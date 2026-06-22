"""Detect waveform-derived components in CASALS L1B receive waveforms.

Scientific boundary
-------------------
CASALS L1B provides one official geolocated ``refh`` point per pulse. This
script detects waveform-derived components for diagnostics, artifact screening,
and downstream refh-quality features. Secondary components produced here are
not georeferenced returns and must not be interpreted as an official
multi-return point cloud product.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
import json

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from waveform_components import (
    RecordIndexGridInfo,
    build_record_index_grid,
    compute_sweep_peak_continuity,
    detect_components_1d,
    estimate_stripe_like_score,
    find_dataset,
    infer_waveform_record_axis,
    prepare_waveform_for_detection,
    read_optional_1d,
    read_waveform_records,
    require_dataset,
    require_scipy,
    safe_argmax_bin,
)


SCRIPT_DIR = Path(__file__).resolve().parent


COMPONENT_COLUMNS = [
    "h5_filename",
    "pulse_index",
    "sweep_num",
    "track_num",
    "component_rank",
    "peak_bin",
    "amplitude_raw",
    "amplitude_processed",
    "prominence",
    "width_bins",
    "left_ips",
    "right_ips",
    "area_processed",
    "rel_bin_to_main",
    "rel_bin_to_raw_argmax",
    "refh",
    "refh_lon",
    "refh_lat",
    "refh_amp",
    "refh_snr",
    "good_snr",
]

PULSE_COLUMNS = [
    "h5_filename",
    "pulse_index",
    "sweep_num",
    "track_num",
    "raw_argmax_bin",
    "raw_argmax_amp",
    "corrected_argmax_bin",
    "n_components",
    "main_peak_bin",
    "main_peak_prominence",
    "main_peak_width",
    "secondary_to_main_amp_ratio",
    "has_clean_single_peak",
    "has_multiple_prominent_peaks",
    "has_broad_main_peak",
    "low_snr_flag",
    "refh",
    "refh_lon",
    "refh_lat",
    "refh_amp",
    "refh_snr",
    "good_snr",
    "bg_mean",
    "bg_std",
    "refh_thres",
]

SWEEP_COLUMNS = [
    "h5_filename",
    "sweep_num",
    "n_tracks",
    "n_valid_pulses",
    "median_n_components",
    "fraction_clean_single_peak",
    "fraction_multi_peak",
    "median_main_width",
    "median_main_prominence",
    "median_refh_snr",
    "fixed_bin_mode",
    "fixed_bin_fraction",
    "stripe_like_flag",
    "continuity_median_abs_residual",
    "continuity_p95_abs_residual",
    "continuity_discontinuity_count",
]


PANDAS_DTYPES = {
    "h5_filename": "string",
    "pulse_index": "Int64",
    "sweep_num": "Int64",
    "track_num": "Int64",
    "component_rank": "Int64",
    "peak_bin": "Float64",
    "amplitude_raw": "Float64",
    "amplitude_processed": "Float64",
    "prominence": "Float64",
    "width_bins": "Float64",
    "left_ips": "Float64",
    "right_ips": "Float64",
    "area_processed": "Float64",
    "rel_bin_to_main": "Float64",
    "rel_bin_to_raw_argmax": "Float64",
    "refh": "Float64",
    "refh_lon": "Float64",
    "refh_lat": "Float64",
    "refh_amp": "Float64",
    "refh_snr": "Float64",
    "good_snr": "boolean",
    "raw_argmax_bin": "Float64",
    "raw_argmax_amp": "Float64",
    "corrected_argmax_bin": "Float64",
    "n_components": "Int64",
    "main_peak_bin": "Float64",
    "main_peak_prominence": "Float64",
    "main_peak_width": "Float64",
    "secondary_to_main_amp_ratio": "Float64",
    "has_clean_single_peak": "boolean",
    "has_multiple_prominent_peaks": "boolean",
    "has_broad_main_peak": "boolean",
    "low_snr_flag": "boolean",
    "bg_mean": "Float64",
    "bg_std": "Float64",
    "refh_thres": "Float64",
    "n_tracks": "Int64",
    "n_valid_pulses": "Int64",
    "median_n_components": "Float64",
    "fraction_clean_single_peak": "Float64",
    "fraction_multi_peak": "Float64",
    "median_main_width": "Float64",
    "median_main_prominence": "Float64",
    "median_refh_snr": "Float64",
    "fixed_bin_mode": "Float64",
    "fixed_bin_fraction": "Float64",
    "stripe_like_flag": "boolean",
    "continuity_median_abs_residual": "Float64",
    "continuity_p95_abs_residual": "Float64",
    "continuity_discontinuity_count": "Int64",
}


@dataclass
class Config:
    h5_path: Path
    out_dir: Path = Path("./outputs/detect_waveform_components")

    selected_sweeps: Optional[Sequence[int]] = None
    sweep_start: int = 0
    sweep_end: Optional[int] = None
    sweep_step: int = 1
    selected_tracks: Optional[Sequence[int]] = None
    max_sweeps: Optional[int] = None

    background_mode: str = "minus_bg_mean"
    clip_negative_after_background: bool = True
    smoothing_method: str = "savgol"
    smoothing_window: int = 7
    smoothing_polyorder: int = 2
    min_height_sigma: float = 3.0
    min_prominence_sigma: float = 3.0
    min_width_bins: float = 1.0
    max_width_bins: Optional[float] = None
    min_distance_bins: int = 3
    max_components_per_pulse: int = 5

    write_component_table: bool = True
    write_pulse_summary: bool = True
    write_sweep_summary: bool = True
    write_diagnostic_png: bool = True
    diagnostic_sweeps: Optional[Sequence[int]] = None


class ParquetChunkWriter:
    """Append row dictionaries to a parquet file with stable column order."""

    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        self.path = path
        self.columns = list(columns)
        self.writer: Optional[pq.ParquetWriter] = None

    def write_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        df = rows_to_dataframe(rows, self.columns)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            ensure_dir(self.path.parent)
            self.writer = pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table)

    def finalize(self) -> None:
        if self.writer is not None:
            self.writer.close()
            return
        ensure_dir(self.path.parent)
        rows_to_dataframe([], self.columns).to_parquet(self.path, index=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, range)):
        return [json_safe(v) for v in obj]
    return obj


def rows_to_dataframe(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    for column in columns:
        if column not in df.columns:
            dtype = PANDAS_DTYPES[column]
            if dtype == "string":
                df[column] = pd.Series([pd.NA] * len(df), dtype="string")
            else:
                df[column] = pd.Series([pd.NA] * len(df), dtype=dtype)
    df = df.loc[:, list(columns)]
    for column in columns:
        df[column] = df[column].astype(PANDAS_DTYPES[column])
    return df


def scalar_attr_to_jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def build_detection_config_dict(cfg: Config, refh_thres: float | None = None) -> dict[str, Any]:
    return {
        "background_mode": cfg.background_mode,
        "clip_negative_after_background": cfg.clip_negative_after_background,
        "smoothing_method": cfg.smoothing_method,
        "smoothing_window": cfg.smoothing_window,
        "smoothing_polyorder": cfg.smoothing_polyorder,
        "min_height_sigma": cfg.min_height_sigma,
        "min_prominence_sigma": cfg.min_prominence_sigma,
        "min_width_bins": cfg.min_width_bins,
        "max_width_bins": cfg.max_width_bins,
        "min_distance_bins": cfg.min_distance_bins,
        "max_components_per_pulse": cfg.max_components_per_pulse,
        "refh_thres": refh_thres,
    }


def choose_sweeps(cfg: Config, n_sweeps: int) -> np.ndarray:
    if cfg.selected_sweeps is not None:
        sweeps = np.asarray(list(cfg.selected_sweeps), dtype=np.int64)
    else:
        start = max(0, int(cfg.sweep_start))
        end = n_sweeps - 1 if cfg.sweep_end is None else min(n_sweeps - 1, int(cfg.sweep_end))
        if end < start:
            raise ValueError(f"Invalid sweep range: start={start}, end={end}")
        step = max(1, int(cfg.sweep_step))
        sweeps = np.arange(start, end + 1, step, dtype=np.int64)
    sweeps = sweeps[(sweeps >= 0) & (sweeps < n_sweeps)]
    if cfg.max_sweeps is not None:
        sweeps = sweeps[: int(cfg.max_sweeps)]
    if sweeps.size == 0:
        raise ValueError("No sweeps were selected for processing.")
    return sweeps


def choose_tracks(cfg: Config, n_tracks: int) -> np.ndarray:
    if cfg.selected_tracks is None:
        return np.arange(n_tracks, dtype=np.int64)
    tracks = np.asarray(list(cfg.selected_tracks), dtype=np.int64)
    tracks = tracks[(tracks >= 0) & (tracks < n_tracks)]
    if tracks.size == 0:
        raise ValueError("No valid tracks were selected for processing.")
    return np.unique(tracks)


def choose_diagnostic_sweeps(cfg: Config, selected_sweeps: np.ndarray) -> set[int]:
    if cfg.diagnostic_sweeps is not None:
        return {int(v) for v in cfg.diagnostic_sweeps}
    idx = {0, selected_sweeps.size // 2, selected_sweeps.size - 1}
    return {int(selected_sweeps[i]) for i in sorted(idx)}


def read_required_fields(h5: h5py.File, n_records: int) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    required_float = [
        "refh",
        "refh_amp",
        "refh_snr",
        "bg_mean",
        "bg_std",
        "refh_thres",
        "rwstart",
        "rwstop",
        "refh_longitude",
        "refh_latitude",
    ]
    required_bool = ["good_snr"]
    for name in required_float:
        arr = read_optional_1d(h5, name, n_records, dtype=np.float64)
        if arr is None:
            raise KeyError(f"Required dataset {name!r} was not found.")
        fields[name] = arr
    for name in required_bool:
        arr = read_optional_1d(h5, name, n_records, dtype=bool)
        if arr is None:
            raise KeyError(f"Required dataset {name!r} was not found.")
        fields[name] = arr
    return fields


def extract_selected_matrix(
    ds: h5py.Dataset,
    record_axis: int,
    info: RecordIndexGridInfo,
    sweep_num: int,
    selected_tracks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    track_grid = info.record_index_grid[int(sweep_num)]
    selected_record_indices = track_grid[selected_tracks]
    valid_mask = selected_record_indices >= 0
    valid_tracks = selected_tracks[valid_mask]
    valid_indices = selected_record_indices[valid_mask]

    n_bins = int(ds.shape[1] if record_axis == 0 else ds.shape[0])
    matrix = np.full((selected_tracks.size, n_bins), np.nan, dtype=np.float32)
    if valid_indices.size:
        matrix[valid_mask] = read_waveform_records(ds, record_axis, valid_indices)
    return matrix, valid_tracks, valid_indices


def finite_median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


def finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def make_diagnostic_figure(
    out_path: Path,
    sweep_num: int,
    track_axis: np.ndarray,
    rx_matrix: np.ndarray,
    corrected_matrix: np.ndarray,
    raw_argmax_bin: np.ndarray,
    corrected_argmax_bin: np.ndarray,
    main_component_bin: np.ndarray,
    secondary_component_bin: np.ndarray,
    n_components: np.ndarray,
    refh_snr: np.ndarray,
    good_snr: np.ndarray,
) -> None:
    ensure_dir(out_path.parent)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    raw_vmax = np.nanpercentile(rx_matrix[np.isfinite(rx_matrix)], 99.5) if np.any(np.isfinite(rx_matrix)) else 1.0
    corr_vmax = np.nanpercentile(corrected_matrix[np.isfinite(corrected_matrix)], 99.5) if np.any(np.isfinite(corrected_matrix)) else 1.0
    raw_vmax = max(float(raw_vmax), 1.0)
    corr_vmax = max(float(corr_vmax), 1.0)

    im1 = ax1.imshow(rx_matrix.T, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
    fig.colorbar(im1, ax=ax1, label="Raw amplitude")
    ax1.plot(track_axis, raw_argmax_bin, color="white", linewidth=1.0, label="raw argmax")
    ax1.plot(track_axis, main_component_bin, color="tab:red", linewidth=1.0, label="main component")
    ax1.set_title(f"Sweep {sweep_num} raw RX matrix diagnostic")
    ax1.set_xlabel("Track")
    ax1.set_ylabel("RX bin")
    ax1.legend(loc="upper right")

    im2 = ax2.imshow(
        corrected_matrix.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
        vmin=0.0 if np.nanmin(corrected_matrix) >= 0 else None,
        vmax=corr_vmax,
    )
    fig.colorbar(im2, ax=ax2, label="Background-adjusted amplitude")
    ax2.plot(track_axis, corrected_argmax_bin, color="cyan", linewidth=1.0, label="corrected argmax")
    ax2.set_title("Background-adjusted RX matrix diagnostic")
    ax2.set_xlabel("Track")
    ax2.set_ylabel("RX bin")
    ax2.legend(loc="upper right")

    ax3.plot(track_axis, raw_argmax_bin, label="raw argmax", color="0.25")
    ax3.plot(track_axis, corrected_argmax_bin, label="corrected argmax", color="tab:blue")
    ax3.plot(track_axis, main_component_bin, label="main component", color="tab:red")
    ax3.plot(track_axis, secondary_component_bin, label="secondary component", color="tab:orange", alpha=0.8)
    ax3.set_title("Track-wise bin trajectories")
    ax3.set_xlabel("Track")
    ax3.set_ylabel("RX bin")
    ax3.legend(loc="upper right")

    ax4.plot(track_axis, n_components, color="tab:purple", label="n_components")
    ax4.set_xlabel("Track")
    ax4.set_ylabel("Component count", color="tab:purple")
    ax4.tick_params(axis="y", labelcolor="tab:purple")
    ax4.set_title("Track-wise component count and SNR diagnostics")
    ax4b = ax4.twinx()
    ax4b.plot(track_axis, refh_snr, color="tab:green", label="refh_snr", alpha=0.9)
    good_tracks = track_axis[np.asarray(good_snr, dtype=bool)]
    if good_tracks.size:
        ax4b.scatter(good_tracks, np.full(good_tracks.size, np.nanmax(refh_snr[np.isfinite(refh_snr)]) if np.any(np.isfinite(refh_snr)) else 1.0), s=8, color="black", label="good_snr=True")
    ax4b.set_ylabel("refh_snr", color="tab:green")
    ax4b.tick_params(axis="y", labelcolor="tab:green")
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    finite_main = main_component_bin[np.isfinite(main_component_bin)]
    if finite_main.size:
        ax5.hist(finite_main, bins=40, color="tab:red", alpha=0.8)
    ax5.set_title("Histogram of main component bin")
    ax5.set_xlabel("Main component bin")
    ax5.set_ylabel("Count")

    finite_counts = n_components[np.isfinite(n_components)]
    if finite_counts.size:
        bins = np.arange(-0.5, np.nanmax(finite_counts) + 1.5, 1.0)
        ax6.hist(finite_counts, bins=bins, color="tab:purple", alpha=0.8)
    ax6.set_title("Histogram of detected component count")
    ax6.set_xlabel("n_components")
    ax6.set_ylabel("Count")

    fig.suptitle(
        f"CASALS L1B waveform component diagnostic sweep {sweep_num} "
        "(waveform-derived only; not georeferenced returns)",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_run_metadata(
    cfg: Config,
    h5_path: Path,
    info: RecordIndexGridInfo,
    dataset_info: dict[str, Any],
    selected_sweeps: np.ndarray,
    selected_tracks: np.ndarray,
) -> dict[str, Any]:
    return {
        "runtime_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": json_safe(asdict(cfg)),
        "inputs": {
            "h5_path": str(h5_path),
            "h5_filename": h5_path.name,
        },
        "dataset_shapes": dataset_info,
        "grid_status": {
            "n_records": info.n_records,
            "n_sweeps": info.n_sweeps,
            "n_tracks": info.n_tracks,
            "complete_rectangular_grid": info.complete_rectangular_grid,
            "duplicate_sweep_track_cells": info.duplicate_sweep_track_cells,
            "missing_sweep_track_cells": info.missing_sweep_track_cells,
        },
        "selected_domain": {
            "selected_sweeps": selected_sweeps.tolist(),
            "selected_tracks": selected_tracks.tolist(),
            "n_selected_sweeps": int(selected_sweeps.size),
            "n_selected_tracks": int(selected_tracks.size),
        },
        "scientific_notes": [
            "CASALS L1B has one official geolocated refh point per pulse.",
            "Waveform-derived secondary components in this run are diagnostic only and are not georeferenced returns.",
            "This workflow is intended for waveform component detection, refh quality diagnostics, and stripe/artifact screening.",
            "No official multi-return point cloud is created by this script.",
        ],
    }


def run_detection(cfg: Config) -> dict[str, Path]:
    require_scipy()

    h5_path = resolve_path(cfg.h5_path)
    out_dir = resolve_path(cfg.out_dir)
    ensure_dir(out_dir)
    diag_dir = out_dir / "diagnostic_figures"
    if cfg.write_diagnostic_png:
        ensure_dir(diag_dir)

    component_path = out_dir / "component_table.parquet"
    pulse_path = out_dir / "pulse_summary.parquet"
    sweep_path = out_dir / "sweep_summary.csv"
    metadata_path = out_dir / "run_metadata.json"

    component_writer = ParquetChunkWriter(component_path, COMPONENT_COLUMNS)
    pulse_writer = ParquetChunkWriter(pulse_path, PULSE_COLUMNS)
    sweep_rows: list[dict[str, Any]] = []

    with h5py.File(h5_path, "r") as h5:
        rx_ds = require_dataset(h5, "rx_waveform")
        tx_ds = require_dataset(h5, "tx_waveform")
        info = build_record_index_grid(h5)
        if info.duplicate_sweep_track_cells > 0:
            raise ValueError(
                f"Detected {info.duplicate_sweep_track_cells} duplicate sweep-track cells; "
                "waveform component detection requires unique pulse indexing."
            )

        rx_record_axis = infer_waveform_record_axis(rx_ds, info.n_records)
        tx_record_axis = infer_waveform_record_axis(tx_ds, info.n_records)
        fields = read_required_fields(h5, info.n_records)

        selected_sweeps = choose_sweeps(cfg, info.n_sweeps)
        selected_tracks = choose_tracks(cfg, info.n_tracks)
        diagnostic_sweeps = choose_diagnostic_sweeps(cfg, selected_sweeps)

        dataset_info = {
            "rx_waveform": {"shape": list(rx_ds.shape), "dtype": str(rx_ds.dtype), "record_axis": int(rx_record_axis)},
            "tx_waveform": {"shape": list(tx_ds.shape), "dtype": str(tx_ds.dtype), "record_axis": int(tx_record_axis)},
        }
        for name in [
            "refh",
            "refh_amp",
            "refh_snr",
            "good_snr",
            "bg_mean",
            "bg_std",
            "refh_thres",
            "rwstart",
            "rwstop",
            "refh_longitude",
            "refh_latitude",
        ]:
            ds = find_dataset(h5, name)
            if ds is not None:
                dataset_info[name] = {"shape": list(ds.shape), "dtype": str(ds.dtype)}

        metadata = build_run_metadata(cfg, h5_path, info, dataset_info, selected_sweeps, selected_tracks)
        metadata["h5_attrs_subset"] = {
            str(k): scalar_attr_to_jsonable(v) for k, v in h5.attrs.items()
        }

        print(f"H5: {h5_path}")
        print(f"Selected sweeps: {selected_sweeps[0]}..{selected_sweeps[-1]} ({selected_sweeps.size} total)")
        print(f"Selected tracks: {selected_tracks[0]}..{selected_tracks[-1]} ({selected_tracks.size} total)")

        for sweep_num in selected_sweeps:
            rx_matrix, valid_tracks, valid_indices = extract_selected_matrix(
                ds=rx_ds,
                record_axis=rx_record_axis,
                info=info,
                sweep_num=int(sweep_num),
                selected_tracks=selected_tracks,
            )
            if valid_tracks.size == 0:
                continue

            corrected_matrix = np.full_like(rx_matrix, np.nan, dtype=np.float32)
            raw_argmax_bin = np.full(selected_tracks.size, np.nan, dtype=np.float64)
            raw_argmax_amp = np.full(selected_tracks.size, np.nan, dtype=np.float64)
            corrected_argmax_bin = np.full(selected_tracks.size, np.nan, dtype=np.float64)
            main_component_bin = np.full(selected_tracks.size, np.nan, dtype=np.float64)
            secondary_component_bin = np.full(selected_tracks.size, np.nan, dtype=np.float64)
            n_components_arr = np.full(selected_tracks.size, np.nan, dtype=np.float64)
            refh_snr_arr = np.full(selected_tracks.size, np.nan, dtype=np.float64)
            good_snr_arr = np.zeros(selected_tracks.size, dtype=bool)

            component_rows: list[dict[str, Any]] = []
            pulse_rows: list[dict[str, Any]] = []

            for local_idx, track_num in enumerate(selected_tracks):
                pulse_index = int(info.record_index_grid[int(sweep_num), int(track_num)])
                if pulse_index < 0:
                    continue

                y_raw = np.asarray(rx_matrix[local_idx], dtype=np.float64)
                bg_mean = float(fields["bg_mean"][pulse_index])
                bg_std = float(fields["bg_std"][pulse_index])
                refh_thres = float(fields["refh_thres"][pulse_index])
                prepared = prepare_waveform_for_detection(
                    y_raw=y_raw,
                    bg_mean=bg_mean,
                    bg_std=bg_std,
                    config=build_detection_config_dict(cfg, refh_thres=refh_thres),
                )
                corrected = np.asarray(prepared["processed"], dtype=np.float64)
                corrected_matrix[local_idx] = corrected.astype(np.float32, copy=False)

                raw_bin, raw_amp = safe_argmax_bin(y_raw)
                corr_bin, _ = safe_argmax_bin(corrected)
                components = detect_components_1d(
                    y_raw=y_raw,
                    bg_mean=bg_mean,
                    bg_std=bg_std,
                    config=build_detection_config_dict(cfg, refh_thres=refh_thres),
                )
                pulse_summary = summarize_for_script(
                    components=components,
                    raw_argmax_bin=raw_bin,
                    corrected_argmax_bin=corr_bin,
                    refh_amp=float(fields["refh_amp"][pulse_index]),
                    refh_snr=float(fields["refh_snr"][pulse_index]),
                    good_snr=bool(fields["good_snr"][pulse_index]),
                )

                raw_argmax_bin[local_idx] = raw_bin
                raw_argmax_amp[local_idx] = raw_amp
                corrected_argmax_bin[local_idx] = corr_bin
                n_components_arr[local_idx] = pulse_summary["n_components"]
                refh_snr_arr[local_idx] = float(fields["refh_snr"][pulse_index])
                good_snr_arr[local_idx] = bool(fields["good_snr"][pulse_index])
                if np.isfinite(pulse_summary["main_peak_bin"]):
                    main_component_bin[local_idx] = pulse_summary["main_peak_bin"]
                if np.isfinite(pulse_summary["secondary_peak_bin"]):
                    secondary_component_bin[local_idx] = pulse_summary["secondary_peak_bin"]

                main_peak_bin = pulse_summary["main_peak_bin"]
                for component in components:
                    peak_bin = float(component["peak_bin"])
                    component_rows.append(
                        {
                            "h5_filename": h5_path.name,
                            "pulse_index": pulse_index,
                            "sweep_num": int(sweep_num),
                            "track_num": int(track_num),
                            "component_rank": int(component["rank_by_prominence"]),
                            "peak_bin": peak_bin,
                            "amplitude_raw": float(component["amplitude_raw"]),
                            "amplitude_processed": float(component["amplitude_processed"]),
                            "prominence": float(component["prominence"]),
                            "width_bins": float(component["width_bins"]),
                            "left_ips": float(component["left_ips"]),
                            "right_ips": float(component["right_ips"]),
                            "area_processed": float(component["area_processed"]),
                            "rel_bin_to_main": peak_bin - main_peak_bin if np.isfinite(main_peak_bin) else np.nan,
                            "rel_bin_to_raw_argmax": peak_bin - raw_bin if np.isfinite(raw_bin) else np.nan,
                            "refh": float(fields["refh"][pulse_index]),
                            "refh_lon": float(fields["refh_longitude"][pulse_index]),
                            "refh_lat": float(fields["refh_latitude"][pulse_index]),
                            "refh_amp": float(fields["refh_amp"][pulse_index]),
                            "refh_snr": float(fields["refh_snr"][pulse_index]),
                            "good_snr": bool(fields["good_snr"][pulse_index]),
                        }
                    )

                pulse_rows.append(
                    {
                        "h5_filename": h5_path.name,
                        "pulse_index": pulse_index,
                        "sweep_num": int(sweep_num),
                        "track_num": int(track_num),
                        "raw_argmax_bin": raw_bin,
                        "raw_argmax_amp": raw_amp,
                        "corrected_argmax_bin": corr_bin,
                        "n_components": pulse_summary["n_components"],
                        "main_peak_bin": pulse_summary["main_peak_bin"],
                        "main_peak_prominence": pulse_summary["main_peak_prominence"],
                        "main_peak_width": pulse_summary["main_peak_width"],
                        "secondary_to_main_amp_ratio": pulse_summary["secondary_to_main_amp_ratio"],
                        "has_clean_single_peak": pulse_summary["has_clean_single_peak"],
                        "has_multiple_prominent_peaks": pulse_summary["has_multiple_prominent_peaks"],
                        "has_broad_main_peak": pulse_summary["has_broad_main_peak"],
                        "low_snr_flag": pulse_summary["low_snr_flag"],
                        "refh": float(fields["refh"][pulse_index]),
                        "refh_lon": float(fields["refh_longitude"][pulse_index]),
                        "refh_lat": float(fields["refh_latitude"][pulse_index]),
                        "refh_amp": float(fields["refh_amp"][pulse_index]),
                        "refh_snr": float(fields["refh_snr"][pulse_index]),
                        "good_snr": bool(fields["good_snr"][pulse_index]),
                        "bg_mean": bg_mean,
                        "bg_std": bg_std,
                        "refh_thres": refh_thres,
                    }
                )

            if cfg.write_component_table:
                component_writer.write_rows(component_rows)
            if cfg.write_pulse_summary:
                pulse_writer.write_rows(pulse_rows)

            if pulse_rows:
                pulse_df = rows_to_dataframe(pulse_rows, PULSE_COLUMNS)
                continuity = compute_sweep_peak_continuity(
                    track=pulse_df["track_num"].to_numpy(dtype=np.float64),
                    peak_bin=pulse_df["main_peak_bin"].to_numpy(dtype=np.float64),
                )
                stripe = estimate_stripe_like_score(pulse_df["main_peak_bin"].to_numpy(dtype=np.float64))
                sweep_rows.append(
                    {
                        "h5_filename": h5_path.name,
                        "sweep_num": int(sweep_num),
                        "n_tracks": int(selected_tracks.size),
                        "n_valid_pulses": int(len(pulse_rows)),
                        "median_n_components": finite_median(pulse_df["n_components"].to_numpy(dtype=np.float64)),
                        "fraction_clean_single_peak": finite_mean(pulse_df["has_clean_single_peak"].astype("Float64")),
                        "fraction_multi_peak": finite_mean(pulse_df["has_multiple_prominent_peaks"].astype("Float64")),
                        "median_main_width": finite_median(pulse_df["main_peak_width"].to_numpy(dtype=np.float64)),
                        "median_main_prominence": finite_median(pulse_df["main_peak_prominence"].to_numpy(dtype=np.float64)),
                        "median_refh_snr": finite_median(pulse_df["refh_snr"].to_numpy(dtype=np.float64)),
                        "fixed_bin_mode": stripe["fixed_bin_mode"],
                        "fixed_bin_fraction": stripe["fixed_bin_fraction"],
                        "stripe_like_flag": stripe["stripe_like_flag"],
                        "continuity_median_abs_residual": continuity["median_abs_residual"],
                        "continuity_p95_abs_residual": continuity["p95_abs_residual"],
                        "continuity_discontinuity_count": continuity["discontinuity_count"],
                    }
                )

            if cfg.write_diagnostic_png and int(sweep_num) in diagnostic_sweeps:
                make_diagnostic_figure(
                    out_path=diag_dir / f"sweep_{int(sweep_num):05d}_rx_components.png",
                    sweep_num=int(sweep_num),
                    track_axis=selected_tracks.astype(np.int64),
                    rx_matrix=np.asarray(rx_matrix, dtype=np.float64),
                    corrected_matrix=np.asarray(corrected_matrix, dtype=np.float64),
                    raw_argmax_bin=raw_argmax_bin,
                    corrected_argmax_bin=corrected_argmax_bin,
                    main_component_bin=main_component_bin,
                    secondary_component_bin=secondary_component_bin,
                    n_components=n_components_arr,
                    refh_snr=refh_snr_arr,
                    good_snr=good_snr_arr,
                )

            print(
                f"Processed sweep {int(sweep_num)}: "
                f"{len(pulse_rows)} pulses, {len(component_rows)} components"
            )

    if cfg.write_component_table:
        component_writer.finalize()
    if cfg.write_pulse_summary:
        pulse_writer.finalize()
    if cfg.write_sweep_summary:
        rows_to_dataframe(sweep_rows, SWEEP_COLUMNS).to_csv(sweep_path, index=False)

    metadata["outputs"] = {
        "component_table_parquet": str(component_path),
        "pulse_summary_parquet": str(pulse_path),
        "sweep_summary_csv": str(sweep_path),
        "run_metadata_json": str(metadata_path),
        "diagnostic_figures_dir": str(diag_dir),
    }
    metadata["output_counts"] = {
        "n_sweep_rows": int(len(sweep_rows)),
    }
    metadata_path.write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")

    print(f"Wrote: {component_path}")
    print(f"Wrote: {pulse_path}")
    print(f"Wrote: {sweep_path}")
    print(f"Wrote: {metadata_path}")

    return {
        "component_table": component_path,
        "pulse_summary": pulse_path,
        "sweep_summary": sweep_path,
        "run_metadata": metadata_path,
    }


def summarize_for_script(
    components: Sequence[Mapping[str, Any]],
    raw_argmax_bin: float | int | None,
    corrected_argmax_bin: float | int | None,
    refh_amp: float | None,
    refh_snr: float | None,
    good_snr: bool | int | None,
) -> dict[str, Any]:
    from waveform_components import summarize_pulse_components

    return summarize_pulse_components(
        components=components,
        raw_argmax_bin=raw_argmax_bin,
        corrected_argmax_bin=corrected_argmax_bin,
        refh_amp=refh_amp,
        refh_snr=refh_snr,
        good_snr=good_snr,
    )


def main() -> None:
    cfg = Config(
        h5_path=Path("./casals_h5_downloads/casals_l1b_20241118T171757_001_02.h5"),
        out_dir=Path("./outputs/detect_waveform_components"),
        selected_sweeps=None,
        sweep_start=5000,
        sweep_end=5002,
        sweep_step=1,
        selected_tracks=None,
        max_sweeps=None,
        background_mode="minus_bg_mean",
        clip_negative_after_background=True,
        smoothing_method="savgol",
        smoothing_window=7,
        smoothing_polyorder=2,
        min_height_sigma=3.0,
        min_prominence_sigma=3.0,
        min_width_bins=1.0,
        max_width_bins=None,
        min_distance_bins=3,
        max_components_per_pulse=5,
        write_component_table=True,
        write_pulse_summary=True,
        write_sweep_summary=True,
        write_diagnostic_png=True,
        diagnostic_sweeps=None,
    )
    run_detection(cfg)


if __name__ == "__main__":
    main()
