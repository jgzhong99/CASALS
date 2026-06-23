"""Batch extraction of CASALS L1B waveform-derived diagnostic features.

Scientific scope
----------------
CASALS L1B is a geolocated waveform product. Each pulse has one official
geolocated ``refh`` point, and ``refh`` is associated with the
maximum-amplitude RX waveform bin. This script extracts waveform-derived
candidate/prominent component features for refh quality diagnosis, waveform
artifact detection, and downstream refh filtering/classification support.

Secondary waveform peaks/components produced here are diagnostic features only.
They are not official returns, not georeferenced returns, and not a
multi-return point cloud.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from waveform_components import (
    SweepTrackIndex,
    build_record_index_grid,
    classify_prominent_components,
    compute_fixed_bin_stripe_features,
    compute_range_window_height_candidate,
    compute_sweep_continuity_features,
    detect_candidate_components_1d,
    find_dataset,
    infer_waveform_record_axis,
    read_attrs_subset,
    read_optional_1d,
    read_sweep_matrix,
    require_dataset,
    require_scipy,
    safe_argmax_bin,
    summarize_pulse_waveform_features,
)


SCRIPT_DIR = Path(__file__).resolve().parent

COMMON_SCIENTIFIC_NOTES = [
    "CASALS L1B is a geolocated waveform product.",
    "Each pulse has one official geolocated refh point.",
    "refh is associated with the maximum-amplitude RX waveform bin.",
    "Secondary waveform peaks/components are waveform-derived diagnostics only.",
    "Secondary components are not official returns, georeferenced returns, or a multi-return point cloud.",
    "This workflow is for refh quality diagnosis, waveform artifact detection, and waveform-derived feature extraction for downstream refh filtering/classification.",
]

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
    "is_main_component",
    "is_prominent_component",
    "is_valid_secondary_candidate",
    "rejected_reason",
    "rel_bin_to_main",
    "rel_bin_to_raw_argmax",
    "tentative_height_rwstart_to_rwstop",
    "tentative_height_offset_from_refh",
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
    "corrected_argmax_amp",
    "n_candidate_components",
    "n_prominent_components",
    "n_valid_secondary_components",
    "main_peak_bin",
    "main_peak_prominence",
    "main_peak_width_bins",
    "main_peak_area",
    "secondary_peak_bin",
    "secondary_to_main_amp_ratio",
    "secondary_to_main_prominence_ratio",
    "has_clear_main_peak",
    "has_valid_secondary_peak",
    "has_broad_main_peak",
    "has_weak_main_peak",
    "low_snr_flag",
    "waveform_reliability_class",
    "fixed_bin_stripe_score",
    "continuity_residual_bins",
    "refh",
    "refh_lon",
    "refh_lat",
    "refh_amp",
    "refh_snr",
    "good_snr",
    "bg_mean",
    "bg_std",
    "refh_thres",
    "rwstart",
    "rwstop",
]

SWEEP_COLUMNS = [
    "h5_filename",
    "sweep_num",
    "n_tracks",
    "n_valid_pulses",
    "median_refh_snr",
    "fraction_good_snr",
    "median_main_peak_prominence",
    "median_main_peak_width_bins",
    "median_n_candidate_components",
    "median_n_prominent_components",
    "fraction_clear_main_peak",
    "fraction_valid_secondary_peak",
    "fraction_weak_main_peak",
    "fraction_broad_main_peak",
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
    "is_main_component": "boolean",
    "is_prominent_component": "boolean",
    "is_valid_secondary_candidate": "boolean",
    "rejected_reason": "string",
    "rel_bin_to_main": "Float64",
    "rel_bin_to_raw_argmax": "Float64",
    "tentative_height_rwstart_to_rwstop": "Float64",
    "tentative_height_offset_from_refh": "Float64",
    "refh": "Float64",
    "refh_lon": "Float64",
    "refh_lat": "Float64",
    "refh_amp": "Float64",
    "refh_snr": "Float64",
    "good_snr": "boolean",
    "raw_argmax_bin": "Float64",
    "raw_argmax_amp": "Float64",
    "corrected_argmax_bin": "Float64",
    "corrected_argmax_amp": "Float64",
    "n_candidate_components": "Int64",
    "n_prominent_components": "Int64",
    "n_valid_secondary_components": "Int64",
    "main_peak_bin": "Float64",
    "main_peak_prominence": "Float64",
    "main_peak_width_bins": "Float64",
    "main_peak_area": "Float64",
    "secondary_peak_bin": "Float64",
    "secondary_to_main_amp_ratio": "Float64",
    "secondary_to_main_prominence_ratio": "Float64",
    "has_clear_main_peak": "boolean",
    "has_valid_secondary_peak": "boolean",
    "has_broad_main_peak": "boolean",
    "has_weak_main_peak": "boolean",
    "low_snr_flag": "boolean",
    "waveform_reliability_class": "string",
    "fixed_bin_stripe_score": "Float64",
    "continuity_residual_bins": "Float64",
    "bg_mean": "Float64",
    "bg_std": "Float64",
    "refh_thres": "Float64",
    "rwstart": "Float64",
    "rwstop": "Float64",
    "n_tracks": "Int64",
    "n_valid_pulses": "Int64",
    "median_refh_snr": "Float64",
    "fraction_good_snr": "Float64",
    "median_main_peak_prominence": "Float64",
    "median_main_peak_width_bins": "Float64",
    "median_n_candidate_components": "Float64",
    "median_n_prominent_components": "Float64",
    "fraction_clear_main_peak": "Float64",
    "fraction_valid_secondary_peak": "Float64",
    "fraction_weak_main_peak": "Float64",
    "fraction_broad_main_peak": "Float64",
    "fixed_bin_mode": "Float64",
    "fixed_bin_fraction": "Float64",
    "stripe_like_flag": "boolean",
    "continuity_median_abs_residual": "Float64",
    "continuity_p95_abs_residual": "Float64",
    "continuity_discontinuity_count": "Int64",
}


@dataclass
class Config:
    input_h5_paths: list[Path]
    output_root: Path = Path("./outputs/extract_waveform_features")
    sweep_start: int | None = None
    sweep_end: int | None = None
    sweep_step: int = 1
    selected_sweeps: list[int] | None = None
    selected_tracks: list[int] | None = None
    max_sweeps: int | None = None
    background_mode: str = "minus_bg_mean"
    clip_negative_after_background: bool = True
    smoothing_method: str = "savgol"
    smoothing_window: int = 7
    smoothing_polyorder: int = 2
    min_height_sigma: float = 3.0
    min_prominence_sigma: float = 4.0
    min_width_bins: float = 1.0
    max_width_bins: float | None = 80.0
    min_distance_bins: int = 5
    max_candidate_components_per_pulse: int = 8
    min_prominent_prominence_sigma: float = 5.0
    min_prominent_relative_to_main: float = 0.20
    min_secondary_separation_bins: int = 8
    broad_main_width_bins: float = 30.0
    weak_main_prominence_sigma: float = 4.0
    edge_exclusion_bins: int = 10
    stripe_bin_tolerance: int = 2
    stripe_min_fraction: float = 0.20
    write_component_table: bool = True
    write_pulse_summary: bool = True
    write_sweep_summary: bool = True
    write_diagnostic_png: bool = True
    diagnostic_sweeps: list[int] | None = None
    random_seed: int = 42


class ParquetChunkWriter:
    """Append stable-schema parquet chunks."""

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
        else:
            ensure_dir(self.path.parent)
            rows_to_dataframe([], self.columns).to_parquet(self.path, index=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


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


def finite_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def finite_median(values: Sequence[float]) -> float:
    arr = finite_values(values)
    return float(np.median(arr)) if arr.size else np.nan


def finite_mean(values: Sequence[float]) -> float:
    arr = finite_values(values)
    return float(np.mean(arr)) if arr.size else np.nan


def choose_sweeps(cfg: Config, index_info: SweepTrackIndex) -> np.ndarray:
    if cfg.selected_sweeps is not None:
        sweeps = np.asarray(cfg.selected_sweeps, dtype=np.int64)
    else:
        start = 0 if cfg.sweep_start is None else max(0, int(cfg.sweep_start))
        end = index_info.n_sweeps - 1 if cfg.sweep_end is None else min(index_info.n_sweeps - 1, int(cfg.sweep_end))
        if end < start:
            raise ValueError(f"Invalid sweep selection: start={start}, end={end}")
        sweeps = np.arange(start, end + 1, max(1, int(cfg.sweep_step)), dtype=np.int64)
    sweeps = sweeps[(sweeps >= 0) & (sweeps < index_info.n_sweeps)]
    if cfg.max_sweeps is not None:
        sweeps = sweeps[: int(cfg.max_sweeps)]
    if sweeps.size == 0:
        raise ValueError("No sweeps were selected.")
    return sweeps


def choose_tracks(cfg: Config, index_info: SweepTrackIndex) -> np.ndarray:
    if cfg.selected_tracks is None:
        return np.arange(index_info.n_tracks, dtype=np.int64)
    tracks = np.asarray(cfg.selected_tracks, dtype=np.int64)
    tracks = tracks[(tracks >= 0) & (tracks < index_info.n_tracks)]
    if tracks.size == 0:
        raise ValueError("No valid tracks were selected.")
    return np.unique(tracks)


def choose_diagnostic_sweeps(cfg: Config, selected_sweeps: np.ndarray) -> set[int]:
    if cfg.diagnostic_sweeps is not None:
        return {int(v) for v in cfg.diagnostic_sweeps}
    idx = {0, selected_sweeps.size // 2, selected_sweeps.size - 1}
    return {int(selected_sweeps[i]) for i in sorted(idx)}


def build_detector_config(cfg: Config) -> dict[str, Any]:
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
        "max_candidate_components_per_pulse": cfg.max_candidate_components_per_pulse,
        "min_prominent_prominence_sigma": cfg.min_prominent_prominence_sigma,
        "min_prominent_relative_to_main": cfg.min_prominent_relative_to_main,
        "min_secondary_separation_bins": cfg.min_secondary_separation_bins,
        "broad_main_width_bins": cfg.broad_main_width_bins,
        "weak_main_prominence_sigma": cfg.weak_main_prominence_sigma,
        "edge_exclusion_bins": cfg.edge_exclusion_bins,
        "exclude_edge_components": True,
        "stripe_bin_tolerance": cfg.stripe_bin_tolerance,
        "stripe_min_fraction": cfg.stripe_min_fraction,
    }


def read_required_fields(h5: h5py.File, n_records: int) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    required_specs = {
        "refh": np.float64,
        "refh_longitude": np.float64,
        "refh_latitude": np.float64,
        "refh_amp": np.float64,
        "refh_snr": np.float64,
        "good_snr": bool,
        "bg_mean": np.float64,
        "bg_std": np.float64,
        "refh_thres": np.float64,
        "rwstart": np.float64,
        "rwstop": np.float64,
    }
    for name, dtype in required_specs.items():
        arr = read_optional_1d(h5, name, n_records, dtype=dtype)
        if arr is None:
            raise KeyError(f"Required dataset {name!r} was not found in the H5 file.")
        fields[name] = arr
    return fields


def collect_dataset_shapes(h5: h5py.File, names: Sequence[str]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for name in names:
        ds = find_dataset(h5, name)
        if ds is None:
            continue
        info[name] = {"shape": list(ds.shape), "dtype": str(ds.dtype)}
    return info


def maybe_read_tx_shape(h5: h5py.File, index_info: SweepTrackIndex) -> dict[str, Any] | None:
    ds = find_dataset(h5, "tx_waveform")
    if ds is None:
        return None
    axis = infer_waveform_record_axis(ds, index_info.n_records)
    return {"shape": list(ds.shape), "dtype": str(ds.dtype), "record_axis": int(axis)}


def compute_reliability_numeric(classes: Sequence[str]) -> np.ndarray:
    order = {
        "clear_main_peak": 4,
        "clear_main_with_secondary_candidate": 5,
        "broad_main_peak": 2,
        "weak_main_peak": 1,
        "low_snr_weak_main_peak": 0,
        "ambiguous_main_peak": 3,
        "no_detected_main_peak": -1,
    }
    return np.asarray([order.get(str(value), -1) for value in classes], dtype=np.float64)


def make_diagnostic_figure(
    out_path: Path,
    h5_filename: str,
    sweep_num: int,
    track_axis: np.ndarray,
    rx_matrix: np.ndarray,
    corrected_matrix: np.ndarray,
    pulse_df: pd.DataFrame,
    component_df: pd.DataFrame,
    stripe_features: Mapping[str, Any],
    continuity_features: Mapping[str, Any],
) -> None:
    ensure_dir(out_path.parent)

    raw_argmax = pulse_df["raw_argmax_bin"].to_numpy(dtype=np.float64)
    corrected_argmax = pulse_df["corrected_argmax_bin"].to_numpy(dtype=np.float64)
    main_peak_bin = pulse_df["main_peak_bin"].to_numpy(dtype=np.float64)
    refh_snr = pulse_df["refh_snr"].to_numpy(dtype=np.float64)
    reliability_num = compute_reliability_numeric(pulse_df["waveform_reliability_class"].astype(str))
    good_snr = pulse_df["good_snr"].astype(bool).to_numpy()

    fig, axes = plt.subplots(3, 2, figsize=(17, 14), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    im1 = ax1.imshow(rx_matrix.T, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
    fig.colorbar(im1, ax=ax1, label="Raw RX amplitude")
    ax1.plot(track_axis, raw_argmax, color="white", linewidth=1.0, label="raw argmax")
    ax1.plot(track_axis, main_peak_bin, color="tab:red", linewidth=1.0, label="main peak")
    ax1.set_title("RX matrix.T with raw argmax and main peak")
    ax1.set_xlabel("Track")
    ax1.set_ylabel("RX bin")
    ax1.legend(loc="upper right")

    im2 = ax2.imshow(corrected_matrix.T, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    fig.colorbar(im2, ax=ax2, label="Background-subtracted amplitude")
    ax2.set_title("Background-subtracted RX matrix.T")
    ax2.set_xlabel("Track")
    ax2.set_ylabel("RX bin")

    ax3.plot(track_axis, raw_argmax, label="raw argmax", color="0.25")
    ax3.plot(track_axis, corrected_argmax, label="corrected argmax", color="tab:blue")
    ax3.plot(track_axis, main_peak_bin, label="main peak", color="tab:red")
    ax3.set_title("Track-wise bin trajectories")
    ax3.set_xlabel("Track")
    ax3.set_ylabel("RX bin")
    ax3.legend(loc="upper right")

    ax4.plot(track_axis, refh_snr, color="tab:green", label="refh_snr")
    ax4.scatter(track_axis, reliability_num, s=10, color="tab:purple", alpha=0.75, label="waveform_reliability_class")
    good_tracks = track_axis[good_snr]
    if good_tracks.size:
        ax4.scatter(good_tracks, np.full(good_tracks.size, np.nanmax(refh_snr[np.isfinite(refh_snr)]) if np.any(np.isfinite(refh_snr)) else 0.0), marker="v", color="black", s=15, label="good_snr=True")
    ax4.set_title("Track vs refh_snr and waveform reliability")
    ax4.set_xlabel("Track")
    ax4.set_ylabel("refh_snr / reliability code")
    ax4.legend(loc="upper right")

    candidate_bins = component_df["peak_bin"].to_numpy(dtype=np.float64) if not component_df.empty else np.array([], dtype=np.float64)
    prominent_bins = component_df.loc[component_df["is_prominent_component"].fillna(False), "peak_bin"].to_numpy(dtype=np.float64) if not component_df.empty else np.array([], dtype=np.float64)
    if candidate_bins.size:
        ax5.hist(candidate_bins[np.isfinite(candidate_bins)], bins=50, color="tab:gray", alpha=0.55, label="candidate components")
    if prominent_bins.size:
        ax5.hist(prominent_bins[np.isfinite(prominent_bins)], bins=50, color="tab:red", alpha=0.75, label="prominent components")
    ax5.hist(main_peak_bin[np.isfinite(main_peak_bin)], bins=50, histtype="step", linewidth=2.0, color="tab:blue", label="main peaks")
    ax5.set_title("Main-peak and candidate/prominent component bins")
    ax5.set_xlabel("RX bin")
    ax5.set_ylabel("Count")
    ax5.legend(loc="upper right")

    continuity_residual = pulse_df["continuity_residual_bins"].to_numpy(dtype=np.float64)
    ax6.plot(track_axis, continuity_residual, color="tab:orange", label="continuity residual bins")
    band_min = stripe_features.get("fixed_bin_band_min", np.nan)
    band_max = stripe_features.get("fixed_bin_band_max", np.nan)
    if np.isfinite(band_min) and np.isfinite(band_max):
        ax6.axhspan(band_min, band_max, color="tab:green", alpha=0.12, label="stripe band")
    ax6.plot(track_axis, main_peak_bin, color="tab:red", linewidth=1.0, alpha=0.7, label="main peak bin")
    ax6.set_title(
        "Stripe and continuity diagnostics\n"
        f"fixed_bin_fraction={stripe_features.get('fixed_bin_fraction', np.nan):.3f}, "
        f"continuity_p95={continuity_features.get('continuity_p95_abs_residual', np.nan):.2f}"
    )
    ax6.set_xlabel("Track")
    ax6.set_ylabel("Bin / residual")
    ax6.legend(loc="upper right")

    fig.suptitle(
        f"CASALS L1B waveform features diagnostic | {h5_filename} | sweep {sweep_num}\n"
        "Waveform-derived diagnostics only; not a multi-return georeferencing product.",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_metadata(
    cfg: Config,
    h5_path: Path,
    output_dir: Path,
    index_info: SweepTrackIndex,
    dataset_shapes: Mapping[str, Any],
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
        "outputs": {
            "component_table_parquet": str(output_dir / "component_table.parquet"),
            "pulse_summary_parquet": str(output_dir / "pulse_summary.parquet"),
            "sweep_summary_csv": str(output_dir / "sweep_summary.csv"),
            "run_metadata_json": str(output_dir / "run_metadata.json"),
            "diagnostic_figures_dir": str(output_dir / "diagnostic_figures"),
        },
        "dataset_shapes": dataset_shapes,
        "grid_status": {
            "n_records": index_info.n_records,
            "n_sweeps": index_info.n_sweeps,
            "n_tracks": index_info.n_tracks,
            "complete_rectangular_grid": index_info.complete_rectangular_grid,
            "duplicate_sweep_track_cells": index_info.duplicate_sweep_track_cells,
            "missing_sweep_track_cells": index_info.missing_sweep_track_cells,
        },
        "selected_domain": {
            "selected_sweeps": selected_sweeps.tolist(),
            "selected_tracks": selected_tracks.tolist(),
            "n_selected_sweeps": int(selected_sweeps.size),
            "n_selected_tracks": int(selected_tracks.size),
        },
        "scientific_notes": COMMON_SCIENTIFIC_NOTES,
        "caveats": [
            "tentative_height_offset_from_refh is not a georeferenced secondary return.",
        ],
    }


def process_one_file(cfg: Config, input_h5_path: Path) -> dict[str, Any]:
    require_scipy()

    h5_path = resolve_path(input_h5_path)
    output_root = resolve_path(cfg.output_root)
    output_dir = output_root / h5_path.stem
    diag_dir = output_dir / "diagnostic_figures"
    ensure_dir(output_dir)
    if cfg.write_diagnostic_png:
        ensure_dir(diag_dir)

    component_path = output_dir / "component_table.parquet"
    pulse_path = output_dir / "pulse_summary.parquet"
    sweep_path = output_dir / "sweep_summary.csv"
    metadata_path = output_dir / "run_metadata.json"

    component_writer = ParquetChunkWriter(component_path, COMPONENT_COLUMNS)
    pulse_writer = ParquetChunkWriter(pulse_path, PULSE_COLUMNS)
    sweep_rows: list[dict[str, Any]] = []
    total_pulses = 0

    with h5py.File(h5_path, "r") as h5:
        rx_ds = require_dataset(h5, "rx_waveform")
        index_info = build_record_index_grid(h5)
        if index_info.duplicate_sweep_track_cells > 0:
            raise ValueError(
                f"Detected {index_info.duplicate_sweep_track_cells} duplicate sweep-track cells; "
                "the waveform feature extraction workflow requires unique pulse indexing."
            )

        fields = read_required_fields(h5, index_info.n_records)
        selected_sweeps = choose_sweeps(cfg, index_info)
        selected_tracks = choose_tracks(cfg, index_info)
        diagnostic_sweeps = choose_diagnostic_sweeps(cfg, selected_sweeps)
        detector_cfg = build_detector_config(cfg)

        rx_axis = infer_waveform_record_axis(rx_ds, index_info.n_records)
        dataset_shapes = collect_dataset_shapes(
            h5,
            [
                "sweep_num",
                "track_num",
                "rx_waveform",
                "tx_waveform",
                "refh",
                "refh_longitude",
                "refh_latitude",
                "refh_amp",
                "refh_snr",
                "good_snr",
                "bg_mean",
                "bg_std",
                "refh_thres",
                "rwstart",
                "rwstop",
            ],
        )
        dataset_shapes["rx_waveform"]["record_axis"] = int(rx_axis)
        tx_info = maybe_read_tx_shape(h5, index_info)
        if tx_info is not None:
            dataset_shapes["tx_waveform"] = tx_info

        metadata = build_metadata(cfg, h5_path, output_dir, index_info, dataset_shapes, selected_sweeps, selected_tracks)
        metadata["h5_attrs_subset"] = read_attrs_subset(h5)

        print(f"H5: {h5_path}")
        print(f"Output dir: {output_dir}")
        print(f"Selected sweeps: {selected_sweeps.tolist()}")

        for sweep_num in selected_sweeps:
            rx_matrix_full, _, record_indices_full = read_sweep_matrix(h5, "rx_waveform", int(sweep_num), index_info)
            rx_matrix = rx_matrix_full[selected_tracks]
            record_indices = record_indices_full[selected_tracks]
            valid_mask = record_indices >= 0
            valid_tracks = selected_tracks[valid_mask]
            missing_tracks = selected_tracks[~valid_mask]

            print(
                f"Processing sweep {int(sweep_num)}: valid_tracks={int(valid_tracks.size)}, "
                f"missing_tracks={int(missing_tracks.size)}"
            )
            if valid_tracks.size == 0:
                continue

            corrected_matrix = np.full_like(rx_matrix, np.nan, dtype=np.float32)
            component_rows: list[dict[str, Any]] = []
            pulse_rows: list[dict[str, Any]] = []

            for local_idx, track_num in enumerate(selected_tracks):
                pulse_index = int(record_indices[local_idx])
                if pulse_index < 0:
                    continue

                y_raw = np.asarray(rx_matrix[local_idx], dtype=np.float64)
                bg_mean = float(fields["bg_mean"][pulse_index])
                bg_std = float(fields["bg_std"][pulse_index])
                refh_thres = float(fields["refh_thres"][pulse_index])

                candidate_components = detect_candidate_components_1d(
                    y_raw=y_raw,
                    bg_mean=bg_mean,
                    bg_std=bg_std,
                    refh_thres=refh_thres,
                    config=detector_cfg,
                )
                components = classify_prominent_components(candidate_components, detector_cfg)

                corrected = y_raw - bg_mean if cfg.background_mode == "minus_bg_mean" else (
                    y_raw - refh_thres if cfg.background_mode == "minus_refh_thres" else y_raw.copy()
                )
                if cfg.clip_negative_after_background:
                    corrected = np.where(np.isfinite(corrected), np.maximum(corrected, 0.0), np.nan)
                corrected_matrix[local_idx] = corrected.astype(np.float32, copy=False)

                raw_argmax_bin, raw_argmax_amp = safe_argmax_bin(y_raw)
                corrected_argmax_bin, corrected_argmax_amp = safe_argmax_bin(corrected)
                pulse_summary = summarize_pulse_waveform_features(
                    components=components,
                    raw_argmax_bin=raw_argmax_bin,
                    corrected_argmax_bin=corrected_argmax_bin,
                    refh_amp=float(fields["refh_amp"][pulse_index]),
                    refh_snr=float(fields["refh_snr"][pulse_index]),
                    good_snr=bool(fields["good_snr"][pulse_index]),
                    bg_mean=bg_mean,
                    bg_std=bg_std,
                    config=detector_cfg,
                )

                main_peak_bin = pulse_summary["main_peak_bin"]
                for component in components:
                    tentative_height = compute_range_window_height_candidate(
                        peak_bin=component["peak_bin"],
                        n_rx_bins=rx_matrix.shape[1],
                        rwstart=float(fields["rwstart"][pulse_index]),
                        rwstop=float(fields["rwstop"][pulse_index]),
                        direction="rwstart_to_rwstop",
                    )
                    component_rows.append(
                        {
                            "h5_filename": h5_path.name,
                            "pulse_index": pulse_index,
                            "sweep_num": int(sweep_num),
                            "track_num": int(track_num),
                            "component_rank": int(component["rank_by_prominence"]),
                            "peak_bin": float(component["peak_bin"]),
                            "amplitude_raw": float(component["amplitude_raw"]),
                            "amplitude_processed": float(component["amplitude_processed"]),
                            "prominence": float(component["prominence"]),
                            "width_bins": float(component["width_bins"]),
                            "left_ips": float(component["left_ips"]),
                            "right_ips": float(component["right_ips"]),
                            "area_processed": float(component["area_processed"]),
                            "is_main_component": bool(component["is_main_component"]),
                            "is_prominent_component": bool(component["is_prominent_component"]),
                            "is_valid_secondary_candidate": bool(component["is_valid_secondary_candidate"]),
                            "rejected_reason": str(component["rejected_reason"]),
                            "rel_bin_to_main": float(component["peak_bin"] - main_peak_bin) if np.isfinite(main_peak_bin) else np.nan,
                            "rel_bin_to_raw_argmax": float(component["peak_bin"] - raw_argmax_bin) if np.isfinite(raw_argmax_bin) else np.nan,
                            "tentative_height_rwstart_to_rwstop": tentative_height,
                            "tentative_height_offset_from_refh": tentative_height - float(fields["refh"][pulse_index]) if np.isfinite(tentative_height) else np.nan,
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
                        "raw_argmax_bin": raw_argmax_bin,
                        "raw_argmax_amp": raw_argmax_amp,
                        "corrected_argmax_bin": corrected_argmax_bin,
                        "corrected_argmax_amp": corrected_argmax_amp,
                        "n_candidate_components": pulse_summary["n_candidate_components"],
                        "n_prominent_components": pulse_summary["n_prominent_components"],
                        "n_valid_secondary_components": pulse_summary["n_valid_secondary_components"],
                        "main_peak_bin": pulse_summary["main_peak_bin"],
                        "main_peak_prominence": pulse_summary["main_peak_prominence"],
                        "main_peak_width_bins": pulse_summary["main_peak_width_bins"],
                        "main_peak_area": pulse_summary["main_peak_area"],
                        "secondary_peak_bin": pulse_summary["secondary_peak_bin"],
                        "secondary_to_main_amp_ratio": pulse_summary["secondary_to_main_amp_ratio"],
                        "secondary_to_main_prominence_ratio": pulse_summary["secondary_to_main_prominence_ratio"],
                        "has_clear_main_peak": pulse_summary["has_clear_main_peak"],
                        "has_valid_secondary_peak": pulse_summary["has_valid_secondary_peak"],
                        "has_broad_main_peak": pulse_summary["has_broad_main_peak"],
                        "has_weak_main_peak": pulse_summary["has_weak_main_peak"],
                        "low_snr_flag": pulse_summary["low_snr_flag"],
                        "waveform_reliability_class": pulse_summary["waveform_reliability_class"],
                        "fixed_bin_stripe_score": np.nan,
                        "continuity_residual_bins": np.nan,
                        "refh": float(fields["refh"][pulse_index]),
                        "refh_lon": float(fields["refh_longitude"][pulse_index]),
                        "refh_lat": float(fields["refh_latitude"][pulse_index]),
                        "refh_amp": float(fields["refh_amp"][pulse_index]),
                        "refh_snr": float(fields["refh_snr"][pulse_index]),
                        "good_snr": bool(fields["good_snr"][pulse_index]),
                        "bg_mean": bg_mean,
                        "bg_std": bg_std,
                        "refh_thres": refh_thres,
                        "rwstart": float(fields["rwstart"][pulse_index]),
                        "rwstop": float(fields["rwstop"][pulse_index]),
                    }
                )

            if not pulse_rows:
                continue

            pulse_df = rows_to_dataframe(pulse_rows, PULSE_COLUMNS)
            component_df = rows_to_dataframe(component_rows, COMPONENT_COLUMNS)

            continuity = compute_sweep_continuity_features(
                track=pulse_df["track_num"].to_numpy(dtype=np.float64),
                main_peak_bin=pulse_df["main_peak_bin"].to_numpy(dtype=np.float64),
            )
            stripe = compute_fixed_bin_stripe_features(
                track=pulse_df["track_num"].to_numpy(dtype=np.float64),
                peak_bin=pulse_df["main_peak_bin"].to_numpy(dtype=np.float64),
                bin_tolerance=cfg.stripe_bin_tolerance,
                min_fraction=cfg.stripe_min_fraction,
            )

            main_peak = pulse_df["main_peak_bin"].to_numpy(dtype=np.float64)
            valid = np.isfinite(main_peak)
            continuity_residual = np.full(main_peak.shape, np.nan, dtype=np.float64)
            if np.any(valid):
                ordered_track = pulse_df.loc[valid, "track_num"].to_numpy(dtype=np.float64)
                ordered_peak = main_peak[valid]
                order = np.argsort(ordered_track)
                ordered_peak = ordered_peak[order]
                baseline = pd.Series(ordered_peak).rolling(window=9, center=True, min_periods=1).median().to_numpy(dtype=np.float64)
                residual = ordered_peak - baseline
                continuity_residual[np.flatnonzero(valid)[order]] = residual
            pulse_df["continuity_residual_bins"] = continuity_residual
            pulse_df["fixed_bin_stripe_score"] = stripe["fixed_bin_fraction"]
            pulse_rows = pulse_df.to_dict(orient="records")

            if cfg.write_component_table:
                component_writer.write_rows(component_rows)
            if cfg.write_pulse_summary:
                pulse_writer.write_rows(pulse_rows)

            sweep_rows.append(
                {
                    "h5_filename": h5_path.name,
                    "sweep_num": int(sweep_num),
                    "n_tracks": int(selected_tracks.size),
                    "n_valid_pulses": int(len(pulse_rows)),
                    "median_refh_snr": finite_median(pulse_df["refh_snr"].to_numpy(dtype=np.float64)),
                    "fraction_good_snr": finite_mean(pulse_df["good_snr"].astype("Float64")),
                    "median_main_peak_prominence": finite_median(pulse_df["main_peak_prominence"].to_numpy(dtype=np.float64)),
                    "median_main_peak_width_bins": finite_median(pulse_df["main_peak_width_bins"].to_numpy(dtype=np.float64)),
                    "median_n_candidate_components": finite_median(pulse_df["n_candidate_components"].to_numpy(dtype=np.float64)),
                    "median_n_prominent_components": finite_median(pulse_df["n_prominent_components"].to_numpy(dtype=np.float64)),
                    "fraction_clear_main_peak": finite_mean(pulse_df["has_clear_main_peak"].astype("Float64")),
                    "fraction_valid_secondary_peak": finite_mean(pulse_df["has_valid_secondary_peak"].astype("Float64")),
                    "fraction_weak_main_peak": finite_mean(pulse_df["has_weak_main_peak"].astype("Float64")),
                    "fraction_broad_main_peak": finite_mean(pulse_df["has_broad_main_peak"].astype("Float64")),
                    "fixed_bin_mode": stripe["fixed_bin_mode"],
                    "fixed_bin_fraction": stripe["fixed_bin_fraction"],
                    "stripe_like_flag": stripe["stripe_like_flag"],
                    "continuity_median_abs_residual": continuity["continuity_median_abs_residual"],
                    "continuity_p95_abs_residual": continuity["continuity_p95_abs_residual"],
                    "continuity_discontinuity_count": continuity["continuity_discontinuity_count"],
                }
            )
            total_pulses += len(pulse_rows)

            if cfg.write_diagnostic_png and int(sweep_num) in diagnostic_sweeps:
                make_diagnostic_figure(
                    out_path=diag_dir / f"sweep_{int(sweep_num):05d}_waveform_features.png",
                    h5_filename=h5_path.name,
                    sweep_num=int(sweep_num),
                    track_axis=selected_tracks.astype(np.int64),
                    rx_matrix=np.asarray(rx_matrix, dtype=np.float64),
                    corrected_matrix=np.asarray(corrected_matrix, dtype=np.float64),
                    pulse_df=pulse_df,
                    component_df=component_df,
                    stripe_features=stripe,
                    continuity_features=continuity,
                )

            prominent_counts = pulse_df["n_prominent_components"].to_numpy(dtype=np.float64)
            print(
                f"Finished sweep {int(sweep_num)}: pulses={len(pulse_rows)}, "
                f"components={len(component_rows)}, median prominent={finite_median(prominent_counts):.2f}"
            )

    if cfg.write_component_table:
        component_writer.finalize()
    if cfg.write_pulse_summary:
        pulse_writer.finalize()
    if cfg.write_sweep_summary:
        rows_to_dataframe(sweep_rows, SWEEP_COLUMNS).to_csv(sweep_path, index=False)

    metadata["processed_counts"] = {
        "processed_sweeps": int(len(sweep_rows)),
        "processed_tracks_per_sweep_target": int(len(selected_tracks)),
        "processed_pulses": int(total_pulses),
    }
    metadata_path.write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")

    return {
        "output_dir": output_dir,
        "component_table": component_path,
        "pulse_summary": pulse_path,
        "sweep_summary": sweep_path,
        "run_metadata": metadata_path,
    }


def run_all(cfg: Config) -> list[dict[str, Any]]:
    results = []
    for h5_path in cfg.input_h5_paths:
        results.append(process_one_file(cfg, h5_path))
    return results


def main() -> None:
    cfg = Config(
        input_h5_paths=[
            Path("./casals_h5_downloads/casals_l1b_20241118T171757_001_02.h5"),
        ],
        output_root=Path("./outputs/extract_waveform_features"),
        sweep_start=5000,
        sweep_end=5002,
        sweep_step=1,
        selected_sweeps=None,
        selected_tracks=None,
        max_sweeps=None,
        background_mode="minus_bg_mean",
        clip_negative_after_background=True,
        smoothing_method="savgol",
        smoothing_window=7,
        smoothing_polyorder=2,
        min_height_sigma=3.0,
        min_prominence_sigma=4.0,
        min_width_bins=1.0,
        max_width_bins=80.0,
        min_distance_bins=5,
        max_candidate_components_per_pulse=8,
        min_prominent_prominence_sigma=5.0,
        min_prominent_relative_to_main=0.20,
        min_secondary_separation_bins=8,
        broad_main_width_bins=30.0,
        weak_main_prominence_sigma=4.0,
        edge_exclusion_bins=10,
        stripe_bin_tolerance=2,
        stripe_min_fraction=0.20,
        write_component_table=True,
        write_pulse_summary=True,
        write_sweep_summary=True,
        write_diagnostic_png=True,
        diagnostic_sweeps=None,
        random_seed=42,
    )
    run_all(cfg)


if __name__ == "__main__":
    main()
