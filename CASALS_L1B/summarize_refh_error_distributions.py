"""Summarize CASALS L1B refh SNR and error distributions from one H5 file.

Scientific meaning:
    Each record is one official CASALS L1B geolocated refh reference-return
    point, corresponding to the Rx waveform maximum-amplitude bin.

Outputs:
    Metadata JSON, a compact per-filter CSV summary, full-file histogram PNGs,
    and a multi-panel comparison PNG across named filters.

This script does not:
    - create point-cloud products,
    - classify ground or non-ground points,
    - resolve vertical datum or reference-frame differences.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np


CONFIG = {
    "h5_path": Path("./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
    "output_dir": Path("./outputs/summarize_refh_error_distributions"),
    "horizontal_error_conversion": {
        "meters_per_degree_lat": 111320.0,
    },
    "filters": [
        {
            "name": "all_valid",
            "use_good_snr_only": False,
            "snr_min": None,
            "snr_max": None,
            "amp_min": None,
            "amp_max": None,
            "track_range": None,
            "sweep_range": None,
            "bbox_lonlat": None,
        },
        {
            "name": "good_snr",
            "use_good_snr_only": True,
            "snr_min": None,
            "snr_max": None,
            "amp_min": None,
            "amp_max": None,
            "track_range": None,
            "sweep_range": None,
            "bbox_lonlat": None,
        },
        {
            "name": "snr_gt_5",
            "use_good_snr_only": False,
            "snr_min": 5.0,
            "snr_max": None,
            "amp_min": None,
            "amp_max": None,
            "track_range": None,
            "sweep_range": None,
            "bbox_lonlat": None,
        },
    ],
    "histograms": {
        "bins": 120,
        "full_file_percentile_clip": {
            "snr": None,
            "vertical_error_m": (0.5, 99.5),
            "horizontal_error_m": (0.5, 99.5),
        },
        "comparison_percentile_clip": {
            "vertical_error_m": (0.5, 99.5),
            "horizontal_error_m": (0.5, 99.5),
        },
        "dpi": 220,
    },
    "thresholds": {
        "vertical_error_abs_m": [0.5, 1.0],
        "horizontal_error_m": [1.0, 2.0],
    },
}


def normalize_h5_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def find_dataset(h5: h5py.File, basename: str, required: bool = True) -> Optional[h5py.Dataset]:
    if basename in h5 and isinstance(h5[basename], h5py.Dataset):
        return h5[basename]

    matches: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == basename:
            matches.append(name)

    h5.visititems(visitor)
    if len(matches) == 1:
        return h5[matches[0]]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple datasets matched {basename!r}: {matches}")
    if required:
        raise KeyError(f"Required dataset not found: {basename}")
    return None


def read_1d_array(
    h5: h5py.File,
    basename: str,
    *,
    required: bool,
    dtype: Any,
    n_expected: Optional[int] = None,
) -> Optional[np.ndarray]:
    ds = find_dataset(h5, basename, required=required)
    if ds is None:
        return None
    arr = np.asarray(ds[...], dtype=dtype).reshape(-1)
    if n_expected is not None and arr.size != n_expected:
        raise ValueError(f"Dataset {basename!r} has size {arr.size}, expected {n_expected}.")
    return arr


def get_refh_snr(amp: Optional[np.ndarray], thres: Optional[np.ndarray], snr_stored: Optional[np.ndarray]) -> np.ndarray:
    if snr_stored is not None:
        return snr_stored.astype(np.float64)
    if amp is None or thres is None:
        raise KeyError("Neither refh_snr nor refh_amp/refh_thres is available.")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(amp.astype(np.float64), thres.astype(np.float64))


def read_h5_data(h5_path: Path) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        lon = read_1d_array(h5, "refh_longitude", required=True, dtype=np.float64)
        assert lon is not None
        n = lon.size
        lat = read_1d_array(h5, "refh_latitude", required=True, dtype=np.float64, n_expected=n)
        refh = read_1d_array(h5, "refh", required=True, dtype=np.float64, n_expected=n)
        refh_amp = read_1d_array(h5, "refh_amp", required=False, dtype=np.float64, n_expected=n)
        refh_thres = read_1d_array(h5, "refh_thres", required=False, dtype=np.float64, n_expected=n)
        refh_snr_stored = read_1d_array(h5, "refh_snr", required=False, dtype=np.float64, n_expected=n)
        good_snr = read_1d_array(h5, "good_snr", required=False, dtype=np.uint8, n_expected=n)
        track_num = read_1d_array(h5, "track_num", required=False, dtype=np.int64, n_expected=n)
        sweep_num = read_1d_array(h5, "sweep_num", required=False, dtype=np.int64, n_expected=n)
        refh_error = read_1d_array(h5, "refh_error", required=True, dtype=np.float64, n_expected=n)
        refh_lon_error = read_1d_array(h5, "refh_longitude_error", required=True, dtype=np.float64, n_expected=n)
        refh_lat_error = read_1d_array(h5, "refh_latitude_error", required=True, dtype=np.float64, n_expected=n)
        attrs = {k: normalize_h5_attr(v) for k, v in h5.attrs.items()}

    assert lat is not None and refh is not None
    refh_snr = get_refh_snr(refh_amp, refh_thres, refh_snr_stored)
    meters_per_degree_lat = float(CONFIG["horizontal_error_conversion"]["meters_per_degree_lat"])
    with np.errstate(invalid="ignore"):
        dx_m = refh_lon_error * meters_per_degree_lat * np.cos(np.deg2rad(lat))
    dy_m = refh_lat_error * meters_per_degree_lat
    horizontal_error_m = np.sqrt(dx_m * dx_m + dy_m * dy_m)

    return {
        "lon": lon,
        "lat": lat,
        "refh": refh,
        "refh_amp": refh_amp,
        "refh_thres": refh_thres,
        "refh_snr": refh_snr,
        "good_snr": good_snr.astype(bool) if good_snr is not None else None,
        "track_num": track_num,
        "sweep_num": sweep_num,
        "vertical_error_m": refh_error,
        "horizontal_error_m": horizontal_error_m,
        "refh_longitude_error_deg": refh_lon_error,
        "refh_latitude_error_deg": refh_lat_error,
        "attrs": attrs,
    }


def build_base_valid_mask(data: dict[str, Any]) -> np.ndarray:
    mask = np.ones(data["refh"].size, dtype=bool)
    mask &= np.isfinite(data["lon"])
    mask &= np.isfinite(data["lat"])
    mask &= np.isfinite(data["refh"])
    mask &= np.isfinite(data["refh_snr"])
    mask &= np.isfinite(data["vertical_error_m"])
    mask &= np.isfinite(data["horizontal_error_m"])
    mask &= (data["lon"] >= -180.0) & (data["lon"] <= 180.0)
    mask &= (data["lat"] >= -90.0) & (data["lat"] <= 90.0)
    return mask


def build_filter_mask(data: dict[str, Any], base_mask: np.ndarray, filter_cfg: dict[str, Any]) -> tuple[np.ndarray, str]:
    mask = base_mask.copy()
    status = "ok"

    if filter_cfg.get("use_good_snr_only"):
        if data["good_snr"] is None:
            warnings.warn(f"Filter {filter_cfg['name']!r} requested good_snr, but dataset is missing; filter skipped.")
            return np.zeros_like(mask, dtype=bool), "skipped_missing_good_snr"
        mask &= data["good_snr"]

    snr_min = filter_cfg.get("snr_min")
    snr_max = filter_cfg.get("snr_max")
    if snr_min is not None:
        mask &= data["refh_snr"] >= float(snr_min)
    if snr_max is not None:
        mask &= data["refh_snr"] <= float(snr_max)

    amp_min = filter_cfg.get("amp_min")
    amp_max = filter_cfg.get("amp_max")
    if amp_min is not None or amp_max is not None:
        if data["refh_amp"] is None:
            warnings.warn(f"Filter {filter_cfg['name']!r} requested amplitude thresholds, but refh_amp is missing.")
            return np.zeros_like(mask, dtype=bool), "skipped_missing_refh_amp"
        mask &= np.isfinite(data["refh_amp"])
        if amp_min is not None:
            mask &= data["refh_amp"] >= float(amp_min)
        if amp_max is not None:
            mask &= data["refh_amp"] <= float(amp_max)

    track_range = filter_cfg.get("track_range")
    if track_range is not None:
        if data["track_num"] is None:
            warnings.warn(f"Filter {filter_cfg['name']!r} requested track_range, but track_num is missing.")
            return np.zeros_like(mask, dtype=bool), "skipped_missing_track_num"
        lo, hi = track_range
        mask &= data["track_num"] >= int(lo)
        mask &= data["track_num"] <= int(hi)

    sweep_range = filter_cfg.get("sweep_range")
    if sweep_range is not None:
        if data["sweep_num"] is None:
            warnings.warn(f"Filter {filter_cfg['name']!r} requested sweep_range, but sweep_num is missing.")
            return np.zeros_like(mask, dtype=bool), "skipped_missing_sweep_num"
        lo, hi = sweep_range
        mask &= data["sweep_num"] >= int(lo)
        mask &= data["sweep_num"] <= int(hi)

    bbox = filter_cfg.get("bbox_lonlat")
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        mask &= data["lon"] >= float(lon_min)
        mask &= data["lon"] <= float(lon_max)
        mask &= data["lat"] >= float(lat_min)
        mask &= data["lat"] <= float(lat_max)

    return mask, status


def metric_summary(values: np.ndarray, mask: np.ndarray, *, abs_value: bool, thresholds: list[float]) -> dict[str, Any]:
    vals = values[mask]
    vals = vals[np.isfinite(vals)]
    if abs_value:
        vals = np.abs(vals)
    if vals.size == 0:
        return {
            "n_valid": 0,
            "min": None,
            "p02": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p98": None,
            "max": None,
            "mean": None,
            "std": None,
            "fractions_below_threshold": {str(t): None for t in thresholds},
        }
    return {
        "n_valid": int(vals.size),
        "min": float(np.nanmin(vals)),
        "p02": float(np.nanpercentile(vals, 2)),
        "p25": float(np.nanpercentile(vals, 25)),
        "p50": float(np.nanpercentile(vals, 50)),
        "p75": float(np.nanpercentile(vals, 75)),
        "p98": float(np.nanpercentile(vals, 98)),
        "max": float(np.nanmax(vals)),
        "mean": float(np.nanmean(vals)),
        "std": float(np.nanstd(vals)),
        "fractions_below_threshold": {str(t): float(np.mean(vals < float(t))) for t in thresholds},
    }


def clip_values(values: np.ndarray, percentile_clip: Optional[tuple[float, float]]) -> np.ndarray:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return vals
    if percentile_clip is None:
        return vals
    lo, hi = percentile_clip
    v_lo = float(np.nanpercentile(vals, lo))
    v_hi = float(np.nanpercentile(vals, hi))
    return vals[(vals >= v_lo) & (vals <= v_hi)]


def plot_single_histogram(
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int,
    percentile_clip: Optional[tuple[float, float]],
    dpi: int,
    abs_value: bool = False,
) -> None:
    vals = values[np.isfinite(values)]
    if abs_value:
        vals = np.abs(vals)
    vals = clip_values(vals, percentile_clip)
    if vals.size == 0:
        warnings.warn(f"No finite values available for histogram: {output_path.name}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=bins, color="0.35")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_filter_comparison(
    filter_results: list[dict[str, Any]],
    hist_cfg: dict[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap("tab10")

    vertical_all = np.concatenate(
        [np.abs(fr["vertical_values"]) for fr in filter_results if fr["status"] == "ok" and fr["vertical_values"].size > 0]
    ) if any(fr["status"] == "ok" and fr["vertical_values"].size > 0 for fr in filter_results) else np.array([])
    horizontal_all = np.concatenate(
        [fr["horizontal_values"] for fr in filter_results if fr["status"] == "ok" and fr["horizontal_values"].size > 0]
    ) if any(fr["status"] == "ok" and fr["horizontal_values"].size > 0 for fr in filter_results) else np.array([])

    vertical_clip = hist_cfg["comparison_percentile_clip"]["vertical_error_m"]
    horizontal_clip = hist_cfg["comparison_percentile_clip"]["horizontal_error_m"]
    vertical_range = clip_values(vertical_all, vertical_clip) if vertical_all.size else np.array([])
    horizontal_range = clip_values(horizontal_all, horizontal_clip) if horizontal_all.size else np.array([])

    if vertical_range.size:
        v_lo = float(np.nanmin(vertical_range))
        v_hi = float(np.nanmax(vertical_range))
    else:
        v_lo, v_hi = 0.0, 1.0
    if horizontal_range.size:
        h_lo = float(np.nanmin(horizontal_range))
        h_hi = float(np.nanmax(horizontal_range))
    else:
        h_lo, h_hi = 0.0, 1.0

    for i, fr in enumerate(filter_results):
        if fr["status"] != "ok" or fr["point_count"] == 0:
            continue
        color = cmap(i % 10)
        vertical_vals = clip_values(np.abs(fr["vertical_values"]), vertical_clip)
        horizontal_vals = clip_values(fr["horizontal_values"], horizontal_clip)
        if vertical_vals.size:
            axes[0].hist(vertical_vals, bins=hist_cfg["bins"], range=(v_lo, v_hi), histtype="step", linewidth=1.5, label=fr["name"], color=color)
        if horizontal_vals.size:
            axes[1].hist(horizontal_vals, bins=hist_cfg["bins"], range=(h_lo, h_hi), histtype="step", linewidth=1.5, label=fr["name"], color=color)

    axes[0].set_title("Absolute vertical error by filter")
    axes[0].set_xlabel("|refh_error| (m)")
    axes[0].set_ylabel("count")
    axes[1].set_title("Horizontal error by filter")
    axes[1].set_xlabel("horizontal error (m)")
    axes[1].set_ylabel("count")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=hist_cfg["dpi"])
    plt.close(fig)


def flatten_filter_row(filter_result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "filter_name": filter_result["name"],
        "status": filter_result["status"],
        "point_count": filter_result["point_count"],
    }
    for metric_name in ("snr", "vertical_error_m", "horizontal_error_m"):
        metric = filter_result[metric_name]
        row[f"{metric_name}_n_valid"] = metric["n_valid"]
        for key in ("min", "p02", "p25", "p50", "p75", "p98", "max", "mean", "std"):
            row[f"{metric_name}_{key}"] = metric[key]
        for threshold, fraction in metric["fractions_below_threshold"].items():
            row[f"{metric_name}_fraction_lt_{threshold}"] = fraction
    return row


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def strip_plot_arrays(filter_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata_filters: list[dict[str, Any]] = []
    for result in filter_results:
        metadata_filters.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"vertical_values", "horizontal_values"}
            }
        )
    return metadata_filters


def main() -> None:
    cfg = CONFIG
    h5_path = Path(cfg["h5_path"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not h5_path.exists():
        raise FileNotFoundError(f"Input H5 does not exist: {h5_path}")

    print(f"Reading H5: {h5_path}")
    data = read_h5_data(h5_path)
    base_mask = build_base_valid_mask(data)
    print(f"Base-valid points: {int(np.sum(base_mask)):,} / {data['refh'].size:,}")

    hist_cfg = cfg["histograms"]
    thresholds_cfg = cfg["thresholds"]

    filter_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for filter_cfg in cfg["filters"]:
        name = str(filter_cfg["name"])
        print(f"Summarizing filter: {name}")
        mask, status = build_filter_mask(data, base_mask, filter_cfg)
        point_count = int(np.sum(mask))

        result = {
            "name": name,
            "status": status,
            "filter_config": filter_cfg,
            "point_count": point_count,
            "snr": metric_summary(data["refh_snr"], mask, abs_value=False, thresholds=[]),
            "vertical_error_m": metric_summary(
                data["vertical_error_m"],
                mask,
                abs_value=True,
                thresholds=[float(t) for t in thresholds_cfg["vertical_error_abs_m"]],
            ),
            "horizontal_error_m": metric_summary(
                data["horizontal_error_m"],
                mask,
                abs_value=False,
                thresholds=[float(t) for t in thresholds_cfg["horizontal_error_m"]],
            ),
            "vertical_values": np.abs(data["vertical_error_m"][mask & np.isfinite(data["vertical_error_m"])]),
            "horizontal_values": data["horizontal_error_m"][mask & np.isfinite(data["horizontal_error_m"])],
        }

        if status == "ok" and point_count == 0:
            result["status"] = "ok_zero_points"

        filter_results.append(result)
        csv_rows.append(flatten_filter_row(result))

    plot_single_histogram(
        data["refh_snr"][base_mask],
        title="CASALS refh_snr histogram",
        xlabel="refh_snr",
        output_path=output_dir / "full_file_refh_snr_hist.png",
        bins=int(hist_cfg["bins"]),
        percentile_clip=hist_cfg["full_file_percentile_clip"]["snr"],
        dpi=int(hist_cfg["dpi"]),
        abs_value=False,
    )
    plot_single_histogram(
        data["vertical_error_m"][base_mask],
        title="CASALS absolute refh vertical error histogram",
        xlabel="|refh_error| (m)",
        output_path=output_dir / "full_file_vertical_error_hist.png",
        bins=int(hist_cfg["bins"]),
        percentile_clip=hist_cfg["full_file_percentile_clip"]["vertical_error_m"],
        dpi=int(hist_cfg["dpi"]),
        abs_value=True,
    )
    plot_single_histogram(
        data["horizontal_error_m"][base_mask],
        title="CASALS horizontal error histogram",
        xlabel="horizontal error (m)",
        output_path=output_dir / "full_file_horizontal_error_hist.png",
        bins=int(hist_cfg["bins"]),
        percentile_clip=hist_cfg["full_file_percentile_clip"]["horizontal_error_m"],
        dpi=int(hist_cfg["dpi"]),
        abs_value=False,
    )
    plot_filter_comparison(
        filter_results,
        hist_cfg=hist_cfg,
        output_path=output_dir / "filter_comparison_vertical_horizontal_error.png",
    )

    csv_path = output_dir / "filter_summary.csv"
    write_csv_rows(csv_path, csv_rows)
    metadata_filters = strip_plot_arrays(filter_results)

    metadata = {
        "script": "summarize_refh_error_distributions.py",
        "source_h5": str(h5_path.resolve()),
        "scientific_notes": [
            "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
            "refh is WGS84 ellipsoidal height unless otherwise documented.",
            "This is not an official multi-return point cloud.",
            "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            "These are descriptive H5-only error summaries, not DEM or terrain-quality metrics.",
        ],
        "base_valid_count": int(np.sum(base_mask)),
        "n_total_records": int(data["refh"].size),
        "horizontal_error_definition": {
            "dx_m": "refh_longitude_error_deg * 111320 * cos(latitude_rad)",
            "dy_m": "refh_latitude_error_deg * 111320",
            "horizontal_error_m": "sqrt(dx_m^2 + dy_m^2)",
        },
        "config": {
            "h5_path": str(cfg["h5_path"]),
            "output_dir": str(cfg["output_dir"]),
            "horizontal_error_conversion": cfg["horizontal_error_conversion"],
            "filters": cfg["filters"],
            "histograms": cfg["histograms"],
            "thresholds": cfg["thresholds"],
        },
        "source_global_attributes_subset": {
            k: data["attrs"].get(k)
            for k in ("start_utca", "end_utca", "n_pulses", "n_sweeps", "n_tracks", "n_rx_bins")
        },
        "filters": metadata_filters,
        "outputs": {
            "metadata_json": str((output_dir / "summary_metadata.json").resolve()),
            "filter_summary_csv": str(csv_path.resolve()),
            "full_file_refh_snr_hist_png": str((output_dir / "full_file_refh_snr_hist.png").resolve()),
            "full_file_vertical_error_hist_png": str((output_dir / "full_file_vertical_error_hist.png").resolve()),
            "full_file_horizontal_error_hist_png": str((output_dir / "full_file_horizontal_error_hist.png").resolve()),
            "filter_comparison_png": str((output_dir / "filter_comparison_vertical_horizontal_error.png").resolve()),
        },
    }

    metadata_path = output_dir / "summary_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
