"""
Baseline CASALS refh point-cloud classification and pseudo-ground-truth evaluation.

This is a baseline rule-based CASALS refh classifier.
CASALS L1B provides one official refh point per pulse, not an official multi-return
classified point cloud.
The transferred 3DEP LAZ is used as pseudo-ground-truth for evaluation, not absolute
ground truth.
No vertical datum correction is applied by this script.
"""

from __future__ import annotations

import json
import math
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import h5py
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from laspy import ExtraBytesParams
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


LABEL_ORDER = [1, 2, 7]
CLASS_NAME_MAP = {
    1: "processed_unclassified",
    2: "ground",
    7: "noise",
}
CLASS_REASON_MAP = {
    0: "unknown",
    1: "ground_abs_hag_within_tol",
    2: "noise_invalid_dtm",
    3: "noise_below_ground",
    4: "noise_above_40m",
    5: "processed_unclassified_valid_hag",
    6: "noise_low_density",
}
EPSG_RE = re.compile(r"EPSG[:\s]*([0-9]{4,6})", re.IGNORECASE)
UTM_RE = re.compile(r"UTM(?:\s+ZONE)?[:\s]*([0-9]{1,2})([NS])?", re.IGNORECASE)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, CRS):
        return obj.to_string()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def safe_json_dump(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=False), encoding="utf-8")


def collect_class_counts(values: np.ndarray) -> Dict[int, int]:
    arr = np.asarray(values, dtype=np.uint8)
    if arr.size == 0:
        return {}
    uniq, counts = np.unique(arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("finite_mask requires at least one array")
    mask = np.ones(np.asarray(arrays[0]).shape[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(np.asarray(arr))
    return mask


def summarize_values(values: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {
            "n": int(arr.size),
            "n_finite": 0,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    arr = arr[finite]
    return {
        "n": int(values.size),
        "n_finite": int(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def maybe_quantiles(values: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float]]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None, None, None
    arr = arr[finite]
    return float(np.percentile(arr, 5)), float(np.median(arr)), float(np.percentile(arr, 95))


def print_console_metrics_table(
    h5_stem: str,
    evaluation_metrics: Dict[str, Any],
) -> None:
    report_df = evaluation_metrics["primary_report_df"]
    confusion_df = evaluation_metrics["primary_confusion_df"]
    if report_df.empty:
        return

    metrics_df = report_df.copy()
    for col in ["precision", "recall", "f1_score"]:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.6f}")
    if "support" in metrics_df.columns:
        metrics_df["support"] = metrics_df["support"].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
    if "class_code" in metrics_df.columns:
        metrics_df["class_code"] = metrics_df["class_code"].map(
            lambda v: "" if pd.isna(v) or v == "" else f"{int(float(v))}"
        )

    print(f"Classification report: {h5_stem}")
    print(metrics_df.to_string(index=False))

    if not confusion_df.empty:
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_df.to_string(index=False))


def print_value_counts_table(
    title: str,
    counts: Dict[int, int],
    name_map: Optional[Dict[int, str]] = None,
) -> None:
    if not counts:
        return
    rows = []
    for code, count in sorted(counts.items()):
        rows.append({
            "code": int(code),
            "name": (name_map or {}).get(int(code), ""),
            "count": int(count),
        })
    print(title)
    print(pd.DataFrame(rows).to_string(index=False))


def print_density_quantiles_table(
    pred_class_baseline: np.ndarray,
    point_density_pts_m3: np.ndarray,
) -> None:
    rows = []
    for cls in LABEL_ORDER:
        mask = np.asarray(pred_class_baseline, dtype=np.uint8) == cls
        density = np.asarray(point_density_pts_m3, dtype=np.float64)[mask]
        p05, median, p95 = maybe_quantiles(density)
        rows.append({
            "class_code": cls,
            "class_name": CLASS_NAME_MAP[cls],
            "density_p05": p05,
            "density_median": median,
            "density_p95": p95,
        })
    print("Density quantiles by predicted class")
    print(pd.DataFrame(rows).to_string(index=False))


def build_output_paths(output_root: Path, h5_stem: str) -> Dict[str, Path]:
    root = Path(output_root)
    return {
        "classified_laz": root / f"{h5_stem}_baseline_classified.laz",
        "classification_summary_csv": root / f"{h5_stem}_classification_summary.csv",
        "classification_report_csv": root / f"{h5_stem}_classification_report.csv",
        "confusion_matrix_csv": root / f"{h5_stem}_confusion_matrix.csv",
        "evaluation_summary_json": root / f"{h5_stem}_evaluation_summary.json",
        "run_metadata_json": root / f"{h5_stem}_run_metadata.json",
        "class_count_png": root / f"{h5_stem}_class_count_bar.png",
        "hag_hist_png": root / f"{h5_stem}_height_above_ground_hist_by_predicted_class.png",
        "confusion_heatmap_png": root / f"{h5_stem}_confusion_matrix_heatmap.png",
    }


def build_initial_run_metadata(
    h5_stem: str,
    h5_path: Path,
    reference_laz_path: Path,
    output_paths: Dict[str, Path],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "status": "running",
        "runtime_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "h5_stem": h5_stem,
        "inputs": {
            "h5_path": str(h5_path),
            "reference_laz_path": str(reference_laz_path),
        },
        "outputs": {key: str(value) for key, value in output_paths.items()},
        "classification_parameters": {
            key: json_safe(value)
            for key, value in config.items()
            if key.startswith(("GROUND_", "GRID_", "MIN_", "DTM_", "LAS_", "IDW_", "LOCAL_", "NOISE_"))
        },
        "evaluation_parameters": {
            "EVAL_REQUIRE_VALID_DTM": bool(config["EVAL_REQUIRE_VALID_DTM"]),
            "EVAL_IGNORE_REFERENCE_NOISE": bool(config["EVAL_IGNORE_REFERENCE_NOISE"]),
            "EVAL_REQUIRE_TRANSFER_STATUS": json_safe(config["EVAL_REQUIRE_TRANSFER_STATUS"]),
            "HIGH_CONF_NEAREST3DEP_DIST_M": float(config["HIGH_CONF_NEAREST3DEP_DIST_M"]),
            "HIGH_CONF_CLASS_VOTE_RATIO_MIN": float(config["HIGH_CONF_CLASS_VOTE_RATIO_MIN"]),
            "ROW_ALIGN_XY_TOL_M": float(config["ROW_ALIGN_XY_TOL_M"]),
            "ROW_ALIGN_LONLAT_TOL_DEG": float(config["ROW_ALIGN_LONLAT_TOL_DEG"]),
        },
        "warnings": [],
    }


def build_common_metadata_payload(
    projected_crs: CRS,
    crs_info: Dict[str, Any],
    casals: Dict[str, Any],
    ground_grid: Dict[str, Any],
    dtm_sample_valid: np.ndarray,
    pred_counts: Dict[int, int],
    classification_reason: np.ndarray,
    point_density_pts_m3: np.ndarray,
) -> Dict[str, Any]:
    density = np.asarray(point_density_pts_m3, dtype=np.float64)
    density_finite = np.isfinite(density)
    return {
        "crs": {
            "chosen_projected_crs": projected_crs.to_string(),
            "chosen_projected_crs_name": projected_crs.name,
            "chosen_projected_crs_source": crs_info["source"],
            "reference_laz_header_crs": crs_info["reference_crs"],
            "h5_detected_attr_crs": crs_info["h5_detected_attr_crs"],
            "h5_detected_attr_crs_source": crs_info["h5_detected_attr_crs_source"],
        },
        "missing_fields": casals["missing_fields"],
        "derived_fields": casals["derived_fields"],
        "source_datasets": casals["source_datasets"],
        "counts": {
            "point_count": int(casals["point_index"].size),
            "ground_support_candidate_count": int(ground_grid["support_count"]),
            "valid_dtm_cell_count": int(ground_grid["valid_cell_count"]),
            "dtm_invalid_count": int(np.count_nonzero(dtm_sample_valid == 0)),
            "predicted_class_counts": {str(k): int(v) for k, v in pred_counts.items()},
            "classification_reason_counts": {
                str(k): int(v) for k, v in collect_class_counts(classification_reason).items()
            },
            "density_noise_count": int(np.count_nonzero(np.asarray(classification_reason, dtype=np.uint8) == 6)),
            "density_valid_count": int(np.count_nonzero(density_finite)),
            "density_valid_fraction": float(np.mean(density_finite)),
        },
        "density_summary": summarize_values(density),
    }


def scalar_attr_to_text(value: object) -> Optional[str]:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if np.isscalar(value):
        return str(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return None


def iter_scalar_attr_strings(h5: h5py.File) -> Iterable[tuple[str, str]]:
    for key, value in h5.attrs.items():
        text = scalar_attr_to_text(value)
        if text is not None:
            yield f"/@{key}", text


def find_dataset_path(h5: h5py.File, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for name in candidates:
        if name in h5:
            return name
    if required:
        raise KeyError(f"Could not find any dataset from candidates: {list(candidates)}")
    return None


def read_optional_dataset(h5: h5py.File, candidates: Sequence[str], n: int) -> tuple[Optional[np.ndarray], Optional[str]]:
    path = find_dataset_path(h5, candidates, required=False)
    if path is None:
        return None, None
    arr = np.asarray(h5[path][...]).reshape(-1)
    if arr.size != n:
        raise ValueError(f"Optional dataset {path} has size {arr.size}, expected {n}")
    return arr, path


def infer_wgs84_utm_epsg(lon: np.ndarray, lat: np.ndarray) -> int:
    lon_med = float(np.nanmedian(lon))
    lat_med = float(np.nanmedian(lat))
    if not (-180.0 <= lon_med <= 180.0 and -90.0 <= lat_med <= 90.0):
        raise ValueError(f"Invalid median lon/lat for UTM inference: {lon_med}, {lat_med}")
    zone = int(math.floor((lon_med + 180.0) / 6.0) + 1)
    zone = max(1, min(zone, 60))
    return 32600 + zone if lat_med >= 0.0 else 32700 + zone


def detect_crs_from_h5_attrs(h5: h5py.File) -> tuple[Optional[CRS], Optional[str]]:
    for path, text in iter_scalar_attr_strings(h5):
        match = EPSG_RE.search(text)
        if match:
            epsg = int(match.group(1))
            try:
                return CRS.from_epsg(epsg), f"attribute_epsg:{path}={text}"
            except Exception:
                pass
        match = UTM_RE.search(text)
        if match:
            zone = int(match.group(1))
            hemisphere = (match.group(2) or "N").upper()
            epsg = (32600 if hemisphere == "N" else 32700) + zone
            try:
                return CRS.from_epsg(epsg), f"attribute_utm:{path}={text}"
            except Exception:
                pass
    return None, None


def horizontal_crs_only(crs: CRS) -> CRS:
    if crs.is_compound:
        for sub_crs in crs.sub_crs_list:
            if not sub_crs.is_vertical:
                return CRS.from_user_input(sub_crs)
        raise RuntimeError(f"Could not identify horizontal CRS from compound CRS: {crs}")
    if crs.is_vertical:
        raise RuntimeError(f"Reference CRS is vertical-only and unusable for XY projection: {crs}")
    return crs


def read_casals_h5_refh_points(h5_path: Path) -> Dict[str, Any]:
    fields: Dict[str, np.ndarray] = {}
    source_datasets: Dict[str, str] = {}
    missing_fields: list[str] = []
    derived_fields: list[str] = []

    with h5py.File(h5_path, "r") as h5:
        lon_path = find_dataset_path(h5, ["refh_longitude", "longitude", "lon"])
        lat_path = find_dataset_path(h5, ["refh_latitude", "latitude", "lat"])
        z_path = find_dataset_path(h5, ["refh", "refh_height", "height", "elevation"])

        lon = np.asarray(h5[lon_path][...], dtype=np.float64).reshape(-1)
        lat = np.asarray(h5[lat_path][...], dtype=np.float64).reshape(-1)
        z = np.asarray(h5[z_path][...], dtype=np.float64).reshape(-1)
        if lon.size != lat.size or lon.size != z.size:
            raise ValueError(f"H5 lon/lat/refh sizes differ: {lon.size}, {lat.size}, {z.size}")

        source_datasets.update({
            "refh_longitude": lon_path,
            "refh_latitude": lat_path,
            "refh": z_path,
        })

        optional_specs = {
            "refh_amp": ["refh_amp", "amp", "amplitude"],
            "refh_snr": ["refh_snr", "snr"],
            "refh_thres": ["refh_thres", "threshold"],
            "good_snr": ["good_snr"],
            "track_num": ["track_num", "track", "track_index"],
            "sweep_num": ["sweep_num", "sweep", "sweep_index"],
            "delta_time": ["delta_time", "time"],
            "refh_error": ["refh_error", "height_error"],
            "bg_mean": ["bg_mean", "background_mean"],
            "bg_std": ["bg_std", "background_std"],
        }
        for field_name, candidates in optional_specs.items():
            arr, path = read_optional_dataset(h5, candidates, lon.size)
            if arr is None:
                missing_fields.append(field_name)
                continue
            fields[field_name] = arr
            source_datasets[field_name] = path

        detected_attr_crs, detected_attr_crs_source = detect_crs_from_h5_attrs(h5)

    if "refh_snr" not in fields:
        if {"refh_amp", "bg_mean", "bg_std"}.issubset(fields):
            amp = np.asarray(fields["refh_amp"], dtype=np.float64)
            bg_mean = np.asarray(fields["bg_mean"], dtype=np.float64)
            bg_std = np.asarray(fields["bg_std"], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                fields["refh_snr"] = np.divide(
                    amp - bg_mean,
                    bg_std,
                    out=np.full_like(amp, np.nan, dtype=np.float64),
                    where=(bg_std != 0),
                )
            source_datasets["refh_snr"] = "derived_from_(refh_amp-bg_mean)/bg_std"
            derived_fields.append("refh_snr")
            if "refh_snr" in missing_fields:
                missing_fields.remove("refh_snr")
        elif {"refh_amp", "refh_thres"}.issubset(fields):
            amp = np.asarray(fields["refh_amp"], dtype=np.float64)
            thres = np.asarray(fields["refh_thres"], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                fields["refh_snr"] = np.divide(
                    amp,
                    thres,
                    out=np.full_like(amp, np.nan, dtype=np.float64),
                    where=(thres != 0),
                )
            source_datasets["refh_snr"] = "derived_from_refh_amp/refh_thres"
            derived_fields.append("refh_snr")
            if "refh_snr" in missing_fields:
                missing_fields.remove("refh_snr")
        else:
            raise KeyError("refh_snr is missing and could not be derived from available fields.")

    return {
        "lon": lon,
        "lat": lat,
        "z": z,
        "fields": fields,
        "point_index": np.arange(lon.size, dtype=np.uint32),
        "source_datasets": source_datasets,
        "missing_fields": sorted(missing_fields),
        "derived_fields": sorted(derived_fields),
        "detected_attr_crs": detected_attr_crs,
        "detected_attr_crs_source": detected_attr_crs_source,
    }


def infer_or_choose_projected_crs(
    casals_points: Dict[str, Any],
    reference_labels: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    reference_crs = None
    if reference_labels is not None:
        reference_crs = reference_labels.get("crs")
    if reference_crs is not None:
        chosen = horizontal_crs_only(reference_crs)
        return {
            "crs": chosen,
            "source": "reference_laz_header",
            "reference_crs": reference_crs.to_string(),
            "h5_detected_attr_crs": casals_points["detected_attr_crs"].to_string() if casals_points["detected_attr_crs"] else None,
            "h5_detected_attr_crs_source": casals_points["detected_attr_crs_source"],
        }

    if casals_points["detected_attr_crs"] is not None:
        chosen = horizontal_crs_only(casals_points["detected_attr_crs"])
        return {
            "crs": chosen,
            "source": casals_points["detected_attr_crs_source"] or "h5_attribute_crs",
            "reference_crs": None,
            "h5_detected_attr_crs": chosen.to_string(),
            "h5_detected_attr_crs_source": casals_points["detected_attr_crs_source"],
        }

    epsg = infer_wgs84_utm_epsg(casals_points["lon"], casals_points["lat"])
    chosen = CRS.from_epsg(epsg)
    return {
        "crs": chosen,
        "source": "inferred_wgs84_utm_from_median_lonlat",
        "reference_crs": None,
        "h5_detected_attr_crs": None,
        "h5_detected_attr_crs_source": None,
    }


def project_lonlat_to_xy(lon: np.ndarray, lat: np.ndarray, projected_crs: CRS) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(CRS.from_epsg(4326), projected_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def compute_local_point_density(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    local_neighbor_count = np.zeros(np.asarray(x).shape[0], dtype=np.uint32)
    point_density_pts_m3 = np.full(np.asarray(x).shape[0], np.nan, dtype=np.float64)

    valid_xyz = finite_mask(x, y, z)
    if not np.any(valid_xyz):
        return {
            "local_neighbor_count": local_neighbor_count,
            "point_density_pts_m3": point_density_pts_m3,
        }

    radius_m = float(config["LOCAL_FEATURE_RADIUS_M"])
    chunk_size = max(1, int(config["LOCAL_FEATURE_QUERY_CHUNK_SIZE"]))
    sphere_volume_m3 = float((4.0 / 3.0) * np.pi * radius_m**3)
    xyz_valid = np.column_stack((
        np.asarray(x, dtype=np.float64)[valid_xyz],
        np.asarray(y, dtype=np.float64)[valid_xyz],
        np.asarray(z, dtype=np.float64)[valid_xyz],
    ))
    valid_indices = np.flatnonzero(valid_xyz)
    tree = cKDTree(xyz_valid)

    for start in range(0, xyz_valid.shape[0], chunk_size):
        stop = min(start + chunk_size, xyz_valid.shape[0])
        query_xyz = xyz_valid[start:stop]
        try:
            counts = np.asarray(
                tree.query_ball_point(query_xyz, r=radius_m, return_length=True, workers=-1),
                dtype=np.uint32,
            )
        except TypeError:
            counts = np.asarray(
                [len(ids) for ids in tree.query_ball_point(query_xyz, r=radius_m, workers=-1)],
                dtype=np.uint32,
            )
        point_ids = valid_indices[start:stop]
        local_neighbor_count[point_ids] = counts
        point_density_pts_m3[point_ids] = counts.astype(np.float64) / sphere_volume_m3

    return {
        "local_neighbor_count": local_neighbor_count,
        "point_density_pts_m3": point_density_pts_m3,
    }


def build_ground_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    refh_snr: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(refh_snr)
    support_mask = finite & (np.asarray(refh_snr, dtype=np.float64) >= float(config["GROUND_SNR_MIN"]))
    if not np.any(support_mask):
        raise RuntimeError("No ground support candidates satisfy finite XYZ and refh_snr threshold.")

    x_support = x[support_mask]
    y_support = y[support_mask]
    z_support = z[support_mask]
    grid_res = float(config["GRID_RES_M"])

    x_min = math.floor(float(np.min(x_support)) / grid_res) * grid_res
    y_min = math.floor(float(np.min(y_support)) / grid_res) * grid_res
    cols = np.floor((x_support - x_min) / grid_res).astype(np.int64)
    rows = np.floor((y_support - y_min) / grid_res).astype(np.int64)
    n_cols = int(np.max(cols)) + 1
    linear = rows * n_cols + cols

    order = np.argsort(linear, kind="mergesort")
    linear_sorted = linear[order]
    z_sorted = z_support[order]
    unique_linear, start_idx, counts = np.unique(linear_sorted, return_index=True, return_counts=True)

    min_points_per_cell = int(config["MIN_POINTS_PER_CELL"])
    ground_z = np.full(unique_linear.shape[0], np.nan, dtype=np.float64)
    valid = counts >= min_points_per_cell
    percentile = float(config["GROUND_CELL_PERCENTILE"])
    for i in np.flatnonzero(valid):
        start = start_idx[i]
        stop = start + counts[i]
        ground_z[i] = float(np.percentile(z_sorted[start:stop], percentile))

    unique_linear = unique_linear[valid]
    counts = counts[valid]
    ground_z = ground_z[valid]
    if unique_linear.size == 0:
        raise RuntimeError("Ground grid contains zero valid cells after MIN_POINTS_PER_CELL filtering.")

    rows_valid = unique_linear // n_cols
    cols_valid = unique_linear % n_cols
    centers_x = x_min + (cols_valid.astype(np.float64) + 0.5) * grid_res
    centers_y = y_min + (rows_valid.astype(np.float64) + 0.5) * grid_res

    return {
        "support_mask": support_mask,
        "support_count": int(np.count_nonzero(support_mask)),
        "grid_res_m": grid_res,
        "x_min": x_min,
        "y_min": y_min,
        "n_cols": n_cols,
        "valid_linear_ids": unique_linear.astype(np.int64),
        "valid_ground_z": ground_z.astype(np.float64),
        "valid_cell_counts": counts.astype(np.int32),
        "valid_centers_xy": np.column_stack((centers_x, centers_y)).astype(np.float64),
        "valid_cell_count": int(unique_linear.size),
    }


def sample_ground_grid_idw(
    x: np.ndarray,
    y: np.ndarray,
    ground_grid: Dict[str, Any],
    config: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    local_ground_z = np.full(x.shape[0], np.nan, dtype=np.float64)
    dtm_valid = np.zeros(x.shape[0], dtype=np.uint8)

    finite_xy = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite_xy):
        return local_ground_z, dtm_valid

    grid_res = float(ground_grid["grid_res_m"])
    x_min = float(ground_grid["x_min"])
    y_min = float(ground_grid["y_min"])
    n_cols = int(ground_grid["n_cols"])
    valid_linear_ids = np.asarray(ground_grid["valid_linear_ids"], dtype=np.int64)
    valid_ground_z = np.asarray(ground_grid["valid_ground_z"], dtype=np.float64)

    cols = np.floor((x[finite_xy] - x_min) / grid_res).astype(np.int64)
    rows = np.floor((y[finite_xy] - y_min) / grid_res).astype(np.int64)
    linear = rows * n_cols + cols
    finite_idx = np.flatnonzero(finite_xy)

    pos = np.searchsorted(valid_linear_ids, linear)
    same_cell = (pos < valid_linear_ids.size) & (valid_linear_ids[np.clip(pos, 0, valid_linear_ids.size - 1)] == linear)
    if np.any(same_cell):
        idx = finite_idx[same_cell]
        local_ground_z[idx] = valid_ground_z[pos[same_cell]]
        dtm_valid[idx] = 1

    remaining_idx = finite_idx[~same_cell]
    if remaining_idx.size == 0:
        return local_ground_z, dtm_valid

    centers = np.asarray(ground_grid["valid_centers_xy"], dtype=np.float64)
    tree = cKDTree(centers)
    k = min(int(config["DTM_IDW_K"]), centers.shape[0])
    power = float(config["DTM_IDW_POWER"])
    max_radius = float(config["DTM_MAX_SEARCH_RADIUS_M"])
    chunk_size = int(config["IDW_QUERY_CHUNK_SIZE"])

    for start in range(0, remaining_idx.size, chunk_size):
        stop = min(start + chunk_size, remaining_idx.size)
        chunk_idx = remaining_idx[start:stop]
        query_xy = np.column_stack((x[chunk_idx], y[chunk_idx]))
        dists, neighbors = tree.query(query_xy, k=k, distance_upper_bound=max_radius, workers=-1)
        if k == 1:
            dists = dists[:, None]
            neighbors = neighbors[:, None]

        valid_neighbors = np.isfinite(dists) & (neighbors < centers.shape[0]) & (dists <= max_radius)
        if not np.any(valid_neighbors):
            continue

        exact_match = valid_neighbors & np.isclose(dists, 0.0)
        exact_rows = np.any(exact_match, axis=1)
        if np.any(exact_rows):
            row_ids = np.flatnonzero(exact_rows)
            exact_cols = np.argmax(exact_match[row_ids], axis=1)
            point_ids = chunk_idx[row_ids]
            local_ground_z[point_ids] = valid_ground_z[neighbors[row_ids, exact_cols]]
            dtm_valid[point_ids] = 1

        non_exact_rows = np.flatnonzero(~exact_rows)
        for row in non_exact_rows:
            row_mask = valid_neighbors[row]
            if not np.any(row_mask):
                continue
            row_d = dists[row, row_mask]
            row_n = neighbors[row, row_mask]
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = 1.0 / np.power(row_d, power)
            weight_sum = float(np.sum(weights))
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                continue
            point_id = chunk_idx[row]
            local_ground_z[point_id] = float(np.sum(weights * valid_ground_z[row_n]) / weight_sum)
            dtm_valid[point_id] = 1

    return local_ground_z, dtm_valid


def classify_points_baseline(
    z: np.ndarray,
    local_ground_z_m: np.ndarray,
    dtm_sample_valid: np.ndarray,
    point_density_pts_m3: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    hag = np.full(z.shape[0], np.nan, dtype=np.float64)
    valid = np.asarray(dtm_sample_valid, dtype=bool)
    hag[valid] = z[valid] - local_ground_z_m[valid]

    pred = np.full(z.shape[0], 7, dtype=np.uint8)
    reason = np.full(z.shape[0], 0, dtype=np.uint8)

    invalid_dtm = ~valid
    pred[invalid_dtm] = 7
    reason[invalid_dtm] = 2

    valid_mask = valid
    density = np.asarray(point_density_pts_m3, dtype=np.float64)
    low_density = valid_mask & np.isfinite(density) & (density < float(config["NOISE_DENSITY_MAX_PTS_M3"]))
    ground = valid_mask & np.isfinite(hag) & (np.abs(hag) <= float(config["GROUND_RESID_TOL_M"])) & ~low_density
    below_ground = valid_mask & np.isfinite(hag) & (hag < 0.0) & ~(low_density | ground)
    above_40m = valid_mask & np.isfinite(hag) & (hag > 40.0) & ~(low_density | ground | below_ground)
    processed = valid_mask & np.isfinite(hag) & ~(low_density | ground | below_ground | above_40m)
    nonfinite_hag = valid_mask & ~np.isfinite(hag)

    pred[low_density] = 7
    reason[low_density] = 6

    pred[ground] = 2
    reason[ground] = 1

    pred[below_ground] = 7
    reason[below_ground] = 3

    pred[above_40m] = 7
    reason[above_40m] = 4

    pred[processed] = 1
    reason[processed] = 5

    pred[nonfinite_hag] = 7
    reason[nonfinite_hag] = 2

    return {
        "pred_class_baseline": pred.astype(np.uint8),
        "classification_reason": reason.astype(np.uint8),
        "height_above_ground_m": hag.astype(np.float64),
    }


def read_reference_labels(reference_laz_path: Path) -> Dict[str, Any]:
    las = laspy.read(str(reference_laz_path))
    dims = set(las.point_format.extra_dimension_names)

    ref: Dict[str, Any] = {
        "point_count": int(las.header.point_count),
        "point_format_id": int(las.header.point_format.id),
        "crs": las.header.parse_crs(),
        "classification": np.asarray(las.classification, dtype=np.uint8),
        "x": np.asarray(las.x, dtype=np.float64),
        "y": np.asarray(las.y, dtype=np.float64),
        "z": np.asarray(las.z, dtype=np.float64),
        "available_extra_dims": sorted(dims),
    }

    for name in [
        "point_index",
        "longitude",
        "latitude",
        "transfer_status",
        "nearest3dep_dist_m",
        "class_vote_ratio",
    ]:
        if name in dims:
            ref[name] = np.asarray(las[name])

    return ref
def align_prediction_to_reference(
    prediction: Dict[str, np.ndarray],
    reference: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    n_pred = int(prediction["point_index"].shape[0])
    n_ref = int(reference["classification"].shape[0])
    result: Dict[str, Any] = {
        "eval_match_valid": np.zeros(n_pred, dtype=np.uint8),
        "eval_gt_class_raw": np.zeros(n_pred, dtype=np.uint8),
        "reference_transfer_status": None,
        "reference_nearest3dep_dist_m": None,
        "reference_class_vote_ratio": None,
        "alignment_method": None,
        "alignment_checks": {},
    }

    if "point_index" in reference:
        ref_idx = np.asarray(reference["point_index"], dtype=np.int64)
        if ref_idx.shape[0] != n_ref:
            raise ValueError("Reference point_index size does not match reference classification size.")
        if np.any(ref_idx < 0) or np.any(ref_idx >= n_pred):
            raise ValueError("Reference point_index contains values outside prediction range.")
        ref_sorted = np.sort(ref_idx, kind="mergesort")
        if np.any(ref_sorted[1:] == ref_sorted[:-1]):
            raise ValueError("Reference point_index contains duplicates.")

        match_valid = np.ones(n_pred, dtype=np.uint8)
        gt = np.zeros(n_pred, dtype=np.uint8)
        gt[ref_idx] = np.asarray(reference["classification"], dtype=np.uint8)
        result["eval_match_valid"] = match_valid
        result["eval_gt_class_raw"] = gt
        result["alignment_method"] = "point_index"

        for name in ["transfer_status", "nearest3dep_dist_m", "class_vote_ratio"]:
            if name in reference:
                aligned = np.full(n_pred, np.nan, dtype=np.float64) if name != "transfer_status" else np.full(n_pred, -1, dtype=np.int16)
                aligned[ref_idx] = np.asarray(reference[name])
                if name == "transfer_status":
                    result["reference_transfer_status"] = aligned.astype(np.int16)
                elif name == "nearest3dep_dist_m":
                    result["reference_nearest3dep_dist_m"] = aligned.astype(np.float64)
                elif name == "class_vote_ratio":
                    result["reference_class_vote_ratio"] = aligned.astype(np.float64)
        return result

    if n_ref != n_pred:
        raise RuntimeError(
            "Reference has no point_index and point counts differ; row-order alignment is not safe."
        )

    fractions = tuple(config["ROW_ALIGN_CHECK_INDICES"])
    sample_idx = sorted(
        {
            int(np.clip(round(frac * (n_pred - 1)), 0, n_pred - 1))
            for frac in fractions
        }
    )
    checks: Dict[str, Any] = {"sample_indices": sample_idx}

    if "longitude" in reference and "latitude" in reference:
        dlon = np.abs(np.asarray(reference["longitude"], dtype=np.float64)[sample_idx] - prediction["longitude"][sample_idx])
        dlat = np.abs(np.asarray(reference["latitude"], dtype=np.float64)[sample_idx] - prediction["latitude"][sample_idx])
        checks["max_abs_dlon_deg"] = float(np.max(dlon))
        checks["max_abs_dlat_deg"] = float(np.max(dlat))
        if float(np.max(dlon)) <= float(config["ROW_ALIGN_LONLAT_TOL_DEG"]) and float(np.max(dlat)) <= float(config["ROW_ALIGN_LONLAT_TOL_DEG"]):
            result["eval_match_valid"] = np.ones(n_pred, dtype=np.uint8)
            result["eval_gt_class_raw"] = np.asarray(reference["classification"], dtype=np.uint8)
            result["alignment_method"] = "row_order_lonlat"
            result["alignment_checks"] = checks
            if "transfer_status" in reference:
                result["reference_transfer_status"] = np.asarray(reference["transfer_status"], dtype=np.int16)
            if "nearest3dep_dist_m" in reference:
                result["reference_nearest3dep_dist_m"] = np.asarray(reference["nearest3dep_dist_m"], dtype=np.float64)
            if "class_vote_ratio" in reference:
                result["reference_class_vote_ratio"] = np.asarray(reference["class_vote_ratio"], dtype=np.float64)
            return result

    dx = np.abs(reference["x"][sample_idx] - prediction["x"][sample_idx])
    dy = np.abs(reference["y"][sample_idx] - prediction["y"][sample_idx])
    checks["max_abs_dx_m"] = float(np.max(dx))
    checks["max_abs_dy_m"] = float(np.max(dy))
    if float(np.max(dx)) <= float(config["ROW_ALIGN_XY_TOL_M"]) and float(np.max(dy)) <= float(config["ROW_ALIGN_XY_TOL_M"]):
        result["eval_match_valid"] = np.ones(n_pred, dtype=np.uint8)
        result["eval_gt_class_raw"] = np.asarray(reference["classification"], dtype=np.uint8)
        result["alignment_method"] = "row_order_xy"
        result["alignment_checks"] = checks
        if "transfer_status" in reference:
            result["reference_transfer_status"] = np.asarray(reference["transfer_status"], dtype=np.int16)
        if "nearest3dep_dist_m" in reference:
            result["reference_nearest3dep_dist_m"] = np.asarray(reference["nearest3dep_dist_m"], dtype=np.float64)
        if "class_vote_ratio" in reference:
            result["reference_class_vote_ratio"] = np.asarray(reference["class_vote_ratio"], dtype=np.float64)
        return result

    raise RuntimeError("Reference and prediction could not be aligned safely by row order.")


def map_reference_labels_to_baseline_classes(reference_class_raw: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_class_raw, dtype=np.uint8)
    out = np.ones(ref.shape[0], dtype=np.uint8)
    out[ref == 2] = 2
    out[(ref == 7) | (ref == 18)] = 7
    return out


def build_classification_summary_row(
    h5_stem: str,
    pred_class_baseline: np.ndarray,
    eval_gt_class: np.ndarray,
    eval_match_valid: np.ndarray,
    dtm_sample_valid: np.ndarray,
    ground_support_candidate_count: int,
    valid_dtm_cell_count: int,
    height_above_ground_m: np.ndarray,
    refh_snr: np.ndarray,
    point_density_pts_m3: np.ndarray,
    classification_reason: np.ndarray,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "h5_stem": h5_stem,
        "point_count": int(pred_class_baseline.size),
        "matched_count": int(np.count_nonzero(eval_match_valid)),
        "unmatched_count": int(pred_class_baseline.size - np.count_nonzero(eval_match_valid)),
        "dtm_invalid_count": int(np.count_nonzero(np.asarray(dtm_sample_valid) == 0)),
        "ground_support_candidate_count": int(ground_support_candidate_count),
        "valid_dtm_cell_count": int(valid_dtm_cell_count),
        "density_noise_count": int(np.count_nonzero(np.asarray(classification_reason, dtype=np.uint8) == 6)),
    }
    pred_counts = collect_class_counts(pred_class_baseline)
    ref_counts = collect_class_counts(eval_gt_class[np.asarray(eval_match_valid, dtype=bool)])
    for cls in LABEL_ORDER:
        row[f"pred_count_{cls}"] = int(pred_counts.get(cls, 0))
        row[f"ref_count_{cls}"] = int(ref_counts.get(cls, 0))

        pred_mask = np.asarray(pred_class_baseline, dtype=np.uint8) == cls
        p05, median, p95 = maybe_quantiles(np.asarray(height_above_ground_m, dtype=np.float64)[pred_mask])
        row[f"hag_pred_{cls}_p05"] = p05
        row[f"hag_pred_{cls}_median"] = median
        row[f"hag_pred_{cls}_p95"] = p95

        s05, smed, s95 = maybe_quantiles(np.asarray(refh_snr, dtype=np.float64)[pred_mask])
        row[f"refh_snr_pred_{cls}_p05"] = s05
        row[f"refh_snr_pred_{cls}_median"] = smed
        row[f"refh_snr_pred_{cls}_p95"] = s95

        d05, dmed, d95 = maybe_quantiles(np.asarray(point_density_pts_m3, dtype=np.float64)[pred_mask])
        row[f"density_pred_{cls}_p05"] = d05
        row[f"density_pred_{cls}_median"] = dmed
        row[f"density_pred_{cls}_p95"] = d95
    return row


def compute_evaluation_metrics(
    pred_class_baseline: np.ndarray,
    eval_gt_class: np.ndarray,
    eval_match_valid: np.ndarray,
    dtm_sample_valid: np.ndarray,
    reference_transfer_status: Optional[np.ndarray],
    reference_nearest3dep_dist_m: Optional[np.ndarray],
    reference_class_vote_ratio: Optional[np.ndarray],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    matched = np.asarray(eval_match_valid, dtype=bool)
    if config["EVAL_REQUIRE_TRANSFER_STATUS"] is not None and reference_transfer_status is None:
        raise RuntimeError("EVAL_REQUIRE_TRANSFER_STATUS is set but reference transfer_status is unavailable.")

    base_mask = matched.copy()
    if bool(config["EVAL_REQUIRE_VALID_DTM"]):
        base_mask &= np.asarray(dtm_sample_valid, dtype=bool)
    if bool(config["EVAL_IGNORE_REFERENCE_NOISE"]):
        base_mask &= np.asarray(eval_gt_class, dtype=np.uint8) != 7
    if config["EVAL_REQUIRE_TRANSFER_STATUS"] is not None:
        required = config["EVAL_REQUIRE_TRANSFER_STATUS"]
        required_values = [int(required)] if np.isscalar(required) else [int(v) for v in required]
        base_mask &= np.isin(np.asarray(reference_transfer_status), required_values)

    subset_masks: Dict[str, np.ndarray] = {"all_matched": base_mask}
    if reference_nearest3dep_dist_m is not None and reference_class_vote_ratio is not None:
        subset_masks["high_confidence_reference"] = (
            base_mask
            & np.isfinite(reference_nearest3dep_dist_m)
            & np.isfinite(reference_class_vote_ratio)
            & (reference_nearest3dep_dist_m <= float(config["HIGH_CONF_NEAREST3DEP_DIST_M"]))
            & (reference_class_vote_ratio >= float(config["HIGH_CONF_CLASS_VOTE_RATIO_MIN"]))
        )

    subset_results: Dict[str, Any] = {}
    evaluation_summary_rows: list[Dict[str, Any]] = []
    primary_report_df = pd.DataFrame()
    primary_confusion_df = pd.DataFrame()
    primary_metrics: Dict[str, Any] = {}

    for subset_name, subset_mask in subset_masks.items():
        y_true = np.asarray(eval_gt_class, dtype=np.uint8)[subset_mask]
        y_pred = np.asarray(pred_class_baseline, dtype=np.uint8)[subset_mask]

        subset_result: Dict[str, Any] = {
            "subset_name": subset_name,
            "n_points": int(y_true.size),
            "mask_count": int(np.count_nonzero(subset_mask)),
        }
        if y_true.size == 0:
            subset_result["status"] = "no_points_after_filtering"
            subset_results[subset_name] = subset_result
            continue

        cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            zero_division=0,
        )
        macro = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
        weighted = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0)
        accuracy = float(accuracy_score(y_true, y_pred))

        report_rows: list[Dict[str, Any]] = []
        for i, cls in enumerate(LABEL_ORDER):
            report_rows.append({
                "row_name": CLASS_NAME_MAP[cls],
                "class_code": cls,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            })
        report_rows.append({
            "row_name": "macro avg",
            "class_code": "",
            "precision": float(macro[0]),
            "recall": float(macro[1]),
            "f1_score": float(macro[2]),
            "support": int(np.sum(support)),
        })
        report_rows.append({
            "row_name": "weighted avg",
            "class_code": "",
            "precision": float(weighted[0]),
            "recall": float(weighted[1]),
            "f1_score": float(weighted[2]),
            "support": int(np.sum(support)),
        })
        report_rows.append({
            "row_name": "accuracy",
            "class_code": "",
            "precision": None,
            "recall": None,
            "f1_score": accuracy,
            "support": int(np.sum(support)),
        })

        confusion_rows = []
        for row_i, true_cls in enumerate(LABEL_ORDER):
            confusion_rows.append({
                "true_class": true_cls,
                "true_name": CLASS_NAME_MAP[true_cls],
                "pred_1": int(cm[row_i, 0]),
                "pred_2": int(cm[row_i, 1]),
                "pred_7": int(cm[row_i, 2]),
                "row_total": int(np.sum(cm[row_i])),
            })

        subset_result.update({
            "status": "ok",
            "accuracy": accuracy,
            "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]),
            "macro_f1": float(macro[2]),
            "weighted_precision": float(weighted[0]),
            "weighted_recall": float(weighted[1]),
            "weighted_f1": float(weighted[2]),
            "confusion_matrix": cm.astype(int).tolist(),
            "report_rows": report_rows,
            "confusion_rows": confusion_rows,
            "y_true": y_true,
            "y_pred": y_pred,
        })
        subset_results[subset_name] = subset_result

        evaluation_summary_rows.append({
            "subset_name": subset_name,
            "n_points": int(y_true.size),
            "accuracy": accuracy,
            "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]),
            "macro_f1": float(macro[2]),
            "weighted_precision": float(weighted[0]),
            "weighted_recall": float(weighted[1]),
            "weighted_f1": float(weighted[2]),
        })

        if subset_name == "all_matched":
            primary_report_df = pd.DataFrame(report_rows)
            primary_confusion_df = pd.DataFrame(confusion_rows)
            primary_metrics = {
                "accuracy": accuracy,
                "macro_f1": float(macro[2]),
                "weighted_f1": float(weighted[2]),
            }

    return {
        "subset_results": subset_results,
        "evaluation_summary_rows": evaluation_summary_rows,
        "primary_report_df": primary_report_df,
        "primary_confusion_df": primary_confusion_df,
        "primary_metrics": primary_metrics,
    }


def write_classified_laz(
    output_path: Path,
    projected_crs: CRS,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
    point_index: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    refh_original_m: np.ndarray,
    local_ground_z_m: np.ndarray,
    height_above_ground_m: np.ndarray,
    local_neighbor_count: np.ndarray,
    point_density_pts_m3: np.ndarray,
    dtm_sample_valid: np.ndarray,
    classification_reason: np.ndarray,
    pred_class_baseline: np.ndarray,
    fields: Dict[str, np.ndarray],
    eval_gt_class: Optional[np.ndarray],
    eval_match_valid: np.ndarray,
    config: Dict[str, Any],
) -> None:
    ensure_dir(output_path.parent)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([config["LAS_XYZ_SCALE_M"]] * 3, dtype=np.float64)
    header.offsets = np.array([
        math.floor(float(np.nanmin(x))),
        math.floor(float(np.nanmin(y))),
        math.floor(float(np.nanmin(z))),
    ], dtype=np.float64)
    header.system_identifier = "CASALS_L1B_REFH"
    header.generating_software = "classify_and_evaluate_casals_refh.py"
    try:
        header.add_crs(projected_crs)
    except Exception as exc:
        print(f"[WARN] Could not write CRS to LAZ header: {exc}")

    extra_dims = [
        ExtraBytesParams(name="point_index", type=np.uint32, description="CASALS point index"),
        ExtraBytesParams(name="longitude", type=np.float64, description="WGS84 longitude"),
        ExtraBytesParams(name="latitude", type=np.float64, description="WGS84 latitude"),
        ExtraBytesParams(name="refh_original_m", type=np.float64, description="Original refh Z"),
        ExtraBytesParams(name="local_ground_z_m", type=np.float64, description="Local ground Z"),
        ExtraBytesParams(name="height_above_ground_m", type=np.float64, description="refh minus ground"),
        ExtraBytesParams(name="local_neighbor_count", type=np.uint32, description="3D neighbor count"),
        ExtraBytesParams(name="point_density_pts_m3", type=np.float32, description="3D point density"),
        ExtraBytesParams(name="dtm_sample_valid", type=np.uint8, description="1 if DTM valid"),
        ExtraBytesParams(name="classification_reason", type=np.uint8, description="Baseline reason"),
        ExtraBytesParams(name="pred_class_baseline", type=np.uint8, description="Baseline class"),
        ExtraBytesParams(name="eval_match_valid", type=np.uint8, description="1 if eval matched"),
    ]
    optional_specs = {
        "refh_snr": np.float32,
        "refh_amp": np.float32,
        "refh_thres": np.float32,
        "good_snr": np.uint8,
        "track_num": np.uint16,
        "sweep_num": np.uint32,
        "delta_time": np.float64,
        "refh_error": np.float32,
        "bg_mean": np.float32,
        "bg_std": np.float32,
    }
    for name, dtype in optional_specs.items():
        if name in fields:
            extra_dims.append(ExtraBytesParams(name=name, type=dtype, description=name[:32]))
    if eval_gt_class is not None:
        extra_dims.append(ExtraBytesParams(name="eval_gt_class", type=np.uint8, description="Pseudo GT class"))

    las = laspy.LasData(header)
    las.add_extra_dims(extra_dims)
    las.x = np.asarray(x, dtype=np.float64)
    las.y = np.asarray(y, dtype=np.float64)
    las.z = np.asarray(z, dtype=np.float64)
    las.classification = np.asarray(classification, dtype=np.uint8)
    if "refh_amp" in fields:
        las.intensity = np.clip(np.nan_to_num(np.asarray(fields["refh_amp"], dtype=np.float64), nan=0.0), 0, 65535).astype(np.uint16)

    las["point_index"] = np.asarray(point_index, dtype=np.uint32)
    las["longitude"] = np.asarray(longitude, dtype=np.float64)
    las["latitude"] = np.asarray(latitude, dtype=np.float64)
    las["refh_original_m"] = np.asarray(refh_original_m, dtype=np.float64)
    las["local_ground_z_m"] = np.asarray(local_ground_z_m, dtype=np.float64)
    las["height_above_ground_m"] = np.asarray(height_above_ground_m, dtype=np.float64)
    las["local_neighbor_count"] = np.asarray(local_neighbor_count, dtype=np.uint32)
    las["point_density_pts_m3"] = np.asarray(point_density_pts_m3, dtype=np.float32)
    las["dtm_sample_valid"] = np.asarray(dtm_sample_valid, dtype=np.uint8)
    las["classification_reason"] = np.asarray(classification_reason, dtype=np.uint8)
    las["pred_class_baseline"] = np.asarray(pred_class_baseline, dtype=np.uint8)
    las["eval_match_valid"] = np.asarray(eval_match_valid, dtype=np.uint8)
    if eval_gt_class is not None:
        las["eval_gt_class"] = np.asarray(eval_gt_class, dtype=np.uint8)

    for name, dtype in optional_specs.items():
        if name not in fields:
            continue
        arr = np.asarray(fields[name])
        if dtype == np.uint8:
            las[name] = arr.astype(np.uint8)
        elif dtype == np.uint16:
            las[name] = arr.astype(np.uint16)
        elif dtype == np.uint32:
            las[name] = arr.astype(np.uint32)
        elif dtype == np.float32:
            las[name] = arr.astype(np.float32)
        else:
            las[name] = arr.astype(np.float64)

    las.write(str(output_path))


def write_summary_outputs(
    h5_stem: str,
    config: Dict[str, Any],
    output_paths: Dict[str, Path],
    classification_summary_row: Dict[str, Any],
    evaluation_metrics: Dict[str, Any],
    metadata: Dict[str, Any],
    pred_class_baseline: np.ndarray,
    height_above_ground_m: np.ndarray,
) -> None:
    pd.DataFrame([classification_summary_row]).to_csv(output_paths["classification_summary_csv"], index=False)
    evaluation_metrics["primary_report_df"].to_csv(output_paths["classification_report_csv"], index=False)
    evaluation_metrics["primary_confusion_df"].to_csv(output_paths["confusion_matrix_csv"], index=False)

    evaluation_json = {
        "h5_stem": h5_stem,
        "label_order": LABEL_ORDER,
        "subsets": {
            key: {
                sub_key: json_safe(sub_value)
                for sub_key, sub_value in value.items()
                if sub_key not in {"y_true", "y_pred"}
            }
            for key, value in evaluation_metrics["subset_results"].items()
        },
    }
    safe_json_dump(evaluation_json, output_paths["evaluation_summary_json"])
    safe_json_dump(metadata, output_paths["run_metadata_json"])

    if not bool(config["WRITE_DIAGNOSTIC_PNG"]):
        return

    pred_counts = collect_class_counts(pred_class_baseline)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar([CLASS_NAME_MAP[c] for c in LABEL_ORDER], [pred_counts.get(c, 0) for c in LABEL_ORDER], color=["#4c78a8", "#59a14f", "#e15759"])
    ax.set_title(f"{h5_stem} predicted class counts")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_paths["class_count_png"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    valid_hag = np.isfinite(height_above_ground_m)
    for cls, color in zip(LABEL_ORDER, ["#4c78a8", "#59a14f", "#e15759"]):
        mask = valid_hag & (pred_class_baseline == cls)
        if np.any(mask):
            ax.hist(height_above_ground_m[mask], bins=100, alpha=0.45, label=CLASS_NAME_MAP[cls], color=color)
    ax.set_title(f"{h5_stem} height_above_ground_m by predicted class")
    ax.set_xlabel("meters")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_paths["hag_hist_png"], dpi=160)
    plt.close(fig)

    if not evaluation_metrics["primary_confusion_df"].empty:
        cm_rows = evaluation_metrics["primary_confusion_df"]
        cm = cm_rows[["pred_1", "pred_2", "pred_7"]].to_numpy(dtype=np.int64)
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(LABEL_ORDER)))
        ax.set_xticklabels([CLASS_NAME_MAP[c] for c in LABEL_ORDER], rotation=25, ha="right")
        ax.set_yticks(range(len(LABEL_ORDER)))
        ax.set_yticklabels([CLASS_NAME_MAP[c] for c in LABEL_ORDER])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Pseudo-GT")
        ax.set_title(f"{h5_stem} confusion matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_paths["confusion_heatmap_png"], dpi=160)
        plt.close(fig)


def process_one_pair(input_pair: Dict[str, Path], config: Dict[str, Any]) -> Dict[str, Any]:
    h5_path = Path(input_pair["h5_path"])
    reference_laz_path = Path(input_pair["reference_laz_path"])
    h5_stem = h5_path.stem
    output_paths = build_output_paths(Path(config["OUTPUT_ROOT"]), h5_stem)
    ensure_dir(output_paths["classified_laz"].parent)
    ensure_dir(output_paths["run_metadata_json"].parent)

    metadata = build_initial_run_metadata(h5_stem, h5_path, reference_laz_path, output_paths, config)

    print(f"Processing: {h5_stem}")

    try:
        if not h5_path.exists():
            raise FileNotFoundError(h5_path)
        if not reference_laz_path.exists():
            raise FileNotFoundError(reference_laz_path)

        casals = read_casals_h5_refh_points(h5_path)
        print(f"Read CASALS points: {casals['point_index'].size:,}")

        reference = read_reference_labels(reference_laz_path)
        crs_info = infer_or_choose_projected_crs(casals, reference)
        projected_crs = crs_info["crs"]
        x, y = project_lonlat_to_xy(casals["lon"], casals["lat"], projected_crs)

        refh_snr = np.asarray(casals["fields"]["refh_snr"], dtype=np.float64)
        ground_grid = build_ground_grid(x, y, casals["z"], refh_snr, config)
        print(f"Ground support candidates: {ground_grid['support_count']:,}")
        print(f"Valid ground grid cells: {ground_grid['valid_cell_count']:,}")

        local_ground_z_m, dtm_sample_valid = sample_ground_grid_idw(x, y, ground_grid, config)
        density_result = compute_local_point_density(x, y, casals["z"], config)
        local_neighbor_count = density_result["local_neighbor_count"]
        point_density_pts_m3 = density_result["point_density_pts_m3"]
        print("Density source: computed_internal")
        classification_result = classify_points_baseline(
            casals["z"],
            local_ground_z_m,
            dtm_sample_valid,
            point_density_pts_m3,
            config,
        )
        pred_class_baseline = classification_result["pred_class_baseline"]
        classification_reason = classification_result["classification_reason"]
        height_above_ground_m = classification_result["height_above_ground_m"]
        pred_counts = collect_class_counts(pred_class_baseline)
        print(f"Predicted class counts: {pred_counts}")
        print_value_counts_table("Predicted class counts table", pred_counts, CLASS_NAME_MAP)
        print_value_counts_table("Classification reason counts", collect_class_counts(classification_reason), CLASS_REASON_MAP)
        print(f"Density valid fraction: {float(np.isfinite(point_density_pts_m3).mean()):.6f}")
        print_density_quantiles_table(pred_class_baseline, point_density_pts_m3)
        common_metadata = build_common_metadata_payload(
            projected_crs=projected_crs,
            crs_info=crs_info,
            casals=casals,
            ground_grid=ground_grid,
            dtm_sample_valid=dtm_sample_valid,
            pred_counts=pred_counts,
            classification_reason=classification_reason,
            point_density_pts_m3=point_density_pts_m3,
        )
        common_laz_kwargs = {
            "output_path": output_paths["classified_laz"],
            "projected_crs": projected_crs,
            "x": x,
            "y": y,
            "z": casals["z"],
            "classification": pred_class_baseline,
            "point_index": casals["point_index"],
            "longitude": casals["lon"],
            "latitude": casals["lat"],
            "refh_original_m": casals["z"],
            "local_ground_z_m": local_ground_z_m,
            "height_above_ground_m": height_above_ground_m,
            "local_neighbor_count": local_neighbor_count,
            "point_density_pts_m3": point_density_pts_m3,
            "dtm_sample_valid": dtm_sample_valid,
            "classification_reason": classification_reason,
            "pred_class_baseline": pred_class_baseline,
            "fields": casals["fields"],
            "config": config,
        }

        eval_match_valid = np.zeros(pred_class_baseline.shape[0], dtype=np.uint8)
        eval_gt_class = np.zeros(pred_class_baseline.shape[0], dtype=np.uint8)
        reference_transfer_status = None
        reference_nearest3dep_dist_m = None
        reference_class_vote_ratio = None
        alignment_method = None
        alignment_checks = {}
        evaluation_metrics = None
        classification_summary_row = None

        try:
            alignment = align_prediction_to_reference(
                prediction={
                    "point_index": casals["point_index"],
                    "x": x,
                    "y": y,
                    "longitude": np.asarray(casals["lon"], dtype=np.float64),
                    "latitude": np.asarray(casals["lat"], dtype=np.float64),
                },
                reference=reference,
                config=config,
            )
            alignment_method = alignment["alignment_method"]
            alignment_checks = alignment.get("alignment_checks", {})
            eval_match_valid = np.asarray(alignment["eval_match_valid"], dtype=np.uint8)
            eval_gt_class = map_reference_labels_to_baseline_classes(alignment["eval_gt_class_raw"])
            reference_transfer_status = alignment["reference_transfer_status"]
            reference_nearest3dep_dist_m = alignment["reference_nearest3dep_dist_m"]
            reference_class_vote_ratio = alignment["reference_class_vote_ratio"]
            print(f"Evaluation alignment: {alignment_method}")

            evaluation_metrics = compute_evaluation_metrics(
                pred_class_baseline=pred_class_baseline,
                eval_gt_class=eval_gt_class,
                eval_match_valid=eval_match_valid,
                dtm_sample_valid=dtm_sample_valid,
                reference_transfer_status=reference_transfer_status,
                reference_nearest3dep_dist_m=reference_nearest3dep_dist_m,
                reference_class_vote_ratio=reference_class_vote_ratio,
                config=config,
            )
            primary = evaluation_metrics["primary_metrics"]
            print(f"Overall accuracy: {primary.get('accuracy', float('nan')):.6f}")
            print(f"Macro F1: {primary.get('macro_f1', float('nan')):.6f}")
            print(f"Weighted F1: {primary.get('weighted_f1', float('nan')):.6f}")
            print_console_metrics_table(h5_stem, evaluation_metrics)

            classification_summary_row = build_classification_summary_row(
                h5_stem=h5_stem,
                pred_class_baseline=pred_class_baseline,
                eval_gt_class=eval_gt_class,
                eval_match_valid=eval_match_valid,
                dtm_sample_valid=dtm_sample_valid,
                ground_support_candidate_count=ground_grid["support_count"],
                valid_dtm_cell_count=ground_grid["valid_cell_count"],
                height_above_ground_m=height_above_ground_m,
                refh_snr=refh_snr,
                point_density_pts_m3=point_density_pts_m3,
                classification_reason=classification_reason,
            )
        except Exception as eval_exc:
            print(f"[ERROR] Evaluation failed for {h5_stem}: {eval_exc}")
            metadata["warnings"].append(f"Evaluation failed after classification: {eval_exc}")
            metadata["alignment_method"] = alignment_method
            metadata["alignment_checks"] = alignment_checks
            metadata["status"] = "failed"
            metadata.update(common_metadata)
            metadata["error"] = {
                "type": type(eval_exc).__name__,
                "message": str(eval_exc),
                "traceback": traceback.format_exc(),
            }
            write_classified_laz(**common_laz_kwargs, eval_gt_class=None, eval_match_valid=np.zeros(pred_class_baseline.shape[0], dtype=np.uint8))
            safe_json_dump(metadata, output_paths["run_metadata_json"])
            print(f"Wrote: {output_paths['classified_laz']}")
            return {
                "status": "failed",
                "h5_stem": h5_stem,
                "metadata_path": output_paths["run_metadata_json"],
                "error": metadata["error"],
            }

        write_classified_laz(**common_laz_kwargs, eval_gt_class=eval_gt_class, eval_match_valid=eval_match_valid)

        metadata["status"] = "success"
        metadata["alignment_method"] = alignment_method
        metadata["alignment_checks"] = alignment_checks
        metadata.update(common_metadata)
        metadata["counts"].update({
            "reference_class_counts": {
                str(k): int(v) for k, v in collect_class_counts(eval_gt_class[np.asarray(eval_match_valid, dtype=bool)]).items()
            },
            "matched_count": int(np.count_nonzero(eval_match_valid)),
            "unmatched_count": int(pred_class_baseline.size - np.count_nonzero(eval_match_valid)),
        })
        metadata["classification_reason_codes"] = CLASS_REASON_MAP
        metadata["class_semantics"] = {
            "1": "processed_unclassified",
            "2": "ground",
            "7": "noise",
        }
        metadata["scientific_notes"] = {
            "baseline_classifier": "rule-based baseline on CASALS refh points",
            "pseudo_ground_truth": "transferred 3DEP labels are used for evaluation and are not absolute ground truth",
            "vertical_datum": "no vertical datum correction is applied; output Z remains CASALS refh height",
        }

        write_summary_outputs(
            h5_stem=h5_stem,
            config=config,
            output_paths=output_paths,
            classification_summary_row=classification_summary_row,
            evaluation_metrics=evaluation_metrics,
            metadata=metadata,
            pred_class_baseline=pred_class_baseline,
            height_above_ground_m=height_above_ground_m,
        )

        print(f"Wrote: {output_paths['classified_laz']}")
        return {
            "status": "success",
            "h5_stem": h5_stem,
            "classification_summary_row": classification_summary_row,
            "evaluation_summary_rows": evaluation_metrics["evaluation_summary_rows"],
            "confusion_rows_by_subset": {
                subset_name: subset_result["confusion_rows"]
                for subset_name, subset_result in evaluation_metrics["subset_results"].items()
                if subset_result.get("status") == "ok"
            },
            "classification_summary_arrays": {
                "pred_class_baseline": pred_class_baseline,
                "eval_gt_class": eval_gt_class,
                "eval_match_valid": eval_match_valid,
                "dtm_sample_valid": dtm_sample_valid,
                "height_above_ground_m": height_above_ground_m,
                "refh_snr": refh_snr,
                "point_density_pts_m3": point_density_pts_m3,
                "classification_reason": classification_reason,
                "ground_support_candidate_count": ground_grid["support_count"],
                "valid_dtm_cell_count": ground_grid["valid_cell_count"],
            },
            "evaluation_arrays_by_subset": {
                subset_name: {
                    "y_true": subset_result["y_true"],
                    "y_pred": subset_result["y_pred"],
                }
                for subset_name, subset_result in evaluation_metrics["subset_results"].items()
                if subset_result.get("status") == "ok"
            },
            "metadata_path": output_paths["run_metadata_json"],
        }
    except Exception as exc:
        print(f"[ERROR] Processing failed for {h5_stem}: {exc}")
        metadata["status"] = "failed"
        metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        safe_json_dump(metadata, output_paths["run_metadata_json"])
        return {
            "status": "failed",
            "h5_stem": h5_stem,
            "metadata_path": output_paths["run_metadata_json"],
            "error": metadata["error"],
        }


def write_all_files_outputs(
    successful_results: list[Dict[str, Any]],
    failed_results: list[Dict[str, Any]],
    config: Dict[str, Any],
    input_pairs: list[Dict[str, Path]],
    all_metadata: Dict[str, Any],
) -> None:
    output_root = Path(config["OUTPUT_ROOT"])
    classification_summary_path = output_root / "all_files_classification_summary.csv"
    evaluation_summary_path = output_root / "all_files_evaluation_summary.csv"
    confusion_summary_path = output_root / "all_files_confusion_matrix.csv"
    metadata_path = output_root / "all_files_run_metadata.json"

    classification_rows = [result["classification_summary_row"] for result in successful_results]
    if successful_results:
        agg_pred = np.concatenate([r["classification_summary_arrays"]["pred_class_baseline"] for r in successful_results])
        agg_gt = np.concatenate([r["classification_summary_arrays"]["eval_gt_class"] for r in successful_results])
        agg_match = np.concatenate([r["classification_summary_arrays"]["eval_match_valid"] for r in successful_results])
        agg_dtm = np.concatenate([r["classification_summary_arrays"]["dtm_sample_valid"] for r in successful_results])
        agg_hag = np.concatenate([r["classification_summary_arrays"]["height_above_ground_m"] for r in successful_results])
        agg_snr = np.concatenate([r["classification_summary_arrays"]["refh_snr"] for r in successful_results])
        agg_density = np.concatenate([r["classification_summary_arrays"]["point_density_pts_m3"] for r in successful_results])
        agg_reason = np.concatenate([r["classification_summary_arrays"]["classification_reason"] for r in successful_results])
        agg_support_count = int(sum(r["classification_summary_arrays"]["ground_support_candidate_count"] for r in successful_results))
        agg_valid_cells = int(sum(r["classification_summary_arrays"]["valid_dtm_cell_count"] for r in successful_results))
        classification_rows.append(
            build_classification_summary_row(
                h5_stem="__all__",
                pred_class_baseline=agg_pred,
                eval_gt_class=agg_gt,
                eval_match_valid=agg_match,
                dtm_sample_valid=agg_dtm,
                ground_support_candidate_count=agg_support_count,
                valid_dtm_cell_count=agg_valid_cells,
                height_above_ground_m=agg_hag,
                refh_snr=agg_snr,
                point_density_pts_m3=agg_density,
                classification_reason=agg_reason,
            )
        )

    classification_columns = [
        "h5_stem",
        "point_count",
        "matched_count",
        "unmatched_count",
        "dtm_invalid_count",
        "density_noise_count",
        "ground_support_candidate_count",
        "valid_dtm_cell_count",
    ]
    for cls in LABEL_ORDER:
        classification_columns.extend([
            f"pred_count_{cls}",
            f"ref_count_{cls}",
            f"hag_pred_{cls}_p05",
            f"hag_pred_{cls}_median",
            f"hag_pred_{cls}_p95",
            f"refh_snr_pred_{cls}_p05",
            f"refh_snr_pred_{cls}_median",
            f"refh_snr_pred_{cls}_p95",
            f"density_pred_{cls}_p05",
            f"density_pred_{cls}_median",
            f"density_pred_{cls}_p95",
        ])
    pd.DataFrame(classification_rows, columns=classification_columns).to_csv(classification_summary_path, index=False)

    evaluation_summary_rows = []
    confusion_rows = []
    aggregate_by_subset: Dict[str, Dict[str, list[np.ndarray]]] = {}
    for result in successful_results:
        for row in result["evaluation_summary_rows"]:
            evaluation_summary_rows.append({"h5_stem": result["h5_stem"], **row})
        for subset_name, rows in result["confusion_rows_by_subset"].items():
            for row in rows:
                confusion_rows.append({"h5_stem": result["h5_stem"], "subset_name": subset_name, **row})
        for subset_name, arrays in result["evaluation_arrays_by_subset"].items():
            aggregate_by_subset.setdefault(subset_name, {"y_true": [], "y_pred": []})
            aggregate_by_subset[subset_name]["y_true"].append(arrays["y_true"])
            aggregate_by_subset[subset_name]["y_pred"].append(arrays["y_pred"])

    for subset_name, arrays in aggregate_by_subset.items():
        y_true = np.concatenate(arrays["y_true"]) if arrays["y_true"] else np.array([], dtype=np.uint8)
        y_pred = np.concatenate(arrays["y_pred"]) if arrays["y_pred"] else np.array([], dtype=np.uint8)
        if y_true.size == 0:
            continue
        cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
        macro = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
        weighted = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0)
        evaluation_summary_rows.append({
            "h5_stem": "__all__",
            "subset_name": subset_name,
            "n_points": int(y_true.size),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]),
            "macro_f1": float(macro[2]),
            "weighted_precision": float(weighted[0]),
            "weighted_recall": float(weighted[1]),
            "weighted_f1": float(weighted[2]),
        })
        for row_i, true_cls in enumerate(LABEL_ORDER):
            confusion_rows.append({
                "h5_stem": "__all__",
                "subset_name": subset_name,
                "true_class": true_cls,
                "true_name": CLASS_NAME_MAP[true_cls],
                "pred_1": int(cm[row_i, 0]),
                "pred_2": int(cm[row_i, 1]),
                "pred_7": int(cm[row_i, 2]),
                "row_total": int(np.sum(cm[row_i])),
            })

    pd.DataFrame(
        evaluation_summary_rows,
        columns=[
            "h5_stem",
            "subset_name",
            "n_points",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ],
    ).to_csv(evaluation_summary_path, index=False)
    pd.DataFrame(
        confusion_rows,
        columns=[
            "h5_stem",
            "subset_name",
            "true_class",
            "true_name",
            "pred_1",
            "pred_2",
            "pred_7",
            "row_total",
        ],
    ).to_csv(confusion_summary_path, index=False)

    all_metadata["totals"] = {
        "n_input_pairs": len(input_pairs),
        "n_successful_pairs": len(successful_results),
        "n_failed_pairs": len(failed_results),
    }
    safe_json_dump(all_metadata, metadata_path)


def main() -> None:
    INPUT_PAIRS = [
        {
            "h5_path": Path("./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
            "reference_laz_path": Path("./outputs/transfer_3dep_labels_to_casals/casals_l1b_20241112T165718_001_02.laz"),
        },
        {
            "h5_path": Path("./casals_h5_downloads/casals_l1b_20241118T171757_001_02.h5"),
            "reference_laz_path": Path("./outputs/transfer_3dep_labels_to_casals/casals_l1b_20241118T171757_001_02.laz"),
        },
    ]

    CONFIG: Dict[str, Any] = {
        "OUTPUT_ROOT": Path("./outputs/classify_and_evaluate_casals_refh"),
        "GROUND_SNR_MIN": 5.0,
        "GRID_RES_M": 10.0,
        "MIN_POINTS_PER_CELL": 1,
        "GROUND_CELL_PERCENTILE": 2,
        "DTM_IDW_K": 12,
        "DTM_IDW_POWER": 2.0,
        "DTM_MAX_SEARCH_RADIUS_M": 30.0,
        "GROUND_RESID_TOL_M": 1.0,
        "LOCAL_FEATURE_RADIUS_M": 5.0,
        "LOCAL_FEATURE_MAX_NEIGHBORS": 24,
        "LOCAL_FEATURE_MIN_NEIGHBORS": 6,
        "LOCAL_FEATURE_QUERY_CHUNK_SIZE": 100_000,
        "NOISE_DENSITY_MAX_PTS_M3": 0.04,
        "EVAL_REQUIRE_VALID_DTM": False,
        "EVAL_IGNORE_REFERENCE_NOISE": False,
        "EVAL_REQUIRE_TRANSFER_STATUS": None,
        "HIGH_CONF_NEAREST3DEP_DIST_M": 1.0,
        "HIGH_CONF_CLASS_VOTE_RATIO_MIN": 0.5,
        "ROW_ALIGN_XY_TOL_M": 0.01,
        "ROW_ALIGN_LONLAT_TOL_DEG": 1e-7,
        "ROW_ALIGN_CHECK_INDICES": (0.0, 0.25, 0.5, 0.75, 1.0),
        "WRITE_DIAGNOSTIC_PNG": True,
        "LAS_XYZ_SCALE_M": 0.001,
        "IDW_QUERY_CHUNK_SIZE": 500_000,
    }

    ensure_dir(Path(CONFIG["OUTPUT_ROOT"]))

    successful_results = []
    failed_results = []
    all_metadata = {
        "runtime_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": json_safe(CONFIG),
        "input_pairs": json_safe(INPUT_PAIRS),
        "successful_files": [],
        "failed_files": [],
        "warnings": [],
        "outputs": {
            "all_files_classification_summary_csv": str(Path(CONFIG["OUTPUT_ROOT"]) / "all_files_classification_summary.csv"),
            "all_files_evaluation_summary_csv": str(Path(CONFIG["OUTPUT_ROOT"]) / "all_files_evaluation_summary.csv"),
            "all_files_confusion_matrix_csv": str(Path(CONFIG["OUTPUT_ROOT"]) / "all_files_confusion_matrix.csv"),
            "all_files_run_metadata_json": str(Path(CONFIG["OUTPUT_ROOT"]) / "all_files_run_metadata.json"),
        },
    }

    for input_pair in INPUT_PAIRS:
        print(f"\n=== Processing pair: {input_pair['h5_path'].stem} ===")
        result = process_one_pair(input_pair, CONFIG)
        if result["status"] == "success":
            successful_results.append(result)
            all_metadata["successful_files"].append({
                "h5_stem": result["h5_stem"],
                "metadata_path": str(result["metadata_path"]),
            })
        else:
            failed_results.append(result)
            all_metadata["failed_files"].append({
                "h5_stem": result["h5_stem"],
                "metadata_path": str(result["metadata_path"]),
                "error": result.get("error"),
            })
    write_all_files_outputs(
        successful_results=successful_results,
        failed_results=failed_results,
        config=CONFIG,
        input_pairs=INPUT_PAIRS,
        all_metadata=all_metadata,
    )


if __name__ == "__main__":
    main()
