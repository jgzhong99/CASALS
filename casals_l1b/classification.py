"""Deterministic CASALS refh classification and feature calculations."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import h5py
import numpy as np
from pyproj import CRS
from scipy.spatial import cKDTree

from .geo import horizontal_crs_only, infer_wgs84_utm_epsg
from .h5 import find_dataset_path, iter_scalar_attr_strings, read_optional_dataset


LABEL_ORDER = [1, 2, 7]

CLASS_NAME_MAP = {
    1: "processed_unclassified",
    2: "ground",
    7: "noise",
}

CLASS_REASON_MAP = {
    18: "noise_nonfinite_height",
    0: "unknown",
    1: "ground_abs_hag_within_tol",
    2: "unclassified_missing_dtm",
    3: "noise_below_ground",
    4: "noise_above_max_hag",
    5: "processed_unclassified_valid_hag",
    6: "noise_low_density_pre_ground",
    7: "processed_negative_hag_allowed",
    8: "noise_low_density_post_ground",
    9: "noise_low_density_processed_only",
    10: "ground_rejected_low_signal_to_processed",
    11: "ground_rejected_weak_dtm_to_processed",
    12: "ground_rejected_low_density_to_processed",
    13: "noise_low_signal_low_density",
    14: "noise_low_amp_low_density",
    15: "noise_scanline_hag_outlier",
    16: "shallow_negative_hag_kept_processed",
    17: "weak_dtm_shallow_negative_kept_processed",
}

SUPPORTED_CLASSIFIER_MODES = {
    "height_only",
    "height_density_pre_ground",
    "height_density_post_ground",
    "height_density_processed_only",
    "height_no_below_noise",
    "rule_combined_v1",
}

RULE_SWITCH_KEYS = [
    "USE_NEAR_GROUND_GUARD",
    "USE_SIGNAL_DENSITY_NOISE",
    "USE_BELOW_GROUND_REFINEMENT",
    "USE_DTM_SUPPORT_CONFIDENCE",
    "USE_SCANLINE_OUTLIER_NOISE",
]

RULE_THRESHOLD_KEYS = [
    "GROUND_SNR_MIN",
    "GRID_RES_M",
    "GROUND_CELL_PERCENTILE",
    "GROUND_RESID_TOL_M",
    "NOISE_HAG_MAX_M",
    "LOCAL_FEATURE_RADIUS_M",
    "NOISE_DENSITY_MAX_PTS_M3",
    "NEAR_GROUND_GUARD_LOW_SNR_MAX",
    "NEAR_GROUND_GUARD_LOW_DENSITY_MAX",
    "NEAR_GROUND_GUARD_HAG_ABS_MAX_M",
    "NOISE_LOW_DENSITY_MAX_PTS_M3",
    "NOISE_LOW_SNR_MAX",
    "NOISE_LOW_AMP_ROBUST_Z_MAX",
    "NOISE_MIN_ABS_HAG_FOR_SIGNAL_RULE_M",
    "BELOW_GROUND_NOISE_MIN_DEPTH_M",
    "BELOW_GROUND_SHALLOW_TO_PROCESSED_M",
    "DTM_SUPPORT_DIST_MAX_FOR_GROUND_M",
    "SCANLINE_USE_TRACK_HAG_Z",
    "SCANLINE_USE_SWEEP_HAG_Z",
    "SCANLINE_HAG_ZSCORE_MIN",
    "SCANLINE_HAG_ZSCORE_MAX",
    "SCANLINE_HAG_MIN_M",
    "SCANLINE_HAG_MAX_M",
    "SCANLINE_DENSITY_MIN_PTS_M3",
    "SCANLINE_MIN_GROUP_SIZE",
    "ROBUST_Z_NMAD_EPS",
]

DEFAULT_CONFIG: Dict[str, Any] = {
    "OUTPUT_ROOT": Path("outputs/classification"),
    "CLASSIFIER_MODE": "rule_combined_v1",
    "GROUND_SNR_MIN": 5.0,
    "GRID_RES_M": 15.0,
    "MIN_POINTS_PER_CELL": 1,
    "GROUND_CELL_PERCENTILE": 2,
    "DTM_IDW_K": 12,
    "DTM_IDW_POWER": 2.0,
    "DTM_MAX_SEARCH_RADIUS_M": 30.0,
    "GROUND_RESID_TOL_M": 1.5,
    "NOISE_HAG_MAX_M": 30.0,
    "LOCAL_FEATURE_RADIUS_M": 5.0,
    "LOCAL_FEATURE_MAX_NEIGHBORS": 24,
    "LOCAL_FEATURE_MIN_NEIGHBORS": 6,
    "LOCAL_FEATURE_QUERY_CHUNK_SIZE": 100_000,
    "NOISE_DENSITY_MAX_PTS_M3": 0.02,
    "USE_NEAR_GROUND_GUARD": False,
    "USE_SIGNAL_DENSITY_NOISE": False,
    "USE_BELOW_GROUND_REFINEMENT": False,
    "USE_DTM_SUPPORT_CONFIDENCE": False,
    "USE_SCANLINE_OUTLIER_NOISE": True,
    "NEAR_GROUND_GUARD_LOW_SNR_MAX": 2.5,
    "NEAR_GROUND_GUARD_LOW_DENSITY_MAX": 0.01,
    "NEAR_GROUND_GUARD_HAG_ABS_MAX_M": 1.5,
    "NEAR_GROUND_GUARD_ACTION": "ground_to_processed",
    "NOISE_LOW_DENSITY_MAX_PTS_M3": 0.02,
    "NOISE_LOW_SNR_MAX": 2.5,
    "NOISE_LOW_AMP_ROBUST_Z_MAX": -1.0,
    "NOISE_MIN_ABS_HAG_FOR_SIGNAL_RULE_M": 1.0,
    "NOISE_SIGNAL_DENSITY_ACTION": "processed_to_noise",
    "BELOW_GROUND_NOISE_MIN_DEPTH_M": 0.0,
    "BELOW_GROUND_SHALLOW_TO_PROCESSED_M": None,
    "BELOW_GROUND_REQUIRE_LOW_DENSITY": False,
    "BELOW_GROUND_REQUIRE_LOW_SNR": False,
    "COMPUTE_DTM_SUPPORT_FEATURES": False,
    "DTM_SUPPORT_DIST_MAX_FOR_GROUND_M": 30.0,
    "DTM_SUPPORT_WEAK_GROUND_ACTION": "ground_to_processed",
    "DTM_SUPPORT_WEAK_SHALLOW_NEGATIVE_ACTION": "processed",
    "COMPUTE_SCANLINE_FEATURES": False,
    "SCANLINE_USE_TRACK_HAG_Z": True,
    "SCANLINE_USE_SWEEP_HAG_Z": False,
    "SCANLINE_HAG_ZSCORE_MIN": 35.0,
    "SCANLINE_HAG_ZSCORE_MAX": 55.0,
    "SCANLINE_HAG_MIN_M": 8.0,
    "SCANLINE_HAG_MAX_M": 15.0,
    "SCANLINE_DENSITY_MIN_PTS_M3": 2.0,
    "SCANLINE_REQUIRE_LOW_DENSITY": False,
    "SCANLINE_REQUIRE_NON_GROUND": True,
    "SCANLINE_MIN_GROUP_SIZE": 30,
    "ROBUST_Z_NMAD_EPS": 1e-6,
    "WRITE_ERROR_FEATURE_SUMMARY": True,
    "WRITE_ERROR_SUBSET_SAMPLES": False,
    "MAX_ERROR_SUBSET_SAMPLE_POINTS": 100_000,
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

EPSG_RE = re.compile(r"EPSG[:\s]*([0-9]{4,6})", re.IGNORECASE)

UTM_RE = re.compile(r"UTM(?:\s+ZONE)?[:\s]*([0-9]{1,2})([NS])?", re.IGNORECASE)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

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

def robust_median_nmad(values: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    median = float(np.median(arr))
    nmad = float(1.4826 * np.median(np.abs(arr - median)))
    return median, max(nmad, float(eps))

def robust_zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out
    median, nmad = robust_median_nmad(arr[finite], eps=eps)
    if not np.isfinite(median) or not np.isfinite(nmad):
        return out
    out[finite] = (arr[finite] - median) / nmad
    return out

def safe_field_array(
    fields: Dict[str, np.ndarray],
    name: str,
    n: int,
    default: Any = np.nan,
    dtype: Any = np.float64,
) -> np.ndarray:
    if name not in fields:
        return np.full(n, default, dtype=dtype)
    arr = np.asarray(fields[name], dtype=dtype).reshape(-1)
    if arr.size != n:
        raise ValueError(f"Field {name} has size {arr.size}, expected {n}")
    return arr

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

def quantile_summary(values: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    summary = {
        "count": int(arr.size),
        "n_finite": int(np.count_nonzero(finite)),
        "min": None,
        "p05": None,
        "p25": None,
        "median": None,
        "p75": None,
        "p95": None,
        "max": None,
    }
    if not np.any(finite):
        return summary
    arr = arr[finite]
    summary.update({
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    })
    return summary

def normalize_config(config: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config)
    warnings: list[str] = []
    if "CLASSIFIER_MODE" not in config:
        warnings.append(
            f"CLASSIFIER_MODE missing in config; defaulting to {DEFAULT_CONFIG['CLASSIFIER_MODE']}."
        )
    mode = str(merged.get("CLASSIFIER_MODE", DEFAULT_CONFIG["CLASSIFIER_MODE"]))
    if mode not in SUPPORTED_CLASSIFIER_MODES:
        raise ValueError(f"Unsupported CLASSIFIER_MODE: {mode}")
    merged["CLASSIFIER_MODE"] = mode
    return merged, warnings

def needs_density_features(config: Dict[str, Any]) -> bool:
    mode = str(config["CLASSIFIER_MODE"])
    if mode in {"height_density_pre_ground", "height_density_post_ground", "height_density_processed_only", "rule_combined_v1"}:
        return config.get("NOISE_DENSITY_MAX_PTS_M3") is not None
    if bool(config["USE_SIGNAL_DENSITY_NOISE"]):
        return True
    if bool(config["USE_NEAR_GROUND_GUARD"]):
        return True
    if bool(config["USE_SCANLINE_OUTLIER_NOISE"]):
        if bool(config["SCANLINE_REQUIRE_LOW_DENSITY"]):
            return True
        if config.get("SCANLINE_DENSITY_MIN_PTS_M3") is not None:
            return True
    if bool(config["USE_BELOW_GROUND_REFINEMENT"]) and bool(config["BELOW_GROUND_REQUIRE_LOW_DENSITY"]):
        return True
    return False

def needs_dtm_support_features(config: Dict[str, Any]) -> bool:
    return bool(config["COMPUTE_DTM_SUPPORT_FEATURES"]) or bool(config["USE_DTM_SUPPORT_CONFIDENCE"])

def needs_scanline_features(config: Dict[str, Any]) -> bool:
    return bool(config["COMPUTE_SCANLINE_FEATURES"]) or bool(config["USE_SCANLINE_OUTLIER_NOISE"])

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
            "h5_detected_attr_crs": (
                casals_points["detected_attr_crs"].to_string() if casals_points["detected_attr_crs"] else None
            ),
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

def compute_local_point_density(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    local_neighbor_count = np.zeros(np.asarray(x).shape[0], dtype=np.uint32)
    point_density_pts_m3 = np.full(np.asarray(x).shape[0], np.nan, dtype=np.float64)

    radius_m = config.get("LOCAL_FEATURE_RADIUS_M")
    if radius_m is None or float(radius_m) <= 0.0:
        return {
            "local_neighbor_count": local_neighbor_count,
            "point_density_pts_m3": point_density_pts_m3,
        }

    valid_xyz = finite_mask(x, y, z)
    if not np.any(valid_xyz):
        return {
            "local_neighbor_count": local_neighbor_count,
            "point_density_pts_m3": point_density_pts_m3,
        }

    radius_m = float(radius_m)
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

def compute_signal_features(fields: Dict[str, np.ndarray], n: int, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    eps = float(config["ROBUST_Z_NMAD_EPS"])
    refh_snr = safe_field_array(fields, "refh_snr", n)
    refh_amp = safe_field_array(fields, "refh_amp", n)
    bg_mean = safe_field_array(fields, "bg_mean", n)
    bg_std = safe_field_array(fields, "bg_std", n)
    track_num = safe_field_array(fields, "track_num", n)
    sweep_num = safe_field_array(fields, "sweep_num", n)
    amp_snr_like = np.full(n, np.nan, dtype=np.float64)
    valid_amp_snr_like = np.isfinite(refh_amp) & np.isfinite(bg_mean) & np.isfinite(bg_std) & (bg_std > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp_snr_like[valid_amp_snr_like] = (
            (refh_amp[valid_amp_snr_like] - bg_mean[valid_amp_snr_like]) / bg_std[valid_amp_snr_like]
        )
    return {
        "refh_snr": refh_snr,
        "refh_amp": refh_amp,
        "bg_mean": bg_mean,
        "bg_std": bg_std,
        "track_num": track_num,
        "sweep_num": sweep_num,
        "refh_amp_robust_z": robust_zscore(refh_amp, eps=eps),
        "refh_snr_robust_z": robust_zscore(refh_snr, eps=eps),
        "amp_snr_like": amp_snr_like,
    }

def compute_dtm_support_features(x: np.ndarray, y: np.ndarray, ground_grid: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    n = x.shape[0]
    nearest_ground_cell_dist_m = np.full(n, np.nan, dtype=np.float64)
    idw_neighbor_count_within_search_radius = np.zeros(n, dtype=np.int32)
    nearest_ground_cell_support_count = np.full(n, np.nan, dtype=np.float64)

    finite_xy = np.isfinite(x) & np.isfinite(y)
    centers = np.asarray(ground_grid["valid_centers_xy"], dtype=np.float64)
    if not np.any(finite_xy) or centers.size == 0:
        return {
            "nearest_ground_cell_dist_m": nearest_ground_cell_dist_m,
            "idw_neighbor_count_within_search_radius": idw_neighbor_count_within_search_radius,
            "nearest_ground_cell_support_count": nearest_ground_cell_support_count,
        }

    search_radius = max(
        float(config["DTM_MAX_SEARCH_RADIUS_M"]),
        float(config["DTM_SUPPORT_DIST_MAX_FOR_GROUND_M"]),
    )
    tree = cKDTree(centers)
    query_xy = np.column_stack((x[finite_xy], y[finite_xy]))
    dists, idx = tree.query(query_xy, k=1, workers=-1)
    nearest_ground_cell_dist_m[finite_xy] = np.asarray(dists, dtype=np.float64)
    cell_counts = np.asarray(ground_grid["valid_cell_counts"], dtype=np.float64)
    nearest_ground_cell_support_count[finite_xy] = cell_counts[np.asarray(idx, dtype=np.int64)]
    try:
        counts = tree.query_ball_point(query_xy, r=search_radius, return_length=True, workers=-1)
    except TypeError:
        counts = [len(ids) for ids in tree.query_ball_point(query_xy, r=search_radius, workers=-1)]
    idw_neighbor_count_within_search_radius[finite_xy] = np.asarray(counts, dtype=np.int32)
    return {
        "nearest_ground_cell_dist_m": nearest_ground_cell_dist_m,
        "idw_neighbor_count_within_search_radius": idw_neighbor_count_within_search_radius,
        "nearest_ground_cell_support_count": nearest_ground_cell_support_count,
    }

def compute_group_robust_z(values: np.ndarray, groups: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    grp = np.asarray(groups, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr) & np.isfinite(grp)
    if not np.any(finite):
        return out

    min_group_size = int(config["SCANLINE_MIN_GROUP_SIZE"])
    eps = float(config["ROBUST_Z_NMAD_EPS"])

    finite_idx = np.flatnonzero(finite)
    order = np.argsort(grp[finite_idx], kind="mergesort")
    sorted_idx = finite_idx[order]
    sorted_groups = grp[sorted_idx]
    sorted_values = arr[sorted_idx]
    _, start_idx, counts = np.unique(sorted_groups, return_index=True, return_counts=True)

    for start, count in zip(start_idx, counts):
        if int(count) < min_group_size:
            continue
        group_slice = slice(int(start), int(start + count))
        group_values = sorted_values[group_slice]
        median, nmad = robust_median_nmad(group_values, eps=eps)
        if not np.isfinite(median) or not np.isfinite(nmad) or nmad <= 0.0:
            continue
        out[sorted_idx[group_slice]] = np.abs(group_values - median) / nmad
    return out

def compute_scanline_features(
    height_above_ground_m: np.ndarray,
    track_num: np.ndarray,
    sweep_num: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    n = np.asarray(height_above_ground_m).shape[0]
    track_hag_robust_z = np.full(n, np.nan, dtype=np.float64)
    sweep_hag_robust_z = np.full(n, np.nan, dtype=np.float64)
    if bool(config.get("SCANLINE_USE_TRACK_HAG_Z", True)):
        track_hag_robust_z = compute_group_robust_z(height_above_ground_m, track_num, config)
    if bool(config.get("SCANLINE_USE_SWEEP_HAG_Z", True)):
        sweep_hag_robust_z = compute_group_robust_z(height_above_ground_m, sweep_num, config)
    return {
        "track_hag_robust_z": track_hag_robust_z,
        "sweep_hag_robust_z": sweep_hag_robust_z,
    }

def classify_points_baseline(
    z: np.ndarray,
    local_ground_z_m: np.ndarray,
    dtm_sample_valid: np.ndarray,
    point_density_pts_m3: np.ndarray,
    config: Dict[str, Any],
    *,
    refh_snr: Optional[np.ndarray] = None,
    refh_amp: Optional[np.ndarray] = None,
    bg_mean: Optional[np.ndarray] = None,
    bg_std: Optional[np.ndarray] = None,
    track_num: Optional[np.ndarray] = None,
    sweep_num: Optional[np.ndarray] = None,
    local_neighbor_count: Optional[np.ndarray] = None,
    dtm_support_features: Optional[Dict[str, np.ndarray]] = None,
    scanline_features: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    del bg_mean, bg_std, track_num, sweep_num, local_neighbor_count

    config = {**DEFAULT_CONFIG, **config}
    mode = str(config.get("CLASSIFIER_MODE", "height_density_processed_only"))
    if mode not in SUPPORTED_CLASSIFIER_MODES:
        raise ValueError(f"Unsupported CLASSIFIER_MODE: {mode}")
    base_mode = "height_density_processed_only" if mode == "rule_combined_v1" else mode

    n = z.shape[0]
    density = np.asarray(point_density_pts_m3, dtype=np.float64)
    hag = np.full(n, np.nan, dtype=np.float64)
    valid_dtm = np.asarray(dtm_sample_valid, dtype=bool)
    hag[valid_dtm] = np.asarray(z, dtype=np.float64)[valid_dtm] - np.asarray(local_ground_z_m, dtype=np.float64)[valid_dtm]

    pred = np.full(n, 7, dtype=np.uint8)
    reason = np.full(n, 0, dtype=np.uint8)
    invalid_dtm = ~valid_dtm
    pred[invalid_dtm] = 1
    reason[invalid_dtm] = 2

    valid_hag = valid_dtm & np.isfinite(hag)
    nonfinite_hag = valid_dtm & ~np.isfinite(hag)
    pred[nonfinite_hag] = 7
    reason[nonfinite_hag] = 18

    density_threshold = config.get("NOISE_DENSITY_MAX_PTS_M3")
    density_enabled = density_threshold is not None and base_mode != "height_only"
    low_density_baseline = np.zeros(n, dtype=bool)
    if density_enabled:
        low_density_baseline = valid_hag & np.isfinite(density) & (density < float(density_threshold))

    ground = valid_hag & (np.abs(hag) <= float(config["GROUND_RESID_TOL_M"]))
    below_ground = valid_hag & (hag < -float(config["BELOW_GROUND_NOISE_MIN_DEPTH_M"]))
    above_max = valid_hag & (hag > float(config["NOISE_HAG_MAX_M"]))
    density_reason_code: Optional[int] = None
    low_density = np.zeros(n, dtype=bool)

    if base_mode == "height_only":
        ground_final = ground
        below_final = below_ground & ~ground_final
        above_final = above_max & ~(ground_final | below_final)
        processed_final = valid_hag & ~(ground_final | below_final | above_final)
    elif base_mode == "height_density_pre_ground":
        low_density = low_density_baseline.copy()
        ground_final = ground & ~low_density
        below_final = below_ground & ~(low_density | ground_final)
        above_final = above_max & ~(low_density | ground_final | below_final)
        processed_final = valid_hag & ~(low_density | ground_final | below_final | above_final)
        density_reason_code = 6
    elif base_mode == "height_density_post_ground":
        ground_final = ground
        low_density = low_density_baseline & ~ground_final
        below_final = below_ground & ~(ground_final | low_density)
        above_final = above_max & ~(ground_final | low_density | below_final)
        processed_final = valid_hag & ~(ground_final | low_density | below_final | above_final)
        density_reason_code = 8
    elif base_mode in {"height_density_processed_only", "rule_combined_v1"}:
        ground_final = ground
        below_final = below_ground & ~ground_final
        above_final = above_max & ~(ground_final | below_final)
        processed_candidates = valid_hag & ~(ground_final | below_final | above_final)
        low_density = low_density_baseline & processed_candidates
        processed_final = processed_candidates & ~low_density
        density_reason_code = 9
    elif base_mode == "height_no_below_noise":
        ground_final = ground
        below_final = np.zeros(n, dtype=bool)
        above_final = above_max & ~ground_final
        processed_final = valid_hag & ~(ground_final | above_final)
    else:
        raise ValueError(f"Unsupported CLASSIFIER_MODE: {mode}")

    pred[ground_final] = 2
    reason[ground_final] = 1
    if density_reason_code is not None:
        pred[low_density] = 7
        reason[low_density] = int(density_reason_code)
    pred[below_final] = 7
    reason[below_final] = 3
    pred[above_final] = 7
    reason[above_final] = 4
    pred[processed_final] = 1
    reason[processed_final] = 5
    if base_mode == "height_no_below_noise":
        negative_processed = processed_final & (hag < 0.0)
        reason[negative_processed] = 7

    low_density_rule_mask = np.isfinite(density) & (density < float(config["NOISE_LOW_DENSITY_MAX_PTS_M3"]))
    low_snr_mask = np.zeros(n, dtype=bool)
    if refh_snr is not None:
        low_snr_mask = np.isfinite(refh_snr) & (np.asarray(refh_snr, dtype=np.float64) <= float(config["NOISE_LOW_SNR_MAX"]))
    low_amp_mask = np.zeros(n, dtype=bool)
    if refh_amp is not None:
        low_amp_mask = np.isfinite(refh_amp) & (
            robust_zscore(np.asarray(refh_amp, dtype=np.float64), eps=float(config["ROBUST_Z_NMAD_EPS"]))
            <= float(config["NOISE_LOW_AMP_ROBUST_Z_MAX"])
        )
    rule_signal_low = low_snr_mask | low_amp_mask

    weak_dtm_mask = np.zeros(n, dtype=bool)
    if dtm_support_features is not None and "nearest_ground_cell_dist_m" in dtm_support_features:
        nearest_dist = np.asarray(dtm_support_features["nearest_ground_cell_dist_m"], dtype=np.float64)
        weak_dtm_mask = np.isfinite(nearest_dist) & (
            nearest_dist > float(config["DTM_SUPPORT_DIST_MAX_FOR_GROUND_M"])
        )

    rule_scanline_outlier = np.zeros(n, dtype=bool)
    if scanline_features is not None:
        z_min = float(config["SCANLINE_HAG_ZSCORE_MIN"])
        z_max_config = config.get("SCANLINE_HAG_ZSCORE_MAX")
        z_max = None if z_max_config is None else float(z_max_config)

        track_z = np.asarray(
            scanline_features.get("track_hag_robust_z", np.full(n, np.nan, dtype=np.float64)),
            dtype=np.float64,
        )
        sweep_z = np.asarray(
            scanline_features.get("sweep_hag_robust_z", np.full(n, np.nan, dtype=np.float64)),
            dtype=np.float64,
        )
        if bool(config.get("SCANLINE_USE_TRACK_HAG_Z", True)):
            track_outlier = np.isfinite(track_z) & (track_z >= z_min)
            if z_max is not None:
                track_outlier &= track_z <= z_max
            rule_scanline_outlier |= track_outlier
        if bool(config.get("SCANLINE_USE_SWEEP_HAG_Z", True)):
            sweep_outlier = np.isfinite(sweep_z) & (sweep_z >= z_min)
            if z_max is not None:
                sweep_outlier &= sweep_z <= z_max
            rule_scanline_outlier |= sweep_outlier

    if bool(config["USE_NEAR_GROUND_GUARD"]):
        near_ground = (pred == 2) & np.isfinite(hag) & (np.abs(hag) <= float(config["NEAR_GROUND_GUARD_HAG_ABS_MAX_M"]))
        signal_guard = near_ground & low_snr_mask
        weak_guard = near_ground & ~signal_guard & weak_dtm_mask
        density_guard = near_ground & ~signal_guard & ~weak_guard & low_density_rule_mask
        if str(config["NEAR_GROUND_GUARD_ACTION"]) == "ground_to_processed":
            pred[signal_guard] = 1
            reason[signal_guard] = 10
            pred[weak_guard] = 1
            reason[weak_guard] = 11
            pred[density_guard] = 1
            reason[density_guard] = 12

    if bool(config["USE_DTM_SUPPORT_CONFIDENCE"]):
        weak_ground = (pred == 2) & weak_dtm_mask
        if str(config["DTM_SUPPORT_WEAK_GROUND_ACTION"]) == "ground_to_processed":
            pred[weak_ground] = 1
            reason[weak_ground] = 11

    if bool(config["USE_SIGNAL_DENSITY_NOISE"]):
        processed = pred == 1
        signal_density_candidate = (
            processed
            & np.isfinite(hag)
            & (np.abs(hag) >= float(config["NOISE_MIN_ABS_HAG_FOR_SIGNAL_RULE_M"]))
            & low_density_rule_mask
        )
        low_signal_noise = signal_density_candidate & low_snr_mask
        low_amp_noise = signal_density_candidate & ~low_signal_noise & low_amp_mask
        if str(config["NOISE_SIGNAL_DENSITY_ACTION"]) == "processed_to_noise":
            pred[low_signal_noise] = 7
            reason[low_signal_noise] = 13
            pred[low_amp_noise] = 7
            reason[low_amp_noise] = 14

    if bool(config["USE_SCANLINE_OUTLIER_NOISE"]):
        processed = pred == 1
        scanline_candidate = processed & rule_scanline_outlier
        if config.get("SCANLINE_HAG_MIN_M") is not None:
            scanline_candidate &= np.isfinite(hag) & (hag >= float(config["SCANLINE_HAG_MIN_M"]))
        if config.get("SCANLINE_HAG_MAX_M") is not None:
            scanline_candidate &= np.isfinite(hag) & (hag <= float(config["SCANLINE_HAG_MAX_M"]))
        if config.get("SCANLINE_DENSITY_MIN_PTS_M3") is not None:
            scanline_candidate &= (
                np.isfinite(density)
                & (density >= float(config["SCANLINE_DENSITY_MIN_PTS_M3"]))
            )
        if bool(config["SCANLINE_REQUIRE_LOW_DENSITY"]):
            scanline_candidate &= low_density_rule_mask
        if bool(config["SCANLINE_REQUIRE_NON_GROUND"]):
            scanline_candidate &= pred != 2
        pred[scanline_candidate] = 7
        reason[scanline_candidate] = 15

    if bool(config["USE_BELOW_GROUND_REFINEMENT"]) and config["BELOW_GROUND_SHALLOW_TO_PROCESSED_M"] is not None:
        shallow_limit = float(config["BELOW_GROUND_SHALLOW_TO_PROCESSED_M"])
        shallow_negative = (pred == 7) & (reason == 3) & np.isfinite(hag) & (hag < 0.0) & (np.abs(hag) <= shallow_limit)
        if bool(config["BELOW_GROUND_REQUIRE_LOW_DENSITY"]):
            shallow_negative &= low_density_rule_mask
        elif np.any(np.isfinite(density)):
            shallow_negative &= ~low_density_rule_mask
        if bool(config["BELOW_GROUND_REQUIRE_LOW_SNR"]):
            shallow_negative &= low_snr_mask
        elif refh_snr is not None and np.any(np.isfinite(refh_snr)):
            shallow_negative &= ~low_snr_mask

        weak_shallow = shallow_negative & weak_dtm_mask
        general_shallow = shallow_negative & ~weak_shallow
        pred[general_shallow] = 1
        reason[general_shallow] = 16
        if str(config["DTM_SUPPORT_WEAK_SHALLOW_NEGATIVE_ACTION"]) == "processed":
            pred[weak_shallow] = 1
            reason[weak_shallow] = 17

    result = {
        "pred_class_baseline": pred.astype(np.uint8),
        "classification_reason": reason.astype(np.uint8),
        "height_above_ground_m": hag.astype(np.float64),
    }
    result["rule_signal_low"] = rule_signal_low.astype(np.uint8)
    result["rule_density_low"] = low_density_rule_mask.astype(np.uint8)
    result["rule_scanline_outlier"] = rule_scanline_outlier.astype(np.uint8)
    result["rule_dtm_support_weak"] = weak_dtm_mask.astype(np.uint8)
    return result
