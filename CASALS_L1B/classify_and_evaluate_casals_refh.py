"""
Deterministic CASALS refh classification and pseudo-reference evaluation.

This script keeps the existing CASALS refh classification workflow and adds
optional deterministic, non-learning rule enhancements and diagnostics.
Transferred 3DEP labels are used only as pseudo-reference labels for evaluation.
No vertical datum correction is applied by this script.
"""

from __future__ import annotations

import json
import math
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

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
    "OUTPUT_ROOT": Path("./outputs/classify_and_evaluate_casals_refh"),
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


def print_console_metrics_table(h5_stem: str, evaluation_metrics: Dict[str, Any]) -> None:
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
        print("Confusion matrix (rows=pseudo-reference, cols=prediction):")
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


def print_density_quantiles_table(pred_class_baseline: np.ndarray, point_density_pts_m3: np.ndarray) -> None:
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
        "error_feature_summary_csv": root / f"{h5_stem}_error_feature_summary.csv",
        "true7_pred1_samples_csv": root / f"{h5_stem}_true7_pred1_samples.csv",
        "true1_pred2_samples_csv": root / f"{h5_stem}_true1_pred2_samples.csv",
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
            if key == "CLASSIFIER_MODE"
            or key.startswith(
                (
                    "GROUND_",
                    "GRID_",
                    "MIN_",
                    "DTM_",
                    "LAS_",
                    "IDW_",
                    "LOCAL_",
                    "NOISE_",
                    "USE_",
                    "WRITE_",
                    "SCANLINE_",
                    "COMPUTE_",
                    "BELOW_",
                    "ROBUST_",
                )
            )
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
    density_source: str,
) -> Dict[str, Any]:
    density = np.asarray(point_density_pts_m3, dtype=np.float64)
    density_finite = np.isfinite(density)
    density_reason_codes = {6, 8, 9, 13, 14}
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
            "density_noise_count": int(
                np.count_nonzero(np.isin(np.asarray(classification_reason, dtype=np.uint8), list(density_reason_codes)))
            ),
            "density_valid_count": int(np.count_nonzero(density_finite)),
            "density_valid_fraction": float(np.mean(density_finite)),
        },
        "density_source": str(density_source),
        "density_summary": summarize_values(density),
    }


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
    pred[invalid_dtm] = 7
    reason[invalid_dtm] = 2

    valid_hag = valid_dtm & np.isfinite(hag)
    nonfinite_hag = valid_dtm & ~np.isfinite(hag)
    pred[nonfinite_hag] = 7
    reason[nonfinite_hag] = 2

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
                aligned = (
                    np.full(n_pred, np.nan, dtype=np.float64)
                    if name != "transfer_status"
                    else np.full(n_pred, -1, dtype=np.int16)
                )
                aligned[ref_idx] = np.asarray(reference[name])
                if name == "transfer_status":
                    result["reference_transfer_status"] = aligned.astype(np.int16)
                elif name == "nearest3dep_dist_m":
                    result["reference_nearest3dep_dist_m"] = aligned.astype(np.float64)
                elif name == "class_vote_ratio":
                    result["reference_class_vote_ratio"] = aligned.astype(np.float64)
        return result

    if n_ref != n_pred:
        raise RuntimeError("Reference has no point_index and point counts differ; row-order alignment is not safe.")

    fractions = tuple(config["ROW_ALIGN_CHECK_INDICES"])
    sample_idx = sorted({int(np.clip(round(frac * (n_pred - 1)), 0, n_pred - 1)) for frac in fractions})
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


def compute_main_error_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_pred = np.asarray(y_pred, dtype=np.uint8)
    true1 = y_true == 1
    true2 = y_true == 2
    true7 = y_true == 7
    true1_pred2_count = int(np.count_nonzero(true1 & (y_pred == 2)))
    true7_pred1_count = int(np.count_nonzero(true7 & (y_pred == 1)))
    true2_pred1_count = int(np.count_nonzero(true2 & (y_pred == 1)))
    true2_pred7_count = int(np.count_nonzero(true2 & (y_pred == 7)))
    return {
        "true1_pred2_count": true1_pred2_count,
        "true7_pred1_count": true7_pred1_count,
        "true2_pred1_count": true2_pred1_count,
        "true2_pred7_count": true2_pred7_count,
        "true1_pred2_fraction_of_true1": (
            None if not np.any(true1) else float(true1_pred2_count / np.count_nonzero(true1))
        ),
        "true7_pred1_fraction_of_true7": (
            None if not np.any(true7) else float(true7_pred1_count / np.count_nonzero(true7))
        ),
    }


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
    density_reason_codes = {6, 8, 9, 13, 14}
    row: Dict[str, Any] = {
        "h5_stem": h5_stem,
        "point_count": int(pred_class_baseline.size),
        "matched_count": int(np.count_nonzero(eval_match_valid)),
        "unmatched_count": int(pred_class_baseline.size - np.count_nonzero(eval_match_valid)),
        "dtm_invalid_count": int(np.count_nonzero(np.asarray(dtm_sample_valid) == 0)),
        "ground_support_candidate_count": int(ground_support_candidate_count),
        "valid_dtm_cell_count": int(valid_dtm_cell_count),
        "density_noise_count": int(
            np.count_nonzero(np.isin(np.asarray(classification_reason, dtype=np.uint8), list(density_reason_codes)))
        ),
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
        present_labels = sorted(int(v) for v in np.unique(np.concatenate([y_true, y_pred])) if int(v) in LABEL_ORDER)
        if present_labels:
            macro_present = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=present_labels,
                average="macro",
                zero_division=0,
            )
            weighted_present = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=present_labels,
                average="weighted",
                zero_division=0,
            )
        else:
            macro_present = (np.nan, np.nan, np.nan, None)
            weighted_present = (np.nan, np.nan, np.nan, None)
        accuracy = float(accuracy_score(y_true, y_pred))
        nonzero_support = [int(v) for v in support if int(v) > 0]
        support_min = min(nonzero_support) if nonzero_support else 0
        error_counts = compute_main_error_counts(y_true, y_pred)

        report_rows: list[Dict[str, Any]] = []
        per_class_metrics: Dict[int, Dict[str, Any]] = {}
        for i, cls in enumerate(LABEL_ORDER):
            per_class_metrics[cls] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            report_rows.append({
                "row_name": CLASS_NAME_MAP[cls],
                "class_code": cls,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            })
        report_rows.extend([
            {
                "row_name": "macro avg",
                "class_code": "",
                "precision": float(macro[0]),
                "recall": float(macro[1]),
                "f1_score": float(macro[2]),
                "support": int(np.sum(support)),
            },
            {
                "row_name": "weighted avg",
                "class_code": "",
                "precision": float(weighted[0]),
                "recall": float(weighted[1]),
                "f1_score": float(weighted[2]),
                "support": int(np.sum(support)),
            },
            {
                "row_name": "accuracy",
                "class_code": "",
                "precision": None,
                "recall": None,
                "f1_score": accuracy,
                "support": int(np.sum(support)),
            },
        ])

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
            "present_labels": present_labels,
            "present_label_names": [CLASS_NAME_MAP[int(v)] for v in present_labels],
            "macro_precision_present": None if not np.isfinite(macro_present[0]) else float(macro_present[0]),
            "macro_recall_present": None if not np.isfinite(macro_present[1]) else float(macro_present[1]),
            "macro_f1_present": None if not np.isfinite(macro_present[2]) else float(macro_present[2]),
            "weighted_precision_present": None if not np.isfinite(weighted_present[0]) else float(weighted_present[0]),
            "weighted_recall_present": None if not np.isfinite(weighted_present[1]) else float(weighted_present[1]),
            "weighted_f1_present": None if not np.isfinite(weighted_present[2]) else float(weighted_present[2]),
            "n_present_labels": int(len(present_labels)),
            "support_min": int(support_min),
            "support_1": int(support[0]),
            "support_2": int(support[1]),
            "support_7": int(support[2]),
            "confusion_matrix": cm.astype(int).tolist(),
            "report_rows": report_rows,
            "confusion_rows": confusion_rows,
            "per_class_metrics": per_class_metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            **error_counts,
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
            "macro_precision_present": None if not np.isfinite(macro_present[0]) else float(macro_present[0]),
            "macro_recall_present": None if not np.isfinite(macro_present[1]) else float(macro_present[1]),
            "macro_f1_present": None if not np.isfinite(macro_present[2]) else float(macro_present[2]),
            "weighted_precision_present": None if not np.isfinite(weighted_present[0]) else float(weighted_present[0]),
            "weighted_recall_present": None if not np.isfinite(weighted_present[1]) else float(weighted_present[1]),
            "weighted_f1_present": None if not np.isfinite(weighted_present[2]) else float(weighted_present[2]),
            "n_present_labels": int(len(present_labels)),
            "support_min": int(support_min),
            "support_1": int(support[0]),
            "support_2": int(support[1]),
            "support_7": int(support[2]),
            **error_counts,
        })

        if subset_name == "all_matched":
            primary_report_df = pd.DataFrame(report_rows)
            primary_confusion_df = pd.DataFrame(confusion_rows)
            primary_metrics = {
                "accuracy": accuracy,
                "macro_f1": float(macro[2]),
                "weighted_f1": float(weighted[2]),
                "per_class_metrics": per_class_metrics,
                **error_counts,
            }

    return {
        "subset_results": subset_results,
        "evaluation_summary_rows": evaluation_summary_rows,
        "primary_report_df": primary_report_df,
        "primary_confusion_df": primary_confusion_df,
        "primary_metrics": primary_metrics,
    }


def build_error_subset_masks(
    eval_gt_class: np.ndarray,
    pred_class_baseline: np.ndarray,
    eval_match_valid: np.ndarray,
) -> Dict[str, np.ndarray]:
    matched = np.asarray(eval_match_valid, dtype=bool)
    y_true = np.asarray(eval_gt_class, dtype=np.uint8)
    y_pred = np.asarray(pred_class_baseline, dtype=np.uint8)
    return {
        "true7_pred1": matched & (y_true == 7) & (y_pred == 1),
        "true1_pred2": matched & (y_true == 1) & (y_pred == 2),
        "correct_ground": matched & (y_true == 2) & (y_pred == 2),
        "correct_noise": matched & (y_true == 7) & (y_pred == 7),
        "correct_processed": matched & (y_true == 1) & (y_pred == 1),
    }


def build_error_feature_rows(
    h5_stem: str,
    subset_masks: Dict[str, np.ndarray],
    feature_map: Dict[str, Optional[np.ndarray]],
) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, np.ndarray]]]:
    rows: list[Dict[str, Any]] = []
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    for subset_name, mask in subset_masks.items():
        subset_cache: Dict[str, np.ndarray] = {}
        subset_count = int(np.count_nonzero(mask))
        for feature_name, values in feature_map.items():
            if values is None:
                continue
            subset_values = np.asarray(values)[mask]
            subset_cache[feature_name] = subset_values
            summary = quantile_summary(subset_values)
            rows.append({
                "h5_stem": h5_stem,
                "subset_name": subset_name,
                "feature_name": feature_name,
                "subset_count": subset_count,
                **summary,
            })
        cache[subset_name] = subset_cache
    return rows, cache


def maybe_write_error_subset_samples(
    output_paths: Dict[str, Path],
    subset_masks: Dict[str, np.ndarray],
    config: Dict[str, Any],
    sample_fields: Dict[str, np.ndarray],
) -> list[str]:
    if not bool(config["WRITE_ERROR_SUBSET_SAMPLES"]):
        return []
    written: list[str] = []
    max_points = int(config["MAX_ERROR_SUBSET_SAMPLE_POINTS"])
    output_specs = {
        "true7_pred1": output_paths["true7_pred1_samples_csv"],
        "true1_pred2": output_paths["true1_pred2_samples_csv"],
    }
    for subset_name, output_path in output_specs.items():
        mask = subset_masks[subset_name]
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        indices = indices[:max_points]
        frame = pd.DataFrame({
            key: np.asarray(values)[indices]
            for key, values in sample_fields.items()
        })
        frame.to_csv(output_path, index=False)
        written.append(str(output_path))
    return written


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
    optional_extra_arrays: Optional[Dict[str, np.ndarray]] = None,
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
        ExtraBytesParams(name="classification_reason", type=np.uint8, description="Deterministic class reason"),
        ExtraBytesParams(name="pred_class_baseline", type=np.uint8, description="Deterministic class"),
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
    optional_extra_arrays = optional_extra_arrays or {}
    optional_extra_specs = {
        "nearest_ground_cell_dist_m": np.float32,
        "track_hag_robust_z": np.float32,
        "sweep_hag_robust_z": np.float32,
    }
    for name, dtype in optional_extra_specs.items():
        if name in optional_extra_arrays:
            extra_dims.append(ExtraBytesParams(name=name, type=dtype, description=name[:32]))
    if eval_gt_class is not None:
        extra_dims.append(ExtraBytesParams(name="eval_gt_class", type=np.uint8, description="Pseudo-reference class"))

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
    for name, dtype in optional_extra_specs.items():
        if name not in optional_extra_arrays:
            continue
        arr = np.asarray(optional_extra_arrays[name])
        if dtype == np.float32:
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
    error_feature_rows: list[Dict[str, Any]],
) -> None:
    pd.DataFrame([classification_summary_row]).to_csv(output_paths["classification_summary_csv"], index=False)
    evaluation_metrics["primary_report_df"].to_csv(output_paths["classification_report_csv"], index=False)
    evaluation_metrics["primary_confusion_df"].to_csv(output_paths["confusion_matrix_csv"], index=False)
    if bool(config["WRITE_ERROR_FEATURE_SUMMARY"]):
        pd.DataFrame(error_feature_rows).to_csv(output_paths["error_feature_summary_csv"], index=False)

    subset_json = {}
    for subset_name, subset_result in evaluation_metrics["subset_results"].items():
        subset_copy = {k: v for k, v in subset_result.items() if k not in {"y_true", "y_pred"}}
        subset_json[subset_name] = subset_copy

    evaluation_json = {
        "h5_stem": h5_stem,
        "label_order": LABEL_ORDER,
        "class_name_map": CLASS_NAME_MAP,
        "subset_results": subset_json,
        "evaluation_summary_rows": evaluation_metrics["evaluation_summary_rows"],
        "true1_pred2_count": metadata.get("true1_pred2_count"),
        "true7_pred1_count": metadata.get("true7_pred1_count"),
        "true2_pred1_count": metadata.get("true2_pred1_count"),
        "true2_pred7_count": metadata.get("true2_pred7_count"),
        "true1_pred2_fraction_of_true1": metadata.get("true1_pred2_fraction_of_true1"),
        "true7_pred1_fraction_of_true7": metadata.get("true7_pred1_fraction_of_true7"),
        "reference_type": "3dep_transferred_pseudo_reference",
    }
    safe_json_dump(evaluation_json, output_paths["evaluation_summary_json"])
    safe_json_dump(metadata, output_paths["run_metadata_json"])

    if bool(config["WRITE_DIAGNOSTIC_PNG"]):
        pred_counts = collect_class_counts(pred_class_baseline)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.bar([CLASS_NAME_MAP[c] for c in LABEL_ORDER], [pred_counts.get(c, 0) for c in LABEL_ORDER], color=["#b98b27", "#4477aa", "#cc6677"])
        ax.set_title(f"{h5_stem} predicted class counts")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(output_paths["class_count_png"], dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for cls in LABEL_ORDER:
            mask = np.asarray(pred_class_baseline, dtype=np.uint8) == cls
            values = np.asarray(height_above_ground_m, dtype=np.float64)[mask]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            ax.hist(values, bins=120, histtype="step", linewidth=1.2, label=CLASS_NAME_MAP[cls])
        ax.set_title(f"{h5_stem} height above ground by predicted class")
        ax.set_xlabel("Height above ground (m)")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_paths["hag_hist_png"], dpi=160)
        plt.close(fig)

        cm_rows = evaluation_metrics["primary_confusion_df"]
        if not cm_rows.empty:
            cm = cm_rows[["pred_1", "pred_2", "pred_7"]].to_numpy(dtype=np.int64)
            fig, ax = plt.subplots(figsize=(5.0, 4.2))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(LABEL_ORDER)))
            ax.set_xticklabels([CLASS_NAME_MAP[c] for c in LABEL_ORDER], rotation=25, ha="right")
            ax.set_yticks(range(len(LABEL_ORDER)))
            ax.set_yticklabels([CLASS_NAME_MAP[c] for c in LABEL_ORDER])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Pseudo-reference")
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

        n_points = casals["point_index"].size
        signal_features = compute_signal_features(casals["fields"], n_points, config)
        refh_snr = np.asarray(signal_features["refh_snr"], dtype=np.float64)
        refh_amp = np.asarray(signal_features["refh_amp"], dtype=np.float64)
        track_num = np.asarray(signal_features["track_num"], dtype=np.float64)
        sweep_num = np.asarray(signal_features["sweep_num"], dtype=np.float64)

        ground_grid = build_ground_grid(x, y, casals["z"], refh_snr, config)
        print(f"Ground support candidates: {ground_grid['support_count']:,}")
        print(f"Valid ground grid cells: {ground_grid['valid_cell_count']:,}")

        local_ground_z_m, dtm_sample_valid = sample_ground_grid_idw(x, y, ground_grid, config)
        density_enabled = needs_density_features(config)
        if density_enabled:
            density_result = compute_local_point_density(x, y, casals["z"], config)
            local_neighbor_count = density_result["local_neighbor_count"]
            point_density_pts_m3 = density_result["point_density_pts_m3"]
            density_source = "computed_internal"
        else:
            local_neighbor_count = np.zeros(n_points, dtype=np.uint32)
            point_density_pts_m3 = np.full(n_points, np.nan, dtype=np.float64)
            density_source = "disabled"

        dtm_support_features = None
        if needs_dtm_support_features(config):
            dtm_support_features = compute_dtm_support_features(x, y, ground_grid, config)

        pre_hag = np.full(n_points, np.nan, dtype=np.float64)
        valid_dtm = np.asarray(dtm_sample_valid, dtype=bool)
        pre_hag[valid_dtm] = np.asarray(casals["z"], dtype=np.float64)[valid_dtm] - local_ground_z_m[valid_dtm]

        scanline_features = None
        if needs_scanline_features(config):
            scanline_features = compute_scanline_features(pre_hag, track_num, sweep_num, config)

        classification_result = classify_points_baseline(
            casals["z"],
            local_ground_z_m,
            dtm_sample_valid,
            point_density_pts_m3,
            config,
            refh_snr=signal_features["refh_snr"],
            refh_amp=signal_features["refh_amp"],
            bg_mean=signal_features["bg_mean"],
            bg_std=signal_features["bg_std"],
            track_num=signal_features["track_num"],
            sweep_num=signal_features["sweep_num"],
            local_neighbor_count=local_neighbor_count,
            dtm_support_features=dtm_support_features,
            scanline_features=scanline_features,
        )
        pred_class_baseline = classification_result["pred_class_baseline"]
        classification_reason = classification_result["classification_reason"]
        height_above_ground_m = classification_result["height_above_ground_m"]
        pred_counts = collect_class_counts(pred_class_baseline)

        print(f"Classifier mode: {config['CLASSIFIER_MODE']}")
        print(f"Density source: {density_source}")
        print(
            "Rule switches: "
            f"near_ground={bool(config['USE_NEAR_GROUND_GUARD'])}, "
            f"signal_density={bool(config['USE_SIGNAL_DENSITY_NOISE'])}, "
            f"dtm_support={bool(config['USE_DTM_SUPPORT_CONFIDENCE'])}, "
            f"scanline={bool(config['USE_SCANLINE_OUTLIER_NOISE'])}, "
            f"below_ground={bool(config['USE_BELOW_GROUND_REFINEMENT'])}"
        )
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
            density_source=density_source,
        )
        optional_extra_arrays: Dict[str, np.ndarray] = {}
        if dtm_support_features is not None and "nearest_ground_cell_dist_m" in dtm_support_features:
            optional_extra_arrays["nearest_ground_cell_dist_m"] = dtm_support_features["nearest_ground_cell_dist_m"]
        if scanline_features is not None and "track_hag_robust_z" in scanline_features:
            optional_extra_arrays["track_hag_robust_z"] = scanline_features["track_hag_robust_z"]
        if scanline_features is not None and "sweep_hag_robust_z" in scanline_features:
            optional_extra_arrays["sweep_hag_robust_z"] = scanline_features["sweep_hag_robust_z"]

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
            "optional_extra_arrays": optional_extra_arrays,
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
        error_feature_rows: list[Dict[str, Any]] = []
        error_feature_cache: Dict[str, Dict[str, np.ndarray]] = {}
        error_subset_masks: Dict[str, np.ndarray] = {}

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
            for cls in LABEL_ORDER:
                per_class = primary.get("per_class_metrics", {}).get(cls, {})
                print(
                    f"Class {cls} F1 / recall: "
                    f"{float(per_class.get('f1', float('nan'))):.6f} / "
                    f"{float(per_class.get('recall', float('nan'))):.6f}"
                )
            print("Main error counts:")
            print(f"  true1 -> pred2: {primary.get('true1_pred2_count', 0)}")
            print(f"  true7 -> pred1: {primary.get('true7_pred1_count', 0)}")
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

            error_subset_masks = build_error_subset_masks(eval_gt_class, pred_class_baseline, eval_match_valid)
            error_feature_map = {
                "height_above_ground_m": height_above_ground_m,
                "refh_snr": signal_features["refh_snr"],
                "refh_amp": signal_features["refh_amp"],
                "point_density_pts_m3": point_density_pts_m3,
                "nearest_ground_cell_dist_m": None if dtm_support_features is None else dtm_support_features["nearest_ground_cell_dist_m"],
                "track_hag_robust_z": None if scanline_features is None else scanline_features["track_hag_robust_z"],
                "sweep_hag_robust_z": None if scanline_features is None else scanline_features["sweep_hag_robust_z"],
            }
            error_feature_rows, error_feature_cache = build_error_feature_rows(h5_stem, error_subset_masks, error_feature_map)
            sample_output_paths = maybe_write_error_subset_samples(
                output_paths=output_paths,
                subset_masks=error_subset_masks,
                config=config,
                sample_fields={
                    "point_index": casals["point_index"],
                    "longitude": casals["lon"],
                    "latitude": casals["lat"],
                    "x": x,
                    "y": y,
                    "z": casals["z"],
                    "pred_class": pred_class_baseline,
                    "pseudo_reference_class": eval_gt_class,
                    "height_above_ground_m": height_above_ground_m,
                    "refh_snr": signal_features["refh_snr"],
                    "refh_amp": signal_features["refh_amp"],
                    "point_density_pts_m3": point_density_pts_m3,
                    "classification_reason": classification_reason,
                    "track_num": signal_features["track_num"],
                    "sweep_num": signal_features["sweep_num"],
                },
            )
            if sample_output_paths:
                metadata["error_subset_sample_outputs"] = sample_output_paths
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
        primary_errors = evaluation_metrics["primary_metrics"]
        metadata["true1_pred2_count"] = int(primary_errors.get("true1_pred2_count", 0))
        metadata["true7_pred1_count"] = int(primary_errors.get("true7_pred1_count", 0))
        metadata["true2_pred1_count"] = int(primary_errors.get("true2_pred1_count", 0))
        metadata["true2_pred7_count"] = int(primary_errors.get("true2_pred7_count", 0))
        metadata["true1_pred2_fraction_of_true1"] = primary_errors.get("true1_pred2_fraction_of_true1")
        metadata["true7_pred1_fraction_of_true7"] = primary_errors.get("true7_pred1_fraction_of_true7")
        metadata["reason_code_map"] = CLASS_REASON_MAP
        metadata["classifier_mode"] = str(config["CLASSIFIER_MODE"])
        metadata["rule_switches"] = {key: bool(config[key]) for key in RULE_SWITCH_KEYS}
        metadata["rule_thresholds"] = {key: json_safe(config.get(key)) for key in RULE_THRESHOLD_KEYS}
        metadata["density_enabled"] = bool(density_enabled)
        metadata["density_source"] = density_source
        metadata["dtm_support_features_computed"] = bool(dtm_support_features is not None)
        metadata["scanline_features_computed"] = bool(scanline_features is not None)
        metadata["reference_type"] = "3dep_transferred_pseudo_reference"
        metadata["learning_based_methods_used"] = False
        metadata["supervised_training_used"] = False
        metadata["pseudo_reference_used_only_for_evaluation"] = True
        metadata["selected_rules_must_be_visually_validated"] = True
        metadata["scientific_notes"] = {
            "baseline_classifier": "deterministic rule-based CASALS refh classifier",
            "pseudo_reference": "transferred 3DEP labels are used only for evaluation and are not ground truth",
            "vertical_datum": "no vertical datum correction is applied; output Z remains CASALS refh height",
        }
        metadata["missing_optional_laz_dimensions"] = [
            name
            for name in ["nearest_ground_cell_dist_m", "track_hag_robust_z", "sweep_hag_robust_z"]
            if name not in optional_extra_arrays
        ]

        write_summary_outputs(
            h5_stem=h5_stem,
            config=config,
            output_paths=output_paths,
            classification_summary_row=classification_summary_row,
            evaluation_metrics=evaluation_metrics,
            metadata=metadata,
            pred_class_baseline=pred_class_baseline,
            height_above_ground_m=height_above_ground_m,
            error_feature_rows=error_feature_rows,
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
            "error_feature_rows": error_feature_rows,
            "error_feature_cache": error_feature_cache,
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
    error_feature_summary_path = output_root / "all_files_error_feature_summary.csv"
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
        per_class = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_ORDER, zero_division=0)
        macro = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
        weighted = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0)
        present_labels = sorted(int(v) for v in np.unique(np.concatenate([y_true, y_pred])) if int(v) in LABEL_ORDER)
        if present_labels:
            macro_present = precision_recall_fscore_support(
                y_true, y_pred, labels=present_labels, average="macro", zero_division=0
            )
            weighted_present = precision_recall_fscore_support(
                y_true, y_pred, labels=present_labels, average="weighted", zero_division=0
            )
        else:
            macro_present = (np.nan, np.nan, np.nan, None)
            weighted_present = (np.nan, np.nan, np.nan, None)
        support = per_class[3]
        nonzero_support = [int(v) for v in support if int(v) > 0]
        error_counts = compute_main_error_counts(y_true, y_pred)
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
            "macro_precision_present": None if not np.isfinite(macro_present[0]) else float(macro_present[0]),
            "macro_recall_present": None if not np.isfinite(macro_present[1]) else float(macro_present[1]),
            "macro_f1_present": None if not np.isfinite(macro_present[2]) else float(macro_present[2]),
            "weighted_precision_present": None if not np.isfinite(weighted_present[0]) else float(weighted_present[0]),
            "weighted_recall_present": None if not np.isfinite(weighted_present[1]) else float(weighted_present[1]),
            "weighted_f1_present": None if not np.isfinite(weighted_present[2]) else float(weighted_present[2]),
            "n_present_labels": int(len(present_labels)),
            "support_min": int(min(nonzero_support)) if nonzero_support else 0,
            "support_1": int(support[0]),
            "support_2": int(support[1]),
            "support_7": int(support[2]),
            **error_counts,
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
            "macro_precision_present",
            "macro_recall_present",
            "macro_f1_present",
            "weighted_precision_present",
            "weighted_recall_present",
            "weighted_f1_present",
            "n_present_labels",
            "support_min",
            "support_1",
            "support_2",
            "support_7",
            "true1_pred2_count",
            "true7_pred1_count",
            "true2_pred1_count",
            "true2_pred7_count",
            "true1_pred2_fraction_of_true1",
            "true7_pred1_fraction_of_true7",
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

    all_error_feature_rows = []
    aggregate_error_cache: Dict[str, Dict[str, list[np.ndarray]]] = {}
    for result in successful_results:
        all_error_feature_rows.extend(result["error_feature_rows"])
        for subset_name, feature_map in result["error_feature_cache"].items():
            aggregate_error_cache.setdefault(subset_name, {})
            for feature_name, values in feature_map.items():
                aggregate_error_cache[subset_name].setdefault(feature_name, []).append(np.asarray(values))
    for subset_name, feature_map in aggregate_error_cache.items():
        subset_count = None
        for feature_name, arrays in feature_map.items():
            merged = np.concatenate(arrays) if arrays else np.array([], dtype=np.float64)
            subset_count = int(merged.size) if subset_count is None else subset_count
            summary = quantile_summary(merged)
            all_error_feature_rows.append({
                "h5_stem": "__all__",
                "subset_name": subset_name,
                "feature_name": feature_name,
                "subset_count": subset_count,
                **summary,
            })
    if bool(config["WRITE_ERROR_FEATURE_SUMMARY"]):
        pd.DataFrame(all_error_feature_rows).to_csv(error_feature_summary_path, index=False)

    all_metadata["totals"] = {
        "n_input_pairs": len(input_pairs),
        "n_successful_pairs": len(successful_results),
        "n_failed_pairs": len(failed_results),
    }
    safe_json_dump(all_metadata, metadata_path)


def main() -> None:
    input_pairs = [
        {
            "h5_path": Path("./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
            "reference_laz_path": Path("./outputs/transfer_3dep_labels_to_casals/casals_l1b_20241112T165718_001_02.laz"),
        },
        {
            "h5_path": Path("./casals_h5_downloads/casals_l1b_20241118T171757_001_02.h5"),
            "reference_laz_path": Path("./outputs/transfer_3dep_labels_to_casals/casals_l1b_20241118T171757_001_02.laz"),
        },
    ]

    config_overrides: Dict[str, Any] = {}
    config, config_warnings = normalize_config(config_overrides)
    ensure_dir(Path(config["OUTPUT_ROOT"]))

    successful_results = []
    failed_results = []
    all_metadata = {
        "runtime_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": json_safe(config),
        "input_pairs": json_safe(input_pairs),
        "successful_files": [],
        "failed_files": [],
        "warnings": config_warnings,
        "outputs": {
            "all_files_classification_summary_csv": str(Path(config["OUTPUT_ROOT"]) / "all_files_classification_summary.csv"),
            "all_files_evaluation_summary_csv": str(Path(config["OUTPUT_ROOT"]) / "all_files_evaluation_summary.csv"),
            "all_files_confusion_matrix_csv": str(Path(config["OUTPUT_ROOT"]) / "all_files_confusion_matrix.csv"),
            "all_files_error_feature_summary_csv": str(Path(config["OUTPUT_ROOT"]) / "all_files_error_feature_summary.csv"),
            "all_files_run_metadata_json": str(Path(config["OUTPUT_ROOT"]) / "all_files_run_metadata.json"),
        },
        "reference_type": "3dep_transferred_pseudo_reference",
        "learning_based_methods_used": False,
        "supervised_training_used": False,
        "pseudo_reference_used_only_for_evaluation": True,
        "selected_rules_must_be_visually_validated": True,
    }

    for input_pair in input_pairs:
        print(f"\n=== Processing pair: {input_pair['h5_path'].stem} ===")
        result = process_one_pair(input_pair, config)
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
        config=config,
        input_pairs=input_pairs,
        all_metadata=all_metadata,
    )


if __name__ == "__main__":
    main()
