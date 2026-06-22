"""
Debug parameter sweep and ablation diagnostics for the baseline CASALS refh classifier.

This script reuses the existing CASALS refh read / project / DTM / evaluation helpers
from classify_and_evaluate_casals_refh.py, but does not write per-config LAZ files.
Its purpose is to compare classifier variants, tune thresholds, and summarize failure
or confusion patterns against the transferred 3DEP pseudo-reference labels.
"""

from __future__ import annotations

import itertools
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import classify_and_evaluate_casals_refh as baseline_mod
    from classify_and_evaluate_casals_refh import (
        CLASS_NAME_MAP,
        CLASS_REASON_MAP,
        LABEL_ORDER,
        align_prediction_to_reference,
        build_classification_summary_row,
        build_ground_grid,
        classify_points_baseline,
        collect_class_counts,
        compute_evaluation_metrics,
        compute_local_point_density,
        ensure_dir,
        infer_or_choose_projected_crs,
        json_safe,
        map_reference_labels_to_baseline_classes,
        maybe_quantiles,
        project_lonlat_to_xy,
        read_casals_h5_refh_points,
        read_reference_labels,
        safe_json_dump,
        sample_ground_grid_idw,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import classify_and_evaluate_casals_refh.py. "
        "Run this script from the CASALS_L1B directory, for example:\n"
        "  cd CASALS_L1B\n"
        "  python debug_casals_refh_classifier_params.py\n"
        f"Original error: {exc}"
    ) from exc


DEBUG_REASON_MAP = dict(CLASS_REASON_MAP)
DEBUG_REASON_MAP[4] = "noise_above_max_hag"
DEBUG_REASON_MAP.update({
    7: "processed_negative_hag_allowed",
    8: "density_applied_after_ground",
    9: "density_applied_processed_only",
})

PARAM_COLUMNS = [
    "classifier_mode",
    "GROUND_SNR_MIN",
    "GRID_RES_M",
    "GROUND_CELL_PERCENTILE",
    "MIN_POINTS_PER_CELL",
    "DTM_IDW_K",
    "DTM_IDW_POWER",
    "DTM_MAX_SEARCH_RADIUS_M",
    "GROUND_RESID_TOL_M",
    "NOISE_HAG_MAX_M",
    "LOCAL_FEATURE_RADIUS_M",
    "NOISE_DENSITY_MAX_PTS_M3",
]

RESULT_COLUMNS = [
    "h5_stem",
    "config_id",
    "subset_name",
    "status",
    "warning",
    *PARAM_COLUMNS,
    "point_count",
    "matched_count",
    "unmatched_count",
    "ground_support_candidate_count",
    "valid_dtm_cell_count",
    "dtm_invalid_count",
    "pred_count_1",
    "pred_count_2",
    "pred_count_7",
    "ref_count_1",
    "ref_count_2",
    "ref_count_7",
    "pred_fraction_1",
    "pred_fraction_2",
    "pred_fraction_7",
    "ref_fraction_1",
    "ref_fraction_2",
    "ref_fraction_7",
    "fraction_abs_diff_sum",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "class_1_precision",
    "class_1_recall",
    "class_1_f1",
    "class_1_support",
    "class_2_precision",
    "class_2_recall",
    "class_2_f1",
    "class_2_support",
    "class_7_precision",
    "class_7_recall",
    "class_7_f1",
    "class_7_support",
    "objective_score",
]

RANKED_COLUMNS = [
    "h5_stem",
    "config_id",
    *PARAM_COLUMNS,
    "n_files_success",
    "mean_objective_score",
    "min_objective_score",
    "std_objective_score",
    "mean_macro_f1",
    "mean_weighted_f1",
    "mean_class_1_f1",
    "mean_class_2_f1",
    "mean_class_7_f1",
    "mean_class_2_recall",
    "mean_class_7_recall",
    "mean_fraction_abs_diff_sum",
    "rank",
]

BEST_BY_FILE_COLUMNS = [
    "h5_stem",
    "config_id",
    "rank_within_file",
    "global_rank",
    *PARAM_COLUMNS,
    "objective_score",
    "macro_f1",
    "weighted_f1",
    "class_2_recall",
    "class_7_recall",
    "fraction_abs_diff_sum",
    "status",
]

ABLATION_COLUMNS = [
    "h5_stem",
    "config_id",
    "classifier_mode",
    "n_runs",
    "mean_objective_score",
    "median_objective_score",
    "mean_macro_f1",
    "median_macro_f1",
    "mean_class_2_recall",
    "mean_class_7_recall",
]

CONFUSION_LONG_COLUMNS = [
    "h5_stem",
    "config_id",
    "subset_name",
    "status",
    "true_class",
    "true_name",
    "pred_class",
    "pred_name",
    "count",
    "row_total",
    "row_fraction",
]

REASON_LONG_COLUMNS = [
    "h5_stem",
    "config_id",
    *PARAM_COLUMNS,
    "reason_code",
    "reason_name",
    "count",
    "fraction",
]

CLASS_FRACTION_COLUMNS = [
    "h5_stem",
    "config_id",
    *PARAM_COLUMNS,
    "class_code",
    "class_name",
    "pred_count",
    "ref_count",
    "pred_fraction",
    "ref_fraction",
    "abs_fraction_diff",
    "fraction_abs_diff_sum",
]


def build_base_config(output_root: Path) -> Dict[str, Any]:
    return {
        "OUTPUT_ROOT": Path(output_root),
        "CLASSIFIER_MODE": "height_density_pre_ground",
        "GROUND_SNR_MIN": 5.0,
        "GRID_RES_M": 10.0,
        "MIN_POINTS_PER_CELL": 1,
        "GROUND_CELL_PERCENTILE": 2,
        "DTM_IDW_K": 12,
        "DTM_IDW_POWER": 2.0,
        "DTM_MAX_SEARCH_RADIUS_M": 30.0,
        "GROUND_RESID_TOL_M": 1.0,
        "NOISE_HAG_MAX_M": 40.0,
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
        "WRITE_DIAGNOSTIC_PNG": False,
        "LAS_XYZ_SCALE_M": 0.001,
        "IDW_QUERY_CHUNK_SIZE": 500_000,
    }


def mode_preference_value(classifier_mode: str) -> int:
    order = {
        "height_only": 0,
        "height_density_post_ground": 1,
        "height_density_processed_only": 2,
        "height_density_pre_ground": 3,
        "height_no_below_noise": 4,
    }
    return order.get(str(classifier_mode), 99)


def config_signature(config: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(config.get(name) for name in PARAM_COLUMNS)


def parameter_values_from_config(config_record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "classifier_mode": config_record.get("CLASSIFIER_MODE", config_record.get("classifier_mode")),
        "GROUND_SNR_MIN": config_record.get("GROUND_SNR_MIN"),
        "GRID_RES_M": config_record.get("GRID_RES_M"),
        "GROUND_CELL_PERCENTILE": config_record.get("GROUND_CELL_PERCENTILE"),
        "MIN_POINTS_PER_CELL": config_record.get("MIN_POINTS_PER_CELL"),
        "DTM_IDW_K": config_record.get("DTM_IDW_K"),
        "DTM_IDW_POWER": config_record.get("DTM_IDW_POWER"),
        "DTM_MAX_SEARCH_RADIUS_M": config_record.get("DTM_MAX_SEARCH_RADIUS_M"),
        "GROUND_RESID_TOL_M": config_record.get("GROUND_RESID_TOL_M"),
        "NOISE_HAG_MAX_M": config_record.get("NOISE_HAG_MAX_M"),
        "LOCAL_FEATURE_RADIUS_M": config_record.get("LOCAL_FEATURE_RADIUS_M"),
        "NOISE_DENSITY_MAX_PTS_M3": config_record.get("NOISE_DENSITY_MAX_PTS_M3"),
    }


def build_parameter_grid(
    base_config: Dict[str, Any],
    stage_1_coarse: bool,
    focus_priority_configs: bool,
    max_configs: Optional[int],
    random_subset_configs: bool,
    random_seed: int,
) -> List[Dict[str, Any]]:
    if stage_1_coarse:
        grid = {
            "classifier_mode": [
                "height_only",
                "height_density_pre_ground",
                "height_density_post_ground",
                "height_density_processed_only",
                "height_no_below_noise",
            ],
            "GROUND_SNR_MIN": [3.0, 5.0, 7.0],
            "GRID_RES_M": [5.0, 10.0, 15.0],
            "GROUND_CELL_PERCENTILE": [2, 5, 10],
            "GROUND_RESID_TOL_M": [0.5, 1.0, 1.5, 2.0],
            "NOISE_HAG_MAX_M": [30.0, 40.0, 60.0],
            "LOCAL_FEATURE_RADIUS_M": [3.0, 5.0, 8.0],
            "NOISE_DENSITY_MAX_PTS_M3": [None, 0.01, 0.02, 0.04, 0.08],
        }
    else:
        grid = {
            "classifier_mode": [
                "height_density_post_ground",
                "height_density_processed_only",
                "height_density_pre_ground",
                "height_only",
            ],
            "GROUND_SNR_MIN": [4.0, 5.0, 6.0],
            "GRID_RES_M": [8.0, 10.0, 12.0],
            "GROUND_CELL_PERCENTILE": [2, 5, 8],
            "GROUND_RESID_TOL_M": [0.75, 1.0, 1.25, 1.5],
            "NOISE_HAG_MAX_M": [35.0, 40.0, 45.0],
            "LOCAL_FEATURE_RADIUS_M": [4.0, 5.0, 6.0],
            "NOISE_DENSITY_MAX_PTS_M3": [None, 0.02, 0.03, 0.04, 0.05],
        }

    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        record = dict(zip(keys, values))
        mode = str(record["classifier_mode"])
        density_threshold = record["NOISE_DENSITY_MAX_PTS_M3"]
        radius = record["LOCAL_FEATURE_RADIUS_M"]

        if mode == "height_only":
            density_threshold = None
            radius = None
        elif density_threshold is None:
            radius = None

        if focus_priority_configs and stage_1_coarse:
            if mode == "height_density_pre_ground":
                if record["GROUND_SNR_MIN"] != 5.0:
                    continue
                if record["GRID_RES_M"] != 10.0:
                    continue
                if record["GROUND_CELL_PERCENTILE"] not in [2, 5]:
                    continue
                if record["GROUND_RESID_TOL_M"] not in [0.5, 1.0, 1.5]:
                    continue
                if record["NOISE_HAG_MAX_M"] not in [30.0, 40.0]:
                    continue
                if radius not in [None, 5.0]:
                    continue
                if density_threshold not in [None, 0.02, 0.04, 0.08]:
                    continue

        config = dict(base_config)
        config.update({
            "CLASSIFIER_MODE": mode,
            "GROUND_SNR_MIN": float(record["GROUND_SNR_MIN"]),
            "GRID_RES_M": float(record["GRID_RES_M"]),
            "GROUND_CELL_PERCENTILE": int(record["GROUND_CELL_PERCENTILE"]),
            "GROUND_RESID_TOL_M": float(record["GROUND_RESID_TOL_M"]),
            "NOISE_HAG_MAX_M": float(record["NOISE_HAG_MAX_M"]),
            "LOCAL_FEATURE_RADIUS_M": None if radius is None else float(radius),
            "NOISE_DENSITY_MAX_PTS_M3": None if density_threshold is None else float(density_threshold),
        })
        key = config_signature({
            "classifier_mode": config["CLASSIFIER_MODE"],
            "GROUND_SNR_MIN": config["GROUND_SNR_MIN"],
            "GRID_RES_M": config["GRID_RES_M"],
            "GROUND_CELL_PERCENTILE": config["GROUND_CELL_PERCENTILE"],
            "MIN_POINTS_PER_CELL": config["MIN_POINTS_PER_CELL"],
            "DTM_IDW_K": config["DTM_IDW_K"],
            "DTM_IDW_POWER": config["DTM_IDW_POWER"],
            "DTM_MAX_SEARCH_RADIUS_M": config["DTM_MAX_SEARCH_RADIUS_M"],
            "GROUND_RESID_TOL_M": config["GROUND_RESID_TOL_M"],
            "NOISE_HAG_MAX_M": config["NOISE_HAG_MAX_M"],
            "LOCAL_FEATURE_RADIUS_M": config["LOCAL_FEATURE_RADIUS_M"],
            "NOISE_DENSITY_MAX_PTS_M3": config["NOISE_DENSITY_MAX_PTS_M3"],
        })
        if key in seen:
            continue
        seen.add(key)
        rows.append(config)

    if max_configs is not None and max_configs >= 0 and len(rows) > max_configs:
        if random_subset_configs:
            rng = np.random.default_rng(int(random_seed))
            chosen = np.sort(rng.choice(np.arange(len(rows)), size=int(max_configs), replace=False))
            rows = [rows[int(i)] for i in chosen]
        else:
            rows = rows[: int(max_configs)]

    for i, row in enumerate(rows, start=1):
        row["config_id"] = f"cfg_{i:05d}"
    return rows


def prepare_one_file_cache(
    input_pair: Dict[str, Path],
    base_config: Dict[str, Any],
    tested_radii: Sequence[Optional[float]],
) -> Dict[str, Any]:
    h5_path = Path(input_pair["h5_path"])
    reference_laz_path = Path(input_pair["reference_laz_path"])
    h5_stem = h5_path.stem

    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not reference_laz_path.exists():
        raise FileNotFoundError(reference_laz_path)

    casals = read_casals_h5_refh_points(h5_path)
    reference = read_reference_labels(reference_laz_path)
    crs_info = infer_or_choose_projected_crs(casals, reference)
    projected_crs = crs_info["crs"]
    x, y = project_lonlat_to_xy(casals["lon"], casals["lat"], projected_crs)
    refh_snr = np.asarray(casals["fields"]["refh_snr"], dtype=np.float64)

    alignment = align_prediction_to_reference(
        prediction={
            "point_index": casals["point_index"],
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
            "longitude": np.asarray(casals["lon"], dtype=np.float64),
            "latitude": np.asarray(casals["lat"], dtype=np.float64),
        },
        reference=reference,
        config=base_config,
    )
    eval_gt_class = map_reference_labels_to_baseline_classes(alignment["eval_gt_class_raw"])

    density_cache: Dict[float, Dict[str, np.ndarray]] = {}
    for radius_m in sorted({float(v) for v in tested_radii if v is not None}):
        density_config = dict(base_config)
        density_config["LOCAL_FEATURE_RADIUS_M"] = float(radius_m)
        density_cache[float(radius_m)] = compute_local_point_density(
            x=np.asarray(x, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            z=np.asarray(casals["z"], dtype=np.float64),
            config=density_config,
        )

    return {
        "h5_stem": h5_stem,
        "input_pair": input_pair,
        "casals": casals,
        "reference": reference,
        "crs_info": crs_info,
        "projected_crs": projected_crs,
        "x": np.asarray(x, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "refh_snr": refh_snr,
        "alignment": alignment,
        "eval_gt_class": np.asarray(eval_gt_class, dtype=np.uint8),
        "density_cache": density_cache,
        "dtm_cache": {},
        "high_conf_available": (
            alignment["reference_nearest3dep_dist_m"] is not None
            and alignment["reference_class_vote_ratio"] is not None
        ),
    }


def dtm_cache_key(config: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        float(config["GROUND_SNR_MIN"]),
        float(config["GRID_RES_M"]),
        int(config["GROUND_CELL_PERCENTILE"]),
        int(config["MIN_POINTS_PER_CELL"]),
        int(config["DTM_IDW_K"]),
        float(config["DTM_IDW_POWER"]),
        float(config["DTM_MAX_SEARCH_RADIUS_M"]),
    )


def get_or_build_dtm(
    cached_file: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    key = dtm_cache_key(config)
    if key in cached_file["dtm_cache"]:
        return cached_file["dtm_cache"][key]

    ground_grid = build_ground_grid(
        x=cached_file["x"],
        y=cached_file["y"],
        z=np.asarray(cached_file["casals"]["z"], dtype=np.float64),
        refh_snr=np.asarray(cached_file["refh_snr"], dtype=np.float64),
        config=config,
    )
    local_ground_z_m, dtm_sample_valid = sample_ground_grid_idw(
        x=cached_file["x"],
        y=cached_file["y"],
        ground_grid=ground_grid,
        config=config,
    )
    payload = {
        "local_ground_z_m": np.asarray(local_ground_z_m, dtype=np.float64),
        "dtm_sample_valid": np.asarray(dtm_sample_valid, dtype=np.uint8),
        "ground_grid_summary": {
            "support_count": int(ground_grid["support_count"]),
            "valid_cell_count": int(ground_grid["valid_cell_count"]),
        },
    }
    cached_file["dtm_cache"][key] = payload
    return payload


def classify_points_debug(
    z: np.ndarray,
    local_ground_z_m: np.ndarray,
    dtm_sample_valid: np.ndarray,
    point_density_pts_m3: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    mode = str(config["CLASSIFIER_MODE"])
    density_threshold = config.get("NOISE_DENSITY_MAX_PTS_M3")
    max_hag = float(config["NOISE_HAG_MAX_M"])

    if (
        mode == "height_density_pre_ground"
        and density_threshold is not None
        and np.isclose(max_hag, 40.0)
    ):
        return classify_points_baseline(
            z=np.asarray(z, dtype=np.float64),
            local_ground_z_m=np.asarray(local_ground_z_m, dtype=np.float64),
            dtm_sample_valid=np.asarray(dtm_sample_valid, dtype=np.uint8),
            point_density_pts_m3=np.asarray(point_density_pts_m3, dtype=np.float64),
            config=config,
        )

    z = np.asarray(z, dtype=np.float64)
    local_ground_z_m = np.asarray(local_ground_z_m, dtype=np.float64)
    dtm_sample_valid = np.asarray(dtm_sample_valid, dtype=np.uint8)
    point_density_pts_m3 = np.asarray(point_density_pts_m3, dtype=np.float64)

    hag = np.full(z.shape[0], np.nan, dtype=np.float64)
    valid_dtm = np.asarray(dtm_sample_valid, dtype=bool)
    hag[valid_dtm] = z[valid_dtm] - local_ground_z_m[valid_dtm]

    pred = np.full(z.shape[0], 7, dtype=np.uint8)
    reason = np.full(z.shape[0], 0, dtype=np.uint8)

    invalid_dtm = ~valid_dtm
    pred[invalid_dtm] = 7
    reason[invalid_dtm] = 2

    valid_hag = valid_dtm & np.isfinite(hag)
    nonfinite_hag = valid_dtm & ~np.isfinite(hag)
    pred[nonfinite_hag] = 7
    reason[nonfinite_hag] = 2

    density_enabled = density_threshold is not None
    if density_enabled:
        low_density = (
            valid_hag
            & np.isfinite(point_density_pts_m3)
            & (point_density_pts_m3 < float(density_threshold))
        )
    else:
        low_density = np.zeros(z.shape[0], dtype=bool)

    ground = valid_hag & (np.abs(hag) <= float(config["GROUND_RESID_TOL_M"]))
    below_ground = valid_hag & (hag < 0.0)
    above_max = valid_hag & (hag > max_hag)

    if mode == "height_only":
        low_density = np.zeros(z.shape[0], dtype=bool)
        ground_final = ground
        below_final = below_ground & ~ground_final
        above_final = above_max & ~(ground_final | below_final)
        processed_final = valid_hag & ~(ground_final | below_final | above_final)
        density_reason_code = None
    elif mode == "height_density_pre_ground":
        ground_final = ground & ~low_density
        below_final = below_ground & ~(low_density | ground_final)
        above_final = above_max & ~(low_density | ground_final | below_final)
        processed_final = valid_hag & ~(low_density | ground_final | below_final | above_final)
        density_reason_code = 6
    elif mode == "height_density_post_ground":
        ground_final = ground
        low_density = low_density & ~ground_final
        below_final = below_ground & ~(ground_final | low_density)
        above_final = above_max & ~(ground_final | low_density | below_final)
        processed_final = valid_hag & ~(ground_final | low_density | below_final | above_final)
        density_reason_code = 8
    elif mode == "height_density_processed_only":
        ground_final = ground
        below_final = below_ground & ~ground_final
        above_final = above_max & ~(ground_final | below_final)
        processed_candidates = valid_hag & ~(ground_final | below_final | above_final)
        low_density = low_density & processed_candidates
        processed_final = processed_candidates & ~low_density
        density_reason_code = 9
    elif mode == "height_no_below_noise":
        low_density = np.zeros(z.shape[0], dtype=bool)
        ground_final = ground
        below_final = np.zeros(z.shape[0], dtype=bool)
        above_final = above_max & ~ground_final
        processed_final = valid_hag & ~(ground_final | above_final)
        density_reason_code = None
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

    if mode == "height_no_below_noise":
        negative_processed = processed_final & (hag < 0.0)
        reason[negative_processed] = 7

    return {
        "pred_class_baseline": pred.astype(np.uint8),
        "classification_reason": reason.astype(np.uint8),
        "height_above_ground_m": hag.astype(np.float64),
    }


def report_rows_to_metrics(report_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_class = {
        int(row["class_code"]): row
        for row in report_rows
        if row.get("class_code") not in [None, ""]
    }
    out: Dict[str, Any] = {}
    for cls in LABEL_ORDER:
        row = by_class.get(cls, {})
        out[f"class_{cls}_precision"] = row.get("precision")
        out[f"class_{cls}_recall"] = row.get("recall")
        out[f"class_{cls}_f1"] = row.get("f1_score")
        out[f"class_{cls}_support"] = row.get("support")
    return out


def summarize_class_fractions(
    h5_stem: str,
    config_record: Dict[str, Any],
    summary_row: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    point_count = max(int(summary_row["point_count"]), 1)
    matched_count = max(int(summary_row["matched_count"]), 1)
    pred_fractions: Dict[int, float] = {}
    ref_fractions: Dict[int, float] = {}
    rows: List[Dict[str, Any]] = []
    for cls in LABEL_ORDER:
        pred_count = int(summary_row.get(f"pred_count_{cls}", 0))
        ref_count = int(summary_row.get(f"ref_count_{cls}", 0))
        pred_fraction = float(pred_count / point_count)
        ref_fraction = float(ref_count / matched_count)
        pred_fractions[cls] = pred_fraction
        ref_fractions[cls] = ref_fraction

    fraction_abs_diff_sum = float(sum(abs(pred_fractions[cls] - ref_fractions[cls]) for cls in LABEL_ORDER))
    for cls in LABEL_ORDER:
        rows.append({
            "h5_stem": h5_stem,
            "config_id": config_record["config_id"],
            **parameter_values_from_config(config_record),
            "class_code": cls,
            "class_name": CLASS_NAME_MAP[cls],
            "pred_count": int(summary_row.get(f"pred_count_{cls}", 0)),
            "ref_count": int(summary_row.get(f"ref_count_{cls}", 0)),
            "pred_fraction": pred_fractions[cls],
            "ref_fraction": ref_fractions[cls],
            "abs_fraction_diff": abs(pred_fractions[cls] - ref_fractions[cls]),
            "fraction_abs_diff_sum": fraction_abs_diff_sum,
        })

    return rows, {
        f"pred_fraction_{cls}": pred_fractions[cls] for cls in LABEL_ORDER
    } | {
        f"ref_fraction_{cls}": ref_fractions[cls] for cls in LABEL_ORDER
    } | {
        "fraction_abs_diff_sum": fraction_abs_diff_sum
    }


def compute_objective_score(metric_row: Dict[str, Any]) -> Optional[float]:
    required = [
        "macro_f1",
        "class_1_f1",
        "class_2_f1",
        "class_7_f1",
        "weighted_f1",
        "class_2_recall",
        "class_7_recall",
        "fraction_abs_diff_sum",
    ]
    for key in required:
        value = metric_row.get(key)
        if value is None or not np.isfinite(value):
            return None

    objective_all = (
        0.35 * float(metric_row["macro_f1"])
        + 0.20 * float(metric_row["class_2_f1"])
        + 0.20 * float(metric_row["class_7_f1"])
        + 0.15 * float(metric_row["class_1_f1"])
        + 0.10 * float(metric_row["weighted_f1"])
    )

    penalty = 0.0
    if float(metric_row["class_2_recall"]) < 0.90:
        penalty += (0.90 - float(metric_row["class_2_recall"])) * 0.5
    if float(metric_row["class_7_recall"]) < 0.50:
        penalty += (0.50 - float(metric_row["class_7_recall"])) * 0.4
    if float(metric_row["fraction_abs_diff_sum"]) > 0.30:
        penalty += (float(metric_row["fraction_abs_diff_sum"]) - 0.30) * 0.2

    return float(objective_all - penalty)


def summarize_reason_counts(
    h5_stem: str,
    config_record: Dict[str, Any],
    classification_reason: np.ndarray,
) -> List[Dict[str, Any]]:
    counts = collect_class_counts(np.asarray(classification_reason, dtype=np.uint8))
    n = max(int(np.asarray(classification_reason).size), 1)
    rows: List[Dict[str, Any]] = []
    for code in sorted(DEBUG_REASON_MAP):
        rows.append({
            "h5_stem": h5_stem,
            "config_id": config_record["config_id"],
            **parameter_values_from_config(config_record),
            "reason_code": int(code),
            "reason_name": DEBUG_REASON_MAP[code],
            "count": int(counts.get(code, 0)),
            "fraction": float(counts.get(code, 0) / n),
        })
    return rows


def compute_metrics_table(
    h5_stem: str,
    config_record: Dict[str, Any],
    summary_row: Dict[str, Any],
    evaluation_metrics: Dict[str, Any],
    high_conf_available: bool,
    warning: str = "",
) -> List[Dict[str, Any]]:
    class_fraction_rows, fraction_summary = summarize_class_fractions(
        h5_stem=h5_stem,
        config_record=config_record,
        summary_row=summary_row,
    )
    del class_fraction_rows

    subset_results = evaluation_metrics["subset_results"]
    rows: List[Dict[str, Any]] = []
    for subset_name in ["all_matched", "high_confidence_reference"]:
        subset_result = subset_results.get(subset_name)
        base_row = {
            "h5_stem": h5_stem,
            "config_id": config_record["config_id"],
            "subset_name": subset_name,
            "status": "ok",
            "warning": warning,
            **parameter_values_from_config(config_record),
            "point_count": int(summary_row["point_count"]),
            "matched_count": int(summary_row["matched_count"]),
            "unmatched_count": int(summary_row["unmatched_count"]),
            "ground_support_candidate_count": int(summary_row["ground_support_candidate_count"]),
            "valid_dtm_cell_count": int(summary_row["valid_dtm_cell_count"]),
            "dtm_invalid_count": int(summary_row["dtm_invalid_count"]),
            "pred_count_1": int(summary_row["pred_count_1"]),
            "pred_count_2": int(summary_row["pred_count_2"]),
            "pred_count_7": int(summary_row["pred_count_7"]),
            "ref_count_1": int(summary_row["ref_count_1"]),
            "ref_count_2": int(summary_row["ref_count_2"]),
            "ref_count_7": int(summary_row["ref_count_7"]),
            **fraction_summary,
        }

        if subset_result is None:
            base_row["status"] = "unavailable" if not high_conf_available else "not_computed"
            for metric_name in RESULT_COLUMNS:
                if metric_name not in base_row and metric_name not in ["h5_stem", "config_id", "subset_name", "status", "warning", *PARAM_COLUMNS]:
                    base_row[metric_name] = np.nan
            rows.append(base_row)
            continue

        if subset_result.get("status") != "ok":
            base_row["status"] = subset_result.get("status", "unavailable")
            for metric_name in [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
                "class_1_precision",
                "class_1_recall",
                "class_1_f1",
                "class_1_support",
                "class_2_precision",
                "class_2_recall",
                "class_2_f1",
                "class_2_support",
                "class_7_precision",
                "class_7_recall",
                "class_7_f1",
                "class_7_support",
                "objective_score",
            ]:
                base_row[metric_name] = np.nan
            rows.append(base_row)
            continue

        class_metrics = report_rows_to_metrics(subset_result["report_rows"])
        metric_row = {
            **base_row,
            "accuracy": float(subset_result["accuracy"]),
            "macro_precision": float(subset_result["macro_precision"]),
            "macro_recall": float(subset_result["macro_recall"]),
            "macro_f1": float(subset_result["macro_f1"]),
            "weighted_precision": float(subset_result["weighted_precision"]),
            "weighted_recall": float(subset_result["weighted_recall"]),
            "weighted_f1": float(subset_result["weighted_f1"]),
            **class_metrics,
        }
        metric_row["objective_score"] = compute_objective_score(metric_row)
        rows.append(metric_row)

    return rows


def summarize_confusion_rows(
    h5_stem: str,
    config_record: Dict[str, Any],
    evaluation_metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for subset_name, subset_result in evaluation_metrics["subset_results"].items():
        status = subset_result.get("status")
        if status != "ok":
            continue
        for wide_row in subset_result["confusion_rows"]:
            row_total = max(int(wide_row["row_total"]), 1)
            for pred_class in LABEL_ORDER:
                count = int(wide_row[f"pred_{pred_class}"])
                rows.append({
                    "h5_stem": h5_stem,
                    "config_id": config_record["config_id"],
                    "subset_name": subset_name,
                    "status": "ok",
                    "true_class": int(wide_row["true_class"]),
                    "true_name": wide_row["true_name"],
                    "pred_class": int(pred_class),
                    "pred_name": CLASS_NAME_MAP[pred_class],
                    "count": count,
                    "row_total": int(wide_row["row_total"]),
                    "row_fraction": float(count / row_total),
                })
    return rows


def run_one_config_on_cached_file(
    cached_file: Dict[str, Any],
    config_record: Dict[str, Any],
) -> Dict[str, Any]:
    config = dict(config_record)
    dtm_payload = get_or_build_dtm(cached_file, config)

    radius = config.get("LOCAL_FEATURE_RADIUS_M")
    if radius is None:
        local_neighbor_count = np.zeros(cached_file["x"].shape[0], dtype=np.uint32)
        point_density_pts_m3 = np.full(cached_file["x"].shape[0], np.nan, dtype=np.float64)
    else:
        density_result = cached_file["density_cache"][float(radius)]
        local_neighbor_count = np.asarray(density_result["local_neighbor_count"], dtype=np.uint32)
        point_density_pts_m3 = np.asarray(density_result["point_density_pts_m3"], dtype=np.float64)

    classification_result = classify_points_debug(
        z=np.asarray(cached_file["casals"]["z"], dtype=np.float64),
        local_ground_z_m=np.asarray(dtm_payload["local_ground_z_m"], dtype=np.float64),
        dtm_sample_valid=np.asarray(dtm_payload["dtm_sample_valid"], dtype=np.uint8),
        point_density_pts_m3=point_density_pts_m3,
        config=config,
    )
    pred_class = np.asarray(classification_result["pred_class_baseline"], dtype=np.uint8)
    classification_reason = np.asarray(classification_result["classification_reason"], dtype=np.uint8)
    height_above_ground_m = np.asarray(classification_result["height_above_ground_m"], dtype=np.float64)

    evaluation_metrics = compute_evaluation_metrics(
        pred_class_baseline=pred_class,
        eval_gt_class=np.asarray(cached_file["eval_gt_class"], dtype=np.uint8),
        eval_match_valid=np.asarray(cached_file["alignment"]["eval_match_valid"], dtype=np.uint8),
        dtm_sample_valid=np.asarray(dtm_payload["dtm_sample_valid"], dtype=np.uint8),
        reference_transfer_status=cached_file["alignment"]["reference_transfer_status"],
        reference_nearest3dep_dist_m=cached_file["alignment"]["reference_nearest3dep_dist_m"],
        reference_class_vote_ratio=cached_file["alignment"]["reference_class_vote_ratio"],
        config=config,
    )

    summary_row = build_classification_summary_row(
        h5_stem=cached_file["h5_stem"],
        pred_class_baseline=pred_class,
        eval_gt_class=np.asarray(cached_file["eval_gt_class"], dtype=np.uint8),
        eval_match_valid=np.asarray(cached_file["alignment"]["eval_match_valid"], dtype=np.uint8),
        dtm_sample_valid=np.asarray(dtm_payload["dtm_sample_valid"], dtype=np.uint8),
        ground_support_candidate_count=int(dtm_payload["ground_grid_summary"]["support_count"]),
        valid_dtm_cell_count=int(dtm_payload["ground_grid_summary"]["valid_cell_count"]),
        height_above_ground_m=height_above_ground_m,
        refh_snr=np.asarray(cached_file["refh_snr"], dtype=np.float64),
        point_density_pts_m3=point_density_pts_m3,
        classification_reason=classification_reason,
    )
    metrics_rows = compute_metrics_table(
        h5_stem=cached_file["h5_stem"],
        config_record=config_record,
        summary_row=summary_row,
        evaluation_metrics=evaluation_metrics,
        high_conf_available=bool(cached_file["high_conf_available"]),
    )
    class_fraction_rows, _ = summarize_class_fractions(
        h5_stem=cached_file["h5_stem"],
        config_record=config_record,
        summary_row=summary_row,
    )
    reason_rows = summarize_reason_counts(
        h5_stem=cached_file["h5_stem"],
        config_record=config_record,
        classification_reason=classification_reason,
    )
    confusion_rows = summarize_confusion_rows(
        h5_stem=cached_file["h5_stem"],
        config_record=config_record,
        evaluation_metrics=evaluation_metrics,
    )

    return {
        "status": "success",
        "metrics_rows": metrics_rows,
        "reason_rows": reason_rows,
        "confusion_rows": confusion_rows,
        "class_fraction_rows": class_fraction_rows,
    }


def run_parameter_sweep(
    cached_files: Sequence[Dict[str, Any]],
    parameter_grid: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    result_rows: List[Dict[str, Any]] = []
    reason_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []
    class_fraction_rows: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []

    total = len(cached_files) * len(parameter_grid)
    progress = 0
    for config_idx, config_record in enumerate(parameter_grid, start=1):
        print("-" * 100)
        print(f"Config {config_idx}/{len(parameter_grid)}: {format_config_brief(config_record)}")
        for cached_file in cached_files:
            progress += 1
            try:
                run_result = run_one_config_on_cached_file(
                    cached_file=cached_file,
                    config_record=config_record,
                )
                result_rows.extend(run_result["metrics_rows"])
                reason_rows.extend(run_result["reason_rows"])
                confusion_rows.extend(run_result["confusion_rows"])
                class_fraction_rows.extend(run_result["class_fraction_rows"])

                all_matched_row = next(
                    row for row in run_result["metrics_rows"] if row["subset_name"] == "all_matched"
                )
                high_conf_row = next(
                    row for row in run_result["metrics_rows"] if row["subset_name"] == "high_confidence_reference"
                )
                high_conf_text = (
                    f"high_conf_obj={float(high_conf_row['objective_score']):.4f} "
                    f"high_conf_macro_f1={float(high_conf_row['macro_f1']):.4f}"
                    if high_conf_row["status"] == "ok"
                    else f"high_conf_status={high_conf_row['status']}"
                )
                print(
                    f"[{progress}/{total}] "
                    f"file={cached_file['h5_stem']} "
                    f"mode={config_record['CLASSIFIER_MODE']} "
                    f"macro_f1={float(all_matched_row['macro_f1']):.4f} "
                    f"ground_f1={float(all_matched_row['class_2_f1']):.4f} "
                    f"noise_f1={float(all_matched_row['class_7_f1']):.4f} "
                    f"ground_recall={float(all_matched_row['class_2_recall']):.4f} "
                    f"noise_recall={float(all_matched_row['class_7_recall']):.4f} "
                    f"frac_diff={float(all_matched_row['fraction_abs_diff_sum']):.4f} "
                    f"obj={float(all_matched_row['objective_score']):.4f} "
                    f"{high_conf_text}"
                )
            except Exception as exc:
                warning = f"{type(exc).__name__}: {exc}"
                failed_runs.append({
                    "h5_stem": cached_file["h5_stem"],
                    "config_id": config_record["config_id"],
                    "classifier_mode": config_record["CLASSIFIER_MODE"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(
                    f"[{progress}/{total}] "
                    f"file={cached_file['h5_stem']} "
                    f"mode={config_record['CLASSIFIER_MODE']} "
                    f"FAILED {warning}"
                )

                empty_row = {
                    "h5_stem": cached_file["h5_stem"],
                    "config_id": config_record["config_id"],
                    "warning": warning,
                    **parameter_values_from_config(config_record),
                }
                for subset_name in ["all_matched", "high_confidence_reference"]:
                    result_rows.append({
                        **empty_row,
                        "subset_name": subset_name,
                        "status": "failed",
                        **{col: np.nan for col in RESULT_COLUMNS if col not in empty_row and col not in ["h5_stem", "config_id", "subset_name", "status", "warning", *PARAM_COLUMNS]},
                    })

    return {
        "result_rows": result_rows,
        "reason_rows": reason_rows,
        "confusion_rows": confusion_rows,
        "class_fraction_rows": class_fraction_rows,
        "failed_runs": failed_runs,
    }


def rank_and_select_best_configs(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset_df = results_df[
        (results_df["subset_name"] == "all_matched")
        & (results_df["status"] == "ok")
    ].copy()
    if subset_df.empty:
        return (
            pd.DataFrame(columns=RANKED_COLUMNS),
            pd.DataFrame(columns=BEST_BY_FILE_COLUMNS),
        )

    grouped_rows: List[Dict[str, Any]] = []
    for config_id, group in subset_df.groupby("config_id", sort=False):
        first = group.iloc[0]
        grouped_rows.append({
            "h5_stem": "__all__",
            "config_id": config_id,
            **{name: first[name] for name in PARAM_COLUMNS},
            "n_files_success": int(group["h5_stem"].nunique()),
            "mean_objective_score": float(group["objective_score"].mean()),
            "min_objective_score": float(group["objective_score"].min()),
            "std_objective_score": float(group["objective_score"].std(ddof=0)),
            "mean_macro_f1": float(group["macro_f1"].mean()),
            "mean_weighted_f1": float(group["weighted_f1"].mean()),
            "mean_class_1_f1": float(group["class_1_f1"].mean()),
            "mean_class_2_f1": float(group["class_2_f1"].mean()),
            "mean_class_7_f1": float(group["class_7_f1"].mean()),
            "mean_class_2_recall": float(group["class_2_recall"].mean()),
            "mean_class_7_recall": float(group["class_7_recall"].mean()),
            "mean_fraction_abs_diff_sum": float(group["fraction_abs_diff_sum"].mean()),
            "_mode_preference": mode_preference_value(str(first["classifier_mode"])),
        })

    ranked_df = pd.DataFrame(grouped_rows)
    ranked_df = ranked_df.sort_values(
        by=[
            "mean_objective_score",
            "min_objective_score",
            "mean_macro_f1",
            "mean_class_7_recall",
            "_mode_preference",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked_df["rank"] = np.arange(1, len(ranked_df) + 1, dtype=np.int64)
    ranked_df = ranked_df.drop(columns=["_mode_preference"])
    ranked_df = ranked_df[RANKED_COLUMNS]

    best_by_file_rows: List[Dict[str, Any]] = []
    global_rank_map = {
        row["config_id"]: int(row["rank"])
        for _, row in ranked_df.iterrows()
    }
    for h5_stem, group in subset_df.groupby("h5_stem", sort=False):
        group_sorted = group.sort_values(
            by=[
                "objective_score",
                "macro_f1",
                "class_7_recall",
                "fraction_abs_diff_sum",
            ],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        best = group_sorted.iloc[0]
        best_by_file_rows.append({
            "h5_stem": h5_stem,
            "config_id": best["config_id"],
            "rank_within_file": 1,
            "global_rank": int(global_rank_map.get(best["config_id"], -1)),
            **{name: best[name] for name in PARAM_COLUMNS},
            "objective_score": float(best["objective_score"]),
            "macro_f1": float(best["macro_f1"]),
            "weighted_f1": float(best["weighted_f1"]),
            "class_2_recall": float(best["class_2_recall"]),
            "class_7_recall": float(best["class_7_recall"]),
            "fraction_abs_diff_sum": float(best["fraction_abs_diff_sum"]),
            "status": best["status"],
        })

    best_by_file_df = pd.DataFrame(best_by_file_rows, columns=BEST_BY_FILE_COLUMNS)
    return ranked_df, best_by_file_df


def build_ablation_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    subset_df = results_df[
        (results_df["subset_name"] == "all_matched")
        & (results_df["status"] == "ok")
    ].copy()
    if subset_df.empty:
        return pd.DataFrame(columns=ABLATION_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for h5_stem, df_h5 in [("__all__", subset_df), *subset_df.groupby("h5_stem", sort=False)]:
        for mode, group in df_h5.groupby("classifier_mode", sort=False):
            rows.append({
                "h5_stem": h5_stem,
                "config_id": "__all__",
                "classifier_mode": mode,
                "n_runs": int(group.shape[0]),
                "mean_objective_score": float(group["objective_score"].mean()),
                "median_objective_score": float(group["objective_score"].median()),
                "mean_macro_f1": float(group["macro_f1"].mean()),
                "median_macro_f1": float(group["macro_f1"].median()),
                "mean_class_2_recall": float(group["class_2_recall"].mean()),
                "mean_class_7_recall": float(group["class_7_recall"].mean()),
            })
    return pd.DataFrame(rows, columns=ABLATION_COLUMNS)


def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def plot_objective_vs_rank(ranked_df: pd.DataFrame, output_path: Path) -> None:
    if ranked_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ranked_df["rank"], ranked_df["mean_objective_score"], marker="o", linewidth=1.5)
    ax.set_xlabel("Config rank")
    ax.set_ylabel("Mean objective score")
    ax.set_title("Objective vs config rank")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_macro_f1_by_mode(results_df: pd.DataFrame, output_path: Path) -> None:
    subset_df = results_df[
        (results_df["subset_name"] == "all_matched")
        & (results_df["status"] == "ok")
    ].copy()
    if subset_df.empty:
        return
    modes = list(dict.fromkeys(subset_df["classifier_mode"].tolist()))
    data = [subset_df.loc[subset_df["classifier_mode"] == mode, "macro_f1"].to_numpy(dtype=np.float64) for mode in modes]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=modes, showfliers=False)
    for i, mode in enumerate(modes, start=1):
        y = subset_df.loc[subset_df["classifier_mode"] == mode, "macro_f1"].to_numpy(dtype=np.float64)
        x = np.full(y.shape[0], i, dtype=np.float64)
        ax.scatter(x, y, s=10, alpha=0.35)
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 by ablation mode")
    ax.grid(True, axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_noise_recall_by_density(results_df: pd.DataFrame, output_path: Path) -> None:
    subset_df = results_df[
        (results_df["subset_name"] == "all_matched")
        & (results_df["status"] == "ok")
        & results_df["NOISE_DENSITY_MAX_PTS_M3"].notna()
    ].copy()
    if subset_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    color_map = {
        "height_density_pre_ground": "#1f77b4",
        "height_density_post_ground": "#d62728",
        "height_density_processed_only": "#2ca02c",
        "height_only": "#7f7f7f",
        "height_no_below_noise": "#9467bd",
    }
    for mode, group in subset_df.groupby("classifier_mode", sort=False):
        ax.scatter(
            group["NOISE_DENSITY_MAX_PTS_M3"],
            group["class_7_recall"],
            s=24,
            alpha=0.65,
            label=mode,
            color=color_map.get(mode),
        )
    ax.set_xlabel("NOISE_DENSITY_MAX_PTS_M3")
    ax.set_ylabel("Noise recall (class 7)")
    ax.set_title("Noise recall by density threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ground_recall_by_tol(results_df: pd.DataFrame, output_path: Path) -> None:
    subset_df = results_df[
        (results_df["subset_name"] == "all_matched")
        & (results_df["status"] == "ok")
    ].copy()
    if subset_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(subset_df["GROUND_RESID_TOL_M"], subset_df["class_2_recall"], s=24, alpha=0.55)
    ax.set_xlabel("GROUND_RESID_TOL_M")
    ax.set_ylabel("Ground recall (class 2)")
    ax.set_title("Ground recall by ground residual tolerance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_best_confusion_matrix(
    confusion_rows_df: pd.DataFrame,
    h5_stem: str,
    config_id: str,
    output_path: Path,
) -> None:
    df = confusion_rows_df[
        (confusion_rows_df["h5_stem"] == h5_stem)
        & (confusion_rows_df["config_id"] == config_id)
        & (confusion_rows_df["subset_name"] == "all_matched")
        & (confusion_rows_df["status"] == "ok")
    ].copy()
    if df.empty:
        return

    cm = np.zeros((len(LABEL_ORDER), len(LABEL_ORDER)), dtype=np.int64)
    for i, true_cls in enumerate(LABEL_ORDER):
        for j, pred_cls in enumerate(LABEL_ORDER):
            match = df[(df["true_class"] == true_cls) & (df["pred_class"] == pred_cls)]
            if not match.empty:
                cm[i, j] = int(match.iloc[0]["count"])

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.divide(cm, row_sums, where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, mat, title, fmt in [
        (axes[0], cm, "Raw counts", "d"),
        (axes[1], cm_norm, "Row-normalized", ".2f"),
    ]:
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(LABEL_ORDER)))
        ax.set_yticks(range(len(LABEL_ORDER)))
        ax.set_xticklabels([CLASS_NAME_MAP[v] for v in LABEL_ORDER], rotation=20, ha="right")
        ax.set_yticklabels([CLASS_NAME_MAP[v] for v in LABEL_ORDER])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                text = f"{int(value)}" if fmt == "d" else f"{float(value):.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Best confusion matrix: {h5_stem} ({config_id})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_best_class_fraction(
    class_fraction_df: pd.DataFrame,
    h5_stem: str,
    config_id: str,
    output_path: Path,
) -> None:
    df = class_fraction_df[
        (class_fraction_df["h5_stem"] == h5_stem)
        & (class_fraction_df["config_id"] == config_id)
    ].copy()
    if df.empty:
        return

    df = df.sort_values("class_code")
    x = np.arange(df.shape[0], dtype=np.float64)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2.0, df["pred_fraction"], width=width, label="Predicted")
    ax.bar(x + width / 2.0, df["ref_fraction"], width=width, label="Reference")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class_name"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("Class fraction")
    ax.set_title(f"Predicted vs reference class fractions: {h5_stem} ({config_id})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_debug_outputs(
    output_root: Path,
    results_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    best_by_file_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    confusion_rows_df: pd.DataFrame,
    reason_rows_df: pd.DataFrame,
    class_fraction_df: pd.DataFrame,
    failed_runs: List[Dict[str, Any]],
    debug_metadata: Dict[str, Any],
    write_png: bool,
) -> None:
    ensure_dir(output_root)

    parameter_sweep_results_path = output_root / "parameter_sweep_results.csv"
    ranked_path = output_root / "parameter_sweep_results_ranked.csv"
    best_config_path = output_root / "best_config.json"
    best_by_file_path = output_root / "best_configs_by_file.csv"
    ablation_path = output_root / "ablation_summary.csv"
    confusion_path = output_root / "confusion_matrices_long.csv"
    reason_path = output_root / "reason_counts_long.csv"
    class_fraction_path = output_root / "class_fraction_diagnostics.csv"
    failed_path = output_root / "failed_runs.json"
    metadata_path = output_root / "debug_run_metadata.json"

    results_df = results_df if not results_df.empty else empty_frame(RESULT_COLUMNS)
    ranked_df = ranked_df if not ranked_df.empty else empty_frame(RANKED_COLUMNS)
    best_by_file_df = best_by_file_df if not best_by_file_df.empty else empty_frame(BEST_BY_FILE_COLUMNS)
    ablation_df = ablation_df if not ablation_df.empty else empty_frame(ABLATION_COLUMNS)
    confusion_rows_df = confusion_rows_df if not confusion_rows_df.empty else empty_frame(CONFUSION_LONG_COLUMNS)
    reason_rows_df = reason_rows_df if not reason_rows_df.empty else empty_frame(REASON_LONG_COLUMNS)
    class_fraction_df = class_fraction_df if not class_fraction_df.empty else empty_frame(CLASS_FRACTION_COLUMNS)

    results_df[RESULT_COLUMNS].to_csv(parameter_sweep_results_path, index=False)
    ranked_df[RANKED_COLUMNS].to_csv(ranked_path, index=False)
    best_by_file_df[BEST_BY_FILE_COLUMNS].to_csv(best_by_file_path, index=False)
    ablation_df[ABLATION_COLUMNS].to_csv(ablation_path, index=False)
    confusion_rows_df[CONFUSION_LONG_COLUMNS].to_csv(confusion_path, index=False)
    reason_rows_df[REASON_LONG_COLUMNS].to_csv(reason_path, index=False)
    class_fraction_df[CLASS_FRACTION_COLUMNS].to_csv(class_fraction_path, index=False)

    best_payload: Dict[str, Any] = {
        "selected_by": "mean_objective_then_min_objective",
        "tested_files": [],
        "per_file_metrics": [],
        "scientific_notes": [
            "The transferred 3DEP labels are pseudo-reference labels, not absolute ground truth.",
            "The selected parameters are optimized against the current pseudo-reference and must be validated visually.",
        ],
    }
    if not ranked_df.empty:
        best_row = ranked_df.iloc[0]
        best_payload["best_config"] = {
            key: json_safe(best_row[key])
            for key in ["config_id", *PARAM_COLUMNS, "rank", "n_files_success", "mean_objective_score", "min_objective_score"]
        }
        best_result_rows = results_df[results_df["config_id"] == best_row["config_id"]].copy()
        best_payload["tested_files"] = sorted(best_result_rows["h5_stem"].dropna().unique().tolist())
        best_payload["per_file_metrics"] = best_result_rows.to_dict(orient="records")
    safe_json_dump(best_payload, best_config_path)
    safe_json_dump({"failed_runs": failed_runs}, failed_path)

    debug_metadata = dict(debug_metadata)
    debug_metadata["outputs"] = {
        "parameter_sweep_results_csv": str(parameter_sweep_results_path),
        "parameter_sweep_results_ranked_csv": str(ranked_path),
        "best_config_json": str(best_config_path),
        "best_configs_by_file_csv": str(best_by_file_path),
        "ablation_summary_csv": str(ablation_path),
        "confusion_matrices_long_csv": str(confusion_path),
        "reason_counts_long_csv": str(reason_path),
        "class_fraction_diagnostics_csv": str(class_fraction_path),
        "failed_runs_json": str(failed_path),
        "debug_run_metadata_json": str(metadata_path),
    }
    safe_json_dump(debug_metadata, metadata_path)

    if not write_png:
        return

    plot_objective_vs_rank(ranked_df, output_root / "objective_vs_config_rank.png")
    plot_macro_f1_by_mode(results_df, output_root / "macro_f1_by_ablation_mode.png")
    plot_noise_recall_by_density(results_df, output_root / "noise_recall_by_density_threshold.png")
    plot_ground_recall_by_tol(results_df, output_root / "ground_recall_by_ground_tol.png")
    for _, row in best_by_file_df.iterrows():
        plot_best_confusion_matrix(
            confusion_rows_df=confusion_rows_df,
            h5_stem=str(row["h5_stem"]),
            config_id=str(row["config_id"]),
            output_path=output_root / f"best_confusion_matrix_{row['h5_stem']}.png",
        )
        plot_best_class_fraction(
            class_fraction_df=class_fraction_df,
            h5_stem=str(row["h5_stem"]),
            config_id=str(row["config_id"]),
            output_path=output_root / f"pred_vs_reference_class_fraction_{row['h5_stem']}.png",
        )


def maybe_write_best_laz(
    ranked_df: pd.DataFrame,
    input_pairs: Sequence[Dict[str, Path]],
    output_root: Path,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    if ranked_df.empty:
        return {
            "requested": True,
            "status": "skipped_no_ranked_results",
        }

    best_row = ranked_df.iloc[0]
    mode = str(best_row["classifier_mode"])
    max_hag = float(best_row["NOISE_HAG_MAX_M"])
    density_threshold = best_row["NOISE_DENSITY_MAX_PTS_M3"]
    compatible = (
        mode == "height_density_pre_ground"
        and pd.notna(density_threshold)
        and math.isclose(max_hag, 40.0)
    )
    if not compatible:
        return {
            "requested": True,
            "status": "skipped_incompatible_with_formal_writer",
            "reason": (
                "Formal writer only reproduces baseline pre-ground density logic "
                "with NOISE_HAG_MAX_M == 40.0."
            ),
            "best_config_id": best_row["config_id"],
        }

    writer_output_root = output_root / "best_laz_export"
    ensure_dir(writer_output_root)
    outputs = []
    for input_pair in input_pairs:
        writer_config = dict(base_config)
        for key in PARAM_COLUMNS:
            value = best_row[key]
            writer_config[key] = None if pd.isna(value) else value
        writer_config["OUTPUT_ROOT"] = writer_output_root
        result = baseline_mod.process_one_pair(input_pair, writer_config)
        outputs.append(result)

    return {
        "requested": True,
        "status": "completed",
        "output_root": str(writer_output_root),
        "results": outputs,
    }


def format_config_brief(config_record: Dict[str, Any]) -> str:
    density_text = (
        "disabled"
        if config_record.get("NOISE_DENSITY_MAX_PTS_M3") is None
        else f"{float(config_record['NOISE_DENSITY_MAX_PTS_M3']):.4f}"
    )
    radius = config_record.get("LOCAL_FEATURE_RADIUS_M")
    radius_text = "disabled" if radius is None else f"{float(radius):.1f}m"
    return (
        f"id={config_record['config_id']} "
        f"mode={config_record['CLASSIFIER_MODE']} "
        f"ground_snr>={float(config_record['GROUND_SNR_MIN']):.1f} "
        f"grid={float(config_record['GRID_RES_M']):.1f}m "
        f"pct={int(config_record['GROUND_CELL_PERCENTILE'])} "
        f"tol={float(config_record['GROUND_RESID_TOL_M']):.2f}m "
        f"max_hag={float(config_record['NOISE_HAG_MAX_M']):.1f}m "
        f"radius={radius_text} "
        f"density={density_text}"
    )


def print_run_overview(
    input_pairs: Sequence[Dict[str, Path]],
    output_root: Path,
    parameter_grid: Sequence[Dict[str, Any]],
    write_diagnostic_png: bool,
    write_best_laz: bool,
    stage_1_coarse: bool,
    focus_priority_configs: bool,
    max_configs: Optional[int],
    random_subset_configs: bool,
    random_seed: int,
) -> None:
    print("=" * 100)
    print("CASALS refh debug parameter sweep")
    print("=" * 100)
    print(f"Output root: {output_root}")
    print(f"Input pair count: {len(input_pairs)}")
    print(f"Config count: {len(parameter_grid)}")
    print(f"Stage-1 coarse grid: {stage_1_coarse}")
    print(f"Focus priority configs: {focus_priority_configs}")
    print(f"MAX_CONFIGS: {max_configs}")
    print(f"RANDOM_SUBSET_CONFIGS: {random_subset_configs}")
    print(f"RANDOM_SEED: {random_seed}")
    print(f"WRITE_DIAGNOSTIC_PNG: {write_diagnostic_png}")
    print(f"WRITE_BEST_LAZ: {write_best_laz}")
    if not write_best_laz:
        print("Point-cloud export is disabled by default. This debug script will only write metrics and diagnostics.")
    print("Input files:")
    for pair in input_pairs:
        print(f"  - {Path(pair['h5_path']).stem}")
    if parameter_grid:
        print("First configs in sweep:")
        for config in parameter_grid[: min(5, len(parameter_grid))]:
            print(f"  - {format_config_brief(config)}")
        if len(parameter_grid) > 5:
            print(f"  - ... {len(parameter_grid) - 5} more configs")
    print("=" * 100)


def print_cache_summary(cached: Dict[str, Any]) -> None:
    ref_counts = collect_class_counts(
        np.asarray(cached["eval_gt_class"], dtype=np.uint8)[
            np.asarray(cached["alignment"]["eval_match_valid"], dtype=bool)
        ]
    )
    print(
        f"Prepared cache: {cached['h5_stem']} "
        f"points={cached['casals']['point_index'].size:,} "
        f"alignment={cached['alignment']['alignment_method']} "
        f"matched={int(np.count_nonzero(cached['alignment']['eval_match_valid'])):,} "
        f"high_conf_available={bool(cached['high_conf_available'])}"
    )
    print(
        f"  density_radii={sorted(cached['density_cache'])} "
        f"reference_counts={ref_counts}"
    )


def print_ranked_summary(ranked_df: pd.DataFrame, best_by_file_df: pd.DataFrame) -> None:
    print("=" * 100)
    print("Sweep summary")
    print("=" * 100)
    if ranked_df.empty:
        print("No successful ranked configs.")
        return

    top_n = min(5, len(ranked_df))
    print(f"Top {top_n} configs:")
    for _, row in ranked_df.head(top_n).iterrows():
        print(
            f"  rank={int(row['rank'])} "
            f"{row['config_id']} "
            f"mode={row['classifier_mode']} "
            f"mean_obj={float(row['mean_objective_score']):.4f} "
            f"min_obj={float(row['min_objective_score']):.4f} "
            f"mean_macro_f1={float(row['mean_macro_f1']):.4f} "
            f"mean_ground_recall={float(row['mean_class_2_recall']):.4f} "
            f"mean_noise_recall={float(row['mean_class_7_recall']):.4f}"
        )

    if not best_by_file_df.empty:
        print("Best config by file:")
        for _, row in best_by_file_df.iterrows():
            print(
                f"  file={row['h5_stem']} "
                f"config={row['config_id']} "
                f"global_rank={int(row['global_rank'])} "
                f"obj={float(row['objective_score']):.4f} "
                f"macro_f1={float(row['macro_f1']):.4f} "
                f"ground_recall={float(row['class_2_recall']):.4f} "
                f"noise_recall={float(row['class_7_recall']):.4f}"
            )


def print_output_summary(output_root: Path) -> None:
    print("=" * 100)
    print("Output files")
    print("=" * 100)
    for name in [
        "parameter_sweep_results.csv",
        "parameter_sweep_results_ranked.csv",
        "best_config.json",
        "best_configs_by_file.csv",
        "ablation_summary.csv",
        "confusion_matrices_long.csv",
        "reason_counts_long.csv",
        "class_fraction_diagnostics.csv",
        "failed_runs.json",
        "debug_run_metadata.json",
    ]:
        print(f"  - {output_root / name}")


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

    OUTPUT_ROOT = Path("./outputs/debug_casals_refh_classifier_params")
    WRITE_DIAGNOSTIC_PNG = True
    WRITE_BEST_LAZ = False

    STAGE_1_COARSE = True
    FOCUS_PRIORITY_CONFIGS = False
    MAX_CONFIGS = None
    RANDOM_SUBSET_CONFIGS = False
    RANDOM_SEED = 42

    base_config = build_base_config(output_root=OUTPUT_ROOT)
    parameter_grid = build_parameter_grid(
        base_config=base_config,
        stage_1_coarse=STAGE_1_COARSE,
        focus_priority_configs=FOCUS_PRIORITY_CONFIGS,
        max_configs=MAX_CONFIGS,
        random_subset_configs=RANDOM_SUBSET_CONFIGS,
        random_seed=RANDOM_SEED,
    )

    tested_radii = [cfg["LOCAL_FEATURE_RADIUS_M"] for cfg in parameter_grid]
    ensure_dir(OUTPUT_ROOT)
    print_run_overview(
        input_pairs=INPUT_PAIRS,
        output_root=OUTPUT_ROOT,
        parameter_grid=parameter_grid,
        write_diagnostic_png=WRITE_DIAGNOSTIC_PNG,
        write_best_laz=WRITE_BEST_LAZ,
        stage_1_coarse=STAGE_1_COARSE,
        focus_priority_configs=FOCUS_PRIORITY_CONFIGS,
        max_configs=MAX_CONFIGS,
        random_subset_configs=RANDOM_SUBSET_CONFIGS,
        random_seed=RANDOM_SEED,
    )

    cached_files: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []
    for input_pair in INPUT_PAIRS:
        try:
            cached = prepare_one_file_cache(
                input_pair=input_pair,
                base_config=base_config,
                tested_radii=tested_radii,
            )
            cached_files.append(cached)
            print_cache_summary(cached)
        except Exception as exc:
            failed_runs.append({
                "h5_stem": Path(input_pair["h5_path"]).stem,
                "config_id": "__prepare__",
                "classifier_mode": None,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"[PREP FAILED] {Path(input_pair['h5_path']).stem}: {type(exc).__name__}: {exc}")

    if not cached_files:
        print("No files were prepared successfully. Only failed_runs.json and debug metadata will be meaningful.")

    sweep = run_parameter_sweep(
        cached_files=cached_files,
        parameter_grid=parameter_grid,
    )
    failed_runs.extend(sweep["failed_runs"])

    results_df = pd.DataFrame(sweep["result_rows"], columns=RESULT_COLUMNS)
    reason_rows_df = pd.DataFrame(sweep["reason_rows"], columns=REASON_LONG_COLUMNS)
    confusion_rows_df = pd.DataFrame(sweep["confusion_rows"], columns=CONFUSION_LONG_COLUMNS)
    class_fraction_df = pd.DataFrame(sweep["class_fraction_rows"], columns=CLASS_FRACTION_COLUMNS)
    ranked_df, best_by_file_df = rank_and_select_best_configs(results_df)
    ablation_df = build_ablation_summary(results_df)

    debug_metadata: Dict[str, Any] = {
        "runtime_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_pairs": json_safe(INPUT_PAIRS),
        "base_config": json_safe(base_config),
        "parameter_grid_size": int(len(parameter_grid)),
        "prepared_files": [cached["h5_stem"] for cached in cached_files],
        "failed_file_preparations": int(sum(1 for item in failed_runs if item["config_id"] == "__prepare__")),
        "failed_config_runs": int(sum(1 for item in failed_runs if item["config_id"] != "__prepare__")),
        "stage_1_coarse": bool(STAGE_1_COARSE),
        "focus_priority_configs": bool(FOCUS_PRIORITY_CONFIGS),
        "max_configs": MAX_CONFIGS,
        "random_subset_configs": bool(RANDOM_SUBSET_CONFIGS),
        "random_seed": int(RANDOM_SEED),
        "write_diagnostic_png": bool(WRITE_DIAGNOSTIC_PNG),
        "write_best_laz": bool(WRITE_BEST_LAZ),
    }

    if WRITE_BEST_LAZ:
        debug_metadata["best_laz_export"] = maybe_write_best_laz(
            ranked_df=ranked_df,
            input_pairs=INPUT_PAIRS,
            output_root=OUTPUT_ROOT,
            base_config=base_config,
        )
        print(f"Best LAZ export status: {debug_metadata['best_laz_export']['status']}")

    write_debug_outputs(
        output_root=OUTPUT_ROOT,
        results_df=results_df,
        ranked_df=ranked_df,
        best_by_file_df=best_by_file_df,
        ablation_df=ablation_df,
        confusion_rows_df=confusion_rows_df,
        reason_rows_df=reason_rows_df,
        class_fraction_df=class_fraction_df,
        failed_runs=failed_runs,
        debug_metadata=debug_metadata,
        write_png=WRITE_DIAGNOSTIC_PNG,
    )

    print_ranked_summary(ranked_df=ranked_df, best_by_file_df=best_by_file_df)
    print(
        f"Failed preparations: {debug_metadata['failed_file_preparations']} | "
        f"Failed config runs: {debug_metadata['failed_config_runs']}"
    )
    print_output_summary(OUTPUT_ROOT)
    print(f"Wrote debug outputs to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
