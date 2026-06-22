"""
Two-stage debug sweep for CASALS refh classifier parameters.

This script reuses the official read / project / DTM / evaluation helpers from
classify_and_evaluate_casals_refh.py, but does not write per-config LAZ files by
default. It is intended for parameter search, ablation, paired comparisons, and
diagnostic summaries.
"""

from __future__ import annotations

import itertools
import json
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
    "mean_high_conf_macro_f1",
    "mean_high_conf_macro_f1_present",
    "mean_high_conf_weighted_f1",
    "mean_high_conf_n_points",
    "mean_high_conf_support_1",
    "mean_high_conf_support_2",
    "mean_high_conf_support_7",
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
    "max_objective_score",
    "best_config_id",
    "mean_macro_f1",
    "median_macro_f1",
    "mean_class_1_f1",
    "mean_class_2_f1",
    "mean_class_7_f1",
    "mean_class_1_recall",
    "mean_class_2_recall",
    "mean_class_7_recall",
    "mean_fraction_abs_diff_sum",
    "mean_pred_fraction_1",
    "mean_pred_fraction_2",
    "mean_pred_fraction_7",
    "mean_ref_fraction_1",
    "mean_ref_fraction_2",
    "mean_ref_fraction_7",
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

MODE_PAIR_DELTA_COLUMNS = [
    "h5_stem",
    "config_id",
    "baseline_config_id",
    *PARAM_COLUMNS,
    "delta_macro_f1",
    "delta_class_1_f1",
    "delta_class_2_f1",
    "delta_class_7_f1",
    "delta_class_2_recall",
    "delta_class_7_recall",
    "delta_fraction_abs_diff_sum",
    "improves_noise_recall",
    "hurts_ground_recall",
    "useful_density_enhancement",
]


def build_base_config(output_root: Path) -> Dict[str, Any]:
    return {
        "OUTPUT_ROOT": Path(output_root),
        "CLASSIFIER_MODE": "height_only",
        "GROUND_SNR_MIN": 5.0,
        "GRID_RES_M": 15.0,
        "MIN_POINTS_PER_CELL": 1,
        "GROUND_CELL_PERCENTILE": 2,
        "DTM_IDW_K": 12,
        "DTM_IDW_POWER": 2.0,
        "DTM_MAX_SEARCH_RADIUS_M": 30.0,
        "GROUND_RESID_TOL_M": 1.5,
        "NOISE_HAG_MAX_M": 30.0,
        "LOCAL_FEATURE_RADIUS_M": None,
        "LOCAL_FEATURE_MAX_NEIGHBORS": 24,
        "LOCAL_FEATURE_MIN_NEIGHBORS": 6,
        "LOCAL_FEATURE_QUERY_CHUNK_SIZE": 100_000,
        "NOISE_DENSITY_MAX_PTS_M3": None,
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


def config_signature(config_record: Dict[str, Any]) -> Tuple[Any, ...]:
    values = parameter_values_from_config(config_record)
    return tuple(values[col] for col in PARAM_COLUMNS)


def format_value(value: Any, fmt: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "None"
    return format(float(value), fmt)


def format_config_brief(config_record: Dict[str, Any]) -> str:
    values = parameter_values_from_config(config_record)
    return (
        f"id={config_record['config_id']} "
        f"mode={values['classifier_mode']} "
        f"snr={format_value(values['GROUND_SNR_MIN'], '.1f')} "
        f"grid={format_value(values['GRID_RES_M'], '.1f')} "
        f"pct={int(values['GROUND_CELL_PERCENTILE'])} "
        f"tol={format_value(values['GROUND_RESID_TOL_M'], '.1f')} "
        f"max_hag={format_value(values['NOISE_HAG_MAX_M'], '.1f')} "
        f"radius={format_value(values['LOCAL_FEATURE_RADIUS_M'], '.1f')} "
        f"density={format_value(values['NOISE_DENSITY_MAX_PTS_M3'], '.2f')}"
    )


def build_grid_records(
    classifier_modes: Sequence[str],
    ground_snr_values: Sequence[float],
    grid_res_values: Sequence[float],
    percentile_values: Sequence[int],
    tol_values: Sequence[float],
    max_hag_values: Sequence[float],
    radius_values: Sequence[float],
    density_values: Sequence[Optional[float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mode, ground_snr, grid_res, percentile, tol, max_hag, radius, density in itertools.product(
        classifier_modes,
        ground_snr_values,
        grid_res_values,
        percentile_values,
        tol_values,
        max_hag_values,
        radius_values,
        density_values,
    ):
        radius_use = None if density is None or mode == "height_only" else float(radius)
        density_use = None if mode == "height_only" else (None if density is None else float(density))
        rows.append({
            "CLASSIFIER_MODE": str(mode),
            "GROUND_SNR_MIN": float(ground_snr),
            "GRID_RES_M": float(grid_res),
            "GROUND_CELL_PERCENTILE": int(percentile),
            "GROUND_RESID_TOL_M": float(tol),
            "NOISE_HAG_MAX_M": float(max_hag),
            "LOCAL_FEATURE_RADIUS_M": radius_use,
            "NOISE_DENSITY_MAX_PTS_M3": density_use,
        })
    return rows


def build_parameter_grid(
    base_config: Dict[str, Any],
    search_mode: str,
    include_pre_ground_ablation: bool,
    max_configs: Optional[int],
    random_subset_configs: bool,
    random_seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if search_mode == "quick":
        rows.extend(build_grid_records(
            classifier_modes=[
                "height_only",
                "height_density_post_ground",
                "height_density_processed_only",
            ],
            ground_snr_values=[5.0],
            grid_res_values=[10.0, 15.0],
            percentile_values=[2, 5],
            tol_values=[1.0, 1.5],
            max_hag_values=[30.0],
            radius_values=[5.0],
            density_values=[None, 0.02, 0.04],
        ))
    elif search_mode == "refined":
        rows.extend(build_grid_records(
            classifier_modes=["height_only"],
            ground_snr_values=[5.0],
            grid_res_values=[10.0, 15.0],
            percentile_values=[2, 5, 10],
            tol_values=[1.0, 1.5, 2.0],
            max_hag_values=[25.0, 30.0, 35.0],
            radius_values=[5.0],
            density_values=[None],
        ))
        rows.extend(build_grid_records(
            classifier_modes=[
                "height_density_post_ground",
                "height_density_processed_only",
            ],
            ground_snr_values=[5.0],
            grid_res_values=[10.0, 15.0],
            percentile_values=[2, 5],
            tol_values=[1.0, 1.5],
            max_hag_values=[30.0],
            radius_values=[3.0, 5.0, 8.0],
            density_values=[None, 0.01, 0.02, 0.04, 0.08],
        ))
        if include_pre_ground_ablation:
            rows.extend(build_grid_records(
                classifier_modes=["height_density_pre_ground"],
                ground_snr_values=[5.0],
                grid_res_values=[10.0, 15.0],
                percentile_values=[2, 5],
                tol_values=[1.0, 1.5],
                max_hag_values=[30.0],
                radius_values=[5.0],
                density_values=[None, 0.02, 0.04],
            ))
    elif search_mode == "full":
        rows.extend(build_grid_records(
            classifier_modes=[
                "height_only",
                "height_density_post_ground",
                "height_density_processed_only",
                "height_no_below_noise",
            ] + (["height_density_pre_ground"] if include_pre_ground_ablation else []),
            ground_snr_values=[3.0, 5.0, 7.0],
            grid_res_values=[5.0, 10.0, 15.0],
            percentile_values=[2, 5, 10],
            tol_values=[0.5, 1.0, 1.5, 2.0],
            max_hag_values=[30.0, 40.0, 60.0],
            radius_values=[3.0, 5.0, 8.0],
            density_values=[None, 0.01, 0.02, 0.04, 0.08],
        ))
    else:
        raise ValueError(f"Unsupported RUN_MODE/search_mode: {search_mode}")

    unique_rows: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()
    for row in rows:
        config = dict(base_config)
        config.update(row)
        key = config_signature(config)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(config)

    if max_configs is not None and max_configs >= 0 and len(unique_rows) > max_configs:
        if random_subset_configs:
            rng = np.random.default_rng(int(random_seed))
            idx = np.sort(rng.choice(np.arange(len(unique_rows)), size=int(max_configs), replace=False))
            unique_rows = [unique_rows[int(i)] for i in idx]
        else:
            unique_rows = unique_rows[: int(max_configs)]

    for i, row in enumerate(unique_rows, start=1):
        row["config_id"] = f"cfg_{i:05d}"
    return unique_rows


def print_run_overview(
    input_pairs: Sequence[Dict[str, Path]],
    output_root: Path,
    parameter_grid: Sequence[Dict[str, Any]],
    run_mode: str,
    write_diagnostic_png: bool,
    write_best_laz: bool,
    resume_from_existing: bool,
    include_pre_ground_ablation: bool,
) -> None:
    print("=" * 100)
    print("CASALS refh debug parameter sweep")
    print("=" * 100)
    print(f"RUN_MODE: {run_mode}")
    print(f"OUTPUT_ROOT: {output_root}")
    print(f"Input pair count: {len(input_pairs)}")
    print(f"Config count: {len(parameter_grid)}")
    print(f"WRITE_DIAGNOSTIC_PNG: {write_diagnostic_png}")
    print(f"WRITE_BEST_LAZ: {write_best_laz}")
    print(f"RESUME_FROM_EXISTING_RESULTS: {resume_from_existing}")
    print(f"INCLUDE_PRE_GROUND_ABLATION: {include_pre_ground_ablation}")
    if not write_best_laz:
        print("Point-cloud export is disabled by default. This debug script writes metrics and diagnostics only.")
    print("Input files:")
    for pair in input_pairs:
        print(f"  - {Path(pair['h5_path']).stem}")
    print("Config preview:")
    for config in parameter_grid[: min(5, len(parameter_grid))]:
        print(f"  - {format_config_brief(config)}")
    if len(parameter_grid) > 5:
        print(f"  - ... {len(parameter_grid) - 5} more configs")
    print("=" * 100)


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
    radii_to_compute = sorted({float(v) for v in tested_radii if v is not None})
    for radius_m in radii_to_compute:
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


def get_or_build_dtm(cached_file: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
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


def density_enabled_for_config(config_record: Dict[str, Any]) -> bool:
    return (
        str(config_record.get("CLASSIFIER_MODE", config_record.get("classifier_mode", ""))) != "height_only"
        and config_record.get("NOISE_DENSITY_MAX_PTS_M3") is not None
    )


def classify_points_debug(
    z: np.ndarray,
    local_ground_z_m: np.ndarray,
    dtm_sample_valid: np.ndarray,
    point_density_pts_m3: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    return classify_points_baseline(
        z=np.asarray(z, dtype=np.float64),
        local_ground_z_m=np.asarray(local_ground_z_m, dtype=np.float64),
        dtm_sample_valid=np.asarray(dtm_sample_valid, dtype=np.uint8),
        point_density_pts_m3=np.asarray(point_density_pts_m3, dtype=np.float64),
        config=config,
    )


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

    summary = {
        "fraction_abs_diff_sum": fraction_abs_diff_sum,
    }
    for cls in LABEL_ORDER:
        summary[f"pred_fraction_{cls}"] = pred_fractions[cls]
        summary[f"ref_fraction_{cls}"] = ref_fractions[cls]
    return rows, summary


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
    if float(metric_row["NOISE_HAG_MAX_M"]) > 40.0:
        penalty += 0.01
    if str(metric_row["classifier_mode"]) == "height_density_pre_ground":
        penalty += 0.015

    return float(objective_all - penalty)


def summarize_reason_counts(
    h5_stem: str,
    config_record: Dict[str, Any],
    classification_reason: np.ndarray,
) -> List[Dict[str, Any]]:
    counts = collect_class_counts(np.asarray(classification_reason, dtype=np.uint8))
    total = max(int(np.asarray(classification_reason).size), 1)
    rows: List[Dict[str, Any]] = []
    for code in sorted(DEBUG_REASON_MAP):
        rows.append({
            "h5_stem": h5_stem,
            "config_id": config_record["config_id"],
            **parameter_values_from_config(config_record),
            "reason_code": int(code),
            "reason_name": DEBUG_REASON_MAP[code],
            "count": int(counts.get(code, 0)),
            "fraction": float(counts.get(code, 0) / total),
        })
    return rows


def summarize_confusion_rows(
    h5_stem: str,
    config_record: Dict[str, Any],
    evaluation_metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for subset_name, subset_result in evaluation_metrics["subset_results"].items():
        if subset_result.get("status") != "ok":
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


def compute_metrics_table(
    h5_stem: str,
    config_record: Dict[str, Any],
    summary_row: Dict[str, Any],
    evaluation_metrics: Dict[str, Any],
    high_conf_available: bool,
    warning: str = "",
) -> List[Dict[str, Any]]:
    _, fraction_summary = summarize_class_fractions(h5_stem, config_record, summary_row)
    rows: List[Dict[str, Any]] = []
    subset_results = evaluation_metrics["subset_results"]
    for subset_name in ["all_matched", "high_confidence_reference"]:
        subset_result = subset_results.get(subset_name)
        row = {
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
        metric_fill_keys = [
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

        if subset_result is None:
            row["status"] = "unavailable" if not high_conf_available else "missing"
            for key in metric_fill_keys:
                row[key] = np.nan
            rows.append(row)
            continue

        if subset_result.get("status") != "ok":
            row["status"] = subset_result.get("status", "unavailable")
            for key in metric_fill_keys:
                row[key] = np.nan
            rows.append(row)
            continue

        class_metrics = report_rows_to_metrics(subset_result["report_rows"])
        row.update({
            "accuracy": float(subset_result["accuracy"]),
            "macro_precision": float(subset_result["macro_precision"]),
            "macro_recall": float(subset_result["macro_recall"]),
            "macro_f1": float(subset_result["macro_f1"]),
            "weighted_precision": float(subset_result["weighted_precision"]),
            "weighted_recall": float(subset_result["weighted_recall"]),
            "weighted_f1": float(subset_result["weighted_f1"]),
            "macro_precision_present": subset_result.get("macro_precision_present"),
            "macro_recall_present": subset_result.get("macro_recall_present"),
            "macro_f1_present": subset_result.get("macro_f1_present"),
            "weighted_precision_present": subset_result.get("weighted_precision_present"),
            "weighted_recall_present": subset_result.get("weighted_recall_present"),
            "weighted_f1_present": subset_result.get("weighted_f1_present"),
            "n_present_labels": subset_result.get("n_present_labels"),
            "support_min": subset_result.get("support_min"),
            "support_1": subset_result.get("support_1"),
            "support_2": subset_result.get("support_2"),
            "support_7": subset_result.get("support_7"),
            **class_metrics,
        })
        row["objective_score"] = compute_objective_score(row) if subset_name == "all_matched" else np.nan
        rows.append(row)
    return rows


def run_one_config_on_cached_file(
    cached_file: Dict[str, Any],
    config_record: Dict[str, Any],
) -> Dict[str, Any]:
    config = dict(config_record)
    dtm_payload = get_or_build_dtm(cached_file, config)
    if density_enabled_for_config(config):
        radius = float(config["LOCAL_FEATURE_RADIUS_M"])
        density_result = cached_file["density_cache"][radius]
        local_neighbor_count = np.asarray(density_result["local_neighbor_count"], dtype=np.uint32)
        point_density_pts_m3 = np.asarray(density_result["point_density_pts_m3"], dtype=np.float64)
    else:
        local_neighbor_count = np.zeros(cached_file["x"].shape[0], dtype=np.uint32)
        point_density_pts_m3 = np.full(cached_file["x"].shape[0], np.nan, dtype=np.float64)

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
    reason_rows = summarize_reason_counts(cached_file["h5_stem"], config_record, classification_reason)
    confusion_rows = summarize_confusion_rows(cached_file["h5_stem"], config_record, evaluation_metrics)
    class_fraction_rows, _ = summarize_class_fractions(cached_file["h5_stem"], config_record, summary_row)

    return {
        "status": "success",
        "metrics_rows": metrics_rows,
        "reason_rows": reason_rows,
        "confusion_rows": confusion_rows,
        "class_fraction_rows": class_fraction_rows,
    }


def results_output_paths(output_root: Path) -> Dict[str, Path]:
    return {
        "results": output_root / "parameter_sweep_results.csv",
        "ranked": output_root / "parameter_sweep_results_ranked.csv",
        "best_config": output_root / "best_config.json",
        "top_configs_json": output_root / "top_10_configs.json",
        "best_by_file": output_root / "best_configs_by_file.csv",
        "ablation": output_root / "ablation_summary.csv",
        "ablation_by_file": output_root / "ablation_summary_by_file.csv",
        "mode_pair_delta": output_root / "mode_pair_delta_vs_height_only.csv",
        "confusion": output_root / "confusion_matrices_long.csv",
        "reason": output_root / "reason_counts_long.csv",
        "class_fraction": output_root / "class_fraction_diagnostics.csv",
        "top_all": output_root / "top_configs_all_matched.csv",
        "top_high": output_root / "top_configs_high_confidence.csv",
        "top_confusion": output_root / "top_config_confusion_matrices.csv",
        "top_reason": output_root / "top_config_reason_counts.csv",
        "top_fraction": output_root / "top_config_class_fractions.csv",
        "top_console_summary": output_root / "top_config_console_summary.txt",
        "failed": output_root / "failed_runs.json",
        "metadata": output_root / "debug_run_metadata.json",
    }


def load_existing_records(path: Path, columns: Sequence[str]) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[list(columns)].to_dict(orient="records")


def load_resume_state(output_root: Path) -> Dict[str, Any]:
    paths = results_output_paths(output_root)
    failed_runs: List[Dict[str, Any]] = []
    if paths["failed"].exists():
        try:
            failed_payload = json.loads(paths["failed"].read_text(encoding="utf-8"))
            failed_runs = list(failed_payload.get("failed_runs", []))
        except Exception:
            failed_runs = []
    return {
        "result_rows": load_existing_records(paths["results"], RESULT_COLUMNS),
        "reason_rows": load_existing_records(paths["reason"], REASON_LONG_COLUMNS),
        "confusion_rows": load_existing_records(paths["confusion"], CONFUSION_LONG_COLUMNS),
        "class_fraction_rows": load_existing_records(paths["class_fraction"], CLASS_FRACTION_COLUMNS),
        "failed_runs": failed_runs,
    }


def config_mean_objective(result_rows: List[Dict[str, Any]], config_id: str, n_files: int) -> Optional[float]:
    matches = [
        row for row in result_rows
        if row["config_id"] == config_id and row["subset_name"] == "all_matched" and row["status"] == "ok"
    ]
    if len(matches) < n_files:
        return None
    scores = [float(row["objective_score"]) for row in matches if row.get("objective_score") is not None and np.isfinite(row["objective_score"])]
    if len(scores) < n_files:
        return None
    return float(np.mean(scores))


def print_progress_line(row: Dict[str, Any], high_conf_row: Dict[str, Any], progress: int, total: int) -> None:
    high_conf_text = (
        f"high_conf_macro_present={float(high_conf_row['macro_f1_present']):.4f} "
        f"support=({int(high_conf_row['support_1'])},{int(high_conf_row['support_2'])},{int(high_conf_row['support_7'])})"
        if high_conf_row["status"] == "ok"
        else f"high_conf_status={high_conf_row['status']}"
    )
    print(
        f"[{progress}/{total}] "
        f"{row['h5_stem']} "
        f"{row['classifier_mode']} "
        f"macro={float(row['macro_f1']):.4f} "
        f"g_f1={float(row['class_2_f1']):.4f} "
        f"n_f1={float(row['class_7_f1']):.4f} "
        f"g_rec={float(row['class_2_recall']):.4f} "
        f"n_rec={float(row['class_7_recall']):.4f} "
        f"frac_diff={float(row['fraction_abs_diff_sum']):.4f} "
        f"obj={float(row['objective_score']):.4f} "
        f"{high_conf_text}"
    )


def run_parameter_sweep(
    cached_files: Sequence[Dict[str, Any]],
    parameter_grid: Sequence[Dict[str, Any]],
    output_root: Path,
    resume_from_existing_results: bool,
    print_every_config: bool,
    print_every_n_configs: int,
    print_only_improvements: bool,
    verbose_per_file_progress: bool,
) -> Dict[str, Any]:
    state = load_resume_state(output_root) if resume_from_existing_results else {
        "result_rows": [],
        "reason_rows": [],
        "confusion_rows": [],
        "class_fraction_rows": [],
        "failed_runs": [],
    }
    result_rows: List[Dict[str, Any]] = list(state["result_rows"])
    reason_rows: List[Dict[str, Any]] = list(state["reason_rows"])
    confusion_rows: List[Dict[str, Any]] = list(state["confusion_rows"])
    class_fraction_rows: List[Dict[str, Any]] = list(state["class_fraction_rows"])
    failed_runs: List[Dict[str, Any]] = list(state["failed_runs"])

    completed_pairs = {
        (row["h5_stem"], row["config_id"])
        for row in result_rows
        if row["subset_name"] == "all_matched" and row["status"] == "ok"
    }
    total = len(cached_files) * len(parameter_grid)
    progress = 0
    best_mean_objective = max(
        [
            value for value in (
                config_mean_objective(result_rows, config["config_id"], len(cached_files))
                for config in parameter_grid
            )
            if value is not None
        ],
        default=None,
    )

    for config_idx, config_record in enumerate(parameter_grid, start=1):
        if print_every_config:
            print("-" * 100)
            print(f"Config {config_idx}/{len(parameter_grid)}: {format_config_brief(config_record)}")
        mode_rows_before = [
            row for row in result_rows
            if row["config_id"] == config_record["config_id"] and row["subset_name"] == "all_matched" and row["status"] == "ok"
        ]
        for cached_file in cached_files:
            progress += 1
            pair_key = (cached_file["h5_stem"], config_record["config_id"])
            if pair_key in completed_pairs:
                if verbose_per_file_progress:
                    print(f"[{progress}/{total}] skip existing {cached_file['h5_stem']} {config_record['config_id']}")
                continue
            try:
                run_result = run_one_config_on_cached_file(cached_file, config_record)
                result_rows.extend(run_result["metrics_rows"])
                reason_rows.extend(run_result["reason_rows"])
                confusion_rows.extend(run_result["confusion_rows"])
                class_fraction_rows.extend(run_result["class_fraction_rows"])
                completed_pairs.add(pair_key)

                all_matched_row = next(row for row in run_result["metrics_rows"] if row["subset_name"] == "all_matched")
                high_conf_row = next(row for row in run_result["metrics_rows"] if row["subset_name"] == "high_confidence_reference")
                if verbose_per_file_progress:
                    print_progress_line(all_matched_row, high_conf_row, progress, total)
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
                empty_row = {
                    "h5_stem": cached_file["h5_stem"],
                    "config_id": config_record["config_id"],
                    "warning": warning,
                    **parameter_values_from_config(config_record),
                }
                for subset_name in ["all_matched", "high_confidence_reference"]:
                    failed_row = dict(empty_row)
                    failed_row["subset_name"] = subset_name
                    failed_row["status"] = "failed"
                    for col in RESULT_COLUMNS:
                        if col not in failed_row:
                            failed_row[col] = np.nan
                    result_rows.append(failed_row)
                print(f"[{progress}/{total}] FAILED {cached_file['h5_stem']} {config_record['config_id']} {warning}")

        mean_obj = config_mean_objective(result_rows, config_record["config_id"], len(cached_files))
        if mean_obj is not None and (best_mean_objective is None or mean_obj > best_mean_objective):
            best_mean_objective = mean_obj
            print(f"[NEW BEST] {format_config_brief(config_record)} mean_objective={mean_obj:.4f}")
        elif print_every_n_configs > 0 and config_idx % print_every_n_configs == 0:
            print(f"[PROGRESS] config {config_idx}/{len(parameter_grid)} latest={format_config_brief(config_record)}")

        mode_rows_after = [
            row for row in result_rows
            if row["config_id"] == config_record["config_id"] and row["subset_name"] == "all_matched" and row["status"] == "ok"
        ]
        if len(mode_rows_after) > len(mode_rows_before) and not print_only_improvements and not verbose_per_file_progress:
            sample_row = mode_rows_after[0]
            high_conf_candidates = [
                row for row in result_rows
                if row["config_id"] == config_record["config_id"]
                and row["h5_stem"] == sample_row["h5_stem"]
                and row["subset_name"] == "high_confidence_reference"
            ]
            high_conf_row = high_conf_candidates[0] if high_conf_candidates else {"status": "unavailable"}
            print_progress_line(sample_row, high_conf_row, progress, total)

    if not verbose_per_file_progress:
        all_df = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
        subset_df = all_df[(all_df["subset_name"] == "all_matched") & (all_df["status"] == "ok")].copy()
        if not subset_df.empty:
            print("-" * 100)
            print("Mode summary")
            for mode, group in subset_df.groupby("classifier_mode", sort=False):
                print(
                    f"  mode={mode} "
                    f"mean_obj={float(group['objective_score'].mean()):.4f} "
                    f"mean_macro={float(group['macro_f1'].mean()):.4f} "
                    f"mean_noise_recall={float(group['class_7_recall'].mean()):.4f} "
                    f"n_rows={int(group.shape[0])}"
                )

    return {
        "result_rows": result_rows,
        "reason_rows": reason_rows,
        "confusion_rows": confusion_rows,
        "class_fraction_rows": class_fraction_rows,
        "failed_runs": failed_runs,
    }


def rank_and_select_best_configs(
    results_df: pd.DataFrame,
    rank_primary_subset: str,
    use_high_conf_for_ranking: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ranking_subset = "high_confidence_reference" if use_high_conf_for_ranking else rank_primary_subset
    subset_df = results_df[
        (results_df["subset_name"] == ranking_subset)
        & (results_df["status"] == "ok")
    ].copy()
    if subset_df.empty:
        return empty_frame(RANKED_COLUMNS), empty_frame(BEST_BY_FILE_COLUMNS)

    high_conf_df = results_df[
        (results_df["subset_name"] == "high_confidence_reference")
        & (results_df["status"] == "ok")
    ].copy()

    grouped_rows: List[Dict[str, Any]] = []
    for config_id, group in subset_df.groupby("config_id", sort=False):
        first = group.iloc[0]
        high_group = high_conf_df[high_conf_df["config_id"] == config_id]
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
            "mean_high_conf_macro_f1": float(high_group["macro_f1"].mean()) if not high_group.empty else np.nan,
            "mean_high_conf_macro_f1_present": float(high_group["macro_f1_present"].mean()) if not high_group.empty else np.nan,
            "mean_high_conf_weighted_f1": float(high_group["weighted_f1"].mean()) if not high_group.empty else np.nan,
            "mean_high_conf_n_points": float((high_group["support_1"] + high_group["support_2"] + high_group["support_7"]).mean()) if not high_group.empty else np.nan,
            "mean_high_conf_support_1": float(high_group["support_1"].mean()) if not high_group.empty else np.nan,
            "mean_high_conf_support_2": float(high_group["support_2"].mean()) if not high_group.empty else np.nan,
            "mean_high_conf_support_7": float(high_group["support_7"].mean()) if not high_group.empty else np.nan,
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
    global_rank_map = {row["config_id"]: int(row["rank"]) for _, row in ranked_df.iterrows()}
    file_source_df = results_df[
        (results_df["subset_name"] == rank_primary_subset)
        & (results_df["status"] == "ok")
    ].copy()
    for h5_stem, group in file_source_df.groupby("h5_stem", sort=False):
        best = group.sort_values(
            by=["objective_score", "macro_f1", "class_7_recall", "fraction_abs_diff_sum"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).iloc[0]
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


def build_ablation_summary(results_df: pd.DataFrame, by_file: bool) -> pd.DataFrame:
    subset_df = results_df[
        (results_df["subset_name"] == "all_matched")
        & (results_df["status"] == "ok")
    ].copy()
    if subset_df.empty:
        return empty_frame(ABLATION_COLUMNS)
    group_keys = ["h5_stem", "classifier_mode"] if by_file else ["classifier_mode"]
    rows: List[Dict[str, Any]] = []
    for keys, group in subset_df.groupby(group_keys, sort=False):
        best_row = group.sort_values("objective_score", ascending=False, kind="mergesort").iloc[0]
        h5_stem = keys[0] if by_file else "__all__"
        classifier_mode = keys[1] if by_file else keys
        rows.append({
            "h5_stem": h5_stem,
            "config_id": "__all__",
            "classifier_mode": classifier_mode,
            "n_runs": int(group.shape[0]),
            "mean_objective_score": float(group["objective_score"].mean()),
            "median_objective_score": float(group["objective_score"].median()),
            "max_objective_score": float(group["objective_score"].max()),
            "best_config_id": best_row["config_id"],
            "mean_macro_f1": float(group["macro_f1"].mean()),
            "median_macro_f1": float(group["macro_f1"].median()),
            "mean_class_1_f1": float(group["class_1_f1"].mean()),
            "mean_class_2_f1": float(group["class_2_f1"].mean()),
            "mean_class_7_f1": float(group["class_7_f1"].mean()),
            "mean_class_1_recall": float(group["class_1_recall"].mean()),
            "mean_class_2_recall": float(group["class_2_recall"].mean()),
            "mean_class_7_recall": float(group["class_7_recall"].mean()),
            "mean_fraction_abs_diff_sum": float(group["fraction_abs_diff_sum"].mean()),
            "mean_pred_fraction_1": float(group["pred_fraction_1"].mean()),
            "mean_pred_fraction_2": float(group["pred_fraction_2"].mean()),
            "mean_pred_fraction_7": float(group["pred_fraction_7"].mean()),
            "mean_ref_fraction_1": float(group["ref_fraction_1"].mean()),
            "mean_ref_fraction_2": float(group["ref_fraction_2"].mean()),
            "mean_ref_fraction_7": float(group["ref_fraction_7"].mean()),
        })
    return pd.DataFrame(rows, columns=ABLATION_COLUMNS)


def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def build_mode_pair_delta_vs_height_only(results_df: pd.DataFrame) -> pd.DataFrame:
    subset_df = results_df[
        (results_df["subset_name"] == "all_matched")
        & (results_df["status"] == "ok")
    ].copy()
    if subset_df.empty:
        return empty_frame(MODE_PAIR_DELTA_COLUMNS)

    match_cols = [
        "GROUND_SNR_MIN",
        "GRID_RES_M",
        "GROUND_CELL_PERCENTILE",
        "MIN_POINTS_PER_CELL",
        "DTM_IDW_K",
        "DTM_IDW_POWER",
        "DTM_MAX_SEARCH_RADIUS_M",
        "GROUND_RESID_TOL_M",
        "NOISE_HAG_MAX_M",
    ]
    baseline_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for _, row in subset_df[subset_df["classifier_mode"] == "height_only"].iterrows():
        key = (row["h5_stem"],) + tuple(row[col] for col in match_cols)
        baseline_map[key] = row.to_dict()

    rows: List[Dict[str, Any]] = []
    for _, row in subset_df[subset_df["classifier_mode"] != "height_only"].iterrows():
        key = (row["h5_stem"],) + tuple(row[col] for col in match_cols)
        baseline = baseline_map.get(key)
        if baseline is None:
            continue
        delta_class_2_recall = float(row["class_2_recall"] - baseline["class_2_recall"])
        delta_class_7_recall = float(row["class_7_recall"] - baseline["class_7_recall"])
        delta_macro_f1 = float(row["macro_f1"] - baseline["macro_f1"])
        rows.append({
            "h5_stem": row["h5_stem"],
            "config_id": row["config_id"],
            "baseline_config_id": baseline["config_id"],
            **{name: row[name] for name in PARAM_COLUMNS},
            "delta_macro_f1": delta_macro_f1,
            "delta_class_1_f1": float(row["class_1_f1"] - baseline["class_1_f1"]),
            "delta_class_2_f1": float(row["class_2_f1"] - baseline["class_2_f1"]),
            "delta_class_7_f1": float(row["class_7_f1"] - baseline["class_7_f1"]),
            "delta_class_2_recall": delta_class_2_recall,
            "delta_class_7_recall": delta_class_7_recall,
            "delta_fraction_abs_diff_sum": float(row["fraction_abs_diff_sum"] - baseline["fraction_abs_diff_sum"]),
            "improves_noise_recall": bool(delta_class_7_recall > 0.0),
            "hurts_ground_recall": bool(delta_class_2_recall < -0.01),
            "useful_density_enhancement": bool(
                delta_class_7_recall > 0.03
                and delta_class_2_recall > -0.01
                and delta_macro_f1 >= -0.005
            ),
        })
    return pd.DataFrame(rows, columns=MODE_PAIR_DELTA_COLUMNS)


def top_off_diagonal_confusions(confusion_rows_df: pd.DataFrame, config_id: str) -> List[str]:
    df = confusion_rows_df[
        (confusion_rows_df["config_id"] == config_id)
        & (confusion_rows_df["subset_name"] == "all_matched")
        & (confusion_rows_df["status"] == "ok")
        & (confusion_rows_df["true_class"] != confusion_rows_df["pred_class"])
    ].copy()
    if df.empty:
        return []
    summary = (
        df.groupby(["true_class", "pred_class", "true_name", "pred_name"], sort=False)["count"]
        .sum()
        .sort_values(ascending=False)
        .head(2)
        .reset_index()
    )
    return [
        f"true class {int(row.true_class)} -> pred {int(row.pred_class)}: {int(row.count)}"
        for row in summary.itertuples(index=False)
    ]


def write_top_config_diagnostics(
    ranked_df: pd.DataFrame,
    results_df: pd.DataFrame,
    confusion_rows_df: pd.DataFrame,
    reason_rows_df: pd.DataFrame,
    class_fraction_df: pd.DataFrame,
    output_root: Path,
    top_n: int,
) -> Dict[str, Path]:
    paths = results_output_paths(output_root)
    top_ids = ranked_df.head(top_n)["config_id"].tolist() if not ranked_df.empty else []
    top_ranked = ranked_df[ranked_df["config_id"].isin(top_ids)].copy()
    all_matched = results_df[
        (results_df["config_id"].isin(top_ids))
        & (results_df["subset_name"] == "all_matched")
    ].copy()
    high_conf = results_df[
        (results_df["config_id"].isin(top_ids))
        & (results_df["subset_name"] == "high_confidence_reference")
    ].copy()
    top_confusion = confusion_rows_df[confusion_rows_df["config_id"].isin(top_ids)].copy()
    top_reason = reason_rows_df[reason_rows_df["config_id"].isin(top_ids)].copy()
    top_fraction = class_fraction_df[class_fraction_df["config_id"].isin(top_ids)].copy()

    top_all_out = all_matched.merge(top_ranked[["config_id", "rank"]], on="config_id", how="left")
    top_high_out = high_conf.merge(top_ranked[["config_id", "rank"]], on="config_id", how="left")
    top_all_out.to_csv(paths["top_all"], index=False)
    top_high_out.to_csv(paths["top_high"], index=False)
    top_confusion.to_csv(paths["top_confusion"], index=False)
    top_reason.to_csv(paths["top_reason"], index=False)
    top_fraction.to_csv(paths["top_fraction"], index=False)

    lines: List[str] = []
    for _, ranked_row in top_ranked.iterrows():
        values = ranked_row.to_dict()
        lines.append(
            f"Rank {int(ranked_row['rank'])} | {ranked_row['config_id']} | "
            f"mode={ranked_row['classifier_mode']} | snr={values['GROUND_SNR_MIN']} | "
            f"grid={values['GRID_RES_M']} | pct={values['GROUND_CELL_PERCENTILE']} | "
            f"tol={values['GROUND_RESID_TOL_M']} | max_hag={values['NOISE_HAG_MAX_M']} | "
            f"density={values['NOISE_DENSITY_MAX_PTS_M3']}"
        )
        lines.append(
            f"  mean_obj={float(ranked_row['mean_objective_score']):.4f}, "
            f"min_obj={float(ranked_row['min_objective_score']):.4f}, "
            f"mean_macro_f1={float(ranked_row['mean_macro_f1']):.4f}, "
            f"mean_noise_recall={float(ranked_row['mean_class_7_recall']):.4f}"
        )
        per_file = all_matched[all_matched["config_id"] == ranked_row["config_id"]].copy()
        for _, row in per_file.sort_values("h5_stem").iterrows():
            lines.append(
                f"  {row['h5_stem']}: macro={float(row['macro_f1']):.4f}, "
                f"ground_f1={float(row['class_2_f1']):.4f}, noise_f1={float(row['class_7_f1']):.4f}, "
                f"ground_recall={float(row['class_2_recall']):.4f}, noise_recall={float(row['class_7_recall']):.4f}, "
                f"frac_diff={float(row['fraction_abs_diff_sum']):.4f}"
            )
        issues = top_off_diagonal_confusions(confusion_rows_df, str(ranked_row["config_id"]))
        if issues:
            lines.append("  main confusion issue:")
            for issue in issues:
                lines.append(f"    {issue}")
        lines.append("")

    paths["top_console_summary"].write_text("\n".join(lines), encoding="utf-8")
    return {
        "top_all": paths["top_all"],
        "top_high": paths["top_high"],
        "top_confusion": paths["top_confusion"],
        "top_reason": paths["top_reason"],
        "top_fraction": paths["top_fraction"],
        "top_console_summary": paths["top_console_summary"],
    }


def build_top_n_configs_payload(
    ranked_df: pd.DataFrame,
    results_df: pd.DataFrame,
    class_fraction_df: pd.DataFrame,
    reason_rows_df: pd.DataFrame,
    top_n: int,
) -> Dict[str, Any]:
    top_payload: Dict[str, Any] = {"top_configs": []}
    top_ranked = ranked_df.head(top_n)
    for _, ranked_row in top_ranked.iterrows():
        config_id = ranked_row["config_id"]
        top_payload["top_configs"].append({
            "rank": int(ranked_row["rank"]),
            "config_id": config_id,
            "config": {key: json_safe(ranked_row[key]) for key in ["config_id", *PARAM_COLUMNS]},
            "per_file_metrics": results_df[results_df["config_id"] == config_id].to_dict(orient="records"),
            "class_fraction_diagnostics": class_fraction_df[class_fraction_df["config_id"] == config_id].to_dict(orient="records"),
            "reason_count_diagnostics": reason_rows_df[reason_rows_df["config_id"] == config_id].to_dict(orient="records"),
        })
    return top_payload


def plot_objective_vs_rank(ranked_df: pd.DataFrame, output_path: Path) -> None:
    if ranked_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ranked_df["rank"], ranked_df["mean_objective_score"], marker="o")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Mean objective")
    ax.set_title("Objective vs config rank")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_macro_f1_by_mode(results_df: pd.DataFrame, output_path: Path) -> None:
    subset_df = results_df[(results_df["subset_name"] == "all_matched") & (results_df["status"] == "ok")]
    if subset_df.empty:
        return
    modes = list(dict.fromkeys(subset_df["classifier_mode"].tolist()))
    data = [subset_df.loc[subset_df["classifier_mode"] == mode, "macro_f1"].to_numpy(dtype=np.float64) for mode in modes]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=modes, showfliers=False)
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
    ]
    if subset_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, group in subset_df.groupby("classifier_mode", sort=False):
        ax.scatter(group["NOISE_DENSITY_MAX_PTS_M3"], group["class_7_recall"], label=mode, alpha=0.7, s=25)
    ax.set_xlabel("NOISE_DENSITY_MAX_PTS_M3")
    ax.set_ylabel("Class 7 recall")
    ax.set_title("Noise recall by density threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ground_recall_by_tol(results_df: pd.DataFrame, output_path: Path) -> None:
    subset_df = results_df[(results_df["subset_name"] == "all_matched") & (results_df["status"] == "ok")]
    if subset_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(subset_df["GROUND_RESID_TOL_M"], subset_df["class_2_recall"], alpha=0.65, s=25)
    ax.set_xlabel("GROUND_RESID_TOL_M")
    ax.set_ylabel("Class 2 recall")
    ax.set_title("Ground recall by ground tolerance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_best_confusion_matrix(confusion_rows_df: pd.DataFrame, h5_stem: str, config_id: str, output_path: Path) -> None:
    df = confusion_rows_df[
        (confusion_rows_df["h5_stem"] == h5_stem)
        & (confusion_rows_df["config_id"] == config_id)
        & (confusion_rows_df["subset_name"] == "all_matched")
        & (confusion_rows_df["status"] == "ok")
    ]
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
                text = f"{int(mat[i, j])}" if fmt == "d" else f"{float(mat[i, j]):.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Best confusion matrix: {h5_stem}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_best_class_fraction(class_fraction_df: pd.DataFrame, h5_stem: str, config_id: str, output_path: Path) -> None:
    df = class_fraction_df[
        (class_fraction_df["h5_stem"] == h5_stem)
        & (class_fraction_df["config_id"] == config_id)
    ].sort_values("class_code")
    if df.empty:
        return
    x = np.arange(df.shape[0], dtype=np.float64)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2.0, df["pred_fraction"], width=width, label="Predicted")
    ax.bar(x + width / 2.0, df["ref_fraction"], width=width, label="Reference")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class_name"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("Class fraction")
    ax.set_title(f"Pred vs reference class fractions: {h5_stem}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_delta_noise_recall(mode_pair_delta_df: pd.DataFrame, output_path: Path) -> None:
    if mode_pair_delta_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, group in mode_pair_delta_df.groupby("classifier_mode", sort=False):
        ax.scatter(group["NOISE_DENSITY_MAX_PTS_M3"], group["delta_class_7_recall"], label=mode, alpha=0.7, s=25)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax.set_xlabel("NOISE_DENSITY_MAX_PTS_M3")
    ax.set_ylabel("Delta class 7 recall vs height_only")
    ax.set_title("Delta noise recall vs density threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_delta_macro_f1(mode_pair_delta_df: pd.DataFrame, output_path: Path) -> None:
    if mode_pair_delta_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, group in mode_pair_delta_df.groupby("classifier_mode", sort=False):
        ax.scatter(group["NOISE_DENSITY_MAX_PTS_M3"], group["delta_macro_f1"], label=mode, alpha=0.7, s=25)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax.set_xlabel("NOISE_DENSITY_MAX_PTS_M3")
    ax.set_ylabel("Delta macro F1 vs height_only")
    ax.set_title("Delta macro F1 vs density threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_objective_heatmap_tol_vs_maxhag(results_df: pd.DataFrame, output_path: Path) -> None:
    df = results_df[
        (results_df["subset_name"] == "all_matched")
        & (results_df["status"] == "ok")
        & (results_df["classifier_mode"] == "height_only")
    ]
    if df.empty:
        return
    pivot = df.pivot_table(
        index="GROUND_RESID_TOL_M",
        columns="NOISE_HAG_MAX_M",
        values="objective_score",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.to_numpy(dtype=np.float64), cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(v) for v in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel("NOISE_HAG_MAX_M")
    ax.set_ylabel("GROUND_RESID_TOL_M")
    ax.set_title("Height-only mean objective heatmap")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{float(value):.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_best_reason_counts(reason_rows_df: pd.DataFrame, h5_stem: str, config_id: str, output_path: Path) -> None:
    df = reason_rows_df[
        (reason_rows_df["h5_stem"] == h5_stem)
        & (reason_rows_df["config_id"] == config_id)
    ].sort_values("reason_code")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(df.shape[0], dtype=np.float64)
    ax.bar(x, df["count"])
    ax.set_xticks(x)
    ax.set_xticklabels(df["reason_name"].tolist(), rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Best config reason counts: {h5_stem}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_debug_outputs(
    output_root: Path,
    results_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    best_by_file_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    ablation_by_file_df: pd.DataFrame,
    mode_pair_delta_df: pd.DataFrame,
    confusion_rows_df: pd.DataFrame,
    reason_rows_df: pd.DataFrame,
    class_fraction_df: pd.DataFrame,
    failed_runs: List[Dict[str, Any]],
    debug_metadata: Dict[str, Any],
    write_png: bool,
    save_top_config_jsons: bool,
    n_top_configs_to_save: int,
) -> None:
    ensure_dir(output_root)
    paths = results_output_paths(output_root)

    results_df = results_df if not results_df.empty else empty_frame(RESULT_COLUMNS)
    ranked_df = ranked_df if not ranked_df.empty else empty_frame(RANKED_COLUMNS)
    best_by_file_df = best_by_file_df if not best_by_file_df.empty else empty_frame(BEST_BY_FILE_COLUMNS)
    ablation_df = ablation_df if not ablation_df.empty else empty_frame(ABLATION_COLUMNS)
    ablation_by_file_df = ablation_by_file_df if not ablation_by_file_df.empty else empty_frame(ABLATION_COLUMNS)
    mode_pair_delta_df = mode_pair_delta_df if not mode_pair_delta_df.empty else empty_frame(MODE_PAIR_DELTA_COLUMNS)
    confusion_rows_df = confusion_rows_df if not confusion_rows_df.empty else empty_frame(CONFUSION_LONG_COLUMNS)
    reason_rows_df = reason_rows_df if not reason_rows_df.empty else empty_frame(REASON_LONG_COLUMNS)
    class_fraction_df = class_fraction_df if not class_fraction_df.empty else empty_frame(CLASS_FRACTION_COLUMNS)

    results_df[RESULT_COLUMNS].to_csv(paths["results"], index=False)
    ranked_df[RANKED_COLUMNS].to_csv(paths["ranked"], index=False)
    best_by_file_df[BEST_BY_FILE_COLUMNS].to_csv(paths["best_by_file"], index=False)
    ablation_df[ABLATION_COLUMNS].to_csv(paths["ablation"], index=False)
    ablation_by_file_df[ABLATION_COLUMNS].to_csv(paths["ablation_by_file"], index=False)
    mode_pair_delta_df[MODE_PAIR_DELTA_COLUMNS].to_csv(paths["mode_pair_delta"], index=False)
    confusion_rows_df[CONFUSION_LONG_COLUMNS].to_csv(paths["confusion"], index=False)
    reason_rows_df[REASON_LONG_COLUMNS].to_csv(paths["reason"], index=False)
    class_fraction_df[CLASS_FRACTION_COLUMNS].to_csv(paths["class_fraction"], index=False)

    top_outputs = write_top_config_diagnostics(
        ranked_df=ranked_df,
        results_df=results_df,
        confusion_rows_df=confusion_rows_df,
        reason_rows_df=reason_rows_df,
        class_fraction_df=class_fraction_df,
        output_root=output_root,
        top_n=n_top_configs_to_save,
    )

    best_payload: Dict[str, Any] = {
        "selected_by": "all_matched_mean_objective_then_min_objective",
        "rank_primary_subset": debug_metadata["rank_primary_subset"],
        "use_high_conf_for_ranking": debug_metadata["use_high_conf_for_ranking"],
        "scientific_notes": [
            "The transferred 3DEP labels are pseudo-reference labels, not absolute ground truth.",
            "The selected parameters are optimized against the current pseudo-reference and must be visually validated.",
            "height_only is treated as the primary baseline; density-based rules are treated as optional enhancement experiments.",
        ],
    }
    if not ranked_df.empty:
        best_row = ranked_df.iloc[0]
        config_id = best_row["config_id"]
        best_payload.update({
            "best_config_id": config_id,
            "best_config": {key: json_safe(best_row[key]) for key in ["config_id", *PARAM_COLUMNS, "rank", "n_files_success", "mean_objective_score", "min_objective_score"]},
            "tested_files": sorted(results_df[results_df["config_id"] == config_id]["h5_stem"].dropna().unique().tolist()),
            "per_file_all_matched_metrics": results_df[
                (results_df["config_id"] == config_id) & (results_df["subset_name"] == "all_matched")
            ].to_dict(orient="records"),
            "per_file_high_confidence_metrics": results_df[
                (results_df["config_id"] == config_id) & (results_df["subset_name"] == "high_confidence_reference")
            ].to_dict(orient="records"),
            "class_fraction_diagnostics": class_fraction_df[class_fraction_df["config_id"] == config_id].to_dict(orient="records"),
            "reason_count_diagnostics": reason_rows_df[reason_rows_df["config_id"] == config_id].to_dict(orient="records"),
        })
    safe_json_dump(best_payload, paths["best_config"])

    top_payload = build_top_n_configs_payload(
        ranked_df=ranked_df,
        results_df=results_df,
        class_fraction_df=class_fraction_df,
        reason_rows_df=reason_rows_df,
        top_n=n_top_configs_to_save,
    )
    if save_top_config_jsons:
        safe_json_dump(top_payload, paths["top_configs_json"])
    else:
        safe_json_dump({"top_configs": []}, paths["top_configs_json"])

    safe_json_dump({"failed_runs": failed_runs}, paths["failed"])

    debug_metadata = dict(debug_metadata)
    debug_metadata["outputs"] = {key: str(path) for key, path in {**paths, **top_outputs}.items()}
    safe_json_dump(debug_metadata, paths["metadata"])

    if not write_png:
        return
    plot_objective_vs_rank(ranked_df, output_root / "objective_vs_config_rank.png")
    plot_macro_f1_by_mode(results_df, output_root / "macro_f1_by_ablation_mode.png")
    plot_noise_recall_by_density(results_df, output_root / "noise_recall_by_density_threshold.png")
    plot_ground_recall_by_tol(results_df, output_root / "ground_recall_by_ground_tol.png")
    plot_delta_noise_recall(mode_pair_delta_df, output_root / "delta_noise_recall_vs_density_threshold.png")
    plot_delta_macro_f1(mode_pair_delta_df, output_root / "delta_macro_f1_vs_density_threshold.png")
    plot_objective_heatmap_tol_vs_maxhag(results_df, output_root / "objective_heatmap_tol_vs_maxhag.png")
    for _, row in best_by_file_df.iterrows():
        h5_stem = str(row["h5_stem"])
        config_id = str(row["config_id"])
        plot_best_confusion_matrix(confusion_rows_df, h5_stem, config_id, output_root / f"best_confusion_matrix_{h5_stem}.png")
        plot_best_class_fraction(class_fraction_df, h5_stem, config_id, output_root / f"pred_vs_reference_class_fraction_{h5_stem}.png")
        plot_best_reason_counts(reason_rows_df, h5_stem, config_id, output_root / f"best_config_reason_counts_{h5_stem}.png")


def maybe_write_best_laz(
    ranked_df: pd.DataFrame,
    input_pairs: Sequence[Dict[str, Path]],
    output_root: Path,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    if ranked_df.empty:
        return {"requested": True, "status": "skipped_no_ranked_results"}
    writer_output_root = output_root / "best_laz_export"
    ensure_dir(writer_output_root)
    best_row = ranked_df.iloc[0]
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


def print_ranked_summary(ranked_df: pd.DataFrame, best_by_file_df: pd.DataFrame, top_n: int) -> None:
    print("=" * 100)
    print("Sweep summary")
    print("=" * 100)
    if ranked_df.empty:
        print("No successful ranked configs.")
        return
    for _, row in ranked_df.head(top_n).iterrows():
        print(
            f"Rank {int(row['rank'])} | {row['config_id']} | mode={row['classifier_mode']} | "
            f"snr={row['GROUND_SNR_MIN']} | grid={row['GRID_RES_M']} | pct={row['GROUND_CELL_PERCENTILE']} | "
            f"tol={row['GROUND_RESID_TOL_M']} | max_hag={row['NOISE_HAG_MAX_M']} | density={row['NOISE_DENSITY_MAX_PTS_M3']}"
        )
        print(
            f"  mean_obj={float(row['mean_objective_score']):.4f}, "
            f"min_obj={float(row['min_objective_score']):.4f}, "
            f"mean_macro_f1={float(row['mean_macro_f1']):.4f}, "
            f"mean_noise_recall={float(row['mean_class_7_recall']):.4f}"
        )
    if not best_by_file_df.empty:
        print("Best config by file:")
        for _, row in best_by_file_df.iterrows():
            print(
                f"  file={row['h5_stem']} config={row['config_id']} global_rank={int(row['global_rank'])} "
                f"obj={float(row['objective_score']):.4f} macro={float(row['macro_f1']):.4f}"
            )


def print_output_summary(output_root: Path) -> None:
    paths = results_output_paths(output_root)
    print("=" * 100)
    print("Output files")
    print("=" * 100)
    for name in [
        "results",
        "ranked",
        "best_config",
        "top_configs_json",
        "best_by_file",
        "ablation",
        "ablation_by_file",
        "mode_pair_delta",
        "confusion",
        "reason",
        "class_fraction",
        "top_all",
        "top_high",
        "top_console_summary",
        "failed",
        "metadata",
    ]:
        print(f"  - {paths[name]}")


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

    RUN_MODE = "refined"  # options: quick, refined, full
    WRITE_TOP_CONFIG_CONSOLE_SUMMARY = True
    PRINT_TOP_N_CONFIGS = 20
    RANK_PRIMARY_SUBSET = "all_matched"
    USE_HIGH_CONF_FOR_RANKING = False
    SAVE_TOP_CONFIG_JSONS = True
    N_TOP_CONFIGS_TO_SAVE = 10
    INCLUDE_PRE_GROUND_ABLATION = False

    WRITE_DIAGNOSTIC_PNG = True
    WRITE_BEST_LAZ = False
    PRINT_EVERY_CONFIG = False
    PRINT_EVERY_N_CONFIGS = 25
    PRINT_ONLY_IMPROVEMENTS = True
    VERBOSE_PER_FILE_PROGRESS = False
    RESUME_FROM_EXISTING_RESULTS = True

    MAX_CONFIGS = None
    RANDOM_SUBSET_CONFIGS = False
    RANDOM_SEED = 42

    OUTPUT_ROOT = Path(f"./outputs/debug_casals_refh_classifier_params_{RUN_MODE}")
    base_config = build_base_config(output_root=OUTPUT_ROOT)
    parameter_grid = build_parameter_grid(
        base_config=base_config,
        search_mode=RUN_MODE,
        include_pre_ground_ablation=INCLUDE_PRE_GROUND_ABLATION,
        max_configs=MAX_CONFIGS,
        random_subset_configs=RANDOM_SUBSET_CONFIGS,
        random_seed=RANDOM_SEED,
    )

    tested_radii = [
        cfg["LOCAL_FEATURE_RADIUS_M"]
        for cfg in parameter_grid
        if density_enabled_for_config(cfg)
    ]
    ensure_dir(OUTPUT_ROOT)
    print_run_overview(
        input_pairs=INPUT_PAIRS,
        output_root=OUTPUT_ROOT,
        parameter_grid=parameter_grid,
        run_mode=RUN_MODE,
        write_diagnostic_png=WRITE_DIAGNOSTIC_PNG,
        write_best_laz=WRITE_BEST_LAZ,
        resume_from_existing=RESUME_FROM_EXISTING_RESULTS,
        include_pre_ground_ablation=INCLUDE_PRE_GROUND_ABLATION,
    )

    cached_files: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []
    for input_pair in INPUT_PAIRS:
        try:
            cached = prepare_one_file_cache(input_pair, base_config, tested_radii)
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
        print("No files prepared successfully.")

    sweep = run_parameter_sweep(
        cached_files=cached_files,
        parameter_grid=parameter_grid,
        output_root=OUTPUT_ROOT,
        resume_from_existing_results=RESUME_FROM_EXISTING_RESULTS,
        print_every_config=PRINT_EVERY_CONFIG,
        print_every_n_configs=PRINT_EVERY_N_CONFIGS,
        print_only_improvements=PRINT_ONLY_IMPROVEMENTS,
        verbose_per_file_progress=VERBOSE_PER_FILE_PROGRESS,
    )
    failed_runs.extend([item for item in sweep["failed_runs"] if item not in failed_runs])

    results_df = pd.DataFrame(sweep["result_rows"], columns=RESULT_COLUMNS)
    reason_rows_df = pd.DataFrame(sweep["reason_rows"], columns=REASON_LONG_COLUMNS)
    confusion_rows_df = pd.DataFrame(sweep["confusion_rows"], columns=CONFUSION_LONG_COLUMNS)
    class_fraction_df = pd.DataFrame(sweep["class_fraction_rows"], columns=CLASS_FRACTION_COLUMNS)
    ranked_df, best_by_file_df = rank_and_select_best_configs(
        results_df=results_df,
        rank_primary_subset=RANK_PRIMARY_SUBSET,
        use_high_conf_for_ranking=USE_HIGH_CONF_FOR_RANKING,
    )
    ablation_df = build_ablation_summary(results_df, by_file=False)
    ablation_by_file_df = build_ablation_summary(results_df, by_file=True)
    mode_pair_delta_df = build_mode_pair_delta_vs_height_only(results_df)

    debug_metadata: Dict[str, Any] = {
        "runtime_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_pairs": json_safe(INPUT_PAIRS),
        "base_config": json_safe(base_config),
        "parameter_grid_size": int(len(parameter_grid)),
        "prepared_files": [cached["h5_stem"] for cached in cached_files],
        "failed_file_preparations": int(sum(1 for item in failed_runs if item["config_id"] == "__prepare__")),
        "failed_config_runs": int(sum(1 for item in failed_runs if item["config_id"] != "__prepare__")),
        "run_mode": RUN_MODE,
        "write_diagnostic_png": bool(WRITE_DIAGNOSTIC_PNG),
        "write_best_laz": bool(WRITE_BEST_LAZ),
        "resume_from_existing_results": bool(RESUME_FROM_EXISTING_RESULTS),
        "rank_primary_subset": RANK_PRIMARY_SUBSET,
        "use_high_conf_for_ranking": bool(USE_HIGH_CONF_FOR_RANKING),
        "save_top_config_jsons": bool(SAVE_TOP_CONFIG_JSONS),
        "n_top_configs_to_save": int(N_TOP_CONFIGS_TO_SAVE),
        "include_pre_ground_ablation": bool(INCLUDE_PRE_GROUND_ABLATION),
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
        ablation_by_file_df=ablation_by_file_df,
        mode_pair_delta_df=mode_pair_delta_df,
        confusion_rows_df=confusion_rows_df,
        reason_rows_df=reason_rows_df,
        class_fraction_df=class_fraction_df,
        failed_runs=failed_runs,
        debug_metadata=debug_metadata,
        write_png=WRITE_DIAGNOSTIC_PNG,
        save_top_config_jsons=SAVE_TOP_CONFIG_JSONS,
        n_top_configs_to_save=N_TOP_CONFIGS_TO_SAVE,
    )

    if WRITE_TOP_CONFIG_CONSOLE_SUMMARY:
        print_ranked_summary(ranked_df, best_by_file_df, PRINT_TOP_N_CONFIGS)
    print(
        f"Failed preparations: {debug_metadata['failed_file_preparations']} | "
        f"Failed config runs: {debug_metadata['failed_config_runs']}"
    )
    print_output_summary(OUTPUT_ROOT)
    print(f"Wrote debug outputs to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
