"""Point-aligned evaluation for CASALS refh classifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import laspy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .classification import (
    CLASS_NAME_MAP,
    DEFAULT_CONFIG,
    LABEL_ORDER,
    collect_class_counts,
    maybe_quantiles,
    normalize_config,
)


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
        "withheld": np.asarray(las.withheld, dtype=bool),
        **point_source(las),
    }
    for name in [
        "point_index",
        "longitude",
        "latitude",
        "transfer_status",
        "nearest3dep_dist_m",
        "class_vote_ratio",
        "refh_original_m",
        "z_original_m",
    ]:
        if name in dims:
            ref[name] = np.asarray(las[name])
    if "z_original_m" in ref:
        ref["refh_original_m"] = ref["z_original_m"]
    return ref

def point_source(las) -> dict:
    for vlr in las.header.vlrs:
        if vlr.user_id == "CASALS" and vlr.record_id == 1:
            return json.loads(vlr.record_data.decode("utf-8"))
    return {}


def align_prediction_to_reference(prediction, reference, config):
    # Indices identify records only within the same original H5.
    src = prediction.get("source_h5")
    ref_src = reference.get("source_h5")
    if not src or not ref_src or Path(src).resolve() != Path(ref_src).resolve():
        raise ValueError("H5 provenance missing or different; evaluation unavailable.")
    pi = np.asarray(prediction["point_index"], dtype=np.int64)
    ri = np.asarray(reference.get("point_index", np.arange(len(reference["classification"]))), dtype=np.int64)
    if len(np.unique(pi)) != len(pi) or len(np.unique(ri)) != len(ri):
        raise ValueError("Duplicate point_index; identity ambiguous.")
    order = np.argsort(pi)
    pos = np.searchsorted(pi[order], ri)
    if np.any(pos >= len(pi)) or np.any(pi[order][np.minimum(pos, len(pi)-1)] != ri):
        raise ValueError("Reference indices absent from prediction.")
    idx = order[pos]
    checks = {}
    for name, tolerance in (("longitude", config["ROW_ALIGN_LONLAT_TOL_DEG"]),
                            ("latitude", config["ROW_ALIGN_LONLAT_TOL_DEG"]),
                            ("refh_original_m", 0.001)):
        if name not in prediction or name not in reference:
            raise ValueError(f"Original {name} missing; coordinate identity unverified.")
        left = np.asarray(prediction[name], dtype=float)[idx]
        right = np.asarray(reference[name], dtype=float)
        equal = np.isclose(left, right, rtol=0, atol=tolerance, equal_nan=True)
        if not np.all(equal):
            raise ValueError(f"Coordinate identity mismatch: {name}.")
        checks[name] = "all_records_verified"
    n = len(pi)
    valid = np.zeros(n, dtype=np.uint8)
    finite = np.isfinite(reference["longitude"]) & np.isfinite(reference["latitude"]) & np.isfinite(reference["refh_original_m"])
    valid[idx] = finite & ~np.asarray(reference.get("withheld", np.zeros(len(ri))), dtype=bool)
    gt = np.zeros(n, dtype=np.uint8)
    gt[idx] = reference["classification"]
    result = dict(eval_match_valid=valid, eval_gt_class_raw=gt,
                  alignment_method="point_index_and_original_coordinates", alignment_checks=checks)
    for field, output in (("transfer_status", "reference_transfer_status"),
                          ("nearest3dep_dist_m", "reference_nearest3dep_dist_m"),
                          ("class_vote_ratio", "reference_class_vote_ratio")):
        result[output] = None
        if field in reference:
            values = np.full(n, -1 if field == "transfer_status" else np.nan)
            values[idx] = reference[field]
            result[output] = values
    return result


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

def evaluate_classification(
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

    base_mask = matched.copy()
    if bool(config["EVAL_REQUIRE_VALID_DTM"]):
        base_mask &= np.asarray(dtm_sample_valid, dtype=bool)
    if bool(config["EVAL_IGNORE_REFERENCE_NOISE"]):
        base_mask &= np.asarray(eval_gt_class, dtype=np.uint8) != 7
    if config["EVAL_REQUIRE_TRANSFER_STATUS"] is not None and reference_transfer_status is not None:
        required = config["EVAL_REQUIRE_TRANSFER_STATUS"]
        required_values = [int(required)] if np.isscalar(required) else [int(v) for v in required]
        base_mask &= np.isin(np.asarray(reference_transfer_status), required_values)

    statuses = np.full(matched.size, -1) if reference_transfer_status is None else np.asarray(reference_transfer_status)
    if reference_nearest3dep_dist_m is not None:
        base_mask &= np.isfinite(reference_nearest3dep_dist_m) & (reference_nearest3dep_dist_m >= 0)
    if reference_class_vote_ratio is not None:
        base_mask &= np.isfinite(reference_class_vote_ratio) & (reference_class_vote_ratio > 0) & (reference_class_vote_ratio <= 1)
    subset_masks = {"strict": base_mask & (statuses == 1),
                    "strict_plus_weak": base_mask & np.isin(statuses, [1, 2])}
    population = {"total_points": int(matched.size), "aligned_points": int(matched.sum()),
                  "valid_reference_points": int(np.count_nonzero(base_mask & (statuses == 1))),
                  "strict_plus_weak_reference_points": int(np.count_nonzero(base_mask & np.isin(statuses, [1, 2]))),
                  "match_status_counts": {str(int(v)): int(np.count_nonzero(statuses == v)) for v in np.unique(statuses)},
                  "metric_semantics": "pseudo-reference agreement",
                  "class_mapping": "vegetation/building and other non-ground/non-noise classes merged into unclassified",
                  "independent_accuracy_claim": False}

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
            subset_result.update(status="evaluation unavailable", reason="missing reliable transfer status" if reference_transfer_status is None else "no eligible reference points")
            evaluation_summary_rows.append({**subset_result, **population, "coverage": 0.0})
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
            **population, "coverage": float(y_true.size / matched.size) if matched.size else 0.0,
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
            **population, "status": "ok", "coverage": float(y_true.size / matched.size) if matched.size else 0.0,
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

        if subset_name == "strict":
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
        "primary_mask": subset_masks["strict"],
        "subset_results": subset_results,
        "evaluation_summary_rows": evaluation_summary_rows,
        "primary_report_df": primary_report_df,
        "primary_confusion_df": primary_confusion_df,
        "primary_metrics": primary_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an existing CASALS refh prediction against an explicitly aligned reference.")
    parser.add_argument("--prediction", type=Path, required=True, help="Classified CASALS LAS/LAZ with point_index and prediction classes.")
    parser.add_argument("--reference", type=Path, required=True, help="Reference LAS/LAZ; transferred 3DEP labels remain pseudo-reference.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classification/evaluation"))
    parser.add_argument("--config", type=Path, help="Optional JSON evaluation-rule overrides.")
    args = parser.parse_args(argv)

    prediction_las = laspy.read(str(args.prediction))
    dims = set(prediction_las.point_format.extra_dimension_names)
    if "point_index" not in dims:
        parser.error("prediction LAS/LAZ must contain the point_index extra dimension")
    prediction: Dict[str, np.ndarray] = {
        **point_source(prediction_las),
        "point_index": np.asarray(prediction_las.point_index, dtype=np.int64),
        "x": np.asarray(prediction_las.x, dtype=np.float64),
        "y": np.asarray(prediction_las.y, dtype=np.float64),
    }
    if "longitude" in dims and "latitude" in dims:
        prediction["longitude"] = np.asarray(prediction_las.longitude, dtype=np.float64)
        prediction["latitude"] = np.asarray(prediction_las.latitude, dtype=np.float64)
    if "refh_original_m" in dims:
        prediction["refh_original_m"] = np.asarray(prediction_las.refh_original_m)
    reference = read_reference_labels(args.reference)
    config_overrides = {} if args.config is None else json.loads(args.config.read_text(encoding="utf-8"))
    config, _ = normalize_config({**DEFAULT_CONFIG, **config_overrides})
    alignment = align_prediction_to_reference(prediction, reference, config)
    eval_gt = map_reference_labels_to_baseline_classes(alignment["eval_gt_class_raw"])
    eval_mask = alignment["eval_match_valid"]
    dtm_valid = (
        np.asarray(prediction_las.dtm_sample_valid, dtype=bool)
        if "dtm_sample_valid" in dims
        else np.ones(len(prediction_las.points), dtype=bool)
    )
    metrics = evaluate_classification(
        pred_class_baseline=np.asarray(prediction_las.classification, dtype=np.uint8),
        eval_gt_class=eval_gt,
        eval_match_valid=eval_mask,
        dtm_sample_valid=dtm_valid,
        reference_transfer_status=alignment["reference_transfer_status"],
        reference_nearest3dep_dist_m=alignment["reference_nearest3dep_dist_m"],
        reference_class_vote_ratio=alignment["reference_class_vote_ratio"],
        config=config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics["primary_report_df"].to_csv(args.output_dir / "classification_report.csv", index=False)
    metrics["primary_confusion_df"].to_csv(args.output_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(metrics["evaluation_summary_rows"]).to_csv(args.output_dir / "evaluation_summary.csv", index=False)
    subsets = {
        name: {key: value for key, value in result.items() if key not in {"y_true", "y_pred"}}
        for name, result in metrics["subset_results"].items()
    }
    report = {
        "prediction": str(args.prediction.resolve()),
        "reference": str(args.reference.resolve()),
        "reference_type": "transferred_3dep_pseudo_reference",
        "reference_alignment_mode": reference.get("alignment_mode", "unknown"),
        "metric_semantics": "pseudo-reference agreement",
        "alignment_method": alignment["alignment_method"],
        "alignment_checks": alignment.get("alignment_checks", {}),
        "evaluation_population": {name: result.get("n_points", 0) for name, result in subsets.items()},
        "subset_results": subsets,
        "independent_accuracy_claim": False,
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(report, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
        encoding="utf-8",
    )
    print(f"Alignment: {alignment['alignment_method']}; matched={int(np.count_nonzero(eval_mask)):,}")
    for name, values in metrics["primary_metrics"].items():
        if name in {"accuracy", "macro_f1", "weighted_f1"}:
            print(f"{name}: {values:.6f}")
    return 0
