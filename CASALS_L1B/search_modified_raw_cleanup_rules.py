#!/usr/bin/env python3
"""
Search simple cleanup rules for the modified raw transfer target.

Scope
-----
- Keep the existing modified raw target behavior, including ignoring the 2101
  raw pseudo class-2 points in `pair2_de`.
- Search a small family of interpretable rules that only *promote*
  `modified_raw_target == 1` to class 2 when a point also looks like stable
  3DEP ground relative to the morphology-guided CASALS DTM evidence.
- Score the cleaned target against the stable-surface target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import classify_casals_h5_to_3dep_like_las as classifier
import evaluate_classify_casals_h5_to_3dep_like_las as evaluator


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "outputs" / "classify_casals_h5_to_3dep_like_las" / "modified_raw_cleanup_rule_search.json"


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    labels = [cls for cls in [1, 2, 7] if np.any(y_true == cls)]
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics = {
        int(cls): {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for cls, p, r, f, s in zip(labels, prec, rec, f1, sup)
    }
    return {
        "metrics": metrics,
        "min_f1": float(min(m["f1"] for m in metrics.values())),
        "macro_f1": float(np.mean([m["f1"] for m in metrics.values()])),
    }


def build_scene_table(scene_name: str, scene_cfg: Dict[str, Path]) -> Dict[str, np.ndarray]:
    cfg = dict(classifier.CONFIG)
    cfg["h5_path"] = str(scene_cfg["h5"])
    cfg["output_dir"] = str(ROOT / "outputs" / "tmp_eval")
    point_data, artifacts = classifier.classify_h5(cfg)

    stable_idx, stable_target, _ = evaluator.build_stable_surface_target(
        scene_cfg=scene_cfg,
        artifacts=artifacts,
        cell_m=float(evaluator.CONFIG["ground_surface_cell_m"]),
        sample_n=None,
    )
    raw_idx, raw_target = evaluator.build_modified_raw_transfer_target(
        scene_name=scene_name,
        scene_cfg=scene_cfg,
        sample_n=None,
    )
    merged = pd.DataFrame({"point_index": stable_idx, "stable": stable_target}).merge(
        pd.DataFrame({"point_index": raw_idx, "raw": raw_target}),
        on="point_index",
        how="inner",
        validate="one_to_one",
    )
    idx = merged["point_index"].to_numpy(dtype=np.int64)
    return {
        "stable": merged["stable"].to_numpy(dtype=np.uint8),
        "raw": merged["raw"].to_numpy(dtype=np.uint8),
        "abs_dtm_resid_m": np.abs(artifacts.dtm_residual_m[idx].astype(np.float32)),
        "nearest_support_xy_distance_m": artifacts.nearest_support_xy_distance_m[idx].astype(np.float32),
        "nearest_ground_seed_xy_distance_m": artifacts.nearest_ground_seed_xy_distance_m[idx].astype(np.float32),
        "local_support_ratio_10m": artifacts.local_support_ratio_10m[idx].astype(np.float32),
        "local_ground_seed_ratio_10m": artifacts.local_ground_seed_ratio_10m[idx].astype(np.float32),
        "refh_snr": point_data.snr[idx].astype(np.float32),
    }


def score_rule(
    scene_tables: Dict[str, Dict[str, np.ndarray]],
    family_name: str,
    params: Dict[str, float],
    promote_mask_fn: Callable[[Dict[str, np.ndarray], Dict[str, float]], np.ndarray],
) -> Dict[str, Any]:
    per_scene: Dict[str, Any] = {}
    min_f1 = 1.0
    macro_f1_mean = 0.0

    for scene_name, table in scene_tables.items():
        y_true = table["stable"]
        y_pred = table["raw"].copy()
        promote_mask = (y_pred == 1) & promote_mask_fn(table, params)
        y_pred[promote_mask] = 2

        eval_result = evaluate_predictions(y_true, y_pred)
        promoted_true = y_true[promote_mask]
        promoted_counts = {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(promoted_true, return_counts=True))
        } if promoted_true.size else {}
        promote_precision_to_stable2 = float(np.mean(promoted_true == 2)) if promoted_true.size else float("nan")

        per_scene[scene_name] = {
            "evaluation": eval_result,
            "promoted_count": int(promote_mask.sum()),
            "promoted_stable_class_counts": promoted_counts,
            "promote_precision_to_stable2": promote_precision_to_stable2,
        }
        min_f1 = min(min_f1, eval_result["min_f1"])
        macro_f1_mean += eval_result["macro_f1"]

    return {
        "family": family_name,
        "params": params,
        "overall_min_f1": float(min_f1),
        "overall_mean_macro_f1": float(macro_f1_mean / len(scene_tables)),
        "per_scene": per_scene,
    }


def search_rules(scene_tables: Dict[str, Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    families: List[Tuple[str, Iterable[Dict[str, float]], Callable[[Dict[str, np.ndarray], Dict[str, float]], np.ndarray]]] = [
        (
            "dtm_plus_nearest_support",
            (
                {"abs_dtm_resid_max_m": abs_max, "nearest_support_max_m": near_max}
                for abs_max in [0.35, 0.5, 0.75, 1.0]
                for near_max in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
            ),
            lambda t, p: (
                (t["abs_dtm_resid_m"] <= p["abs_dtm_resid_max_m"])
                & (t["nearest_support_xy_distance_m"] <= p["nearest_support_max_m"])
            ),
        ),
        (
            "dtm_plus_nearest_ground_seed",
            (
                {"abs_dtm_resid_max_m": abs_max, "nearest_ground_seed_max_m": near_max}
                for abs_max in [0.35, 0.5, 0.75, 1.0]
                for near_max in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
            ),
            lambda t, p: (
                (t["abs_dtm_resid_m"] <= p["abs_dtm_resid_max_m"])
                & (t["nearest_ground_seed_xy_distance_m"] <= p["nearest_ground_seed_max_m"])
            ),
        ),
        (
            "dtm_plus_nearest_support_plus_snr",
            (
                {
                    "abs_dtm_resid_max_m": abs_max,
                    "nearest_support_max_m": near_max,
                    "refh_snr_min": snr_min,
                }
                for abs_max in [0.35, 0.5, 0.75, 1.0]
                for near_max in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
                for snr_min in [1.8, 2.0, 2.2, 2.5, 3.0]
            ),
            lambda t, p: (
                (t["abs_dtm_resid_m"] <= p["abs_dtm_resid_max_m"])
                & (t["nearest_support_xy_distance_m"] <= p["nearest_support_max_m"])
                & (t["refh_snr"] >= p["refh_snr_min"])
            ),
        ),
        (
            "dtm_plus_nearest_support_plus_local_support_ratio",
            (
                {
                    "abs_dtm_resid_max_m": abs_max,
                    "nearest_support_max_m": near_max,
                    "local_support_ratio_min": ratio_min,
                }
                for abs_max in [0.35, 0.5, 0.75, 1.0]
                for near_max in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
                for ratio_min in [0.0, 0.001, 0.003, 0.005, 0.01]
            ),
            lambda t, p: (
                (t["abs_dtm_resid_m"] <= p["abs_dtm_resid_max_m"])
                & (t["nearest_support_xy_distance_m"] <= p["nearest_support_max_m"])
                & (t["local_support_ratio_10m"] >= p["local_support_ratio_min"])
            ),
        ),
    ]

    for family_name, param_iter, promote_mask_fn in families:
        family_results = [
            score_rule(scene_tables, family_name, params, promote_mask_fn)
            for params in param_iter
        ]
        family_results.sort(key=lambda r: (r["overall_min_f1"], r["overall_mean_macro_f1"]), reverse=True)
        results.extend(family_results[:5])

    results.sort(key=lambda r: (r["overall_min_f1"], r["overall_mean_macro_f1"]), reverse=True)
    return results


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    scene_tables = {
        scene_name: build_scene_table(scene_name, scene_cfg)
        for scene_name, scene_cfg in evaluator.SCENES.items()
    }
    top_rules = search_rules(scene_tables)
    summary = {
        "script": "search_modified_raw_cleanup_rules.py",
        "note": "modified raw target keeps pair2_de raw pseudo class-2 ignored before cleanup-rule search",
        "top_rules": top_rules,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
