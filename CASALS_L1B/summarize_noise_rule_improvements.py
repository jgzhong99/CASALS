#!/usr/bin/env python3
"""
Summarize the effect of the density-aware noise rules in the main classifier.

This compares:
- baseline_without_new_noise_rules
- current_density_aware_noise

against the stable-surface target on the three local scenes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import classify_casals_h5_to_3dep_like_las as classifier
import evaluate_classify_casals_h5_to_3dep_like_las as evaluator


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "classify_casals_h5_to_3dep_like_las"
OUT_JSON = OUT_DIR / "noise_rule_improvement_summary.json"
OUT_CSV = OUT_DIR / "noise_rule_improvement_summary.csv"
OUT_MD = OUT_DIR / "noise_rule_improvement_summary.md"


BASELINE_OVERRIDES = {
    "rule_positive_dense_nonground_lower_m": 1e9,
    "rule_positive_dense_nonground_upper_m": 1e9,
    "rule_positive_dense_nonground_cell_min": 10**9,
    "rule_positive_dense_nonground_support_ratio_min": 2.0,
    "rule_positive_dense_nonground_nearest_support_max_m": 0.0,
    "rule_negative_sparse_noise_m": -1e9,
    "rule_negative_sparse_support_ratio_max": -1.0,
    "rule_negative_sparse_nearest_support_min_m": 1e9,
    "rule_negative_sparse_snr_max": -1.0,
}


def evaluate_mode(scene_name: str, scene_cfg: Dict[str, Path], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(classifier.CONFIG)
    cfg["h5_path"] = str(scene_cfg["h5"])
    cfg["output_dir"] = str(ROOT / "outputs" / "tmp_noise_summary" / scene_name)
    cfg.update(overrides)
    _, artifacts = classifier.classify_h5(cfg)
    point_index, y_true, stable_info = evaluator.build_stable_surface_target(
        scene_cfg=scene_cfg,
        artifacts=artifacts,
        cell_m=float(evaluator.CONFIG["ground_surface_cell_m"]),
        sample_n=None,
    )
    eval_result = evaluator.evaluate_predictions(y_true, artifacts.classification[point_index])
    return {
        "evaluation": eval_result,
        "classifier_mode": artifacts.classifier_mode,
        "stable_surface_target_info": stable_info,
    }


def get_f1(block: Dict[str, Any], cls: int) -> float:
    metrics = block["evaluation"]["metrics"]
    return float((metrics.get(str(cls)) or metrics.get(cls))["f1"])


def get_precision(block: Dict[str, Any], cls: int) -> float:
    metrics = block["evaluation"]["metrics"]
    return float((metrics.get(str(cls)) or metrics.get(cls))["precision"])


def get_recall(block: Dict[str, Any], cls: int) -> float:
    metrics = block["evaluation"]["metrics"]
    return float((metrics.get(str(cls)) or metrics.get(cls))["recall"])


def build_rows(results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene_name in sorted(results):
        base = results[scene_name]["baseline_without_new_noise_rules"]
        curr = results[scene_name]["current_density_aware_noise"]
        rows.append({
            "scene": scene_name,
            "baseline_class1_f1": get_f1(base, 1),
            "current_class1_f1": get_f1(curr, 1),
            "delta_class1_f1": get_f1(curr, 1) - get_f1(base, 1),
            "baseline_class2_f1": get_f1(base, 2),
            "current_class2_f1": get_f1(curr, 2),
            "delta_class2_f1": get_f1(curr, 2) - get_f1(base, 2),
            "baseline_class7_precision": get_precision(base, 7),
            "current_class7_precision": get_precision(curr, 7),
            "delta_class7_precision": get_precision(curr, 7) - get_precision(base, 7),
            "baseline_class7_recall": get_recall(base, 7),
            "current_class7_recall": get_recall(curr, 7),
            "delta_class7_recall": get_recall(curr, 7) - get_recall(base, 7),
            "baseline_class7_f1": get_f1(base, 7),
            "current_class7_f1": get_f1(curr, 7),
            "delta_class7_f1": get_f1(curr, 7) - get_f1(base, 7),
            "baseline_min_f1": float(base["evaluation"]["min_f1"]),
            "current_min_f1": float(curr["evaluation"]["min_f1"]),
            "delta_min_f1": float(curr["evaluation"]["min_f1"]) - float(base["evaluation"]["min_f1"]),
        })
    return rows


def build_markdown(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Noise Rule Improvement Summary")
    lines.append("")
    lines.append("## New Rules")
    lines.append("- Rescue dense moderate positive-height local outliers back to `class 1`.")
    lines.append("- Add sparse low-support below-ground points to `class 7`.")
    lines.append("")
    lines.append("## Scene Table")
    lines.append("")
    lines.append(df.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- The main gain should appear in `class 7` precision and F1, especially on `pair3_nc`.")
    lines.append("- Small losses in `pair1_md` / `pair2_de` are acceptable only if they stay minor while the `pair3_nc` noise gain is materially larger.")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}
    for scene_name, scene_cfg in evaluator.SCENES.items():
        results[scene_name] = {
            "baseline_without_new_noise_rules": evaluate_mode(scene_name, scene_cfg, BASELINE_OVERRIDES),
            "current_density_aware_noise": evaluate_mode(scene_name, scene_cfg, {}),
        }

    df = pd.DataFrame(build_rows(results))
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    OUT_MD.write_text(build_markdown(df), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"[INFO] Wrote JSON: {OUT_JSON}")
    print(f"[INFO] Wrote CSV: {OUT_CSV}")
    print(f"[INFO] Wrote Markdown: {OUT_MD}")


if __name__ == "__main__":
    main()
