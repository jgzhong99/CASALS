#!/usr/bin/env python3
"""
Summarize balanced-vs-strict cleaned-target tradeoffs.

Inputs
------
- three_scene_multi_target_evaluation.json
- cleaned_modified_raw_targets_summary.json

Outputs
-------
- CSV table with the key per-scene metrics
- Markdown report with a short recommendation section
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "classify_casals_h5_to_3dep_like_las"
EVAL_JSON = OUT_DIR / "three_scene_multi_target_evaluation.json"
TARGET_SUMMARY_JSON = OUT_DIR / "cleaned_modified_raw_targets" / "cleaned_modified_raw_targets_summary.json"
OUT_CSV = OUT_DIR / "cleaned_modified_raw_tradeoff_summary.csv"
OUT_MD = OUT_DIR / "cleaned_modified_raw_tradeoff_summary.md"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_f1(block: Dict[str, Any], cls: int) -> float | None:
    metrics = block.get("metrics", {})
    cls_metrics = metrics.get(str(cls)) or metrics.get(cls)
    if cls_metrics is None:
        return None
    return float(cls_metrics["f1"])


def build_rows(eval_data: Dict[str, Any], target_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene_name in sorted(eval_data):
        ev = eval_data[scene_name]
        ts = target_summary[scene_name]
        rows.append({
            "scene": scene_name,
            "modified_min_f1": float(ev["modified_raw_transfer_target"]["min_f1"]),
            "balanced_min_f1": float(ev["cleaned_modified_raw_target_balanced"]["min_f1"]),
            "strict_min_f1": float(ev["cleaned_modified_raw_target_strict"]["min_f1"]),
            "balanced_class1_f1": metric_f1(ev["cleaned_modified_raw_target_balanced"], 1),
            "balanced_class2_f1": metric_f1(ev["cleaned_modified_raw_target_balanced"], 2),
            "strict_class1_f1": metric_f1(ev["cleaned_modified_raw_target_strict"], 1),
            "strict_class2_f1": metric_f1(ev["cleaned_modified_raw_target_strict"], 2),
            "balanced_promoted_count": int(ts["cleaned_rule_balanced"]["promoted_count"]),
            "strict_promoted_count": int(ts["cleaned_rule_strict"]["promoted_count"]),
            "balanced_promotion_precision_to_stable2": float(ts["promotion_precision_to_stable2_balanced"]),
            "strict_promotion_precision_to_stable2": float(ts["promotion_precision_to_stable2_strict"]),
            "balanced_false_promotions": int(ts["promoted_stable_class_counts_balanced"].get("1", 0)),
            "strict_false_promotions": int(ts["promoted_stable_class_counts_strict"].get("1", 0)),
        })
    return rows


def recommend_default(df: pd.DataFrame) -> str:
    balanced_gain = (df["balanced_min_f1"] - df["modified_min_f1"]).mean()
    strict_gain = (df["strict_min_f1"] - df["modified_min_f1"]).mean()
    balanced_precision = df["balanced_promotion_precision_to_stable2"].mean()
    strict_precision = df["strict_promotion_precision_to_stable2"].mean()

    if (balanced_gain - strict_gain) > 0.01 and balanced_precision > 0.97:
        return "balanced"
    if strict_precision > balanced_precision + 0.01:
        return "strict"
    return "balanced"


def build_markdown(df: pd.DataFrame, default_mode: str) -> str:
    lines: List[str] = []
    lines.append("# Cleaned Modified Raw Tradeoff Summary")
    lines.append("")
    lines.append("## Recommendation")
    if default_mode == "balanced":
        lines.append("- Default mode: `balanced`.")
        lines.append("- Reason: it gives the larger recovery in `pair2_de` while keeping promotion precision very high.")
        lines.append("- Use `strict` only when you want a slightly more conservative version for scenes like `pair3_nc`.")
    else:
        lines.append("- Default mode: `strict`.")
        lines.append("- Reason: it reduces false promotions enough to outweigh the smaller recovery.")
    lines.append("")
    lines.append("## Per-Scene Table")
    lines.append("")
    lines.append(df.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    lines.append("## Notes")
    lines.append("- `promotion_precision_to_stable2` is measured only where the stable-surface target is available.")
    lines.append("- `false_promotions` counts promoted points that align with stable class 1 instead of stable class 2.")
    return "\n".join(lines)


def main() -> None:
    eval_data = load_json(EVAL_JSON)
    target_summary = load_json(TARGET_SUMMARY_JSON)

    df = pd.DataFrame(build_rows(eval_data, target_summary))
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")

    default_mode = recommend_default(df)
    OUT_MD.write_text(build_markdown(df, default_mode), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"[INFO] Default mode recommendation: {default_mode}")
    print(f"[INFO] Wrote CSV: {OUT_CSV}")
    print(f"[INFO] Wrote Markdown: {OUT_MD}")


if __name__ == "__main__":
    main()
