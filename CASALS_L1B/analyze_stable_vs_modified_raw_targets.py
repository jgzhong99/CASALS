#!/usr/bin/env python3
"""
Compare the stable-surface target with the modified raw transfer target.

Purpose
-------
- Keep the `pair2_de` special handling explicit: ignore its 2101 raw pseudo
  class-2 points in the modified raw target.
- Quantify where the modified raw target still disagrees with the more stable
  empirical-dz + 3DEP-ground-surface target.
- Summarize a few directly interpretable features for the disagreement groups.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import classify_casals_h5_to_3dep_like_las as classifier
import evaluate_classify_casals_h5_to_3dep_like_las as evaluator


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "outputs" / "classify_casals_h5_to_3dep_like_las" / "stable_vs_modified_raw_target_analysis.json"


FEATURE_COLUMNS = [
    "refh_snr",
    "refh_amp",
    "bg_mean",
    "bg_std",
    "refh_thres",
]

AUDIT_COLUMNS = [
    "dtm_residual_m",
    "nearest_support_xy_distance_m",
    "nearest_ground_seed_xy_distance_m",
    "local_support_ratio_10m",
    "local_ground_seed_ratio_10m",
]


def summarize_series(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "p10": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
        }
    return {
        "n": int(values.size),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def build_confusion_table(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    table = pd.crosstab(df["stable_target"], df["modified_raw_target"], dropna=False)
    out: Dict[str, Dict[str, int]] = {}
    for stable_cls in table.index.tolist():
        out[str(int(stable_cls))] = {
            str(int(raw_cls)): int(table.loc[stable_cls, raw_cls])
            for raw_cls in table.columns.tolist()
        }
    return out


def build_scene_analysis(scene_name: str, scene_cfg: Dict[str, Path]) -> Dict[str, Any]:
    cfg = dict(classifier.CONFIG)
    cfg["h5_path"] = str(scene_cfg["h5"])
    cfg["output_dir"] = str(ROOT / "outputs" / "tmp_eval")
    point_data, artifacts = classifier.classify_h5(cfg)

    stable_idx, stable_target, stable_info = evaluator.build_stable_surface_target(
        scene_cfg=scene_cfg,
        artifacts=artifacts,
        cell_m=float(evaluator.CONFIG["ground_surface_cell_m"]),
        sample_n=evaluator.CONFIG["sample_n"],
    )
    raw_idx, raw_target = evaluator.build_modified_raw_transfer_target(
        scene_name=scene_name,
        scene_cfg=scene_cfg,
        sample_n=evaluator.CONFIG["sample_n"],
    )

    stable_df = pd.DataFrame({"point_index": stable_idx, "stable_target": stable_target})
    raw_df = pd.DataFrame({"point_index": raw_idx, "modified_raw_target": raw_target})
    merged = stable_df.merge(raw_df, on="point_index", how="inner", validate="one_to_one")
    idx = merged["point_index"].to_numpy(dtype=np.int64)

    feature_df = pd.read_parquet(scene_cfg["feature_parquet"], columns=["point_index", *FEATURE_COLUMNS]).set_index("point_index")
    feature_df = feature_df.loc[idx].reset_index(drop=True)

    merged["classifier_prediction"] = artifacts.classification[idx].astype(np.uint8)
    merged["dtm_residual_m"] = artifacts.dtm_residual_m[idx].astype(np.float32)
    merged["nearest_support_xy_distance_m"] = artifacts.nearest_support_xy_distance_m[idx].astype(np.float32)
    merged["nearest_ground_seed_xy_distance_m"] = artifacts.nearest_ground_seed_xy_distance_m[idx].astype(np.float32)
    merged["local_support_ratio_10m"] = artifacts.local_support_ratio_10m[idx].astype(np.float32)
    merged["local_ground_seed_ratio_10m"] = artifacts.local_ground_seed_ratio_10m[idx].astype(np.float32)
    for col in FEATURE_COLUMNS:
        merged[col] = feature_df[col].to_numpy()

    disagreements = merged[merged["stable_target"] != merged["modified_raw_target"]].copy()
    disagreement_counts = (
        disagreements.groupby(["stable_target", "modified_raw_target"], sort=True)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )

    group_summaries: Dict[str, Any] = {}
    for stable_cls, raw_cls in [(2, 1), (1, 1), (2, 2), (7, 7), (1, 7), (2, 7)]:
        sub = merged[(merged["stable_target"] == stable_cls) & (merged["modified_raw_target"] == raw_cls)]
        if sub.empty:
            continue
        key = f"stable_{stable_cls}_raw_{raw_cls}"
        group_summaries[key] = {
            "n": int(len(sub)),
            "classifier_prediction_counts": {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(sub["classifier_prediction"], return_counts=True))
            },
            "features": {
                col: summarize_series(sub[col].to_numpy())
                for col in [*FEATURE_COLUMNS, *AUDIT_COLUMNS]
            },
        }

    return {
        "scene": scene_name,
        "stable_surface_target_info": stable_info,
        "n_common_eval_points": int(len(merged)),
        "stable_vs_modified_raw_confusion": build_confusion_table(merged),
        "disagreement_counts_sorted": disagreement_counts.to_dict(orient="records"),
        "group_summaries": group_summaries,
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    results = {
        scene_name: build_scene_analysis(scene_name, scene_cfg)
        for scene_name, scene_cfg in evaluator.SCENES.items()
    }
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[INFO] Wrote JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
