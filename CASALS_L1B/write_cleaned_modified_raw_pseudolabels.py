#!/usr/bin/env python3
"""
Write cleaned modified-raw pseudo-label artifacts for the three local scenes.

Output
------
- One parquet per scene containing:
  - point_index
  - modified_raw_target
  - cleaned_modified_raw_target_balanced
  - cleaned_modified_raw_target_strict
  - classifier_prediction
  - dtm_residual_m
  - nearest_support_xy_distance_m
  - nearest_ground_seed_xy_distance_m
  - local_support_ratio_10m
  - local_ground_seed_ratio_10m
  - optional stable_surface_target when 3DEP is available
- One summary JSON describing counts and promotion precision to the stable
  surface target where available.
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
OUT_DIR = ROOT / "outputs" / "classify_casals_h5_to_3dep_like_las" / "cleaned_modified_raw_targets"
SUMMARY_JSON = OUT_DIR / "cleaned_modified_raw_targets_summary.json"


def build_scene_artifact(scene_name: str, scene_cfg: Dict[str, Path]) -> Dict[str, Any]:
    cfg = dict(classifier.CONFIG)
    cfg["h5_path"] = str(scene_cfg["h5"])
    cfg["output_dir"] = str(ROOT / "outputs" / "tmp_eval")
    point_data, artifacts = classifier.classify_h5(cfg)

    point_index, modified_raw_target = evaluator.build_modified_raw_transfer_target(
        scene_name=scene_name,
        scene_cfg=scene_cfg,
        sample_n=evaluator.CONFIG["sample_n"],
    )
    _, cleaned_targets = evaluator.build_cleaned_modified_raw_target(
        scene_name=scene_name,
        scene_cfg=scene_cfg,
        artifacts=artifacts,
        sample_n=evaluator.CONFIG["sample_n"],
    )
    cleaned_modified_raw_target_balanced, cleaned_info_balanced = cleaned_targets["balanced"]
    cleaned_modified_raw_target_strict, cleaned_info_strict = cleaned_targets["strict"]

    df = pd.DataFrame({
        "point_index": point_index.astype(np.int64),
        "modified_raw_target": modified_raw_target.astype(np.uint8),
        "cleaned_modified_raw_target_balanced": cleaned_modified_raw_target_balanced.astype(np.uint8),
        "cleaned_modified_raw_target_strict": cleaned_modified_raw_target_strict.astype(np.uint8),
        "classifier_prediction": artifacts.classification[point_index].astype(np.uint8),
        "dtm_residual_m": artifacts.dtm_residual_m[point_index].astype(np.float32),
        "nearest_support_xy_distance_m": artifacts.nearest_support_xy_distance_m[point_index].astype(np.float32),
        "nearest_ground_seed_xy_distance_m": artifacts.nearest_ground_seed_xy_distance_m[point_index].astype(np.float32),
        "local_support_ratio_10m": artifacts.local_support_ratio_10m[point_index].astype(np.float32),
        "local_ground_seed_ratio_10m": artifacts.local_ground_seed_ratio_10m[point_index].astype(np.float32),
        "refh_snr": point_data.snr[point_index].astype(np.float32),
        "refh_amp": point_data.amp[point_index].astype(np.float32),
    })

    summary: Dict[str, Any] = {
        "scene": scene_name,
        "point_count": int(len(df)),
        "cleaned_rule_balanced": cleaned_info_balanced,
        "cleaned_rule_strict": cleaned_info_strict,
        "modified_raw_counts": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(modified_raw_target, return_counts=True))
        },
        "cleaned_modified_raw_counts_balanced": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(cleaned_modified_raw_target_balanced, return_counts=True))
        },
        "cleaned_modified_raw_counts_strict": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(cleaned_modified_raw_target_strict, return_counts=True))
        },
    }

    if scene_cfg["dep3_laz"].exists():
        stable_idx, stable_target, stable_info = evaluator.build_stable_surface_target(
            scene_cfg=scene_cfg,
            artifacts=artifacts,
            cell_m=float(evaluator.CONFIG["ground_surface_cell_m"]),
            sample_n=evaluator.CONFIG["sample_n"],
        )
        stable_df = pd.DataFrame({
            "point_index": stable_idx.astype(np.int64),
            "stable_surface_target": stable_target.astype(np.uint8),
        })
        df = df.merge(stable_df, on="point_index", how="left", validate="one_to_one")

        summary["stable_surface_target_info"] = stable_info
        for mode in ["balanced", "strict"]:
            col = f"cleaned_modified_raw_target_{mode}"
            promoted_mask = df[col].to_numpy() != df["modified_raw_target"].to_numpy()
            promoted_stable = df.loc[promoted_mask, "stable_surface_target"].dropna().to_numpy(dtype=np.uint8)
            summary[f"promoted_stable_class_counts_{mode}"] = {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(promoted_stable, return_counts=True))
            } if promoted_stable.size else {}
            summary[f"promotion_precision_to_stable2_{mode}"] = (
                float(np.mean(promoted_stable == 2))
                if promoted_stable.size else float("nan")
            )

    out_path = OUT_DIR / f"{scene_name}_cleaned_modified_raw_targets.parquet"
    df.to_parquet(out_path, index=False)
    summary["output_parquet"] = str(out_path)
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        scene_name: build_scene_artifact(scene_name, scene_cfg)
        for scene_name, scene_cfg in evaluator.SCENES.items()
    }
    SUMMARY_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[INFO] Wrote JSON: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
