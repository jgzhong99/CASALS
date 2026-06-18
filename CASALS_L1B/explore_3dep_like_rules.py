#!/usr/bin/env python3
"""
Explore simple 3DEP-like CASALS rule sets on the three local pseudo-labeled pairs.

This is a diagnostic workflow. It does not write the final classified LAS.
Its job is to:
  1. read the existing `transfer_3dep_labels_to_casals_refh.py` outputs,
  2. build a small geometric feature table per scene,
  3. evaluate interpretable heuristic rules across scenes,
  4. report where the rule family succeeds or fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "outputs" / "rule_search_features"
OUT_DIR = ROOT / "outputs" / "explore_3dep_like_rules"


PAIR_FILES = {
    "pair1_md": FEATURE_DIR / "pair1_md_features.parquet",
    "pair2_de": FEATURE_DIR / "pair2_de_features.parquet",
    "pair3_nc": FEATURE_DIR / "pair3_nc_features.parquet",
}


RULE_LIBRARY = [
    {
        "name": "baseline_hag_roughness",
        "noise_snr_max": 1.8,
        "noise_amp_max": 450.0,
        "noise_abs_hag5_min": 120.0,
        "ground_hag5_min": -0.5,
        "ground_hag5_max": 0.45,
        "ground_hspan5_max": 0.5,
        "ground_znmad5_max": 0.3,
        "ground_snr_min": 1.8,
        "ground_amp_min": 450.0,
    },
    {
        "name": "slightly_looser_ground",
        "noise_snr_max": 1.8,
        "noise_amp_max": 450.0,
        "noise_abs_hag5_min": 120.0,
        "ground_hag5_min": -0.5,
        "ground_hag5_max": 0.60,
        "ground_hspan5_max": 1.0,
        "ground_znmad5_max": 0.5,
        "ground_snr_min": 1.8,
        "ground_amp_min": 450.0,
    },
    {
        "name": "high_precision_ground",
        "noise_snr_max": 1.7,
        "noise_amp_max": 450.0,
        "noise_abs_hag5_min": 100.0,
        "ground_hag5_min": -0.5,
        "ground_hag5_max": 0.35,
        "ground_hspan5_max": 0.3,
        "ground_znmad5_max": 0.2,
        "ground_snr_min": 1.9,
        "ground_amp_min": 500.0,
    },
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_scene(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    keep = (
        ((df["match_status"].isin(["strict_pseudolabel", "weak_pseudolabel"])) & (df["pseudo_3dep_class"].isin([1, 2])))
        | (df["match_status"] == "internal_noise_flagged")
    )
    df = df.loc[keep].copy()
    df["target"] = 1
    df.loc[df["pseudo_3dep_class"] == 2, "target"] = 2
    df.loc[df["match_status"] == "internal_noise_flagged", "target"] = 7
    numeric_cols = ["refh_snr", "refh_amp", "hag5", "hspan5", "znmad5", "hag10", "hspan10", "znmad10"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=numeric_cols)


def summarize_feature_quantiles(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cls in [1, 2, 7]:
        g = df[df["target"] == cls]
        out[str(cls)] = {}
        for col in cols:
            vals = g[col].dropna()
            out[str(cls)][col] = {
                "p10": float(vals.quantile(0.10)),
                "p25": float(vals.quantile(0.25)),
                "p50": float(vals.quantile(0.50)),
                "p75": float(vals.quantile(0.75)),
                "p90": float(vals.quantile(0.90)),
            }
    return out


def apply_rule(df: pd.DataFrame, rule: Dict[str, Any]) -> np.ndarray:
    pred = np.full(len(df), 1, dtype=np.uint8)
    noise = ((df["refh_snr"] < rule["noise_snr_max"]) & (df["refh_amp"] < rule["noise_amp_max"])) | (
        np.abs(df["hag5"]) > rule["noise_abs_hag5_min"]
    )
    ground = (
        (df["hag5"] >= rule["ground_hag5_min"])
        & (df["hag5"] <= rule["ground_hag5_max"])
        & (df["hspan5"] <= rule["ground_hspan5_max"])
        & (df["znmad5"] <= rule["ground_znmad5_max"])
        & (df["refh_snr"] >= rule["ground_snr_min"])
        & (df["refh_amp"] >= rule["ground_amp_min"])
    )
    pred[noise.to_numpy()] = 7
    pred[(pred != 7) & ground.to_numpy()] = 2
    return pred


def f1_dict(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "class_1_f1": float(f1_score(y_true == 1, y_pred == 1)),
        "class_2_f1": float(f1_score(y_true == 2, y_pred == 2)),
        "class_7_f1": float(f1_score(y_true == 7, y_pred == 7)),
    }


def evaluate_rule(name: str, df: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
    pred = apply_rule(df, rule)
    f1s = f1_dict(df["target"], pred)
    report = classification_report(df["target"], pred, labels=[1, 2, 7], output_dict=True, zero_division=0)
    return {
        "rule_name": name,
        "n_points": int(len(df)),
        "f1": f1s,
        "classification_report": report,
    }


def main() -> None:
    ensure_dir(OUT_DIR)

    scenes = {name: load_scene(path) for name, path in PAIR_FILES.items()}
    feature_summary = {
        name: summarize_feature_quantiles(df, ["refh_snr", "refh_amp", "hag5", "hspan5", "znmad5"])
        for name, df in scenes.items()
    }

    rule_results: List[Dict[str, Any]] = []
    for rule in RULE_LIBRARY:
        per_scene = {name: evaluate_rule(name, df, rule) for name, df in scenes.items()}
        min_f1 = min(min(scene["f1"].values()) for scene in per_scene.values())
        mean_macro_f1 = float(
            np.mean([np.mean(list(scene["f1"].values())) for scene in per_scene.values()])
        )
        rule_results.append({
            "rule": rule,
            "min_scene_class_f1": float(min_f1),
            "mean_scene_macro_f1": mean_macro_f1,
            "per_scene": per_scene,
        })

    rule_results.sort(key=lambda r: (r["min_scene_class_f1"], r["mean_scene_macro_f1"]), reverse=True)

    summary = {
        "script": "explore_3dep_like_rules.py",
        "feature_summary": feature_summary,
        "rule_results_sorted": rule_results,
    }
    out_json = OUT_DIR / "rule_exploration_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 88)
    print("3DEP-like rule exploration summary")
    print("=" * 88)
    for result in rule_results[:3]:
        print("\nRULE:", result["rule"]["name"])
        print("  min_scene_class_f1:", round(result["min_scene_class_f1"], 4))
        print("  mean_scene_macro_f1:", round(result["mean_scene_macro_f1"], 4))
        for scene_name, scene in result["per_scene"].items():
            f1s = {k: round(v, 4) for k, v in scene["f1"].items()}
            print(" ", scene_name, f1s)
    print("\nJSON:", out_json)


if __name__ == "__main__":
    main()
