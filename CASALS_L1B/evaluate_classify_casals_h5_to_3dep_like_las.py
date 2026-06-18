#!/usr/bin/env python3
"""
Evaluate classify_casals_h5_to_3dep_like_las.py on the three local CASALS/3DEP pairs.

This script reports four benchmarks:
1. `stable_surface_target`
   Uses residual to the 3DEP class-2 ground surface after estimating a single
   empirical vertical offset `dz` from CASALS ground seeds. This is the more
   stable ground/non-ground/noise target and does not depend on an old aligned
   LAS artifact.
2. `raw_transfer_pseudolabel_target`
   Uses the direct pointwise transfer pseudo-labels from the earlier workflow.
   This is useful for audit, but it is known to be semantically inconsistent in
   pair2_de, where many raw pseudo class-1 points are actually ground-surface
   points relative to 3DEP class-2.
3. `cleaned_modified_raw_target_balanced`
   Starts from the modified raw transfer target, then applies the balanced
   cleanup rule found in the later cross-scene study.
4. `cleaned_modified_raw_target_strict`
   Same rule family, but with a tighter DTM residual gate to reduce false
   promotions in high-support scenes such as pair3_nc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import precision_recall_fscore_support

import classify_casals_h5_to_3dep_like_las as classifier
from clean_modified_raw_target import CLEAN_MODIFIED_RAW_RULES, apply_clean_modified_raw_rule


ROOT = Path(__file__).resolve().parent
OUTPUT_JSON = ROOT / "outputs" / "classify_casals_h5_to_3dep_like_las" / "three_scene_multi_target_evaluation.json"
LEGACY_OUTPUT_JSON = ROOT / "outputs" / "classify_casals_h5_to_3dep_like_las" / "three_scene_dual_target_evaluation.json"

CONFIG = {
    "ground_surface_cell_m": 3.0,
    "ground_class_resid_abs_max_m": 0.5,
    "noise_class_resid_below_m": -2.0,
    "nonground_class_resid_above_m": 2.0,
    "sample_n": None,
    "ignore_pair2_de_raw_pseudo_class2_in_modified_raw_eval": True,
    "default_clean_modified_raw_mode": "balanced",
}

SCENES = {
    "pair1_md": {
        "h5": ROOT / "casals_h5_downloads" / "casals_l1b_20241112T165718_001_02.h5",
        "aligned_las": ROOT / "outputs" / "transfer_3dep_labels_to_casals_refh_multi" / "pair1_md_casals_3dep_pseudolabeled_aligned.las",
        "dep3_laz": ROOT / "point_cloud_data" / "download_3dep_lpc" / "casals_l1b_20241112T165718_001_02_MD_Southeast_1_2019_EPSG6347_39a068a77804.laz",
        "feature_parquet": ROOT / "outputs" / "rule_search_features" / "pair1_md_features.parquet",
    },
    "pair2_de": {
        "h5": ROOT / "casals_h5_downloads" / "casals_l1b_20241112T170442_001_02.h5",
        "aligned_las": ROOT / "outputs" / "transfer_3dep_labels_to_casals_refh_multi" / "pair2_de_casals_3dep_pseudolabeled_aligned.las",
        "dep3_laz": ROOT / "point_cloud_data" / "download_3dep_lpc" / "casals_l1b_20241112T170442_001_02_DE_Statewide_1_B23_EPSG6347_b60d6cbd5f2f.laz",
        "feature_parquet": ROOT / "outputs" / "rule_search_features" / "pair2_de_features.parquet",
    },
    "pair3_nc": {
        "h5": ROOT / "casals_h5_downloads" / "casals_l1b_20241118T171757_001_02.h5",
        "aligned_las": ROOT / "outputs" / "transfer_3dep_labels_to_casals_refh_multi" / "pair3_nc_casals_3dep_pseudolabeled_aligned.las",
        "dep3_laz": ROOT / "point_cloud_data" / "download_3dep_lpc" / "casals_l1b_20241118T171757_001_02_NC_HurricaneFlorence_9_2020_EPSG6347_f50533a04725.laz",
        "feature_parquet": ROOT / "outputs" / "rule_search_features" / "pair3_nc_features.parquet",
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_ground_grid(dep3_path: Path, cell_m: float) -> tuple[np.ndarray, np.ndarray, cKDTree]:
    with laspy.open(dep3_path) as reader:
        hdr = reader.header
        xmin, ymin = hdr.mins[0], hdr.mins[1]
        xmax, ymax = hdr.maxs[0], hdr.maxs[1]
        ncols = int(np.ceil((xmax - xmin) / cell_m)) + 1
        nrows = int(np.ceil((ymax - ymin) / cell_m)) + 1
        sums = np.zeros(nrows * ncols, dtype=np.float64)
        counts = np.zeros(nrows * ncols, dtype=np.uint32)

        for chunk in reader.chunk_iterator(5_000_000):
            cls = np.asarray(chunk.classification)
            mask = cls == 2
            if not np.any(mask):
                continue
            x = np.asarray(chunk.x)[mask]
            y = np.asarray(chunk.y)[mask]
            z = np.asarray(chunk.z)[mask]
            col = np.clip(np.floor((x - xmin) / cell_m).astype(np.int64), 0, ncols - 1)
            row = np.clip(np.floor((y - ymin) / cell_m).astype(np.int64), 0, nrows - 1)
            cid = row * ncols + col
            np.add.at(sums, cid, z)
            np.add.at(counts, cid, 1)

    valid = counts > 0
    grid_z = np.full(nrows * ncols, np.nan, dtype=np.float64)
    grid_z[valid] = sums[valid] / counts[valid]
    vcid = np.flatnonzero(valid)
    vr = vcid // ncols
    vc = vcid % ncols
    vx = xmin + (vc.astype(np.float64) + 0.5) * cell_m
    vy = ymin + (vr.astype(np.float64) + 0.5) * cell_m
    tree = cKDTree(np.column_stack((vx, vy)))
    return grid_z, vcid, tree


def build_stable_surface_target(
    scene_cfg: Dict[str, Path],
    artifacts: classifier.ClassificationArtifacts,
    cell_m: float,
    sample_n: int | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    feat = pd.read_parquet(scene_cfg["feature_parquet"], columns=["point_index", "pseudo_3dep_class", "match_status"])
    keep = (
        ((feat["match_status"].isin(["strict_pseudolabel", "weak_pseudolabel"])) & (feat["pseudo_3dep_class"].isin([1, 2])))
        | (feat["match_status"] == "internal_noise_flagged")
    )
    feat = feat.loc[keep].copy().sort_values("point_index").reset_index(drop=True)
    if sample_n is not None and len(feat) > sample_n:
        feat = feat.sample(n=sample_n, random_state=42).sort_values("point_index").reset_index(drop=True)

    idx = feat["point_index"].to_numpy()
    grid_z, vcid, tree = build_ground_grid(scene_cfg["dep3_laz"], cell_m)
    seed_xyz = artifacts.xyz[artifacts.ground_seed_mask]
    _, qi_seed = tree.query(seed_xyz[:, :2], k=1, workers=-1)
    seed_surface_z = grid_z[vcid[qi_seed]]
    seed_resid = seed_xyz[:, 2] - seed_surface_z
    seed_resid = seed_resid[np.isfinite(seed_resid)]
    if seed_resid.size == 0:
        raise ValueError("Could not estimate dz for stable-surface evaluation because no ground-seed/surface pairs were finite.")

    dz = float(np.median(seed_resid))
    dz_nmad = float(classifier.robust_nmad(seed_resid))

    xyz = artifacts.xyz[idx]
    _, qi = tree.query(xyz[:, :2], k=1, workers=-1)
    surface_z = grid_z[vcid[qi]]
    surface_resid = (xyz[:, 2] - dz) - surface_z

    target = np.full(len(feat), -1, dtype=np.int16)
    match_status = feat["match_status"].to_numpy()
    target[match_status == "internal_noise_flagged"] = 7
    unresolved = target == -1
    target[unresolved & (surface_resid < float(CONFIG["noise_class_resid_below_m"]))] = 7
    target[unresolved & (np.abs(surface_resid) <= float(CONFIG["ground_class_resid_abs_max_m"]))] = 2
    target[unresolved & (surface_resid > float(CONFIG["nonground_class_resid_above_m"]))] = 1

    keep_labeled = np.isin(target, [1, 2, 7])
    info = {
        "method": "direct_projected_xy_plus_empirical_dz_from_ground_seeds",
        "estimated_dz_m": dz,
        "ground_seed_surface_pairs": int(seed_resid.size),
        "ground_seed_surface_residual_nmad_m": dz_nmad,
    }
    return idx[keep_labeled], target[keep_labeled].astype(np.uint8), info


def build_raw_transfer_target(scene_cfg: Dict[str, Path], sample_n: int | None) -> tuple[np.ndarray, np.ndarray]:
    feat = pd.read_parquet(scene_cfg["feature_parquet"], columns=["point_index", "pseudo_3dep_class", "match_status"])
    keep = (
        ((feat["match_status"].isin(["strict_pseudolabel", "weak_pseudolabel"])) & (feat["pseudo_3dep_class"].isin([1, 2])))
        | (feat["match_status"] == "internal_noise_flagged")
    )
    feat = feat.loc[keep].copy().sort_values("point_index").reset_index(drop=True)
    if sample_n is not None and len(feat) > sample_n:
        feat = feat.sample(n=sample_n, random_state=42).sort_values("point_index").reset_index(drop=True)

    target = np.full(len(feat), 1, dtype=np.uint8)
    target[feat["pseudo_3dep_class"].to_numpy() == 2] = 2
    target[feat["match_status"].to_numpy() == "internal_noise_flagged"] = 7
    return feat["point_index"].to_numpy(), target


def build_modified_raw_transfer_target(scene_name: str, scene_cfg: Dict[str, Path], sample_n: int | None) -> tuple[np.ndarray, np.ndarray]:
    point_index, target = build_raw_transfer_target(scene_cfg=scene_cfg, sample_n=sample_n)
    if bool(CONFIG["ignore_pair2_de_raw_pseudo_class2_in_modified_raw_eval"]) and scene_name == "pair2_de":
        keep = target != 2
        return point_index[keep], target[keep]
    return point_index, target


def build_cleaned_modified_raw_target(
    scene_name: str,
    scene_cfg: Dict[str, Path],
    artifacts: classifier.ClassificationArtifacts,
    sample_n: int | None,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, dict]]]:
    point_index, modified_raw_target = build_modified_raw_transfer_target(
        scene_name=scene_name,
        scene_cfg=scene_cfg,
        sample_n=sample_n,
    )
    per_mode: dict[str, tuple[np.ndarray, dict]] = {}
    for mode in CLEAN_MODIFIED_RAW_RULES:
        cleaned, info = apply_clean_modified_raw_rule(
            point_index=point_index,
            modified_raw_target=modified_raw_target,
            artifacts=artifacts,
            mode=mode,
        )
        per_mode[mode] = (cleaned, info)
    return point_index, per_mode


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    present_labels = [cls for cls in [1, 2, 7] if np.any(y_true == cls)]
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=present_labels, zero_division=0)
    metrics = {
        int(cls): {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for cls, p, r, f, s in zip(present_labels, prec, rec, f1, sup)
    }
    return {
        "n_eval_points": int(len(y_true)),
        "metrics": metrics,
        "min_f1": float(min(m["f1"] for m in metrics.values())),
    }


def evaluate_scene(name: str, scene_cfg: Dict[str, Path]) -> dict:
    cfg = dict(classifier.CONFIG)
    cfg["h5_path"] = str(scene_cfg["h5"])
    cfg["output_dir"] = str(ROOT / "outputs" / "tmp_eval")
    _, artifacts = classifier.classify_h5(cfg)

    stable_result = None
    stable_info = None
    if scene_cfg["dep3_laz"].exists():
        point_index_stable, y_true_stable, stable_info = build_stable_surface_target(
            scene_cfg=scene_cfg,
            artifacts=artifacts,
            cell_m=float(CONFIG["ground_surface_cell_m"]),
            sample_n=CONFIG["sample_n"],
        )
        stable_result = evaluate_predictions(y_true_stable, artifacts.classification[point_index_stable])

    point_index_raw, y_true_raw = build_raw_transfer_target(
        scene_cfg=scene_cfg,
        sample_n=CONFIG["sample_n"],
    )
    point_index_raw_modified, y_true_raw_modified = build_modified_raw_transfer_target(
        scene_name=name,
        scene_cfg=scene_cfg,
        sample_n=CONFIG["sample_n"],
    )
    point_index_cleaned_raw, cleaned_targets = build_cleaned_modified_raw_target(
        scene_name=name,
        scene_cfg=scene_cfg,
        artifacts=artifacts,
        sample_n=CONFIG["sample_n"],
    )
    balanced_target, balanced_info = cleaned_targets["balanced"]
    strict_target, strict_info = cleaned_targets["strict"]
    return {
        "stable_surface_target": stable_result,
        "stable_surface_target_info": stable_info,
        "raw_transfer_pseudolabel_target": evaluate_predictions(y_true_raw, artifacts.classification[point_index_raw]),
        "modified_raw_transfer_target": evaluate_predictions(y_true_raw_modified, artifacts.classification[point_index_raw_modified]),
        "cleaned_modified_raw_target": evaluate_predictions(balanced_target, artifacts.classification[point_index_cleaned_raw]),
        "cleaned_modified_raw_target_info": balanced_info,
        "cleaned_modified_raw_target_balanced": evaluate_predictions(balanced_target, artifacts.classification[point_index_cleaned_raw]),
        "cleaned_modified_raw_target_balanced_info": balanced_info,
        "cleaned_modified_raw_target_strict": evaluate_predictions(strict_target, artifacts.classification[point_index_cleaned_raw]),
        "cleaned_modified_raw_target_strict_info": strict_info,
    }


def main() -> None:
    ensure_dir(OUTPUT_JSON.parent)
    results = {
        name: evaluate_scene(name, scene_cfg)
        for name, scene_cfg in SCENES.items()
    }
    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LEGACY_OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[INFO] Wrote evaluation JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
