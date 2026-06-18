#!/usr/bin/env python3
"""
Reusable cleanup rules for the modified raw pseudo-label target.

Modes
-----
- `balanced`: maximizes cross-scene recovery of raw class-1 points that look
  like stable ground.
- `strict`: slightly tighter DTM residual gate that reduces false promotions in
  pair3_nc while keeping the same simple interpretation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import classify_casals_h5_to_3dep_like_las as classifier


CLEAN_MODIFIED_RAW_RULES: Dict[str, Dict[str, float | str]] = {
    "balanced": {
        "mode": "balanced",
        "description": "Promote modified raw class-1 to class-2 when close to the CASALS DTM and nearby morphology support.",
        "abs_dtm_resid_max_m": 1.0,
        "nearest_support_max_m": 5.0,
    },
    "strict": {
        "mode": "strict",
        "description": "Same as balanced, but with a tighter DTM residual threshold to reduce false promotions in high-support scenes.",
        "abs_dtm_resid_max_m": 0.75,
        "nearest_support_max_m": 5.0,
    },
}


def apply_clean_modified_raw_rule(
    point_index: np.ndarray,
    modified_raw_target: np.ndarray,
    artifacts: classifier.ClassificationArtifacts,
    mode: str = "balanced",
) -> Tuple[np.ndarray, dict]:
    if mode not in CLEAN_MODIFIED_RAW_RULES:
        raise ValueError(f"Unknown clean-modified-raw mode: {mode}")

    rule = CLEAN_MODIFIED_RAW_RULES[mode]
    cleaned = modified_raw_target.copy()
    promote_mask = (
        (cleaned == 1)
        & (np.abs(artifacts.dtm_residual_m[point_index]) <= float(rule["abs_dtm_resid_max_m"]))
        & (artifacts.nearest_support_xy_distance_m[point_index] <= float(rule["nearest_support_max_m"]))
    )
    cleaned[promote_mask] = 2
    info = {
        "method": "promote_modified_raw_class1_to_class2_when_close_to_dtm_and_support",
        "mode": str(rule["mode"]),
        "description": str(rule["description"]),
        "abs_dtm_resid_max_m": float(rule["abs_dtm_resid_max_m"]),
        "nearest_support_max_m": float(rule["nearest_support_max_m"]),
        "promoted_count": int(promote_mask.sum()),
    }
    return cleaned, info
