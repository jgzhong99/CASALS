#!/usr/bin/env python3
"""
Classify one CASALS L1B H5 granule into a simple 3DEP-like LAS.

Goal
----
Produce an interpretable point-level classification with:
  - class 2: ground-like,
  - class 1: retained non-ground,
  - class 7: likely noise / outlier.

Method
------
1. Read CASALS refh points from one H5 granule.
2. Project lon/lat to a local projected CRS.
3. Use high-SNR points to build a preliminary low surface with progressive
   morphology.
4. Select tentative ground seeds from high-SNR residuals to that low surface.
5. Build a smoothed IDW DTM from those seeds.
6. Classify all finite points using residual-to-DTM plus a few simple
   low-signal / local-outlier rules.

Scientific scope
----------------
- This is a heuristic CASALS refh classifier tuned to behave more like 3DEP
  ground / non-ground / noise semantics.
- It is not official truth and not a rigorous geodetic product.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import laspy
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.spatial import cKDTree

from extract_refh_ground import (
    Config as MorphConfig,
    build_preliminary_ground_surface,
    classify_highsnr_points_against_prelim,
    idw_interpolate_ground,
    infer_wgs84_utm_epsg,
    make_grid_from_points,
    make_support_mask_from_points,
    normalized_gaussian_smooth,
    read_point_data,
    sample_grid_nearest,
    transform_lonlat_to_projected,
    valid_base_mask,
    write_float_geotiff,
)


CONFIG = {
    #"h5_path": r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5",
    #"h5_path": r"./casals_h5_downloads/casals_l1b_20241112T170442_001_02.h5",
    "h5_path": r"./casals_h5_downloads/casals_l1b_20241118T171757_001_02.h5",
    "output_dir": r"./outputs/classify_casals_h5_to_3dep_like_las",
    "output_prefix": None,
    "output_epsg_override": None,

    # Support points used to build the low surface / DTM.
    "support_snr_threshold": 5.0,
    "require_good_snr_for_support_if_available": True,

    # Preliminary morphology surface.
    "prelim_grid_resolution_m": 10.0,
    "prelim_low_percentile": 10.0,
    "prelim_support_buffer_m": 35.0,
    "prelim_support_closing_m": 30.0,
    "prelim_smoothing_sigma_cells": 0.8,
    "use_progressive_morphology": True,
    "morphology_window_sizes_m": (30.0, 50.0, 80.0, 120.0, 180.0),
    "base_threshold_m": 1.0,
    "slope_threshold": 0.06,
    "max_threshold_m": 8.0,
    "final_median_smooth_size_cells": 3,

    # Residual thresholds for the high-SNR seed selection stage.
    "prelim_ground_above_tolerance_m": 2.5,
    "prelim_ground_below_tolerance_m": 3.0,

    # Final DTM interpolation.
    "dtm_resolution_m": 10.0,
    "extent_buffer_m": 0.0,
    "dtm_support_source": "support_points",
    "support_buffer_m": 35.0,
    "support_closing_m": 30.0,
    "support_fill_holes": True,
    "idw_radius_m": 45.0,
    "idw_k": 12,
    "idw_power": 2.0,
    "idw_min_neighbors": 3,
    "idw_chunk_size": 250_000,
    "fill_internal_holes": True,
    "max_internal_fill_distance_m": 60.0,
    "dtm_smoothing_sigma_cells": 0.7,

    # Final classification rules. These thresholds come from a shallow
    # cross-scene search against stable 3DEP-ground-surface targets.
    "rule_abs_dtm_ground_m": 1.507,
    "rule_dtm_ground_upper_m": 0.923,
    "rule_dtm_low_noise_m": -0.002,
    "rule_low_snr_noise_max": 2.506,
    "rule_no_surface_ground_amp_min": 481.5,
    "rule_positive_dense_nonground_lower_m": 8.0,
    "rule_positive_dense_nonground_upper_m": 30.0,
    "rule_positive_dense_nonground_cell_min": 1500,
    "rule_positive_dense_nonground_support_ratio_min": 0.08,
    "rule_positive_dense_nonground_nearest_support_max_m": 1.0,
    "rule_negative_sparse_noise_m": -2.0,
    "rule_negative_sparse_support_ratio_max": 0.01,
    "rule_negative_sparse_nearest_support_min_m": 2.0,
    "rule_negative_sparse_snr_max": 3.0,

    # Conservative local-Z outlier flag. Flagged points are kept as class 7.
    "enable_local_z_outlier_flag": True,
    "local_z_cell_size_m": 5.0,
    "local_z_min_points_per_cell": 8,
    "local_z_nmad_multiplier": 8.0,
    "local_z_min_abs_threshold_m": 20.0,

    # Audit-only support evidence summaries used to explain why a point was
    # considered close to the morphology/DTM evidence.
    "audit_local_support_cell_size_m": 10.0,
    "audit_query_chunk_size": 250_000,

    # Output control.
    "write_point_csv": True,
    "write_dtm_tif": True,
    "csv_float_precision": "%.6f",
    "las_scale_m": 0.001,
}


CLASS_NON_GROUND = 1
CLASS_GROUND = 2
CLASS_NOISE = 7

QF_NONFINITE = 1 << 0
QF_LOW_SIGNAL = 1 << 1
QF_LOCAL_Z_OUTLIER = 1 << 2
QF_OUTSIDE_DTM_SUPPORT = 1 << 3
QF_BAD_GOOD_SNR = 1 << 4

RF_GROUND_WINDOW = 1 << 0
RF_NOISE_BELOW_DTM = 1 << 1
RF_NOISE_LOW_SIGNAL = 1 << 2
RF_NOISE_LOCAL_Z_OUTLIER = 1 << 3
RF_SUPPORT_POINT = 1 << 4
RF_GROUND_SEED = 1 << 5
RF_DTM_SUPPORTED = 1 << 6


@dataclass
class ClassificationArtifacts:
    base_mask: np.ndarray
    xyz: np.ndarray
    out_crs: CRS
    support_mask: np.ndarray
    ground_seed_mask: np.ndarray
    local_z_outlier_mask: np.ndarray
    quality_flags: np.ndarray
    rule_flags: np.ndarray
    preliminary_residual_m: np.ndarray
    dtm_z_m: np.ndarray
    dtm_residual_m: np.ndarray
    nearest_support_xy_distance_m: np.ndarray
    nearest_ground_seed_xy_distance_m: np.ndarray
    local_point_count_10m: np.ndarray
    local_support_count_10m: np.ndarray
    local_ground_seed_count_10m: np.ndarray
    local_support_ratio_10m: np.ndarray
    local_ground_seed_ratio_10m: np.ndarray
    classification: np.ndarray
    dtm_grid: Any
    dtm_raster: np.ndarray
    prelim_grid: Any
    prelim_surface: np.ndarray
    classifier_mode: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def robust_nmad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def summarize_array(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "nmad": float("nan"),
        }
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "nmad": robust_nmad(values),
    }


def compute_local_z_outliers(xyz: np.ndarray, cfg: dict) -> np.ndarray:
    n = xyz.shape[0]
    out = np.zeros(n, dtype=bool)
    if not cfg["enable_local_z_outlier_flag"]:
        return out

    finite = np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1]) & np.isfinite(xyz[:, 2])
    if not np.any(finite):
        return out

    cell = float(cfg["local_z_cell_size_m"])
    min_n = int(cfg["local_z_min_points_per_cell"])
    mult = float(cfg["local_z_nmad_multiplier"])
    min_abs = float(cfg["local_z_min_abs_threshold_m"])

    ix = np.floor(xyz[:, 0] / cell).astype(np.int64)
    iy = np.floor(xyz[:, 1] / cell).astype(np.int64)
    df = pd.DataFrame({"idx": np.arange(n), "ix": ix, "iy": iy, "z": xyz[:, 2], "finite": finite})
    for (_, _), group in df[df["finite"]].groupby(["ix", "iy"], sort=False):
        if len(group) < min_n:
            continue
        idx = group["idx"].to_numpy()
        z = group["z"].to_numpy(dtype=float)
        med = np.median(z)
        sigma = max(robust_nmad(z), 1e-6)
        thresh = max(min_abs, mult * sigma)
        out[idx] = np.abs(z - med) > thresh
    return out


def compute_nearest_xy_distance(xy: np.ndarray, source_mask: np.ndarray, chunk_size: int) -> np.ndarray:
    out = np.full(xy.shape[0], np.nan, dtype=np.float32)
    if not np.any(source_mask):
        return out

    tree = cKDTree(xy[source_mask])
    for start in range(0, xy.shape[0], chunk_size):
        stop = min(start + chunk_size, xy.shape[0])
        dist, _ = tree.query(xy[start:stop], k=1, workers=-1)
        out[start:stop] = dist.astype(np.float32)
    return out


def compute_local_support_stats(
    xyz: np.ndarray,
    base_mask: np.ndarray,
    support_mask: np.ndarray,
    ground_seed_mask: np.ndarray,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = xyz.shape[0]
    local_point_count = np.zeros(n, dtype=np.uint32)
    local_support_count = np.zeros(n, dtype=np.uint32)
    local_ground_seed_count = np.zeros(n, dtype=np.uint32)
    local_support_ratio = np.full(n, np.nan, dtype=np.float32)
    local_ground_seed_ratio = np.full(n, np.nan, dtype=np.float32)

    if not np.any(base_mask):
        return (
            local_point_count,
            local_support_count,
            local_ground_seed_count,
            local_support_ratio,
            local_ground_seed_ratio,
        )

    finite_xy = np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1])
    work_mask = base_mask & finite_xy
    if not np.any(work_mask):
        return (
            local_point_count,
            local_support_count,
            local_ground_seed_count,
            local_support_ratio,
            local_ground_seed_ratio,
        )

    idx = np.flatnonzero(work_mask)
    ix = np.floor(xyz[idx, 0] / cell_size_m).astype(np.int64)
    iy = np.floor(xyz[idx, 1] / cell_size_m).astype(np.int64)
    cell_coords = np.column_stack((ix, iy))
    _, inverse, counts = np.unique(cell_coords, axis=0, return_inverse=True, return_counts=True)
    support_counts = np.bincount(inverse, weights=support_mask[idx].astype(np.uint32), minlength=counts.size).astype(np.uint32)
    ground_seed_counts = np.bincount(inverse, weights=ground_seed_mask[idx].astype(np.uint32), minlength=counts.size).astype(np.uint32)

    local_point_count[idx] = counts[inverse].astype(np.uint32)
    local_support_count[idx] = support_counts[inverse].astype(np.uint32)
    local_ground_seed_count[idx] = ground_seed_counts[inverse].astype(np.uint32)
    local_support_ratio[idx] = (local_support_count[idx] / np.maximum(local_point_count[idx], 1)).astype(np.float32)
    local_ground_seed_ratio[idx] = (local_ground_seed_count[idx] / np.maximum(local_point_count[idx], 1)).astype(np.float32)
    return (
        local_point_count,
        local_support_count,
        local_ground_seed_count,
        local_support_ratio,
        local_ground_seed_ratio,
    )


def build_morph_config(cfg: dict, h5_path: Path, output_dir: Path) -> MorphConfig:
    return MorphConfig(
        h5_path=h5_path,
        point_cloud_dir=output_dir,
        out_dir=output_dir,
        snr_threshold=float(cfg["support_snr_threshold"]),
        output_epsg_override=cfg["output_epsg_override"],
        prelim_grid_resolution_m=float(cfg["prelim_grid_resolution_m"]),
        prelim_low_percentile=float(cfg["prelim_low_percentile"]),
        prelim_support_buffer_m=float(cfg["prelim_support_buffer_m"]),
        prelim_support_closing_m=float(cfg["prelim_support_closing_m"]),
        use_progressive_morphology=bool(cfg["use_progressive_morphology"]),
        morphology_window_sizes_m=tuple(float(v) for v in cfg["morphology_window_sizes_m"]),
        base_threshold_m=float(cfg["base_threshold_m"]),
        slope_threshold=float(cfg["slope_threshold"]),
        max_threshold_m=float(cfg["max_threshold_m"]),
        final_median_smooth_size_cells=int(cfg["final_median_smooth_size_cells"]),
        ground_above_prelim_tolerance_m=float(cfg["prelim_ground_above_tolerance_m"]),
        ground_below_prelim_tolerance_m=float(cfg["prelim_ground_below_tolerance_m"]),
        dtm_resolution_m=float(cfg["dtm_resolution_m"]),
        extent_buffer_m=float(cfg["extent_buffer_m"]),
        support_buffer_m=float(cfg["support_buffer_m"]),
        support_closing_m=float(cfg["support_closing_m"]),
        support_fill_holes=bool(cfg["support_fill_holes"]),
        idw_radius_m=float(cfg["idw_radius_m"]),
        idw_k=int(cfg["idw_k"]),
        idw_power=float(cfg["idw_power"]),
        idw_min_neighbors=int(cfg["idw_min_neighbors"]),
        idw_chunk_size=int(cfg["idw_chunk_size"]),
        fill_internal_holes=bool(cfg["fill_internal_holes"]),
        max_internal_fill_distance_m=cfg["max_internal_fill_distance_m"],
        prelim_smoothing_sigma_cells=float(cfg["prelim_smoothing_sigma_cells"]),
        smoothing_sigma_cells=float(cfg["dtm_smoothing_sigma_cells"]),
        write_classified_highsnr_las=False,
        write_ground_only_las=False,
        write_support_mask_tif=False,
        write_source_mask_tif=False,
        write_distance_tif=False,
        write_morphology_debug_tifs=False,
        write_preview_png=False,
        write_mesh=False,
        las_scale_m=float(cfg["las_scale_m"]),
    )


def classify_h5(cfg: dict) -> tuple[Any, ClassificationArtifacts]:
    h5_path = Path(cfg["h5_path"]).resolve()
    output_dir = Path(cfg["output_dir"]).resolve()
    ensure_dir(output_dir)

    point_data = read_point_data(h5_path)
    base_mask = valid_base_mask(point_data)
    if not np.any(base_mask):
        raise ValueError("No finite CASALS refh points were found.")

    epsg = cfg["output_epsg_override"] or infer_wgs84_utm_epsg(point_data.lon[base_mask], point_data.lat[base_mask])
    x_all, y_all, out_crs = transform_lonlat_to_projected(point_data.lon, point_data.lat, int(epsg))
    xyz = np.column_stack((x_all, y_all, point_data.z.astype(np.float64)))

    support_mask = base_mask & (point_data.snr >= float(cfg["support_snr_threshold"]))
    if bool(cfg["require_good_snr_for_support_if_available"]) and point_data.good_snr is not None:
        support_mask &= np.asarray(point_data.good_snr).astype(bool)
    if not np.any(support_mask):
        raise ValueError("No support points survived the morphology SNR threshold.")

    morph_cfg = build_morph_config(cfg, h5_path, output_dir)
    prelim_surface, prelim_grid, _, _, _, _ = build_preliminary_ground_surface(
        x_all[support_mask],
        y_all[support_mask],
        point_data.z[support_mask],
        morph_cfg,
    )

    prelim_resid_all = point_data.z - sample_grid_nearest(x_all, y_all, prelim_surface, prelim_grid)
    prelim_cls_support, _ = classify_highsnr_points_against_prelim(
        x_all[support_mask],
        y_all[support_mask],
        point_data.z[support_mask],
        prelim_surface,
        prelim_grid,
        morph_cfg,
    )
    ground_seed_mask = np.zeros(point_data.z.size, dtype=bool)
    support_indices = np.flatnonzero(support_mask)
    ground_seed_mask[support_indices[prelim_cls_support == CLASS_GROUND]] = True
    if int(ground_seed_mask.sum()) < 10:
        raise ValueError("Too few ground seeds survived preliminary morphology filtering.")

    dtm_support_source = str(cfg["dtm_support_source"]).strip().lower()
    if dtm_support_source == "support_points":
        support_xy_mask = support_mask
    elif dtm_support_source == "ground_seeds":
        support_xy_mask = ground_seed_mask
    else:
        raise ValueError("dtm_support_source must be 'support_points' or 'ground_seeds'.")

    dtm_grid = make_grid_from_points(x_all[support_xy_mask], y_all[support_xy_mask], float(cfg["dtm_resolution_m"]), buffer_m=float(cfg["extent_buffer_m"]))
    dtm_support = make_support_mask_from_points(
        x_all[support_xy_mask],
        y_all[support_xy_mask],
        dtm_grid,
        buffer_m=float(cfg["support_buffer_m"]),
        closing_m=float(cfg["support_closing_m"]),
        fill_holes=bool(cfg["support_fill_holes"]),
    )
    dtm_idw, _, _ = idw_interpolate_ground(
        x_all[ground_seed_mask],
        y_all[ground_seed_mask],
        point_data.z[ground_seed_mask],
        dtm_grid,
        dtm_support,
        morph_cfg,
    )
    dtm_raster = normalized_gaussian_smooth(
        dtm_idw,
        dtm_support & np.isfinite(dtm_idw),
        float(cfg["dtm_smoothing_sigma_cells"]),
    )
    dtm_raster[~dtm_support] = np.nan

    dtm_z_all = sample_grid_nearest(x_all, y_all, dtm_raster, dtm_grid)
    dtm_resid_all = point_data.z - dtm_z_all

    nearest_support_xy_distance_m = compute_nearest_xy_distance(
        xyz[:, :2],
        support_mask,
        chunk_size=int(cfg["audit_query_chunk_size"]),
    )
    nearest_ground_seed_xy_distance_m = compute_nearest_xy_distance(
        xyz[:, :2],
        ground_seed_mask,
        chunk_size=int(cfg["audit_query_chunk_size"]),
    )
    (
        local_point_count_10m,
        local_support_count_10m,
        local_ground_seed_count_10m,
        local_support_ratio_10m,
        local_ground_seed_ratio_10m,
    ) = compute_local_support_stats(
        xyz,
        base_mask=base_mask,
        support_mask=support_mask,
        ground_seed_mask=ground_seed_mask,
        cell_size_m=float(cfg["audit_local_support_cell_size_m"]),
    )

    local_z_outlier_mask = compute_local_z_outliers(xyz, cfg)
    quality_flags = np.zeros(point_data.z.size, dtype=np.uint8)
    rule_flags = np.zeros(point_data.z.size, dtype=np.uint8)

    quality_flags[~base_mask] |= QF_NONFINITE
    if point_data.good_snr is not None:
        quality_flags[base_mask & ~np.asarray(point_data.good_snr).astype(bool)] |= QF_BAD_GOOD_SNR

    low_signal_mask = base_mask & (point_data.snr < float(cfg["rule_low_snr_noise_max"]))
    quality_flags[low_signal_mask] |= QF_LOW_SIGNAL
    quality_flags[local_z_outlier_mask] |= QF_LOCAL_Z_OUTLIER
    quality_flags[base_mask & ~np.isfinite(dtm_z_all)] |= QF_OUTSIDE_DTM_SUPPORT

    rule_flags[support_mask] |= RF_SUPPORT_POINT
    rule_flags[ground_seed_mask] |= RF_GROUND_SEED
    rule_flags[np.isfinite(dtm_z_all)] |= RF_DTM_SUPPORTED

    dtm_supported = np.isfinite(dtm_resid_all)
    abs_dtm_resid = np.abs(dtm_resid_all)

    near_dtm_ground_mask = (
        dtm_supported
        & (abs_dtm_resid <= float(cfg["rule_abs_dtm_ground_m"]))
        & (dtm_resid_all <= float(cfg["rule_dtm_ground_upper_m"]))
    )
    positive_dense_nonground_mask = (
        dtm_supported
        & local_z_outlier_mask
        & (dtm_resid_all >= float(cfg["rule_positive_dense_nonground_lower_m"]))
        & (dtm_resid_all <= float(cfg["rule_positive_dense_nonground_upper_m"]))
        & (local_point_count_10m >= int(cfg["rule_positive_dense_nonground_cell_min"]))
        & (local_support_ratio_10m >= float(cfg["rule_positive_dense_nonground_support_ratio_min"]))
        & (nearest_support_xy_distance_m <= float(cfg["rule_positive_dense_nonground_nearest_support_max_m"]))
    )
    high_resid_local_outlier_noise_mask = (
        dtm_supported
        & (abs_dtm_resid > float(cfg["rule_abs_dtm_ground_m"]))
        & local_z_outlier_mask
        & (~positive_dense_nonground_mask)
    )
    deep_negative_noise_mask = (
        dtm_supported
        & (abs_dtm_resid > float(cfg["rule_abs_dtm_ground_m"]))
        & (~local_z_outlier_mask)
        & (dtm_resid_all <= float(cfg["rule_dtm_low_noise_m"]))
        & (point_data.snr <= float(cfg["rule_low_snr_noise_max"]))
    )
    negative_sparse_noise_mask = (
        dtm_supported
        & (dtm_resid_all <= float(cfg["rule_negative_sparse_noise_m"]))
        & (
            (local_support_ratio_10m <= float(cfg["rule_negative_sparse_support_ratio_max"]))
            | (nearest_support_xy_distance_m >= float(cfg["rule_negative_sparse_nearest_support_min_m"]))
        )
        & (point_data.snr <= float(cfg["rule_negative_sparse_snr_max"]))
        & (~positive_dense_nonground_mask)
    )
    deep_negative_ground_mask = (
        dtm_supported
        & (abs_dtm_resid > float(cfg["rule_abs_dtm_ground_m"]))
        & (~local_z_outlier_mask)
        & (dtm_resid_all <= float(cfg["rule_dtm_low_noise_m"]))
        & (point_data.snr > float(cfg["rule_low_snr_noise_max"]))
    )

    no_surface_mask = base_mask & (~dtm_supported)
    no_surface_noise_mask = no_surface_mask & local_z_outlier_mask
    no_surface_ground_mask = no_surface_mask & (~local_z_outlier_mask) & (point_data.amp > float(cfg["rule_no_surface_ground_amp_min"]))

    noise_mask = (
        (~base_mask)
        | high_resid_local_outlier_noise_mask
        | deep_negative_noise_mask
        | negative_sparse_noise_mask
        | no_surface_noise_mask
    )
    ground_mask = near_dtm_ground_mask | deep_negative_ground_mask | no_surface_ground_mask

    rule_flags[near_dtm_ground_mask] |= RF_GROUND_WINDOW
    rule_flags[deep_negative_ground_mask] |= RF_GROUND_WINDOW
    rule_flags[no_surface_ground_mask] |= RF_GROUND_WINDOW
    rule_flags[deep_negative_noise_mask | negative_sparse_noise_mask] |= RF_NOISE_BELOW_DTM
    rule_flags[high_resid_local_outlier_noise_mask | no_surface_noise_mask] |= RF_NOISE_LOCAL_Z_OUTLIER
    rule_flags[deep_negative_noise_mask] |= RF_NOISE_LOW_SIGNAL
    rule_flags[local_z_outlier_mask] |= RF_NOISE_LOCAL_Z_OUTLIER

    classification = np.full(point_data.z.size, CLASS_NON_GROUND, dtype=np.uint8)
    classification[noise_mask] = CLASS_NOISE
    classification[ground_mask] = CLASS_GROUND

    artifacts = ClassificationArtifacts(
        base_mask=base_mask,
        xyz=xyz,
        out_crs=out_crs,
        support_mask=support_mask,
        ground_seed_mask=ground_seed_mask,
        local_z_outlier_mask=local_z_outlier_mask,
        quality_flags=quality_flags,
        rule_flags=rule_flags,
        preliminary_residual_m=prelim_resid_all,
        dtm_z_m=dtm_z_all,
        dtm_residual_m=dtm_resid_all,
        nearest_support_xy_distance_m=nearest_support_xy_distance_m,
        nearest_ground_seed_xy_distance_m=nearest_ground_seed_xy_distance_m,
        local_point_count_10m=local_point_count_10m,
        local_support_count_10m=local_support_count_10m,
        local_ground_seed_count_10m=local_ground_seed_count_10m,
        local_support_ratio_10m=local_support_ratio_10m,
        local_ground_seed_ratio_10m=local_ground_seed_ratio_10m,
        classification=classification,
        dtm_grid=dtm_grid,
        dtm_raster=dtm_raster,
        prelim_grid=prelim_grid,
        prelim_surface=prelim_surface,
        classifier_mode="generic_handcrafted_rule_density_aware_noise",
    )
    return point_data, artifacts


def write_classified_las(output_path: Path, point_data: Any, artifacts: ClassificationArtifacts, cfg: dict) -> None:
    mask = artifacts.base_mask
    x = artifacts.xyz[mask, 0]
    y = artifacts.xyz[mask, 1]
    z = artifacts.xyz[mask, 2]
    cls = artifacts.classification[mask]

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([cfg["las_scale_m"], cfg["las_scale_m"], cfg["las_scale_m"]], dtype=np.float64)
    header.offsets = np.array([
        math.floor(float(np.nanmin(x))),
        math.floor(float(np.nanmin(y))),
        math.floor(float(np.nanmin(z))),
    ], dtype=np.float64)
    header.system_identifier = "CASALS_REFH_3DEP_LIKE"
    header.generating_software = "classify_casals_h5_to_3dep_like_las.py"
    header.add_crs(artifacts.out_crs)

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.intensity = np.clip(np.rint(point_data.amp[mask]), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    las.classification = cls

    red = np.zeros(x.size, dtype=np.uint16)
    green = np.zeros(x.size, dtype=np.uint16)
    blue = np.zeros(x.size, dtype=np.uint16)
    red[cls == CLASS_NON_GROUND] = 65535
    green[cls == CLASS_GROUND] = 65535
    blue[cls == CLASS_NOISE] = 65535
    las.red = red
    las.green = green
    las.blue = blue

    extra_dims = [
        laspy.ExtraBytesParams(name="longitude", type=np.float64, description="CASALS lon"),
        laspy.ExtraBytesParams(name="latitude", type=np.float64, description="CASALS lat"),
        laspy.ExtraBytesParams(name="refh_snr", type=np.float32, description="CASALS refh snr"),
        laspy.ExtraBytesParams(name="refh_amp", type=np.float32, description="CASALS refh amp"),
        laspy.ExtraBytesParams(name="refh_thres", type=np.float32, description="CASALS refh thres"),
        laspy.ExtraBytesParams(name="good_snr", type=np.uint8, description="CASALS good_snr"),
        laspy.ExtraBytesParams(name="quality_flag", type=np.uint8, description="quality bits"),
        laspy.ExtraBytesParams(name="rule_flag", type=np.uint8, description="rule bits"),
        laspy.ExtraBytesParams(name="support_pt", type=np.uint8, description="high-SNR support"),
        laspy.ExtraBytesParams(name="ground_seed", type=np.uint8, description="prelim seed"),
        laspy.ExtraBytesParams(name="prelim_resid", type=np.float32, description="z-prelim"),
        laspy.ExtraBytesParams(name="dtm_z_m", type=np.float32, description="final dtm z"),
        laspy.ExtraBytesParams(name="dtm_resid_m", type=np.float32, description="z-dtm"),
        laspy.ExtraBytesParams(name="near_supp_m", type=np.float32, description="nearest support xy dist m"),
        laspy.ExtraBytesParams(name="near_seed_m", type=np.float32, description="nearest seed xy dist m"),
        laspy.ExtraBytesParams(name="cell_n_10m", type=np.uint32, description="points in 10m cell"),
        laspy.ExtraBytesParams(name="supp_n_10m", type=np.uint32, description="support pts in 10m cell"),
        laspy.ExtraBytesParams(name="seed_n_10m", type=np.uint32, description="seed pts in 10m cell"),
        laspy.ExtraBytesParams(name="supp_r10m", type=np.float32, description="support ratio in 10m cell"),
        laspy.ExtraBytesParams(name="seed_r10m", type=np.float32, description="seed ratio in 10m cell"),
        laspy.ExtraBytesParams(name="local_z_out", type=np.uint8, description="local z outlier"),
        laspy.ExtraBytesParams(name="pulse_index", type=np.uint32, description="refh index"),
    ]
    if point_data.track_num is not None:
        extra_dims.append(laspy.ExtraBytesParams(name="track_num", type=np.uint16, description="CASALS track_num"))
    if point_data.sweep_num is not None:
        extra_dims.append(laspy.ExtraBytesParams(name="sweep_num", type=np.uint32, description="CASALS sweep_num"))
    las.add_extra_dims(extra_dims)

    las["longitude"] = point_data.lon[mask].astype(np.float64)
    las["latitude"] = point_data.lat[mask].astype(np.float64)
    las["refh_snr"] = point_data.snr[mask].astype(np.float32)
    las["refh_amp"] = point_data.amp[mask].astype(np.float32)
    las["refh_thres"] = point_data.thres[mask].astype(np.float32)
    las["good_snr"] = np.asarray(point_data.good_snr[mask]).astype(np.uint8)
    las["quality_flag"] = artifacts.quality_flags[mask].astype(np.uint8)
    las["rule_flag"] = artifacts.rule_flags[mask].astype(np.uint8)
    las["support_pt"] = artifacts.support_mask[mask].astype(np.uint8)
    las["ground_seed"] = artifacts.ground_seed_mask[mask].astype(np.uint8)
    las["prelim_resid"] = artifacts.preliminary_residual_m[mask].astype(np.float32)
    las["dtm_z_m"] = artifacts.dtm_z_m[mask].astype(np.float32)
    las["dtm_resid_m"] = artifacts.dtm_residual_m[mask].astype(np.float32)
    las["near_supp_m"] = artifacts.nearest_support_xy_distance_m[mask].astype(np.float32)
    las["near_seed_m"] = artifacts.nearest_ground_seed_xy_distance_m[mask].astype(np.float32)
    las["cell_n_10m"] = artifacts.local_point_count_10m[mask].astype(np.uint32)
    las["supp_n_10m"] = artifacts.local_support_count_10m[mask].astype(np.uint32)
    las["seed_n_10m"] = artifacts.local_ground_seed_count_10m[mask].astype(np.uint32)
    las["supp_r10m"] = artifacts.local_support_ratio_10m[mask].astype(np.float32)
    las["seed_r10m"] = artifacts.local_ground_seed_ratio_10m[mask].astype(np.float32)
    las["local_z_out"] = artifacts.local_z_outlier_mask[mask].astype(np.uint8)
    las["pulse_index"] = np.flatnonzero(mask).astype(np.uint32)

    if point_data.track_num is not None:
        las["track_num"] = point_data.track_num[mask].astype(np.uint16)
    if point_data.sweep_num is not None:
        las["sweep_num"] = point_data.sweep_num[mask].astype(np.uint32)

    las.write(str(output_path))


def build_point_csv(point_data: Any, artifacts: ClassificationArtifacts) -> pd.DataFrame:
    mask = artifacts.base_mask
    df = pd.DataFrame({
        "point_index": np.flatnonzero(mask).astype(np.int64),
        "lon": point_data.lon[mask].astype(np.float64),
        "lat": point_data.lat[mask].astype(np.float64),
        "x_m": artifacts.xyz[mask, 0].astype(np.float32),
        "y_m": artifacts.xyz[mask, 1].astype(np.float32),
        "z_m": artifacts.xyz[mask, 2].astype(np.float32),
        "classification": artifacts.classification[mask].astype(np.uint8),
        "support_point": artifacts.support_mask[mask].astype(np.uint8),
        "ground_seed": artifacts.ground_seed_mask[mask].astype(np.uint8),
        "local_z_outlier": artifacts.local_z_outlier_mask[mask].astype(np.uint8),
        "quality_flag": artifacts.quality_flags[mask].astype(np.uint8),
        "rule_flag": artifacts.rule_flags[mask].astype(np.uint8),
        "refh_snr": point_data.snr[mask].astype(np.float32),
        "refh_amp": point_data.amp[mask].astype(np.float32),
        "refh_thres": point_data.thres[mask].astype(np.float32),
        "prelim_resid_m": artifacts.preliminary_residual_m[mask].astype(np.float32),
        "dtm_z_m": artifacts.dtm_z_m[mask].astype(np.float32),
        "dtm_resid_m": artifacts.dtm_residual_m[mask].astype(np.float32),
        "nearest_support_xy_distance_m": artifacts.nearest_support_xy_distance_m[mask].astype(np.float32),
        "nearest_ground_seed_xy_distance_m": artifacts.nearest_ground_seed_xy_distance_m[mask].astype(np.float32),
        "local_point_count_10m": artifacts.local_point_count_10m[mask].astype(np.uint32),
        "local_support_count_10m": artifacts.local_support_count_10m[mask].astype(np.uint32),
        "local_ground_seed_count_10m": artifacts.local_ground_seed_count_10m[mask].astype(np.uint32),
        "local_support_ratio_10m": artifacts.local_support_ratio_10m[mask].astype(np.float32),
        "local_ground_seed_ratio_10m": artifacts.local_ground_seed_ratio_10m[mask].astype(np.float32),
    })
    if point_data.track_num is not None:
        df["track_num"] = point_data.track_num[mask]
    if point_data.sweep_num is not None:
        df["sweep_num"] = point_data.sweep_num[mask]
    return df


def safe_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, CRS):
        return obj.to_string()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    return obj


def main() -> None:
    cfg = dict(CONFIG)
    h5_path = Path(cfg["h5_path"]).resolve()
    output_dir = Path(cfg["output_dir"]).resolve()
    ensure_dir(output_dir)

    prefix = cfg["output_prefix"] or h5_path.stem
    las_path = output_dir / f"{prefix}_3dep_like_classified.las"
    json_path = output_dir / f"{prefix}_3dep_like_classification_summary.json"
    csv_path = output_dir / f"{prefix}_3dep_like_classified_points.csv"
    dtm_path = output_dir / f"{prefix}_3dep_like_dtm.tif"

    print(f"[INFO] Reading H5 and building morphology-guided 3DEP-like classification: {h5_path}")
    point_data, artifacts = classify_h5(cfg)

    print("[INFO] Writing classified LAS...")
    write_classified_las(las_path, point_data, artifacts, cfg)

    if cfg["write_point_csv"]:
        df = build_point_csv(point_data, artifacts)
        df.to_csv(csv_path, index=False, float_format=cfg["csv_float_precision"])
        print(f"[INFO] Wrote point CSV: {csv_path}")

    if cfg["write_dtm_tif"]:
        write_float_geotiff(dtm_path, artifacts.dtm_raster, artifacts.out_crs, artifacts.dtm_grid.transform, -9999.0)
        print(f"[INFO] Wrote DTM TIFF: {dtm_path}")

    class_counts = {
        int(c): int(n)
        for c, n in zip(*np.unique(artifacts.classification[artifacts.base_mask], return_counts=True))
    }
    summary = {
        "script": "classify_casals_h5_to_3dep_like_las.py",
        "scientific_scope": "heuristic CASALS refh classification approximating 3DEP-like ground/non-ground/noise semantics",
        "input_h5": h5_path,
        "classifier_mode": artifacts.classifier_mode,
        "output_las": las_path,
        "output_csv": csv_path if cfg["write_point_csv"] else None,
        "output_dtm_tif": dtm_path if cfg["write_dtm_tif"] else None,
        "config": cfg,
        "counts": {
            "total_refh_points": int(point_data.lon.size),
            "base_valid_points": int(artifacts.base_mask.sum()),
            "support_points": int(artifacts.support_mask.sum()),
            "ground_seed_points": int(artifacts.ground_seed_mask.sum()),
            "class_counts": class_counts,
        },
        "residual_summary": {
            "preliminary_residual_m": summarize_array(artifacts.preliminary_residual_m[artifacts.base_mask]),
            "dtm_residual_m": summarize_array(artifacts.dtm_residual_m[artifacts.base_mask]),
            "ground_only_dtm_residual_m": summarize_array(artifacts.dtm_residual_m[(artifacts.base_mask) & (artifacts.classification == CLASS_GROUND)]),
            "noise_only_dtm_residual_m": summarize_array(artifacts.dtm_residual_m[(artifacts.base_mask) & (artifacts.classification == CLASS_NOISE)]),
        },
        "audit_feature_summary": {
            "nearest_support_xy_distance_m": summarize_array(artifacts.nearest_support_xy_distance_m[artifacts.base_mask]),
            "nearest_ground_seed_xy_distance_m": summarize_array(artifacts.nearest_ground_seed_xy_distance_m[artifacts.base_mask]),
            "local_point_count_10m": summarize_array(artifacts.local_point_count_10m[artifacts.base_mask]),
            "local_support_count_10m": summarize_array(artifacts.local_support_count_10m[artifacts.base_mask]),
            "local_ground_seed_count_10m": summarize_array(artifacts.local_ground_seed_count_10m[artifacts.base_mask]),
            "local_support_ratio_10m": summarize_array(artifacts.local_support_ratio_10m[artifacts.base_mask]),
            "local_ground_seed_ratio_10m": summarize_array(artifacts.local_ground_seed_ratio_10m[artifacts.base_mask]),
        },
        "quality_flag_counts": {
            int(c): int(n)
            for c, n in zip(*np.unique(artifacts.quality_flags[artifacts.base_mask], return_counts=True))
        },
        "rule_flag_counts": {
            int(c): int(n)
            for c, n in zip(*np.unique(artifacts.rule_flags[artifacts.base_mask], return_counts=True))
        },
    }
    json_path.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
    print(f"[INFO] Wrote summary JSON: {json_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
