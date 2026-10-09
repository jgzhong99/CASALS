#!/usr/bin/env python3
"""
Ground-align 3DEP labels to CASALS L1B refh points and export labeled CASALS LAS/LAZ.

Purpose
-------
Input:
  1. A CASALS L1B H5 file containing refh longitude/latitude/height.
  2. A clipped 3DEP LAS/LAZ file containing XYZ and LAS classification.

Workflow:
  Verified complete source/target CRS: audit and apply the PROJ XYZ operation.
  Explicit empirical_diagnostic: project XY and fit exploratory ground dz.
  Transfer reliable 3DEP classes; far/ambiguous points remain unclassified.

Scientific notes
----------------
- CASALS L1B refh points are official geolocated reference-return / max-Rx-bin
  points, not a formal waveform-decomposed multi-return point cloud.
- The estimated dz is an empirical ground-to-ground vertical alignment term.
  It is NOT a rigorous geoid/datum transformation.
- Output coordinates use the audited target XYZ frame in verified_crs mode,
  or projected XY plus exploratory dz in empirical_diagnostic mode. Transferred
  labels remain pseudo-reference rather than independent ground truth.
- LAS classification stores the transferred/evaluation class. Extra dimensions
  store transfer status, vote ratio, nearest 3DEP distance, and alignment audit
  fields so that labels can be filtered later.

Dependencies
------------
conda install -c conda-forge h5py laspy lazrs pyproj scipy pandas numpy

Example
-------
python -m casals_l1b reference transfer ^
  --casals-h5 data/raw/casals_l1b/casals_l1b_20241112T165718_001_02.h5 ^
  --dep3-las data/reference/3dep/example_3dep_clip.laz ^
  --output-dir outputs/reference/casals_l1b_20241112T165718_001_02/transfer
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import laspy
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

from casals_l1b.geo import horizontal_crs_only
from research.reference.diagnose_3dep_offsets import (resolve_reference_frame, build_target_compound_crs, transform_verified_xyz)
from dataclasses import asdict


LAS_CLASS_LOW_NOISE = 7

STATUS_FAR_NO_MATCH = 0
STATUS_STRICT = 1
STATUS_WEAK = 2
STATUS_AMBIGUOUS_NEAR = 3
STATUS_NONFINITE_NOISE = 4

STATUS_NAMES = {
    STATUS_FAR_NO_MATCH: "far_no_match",
    STATUS_STRICT: "strict_3dep_label",
    STATUS_WEAK: "weak_3dep_label",
    STATUS_AMBIGUOUS_NEAR: "ambiguous_near_3dep",
    STATUS_NONFINITE_NOISE: "nonfinite_noise",
}

QF_NONFINITE = 1 << 0
QF_BAD_GOOD_SNR = 1 << 1
QF_LOW_REFH_SNR = 1 << 2
QF_LOCAL_Z_OUTLIER = 1 << 3
QF_NOT_GROUNDLIKE_FOR_ALIGNMENT = 1 << 4
QF_NO_3DEP_GROUND_FOR_ALIGNMENT = 1 << 5


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("finite_mask requires at least one array")
    mask = np.ones(np.asarray(arrays[0]).shape[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(np.asarray(arr))
    return mask


def robust_nmad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def summarize_values(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "nmad": None,
        }
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "nmad": robust_nmad(x),
    }


def safe_json_dump(obj: Dict[str, Any], path: Path) -> None:
    def convert(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            if np.isfinite(o):
                return float(o)
            return None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, CRS):
            return o.to_string()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)


def collect_class_counts(cls: np.ndarray) -> Dict[int, int]:
    cls = np.asarray(cls, dtype=np.uint8)
    vals, counts = np.unique(cls, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}



def find_dataset_path(h5: h5py.File, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    all_paths: List[str] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            all_paths.append(name)

    h5.visititems(visitor)
    norm_candidates = [c.strip("/") for c in candidates]

    for cand in norm_candidates:
        if cand in h5 and isinstance(h5[cand], h5py.Dataset):
            return cand

    leaves: Dict[str, str] = {}
    for p in all_paths:
        leaves[p.split("/")[-1]] = p

    for cand in norm_candidates:
        if cand in leaves:
            return leaves[cand]

    for cand in norm_candidates:
        for p in all_paths:
            if p.endswith("/" + cand) or p.endswith(cand):
                return p

    if required:
        raise KeyError(f"Could not find required H5 dataset among candidates: {candidates}")
    return None


def read_optional_dataset(h5: h5py.File, candidates: Sequence[str], n: int, selection=slice(None)) -> Tuple[Optional[np.ndarray], Optional[str]]:
    path = find_dataset_path(h5, candidates, required=False)
    if path is None:
        return None, None
    arr = np.asarray(h5[path][selection]).reshape(-1)
    if arr.size != n:
        print(f"[WARN] Optional dataset {path} has size {arr.size}, expected {n}; ignored.")
        return None, None
    return arr, path


def read_casals_h5(h5_path: Path, selection=slice(None)) -> Dict[str, Any]:
    print(f"[INFO] Reading CASALS H5: {h5_path}")
    fields: Dict[str, np.ndarray] = {}
    source: Dict[str, str] = {}

    with h5py.File(h5_path, "r") as h5:
        lon_path = find_dataset_path(h5, ["refh_longitude", "longitude", "lon"])
        lat_path = find_dataset_path(h5, ["refh_latitude", "latitude", "lat"])
        z_path = find_dataset_path(h5, ["refh", "refh_height", "height", "elevation"])

        lon = np.asarray(h5[lon_path][selection], dtype=np.float64).reshape(-1)
        lat = np.asarray(h5[lat_path][selection], dtype=np.float64).reshape(-1)
        z = np.asarray(h5[z_path][selection], dtype=np.float64).reshape(-1)

        if lat.size != lon.size or z.size != lon.size:
            raise ValueError(f"CASALS lon/lat/z sizes differ: {lon.size}, {lat.size}, {z.size}")

        source.update({"lon": lon_path, "lat": lat_path, "z": z_path})
        point_index = np.arange(h5[lon_path].size, dtype=np.uint32)[selection]
        n = lon.size

        optional_specs = {
            "refh_amp": ["refh_amp", "amp", "amplitude"],
            "refh_snr": ["refh_snr", "snr"],
            "good_snr": ["good_snr"],
            "bg_mean": ["bg_mean", "background_mean"],
            "bg_std": ["bg_std", "background_std"],
            "refh_thres": ["refh_thres", "threshold"],
            "track_num": ["track_num", "track", "track_index"],
            "sweep_num": ["sweep_num", "sweep", "sweep_index"],
            "delta_time": ["delta_time", "time"],
            "angle": ["angle", "incidence_angle", "scan_angle"],
            "refh_error": ["refh_error", "height_error"],
            "horizontal_error": ["horizontal_error", "geolocation_error"],
        }

        for out_name, candidates in optional_specs.items():
            arr, path = read_optional_dataset(h5, candidates, n, selection)
            if arr is not None:
                fields[out_name] = arr
                source[out_name] = path

    if "refh_snr" not in fields:
        if {"refh_amp", "bg_mean", "bg_std"}.issubset(fields.keys()):
            amp = np.asarray(fields["refh_amp"], dtype=np.float64)
            bg_mean = np.asarray(fields["bg_mean"], dtype=np.float64)
            bg_std = np.asarray(fields["bg_std"], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                fields["refh_snr"] = np.divide(
                    amp - bg_mean,
                    bg_std,
                    out=np.full_like(amp, np.nan, dtype=np.float64),
                    where=(bg_std != 0),
                )
            source["refh_snr"] = "derived_from_(refh_amp-bg_mean)/bg_std"
            print("[INFO] Derived refh_snr from (refh_amp - bg_mean) / bg_std.")
        elif {"refh_amp", "refh_thres"}.issubset(fields.keys()):
            amp = np.asarray(fields["refh_amp"], dtype=np.float64)
            thres = np.asarray(fields["refh_thres"], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                fields["refh_snr"] = np.divide(
                    amp,
                    thres,
                    out=np.full_like(amp, np.nan, dtype=np.float64),
                    where=(thres != 0),
                )
            source["refh_snr"] = "derived_from_refh_amp/refh_thres"
            print("[INFO] Derived refh_snr from refh_amp / refh_thres.")

    print(f"[INFO] CASALS points: {lon.size:,}")
    print(f"[INFO] Optional CASALS fields found: {sorted(fields.keys())}")

    return {
        "lon": lon,
        "lat": lat,
        "z": z,
        "fields": fields,
        "source_datasets": source,
        "source_h5": str(h5_path.resolve()),
        "point_index": point_index,
    }


def read_3dep_las(las_path: Path, xy_bounds=None, max_points=None) -> Dict[str, Any]:
    """Stream source clips; bounded selection is available for local smoke checks."""
    arrays, classes = [], []
    selected = scanned = 0
    truncated = False
    with laspy.open(las_path) as reader:
        header = reader.header
        for points in reader.chunk_iterator(500_000):
            xyz = np.column_stack([points.x, points.y, points.z])
            scanned += len(points)
            keep = np.all(np.isfinite(xyz), axis=1) & ~np.asarray(points.withheld, dtype=bool)
            if xy_bounds is not None:
                xmin, ymin, xmax, ymax = xy_bounds
                keep &= (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
            idx = np.flatnonzero(keep)
            if max_points is not None and selected + len(idx) > max_points:
                idx = idx[:max_points-selected]
                truncated = True
            if len(idx):
                arrays.append(xyz[idx]); classes.append(np.asarray(points.classification)[idx])
                selected += len(idx)
            if truncated:
                break
    return {"xyz": np.concatenate(arrays) if arrays else np.empty((0, 3)),
            "classification": np.concatenate(classes) if classes else np.empty(0, dtype=np.uint8),
            "crs": header.parse_crs(),
            "header_summary": {"point_count": int(header.point_count), "selected_count": selected,
                               "scanned_count": scanned, "selection_truncated": truncated,
                               "version": str(header.version), "point_format_id": header.point_format.id,
                               "scales": header.scales.tolist(), "mins": header.mins.tolist(), "maxs": header.maxs.tolist()}}


def project_casals_to_target(casals: Dict[str, Any], target_horizontal_crs: CRS) -> np.ndarray:
    transformer = Transformer.from_crs(CRS.from_epsg(4326), target_horizontal_crs, always_xy=True)
    x, y = transformer.transform(casals["lon"], casals["lat"])
    xyz = np.column_stack((
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(casals["z"], dtype=np.float64),
    ))
    return xyz


def build_quality_flags(casals: Dict[str, Any], xyz: np.ndarray, min_snr: float) -> np.ndarray:
    n = xyz.shape[0]
    fields = casals["fields"]
    flags = np.zeros(n, dtype=np.uint8)

    nonfinite = ~finite_mask(xyz[:, 0], xyz[:, 1], xyz[:, 2], casals["lon"], casals["lat"])
    flags[nonfinite] |= QF_NONFINITE

    if "good_snr" in fields:
        good = np.asarray(fields["good_snr"]).astype(bool)
        flags[~good] |= QF_BAD_GOOD_SNR

    if "refh_snr" in fields:
        snr = np.asarray(fields["refh_snr"], dtype=np.float64)
        flags[snr < float(min_snr)] |= QF_LOW_REFH_SNR

    return flags


def compute_local_z_outliers(
    xyz: np.ndarray,
    base_mask: np.ndarray,
    cell_size_m: float,
    min_points_per_cell: int,
    nmad_multiplier: float,
    min_abs_threshold_m: float,
) -> np.ndarray:
    n = xyz.shape[0]
    out = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(base_mask & finite_mask(xyz[:, 0], xyz[:, 1], xyz[:, 2]))
    if idx.size == 0:
        return out

    x0 = float(np.nanmin(xyz[idx, 0]))
    y0 = float(np.nanmin(xyz[idx, 1]))
    ix = np.floor((xyz[idx, 0] - x0) / cell_size_m).astype(np.int64)
    iy = np.floor((xyz[idx, 1] - y0) / cell_size_m).astype(np.int64)

    df = pd.DataFrame({"idx": idx, "ix": ix, "iy": iy, "z": xyz[idx, 2]})
    for _, g in df.groupby(["ix", "iy"], sort=False):
        if len(g) < min_points_per_cell:
            continue
        gidx = g["idx"].to_numpy()
        z = g["z"].to_numpy(dtype=np.float64)
        med = np.median(z)
        sigma = max(robust_nmad(z), 1e-6)
        threshold = max(float(min_abs_threshold_m), float(nmad_multiplier) * sigma)
        out[gidx] = np.abs(z - med) > threshold
    return out


def compute_local_low_surface_for_candidates(
    xyz: np.ndarray,
    candidate_mask: np.ndarray,
    cell_size_m: float,
    min_points_per_cell: int,
    low_percentile: float,
) -> np.ndarray:
    n = xyz.shape[0]
    local_low = np.full(n, np.nan, dtype=np.float64)
    idx = np.flatnonzero(candidate_mask & finite_mask(xyz[:, 0], xyz[:, 1], xyz[:, 2]))
    if idx.size == 0:
        return local_low

    x0 = float(np.nanmin(xyz[idx, 0]))
    y0 = float(np.nanmin(xyz[idx, 1]))
    ix = np.floor((xyz[idx, 0] - x0) / cell_size_m).astype(np.int64)
    iy = np.floor((xyz[idx, 1] - y0) / cell_size_m).astype(np.int64)

    df = pd.DataFrame({"idx": idx, "ix": ix, "iy": iy, "z": xyz[idx, 2]})
    grouped = df.groupby(["ix", "iy"], sort=False)
    df["cell_n"] = grouped["z"].transform("size")
    df["cell_low_z"] = grouped["z"].transform(lambda s: float(np.percentile(s.to_numpy(dtype=np.float64), low_percentile)))

    valid = df["cell_n"].to_numpy() >= int(min_points_per_cell)
    local_low[df.loc[valid, "idx"].to_numpy(dtype=np.int64)] = df.loc[valid, "cell_low_z"].to_numpy(dtype=np.float64)
    return local_low


def sample_reference_z_idw(
    query_xy: np.ndarray,
    ref_xy: np.ndarray,
    ref_z: np.ndarray,
    tree: cKDTree,
    k: int,
    radius_m: float,
    power: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d, idx = tree.query(query_xy, k=int(k), distance_upper_bound=float(radius_m), workers=-1)
    if int(k) == 1:
        d = d[:, None]
        idx = idx[:, None]

    valid = np.isfinite(d) & (idx < ref_z.shape[0])
    count = np.sum(valid, axis=1).astype(np.uint16)

    nearest = np.full(query_xy.shape[0], np.inf, dtype=np.float64)
    any_valid = count > 0
    nearest[any_valid] = np.min(np.where(valid, d, np.inf), axis=1)[any_valid]

    z_neigh = np.zeros_like(d, dtype=np.float64)
    z_neigh[valid] = ref_z[idx[valid]]

    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(valid, 1.0 / np.maximum(d, 1e-6) ** float(power), 0.0)

    weight_sum = np.sum(weights, axis=1)
    z_sample = np.full(query_xy.shape[0], np.nan, dtype=np.float64)
    ok = weight_sum > 0
    z_sample[ok] = np.sum(weights[ok] * z_neigh[ok], axis=1) / weight_sum[ok]

    return z_sample, count, nearest


def estimate_ground_vertical_shift(
    casals: Dict[str, Any],
    casals_xyz: np.ndarray,
    dep3_xyz: np.ndarray,
    dep3_cls: np.ndarray,
    cfg: Dict[str, Any],
    quality_flags: np.ndarray,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    fields = casals["fields"]
    n = casals_xyz.shape[0]

    dep_ground_mask = (
        (dep3_cls == int(cfg["ground_class"])) &
        finite_mask(dep3_xyz[:, 0], dep3_xyz[:, 1], dep3_xyz[:, 2])
    )
    dep_ground_xyz = dep3_xyz[dep_ground_mask]
    if dep_ground_xyz.shape[0] < int(cfg["min_ground_alignment_pairs"]):
        raise RuntimeError(
            f"Too few 3DEP ground points: {dep_ground_xyz.shape[0]:,}. "
            f"Expected at least {cfg['min_ground_alignment_pairs']:,}."
        )

    dep_ground_xy = dep_ground_xyz[:, :2]
    dep_ground_z = dep_ground_xyz[:, 2]
    dep_ground_tree = cKDTree(dep_ground_xy)

    base_mask = finite_mask(casals_xyz[:, 0], casals_xyz[:, 1], casals_xyz[:, 2])
    if cfg["require_good_snr_for_ground"] and "good_snr" in fields:
        base_mask &= np.asarray(fields["good_snr"]).astype(bool)
    if "refh_snr" in fields:
        base_mask &= np.asarray(fields["refh_snr"], dtype=np.float64) >= float(cfg["min_ground_snr"])

    local_out = compute_local_z_outliers(
        xyz=casals_xyz,
        base_mask=base_mask,
        cell_size_m=float(cfg["local_z_cell_size_m"]),
        min_points_per_cell=int(cfg["local_z_min_points_per_cell"]),
        nmad_multiplier=float(cfg["local_z_nmad_multiplier"]),
        min_abs_threshold_m=float(cfg["local_z_min_abs_threshold_m"]),
    )
    quality_flags[local_out] |= QF_LOCAL_Z_OUTLIER
    base_mask &= ~local_out

    local_low = compute_local_low_surface_for_candidates(
        xyz=casals_xyz,
        candidate_mask=base_mask,
        cell_size_m=float(cfg["casals_low_cell_size_m"]),
        min_points_per_cell=int(cfg["casals_low_min_points_per_cell"]),
        low_percentile=float(cfg["casals_low_percentile"]),
    )
    residual_to_local_low = casals_xyz[:, 2] - local_low

    groundlike_mask = (
        base_mask &
        np.isfinite(local_low) &
        (residual_to_local_low >= float(cfg["groundlike_min_residual_to_low_m"])) &
        (residual_to_local_low <= float(cfg["groundlike_max_residual_to_low_m"]))
    )

    quality_flags[base_mask & ~groundlike_mask] |= QF_NOT_GROUNDLIKE_FOR_ALIGNMENT

    candidate_idx = np.flatnonzero(groundlike_mask)
    if candidate_idx.size > int(cfg["max_ground_alignment_casals_points"]):
        rng = np.random.default_rng(int(cfg["random_seed"]))
        candidate_idx = np.sort(
            rng.choice(candidate_idx, size=int(cfg["max_ground_alignment_casals_points"]), replace=False)
        )

    if candidate_idx.size < int(cfg["min_ground_alignment_pairs"]):
        raise RuntimeError(
            f"Too few CASALS ground-like candidates: {candidate_idx.size:,}. "
            "Relax min_ground_snr, low-envelope, or groundlike residual thresholds."
        )

    dep_ground_z_at_casals, dep_ground_count, dep_ground_dist = sample_reference_z_idw(
        query_xy=casals_xyz[candidate_idx, :2],
        ref_xy=dep_ground_xy,
        ref_z=dep_ground_z,
        tree=dep_ground_tree,
        k=int(cfg["ground_idw_k"]),
        radius_m=float(cfg["ground_idw_radius_m"]),
        power=float(cfg["ground_idw_power"]),
    )

    has_ground_reference = (
        np.isfinite(dep_ground_z_at_casals) &
        (dep_ground_count >= int(cfg["ground_idw_min_neighbors"]))
    )
    quality_flags[candidate_idx[~has_ground_reference]] |= QF_NO_3DEP_GROUND_FOR_ALIGNMENT

    pair_idx = candidate_idx[has_ground_reference]
    if pair_idx.size < int(cfg["min_ground_alignment_pairs"]):
        raise RuntimeError(
            f"Too few valid CASALS-3DEP ground alignment pairs: {pair_idx.size:,}. "
            "Increase ground_idw_radius_m or inspect overlap."
        )

    sampled_ground_z = dep_ground_z_at_casals[has_ground_reference]
    raw_dz_samples = sampled_ground_z - casals_xyz[pair_idx, 2]

    inliers = np.isfinite(raw_dz_samples)
    dz = float(np.median(raw_dz_samples[inliers]))

    for _ in range(int(cfg["alignment_clip_iterations"])):
        residual = casals_xyz[pair_idx, 2] + dz - sampled_ground_z
        med = float(np.median(residual[inliers]))
        nmad = robust_nmad(residual[inliers])
        threshold = max(float(cfg["alignment_clip_min_abs_m"]), float(cfg["alignment_clip_nmad_multiplier"]) * nmad)
        new_inliers = np.isfinite(residual) & (np.abs(residual - med) <= threshold)
        if np.count_nonzero(new_inliers) < int(cfg["min_ground_alignment_pairs"]):
            break
        if np.array_equal(new_inliers, inliers):
            inliers = new_inliers
            break
        inliers = new_inliers
        dz = float(np.median(raw_dz_samples[inliers]))

    final_residual = casals_xyz[pair_idx, 2] + dz - sampled_ground_z
    inlier_pair_idx = pair_idx[inliers]

    ground_alignment_inlier = np.zeros(n, dtype=bool)
    ground_alignment_inlier[inlier_pair_idx] = True

    print(f"[INFO] 3DEP ground points: {dep_ground_xyz.shape[0]:,}")
    print(f"[INFO] CASALS high-SNR/local-low ground-like candidates: {candidate_idx.size:,}")
    print(f"[INFO] Valid CASALS-3DEP ground alignment pairs: {pair_idx.size:,}")
    print(f"[INFO] Ground alignment inliers: {np.count_nonzero(inliers):,}")
    print(f"[INFO] Empirical dz added to CASALS Z: {dz:.4f} m")
    print(f"[INFO] Ground residual after dz, NMAD: {robust_nmad(final_residual[inliers]):.4f} m")

    summary = {
        "method": "3DEP class-2 ground IDW sampled at CASALS high-SNR local-low ground-like points",
        "ground_class": int(cfg["ground_class"]),
        "dep3_ground_points": int(dep_ground_xyz.shape[0]),
        "casals_base_candidate_points": int(np.count_nonzero(base_mask)),
        "casals_groundlike_candidate_points_after_subsample": int(candidate_idx.size),
        "valid_ground_alignment_pairs": int(pair_idx.size),
        "ground_alignment_inliers": int(np.count_nonzero(inliers)),
        "dz_m_added_to_casals": float(dz),
        "raw_dz_samples_m_summary": summarize_values(raw_dz_samples),
        "nearest_3dep_ground_horizontal_distance_m_summary": summarize_values(dep_ground_dist[has_ground_reference]),
        "ground_residual_after_dz_all_pairs_m_summary": summarize_values(final_residual),
        "ground_residual_after_dz_inliers_m_summary": summarize_values(final_residual[inliers]),
        "parameters": {
            "min_ground_snr": float(cfg["min_ground_snr"]),
            "require_good_snr_for_ground": bool(cfg["require_good_snr_for_ground"]),
            "casals_low_cell_size_m": float(cfg["casals_low_cell_size_m"]),
            "casals_low_percentile": float(cfg["casals_low_percentile"]),
            "groundlike_min_residual_to_low_m": float(cfg["groundlike_min_residual_to_low_m"]),
            "groundlike_max_residual_to_low_m": float(cfg["groundlike_max_residual_to_low_m"]),
            "ground_idw_radius_m": float(cfg["ground_idw_radius_m"]),
            "ground_idw_k": int(cfg["ground_idw_k"]),
            "ground_idw_min_neighbors": int(cfg["ground_idw_min_neighbors"]),
        },
    }
    return dz, ground_alignment_inlier, summary


def transfer_labels_to_casals(
    casals_aligned_xyz: np.ndarray,
    dep3_xyz: np.ndarray,
    dep3_cls: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    n = casals_aligned_xyz.shape[0]
    k = int(cfg["label_knn"])
    max_dist = float(cfg["label_max_3d_distance_m"])
    strict_dist = float(cfg["label_strict_max_3d_distance_m"])
    min_neighbors = int(cfg["label_min_neighbors"])
    min_ratio = float(cfg["label_min_vote_ratio"])
    chunk_size = int(cfg["label_query_chunk_size"])

    dep3_valid = finite_mask(dep3_xyz[:, 0], dep3_xyz[:, 1], dep3_xyz[:, 2])
    if np.count_nonzero(dep3_valid) == 0:
        raise RuntimeError("No finite 3DEP points available for label transfer.")

    dep3_xyz_valid = dep3_xyz[dep3_valid]
    dep3_cls_valid = dep3_cls[dep3_valid].astype(np.uint8)
    tree = cKDTree(dep3_xyz_valid)
    observed_classes = np.unique(dep3_cls_valid).astype(np.uint8)

    final_class = np.full(n, 1, dtype=np.uint8)
    transfer_status = np.full(n, STATUS_FAR_NO_MATCH, dtype=np.uint8)
    nearest_dist = np.full(n, np.inf, dtype=np.float32)
    nearest_class = np.full(n, 1, dtype=np.uint8)
    dominant_class = np.full(n, 1, dtype=np.uint8)
    vote_ratio = np.zeros(n, dtype=np.float32)
    neighbor_count = np.zeros(n, dtype=np.uint16)

    finite_query = finite_mask(casals_aligned_xyz[:, 0], casals_aligned_xyz[:, 1], casals_aligned_xyz[:, 2])
    transfer_status[~finite_query] = STATUS_NONFINITE_NOISE
    final_class[~finite_query] = LAS_CLASS_LOW_NOISE

    finite_indices = np.flatnonzero(finite_query)

    print(
        f"[INFO] Transferring 3DEP labels to CASALS: "
        f"k={k}, max_3d_dist={max_dist:.2f} m, strict_3d_dist={strict_dist:.2f} m"
    )

    for start in range(0, finite_indices.size, chunk_size):
        end = min(start + chunk_size, finite_indices.size)
        global_idx = finite_indices[start:end]
        q = casals_aligned_xyz[global_idx]

        d, idx = tree.query(q, k=k, distance_upper_bound=max_dist, workers=-1)
        if k == 1:
            d = d[:, None]
            idx = idx[:, None]

        valid = np.isfinite(d) & (idx < dep3_xyz_valid.shape[0])
        counts = np.sum(valid, axis=1).astype(np.uint16)

        chunk_nearest_dist = np.full(global_idx.size, np.inf, dtype=np.float32)
        has_any = counts > 0
        chunk_nearest_dist[has_any] = np.min(np.where(valid, d, np.inf), axis=1)[has_any].astype(np.float32)

        chunk_nearest_class = np.full(global_idx.size, LAS_CLASS_LOW_NOISE, dtype=np.uint8)
        for row in np.flatnonzero(has_any):
            first_col = int(np.flatnonzero(valid[row])[0])
            chunk_nearest_class[row] = dep3_cls_valid[idx[row, first_col]]

        neighbor_cls = np.full(valid.shape, 255, dtype=np.uint8)
        neighbor_cls[valid] = dep3_cls_valid[idx[valid]]

        class_count_stack = np.stack(
            [(neighbor_cls == c).sum(axis=1) for c in observed_classes],
            axis=1,
        )
        winner_col = np.argmax(class_count_stack, axis=1)
        winner_count = class_count_stack[np.arange(global_idx.size), winner_col]
        chunk_dominant_class = observed_classes[winner_col].astype(np.uint8)

        with np.errstate(divide="ignore", invalid="ignore"):
            chunk_ratio = np.divide(
                winner_count,
                counts,
                out=np.zeros_like(winner_count, dtype=np.float32),
                where=(counts > 0),
            ).astype(np.float32)

        reliable_vote = (counts >= min_neighbors) & (chunk_ratio >= min_ratio)
        strict = has_any & reliable_vote & (chunk_nearest_dist <= strict_dist)
        weak = has_any & reliable_vote & (chunk_nearest_dist > strict_dist) & (chunk_nearest_dist <= max_dist)
        ambiguous = has_any & ~(strict | weak)

        chunk_status = np.full(global_idx.size, STATUS_FAR_NO_MATCH, dtype=np.uint8)
        chunk_status[strict] = STATUS_STRICT
        chunk_status[weak] = STATUS_WEAK
        chunk_status[ambiguous] = STATUS_AMBIGUOUS_NEAR

        chunk_final_class = np.full(global_idx.size, 1, dtype=np.uint8)
        chunk_final_class[strict | weak] = chunk_dominant_class[strict | weak]

        final_class[global_idx] = chunk_final_class
        transfer_status[global_idx] = chunk_status
        nearest_dist[global_idx] = chunk_nearest_dist
        nearest_class[global_idx] = chunk_nearest_class
        dominant_class[global_idx] = chunk_dominant_class
        vote_ratio[global_idx] = chunk_ratio
        neighbor_count[global_idx] = counts

        if start == 0 or end == finite_indices.size or ((start // chunk_size) + 1) % 10 == 0:
            print(f"[INFO] Label transfer progress: {end:,} / {finite_indices.size:,} finite CASALS points")

    print("[INFO] Transfer status counts:")
    vals, counts = np.unique(transfer_status, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"  - {STATUS_NAMES.get(int(v), 'unknown')}: {int(c):,}")

    print("[INFO] Output LAS class counts:")
    for cls, count in collect_class_counts(final_class).items():
        print(f"  - class {cls}: {count:,}")

    return {
        "classification": final_class,
        "transfer_status": transfer_status,
        "nearest_3dep_distance_m": nearest_dist,
        "nearest_3dep_class": nearest_class,
        "dominant_3dep_class": dominant_class,
        "class_vote_ratio": vote_ratio,
        "n_3dep_neighbors": neighbor_count,
    }


def add_extra_dims(header: laspy.LasHeader, specs: Iterable[Tuple[str, Any, str]]) -> None:
    dims = [laspy.ExtraBytesParams(name=name, type=dtype, description=desc[:32]) for name, dtype, desc in specs]
    header.add_extra_dims(dims)


def write_labeled_las(
    output_path: Path,
    casals: Dict[str, Any],
    casals_original_xyz: np.ndarray,
    casals_aligned_xyz: np.ndarray,
    quality_flags: np.ndarray,
    ground_alignment_inlier: np.ndarray,
    transfer: Dict[str, np.ndarray],
    target_horizontal_crs: CRS,
    dz_m: float,
) -> None:
    print(f"[INFO] Writing labeled CASALS point cloud: {output_path}")
    ensure_dir(output_path.parent)

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([
        math.floor(float(np.nanmin(casals_aligned_xyz[:, 0]))),
        math.floor(float(np.nanmin(casals_aligned_xyz[:, 1]))),
        math.floor(float(np.nanmin(casals_aligned_xyz[:, 2]))),
    ], dtype=np.float64)
    header.scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    header.vlrs.append(laspy.VLR(user_id="CASALS", record_id=1, record_data=json.dumps({"source_h5": casals["source_h5"], "alignment_mode": casals["alignment_mode"], "independent_accuracy_claim": False}).encode()))

    try:
        header.add_crs(target_horizontal_crs)
    except Exception as exc:
        print(f"[WARN] Could not write CRS to LAS header: {exc}")

    extra_specs: List[Tuple[str, Any, str]] = [
        ("point_index", np.uint32, "CASALS refh point index"),
        ("longitude", np.float64, "Original CASALS longitude"),
        ("latitude", np.float64, "Original CASALS latitude"),
        ("x_original_m", np.float64, "CASALS projected original X"),
        ("y_original_m", np.float64, "CASALS projected original Y"),
        ("z_original_m", np.float64, "Original CASALS refh Z"),
        ("empirical_dz_m", np.float32, "Empirical dz added to Z"),
        ("quality_flag", np.uint8, "CASALS quality bit flag"),
        ("ground_align_inlier", np.uint8, "Used as dz inlier"),
        ("transfer_status", np.uint8, "3DEP transfer status"),
        ("nearest3dep_dist_m", np.float32, "Nearest 3DEP 3D distance"),
        ("nearest3dep_class", np.uint8, "Nearest 3DEP class"),
        ("dominant3dep_class", np.uint8, "Dominant 3DEP class"),
        ("class_vote_ratio", np.float32, "Dominant class vote ratio"),
        ("n_3dep_neighbors", np.uint16, "3DEP neighbors in radius"),
    ]

    optional_field_specs = {
        "refh_snr": np.float32,
        "refh_amp": np.float32,
        "bg_mean": np.float32,
        "bg_std": np.float32,
        "refh_thres": np.float32,
        "good_snr": np.uint8,
        "track_num": np.uint16,
        "sweep_num": np.uint32,
        "delta_time": np.float64,
        "angle": np.float32,
        "refh_error": np.float32,
        "horizontal_error": np.float32,
    }

    for name, dtype in optional_field_specs.items():
        if name in casals["fields"]:
            extra_specs.append((name, dtype, f"CASALS {name}"))

    add_extra_dims(header, extra_specs)

    las = laspy.LasData(header)
    las.x = casals_aligned_xyz[:, 0]
    las.y = casals_aligned_xyz[:, 1]
    las.z = casals_aligned_xyz[:, 2]
    las.classification = transfer["classification"].astype(np.uint8)

    if "refh_amp" in casals["fields"]:
        amp = np.asarray(casals["fields"]["refh_amp"], dtype=np.float64)
        las.intensity = np.clip(np.nan_to_num(amp, nan=0.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    n = casals_aligned_xyz.shape[0]
    las["point_index"] = casals.get("point_index", np.arange(n, dtype=np.uint32))
    las["longitude"] = np.asarray(casals["lon"], dtype=np.float64)
    las["latitude"] = np.asarray(casals["lat"], dtype=np.float64)
    las["x_original_m"] = casals_original_xyz[:, 0].astype(np.float64)
    las["y_original_m"] = casals_original_xyz[:, 1].astype(np.float64)
    las["z_original_m"] = np.asarray(casals["z"], dtype=np.float64)
    las["empirical_dz_m"] = np.full(n, float(dz_m), dtype=np.float32)
    las["quality_flag"] = quality_flags.astype(np.uint8)
    las["ground_align_inlier"] = ground_alignment_inlier.astype(np.uint8)
    las["transfer_status"] = transfer["transfer_status"].astype(np.uint8)
    las["nearest3dep_dist_m"] = transfer["nearest_3dep_distance_m"].astype(np.float32)
    las["nearest3dep_class"] = transfer["nearest_3dep_class"].astype(np.uint8)
    las["dominant3dep_class"] = transfer["dominant_3dep_class"].astype(np.uint8)
    las["class_vote_ratio"] = transfer["class_vote_ratio"].astype(np.float32)
    las["n_3dep_neighbors"] = transfer["n_3dep_neighbors"].astype(np.uint16)

    for name, dtype in optional_field_specs.items():
        if name not in casals["fields"]:
            continue
        arr = casals["fields"][name]
        if dtype == np.uint8:
            arr_out = np.asarray(arr).astype(np.uint8)
        elif dtype == np.uint16:
            arr_out = np.asarray(arr).astype(np.uint16)
        elif dtype == np.uint32:
            arr_out = np.asarray(arr).astype(np.uint32)
        elif dtype == np.float64:
            arr_out = np.asarray(arr, dtype=np.float64)
        else:
            arr_out = np.asarray(arr, dtype=np.float32)
        las[name] = arr_out

    las.write(str(output_path))
    print(f"[INFO] Wrote: {output_path}")


def write_point_csv(
    csv_path: Path,
    casals: Dict[str, Any],
    original_xyz: np.ndarray,
    aligned_xyz: np.ndarray,
    quality_flags: np.ndarray,
    ground_alignment_inlier: np.ndarray,
    transfer: Dict[str, np.ndarray],
) -> None:
    print(f"[INFO] Writing point CSV: {csv_path}")
    ensure_dir(csv_path.parent)
    df = pd.DataFrame({
        "point_index": np.arange(original_xyz.shape[0], dtype=np.int64),
        "lon": casals["lon"],
        "lat": casals["lat"],
        "x_original_m": original_xyz[:, 0],
        "y_original_m": original_xyz[:, 1],
        "z_original_m": original_xyz[:, 2],
        "x_aligned_m": aligned_xyz[:, 0],
        "y_aligned_m": aligned_xyz[:, 1],
        "z_aligned_m": aligned_xyz[:, 2],
        "classification": transfer["classification"].astype(np.uint8),
        "transfer_status_code": transfer["transfer_status"].astype(np.uint8),
        "transfer_status": [STATUS_NAMES.get(int(s), "unknown") for s in transfer["transfer_status"]],
        "nearest_3dep_distance_m": transfer["nearest_3dep_distance_m"],
        "nearest_3dep_class": transfer["nearest_3dep_class"],
        "dominant_3dep_class": transfer["dominant_3dep_class"],
        "class_vote_ratio": transfer["class_vote_ratio"],
        "n_3dep_neighbors": transfer["n_3dep_neighbors"],
        "quality_flag": quality_flags,
        "ground_alignment_inlier": ground_alignment_inlier.astype(np.uint8),
    })
    for name, arr in casals["fields"].items():
        if np.asarray(arr).shape[0] == original_xyz.shape[0]:
            df[name] = arr
    df.to_csv(csv_path, index=False, float_format="%.6f")


def build_summary(
    cfg: Dict[str, Any],
    casals: Dict[str, Any],
    dep3: Dict[str, Any],
    original_xyz: np.ndarray,
    target_horizontal_crs: CRS,
    dz_m: float,
    ground_summary: Dict[str, Any],
    quality_flags: np.ndarray,
    transfer: Dict[str, np.ndarray],
    output_las: Path,
    output_csv: Optional[Path],
    output_json: Path,
) -> Dict[str, Any]:
    status_counts = {
        STATUS_NAMES.get(int(v), str(int(v))): int(c)
        for v, c in zip(*np.unique(transfer["transfer_status"], return_counts=True))
    }
    class_counts = collect_class_counts(transfer["classification"])
    qflag_counts = {str(int(v)): int(c) for v, c in zip(*np.unique(quality_flags, return_counts=True))}

    nearest = transfer["nearest_3dep_distance_m"]
    nearest_finite = np.asarray(nearest, dtype=np.float64)
    nearest_finite[~np.isfinite(nearest_finite)] = np.nan

    return {
        "script_scope": {
            "purpose": "Ground-align CASALS refh Z to 3DEP class-2 ground, transfer 3DEP LAS labels to CASALS, and write labeled CASALS LAS/LAZ.",
            "casals_point_semantics": "CASALS L1B refh/reference-return points, not formal waveform-decomposed multi-return lidar.",
            "vertical_alignment_semantics": cfg["alignment_mode"],
            "far_point_policy": "CASALS points without nearby 3DEP support are written as LAS class 1 unclassified.",
        },
        "inputs": {
            "casals_h5": str(Path(cfg["casals_h5"]).resolve()),
            "dep3_las": str(Path(cfg["dep3_las"]).resolve()),
            "casals_source_datasets": casals["source_datasets"],
            "dep3_header_crs": dep3["crs"].to_string() if dep3["crs"] else None,
            "target_horizontal_crs": target_horizontal_crs.to_string(),
            "dep3_header_summary": dep3["header_summary"],
            "dep3_class_counts": collect_class_counts(dep3["classification"]),
        },
        "counts": {
            "casals_points": int(original_xyz.shape[0]),
            "dep3_points": int(dep3["xyz"].shape[0]),
            "output_class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "transfer_status_counts": status_counts,
            "quality_flag_counts": qflag_counts,
        },
        "alignment": ground_summary,
        "transfer": {
            "label_knn": int(cfg["label_knn"]),
            "label_max_3d_distance_m": float(cfg["label_max_3d_distance_m"]),
            "label_strict_max_3d_distance_m": float(cfg["label_strict_max_3d_distance_m"]),
            "label_min_neighbors": int(cfg["label_min_neighbors"]),
            "label_min_vote_ratio": float(cfg["label_min_vote_ratio"]),

            "nearest_3dep_distance_m_summary": summarize_values(nearest_finite),
            "class_vote_ratio_summary": summarize_values(transfer["class_vote_ratio"]),
            "n_3dep_neighbors_summary": summarize_values(transfer["n_3dep_neighbors"]),
        },
        "coordinate_output": {
            "las_xyz": "audited PROJ XYZ in target CRS" if cfg["alignment_mode"] == "verified_crs" else "CASALS projected XY plus exploratory empirical dz",
            "empirical_dz_m_added_to_all_casals_z": float(dz_m),
            "x_y_crs": target_horizontal_crs.to_2d().to_string(),
            "z_warning": "empirical alignment is not a datum transformation" if cfg["alignment_mode"] == "empirical_diagnostic" else "declared verified CRS conversion is not independent accuracy validation",
        },
        "outputs": {
            "labeled_las_or_laz": str(output_las),
            "point_csv": str(output_csv) if output_csv else None,
            "summary_json": str(output_json),
        },
    }



# -----------------------------------------------------------------------------
# Project-style configuration
# -----------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {

    # Ground alignment settings.
    # 3DEP side uses LAS class 2 by default.
    "ground_class": 2,

    # CASALS side: high-SNR local-low points are treated as tentative ground-like
    # candidates for estimating a single empirical dz.
    "min_ground_snr": 5.0,
    "require_good_snr_for_ground": True,
    "max_ground_alignment_casals_points": 150_000,
    "min_ground_alignment_pairs": 100,

    # Conservative local-Z outlier rejection among CASALS high-SNR candidates.
    "local_z_cell_size_m": 5.0,
    "local_z_min_points_per_cell": 8,
    "local_z_nmad_multiplier": 8.0,
    "local_z_min_abs_threshold_m": 20.0,

    # CASALS local-low surface parameters for extracting ground-like candidates.
    "casals_low_cell_size_m": 5.0,
    "casals_low_min_points_per_cell": 3,
    "casals_low_percentile": 10.0,
    "groundlike_min_residual_to_low_m": -0.5,
    "groundlike_max_residual_to_low_m": 2.0,

    # Sampling 3DEP class-2 ground height at CASALS ground-like XY positions.
    "ground_idw_radius_m": 5.0,
    "ground_idw_k": 8,
    "ground_idw_min_neighbors": 1,
    "ground_idw_power": 3.0,

    # Robust clipping for dz estimation.
    "alignment_clip_iterations": 3,
    "alignment_clip_nmad_multiplier": 4.0,
    "alignment_clip_min_abs_m": 1.0,

    # Label transfer settings.
    # For each aligned CASALS point, search nearby 3DEP points in 3D and assign
    # the dominant 3DEP LAS class. No-match/far points remain class 1.
    "label_knn": 12,
    "label_max_3d_distance_m": 3.0,
    "label_strict_max_3d_distance_m": 2.0,
    "label_min_neighbors": 3,
    "label_min_vote_ratio": 0.65,
    "label_query_chunk_size": 200_000,

    # Ambiguous points remain unclassified. Verified CRS is the default.
    "alignment_mode": "verified_crs",
    "casals_crs_evidence": None,
    "reference_sidecar": None,

    "random_seed": 42,

    # Optional point-level CSV. This can be large. Keep False unless needed.
    "write_point_csv": False,
}


DEFAULT_OUTPUT_DIR = Path("outputs/reference")


def derive_output_stem(casals_h5: Path, dep3_las: Path) -> str:
    del dep3_las
    return casals_h5.stem


def resolve_output_las_path(cfg: Dict[str, Any]) -> Path:
    if cfg.get("output"):
        output_las = Path(cfg["output"]).expanduser().resolve()
    else:
        casals_h5 = Path(cfg["casals_h5"]).expanduser()
        dep3_las = Path(cfg["dep3_las"]).expanduser()
        output_dir = Path(cfg.get("output_dir", DEFAULT_OUTPUT_DIR)).expanduser().resolve()
        output_ext = str(cfg.get("output_ext") or dep3_las.suffix or ".laz").lower()
        output_las = output_dir / f"{derive_output_stem(casals_h5, dep3_las)}{output_ext}"

    if output_las.suffix.lower() not in [".las", ".laz"]:
        raise ValueError("Output path must end with .las or .laz")

    return output_las


def resolve_output_paths(cfg: Dict[str, Any]) -> Tuple[Path, Path, Optional[Path]]:
    output_las = resolve_output_las_path(cfg)

    output_json = (
        Path(cfg["summary_json"]).expanduser().resolve()
        if cfg.get("summary_json")
        else output_las.with_name(output_las.stem + "_summary.json")
    )

    output_csv = None
    if cfg.get("point_csv"):
        output_csv = Path(cfg["point_csv"]).expanduser().resolve()
    elif bool(cfg.get("write_point_csv", False)):
        output_csv = output_las.with_name(output_las.stem + "_points.csv")

    return output_las, output_json, output_csv


def run_job(job_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(CONFIG)
    cfg.update(job_cfg)

    h5_path = Path(cfg["casals_h5"]).expanduser().resolve()
    las_path = Path(cfg["dep3_las"]).expanduser().resolve()
    output_las, output_json, output_csv = resolve_output_paths(cfg)

    print(f"[INFO] Output LAS/LAZ: {output_las}")
    print(f"[INFO] Output summary JSON: {output_json}")
    if output_csv is not None:
        print(f"[INFO] Output point CSV: {output_csv}")

    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not las_path.exists():
        raise FileNotFoundError(las_path)

    ensure_dir(output_las.parent)
    ensure_dir(output_json.parent)
    if output_csv:
        ensure_dir(output_csv.parent)

    casals = read_casals_h5(h5_path)
    dep3 = read_3dep_las(las_path)

    if dep3["crs"] is None:
        raise ValueError("3DEP horizontal CRS unknown; provide documented CRS metadata.")
    horizontal, vertical, frame = resolve_reference_frame(dep3["crs"], cfg.get("reference_sidecar"), cfg.get("ept_metadata"), cfg.get("workunit_metadata"))
    mode = cfg["alignment_mode"]
    casals["alignment_mode"] = mode
    casals_xyz = project_casals_to_target(casals, horizontal)
    quality_flags = build_quality_flags(casals, casals_xyz, min_snr=float(cfg["min_ground_snr"]))
    target_horizontal_crs = horizontal
    if mode == "verified_crs":
        if vertical is None:
            raise ValueError("vertical_crs_unknown; explicitly choose empirical_diagnostic for exploratory labels.")
        target_horizontal_crs = build_target_compound_crs(horizontal, vertical)
        casals_aligned_xyz, operation = transform_verified_xyz(
            np.column_stack([casals["lon"], casals["lat"], casals["z"]]), CRS.from_epsg(4979), target_horizontal_crs,
            source_verified=bool(cfg.get("casals_crs_evidence")),
            target_verified=frame.reference_frame_status.startswith("reference_frame_vertical_verified"))
        dz_m = 0.0
        ground_alignment_inlier = np.zeros(len(casals_xyz), dtype=np.uint8)
        ground_summary = {"operation": operation, "empirical_alignment": False}
    elif mode == "empirical_diagnostic":
        dz_m, ground_alignment_inlier, ground_summary = estimate_ground_vertical_shift(
            casals, casals_xyz, dep3["xyz"], dep3["classification"], cfg, quality_flags)
        casals_aligned_xyz = casals_xyz.copy()
        casals_aligned_xyz[:, 2] += dz_m
        ground_summary.update(empirical_alignment=True, residual_semantics="fitting diagnostics only; no independent elevation validation")
    else:
        raise ValueError("alignment_mode must be verified_crs or empirical_diagnostic")

    transfer = transfer_labels_to_casals(
        casals_aligned_xyz=casals_aligned_xyz,
        dep3_xyz=dep3["xyz"],
        dep3_cls=dep3["classification"],
        cfg=cfg,
    )

    write_labeled_las(
        output_path=output_las,
        casals=casals,
        casals_original_xyz=casals_xyz,
        casals_aligned_xyz=casals_aligned_xyz,
        quality_flags=quality_flags,
        ground_alignment_inlier=ground_alignment_inlier,
        transfer=transfer,
        target_horizontal_crs=target_horizontal_crs,
        dz_m=dz_m,
    )

    if output_csv is not None:
        write_point_csv(
            csv_path=output_csv,
            casals=casals,
            original_xyz=casals_xyz,
            aligned_xyz=casals_aligned_xyz,
            quality_flags=quality_flags,
            ground_alignment_inlier=ground_alignment_inlier,
            transfer=transfer,
        )

    summary = build_summary(
        cfg=cfg,
        casals=casals,
        dep3=dep3,
        original_xyz=casals_xyz,
        target_horizontal_crs=target_horizontal_crs,
        dz_m=dz_m,
        ground_summary=ground_summary,
        quality_flags=quality_flags,
        transfer=transfer,
        output_las=output_las,
        output_csv=output_csv,
        output_json=output_json,
    )
    summary.update(alignment_mode=mode, empirical_alignment=mode == "empirical_diagnostic", reference_frame=asdict(frame), casals_frame={"horizontal": "WGS84", "vertical": "WGS84 ellipsoidal height", "status": "verified by supplied evidence" if cfg.get("casals_crs_evidence") else "pending verification assumption", "evidence": cfg.get("casals_crs_evidence")}, independent_accuracy_claim=False)
    safe_json_dump(summary, output_json)
    print(f"[INFO] Wrote summary JSON: {output_json}")
    print("[INFO] Done.")

    return {
        "output_las": output_las,
        "summary_json": output_json,
        "point_csv": output_csv,
        "summary": summary,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casals-h5", action="append", type=Path, help="CASALS H5 input; repeat in pair order.")
    parser.add_argument("--dep3-las", action="append", type=Path, help="3DEP reference LAS/LAZ; repeat in pair order.")
    parser.add_argument("--output-dir", type=Path, help="Output root (default: outputs/reference/<h5-stem>/transfer).")
    parser.add_argument("--write-point-csv", action="store_true", help="Write a point-level CSV, which may be large.")
    parser.add_argument("--alignment-mode", choices=["verified_crs", "empirical_diagnostic"], default="verified_crs")
    parser.add_argument("--reference-sidecar", type=Path)
    parser.add_argument("--casals-crs-evidence", help="Verified source frame evidence, not an empirical fit.")
    args = parser.parse_args(argv)

    if args.casals_h5 is None and args.dep3_las is None:
        pairs = [
            (
                Path("data/raw/casals_l1b/casals_l1b_20241112T165718_001_02.h5"),
                Path("data/reference/3dep/casals_l1b_20241112T165718_001_02_MD_Southeast_1_2019_EPSG6347_39a068a77804.laz"),
            ),
            (
                Path("data/raw/casals_l1b/casals_l1b_20241118T171757_001_02.h5"),
                Path("data/reference/3dep/casals_l1b_20241118T171757_001_02_NC_HurricaneFlorence_9_2020_EPSG6347_f50533a04725.laz"),
            ),
        ]
    elif args.casals_h5 is None or args.dep3_las is None:
        parser.error("provide both --casals-h5 and --dep3-las, with matching counts")
    else:
        if len(args.casals_h5) != len(args.dep3_las):
            parser.error("provide the same number of --casals-h5 and --dep3-las arguments")
        pairs = list(zip(args.casals_h5, args.dep3_las))

    jobs = [
        {
            "alignment_mode": args.alignment_mode,
            "reference_sidecar": args.reference_sidecar,
            "casals_crs_evidence": args.casals_crs_evidence,
            "casals_h5": h5_path,
            "dep3_las": dep3_path,
            "output_dir": args.output_dir or DEFAULT_OUTPUT_DIR / h5_path.stem / "transfer",
            "write_point_csv": args.write_point_csv,
        }
        for h5_path, dep3_path in pairs
    ]

    for i, job in enumerate(jobs, start=1):
        preview_cfg = dict(CONFIG)
        preview_cfg.update(job)
        job_name = resolve_output_las_path(preview_cfg).stem
        print("=" * 80)
        print(f"[INFO] Running job {i}/{len(jobs)}: {job_name}")
        print("=" * 80)
        run_job(job)


if __name__ == "__main__":
    main()
