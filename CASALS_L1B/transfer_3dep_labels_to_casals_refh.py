#!/usr/bin/env python3
"""
Transfer 3DEP LAS classification labels to CASALS L1B refh points.

This script is intentionally self-contained and flat. It is meant for the
current CASALS L1B exploratory workflow, where one CASALS H5 granule is compared
with one already-clipped 3DEP LAS/LAZ file.

Scientific scope
----------------
- Input CASALS points are L1B refh/reference-return points, not a formal
  waveform-decomposed multi-return point cloud.
- 3DEP classification is used only as an external reference for pseudo-label
  transfer. The output labels should be interpreted as
  `3DEP-reference pseudo-labels`, not CASALS ground-truth classes.
- The alignment implemented here is an empirical translation that minimizes
  residuals between the two point sets. It is not a rigorous datum/frame
  transformation and should not be used to make geodetic claims by itself.
- Points far from the aligned 3DEP cloud are kept and explicitly flagged as
  no-reference-match / spatial-outlier candidates, because their SNR/amp/bg
  statistics are useful for later CASALS noise-rule design.

Main outputs
------------
1. *_casals_3dep_pseudolabeled_aligned.las
   CASALS refh points in the 3DEP horizontal CRS, shifted by the estimated
   empirical offset, with LAS classification populated from strict pseudo-labels
   where possible and extra dimensions storing match confidence and flags.
2. *_casals_3dep_pseudolabeled_points.csv
   Point-wise table for statistics/debugging.
3. *_label_transfer_summary_by_group.csv
   Summary by match status and pseudo-label.
4. *_alignment_and_transfer_summary.json
   Configuration, estimated offset, residual metrics, validation checks, and
   label-transfer counts.

Dependencies
------------
conda install -c conda-forge h5py laspy lazrs pyproj scipy pandas numpy

If LAZ reading fails, ensure either `lazrs` or `laszip` support is installed for
laspy in the active environment.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import laspy
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree


# =============================================================================
# User configuration
# =============================================================================

CONFIG = {
    "casals_h5": r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5",
    "dep3_las": r"./point_cloud_data/download_3dep_lpc/casals_l1b_20241112T165718_001_02_MD_Southeast_1_2019_EPSG6347_39a068a77804.laz",
    "output_dir": r"./outputs/transfer_3dep_labels_to_casals_refh",
    "output_prefix": "casals_l1b_single",

    # If the 3DEP LAS/LAZ header has no parseable CRS, use this EPSG code.
    # Leave as None to infer a WGS84 UTM zone from the CASALS lon/lat footprint.
    "fallback_3dep_epsg": None,

    # CASALS point preselection. These are not final classification thresholds;
    # they only make alignment less sensitive to obvious weak/noisy refh points.
    "use_good_snr_for_alignment_if_available": True,
    "min_refh_snr_for_alignment": 5.0,
    "max_alignment_casals_points": 150_000,
    "random_seed": 42,

    # Local-Z outlier flag on CASALS before reference matching. It is deliberately
    # conservative. Flagged points are kept in output and can still be analyzed.
    "enable_local_z_outlier_flag": True,
    "local_z_cell_size_m": 5.0,
    "local_z_min_points_per_cell": 8,
    "local_z_nmad_multiplier": 8.0,
    "local_z_min_abs_threshold_m": 20.0,

    # 3DEP classes used only for estimating empirical alignment. Ground and
    # building are usually more stable than vegetation, but this is scene-dependent.
    # If too few points remain, the script falls back to all non-noise 3DEP classes.
    "alignment_3dep_classes": [2, 6],
    "fallback_min_alignment_3dep_points": 1000,
    "exclude_3dep_noise_classes_from_fallback": [7, 18],

    # Empirical translation search. The script adds (dx, dy, dz) to CASALS points.
    # Coarse/fine XY search minimizes robust vertical residuals after nearest-XY
    # matching to selected 3DEP alignment points.
    "xy_query_radius_m": 5.0,
    "coarse_xy_range_m": 10.0,
    "coarse_xy_step_m": 1.0,
    "fine_xy_range_m": 1.5,
    "fine_xy_step_m": 0.25,
    "min_alignment_pairs": 100,
    "estimate_xy_offset": False,

    # Label transfer. kNN is done in aligned 3D space against the full 3DEP cloud.
    "label_knn": 12,
    "label_max_3d_distance_m": 4.0,
    "label_strict_max_3d_distance_m": 2.0,
    "label_min_neighbors": 3,
    "label_min_vote_ratio": 0.65,
    "label_query_chunk_size": 200_000,

    # Validation heuristics. These are warnings, not geodetic truth claims.
    "validation_max_abs_offset_m": 50.0,
    "validation_warn_if_offset_hits_grid_edge": True,
    "validation_warn_if_no_match_fraction_gt": 0.90,
    "validation_warn_if_ambiguous_fraction_gt": 0.90,
    "validation_warn_if_strict_weak_fraction_lt": 0.01,
    "validation_warn_if_nearest_distance_p95_gt_m": 3.5,

    # Output control.
    "write_point_csv": True,
    "csv_float_precision": "%.6f",
}


PSEUDO_CLASS_AMBIGUOUS = 254
PSEUDO_CLASS_NO_MATCH = 255

MATCH_STATUS_NO_MATCH = 0
MATCH_STATUS_STRICT = 1
MATCH_STATUS_WEAK = 2
MATCH_STATUS_AMBIGUOUS = 3
MATCH_STATUS_INTERNAL_NOISE = 4

MATCH_STATUS_NAMES = {
    MATCH_STATUS_NO_MATCH: "no_reference_match",
    MATCH_STATUS_STRICT: "strict_pseudolabel",
    MATCH_STATUS_WEAK: "weak_pseudolabel",
    MATCH_STATUS_AMBIGUOUS: "ambiguous_match",
    MATCH_STATUS_INTERNAL_NOISE: "internal_noise_flagged",
}

QF_BAD_GOOD_SNR = 1 << 0
QF_LOW_REFH_SNR = 1 << 1
QF_LOCAL_Z_OUTLIER = 1 << 2
QF_NONFINITE = 1 << 3


@dataclass
class CasalsPointData:
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray
    fields: Dict[str, np.ndarray]
    source_datasets: Dict[str, str]


@dataclass
class Dep3PointData:
    xyz: np.ndarray
    classification: np.ndarray
    crs: Optional[CRS]
    class_counts: Dict[int, int]
    header_summary: Dict[str, Any]


@dataclass
class AlignmentResult:
    alignment_mode: str
    dx_m: float
    dy_m: float
    dz_m: float
    score_nmad_m: float
    score_median_abs_m: float
    median_residual_after_m: float
    n_pairs: int
    xy_query_radius_m: float
    alignment_3dep_classes_used: List[int]
    alignment_3dep_class_counts: Dict[int, int]
    matched_alignment_class_counts: Dict[int, int]
    casals_alignment_sample_size: int
    dep3_alignment_point_count: int
    fallback_alignment_classes_used: bool
    coarse_best_dx_m: float
    coarse_best_dy_m: float
    coarse_best_dz_m: float
    coarse_best_score_nmad_m: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def robust_nmad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def summarize_values(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "nmad": np.nan,
        }
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "nmad": robust_nmad(x),
    }


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def infer_wgs84_utm_crs(lon: np.ndarray, lat: np.ndarray) -> CRS:
    lon_med = float(np.nanmedian(lon))
    lat_med = float(np.nanmedian(lat))
    zone = int(math.floor((lon_med + 180.0) / 6.0) + 1)
    epsg = (32600 if lat_med >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def safe_json_dump(obj: dict, path: Path) -> None:
    def convert(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)


def collect_class_counts(classes: np.ndarray) -> Dict[int, int]:
    vals, counts = np.unique(np.asarray(classes, dtype=np.uint8), return_counts=True)
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

    leaves = {p.split("/")[-1]: p for p in all_paths}
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


def read_optional_dataset(h5: h5py.File, candidates: Sequence[str], n: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    path = find_dataset_path(h5, candidates, required=False)
    if path is None:
        return None, None
    arr = np.asarray(h5[path][...]).reshape(-1)
    if arr.size != n:
        print(f"[WARN] Optional dataset {path} has size {arr.size}, expected {n}; ignoring.")
        return None, None
    return arr, path


def read_casals_h5(h5_path: Path) -> CasalsPointData:
    print(f"[INFO] Reading CASALS H5: {h5_path}")
    fields: Dict[str, np.ndarray] = {}
    source: Dict[str, str] = {}

    with h5py.File(h5_path, "r") as h5:
        lon_path = find_dataset_path(h5, ["refh_longitude", "longitude", "lon"])
        lat_path = find_dataset_path(h5, ["refh_latitude", "latitude", "lat"])
        z_path = find_dataset_path(h5, ["refh", "refh_height", "height", "elevation"])

        lon = np.asarray(h5[lon_path][...], dtype=np.float64).reshape(-1)
        lat = np.asarray(h5[lat_path][...], dtype=np.float64).reshape(-1)
        z = np.asarray(h5[z_path][...], dtype=np.float64).reshape(-1)
        n = lon.size
        if lat.size != n or z.size != n:
            raise ValueError(f"CASALS lon/lat/z sizes differ: {lon.size}, {lat.size}, {z.size}")

        source.update({"lon": lon_path, "lat": lat_path, "z": z_path})

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
            arr, path = read_optional_dataset(h5, candidates, n)
            if arr is not None:
                fields[out_name] = arr
                source[out_name] = path

    if "refh_snr" not in fields and "refh_amp" in fields and "refh_thres" in fields:
        amp = np.asarray(fields["refh_amp"], dtype=np.float64)
        thres = np.asarray(fields["refh_thres"], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            fields["refh_snr"] = np.divide(amp, thres, out=np.full_like(amp, np.nan), where=(thres != 0))
        source["refh_snr"] = "derived_from_refh_amp/refh_thres"
        print("[INFO] Derived refh_snr from refh_amp / refh_thres because refh_snr dataset was absent.")

    print(f"[INFO] CASALS points: {lon.size:,}")
    print(f"[INFO] Optional CASALS fields found: {sorted(fields.keys())}")
    return CasalsPointData(lon=lon, lat=lat, z=z, fields=fields, source_datasets=source)


def read_3dep_las(las_path: Path) -> Dep3PointData:
    print(f"[INFO] Reading 3DEP LAS/LAZ: {las_path}")
    las = laspy.read(str(las_path))
    xyz = np.column_stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z))).astype(np.float64)
    cls = np.asarray(las.classification, dtype=np.uint8)
    crs = las.header.parse_crs()
    class_counts = collect_class_counts(cls)
    header_summary = {
        "version": str(las.header.version),
        "point_format_id": int(las.header.point_format.id),
        "point_count": int(las.header.point_count),
        "mins": [float(v) for v in las.header.mins],
        "maxs": [float(v) for v in las.header.maxs],
        "scales": [float(v) for v in las.header.scales],
        "offsets": [float(v) for v in las.header.offsets],
    }
    print(f"[INFO] 3DEP points: {xyz.shape[0]:,}")
    print(f"[INFO] 3DEP CRS from LAS header: {crs.to_string() if crs else 'None'}")
    print(f"[INFO] 3DEP class counts: {class_counts}")
    return Dep3PointData(
        xyz=xyz,
        classification=cls,
        crs=crs,
        class_counts=class_counts,
        header_summary=header_summary,
    )


def choose_target_crs(casals: CasalsPointData, dep3_crs: Optional[CRS], fallback_epsg: Optional[int]) -> CRS:
    if dep3_crs is not None:
        return dep3_crs
    if fallback_epsg is not None:
        crs = CRS.from_epsg(int(fallback_epsg))
        print(f"[WARN] LAS has no CRS; using fallback EPSG:{fallback_epsg}")
        return crs
    crs = infer_wgs84_utm_crs(casals.lon, casals.lat)
    print(f"[WARN] LAS has no CRS and no fallback EPSG; inferred {crs.to_string()} from CASALS lon/lat.")
    return crs


def project_casals_to_target(casals: CasalsPointData, target_crs: CRS) -> np.ndarray:
    transformer = Transformer.from_crs(CRS.from_epsg(4326), target_crs, always_xy=True)
    x, y = transformer.transform(casals.lon, casals.lat)
    return np.column_stack((np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), casals.z))


def compute_local_z_outliers(xyz: np.ndarray, cfg: dict) -> np.ndarray:
    n = xyz.shape[0]
    out = np.zeros(n, dtype=bool)
    if not cfg["enable_local_z_outlier_flag"]:
        return out

    finite = finite_mask(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    if not np.any(finite):
        return out

    cell = float(cfg["local_z_cell_size_m"])
    min_n = int(cfg["local_z_min_points_per_cell"])
    mult = float(cfg["local_z_nmad_multiplier"])
    min_abs = float(cfg["local_z_min_abs_threshold_m"])

    ix = np.floor(xyz[:, 0] / cell).astype(np.int64)
    iy = np.floor(xyz[:, 1] / cell).astype(np.int64)
    df = pd.DataFrame({"idx": np.arange(n), "ix": ix, "iy": iy, "z": xyz[:, 2], "finite": finite})
    for (_, _), g in df[df["finite"]].groupby(["ix", "iy"], sort=False):
        if len(g) < min_n:
            continue
        idx = g["idx"].to_numpy()
        z = g["z"].to_numpy(dtype=float)
        med = np.median(z)
        sigma = max(robust_nmad(z), 1e-6)
        thresh = max(min_abs, mult * sigma)
        out[idx] = np.abs(z - med) > thresh
    return out


def build_casals_quality_flags(casals: CasalsPointData, xyz: np.ndarray, cfg: dict) -> np.ndarray:
    n = xyz.shape[0]
    flags = np.zeros(n, dtype=np.uint8)
    nonfinite = ~finite_mask(xyz[:, 0], xyz[:, 1], xyz[:, 2], casals.lon, casals.lat)
    flags[nonfinite] |= QF_NONFINITE

    if "good_snr" in casals.fields:
        good = np.asarray(casals.fields["good_snr"]).astype(bool)
        flags[~good] |= QF_BAD_GOOD_SNR

    if "refh_snr" in casals.fields:
        snr = np.asarray(casals.fields["refh_snr"], dtype=float)
        flags[snr < float(cfg["min_refh_snr_for_alignment"])] |= QF_LOW_REFH_SNR

    local_out = compute_local_z_outliers(xyz, cfg)
    flags[local_out] |= QF_LOCAL_Z_OUTLIER
    return flags


def make_alignment_casals_mask(casals: CasalsPointData, xyz: np.ndarray, quality_flags: np.ndarray, cfg: dict) -> np.ndarray:
    mask = finite_mask(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    mask &= (quality_flags & QF_LOCAL_Z_OUTLIER) == 0
    mask &= (quality_flags & QF_NONFINITE) == 0

    if cfg["use_good_snr_for_alignment_if_available"] and "good_snr" in casals.fields:
        mask &= np.asarray(casals.fields["good_snr"]).astype(bool)

    if "refh_snr" in casals.fields:
        mask &= np.asarray(casals.fields["refh_snr"], dtype=float) >= float(cfg["min_refh_snr_for_alignment"])

    return mask


def choose_3dep_alignment_points(dep3_xyz: np.ndarray, dep3_cls: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray, List[int], bool]:
    classes = [int(c) for c in cfg["alignment_3dep_classes"]]
    mask = np.isin(dep3_cls, classes) & finite_mask(dep3_xyz[:, 0], dep3_xyz[:, 1], dep3_xyz[:, 2])
    fallback_used = False

    if int(np.count_nonzero(mask)) < int(cfg["fallback_min_alignment_3dep_points"]):
        noise = [int(c) for c in cfg["exclude_3dep_noise_classes_from_fallback"]]
        mask = (~np.isin(dep3_cls, noise)) & finite_mask(dep3_xyz[:, 0], dep3_xyz[:, 1], dep3_xyz[:, 2])
        fallback_used = True
        print(
            "[WARN] Too few selected 3DEP alignment points; "
            "falling back to all non-noise classes."
        )

    used_classes = sorted(int(c) for c in np.unique(dep3_cls[mask]))
    return dep3_xyz[mask], dep3_cls[mask], used_classes, fallback_used


def subsample_indices(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size <= max_points:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_points, replace=False))


def score_xy_shift(
    casals_xyz: np.ndarray,
    tree_xy: cKDTree,
    dep3_align_xyz: np.ndarray,
    dx: float,
    dy: float,
    query_radius: float,
    min_pairs: int,
) -> Optional[Tuple[float, float, float, int, float]]:
    q = casals_xyz[:, :2].copy()
    q[:, 0] += dx
    q[:, 1] += dy
    dxy, nn = tree_xy.query(q, k=1, distance_upper_bound=query_radius, workers=-1)
    valid = np.isfinite(dxy) & (nn < dep3_align_xyz.shape[0])
    if np.count_nonzero(valid) < min_pairs:
        return None

    z_c = casals_xyz[valid, 2]
    z_r = dep3_align_xyz[nn[valid], 2]
    dz = float(np.median(z_r - z_c))
    residual = z_c + dz - z_r
    score = robust_nmad(residual)
    med_abs = float(np.median(np.abs(residual)))
    med_res = float(np.median(residual))
    return score, med_abs, med_res, int(np.count_nonzero(valid)), dz


def grid_values(center: float, half_range: float, step: float) -> np.ndarray:
    n = int(round((2 * half_range) / step))
    return np.round(center - half_range + step * np.arange(n + 1), 8)


def summarize_alignment_pairs(
    casals_align_xyz: np.ndarray,
    dep3_align_xyz: np.ndarray,
    dep3_align_cls: np.ndarray,
    dx: float,
    dy: float,
    query_radius: float,
) -> Dict[int, int]:
    q = casals_align_xyz[:, :2].copy()
    q[:, 0] += dx
    q[:, 1] += dy
    tree_xy = cKDTree(dep3_align_xyz[:, :2])
    dxy, nn = tree_xy.query(q, k=1, distance_upper_bound=query_radius, workers=-1)
    valid = np.isfinite(dxy) & (nn < dep3_align_xyz.shape[0])
    if not np.any(valid):
        return {}
    return collect_class_counts(dep3_align_cls[nn[valid]])


def estimate_empirical_translation(
    casals_xyz: np.ndarray,
    dep3_xyz: np.ndarray,
    dep3_cls: np.ndarray,
    casals_align_mask: np.ndarray,
    cfg: dict,
) -> AlignmentResult:
    dep3_align_xyz, dep3_align_cls, used_classes, fallback_used = choose_3dep_alignment_points(dep3_xyz, dep3_cls, cfg)
    dep3_align_counts = collect_class_counts(dep3_align_cls)
    casals_idx = subsample_indices(
        casals_align_mask,
        int(cfg["max_alignment_casals_points"]),
        int(cfg["random_seed"]),
    )
    casals_align_xyz = casals_xyz[casals_idx]

    if casals_align_xyz.shape[0] < int(cfg["min_alignment_pairs"]):
        raise RuntimeError(
            f"Too few CASALS alignment candidates: {casals_align_xyz.shape[0]:,}. "
            "Relax SNR/quality thresholds or inspect input data."
        )
    if dep3_align_xyz.shape[0] < int(cfg["min_alignment_pairs"]):
        raise RuntimeError(
            f"Too few 3DEP alignment candidates: {dep3_align_xyz.shape[0]:,}. "
            "Check classification values or alignment_3dep_classes."
        )

    print(f"[INFO] CASALS alignment sample: {casals_align_xyz.shape[0]:,}")
    print(f"[INFO] 3DEP alignment points: {dep3_align_xyz.shape[0]:,}; classes={used_classes}")

    tree_xy = cKDTree(dep3_align_xyz[:, :2])
    query_radius = float(cfg["xy_query_radius_m"])
    min_pairs = int(cfg["min_alignment_pairs"])

    best: Optional[Tuple[Tuple[float, float, int], float, float, float, int]] = None
    best_dx = best_dy = best_dz = 0.0
    alignment_mode = "dz_only" if not bool(cfg["estimate_xy_offset"]) else "xyz_translation"

    if not bool(cfg["estimate_xy_offset"]):
        print("[INFO] XY offset estimation disabled; solving dz only at fixed dx=0, dy=0.")
        result = score_xy_shift(casals_align_xyz, tree_xy, dep3_align_xyz, 0.0, 0.0, query_radius, min_pairs)
        if result is None:
            raise RuntimeError("No valid dz-only alignment solution at dx=0, dy=0. Increase xy_query_radius_m or inspect overlap.")
        score, med_abs, med_res, n_pairs, dz = result
        key = (score, med_abs, -n_pairs)
        best = (key, score, med_abs, med_res, n_pairs)
        best_dx, best_dy, best_dz = 0.0, 0.0, float(dz)
        coarse_best = {
            "dx_m": 0.0,
            "dy_m": 0.0,
            "dz_m": float(dz),
            "score_nmad_m": float(score),
        }
        fine_best = best
        fine_dx, fine_dy, fine_dz = best_dx, best_dy, best_dz
        print(
            f"[INFO] dz-only best: dx=0.000, dy=0.000, dz={best_dz:.3f}, "
            f"NMAD={score:.3f}, pairs={n_pairs:,}"
        )
    else:
        print("[INFO] Coarse XY grid search...")
        for dx in grid_values(0.0, float(cfg["coarse_xy_range_m"]), float(cfg["coarse_xy_step_m"])):
            for dy in grid_values(0.0, float(cfg["coarse_xy_range_m"]), float(cfg["coarse_xy_step_m"])):
                result = score_xy_shift(casals_align_xyz, tree_xy, dep3_align_xyz, float(dx), float(dy), query_radius, min_pairs)
                if result is None:
                    continue
                score, med_abs, med_res, n_pairs, dz = result
                key = (score, med_abs, -n_pairs)
                if best is None or key < best[0]:
                    best = (key, score, med_abs, med_res, n_pairs)
                    best_dx, best_dy, best_dz = float(dx), float(dy), float(dz)

        if best is None:
            raise RuntimeError("No valid coarse alignment solution. Increase xy_query_radius_m or inspect CRS/overlap.")

        coarse_best = {
            "dx_m": best_dx,
            "dy_m": best_dy,
            "dz_m": best_dz,
            "score_nmad_m": float(best[1]),
        }
        print(
            f"[INFO] Coarse best: dx={best_dx:.3f}, dy={best_dy:.3f}, dz={best_dz:.3f}, "
            f"NMAD={best[1]:.3f}, pairs={best[4]:,}"
        )

        fine_best = best
        fine_dx, fine_dy, fine_dz = best_dx, best_dy, best_dz
        print("[INFO] Fine XY grid search...")
        for dx in grid_values(best_dx, float(cfg["fine_xy_range_m"]), float(cfg["fine_xy_step_m"])):
            for dy in grid_values(best_dy, float(cfg["fine_xy_range_m"]), float(cfg["fine_xy_step_m"])):
                result = score_xy_shift(casals_align_xyz, tree_xy, dep3_align_xyz, float(dx), float(dy), query_radius, min_pairs)
                if result is None:
                    continue
                score, med_abs, med_res, n_pairs, dz = result
                key = (score, med_abs, -n_pairs)
                if key < fine_best[0]:
                    fine_best = (key, score, med_abs, med_res, n_pairs)
                    fine_dx, fine_dy, fine_dz = float(dx), float(dy), float(dz)

    matched_class_counts = summarize_alignment_pairs(
        casals_align_xyz=casals_align_xyz,
        dep3_align_xyz=dep3_align_xyz,
        dep3_align_cls=dep3_align_cls,
        dx=fine_dx,
        dy=fine_dy,
        query_radius=query_radius,
    )

    print(
        f"[INFO] Final empirical offset added to CASALS: "
        f"dx={fine_dx:.3f} m, dy={fine_dy:.3f} m, dz={fine_dz:.3f} m"
    )
    print(
        f"[INFO] Alignment residual: NMAD={fine_best[1]:.3f} m, "
        f"median_abs={fine_best[2]:.3f} m, median={fine_best[3]:.3f} m, pairs={fine_best[4]:,}"
    )

    return AlignmentResult(
        alignment_mode=alignment_mode,
        dx_m=fine_dx,
        dy_m=fine_dy,
        dz_m=fine_dz,
        score_nmad_m=float(fine_best[1]),
        score_median_abs_m=float(fine_best[2]),
        median_residual_after_m=float(fine_best[3]),
        n_pairs=int(fine_best[4]),
        xy_query_radius_m=query_radius,
        alignment_3dep_classes_used=used_classes,
        alignment_3dep_class_counts=dep3_align_counts,
        matched_alignment_class_counts=matched_class_counts,
        casals_alignment_sample_size=int(casals_align_xyz.shape[0]),
        dep3_alignment_point_count=int(dep3_align_xyz.shape[0]),
        fallback_alignment_classes_used=bool(fallback_used),
        coarse_best_dx_m=coarse_best["dx_m"],
        coarse_best_dy_m=coarse_best["dy_m"],
        coarse_best_dz_m=coarse_best["dz_m"],
        coarse_best_score_nmad_m=coarse_best["score_nmad_m"],
    )


def status_name_array(status: np.ndarray) -> np.ndarray:
    return np.asarray([MATCH_STATUS_NAMES.get(int(s), "unknown") for s in status], dtype=object)


def transfer_labels_from_3dep(
    casals_aligned_xyz: np.ndarray,
    dep3_xyz: np.ndarray,
    dep3_cls: np.ndarray,
    quality_flags: np.ndarray,
    cfg: dict,
) -> Dict[str, np.ndarray]:
    n = casals_aligned_xyz.shape[0]
    k = int(cfg["label_knn"])
    max_dist = float(cfg["label_max_3d_distance_m"])
    strict_dist = float(cfg["label_strict_max_3d_distance_m"])
    min_neighbors = int(cfg["label_min_neighbors"])
    min_ratio = float(cfg["label_min_vote_ratio"])
    chunk_size = int(cfg["label_query_chunk_size"])

    print("[INFO] Building 3DEP 3D KD-tree for label transfer...")
    dep3_valid = finite_mask(dep3_xyz[:, 0], dep3_xyz[:, 1], dep3_xyz[:, 2])
    dep3_xyz_valid = dep3_xyz[dep3_valid]
    dep3_cls_valid = dep3_cls[dep3_valid]
    tree = cKDTree(dep3_xyz_valid)

    pseudo_class = np.full(n, PSEUDO_CLASS_NO_MATCH, dtype=np.uint8)
    match_status = np.full(n, MATCH_STATUS_NO_MATCH, dtype=np.uint8)
    nearest_dist = np.full(n, np.inf, dtype=np.float32)
    nearest_class = np.full(n, PSEUDO_CLASS_NO_MATCH, dtype=np.uint8)
    vote_ratio = np.zeros(n, dtype=np.float32)
    neighbor_count = np.zeros(n, dtype=np.uint16)

    print(
        f"[INFO] Querying 3DEP neighbors in chunks: k={k}, max_dist={max_dist:.2f} m, "
        f"chunk_size={chunk_size:,}"
    )
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        q = casals_aligned_xyz[start:end]
        d, idx = tree.query(q, k=k, distance_upper_bound=max_dist, workers=-1)
        if k == 1:
            d = d[:, None]
            idx = idx[:, None]

        valid_neighbor = np.isfinite(d) & (idx < dep3_xyz_valid.shape[0])
        nearest_valid = valid_neighbor[:, 0]
        nearest_dist[start:end][nearest_valid] = d[nearest_valid, 0].astype(np.float32)
        nearest_class[start:end][nearest_valid] = dep3_cls_valid[idx[nearest_valid, 0]]
        neighbor_count[start:end] = np.sum(valid_neighbor, axis=1).astype(np.uint16)

        chunk_pseudo = pseudo_class[start:end]
        chunk_status = match_status[start:end]
        chunk_ratio = vote_ratio[start:end]
        chunk_nearest = nearest_dist[start:end]
        chunk_neighbors = neighbor_count[start:end]

        for i in range(end - start):
            m = valid_neighbor[i]
            count = int(np.count_nonzero(m))
            if count == 0:
                continue
            classes = dep3_cls_valid[idx[i, m]].astype(np.uint8)
            vals, counts = np.unique(classes, return_counts=True)
            winner_j = int(np.argmax(counts))
            winner = int(vals[winner_j])
            ratio = float(counts[winner_j] / count)
            chunk_pseudo[i] = winner
            chunk_ratio[i] = ratio

            if count >= min_neighbors and ratio >= min_ratio and chunk_nearest[i] <= strict_dist:
                chunk_status[i] = MATCH_STATUS_STRICT
            elif count >= min_neighbors and ratio >= min_ratio and chunk_nearest[i] <= max_dist:
                chunk_status[i] = MATCH_STATUS_WEAK
            else:
                chunk_pseudo[i] = PSEUDO_CLASS_AMBIGUOUS
                chunk_status[i] = MATCH_STATUS_AMBIGUOUS

        pseudo_class[start:end] = chunk_pseudo
        match_status[start:end] = chunk_status
        vote_ratio[start:end] = chunk_ratio
        if start == 0 or end == n or ((start // chunk_size) + 1) % 10 == 0:
            print(f"[INFO] Label transfer progress: {end:,} / {n:,} CASALS points")

    internal_noise = (quality_flags & (QF_LOCAL_Z_OUTLIER | QF_NONFINITE)) != 0
    overwrite = internal_noise & (match_status != MATCH_STATUS_STRICT)
    match_status[overwrite] = MATCH_STATUS_INTERNAL_NOISE

    counts = pd.Series(match_status).map(MATCH_STATUS_NAMES).value_counts(dropna=False)
    print("[INFO] Match status counts:")
    for name, count in counts.items():
        print(f"  - {name}: {int(count):,}")

    return {
        "pseudo_class": pseudo_class,
        "match_status": match_status,
        "nearest_3dep_distance_m": nearest_dist,
        "nearest_3dep_class": nearest_class,
        "class_vote_ratio": vote_ratio,
        "n_3dep_neighbors": neighbor_count,
    }


def build_point_dataframe(
    casals: CasalsPointData,
    original_xyz: np.ndarray,
    aligned_xyz: np.ndarray,
    quality_flags: np.ndarray,
    alignment_mask: np.ndarray,
    transfer: Dict[str, np.ndarray],
) -> pd.DataFrame:
    status = transfer["match_status"].astype(np.uint8)
    df = pd.DataFrame({
        "point_index": np.arange(original_xyz.shape[0], dtype=np.int64),
        "lon": casals.lon.astype(np.float64),
        "lat": casals.lat.astype(np.float64),
        "x_original_m": original_xyz[:, 0].astype(np.float32),
        "y_original_m": original_xyz[:, 1].astype(np.float32),
        "z_original_m": original_xyz[:, 2].astype(np.float32),
        "x_aligned_m": aligned_xyz[:, 0].astype(np.float32),
        "y_aligned_m": aligned_xyz[:, 1].astype(np.float32),
        "z_aligned_m": aligned_xyz[:, 2].astype(np.float32),
        "alignment_candidate": alignment_mask.astype(np.uint8),
        "casals_quality_flag": quality_flags.astype(np.uint8),
        "pseudo_3dep_class": transfer["pseudo_class"].astype(np.uint8),
        "match_status_code": status,
        "match_status": pd.Categorical(status_name_array(status)),
        "strict_pseudolabel": (status == MATCH_STATUS_STRICT),
        "weak_pseudolabel": (status == MATCH_STATUS_WEAK),
        "ambiguous_match": (status == MATCH_STATUS_AMBIGUOUS),
        "no_reference_match": (status == MATCH_STATUS_NO_MATCH),
        "internal_noise_flagged": (status == MATCH_STATUS_INTERNAL_NOISE),
        "nearest_3dep_distance_m": transfer["nearest_3dep_distance_m"].astype(np.float32),
        "nearest_3dep_class": transfer["nearest_3dep_class"].astype(np.uint8),
        "class_vote_ratio": transfer["class_vote_ratio"].astype(np.float32),
        "n_3dep_neighbors": transfer["n_3dep_neighbors"].astype(np.uint16),
    })

    for name, arr in casals.fields.items():
        if arr.shape[0] != original_xyz.shape[0]:
            continue
        if name == "good_snr":
            df[name] = np.asarray(arr).astype(np.uint8)
        elif np.issubdtype(np.asarray(arr).dtype, np.integer):
            df[name] = arr
        else:
            df[name] = np.asarray(arr, dtype=np.float32)
    return df


def summarize_group_rows(group_level: str, group_key: Dict[str, Any], df: pd.DataFrame, total_n: int) -> Dict[str, Any]:
    numeric_cols = [
        c for c in [
            "refh_snr",
            "refh_amp",
            "bg_mean",
            "bg_std",
            "refh_thres",
            "angle",
            "refh_error",
            "horizontal_error",
            "nearest_3dep_distance_m",
            "class_vote_ratio",
            "n_3dep_neighbors",
            "z_aligned_m",
        ] if c in df.columns
    ]

    row: Dict[str, Any] = {
        "group_level": group_level,
        "n_points": int(len(df)),
        "fraction_of_all_points": float(len(df) / total_n) if total_n else np.nan,
    }
    row.update(group_key)
    for col in numeric_cols:
        s = summarize_values(df[col].to_numpy())
        for stat_name in ["mean", "std", "p25", "median", "p75", "p95", "nmad"]:
            row[f"{col}_{stat_name}"] = s[stat_name]
    return row


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total_n = len(df)
    rows.append(summarize_group_rows("overall", {"match_status": "all", "pseudo_3dep_class": -1}, df, total_n))
    for status, g in df.groupby("match_status", dropna=False, observed=False):
        rows.append(summarize_group_rows("by_match_status", {"match_status": str(status), "pseudo_3dep_class": -1}, g, total_n))
    for keys, g in df.groupby(["match_status", "pseudo_3dep_class"], dropna=False, observed=False):
        status, pseudo_class = keys
        rows.append(
            summarize_group_rows(
                "by_match_status_and_pseudo_class",
                {"match_status": str(status), "pseudo_3dep_class": int(pseudo_class)},
                g,
                total_n,
            )
        )
    return pd.DataFrame(rows)


def write_las_output(
    output_path: Path,
    casals: CasalsPointData,
    aligned_xyz: np.ndarray,
    quality_flags: np.ndarray,
    alignment_mask: np.ndarray,
    transfer: Dict[str, np.ndarray],
    target_crs: CRS,
) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([
        math.floor(float(np.nanmin(aligned_xyz[:, 0]))),
        math.floor(float(np.nanmin(aligned_xyz[:, 1]))),
        math.floor(float(np.nanmin(aligned_xyz[:, 2]))),
    ], dtype=np.float64)
    header.scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    try:
        header.add_crs(target_crs)
    except Exception as exc:
        print(f"[WARN] Could not write CRS to LAS header: {exc}")

    extra_dims = [
        laspy.ExtraBytesParams(name="longitude", type=np.float64, description="CASALS lon deg"),
        laspy.ExtraBytesParams(name="latitude", type=np.float64, description="CASALS lat deg"),
        laspy.ExtraBytesParams(name="align_candidate", type=np.uint8, description="Used in alignment"),
        laspy.ExtraBytesParams(name="casals_quality_flag", type=np.uint8, description="CASALS qflag"),
        laspy.ExtraBytesParams(name="pseudo_3dep_class", type=np.uint8, description="3DEP pseudo class"),
        laspy.ExtraBytesParams(name="match_status", type=np.uint8, description="match status code"),
        laspy.ExtraBytesParams(name="nearest_3dep_dist_m", type=np.float32, description="nearest 3DEP dist m"),
        laspy.ExtraBytesParams(name="nearest_3dep_class", type=np.uint8, description="nearest 3DEP class"),
        laspy.ExtraBytesParams(name="class_vote_ratio", type=np.float32, description="dominant vote ratio"),
        laspy.ExtraBytesParams(name="n_3dep_neighbors", type=np.uint16, description="3DEP neighbors in maxdist"),
    ]

    optional_las_fields = {
        "refh_snr": np.float32,
        "refh_amp": np.float32,
        "bg_mean": np.float32,
        "bg_std": np.float32,
        "refh_thres": np.float32,
        "good_snr": np.uint8,
        "track_num": np.uint16,
        "sweep_num": np.uint32,
        "angle": np.float32,
        "refh_error": np.float32,
        "horizontal_error": np.float32,
    }
    for name, dtype in optional_las_fields.items():
        if name in casals.fields:
            extra_dims.append(laspy.ExtraBytesParams(name=name, type=dtype, description=f"CASALS {name}"[:32]))

    las = laspy.LasData(header)
    las.add_extra_dims(extra_dims)
    las.x = aligned_xyz[:, 0]
    las.y = aligned_xyz[:, 1]
    las.z = aligned_xyz[:, 2]

    pseudo = transfer["pseudo_class"].astype(np.uint8)
    status = transfer["match_status"].astype(np.uint8)
    out_cls = np.ones(pseudo.shape[0], dtype=np.uint8)
    valid_label = np.isin(status, [MATCH_STATUS_STRICT, MATCH_STATUS_WEAK]) & (pseudo < 254)
    out_cls[valid_label] = pseudo[valid_label]
    out_cls[np.isin(status, [MATCH_STATUS_NO_MATCH, MATCH_STATUS_INTERNAL_NOISE])] = 7
    las.classification = out_cls

    las["longitude"] = casals.lon
    las["latitude"] = casals.lat
    las["align_candidate"] = alignment_mask.astype(np.uint8)
    las["casals_quality_flag"] = quality_flags.astype(np.uint8)
    las["pseudo_3dep_class"] = pseudo
    las["match_status"] = status
    las["nearest_3dep_dist_m"] = transfer["nearest_3dep_distance_m"].astype(np.float32)
    las["nearest_3dep_class"] = transfer["nearest_3dep_class"].astype(np.uint8)
    las["class_vote_ratio"] = transfer["class_vote_ratio"].astype(np.float32)
    las["n_3dep_neighbors"] = transfer["n_3dep_neighbors"].astype(np.uint16)

    for name in optional_las_fields:
        if name not in casals.fields:
            continue
        arr = casals.fields[name]
        if name == "good_snr":
            arr = np.asarray(arr).astype(np.uint8)
        elif np.issubdtype(np.asarray(arr).dtype, np.integer):
            arr = np.asarray(arr)
        else:
            arr = np.asarray(arr, dtype=np.float32)
        las[name] = arr

    las.write(str(output_path))
    print(f"[INFO] Wrote LAS: {output_path}")


def verify_las_roundtrip(
    las_path: Path,
    expected_point_count: int,
    target_crs: CRS,
    quality_flags: np.ndarray,
    alignment_mask: np.ndarray,
    transfer: Dict[str, np.ndarray],
    casals: CasalsPointData,
) -> Dict[str, Any]:
    reread = laspy.read(str(las_path))
    failures: List[str] = []
    extra_dim_names = list(reread.point_format.extra_dimension_names)

    def check_exact(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            failures.append(f"{name} mismatch after LAS round-trip")

    def check_close(name: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-6) -> None:
        if actual.shape != expected.shape or not np.allclose(actual, expected, equal_nan=True, atol=atol):
            failures.append(f"{name} mismatch after LAS round-trip")

    check_exact("point_count", np.asarray([reread.header.point_count]), np.asarray([expected_point_count]))
    check_exact("pseudo_3dep_class", np.asarray(reread["pseudo_3dep_class"]), transfer["pseudo_class"].astype(np.uint8))
    check_exact("match_status", np.asarray(reread["match_status"]), transfer["match_status"].astype(np.uint8))
    check_exact("casals_quality_flag", np.asarray(reread["casals_quality_flag"]), quality_flags.astype(np.uint8))
    check_exact("align_candidate", np.asarray(reread["align_candidate"]), alignment_mask.astype(np.uint8))
    check_close("nearest_3dep_dist_m", np.asarray(reread["nearest_3dep_dist_m"]), transfer["nearest_3dep_distance_m"].astype(np.float32), atol=1e-4)
    check_close("class_vote_ratio", np.asarray(reread["class_vote_ratio"]), transfer["class_vote_ratio"].astype(np.float32), atol=1e-6)
    check_exact("n_3dep_neighbors", np.asarray(reread["n_3dep_neighbors"]), transfer["n_3dep_neighbors"].astype(np.uint16))
    check_close("longitude", np.asarray(reread["longitude"]), casals.lon.astype(np.float64), atol=1e-10)
    check_close("latitude", np.asarray(reread["latitude"]), casals.lat.astype(np.float64), atol=1e-10)

    for name in ["refh_snr", "refh_amp", "bg_mean", "bg_std", "refh_thres", "angle", "refh_error", "horizontal_error"]:
        if name in casals.fields and name in extra_dim_names:
            check_close(name, np.asarray(reread[name]), np.asarray(casals.fields[name], dtype=np.float32), atol=1e-4)
    if "good_snr" in casals.fields and "good_snr" in extra_dim_names:
        check_exact("good_snr", np.asarray(reread["good_snr"]), np.asarray(casals.fields["good_snr"]).astype(np.uint8))

    reread_crs = reread.header.parse_crs()
    crs_match = False
    if reread_crs is not None:
        try:
            crs_match = reread_crs == target_crs
        except Exception:
            crs_match = reread_crs.to_wkt() == target_crs.to_wkt()
    if not crs_match:
        failures.append("CRS mismatch after LAS round-trip")

    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "point_count": int(reread.header.point_count),
        "extra_dimension_names": extra_dim_names,
        "reread_crs": reread_crs.to_string() if reread_crs else None,
    }


def build_validation_summary(
    df: pd.DataFrame,
    align: AlignmentResult,
    dep3: Dep3PointData,
    cfg: dict,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    total = len(df)
    strict_n = int(df["strict_pseudolabel"].sum())
    weak_n = int(df["weak_pseudolabel"].sum())
    ambiguous_n = int(df["ambiguous_match"].sum())
    no_match_n = int(df["no_reference_match"].sum())
    internal_noise_n = int(df["internal_noise_flagged"].sum())

    def add_check(name: str, passed: bool, detail: Dict[str, Any]) -> None:
        checks.append({"name": name, "passed": bool(passed), **detail})

    add_check(
        "output_point_count_equals_casals_refh_count",
        int(total) == int(df["point_index"].nunique()),
        {
            "point_count": int(total),
            "unique_point_index_count": int(df["point_index"].nunique()),
        },
    )

    add_check(
        "status_counts_not_degenerate",
        (strict_n + weak_n) / max(total, 1) >= float(cfg["validation_warn_if_strict_weak_fraction_lt"]) and
        no_match_n / max(total, 1) <= float(cfg["validation_warn_if_no_match_fraction_gt"]) and
        ambiguous_n / max(total, 1) <= float(cfg["validation_warn_if_ambiguous_fraction_gt"]),
        {
            "strict": strict_n,
            "weak": weak_n,
            "ambiguous": ambiguous_n,
            "no_match": no_match_n,
            "internal_noise": internal_noise_n,
            "strict_plus_weak_fraction": float((strict_n + weak_n) / max(total, 1)),
            "no_match_fraction": float(no_match_n / max(total, 1)),
            "ambiguous_fraction": float(ambiguous_n / max(total, 1)),
        },
    )

    nearest_stats = summarize_values(df["nearest_3dep_distance_m"].replace(np.inf, np.nan).to_numpy())
    add_check(
        "nearest_distance_distribution_reasonable",
        np.isfinite(nearest_stats["p95"]) and nearest_stats["p95"] <= float(cfg["validation_warn_if_nearest_distance_p95_gt_m"]),
        nearest_stats,
    )

    max_offset = float(cfg["validation_max_abs_offset_m"])
    offset_hits_grid_edge = (
        abs(align.dx_m) >= float(cfg["coarse_xy_range_m"]) or
        abs(align.dy_m) >= float(cfg["coarse_xy_range_m"])
    )
    add_check(
        "dx_dy_dz_not_obviously_abnormal",
        abs(align.dx_m) <= max_offset and abs(align.dy_m) <= max_offset and abs(align.dz_m) <= max_offset and
        (not cfg["validation_warn_if_offset_hits_grid_edge"] or not offset_hits_grid_edge),
        {
            "dx_m": align.dx_m,
            "dy_m": align.dy_m,
            "dz_m": align.dz_m,
            "offset_hits_coarse_grid_edge": bool(offset_hits_grid_edge),
            "abs_offset_threshold_m": max_offset,
        },
    )

    bad_alignment_share = 0.0
    matched_total = sum(align.matched_alignment_class_counts.values())
    if matched_total > 0:
        bad_alignment_share = sum(align.matched_alignment_class_counts.get(c, 0) for c in [7, 18]) / matched_total
    add_check(
        "class_7_18_not_dominating_alignment",
        bad_alignment_share <= 0.10,
        {
            "matched_alignment_class_counts": align.matched_alignment_class_counts,
            "bad_alignment_share_classes_7_18": float(bad_alignment_share),
            "alignment_3dep_classes_used": align.alignment_3dep_classes_used,
            "full_3dep_class_counts": dep3.class_counts,
        },
    )

    return checks


def run_transfer(cfg: dict) -> Dict[str, Any]:
    h5_path = Path(cfg["casals_h5"]).expanduser().resolve()
    las_path = Path(cfg["dep3_las"]).expanduser().resolve()
    out_dir = Path(cfg["output_dir"]).expanduser().resolve()
    prefix = str(cfg["output_prefix"])
    ensure_dir(out_dir)

    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not las_path.exists():
        raise FileNotFoundError(las_path)

    casals = read_casals_h5(h5_path)
    dep3 = read_3dep_las(las_path)
    target_crs = choose_target_crs(casals, dep3.crs, cfg["fallback_3dep_epsg"])
    casals_xyz = project_casals_to_target(casals, target_crs)

    quality_flags = build_casals_quality_flags(casals, casals_xyz, cfg)
    casals_align_mask = make_alignment_casals_mask(casals, casals_xyz, quality_flags, cfg)

    print(f"[INFO] CASALS finite points: {np.count_nonzero(finite_mask(casals_xyz[:, 0], casals_xyz[:, 1], casals_xyz[:, 2])):,}")
    print(f"[INFO] CASALS alignment-candidate points: {np.count_nonzero(casals_align_mask):,}")

    align = estimate_empirical_translation(casals_xyz, dep3.xyz, dep3.classification, casals_align_mask, cfg)
    casals_aligned_xyz = casals_xyz.copy()
    casals_aligned_xyz[:, 0] += align.dx_m
    casals_aligned_xyz[:, 1] += align.dy_m
    casals_aligned_xyz[:, 2] += align.dz_m

    transfer = transfer_labels_from_3dep(casals_aligned_xyz, dep3.xyz, dep3.classification, quality_flags, cfg)
    df = build_point_dataframe(casals, casals_xyz, casals_aligned_xyz, quality_flags, casals_align_mask, transfer)
    summary_df = make_summary_table(df)

    las_out = out_dir / f"{prefix}_casals_3dep_pseudolabeled_aligned.las"
    csv_out = out_dir / f"{prefix}_casals_3dep_pseudolabeled_points.csv"
    summary_csv_out = out_dir / f"{prefix}_label_transfer_summary_by_group.csv"
    json_out = out_dir / f"{prefix}_alignment_and_transfer_summary.json"

    write_las_output(las_out, casals, casals_aligned_xyz, quality_flags, casals_align_mask, transfer, target_crs)
    las_roundtrip = verify_las_roundtrip(
        las_path=las_out,
        expected_point_count=int(casals_xyz.shape[0]),
        target_crs=target_crs,
        quality_flags=quality_flags,
        alignment_mask=casals_align_mask,
        transfer=transfer,
        casals=casals,
    )
    if las_roundtrip["ok"]:
        print("[INFO] LAS extra-dimension round-trip check passed.")
    else:
        print(f"[WARN] LAS round-trip check found issues: {las_roundtrip['failures']}")

    if cfg["write_point_csv"]:
        df.to_csv(csv_out, index=False, float_format=cfg["csv_float_precision"])
        print(f"[INFO] Wrote point CSV: {csv_out}")

    summary_df.to_csv(summary_csv_out, index=False, float_format=cfg["csv_float_precision"])
    print(f"[INFO] Wrote summary CSV: {summary_csv_out}")

    status_counts = df["match_status"].value_counts().to_dict()
    pseudo_counts = df["pseudo_3dep_class"].value_counts().sort_index().to_dict()
    qflag_counts = df["casals_quality_flag"].value_counts().sort_index().to_dict()
    nearest_stats = summarize_values(df["nearest_3dep_distance_m"].replace(np.inf, np.nan).to_numpy())
    validation_checks = build_validation_summary(df, align, dep3, cfg)

    out_summary = {
        "script_scope": {
            "label_semantics": "3DEP-reference pseudo-labels transferred to CASALS L1B refh points",
            "alignment_semantics": "empirical translation minimizing robust residuals; not a rigorous datum/frame transformation",
            "casals_point_semantics": "L1B refh/reference-return points, not official multi-return classified lidar",
        },
        "inputs": {
            "casals_h5": str(h5_path),
            "dep3_las": str(las_path),
            "casals_source_datasets": casals.source_datasets,
            "target_crs": target_crs.to_string(),
            "dep3_header_crs": dep3.crs.to_string() if dep3.crs else None,
            "dep3_header_summary": dep3.header_summary,
            "dep3_class_counts": dep3.class_counts,
        },
        "config": cfg,
        "alignment": asdict(align),
        "counts": {
            "casals_points": int(casals_xyz.shape[0]),
            "dep3_points": int(dep3.xyz.shape[0]),
            "alignment_candidate_points": int(np.count_nonzero(casals_align_mask)),
            "match_status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "pseudo_3dep_class_counts": {str(k): int(v) for k, v in pseudo_counts.items()},
            "casals_quality_flag_counts": {str(k): int(v) for k, v in qflag_counts.items()},
        },
        "nearest_3dep_distance_m_summary": nearest_stats,
        "las_roundtrip_check": las_roundtrip,
        "validation_checks": validation_checks,
        "outputs": {
            "las": str(las_out),
            "point_csv": str(csv_out) if cfg["write_point_csv"] else None,
            "summary_csv": str(summary_csv_out),
            "json": str(json_out),
        },
    }
    safe_json_dump(out_summary, json_out)
    print(f"[INFO] Wrote JSON summary: {json_out}")
    print("[INFO] Done.")
    return {
        "config": cfg,
        "summary": out_summary,
        "point_csv": csv_out if cfg["write_point_csv"] else None,
        "summary_csv": summary_csv_out,
        "json": json_out,
        "las": las_out,
    }


def main() -> None:
    # -------------------------------------------------------------------------
    # USER SETTINGS: edit here.
    # -------------------------------------------------------------------------
    cfg = dict(CONFIG)
    cfg.update({
        "casals_h5": r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5",
        "dep3_las": r"./point_cloud_data/download_3dep_lpc/casals_l1b_20241112T165718_001_02_MD_Southeast_1_2019_EPSG6347_39a068a77804.laz",
        "output_dir": r"./outputs/transfer_3dep_labels_to_casals_refh",
        "output_prefix": "casals_l1b_single",
        "fallback_3dep_epsg": None,
        "write_point_csv": True,
    })
    # -------------------------------------------------------------------------

    run_transfer(cfg)


if __name__ == "__main__":
    main()
