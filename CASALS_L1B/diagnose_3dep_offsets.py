#!/usr/bin/env python3
"""Diagnose multi-pair CASALS vs 3DEP vertical offsets.

Purpose
-------
Run the same vertical-offset diagnosis for multiple (CASALS H5, 3DEP LAZ/LAS)
pairs, then compare the offsets across pairs. This is designed to answer whether
CASALS--3DEP vertical discrepancies are:

  1. explained by 3DEP NAVD88/GEOID18 -> ellipsoidal conversion,
  2. explained by adding NAD83(2011) -> WGS84/ITRF frame transformation,
  3. still leaving a repeatable CASALS-side residual term, or
  4. varying by scene/project, indicating local ground extraction/metadata issues.

No argparse is used. Edit main() only.

Outputs
-------
<out_root>/
  per_pair_summary.csv
  per_pair_summary.json
  cross_pair_vertical_diagnosis_report.md
  pair_<label>/
    matched_samples.csv
    pair_report.json
    residual_histograms.png
  figures/
    empirical_offsets_by_pair.png
    formal_residuals_by_pair.png
    offset_decomposition_by_pair.png

Dependencies
------------
conda install -c conda-forge h5py numpy pandas scipy pyproj laspy matplotlib lazrs

Notes
-----
- This script treats CASALS refh as WGS84-like ellipsoidal height, following the
  CASALS documentation convention used in this project.
- This script treats 3DEP LAZ Z as NAVD88 orthometric height by default. For a
  formal transformation, it constructs a compound CRS:
      <3DEP horizontal EPSG> + EPSG:5703 (NAVD88 height)
  and transforms to NAD83(2011) 3D / WGS84 3D / ITRF targets.
- The empirical offset is estimated from CASALS high-SNR refh points and an IDW
  3DEP class-2 ground surface sampled at the CASALS point locations:
      offset = median(CASALS_refh - 3DEP_ground_NAVD88_IDW)
- This is a diagnostic workflow. It does not write corrected LAS files and does
  not replace project-specific CRS and geolocation documentation.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.request
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer, network
try:
    from pyproj.transformer import TransformerGroup
except Exception:  # pragma: no cover
    TransformerGroup = None
from scipy.spatial import cKDTree


# ------------------------------- configuration -------------------------------

@dataclass
class PairConfig:
    label: str
    h5_path: Path
    dep_las_path: Path
    dep_input_epsg_override: Optional[int] = None
    dep_vertical_epsg: int = 5703  # NAVD88 height
    notes: str = ""


@dataclass
class Config:
    pairs: List[PairConfig]
    out_root: Path = Path("./casals_3dep_multipair_vertical_diagnosis")

    # CASALS high-SNR selection
    snr_threshold: float = 5.0
    max_casals_points_per_pair: int = 250_000
    random_seed: int = 20260615

    # 3DEP ground selection / interpolation
    ground_class_code: int = 2
    max_3dep_ground_points_for_tree: Optional[int] = None  # None = all class-2 ground
    idw_k: int = 12
    idw_radius_m: float = 35.0
    idw_power: float = 2.0
    idw_min_neighbors: int = 3

    # Robust statistics
    robust_nmad_factor: float = 1.4826
    robust_clip_nmad_multiplier: float = 3.0
    robust_clip_min_abs_m: float = 0.35

    # Formal transformations
    enable_pyproj_network: bool = True
    formal_targets: Dict[str, str] = field(default_factory=lambda: {
        "NAD83_2011_3D": "EPSG:6319",      # NAD83(2011) geographic 3D
        "WGS84_3D": "EPSG:4979",           # WGS84 geographic 3D
        "ITRF2014_3D": "EPSG:7912",        # ITRF2014 geographic 3D
        "ITRF2020_3D": "EPSG:9989",        # ITRF2020 geographic 3D; may fail on older PROJ
    })

    # Optional NGS geoid API. Online call; can be disabled.
    query_ngs_geoid_api: bool = True
    ngs_geoid_models: Tuple[int, ...] = (14, 13, 12, 11)  # 14=GEOID18 in current NGS API
    ngs_api_pause_s: float = 0.15

    # CSV outputs
    save_matched_sample_limit: int = 50_000

    # Plotting
    dpi: int = 180


# ------------------------------- utilities ------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64, np.uint32, np.uint64)):
        return int(x)
    if isinstance(x, dict):
        return {str(k): as_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [as_jsonable(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(as_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def nmad(arr: np.ndarray, factor: float = 1.4826) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.nanmedian(arr)
    return float(factor * np.nanmedian(np.abs(arr - med)))


def robust_stats(arr: np.ndarray, factor: float = 1.4826) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
            "std": math.nan,
            "nmad": math.nan,
            "p01": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "p50": math.nan,
            "p75": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "abs_le_0p25m": math.nan,
            "abs_le_0p50m": math.nan,
            "abs_le_1p00m": math.nan,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "mae": float(np.mean(np.abs(arr))),
        "std": float(np.std(arr)),
        "nmad": nmad(arr, factor=factor),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "abs_le_0p25m": float(np.mean(np.abs(arr) <= 0.25)),
        "abs_le_0p50m": float(np.mean(np.abs(arr) <= 0.50)),
        "abs_le_1p00m": float(np.mean(np.abs(arr) <= 1.00)),
    }


def robust_inlier_mask(arr: np.ndarray, cfg: Config) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    mask = np.zeros_like(finite, dtype=bool)
    if not finite.any():
        return mask
    med = np.nanmedian(arr[finite])
    scale = nmad(arr[finite], factor=cfg.robust_nmad_factor)
    tol = max(cfg.robust_clip_min_abs_m, cfg.robust_clip_nmad_multiplier * scale)
    mask[finite] = np.abs(arr[finite] - med) <= tol
    return mask


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "pair"


# ------------------------------- H5 reading -----------------------------------

def find_dataset(h5: h5py.File, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in h5:
            return c
    # allow suffix matching in case variables are nested
    names: List[str] = []
    h5.visit(lambda name: names.append(name))
    for c in candidates:
        for name in names:
            if name.split("/")[-1] == c:
                obj = h5[name]
                if isinstance(obj, h5py.Dataset):
                    return name
    raise KeyError(f"None of candidate datasets found: {candidates}")


def read_h5_scalar_attrs(h5: h5py.File) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in h5.attrs.items():
        try:
            if isinstance(v, bytes):
                out[k] = v.decode("utf-8", errors="replace")
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, np.generic):
                out[k] = v.item()
            else:
                out[k] = str(v)
        except Exception:
            out[k] = repr(v)
    return out


def read_casals_high_snr_points(h5_path: Path, cfg: Config) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.random_seed)
    with h5py.File(h5_path, "r") as h5:
        lon_name = find_dataset(h5, ["refh_longitude", "longitude"])
        lat_name = find_dataset(h5, ["refh_latitude", "latitude"])
        refh_name = find_dataset(h5, ["refh"])
        snr_name = find_dataset(h5, ["refh_snr"])

        lon = np.asarray(h5[lon_name][:], dtype=np.float64)
        lat = np.asarray(h5[lat_name][:], dtype=np.float64)
        z = np.asarray(h5[refh_name][:], dtype=np.float64)
        snr = np.asarray(h5[snr_name][:], dtype=np.float64)

        n_total = lon.size
        mask = (
            np.isfinite(lon) & np.isfinite(lat) & np.isfinite(z) & np.isfinite(snr) &
            (snr >= cfg.snr_threshold) &
            (lon >= -180) & (lon <= 180) & (lat >= -90) & (lat <= 90)
        )
        idx = np.flatnonzero(mask)
        n_high = int(idx.size)
        if cfg.max_casals_points_per_pair and idx.size > cfg.max_casals_points_per_pair:
            idx = rng.choice(idx, size=cfg.max_casals_points_per_pair, replace=False)
            idx.sort()

        # Optional variables useful for output/diagnostics.
        optional: Dict[str, np.ndarray] = {}
        for name in [
            "refh_amp", "refh_thres", "track_num", "sweep_num", "delta_time",
            "range_bias_correction", "refh_neutat_delay_total", "geoid", "dac",
            "tide_earth", "tide_load", "tide_ocean", "tide_ocean_pole", "tide_pole",
            "refh_error", "refh_bounce_time_offset",
        ]:
            try:
                ds_name = find_dataset(h5, [name])
                optional[name] = np.asarray(h5[ds_name][idx])
            except Exception:
                pass

        attrs = read_h5_scalar_attrs(h5)

    return {
        "lon": lon[idx],
        "lat": lat[idx],
        "refh": z[idx],
        "snr": snr[idx],
        "record_index": idx,
        "n_total": int(n_total),
        "n_high_snr_total": n_high,
        "n_sampled": int(idx.size),
        "footprint": {
            "lon_min": float(np.nanmin(lon[np.isfinite(lon)])),
            "lon_max": float(np.nanmax(lon[np.isfinite(lon)])),
            "lat_min": float(np.nanmin(lat[np.isfinite(lat)])),
            "lat_max": float(np.nanmax(lat[np.isfinite(lat)])),
            "lon_median": float(np.nanmedian(lon[np.isfinite(lon)])),
            "lat_median": float(np.nanmedian(lat[np.isfinite(lat)])),
        },
        "attrs": attrs,
        "optional": optional,
    }


# ------------------------------- LAS reading ----------------------------------

def get_las_horizontal_crs(las: laspy.LasData, override_epsg: Optional[int]) -> CRS:
    if override_epsg is not None:
        return CRS.from_epsg(override_epsg)
    try:
        crs = las.header.parse_crs()
        if crs is not None:
            return crs
    except Exception:
        pass
    raise RuntimeError("Could not parse 3DEP horizontal CRS from LAS header. Set dep_input_epsg_override in PairConfig.")


def read_dep_ground_points(dep_las_path: Path, pair: PairConfig, cfg: Config) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.random_seed)
    las = laspy.read(dep_las_path)
    crs = get_las_horizontal_crs(las, pair.dep_input_epsg_override)
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    try:
        cls = np.asarray(las.classification)
    except Exception:
        cls = np.ones_like(z, dtype=np.uint8)

    cls_unique, cls_count = np.unique(cls, return_counts=True)
    cls_counts = {int(k): int(v) for k, v in zip(cls_unique, cls_count)}

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (cls == cfg.ground_class_code)
    idx = np.flatnonzero(mask)
    n_ground_total = int(idx.size)
    if cfg.max_3dep_ground_points_for_tree and idx.size > cfg.max_3dep_ground_points_for_tree:
        idx = rng.choice(idx, size=cfg.max_3dep_ground_points_for_tree, replace=False)
        idx.sort()

    return {
        "x": x[idx],
        "y": y[idx],
        "z": z[idx],
        "n_total": int(z.size),
        "n_ground_total": n_ground_total,
        "n_ground_used": int(idx.size),
        "classification_counts": cls_counts,
        "crs": crs,
        "crs_epsg": epsg,
        "crs_name": crs.name,
        "header_summary": {
            "version": str(las.header.version),
            "point_format_id": int(las.header.point_format.id),
            "point_count": int(las.header.point_count),
            "mins": [float(v) for v in las.header.mins],
            "maxs": [float(v) for v in las.header.maxs],
            "scales": [float(v) for v in las.header.scales],
            "offsets": [float(v) for v in las.header.offsets],
        },
    }


# ------------------------------- geoid / transforms ----------------------------

def query_ngs_geoid(lat: float, lon: float, model: int, timeout_s: float = 20.0) -> Optional[Dict[str, Any]]:
    """Query NGS geoid height API. Returns None on failure.

    NGS endpoint usually accepts model id; model=14 is GEOID18. The response JSON
    keys have changed slightly over time, so this function preserves the full response
    and attempts to parse geoidHeight-like fields.
    """
    base = "https://geodesy.noaa.gov/api/geoid/ght"
    params = urllib.parse.urlencode({"lat": f"{lat:.10f}", "lon": f"{lon:.10f}", "model": str(model)})
    url = f"{base}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
    except Exception as e:
        return {"ok": False, "model": model, "lat": lat, "lon": lon, "error": str(e), "url": url}

    h = None
    for key in ["geoidHeight", "geoid_height", "GeoidHeight", "height", "geoid"]:
        if key in data:
            try:
                h = float(data[key])
                break
            except Exception:
                pass
    return {"ok": h is not None, "model": model, "lat": lat, "lon": lon, "geoid_height_m": h, "response": data, "url": url}


def sample_ngs_geoid_over_footprint(footprint: Dict[str, float], cfg: Config) -> Dict[str, Any]:
    if not cfg.query_ngs_geoid_api:
        return {"enabled": False}
    lon_min, lon_max = footprint["lon_min"], footprint["lon_max"]
    lat_min, lat_max = footprint["lat_min"], footprint["lat_max"]
    points = {
        "center_median": (footprint["lat_median"], footprint["lon_median"]),
        "southwest": (lat_min, lon_min),
        "southeast": (lat_min, lon_max),
        "northwest": (lat_max, lon_min),
        "northeast": (lat_max, lon_max),
    }
    rows: List[Dict[str, Any]] = []
    for model in cfg.ngs_geoid_models:
        for loc, (lat, lon) in points.items():
            rec = query_ngs_geoid(lat, lon, model)
            if rec is None:
                rec = {"ok": False, "model": model, "lat": lat, "lon": lon, "error": "None response"}
            rec["location"] = loc
            rows.append(rec)
            time.sleep(cfg.ngs_api_pause_s)
    df = pd.DataFrame(rows)
    summary: Dict[str, Any] = {"enabled": True, "rows": rows, "by_model": {}}
    for model, g in df.groupby("model"):
        vals = pd.to_numeric(g.get("geoid_height_m"), errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        summary["by_model"][str(model)] = robust_stats(vals) if vals.size else {"n": 0}
    return summary


def make_source_compound_crs(dep_horizontal_crs: CRS, vertical_epsg: int) -> CRS:
    epsg = dep_horizontal_crs.to_epsg()
    if epsg is None:
        # Fall back to WKT compound construction may fail; explicit EPSG is strongly preferred.
        raise RuntimeError("3DEP horizontal CRS has no EPSG. Set dep_input_epsg_override for formal transforms.")
    # pyproj accepts EPSG:6347+5703 in current versions.
    return CRS.from_user_input(f"EPSG:{epsg}+{vertical_epsg}")


def formal_transform_z(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    source_compound: CRS,
    target_crs_str: str,
) -> Dict[str, Any]:
    try:
        target = CRS.from_user_input(target_crs_str)
        transformer = Transformer.from_crs(source_compound, target, always_xy=True)
        xo, yo, zo = transformer.transform(x, y, z)
        zo = np.asarray(zo, dtype=float)
        return {
            "success": True,
            "target": target_crs_str,
            "target_name": target.name,
            "z": zo,
            "z_shift": zo - z,
            "error": None,
        }
    except Exception as e:
        return {"success": False, "target": target_crs_str, "error": str(e)}


def transformer_group_info(source: CRS, target_str: str) -> Dict[str, Any]:
    if TransformerGroup is None:
        return {"available": False, "reason": "TransformerGroup unavailable"}
    try:
        target = CRS.from_user_input(target_str)
        tg = TransformerGroup(source, target, always_xy=True)
        info: Dict[str, Any] = {
            "available": True,
            "target": target_str,
            "best_available": bool(getattr(tg, "best_available", False)),
            "n_transformers": len(tg.transformers),
            "unavailable_operations": [],
            "transformers": [],
        }
        for tr in tg.transformers[:5]:
            info["transformers"].append({
                "description": getattr(tr, "description", None),
                "accuracy": getattr(tr, "accuracy", None),
            })
        for op in getattr(tg, "unavailable_operations", [])[:10]:
            grids = []
            for grid in getattr(op, "grids", []) or []:
                grids.append({
                    "short_name": getattr(grid, "short_name", None),
                    "url": getattr(grid, "url", None),
                    "available": getattr(grid, "available", None),
                })
            info["unavailable_operations"].append({
                "name": getattr(op, "name", None),
                "accuracy": getattr(op, "accuracy", None),
                "grids": grids,
            })
        return info
    except Exception as e:
        return {"available": False, "target": target_str, "error": str(e)}


# ------------------------------- matching -------------------------------------

def idw_ground_z_at_points(
    xq: np.ndarray,
    yq: np.ndarray,
    dep_x: np.ndarray,
    dep_y: np.ndarray,
    dep_z: np.ndarray,
    cfg: Config,
) -> Dict[str, np.ndarray]:
    if dep_x.size == 0:
        raise RuntimeError("No 3DEP ground points available.")
    tree = cKDTree(np.column_stack([dep_x, dep_y]))
    d, ii = tree.query(
        np.column_stack([xq, yq]),
        k=cfg.idw_k,
        distance_upper_bound=cfg.idw_radius_m,
        workers=-1,
    )
    if cfg.idw_k == 1:
        d = d[:, None]
        ii = ii[:, None]
    valid = np.isfinite(d) & (ii >= 0) & (ii < dep_z.size)
    n_valid = valid.sum(axis=1)
    ok = n_valid >= cfg.idw_min_neighbors

    ii_safe = np.where(valid, ii, 0)
    z_nei = dep_z[ii_safe]
    weights = np.zeros_like(d, dtype=float)
    exact = valid & (d <= 1e-9)
    non_exact = valid & ~exact
    weights[non_exact] = 1.0 / np.power(d[non_exact], cfg.idw_power)
    # If one or more exact neighbor exists, average exact neighbors only.
    has_exact = exact.any(axis=1)
    weights[exact] = 1.0
    # For rows with exact points, zero non-exact weights.
    weights[has_exact[:, None] & ~exact] = 0.0

    wsum = weights.sum(axis=1)
    z_idw = np.full(xq.shape, np.nan, dtype=float)
    good = ok & (wsum > 0)
    z_idw[good] = (weights[good] * z_nei[good]).sum(axis=1) / wsum[good]
    nearest_dist = np.full(xq.shape, np.nan, dtype=float)
    nearest_dist[valid.any(axis=1)] = np.nanmin(np.where(valid, d, np.nan), axis=1)[valid.any(axis=1)]
    return {"z": z_idw, "ok": good, "n_neighbors": n_valid, "nearest_distance_m": nearest_dist}


def process_pair(pair: PairConfig, cfg: Config) -> Dict[str, Any]:
    print("\n" + "=" * 96)
    print(f"Processing pair: {pair.label}")
    print(f"  H5 : {pair.h5_path}")
    print(f"  3DEP: {pair.dep_las_path}")
    pair_dir = ensure_dir(cfg.out_root / f"pair_{safe_label(pair.label)}")

    casals = read_casals_high_snr_points(pair.h5_path, cfg)
    dep = read_dep_ground_points(pair.dep_las_path, pair, cfg)
    print(f"  CASALS high-SNR sampled: {casals['n_sampled']:,} / high total {casals['n_high_snr_total']:,}")
    print(f"  3DEP ground used: {dep['n_ground_used']:,} / ground total {dep['n_ground_total']:,}")
    print(f"  3DEP CRS: {dep['crs_name']} EPSG={dep['crs_epsg']}")

    if cfg.enable_pyproj_network:
        try:
            network.set_network_enabled(True)
        except Exception:
            pass

    # Project CASALS lon/lat into 3DEP horizontal CRS.
    transformer_xy = Transformer.from_crs(CRS.from_epsg(4326), dep["crs"], always_xy=True)
    xq, yq = transformer_xy.transform(casals["lon"], casals["lat"])
    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)

    # Match to raw 3DEP NAVD88 ground surface by IDW.
    interp_raw = idw_ground_z_at_points(xq, yq, dep["x"], dep["y"], dep["z"], cfg)
    ok = interp_raw["ok"] & np.isfinite(casals["refh"])
    if not ok.any():
        raise RuntimeError(f"No matched CASALS/3DEP samples for pair {pair.label}.")

    x_m = xq[ok]
    y_m = yq[ok]
    lon_m = casals["lon"][ok]
    lat_m = casals["lat"][ok]
    cas_z = casals["refh"][ok]
    cas_snr = casals["snr"][ok]
    dep_z_raw = interp_raw["z"][ok]
    raw_residual = cas_z - dep_z_raw
    empirical_offset = float(np.nanmedian(raw_residual))
    inlier = robust_inlier_mask(raw_residual, cfg)

    print(f"  Matched samples: {ok.sum():,}")
    print(f"  Empirical offset median(CASALS - 3DEP raw): {empirical_offset:.3f} m")
    print(f"  Raw residual NMAD: {nmad(raw_residual, cfg.robust_nmad_factor):.3f} m")

    # NGS geoid at footprint points.
    ngs = sample_ngs_geoid_over_footprint(casals["footprint"], cfg)
    primary_geoid = math.nan
    if ngs.get("enabled") and "14" in ngs.get("by_model", {}):
        primary_geoid = ngs["by_model"]["14"].get("median", math.nan)

    # Formal transforms using raw 3DEP IDW point samples at CASALS points.
    formal_results: Dict[str, Any] = {}
    transformer_groups: Dict[str, Any] = {}
    source_compound = None
    try:
        source_compound = make_source_compound_crs(dep["crs"], pair.dep_vertical_epsg)
        for target_name, target_crs in cfg.formal_targets.items():
            transformer_groups[target_name] = transformer_group_info(source_compound, target_crs)
            fr = formal_transform_z(x_m, y_m, dep_z_raw, source_compound, target_crs)
            if fr.get("success"):
                residual = cas_z - fr["z"]
                fr_summary = {
                    "success": True,
                    "target_crs": target_crs,
                    "target_name": fr.get("target_name"),
                    "z_shift_stats": robust_stats(fr["z_shift"], cfg.robust_nmad_factor),
                    "residual_stats": robust_stats(residual, cfg.robust_nmad_factor),
                    "residual_inlier_stats": robust_stats(residual[robust_inlier_mask(residual, cfg)], cfg.robust_nmad_factor),
                }
                formal_results[target_name] = fr_summary
            else:
                formal_results[target_name] = {"success": False, "target_crs": target_crs, "error": fr.get("error")}
    except Exception as e:
        formal_results["_formal_transform_error"] = str(e)

    # Candidate shifts applied to raw 3DEP ground. Residual = casals - (raw_3dep + shift).
    candidate_rows: List[Dict[str, Any]] = []
    def add_candidate(name: str, shift: float, category: str) -> None:
        if not np.isfinite(shift):
            return
        residual = raw_residual - shift
        stats = robust_stats(residual, cfg.robust_nmad_factor)
        candidate_rows.append({
            "pair_label": pair.label,
            "candidate": name,
            "category": category,
            "applied_shift_m": float(shift),
            **{f"residual_{k}": v for k, v in stats.items()},
        })

    add_candidate("empirical_alignment", empirical_offset, "empirical")
    if np.isfinite(primary_geoid):
        add_candidate("NGS_GEOID18_model14_only", primary_geoid, "geoid_only")
        add_candidate(
            f"NGS_GEOID18_model14_plus_required_extra_{empirical_offset-primary_geoid:+.3f}m",
            empirical_offset,
            "geoid_plus_required_extra",
        )
    # Add all NGS model medians.
    for model, stats in ngs.get("by_model", {}).items():
        med = stats.get("median")
        if med is not None and np.isfinite(med):
            add_candidate(f"NGS_geoid_model_{model}_only", float(med), "geoid_only")
    # Add formal shifts.
    for target_name, fr in formal_results.items():
        if isinstance(fr, dict) and fr.get("success"):
            zshift_med = fr["z_shift_stats"].get("median")
            add_candidate(f"formal_{target_name}", zshift_med, "formal_transform")
            add_candidate(
                f"formal_{target_name}_plus_required_extra_{empirical_offset - zshift_med:+.3f}m",
                empirical_offset,
                "formal_plus_required_extra",
            )
    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(pair_dir / "height_shift_candidate_residuals.csv", index=False)

    # Save matched sample subset for auditing.
    sample_df = pd.DataFrame({
        "lon": lon_m,
        "lat": lat_m,
        "x_dep_crs": x_m,
        "y_dep_crs": y_m,
        "casals_refh": cas_z,
        "casals_snr": cas_snr,
        "dep_ground_z_raw": dep_z_raw,
        "raw_residual_casals_minus_3dep": raw_residual,
        "nearest_3dep_ground_distance_m": interp_raw["nearest_distance_m"][ok],
        "n_3dep_neighbors": interp_raw["n_neighbors"][ok],
        "robust_inlier_raw_residual": inlier,
    })
    # Add optional CASALS fields if they align with ok mask.
    for name, arr in casals["optional"].items():
        try:
            sample_df[f"casals_{name}"] = arr[ok]
        except Exception:
            pass
    if cfg.save_matched_sample_limit and len(sample_df) > cfg.save_matched_sample_limit:
        sample_df_save = sample_df.sample(n=cfg.save_matched_sample_limit, random_state=cfg.random_seed).sort_index()
    else:
        sample_df_save = sample_df
    sample_df_save.to_csv(pair_dir / "matched_samples.csv", index=False)

    # Plots per pair.
    fig, ax = plt.subplots(figsize=(8, 5), dpi=cfg.dpi)
    ax.hist(raw_residual, bins=120, alpha=0.45, label="raw CASALS - 3DEP NAVD88")
    ax.axvline(empirical_offset, color="k", linewidth=2, label=f"empirical median={empirical_offset:.3f} m")
    if np.isfinite(primary_geoid):
        ax.axvline(primary_geoid, linestyle="--", linewidth=1.7, label=f"GEOID18={primary_geoid:.3f} m")
    for target_name in ["WGS84_3D", "ITRF2014_3D", "ITRF2020_3D"]:
        fr = formal_results.get(target_name)
        if fr and fr.get("success"):
            shift = fr["z_shift_stats"].get("median")
            ax.axvline(shift, linestyle=":", linewidth=1.7, label=f"{target_name} shift={shift:.3f} m")
    ax.set_xlabel("CASALS refh - raw 3DEP ground Z (m)")
    ax.set_ylabel("count")
    ax.set_title(f"Vertical offset candidates: {pair.label}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pair_dir / "residual_histograms.png")
    plt.close(fig)

    pair_report = {
        "script": "diagnose_3dep_offsets.py",
        "scientific_notes": [
            "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
            "refh is WGS84 ellipsoidal height unless otherwise documented.",
            "This is not an official multi-return point cloud.",
            "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            "3DEP comparisons must explicitly document vertical datum and terrestrial reference frame differences.",
        ],
        "pair": asdict(pair),
        "casals": {
            "n_total": casals["n_total"],
            "n_high_snr_total": casals["n_high_snr_total"],
            "n_sampled": casals["n_sampled"],
            "footprint": casals["footprint"],
            "root_attrs_subset": {k: v for k, v in casals["attrs"].items() if any(s in k.lower() for s in ["epoch", "time", "date", "crs", "datum", "ellipsoid", "geoid"])}
        },
        "dep": {
            "path": str(pair.dep_las_path),
            "n_total": dep["n_total"],
            "n_ground_total": dep["n_ground_total"],
            "n_ground_used": dep["n_ground_used"],
            "classification_counts": dep["classification_counts"],
            "crs_name": dep["crs_name"],
            "crs_epsg": dep["crs_epsg"],
            "header_summary": dep["header_summary"],
        },
        "matching": {
            "n_matched": int(ok.sum()),
            "idw_k": cfg.idw_k,
            "idw_radius_m": cfg.idw_radius_m,
            "idw_min_neighbors": cfg.idw_min_neighbors,
        },
        "empirical_offset": {
            "median_raw_residual_m": empirical_offset,
            "raw_residual_stats": robust_stats(raw_residual, cfg.robust_nmad_factor),
            "raw_residual_inlier_stats": robust_stats(raw_residual[inlier], cfg.robust_nmad_factor),
        },
        "ngs_geoid": ngs,
        "source_compound_crs": source_compound.to_string() if source_compound is not None else None,
        "formal_transform_results": formal_results,
        "transformer_group_info": transformer_groups,
        "candidate_residuals_csv": str(pair_dir / "height_shift_candidate_residuals.csv"),
        "matched_samples_csv": str(pair_dir / "matched_samples.csv"),
    }
    write_json(pair_dir / "pair_report.json", pair_report)

    # Summary row.
    row: Dict[str, Any] = {
        "label": pair.label,
        "h5_path": str(pair.h5_path),
        "dep_las_path": str(pair.dep_las_path),
        "dep_crs_epsg": dep["crs_epsg"],
        "dep_crs_name": dep["crs_name"],
        "casals_lon_median": casals["footprint"]["lon_median"],
        "casals_lat_median": casals["footprint"]["lat_median"],
        "casals_n_total": casals["n_total"],
        "casals_n_high_snr_total": casals["n_high_snr_total"],
        "casals_n_sampled": casals["n_sampled"],
        "dep_n_total": dep["n_total"],
        "dep_n_ground_total": dep["n_ground_total"],
        "n_matched": int(ok.sum()),
        "empirical_offset_m": empirical_offset,
        "empirical_offset_nmad_m": nmad(raw_residual, cfg.robust_nmad_factor),
        "empirical_offset_p05_m": float(np.percentile(raw_residual, 5)),
        "empirical_offset_p95_m": float(np.percentile(raw_residual, 95)),
        "ngs_model14_geoid_median_m": primary_geoid,
        "empirical_minus_ngs_model14_m": empirical_offset - primary_geoid if np.isfinite(primary_geoid) else math.nan,
    }
    for target_name, fr in formal_results.items():
        if isinstance(fr, dict) and fr.get("success"):
            shift = fr["z_shift_stats"].get("median")
            res_med = fr["residual_stats"].get("median")
            row[f"formal_{target_name}_shift_median_m"] = shift
            row[f"formal_{target_name}_residual_median_m"] = res_med
            row[f"empirical_minus_formal_{target_name}_m"] = empirical_offset - shift
        else:
            row[f"formal_{target_name}_error"] = fr.get("error") if isinstance(fr, dict) else str(fr)

    return {"row": row, "pair_report": pair_report, "candidate_df": candidate_df}


# ------------------------------- aggregate ------------------------------------

def make_aggregate_plots(summary_df: pd.DataFrame, out_root: Path, cfg: Config) -> None:
    fig_dir = ensure_dir(out_root / "figures")
    if summary_df.empty or "empirical_offset_m" not in summary_df.columns:
        return
    summary_df = summary_df.loc[np.isfinite(pd.to_numeric(summary_df["empirical_offset_m"], errors="coerce"))].copy()
    if summary_df.empty:
        return

    labels = summary_df["label"].astype(str).tolist()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(labels)), 5), dpi=cfg.dpi)
    ax.bar(x, summary_df["empirical_offset_m"].to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("offset (m)")
    ax.set_title("Empirical CASALS - raw 3DEP ground offset by pair")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "empirical_offsets_by_pair.png")
    plt.close(fig)

    # Residuals after candidate formal transformations.
    residual_cols = [c for c in summary_df.columns if c.startswith("formal_") and c.endswith("_residual_median_m")]
    if residual_cols:
        fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(labels)), 5), dpi=cfg.dpi)
        width = 0.8 / max(1, len(residual_cols))
        for i, c in enumerate(residual_cols):
            vals = summary_df[c].to_numpy(dtype=float)
            ax.bar(x + (i - (len(residual_cols) - 1) / 2) * width, vals, width=width, label=c.replace("formal_", "").replace("_residual_median_m", ""))
        ax.axhline(0, color="k", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("median residual after formal transform (m)")
        ax.set_title("Does formal datum/frame transformation remove the bias?")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "formal_residuals_by_pair.png")
        plt.close(fig)

    # Decompose: NGS GEOID18 + extra to empirical; and WGS84/ITRF + extra if available.
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(labels)), 5), dpi=cfg.dpi)
    geoid = summary_df.get("ngs_model14_geoid_median_m", pd.Series([np.nan] * len(summary_df))).to_numpy(dtype=float)
    emp = summary_df["empirical_offset_m"].to_numpy(dtype=float)
    extra_geoid = emp - geoid
    ax.bar(x - 0.18, geoid, width=0.35, label="GEOID18 term")
    ax.bar(x - 0.18, extra_geoid, bottom=geoid, width=0.35, label="extra after GEOID18")
    wgs_col = "formal_WGS84_3D_shift_median_m"
    if wgs_col in summary_df.columns:
        wgs = summary_df[wgs_col].to_numpy(dtype=float)
        ax.bar(x + 0.18, wgs, width=0.35, label="formal WGS84 shift")
        ax.bar(x + 0.18, emp - wgs, bottom=wgs, width=0.35, label="extra after formal WGS84")
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("height shift (m)")
    ax.set_title("Offset decomposition by pair")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "offset_decomposition_by_pair.png")
    plt.close(fig)


def make_markdown_report(summary_df: pd.DataFrame, out_root: Path, cfg: Config) -> None:
    lines: List[str] = []
    lines.append("# Multi-pair CASALS–3DEP vertical-offset diagnosis")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This report compares multiple CASALS H5 / 3DEP LAZ pairs to test whether the vertical discrepancy is stable across scenes and 3DEP projects.")
    lines.append("")
    lines.append("## Method summary")
    lines.append("")
    lines.append(f"- CASALS points: `refh_snr >= {cfg.snr_threshold}` high-SNR reference-height points.")
    lines.append(f"- 3DEP ground: LAS/LAZ `Classification == {cfg.ground_class_code}`.")
    lines.append("- Empirical offset: median of `CASALS_refh - raw_3DEP_ground_IDW`.")
    lines.append("- Formal transformations: 3DEP horizontal CRS plus NAVD88 height EPSG:5703 transformed to NAD83(2011) 3D, WGS84 3D, and ITRF targets when available.")
    lines.append("")
    if "empirical_offset_m" in summary_df.columns:
        successful_df = summary_df.loc[np.isfinite(pd.to_numeric(summary_df["empirical_offset_m"], errors="coerce"))].copy()
    else:
        successful_df = pd.DataFrame()
    if not successful_df.empty:
        emp = successful_df["empirical_offset_m"].to_numpy(dtype=float)
        lines.append("## Cross-pair headline")
        lines.append("")
        lines.append(f"- Number of successful pairs: `{len(successful_df)}`")
        lines.append(f"- Empirical offset median across pairs: `{np.nanmedian(emp):.4f}` m")
        lines.append(f"- Empirical offset range across pairs: `{np.nanmin(emp):.4f}` to `{np.nanmax(emp):.4f}` m")
        lines.append(f"- Empirical offset NMAD across pairs: `{nmad(emp):.4f}` m")
        lines.append("")
        if "ngs_model14_geoid_median_m" in successful_df.columns:
            extra = successful_df["empirical_minus_ngs_model14_m"].to_numpy(dtype=float)
            lines.append(f"- Extra term after NGS model 14 / GEOID18 median: median `{np.nanmedian(extra):.4f}` m, range `{np.nanmin(extra):.4f}` to `{np.nanmax(extra):.4f}` m")
        if "empirical_minus_formal_WGS84_3D_m" in successful_df.columns:
            extra = successful_df["empirical_minus_formal_WGS84_3D_m"].to_numpy(dtype=float)
            lines.append(f"- Extra term after formal WGS84 3D transform: median `{np.nanmedian(extra):.4f}` m, range `{np.nanmin(extra):.4f}` to `{np.nanmax(extra):.4f}` m")
        if "empirical_minus_formal_ITRF2014_3D_m" in successful_df.columns:
            extra = successful_df["empirical_minus_formal_ITRF2014_3D_m"].to_numpy(dtype=float)
            lines.append(f"- Extra term after formal ITRF2014 3D transform: median `{np.nanmedian(extra):.4f}` m, range `{np.nanmin(extra):.4f}` to `{np.nanmax(extra):.4f}` m")
        lines.append("")

        display_cols = [
            "label", "empirical_offset_m", "empirical_offset_nmad_m",
            "ngs_model14_geoid_median_m", "empirical_minus_ngs_model14_m",
            "formal_WGS84_3D_shift_median_m", "empirical_minus_formal_WGS84_3D_m",
            "formal_ITRF2014_3D_shift_median_m", "empirical_minus_formal_ITRF2014_3D_m",
            "n_matched",
        ]
        display_cols = [c for c in display_cols if c in successful_df.columns]
        lines.append("## Per-pair summary")
        lines.append("")
        lines.append(successful_df[display_cols].to_markdown(index=False))
        lines.append("")

    if "error" in summary_df.columns:
        failed_df = summary_df.loc[summary_df["error"].notna()].copy()
    else:
        failed_df = pd.DataFrame()
    if not failed_df.empty:
        display_cols = [c for c in ["label", "h5_path", "dep_las_path", "error"] if c in failed_df.columns]
        lines.append("## Failed pairs")
        lines.append("")
        lines.append(failed_df[display_cols].to_markdown(index=False))
        lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- If `empirical_minus_formal_WGS84_3D_m` or `empirical_minus_formal_ITRF2014_3D_m` is nearly constant across pairs, the remaining term is likely CASALS-side or a common frame/epoch/correction convention.")
    lines.append("- If that term changes by 3DEP project, investigate project metadata, vertical datum, geoid model, and LAS header issues.")
    lines.append("- If that term changes by CASALS H5/granule while the same 3DEP project is used, investigate CASALS geolocation, correction fields, range-bin convention, or acquisition-specific calibration.")
    lines.append("- If the empirical offset is noisy within a pair, inspect high-SNR ground purity, local support distance, and 3DEP class-2 coverage.")
    lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append("This is a diagnostic analysis. It does not replace project-specific vertical datum metadata, CASALS geolocation processor documentation, or independent ground control validation.")

    (out_root / "cross_pair_vertical_diagnosis_report.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------- main --------------------------------------

def main() -> None:
    # Edit this list. Each group contains one CASALS H5 path and one 3DEP LAZ/LAS path.
    # You may repeat an H5 with different 3DEP clips, or repeat a 3DEP project with different H5s.
    cfg = Config(
        pairs=[
            PairConfig(
                label="casals_20241112_MD_Southeast_1_2019",
                h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
                dep_las_path=Path(r"./point_cloud_data/download_3dep_lpc/casals_l1b_20241112T165718_001_02_MD_Southeast_1_2019_EPSG6347_39a068a77804.laz"),
                dep_input_epsg_override=None,
            ),
            PairConfig(
                label="casals_20241112_DE_Statewide_1_B23",
                h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241112T170442_001_02.h5"),
                dep_las_path=Path(r"./point_cloud_data/download_3dep_lpc/casals_l1b_20241112T170442_001_02_DE_Statewide_1_B23_EPSG6347_b60d6cbd5f2f.laz"),
                dep_input_epsg_override=None,  # set e.g. 6347 if LAS CRS is missing
            ),
            PairConfig(
                label="casals_20241118_NC_HurricaneFlorence_9_2020",
                h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241118T171757_001_02.h5"),
                dep_las_path=Path(r"./point_cloud_data/download_3dep_lpc/casals_l1b_20241118T171757_001_02_NC_HurricaneFlorence_9_2020_EPSG6347_f50533a04725.laz"),
                dep_input_epsg_override=None,
            ),
        ],
        out_root=Path(r"./outputs/diagnose_3dep_offsets"),
        snr_threshold=5.0,
        max_casals_points_per_pair=250_000,
        idw_k=12,
        idw_radius_m=35.0,
        idw_min_neighbors=3,
        enable_pyproj_network=True,
        query_ngs_geoid_api=True,
    )

    ensure_dir(cfg.out_root)
    print("=" * 96)
    print("Multi-pair CASALS vs 3DEP vertical-offset diagnosis")
    print("=" * 96)
    print(f"Output root: {cfg.out_root.resolve()}")
    print(f"Pairs: {len(cfg.pairs)}")
    print(f"SNR threshold: {cfg.snr_threshold}")

    rows: List[Dict[str, Any]] = []
    pair_reports: Dict[str, Any] = {}
    all_candidates: List[pd.DataFrame] = []

    for pair in cfg.pairs:
        try:
            result = process_pair(pair, cfg)
            rows.append(result["row"])
            pair_reports[pair.label] = result["pair_report"]
            all_candidates.append(result["candidate_df"])
        except Exception as e:
            warnings.warn(f"Pair failed: {pair.label}: {e}")
            rows.append({
                "label": pair.label,
                "h5_path": str(pair.h5_path),
                "dep_las_path": str(pair.dep_las_path),
                "error": str(e),
            })

    summary_df = pd.DataFrame(rows)
    summary_csv = cfg.out_root / "per_pair_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    write_json(
        cfg.out_root / "per_pair_summary.json",
        {
            "script": "diagnose_3dep_offsets.py",
            "scientific_notes": [
                "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
                "refh is WGS84 ellipsoidal height unless otherwise documented.",
                "This is not an official multi-return point cloud.",
                "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
                "CASALS versus 3DEP comparisons must explicitly document vertical datum and frame issues.",
            ],
            "config": asdict(cfg),
            "summary": rows,
            "pair_reports": pair_reports,
        },
    )

    if all_candidates:
        pd.concat(all_candidates, ignore_index=True).to_csv(cfg.out_root / "all_pairs_height_shift_candidate_residuals.csv", index=False)

    make_aggregate_plots(summary_df, cfg.out_root, cfg)
    make_markdown_report(summary_df, cfg.out_root, cfg)

    print("\nDone.")
    print(json.dumps({
        "summary_csv": str(summary_csv),
        "summary_json": str(cfg.out_root / "per_pair_summary.json"),
        "report_md": str(cfg.out_root / "cross_pair_vertical_diagnosis_report.md"),
        "figures_dir": str(cfg.out_root / "figures"),
    }, indent=2))


if __name__ == "__main__":
    main()
