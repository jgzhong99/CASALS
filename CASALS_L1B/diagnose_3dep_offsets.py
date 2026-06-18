#!/usr/bin/env python3
"""Diagnose CASALS-3DEP ground-to-ground vertical offsets.

Purpose
-------
This script performs a research-auditable CASALS-3DEP ground-to-ground vertical
offset diagnosis workflow.

- CASALS coordinates/heights are treated as WGS84 3D (`EPSG:4979`) with
  ellipsoidal height.
- 3DEP LAS/LAZ coordinates are treated as NAD83(2011) horizontal plus NAVD88
  orthometric height, but the vertical CRS provenance is explicitly recorded as
  metadata-verified, sidecar-verified, manually verified, or assumed.
- Raw `CASALS refh - 3DEP raw Z` is only a height-system sanity check.
- The main residual is:
  `CASALS-derived ground height - transformed 3DEP ground height`.
- `SNR > threshold` identifies CASALS ground candidates only. It is not the
  final ground residual sample definition.
- This script does not write corrected LAS, does not overwrite raw CASALS
  `refh`, and does not claim absolute-accuracy certification.

No argparse is used. Edit `main()` only.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from pyproj import CRS, Transformer, network
from pyproj.crs import CompoundCRS
from scipy.spatial import cKDTree

try:
    from pyproj.transformer import TransformerGroup
except Exception:  # pragma: no cover
    TransformerGroup = None

try:  # pragma: no cover
    from osgeo import gdal, osr
except Exception:  # pragma: no cover
    gdal = None
    osr = None


INTERPRETATION_GUIDANCE = [
    "Raw CASALS-3DEP difference is a height-system discrepancy, not same-datum ground error.",
    "The main residual is computed only after transforming 3DEP ground heights to the CASALS target height system and after deriving CASALS ground points from high-SNR candidates.",
    "SNR threshold identifies strong CASALS returns, not guaranteed ground, so low-envelope and robust filtering are required before ground-to-ground residual estimation.",
    "The pointwise residual is CASALS-derived ground height minus transformed 3DEP ground height sampled at the same horizontal location.",
    "The grid residual is CASALS-derived DTM minus transformed 3DEP DTM on a common grid.",
    "GDAL/PROJ command-line results are cross-checks only. Zero-shift results are marked failed or degraded, not averaged into the final result.",
    "The final residual is a diagnostic ground-to-ground agreement metric, not an absolute-accuracy validation.",
]

NGS_GEOID_MODELS = (14, 13, 12, 11)
NGS_API_PAUSE_S = 0.15
TRANSFORM_VALIDATION_SAMPLE_LIMIT = 5000
IDW_POWER = 2.0
LEGACY_FIELD_PATTERNS = (
    "empirical_offset",
    "dep_input_epsg_override",
    "height_shift_candidate",
    "candidate_residual",
)


@dataclass
class PairConfig:
    label: str
    h5_path: Path
    dep_las_path: Path

    dep_horizontal_epsg_override: Optional[int] = None
    dep_vertical_epsg: int = 5703
    dep_vertical_crs_source: str = "assumed_config"
    dep_vertical_crs_verified: bool = False

    casals_crs: str = "EPSG:4979"
    casals_vertical_reference: str = "WGS84 ellipsoidal height"
    casals_vertical_reference_verified: bool = True

    casals_epoch: Optional[float] = None
    dep_epoch: Optional[float] = None

    notes: str = ""


@dataclass
class Config:
    pairs: List[PairConfig]
    out_root: Path = Path("./outputs/diagnose_3dep_ground_offsets")

    casals_snr_ground_candidate_threshold: float = 5.0
    max_casals_points_per_pair: int = 500_000
    random_seed: int = 20260618

    casals_ground_grid_m: float = 5.0
    casals_low_quantile: float = 0.10
    casals_ground_seed_resid_low_m: float = -0.50
    casals_ground_seed_resid_high_m: float = 1.00
    casals_local_outlier_cell_m: float = 5.0
    casals_local_outlier_nmad_multiplier: float = 6.0
    casals_local_outlier_min_abs_m: float = 5.0
    casals_min_neighbors_radius_m: float = 3.0
    casals_min_neighbors_count: int = 3
    casals_dtm_idw_k: int = 12
    casals_dtm_idw_radius_m: float = 35.0
    casals_dtm_idw_min_neighbors: int = 3

    dep_ground_class_code: int = 2
    dep_dtm_idw_k: int = 12
    dep_dtm_idw_radius_m: float = 35.0
    dep_dtm_idw_min_neighbors: int = 3
    max_3dep_ground_points_for_tree: Optional[int] = None

    residual_grid_m: float = 5.0
    min_points_per_grid_cell: int = 3

    enable_pyproj_check: bool = True
    enable_gdal_osr_check: bool = True
    enable_gdaltransform_check: bool = True
    enable_proj_cct_check: bool = True
    enable_ngs_geoid_check: bool = True
    enable_pdal_smrf_casals_ground_check: bool = False

    gdaltransform_exe: Optional[str] = None
    cct_exe: Optional[str] = None
    projinfo_exe: Optional[str] = None
    pdal_exe: Optional[str] = None
    manual_proj_pipeline: Optional[str] = None

    main_target_crs: str = "EPSG:4979"
    auxiliary_target_crs: Dict[str, str] = field(default_factory=lambda: {
        "NAD83_2011_3D": "EPSG:6319",
        "WGS84_3D": "EPSG:4979",
        "ITRF2014_3D": "EPSG:7912",
        "ITRF2020_3D": "EPSG:9989",
    })

    transform_shift_agreement_tol_m: float = 0.05
    two_stage_shift_agreement_tol_m: float = 0.30
    zero_vertical_shift_abs_tol_m: float = 0.01
    warning_residual_abs_m: float = 1.0
    pyproj_vs_ngs_geoid_warning_abs_m: float = 5.0

    robust_nmad_factor: float = 1.4826

    save_debug_point_limit: int = 100_000
    dpi: int = 180


@dataclass
class TransformResult:
    method: str
    source_crs: str
    target_crs: str
    status: str
    z_transformed: Optional[np.ndarray]
    z_shift: Optional[np.ndarray]
    operation_description: Optional[str]
    operation_accuracy: Optional[float]
    best_available: Optional[bool]
    missing_grids: List[Dict[str, Any]]
    command: Optional[str]
    software: Dict[str, Any]
    notes: str
    input_coordinate_type: Optional[str] = None
    z_shift_expected_nonzero: bool = True
    failed_zero_vertical_shift: bool = False
    median_diff_to_pyproj_m: Optional[float] = None
    nmad_diff_to_pyproj_m: Optional[float] = None
    p95_abs_diff_to_pyproj_m: Optional[float] = None
    pipeline_available: Optional[bool] = None
    pipeline_validated: Optional[bool] = None
    projinfo_status: Optional[str] = None
    ngs_geoid_status: Optional[str] = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "pair"


def as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, CRS):
        return value.to_string()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(as_jsonable(value), indent=2, ensure_ascii=False), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def nmad(values: np.ndarray, factor: float = 1.4826) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.nanmedian(arr)
    return float(factor * np.nanmedian(np.abs(arr - med)))


def robust_stats(values: np.ndarray, factor: float = 1.4826) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
            "nmad": math.nan,
            "p01": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "p50": math.nan,
            "p75": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "max_abs": math.nan,
            "fraction_abs_le_0p25m": math.nan,
            "fraction_abs_le_0p50m": math.nan,
            "fraction_abs_le_1m": math.nan,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "mae": float(np.mean(np.abs(arr))),
        "nmad": nmad(arr, factor=factor),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max_abs": float(np.max(np.abs(arr))),
        "fraction_abs_le_0p25m": float(np.mean(np.abs(arr) <= 0.25)),
        "fraction_abs_le_0p50m": float(np.mean(np.abs(arr) <= 0.50)),
        "fraction_abs_le_1m": float(np.mean(np.abs(arr) <= 1.0)),
    }


def robust_inlier_mask(values: np.ndarray, cfg: Config, nmad_multiplier: float = 3.0, min_abs_m: float = 0.5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    mask = np.zeros(arr.shape, dtype=bool)
    if not finite.any():
        return mask
    med = float(np.nanmedian(arr[finite]))
    scale = nmad(arr[finite], cfg.robust_nmad_factor)
    tol = max(min_abs_m, nmad_multiplier * scale if np.isfinite(scale) else min_abs_m)
    mask[finite] = np.abs(arr[finite] - med) <= tol
    return mask


def clipped_stats(values: np.ndarray, cfg: Config, nmad_multiplier: float = 3.0, min_abs_m: float = 0.5) -> Dict[str, Any]:
    mask = robust_inlier_mask(values, cfg, nmad_multiplier=nmad_multiplier, min_abs_m=min_abs_m)
    stats = robust_stats(np.asarray(values, dtype=float)[mask], factor=cfg.robust_nmad_factor)
    stats["clip_n_inlier"] = int(mask.sum())
    stats["clip_fraction_inlier"] = float(np.mean(mask)) if mask.size else math.nan
    return stats


def stats_to_prefixed_row(prefix: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        f"{prefix}_n": stats.get("n"),
        f"{prefix}_mean_m": stats.get("mean"),
        f"{prefix}_median_m": stats.get("median"),
        f"{prefix}_std_m": stats.get("std"),
        f"{prefix}_rmse_m": stats.get("rmse"),
        f"{prefix}_mae_m": stats.get("mae"),
        f"{prefix}_nmad_m": stats.get("nmad"),
        f"{prefix}_p01_m": stats.get("p01"),
        f"{prefix}_p05_m": stats.get("p05"),
        f"{prefix}_p25_m": stats.get("p25"),
        f"{prefix}_p50_m": stats.get("p50"),
        f"{prefix}_p75_m": stats.get("p75"),
        f"{prefix}_p95_m": stats.get("p95"),
        f"{prefix}_p99_m": stats.get("p99"),
        f"{prefix}_fraction_abs_le_0p25m": stats.get("fraction_abs_le_0p25m"),
        f"{prefix}_fraction_abs_le_0p50m": stats.get("fraction_abs_le_0p50m"),
        f"{prefix}_fraction_abs_le_1m": stats.get("fraction_abs_le_1m"),
    }


def decode_h5_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_h5_scalar_attrs(h5: h5py.File) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in h5.attrs.items():
        try:
            out[str(key)] = decode_h5_value(value)
        except Exception:
            out[str(key)] = repr(value)
    return out


def list_h5_dataset_names(h5: h5py.File) -> List[str]:
    names: List[str] = []
    h5.visit(lambda name: names.append(name))
    return names


def find_dataset(h5: h5py.File, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in h5:
            return candidate
    names = list_h5_dataset_names(h5)
    for candidate in candidates:
        for name in names:
            if name.split("/")[-1] == candidate and isinstance(h5[name], h5py.Dataset):
                return name
    raise KeyError(f"Dataset not found from candidates: {candidates}")


def resolve_executable(configured: Optional[str], fallback_name: str) -> Optional[str]:
    if configured:
        path = Path(configured)
        if path.exists():
            return str(path)
        found = shutil.which(configured)
        if found:
            return found
    found = shutil.which(fallback_name)
    if found:
        return found
    candidate_dirs: List[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate_dirs.extend([
            Path(conda_prefix),
            Path(conda_prefix) / "Library" / "bin",
            Path(conda_prefix) / "Scripts",
            Path(conda_prefix) / "bin",
        ])
    candidate_dirs.extend([
        Path(sys.prefix),
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.prefix) / "Scripts",
        Path(sys.prefix) / "bin",
    ])
    suffixes = [fallback_name]
    if not fallback_name.lower().endswith(".exe"):
        suffixes.append(f"{fallback_name}.exe")
    for directory in candidate_dirs:
        for suffix in suffixes:
            candidate = directory / suffix
            if candidate.exists():
                return str(candidate)
    return None


def get_command_version(command: Sequence[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return None
    text = (proc.stdout or proc.stderr or "").strip()
    return text.splitlines()[0].strip() if text else None


def get_proj_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PROJ_NETWORK", "ON")
    return env


def choose_indices(n: int, limit: Optional[int], seed: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if limit is None or limit <= 0 or n <= limit:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False)).astype(int)


def percent_str(num: int, den: int) -> str:
    if den <= 0:
        return "nan"
    return f"{100.0 * num / den:.2f}%"


def is_near_zero_vertical_shift(z_shift: Optional[np.ndarray], cfg: Config) -> bool:
    if z_shift is None:
        return False
    arr = np.asarray(z_shift, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    return bool(np.nanpercentile(np.abs(arr), 95) <= cfg.zero_vertical_shift_abs_tol_m)


def print_pair_header(pair: PairConfig) -> None:
    print("\n" + "=" * 80)
    print(f"PAIR: {pair.label}")
    print("-" * 80)


def print_step(step_num: int, total: int, title: str) -> None:
    print(f"[{step_num}/{total}] {title}")


def read_casals_raw_points(h5_path: Path) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        lon_name = find_dataset(h5, ["refh_longitude", "longitude"])
        lat_name = find_dataset(h5, ["refh_latitude", "latitude"])
        refh_name = find_dataset(h5, ["refh"])
        snr_name = find_dataset(h5, ["refh_snr"])

        lon = np.asarray(h5[lon_name][:], dtype=np.float64)
        lat = np.asarray(h5[lat_name][:], dtype=np.float64)
        refh = np.asarray(h5[refh_name][:], dtype=np.float64)
        snr = np.asarray(h5[snr_name][:], dtype=np.float64)

        optional_arrays: Dict[str, np.ndarray] = {}
        for name in ["refh_amp", "amp", "geoid"]:
            try:
                optional_arrays[name] = np.asarray(h5[find_dataset(h5, [name])][:], dtype=np.float64)
            except Exception:
                continue

        attrs = read_h5_scalar_attrs(h5)
        dataset_names = list_h5_dataset_names(h5)

    return {
        "lon": lon,
        "lat": lat,
        "refh": refh,
        "refh_snr": snr,
        "record_index": np.arange(lon.size, dtype=int),
        "optional": optional_arrays,
        "attrs": attrs,
        "dataset_names": dataset_names,
        "n_total": int(lon.size),
    }


def inspect_casals_crs(h5_path: Path, pair: PairConfig) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        attrs = read_h5_scalar_attrs(h5)
        dataset_names = list_h5_dataset_names(h5)
        geoid_stats = None
        try:
            geoid_ds = find_dataset(h5, ["geoid"])
            geoid_stats = robust_stats(np.asarray(h5[geoid_ds][:], dtype=np.float64))
            has_geoid_dataset = True
        except Exception:
            has_geoid_dataset = False

    keywords = ("datum", "ellipsoid", "geoid", "height", "vertical", "crs", "reference", "epoch")
    metadata_hits: List[Dict[str, Any]] = []
    for key, value in attrs.items():
        text = f"{key}: {value}"
        if any(word in text.lower() for word in keywords):
            metadata_hits.append({"source": "root_attr", "key": key, "value": value})
    for name in dataset_names:
        if any(word in name.lower() for word in keywords):
            metadata_hits.append({"source": "dataset_name", "path": name})

    epoch_attrs = {key: value for key, value in attrs.items() if "epoch" in key.lower() or "time" in key.lower()}

    return {
        "casals_crs": pair.casals_crs,
        "casals_vertical_reference": pair.casals_vertical_reference,
        "casals_vertical_reference_verified": bool(pair.casals_vertical_reference_verified),
        "h5_metadata_hits": metadata_hits,
        "h5_contains_geoid_dataset": has_geoid_dataset,
        "h5_contains_refh_dataset": any(name.split("/")[-1] == "refh" for name in dataset_names),
        "h5_contains_refh_snr_dataset": any(name.split("/")[-1] == "refh_snr" for name in dataset_names),
        "h5_epoch_attrs": epoch_attrs,
        "geoid_dataset_stats": geoid_stats,
    }


def inspect_las_crs(las: laspy.LasData, pair: PairConfig) -> Dict[str, Any]:
    parsed_crs: Optional[CRS] = None
    parse_error = None
    try:
        parsed_crs = las.header.parse_crs()
    except Exception as exc:
        parse_error = str(exc)

    header_horizontal: Optional[CRS] = None
    header_vertical: Optional[CRS] = None
    if parsed_crs is not None and parsed_crs.is_compound:
        for sub in parsed_crs.sub_crs_list:
            if sub.is_vertical and header_vertical is None:
                header_vertical = sub
            elif header_horizontal is None:
                header_horizontal = sub
    elif parsed_crs is not None:
        header_horizontal = parsed_crs

    horizontal_crs = header_horizontal
    if pair.dep_horizontal_epsg_override is not None:
        horizontal_crs = CRS.from_epsg(pair.dep_horizontal_epsg_override)

    vertical_crs = header_vertical
    vertical_epsg: Optional[int] = None
    vertical_name: Optional[str] = None
    vertical_crs_source = pair.dep_vertical_crs_source
    vertical_crs_verified = bool(pair.dep_vertical_crs_verified)

    if header_vertical is not None:
        vertical_crs = header_vertical
        vertical_epsg = header_vertical.to_epsg()
        vertical_name = header_vertical.name
        vertical_crs_source = "las_header"
        vertical_crs_verified = True
    else:
        vertical_crs = CRS.from_epsg(pair.dep_vertical_epsg)
        vertical_epsg = pair.dep_vertical_epsg
        vertical_name = vertical_crs.name

    horizontal_epsg = horizontal_crs.to_epsg() if horizontal_crs is not None else None
    horizontal_name = horizontal_crs.name if horizontal_crs is not None else None
    horizontal_crs_wkt = horizontal_crs.to_wkt() if horizontal_crs is not None else None

    source_compound: Optional[CRS] = None
    source_compound_error = None
    if parsed_crs is not None and parsed_crs.is_compound and pair.dep_horizontal_epsg_override is None:
        source_compound = parsed_crs
    elif horizontal_crs is not None and vertical_crs is not None:
        try:
            source_compound = CompoundCRS(
                name=f"{horizontal_name or 'horizontal'} + {vertical_name or f'EPSG:{vertical_epsg}'}",
                components=[horizontal_crs, vertical_crs],
            )
        except Exception as exc:
            source_compound_error = str(exc)
    else:
        source_compound_error = "missing_horizontal_or_vertical_component"

    if source_compound is None:
        source_compound_string = None
        source_compound_wkt = None
    else:
        if horizontal_epsg is not None and vertical_epsg is not None:
            source_compound_string = f"EPSG:{horizontal_epsg}+{vertical_epsg}"
        else:
            source_compound_string = source_compound.to_string()
        source_compound_wkt = source_compound.to_wkt()

    return {
        "las_header_parse_crs_success": parsed_crs is not None,
        "las_crs_is_compound": bool(parsed_crs.is_compound) if parsed_crs is not None else False,
        "las_crs_wkt": parsed_crs.to_wkt() if parsed_crs is not None else None,
        "horizontal_crs": horizontal_crs,
        "horizontal_crs_wkt": horizontal_crs_wkt,
        "horizontal_epsg": horizontal_epsg,
        "horizontal_name": horizontal_name,
        "vertical_crs": vertical_crs,
        "vertical_epsg": vertical_epsg,
        "vertical_name": vertical_name,
        "vertical_crs_source": vertical_crs_source,
        "vertical_crs_verified": vertical_crs_verified,
        "source_compound_crs": source_compound,
        "source_compound_crs_wkt": source_compound_wkt,
        "source_compound_crs_string": source_compound_string,
        "source_compound_crs_error": source_compound_error,
        "parse_error": parse_error,
    }


def read_dep_ground_points(dep_las_path: Path, pair: PairConfig, cfg: Config) -> Dict[str, Any]:
    las = laspy.read(dep_las_path)
    crs_audit = inspect_las_crs(las, pair)
    horizontal_crs = crs_audit["horizontal_crs"]
    if horizontal_crs is None:
        raise RuntimeError("3DEP horizontal CRS could not be resolved.")
    if crs_audit["source_compound_crs"] is None:
        raise RuntimeError(f"Could not construct source compound CRS: {crs_audit['source_compound_crs_error']}")

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    try:
        classification = np.asarray(las.classification)
    except Exception:
        classification = np.zeros(z.shape, dtype=np.uint8)

    cls_unique, cls_count = np.unique(classification, return_counts=True)
    classification_counts = {int(k): int(v) for k, v in zip(cls_unique, cls_count)}

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (classification == cfg.dep_ground_class_code)
    idx = np.flatnonzero(mask)
    n_ground_total = int(idx.size)
    if cfg.max_3dep_ground_points_for_tree and idx.size > cfg.max_3dep_ground_points_for_tree:
        idx = choose_indices(idx.size, cfg.max_3dep_ground_points_for_tree, cfg.random_seed)
        idx = np.flatnonzero(mask)[idx]

    return {
        "x": x[idx],
        "y": y[idx],
        "z": z[idx],
        "n_total": int(z.size),
        "n_ground_total": n_ground_total,
        "n_ground_used": int(idx.size),
        "classification_counts": classification_counts,
        "header_summary": {
            "version": str(las.header.version),
            "point_format_id": int(las.header.point_format.id),
            "point_count": int(las.header.point_count),
            "mins": [float(v) for v in las.header.mins],
            "maxs": [float(v) for v in las.header.maxs],
            "scales": [float(v) for v in las.header.scales],
            "offsets": [float(v) for v in las.header.offsets],
        },
        "crs_audit": crs_audit,
    }


def build_casals_points_in_dep_horizontal(raw: Dict[str, Any], pair: PairConfig, dep_horizontal_crs: CRS) -> Dict[str, Any]:
    transformer = Transformer.from_crs(CRS.from_user_input(pair.casals_crs), dep_horizontal_crs, always_xy=True)
    x, y = transformer.transform(raw["lon"], raw["lat"])
    return {
        "lon": np.asarray(raw["lon"], dtype=float),
        "lat": np.asarray(raw["lat"], dtype=float),
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "refh": np.asarray(raw["refh"], dtype=float),
        "refh_snr": np.asarray(raw["refh_snr"], dtype=float),
        "amp": np.asarray(raw["optional"].get("refh_amp", raw["optional"].get("amp", np.full_like(raw["refh"], np.nan))), dtype=float),
        "geoid": np.asarray(raw["optional"].get("geoid", np.full_like(raw["refh"], np.nan)), dtype=float),
        "record_index": np.asarray(raw["record_index"], dtype=int),
        "n_total": raw["n_total"],
    }


def query_neighbors(tree: cKDTree, query_xy: np.ndarray, k: int, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distances, indices = tree.query(query_xy, k=k, distance_upper_bound=radius, workers=-1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    valid = np.isfinite(distances) & (indices >= 0) & (indices < tree.n)
    counts = valid.sum(axis=1)
    nearest = np.full(query_xy.shape[0], np.nan, dtype=float)
    any_valid = valid.any(axis=1)
    nearest[any_valid] = np.nanmin(np.where(valid, distances, np.nan), axis=1)[any_valid]
    return distances, indices, valid, counts.astype(int), nearest


def idw_from_neighbors(
    source_z: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    idx_safe = np.where(valid, indices, 0)
    z_neighbors = source_z[idx_safe]
    weights = np.zeros_like(distances, dtype=float)
    exact = valid & (distances <= 1e-9)
    non_exact = valid & ~exact
    weights[non_exact] = 1.0 / np.power(distances[non_exact], IDW_POWER)
    has_exact = exact.any(axis=1)
    weights[exact] = 1.0
    weights[has_exact[:, None] & ~exact] = 0.0
    sampled = np.full(distances.shape[0], np.nan, dtype=float)
    denom = weights.sum(axis=1)
    good = denom > 0
    sampled[good] = np.sum(weights[good] * z_neighbors[good], axis=1) / denom[good]
    return sampled


def sample_surface_idw(
    query_x: np.ndarray,
    query_y: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_z: np.ndarray,
    k: int,
    radius: float,
    min_neighbors: int,
) -> Dict[str, np.ndarray]:
    if source_x.size == 0:
        raise RuntimeError("No source points available for surface sampling.")
    tree = cKDTree(np.column_stack([source_x, source_y]))
    qxy = np.column_stack([query_x, query_y])
    d, idx, valid, counts, nearest = query_neighbors(tree, qxy, k=k, radius=radius)
    sampled = idw_from_neighbors(source_z, d, idx, valid)
    ok = (counts >= min_neighbors) & np.isfinite(sampled)
    return {
        "z": sampled,
        "ok": ok,
        "n_neighbors": counts,
        "nearest_distance_m": nearest,
        "neighbor_distances": d,
        "neighbor_indices": idx,
        "neighbor_valid": valid,
    }


def map_by_record_index(base_df: pd.DataFrame, source_df: pd.DataFrame, value_columns: Sequence[str]) -> pd.DataFrame:
    out = base_df.copy()
    keyed = source_df.set_index("record_index")
    rec = out["record_index"]
    for column in value_columns:
        if column not in keyed.columns:
            continue
        out[column] = rec.map(keyed[column])
    return out


def fill_missing_bool(series: pd.Series, default: bool = False) -> pd.Series:
    values = series.to_numpy(dtype=object)
    out = np.empty(values.shape[0], dtype=bool)
    for i, value in enumerate(values):
        out[i] = bool(value) if pd.notna(value) else default
    return pd.Series(out, index=series.index, dtype=bool)


def build_grid_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell_size: float,
    quantile: float,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], float], float, float]:
    x0 = float(np.nanmin(x))
    y0 = float(np.nanmin(y))
    gx = np.floor((x - x0) / cell_size).astype(int)
    gy = np.floor((y - y0) / cell_size).astype(int)
    df = pd.DataFrame({"gx": gx, "gy": gy, "z": z})
    grouped = df.groupby(["gx", "gy"], sort=False)
    grid = grouped["z"].quantile(quantile).reset_index(name="z_low")
    grid["n_points"] = grouped.size().to_numpy(dtype=int)

    z_lookup = {(int(row.gx), int(row.gy)): float(row.z_low) for row in grid.itertuples(index=False)}
    smooth_values = []
    for row in grid.itertuples(index=False):
        values = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                val = z_lookup.get((int(row.gx) + dx, int(row.gy) + dy))
                if val is not None and np.isfinite(val):
                    values.append(val)
        smooth_values.append(float(np.mean(values)) if values else float(row.z_low))
    grid["z_low_smoothed"] = np.asarray(smooth_values, dtype=float)
    grid["center_x"] = x0 + (grid["gx"].to_numpy(dtype=float) + 0.5) * cell_size
    grid["center_y"] = y0 + (grid["gy"].to_numpy(dtype=float) + 0.5) * cell_size
    smooth_lookup = {(int(row.gx), int(row.gy)): float(row.z_low_smoothed) for row in grid.itertuples(index=False)}
    return grid, smooth_lookup, x0, y0


def sample_grid_lookup(
    x: np.ndarray,
    y: np.ndarray,
    cell_size: float,
    x0: float,
    y0: float,
    lookup: Dict[Tuple[int, int], float],
    fallback_grid: pd.DataFrame,
) -> np.ndarray:
    gx = np.floor((x - x0) / cell_size).astype(int)
    gy = np.floor((y - y0) / cell_size).astype(int)
    sampled = np.full(x.shape, np.nan, dtype=float)
    for i, key in enumerate(zip(gx, gy)):
        value = lookup.get(key)
        if value is not None:
            sampled[i] = value
    if np.isfinite(sampled).all() or fallback_grid.empty:
        return sampled

    centers = fallback_grid[["center_x", "center_y"]].to_numpy(dtype=float)
    tree = cKDTree(centers)
    missing = ~np.isfinite(sampled)
    d, idx = tree.query(np.column_stack([x[missing], y[missing]]), k=1, workers=-1)
    idx = np.atleast_1d(idx)
    sampled[missing] = fallback_grid["z_low_smoothed"].to_numpy(dtype=float)[idx]
    return sampled


def fit_local_plane_slope(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    if x.size < 3:
        return math.nan
    a = np.column_stack([x - np.mean(x), y - np.mean(y), np.ones_like(x)])
    try:
        coeff, _, _, _ = np.linalg.lstsq(a, z, rcond=None)
    except np.linalg.LinAlgError:
        return math.nan
    dzdx, dzdy = float(coeff[0]), float(coeff[1])
    slope_rad = math.atan(math.sqrt(dzdx ** 2 + dzdy ** 2))
    return float(np.degrees(slope_rad))


def compute_cell_counts(x: np.ndarray, y: np.ndarray, x0: float, y0: float, cell_size: float) -> pd.DataFrame:
    gx = np.floor((x - x0) / cell_size).astype(int)
    gy = np.floor((y - y0) / cell_size).astype(int)
    df = pd.DataFrame({"gx": gx, "gy": gy})
    out = df.groupby(["gx", "gy"], sort=False).size().reset_index(name="n_points")
    return out


def derive_casals_ground_points(casals_points: Dict[str, Any], cfg: Config, pair_dir: Path) -> Dict[str, Any]:
    out_dir = ensure_dir(pair_dir / "02_ground_extraction")

    lon = np.asarray(casals_points["lon"], dtype=float)
    lat = np.asarray(casals_points["lat"], dtype=float)
    x = np.asarray(casals_points["x"], dtype=float)
    y = np.asarray(casals_points["y"], dtype=float)
    z = np.asarray(casals_points["refh"], dtype=float)
    snr = np.asarray(casals_points["refh_snr"], dtype=float)
    amp = np.asarray(casals_points["amp"], dtype=float)
    record_index = np.asarray(casals_points["record_index"], dtype=int)

    finite = (
        np.isfinite(lon) & np.isfinite(lat) & np.isfinite(x) & np.isfinite(y) &
        np.isfinite(z) & np.isfinite(snr) &
        (lon >= -180) & (lon <= 180) & (lat >= -90) & (lat <= 90)
    )
    snr_candidate_mask = finite & (snr >= cfg.casals_snr_ground_candidate_threshold)
    candidate_idx_all = np.flatnonzero(snr_candidate_mask)
    n_snr_candidates_full = int(candidate_idx_all.size)
    sampled_idx = candidate_idx_all
    if cfg.max_casals_points_per_pair and candidate_idx_all.size > cfg.max_casals_points_per_pair:
        sampled_rel = choose_indices(candidate_idx_all.size, cfg.max_casals_points_per_pair, cfg.random_seed)
        sampled_idx = candidate_idx_all[sampled_rel]

    cand = pd.DataFrame({
        "record_index": record_index[sampled_idx],
        "lon": lon[sampled_idx],
        "lat": lat[sampled_idx],
        "x_dep_horizontal": x[sampled_idx],
        "y_dep_horizontal": y[sampled_idx],
        "refh_wgs84_ellipsoidal_m": z[sampled_idx],
        "refh_snr": snr[sampled_idx],
        "amp": amp[sampled_idx],
    })
    cand["isolated_noise_flag"] = False
    cand["local_z_outlier_flag"] = False
    cand["ground_seed_flag"] = False
    cand["secondary_ground_outlier_flag"] = False
    cand["final_ground_flag"] = False

    if cand.empty:
        raise RuntimeError("No CASALS SNR candidates after finite screening.")

    tree = cKDTree(cand[["x_dep_horizontal", "y_dep_horizontal"]].to_numpy(dtype=float))
    k_for_neighbor = max(2, cfg.casals_min_neighbors_count + 1)
    d, _, valid, counts, _ = query_neighbors(
        tree,
        cand[["x_dep_horizontal", "y_dep_horizontal"]].to_numpy(dtype=float),
        k=k_for_neighbor,
        radius=cfg.casals_min_neighbors_radius_m,
    )
    _ = d
    neighbor_count_excluding_self = np.maximum(counts - 1, 0)
    isolated = neighbor_count_excluding_self < cfg.casals_min_neighbors_count
    cand.loc[isolated, "isolated_noise_flag"] = True

    kept_after_neighbor = cand.loc[~cand["isolated_noise_flag"]].copy()
    if kept_after_neighbor.empty:
        raise RuntimeError("CASALS candidate set collapsed after neighbor filter.")

    x0_local = float(kept_after_neighbor["x_dep_horizontal"].min())
    y0_local = float(kept_after_neighbor["y_dep_horizontal"].min())
    gx = np.floor((kept_after_neighbor["x_dep_horizontal"].to_numpy(dtype=float) - x0_local) / cfg.casals_local_outlier_cell_m).astype(int)
    gy = np.floor((kept_after_neighbor["y_dep_horizontal"].to_numpy(dtype=float) - y0_local) / cfg.casals_local_outlier_cell_m).astype(int)
    kept_after_neighbor["gx_local"] = gx
    kept_after_neighbor["gy_local"] = gy

    local_groups = kept_after_neighbor.groupby(["gx_local", "gy_local"], sort=False)["refh_wgs84_ellipsoidal_m"]
    local_median = local_groups.transform("median")
    local_nmad = local_groups.transform(lambda s: nmad(s.to_numpy(dtype=float), cfg.robust_nmad_factor))
    local_tol = np.maximum(
        cfg.casals_local_outlier_min_abs_m,
        cfg.casals_local_outlier_nmad_multiplier * local_nmad.to_numpy(dtype=float),
    )
    local_outlier = np.abs(kept_after_neighbor["refh_wgs84_ellipsoidal_m"].to_numpy(dtype=float) - local_median.to_numpy(dtype=float)) > local_tol
    kept_after_neighbor["local_z_outlier_flag"] = local_outlier
    cand = map_by_record_index(cand, kept_after_neighbor[["record_index", "local_z_outlier_flag"]], ["local_z_outlier_flag"])
    cand["local_z_outlier_flag"] = fill_missing_bool(cand["local_z_outlier_flag"], default=False)

    filtered = cand.loc[~cand["isolated_noise_flag"] & ~cand["local_z_outlier_flag"]].copy()
    if filtered.empty:
        raise RuntimeError("CASALS candidate set collapsed after local Z outlier filter.")

    low_grid, low_lookup, low_x0, low_y0 = build_grid_surface(
        filtered["x_dep_horizontal"].to_numpy(dtype=float),
        filtered["y_dep_horizontal"].to_numpy(dtype=float),
        filtered["refh_wgs84_ellipsoidal_m"].to_numpy(dtype=float),
        cfg.casals_ground_grid_m,
        cfg.casals_low_quantile,
    )
    low_grid.to_csv(out_dir / "casals_prelim_low_surface.csv", index=False)
    prelim_low = sample_grid_lookup(
        cand["x_dep_horizontal"].to_numpy(dtype=float),
        cand["y_dep_horizontal"].to_numpy(dtype=float),
        cfg.casals_ground_grid_m,
        low_x0,
        low_y0,
        low_lookup,
        low_grid,
    )
    cand["prelim_low_surface_z_m"] = prelim_low
    cand["resid_to_prelim_low_surface_m"] = cand["refh_wgs84_ellipsoidal_m"] - cand["prelim_low_surface_z_m"]

    seed_mask = (
        ~cand["isolated_noise_flag"] &
        ~cand["local_z_outlier_flag"] &
        np.isfinite(cand["resid_to_prelim_low_surface_m"]) &
        (cand["resid_to_prelim_low_surface_m"] >= cfg.casals_ground_seed_resid_low_m) &
        (cand["resid_to_prelim_low_surface_m"] <= cfg.casals_ground_seed_resid_high_m)
    )
    cand.loc[seed_mask, "ground_seed_flag"] = True

    seeds = cand.loc[cand["ground_seed_flag"]].copy()
    if seeds.empty:
        raise RuntimeError("No CASALS ground seeds survived low-envelope screening.")

    seed_xy = seeds[["x_dep_horizontal", "y_dep_horizontal"]].to_numpy(dtype=float)
    seed_tree = cKDTree(seed_xy)
    k_loocv = max(2, cfg.casals_dtm_idw_k + 1)
    d_seed, idx_seed, valid_seed, counts_seed, _ = query_neighbors(
        seed_tree,
        seed_xy,
        k=k_loocv,
        radius=cfg.casals_dtm_idw_radius_m,
    )
    non_self = valid_seed & (d_seed > 1e-9)
    z_seed = seeds["refh_wgs84_ellipsoidal_m"].to_numpy(dtype=float)
    sampled_seed_dtm = idw_from_neighbors(z_seed, d_seed, idx_seed, non_self)
    ok_seed_dtm = (non_self.sum(axis=1) >= cfg.casals_dtm_idw_min_neighbors) & np.isfinite(sampled_seed_dtm)
    seed_residual = z_seed - sampled_seed_dtm
    secondary_keep = np.zeros(z_seed.shape, dtype=bool)
    if ok_seed_dtm.any():
        sec_mask = robust_inlier_mask(seed_residual[ok_seed_dtm], cfg, nmad_multiplier=3.0, min_abs_m=0.5)
        secondary_keep[ok_seed_dtm] = sec_mask
    else:
        secondary_keep[:] = True
    seeds["seed_to_dtm_residual_m"] = seed_residual
    seeds["secondary_ground_outlier_flag"] = ~secondary_keep
    final_ground = seeds.loc[~seeds["secondary_ground_outlier_flag"]].copy()
    if final_ground.empty:
        raise RuntimeError("No final CASALS ground points survived secondary robust filtering.")

    cand = map_by_record_index(
        cand,
        seeds[["record_index", "seed_to_dtm_residual_m", "secondary_ground_outlier_flag"]],
        ["seed_to_dtm_residual_m", "secondary_ground_outlier_flag"],
    )
    cand["secondary_ground_outlier_flag"] = fill_missing_bool(cand["secondary_ground_outlier_flag"], default=False)
    cand.loc[cand["record_index"].isin(final_ground["record_index"]), "final_ground_flag"] = True
    cand.to_csv(out_dir / "casals_ground_candidates.csv", index=False)

    final_ground = final_ground.assign(
        casals_ground_z_target=final_ground["refh_wgs84_ellipsoidal_m"],
    )
    final_ground.to_csv(out_dir / "casals_ground_points.csv", index=False)

    pdal_smrf_summary = {
        "status": "disabled",
        "precision_like_overlap": math.nan,
        "recall_like_overlap": math.nan,
        "disagreement_count": None,
    }
    if cfg.enable_pdal_smrf_casals_ground_check:
        pdal_exe = resolve_executable(cfg.pdal_exe, "pdal")
        pdal_smrf_summary = {"status": "skipped_pdal_not_found" if not pdal_exe else "not_implemented"}

    summary = {
        "n_total_records": int(casals_points["n_total"]),
        "n_finite_records": int(finite.sum()),
        "n_snr_candidates": n_snr_candidates_full,
        "n_snr_candidates_used": int(len(cand)),
        "n_after_neighbor_filter": int((~cand["isolated_noise_flag"]).sum()),
        "n_after_local_outlier_filter": int((~cand["isolated_noise_flag"] & ~cand["local_z_outlier_flag"]).sum()),
        "n_ground_seeds": int(cand["ground_seed_flag"].sum()),
        "n_final_ground_points": int(final_ground.shape[0]),
        "final_ground_fraction_of_used_candidates": float(final_ground.shape[0] / len(cand)),
        "candidate_z_range_m": [
            float(np.nanmin(cand["refh_wgs84_ellipsoidal_m"])),
            float(np.nanmax(cand["refh_wgs84_ellipsoidal_m"])),
        ],
        "ground_z_range_m": [
            float(np.nanmin(final_ground["refh_wgs84_ellipsoidal_m"])),
            float(np.nanmax(final_ground["refh_wgs84_ellipsoidal_m"])),
        ],
        "pdal_smrf_cross_check": pdal_smrf_summary,
    }
    write_json(out_dir / "casals_ground_extraction_summary.json", summary)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=cfg.dpi)
    stage_counts = [
        len(cand),
        int((~cand["isolated_noise_flag"]).sum()),
        int((~cand["isolated_noise_flag"] & ~cand["local_z_outlier_flag"]).sum()),
        int(cand["ground_seed_flag"].sum()),
        int(final_ground.shape[0]),
    ]
    axes[0, 0].bar(np.arange(len(stage_counts)), stage_counts)
    axes[0, 0].set_xticks(np.arange(len(stage_counts)))
    axes[0, 0].set_xticklabels(["SNR", "neighbor", "local-Z", "seed", "final"], rotation=20)
    axes[0, 0].set_title("CASALS ground extraction counts")

    axes[0, 1].hist(cand["resid_to_prelim_low_surface_m"].dropna().to_numpy(dtype=float), bins=100, color="0.35")
    axes[0, 1].axvline(cfg.casals_ground_seed_resid_low_m, color="r", linestyle="--")
    axes[0, 1].axvline(cfg.casals_ground_seed_resid_high_m, color="r", linestyle="--")
    axes[0, 1].set_title("Residual to preliminary low surface")
    axes[0, 1].set_xlabel("m")

    axes[1, 0].scatter(
        cand["x_dep_horizontal"],
        cand["y_dep_horizontal"],
        c=cand["final_ground_flag"].astype(int),
        s=2,
        cmap="viridis",
    )
    axes[1, 0].set_title("Final CASALS ground mask")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")

    axes[1, 1].hist(final_ground["refh_wgs84_ellipsoidal_m"].to_numpy(dtype=float), bins=100, color="0.25")
    axes[1, 1].set_title("Final CASALS ground heights")
    axes[1, 1].set_xlabel("m")

    fig.tight_layout()
    fig.savefig(out_dir / "casals_ground_extraction_diagnostics.png")
    plt.close(fig)

    return {
        "summary": summary,
        "candidate_df": cand,
        "ground_df": final_ground,
        "prelim_low_surface": low_grid,
    }


def build_transformer_group_info(source_crs: CRS, target_crs: CRS) -> Dict[str, Any]:
    if TransformerGroup is None:
        return {
            "available": False,
            "best_available": None,
            "selected_transformer_description": None,
            "selected_transformer_accuracy": None,
            "missing_grids": [],
            "transformer": None,
        }
    try:
        group = TransformerGroup(source_crs, target_crs, always_xy=True)
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "best_available": None,
            "selected_transformer_description": None,
            "selected_transformer_accuracy": None,
            "missing_grids": [],
            "transformer": None,
        }

    missing_grids: List[Dict[str, Any]] = []
    for op in getattr(group, "unavailable_operations", []) or []:
        for grid in getattr(op, "grids", []) or []:
            if getattr(grid, "available", None) is False:
                missing_grids.append({
                    "short_name": getattr(grid, "short_name", None),
                    "url": getattr(grid, "url", None),
                    "package_name": getattr(grid, "package_name", None),
                })

    transformer = group.transformers[0] if group.transformers else None
    return {
        "available": True,
        "best_available": bool(getattr(group, "best_available", False)),
        "selected_transformer_description": getattr(transformer, "description", None),
        "selected_transformer_accuracy": getattr(transformer, "accuracy", None),
        "missing_grids": missing_grids,
        "transformer": transformer,
    }


def pipeline_has_explicit_vertical_step(pipeline: str) -> bool:
    text = pipeline.lower()
    vertical_tokens = [
        "vgridshift",
        "geoidgrids",
        "vertcon",
        "xyzgridshift",
        "vertoffset",
        "proj=cart",
        "proj=helmert",
        "push",
        "pop",
    ]
    return any(token in text for token in vertical_tokens)


def normalize_proj_pipeline(pipeline: str) -> Optional[str]:
    if not isinstance(pipeline, str):
        return None
    text = " ".join(pipeline.strip().split())
    if not text:
        return None
    if text.startswith("+proj=pipeline"):
        return text
    if "proj=pipeline" not in text.lower():
        return None
    tokens = text.split()
    normalized: List[str] = []
    for token in tokens:
        lower = token.lower()
        if token.startswith("+"):
            normalized.append(token)
        elif lower == "step":
            normalized.append("+step")
        elif "=" in token or lower == "proj=pipeline":
            normalized.append(f"+{token}")
        else:
            normalized.append(token)
    normalized_text = " ".join(normalized)
    return normalized_text if normalized_text.startswith("+proj=pipeline") else None


def extract_executable_proj_pipeline(transformer: Optional[Transformer]) -> Optional[str]:
    if transformer is None:
        return None
    candidate_values: List[Optional[str]] = []
    try:
        candidate_values.append(transformer.to_proj4())
    except Exception:
        candidate_values.append(None)
    candidate_values.append(getattr(transformer, "definition", None))
    try:
        last_used = transformer.get_last_used_operation()
    except Exception:
        last_used = None
    if last_used is not None:
        try:
            candidate_values.append(last_used.to_proj4())
        except Exception:
            candidate_values.append(None)
        candidate_values.append(getattr(last_used, "definition", None))

    for candidate in candidate_values:
        if not isinstance(candidate, str):
            continue
        normalized = normalize_proj_pipeline(candidate)
        if normalized and pipeline_has_explicit_vertical_step(normalized):
            return normalized
    return None


def infer_pipeline_input_type(pipeline: str) -> str:
    text = " ".join((pipeline or "").lower().split())
    if not text.startswith("+proj=pipeline"):
        return "unknown"
    if text.startswith("+proj=pipeline +step +inv +proj=utm"):
        return "projected"
    if text.startswith("+proj=pipeline +step +inv +proj=tmerc"):
        return "projected"
    if text.startswith("+proj=pipeline +step +inv +proj=lcc"):
        return "projected"
    if text.startswith("+proj=pipeline +step +proj=axisswap +order=2,1 +step +proj=unitconvert +xy_in=deg +xy_out=rad"):
        return "geographic_degree"
    if text.startswith("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad"):
        return "geographic_degree"
    if text.startswith("+proj=pipeline +step +proj=unitconvert +xy_in=rad +xy_out=deg"):
        return "geographic_radian"
    if text.startswith("+proj=pipeline +step +proj=cart"):
        return "geographic_radian"
    return "unknown"


def prepare_input_for_pipeline(
    x_projected: np.ndarray,
    y_projected: np.ndarray,
    z: np.ndarray,
    horizontal_crs: CRS,
    pipeline_input_type: str,
) -> pd.DataFrame:
    if pipeline_input_type == "projected":
        return pd.DataFrame({"x_in": x_projected, "y_in": y_projected, "z_in": z})
    if pipeline_input_type in {"geographic_degree", "geographic_radian"}:
        geodetic_crs = horizontal_crs.geodetic_crs or CRS.from_epsg(6318)
        transformer = Transformer.from_crs(horizontal_crs, geodetic_crs, always_xy=True)
        lon, lat = transformer.transform(x_projected, y_projected)
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if pipeline_input_type == "geographic_radian":
            lon = np.deg2rad(lon)
            lat = np.deg2rad(lat)
        return pd.DataFrame({"x_in": lon, "y_in": lat, "z_in": z})
    raise ValueError(f"Unsupported or ambiguous pipeline input type: {pipeline_input_type}")


def validate_proj_pipeline_with_cct_or_projinfo(
    pipeline: str,
    sample_input: str,
    cfg: Config,
) -> Tuple[bool, str]:
    cct_reason = None
    cct_exe = resolve_executable(cfg.cct_exe, "cct")
    if cct_exe:
        command = [cct_exe] + shlex.split(pipeline, posix=False)
        try:
            return_code, stdout_text, stderr_text = run_external_transform_command(command, sample_input)
            xyz_out = parse_xyz_stdout(stdout_text)
            if return_code == 0 and xyz_out is not None and xyz_out.shape[1] >= 3:
                return True, "validated_with_cct"
            cct_reason = f"cct_failed: {stderr_text.splitlines()[:2]}"
        except Exception as exc:
            cct_reason = f"cct_exception: {exc}"
    projinfo_exe = resolve_executable(cfg.projinfo_exe, "projinfo")
    if projinfo_exe:
        try:
            proc = subprocess.run([projinfo_exe, pipeline], capture_output=True, text=True, timeout=60, check=False, env=get_proj_subprocess_env())
            if proc.returncode == 0:
                return True, "validated_with_projinfo" if cct_reason is None else f"validated_with_projinfo_after_{cct_reason}"
            return False, f"projinfo_failed: {(proc.stderr or proc.stdout).splitlines()[:2]}"
        except Exception as exc:
            return False, f"projinfo_exception: {exc}"
    return False, cct_reason or "no_cct_or_projinfo_available"


def extract_proj_pipelines_from_projinfo_text(text: str) -> List[str]:
    pipelines: List[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "PROJ string:":
            block: List[str] = []
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if not stripped:
                    break
                if stripped.startswith("WKT2:"):
                    break
                block.append(stripped)
                i += 1
            pipeline = normalize_proj_pipeline(" ".join(block))
            if pipeline:
                pipelines.append(pipeline)
        i += 1
    return pipelines


def run_projinfo_audit(source_crs: CRS, target_crs: CRS, out_dir: Path, cfg: Config) -> Dict[str, Any]:
    exe = resolve_executable(cfg.projinfo_exe, "projinfo")
    if not exe:
        return {
            "status": "skipped_executable_not_found",
            "projinfo_mentions_vertical_grid": False,
            "projinfo_mentions_navd88": False,
            "projinfo_mentions_wgs84_or_itrf": False,
            "operation_names": [],
            "pipelines": [],
        }
    command = [exe, "-s", source_crs.to_wkt(), "-t", target_crs.to_string(), "--spatial-test", "intersects"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False, env=get_proj_subprocess_env())
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "projinfo_mentions_vertical_grid": False,
            "projinfo_mentions_navd88": False,
            "projinfo_mentions_wgs84_or_itrf": False,
            "operation_names": [],
            "pipelines": [],
        }
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    (out_dir / "projinfo_full.txt").write_text(text, encoding="utf-8")
    operation_names = re.findall(r"Operation No\.\s+\d+:\s*\n\s*\n?([^\n]+)", text)
    pipelines = extract_proj_pipelines_from_projinfo_text(text)
    mentions_vertical = any(token in text.lower() for token in ["vgridshift", "geoid", "vertical"])
    mentions_navd88 = "navd88" in text.lower()
    mentions_wgs84_or_itrf = ("wgs 84" in text.lower()) or ("itrf" in text.lower())
    summary_lines = [
        f"status: {'ok' if proc.returncode == 0 else 'failed'}",
        f"n_operations: {len(operation_names)}",
        f"n_pipelines: {len(pipelines)}",
        f"mentions_vertical_grid: {mentions_vertical}",
        f"mentions_navd88: {mentions_navd88}",
        f"mentions_wgs84_or_itrf: {mentions_wgs84_or_itrf}",
    ]
    (out_dir / "projinfo_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "command": subprocess.list2cmdline(command),
        "projinfo_mentions_vertical_grid": mentions_vertical,
        "projinfo_mentions_navd88": mentions_navd88,
        "projinfo_mentions_wgs84_or_itrf": mentions_wgs84_or_itrf,
        "operation_names": operation_names[:10],
        "pipelines": pipelines,
        "stdout_preview": "\n".join(text.splitlines()[:30]),
    }


def select_validated_pipeline(
    candidate_pipelines: Sequence[str],
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_z: np.ndarray,
    horizontal_crs: CRS,
    cfg: Config,
) -> Dict[str, Any]:
    def pipeline_score(pipeline_text: str) -> int:
        text = pipeline_text.lower()
        score = 0
        for token, weight in [
            ("proj=helmert", 4),
            ("proj=cart", 3),
            ("push", 2),
            ("pop", 2),
            ("vgridshift", 2),
            ("gridshift", 1),
        ]:
            if token in text:
                score += weight
        return score

    best_candidate: Optional[Dict[str, Any]] = None
    for pipeline in candidate_pipelines:
        input_type = infer_pipeline_input_type(pipeline)
        if input_type in {"unknown", "geocentric"}:
            continue
        try:
            prepared = prepare_input_for_pipeline(sample_x[:1], sample_y[:1], sample_z[:1], horizontal_crs, input_type)
        except Exception:
            continue
        sample_input = format_xyz_input_text(
            (row.x_in, row.y_in, row.z_in) for row in prepared.itertuples(index=False)
        )
        valid, reason = validate_proj_pipeline_with_cct_or_projinfo(pipeline, sample_input, cfg)
        if valid:
            candidate = {
                "pipeline": pipeline,
                "pipeline_validated": True,
                "pipeline_validation_reason": reason,
                "input_coordinate_type": input_type,
                "pipeline_score": pipeline_score(pipeline),
            }
            if best_candidate is None or candidate["pipeline_score"] > best_candidate["pipeline_score"]:
                best_candidate = candidate
    if best_candidate is not None:
        return best_candidate
    return {
        "pipeline": None,
        "pipeline_validated": False,
        "pipeline_validation_reason": "no_pipeline_validated",
        "input_coordinate_type": None,
    }


def parse_ngs_geoid_height(response: Dict[str, Any]) -> Optional[float]:
    candidate_keys = [
        "geoidHeight",
        "geoid_height",
        "GeoidHeight",
        "height",
        "geoid",
        "geoidHt",
        "geoid_ht",
    ]
    for key in candidate_keys:
        if key in response:
            try:
                return float(response[key])
            except Exception:
                continue

    stack: List[Any] = [response]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                key_lower = str(key).lower()
                if "geoid" in key_lower and ("height" in key_lower or "ht" in key_lower):
                    try:
                        return float(value)
                    except Exception:
                        pass
                if isinstance(value, (dict, list, tuple)):
                    stack.append(value)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return None


def query_ngs_geoid_variants(lat: float, lon: float, model: int) -> List[Dict[str, Any]]:
    variant_param_sets: List[Dict[str, Any]] = [
        {"lat": f"{lat:.10f}", "lon": f"{lon:.10f}", "model": model},
        {"lat": f"{lat:.10f}", "lon": f"{lon:.10f}", "model": model, "units": "m"},
    ]
    if model == 14:
        variant_param_sets.append({"lat": f"{lat:.10f}", "lon": f"{lon:.10f}"})

    results: List[Dict[str, Any]] = []
    for params in variant_param_sets[:3]:
        url = f"https://geodesy.noaa.gov/api/geoid/ght?{urllib.parse.urlencode(params)}"
        raw_response: Any = None
        error = None
        try:
            with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310
                raw_text = response.read().decode("utf-8")
            raw_response = json.loads(raw_text)
        except Exception as exc:
            error = str(exc)
        results.append({
            "url": url,
            "raw_response": raw_response,
            "error": error,
            "geoid_height_m": parse_ngs_geoid_height(raw_response) if isinstance(raw_response, dict) else None,
            "status": "ok" if isinstance(raw_response, dict) and parse_ngs_geoid_height(raw_response) is not None else "failed",
        })
        time.sleep(NGS_API_PAUSE_S)
        if results[-1]["status"] == "ok":
            break
    return results


def ngs_geoid_check(casals_points: Dict[str, Any], cfg: Config, out_dir: Path) -> Dict[str, Any]:
    if not cfg.enable_ngs_geoid_check:
        return {"enabled": False, "status": "disabled", "summary_rows": [], "response_rows": []}
    lon = np.asarray(casals_points["lon"], dtype=float)
    lat = np.asarray(casals_points["lat"], dtype=float)
    finite = np.isfinite(lon) & np.isfinite(lat)
    if not finite.any():
        return {"enabled": True, "status": "failed", "summary_rows": [], "response_rows": [], "error": "no_finite_lon_lat"}

    lon_f = lon[finite]
    lat_f = lat[finite]
    sample_points = [
        ("center", float(np.nanmedian(lat_f)), float(np.nanmedian(lon_f))),
        ("corner_sw", float(np.nanmin(lat_f)), float(np.nanmin(lon_f))),
        ("corner_se", float(np.nanmin(lat_f)), float(np.nanmax(lon_f))),
        ("corner_nw", float(np.nanmax(lat_f)), float(np.nanmin(lon_f))),
        ("corner_ne", float(np.nanmax(lat_f)), float(np.nanmax(lon_f))),
    ]

    response_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for model in NGS_GEOID_MODELS:
        values: List[float] = []
        for location_name, lat_i, lon_i in sample_points:
            variant_results = query_ngs_geoid_variants(lat_i, lon_i, model)
            chosen = variant_results[-1] if variant_results else {"status": "failed", "geoid_height_m": None, "url": None, "raw_response": None, "error": "no_variants"}
            for variant in variant_results:
                response_rows.append({
                    "ngs_model": model,
                    "location": location_name,
                    "lat": lat_i,
                    "lon": lon_i,
                    "status": variant["status"],
                    "geoid_height_m": variant["geoid_height_m"],
                    "raw_response": variant["raw_response"],
                    "url": variant["url"],
                    "error": variant["error"],
                })
            if chosen.get("geoid_height_m") is not None:
                values.append(float(chosen["geoid_height_m"]))
        if values:
            stats = robust_stats(np.asarray(values, dtype=float), cfg.robust_nmad_factor)
            summary_rows.append({
                "ngs_model": model,
                "n_samples": len(values),
                "ngs_geoid_median_m": stats["median"],
                "ngs_geoid_nmad_m": stats["nmad"],
                "diagnostic_only_not_full_frame_transform": True,
            })

    write_json(out_dir / "ngs_geoid_api_responses.json", response_rows)
    status = "ok" if summary_rows else "failed"
    preferred = next((row for row in summary_rows if row["ngs_model"] == 14), summary_rows[0] if summary_rows else None)
    return {
        "enabled": True,
        "status": status,
        "summary_rows": summary_rows,
        "response_rows": response_rows,
        "preferred": preferred,
    }


def agreement_tolerance_for_method(method: str, cfg: Config) -> float:
    if method == "two_stage_ngs_geoid_plus_pyproj_frame":
        return cfg.two_stage_shift_agreement_tol_m
    return cfg.transform_shift_agreement_tol_m


def compare_with_pyproj(
    method: str,
    result: TransformResult,
    reference: TransformResult,
    cfg: Config,
) -> Dict[str, Any]:
    comparison = {
        "agree": None,
        "median_difference_m": math.nan,
        "nmad_difference_m": math.nan,
        "p05_difference_m": math.nan,
        "p95_difference_m": math.nan,
        "p95_abs_difference_m": math.nan,
        "max_abs_difference_m": math.nan,
        "failed_zero_vertical_shift": False,
    }
    if result.z_shift is None or reference.z_shift is None or len(result.z_shift) != len(reference.z_shift):
        return comparison
    diff = np.asarray(result.z_shift, dtype=float) - np.asarray(reference.z_shift, dtype=float)
    stats = robust_stats(diff, factor=cfg.robust_nmad_factor)
    tol = agreement_tolerance_for_method(method, cfg)
    failed_zero = is_near_zero_vertical_shift(result.z_shift, cfg) and not is_near_zero_vertical_shift(reference.z_shift, cfg)
    p95_abs = float(np.nanpercentile(np.abs(diff), 95)) if np.isfinite(diff).any() else math.nan
    agree = (
        abs(stats["median"]) <= tol and
        stats["nmad"] <= tol and
        p95_abs <= max(tol, 0.3 if method == "two_stage_ngs_geoid_plus_pyproj_frame" else tol)
    )
    comparison.update({
        "agree": bool(agree) and not failed_zero,
        "median_difference_m": stats["median"],
        "nmad_difference_m": stats["nmad"],
        "p05_difference_m": stats["p05"],
        "p95_difference_m": stats["p95"],
        "p95_abs_difference_m": p95_abs,
        "max_abs_difference_m": stats["max_abs"],
        "failed_zero_vertical_shift": failed_zero,
    })
    result.median_diff_to_pyproj_m = stats["median"]
    result.nmad_diff_to_pyproj_m = stats["nmad"]
    result.p95_abs_diff_to_pyproj_m = p95_abs
    result.failed_zero_vertical_shift = failed_zero
    if failed_zero:
        result.status = "failed_zero_vertical_shift"
    elif method == "two_stage_ngs_geoid_plus_pyproj_frame":
        pass
    elif result.status == "ok" and not comparison["agree"]:
        result.status = "failed_incomplete_vertical_operation"
    return comparison


def transform_with_pyproj(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    source_crs: CRS,
    target_crs: CRS,
    cfg: Config,
) -> Tuple[TransformResult, Dict[str, Any]]:
    group_info = build_transformer_group_info(source_crs, target_crs)
    transformer = group_info.get("transformer")
    if transformer is None:
        try:
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        except Exception as exc:
            result = TransformResult(
                method="pyproj",
                source_crs=source_crs.to_string(),
                target_crs=target_crs.to_string(),
                status="failed",
                z_transformed=None,
                z_shift=None,
                operation_description=group_info.get("selected_transformer_description"),
                operation_accuracy=group_info.get("selected_transformer_accuracy"),
                best_available=group_info.get("best_available"),
                missing_grids=group_info.get("missing_grids", []),
                command=None,
                software={"pyproj_version": pyproj.__version__, "error": str(exc)},
                notes="pyproj failed to construct a transformer.",
            )
            return result, {"transformer": None, "pipeline": None, "input_coordinate_type": None, "pipeline_validated": False}

    try:
        _, _, z_out = transformer.transform(x, y, z)
        z_out = np.asarray(z_out, dtype=float)
        z_shift = z_out - np.asarray(z, dtype=float)
    except Exception as exc:
        result = TransformResult(
            method="pyproj",
            source_crs=source_crs.to_string(),
            target_crs=target_crs.to_string(),
            status="failed",
            z_transformed=None,
            z_shift=None,
            operation_description=getattr(transformer, "description", None),
            operation_accuracy=getattr(transformer, "accuracy", None),
            best_available=group_info.get("best_available"),
            missing_grids=group_info.get("missing_grids", []),
            command=None,
            software={"pyproj_version": pyproj.__version__, "error": str(exc)},
            notes="pyproj transform execution failed.",
        )
        pipeline = extract_executable_proj_pipeline(transformer)
        return result, {
            "transformer": transformer,
            "pipeline": pipeline,
            "input_coordinate_type": infer_pipeline_input_type(pipeline) if pipeline else None,
            "pipeline_validated": False,
        }

    status = "ok"
    if is_near_zero_vertical_shift(z_shift, cfg):
        status = "failed_zero_vertical_shift"
    elif group_info.get("best_available") is False:
        status = "degraded_missing_grid"

    result = TransformResult(
        method="pyproj",
        source_crs=source_crs.to_string(),
        target_crs=target_crs.to_string(),
        status=status,
        z_transformed=z_out,
        z_shift=z_shift,
        operation_description=getattr(transformer, "description", None),
        operation_accuracy=getattr(transformer, "accuracy", None),
        best_available=group_info.get("best_available"),
        missing_grids=group_info.get("missing_grids", []),
        command=None,
        software={
            "pyproj_version": pyproj.__version__,
            "proj_version": pyproj.proj_version_str,
            "transformer_definition": getattr(transformer, "definition", None),
            "transformer_proj4": getattr(transformer, "to_proj4", lambda: None)(),
        },
        notes="pyproj/PROJ reference transform.",
        input_coordinate_type="projected",
        pipeline_available=False,
        pipeline_validated=False,
    )
    pipeline = extract_executable_proj_pipeline(transformer)
    result.pipeline_available = bool(pipeline)
    result.input_coordinate_type = infer_pipeline_input_type(pipeline) if pipeline else "projected"
    return result, {
        "transformer": transformer,
        "pipeline": pipeline,
        "input_coordinate_type": result.input_coordinate_type,
        "pipeline_validated": False,
    }


def transform_with_gdal_osr(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    source_crs: CRS,
    target_crs: CRS,
    cfg: Config,
) -> TransformResult:
    if not cfg.enable_gdal_osr_check:
        return TransformResult(
            method="gdal_osr",
            source_crs=source_crs.to_string(),
            target_crs=target_crs.to_string(),
            status="skipped_disabled",
            z_transformed=None,
            z_shift=None,
            operation_description=None,
            operation_accuracy=None,
            best_available=None,
            missing_grids=[],
            command=None,
            software={"available": False},
            notes="GDAL OSR check disabled in config.",
            input_coordinate_type="projected",
        )
    if osr is None:
        return TransformResult(
            method="gdal_osr",
            source_crs=source_crs.to_string(),
            target_crs=target_crs.to_string(),
            status="skipped_missing_bindings",
            z_transformed=None,
            z_shift=None,
            operation_description=None,
            operation_accuracy=None,
            best_available=None,
            missing_grids=[],
            command=None,
            software={"available": False},
            notes="osgeo.osr bindings are not available.",
            input_coordinate_type="projected",
        )

    try:
        src = osr.SpatialReference()
        src.ImportFromWkt(source_crs.to_wkt())
        dst = osr.SpatialReference()
        dst.ImportFromWkt(target_crs.to_wkt())
        if hasattr(src, "SetAxisMappingStrategy"):
            src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if hasattr(dst, "SetAxisMappingStrategy"):
            dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(src, dst)
        pts = np.column_stack([x, y, z]).tolist()
        out = np.asarray(transform.TransformPoints(pts), dtype=float)
        z_out = out[:, 2]
        z_shift = z_out - np.asarray(z, dtype=float)
    except Exception as exc:
        return TransformResult(
            method="gdal_osr",
            source_crs=source_crs.to_string(),
            target_crs=target_crs.to_string(),
            status="failed",
            z_transformed=None,
            z_shift=None,
            operation_description=None,
            operation_accuracy=None,
            best_available=None,
            missing_grids=[],
            command=None,
            software={"gdal_version": getattr(gdal, "__version__", None), "error": str(exc)},
            notes="GDAL OSR transformation failed.",
            input_coordinate_type="projected",
        )

    failed_zero = is_near_zero_vertical_shift(z_shift, cfg)
    status = "failed_zero_vertical_shift" if failed_zero else "ok"
    return TransformResult(
        method="gdal_osr",
        source_crs=source_crs.to_string(),
        target_crs=target_crs.to_string(),
        status=status,
        z_transformed=z_out,
        z_shift=z_shift,
        operation_description="GDAL OSR CoordinateTransformation",
        operation_accuracy=None,
        best_available=None,
        missing_grids=[],
        command=None,
        software={"gdal_version": getattr(gdal, "__version__", None), "operation_queryable": False},
        notes="Independent GDAL Python bindings path.",
        input_coordinate_type="projected",
        failed_zero_vertical_shift=failed_zero,
    )


def run_external_transform_command(command: List[str], input_text: str) -> Tuple[int, str, str]:
    proc = subprocess.run(
        command,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=get_proj_subprocess_env(),
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def parse_xyz_stdout(stdout_text: str) -> Optional[np.ndarray]:
    rows = []
    for line in stdout_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except Exception:
            continue
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def format_xyz_input_text(rows: Iterable[Tuple[float, float, float]]) -> str:
    lines = [f"{x:.10f} {y:.10f} {z:.10f}" for x, y, z in rows]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def transform_with_gdaltransform_cli(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    source_crs: str,
    target_crs: str,
    cfg: Config,
    out_dir: Path,
    mode: str,
    proj_pipeline: Optional[str] = None,
    horizontal_crs: Optional[CRS] = None,
    input_coordinate_type: Optional[str] = None,
) -> TransformResult:
    method = "gdaltransform_ct" if mode == "ct" else "gdaltransform_noct"
    if not cfg.enable_gdaltransform_check:
        return TransformResult(
            method=method,
            source_crs=source_crs,
            target_crs=target_crs,
            status="skipped_disabled",
            z_transformed=None,
            z_shift=None,
            operation_description=None,
            operation_accuracy=None,
            best_available=None,
            missing_grids=[],
            command=None,
            software={},
            notes="gdaltransform CLI check disabled in config.",
            input_coordinate_type="projected",
        )
    exe = resolve_executable(cfg.gdaltransform_exe, "gdaltransform")
    if not exe:
        return TransformResult(
            method=method,
            source_crs=source_crs,
            target_crs=target_crs,
            status="skipped_executable_not_found",
            z_transformed=None,
            z_shift=None,
            operation_description=None,
            operation_accuracy=None,
            best_available=None,
            missing_grids=[],
            command=None,
            software={},
            notes="gdaltransform executable not found.",
            input_coordinate_type="projected",
        )
    if mode == "ct" and not proj_pipeline:
        return TransformResult(
            method=method,
            source_crs=source_crs,
            target_crs=target_crs,
            status="skipped_missing_executable_pipeline",
            z_transformed=None,
            z_shift=None,
            operation_description=None,
            operation_accuracy=None,
            best_available=None,
            missing_grids=[],
            command=None,
            software={},
            notes="No explicit PROJ pipeline available for gdaltransform -ct.",
            input_coordinate_type=None,
            pipeline_available=False,
            pipeline_validated=False,
        )
    if mode == "ct" and input_coordinate_type in {"unknown", "geocentric"}:
        return TransformResult(
            method=method,
            source_crs=source_crs,
            target_crs=target_crs,
            status="skipped_ambiguous_pipeline_input_type",
            z_transformed=None,
            z_shift=None,
            operation_description=proj_pipeline,
            operation_accuracy=None,
            best_available=None,
            missing_grids=[],
            command=None,
            software={},
            notes="Pipeline input type is ambiguous for gdaltransform -ct.",
            input_coordinate_type=input_coordinate_type,
            pipeline_available=bool(proj_pipeline),
            pipeline_validated=False,
        )

    input_path = out_dir / f"{method}_input_xyz.txt"
    output_path = out_dir / f"{method}_output_xyz.csv"
    if mode == "ct" and input_coordinate_type and input_coordinate_type != "projected":
        if horizontal_crs is None:
            return TransformResult(
                method=method,
                source_crs=source_crs,
                target_crs=target_crs,
                status="skipped_ambiguous_pipeline_input_type",
                z_transformed=None,
                z_shift=None,
                operation_description=proj_pipeline,
                operation_accuracy=None,
                best_available=None,
                missing_grids=[],
                command=None,
                software={},
                notes="Horizontal CRS missing for non-projected gdaltransform -ct input preparation.",
                input_coordinate_type=input_coordinate_type,
                pipeline_available=True,
                pipeline_validated=False,
            )
        prepared = prepare_input_for_pipeline(x, y, z, horizontal_crs, input_coordinate_type)
        input_text = format_xyz_input_text(
            (row.x_in, row.y_in, row.z_in) for row in prepared.itertuples(index=False)
        )
    else:
        input_text = format_xyz_input_text(zip(x, y, z))
    input_path.write_text(input_text, encoding="utf-8")

    command = [exe, "-s_srs", source_crs, "-t_srs", target_crs]
    if mode == "ct":
        command.extend(["-ct", proj_pipeline or ""])
    return_code, stdout_text, stderr_text = run_external_transform_command(command, input_text)
    xyz_out = parse_xyz_stdout(stdout_text)
    if return_code != 0 or xyz_out is None or xyz_out.shape[0] != len(x):
        return TransformResult(
            method=method,
            source_crs=source_crs,
            target_crs=target_crs,
            status="failed",
            z_transformed=None,
            z_shift=None,
            operation_description=None,
            operation_accuracy=None,
            best_available=None,
            missing_grids=[],
            command=subprocess.list2cmdline(command),
            software={
                "return_code": return_code,
                "stdout_preview": "\n".join(stdout_text.splitlines()[:8]),
                "stderr_preview": "\n".join(stderr_text.splitlines()[:8]),
                "input_xyz_txt": str(input_path),
            },
            notes="gdaltransform CLI execution failed or produced unparseable output.",
            input_coordinate_type=input_coordinate_type or "projected",
            pipeline_available=(mode == "ct"),
            pipeline_validated=(mode == "ct" and bool(proj_pipeline)),
        )

    out_df = pd.DataFrame(xyz_out, columns=["x_out", "y_out", "z_out"])
    out_df.to_csv(output_path, index=False)
    z_out = out_df["z_out"].to_numpy(dtype=float)
    z_shift = z_out - np.asarray(z, dtype=float)
    failed_zero = is_near_zero_vertical_shift(z_shift, cfg)
    status = "failed_zero_vertical_shift" if failed_zero else "ok"
    return TransformResult(
        method=method,
        source_crs=source_crs,
        target_crs=target_crs,
        status=status,
        z_transformed=z_out,
        z_shift=z_shift,
        operation_description="gdaltransform -ct pipeline" if mode == "ct" else "gdaltransform inferred operation",
        operation_accuracy=None,
        best_available=None,
        missing_grids=[],
        command=subprocess.list2cmdline(command),
        software={
            "return_code": return_code,
            "stdout_preview": "\n".join(stdout_text.splitlines()[:8]),
            "stderr_preview": "\n".join(stderr_text.splitlines()[:8]),
            "input_xyz_txt": str(input_path),
            "output_xyz_csv": str(output_path),
        },
        notes="GDAL gdaltransform command-line cross-check.",
        input_coordinate_type=input_coordinate_type or "projected",
        failed_zero_vertical_shift=failed_zero,
        pipeline_available=(mode == "ct"),
        pipeline_validated=(mode == "ct" and bool(proj_pipeline)),
    )


def run_cct_transform(
    method: str,
    command_args: List[str],
    input_df: pd.DataFrame,
    out_dir: Path,
    cfg: Config,
    operation_description: str,
    notes: str,
    input_coordinate_type: str,
) -> TransformResult:
    input_path = out_dir / f"{method}_input_xyz.txt"
    output_path = out_dir / f"{method}_output_xyz.csv"
    input_text = format_xyz_input_text(
        (row.x_in, row.y_in, row.z_in) for row in input_df.itertuples(index=False)
    )
    input_path.write_text(input_text, encoding="utf-8")
    return_code, stdout_text, stderr_text = run_external_transform_command(command_args, input_text)
    xyz_out = parse_xyz_stdout(stdout_text)
    if return_code != 0 or xyz_out is None or xyz_out.shape[0] != len(input_df):
        return TransformResult(
            method=method,
            source_crs="pipeline_input",
            target_crs=cfg.main_target_crs,
            status="failed",
            z_transformed=None,
            z_shift=None,
            operation_description=operation_description,
            operation_accuracy=None,
            best_available=None,
            missing_grids=[],
            command=subprocess.list2cmdline(command_args),
            software={
                "return_code": return_code,
                "stdout_preview": "\n".join(stdout_text.splitlines()[:8]),
                "stderr_preview": "\n".join(stderr_text.splitlines()[:8]),
                "input_xyz_txt": str(input_path),
            },
            notes=notes,
            input_coordinate_type=input_coordinate_type,
        )
    out_df = pd.DataFrame(xyz_out, columns=["x_out", "y_out", "z_out"])
    out_df.to_csv(output_path, index=False)
    z_out = out_df["z_out"].to_numpy(dtype=float)
    z_shift = z_out - input_df["z_in"].to_numpy(dtype=float)
    failed_zero = is_near_zero_vertical_shift(z_shift, cfg)
    return TransformResult(
        method=method,
        source_crs="pipeline_input",
        target_crs=cfg.main_target_crs,
        status="failed_zero_vertical_shift" if failed_zero else "ok",
        z_transformed=z_out,
        z_shift=z_shift,
        operation_description=operation_description,
        operation_accuracy=None,
        best_available=None,
        missing_grids=[],
        command=subprocess.list2cmdline(command_args),
        software={
            "return_code": return_code,
            "stdout_preview": "\n".join(stdout_text.splitlines()[:8]),
            "stderr_preview": "\n".join(stderr_text.splitlines()[:8]),
            "input_xyz_txt": str(input_path),
            "output_xyz_csv": str(output_path),
        },
        notes=notes,
        input_coordinate_type=input_coordinate_type,
        failed_zero_vertical_shift=failed_zero,
    )


def is_referenceable_cct_operation_name(operation_name: Optional[str]) -> bool:
    if not operation_name or not isinstance(operation_name, str):
        return False
    text = operation_name.strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith("unknown id,"):
        return False
    if lowered.startswith("inverse of "):
        return False
    return True


def transform_with_proj_cct_pipeline(
    x_projected: np.ndarray,
    y_projected: np.ndarray,
    z: np.ndarray,
    source_horizontal_crs: CRS,
    cfg: Config,
    out_dir: Path,
    proj_pipeline: Optional[str],
    pipeline_validated: bool,
) -> TransformResult:
    method = "proj_cct_pipeline"
    if not cfg.enable_proj_cct_check:
        return TransformResult(method, source_horizontal_crs.to_string(), cfg.main_target_crs, "skipped_disabled", None, None, None, None, None, [], None, {}, "PROJ cct check disabled in config.")
    exe = resolve_executable(cfg.cct_exe, "cct")
    if not exe:
        return TransformResult(method, source_horizontal_crs.to_string(), cfg.main_target_crs, "skipped_executable_not_found", None, None, None, None, None, [], None, {}, "PROJ cct executable not found.")
    if not proj_pipeline:
        return TransformResult(method, source_horizontal_crs.to_string(), cfg.main_target_crs, "skipped_missing_executable_pipeline", None, None, None, None, None, [], None, {}, "No validated explicit pipeline available.", pipeline_available=False, pipeline_validated=False)
    input_coordinate_type = infer_pipeline_input_type(proj_pipeline)
    if input_coordinate_type == "unknown":
        return TransformResult(method, source_horizontal_crs.to_string(), cfg.main_target_crs, "skipped_ambiguous_pipeline_input_type", None, None, proj_pipeline, None, None, [], None, {}, "Pipeline input coordinate type could not be inferred safely.", input_coordinate_type=input_coordinate_type, pipeline_available=True, pipeline_validated=pipeline_validated)
    if input_coordinate_type == "geocentric":
        return TransformResult(method, source_horizontal_crs.to_string(), cfg.main_target_crs, "skipped_ambiguous_pipeline_input_type", None, None, proj_pipeline, None, None, [], None, {}, "Geocentric pipeline input is not auto-handled.", input_coordinate_type=input_coordinate_type, pipeline_available=True, pipeline_validated=pipeline_validated)
    try:
        input_df = prepare_input_for_pipeline(x_projected, y_projected, z, source_horizontal_crs, input_coordinate_type)
    except Exception as exc:
        return TransformResult(method, source_horizontal_crs.to_string(), cfg.main_target_crs, "failed", None, None, proj_pipeline, None, None, [], None, {"error": str(exc)}, "Could not prepare cct pipeline input.", input_coordinate_type=input_coordinate_type, pipeline_available=True, pipeline_validated=pipeline_validated)
    command_args = [exe] + shlex.split(proj_pipeline, posix=False)
    result = run_cct_transform(
        method,
        command_args,
        input_df,
        out_dir,
        cfg,
        proj_pipeline,
        "Explicit PROJ cct pipeline cross-check.",
        input_coordinate_type,
    )
    result.pipeline_available = True
    result.pipeline_validated = pipeline_validated
    return result


def transform_with_proj_cct_operation(
    x_projected: np.ndarray,
    y_projected: np.ndarray,
    z: np.ndarray,
    source_horizontal_crs: CRS,
    cfg: Config,
    out_dir: Path,
    operation_name: Optional[str],
) -> TransformResult:
    method = "proj_cct_operation"
    if not cfg.enable_proj_cct_check:
        return TransformResult(method, source_horizontal_crs.to_string(), cfg.main_target_crs, "skipped_disabled", None, None, None, None, None, [], None, {}, "PROJ cct operation check disabled in config.")
    exe = resolve_executable(cfg.cct_exe, "cct")
    if not exe:
        return TransformResult(method, source_horizontal_crs.to_string(), cfg.main_target_crs, "skipped_executable_not_found", None, None, None, None, None, [], None, {}, "PROJ cct executable not found.")
    if not operation_name:
        return TransformResult(method, source_horizontal_crs.to_string(), cfg.main_target_crs, "skipped_missing_operation_name", None, None, None, None, None, [], None, {}, "No operation name available for cct operation check.")
    if not is_referenceable_cct_operation_name(operation_name):
        return TransformResult(
            method,
            source_horizontal_crs.to_string(),
            cfg.main_target_crs,
            "skipped_unreferenceable_operation_name",
            None,
            None,
            operation_name,
            None,
            None,
            [],
            None,
            {"candidate_operation_name": operation_name},
            "projinfo/pyproj only provided a descriptive operation title, not a cct-referenceable operation identifier.",
            input_coordinate_type="geographic_degree",
        )
    geodetic_crs = source_horizontal_crs.geodetic_crs or CRS.from_epsg(6318)
    transformer = Transformer.from_crs(source_horizontal_crs, geodetic_crs, always_xy=True)
    lon, lat = transformer.transform(x_projected, y_projected)
    input_df = pd.DataFrame({"x_in": lon, "y_in": lat, "z_in": z})
    command_args = [exe, operation_name]
    return run_cct_transform(
        method,
        command_args,
        input_df,
        out_dir,
        cfg,
        operation_name,
        "Operation-name-based PROJ cct optional cross-check.",
        "geographic_degree",
    )


def two_stage_ngs_geoid_plus_pyproj_frame(
    x_projected: np.ndarray,
    y_projected: np.ndarray,
    z_navd88: np.ndarray,
    horizontal_crs: CRS,
    target_crs: CRS,
    ngs_check: Dict[str, Any],
    cfg: Config,
) -> TransformResult:
    method = "two_stage_ngs_geoid_plus_pyproj_frame"
    preferred = ngs_check.get("preferred")
    if not preferred or preferred.get("ngs_geoid_median_m") is None:
        return TransformResult(method, "EPSG:6319", target_crs.to_string(), "skipped_missing_ngs_geoid", None, None, None, None, None, [], None, {}, "NGS geoid median unavailable for two-stage check.", ngs_geoid_status=ngs_check.get("status"))
    geoid_height = float(preferred["ngs_geoid_median_m"])
    geodetic_crs = horizontal_crs.geodetic_crs or CRS.from_epsg(6318)
    xy_transformer = Transformer.from_crs(horizontal_crs, geodetic_crs, always_xy=True)
    lon, lat = xy_transformer.transform(x_projected, y_projected)
    z_nad83_ellip = np.asarray(z_navd88, dtype=float) + geoid_height
    try:
        frame_transformer = Transformer.from_crs(CRS.from_epsg(6319), target_crs, always_xy=True)
        _, _, z_out = frame_transformer.transform(lon, lat, z_nad83_ellip)
        z_out = np.asarray(z_out, dtype=float)
        z_shift = z_out - np.asarray(z_navd88, dtype=float)
    except Exception as exc:
        return TransformResult(method, "EPSG:6319", target_crs.to_string(), "failed", None, None, None, None, None, [], None, {"error": str(exc)}, "Two-stage NGS geoid plus pyproj frame transform failed.", input_coordinate_type="geographic_degree", ngs_geoid_status=ngs_check.get("status"))
    failed_zero = is_near_zero_vertical_shift(z_shift, cfg)
    return TransformResult(
        method=method,
        source_crs="EPSG:6319",
        target_crs=target_crs.to_string(),
        status="failed_zero_vertical_shift" if failed_zero else "ok",
        z_transformed=z_out,
        z_shift=z_shift,
        operation_description="Approximate two-stage NAVD88->NAD83 ellipsoidal via NGS geoid median, then pyproj 3D frame transform.",
        operation_accuracy=None,
        best_available=None,
        missing_grids=[],
        command=None,
        software={"ngs_geoid_median_m": geoid_height},
        notes="Approximate independent sanity check only, not the main transform.",
        input_coordinate_type="geographic_degree",
        failed_zero_vertical_shift=failed_zero,
        ngs_geoid_status=ngs_check.get("status"),
    )

def build_transform_cross_validation_rows(
    pair_label: str,
    transform_results: Dict[str, TransformResult],
    agreement_summary: Dict[str, Any],
    cfg: Config,
) -> pd.DataFrame:
    rows = []
    pyproj_method = agreement_summary.get("reference_method", "pyproj")
    for method, result in transform_results.items():
        shift_stats = robust_stats(result.z_shift, factor=cfg.robust_nmad_factor) if result.z_shift is not None else robust_stats(np.array([]), factor=cfg.robust_nmad_factor)
        compare = agreement_summary.get("per_method", {}).get(method, {})
        rows.append({
            "pair_label": pair_label,
            "method": method,
            "target_crs": result.target_crs,
            "status": result.status,
            "median_z_shift_m": shift_stats["median"],
            "nmad_z_shift_m": shift_stats["nmad"],
            "p05_z_shift_m": shift_stats["p05"],
            "p95_z_shift_m": shift_stats["p95"],
            "best_available": result.best_available,
            "missing_grid_count": len(result.missing_grids),
            "operation_accuracy": result.operation_accuracy,
            "operation_description": result.operation_description,
            "agrees_with_pyproj": True if method == pyproj_method else compare.get("agree"),
            "input_coordinate_type": result.input_coordinate_type,
            "z_shift_expected_nonzero": result.z_shift_expected_nonzero,
            "failed_zero_vertical_shift": result.failed_zero_vertical_shift,
            "median_diff_to_pyproj_m": result.median_diff_to_pyproj_m,
            "nmad_diff_to_pyproj_m": result.nmad_diff_to_pyproj_m,
            "p95_abs_diff_to_pyproj_m": result.p95_abs_diff_to_pyproj_m,
            "pipeline_available": result.pipeline_available,
            "pipeline_validated": result.pipeline_validated,
            "projinfo_status": result.projinfo_status,
            "ngs_geoid_status": result.ngs_geoid_status,
            "notes": result.notes,
        })
    return pd.DataFrame(rows)


def evaluate_transform_agreement(transform_results: Dict[str, TransformResult], cfg: Config) -> Dict[str, Any]:
    reference = transform_results.get("pyproj")
    if reference is None or reference.z_shift is None:
        return {
            "reference_method": "pyproj",
            "transform_agreement_status": "transform_failed",
            "single_transform_path_only": False,
            "n_transform_methods_ok": 0,
            "n_transform_methods_agree_with_pyproj": 0,
            "per_method": {},
        }

    per_method: Dict[str, Any] = {}
    n_ok = 0
    n_agree = 0
    external_method_names = {
        "gdal_osr",
        "gdaltransform_noct",
        "gdaltransform_ct",
        "proj_cct_pipeline",
        "proj_cct_operation",
    }
    two_stage_agree = False
    external_agree = False
    for method, result in transform_results.items():
        method_ok = result.status in {"ok", "degraded_missing_grid"}
        if method_ok:
            n_ok += 1
        if method == "pyproj":
            continue
        comparison = compare_with_pyproj(method, result, reference, cfg)
        if comparison["agree"]:
            n_agree += 1
            if method == "two_stage_ngs_geoid_plus_pyproj_frame":
                two_stage_agree = True
            if method in external_method_names:
                external_agree = True
        per_method[method] = comparison

    single_transform_path_only = n_ok == 1 and reference.status in {"ok", "degraded_missing_grid"}
    nonzero_disagreement = any(
        value.get("agree") is False and not value.get("failed_zero_vertical_shift", False)
        for value in per_method.values()
    )
    if reference.status not in {"ok", "degraded_missing_grid"}:
        status = "transform_failed"
    elif external_agree:
        status = "multi_method_agree"
    elif two_stage_agree:
        status = "pyproj_plus_two_stage_agree"
    elif nonzero_disagreement:
        status = "transform_disagreement"
    elif single_transform_path_only:
        status = "single_transform_path_only"
    else:
        status = "single_transform_path_only"

    return {
        "reference_method": "pyproj",
        "transform_agreement_status": status,
        "single_transform_path_only": single_transform_path_only,
        "n_transform_methods_ok": n_ok,
        "n_transform_methods_agree_with_pyproj": n_agree,
        "gdal_osr_agree": per_method.get("gdal_osr", {}).get("agree"),
        "gdaltransform_noct_agree": per_method.get("gdaltransform_noct", {}).get("agree"),
        "gdaltransform_ct_agree": per_method.get("gdaltransform_ct", {}).get("agree"),
        "proj_cct_pipeline_agree": per_method.get("proj_cct_pipeline", {}).get("agree"),
        "proj_cct_operation_agree": per_method.get("proj_cct_operation", {}).get("agree"),
        "two_stage_agree": per_method.get("two_stage_ngs_geoid_plus_pyproj_frame", {}).get("agree"),
        "per_method": per_method,
    }


def make_transform_histogram(transform_results: Dict[str, TransformResult], out_path: Path, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=cfg.dpi)
    plotted = False
    for method, result in transform_results.items():
        if result.z_shift is None:
            continue
        arr = np.asarray(result.z_shift, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        ax.hist(arr, bins=100, alpha=0.35, label=method)
        plotted = True
    if plotted:
        ax.legend(fontsize=8)
    ax.set_xlabel("Vertical shift (m)")
    ax.set_ylabel("count")
    ax.set_title("Coordinate transform vertical-shift cross-validation")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_3dep_ground_surface(dep_ground_points: Dict[str, Any], transform_result: TransformResult, cfg: Config, pair_dir: Path) -> Dict[str, Any]:
    out_dir = ensure_dir(pair_dir / "02_ground_extraction")
    if transform_result.z_transformed is None:
        raise RuntimeError("Main transform did not produce transformed 3DEP ground heights.")

    x = np.asarray(dep_ground_points["x"], dtype=float)
    y = np.asarray(dep_ground_points["y"], dtype=float)
    z_raw = np.asarray(dep_ground_points["z"], dtype=float)
    z_target = np.asarray(transform_result.z_transformed, dtype=float)
    if z_target.shape != z_raw.shape:
        raise RuntimeError("Transformed 3DEP ground height array size mismatch.")

    sample_idx = choose_indices(len(x), min(cfg.save_debug_point_limit, len(x)), cfg.random_seed)
    sample_df = pd.DataFrame({
        "x_dep_horizontal": x[sample_idx],
        "y_dep_horizontal": y[sample_idx],
        "dep_ground_z_raw_navd88_m": z_raw[sample_idx],
        "dep_ground_z_target_m": z_target[sample_idx],
        "dep_ground_z_shift_to_target_m": z_target[sample_idx] - z_raw[sample_idx],
    })
    sample_df.to_csv(out_dir / "dep_ground_transformed_points_sample.csv", index=False)

    summary = {
        "n_3dep_total": int(dep_ground_points["n_total"]),
        "n_3dep_ground": int(dep_ground_points["n_ground_used"]),
        "transform_main_method": transform_result.method,
        "transform_main_status": transform_result.status,
        "transform_main_target_crs": transform_result.target_crs,
        "transform_main_shift_stats": robust_stats(z_target - z_raw, factor=cfg.robust_nmad_factor),
    }
    write_json(out_dir / "dep_ground_surface_summary.json", summary)

    return {
        "x": x,
        "y": y,
        "z_raw": z_raw,
        "z_target": z_target,
        "summary": summary,
    }


def make_residual_summary_rows(stats_map: Dict[str, Dict[str, Any]], cfg: Config) -> pd.DataFrame:
    rows = []
    for residual_name, stats in stats_map.items():
        row = {"residual_name": residual_name}
        row.update(stats_to_prefixed_row("residual", stats))
        clipped = clipped_stats(np.asarray(stats.get("_values", []), dtype=float), cfg) if "_values" in stats else None
        if clipped is not None:
            row.update({
                "robust_clip_n": clipped.get("n"),
                "robust_clip_median_m": clipped.get("median"),
                "robust_clip_nmad_m": clipped.get("nmad"),
                "robust_clip_rmse_m": clipped.get("rmse"),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def compute_ground_to_ground_residuals(
    casals_ground: Dict[str, Any],
    dep_ground_surface: Dict[str, Any],
    casals_ground_surface: Dict[str, Any],
    ngs_check: Dict[str, Any],
    cfg: Config,
    pair_dir: Path,
) -> Dict[str, Any]:
    out_dir = ensure_dir(pair_dir / "03_residuals")
    ground_df = casals_ground["ground_df"].copy()
    candidate_df = casals_ground["candidate_df"].copy()

    dep_pointwise = sample_surface_idw(
        ground_df["x_dep_horizontal"].to_numpy(dtype=float),
        ground_df["y_dep_horizontal"].to_numpy(dtype=float),
        dep_ground_surface["x"],
        dep_ground_surface["y"],
        dep_ground_surface["z_target"],
        k=cfg.dep_dtm_idw_k,
        radius=cfg.dep_dtm_idw_radius_m,
        min_neighbors=cfg.dep_dtm_idw_min_neighbors,
    )
    ok_point = dep_pointwise["ok"]
    if not ok_point.any():
        raise RuntimeError("No pointwise ground-to-ground residuals could be computed.")

    slope = np.full(len(ground_df), np.nan, dtype=float)
    dep_x = dep_ground_surface["x"]
    dep_y = dep_ground_surface["y"]
    dep_z_target = dep_ground_surface["z_target"]
    idx_neighbors = dep_pointwise["neighbor_indices"]
    valid_neighbors = dep_pointwise["neighbor_valid"]
    for i in range(len(ground_df)):
        valid_i = valid_neighbors[i]
        idx_i = idx_neighbors[i][valid_i]
        if idx_i.size < 3:
            continue
        slope[i] = fit_local_plane_slope(dep_x[idx_i], dep_y[idx_i], dep_z_target[idx_i])

    point_df = ground_df.copy()
    point_df["casals_ground_z_target"] = point_df["refh_wgs84_ellipsoidal_m"]
    point_df["dep_ground_z_target_at_casals_xy"] = dep_pointwise["z"]
    point_df["residual_casals_minus_3dep"] = point_df["casals_ground_z_target"] - point_df["dep_ground_z_target_at_casals_xy"]
    point_df["nearest_3dep_ground_distance_m"] = dep_pointwise["nearest_distance_m"]
    point_df["n_3dep_neighbors"] = dep_pointwise["n_neighbors"]
    point_df["local_slope_deg"] = slope
    point_df["slope_bin"] = pd.cut(
        point_df["local_slope_deg"],
        bins=[-np.inf, 5.0, 15.0, np.inf],
        labels=["low", "medium", "high"],
    )
    point_df = point_df.loc[ok_point].copy()
    point_df.to_csv(out_dir / "ground_to_ground_point_residuals.csv", index=False)

    raw_dep_sample = sample_surface_idw(
        candidate_df["x_dep_horizontal"].to_numpy(dtype=float),
        candidate_df["y_dep_horizontal"].to_numpy(dtype=float),
        dep_ground_surface["x"],
        dep_ground_surface["y"],
        dep_ground_surface["z_raw"],
        k=cfg.dep_dtm_idw_k,
        radius=cfg.dep_dtm_idw_radius_m,
        min_neighbors=cfg.dep_dtm_idw_min_neighbors,
    )
    transformed_dep_sample = sample_surface_idw(
        candidate_df["x_dep_horizontal"].to_numpy(dtype=float),
        candidate_df["y_dep_horizontal"].to_numpy(dtype=float),
        dep_ground_surface["x"],
        dep_ground_surface["y"],
        dep_ground_surface["z_target"],
        k=cfg.dep_dtm_idw_k,
        radius=cfg.dep_dtm_idw_radius_m,
        min_neighbors=cfg.dep_dtm_idw_min_neighbors,
    )
    sanity_df = candidate_df.copy()
    sanity_df["dep_ground_raw_navd88_at_casals_xy_m"] = raw_dep_sample["z"]
    sanity_df["dep_ground_target_at_casals_xy_m"] = transformed_dep_sample["z"]
    sanity_df["raw_height_system_offset_m"] = sanity_df["refh_wgs84_ellipsoidal_m"] - sanity_df["dep_ground_raw_navd88_at_casals_xy_m"]
    sanity_df["transformed_high_snr_sanity_residual_m"] = sanity_df["refh_wgs84_ellipsoidal_m"] - sanity_df["dep_ground_target_at_casals_xy_m"]
    geoid_median = math.nan
    preferred_ngs = ngs_check.get("preferred")
    if preferred_ngs and preferred_ngs.get("ngs_geoid_median_m") is not None:
        geoid_median = float(preferred_ngs["ngs_geoid_median_m"])
    sanity_df["geoid_only_residual_m"] = sanity_df["raw_height_system_offset_m"] - geoid_median if np.isfinite(geoid_median) else np.nan
    sanity_df["diagnostic_only_not_full_frame_transform"] = True
    sanity_df.to_csv(out_dir / "height_system_sanity_check.csv", index=False)

    x0 = float(min(np.nanmin(point_df["x_dep_horizontal"]), np.nanmin(dep_ground_surface["x"])))
    y0 = float(min(np.nanmin(point_df["y_dep_horizontal"]), np.nanmin(dep_ground_surface["y"])))
    cas_counts = compute_cell_counts(
        point_df["x_dep_horizontal"].to_numpy(dtype=float),
        point_df["y_dep_horizontal"].to_numpy(dtype=float),
        x0,
        y0,
        cfg.residual_grid_m,
    ).rename(columns={"n_points": "n_casals_points"})
    dep_counts = compute_cell_counts(
        dep_ground_surface["x"],
        dep_ground_surface["y"],
        x0,
        y0,
        cfg.residual_grid_m,
    ).rename(columns={"n_points": "n_3dep_points"})

    common_cells = cas_counts.merge(dep_counts, on=["gx", "gy"], how="inner")
    common_cells = common_cells.loc[
        (common_cells["n_casals_points"] >= cfg.min_points_per_grid_cell) &
        (common_cells["n_3dep_points"] >= cfg.min_points_per_grid_cell)
    ].copy()
    common_cells["center_x"] = x0 + (common_cells["gx"].to_numpy(dtype=float) + 0.5) * cfg.residual_grid_m
    common_cells["center_y"] = y0 + (common_cells["gy"].to_numpy(dtype=float) + 0.5) * cfg.residual_grid_m

    if common_cells.empty:
        raise RuntimeError("No common grid cells met the minimum point requirement.")

    cas_grid_sample = sample_surface_idw(
        common_cells["center_x"].to_numpy(dtype=float),
        common_cells["center_y"].to_numpy(dtype=float),
        point_df["x_dep_horizontal"].to_numpy(dtype=float),
        point_df["y_dep_horizontal"].to_numpy(dtype=float),
        point_df["casals_ground_z_target"].to_numpy(dtype=float),
        k=cfg.casals_dtm_idw_k,
        radius=cfg.casals_dtm_idw_radius_m,
        min_neighbors=cfg.casals_dtm_idw_min_neighbors,
    )
    dep_grid_sample = sample_surface_idw(
        common_cells["center_x"].to_numpy(dtype=float),
        common_cells["center_y"].to_numpy(dtype=float),
        dep_ground_surface["x"],
        dep_ground_surface["y"],
        dep_ground_surface["z_target"],
        k=cfg.dep_dtm_idw_k,
        radius=cfg.dep_dtm_idw_radius_m,
        min_neighbors=cfg.dep_dtm_idw_min_neighbors,
    )
    ok_grid = cas_grid_sample["ok"] & dep_grid_sample["ok"]
    grid_df = common_cells.loc[ok_grid].copy()
    grid_df["casals_ground_dtm_target_m"] = cas_grid_sample["z"][ok_grid]
    grid_df["dep_ground_dtm_target_m"] = dep_grid_sample["z"][ok_grid]
    grid_df["residual_casals_minus_3dep"] = grid_df["casals_ground_dtm_target_m"] - grid_df["dep_ground_dtm_target_m"]
    grid_df.to_csv(out_dir / "ground_to_ground_grid_residuals.csv", index=False)

    dep_grid_export = grid_df[["gx", "gy", "center_x", "center_y", "n_3dep_points", "dep_ground_dtm_target_m"]].copy()
    dep_grid_export.to_csv(pair_dir / "02_ground_extraction" / "dep_ground_dtm_grid.csv", index=False)

    point_values = point_df["residual_casals_minus_3dep"].to_numpy(dtype=float)
    grid_values = grid_df["residual_casals_minus_3dep"].to_numpy(dtype=float)
    raw_values = sanity_df["raw_height_system_offset_m"].to_numpy(dtype=float)
    geoid_values = sanity_df["geoid_only_residual_m"].to_numpy(dtype=float)
    transformed_sanity_values = sanity_df["transformed_high_snr_sanity_residual_m"].to_numpy(dtype=float)

    stats_map = {
        "raw_height_system_offset": {**robust_stats(raw_values, cfg.robust_nmad_factor), "_values": raw_values},
        "geoid_only_residual": {**robust_stats(geoid_values, cfg.robust_nmad_factor), "_values": geoid_values},
        "transformed_high_snr_sanity_residual": {**robust_stats(transformed_sanity_values, cfg.robust_nmad_factor), "_values": transformed_sanity_values},
        "point_ground_to_ground_all": {**robust_stats(point_values, cfg.robust_nmad_factor), "_values": point_values},
        "grid_ground_to_ground_all": {**robust_stats(grid_values, cfg.robust_nmad_factor), "_values": grid_values},
    }

    for slope_name in ["low", "medium", "high"]:
        subset = point_df.loc[point_df["slope_bin"].astype(str) == slope_name, "residual_casals_minus_3dep"].to_numpy(dtype=float)
        stats_map[f"point_ground_to_ground_{slope_name}_slope"] = {**robust_stats(subset, cfg.robust_nmad_factor), "_values": subset}

    summary_rows = make_residual_summary_rows(stats_map, cfg)
    summary_rows.to_csv(out_dir / "ground_to_ground_offset_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=cfg.dpi)
    ax.hist(point_values[np.isfinite(point_values)], bins=120, color="0.35")
    ax.set_xlabel("Residual (m)")
    ax.set_ylabel("count")
    ax.set_title("Pointwise ground-to-ground residual")
    fig.tight_layout()
    fig.savefig(out_dir / "residual_histogram_pointwise.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=cfg.dpi)
    ax.hist(grid_values[np.isfinite(grid_values)], bins=120, color="0.35")
    ax.set_xlabel("Residual (m)")
    ax.set_ylabel("count")
    ax.set_title("Grid-to-grid ground residual")
    fig.tight_layout()
    fig.savefig(out_dir / "residual_histogram_grid.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=cfg.dpi)
    axes[0].scatter(point_df["local_slope_deg"], point_df["residual_casals_minus_3dep"], s=5, alpha=0.35)
    axes[0].set_xlabel("Local slope (deg)")
    axes[0].set_ylabel("Residual (m)")
    axes[0].set_title("Residual vs local slope")
    for slope_name, color in [("low", "#1b9e77"), ("medium", "#d95f02"), ("high", "#7570b3")]:
        vals = point_df.loc[point_df["slope_bin"].astype(str) == slope_name, "residual_casals_minus_3dep"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            axes[1].hist(vals, bins=60, alpha=0.4, label=slope_name, color=color)
    axes[1].set_xlabel("Residual (m)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Residual histogram by slope bin")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_vs_slope.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=cfg.dpi)
    ax.scatter(point_df["nearest_3dep_ground_distance_m"], point_df["residual_casals_minus_3dep"], s=5, alpha=0.35)
    ax.set_xlabel("Nearest 3DEP ground distance (m)")
    ax.set_ylabel("Residual (m)")
    ax.set_title("Residual vs nearest 3DEP ground distance")
    fig.tight_layout()
    fig.savefig(out_dir / "residual_vs_neighbor_distance.png")
    plt.close(fig)

    return {
        "point_df": point_df,
        "grid_df": grid_df,
        "sanity_df": sanity_df,
        "summary_rows": summary_rows,
        "raw_stats": stats_map["raw_height_system_offset"],
        "geoid_stats": stats_map["geoid_only_residual"],
        "transformed_high_snr_stats": stats_map["transformed_high_snr_sanity_residual"],
        "point_stats": stats_map["point_ground_to_ground_all"],
        "grid_stats": stats_map["grid_ground_to_ground_all"],
        "low_slope_stats": stats_map["point_ground_to_ground_low_slope"],
        "medium_slope_stats": stats_map["point_ground_to_ground_medium_slope"],
        "high_slope_stats": stats_map["point_ground_to_ground_high_slope"],
    }


def build_software_audit(cfg: Config) -> Dict[str, Any]:
    gdaltransform = resolve_executable(cfg.gdaltransform_exe, "gdaltransform")
    cct_exe = resolve_executable(cfg.cct_exe, "cct")
    projinfo_exe = resolve_executable(cfg.projinfo_exe, "projinfo")
    pdal_exe = resolve_executable(cfg.pdal_exe, "pdal")
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pyproj_version": pyproj.__version__,
        "proj_version": pyproj.proj_version_str,
        "gdal_python_available": osr is not None,
        "gdal_python_version": getattr(gdal, "__version__", None),
        "gdaltransform_version": get_command_version([gdaltransform, "--version"]) if gdaltransform else None,
        "cct_version": get_command_version([cct_exe, "-V"]) if cct_exe else None,
        "projinfo_version": get_command_version([projinfo_exe, "--version"]) if projinfo_exe else None,
        "pdal_version": get_command_version([pdal_exe, "--version"]) if pdal_exe else None,
        "PROJ_DATA": os.environ.get("PROJ_DATA"),
        "PROJ_LIB": os.environ.get("PROJ_LIB"),
        "PROJ_NETWORK": os.environ.get("PROJ_NETWORK"),
    }


def run_self_checks(pair_result: Dict[str, Any], cfg: Config) -> None:
    transform_results = pair_result["transform_results"]
    pyproj_result = transform_results["pyproj"]
    if pyproj_result.status == "ok":
        pyproj_shift_median = robust_stats(pyproj_result.z_shift, factor=cfg.robust_nmad_factor)["median"]
        if abs(pyproj_shift_median) <= cfg.zero_vertical_shift_abs_tol_m:
            raise AssertionError("pyproj transform unexpectedly produced near-zero vertical shift.")

    pyproj_nonzero = pyproj_result.z_shift is not None and not is_near_zero_vertical_shift(pyproj_result.z_shift, cfg)
    for method in ["gdal_osr", "gdaltransform_noct", "gdaltransform_ct", "proj_cct_pipeline", "proj_cct_operation"]:
        result = transform_results.get(method)
        if result is None or result.z_shift is None:
            continue
        if pyproj_nonzero and is_near_zero_vertical_shift(result.z_shift, cfg) and result.status == "ok":
            raise AssertionError(f"{method} produced zero-shift but was marked ok.")
    gdal_ct = transform_results.get("gdaltransform_ct")
    if gdal_ct is not None and gdal_ct.status == "ok":
        if not pair_result["transform_agreement"].get("per_method", {}).get("gdaltransform_ct", {}).get("agree", False):
            raise AssertionError("gdaltransform_ct was marked ok but does not agree with pyproj.")

    two_stage = transform_results.get("two_stage_ngs_geoid_plus_pyproj_frame")
    if two_stage is not None and two_stage.status == "ok":
        diff_med = abs(two_stage.median_diff_to_pyproj_m or math.nan)
        if np.isfinite(diff_med) and diff_med > cfg.two_stage_shift_agreement_tol_m:
            pair_result.setdefault("warnings_runtime", []).append("two_stage_check_differs_from_pyproj")

    point_df = pair_result["residuals"]["point_df"]
    ground_count = len(pair_result["casals_ground"]["ground_df"])
    if len(point_df) > ground_count:
        raise AssertionError("Point residuals were not restricted to final CASALS ground points.")

    required_cols = {
        "casals_ground_z_target",
        "dep_ground_z_target_at_casals_xy",
        "residual_casals_minus_3dep",
        "nearest_3dep_ground_distance_m",
        "local_slope_deg",
    }
    if not required_cols.issubset(point_df.columns):
        raise AssertionError("Point residual CSV schema is missing required fields.")

    summary_row = pair_result["summary_row"]
    required_summary_prefixes = [
        "raw_height_system",
        "geoid_only",
        "transformed_high_snr_sanity",
        "main_point_ground_residual",
        "main_grid_ground_residual",
    ]
    for prefix in required_summary_prefixes:
        if not any(key.startswith(prefix) for key in summary_row):
            raise AssertionError(f"Summary row is missing {prefix} fields.")

    for key in summary_row:
        lowered = key.lower()
        if any(pattern in lowered for pattern in LEGACY_FIELD_PATTERNS):
            raise AssertionError(f"Legacy field name detected in final summary: {key}")

    ngs_status = summary_row.get("ngs_geoid_status")
    geoid_only_resid = summary_row.get("geoid_only_residual_median_m")
    if (geoid_only_resid is None or not np.isfinite(geoid_only_resid)) and ngs_status != "failed":
        raise AssertionError("geoid_only_residual_median_m is NaN without ngs_geoid_status=failed.")
    if ngs_status == "ok":
        pyproj_median = summary_row.get("transform_main_median_z_shift_m")
        ngs_median = summary_row.get("ngs_geoid_median_m")
        if np.isfinite(pyproj_median) and np.isfinite(ngs_median):
            if abs(pyproj_median - ngs_median) > cfg.pyproj_vs_ngs_geoid_warning_abs_m:
                pair_result.setdefault("warnings_runtime", []).append("pyproj_vs_ngs_geoid_large_difference")


def process_pair(pair: PairConfig, cfg: Config) -> Dict[str, Any]:
    pair_dir = ensure_dir(cfg.out_root / f"pair_{safe_label(pair.label)}")
    metadata_dir = ensure_dir(pair_dir / "00_metadata")
    transform_dir = ensure_dir(pair_dir / "01_transform")
    ensure_dir(pair_dir / "02_ground_extraction")
    ensure_dir(pair_dir / "03_residuals")

    print_pair_header(pair)

    print_step(1, 7, "Input metadata")
    print(f"  CASALS H5: {pair.h5_path}")
    print(f"  3DEP LAS:  {pair.dep_las_path}")
    raw_casals = read_casals_raw_points(pair.h5_path)
    dep = read_dep_ground_points(pair.dep_las_path, pair, cfg)
    casals_crs_audit = inspect_casals_crs(pair.h5_path, pair)
    crs_audit = dep["crs_audit"]
    write_json(metadata_dir / "crs_audit.json", crs_audit)
    write_json(metadata_dir / "casals_metadata_audit.json", casals_crs_audit)
    write_json(metadata_dir / "software_audit.json", build_software_audit(cfg))

    print(f"  CASALS CRS: {pair.casals_crs} WGS84 3D, verified={pair.casals_vertical_reference_verified}")
    print(f"  3DEP horizontal CRS: EPSG:{crs_audit['horizontal_epsg']} {crs_audit['horizontal_name']}")
    print(f"  3DEP vertical CRS: EPSG:{crs_audit['vertical_epsg']} {crs_audit['vertical_name']}, source={crs_audit['vertical_crs_source']}, verified={crs_audit['vertical_crs_verified']}")
    if not crs_audit["vertical_crs_verified"]:
        print("  WARNING: 3DEP vertical CRS not metadata-verified")

    if cfg.enable_pyproj_check:
        try:
            network.set_network_enabled(True)
        except Exception:
            pass

    casals_points = build_casals_points_in_dep_horizontal(raw_casals, pair, crs_audit["horizontal_crs"])
    print(f"  CASALS total records: {casals_points['n_total']:,}")
    print(f"  3DEP total points: {dep['n_total']:,}; class-{cfg.dep_ground_class_code} ground used: {dep['n_ground_used']:,}")

    print_step(2, 7, "Coordinate transformation")
    source_compound = crs_audit["source_compound_crs"]
    if source_compound is None:
        raise RuntimeError("Source compound CRS could not be constructed.")
    main_target_crs = CRS.from_user_input(cfg.main_target_crs)
    print(f"  Source compound CRS: {crs_audit['source_compound_crs_string']}")
    print(f"  Target CRS: {cfg.main_target_crs}")

    ngs = ngs_geoid_check(casals_points, cfg, transform_dir)
    projinfo_audit = run_projinfo_audit(source_compound, main_target_crs, transform_dir, cfg)
    pyproj_full, pyproj_context = transform_with_pyproj(dep["x"], dep["y"], dep["z"], source_compound, main_target_crs, cfg)
    if pyproj_full.status not in {"ok", "degraded_missing_grid"}:
        raise RuntimeError(f"pyproj main transform failed: {pyproj_full.status}")

    dep_xy = np.column_stack([dep["x"], dep["y"]])
    bbox_mask = (
        (dep["x"] >= np.nanmin(casals_points["x"])) &
        (dep["x"] <= np.nanmax(casals_points["x"])) &
        (dep["y"] >= np.nanmin(casals_points["y"])) &
        (dep["y"] <= np.nanmax(casals_points["y"]))
    )
    sample_pool = np.flatnonzero(bbox_mask)
    if sample_pool.size == 0:
        sample_pool = np.arange(len(dep["x"]), dtype=int)
    sample_rel = choose_indices(sample_pool.size, min(TRANSFORM_VALIDATION_SAMPLE_LIMIT, sample_pool.size), cfg.random_seed)
    sample_idx = sample_pool[sample_rel]
    sample_x = dep["x"][sample_idx]
    sample_y = dep["y"][sample_idx]
    sample_z = dep["z"][sample_idx]
    pyproj_sample = TransformResult(
        method="pyproj",
        source_crs=pyproj_full.source_crs,
        target_crs=pyproj_full.target_crs,
        status=pyproj_full.status,
        z_transformed=pyproj_full.z_transformed[sample_idx],
        z_shift=pyproj_full.z_shift[sample_idx],
        operation_description=pyproj_full.operation_description,
        operation_accuracy=pyproj_full.operation_accuracy,
        best_available=pyproj_full.best_available,
        missing_grids=pyproj_full.missing_grids,
        command=pyproj_full.command,
        software=pyproj_full.software,
        notes=pyproj_full.notes,
        input_coordinate_type=pyproj_full.input_coordinate_type,
        pipeline_available=pyproj_full.pipeline_available,
        pipeline_validated=pyproj_full.pipeline_validated,
    )

    source_crs_string = crs_audit["source_compound_crs_string"] or source_compound.to_string()
    target_crs_string = main_target_crs.to_string()
    candidate_pipelines: List[str] = []
    if pyproj_context.get("pipeline"):
        candidate_pipelines.append(pyproj_context["pipeline"])
    candidate_pipelines.extend([pipe for pipe in projinfo_audit.get("pipelines", []) if pipe not in candidate_pipelines])
    validated_pipeline_info = select_validated_pipeline(
        candidate_pipelines,
        sample_x,
        sample_y,
        sample_z,
        crs_audit["horizontal_crs"],
        cfg,
    )
    executable_pipeline = validated_pipeline_info["pipeline"]
    executable_input_type = validated_pipeline_info["input_coordinate_type"]
    pipeline_validated = validated_pipeline_info["pipeline_validated"]
    transform_results = {
        "pyproj": pyproj_sample,
        "gdal_osr": transform_with_gdal_osr(sample_x, sample_y, sample_z, source_compound, main_target_crs, cfg),
        "gdaltransform_noct": transform_with_gdaltransform_cli(
            sample_x, sample_y, sample_z, source_crs_string, target_crs_string, cfg, transform_dir, "noct"
        ),
        "gdaltransform_ct": transform_with_gdaltransform_cli(
            sample_x, sample_y, sample_z, source_crs_string, target_crs_string, cfg, transform_dir, "ct", executable_pipeline, crs_audit["horizontal_crs"], executable_input_type
        ),
        "proj_cct_pipeline": transform_with_proj_cct_pipeline(
            sample_x, sample_y, sample_z, crs_audit["horizontal_crs"], cfg, transform_dir, executable_pipeline, pipeline_validated
        ),
        "proj_cct_operation": transform_with_proj_cct_operation(
            sample_x, sample_y, sample_z, crs_audit["horizontal_crs"], cfg, transform_dir, projinfo_audit.get("operation_names", [None])[0]
        ),
        "two_stage_ngs_geoid_plus_pyproj_frame": two_stage_ngs_geoid_plus_pyproj_frame(
            sample_x, sample_y, sample_z, crs_audit["horizontal_crs"], main_target_crs, ngs, cfg
        ),
    }
    for result in transform_results.values():
        result.projinfo_status = projinfo_audit.get("status")
        result.ngs_geoid_status = ngs.get("status")
        if result.method in {"gdaltransform_ct", "proj_cct_pipeline"}:
            result.pipeline_available = bool(executable_pipeline)
            result.pipeline_validated = pipeline_validated
            result.input_coordinate_type = executable_input_type or result.input_coordinate_type
    transform_agreement = evaluate_transform_agreement(transform_results, cfg)

    pyproj_shift_stats = robust_stats(pyproj_sample.z_shift, factor=cfg.robust_nmad_factor)
    ngs_preferred = ngs.get("preferred")
    if ngs_preferred:
        print(f"  NGS GEOID model {ngs_preferred['ngs_model']}: status={ngs['status']}, median={ngs_preferred['ngs_geoid_median_m']:.3f} m, nmad={ngs_preferred['ngs_geoid_nmad_m']:.3f} m")
    else:
        print(f"  NGS GEOID model 14: status={ngs.get('status')}, median=nan m")
    print(f"  pyproj: status={pyproj_sample.status}, median shift={pyproj_shift_stats['median']:.3f} m, best_available={pyproj_sample.best_available}, missing_grids={len(pyproj_sample.missing_grids)}")
    print(f"  pyproj operation: {pyproj_sample.operation_description}")
    print(
        f"  projinfo: status={projinfo_audit.get('status')}, "
        f"vertical_grid_detected={projinfo_audit.get('projinfo_mentions_vertical_grid')}, "
        f"navd88_detected={projinfo_audit.get('projinfo_mentions_navd88')}"
    )
    for method_name in ["gdal_osr", "gdaltransform_noct", "gdaltransform_ct", "proj_cct_pipeline", "proj_cct_operation", "two_stage_ngs_geoid_plus_pyproj_frame"]:
        result = transform_results[method_name]
        stats = robust_stats(result.z_shift, factor=cfg.robust_nmad_factor) if result.z_shift is not None else {"median": math.nan}
        line = f"  {method_name}: status={result.status}, median shift={stats['median']:.3f} m"
        compare = transform_agreement.get("per_method", {}).get(method_name)
        if compare:
            line += f", diff-to-pyproj median={compare.get('median_difference_m'):.3f} m, agree_with_pyproj={compare.get('agree')}"
        print(line)
        if result.z_shift is not None and is_near_zero_vertical_shift(result.z_shift, cfg) and not is_near_zero_vertical_shift(pyproj_sample.z_shift, cfg):
            print(f"  WARNING: {method_name} produced near-zero vertical shift while pyproj did not")
    print(f"  Transform agreement: {transform_agreement['transform_agreement_status']}")
    if transform_agreement["single_transform_path_only"]:
        print("  WARNING: only pyproj produced a valid non-zero vertical transform.")

    cross_validation_df = build_transform_cross_validation_rows(pair.label, transform_results, transform_agreement, cfg)
    cross_validation_df.to_csv(transform_dir / "coordinate_transform_cross_validation.csv", index=False)
    write_json(transform_dir / "transform_agreement_summary.json", transform_agreement)
    (transform_dir / "pyproj_operation.txt").write_text(
        "\n".join([
            f"description: {pyproj_sample.operation_description}",
            f"accuracy: {pyproj_sample.operation_accuracy}",
            f"best_available: {pyproj_sample.best_available}",
            f"raw_pipeline_candidate: {pyproj_context.get('pipeline')}",
            f"validated_executable_pipeline: {executable_pipeline}",
            f"pipeline_validation_reason: {validated_pipeline_info.get('pipeline_validation_reason')}",
            f"missing_grids: {json.dumps(as_jsonable(pyproj_sample.missing_grids), ensure_ascii=False)}",
        ]),
        encoding="utf-8",
    )
    gdal_commands = []
    for method_name in ["gdaltransform_noct", "gdaltransform_ct"]:
        result = transform_results[method_name]
        if result.command:
            gdal_commands.append(f"{method_name}: {result.command}")
    (transform_dir / "gdaltransform_commands.txt").write_text("\n".join(gdal_commands), encoding="utf-8")
    make_transform_histogram(transform_results, transform_dir / "transform_shift_histograms.png", cfg)

    print_step(3, 7, "CASALS ground extraction")
    casals_ground = derive_casals_ground_points(casals_points, cfg, pair_dir)
    ground_summary = casals_ground["summary"]
    print(f"  total records={ground_summary['n_total_records']:,}")
    print(f"  finite records={ground_summary['n_finite_records']:,}")
    print(f"  SNR>{cfg.casals_snr_ground_candidate_threshold:g} candidates={ground_summary['n_snr_candidates']:,} used={ground_summary['n_snr_candidates_used']:,}")
    print(f"  after neighbor filter={ground_summary['n_after_neighbor_filter']:,}")
    print(f"  after local Z outlier filter={ground_summary['n_after_local_outlier_filter']:,}")
    print(f"  ground seeds={ground_summary['n_ground_seeds']:,}")
    print(f"  final CASALS ground points={ground_summary['n_final_ground_points']:,}")
    print(f"  final ground fraction={percent_str(ground_summary['n_final_ground_points'], ground_summary['n_snr_candidates_used'])}")
    print(f"  candidate z range={ground_summary['candidate_z_range_m'][0]:.3f} to {ground_summary['candidate_z_range_m'][1]:.3f} m")
    print(f"  ground z range={ground_summary['ground_z_range_m'][0]:.3f} to {ground_summary['ground_z_range_m'][1]:.3f} m")

    print_step(4, 7, "3DEP ground surface")
    dep_surface = build_3dep_ground_surface(dep, pyproj_full, cfg, pair_dir)
    print(f"  total 3DEP points={dep['n_total']:,}")
    print(f"  class-2 ground points={dep['n_ground_used']:,}")
    print(f"  transformed ground points={len(dep_surface['z_target']):,}")
    print(f"  KDTree points={len(dep_surface['x']):,}")

    print_step(5, 7, "Pointwise ground-to-ground residual")
    residuals = compute_ground_to_ground_residuals(casals_ground, dep_surface, casals_ground, ngs, cfg, pair_dir)
    point_stats = residuals["point_stats"]
    print(f"  n={point_stats['n']:,}")
    print(f"  median={point_stats['median']:.3f} m")
    print(f"  NMAD={point_stats['nmad']:.3f} m")
    print(f"  RMSE={point_stats['rmse']:.3f} m")
    print(f"  p05/p95={point_stats['p05']:.3f}/{point_stats['p95']:.3f} m")
    print(f"  abs<=0.5m={point_stats['fraction_abs_le_0p50m']:.3f}")
    print(f"  abs<=1.0m={point_stats['fraction_abs_le_1m']:.3f}")

    print_step(6, 7, "Grid-to-grid ground residual")
    grid_stats = residuals["grid_stats"]
    print(f"  n_cells={grid_stats['n']:,}")
    print(f"  median={grid_stats['median']:.3f} m")
    print(f"  NMAD={grid_stats['nmad']:.3f} m")
    print(f"  RMSE={grid_stats['rmse']:.3f} m")
    print(f"  p05/p95={grid_stats['p05']:.3f}/{grid_stats['p95']:.3f} m")

    warnings_list: List[str] = []
    if not crs_audit["vertical_crs_verified"]:
        warnings_list.append("3dep_vertical_crs_unverified")
    if ngs.get("status") != "ok":
        warnings_list.append("ngs_geoid_check_failed")
    if transform_agreement["single_transform_path_only"]:
        warnings_list.append("single_transform_path_only")
    if transform_agreement["transform_agreement_status"] == "transform_disagreement":
        warnings_list.append("transform_disagreement")
    if transform_results["gdal_osr"].status == "failed_zero_vertical_shift" or transform_results["gdaltransform_noct"].status == "failed_zero_vertical_shift":
        warnings_list.append("gdal_default_path_no_vertical_shift")
    if transform_results["gdaltransform_ct"].status == "skipped_missing_executable_pipeline":
        warnings_list.append("gdal_ct_pipeline_unavailable")
    if ngs_preferred is not None and abs(pyproj_shift_stats["median"] - ngs_preferred["ngs_geoid_median_m"]) > cfg.pyproj_vs_ngs_geoid_warning_abs_m:
        warnings_list.append("pyproj_vs_ngs_geoid_large_difference")
    two_stage_cmp = transform_agreement.get("per_method", {}).get("two_stage_ngs_geoid_plus_pyproj_frame", {})
    if transform_results["two_stage_ngs_geoid_plus_pyproj_frame"].status == "ok" and np.isfinite(two_stage_cmp.get("median_difference_m", math.nan)):
        if abs(two_stage_cmp["median_difference_m"]) > cfg.two_stage_shift_agreement_tol_m:
            warnings_list.append("two_stage_check_differs_from_pyproj")
    if point_stats["median"] is not None and np.isfinite(point_stats["median"]) and abs(point_stats["median"]) > cfg.warning_residual_abs_m:
        warnings_list.append("point_ground_residual_exceeds_warning_threshold")
    if grid_stats["median"] is not None and np.isfinite(grid_stats["median"]) and abs(grid_stats["median"]) > cfg.warning_residual_abs_m:
        warnings_list.append("grid_ground_residual_exceeds_warning_threshold")

    print_step(7, 7, "Pair conclusion")
    print(f"  raw height-system offset={residuals['raw_stats']['median']:.3f} m")
    if ngs_preferred:
        print(f"  NGS GEOID model {ngs_preferred['ngs_model']}: status={ngs['status']}, median geoid={ngs_preferred['ngs_geoid_median_m']:.3f} m, geoid-only residual median={residuals['geoid_stats']['median']:.3f} m")
    print(f"  transformed ground-to-ground offset={point_stats['median']:.3f} m")
    print(f"  warnings={';'.join(warnings_list) if warnings_list else 'none'}")

    summary_row = {
        "label": pair.label,
        "casals_crs": pair.casals_crs,
        "casals_vertical_reference_verified": pair.casals_vertical_reference_verified,
        "dep_horizontal_epsg": crs_audit["horizontal_epsg"],
        "dep_vertical_epsg": crs_audit["vertical_epsg"],
        "dep_vertical_crs_source": crs_audit["vertical_crs_source"],
        "dep_vertical_crs_verified": crs_audit["vertical_crs_verified"],
        "transform_main_method": "pyproj",
        "transform_main_status": pyproj_full.status,
        "transform_main_median_z_shift_m": robust_stats(pyproj_full.z_shift, cfg.robust_nmad_factor)["median"],
        "transform_agreement_status": transform_agreement["transform_agreement_status"],
        "ngs_geoid_status": ngs.get("status"),
        "ngs_geoid_median_m": ngs_preferred["ngs_geoid_median_m"] if ngs_preferred else math.nan,
        "ngs_geoid_nmad_m": ngs_preferred["ngs_geoid_nmad_m"] if ngs_preferred else math.nan,
        "two_stage_check_status": transform_results["two_stage_ngs_geoid_plus_pyproj_frame"].status,
        "external_transform_check_status": "ok" if transform_agreement["transform_agreement_status"] == "multi_method_agree" else "limited",
        "n_casals_total": casals_points["n_total"],
        "n_casals_snr_candidates": ground_summary["n_snr_candidates"],
        "n_casals_ground_final": ground_summary["n_final_ground_points"],
        "n_3dep_total": dep["n_total"],
        "n_3dep_ground": dep["n_ground_used"],
        "raw_height_system_offset_median_m": residuals["raw_stats"]["median"],
        "raw_height_system_offset_nmad_m": residuals["raw_stats"]["nmad"],
        "geoid_only_residual_median_m": residuals["geoid_stats"]["median"],
        "geoid_only_residual_nmad_m": residuals["geoid_stats"]["nmad"],
        "transformed_high_snr_sanity_residual_median_m": residuals["transformed_high_snr_stats"]["median"],
        "main_point_ground_residual_median_m": residuals["point_stats"]["median"],
        "main_point_ground_residual_nmad_m": residuals["point_stats"]["nmad"],
        "main_point_ground_residual_rmse_m": residuals["point_stats"]["rmse"],
        "main_grid_ground_residual_median_m": residuals["grid_stats"]["median"],
        "main_grid_ground_residual_nmad_m": residuals["grid_stats"]["nmad"],
        "main_grid_ground_residual_rmse_m": residuals["grid_stats"]["rmse"],
        "low_slope_point_ground_residual_median_m": residuals["low_slope_stats"]["median"],
        "low_slope_point_ground_residual_nmad_m": residuals["low_slope_stats"]["nmad"],
        "warnings": ";".join(sorted(set(warnings_list))),
        "notes": pair.notes,
    }

    pair_report = {
        "script": "diagnose_3dep_offsets.py",
        "interpretation_guidance": INTERPRETATION_GUIDANCE,
        "pair": asdict(pair),
        "crs_audit": crs_audit,
        "casals_crs_audit": casals_crs_audit,
        "ngs_geoid_check": ngs,
        "projinfo_audit": projinfo_audit,
        "transform_agreement_summary": transform_agreement,
        "ground_extraction_summary": ground_summary,
        "dep_ground_surface_summary": dep_surface["summary"],
        "residual_overview": {
            "raw_height_system_offset": residuals["raw_stats"],
            "geoid_only_residual": residuals["geoid_stats"],
            "transformed_high_snr_sanity_residual": residuals["transformed_high_snr_stats"],
            "point_ground_to_ground": residuals["point_stats"],
            "grid_ground_to_ground": residuals["grid_stats"],
            "low_slope_point_ground_to_ground": residuals["low_slope_stats"],
        },
        "warnings": warnings_list,
        "outputs": {
            "metadata_dir": str(metadata_dir),
            "transform_dir": str(transform_dir),
            "ground_extraction_dir": str(pair_dir / "02_ground_extraction"),
            "residual_dir": str(pair_dir / "03_residuals"),
        },
    }
    write_json(pair_dir / "pair_report.json", pair_report)

    pair_result = {
        "summary_row": summary_row,
        "pair_report": pair_report,
        "transform_results": transform_results,
        "transform_agreement": transform_agreement,
        "cross_validation_df": cross_validation_df,
        "casals_ground": casals_ground,
        "dep_surface": dep_surface,
        "residuals": residuals,
    }
    run_self_checks(pair_result, cfg)
    return pair_result


def make_aggregate_plots(summary_df: pd.DataFrame, transform_df: pd.DataFrame, out_root: Path, cfg: Config) -> None:
    fig_dir = ensure_dir(out_root / "figures")
    if summary_df.empty:
        return

    labels = summary_df["label"].astype(str).tolist()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels)), 5), dpi=cfg.dpi)
    ax.bar(x, summary_df["raw_height_system_offset_median_m"].to_numpy(dtype=float), color="#8c510a", alpha=0.7, label="raw height-system")
    ax.bar(x, summary_df["main_point_ground_residual_median_m"].to_numpy(dtype=float), color="#01665e", alpha=0.7, label="point ground-to-ground")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Median residual (m)")
    ax.set_title("Raw vs transformed ground-to-ground offsets by pair")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "ground_offset_overview_by_pair.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels)), 5), dpi=cfg.dpi)
    width = 0.35
    ax.bar(x - width / 2, summary_df["main_point_ground_residual_nmad_m"].to_numpy(dtype=float), width=width, label="point NMAD")
    ax.bar(x + width / 2, summary_df["main_grid_ground_residual_nmad_m"].to_numpy(dtype=float), width=width, label="grid NMAD")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("NMAD (m)")
    ax.set_title("Ground-to-ground residual spread by pair")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "ground_offset_spread_by_pair.png")
    plt.close(fig)

    if not transform_df.empty:
        pivot = transform_df.pivot(index="pair_label", columns="method", values="median_z_shift_m")
        pivot = pivot.reindex(labels)
        fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(labels)), 5), dpi=cfg.dpi)
        width = max(0.12, 0.7 / max(1, len(pivot.columns)))
        methods = list(pivot.columns)
        for i, method in enumerate(methods):
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, pivot[method].to_numpy(dtype=float), width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Median z shift (m)")
        ax.set_title("Transform cross-validation median shift by pair")
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "transform_cross_validation_by_pair.png")
        plt.close(fig)


def make_markdown_report(summary_df: pd.DataFrame, transform_df: pd.DataFrame, out_root: Path) -> None:
    lines: List[str] = []
    lines.append("# CASALS-3DEP Ground Offset Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This workflow diagnoses CASALS-3DEP vertical agreement using a ground-to-ground interpretation, not a raw mixed-datum height difference.")
    lines.append("")
    lines.append("## 2. Coordinate and height systems")
    lines.append("")
    lines.append("- CASALS is treated as WGS84 ellipsoidal height (`EPSG:4979`).")
    lines.append("- 3DEP is treated as NAD83(2011) horizontal + NAVD88 orthometric height, with per-pair provenance recorded as metadata-verified or assumed.")
    lines.append("")
    lines.append("## 3. Coordinate transformation validation")
    lines.append("")
    lines.append("- pyproj/PROJ remains the main transform path.")
    lines.append("- GDAL OSR and `gdaltransform` without explicit `-ct` are retained as diagnostic checks; near-zero vertical shift means the default GDAL path did not apply the vertical operation.")
    lines.append("- `gdaltransform -ct` and PROJ `cct` are used only when an explicit executable pipeline can be validated safely.")
    lines.append("- The NGS geoid-only check is a magnitude sanity check, not a full WGS84 or ITRF transformation.")
    lines.append("")
    if not transform_df.empty:
        display_cols = [
            "pair_label",
            "method",
            "status",
            "median_z_shift_m",
            "nmad_z_shift_m",
            "median_diff_to_pyproj_m",
            "nmad_diff_to_pyproj_m",
            "best_available",
            "agrees_with_pyproj",
            "input_coordinate_type",
            "pipeline_validated",
            "projinfo_status",
            "ngs_geoid_status",
        ]
        display_cols = [col for col in display_cols if col in transform_df.columns]
        lines.append(dataframe_to_markdown(transform_df[display_cols]))
    else:
        lines.append("No transform cross-validation rows were produced.")
    lines.append("")
    lines.append("## 4. CASALS ground extraction")
    lines.append("")
    lines.append("- SNR thresholding produces initial candidates only.")
    lines.append("- Neighbor filtering removes isolated sparse noise.")
    lines.append("- Local Z filtering removes local outliers.")
    lines.append("- A low-envelope surface identifies likely ground seeds.")
    lines.append("- Final CASALS ground points are derived after secondary robust filtering.")
    lines.append("")
    lines.append("## 5. Main residual definition")
    lines.append("")
    lines.append("- Pointwise residual: `CASALS-derived ground height - transformed 3DEP ground height sampled at the same horizontal location`.")
    lines.append("- Grid residual: `CASALS-derived DTM - transformed 3DEP DTM` on common occupied grid cells.")
    lines.append("- Raw high-SNR residual is reported only as a sanity check and is not the final residual.")
    lines.append("")
    lines.append("## 6. Results")
    lines.append("")
    if summary_df.empty:
        lines.append("No per-pair summary rows were produced.")
    else:
        display_cols = [
            "label",
            "transform_agreement_status",
            "ngs_geoid_status",
            "two_stage_check_status",
            "external_transform_check_status",
            "ngs_geoid_median_m",
            "raw_height_system_offset_median_m",
            "geoid_only_residual_median_m",
            "main_point_ground_residual_median_m",
            "main_point_ground_residual_nmad_m",
            "main_grid_ground_residual_median_m",
            "main_grid_ground_residual_nmad_m",
            "low_slope_point_ground_residual_median_m",
            "warnings",
        ]
        display_cols = [col for col in display_cols if col in summary_df.columns]
        lines.append(dataframe_to_markdown(summary_df[display_cols]))
    lines.append("")
    lines.append("## 7. Interpretation")
    lines.append("")
    for item in INTERPRETATION_GUIDANCE:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 8. Warnings and limitations")
    lines.append("")
    lines.append("- 3DEP vertical CRS may remain assumed rather than metadata-verified for some pairs.")
    lines.append("- `ngs_geoid_check_failed` means the API sanity check was unavailable or returned unusable responses.")
    lines.append("- `gdal_default_path_no_vertical_shift` means default GDAL execution did not apply a vertical transform and is not used as evidence.")
    lines.append("- `gdal_ct_pipeline_unavailable` means no validated explicit pipeline was available for `gdaltransform -ct`.")
    lines.append("- `single_transform_path_only` means only pyproj produced a valid non-zero vertical transform.")
    lines.append("- GDAL/PROJ external methods may still be skipped when executables or bindings are unavailable.")
    lines.append("- CASALS ground extraction remains a derived product from pointwise candidates, not a native LAS ground classification.")
    lines.append("- No corrected LAS is written.")
    (out_root / "cross_pair_ground_offset_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = Config(
        pairs=[
            PairConfig(
                label="casals_20241112_MD_Southeast_1_2019",
                h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
                dep_las_path=Path(r"./point_cloud_data/download_3dep_lpc/casals_l1b_20241112T165718_001_02_MD_Southeast_1_2019_EPSG6347_39a068a77804.laz"),
            ),
            PairConfig(
                label="casals_20241112_DE_Statewide_1_B23",
                h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241112T170442_001_02.h5"),
                dep_las_path=Path(r"./point_cloud_data/download_3dep_lpc/casals_l1b_20241112T170442_001_02_DE_Statewide_1_B23_EPSG6347_b60d6cbd5f2f.laz"),
            ),
            PairConfig(
                label="casals_20241118_NC_HurricaneFlorence_9_2020",
                h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241118T171757_001_02.h5"),
                dep_las_path=Path(r"./point_cloud_data/download_3dep_lpc/casals_l1b_20241118T171757_001_02_NC_HurricaneFlorence_9_2020_EPSG6347_f50533a04725.laz"),
            ),
        ],
        out_root=Path(r"./outputs/diagnose_3dep_ground_offsets"),
    )

    ensure_dir(cfg.out_root)
    print("=" * 80)
    print("CASALS-3DEP ground-to-ground vertical residual diagnosis")
    print("=" * 80)
    print(f"Output root: {cfg.out_root.resolve()}")
    print(f"Pairs: {len(cfg.pairs)}")
    print(f"CASALS SNR threshold: {cfg.casals_snr_ground_candidate_threshold}")
    print(f"Main target CRS: {cfg.main_target_crs}")

    summary_rows: List[Dict[str, Any]] = []
    pair_reports: Dict[str, Any] = {}
    transform_tables: List[pd.DataFrame] = []
    failed_pairs: List[str] = []

    for pair in cfg.pairs:
        try:
            result = process_pair(pair, cfg)
            summary_rows.append(result["summary_row"])
            pair_reports[pair.label] = result["pair_report"]
            transform_tables.append(result["cross_validation_df"])
        except Exception as exc:
            failed_pairs.append(pair.label)
            summary_rows.append({
                "label": pair.label,
                "casals_crs": pair.casals_crs,
                "casals_vertical_reference_verified": pair.casals_vertical_reference_verified,
                "dep_horizontal_epsg": pair.dep_horizontal_epsg_override,
                "dep_vertical_epsg": pair.dep_vertical_epsg,
                "dep_vertical_crs_source": pair.dep_vertical_crs_source,
                "dep_vertical_crs_verified": pair.dep_vertical_crs_verified,
                "transform_main_method": "pyproj",
                "transform_main_status": "failed",
                "transform_main_median_z_shift_m": math.nan,
                "transform_agreement_status": "failed_pair",
                "ngs_geoid_status": "failed",
                "ngs_geoid_median_m": math.nan,
                "ngs_geoid_nmad_m": math.nan,
                "two_stage_check_status": "failed",
                "external_transform_check_status": "failed",
                "n_casals_total": math.nan,
                "n_casals_snr_candidates": math.nan,
                "n_casals_ground_final": math.nan,
                "n_3dep_total": math.nan,
                "n_3dep_ground": math.nan,
                "raw_height_system_offset_median_m": math.nan,
                "raw_height_system_offset_nmad_m": math.nan,
                "geoid_only_residual_median_m": math.nan,
                "geoid_only_residual_nmad_m": math.nan,
                "transformed_high_snr_sanity_residual_median_m": math.nan,
                "main_point_ground_residual_median_m": math.nan,
                "main_point_ground_residual_nmad_m": math.nan,
                "main_point_ground_residual_rmse_m": math.nan,
                "main_grid_ground_residual_median_m": math.nan,
                "main_grid_ground_residual_nmad_m": math.nan,
                "main_grid_ground_residual_rmse_m": math.nan,
                "low_slope_point_ground_residual_median_m": math.nan,
                "low_slope_point_ground_residual_nmad_m": math.nan,
                "warnings": f"pair_failed:{exc}",
            })
            pair_reports[pair.label] = {"error": str(exc)}

    summary_df = pd.DataFrame(summary_rows)
    summary_columns = [
        "label",
        "casals_crs",
        "casals_vertical_reference_verified",
        "dep_horizontal_epsg",
        "dep_vertical_epsg",
        "dep_vertical_crs_source",
        "dep_vertical_crs_verified",
        "transform_main_method",
        "transform_main_status",
        "transform_main_median_z_shift_m",
        "transform_agreement_status",
        "ngs_geoid_status",
        "ngs_geoid_median_m",
        "ngs_geoid_nmad_m",
        "two_stage_check_status",
        "external_transform_check_status",
        "n_casals_total",
        "n_casals_snr_candidates",
        "n_casals_ground_final",
        "n_3dep_total",
        "n_3dep_ground",
        "raw_height_system_offset_median_m",
        "raw_height_system_offset_nmad_m",
        "geoid_only_residual_median_m",
        "geoid_only_residual_nmad_m",
        "transformed_high_snr_sanity_residual_median_m",
        "main_point_ground_residual_median_m",
        "main_point_ground_residual_nmad_m",
        "main_point_ground_residual_rmse_m",
        "main_grid_ground_residual_median_m",
        "main_grid_ground_residual_nmad_m",
        "main_grid_ground_residual_rmse_m",
        "low_slope_point_ground_residual_median_m",
        "low_slope_point_ground_residual_nmad_m",
        "warnings",
        "notes",
    ]
    summary_df = summary_df.reindex(columns=[col for col in summary_columns if col in summary_df.columns])
    summary_df.to_csv(cfg.out_root / "per_pair_ground_offset_summary.csv", index=False)
    summary_df.to_csv(cfg.out_root / "all_pairs_ground_to_ground_offset_summary.csv", index=False)
    write_json(
        cfg.out_root / "per_pair_ground_offset_summary.json",
        {"config": asdict(cfg), "summary": summary_rows, "pair_reports": pair_reports},
    )

    transform_df = pd.concat(transform_tables, ignore_index=True) if transform_tables else pd.DataFrame()
    transform_df.to_csv(cfg.out_root / "all_pairs_transform_cross_validation.csv", index=False)

    make_aggregate_plots(summary_df, transform_df, cfg.out_root, cfg)
    make_markdown_report(summary_df, transform_df, cfg.out_root)

    print("\n" + "=" * 80)
    print("GLOBAL SUMMARY")
    print("-" * 80)
    print(f"pairs processed: {len(cfg.pairs)}")
    print(f"failed pairs: {len(failed_pairs)}")
    if not transform_df.empty and "agrees_with_pyproj" in transform_df.columns:
        agree_counts = transform_df.groupby("method")["agrees_with_pyproj"].apply(lambda s: int(np.nansum(s == True))).to_dict()
        print(f"transform methods agreement summary: {agree_counts}")
    if not summary_df.empty and "main_point_ground_residual_median_m" in summary_df.columns:
        med = pd.to_numeric(summary_df["main_point_ground_residual_median_m"], errors="coerce").to_numpy(dtype=float)
        med = med[np.isfinite(med)]
        if med.size:
            print(f"main residual median range across pairs: {med.min():.3f} to {med.max():.3f} m")
    print(f"output report path: {cfg.out_root / 'cross_pair_ground_offset_report.md'}")


if __name__ == "__main__":
    main()
