#!/usr/bin/env python3
"""
Evaluate CASALS L1B reference-height points against 3DEP LPC ground points
under an explicit 3DEP reference-frame contract.

This script separates:
    - raw mixed-datum diagnostics,
    - empirical median-removed diagnostics,
    - H5 geoid-based height-conversion candidates,
    - final reference-frame residuals, only when CASALS heights are explicitly
      converted into the verified 3DEP reference frame.

It intentionally fails closed in strict reference mode whenever the 3DEP
vertical datum is unknown or the CASALS-to-3DEP height conversion is not
verified and selected.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import laspy
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer, datadir, network
from pyproj.crs import CompoundCRS
from pyproj.transformer import AreaOfInterest, TransformerGroup
from scipy.spatial import cKDTree

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    plt = None

try:
    from osgeo import osr

    HAS_GDAL_OSR = True
except Exception:
    HAS_GDAL_OSR = False
    osr = None


FINAL_NOT_COMPUTED_MESSAGE = (
    "Final CASALS accuracy against 3DEP was not computed because CASALS heights "
    "were not verified or converted into the 3DEP vertical reference frame."
)
FINAL_COMPUTED_MESSAGE = (
    "Final residuals are computed as CASALS height in the verified 3DEP "
    "reference frame minus local 3DEP class-2 ground median height."
)
USER_SELECTED_FORMULA_WARNING = (
    "The CASALS height conversion formula was explicitly selected by "
    "configuration. This result is suitable for controlled evaluation only if "
    "the selected formula is independently verified from CASALS product "
    "documentation or provider clarification."
)
STRICT_VERTICAL_UNAVAILABLE_WARNING = (
    "3DEP vertical CRS is unknown; final CASALS accuracy metrics cannot be "
    "computed in strict reference mode."
)
H5_DIAGNOSTIC_ONLY_MESSAGE = (
    "H5 geoid formula is diagnostic-only. Select a verified formula before "
    "final accuracy assessment."
)

FORMULA_TO_COLUMN = {
    "orthometric_height = refh - geoid": "casals_z_refh_minus_geoid_m",
    "orthometric_height = refh + geoid": "casals_z_refh_plus_geoid_m",
    "orthometric_height = refh - geoid - geoid_free2mean": "casals_z_refh_minus_geoid_minus_free2mean_m",
    "orthometric_height = refh - geoid + geoid_free2mean": "casals_z_refh_minus_geoid_plus_free2mean_m",
    "orthometric_height = refh + geoid - geoid_free2mean": "casals_z_refh_plus_geoid_minus_free2mean_m",
    "orthometric_height = refh + geoid + geoid_free2mean": "casals_z_refh_plus_geoid_plus_free2mean_m",
}


@dataclass
class CRSInfo:
    source_name: str
    crs_present: bool
    crs_name: Optional[str]
    authority: Optional[str]
    authority_code: Optional[str]
    is_geographic: Optional[bool]
    is_projected: Optional[bool]
    is_vertical: Optional[bool]
    is_compound: Optional[bool]
    axis_info: list[str]
    area_of_use: Optional[str]
    wkt: Optional[str]
    remarks: list[str]


@dataclass
class LasTileInfo:
    path: str
    file_name: str
    point_count: int
    point_format: str
    version: str
    scales: list[float]
    offsets: list[float]
    mins: list[float]
    maxs: list[float]
    vlr_count: int
    crs_parse_success: bool
    crs_name: Optional[str]
    crs_authority: Optional[str]
    crs_wkt: Optional[str]
    ground_point_count: int
    classification_counts: dict[str, int]


@dataclass
class TransformAudit:
    name: str
    source_crs: str
    target_crs: str
    allow_ballpark: bool
    always_xy: bool
    best_available: Optional[bool]
    transformer_count: Optional[int]
    unavailable_operations_count: Optional[int]
    first_transformer_description: Optional[str]
    first_transformer_accuracy_m: Optional[float]
    missing_grids: list[dict[str, Any]]
    z_change_median_m: Optional[float]
    z_change_p05_m: Optional[float]
    z_change_p95_m: Optional[float]
    status: str
    warnings: list[str]


@dataclass
class ResidualStats:
    name: str
    n: int
    mean_m: Optional[float]
    median_m: Optional[float]
    std_m: Optional[float]
    nmad_m: Optional[float]
    p05_m: Optional[float]
    p25_m: Optional[float]
    p75_m: Optional[float]
    p95_m: Optional[float]
    min_m: Optional[float]
    max_m: Optional[float]


@dataclass
class ReferenceFrameInfo:
    horizontal_crs_name: str
    horizontal_crs_authority: Optional[str]
    vertical_crs_name: Optional[str]
    vertical_crs_authority: Optional[str]
    vertical_crs_source: str
    geoid_model_or_source: Optional[str]
    reference_frame_status: str
    warnings: list[str]


@dataclass
class HeightConversionCandidateStats:
    candidate_name: str
    formula: str
    used_as_final: bool
    n: int
    mean_m: Optional[float]
    median_m: Optional[float]
    std_m: Optional[float]
    rmse_m: Optional[float]
    mae_m: Optional[float]
    nmad_m: Optional[float]
    p05_m: Optional[float]
    p25_m: Optional[float]
    p75_m: Optional[float]
    p95_m: Optional[float]
    min_m: Optional[float]
    max_m: Optional[float]
    interpretation: str


@dataclass
class FinalAccuracyStats:
    residual_name: str
    n: int
    bias_m: Optional[float]
    median_m: Optional[float]
    mean_m: Optional[float]
    std_m: Optional[float]
    rmse_m: Optional[float]
    mae_m: Optional[float]
    nmad_m: Optional[float]
    p05_m: Optional[float]
    p25_m: Optional[float]
    p75_m: Optional[float]
    p95_m: Optional[float]
    abs_p50_m: Optional[float]
    abs_p68_m: Optional[float]
    abs_p90_m: Optional[float]
    abs_p95_m: Optional[float]
    min_m: Optional[float]
    max_m: Optional[float]


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object is not JSON serializable: {type(obj)}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=to_jsonable)


def format_int(value: int) -> str:
    return f"{int(value):,}"


def log_status(message: str) -> None:
    print(f"[status] {message}", flush=True)


def log_step(step_num: int, total_steps: int, title: str) -> None:
    print(f"[{step_num}/{total_steps}] {title}", flush=True)


def is_blank(value: Optional[str]) -> bool:
    return value is None or not str(value).strip()


def robust_nmad(values: np.ndarray) -> Optional[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    med = float(np.median(arr))
    return float(1.4826 * np.median(np.abs(arr - med)))


def compute_distribution_metrics(values: np.ndarray) -> dict[str, Optional[float]]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean_m": None,
            "median_m": None,
            "std_m": None,
            "rmse_m": None,
            "mae_m": None,
            "nmad_m": None,
            "p05_m": None,
            "p25_m": None,
            "p75_m": None,
            "p95_m": None,
            "min_m": None,
            "max_m": None,
            "abs_p50_m": None,
            "abs_p68_m": None,
            "abs_p90_m": None,
            "abs_p95_m": None,
        }

    abs_arr = np.abs(arr)
    return {
        "n": int(arr.size),
        "mean_m": float(np.mean(arr)),
        "median_m": float(np.median(arr)),
        "std_m": float(np.std(arr)),
        "rmse_m": float(np.sqrt(np.mean(arr ** 2))),
        "mae_m": float(np.mean(abs_arr)),
        "nmad_m": robust_nmad(arr),
        "p05_m": float(np.percentile(arr, 5)),
        "p25_m": float(np.percentile(arr, 25)),
        "p75_m": float(np.percentile(arr, 75)),
        "p95_m": float(np.percentile(arr, 95)),
        "min_m": float(np.min(arr)),
        "max_m": float(np.max(arr)),
        "abs_p50_m": float(np.percentile(abs_arr, 50)),
        "abs_p68_m": float(np.percentile(abs_arr, 68)),
        "abs_p90_m": float(np.percentile(abs_arr, 90)),
        "abs_p95_m": float(np.percentile(abs_arr, 95)),
    }


def robust_stats(name: str, values: np.ndarray) -> ResidualStats:
    metrics = compute_distribution_metrics(values)
    return ResidualStats(
        name=name,
        n=int(metrics["n"]),
        mean_m=metrics["mean_m"],
        median_m=metrics["median_m"],
        std_m=metrics["std_m"],
        nmad_m=metrics["nmad_m"],
        p05_m=metrics["p05_m"],
        p25_m=metrics["p25_m"],
        p75_m=metrics["p75_m"],
        p95_m=metrics["p95_m"],
        min_m=metrics["min_m"],
        max_m=metrics["max_m"],
    )


def final_accuracy_stats_from_values(name: str, values: np.ndarray) -> FinalAccuracyStats:
    metrics = compute_distribution_metrics(values)
    return FinalAccuracyStats(
        residual_name=name,
        n=int(metrics["n"]),
        bias_m=metrics["mean_m"],
        median_m=metrics["median_m"],
        mean_m=metrics["mean_m"],
        std_m=metrics["std_m"],
        rmse_m=metrics["rmse_m"],
        mae_m=metrics["mae_m"],
        nmad_m=metrics["nmad_m"],
        p05_m=metrics["p05_m"],
        p25_m=metrics["p25_m"],
        p75_m=metrics["p75_m"],
        p95_m=metrics["p95_m"],
        abs_p50_m=metrics["abs_p50_m"],
        abs_p68_m=metrics["abs_p68_m"],
        abs_p90_m=metrics["abs_p90_m"],
        abs_p95_m=metrics["abs_p95_m"],
        min_m=metrics["min_m"],
        max_m=metrics["max_m"],
    )


def summarize_crs(source_name: str, crs: Optional[CRS], remarks: Optional[list[str]] = None) -> CRSInfo:
    remarks = list(remarks or [])
    if crs is None:
        return CRSInfo(
            source_name=source_name,
            crs_present=False,
            crs_name=None,
            authority=None,
            authority_code=None,
            is_geographic=None,
            is_projected=None,
            is_vertical=None,
            is_compound=None,
            axis_info=[],
            area_of_use=None,
            wkt=None,
            remarks=remarks + ["CRS is missing or could not be parsed."],
        )

    authority = crs.to_authority()
    axis_info = [f"{axis.name} | {axis.direction} | {axis.unit_name}" for axis in crs.axis_info]
    try:
        area_of_use = str(crs.area_of_use)
    except Exception:
        area_of_use = None

    return CRSInfo(
        source_name=source_name,
        crs_present=True,
        crs_name=crs.name,
        authority=authority[0] if authority else None,
        authority_code=authority[1] if authority else None,
        is_geographic=crs.is_geographic,
        is_projected=crs.is_projected,
        is_vertical=crs.is_vertical,
        is_compound=crs.is_compound,
        axis_info=axis_info,
        area_of_use=area_of_use,
        wkt=crs.to_wkt(),
        remarks=remarks,
    )


def split_horizontal_vertical_crs(crs: CRS) -> tuple[CRS, Optional[CRS], list[str]]:
    remarks: list[str] = []
    if crs.is_compound:
        horizontal = None
        vertical = None
        for sub in crs.sub_crs_list:
            if sub.is_vertical:
                vertical = sub
            elif sub.is_geographic or sub.is_projected:
                horizontal = sub
        if horizontal is None:
            raise RuntimeError(f"Could not identify horizontal component from compound CRS: {crs}")
        return horizontal.to_2d(), vertical, remarks

    if crs.is_vertical:
        raise RuntimeError(f"LAS CRS appears to be vertical-only, not usable as point cloud horizontal CRS: {crs}")

    remarks.append("CRS is not compound; vertical CRS is not explicitly available from LAS metadata.")
    return crs.to_2d(), None, remarks


def build_3dep_reference_frame(
    dep_crs: CRS,
    explicit_3dep_vertical_crs: Optional[CRS],
    explicit_3dep_vertical_crs_reason: Optional[str],
) -> tuple[CRS, Optional[CRS], ReferenceFrameInfo]:
    horizontal_crs, vertical_from_las, split_remarks = split_horizontal_vertical_crs(dep_crs)
    warnings_list = list(split_remarks)

    if vertical_from_las is not None:
        vertical_crs = vertical_from_las
        vertical_source = "LAS compound CRS"
        reference_frame_status = "reference_frame_vertical_verified_from_las"
    elif explicit_3dep_vertical_crs is not None:
        vertical_crs = explicit_3dep_vertical_crs
        vertical_source = "explicit verified configuration"
        reference_frame_status = "reference_frame_vertical_verified_from_configuration"
        if is_blank(explicit_3dep_vertical_crs_reason):
            warnings_list.append("Explicit 3DEP vertical CRS was provided without a non-empty reason.")
    else:
        vertical_crs = None
        vertical_source = "unknown"
        reference_frame_status = "vertical_crs_unknown"
        warnings_list.append(STRICT_VERTICAL_UNAVAILABLE_WARNING)

    horizontal_auth = horizontal_crs.to_authority()
    vertical_auth = vertical_crs.to_authority() if vertical_crs is not None else None
    info = ReferenceFrameInfo(
        horizontal_crs_name=horizontal_crs.name,
        horizontal_crs_authority=(
            f"{horizontal_auth[0]}:{horizontal_auth[1]}" if horizontal_auth else None
        ),
        vertical_crs_name=vertical_crs.name if vertical_crs is not None else None,
        vertical_crs_authority=(
            f"{vertical_auth[0]}:{vertical_auth[1]}" if vertical_auth else None
        ),
        vertical_crs_source=vertical_source,
        geoid_model_or_source=None,
        reference_frame_status=reference_frame_status,
        warnings=warnings_list,
    )
    return horizontal_crs, vertical_crs, info


def list_h5_datasets(h5: h5py.File) -> list[str]:
    names: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    h5.visititems(visitor)
    return names


def find_dataset_name(available: list[str], candidates: list[str]) -> str:
    available_set = set(available)
    for candidate in candidates:
        if candidate in available_set:
            return candidate

    basenames: dict[str, str] = {}
    for item in available:
        basenames.setdefault(Path(item).name, item)
    for candidate in candidates:
        if candidate in basenames:
            return basenames[candidate]

    raise KeyError(
        "Could not find any candidate dataset. "
        f"Candidates={candidates}. Available datasets include first 50: {available[:50]}"
    )


def read_casals_refh_points(h5_path: Path, snr_threshold: Optional[float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    log_status(f"Reading CASALS H5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        datasets = list_h5_datasets(f)
        log_status(f"Found {format_int(len(datasets))} H5 datasets; locating core fields.")

        lon_name = find_dataset_name(
            datasets,
            [
                "refh_longitude",
                "longitude",
                "lon",
                "ref_lon",
                "geolocation/refh_longitude",
                "geolocation/longitude",
            ],
        )
        lat_name = find_dataset_name(
            datasets,
            [
                "refh_latitude",
                "latitude",
                "lat",
                "ref_lat",
                "geolocation/refh_latitude",
                "geolocation/latitude",
            ],
        )
        z_name = find_dataset_name(
            datasets,
            [
                "refh",
                "refh_height",
                "height",
                "elevation",
                "ref_height",
                "geolocation/refh",
                "geolocation/height",
            ],
        )

        lon = np.asarray(f[lon_name][:], dtype=np.float64).reshape(-1)
        lat = np.asarray(f[lat_name][:], dtype=np.float64).reshape(-1)
        z = np.asarray(f[z_name][:], dtype=np.float64).reshape(-1)
        if not (lon.size == lat.size == z.size):
            raise RuntimeError(
                f"CASALS lon/lat/refh sizes do not match: lon={lon.size}, lat={lat.size}, z={z.size}"
            )
        log_status(
            "Loaded CASALS core arrays: "
            f"lon={format_int(lon.size)}, lat={format_int(lat.size)}, refh={format_int(z.size)}"
        )

        optional_candidates = {
            "geoid": ["geoid", "geolocation/geoid"],
            "geoid_free2mean": ["geoid_free2mean", "geolocation/geoid_free2mean"],
            "tide_earth": ["tide_earth", "geolocation/tide_earth"],
            "tide_earth_free2mean": ["tide_earth_free2mean", "geolocation/tide_earth_free2mean"],
            "tide_load": ["tide_load", "geolocation/tide_load"],
            "tide_ocean": ["tide_ocean", "geolocation/tide_ocean"],
            "tide_ocean_pole": ["tide_ocean_pole", "geolocation/tide_ocean_pole"],
            "tide_pole": ["tide_pole", "geolocation/tide_pole"],
            "range_bias_correction": ["range_bias_correction", "geolocation/range_bias_correction"],
            "refh_error": ["refh_error", "geolocation/refh_error"],
            "refh_snr": ["refh_snr", "snr", "signal_to_noise", "geolocation/refh_snr"],
            "refh_amp": ["refh_amp", "amp", "amplitude", "geolocation/refh_amp"],
            "track_num": ["track_num", "track", "track_index", "channel", "geolocation/track_num", "geolocation/track"],
            "sweep_num": ["sweep_num", "sweep", "row", "scan", "geolocation/sweep_num", "geolocation/sweep"],
            "good_snr": ["good_snr", "geolocation/good_snr"],
            "bg_mean": ["bg_mean", "background_mean", "geolocation/bg_mean"],
            "bg_std": ["bg_std", "background_std", "geolocation/bg_std"],
        }

        fields: dict[str, np.ndarray] = {}
        for out_name, candidates in optional_candidates.items():
            try:
                ds_name = find_dataset_name(datasets, candidates)
                arr = np.asarray(f[ds_name][:]).reshape(-1)
                if arr.size == z.size:
                    fields[out_name] = arr
            except Exception:
                continue

        attrs: dict[str, Any] = {}
        for key, value in f.attrs.items():
            try:
                if isinstance(value, bytes):
                    attrs[key] = value.decode("utf-8", errors="replace")
                elif isinstance(value, np.ndarray):
                    attrs[key] = value.tolist()
                elif hasattr(value, "item"):
                    attrs[key] = value.item()
                else:
                    attrs[key] = str(value)
            except Exception:
                attrs[key] = str(value)

    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(z)
    valid &= (lon >= -180.0) & (lon <= 180.0) & (lat >= -90.0) & (lat <= 90.0)

    df = pd.DataFrame(
        {
            "lon": lon[valid],
            "lat": lat[valid],
            "casals_refh_raw_m": z[valid],
        }
    )
    for key, arr in fields.items():
        df[key] = np.asarray(arr)[valid]

    if snr_threshold is not None:
        if "refh_snr" in df.columns:
            df = df[df["refh_snr"].astype(float) >= float(snr_threshold)].copy()
        elif "good_snr" in df.columns:
            df = df[df["good_snr"].astype(bool)].copy()
        else:
            warnings.warn(
                "SNR threshold requested but no refh_snr/good_snr field was found. No SNR filtering applied."
            )

    df = df.reset_index(drop=True)
    log_status(
        f"CASALS filtering complete: {format_int(len(df))} points remain after validity and SNR screening."
    )

    meta = {
        "h5_path": str(h5_path),
        "dataset_lon": lon_name,
        "dataset_lat": lat_name,
        "dataset_height": z_name,
        "h5_attrs": attrs,
        "available_dataset_count": len(datasets),
        "available_datasets_first_100": datasets[:100],
        "available_height_related_columns": [col for col in df.columns if "geoid" in col or "tide" in col or "refh" in col],
        "height_interpretation": (
            "CASALS refh is treated as raw reference height. It is assumed WGS84 "
            "ellipsoidal-like only for tested transformation paths, not as a verified fact."
        ),
        "n_after_basic_and_snr_filter": int(len(df)),
    }
    return df, meta


def read_las_tile_info(las_path: Path) -> tuple[Optional[CRS], LasTileInfo]:
    log_status(f"Scanning LAS metadata and class counts: {las_path.name}")
    with laspy.open(las_path) as reader:
        header = reader.header
        crs = header.parse_crs(prefer_wkt=True)
        classification_counts: dict[str, int] = {}
        ground_count = 0
        chunk_points_read = 0
        for chunk_index, points in enumerate(reader.chunk_iterator(2_000_000), start=1):
            cls = np.asarray(points.classification)
            chunk_points_read += int(len(cls))
            unique, counts = np.unique(cls, return_counts=True)
            for u, c in zip(unique, counts):
                key = str(int(u))
                classification_counts[key] = classification_counts.get(key, 0) + int(c)
            ground_count += int(np.sum(cls == 2))
            if chunk_index == 1 or chunk_index % 10 == 0:
                log_status(
                    f"  {las_path.name}: metadata chunk {chunk_index}, processed "
                    f"{format_int(chunk_points_read)} / {format_int(header.point_count)} points."
                )

        authority = crs.to_authority() if crs is not None else None
        info = LasTileInfo(
            path=str(las_path),
            file_name=las_path.name,
            point_count=int(header.point_count),
            point_format=str(header.point_format),
            version=str(header.version),
            scales=[float(x) for x in header.scales],
            offsets=[float(x) for x in header.offsets],
            mins=[float(x) for x in header.mins],
            maxs=[float(x) for x in header.maxs],
            vlr_count=len(header.vlrs),
            crs_parse_success=crs is not None,
            crs_name=crs.name if crs is not None else None,
            crs_authority=f"{authority[0]}:{authority[1]}" if authority else None,
            crs_wkt=crs.to_wkt() if crs is not None else None,
            ground_point_count=ground_count,
            classification_counts=classification_counts,
        )

    log_status(
        f"Finished LAS metadata scan for {las_path.name}: {format_int(info.ground_point_count)} class-2 ground points found."
    )
    return crs, info


def read_3dep_ground_points(
    las_paths: list[Path],
    max_points_per_tile: Optional[int] = None,
) -> tuple[pd.DataFrame, list[LasTileInfo], CRS]:
    all_rows: list[pd.DataFrame] = []
    tile_infos: list[LasTileInfo] = []
    first_crs: Optional[CRS] = None
    rng = np.random.default_rng(0)

    for tile_index, las_path in enumerate(las_paths, start=1):
        log_status(f"Loading 3DEP tile {tile_index}/{len(las_paths)}: {las_path.name}")
        crs, info = read_las_tile_info(las_path)
        tile_infos.append(info)

        if crs is None:
            raise RuntimeError(f"Cannot parse CRS from LAS/LAZ file: {las_path}")
        if first_crs is None:
            first_crs = crs
        elif not crs.equals(first_crs):
            raise RuntimeError(
                "Multiple 3DEP LAS files have different CRS definitions. "
                f"First={first_crs.to_string()}, current={crs.to_string()}, file={las_path}"
            )

        las = laspy.read(las_path)
        cls = np.asarray(las.classification)
        mask = cls == 2
        x = np.asarray(las.x[mask], dtype=np.float64)
        y = np.asarray(las.y[mask], dtype=np.float64)
        z = np.asarray(las.z[mask], dtype=np.float64)

        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x = x[finite]
        y = y[finite]
        z = z[finite]

        if max_points_per_tile is not None and len(x) > max_points_per_tile:
            idx = rng.choice(len(x), size=max_points_per_tile, replace=False)
            x = x[idx]
            y = y[idx]
            z = z[idx]
            log_status(f"  Applied per-tile sampling cap: kept {format_int(len(x))} ground points for {las_path.name}.")

        all_rows.append(
            pd.DataFrame(
                {
                    "x_3dep": x,
                    "y_3dep": y,
                    "z_3dep_raw_m": z,
                    "tile": las_path.name,
                }
            )
        )
        log_status(f"  Loaded {format_int(len(x))} finite class-2 ground points from {las_path.name}.")

    if first_crs is None:
        raise RuntimeError("No 3DEP LAS/LAZ CRS could be read.")
    if not all_rows:
        raise RuntimeError("No 3DEP ground points were loaded.")

    merged = pd.concat(all_rows, ignore_index=True)
    log_status(
        f"3DEP ground loading complete: {format_int(len(merged))} total ground points across "
        f"{format_int(len(tile_infos))} tile(s)."
    )
    return merged, tile_infos, first_crs


def lonlat_bounds_from_df(df: pd.DataFrame) -> tuple[float, float, float, float]:
    return (
        float(np.nanmin(df["lon"])),
        float(np.nanmin(df["lat"])),
        float(np.nanmax(df["lon"])),
        float(np.nanmax(df["lat"])),
    )


def sample_indices(n: int, max_n: int = 1000) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_n).astype(int)


def build_transformer_group(
    src_crs: CRS,
    dst_crs: CRS,
    bounds_lonlat: Optional[tuple[float, float, float, float]] = None,
) -> TransformerGroup:
    kwargs: dict[str, Any] = {"always_xy": True, "allow_ballpark": False}
    if bounds_lonlat is not None:
        west, south, east, north = bounds_lonlat
        kwargs["area_of_interest"] = AreaOfInterest(
            west_lon_degree=west,
            south_lat_degree=south,
            east_lon_degree=east,
            north_lat_degree=north,
        )
    try:
        return TransformerGroup(src_crs, dst_crs, **kwargs)
    except TypeError:
        kwargs.pop("area_of_interest", None)
        return TransformerGroup(src_crs, dst_crs, **kwargs)


def build_transformer(
    src_crs: CRS,
    dst_crs: CRS,
    bounds_lonlat: Optional[tuple[float, float, float, float]] = None,
) -> Transformer:
    kwargs: dict[str, Any] = {"always_xy": True, "allow_ballpark": False}
    if bounds_lonlat is not None:
        west, south, east, north = bounds_lonlat
        kwargs["area_of_interest"] = AreaOfInterest(
            west_lon_degree=west,
            south_lat_degree=south,
            east_lon_degree=east,
            north_lat_degree=north,
        )
    try:
        return Transformer.from_crs(src_crs, dst_crs, **kwargs)
    except TypeError:
        kwargs.pop("area_of_interest", None)
        return Transformer.from_crs(src_crs, dst_crs, **kwargs)


def collect_missing_grids(group: TransformerGroup) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for operation in getattr(group, "unavailable_operations", []):
        grids = getattr(operation, "grids", [])
        if not grids:
            rows.append({"operation_name": getattr(operation, "name", None), "reason": "operation_unavailable"})
            continue
        for grid in grids:
            rows.append(
                {
                    "operation_name": getattr(operation, "name", None),
                    "grid_name": getattr(grid, "short_name", None),
                    "full_name": getattr(grid, "full_name", None),
                    "package_name": getattr(grid, "package_name", None),
                    "url": getattr(grid, "url", None),
                    "available": getattr(grid, "available", None),
                }
            )
    return rows


def audit_transformer_group(
    name: str,
    src_crs: CRS,
    dst_crs: CRS,
    bounds_lonlat: Optional[tuple[float, float, float, float]] = None,
    sample_lon: Optional[np.ndarray] = None,
    sample_lat: Optional[np.ndarray] = None,
    sample_z: Optional[np.ndarray] = None,
    expect_vertical_change: bool = False,
) -> TransformAudit:
    warnings_list: list[str] = []
    group = build_transformer_group(src_crs, dst_crs, bounds_lonlat=bounds_lonlat)
    transformers = list(group.transformers)
    first = transformers[0] if transformers else None

    transformer_count = len(transformers)
    unavailable_count = len(getattr(group, "unavailable_operations", []))
    missing_grids = collect_missing_grids(group)
    best_available = getattr(group, "best_available", None)
    status = "ok"

    if transformer_count == 0:
        status = "no_transformer_available"
        warnings_list.append("No non-ballpark transformation path is available.")
    elif best_available is False:
        status = "best_transformation_unavailable"
        warnings_list.append("Best transformation is unavailable; required grids may be missing.")

    if missing_grids:
        warnings_list.append("Missing or unavailable grid-based operations were reported by pyproj.")

    description = None
    accuracy = None
    if first is not None:
        description = getattr(first, "description", None)
        raw_accuracy = getattr(first, "accuracy", None)
        if raw_accuracy not in (None, -1):
            accuracy = float(raw_accuracy)
        if description and "ballpark" in description.lower():
            warnings_list.append("First available transformation description contains 'ballpark'.")
            if status == "ok":
                status = "ballpark_operation_reported"

    z_change_median = None
    z_change_p05 = None
    z_change_p95 = None
    if expect_vertical_change and first is not None and sample_lon is not None and sample_lat is not None and sample_z is not None:
        try:
            _, _, z_out = first.transform(sample_lon, sample_lat, sample_z)
            z_out = np.asarray(z_out, dtype=np.float64)
            z_change = z_out - np.asarray(sample_z, dtype=np.float64)
            z_change = z_change[np.isfinite(z_change)]
            if z_change.size:
                z_change_median = float(np.median(z_change))
                z_change_p05 = float(np.percentile(z_change, 5))
                z_change_p95 = float(np.percentile(z_change, 95))
                if np.percentile(np.abs(z_change), 95) <= 0.01:
                    status = "z_passthrough_or_no_vertical_operation"
                    warnings_list.append("Vertical transformation appears to be a z passthrough or no-op.")
        except Exception as exc:
            warnings_list.append(f"Sample z audit failed: {exc}")
            if status == "ok":
                status = "z_audit_failed"

    return TransformAudit(
        name=name,
        source_crs=src_crs.to_string(),
        target_crs=dst_crs.to_string(),
        allow_ballpark=False,
        always_xy=True,
        best_available=best_available,
        transformer_count=transformer_count,
        unavailable_operations_count=unavailable_count,
        first_transformer_description=description,
        first_transformer_accuracy_m=accuracy,
        missing_grids=missing_grids,
        z_change_median_m=z_change_median,
        z_change_p05_m=z_change_p05,
        z_change_p95_m=z_change_p95,
        status=status,
        warnings=warnings_list,
    )


def project_casals_to_horizontal_crs(
    casals_df: pd.DataFrame,
    target_horizontal_crs: CRS,
) -> tuple[pd.DataFrame, TransformAudit]:
    log_status(
        f"Projecting {format_int(len(casals_df))} CASALS points into 3DEP horizontal CRS ({target_horizontal_crs.to_string()})."
    )
    bounds = lonlat_bounds_from_df(casals_df)
    audit = audit_transformer_group(
        name="casals_lonlat_to_3dep_horizontal",
        src_crs=CRS.from_epsg(4326),
        dst_crs=target_horizontal_crs,
        bounds_lonlat=bounds,
    )
    transformer = build_transformer(CRS.from_epsg(4326), target_horizontal_crs, bounds_lonlat=bounds)
    x, y = transformer.transform(
        casals_df["lon"].to_numpy(dtype=np.float64),
        casals_df["lat"].to_numpy(dtype=np.float64),
    )
    projected = casals_df.copy()
    projected["x_3dep_horizontal_m"] = np.asarray(x, dtype=np.float64)
    projected["y_3dep_horizontal_m"] = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(projected["x_3dep_horizontal_m"]) & np.isfinite(projected["y_3dep_horizontal_m"])
    projected = projected.loc[finite].reset_index(drop=True)
    log_status(f"Horizontal projection complete: {format_int(len(projected))} points retained.")
    return projected, audit


def filter_casals_ground_like_by_local_low_surface(
    casals_projected: pd.DataFrame,
    cell_size_m: float,
    max_above_local_low_m: float,
    min_points_per_cell: int,
    local_low_method: str = "min",
) -> pd.DataFrame:
    log_status(
        "Filtering CASALS ground-like candidates using local low surface: "
        f"method={local_low_method}, cell_size={cell_size_m} m, "
        f"max_above_local_low={max_above_local_low_m} m."
    )
    df = casals_projected.copy()
    df["cell_ix"] = np.floor(df["x_3dep_horizontal_m"] / float(cell_size_m)).astype(np.int64)
    df["cell_iy"] = np.floor(df["y_3dep_horizontal_m"] / float(cell_size_m)).astype(np.int64)

    grouped_series = df.groupby(["cell_ix", "cell_iy"], sort=False)["casals_refh_raw_m"]
    if local_low_method == "min":
        low_surface = grouped_series.min()
    elif local_low_method == "p05":
        low_surface = grouped_series.quantile(0.05)
    elif local_low_method == "p10":
        low_surface = grouped_series.quantile(0.10)
    else:
        raise ValueError(f"Unsupported local_low_method={local_low_method!r}")

    counts = grouped_series.count()
    grouped = pd.concat(
        [counts.rename("casals_local_cell_count"), low_surface.rename("casals_local_low_m")],
        axis=1,
    ).reset_index()

    df = df.merge(grouped, on=["cell_ix", "cell_iy"], how="left")
    df["casals_local_low_method"] = local_low_method
    df["casals_above_local_low_m"] = df["casals_refh_raw_m"] - df["casals_local_low_m"]
    keep = (
        (df["casals_local_cell_count"] >= int(min_points_per_cell))
        & (df["casals_above_local_low_m"] <= float(max_above_local_low_m))
    )
    df = df.loc[keep].copy().reset_index(drop=True)
    df["casals_ground_like"] = True
    log_status(
        f"CASALS ground-like filtering complete: {format_int(len(df))} / {format_int(len(casals_projected))} points retained."
    )
    return df


def _mode_string(values: np.ndarray) -> str:
    if values.size == 0:
        return "unknown"
    unique, counts = np.unique(values.astype(str), return_counts=True)
    return str(unique[np.argmax(counts)])


def match_3dep_ground_local_median(
    casals_ground_like: pd.DataFrame,
    dep_ground_df: pd.DataFrame,
    radius_m: float,
    min_neighbors: int,
) -> pd.DataFrame:
    log_status(
        "Matching CASALS ground-like points to local 3DEP ground neighborhoods: "
        f"{format_int(len(casals_ground_like))} queries, radius={radius_m} m."
    )
    tree = cKDTree(dep_ground_df[["x_3dep", "y_3dep"]].to_numpy(dtype=np.float64))
    query_xy = casals_ground_like[["x_3dep_horizontal_m", "y_3dep_horizontal_m"]].to_numpy(dtype=np.float64)
    neighbor_lists = tree.query_ball_point(query_xy, r=float(radius_m))

    rows: list[dict[str, Any]] = []
    dep_z = dep_ground_df["z_3dep_raw_m"].to_numpy(dtype=np.float64)
    dep_tile = dep_ground_df["tile"].to_numpy(dtype=object)
    total_queries = len(neighbor_lists)

    for i, neighbors in enumerate(neighbor_lists):
        if len(neighbors) < int(min_neighbors):
            if (i + 1) == 1 or (i + 1) % 100000 == 0 or (i + 1) == total_queries:
                log_status(
                    f"  Matching progress: processed {format_int(i + 1)} / {format_int(total_queries)} CASALS candidates."
                )
            continue

        z_local = dep_z[np.asarray(neighbors, dtype=int)]
        z_local = z_local[np.isfinite(z_local)]
        if z_local.size < int(min_neighbors):
            if (i + 1) == 1 or (i + 1) % 100000 == 0 or (i + 1) == total_queries:
                log_status(
                    f"  Matching progress: processed {format_int(i + 1)} / {format_int(total_queries)} CASALS candidates."
                )
            continue

        tile_local = dep_tile[np.asarray(neighbors, dtype=int)]
        row = casals_ground_like.iloc[i].to_dict()
        median = float(np.median(z_local))
        p05 = float(np.percentile(z_local, 5))
        p95 = float(np.percentile(z_local, 95))
        nmad = robust_nmad(z_local)
        row.update(
            {
                "z_3dep_ground_median_raw_m": median,
                "z_3dep_ground_nmad_m": nmad,
                "z_3dep_ground_p05_m": p05,
                "z_3dep_ground_p95_m": p95,
                "z_3dep_ground_p95_minus_p05_m": p95 - p05,
                "n_3dep_neighbors": int(z_local.size),
                "tile": _mode_string(np.asarray(tile_local)),
            }
        )
        rows.append(row)

        if (i + 1) == 1 or (i + 1) % 100000 == 0 or (i + 1) == total_queries:
            log_status(
                f"  Matching progress: processed {format_int(i + 1)} / {format_int(total_queries)} CASALS candidates."
            )

    matched = pd.DataFrame(rows)
    log_status(f"Local 3DEP ground matching complete: {format_int(len(matched))} matched points retained.")
    return matched


def build_target_compound_crs(horizontal_crs: CRS, vertical_crs: CRS) -> CRS:
    compound = CompoundCRS(
        name=f"{horizontal_crs.name} + {vertical_crs.name}",
        components=[horizontal_crs, vertical_crs],
    )
    return CRS.from_wkt(compound.to_wkt())


def add_h5_geoid_height_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    warnings_list: list[str] = []

    if "geoid" not in out.columns:
        warnings_list.append("CASALS H5 does not contain geoid. No H5 geoid candidates were created.")
        return out, warnings_list

    out["casals_z_refh_minus_geoid_m"] = out["casals_refh_raw_m"] - out["geoid"]
    out["casals_z_refh_plus_geoid_m"] = out["casals_refh_raw_m"] + out["geoid"]

    if "geoid_free2mean" in out.columns:
        out["casals_z_refh_minus_geoid_minus_free2mean_m"] = (
            out["casals_refh_raw_m"] - out["geoid"] - out["geoid_free2mean"]
        )
        out["casals_z_refh_minus_geoid_plus_free2mean_m"] = (
            out["casals_refh_raw_m"] - out["geoid"] + out["geoid_free2mean"]
        )
        out["casals_z_refh_plus_geoid_minus_free2mean_m"] = (
            out["casals_refh_raw_m"] + out["geoid"] - out["geoid_free2mean"]
        )
        out["casals_z_refh_plus_geoid_plus_free2mean_m"] = (
            out["casals_refh_raw_m"] + out["geoid"] + out["geoid_free2mean"]
        )
    else:
        warnings_list.append(
            "CASALS H5 does not contain geoid_free2mean. Candidates requiring geoid_free2mean were not created."
        )
    return out, warnings_list


def formula_for_candidate_column(candidate_column: str) -> str:
    for formula, column in FORMULA_TO_COLUMN.items():
        if column == candidate_column:
            return formula
    return candidate_column


def residual_column_name_from_candidate(candidate_column: str) -> str:
    suffix = candidate_column
    if suffix.startswith("casals_z_"):
        suffix = suffix[len("casals_z_"):]
    return f"candidate_residual_{suffix}"


def evaluate_height_conversion_candidates(
    matched: pd.DataFrame,
    candidate_columns: list[str],
    reference_z_col: str = "z_3dep_ground_median_raw_m",
) -> tuple[pd.DataFrame, list[HeightConversionCandidateStats]]:
    out = matched.copy()
    stats_rows: list[HeightConversionCandidateStats] = []

    for candidate_col in candidate_columns:
        if candidate_col not in out.columns:
            continue
        residual_col = residual_column_name_from_candidate(candidate_col)
        out[residual_col] = out[candidate_col] - out[reference_z_col]
        metrics = compute_distribution_metrics(out[residual_col].to_numpy(dtype=np.float64))
        stats_rows.append(
            HeightConversionCandidateStats(
                candidate_name=candidate_col,
                formula=formula_for_candidate_column(candidate_col),
                used_as_final=False,
                n=int(metrics["n"]),
                mean_m=metrics["mean_m"],
                median_m=metrics["median_m"],
                std_m=metrics["std_m"],
                rmse_m=metrics["rmse_m"],
                mae_m=metrics["mae_m"],
                nmad_m=metrics["nmad_m"],
                p05_m=metrics["p05_m"],
                p25_m=metrics["p25_m"],
                p75_m=metrics["p75_m"],
                p95_m=metrics["p95_m"],
                min_m=metrics["min_m"],
                max_m=metrics["max_m"],
                interpretation="Diagnostic only; not a final reference-frame residual unless explicitly selected and verified.",
            )
        )

    return out, stats_rows


def convert_casals_to_target_compound_crs_with_pyproj(
    df: pd.DataFrame,
    target_horizontal_crs: CRS,
    target_vertical_crs: CRS,
    bounds_lonlat: tuple[float, float, float, float],
) -> tuple[pd.DataFrame, TransformAudit]:
    log_status(
        f"Running pyproj / PROJ compound conversion for {format_int(len(df))} points into the 3DEP reference frame."
    )
    dst_crs = build_target_compound_crs(target_horizontal_crs, target_vertical_crs)
    idx = sample_indices(len(df), max_n=1000)
    sample_lon = df["lon"].to_numpy(dtype=np.float64)[idx]
    sample_lat = df["lat"].to_numpy(dtype=np.float64)[idx]
    sample_z = df["casals_refh_raw_m"].to_numpy(dtype=np.float64)[idx]

    audit = audit_transformer_group(
        name="casals_4979_to_3dep_reference_pyproj",
        src_crs=CRS.from_epsg(4979),
        dst_crs=dst_crs,
        bounds_lonlat=bounds_lonlat,
        sample_lon=sample_lon,
        sample_lat=sample_lat,
        sample_z=sample_z,
        expect_vertical_change=True,
    )

    out = df.copy()
    try:
        transformer = build_transformer(CRS.from_epsg(4979), dst_crs, bounds_lonlat=bounds_lonlat)
        x_out, y_out, z_out = transformer.transform(
            out["lon"].to_numpy(dtype=np.float64),
            out["lat"].to_numpy(dtype=np.float64),
            out["casals_refh_raw_m"].to_numpy(dtype=np.float64),
        )
        out["x_in_3dep_reference_m"] = np.asarray(x_out, dtype=np.float64)
        out["y_in_3dep_reference_m"] = np.asarray(y_out, dtype=np.float64)
        out["casals_z_in_3dep_reference_proj_m"] = np.asarray(z_out, dtype=np.float64)
        if audit.status == "ok":
            audit.status = "reference_frame_transform_pyproj_ok"
    except Exception as exc:
        audit.status = "reference_frame_transform_pyproj_failed"
        audit.warnings.append(f"pyproj compound transformation failed: {exc}")

    return out, audit


def enforce_pyproj_reference_audit(
    audit: TransformAudit,
    strict_reference_mode: bool,
) -> None:
    if not strict_reference_mode:
        return
    if audit.best_available is False:
        raise RuntimeError("pyproj reports best_available=False for the CASALS-to-3DEP reference transform.")
    if audit.missing_grids:
        raise RuntimeError("pyproj reports missing grids for the CASALS-to-3DEP reference transform.")
    if audit.status in {
        "no_transformer_available",
        "best_transformation_unavailable",
        "ballpark_operation_reported",
        "z_passthrough_or_no_vertical_operation",
        "z_audit_failed",
        "reference_frame_transform_pyproj_failed",
    }:
        raise RuntimeError(f"CASALS-to-3DEP reference transform failed strict audit: {audit.status}")
    if audit.first_transformer_accuracy_m is None:
        raise RuntimeError("Transformer accuracy is unknown; strict reference mode will not accept the transform.")


def transformation_crosscheck(
    casals_df: pd.DataFrame,
    src_crs: CRS,
    dst_crs: CRS,
    output_csv: Path,
    output_json: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    log_status(
        f"Running pyproj / GDAL reference-frame cross-check on {format_int(min(len(casals_df), 200))} sample points."
    )
    idx = sample_indices(len(casals_df), max_n=200)
    lon = casals_df["lon"].to_numpy(dtype=np.float64)[idx]
    lat = casals_df["lat"].to_numpy(dtype=np.float64)[idx]
    z = casals_df["casals_refh_raw_m"].to_numpy(dtype=np.float64)[idx]
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "cwd": os.getcwd(),
        "n_samples": int(len(idx)),
        "pyproj_status": "not_run",
        "gdal_osr_status": "not_run",
        "gdaltransform_status": "not_run",
    }

    pyproj_transformer = build_transformer(src_crs, dst_crs, bounds_lonlat=lonlat_bounds_from_df(casals_df))
    x_ref, y_ref, z_ref = pyproj_transformer.transform(lon, lat, z)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)
    z_ref = np.asarray(z_ref, dtype=np.float64)
    meta["pyproj_status"] = "ok"

    rows.append(
        {
            "method": "pyproj",
            "status": "ok",
            "n_samples": int(len(idx)),
            "median_abs_horizontal_diff_m": 0.0,
            "p95_abs_horizontal_diff_m": 0.0,
            "median_abs_vertical_diff_m": 0.0,
            "p95_abs_vertical_diff_m": 0.0,
            "max_abs_vertical_diff_m": 0.0,
        }
    )

    def compare_against_pyproj(method_name: str, x: np.ndarray, y: np.ndarray, z_values: np.ndarray, status: str) -> dict[str, Any]:
        dx = np.asarray(x, dtype=np.float64) - x_ref
        dy = np.asarray(y, dtype=np.float64) - y_ref
        dz = np.asarray(z_values, dtype=np.float64) - z_ref
        dxy = np.sqrt(dx * dx + dy * dy)
        abs_dz = np.abs(dz)
        return {
            "method": method_name,
            "status": status,
            "n_samples": int(len(x_ref)),
            "median_abs_horizontal_diff_m": float(np.median(np.abs(dxy))),
            "p95_abs_horizontal_diff_m": float(np.percentile(np.abs(dxy), 95)),
            "median_abs_vertical_diff_m": float(np.median(abs_dz)),
            "p95_abs_vertical_diff_m": float(np.percentile(abs_dz, 95)),
            "max_abs_vertical_diff_m": float(np.max(abs_dz)),
        }

    if HAS_GDAL_OSR and osr is not None:
        try:
            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(src_crs.to_wkt())
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromWkt(dst_crs.to_wkt())
            if hasattr(src_srs, "SetAxisMappingStrategy"):
                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            transform = osr.CoordinateTransformation(src_srs, dst_srs)
            xyz = np.array(
                [transform.TransformPoint(float(lo), float(la), float(zz)) for lo, la, zz in zip(lon, lat, z)]
            )
            rows.append(compare_against_pyproj("gdal_osr", xyz[:, 0], xyz[:, 1], xyz[:, 2], "ok"))
            meta["gdal_osr_status"] = "ok"
            log_status("  GDAL OSR cross-check complete.")
        except Exception as exc:
            meta["gdal_osr_status"] = "gdal_osr_failed"
            meta["gdal_osr_error"] = str(exc)
            log_status(f"  GDAL OSR cross-check failed: {exc}")
    else:
        meta["gdal_osr_status"] = "gdal_osr_not_available"
        log_status("  GDAL OSR bindings not available; skipping OSR cross-check.")

    gdaltransform_exe = shutil.which("gdaltransform")
    if gdaltransform_exe is None:
        meta["gdaltransform_status"] = "gdaltransform_not_found"
        log_status("  gdaltransform executable not found; skipping CLI cross-check.")
    else:
        try:
            cmd = [gdaltransform_exe, "-s_srs", src_crs.to_wkt(), "-t_srs", dst_crs.to_wkt()]
            stdin_text = "".join(f"{lo} {la} {zz}\n" for lo, la, zz in zip(lon, lat, z))
            proc = subprocess.run(
                cmd,
                input=stdin_text,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "gdaltransform failed")
            parsed_rows: list[list[float]] = []
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    raise RuntimeError(f"Unexpected gdaltransform output line: {line}")
                parsed_rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            xyz = np.asarray(parsed_rows, dtype=np.float64)
            if xyz.shape[0] != len(idx):
                raise RuntimeError(f"gdaltransform returned {xyz.shape[0]} rows for {len(idx)} inputs.")
            rows.append(compare_against_pyproj("gdaltransform", xyz[:, 0], xyz[:, 1], xyz[:, 2], "ok"))
            meta["gdaltransform_status"] = "ok"
            log_status("  gdaltransform cross-check complete.")
        except Exception as exc:
            meta["gdaltransform_status"] = "gdaltransform_failed"
            meta["gdaltransform_error"] = str(exc)
            log_status(f"  gdaltransform cross-check failed: {exc}")

    crosscheck_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    crosscheck_df.to_csv(output_csv, index=False)
    meta["output_csv"] = str(output_csv)
    meta["output_json"] = str(output_json)
    meta["rows_written"] = int(len(crosscheck_df))
    write_json(output_json, {"meta": meta, "rows": crosscheck_df.to_dict(orient="records")})
    return crosscheck_df, meta


def enforce_crosscheck_thresholds(
    crosscheck_df: pd.DataFrame,
    crosscheck_meta: dict[str, Any],
    strict_reference_mode: bool,
    max_pyproj_gdal_vertical_disagreement_m: float,
    max_pyproj_gdal_horizontal_disagreement_m: float,
    check_gdal_osr: bool,
    check_gdal_cli: bool,
) -> None:
    if not strict_reference_mode:
        return
    method_flags = {
        "gdal_osr": check_gdal_osr,
        "gdaltransform": check_gdal_cli,
    }
    for method_name in ("gdal_osr", "gdaltransform"):
        if not method_flags[method_name]:
            continue
        row = crosscheck_df.loc[crosscheck_df["method"] == method_name]
        if row.empty:
            continue
        h = float(row["p95_abs_horizontal_diff_m"].iloc[0])
        v = float(row["p95_abs_vertical_diff_m"].iloc[0])
        if h > float(max_pyproj_gdal_horizontal_disagreement_m):
            raise RuntimeError(
                f"{method_name} horizontal disagreement exceeds threshold: {h} m > {max_pyproj_gdal_horizontal_disagreement_m} m."
            )
        if v > float(max_pyproj_gdal_vertical_disagreement_m):
            raise RuntimeError(
                f"{method_name} vertical disagreement exceeds threshold: {v} m > {max_pyproj_gdal_vertical_disagreement_m} m."
            )

    if check_gdal_osr and crosscheck_meta.get("gdal_osr_status") not in {"ok", "gdal_osr_not_available"}:
        raise RuntimeError(f"GDAL OSR cross-check failed: {crosscheck_meta.get('gdal_osr_error')}")
    if check_gdal_cli and crosscheck_meta.get("gdaltransform_status") not in {"ok", "gdaltransform_not_found"}:
        raise RuntimeError(f"gdaltransform cross-check failed: {crosscheck_meta.get('gdaltransform_error')}")


def select_casals_reference_height(
    matched: pd.DataFrame,
    mode: str,
    h5_geoid_formula: str,
    selected_h5_geoid_formula_for_final: Optional[str],
    target_vertical_crs: Optional[CRS],
    strict_reference_mode: bool,
    compare_h5_proj_agreement_threshold_m: float = 0.02,
) -> tuple[pd.DataFrame, str, list[str]]:
    out = matched.copy()
    warnings_list: list[str] = []

    if target_vertical_crs is None:
        warnings_list.append(STRICT_VERTICAL_UNAVAILABLE_WARNING)
        if strict_reference_mode:
            raise RuntimeError(STRICT_VERTICAL_UNAVAILABLE_WARNING)
        return out, "reference_frame_not_selected_vertical_unknown", warnings_list

    if mode == "raw_only_diagnostic":
        warnings_list.append("CASALS height conversion mode is raw_only_diagnostic. No final reference-frame height was created.")
        if strict_reference_mode:
            raise RuntimeError("strict_reference_mode=True but casals_height_conversion_mode='raw_only_diagnostic'.")
        return out, "raw_only_diagnostic", warnings_list

    if mode == "h5_geoid":
        if h5_geoid_formula == "auto_diagnostic_only" and selected_h5_geoid_formula_for_final is None:
            warnings_list.append(H5_DIAGNOSTIC_ONLY_MESSAGE)
            if strict_reference_mode:
                raise RuntimeError(H5_DIAGNOSTIC_ONLY_MESSAGE)
            return out, "h5_geoid_diagnostic_only", warnings_list

        selected_formula = selected_h5_geoid_formula_for_final or h5_geoid_formula
        candidate_col = FORMULA_TO_COLUMN.get(selected_formula)
        if candidate_col is None:
            raise ValueError(f"Unsupported H5 geoid formula selection: {selected_formula}")
        if candidate_col not in out.columns:
            raise RuntimeError(f"Selected H5 geoid candidate column is missing: {candidate_col}")
        out["casals_z_in_3dep_reference_m"] = out[candidate_col]
        out["casals_height_conversion_status"] = "h5_geoid_formula_selected"
        out["casals_height_conversion_formula"] = selected_formula
        return out, "h5_geoid_formula_selected", warnings_list

    if mode == "proj_grid":
        if "casals_z_in_3dep_reference_proj_m" not in out.columns:
            raise RuntimeError("PROJ grid conversion result is missing. Cannot create final reference-frame height.")
        out["casals_z_in_3dep_reference_m"] = out["casals_z_in_3dep_reference_proj_m"]
        out["casals_height_conversion_status"] = "proj_grid_selected"
        out["casals_height_conversion_formula"] = "pyproj / PROJ compound transform from EPSG:4979 to 3DEP compound CRS"
        return out, "proj_grid_selected", warnings_list

    if mode == "compare_h5_geoid_and_proj_grid":
        selected_formula = selected_h5_geoid_formula_for_final or h5_geoid_formula
        candidate_col = FORMULA_TO_COLUMN.get(selected_formula)
        if candidate_col is None:
            raise ValueError(
                "compare_h5_geoid_and_proj_grid requires a selected H5 geoid formula for comparison."
            )
        if candidate_col not in out.columns:
            raise RuntimeError(f"Selected H5 geoid candidate column is missing: {candidate_col}")
        if "casals_z_in_3dep_reference_proj_m" not in out.columns:
            raise RuntimeError("PROJ grid conversion result is missing for compare_h5_geoid_and_proj_grid.")

        out["casals_z_in_3dep_reference_h5_m"] = out[candidate_col]
        out["casals_z_in_3dep_reference_proj_m"] = out["casals_z_in_3dep_reference_proj_m"]
        out["casals_h5_minus_proj_reference_height_m"] = (
            out["casals_z_in_3dep_reference_h5_m"] - out["casals_z_in_3dep_reference_proj_m"]
        )
        abs_diff = np.abs(out["casals_h5_minus_proj_reference_height_m"].to_numpy(dtype=np.float64))
        p95_abs_diff = float(np.percentile(abs_diff[np.isfinite(abs_diff)], 95)) if np.isfinite(abs_diff).any() else math.inf

        if p95_abs_diff <= float(compare_h5_proj_agreement_threshold_m):
            out["casals_z_in_3dep_reference_m"] = out["casals_z_in_3dep_reference_proj_m"]
            out["casals_height_conversion_status"] = "compare_h5_geoid_and_proj_grid_agree_use_proj"
            out["casals_height_conversion_formula"] = (
                "pyproj / PROJ compound transform accepted after agreement with selected H5 geoid formula"
            )
            return out, "compare_agree_use_proj", warnings_list

        warnings_list.append(
            "H5 geoid and PROJ-grid reference heights do not agree within the configured threshold."
        )
        if strict_reference_mode:
            raise RuntimeError(
                "H5 geoid and PROJ-grid reference heights do not agree within threshold."
            )
        return out, "compare_disagree_no_final_selection", warnings_list

    raise ValueError(f"Unsupported casals_height_conversion_mode={mode!r}")


def compute_diagnostic_residuals(matched: pd.DataFrame) -> tuple[pd.DataFrame, list[ResidualStats]]:
    out = matched.copy()
    out["raw_mixed_datum_difference_m"] = (
        out["casals_refh_raw_m"] - out["z_3dep_ground_median_raw_m"]
    )
    global_median = float(np.median(out["raw_mixed_datum_difference_m"].to_numpy(dtype=np.float64)))
    out["empirical_median_removed_residual_m"] = out["raw_mixed_datum_difference_m"] - global_median

    stats_rows = [
        robust_stats("raw_mixed_datum_difference_m", out["raw_mixed_datum_difference_m"].to_numpy(dtype=np.float64)),
        robust_stats(
            "empirical_median_removed_residual_m",
            out["empirical_median_removed_residual_m"].to_numpy(dtype=np.float64),
        ),
    ]
    return out, stats_rows


def compute_final_reference_frame_accuracy(
    matched: pd.DataFrame,
    strict_reference_mode: bool,
) -> tuple[pd.DataFrame, Optional[FinalAccuracyStats]]:
    out = matched.copy()
    if "casals_z_in_3dep_reference_m" not in out.columns:
        if strict_reference_mode:
            raise RuntimeError(FINAL_NOT_COMPUTED_MESSAGE)
        return out, None

    out["z_3dep_ground_median_reference_m"] = out["z_3dep_ground_median_raw_m"]
    out["residual_casals_minus_3dep_reference_m"] = (
        out["casals_z_in_3dep_reference_m"] - out["z_3dep_ground_median_reference_m"]
    )
    stats = final_accuracy_stats_from_values(
        "residual_casals_minus_3dep_reference_m",
        out["residual_casals_minus_3dep_reference_m"].to_numpy(dtype=np.float64),
    )
    return out, stats


def mark_outliers(matched: pd.DataFrame) -> pd.DataFrame:
    out = matched.copy()
    if "residual_casals_minus_3dep_reference_m" not in out.columns:
        return out
    out["abs_residual_reference_m"] = np.abs(out["residual_casals_minus_3dep_reference_m"])
    out["is_outlier_abs_gt_0p5m"] = out["abs_residual_reference_m"] > 0.5
    out["is_outlier_abs_gt_1m"] = out["abs_residual_reference_m"] > 1.0
    out["is_outlier_abs_gt_2m"] = out["abs_residual_reference_m"] > 2.0
    out["residual_sign"] = np.where(
        out["residual_casals_minus_3dep_reference_m"] > 0,
        "positive",
        np.where(out["residual_casals_minus_3dep_reference_m"] < 0, "negative", "zero"),
    )
    return out


def export_outlier_csvs(matched: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    counts = {"abs_gt_0p5m": 0, "abs_gt_1m": 0, "abs_gt_2m": 0}
    if "abs_residual_reference_m" not in matched.columns:
        return counts

    export_cols = [
        "lon",
        "lat",
        "x_3dep_horizontal_m",
        "y_3dep_horizontal_m",
        "x_in_3dep_reference_m",
        "y_in_3dep_reference_m",
        "casals_refh_raw_m",
        "casals_z_in_3dep_reference_m",
        "z_3dep_ground_median_reference_m",
        "residual_casals_minus_3dep_reference_m",
        "abs_residual_reference_m",
        "residual_sign",
        "track_num",
        "sweep_num",
        "refh_snr",
        "refh_amp",
        "z_3dep_ground_nmad_m",
        "z_3dep_ground_p95_minus_p05_m",
        "n_3dep_neighbors",
        "tile",
    ]
    export_cols = [col for col in export_cols if col in matched.columns]

    for threshold, suffix, key in [
        (0.5, "0p5m", "abs_gt_0p5m"),
        (1.0, "1m", "abs_gt_1m"),
        (2.0, "2m", "abs_gt_2m"),
    ]:
        subset = matched.loc[matched["abs_residual_reference_m"] > threshold, export_cols].copy()
        counts[key] = int(len(subset))
        subset.to_csv(output_dir / f"outliers_reference_frame_abs_gt_{suffix}.csv", index=False)
    return counts


def per_tile_residual_summary(residual_df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "tile" not in residual_df.columns:
        return pd.DataFrame(rows)

    for tile_name, group in residual_df.groupby("tile", sort=True):
        for col in value_cols:
            if col not in group.columns:
                continue
            st = robust_stats(col, group[col].to_numpy(dtype=np.float64))
            rows.append(
                {
                    "tile": tile_name,
                    "residual_name": st.name,
                    "n": st.n,
                    "mean_m": st.mean_m,
                    "median_m": st.median_m,
                    "std_m": st.std_m,
                    "nmad_m": st.nmad_m,
                    "p05_m": st.p05_m,
                    "p25_m": st.p25_m,
                    "p75_m": st.p75_m,
                    "p95_m": st.p95_m,
                    "min_m": st.min_m,
                    "max_m": st.max_m,
                }
            )
    return pd.DataFrame(rows)


def plot_hist_reference_residual(matched: pd.DataFrame, path: Path) -> None:
    if not HAS_MATPLOTLIB or "residual_casals_minus_3dep_reference_m" not in matched.columns:
        return
    values = matched["residual_casals_minus_3dep_reference_m"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=120, color="0.3")
    plt.xlabel("residual_casals_minus_3dep_reference_m")
    plt.ylabel("count")
    plt.title("CASALS minus 3DEP reference residual")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_hist_height_conversion_candidates(matched: pd.DataFrame, candidate_stats: list[HeightConversionCandidateStats], path: Path) -> None:
    if not HAS_MATPLOTLIB or not candidate_stats:
        return
    plt.figure(figsize=(9, 6))
    used_any = False
    for row in candidate_stats:
        residual_col = residual_column_name_from_candidate(row.candidate_name)
        if residual_col not in matched.columns:
            continue
        values = matched[residual_col].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        plt.hist(values, bins=80, histtype="step", linewidth=1.2, label=row.candidate_name)
        used_any = True
    if not used_any:
        plt.close()
        return
    plt.xlabel("candidate residual (m)")
    plt.ylabel("count")
    plt.title("H5 height-conversion candidate residuals")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_scatter_raw_vs_geoid(matched: pd.DataFrame, path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    if "geoid" not in matched.columns or "raw_mixed_datum_difference_m" not in matched.columns:
        return
    x = matched["geoid"].to_numpy(dtype=np.float64)
    y = matched["raw_mixed_datum_difference_m"].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return
    plt.figure(figsize=(7, 6))
    plt.scatter(x[finite], y[finite], s=2, alpha=0.4, linewidths=0)
    plt.xlabel("geoid")
    plt.ylabel("raw_mixed_datum_difference_m")
    plt.title("Raw mixed-datum difference vs H5 geoid")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_map_reference_residual_xy(matched: pd.DataFrame, path: Path) -> None:
    if not HAS_MATPLOTLIB or "residual_casals_minus_3dep_reference_m" not in matched.columns:
        return
    x_col = "x_in_3dep_reference_m" if "x_in_3dep_reference_m" in matched.columns else "x_3dep_horizontal_m"
    y_col = "y_in_3dep_reference_m" if "y_in_3dep_reference_m" in matched.columns else "y_3dep_horizontal_m"
    finite = (
        np.isfinite(matched[x_col].to_numpy(dtype=np.float64))
        & np.isfinite(matched[y_col].to_numpy(dtype=np.float64))
        & np.isfinite(matched["residual_casals_minus_3dep_reference_m"].to_numpy(dtype=np.float64))
    )
    if not finite.any():
        return
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        matched.loc[finite, x_col],
        matched.loc[finite, y_col],
        c=matched.loc[finite, "residual_casals_minus_3dep_reference_m"],
        s=2,
        cmap="coolwarm",
        linewidths=0,
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Reference-frame residual map")
    plt.colorbar(sc, label="residual_casals_minus_3dep_reference_m")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_map_outliers_reference_xy(matched: pd.DataFrame, path: Path) -> None:
    if not HAS_MATPLOTLIB or "abs_residual_reference_m" not in matched.columns:
        return
    x_col = "x_in_3dep_reference_m" if "x_in_3dep_reference_m" in matched.columns else "x_3dep_horizontal_m"
    y_col = "y_in_3dep_reference_m" if "y_in_3dep_reference_m" in matched.columns else "y_3dep_horizontal_m"
    finite = np.isfinite(matched[x_col].to_numpy(dtype=np.float64)) & np.isfinite(matched[y_col].to_numpy(dtype=np.float64))
    if not finite.any():
        return
    outlier_mask = matched["abs_residual_reference_m"] > 1.0
    plt.figure(figsize=(8, 6))
    plt.scatter(matched.loc[finite, x_col], matched.loc[finite, y_col], s=1, color="0.8", linewidths=0)
    if outlier_mask.any():
        plt.scatter(
            matched.loc[outlier_mask & finite, x_col],
            matched.loc[outlier_mask & finite, y_col],
            s=4,
            color="red",
            linewidths=0,
        )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Reference-frame outlier map")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_reference_frame_sensitivity_analysis(
    casals_projected: pd.DataFrame,
    dep_ground_df: pd.DataFrame,
    target_vertical_crs: Optional[CRS],
    casals_local_cell_size_m: float,
    casals_max_above_local_low_m: float,
    casals_min_points_per_cell: int,
    match_min_3dep_neighbors: int,
    max_3dep_ground_p95_minus_p05_m: float,
    casals_height_conversion_mode: str,
    h5_geoid_formula: str,
    selected_h5_geoid_formula_for_final: Optional[str],
    strict_reference_mode: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for match_radius_m in [0.5, 1.0, 2.0, 3.0]:
        for max_3dep_ground_nmad_m in [0.15, 0.30, 0.50]:
            for casals_local_low_method in ["min", "p05", "p10"]:
                try:
                    casals_ground_like = filter_casals_ground_like_by_local_low_surface(
                        casals_projected,
                        cell_size_m=casals_local_cell_size_m,
                        max_above_local_low_m=casals_max_above_local_low_m,
                        min_points_per_cell=casals_min_points_per_cell,
                        local_low_method=casals_local_low_method,
                    )
                    matched = match_3dep_ground_local_median(
                        casals_ground_like,
                        dep_ground_df,
                        radius_m=match_radius_m,
                        min_neighbors=match_min_3dep_neighbors,
                    )
                    matched = matched[
                        (matched["z_3dep_ground_nmad_m"] <= float(max_3dep_ground_nmad_m))
                        & (matched["z_3dep_ground_p95_minus_p05_m"] <= float(max_3dep_ground_p95_minus_p05_m))
                    ].copy()
                    matched = matched.reset_index(drop=True)
                    n_matched = int(len(matched))
                    n_final = 0
                    final_stats = None
                    status = "diagnostic_only"

                    if n_matched > 0:
                        matched, _ = compute_diagnostic_residuals(matched)
                        matched, _ = add_h5_geoid_height_candidates(matched)
                        candidate_cols = [col for col in FORMULA_TO_COLUMN.values() if col in matched.columns]
                        matched, _ = evaluate_height_conversion_candidates(matched, candidate_cols)
                        matched, _, _ = select_casals_reference_height(
                            matched=matched,
                            mode=casals_height_conversion_mode,
                            h5_geoid_formula=h5_geoid_formula,
                            selected_h5_geoid_formula_for_final=selected_h5_geoid_formula_for_final,
                            target_vertical_crs=target_vertical_crs,
                            strict_reference_mode=False if not strict_reference_mode else False,
                        )
                        matched, final_stats = compute_final_reference_frame_accuracy(matched, strict_reference_mode=False)
                        if final_stats is not None:
                            matched = mark_outliers(matched)
                            n_final = int(final_stats.n)
                            status = "final_available"
                        else:
                            status = "no_final_reference_height"

                    rows.append(
                        {
                            "match_radius_m": match_radius_m,
                            "max_3dep_ground_nmad_m": max_3dep_ground_nmad_m,
                            "casals_local_low_method": casals_local_low_method,
                            "n_matched": n_matched,
                            "n_final": n_final,
                            "final_bias_m": None if final_stats is None else final_stats.bias_m,
                            "final_median_m": None if final_stats is None else final_stats.median_m,
                            "final_rmse_m": None if final_stats is None else final_stats.rmse_m,
                            "final_nmad_m": None if final_stats is None else final_stats.nmad_m,
                            "final_p05_m": None if final_stats is None else final_stats.p05_m,
                            "final_p95_m": None if final_stats is None else final_stats.p95_m,
                            "outlier_abs_gt_0p5m_fraction": None if final_stats is None or n_final == 0 else float(np.mean(matched["abs_residual_reference_m"] > 0.5)),
                            "outlier_abs_gt_1m_fraction": None if final_stats is None or n_final == 0 else float(np.mean(matched["abs_residual_reference_m"] > 1.0)),
                            "outlier_abs_gt_2m_fraction": None if final_stats is None or n_final == 0 else float(np.mean(matched["abs_residual_reference_m"] > 2.0)),
                            "status": status,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "match_radius_m": match_radius_m,
                            "max_3dep_ground_nmad_m": max_3dep_ground_nmad_m,
                            "casals_local_low_method": casals_local_low_method,
                            "n_matched": 0,
                            "n_final": 0,
                            "final_bias_m": None,
                            "final_median_m": None,
                            "final_rmse_m": None,
                            "final_nmad_m": None,
                            "final_p05_m": None,
                            "final_p95_m": None,
                            "outlier_abs_gt_0p5m_fraction": None,
                            "outlier_abs_gt_1m_fraction": None,
                            "outlier_abs_gt_2m_fraction": None,
                            "status": f"failed:{exc}",
                        }
                    )
    return pd.DataFrame(rows)


def reference_frame_info_to_dict(reference_frame_info: ReferenceFrameInfo) -> dict[str, Any]:
    return asdict(reference_frame_info)


def write_diagnosis_report(
    path: Path,
    h5_path: Path,
    las_paths: list[Path],
    casals_meta: dict[str, Any],
    las_infos: list[LasTileInfo],
    reference_frame_info: ReferenceFrameInfo,
    crs_infos: list[CRSInfo],
    transform_audits: list[TransformAudit],
    crosscheck_meta: dict[str, Any],
    crosscheck_df: pd.DataFrame,
    candidate_stats: list[HeightConversionCandidateStats],
    diagnostic_residual_stats: list[ResidualStats],
    final_accuracy_stats: Optional[FinalAccuracyStats],
    sensitivity_df: pd.DataFrame,
    matched_count: int,
    final_selection_status: str,
    final_selection_warnings: list[str],
    strict_block_reason: Optional[str],
    outlier_counts: dict[str, int],
    config: dict[str, Any],
    user_selected_formula_warning: bool,
) -> None:
    lines: list[str] = []
    lines.append("# CASALS–3DEP Reference-Frame Accuracy Evaluation")
    lines.append("")
    lines[0] = "# CASALS\u20133DEP Reference-Frame Accuracy Evaluation"
    lines.append("## Final evaluation status")
    if final_accuracy_stats is None:
        lines.append(FINAL_NOT_COMPUTED_MESSAGE)
    else:
        lines.append(FINAL_COMPUTED_MESSAGE)
    if user_selected_formula_warning:
        lines.append("")
        lines.append(USER_SELECTED_FORMULA_WARNING)
    if strict_block_reason:
        lines.append("")
        lines.append(f"Blocked reason: {strict_block_reason}")
    lines.append("")

    lines.append("## Reference-frame contract")
    lines.append("```json")
    lines.append(json.dumps(reference_frame_info_to_dict(reference_frame_info), indent=2, default=to_jsonable))
    lines.append("```")
    lines.append("")

    lines.append("## Input files")
    lines.append(f"- CASALS H5: `{h5_path}`")
    for p in las_paths:
        lines.append(f"- 3DEP LAS/LAZ: `{p}`")
    lines.append("")

    lines.append("## CASALS height fields")
    casals_height_cols = [
        col
        for col in [
            "casals_refh_raw_m",
            "geoid",
            "geoid_free2mean",
            "tide_earth",
            "tide_earth_free2mean",
            "tide_load",
            "tide_ocean",
            "tide_ocean_pole",
            "tide_pole",
            "range_bias_correction",
            "refh_error",
            "refh_snr",
            "refh_amp",
            "track_num",
            "sweep_num",
        ]
        if col in casals_meta.get("available_height_related_columns", []) or col in casals_meta.get("available_datasets_first_100", []) or col in config.get("casals_dataframe_columns", [])
    ]
    lines.append("```json")
    lines.append(json.dumps({"available_height_fields": casals_height_cols, "casals_meta": casals_meta}, indent=2, default=to_jsonable)[:25000])
    lines.append("```")
    lines.append("")

    lines.append("## 3DEP reference metadata")
    lines.append("```json")
    lines.append(json.dumps({"reference_frame_info": asdict(reference_frame_info), "las_tile_info": [asdict(x) for x in las_infos]}, indent=2, default=to_jsonable)[:30000])
    lines.append("```")
    lines.append("")

    lines.append("## CRS and vertical datum audit")
    lines.append("```json")
    lines.append(json.dumps({"crs_infos": [asdict(x) for x in crs_infos], "transform_audits": [asdict(x) for x in transform_audits]}, indent=2, default=to_jsonable)[:30000])
    lines.append("```")
    lines.append("")

    lines.append("## CASALS height conversion to 3DEP reference frame")
    lines.append("```json")
    lines.append(json.dumps({
        "casals_height_conversion_mode": config["casals_height_conversion_mode"],
        "h5_geoid_formula": config["h5_geoid_formula"],
        "selected_h5_geoid_formula_for_final": config["selected_h5_geoid_formula_for_final"],
        "final_selection_status": final_selection_status,
        "final_selection_warnings": final_selection_warnings,
    }, indent=2, default=to_jsonable))
    lines.append("```")
    lines.append("")

    lines.append("## Height-conversion candidate diagnostics")
    if candidate_stats:
        lines.append("| candidate | formula | used_as_final | n | median residual m | NMAD m | P05 m | P95 m |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in candidate_stats:
            lines.append(
                f"| {row.candidate_name} | {row.formula} | {int(row.used_as_final)} | {row.n} | "
                f"{row.median_m} | {row.nmad_m} | {row.p05_m} | {row.p95_m} |"
            )
    else:
        lines.append("No H5 height-conversion candidate residuals were produced.")
    lines.append("")

    lines.append("## pyproj / PROJ / GDAL cross-validation")
    lines.append("```json")
    lines.append(json.dumps(crosscheck_meta, indent=2, default=to_jsonable))
    lines.append("```")
    if not crosscheck_df.empty:
        lines.append("")
        lines.append(crosscheck_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Ground filtering and local 3DEP reference construction")
    lines.append(f"- Matched CASALS ground-like points after all main filters: {matched_count}")
    lines.append("- Local 3DEP reference height is the class-2 ground median within the configured radius.")
    lines.append("- Single nearest-neighbor Z is not used as the final reference height.")
    lines.append("")

    lines.append("## Final CASALS-minus-3DEP accuracy metrics")
    if final_accuracy_stats is None:
        lines.append(FINAL_NOT_COMPUTED_MESSAGE)
    else:
        lines.append(FINAL_COMPUTED_MESSAGE)
        lines.append("")
        lines.append("| metric | value_m |")
        lines.append("|---|---:|")
        lines.append(f"| n | {final_accuracy_stats.n} |")
        lines.append(f"| bias_mean | {final_accuracy_stats.bias_m} |")
        lines.append(f"| median | {final_accuracy_stats.median_m} |")
        lines.append(f"| std | {final_accuracy_stats.std_m} |")
        lines.append(f"| RMSE | {final_accuracy_stats.rmse_m} |")
        lines.append(f"| MAE | {final_accuracy_stats.mae_m} |")
        lines.append(f"| NMAD | {final_accuracy_stats.nmad_m} |")
        lines.append(f"| P05 | {final_accuracy_stats.p05_m} |")
        lines.append(f"| P25 | {final_accuracy_stats.p25_m} |")
        lines.append(f"| P75 | {final_accuracy_stats.p75_m} |")
        lines.append(f"| P95 | {final_accuracy_stats.p95_m} |")
        lines.append(f"| abs_P50 | {final_accuracy_stats.abs_p50_m} |")
        lines.append(f"| abs_P68 | {final_accuracy_stats.abs_p68_m} |")
        lines.append(f"| abs_P90 | {final_accuracy_stats.abs_p90_m} |")
        lines.append(f"| abs_P95 | {final_accuracy_stats.abs_p95_m} |")
    lines.append("")

    lines.append("## Residual distribution")
    lines.append("| residual | n | median m | NMAD m | mean m | std m | p05 m | p95 m |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in diagnostic_residual_stats:
        lines.append(
            f"| {row.name} | {row.n} | {row.median_m} | {row.nmad_m} | {row.mean_m} | {row.std_m} | {row.p05_m} | {row.p95_m} |"
        )
    lines.append("")

    lines.append("## Sensitivity analysis")
    if sensitivity_df.empty:
        lines.append("No sensitivity-analysis rows were produced.")
    else:
        display_cols = [
            "match_radius_m",
            "max_3dep_ground_nmad_m",
            "casals_local_low_method",
            "n_matched",
            "n_final",
            "final_bias_m",
            "final_median_m",
            "final_rmse_m",
            "final_nmad_m",
            "status",
        ]
        lines.append(sensitivity_df[display_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Outlier diagnostics")
    lines.append("```json")
    lines.append(json.dumps(outlier_counts, indent=2, default=to_jsonable))
    lines.append("```")
    lines.append("")

    lines.append("## Limitations and remaining assumptions")
    limitations = []
    limitations.extend(reference_frame_info.warnings)
    limitations.extend(final_selection_warnings)
    if config["casals_height_conversion_mode"] != "proj_grid":
        limitations.append("H5 geoid candidate formulas remain interpretive until verified from CASALS product metadata.")
    if not HAS_GDAL_OSR:
        limitations.append("GDAL OSR was not available in this environment.")
    if shutil.which("gdaltransform") is None:
        limitations.append("gdaltransform CLI was not available in this environment.")
    if not HAS_MATPLOTLIB:
        limitations.append("matplotlib was not available, so optional plots were skipped.")
    if not limitations:
        limitations.append("No additional limitations were recorded.")
    for item in limitations:
        lines.append(f"- {item}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # ----------------------------
    # User-editable configuration
    # ----------------------------
    h5_path = Path(r"./casals_h5_downloads/casals_l1b_20241118T171757_001_02.h5")
    las_paths = [
        Path(
            r"./point_cloud_data/download_3dep_lpc/"
            r"casals_l1b_20241118T171757_001_02_NC_HurricaneFlorence_9_2020_"
            r"EPSG6347_f50533a04725.laz"
        ),
    ]
    output_dir = Path(r"./outputs/diagnose_3dep_offsets_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    snr_threshold = 5.0
    max_3dep_points_per_tile = None

    casals_local_cell_size_m = 2.0
    casals_max_above_local_low_m = 1.0
    casals_min_points_per_cell = 3
    casals_local_low_method = "min"

    match_radius_m = 1.0
    match_min_3dep_neighbors = 5

    max_3dep_ground_nmad_m = 0.30
    max_3dep_ground_p95_minus_p05_m = 1.00

    strict_reference_mode = False
    allow_empirical_as_final = False

    explicit_3dep_vertical_crs: Optional[CRS] = None
    explicit_3dep_vertical_crs_reason = None
    # explicit_3dep_vertical_crs = CRS.from_epsg(5703)
    # explicit_3dep_vertical_crs_reason = "Verified from 3DEP project metadata / USGS metadata."

    casals_height_conversion_mode = "h5_geoid"
    h5_geoid_formula = "auto_diagnostic_only"
    selected_h5_geoid_formula_for_final = None

    run_proj_grid_crosscheck = True
    run_gdal_cli_crosscheck = True
    run_gdal_osr_crosscheck = True

    max_pyproj_gdal_vertical_disagreement_m = 0.02
    max_pyproj_gdal_horizontal_disagreement_m = 0.001
    max_allowed_reference_frame_median_abs_residual_m = 2.0

    config = {
        "h5_path": str(h5_path),
        "las_paths": [str(p) for p in las_paths],
        "output_dir": str(output_dir),
        "snr_threshold": snr_threshold,
        "max_3dep_points_per_tile": max_3dep_points_per_tile,
        "casals_local_cell_size_m": casals_local_cell_size_m,
        "casals_max_above_local_low_m": casals_max_above_local_low_m,
        "casals_min_points_per_cell": casals_min_points_per_cell,
        "casals_local_low_method": casals_local_low_method,
        "match_radius_m": match_radius_m,
        "match_min_3dep_neighbors": match_min_3dep_neighbors,
        "max_3dep_ground_nmad_m": max_3dep_ground_nmad_m,
        "max_3dep_ground_p95_minus_p05_m": max_3dep_ground_p95_minus_p05_m,
        "strict_reference_mode": strict_reference_mode,
        "allow_empirical_as_final": allow_empirical_as_final,
        "explicit_3dep_vertical_crs": (
            explicit_3dep_vertical_crs.to_string() if explicit_3dep_vertical_crs else None
        ),
        "explicit_3dep_vertical_crs_reason": explicit_3dep_vertical_crs_reason,
        "casals_height_conversion_mode": casals_height_conversion_mode,
        "h5_geoid_formula": h5_geoid_formula,
        "selected_h5_geoid_formula_for_final": selected_h5_geoid_formula_for_final,
        "run_proj_grid_crosscheck": run_proj_grid_crosscheck,
        "run_gdal_cli_crosscheck": run_gdal_cli_crosscheck,
        "run_gdal_osr_crosscheck": run_gdal_osr_crosscheck,
        "max_pyproj_gdal_vertical_disagreement_m": max_pyproj_gdal_vertical_disagreement_m,
        "max_pyproj_gdal_horizontal_disagreement_m": max_pyproj_gdal_horizontal_disagreement_m,
        "max_allowed_reference_frame_median_abs_residual_m": max_allowed_reference_frame_median_abs_residual_m,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "pyproj_data_dir": datadir.get_data_dir(),
        "pyproj_network_enabled": network.is_network_enabled(),
        "gdal_osr_available": HAS_GDAL_OSR,
        "gdaltransform_available": shutil.which("gdaltransform") is not None,
        "matplotlib_available": HAS_MATPLOTLIB,
    }

    print("=" * 96, flush=True)
    print("CASALS 3DEP reference-frame accuracy evaluation", flush=True)
    print("=" * 96, flush=True)
    print(f"CASALS H5: {h5_path}", flush=True)
    print(f"3DEP tile count: {len(las_paths)}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"strict_reference_mode: {strict_reference_mode}", flush=True)
    print(f"casals_height_conversion_mode: {casals_height_conversion_mode}", flush=True)
    print(flush=True)

    total_steps = 11
    strict_block_reason: Optional[str] = None

    log_step(1, total_steps, "Read CASALS H5 points")
    casals_df, casals_meta = read_casals_refh_points(h5_path, snr_threshold=snr_threshold)
    config["casals_dataframe_columns"] = list(casals_df.columns)

    log_step(2, total_steps, "Read 3DEP LAS metadata and class-2 ground points")
    dep_ground_df, las_infos, dep_crs = read_3dep_ground_points(
        las_paths,
        max_points_per_tile=max_3dep_points_per_tile,
    )

    log_step(3, total_steps, "Build 3DEP reference-frame contract")
    dep_horizontal_crs, target_vertical_crs, reference_frame_info = build_3dep_reference_frame(
        dep_crs,
        explicit_3dep_vertical_crs=explicit_3dep_vertical_crs,
        explicit_3dep_vertical_crs_reason=explicit_3dep_vertical_crs_reason,
    )
    if explicit_3dep_vertical_crs is not None and is_blank(explicit_3dep_vertical_crs_reason):
        if strict_reference_mode:
            raise ValueError(
                "explicit_3dep_vertical_crs_reason must be non-empty when explicit_3dep_vertical_crs is used in strict mode."
            )

    log_step(4, total_steps, "Build CRS audit summaries")
    crs_infos = [
        summarize_crs(
            "CASALS assumed source CRS for tested transformations",
            CRS.from_epsg(4979),
            remarks=[
                "This is an assumption for testing. CASALS refh vertical datum must be verified from product metadata.",
            ],
        ),
        summarize_crs("3DEP LAS parsed CRS", dep_crs, remarks=reference_frame_info.warnings),
        summarize_crs("3DEP horizontal CRS", dep_horizontal_crs),
        summarize_crs(
            "3DEP target vertical CRS",
            target_vertical_crs,
            remarks=[
                f"Vertical CRS source: {reference_frame_info.vertical_crs_source}",
                f"Explicit vertical CRS reason: {explicit_3dep_vertical_crs_reason}",
            ] if target_vertical_crs is not None else reference_frame_info.warnings,
        ),
    ]

    log_step(5, total_steps, "Project CASALS points into 3DEP horizontal CRS")
    casals_projected, horizontal_audit = project_casals_to_horizontal_crs(casals_df, dep_horizontal_crs)

    log_step(6, total_steps, "Filter CASALS ground-like points")
    casals_ground_like = filter_casals_ground_like_by_local_low_surface(
        casals_projected,
        cell_size_m=casals_local_cell_size_m,
        max_above_local_low_m=casals_max_above_local_low_m,
        min_points_per_cell=casals_min_points_per_cell,
        local_low_method=casals_local_low_method,
    )

    log_step(7, total_steps, "Match local 3DEP class-2 ground surface")
    matched = match_3dep_ground_local_median(
        casals_ground_like,
        dep_ground_df,
        radius_m=match_radius_m,
        min_neighbors=match_min_3dep_neighbors,
    )
    if matched.empty:
        raise RuntimeError("No CASALS ground-like points matched local 3DEP ground. Check extent, CRS, or thresholds.")

    matched = matched[
        (matched["z_3dep_ground_nmad_m"] <= float(max_3dep_ground_nmad_m))
        & (matched["z_3dep_ground_p95_minus_p05_m"] <= float(max_3dep_ground_p95_minus_p05_m))
    ].copy()
    matched = matched.reset_index(drop=True)
    if matched.empty:
        raise RuntimeError("All matches were rejected by local 3DEP roughness filters.")
    log_status(
        f"Local roughness screening complete: {format_int(len(matched))} matched points remain after 3DEP neighborhood filtering."
    )

    log_step(8, total_steps, "Compute raw diagnostics and H5 height-conversion candidates")
    matched, diagnostic_residual_stats = compute_diagnostic_residuals(matched)
    matched, geoid_candidate_warnings = add_h5_geoid_height_candidates(matched)
    candidate_columns = [col for col in FORMULA_TO_COLUMN.values() if col in matched.columns]
    matched, candidate_stats = evaluate_height_conversion_candidates(matched, candidate_columns)

    log_step(9, total_steps, "Run reference-frame conversion and pyproj / GDAL cross-checks")
    bounds = lonlat_bounds_from_df(matched)
    reference_transform_audit = TransformAudit(
        name="casals_4979_to_3dep_reference_pyproj",
        source_crs=CRS.from_epsg(4979).to_string(),
        target_crs=dep_horizontal_crs.to_string(),
        allow_ballpark=False,
        always_xy=True,
        best_available=None,
        transformer_count=None,
        unavailable_operations_count=None,
        first_transformer_description=None,
        first_transformer_accuracy_m=None,
        missing_grids=[],
        z_change_median_m=None,
        z_change_p05_m=None,
        z_change_p95_m=None,
        status="not_attempted",
        warnings=[],
    )
    reference_crosscheck_df = pd.DataFrame(
        columns=[
            "method",
            "status",
            "n_samples",
            "median_abs_horizontal_diff_m",
            "p95_abs_horizontal_diff_m",
            "median_abs_vertical_diff_m",
            "p95_abs_vertical_diff_m",
            "max_abs_vertical_diff_m",
        ]
    )
    reference_crosscheck_meta: dict[str, Any] = {
        "status": "not_attempted",
        "reason": "reference_frame_transform_not_attempted",
    }

    if casals_height_conversion_mode in {"proj_grid", "compare_h5_geoid_and_proj_grid"}:
        if target_vertical_crs is not None:
            matched, reference_transform_audit = convert_casals_to_target_compound_crs_with_pyproj(
                matched,
                target_horizontal_crs=dep_horizontal_crs,
                target_vertical_crs=target_vertical_crs,
                bounds_lonlat=bounds,
            )
            if run_proj_grid_crosscheck:
                try:
                    dst_crs = build_target_compound_crs(dep_horizontal_crs, target_vertical_crs)
                    reference_crosscheck_df, reference_crosscheck_meta = transformation_crosscheck(
                        casals_df=matched,
                        src_crs=CRS.from_epsg(4979),
                        dst_crs=dst_crs,
                        output_csv=output_dir / "reference_frame_transform_crosscheck.csv",
                        output_json=output_dir / "reference_frame_transform_crosscheck.json",
                    )
                    if not run_gdal_osr_crosscheck:
                        reference_crosscheck_df = reference_crosscheck_df.loc[
                            reference_crosscheck_df["method"] != "gdal_osr"
                        ].reset_index(drop=True)
                    if not run_gdal_cli_crosscheck:
                        reference_crosscheck_df = reference_crosscheck_df.loc[
                            reference_crosscheck_df["method"] != "gdaltransform"
                        ].reset_index(drop=True)
                    enforce_crosscheck_thresholds(
                        reference_crosscheck_df,
                        reference_crosscheck_meta,
                        strict_reference_mode=strict_reference_mode,
                        max_pyproj_gdal_vertical_disagreement_m=max_pyproj_gdal_vertical_disagreement_m,
                        max_pyproj_gdal_horizontal_disagreement_m=max_pyproj_gdal_horizontal_disagreement_m,
                        check_gdal_osr=run_gdal_osr_crosscheck,
                        check_gdal_cli=run_gdal_cli_crosscheck,
                    )
                except Exception as exc:
                    reference_crosscheck_meta = {
                        "status": "crosscheck_failed",
                        "error": str(exc),
                    }
                    if strict_reference_mode:
                        raise
            enforce_pyproj_reference_audit(reference_transform_audit, strict_reference_mode=strict_reference_mode)
        else:
            reference_transform_audit.status = "target_vertical_crs_unknown"
            reference_transform_audit.warnings.append(STRICT_VERTICAL_UNAVAILABLE_WARNING)

    transform_audits = [horizontal_audit, reference_transform_audit]

    log_step(10, total_steps, "Select final CASALS reference height and compute final accuracy")
    final_selection_warnings = list(geoid_candidate_warnings)
    try:
        matched, final_selection_status, selection_warnings = select_casals_reference_height(
            matched=matched,
            mode=casals_height_conversion_mode,
            h5_geoid_formula=h5_geoid_formula,
            selected_h5_geoid_formula_for_final=selected_h5_geoid_formula_for_final,
            target_vertical_crs=target_vertical_crs,
            strict_reference_mode=strict_reference_mode,
            compare_h5_proj_agreement_threshold_m=max_pyproj_gdal_vertical_disagreement_m,
        )
        final_selection_warnings.extend(selection_warnings)
    except Exception as exc:
        final_selection_status = "selection_failed"
        strict_block_reason = str(exc)
        if strict_reference_mode:
            raise

    for row in candidate_stats:
        if FORMULA_TO_COLUMN.get(selected_h5_geoid_formula_for_final or h5_geoid_formula) == row.candidate_name and "casals_z_in_3dep_reference_m" in matched.columns:
            if final_selection_status == "h5_geoid_formula_selected":
                row.used_as_final = True

    matched, final_accuracy_stats = compute_final_reference_frame_accuracy(
        matched,
        strict_reference_mode=False if strict_block_reason is not None else strict_reference_mode,
    )
    if final_accuracy_stats is not None:
        matched = mark_outliers(matched)
        if (
            strict_reference_mode
            and final_accuracy_stats.abs_p50_m is not None
            and final_accuracy_stats.abs_p50_m > float(max_allowed_reference_frame_median_abs_residual_m)
        ):
            raise RuntimeError(
                f"Final median absolute residual exceeds threshold: {final_accuracy_stats.abs_p50_m} m > {max_allowed_reference_frame_median_abs_residual_m} m."
            )
    elif strict_reference_mode and strict_block_reason is None:
        strict_block_reason = FINAL_NOT_COMPUTED_MESSAGE
        raise RuntimeError(FINAL_NOT_COMPUTED_MESSAGE)

    if final_accuracy_stats is None and strict_block_reason is None:
        strict_block_reason = FINAL_NOT_COMPUTED_MESSAGE

    if final_accuracy_stats is not None:
        reference_frame_info.geoid_model_or_source = matched.get("casals_height_conversion_formula", pd.Series(["unknown"])).iloc[0]
        reference_frame_info.reference_frame_status = "final_reference_frame_available"
    else:
        if casals_height_conversion_mode in {"h5_geoid", "compare_h5_geoid_and_proj_grid"}:
            reference_frame_info.geoid_model_or_source = "H5 geoid candidate diagnostics"
        elif casals_height_conversion_mode == "proj_grid":
            reference_frame_info.geoid_model_or_source = "pyproj / PROJ grid transform attempt"

    log_step(11, total_steps, "Write outputs, plots, report, and sensitivity analysis")
    candidate_summary_df = pd.DataFrame([asdict(x) for x in candidate_stats])
    candidate_summary_df.to_csv(output_dir / "height_conversion_candidates_summary.csv", index=False)
    write_json(output_dir / "height_conversion_candidates_summary.json", [asdict(x) for x in candidate_stats])

    write_json(output_dir / "reference_frame_info.json", asdict(reference_frame_info))
    write_json(output_dir / "reference_frame_transform_audit.json", asdict(reference_transform_audit))
    if reference_crosscheck_df.empty:
        reference_crosscheck_df.to_csv(output_dir / "reference_frame_transform_crosscheck.csv", index=False)
        write_json(output_dir / "reference_frame_transform_crosscheck.json", reference_crosscheck_meta)

    write_json(output_dir / "config.json", config)
    write_json(output_dir / "casals_metadata.json", casals_meta)
    write_json(output_dir / "las_tile_metadata.json", [asdict(x) for x in las_infos])
    write_json(output_dir / "crs_audit.json", [asdict(x) for x in crs_infos])
    write_json(output_dir / "transform_audit.json", [asdict(x) for x in transform_audits])
    write_json(output_dir / "reference_frame_crosscheck_meta.json", reference_crosscheck_meta)
    write_json(output_dir / "diagnostic_residual_summary.json", [asdict(x) for x in diagnostic_residual_stats])

    matched.to_csv(output_dir / "matched_ground_points_reference_frame.csv", index=False)
    matched.to_csv(output_dir / "matched_ground_points_with_residuals.csv", index=False)

    final_summary_df = pd.DataFrame([asdict(final_accuracy_stats)]) if final_accuracy_stats is not None else pd.DataFrame(
        [{"status": "not_computed", "reason": strict_block_reason}]
    )
    final_summary_df.to_csv(output_dir / "final_reference_frame_accuracy_summary.csv", index=False)
    write_json(
        output_dir / "final_reference_frame_accuracy_summary.json",
        asdict(final_accuracy_stats) if final_accuracy_stats is not None else {"status": "not_computed", "reason": strict_block_reason},
    )

    per_tile_df = per_tile_residual_summary(
        matched,
        [
            "raw_mixed_datum_difference_m",
            "empirical_median_removed_residual_m",
            "residual_casals_minus_3dep_reference_m",
        ],
    )
    per_tile_df.to_csv(output_dir / "per_tile_residual_summary.csv", index=False)

    outlier_counts = export_outlier_csvs(matched, output_dir)

    sensitivity_df = run_reference_frame_sensitivity_analysis(
        casals_projected=casals_projected,
        dep_ground_df=dep_ground_df,
        target_vertical_crs=target_vertical_crs,
        casals_local_cell_size_m=casals_local_cell_size_m,
        casals_max_above_local_low_m=casals_max_above_local_low_m,
        casals_min_points_per_cell=casals_min_points_per_cell,
        match_min_3dep_neighbors=match_min_3dep_neighbors,
        max_3dep_ground_p95_minus_p05_m=max_3dep_ground_p95_minus_p05_m,
        casals_height_conversion_mode=casals_height_conversion_mode,
        h5_geoid_formula=h5_geoid_formula,
        selected_h5_geoid_formula_for_final=selected_h5_geoid_formula_for_final,
        strict_reference_mode=strict_reference_mode,
    )
    sensitivity_df.to_csv(output_dir / "reference_frame_sensitivity_summary.csv", index=False)

    if HAS_MATPLOTLIB:
        plot_hist_reference_residual(matched, output_dir / "hist_residual_casals_minus_3dep_reference.png")
        plot_hist_height_conversion_candidates(matched, candidate_stats, output_dir / "hist_height_conversion_candidates.png")
        plot_scatter_raw_vs_geoid(matched, output_dir / "scatter_raw_vs_geoid.png")
        plot_map_reference_residual_xy(matched, output_dir / "map_residual_reference_frame_xy.png")
        plot_map_outliers_reference_xy(matched, output_dir / "map_outliers_reference_frame_xy.png")

    write_diagnosis_report(
        path=output_dir / "diagnosis_report.md",
        h5_path=h5_path,
        las_paths=las_paths,
        casals_meta=casals_meta,
        las_infos=las_infos,
        reference_frame_info=reference_frame_info,
        crs_infos=crs_infos,
        transform_audits=transform_audits,
        crosscheck_meta=reference_crosscheck_meta,
        crosscheck_df=reference_crosscheck_df,
        candidate_stats=candidate_stats,
        diagnostic_residual_stats=diagnostic_residual_stats,
        final_accuracy_stats=final_accuracy_stats,
        sensitivity_df=sensitivity_df,
        matched_count=len(matched),
        final_selection_status=final_selection_status,
        final_selection_warnings=final_selection_warnings,
        strict_block_reason=strict_block_reason,
        outlier_counts=outlier_counts,
        config=config,
        user_selected_formula_warning=selected_h5_geoid_formula_for_final is not None,
    )

    log_status(
        f"Output writing complete: matched rows={format_int(len(matched))}, "
        f"candidate summaries={format_int(len(candidate_stats))}, sensitivity rows={format_int(len(sensitivity_df))}."
    )
    print(f"Done. Outputs written to: {output_dir}")
    if final_accuracy_stats is None:
        print(FINAL_NOT_COMPUTED_MESSAGE)
    else:
        print(FINAL_COMPUTED_MESSAGE)


if __name__ == "__main__":
    main()
