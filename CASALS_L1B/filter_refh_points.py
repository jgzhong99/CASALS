# -*- coding: utf-8 -*-
"""Filter CASALS L1B refh points into raw, noise-labeled, and clean LAS outputs.

Scientific meaning:
    Each input point is the official CASALS L1B geolocated refh reference point,
    corresponding to the Rx waveform maximum-amplitude bin.

Outputs:
    `raw_refh.las`, `noise_labeled_refh.las`, `clean_refh.las`, one metadata JSON,
    and core preview PNGs.

This script does not:
    - create a ground DEM,
    - create an official multi-return point cloud,
    - use waveform peak detection as a primary geolocation workflow,
    - resolve vertical datum differences with 3DEP/NAVD88 products.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
from pyproj import CRS, Transformer

import laspy
from laspy import ExtraBytesParams

import matplotlib.pyplot as plt

from refh_noise import NoiseResult as FilterResult, label_noise as shared_label_noise


# =============================================================================
# Configuration dataclass
# =============================================================================

@dataclass(frozen=True)
class Config:
    # Input/output.
    h5_path: Path
    point_cloud_dir: Path
    output_dir: Path

    # CRS handling.
    # If None, infer WGS84 UTM zone from median refh lon/lat.
    utm_epsg_override: Optional[int] = None
    las_xyz_scale_m: float = 0.001

    # Required filter behavior.
    # These validity checks are always applied: finite lon/lat/refh, valid lon/lat,
    # finite refh_amp/refh_snr.
    filter_good_snr_only: bool = False
    refh_snr_min_for_input: Optional[float] = None
    track_range: Optional[Tuple[int, int]] = None
    sweep_range: Optional[Tuple[int, int]] = None

    # Signal-quality labeling. These DO NOT affect raw output; they label noise.
    use_snr_amp_noise_label: bool = True
    snr_hard_min: float = 1.5
    snr_soft_min: float = 1.8
    amp_low_percentile: float = 5.0
    # If True, label as noise when both snr < snr_soft_min and amp below percentile.
    # Also label as noise when snr < snr_hard_min regardless of amplitude.
    low_signal_requires_low_amp_for_soft_snr: bool = True

    # Per-pulse threshold field if present.
    use_refh_threshold_if_available: bool = True
    refh_threshold_margin_min: float = 0.0

    # Optional geolocation / refh error labeling if available.
    use_error_fields_if_available: bool = True
    max_refh_error_m: Optional[float] = None
    max_refh_horizontal_error_deg: Optional[float] = None

    # Optional global z percentile guard for catastrophic high/low returns.
    # Keep disabled for formal products; useful for quick visualization.
    use_global_z_percentile_guard: bool = False
    z_low_percentile: float = 0.05
    z_high_percentile: float = 99.95

    # Local height consistency labeling in projected coordinates.
    use_local_height_filter: bool = True
    local_grid_cell_size_m: float = 5.0
    local_min_points_per_cell: int = 12
    local_abs_residual_threshold_m: float = 25.0
    local_mad_multiplier: float = 8.0
    local_min_sigma_m: float = 0.75

    # Optional Open3D statistical outlier removal.
    # This can be slow/memory-heavy on millions of points. It is disabled by default.
    use_open3d_statistical_outlier: bool = False
    open3d_sor_nb_neighbors: int = 20
    open3d_sor_std_ratio: float = 2.5
    open3d_sor_max_points: int = 750_000
    open3d_sor_seed: int = 42
    # If False and points exceed max, skip SOR rather than sampling for final labeling.
    # Sampling-based SOR is useful for preview but not rigorous for final all-point labels.
    allow_sampled_sor_for_labeling: bool = False

    # Output controls.
    write_raw_las: bool = True
    write_noise_labeled_las: bool = True
    write_clean_las: bool = True
    write_noise_only_las: bool = False
    write_metadata_json: bool = True
    write_preview_png: bool = True

    # Visualization / preview controls.
    preview_max_points: int = 300_000
    preview_seed: int = 42
    rgb_color_by: str = "refh_amp"  # "refh_amp", "refh", "refh_snr", "classification"
    robust_color_percentiles: Tuple[float, float] = (2.0, 98.0)
    visualize_open3d: bool = False
    open3d_visual_max_points: int = 350_000
    open3d_vertical_exaggeration: float = 1.0
    open3d_color_mode: str = "classification"  # "classification", "amp", "snr", "height"

    # Behavior.
    overwrite: bool = True


# =============================================================================
# Constants / bit masks
# =============================================================================

REQUIRED_DATASETS = [
    "refh_longitude",
    "refh_latitude",
    "refh",
    "refh_amp",
    "refh_snr",
    "good_snr",
    "track_num",
    "sweep_num",
]

OPTIONAL_DATASETS = [
    "delta_time",
    "refh_thres",
    "bg_mean",
    "bg_std",
    "refh_error",
    "refh_longitude_error",
    "refh_latitude_error",
]

# =============================================================================
# Generic utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_h5_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def read_global_attributes(h5: h5py.File) -> Dict[str, Any]:
    return {k: normalize_h5_attr(v) for k, v in h5.attrs.items()}


def require_datasets(h5: h5py.File, required: list[str]) -> None:
    missing = [name for name in required if name not in h5]
    if missing:
        raise KeyError("Missing required datasets: " + ", ".join(missing))


def as_1d_array(h5: h5py.File, name: str) -> np.ndarray:
    arr = np.asarray(h5[name][...])
    if arr.ndim != 1:
        raise ValueError(f"Dataset {name!r} is expected to be 1D; got shape={arr.shape}")
    return arr


def optional_1d_arrays(h5: h5py.File, names: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in names:
        if name in h5:
            try:
                out[name] = as_1d_array(h5, name)
            except Exception as exc:
                warnings.warn(f"Optional dataset {name!r} was not loaded: {type(exc).__name__}: {exc}")
    return out


def infer_utm_epsg_from_lonlat(lon: np.ndarray, lat: np.ndarray) -> int:
    lon_med = float(np.nanmedian(lon))
    lat_med = float(np.nanmedian(lat))
    if not (-180.0 <= lon_med <= 180.0 and -90.0 <= lat_med <= 90.0):
        raise ValueError(f"Invalid median lon/lat for UTM inference: {lon_med}, {lat_med}")
    zone = int(math.floor((lon_med + 180.0) / 6.0) + 1)
    zone = max(1, min(zone, 60))
    return 32600 + zone if lat_med >= 0 else 32700 + zone


def robust_min_max(values: np.ndarray, percentiles: Tuple[float, float]) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return 0.0, 1.0
    p_low, p_high = np.nanpercentile(vals[finite], percentiles)
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        vmin = float(np.nanmin(vals[finite]))
        vmax = float(np.nanmax(vals[finite]))
        return vmin, vmax if vmax > vmin else vmin + 1.0
    return float(p_low), float(p_high)


def summarize_array(name: str, arr: np.ndarray) -> Dict[str, Any]:
    vals = np.asarray(arr)
    try:
        vals_float = vals.astype(np.float64, copy=False)
    except Exception:
        return {"name": name, "n": int(vals.size), "dtype": str(vals.dtype), "summary": "non_numeric"}
    finite = np.isfinite(vals_float)
    if not np.any(finite):
        return {"name": name, "n": int(vals.size), "n_finite": 0, "min": None, "p02": None, "p50": None, "p98": None, "max": None}
    q = np.nanpercentile(vals_float[finite], [2, 50, 98])
    return {
        "name": name,
        "n": int(vals.size),
        "n_finite": int(np.sum(finite)),
        "min": float(np.nanmin(vals_float[finite])),
        "p02": float(q[0]),
        "p50": float(q[1]),
        "p98": float(q[2]),
        "max": float(np.nanmax(vals_float[finite])),
    }


def validate_inverse_projection(
    x: np.ndarray,
    y: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    utm_epsg: int,
    sample_size: int = 50_000,
    seed: int = 42,
) -> Dict[str, float]:
    n = len(x)
    if n == 0:
        return {
            "inverse_projection_sample_size": 0,
            "max_abs_lon_error_deg": float("nan"),
            "max_abs_lat_error_deg": float("nan"),
            "approx_max_horizontal_error_m": float("nan"),
        }
    if n > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_size, replace=False)
    else:
        idx = np.arange(n)
    transformer_back = Transformer.from_crs(CRS.from_epsg(utm_epsg), CRS.from_epsg(4326), always_xy=True)
    lon_back, lat_back = transformer_back.transform(x[idx], y[idx])
    lon_err = np.asarray(lon_back) - lon[idx]
    lat_err = np.asarray(lat_back) - lat[idx]
    max_abs_lon_error_deg = float(np.nanmax(np.abs(lon_err)))
    max_abs_lat_error_deg = float(np.nanmax(np.abs(lat_err)))
    lat_med_rad = np.deg2rad(float(np.nanmedian(lat[idx])))
    meters_per_deg_lon = 111_320.0 * max(math.cos(lat_med_rad), 1e-6)
    approx_max_horizontal_error_m = max(
        max_abs_lon_error_deg * meters_per_deg_lon,
        max_abs_lat_error_deg * 111_320.0,
    )
    return {
        "inverse_projection_sample_size": int(len(idx)),
        "max_abs_lon_error_deg": max_abs_lon_error_deg,
        "max_abs_lat_error_deg": max_abs_lat_error_deg,
        "approx_max_horizontal_error_m": float(approx_max_horizontal_error_m),
    }


# =============================================================================
# Input and georeference
# =============================================================================

@dataclass
class RefhPointData:
    lon: np.ndarray
    lat: np.ndarray
    z_refh: np.ndarray
    refh_amp: np.ndarray
    refh_snr: np.ndarray
    good_snr: np.ndarray
    track_num: np.ndarray
    sweep_num: np.ndarray
    pulse_index: np.ndarray
    optional: dict[str, np.ndarray]
    attrs: dict[str, Any]
    n_input_records: int
    n_valid_records: int
    input_mask_summary: dict[str, Any]


def build_input_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    z: np.ndarray,
    refh_amp: np.ndarray,
    refh_snr: np.ndarray,
    good_snr: np.ndarray,
    track_num: np.ndarray,
    sweep_num: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    mask = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & np.isfinite(z)
        & (lon >= -180.0)
        & (lon <= 180.0)
        & (lat >= -90.0)
        & (lat <= 90.0)
        & np.isfinite(np.asarray(refh_amp, dtype=np.float64))
        & np.isfinite(np.asarray(refh_snr, dtype=np.float64))
    )
    if cfg.filter_good_snr_only:
        mask &= good_snr.astype(bool)
    if cfg.refh_snr_min_for_input is not None:
        mask &= np.asarray(refh_snr, dtype=np.float64) >= float(cfg.refh_snr_min_for_input)
    if cfg.track_range is not None:
        t0, t1 = cfg.track_range
        mask &= (track_num >= t0) & (track_num <= t1)
    if cfg.sweep_range is not None:
        s0, s1 = cfg.sweep_range
        mask &= (sweep_num >= s0) & (sweep_num <= s1)
    return mask


def read_refh_point_data(cfg: Config) -> RefhPointData:
    with h5py.File(cfg.h5_path, "r") as h5:
        attrs = read_global_attributes(h5)
        require_datasets(h5, REQUIRED_DATASETS)
        lon = as_1d_array(h5, "refh_longitude").astype(np.float64)
        lat = as_1d_array(h5, "refh_latitude").astype(np.float64)
        z_refh = as_1d_array(h5, "refh").astype(np.float64)
        refh_amp = as_1d_array(h5, "refh_amp").astype(np.float64)
        refh_snr = as_1d_array(h5, "refh_snr").astype(np.float64)
        good_snr = as_1d_array(h5, "good_snr").astype(bool)
        track_num = as_1d_array(h5, "track_num")
        sweep_num = as_1d_array(h5, "sweep_num")
        optional_all = optional_1d_arrays(h5, OPTIONAL_DATASETS)

    shapes = {
        "refh_longitude": lon.shape,
        "refh_latitude": lat.shape,
        "refh": z_refh.shape,
        "refh_amp": refh_amp.shape,
        "refh_snr": refh_snr.shape,
        "good_snr": good_snr.shape,
        "track_num": track_num.shape,
        "sweep_num": sweep_num.shape,
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Required arrays have inconsistent shapes: {shapes}")

    n_input = len(lon)
    if "n_pulses" in attrs and int(attrs["n_pulses"]) != n_input:
        raise ValueError(f"HDF5 n_pulses={attrs['n_pulses']} but refh length={n_input}")

    input_mask = build_input_mask(lon, lat, z_refh, refh_amp, refh_snr, good_snr, track_num, sweep_num, cfg)
    n_valid = int(np.sum(input_mask))
    if n_valid == 0:
        raise RuntimeError("No valid refh points after input validity filtering.")

    pulse_index = np.arange(n_input, dtype=np.uint32)
    optional = {name: arr[input_mask] for name, arr in optional_all.items() if len(arr) == n_input}

    input_mask_summary = {
        "n_input_records": int(n_input),
        "n_valid_records": int(n_valid),
        "filter_good_snr_only": bool(cfg.filter_good_snr_only),
        "refh_snr_min_for_input": cfg.refh_snr_min_for_input,
        "track_range": cfg.track_range,
        "sweep_range": cfg.sweep_range,
        "good_snr_fraction_input": float(np.mean(good_snr)),
        "good_snr_fraction_valid": float(np.mean(good_snr[input_mask])),
    }

    return RefhPointData(
        lon=lon[input_mask],
        lat=lat[input_mask],
        z_refh=z_refh[input_mask],
        refh_amp=refh_amp[input_mask],
        refh_snr=refh_snr[input_mask],
        good_snr=good_snr[input_mask],
        track_num=track_num[input_mask],
        sweep_num=sweep_num[input_mask],
        pulse_index=pulse_index[input_mask],
        optional=optional,
        attrs=attrs,
        n_input_records=n_input,
        n_valid_records=n_valid,
        input_mask_summary=input_mask_summary,
    )


@dataclass
class ProjectedData:
    easting: np.ndarray
    northing: np.ndarray
    utm_epsg: int
    utm_crs_name: str
    projection_check: dict[str, Any]


def project_lonlat_to_utm(data: RefhPointData, cfg: Config) -> ProjectedData:
    utm_epsg = int(cfg.utm_epsg_override) if cfg.utm_epsg_override is not None else infer_utm_epsg_from_lonlat(data.lon, data.lat)
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(utm_epsg), always_xy=True)
    easting, northing = transformer.transform(data.lon, data.lat)
    easting = np.asarray(easting, dtype=np.float64)
    northing = np.asarray(northing, dtype=np.float64)
    projection_check = validate_inverse_projection(easting, northing, data.lon, data.lat, utm_epsg)
    return ProjectedData(
        easting=easting,
        northing=northing,
        utm_epsg=utm_epsg,
        utm_crs_name=CRS.from_epsg(utm_epsg).name,
        projection_check=projection_check,
    )


# =============================================================================
# Noise labeling
# =============================================================================

def label_noise(data: RefhPointData, proj: ProjectedData, cfg: Config) -> FilterResult:
    return shared_label_noise(data, proj, cfg)


# =============================================================================
# LAS writing
# =============================================================================

def values_to_rgb16(
    values: np.ndarray,
    percentiles: Tuple[float, float] = (2.0, 98.0),
    cmap_name: str = "viridis",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    vals = np.asarray(values, dtype=np.float64)
    vmin, vmax = robust_min_max(vals, percentiles)
    norm = (vals - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    norm[~np.isfinite(norm)] = 0.0
    cmap = plt.get_cmap(cmap_name)
    rgb_float = cmap(norm)[:, :3]
    rgb16 = np.round(rgb_float * 65535.0).astype(np.uint16)
    return rgb16[:, 0], rgb16[:, 1], rgb16[:, 2], (vmin, vmax)


def classification_to_rgb16(classification: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    cls = np.asarray(classification, dtype=np.uint8)
    rgb = np.zeros((len(cls), 3), dtype=np.uint16)
    # Class 1 retained/unclassified: cool gray/blue.
    keep = cls != 7
    rgb[keep] = np.array([15000, 36000, 65535], dtype=np.uint16)
    # Class 7 likely noise: red/orange.
    rgb[~keep] = np.array([65535, 9000, 5000], dtype=np.uint16)
    return rgb[:, 0], rgb[:, 1], rgb[:, 2], (0.0, 7.0)


def subset_optional(optional: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: arr[mask] for name, arr in optional.items() if len(arr) == len(mask)}


def extra_dims_for_optional(optional: dict[str, np.ndarray]) -> list[ExtraBytesParams]:
    dims: list[ExtraBytesParams] = []
    for name, arr in optional.items():
        safe_name = name[:32]
        if safe_name == "delta_time":
            desc = "delta_time_sec"
        else:
            desc = safe_name[:31]
        dtype = np.asarray(arr).dtype
        if dtype.kind in "iu":
            typ = np.uint32 if dtype.itemsize > 2 else np.uint16 if dtype.itemsize > 1 else np.uint8
        else:
            typ = np.float64
        dims.append(ExtraBytesParams(name=safe_name, type=typ, description=desc))
    return dims


def write_las(
    path: Path,
    data: RefhPointData,
    proj: ProjectedData,
    cfg: Config,
    mask: np.ndarray,
    classification: np.ndarray,
    reason_code: Optional[np.ndarray] = None,
    local_z_residual: Optional[np.ndarray] = None,
    local_median_z: Optional[np.ndarray] = None,
    local_mad_z: Optional[np.ndarray] = None,
    rgb_color_by: Optional[str] = None,
) -> dict[str, Any]:
    ensure_dir(path.parent)
    if path.exists() and cfg.overwrite:
        path.unlink()
    if path.exists() and not cfg.overwrite:
        return {"path": str(path), "status": "exists_skipped", "n_points": None}

    idx = np.where(mask)[0]
    n = len(idx)
    if n <= 0:
        return {"path": str(path), "status": "skipped_no_points", "n_points": 0}

    x = proj.easting[idx]
    y = proj.northing[idx]
    z = data.z_refh[idx]
    cls = classification[idx].astype(np.uint8)
    optional = subset_optional(data.optional, mask)

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([cfg.las_xyz_scale_m, cfg.las_xyz_scale_m, cfg.las_xyz_scale_m], dtype=np.float64)
    header.offsets = np.array([
        math.floor(float(np.nanmin(x))),
        math.floor(float(np.nanmin(y))),
        math.floor(float(np.nanmin(z))),
    ], dtype=np.float64)
    header.system_identifier = "CASALS_L1B_REFH"
    header.generating_software = "filter_refh_points.py"
    header.add_crs(CRS.from_epsg(proj.utm_epsg))
    las = laspy.LasData(header)

    extra_dims = [
        ExtraBytesParams(name="longitude", type=np.float64, description="refh_lon_deg_WGS84"),
        ExtraBytesParams(name="latitude", type=np.float64, description="refh_lat_deg_WGS84"),
        ExtraBytesParams(name="refh_amp_raw", type=np.float64, description="refh_amp_raw_counts"),
        ExtraBytesParams(name="refh_snr", type=np.float64, description="refh_snr"),
        ExtraBytesParams(name="good_snr", type=np.uint8, description="good_snr_1_true"),
        ExtraBytesParams(name="track_num", type=np.uint16, description="track_channel"),
        ExtraBytesParams(name="sweep_num", type=np.uint32, description="sweep_number"),
        ExtraBytesParams(name="pulse_index", type=np.uint32, description="pulse_index"),
    ]
    if reason_code is not None:
        extra_dims.append(ExtraBytesParams(name="noise_reason", type=np.uint16, description="noise_reason_bits"))
    if local_z_residual is not None:
        extra_dims.append(ExtraBytesParams(name="local_z_resid", type=np.float64, description="local_z_resid_m"))
    if local_median_z is not None:
        extra_dims.append(ExtraBytesParams(name="local_z_med", type=np.float64, description="local_z_median_m"))
    if local_mad_z is not None:
        extra_dims.append(ExtraBytesParams(name="local_z_mad", type=np.float64, description="local_z_mad_m"))
    extra_dims.extend(extra_dims_for_optional(optional))
    las.add_extra_dims(extra_dims)

    las.x = x.astype(np.float64)
    las.y = y.astype(np.float64)
    las.z = z.astype(np.float64)
    las.intensity = np.clip(data.refh_amp[idx], 0, 65535).astype(np.uint16)
    las.classification = cls

    color_mode = rgb_color_by or cfg.rgb_color_by
    if color_mode == "classification":
        r, g, b, rgb_range = classification_to_rgb16(cls)
    elif color_mode == "refh":
        r, g, b, rgb_range = values_to_rgb16(z, cfg.robust_color_percentiles)
    elif color_mode == "refh_snr":
        r, g, b, rgb_range = values_to_rgb16(data.refh_snr[idx], cfg.robust_color_percentiles)
    elif color_mode == "refh_amp":
        r, g, b, rgb_range = values_to_rgb16(data.refh_amp[idx], cfg.robust_color_percentiles)
    else:
        raise ValueError(f"Unsupported rgb_color_by={color_mode!r}")
    las.red, las.green, las.blue = r, g, b

    las.longitude = data.lon[idx].astype(np.float64)
    las.latitude = data.lat[idx].astype(np.float64)
    las.refh_amp_raw = data.refh_amp[idx].astype(np.float64)
    las.refh_snr = data.refh_snr[idx].astype(np.float64)
    las.good_snr = data.good_snr[idx].astype(np.uint8)
    las.track_num = data.track_num[idx].astype(np.uint16)
    las.sweep_num = data.sweep_num[idx].astype(np.uint32)
    las.pulse_index = data.pulse_index[idx].astype(np.uint32)
    if reason_code is not None:
        las.noise_reason = reason_code[idx].astype(np.uint16)
    if local_z_residual is not None:
        las.local_z_resid = local_z_residual[idx].astype(np.float64)
    if local_median_z is not None:
        las.local_z_med = local_median_z[idx].astype(np.float64)
    if local_mad_z is not None:
        las.local_z_mad = local_mad_z[idx].astype(np.float64)

    for name, arr in optional.items():
        safe_name = name[:32]
        if np.asarray(arr).dtype.kind in "iu":
            setattr(las, safe_name, np.asarray(arr, dtype=np.uint32))
        else:
            setattr(las, safe_name, np.asarray(arr, dtype=np.float64))

    las.write(str(path))
    return {
        "path": str(path),
        "status": "written",
        "n_points": int(n),
        "bytes": int(path.stat().st_size),
        "classification_counts": {str(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True))},
        "rgb_color_by": color_mode,
        "rgb_range": [float(rgb_range[0]), float(rgb_range[1])],
        "horizontal_crs_epsg": int(proj.utm_epsg),
        "z_convention": "CASALS refh WGS84 ellipsoidal height; no vertical datum conversion",
    }


# =============================================================================
# Preview plots and Open3D visualization
# =============================================================================

def sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=int(max_points), replace=False)


def make_debug_previews(
    out_dir: Path,
    stem: str,
    data: RefhPointData,
    proj: ProjectedData,
    filt: FilterResult,
    cfg: Config,
) -> dict[str, str]:
    ensure_dir(out_dir)
    idx = sample_indices(len(data.z_refh), cfg.preview_max_points, cfg.preview_seed)
    outputs: dict[str, str] = {}

    def save_current(name: str) -> None:
        path = out_dir / f"{stem}_{name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()
        outputs[name] = str(path)

    # Spatial classification preview.
    plt.figure(figsize=(12, 8))
    keep = filt.keep_mask[idx]
    plt.scatter(proj.easting[idx][keep], proj.northing[idx][keep], s=0.2, c="tab:blue", linewidths=0, alpha=0.45, label="kept")
    plt.scatter(proj.easting[idx][~keep], proj.northing[idx][~keep], s=0.35, c="tab:red", linewidths=0, alpha=0.55, label="likely noise")
    plt.axis("equal")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("CASALS refh noise labeling preview")
    plt.legend(markerscale=8)
    save_current("classification_noise_mask")

    # Raw amplitude spatial map.
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(proj.easting[idx], proj.northing[idx], c=data.refh_amp[idx], s=0.25, linewidths=0, cmap="viridis")
    plt.axis("equal")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("CASALS refh_amp spatial preview")
    plt.colorbar(sc, label="refh_amp")
    save_current("raw_refh_amp")

    # SNR map.
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(proj.easting[idx], proj.northing[idx], c=data.refh_snr[idx], s=0.25, linewidths=0, cmap="viridis")
    plt.axis("equal")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("CASALS refh_snr spatial preview")
    plt.colorbar(sc, label="refh_snr")
    save_current("snr_map")

    return outputs


def open3d_visualize(data: RefhPointData, proj: ProjectedData, filt: FilterResult, cfg: Config) -> None:
    if not cfg.visualize_open3d:
        return
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        print(f"Open3D visualization skipped: {type(exc).__name__}: {exc}")
        return

    n = len(data.z_refh)
    idx = sample_indices(n, cfg.open3d_visual_max_points, cfg.preview_seed)
    x0 = float(np.nanmedian(proj.easting[idx]))
    y0 = float(np.nanmedian(proj.northing[idx]))
    z0 = float(np.nanmedian(data.z_refh[idx]))
    pts = np.column_stack([
        proj.easting[idx] - x0,
        proj.northing[idx] - y0,
        (data.z_refh[idx] - z0) * float(cfg.open3d_vertical_exaggeration),
    ]).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    colors = np.zeros((len(idx), 3), dtype=np.float64)
    mode = cfg.open3d_color_mode
    if mode == "classification":
        colors[filt.keep_mask[idx]] = np.array([0.1, 0.45, 1.0])
        colors[filt.noise_mask[idx]] = np.array([1.0, 0.1, 0.02])
    else:
        if mode == "amp":
            vals = data.refh_amp[idx]
        elif mode == "snr":
            vals = data.refh_snr[idx]
        elif mode == "height":
            vals = data.z_refh[idx]
        else:
            vals = data.refh_amp[idx]
        vmin, vmax = robust_min_max(vals, cfg.robust_color_percentiles)
        norm = np.clip((vals - vmin) / (vmax - vmin), 0, 1)
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("Open3D visualization:")
    print(f"  sampled points: {len(idx):,} / {n:,}")
    print(f"  coordinates are centered at easting={x0:.3f}, northing={y0:.3f}, refh={z0:.3f}")
    print(f"  vertical exaggeration: {cfg.open3d_vertical_exaggeration}")
    print("  color mode:", mode)
    o3d.visualization.draw_geometries([pcd], window_name="CASALS refh filter view")


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    # -------------------------------------------------------------------------
    # Edit parameters here. No argparse is used by design.
    # -------------------------------------------------------------------------
    cfg = Config(
        h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
        point_cloud_dir=Path(r"./point_cloud_data/filter_refh_points"),
        output_dir=Path(r"./outputs/filter_refh_points"),

        # Input-level filtering: keep broad for raw/reference product.
        filter_good_snr_only=True,
        refh_snr_min_for_input=None,
        track_range=None,
        sweep_range=None,

        # Tunable signal thresholds. Start conservative; adjust and rerun.
        use_snr_amp_noise_label=True,
        snr_hard_min=1.35,
        snr_soft_min=1.80,
        amp_low_percentile=5.0,
        low_signal_requires_low_amp_for_soft_snr=True,

        # Use per-pulse threshold if H5 provides it.
        use_refh_threshold_if_available=True,
        refh_threshold_margin_min=0.0,

        # Use optional error fields only if you set thresholds.
        use_error_fields_if_available=True,
        max_refh_error_m=None,
        max_refh_horizontal_error_deg=None,

        # Useful for quick visualization, but keep False for formal raw products.
        use_global_z_percentile_guard=False,
        z_low_percentile=0.05,
        z_high_percentile=99.95,

        # Main spatial consistency filter.
        use_local_height_filter=True,
        local_grid_cell_size_m=5.0,
        local_min_points_per_cell=12,
        local_abs_residual_threshold_m=25.0,
        local_mad_multiplier=8.0,
        local_min_sigma_m=0.75,

        # Optional Open3D SOR final labeling. Disabled by default for 3.6M points.
        use_open3d_statistical_outlier=False,
        open3d_sor_nb_neighbors=20,
        open3d_sor_std_ratio=2.5,
        open3d_sor_max_points=750_000,
        allow_sampled_sor_for_labeling=False,

        # Outputs.
        write_raw_las=True,
        write_noise_labeled_las=True,
        write_clean_las=True,
        write_noise_only_las=False,
        write_metadata_json=True,
        write_preview_png=True,

        # Interactive visualization. Turn on after first successful run.
        visualize_open3d=False,
        open3d_visual_max_points=350_000,
        open3d_vertical_exaggeration=1.0,
        open3d_color_mode="classification",

        rgb_color_by="classification",
        preview_max_points=300_000,
        overwrite=True,
    )

    t0 = time.time()
    ensure_dir(cfg.point_cloud_dir)
    ensure_dir(cfg.output_dir)
    if not cfg.h5_path.exists():
        raise FileNotFoundError(f"H5 file does not exist: {cfg.h5_path}")

    print("=" * 80)
    print("CASALS L1B refh filtering and noise labeling")
    print("=" * 80)
    print(f"H5: {cfg.h5_path.resolve()}")
    print(f"Point-cloud directory: {cfg.point_cloud_dir.resolve()}")
    print(f"Output directory: {cfg.output_dir.resolve()}")
    print()

    print("Reading L1B reference-return fields...")
    data = read_refh_point_data(cfg)
    print(f"  input records: {data.n_input_records:,}")
    print(f"  valid records after input mask: {data.n_valid_records:,}")
    print(f"  start UTC: {data.attrs.get('start_utca', 'unknown')}")
    print(f"  end UTC:   {data.attrs.get('end_utca', 'unknown')}")
    print(f"  good_snr fraction valid: {data.input_mask_summary['good_snr_fraction_valid']:.6f}")
    print()

    print("Projecting lon/lat to UTM...")
    proj = project_lonlat_to_utm(data, cfg)
    print(f"  output CRS: EPSG:{proj.utm_epsg}, {proj.utm_crs_name}")
    print(f"  inverse projection max horizontal error approx: {proj.projection_check['approx_max_horizontal_error_m']:.3e} m")
    print()

    print("Labeling likely noise with current parameters...")
    filt = label_noise(data, proj, cfg)
    print(json.dumps(filt.counts, indent=2))
    print()

    print("Coordinate and attribute summaries:")
    summaries = [
        summarize_array("longitude_deg", data.lon),
        summarize_array("latitude_deg", data.lat),
        summarize_array("easting_m", proj.easting),
        summarize_array("northing_m", proj.northing),
        summarize_array("refh_ellipsoidal_height_m", data.z_refh),
        summarize_array("refh_amp", data.refh_amp),
        summarize_array("refh_snr", data.refh_snr),
        summarize_array("track_num", data.track_num),
        summarize_array("sweep_num", data.sweep_num),
    ]
    for s in summaries:
        print(json.dumps(s, indent=2))
    print()

    stem = cfg.h5_path.stem
    suffix = f"epsg{proj.utm_epsg}"
    outputs: dict[str, Any] = {}

    all_mask = np.ones(len(data.z_refh), dtype=bool)
    raw_cls = np.ones(len(data.z_refh), dtype=np.uint8)
    labeled_cls = np.where(filt.noise_mask, 7, 1).astype(np.uint8)
    clean_mask = filt.keep_mask
    noise_only_mask = filt.noise_mask

    if cfg.write_raw_las:
        path = cfg.point_cloud_dir / f"{stem}_raw_refh_{suffix}.las"
        print(f"Writing raw LAS: {path}")
        outputs["raw_las"] = write_las(path, data, proj, cfg, all_mask, raw_cls, rgb_color_by="refh_amp")

    if cfg.write_noise_labeled_las:
        path = cfg.point_cloud_dir / f"{stem}_noise_labeled_refh_{suffix}.las"
        print(f"Writing noise-labeled LAS: {path}")
        outputs["noise_labeled_las"] = write_las(
            path,
            data,
            proj,
            cfg,
            all_mask,
            labeled_cls,
            reason_code=filt.reason_code,
            local_z_residual=filt.local_z_residual,
            local_median_z=filt.local_median_z,
            local_mad_z=filt.local_mad_z,
            rgb_color_by="classification",
        )

    if cfg.write_clean_las:
        path = cfg.point_cloud_dir / f"{stem}_clean_refh_{suffix}.las"
        print(f"Writing clean LAS: {path}")
        outputs["clean_las"] = write_las(
            path,
            data,
            proj,
            cfg,
            clean_mask,
            labeled_cls,
            reason_code=filt.reason_code,
            local_z_residual=filt.local_z_residual,
            local_median_z=filt.local_median_z,
            local_mad_z=filt.local_mad_z,
            rgb_color_by="refh_amp",
        )

    if cfg.write_noise_only_las:
        path = cfg.point_cloud_dir / f"{stem}_noise_only_refh_{suffix}.las"
        print(f"Writing noise-only LAS: {path}")
        outputs["noise_only_las"] = write_las(
            path,
            data,
            proj,
            cfg,
            noise_only_mask,
            labeled_cls,
            reason_code=filt.reason_code,
            local_z_residual=filt.local_z_residual,
            local_median_z=filt.local_median_z,
            local_mad_z=filt.local_mad_z,
            rgb_color_by="classification",
        )

    preview_outputs: dict[str, str] = {}
    if cfg.write_preview_png:
        print("Writing preview figures...")
        preview_outputs = make_debug_previews(cfg.output_dir, stem, data, proj, filt, cfg)
        outputs["preview_pngs"] = preview_outputs

    metadata = {
        "script": "filter_refh_points.py",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_h5": str(cfg.h5_path.resolve()),
        "casals_product_level": "L1B",
        "point_cloud_level": "Level-A max-Rx-bin / refh reference-return point cloud",
        "scientific_notes": [
            "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
            "refh is WGS84 ellipsoidal height unless otherwise documented.",
            "This is not an official multi-return point cloud.",
            "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            "Waveform peak detection beyond the official refh point is experimental or diagnostic only.",
        ],
        "method_scope": {
            "x_source": "refh_longitude projected from EPSG:4326 to inferred/overridden UTM",
            "y_source": "refh_latitude projected from EPSG:4326 to inferred/overridden UTM",
            "z_source": "refh, WGS84 ellipsoidal height",
            "waveform_decomposition": "not applied",
            "ground_classification": "not applied",
            "vertical_datum_conversion": "not applied",
            "noise_handling": "likely noise labeled as LAS class 7; raw points retained in raw output",
        },
        "config": asdict(cfg),
        "source_global_attributes_subset": {
            k: data.attrs.get(k)
            for k in (
                "tdms_file", "l1a_file", "ard_file", "geoloc_file",
                "start_utca", "end_utca", "n_pulses", "n_sweeps", "n_tracks", "n_rx_bins", "n_tx_bins",
            )
        },
        "input_mask_summary": data.input_mask_summary,
        "crs": {
            "input_horizontal_crs": "EPSG:4326 WGS84 geographic",
            "output_horizontal_crs_epsg": int(proj.utm_epsg),
            "output_horizontal_crs_name": proj.utm_crs_name,
            "z_height_convention": "WGS84 ellipsoidal height from CASALS refh",
        },
        "projection_validation": proj.projection_check,
        "filter_thresholds": filt.thresholds,
        "filter_counts": filt.counts,
        "noise_reason_count": filt.counts.get("counts_by_reason", {}),
        "summaries": summaries,
        "outputs": outputs,
        "runtime_seconds": float(time.time() - t0),
    }
    metadata_path = cfg.output_dir / f"{stem}_filter_metadata.json"
    if cfg.write_metadata_json:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        print(f"Metadata written: {metadata_path}")

    print()
    print("=" * 80)
    print("Done.")
    print("=" * 80)
    print(f"Keep points:  {filt.counts['n_keep']:,}")
    print(f"Noise points: {filt.counts['n_noise_labeled']:,}")
    print(f"Metadata: {metadata_path}")
    print("Reminder: Z is CASALS refh WGS84 ellipsoidal height; this is not a DEM or ground-classified product.")

    open3d_visualize(data, proj, filt, cfg)


if __name__ == "__main__":
    main()
