"""Classify CASALS L1B refh points with simple heuristic classes and Open3D.

Scientific meaning:
    Each point is one official CASALS L1B geolocated refh reference-return
    point, corresponding to the Rx waveform maximum-amplitude bin.

Outputs:
    Classification summary JSON, optional sampled PLY, and optional Open3D view.

This script does not:
    - create an official ground-classified point cloud,
    - create an official multi-return point cloud,
    - replace the formal CASALS refh geolocation definition.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import numpy as np
from pyproj import CRS, Transformer

from refh_noise import NoiseConfig, NoiseResult, label_noise_arrays


CONFIG = {
    "h5_path": Path("./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
    "output_dir": Path("./outputs/classify_refh_points_open3d"),
    "point_cloud_dir": Path("./point_cloud_data/classify_refh_points_open3d"),
    "metadata_name": "classification_summary.json",
    "sampled_ply_name": "classification_view_sample.ply",
    "input_filter": {
        "require_finite_xyz": True,
        "require_finite_snr": True,
        "require_finite_amp": False,
        "use_good_snr_only": False,
        "track_range": None,
        "sweep_range": None,
        "bbox_lonlat": None,
    },
    "ground": {
        "method": "local_low_surface",
        "snr_min": 5.0,
        "local_cell_size_m": 5.0,
        "local_low_percentile": 20.0,
        "min_points_per_cell": 3,
        "max_above_low_surface_m": 1.0,
        "max_below_low_surface_m": 2.0,
    },
    "target_class": {
        "name": "candidate_object",
        "snr_min": 4.0,
        "snr_max": None,
        "amp_min": None,
        "amp_max": None,
        "amp_percentile_min": 70.0,
        "bg_mean_min": None,
        "bg_mean_max": None,
        "bg_std_min": None,
        "bg_std_max": None,
        "height_above_ground_min_m": 0.1,
        "height_above_ground_max_m": 30,
        "amp_minus_bg_mean_min": None,
        "amp_minus_bg_mean_max": None,
        "amp_over_bg_mean_min": None,
        "amp_over_bg_mean_max": None,
    },
    "noise": {
        "use_snr_amp_noise_label": True,
        "snr_hard_min": 1.5,
        "snr_soft_min": 1.8,
        "amp_low_percentile": 5.0,
        "low_signal_requires_low_amp_for_soft_snr": True,
        "use_refh_threshold_if_available": True,
        "refh_threshold_margin_min": 0.0,
        "use_error_fields_if_available": True,
        "max_refh_error_m": None,
        "max_refh_horizontal_error_deg": None,
        "use_global_z_percentile_guard": False,
        "z_low_percentile": 0.05,
        "z_high_percentile": 99.95,
        "use_local_height_filter": True,
        "local_grid_cell_size_m": 5.0,
        "local_min_points_per_cell": 12,
        "local_abs_residual_threshold_m": 25.0,
        "local_mad_multiplier": 8.0,
        "local_min_sigma_m": 0.75,
        "use_open3d_statistical_outlier": False,
        "open3d_sor_nb_neighbors": 20,
        "open3d_sor_std_ratio": 2.5,
        "open3d_sor_max_points": 750_000,
        "open3d_sor_seed": 42,
        "allow_sampled_sor_for_labeling": False,
    },
    "projection": {
        "output_crs": None,
    },
    "visualization": {
        "enabled": True,
        "center_xy": True,
        "center_z": True,
        "z_scale": 1.0,
        "show_ground_default": True,
        "show_target_default": True,
        "show_noise_default": False,
        "show_other_default": True,
        "ground_color": (0.45, 0.28, 0.12),
        "target_color": (0.10, 0.95, 0.20),
        "noise_color": (1.00, 0.12, 0.02),
        "other_color": (0.55, 0.55, 0.55),
        "max_display_points_ground": 300_000,
        "max_display_points_target": 300_000,
        "max_display_points_noise": 200_000,
        "max_display_points_other": 200_000,
        "random_seed": 42,
        "point_size": 2.5,
        "show_coordinate_frame": True,
        "coordinate_frame_size_m": 50.0,
        "save_sampled_ply": False,
    },
}


@dataclass(frozen=True)
class InputFilterConfig:
    require_finite_xyz: bool = True
    require_finite_snr: bool = True
    require_finite_amp: bool = False
    use_good_snr_only: bool = False
    track_range: Optional[Tuple[int, int]] = None
    sweep_range: Optional[Tuple[int, int]] = None
    bbox_lonlat: Optional[Tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class GroundConfig:
    method: str = "local_low_surface"  # "snr_only" or "local_low_surface"
    snr_min: float = 5.0
    local_cell_size_m: float = 5.0
    local_low_percentile: float = 20.0
    min_points_per_cell: int = 3
    max_above_low_surface_m: float = 1.0
    max_below_low_surface_m: float = 2.0


@dataclass(frozen=True)
class TargetClassConfig:
    name: str = "candidate_object"
    snr_min: Optional[float] = 2.0
    snr_max: Optional[float] = None
    amp_min: Optional[float] = None
    amp_max: Optional[float] = None
    amp_percentile_min: Optional[float] = 70.0
    bg_mean_min: Optional[float] = None
    bg_mean_max: Optional[float] = None
    bg_std_min: Optional[float] = None
    bg_std_max: Optional[float] = None
    height_above_ground_min_m: Optional[float] = 1.5
    height_above_ground_max_m: Optional[float] = None
    amp_minus_bg_mean_min: Optional[float] = None
    amp_minus_bg_mean_max: Optional[float] = None
    amp_over_bg_mean_min: Optional[float] = None
    amp_over_bg_mean_max: Optional[float] = None


@dataclass(frozen=True)
class ProjectionConfig:
    output_crs: Optional[str] = None


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool = True
    center_xy: bool = True
    center_z: bool = True
    z_scale: float = 1.0
    show_ground_default: bool = True
    show_target_default: bool = True
    show_noise_default: bool = False
    show_other_default: bool = False
    ground_color: Tuple[float, float, float] = (0.45, 0.28, 0.12)
    target_color: Tuple[float, float, float] = (0.10, 0.75, 0.20)
    noise_color: Tuple[float, float, float] = (1.00, 0.12, 0.02)
    other_color: Tuple[float, float, float] = (0.55, 0.55, 0.55)
    max_display_points_ground: int = 300_000
    max_display_points_target: int = 300_000
    max_display_points_noise: int = 200_000
    max_display_points_other: int = 200_000
    random_seed: int = 42
    point_size: float = 2.0
    show_coordinate_frame: bool = True
    coordinate_frame_size_m: float = 50.0
    save_sampled_ply: bool = False


@dataclass(frozen=True)
class Config:
    h5_path: Path
    output_dir: Path
    point_cloud_dir: Path
    metadata_name: str = "classification_summary.json"
    sampled_ply_name: str = "classification_view_sample.ply"
    input_filter: InputFilterConfig = InputFilterConfig()
    ground: GroundConfig = GroundConfig()
    target_class: TargetClassConfig = TargetClassConfig()
    noise: NoiseConfig = NoiseConfig()
    projection: ProjectionConfig = ProjectionConfig()
    visualization: VisualizationConfig = VisualizationConfig()


@dataclass
class PointData:
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray
    snr: np.ndarray
    amp: Optional[np.ndarray]
    thres: Optional[np.ndarray]
    good_snr: Optional[np.ndarray]
    track_num: Optional[np.ndarray]
    sweep_num: Optional[np.ndarray]
    bg_mean: Optional[np.ndarray]
    bg_std: Optional[np.ndarray]
    refh_error: Optional[np.ndarray]
    refh_longitude_error: Optional[np.ndarray]
    refh_latitude_error: Optional[np.ndarray]
    delta_time: Optional[np.ndarray]
    attrs: dict[str, Any]


def config_from_dict(config_dict: dict[str, Any]) -> Config:
    return Config(
        h5_path=Path(config_dict["h5_path"]),
        output_dir=Path(config_dict["output_dir"]),
        point_cloud_dir=Path(config_dict["point_cloud_dir"]),
        metadata_name=str(config_dict.get("metadata_name", "classification_summary.json")),
        sampled_ply_name=str(config_dict.get("sampled_ply_name", "classification_view_sample.ply")),
        input_filter=InputFilterConfig(**config_dict.get("input_filter", {})),
        ground=GroundConfig(**config_dict.get("ground", {})),
        target_class=TargetClassConfig(**config_dict.get("target_class", {})),
        noise=NoiseConfig(**config_dict.get("noise", {})),
        projection=ProjectionConfig(**config_dict.get("projection", {})),
        visualization=VisualizationConfig(**config_dict.get("visualization", {})),
    )


def normalize_h5_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def find_dataset(h5: h5py.File, basename: str, required: bool = True) -> Optional[h5py.Dataset]:
    if basename in h5 and isinstance(h5[basename], h5py.Dataset):
        return h5[basename]

    matches: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == basename:
            matches.append(name)

    h5.visititems(visitor)

    if len(matches) == 1:
        return h5[matches[0]]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple datasets matched {basename!r}: {matches}")
    if required:
        raise KeyError(f"Required dataset not found: {basename}")
    return None


def read_1d_array(
    h5: h5py.File,
    basename: str,
    *,
    required: bool,
    dtype: Any,
    n_expected: Optional[int] = None,
) -> Optional[np.ndarray]:
    ds = find_dataset(h5, basename, required=required)
    if ds is None:
        return None
    arr = np.asarray(ds[...], dtype=dtype).reshape(-1)
    if n_expected is not None and arr.size != n_expected:
        raise ValueError(f"Dataset {basename!r} has size {arr.size}, expected {n_expected}.")
    return arr


def infer_utm_crs(lon: np.ndarray, lat: np.ndarray) -> CRS:
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    if not (-180.0 <= lon0 <= 180.0 and -90.0 <= lat0 <= 90.0):
        raise ValueError(f"Invalid lon/lat for UTM inference: {lon0}, {lat0}")
    zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
    zone = max(1, min(zone, 60))
    epsg = 32600 + zone if lat0 >= 0.0 else 32700 + zone
    return CRS.from_epsg(epsg)


def project_lonlat(lon: np.ndarray, lat: np.ndarray, output_crs: Optional[str]) -> tuple[np.ndarray, np.ndarray, CRS]:
    crs = infer_utm_crs(lon, lat) if output_crs is None else CRS.from_user_input(output_crs)
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), crs


def validate_inverse_projection(
    x: np.ndarray,
    y: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    crs: CRS,
    sample_size: int = 50_000,
    seed: int = 42,
) -> dict[str, float]:
    n = x.size
    if n == 0:
        return {
            "inverse_projection_sample_size": 0,
            "max_abs_lon_error_deg": float("nan"),
            "max_abs_lat_error_deg": float("nan"),
        }
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    inv = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon_back, lat_back = inv.transform(x[idx], y[idx])
    return {
        "inverse_projection_sample_size": int(idx.size),
        "max_abs_lon_error_deg": float(np.nanmax(np.abs(lon_back - lon[idx]))),
        "max_abs_lat_error_deg": float(np.nanmax(np.abs(lat_back - lat[idx]))),
    }


def get_refh_snr(amp: Optional[np.ndarray], thres: Optional[np.ndarray], snr: Optional[np.ndarray]) -> np.ndarray:
    if snr is not None:
        return snr.astype(np.float64)
    if amp is None or thres is None:
        raise KeyError("Neither refh_snr nor refh_amp/refh_thres is available.")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(amp.astype(np.float64), thres.astype(np.float64))


def read_h5_refh_points(h5_path: Path) -> PointData:
    with h5py.File(h5_path, "r") as h5:
        lon = read_1d_array(h5, "refh_longitude", required=True, dtype=np.float64)
        assert lon is not None
        n = lon.size
        lat = read_1d_array(h5, "refh_latitude", required=True, dtype=np.float64, n_expected=n)
        z = read_1d_array(h5, "refh", required=True, dtype=np.float64, n_expected=n)
        amp = read_1d_array(h5, "refh_amp", required=False, dtype=np.float64, n_expected=n)
        thres = read_1d_array(h5, "refh_thres", required=False, dtype=np.float64, n_expected=n)
        snr_stored = read_1d_array(h5, "refh_snr", required=False, dtype=np.float64, n_expected=n)
        good_snr = read_1d_array(h5, "good_snr", required=False, dtype=np.uint8, n_expected=n)
        track_num = read_1d_array(h5, "track_num", required=False, dtype=np.int64, n_expected=n)
        sweep_num = read_1d_array(h5, "sweep_num", required=False, dtype=np.int64, n_expected=n)
        bg_mean = read_1d_array(h5, "bg_mean", required=False, dtype=np.float64, n_expected=n)
        bg_std = read_1d_array(h5, "bg_std", required=False, dtype=np.float64, n_expected=n)
        refh_error = read_1d_array(h5, "refh_error", required=False, dtype=np.float64, n_expected=n)
        refh_longitude_error = read_1d_array(h5, "refh_longitude_error", required=False, dtype=np.float64, n_expected=n)
        refh_latitude_error = read_1d_array(h5, "refh_latitude_error", required=False, dtype=np.float64, n_expected=n)
        delta_time = read_1d_array(h5, "delta_time", required=False, dtype=np.float64, n_expected=n)
        attrs = {k: normalize_h5_attr(v) for k, v in h5.attrs.items()}

    assert lat is not None and z is not None
    snr = get_refh_snr(amp, thres, snr_stored)
    return PointData(
        lon=lon,
        lat=lat,
        z=z,
        snr=snr,
        amp=amp,
        thres=thres,
        good_snr=good_snr.astype(bool) if good_snr is not None else None,
        track_num=track_num,
        sweep_num=sweep_num,
        bg_mean=bg_mean,
        bg_std=bg_std,
        refh_error=refh_error,
        refh_longitude_error=refh_longitude_error,
        refh_latitude_error=refh_latitude_error,
        delta_time=delta_time,
        attrs=attrs,
    )


def build_input_mask(data: PointData, config: InputFilterConfig) -> np.ndarray:
    mask = np.ones(data.z.size, dtype=bool)

    if config.require_finite_xyz:
        mask &= np.isfinite(data.lon)
        mask &= np.isfinite(data.lat)
        mask &= np.isfinite(data.z)
        mask &= (data.lon >= -180.0) & (data.lon <= 180.0)
        mask &= (data.lat >= -90.0) & (data.lat <= 90.0)

    if config.require_finite_snr:
        mask &= np.isfinite(data.snr)

    if config.require_finite_amp and data.amp is not None:
        mask &= np.isfinite(data.amp)

    if config.use_good_snr_only and data.good_snr is not None:
        mask &= data.good_snr

    if config.track_range is not None and data.track_num is not None:
        lo, hi = config.track_range
        mask &= (data.track_num >= lo) & (data.track_num <= hi)

    if config.sweep_range is not None and data.sweep_num is not None:
        lo, hi = config.sweep_range
        mask &= (data.sweep_num >= lo) & (data.sweep_num <= hi)

    if config.bbox_lonlat is not None:
        lon_min, lat_min, lon_max, lat_max = config.bbox_lonlat
        mask &= (data.lon >= lon_min) & (data.lon <= lon_max)
        mask &= (data.lat >= lat_min) & (data.lat <= lat_max)

    return mask


def compute_cell_low_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    source_mask: np.ndarray,
    cell_size_m: float,
    low_percentile: float,
    min_points_per_cell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cell_size_m <= 0:
        raise ValueError("local_cell_size_m must be > 0.")
    if min_points_per_cell <= 0:
        raise ValueError("min_points_per_cell must be > 0.")
    if not np.any(source_mask):
        n = z.size
        return np.full(n, np.nan), np.empty(0, dtype=float), np.empty(0, dtype=np.int32)

    xmin = float(np.nanmin(x[source_mask]))
    ymin = float(np.nanmin(y[source_mask]))
    xmax = float(np.nanmax(x[source_mask]))
    ymax = float(np.nanmax(y[source_mask]))

    n_cols = int(np.ceil((xmax - xmin) / cell_size_m)) + 1
    n_rows = int(np.ceil((ymax - ymin) / cell_size_m)) + 1
    n_cells = n_rows * n_cols

    col = np.floor((x - xmin) / cell_size_m).astype(np.int64)
    row = np.floor((y - ymin) / cell_size_m).astype(np.int64)
    col = np.clip(col, 0, n_cols - 1)
    row = np.clip(row, 0, n_rows - 1)
    cell_id = row * n_cols + col

    source_idx = np.flatnonzero(source_mask)
    source_cell = cell_id[source_idx]
    source_z = z[source_idx]

    order = np.argsort(source_cell)
    source_cell_sorted = source_cell[order]
    source_z_sorted = source_z[order]

    cell_low = np.full(n_cells, np.nan, dtype=np.float64)
    cell_count = np.zeros(n_cells, dtype=np.int32)

    starts = np.r_[0, np.flatnonzero(np.diff(source_cell_sorted)) + 1]
    ends = np.r_[starts[1:], source_cell_sorted.size]
    for s, e in zip(starts, ends):
        cid = int(source_cell_sorted[s])
        count = int(e - s)
        cell_count[cid] = count
        if count >= min_points_per_cell:
            cell_low[cid] = float(np.nanpercentile(source_z_sorted[s:e], low_percentile))

    local_low = cell_low[cell_id]
    return local_low, cell_low, cell_count


def classify_ground(
    data: PointData,
    x: np.ndarray,
    y: np.ndarray,
    input_mask: np.ndarray,
    config: GroundConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    high_snr = input_mask & np.isfinite(data.snr) & (data.snr >= config.snr_min)
    if config.method == "snr_only":
        return high_snr.copy(), np.full(data.z.size, np.nan), {"high_snr_source_points": int(np.sum(high_snr))}
    if config.method != "local_low_surface":
        raise ValueError(f"Unsupported ground method: {config.method}")

    local_low, _, cell_count = compute_cell_low_surface(
        x=x,
        y=y,
        z=data.z,
        source_mask=high_snr,
        cell_size_m=config.local_cell_size_m,
        low_percentile=config.local_low_percentile,
        min_points_per_cell=config.min_points_per_cell,
    )

    residual = data.z - local_low
    ground_mask = high_snr & np.isfinite(local_low)
    ground_mask &= residual <= config.max_above_low_surface_m
    ground_mask &= residual >= -config.max_below_low_surface_m
    return ground_mask, local_low, {
        "high_snr_source_points": int(np.sum(high_snr)),
        "local_low_surface_supported_points": int(np.sum(np.isfinite(local_low) & input_mask)),
        "cells_with_support": int(np.sum(cell_count >= config.min_points_per_cell)) if cell_count.size else 0,
    }


def range_filter(values: Optional[np.ndarray], min_value: Optional[float], max_value: Optional[float]) -> Optional[np.ndarray]:
    if values is None:
        return None
    mask = np.isfinite(values)
    if min_value is not None:
        mask &= values >= min_value
    if max_value is not None:
        mask &= values <= max_value
    return mask


def classify_target(
    data: PointData,
    input_mask: np.ndarray,
    ground_mask: np.ndarray,
    local_low: np.ndarray,
    config: TargetClassConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate = input_mask & (~ground_mask)
    details: dict[str, Any] = {}

    snr_mask = range_filter(data.snr, config.snr_min, config.snr_max)
    if snr_mask is not None:
        candidate &= snr_mask

    if data.amp is not None:
        amp_min = config.amp_min
        if config.amp_percentile_min is not None:
            amp_valid = data.amp[input_mask & np.isfinite(data.amp)]
            if amp_valid.size > 0:
                amp_pct = float(np.nanpercentile(amp_valid, config.amp_percentile_min))
                details["amp_percentile_min_value"] = amp_pct
                amp_min = amp_pct if amp_min is None else max(amp_min, amp_pct)
        amp_mask = range_filter(data.amp, amp_min, config.amp_max)
        if amp_mask is not None:
            candidate &= amp_mask

    bg_mean_mask = range_filter(data.bg_mean, config.bg_mean_min, config.bg_mean_max)
    if bg_mean_mask is not None:
        candidate &= bg_mean_mask

    bg_std_mask = range_filter(data.bg_std, config.bg_std_min, config.bg_std_max)
    if bg_std_mask is not None:
        candidate &= bg_std_mask

    if config.height_above_ground_min_m is not None or config.height_above_ground_max_m is not None:
        if np.any(np.isfinite(local_low)):
            height_above_ground = data.z - local_low
            hag_mask = range_filter(height_above_ground, config.height_above_ground_min_m, config.height_above_ground_max_m)
            if hag_mask is not None:
                candidate &= hag_mask
                details["height_above_ground_filter_used"] = True
        else:
            warnings.warn("height_above_ground_* requested, but no finite local_low surface is available; filter ignored.")
            details["height_above_ground_filter_used"] = False

    if data.amp is not None and data.bg_mean is not None:
        amp_minus_bg = data.amp - data.bg_mean
        diff_mask = range_filter(amp_minus_bg, config.amp_minus_bg_mean_min, config.amp_minus_bg_mean_max)
        if diff_mask is not None:
            candidate &= diff_mask

        with np.errstate(divide="ignore", invalid="ignore"):
            amp_over_bg = np.divide(data.amp, data.bg_mean)
        ratio_mask = range_filter(amp_over_bg, config.amp_over_bg_mean_min, config.amp_over_bg_mean_max)
        if ratio_mask is not None:
            candidate &= ratio_mask

    return candidate, details


def build_noise_optional_fields(data: PointData, mask: np.ndarray) -> dict[str, np.ndarray]:
    optional: dict[str, np.ndarray] = {}
    for name, values in {
        "refh_thres": data.thres,
        "refh_error": data.refh_error,
        "refh_longitude_error": data.refh_longitude_error,
        "refh_latitude_error": data.refh_latitude_error,
    }.items():
        if values is not None:
            optional[name] = values[mask]
    return optional


def classify_noise(
    data: PointData,
    x: np.ndarray,
    y: np.ndarray,
    input_mask: np.ndarray,
    config: NoiseConfig,
) -> tuple[np.ndarray, np.ndarray, NoiseResult]:
    result = label_noise_arrays(
        z_refh=data.z[input_mask],
        refh_amp=data.amp[input_mask] if data.amp is not None else None,
        refh_snr=data.snr[input_mask],
        easting=x[input_mask],
        northing=y[input_mask],
        optional=build_noise_optional_fields(data, input_mask),
        cfg=config,
    )

    noise_mask = np.zeros(data.z.size, dtype=bool)
    reason_code = np.zeros(data.z.size, dtype=np.uint16)
    noise_mask[input_mask] = result.noise_mask
    reason_code[input_mask] = result.reason_code
    return noise_mask, reason_code, result


def sample_indices(mask: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size <= max_points:
        return idx
    return np.sort(rng.choice(idx, size=max_points, replace=False))


def make_o3d_cloud(points: np.ndarray, color: Tuple[float, float, float]):
    import open3d as o3d  # type: ignore

    pcd = o3d.geometry.PointCloud()
    if points.size == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    colors = np.tile(np.asarray(color, dtype=np.float64).reshape(1, 3), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def save_sampled_ply(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_mask: np.ndarray,
    target_mask: np.ndarray,
    noise_mask: np.ndarray,
    other_mask: np.ndarray,
    config: VisualizationConfig,
) -> Optional[Path]:
    if not config.save_sampled_ply:
        return None
    import open3d as o3d  # type: ignore

    rng = np.random.default_rng(config.random_seed)
    idx_ground = sample_indices(ground_mask, config.max_display_points_ground, rng)
    idx_target = sample_indices(target_mask, config.max_display_points_target, rng)
    idx_noise = sample_indices(noise_mask, config.max_display_points_noise, rng)
    idx_other = sample_indices(other_mask, config.max_display_points_other, rng)
    idx = np.r_[idx_ground, idx_target, idx_noise, idx_other]
    if idx.size == 0:
        return None

    x0 = float(np.nanmedian(x[idx])) if config.center_xy else 0.0
    y0 = float(np.nanmedian(y[idx])) if config.center_xy else 0.0
    z0 = float(np.nanmedian(z[idx])) if config.center_z else 0.0
    pts = np.column_stack([x[idx] - x0, y[idx] - y0, (z[idx] - z0) * config.z_scale])

    colors = np.tile(np.asarray(config.other_color, dtype=np.float64).reshape(1, 3), (idx.size, 1))
    colors[: idx_ground.size] = np.asarray(config.ground_color, dtype=np.float64)
    colors[idx_ground.size : idx_ground.size + idx_target.size] = np.asarray(config.target_color, dtype=np.float64)
    noise_start = idx_ground.size + idx_target.size
    colors[noise_start : noise_start + idx_noise.size] = np.asarray(config.noise_color, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)
    return path


def visualize_open3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_mask: np.ndarray,
    target_mask: np.ndarray,
    noise_mask: np.ndarray,
    other_mask: np.ndarray,
    config: VisualizationConfig,
) -> None:
    import open3d as o3d  # type: ignore

    rng = np.random.default_rng(config.random_seed)
    idx_ground = sample_indices(ground_mask, config.max_display_points_ground, rng)
    idx_target = sample_indices(target_mask, config.max_display_points_target, rng)
    idx_noise = sample_indices(noise_mask, config.max_display_points_noise, rng)
    idx_other = sample_indices(other_mask, config.max_display_points_other, rng)

    all_idx = np.r_[idx_ground, idx_target, idx_noise, idx_other]
    x0 = float(np.nanmedian(x[all_idx])) if config.center_xy and all_idx.size else 0.0
    y0 = float(np.nanmedian(y[all_idx])) if config.center_xy and all_idx.size else 0.0
    z0 = float(np.nanmedian(z[all_idx])) if config.center_z and all_idx.size else 0.0

    def make_points(idx: np.ndarray) -> np.ndarray:
        return np.column_stack([x[idx] - x0, y[idx] - y0, (z[idx] - z0) * config.z_scale]).astype(np.float64)

    ground_cloud = make_o3d_cloud(make_points(idx_ground), config.ground_color)
    target_cloud = make_o3d_cloud(make_points(idx_target), config.target_color)
    noise_cloud = make_o3d_cloud(make_points(idx_noise), config.noise_color)
    other_cloud = make_o3d_cloud(make_points(idx_other), config.other_color)

    print("\nOpen3D visualization")
    print("--------------------")
    print(f"Ground displayed: {idx_ground.size:,}")
    print(f"Target displayed: {idx_target.size:,}")
    print(f"Noise displayed:  {idx_noise.size:,}")
    print(f"Other displayed:  {idx_other.size:,}")
    print("Controls: G toggle ground, T toggle target, N toggle noise, O toggle other, Q/Esc quit")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="CASALS refh heuristic classification", width=1400, height=900)

    state = {"ground": False, "target": False, "noise": False, "other": False}
    first_added = {"done": False}

    def add_or_remove(name: str, geom, visible: bool) -> None:
        if visible and not state[name]:
            reset_bbox = not first_added["done"]
            vis.add_geometry(geom, reset_bounding_box=reset_bbox)
            state[name] = True
            first_added["done"] = True
        elif not visible and state[name]:
            vis.remove_geometry(geom, reset_bounding_box=False)
            state[name] = False

    add_or_remove("ground", ground_cloud, config.show_ground_default)
    add_or_remove("target", target_cloud, config.show_target_default)
    add_or_remove("noise", noise_cloud, config.show_noise_default)
    add_or_remove("other", other_cloud, config.show_other_default)

    if config.show_coordinate_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(config.coordinate_frame_size_m),
            origin=(0.0, 0.0, 0.0),
        )
        vis.add_geometry(frame, reset_bounding_box=not first_added["done"])
        first_added["done"] = True

    def toggle_ground(_vis) -> bool:
        add_or_remove("ground", ground_cloud, not state["ground"])
        _vis.update_renderer()
        return False

    def toggle_target(_vis) -> bool:
        add_or_remove("target", target_cloud, not state["target"])
        _vis.update_renderer()
        return False

    def toggle_noise(_vis) -> bool:
        add_or_remove("noise", noise_cloud, not state["noise"])
        _vis.update_renderer()
        return False

    def toggle_other(_vis) -> bool:
        add_or_remove("other", other_cloud, not state["other"])
        _vis.update_renderer()
        return False

    vis.register_key_callback(ord("G"), toggle_ground)
    vis.register_key_callback(ord("T"), toggle_target)
    vis.register_key_callback(ord("N"), toggle_noise)
    vis.register_key_callback(ord("O"), toggle_other)

    render_option = vis.get_render_option()
    render_option.point_size = float(config.point_size)
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])
    vis.poll_events()
    vis.update_renderer()
    try:
        vis.reset_view_point(True)
    except Exception:
        pass

    vis.run()
    vis.destroy_window()


def summarize_array(values: np.ndarray, mask: np.ndarray) -> Optional[dict[str, float]]:
    vals = values[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return {
        "min": float(np.nanmin(vals)),
        "p25": float(np.nanpercentile(vals, 25)),
        "p50": float(np.nanpercentile(vals, 50)),
        "p75": float(np.nanpercentile(vals, 75)),
        "max": float(np.nanmax(vals)),
    }


def build_summary(
    data: PointData,
    input_mask: np.ndarray,
    clean_mask: np.ndarray,
    noise_mask: np.ndarray,
    ground_mask: np.ndarray,
    target_mask: np.ndarray,
    other_mask: np.ndarray,
    crs: CRS,
    config: Config,
    projection_validation: dict[str, float],
    ground_details: dict[str, Any],
    target_details: dict[str, Any],
    noise_result: NoiseResult,
    sampled_ply_path: Optional[Path],
) -> dict[str, Any]:
    input_valid_count = int(np.sum(input_mask))
    clean_valid_count = int(np.sum(clean_mask))
    return {
        "script": "classify_refh_points_open3d.py",
        "source_h5": str(config.h5_path.resolve()),
        "output_crs": crs.to_string(),
        "output_crs_wkt": crs.to_wkt(),
        "projection_validation": projection_validation,
        "target_class_name": config.target_class.name,
        "scientific_notes": [
            "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
            "refh is WGS84 ellipsoidal height unless otherwise documented.",
            "This is not an official multi-return point cloud.",
            "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            "Noise/outlier is a QA layer excluded before ground/target heuristic classification.",
            "Ground and target classes here are heuristic exploratory classes only.",
        ],
        "counts": {
            "total_records": int(data.z.size),
            "invalid_input": int(data.z.size - input_valid_count),
            "input_valid": input_valid_count,
            "noise_outlier": int(np.sum(noise_mask)),
            "clean_valid": clean_valid_count,
            "ground": int(np.sum(ground_mask)),
            "target": int(np.sum(target_mask)),
            "other": int(np.sum(other_mask)),
        },
        "fractions_of_input_valid": {
            "noise_outlier": float(np.sum(noise_mask) / max(input_valid_count, 1)),
            "clean_valid": float(clean_valid_count / max(input_valid_count, 1)),
            "ground": float(np.sum(ground_mask) / max(input_valid_count, 1)),
            "target": float(np.sum(target_mask) / max(input_valid_count, 1)),
            "other": float(np.sum(other_mask) / max(input_valid_count, 1)),
        },
        "fractions_of_clean_valid": {
            "ground": float(np.sum(ground_mask) / max(clean_valid_count, 1)),
            "target": float(np.sum(target_mask) / max(clean_valid_count, 1)),
            "other": float(np.sum(other_mask) / max(clean_valid_count, 1)),
        },
        "ground_details": ground_details,
        "target_details": target_details,
        "noise_details": {
            "thresholds": noise_result.thresholds,
            "counts": noise_result.counts,
        },
        "noise_reason_counts": noise_result.counts.get("counts_by_reason", {}),
        "summaries": {
            "snr_valid": summarize_array(data.snr, input_mask),
            "snr_clean_valid": summarize_array(data.snr, clean_mask),
            "snr_noise": summarize_array(data.snr, noise_mask),
            "snr_ground": summarize_array(data.snr, ground_mask),
            "snr_target": summarize_array(data.snr, target_mask),
            "refh_valid": summarize_array(data.z, input_mask),
            "refh_clean_valid": summarize_array(data.z, clean_mask),
            "refh_noise": summarize_array(data.z, noise_mask),
            "refh_ground": summarize_array(data.z, ground_mask),
            "refh_target": summarize_array(data.z, target_mask),
            "amp_valid": summarize_array(data.amp, input_mask) if data.amp is not None else None,
            "amp_clean_valid": summarize_array(data.amp, clean_mask) if data.amp is not None else None,
            "amp_noise": summarize_array(data.amp, noise_mask) if data.amp is not None else None,
            "amp_ground": summarize_array(data.amp, ground_mask) if data.amp is not None else None,
            "amp_target": summarize_array(data.amp, target_mask) if data.amp is not None else None,
        },
        "source_global_attributes_subset": {
            k: data.attrs.get(k)
            for k in ("start_utca", "end_utca", "n_pulses", "n_sweeps", "n_tracks", "n_rx_bins")
        },
        "config": asdict(config),
        "outputs": {
            "metadata_json": str((config.output_dir / config.metadata_name).resolve()),
            "sampled_ply": str(sampled_ply_path.resolve()) if sampled_ply_path is not None else None,
        },
    }


def main() -> None:
    config = config_from_dict(CONFIG)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.point_cloud_dir.mkdir(parents=True, exist_ok=True)
    if not config.h5_path.exists():
        raise FileNotFoundError(f"Input H5 does not exist: {config.h5_path}")

    print(f"Reading H5: {config.h5_path}")
    data = read_h5_refh_points(config.h5_path)

    print("Building input mask...")
    input_mask = build_input_mask(data, config.input_filter)
    print(f"Input-valid points: {np.sum(input_mask):,} / {data.z.size:,}")
    if not np.any(input_mask):
        raise RuntimeError("No points survived input filtering.")

    print("Projecting lon/lat to local metric CRS...")
    x, y, crs = project_lonlat(data.lon[input_mask], data.lat[input_mask], config.projection.output_crs)
    x_all = np.full(data.z.size, np.nan, dtype=np.float64)
    y_all = np.full(data.z.size, np.nan, dtype=np.float64)
    x_all[input_mask] = x
    y_all[input_mask] = y
    projection_validation = validate_inverse_projection(x, y, data.lon[input_mask], data.lat[input_mask], crs)

    print("Labeling noise/outlier points...")
    noise_mask, _noise_reason_code, noise_result = classify_noise(data, x_all, y_all, input_mask, config.noise)
    clean_mask = input_mask & (~noise_mask)
    print(f"Noise/outlier points: {np.sum(noise_mask):,}")
    print(f"Clean valid points: {np.sum(clean_mask):,}")
    if not np.any(clean_mask):
        raise RuntimeError("No clean points remained after noise/outlier labeling.")

    print("Classifying tentative ground points...")
    ground_mask, local_low, ground_details = classify_ground(data, x_all, y_all, clean_mask, config.ground)
    print(f"Ground candidates: {np.sum(ground_mask):,}")

    print(f"Classifying target non-ground class: {config.target_class.name}")
    target_mask, target_details = classify_target(data, clean_mask, ground_mask, local_low, config.target_class)
    print(f"Target-class points: {np.sum(target_mask):,}")

    other_mask = clean_mask & (~ground_mask) & (~target_mask)
    print(f"Other clean valid points: {np.sum(other_mask):,}")

    partition_count = int(np.sum(noise_mask) + np.sum(ground_mask) + np.sum(target_mask) + np.sum(other_mask))
    if partition_count != int(np.sum(input_mask)):
        raise RuntimeError(
            "Classification partition mismatch: "
            f"noise + ground + target + other = {partition_count}, input_valid = {int(np.sum(input_mask))}."
        )

    sampled_ply_path = save_sampled_ply(
        config.point_cloud_dir / config.sampled_ply_name,
        x_all,
        y_all,
        data.z,
        ground_mask,
        target_mask,
        noise_mask,
        other_mask,
        config.visualization,
    )

    summary = build_summary(
        data=data,
        input_mask=input_mask,
        clean_mask=clean_mask,
        noise_mask=noise_mask,
        ground_mask=ground_mask,
        target_mask=target_mask,
        other_mask=other_mask,
        crs=crs,
        config=config,
        projection_validation=projection_validation,
        ground_details=ground_details,
        target_details=target_details,
        noise_result=noise_result,
        sampled_ply_path=sampled_ply_path,
    )
    metadata_path = config.output_dir / config.metadata_name
    metadata_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote metadata: {metadata_path}")

    if config.visualization.enabled:
        try:
            visualize_open3d(
                x=x_all,
                y=y_all,
                z=data.z,
                ground_mask=ground_mask,
                target_mask=target_mask,
                noise_mask=noise_mask,
                other_mask=other_mask,
                config=config.visualization,
            )
        except Exception as exc:
            warnings.warn(f"Open3D visualization failed or was skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
