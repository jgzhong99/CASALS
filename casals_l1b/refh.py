"""Read, project, export, and label official CASALS L1B ``refh`` points."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import laspy
import numpy as np
from laspy import ExtraBytesParams
from pyproj import CRS

from .geo import infer_wgs84_utm_epsg, transform_lonlat_to_projected, validate_inverse_projection
from .h5 import (
    find_dataset,
    read_global_attributes,
    read_optional_array,
    read_root_1d,
    require_dataset,
)
from .noise import NoiseResult


REQUIRED_REFH_FIELDS = (
    "refh_longitude",
    "refh_latitude",
    "refh",
    "refh_amp",
    "refh_snr",
    "good_snr",
    "track_num",
    "sweep_num",
)
DEFAULT_OPTIONAL_REFH_FIELDS = (
    "delta_time",
    "refh_thres",
    "bg_mean",
    "bg_std",
    "refh_error",
    "refh_longitude_error",
    "refh_latitude_error",
)


@dataclass(frozen=True)
class RefhSelection:
    filter_good_snr_only: bool = False
    refh_snr_min: Optional[float] = None
    track_range: Optional[tuple[int, int]] = None
    sweep_range: Optional[tuple[int, int]] = None


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


@dataclass
class ProjectedRefh:
    easting: np.ndarray
    northing: np.ndarray
    utm_epsg: int
    utm_crs_name: str
    projection_check: dict[str, Any]


@dataclass
class RefhSurfaceData:
    """Unfiltered refh fields used by the DSM and tentative DTM workflows."""

    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray
    snr: np.ndarray
    amp: np.ndarray
    thres: np.ndarray
    good_snr: np.ndarray
    track_num: Optional[np.ndarray]
    sweep_num: Optional[np.ndarray]
    pulse_index: np.ndarray
    attrs: dict[str, Any]


def build_valid_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    z: np.ndarray,
    refh_amp: np.ndarray,
    refh_snr: np.ndarray,
    good_snr: np.ndarray,
    track_num: np.ndarray,
    sweep_num: np.ndarray,
    selection: RefhSelection,
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
    if selection.filter_good_snr_only:
        mask &= np.asarray(good_snr, dtype=bool)
    if selection.refh_snr_min is not None:
        mask &= np.asarray(refh_snr, dtype=np.float64) >= float(selection.refh_snr_min)
    if selection.track_range is not None:
        low, high = selection.track_range
        mask &= (track_num >= low) & (track_num <= high)
    if selection.sweep_range is not None:
        low, high = selection.sweep_range
        mask &= (sweep_num >= low) & (sweep_num <= high)
    return mask


def read_refh_points(
    h5_path: Path,
    *,
    selection: RefhSelection = RefhSelection(),
    optional_fields: tuple[str, ...] = DEFAULT_OPTIONAL_REFH_FIELDS,
    ignore_bad_optional_fields: bool = True,
    ignore_optional_size_mismatch: bool = True,
) -> RefhPointData:
    """Read the shared root-level refh contract used by export and filtering.

    ``ignore_bad_optional_fields`` preserves the filter workflow's historical
    warning-and-skip behavior. Export can set it false for its strict delta_time
    read, matching its original contract.
    """
    with h5py.File(h5_path, "r") as h5:
        attrs = read_global_attributes(h5)
        required = {name: read_root_1d(h5, name) for name in REQUIRED_REFH_FIELDS}
        optional_all: dict[str, np.ndarray] = {}
        for name in optional_fields:
            if name not in h5:
                continue
            try:
                optional_all[name] = read_root_1d(h5, name)
            except (KeyError, ValueError) as exc:
                if not ignore_bad_optional_fields:
                    raise
                warnings.warn(
                    f"Optional dataset {name!r} was not loaded: {type(exc).__name__}: {exc}",
                    stacklevel=2,
                )

    arrays = {
        "refh_longitude": np.asarray(required["refh_longitude"], dtype=np.float64),
        "refh_latitude": np.asarray(required["refh_latitude"], dtype=np.float64),
        "refh": np.asarray(required["refh"], dtype=np.float64),
        "refh_amp": np.asarray(required["refh_amp"], dtype=np.float64),
        "refh_snr": np.asarray(required["refh_snr"], dtype=np.float64),
        "good_snr": np.asarray(required["good_snr"], dtype=bool),
        "track_num": required["track_num"],
        "sweep_num": required["sweep_num"],
    }
    shapes = {name: values.shape for name, values in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Required refh arrays have inconsistent shapes: {shapes}")

    n_input = len(arrays["refh_longitude"])
    if "n_pulses" in attrs and int(attrs["n_pulses"]) != n_input:
        raise ValueError(f"HDF5 n_pulses={attrs['n_pulses']} but refh length={n_input}")
    optional: dict[str, np.ndarray] = {}
    for name, values in optional_all.items():
        if len(values) != n_input:
            if ignore_optional_size_mismatch:
                warnings.warn(
                    f"Optional dataset {name!r} has size {len(values)}, expected {n_input}; ignored.",
                    stacklevel=2,
                )
                continue
            raise ValueError(f"Optional dataset {name!r} has size {len(values)}, expected {n_input}.")
        optional[name] = values

    mask = build_valid_mask(
        arrays["refh_longitude"],
        arrays["refh_latitude"],
        arrays["refh"],
        arrays["refh_amp"],
        arrays["refh_snr"],
        arrays["good_snr"],
        arrays["track_num"],
        arrays["sweep_num"],
        selection,
    )
    n_valid = int(np.sum(mask))
    if n_valid == 0:
        raise RuntimeError("No valid refh points after input validity filtering.")

    optional = {name: values[mask] for name, values in optional.items()}
    good_snr = arrays["good_snr"]
    summary = {
        "n_input_records": int(n_input),
        "n_valid_records": n_valid,
        "filter_good_snr_only": bool(selection.filter_good_snr_only),
        "refh_snr_min": selection.refh_snr_min,
        "track_range": selection.track_range,
        "sweep_range": selection.sweep_range,
        "good_snr_fraction_input": float(np.mean(good_snr)),
        "good_snr_fraction_valid": float(np.mean(good_snr[mask])),
    }
    return RefhPointData(
        lon=arrays["refh_longitude"][mask],
        lat=arrays["refh_latitude"][mask],
        z_refh=arrays["refh"][mask],
        refh_amp=arrays["refh_amp"][mask],
        refh_snr=arrays["refh_snr"][mask],
        good_snr=good_snr[mask],
        track_num=arrays["track_num"][mask],
        sweep_num=arrays["sweep_num"][mask],
        pulse_index=np.arange(n_input, dtype=np.uint32)[mask],
        optional=optional,
        attrs=attrs,
        n_input_records=n_input,
        n_valid_records=n_valid,
        input_mask_summary=summary,
    )


def read_refh_surface_data(h5_path: Path) -> RefhSurfaceData:
    """Read the full, unfiltered point arrays used by DSM and DTM workflows.

    Unlike :func:`read_refh_points`, this preserves invalid rows so each
    surface algorithm can apply its established input mask and retain pulse
    indexing exactly as before.
    """
    with h5py.File(h5_path, "r") as h5:
        lon = np.asarray(require_dataset(h5, "refh_longitude")[...], dtype=np.float64).reshape(-1)
        lat = np.asarray(require_dataset(h5, "refh_latitude")[...], dtype=np.float64).reshape(-1)
        z = np.asarray(require_dataset(h5, "refh")[...], dtype=np.float64).reshape(-1)
        amp = np.asarray(require_dataset(h5, "refh_amp")[...], dtype=np.float64).reshape(-1)

        thres_ds = find_dataset(h5, "refh_thres")
        thres = (
            np.asarray(thres_ds[...], dtype=np.float64).reshape(-1)
            if thres_ds is not None
            else np.full(lon.shape, np.nan, dtype=np.float64)
        )
        snr_ds = find_dataset(h5, "refh_snr")
        if snr_ds is not None:
            snr = np.asarray(snr_ds[...], dtype=np.float64).reshape(-1)
        elif thres_ds is not None:
            snr = np.divide(amp, thres, out=np.full_like(amp, np.nan), where=(thres != 0))
        else:
            raise KeyError("Neither refh_snr nor refh_thres was found; cannot compute SNR.")

        good_snr = read_optional_array(h5, "good_snr", lon.size)
        good_snr = good_snr.astype(bool) if good_snr is not None else (snr >= 5.0)
        track_num = read_optional_array(h5, "track_num", lon.size)
        sweep_num = read_optional_array(h5, "sweep_num", lon.size)
        attrs = read_global_attributes(h5)

    sizes = {
        "lon": lon.size,
        "lat": lat.size,
        "z": z.size,
        "snr": snr.size,
        "amp": amp.size,
        "thres": thres.size,
        "good_snr": good_snr.size,
    }
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Required datasets do not have matching sizes: {sizes}")

    return RefhSurfaceData(
        lon=lon,
        lat=lat,
        z=z,
        snr=snr,
        amp=amp,
        thres=thres,
        good_snr=good_snr,
        track_num=track_num,
        sweep_num=sweep_num,
        pulse_index=np.arange(lon.size, dtype=np.uint32),
        attrs=attrs,
    )


def summarize_refh_array(name: str, values: np.ndarray) -> dict[str, Any]:
    """Return the shared finite-value summary used by refh export/filter."""
    array = np.asarray(values)
    try:
        numeric = array.astype(np.float64, copy=False)
    except (TypeError, ValueError):
        return {"name": name, "n": int(array.size), "dtype": str(array.dtype), "summary": "non_numeric"}
    finite = np.isfinite(numeric)
    if not np.any(finite):
        return {
            "name": name,
            "n": int(array.size),
            "n_finite": 0,
            "min": None,
            "p02": None,
            "p50": None,
            "p98": None,
            "max": None,
        }
    percentiles = np.nanpercentile(numeric[finite], [2, 50, 98])
    return {
        "name": name,
        "n": int(array.size),
        "n_finite": int(np.sum(finite)),
        "min": float(np.nanmin(numeric[finite])),
        "p02": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p98": float(percentiles[2]),
        "max": float(np.nanmax(numeric[finite])),
    }


def summarize_surface_array(
    values: np.ndarray,
    percentiles: tuple[float, ...] = (0, 1, 2, 5, 50, 95, 98, 99, 100),
) -> dict[str, Any]:
    """Return the identical array summary used by DSM and tentative DTM."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"n": int(array.size), "n_finite": 0}
    return {
        "n": int(array.size),
        "n_finite": int(finite.size),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "mean": float(np.nanmean(finite)),
        "std": float(np.nanstd(finite)),
        "percentiles": {
            str(percentile): float(np.nanpercentile(finite, percentile))
            for percentile in percentiles
        },
    }


def project_refh_points(
    data: RefhPointData,
    utm_epsg_override: Optional[int] = None,
    *,
    seed: int = 42,
) -> ProjectedRefh:
    epsg = int(utm_epsg_override) if utm_epsg_override is not None else infer_wgs84_utm_epsg(data.lon, data.lat)
    x, y, crs = transform_lonlat_to_projected(data.lon, data.lat, epsg)
    check = validate_inverse_projection(x, y, data.lon, data.lat, epsg, seed=seed)
    return ProjectedRefh(x, y, epsg, crs.name, check)


def _color_values(
    values: np.ndarray,
    percentiles: tuple[float, float],
    cmap_name: str = "viridis",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        low, high = 0.0, 1.0
    else:
        low, high = (float(v) for v in np.nanpercentile(values[finite], percentiles))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.nanmin(values[finite]))
            high = float(np.nanmax(values[finite]))
            if high <= low:
                high = low + 1.0
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    rgb = np.round(plt.get_cmap(cmap_name)(normalized)[:, :3] * 65535.0).astype(np.uint16)
    return rgb[:, 0], rgb[:, 1], rgb[:, 2], (low, high)


def _optional_extra_dims(
    optional: dict[str, np.ndarray], descriptions: Mapping[str, str] | None = None
) -> list[ExtraBytesParams]:
    dims = []
    for name, values in optional.items():
        safe_name = name[:32]
        dtype = np.asarray(values).dtype
        if dtype.kind in "iu":
            point_type = np.uint32 if dtype.itemsize > 2 else np.uint16 if dtype.itemsize > 1 else np.uint8
        else:
            point_type = np.float64
        default_description = "delta_time_sec" if name == "delta_time" else safe_name[:31]
        description = (descriptions or {}).get(name, default_description)
        dims.append(ExtraBytesParams(name=safe_name, type=point_type, description=description[:31]))
    return dims


def write_refh_las(
    path: Path,
    data: RefhPointData,
    projected: ProjectedRefh,
    *,
    mask: Optional[np.ndarray] = None,
    classification: Optional[np.ndarray] = None,
    noise: Optional[NoiseResult] = None,
    rgb_color_by: str = "refh_amp",
    color_percentiles: tuple[float, float] = (2.0, 98.0),
    xyz_scale_m: float = 0.001,
    include_optional_fields: bool = True,
    generating_software: str = "casals_l1b",
    track_num_description: str = "track_channel",
    pulse_index_description: str = "pulse_index",
    optional_descriptions: Mapping[str, str] | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Write a refh-derived LAS while retaining pulse order and source fields."""
    n_all = len(data.z_refh)
    selected = np.ones(n_all, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != (n_all,):
        raise ValueError(f"LAS selection mask has shape {selected.shape}; expected {(n_all,)}.")
    idx = np.flatnonzero(selected)
    if idx.size == 0:
        return {"path": str(path), "status": "skipped_no_points", "n_points": 0}
    if path.exists() and not overwrite:
        return {"path": str(path), "status": "exists_skipped", "n_points": None}
    path.parent.mkdir(parents=True, exist_ok=True)

    cls = np.ones(n_all, dtype=np.uint8) if classification is None else np.asarray(classification, dtype=np.uint8)
    if cls.shape != (n_all,):
        raise ValueError(f"Classification array has shape {cls.shape}; expected {(n_all,)}.")
    optional = {name: values for name, values in data.optional.items() if include_optional_fields}
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([xyz_scale_m] * 3, dtype=np.float64)
    header.offsets = np.array([
        math.floor(float(np.nanmin(projected.easting[idx]))),
        math.floor(float(np.nanmin(projected.northing[idx]))),
        math.floor(float(np.nanmin(data.z_refh[idx]))),
    ], dtype=np.float64)
    header.system_identifier = "CASALS_L1B_REFH"
    header.generating_software = generating_software
    header.add_crs(CRS.from_epsg(projected.utm_epsg))

    extra_dims = [
        ExtraBytesParams(name="longitude", type=np.float64, description="refh_lon_deg_WGS84"),
        ExtraBytesParams(name="latitude", type=np.float64, description="refh_lat_deg_WGS84"),
        ExtraBytesParams(name="refh_amp_raw", type=np.float64, description="refh_amp_raw_counts"),
        ExtraBytesParams(name="refh_snr", type=np.float64, description="refh_snr"),
        ExtraBytesParams(name="good_snr", type=np.uint8, description="good_snr_1_true"),
        ExtraBytesParams(name="track_num", type=np.uint16, description=track_num_description),
        ExtraBytesParams(name="sweep_num", type=np.uint32, description="sweep_number"),
        ExtraBytesParams(name="pulse_index", type=np.uint32, description=pulse_index_description),
    ]
    if noise is not None:
        extra_dims.extend([
            ExtraBytesParams(name="noise_reason", type=np.uint16, description="noise_reason_bits"),
            ExtraBytesParams(name="local_z_resid", type=np.float64, description="local_z_resid_m"),
            ExtraBytesParams(name="local_z_med", type=np.float64, description="local_z_median_m"),
            ExtraBytesParams(name="local_z_mad", type=np.float64, description="local_z_mad_m"),
        ])
    extra_dims.extend(_optional_extra_dims(optional, optional_descriptions))
    header.add_extra_dims(extra_dims)

    las = laspy.LasData(header)
    las.x = projected.easting[idx].astype(np.float64)
    las.y = projected.northing[idx].astype(np.float64)
    las.z = data.z_refh[idx].astype(np.float64)
    las.intensity = np.clip(data.refh_amp[idx], 0.0, 65535.0).astype(np.uint16)
    las.classification = cls[idx]
    values = {
        "refh": data.z_refh,
        "refh_snr": data.refh_snr,
        "refh_amp": data.refh_amp,
    }.get(rgb_color_by)
    if values is None:
        if rgb_color_by != "classification":
            raise ValueError(f"Unsupported RGB mode: {rgb_color_by!r}")
        rgb = np.zeros((len(idx), 3), dtype=np.uint16)
        keep = cls[idx] != 7
        rgb[keep] = np.array([15000, 36000, 65535], dtype=np.uint16)
        rgb[~keep] = np.array([65535, 9000, 5000], dtype=np.uint16)
        color_range = (0.0, 7.0)
    else:
        red, green, blue, color_range = _color_values(values[idx], color_percentiles)
        rgb = np.column_stack((red, green, blue))
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    las.longitude = data.lon[idx].astype(np.float64)
    las.latitude = data.lat[idx].astype(np.float64)
    las.refh_amp_raw = data.refh_amp[idx].astype(np.float64)
    las.refh_snr = data.refh_snr[idx].astype(np.float64)
    las.good_snr = data.good_snr[idx].astype(np.uint8)
    las.track_num = data.track_num[idx].astype(np.uint16)
    las.sweep_num = data.sweep_num[idx].astype(np.uint32)
    las.pulse_index = data.pulse_index[idx].astype(np.uint32)
    if noise is not None:
        las.noise_reason = noise.reason_code[idx].astype(np.uint16)
        las.local_z_resid = noise.local_z_residual[idx].astype(np.float64)
        las.local_z_med = noise.local_median_z[idx].astype(np.float64)
        las.local_z_mad = noise.local_mad_z[idx].astype(np.float64)
    for name, values in optional.items():
        if len(values) != n_all:
            continue
        setattr(las, name[:32], np.asarray(values[idx]))

    las.write(str(path))
    return {
        "path": str(path),
        "status": "written",
        "n_points": int(len(idx)),
        "bytes": int(path.stat().st_size),
        "classification_counts": {
            str(int(code)): int(count) for code, count in zip(*np.unique(cls[idx], return_counts=True))
        },
        "horizontal_crs_epsg": int(projected.utm_epsg),
        "z_convention": "CASALS refh WGS84 ellipsoidal height; no vertical datum conversion",
        "offsets": [float(v) for v in header.offsets],
        "rgb_color_by": rgb_color_by,
        "rgb_range": [float(color_range[0]), float(color_range[1])],
        "xyz_scale_m": float(xyz_scale_m),
    }
