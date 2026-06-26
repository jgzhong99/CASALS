#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CASALS beam-angle convention audit with broad convention search and H5 height sanity checks.

For a selected CASALS L1B sweep, this diagnostic script tests many interpretations of
local_beam_azimuth/local_beam_elevation. For each interpretation it builds 3-D beam
lines from either the H5 instrument point or the official refh point, checks whether the
lines converge, and, more importantly, checks whether the ray geometry is physically
consistent with H5 instrument position/altitude and refh/rwstart/rwstop heights.

This is diagnostic only. It validates official refh beam-line geometry. It does not, by
itself, establish full waveform-bin geolocation.

Typical run:
    python geolocate_sweep_bins_from_beam_angle_rules.py

Optional:
    python geolocate_sweep_bins_from_beam_angle_rules.py \
        --h5 ./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5 \
        --sweep 7040 \
        --output-root ./beam_line_origin \
        --expected-agl-bands 3500:6500,7500:9500
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

DEFAULT_H5_PATH = Path("./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5")
DEFAULT_SWEEP_INDEX = 7040
DEFAULT_OUTPUT_ROOT = Path("./outputs/beam_line_origin")

# These bands are not used as hard truth. They are only a sanity check against the
# two airborne CASALS operating-height regimes that appear in current CASALS materials:
# ~4.2-5.3 km and ~8.5 km above the surface.
DEFAULT_EXPECTED_AGL_BANDS = "3500:6500,7500:9500"

DATASET_CANDIDATES: Dict[str, Sequence[str]] = {
    "refh_lon": ("refh_longitude", "longitude", "lon"),
    "refh_lat": ("refh_latitude", "latitude", "lat"),
    "refh_z": ("refh", "refh_height", "height", "elevation"),
    "instrument_lon": ("instrument_longitude", "sensor_longitude", "platform_longitude"),
    "instrument_lat": ("instrument_latitude", "sensor_latitude", "platform_latitude"),
    "instrument_z": ("instrument_altitude", "instrument_height", "sensor_altitude", "platform_altitude"),
    "instrument_z_error": ("instrument_altitude_error", "instrument_height_error", "sensor_altitude_error"),
    "beam_az": ("local_beam_azimuth", "beam_azimuth", "azimuth"),
    "beam_el": ("local_beam_elevation", "beam_elevation", "elevation"),
    "beam_az_error": ("local_beam_azimuth_error", "beam_azimuth_error", "azimuth_error"),
    "beam_el_error": ("local_beam_elevation_error", "beam_elevation_error", "elevation_error"),
    "sweep_num": ("sweep_num", "sweep", "sweep_index"),
    "track_num": ("track_num", "track", "track_index"),
    "delta_time": ("delta_time", "time"),
    "bin_size": ("bin_size", "rx_bin_size", "range_bin_size"),
    "geoid": ("geoid",),
    "geoid_free2mean": ("geoid_free2mean",),
    "rwstart_lon": ("rwstart_longitude", "range_window_start_longitude"),
    "rwstart_lat": ("rwstart_latitude", "range_window_start_latitude"),
    "rwstart_z": ("rwstart", "rwstart_height", "range_window_start_height"),
    "rwstop_lon": ("rwstop_longitude", "range_window_stop_longitude"),
    "rwstop_lat": ("rwstop_latitude", "range_window_stop_latitude"),
    "rwstop_z": ("rwstop", "rwstop_height", "range_window_stop_height"),
}

UNITS = ("radian", "degree")
AZ_ZERO_DIRECTIONS = ("north", "east", "south", "west")
AZ_ROTATION_DIRECTIONS = ("clockwise", "counterclockwise")
EL_CONVENTIONS = (
    "elevation_from_horizon",
    "depression_from_horizon",
    "angle_from_up_vertical",
    "angle_from_down_vertical",
)
SIGNS = ("as_encoded", "negated")
BASES = ("instrument", "refh")
ANGLE_TRANSFORMS = (
    "as_named",
    "swap_az_el",
    "negate_az",
    "negate_el",
    "negate_both",
    "swap_and_negate_both",
)


def find_dataset_path(h5: h5py.File, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    """Find a dataset at root level or by basename anywhere in the H5 tree.

    Matching is case-insensitive for robustness, but the returned path preserves the
    exact H5 dataset name.
    """
    candidate_lower = {c.lower(): c for c in candidates}

    for name in candidates:
        if name in h5 and isinstance(h5[name], h5py.Dataset):
            return name

    basename_to_path: Dict[str, str] = {}

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            basename_to_path[name.split("/")[-1].lower()] = name

    h5.visititems(visitor)
    for lower_name in candidate_lower:
        if lower_name in basename_to_path:
            return basename_to_path[lower_name]

    if required:
        raise KeyError(f"Could not find any dataset from candidates: {list(candidates)}")
    return None


def read_1d(h5: h5py.File, logical_name: str, required: bool = True) -> Tuple[Optional[np.ndarray], Optional[str]]:
    path = find_dataset_path(h5, DATASET_CANDIDATES[logical_name], required=required)
    if path is None:
        return None, None
    arr = np.asarray(h5[path][...]).reshape(-1)
    return arr, path


def infer_wgs84_utm_epsg(lon: np.ndarray, lat: np.ndarray) -> int:
    lon_med = float(np.nanmedian(lon))
    lat_med = float(np.nanmedian(lat))
    if not (-180.0 <= lon_med <= 180.0 and -90.0 <= lat_med <= 90.0):
        raise ValueError(f"Cannot infer UTM zone from lon/lat median: lon={lon_med}, lat={lat_med}")
    zone = int(math.floor((lon_med + 180.0) / 6.0) + 1)
    zone = max(1, min(zone, 60))
    return (32600 if lat_med >= 0.0 else 32700) + zone


@dataclass
class SweepData:
    sweep_index: int
    record_indices: np.ndarray
    track_num: np.ndarray
    refh_lon: np.ndarray
    refh_lat: np.ndarray
    refh_z: np.ndarray
    instrument_lon: Optional[np.ndarray]
    instrument_lat: Optional[np.ndarray]
    instrument_z: Optional[np.ndarray]
    instrument_z_error: Optional[np.ndarray]
    beam_az: np.ndarray
    beam_el: np.ndarray
    beam_az_error: Optional[np.ndarray]
    beam_el_error: Optional[np.ndarray]
    delta_time: Optional[np.ndarray]
    bin_size: Optional[np.ndarray]
    geoid: Optional[np.ndarray]
    geoid_free2mean: Optional[np.ndarray]
    rwstart_lon: Optional[np.ndarray]
    rwstart_lat: Optional[np.ndarray]
    rwstart_z: Optional[np.ndarray]
    rwstop_lon: Optional[np.ndarray]
    rwstop_lat: Optional[np.ndarray]
    rwstop_z: Optional[np.ndarray]
    source_datasets: Dict[str, Optional[str]]

    @property
    def n(self) -> int:
        return int(self.record_indices.size)


def load_sweep_data(h5_path: Path, sweep_index: int) -> SweepData:
    with h5py.File(h5_path, "r") as h5:
        sweep_num, sweep_path = read_1d(h5, "sweep_num", required=True)
        track_num_all, track_path = read_1d(h5, "track_num", required=True)
        sweep_mask = np.asarray(sweep_num) == sweep_index
        record_indices = np.flatnonzero(sweep_mask)
        if record_indices.size == 0:
            raise RuntimeError(f"No records found for sweep_index={sweep_index}")

        track_sel = np.asarray(track_num_all[record_indices], dtype=np.int64)
        order = np.argsort(track_sel, kind="mergesort")
        record_indices = record_indices[order]
        track_sel = track_sel[order]

        unique_tracks, first_pos = np.unique(track_sel, return_index=True)
        duplicate_count = int(track_sel.size - unique_tracks.size)
        if duplicate_count > 0:
            keep = np.sort(first_pos)
            record_indices = record_indices[keep]
            track_sel = track_sel[keep]

        def read_selected(logical_name: str, required: bool = True) -> Tuple[Optional[np.ndarray], Optional[str]]:
            arr, path = read_1d(h5, logical_name, required=required)
            if arr is None:
                return None, path
            if arr.shape[0] <= int(np.max(record_indices)):
                raise ValueError(
                    f"Dataset {path} has length {arr.shape[0]}, but max selected index is {int(np.max(record_indices))}."
                )
            return np.asarray(arr[record_indices]), path

        selected: Dict[str, Tuple[Optional[np.ndarray], Optional[str]]] = {}
        for logical_name in DATASET_CANDIDATES:
            required = logical_name in {"refh_lon", "refh_lat", "refh_z", "beam_az", "beam_el", "sweep_num", "track_num"}
            if logical_name in {"sweep_num", "track_num"}:
                continue
            selected[logical_name] = read_selected(logical_name, required=required)

    def arr(name: str) -> Optional[np.ndarray]:
        return selected[name][0]

    source_datasets = {
        "sweep_num": sweep_path,
        "track_num": track_path,
        **{name: path for name, (_, path) in selected.items()},
        "duplicate_records_removed_for_same_track": str(duplicate_count),
    }

    return SweepData(
        sweep_index=int(sweep_index),
        record_indices=np.asarray(record_indices, dtype=np.int64),
        track_num=np.asarray(track_sel, dtype=np.int64),
        refh_lon=np.asarray(arr("refh_lon"), dtype=np.float64),
        refh_lat=np.asarray(arr("refh_lat"), dtype=np.float64),
        refh_z=np.asarray(arr("refh_z"), dtype=np.float64),
        instrument_lon=None if arr("instrument_lon") is None else np.asarray(arr("instrument_lon"), dtype=np.float64),
        instrument_lat=None if arr("instrument_lat") is None else np.asarray(arr("instrument_lat"), dtype=np.float64),
        instrument_z=None if arr("instrument_z") is None else np.asarray(arr("instrument_z"), dtype=np.float64),
        instrument_z_error=None if arr("instrument_z_error") is None else np.asarray(arr("instrument_z_error"), dtype=np.float64),
        beam_az=np.asarray(arr("beam_az"), dtype=np.float64),
        beam_el=np.asarray(arr("beam_el"), dtype=np.float64),
        beam_az_error=None if arr("beam_az_error") is None else np.asarray(arr("beam_az_error"), dtype=np.float64),
        beam_el_error=None if arr("beam_el_error") is None else np.asarray(arr("beam_el_error"), dtype=np.float64),
        delta_time=None if arr("delta_time") is None else np.asarray(arr("delta_time"), dtype=np.float64),
        bin_size=None if arr("bin_size") is None else np.asarray(arr("bin_size"), dtype=np.float64),
        geoid=None if arr("geoid") is None else np.asarray(arr("geoid"), dtype=np.float64),
        geoid_free2mean=None if arr("geoid_free2mean") is None else np.asarray(arr("geoid_free2mean"), dtype=np.float64),
        rwstart_lon=None if arr("rwstart_lon") is None else np.asarray(arr("rwstart_lon"), dtype=np.float64),
        rwstart_lat=None if arr("rwstart_lat") is None else np.asarray(arr("rwstart_lat"), dtype=np.float64),
        rwstart_z=None if arr("rwstart_z") is None else np.asarray(arr("rwstart_z"), dtype=np.float64),
        rwstop_lon=None if arr("rwstop_lon") is None else np.asarray(arr("rwstop_lon"), dtype=np.float64),
        rwstop_lat=None if arr("rwstop_lat") is None else np.asarray(arr("rwstop_lat"), dtype=np.float64),
        rwstop_z=None if arr("rwstop_z") is None else np.asarray(arr("rwstop_z"), dtype=np.float64),
        source_datasets=source_datasets,
    )


def project_lonlat(lon: np.ndarray, lat: np.ndarray, epsg: int) -> Tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def inverse_project_xy(x: float, y: float, epsg: int) -> Tuple[float, float]:
    transformer = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(float(x), float(y))
    return float(lon), float(lat)


def finite_rows(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("finite_rows requires at least one array")
    mask = np.ones(np.asarray(arrays[0]).shape[0], dtype=bool)
    for arr in arrays:
        a = np.asarray(arr)
        if a.ndim == 1:
            mask &= np.isfinite(a)
        else:
            mask &= np.all(np.isfinite(a), axis=1)
    return mask


def normalize_vectors(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1)
    out = np.full_like(v, np.nan, dtype=np.float64)
    valid = np.isfinite(norms) & (norms > 0.0)
    out[valid] = v[valid] / norms[valid, None]
    return out


def line_point_distances(points: np.ndarray, line_points: np.ndarray, directions: np.ndarray) -> np.ndarray:
    w = points - line_points
    proj = np.sum(w * directions, axis=1)
    perp = w - proj[:, None] * directions
    return np.linalg.norm(perp, axis=1)


def signed_along_line(points: np.ndarray, line_points: np.ndarray, directions: np.ndarray) -> np.ndarray:
    return np.sum((points - line_points) * directions, axis=1)


def fit_common_origin(line_points: np.ndarray, directions: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """Least-squares closest point to a bundle of 3-D lines."""
    eye = np.eye(3, dtype=np.float64)
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for p, u in zip(line_points, directions):
        M = eye - np.outer(u, u)
        A += M
        b += M @ p
    try:
        cond = float(np.linalg.cond(A))
    except Exception:
        cond = np.inf
    try:
        origin = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        origin = np.linalg.lstsq(A, b, rcond=None)[0]
    solved = bool(np.all(np.isfinite(origin)))
    return origin.astype(np.float64), cond, solved


def q(values: np.ndarray, p: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, p))


def median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return np.nan if arr.size == 0 else float(np.median(arr))


def robust_nmad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    return float(1.4826 * np.median(np.abs(arr - med)))


@dataclass(frozen=True)
class AngleRule:
    unit: str
    az_zero: str
    az_rotation: str
    el_convention: str
    sign: str
    basis: str
    angle_transform: str
    az_bias_deg: float = 0.0
    el_bias_deg: float = 0.0
    refinement_stage: str = "base"

    @property
    def key(self) -> str:
        return (
            f"{self.unit}|az0={self.az_zero}_{self.az_rotation}|el={self.el_convention}|"
            f"sign={self.sign}|basis={self.basis}|transform={self.angle_transform}|"
            f"azbias={self.az_bias_deg:+.4f}deg|elbias={self.el_bias_deg:+.4f}deg|stage={self.refinement_stage}"
        )


def iter_base_angle_rules() -> Iterable[AngleRule]:
    for unit in UNITS:
        for az_zero in AZ_ZERO_DIRECTIONS:
            for az_rotation in AZ_ROTATION_DIRECTIONS:
                for el_convention in EL_CONVENTIONS:
                    for sign in SIGNS:
                        for basis in BASES:
                            for transform in ANGLE_TRANSFORMS:
                                yield AngleRule(unit, az_zero, az_rotation, el_convention, sign, basis, transform)


def parse_float_list(text: str) -> List[float]:
    vals: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def parse_agl_bands(text: str) -> List[Tuple[float, float]]:
    bands: List[Tuple[float, float]] = []
    text = str(text or "").strip()
    if not text:
        return bands
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Bad AGL band token {token!r}; expected min:max")
        lo_s, hi_s = token.split(":", 1)
        lo = float(lo_s)
        hi = float(hi_s)
        if hi < lo:
            lo, hi = hi, lo
        bands.append((lo, hi))
    return bands


def nearest_band_error(value: float, bands: Sequence[Tuple[float, float]]) -> Tuple[float, str]:
    if not np.isfinite(value) or not bands:
        return np.nan, ""
    best_error = np.inf
    best_label = ""
    for lo, hi in bands:
        if lo <= value <= hi:
            error = 0.0
        else:
            error = min(abs(value - lo), abs(value - hi))
        if error < best_error:
            best_error = error
            best_label = f"{lo:g}:{hi:g}"
    return float(best_error), best_label


def apply_angle_transform(az: np.ndarray, el: np.ndarray, transform: str) -> Tuple[np.ndarray, np.ndarray]:
    if transform == "as_named":
        return az, el
    if transform == "swap_az_el":
        return el, az
    if transform == "negate_az":
        return -az, el
    if transform == "negate_el":
        return az, -el
    if transform == "negate_both":
        return -az, -el
    if transform == "swap_and_negate_both":
        return -el, -az
    raise ValueError(f"Unsupported angle transform: {transform}")


def encoded_angles_to_radians(az: np.ndarray, el: np.ndarray, unit: str) -> Tuple[np.ndarray, np.ndarray]:
    if unit == "radian":
        return np.asarray(az, dtype=np.float64), np.asarray(el, dtype=np.float64)
    if unit == "degree":
        return np.deg2rad(np.asarray(az, dtype=np.float64)), np.deg2rad(np.asarray(el, dtype=np.float64))
    raise ValueError(f"Unsupported angle unit: {unit}")


def azimuth_to_east_north(az: np.ndarray, zero: str, rotation: str) -> Tuple[np.ndarray, np.ndarray]:
    # Convert all azimuths to the conventional north-clockwise frame.
    zero_to_north_clockwise_deg = {
        "north": 0.0,
        "east": 90.0,
        "south": 180.0,
        "west": 270.0,
    }
    if zero not in zero_to_north_clockwise_deg:
        raise ValueError(f"Unsupported azimuth zero direction: {zero}")
    zero_rad = math.radians(zero_to_north_clockwise_deg[zero])
    if rotation == "clockwise":
        a_ncw = zero_rad + az
    elif rotation == "counterclockwise":
        a_ncw = zero_rad - az
    else:
        raise ValueError(f"Unsupported azimuth rotation: {rotation}")
    east = np.sin(a_ncw)
    north = np.cos(a_ncw)
    return east, north


def direction_vectors_from_rule(az_raw: np.ndarray, el_raw: np.ndarray, rule: AngleRule) -> np.ndarray:
    az, el = encoded_angles_to_radians(az_raw, el_raw, rule.unit)
    az, el = apply_angle_transform(az, el, rule.angle_transform)
    az = az + math.radians(float(rule.az_bias_deg))
    el = el + math.radians(float(rule.el_bias_deg))

    east_h, north_h = azimuth_to_east_north(az, rule.az_zero, rule.az_rotation)

    if rule.el_convention == "elevation_from_horizon":
        horizontal_scale = np.cos(el)
        up = np.sin(el)
    elif rule.el_convention == "depression_from_horizon":
        horizontal_scale = np.cos(el)
        up = -np.sin(el)
    elif rule.el_convention == "angle_from_up_vertical":
        horizontal_scale = np.sin(el)
        up = np.cos(el)
    elif rule.el_convention == "angle_from_down_vertical":
        horizontal_scale = np.sin(el)
        up = -np.cos(el)
    else:
        raise ValueError(f"Unsupported elevation convention: {rule.el_convention}")

    u = np.column_stack((horizontal_scale * east_h, horizontal_scale * north_h, up)).astype(np.float64)
    u = normalize_vectors(u)
    if rule.sign == "negated":
        u = -u
    elif rule.sign != "as_encoded":
        raise ValueError(f"Unsupported sign convention: {rule.sign}")
    return normalize_vectors(u)


def physical_score_from_row(row: Dict[str, Any]) -> float:
    def val(name: str, default: float = np.nan) -> float:
        try:
            out = float(row.get(name, default))
        except Exception:
            out = default
        return out

    # Main physical target: a ray should pass through both H5 instrument position and refh,
    # point forward from instrument to refh, and have a sensor height consistent with H5.
    score = 0.0
    score += 1.0 * val("median_cross_anchor_line_error_m", 1e6)
    score += 0.5 * val("p90_cross_anchor_line_error_m", 1e6)
    score += 0.2 * val("median_line_to_fit_origin_m", 1e6)
    score += 0.05 * val("p90_line_to_fit_origin_m", 1e6)
    score += 0.03 * val("origin_fit_condition_log10", 8.0)

    # Height agreement against H5 instrument altitude is used to reject arbitrary intersections.
    score += 0.10 * abs(val("fit_origin_vertical_diff_to_median_instrument_m", 1e6))
    score += 0.02 * val("fit_origin_horizontal_to_median_instrument_m", 1e6)
    score += 0.05 * abs(val("fit_origin_height_error_vs_h5_platform_height_m", 1e6))

    # Optional range-window endpoints, if present.
    if np.isfinite(val("median_line_to_rwstart_m", np.nan)):
        score += 0.15 * val("median_line_to_rwstart_m", 1e6)
    if np.isfinite(val("median_line_to_rwstop_m", np.nan)):
        score += 0.15 * val("median_line_to_rwstop_m", 1e6)

    med_cos = val("median_cosine_to_inst_to_ref", np.nan)
    if not np.isfinite(med_cos):
        score += 1e6
    elif med_cos < 0.0:
        score += 1e6 + 1e5 * abs(med_cos)
    else:
        score += 1e5 * max(0.0, 1.0 - med_cos)
    score += 100.0 * val("median_angle_error_to_segment_deg", 1e6)

    # Penalize impossible or implausible platform heights/ranges, but do not hard-code a single altitude.
    agl = val("fit_origin_height_above_median_refh_m", np.nan)
    if np.isfinite(agl):
        if agl < 100.0:
            score += 1e5 + 100.0 * (100.0 - agl)
        elif agl > 50_000.0:
            score += 1e5 + 0.1 * (agl - 50_000.0)
    h5_agl = val("h5_median_instrument_height_above_median_refh_m", np.nan)
    if np.isfinite(h5_agl):
        if h5_agl < 100.0 or h5_agl > 50_000.0:
            score += 1e5

    range_m = val("median_instrument_to_refh_range_m", np.nan)
    if np.isfinite(range_m):
        if range_m < 100.0:
            score += 1e5 + 100.0 * (100.0 - range_m)
        elif range_m > 50_000.0:
            score += 1e5 + 0.1 * (range_m - 50_000.0)

    documented_err = val("h5_platform_height_error_to_expected_agl_band_m", np.nan)
    if np.isfinite(documented_err):
        score += 0.01 * documented_err
    return float(score)


def score_rule(
    rule: AngleRule,
    refh_xyz: np.ndarray,
    instr_xyz: Optional[np.ndarray],
    rwstart_xyz: Optional[np.ndarray],
    rwstop_xyz: Optional[np.ndarray],
    beam_az: np.ndarray,
    beam_el: np.ndarray,
    track_num: np.ndarray,
    epsg: int,
    expected_agl_bands: Sequence[Tuple[float, float]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    directions = direction_vectors_from_rule(beam_az, beam_el, rule)
    valid = finite_rows(refh_xyz, directions)
    if instr_xyz is not None:
        valid &= finite_rows(instr_xyz)

    if np.count_nonzero(valid) < 3:
        row = {"rule_key": rule.key, **asdict(rule), "valid_count": int(np.count_nonzero(valid)), "status": "insufficient_valid_lines"}
        row["physical_score"] = np.inf
        return row, pd.DataFrame()

    P = refh_xyz[valid]
    U = directions[valid]
    T = track_num[valid]
    I = None if instr_xyz is None else instr_xyz[valid]
    RWS = None if rwstart_xyz is None else rwstart_xyz[valid]
    RWE = None if rwstop_xyz is None else rwstop_xyz[valid]

    if rule.basis == "refh":
        line_points = P
    elif rule.basis == "instrument":
        if I is None:
            row = {"rule_key": rule.key, **asdict(rule), "valid_count": int(np.count_nonzero(valid)), "status": "missing_instrument_for_instrument_basis"}
            row["physical_score"] = np.inf
            return row, pd.DataFrame()
        line_points = I
    else:
        raise ValueError(f"Unsupported basis: {rule.basis}")

    origin, cond, solved = fit_common_origin(line_points, U)
    origin_repeated = np.repeat(origin[None, :], P.shape[0], axis=0)
    line_to_origin = line_point_distances(origin_repeated, line_points, U)
    line_to_refh = line_point_distances(P, line_points, U)
    lon, lat = inverse_project_xy(origin[0], origin[1], epsg)

    row: Dict[str, Any] = {
        "rule_key": rule.key,
        **asdict(rule),
        "valid_count": int(P.shape[0]),
        "status": "ok" if solved else "origin_lstsq_or_problematic",
        "origin_fit_condition_number": float(cond),
        "origin_fit_condition_log10": float(np.log10(cond)) if np.isfinite(cond) and cond > 0.0 else np.nan,
        "fit_origin_x_m": float(origin[0]),
        "fit_origin_y_m": float(origin[1]),
        "fit_origin_z_m": float(origin[2]),
        "fit_origin_longitude": lon,
        "fit_origin_latitude": lat,
        "fit_origin_altitude_m": float(origin[2]),
        "median_line_to_fit_origin_m": median(line_to_origin),
        "p90_line_to_fit_origin_m": q(line_to_origin, 90),
        "p95_line_to_fit_origin_m": q(line_to_origin, 95),
        "median_line_to_refh_m": median(line_to_refh),
        "p90_line_to_refh_m": q(line_to_refh, 90),
        "median_refh_z_m": float(np.nanmedian(P[:, 2])),
        "fit_origin_height_above_median_refh_m": float(origin[2] - np.nanmedian(P[:, 2])),
    }

    if I is not None:
        line_to_instr = line_point_distances(I, line_points, U)
        seg = P - I
        seg_range = np.linalg.norm(seg, axis=1)
        seg_unit = normalize_vectors(seg)
        cos_to_inst_ref = np.sum(U * seg_unit, axis=1)
        angle_err = np.degrees(np.arccos(np.clip(cos_to_inst_ref, -1.0, 1.0)))
        signed_ref_to_instr = signed_along_line(I, P, U)

        med_instr = np.nanmedian(I, axis=0)
        mean_instr = np.nanmean(I, axis=0)
        fit_to_med = origin - med_instr
        fit_to_mean = origin - mean_instr
        med_instr_lon, med_instr_lat = inverse_project_xy(med_instr[0], med_instr[1], epsg)

        if rule.basis == "instrument":
            cross_anchor_error = line_to_refh
        else:
            cross_anchor_error = line_to_instr

        h5_platform_height = med_instr[2] - np.nanmedian(P[:, 2])
        band_err, band_label = nearest_band_error(h5_platform_height, expected_agl_bands)

        row.update({
            "median_cross_anchor_line_error_m": median(cross_anchor_error),
            "p90_cross_anchor_line_error_m": q(cross_anchor_error, 90),
            "median_line_to_own_instrument_m": median(line_to_instr),
            "p90_line_to_own_instrument_m": q(line_to_instr, 90),
            "median_angle_error_to_segment_deg": median(angle_err),
            "p90_angle_error_to_segment_deg": q(angle_err, 90),
            "median_cosine_to_inst_to_ref": median(cos_to_inst_ref),
            "median_signed_range_ref_to_inst_along_rule_m": median(signed_ref_to_instr),
            "median_instrument_to_refh_range_m": median(seg_range),
            "p90_instrument_to_refh_range_m": q(seg_range, 90),
            "median_horizontal_instrument_to_refh_offset_m": median(np.linalg.norm(seg[:, :2], axis=1)),
            "median_vertical_instrument_minus_refh_m": median(I[:, 2] - P[:, 2]),
            "fit_origin_to_median_instrument_m": float(np.linalg.norm(fit_to_med)),
            "fit_origin_to_mean_instrument_m": float(np.linalg.norm(fit_to_mean)),
            "fit_origin_horizontal_to_median_instrument_m": float(np.linalg.norm(fit_to_med[:2])),
            "fit_origin_horizontal_to_mean_instrument_m": float(np.linalg.norm(fit_to_mean[:2])),
            "fit_origin_vertical_diff_to_median_instrument_m": float(fit_to_med[2]),
            "fit_origin_height_error_vs_h5_platform_height_m": float((origin[2] - np.nanmedian(P[:, 2])) - h5_platform_height),
            "median_instrument_x_m": float(med_instr[0]),
            "median_instrument_y_m": float(med_instr[1]),
            "median_instrument_z_m": float(med_instr[2]),
            "mean_instrument_x_m": float(mean_instr[0]),
            "mean_instrument_y_m": float(mean_instr[1]),
            "mean_instrument_z_m": float(mean_instr[2]),
            "median_instrument_altitude_m": float(med_instr[2]),
            "median_instrument_longitude": med_instr_lon,
            "median_instrument_latitude": med_instr_lat,
            "h5_median_instrument_height_above_median_refh_m": float(h5_platform_height),
            "h5_platform_height_error_to_expected_agl_band_m": band_err,
            "nearest_expected_agl_band_m": band_label,
        })
        per_track = pd.DataFrame({
            "track_num": T,
            "line_to_fit_origin_m": line_to_origin,
            "line_to_refh_m": line_to_refh,
            "line_to_own_instrument_m": line_to_instr,
            "cross_anchor_line_error_m": cross_anchor_error,
            "angle_error_to_segment_deg": angle_err,
            "cosine_to_inst_to_ref": cos_to_inst_ref,
            "signed_range_ref_to_inst_along_rule_m": signed_ref_to_instr,
            "instrument_to_refh_range_m": seg_range,
            "horizontal_instrument_to_refh_offset_m": np.linalg.norm(seg[:, :2], axis=1),
            "vertical_instrument_minus_refh_m": I[:, 2] - P[:, 2],
        })
    else:
        row.update({
            "median_cross_anchor_line_error_m": np.nan,
            "p90_cross_anchor_line_error_m": np.nan,
            "median_line_to_own_instrument_m": np.nan,
            "p90_line_to_own_instrument_m": np.nan,
            "median_angle_error_to_segment_deg": np.nan,
            "p90_angle_error_to_segment_deg": np.nan,
            "median_cosine_to_inst_to_ref": np.nan,
            "median_signed_range_ref_to_inst_along_rule_m": np.nan,
            "median_instrument_to_refh_range_m": np.nan,
            "median_horizontal_instrument_to_refh_offset_m": np.nan,
            "median_vertical_instrument_minus_refh_m": np.nan,
            "fit_origin_to_median_instrument_m": np.nan,
            "fit_origin_to_mean_instrument_m": np.nan,
            "fit_origin_horizontal_to_median_instrument_m": np.nan,
            "fit_origin_vertical_diff_to_median_instrument_m": np.nan,
            "fit_origin_height_error_vs_h5_platform_height_m": np.nan,
            "median_instrument_altitude_m": np.nan,
            "h5_median_instrument_height_above_median_refh_m": np.nan,
            "h5_platform_height_error_to_expected_agl_band_m": np.nan,
            "nearest_expected_agl_band_m": "",
        })
        per_track = pd.DataFrame({"track_num": T, "line_to_fit_origin_m": line_to_origin, "line_to_refh_m": line_to_refh})

    if RWS is not None and finite_rows(RWS).any():
        line_to_rwstart = line_point_distances(RWS, line_points, U)
        row["median_line_to_rwstart_m"] = median(line_to_rwstart)
        row["p90_line_to_rwstart_m"] = q(line_to_rwstart, 90)
        row["median_signed_range_refh_to_rwstart_along_rule_m"] = median(signed_along_line(RWS, P, U))
        per_track["line_to_rwstart_m"] = line_to_rwstart
        per_track["signed_range_refh_to_rwstart_along_rule_m"] = signed_along_line(RWS, P, U)
    else:
        row["median_line_to_rwstart_m"] = np.nan
        row["p90_line_to_rwstart_m"] = np.nan
        row["median_signed_range_refh_to_rwstart_along_rule_m"] = np.nan

    if RWE is not None and finite_rows(RWE).any():
        line_to_rwstop = line_point_distances(RWE, line_points, U)
        row["median_line_to_rwstop_m"] = median(line_to_rwstop)
        row["p90_line_to_rwstop_m"] = q(line_to_rwstop, 90)
        row["median_signed_range_refh_to_rwstop_along_rule_m"] = median(signed_along_line(RWE, P, U))
        per_track["line_to_rwstop_m"] = line_to_rwstop
        per_track["signed_range_refh_to_rwstop_along_rule_m"] = signed_along_line(RWE, P, U)
    else:
        row["median_line_to_rwstop_m"] = np.nan
        row["p90_line_to_rwstop_m"] = np.nan
        row["median_signed_range_refh_to_rwstop_along_rule_m"] = np.nan

    row["direction_forward_to_refh"] = bool(np.isfinite(row.get("median_cosine_to_inst_to_ref", np.nan)) and float(row["median_cosine_to_inst_to_ref"]) > 0.0)
    row["origin_near_instrument_3d_lt_100m"] = bool(np.isfinite(row.get("fit_origin_to_median_instrument_m", np.nan)) and float(row["fit_origin_to_median_instrument_m"]) < 100.0)
    row["origin_height_plausible_100m_to_50km_agl"] = bool(np.isfinite(row.get("fit_origin_height_above_median_refh_m", np.nan)) and 100.0 <= float(row["fit_origin_height_above_median_refh_m"]) <= 50000.0)
    row["range_plausible_100m_to_50km"] = bool(np.isfinite(row.get("median_instrument_to_refh_range_m", np.nan)) and 100.0 <= float(row["median_instrument_to_refh_range_m"]) <= 50000.0)
    row["h5_height_matches_expected_agl_band"] = bool(np.isfinite(row.get("h5_platform_height_error_to_expected_agl_band_m", np.nan)) and float(row["h5_platform_height_error_to_expected_agl_band_m"]) == 0.0)
    row["rule_passes_strict_physical_gate"] = bool(
        np.isfinite(row.get("median_cosine_to_inst_to_ref", np.nan))
        and float(row["median_cosine_to_inst_to_ref"]) > 0.9999
        and np.isfinite(row.get("median_angle_error_to_segment_deg", np.nan))
        and float(row["median_angle_error_to_segment_deg"]) < 0.05
        and np.isfinite(row.get("median_cross_anchor_line_error_m", np.nan))
        and float(row["median_cross_anchor_line_error_m"]) < 5.0
        and np.isfinite(row.get("median_instrument_to_refh_range_m", np.nan))
        and 100.0 <= float(row["median_instrument_to_refh_range_m"]) <= 50000.0
    )
    row["physical_score"] = physical_score_from_row(row)
    return row, per_track


def build_aux_points(sweep: SweepData, epsg: int, prefix: str) -> Optional[np.ndarray]:
    lon = getattr(sweep, f"{prefix}_lon")
    lat = getattr(sweep, f"{prefix}_lat")
    z = getattr(sweep, f"{prefix}_z")
    if lon is None or lat is None or z is None:
        return None
    x, y = project_lonlat(lon, lat, epsg)
    return np.column_stack((x, y, z)).astype(np.float64)


def height_sanity_summary(sweep: SweepData, refh_xyz: np.ndarray, instr_xyz: Optional[np.ndarray], expected_agl_bands: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "instrument_fields_present": bool(instr_xyz is not None),
        "instrument_altitude_field_found": bool(sweep.instrument_z is not None),
        "instrument_altitude_dataset_path": sweep.source_datasets.get("instrument_z"),
        "refh_dataset_path": sweep.source_datasets.get("refh_z"),
        "height_reference_note": "instrument_altitude and refh are compared only as H5-internal height fields; this script does not prove the vertical datum.",
        "expected_agl_bands_m": [list(b) for b in expected_agl_bands],
    }
    out["refh_z_median_m"] = median(refh_xyz[:, 2])
    out["refh_z_p10_m"] = q(refh_xyz[:, 2], 10)
    out["refh_z_p90_m"] = q(refh_xyz[:, 2], 90)
    if instr_xyz is not None:
        dz = instr_xyz[:, 2] - refh_xyz[:, 2]
        horizontal = np.linalg.norm(instr_xyz[:, :2] - refh_xyz[:, :2], axis=1)
        slant = np.linalg.norm(instr_xyz - refh_xyz, axis=1)
        out.update({
            "instrument_altitude_median_m": median(instr_xyz[:, 2]),
            "instrument_altitude_p10_m": q(instr_xyz[:, 2], 10),
            "instrument_altitude_p90_m": q(instr_xyz[:, 2], 90),
            "instrument_minus_refh_vertical_median_m": median(dz),
            "instrument_minus_refh_vertical_nmad_m": robust_nmad(dz),
            "instrument_refh_horizontal_offset_median_m": median(horizontal),
            "instrument_refh_horizontal_offset_p90_m": q(horizontal, 90),
            "instrument_refh_slant_range_median_m": median(slant),
            "instrument_refh_slant_range_p90_m": q(slant, 90),
        })
        band_err, band_label = nearest_band_error(float(out["instrument_minus_refh_vertical_median_m"]), expected_agl_bands)
        out["instrument_minus_refh_vertical_error_to_expected_agl_band_m"] = band_err
        out["nearest_expected_agl_band_m"] = band_label
        if sweep.instrument_z_error is not None:
            out["instrument_altitude_error_median_m"] = median(sweep.instrument_z_error)
            out["instrument_altitude_error_p90_m"] = q(sweep.instrument_z_error, 90)
    if sweep.geoid is not None:
        out["geoid_median_m"] = median(sweep.geoid)
        out["geoid_p10_m"] = q(sweep.geoid, 10)
        out["geoid_p90_m"] = q(sweep.geoid, 90)
    if sweep.geoid_free2mean is not None:
        out["geoid_free2mean_median_m"] = median(sweep.geoid_free2mean)
    if sweep.bin_size is not None:
        out["bin_size_median"] = median(sweep.bin_size)
        out["bin_size_p10"] = q(sweep.bin_size, 10)
        out["bin_size_p90"] = q(sweep.bin_size, 90)
    return out


def write_ply_points(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_ply_edges(path: Path, vertices: np.ndarray, colors: np.ndarray, edges: Sequence[Tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\nend_header\n")
        for p, c in zip(vertices, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for a, b in edges:
            f.write(f"{int(a)} {int(b)}\n")


def export_markers_ply(path: Path, refh_xyz: np.ndarray, instr_xyz: Optional[np.ndarray], top_rows: pd.DataFrame) -> None:
    pts: List[np.ndarray] = []
    cols: List[Tuple[int, int, int]] = []
    sample_ids = np.linspace(0, refh_xyz.shape[0] - 1, min(refh_xyz.shape[0], 64)).round().astype(int)
    for p in refh_xyz[sample_ids]:
        pts.append(p); cols.append((30, 120, 220))
    if instr_xyz is not None:
        for p in instr_xyz[sample_ids]:
            pts.append(p); cols.append((40, 40, 40))
    palette = [(255, 60, 60), (255, 140, 0), (180, 100, 255), (0, 170, 80), (0, 180, 180), (120, 120, 120)]
    for j, (_, row) in enumerate(top_rows.iterrows()):
        pts.append(np.array([row["fit_origin_x_m"], row["fit_origin_y_m"], row["fit_origin_z_m"]], dtype=np.float64))
        cols.append(palette[j % len(palette)])
    if pts:
        write_ply_points(path, np.vstack(pts), np.asarray(cols, dtype=np.uint8))


def export_ray_bundle_ply(
    path: Path,
    rule: AngleRule,
    refh_xyz: np.ndarray,
    instr_xyz: Optional[np.ndarray],
    beam_az: np.ndarray,
    beam_el: np.ndarray,
    origin_xyz: np.ndarray,
    max_tracks: int = 64,
) -> None:
    valid = finite_rows(refh_xyz)
    if instr_xyz is not None:
        valid &= finite_rows(instr_xyz)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        return
    sample_ids = valid_ids[np.linspace(0, valid_ids.size - 1, min(max_tracks, valid_ids.size)).round().astype(int)]
    vertices: List[np.ndarray] = []
    colors: List[Tuple[int, int, int]] = []
    edges: List[Tuple[int, int]] = []
    U_all = direction_vectors_from_rule(beam_az, beam_el, rule)
    for idx in sample_ids:
        if rule.basis == "instrument" and instr_xyz is not None:
            p0 = instr_xyz[idx]
        else:
            p0 = refh_xyz[idx]
        p1 = refh_xyz[idx]
        # Also add a short forward ray segment to make orientation visible when opened in CloudCompare.
        u = U_all[idx]
        p2 = p0 + 250.0 * u
        a = len(vertices)
        vertices.append(p0); colors.append((40, 40, 40))
        vertices.append(p1); colors.append((30, 120, 220))
        vertices.append(p2); colors.append((255, 170, 0))
        edges.append((a, a + 1))
        edges.append((a, a + 2))
    vertices.append(origin_xyz); colors.append((255, 60, 60))
    write_ply_edges(path, np.vstack(vertices), np.asarray(colors, dtype=np.uint8), edges)


def score_many_rules(
    rules: Sequence[AngleRule],
    refh_xyz: np.ndarray,
    instr_xyz: Optional[np.ndarray],
    rwstart_xyz: Optional[np.ndarray],
    rwstop_xyz: Optional[np.ndarray],
    sweep: SweepData,
    epsg: int,
    expected_agl_bands: Sequence[Tuple[float, float]],
    progress_prefix: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for k, rule in enumerate(rules, start=1):
        try:
            row, _ = score_rule(rule, refh_xyz, instr_xyz, rwstart_xyz, rwstop_xyz, sweep.beam_az, sweep.beam_el, sweep.track_num, epsg, expected_agl_bands)
        except Exception as exc:
            row = {"rule_key": rule.key, **asdict(rule), "valid_count": 0, "status": f"error: {type(exc).__name__}: {exc}", "physical_score": np.inf}
        rows.append(row)
        if k % 100 == 0 or k == len(rules):
            print(f"{progress_prefix}: scored {k}/{len(rules)} rules")
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CASALS beam-angle conventions using broad convention search and H5 height sanity checks.")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH, help="Input CASALS L1B H5 file.")
    parser.add_argument("--sweep", type=int, default=DEFAULT_SWEEP_INDEX, help="Sweep index to analyze.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output directory.")
    parser.add_argument("--epsg", type=int, default=None, help="Projected CRS EPSG. Default: infer WGS84 UTM from refh lon/lat.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top rows to print/export in top origins CSV.")
    parser.add_argument("--top-n-ply", type=int, default=8, help="Number of top physical rules to export as PLY ray bundles.")
    parser.add_argument("--ply-max-tracks", type=int, default=64, help="Maximum tracks per ray-bundle PLY.")
    parser.add_argument("--sort-by", choices=("physical_score", "median_cross_anchor_line_error_m", "median_line_to_fit_origin_m"), default="physical_score")
    parser.add_argument("--expected-agl-bands", type=str, default=DEFAULT_EXPECTED_AGL_BANDS, help="Comma-separated AGL sanity bands in meters, e.g. 3500:6500,7500:9500. Use empty string to disable.")
    parser.add_argument("--refine-top-n", type=int, default=12, help="Refine this many top base rules using az/el bias grid. Set 0 to disable.")
    parser.add_argument("--bias-grid-deg", type=str, default="0,-0.02,0.02,-0.05,0.05,-0.1,0.1", help="Comma-separated angular bias grid in degrees used for top-rule refinement.")
    return parser.parse_args()


def add_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy()
    for col, rank_col in [
        ("physical_score", "rank_by_physical_score"),
        ("median_cross_anchor_line_error_m", "rank_by_cross_anchor"),
        ("median_line_to_fit_origin_m", "rank_by_convergence"),
        ("median_angle_error_to_segment_deg", "rank_by_angle_error"),
    ]:
        if col in scores.columns:
            scores[rank_col] = scores[col].rank(method="min", ascending=True, na_option="bottom").astype(int)
    return scores


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    scores_dir = output_root / "scores"
    pc_dir = output_root / "pc"
    scores_dir.mkdir(parents=True, exist_ok=True)
    pc_dir.mkdir(parents=True, exist_ok=True)
    expected_agl_bands = parse_agl_bands(args.expected_agl_bands)

    print("=" * 100)
    print("CASALS broad beam-line convention + H5 height sanity audit")
    print(f"H5: {args.h5.resolve()}")
    print(f"Sweep: {args.sweep}")
    print(f"Output: {output_root.resolve()}")
    print(f"Expected AGL bands: {expected_agl_bands if expected_agl_bands else 'disabled'}")
    print("=" * 100)

    sweep = load_sweep_data(args.h5, args.sweep)
    instr_present = sweep.instrument_lon is not None and sweep.instrument_lat is not None and sweep.instrument_z is not None
    print("Sweep loaded")
    print(f"Tracks: {sweep.n}")
    print(f"Track range: {int(np.min(sweep.track_num))}..{int(np.max(sweep.track_num))}")
    print(f"Instrument fields present: {instr_present}")
    print(f"RW start/stop fields present: {sweep.rwstart_lon is not None and sweep.rwstop_lon is not None}")

    epsg = args.epsg if args.epsg is not None else infer_wgs84_utm_epsg(sweep.refh_lon, sweep.refh_lat)
    print(f"Using projected CRS EPSG:{epsg}")

    refh_x, refh_y = project_lonlat(sweep.refh_lon, sweep.refh_lat, epsg)
    refh_xyz = np.column_stack((refh_x, refh_y, sweep.refh_z)).astype(np.float64)
    instr_xyz: Optional[np.ndarray] = None
    if instr_present:
        instr_x, instr_y = project_lonlat(sweep.instrument_lon, sweep.instrument_lat, epsg)
        instr_xyz = np.column_stack((instr_x, instr_y, sweep.instrument_z)).astype(np.float64)

    rwstart_xyz = build_aux_points(sweep, epsg, "rwstart")
    rwstop_xyz = build_aux_points(sweep, epsg, "rwstop")

    height_summary = height_sanity_summary(sweep, refh_xyz, instr_xyz, expected_agl_bands)
    height_json_path = scores_dir / f"sw_{args.sweep}_h5_height_sanity.json"
    height_csv_path = scores_dir / f"sw_{args.sweep}_h5_height_sanity.csv"
    with height_json_path.open("w", encoding="utf-8") as f:
        json.dump(height_summary, f, indent=2)
    pd.DataFrame([height_summary]).to_csv(height_csv_path, index=False)
    print("H5 height sanity:")
    print(json.dumps(height_summary, indent=2))

    base_rules = list(iter_base_angle_rules())
    print("=" * 100)
    print("Scoring broad base angle conventions")
    print(f"Number of base angle rules: {len(base_rules)}")
    base_scores = score_many_rules(base_rules, refh_xyz, instr_xyz, rwstart_xyz, rwstop_xyz, sweep, epsg, expected_agl_bands, "base")
    base_scores = add_ranks(base_scores)

    refined_scores = pd.DataFrame()
    if args.refine_top_n > 0:
        bias_values = parse_float_list(args.bias_grid_deg)
        base_top = base_scores.sort_values(args.sort_by, ascending=True, na_position="last").head(int(args.refine_top_n))
        refine_rules: List[AngleRule] = []
        seen: set[str] = set()
        for _, row in base_top.iterrows():
            base_rule = AngleRule(
                unit=str(row["unit"]),
                az_zero=str(row["az_zero"]),
                az_rotation=str(row["az_rotation"]),
                el_convention=str(row["el_convention"]),
                sign=str(row["sign"]),
                basis=str(row["basis"]),
                angle_transform=str(row["angle_transform"]),
                az_bias_deg=0.0,
                el_bias_deg=0.0,
                refinement_stage="bias_refined",
            )
            for az_bias in bias_values:
                for el_bias in bias_values:
                    r = replace(base_rule, az_bias_deg=float(az_bias), el_bias_deg=float(el_bias))
                    if r.key not in seen:
                        refine_rules.append(r)
                        seen.add(r.key)
        print("=" * 100)
        print("Refining top base rules with small az/el bias grid")
        print(f"Top base rules: {int(args.refine_top_n)}; bias values: {bias_values}; refined rules: {len(refine_rules)}")
        refined_scores = score_many_rules(refine_rules, refh_xyz, instr_xyz, rwstart_xyz, rwstop_xyz, sweep, epsg, expected_agl_bands, "refine")
        refined_scores = add_ranks(refined_scores)

    if refined_scores.empty:
        scores = base_scores.copy()
    else:
        scores = pd.concat([base_scores, refined_scores], ignore_index=True, sort=False)
        scores = add_ranks(scores)

    all_scores_path = scores_dir / f"sw_{args.sweep}_origin_scores_all.csv"
    base_scores_path = scores_dir / f"sw_{args.sweep}_origin_scores_base_only.csv"
    refined_scores_path = scores_dir / f"sw_{args.sweep}_origin_scores_refined_only.csv"
    scores.to_csv(all_scores_path, index=False)
    base_scores.to_csv(base_scores_path, index=False)
    if not refined_scores.empty:
        refined_scores.to_csv(refined_scores_path, index=False)

    compact_cols = [
        "rank_by_physical_score", "rank_by_cross_anchor", "rank_by_convergence", "rank_by_angle_error",
        "rule_key", "status", "valid_count", "physical_score", "rule_passes_strict_physical_gate",
        "median_cross_anchor_line_error_m", "p90_cross_anchor_line_error_m",
        "median_line_to_fit_origin_m", "p90_line_to_fit_origin_m",
        "fit_origin_to_median_instrument_m", "fit_origin_horizontal_to_median_instrument_m",
        "fit_origin_vertical_diff_to_median_instrument_m", "fit_origin_height_error_vs_h5_platform_height_m",
        "fit_origin_altitude_m", "median_instrument_altitude_m", "median_refh_z_m",
        "fit_origin_height_above_median_refh_m", "h5_median_instrument_height_above_median_refh_m",
        "h5_platform_height_error_to_expected_agl_band_m", "nearest_expected_agl_band_m",
        "median_line_to_refh_m", "median_line_to_own_instrument_m",
        "median_line_to_rwstart_m", "median_line_to_rwstop_m",
        "median_signed_range_refh_to_rwstart_along_rule_m", "median_signed_range_refh_to_rwstop_along_rule_m",
        "median_angle_error_to_segment_deg", "median_cosine_to_inst_to_ref",
        "median_signed_range_ref_to_inst_along_rule_m", "median_instrument_to_refh_range_m",
        "median_horizontal_instrument_to_refh_offset_m", "median_vertical_instrument_minus_refh_m",
        "origin_fit_condition_number", "origin_fit_condition_log10",
        "direction_forward_to_refh", "origin_near_instrument_3d_lt_100m",
        "origin_height_plausible_100m_to_50km_agl", "range_plausible_100m_to_50km",
        "h5_height_matches_expected_agl_band", "unit", "az_zero", "az_rotation", "el_convention",
        "sign", "basis", "angle_transform", "az_bias_deg", "el_bias_deg", "refinement_stage",
        "fit_origin_longitude", "fit_origin_latitude", "median_instrument_longitude", "median_instrument_latitude",
    ]
    compact_cols = [c for c in compact_cols if c in scores.columns]
    compact = scores.sort_values(args.sort_by, ascending=True, na_position="last")[compact_cols].copy()
    compact_path = scores_dir / f"sw_{args.sweep}_origin_scores_compact.csv"
    compact.to_csv(compact_path, index=False)

    recommended = compact[compact.get("rule_passes_strict_physical_gate", False) == True].copy() if "rule_passes_strict_physical_gate" in compact.columns else compact.head(0).copy()
    recommended_path = scores_dir / f"sw_{args.sweep}_recommended_physical_rules.csv"
    recommended.to_csv(recommended_path, index=False)

    top = compact.head(int(args.top_n)).copy()
    top_origins_path = scores_dir / f"sw_{args.sweep}_fit_origins_top.csv"
    top.to_csv(top_origins_path, index=False)

    print("=" * 100)
    print(f"Top {min(args.top_n, len(top))} rules by {args.sort_by}")
    with pd.option_context("display.max_colwidth", 160, "display.width", 260):
        print(top.to_string(index=False))

    top_physical = scores.sort_values("physical_score", ascending=True, na_position="last").head(int(args.top_n_ply))
    per_track_tables = []
    for rank, (_, row) in enumerate(top_physical.iterrows(), start=1):
        rule = AngleRule(
            unit=str(row["unit"]),
            az_zero=str(row["az_zero"]),
            az_rotation=str(row["az_rotation"]),
            el_convention=str(row["el_convention"]),
            sign=str(row["sign"]),
            basis=str(row["basis"]),
            angle_transform=str(row["angle_transform"]),
            az_bias_deg=float(row.get("az_bias_deg", 0.0)),
            el_bias_deg=float(row.get("el_bias_deg", 0.0)),
            refinement_stage=str(row.get("refinement_stage", "base")),
        )
        print("-" * 100)
        print(f"Exporting physical-rank {rank}: {rule.key}")
        _, per_track = score_rule(rule, refh_xyz, instr_xyz, rwstart_xyz, rwstop_xyz, sweep.beam_az, sweep.beam_el, sweep.track_num, epsg, expected_agl_bands)
        if not per_track.empty:
            per_track.insert(0, "physical_rank", rank)
            per_track.insert(1, "rule_key", rule.key)
            per_track_tables.append(per_track)
        origin_xyz = np.array([row["fit_origin_x_m"], row["fit_origin_y_m"], row["fit_origin_z_m"]], dtype=np.float64)
        export_ray_bundle_ply(pc_dir / f"sw_{args.sweep}_rank_{rank:02d}_ray_bundle.ply", rule, refh_xyz, instr_xyz, sweep.beam_az, sweep.beam_el, origin_xyz, max_tracks=int(args.ply_max_tracks))

    per_track_path = None
    if per_track_tables:
        per_track_path = scores_dir / f"sw_{args.sweep}_top_rule_per_track_diagnostics.csv"
        pd.concat(per_track_tables, ignore_index=True).to_csv(per_track_path, index=False)

    export_markers_ply(pc_dir / f"sw_{args.sweep}_markers.ply", refh_xyz, instr_xyz, top_physical)

    metadata = {
        "h5_path": str(args.h5.resolve()),
        "sweep_index": int(args.sweep),
        "output_root": str(output_root.resolve()),
        "epsg": int(epsg),
        "record_count": int(sweep.n),
        "tracks_present_count": int(np.unique(sweep.track_num).size),
        "track_min": int(np.min(sweep.track_num)),
        "track_max": int(np.max(sweep.track_num)),
        "instrument_fields_present": bool(instr_present),
        "rwstart_fields_present": bool(rwstart_xyz is not None),
        "rwstop_fields_present": bool(rwstop_xyz is not None),
        "source_datasets": sweep.source_datasets,
        "base_rule_count": int(len(base_rules)),
        "combined_rule_count": int(len(scores)),
        "rule_family_count_explanation": {
            "units": len(UNITS),
            "az_zero_directions": len(AZ_ZERO_DIRECTIONS),
            "az_rotation_directions": len(AZ_ROTATION_DIRECTIONS),
            "elevation_conventions": len(EL_CONVENTIONS),
            "signs": len(SIGNS),
            "bases": len(BASES),
            "angle_transforms": len(ANGLE_TRANSFORMS),
            "base_total": int(len(base_rules)),
            "refine_top_n": int(args.refine_top_n),
            "bias_grid_deg": parse_float_list(args.bias_grid_deg) if args.refine_top_n > 0 else [],
            "combined_total": int(len(scores)),
        },
        "height_sanity": height_summary,
        "outputs": {
            "all_scores_csv": str(all_scores_path),
            "base_scores_csv": str(base_scores_path),
            "refined_scores_csv": None if refined_scores.empty else str(refined_scores_path),
            "compact_scores_csv": str(compact_path),
            "recommended_physical_rules_csv": str(recommended_path),
            "top_origins_csv": str(top_origins_path),
            "height_sanity_json": str(height_json_path),
            "height_sanity_csv": str(height_csv_path),
            "top_rule_per_track_diagnostics_csv": None if per_track_path is None else str(per_track_path),
            "marker_ply": str(pc_dir / f"sw_{args.sweep}_markers.ply"),
            "ray_bundle_ply_dir": str(pc_dir),
        },
        "notes": [
            "physical_score is a convenience ranking only; inspect raw metrics before choosing a convention.",
            "instrument_altitude and refh are compared as H5-internal height fields. This script does not prove their external vertical datum.",
            "rwstart/rwstop line checks are included when those H5 fields exist, but they are still H5-internal consistency checks.",
            "This script validates refh beam-line geometry only; it does not establish waveform-bin geolocation.",
        ],
    }
    meta_path = output_root / f"sw_{args.sweep}_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 100)
    print("Done.")
    print(f"All scores: {all_scores_path}")
    print(f"Base scores: {base_scores_path}")
    if not refined_scores.empty:
        print(f"Refined scores: {refined_scores_path}")
    print(f"Compact scores: {compact_path}")
    print(f"Recommended physical rules: {recommended_path}")
    print(f"Top origins: {top_origins_path}")
    print(f"Height sanity: {height_json_path}")
    if per_track_path is not None:
        print(f"Per-track diagnostics: {per_track_path}")
    print(f"Marker PLY: {pc_dir / f'sw_{args.sweep}_markers.ply'}")
    print(f"Metadata: {meta_path}")
    print(f"Point clouds: {pc_dir}")


if __name__ == "__main__":
    main()
