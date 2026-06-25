#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CASALS beam-angle convention audit with physically meaningful sensor-origin checks.

For a selected CASALS L1B sweep, this script tests multiple interpretations of
local_beam_azimuth/local_beam_elevation. For each interpretation, it builds
3-D beam lines from the refh point or instrument point, estimates the common
line-intersection origin, and evaluates whether that origin is physically
consistent with the H5 instrument position.

This script is diagnostic only. It validates official refh beam-line geometry;
it does not georeference all waveform bins.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

DEFAULT_H5_PATH = Path("./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5")
DEFAULT_SWEEP_INDEX = 7040
DEFAULT_OUTPUT_ROOT = Path("./beam_line_origin")

DATASET_CANDIDATES: Dict[str, Sequence[str]] = {
    "refh_lon": ("refh_longitude", "longitude", "lon"),
    "refh_lat": ("refh_latitude", "latitude", "lat"),
    "refh_z": ("refh", "refh_height", "height", "elevation"),
    "instrument_lon": ("instrument_longitude", "sensor_longitude", "platform_longitude"),
    "instrument_lat": ("instrument_latitude", "sensor_latitude", "platform_latitude"),
    "instrument_z": ("instrument_altitude", "instrument_height", "sensor_altitude", "platform_altitude"),
    "beam_az": ("local_beam_azimuth", "beam_azimuth", "azimuth"),
    "beam_el": ("local_beam_elevation", "beam_elevation", "elevation"),
    "sweep_num": ("sweep_num", "sweep", "sweep_index"),
    "track_num": ("track_num", "track", "track_index"),
    "delta_time": ("delta_time", "time"),
}

UNITS = ("radian", "degree")
AZ_CONVENTIONS = (
    "north_clockwise",
    "north_counterclockwise",
    "south_clockwise",
    "south_counterclockwise",
)
EL_CONVENTIONS = (
    "elevation_from_horizon",
    "depression_from_horizon",
    "angle_from_up_vertical",
)
SIGNS = ("as_encoded", "negated")
BASES = ("refh", "instrument")
ANGLE_TRANSFORMS = (
    "as_named",
    "swap_az_el",
    "negate_az",
    "negate_el",
    "negate_both",
    "swap_and_negate_both",
)


def find_dataset_path(h5: h5py.File, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    """Find a dataset at root level or by basename anywhere in the H5 tree."""
    for name in candidates:
        if name in h5 and isinstance(h5[name], h5py.Dataset):
            return name

    basename_to_path: Dict[str, str] = {}

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            basename_to_path[name.split("/")[-1]] = name

    h5.visititems(visitor)
    for name in candidates:
        if name in basename_to_path:
            return basename_to_path[name]

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
    beam_az: np.ndarray
    beam_el: np.ndarray
    delta_time: Optional[np.ndarray]
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

        refh_lon, refh_lon_path = read_selected("refh_lon", required=True)
        refh_lat, refh_lat_path = read_selected("refh_lat", required=True)
        refh_z, refh_z_path = read_selected("refh_z", required=True)
        instr_lon, instr_lon_path = read_selected("instrument_lon", required=False)
        instr_lat, instr_lat_path = read_selected("instrument_lat", required=False)
        instr_z, instr_z_path = read_selected("instrument_z", required=False)
        beam_az, beam_az_path = read_selected("beam_az", required=True)
        beam_el, beam_el_path = read_selected("beam_el", required=True)
        delta_time, delta_time_path = read_selected("delta_time", required=False)

    source_datasets = {
        "sweep_num": sweep_path,
        "track_num": track_path,
        "refh_longitude": refh_lon_path,
        "refh_latitude": refh_lat_path,
        "refh": refh_z_path,
        "instrument_longitude": instr_lon_path,
        "instrument_latitude": instr_lat_path,
        "instrument_altitude": instr_z_path,
        "local_beam_azimuth": beam_az_path,
        "local_beam_elevation": beam_el_path,
        "delta_time": delta_time_path,
        "duplicate_records_removed_for_same_track": str(duplicate_count),
    }

    return SweepData(
        sweep_index=int(sweep_index),
        record_indices=np.asarray(record_indices, dtype=np.int64),
        track_num=np.asarray(track_sel, dtype=np.int64),
        refh_lon=np.asarray(refh_lon, dtype=np.float64),
        refh_lat=np.asarray(refh_lat, dtype=np.float64),
        refh_z=np.asarray(refh_z, dtype=np.float64),
        instrument_lon=None if instr_lon is None else np.asarray(instr_lon, dtype=np.float64),
        instrument_lat=None if instr_lat is None else np.asarray(instr_lat, dtype=np.float64),
        instrument_z=None if instr_z is None else np.asarray(instr_z, dtype=np.float64),
        beam_az=np.asarray(beam_az, dtype=np.float64),
        beam_el=np.asarray(beam_el, dtype=np.float64),
        delta_time=None if delta_time is None else np.asarray(delta_time, dtype=np.float64),
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
    """
    Least-squares closest point to a bundle of 3-D lines.

    Minimize sum_i ||(I - u_i u_i^T)(x - p_i)||^2.
    """
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


@dataclass(frozen=True)
class AngleRule:
    unit: str
    az_convention: str
    el_convention: str
    sign: str
    basis: str
    angle_transform: str

    @property
    def key(self) -> str:
        return (
            f"{self.unit}|az={self.az_convention}|el={self.el_convention}|"
            f"sign={self.sign}|basis={self.basis}|transform={self.angle_transform}"
        )


def iter_angle_rules() -> Iterable[AngleRule]:
    for unit in UNITS:
        for az_convention in AZ_CONVENTIONS:
            for el_convention in EL_CONVENTIONS:
                for sign in SIGNS:
                    for basis in BASES:
                        for transform in ANGLE_TRANSFORMS:
                            yield AngleRule(unit, az_convention, el_convention, sign, basis, transform)


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


def horizontal_components_from_azimuth(az: np.ndarray, az_convention: str) -> Tuple[np.ndarray, np.ndarray]:
    if az_convention == "north_clockwise":
        east = np.sin(az)
        north = np.cos(az)
    elif az_convention == "north_counterclockwise":
        east = -np.sin(az)
        north = np.cos(az)
    elif az_convention == "south_clockwise":
        east = -np.sin(az)
        north = -np.cos(az)
    elif az_convention == "south_counterclockwise":
        east = np.sin(az)
        north = -np.cos(az)
    else:
        raise ValueError(f"Unsupported azimuth convention: {az_convention}")
    return east, north


def direction_vectors_from_rule(az_raw: np.ndarray, el_raw: np.ndarray, rule: AngleRule) -> np.ndarray:
    az, el = encoded_angles_to_radians(az_raw, el_raw, rule.unit)
    az, el = apply_angle_transform(az, el, rule.angle_transform)
    east_h, north_h = horizontal_components_from_azimuth(az, rule.az_convention)

    if rule.el_convention == "elevation_from_horizon":
        horizontal_scale = np.cos(el)
        up = np.sin(el)
    elif rule.el_convention == "depression_from_horizon":
        horizontal_scale = np.cos(el)
        up = -np.sin(el)
    elif rule.el_convention == "angle_from_up_vertical":
        horizontal_scale = np.sin(el)
        up = np.cos(el)
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
            return float(row.get(name, default))
        except Exception:
            return default

    score = 0.0
    score += val("median_line_to_fit_origin_m", 1e6)
    score += 0.2 * val("p90_line_to_fit_origin_m", 1e6)
    score += val("median_line_to_refh_m", 1e6)
    score += val("median_line_to_own_instrument_m", 1e6)
    score += 0.05 * val("fit_origin_to_median_instrument_m", 1e6)
    score += 0.05 * abs(val("fit_origin_vertical_diff_to_median_instrument_m", 1e6))

    med_cos = val("median_cosine_to_inst_to_ref", np.nan)
    if not np.isfinite(med_cos):
        score += 1e6
    elif med_cos < 0.0:
        score += 1e6 + 1e5 * abs(med_cos)
    else:
        score += 1e4 * max(0.0, 1.0 - med_cos)
    score += 10.0 * val("median_angle_error_to_segment_deg", 1e6)

    agl = val("fit_origin_height_above_median_refh_m", np.nan)
    if np.isfinite(agl):
        if agl < 100.0:
            score += 1e5 + 100.0 * (100.0 - agl)
        elif agl > 30_000.0:
            score += 1e5 + 0.1 * (agl - 30_000.0)

    range_m = abs(val("median_signed_range_ref_to_inst_along_rule_m", np.nan))
    if np.isfinite(range_m):
        if range_m < 100.0:
            score += 1e5 + 100.0 * (100.0 - range_m)
        elif range_m > 50_000.0:
            score += 1e5 + 0.1 * (range_m - 50_000.0)

    cond = val("origin_fit_condition_number", np.nan)
    if np.isfinite(cond) and cond > 1e8:
        score += math.log10(cond / 1e8 + 1.0) * 1000.0
    return float(score)


def score_rule(
    rule: AngleRule,
    refh_xyz: np.ndarray,
    instr_xyz: Optional[np.ndarray],
    beam_az: np.ndarray,
    beam_el: np.ndarray,
    track_num: np.ndarray,
    epsg: int,
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

        row.update({
            "median_line_to_own_instrument_m": median(line_to_instr),
            "p90_line_to_own_instrument_m": q(line_to_instr, 90),
            "median_angle_error_to_segment_deg": median(angle_err),
            "p90_angle_error_to_segment_deg": q(angle_err, 90),
            "median_cosine_to_inst_to_ref": median(cos_to_inst_ref),
            "median_signed_range_ref_to_inst_along_rule_m": median(signed_ref_to_instr),
            "median_instrument_to_refh_range_m": median(seg_range),
            "fit_origin_to_median_instrument_m": float(np.linalg.norm(fit_to_med)),
            "fit_origin_to_mean_instrument_m": float(np.linalg.norm(fit_to_mean)),
            "fit_origin_horizontal_to_median_instrument_m": float(np.linalg.norm(fit_to_med[:2])),
            "fit_origin_horizontal_to_mean_instrument_m": float(np.linalg.norm(fit_to_mean[:2])),
            "fit_origin_vertical_diff_to_median_instrument_m": float(fit_to_med[2]),
            "median_instrument_x_m": float(med_instr[0]),
            "median_instrument_y_m": float(med_instr[1]),
            "median_instrument_z_m": float(med_instr[2]),
            "mean_instrument_x_m": float(mean_instr[0]),
            "mean_instrument_y_m": float(mean_instr[1]),
            "mean_instrument_z_m": float(mean_instr[2]),
            "median_instrument_altitude_m": float(med_instr[2]),
            "median_instrument_longitude": med_instr_lon,
            "median_instrument_latitude": med_instr_lat,
            "fit_origin_height_above_median_refh_m": float(origin[2] - np.nanmedian(P[:, 2])),
            "median_instrument_height_above_median_refh_m": float(med_instr[2] - np.nanmedian(P[:, 2])),
        })
        per_track = pd.DataFrame({
            "track_num": T,
            "line_to_fit_origin_m": line_to_origin,
            "line_to_refh_m": line_to_refh,
            "line_to_own_instrument_m": line_to_instr,
            "angle_error_to_segment_deg": angle_err,
            "cosine_to_inst_to_ref": cos_to_inst_ref,
            "signed_range_ref_to_inst_along_rule_m": signed_ref_to_instr,
            "instrument_to_refh_range_m": seg_range,
        })
    else:
        row.update({
            "median_line_to_own_instrument_m": np.nan,
            "p90_line_to_own_instrument_m": np.nan,
            "median_angle_error_to_segment_deg": np.nan,
            "p90_angle_error_to_segment_deg": np.nan,
            "median_cosine_to_inst_to_ref": np.nan,
            "median_signed_range_ref_to_inst_along_rule_m": np.nan,
            "median_instrument_to_refh_range_m": np.nan,
            "fit_origin_to_median_instrument_m": np.nan,
            "fit_origin_to_mean_instrument_m": np.nan,
            "fit_origin_horizontal_to_median_instrument_m": np.nan,
            "fit_origin_vertical_diff_to_median_instrument_m": np.nan,
            "median_instrument_altitude_m": np.nan,
            "fit_origin_height_above_median_refh_m": np.nan,
            "median_instrument_height_above_median_refh_m": np.nan,
        })
        per_track = pd.DataFrame({"track_num": T, "line_to_fit_origin_m": line_to_origin, "line_to_refh_m": line_to_refh})

    row["direction_forward_to_refh"] = bool(np.isfinite(row.get("median_cosine_to_inst_to_ref", np.nan)) and float(row["median_cosine_to_inst_to_ref"]) > 0.0)
    row["origin_near_instrument_3d_lt_100m"] = bool(np.isfinite(row.get("fit_origin_to_median_instrument_m", np.nan)) and float(row["fit_origin_to_median_instrument_m"]) < 100.0)
    row["origin_height_plausible_100m_to_30km_agl"] = bool(np.isfinite(row.get("fit_origin_height_above_median_refh_m", np.nan)) and 100.0 <= float(row["fit_origin_height_above_median_refh_m"]) <= 30000.0)
    row["range_plausible_100m_to_50km"] = bool(np.isfinite(row.get("median_instrument_to_refh_range_m", np.nan)) and 100.0 <= float(row["median_instrument_to_refh_range_m"]) <= 50000.0)
    row["physical_score"] = physical_score_from_row(row)
    return row, per_track


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


def export_ray_bundle_ply(path: Path, rule: AngleRule, refh_xyz: np.ndarray, instr_xyz: Optional[np.ndarray], beam_az: np.ndarray, beam_el: np.ndarray, origin_xyz: np.ndarray, max_tracks: int = 64) -> None:
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
    for idx in sample_ids:
        p0 = origin_xyz if instr_xyz is None else instr_xyz[idx]
        p1 = refh_xyz[idx]
        a = len(vertices)
        vertices.append(p0); colors.append((40, 40, 40))
        vertices.append(p1); colors.append((30, 120, 220))
        edges.append((a, a + 1))
    vertices.append(origin_xyz); colors.append((255, 60, 60))
    write_ply_edges(path, np.vstack(vertices), np.asarray(colors, dtype=np.uint8), edges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CASALS beam-angle conventions using common-origin convergence and sensor sanity checks.")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH, help="Input CASALS L1B H5 file.")
    parser.add_argument("--sweep", type=int, default=DEFAULT_SWEEP_INDEX, help="Sweep index to analyze.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output directory.")
    parser.add_argument("--epsg", type=int, default=None, help="Projected CRS EPSG. Default: infer WGS84 UTM from refh lon/lat.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top rows to print/export in top origins CSV.")
    parser.add_argument("--top-n-ply", type=int, default=8, help="Number of top physical rules to export as PLY ray bundles.")
    parser.add_argument("--ply-max-tracks", type=int, default=64, help="Maximum tracks per ray-bundle PLY.")
    parser.add_argument("--sort-by", choices=("physical_score", "median_line_to_fit_origin_m"), default="physical_score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    scores_dir = output_root / "scores"
    pc_dir = output_root / "pc"
    scores_dir.mkdir(parents=True, exist_ok=True)
    pc_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CASALS beam-line origin convergence and sensor sanity audit")
    print(f"H5: {args.h5.resolve()}")
    print(f"Sweep: {args.sweep}")
    print(f"Output: {output_root.resolve()}")
    print("=" * 100)

    sweep = load_sweep_data(args.h5, args.sweep)
    instr_present = sweep.instrument_lon is not None and sweep.instrument_lat is not None and sweep.instrument_z is not None
    print("Sweep loaded")
    print(f"Tracks: {sweep.n}")
    print(f"Track range: {int(np.min(sweep.track_num))}..{int(np.max(sweep.track_num))}")
    print(f"Instrument fields present: {instr_present}")

    epsg = args.epsg if args.epsg is not None else infer_wgs84_utm_epsg(sweep.refh_lon, sweep.refh_lat)
    print(f"Using UTM EPSG:{epsg}")

    refh_x, refh_y = project_lonlat(sweep.refh_lon, sweep.refh_lat, epsg)
    refh_xyz = np.column_stack((refh_x, refh_y, sweep.refh_z)).astype(np.float64)
    instr_xyz: Optional[np.ndarray] = None
    if instr_present:
        instr_x, instr_y = project_lonlat(sweep.instrument_lon, sweep.instrument_lat, epsg)
        instr_xyz = np.column_stack((instr_x, instr_y, sweep.instrument_z)).astype(np.float64)

    rules = list(iter_angle_rules())
    print("=" * 100)
    print("Scoring all angle conventions by line convergence and physical sensor sanity")
    print(f"Number of angle rules to score: {len(rules)}")

    rows: List[Dict[str, Any]] = []
    for k, rule in enumerate(rules, start=1):
        try:
            row, _ = score_rule(rule, refh_xyz, instr_xyz, sweep.beam_az, sweep.beam_el, sweep.track_num, epsg)
        except Exception as exc:
            row = {"rule_key": rule.key, **asdict(rule), "valid_count": 0, "status": f"error: {type(exc).__name__}: {exc}", "physical_score": np.inf}
        rows.append(row)
        if k % 50 == 0 or k == len(rules):
            print(f"Scored {k}/{len(rules)} rules")

    scores = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    scores["rank_by_convergence"] = scores["median_line_to_fit_origin_m"].rank(method="min", ascending=True, na_option="bottom").astype(int)
    scores["rank_by_physical_score"] = scores["physical_score"].rank(method="min", ascending=True, na_option="bottom").astype(int)

    all_scores_path = scores_dir / f"sw_{args.sweep}_origin_scores_all.csv"
    scores.to_csv(all_scores_path, index=False)

    compact_cols = [
        "rank_by_physical_score", "rank_by_convergence", "rule_key", "status", "valid_count", "physical_score",
        "median_line_to_fit_origin_m", "p90_line_to_fit_origin_m", "fit_origin_to_median_instrument_m",
        "fit_origin_horizontal_to_median_instrument_m", "fit_origin_vertical_diff_to_median_instrument_m",
        "fit_origin_altitude_m", "median_instrument_altitude_m", "fit_origin_height_above_median_refh_m",
        "median_line_to_refh_m", "median_line_to_own_instrument_m", "median_angle_error_to_segment_deg",
        "median_cosine_to_inst_to_ref", "median_signed_range_ref_to_inst_along_rule_m",
        "median_instrument_to_refh_range_m", "origin_fit_condition_number",
        "direction_forward_to_refh", "origin_near_instrument_3d_lt_100m", "origin_height_plausible_100m_to_30km_agl",
        "range_plausible_100m_to_50km", "fit_origin_longitude", "fit_origin_latitude", "median_instrument_longitude", "median_instrument_latitude",
    ]
    compact_cols = [c for c in compact_cols if c in scores.columns]
    compact = scores.sort_values(args.sort_by, ascending=True, na_position="last")[compact_cols].copy()
    compact_path = scores_dir / f"sw_{args.sweep}_origin_scores_compact.csv"
    compact.to_csv(compact_path, index=False)

    top = compact.head(int(args.top_n)).copy()
    top_origins_path = scores_dir / f"sw_{args.sweep}_fit_origins_top.csv"
    top.to_csv(top_origins_path, index=False)

    print("=" * 100)
    print(f"Top {min(args.top_n, len(top))} rules by {args.sort_by}")
    with pd.option_context("display.max_colwidth", 140, "display.width", 240):
        print(top.to_string(index=False))

    top_physical = scores.sort_values("physical_score", ascending=True, na_position="last").head(int(args.top_n_ply))
    per_track_tables = []
    for rank, (_, row) in enumerate(top_physical.iterrows(), start=1):
        rule = AngleRule(
            unit=str(row["unit"]),
            az_convention=str(row["az_convention"]),
            el_convention=str(row["el_convention"]),
            sign=str(row["sign"]),
            basis=str(row["basis"]),
            angle_transform=str(row["angle_transform"]),
        )
        print("-" * 100)
        print(f"Exporting physical-rank {rank}: {rule.key}")
        _, per_track = score_rule(rule, refh_xyz, instr_xyz, sweep.beam_az, sweep.beam_el, sweep.track_num, epsg)
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
        "source_datasets": sweep.source_datasets,
        "rule_count": int(len(rules)),
        "rule_family_count_explanation": {
            "units": len(UNITS),
            "azimuth_conventions": len(AZ_CONVENTIONS),
            "elevation_conventions": len(EL_CONVENTIONS),
            "signs": len(SIGNS),
            "bases": len(BASES),
            "angle_transforms": len(ANGLE_TRANSFORMS),
            "total": int(len(rules)),
        },
        "outputs": {
            "all_scores_csv": str(all_scores_path),
            "compact_scores_csv": str(compact_path),
            "top_origins_csv": str(top_origins_path),
            "top_rule_per_track_diagnostics_csv": None if per_track_path is None else str(per_track_path),
            "marker_ply": str(pc_dir / f"sw_{args.sweep}_markers.ply"),
            "ray_bundle_ply_dir": str(pc_dir),
        },
        "notes": [
            "physical_score is a convenience ranking only; inspect raw metrics before choosing a convention.",
            "fit_origin_altitude_m and median_instrument_altitude_m are compared in the same height field as stored in the H5.",
            "This script validates refh beam-line geometry only; it does not establish waveform-bin geolocation.",
        ],
    }
    meta_path = output_root / f"sw_{args.sweep}_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 100)
    print("Done.")
    print(f"All scores: {all_scores_path}")
    print(f"Compact scores: {compact_path}")
    print(f"Top origins: {top_origins_path}")
    if per_track_path is not None:
        print(f"Per-track diagnostics: {per_track_path}")
    print(f"Marker PLY: {pc_dir / f'sw_{args.sweep}_markers.ply'}")
    print(f"Metadata: {meta_path}")
    print(f"Point clouds: {pc_dir}")


if __name__ == "__main__":
    main()
