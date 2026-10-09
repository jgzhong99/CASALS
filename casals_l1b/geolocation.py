"""Experimental geometry for already detected CASALS RX waveform peaks.

The segment model interpolates the H5 ``rwstart`` to ``rwstop`` geodetic
endpoints by RX-bin fraction. The beam model uses the audited local-angle
convention and an explicit official-refh anchor. Neither model is an official
CASALS multi-return geolocation product.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer

from .h5 import find_dataset
from .waveform import infer_waveform_record_axis, read_waveform_records


ECEF_CRS = "EPSG:4978"
GEODETIC_CRS = "EPSG:4979"
SEGMENT_METHOD = "segment_rwstart_to_rwstop_linear_bin_fraction"
BEAM_METHOD = "beam_refh_anchor_stored_bin_size"
REQUIRED_COMPONENT_COLUMNS = {
    "pulse_index", "sweep_num", "track_num", "component_rank", "peak_bin",
    "amplitude_raw", "prominence", "is_main_component", "is_valid_secondary_candidate",
}
OUTPUT_COLUMNS = [
    "source_h5", "pulse_index", "sweep_num", "track_num", "component_rank", "peak_bin",
    "is_main_component", "is_valid_secondary_candidate", "amplitude_raw", "prominence",
    "geolocation_method", "candidate_x", "candidate_y", "candidate_z", "candidate_lon",
    "candidate_lat", "candidate_height", "coordinate_crs", "height_reference",
    "geometry_valid", "geometry_status", "segment_fraction", "raw_argmax_bin",
    "reference_bin", "used_refh_anchor", "model_variant", "closure_error_3d_m",
    "closure_horizontal_error_m", "closure_vertical_difference_m",
    "beam_segment_discrepancy_m",
]

_TO_ECEF = Transformer.from_crs(GEODETIC_CRS, ECEF_CRS, always_xy=True)
_FROM_ECEF = Transformer.from_crs(ECEF_CRS, GEODETIC_CRS, always_xy=True)
_GEOD = Geod(ellps="WGS84")


def geodetic_to_ecef(lon: Any, lat: Any, height: Any) -> np.ndarray:
    """Convert WGS84 longitude/latitude and ellipsoidal height to ECEF meters."""
    x, y, z = _TO_ECEF.transform(lon, lat, height)
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1).astype(np.float64, copy=False)


def ecef_to_geodetic(xyz: Any) -> np.ndarray:
    """Convert ECEF meter coordinates to WGS84 longitude/latitude/height."""
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError("ECEF coordinates must have a final dimension of length 3")
    lon, lat, height = _FROM_ECEF.transform(arr[..., 0], arr[..., 1], arr[..., 2])
    return np.stack(np.broadcast_arrays(lon, lat, height), axis=-1).astype(np.float64, copy=False)


def interpolate_segment(start_xyz: Any, stop_xyz: Any, fraction: Any) -> np.ndarray:
    """Interpolate ECEF endpoints; fraction is RX bin / (number of bins - 1)."""
    start = np.asarray(start_xyz, dtype=np.float64)
    stop = np.asarray(stop_xyz, dtype=np.float64)
    u = np.asarray(fraction, dtype=np.float64)
    if start.shape[-1] != 3 or stop.shape[-1] != 3:
        raise ValueError("Segment endpoints must have a final dimension of length 3")
    return start + np.expand_dims(u, -1) * (stop - start)


def beam_direction_enu(azimuth_rad: Any, elevation_rad: Any) -> np.ndarray:
    """Return the audited down-looking ENU unit vector.

    Convention: azimuth is radians clockwise from north, elevation is radians
    above the horizon, and the sign is reversed to point from the aircraft
    toward the ground.
    """
    az = np.asarray(azimuth_rad, dtype=np.float64)
    el = np.asarray(elevation_rad, dtype=np.float64)
    return np.stack(
        (-np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), -np.sin(el)), axis=-1
    )


def enu_to_ecef_direction(direction_enu: Any, lon: Any, lat: Any) -> np.ndarray:
    """Rotate local east/north/up vectors into ECEF at the supplied location."""
    vec = np.asarray(direction_enu, dtype=np.float64)
    lon_r, lat_r = np.deg2rad(np.asarray(lon, dtype=np.float64)), np.deg2rad(np.asarray(lat, dtype=np.float64))
    east = np.stack((-np.sin(lon_r), np.cos(lon_r), np.zeros_like(lon_r)), axis=-1)
    north = np.stack(
        (-np.sin(lat_r) * np.cos(lon_r), -np.sin(lat_r) * np.sin(lon_r), np.cos(lat_r)), axis=-1
    )
    up = np.stack(
        (np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)), axis=-1
    )
    out = vec[..., 0, None] * east + vec[..., 1, None] * north + vec[..., 2, None] * up
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return np.divide(out, norm, out=np.full_like(out, np.nan), where=norm > 0)


def _finite_triplet(lon: float, lat: float, height: float) -> bool:
    return bool(np.isfinite(lon) and np.isfinite(lat) and np.isfinite(height) and abs(lat) <= 90)


def _read_records(ds: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read sorted unique record indices and restore their requested order."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.empty(0, dtype=ds.dtype)
    unique, inverse = np.unique(idx, return_inverse=True)
    if unique[0] < 0 or unique[-1] >= ds.shape[0]:
        raise IndexError(f"Pulse index outside {ds.name}: {unique[0]}..{unique[-1]}")
    return np.asarray(ds[unique])[inverse]


def _read_context(h5: h5py.File, pulse_ids: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    names = [
        "sweep_num", "track_num", "rwstart_longitude", "rwstart_latitude", "rwstart",
        "rwstop_longitude", "rwstop_latitude", "rwstop", "refh_longitude", "refh_latitude",
        "refh", "bin_size", "local_beam_azimuth", "local_beam_elevation", "instrument_longitude",
        "instrument_latitude", "instrument_altitude", "refh_bounce_time_offset",
        "rwstart_bounce_time_offset", "rwstop_bounce_time_offset",
    ]
    fields: dict[str, np.ndarray] = {}
    for name in names:
        ds = find_dataset(h5, name)
        if ds is None:
            continue
        values = []
        for start in range(0, pulse_ids.size, batch_size):
            values.append(_read_records(ds, pulse_ids[start : start + batch_size]))
        fields[name] = np.concatenate(values) if values else np.empty(0, dtype=ds.dtype)

    rx = find_dataset(h5, "rx_waveform")
    if rx is None:
        fields["raw_argmax_bin"] = np.full(pulse_ids.size, np.nan)
        fields["n_rx_bins"] = np.full(pulse_ids.size, np.nan)
    else:
        sweep_ds = find_dataset(h5, "sweep_num")
        n_records = int(sweep_ds.shape[0])
        axis = infer_waveform_record_axis(rx, n_records)
        n_bins = int(rx.shape[1] if axis == 0 else rx.shape[0])
        argmax = np.full(pulse_ids.size, np.nan, dtype=np.float64)
        for start in range(0, pulse_ids.size, batch_size):
            stop = min(start + batch_size, pulse_ids.size)
            matrix = read_waveform_records(rx, axis, pulse_ids[start:stop])
            finite = np.isfinite(matrix)
            has_values = finite.any(axis=1)
            safe = np.where(finite, matrix, -np.inf)
            argmax[start:stop][has_values] = np.argmax(safe[has_values], axis=1)
        fields["raw_argmax_bin"] = argmax
        fields["n_rx_bins"] = np.full(pulse_ids.size, n_bins, dtype=np.int32)
    return fields


def _metrics(errors: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(errors, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None}
    return {
        "count": int(vals.size),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
    }


def _closure_rows(fields: dict[str, np.ndarray], pulse_ids: np.ndarray) -> pd.DataFrame:
    n = pulse_ids.size
    rows: list[dict[str, Any]] = []
    names = ("rwstart_longitude", "rwstart_latitude", "rwstart", "rwstop_longitude", "rwstop_latitude", "rwstop", "refh_longitude", "refh_latitude", "refh", "raw_argmax_bin", "n_rx_bins", "sweep_num")
    missing = [name for name in names if name not in fields]
    if missing:
        return pd.DataFrame([{"scope": "refh_closure", "method": SEGMENT_METHOD, "status": "unavailable", "reason": "missing_fields:" + ",".join(missing), "n_total": int(n), "n_valid": 0, "valid_fraction": 0.0}])
    for i, pulse in enumerate(pulse_ids):
        row = {"pulse_index": int(pulse), "sweep_num": int(fields["sweep_num"][i]), "status": "ok", "reason": ""}
        bin_idx, n_bins = float(fields["raw_argmax_bin"][i]), int(fields["n_rx_bins"][i])
        values = [float(fields[k][i]) for k in names[:9]]
        if not np.all(np.isfinite(values)):
            row.update(status="invalid_geometry", reason="missing_or_nonfinite_endpoint_or_refh")
        elif not np.isfinite(bin_idx) or n_bins <= 1 or bin_idx < 0 or bin_idx > n_bins - 1:
            row.update(status="invalid_geometry", reason="invalid_raw_argmax_bin")
        else:
            start_xyz = geodetic_to_ecef(values[0], values[1], values[2])
            stop_xyz = geodetic_to_ecef(values[3], values[4], values[5])
            pred_xyz = interpolate_segment(start_xyz, stop_xyz, bin_idx / (n_bins - 1))
            ref_xyz = geodetic_to_ecef(values[6], values[7], values[8])
            delta = pred_xyz - ref_xyz
            pred_geo = ecef_to_geodetic(pred_xyz)
            _, _, horizontal = _GEOD.inv(values[6], values[7], float(pred_geo[0]), float(pred_geo[1]))
            row.update(
                error_3d_m=float(np.linalg.norm(delta)),
                horizontal_error_m=float(abs(horizontal)),
                vertical_difference_m=float(pred_geo[2] - values[8]),
                status="ok",
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_row(scope: str, method: str, values: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"scope": scope, "method": method, **values, **extra}


def _validation_summary(components: pd.DataFrame, candidates: pd.DataFrame, closure: pd.DataFrame, fields: dict[str, np.ndarray], pulse_ids: np.ndarray) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    if not closure.empty and "scope" not in closure.columns:
        valid = closure[closure["status"] == "ok"]
        total = int(len(closure))
        for metric, target in [("error_3d_m", "3d"), ("horizontal_error_m", "horizontal"), ("vertical_difference_m", "vertical")]:
            m = _metrics(valid[metric].to_numpy() if metric in valid else np.array([]))
            summaries.append(_summary_row("refh_closure", SEGMENT_METHOD, {f"{target}_{k}": v for k, v in m.items()}, n_total=total, n_valid=int(len(valid)), valid_fraction=float(len(valid) / total) if total else 0.0, note="raw RX argmax bin; no fitted parameters"))
        for sweep, group in valid.groupby("sweep_num"):
            m = _metrics(group["error_3d_m"].to_numpy())
            summaries.append(_summary_row("refh_closure_by_sweep", SEGMENT_METHOD, {f"3d_{k}": v for k, v in m.items()}, sweep_num=int(sweep), n_total=int((closure["sweep_num"] == sweep).sum()), n_valid=int(len(group))))
        invalid = closure[closure["status"] != "ok"]
        for reason, group in invalid.groupby("reason"):
            summaries.append(_summary_row("invalid_reason", SEGMENT_METHOD, {"count": int(len(group))}, reason=str(reason)))
    elif not closure.empty:
        summaries.append(_summary_row("refh_closure", SEGMENT_METHOD, {"status": "unavailable", "reason": str(closure.iloc[0].get("reason", "unknown")), "n_total": int(closure.iloc[0].get("n_total", 0)), "n_valid": 0, "valid_fraction": 0.0}))

    for method, group in candidates.groupby("geolocation_method", dropna=False):
        valid = group[group["geometry_valid"].astype(bool)]
        summaries.append(_summary_row("candidate_geometry", str(method), {"n_total": int(len(group)), "n_valid": int(len(valid)), "valid_fraction": float(len(valid) / len(group)) if len(group) else 0.0}, note="valid means finite modeled coordinates; not independent accuracy"))
        for reason, rejected in group.loc[~group["geometry_valid"].astype(bool)].groupby("geometry_status"):
            summaries.append(_summary_row("invalid_reason", str(method), {"count": int(len(rejected))}, reason=str(reason)))

    if not candidates.empty:
        wide = candidates.pivot_table(index=["pulse_index", "component_rank"], columns="geolocation_method", values=["candidate_x", "candidate_y", "candidate_z", "geometry_valid"], aggfunc="first")
        if SEGMENT_METHOD in wide.columns.get_level_values(1) and BEAM_METHOD in wide.columns.get_level_values(1):
            both = wide[("geometry_valid", SEGMENT_METHOD)].fillna(False).astype(bool) & wide[("geometry_valid", BEAM_METHOD)].fillna(False).astype(bool)
            if both.any():
                dx = wide.loc[both, ("candidate_x", SEGMENT_METHOD)] - wide.loc[both, ("candidate_x", BEAM_METHOD)]
                dy = wide.loc[both, ("candidate_y", SEGMENT_METHOD)] - wide.loc[both, ("candidate_y", BEAM_METHOD)]
                dz = wide.loc[both, ("candidate_z", SEGMENT_METHOD)] - wide.loc[both, ("candidate_z", BEAM_METHOD)]
                m = _metrics(np.sqrt(dx.to_numpy() ** 2 + dy.to_numpy() ** 2 + dz.to_numpy() ** 2))
                summaries.append(_summary_row("beam_segment_agreement", "segment_vs_beam", {f"xyz_discrepancy_{k}_m": v for k, v in m.items()}, n_common=int(both.sum()), note="internal method agreement only; beam candidates use refh anchors"))
            else:
                summaries.append(_summary_row("beam_segment_agreement", "segment_vs_beam", {"n_common": 0, "status": "no_common_valid_candidates"}))

    if pulse_ids.size and all(k in fields for k in ("rwstart_longitude", "rwstart_latitude", "rwstart", "rwstop_longitude", "rwstop_latitude", "rwstop", "bin_size", "n_rx_bins")):
        start = geodetic_to_ecef(fields["rwstart_longitude"], fields["rwstart_latitude"], fields["rwstart"])
        stop = geodetic_to_ecef(fields["rwstop_longitude"], fields["rwstop_latitude"], fields["rwstop"])
        implied = np.linalg.norm(stop - start, axis=1) / (fields["n_rx_bins"] - 1)
        stored = np.asarray(fields["bin_size"], dtype=np.float64)
        valid = np.isfinite(implied) & np.isfinite(stored)
        if np.any(valid):
            diff = implied[valid] - stored[valid]
            summaries.append(_summary_row("bin_mapping", "segment_length_over_n_bins_minus_1_vs_stored_bin_size", {"n_total": int(pulse_ids.size), "n_valid": int(valid.sum()), "implied_bin_size_median_m": float(np.median(implied[valid])), "stored_bin_size_median_m": float(np.median(stored[valid])), "difference_median_m": float(np.median(diff)), "difference_p95_abs_m": float(np.percentile(np.abs(diff), 95))}, note="stored bin_size is interpreted in meters from the audited H5 product convention"))
    return pd.DataFrame(summaries)


def _attach_method_discrepancy(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or candidates["geolocation_method"].nunique() < 2:
        return candidates
    key = ["pulse_index", "component_rank"]
    values = candidates.pivot_table(index=key, columns="geolocation_method", values=["candidate_x", "candidate_y", "candidate_z", "geometry_valid"], aggfunc="first")
    if not all(("geometry_valid", method) in values.columns for method in (SEGMENT_METHOD, BEAM_METHOD)):
        return candidates
    both = values[("geometry_valid", SEGMENT_METHOD)].fillna(False).astype(bool) & values[("geometry_valid", BEAM_METHOD)].fillna(False).astype(bool)
    discrepancy: dict[tuple[int, int], float] = {}
    for idx in values.index[both]:
        dx = values.loc[idx, ("candidate_x", SEGMENT_METHOD)] - values.loc[idx, ("candidate_x", BEAM_METHOD)]
        dy = values.loc[idx, ("candidate_y", SEGMENT_METHOD)] - values.loc[idx, ("candidate_y", BEAM_METHOD)]
        dz = values.loc[idx, ("candidate_z", SEGMENT_METHOD)] - values.loc[idx, ("candidate_z", BEAM_METHOD)]
        discrepancy[(int(idx[0]), int(idx[1]))] = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    candidates = candidates.copy()
    candidates["beam_segment_discrepancy_m"] = [
        discrepancy.get((int(p), int(rank)), np.nan) if pd.notna(p) and pd.notna(rank) else np.nan
        for p, rank in zip(candidates["pulse_index"], candidates["component_rank"])
    ]
    return candidates


def locate_peaks(
    h5_path: str | Path,
    component_table: str | Path,
    output_dir: str | Path,
    method: str = "both",
    *,
    batch_size: int = 512,
) -> dict[str, Path]:
    """Map existing detector components to experimental segment/beam candidates."""
    h5_path, component_table, output_dir = Path(h5_path), Path(component_table), Path(output_dir)
    if method not in {"segment", "beam", "both"}:
        raise ValueError("method must be 'segment', 'beam', or 'both'")
    components = pd.read_parquet(component_table).reset_index(drop=True)
    missing = sorted(REQUIRED_COMPONENT_COLUMNS - set(components.columns))
    if missing:
        raise ValueError(f"component table is missing required columns: {missing}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    pulse_values = pd.to_numeric(components["pulse_index"], errors="coerce").to_numpy(dtype=np.float64)
    valid_pulse = np.isfinite(pulse_values) & (pulse_values >= 0) & (pulse_values == np.floor(pulse_values))
    pulse_ids = np.unique(pulse_values[valid_pulse].astype(np.int64))
    with h5py.File(h5_path, "r") as h5:
        fields = _read_context(h5, pulse_ids, batch_size)
        if "sweep_num" not in fields or "track_num" not in fields:
            raise ValueError("H5 is missing sweep_num or track_num; pulse associations cannot be checked")
        pulse_pos = np.searchsorted(pulse_ids, np.where(valid_pulse, pulse_values, 0).astype(np.int64)) if pulse_ids.size else np.zeros(len(components), dtype=np.int64)
        rows: list[dict[str, Any]] = []
        methods = [SEGMENT_METHOD, BEAM_METHOD] if method == "both" else [SEGMENT_METHOD if method == "segment" else BEAM_METHOD]
        for i, component in components.iterrows():
            p = float(pulse_values[i])
            common = {
                "source_h5": h5_path.name,
                "pulse_index": int(p) if valid_pulse[i] else None,
                "sweep_num": int(component["sweep_num"]) if pd.notna(component["sweep_num"]) else None,
                "track_num": int(component["track_num"]) if pd.notna(component["track_num"]) else None,
                "component_rank": int(component["component_rank"]) if pd.notna(component["component_rank"]) else None,
                "peak_bin": float(component["peak_bin"]) if pd.notna(component["peak_bin"]) else np.nan,
                "is_main_component": bool(component["is_main_component"]) if pd.notna(component["is_main_component"]) else False,
                "is_valid_secondary_candidate": bool(component["is_valid_secondary_candidate"]) if pd.notna(component["is_valid_secondary_candidate"]) else False,
                "amplitude_raw": float(component["amplitude_raw"]) if pd.notna(component["amplitude_raw"]) else np.nan,
                "prominence": float(component["prominence"]) if pd.notna(component["prominence"]) else np.nan,
            }
            if not valid_pulse[i]:
                context = None
                initial_status = "invalid_pulse_index"
            else:
                pos = int(pulse_pos[i])
                context = {key: value[pos] for key, value in fields.items()}
                initial_status = "ok"
                if int(context["sweep_num"]) != common["sweep_num"] or int(context["track_num"]) != common["track_num"]:
                    initial_status = "pulse_index_sweep_track_mismatch"

            for chosen_method in methods:
                point = {**common, "geolocation_method": chosen_method, "candidate_x": np.nan, "candidate_y": np.nan, "candidate_z": np.nan, "candidate_lon": np.nan, "candidate_lat": np.nan, "candidate_height": np.nan, "coordinate_crs": ECEF_CRS, "height_reference": "WGS84 ellipsoidal", "geometry_valid": False, "geometry_status": initial_status, "segment_fraction": np.nan, "raw_argmax_bin": np.nan, "reference_bin": np.nan, "used_refh_anchor": chosen_method == BEAM_METHOD, "model_variant": chosen_method, "closure_error_3d_m": np.nan, "closure_horizontal_error_m": np.nan, "closure_vertical_difference_m": np.nan, "beam_segment_discrepancy_m": np.nan}
                if context is None or initial_status != "ok":
                    rows.append(point)
                    continue
                peak_bin = common["peak_bin"]
                n_bins = int(context.get("n_rx_bins", np.nan)) if np.isfinite(context.get("n_rx_bins", np.nan)) else 0
                raw_bin = float(context.get("raw_argmax_bin", np.nan))
                point["raw_argmax_bin"] = raw_bin
                point["reference_bin"] = raw_bin
                if n_bins <= 1 or not np.isfinite(peak_bin) or peak_bin < 0 or peak_bin > n_bins - 1:
                    point["geometry_status"] = "invalid_peak_bin"
                    rows.append(point)
                    continue

                xyz = None
                if chosen_method == SEGMENT_METHOD:
                    endpoint_names = ("rwstart_longitude", "rwstart_latitude", "rwstart", "rwstop_longitude", "rwstop_latitude", "rwstop")
                    if not all(name in context for name in endpoint_names):
                        point["geometry_status"] = "missing_segment_fields"
                    elif not _finite_triplet(*[float(context[name]) for name in endpoint_names[:3]]) or not _finite_triplet(*[float(context[name]) for name in endpoint_names[3:]]):
                        point["geometry_status"] = "invalid_segment_endpoint"
                    else:
                        u = peak_bin / (n_bins - 1)
                        point["segment_fraction"] = u
                        start_xyz = geodetic_to_ecef(context[endpoint_names[0]], context[endpoint_names[1]], context[endpoint_names[2]])
                        stop_xyz = geodetic_to_ecef(context[endpoint_names[3]], context[endpoint_names[4]], context[endpoint_names[5]])
                        xyz = interpolate_segment(start_xyz, stop_xyz, u)
                else:
                    beam_names = ("local_beam_azimuth", "local_beam_elevation", "instrument_longitude", "instrument_latitude", "instrument_altitude", "refh_longitude", "refh_latitude", "refh", "bin_size")
                    if not all(name in context for name in beam_names) or not np.isfinite(raw_bin):
                        point["geometry_status"] = "missing_beam_range_or_refh_anchor"
                    elif not _finite_triplet(context["instrument_longitude"], context["instrument_latitude"], context["instrument_altitude"]) or not _finite_triplet(context["refh_longitude"], context["refh_latitude"], context["refh"]):
                        point["geometry_status"] = "invalid_beam_anchor_or_instrument_position"
                    elif not np.isfinite(context["bin_size"]) or float(context["bin_size"]) <= 0:
                        point["geometry_status"] = "invalid_bin_size"
                    else:
                        enu = beam_direction_enu(context["local_beam_azimuth"], context["local_beam_elevation"])
                        direction = enu_to_ecef_direction(enu, context["instrument_longitude"], context["instrument_latitude"])
                        anchor = geodetic_to_ecef(context["refh_longitude"], context["refh_latitude"], context["refh"])
                        xyz = anchor + (peak_bin - raw_bin) * float(context["bin_size"]) * direction
                if xyz is not None and np.all(np.isfinite(xyz)):
                    geo = ecef_to_geodetic(xyz)
                    point.update(candidate_x=float(xyz[0]), candidate_y=float(xyz[1]), candidate_z=float(xyz[2]), candidate_lon=float(geo[0]), candidate_lat=float(geo[1]), candidate_height=float(geo[2]), geometry_valid=True, geometry_status="ok")
                    if all(name in context for name in ("refh_longitude", "refh_latitude", "refh")) and _finite_triplet(context["refh_longitude"], context["refh_latitude"], context["refh"]):
                        refh_xyz = geodetic_to_ecef(context["refh_longitude"], context["refh_latitude"], context["refh"])
                        point["closure_error_3d_m"] = float(np.linalg.norm(xyz - refh_xyz))
                        _, _, horizontal = _GEOD.inv(context["refh_longitude"], context["refh_latitude"], geo[0], geo[1])
                        point["closure_horizontal_error_m"] = float(abs(horizontal))
                        point["closure_vertical_difference_m"] = float(geo[2] - context["refh"])
                elif xyz is not None:
                    point["geometry_status"] = "invalid_calculated_coordinate"
                rows.append(point)

    candidate_df = _attach_method_discrepancy(pd.DataFrame(rows, columns=OUTPUT_COLUMNS))
    # The segment closure is computed on raw waveform argmax bins, independently
    # of detector component ranks and without fitting to refh.
    closure = _closure_rows(fields, pulse_ids)
    summary = _validation_summary(components, candidate_df, closure, fields, pulse_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_points.parquet"
    closure_path = output_dir / "closure_residuals.csv"
    summary_path = output_dir / "validation_summary.csv"
    metadata_path = output_dir / "run_metadata.json"
    candidate_df.to_parquet(candidate_path, index=False)
    closure.to_csv(closure_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "source_h5": str(h5_path.resolve()),
        "component_table": str(component_table.resolve()),
        "method_selection": method,
        "methods": methods,
        "n_component_rows": int(len(components)),
        "n_unique_pulses": int(len(pulse_ids)),
        "candidate_points": str(candidate_path.resolve()),
        "closure_residuals": str(closure_path.resolve()),
        "validation_summary": str(summary_path.resolve()),
        "coordinates": {"candidate_x_y_z": "EPSG:4978 ECEF meters", "candidate_lon_lat_height": "EPSG:4979 WGS84 geodetic; ellipsoidal height"},
        "segment_model": {"formula": "rwstart ECEF + (peak_bin / (n_rx_bins - 1)) * (rwstop ECEF - rwstart ECEF)", "variant": SEGMENT_METHOD, "refh_anchor_used": False},
        "beam_model": {"variant": BEAM_METHOD, "angle_unit": "radian", "azimuth": "clockwise from north", "elevation": "above horizon", "direction_sign": "negated toward ground", "range": "(peak_bin - raw_argmax_bin) * H5 bin_size in meters", "refh_anchor_used": True, "closure_is_independent": False},
        "status_semantics": {"ok": "finite experimental candidate coordinates were calculated", "other": "candidate unavailable; coordinates remain NaN"},
        "notes": ["No peak detector is run in this module.", "Raw RX argmax is computed only for bin-mapping and refh-closure diagnostics.", "Refh closure tests internal consistency and is not independent absolute geolocation accuracy."],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"candidate_points": candidate_path, "closure_residuals": closure_path, "validation_summary": summary_path, "run_metadata": metadata_path}


def validate_candidate_table(h5_path: str | Path, candidates_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Recheck H5 pulse identity and candidate-table validity without peak detection."""
    h5_path, candidates_path, output_dir = Path(h5_path), Path(candidates_path), Path(output_dir)
    candidates = pd.read_parquet(candidates_path).reset_index(drop=True)
    missing = sorted(set(OUTPUT_COLUMNS) - set(candidates.columns))
    if missing:
        raise ValueError(f"candidate table is missing required columns: {missing}")
    pulse_values = pd.to_numeric(candidates["pulse_index"], errors="coerce").dropna().to_numpy(dtype=np.int64)
    pulse_ids = np.unique(pulse_values)
    with h5py.File(h5_path, "r") as h5:
        fields = _read_context(h5, pulse_ids, 512)
        matches = np.ones(len(candidates), dtype=bool)
        for i, row in candidates.iterrows():
            if pd.isna(row["pulse_index"]):
                matches[i] = False
                continue
            pos = int(np.searchsorted(pulse_ids, int(row["pulse_index"])))
            matches[i] = int(fields["sweep_num"][pos]) == int(row["sweep_num"]) and int(fields["track_num"][pos]) == int(row["track_num"])
    valid = candidates["geometry_valid"].astype(bool).to_numpy()
    xyz_finite = np.isfinite(candidates[["candidate_x", "candidate_y", "candidate_z"]].to_numpy(dtype=np.float64)).all(axis=1)
    geo_finite = np.isfinite(candidates[["candidate_lon", "candidate_lat", "candidate_height"]].to_numpy(dtype=np.float64)).all(axis=1)
    rows = []
    for method, group_idx in candidates.groupby("geolocation_method").groups.items():
        idx = np.asarray(list(group_idx), dtype=np.int64)
        group_valid = valid[idx]
        good = matches[idx] & ((~group_valid) | (xyz_finite[idx] & geo_finite[idx]))
        rows.append({"scope": "candidate_table", "method": str(method), "n_total": int(idx.size), "n_valid": int(group_valid.sum()), "valid_fraction": float(group_valid.mean()) if idx.size else 0.0, "n_pulse_identity_mismatch": int((~matches[idx]).sum()), "n_coordinate_status_mismatch": int((~good).sum()), "status": "pass" if bool(np.all(good)) else "fail", "note": "schema and H5 sweep/track identity checks; geometric plausibility is summarized separately"})
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "validation_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps({"source_h5": str(h5_path.resolve()), "candidate_table": str(candidates_path.resolve()), "checks": ["required output schema", "pulse_index to H5 sweep_num/track_num identity", "geometry_valid coordinates are finite"], "passed": bool(all(row["status"] == "pass" for row in rows))}, indent=2), encoding="utf-8")
    return {"validation_summary": summary_path, "run_metadata": metadata_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    locate = sub.add_parser("locate", help="map an existing component table to candidate positions")
    locate.add_argument("--h5", type=Path, required=True)
    locate.add_argument("--components", type=Path, required=True)
    locate.add_argument("--method", choices=["segment", "beam", "both"], default="both")
    locate.add_argument("--output-dir", type=Path, default=Path("outputs/peaks/geolocation"))
    validate = sub.add_parser("validate", help="validate an existing candidate table")
    validate.add_argument("--h5", type=Path, required=True)
    validate.add_argument("--candidates", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, default=Path("outputs/peaks/validation"))
    args = parser.parse_args(argv)
    if args.action == "locate":
        result = locate_peaks(args.h5, args.components, args.output_dir, args.method)
    else:
        result = validate_candidate_table(args.h5, args.candidates, args.output_dir)
    for name, path in result.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
