"""Small, diagnostic helpers for CASALS L1B research validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS
from pyproj.transformer import AreaOfInterest, TransformDirection, TransformerGroup
from scipy.signal import find_peaks, peak_prominences, peak_widths


_REFERENCE_EVIDENCE = (
    (
        "casals_horizontal_reference_verified",
        "CASALS_HORIZONTAL_FRAME_UNVERIFIED",
        "CASALS horizontal datum realization and coordinate epoch are not verified.",
    ),
    (
        "casals_vertical_reference_verified",
        "CASALS_VERTICAL_FRAME_UNVERIFIED",
        "CASALS Refh vertical frame and processor corrections are not verified.",
    ),
    (
        "reference_horizontal_crs_verified",
        "REFERENCE_HORIZONTAL_CRS_UNVERIFIED",
        "The 3DEP horizontal reference metadata is not independently complete.",
    ),
    (
        "reference_vertical_frame_verified",
        "REFERENCE_VERTICAL_FRAME_UNVERIFIED",
        "The 3DEP vertical CRS is absent or unverified in the available source metadata.",
    ),
    (
        "historical_z_processing_verified",
        "REFERENCE_Z_PROCESSING_UNVERIFIED",
        "The historical 3DEP clipping and Z handling have not been verified.",
    ),
    (
        "reference_source_provenance_verified",
        "REFERENCE_SOURCE_PROVENANCE_UNVERIFIED",
        "The original 3DEP source/work-unit provenance is not fully available.",
    ),
    (
        "best_required_transform_available",
        "REQUIRED_TRANSFORM_UNAVAILABLE",
        "The required non-ballpark coordinate operation and grids are unavailable.",
    ),
)


def reference_readiness(evidence: Mapping[str, bool]) -> dict[str, Any]:
    """Gate absolute elevation errors on explicit, complete frame evidence.

    Missing keys are treated as unverified. No empirical offset or other
    fallback can make a blocked reference comparison pass this gate.
    """

    blockers = []
    for key, code, detail in _REFERENCE_EVIDENCE:
        if evidence.get(key) is not True:
            blockers.append({"evidence_key": key, "code": code, "detail": detail})
    return {
        "status": "VERIFIED" if not blockers else "BLOCKED",
        "absolute_elevation_errors_permitted": not blockers,
        "blockers": blockers,
    }


def transform_xyz_with_verified_frames(
    coordinates: np.ndarray,
    source_crs: Any,
    target_crs: Any,
    evidence: Mapping[str, bool],
    *,
    area_of_interest: tuple[float, float, float, float] | None = None,
    roundtrip_tolerance: tuple[float, float, float] = (1e-8, 1e-8, 1e-4),
) -> dict[str, Any]:
    """Apply an available non-ballpark 3D operation only after the evidence gate passes.

    Coordinates use x, y, z order. Round-trip tolerances use each source axis unit.
    """

    readiness = reference_readiness(evidence)
    if readiness["status"] != "VERIFIED":
        codes = ", ".join(item["code"] for item in readiness["blockers"])
        raise ValueError(f"Reference evidence blocks XYZ transformation: {codes}")

    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim == 0 or xyz.shape[-1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError("Coordinates must be finite x, y, z values.")
    source = CRS.from_user_input(source_crs)
    target = CRS.from_user_input(target_crs)
    if len(source.axis_info) != 3 or len(target.axis_info) != 3:
        raise ValueError("Both source and target CRS must define three axes.")
    if any(not axis.unit_name for axis in (*source.axis_info, *target.axis_info)):
        raise ValueError("All source and target axes must declare units.")

    aoi = AreaOfInterest(*area_of_interest) if area_of_interest is not None else None
    group = TransformerGroup(
        source,
        target,
        always_xy=True,
        area_of_interest=aoi,
        allow_ballpark=False,
    )
    if not group.best_available or not group.transformers:
        raise RuntimeError("The best non-ballpark 3D coordinate operation is unavailable.")

    tolerance = np.asarray(roundtrip_tolerance, dtype=np.float64)
    if tolerance.shape != (3,) or not np.all(np.isfinite(tolerance)) or np.any(tolerance <= 0):
        raise ValueError("Round-trip tolerance must contain three finite positive values.")

    transformer = group.transformers[0]
    transformed = np.stack(
        transformer.transform(xyz[..., 0], xyz[..., 1], xyz[..., 2], errcheck=True),
        axis=-1,
    )
    returned = np.stack(
        transformer.transform(
            transformed[..., 0],
            transformed[..., 1],
            transformed[..., 2],
            direction=TransformDirection.INVERSE,
            errcheck=True,
        ),
        axis=-1,
    )
    max_error = np.max(np.abs(returned - xyz).reshape(-1, 3), axis=0)
    if not np.all(np.isfinite(transformed)) or np.any(max_error > tolerance):
        raise RuntimeError("XYZ transformation failed its finite-value or round-trip check.")
    return {
        "status": "VERIFIED",
        "transformed_xyz": transformed.tolist(),
        "roundtrip_max_abs_error": max_error.tolist(),
        "roundtrip_tolerance": tolerance.tolist(),
        "source_axis_units": [axis.unit_name for axis in source.axis_info],
        "target_axis_units": [axis.unit_name for axis in target.axis_info],
        "operation": transformer.description,
        "operation_accuracy_m": float(transformer.accuracy),
    }


def waveform_response_features(
    waveform: np.ndarray,
    background_mean: float,
    background_std: float,
    *,
    threshold_sigma: float = 3.0,
    minimum_peak_distance: int = 5,
    minimum_finite_fraction: float = 0.95,
) -> dict[str, Any]:
    """Measure descriptive RX morphology without dropping no-detect pulses.

    The threshold is a screening convention, not the production detector and
    not a calibrated return criterion. Width is the contiguous half-maximum
    span of the strongest positive background-relative peak.
    """

    source = np.asarray(waveform)
    values = np.asarray(source, dtype=np.float64).reshape(-1)
    clipped = False
    clipped_bin_count = 0
    if np.issubdtype(source.dtype, np.integer):
        dtype_limits = np.iinfo(source.dtype)
        clipped_mask = (values == dtype_limits.min) | (values == dtype_limits.max)
        clipped = bool(np.any(clipped_mask))
        clipped_bin_count = int(np.count_nonzero(clipped_mask))

    if values.size == 0:
        return {
            "waveform_status": "invalid_waveform",
            "finite_fraction": 0.0,
            "raw_max_counts": None,
            "is_clipped": clipped,
            "clipped_bin_count": clipped_bin_count,
            "peak_excess_counts": None,
            "peak_snr_sigma": None,
            "peak_bin": None,
            "width_halfmax_bins": None,
            "active_bin_count": 0,
            "signal_extent_bins": None,
            "signal_support_component_count": 0,
            "dominant_support_start_bin": None,
            "dominant_support_end_bin": None,
            "dominant_support_width_bins": None,
            "dominant_support_energy_counts_bin": None,
            "energy_centroid_bin": None,
            "energy_p10_bin": None,
            "energy_p50_bin": None,
            "energy_p90_bin": None,
            "asymmetry_skewness": None,
            "signal_support_components": [],
            "peak_count": 0,
        }

    finite = np.isfinite(values)
    finite_fraction = float(np.mean(finite))
    invalid_result = {
        "finite_fraction": finite_fraction,
        "raw_max_counts": float(np.max(values[finite])) if np.any(finite) else None,
        "is_clipped": clipped,
        "clipped_bin_count": clipped_bin_count,
        "peak_excess_counts": None,
        "peak_snr_sigma": None,
        "peak_bin": None,
        "width_halfmax_bins": None,
        "active_bin_count": 0,
        "signal_extent_bins": None,
        "signal_support_component_count": 0,
        "dominant_support_start_bin": None,
        "dominant_support_end_bin": None,
        "dominant_support_width_bins": None,
        "dominant_support_energy_counts_bin": None,
        "energy_centroid_bin": None,
        "energy_p10_bin": None,
        "energy_p50_bin": None,
        "energy_p90_bin": None,
        "asymmetry_skewness": None,
        "signal_support_components": [],
        "peak_count": 0,
    }
    if finite_fraction < float(minimum_finite_fraction):
        return {"waveform_status": "invalid_waveform", **invalid_result}
    if not np.isfinite(background_mean) or not np.isfinite(background_std) or background_std <= 0:
        return {"waveform_status": "invalid_background", **invalid_result}

    excess = values - float(background_mean)
    sigma = float(background_std)
    threshold = float(threshold_sigma) * sigma
    valid_excess = excess[finite]
    peak_bin = int(np.flatnonzero(finite)[int(np.argmax(valid_excess))])
    peak_excess = max(float(excess[peak_bin]), 0.0)
    z = excess / sigma
    finite_excess = np.where(finite, excess, -np.inf)
    peak_indices, _ = find_peaks(
        finite_excess,
        height=threshold,
        distance=max(1, int(minimum_peak_distance)),
    )
    edge_candidates = []
    if values.size == 1 and finite[0] and excess[0] >= threshold:
        edge_candidates.append(0)
    elif values.size > 1:
        if finite[0] and excess[0] >= threshold and excess[0] > excess[1]:
            edge_candidates.append(0)
        if finite[-1] and excess[-1] >= threshold and excess[-1] > excess[-2]:
            edge_candidates.append(values.size - 1)
    if edge_candidates:
        peak_indices = np.unique(np.concatenate((peak_indices, np.asarray(edge_candidates, dtype=np.int64))))
    if peak_excess >= threshold and peak_bin not in peak_indices:
        peak_indices = np.append(peak_indices, peak_bin)
    active = finite & (excess >= threshold)
    active_bin_count = int(np.count_nonzero(active))
    active_positions = np.flatnonzero(active)
    components = []
    if active_positions.size:
        breaks = np.flatnonzero(np.diff(active_positions) > 1) + 1
        runs = np.split(active_positions, breaks)
        for run in runs:
            start, end = int(run[0]), int(run[-1])
            weights = np.maximum(excess[start : end + 1], 0.0)
            total = float(np.sum(weights))
            components.append({
                "start_bin": start,
                "end_bin": end,
                "width_bins": int(end - start + 1),
                "positive_energy_counts_bin": total,
            })
    dominant_component = next(
        (part for part in components if part["start_bin"] <= peak_bin <= part["end_bin"]),
        None,
    )
    centroid = p10 = p50 = p90 = skewness = None
    if peak_excess >= threshold and dominant_component is not None:
        start = int(dominant_component["start_bin"])
        end = int(dominant_component["end_bin"])
        support_bins = np.arange(start, end + 1, dtype=np.float64)
        weights = np.maximum(excess[start : end + 1], 0.0)
        energy = float(np.sum(weights))
        if energy > 0:
            centroid = float(np.sum(support_bins * weights) / energy)
            cdf = np.cumsum(weights) / energy
            p10, p50, p90 = [float(support_bins[min(int(np.searchsorted(cdf, q, side="left")), support_bins.size - 1)]) for q in (0.1, 0.5, 0.9)]
            variance = float(np.sum(weights * (support_bins - centroid) ** 2) / energy)
            if variance > 0:
                skewness = float(np.sum(weights * (support_bins - centroid) ** 3) / energy / variance ** 1.5)
    if peak_excess >= threshold:
        halfmax = max(threshold, peak_excess / 2.0)
        above_half = finite & (excess >= halfmax)
        left = peak_bin
        right = peak_bin
        while left > 0 and above_half[left - 1]:
            left -= 1
        while right + 1 < values.size and above_half[right + 1]:
            right += 1
        width = int(right - left + 1)
    else:
        width = None

    return {
        "waveform_status": "signal_present_3sigma" if peak_excess >= threshold else "no_3sigma_peak",
        "finite_fraction": finite_fraction,
        "raw_max_counts": float(np.max(values[finite])),
        "is_clipped": clipped,
        "clipped_bin_count": clipped_bin_count,
        "peak_excess_counts": peak_excess,
        "peak_snr_sigma": float(np.max(z[finite])),
        "peak_bin": peak_bin,
        "width_halfmax_bins": width,
        "active_bin_count": active_bin_count,
        "signal_extent_bins": int(active_positions[-1] - active_positions[0] + 1) if active_positions.size else None,
        "signal_support_component_count": int(len(components)),
        "dominant_support_start_bin": int(dominant_component["start_bin"]) if dominant_component else None,
        "dominant_support_end_bin": int(dominant_component["end_bin"]) if dominant_component else None,
        "dominant_support_width_bins": int(dominant_component["width_bins"]) if dominant_component else None,
        "dominant_support_energy_counts_bin": float(dominant_component["positive_energy_counts_bin"]) if dominant_component else None,
        "energy_centroid_bin": centroid,
        "energy_p10_bin": p10,
        "energy_p50_bin": p50,
        "energy_p90_bin": p90,
        "asymmetry_skewness": skewness,
        "signal_support_components": components,
        "peak_count": int(len(peak_indices)),
    }


def roi_primary_eligibility(
    properties: Mapping[str, Any],
    *,
    expected_granule_id: str,
    observed_image_date: str,
) -> dict[str, Any]:
    """Require an included ROI, exact granule identity, and verified image date."""

    reasons = []
    if properties.get("analysis_status") != "Included":
        reasons.append("roi_not_marked_included")
    if properties.get("granule_id") != expected_granule_id:
        reasons.append("granule_identity_mismatch")
    if str(properties.get("source_date", "")) != str(observed_image_date):
        reasons.append("source_date_mismatch")
    return {
        "primary_eligible": not reasons,
        "roi_id": properties.get("roi_id"),
        "site_id": properties.get("site_id"),
        "granule_id": properties.get("granule_id"),
        "source_date": properties.get("source_date"),
        "observed_image_date": observed_image_date,
        "reasons": reasons,
    }


def common_track_ids(left_tracks: Any, right_tracks: Any) -> list[int]:
    """Return sorted shared track IDs for an in-granule ROI pair."""

    left = {int(value) for value in left_tracks if np.isfinite(value)}
    right = {int(value) for value in right_tracks if np.isfinite(value)}
    return sorted(left & right)


def aggregate_roi_track_responses(
    rows: Any,
    *,
    metric_columns: tuple[str, ...] = (
        "refh_snr",
        "refh_amp_counts",
        "width_halfmax_bins",
        "dominant_support_energy_counts_bin",
        "asymmetry_skewness",
    ),
) -> list[dict[str, Any]]:
    """Robustly aggregate every pulse in each ROI-track, retaining no-detects."""

    required = {"granule_id", "roi_id", "site_id", "class_label", "track_num", "candidate_signal_detected"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"ROI-track response table is missing columns: {sorted(missing)}")
    keys = ["granule_id", "roi_id", "site_id", "class_label", "track_num"]
    output = []
    for group_key, group in rows.groupby(keys, sort=True, dropna=False):
        record = dict(zip(keys, group_key))
        detected = group["candidate_signal_detected"].fillna(False).astype(bool)
        record.update({
            "n_pulses": int(len(group)),
            "n_candidate_signal": int(detected.sum()),
            "candidate_signal_rate": float(detected.mean()) if len(group) else None,
        })
        if "waveform_status" in group:
            record["n_valid_waveforms"] = int((group.waveform_status != "invalid_waveform").sum())
        else:
            record["n_valid_waveforms"] = int(len(group))
        for column in ("sweep_num", "delta_time_s"):
            if column in group:
                values = np.asarray(group[column], dtype=np.float64)
                values = values[np.isfinite(values)]
                record[f"{column}_median"] = float(np.median(values)) if values.size else None
        for column in metric_columns:
            if column not in group:
                continue
            values = np.asarray(group[column], dtype=np.float64)
            values = values[np.isfinite(values)]
            record[f"{column}_n"] = int(values.size)
            record[f"{column}_median"] = float(np.median(values)) if values.size else None
            record[f"{column}_p10"] = float(np.percentile(values, 10)) if values.size else None
            record[f"{column}_p90"] = float(np.percentile(values, 90)) if values.size else None
        output.append(record)
    return output


def matched_track_differences(
    track_rows: Any,
    *,
    roi_a: str,
    roi_b: str,
    metric: str,
) -> list[dict[str, Any]]:
    """Compare ROI-track medians only on common track IDs; never sample pulses."""

    by_roi: dict[str, dict[int, Any]] = {roi_a: {}, roi_b: {}}
    for row in track_rows:
        roi = row.get("roi_id")
        if roi in by_roi:
            by_roi[roi][int(row["track_num"])] = row
    result = []
    for track in sorted(set(by_roi[roi_a]) & set(by_roi[roi_b])):
        left = by_roi[roi_a][track]
        right = by_roi[roi_b][track]
        left_value = left.get(f"{metric}_median")
        right_value = right.get(f"{metric}_median")
        if left_value is None or right_value is None:
            continue
        result.append({
            "track_num": track,
            "roi_a": roi_a,
            "roi_b": roi_b,
            "metric": metric,
            "roi_a_median": float(left_value),
            "roi_b_median": float(right_value),
            "difference_a_minus_b": float(left_value) - float(right_value),
            "roi_a_pulse_n": int(left["n_pulses"]),
            "roi_b_pulse_n": int(right["n_pulses"]),
        })
    return result


def sample_real_background_windows(
    h5_path: str,
    *,
    window: tuple[int, int] = (2600, 2728),
    block_count: int = 24,
    rows_per_block: int = 256,
) -> np.ndarray:
    """Read deterministic late-window noise samples from a real CASALS H5."""

    import h5py

    windows = []
    with h5py.File(h5_path, "r") as h5:
        rx = h5["rx_waveform"]
        n, bin_count = rx.shape
        lo, hi = map(int, window)
        if lo < 0 or hi > bin_count or hi - lo < 16 or block_count < 1 or rows_per_block < 1:
            raise ValueError(f"Invalid background sampling window or block count: {window}.")
        starts = np.linspace(0, max(0, n - rows_per_block), block_count, dtype=int)
        for start in starts:
            stop = min(start + rows_per_block, n)
            wave = np.asarray(rx[start:stop, lo:hi], dtype=np.float32)
            mean = np.asarray(h5["bg_mean"][start:stop], dtype=np.float32)
            std = np.asarray(h5["bg_std"][start:stop], dtype=np.float32)
            valid = np.isfinite(mean) & np.isfinite(std) & (std > 0) & np.all(np.isfinite(wave), axis=1)
            if valid.any():
                windows.append((wave[valid] - mean[valid, None]) / std[valid, None])
    if not windows:
        raise ValueError(f"No valid background windows were found in {h5_path}.")
    return np.concatenate(windows, axis=0).astype(np.float32, copy=False)


def background_stationarity_summary(windows: np.ndarray) -> tuple[Any, float | None]:
    """Summarize four tail segments before using a block-bootstrap null."""

    import pandas as pd

    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 16 or values.shape[1] % 4:
        raise ValueError("Background windows must have a bin count divisible by four.")
    rows = []
    width = values.shape[1] // 4
    for segment in range(4):
        start, stop = segment * width, (segment + 1) * width
        part = values[:, start:stop]
        centered = part - part.mean(axis=1, keepdims=True)
        denom = np.sqrt(np.sum(centered[:, :-1] ** 2, axis=1) * np.sum(centered[:, 1:] ** 2, axis=1))
        lag1 = np.divide(
            np.sum(centered[:, :-1] * centered[:, 1:], axis=1),
            denom,
            out=np.zeros(len(part), dtype=np.float64),
            where=denom > 0,
        )
        rows.append({
            "tail_segment": f"{start}:{stop}",
            "bin_count": stop - start,
            "window_count": len(part),
            "median_standardized_mean": float(np.median(part.mean(axis=1))),
            "median_standardized_sd": float(np.median(part.std(axis=1))),
            "p95_absolute_standardized_sample": float(np.percentile(np.abs(part), 95)),
            "median_lag1_correlation": float(np.median(lag1)),
        })
    table = pd.DataFrame(rows)
    sd = table["median_standardized_sd"].to_numpy(dtype=np.float64)
    ratio = float(sd.max() / sd.min()) if sd.size and sd.min() > 0 else None
    return table, ratio


def extract_roi_waveform_features(
    h5_path: str,
    membership: Any,
    *,
    prominence_threshold_sigma: float,
    minimum_width_bins: float = 2.0,
    minimum_peak_distance: int = 5,
) -> tuple[Any, Any]:
    """Re-read selected real H5 waveforms and recompute all candidate fields."""

    import h5py
    import pandas as pd

    required = {"granule_id", "pulse_index", "sweep_num", "track_num", "roi_id"}
    missing = required - set(membership.columns)
    if missing:
        raise ValueError(f"ROI membership is missing columns: {sorted(missing)}")
    stem = Path(h5_path).stem
    if set(membership["granule_id"].astype(str)) != {stem}:
        raise ValueError("ROI membership granule IDs do not match the source H5.")
    pulse_ids = np.unique(membership["pulse_index"].to_numpy(dtype=np.int64))
    if not pulse_ids.size or pulse_ids[0] < 0:
        raise ValueError(f"No valid ROI pulse IDs were supplied for {stem}.")

    scalar_names = (
        "refh", "refh_amp", "refh_snr", "good_snr", "bg_mean", "bg_std",
        "sweep_num", "track_num", "delta_time", "refh_longitude", "refh_latitude",
    )
    rows = []
    with h5py.File(h5_path, "r") as h5:
        rx = h5["rx_waveform"]
        if pulse_ids[-1] >= rx.shape[0]:
            raise ValueError("An ROI pulse ID exceeds the source H5 record count.")
        row_chunk = int(rx.chunks[0]) if rx.chunks else 512
        for start in range(0, rx.shape[0], row_chunk):
            stop = min(start + row_chunk, rx.shape[0])
            left = int(np.searchsorted(pulse_ids, start, side="left"))
            right = int(np.searchsorted(pulse_ids, stop, side="left"))
            if left == right:
                continue
            ids = pulse_ids[left:right]
            waves = np.asarray(rx[start:stop, :])
            scalar = {name: np.asarray(h5[name][start:stop]) for name in scalar_names}
            for pulse_id in ids:
                row = int(pulse_id - start)
                waveform = np.asarray(waves[row], dtype=np.float64)
                bg_mean, bg_std = float(scalar["bg_mean"][row]), float(scalar["bg_std"][row])
                detected = candidate_signal_features(
                    waveform,
                    bg_mean,
                    bg_std,
                    prominence_threshold_sigma=prominence_threshold_sigma,
                    minimum_width_bins=minimum_width_bins,
                    minimum_peak_distance=minimum_peak_distance,
                )
                peak = detected["candidate_peak_bin"]
                feature = {
                    "candidate_signal_status": detected["candidate_signal_status"],
                    "waveform_status": (
                        "invalid_waveform"
                        if detected["candidate_signal_status"].startswith("invalid_")
                        else ("candidate" if detected["candidate_signal_detected"] else "no_candidate")
                    ),
                    "waveform_quality_reason": detected["candidate_signal_status"],
                    "candidate_signal_detected": bool(detected["candidate_signal_detected"]),
                    "candidate_peak_bin": peak,
                    "candidate_peak_height_sigma": detected["candidate_peak_height_sigma"],
                    "candidate_peak_prominence_sigma": detected["candidate_peak_prominence_sigma"],
                    "maximum_interior_peak_prominence_sigma": detected["maximum_local_peak_prominence_sigma"],
                    "candidate_peak_width_bins": detected["candidate_peak_width_bins"],
                    "width_halfmax_bins": detected["candidate_peak_width_bins"],
                    "candidate_local_peak_count": int(detected["candidate_local_peak_count"]),
                    "finite_waveform_fraction": float(detected["finite_fraction"]),
                    "raw_max_bin": int(np.nanargmax(waveform)) if np.isfinite(waveform).any() else None,
                    "raw_max_counts": float(np.nanmax(waveform)) if np.isfinite(waveform).any() else None,
                    "dominant_support_energy_counts_bin": None,
                    "signal_extent_bins": None,
                    "asymmetry_skewness": None,
                    "candidate_peak_excess_counts": None,
                }
                if detected["candidate_signal_detected"]:
                    peak = int(peak)
                    z = (waveform - bg_mean) / bg_std
                    above = np.isfinite(z) & (z >= 1.0)
                    lo = peak
                    hi = peak + 1
                    while lo > 0 and above[lo - 1]:
                        lo -= 1
                    while hi < len(z) and above[hi]:
                        hi += 1
                    bins = np.arange(lo, hi, dtype=np.float64)
                    excess = np.maximum(waveform[lo:hi] - bg_mean, 0.0)
                    energy = float(excess.sum())
                    feature.update({
                        "dominant_support_energy_counts_bin": energy,
                        "signal_extent_bins": int(hi - lo),
                        "candidate_peak_excess_counts": float(waveform[peak] - bg_mean),
                    })
                    if energy > 0 and len(bins) >= 3:
                        center = float(np.dot(bins, excess) / energy)
                        variance = float(np.dot((bins - center) ** 2, excess) / energy)
                        feature["asymmetry_skewness"] = (
                            float(np.dot((bins - center) ** 3, excess) / energy / variance ** 1.5)
                            if variance > 0 else None
                        )
                row_data = {
                    "granule_id": stem,
                    "pulse_index": int(pulse_id),
                    "sweep_num": int(scalar["sweep_num"][row]),
                    "track_num": int(scalar["track_num"][row]),
                    "refh_m": float(scalar["refh"][row]) if np.isfinite(scalar["refh"][row]) else np.nan,
                    "refh_amp_counts": int(scalar["refh_amp"][row]),
                    "refh_snr": float(scalar["refh_snr"][row]) if np.isfinite(scalar["refh_snr"][row]) else np.nan,
                    "good_snr": bool(scalar["good_snr"][row]),
                    "bg_mean_counts": bg_mean,
                    "bg_std_counts": bg_std,
                    "delta_time_s": float(scalar["delta_time"][row]),
                    "longitude_deg": float(scalar["refh_longitude"][row]),
                    "latitude_deg": float(scalar["refh_latitude"][row]),
                    "noise_calibrated_prominence_threshold_sigma": float(prominence_threshold_sigma),
                    **feature,
                }
                row_data["refh_amp_matches_raw_max"] = bool(
                    row_data["raw_max_counts"] is not None
                    and np.isclose(row_data["refh_amp_counts"], row_data["raw_max_counts"], rtol=0.0, atol=0.0)
                )
                rows.append(row_data)

    pulse_features = pd.DataFrame(rows)
    if len(pulse_features) != len(pulse_ids):
        raise RuntimeError(f"Waveform extraction lost selected pulse records for {stem}.")
    actual = pulse_features.set_index("pulse_index")
    membership_tracks = membership.drop_duplicates("pulse_index").set_index("pulse_index")
    if not np.array_equal(actual.loc[membership_tracks.index, "sweep_num"], membership_tracks["sweep_num"]):
        raise RuntimeError("ROI sweep IDs do not match the current source H5.")
    if not np.array_equal(actual.loc[membership_tracks.index, "track_num"], membership_tracks["track_num"]):
        raise RuntimeError("ROI track IDs do not match the current source H5.")
    per_roi = membership.merge(
        pulse_features,
        on=["granule_id", "pulse_index", "sweep_num", "track_num"],
        how="left",
        validate="many_to_one",
    )
    if per_roi["candidate_signal_detected"].isna().any():
        raise RuntimeError("Some ROI memberships did not receive a recomputed H5 waveform result.")
    return pulse_features, per_roi


def _max_peak_prominence(values: np.ndarray, minimum_peak_distance: int) -> tuple[float, float, int]:
    peaks, _ = find_peaks(values, distance=max(1, int(minimum_peak_distance)))
    if not peaks.size:
        return 0.0, 0.0, -1
    prominences = peak_prominences(values, peaks)[0]
    widths = peak_widths(values, peaks, rel_height=0.5)[0]
    best = int(np.argmax(prominences))
    return float(prominences[best]), float(widths[best]), int(peaks[best])


def calibrate_noise_peak_threshold(
    real_background_windows: np.ndarray,
    *,
    waveform_length: int = 2728,
    quantile: float = 0.995,
    minimum_peak_distance: int = 5,
    calibration_simulations: int = 10_000,
    validation_simulations: int = 10_000,
    seed: int = 20261012,
) -> dict[str, Any]:
    """Block-bootstrap full-waveform null maxima from standardized real noise windows."""

    windows = np.asarray(real_background_windows, dtype=np.float32)
    if windows.ndim != 2 or windows.shape[0] < 20 or windows.shape[1] < 16:
        raise ValueError("Provide at least 20 real background windows of 16 or more bins.")
    if not 0.90 <= quantile < 1.0 or waveform_length < windows.shape[1]:
        raise ValueError("Invalid null quantile or waveform length.")
    keep = np.mean(np.isfinite(windows), axis=1) >= 0.95
    windows = windows[keep]
    windows = windows[np.all(np.isfinite(windows), axis=1)]
    if windows.shape[0] < 20:
        raise ValueError("Fewer than 20 finite real background windows remain.")

    block_length = windows.shape[1]
    n_blocks = int(np.ceil(waveform_length / block_length))
    rng = np.random.default_rng(seed)

    def simulate(replicates: int) -> tuple[np.ndarray, np.ndarray]:
        maxima = np.zeros(replicates, dtype=np.float32)
        widths = np.zeros(replicates, dtype=np.float32)
        batch_size = 256
        for start in range(0, replicates, batch_size):
            stop = min(start + batch_size, replicates)
            count = stop - start
            block_ids = rng.integers(0, len(windows), size=(count, n_blocks))
            synthetic = np.empty((count, n_blocks * block_length), dtype=np.float32)
            for block in range(n_blocks):
                synthetic[:, block * block_length : (block + 1) * block_length] = windows[block_ids[:, block]]
            synthetic = synthetic[:, :waveform_length]
            for i, values in enumerate(synthetic):
                maxima[start + i], widths[start + i], _ = _max_peak_prominence(values, minimum_peak_distance)
        return maxima, widths

    calibration_maxima, calibration_widths = simulate(int(calibration_simulations))
    threshold = float(np.quantile(calibration_maxima, quantile))
    validation_maxima, validation_widths = simulate(int(validation_simulations))
    false_alarms = (validation_maxima >= threshold) & (validation_widths >= 2.0)
    return {
        "threshold_peak_prominence_sigma": threshold,
        "threshold_quantile": float(quantile),
        "real_background_window_count": int(len(windows)),
        "background_window_length_bins": int(block_length),
        "synthetic_null_waveform_length_bins": int(waveform_length),
        "calibration_simulation_count": int(calibration_simulations),
        "validation_simulation_count": int(validation_simulations),
        "validation_false_alarm_count": int(false_alarms.sum()),
        "validation_false_alarm_rate": float(false_alarms.mean()),
        "validation_null_peak_prominence_p50_sigma": float(np.median(validation_maxima)),
        "validation_null_peak_prominence_p95_sigma": float(np.percentile(validation_maxima, 95)),
        "minimum_detected_width_bins": 2.0,
        "multiple_testing_method": "block bootstrap of real standardized noise windows to the full RX length",
        "seed": int(seed),
    }


def candidate_signal_features(
    waveform: np.ndarray,
    background_mean: float,
    background_std: float,
    *,
    prominence_threshold_sigma: float,
    minimum_width_bins: float = 2.0,
    minimum_peak_distance: int = 5,
    minimum_finite_fraction: float = 0.95,
) -> dict[str, Any]:
    """Flag a waveform candidate using noise-calibrated peak prominence."""

    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    finite_fraction = float(finite.mean()) if values.size else 0.0
    empty = {
        "candidate_signal_detected": False,
        "candidate_peak_bin": None,
        "candidate_peak_height_sigma": None,
        "candidate_peak_prominence_sigma": None,
        "candidate_peak_width_bins": None,
        "candidate_local_peak_count": 0,
        "maximum_local_peak_prominence_sigma": None,
    }
    if not values.size or finite_fraction < minimum_finite_fraction:
        return {"candidate_signal_status": "invalid_waveform", "finite_fraction": finite_fraction, **empty}
    if not np.isfinite(background_mean) or not np.isfinite(background_std) or background_std <= 0:
        return {"candidate_signal_status": "invalid_background", "finite_fraction": finite_fraction, **empty}
    if prominence_threshold_sigma <= 0 or minimum_width_bins < 0:
        raise ValueError("The calibrated prominence threshold must be positive and width non-negative.")

    z = (values - float(background_mean)) / float(background_std)
    z_for_peaks = np.where(finite, z, -np.inf)
    peaks, _ = find_peaks(z_for_peaks, distance=max(1, int(minimum_peak_distance)))
    prominences = peak_prominences(z_for_peaks, peaks)[0] if peaks.size else np.array([], dtype=float)
    widths = peak_widths(z_for_peaks, peaks, rel_height=0.5)[0] if peaks.size else np.array([], dtype=float)
    if not peaks.size:
        return {"candidate_signal_status": "no_candidate", "finite_fraction": finite_fraction, **empty}
    heights = z_for_peaks[peaks]
    qualified = np.flatnonzero(
        (heights > 0.0)
        & (prominences >= prominence_threshold_sigma)
        & (widths >= minimum_width_bins)
    )
    maximum_prominence = float(np.max(prominences))
    if not qualified.size:
        return {
            "candidate_signal_status": "no_candidate",
            "finite_fraction": finite_fraction,
            **empty,
            "maximum_local_peak_prominence_sigma": maximum_prominence,
        }

    # Prefer the strongest qualified peak, then the wider peak, then the first bin.
    best = max(
        qualified.tolist(),
        key=lambda i: (float(prominences[i]), float(widths[i]), -int(peaks[i])),
    )
    return {
        "candidate_signal_status": "candidate",
        "finite_fraction": finite_fraction,
        "candidate_signal_detected": True,
        "candidate_peak_bin": int(peaks[best]),
        "candidate_peak_height_sigma": float(heights[best]),
        "candidate_peak_prominence_sigma": float(prominences[best]),
        "candidate_peak_width_bins": float(widths[best]),
        "candidate_local_peak_count": int(qualified.size),
        "maximum_local_peak_prominence_sigma": maximum_prominence,
    }


def site_level_unit_counts(rows: Any) -> dict[str, Any]:
    """Count independent site IDs by land-cover class, not by pulse or ROI."""

    if not {"site_id", "class_label"}.issubset(rows.columns):
        raise ValueError("Site summary needs site_id and class_label columns.")
    sites_by_class: dict[str, set[str]] = {}
    for row in rows[["site_id", "class_label"]].drop_duplicates().to_dict("records"):
        sites_by_class.setdefault(str(row["class_label"]), set()).add(str(row["site_id"]))
    return {
        class_label: {"n_independent_sites": len(sites), "site_ids": sorted(sites)}
        for class_label, sites in sorted(sites_by_class.items())
    }
