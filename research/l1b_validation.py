"""Small, diagnostic helpers for CASALS L1B research validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS
from pyproj.transformer import AreaOfInterest, TransformDirection, TransformerGroup
from scipy.signal import find_peaks, peak_prominences, peak_widths


def sha256_file(path: str | Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a source or result file."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive.")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def build_landcover_response_analysis(
    project_root: str | Path,
    h5_by_stem: Mapping[str, str | Path],
    naip_paths: Mapping[str, tuple[str | Path, str | Path]],
    naip_dates: Mapping[str, str],
    roi_manifest: Mapping[str, Any],
    previous_root: str | Path,
    output_root: str | Path,
    *,
    prominence_quantile: float,
    background_window: tuple[int, int],
    minimum_width_bins: float,
    minimum_peak_distance: int,
) -> dict[str, Any]:
    """Recompute current-H5 ROI responses and write Notebook 11 research tables."""

    import gc
    import h5py
    import inspect
    import pandas as pd
    import rasterio
    from datetime import date
    from matplotlib.path import Path as PolygonPath
    from pyproj import Transformer

    from research.validation_figures import plot_naip_roi_board

    ROOT = Path(project_root)
    H5_BY_STEM = {str(key): Path(value) for key, value in h5_by_stem.items()}
    NAIP_PATHS = {str(key): tuple(Path(value) for value in paths) for key, paths in naip_paths.items()}
    NAIP_DATES = naip_dates
    ROI_MANIFEST = roi_manifest
    OLD_ROOT = Path(previous_root)
    OUT_ROOT = Path(output_root)
    PROMINENCE_QUANTILE = float(prominence_quantile)
    BACKGROUND_WINDOW = tuple(background_window)
    MINIMUM_WIDTH_BINS = float(minimum_width_bins)
    MINIMUM_PEAK_DISTANCE = int(minimum_peak_distance)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    roi_features_by_stem = {}
    roi_audit_parts, memberships, features_by_stem = [], {}, {}
    noise_parts, noise_summary = [], {}
    for stem, h5_path in H5_BY_STEM.items():
        image_path, metadata_path = NAIP_PATHS[stem]
        image_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_date = NAIP_DATES[stem]
        if image_metadata.get("acquisition_dates") != [expected_date]:
            raise ValueError(f"NAIP date mismatch for {stem}: {image_metadata.get('acquisition_dates')}.")
        granule_features = [feature for feature in ROI_MANIFEST["features"]
                            if feature["properties"].get("granule_id") == stem]
        roi_features_by_stem[stem] = granule_features
        with h5py.File(h5_path, "r") as h5:
            h5_date = str(h5.attrs["start_utca"])[:10]
            h5_count = int(h5["refh"].shape[0])
            if not np.array_equal(np.asarray(h5["good_snr"]), np.asarray(h5["refh_snr"]) >= 5.0):
                raise ValueError(f"H5 good_snr differs from refh_snr >= 5 for {stem}.")

        audit_rows = []
        with rasterio.open(image_path) as raster:
            transform = raster.transform
            if raster.crs is None or not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0) or transform.a <= 0 or transform.e >= 0:
                raise ValueError(f"NAIP raster is not north-up with projected coordinates: {image_path}")
            raster_transformer = Transformer.from_crs("EPSG:4326", raster.crs, always_xy=True)
            for feature in granule_features:
                props = feature["properties"]
                coords = np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float64)
                px, py = raster_transformer.transform(coords[:, 0], coords[:, 1])
                inside = ((px >= raster.bounds.left) & (px <= raster.bounds.right)
                          & (py >= raster.bounds.bottom) & (py <= raster.bounds.top))
                eligibility = roi_primary_eligibility(
                    props, expected_granule_id=stem, observed_image_date=expected_date,
                )
                full_footprint = bool(inside.all())
                primary = bool(eligibility["primary_eligible"] and full_footprint)
                tier = "primary" if primary else ("exploratory" if props.get("analysis_status") in {"Included", "Candidate"} else "excluded")
                audit_rows.append({
                    "granule_id": stem, "roi_id": props["roi_id"], "site_id": props["site_id"],
                    "class_label": props["class_label"], "analysis_status": props["analysis_status"],
                    "analysis_tier": tier, "primary_eligible": primary,
                    "eligibility_reasons": "|".join(eligibility["reasons"] + ([] if full_footprint else ["roi_outside_naip_bounds"])),
                    "h5_acquisition_date": h5_date, "naip_acquisition_date": expected_date,
                    "naip_age_days_at_h5": (date.fromisoformat(h5_date) - date.fromisoformat(expected_date)).days,
                    "naip_crs": raster.crs.to_string(), "naip_affine": tuple(transform),
                    "naip_bounds": tuple(raster.bounds), "roi_vertices_inside_naip_fraction": float(inside.mean()),
                    "roi_fully_inside_naip_footprint": full_footprint,
                    "label_source": props.get("label_source"), "evidence_source": props.get("evidence_source"),
                    "confidence": props.get("confidence"), "temporal_change_risk": props.get("temporal_change_risk"),
                    "boundary_risk": props.get("boundary_risk"),
                })
        audit = pd.DataFrame(audit_rows)
        roi_audit_parts.append(audit)

        membership_path = OLD_ROOT / stem / "roi_pulse_features.csv.gz"
        membership = pd.read_csv(
            membership_path, compression="gzip",
            usecols=["granule_id", "pulse_index", "sweep_num", "track_num", "roi_id"],
        ).drop_duplicates(["granule_id", "pulse_index", "roi_id"]).copy()
        if set(membership["granule_id"].astype(str)) != {stem}:
            raise ValueError(f"ROI membership granule IDs do not match {stem}.")
        if membership.empty or membership.pulse_index.min() < 0 or membership.pulse_index.max() >= h5_count:
            raise ValueError(f"Invalid H5 pulse IDs in preserved ROI membership for {stem}.")
        membership["site_id"] = membership.roi_id.map({p["roi_id"]: p["site_id"] for p in [f["properties"] for f in granule_features]})
        membership["class_label"] = membership.roi_id.map({p["roi_id"]: p["class_label"] for p in [f["properties"] for f in granule_features]})
        membership["analysis_status"] = membership.roi_id.map({p["roi_id"]: p["analysis_status"] for p in [f["properties"] for f in granule_features]})
        membership["analysis_tier"] = membership.roi_id.map(audit.set_index("roi_id").analysis_tier.to_dict())
        if membership[["site_id", "class_label", "analysis_status", "analysis_tier"]].isna().any().any():
            raise ValueError(f"Membership includes an ROI missing from the current manifest for {stem}.")

        windows = sample_real_background_windows(
            str(h5_path), window=BACKGROUND_WINDOW, block_count=24, rows_per_block=256,
        )
        tail_table, sd_ratio = background_stationarity_summary(windows)
        calibration_path = OLD_ROOT / stem / "noise_peak_calibration.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        expected_seed = 20261024 if "20241112" in stem else 20261030
        checks = {
            "granule_id": calibration.get("granule_id") == stem,
            "background_window_bins": calibration.get("background_window_bins") == "2600:2728",
            "real_background_window_count": int(calibration.get("real_background_window_count", -1)) == len(windows),
            "calibration_simulation_count": int(calibration.get("calibration_simulation_count", -1)) == 10_000,
            "validation_simulation_count": int(calibration.get("validation_simulation_count", -1)) == 10_000,
            "seed": int(calibration.get("seed", -1)) == expected_seed,
            "width_condition": float(calibration.get("minimum_detected_width_bins", -1)) == MINIMUM_WIDTH_BINS,
            "threshold_quantile": float(calibration.get("threshold_quantile", -1)) == PROMINENCE_QUANTILE,
            "stationarity_ratio_reproduced": np.isclose(
                float(calibration.get("tail_segment_sd_ratio_max_min", np.nan)), float(sd_ratio), rtol=0, atol=5e-8,
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"Preserved noise threshold did not reproduce for {stem}: {checks}")
        calibration = {
            **calibration, "calibration_reused_from": str(calibration_path.relative_to(ROOT)),
            "current_real_tail_windows_rechecked": True, "current_tail_sd_ratio": float(sd_ratio),
            "calibration_parameters_verified": checks,
            "candidate_detector": "all positive-height local peaks; prominence and width qualified jointly; maximum prominence among qualified peaks",
            "candidate_detector_source_sha256": hashlib.sha256(
                inspect.getsource(__import__("research.l1b_validation", fromlist=["candidate_signal_features"]).candidate_signal_features).encode()
            ).hexdigest(),
            "official_refh_rx_bin_available": False,
            "interpretation": "Research screening threshold only; the block-bootstrap null does not calibrate real-return detection probability.",
        }
        noise_summary[stem] = calibration
        tail_table.insert(0, "granule_id", stem)
        noise_parts.append(tail_table)
        (OUT_ROOT / f"{stem}_noise_peak_calibration.json").write_text(
            json.dumps(_json_safe(calibration), indent=2, allow_nan=False), encoding="utf-8",
        )

        threshold = float(calibration["threshold_peak_prominence_sigma"])
        pulse_features, per_roi = extract_roi_waveform_features(
            str(h5_path), membership, prominence_threshold_sigma=threshold,
            minimum_width_bins=MINIMUM_WIDTH_BINS, minimum_peak_distance=MINIMUM_PEAK_DISTANCE,
        )
        for feature in granule_features:
            props = feature["properties"]
            roi_id = str(props["roi_id"])
            rows = per_roi.loc[per_roi.roi_id == roi_id]
            if rows.empty:
                continue
            polygon = PolygonPath(np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float64))
            coords = rows[["longitude_deg", "latitude_deg"]].to_numpy(dtype=np.float64)
            if not polygon.contains_points(coords).all():
                raise ValueError(f"Some current H5 centers fall outside their saved ROI: {stem}/{roi_id}.")
        per_roi = per_roi.merge(
            audit[["granule_id", "roi_id", "primary_eligible"]],
            on=["granule_id", "roi_id"], how="left", validate="many_to_one",
        )
        if len(pulse_features) != membership.pulse_index.nunique():
            raise ValueError(f"Current H5 waveform extraction lost pulse IDs for {stem}.")
        pulse_features.to_csv(OUT_ROOT / f"{stem}_recomputed_pulse_features.csv.gz",
                              index=False, compression="gzip")
        per_roi.to_csv(OUT_ROOT / f"{stem}_recomputed_roi_waveforms.csv.gz",
                       index=False, compression="gzip")
        memberships[stem] = membership
        features_by_stem[stem] = per_roi

        roi_counts = membership.groupby("roi_id").size().to_dict()
        audit["n_nominal_pulse_centers"] = audit.roi_id.map(roi_counts).fillna(0).astype(int)
        audit["n_tracks"] = audit.roi_id.map(membership.groupby("roi_id").track_num.nunique()).fillna(0).astype(int)
        roi_audit_parts[-1] = audit
        del pulse_features, per_roi, membership, windows
        gc.collect()

    ROI_AUDIT = pd.concat(roi_audit_parts, ignore_index=True)
    ROI_PULSE_FEATURES = pd.concat(
        [frame for frame in features_by_stem.values()], ignore_index=True,
    )
    NOISE_SEGMENTS = pd.concat(noise_parts, ignore_index=True)
    ROI_AUDIT.to_csv(OUT_ROOT / "roi_audit.csv", index=False)
    NOISE_SEGMENTS.to_csv(OUT_ROOT / "noise_tail_stationarity.csv", index=False)
    (OUT_ROOT / "noise_peak_calibration.json").write_text(
        json.dumps(_json_safe(noise_summary), indent=2, allow_nan=False), encoding="utf-8",
    )

    roi_rows = []
    for keys, group in ROI_PULSE_FEATURES.groupby(
        ["granule_id", "roi_id", "site_id", "class_label", "analysis_status", "analysis_tier"], sort=True,
    ):
        stem, roi_id, site_id, class_label, status, tier = keys
        detected = group.candidate_signal_detected.fillna(False).astype(bool)
        valid_wave = group.waveform_status.isin(["candidate", "no_candidate"])
        row = {
            "granule_id": stem, "roi_id": roi_id, "site_id": site_id, "class_label": class_label,
            "analysis_status": status, "analysis_tier": tier,
            "n_roi_pulses": int(len(group)), "n_valid_waveforms": int(valid_wave.sum()),
            "n_invalid_waveforms": int((~valid_wave).sum()), "n_candidate_signals": int(detected.sum()),
            "candidate_signal_rate_all_roi_pulses": float(detected.mean()),
            "track_count": int(group.track_num.nunique()),
            "refh_snr_median": float(group.refh_snr.median()),
            "refh_snr_p10": float(group.refh_snr.quantile(0.10)),
            "refh_snr_p90": float(group.refh_snr.quantile(0.90)),
            "refh_amp_counts_median": float(group.refh_amp_counts.median()),
            "refh_amp_counts_p10": float(group.refh_amp_counts.quantile(0.10)),
            "refh_amp_counts_p90": float(group.refh_amp_counts.quantile(0.90)),
            "candidate_width_median_bins": float(group.loc[detected, "width_halfmax_bins"].median()) if detected.any() else np.nan,
            "candidate_width_p10_bins": float(group.loc[detected, "width_halfmax_bins"].quantile(0.10)) if detected.any() else np.nan,
            "candidate_width_p90_bins": float(group.loc[detected, "width_halfmax_bins"].quantile(0.90)) if detected.any() else np.nan,
            "candidate_energy_median_counts_bin": float(group.loc[detected, "dominant_support_energy_counts_bin"].median()) if detected.any() else np.nan,
            "candidate_asymmetry_median": float(group.loc[detected, "asymmetry_skewness"].median()) if detected.any() else np.nan,
            "candidate_feature_n": int(detected.sum()),
            "refh_amp_matches_raw_max_n": int(group.refh_amp_matches_raw_max.sum()),
        }
        roi_rows.append(row)
    ROI_RESPONSE = pd.DataFrame(roi_rows)
    ROI_RESPONSE.to_csv(OUT_ROOT / "roi_response_summary.csv", index=False)

    site_pulses = ROI_PULSE_FEATURES.drop_duplicates(["granule_id", "site_id", "pulse_index"]).copy()
    site_rows = []
    for keys, group in site_pulses.groupby(["granule_id", "site_id", "class_label", "analysis_tier"], sort=True):
        stem, site_id, class_label, tier = keys
        detected = group.candidate_signal_detected.fillna(False).astype(bool)
        site_rows.append({
            "granule_id": stem, "site_id": site_id, "class_label": class_label, "analysis_tier": tier,
            "n_rois": int(ROI_PULSE_FEATURES.loc[
                (ROI_PULSE_FEATURES.granule_id == stem) & (ROI_PULSE_FEATURES.site_id == site_id), "roi_id"
            ].nunique()),
            "n_unique_site_pulses": int(len(group)), "n_candidate_signals": int(detected.sum()),
            "candidate_signal_rate": float(detected.mean()),
            "refh_snr_median": float(group.refh_snr.median()),
            "refh_amp_counts_median": float(group.refh_amp_counts.median()),
            "candidate_width_median_bins": float(group.loc[detected, "width_halfmax_bins"].median()) if detected.any() else np.nan,
            "candidate_energy_median_counts_bin": float(group.loc[detected, "dominant_support_energy_counts_bin"].median()) if detected.any() else np.nan,
        })
    SITE_RESPONSE = pd.DataFrame(site_rows)
    SITE_RESPONSE.to_csv(OUT_ROOT / "site_response_summary.csv", index=False)

    variability_rows = []
    for (stem, tier, class_label), group in SITE_RESPONSE.groupby(
        ["granule_id", "analysis_tier", "class_label"], sort=True,
    ):
        for metric in ("refh_snr_median", "refh_amp_counts_median", "candidate_signal_rate",
                       "candidate_width_median_bins", "candidate_energy_median_counts_bin"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            variability_rows.append({
                "granule_id": stem, "analysis_tier": tier, "class_label": class_label, "metric": metric,
                "independent_site_n": int(group.site_id.nunique()),
                "median_across_sites": float(np.median(values)) if len(values) else np.nan,
                "site_p10": float(np.percentile(values, 10)) if len(values) else np.nan,
                "site_p90": float(np.percentile(values, 90)) if len(values) else np.nan,
                "site_range": float(np.ptp(values)) if len(values) > 1 else np.nan,
                "site_ids": "|".join(sorted(group.site_id.astype(str).unique())),
            })
    SITE_VARIABILITY = pd.DataFrame(variability_rows)
    SITE_VARIABILITY.to_csv(OUT_ROOT / "between_site_variability.csv", index=False)

    old_summary = pd.read_csv(OLD_ROOT / "roi_response_summary.csv")
    candidate_rate_change = ROI_RESPONSE.merge(
        old_summary[["granule_id", "roi_id", "n_candidate_signals", "candidate_signal_rate_all_roi_pulses"]],
        on=["granule_id", "roi_id"], how="left", suffixes=("_current", "_previous"), validate="one_to_one",
    )
    candidate_rate_change["rate_change_percentage_points"] = 100 * (
        candidate_rate_change.candidate_signal_rate_all_roi_pulses_current
        - candidate_rate_change.candidate_signal_rate_all_roi_pulses_previous
    )
    candidate_rate_change["candidate_count_change"] = (
        candidate_rate_change.n_candidate_signals_current
        - candidate_rate_change.n_candidate_signals_previous
    )
    candidate_rate_change["rate_change_percentage_points"] = candidate_rate_change["rate_change_percentage_points"].round(8)
    candidate_rate_change["previous_values_use"] = "summary-only comparison; all current candidate features were recomputed from source H5"
    candidate_rate_change.to_csv(OUT_ROOT / "candidate_rate_change_after_detector_fix.csv", index=False)

    transition_rows = []
    for stem, current in features_by_stem.items():
        old_path = OLD_ROOT / stem / "roi_pulse_features.csv.gz"
        old_flags = pd.read_csv(
            old_path, compression="gzip",
            usecols=["granule_id", "roi_id", "pulse_index", "candidate_signal_detected"],
        )
        current_flags = current[["granule_id", "roi_id", "pulse_index", "candidate_signal_detected",
                                 "site_id", "class_label", "analysis_tier"]]
        comparison = old_flags.merge(
            current_flags, on=["granule_id", "roi_id", "pulse_index"],
            how="outer", suffixes=("_previous", "_current"), indicator=True, validate="one_to_one",
        )
        if not comparison._merge.eq("both").all():
            raise ValueError(f"Previous/current candidate comparison has unmatched ROI pulse IDs for {stem}.")
        previous = comparison.candidate_signal_detected_previous.astype(bool)
        updated = comparison.candidate_signal_detected_current.astype(bool)
        comparison["gained"] = ~previous & updated
        comparison["lost"] = previous & ~updated
        for keys, group in comparison.groupby(
            ["granule_id", "roi_id", "site_id", "class_label", "analysis_tier"], sort=True,
        ):
            transition_rows.append({
                **dict(zip(["granule_id", "roi_id", "site_id", "class_label", "analysis_tier"], keys)),
                "compared_pulse_n": int(len(group)),
                "previous_candidate_n": int(group.candidate_signal_detected_previous.astype(bool).sum()),
                "current_candidate_n": int(group.candidate_signal_detected_current.astype(bool).sum()),
                "gained_candidate_n": int(group.gained.sum()), "lost_candidate_n": int(group.lost.sum()),
                "changed_flag_n": int((group.gained | group.lost).sum()),
                "comparison_use": "before-after detector audit only; current summaries use only recomputed H5 features",
            })
    CANDIDATE_TRANSITIONS = pd.DataFrame(transition_rows)
    CANDIDATE_TRANSITIONS.to_csv(OUT_ROOT / "candidate_detection_transition_summary.csv", index=False)

    track_rows = []
    for stem, frame in features_by_stem.items():
        primary = frame.loc[frame.analysis_tier == "primary"].drop_duplicates(
            ["granule_id", "site_id", "pulse_index"],
        ).copy()
        primary["roi_id"] = primary.site_id.astype(str)
        track_rows.extend(aggregate_roi_track_responses(primary))
    TRACK_SUMMARY = pd.DataFrame(track_rows)
    TRACK_SUMMARY.to_csv(OUT_ROOT / "primary_site_track_responses.csv", index=False)

    track_pair_rows, track_pair_summary = [], []
    for stem in H5_BY_STEM:
        part = TRACK_SUMMARY.loc[TRACK_SUMMARY.granule_id == stem]
        open_sites = part.loc[part.class_label == "open_field", "site_id"].drop_duplicates().tolist()
        compare_sites = part.loc[part.class_label.isin(["tree_canopy", "road_surface"]), ["site_id", "class_label"]].drop_duplicates()
        by_site = {site: group.set_index("track_num", drop=False).to_dict("index")
                   for site, group in part.groupby("site_id")}
        pulse_counts = {
            (row.granule_id, row.site_id): int(row.n_unique_site_pulses)
            for row in SITE_RESPONSE.loc[SITE_RESPONSE.analysis_tier == "primary"].itertuples()
        }
        for open_site in open_sites:
            for target in compare_sites.itertuples(index=False):
                compare_site, target_class = str(target.site_id), str(target.class_label)
                if compare_site == open_site:
                    continue
                common = common_track_ids(by_site.get(open_site, {}).keys(), by_site.get(compare_site, {}).keys())
                comparison = f"{open_site} vs {compare_site} ({target_class})"
                date_label = "Nov 12" if "20241112" in stem else "Nov 18"
                field_label = "F1" if open_site.endswith(("01", "field_site_01")) else "F2"
                compare_label = "road" if target_class == "road_surface" else ("woodlot" if "northwest" in compare_site else "forest")
                comparison_short = f"{date_label}\n{field_label} / {compare_label}\nn={len(common)}"
                left = by_site.get(open_site, {})
                right = by_site.get(compare_site, {})
                summary = {
                    "granule_id": stem, "site_a": open_site, "site_b": compare_site, "class_b": target_class,
                    "comparison": comparison, "comparison_short": comparison_short, "common_track_n": len(common),
                    "site_a_pulse_n": pulse_counts.get((stem, open_site), 0),
                    "site_b_pulse_n": pulse_counts.get((stem, compare_site), 0),
                    "comparability": "common-track descriptive pair" if common else "no common track IDs",
                }
                values_by_metric = {
                    "snr_difference_a_minus_b": [], "amplitude_difference_a_minus_b": [],
                    "candidate_rate_difference_a_minus_b": [], "sweep_separation": [], "time_separation_s": [],
                }
                for track in common:
                    a, b = left[track], right[track]
                    row = {
                        "granule_id": stem, "comparison": comparison, "comparison_short": comparison_short,
                        "site_a": open_site, "site_b": compare_site, "class_b": target_class, "track_num": int(track),
                        "site_a_pulse_n": int(a["n_pulses"]), "site_b_pulse_n": int(b["n_pulses"]),
                        "site_a_sweep_median": a.get("sweep_num_median"),
                        "site_b_sweep_median": b.get("sweep_num_median"),
                        "site_a_time_median_s": a.get("delta_time_s_median"),
                        "site_b_time_median_s": b.get("delta_time_s_median"),
                    }
                    row["sweep_separation"] = abs(float(row["site_a_sweep_median"]) - float(row["site_b_sweep_median"]))
                    row["time_separation_s"] = abs(float(row["site_a_time_median_s"]) - float(row["site_b_time_median_s"]))
                    row["snr_difference_a_minus_b"] = float(a["refh_snr_median"] - b["refh_snr_median"])
                    row["amplitude_difference_a_minus_b"] = float(a["refh_amp_counts_median"] - b["refh_amp_counts_median"])
                    row["candidate_rate_difference_a_minus_b"] = float(a["candidate_signal_rate"] - b["candidate_signal_rate"])
                    for key in values_by_metric:
                        values_by_metric[key].append(row[key])
                    track_pair_rows.append(row)
                for key, values in values_by_metric.items():
                    summary[f"{key}_median"] = float(np.median(values)) if values else np.nan
                if common:
                    summary["median_sweep_separation"] = summary.pop("sweep_separation_median")
                    summary["median_time_separation_s"] = summary.pop("time_separation_s_median")
                track_pair_summary.append(summary)

    TRACK_PAIRS = pd.DataFrame(track_pair_rows)
    TRACK_PAIR_SUMMARY = pd.DataFrame(track_pair_summary)
    TRACK_PAIRS.to_csv(OUT_ROOT / "matched_track_comparisons.csv", index=False)
    TRACK_PAIR_SUMMARY.to_csv(OUT_ROOT / "matched_track_site_pair_summary.csv", index=False)

    (OUT_ROOT / "study_summary.json").write_text(json.dumps(_json_safe({
        "interpretation": "Within-granule descriptive response only; amplitude is native counts, morphology is candidate-only, and candidate rates are research screening rates.",
        "strict_detection_probability_calibrated": False,
        "official_refh_rx_bin_available": False,
        "candidate_detector_parameters": {
            "prominence_threshold_quantile": PROMINENCE_QUANTILE,
            "width_minimum_bins": MINIMUM_WIDTH_BINS, "minimum_peak_distance_bins": MINIMUM_PEAK_DISTANCE,
            "positive_peak_height_required": True,
            "qualification": "height above background, prominence, and width are checked jointly for every local peak",
        },
        "roi_count": int(len(ROI_AUDIT)), "roi_response_count": int(len(ROI_RESPONSE)),
        "site_response": SITE_RESPONSE.to_dict("records"),
        "matched_track_site_pair_summary": TRACK_PAIR_SUMMARY.to_dict("records"),
        "noise_thresholds": noise_summary,
    }), indent=2, allow_nan=False), encoding="utf-8")

    naip_figure_paths = {}
    for stem in H5_BY_STEM:
        assets = NAIP_PATHS[stem]
        audit = ROI_AUDIT.loc[ROI_AUDIT.granule_id == stem]
        membership = features_by_stem[stem]
        overview = OUT_ROOT / f"11-A_naip_roi_context_{stem}.png"
        plot_naip_roi_board(
            granule_id=stem, h5_date=audit.h5_acquisition_date.iloc[0],
            expected_naip_date=NAIP_DATES[stem], image_path=assets[0], metadata_path=assets[1],
            roi_features=roi_features_by_stem[stem], roi_audit=audit,
            membership=membership, output_path=overview,
        )
        naip_figure_paths[stem] = overview

    return {
        "roi_features_by_stem": roi_features_by_stem,
        "roi_audit": ROI_AUDIT,
        "roi_pulse_features": ROI_PULSE_FEATURES,
        "noise_segments": NOISE_SEGMENTS,
        "roi_response": ROI_RESPONSE,
        "site_response": SITE_RESPONSE,
        "site_variability": SITE_VARIABILITY,
        "candidate_rate_change": candidate_rate_change,
        "candidate_transitions": CANDIDATE_TRANSITIONS,
        "track_summary": TRACK_SUMMARY,
        "track_pairs": TRACK_PAIRS,
        "track_pair_summary": TRACK_PAIR_SUMMARY,
        "features_by_stem": features_by_stem,
        "naip_figure_paths": naip_figure_paths,
    }
