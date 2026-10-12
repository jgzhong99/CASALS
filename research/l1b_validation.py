"""Small, diagnostic helpers for CASALS L1B research validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from pyproj import CRS
from pyproj.transformer import AreaOfInterest, TransformDirection, TransformerGroup
from scipy.signal import find_peaks


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
