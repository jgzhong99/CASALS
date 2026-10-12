"""Bounded research helpers for empirical CASALS–3DEP comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.transformer import TransformerGroup
from scipy.spatial import cKDTree


def horizontal_transform_audit(
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    target_crs: Any = "EPSG:6347",
    area_of_interest: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Transform nominal H5 longitude/latitude in 2D and record the operation."""

    lon = np.asarray(longitude, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)
    if lon.shape != lat.shape or not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError("Longitude and latitude must be same-shaped finite arrays.")
    source = CRS.from_epsg(4326)
    target = CRS.from_user_input(target_crs)
    aoi = AreaOfInterest(*area_of_interest) if area_of_interest is not None else None
    group = TransformerGroup(
        source,
        target,
        always_xy=True,
        area_of_interest=aoi,
        allow_ballpark=True,
    )
    if not group.transformers:
        raise RuntimeError("No usable 2D horizontal transformation is available.")
    transformer = group.transformers[0]
    x, y = transformer.transform(lon, lat, errcheck=True)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise RuntimeError("Horizontal transformation returned non-finite coordinates.")
    accuracy = float(transformer.accuracy)
    return x, y, {
        "source_crs_assumption": "EPSG:4326 nominal H5 longitude/latitude; realization and epoch unresolved",
        "target_crs": target.to_string(),
        "selected_operation": transformer.description,
        "operation_definition": transformer.definition,
        "operation_accuracy_m": accuracy if accuracy >= 0 else None,
        "ballpark_operation": "ballpark" in transformer.description.lower(),
        "preferred_operation_available": bool(group.best_available),
        "unavailable_operation_count": len(group.unavailable_operations),
        "coordinate_status": "2D exploratory horizontal transformation; no vertical operation applied",
        "area_of_interest_lonlat": list(area_of_interest) if area_of_interest is not None else None,
    }


def class2_ground_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
) -> np.ndarray:
    """Return finite XYZ rows whose LAS classification is ASPRS class 2."""

    arrays = [np.asarray(v) for v in (x, y, z, classification)]
    if any(a.ndim != 1 for a in arrays) or len({a.size for a in arrays}) != 1:
        raise ValueError("x, y, z, and classification must be same-length 1D arrays.")
    xyz = np.column_stack([arrays[0], arrays[1], arrays[2]]).astype(np.float64, copy=False)
    keep = (arrays[3] == 2) & np.all(np.isfinite(xyz), axis=1)
    return xyz[keep]


def interpolate_ground_surface(
    query_xy: np.ndarray,
    ground_xy: np.ndarray,
    ground_z: np.ndarray,
    *,
    search_radius_m: float = 3.0,
    min_neighbors: int = 6,
    max_neighbors: int = 32,
    max_slope_deg: float = 5.0,
    max_plane_residual_nmad_m: float = 0.20,
    chunk_size: int = 20_000,
    ground_tree: cKDTree | None = None,
) -> dict[str, np.ndarray]:
    """Fit robust local planes without extrapolating beyond ground-point support.

    The independent local-median surface uses the same retained class-2
    neighbors. ``plane_fit_valid`` records geometrically supported fits;
    ``surface_acceptable`` also applies the slope and roughness limits.
    """

    q = np.asarray(query_xy, dtype=np.float64)
    xy = np.asarray(ground_xy, dtype=np.float64)
    z = np.asarray(ground_z, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 2 or not np.all(np.isfinite(q)):
        raise ValueError("query_xy must be a finite (n, 2) array.")
    if xy.ndim != 2 or xy.shape[1] != 2 or z.ndim != 1 or len(xy) != len(z):
        raise ValueError("ground_xy and ground_z must have shapes (m, 2) and (m,).")
    if not len(xy) or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(z)):
        raise ValueError("Ground points must be non-empty and finite.")
    if search_radius_m <= 0 or min_neighbors < 3 or max_neighbors < min_neighbors:
        raise ValueError("Invalid radius or neighborhood size.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    tree = ground_tree if ground_tree is not None else cKDTree(xy)
    n = len(q)
    out: dict[str, np.ndarray] = {
        "reference_height_plane_m": np.full(n, np.nan),
        "reference_height_median_m": np.full(n, np.nan),
        "neighbor_count": np.zeros(n, dtype=np.int16),
        "robust_neighbor_count": np.zeros(n, dtype=np.int16),
        "nearest_ground_distance_m": np.full(n, np.nan),
        "local_slope_deg": np.full(n, np.nan),
        "plane_residual_nmad_m": np.full(n, np.nan),
        "ground_roughness_nmad_m": np.full(n, np.nan),
        "plane_condition_number": np.full(n, np.nan),
    }
    out["plane_fit_valid"] = np.zeros(n, dtype=bool)
    out["surface_acceptable"] = np.zeros(n, dtype=bool)
    out["surface_status"] = np.full(n, "insufficient_neighbors", dtype="U32")

    k = min(max_neighbors, len(xy))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        qpart = q[start:stop]
        distances, indices = tree.query(
            qpart, k=k, distance_upper_bound=search_radius_m, workers=1
        )
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        valid = np.isfinite(distances) & (indices < len(xy))
        counts = valid.sum(axis=1)
        out["neighbor_count"][start:stop] = counts.astype(np.int16)
        nearest = np.where(valid.any(axis=1), np.min(np.where(valid, distances, np.inf), axis=1), np.nan)
        out["nearest_ground_distance_m"][start:stop] = nearest
        rows = np.flatnonzero(counts >= min_neighbors)
        if not rows.size:
            continue

        local_q = qpart[rows]
        local_valid = valid[rows]
        local_indices = np.minimum(indices[rows], len(xy) - 1)
        local_xy = xy[local_indices]
        local_z = z[local_indices]
        dx = local_xy[:, :, 0] - local_q[:, None, 0]
        dy = local_xy[:, :, 1] - local_q[:, None, 1]
        design = np.stack((dx, dy, np.ones_like(dx)), axis=2)
        z_masked = np.where(local_valid, local_z, np.nan)
        z_base = np.nanmedian(z_masked, axis=1)
        z_rel = np.where(local_valid, local_z - z_base[:, None], 0.0)

        def solve(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            weight = mask.astype(np.float64)
            gram = np.einsum("nki,nkj->nij", design, design * weight[:, :, None])
            rhs = np.einsum("nki,nk->ni", design, z_rel * weight)
            eigen = np.linalg.eigvalsh(gram)
            condition = np.divide(
                eigen[:, -1], eigen[:, 0], out=np.full(len(eigen), np.inf), where=eigen[:, 0] > 1e-12
            )
            enough = (mask.sum(axis=1) >= min_neighbors) & np.isfinite(condition) & (condition <= 1e8)
            beta = np.full((len(mask), 3), np.nan)
            if enough.any():
                beta[enough] = np.linalg.solve(gram[enough], rhs[enough, :, None])[:, :, 0]
            return beta, condition, enough

        beta, condition, fit_ok = solve(local_valid)
        predicted = np.einsum("nki,ni->nk", design, np.nan_to_num(beta))
        residual = z_rel - predicted
        residual = np.where(local_valid & fit_ok[:, None], residual, np.nan)
        residual_center = np.nanmedian(residual, axis=1)
        initial_nmad = 1.4826 * np.nanmedian(np.abs(residual - residual_center[:, None]), axis=1)
        cutoff = np.maximum(3.0 * np.nan_to_num(initial_nmad, nan=0.0), 0.02)
        inliers = local_valid & (np.abs(residual - residual_center[:, None]) <= cutoff[:, None])
        beta2, condition2, fit_ok2 = solve(inliers)
        final_ok = fit_ok & fit_ok2
        if not final_ok.any():
            out["surface_status"][start + rows] = "ill_conditioned_plane"
            continue

        pred2 = np.einsum("nki,ni->nk", design, np.nan_to_num(beta2))
        resid2 = z_rel - pred2
        resid2 = np.where(inliers & final_ok[:, None], resid2, np.nan)
        resid_center2 = np.full(len(rows), np.nan)
        residual_nmad = np.full(len(rows), np.nan)
        roughness = np.full(len(rows), np.nan)
        median_height = np.full(len(rows), np.nan)
        if final_ok.any():
            resid_center2[final_ok] = np.nanmedian(resid2[final_ok], axis=1)
            residual_nmad[final_ok] = 1.4826 * np.nanmedian(
                np.abs(resid2[final_ok] - resid_center2[final_ok, None]), axis=1
            )
            roughness[final_ok] = 1.4826 * np.nanmedian(
                np.abs(z_masked[final_ok] - np.nanmedian(z_masked[final_ok], axis=1)[:, None]), axis=1
            )
            median_height[final_ok] = np.nanmedian(
                np.where(inliers[final_ok], local_z[final_ok], np.nan), axis=1
            )
        slope = np.degrees(np.arctan(np.hypot(beta2[:, 0], beta2[:, 1])))
        plane_height = z_base + beta2[:, 2] + resid_center2

        absolute_rows = start + rows
        valid_rows = absolute_rows[final_ok]
        out["reference_height_plane_m"][valid_rows] = plane_height[final_ok]
        out["reference_height_median_m"][valid_rows] = median_height[final_ok]
        out["robust_neighbor_count"][valid_rows] = inliers.sum(axis=1)[final_ok].astype(np.int16)
        out["local_slope_deg"][valid_rows] = slope[final_ok]
        out["plane_residual_nmad_m"][valid_rows] = residual_nmad[final_ok]
        out["ground_roughness_nmad_m"][valid_rows] = roughness[final_ok]
        out["plane_condition_number"][valid_rows] = condition2[final_ok]
        out["plane_fit_valid"][valid_rows] = True
        acceptable = (
            (slope[final_ok] <= max_slope_deg)
            & (residual_nmad[final_ok] <= max_plane_residual_nmad_m)
        )
        out["surface_acceptable"][valid_rows] = acceptable
        out["surface_status"][valid_rows] = np.where(
            slope[final_ok] > max_slope_deg,
            "steep_local_surface",
            np.where(residual_nmad[final_ok] > max_plane_residual_nmad_m, "rough_local_surface", "matched"),
        )
        failed_rows = absolute_rows[~final_ok]
        out["surface_status"][failed_rows] = "ill_conditioned_plane"

    return out


def elevation_metrics(values: np.ndarray) -> dict[str, float | int | None]:
    """Summarize signed elevation differences; these metrics make no accuracy claim."""

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not x.size:
        return {
            "n": 0,
            "bias_m": None,
            "median_m": None,
            "std_m": None,
            "rmse_m": None,
            "mae_m": None,
            "nmad_m": None,
            "p05_m": None,
            "p50_m": None,
            "p95_m": None,
            "absolute_p90_m": None,
            "absolute_p95_m": None,
        }
    median = float(np.median(x))
    absolute = np.abs(x)
    return {
        "n": int(x.size),
        "bias_m": float(np.mean(x)),
        "median_m": median,
        "std_m": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "rmse_m": float(np.sqrt(np.mean(np.square(x)))),
        "mae_m": float(np.mean(absolute)),
        "nmad_m": float(1.4826 * np.median(np.abs(x - median))),
        "p05_m": float(np.percentile(x, 5)),
        "p50_m": median,
        "p95_m": float(np.percentile(x, 95)),
        "absolute_p90_m": float(np.percentile(absolute, 90)),
        "absolute_p95_m": float(np.percentile(absolute, 95)),
    }


def empirical_alignment_metrics(
    differences_m: np.ndarray,
    calibration_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    test_denominator: int | None = None,
) -> dict[str, Any]:
    """Estimate a median offset on calibration points and score disjoint test points."""

    d = np.asarray(differences_m, dtype=np.float64).reshape(-1)
    calibration = np.asarray(calibration_mask, dtype=bool).reshape(-1)
    test = np.asarray(test_mask, dtype=bool).reshape(-1)
    if d.shape != calibration.shape or d.shape != test.shape:
        raise ValueError("Differences and split masks must have the same length.")
    if np.any(calibration & test):
        raise ValueError("Calibration and test samples must be disjoint.")
    calibration &= np.isfinite(d)
    test &= np.isfinite(d)
    if not calibration.any() or not test.any():
        raise ValueError("Both calibration and independent test samples must be non-empty.")
    offset = float(np.median(d[calibration]))
    residual = d[test] - offset
    summary = elevation_metrics(residual)
    denom = int(test_denominator) if test_denominator is not None else int(test.sum())
    summary.update({
        "empirical_vertical_offset_m": offset,
        "calibration_n": int(calibration.sum()),
        "test_n": int(test.sum()),
        "test_denominator_n": denom,
        "test_coverage": float(test.sum() / denom) if denom else None,
        "evaluation_semantics": "spatial holdout agreement after empirical median alignment; not strict absolute accuracy",
    })
    return summary


def spatial_site_masks(
    site_ids: np.ndarray,
    calibration_site_ids: set[str] | tuple[str, ...] | list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Split by predeclared spatial site identity, independently of residuals."""

    sites = np.asarray(site_ids, dtype=object).reshape(-1)
    if any(value is None or str(value) == "" for value in sites):
        raise ValueError("Every evaluation point must have a spatial site identity.")
    calibration_sites = {str(value) for value in calibration_site_ids}
    if not calibration_sites:
        raise ValueError("At least one calibration site must be declared.")
    calibration = np.isin(sites.astype(str), list(calibration_sites))
    test = ~calibration
    if not calibration.any() or not test.any():
        raise ValueError("Calibration and test sites must both be represented.")
    return calibration, test


def snr_strata(snr: np.ndarray) -> dict[str, np.ndarray]:
    """Return the complete valid population and nested SNR sensitivity masks."""

    value = np.asarray(snr, dtype=np.float64).reshape(-1)
    valid = np.isfinite(value)
    return {
        "all_valid_refh": valid,
        "snr_ge_3": valid & (value >= 3.0),
        "snr_ge_5": valid & (value >= 5.0),
    }
