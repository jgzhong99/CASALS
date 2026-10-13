"""Ground-target and reference-return evaluation for CASALS Notebook 10.

This analysis reuses the saved full-granule 3DEP matches. Road interiors are
treated as expected-ground target areas based on independently reviewed NAIP
geometry; the H5 Refh return origin remains unresolved for every pulse.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research.reference import vertical_datum_comparison as datum


ROOT = Path(__file__).resolve().parents[2]
GROUND_OUTPUT_ROOT = ROOT / "outputs/notebooks/10_3dep_elevation_validation/ground_aware"
ROI_PATH = ROOT / "data/reference/landcover_rois/roi_manifest.geojson"
NAIP_MANIFEST_PATH = ROOT / "outputs/notebooks/11_landcover_lidar_response/visual_revision/input_manifest.json"
LEGACY_DATUM_ROOT = ROOT / "outputs/notebooks/10_3dep_elevation_validation/datum_comparison"
LEGACY_VISUAL_ROOT = ROOT / "outputs/notebooks/10_3dep_elevation_validation/visual_revision"
REVISED_ROOT = ROOT / "outputs/notebooks/10_3dep_elevation_validation/revised"
ROAD_BOUNDARY_TRIM_M = {
    "road_core_12_01": 2.0,
    "road_core_01": 0.5,
}
SPLIT_GUARD_GAP_M = 2.0
GROUND_CONTEXT_VERSION = "ground-context-v1"

_PARQUET_COLUMNS = [
    "pulse_index", "sweep_num", "track_num", "longitude_deg", "latitude_deg",
    "casals_refh_m", "refh_snr", "easting_m_epsg6347", "northing_m_epsg6347",
    "reference_height_local_plane_m", "reference_height_local_median_m",
    "neighbor_count_within_3m", "local_slope_deg", "plane_residual_nmad_m",
    "ground_roughness_nmad_m", "surface_acceptable", "surface_status", "spatial_split",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def _safe_metrics(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "bias_m": None, "median_m": None, "rmse_m": None,
                "mae_m": None, "nmad_m": None, "abs_p95_m": None, "max_abs_m": None}
    median = float(np.median(array))
    return {
        "n": int(len(array)), "bias_m": float(np.mean(array)), "median_m": median,
        "rmse_m": float(np.sqrt(np.mean(array**2))), "mae_m": float(np.mean(np.abs(array))),
        "nmad_m": float(1.4826 * np.median(np.abs(array - median))),
        "abs_p95_m": float(np.percentile(np.abs(array), 95)),
        "max_abs_m": float(np.max(np.abs(array))),
    }


def _load_naip_sources() -> dict[str, dict[str, Path]]:
    manifest = json.loads(NAIP_MANIFEST_PATH.read_text(encoding="utf-8"))
    sources: dict[str, dict[str, Path]] = {}
    for granule_id, item in manifest.get("naip_assets", {}).items():
        sources[str(granule_id)] = {
            "image": ROOT / item["image_path"],
            "metadata": ROOT / item["metadata_path"],
        }
    if not sources:
        raise ValueError(f"No NAIP assets were found in {NAIP_MANIFEST_PATH}.")
    return sources


def _context_label(properties: Mapping[str, Any]) -> tuple[str, str]:
    landcover = str(properties.get("landcover_class", properties.get("class_label", "unknown")))
    roi_id = str(properties.get("roi_id", ""))
    if landcover == "road_surface" and roi_id in ROAD_BOUNDARY_TRIM_M:
        return "ground_target_eligible", "Reviewed road-core geometry; likely ground target, return origin unverified."
    if landcover == "tree_canopy":
        return "non_ground_context", "Canopy context; Refh may represent an unresolved return and is not a ground label."
    return "uncertain", "Historical imagery or water context does not establish stable ground at the CASALS date."


def build_ground_context_evidence(
    roi_manifest: Mapping[str, Any],
    naip_sources: Mapping[str, Mapping[str, Path]],
    h5_dates: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copy ROI features and add evidence fields without changing source labels."""
    import rasterio

    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as transform_geometry

    acquisition_dates = roi_manifest.get("metadata", {}).get("naip_acquisition_dates_by_granule", {})
    features_out = []
    geometries: dict[str, Any] = {}
    audit_rows = []
    for source_feature in roi_manifest.get("features", []):
        feature = json.loads(json.dumps(source_feature))
        properties = feature["properties"]
        granule_id = str(properties["granule_id"])
        roi_id = str(properties["roi_id"])
        if granule_id not in naip_sources or granule_id not in h5_dates:
            raise ValueError(f"No current image/H5 metadata is available for {granule_id}.")
        naip_metadata = json.loads(naip_sources[granule_id]["metadata"].read_text(encoding="utf-8"))
        expected_date = str(acquisition_dates.get(granule_id, ""))
        actual_dates = naip_metadata.get("acquisition_dates", [])
        if actual_dates != [expected_date]:
            raise ValueError(f"NAIP date mismatch for {granule_id}: {actual_dates} != {[expected_date]}.")
        with rasterio.open(naip_sources[granule_id]["image"]) as image:
            transform = image.transform
            if image.crs is None or not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0) or transform.a <= 0 or transform.e >= 0:
                raise ValueError(f"NAIP raster must be projected and north-up: {naip_sources[granule_id]['image']}.")
            raster_crs = image.crs.to_string()
            raster_bounds = tuple(float(v) for v in image.bounds)
            affine = tuple(float(v) for v in transform)
        h5_date = str(h5_dates[granule_id])
        age_days = (date.fromisoformat(h5_date) - date.fromisoformat(expected_date)).days
        status, reason = _context_label(properties)
        trim = float(ROAD_BOUNDARY_TRIM_M.get(roi_id, 0.0))
        properties.update({
            "ground_context_status": status,
            "ground_context_evidence": reason,
            "ground_target_boundary_trim_m": trim,
            "ground_target_geometry_method": "inward buffer of manually reviewed NAIP polygon in EPSG:6347" if trim else "no inward buffer; context only",
            "return_origin_status": "unresolved_for_CASALS_Refh",
            "h5_acquisition_date_verified": h5_date,
            "naip_acquisition_date_verified": expected_date,
            "naip_age_days_at_h5": int(age_days),
            "naip_crs_verified": raster_crs,
            "naip_affine_verified": list(affine),
            "naip_bounds_verified": list(raster_bounds),
            "context_method_version": GROUND_CONTEXT_VERSION,
        })
        feature_geometry = shape(feature["geometry"])
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:6347", always_xy=True)
        projected = transform_geometry(transformer.transform, feature_geometry)
        interior = projected.buffer(-trim) if status == "ground_target_eligible" and trim > 0 else projected
        if interior.is_empty:
            raise ValueError(f"Road boundary trim removed the entire target polygon: {roi_id}.")
        geometries[roi_id] = {
            "polygon": projected,
            "interior": interior,
            "granule_id": granule_id,
            "properties": properties,
        }
        features_out.append(feature)
        audit_rows.append({
            "granule_id": granule_id, "roi_id": roi_id,
            "site_id": str(properties.get("site_id", "")),
            "landcover_class": str(properties.get("landcover_class", properties.get("class_label", "unknown"))),
            "ground_context_status": status, "return_origin_status": "unresolved_for_CASALS_Refh",
            "source_analysis_status": properties.get("analysis_status"),
            "source_visual_confidence": properties.get("visual_interpretation_confidence", properties.get("confidence")),
            "naip_acquisition_date": expected_date, "h5_acquisition_date": h5_date,
            "naip_age_days_at_h5": int(age_days), "boundary_trim_m": trim,
            "naip_crs": raster_crs, "naip_affine": affine, "naip_bounds": raster_bounds,
            "evidence_note": reason,
            "temporal_and_registration_limit": "NAIP predates H5; nominal H5 centers and polygon registration are not calibrated.",
        })
    return features_out, {"geometries": geometries, "audit": pd.DataFrame(audit_rows)}


def build_reviewed_pulse_population(
    matched: pd.DataFrame,
    geometry_evidence: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Assign pulse context from projected polygons only, never from signal/error."""
    import shapely

    x = matched.easting_m_epsg6347.to_numpy(dtype=np.float64)
    y = matched.northing_m_epsg6347.to_numpy(dtype=np.float64)
    hits: list[pd.DataFrame] = []
    for roi_id, item in geometry_evidence.items():
        polygon = item["polygon"]
        min_x, min_y, max_x, max_y = polygon.bounds
        candidates = np.flatnonzero((x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y))
        if not candidates.size:
            continue
        inside = shapely.contains_xy(polygon, x[candidates], y[candidates])
        candidates = candidates[inside]
        if not candidates.size:
            continue
        rows = matched.iloc[candidates].copy()
        rows["roi_id"] = roi_id
        rows["site_id"] = str(item["properties"].get("site_id", ""))
        rows["landcover_class"] = str(item["properties"].get("landcover_class", item["properties"].get("class_label", "unknown")))
        rows["ground_context_status"] = str(item["properties"]["ground_context_status"])
        rows["return_origin_status"] = "unresolved_for_CASALS_Refh"
        rows["naip_acquisition_date"] = str(item["properties"]["naip_acquisition_date_verified"])
        rows["h5_acquisition_date"] = str(item["properties"]["h5_acquisition_date_verified"])
        rows["naip_age_days_at_h5"] = int(item["properties"]["naip_age_days_at_h5"])
        trim = float(item["properties"]["ground_target_boundary_trim_m"])
        rows["ground_target_boundary_trim_m"] = trim
        if trim > 0:
            interior = item["interior"]
            is_target = shapely.contains_xy(interior, x[candidates], y[candidates])
            rows["ground_context_status"] = np.where(is_target, "ground_target_eligible", "uncertain")
            rows["eligibility_reason"] = np.where(
                is_target,
                "inside reviewed road core after fixed boundary trim; likely-ground target only",
                "inside road polygon but within fixed boundary trim; excluded from primary target evaluation",
            )
        else:
            is_target = np.zeros(len(rows), dtype=bool)
            rows["eligibility_reason"] = str(item["properties"]["ground_context_evidence"])
        rows["is_ground_target_eligible"] = is_target.astype(bool)
        rows["original_surface_acceptable"] = rows.surface_acceptable.astype(bool)
        hits.append(rows)
    if not hits:
        raise ValueError("No matched CASALS pulse centers intersect the reviewed ROI contexts.")
    population = pd.concat(hits, ignore_index=True)
    duplicate = population.duplicated(["granule_id", "pulse_index"], keep=False)
    if duplicate.any():
        overlap = population.loc[duplicate, ["granule_id", "pulse_index"]].drop_duplicates()
        if len(overlap):
            population.loc[duplicate, "ground_context_status"] = "uncertain"
            population.loc[duplicate, "is_ground_target_eligible"] = False
            population.loc[duplicate, "eligibility_reason"] = "overlapping reviewed ROI polygons; excluded from primary evaluation"
            population.loc[duplicate, "roi_id"] = "ambiguous_overlap"
            population.loc[duplicate, "site_id"] = "ambiguous_overlap"
            population.loc[duplicate, "landcover_class"] = "ambiguous_overlap"
            population = population.drop_duplicates(["granule_id", "pulse_index"], keep="first")
    population["pulse_index"] = population.pulse_index.astype(np.int64)
    return population.sort_values(["granule_id", "pulse_index"], kind="mergesort").reset_index(drop=True)


def _principal_axis(polygon: Any) -> tuple[np.ndarray, float, float]:
    coords = np.asarray(polygon.exterior.coords, dtype=np.float64)[:, :2]
    center = coords[:-1].mean(axis=0)
    covariance = np.cov((coords[:-1] - center).T, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    if axis[0] < 0 or (np.isclose(axis[0], 0.0) and axis[1] < 0):
        axis = -axis
    projection = (coords[:-1] - center) @ axis
    return axis, float(projection.min()), float(projection.max())


def freeze_ground_spatial_split(
    population: pd.DataFrame,
    geometry_evidence: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    *,
    guard_gap_m: float = SPLIT_GUARD_GAP_M,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Freeze road roles using only site geometry and pulse coordinates."""
    if guard_gap_m < 0:
        raise ValueError("guard_gap_m must be non-negative.")
    ground = population.loc[population.is_ground_target_eligible.astype(bool)].copy()
    if ground.empty:
        raise ValueError("No road-core pulses remain after the fixed geometry boundary trim.")
    rows = []
    for roi_id, group in ground.groupby("roi_id", sort=True):
        item = geometry_evidence[str(roi_id)]
        polygon = item["interior"]
        axis, low, high = _principal_axis(polygon)
        xy = group[["easting_m_epsg6347", "northing_m_epsg6347"]].to_numpy(dtype=np.float64)
        center = np.asarray(polygon.centroid.coords[0], dtype=np.float64)
        along = (xy - center) @ axis
        span = high - low
        cut1, cut2 = low + span / 3.0, low + 2.0 * span / 3.0
        half_gap = guard_gap_m / 2.0
        role = np.full(len(group), "guard_gap", dtype=object)
        role[along < cut1 - half_gap] = "calibration_offset"
        role[(along > cut1 + half_gap) & (along < cut2 - half_gap)] = "candidate_selection"
        role[along > cut2 + half_gap] = "independent_test"
        for local, row in enumerate(group.itertuples(index=False)):
            rows.append({
                "granule_id": str(row.granule_id), "pulse_index": int(row.pulse_index),
                "roi_id": str(roi_id), "site_id": str(row.site_id),
                "research_role": str(role[local]),
                "axis_distance_m": float(along[local]),
                "axis_unit_easting": float(axis[0]), "axis_unit_northing": float(axis[1]),
                "axis_extent_min_m": low, "axis_extent_max_m": high,
                "guard_gap_m": float(guard_gap_m),
                "easting_m_epsg6347": float(row.easting_m_epsg6347),
                "northing_m_epsg6347": float(row.northing_m_epsg6347),
                "fixed_before_candidate_residuals": True,
                "membership_basis": "reviewed polygon interior and projected position along dominant polygon axis; no Refh residual, candidate RMSE, SNR, or support metric used",
            })
    manifest = pd.DataFrame(rows).sort_values(["granule_id", "roi_id", "pulse_index"], kind="mergesort").reset_index(drop=True)
    if manifest.duplicated(["granule_id", "pulse_index"]).any():
        raise RuntimeError("A pulse received multiple supplemental roles.")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ground_spatial_split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    role_counts = {f"{k[0]}::{k[1]}": int(v) for k, v in manifest.groupby(["granule_id", "research_role"], sort=True).size().items()}
    summary = {
        "method": "Geometry-only supplemental split within each road-core polygon; axis positions divided into thirds with 2 m total guard gaps around the two cuts.",
        "frozen_before_candidate_residuals": True,
        "roles": ["calibration_offset", "candidate_selection", "independent_test", "guard_gap"],
        "guard_gap_m": float(guard_gap_m), "role_counts": role_counts,
        "independent_sites": sorted(manifest.loc[manifest.research_role == "independent_test", "site_id"].unique().tolist()),
        "independent_site_limit": "Segments are spatially separated portions of one road site per granule, not independent sites.",
        "manifest_sha256": _sha256(manifest_path),
    }
    _json_write(output_dir / "ground_spatial_split_audit.json", summary)
    return manifest, summary


def _load_ground_candidate_frame(
    granule_id: str,
    population: pd.DataFrame,
    split_manifest: pd.DataFrame,
    h5_path: Path,
) -> pd.DataFrame:
    import h5py

    from research.reference.vertical_datum_comparison import build_height_candidates

    target = population.loc[
        (population.granule_id.astype(str) == granule_id)
        & population.is_ground_target_eligible.astype(bool)
    ].copy()
    roles = split_manifest.loc[split_manifest.granule_id.astype(str) == granule_id]
    target = target.merge(roles[["pulse_index", "research_role", "axis_distance_m"]], on="pulse_index", how="inner", validate="one_to_one")
    target = target.loc[target.research_role != "guard_gap"].copy()
    target.sort_values("pulse_index", inplace=True, kind="mergesort")
    ids = target.pulse_index.to_numpy(dtype=np.int64)
    with h5py.File(h5_path, "r") as h5:
        if int(h5["refh"].shape[0]) <= int(ids.max(initial=-1)):
            raise ValueError(f"Ground context pulse IDs exceed the H5 pulse population for {granule_id}.")
        refh = np.asarray(h5["refh"][ids], dtype=np.float64)
        geoid = np.asarray(h5["geoid"][ids], dtype=np.float64) if "geoid" in h5 else None
        free = np.asarray(h5["geoid_free2mean"][ids], dtype=np.float64) if "geoid_free2mean" in h5 else None
        lon = np.asarray(h5["refh_longitude"][ids], dtype=np.float64)
        lat = np.asarray(h5["refh_latitude"][ids], dtype=np.float64)
        snr = np.asarray(h5["refh_snr"][ids], dtype=np.float64)
        if not np.allclose(target.casals_refh_m, refh, rtol=0.0, atol=1e-9):
            raise RuntimeError(f"Ground pulse Refh differs from its H5 source rows for {granule_id}.")
        if not np.allclose(target.longitude_deg, lon, rtol=0.0, atol=1e-10) or not np.allclose(target.latitude_deg, lat, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"Ground pulse coordinates differ from their H5 source rows for {granule_id}.")
        if not np.allclose(target.refh_snr, snr, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Ground pulse SNR differs from its H5 source rows for {granule_id}.")
    candidates, warnings = build_height_candidates(refh, geoid, free)
    for candidate_id, values in candidates.items():
        target[f"{candidate_id}_height_m"] = values
    target["ground_candidate_warnings"] = "; ".join(warnings)
    return target


def _ground_metric_rows(frame: pd.DataFrame, *, roles: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in roles:
        part = frame.loc[frame.research_role.astype(str) == role]
        if part.empty:
            continue
        common = part.common_support_all_candidates.to_numpy(dtype=bool)
        strata = {
            "all_valid_refh": np.ones(len(part), dtype=bool),
            "snr_ge_3": part.refh_snr.to_numpy(dtype=np.float64) >= 3.0,
            "snr_ge_5": part.refh_snr.to_numpy(dtype=np.float64) >= 5.0,
        }
        for stratum, snr_mask in strata.items():
            keep = common & snr_mask
            denominator = int(snr_mask.sum())
            for candidate_id in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"):
                candidate_col = f"{candidate_id}_height_m"
                reference_col = "c7_reference_height_local_plane_m" if candidate_id == "C7" else "reference_height_local_plane_m"
                if candidate_col not in part or reference_col not in part:
                    continue
                residual = part[candidate_col].to_numpy(dtype=np.float64)[keep] - part[reference_col].to_numpy(dtype=np.float64)[keep]
                metrics = _safe_metrics(residual)
                rows.append({
                    "granule_id": str(part.granule_id.iloc[0]), "candidate_id": candidate_id,
                    "candidate_label": datum.CANDIDATE_LABELS[candidate_id],
                    "research_role": role, "snr_stratum": stratum,
                    "analysis_population": "conditional likely-ground target; reviewed road-core interior only",
                    "return_origin_status": "unresolved_for_CASALS_Refh",
                    "n_geometry_role_and_snr": denominator,
                    "n_common_c0_c7_support": int(keep.sum()),
                    "coverage": float(keep.sum() / denominator) if denominator else 0.0,
                    "difference_definition": "candidate height minus local 3DEP class-2-derived plane; C7 uses plane re-queried at transformed XY",
                    **metrics,
                })
    return rows


def _ground_test_metrics(frames: Sequence[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    residual_frames = []
    empirical_audit: dict[str, Any] = {
        "method": "C8 subtracts the median C0 minus local-plane residual fitted on all common-support calibration_offset road pulses; the same frozen offset is applied to the independent test.",
        "test_data_used_for_fit": False,
        "granules": {},
    }
    for frame in frames:
        granule_id = str(frame.granule_id.iloc[0])
        common = frame.common_support_all_candidates.to_numpy(dtype=bool)
        calibration = frame.research_role.astype(str).to_numpy() == "calibration_offset"
        test = frame.research_role.astype(str).to_numpy() == "independent_test"
        raw = frame.C0_height_m.to_numpy(dtype=np.float64) - frame.reference_height_local_plane_m.to_numpy(dtype=np.float64)
        fit = calibration & common & np.isfinite(raw)
        offset = float(np.median(raw[fit])) if fit.any() else None
        empirical_audit["granules"][granule_id] = {
            "calibration_offset_m": offset, "calibration_common_n": int(fit.sum()),
            "test_common_n": int((test & common).sum()),
            "fit_population": "all valid SNR values; no SNR stratum, residual truncation, or test rows used",
        }
        test_part = frame.loc[test].copy()
        for candidate_id in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"):
            candidate_col = f"{candidate_id}_height_m"
            reference_col = "c7_reference_height_local_plane_m" if candidate_id == "C7" else "reference_height_local_plane_m"
            test_part[f"{candidate_id}_residual_m"] = test_part[candidate_col] - test_part[reference_col]
        if offset is not None:
            test_part["C8_height_m"] = test_part.C0_height_m - offset
            test_part["C8_residual_m"] = test_part.C8_height_m - test_part.reference_height_local_plane_m
        else:
            test_part["C8_height_m"] = np.nan
            test_part["C8_residual_m"] = np.nan
        common_test = test_part.loc[test_part.common_support_all_candidates.astype(bool)].copy()
        for stratum, mask in {
            "all_valid_refh": np.ones(len(common_test), dtype=bool),
            "snr_ge_3": common_test.refh_snr.to_numpy(dtype=np.float64) >= 3.0,
            "snr_ge_5": common_test.refh_snr.to_numpy(dtype=np.float64) >= 5.0,
        }.items():
            denominator_mask = {
                "all_valid_refh": np.ones(len(test_part), dtype=bool),
                "snr_ge_3": test_part.refh_snr.to_numpy(dtype=np.float64) >= 3.0,
                "snr_ge_5": test_part.refh_snr.to_numpy(dtype=np.float64) >= 5.0,
            }[stratum]
            denominator = int(denominator_mask.sum())
            selected = common_test.loc[mask]
            for configuration in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):
                residual_col = f"{configuration}_residual_m"
                metrics = _safe_metrics(selected[residual_col].to_numpy(dtype=np.float64))
                rows.append({
                    "granule_id": granule_id, "configuration_id": configuration,
                    "configuration_label": datum.CANDIDATE_LABELS.get(configuration, "C8 empirical alignment"),
                    "configuration_status": "PROVISIONAL_NOT_DATUM_VERIFIED" if configuration == "C7" else ("EMPIRICAL_ONLY" if configuration == "C8" else "diagnostic candidate"),
                    "research_role": "independent_test", "snr_stratum": stratum,
                    "n_geometry_role_and_snr": denominator, "n_common_c0_c7_support": int(len(selected)),
                    "coverage": float(len(selected) / denominator) if denominator else 0.0,
                    "difference_definition": "configuration minus local 3DEP class-2-derived ground plane; C7 reference uses transformed XY; C8 uses calibration-only offset",
                    **metrics,
                })
        test_part["empirical_offset_calibration_m"] = offset
        residual_frames.append(test_part)
    return pd.DataFrame(rows), pd.concat(residual_frames, ignore_index=True), empirical_audit


def _query_c7_descriptive_context(
    population: pd.DataFrame,
    training_surfaces: Mapping[str, tuple[np.ndarray, np.ndarray]],
    empirical_audit: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add provisional C7/C8 context differences using already-built class-2 grids."""
    from research.elevation_comparison import interpolate_ground_surface

    output = population.reset_index(drop=True).copy()
    n = len(output)
    float_columns = (
        "c7_easting_m_epsg6347", "c7_northing_m_epsg6347", "c7_refh_height_m",
        "c7_reference_height_local_plane_m", "c7_reference_height_local_median_m",
        "c7_nearest_ground_distance_m", "c7_local_slope_deg", "c7_ground_roughness_nmad_m",
        "C0_refh_minus_original_ground_m", "C7_refh_minus_transformed_ground_m",
        "C8_empirical_refh_minus_original_ground_m", "c8_empirical_offset_m",
    )
    for column in float_columns:
        output[column] = np.full(n, np.nan, dtype=np.float64)
    output["c7_neighbor_count"] = np.zeros(n, dtype=np.int16)
    output["c7_surface_acceptable"] = False
    output["c7_surface_status"] = "transform_unavailable"

    audit: dict[str, Any] = {
        "purpose": "Descriptive context comparison only; does not identify the physical origin of CASALS Refh.",
        "training_surface_reused": True,
        "additional_laz_scan": False,
        "granules": {},
    }
    for granule_id, positions in output.groupby("granule_id", sort=True).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        group = output.iloc[positions]
        lon = group.longitude_deg.to_numpy(dtype=np.float64)
        lat = group.latitude_deg.to_numpy(dtype=np.float64)
        refh = group.casals_refh_m.to_numpy(dtype=np.float64)
        valid_input = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(refh)
        if not valid_input.any():
            audit["granules"][str(granule_id)] = {
                "context_pulse_count": int(len(group)), "finite_input_count": 0,
                "finite_transformed_count": 0, "c7_surface_acceptable_count": 0,
                "status": "NO_FINITE_CONTEXT_INPUTS",
            }
            continue

        x, y, z, transform_audit = datum.transform_c7(
            lon[valid_input], lat[valid_input], refh[valid_input], return_audit=True,
        )
        x, y, z = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
        finite_output = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        valid_rows = np.flatnonzero(valid_input)
        c7_positions = positions[valid_rows[finite_output]]
        empirical = empirical_audit["granules"].get(str(granule_id), {})
        offset = empirical.get("calibration_offset_m")
        if finite_output.any():
            for column, values in (
                ("c7_easting_m_epsg6347", x[finite_output]),
                ("c7_northing_m_epsg6347", y[finite_output]),
                ("c7_refh_height_m", z[finite_output]),
            ):
                output.loc[c7_positions, column] = values

            training_xy, training_z = training_surfaces[str(granule_id)]
            surface = interpolate_ground_surface(
                np.column_stack((x[finite_output], y[finite_output])), training_xy, training_z,
            )
            surface_columns = {
                "c7_reference_height_local_plane_m": "reference_height_plane_m",
                "c7_reference_height_local_median_m": "reference_height_median_m",
                "c7_nearest_ground_distance_m": "nearest_ground_distance_m",
                "c7_local_slope_deg": "local_slope_deg",
                "c7_ground_roughness_nmad_m": "ground_roughness_nmad_m",
                "c7_neighbor_count": "neighbor_count",
                "c7_surface_acceptable": "surface_acceptable",
                "c7_surface_status": "surface_status",
            }
            for output_column, surface_column in surface_columns.items():
                output.loc[c7_positions, output_column] = surface[surface_column]

            original_plane = group.reference_height_local_plane_m.to_numpy(dtype=np.float64)[valid_rows[finite_output]]
            original_ok = group.original_surface_acceptable.to_numpy(dtype=bool)[valid_rows[finite_output]]
            c0_values = refh[valid_rows[finite_output]] - original_plane
            c0_values[~original_ok | ~np.isfinite(original_plane)] = np.nan
            c7_values = z[finite_output] - surface["reference_height_plane_m"]
            c7_values[~surface["surface_acceptable"]] = np.nan
            c8_values = np.full(len(c7_positions), np.nan, dtype=np.float64)
            if offset is not None:
                c8_values = c0_values - float(offset)
            for column, values in (
                ("C0_refh_minus_original_ground_m", c0_values),
                ("C7_refh_minus_transformed_ground_m", c7_values),
                ("C8_empirical_refh_minus_original_ground_m", c8_values),
                ("c8_empirical_offset_m", np.full(len(c7_positions), np.nan if offset is None else float(offset))),
            ):
                output.loc[c7_positions, column] = values

        audit["granules"][str(granule_id)] = {
            "context_pulse_count": int(len(group)),
            "finite_input_count": int(valid_input.sum()),
            "finite_transformed_count": int(finite_output.sum()),
            "c7_surface_acceptable_count": int(surface["surface_acceptable"].sum()) if finite_output.any() else 0,
            "c7_surface_supported_count": int(surface["plane_fit_valid"].sum()) if finite_output.any() else 0,
            "c8_calibration_offset_m": empirical.get("calibration_offset_m"),
            "c8_calibration_count": empirical.get("calibration_common_n"),
            "transform": transform_audit,
            "status": "PROVISIONAL_NOT_DATUM_VERIFIED",
        }

    output["C0_refh_minus_original_ground_m"] = (
        output.casals_refh_m - output.reference_height_local_plane_m
    ).where(output.original_surface_acceptable.astype(bool))
    return output, audit


def _descriptive_reference_metrics(population: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in population.groupby(["granule_id", "roi_id", "site_id", "landcover_class", "ground_context_status"], sort=True):
        granule_id, roi_id, site_id, landcover, status = keys
        supported = group.loc[
            group.original_surface_acceptable.astype(bool)
            & np.isfinite(group.casals_refh_m)
            & np.isfinite(group.reference_height_local_plane_m)
        ]
        descriptive = supported.casals_refh_m.to_numpy(dtype=np.float64) - supported.reference_height_local_plane_m.to_numpy(dtype=np.float64)
        row = {
            "granule_id": granule_id, "roi_id": roi_id, "site_id": site_id,
            "landcover_class": landcover, "ground_context_status": status,
            "interpretation": "descriptive Refh minus local ground surface; not canopy height or a ground-return error, and C7 remains provisional",
            "n_context_pulses": int(len(group)), "n_original_surface_acceptable": int(len(supported)),
            "support_fraction": float(len(supported) / len(group)) if len(group) else 0.0,
            **_safe_metrics(descriptive),
        }
        if "C7_refh_minus_transformed_ground_m" in group:
            c7_supported = group.loc[np.isfinite(group.C7_refh_minus_transformed_ground_m)]
            c8_supported = group.loc[np.isfinite(group.C8_empirical_refh_minus_original_ground_m)]
            row.update({
                "n_c7_surface_acceptable": int(group.c7_surface_acceptable.astype(bool).sum()),
                "c7_support_fraction": float(group.c7_surface_acceptable.astype(bool).sum() / len(group)) if len(group) else 0.0,
                "n_c7_descriptive_differences": int(len(c7_supported)),
                "n_c8_descriptive_differences": int(len(c8_supported)),
                "C7_interpretation": "Provisional transformed Refh minus class-2 local plane; return origin unresolved.",
                "C8_interpretation": "Calibration-only empirical offset applied to native Refh-minus-original-plane; not a datum conversion.",
                **{f"C0_{key}": value for key, value in _safe_metrics(
                    group.C0_refh_minus_original_ground_m.to_numpy(dtype=np.float64)
                ).items()},
                **{f"C7_{key}": value for key, value in _safe_metrics(
                    c7_supported.C7_refh_minus_transformed_ground_m.to_numpy(dtype=np.float64)
                ).items()},
                **{f"C8_{key}": value for key, value in _safe_metrics(
                    c8_supported.C8_empirical_refh_minus_original_ground_m.to_numpy(dtype=np.float64)
                ).items()},
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _select_waveform_cases(
    population: pd.DataFrame,
    geometry_evidence: Mapping[str, Mapping[str, Any]],
    candidate_frames: Sequence[pd.DataFrame],
    h5_by_granule: Mapping[str, Path],
    test_residuals: pd.DataFrame,
    training_surfaces: Mapping[str, tuple[np.ndarray, np.ndarray]],
    empirical_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import h5py
    from pyproj import Transformer
    import rasterio
    from shapely.geometry import Point

    from research.elevation_comparison import interpolate_ground_surface

    candidate_by_granule = {}
    for frame in candidate_frames:
        granule_id = str(frame.granule_id.iloc[0])
        candidate_by_granule[granule_id] = frame.set_index("pulse_index", drop=False)
    common_test = test_residuals.loc[
        test_residuals.common_support_all_candidates.astype(bool)
        & (test_residuals.research_role.astype(str) == "independent_test")
        & np.isfinite(test_residuals.C7_residual_m)
    ].copy()
    if common_test.empty:
        raise ValueError("No common-support independent-test pulse is available for deterministic waveform cases.")
    common_test["absolute_C7_residual_m"] = common_test.C7_residual_m.abs()
    low_ground = common_test.sort_values(["absolute_C7_residual_m", "pulse_index"], kind="mergesort").iloc[0]
    high_ground = common_test.sort_values(
        ["absolute_C7_residual_m", "pulse_index"], ascending=[False, True], kind="mergesort",
    ).iloc[0]

    canopy = population.loc[
        (population.ground_context_status.astype(str) == "non_ground_context")
        & population.original_surface_acceptable.astype(bool)
    ].copy()
    if canopy.empty:
        raise ValueError("No supported non-ground-context pulse is available for case review.")
    canopy["absolute_C0_difference_m"] = (
        canopy.casals_refh_m - canopy.reference_height_local_plane_m
    ).abs()
    large_context = canopy.sort_values(
        ["absolute_C0_difference_m", "pulse_index"], ascending=[False, True], kind="mergesort",
    ).iloc[0]

    boundary = population.loc[
        population.ground_context_status.astype(str).eq("uncertain")
        & population.roi_id.astype(str).str.startswith("road_core_")
        & population.original_surface_acceptable.astype(bool)
    ].copy()
    if boundary.empty:
        raise ValueError("No supported uncertain road-boundary pulse is available for case review.")
    boundary_distance = []
    for row in boundary.itertuples(index=False):
        interior = geometry_evidence[str(row.roi_id)]["interior"]
        boundary_distance.append(interior.boundary.distance(Point(row.easting_m_epsg6347, row.northing_m_epsg6347)))
    boundary["distance_to_target_boundary_m"] = boundary_distance
    mixed_context = boundary.sort_values(
        ["distance_to_target_boundary_m", "pulse_index"], kind="mergesort",
    ).iloc[0]

    cases_to_build = [
        ("Ground-target low residual", low_ground,
         "Minimum absolute C7 residual among common-support independent-test road pulses; ties use lowest pulse_index."),
        ("Ground-target large residual", high_ground,
         "Maximum absolute C7 residual among all common-support independent-test road pulses; no tail filtering; ties use lowest pulse_index."),
        ("Non-ground-context large difference", large_context,
         "Maximum absolute native Refh-minus-original-local-ground difference among supported non-ground-context pulses; ties use lowest pulse_index."),
        ("Uncertain road-boundary case", mixed_context,
         "Supported uncertain road-boundary pulse nearest the fixed target-interior boundary; ties use lowest pulse_index."),
    ]
    sources = _load_naip_sources()
    selected: list[dict[str, Any]] = []
    for case_label, chosen, selection_rule in cases_to_build:
        granule_id = str(chosen.granule_id)
        pulse_index = int(chosen.pulse_index)
        lon, lat, refh = float(chosen.longitude_deg), float(chosen.latitude_deg), float(chosen.casals_refh_m)
        frame = candidate_by_granule.get(granule_id)
        if frame is not None and pulse_index in frame.index:
            candidate = frame.loc[pulse_index]
            c7_x, c7_y = float(candidate.c7_easting_m_epsg6347), float(candidate.c7_northing_m_epsg6347)
            c7_height = float(candidate.C7_height_m)
            c7_ground = float(candidate.c7_reference_height_local_plane_m)
            c7_acceptable = bool(candidate.c7_surface_acceptable)
            c7_neighbor_count = int(candidate.c7_neighbor_count)
            c7_difference = float(c7_height - c7_ground)
            c8_difference = float(
                test_residuals.loc[
                    test_residuals.granule_id.astype(str).eq(granule_id)
                    & (test_residuals.pulse_index == pulse_index), "C8_residual_m",
                ].iloc[0]
            )
            c7_common = bool(candidate.common_support_all_candidates)
        else:
            c7_x, c7_y, c7_z, _ = datum.transform_c7([lon], [lat], [refh])
            c7_x, c7_y, c7_height = float(c7_x[0]), float(c7_y[0]), float(c7_z[0])
            ground_support = interpolate_ground_surface(
                np.asarray([[c7_x, c7_y]], dtype=np.float64), *training_surfaces[granule_id],
            )
            c7_ground = float(ground_support["reference_height_plane_m"][0])
            c7_acceptable = bool(ground_support["surface_acceptable"][0])
            c7_neighbor_count = int(ground_support["neighbor_count"][0])
            c7_difference = c7_height - c7_ground if c7_acceptable else np.nan
            offset = empirical_audit["granules"][granule_id]["calibration_offset_m"]
            c8_difference = (
                refh - float(offset) - float(chosen.reference_height_local_plane_m)
                if bool(chosen.original_surface_acceptable) and offset is not None else np.nan
            )
            c7_common = False
        training_xy, training_z = training_surfaces[granule_id]
        distances = np.linalg.norm(training_xy - np.asarray([c7_x, c7_y]), axis=1)
        support = distances <= 3.0
        waveform_path = h5_by_granule[granule_id]
        with h5py.File(waveform_path, "r") as h5:
            waveform = np.asarray(h5["rx_waveform"][pulse_index], dtype=np.float64)
            bins = np.asarray(h5["rx_bins"][:], dtype=np.float64) if h5["rx_bins"].ndim == 1 else np.arange(waveform.size, dtype=np.float64)
            background = float(np.mean(waveform[2600:2728]))
            snr = float(h5["refh_snr"][pulse_index])
            h5_refh = float(h5["refh"][pulse_index])
            if not np.isclose(h5_refh, refh, rtol=0.0, atol=1e-9):
                raise RuntimeError(f"Case pulse Refh does not match H5 source at {granule_id} pulse {pulse_index}.")
            sweep, track = int(h5["sweep_num"][pulse_index]), int(h5["track_num"][pulse_index])
        image_path = sources[granule_id]["image"]
        metadata_path = sources[granule_id]["metadata"]
        with rasterio.open(image_path) as raster:
            if raster.crs is None or not np.isclose(raster.transform.b, 0.0) or not np.isclose(raster.transform.d, 0.0) or raster.transform.a <= 0 or raster.transform.e >= 0:
                raise ValueError(f"NAIP raster affine/CRS check failed for {granule_id}.")
            transformer = Transformer.from_crs("EPSG:4326", raster.crs, always_xy=True)
            image_x, image_y = transformer.transform(lon, lat)
        raw_difference = float(refh - chosen.reference_height_local_plane_m) if bool(chosen.original_surface_acceptable) else np.nan
        selected.append({
            "case_label": case_label, "selection_rule": selection_rule,
            "granule_id": granule_id, "roi_id": str(chosen.roi_id), "site_id": str(chosen.site_id),
            "target_context": str(chosen.ground_context_status), "landcover_class": str(chosen.landcover_class),
            "pulse_index": pulse_index, "track_num": track, "sweep_num": sweep,
            "snr": snr, "refh_m": refh,
            "original_surface_acceptable": bool(chosen.original_surface_acceptable),
            "reference_ground_height_m": float(chosen.reference_height_local_plane_m) if bool(chosen.original_surface_acceptable) else np.nan,
            "refh_minus_ground_m": raw_difference,
            "c7_common_support": c7_common, "c7_surface_acceptable": c7_acceptable,
            "c7_neighbor_count": c7_neighbor_count, "C7_residual_m": c7_difference,
            "C8_residual_m": c8_difference,
            "c7_height_m": c7_height, "c7_ground_height_m": c7_ground,
            "c7_easting_m_epsg6347": c7_x, "c7_northing_m_epsg6347": c7_y,
            "class2_grid_support_count_within_3m": int(support.sum()),
            "image_path": image_path, "image_metadata_path": metadata_path,
            "image_x": float(image_x), "image_y": float(image_y),
            "waveform": waveform, "rx_bins": bins, "background_mean": background,
            "support_points_xy": training_xy[support], "support_points_height_m": training_z[support],
        })
    return selected


def _reference_validation(
    held_out: pd.DataFrame,
    training_grid_xy: np.ndarray,
    training_grid_z: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from research.elevation_comparison import interpolate_ground_surface

    if held_out.empty:
        return pd.DataFrame(), {"n_held_out_class2_points": 0, "status": "NO_VALIDATION_POINTS_NEAR_CONTEXT"}
    query_xy = held_out[["easting_m_epsg6347", "northing_m_epsg6347"]].to_numpy(dtype=np.float64)
    surface = interpolate_ground_surface(query_xy, training_grid_xy, training_grid_z)
    result = held_out.copy()
    result["plane_height_m"] = surface["reference_height_plane_m"]
    result["median_height_m"] = surface["reference_height_median_m"]
    result["neighbor_count"] = surface["neighbor_count"]
    result["nearest_training_ground_distance_m"] = surface["nearest_ground_distance_m"]
    result["plane_fit_valid"] = surface["plane_fit_valid"]
    result["surface_acceptable"] = surface["surface_acceptable"]
    result["plane_prediction_error_m"] = result.plane_height_m - result.elevation_m
    result["median_prediction_error_m"] = result.median_height_m - result.elevation_m
    accepted = result.loc[result.surface_acceptable.astype(bool)]
    audit = {
        "n_held_out_class2_points_near_queries": int(len(result)),
        "n_plane_fit_valid": int(result.plane_fit_valid.sum()),
        "n_surface_acceptable": int(result.surface_acceptable.sum()),
        "plane_all_fit_metrics_m": _safe_metrics(result.plane_prediction_error_m.to_numpy()),
        "plane_acceptable_metrics_m": _safe_metrics(accepted.plane_prediction_error_m.to_numpy()),
        "local_median_acceptable_metrics_m": _safe_metrics(accepted.median_prediction_error_m.to_numpy()),
        "method": "Real LAZ class-2 records held out by LAS record index modulo 10000; queried against the training-only 0.75 m cell grid and same robust local-plane/median methods.",
    }
    return result, audit


def _make_views(
    features: Sequence[Mapping[str, Any]],
    geometry_evidence: Mapping[str, Mapping[str, Any]],
    population: pd.DataFrame,
    split_manifest: pd.DataFrame,
    candidate_frames: Sequence[pd.DataFrame],
    naip_sources: Mapping[str, Mapping[str, Path]],
    h5_dates: Mapping[str, str],
) -> list[dict[str, Any]]:
    views = []
    for granule_id in sorted(h5_dates):
        ground_rows = population.loc[
            (population.granule_id.astype(str) == granule_id)
            & population.is_ground_target_eligible.astype(bool)
        ].copy()
        context_rows = population.loc[population.granule_id.astype(str) == granule_id].copy()
        roles = split_manifest.loc[
            split_manifest.granule_id.astype(str) == granule_id,
            ["granule_id", "pulse_index", "research_role", "axis_distance_m"],
        ]
        ground_rows = ground_rows.merge(
            roles, on=["granule_id", "pulse_index"], how="left", validate="one_to_one",
        )
        for frame in candidate_frames:
            if str(frame.granule_id.iloc[0]) != granule_id:
                continue
            extras = [c for c in ["pulse_index", "C0_height_m", "C7_height_m",
                                  "C8_height_m", "c7_reference_height_local_plane_m", "c7_surface_acceptable",
                                  "common_support_all_candidates", "c7_local_slope_deg", "c7_ground_roughness_nmad_m",
                                  "c7_neighbor_count"] if c in frame.columns]
            ground_rows = ground_rows.merge(frame[["granule_id", *extras]], on=["granule_id", "pulse_index"], how="left", validate="one_to_one")
        feature_items = []
        for feature in features:
            props = feature["properties"]
            if str(props["granule_id"]) != granule_id:
                continue
            roi_id = str(props["roi_id"])
            item = geometry_evidence[roi_id]
            polygon = item["polygon"]
            interior = item["interior"]
            feature_items.append({
                "roi_id": roi_id, "landcover_class": str(props.get("landcover_class", props.get("class_label", "unknown"))),
                "ground_context_status": str(props["ground_context_status"]),
                "polygon_xy": np.asarray(polygon.exterior.coords, dtype=np.float64),
                "interior_xy": np.asarray(interior.exterior.coords, dtype=np.float64),
            })
        sources = naip_sources[granule_id]
        image_meta = json.loads(sources["metadata"].read_text(encoding="utf-8"))
        views.append({
            "granule_id": granule_id, "h5_date": h5_dates[granule_id],
            "naip_date": str(image_meta["acquisition_dates"][0]),
            "image_path": sources["image"], "image_metadata_path": sources["metadata"],
            "features": feature_items, "population": context_rows,
            "ground_rows": ground_rows,
        })
    return views


def _write_artifact_manifest(output_dir: Path) -> pd.DataFrame:
    rows = []
    manifest_path = output_dir / "ground_aware_artifact_manifest.csv"
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path != manifest_path:
            rows.append({
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            })
    result = pd.DataFrame(rows)
    result.to_csv(manifest_path, index=False)
    return result


def run_ground_aware_evaluation(
    output_dir: str | Path = GROUND_OUTPUT_ROOT,
    *,
    make_figures: bool = True,
) -> dict[str, Any]:
    """Re-evaluate likely-ground road targets and retain descriptive context."""
    import h5py
    import rasterio
    from pyproj import datadir, network

    from research.validation_figures import (
        plot_ground_candidate_comparison,
        plot_ground_elevation_profile,
        plot_ground_residual_distributions,
        plot_ground_scene_overview,
        plot_ground_vs_nonground_cases,
        plot_reference_ground_context,
    )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_paths = {spec.granule_id: spec.h5_path for spec in datum.GRANULES}
    naip_sources = _load_naip_sources()
    if set(naip_sources) != set(h5_paths):
        raise ValueError("The preserved NAIP input manifest does not contain both CASALS granules.")
    with ROI_PATH.open("r", encoding="utf-8") as stream:
        roi_manifest = json.load(stream)
    h5_dates, h5_counts = {}, {}
    for granule_id, h5_path in h5_paths.items():
        with h5py.File(h5_path, "r") as h5:
            h5_dates[granule_id] = str(h5.attrs["start_utca"])[:10]
            h5_counts[granule_id] = int(h5["refh"].shape[0])
    copied_features, evidence = build_ground_context_evidence(roi_manifest, naip_sources, h5_dates)
    geometries = evidence["geometries"]
    pd.DataFrame(evidence["audit"]).to_csv(output_dir / "ground_context_manifest.csv", index=False)
    evidence_collection = {"type": "FeatureCollection", "features": copied_features, "metadata": {
        "source_roi_manifest": str(ROI_PATH.relative_to(ROOT)),
        "context_method_version": GROUND_CONTEXT_VERSION,
        "return_origin_status": "unresolved_for_CASALS_Refh",
        "ground_target_note": "Road-core polygons identify likely-ground target areas only; no pulse is independently labeled as a ground return.",
    }}
    _json_write(output_dir / "ground_context_manifest.geojson", evidence_collection)

    matched_parts = []
    matched_counts = {}
    for spec in datum.GRANULES:
        path = spec.revised_dir / "matched_casals_3dep_points.parquet"
        matched = pd.read_parquet(path, columns=_PARQUET_COLUMNS)
        matched["granule_id"] = spec.granule_id
        matched_counts[spec.granule_id] = int(len(matched))
        matched_parts.append(build_reviewed_pulse_population(matched, geometries))
    population = pd.concat(matched_parts, ignore_index=True)
    population.to_parquet(output_dir / "reviewed_context_pulse_population.parquet", index=False)
    context_summary = (
        population.groupby(["granule_id", "roi_id", "site_id", "landcover_class", "ground_context_status"], sort=True)
        .agg(n_context_pulses=("pulse_index", "size"),
             n_geometry_ground_target=("is_ground_target_eligible", "sum"),
             n_original_surface_acceptable=("original_surface_acceptable", "sum"),
             median_refh_snr=("refh_snr", "median"))
        .reset_index()
    )
    context_summary.to_csv(output_dir / "ground_context_population_summary.csv", index=False)

    historical_context_rows = []
    historical_case_sources = {}
    context_identity = population[[
        "granule_id", "pulse_index", "ground_context_status", "roi_id", "landcover_class", "return_origin_status",
    ]]
    for spec in datum.GRANULES:
        case_path = LEGACY_VISUAL_ROOT / f"{spec.granule_id}_all_gross_errors_snr_ge_3.csv"
        if not case_path.is_file():
            continue
        cases = pd.read_csv(case_path)
        if "pulse_index" not in cases:
            raise ValueError(f"The preserved large-difference case table lacks pulse_index: {case_path}.")
        cases["granule_id"] = spec.granule_id
        cases = cases.merge(context_identity, on=["granule_id", "pulse_index"], how="left", validate="one_to_one")
        cases["reviewed_context_status"] = cases.ground_context_status.fillna("outside_reviewed_context")
        cases["historical_case_source"] = str(case_path.relative_to(ROOT))
        historical_context_rows.append(cases)
        historical_case_sources[spec.granule_id] = {
            "path": str(case_path.relative_to(ROOT)), "sha256": _sha256(case_path), "rows": int(len(cases)),
        }
    historical_context_crosswalk = (
        pd.concat(historical_context_rows, ignore_index=True)
        if historical_context_rows else pd.DataFrame()
    )
    historical_context_crosswalk.to_csv(output_dir / "historical_large_difference_context_crosswalk.csv", index=False)
    historical_context_counts = {
        f"{granule}::{status}": int(count)
        for (granule, status), count in historical_context_crosswalk.groupby(
            ["granule_id", "reviewed_context_status"], dropna=False, sort=True,
        ).size().items()
    } if not historical_context_crosswalk.empty else {}

    # Freeze all geometry-only road roles before any C7 transform or residual calculation.
    split_manifest, split_audit = freeze_ground_spatial_split(population, geometries, output_dir)

    saved_config_path = LEGACY_DATUM_ROOT / "selected_vertical_configuration.json"
    if not saved_config_path.is_file():
        raise FileNotFoundError(f"The already frozen C7 configuration is missing: {saved_config_path}.")
    saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
    if saved_config.get("candidate_id") != "C7" or saved_config.get("status") != "PROVISIONAL_NOT_DATUM_VERIFIED":
        raise ValueError("The preserved C7 configuration is not the expected provisional selection.")
    proj_dir = LEGACY_DATUM_ROOT / "proj_data"
    if not (proj_dir / "proj.db").is_file() or not (proj_dir / datum.GEOID18_GRID).is_file():
        raise FileNotFoundError("The existing private PROJ database/GEOID18 grid is required and is read-only for this run.")
    datadir.set_data_dir(str(proj_dir))
    network.set_network_enabled(False)

    candidate_frames = []
    training_surfaces = {}
    interpolation_rows = []
    interpolation_audits = {}
    query_audits = {}
    for spec in datum.GRANULES:
        frame = _load_ground_candidate_frame(spec.granule_id, population, split_manifest, spec.h5_path)
        validation_context = population.loc[population.granule_id.astype(str) == spec.granule_id]
        validation_query_xy = validation_context[["easting_m_epsg6347", "northing_m_epsg6347"]].to_numpy(dtype=np.float64)
        queried, query_audit, held_out, training_xy, training_z = datum._surface_query(
            spec, frame, return_validation_points=True, validation_query_xy=validation_query_xy,
        )
        candidate_frames.append(queried)
        training_surfaces[spec.granule_id] = (training_xy, training_z)
        query_audits[spec.granule_id] = query_audit
        validation, validation_audit = _reference_validation(held_out, training_xy, training_z)
        if not validation.empty:
            validation.insert(0, "granule_id", spec.granule_id)
            interpolation_rows.append(validation)
        interpolation_audits[spec.granule_id] = validation_audit

    candidate_metrics_rows = []
    for frame in candidate_frames:
        candidate_metrics_rows.extend(_ground_metric_rows(frame, roles=("candidate_selection", "independent_test")))
    candidate_metrics = pd.DataFrame(candidate_metrics_rows)
    candidate_metrics.to_csv(output_dir / "ground_conditional_candidate_metrics.csv", index=False)
    test_metrics, test_residuals, empirical_audit = _ground_test_metrics(candidate_frames)
    test_metrics.to_csv(output_dir / "ground_conditional_test_metrics.csv", index=False)
    test_residuals.to_parquet(output_dir / "ground_target_test_population.parquet", index=False)
    highlight = test_residuals.loc[
        test_residuals.common_support_all_candidates.astype(bool)
        & (test_residuals.C7_residual_m.abs() > 2.0)
    ].copy()
    highlight["display_highlight_threshold_m"] = 2.0
    highlight["threshold_is_display_only"] = True
    highlight_columns = [
        "granule_id", "pulse_index", "track_num", "sweep_num", "research_role", "refh_snr",
        "C7_residual_m", "reference_height_local_plane_m", "c7_reference_height_local_plane_m",
        "neighbor_count_within_3m", "local_slope_deg", "ground_roughness_nmad_m",
        "c7_neighbor_count", "c7_local_slope_deg", "c7_ground_roughness_nmad_m",
        "surface_acceptable", "c7_surface_acceptable", "display_highlight_threshold_m", "threshold_is_display_only",
    ]
    highlight[[column for column in highlight_columns if column in highlight]].to_csv(
        output_dir / "large_ground_target_residuals_for_review.csv", index=False,
    )
    empirical_audit["existing_c7_status"] = saved_config["status"]
    empirical_audit["existing_c7_configuration_sha256"] = _sha256(saved_config_path)
    _json_write(output_dir / "ground_empirical_alignment_audit.json", empirical_audit)

    common_frames = []
    for frame in candidate_frames:
        common_frames.append(frame.loc[frame.common_support_all_candidates.astype(bool)].copy())
    common_sample = pd.concat(common_frames, ignore_index=True)
    common_sample.to_parquet(output_dir / "ground_c0_c7_common_sample.parquet", index=False)
    descriptive_population, descriptive_query_audit = _query_c7_descriptive_context(
        population, training_surfaces, empirical_audit,
    )
    descriptive_population.to_parquet(
        output_dir / "descriptive_reference_return_to_ground_population.parquet", index=False,
    )
    descriptive = _descriptive_reference_metrics(descriptive_population)
    descriptive.to_csv(output_dir / "descriptive_reference_return_to_ground_metrics.csv", index=False)
    _json_write(output_dir / "descriptive_context_c7_query_audit.json", descriptive_query_audit)

    if interpolation_rows:
        interpolation = pd.concat(interpolation_rows, ignore_index=True)
    else:
        interpolation = pd.DataFrame()
    interpolation.to_csv(output_dir / "independent_class2_interpolation_check.csv", index=False)

    cases = _select_waveform_cases(
        population, geometries, candidate_frames, h5_paths, test_residuals, training_surfaces, empirical_audit,
    )
    case_rows = []
    for case in cases:
        case_rows.append({
            k: v for k, v in case.items()
            if k not in {"waveform", "rx_bins", "support_points_xy", "support_points_height_m"}
        })
    pd.DataFrame(case_rows).to_csv(output_dir / "ground_context_waveform_case_manifest.csv", index=False)

    views = _make_views(copied_features, geometries, population, split_manifest, candidate_frames, naip_sources, h5_dates)
    figure_paths = []
    if make_figures:
        fig_dir = output_dir / "figures"
        figure_paths = [
            plot_ground_scene_overview(views, fig_dir / "ground_scene_overview.png"),
            plot_reference_ground_context(views, fig_dir / "reference_ground_context.png"),
            plot_ground_candidate_comparison(candidate_metrics, test_metrics, fig_dir / "ground_candidate_comparison.png"),
            plot_ground_residual_distributions(test_residuals, fig_dir / "ground_residual_distributions.png"),
            plot_ground_vs_nonground_cases(cases, fig_dir / "ground_vs_nonground_cases.png"),
            plot_ground_elevation_profile(test_residuals, fig_dir / "ground_elevation_profile.png"),
        ]

    source_manifest = {"inputs": {}}
    preserved_identity_path = LEGACY_VISUAL_ROOT / "input_manifest.json"
    if preserved_identity_path.is_file():
        source_manifest["preserved_notebook10_input_manifest_sha256"] = _sha256(preserved_identity_path)
        preserved = json.loads(preserved_identity_path.read_text(encoding="utf-8"))
        source_manifest["verified_source_hashes_from_preserved_manifest"] = preserved.get("inputs", {})
    for spec in datum.GRANULES:
        source_manifest["inputs"][spec.granule_id] = {
            "h5_path": str(spec.h5_path.relative_to(ROOT)), "h5_size_bytes": int(spec.h5_path.stat().st_size),
            "h5_sha256": _sha256(spec.h5_path),
            "laz_path": str(spec.laz_path.relative_to(ROOT)), "laz_size_bytes": int(spec.laz_path.stat().st_size),
            "laz_sha256": _sha256(spec.laz_path),
            "naip_image_path": str(naip_sources[spec.granule_id]["image"].relative_to(ROOT)),
            "naip_image_size_bytes": int(naip_sources[spec.granule_id]["image"].stat().st_size),
            "naip_image_sha256": _sha256(naip_sources[spec.granule_id]["image"]),
            "naip_metadata_path": str(naip_sources[spec.granule_id]["metadata"].relative_to(ROOT)),
            "naip_metadata_sha256": _sha256(naip_sources[spec.granule_id]["metadata"]),
            "h5_pulse_count": h5_counts[spec.granule_id],
            "matched_points_available": matched_counts[spec.granule_id],
            "c7_transform_and_query_audit": query_audits[spec.granule_id],
            "class2_interpolation_validation": interpolation_audits[spec.granule_id],
        }
    source_manifest["roi_manifest"] = {
        "path": str(ROI_PATH.relative_to(ROOT)), "sha256": _sha256(ROI_PATH),
        "naip_asset_manifest_sha256": _sha256(NAIP_MANIFEST_PATH),
    }
    source_manifest["c7_configuration"] = {
        "path": str(saved_config_path.relative_to(ROOT)), "sha256": _sha256(saved_config_path),
        "candidate_id": saved_config["candidate_id"], "status": saved_config["status"],
        "proj_db_sha256": _sha256(proj_dir / "proj.db"),
        "geoid18_grid_sha256": _sha256(proj_dir / datum.GEOID18_GRID),
    }
    source_manifest["split"] = split_audit
    source_manifest["descriptive_context_query"] = descriptive_query_audit
    source_manifest["preserved_historical_large_difference_case_tables"] = historical_case_sources
    source_manifest["selection_and_membership_exclusions"] = [
        "Only road-core points inside fixed inward buffers are ground_target_eligible; this denotes likely-ground target geography, not a ground return label.",
        "Road boundary points inside the original polygon but outside its fixed inward buffer are uncertain and excluded from primary metrics.",
        "Open fields are uncertain because the NAIP scenes precede the November 2024 observations by 529 and 887 days; crop/vegetation state is unknown.",
        "Canopy pulses are retained as non_ground_context and their Refh-minus-ground differences are descriptive only.",
        "Water is uncertain; the November 18 water ROI has no nominal pulse centers.",
        "No eligibility, role assignment, or case selection uses Refh residual, candidate RMSE, SNR, or waveform peak shape.",
        "Candidate metrics require common C0-C7 finite heights and acceptable original and C7-requeried class-2 surface support; no residual tail is filtered.",
    ]
    _json_write(output_dir / "ground_aware_provenance.json", source_manifest)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "context_population_rows": int(len(population)),
        "context_population_by_status": {str(k): int(v) for k, v in population.ground_context_status.value_counts().items()},
        "historical_large_difference_context_counts": historical_context_counts,
        "ground_target_eligible_pulses": int(population.is_ground_target_eligible.sum()),
        "h5_pulse_counts": h5_counts,
        "matched_point_counts": matched_counts,
        "split_role_counts": split_audit["role_counts"],
        "c0_c7_common_sample_rows": int(len(common_sample)),
        "conditional_test_metric_rows": int(len(test_metrics)),
        "waveform_case_count": int(len(cases)),
        "descriptive_context_metric_rows": int(len(descriptive)),
        "descriptive_context_c7_acceptable_rows": int(descriptive_population.c7_surface_acceptable.astype(bool).sum()),
        "c7_status": saved_config["status"],
        "strict_absolute_elevation_error_available": False,
        "primary_analysis": "Conditional candidate-to-3DEP agreement inside reviewed road-core target interiors with common C0-C7 support; per-pulse Refh ground-return origin remains unresolved.",
        "secondary_analysis": "Descriptive Refh-minus-local-ground distributions by reviewed context; canopy and uncertain polygons are not ground-error populations.",
        "figure_files": [path.relative_to(output_dir).as_posix() for path in figure_paths],
    }
    _json_write(output_dir / "ground_aware_summary.json", summary)
    artifacts = _write_artifact_manifest(output_dir)
    summary["artifact_count"] = int(len(artifacts))
    _json_write(output_dir / "ground_aware_summary.json", summary)
    artifacts = _write_artifact_manifest(output_dir)
    return {
        "summary": summary, "population": population, "context_summary": context_summary,
        "split_manifest": split_manifest, "split_audit": split_audit,
        "candidate_metrics": candidate_metrics, "test_metrics": test_metrics,
        "test_residuals": test_residuals, "common_sample": common_sample,
        "descriptive_metrics": descriptive, "interpolation_audits": interpolation_audits,
        "descriptive_population": descriptive_population,
        "cases": case_rows, "artifact_manifest": artifacts,
    }

