from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import box

from research import elevation_comparison
from research.reference import vertical_datum_comparison as datum
from research.reference.ground_evaluation import (
    _descriptive_reference_metrics,
    _query_c7_descriptive_context,
    build_reviewed_pulse_population,
    freeze_ground_spatial_split,
)


def _matched_rows() -> pd.DataFrame:
    rows = [
        (1, "g", 5.0, 5.0, 100.0, 100.0, "road"),
        (2, "g", 15.0, 5.0, 115.0, 100.0, "canopy"),
        (3, "g", 25.0, 5.0, 101.0, 100.0, "field"),
    ]
    return pd.DataFrame(rows, columns=[
        "pulse_index", "granule_id", "easting_m_epsg6347", "northing_m_epsg6347",
        "casals_refh_m", "reference_height_local_plane_m", "roi_kind",
    ]).assign(
        sweep_num=0, track_num=0, longitude_deg=-75.0, latitude_deg=35.0,
        refh_snr=4.0, reference_height_local_median_m=100.0,
        neighbor_count_within_3m=8, local_slope_deg=1.0,
        plane_residual_nmad_m=0.02, ground_roughness_nmad_m=0.02,
        surface_acceptable=True, surface_status="matched", spatial_split="outside_primary_open_field",
    )


def _geometry_evidence() -> dict[str, dict[str, object]]:
    return {
        "road": {
            "polygon": box(0, 0, 10, 10), "interior": box(1, 1, 9, 9),
            "properties": {"site_id": "road_site", "landcover_class": "road_surface",
                           "ground_context_status": "ground_target_eligible",
                           "ground_target_boundary_trim_m": 1.0,
                           "h5_acquisition_date_verified": "2024-11-12",
                           "naip_acquisition_date_verified": "2023-06-02",
                           "naip_age_days_at_h5": 529,
                           "ground_context_evidence": "reviewed road core"},
        },
        "canopy": {
            "polygon": box(10, 0, 20, 10), "interior": box(10, 0, 20, 10),
            "properties": {"site_id": "forest_site", "landcover_class": "tree_canopy",
                           "ground_context_status": "non_ground_context",
                           "ground_target_boundary_trim_m": 0.0,
                           "h5_acquisition_date_verified": "2024-11-12",
                           "naip_acquisition_date_verified": "2023-06-02",
                           "naip_age_days_at_h5": 529,
                           "ground_context_evidence": "canopy context"},
        },
        "field": {
            "polygon": box(20, 0, 30, 10), "interior": box(20, 0, 30, 10),
            "properties": {"site_id": "field_site", "landcover_class": "open_field",
                           "ground_context_status": "uncertain",
                           "ground_target_boundary_trim_m": 0.0,
                           "h5_acquisition_date_verified": "2024-11-12",
                           "naip_acquisition_date_verified": "2023-06-02",
                           "naip_age_days_at_h5": 529,
                           "ground_context_evidence": "seasonal state unknown"},
        },
    }


def test_context_classification_uses_polygon_evidence_not_height_or_snr() -> None:
    matched = _matched_rows()
    first = build_reviewed_pulse_population(matched, _geometry_evidence()).set_index("pulse_index")
    changed = matched.copy()
    changed["casals_refh_m"] += 15.0
    changed["refh_snr"] = -100.0
    second = build_reviewed_pulse_population(changed, _geometry_evidence()).set_index("pulse_index")

    assert first.loc[1, "ground_context_status"] == "ground_target_eligible"
    assert first.loc[1, "is_ground_target_eligible"]
    assert first.loc[2, "ground_context_status"] == "non_ground_context"
    assert not first.loc[2, "is_ground_target_eligible"]
    assert first.loc[3, "ground_context_status"] == "uncertain"
    assert not first.loc[3, "is_ground_target_eligible"]
    np.testing.assert_array_equal(first.is_ground_target_eligible, second.is_ground_target_eligible)
    np.testing.assert_array_equal(first.ground_context_status, second.ground_context_status)


def test_canopy_reference_return_difference_is_descriptive_not_ground_eligibility() -> None:
    population = build_reviewed_pulse_population(_matched_rows(), _geometry_evidence())
    metrics = _descriptive_reference_metrics(population).set_index("roi_id")

    assert metrics.loc["canopy", "median_m"] == 15.0
    assert metrics.loc["road", "median_m"] == 0.0
    assert metrics.loc["canopy", "interpretation"].startswith("descriptive Refh minus local ground")
    assert metrics.loc["canopy", "ground_context_status"] == "non_ground_context"


def test_c7_context_metrics_reuse_grid_and_keep_c0_c7_c8_definitions_separate(monkeypatch) -> None:
    population = build_reviewed_pulse_population(_matched_rows(), _geometry_evidence())

    def fake_transform(lon, lat, refh, *, return_audit=False):
        refh = np.asarray(refh, dtype=np.float64)
        audit = {"input_count": int(len(refh)), "finite_output_count": int(len(refh))}
        return np.arange(len(refh), dtype=float), np.zeros(len(refh)), refh + 10.0, audit

    def fake_interpolate(query_xy, ground_xy, ground_z):
        n = len(query_xy)
        accepted = np.array([True, False, True])[:n]
        return {
            "reference_height_plane_m": np.full(n, 95.0),
            "reference_height_median_m": np.full(n, 95.0),
            "neighbor_count": np.full(n, 8, dtype=np.int16),
            "nearest_ground_distance_m": np.full(n, 0.25),
            "local_slope_deg": np.full(n, 1.0),
            "ground_roughness_nmad_m": np.full(n, 0.02),
            "surface_acceptable": accepted,
            "plane_fit_valid": np.ones(n, dtype=bool),
            "surface_status": np.where(accepted, "acceptable", "rejected_test"),
        }

    monkeypatch.setattr(datum, "transform_c7", fake_transform)
    monkeypatch.setattr(elevation_comparison, "interpolate_ground_surface", fake_interpolate)
    surfaces = {"g": (np.array([[0.0, 0.0]]), np.array([100.0]))}
    empirical = {"granules": {"g": {"calibration_offset_m": 2.0, "calibration_common_n": 4}}}

    detailed, audit = _query_c7_descriptive_context(population, surfaces, empirical)
    metrics = _descriptive_reference_metrics(detailed).set_index("roi_id")
    by_pulse = detailed.set_index("pulse_index")

    assert by_pulse.loc[1, "C0_refh_minus_original_ground_m"] == 0.0
    assert by_pulse.loc[1, "C7_refh_minus_transformed_ground_m"] == 15.0
    assert by_pulse.loc[1, "C8_empirical_refh_minus_original_ground_m"] == -2.0
    assert np.isnan(by_pulse.loc[2, "C7_refh_minus_transformed_ground_m"])
    assert by_pulse.loc[2, "C8_empirical_refh_minus_original_ground_m"] == 13.0
    assert metrics.loc["canopy", "C7_n"] == 0
    assert metrics.loc["canopy", "C8_n"] == 1
    assert audit["training_surface_reused"]
    assert not audit["additional_laz_scan"]


def test_supplemental_roles_are_geometry_only_and_non_overlapping(tmp_path) -> None:
    polygon = box(0.0, 0.0, 300.0, 10.0)
    interior = polygon.buffer(-1.0)
    x = np.linspace(2.0, 298.0, 297)
    population = pd.DataFrame({
        "granule_id": "g", "pulse_index": np.arange(len(x)), "roi_id": "road",
        "site_id": "one_road", "is_ground_target_eligible": True,
        "easting_m_epsg6347": x, "northing_m_epsg6347": 5.0,
        "refh_snr": np.linspace(-5.0, 20.0, len(x)),
        "casals_refh_m": np.linspace(-100.0, 100.0, len(x)),
    })
    evidence = {"road": {"polygon": polygon, "interior": interior,
                         "properties": {"site_id": "one_road"}}}
    first, audit = freeze_ground_spatial_split(population, evidence, tmp_path)
    changed = population.assign(refh_snr=999.0, casals_refh_m=-999999.0)
    second, _ = freeze_ground_spatial_split(changed, evidence, tmp_path / "changed")

    assert audit["frozen_before_candidate_residuals"]
    assert not first.duplicated(["granule_id", "pulse_index"]).any()
    assert first.set_index("pulse_index").research_role.equals(second.set_index("pulse_index").research_role)
    assert {"calibration_offset", "candidate_selection", "independent_test", "guard_gap"}.issubset(set(first.research_role))
    assert set(first.loc[first.research_role == "guard_gap", "pulse_index"]).isdisjoint(
        set(first.loc[first.research_role == "independent_test", "pulse_index"])
    )

