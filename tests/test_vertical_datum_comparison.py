from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.reference.vertical_datum_comparison import (
    build_height_candidates,
    build_target_compound_crs,
    choose_available_non_ballpark_operation,
    common_support_mask,
    distribution_metrics,
    require_selected_operation_grids,
    selected_operation_grid_report,
    select_vertical_configuration,
    source_frame_evidence_status,
    split_calibration_geometry,
    validate_xyz_crs,
    xyz_round_trip_metrics,
    _freeze_selected_configuration,
)


def test_known_ellipsoid_geoid_height_recovers_known_navd88_height() -> None:
    known_navd88 = np.array([12.3, 4.7])
    geoid_undulation = np.array([-37.1, -35.8])
    ellipsoidal = known_navd88 + geoid_undulation

    candidates, warnings = build_height_candidates(ellipsoidal, geoid_undulation, np.zeros(2))

    assert not warnings
    np.testing.assert_allclose(candidates["C1"], known_navd88)
    assert np.all(candidates["C2"] < -60.0)


def test_all_h5_formula_sign_and_free_to_mean_combinations() -> None:
    refh = np.array([-20.0, 7.0])
    geoid = np.array([-36.0, -38.0])
    free = np.array([-0.02, 0.01])

    candidates, warnings = build_height_candidates(refh, geoid, free)

    assert not warnings
    np.testing.assert_array_equal(candidates["C0"], refh)
    np.testing.assert_array_equal(candidates["C1"], refh - geoid)
    np.testing.assert_array_equal(candidates["C2"], refh + geoid)
    np.testing.assert_array_equal(candidates["C3"], refh - geoid - free)
    np.testing.assert_array_equal(candidates["C4"], refh - geoid + free)
    np.testing.assert_array_equal(candidates["C5"], refh + geoid - free)
    np.testing.assert_array_equal(candidates["C6"], refh + geoid + free)


def test_missing_geoid_and_free_to_mean_remain_explicit() -> None:
    only_native, warning = build_height_candidates([1.0], None, None)
    assert set(only_native) == {"C0"}
    assert "geoid is missing" in warning[0]

    without_free, warning = build_height_candidates([1.0], [-2.0], None)
    assert set(without_free) == {"C0", "C1", "C2"}
    assert "geoid_free2mean is missing" in warning[0]


def test_nonfinite_corrections_propagate_and_leave_common_population() -> None:
    values, _ = build_height_candidates([2.0, 3.0], [np.nan, -5.0], [0.0, np.inf])
    assert np.isnan(values["C1"][0])
    assert not np.isfinite(values["C3"][1])
    common = common_support_mask(
        values,
        [True, True],
        [True, True],
        [1.0, 2.0],
        [1.0, 2.0],
    )
    np.testing.assert_array_equal(common, [False, False])


def test_formula_units_must_be_meters() -> None:
    with pytest.raises(ValueError, match="metres"):
        build_height_candidates([1.0], [2.0], [0.0], units="feet")


def test_candidate_arrays_must_have_same_pulse_order_length() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        build_height_candidates([1.0, 2.0], [0.0], [0.0, 0.0])


def test_common_support_uses_same_pulses_for_all_candidates_and_xy_surfaces() -> None:
    candidates = {"C0": np.array([1.0, 2.0, 3.0]), "C1": np.array([1.0, np.nan, 3.0])}
    # The second point loses C7 support because its transformed XY has no local ground.
    common = common_support_mask(
        candidates,
        [True, True, True],
        [True, False, True],
        [10.0, 11.0, 12.0],
        [10.1, np.nan, 12.1],
    )
    np.testing.assert_array_equal(common, [True, False, True])


def test_metrics_are_uncorrected_distribution_statistics() -> None:
    metrics = distribution_metrics(np.array([-1.0, 0.0, 1.0, 10.0]))
    assert metrics["n"] == 4
    assert metrics["bias_m"] == pytest.approx(2.5)
    assert metrics["median_m"] == pytest.approx(0.5)
    assert metrics["rmse_m"] == pytest.approx(np.sqrt(25.5))
    assert metrics["gross_error_rate_gt_2m"] == pytest.approx(0.25)


def test_geometry_split_is_stable_and_disjoint() -> None:
    input_frame = pd.DataFrame({
        "pulse_index": [10, 11, 12, 13, 14, 15],
        "easting_m_epsg6347": [5, 1, 3, 3, 6, 2],
        "northing_m_epsg6347": [0, 0, 2, 1, 0, 0],
    })
    a = split_calibration_geometry(input_frame)
    b = split_calibration_geometry(input_frame.sample(frac=1.0, random_state=99))

    assert a.groupby("research_role").size().to_dict() == {
        "calibration_offset": 3,
        "candidate_selection": 3,
    }
    assert set(a.loc[a.research_role == "calibration_offset", "pulse_index"]).isdisjoint(
        set(a.loc[a.research_role == "candidate_selection", "pulse_index"])
    )
    pd.testing.assert_frame_equal(a, b)


def test_xyz_crs_requires_full_3d_source_and_compound_target() -> None:
    from pyproj import CRS

    target = build_target_compound_crs()
    validate_xyz_crs(CRS.from_epsg(4979), target)
    with pytest.raises(ValueError, match="three axes"):
        validate_xyz_crs(CRS.from_epsg(4326), target)
    with pytest.raises(ValueError, match="compound"):
        validate_xyz_crs(CRS.from_epsg(4979), CRS.from_epsg(6347))


def test_unresolved_coordinate_epoch_keeps_source_assumed() -> None:
    assert source_frame_evidence_status(None, None) == "ASSUMED_SOURCE_FRAME"
    assert source_frame_evidence_status("WGS84 (G2139)", None) == "ASSUMED_SOURCE_FRAME"
    assert source_frame_evidence_status("NAD83(2011)", 2010.0) == "VERIFIED_SOURCE_FRAME"


def test_missing_required_grid_is_reported_and_not_accepted() -> None:
    grid = SimpleNamespace(short_name="geoid.tif", url="https://example.test/geoid.tif", available=False, open_license=True)
    operation = SimpleNamespace(name="vertical grid operation", grids=[grid])
    transformer = SimpleNamespace(operations=[operation])
    rows, available = selected_operation_grid_report(transformer)
    assert rows[0]["name"] == "geoid.tif"
    assert not available
    with pytest.raises(RuntimeError, match="unavailable required grids"):
        require_selected_operation_grids(transformer)


def test_available_non_ballpark_operation_can_be_used_when_best_alternative_missing() -> None:
    selected = object()
    group = SimpleNamespace(transformers=[selected], best_available=False)
    assert choose_available_non_ballpark_operation(group) is selected
    with pytest.raises(RuntimeError, match="No available"):
        choose_available_non_ballpark_operation(SimpleNamespace(transformers=[]))


def test_operation_selection_can_require_the_product_geoid_grid() -> None:
    geoid99 = SimpleNamespace(short_name="us_noaa_g1999u08.tif", available=True)
    geoid18 = SimpleNamespace(short_name="us_noaa_g2018u0.tif", available=True)
    ranked_old_model = SimpleNamespace(
        is_ballpark=False, operations=[SimpleNamespace(name="GEOID99 route", grids=[geoid99])]
    )
    product_model = SimpleNamespace(
        is_ballpark=False, operations=[SimpleNamespace(name="GEOID18 route", grids=[geoid18])]
    )
    group = SimpleNamespace(transformers=[ranked_old_model, product_model], best_available=True)

    assert choose_available_non_ballpark_operation(
        group, required_grid_name="us_noaa_g2018u0.tif"
    ) is product_model


def test_required_product_geoid_grid_must_be_available() -> None:
    other_grid = SimpleNamespace(short_name="other.tif", available=True)
    transformer = SimpleNamespace(
        is_ballpark=False, operations=[SimpleNamespace(name="other route", grids=[other_grid])]
    )
    with pytest.raises(RuntimeError, match="required grid"):
        choose_available_non_ballpark_operation(
            SimpleNamespace(transformers=[transformer]),
            required_grid_name="us_noaa_g2018u0.tif",
        )


def test_ballpark_operation_is_rejected() -> None:
    group = SimpleNamespace(transformers=[SimpleNamespace(is_ballpark=True)])
    with pytest.raises(RuntimeError, match="Ballpark"):
        choose_available_non_ballpark_operation(group)


def test_synthetic_xyz_round_trip_returns_known_geodetic_height() -> None:
    from pyproj import Transformer

    longitude, latitude, height = np.array([-75.5]), np.array([38.0]), np.array([-24.8])
    transformer = Transformer.from_crs(4979, 4978, always_xy=True)
    x, y, z = transformer.transform(longitude, latitude, height)
    metrics = xyz_round_trip_metrics(transformer, longitude, latitude, height, x, y, z)
    assert metrics["finite"]
    assert metrics["horizontal_error_m"]["max"] < 1e-6
    assert abs(metrics["vertical_error_m"]["max"]) < 1e-8


def test_selection_rejects_test_metrics_and_does_not_use_c8() -> None:
    metrics = pd.DataFrame({
        "candidate_id": ["C7", "C1", "C8"],
        "research_role": ["candidate_selection"] * 3,
        "rmse_m": [0.2, 0.1, 0.0001],
    })
    evidence = {
        "C7": {"evidence_status": "PROVISIONAL_NOT_DATUM_VERIFIED", "evidence_rank": 100,
               "provisional_eligible": True, "operation_complete": True, "source_frame_verified": False},
        "C1": {"evidence_status": "DIAGNOSTIC_ONLY", "evidence_rank": 3,
               "provisional_eligible": True, "operation_complete": True, "source_frame_verified": False},
    }
    chosen = select_vertical_configuration(metrics, evidence)
    assert chosen["candidate_id"] == "C7"
    assert chosen["test_metrics_used"] is False
    with pytest.raises(ValueError, match="forbidden populations"):
        select_vertical_configuration(
            pd.concat([metrics, pd.DataFrame([{"candidate_id": "C7", "research_role": "independent_test", "rmse_m": 0.01}])]),
            evidence,
        )


def test_verified_path_is_preferred_even_if_diagnostic_rmse_is_smaller() -> None:
    metrics = pd.DataFrame({
        "candidate_id": ["C1", "C7"], "research_role": ["candidate_selection"] * 2,
        "rmse_m": [0.01, 0.30],
    })
    evidence = {
        "C1": {"evidence_status": "DIAGNOSTIC_ONLY", "evidence_rank": 100,
               "provisional_eligible": True, "operation_complete": True, "source_frame_verified": False},
        "C7": {"evidence_status": "VERIFIED_GEODETIC", "evidence_rank": 1,
               "provisional_eligible": True, "operation_complete": True, "source_frame_verified": True},
    }
    assert select_vertical_configuration(metrics, evidence)["candidate_id"] == "C7"


def test_no_verified_path_yields_explicit_provisional_fallback() -> None:
    metrics = pd.DataFrame({"candidate_id": ["C1"], "research_role": ["candidate_selection"], "rmse_m": [0.4]})
    evidence = {"C1": {"evidence_status": "DIAGNOSTIC_ONLY", "evidence_rank": 3,
                       "provisional_eligible": True, "operation_complete": True, "source_frame_verified": False}}
    selected = select_vertical_configuration(metrics, evidence)
    assert selected["candidate_id"] == "C1"
    assert selected["evidence_status"] == "PROVISIONAL_NOT_DATUM_VERIFIED"


def test_c7_uses_its_transformed_xy_reference_plane() -> None:
    from research.reference.vertical_datum_comparison import _metric_rows_for_partition

    frame = pd.DataFrame({
        "granule_id": ["g", "g"], "research_role": ["candidate_selection"] * 2,
        "common_support_all_candidates": [True, True], "refh_snr": [4.0, 4.0],
        "C0_height_m": [10.0, 11.0], "C7_height_m": [10.0, 11.0],
        "reference_height_local_plane_m": [9.0, 10.0],
        "c7_reference_height_local_plane_m": [8.0, 9.0],
    })
    rows = _metric_rows_for_partition(frame, partition="candidate_selection", stage="test")
    c0 = next(row for row in rows if row["candidate_id"] == "C0" and row["snr_stratum"] == "all_valid_refh")
    c7 = next(row for row in rows if row["candidate_id"] == "C7" and row["snr_stratum"] == "all_valid_refh")
    assert c0["bias_m"] == pytest.approx(1.0)
    assert c7["bias_m"] == pytest.approx(2.0)


def test_frozen_configuration_records_provisional_status_and_no_test_usage(tmp_path: Path) -> None:
    selection = {
        "candidate_id": "C7", "candidate_label": "test", "evidence_status": "PROVISIONAL_NOT_DATUM_VERIFIED",
        "selection_reason": "evidence first", "selection_metrics": [],
    }
    evidence = {"C7": {"operation_complete": True}}
    result = _freeze_selected_configuration(selection, evidence, pd.DataFrame(),
                                            {"manifest_sha256": "abc"}, {},
                                            {"overall_status": "VERIFIED_SAMPLE_PASSTHROUGH"}, tmp_path)
    saved = pd.read_json(tmp_path / "selected_vertical_configuration.json", typ="series")
    assert result["candidate_id"] == "C7"
    assert result["test_metrics_used"] is False
    assert result["source_frame_status"] == "assumed_source_frame"
    assert saved["configuration_sha256"] == result["configuration_sha256"]

