from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from research.geolocation.bin_georeferencing_audit.run_audit import inventory
from research.reference.refh_correction_audit import (
    VALID_APPLIED_STATUS,
    VALID_CANDIDATE_STATUS,
    add_sensitivity,
    bilinear_grid,
    candidate_evidence_status,
    collect_values,
    compare_common_pulses,
    distribution,
    field_inventory,
    geoid_model_comparison,
    height_sensitivity,
    residual_metrics,
    select_independent_common_test,
    tide_free_to_mean_formula,
)


def test_h5_correction_field_extraction_and_declared_fill_values(tmp_path):
    path = tmp_path / "small.h5"
    with h5py.File(path, "w") as h5:
        ds = h5.create_dataset("refh_neutat_delay_total", data=[1.0, -999.0, 2.0])
        ds.attrs["_FillValue"] = -999.0
        ds.attrs["scale_factor"] = 2.0
        ds.attrs["add_offset"] = 1.0
        inv, paths = inventory(h5)
        values = collect_values(h5[paths["refh_neutat_delay_total"]], chunk_size=2)
        rows = field_inventory(inv, paths)
    np.testing.assert_allclose(values, [3.0, np.nan, 5.0], equal_nan=True)
    row = next(x for x in rows if x["field"] == "refh_neutat_delay_total")
    assert row["presence_status"] == "PRESENT"
    assert "_FillValue" in row["declared_missing_values"]


def test_missing_h5_fields_are_explicitly_not_present():
    rows = field_inventory({"objects": {}}, {})
    row = next(x for x in rows if x["field"] == "range_bias_correction")
    assert row["presence_status"] == "NOT_PRESENT"
    assert row["path"] == "NOT_PRESENT"


def test_units_and_applied_status_are_not_inferred_from_field_names():
    assert "UNKNOWN" in VALID_APPLIED_STATUS
    row = next(x for x in field_inventory({"objects": {}}, {})
               if x["field"] == "geoid")
    assert row["units_attribute"] == "UNKNOWN"
    assert candidate_evidence_status("UNKNOWN", True) == "PHYSICAL_SENSITIVITY_ONLY"


def test_sensitivity_does_not_mutate_refh_or_apply_corrections_automatically():
    refh = np.array([10.0, 20.0])
    before = refh.copy()
    adjusted = add_sensitivity(refh, np.array([0.5, 0.25]))
    np.testing.assert_array_equal(refh, before)
    np.testing.assert_allclose(adjusted, [10.5, 20.25])
    assert not np.shares_memory(adjusted, refh)


def test_positive_excess_range_removal_raises_height_for_downward_beam():
    change = height_sensitivity([2.0], [0.0], [0.0], [0.0], [np.pi / 2])
    assert change[0] == pytest.approx(2.0, abs=1e-12)


def test_geometry_sensitivity_uses_observed_beam_orientation():
    near_nadir = height_sensitivity([1.0], [-75.0], [39.0], [0.0], [np.pi / 2])
    oblique = height_sensitivity([1.0], [-75.0], [39.0], [0.0], [np.pi / 4])
    assert 0.99 <= near_nadir[0] <= 1.0
    assert oblique[0] == pytest.approx(np.sqrt(.5), abs=1e-12)


def test_synthetic_range_bias_moves_target_toward_sensor_and_raises_ellipsoid_height():
    to_xyz = Transformer.from_crs(4979, 4978, always_xy=True)
    to_geo = Transformer.from_crs(4978, 4979, always_xy=True)
    sensor = np.array(to_xyz.transform(0.0, 90.0, 1000.0))
    target = np.array(to_xyz.transform(0.0, 90.0, 0.0))
    direction = (target - sensor) / np.linalg.norm(target - sensor)
    measured_range = np.linalg.norm(target - sensor) + 2.0
    raw_target = sensor + measured_range * direction
    corrected = sensor + (measured_range - 2.0) * direction
    corrected_lon, corrected_lat, corrected_height = to_geo.transform(*corrected)
    _, _, raw_height = to_geo.transform(*raw_target)
    assert np.dot(corrected - raw_target, -direction) > 1.99
    assert corrected_height > raw_height
    assert corrected_height - raw_height == pytest.approx(2.0, abs=1e-3)


def test_bilinear_geoid_interpolation_uses_common_coordinates():
    grid = np.array([[0.0, 1.0], [10.0, 11.0]])
    # Grid rows start at 90N, columns at 0E, with one-degree spacing here.
    value = bilinear_grid(grid, [0.25], [89.75], spacing_deg=1.0)
    assert value[0] == pytest.approx(2.75)


def test_geoid_models_require_same_coordinates_and_report_pairwise_differences():
    result = geoid_model_comparison([1.0, 2.0], [1.1, 2.1], [0.5, 1.5])
    assert result["H5 minus EGM2008"]["median"] == pytest.approx(-0.1)
    assert result["H5 minus GEOID18"]["median"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="common coordinates"):
        geoid_model_comparison([1.0], [1.0, 2.0], [0.0])


def test_tide_formula_is_numerically_reproducible():
    latitude = np.array([38.4, 39.0])
    expected = 0.1287 - 0.3848 * np.sin(np.deg2rad(latitude)) ** 2
    np.testing.assert_allclose(tide_free_to_mean_formula(latitude), expected, atol=1e-12)


def test_no_arbitrary_bias_fit_is_included_in_sensitivity_application():
    baseline = np.array([-1.7, -1.5, -1.6])
    physical_delta = np.array([1.2, 1.3, 1.4])
    candidate = add_sensitivity(baseline, physical_delta)
    np.testing.assert_allclose(candidate, baseline + physical_delta)
    assert not np.allclose(candidate - baseline, np.full(3, np.mean(physical_delta)))


def test_test_population_selector_excludes_calibration_rows_without_mutation():
    frame = pd.DataFrame({
        "pulse_index": [3, 8, 11],
        "research_role": ["independent_test", "calibration_offset", "independent_test"],
        "common_support_all_candidates": [True, True, False],
    })
    original = frame.copy(deep=True)
    selected = select_independent_common_test(frame)
    assert selected.pulse_index.tolist() == [3]
    pd.testing.assert_frame_equal(frame, original)


def test_common_id_metrics_keep_all_candidate_scenarios_on_same_pulses():
    ids = np.array([10, 20, 30])
    result = compare_common_pulses(ids, {"C7": [-1, 0, 1], "H1": [0, 1, 2]})
    assert result.n.tolist() == [3, 3]
    with pytest.raises(ValueError, match="common pulse IDs"):
        compare_common_pulses(ids, {"C7": [-1, 0]})
    with pytest.raises(ValueError, match="unique"):
        compare_common_pulses([10, 10], {"C7": [-1, 0]})


def test_c7_baseline_and_ground_support_mask_are_immutable():
    c7 = np.array([-1.6, -1.5])
    support = np.array([True, True])
    c7_before, support_before = c7.copy(), support.copy()
    _ = add_sensitivity(c7, [0.4, 0.5])
    np.testing.assert_array_equal(c7, c7_before)
    np.testing.assert_array_equal(support, support_before)


def test_bias_rmse_nmad_and_count_are_correct():
    values = np.array([-1.0, 0.0, 1.0])
    metrics = residual_metrics(values)
    assert metrics["n"] == 3
    assert metrics["mean_bias_m"] == pytest.approx(0.0)
    assert metrics["median_bias_m"] == pytest.approx(0.0)
    assert metrics["RMSE_m"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["NMAD_m"] == pytest.approx(1.4826)
    assert metrics["absolute_P95_m"] == pytest.approx(1.0)
    assert metrics["cRMSE_m"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["median_absolute_residual_m"] == pytest.approx(1.0)
    assert metrics["MAE_m"] == pytest.approx(2 / 3)
    assert metrics["residual_P05_m"] == pytest.approx(-0.9)
    assert metrics["residual_P95_m"] == pytest.approx(0.9)
    assert metrics["bias_squared_fraction_of_MSE"] + metrics["dispersion_squared_fraction_of_MSE"] == pytest.approx(1.0)
    assert distribution([1, 2, np.nan])["valid_fraction"] == pytest.approx(2 / 3)


def test_centered_rmse_uses_population_denominator_and_decomposes_rmse():
    values = np.array([-1.9, -1.8, -1.6, -1.5])
    metrics = residual_metrics(values)
    assert metrics["cRMSE_m"] == pytest.approx(np.std(values, ddof=0))
    assert metrics["RMSE_m"] ** 2 == pytest.approx(
        metrics["mean_bias_m"] ** 2 + metrics["cRMSE_m"] ** 2, abs=1e-14
    )


def test_constant_offset_preserves_centered_and_robust_dispersion():
    values = np.array([-0.21, -0.10, -0.03, 0.02, 0.08, 0.31])
    baseline = residual_metrics(values)
    shifted = residual_metrics(values + 4.75)
    assert shifted["cRMSE_m"] == pytest.approx(baseline["cRMSE_m"])
    assert shifted["NMAD_m"] == pytest.approx(baseline["NMAD_m"])
    assert shifted["RMSE_m"] != pytest.approx(baseline["RMSE_m"])


def test_nonconstant_sensitivity_can_change_centered_dispersion():
    baseline_values = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    varied_candidate = add_sensitivity(baseline_values, [-0.2, -0.1, 0.0, 0.1, 0.2])
    assert residual_metrics(varied_candidate)["cRMSE_m"] > residual_metrics(baseline_values)["cRMSE_m"]


def test_single_outlier_increases_classical_dispersion_more_than_nmad():
    rng = np.random.default_rng(510)
    baseline = -1.6 + rng.normal(0.0, 0.1, size=1000)
    outlier = baseline.copy()
    outlier[-1] = 1.5
    base_metrics = residual_metrics(baseline)
    outlier_metrics = residual_metrics(outlier)
    assert outlier_metrics["cRMSE_m"] - base_metrics["cRMSE_m"] > (
        outlier_metrics["NMAD_m"] - base_metrics["NMAD_m"]
    )


def test_nonfinite_values_are_counted_and_zero_nmad_ratio_is_not_reported():
    metrics = residual_metrics([1.0, np.nan, np.inf, -np.inf])
    assert metrics["n"] == 1
    assert metrics["n_input"] == 4
    assert metrics["n_nonfinite"] == 3
    assert np.isnan(residual_metrics([2.0, 2.0, 2.0])["cRMSE_NMAD_ratio"])
    empty = residual_metrics([])
    assert empty["n"] == 0 and empty["n_input"] == 0


def test_minus_1p6_bias_with_point_1m_dispersion_does_not_certify_a_physical_correction():
    rng = np.random.default_rng(20261013)
    residual = -1.6 + rng.normal(0.0, 0.1, size=20_000)
    metrics = residual_metrics(residual)
    centered_as_description = residual - metrics["mean_bias_m"]
    centered_metrics = residual_metrics(centered_as_description)
    assert metrics["mean_bias_m"] == pytest.approx(-1.6, abs=0.002)
    assert metrics["RMSE_m"] == pytest.approx(1.603, abs=0.003)
    assert metrics["cRMSE_m"] == pytest.approx(0.1, abs=0.002)
    assert metrics["NMAD_m"] == pytest.approx(0.1, abs=0.003)
    assert centered_metrics["RMSE_m"] < metrics["RMSE_m"]
    assert centered_metrics["RMSE_m"] == pytest.approx(metrics["cRMSE_m"])


def test_evidence_status_is_closed_and_unknown_cannot_become_verified():
    assert candidate_evidence_status("CONFIRMED_NOT_APPLIED", True) == "VERIFIED"
    assert candidate_evidence_status("CONFIRMED_APPLIED", True) == "REJECTED"
    assert candidate_evidence_status("UNKNOWN", False) == "UNAVAILABLE"
    assert candidate_evidence_status("UNKNOWN", True) in VALID_CANDIDATE_STATUS
    assert candidate_evidence_status("UNKNOWN", True) != "VERIFIED"


def test_unknown_applied_status_stays_sensitivity_only_even_if_rmse_improves():
    baseline = np.array([-2.0, -1.8, -2.2])
    candidate = add_sensitivity(baseline, [1.9, 1.7, 2.1])
    assert residual_metrics(candidate)["RMSE_m"] < residual_metrics(baseline)["RMSE_m"]
    assert candidate_evidence_status("UNKNOWN", physical_basis=True) == "PHYSICAL_SENSITIVITY_ONLY"
