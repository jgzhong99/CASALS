import numpy as np
import pandas as pd
import pytest

from research.reference.refh_correction_audit import add_sensitivity, residual_metrics
from research.reference.residual_diagnostics import (
    SCENARIOS,
    _high_influence_table,
    _make_bias_decomposition,
    _spatial_summary_rows,
    confidence_interval_status,
    pulse_id_sha256,
    stratum_masks,
)


def test_fixed_snr_masks_are_nested_and_keep_the_common_population():
    masks = stratum_masks([1.0, 3.0, 4.0, 5.0, np.nan])
    assert masks["all_common_support"].tolist() == [True] * 5
    assert masks["snr_ge_3"].tolist() == [False, True, True, True, False]
    assert masks["snr_ge_5"].tolist() == [False, False, False, True, False]
    assert np.all(masks["snr_ge_5"] <= masks["snr_ge_3"])


def test_pulse_id_hash_is_stable_and_rejects_duplicate_ids():
    assert pulse_id_sha256([2, 5, 9]) == pulse_id_sha256(np.array([2, 5, 9], dtype="<i8"))
    with pytest.raises(ValueError, match="unique"):
        pulse_id_sha256([2, 2])


def test_sensitivity_is_a_separate_copy_and_constant_shift_preserves_dispersion():
    baseline = np.array([-1.7, -1.6, -1.5])
    original = baseline.copy()
    shifted = add_sensitivity(baseline, np.full(3, 1.6))
    before = residual_metrics(baseline)
    after = residual_metrics(shifted)
    assert np.array_equal(baseline, original)
    assert np.isclose(after["mean_bias_m"], before["mean_bias_m"] + 1.6)
    assert np.isclose(after["cRMSE_m"], before["cRMSE_m"])
    assert np.isclose(after["NMAD_m"], before["NMAD_m"])


def test_bias_decomposition_is_population_squared_error_identity():
    values = np.array([-1.75, -1.65, -1.55, -1.45])
    metrics = residual_metrics(values)
    row = {
        "granule_id": "g", "stratum": "snr_ge_5", "hypothesis": "C7",
        "scenario_label": "C7", "candidate_status": "PROVISIONAL",
        "pulse_id_sha256": "hash", **metrics,
    }
    decomposition = _make_bias_decomposition(pd.DataFrame([row])).iloc[0]
    assert decomposition.decomposition_identity_status == "PASS"
    assert np.isclose(decomposition.bias_squared_m2 + decomposition.dispersion_squared_m2,
                      decomposition.MSE_m2)
    assert np.isclose(decomposition.bias_squared_fraction_of_MSE
                      + decomposition.dispersion_squared_fraction_of_MSE, 1.0)
    assert "not causal" in decomposition.interpretation


def test_insufficient_independent_sites_do_not_get_confidence_intervals():
    assert confidence_interval_status(1).startswith("NOT_ESTIMATED: fewer than 3")
    assert confidence_interval_status(2).startswith("NOT_ESTIMATED: fewer than 3")


def test_track_dispersion_is_withheld_below_minimum_group_size():
    frame = pd.DataFrame({
        "granule_id": ["g", "g"], "track_num": [11, 11], "sweep_num": [1, 1],
        "pulse_index": [1, 2], "longitude_deg": [-75.0, -75.00001],
        "latitude_deg": [38.0, 38.00001], "axis_distance_m": [0.0, 1.0],
        "reference_height_local_plane_m": [10.0, 10.1], "c7_local_slope_deg": [2.0, 2.1],
        "refh_snr": [5.1, 5.2],
    })
    rows = _spatial_summary_rows(frame, "snr_ge_5", {"C7": np.array([-1.0, -1.1])})
    track = next(row for row in rows if row["spatial_unit_type"] == "track")
    assert track["n"] == 2
    assert track["dispersion_status"] == "WITHHELD: n<3"
    assert np.isnan(track["cRMSE_m"])
    assert "NOT_ESTIMATED" in track["confidence_interval_status"]


def test_november_12_influence_table_keeps_all_rows_and_marks_loo_as_diagnostic():
    n = 14
    frame = pd.DataFrame({
        "granule_id": ["g"] * n, "pulse_index": np.arange(n), "track_num": np.arange(n),
        "sweep_num": np.arange(n), "longitude_deg": np.linspace(-75.0, -75.01, n),
        "latitude_deg": np.linspace(38.0, 38.01, n), "refh_snr": np.linspace(5.1, 6.4, n),
        "reference_height_local_plane_m": np.full(n, 10.0), "surface_acceptable": [True] * n,
        "c7_surface_acceptable": [True] * n, "c7_plane_fit_valid": [True] * n,
        "c7_neighbor_count": [32] * n, "c7_nearest_ground_distance_m": np.full(n, 0.15),
        "c7_local_slope_deg": np.full(n, 2.0), "c7_plane_residual_nmad_m": np.full(n, 0.1),
        "c7_ground_roughness_nmad_m": np.full(n, 0.12), "axis_distance_m": np.arange(n, dtype=float),
        "stored_delay_numeric": np.full(n, 1.59), "height_effect_if_m": np.full(n, 1.58),
        "height_effect_ERA5_profile": np.full(n, 1.60),
    })
    c7 = -1.6 + np.linspace(-0.1, 0.1, n)
    scenarios = {name: c7.copy() for name in SCENARIOS}
    scenarios["H1_stored_as_m"] = c7 + 1.58
    scenarios["H1_ERA5_profile"] = c7 + 1.60
    table = _high_influence_table(frame, scenarios)
    assert len(table) == n
    assert table.pulse_index.nunique() == n
    assert not table.deletion_performed.any()
    assert np.isclose(table.C7_centered_squared_error_share.sum(), 1.0)
    assert table.C7_absolute_residual_rank.min() == 1
    assert table.LOO_delta_C7_cRMSE_m.notna().all()
