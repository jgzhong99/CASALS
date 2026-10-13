import numpy as np
import pandas as pd

from research.reference.final_bias_validation import (
    _ciddor_group_refractivity,
    _write_artifact_hash_manifest,
    compare_refh_application_state,
    eligible_independent_test,
    integrate_atmospheric_profile,
    range_effect_from_delay,
    removed_excess_height_effect,
    select_prospective_granules,
    source_frame_status,
    target_xyz_from_range,
)
from research.reference.refh_correction_audit import candidate_evidence_status, compare_common_pulses


def test_artifact_hash_manifest_covers_outputs_and_excludes_itself(tmp_path):
    (tmp_path / "table.csv").write_bytes(b"a,b\n1,2\n")
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "plot.png").write_bytes(b"example image bytes")

    manifest = _write_artifact_hash_manifest(tmp_path)

    assert manifest.relative_path.tolist() == ["figures/plot.png", "table.csv"]
    assert manifest.bytes.tolist() == [19, 8]
    assert "artifact_sha256_manifest.csv" not in manifest.relative_path.tolist()


def test_raw_range_synthetic_end_to_end_identifies_applied_and_unapplied_paths():
    target = np.array([100.0, 200.0, 300.0])
    sensor = np.array([100.0, 200.0, 1300.0])
    beam = (target - sensor) / np.linalg.norm(target - sensor)
    geometric_range = np.linalg.norm(target - sensor)
    atmospheric_excess_m = 1.6
    calibration_bias_m = 0.2
    raw_range = geometric_range + atmospheric_excess_m + calibration_bias_m

    reconstructed_no_atmosphere = target_xyz_from_range(
        sensor, beam, raw_range - calibration_bias_m
    )
    reconstructed_with_atmosphere = target_xyz_from_range(
        sensor, beam, raw_range - atmospheric_excess_m - calibration_bias_m
    )
    assert np.isclose(np.linalg.norm(reconstructed_no_atmosphere - sensor), geometric_range + atmospheric_excess_m)
    assert np.allclose(reconstructed_with_atmosphere, target)

    no_atmosphere_state = compare_refh_application_state(
        geometric_range + atmospheric_excess_m, raw_range, atmospheric_excess_m,
        calibration_bias_m=calibration_bias_m,
    )
    applied_state = compare_refh_application_state(
        geometric_range, raw_range, atmospheric_excess_m,
        calibration_bias_m=calibration_bias_m,
    )
    assert no_atmosphere_state["status"] == "CONFIRMED_NOT_APPLIED"
    assert applied_state["status"] == "CONFIRMED_APPLIED"


def test_processed_refh_and_delay_without_independent_raw_range_stays_unknown():
    result = compare_refh_application_state(
        processed_range_m=500.0,
        independent_raw_range_m=None,
        atmospheric_excess_m=1.5,
        calibration_bias_m=None,
    )
    assert result["status"] == "APPLICATION_STATE_NOT_IDENTIFIABLE_FROM_AVAILABLE_INPUTS"


def test_one_way_two_way_and_nanosecond_conversion_factors():
    value = np.array([1.0])
    assert np.allclose(range_effect_from_delay(value, "m_one_way_range"), [1.0])
    assert np.allclose(range_effect_from_delay(value, "ns_two_way_time"), [0.149896229])
    assert np.allclose(range_effect_from_delay(value, "ns_one_way_time"), [0.299792458])
    with np.testing.assert_raises(ValueError):
        range_effect_from_delay(value, "unknown")


def test_removed_positive_excess_range_raises_downward_target_height_by_projection():
    assert np.isclose(removed_excess_height_effect(2.0, np.pi / 2), 2.0)
    assert np.isclose(removed_excess_height_effect(2.0, np.pi / 6), 1.0)


def test_ciddor_group_refractivity_and_profile_integration_have_physical_scale():
    surface_n = float(_ciddor_group_refractivity(1.040, 15.0, 1013.25, 1200.0))
    assert 2.5e-4 < surface_n < 3.1e-4
    pressure = np.array([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 20], dtype=float)
    heights = np.array([100, 950, 1800, 2800, 4100, 5600, 7300, 9400, 11800, 16000, 20000, 26000], dtype=float)
    temp_k = 288.15 - 0.0065 * heights
    q = 0.010 * np.exp(-heights / 2500)
    geopotential = heights * 9.80665
    path = integrate_atmospheric_profile(
        pressure, temp_k, q, geopotential,
        surface_pressure_pa=101325.0, ground_height_m=0.0,
        instrument_height_m=8500.0, beam_elevation_rad=np.deg2rad(87.0),
    )
    assert 1.0 < path["slant_one_way_group_excess_path_m"] < 3.5
    assert path["surface_group_refractivity_m_per_m"] > path["instrument_group_refractivity_m_per_m"]


def test_ground_eligibility_and_snr_selection_do_not_depend_on_residual_values():
    population = pd.DataFrame({
        "pulse_index": [1, 2, 3, 4],
        "granule_id": ["g"] * 4,
        "research_role": ["independent_test"] * 4,
        "is_ground_target_eligible": [True, True, False, True],
        "common_support_all_candidates": [True, True, True, False],
        "refh_snr": [6.0, 4.9, 9.0, 9.0],
        "C7_residual_m": [-1e9, 1e9, 0.0, 0.0],
    })
    chosen = eligible_independent_test(population)
    assert chosen.pulse_index.tolist() == [1]
    changed = population.assign(C7_residual_m=[0.0, 0.0, -1e20, 1e20])
    chosen_again = eligible_independent_test(changed)
    assert chosen_again.pulse_index.tolist() == chosen.pulse_index.tolist()


def test_prospective_granule_selection_uses_metadata_not_residuals():
    metadata = [
        {"object_key": "old", "acquisition_date": "2024-11-12", "scene_evidence": True,
         "eligible_ground_site_evidence": True, "residual": -999.0},
        {"object_key": "new_good", "acquisition_date": "2025-05-20", "scene_evidence": True,
         "eligible_ground_site_evidence": True, "residual": 999.0},
        {"object_key": "new_no_site", "acquisition_date": "2025-06-20", "scene_evidence": True,
         "eligible_ground_site_evidence": False, "residual": -9999.0},
    ]
    selected = select_prospective_granules(metadata, {"2024-11-12", "2024-11-18"})
    assert [row["object_key"] for row in selected] == ["new_good"]
    assert metadata[0]["prospective_selection_status"] == "NO_NEW_ACQUISITION_DATE"


def test_source_frame_numeric_noop_does_not_prove_realization_or_epoch():
    assert source_frame_status(True, None, None) == "UNRESOLVED_SOURCE_FRAME_NUMERIC_NOOP"
    assert source_frame_status(False, "NAD83(2011)", "2024.87") == "SOURCE_FRAME_PROVENANCE_AVAILABLE"


def test_same_pulse_comparison_and_unknown_status_never_promote_to_verified():
    ids = np.array([11, 12, 13])
    rows = compare_common_pulses(ids, {"C7": np.array([-1.0, -1.1, -0.9]),
                                      "H1": np.array([0.2, 0.1, 0.3])})
    assert rows.n.tolist() == [3, 3]
    assert rows.hypothesis.tolist() == ["C7", "H1"]
    with np.testing.assert_raises(ValueError):
        compare_common_pulses(ids, {"C7": np.array([-1.0, -1.1, -0.9]),
                                    "H1": np.array([0.2, 0.1])})
    assert candidate_evidence_status("UNKNOWN", physical_basis=True) == "PHYSICAL_SENSITIVITY_ONLY"
    assert candidate_evidence_status("UNKNOWN", physical_basis=False) == "UNAVAILABLE"

