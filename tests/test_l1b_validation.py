import numpy as np
import pytest

import research.l1b_validation as validation
from research.l1b_validation import (
    candidate_signal_features,
    reference_readiness,
    transform_xyz_with_verified_frames,
    waveform_response_features,
)


def _complete_reference_evidence():
    return {
        "casals_horizontal_reference_verified": True,
        "casals_vertical_reference_verified": True,
        "reference_horizontal_crs_verified": True,
        "reference_vertical_frame_verified": True,
        "historical_z_processing_verified": True,
        "reference_source_provenance_verified": True,
        "best_required_transform_available": True,
    }


def test_reference_readiness_blocks_missing_or_non_boolean_evidence():
    missing = reference_readiness({})
    assert missing["status"] == "BLOCKED"
    assert missing["absolute_elevation_errors_permitted"] is False
    assert {item["code"] for item in missing["blockers"]} >= {
        "CASALS_VERTICAL_FRAME_UNVERIFIED",
        "REFERENCE_VERTICAL_FRAME_UNVERIFIED",
        "REQUIRED_TRANSFORM_UNAVAILABLE",
    }

    evidence = _complete_reference_evidence()
    evidence["reference_vertical_frame_verified"] = "True"
    string_boolean = reference_readiness(evidence)
    assert string_boolean["status"] == "BLOCKED"
    assert string_boolean["absolute_elevation_errors_permitted"] is False


def test_reference_readiness_only_passes_complete_explicit_evidence():
    ready = reference_readiness(_complete_reference_evidence())
    assert ready["status"] == "VERIFIED"
    assert ready["absolute_elevation_errors_permitted"] is True
    assert ready["blockers"] == []


@pytest.mark.parametrize("key", _complete_reference_evidence())
def test_each_unverified_reference_gate_blocks_absolute_errors(key):
    evidence = _complete_reference_evidence()
    evidence[key] = False
    result = reference_readiness(evidence)
    assert result["status"] == "BLOCKED"
    assert result["absolute_elevation_errors_permitted"] is False
    assert len(result["blockers"]) == 1


def test_reference_readiness_empirical_offset_cannot_clear_frame_blocker():
    evidence = _complete_reference_evidence()
    evidence["casals_vertical_reference_verified"] = False
    evidence["empirical_dz_m"] = -37.8
    result = reference_readiness(evidence)
    assert result["status"] == "BLOCKED"
    assert result["absolute_elevation_errors_permitted"] is False


def test_xyz_transform_is_blocked_before_any_crs_operation_when_evidence_is_incomplete():
    evidence = _complete_reference_evidence()
    evidence["historical_z_processing_verified"] = False
    evidence["empirical_dz_m"] = 42.0
    with pytest.raises(ValueError, match="blocks XYZ transformation"):
        transform_xyz_with_verified_frames([[0.0, 0.0, 0.0]], "invalid CRS", "EPSG:4978", evidence)


def test_xyz_transform_requires_three_dimensional_crs_and_finite_xyz():
    evidence = _complete_reference_evidence()
    with pytest.raises(ValueError, match="three axes"):
        transform_xyz_with_verified_frames([[0.0, 0.0, 0.0]], "EPSG:4326", "EPSG:4978", evidence)
    with pytest.raises(ValueError, match="finite x, y, z"):
        transform_xyz_with_verified_frames([[0.0, 0.0, np.nan]], "EPSG:4979", "EPSG:4978", evidence)


def test_xyz_transform_applies_known_complete_3d_operation_and_roundtrips():
    result = transform_xyz_with_verified_frames(
        [[0.0, 0.0, 100.0]],
        "EPSG:4979",
        "EPSG:4978",
        _complete_reference_evidence(),
    )
    assert result["status"] == "VERIFIED"
    assert np.allclose(result["transformed_xyz"], [[6378237.0, 0.0, 0.0]], atol=1e-8)
    assert np.all(np.asarray(result["roundtrip_max_abs_error"]) <= result["roundtrip_tolerance"])
    assert result["source_axis_units"] == ["degree", "degree", "metre"]
    assert result["target_axis_units"] == ["metre", "metre", "metre"]


def test_xyz_transform_rejects_unavailable_non_ballpark_operation(monkeypatch):
    class UnavailableGroup:
        best_available = False
        transformers = []

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(validation, "TransformerGroup", UnavailableGroup)
    with pytest.raises(RuntimeError, match="best non-ballpark 3D coordinate operation is unavailable"):
        transform_xyz_with_verified_frames(
            [[0.0, 0.0, 100.0]],
            "EPSG:4979",
            "EPSG:4978",
            _complete_reference_evidence(),
        )


def test_waveform_response_features_measure_single_and_broad_peaks():
    bins = np.arange(101, dtype=float)
    single = 100.0 + 20.0 * np.exp(-0.5 * ((bins - 50.0) / 4.0) ** 2)
    result = waveform_response_features(single, 100.0, 2.0)
    assert result["waveform_status"] == "signal_present_3sigma"
    assert result["peak_bin"] == 50
    assert result["peak_snr_sigma"] == 10.0
    assert result["peak_count"] == 1
    assert 7 <= result["width_halfmax_bins"] <= 11
    assert result["dominant_support_energy_counts_bin"] > 0
    assert result["energy_centroid_bin"] == 50.0
    assert result["signal_support_component_count"] == 1

    broad = 100.0 + 15.0 * np.exp(-0.5 * ((bins - 50.0) / 9.0) ** 2)
    broad_result = waveform_response_features(broad, 100.0, 2.0)
    assert broad_result["waveform_status"] == "signal_present_3sigma"
    assert broad_result["width_halfmax_bins"] > result["width_halfmax_bins"]


def test_waveform_response_features_keeps_low_signal_and_noise_only_records():
    low_signal = 20.0 + 0.4 * np.exp(-0.5 * ((np.arange(32) - 16.0) / 2.0) ** 2)
    result = waveform_response_features(low_signal, 20.0, 1.0)
    assert result["waveform_status"] == "no_3sigma_peak"
    assert result["peak_count"] == 0
    assert result["peak_excess_counts"] < 3.0
    assert result["width_halfmax_bins"] is None
    assert result["dominant_support_energy_counts_bin"] is None
    assert result["finite_fraction"] == 1.0


def test_waveform_response_features_separates_disjoint_supports_and_integrates_dominant_only():
    values = np.full(120, 10.0)
    values[20:24] = [14.0, 18.0, 15.0, 13.0]
    values[80:83] = [14.0, 16.0, 14.0]
    result = waveform_response_features(values, 10.0, 1.0, minimum_peak_distance=5)
    assert result["signal_support_component_count"] == 2
    assert len(result["signal_support_components"]) == 2
    assert result["peak_count"] == 2
    assert result["dominant_support_start_bin"] == 20
    assert result["dominant_support_end_bin"] == 23
    assert result["dominant_support_energy_counts_bin"] == 20.0
    assert result["signal_extent_bins"] == 63


def test_waveform_response_features_handles_missing_data_and_empty_inputs():
    values = np.full(20, 1.0)
    values[:2] = np.nan
    result = waveform_response_features(values, 1.0, 1.0)
    assert result["waveform_status"] == "invalid_waveform"
    assert result["finite_fraction"] == 0.9
    assert result["peak_snr_sigma"] is None

    empty = waveform_response_features(np.array([]), 1.0, 1.0)
    assert empty["waveform_status"] == "invalid_waveform"
    assert empty["finite_fraction"] == 0.0


def test_waveform_response_features_flags_bad_background_and_constant_baseline():
    no_signal = waveform_response_features(np.full(32, 20.0), 20.0, 0.0)
    assert no_signal["waveform_status"] == "invalid_background"
    assert no_signal["peak_snr_sigma"] is None

    constant = waveform_response_features(np.full(32, 20.0), 20.0, 2.0)
    assert constant["waveform_status"] == "no_3sigma_peak"
    assert constant["peak_excess_counts"] == 0.0


def test_waveform_response_features_detects_edge_peaks_and_integer_clipping():
    edge = np.full(25, 10, dtype=np.int16)
    edge[0] = 20
    result = waveform_response_features(edge, 10.0, 1.0)
    assert result["waveform_status"] == "signal_present_3sigma"
    assert result["peak_bin"] == 0
    assert result["peak_count"] == 1

    clipped = np.full(25, 10, dtype=np.int16)
    clipped[-1] = np.iinfo(np.int16).max
    clipped_result = waveform_response_features(clipped, 10.0, 1.0)
    assert clipped_result["is_clipped"] is True
    assert clipped_result["clipped_bin_count"] == 1


def test_candidate_detector_uses_weaker_qualified_peak_when_strongest_is_too_narrow():
    bins = np.arange(201, dtype=float)
    waveform = 24.0 * np.exp(-0.5 * ((bins - 42.0) / 0.5) ** 2)
    waveform += 8.0 * np.exp(-0.5 * ((bins - 130.0) / 4.0) ** 2)

    result = candidate_signal_features(
        waveform, 0.0, 1.0, prominence_threshold_sigma=5.0, minimum_width_bins=4.0
    )

    assert result["candidate_signal_detected"] is True
    assert result["candidate_peak_bin"] == 130
    assert result["candidate_peak_width_bins"] >= 4.0
    assert result["maximum_local_peak_prominence_sigma"] > result["candidate_peak_prominence_sigma"]


def test_candidate_detector_rejects_deterministic_noise_only_waveform():
    noise = np.random.default_rng(20261012).normal(0.0, 1.0, 2728)
    result = candidate_signal_features(
        noise, 0.0, 1.0, prominence_threshold_sigma=15.0, minimum_width_bins=2.0
    )

    assert result["candidate_signal_detected"] is False
    assert result["candidate_signal_status"] == "no_candidate"
    assert result["candidate_peak_bin"] is None


def test_candidate_detector_accepts_a_normal_positive_peak():
    bins = np.arange(101, dtype=float)
    waveform = 12.0 * np.exp(-0.5 * ((bins - 50.0) / 4.0) ** 2)
    result = candidate_signal_features(
        waveform, 0.0, 1.0, prominence_threshold_sigma=8.0, minimum_width_bins=2.0
    )

    assert result["candidate_signal_detected"] is True
    assert result["candidate_peak_bin"] == 50
    assert result["candidate_peak_height_sigma"] == pytest.approx(12.0)


def test_candidate_detector_rejects_negative_peak_with_large_prominence():
    bins = np.arange(101, dtype=float)
    waveform = -10.0 + 9.0 * np.exp(-0.5 * ((bins - 28.0) / 4.0) ** 2)
    waveform[70] = -30.0
    result = candidate_signal_features(
        waveform, 0.0, 1.0, prominence_threshold_sigma=5.0, minimum_width_bins=2.0
    )

    assert result["maximum_local_peak_prominence_sigma"] >= 5.0
    assert result["candidate_signal_detected"] is False
    assert result["candidate_peak_bin"] is None


def test_candidate_detector_selects_strongest_among_multiple_qualified_peaks():
    bins = np.arange(181, dtype=float)
    waveform = 8.0 * np.exp(-0.5 * ((bins - 42.0) / 4.0) ** 2)
    waveform += 12.0 * np.exp(-0.5 * ((bins - 132.0) / 5.0) ** 2)
    result = candidate_signal_features(
        waveform, 0.0, 1.0, prominence_threshold_sigma=5.0, minimum_width_bins=3.0
    )

    assert result["candidate_signal_detected"] is True
    assert result["candidate_local_peak_count"] == 2
    assert result["candidate_peak_bin"] == 132


def test_candidate_detector_returns_no_candidate_when_no_peak_qualifies():
    bins = np.arange(101, dtype=float)
    waveform = 3.0 * np.exp(-0.5 * ((bins - 50.0) / 4.0) ** 2)
    result = candidate_signal_features(
        waveform, 0.0, 1.0, prominence_threshold_sigma=8.0, minimum_width_bins=3.0
    )

    assert result["candidate_signal_detected"] is False
    assert result["candidate_signal_status"] == "no_candidate"
    assert result["candidate_peak_bin"] is None
    assert result["candidate_peak_prominence_sigma"] is None

