import numpy as np
import pandas as pd

from research.l1b_validation import (
    aggregate_roi_track_responses,
    calibrate_noise_peak_threshold,
    candidate_signal_features,
    common_track_ids,
    matched_track_differences,
    roi_primary_eligibility,
    site_level_unit_counts,
    waveform_response_features,
)


def test_roi_identity_candidate_exclusion_and_naip_date_provenance():
    properties = {
        "analysis_status": "Included",
        "granule_id": "granule_a",
        "source_date": "2023-06-02",
        "roi_id": "field_1",
        "site_id": "field_site_1",
    }
    result = roi_primary_eligibility(
        properties, expected_granule_id="granule_a", observed_image_date="2023-06-02"
    )
    assert result["primary_eligible"] is True
    assert roi_primary_eligibility(
        properties, expected_granule_id="granule_b", observed_image_date="2023-06-02"
    )["reasons"] == ["granule_identity_mismatch"]
    assert roi_primary_eligibility(
        properties, expected_granule_id="granule_a", observed_image_date="2022-06-15"
    )["reasons"] == ["source_date_mismatch"]
    candidate = {**properties, "analysis_status": "Candidate"}
    assert roi_primary_eligibility(
        candidate, expected_granule_id="granule_a", observed_image_date="2023-06-02"
    )["primary_eligible"] is False


def test_real_background_block_bootstrap_controls_full_waveform_false_alarms():
    rng = np.random.default_rng(21)
    background_windows = rng.normal(0.0, 1.0, size=(512, 128)).astype(np.float32)
    calibration = calibrate_noise_peak_threshold(
        background_windows,
        quantile=0.995,
        calibration_simulations=2000,
        validation_simulations=2000,
        seed=97,
    )
    assert calibration["threshold_peak_prominence_sigma"] > 4.0
    assert calibration["real_background_window_count"] == 512
    assert calibration["validation_false_alarm_rate"] <= 0.02


def test_low_snr_clear_and_multi_peak_waveforms_use_calibrated_prominence():
    rng = np.random.default_rng(33)
    bins = np.arange(2728, dtype=float)
    noise = rng.normal(0.0, 1.0, bins.size)
    low = noise + 0.5 * np.exp(-0.5 * ((bins - 1300) / 4.0) ** 2)
    low_result = candidate_signal_features(low, 0.0, 1.0, prominence_threshold_sigma=8.0)
    assert low_result["candidate_signal_detected"] is False

    clear = noise + 14.0 * np.exp(-0.5 * ((bins - 1300) / 4.0) ** 2)
    clear_result = candidate_signal_features(clear, 0.0, 1.0, prominence_threshold_sigma=8.0)
    assert clear_result["candidate_signal_detected"] is True
    assert clear_result["candidate_peak_bin"] == 1300
    assert clear_result["candidate_peak_width_bins"] >= 2.0

    multi = noise.copy()
    multi += 14.0 * np.exp(-0.5 * ((bins - 900) / 4.0) ** 2)
    multi += 12.0 * np.exp(-0.5 * ((bins - 1700) / 5.0) ** 2)
    multi_result = candidate_signal_features(multi, 0.0, 1.0, prominence_threshold_sigma=8.0)
    assert multi_result["candidate_signal_detected"] is True
    assert multi_result["candidate_local_peak_count"] >= 2


def test_missing_waveform_is_invalid_and_not_counted_as_detection():
    result = candidate_signal_features(np.array([1.0, np.nan]), 0.0, 1.0, prominence_threshold_sigma=8.0)
    assert result["candidate_signal_status"] == "invalid_waveform"
    assert result["candidate_signal_detected"] is False
    assert result["candidate_peak_prominence_sigma"] is None


def test_every_pulse_stays_in_roi_track_denominator_and_common_track_pairs_use_medians():
    rows = pd.DataFrame([
        {"granule_id":"g", "roi_id":"field", "site_id":"s1", "class_label":"open_field", "track_num":10, "candidate_signal_detected":True, "refh_snr":5.0, "refh_amp_counts":100, "width_halfmax_bins":4.0},
        {"granule_id":"g", "roi_id":"field", "site_id":"s1", "class_label":"open_field", "track_num":10, "candidate_signal_detected":False, "refh_snr":1.0, "refh_amp_counts":20, "width_halfmax_bins":np.nan},
        {"granule_id":"g", "roi_id":"forest", "site_id":"s2", "class_label":"tree_canopy", "track_num":10, "candidate_signal_detected":False, "refh_snr":2.0, "refh_amp_counts":30, "width_halfmax_bins":np.nan},
    ])
    track_rows = aggregate_roi_track_responses(rows)
    field = next(row for row in track_rows if row["roi_id"] == "field")
    assert field["n_pulses"] == 2
    assert field["n_candidate_signal"] == 1
    assert field["candidate_signal_rate"] == 0.5
    assert field["width_halfmax_bins_n"] == 1
    difference = matched_track_differences(
        track_rows, roi_a="field", roi_b="forest", metric="refh_snr"
    )
    assert len(difference) == 1
    assert difference[0]["track_num"] == 10
    assert difference[0]["difference_a_minus_b"] == 1.0
    assert common_track_ids([10, 11], [10, 12]) == [10]


def test_site_denominator_counts_unique_sites_not_pulses_or_roi_polygons():
    rows = pd.DataFrame([
        {"roi_id":"forest_a", "site_id":"same_forest", "class_label":"tree_canopy"},
        {"roi_id":"forest_b", "site_id":"same_forest", "class_label":"tree_canopy"},
        {"roi_id":"forest_c", "site_id":"other_forest", "class_label":"tree_canopy"},
        {"roi_id":"field_a", "site_id":"field_site", "class_label":"open_field"},
    ])
    counts = site_level_unit_counts(rows)
    assert counts["tree_canopy"]["n_independent_sites"] == 2
    assert counts["open_field"]["n_independent_sites"] == 1


def test_waveform_width_and_energy_are_bins_and_counts_not_reflectance():
    bins = np.arange(101, dtype=float)
    waveform = 5.0 + 15.0 * np.exp(-0.5 * ((bins - 50.0) / 4.0) ** 2)
    features = waveform_response_features(waveform, 5.0, 1.0, threshold_sigma=3.0)
    assert features["width_halfmax_bins"] > 0
    assert features["dominant_support_energy_counts_bin"] > 0
    assert np.isclose(features["energy_centroid_bin"], 50.0)
