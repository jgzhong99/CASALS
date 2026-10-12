import numpy as np

from research.nadir_window_reflection import (
    extract_candidate_bands,
    flag_bins,
    mask_bins,
    scan_all_bins_events,
    score_track_bins,
    subtract_template,
)


def _synthetic_block():
    rng = np.random.default_rng(8)
    rx = rng.normal(0.0, 1.0, size=(24, 3, 300)).astype(np.float32)
    rx[:18, 1, 100:103] += 8.0
    mean = np.zeros((24, 3), dtype=np.float32)
    std = np.ones((24, 3), dtype=np.float32)
    refh = np.tile(np.linspace(-3.0, 3.0, 24)[:, None], (1, 3))
    return rx, mean, std, refh


def test_occupancy_amplitude_and_candidate_band_are_reported():
    rx, mean, std, refh = _synthetic_block()
    scores = score_track_bins(rx, mean, std, refh, early_bins=200, threshold_sigma=5.0)
    row = scores[(scores.track_num == 1) & (scores.bin == 101)].iloc[0]
    assert np.isclose(row.occupancy, 18 / 24)
    assert row.occupancy_ci95_low < row.occupancy < row.occupancy_ci95_high
    assert row.median_excess_sigma > 7.0
    assert row.ghost_score > 0.45

    bands = extract_candidate_bands(scores, occupancy_min=0.5, score_min=0.45)
    band = bands[(bands.track_num == 1) & (bands.start_bin == 100)].iloc[0]
    assert band.end_bin == 102
    assert band.max_persistence_sweeps >= 18
    assert band.max_persistence_sweeps <= 24
    assert scores.valid_sweep_count.eq(24).all()


def test_missing_waveform_samples_use_per_bin_valid_denominator():
    rx = np.zeros((10, 1, 20), dtype=np.float32)
    rx[:4, 0, 7] = 8.0
    rx[4, 0, 7] = np.nan
    mean = np.zeros((10, 1), dtype=np.float32)
    std = np.ones((10, 1), dtype=np.float32)
    refh = np.zeros((10, 1), dtype=float)
    scores = score_track_bins(rx, mean, std, refh, early_bins=12, threshold_sigma=5)
    row = scores[(scores.track_num == 0) & (scores.bin == 7)].iloc[0]
    assert row.valid_sweep_count == 9
    assert row.event_count == 4
    assert np.isclose(row.occupancy, 4 / 9)


def test_median_excess_counts_uses_actual_hit_samples_with_variable_noise():
    rx = np.zeros((4, 1, 12), dtype=np.float32)
    rx[:, 0, 5] = [6.0, 12.0, 24.0, 32.0]
    mean = np.zeros((4, 1), dtype=np.float32)
    std = np.array([[1.0], [2.0], [4.0], [8.0]], dtype=np.float32)
    refh = np.zeros((4, 1), dtype=float)
    row = score_track_bins(rx, mean, std, refh, early_bins=10, threshold_sigma=5.0)
    target = row[(row.track_num == 0) & (row.bin == 5)].iloc[0]
    assert target.event_count == 3
    assert np.isclose(target.median_excess_counts, 12.0)
    assert np.isclose(target.median_excess_sigma, 6.0)


def test_all_bin_stage_a_finds_a_delayed_event_and_tracks_bin_drift():
    rng = np.random.default_rng(41)
    rx = rng.normal(0.0, 1.0, size=(24, 1, 1800)).astype(np.float32)
    mean = np.zeros((24, 1), dtype=np.float32)
    std = np.ones((24, 1), dtype=np.float32)
    refh = np.zeros((24, 1), dtype=float)
    injected_bins = 1400 + np.arange(24)
    rx[np.arange(24), 0, injected_bins] += 12.0

    events = scan_all_bins_events(rx, mean, std, threshold_sigma=8.0,
                                  min_prominence_sigma=4.0, sweep_numbers=np.arange(100, 124))
    assert len(events) == 24
    assert events.sweep_num.tolist() == list(range(100, 124))
    np.testing.assert_array_equal(events.bin.to_numpy(), injected_bins)
    assert events.bin.max() > 1200

    scores = score_track_bins(rx, mean, std, refh, early_bins=1800, threshold_sigma=5.0)
    bands = extract_candidate_bands(scores, occupancy_min=0.20,
                                    median_excess_sigma_min=5.0, score_min=0.45)
    assert bands.empty  # the fixed-bin gate misses this deliberately drifting sequence


def test_all_bin_stage_a_does_not_bridge_missing_sample_gaps():
    rx = np.zeros((1, 1, 32), dtype=np.float32)
    rx[0, 0, 8] = 12.0
    rx[0, 0, 10] = 12.0
    rx[0, 0, 9] = np.nan
    rx[0, 0, 20] = 12.0
    mean = np.zeros((1, 1), dtype=np.float32)
    std = np.ones((1, 1), dtype=np.float32)
    events = scan_all_bins_events(rx, mean, std, threshold_sigma=8, min_prominence_sigma=4)
    assert events.bin.tolist() == [20]


def test_quiet_noise_block_does_not_create_a_candidate_band():
    rng = np.random.default_rng(2026)
    rx = rng.normal(0.0, 1.0, size=(90, 2, 128)).astype(np.float32)
    mean = np.zeros((90, 2), dtype=np.float32)
    std = np.ones((90, 2), dtype=np.float32)
    refh = np.zeros((90, 2), dtype=float)
    scores = score_track_bins(rx, mean, std, refh, early_bins=64, threshold_sigma=5)
    bands = extract_candidate_bands(scores, occupancy_min=0.20,
                                   median_excess_sigma_min=5, score_min=0.45)
    assert bands.empty
    np.testing.assert_array_equal(mask_bins(rx[0, 0], []), rx[0, 0])


def test_flag_and_mask_cover_inclusive_candidate_interval():
    flags = flag_bins(200, [(4, 6)])
    assert np.flatnonzero(flags).tolist() == [4, 5, 6]
    masked = mask_bins(np.arange(10, dtype=float), [(4, 6)])
    assert np.isnan(masked[4:7]).all()
    np.testing.assert_array_equal(masked[:4], np.arange(4, dtype=float))


def test_template_subtraction_preserves_shape_and_rejects_mismatch():
    y = np.arange(8, dtype=float)
    template = np.ones(8, dtype=float)
    corrected = subtract_template(y, template, scale=0.5)
    assert corrected.shape == y.shape
    np.testing.assert_allclose(corrected, y - 0.5)
    try:
        subtract_template(y, template[:4])
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch should fail")
