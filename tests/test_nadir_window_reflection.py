import numpy as np

from research.nadir_window_reflection import (
    extract_candidate_bands,
    flag_bins,
    mask_bins,
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
    assert row.median_excess_sigma > 7.0
    assert row.ghost_score > 0.45

    bands = extract_candidate_bands(scores, occupancy_min=0.5, score_min=0.45)
    band = bands[(bands.track_num == 1) & (bands.start_bin == 100)].iloc[0]
    assert band.end_bin == 102
    assert band.max_persistence_sweeps >= 18


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
