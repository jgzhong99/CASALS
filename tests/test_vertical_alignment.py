import numpy as np
import pandas as pd
import pytest

from research.reference.vertical_alignment import METHODS, empirical_alignment, summarize


def population():
    frame = pd.DataFrame({
        "granule_id": ["example"] * 4, "pulse_index": [1, 2, 3, 4],
        "research_role": ["calibration_offset"] * 2 + ["independent_test"] * 2,
        "common_support_all_candidates": [True] * 4, "refh_snr": [6.] * 4,
        "track_num": [1] * 4, "easting_m_epsg6347": [0., 1., 20., 21.],
        "northing_m_epsg6347": [0.] * 4, "c7_reference_height_local_plane_m": [0.] * 4,
    })
    for method in METHODS:
        frame[f"{method}_residual_m"] = [-2., -4., -1., -3.]
    return frame


def test_calibration_median_sign_and_no_test_leakage():
    frame = population()
    metrics, _ = empirical_alignment(frame)
    after = metrics.query("method == 'GEOID18' and state == 'aligned'")
    assert (after.offset_m == -3.).all()
    np.testing.assert_allclose(after.mean_bias_m, 1.)
    np.testing.assert_allclose(after.RMSE_m, np.sqrt(2))
    frame.loc[frame.research_role.eq("independent_test"), "GEOID18_residual_m"] += 100
    changed, _ = empirical_alignment(frame)
    assert (changed.query("method == 'GEOID18'").offset_m == -3.).all()


def test_shared_population_retains_large_residual_and_coverage():
    frame = population()
    frame.loc[0, "common_support_all_candidates"] = False
    for method in METHODS:
        frame.loc[3, f"{method}_residual_m"] = 1000.
    metrics = summarize(frame, population="test")
    assert set(metrics.n) == {3}
    assert set(metrics.coverage) == {.75}
    assert metrics.pulse_sha256.nunique() == 1
    assert metrics.RMSE_m.min() > 500


def test_overlapping_calibration_geometry_is_rejected():
    frame = population()
    frame.loc[2:, "easting_m_epsg6347"] = [0., 1.]
    with pytest.raises(AssertionError):
        empirical_alignment(frame)
