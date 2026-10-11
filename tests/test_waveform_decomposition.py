import numpy as np

from research.waveform_decomposition import estimate_baseline, fit_gaussian_components


def test_baseline_uses_h5_background_and_has_robust_fallback():
    y = np.array([8.0, 9.0, 10.0, 11.0, 50.0])
    result = estimate_baseline(y, background_mean=9.5, background_std=1.2)
    assert result["baseline"] == 9.5
    assert result["noise_sigma"] == 1.2
    assert result["method"] == "h5_background"

    fallback = estimate_baseline(y, fallback_bins=4)
    assert fallback["baseline"] == 9.5
    assert fallback["noise_sigma"] > 0


def test_gaussian_fit_recovers_two_components_and_residual():
    rng = np.random.default_rng(12)
    x = np.arange(128, dtype=float)
    y = (
        4.0
        + 12.0 * np.exp(-0.5 * ((x - 34.0) / 2.2) ** 2)
        + 8.0 * np.exp(-0.5 * ((x - 81.0) / 3.1) ** 2)
        + rng.normal(0.0, 0.25, x.size)
    )
    result = fit_gaussian_components(y, baseline=4.0, noise_sigma=0.25, max_components=4)

    assert result["success"]
    assert result["chosen_k"] == 2
    centers = sorted(component["center_bin"] for component in result["components"])
    np.testing.assert_allclose(centers, [34.0, 81.0], atol=0.5)
    np.testing.assert_allclose(result["residual"], y - result["fit"], rtol=0, atol=1e-12)
    assert np.isclose(result["rmse"], np.sqrt(np.mean(result["residual"] ** 2)))
    assert result["parameter_count"] == 2 + 3 * result["chosen_k"]
    assert all(model["parameter_count"] == 2 + 3 * model["k"] for model in result["models"])


def test_fit_window_uses_original_bin_coordinates_and_reports_its_denominator():
    rng = np.random.default_rng(19)
    x = np.arange(128, dtype=float)
    y = (4.0 + 12.0 * np.exp(-0.5 * ((x - 34.0) / 2.2) ** 2)
         + 8.0 * np.exp(-0.5 * ((x - 81.0) / 3.1) ** 2)
         + rng.normal(0.0, 0.25, x.size))
    indices = np.r_[np.arange(20, 50), np.arange(68, 96)]
    result = fit_gaussian_components(
        y, 4.0, 0.25, detector_peaks=[{"peak_bin": 34.0, "width_bins": 5.0},
                                     {"peak_bin": 81.0, "width_bins": 7.0}],
        fit_indices=indices, max_components=4)

    assert result["chosen_k"] == 2
    assert result["fit_sample_count"] == indices.size
    assert result["fit_sample_fraction"] == indices.size / y.size
    np.testing.assert_allclose(
        sorted(component["center_bin"] for component in result["components"]),
        [34.0, 81.0], atol=0.5,
    )
    assert np.isfinite(result["full_window_rmse"])


def test_fit_window_rejects_invalid_bin_indices():
    y = np.ones(16, dtype=float)
    with np.testing.assert_raises(ValueError):
        fit_gaussian_components(y, 1.0, 1.0, fit_indices=[1, 1, 2, 3, 4])


def test_low_snr_noise_does_not_create_components():
    rng = np.random.default_rng(5)
    y = 20.0 + rng.normal(0.0, 1.0, 256)
    result = fit_gaussian_components(y, baseline=20.0, noise_sigma=1.0)
    assert result["chosen_k"] == 0
    assert result["fit_status"] == "no_detectable_peak"
