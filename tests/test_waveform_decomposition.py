from types import SimpleNamespace

import numpy as np

import research.waveform_decomposition as decomposition
from research.waveform_decomposition import estimate_baseline, fit_gaussian_components, residual_diagnostics


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
    assert result["parameter_count"] == 1 + 3 * result["chosen_k"]
    assert all(model["parameter_count"] == 1 + 3 * model["k"] for model in result["models"])


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


def test_baseline_only_is_compared_on_the_same_samples_and_fixed():
    x = np.arange(96, dtype=float)
    y = 3.0 + 9.0 * np.exp(-0.5 * ((x - 44.0) / 2.0) ** 2)
    result = fit_gaussian_components(y, 3.0, 0.5,
                                     detector_peaks=[{"peak_bin": 44.0, "width_bins": 5.0}],
                                     max_components=3)

    models = {model["chosen_k"]: model for model in result["models"]}
    assert set(models) == {0, 1}
    assert models[0]["parameter_count"] == 1
    assert models[1]["parameter_count"] == 4
    assert models[0]["fit_sample_count"] == models[1]["fit_sample_count"] == y.size
    assert models[0]["baseline"] == 3.0
    assert result["chosen_k"] == 1


def test_failed_optimizer_run_is_retained_but_not_selectable(monkeypatch):
    def fail(residuals, x0, bounds, **kwargs):
        return SimpleNamespace(success=False, status=0, message="evaluation limit", nfev=1,
                                x=np.asarray(x0, dtype=float))

    monkeypatch.setattr(decomposition, "least_squares", fail)
    x = np.arange(64, dtype=float)
    y = 2.0 + 10.0 * np.exp(-0.5 * ((x - 30.0) / 2.0) ** 2)
    result = fit_gaussian_components(y, 2.0, 0.5,
                                     detector_peaks=[{"peak_bin": 30.0, "width_bins": 5.0}],
                                     max_components=2)

    model1 = next(model for model in result["models"] if model["chosen_k"] == 1)
    assert not model1["success"]
    assert model1["fit_status"] == "optimizer_failed"
    assert all(not attempt["optimizer_success"] for attempt in model1["attempts"])
    assert result["chosen_k"] == 0
    assert result["success"]
    assert not result["stable_order"]


def test_boundary_hits_are_reported_per_parameter(monkeypatch):
    def lower_bound_fit(residuals, x0, bounds, **kwargs):
        return SimpleNamespace(success=True, status=1, message="synthetic bound solution", nfev=1,
                                x=np.asarray(bounds[0], dtype=float))

    monkeypatch.setattr(decomposition, "least_squares", lower_bound_fit)
    x = np.arange(64, dtype=float)
    y = 2.0 + 10.0 * np.exp(-0.5 * ((x - 30.0) / 2.0) ** 2)
    result = fit_gaussian_components(y, 2.0, 0.5,
                                     detector_peaks=[{"peak_bin": 30.0, "width_bins": 5.0}],
                                     max_components=1)

    model1 = next(model for model in result["models"] if model["chosen_k"] == 1)
    component = model1["boundary_diagnostics"]["components"][0]
    assert component["boundary_hit"]
    assert component["amplitude"]["at_lower"]
    assert component["sigma_bins"]["at_lower"]
    assert model1["boundary_hit_components"] == [1]
    assert not model1["is_admissible"]
    assert model1["admissibility_reason"] == "boundary_hit"
    assert result["chosen_k"] == 0


def test_seed_cap_and_residual_diagnostics_are_explicit():
    x = np.arange(128, dtype=float)
    y = (4.0 + 12.0 * np.exp(-0.5 * ((x - 34.0) / 2.2) ** 2)
         + 8.0 * np.exp(-0.5 * ((x - 81.0) / 3.1) ** 2))
    result = fit_gaussian_components(
        y, 4.0, 0.25,
        detector_peaks=[{"peak_bin": 34.0, "width_bins": 5.0},
                        {"peak_bin": 81.0, "width_bins": 7.0}],
        max_components=5,
    )
    assert result["configured_cap"] == 5
    assert result["available_seed_cap"] == 2
    assert result["searched_k_cap"] == 2
    assert result["seed_limited"]
    diagnostics = residual_diagnostics(np.array([1.0, 2.0, 3.0, 4.0]), 1.0,
                                       sample_indices=[0, 1, 3])
    assert diagnostics["sample_count"] == 3
    assert diagnostics["adjacent_pairs"] == 1
    assert diagnostics["lag1_autocorrelation_adjacent_bins"] is None
    assert diagnostics["adjacent_sign_change_fraction"] == 0.0


def test_selected_model_reports_component_separability_screen():
    x = np.arange(128, dtype=float)
    y = (2.0 + 12.0 * np.exp(-0.5 * ((x - 50.0) / 6.0) ** 2)
         + 10.0 * np.exp(-0.5 * ((x - 55.5) / 6.0) ** 2))
    result = fit_gaussian_components(
        y, 2.0, 0.2,
        detector_peaks=[{"peak_bin": 50.0, "width_bins": 14.0},
                        {"peak_bin": 55.5, "width_bins": 14.0}],
        max_components=2,
    )
    two_component_model = next(model for model in result["models"] if model["chosen_k"] == 2)
    assert len(two_component_model["pairwise_component_overlap"]) == 1
    assert two_component_model["pairwise_component_overlap"][0]["strong_overlap_screen"]


def test_low_snr_noise_does_not_create_components():
    rng = np.random.default_rng(5)
    y = 20.0 + rng.normal(0.0, 1.0, 256)
    result = fit_gaussian_components(y, baseline=20.0, noise_sigma=1.0)
    assert result["chosen_k"] == 0
    assert result["fit_status"] == "no_detectable_peak"


def test_noise_order_is_conditionally_stable_without_seeds():
    y = 90.0 + np.random.default_rng(20261011).normal(0, 1, 192)
    result = fit_gaussian_components(y, 90.0, 1.0)
    assert result["chosen_k"] == result["searched_k_cap"] == 0
    assert result["stable_order"]
    assert result["delta_bic"] is None


def test_bounds_and_overlap_are_rejected_with_raw_order_retained():
    x = np.arange(192, dtype=float)
    y = (90 + 12*np.exp(-0.5*((x-78)/6)**2)
         + 10*np.exp(-0.5*((x-83.5)/6)**2))
    result = fit_gaussian_components(y, 90, 0.12,
        detector_peaks=[{"peak_bin": 78, "width_bins": 14.13},
                        {"peak_bin": 83.5, "width_bins": 14.13}], max_components=3)
    model = next(m for m in result["models"] if m["k"] == 2)
    assert not model["is_admissible"]
    assert model["admissibility_reason"] == "excessive_component_overlap"
    assert result["raw_bic_chosen_k"] == 2
    assert result["chosen_k"] == 1
    assert not result["selected_at_boundary"]


def test_fixed_baseline_prevents_local_outside_window_offset():
    x = np.arange(256, dtype=float)
    y = 5 + 10*np.exp(-0.5*((x-100)/3)**2)
    result = fit_gaussian_components(y, 5, 0.25, fit_indices=np.arange(95, 106))
    assert all(m["baseline"] == 5 for m in result["models"] if m["success"])
    assert result["parameter_count"] == 1 + 3*result["chosen_k"]
    assert abs(result["fit"][0] - 5) < 1e-10


def test_flat_top_is_a_suspicion_and_not_an_int16_rail():
    y = np.zeros(128); y[60:63] = 4095
    stats = decomposition.morphology_stats(y)
    assert stats["peak_flat_top_samples"] == 3
    assert stats["suspected_flat_top_clipping"]
    assert stats["digital_high_rail_hits"] == 0
    assert not decomposition.morphology_stats(np.zeros(128))["suspected_flat_top_clipping"]
    y[61] = 4096
    assert decomposition.morphology_stats(y)["peak_flat_top_samples"] == 1
    with np.testing.assert_raises(ValueError):
        decomposition.morphology_stats([np.nan])
