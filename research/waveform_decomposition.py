"""Small, non-production waveform decomposition helpers for CASALS research."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths


def _robust_sigma(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    median = float(np.median(finite))
    sigma = float(1.4826 * np.median(np.abs(finite - median)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(finite))
    return sigma if np.isfinite(sigma) and sigma > 0 else 1.0


def estimate_baseline(
    waveform: Sequence[float],
    background_mean: float | None = None,
    background_std: float | None = None,
    fallback_bins: int = 256,
) -> dict[str, Any]:
    """Use H5 background fields when present; otherwise estimate from leading bins."""

    y = np.asarray(waveform, dtype=np.float64).reshape(-1)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        raise ValueError("waveform contains no finite samples")
    if background_mean is not None and np.isfinite(background_mean):
        noise = float(background_std) if background_std is not None and np.isfinite(background_std) and background_std > 0 else _robust_sigma(finite)
        return {"baseline": float(background_mean), "noise_sigma": noise, "method": "h5_background"}

    lead = y[: max(1, min(int(fallback_bins), y.size))]
    lead = lead[np.isfinite(lead)]
    baseline = float(np.median(lead if lead.size else finite))
    return {"baseline": baseline, "noise_sigma": _robust_sigma(lead if lead.size else finite), "method": "leading_window_median"}


def _gaussian_sum(x: np.ndarray, params: np.ndarray, count: int) -> np.ndarray:
    fitted = np.full(x.shape, params[0], dtype=np.float64)
    for index in range(count):
        amplitude, center, sigma = params[1 + 3 * index : 4 + 3 * index]
        fitted += amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return fitted


def _peak_seeds(
    signal: np.ndarray,
    noise_sigma: float,
    detector_peaks: Sequence[Any] | None,
) -> list[tuple[float, float, float]]:
    """Return (center, amplitude, sigma) seeds without changing detector behavior."""

    if detector_peaks is None:
        peaks, properties = find_peaks(signal, height=3.0 * noise_sigma, prominence=3.0 * noise_sigma, distance=3)
        widths = peak_widths(signal, peaks, rel_height=0.5)[0] if peaks.size else np.empty(0)
        seeds = [
            (float(p), float(signal[p]), float(np.clip(w / 2.355, 0.8, 30.0)))
            for p, w in zip(peaks, widths)
        ]
        return sorted(seeds, key=lambda item: item[1], reverse=True)

    seeds: list[tuple[float, float, float]] = []
    for item in detector_peaks:
        if isinstance(item, dict):
            center = float(item.get("peak_bin", item.get("center_bin", np.nan)))
            width = float(item.get("width_bins", item.get("fwhm_bins", 6.0)))
        else:
            center, width = float(item), 6.0
        index = int(round(center))
        if not np.isfinite(center) or not 0 <= index < signal.size:
            continue
        amplitude = float(signal[index])
        if amplitude >= 3.0 * noise_sigma:
            seeds.append((center, amplitude, float(np.clip(width / 2.355, 0.8, 30.0))))
    # Keep detector peaks as the first-choice initializers, then add distinct
    # raw-signal maxima so the research model-order cap is not silently equal
    # to the number of detector components. These extras remain heuristic seeds.
    raw_peaks, _ = find_peaks(signal, height=3.0 * noise_sigma,
                              prominence=3.0 * noise_sigma, distance=3)
    raw_widths = peak_widths(signal, raw_peaks, rel_height=0.5)[0] if raw_peaks.size else np.empty(0)
    raw_candidates = [
        (float(p), float(signal[p]), float(np.clip(w / 2.355, 0.8, 30.0)))
        for p, w in zip(raw_peaks, raw_widths)
    ]
    for candidate in sorted(raw_candidates, key=lambda item: item[1], reverse=True):
        if all(abs(candidate[0] - kept[0]) >= 5.0 for kept in seeds):
            seeds.append(candidate)
    seeds.sort(key=lambda item: item[1], reverse=True)
    unique: list[tuple[float, float, float]] = []
    for seed in seeds:
        if all(abs(seed[0] - kept[0]) >= 5.0 for kept in unique):
            unique.append(seed)
    return unique


def _check_model_admissibility(model: dict[str, Any], noise_sigma: float) -> tuple[bool, str]:
    """Empirical shape screens, not calibrated physical-return resolution."""
    if not model.get("success", False):
        return False, "optimizer_failed"
    if model["chosen_k"] == 0:
        return True, "baseline_only"
    if model.get("boundary_hit_components"):
        return False, "boundary_hit"
    if any(c["amplitude"] < 2.0 * noise_sigma for c in model["components"]):
        return False, "sub_noise_amplitude"
    if any(p["center_distance_bins"] < p["combined_sigma_bins"]
           for p in model["pairwise_component_overlap"]):
        return False, "excessive_component_overlap"
    return True, "admissible"


def fit_gaussian_components(
    waveform: Sequence[float],
    baseline: float,
    noise_sigma: float,
    detector_peaks: Sequence[Any] | None = None,
    max_components: int = 5,
    fit_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Fit baseline-only and 1..K Gaussian models on identical selected bins.

    The supplied global baseline is fixed. BIC uses ``n*log(RSS/n) + p*log(n)``
    with ``p=1+3*K`` (Gaussian terms and unknown residual variance), conditional
    on that baseline. A six-unit BIC improvement per added component is required
    among admissible models. Raw BIC selection and rejected models are retained.
    The components describe
    waveform shape; they are not validated physical returns.
    """

    y = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if y.size < 5 or not np.all(np.isfinite(y)):
        raise ValueError("waveform must contain at least five finite samples")
    configured_cap = int(max_components)
    if configured_cap < 1:
        raise ValueError("max_components must be at least one")
    sigma_noise = float(noise_sigma) if np.isfinite(noise_sigma) and noise_sigma > 0 else _robust_sigma(y)
    base = float(baseline) if np.isfinite(baseline) else float(np.median(y[: min(256, y.size)]))
    x = np.arange(y.size, dtype=np.float64)
    if fit_indices is None:
        indices = np.arange(y.size, dtype=np.int64)
    else:
        indices = np.asarray(fit_indices, dtype=np.int64).reshape(-1)
        if (indices.size < 5 or np.any(indices < 0) or np.any(indices >= y.size)
                or np.unique(indices).size != indices.size):
            raise ValueError("fit_indices must contain at least five unique in-range bins")
    x_fit, y_fit = x[indices], y[indices]
    seed_signal = y - base
    seed_signal[np.setdiff1d(np.arange(y.size), indices)] = 0.0
    seeds = _peak_seeds(seed_signal, sigma_noise, detector_peaks)
    n = int(indices.size)

    def fit_metrics(params: np.ndarray, count: int) -> dict[str, Any]:
        fitted = _gaussian_sum(x, params, count)
        fit_residual = y_fit - _gaussian_sum(x_fit, params, count)
        residual = y - fitted
        rss = max(float(np.dot(fit_residual, fit_residual)), np.finfo(float).tiny)
        parameter_count = 1 + 3 * count
        return {
            "success": True,
            "fit_status": "ok" if count else "baseline_only",
            "chosen_k": count,
            "k": count,
            "baseline": float(params[0]),
            "baseline_fixed": True,
            "noise_sigma": sigma_noise,
            "rss": rss,
            "rmse": float(np.sqrt(rss / n)),
            "full_window_rmse": float(np.sqrt(np.mean(residual**2))),
            "bic": float(n * np.log(rss / n) + parameter_count * np.log(n)),
            "fit_sample_count": n,
            "fit_sample_fraction": float(n / y.size),
            "parameter_count": parameter_count,
            "fit": fitted,
            "residual": residual,
        }

    def bound_state(value: float, low: float, high: float) -> dict[str, Any]:
        tolerance = max(1e-8, 1e-6 * (high - low))
        return {
            "at_lower": bool(value - low <= tolerance),
            "at_upper": bool(high - value <= tolerance),
            "distance_to_lower": float(value - low),
            "distance_to_upper": float(high - value),
        }

    baseline0 = base
    k0 = fit_metrics(np.asarray([baseline0]), 0)
    k0.update({
        "optimizer_status": "fixed_baseline",
        "optimizer_success": True,
        "boundary_diagnostics": {"baseline": {"fixed": True}},
        "boundary_hit_components": [],
        "components": [],
        "attempts": [],
    })
    model_rows: list[dict[str, Any]] = [k0]
    selectable = [k0]
    search_cap = min(configured_cap, len(seeds))
    max_amp = max(float(np.max(y_fit - base)), sigma_noise) * 2.0 + sigma_noise

    for count in range(1, search_cap + 1):
        chosen_seeds = seeds[:count]
        lower = [base]
        upper = [base]
        for center, _, width in chosen_seeds:
            radius = max(5.0, 2.0 * width)
            lower.extend([0.0, max(0.0, center - radius), 0.6])
            upper.extend([max_amp, min(float(x.size - 1), center + radius), 80.0])
        lower_array, upper_array = np.asarray(lower), np.asarray(upper)
        attempts: list[dict[str, Any]] = []
        successful: list[tuple[float, dict[str, Any], np.ndarray, dict[str, Any]]] = []

        # A nominal start and one small deterministic center/width perturbation.
        for start_index, offset in enumerate((0.0, 0.2)):
            initial = [base]
            for center, amplitude, width in chosen_seeds:
                radius = max(5.0, 2.0 * width)
                center0 = float(np.clip(center + offset * width, max(0.0, center - radius) + 1e-8,
                                         min(float(x.size - 1), center + radius) - 1e-8))
                initial.extend([amplitude, center0, width])
            initial_array = np.asarray(initial, dtype=np.float64)

            def residuals(params: np.ndarray) -> np.ndarray:
                return _gaussian_sum(x_fit, np.r_[base, params], count) - y_fit

            try:
                result = least_squares(
                    residuals, initial_array[1:], bounds=(lower_array[1:], upper_array[1:]), max_nfev=2000,
                )
                params = np.r_[base, np.asarray(result.x, dtype=np.float64)]
                finite_params = params.shape == initial_array.shape and np.isfinite(params).all()
                rss = float(np.sum(residuals(params[1:]) ** 2)) if finite_params else None
                attempt = {
                    "start_index": start_index, "optimizer_success": bool(result.success),
                    "optimizer_status": int(result.status), "message": str(result.message),
                    "nfev": int(result.nfev), "rss": rss,
                }
                if bool(result.success) and finite_params and rss is not None and np.isfinite(rss):
                    metrics = fit_metrics(params, count)
                    boundary = {"baseline": {"fixed": True}}
                    component_boundaries = []
                    components = []
                    for component_index in range(count):
                        amplitude, center, width = params[1 + 3 * component_index:4 + 3 * component_index]
                        bounds = {
                            "amplitude": bound_state(amplitude, lower_array[1 + 3 * component_index], upper_array[1 + 3 * component_index]),
                            "center_bin": bound_state(center, lower_array[2 + 3 * component_index], upper_array[2 + 3 * component_index]),
                            "sigma_bins": bound_state(width, lower_array[3 + 3 * component_index], upper_array[3 + 3 * component_index]),
                        }
                        hit = any(v["at_lower"] or v["at_upper"] for v in bounds.values())
                        component_boundaries.append({"component_index": component_index + 1, **bounds, "boundary_hit": hit})
                        components.append({"center_bin": float(center), "amplitude": float(amplitude),
                                           "sigma_bins": float(width), "fwhm_bins": float(2.354820045 * width),
                                           "boundary_hit": hit})
                    boundary["components"] = component_boundaries
                    metrics.update({
                        "optimizer_status": int(result.status), "optimizer_success": True,
                        "optimizer_message": str(result.message), "nfev": int(result.nfev),
                        "boundary_diagnostics": boundary,
                        "boundary_hit_components": [int(i + 1) for i, item in enumerate(component_boundaries)
                                                     if item["boundary_hit"]],
                        "components": sorted(components, key=lambda item: item["center_bin"]),
                    })
                    successful.append((metrics["bic"], metrics, params, boundary))
                attempts.append(attempt)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
                attempts.append({"start_index": start_index, "optimizer_success": False,
                                 "optimizer_status": None, "message": f"{type(exc).__name__}: {exc}",
                                 "nfev": None, "rss": None})

        if successful:
            _, best_model, _, _ = min(successful, key=lambda item: item[0])
            best_model["attempts"] = attempts
            model_rows.append(best_model)
            selectable.append(best_model)
        else:
            model_rows.append({
                "chosen_k": count, "success": False, "optimizer_success": False,
                "k": count,
                "fit_status": "optimizer_failed", "optimizer_status": None,
                "rss": None, "rmse": None, "bic": None, "parameter_count": 1 + 3 * count,
                "fit_sample_count": n, "fit_sample_fraction": float(n / y.size),
                "attempts": attempts, "components": [], "boundary_diagnostics": {},
                "boundary_hit_components": [],
            })

    for model in model_rows:
        component_overlap = []
        components = model.get("components", [])
        for left_index, left in enumerate(components):
            for right_index in range(left_index + 1, len(components)):
                right = components[right_index]
                center_distance = abs(float(right["center_bin"]) - float(left["center_bin"]))
                width_sum = float(left["sigma_bins"]) + float(right["sigma_bins"])
                component_overlap.append({
                    "component_pair": [left_index + 1, right_index + 1],
                    "center_distance_bins": center_distance,
                    "sum_sigma_bins": width_sum,
                    "combined_sigma_bins": float(np.hypot(left["sigma_bins"], right["sigma_bins"])),
                    "distance_over_sum_sigma": center_distance / width_sum if width_sum > 0 else None,
                    "strong_overlap_screen": bool(width_sum > 0 and center_distance < 0.5 * width_sum),
                })
        model["pairwise_component_overlap"] = component_overlap
        model["is_admissible"], model["admissibility_reason"] = _check_model_admissibility(model, sigma_noise)

    selectable.sort(key=lambda item: item["bic"])
    raw_best = selectable[0]
    valid_models = sorted((m for m in selectable if m["is_admissible"]), key=lambda m: m["chosen_k"])
    selected = valid_models[0]
    for candidate in valid_models[1:]:
        added = candidate["chosen_k"] - selected["chosen_k"]
        if selected["bic"] - candidate["bic"] >= 6.0 * added:
            selected = candidate
    best = selected.copy()
    competitors = [m for m in valid_models if m["chosen_k"] != best["chosen_k"]]
    competitor = min(competitors, key=lambda m: m["bic"]) if competitors else None
    best["delta_bic"] = float(competitor["bic"] - best["bic"]) if competitor else None
    best["raw_bic_chosen_k"] = int(raw_best["chosen_k"])
    best["unconstrained_bic_k"] = int(raw_best["chosen_k"])
    best["bic_improvement_per_component"] = 6.0
    best["stable_order"] = bool(search_cap == 0 or (
        best["delta_bic"] is not None and best["delta_bic"] >= 6.0
        and best["chosen_k"] < configured_cap))
    best["order_stability_scope"] = "conditional_on_seed_gate_and_tested_admissible_models"
    best["selected_parameter_boundary_hit"] = bool(best["boundary_hit_components"])
    best["configured_cap"] = configured_cap
    best["available_seed_cap"] = len(seeds)
    best["searched_k_cap"] = search_cap
    best["seed_limited"] = len(seeds) < configured_cap
    best["selected_at_boundary"] = best["selected_parameter_boundary_hit"]
    best["selected_at_configured_cap"] = bool(best["chosen_k"] == configured_cap and len(seeds) >= configured_cap)
    best["at_search_limit"] = bool(best["chosen_k"] > 0 and best["chosen_k"] == search_cap)
    best["pairwise_component_overlap"] = next(
        model["pairwise_component_overlap"] for model in model_rows
        if model.get("chosen_k") == best["chosen_k"]
    )
    if best["chosen_k"] == 0:
        best["fit_status"] = "no_detectable_peak" if not seeds else "no_components_selected"
    best["models"] = model_rows
    return best


def morphology_stats(values: Sequence[float]) -> dict[str, Any]:
    """Report integer rails and suspected flat tops without asserting saturation."""
    values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("waveform contains no finite samples")
    base = float(np.median(finite_values))
    peak = int(np.argmax(np.where(np.isfinite(values), values, -np.inf)))
    peak_value = float(values[peak])
    excess = peak_value - base
    above = np.isfinite(values) & (values >= base + 0.5 * excess)
    lo = hi = peak
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi + 1 < values.size and above[hi + 1]:
        hi += 1
    flat_lo = flat_hi = peak
    while flat_lo > 0 and values[flat_lo - 1] == peak_value:
        flat_lo -= 1
    while flat_hi + 1 < values.size and values[flat_hi + 1] == peak_value:
        flat_hi += 1
    run = int(flat_hi - flat_lo + 1)
    return {
        "median_baseline_counts": base, "peak_sample_index": peak,
        "peak_excess_counts": excess, "half_height_width_samples": int(hi - lo + 1),
        "digital_low_rail_hits": int(np.sum(values == np.iinfo(np.int16).min)),
        "digital_high_rail_hits": int(np.sum(values == np.iinfo(np.int16).max)),
        "peak_flat_top_samples": run,
        "suspected_flat_top_clipping": bool(
            run >= 3 and peak_value.is_integer() and excess > 5.0 * np.std(finite_values)),
    }


def residual_diagnostics(
    residual: Sequence[float],
    noise_sigma: float,
    sample_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Summarize residual shape without assuming independent samples."""

    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    indices = (np.arange(values.size, dtype=np.int64) if sample_indices is None
               else np.asarray(sample_indices, dtype=np.int64).reshape(-1))
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= values.size):
        raise ValueError("sample_indices must be a non-empty in-range selection")
    selected = values[indices]
    finite = np.isfinite(selected)
    selected, indices = selected[finite], indices[finite]
    if not selected.size:
        raise ValueError("selected residual samples contain no finite values")
    adjacent = np.diff(indices) == 1
    left, right = selected[:-1][adjacent], selected[1:][adjacent]
    autocorrelation = None
    if left.size >= 2 and np.std(left) > 0 and np.std(right) > 0:
        autocorrelation = float(np.corrcoef(left, right)[0, 1])
    sigma = float(noise_sigma) if np.isfinite(noise_sigma) and noise_sigma > 0 else _robust_sigma(selected)
    centered = selected - float(np.mean(selected))
    std = float(np.std(selected))
    skew = float(np.mean(centered**3) / std**3) if std > 0 else 0.0
    kurtosis = float(np.mean(centered**4) / std**4 - 3.0) if std > 0 else 0.0
    return {
        "sample_count": int(selected.size), "mean_counts": float(np.mean(selected)),
        "std_counts": std, "median_absolute_counts": float(np.median(np.abs(selected))),
        "rmse_counts": float(np.sqrt(np.mean(selected**2))),
        "rmse_over_h5_noise_sigma": float(np.sqrt(np.mean(selected**2)) / sigma),
        "fraction_abs_over_3_h5_sigma": float(np.mean(np.abs(selected) > 3.0 * sigma)),
        "skewness": skew, "excess_kurtosis": kurtosis,
        "lag1_autocorrelation_adjacent_bins": autocorrelation,
        "adjacent_sign_change_fraction": (
            float(np.mean((left < 0) != (right < 0))) if left.size else None
        ),
        "adjacent_pairs": int(left.size),
    }
