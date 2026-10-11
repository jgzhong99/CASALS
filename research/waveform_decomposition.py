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
    seeds.sort(key=lambda item: item[1], reverse=True)
    unique: list[tuple[float, float, float]] = []
    for seed in seeds:
        if all(abs(seed[0] - kept[0]) >= 5.0 for kept in unique):
            unique.append(seed)
    return unique


def fit_gaussian_components(
    waveform: Sequence[float],
    baseline: float,
    noise_sigma: float,
    detector_peaks: Sequence[Any] | None = None,
    max_components: int = 5,
) -> dict[str, Any]:
    """Fit 0..K Gaussian components and choose K by BIC.

    The fit is an exploratory waveform representation. Its components are not
    validated physical returns or official geolocated points.
    """

    y = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if y.size < 5 or not np.all(np.isfinite(y)):
        raise ValueError("waveform must contain at least five finite samples")
    if int(max_components) < 1:
        raise ValueError("max_components must be at least one")
    sigma_noise = float(noise_sigma) if np.isfinite(noise_sigma) and noise_sigma > 0 else _robust_sigma(y)
    base = float(baseline) if np.isfinite(baseline) else float(np.median(y[: min(256, y.size)]))
    x = np.arange(y.size, dtype=np.float64)
    seeds = _peak_seeds(y - base, sigma_noise, detector_peaks)

    if not seeds:
        fitted = np.full_like(y, base)
        residual = y - fitted
        return {
            "success": True,
            "fit_status": "no_detectable_peak",
            "chosen_k": 0,
            "baseline": base,
            "noise_sigma": sigma_noise,
            "bic": None,
            "delta_bic": None,
            "at_search_limit": False,
            "stable_order": False,
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "fit": fitted,
            "residual": residual,
            "components": [],
            "models": [],
        }

    n = y.size
    max_amp = max(float(np.max(y - base)), sigma_noise) * 2.0 + sigma_noise
    lower_baseline = base - 3.0 * sigma_noise
    upper_baseline = base + 3.0 * sigma_noise
    candidates: list[dict[str, Any]] = []
    for count in range(1, min(int(max_components), len(seeds)) + 1):
        chosen_seeds = seeds[:count]
        initial = [base]
        lower = [lower_baseline]
        upper = [upper_baseline]
        for center, amplitude, width in chosen_seeds:
            radius = max(5.0, 2.0 * width)
            initial.extend([amplitude, center, width])
            lower.extend([0.0, max(0.0, center - radius), 0.6])
            upper.extend([max_amp, min(float(n - 1), center + radius), 80.0])

        def residuals(params: np.ndarray) -> np.ndarray:
            return _gaussian_sum(x, params, count) - y

        try:
            result = least_squares(
                residuals,
                np.asarray(initial, dtype=np.float64),
                bounds=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
                max_nfev=3000,
            )
            fitted = _gaussian_sum(x, result.x, count)
            resid = y - fitted
            rss = max(float(np.dot(resid, resid)), np.finfo(float).tiny)
            parameter_count = 1 + 3 * count
            bic = n * np.log(rss / n) + parameter_count * np.log(n)
            components = []
            for index in range(count):
                amplitude, center, width = result.x[1 + 3 * index : 4 + 3 * index]
                components.append(
                    {
                        "center_bin": float(center),
                        "amplitude": float(amplitude),
                        "sigma_bins": float(width),
                        "fwhm_bins": float(2.354820045 * width),
                    }
                )
            components.sort(key=lambda item: item["center_bin"])
            candidates.append(
                {
                    "success": bool(result.success),
                    "fit_status": "ok" if result.success else "optimizer_not_converged",
                    "chosen_k": count,
                    "baseline": float(result.x[0]),
                    "noise_sigma": sigma_noise,
                    "bic": float(bic),
                    "rmse": float(np.sqrt(np.mean(resid**2))),
                    "fit": fitted,
                    "residual": resid,
                    "components": components,
                    "models": [],
                }
            )
        except (ValueError, FloatingPointError):
            continue

    if not candidates:
        fitted = np.full_like(y, base)
        residual = y - fitted
        return {
            "success": False,
            "fit_status": "fit_failed",
            "chosen_k": 0,
            "baseline": base,
            "noise_sigma": sigma_noise,
            "bic": None,
            "delta_bic": None,
            "at_search_limit": False,
            "stable_order": False,
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "fit": fitted,
            "residual": residual,
            "components": [],
            "models": [],
        }

    candidates.sort(key=lambda item: item["bic"])
    best = candidates[0]
    best["delta_bic"] = float(candidates[1]["bic"] - best["bic"]) if len(candidates) > 1 else None
    best["stable_order"] = bool(best["success"] and best["delta_bic"] is not None and best["delta_bic"] >= 6.0)
    best["at_search_limit"] = bool(best["chosen_k"] >= min(int(max_components), len(seeds)))
    best["models"] = [
        {"k": int(item["chosen_k"]), "bic": float(item["bic"]), "rmse": float(item["rmse"])}
        for item in candidates
    ]
    return best
