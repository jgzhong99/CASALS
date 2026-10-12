"""Bounded waveform diagnostics for a nadir-window reflection hypothesis."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from casals_l1b.waveform import build_record_index_grid, infer_waveform_record_axis, read_waveform_records


def read_sweep_block(
    h5: h5py.File,
    waveform_name: str,
    start_sweep: int,
    stop_sweep: int,
) -> tuple[np.ndarray, Any, np.ndarray]:
    """Read only [start_sweep, stop_sweep) and return sweep x track x bin data."""

    index = build_record_index_grid(h5)
    start, stop = int(start_sweep), int(stop_sweep)
    if start < 0 or stop > index.n_sweeps or start >= stop:
        raise ValueError(f"invalid sweep block [{start}, {stop}) for {index.n_sweeps} sweeps")
    record_ids = np.asarray(index.record_index_grid[start:stop], dtype=np.int64)
    if np.any(record_ids < 0):
        raise ValueError("selected sweep block contains missing sweep/track records")
    dataset = h5[waveform_name]
    axis = infer_waveform_record_axis(dataset, index.n_records)
    flat = read_waveform_records(dataset, axis, record_ids.reshape(-1))
    return flat.reshape(stop - start, index.n_tracks, -1), index, record_ids


def score_track_bins(
    waveforms: np.ndarray,
    background_mean: np.ndarray,
    background_std: np.ndarray,
    refh: np.ndarray,
    *,
    early_bins: int = 1200,
    threshold_sigma: float = 5.0,
) -> pd.DataFrame:
    """Summarize recurrence, excess amplitude, early prior, and Refh association."""

    rx = np.asarray(waveforms, dtype=np.float32)
    if rx.ndim != 3:
        raise ValueError("waveforms must have shape (sweeps, tracks, bins)")
    n_sweeps, n_tracks, n_bins = rx.shape
    limit = min(int(early_bins), n_bins)
    if limit < 1:
        raise ValueError("early_bins must be positive")
    mean = np.asarray(background_mean, dtype=np.float32)
    noise = np.asarray(background_std, dtype=np.float32)
    scene = np.asarray(refh, dtype=np.float64)
    if mean.shape != (n_sweeps, n_tracks) or noise.shape != mean.shape or scene.shape != mean.shape:
        raise ValueError("background fields and Refh must match the sweep x track dimensions")

    noise = np.where(np.isfinite(noise) & (noise > 0), noise, np.nan)
    z = (rx[:, :, :limit] - mean[:, :, None]) / noise[:, :, None]
    valid = np.isfinite(z)
    detected = valid & (z >= float(threshold_sigma))
    count = detected.sum(axis=0)
    valid_sweeps = valid.sum(axis=0)
    occupancy = count / np.maximum(valid_sweeps, 1)
    # Wilson 95% intervals make the small-block recurrence denominator visible.
    # These are descriptive binomial intervals; sweep dependence is not modeled.
    wilson_z = 1.959963984540054
    wilson_denom = 1.0 + wilson_z**2 / np.maximum(valid_sweeps, 1)
    wilson_center = (occupancy + wilson_z**2 / (2.0 * np.maximum(valid_sweeps, 1))) / wilson_denom
    wilson_half = (
        wilson_z * np.sqrt(
            occupancy * (1.0 - occupancy) / np.maximum(valid_sweeps, 1)
            + wilson_z**2 / (4.0 * np.maximum(valid_sweeps, 1) ** 2)
        ) / wilson_denom
    )
    occupancy_ci_low = np.where(valid_sweeps > 0, np.clip(wilson_center - wilson_half, 0.0, 1.0), np.nan)
    occupancy_ci_high = np.where(valid_sweeps > 0, np.clip(wilson_center + wilson_half, 0.0, 1.0), np.nan)

    # The median is computed over threshold exceedances only; no-hit bins stay NaN.
    hit_values = np.where(detected, z, np.inf)
    sorted_hits = np.sort(hit_values, axis=0)
    median_rank = np.maximum(count - 1, 0) // 2
    median_z = np.take_along_axis(sorted_hits, median_rank[None, :, :], axis=0)[0]
    median_z = np.where(count > 0, median_z, np.nan)
    hit_excess_counts = np.where(detected, rx[:, :, :limit] - mean[:, :, None], np.inf)
    sorted_excess_counts = np.sort(hit_excess_counts, axis=0)
    median_excess = np.take_along_axis(sorted_excess_counts, median_rank[None, :, :], axis=0)[0]
    median_excess = np.where(count > 0, median_excess, np.nan)

    current = np.zeros((n_tracks, limit), dtype=np.uint16)
    longest = np.zeros_like(current)
    for row in detected:
        current = np.where(row, current + 1, 0).astype(np.uint16, copy=False)
        longest = np.maximum(longest, current)

    corr = np.full((n_tracks, limit), np.nan, dtype=np.float32)
    scene_assessed = np.zeros(n_tracks, dtype=bool)
    for track in range(n_tracks):
        y = scene[:, track]
        finite_y = np.isfinite(y)
        if finite_y.sum() < 10 or np.nanstd(y) < 0.05:
            continue
        scene_assessed[track] = True
        x = z[finite_y, track, :]
        centered_x = x - np.nanmean(x, axis=0)
        centered_y = y[finite_y] - np.mean(y[finite_y])
        numerator = np.nansum(centered_x * centered_y[:, None], axis=0)
        denominator = np.sqrt(np.nansum(centered_x**2, axis=0) * np.sum(centered_y**2))
        corr[track] = np.divide(numerator, denominator, out=np.full(limit, np.nan), where=denominator > 0)

    bin_index = np.broadcast_to(np.arange(limit, dtype=np.float32), (n_tracks, limit))
    early_weight = 1.0 - bin_index / max(limit - 1, 1)
    scene_independence = np.where(np.isfinite(corr), 1.0 - np.abs(corr), 0.5)
    occupancy_score = np.clip(occupancy / 0.5, 0.0, 1.0)
    amplitude_score = np.clip(np.nan_to_num(median_z, nan=0.0) / 12.0, 0.0, 1.0)
    score = 0.45 * occupancy_score + 0.35 * amplitude_score + 0.10 * early_weight + 0.10 * scene_independence

    track_grid, bin_grid = np.indices((n_tracks, limit))
    return pd.DataFrame(
        {
            "track_num": track_grid.ravel(),
            "bin": bin_grid.ravel(),
            "occupancy": occupancy.ravel(),
            "occupancy_ci95_low": occupancy_ci_low.ravel(),
            "occupancy_ci95_high": occupancy_ci_high.ravel(),
            "event_count": count.ravel(),
            "valid_sweep_count": valid_sweeps.ravel(),
            "median_excess_counts": median_excess.ravel(),
            "median_excess_sigma": median_z.ravel(),
            "early_weight": early_weight.ravel(),
            "scene_correlation": corr.ravel(),
            "scene_assessed": np.broadcast_to(scene_assessed[:, None], (n_tracks, limit)).ravel(),
            "longest_persistence_sweeps": longest.ravel(),
            "ghost_score": score.ravel(),
        }
    )


def scan_all_bins_events(
    waveforms: np.ndarray,
    background_mean: np.ndarray,
    background_std: np.ndarray,
    *,
    threshold_sigma: float = 8.0,
    min_prominence_sigma: float = 4.0,
    min_distance_bins: int = 5,
    sweep_numbers: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Find prominent standardized peaks over every RX bin in a bounded block.

    This is a sensitive Stage-A anomaly list. Its thresholds are screening
    heuristics and its events do not identify an instrument mechanism.
    """

    rx = np.asarray(waveforms, dtype=np.float32)
    mean = np.asarray(background_mean, dtype=np.float32)
    noise = np.asarray(background_std, dtype=np.float32)
    if rx.ndim != 3 or mean.shape != rx.shape[:2] or noise.shape != mean.shape:
        raise ValueError("waveforms must be sweeps x tracks x bins; background fields must match sweeps x tracks")
    if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be finite and positive")
    if not np.isfinite(min_prominence_sigma) or min_prominence_sigma < 0:
        raise ValueError("min_prominence_sigma must be finite and non-negative")
    if int(min_distance_bins) < 1:
        raise ValueError("min_distance_bins must be positive")
    sweeps = np.arange(rx.shape[0]) if sweep_numbers is None else np.asarray(sweep_numbers, dtype=np.int64)
    if sweeps.shape != (rx.shape[0],):
        raise ValueError("sweep_numbers must contain one value per waveform sweep")

    rows: list[dict[str, Any]] = []
    for sweep_offset in range(rx.shape[0]):
        for track in range(rx.shape[1]):
            if not np.isfinite(mean[sweep_offset, track]) or not np.isfinite(noise[sweep_offset, track]) or noise[sweep_offset, track] <= 0:
                continue
            z = (rx[sweep_offset, track] - mean[sweep_offset, track]) / noise[sweep_offset, track]
            finite = np.isfinite(z)
            if not finite.any():
                continue
            starts = np.flatnonzero(finite & ~np.r_[False, finite[:-1]])
            stops = np.flatnonzero(finite & ~np.r_[finite[1:], False]) + 1
            for start, stop in zip(starts, stops):
                segment = z[start:stop]
                if segment.size < 3:
                    continue
                peaks, properties = find_peaks(
                    segment, height=float(threshold_sigma), prominence=float(min_prominence_sigma),
                    distance=int(min_distance_bins),
                )
                if not peaks.size:
                    continue
                widths = peak_widths(segment, peaks, rel_height=0.5)[0]
                prominences = properties.get("prominences", np.full(peaks.size, np.nan))
                for position, local_bin in enumerate(peaks):
                    bin_index = int(start + local_bin)
                    rows.append({
                        "sweep_num": int(sweeps[sweep_offset]), "sweep_offset": int(sweep_offset),
                        "track_num": int(track), "bin": bin_index,
                        "excess_sigma": float(z[bin_index]),
                        "prominence_sigma": float(prominences[position]),
                        "width_bins": float(widths[position]),
                        "raw_count": float(rx[sweep_offset, track, bin_index]),
                        "valid_rx_samples": int(finite.sum()),
                    })
    return pd.DataFrame(rows, columns=[
        "sweep_num", "sweep_offset", "track_num", "bin", "excess_sigma",
        "prominence_sigma", "width_bins", "raw_count", "valid_rx_samples",
    ])


def extract_candidate_bands(
    scores: pd.DataFrame,
    *,
    occupancy_min: float = 0.20,
    median_excess_sigma_min: float = 5.0,
    score_min: float = 0.45,
) -> pd.DataFrame:
    """Merge adjacent qualifying bins within each track into candidate bands."""

    required = {"track_num", "bin", "occupancy", "median_excess_sigma", "ghost_score"}
    if not required.issubset(scores.columns):
        raise ValueError(f"scores is missing columns: {sorted(required - set(scores.columns))}")
    qualifies = (
        scores["occupancy"].ge(float(occupancy_min))
        & scores["median_excess_sigma"].ge(float(median_excess_sigma_min))
        & scores["ghost_score"].ge(float(score_min))
    )
    rows: list[dict[str, Any]] = []
    selected = scores.loc[qualifies].sort_values(["track_num", "bin"], kind="stable")
    for track, group in selected.groupby("track_num", sort=True):
        bins = group["bin"].to_numpy(dtype=int)
        split = np.flatnonzero(np.diff(bins) > 1) + 1
        bounds = np.r_[0, split, len(group)]
        for start, stop in zip(bounds[:-1], bounds[1:]):
            run = group.iloc[start:stop]
            rows.append(
                {
                    "track_num": int(track),
                    "start_bin": int(run["bin"].min()),
                    "end_bin": int(run["bin"].max()),
                    "n_bins": int(len(run)),
                    "mean_occupancy": float(run["occupancy"].mean()),
                    "max_occupancy": float(run["occupancy"].max()),
                    "occupancy_ci95_low_min": float(run["occupancy_ci95_low"].min()) if "occupancy_ci95_low" in run else None,
                    "occupancy_ci95_high_max": float(run["occupancy_ci95_high"].max()) if "occupancy_ci95_high" in run else None,
                    "median_excess_counts": float(run["median_excess_counts"].median()),
                    "median_excess_sigma": float(run["median_excess_sigma"].median()),
                    "mean_ghost_score": float(run["ghost_score"].mean()),
                    "mean_scene_correlation": float(run["scene_correlation"].mean()),
                    "scene_assessed": bool(run["scene_assessed"].all()) if "scene_assessed" in run else False,
                    "max_persistence_sweeps": int(run["longest_persistence_sweeps"].max()) if "longest_persistence_sweeps" in run else 0,
                }
            )
    return pd.DataFrame(rows, columns=[
        "track_num", "start_bin", "end_bin", "n_bins", "mean_occupancy", "max_occupancy",
        "occupancy_ci95_low_min", "occupancy_ci95_high_max",
        "median_excess_counts", "median_excess_sigma", "mean_ghost_score",
        "mean_scene_correlation", "scene_assessed", "max_persistence_sweeps",
    ])


def flag_bins(n_bins: int, intervals: Iterable[Any]) -> np.ndarray:
    """Return a Boolean mask for inclusive (start_bin, end_bin) intervals."""

    flags = np.zeros(int(n_bins), dtype=bool)
    for interval in intervals:
        if isinstance(interval, dict):
            start, stop = int(interval["start_bin"]), int(interval["end_bin"])
        else:
            start, stop = int(interval[0]), int(interval[1])
        if stop < start:
            raise ValueError("interval end_bin must be >= start_bin")
        flags[max(0, start) : min(flags.size, stop + 1)] = True
    return flags


def mask_bins(waveform: Sequence[float], intervals: Iterable[Any]) -> np.ndarray:
    """Copy a waveform and mark flagged candidate bins as NaN."""

    result = np.asarray(waveform, dtype=np.float64).copy()
    result[flag_bins(result.size, intervals)] = np.nan
    return result


def subtract_template(waveform: Sequence[float], template: Sequence[float], scale: float = 1.0) -> np.ndarray:
    """Subtract a same-shape exploratory template; no automatic correction is implied."""

    y = np.asarray(waveform, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    if y.shape != t.shape:
        raise ValueError(f"template shape {t.shape} does not match waveform shape {y.shape}")
    if not np.isfinite(scale):
        raise ValueError("scale must be finite")
    return y - float(scale) * t
