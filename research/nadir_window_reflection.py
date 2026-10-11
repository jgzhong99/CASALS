"""Bounded waveform diagnostics for a nadir-window reflection hypothesis."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import pandas as pd

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

    # The median is computed over threshold exceedances only; no-hit bins stay NaN.
    hit_values = np.where(detected, z, np.inf)
    sorted_hits = np.sort(hit_values, axis=0)
    median_rank = np.maximum(count - 1, 0) // 2
    median_z = np.take_along_axis(sorted_hits, median_rank[None, :, :], axis=0)[0]
    median_z = np.where(count > 0, median_z, np.nan)
    typical_noise = np.nanmedian(noise, axis=0)
    median_excess = median_z * typical_noise[:, None]

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
