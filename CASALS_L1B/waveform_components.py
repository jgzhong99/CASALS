"""Shared waveform diagnostics utilities for CASALS L1B.

Scientific scope
----------------
CASALS L1B is treated here as a geolocated waveform product. Each pulse has
one official geolocated ``refh`` point, and ``refh`` is associated with the
maximum-amplitude RX waveform bin. Secondary waveform peaks/components handled
by this module are waveform-derived diagnostics only. They are not official
returns, not georeferenced returns, and not a multi-return point cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence
import warnings

import h5py
import numpy as np
import pandas as pd

try:  # pragma: no cover - guarded at runtime
    from scipy.signal import find_peaks, medfilt, peak_widths, savgol_filter
except Exception as exc:  # pragma: no cover
    find_peaks = None
    medfilt = None
    peak_widths = None
    savgol_filter = None
    _SCIPY_IMPORT_ERROR = exc
else:  # pragma: no cover
    _SCIPY_IMPORT_ERROR = None


DEFAULT_ATTR_KEYS = (
    "n_pulses",
    "n_sweeps",
    "n_tracks",
    "n_rx_bins",
    "n_tx_bins",
    "start_utca",
    "end_utca",
)


@dataclass(frozen=True)
class SweepTrackIndex:
    """Sweep/track lookup table for waveform records.

    ``record_index_grid[sweep, track]`` stores the flat H5 pulse index, or
    ``-1`` when that sweep/track cell is missing. This structure is for
    waveform bookkeeping and diagnostics only; it does not georeference
    secondary waveform-derived components.
    """

    sweep_num: np.ndarray
    track_num: np.ndarray
    record_index_grid: np.ndarray
    n_sweeps: int
    n_tracks: int
    n_records: int
    complete_rectangular_grid: bool
    duplicate_sweep_track_cells: int
    missing_sweep_track_cells: int


RecordIndexGridInfo = SweepTrackIndex


def require_scipy() -> None:
    """Raise a clear runtime error when SciPy peak tools are unavailable."""

    if find_peaks is None or medfilt is None or peak_widths is None or savgol_filter is None:
        raise RuntimeError(
            "SciPy is required for CASALS waveform diagnostics. Please install scipy."
        ) from _SCIPY_IMPORT_ERROR


def find_dataset(h5: h5py.File, name: str) -> Optional[h5py.Dataset]:
    """Find an H5 dataset by exact root name or recursive basename match.

    This lookup is used to support diagnostic waveform workflows across H5
    files with slightly different internal paths. Any located dataset remains
    pulse-level metadata or waveform storage only; it does not georeference
    secondary waveform-derived components.
    """

    normalized = name.strip("/")
    if normalized in h5 and isinstance(h5[normalized], h5py.Dataset):
        return h5[normalized]

    matches: list[str] = []

    def visitor(path: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and path.split("/")[-1] == normalized:
            matches.append(path)

    h5.visititems(visitor)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple datasets matched basename {name!r}: {matches}")
    return h5[matches[0]]


def require_dataset(h5: h5py.File, name: str) -> h5py.Dataset:
    """Return a required dataset or raise a clear error."""

    ds = find_dataset(h5, name)
    if ds is None:
        raise KeyError(f"Required dataset {name!r} was not found in the H5 file.")
    return ds


def read_optional_1d(
    h5: h5py.File,
    name: str,
    n_expected: int,
    dtype: Any = np.float64,
) -> Optional[np.ndarray]:
    """Read an optional 1D dataset and validate its length.

    Missing datasets return ``None``. Size mismatches raise so that waveform
    records cannot be silently paired with the wrong pulse metadata.
    """

    ds = find_dataset(h5, name)
    if ds is None:
        return None
    arr = np.asarray(ds[...], dtype=dtype).reshape(-1)
    if arr.size != int(n_expected):
        raise ValueError(f"Optional dataset {name!r} has size {arr.size}, expected {int(n_expected)}.")
    return arr


def read_attrs_subset(h5: h5py.File, keys: Sequence[str] = DEFAULT_ATTR_KEYS) -> dict[str, Any]:
    """Read a small JSON-safe H5 attribute subset for diagnostics metadata."""

    out: dict[str, Any] = {}
    for key in keys:
        if key not in h5.attrs:
            continue
        value = h5.attrs[key]
        if isinstance(value, bytes):
            out[key] = value.decode("utf-8", errors="replace")
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif hasattr(value, "item"):
            try:
                out[key] = value.item()
            except Exception:
                out[key] = str(value)
        else:
            out[key] = value
    return out


def build_record_index_grid(h5: h5py.File) -> SweepTrackIndex:
    """Build ``record_index_grid[sweep, track] = flat pulse index``.

    Duplicate sweep-track cells are detected explicitly. The first flat record
    index is retained in the grid and duplicates are counted so callers can
    decide whether to stop.
    """

    sweep = np.asarray(require_dataset(h5, "sweep_num")[...], dtype=np.int64).reshape(-1)
    track = np.asarray(require_dataset(h5, "track_num")[...], dtype=np.int64).reshape(-1)
    if sweep.size != track.size:
        raise ValueError(f"sweep_num and track_num sizes differ: {sweep.size} vs {track.size}")
    if sweep.size == 0:
        raise ValueError("No waveform records were found in sweep_num/track_num.")
    if int(np.nanmin(sweep)) < 0 or int(np.nanmin(track)) < 0:
        raise ValueError("Negative sweep_num or track_num values are not supported.")

    n_records = int(sweep.size)
    n_sweeps = int(np.nanmax(sweep)) + 1
    n_tracks = int(np.nanmax(track)) + 1
    n_expected = n_sweeps * n_tracks

    flat = sweep * n_tracks + track
    unique_flat, first_indices, counts = np.unique(flat, return_index=True, return_counts=True)
    grid = np.full((n_sweeps, n_tracks), -1, dtype=np.int64)
    grid.reshape(-1)[unique_flat] = first_indices.astype(np.int64, copy=False)

    duplicate_cells = int(np.sum(counts > 1))
    missing_cells = int(n_expected - unique_flat.size)
    return SweepTrackIndex(
        sweep_num=sweep,
        track_num=track,
        record_index_grid=grid,
        n_sweeps=n_sweeps,
        n_tracks=n_tracks,
        n_records=n_records,
        complete_rectangular_grid=bool(duplicate_cells == 0 and missing_cells == 0 and n_records == n_expected),
        duplicate_sweep_track_cells=duplicate_cells,
        missing_sweep_track_cells=missing_cells,
    )


def infer_waveform_record_axis(ds: h5py.Dataset, n_records: int) -> int:
    """Infer whether waveform records are stored on axis 0 or 1."""

    if ds.ndim != 2:
        raise ValueError(f"{ds.name} must be 2D, got shape={ds.shape}")
    if ds.shape[0] == int(n_records):
        return 0
    if ds.shape[1] == int(n_records):
        return 1
    raise ValueError(f"{ds.name} shape {ds.shape} does not match n_records={n_records} on either axis.")


def read_waveform_records(ds: h5py.Dataset, record_axis: int, indices: Sequence[int] | np.ndarray) -> np.ndarray:
    """Read waveform records with sorted h5py fancy indexing, then restore order."""

    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    n_bins = int(ds.shape[1] if record_axis == 0 else ds.shape[0])
    if idx.size == 0:
        return np.empty((0, n_bins), dtype=np.float32)

    order = np.argsort(idx)
    sorted_idx = idx[order]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size, dtype=order.dtype)

    if record_axis == 0:
        values_sorted = np.asarray(ds[sorted_idx, :], dtype=np.float32)
    elif record_axis == 1:
        values_sorted = np.asarray(ds[:, sorted_idx], dtype=np.float32).T
    else:
        raise ValueError(f"record_axis must be 0 or 1, got {record_axis}")
    return values_sorted[inverse]


def read_sweep_matrix(
    h5: h5py.File,
    waveform_name: str,
    sweep_idx: int,
    index_info: SweepTrackIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one sweep into ``matrix[track, bin]``.

    The returned matrix spans all tracks declared by ``index_info.n_tracks``.
    Missing tracks are filled with ``NaN`` rows. ``valid_tracks`` contains only
    tracks present in the selected sweep, and ``record_indices_by_track`` is a
    length-``n_tracks`` array of flat pulse indices or ``-1``.
    """

    ds = require_dataset(h5, waveform_name)
    record_axis = infer_waveform_record_axis(ds, index_info.n_records)
    record_indices_by_track = np.asarray(index_info.record_index_grid[int(sweep_idx)], dtype=np.int64)
    valid_tracks = np.flatnonzero(record_indices_by_track >= 0)

    n_bins = int(ds.shape[1] if record_axis == 0 else ds.shape[0])
    matrix = np.full((index_info.n_tracks, n_bins), np.nan, dtype=np.float32)
    if valid_tracks.size:
        matrix[valid_tracks] = read_waveform_records(ds, record_axis, record_indices_by_track[valid_tracks])
    return matrix, valid_tracks, record_indices_by_track


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr.copy()
    if np.all(finite):
        return arr.copy()
    out = arr.copy()
    valid_x = np.flatnonzero(finite)
    out[~finite] = np.interp(np.flatnonzero(~finite), valid_x, out[finite])
    return out


def smooth_waveform(
    y: Sequence[float],
    method: str = "none",
    window: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Smooth a 1D waveform using ``none``, ``median``, or ``savgol``."""

    arr = np.asarray(y, dtype=np.float64)
    method_l = str(method).lower()
    if method_l == "none":
        return arr.copy()

    require_scipy()
    window = int(window)
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    if window > arr.size:
        window = arr.size if arr.size % 2 == 1 else max(1, arr.size - 1)
    if window <= 1:
        return arr.copy()

    work = _interpolate_nans(arr)
    if method_l == "median":
        return np.asarray(medfilt(work, kernel_size=window), dtype=np.float64)
    if method_l == "savgol":
        polyorder = min(int(polyorder), max(0, window - 1))
        return np.asarray(savgol_filter(work, window_length=window, polyorder=polyorder, mode="interp"), dtype=np.float64)
    raise ValueError("smoothing method must be one of: none, median, savgol")


def safe_argmax_bin(values: Sequence[float]) -> tuple[float, float]:
    """Return ``(argmax_bin, argmax_value)`` for a waveform or ``(nan, nan)``."""

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return np.nan, np.nan
    safe = np.where(np.isfinite(arr), arr, -np.inf)
    idx = int(np.argmax(safe))
    return float(idx), float(arr[idx]) if np.isfinite(arr[idx]) else np.nan


def preprocess_waveform(
    y_raw: Sequence[float],
    bg_mean: float | None,
    bg_std: float | None,
    refh_thres: float | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Preprocess one RX waveform for diagnostics.

    Supported configuration:
    - ``background_mode``: ``raw``, ``minus_bg_mean``, ``minus_refh_thres``
    - ``clip_negative_after_background``
    - ``smoothing_method``: ``none``, ``median``, ``savgol``
    - ``smoothing_window``
    - ``smoothing_polyorder``

    The result contains raw, processed, and smoothed waveforms plus the sigma
    scale used for thresholding. These are diagnostic representations only.
    """

    y_raw_arr = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    finite_raw = np.isfinite(y_raw_arr)
    if not np.any(finite_raw):
        return {
            "raw": y_raw_arr,
            "processed": y_raw_arr.copy(),
            "smoothed": y_raw_arr.copy(),
            "sigma": np.nan,
            "background_offset": np.nan,
            "valid": False,
        }

    background_mode = str(config.get("background_mode", "raw")).lower()
    if background_mode == "raw":
        background_offset = 0.0
    elif background_mode == "minus_bg_mean":
        background_offset = float(bg_mean) if bg_mean is not None and np.isfinite(bg_mean) else 0.0
    elif background_mode == "minus_refh_thres":
        background_offset = float(refh_thres) if refh_thres is not None and np.isfinite(refh_thres) else 0.0
    else:
        raise ValueError("background_mode must be one of: raw, minus_bg_mean, minus_refh_thres")

    processed = y_raw_arr - background_offset
    if bool(config.get("clip_negative_after_background", False)):
        processed = np.where(np.isfinite(processed), np.maximum(processed, 0.0), np.nan)

    smoothed = smooth_waveform(
        processed,
        method=str(config.get("smoothing_method", "none")).lower(),
        window=int(config.get("smoothing_window", 7)),
        polyorder=int(config.get("smoothing_polyorder", 2)),
    )

    sigma = float(bg_std) if bg_std is not None and np.isfinite(bg_std) and float(bg_std) > 0 else np.nan
    if not np.isfinite(sigma) or sigma <= 0:
        finite_processed = processed[np.isfinite(processed)]
        if finite_processed.size:
            med = float(np.median(finite_processed))
            sigma = float(1.4826 * np.median(np.abs(finite_processed - med)))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.nanstd(processed))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

    return {
        "raw": y_raw_arr,
        "processed": processed,
        "smoothed": smoothed,
        "sigma": sigma,
        "background_offset": background_offset,
        "valid": True,
    }


def detect_candidate_components_1d(
    y_raw: Sequence[float],
    bg_mean: float | None,
    bg_std: float | None,
    refh_thres: float | None,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Detect candidate waveform components with ``scipy.signal.find_peaks``.

    The output is a list of candidate components, not a claim of true physical
    returns. Each component is a waveform-derived diagnostic object only.
    """

    require_scipy()
    prepared = preprocess_waveform(
        y_raw=y_raw,
        bg_mean=bg_mean,
        bg_std=bg_std,
        refh_thres=refh_thres,
        config=config,
    )
    if not prepared["valid"]:
        return []

    raw = np.asarray(prepared["raw"], dtype=np.float64)
    processed = np.asarray(prepared["processed"], dtype=np.float64)
    smoothed = np.asarray(prepared["smoothed"], dtype=np.float64)
    sigma = float(prepared["sigma"])

    if not np.any(np.isfinite(smoothed)):
        return []

    work = np.where(np.isfinite(smoothed), smoothed, -np.inf)
    min_height_sigma = float(config.get("min_height_sigma", 3.0))
    min_prominence_sigma = float(config.get("min_prominence_sigma", 3.0))
    min_width_bins = float(config.get("min_width_bins", 1.0))
    max_width_bins = config.get("max_width_bins")
    min_distance_bins = max(1, int(config.get("min_distance_bins", 3)))

    width_arg: tuple[float, float | None] | float
    if max_width_bins is None:
        width_arg = max(min_width_bins, 0.0)
    else:
        width_arg = (max(min_width_bins, 0.0), float(max_width_bins))

    peaks, properties = find_peaks(
        work,
        height=max(0.0, min_height_sigma * sigma),
        prominence=max(0.0, min_prominence_sigma * sigma),
        width=width_arg,
        distance=min_distance_bins,
    )
    if peaks.size == 0:
        return []

    widths, width_heights, left_ips, right_ips = peak_widths(work, peaks, rel_height=0.5)
    prominences = np.asarray(properties.get("prominences", np.full(peaks.size, np.nan)), dtype=np.float64)
    left_bases = np.asarray(properties.get("left_bases", np.full(peaks.size, -1)), dtype=np.int64)
    right_bases = np.asarray(properties.get("right_bases", np.full(peaks.size, -1)), dtype=np.int64)

    order_prom = np.argsort(-np.nan_to_num(prominences, nan=-np.inf), kind="stable")
    order_amp = np.argsort(
        -np.nan_to_num(np.where(np.isfinite(processed[peaks]), processed[peaks], np.nan), nan=-np.inf),
        kind="stable",
    )
    rank_by_prominence = np.empty_like(order_prom)
    rank_by_prominence[order_prom] = np.arange(1, peaks.size + 1)
    rank_by_amplitude = np.empty_like(order_amp)
    rank_by_amplitude[order_amp] = np.arange(1, peaks.size + 1)

    keep_n = int(config.get("max_candidate_components_per_pulse", config.get("max_components_per_pulse", peaks.size)))
    keep_idx = order_prom[:keep_n]

    components: list[dict[str, Any]] = []
    edge_exclusion = int(config.get("edge_exclusion_bins", 0))
    for idx in keep_idx:
        peak_bin = int(peaks[idx])
        left = float(left_ips[idx])
        right = float(right_ips[idx])
        lo = max(0, int(np.floor(left)))
        hi = min(processed.size - 1, int(np.ceil(right)))
        area_processed = float(np.nansum(np.where(np.isfinite(processed[lo : hi + 1]), np.maximum(processed[lo : hi + 1], 0.0), 0.0)))
        amplitude_processed = float(processed[peak_bin]) if np.isfinite(processed[peak_bin]) else np.nan
        amplitude_raw = float(raw[peak_bin]) if np.isfinite(raw[peak_bin]) else np.nan
        components.append(
            {
                "peak_bin": peak_bin,
                "amplitude_raw": amplitude_raw,
                "amplitude_processed": amplitude_processed,
                "prominence": float(prominences[idx]),
                "prominence_sigma": float(prominences[idx] / sigma) if np.isfinite(prominences[idx]) and sigma > 0 else np.nan,
                "width_bins": float(widths[idx]),
                "left_ips": left,
                "right_ips": right,
                "left_base": int(left_bases[idx]),
                "right_base": int(right_bases[idx]),
                "width_height": float(width_heights[idx]),
                "area_processed": area_processed,
                "rank_by_prominence": int(rank_by_prominence[idx]),
                "rank_by_amplitude": int(rank_by_amplitude[idx]),
                "is_edge_peak": bool(peak_bin < edge_exclusion or peak_bin > raw.size - 1 - edge_exclusion),
                "sigma": sigma,
                "background_offset": float(prepared["background_offset"]),
            }
        )
    components.sort(key=lambda item: (item["rank_by_prominence"], item["rank_by_amplitude"]))
    return components


def classify_prominent_components(components: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Classify candidate components into main/prominent/secondary diagnostic roles.

    Prominent-component classification is deliberately stricter than candidate
    detection. This prevents the raw candidate count from being misread as a
    true return count.
    """

    items = [dict(component) for component in components]
    if not items:
        return items

    main_idx = int(np.argmin([item["rank_by_prominence"] for item in items]))
    main_peak = items[main_idx]
    main_prominence = float(main_peak.get("prominence", np.nan))
    main_bin = float(main_peak.get("peak_bin", np.nan))

    min_prominent_sigma = float(config.get("min_prominent_prominence_sigma", 5.0))
    min_relative_to_main = float(config.get("min_prominent_relative_to_main", 0.20))
    min_secondary_separation = float(config.get("min_secondary_separation_bins", 8))
    min_width_bins = float(config.get("min_width_bins", 1.0))
    max_width_bins = config.get("max_width_bins")
    edge_exclusion_bins = int(config.get("edge_exclusion_bins", 0))
    reject_edge = bool(config.get("exclude_edge_components", True))

    for idx, item in enumerate(items):
        prominence = float(item.get("prominence", np.nan))
        prominence_sigma = float(item.get("prominence_sigma", np.nan))
        peak_bin = float(item.get("peak_bin", np.nan))
        width_bins = float(item.get("width_bins", np.nan))
        relative_to_main = prominence / main_prominence if np.isfinite(prominence) and np.isfinite(main_prominence) and main_prominence > 0 else np.nan
        bin_separation = abs(peak_bin - main_bin) if np.isfinite(peak_bin) and np.isfinite(main_bin) else np.nan

        rejected_reasons: list[str] = []
        is_main = idx == main_idx
        if not np.isfinite(prominence_sigma) or prominence_sigma < min_prominent_sigma:
            rejected_reasons.append("low_absolute_prominence")
        if not is_main and (not np.isfinite(relative_to_main) or relative_to_main < min_relative_to_main):
            rejected_reasons.append("low_relative_prominence_to_main")
        if not is_main and (not np.isfinite(bin_separation) or bin_separation < min_secondary_separation):
            rejected_reasons.append("too_close_to_main")
        if np.isfinite(width_bins) and width_bins < min_width_bins:
            rejected_reasons.append("too_narrow")
        if max_width_bins is not None and np.isfinite(width_bins) and width_bins > float(max_width_bins):
            rejected_reasons.append("too_wide")
        if reject_edge and bool(item.get("is_edge_peak", False)):
            rejected_reasons.append(f"edge_exclusion_{edge_exclusion_bins}bins")

        is_prominent = is_main or len(rejected_reasons) == 0
        is_valid_secondary = (not is_main) and is_prominent
        item["relative_prominence_to_main"] = relative_to_main
        item["bin_separation_from_main"] = bin_separation
        item["is_main_component"] = bool(is_main)
        item["is_prominent_component"] = bool(is_prominent)
        item["is_valid_secondary_candidate"] = bool(is_valid_secondary)
        item["rejected_reason"] = "" if is_prominent else ";".join(rejected_reasons)
    return items


def summarize_pulse_waveform_features(
    components: Sequence[Mapping[str, Any]],
    raw_argmax_bin: float | int | None,
    corrected_argmax_bin: float | int | None,
    refh_amp: float | None,
    refh_snr: float | None,
    good_snr: bool | int | None,
    bg_mean: float | None,
    bg_std: float | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize pulse-level waveform features for refh-quality analysis."""

    items = [dict(component) for component in components]
    n_candidate = int(len(items))
    main_components = [item for item in items if item.get("is_main_component", False)]
    main = main_components[0] if main_components else (items[0] if items else None)
    prominent = [item for item in items if item.get("is_prominent_component", False)]
    secondaries = [item for item in items if item.get("is_valid_secondary_candidate", False)]
    secondary = secondaries[0] if secondaries else None

    raw_argmax = float(raw_argmax_bin) if raw_argmax_bin is not None and np.isfinite(raw_argmax_bin) else np.nan
    corrected_argmax = float(corrected_argmax_bin) if corrected_argmax_bin is not None and np.isfinite(corrected_argmax_bin) else np.nan
    refh_snr_value = float(refh_snr) if refh_snr is not None and np.isfinite(refh_snr) else np.nan
    low_snr_flag = bool((good_snr is not None and not bool(good_snr)) or (np.isfinite(refh_snr_value) and refh_snr_value < 3.0))

    main_peak_bin = float(main["peak_bin"]) if main else np.nan
    main_peak_prominence = float(main["prominence"]) if main else np.nan
    main_peak_width_bins = float(main["width_bins"]) if main else np.nan
    main_peak_area = float(main["area_processed"]) if main else np.nan
    main_peak_amp = float(main["amplitude_raw"]) if main else np.nan
    secondary_peak_bin = float(secondary["peak_bin"]) if secondary else np.nan
    secondary_amp = float(secondary["amplitude_raw"]) if secondary else np.nan
    secondary_prominence = float(secondary["prominence"]) if secondary else np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        secondary_to_main_amp_ratio = secondary_amp / main_peak_amp if np.isfinite(secondary_amp) and np.isfinite(main_peak_amp) and main_peak_amp != 0 else np.nan
        secondary_to_main_prominence_ratio = secondary_prominence / main_peak_prominence if np.isfinite(secondary_prominence) and np.isfinite(main_peak_prominence) and main_peak_prominence != 0 else np.nan

    abs_main_minus_raw = abs(main_peak_bin - raw_argmax) if np.isfinite(main_peak_bin) and np.isfinite(raw_argmax) else np.nan
    abs_corrected_minus_raw = abs(corrected_argmax - raw_argmax) if np.isfinite(corrected_argmax) and np.isfinite(raw_argmax) else np.nan

    broad_main_width_bins = float(config.get("broad_main_width_bins", 30.0))
    weak_main_prominence_sigma = float(config.get("weak_main_prominence_sigma", 4.0))
    sigma_value = float(main["sigma"]) if main and np.isfinite(main.get("sigma", np.nan)) else (float(bg_std) if bg_std is not None and np.isfinite(bg_std) and float(bg_std) > 0 else np.nan)
    main_prominence_sigma = main_peak_prominence / sigma_value if np.isfinite(main_peak_prominence) and np.isfinite(sigma_value) and sigma_value > 0 else np.nan

    has_clear_main_peak = bool(main is not None and main.get("is_prominent_component", False) and not low_snr_flag)
    has_valid_secondary_peak = bool(secondary is not None)
    has_broad_main_peak = bool(np.isfinite(main_peak_width_bins) and main_peak_width_bins >= broad_main_width_bins)
    has_weak_main_peak = bool(not np.isfinite(main_prominence_sigma) or main_prominence_sigma < weak_main_prominence_sigma)

    if main is None:
        reliability = "no_detected_main_peak"
    elif low_snr_flag and has_weak_main_peak:
        reliability = "low_snr_weak_main_peak"
    elif has_broad_main_peak:
        reliability = "broad_main_peak"
    elif has_valid_secondary_peak:
        reliability = "clear_main_with_secondary_candidate"
    elif has_clear_main_peak:
        reliability = "clear_main_peak"
    elif has_weak_main_peak:
        reliability = "weak_main_peak"
    else:
        reliability = "ambiguous_main_peak"

    return {
        "n_candidate_components": n_candidate,
        "n_prominent_components": int(len(prominent)),
        "n_valid_secondary_components": int(len(secondaries)),
        "main_peak_bin": main_peak_bin,
        "main_peak_prominence": main_peak_prominence,
        "main_peak_width_bins": main_peak_width_bins,
        "main_peak_area": main_peak_area,
        "secondary_peak_bin": secondary_peak_bin,
        "secondary_to_main_amp_ratio": secondary_to_main_amp_ratio,
        "secondary_to_main_prominence_ratio": secondary_to_main_prominence_ratio,
        "raw_argmax_bin": raw_argmax,
        "corrected_argmax_bin": corrected_argmax,
        "abs_main_minus_raw_argmax_bin": abs_main_minus_raw,
        "abs_corrected_minus_raw_argmax_bin": abs_corrected_minus_raw,
        "has_clear_main_peak": has_clear_main_peak,
        "has_valid_secondary_peak": has_valid_secondary_peak,
        "has_broad_main_peak": has_broad_main_peak,
        "has_weak_main_peak": has_weak_main_peak,
        "low_snr_flag": low_snr_flag,
        "waveform_reliability_class": reliability,
        "main_peak_prominence_sigma": main_prominence_sigma,
        "refh_amp": float(refh_amp) if refh_amp is not None and np.isfinite(refh_amp) else np.nan,
        "refh_snr": refh_snr_value,
        "bg_mean": float(bg_mean) if bg_mean is not None and np.isfinite(bg_mean) else np.nan,
        "bg_std": float(bg_std) if bg_std is not None and np.isfinite(bg_std) else np.nan,
    }


def compute_sweep_continuity_features(track: Sequence[float], main_peak_bin: Sequence[float]) -> dict[str, Any]:
    """Compute continuity diagnostics for the main peak trajectory across a sweep."""

    track_arr = np.asarray(track, dtype=np.float64).reshape(-1)
    peak_arr = np.asarray(main_peak_bin, dtype=np.float64).reshape(-1)
    valid = np.isfinite(track_arr) & np.isfinite(peak_arr)
    if not np.any(valid):
        return {
            "continuity_median_abs_residual": np.nan,
            "continuity_p95_abs_residual": np.nan,
            "continuity_max_abs_residual": np.nan,
            "continuity_discontinuity_count": 0,
        }

    ordered_track = track_arr[valid]
    ordered_peak = peak_arr[valid]
    order = np.argsort(ordered_track)
    ordered_peak = ordered_peak[order]

    series = pd.Series(ordered_peak)
    baseline = series.rolling(window=9, center=True, min_periods=1).median().to_numpy(dtype=np.float64)
    residual = ordered_peak - baseline
    abs_residual = np.abs(residual)
    discontinuity_threshold = 6.0
    return {
        "continuity_median_abs_residual": float(np.median(abs_residual)),
        "continuity_p95_abs_residual": float(np.percentile(abs_residual, 95)),
        "continuity_max_abs_residual": float(np.max(abs_residual)),
        "continuity_discontinuity_count": int(np.count_nonzero(abs_residual >= discontinuity_threshold)),
    }


def compute_fixed_bin_stripe_features(
    track: Sequence[float],
    peak_bin: Sequence[float],
    bin_tolerance: float,
    min_fraction: float,
) -> dict[str, Any]:
    """Compute fixed-bin stripe diagnostics from peak-bin concentrations."""

    del track
    bins = np.asarray(peak_bin, dtype=np.float64).reshape(-1)
    valid = bins[np.isfinite(bins)]
    if valid.size == 0:
        return {
            "fixed_bin_mode": np.nan,
            "fixed_bin_fraction": np.nan,
            "fixed_bin_band_min": np.nan,
            "fixed_bin_band_max": np.nan,
            "stripe_like_flag": False,
        }

    rounded = np.rint(valid).astype(np.int64)
    uniq, counts = np.unique(rounded, return_counts=True)
    mode = int(uniq[int(np.argmax(counts))])
    band_min = float(mode - float(bin_tolerance))
    band_max = float(mode + float(bin_tolerance))
    in_band = (valid >= band_min) & (valid <= band_max)
    fixed_fraction = float(np.mean(in_band))
    return {
        "fixed_bin_mode": float(mode),
        "fixed_bin_fraction": fixed_fraction,
        "fixed_bin_band_min": band_min,
        "fixed_bin_band_max": band_max,
        "stripe_like_flag": bool(fixed_fraction >= float(min_fraction)),
    }


def compute_range_window_height_candidate(
    peak_bin: float | int | None,
    n_rx_bins: int,
    rwstart: float | None,
    rwstop: float | None,
    direction: str = "rwstart_to_rwstop",
) -> float:
    """Compute a tentative relative bin-height bookkeeping candidate.

    This is a tentative relative bin-height bookkeeping function and does not
    georeference secondary waveform components in lon/lat.
    """

    if peak_bin is None or rwstart is None or rwstop is None:
        return np.nan
    if not np.isfinite(peak_bin) or not np.isfinite(rwstart) or not np.isfinite(rwstop):
        return np.nan
    if int(n_rx_bins) <= 1:
        return np.nan

    t = float(peak_bin) / float(int(n_rx_bins) - 1)
    direction_l = str(direction).lower()
    if direction_l == "rwstart_to_rwstop":
        return float(rwstart + t * (rwstop - rwstart))
    if direction_l == "rwstop_to_rwstart":
        return float(rwstop + t * (rwstart - rwstop))
    raise ValueError("direction must be 'rwstart_to_rwstop' or 'rwstop_to_rwstart'")


def warn_if_duplicate_cells(index_info: SweepTrackIndex) -> None:
    """Warn when a sweep-track grid contains duplicate cells."""

    if index_info.duplicate_sweep_track_cells > 0:
        warnings.warn(
            f"Detected {index_info.duplicate_sweep_track_cells} duplicate sweep-track cells; "
            "the first flat record index was retained in the grid.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Backward-compatible aliases for earlier notebook/script revisions.
# ---------------------------------------------------------------------------


def prepare_waveform_for_detection(
    y_raw: Sequence[float],
    bg_mean: float | None,
    bg_std: float | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Backward-compatible alias for older notebook/script revisions."""

    return preprocess_waveform(
        y_raw=y_raw,
        bg_mean=bg_mean,
        bg_std=bg_std,
        refh_thres=config.get("refh_thres"),
        config=config,
    )


def detect_components_1d(
    y_raw: Sequence[float],
    bg_mean: float | None,
    bg_std: float | None,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Backward-compatible alias for older notebook/script revisions."""

    return detect_candidate_components_1d(
        y_raw=y_raw,
        bg_mean=bg_mean,
        bg_std=bg_std,
        refh_thres=config.get("refh_thres"),
        config=config,
    )


def summarize_pulse_components(
    components: Sequence[Mapping[str, Any]],
    raw_argmax_bin: float | int | None,
    corrected_argmax_bin: float | int | None,
    refh_amp: float | None,
    refh_snr: float | None,
    good_snr: bool | int | None,
) -> dict[str, Any]:
    """Backward-compatible alias for older notebook/script revisions."""

    return summarize_pulse_waveform_features(
        components=components,
        raw_argmax_bin=raw_argmax_bin,
        corrected_argmax_bin=corrected_argmax_bin,
        refh_amp=refh_amp,
        refh_snr=refh_snr,
        good_snr=good_snr,
        bg_mean=None,
        bg_std=None,
        config={},
    )


def compute_sweep_peak_continuity(track: Sequence[float], peak_bin: Sequence[float]) -> dict[str, Any]:
    """Backward-compatible alias for older notebook/script revisions."""

    return compute_sweep_continuity_features(track=track, main_peak_bin=peak_bin)


def estimate_stripe_like_score(component_bins_across_tracks: Sequence[float]) -> dict[str, Any]:
    """Backward-compatible alias for older notebook/script revisions."""

    track = np.arange(len(component_bins_across_tracks), dtype=np.float64)
    return compute_fixed_bin_stripe_features(
        track=track,
        peak_bin=component_bins_across_tracks,
        bin_tolerance=2.0,
        min_fraction=0.40,
    )
