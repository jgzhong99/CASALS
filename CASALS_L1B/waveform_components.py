"""Reusable helpers for CASALS L1B waveform component diagnostics.

Scientific boundary
-------------------
CASALS L1B provides one official geolocated ``refh`` point per pulse. The
secondary components detected here are waveform-derived diagnostics only. They
are not georeferenced returns and must not be interpreted as an official
multi-return point cloud product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence
import warnings

import h5py
import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised indirectly by runtime checks
    from scipy.signal import find_peaks, medfilt, peak_widths, savgol_filter
except Exception as exc:  # pragma: no cover
    find_peaks = None
    medfilt = None
    peak_widths = None
    savgol_filter = None
    _SCIPY_IMPORT_ERROR = exc
else:  # pragma: no cover
    _SCIPY_IMPORT_ERROR = None


DEFAULT_MULTI_PEAK_PROMINENCE_RATIO = 0.35
DEFAULT_BROAD_MAIN_WIDTH_BINS = 12.0
DEFAULT_CONTINUITY_WINDOW = 9
DEFAULT_DISCONTINUITY_THRESHOLD_BINS = 6.0
DEFAULT_STRIPE_BAND_HALF_WIDTH = 2.0
DEFAULT_STRIPE_FRACTION_THRESHOLD = 0.40


@dataclass(frozen=True)
class RecordIndexGridInfo:
    """Sweep/track indexing summary for waveform records.

    The returned grid maps ``record_index_grid[sweep, track]`` to the flat pulse
    record index in the H5 file. Missing cells are ``-1``. The mapping is a
    bookkeeping structure for waveform diagnostics only; it does not geolocate
    secondary waveform components.
    """

    record_index_grid: np.ndarray
    sweep_num: np.ndarray
    track_num: np.ndarray
    n_sweeps: int
    n_tracks: int
    n_records: int
    duplicate_sweep_track_cells: int
    missing_sweep_track_cells: int
    complete_rectangular_grid: bool


def require_scipy() -> None:
    """Raise a clear error if SciPy is unavailable.

    Waveform component detection relies on ``scipy.signal`` peak utilities. This
    check only guards the local analysis workflow; it does not affect official
    CASALS geolocation products.
    """

    if find_peaks is None or peak_widths is None or savgol_filter is None or medfilt is None:
        raise RuntimeError(
            "SciPy is required for CASALS waveform component detection. "
            "Please install scipy."
        ) from _SCIPY_IMPORT_ERROR


def find_dataset(h5: h5py.File, name: str) -> Optional[h5py.Dataset]:
    """Return an H5 dataset by exact root name or recursive basename match.

    Matching is limited to dataset lookup convenience for diagnostic workflows.
    Secondary waveform components derived later remain waveform-only diagnostics,
    not georeferenced points.
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
    """Return a required dataset or raise a clear error.

    The lookup is used for waveform diagnostics and metadata inspection. It does
    not change the single official CASALS ``refh`` geolocation per pulse.
    """

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

    Missing optional fields return ``None``. Length mismatches raise so that the
    diagnostic workflow cannot silently mix waveform records with wrong metadata.
    Any data returned here are pulse-level fields only; they do not georeference
    secondary waveform-derived components.
    """

    ds = find_dataset(h5, name)
    if ds is None:
        return None
    arr = np.asarray(ds[...], dtype=dtype).reshape(-1)
    if arr.size != int(n_expected):
        raise ValueError(
            f"Optional dataset {name!r} has size {arr.size}, expected {int(n_expected)}."
        )
    return arr


def build_record_index_grid(h5: h5py.File) -> RecordIndexGridInfo:
    """Build ``record_index_grid[sweep, track] = flat pulse index``.

    The returned grid summarizes the waveform storage layout. Duplicate
    ``(sweep_num, track_num)`` cells are retained only through their first flat
    record index and reported explicitly so callers can decide whether to stop.
    Missing cells are filled with ``-1``. This indexing utility supports
    waveform-derived diagnostics only.
    """

    sweep = np.asarray(require_dataset(h5, "sweep_num")[...], dtype=np.int64).reshape(-1)
    track = np.asarray(require_dataset(h5, "track_num")[...], dtype=np.int64).reshape(-1)
    if sweep.size != track.size:
        raise ValueError(f"sweep_num and track_num sizes differ: {sweep.size} vs {track.size}")
    if sweep.size == 0:
        raise ValueError("No pulse records were found in sweep_num/track_num.")
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
    return RecordIndexGridInfo(
        record_index_grid=grid,
        sweep_num=sweep,
        track_num=track,
        n_sweeps=n_sweeps,
        n_tracks=n_tracks,
        n_records=n_records,
        duplicate_sweep_track_cells=duplicate_cells,
        missing_sweep_track_cells=missing_cells,
        complete_rectangular_grid=bool(duplicate_cells == 0 and missing_cells == 0 and n_records == n_expected),
    )


def infer_waveform_record_axis(ds: h5py.Dataset, n_records: int) -> int:
    """Infer which waveform axis stores pulse records.

    CASALS waveform datasets are expected to be 2D and stored either as
    ``[n_records, n_bins]`` or ``[n_bins, n_records]``. This identifies the
    record axis for diagnostic reads only; it does not georeference any
    waveform-derived secondary components.
    """

    if ds.ndim != 2:
        raise ValueError(f"{ds.name} must be 2D, got shape={ds.shape}")
    if ds.shape[0] == int(n_records):
        return 0
    if ds.shape[1] == int(n_records):
        return 1
    raise ValueError(f"{ds.name} shape {ds.shape} does not match n_records={n_records} on either axis.")


def read_waveform_records(
    ds: h5py.Dataset,
    record_axis: int,
    indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Read waveform rows/columns by record index while respecting h5py sorting.

    ``h5py`` fancy indexing requires monotonically increasing integer indices.
    This helper sorts indices for the read and then restores the caller's
    original order. Returned arrays are waveform samples only; they do not imply
    any new geolocation for secondary components.
    """

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


def smooth_waveform(y: Sequence[float], method: str = "none", window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Smooth a 1D waveform using ``none``, ``median``, or ``savgol``.

    Smoothing is a diagnostic aid for waveform component detection. It does not
    create new georeferenced returns or alter the official single-pulse ``refh``
    geolocation.
    """

    arr = np.asarray(y, dtype=np.float64)
    method_l = str(method).lower()
    if method_l == "none":
        return arr.copy()

    require_scipy()
    if window is None:
        return arr.copy()
    window = int(window)
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    if window > arr.size:
        window = arr.size if arr.size % 2 == 1 else max(1, arr.size - 1)
    if window <= 1:
        return arr.copy()

    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr.copy()

    work = arr.copy()
    if not np.all(finite):
        valid_x = np.flatnonzero(finite)
        work[~finite] = np.interp(np.flatnonzero(~finite), valid_x, work[finite])

    if method_l == "median":
        return np.asarray(medfilt(work, kernel_size=window), dtype=np.float64)
    if method_l == "savgol":
        polyorder = int(polyorder)
        if polyorder >= window:
            polyorder = max(0, window - 1)
        return np.asarray(savgol_filter(work, window_length=window, polyorder=polyorder, mode="interp"), dtype=np.float64)
    raise ValueError("method must be one of: none, median, savgol")


def prepare_waveform_for_detection(
    y_raw: Sequence[float],
    bg_mean: float | None,
    bg_std: float | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare a waveform for component detection.

    The output includes raw, background-adjusted, and smoothed arrays plus the
    sigma scale used for peak thresholds. These arrays are intended for
    diagnostics only and do not georeference secondary waveform components.
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
        refh_thres = config.get("refh_thres")
        background_offset = float(refh_thres) if refh_thres is not None and np.isfinite(refh_thres) else 0.0
    else:
        raise ValueError("background_mode must be one of: raw, minus_bg_mean, minus_refh_thres")

    processed = y_raw_arr - background_offset
    if bool(config.get("clip_negative_after_background", False)):
        processed = np.where(np.isfinite(processed), np.maximum(processed, 0.0), np.nan)

    smoothing_method = str(config.get("smoothing_method", config.get("smoothing", "none"))).lower()
    smoothing_window = int(config.get("smoothing_window", 7) or 7)
    smoothing_polyorder = int(config.get("smoothing_polyorder", 2) or 2)
    smoothed = smooth_waveform(processed, method=smoothing_method, window=smoothing_window, polyorder=smoothing_polyorder)

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


def detect_components_1d(
    y_raw: Sequence[float],
    bg_mean: float | None,
    bg_std: float | None,
    config: Mapping[str, Any],
) -> list[dict[str, float | int]]:
    """Detect waveform components in one CASALS L1B receive waveform.

    Peaks are found from a background-adjusted and optionally smoothed waveform
    using ``scipy.signal.find_peaks``. Returned components are waveform-derived
    diagnostics only. They are not georeferenced returns and must not be used to
    construct an official multi-return point cloud.
    """

    prepared = prepare_waveform_for_detection(y_raw=y_raw, bg_mean=bg_mean, bg_std=bg_std, config=config)
    if not prepared["valid"]:
        return []

    smoothed = np.asarray(prepared["smoothed"], dtype=np.float64)
    processed = np.asarray(prepared["processed"], dtype=np.float64)
    raw = np.asarray(prepared["raw"], dtype=np.float64)
    sigma = float(prepared["sigma"])

    finite = np.isfinite(smoothed)
    if not np.any(finite):
        return []

    require_scipy()
    work = np.where(finite, smoothed, -np.inf)
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

    sort_key = np.lexsort((
        -np.asarray(peaks, dtype=np.float64),
        -np.nan_to_num(np.where(np.isfinite(processed[peaks]), processed[peaks], np.nan), nan=-np.inf),
        -np.nan_to_num(prominences, nan=-np.inf),
    ))
    max_components = int(config.get("max_components_per_pulse", peaks.size))
    keep = sort_key[:max_components]

    components: list[dict[str, float | int]] = []
    for rank, idx in enumerate(keep, start=1):
        peak_bin = int(peaks[idx])
        left = float(left_ips[idx])
        right = float(right_ips[idx])
        lo = max(0, int(np.floor(left)))
        hi = min(processed.size - 1, int(np.ceil(right)))
        segment = processed[lo : hi + 1]
        area_processed = float(np.nansum(np.where(np.isfinite(segment), np.maximum(segment, 0.0), 0.0)))
        components.append(
            {
                "peak_bin": peak_bin,
                "amplitude_raw": float(raw[peak_bin]) if np.isfinite(raw[peak_bin]) else np.nan,
                "amplitude_processed": float(processed[peak_bin]) if np.isfinite(processed[peak_bin]) else np.nan,
                "prominence": float(prominences[idx]),
                "width_bins": float(widths[idx]),
                "left_ips": left,
                "right_ips": right,
                "width_height": float(width_heights[idx]),
                "area_processed": area_processed,
                "rank_by_prominence": int(rank),
            }
        )
    return components


def summarize_pulse_components(
    components: Sequence[Mapping[str, Any]],
    raw_argmax_bin: float | int | None,
    corrected_argmax_bin: float | int | None,
    refh_amp: float | None,
    refh_snr: float | None,
    good_snr: bool | int | None,
) -> dict[str, Any]:
    """Summarize component diagnostics for a single pulse.

    The returned features are intended for refh-quality analysis and artifact
    screening. They do not define new georeferenced returns or replace the
    official single-pulse CASALS ``refh`` geolocation.
    """

    n_components = int(len(components))
    refh_snr_value = float(refh_snr) if refh_snr is not None and np.isfinite(refh_snr) else np.nan
    low_snr_flag = bool((good_snr is not None and not bool(good_snr)) or (np.isfinite(refh_snr_value) and refh_snr_value < 3.0))

    def finite_value(name: str, idx: int = 0) -> float:
        if idx >= n_components:
            return np.nan
        value = components[idx].get(name)
        return float(value) if value is not None and np.isfinite(value) else np.nan

    main_peak_bin = finite_value("peak_bin", 0)
    main_peak_amp = finite_value("amplitude_raw", 0)
    main_peak_prominence = finite_value("prominence", 0)
    main_peak_width = finite_value("width_bins", 0)
    secondary_peak_bin = finite_value("peak_bin", 1)
    secondary_amp = finite_value("amplitude_raw", 1)
    secondary_prominence = finite_value("prominence", 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        secondary_to_main_amp_ratio = float(secondary_amp / main_peak_amp) if np.isfinite(secondary_amp) and np.isfinite(main_peak_amp) and main_peak_amp != 0 else np.nan
        secondary_to_main_prominence_ratio = (
            float(secondary_prominence / main_peak_prominence)
            if np.isfinite(secondary_prominence) and np.isfinite(main_peak_prominence) and main_peak_prominence != 0
            else np.nan
        )

    raw_argmax = float(raw_argmax_bin) if raw_argmax_bin is not None and np.isfinite(raw_argmax_bin) else np.nan
    corrected_argmax = float(corrected_argmax_bin) if corrected_argmax_bin is not None and np.isfinite(corrected_argmax_bin) else np.nan
    abs_main_minus_raw = abs(main_peak_bin - raw_argmax) if np.isfinite(main_peak_bin) and np.isfinite(raw_argmax) else np.nan
    abs_corrected_minus_raw = abs(corrected_argmax - raw_argmax) if np.isfinite(corrected_argmax) and np.isfinite(raw_argmax) else np.nan

    has_broad_main_peak = bool(np.isfinite(main_peak_width) and main_peak_width >= DEFAULT_BROAD_MAIN_WIDTH_BINS)
    has_multiple_prominent_peaks = bool(
        n_components >= 2 and np.isfinite(secondary_to_main_prominence_ratio)
        and secondary_to_main_prominence_ratio >= DEFAULT_MULTI_PEAK_PROMINENCE_RATIO
    )
    has_clean_single_peak = bool(
        n_components == 1 and np.isfinite(main_peak_bin) and not has_broad_main_peak and not low_snr_flag
    )

    return {
        "n_components": n_components,
        "main_peak_bin": main_peak_bin,
        "main_peak_amp": main_peak_amp,
        "main_peak_prominence": main_peak_prominence,
        "main_peak_width": main_peak_width,
        "secondary_peak_bin": secondary_peak_bin,
        "secondary_to_main_amp_ratio": secondary_to_main_amp_ratio,
        "secondary_to_main_prominence_ratio": secondary_to_main_prominence_ratio,
        "raw_argmax_bin": raw_argmax,
        "corrected_argmax_bin": corrected_argmax,
        "abs_main_minus_raw_argmax_bin": abs_main_minus_raw,
        "abs_corrected_minus_raw_argmax_bin": abs_corrected_minus_raw,
        "has_clean_single_peak": has_clean_single_peak,
        "has_multiple_prominent_peaks": has_multiple_prominent_peaks,
        "has_broad_main_peak": has_broad_main_peak,
        "low_snr_flag": low_snr_flag,
        "refh_amp": float(refh_amp) if refh_amp is not None and np.isfinite(refh_amp) else np.nan,
        "refh_snr": refh_snr_value,
    }


def compute_sweep_peak_continuity(track: Sequence[float], peak_bin: Sequence[float]) -> dict[str, Any]:
    """Measure how smoothly a peak-bin trajectory evolves across a sweep.

    A rolling median baseline is used to quantify residual jumps. The result is
    a diagnostic continuity score for waveform behavior only and does not imply
    any new geolocation for waveform-derived components.
    """

    track_arr = np.asarray(track, dtype=np.float64).reshape(-1)
    peak_arr = np.asarray(peak_bin, dtype=np.float64).reshape(-1)
    valid = np.isfinite(track_arr) & np.isfinite(peak_arr)
    if np.count_nonzero(valid) == 0:
        return {
            "median_abs_residual": np.nan,
            "p95_abs_residual": np.nan,
            "max_abs_residual": np.nan,
            "discontinuity_count": 0,
        }

    order = np.argsort(track_arr[valid])
    ordered_peak = peak_arr[valid][order]
    series = pd.Series(ordered_peak)
    baseline = series.rolling(window=DEFAULT_CONTINUITY_WINDOW, center=True, min_periods=1).median().to_numpy(dtype=np.float64)
    residual = ordered_peak - baseline
    abs_residual = np.abs(residual)
    return {
        "median_abs_residual": float(np.median(abs_residual)),
        "p95_abs_residual": float(np.percentile(abs_residual, 95)),
        "max_abs_residual": float(np.max(abs_residual)),
        "discontinuity_count": int(np.count_nonzero(abs_residual >= DEFAULT_DISCONTINUITY_THRESHOLD_BINS)),
    }


def estimate_stripe_like_score(component_bins_across_tracks: Sequence[float]) -> dict[str, Any]:
    """Estimate whether many tracks collapse into a narrow fixed-bin band.

    This is intended to flag stripe-like waveform artifacts across a sweep.
    Even if a narrow band is detected, the result remains a waveform diagnostic
    and does not georeference any secondary component.
    """

    bins = np.asarray(component_bins_across_tracks, dtype=np.float64).reshape(-1)
    valid = bins[np.isfinite(bins)]
    if valid.size == 0:
        return {
            "fixed_bin_mode": np.nan,
            "fixed_bin_fraction": np.nan,
            "stripe_like_flag": False,
        }

    rounded = np.rint(valid).astype(np.int64)
    unique_vals, counts = np.unique(rounded, return_counts=True)
    mode_idx = int(np.argmax(counts))
    fixed_bin_mode = int(unique_vals[mode_idx])
    band_mask = np.abs(valid - fixed_bin_mode) <= DEFAULT_STRIPE_BAND_HALF_WIDTH
    fixed_bin_fraction = float(np.mean(band_mask))
    stripe_like_flag = bool(fixed_bin_fraction >= DEFAULT_STRIPE_FRACTION_THRESHOLD)
    return {
        "fixed_bin_mode": fixed_bin_mode,
        "fixed_bin_fraction": fixed_bin_fraction,
        "stripe_like_flag": stripe_like_flag,
    }


def safe_argmax_bin(values: Sequence[float]) -> tuple[float, float]:
    """Return ``(argmax_bin, argmax_value)`` for a 1D waveform.

    All-NaN inputs return ``(nan, nan)``. This helper is purely diagnostic and
    does not redefine the official ``refh`` geolocation semantics.
    """

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return np.nan, np.nan
    safe = np.where(np.isfinite(arr), arr, -np.inf)
    idx = int(np.argmax(safe))
    return float(idx), float(arr[idx]) if np.isfinite(arr[idx]) else np.nan


def warn_if_duplicate_cells(info: RecordIndexGridInfo) -> None:
    """Emit a warning when sweep-track indexing is not one-to-one.

    Duplicate sweep-track cells make waveform diagnostics ambiguous because one
    pulse index cannot represent multiple stored records. This warning is about
    data integrity only and does not concern official CASALS geolocation.
    """

    if info.duplicate_sweep_track_cells > 0:
        warnings.warn(
            f"Detected {info.duplicate_sweep_track_cells} duplicate sweep-track cells; "
            "only the first flat record index was retained in the grid.",
            stacklevel=2,
        )
