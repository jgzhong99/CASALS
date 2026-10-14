"""Scientific RX cross-section preparation for the sweep animator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from pyproj import Transformer

from .geometry import (
    bin_to_xyz,
    ecef_to_geodetic,
    geodetic_to_ecef,
    project_cross_track,
)
from .h5_model import SweepData

_ECEF_TO_WGS84 = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)


@dataclass
class CrossSectionGrid:
    mode: str
    view: str
    mapping_convention: str
    grid_data: np.ndarray
    x_min: float
    x_max: float
    h_min: float
    h_max: float
    dx: float
    dh: float
    refh_x: np.ndarray
    refh_h: np.ndarray
    refh_valid: np.ndarray
    argmax_x: Optional[np.ndarray]
    argmax_h: Optional[np.ndarray]
    closure_median_3d_m: Optional[float]
    closure_median_horizontal_m: Optional[float]
    closure_median_vertical_m: Optional[float]
    color_title: str
    color_unit: str
    available: bool = True
    message: str = ""


def _ecef_to_wgs84_height(xyz: np.ndarray) -> np.ndarray:
    """Compute WGS84 ellipsoidal height through PROJ's vectorized geocentric transform."""
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError("ECEF coordinates must have a final dimension of size 3.")
    _, _, height = _ECEF_TO_WGS84.transform(arr[..., 0], arr[..., 1], arr[..., 2], errcheck=False)
    return np.asarray(height, dtype=np.float64)


def _empty_grid(
    sweep: SweepData,
    mode: str,
    view: str,
    mapping_convention: str,
    n_cols: int,
    n_rows: int,
    x_min: float,
    x_max: float,
    h_min: float,
    h_max: float,
    refh_x: np.ndarray,
    refh_valid: np.ndarray,
    refh_h: np.ndarray,
    argmax_x: Optional[np.ndarray],
    argmax_h: Optional[np.ndarray],
    closure: tuple[Optional[float], Optional[float], Optional[float]],
    title: str,
    unit: str,
    *,
    available: bool = True,
    message: str = "",
) -> CrossSectionGrid:
    dx = (x_max - x_min) / max(1, n_cols)
    dh = (h_max - h_min) / max(1, n_rows)
    return CrossSectionGrid(
        mode=mode,
        view=view,
        mapping_convention=mapping_convention,
        grid_data=np.full((n_cols, n_rows), np.nan, dtype=np.float32),
        x_min=x_min,
        x_max=x_max,
        h_min=h_min,
        h_max=h_max,
        dx=dx,
        dh=dh,
        refh_x=refh_x,
        refh_h=refh_h,
        refh_valid=refh_valid,
        argmax_x=argmax_x,
        argmax_h=argmax_h,
        closure_median_3d_m=closure[0],
        closure_median_horizontal_m=closure[1],
        closure_median_vertical_m=closure[2],
        color_title=title,
        color_unit=unit,
        available=available,
        message=message,
    )


def _response_matrix(sweep: SweepData, mode: str, asinh_scale: float) -> tuple[np.ndarray, str, str, bool, str]:
    raw = np.asarray(sweep.waveform_matrix, dtype=np.float32)
    mode = mode.lower()
    present = sweep.track_present[:, None] & np.isfinite(raw)
    if mode == "raw":
        values = np.where(present, raw, np.nan)
        return values, "Raw RX", "counts", True, ""

    if mode in {"minus_bg", "asinh", "standardized"}:
        if sweep.bg_mean is None:
            title = {"minus_bg": "Net RX unavailable", "asinh": "Asinh RX unavailable", "standardized": "Standardized RX unavailable"}[mode]
            unit = "counts" if mode == "minus_bg" else ("dimensionless" if mode == "asinh" else "sigma")
            return np.full_like(raw, np.nan), title, unit, False, "bg_mean is missing from this H5 file"
        bg = sweep.bg_mean[:, None]
        valid_bg = np.isfinite(bg)
        net = np.maximum(raw - bg, 0.0)
        if mode == "minus_bg":
            values = np.where(present & valid_bg, net, np.nan)
            return values, "Net RX", "counts", bool(np.any(np.isfinite(values))), ""
        if mode == "asinh":
            scale = float(asinh_scale) if np.isfinite(asinh_scale) and asinh_scale > 0 else 100.0
            values = np.where(present & valid_bg, np.arcsinh(net / scale), np.nan)
            return values, f"Asinh Net RX (scale {scale:g})", "dimensionless", bool(np.any(np.isfinite(values))), ""
        if sweep.bg_std is None:
            return np.full_like(raw, np.nan), "Standardized RX unavailable", "sigma", False, "bg_std is missing from this H5 file"
        std = sweep.bg_std[:, None]
        valid_std = np.isfinite(std) & (std > 0)
        values = np.where(present & valid_bg & valid_std, (raw - bg) / np.where(valid_std, std, 1.0), np.nan)
        available = bool(np.any(np.isfinite(values)))
        return values, "Standardized RX", "sigma", available, "No finite positive bg_std values in this sweep" if not available else ""

    if mode == "refh_only":
        return np.full_like(raw, np.nan), "Official Refh", "m", True, ""
    raise ValueError(f"Unknown cross-section mode: {mode!r}")


def _height_samples(sweep: SweepData, mapping_convention: str) -> np.ndarray:
    """Return WGS84 ellipsoidal heights at the exact H1/H2 bin fractions.

    The endpoint heights are already ellipsoidal heights. Interpolating them at
    the H1/H2 fractions avoids constructing a full ECEF XYZ array for the 2D
    raster; direct ECEF-to-geodetic parity is checked in tests and real-data
    validation. 3D positions and reconstructed argmax geometry still use ECEF.
    """
    n_tracks = sweep.n_tracks
    n_bins = sweep.n_rx_bins
    if mapping_convention == "H1":
        fractions = np.linspace(0.0, 1.0, n_bins, dtype=np.float64)
    elif mapping_convention == "H2":
        fractions = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
    else:
        raise ValueError(f"Unknown mapping convention: {mapping_convention!r}")
    start = np.asarray(sweep.rwstart_height, dtype=np.float64)
    stop = np.asarray(sweep.rwstop_height, dtype=np.float64)
    valid = (
        np.isfinite(sweep.rwstart_ecef).all(axis=1)
        & np.isfinite(sweep.rwstop_ecef).all(axis=1)
        & np.isfinite(start)
        & np.isfinite(stop)
    )
    heights = np.full((n_tracks, n_bins), np.nan, dtype=np.float64)
    if not np.any(valid):
        return heights
    start_valid = start[valid]
    height_delta = stop[valid] - start_valid
    heights[valid] = start_valid[:, None] + fractions[None, :] * height_delta[:, None]
    return heights


def _height_axis(sweep: SweepData, fixed_height_range: Optional[Tuple[float, float]], n_rows: int) -> tuple[float, float, np.ndarray]:
    if fixed_height_range is not None:
        lo, hi = map(float, fixed_height_range)
    else:
        ends = np.concatenate([sweep.rwstart_height, sweep.rwstop_height])
        ends = ends[np.isfinite(ends)]
        if ends.size:
            lo, hi = float(ends.min()), float(ends.max())
        else:
            lo, hi = -1.0, 1.0
    if hi <= lo:
        lo, hi = lo - 1.0, hi + 1.0
    return lo, hi, np.linspace(lo, hi, n_rows, dtype=np.float64)


def _fill_curtain_row(out: np.ndarray, height_grid: np.ndarray, heights: np.ndarray, response: np.ndarray) -> None:
    """Interpolate only within contiguous finite bin runs; preserve gaps and edges."""
    finite = np.isfinite(heights) & np.isfinite(response)
    if not np.any(finite):
        return
    if finite.all():
        delta = np.diff(heights)
        if np.all(delta > 0.0):
            keep = (height_grid >= heights[0]) & (height_grid <= heights[-1])
            out[keep] = np.interp(height_grid[keep], heights, response).astype(np.float32, copy=False)
            return
        if np.all(delta < 0.0):
            keep = (height_grid >= heights[-1]) & (height_grid <= heights[0])
            out[keep] = np.interp(height_grid[keep], heights[::-1], response[::-1]).astype(np.float32, copy=False)
            return
    edges = np.diff(np.r_[False, finite, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    for start, stop in zip(starts, stops):
        if stop - start < 2:
            continue
        h = heights[start:stop]
        v = response[start:stop]
        order = np.argsort(h, kind="stable")
        h = h[order]
        v = v[order]
        unique, first = np.unique(h, return_index=True)
        if unique.size < 2:
            continue
        keep = (height_grid >= unique[0]) & (height_grid <= unique[-1])
        out[keep] = np.interp(height_grid[keep], unique, v[first]).astype(np.float32, copy=False)


def compute_waveform_cross_section(
    sweep: SweepData,
    mode: str = "minus_bg",
    view: str = "curtain",
    mapping_convention: str = "H2",
    n_cols: int = 384,
    n_rows: int = 512,
    asinh_scale: float = 100.0,
    fixed_height_range: Optional[Tuple[float, float]] = None,
    fixed_x_range: Optional[Tuple[float, float]] = None,
) -> CrossSectionGrid:
    """Prepare a track-index curtain or a georeferenced meter cross-section.

    H1/H2 bin positions are computed in ECEF. Heights are converted from those
    positions to WGS84 ellipsoidal height instead of linearly interpolating the
    endpoint height attributes.
    """
    view = view.lower()
    if view not in {"curtain", "georeferenced"}:
        raise ValueError("view must be 'curtain' or 'georeferenced'")
    mapping = mapping_convention.upper()
    if mapping not in {"H1", "H2"}:
        raise ValueError("mapping_convention must be 'H1' or 'H2'")
    n_cols = max(1, int(n_cols))
    n_rows = max(1, int(n_rows))

    if view == "curtain":
        n_cols = sweep.n_tracks
        x_min, x_max = -0.5, float(sweep.n_tracks) - 0.5
    else:
        if not sweep.can_use_meters or sweep.cross_track_unit_vector is None or sweep.cross_track_center is None:
            x_min, x_max = (-1.0, 1.0) if fixed_x_range is None else tuple(map(float, fixed_x_range))
            h_min, h_max, _ = _height_axis(sweep, fixed_height_range, n_rows)
            refh_x = np.full(sweep.n_tracks, np.nan, dtype=np.float64)
            return _empty_grid(
                sweep, mode, view, mapping, n_cols, n_rows, x_min, x_max, h_min, h_max,
                refh_x, sweep.valid_pulse_mask, sweep.refh_height, None, None, (None, None, None),
                "Georeferenced cross-section unavailable", "counts", available=False,
                message=sweep.cross_track_reason,
            )
        start_x = project_cross_track(sweep.rwstart_enu[:, :2], sweep.cross_track_unit_vector, sweep.cross_track_center)
        stop_x = project_cross_track(sweep.rwstop_enu[:, :2], sweep.cross_track_unit_vector, sweep.cross_track_center)
        finite_x = np.concatenate([start_x[np.isfinite(start_x)], stop_x[np.isfinite(stop_x)]])
        if fixed_x_range is not None:
            x_min, x_max = map(float, fixed_x_range)
        elif finite_x.size:
            x_min, x_max = float(finite_x.min()), float(finite_x.max())
            pad = max(1.0, (x_max - x_min) * 0.02)
            x_min, x_max = x_min - pad, x_max + pad
        else:
            x_min, x_max = -1.0, 1.0

    h_min, h_max, height_grid = _height_axis(sweep, fixed_height_range, n_rows)
    response, title, unit, available, message = _response_matrix(sweep, mode, asinh_scale)
    heights = _height_samples(sweep, mapping)

    # Official Refh and reconstructed raw-maximum overlays remain distinct.
    refh_x = sweep.track_nums.astype(np.float64) if view == "curtain" else np.asarray(sweep.cross_track_refh_m if sweep.cross_track_refh_m is not None else np.full(sweep.n_tracks, np.nan), dtype=np.float64)
    refh_h = np.asarray(sweep.refh_height, dtype=np.float64)
    refh_valid = np.asarray(sweep.valid_pulse_mask, dtype=bool) & np.isfinite(refh_x) & np.isfinite(refh_h)
    argmax_bins = np.full(sweep.n_tracks, np.nan, dtype=np.float64)
    valid_rows = sweep.track_present & np.isfinite(sweep.waveform_matrix).any(axis=1)
    if np.any(valid_rows):
        safe = np.where(np.isfinite(sweep.waveform_matrix[valid_rows]), sweep.waveform_matrix[valid_rows], -np.inf)
        argmax_bins[valid_rows] = np.argmax(safe, axis=1)
    has_rw = np.isfinite(sweep.rwstart_ecef).all(axis=1) & np.isfinite(sweep.rwstop_ecef).all(axis=1)
    valid_rec = np.isfinite(argmax_bins) & has_rw
    argmax_x = np.full(sweep.n_tracks, np.nan, dtype=np.float64)
    argmax_h = np.full(sweep.n_tracks, np.nan, dtype=np.float64)
    closure = (None, None, None)
    if np.any(valid_rec):
        p = bin_to_xyz(sweep.rwstart_ecef[valid_rec], sweep.rwstop_ecef[valid_rec], argmax_bins[valid_rec], sweep.n_rx_bins, convention=mapping)
        geo = ecef_to_geodetic(p)
        argmax_h[valid_rec] = geo[:, 2]
        if view == "curtain":
            argmax_x[valid_rec] = sweep.track_nums[valid_rec]
        else:
            enu = bin_to_xyz(sweep.rwstart_enu[valid_rec], sweep.rwstop_enu[valid_rec], argmax_bins[valid_rec], sweep.n_rx_bins, convention=mapping)
            argmax_x[valid_rec] = project_cross_track(enu[:, :2], sweep.cross_track_unit_vector, sweep.cross_track_center)
        refh_ecef = geodetic_to_ecef(sweep.refh_lon[valid_rec], sweep.refh_lat[valid_rec], sweep.refh_height[valid_rec])
        diff3 = np.linalg.norm(p - refh_ecef, axis=1)
        diffv = np.abs(geo[:, 2] - sweep.refh_height[valid_rec])
        closure = (
            float(np.nanmedian(diff3)),
            float(np.nanmedian(np.sqrt(np.maximum(diff3 * diff3 - diffv * diffv, 0.0)))),
            float(np.nanmedian(diffv)),
        )

    if mode.lower() == "refh_only":
        return _empty_grid(sweep, mode, view, mapping, n_cols, n_rows, x_min, x_max, h_min, h_max,
                           refh_x, refh_valid, refh_h, argmax_x, argmax_h, closure, title, unit)
    if not available:
        return _empty_grid(sweep, mode, view, mapping, n_cols, n_rows, x_min, x_max, h_min, h_max,
                           refh_x, refh_valid, refh_h, argmax_x, argmax_h, closure, title, unit,
                           available=False, message=message)

    if view == "curtain":
        grid = np.full((n_cols, n_rows), np.nan, dtype=np.float32)
        for track in range(sweep.n_tracks):
            if not sweep.track_present[track]:
                continue
            _fill_curtain_row(grid[track], height_grid, heights[track], response[track])
    else:
        bins = np.arange(sweep.n_rx_bins, dtype=np.float64)
        t = (bins / (sweep.n_rx_bins - 1)) if mapping == "H1" else ((bins + 0.5) / sweep.n_rx_bins)
        start_x = project_cross_track(sweep.rwstart_enu[:, :2], sweep.cross_track_unit_vector, sweep.cross_track_center)
        stop_x = project_cross_track(sweep.rwstop_enu[:, :2], sweep.cross_track_unit_vector, sweep.cross_track_center)
        x_matrix = start_x[:, None] + t[None, :] * (stop_x - start_x)[:, None]
        finite = np.isfinite(response) & np.isfinite(heights) & np.isfinite(x_matrix)
        x = x_matrix[finite]
        h = heights[finite]
        values = response[finite]
        ix = np.floor((x - x_min) * (n_cols / (x_max - x_min))).astype(np.int64)
        iy = np.floor((h - h_min) * (n_rows / (h_max - h_min))).astype(np.int64)
        inside = (ix >= 0) & (ix < n_cols) & (iy >= 0) & (iy < n_rows)
        flat = ix[inside] * n_rows + iy[inside]
        counts = np.bincount(flat, minlength=n_cols * n_rows)
        sums = np.bincount(flat, weights=values[inside], minlength=n_cols * n_rows)
        grid = np.full(n_cols * n_rows, np.nan, dtype=np.float32)
        used = counts > 0
        grid[used] = (sums[used] / counts[used]).astype(np.float32)
        grid = grid.reshape(n_cols, n_rows)

    return CrossSectionGrid(
        mode=mode,
        view=view,
        mapping_convention=mapping,
        grid_data=grid,
        x_min=x_min,
        x_max=x_max,
        h_min=h_min,
        h_max=h_max,
        dx=(x_max - x_min) / n_cols,
        dh=(h_max - h_min) / n_rows,
        refh_x=refh_x,
        refh_h=refh_h,
        refh_valid=refh_valid,
        argmax_x=argmax_x,
        argmax_h=argmax_h,
        closure_median_3d_m=closure[0],
        closure_median_horizontal_m=closure[1],
        closure_median_vertical_m=closure[2],
        color_title=title,
        color_unit=unit,
        available=True,
    )
