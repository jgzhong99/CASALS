"""Cross-section processing and 2D height mapping for CASALS L1B sweeps.

Scientific meaning:
- Maps RX waveform bins to WGS84 ellipsoidal height and horizontal cross-track position.
- Supports audited H2 mapping (window edges with centered bins) and production H1 mapping.
- Grids discrete waveform bins into a regular (cross-track, height) physical coordinate grid.
- Produces Raw, Minus-BG, Asinh, Standardized response (Z-score), and Refh spatial sections.
- Computes Refh and reconstructed argmax overlay coordinates and closure statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .geometry import bin_to_xyz, ecef_to_geodetic, geodetic_to_ecef, project_cross_track
from .h5_model import SweepData


@dataclass
class CrossSectionGrid:
    """A regular 2D grid containing the waveform cross-section."""

    mode: str
    x_axis_mode: str  # 'meters' or 'track'
    mapping_convention: str  # 'H2' or 'H1'

    # 2D values of shape [n_x, n_h] for pyqtgraph col-major axis order
    # (or [n_h, n_x] where specified)
    grid_data: np.ndarray  # Shape (n_x, n_h)

    # Coordinate extents
    x_min: float
    x_max: float
    h_min: float
    h_max: float
    dx: float
    dh: float

    # Refh overlay
    refh_x: np.ndarray
    refh_h: np.ndarray
    refh_valid: np.ndarray

    # Argmax overlay (if computed)
    argmax_x: Optional[np.ndarray]
    argmax_h: Optional[np.ndarray]

    # Closure statistics (between Refh and reconstructed argmax)
    closure_median_3d_m: Optional[float]
    closure_median_horizontal_m: Optional[float]
    closure_median_vertical_m: Optional[float]

    # Data units and color title
    color_title: str
    color_unit: str


def compute_waveform_cross_section(
    sweep: SweepData,
    mode: str = "minus_bg",  # 'raw', 'minus_bg', 'asinh', 'standardized', 'refh_only'
    x_axis_mode: str = "meters",  # 'meters' or 'track'
    mapping_convention: str = "H2",  # 'H2' or 'H1'
    n_cols: int = 384,
    n_rows: int = 512,
    asinh_scale: float = 100.0,
    fixed_height_range: Optional[Tuple[float, float]] = None,
) -> CrossSectionGrid:
    """Transform sweep waveforms into a regular 2D physical or track cross-section grid.

    Parameters
    ----------
    sweep : SweepData
        The sweep data object.
    mode : str
        'raw', 'minus_bg', 'asinh', 'standardized', 'refh_only'
    x_axis_mode : str
        'meters' or 'track'. If meters cannot be determined, falls back to track.
    mapping_convention : str
        'H2' or 'H1'.
    n_cols : int
        Number of horizontal grid bins.
    n_rows : int
        Number of vertical height bins.
    asinh_scale : float
        Scale parameter for asinh display transformation.
    fixed_height_range : tuple of (h_min, h_max), optional
        If provided, fixes the vertical grid extent instead of auto-fitting sweep extent.
    """
    n_tracks = sweep.n_tracks
    n_bins = sweep.n_rx_bins
    use_meters = x_axis_mode.lower() == "meters" and sweep.can_use_meters

    # Prepare values per (track, bin) based on display mode
    raw_matrix = sweep.waveform_matrix  # Shape (n_tracks, n_bins)
    mode_l = mode.lower()

    if mode_l == "raw":
        response = raw_matrix.copy()
        color_title = "Raw RX Amplitude"
        color_unit = "counts (ADU)"
    elif mode_l == "minus_bg":
        if sweep.bg_mean is not None and len(sweep.bg_mean) == n_tracks:
            bg = sweep.bg_mean[:, None]
            response = np.maximum(raw_matrix - bg, 0.0)
        else:
            response = raw_matrix.copy()
        color_title = "Background-subtracted RX Response"
        color_unit = "counts"
    elif mode_l == "asinh":
        if sweep.bg_mean is not None and len(sweep.bg_mean) == n_tracks:
            bg = sweep.bg_mean[:, None]
            net = np.maximum(raw_matrix - bg, 0.0)
        else:
            net = raw_matrix.copy()
        scale = float(asinh_scale) if asinh_scale > 0 else 100.0
        response = np.arcsinh(net / scale)
        color_title = f"Asinh-scaled Net RX (scale={scale:g})"
        color_unit = "asinh(counts)"
    elif mode_l == "standardized":
        if (
            sweep.bg_mean is not None
            and sweep.bg_std is not None
            and len(sweep.bg_mean) == n_tracks
            and len(sweep.bg_std) == n_tracks
        ):
            bg = sweep.bg_mean[:, None]
            std = sweep.bg_std[:, None]
            valid_std = np.isfinite(std) & (std > 0)
            safe_std = np.where(valid_std, std, 1.0)
            z_score = (raw_matrix - bg) / safe_std
            z_score[~np.broadcast_to(valid_std, raw_matrix.shape)] = np.nan
            response = z_score
            color_title = "Standardized RX excess (σ)"
            color_unit = "standard deviations (σ)"
        else:
            # Fall back to raw if bg_std missing
            response = raw_matrix.copy()
            color_title = "Raw RX (bg_std unavailable)"
            color_unit = "counts"
    elif mode_l == "refh_only":
        response = np.full_like(raw_matrix, np.nan)
        color_title = "Refh Spatial Section"
        color_unit = "N/A"
    else:
        raise ValueError(f"Unknown cross-section mode: {mode!r}")

    # Compute official Refh coordinates for overlay
    refh_valid = sweep.valid_pulse_mask
    if use_meters and sweep.cross_track_refh_m is not None:
        refh_x = sweep.cross_track_refh_m
    else:
        refh_x = sweep.track_nums.astype(np.float64)
    refh_h = sweep.refh_height

    # Compute argmax bin for each track and reconstruct position via H1/H2
    valid_row = np.isfinite(raw_matrix).any(axis=1)
    safe_wave = np.where(np.isfinite(raw_matrix), raw_matrix, -np.inf)
    argmax_bins = np.full(n_tracks, np.nan, dtype=np.float64)
    argmax_bins[valid_row] = np.argmax(safe_wave[valid_row], axis=1).astype(np.float64)

    # Argmax 3D ECEF positions
    valid_argmax = np.isfinite(argmax_bins) & (sweep.track_nums >= 0)
    has_rw = np.isfinite(sweep.rwstart_ecef[:, 0]) & np.isfinite(sweep.rwstop_ecef[:, 0])
    valid_rec = valid_argmax & has_rw

    argmax_x = np.full(n_tracks, np.nan, dtype=np.float64)
    argmax_h = np.full(n_tracks, np.nan, dtype=np.float64)
    closure_3d_m = None
    closure_horiz_m = None
    closure_vert_m = None

    if np.any(valid_rec):
        bins_rec = argmax_bins[valid_rec]
        p_ecef = bin_to_xyz(
            sweep.rwstart_ecef[valid_rec],
            sweep.rwstop_ecef[valid_rec],
            bins_rec,
            n_bins,
            convention=mapping_convention,
        )
        geo_rec = ecef_to_geodetic(p_ecef)
        argmax_h[valid_rec] = geo_rec[:, 2]

        if use_meters and sweep.cross_track_unit_vector is not None and sweep.cross_track_center is not None:
            enu_rec = sweep.rwstart_enu[valid_rec] + np.expand_dims(
                (bins_rec + 0.5) / float(n_bins)
                if mapping_convention == "H2"
                else bins_rec / float(n_bins - 1),
                axis=-1,
            ) * (sweep.rwstop_enu[valid_rec] - sweep.rwstart_enu[valid_rec])
            argmax_x[valid_rec] = project_cross_track(
                enu_rec[:, :2], sweep.cross_track_unit_vector, sweep.cross_track_center
            )
        else:
            argmax_x[valid_rec] = sweep.track_nums[valid_rec].astype(np.float64)

        # Compute closure residuals against official Refh
        refh_ecef = geodetic_to_ecef(
            sweep.refh_lon[valid_rec],
            sweep.refh_lat[valid_rec],
            sweep.refh_height[valid_rec],
        )
        diff_3d = np.linalg.norm(p_ecef - refh_ecef, axis=1)
        closure_3d_m = float(np.nanmedian(diff_3d))
        diff_h = np.abs(geo_rec[:, 2] - sweep.refh_height[valid_rec])
        closure_vert_m = float(np.nanmedian(diff_h))

        # Horizontal error approximation
        diff_horiz = np.sqrt(np.maximum(diff_3d**2 - diff_h**2, 0.0))
        closure_horiz_m = float(np.nanmedian(diff_horiz))

    # Determine X extents
    if use_meters:
        # Cross-track positions for all bins across range window
        # P(b) = P0 + t*(P1 - P0). Cross-track distance is linear in t.
        v = sweep.cross_track_unit_vector
        c = sweep.cross_track_center
        assert v is not None and c is not None

        x_start = project_cross_track(sweep.rwstart_enu[:, :2], v, c)
        x_stop = project_cross_track(sweep.rwstop_enu[:, :2], v, c)

        x_all = np.concatenate([x_start[np.isfinite(x_start)], x_stop[np.isfinite(x_stop)]])
        if len(x_all) >= 2:
            x_min = float(np.nanmin(x_all)) - 5.0
            x_max = float(np.nanmax(x_all)) + 5.0
        else:
            x_min, x_max = -150.0, 150.0
    else:
        x_min = -0.5
        x_max = float(n_tracks) - 0.5

    # Determine Height extents
    if fixed_height_range is not None:
        h_min, h_max = float(fixed_height_range[0]), float(fixed_height_range[1])
    else:
        h_all = np.concatenate(
            [
                sweep.rwstart_height[np.isfinite(sweep.rwstart_height)],
                sweep.rwstop_height[np.isfinite(sweep.rwstop_height)],
            ]
        )
        if len(h_all) >= 2:
            h_min = float(np.nanmin(h_all)) - 5.0
            h_max = float(np.nanmax(h_all)) + 5.0
        else:
            h_min, h_max = 2000.0, 2400.0

    if h_max <= h_min:
        h_max = h_min + 50.0
    if x_max <= x_min:
        x_max = x_min + 50.0

    dx = (x_max - x_min) / float(n_cols)
    dh = (h_max - h_min) / float(n_rows)

    # If mode is refh_only, we don't need to grid waveform bins
    if mode_l == "refh_only":
        grid_data = np.full((n_cols, n_rows), np.nan, dtype=np.float32)
        return CrossSectionGrid(
            mode=mode,
            x_axis_mode="meters" if use_meters else "track",
            mapping_convention=mapping_convention,
            grid_data=grid_data,
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
            closure_median_3d_m=closure_3d_m,
            closure_median_horizontal_m=closure_horiz_m,
            closure_median_vertical_m=closure_vert_m,
            color_title=color_title,
            color_unit=color_unit,
        )

    # Grid waveform responses:
    # Bin indices fraction t along range window
    b_indices = np.arange(n_bins, dtype=np.float64)
    if mapping_convention == "H1":
        t_frac = b_indices / float(n_bins - 1)
    else:
        t_frac = (b_indices + 0.5) / float(n_bins)

    # Compute (X, Height) for each (track, bin)
    valid_track_mask = has_rw & np.isfinite(response).any(axis=1)
    valid_tr_indices = np.flatnonzero(valid_track_mask)

    if len(valid_tr_indices) == 0:
        grid_data = np.full((n_cols, n_rows), np.nan, dtype=np.float32)
        return CrossSectionGrid(
            mode=mode,
            x_axis_mode="meters" if use_meters else "track",
            mapping_convention=mapping_convention,
            grid_data=grid_data,
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
            closure_median_3d_m=closure_3d_m,
            closure_median_horizontal_m=closure_horiz_m,
            closure_median_vertical_m=closure_vert_m,
            color_title=color_title,
            color_unit=color_unit,
        )

    # Sub-select valid tracks
    resp_sub = response[valid_tr_indices, :]  # Shape (M, n_bins)
    rw_start_h_sub = sweep.rwstart_height[valid_tr_indices, None]
    rw_stop_h_sub = sweep.rwstop_height[valid_tr_indices, None]
    t_row = t_frac[None, :]

    # Heights for all bins
    h_matrix = rw_start_h_sub + t_row * (rw_stop_h_sub - rw_start_h_sub)

    # X coordinates for all bins
    if use_meters:
        v = sweep.cross_track_unit_vector
        c = sweep.cross_track_center
        assert v is not None and c is not None
        xs_sub = project_cross_track(sweep.rwstart_enu[valid_tr_indices, :2], v, c)[:, None]
        xe_sub = project_cross_track(sweep.rwstop_enu[valid_tr_indices, :2], v, c)[:, None]
        x_matrix = xs_sub + t_row * (xe_sub - xs_sub)
    else:
        tr_nums = sweep.track_nums[valid_tr_indices, None]
        x_matrix = np.broadcast_to(tr_nums, h_matrix.shape)

    # Flatten finite samples for binning
    fin_mask = np.isfinite(resp_sub) & np.isfinite(x_matrix) & np.isfinite(h_matrix)
    x_flat = x_matrix[fin_mask]
    h_flat = h_matrix[fin_mask]
    v_flat = resp_sub[fin_mask]

    # Compute column and row indices
    ix = np.floor((x_flat - x_min) / dx).astype(np.int64)
    iy = np.floor((h_flat - h_min) / dh).astype(np.int64)

    # Filter bounds
    in_bounds = (ix >= 0) & (ix < n_cols) & (iy >= 0) & (iy < n_rows)
    ix = ix[in_bounds]
    iy = iy[in_bounds]
    v_flat = v_flat[in_bounds]

    # Notice pyqtgraph axisOrder='col-major': array[x, y]
    # flat_idx = ix * n_rows + iy
    flat_idx = ix * n_rows + iy
    total_cells = n_cols * n_rows

    counts = np.bincount(flat_idx, minlength=total_cells)
    sums = np.bincount(flat_idx, weights=v_flat, minlength=total_cells)

    grid_flat = np.full(total_cells, np.nan, dtype=np.float32)
    has_samples = counts > 0
    grid_flat[has_samples] = (sums[has_samples] / counts[has_samples]).astype(np.float32)

    grid_data = grid_flat.reshape((n_cols, n_rows))

    return CrossSectionGrid(
        mode=mode,
        x_axis_mode="meters" if use_meters else "track",
        mapping_convention=mapping_convention,
        grid_data=grid_data,
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
        closure_median_3d_m=closure_3d_m,
        closure_median_horizontal_m=closure_horiz_m,
        closure_median_vertical_m=closure_vert_m,
        color_title=color_title,
        color_unit=color_unit,
    )
