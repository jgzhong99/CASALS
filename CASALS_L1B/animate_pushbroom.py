"""Animate the CASALS L1B refh pushbroom acquisition process.

Scientific meaning:
    Frames are reconstructed from explicit sweep_num and track_num indexing of
    official CASALS L1B refh records.

Outputs:
    Video, first-frame PNG, and metadata JSON.

This script does not:
    - reshape blindly by linear record order,
    - analyze raw waveform bins,
    - generate point-cloud products or DEMs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import math
import warnings

import h5py
import numpy as np
from pyproj import CRS, Transformer

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    h5_path: Path
    out_dir: Path

    # Output animation base name. Extension is chosen from output_format.
    output_stem: str = "casals_pushbroom_sweep_animation"
    output_format: str = "mp4"  # "mp4" or "avi". No GIF fallback.
    preferred_fourcc: str = "mp4v"  # Common MP4 option in OpenCV. Try "avc1" or "H264" if supported.
    allow_avi_fallback: bool = False  # If True, falls back to AVI/MJPG when MP4 codecs are unavailable.
    fps: int = 10
    dpi: int = 160

    # Sweep selection. Use sweep_step/max_frames to keep animation practical.
    start_sweep: int = 0
    end_sweep: Optional[int] = None
    sweep_step: int = 40
    max_frames: Optional[int] = 350

    # Show recent sweeps behind the current sweep.
    trail_sweeps: int = 160
    current_sweep_point_size: float = 18.0
    trail_point_size: float = 3.0
    trail_alpha: float = 0.20

    # Optional display filters. These filter displayed points only, not the H5 data.
    # Set display_snr_min=None to show all finite refh points.
    display_snr_min: Optional[float] = 2.0
    display_snr_max: Optional[float] = None
    display_good_snr_only: bool = False

    # Color variable for map/current-sweep plots.
    # Options: "snr", "refh", "amp", "track".
    color_by: str = "snr"
    color_min: Optional[float] = 1.0
    color_max: Optional[float] = 6.0
    cmap: str = "viridis"

    # Display-only orientation control. If enabled, the script checks the
    # displayed footprint aspect ratio after filtering. If the footprint is
    # much taller than wide, it rotates the plot coordinates by 90 degrees
    # so the single map panel uses the frame more efficiently. This does not
    # modify the georeferenced coordinates or H5-derived data.
    auto_rotate_display_by_aspect: bool = True
    auto_rotate_aspect_threshold: float = 1.25

    # Coordinate projection. If None, infer WGS84 UTM EPSG from median lon/lat.
    output_epsg_override: Optional[int] = None
    use_local_xy_origin: bool = True

    # Animation layout and styling.
    # If figsize is None, the script estimates a single-panel figure size from
    # the map extent and a detected/fallback screen aspect ratio.  This keeps
    # portrait-like and landscape-like CASALS footprints from being forced into
    # the old fixed two-panel layout.
    figsize: Optional[Tuple[float, float]] = None
    screen_aspect_override: Optional[float] = None
    fallback_screen_aspect: float = 16.0 / 9.0
    max_fig_width_in: float = 13.5
    min_fig_width_in: float = 7.5
    min_fig_height_in: float = 5.0
    min_fig_aspect: float = 0.75
    map_aspect_padding: float = 1.12
    map_xy_percentile_range: Optional[Tuple[float, float]] = None
    axis_padding_fraction: float = 0.02
    show_all_display_points_as_context: bool = True
    context_point_size: float = 0.25
    context_alpha: float = 0.05
    context_max_points: int = 500_000
    random_seed: int = 42

    # Save one static first-frame PNG for quick checking.
    write_first_frame_png: bool = True


@dataclass
class PulseArrays:
    lon: np.ndarray
    lat: np.ndarray
    refh: np.ndarray
    snr: np.ndarray
    amp: np.ndarray
    thres: np.ndarray
    good_snr: np.ndarray
    sweep_num: np.ndarray
    track_num: np.ndarray
    attrs: Dict[str, Any]


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def find_dataset(h5: h5py.File, name: str) -> Optional[h5py.Dataset]:
    """Return dataset by exact root name or recursive basename match."""
    if name in h5 and isinstance(h5[name], h5py.Dataset):
        return h5[name]

    matches = []

    def visitor(path: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and path.split("/")[-1] == name:
            matches.append(path)

    h5.visititems(visitor)
    if len(matches) == 1:
        return h5[matches[0]]
    if len(matches) > 1:
        raise ValueError(f"Multiple datasets named {name!r}: {matches}")
    return None


def require_dataset(h5: h5py.File, name: str) -> h5py.Dataset:
    ds = find_dataset(h5, name)
    if ds is None:
        raise KeyError(f"Required dataset {name!r} not found in H5 file.")
    return ds


def optional_array(h5: h5py.File, name: str, n_expected: int, dtype: Any = None) -> Optional[np.ndarray]:
    ds = find_dataset(h5, name)
    if ds is None:
        return None
    arr = np.asarray(ds[...], dtype=dtype).reshape(-1) if dtype is not None else np.asarray(ds[...]).reshape(-1)
    if arr.size != n_expected:
        warnings.warn(f"Optional dataset {name!r} has size {arr.size}, expected {n_expected}; ignoring.")
        return None
    return arr


def read_pulse_arrays(h5_path: Path) -> PulseArrays:
    with h5py.File(h5_path, "r") as h5:
        lon = np.asarray(require_dataset(h5, "refh_longitude")[...], dtype=np.float64).reshape(-1)
        lat = np.asarray(require_dataset(h5, "refh_latitude")[...], dtype=np.float64).reshape(-1)
        refh = np.asarray(require_dataset(h5, "refh")[...], dtype=np.float64).reshape(-1)
        amp = np.asarray(require_dataset(h5, "refh_amp")[...], dtype=np.float64).reshape(-1)

        thres_ds = find_dataset(h5, "refh_thres")
        if thres_ds is not None:
            thres = np.asarray(thres_ds[...], dtype=np.float64).reshape(-1)
        else:
            thres = np.full(lon.shape, np.nan, dtype=np.float64)

        snr_ds = find_dataset(h5, "refh_snr")
        if snr_ds is not None:
            snr = np.asarray(snr_ds[...], dtype=np.float64).reshape(-1)
        elif thres_ds is not None:
            snr = np.divide(amp, thres, out=np.full_like(amp, np.nan), where=(thres != 0))
        else:
            raise KeyError("Neither refh_snr nor refh_thres was found; cannot obtain SNR.")

        sweep_num = np.asarray(require_dataset(h5, "sweep_num")[...], dtype=np.int64).reshape(-1)
        track_num = np.asarray(require_dataset(h5, "track_num")[...], dtype=np.int64).reshape(-1)

        good = optional_array(h5, "good_snr", lon.size)
        good_snr = good.astype(bool) if good is not None else (snr >= 5.0)

        sizes = {
            "lon": lon.size,
            "lat": lat.size,
            "refh": refh.size,
            "amp": amp.size,
            "thres": thres.size,
            "snr": snr.size,
            "sweep_num": sweep_num.size,
            "track_num": track_num.size,
            "good_snr": good_snr.size,
        }
        if len(set(sizes.values())) != 1:
            raise ValueError(f"Dataset size mismatch: {sizes}")

        attrs: Dict[str, Any] = {}
        for key, value in h5.attrs.items():
            try:
                if isinstance(value, bytes):
                    attrs[key] = value.decode("utf-8", errors="replace")
                elif isinstance(value, np.ndarray):
                    attrs[key] = value.tolist()
                else:
                    attrs[key] = value.item() if hasattr(value, "item") else value
            except Exception:
                attrs[key] = str(value)

    return PulseArrays(
        lon=lon,
        lat=lat,
        refh=refh,
        snr=snr,
        amp=amp,
        thres=thres,
        good_snr=good_snr,
        sweep_num=sweep_num,
        track_num=track_num,
        attrs=attrs,
    )


def infer_wgs84_utm_epsg(lon: np.ndarray, lat: np.ndarray) -> int:
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    zone = int(math.floor((lon0 + 180.0) / 6.0) + 1)
    if not (1 <= zone <= 60):
        raise ValueError(f"Cannot infer UTM zone from median longitude {lon0}")
    return (32600 if lat0 >= 0 else 32700) + zone


def lonlat_to_projected(lon: np.ndarray, lat: np.ndarray, epsg: int) -> Tuple[np.ndarray, np.ndarray, CRS]:
    out_crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(CRS.from_epsg(4326), out_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), out_crs


def reshape_by_sweep_track(
    sweep: np.ndarray,
    track: np.ndarray,
    values: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build [sweep, track] grids using explicit sweep_num/track_num indexing."""
    if np.nanmin(sweep) < 0 or np.nanmin(track) < 0:
        raise ValueError("Negative sweep_num or track_num found; cannot use as direct grid indices.")

    n_sweeps = int(np.nanmax(sweep)) + 1
    n_tracks = int(np.nanmax(track)) + 1
    n_expected = n_sweeps * n_tracks

    flat = sweep.astype(np.int64) * n_tracks + track.astype(np.int64)
    unique_flat, counts = np.unique(flat, return_counts=True)
    duplicate_count = int(np.sum(counts > 1))
    missing_count = int(n_expected - unique_flat.size)

    grids: Dict[str, np.ndarray] = {}
    for name, arr in values.items():
        dtype = np.float32 if name not in {"good_snr"} else np.float32
        grid = np.full((n_sweeps, n_tracks), np.nan, dtype=dtype)
        # If duplicates exist, later records overwrite earlier ones. Duplicates are reported.
        grid[sweep, track] = arr.astype(dtype, copy=False)
        grids[name] = grid

    info = {
        "n_records": int(sweep.size),
        "n_sweeps": n_sweeps,
        "n_tracks": n_tracks,
        "expected_records_if_complete_rectangular": int(n_expected),
        "unique_sweep_track_cells": int(unique_flat.size),
        "missing_sweep_track_cells": missing_count,
        "duplicate_sweep_track_cells": duplicate_count,
        "complete_rectangular_sweep_track_grid": bool(missing_count == 0 and duplicate_count == 0 and sweep.size == n_expected),
    }
    return grids, info


def choose_color_values(grids: Dict[str, np.ndarray], cfg: Config) -> np.ndarray:
    mode = cfg.color_by.lower()
    if mode == "snr":
        return grids["snr"]
    if mode == "refh":
        return grids["refh"]
    if mode == "amp":
        return grids["amp"]
    if mode == "track":
        n_tracks = grids["snr"].shape[1]
        return np.tile(np.arange(n_tracks, dtype=np.float32), (grids["snr"].shape[0], 1))
    raise ValueError(f"Unsupported color_by={cfg.color_by!r}; use snr, refh, amp, or track.")



def display_mask_for_grid(grids: Dict[str, np.ndarray], cfg: Config) -> np.ndarray:
    mask = np.isfinite(grids["x"]) & np.isfinite(grids["y"]) & np.isfinite(grids["refh"]) & np.isfinite(grids["snr"])
    if cfg.display_snr_min is not None:
        mask &= grids["snr"] >= float(cfg.display_snr_min)
    if cfg.display_snr_max is not None:
        mask &= grids["snr"] <= float(cfg.display_snr_max)
    if cfg.display_good_snr_only:
        mask &= grids["good_snr"] > 0.5
    return mask


def robust_range(arr: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> Tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(finite, p_lo))
    hi = float(np.nanpercentile(finite, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def stable_axis_limits(
    arr: np.ndarray,
    percentile_range: Optional[Tuple[float, float]] = None,
    pad_fraction: float = 0.02,
    absolute_pad: float = 0.0,
) -> Tuple[float, float]:
    finite = np.asarray(arr, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0

    if percentile_range is not None:
        lo, hi = percentile_range
        lo_v = float(np.nanpercentile(finite, lo))
        hi_v = float(np.nanpercentile(finite, hi))
    else:
        lo_v = float(np.nanmin(finite))
        hi_v = float(np.nanmax(finite))

    if not np.isfinite(lo_v) or not np.isfinite(hi_v) or hi_v < lo_v:
        lo_v = float(np.nanmin(finite))
        hi_v = float(np.nanmax(finite))

    span = hi_v - lo_v
    if span <= 0.0:
        base = max(abs(lo_v), 1.0)
        span = base * 0.05

    pad = max(float(absolute_pad), span * max(0.0, float(pad_fraction)))
    return lo_v - pad, hi_v + pad


def apply_display_rotation_if_needed(
    grids: Dict[str, np.ndarray],
    disp_mask: np.ndarray,
    cfg: Config,
    base_x_label: str,
    base_y_label: str,
) -> Tuple[Dict[str, np.ndarray], str, str, Dict[str, Any]]:
    """Rotate display coordinates by 90 degrees when the map footprint is tall.

    This is intentionally display-only. It modifies the plotting grids in memory
    after projection and sweep/track reshaping, but the metadata records the
    operation so the original projected-coordinate meaning remains explicit.
    """
    x = grids["x"]
    y = grids["y"]
    x_vals = np.asarray(x[disp_mask], dtype=np.float64)
    y_vals = np.asarray(y[disp_mask], dtype=np.float64)

    x_lo, x_hi = stable_axis_limits(x_vals, percentile_range=cfg.map_xy_percentile_range, pad_fraction=0.0)
    y_lo, y_hi = stable_axis_limits(y_vals, percentile_range=cfg.map_xy_percentile_range, pad_fraction=0.0)
    x_span = max(float(x_hi - x_lo), 1e-9)
    y_span = max(float(y_hi - y_lo), 1e-9)
    aspect_y_over_x = y_span / x_span

    rotate = bool(cfg.auto_rotate_display_by_aspect and aspect_y_over_x >= float(cfg.auto_rotate_aspect_threshold))
    info = {
        "auto_rotate_display_by_aspect": bool(cfg.auto_rotate_display_by_aspect),
        "auto_rotate_aspect_threshold": float(cfg.auto_rotate_aspect_threshold),
        "pre_rotation_x_span_m": float(x_span),
        "pre_rotation_y_span_m": float(y_span),
        "pre_rotation_y_over_x_aspect": float(aspect_y_over_x),
        "display_rotated_90deg": rotate,
        "rotation_semantics": "plot_x = original_y, plot_y = -original_x" if rotate else "none",
    }

    if not rotate:
        return grids, base_x_label, base_y_label, info

    new_grids = dict(grids)
    new_grids["x"] = y.copy()
    new_grids["y"] = -x.copy()
    x_label = f"{base_y_label} (display X; auto-rotated)"
    y_label = f"-{base_x_label} (display Y; auto-rotated)"
    return new_grids, x_label, y_label, info



def detect_screen_aspect(fallback: float = 16.0 / 9.0) -> float:
    """Return the primary screen width/height ratio when Tk can see a display.

    The function is deliberately optional: batch/HPC/headless runs fall back to
    a conventional 16:9-like ratio without failing the animation.
    """
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = float(root.winfo_screenwidth())
        height = float(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0 and np.isfinite(width / height):
            return width / height
    except Exception:
        pass
    return float(fallback)


def infer_single_panel_figsize(
    map_x_range: Tuple[float, float],
    map_y_range: Tuple[float, float],
    cfg: Config,
) -> Tuple[Tuple[float, float], Dict[str, float]]:
    """Infer a readable one-panel figure size from map and screen aspect.

    The map axes keep equal x/y scaling.  The figure aspect is therefore based
    mainly on the displayed map extent, but it is bounded so the rendered video
    remains practical on a normal monitor.
    """
    if cfg.figsize is not None:
        w, h = float(cfg.figsize[0]), float(cfg.figsize[1])
        return (w, h), {
            "figure_width_in": w,
            "figure_height_in": h,
            "screen_aspect": float("nan"),
            "map_aspect": float("nan"),
            "target_aspect": w / h if h > 0 else float("nan"),
            "figsize_mode": "manual",
        }

    screen_aspect = (
        float(cfg.screen_aspect_override)
        if cfg.screen_aspect_override is not None
        else detect_screen_aspect(cfg.fallback_screen_aspect)
    )
    if not np.isfinite(screen_aspect) or screen_aspect <= 0:
        screen_aspect = float(cfg.fallback_screen_aspect)

    dx = max(float(map_x_range[1] - map_x_range[0]), 1e-6)
    dy = max(float(map_y_range[1] - map_y_range[0]), 1e-6)
    map_aspect = dx / dy

    # Add mild padding for labels/colorbar/title, then bound to a monitor-
    # friendly range.  Very tall footprints remain portrait-ish instead of
    # being forced into 16:9, but not so narrow that labels become unreadable.
    target_aspect = map_aspect * float(cfg.map_aspect_padding)
    lower = max(float(cfg.min_fig_aspect), 1.0 / max(screen_aspect, 1e-6))
    upper = max(lower, screen_aspect)
    target_aspect = min(max(target_aspect, lower), upper)

    max_w = float(cfg.max_fig_width_in)
    max_h = max(float(cfg.min_fig_height_in), max_w / screen_aspect)
    if target_aspect >= screen_aspect:
        fig_w = max_w
        fig_h = fig_w / target_aspect
    else:
        fig_h = max_h
        fig_w = fig_h * target_aspect

    if fig_w < float(cfg.min_fig_width_in):
        fig_w = float(cfg.min_fig_width_in)
        fig_h = fig_w / target_aspect
    if fig_h < float(cfg.min_fig_height_in):
        fig_h = float(cfg.min_fig_height_in)
        fig_w = fig_h * target_aspect

    return (float(fig_w), float(fig_h)), {
        "figure_width_in": float(fig_w),
        "figure_height_in": float(fig_h),
        "screen_aspect": float(screen_aspect),
        "map_aspect": float(map_aspect),
        "target_aspect": float(target_aspect),
        "figsize_mode": "auto_single_panel",
    }


def resolve_config_path(path: Path) -> Path:
    """Resolve config paths relative to the script location, not the launch CWD."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def make_frame_sweeps(n_sweeps: int, cfg: Config) -> np.ndarray:
    start = max(0, int(cfg.start_sweep))
    end = n_sweeps - 1 if cfg.end_sweep is None else min(n_sweeps - 1, int(cfg.end_sweep))
    if end < start:
        raise ValueError(f"Invalid sweep range: start={start}, end={end}")

    step = max(1, int(cfg.sweep_step))
    sweeps = np.arange(start, end + 1, step, dtype=np.int64)
    if cfg.max_frames is not None and sweeps.size > cfg.max_frames:
        indices = np.linspace(0, sweeps.size - 1, cfg.max_frames).round().astype(np.int64)
        sweeps = sweeps[np.unique(indices)]
    return sweeps



def canvas_to_bgr_frame(fig: plt.Figure, even_size: bool = True) -> np.ndarray:
    """Render a Matplotlib figure canvas to an OpenCV BGR uint8 frame."""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
    if even_size:
        # Many MP4 encoders prefer even dimensions. Crop one pixel if needed.
        h_even = height - (height % 2)
        w_even = width - (width % 2)
        rgba = rgba[:h_even, :w_even, :]
    rgb = rgba[:, :, :3]
    return rgb[:, :, ::-1].copy()  # RGB -> BGR for OpenCV


def build_video_writer(path: Path, fps: int, frame_size: Tuple[int, int], fourcc_text: str):
    """Create and validate an OpenCV VideoWriter."""
    if cv2 is None:
        raise RuntimeError(
            "OpenCV/cv2 is not available. Install it with: conda install -c conda-forge opencv. "
            f"Original import error: {_CV2_IMPORT_ERROR}"
        )
    fourcc = cv2.VideoWriter_fourcc(*fourcc_text)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), frame_size, True)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"OpenCV VideoWriter failed to open: path={path}, fourcc={fourcc_text}, size={frame_size}")
    return writer


def open_video_writer_with_fallbacks(cfg: Config, first_frame_bgr: np.ndarray) -> Tuple[Any, Path, str, str]:
    """Open a video writer. Prefer MP4, optionally fall back to AVI. Never writes GIF."""
    h, w = first_frame_bgr.shape[:2]
    frame_size = (int(w), int(h))
    fmt = cfg.output_format.lower().lstrip(".")

    attempts = []
    if fmt == "mp4":
        # MP4 support depends on how OpenCV was built. Try common FourCCs.
        fourccs = []
        for code in [cfg.preferred_fourcc, "mp4v", "MP4V", "avc1", "H264"]:
            if code and code not in fourccs:
                fourccs.append(code)
        attempts.extend([(cfg.out_dir / f"{cfg.output_stem}.mp4", code) for code in fourccs])
        if cfg.allow_avi_fallback:
            attempts.append((cfg.out_dir / f"{cfg.output_stem}.avi", "MJPG"))
    elif fmt == "avi":
        attempts.extend([
            (cfg.out_dir / f"{cfg.output_stem}.avi", cfg.preferred_fourcc if cfg.preferred_fourcc else "MJPG"),
            (cfg.out_dir / f"{cfg.output_stem}.avi", "MJPG"),
            (cfg.out_dir / f"{cfg.output_stem}.avi", "XVID"),
        ])
    else:
        raise ValueError("output_format must be 'mp4' or 'avi'. GIF output is intentionally disabled.")

    errors = []
    for out_path, fourcc_text in attempts:
        try:
            writer = build_video_writer(out_path, cfg.fps, frame_size, fourcc_text)
            return writer, out_path, out_path.suffix.lower().lstrip("."), fourcc_text
        except Exception as exc:
            errors.append(f"{out_path.name} / {fourcc_text}: {exc}")

    raise RuntimeError(
        "Could not open any OpenCV VideoWriter. No GIF fallback is used.\n"
        "Tried:\n  - " + "\n  - ".join(errors) + "\n"
        "Try installing a conda-forge OpenCV build, or set output_format='avi', preferred_fourcc='MJPG', "
        "or set allow_avi_fallback=True."
    )

def write_metadata(path: Path, cfg: Config, grid_info: Dict[str, Any], out_crs: CRS, outputs: Dict[str, Any], extra: Dict[str, Any]) -> None:
    meta = {
        "script": "animate_pushbroom.py",
        "script_semantics": "CASALS_L1B_sweep_pushbroom_animation_from_refh_reference_return_records",
        "scientific_notes": [
            "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
            "refh is WGS84 ellipsoidal height unless otherwise documented.",
            "This is not an official multi-return point cloud.",
            "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            "Sweeps are reconstructed using explicit sweep_num and track_num indexing.",
        ],
        "config": asdict(cfg),
        "output_crs": out_crs.to_string(),
        "grid_info": grid_info,
        "outputs": outputs,
        "extra": extra,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(meta), f, indent=2)


def main() -> None:
    # -------------------------------------------------------------------------
    # USER SETTINGS: edit here.
    # -------------------------------------------------------------------------
    cfg = Config(
        h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
        out_dir=Path(r"./outputs/animate_pushbroom"),

        output_stem="casals_20241112_pushbroom_refh_snr",
        output_format="mp4",  # written with OpenCV VideoWriter. No GIF fallback.
        preferred_fourcc="mp4v",
        allow_avi_fallback=False,
        fps=5,
        dpi=300,

        # Current file has 14080 sweeps. To avoid a huge video, render a subset.
        start_sweep=0,
        end_sweep=None,
        sweep_step=40,
        max_frames=3500,

        # Trail controls how much previous along-track history appears behind current sweep.
        trail_sweeps=160,

        # SNR display filter. Set to None to show all finite points.
        display_snr_min=3.0,
        display_snr_max=None,
        display_good_snr_only=False,

        # Coloring.
        color_by="snr",            # snr, refh, amp, or track
        color_min=1.0,
        color_max=6.0,
        cmap="viridis",

        # Display orientation. Auto-rotation is display-only and is recorded in metadata.
        auto_rotate_display_by_aspect=True,
        auto_rotate_aspect_threshold=1.25,

        # Coordinate output. For this granule, automatic inference should give EPSG:32618.
        output_epsg_override=None,
        use_local_xy_origin=True,

        show_all_display_points_as_context=True,
        context_max_points=500_000,
        write_first_frame_png=True,
    )
    # -------------------------------------------------------------------------

    cfg.h5_path = resolve_config_path(cfg.h5_path)
    cfg.out_dir = resolve_config_path(cfg.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("CASALS L1B sweep / pushbroom animation")
    print("=" * 88)
    print(f"H5: {cfg.h5_path.resolve()}")
    print(f"Output dir: {cfg.out_dir.resolve()}")

    pd = read_pulse_arrays(cfg.h5_path)

    finite_geo = (
        np.isfinite(pd.lon)
        & np.isfinite(pd.lat)
        & np.isfinite(pd.refh)
        & np.isfinite(pd.snr)
        & (pd.lon >= -180.0)
        & (pd.lon <= 180.0)
        & (pd.lat >= -90.0)
        & (pd.lat <= 90.0)
    )
    if finite_geo.sum() == 0:
        raise ValueError("No finite georeferenced refh records found.")

    epsg = cfg.output_epsg_override or infer_wgs84_utm_epsg(pd.lon[finite_geo], pd.lat[finite_geo])
    x, y, out_crs = lonlat_to_projected(pd.lon, pd.lat, epsg)

    if cfg.use_local_xy_origin:
        x0 = float(np.nanmedian(x[finite_geo]))
        y0 = float(np.nanmedian(y[finite_geo]))
        x_plot = x - x0
        y_plot = y - y0
        xy_units = f"local projected meters, origin=median EPSG:{epsg} ({x0:.3f}, {y0:.3f})"
        x_label = "Local X (m)"
        y_label = "Local Y (m)"
        map_title = f"Map view | local EPSG:{epsg}"
    else:
        x0 = 0.0
        y0 = 0.0
        x_plot = x
        y_plot = y
        xy_units = f"EPSG:{epsg} projected meters"
        x_label = "Projected X (m)"
        y_label = "Projected Y (m)"
        map_title = f"Map view | EPSG:{epsg}"

    grids, grid_info = reshape_by_sweep_track(
        pd.sweep_num,
        pd.track_num,
        {
            "x": x_plot,
            "y": y_plot,
            "refh": pd.refh,
            "snr": pd.snr,
            "amp": pd.amp,
            "thres": pd.thres,
            "good_snr": pd.good_snr.astype(np.float32),
        },
    )

    n_sweeps = grid_info["n_sweeps"]
    n_tracks = grid_info["n_tracks"]
    frame_sweeps = make_frame_sweeps(n_sweeps, cfg)
    disp_mask = display_mask_for_grid(grids, cfg)

    x_disp_pre = grids["x"][disp_mask]
    if x_disp_pre.size == 0:
        raise ValueError("Display filter removed all points. Lower display_snr_min or disable display_good_snr_only.")

    grids, x_label, y_label, rotation_info = apply_display_rotation_if_needed(
        grids, disp_mask, cfg, x_label, y_label
    )
    if rotation_info["display_rotated_90deg"]:
        map_title = (
            f"Map view | local EPSG:{epsg} | display rotated 90°"
            if cfg.use_local_xy_origin
            else f"Map view | EPSG:{epsg} | display rotated 90°"
        )
    color_grid = choose_color_values(grids, cfg)

    cmin, cmax = robust_range(color_grid[disp_mask], 2, 98)
    if cfg.color_min is not None:
        cmin = float(cfg.color_min)
    if cfg.color_max is not None:
        cmax = float(cfg.color_max)
    if cmax <= cmin:
        cmax = cmin + 1.0

    x_disp = grids["x"][disp_mask]
    y_disp = grids["y"][disp_mask]

    map_x_lo, map_x_hi = stable_axis_limits(
        x_disp,
        percentile_range=cfg.map_xy_percentile_range,
        pad_fraction=cfg.axis_padding_fraction,
    )
    map_y_lo, map_y_hi = stable_axis_limits(
        y_disp,
        percentile_range=cfg.map_xy_percentile_range,
        pad_fraction=cfg.axis_padding_fraction,
    )

    print(f"Records: {pd.lon.size:,}")
    print(f"Sweeps x tracks: {n_sweeps} x {n_tracks} = {n_sweeps*n_tracks:,}")
    print(f"Complete rectangular sweep-track grid: {grid_info['complete_rectangular_sweep_track_grid']}")
    print(f"Display points after filter: {int(disp_mask.sum()):,}")
    print(f"Rendered animation frames: {frame_sweeps.size}")
    print(f"Color by: {cfg.color_by}, color range: {cmin:.3f} to {cmax:.3f}")
    print(f"Display rotation: {rotation_info['rotation_semantics']} (pre y/x aspect={rotation_info['pre_rotation_y_over_x_aspect']:.3f})")

    fig_size, layout_info = infer_single_panel_figsize((map_x_lo, map_x_hi), (map_y_lo, map_y_hi), cfg)
    print(
        "Figure layout: "
        f"{fig_size[0]:.2f} x {fig_size[1]:.2f} in | "
        f"map aspect={layout_info['map_aspect']:.3f} | "
        f"screen aspect={layout_info['screen_aspect']:.3f}"
    )

    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.0, 0.045],
        hspace=0.08,
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.11,
    )
    ax_map = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[1, 0])

    cmap = plt.get_cmap(cfg.cmap)
    norm = plt.Normalize(cmin, cmax)

    # Optional context: a very light scatter of all display-selected points.
    context_artist = None
    if cfg.show_all_display_points_as_context:
        rng = np.random.default_rng(cfg.random_seed)
        valid_indices = np.flatnonzero(disp_mask.ravel())
        if valid_indices.size > cfg.context_max_points:
            valid_indices = rng.choice(valid_indices, size=cfg.context_max_points, replace=False)
        all_x = grids["x"].ravel()[valid_indices]
        all_y = grids["y"].ravel()[valid_indices]
        context_artist = ax_map.scatter(
            all_x,
            all_y,
            s=cfg.context_point_size,
            c="0.5",
            alpha=cfg.context_alpha,
            linewidths=0,
            rasterized=True,
        )

    trail_scatter = ax_map.scatter([], [], s=cfg.trail_point_size, c=[], cmap=cmap, norm=norm, alpha=cfg.trail_alpha, linewidths=0, rasterized=True)
    current_scatter = ax_map.scatter([], [], s=cfg.current_sweep_point_size, c=[], cmap=cmap, norm=norm, alpha=0.95, edgecolors="k", linewidths=0.25)
    current_line = LineCollection([], colors="black", linewidths=0.8, alpha=0.8)
    ax_map.add_collection(current_line)

    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel(x_label)
    ax_map.set_ylabel(y_label)
    ax_map.set_xlim(map_x_lo, map_x_hi)
    ax_map.set_ylim(map_y_lo, map_y_hi)
    ax_map.grid(True, linewidth=0.3, alpha=0.35)
    ax_map.set_title(map_title, fontsize=10)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(cfg.color_by)
    cb.outline.set_linewidth(0.6)

    title = fig.suptitle("", fontsize=11.5)

    def get_sweep_data(s: int, include_filter: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = np.isfinite(grids["x"][s]) & np.isfinite(grids["y"][s]) & np.isfinite(color_grid[s])
        if include_filter:
            m &= disp_mask[s]
        tracks = np.arange(n_tracks, dtype=np.float64)[m]
        xs = grids["x"][s, m]
        ys = grids["y"][s, m]
        cols = color_grid[s, m]
        return tracks, xs, ys, cols

    def update(frame_idx: int):
        s = int(frame_sweeps[frame_idx])
        trail_start = max(0, s - int(cfg.trail_sweeps))
        trail_rows = slice(trail_start, s + 1)
        trail_m = disp_mask[trail_rows]
        tx = grids["x"][trail_rows][trail_m]
        ty = grids["y"][trail_rows][trail_m]
        tc = color_grid[trail_rows][trail_m]

        if tx.size:
            trail_scatter.set_offsets(np.column_stack((tx, ty)))
            trail_scatter.set_array(tc.astype(np.float64))
        else:
            trail_scatter.set_offsets(np.empty((0, 2)))
            trail_scatter.set_array(np.asarray([], dtype=np.float64))

        tracks, xs, ys, cols = get_sweep_data(s, include_filter=True)
        if xs.size:
            current_scatter.set_offsets(np.column_stack((xs, ys)))
            current_scatter.set_array(cols.astype(np.float64))
            # Draw a polyline connecting the current sweep in track order.
            segments = []
            if xs.size >= 2:
                order = np.argsort(tracks)
                pts = np.column_stack((xs[order], ys[order]))
                segments = [pts]
            current_line.set_segments(segments)
        else:
            current_scatter.set_offsets(np.empty((0, 2)))
            current_scatter.set_array(np.asarray([], dtype=np.float64))
            current_line.set_segments([])

        title.set_text(
            f"CASALS L1B pushbroom | sweep {s:,}/{n_sweeps-1:,} | "
            f"shown {xs.size}/{n_tracks} | trail {trail_start:,}-{s:,}"
        )
        return trail_scatter, current_scatter, current_line, title

    # Render the first frame once to determine exact pixel size for OpenCV VideoWriter.
    update(0)
    first_frame_bgr = canvas_to_bgr_frame(fig, even_size=True)

    if cfg.write_first_frame_png:
        first_png = cfg.out_dir / f"{cfg.output_stem}_first_frame.png"
        fig.savefig(first_png, dpi=cfg.dpi, bbox_inches="tight")
    else:
        first_png = None

    writer, out_path, actual_format, actual_fourcc = open_video_writer_with_fallbacks(cfg, first_frame_bgr)

    # Important: write all frames with exactly the same pixel size.
    for frame_idx in range(frame_sweeps.size):
        update(frame_idx)
        frame_bgr = canvas_to_bgr_frame(fig, even_size=True)
        if frame_bgr.shape[:2] != first_frame_bgr.shape[:2]:
            raise RuntimeError(
                f"Frame size changed at frame {frame_idx}: {frame_bgr.shape[:2]} vs {first_frame_bgr.shape[:2]}"
            )
        writer.write(frame_bgr)
        if (frame_idx + 1) % 25 == 0 or frame_idx == 0 or frame_idx + 1 == frame_sweeps.size:
            print(f"  wrote video frame {frame_idx + 1:,}/{frame_sweeps.size:,}")
    writer.release()
    plt.close(fig)

    outputs = {
        "animation": str(out_path),
        "animation_format": actual_format,
        "video_fourcc": actual_fourcc,
        "first_frame_png": str(first_png) if first_png is not None else None,
        "metadata_json": str(cfg.out_dir / f"{cfg.output_stem}_metadata.json"),
    }
    extra = {
        "h5_path": str(cfg.h5_path),
        "records": int(pd.lon.size),
        "finite_georeferenced_records": int(finite_geo.sum()),
        "display_points_after_filter": int(disp_mask.sum()),
        "frame_sweep_count": int(frame_sweeps.size),
        "first_frame_sweep": int(frame_sweeps[0]),
        "last_frame_sweep": int(frame_sweeps[-1]),
        "sweep_numbers_rendered_sample": [int(v) for v in frame_sweeps[:10]],
        "xy_origin_offset_applied": {"x0": x0, "y0": y0} if cfg.use_local_xy_origin else None,
        "color_range": {"min": cmin, "max": cmax},
        "map_x_range": {"min": map_x_lo, "max": map_x_hi},
        "map_y_range": {"min": map_y_lo, "max": map_y_hi},
        "display_rotation": rotation_info,
        "single_panel_layout": layout_info,
        "source_attrs_subset": pd.attrs,
    }
    metadata_path = cfg.out_dir / f"{cfg.output_stem}_metadata.json"
    write_metadata(metadata_path, cfg, grid_info, out_crs, outputs, extra)

    print("\nDone.")
    print(json.dumps(_json_safe({
        "animation": str(out_path),
        "metadata": str(metadata_path),
        "first_frame_png": str(first_png) if first_png is not None else None,
        "n_sweeps": n_sweeps,
        "n_tracks": n_tracks,
        "n_frames": int(frame_sweeps.size),
        "display_points_after_filter": int(disp_mask.sum()),
        "output_crs": out_crs.to_string(),
    }), indent=2))


if __name__ == "__main__":
    main()
