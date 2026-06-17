"""Animate CASALS L1B receive waveforms sweep by sweep.

Scientific meaning:
    Each frame is an `rx_bins x tracks` image built from explicit sweep_num and
    track_num indexing of the raw `rx_waveform` dataset.

Outputs:
    Video, first-frame PNG, and metadata JSON.

This script does not:
    - perform waveform decomposition,
    - generate point-cloud products,
    - infer DEM/DSM surfaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import json
import math
import warnings

import h5py
import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class Config:
    h5_path: Path
    out_dir: Path

    output_stem: str = "casals_20241112_rx_waveform_sweep_movie"
    output_format: str = "mp4"  # "mp4" or "avi"; no GIF fallback.
    preferred_fourcc: str = "mp4v"  # Try "avc1" or "H264" if your OpenCV build supports them.
    allow_avi_fallback: bool = False
    fps: float = 30.0

    # Frame selection. sweep_step=1 means one video frame for every sweep.
    start_sweep: int = 0
    end_sweep: Optional[int] = None
    sweep_step: int = 1
    max_frames: Optional[int] = None  # Leave None to keep one frame per selected sweep.

    # Waveform display transformation.
    # "raw" uses rx_waveform amplitude directly.
    # "minus_bg_mean" subtracts bg_mean per pulse if bg_mean exists.
    # "minus_refh_thres" subtracts refh_thres per pulse if refh_thres exists.
    background_mode: str = "raw"  # "raw", "minus_bg_mean", "minus_refh_thres"
    clip_negative_after_background: bool = True

    # Value transform applied after background processing.
    # log1p or asinh usually makes weaker returns more visible than linear.
    value_transform: str = "log1p"  # "linear", "log1p", "sqrt", "asinh"
    asinh_scale: float = 100.0

    # Global display range. If None, estimated from sampled sweeps after transform.
    display_vmin: Optional[float] = None
    display_vmax: Optional[float] = None
    range_percentiles: Tuple[float, float] = (1.0, 99.7)
    range_estimation_max_sweeps: int = 80
    range_estimation_bin_stride: int = 4
    range_estimation_track_stride: int = 1

    # Colormap. Options depend on OpenCV version; TURBO/VIRIDIS/INFERNO are common.
    colormap: str = "turbo"  # "turbo", "viridis", "inferno", "magma", "plasma", "gray"

    # Native waveform matrix is n_rx_bins x n_tracks, e.g. 2728 x 256.
    # Scaling changes video display size but not the data model.
    # Use x=1, y=1 for exact 256x2728 frame size.
    display_scale_x: float = 4.0
    display_scale_y: float = 0.5
    resize_interpolation: str = "area"  # "nearest", "linear", "area"

    # Bin orientation. True means bin 0 appears at the top row.
    bin0_at_top: bool = True

    # Optional visual overlays. These modify the video pixels but not the data.
    overlay_max_bin: bool = True
    overlay_refh_bin_from_waveform_max: bool = True  # max bin is computed from this sweep's rx waveform.
    overlay_color_bgr: Tuple[int, int, int] = (0, 0, 255)  # red in BGR.
    overlay_radius_px: int = 1
    overlay_polyline: bool = True

    # Optional text. Turn off for a pure waveform image.
    add_text_overlay: bool = True
    text_color_bgr: Tuple[int, int, int] = (255, 255, 255)
    text_shadow_bgr: Tuple[int, int, int] = (0, 0, 0)
    text_scale: float = 0.5
    text_thickness: int = 1

    # Output diagnostics.
    write_first_frame_png: bool = True
    write_metadata_json: bool = True
    validate_written_video: bool = True

    # Reproducibility / progress.
    random_seed: int = 42
    progress_every_n_frames: int = 100


@dataclass
class PulseIndexInfo:
    sweep_num: np.ndarray
    track_num: np.ndarray
    record_index_grid: np.ndarray
    n_sweeps: int
    n_tracks: int
    n_records: int
    complete_rectangular_grid: bool
    duplicate_sweep_track_cells: int
    missing_sweep_track_cells: int
    attrs: Dict[str, Any]


# -----------------------------------------------------------------------------
# H5 helpers
# -----------------------------------------------------------------------------


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
        raise KeyError(f"Required dataset {name!r} was not found in H5 file.")
    return ds


def optional_1d_array(h5: h5py.File, name: str, n_expected: int, dtype: Any = np.float64) -> Optional[np.ndarray]:
    ds = find_dataset(h5, name)
    if ds is None:
        return None
    arr = np.asarray(ds[...], dtype=dtype).reshape(-1)
    if arr.size != n_expected:
        warnings.warn(f"Optional dataset {name!r} has size {arr.size}, expected {n_expected}; ignored.")
        return None
    return arr


def read_attrs(h5: h5py.File) -> Dict[str, Any]:
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
    return attrs


def build_record_index_grid(h5: h5py.File, rx_ds: h5py.Dataset) -> Tuple[PulseIndexInfo, int, int]:
    """Build record_index_grid[sweep, track] = flat pulse record index.

    Returns (index_info, n_records, n_rx_bins). Handles rx_waveform stored as
    either [n_records, n_bins] or [n_bins, n_records], though the first form is
    expected for current CASALS L1B files.
    """
    sweep = np.asarray(require_dataset(h5, "sweep_num")[...], dtype=np.int64).reshape(-1)
    track = np.asarray(require_dataset(h5, "track_num")[...], dtype=np.int64).reshape(-1)
    if sweep.size != track.size:
        raise ValueError(f"sweep_num and track_num sizes differ: {sweep.size} vs {track.size}")
    if sweep.size == 0:
        raise ValueError("No sweep/track records found.")
    if int(np.nanmin(sweep)) < 0 or int(np.nanmin(track)) < 0:
        raise ValueError("Negative sweep_num or track_num found; cannot use direct grid indexing.")

    n_records = int(sweep.size)
    if rx_ds.ndim != 2:
        raise ValueError(f"Expected rx_waveform to be 2D, got shape={rx_ds.shape}")

    if rx_ds.shape[0] == n_records:
        record_axis = 0
        n_rx_bins = int(rx_ds.shape[1])
    elif rx_ds.shape[1] == n_records:
        record_axis = 1
        n_rx_bins = int(rx_ds.shape[0])
    else:
        raise ValueError(
            f"rx_waveform shape {rx_ds.shape} does not match n_records={n_records} on either axis."
        )

    n_sweeps = int(np.nanmax(sweep)) + 1
    n_tracks = int(np.nanmax(track)) + 1
    n_expected = n_sweeps * n_tracks

    flat = sweep * n_tracks + track
    counts = np.bincount(flat, minlength=n_expected)
    duplicate_cells = int(np.sum(counts > 1))
    missing_cells = int(np.sum(counts == 0))

    record_index_grid = np.full((n_sweeps, n_tracks), -1, dtype=np.int64)
    record_index_grid[sweep, track] = np.arange(n_records, dtype=np.int64)

    info = PulseIndexInfo(
        sweep_num=sweep,
        track_num=track,
        record_index_grid=record_index_grid,
        n_sweeps=n_sweeps,
        n_tracks=n_tracks,
        n_records=n_records,
        complete_rectangular_grid=bool(missing_cells == 0 and duplicate_cells == 0 and n_records == n_expected),
        duplicate_sweep_track_cells=duplicate_cells,
        missing_sweep_track_cells=missing_cells,
        attrs=read_attrs(h5),
    )
    return info, record_axis, n_rx_bins


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------


def selected_sweeps(n_sweeps: int, cfg: Config) -> np.ndarray:
    start = max(0, int(cfg.start_sweep))
    end = n_sweeps - 1 if cfg.end_sweep is None else min(n_sweeps - 1, int(cfg.end_sweep))
    if end < start:
        raise ValueError(f"Invalid sweep range: start={start}, end={end}")
    step = max(1, int(cfg.sweep_step))
    sweeps = np.arange(start, end + 1, step, dtype=np.int64)
    if cfg.max_frames is not None and sweeps.size > int(cfg.max_frames):
        # This intentionally reduces frames; leave max_frames=None for exact one-frame-per-selected-sweep.
        idx = np.linspace(0, sweeps.size - 1, int(cfg.max_frames)).round().astype(np.int64)
        sweeps = sweeps[np.unique(idx)]
    return sweeps


def get_cv2_colormap(name: str) -> Optional[int]:
    name_l = name.lower()
    if name_l == "gray":
        return None
    mapping = {
        "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
        "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
        "inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
        "magma": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
        "plasma": getattr(cv2, "COLORMAP_PLASMA", cv2.COLORMAP_JET),
        "jet": cv2.COLORMAP_JET,
    }
    if name_l not in mapping:
        raise ValueError(f"Unsupported colormap={name!r}; use {sorted(mapping) + ['gray']}")
    return mapping[name_l]


def transform_values(values: np.ndarray, cfg: Config) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if cfg.clip_negative_after_background:
        arr = np.maximum(arr, 0.0)
    mode = cfg.value_transform.lower()
    if mode == "linear":
        return arr
    if mode == "log1p":
        return np.log1p(arr)
    if mode == "sqrt":
        return np.sqrt(np.maximum(arr, 0.0))
    if mode == "asinh":
        scale = float(cfg.asinh_scale)
        if not np.isfinite(scale) or scale <= 0:
            scale = 100.0
        return np.arcsinh(arr / scale)
    raise ValueError("value_transform must be one of: linear, log1p, sqrt, asinh")


def interpolation_flag(name: str) -> int:
    name_l = name.lower()
    if name_l == "nearest":
        return cv2.INTER_NEAREST
    if name_l == "linear":
        return cv2.INTER_LINEAR
    if name_l == "area":
        return cv2.INTER_AREA
    raise ValueError("resize_interpolation must be nearest, linear, or area")


def read_optional_background_values(h5: h5py.File, info: PulseIndexInfo, cfg: Config) -> Optional[np.ndarray]:
    mode = cfg.background_mode.lower()
    if mode == "raw":
        return None
    name = None
    if mode == "minus_bg_mean":
        name = "bg_mean"
    elif mode == "minus_refh_thres":
        name = "refh_thres"
    else:
        raise ValueError("background_mode must be raw, minus_bg_mean, or minus_refh_thres")
    arr = optional_1d_array(h5, name, info.n_records, dtype=np.float32)
    if arr is None:
        raise KeyError(f"background_mode={cfg.background_mode!r} requires dataset {name!r}, but it was not found.")
    return arr


def read_sweep_waveform(
    rx_ds: h5py.Dataset,
    info: PulseIndexInfo,
    sweep_idx: int,
    record_axis: int,
    n_rx_bins: int,
    background_values: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return waveform image as [n_bins, n_tracks], track_valid, max_bin_by_track.

    The returned waveform matrix is in track-column order. Missing sweep-track
    cells are filled with NaN. max_bin_by_track is -1 for missing tracks.
    """
    idx_by_track = info.record_index_grid[int(sweep_idx)]
    valid_tracks = np.flatnonzero(idx_by_track >= 0)
    valid_idx = idx_by_track[valid_tracks]

    wave_track_bin = np.full((info.n_tracks, n_rx_bins), np.nan, dtype=np.float32)
    if valid_idx.size == 0:
        return wave_track_bin.T, valid_tracks, np.full(info.n_tracks, -1, dtype=np.int32)

    # h5py fancy indexing requires explicit indices in increasing order. Read sorted,
    # then reorder back to track order.
    sort_order = np.argsort(valid_idx)
    sorted_idx = valid_idx[sort_order]
    inverse_order = np.empty_like(sort_order)
    inverse_order[sort_order] = np.arange(sort_order.size)

    if record_axis == 0:
        data_sorted = np.asarray(rx_ds[sorted_idx, :], dtype=np.float32)
    else:
        data_sorted = np.asarray(rx_ds[:, sorted_idx], dtype=np.float32).T
    data = data_sorted[inverse_order, :]

    if background_values is not None:
        bg_sorted = np.asarray(background_values[sorted_idx], dtype=np.float32)
        bg = bg_sorted[inverse_order]
        data = data - bg[:, None]

    wave_track_bin[valid_tracks, :] = data

    max_bin_by_track = np.full(info.n_tracks, -1, dtype=np.int32)
    finite_row = np.isfinite(wave_track_bin).any(axis=1)
    if finite_row.any():
        # nanargmax fails on all-NaN rows; fill NaNs with -inf first.
        safe = np.where(np.isfinite(wave_track_bin), wave_track_bin, -np.inf)
        max_bin_by_track[finite_row] = np.argmax(safe[finite_row], axis=1).astype(np.int32)

    return wave_track_bin.T, valid_tracks, max_bin_by_track


def estimate_display_range(
    h5: h5py.File,
    rx_ds: h5py.Dataset,
    info: PulseIndexInfo,
    record_axis: int,
    n_rx_bins: int,
    frame_sweeps: np.ndarray,
    background_values: Optional[np.ndarray],
    cfg: Config,
) -> Tuple[float, float]:
    if cfg.display_vmin is not None and cfg.display_vmax is not None:
        return float(cfg.display_vmin), float(cfg.display_vmax)

    n_sample = min(int(cfg.range_estimation_max_sweeps), int(frame_sweeps.size))
    if n_sample <= 0:
        raise ValueError("No sweeps available for display range estimation.")
    if n_sample == frame_sweeps.size:
        sample_sweeps = frame_sweeps
    else:
        rng = np.random.default_rng(cfg.random_seed)
        sample_sweeps = np.sort(rng.choice(frame_sweeps, size=n_sample, replace=False))

    values = []
    iterator: Iterable[int]
    iterator = sample_sweeps
    if tqdm is not None:
        iterator = tqdm(sample_sweeps, desc="Estimating waveform display range")
    for s in iterator:
        wf, _, _ = read_sweep_waveform(rx_ds, info, int(s), record_axis, n_rx_bins, background_values)
        wf = wf[:: max(1, int(cfg.range_estimation_bin_stride)), :: max(1, int(cfg.range_estimation_track_stride))]
        tv = transform_values(wf, cfg)
        finite = tv[np.isfinite(tv)]
        if finite.size:
            values.append(finite.astype(np.float32, copy=False))
    if not values:
        raise ValueError("No finite waveform values found for display range estimation.")
    sample = np.concatenate(values)
    lo_p, hi_p = cfg.range_percentiles
    vmin = float(np.nanpercentile(sample, lo_p)) if cfg.display_vmin is None else float(cfg.display_vmin)
    vmax = float(np.nanpercentile(sample, hi_p)) if cfg.display_vmax is None else float(cfg.display_vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(sample))
        vmax = float(np.nanmax(sample))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def waveform_to_bgr_frame(
    waveform_bins_tracks: np.ndarray,
    max_bin_by_track: np.ndarray,
    sweep_idx: int,
    frame_idx: int,
    n_frames: int,
    cfg: Config,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    arr = transform_values(waveform_bins_tracks, cfg)
    if not cfg.bin0_at_top:
        arr = arr[::-1, :]

    norm = (arr - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    img8 = np.where(np.isfinite(norm), np.rint(norm * 255.0), 0).astype(np.uint8)

    cmap_code = get_cv2_colormap(cfg.colormap)
    if cmap_code is None:
        bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.applyColorMap(img8, cmap_code)

    # Resize for display. Raw matrix size remains n_bins x n_tracks conceptually.
    scale_x = float(cfg.display_scale_x)
    scale_y = float(cfg.display_scale_y)
    if not np.isfinite(scale_x) or scale_x <= 0:
        scale_x = 1.0
    if not np.isfinite(scale_y) or scale_y <= 0:
        scale_y = 1.0
    out_w = max(2, int(round(bgr.shape[1] * scale_x)))
    out_h = max(2, int(round(bgr.shape[0] * scale_y)))
    # Many video encoders prefer even dimensions.
    out_w -= out_w % 2
    out_h -= out_h % 2
    out_w = max(out_w, 2)
    out_h = max(out_h, 2)
    if out_w != bgr.shape[1] or out_h != bgr.shape[0]:
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=interpolation_flag(cfg.resize_interpolation))

    if cfg.overlay_max_bin:
        n_bins = waveform_bins_tracks.shape[0]
        n_tracks = waveform_bins_tracks.shape[1]
        sx = out_w / float(n_tracks)
        sy = out_h / float(n_bins)
        points = []
        for track in range(n_tracks):
            b = int(max_bin_by_track[track])
            if b < 0:
                continue
            y_bin = b if cfg.bin0_at_top else (n_bins - 1 - b)
            x_px = int(round((track + 0.5) * sx))
            y_px = int(round((y_bin + 0.5) * sy))
            if 0 <= x_px < out_w and 0 <= y_px < out_h:
                points.append((x_px, y_px))
                radius = max(1, int(cfg.overlay_radius_px))
                cv2.circle(bgr, (x_px, y_px), radius, cfg.overlay_color_bgr, thickness=-1, lineType=cv2.LINE_AA)
        if cfg.overlay_polyline and len(points) >= 2:
            pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(bgr, [pts], isClosed=False, color=cfg.overlay_color_bgr, thickness=1, lineType=cv2.LINE_AA)

    if cfg.add_text_overlay:
        text = f"sweep={sweep_idx} | frame={frame_idx+1}/{n_frames} | rows=bins {waveform_bins_tracks.shape[0]} | cols=tracks {waveform_bins_tracks.shape[1]}"
        org = (8, 18)
        cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, cfg.text_scale, cfg.text_shadow_bgr, cfg.text_thickness + 2, cv2.LINE_AA)
        cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, cfg.text_scale, cfg.text_color_bgr, cfg.text_thickness, cv2.LINE_AA)

    return bgr


# -----------------------------------------------------------------------------
# Video helpers
# -----------------------------------------------------------------------------


def build_video_writer(path: Path, fps: float, frame_size: Tuple[int, int], fourcc_text: str):
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
    h, w = first_frame_bgr.shape[:2]
    frame_size = (int(w), int(h))
    fmt = cfg.output_format.lower().lstrip(".")

    attempts = []
    if fmt == "mp4":
        fourccs = []
        for code in [cfg.preferred_fourcc, "mp4v", "MP4V", "avc1", "H264"]:
            if code and code not in fourccs:
                fourccs.append(code)
        attempts.extend([(cfg.out_dir / f"{cfg.output_stem}.mp4", code) for code in fourccs])
        if cfg.allow_avi_fallback:
            attempts.append((cfg.out_dir / f"{cfg.output_stem}.avi", "MJPG"))
    elif fmt == "avi":
        fourccs = []
        for code in [cfg.preferred_fourcc, "MJPG", "XVID"]:
            if code and code not in fourccs:
                fourccs.append(code)
        attempts.extend([(cfg.out_dir / f"{cfg.output_stem}.avi", code) for code in fourccs])
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
        "Could not open an OpenCV VideoWriter. No GIF fallback is used.\n"
        "Tried:\n  - " + "\n  - ".join(errors) + "\n"
        "Try output_format='avi', preferred_fourcc='MJPG', or allow_avi_fallback=True."
    )


def validate_video_file(path: Path) -> Dict[str, Any]:
    if cv2 is None:
        return {"validated": False, "reason": "cv2 unavailable"}
    if not path.exists():
        raise RuntimeError(f"Video file was not created: {path}")
    size = path.stat().st_size
    if size < 1024:
        raise RuntimeError(f"Video file is suspiciously small ({size} bytes): {path}")
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if ok else None
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) if ok else None
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if ok else None
    fps = cap.get(cv2.CAP_PROP_FPS) if ok else None
    cap.release()
    if not ok:
        raise RuntimeError(f"Written video cannot be opened by OpenCV: {path}")
    return {
        "validated": True,
        "file_size_bytes": int(size),
        "opencv_frame_count": float(frame_count),
        "opencv_width": float(width),
        "opencv_height": float(height),
        "opencv_fps": float(fps),
    }


def write_metadata(path: Path, cfg: Config, info: PulseIndexInfo, n_rx_bins: int, outputs: Dict[str, Any], extra: Dict[str, Any]) -> None:
    meta = {
        "script": "animate_rx_waveforms.py",
        "script_semantics": "CASALS_L1B_rx_waveform_video_one_frame_per_sweep",
        "scientific_notes": [
            "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
            "refh is WGS84 ellipsoidal height unless otherwise documented.",
            "This is not an official multi-return point cloud.",
            "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            "The optional waveform-max overlay is diagnostic only and does not replace the official refh geolocation product.",
        ],
        "config": asdict(cfg),
        "sweep_track_index": {
            "n_records": info.n_records,
            "n_sweeps": info.n_sweeps,
            "n_tracks": info.n_tracks,
            "n_rx_bins": n_rx_bins,
            "expected_records_if_complete": info.n_sweeps * info.n_tracks,
            "complete_rectangular_grid": info.complete_rectangular_grid,
            "duplicate_sweep_track_cells": info.duplicate_sweep_track_cells,
            "missing_sweep_track_cells": info.missing_sweep_track_cells,
        },
        "outputs": outputs,
        "extra": extra,
        "source_attrs_subset": info.attrs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(meta), f, indent=2)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    # -------------------------------------------------------------------------
    # USER SETTINGS: edit here.
    # -------------------------------------------------------------------------
    cfg = Config(
        h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
        out_dir=Path(r"./outputs/animate_rx_waveforms"),

        output_stem="casals_20241112_rx_waveform_by_sweep",
        output_format="mp4",
        preferred_fourcc="mp4v",
        allow_avi_fallback=False,
        fps=30.0,

        # One frame per sweep by default. Current file: 14080 frames.
        start_sweep=0,
        end_sweep=None,
        sweep_step=1,
        max_frames=None,

        # Raw waveform display. For a cleaner background try "minus_bg_mean".
        background_mode="raw",
        value_transform="log1p",
        range_percentiles=(1.0, 99.7),

        # Native waveform matrix is about 2728 x 256.
        # Set both to 1.0 for exact 256 x 2728 video frames.
        display_scale_x=4.0,
        display_scale_y=0.5,
        resize_interpolation="area",

        bin0_at_top=True,
        colormap="turbo",

        overlay_max_bin=True,
        overlay_polyline=True,
        add_text_overlay=True,

        write_first_frame_png=True,
        validate_written_video=True,
    )
    # -------------------------------------------------------------------------

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("CASALS L1B RX waveform sweep video")
    print("=" * 96)
    print(f"H5: {cfg.h5_path.resolve()}")
    print(f"Output dir: {cfg.out_dir.resolve()}")
    print("Interpretation: one video frame = one sweep; columns = tracks; rows = RX waveform bins.")

    if cv2 is None:
        raise RuntimeError(
            "OpenCV/cv2 is not available. Install it with: conda install -c conda-forge opencv. "
            f"Original import error: {_CV2_IMPORT_ERROR}"
        )

    with h5py.File(cfg.h5_path, "r") as h5:
        rx_ds = require_dataset(h5, "rx_waveform")
        info, record_axis, n_rx_bins = build_record_index_grid(h5, rx_ds)
        frame_sweeps = selected_sweeps(info.n_sweeps, cfg)
        background_values = read_optional_background_values(h5, info, cfg)

        print(f"rx_waveform shape: {rx_ds.shape}")
        print(f"record axis: {record_axis}; n_rx_bins: {n_rx_bins}")
        print(f"Sweeps x tracks: {info.n_sweeps} x {info.n_tracks} = {info.n_sweeps * info.n_tracks:,}")
        print(f"Complete rectangular sweep-track grid: {info.complete_rectangular_grid}")
        print(f"Frames to render: {frame_sweeps.size:,}")

        vmin, vmax = estimate_display_range(h5, rx_ds, info, record_axis, n_rx_bins, frame_sweeps, background_values, cfg)
        print(f"Display range after {cfg.background_mode} + {cfg.value_transform}: {vmin:.6g} to {vmax:.6g}")

        # First frame determines video size and writer initialization.
        first_sweep = int(frame_sweeps[0])
        first_wf, _, first_max_bins = read_sweep_waveform(rx_ds, info, first_sweep, record_axis, n_rx_bins, background_values)
        first_frame = waveform_to_bgr_frame(first_wf, first_max_bins, first_sweep, 0, int(frame_sweeps.size), cfg, vmin, vmax)
        frame_h, frame_w = first_frame.shape[:2]
        print(f"Video frame size: {frame_w} x {frame_h} pixels")
        print(f"Underlying waveform image size: {info.n_tracks} columns x {n_rx_bins} rows")

        if cfg.write_first_frame_png:
            first_png = cfg.out_dir / f"{cfg.output_stem}_first_frame.png"
            cv2.imwrite(str(first_png), first_frame)
        else:
            first_png = None

        writer, out_path, actual_format, actual_fourcc = open_video_writer_with_fallbacks(cfg, first_frame)

        iterator: Iterable[Tuple[int, int]] = enumerate(frame_sweeps)
        if tqdm is not None:
            iterator = tqdm(list(enumerate(frame_sweeps)), desc="Writing sweep video")

        for frame_idx, sweep in iterator:
            sweep_i = int(sweep)
            if frame_idx == 0:
                frame = first_frame
            else:
                wf, _, max_bins = read_sweep_waveform(rx_ds, info, sweep_i, record_axis, n_rx_bins, background_values)
                frame = waveform_to_bgr_frame(wf, max_bins, sweep_i, int(frame_idx), int(frame_sweeps.size), cfg, vmin, vmax)
            if frame.shape[:2] != (frame_h, frame_w):
                writer.release()
                raise RuntimeError(
                    f"Frame size changed at frame {frame_idx}: {frame.shape[:2]} vs {(frame_h, frame_w)}"
                )
            writer.write(frame)
            if tqdm is None and (
                frame_idx == 0
                or (frame_idx + 1) % int(cfg.progress_every_n_frames) == 0
                or frame_idx + 1 == frame_sweeps.size
            ):
                print(f"  wrote frame {frame_idx + 1:,}/{frame_sweeps.size:,} (sweep {sweep_i:,})")

        writer.release()

    validation = validate_video_file(out_path) if cfg.validate_written_video else {"validated": False, "reason": "disabled"}

    metadata_path = cfg.out_dir / f"{cfg.output_stem}_metadata.json"
    outputs = {
        "video": str(out_path),
        "format": actual_format,
        "fourcc": actual_fourcc,
        "first_frame_png": str(first_png) if first_png is not None else None,
        "metadata_json": str(metadata_path),
    }
    extra = {
        "frame_count_requested": int(frame_sweeps.size),
        "first_sweep": int(frame_sweeps[0]),
        "last_sweep": int(frame_sweeps[-1]),
        "frame_width_px": int(frame_w),
        "frame_height_px": int(frame_h),
        "underlying_waveform_columns_tracks": int(info.n_tracks),
        "underlying_waveform_rows_rx_bins": int(n_rx_bins),
        "record_axis_in_rx_waveform": int(record_axis),
        "display_vmin": float(vmin),
        "display_vmax": float(vmax),
        "video_validation": validation,
    }
    if cfg.write_metadata_json:
        write_metadata(metadata_path, cfg, info, n_rx_bins, outputs, extra)

    print("\nDone.")
    print(json.dumps(_json_safe({
        "video": str(out_path),
        "metadata": str(metadata_path) if cfg.write_metadata_json else None,
        "first_frame_png": str(first_png) if first_png is not None else None,
        "n_sweeps": int(info.n_sweeps),
        "n_tracks": int(info.n_tracks),
        "n_rx_bins": int(n_rx_bins),
        "n_frames": int(frame_sweeps.size),
        "frame_size_px": [int(frame_w), int(frame_h)],
        "underlying_waveform_image": f"{info.n_tracks} columns x {n_rx_bins} rows",
        "video_validation": validation,
    }), indent=2))


if __name__ == "__main__":
    main()
