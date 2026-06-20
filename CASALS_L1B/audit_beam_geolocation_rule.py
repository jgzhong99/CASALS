from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Callable, Iterable

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

try:
    from scipy.signal import find_peaks

    HAS_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    HAS_SCIPY = False


C_LIGHT = 299792458.0
GEODETIC_CRS = "EPSG:4979"
ECEF_CRS = "EPSG:4978"
DEFAULT_CHUNK_SIZE = 2048
DEFAULT_SAMPLE_PULSES = 64
DEFAULT_TOP_PEAKS_PER_WAVEFORM = 6
RANDOM_SEED = 42
STRONG_PEAK_FRACTION = 0.5
PEAK_MIN_DISTANCE_BINS = 5
PEAK_MIN_PROMINENCE_SIGMA = 5.0

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_GLOBS = [
    str(SCRIPT_DIR / "casals_h5_downloads" / "*.h5"),
    str(SCRIPT_DIR / "casals_h5_downloads" / "*.hdf5"),
]
DEFAULT_OUT_DIR = SCRIPT_DIR / "beam_geolocation_rule_audit_outputs"

to_ecef = Transformer.from_crs(GEODETIC_CRS, ECEF_CRS, always_xy=True)
from_ecef = Transformer.from_crs(ECEF_CRS, GEODETIC_CRS, always_xy=True)

REQUIRED_FIELDS = [
    "bin_size",
    "local_beam_azimuth",
    "local_beam_elevation",
    "refh",
    "refh_bounce_time_offset",
    "refh_latitude",
    "refh_longitude",
    "rwstart",
    "rwstart_bounce_time_offset",
    "rwstart_latitude",
    "rwstart_longitude",
    "rwstop",
    "rwstop_bounce_time_offset",
    "rwstop_latitude",
    "rwstop_longitude",
    "rx_bins",
    "rx_waveform",
]

OPTIONAL_FIELDS = [
    "refh_snr",
    "sweep_num",
    "track_num",
]


REFERENCE_BIN_MODEL_SPECS: list[tuple[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]]] = [
    ("argmax_rx_waveform", lambda ref, start, stop, bs, n_bins: np.full_like(ref, np.nan, dtype=float)),
    ("(refh_offset - rwstart_offset) / bin_size", lambda ref, start, stop, bs, n_bins: (ref - start) / bs),
    ("(rwstop_offset - refh_offset) / bin_size", lambda ref, start, stop, bs, n_bins: (stop - ref) / bs),
    ("(refh_offset - rwstart_offset) * C / 2 / bin_size", lambda ref, start, stop, bs, n_bins: (ref - start) * C_LIGHT / 2.0 / bs),
    ("(rwstop_offset - refh_offset) * C / 2 / bin_size", lambda ref, start, stop, bs, n_bins: (stop - ref) * C_LIGHT / 2.0 / bs),
    ("(refh_offset - rwstart_offset) * C / bin_size", lambda ref, start, stop, bs, n_bins: (ref - start) * C_LIGHT / bs),
    ("(rwstop_offset - refh_offset) * C / bin_size", lambda ref, start, stop, bs, n_bins: (stop - ref) * C_LIGHT / bs),
    (
        "((refh_offset - rwstart_offset) / (rwstop_offset - rwstart_offset)) * (n_rx_bins - 1)",
        lambda ref, start, stop, bs, n_bins: np.divide(
            ref - start,
            stop - start,
            out=np.full_like(ref, np.nan, dtype=float),
            where=np.abs(stop - start) > 0,
        )
        * (n_bins - 1),
    ),
    (
        "((rwstop_offset - refh_offset) / (rwstop_offset - rwstart_offset)) * (n_rx_bins - 1)",
        lambda ref, start, stop, bs, n_bins: np.divide(
            stop - ref,
            stop - start,
            out=np.full_like(ref, np.nan, dtype=float),
            where=np.abs(stop - start) > 0,
        )
        * (n_bins - 1),
    ),
]

WHOLE_WINDOW_MODEL_SPECS: list[tuple[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]] = [
    ("(rwstop_offset - rwstart_offset) * C / bin_size", lambda start, stop, bs: (stop - start) * C_LIGHT / bs),
    ("(rwstop_offset - rwstart_offset) * C / 2 / bin_size", lambda start, stop, bs: (stop - start) * C_LIGHT / 2.0 / bs),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit beam-based geolocation rules for CASALS L1B H5 files."
    )
    parser.add_argument("--h5", nargs="*", default=[], help="One or more explicit H5 paths.")
    parser.add_argument(
        "--glob",
        nargs="*",
        default=DEFAULT_GLOBS,
        help="Glob patterns used when --h5 is not provided.",
    )
    parser.add_argument(
        "--max-pulses",
        type=int,
        default=None,
        help="Optional cap on pulses per file for faster audits.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size used when streaming full-file metrics.",
    )
    parser.add_argument(
        "--sample-pulses",
        type=int,
        default=DEFAULT_SAMPLE_PULSES,
        help="Sampled pulses per file for peak-level beam scoring.",
    )
    parser.add_argument(
        "--top-peaks-per-waveform",
        type=int,
        default=DEFAULT_TOP_PEAKS_PER_WAVEFORM,
        help="Number of strongest detected peaks per sampled waveform.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for CSV and summary outputs.",
    )
    return parser.parse_args()


def resolve_h5_files(h5_files: list[str], h5_globs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw in h5_files:
        p = Path(raw).expanduser()
        if p.exists():
            rp = p.resolve()
            if rp not in seen:
                resolved.append(rp)
                seen.add(rp)

    if not resolved:
        for pattern in h5_globs:
            for match in glob.glob(pattern):
                p = Path(match)
                if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}:
                    rp = p.resolve()
                    if rp not in seen:
                        resolved.append(rp)
                        seen.add(rp)

    if not resolved:
        raise FileNotFoundError("No H5 files found.")
    return sorted(resolved)


def sample_indices(n_records: int, n_sample: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = min(int(n_records), int(n_sample))
    return np.sort(rng.choice(int(n_records), size=n, replace=False))


def infer_angle_unit_from_values(x: np.ndarray) -> str:
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "unknown"
    p99_abs = float(np.nanpercentile(np.abs(finite), 99))
    if p99_abs <= 2 * np.pi + 0.1:
        return "radian_likely"
    if p99_abs <= 360.0 + 1.0:
        return "degree_likely"
    return "unknown_or_non_angle"


def geodetic_to_ecef(lon: np.ndarray, lat: np.ndarray, h: np.ndarray) -> np.ndarray:
    x, y, z = to_ecef.transform(lon, lat, h)
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(xyz: np.ndarray) -> np.ndarray:
    lon, lat, h = from_ecef.transform(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    return np.stack([lon, lat, h], axis=-1)


def enu_to_ecef_delta_matrix(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    east = np.stack([-sin_lon, cos_lon, np.zeros_like(lon)], axis=-1)
    north = np.stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], axis=-1)
    up = np.stack([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], axis=-1)
    return np.stack([east, north, up], axis=-1)


def beam_unit_enu(
    az: np.ndarray,
    el: np.ndarray,
    angle_unit: str,
    vector_direction: str,
) -> np.ndarray:
    az = np.asarray(az, dtype=float)
    el = np.asarray(el, dtype=float)
    if angle_unit == "degree":
        az = np.deg2rad(az)
        el = np.deg2rad(el)
    east = np.cos(el) * np.sin(az)
    north = np.cos(el) * np.cos(az)
    up = np.sin(el)
    vec = np.stack([east, north, up], axis=-1)
    norm = np.linalg.norm(vec, axis=-1)
    vec = vec / norm[:, None]
    if vector_direction == "sensor_to_ref":
        vec = -vec
    return vec


def interpolate_segment_geodetic(
    rwstart_lon: np.ndarray,
    rwstart_lat: np.ndarray,
    rwstart_h: np.ndarray,
    rwstop_lon: np.ndarray,
    rwstop_lat: np.ndarray,
    rwstop_h: np.ndarray,
    t_peak: np.ndarray,
) -> dict[str, np.ndarray]:
    p0 = geodetic_to_ecef(rwstart_lon, rwstart_lat, rwstart_h)
    p1 = geodetic_to_ecef(rwstop_lon, rwstop_lat, rwstop_h)
    t_peak = np.asarray(t_peak, dtype=float)
    xyz = p0 + t_peak[:, None] * (p1 - p0)
    llh = ecef_to_geodetic(xyz)
    return {
        "segment_lon": llh[:, 0],
        "segment_lat": llh[:, 1],
        "segment_h": llh[:, 2],
        "segment_ecef": xyz,
    }


def propagate_from_refh_with_beam(
    ref_lon: np.ndarray,
    ref_lat: np.ndarray,
    ref_h: np.ndarray,
    az: np.ndarray,
    el: np.ndarray,
    ref_bin: np.ndarray,
    peak_bin: np.ndarray,
    bin_size: np.ndarray,
    angle_unit: str,
    range_spacing_interpretation: str,
    vector_direction: str,
) -> dict[str, np.ndarray]:
    ref_lon = np.asarray(ref_lon, dtype=float)
    ref_lat = np.asarray(ref_lat, dtype=float)
    ref_h = np.asarray(ref_h, dtype=float)
    ref_bin = np.asarray(ref_bin, dtype=float)
    peak_bin = np.asarray(peak_bin, dtype=float)
    bin_size = np.asarray(bin_size, dtype=float)

    if range_spacing_interpretation == "bin_size_as_seconds_times_c":
        spacing_m = bin_size * C_LIGHT
    elif range_spacing_interpretation == "bin_size_as_seconds_two_way_c_over_2":
        spacing_m = bin_size * C_LIGHT / 2.0
    elif range_spacing_interpretation == "bin_size_as_m_per_bin":
        spacing_m = bin_size
    else:
        raise ValueError(range_spacing_interpretation)

    delta_bins = peak_bin - ref_bin
    delta_range_m = delta_bins * spacing_m
    beam_enu = beam_unit_enu(az, el, angle_unit=angle_unit, vector_direction=vector_direction)
    delta_enu = delta_range_m[:, None] * beam_enu
    rot = enu_to_ecef_delta_matrix(ref_lon, ref_lat)
    delta_ecef = np.einsum("nij,nj->ni", rot, delta_enu)
    ref_ecef = geodetic_to_ecef(ref_lon, ref_lat, ref_h)
    peak_ecef = ref_ecef + delta_ecef
    peak_llh = ecef_to_geodetic(peak_ecef)
    return {
        "beam_lon": peak_llh[:, 0],
        "beam_lat": peak_llh[:, 1],
        "beam_h": peak_llh[:, 2],
        "beam_ecef": peak_ecef,
        "ref_ecef": ref_ecef,
        "delta_range_m": delta_range_m,
        "spacing_m_per_bin": spacing_m,
        "beam_enu": beam_enu,
        "delta_ecef": delta_ecef,
    }


def make_strong_peak_mask(rx_chunk: np.ndarray, strong_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rx_chunk = np.asarray(rx_chunk)
    n_rows, n_bins = rx_chunk.shape
    argmax_bins = np.argmax(rx_chunk, axis=1)
    argmax_vals = rx_chunk[np.arange(n_rows), argmax_bins]
    threshold = strong_fraction * argmax_vals
    mask = np.zeros((n_rows, n_bins), dtype=bool)
    if n_bins == 1:
        mask[:, 0] = True
        return argmax_bins, mask
    mask[:, 0] = (rx_chunk[:, 0] >= rx_chunk[:, 1]) & (rx_chunk[:, 0] >= threshold)
    mask[:, -1] = (rx_chunk[:, -1] >= rx_chunk[:, -2]) & (rx_chunk[:, -1] >= threshold)
    mid = rx_chunk[:, 1:-1]
    mask[:, 1:-1] = (mid >= rx_chunk[:, :-2]) & (mid >= rx_chunk[:, 2:]) & (mid >= threshold[:, None])
    empty = ~mask.any(axis=1)
    if np.any(empty):
        mask[empty, argmax_bins[empty]] = True
    return argmax_bins, mask


def strong_peak_lookup_tables(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_bins = mask.shape
    cols = np.broadcast_to(np.arange(n_bins, dtype=np.int32), (n_rows, n_bins))
    left = np.where(mask, cols, -n_bins).astype(np.int32, copy=False)
    np.maximum.accumulate(left, axis=1, out=left)
    right = np.where(mask, cols, 2 * n_bins).astype(np.int32, copy=False)
    right = right[:, ::-1]
    np.minimum.accumulate(right, axis=1, out=right)
    right = right[:, ::-1]
    return left, right


def init_hist_state(n_bins: int) -> dict[str, np.ndarray | int]:
    return {
        "n_total": 0,
        "n_valid": 0,
        "hist_argmax": np.zeros(n_bins, dtype=np.int64),
        "hist_strong": np.zeros(n_bins, dtype=np.int64),
    }


def hist_quantile(hist: np.ndarray, q: float) -> float:
    hist = np.asarray(hist, dtype=np.int64)
    total = int(hist.sum())
    if total == 0:
        return np.nan
    target = q * (total - 1)
    csum = np.cumsum(hist)
    return float(np.searchsorted(csum, target, side="left"))


def build_reference_model_predictions(
    ref: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    bs: np.ndarray,
    n_bins: int,
    argmax_bins: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    pred: dict[str, np.ndarray] = {}
    for name, fn in REFERENCE_BIN_MODEL_SPECS:
        if name == "argmax_rx_waveform":
            if argmax_bins is None:
                pred[name] = np.full_like(ref, np.nan, dtype=float)
            else:
                pred[name] = np.asarray(argmax_bins, dtype=float)
            continue
        try:
            pred[name] = np.asarray(fn(ref, start, stop, bs, n_bins), dtype=float)
        except Exception:
            pred[name] = np.full_like(ref, np.nan, dtype=float)
    return pred


def classify_reference_closure(valid_fraction: float, median_strong: float, p90_strong: float) -> str:
    if valid_fraction >= 0.999 and median_strong <= 1 and p90_strong <= 3:
        return "strong_internal_closure"
    if valid_fraction >= 0.99 and median_strong <= 5 and p90_strong <= 25:
        return "moderate_internal_closure"
    return "weak_or_failed_internal_closure"


def classify_endpoint_quality(segment_median_m: float, tdiff_median: float, ref_model_median: float, ref_model_p90: float) -> str:
    if segment_median_m <= 0.01 and tdiff_median <= 1e-4 and ref_model_median <= 1 and ref_model_p90 <= 3:
        return "strong_endpoint_segment_closure"
    if segment_median_m <= 0.1 and tdiff_median <= 1e-3 and ref_model_median <= 5 and ref_model_p90 <= 25:
        return "moderate_endpoint_segment_closure"
    return "weak_or_failed_endpoint_segment_closure"


def classify_beam_agreement(median_m: float, p90_m: float) -> str:
    if median_m <= 1.0 and p90_m <= 5.0:
        return "strong_beam_agreement"
    if median_m <= 5.0 and p90_m <= 20.0:
        return "moderate_beam_agreement"
    return "weak_beam_agreement"


def classify_refh_return_error(median_m: float, p90_m: float) -> str:
    if median_m <= 1.0 and p90_m <= 5.0:
        return "strong_refh_return_closure"
    if median_m <= 5.0 and p90_m <= 20.0:
        return "moderate_refh_return_closure"
    return "weak_refh_return_closure"


def classify_direction_consistency(median_cosine: float, p10_cosine: float) -> str:
    if median_cosine >= 0.95 and p10_cosine >= 0.50:
        return "strong_direction_consistency"
    if median_cosine >= 0.80 and p10_cosine >= 0.00:
        return "moderate_direction_consistency"
    return "weak_direction_consistency"


def safe_find_peaks_1d(y: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if HAS_SCIPY:
        edge = max(20, int(0.1 * len(y)))
        bg = np.concatenate([y[:edge], y[-edge:]])
        med = float(np.nanmedian(bg))
        mad = float(np.nanmedian(np.abs(bg - med)))
        sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(bg))
        prominence = PEAK_MIN_PROMINENCE_SIGMA * sigma if np.isfinite(sigma) and sigma > 0 else None
        peaks, _ = find_peaks(y, prominence=prominence, distance=PEAK_MIN_DISTANCE_BINS)
        if peaks.size == 0:
            peaks, _ = find_peaks(y, distance=PEAK_MIN_DISTANCE_BINS)
    else:
        peaks = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    if peaks.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    strengths = y[peaks]
    order = np.argsort(strengths)[::-1][:top_k]
    return peaks[order], strengths[order]


def rms_distance_m(xyz_a: np.ndarray, xyz_b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(xyz_a, dtype=float) - np.asarray(xyz_b, dtype=float), axis=1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    out = np.full(len(a), np.nan, dtype=float)
    valid = (an > 0) & (bn > 0)
    out[valid] = np.sum(a[valid] * b[valid], axis=1) / (an[valid] * bn[valid])
    both_zero = (an == 0) & (bn == 0)
    out[both_zero] = 1.0
    return out


def serialize_attr_value(v: object) -> object:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def dataset_path_map(h5: h5py.File) -> dict[str, str]:
    name_to_path: dict[str, list[str]] = {}

    def visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            base = name.split("/")[-1]
            name_to_path.setdefault(base, []).append("/" + name)

    h5.visititems(visitor)

    resolved: dict[str, str] = {}
    for base, paths in name_to_path.items():
        if len(paths) == 1:
            resolved[base] = paths[0]
            continue
        root_match = [p for p in paths if p == f"/{base}"]
        if len(root_match) == 1:
            resolved[base] = root_match[0]
        else:
            raise ValueError(f"Ambiguous dataset basename {base}: {paths}")
    return resolved


def require_paths(h5: h5py.File) -> dict[str, str]:
    paths = dataset_path_map(h5)
    missing = [field for field in REQUIRED_FIELDS if field not in paths]
    if missing:
        raise KeyError(f"Missing required datasets: {missing}")
    return paths


def whole_window_stats(start: np.ndarray, stop: np.ndarray, bs: np.ndarray, n_bins: int) -> list[dict[str, object]]:
    rows = []
    target_bins = n_bins - 1
    for model_name, fn in WHOLE_WINDOW_MODEL_SPECS:
        vals = np.asarray(fn(start, stop, bs), dtype=float)
        abs_diff = np.abs(vals - target_bins)
        rows.append(
            {
                "model": model_name,
                "n_pulses_scored": int(len(vals)),
                "target_n_rx_bins_minus_1": int(target_bins),
                "median_window_bins": float(np.nanmedian(vals)),
                "median_abs_diff_to_n_rx_bins_minus_1": float(np.nanmedian(abs_diff)),
                "p90_abs_diff_to_n_rx_bins_minus_1": float(np.nanpercentile(abs_diff, 90)),
            }
        )
    return rows


def choose_spacing(rows: pd.DataFrame) -> pd.Series:
    best = rows.sort_values(
        ["median_abs_diff_to_n_rx_bins_minus_1", "p90_abs_diff_to_n_rx_bins_minus_1"],
        ascending=[True, True],
    ).iloc[0]
    interpretation = (
        "bin_size_as_seconds_times_c"
        if best["model"] == "(rwstop_offset - rwstart_offset) * C / bin_size"
        else "bin_size_as_seconds_two_way_c_over_2"
    )
    out = best.copy()
    out["range_spacing_interpretation"] = interpretation
    return out


def choose_angle_unit(az_sample: np.ndarray, el_sample: np.ndarray) -> tuple[str, str, str]:
    az_guess = infer_angle_unit_from_values(az_sample)
    el_guess = infer_angle_unit_from_values(el_sample)
    if az_guess == "degree_likely" and el_guess == "degree_likely":
        return "degree", az_guess, el_guess
    return "radian", az_guess, el_guess


def load_archive_time_evidence() -> list[str]:
    notes: list[str] = []
    guide_path = REPO_ROOT / "Archive" / "sample data and viewer 2024 collection" / "CASALS_viewer_v2_user_guide_23apr2025.pdf"
    if not guide_path.exists() or PdfReader is None:
        return notes
    try:
        reader = PdfReader(str(guide_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:8])
    except Exception:
        return notes
    text_lower = text.lower()
    if "1ghz" in text_lower or "1 ghz" in text_lower:
        notes.append("Archive viewer guide mentions a 1 GHz waveform digitizer.")
    if "2740ns" in text_lower or "2,740ns" in text_lower or "2740 ns" in text_lower:
        notes.append("Archive viewer guide mentions an approximately 2740 ns return range window.")
    return notes


def audit_file(
    h5_path: Path,
    args: argparse.Namespace,
    archive_notes: list[str],
) -> dict[str, pd.DataFrame | dict[str, object]]:
    with h5py.File(h5_path, "r") as h5:
        paths = require_paths(h5)
        attrs = {str(k): serialize_attr_value(v) for k, v in h5.attrs.items()}

        rx_ds = h5[paths["rx_waveform"]]
        n_records, n_bins = rx_ds.shape
        n_take = min(int(n_records), int(args.max_pulses) if args.max_pulses is not None else int(n_records))
        idx = sample_indices(n_records, args.sample_pulses, RANDOM_SEED)

        az_sample = np.asarray(h5[paths["local_beam_azimuth"]][idx], dtype=float)
        el_sample = np.asarray(h5[paths["local_beam_elevation"]][idx], dtype=float)
        angle_unit_selected, az_guess, el_guess = choose_angle_unit(az_sample, el_sample)

        whole_window_rows: list[dict[str, object]] = []
        states = {name: init_hist_state(n_bins) for name, _ in REFERENCE_BIN_MODEL_SPECS}
        dist_pieces: list[np.ndarray] = []
        tdiff_pieces: list[np.ndarray] = []
        off_segment_count = 0
        total_segment_count = 0

        for start_i in range(0, n_take, int(args.chunk_size)):
            stop_i = min(n_take, start_i + int(args.chunk_size))
            ref = np.asarray(h5[paths["refh_bounce_time_offset"]][start_i:stop_i], dtype=float)
            rwstart = np.asarray(h5[paths["rwstart_bounce_time_offset"]][start_i:stop_i], dtype=float)
            rwstop = np.asarray(h5[paths["rwstop_bounce_time_offset"]][start_i:stop_i], dtype=float)
            bs = np.asarray(h5[paths["bin_size"]][start_i:stop_i], dtype=float)
            rx = np.asarray(rx_ds[start_i:stop_i, :], dtype=np.float32)

            for row in whole_window_stats(rwstart, rwstop, bs, n_bins):
                row["h5_file"] = str(h5_path)
                whole_window_rows.append(row)

            argmax_bins, strong_mask = make_strong_peak_mask(rx, strong_fraction=STRONG_PEAK_FRACTION)
            left_lookup, right_lookup = strong_peak_lookup_tables(strong_mask)
            pred_dict = build_reference_model_predictions(ref, rwstart, rwstop, bs, n_bins, argmax_bins=argmax_bins)
            chunk_size = len(ref)

            for model_name, pred in pred_dict.items():
                state = states[model_name]
                state["n_total"] += chunk_size
                pred_round = np.rint(np.asarray(pred, dtype=float))
                valid = np.isfinite(pred_round) & (pred_round >= 0) & (pred_round < n_bins)
                valid_count = int(valid.sum())
                state["n_valid"] += valid_count
                if valid_count == 0:
                    continue
                row_idx = np.flatnonzero(valid)
                pred_bins = pred_round[valid].astype(np.int32, copy=False)
                d_argmax = np.abs(pred_bins - argmax_bins[valid].astype(np.int32))
                left_dist = pred_bins - left_lookup[row_idx, pred_bins]
                right_dist = right_lookup[row_idx, pred_bins] - pred_bins
                d_strong = np.minimum(left_dist, right_dist).astype(np.int32, copy=False)
                state["hist_argmax"] += np.bincount(d_argmax, minlength=n_bins)[:n_bins]
                state["hist_strong"] += np.bincount(d_strong, minlength=n_bins)[:n_bins]

            rwstart_lon = np.asarray(h5[paths["rwstart_longitude"]][start_i:stop_i], dtype=float)
            rwstart_lat = np.asarray(h5[paths["rwstart_latitude"]][start_i:stop_i], dtype=float)
            rwstart_h = np.asarray(h5[paths["rwstart"]][start_i:stop_i], dtype=float)
            rwstop_lon = np.asarray(h5[paths["rwstop_longitude"]][start_i:stop_i], dtype=float)
            rwstop_lat = np.asarray(h5[paths["rwstop_latitude"]][start_i:stop_i], dtype=float)
            rwstop_h = np.asarray(h5[paths["rwstop"]][start_i:stop_i], dtype=float)
            refh_lon = np.asarray(h5[paths["refh_longitude"]][start_i:stop_i], dtype=float)
            refh_lat = np.asarray(h5[paths["refh_latitude"]][start_i:stop_i], dtype=float)
            refh_h = np.asarray(h5[paths["refh"]][start_i:stop_i], dtype=float)

            p0 = geodetic_to_ecef(rwstart_lon, rwstart_lat, rwstart_h)
            p1 = geodetic_to_ecef(rwstop_lon, rwstop_lat, rwstop_h)
            pr = geodetic_to_ecef(refh_lon, refh_lat, refh_h)
            v = p1 - p0
            vv = np.sum(v * v, axis=1)
            t_geom = np.divide(np.sum((pr - p0) * v, axis=1), vv, out=np.full(len(vv), np.nan), where=vv > 0)
            t_time = np.divide(
                ref - rwstart,
                rwstop - rwstart,
                out=np.full(len(vv), np.nan),
                where=np.abs(rwstop - rwstart) > 0,
            )
            proj = p0 + t_geom[:, None] * v
            dist = np.linalg.norm(pr - proj, axis=1).astype(np.float32)
            abs_t_diff = np.abs(t_geom - t_time).astype(np.float32)
            dist_pieces.append(dist)
            tdiff_pieces.append(abs_t_diff)
            off_segment_count += int(np.sum((t_geom < 0.0) | (t_geom > 1.0)))
            total_segment_count += int(len(t_geom))

        whole_window_df = (
            pd.DataFrame(whole_window_rows)
            .groupby(
                ["h5_file", "model", "target_n_rx_bins_minus_1"],
                as_index=False,
            )
            .agg(
                n_pulses_scored=("n_pulses_scored", "sum"),
                median_window_bins=("median_window_bins", "median"),
                median_abs_diff_to_n_rx_bins_minus_1=("median_abs_diff_to_n_rx_bins_minus_1", "median"),
                p90_abs_diff_to_n_rx_bins_minus_1=("p90_abs_diff_to_n_rx_bins_minus_1", "median"),
            )
        )

        reference_rows = []
        for model_name, state in states.items():
            valid_fraction = state["n_valid"] / state["n_total"] if state["n_total"] else np.nan
            median_argmax = hist_quantile(state["hist_argmax"], 0.50)
            median_strong = hist_quantile(state["hist_strong"], 0.50)
            p90_strong = hist_quantile(state["hist_strong"], 0.90)
            reference_rows.append(
                {
                    "h5_file": str(h5_path),
                    "candidate_model": model_name,
                    "n_pulses_scored": int(state["n_total"]),
                    "valid_fraction": float(valid_fraction),
                    "median_abs_bin_diff_to_argmax": float(median_argmax) if np.isfinite(median_argmax) else np.nan,
                    "median_abs_bin_diff_to_nearest_strong_peak": float(median_strong) if np.isfinite(median_strong) else np.nan,
                    "p90_abs_bin_diff_to_nearest_strong_peak": float(p90_strong) if np.isfinite(p90_strong) else np.nan,
                }
            )
        reference_df = pd.DataFrame(reference_rows)
        eligible_reference_df = reference_df[reference_df["candidate_model"] != "argmax_rx_waveform"].copy()
        if eligible_reference_df.empty:
            eligible_reference_df = reference_df.copy()
        best_reference = eligible_reference_df.sort_values(
            [
                "valid_fraction",
                "median_abs_bin_diff_to_nearest_strong_peak",
                "p90_abs_bin_diff_to_nearest_strong_peak",
                "median_abs_bin_diff_to_argmax",
            ],
            ascending=[False, True, True, True],
        ).iloc[0].copy()
        best_reference["closure_quality"] = classify_reference_closure(
            float(best_reference["valid_fraction"]),
            float(best_reference["median_abs_bin_diff_to_nearest_strong_peak"]),
            float(best_reference["p90_abs_bin_diff_to_nearest_strong_peak"]),
        )
        best_reference_df = pd.DataFrame([best_reference])

        spacing_choice = choose_spacing(whole_window_df[whole_window_df["h5_file"] == str(h5_path)])
        spacing_choice_df = pd.DataFrame([spacing_choice])

        dist_all = np.concatenate(dist_pieces) if dist_pieces else np.array([], dtype=np.float32)
        tdiff_all = np.concatenate(tdiff_pieces) if tdiff_pieces else np.array([], dtype=np.float32)
        frac_model_name = "((refh_offset - rwstart_offset) / (rwstop_offset - rwstart_offset)) * (n_rx_bins - 1)"
        frac_row = reference_df[reference_df["candidate_model"] == frac_model_name].iloc[0]
        endpoint_row = {
            "h5_file": str(h5_path),
            "median_refh_distance_to_rwstart_rwstop_segment_m": float(np.nanmedian(dist_all)) if len(dist_all) else np.nan,
            "p90_refh_distance_to_rwstart_rwstop_segment_m": float(np.nanpercentile(dist_all, 90)) if len(dist_all) else np.nan,
            "median_abs_t_geom_minus_t_time": float(np.nanmedian(tdiff_all)) if len(tdiff_all) else np.nan,
            "p90_abs_t_geom_minus_t_time": float(np.nanpercentile(tdiff_all, 90)) if len(tdiff_all) else np.nan,
            "off_segment_fraction": off_segment_count / total_segment_count if total_segment_count else np.nan,
            "t_time_bin_model": frac_model_name,
            "t_time_bin_model_median_abs_diff_to_nearest_strong_peak": float(frac_row["median_abs_bin_diff_to_nearest_strong_peak"]),
            "t_time_bin_model_p90_abs_diff_to_nearest_strong_peak": float(frac_row["p90_abs_bin_diff_to_nearest_strong_peak"]),
        }
        endpoint_row["endpoint_segment_closure_quality"] = classify_endpoint_quality(
            float(endpoint_row["median_refh_distance_to_rwstart_rwstop_segment_m"]),
            float(endpoint_row["median_abs_t_geom_minus_t_time"]),
            float(endpoint_row["t_time_bin_model_median_abs_diff_to_nearest_strong_peak"]),
            float(endpoint_row["t_time_bin_model_p90_abs_diff_to_nearest_strong_peak"]),
        )
        endpoint_df = pd.DataFrame([endpoint_row])

        sampled_rx = np.asarray(rx_ds[idx, :], dtype=np.float32)
        sampled_ref = np.asarray(h5[paths["refh_bounce_time_offset"]][idx], dtype=float)
        sampled_rwstart = np.asarray(h5[paths["rwstart_bounce_time_offset"]][idx], dtype=float)
        sampled_rwstop = np.asarray(h5[paths["rwstop_bounce_time_offset"]][idx], dtype=float)
        sampled_bs = np.asarray(h5[paths["bin_size"]][idx], dtype=float)
        sampled_argmax_bins = np.argmax(sampled_rx, axis=1)
        sampled_pred_bins = build_reference_model_predictions(
            sampled_ref,
            sampled_rwstart,
            sampled_rwstop,
            sampled_bs,
            n_bins,
            argmax_bins=sampled_argmax_bins,
        )[str(best_reference["candidate_model"])]

        ref_lon = np.asarray(h5[paths["refh_longitude"]][idx], dtype=float)
        ref_lat = np.asarray(h5[paths["refh_latitude"]][idx], dtype=float)
        ref_h = np.asarray(h5[paths["refh"]][idx], dtype=float)
        rwstart_lon = np.asarray(h5[paths["rwstart_longitude"]][idx], dtype=float)
        rwstart_lat = np.asarray(h5[paths["rwstart_latitude"]][idx], dtype=float)
        rwstart_h = np.asarray(h5[paths["rwstart"]][idx], dtype=float)
        rwstop_lon = np.asarray(h5[paths["rwstop_longitude"]][idx], dtype=float)
        rwstop_lat = np.asarray(h5[paths["rwstop_latitude"]][idx], dtype=float)
        rwstop_h = np.asarray(h5[paths["rwstop"]][idx], dtype=float)
        az = np.asarray(h5[paths["local_beam_azimuth"]][idx], dtype=float)
        el = np.asarray(h5[paths["local_beam_elevation"]][idx], dtype=float)

        optional = {}
        for field in OPTIONAL_FIELDS:
            path = paths.get(field)
            optional[field] = np.asarray(h5[path][idx], dtype=float) if path else np.full(len(idx), np.nan)

        derived_base_rows: list[dict[str, object]] = []
        for local_i, pulse_index in enumerate(idx):
            peaks, strengths = safe_find_peaks_1d(sampled_rx[local_i], top_k=int(args.top_peaks_per_waveform))
            if peaks.size == 0:
                continue
            for rank, (peak_bin, peak_amp) in enumerate(zip(peaks, strengths), start=1):
                t_peak = peak_bin / (n_bins - 1)
                segment = interpolate_segment_geodetic(
                    np.array([rwstart_lon[local_i]]),
                    np.array([rwstart_lat[local_i]]),
                    np.array([rwstart_h[local_i]]),
                    np.array([rwstop_lon[local_i]]),
                    np.array([rwstop_lat[local_i]]),
                    np.array([rwstop_h[local_i]]),
                    np.array([t_peak]),
                )
                derived_base_rows.append(
                    {
                        "h5_file": str(h5_path),
                        "source_pulse_index": int(pulse_index),
                        "peak_rank_by_amplitude": int(rank),
                        "peak_bin": int(peak_bin),
                        "peak_amp": float(peak_amp),
                        "argmax_bin": int(sampled_argmax_bins[local_i]),
                        "predicted_refh_bin": float(sampled_pred_bins[local_i]) if np.isfinite(sampled_pred_bins[local_i]) else np.nan,
                        "t_peak": float(t_peak),
                        "ref_lon": float(ref_lon[local_i]),
                        "ref_lat": float(ref_lat[local_i]),
                        "ref_h": float(ref_h[local_i]),
                        "rwstart_lon": float(rwstart_lon[local_i]),
                        "rwstart_lat": float(rwstart_lat[local_i]),
                        "rwstart_h": float(rwstart_h[local_i]),
                        "rwstop_lon": float(rwstop_lon[local_i]),
                        "rwstop_lat": float(rwstop_lat[local_i]),
                        "rwstop_h": float(rwstop_h[local_i]),
                        "local_beam_azimuth": float(az[local_i]),
                        "local_beam_elevation": float(el[local_i]),
                        "bin_size": float(sampled_bs[local_i]),
                        "refh_snr": float(optional["refh_snr"][local_i]) if np.isfinite(optional["refh_snr"][local_i]) else np.nan,
                        "sweep_num": int(optional["sweep_num"][local_i]) if np.isfinite(optional["sweep_num"][local_i]) else np.nan,
                        "track_num": int(optional["track_num"][local_i]) if np.isfinite(optional["track_num"][local_i]) else np.nan,
                        "segment_lon": float(segment["segment_lon"][0]),
                        "segment_lat": float(segment["segment_lat"][0]),
                        "segment_h": float(segment["segment_h"][0]),
                        "segment_ecef_x": float(segment["segment_ecef"][0, 0]),
                        "segment_ecef_y": float(segment["segment_ecef"][0, 1]),
                        "segment_ecef_z": float(segment["segment_ecef"][0, 2]),
                    }
                )
        derived_base_df = pd.DataFrame(derived_base_rows)

        beam_rule_rows: list[dict[str, object]] = []
        if len(derived_base_df):
            spacing = str(spacing_choice["range_spacing_interpretation"])
            for angle_unit in ["radian", "degree"]:
                for vector_direction in ["ref_to_sensor", "sensor_to_ref"]:
                    beam = propagate_from_refh_with_beam(
                        derived_base_df["ref_lon"].to_numpy(),
                        derived_base_df["ref_lat"].to_numpy(),
                        derived_base_df["ref_h"].to_numpy(),
                        derived_base_df["local_beam_azimuth"].to_numpy(),
                        derived_base_df["local_beam_elevation"].to_numpy(),
                        derived_base_df["predicted_refh_bin"].to_numpy(),
                        derived_base_df["peak_bin"].to_numpy(),
                        derived_base_df["bin_size"].to_numpy(),
                        angle_unit=angle_unit,
                        range_spacing_interpretation=spacing,
                        vector_direction=vector_direction,
                    )
                    segment_xyz = derived_base_df[["segment_ecef_x", "segment_ecef_y", "segment_ecef_z"]].to_numpy()
                    segment_delta = segment_xyz - beam["ref_ecef"]
                    disagreement = rms_distance_m(beam["beam_ecef"], segment_xyz)
                    direction_cos = cosine_similarity(beam["delta_ecef"], segment_delta)

                    argmax_beam = propagate_from_refh_with_beam(
                        ref_lon,
                        ref_lat,
                        ref_h,
                        az,
                        el,
                        sampled_pred_bins,
                        sampled_argmax_bins,
                        sampled_bs,
                        angle_unit=angle_unit,
                        range_spacing_interpretation=spacing,
                        vector_direction=vector_direction,
                    )
                    ref_xyz = geodetic_to_ecef(ref_lon, ref_lat, ref_h)
                    refh_return_error = rms_distance_m(argmax_beam["beam_ecef"], ref_xyz)

                    median_disagreement = float(np.nanmedian(disagreement))
                    p90_disagreement = float(np.nanpercentile(disagreement, 90))
                    median_refh_error = float(np.nanmedian(refh_return_error))
                    p90_refh_error = float(np.nanpercentile(refh_return_error, 90))
                    median_dir_cos = float(np.nanmedian(direction_cos)) if np.isfinite(direction_cos).any() else np.nan
                    p10_dir_cos = float(np.nanpercentile(direction_cos[np.isfinite(direction_cos)], 10)) if np.isfinite(direction_cos).any() else np.nan

                    beam_row = {
                        "h5_file": str(h5_path),
                        "angle_unit": angle_unit,
                        "angle_unit_inferred_from_values": angle_unit_selected,
                        "range_spacing_interpretation": spacing,
                        "beam_vector_direction": vector_direction,
                        "n_peak_rows": int(len(derived_base_df)),
                        "n_refh_return_rows": int(len(refh_return_error)),
                        "median_beam_vs_segment_disagreement_m": median_disagreement,
                        "p90_beam_vs_segment_disagreement_m": p90_disagreement,
                        "median_refh_return_error_m": median_refh_error,
                        "p90_refh_return_error_m": p90_refh_error,
                        "median_direction_cosine": median_dir_cos,
                        "p10_direction_cosine": p10_dir_cos,
                        "beam_model_agreement_quality": classify_beam_agreement(median_disagreement, p90_disagreement),
                        "refh_return_closure_quality": classify_refh_return_error(median_refh_error, p90_refh_error),
                        "direction_consistency_quality": classify_direction_consistency(median_dir_cos, p10_dir_cos),
                    }
                    beam_row["beam_rule_key"] = (
                        f"{beam_row['angle_unit']}|{beam_row['range_spacing_interpretation']}|{beam_row['beam_vector_direction']}"
                    )
                    beam_rule_rows.append(beam_row)
        beam_rule_scores_df = pd.DataFrame(beam_rule_rows)
        if len(beam_rule_scores_df):
            best_beam = beam_rule_scores_df.sort_values(
                [
                    "median_beam_vs_segment_disagreement_m",
                    "p90_beam_vs_segment_disagreement_m",
                    "median_refh_return_error_m",
                    "p90_refh_return_error_m",
                    "median_direction_cosine",
                ],
                ascending=[True, True, True, True, False],
            ).iloc[0].copy()
            best_beam_rule_df = pd.DataFrame([best_beam])
        else:
            best_beam_rule_df = pd.DataFrame(
                [
                    {
                        "h5_file": str(h5_path),
                        "angle_unit": np.nan,
                        "range_spacing_interpretation": spacing_choice["range_spacing_interpretation"],
                        "beam_vector_direction": np.nan,
                        "beam_rule_key": np.nan,
                    }
                ]
            )

        file_meta = {
            "h5_file": str(h5_path),
            "n_pulses": int(n_records),
            "n_rx_bins": int(n_bins),
            "sec_to_meters_attr": attrs.get("sec_to_meters"),
            "start_utca": attrs.get("start_utca"),
            "end_utca": attrs.get("end_utca"),
            "rx_bins_are_indices": bool(np.array_equal(np.asarray(h5[paths["rx_bins"]][:], dtype=float), np.arange(n_bins, dtype=float))),
            "angle_unit_inferred": angle_unit_selected,
            "angle_unit_guess_azimuth": az_guess,
            "angle_unit_guess_elevation": el_guess,
            "archive_time_notes": " | ".join(archive_notes),
        }
        return {
            "reference_df": reference_df,
            "best_reference_df": best_reference_df,
            "whole_window_df": whole_window_df,
            "spacing_choice_df": spacing_choice_df,
            "endpoint_df": endpoint_df,
            "derived_base_df": derived_base_df,
            "beam_rule_scores_df": beam_rule_scores_df,
            "best_beam_rule_df": best_beam_rule_df,
            "file_meta": file_meta,
        }


def combine_frames(audits: list[dict[str, pd.DataFrame | dict[str, object]]], key: str) -> pd.DataFrame:
    frames = [audit[key] for audit in audits if isinstance(audit[key], pd.DataFrame)]
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_final_decision(
    best_reference_df: pd.DataFrame,
    endpoint_df: pd.DataFrame,
    best_beam_rule_df: pd.DataFrame,
    file_meta_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = (
        file_meta_df
        .merge(best_reference_df[["h5_file", "candidate_model", "closure_quality", "valid_fraction", "median_abs_bin_diff_to_nearest_strong_peak", "p90_abs_bin_diff_to_nearest_strong_peak"]], on="h5_file", how="left")
        .merge(endpoint_df[["h5_file", "endpoint_segment_closure_quality", "median_refh_distance_to_rwstart_rwstop_segment_m", "median_abs_t_geom_minus_t_time", "off_segment_fraction"]], on="h5_file", how="left")
        .merge(best_beam_rule_df, on="h5_file", how="left")
    )
    rule_counts = merged["beam_rule_key"].value_counts(dropna=True)
    stable_rule_key = rule_counts.index[0] if len(rule_counts) and rule_counts.iloc[0] == len(merged) else None
    rule_stable_across_files = stable_rule_key is not None and len(merged) > 0

    status_rows = []
    for _, row in merged.iterrows():
        strong_ref = row["closure_quality"] == "strong_internal_closure"
        strong_endpoint = row["endpoint_segment_closure_quality"] == "strong_endpoint_segment_closure"
        strong_beam = (
            row.get("beam_model_agreement_quality") == "strong_beam_agreement"
            and row.get("refh_return_closure_quality") == "strong_refh_return_closure"
            and row.get("direction_consistency_quality") == "strong_direction_consistency"
        )
        moderate_beam = (
            row.get("beam_model_agreement_quality") in {"strong_beam_agreement", "moderate_beam_agreement"}
            and row.get("refh_return_closure_quality") in {"strong_refh_return_closure", "moderate_refh_return_closure"}
            and row.get("direction_consistency_quality") in {"strong_direction_consistency", "moderate_direction_consistency"}
        )

        notes: list[str] = []
        if pd.notna(row.get("sec_to_meters_attr")) and row.get("range_spacing_interpretation") == "bin_size_as_seconds_times_c":
            notes.append("Root attr sec_to_meters implies c/2, but whole-window closure favored bin_size * c.")
        if row.get("archive_time_notes"):
            notes.append(str(row["archive_time_notes"]))
        if not rule_stable_across_files:
            notes.append("Best beam rule is not stable across all audited files.")
        if row.get("angle_unit") != row.get("angle_unit_inferred"):
            notes.append("Best beam angle unit differs from value-range inference.")

        if strong_ref and strong_endpoint and strong_beam and rule_stable_across_files:
            status = "STRICT_BEAM_RULE_SUPPORTED"
        elif strong_ref and strong_endpoint and moderate_beam:
            status = "BEST_BEAM_RULE_LOW_CONFIDENCE"
        else:
            status = "NO_STRICT_BEAM_RULE_CONFIRMED"

        status_rows.append(
            {
                "scope": "file",
                "h5_file": row["h5_file"],
                "status": status,
                "best_reference_bin_model": row["candidate_model"],
                "reference_closure_quality": row["closure_quality"],
                "endpoint_segment_closure_quality": row["endpoint_segment_closure_quality"],
                "best_beam_angle_unit": row.get("angle_unit"),
                "best_beam_vector_direction": row.get("beam_vector_direction"),
                "range_spacing_interpretation": row.get("range_spacing_interpretation"),
                "beam_rule_key": row.get("beam_rule_key"),
                "beam_model_agreement_quality": row.get("beam_model_agreement_quality"),
                "refh_return_closure_quality": row.get("refh_return_closure_quality"),
                "direction_consistency_quality": row.get("direction_consistency_quality"),
                "median_beam_vs_segment_disagreement_m": row.get("median_beam_vs_segment_disagreement_m"),
                "p90_beam_vs_segment_disagreement_m": row.get("p90_beam_vs_segment_disagreement_m"),
                "median_refh_return_error_m": row.get("median_refh_return_error_m"),
                "p90_refh_return_error_m": row.get("p90_refh_return_error_m"),
                "median_direction_cosine": row.get("median_direction_cosine"),
                "p10_direction_cosine": row.get("p10_direction_cosine"),
                "rule_stable_across_files": rule_stable_across_files,
                "stable_rule_key": stable_rule_key,
                "notes": " | ".join(notes),
            }
        )

    file_status_df = pd.DataFrame(status_rows)
    file_statuses = set(file_status_df["status"])
    if file_statuses == {"STRICT_BEAM_RULE_SUPPORTED"} and rule_stable_across_files:
        global_status = "STRICT_BEAM_RULE_SUPPORTED"
    elif "BEST_BEAM_RULE_LOW_CONFIDENCE" in file_statuses or "STRICT_BEAM_RULE_SUPPORTED" in file_statuses:
        global_status = "BEST_BEAM_RULE_LOW_CONFIDENCE"
    else:
        global_status = "NO_STRICT_BEAM_RULE_CONFIRMED"

    global_row = {
        "scope": "global",
        "h5_file": "__GLOBAL__",
        "status": global_status,
        "best_reference_bin_model": None,
        "reference_closure_quality": None,
        "endpoint_segment_closure_quality": None,
        "best_beam_angle_unit": None,
        "best_beam_vector_direction": None,
        "range_spacing_interpretation": None,
        "beam_rule_key": stable_rule_key,
        "beam_model_agreement_quality": None,
        "refh_return_closure_quality": None,
        "direction_consistency_quality": None,
        "median_beam_vs_segment_disagreement_m": None,
        "p90_beam_vs_segment_disagreement_m": None,
        "median_refh_return_error_m": None,
        "p90_refh_return_error_m": None,
        "median_direction_cosine": None,
        "p10_direction_cosine": None,
        "rule_stable_across_files": rule_stable_across_files,
        "stable_rule_key": stable_rule_key,
        "notes": "Best beam rule is stable across files." if rule_stable_across_files else "No single beam rule was stable across all audited files.",
    }
    if file_status_df.empty:
        return pd.DataFrame([global_row])
    rows = file_status_df.to_dict(orient="records")
    rows.append({col: global_row.get(col) for col in file_status_df.columns})
    return pd.DataFrame(rows, columns=file_status_df.columns)


def write_summary(
    out_path: Path,
    h5_files: list[Path],
    file_meta_df: pd.DataFrame,
    best_reference_df: pd.DataFrame,
    spacing_choice_df: pd.DataFrame,
    best_beam_rule_df: pd.DataFrame,
    final_rule_df: pd.DataFrame,
) -> None:
    lines = [
        "# CASALS Beam Geolocation Rule Audit",
        "",
        "## Files",
    ]
    lines.extend([f"- `{p.name}`" for p in h5_files])
    lines.extend(["", "## Global Decision"])
    global_row = final_rule_df[final_rule_df["scope"] == "global"].iloc[0]
    lines.append(f"- Status: `{global_row['status']}`")
    lines.append(f"- Stable beam rule key: `{global_row['stable_rule_key']}`")
    lines.append(f"- Notes: {global_row['notes']}")
    lines.extend(["", "## Per-File Summary"])

    merged = (
        file_meta_df
        .merge(best_reference_df[["h5_file", "candidate_model", "closure_quality"]], on="h5_file", how="left")
        .merge(spacing_choice_df[["h5_file", "range_spacing_interpretation"]], on="h5_file", how="left")
        .merge(best_beam_rule_df[["h5_file", "angle_unit", "beam_vector_direction", "beam_rule_key", "median_beam_vs_segment_disagreement_m", "p90_beam_vs_segment_disagreement_m", "median_refh_return_error_m", "p90_refh_return_error_m", "median_direction_cosine", "p10_direction_cosine"]], on="h5_file", how="left")
        .merge(final_rule_df[final_rule_df["scope"] == "file"][["h5_file", "status", "notes"]], on="h5_file", how="left")
    )
    for _, row in merged.iterrows():
        lines.append(f"### {Path(row['h5_file']).name}")
        lines.append(f"- Status: `{row['status']}`")
        lines.append(f"- Best reference-bin model: `{row['candidate_model']}`")
        lines.append(f"- Reference closure: `{row['closure_quality']}`")
        lines.append(f"- Angle unit inferred from values: `{row['angle_unit_inferred']}`")
        lines.append(f"- Selected range spacing: `{row['range_spacing_interpretation']}`")
        lines.append(f"- Best beam rule: `{row['beam_rule_key']}`")
        lines.append(
            f"- Beam disagreement median/p90 (m): `{row['median_beam_vs_segment_disagreement_m']}` / `{row['p90_beam_vs_segment_disagreement_m']}`"
        )
        lines.append(
            f"- Refh-return error median/p90 (m): `{row['median_refh_return_error_m']}` / `{row['p90_refh_return_error_m']}`"
        )
        lines.append(
            f"- Direction cosine median/p10: `{row['median_direction_cosine']}` / `{row['p10_direction_cosine']}`"
        )
        lines.append(f"- Notes: {row['notes']}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    h5_files = resolve_h5_files(args.h5, args.glob)
    archive_notes = load_archive_time_evidence()

    audits = [audit_file(h5_path, args, archive_notes) for h5_path in h5_files]

    reference_df = combine_frames(audits, "reference_df")
    best_reference_df = combine_frames(audits, "best_reference_df")
    whole_window_df = combine_frames(audits, "whole_window_df")
    spacing_choice_df = combine_frames(audits, "spacing_choice_df")
    endpoint_df = combine_frames(audits, "endpoint_df")
    derived_base_df = combine_frames(audits, "derived_base_df")
    beam_rule_scores_df = combine_frames(audits, "beam_rule_scores_df")
    best_beam_rule_df = combine_frames(audits, "best_beam_rule_df")
    file_meta_df = pd.DataFrame([audit["file_meta"] for audit in audits])

    final_rule_df = build_final_decision(
        best_reference_df=best_reference_df,
        endpoint_df=endpoint_df,
        best_beam_rule_df=best_beam_rule_df,
        file_meta_df=file_meta_df,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    reference_df.to_csv(out_dir / "reference_bin_model_scores.csv", index=False)
    best_reference_df.to_csv(out_dir / "best_reference_bin_model.csv", index=False)
    whole_window_df.to_csv(out_dir / "whole_window_closure.csv", index=False)
    spacing_choice_df.to_csv(out_dir / "range_spacing_choice.csv", index=False)
    endpoint_df.to_csv(out_dir / "endpoint_segment_closure.csv", index=False)
    derived_base_df.to_csv(out_dir / "derived_peak_base_rows.csv", index=False)
    beam_rule_scores_df.to_csv(out_dir / "beam_rule_scores.csv", index=False)
    best_beam_rule_df.to_csv(out_dir / "best_beam_rule.csv", index=False)
    file_meta_df.to_csv(out_dir / "file_metadata.csv", index=False)
    final_rule_df.to_csv(out_dir / "final_rule_decision.csv", index=False)
    write_summary(
        out_path=out_dir / "summary.md",
        h5_files=h5_files,
        file_meta_df=file_meta_df,
        best_reference_df=best_reference_df,
        spacing_choice_df=spacing_choice_df,
        best_beam_rule_df=best_beam_rule_df,
        final_rule_df=final_rule_df,
    )

    print(f"Wrote outputs to: {out_dir}")
    print(final_rule_df.to_string(index=False))


if __name__ == "__main__":
    main()
