"""Export a CASALS L1B Level-A refh LAS.

Scientific meaning:
    Each point is the official CASALS L1B geolocated max-Rx-bin / refh
    reference-return point for one pulse.

Inputs:
    One CASALS L1B H5 file with refh lon/lat/height and quality fields.

Outputs:
    One Level-A refh LAS, one metadata JSON, and one preview PNG.

This script does not:
    - perform waveform decomposition,
    - create an official multi-return point cloud,
    - classify ground/canopy/buildings,
    - resolve vertical datum differences with 3DEP or NAVD88 products.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
from pyproj import CRS, Transformer

import laspy
from laspy import ExtraBytesParams

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Config:
    h5_path: Path
    point_cloud_dir: Path
    output_dir: Path
    filter_good_snr_only: bool = False
    refh_snr_min: Optional[float] = None
    sweep_range: Optional[Tuple[int, int]] = None
    track_range: Optional[Tuple[int, int]] = None
    write_las: bool = True
    write_metadata_json: bool = True
    write_preview_png: bool = True
    rgb_color_by: str = "refh_amp"
    robust_color_percentiles: Tuple[float, float] = (2.0, 98.0)
    preview_max_points: int = 300_000
    random_seed: int = 42
    las_xyz_scale_m: float = 0.001


# =============================================================================
# Utility functions
# =============================================================================

def normalize_h5_attr(value: Any) -> Any:
    """Convert HDF5 attributes to JSON-friendly Python scalars/lists."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def read_global_attributes(h5: h5py.File) -> Dict[str, Any]:
    """Read root-level HDF5 attributes into JSON-friendly values."""
    return {k: normalize_h5_attr(v) for k, v in h5.attrs.items()}


def require_datasets(h5: h5py.File, required: list[str]) -> None:
    """Fail early if required datasets are missing."""
    missing = [name for name in required if name not in h5]
    if missing:
        raise KeyError(
            "Missing required CASALS L1B datasets for Level-A refh point cloud: "
            + ", ".join(missing)
        )


def as_1d_array(h5: h5py.File, name: str) -> np.ndarray:
    """Read a 1D dataset and validate its dimensionality."""
    arr = np.asarray(h5[name][...])
    if arr.ndim != 1:
        raise ValueError(f"Dataset {name!r} is expected to be 1D, got shape {arr.shape}.")
    return arr


def infer_utm_epsg_from_lonlat(lon: np.ndarray, lat: np.ndarray) -> int:
    """
    Infer a UTM EPSG code from median longitude/latitude.

    Northern hemisphere: EPSG 326xx.
    Southern hemisphere: EPSG 327xx.
    """
    lon_med = float(np.nanmedian(lon))
    lat_med = float(np.nanmedian(lat))

    if not (-180.0 <= lon_med <= 180.0 and -90.0 <= lat_med <= 90.0):
        raise ValueError(f"Invalid median lon/lat for UTM inference: {lon_med}, {lat_med}")

    zone = int(math.floor((lon_med + 180.0) / 6.0) + 1)
    zone = max(1, min(zone, 60))

    return 32600 + zone if lat_med >= 0 else 32700 + zone


def build_valid_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    z: np.ndarray,
    refh_amp: np.ndarray,
    refh_snr: np.ndarray,
    good_snr: np.ndarray,
    track_num: np.ndarray,
    sweep_num: np.ndarray,
    *,
    filter_good_snr_only: bool,
    refh_snr_min: Optional[float],
    track_range: Optional[Tuple[int, int]],
    sweep_range: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Build a conservative mask for finite, geospatially valid refh points."""
    mask = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & np.isfinite(z)
        & (lon >= -180.0)
        & (lon <= 180.0)
        & (lat >= -90.0)
        & (lat <= 90.0)
    )

    # Preserve all finite amplitude values, including negative values if present.
    mask &= np.isfinite(refh_amp.astype(np.float64))
    mask &= np.isfinite(refh_snr.astype(np.float64))

    if filter_good_snr_only:
        mask &= good_snr.astype(bool)

    if refh_snr_min is not None:
        mask &= refh_snr.astype(np.float64) >= float(refh_snr_min)

    if track_range is not None:
        t0, t1 = track_range
        mask &= (track_num >= t0) & (track_num <= t1)

    if sweep_range is not None:
        s0, s1 = sweep_range
        mask &= (sweep_num >= s0) & (sweep_num <= s1)

    return mask


def robust_min_max(values: np.ndarray, percentiles: Tuple[float, float]) -> Tuple[float, float]:
    """Return robust min/max for color scaling."""
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return 0.0, 1.0

    p_low, p_high = np.nanpercentile(vals[finite], percentiles)
    p_low = float(p_low)
    p_high = float(p_high)

    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        vmin = float(np.nanmin(vals[finite]))
        vmax = float(np.nanmax(vals[finite]))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return vmin, vmax

    return p_low, p_high


def values_to_rgb16(
    values: np.ndarray,
    cmap_name: str = "viridis",
    percentiles: Tuple[float, float] = (2.0, 98.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Convert scalar values to LAS RGB uint16 using robust color scaling.

    This is only for visualization. It does not alter coordinates or georeference.
    """
    vals = np.asarray(values, dtype=np.float64)
    vmin, vmax = robust_min_max(vals, percentiles)

    norm = (vals - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    norm[~np.isfinite(norm)] = 0.0

    cmap = plt.get_cmap(cmap_name)
    rgb_float = cmap(norm)[:, :3]
    rgb16 = np.round(rgb_float * 65535.0).astype(np.uint16)

    return rgb16[:, 0], rgb16[:, 1], rgb16[:, 2], (vmin, vmax)


def make_preview_png(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    title: str,
    colorbar_label: str,
    max_points: int = 300_000,
    seed: int = 42,
) -> None:
    """Create a 2D projected-coordinate preview scatter plot."""
    n = len(x)
    if n == 0:
        warnings.warn(f"No points available for preview: {output_path}")
        return

    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
    else:
        idx = np.arange(n)

    plt.figure(figsize=(12, 8))
    sc = plt.scatter(
        x[idx],
        y[idx],
        c=np.asarray(values)[idx],
        s=0.25,
        linewidths=0,
        cmap="viridis",
    )
    plt.axis("equal")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title(title)
    plt.colorbar(sc, label=colorbar_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def validate_inverse_projection(
    x: np.ndarray,
    y: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    utm_epsg: int,
    sample_size: int = 50_000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Validate projected coordinates by inverse-transforming a random sample.

    This checks that the UTM transform was applied with the correct axis order.
    """
    n = len(x)
    if n == 0:
        return {
            "inverse_projection_sample_size": 0,
            "max_abs_lon_error_deg": float("nan"),
            "max_abs_lat_error_deg": float("nan"),
            "approx_max_horizontal_error_m": float("nan"),
        }

    idx = np.random.default_rng(seed).choice(n, size=sample_size, replace=False) if n > sample_size else np.arange(n)

    transformer_back = Transformer.from_crs(
        CRS.from_epsg(utm_epsg),
        CRS.from_epsg(4326),
        always_xy=True,
    )
    lon_back, lat_back = transformer_back.transform(x[idx], y[idx])

    lon_err = np.asarray(lon_back) - lon[idx]
    lat_err = np.asarray(lat_back) - lat[idx]

    max_abs_lon_error_deg = float(np.nanmax(np.abs(lon_err)))
    max_abs_lat_error_deg = float(np.nanmax(np.abs(lat_err)))

    lat_med_rad = np.deg2rad(float(np.nanmedian(lat[idx])))
    meters_per_deg_lon = 111_320.0 * max(math.cos(lat_med_rad), 1e-6)
    approx_max_horizontal_error_m = max(
        max_abs_lon_error_deg * meters_per_deg_lon,
        max_abs_lat_error_deg * 111_320.0,
    )

    return {
        "inverse_projection_sample_size": int(len(idx)),
        "max_abs_lon_error_deg": max_abs_lon_error_deg,
        "max_abs_lat_error_deg": max_abs_lat_error_deg,
        "approx_max_horizontal_error_m": float(approx_max_horizontal_error_m),
    }


def write_las_point_cloud(
    las_path: Path,
    easting: np.ndarray,
    northing: np.ndarray,
    z_refh: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    refh_amp: np.ndarray,
    refh_snr: np.ndarray,
    good_snr: np.ndarray,
    track_num: np.ndarray,
    sweep_num: np.ndarray,
    pulse_index: np.ndarray,
    utm_epsg: int,
    rgb_color_by: str,
    las_xyz_scale_m: float,
    robust_color_percentiles: Tuple[float, float],
    delta_time: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Write a georeferenced LAS file with CASALS Level-A refh points."""
    n = len(easting)
    if n == 0:
        raise ValueError("No points to write.")

    utm_crs = CRS.from_epsg(utm_epsg)

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([las_xyz_scale_m, las_xyz_scale_m, las_xyz_scale_m], dtype=np.float64)
    header.offsets = np.array(
        [
            math.floor(float(np.nanmin(easting))),
            math.floor(float(np.nanmin(northing))),
            math.floor(float(np.nanmin(z_refh))),
        ],
        dtype=np.float64,
    )
    header.system_identifier = "CASALS_L1B_REFH"
    header.generating_software = "export_refh_las.py"

    # Add projected horizontal CRS to LAS header.
    # The vertical coordinate is preserved as CASALS WGS84 ellipsoidal height,
    # documented in metadata JSON and extra dimensions.
    header.add_crs(utm_crs)

    las = laspy.LasData(header)

    extra_dims = [
        ExtraBytesParams(name="longitude",   type=np.float64, description="refh_lon_deg_WGS84"),
        ExtraBytesParams(name="latitude",    type=np.float64, description="refh_lat_deg_WGS84"),
        ExtraBytesParams(name="refh_amp_raw",type=np.float64, description="refh_amp_raw_counts"),
        ExtraBytesParams(name="refh_snr",    type=np.float64, description="refh_snr"),
        ExtraBytesParams(name="good_snr",    type=np.uint8,   description="good_snr_1_true"),
        ExtraBytesParams(name="track_num",   type=np.uint16,  description="track_channel_number"),
        ExtraBytesParams(name="sweep_num",   type=np.uint32,  description="sweep_number"),
        ExtraBytesParams(name="pulse_index", type=np.uint32,  description="zero_based_pulse_index"),
    ]
    if delta_time is not None:
        extra_dims.append(ExtraBytesParams(name="delta_time", type=np.float64, description="delta_time_sec_2018"))

    las.add_extra_dims(extra_dims)

    las.x = easting.astype(np.float64)
    las.y = northing.astype(np.float64)
    las.z = z_refh.astype(np.float64)

    # LAS intensity is unsigned 16-bit. Preserve original refh_amp in extra dimension.
    las.intensity = np.clip(np.asarray(refh_amp, dtype=np.float64), 0, 65535).astype(np.uint16)

    # LAS classification code 1 = unclassified.
    # Do NOT label as ground/canopy/building here.
    las.classification = np.ones(n, dtype=np.uint8)

    rgb_map = {"refh": z_refh, "refh_snr": refh_snr, "refh_amp": refh_amp}
    if rgb_color_by not in rgb_map:
        raise ValueError(f"Unsupported rgb_color_by={rgb_color_by!r}")
    rgb_label = rgb_color_by
    red, green, blue, rgb_range = values_to_rgb16(rgb_map[rgb_color_by], percentiles=robust_color_percentiles)
    las.red, las.green, las.blue = red, green, blue

    las.longitude    = lon.astype(np.float64)
    las.latitude     = lat.astype(np.float64)
    las.refh_amp_raw = np.asarray(refh_amp, dtype=np.float64)
    las.refh_snr     = np.asarray(refh_snr, dtype=np.float64)
    las.good_snr     = np.asarray(good_snr, dtype=np.uint8)
    las.track_num    = np.asarray(track_num, dtype=np.uint16)
    las.sweep_num    = np.asarray(sweep_num, dtype=np.uint32)
    las.pulse_index  = np.asarray(pulse_index, dtype=np.uint32)
    if delta_time is not None:
        las.delta_time = np.asarray(delta_time, dtype=np.float64)

    las.write(str(las_path))

    return {
        "las_path": str(las_path),
        "n_points_written": int(n),
        "las_horizontal_crs_epsg": int(utm_epsg),
        "las_horizontal_crs_name": utm_crs.name,
        "las_z_height_convention": "CASALS refh; WGS84 ellipsoidal height; no geoid/vertical datum conversion applied",
        "las_xyz_scale_m": float(las_xyz_scale_m),
        "las_offsets": [float(v) for v in header.offsets],
        "las_rgb_color_by": rgb_label,
        "las_rgb_robust_range": [float(rgb_range[0]), float(rgb_range[1])],
        "las_classification": "1 = unclassified for all points",
        "las_intensity": "refh_amp clipped to uint16 [0, 65535]; raw refh_amp preserved in extra dimension refh_amp_raw",
    }


def summarize_array(name: str, arr: np.ndarray) -> Dict[str, Any]:
    """Basic summary for numeric arrays."""
    vals = np.asarray(arr)
    vals_float = vals.astype(np.float64, copy=False)
    finite = np.isfinite(vals_float)

    if not np.any(finite):
        return {"name": name, "n": int(vals.size), "n_finite": 0,
                "min": None, "p02": None, "p50": None, "p98": None, "max": None}

    q = np.nanpercentile(vals_float[finite], [2, 50, 98])
    return {
        "name": name,
        "n": int(vals.size),
        "n_finite": int(np.sum(finite)),
        "min": float(np.nanmin(vals_float[finite])),
        "p02": float(q[0]),
        "p50": float(q[1]),
        "p98": float(q[2]),
        "max": float(np.nanmax(vals_float[finite])),
    }


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    cfg = Config(
        h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
        point_cloud_dir=Path(r"./point_cloud_data/export_refh_las"),
        output_dir=Path(r"./outputs/export_refh_las"),
        filter_good_snr_only=False,
        refh_snr_min=None,
        sweep_range=None,
        track_range=None,
        write_las=True,
        write_metadata_json=True,
        write_preview_png=True,
        rgb_color_by="refh_amp",
        robust_color_percentiles=(2.0, 98.0),
        preview_max_points=300_000,
        random_seed=42,
        las_xyz_scale_m=0.001,
    )

    required_datasets = [
        "refh_longitude", "refh_latitude", "refh",
        "refh_amp", "refh_snr", "good_snr", "track_num", "sweep_num",
    ]

    # -------------------------------------------------------------------------
    # Workflow
    # -------------------------------------------------------------------------
    cfg.point_cloud_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.h5_path.exists():
        raise FileNotFoundError(f"Input H5 file does not exist: {cfg.h5_path}")

    print("=" * 80)
    print("CASALS Level-A max-Rx-bin / refh point cloud generation")
    print("=" * 80)
    print(f"Input H5: {cfg.h5_path.resolve()}")
    print(f"Point-cloud directory: {cfg.point_cloud_dir.resolve()}")
    print(f"Output directory: {cfg.output_dir.resolve()}")
    print()

    with h5py.File(cfg.h5_path, "r") as h5:
        attrs = read_global_attributes(h5)
        require_datasets(h5, required_datasets)

        print("Reading required 1D georeference/reference-return datasets...")
        lon      = as_1d_array(h5, "refh_longitude").astype(np.float64)
        lat      = as_1d_array(h5, "refh_latitude").astype(np.float64)
        z_refh   = as_1d_array(h5, "refh").astype(np.float64)
        refh_amp = as_1d_array(h5, "refh_amp")
        refh_snr = as_1d_array(h5, "refh_snr").astype(np.float64)
        good_snr = as_1d_array(h5, "good_snr").astype(bool)
        track_num = as_1d_array(h5, "track_num")
        sweep_num = as_1d_array(h5, "sweep_num")

        delta_time = as_1d_array(h5, "delta_time").astype(np.float64) if "delta_time" in h5 else None

        n_pulses    = len(lon)
        pulse_index = np.arange(n_pulses, dtype=np.uint32)

        shapes = {
            "refh_longitude": lon.shape, "refh_latitude": lat.shape, "refh": z_refh.shape,
            "refh_amp": refh_amp.shape,  "refh_snr": refh_snr.shape, "good_snr": good_snr.shape,
            "track_num": track_num.shape, "sweep_num": sweep_num.shape,
        }
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Required arrays have inconsistent shapes: {shapes}")

        if "n_pulses" in attrs and int(attrs["n_pulses"]) != n_pulses:
            raise ValueError(
                f"HDF5 attribute n_pulses={attrs['n_pulses']} does not match array length {n_pulses}."
            )

        print(f"Total pulse/reference records: {n_pulses:,}")
        print(f"File start UTC: {attrs.get('start_utca', 'unknown')}")
        print(f"File end UTC:   {attrs.get('end_utca', 'unknown')}")
        print()

        mask = build_valid_mask(
            lon=lon, lat=lat, z=z_refh,
            refh_amp=refh_amp, refh_snr=refh_snr, good_snr=good_snr,
            track_num=track_num, sweep_num=sweep_num,
            filter_good_snr_only=cfg.filter_good_snr_only,
            refh_snr_min=cfg.refh_snr_min,
            track_range=cfg.track_range,
            sweep_range=cfg.sweep_range,
        )

        n_valid = int(np.sum(mask))
        if n_valid == 0:
            raise RuntimeError("No valid refh points after filtering.")

        print("Filtering summary:")
        print(f"  finite + valid lon/lat/refh records: {n_valid:,} / {n_pulses:,}")
        print(f"  filter_good_snr_only: {cfg.filter_good_snr_only}")
        print(f"  refh_snr_min: {cfg.refh_snr_min}")
        print(f"  track_range: {cfg.track_range}")
        print(f"  sweep_range: {cfg.sweep_range}")
        print(f"  good_snr fraction in full file: {float(np.mean(good_snr)):.6f}")
        print(f"  good_snr fraction after filtering: {float(np.mean(good_snr[mask])):.6f}")
        print()

        # Apply mask.
        lon_f       = lon[mask]
        lat_f       = lat[mask]
        z_f         = z_refh[mask]
        amp_f       = refh_amp[mask]
        snr_f       = refh_snr[mask]
        good_f      = good_snr[mask]
        track_f     = track_num[mask]
        sweep_f     = sweep_num[mask]
        pulse_f     = pulse_index[mask]
        delta_time_f = delta_time[mask] if delta_time is not None else None

    # Infer projected CRS from valid lon/lat.
    utm_epsg      = infer_utm_epsg_from_lonlat(lon_f, lat_f)
    geographic_crs = CRS.from_epsg(4326)
    utm_crs        = CRS.from_epsg(utm_epsg)

    print("Coordinate reference system:")
    print("  Input horizontal CRS: EPSG:4326, WGS84 geographic lon/lat")
    print(f"  Output horizontal CRS: EPSG:{utm_epsg}, {utm_crs.name}")
    print("  Output Z: CASALS refh, WGS84 ellipsoidal height; no vertical datum conversion")
    print()

    transformer = Transformer.from_crs(geographic_crs, utm_crs, always_xy=True)
    easting, northing = transformer.transform(lon_f, lat_f)
    easting   = np.asarray(easting,   dtype=np.float64)
    northing  = np.asarray(northing,  dtype=np.float64)

    projection_check = validate_inverse_projection(
        x=easting, y=northing, lon=lon_f, lat=lat_f, utm_epsg=utm_epsg, seed=cfg.random_seed,
    )

    print("Projection validation:")
    for key, value in projection_check.items():
        print(f"  {key}: {value}")
    print()

    print("Coordinate and attribute summaries after filtering:")
    for summary in [
        summarize_array("longitude_deg",             lon_f),
        summarize_array("latitude_deg",              lat_f),
        summarize_array("easting_m",                 easting),
        summarize_array("northing_m",                northing),
        summarize_array("refh_ellipsoidal_height_m", z_f),
        summarize_array("refh_amp",                  amp_f),
        summarize_array("refh_snr",                  snr_f),
        summarize_array("track_num",                 track_f),
        summarize_array("sweep_num",                 sweep_f),
    ]:
        print(json.dumps(summary, indent=2))
    print()

    base = cfg.h5_path.stem
    las_path = cfg.point_cloud_dir / f"{base}_levelA_refh_epsg{utm_epsg}.las"
    metadata_path = cfg.output_dir / f"{base}_levelA_refh_metadata.json"
    preview_path = cfg.output_dir / f"{base}_levelA_refh_preview.png"

    las_info: Dict[str, Any] = {}
    if cfg.write_las:
        print(f"Writing LAS point cloud: {las_path}")
        las_info = write_las_point_cloud(
            las_path=las_path,
            easting=easting, northing=northing, z_refh=z_f,
            lon=lon_f, lat=lat_f,
            refh_amp=amp_f, refh_snr=snr_f, good_snr=good_f,
            track_num=track_f, sweep_num=sweep_f, pulse_index=pulse_f,
            utm_epsg=utm_epsg,
            rgb_color_by=cfg.rgb_color_by,
            las_xyz_scale_m=cfg.las_xyz_scale_m,
            robust_color_percentiles=cfg.robust_color_percentiles,
            delta_time=delta_time_f,
        )
        print("LAS writing complete.")
        print()

    if cfg.write_preview_png:
        print("Writing preview PNG...")
        make_preview_png(
            x=easting, y=northing, values=amp_f.astype(np.float64),
            output_path=preview_path,
            title="CASALS Level-A refh preview: max-Rx-bin amplitude",
            colorbar_label="refh_amp, raw counts",
            max_points=cfg.preview_max_points, seed=cfg.random_seed,
        )
        print(f"  {preview_path}")
        print()

    metadata = {
        "script": "export_refh_las.py",
        "source_h5": str(cfg.h5_path.resolve()),
        "source_file_stem": base,
        "casals_product_level": "L1B",
        "point_cloud_level": "Level-A max-Rx-bin / refh reference-return point cloud",
        "scientific_notes": [
            "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
            "refh is WGS84 ellipsoidal height unless otherwise documented.",
            "This is not an official multi-return point cloud.",
            "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            "The official geolocated refh point corresponds to the Rx waveform maximum-amplitude bin.",
        ],
        "method": {
            "x_source": "refh_longitude projected from EPSG:4326 to inferred UTM",
            "y_source": "refh_latitude projected from EPSG:4326 to inferred UTM",
            "z_source": "refh, WGS84 ellipsoidal height",
            "intensity_source": "refh_amp clipped to LAS uint16; raw preserved in refh_amp_raw extra dimension",
            "classification": "All points are LAS class 1 unclassified; no ground/canopy/building classification applied",
            "waveform_decomposition": "Not applied",
            "vertical_datum_conversion": "Not applied",
        },
        "crs": {
            "input_horizontal_crs": "EPSG:4326 WGS84 geographic",
            "output_horizontal_crs_epsg": int(utm_epsg),
            "output_horizontal_crs_name": utm_crs.name,
            "output_horizontal_crs_wkt": utm_crs.to_wkt(),
            "z_height_convention": "WGS84 ellipsoidal height from CASALS refh",
        },
        "filters": {
            "finite_valid_lon_lat_refh": True,
            "filter_good_snr_only": bool(cfg.filter_good_snr_only),
            "refh_snr_min": cfg.refh_snr_min,
            "track_range_inclusive": cfg.track_range,
            "sweep_range_inclusive": cfg.sweep_range,
        },
        "counts": {
            "n_input_records": int(n_pulses),
            "n_output_points": int(n_valid),
            "good_snr_fraction_input": float(np.mean(good_snr)),
            "good_snr_fraction_output": float(np.mean(good_f)),
        },
        "bounds": {
            "longitude_min": float(np.nanmin(lon_f)),
            "longitude_max": float(np.nanmax(lon_f)),
            "latitude_min": float(np.nanmin(lat_f)),
            "latitude_max": float(np.nanmax(lat_f)),
            "easting_min_m": float(np.nanmin(easting)),
            "easting_max_m": float(np.nanmax(easting)),
            "northing_min_m": float(np.nanmin(northing)),
            "northing_max_m": float(np.nanmax(northing)),
            "refh_min_m": float(np.nanmin(z_f)),
            "refh_max_m": float(np.nanmax(z_f)),
        },
        "projection_validation": projection_check,
        "outputs": {
            "las": las_info.get("las_path") if las_info else None,
            "metadata_json": str(metadata_path) if cfg.write_metadata_json else None,
            "preview_png": str(preview_path) if cfg.write_preview_png else None,
        },
        "config": asdict(cfg),
        "source_global_attributes_subset": {
            k: attrs.get(k) for k in (
                "tdms_file", "l1a_file", "ard_file", "geoloc_file",
                "start_utca", "end_utca",
                "n_pulses", "n_sweeps", "n_tracks", "n_rx_bins", "n_tx_bins",
            )
        },
    }

    if las_info:
        metadata["las"] = las_info

    if cfg.write_metadata_json:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata JSON written: {metadata_path}")

    print()
    print("=" * 80)
    print("Done.")
    print("=" * 80)
    if cfg.write_las:
        print(f"LAS point cloud: {las_path}")
    if cfg.write_metadata_json:
        print(f"Metadata: {metadata_path}")
    if cfg.write_preview_png:
        print(f"Preview: {preview_path}")
    print()
    print("Reminder:")
    print("  Z is CASALS refh WGS84 ellipsoidal height.")
    print("  This is not an orthometric DEM height and not a ground-classified point cloud.")


if __name__ == "__main__":
    main()
