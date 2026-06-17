"""Visualize CASALS L1B refh points in Open3D.

Scientific meaning:
    Each displayed point is one CASALS L1B geolocated max-Rx-bin / refh
    reference-return point.

Outputs:
    An interactive Open3D view, optional sampled PLY, metadata JSON, and colorbar PNG.

This script does not:
    - write formal LAS products,
    - perform filtering outputs,
    - replace the official CASALS refh point definition.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
from pyproj import CRS, Transformer
import laspy

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required for SNR color mapping and colorbar export.") from exc

try:
    import open3d as o3d
except Exception as exc:  # pragma: no cover
    raise ImportError("open3d is required. Install with: conda install -c conda-forge open3d") from exc


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    h5_path: Path
    point_cloud_dir: Path
    out_dir: Path
    classification_las_path: Optional[Path] = None

    # SNR source:
    #   "refh_snr"        -> use stored H5 refh_snr
    #   "amp_over_thres"  -> compute refh_amp / refh_thres
    # They should be identical for the tested CASALS L1B granule.
    snr_source: str = "refh_snr"

    # Keep points in this SNR interval before visualization.
    # Use None to disable one side.
    snr_min: Optional[float] = 2.0
    snr_max: Optional[float] = None

    # Optional spatial/track/sweep subsets. Keep None for full H5.
    track_range: Optional[Tuple[int, int]] = None       # e.g., (0, 255)
    sweep_range: Optional[Tuple[int, int]] = None       # e.g., (5000, 6000)
    bbox: Optional[Tuple[float, float, float, float]] = None
    # bbox = (lon_min, lat_min, lon_max, lat_max)

    # Visualization sampling. Open3D can struggle with millions of points.
    max_display_points: int = 700_000
    random_seed: int = 42

    # Sampling mode:
    #   "random"        -> random sample from filtered points
    #   "snr_stratified"-> preserve all SNR ranges more evenly
    sampling_mode: str = "snr_stratified"

    # Coordinate handling for Open3D visualization.
    # Centering avoids large-coordinate numerical/viewing issues.
    center_xy_for_view: bool = True
    z_scale_for_view: float = 1.0

    # Color mode: "snr", "amp", "height", "good_snr", or "classification"
    color_mode: str = "snr"

    # Continuous color stretch for scalar color modes. Values outside are clipped for color only.
    color_min: Optional[float] = None
    color_max: Optional[float] = None
    colormap_name: str = "viridis"

    # Open3D render controls.
    point_size: float = 2.0
    show_coordinate_frame: bool = True
    coordinate_frame_size_m: float = 60.0

    # Optional outputs.
    save_sampled_ply: bool = True
    save_metadata_json: bool = True
    save_snr_colorbar_png: bool = True


# =============================================================================
# Basic utilities
# =============================================================================

def require_dataset(h5: h5py.File, name: str) -> None:
    if name not in h5:
        raise KeyError(f"Required H5 dataset missing: {name}")


def read_1d(h5: h5py.File, name: str, dtype=None) -> np.ndarray:
    require_dataset(h5, name)
    arr = np.asarray(h5[name][...], dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"Dataset {name!r} must be 1D, got shape {arr.shape}")
    return arr


def infer_utm_epsg_from_lonlat(lon: np.ndarray, lat: np.ndarray) -> int:
    lon_med = float(np.nanmedian(lon))
    lat_med = float(np.nanmedian(lat))
    if not (-180.0 <= lon_med <= 180.0 and -90.0 <= lat_med <= 90.0):
        raise ValueError(f"Invalid median lon/lat: {lon_med}, {lat_med}")
    zone = int(math.floor((lon_med + 180.0) / 6.0) + 1)
    zone = max(1, min(zone, 60))
    return 32600 + zone if lat_med >= 0 else 32700 + zone


def safe_percentiles(x: np.ndarray, qs=(0, 1, 2, 5, 25, 50, 75, 95, 98, 99, 100)) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    out: Dict[str, float] = {"n": int(x.size), "n_finite": int(np.sum(finite))}
    if not np.any(finite):
        return out
    vals = np.nanpercentile(x[finite], qs)
    for q, v in zip(qs, vals):
        out[f"p{q:g}"] = float(v)
    return out


def normalize_to_unit(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Invalid color normalization range: {vmin}, {vmax}")
    t = (vals - vmin) / (vmax - vmin)
    t = np.clip(t, 0.0, 1.0)
    t[~np.isfinite(t)] = 0.0
    return t


def continuous_colormap(values: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    t = normalize_to_unit(values, vmin, vmax)
    cmap = plt.get_cmap(cmap_name)
    return cmap(t)[:, :3].astype(np.float64)


def good_snr_colors(good_snr: np.ndarray) -> np.ndarray:
    g = np.asarray(good_snr, dtype=bool)
    colors = np.zeros((g.size, 3), dtype=np.float64)
    colors[~g] = np.array([0.45, 0.45, 0.45])
    colors[g] = np.array([1.00, 0.05, 0.20])
    return colors


def classification_colors(classification: np.ndarray) -> np.ndarray:
    cls = np.asarray(classification)
    colors = np.zeros((cls.size, 3), dtype=np.float64)
    colors[:] = np.array([0.45, 0.45, 0.45])
    colors[cls == 1] = np.array([0.15, 0.45, 0.95])
    colors[cls == 2] = np.array([0.10, 0.75, 0.20])
    colors[cls == 7] = np.array([0.95, 0.15, 0.15])
    return colors


def save_colorbar(out_path: Path, cmap_name: str, vmin: float, vmax: float, label: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 1.1))
    fig.subplots_adjust(bottom=0.45)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label(label)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Data reading and filtering
# =============================================================================

def read_casals_refh_fields(h5_path: Path, snr_source: str) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        lon = read_1d(h5, "refh_longitude", np.float64)
        lat = read_1d(h5, "refh_latitude", np.float64)
        refh = read_1d(h5, "refh", np.float64)
        refh_amp = read_1d(h5, "refh_amp", np.float64)
        good_snr = read_1d(h5, "good_snr").astype(bool)
        track_num = read_1d(h5, "track_num")
        sweep_num = read_1d(h5, "sweep_num")

        refh_snr_stored = read_1d(h5, "refh_snr", np.float64)

        refh_thres = None
        if "refh_thres" in h5:
            refh_thres = read_1d(h5, "refh_thres", np.float64)

        if snr_source == "refh_snr":
            snr = refh_snr_stored
        elif snr_source == "amp_over_thres":
            if refh_thres is None:
                raise KeyError("snr_source='amp_over_thres' requires H5 dataset 'refh_thres'.")
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = refh_amp / refh_thres
        else:
            raise ValueError("snr_source must be 'refh_snr' or 'amp_over_thres'.")

        out = {
            "lon": lon,
            "lat": lat,
            "refh": refh,
            "refh_amp": refh_amp,
            "refh_snr": snr.astype(np.float64),
            "refh_snr_stored": refh_snr_stored,
            "good_snr": good_snr,
            "track_num": track_num,
            "sweep_num": sweep_num,
            "pulse_index": np.arange(lon.size, dtype=np.uint32),
        }
        if refh_thres is not None:
            out["refh_thres"] = refh_thres
        return out


def build_mask(data: Dict[str, np.ndarray], cfg: Config) -> np.ndarray:
    lon = data["lon"]
    lat = data["lat"]
    refh = data["refh"]
    snr = data["refh_snr"]
    track = data["track_num"]
    sweep = data["sweep_num"]

    mask = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & np.isfinite(refh)
        & np.isfinite(snr)
        & (lon >= -180.0)
        & (lon <= 180.0)
        & (lat >= -90.0)
        & (lat <= 90.0)
    )

    if cfg.snr_min is not None:
        mask &= snr >= float(cfg.snr_min)
    if cfg.snr_max is not None:
        mask &= snr <= float(cfg.snr_max)

    if cfg.track_range is not None:
        t0, t1 = cfg.track_range
        mask &= (track >= t0) & (track <= t1)

    if cfg.sweep_range is not None:
        s0, s1 = cfg.sweep_range
        mask &= (sweep >= s0) & (sweep <= s1)

    if cfg.bbox is not None:
        lon_min, lat_min, lon_max, lat_max = cfg.bbox
        mask &= (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

    return mask


def choose_display_indices(snr: np.ndarray, max_points: int, mode: str, seed: int) -> np.ndarray:
    n = snr.size
    if n <= max_points:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    if mode == "random":
        return np.sort(rng.choice(n, size=max_points, replace=False))

    if mode != "snr_stratified":
        raise ValueError("sampling_mode must be 'random' or 'snr_stratified'.")

    # Stratify by SNR bands so high-SNR rare points remain visible.
    bins = [
        (-np.inf, 2.0),
        (2.0, 3.0),
        (3.0, 4.0),
        (4.0, 4.5),
        (4.5, 5.0),
        (5.0, np.inf),
    ]
    groups = []
    for lo, hi in bins:
        groups.append(np.flatnonzero((snr >= lo) & (snr < hi)))

    nonempty = [g for g in groups if g.size > 0]
    if not nonempty:
        return np.sort(rng.choice(n, size=max_points, replace=False))

    # Allocate half uniformly across bands, half proportional to band size.
    base_each = max(1, int(0.5 * max_points / len(nonempty)))
    selected = []
    remaining_budget = max_points

    for g in nonempty:
        take = min(g.size, base_each, remaining_budget)
        if take > 0:
            selected.append(rng.choice(g, size=take, replace=False))
            remaining_budget -= take

    if remaining_budget > 0:
        already = np.concatenate(selected) if selected else np.array([], dtype=int)
        already_mask = np.zeros(n, dtype=bool)
        already_mask[already] = True
        rest = np.flatnonzero(~already_mask)
        take = min(rest.size, remaining_budget)
        if take > 0:
            selected.append(rng.choice(rest, size=take, replace=False))

    return np.sort(np.concatenate(selected))


# =============================================================================
# Open3D visualization
# =============================================================================

def make_open3d_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def visualize_open3d(pcd: o3d.geometry.PointCloud, cfg: Config) -> None:
    geometries = [pcd]
    if cfg.show_coordinate_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cfg.coordinate_frame_size_m)
        geometries.append(frame)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="CASALS refh points colored by SNR", width=1400, height=950)
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = float(cfg.point_size)
    opt.background_color = np.asarray([0.0, 0.0, 0.0])

    print("\nOpen3D controls:")
    print("  Mouse drag: rotate")
    print("  Mouse wheel: zoom")
    print("  Shift/Ctrl + drag: pan")
    print("  Press 'Q' or close the window to exit")
    print()

    vis.run()
    vis.destroy_window()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    cfg = Config(
        h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
        point_cloud_dir=Path(r"./point_cloud_data/view_refh_points"),
        out_dir=Path(r"./outputs/view_refh_points"),

        # Recommended choices:
        #   2.0: many points, noisy but spatially complete
        #   3.0: moderate filtering
        #   4.0: cleaner visualization
        #   4.5: strict visualization
        #   5.0: same as good_snr=True for the tested granule, very sparse
        snr_min=3.0,
        snr_max=None,

        # Try these color modes:
        color_mode="snr",
        color_min=None,
        color_max=None,
        colormap_name="viridis",

        max_display_points=700_000,
        sampling_mode="snr_stratified",
        point_size=2.0,

        save_sampled_ply=False,
        save_metadata_json=True,
        save_snr_colorbar_png=True,
    )

    cfg.point_cloud_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.h5_path.exists():
        raise FileNotFoundError(f"Input H5 does not exist: {cfg.h5_path}")

    print("=" * 88)
    print("CASALS L1B refh Open3D visualization")
    print("=" * 88)
    print(f"H5: {cfg.h5_path.resolve()}")
    print(f"Point-cloud directory: {cfg.point_cloud_dir.resolve()}")
    print(f"Output directory: {cfg.out_dir.resolve()}")
    print(f"SNR source: {cfg.snr_source}")
    print(f"SNR display filter: [{cfg.snr_min}, {cfg.snr_max}]")
    print(f"Color mode: {cfg.color_mode}")
    print()

    data = read_casals_refh_fields(cfg.h5_path, cfg.snr_source)
    classification_full = np.ones(data["lon"].size, dtype=np.uint8)
    classification_source = "default_all_class_1"
    if cfg.classification_las_path is not None and cfg.classification_las_path.exists():
        las = laspy.read(cfg.classification_las_path)
        las_class = np.asarray(las.classification, dtype=np.uint8)
        if hasattr(las, "pulse_index"):
            pulse_idx = np.asarray(las.pulse_index, dtype=np.int64)
            valid = (pulse_idx >= 0) & (pulse_idx < classification_full.size)
            classification_full[pulse_idx[valid]] = las_class[valid]
            classification_source = f"pulse_index_from_{cfg.classification_las_path.name}"
        elif las_class.size == classification_full.size:
            classification_full = las_class
            classification_source = f"same_order_from_{cfg.classification_las_path.name}"
    n_total = data["lon"].size

    if "refh_thres" in data:
        diff = data["refh_snr_stored"] - (data["refh_amp"] / data["refh_thres"])
        finite_diff = diff[np.isfinite(diff)]
        print("SNR identity check: refh_snr - refh_amp/refh_thres")
        print(f"  max abs diff: {float(np.nanmax(np.abs(finite_diff))):.6g}")
        print(f"  p95 abs diff: {float(np.nanpercentile(np.abs(finite_diff), 95)):.6g}")
        print()

    mask = build_mask(data, cfg)
    n_keep = int(np.sum(mask))
    if n_keep == 0:
        raise RuntimeError("No points left after filtering. Lower snr_min or relax subset settings.")

    print("Filtering summary:")
    print(f"  total records: {n_total:,}")
    print(f"  kept before display sampling: {n_keep:,} ({n_keep / n_total:.4%})")
    print(f"  good_snr fraction in full file: {float(np.mean(data['good_snr'])):.6%}")
    print(f"  good_snr fraction after filter: {float(np.mean(data['good_snr'][mask])):.6%}")
    print("  SNR summary after filter:")
    print(json.dumps(safe_percentiles(data["refh_snr"][mask]), indent=2))
    print()

    lon = data["lon"][mask]
    lat = data["lat"][mask]
    refh = data["refh"][mask]
    snr = data["refh_snr"][mask]
    amp = data["refh_amp"][mask]
    good_snr = data["good_snr"][mask]
    classification = classification_full[mask]

    utm_epsg = infer_utm_epsg_from_lonlat(lon, lat)
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(utm_epsg), always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    easting = np.asarray(easting, dtype=np.float64)
    northing = np.asarray(northing, dtype=np.float64)
    z = np.asarray(refh, dtype=np.float64) * float(cfg.z_scale_for_view)

    display_idx = choose_display_indices(
        snr=snr,
        max_points=int(cfg.max_display_points),
        mode=cfg.sampling_mode,
        seed=int(cfg.random_seed),
    )

    x_display = easting[display_idx]
    y_display = northing[display_idx]
    z_display = z[display_idx]
    snr_display = snr[display_idx]
    amp_display = amp[display_idx]
    good_display = good_snr[display_idx]
    class_display = classification[display_idx]

    view_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if cfg.center_xy_for_view:
        view_offset = np.array([
            float(np.nanmedian(x_display)),
            float(np.nanmedian(y_display)),
            0.0,
        ])

    points = np.column_stack([x_display, y_display, z_display]) - view_offset

    if cfg.color_mode == "snr":
        scalar_values = snr_display
        color_label = "refh_snr"
    elif cfg.color_mode == "amp":
        scalar_values = amp_display
        color_label = "refh_amp"
    elif cfg.color_mode == "height":
        scalar_values = z_display
        color_label = "refh height (m)"
    elif cfg.color_mode == "good_snr":
        scalar_values = None
        color_label = "good_snr"
        colors = good_snr_colors(good_display)
    elif cfg.color_mode == "classification":
        scalar_values = None
        color_label = "classification"
        colors = classification_colors(class_display)
    else:
        raise ValueError("color_mode must be 'snr', 'amp', 'height', 'good_snr', or 'classification'.")

    if cfg.color_mode in {"snr", "amp", "height"}:
        vmin = float(cfg.color_min) if cfg.color_min is not None else float(np.nanpercentile(scalar_values, 2))
        vmax = float(cfg.color_max) if cfg.color_max is not None else float(np.nanpercentile(scalar_values, 98))
        colors = continuous_colormap(
            scalar_values,
            vmin=vmin,
            vmax=vmax,
            cmap_name=cfg.colormap_name,
        )

    print("Display summary:")
    print(f"  UTM EPSG: {utm_epsg} ({CRS.from_epsg(utm_epsg).name})")
    print(f"  display points: {len(display_idx):,}")
    print(f"  view offset subtracted from XYZ: {view_offset.tolist()}")
    print(f"  classification source: {classification_source}")
    if cfg.color_mode in {'snr', 'amp', 'height'}:
        print(f"  color stretch: [{vmin}, {vmax}]")
    print()

    pcd = make_open3d_point_cloud(points, colors)

    base = cfg.h5_path.stem
    if cfg.save_sampled_ply:
        ply_path = cfg.point_cloud_dir / f"{base}_refh_view_sample_{cfg.color_mode}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=False)
        print(f"Sampled colored PLY written: {ply_path}")

    if cfg.save_snr_colorbar_png and cfg.color_mode in {"snr", "amp", "height"}:
        colorbar_path = cfg.out_dir / f"{base}_{cfg.color_mode}_colorbar.png"
        save_colorbar(
            colorbar_path,
            cmap_name=cfg.colormap_name,
            vmin=vmin,
            vmax=vmax,
            label=color_label,
        )
        print(f"Colorbar written: {colorbar_path}")

    if cfg.save_metadata_json:
        metadata = {
            "script": "view_refh_points.py",
            "source_h5": str(cfg.h5_path.resolve()),
            "config": {**asdict(cfg), "h5_path": str(cfg.h5_path), "out_dir": str(cfg.out_dir)},
            "n_total_records": int(n_total),
            "n_kept_after_filter": int(n_keep),
            "n_display_points": int(len(display_idx)),
            "utm_epsg": int(utm_epsg),
            "utm_crs_name": CRS.from_epsg(utm_epsg).name,
            "view_offset_xyz_subtracted": [float(v) for v in view_offset],
            "classification_source": classification_source,
            "snr_summary_full": safe_percentiles(data["refh_snr"]),
            "snr_summary_after_filter": safe_percentiles(snr),
            "snr_summary_display": safe_percentiles(snr_display),
            "good_snr_fraction_full": float(np.mean(data["good_snr"])),
            "good_snr_fraction_after_filter": float(np.mean(good_snr)),
            "good_snr_fraction_display": float(np.mean(good_display)),
            "scientific_notes": [
                "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
                "refh is WGS84 ellipsoidal height unless otherwise documented.",
                "This is not an official multi-return point cloud.",
                "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            ],
        }
        metadata_path = cfg.out_dir / f"{base}_view_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata written: {metadata_path}")

    visualize_open3d(pcd, cfg)


if __name__ == "__main__":
    main()
