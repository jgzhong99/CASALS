# -*- coding: utf-8 -*-
"""Filter CASALS L1B refh points into raw, noise-labeled, and clean LAS outputs.

Scientific meaning:
    Each input point is the official CASALS L1B geolocated refh reference point,
    corresponding to the Rx waveform maximum-amplitude bin.

Outputs:
    `raw_refh.las`, `noise_labeled_refh.las`, `clean_refh.las`, one metadata JSON,
    and core preview PNGs.

This script does not:
    - create a ground DEM,
    - create an official multi-return point cloud,
    - use waveform peak detection as a primary geolocation workflow,
    - resolve vertical datum differences with 3DEP/NAVD88 products.
"""

from __future__ import annotations

import json
import time
import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .noise import NoiseResult as FilterResult, label_noise
from .refh import (
    ProjectedRefh,
    RefhPointData,
    RefhSelection,
    project_refh_points,
    read_refh_points,
    write_refh_las,
    summarize_refh_array,
)


# =============================================================================
# Configuration dataclass
# =============================================================================

@dataclass(frozen=True)
class Config:
    # Input/output.
    h5_path: Path
    point_cloud_dir: Path
    output_dir: Path

    # CRS handling.
    # If None, infer WGS84 UTM zone from median refh lon/lat.
    utm_epsg_override: Optional[int] = None
    las_xyz_scale_m: float = 0.001

    # Required filter behavior.
    # These validity checks are always applied: finite lon/lat/refh, valid lon/lat,
    # finite refh_amp/refh_snr.
    filter_good_snr_only: bool = False
    refh_snr_min_for_input: Optional[float] = None
    track_range: Optional[Tuple[int, int]] = None
    sweep_range: Optional[Tuple[int, int]] = None

    # Signal-quality labeling. These DO NOT affect raw output; they label noise.
    use_snr_amp_noise_label: bool = True
    snr_hard_min: float = 1.5
    snr_soft_min: float = 1.8
    amp_low_percentile: float = 5.0
    # If True, label as noise when both snr < snr_soft_min and amp below percentile.
    # Also label as noise when snr < snr_hard_min regardless of amplitude.
    low_signal_requires_low_amp_for_soft_snr: bool = True

    # Per-pulse threshold field if present.
    use_refh_threshold_if_available: bool = True
    refh_threshold_margin_min: float = 0.0

    # Optional geolocation / refh error labeling if available.
    use_error_fields_if_available: bool = True
    max_refh_error_m: Optional[float] = None
    max_refh_horizontal_error_deg: Optional[float] = None

    # Optional global z percentile guard for catastrophic high/low returns.
    # Keep disabled for formal products; useful for quick visualization.
    use_global_z_percentile_guard: bool = False
    z_low_percentile: float = 0.05
    z_high_percentile: float = 99.95

    # Local height consistency labeling in projected coordinates.
    use_local_height_filter: bool = True
    local_grid_cell_size_m: float = 5.0
    local_min_points_per_cell: int = 12
    local_abs_residual_threshold_m: float = 25.0
    local_mad_multiplier: float = 8.0
    local_min_sigma_m: float = 0.75

    # Optional Open3D statistical outlier removal.
    # This can be slow/memory-heavy on millions of points. It is disabled by default.
    use_open3d_statistical_outlier: bool = False
    open3d_sor_nb_neighbors: int = 20
    open3d_sor_std_ratio: float = 2.5
    open3d_sor_max_points: int = 750_000
    open3d_sor_seed: int = 42
    # If False and points exceed max, skip SOR rather than sampling for final labeling.
    # Sampling-based SOR is useful for preview but not rigorous for final all-point labels.
    allow_sampled_sor_for_labeling: bool = False

    # Output controls.
    write_raw_las: bool = True
    write_noise_labeled_las: bool = True
    write_clean_las: bool = True
    write_noise_only_las: bool = False
    write_metadata_json: bool = True
    write_preview_png: bool = True

    # Visualization / preview controls.
    preview_max_points: int = 300_000
    preview_seed: int = 42
    rgb_color_by: str = "refh_amp"  # "refh_amp", "refh", "refh_snr", "classification"
    robust_color_percentiles: Tuple[float, float] = (2.0, 98.0)
    visualize_open3d: bool = False
    open3d_visual_max_points: int = 350_000
    open3d_vertical_exaggeration: float = 1.0
    open3d_color_mode: str = "classification"  # "classification", "amp", "snr", "height"

    # Behavior.
    overwrite: bool = True


# =============================================================================
# Constants / bit masks
# =============================================================================

# =============================================================================
# Generic utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)














def robust_min_max(values: np.ndarray, percentiles: Tuple[float, float]) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return 0.0, 1.0
    p_low, p_high = np.nanpercentile(vals[finite], percentiles)
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        vmin = float(np.nanmin(vals[finite]))
        vmax = float(np.nanmax(vals[finite]))
        return vmin, vmax if vmax > vmin else vmin + 1.0
    return float(p_low), float(p_high)


# =============================================================================
# Input and georeference
# =============================================================================

# =============================================================================
# Noise labeling
# =============================================================================



# =============================================================================
# LAS writing
# =============================================================================











# =============================================================================
# Preview plots and Open3D visualization
# =============================================================================

def sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=int(max_points), replace=False)


def make_debug_previews(
    out_dir: Path,
    stem: str,
    data: RefhPointData,
    proj: ProjectedRefh,
    filt: FilterResult,
    cfg: Config,
) -> dict[str, str]:
    ensure_dir(out_dir)
    idx = sample_indices(len(data.z_refh), cfg.preview_max_points, cfg.preview_seed)
    outputs: dict[str, str] = {}

    def save_current(name: str) -> None:
        path = out_dir / f"{stem}_{name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()
        outputs[name] = str(path)

    # Spatial classification preview.
    plt.figure(figsize=(12, 8))
    keep = filt.keep_mask[idx]
    plt.scatter(proj.easting[idx][keep], proj.northing[idx][keep], s=0.2, c="tab:blue", linewidths=0, alpha=0.45, label="kept")
    plt.scatter(proj.easting[idx][~keep], proj.northing[idx][~keep], s=0.35, c="tab:red", linewidths=0, alpha=0.55, label="likely noise")
    plt.axis("equal")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("CASALS refh noise labeling preview")
    plt.legend(markerscale=8)
    save_current("classification_noise_mask")

    # Raw amplitude spatial map.
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(proj.easting[idx], proj.northing[idx], c=data.refh_amp[idx], s=0.25, linewidths=0, cmap="viridis")
    plt.axis("equal")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("CASALS refh_amp spatial preview")
    plt.colorbar(sc, label="refh_amp")
    save_current("raw_refh_amp")

    # SNR map.
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(proj.easting[idx], proj.northing[idx], c=data.refh_snr[idx], s=0.25, linewidths=0, cmap="viridis")
    plt.axis("equal")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("CASALS refh_snr spatial preview")
    plt.colorbar(sc, label="refh_snr")
    save_current("snr_map")

    return outputs


def open3d_visualize(data: RefhPointData, proj: ProjectedRefh, filt: FilterResult, cfg: Config) -> None:
    if not cfg.visualize_open3d:
        return
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        print(f"Open3D visualization skipped: {type(exc).__name__}: {exc}")
        return

    n = len(data.z_refh)
    idx = sample_indices(n, cfg.open3d_visual_max_points, cfg.preview_seed)
    x0 = float(np.nanmedian(proj.easting[idx]))
    y0 = float(np.nanmedian(proj.northing[idx]))
    z0 = float(np.nanmedian(data.z_refh[idx]))
    pts = np.column_stack([
        proj.easting[idx] - x0,
        proj.northing[idx] - y0,
        (data.z_refh[idx] - z0) * float(cfg.open3d_vertical_exaggeration),
    ]).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    colors = np.zeros((len(idx), 3), dtype=np.float64)
    mode = cfg.open3d_color_mode
    if mode == "classification":
        colors[filt.keep_mask[idx]] = np.array([0.1, 0.45, 1.0])
        colors[filt.noise_mask[idx]] = np.array([1.0, 0.1, 0.02])
    else:
        if mode == "amp":
            vals = data.refh_amp[idx]
        elif mode == "snr":
            vals = data.refh_snr[idx]
        elif mode == "height":
            vals = data.z_refh[idx]
        else:
            vals = data.refh_amp[idx]
        vmin, vmax = robust_min_max(vals, cfg.robust_color_percentiles)
        norm = np.clip((vals - vmin) / (vmax - vmin), 0, 1)
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("Open3D visualization:")
    print(f"  sampled points: {len(idx):,} / {n:,}")
    print(f"  coordinates are centered at easting={x0:.3f}, northing={y0:.3f}, refh={z0:.3f}")
    print(f"  vertical exaggeration: {cfg.open3d_vertical_exaggeration}")
    print("  color mode:", mode)
    o3d.visualization.draw_geometries([pcd], window_name="CASALS refh filter view")


# =============================================================================
# Main workflow
# =============================================================================

def default_config(
    h5_path: Path,
    output_dir: Optional[Path] = None,
    point_cloud_dir: Optional[Path] = None,
) -> Config:
    """Build the established filter defaults for direct or CLI use."""
    output_dir = output_dir or Path("outputs/refh") / h5_path.stem / "filter"
    point_cloud_dir = point_cloud_dir or output_dir / "point_clouds"
    return Config(
        h5_path=h5_path,
        point_cloud_dir=point_cloud_dir,
        output_dir=output_dir,

        # Input-level filtering: keep broad for raw/reference product.
        filter_good_snr_only=True,
        refh_snr_min_for_input=None,
        track_range=None,
        sweep_range=None,

        # Tunable signal thresholds. Start conservative; adjust and rerun.
        use_snr_amp_noise_label=True,
        snr_hard_min=1.35,
        snr_soft_min=1.80,
        amp_low_percentile=5.0,
        low_signal_requires_low_amp_for_soft_snr=True,

        # Use per-pulse threshold if H5 provides it.
        use_refh_threshold_if_available=True,
        refh_threshold_margin_min=0.0,

        # Use optional error fields only if you set thresholds.
        use_error_fields_if_available=True,
        max_refh_error_m=None,
        max_refh_horizontal_error_deg=None,

        # Useful for quick visualization, but keep False for formal raw products.
        use_global_z_percentile_guard=False,
        z_low_percentile=0.05,
        z_high_percentile=99.95,

        # Main spatial consistency filter.
        use_local_height_filter=True,
        local_grid_cell_size_m=5.0,
        local_min_points_per_cell=12,
        local_abs_residual_threshold_m=25.0,
        local_mad_multiplier=8.0,
        local_min_sigma_m=0.75,

        # Optional Open3D SOR final labeling. Disabled by default for 3.6M points.
        use_open3d_statistical_outlier=False,
        open3d_sor_nb_neighbors=20,
        open3d_sor_std_ratio=2.5,
        open3d_sor_max_points=750_000,
        allow_sampled_sor_for_labeling=False,

        # Outputs.
        write_raw_las=True,
        write_noise_labeled_las=True,
        write_clean_las=True,
        write_noise_only_las=False,
        write_metadata_json=True,
        write_preview_png=True,

        # Interactive visualization. Turn on after first successful run.
        visualize_open3d=False,
        open3d_visual_max_points=350_000,
        open3d_vertical_exaggeration=1.0,
        open3d_color_mode="classification",

        rgb_color_by="classification",
        preview_max_points=300_000,
        overwrite=True,
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True, help="CASALS L1B input H5 file")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: outputs/refh/<h5-stem>/filter).")
    parser.add_argument(
        "--point-cloud-dir",
        type=Path,
        help="Directory for LAS output (default: <output-dir>/point_clouds)",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON object of Config fields")
    args = parser.parse_args(argv)
    # -------------------------------------------------------------------------
    # Apply only the explicit JSON overrides to the shared defaults.
    # -------------------------------------------------------------------------
    cfg = default_config(args.h5, args.output_dir, args.point_cloud_dir)
    if args.config:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
        cfg = replace(
            cfg,
            **{
                key: value
                for key, value in overrides.items()
                if key not in {"h5_path", "point_cloud_dir", "output_dir"}
            },
        )

    filter_refh(cfg)


def filter_refh(cfg: Config) -> Dict[str, Any]:
    t0 = time.time()
    ensure_dir(cfg.point_cloud_dir)
    ensure_dir(cfg.output_dir)
    if not cfg.h5_path.exists():
        raise FileNotFoundError(f"H5 file does not exist: {cfg.h5_path}")

    print("=" * 80)
    print("CASALS L1B refh filtering and noise labeling")
    print("=" * 80)
    print(f"H5: {cfg.h5_path.resolve()}")
    print(f"Point-cloud directory: {cfg.point_cloud_dir.resolve()}")
    print(f"Output directory: {cfg.output_dir.resolve()}")
    print()

    print("Reading L1B reference-return fields...")
    data = read_refh_points(
        cfg.h5_path,
        selection=RefhSelection(
            filter_good_snr_only=cfg.filter_good_snr_only,
            refh_snr_min=cfg.refh_snr_min_for_input,
            track_range=cfg.track_range,
            sweep_range=cfg.sweep_range,
        ),
    )
    print(f"  input records: {data.n_input_records:,}")
    print(f"  valid records after input mask: {data.n_valid_records:,}")
    print(f"  start UTC: {data.attrs.get('start_utca', 'unknown')}")
    print(f"  end UTC:   {data.attrs.get('end_utca', 'unknown')}")
    print(f"  good_snr fraction valid: {data.input_mask_summary['good_snr_fraction_valid']:.6f}")
    print()

    print("Projecting lon/lat to UTM...")
    proj = project_refh_points(data, cfg.utm_epsg_override)
    print(f"  output CRS: EPSG:{proj.utm_epsg}, {proj.utm_crs_name}")
    print(f"  inverse projection max horizontal error approx: {proj.projection_check['approx_max_horizontal_error_m']:.3e} m")
    print()

    print("Labeling likely noise with current parameters...")
    filt = label_noise(data, proj, cfg)
    print(json.dumps(filt.counts, indent=2))
    print()

    print("Coordinate and attribute summaries:")
    summaries = [
        summarize_refh_array("longitude_deg", data.lon),
        summarize_refh_array("latitude_deg", data.lat),
        summarize_refh_array("easting_m", proj.easting),
        summarize_refh_array("northing_m", proj.northing),
        summarize_refh_array("refh_ellipsoidal_height_m", data.z_refh),
        summarize_refh_array("refh_amp", data.refh_amp),
        summarize_refh_array("refh_snr", data.refh_snr),
        summarize_refh_array("track_num", data.track_num),
        summarize_refh_array("sweep_num", data.sweep_num),
    ]
    for s in summaries:
        print(json.dumps(s, indent=2))
    print()

    stem = cfg.h5_path.stem
    suffix = f"epsg{proj.utm_epsg}"
    outputs: dict[str, Any] = {}

    all_mask = np.ones(len(data.z_refh), dtype=bool)
    raw_cls = np.ones(len(data.z_refh), dtype=np.uint8)
    labeled_cls = np.where(filt.noise_mask, 7, 1).astype(np.uint8)
    clean_mask = filt.keep_mask
    noise_only_mask = filt.noise_mask

    if cfg.write_raw_las:
        path = cfg.point_cloud_dir / f"{stem}_raw_refh_{suffix}.las"
        print(f"Writing raw LAS: {path}")
        outputs["raw_las"] = write_refh_las(
            path, data, proj, mask=all_mask, classification=raw_cls,
            rgb_color_by="refh_amp", color_percentiles=cfg.robust_color_percentiles,
            xyz_scale_m=cfg.las_xyz_scale_m, generating_software="filter_refh_points.py",
            overwrite=cfg.overwrite,
        )

    if cfg.write_noise_labeled_las:
        path = cfg.point_cloud_dir / f"{stem}_noise_labeled_refh_{suffix}.las"
        print(f"Writing noise-labeled LAS: {path}")
        outputs["noise_labeled_las"] = write_refh_las(
            path, data, proj, mask=all_mask, classification=labeled_cls, noise=filt,
            rgb_color_by="classification", color_percentiles=cfg.robust_color_percentiles,
            xyz_scale_m=cfg.las_xyz_scale_m, generating_software="filter_refh_points.py",
            overwrite=cfg.overwrite,
        )

    if cfg.write_clean_las:
        path = cfg.point_cloud_dir / f"{stem}_clean_refh_{suffix}.las"
        print(f"Writing clean LAS: {path}")
        outputs["clean_las"] = write_refh_las(
            path, data, proj, mask=clean_mask, classification=labeled_cls, noise=filt,
            rgb_color_by="refh_amp", color_percentiles=cfg.robust_color_percentiles,
            xyz_scale_m=cfg.las_xyz_scale_m, generating_software="filter_refh_points.py",
            overwrite=cfg.overwrite,
        )

    if cfg.write_noise_only_las:
        path = cfg.point_cloud_dir / f"{stem}_noise_only_refh_{suffix}.las"
        print(f"Writing noise-only LAS: {path}")
        outputs["noise_only_las"] = write_refh_las(
            path, data, proj, mask=noise_only_mask, classification=labeled_cls, noise=filt,
            rgb_color_by="classification", color_percentiles=cfg.robust_color_percentiles,
            xyz_scale_m=cfg.las_xyz_scale_m, generating_software="filter_refh_points.py",
            overwrite=cfg.overwrite,
        )

    preview_outputs: dict[str, str] = {}
    if cfg.write_preview_png:
        print("Writing preview figures...")
        preview_outputs = make_debug_previews(cfg.output_dir, stem, data, proj, filt, cfg)
        outputs["preview_pngs"] = preview_outputs

    metadata = {
        "script": "filter_refh_points.py",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_h5": str(cfg.h5_path.resolve()),
        "casals_product_level": "L1B",
        "point_cloud_level": "Level-A max-Rx-bin / refh reference-return point cloud",
        "scientific_notes": [
            "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
            "refh is WGS84 ellipsoidal height unless otherwise documented.",
            "This is not an official multi-return point cloud.",
            "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            "Waveform peak detection beyond the official refh point is experimental or diagnostic only.",
        ],
        "method_scope": {
            "x_source": "refh_longitude projected from EPSG:4326 to inferred/overridden UTM",
            "y_source": "refh_latitude projected from EPSG:4326 to inferred/overridden UTM",
            "z_source": "refh, WGS84 ellipsoidal height",
            "waveform_decomposition": "not applied",
            "ground_classification": "not applied",
            "vertical_datum_conversion": "not applied",
            "noise_handling": "likely noise labeled as LAS class 7; raw points retained in raw output",
        },
        "config": asdict(cfg),
        "source_global_attributes_subset": {
            k: data.attrs.get(k)
            for k in (
                "tdms_file", "l1a_file", "ard_file", "geoloc_file",
                "start_utca", "end_utca", "n_pulses", "n_sweeps", "n_tracks", "n_rx_bins", "n_tx_bins",
            )
        },
        "input_mask_summary": data.input_mask_summary,
        "crs": {
            "input_horizontal_crs": "EPSG:4326 WGS84 geographic",
            "output_horizontal_crs_epsg": int(proj.utm_epsg),
            "output_horizontal_crs_name": proj.utm_crs_name,
            "z_height_convention": "WGS84 ellipsoidal height from CASALS refh",
        },
        "projection_validation": proj.projection_check,
        "filter_thresholds": filt.thresholds,
        "filter_counts": filt.counts,
        "noise_reason_count": filt.counts.get("counts_by_reason", {}),
        "summaries": summaries,
        "outputs": outputs,
        "runtime_seconds": float(time.time() - t0),
    }
    metadata_path = cfg.output_dir / f"{stem}_filter_metadata.json"
    if cfg.write_metadata_json:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        print(f"Metadata written: {metadata_path}")

    print()
    print("=" * 80)
    print("Done.")
    print("=" * 80)
    print(f"Keep points:  {filt.counts['n_keep']:,}")
    print(f"Noise points: {filt.counts['n_noise_labeled']:,}")
    print(f"Metadata: {metadata_path}")
    print("Reminder: Z is CASALS refh WGS84 ellipsoidal height; this is not a DEM or ground-classified product.")

    open3d_visualize(data, proj, filt, cfg)

    return {
        "metadata_path": metadata_path,
        "outputs": outputs,
        "counts": filt.counts,
        "metadata": metadata,
    }


if __name__ == "__main__":
    main()
