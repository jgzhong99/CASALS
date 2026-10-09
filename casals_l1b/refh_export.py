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
import warnings
import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pyproj import CRS

import matplotlib.pyplot as plt

from .refh import (
    RefhSelection,
    project_refh_points,
    read_refh_points,
    write_refh_las,
    summarize_refh_array,
)


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






# =============================================================================
# Main workflow
# =============================================================================

def default_config(
    h5_path: Path,
    output_dir: Optional[Path] = None,
    point_cloud_dir: Optional[Path] = None,
) -> Config:
    """Build the established export defaults for direct or CLI use."""
    output_dir = output_dir or Path("outputs/refh") / h5_path.stem / "export"
    point_cloud_dir = point_cloud_dir or output_dir / "point_clouds"
    return Config(
        h5_path=h5_path,
        point_cloud_dir=point_cloud_dir,
        output_dir=output_dir,
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


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True, help="CASALS L1B input H5 file")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: outputs/refh/<h5-stem>/export).")
    parser.add_argument(
        "--point-cloud-dir",
        type=Path,
        help="Directory for LAS output (default: <output-dir>/point_clouds)",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON object of Config fields")
    args = parser.parse_args(argv)
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

    export_refh(cfg)


def export_refh(cfg: Config) -> Dict[str, Any]:
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

    print("Reading required 1D georeference/reference-return datasets...")
    data = read_refh_points(
        cfg.h5_path,
        selection=RefhSelection(
            filter_good_snr_only=cfg.filter_good_snr_only,
            refh_snr_min=cfg.refh_snr_min,
            track_range=cfg.track_range,
            sweep_range=cfg.sweep_range,
        ),
        optional_fields=("delta_time",),
        ignore_bad_optional_fields=False,
        ignore_optional_size_mismatch=False,
    )
    attrs = data.attrs
    n_pulses = data.n_input_records
    n_valid = data.n_valid_records
    lon_f, lat_f, z_f = data.lon, data.lat, data.z_refh
    amp_f, snr_f = data.refh_amp, data.refh_snr
    good_f, track_f, sweep_f = data.good_snr, data.track_num, data.sweep_num
    print(f"Total pulse/reference records: {n_pulses:,}")
    print(f"File start UTC: {attrs.get('start_utca', 'unknown')}")
    print(f"File end UTC:   {attrs.get('end_utca', 'unknown')}")
    print("Filtering summary:")
    print(f"  finite + valid lon/lat/refh records: {n_valid:,} / {n_pulses:,}")
    print(f"  filter_good_snr_only: {cfg.filter_good_snr_only}")
    print(f"  refh_snr_min: {cfg.refh_snr_min}")
    print(f"  track_range: {cfg.track_range}")
    print(f"  sweep_range: {cfg.sweep_range}")
    print(f"  good_snr fraction in full file: {data.input_mask_summary['good_snr_fraction_input']:.6f}")
    print(f"  good_snr fraction after filtering: {data.input_mask_summary['good_snr_fraction_valid']:.6f}")
    print()

    projected = project_refh_points(data, seed=cfg.random_seed)
    utm_epsg = projected.utm_epsg
    utm_crs = CRS.from_epsg(utm_epsg)
    easting, northing = projected.easting, projected.northing
    projection_check = projected.projection_check

    print("Coordinate reference system:")
    print("  Input horizontal CRS: EPSG:4326, WGS84 geographic lon/lat")
    print(f"  Output horizontal CRS: EPSG:{utm_epsg}, {utm_crs.name}")
    print("  Output Z: CASALS refh, WGS84 ellipsoidal height; no vertical datum conversion")
    print()

    print("Projection validation:")
    for key, value in projection_check.items():
        print(f"  {key}: {value}")
    print()

    print("Coordinate and attribute summaries after filtering:")
    for summary in [
        summarize_refh_array("longitude_deg",             lon_f),
        summarize_refh_array("latitude_deg",              lat_f),
        summarize_refh_array("easting_m",                 easting),
        summarize_refh_array("northing_m",                northing),
        summarize_refh_array("refh_ellipsoidal_height_m", z_f),
        summarize_refh_array("refh_amp",                  amp_f),
        summarize_refh_array("refh_snr",                  snr_f),
        summarize_refh_array("track_num",                 track_f),
        summarize_refh_array("sweep_num",                 sweep_f),
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
        written = write_refh_las(
            las_path,
            data,
            projected,
            rgb_color_by=cfg.rgb_color_by,
            color_percentiles=cfg.robust_color_percentiles,
            xyz_scale_m=cfg.las_xyz_scale_m,
            include_optional_fields=True,
            generating_software="export_refh_las.py",
            track_num_description="track_channel_number",
            pulse_index_description="zero_based_pulse_index",
            optional_descriptions={"delta_time": "delta_time_sec_2018"},
        )
        las_info = {
            "las_path": written["path"],
            "n_points_written": written["n_points"],
            "las_horizontal_crs_epsg": written["horizontal_crs_epsg"],
            "las_horizontal_crs_name": utm_crs.name,
            "las_z_height_convention": written["z_convention"],
            "las_xyz_scale_m": written["xyz_scale_m"],
            "las_offsets": written["offsets"],
            "las_rgb_color_by": written["rgb_color_by"],
            "las_rgb_robust_range": written["rgb_range"],
            "las_classification": "1 = unclassified for all points",
            "las_intensity": "refh_amp clipped to uint16 [0, 65535]; raw refh_amp preserved in extra dimension refh_amp_raw",
        }
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
            "good_snr_fraction_input": data.input_mask_summary["good_snr_fraction_input"],
            "good_snr_fraction_output": data.input_mask_summary["good_snr_fraction_valid"],
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

    return {
        "las_path": las_path if cfg.write_las else None,
        "metadata_path": metadata_path if cfg.write_metadata_json else None,
        "preview_path": preview_path if cfg.write_preview_png else None,
    }


if __name__ == "__main__":
    main()
