"""Small fixed-point NGS HTDP cross-check for the frame/epoch assessment."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

import h5py
import numpy as np
import pandas as pd
from pyproj import CRS

from research.reference import vertical_datum_comparison as datum
from research.reference.frame_hypotheses import (
    apply_frame, observation_epoch, xyz_checkpoint,
)

HTDP_DOWNLOAD_URL = "https://geodesy.noaa.gov/TOOLS/Htdp/HTDP-download.zip"
HTDP_USER_GUIDE_URL = "https://geodesy.noaa.gov/TOOLS/Htdp/HTDP-user-guide.pdf"
HTDP_ZIP_SHA256 = "cbc3bd587dd6132f57dee4d9d01e5ca171f86ddb96f8f11b9242260fbe691474"
HTDP_EXE_SHA256 = "15dbe2f4f01aae1cd63dd4013d3aea170d6249bf9083aa0c1d5858dcfbcc2054"
HTDP_FRAME_KEY = {10605: 12, 7912: 25, 9989: 26}
HTDP_FRAME_NAME = {10605: "WGS84_G2296", 7912: "ITRF2014", 9989: "ITRF2020"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_point_output(text: str, point_name: str) -> tuple[float, float, float]:
    """Parse one named LAT,LON-w-positive,EHT row from the official text output."""
    matches = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[-1] == point_name:
            try:
                matches.append(tuple(float(v) for v in fields[:3]))
            except ValueError:
                continue
    if len(matches) != 1:
        raise ValueError(f"Expected one HTDP output row for {point_name}, found {len(matches)}")
    return matches[0]


def _run_htdp(executable: Path, output_dir: Path, input_path: Path, *,
              input_frame_key: int, input_epoch: float, output_epoch: float,
              point_name: str, run_name: str) -> tuple[tuple[float, float, float], str, str]:
    output_name = f"{run_name}.out"
    answers = [
        "4", output_name, str(input_frame_key), "1",
        "2", format(float(input_epoch), ".12f"),
        "2", format(float(output_epoch), ".12f"),
        "3", input_path.name, "0",
    ]
    (output_dir / f"{run_name}.menu.txt").write_text("\n".join(answers) + "\n", encoding="ascii")
    completed = subprocess.run(
        [str(executable)], input="\n".join(answers) + "\n", text=True,
        capture_output=True, cwd=output_dir, timeout=120,
    )
    (output_dir / f"{run_name}.console.txt").write_text(
        completed.stdout + ("\n--- STDERR ---\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    result_path = output_dir / output_name
    if completed.returncode != 0 or "ABNORMAL TERMINATION" in completed.stdout or not result_path.is_file():
        raise RuntimeError(f"HTDP failed for {run_name}; see {run_name}.console.txt")
    result_text = result_path.read_text(encoding="utf-8", errors="replace")
    return _parse_point_output(result_text, point_name), result_text, output_name


def run_htdp_checkpoints(R: dict, output: str | Path, executable: str | Path | None) -> dict:
    """Compare PROJ with official HTDP at two real H5 pulse-0 points.

    Three source frames are compared at identical observation input/output epochs.
    HTDP's ITRF2014 case is separately propagated to NAD83(2011) epoch 2010.0.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if executable is None or not Path(executable).is_file():
        status = {
            "status": "unavailable",
            "reason": "Set CASALS_HTDP_EXE to the official HTDP 3.6.0 executable to run the fixed-point check.",
            "download_url": HTDP_DOWNLOAD_URL,
            "user_guide_url": HTDP_USER_GUIDE_URL,
            "fixed_inputs": "H5 pulse_index=0 for each current granule; no 3DEP values are read.",
        }
        (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return {"status": status, "same_epoch": pd.DataFrame(),
                "epoch_propagation": pd.DataFrame(), "points_2010": pd.DataFrame(), "output": output}

    executable = Path(executable).resolve()
    same_epoch_rows, propagation_rows, points_2010, fixed_inputs = [], [], [], []
    h5_hashes = {}
    manifest_path = Path(R["output"]) / "input_sha256.csv"
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        for spec in datum.GRANULES:
            row = manifest.loc[manifest.path.eq(spec.h5_path.relative_to(datum.PROJECT_ROOT).as_posix())]
            if len(row):
                h5_hashes[spec.granule_id] = str(row.iloc[0].sha256)

    version_probe = subprocess.run([str(executable)], input="", text=True, capture_output=True,
                                   cwd=output, timeout=30)
    version_text = version_probe.stdout
    version_match = re.search(r"SOFTWARE VERSION:\s*([^\s]+).*?VERSION DATE:\s*(.+)", version_text, re.S)
    if not version_match:
        raise RuntimeError("Could not identify the HTDP executable version from its startup banner")
    htdp_version, htdp_date = version_match.group(1), version_match.group(2).splitlines()[0].strip()
    exe_hash = _sha256(executable)

    for spec in datum.GRANULES:
        with h5py.File(spec.h5_path, "r") as h5:
            pulse_index = 0
            lon, lat, height = (float(h5[name][pulse_index]) for name in
                                ("refh_longitude", "refh_latitude", "refh"))
        if not np.isfinite([lon, lat, height]).all():
            raise ValueError(f"H5 pulse 0 is not a finite fixed HTDP point in {spec.granule_id}")
        epoch = float(observation_epoch(spec.granule_id))
        point_name = f"{spec.granule_id[11:19]}_pulse0"
        input_name = f"{spec.granule_id[11:19]}_pulse0_input.csv"
        input_path = output / input_name
        # HTDP delimited files use longitude positive west; CASALS stores it positive east.
        input_path.write_text(f"{lat:.15g},{-lon:.15g},{height:.15g},{point_name}\n", encoding="ascii")
        fixed_inputs.append({
            "granule_id": spec.granule_id, "pulse_index": pulse_index,
            "point_name": point_name, "longitude_east_deg": lon,
            "longitude_positive_west_deg_for_HTDP": -lon, "latitude_deg": lat,
            "ellipsoidal_height_m": height, "input_coordinate_epoch_assumption": epoch,
            "source_h5": str(spec.h5_path.relative_to(datum.PROJECT_ROOT)),
            "source_h5_sha256": h5_hashes.get(spec.granule_id),
            "fixed_input_sha256": _sha256(input_path),
        })
        for epsg, frame_key in HTDP_FRAME_KEY.items():
            proj_lon, proj_lat, proj_h, proj_audit = apply_frame(
                epsg, epoch, [lon], [lat], [height], coordinate_time=epoch)
            run_name = f"{spec.granule_id[11:19]}_{HTDP_FRAME_NAME[epsg]}_same_epoch"
            htdp_row, htdp_text, output_name = _run_htdp(
                executable, output, input_path, input_frame_key=frame_key,
                input_epoch=epoch, output_epoch=epoch, point_name=point_name, run_name=run_name)
            htdp_lat, htdp_lon_west, htdp_h = htdp_row
            htdp_lon = -htdp_lon_west
            geod = CRS.from_epsg(4326).get_geod()
            _, _, horizontal = geod.inv(float(proj_lon[0]), float(proj_lat[0]), htdp_lon, htdp_lat)
            same_epoch_rows.append({
                "granule_id": spec.granule_id, "pulse_index": pulse_index,
                "source_frame": HTDP_FRAME_NAME[epsg], "source_epsg": epsg,
                "input_epoch": epoch, "output_epoch": epoch,
                "PROJ_lon_east_deg": float(proj_lon[0]), "PROJ_lat_deg": float(proj_lat[0]),
                "PROJ_h_m": float(proj_h[0]), "HTDP_lon_east_deg": htdp_lon,
                "HTDP_lat_deg": htdp_lat, "HTDP_h_m_rounded_to_mm": htdp_h,
                "PROJ_minus_HTDP_horizontal_m": float(horizontal),
                "PROJ_minus_HTDP_height_m": float(proj_h[0] - htdp_h),
                "PROJ_output_coordinate_time": float(proj_audit["coordinate_time_out"]),
                "HTDP_output_file": output_name,
                "HTDP_header": next((line.strip() for line in htdp_text.splitlines()
                                     if line.strip().startswith("TO NAD_83")), None),
            })
            if epsg == 7912:
                xyz = xyz_checkpoint(epsg, lon, lat, height, epoch)
                _, _, xyz_distance = geod.inv(float(proj_lon[0]), float(proj_lat[0]),
                                               xyz["output_geographic_lon_lat_h"][0],
                                               xyz["output_geographic_lon_lat_h"][1])
                xyz_dh = float(proj_h[0] - xyz["output_geographic_lon_lat_h"][2])
                if abs(float(xyz_distance)) > 1e-5 or abs(xyz_dh) > 1e-5:
                    raise RuntimeError("Independent geocentric XYZ checkpoint disagrees with 3D PROJ transform")
                (output / f"{spec.granule_id[11:19]}_ITRF2014_xyz_checkpoint.json").write_text(
                    json.dumps(xyz, indent=2), encoding="utf-8")
                propagated_name = f"{spec.granule_id[11:19]}_ITRF2014_to_2010"
                propagated_row, propagated_text, propagated_file = _run_htdp(
                    executable, output, input_path, input_frame_key=frame_key,
                    input_epoch=epoch, output_epoch=2010.0, point_name=point_name,
                    run_name=propagated_name)
                target_lat, target_lon_west, target_h = propagated_row
                target_lon = -target_lon_west
                _, _, propagation_horizontal = geod.inv(htdp_lon, htdp_lat, target_lon, target_lat)
                propagation_rows.append({
                    "granule_id": spec.granule_id, "pulse_index": pulse_index,
                    "source_frame": "ITRF2014 assumed", "source_coordinate_epoch_assumption": epoch,
                    "frame_output": "NAD83(2011)", "target_coordinate_epoch": 2010.0,
                    "same_epoch_NAD83_lon_east_deg": htdp_lon,
                    "same_epoch_NAD83_lat_deg": htdp_lat, "same_epoch_NAD83_h_m": htdp_h,
                    "epoch_2010_NAD83_lon_east_deg": target_lon,
                    "epoch_2010_NAD83_lat_deg": target_lat, "epoch_2010_NAD83_h_m_rounded_to_mm": target_h,
                    "B_to_C_horizontal_displacement_m": float(propagation_horizontal),
                    "B_to_C_ellipsoidal_height_change_m": float(target_h - htdp_h),
                    "HTDP_output_file": propagated_file,
                    "HTDP_header": next((line.strip() for line in propagated_text.splitlines()
                                         if line.strip().startswith("TO NAD_83")), None),
                    "velocity_model": "NGS HTDP 3.6.0 model; conditional on the ITRF2014 source-frame assumption",
                })
                points_2010.append({
                    "granule_id": spec.granule_id, "pulse_index": pulse_index,
                    "longitude_deg": target_lon, "latitude_deg": target_lat,
                    "ellipsoidal_height_m": target_h, "source_epsg_assumption": 7912,
                    "coordinate_epoch": 2010.0,
                    "input_coordinate_epoch_assumption": epoch,
                })

    same = pd.DataFrame(same_epoch_rows)
    propagation = pd.DataFrame(propagation_rows)
    points = pd.DataFrame(points_2010)
    same.to_csv(output / "same_epoch_comparison.csv", index=False)
    propagation.to_csv(output / "epoch_2010_propagation.csv", index=False)
    points.to_csv(output / "nad83_2011_epoch_2010_checkpoints.csv", index=False)
    (output / "fixed_input_manifest.json").write_text(json.dumps(fixed_inputs, indent=2), encoding="utf-8")
    status = {
        "status": "completed", "tool": "NGS HTDP", "version": htdp_version,
        "version_date": htdp_date, "executable": str(executable), "executable_sha256": exe_hash,
        "official_download_url": HTDP_DOWNLOAD_URL, "official_user_guide_url": HTDP_USER_GUIDE_URL,
        "official_zip_sha256_if_unmodified": HTDP_ZIP_SHA256,
        "same_epoch_frames": ["WGS84(G2296)", "ITRF2014", "ITRF2020"],
        "epoch_propagation": "ITRF2014 assumed at observation epoch -> NAD83(2011), 2010.0; HTDP model applied",
        "source_frame_identity_confirmed_for_CASALS": False,
        "all_pulses_propagated": False,
        "interpretation": "Fixed-point numerical verification only; source realization and coordinate epoch remain conditional.",
    }
    (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return {"status": status, "same_epoch": same, "epoch_propagation": propagation,
            "points_2010": points, "output": output}
