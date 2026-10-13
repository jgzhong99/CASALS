"""Read-only CASALS L1B correction audit for the two validation granules."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer, datadir
from pyproj.transformer import TransformerGroup

from research.geolocation.bin_georeferencing_audit.run_audit import (
    beam,
    inventory as h5_inventory,
    read_values,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/notebooks/10_3dep_elevation_validation/correction_audit"
H5_DIR = ROOT / "data/raw/casals_l1b"
GRANULES = (
    "casals_l1b_20241112T165718_001_02",
    "casals_l1b_20241118T171757_001_02",
)
H5_SUFFIX = ".h5"
NGA_ZIP = OUT / "reference_data/egm2008_2p5min_nga.zip"
NGA_SHA256 = "0f65f16e6fd3f89a6b8022d7a89375d0c29fb275a551927175669bb610904cd0"
NGA_MEMBER = "Und_min2.5x2.5_egm2008_isw=82_WGS84_TideFree_SE"
NGA_ROWS, NGA_COLS, NGA_STEP_DEG = 4321, 8640, 2.5 / 60
TARGET_COMMON = OUT.parent / "ground_aware/ground_c0_c7_common_sample.parquet"
TARGET_TEST = OUT.parent / "ground_aware/ground_target_test_population.parquet"
GEOID18_SAMPLE = OUT.parent / "datum_comparison/geoid18_sample_comparison.csv"
PROJ_DATA = OUT.parent / "datum_comparison/proj_data"
VALID_APPLIED_STATUS = {
    "CONFIRMED_APPLIED", "CONFIRMED_NOT_APPLIED", "NUMERICALLY_SUPPORTED",
    "UNKNOWN", "NOT_APPLICABLE",
}
VALID_CANDIDATE_STATUS = {
    "VERIFIED", "DOCUMENTED_ALTERNATIVE", "PHYSICAL_SENSITIVITY_ONLY",
    "UNAVAILABLE", "REJECTED",
}

CORRECTION_FIELDS = (
    "refh_neutat_delay_total", "refh_neutat_delay_derivative",
    "rwstart_neutat_delay_total", "rwstart_neutat_delay_derivative",
    "rwstop_neutat_delay_total", "rwstop_neutat_delay_derivative",
    "range_bias_correction", "geoid", "geoid_free2mean", "tide_earth",
    "tide_earth_free2mean", "tide_load", "tide_pole", "tide_ocean",
    "tide_ocean_pole", "dac",
)
METADATA_FIELDS = (
    "refh", "refh_error", "refh_snr", "refh_longitude", "refh_latitude", "instrument_longitude",
    "instrument_latitude", "instrument_altitude", "local_beam_azimuth",
    "local_beam_elevation", "rwstart", "rwstop", "rwstart_longitude",
    "rwstart_latitude", "rwstop_longitude", "rwstop_latitude",
    "rwstart_bounce_time_offset", "rwstop_bounce_time_offset",
    "refh_bounce_time_offset", "bin_size", "delta_time", "track_num",
    "sweep_num",
)
AUDIT_FIELDS = tuple(dict.fromkeys(CORRECTION_FIELDS + METADATA_FIELDS))


def json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: Any) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float).ravel()
    valid = x[np.isfinite(x)]
    result: dict[str, float | int] = {
        "n_total": int(x.size), "n_valid": int(valid.size),
        "n_missing": int(x.size - valid.size),
        "valid_fraction": float(valid.size / x.size) if x.size else np.nan,
    }
    if valid.size:
        median = float(np.median(valid))
        result.update(
            minimum=float(np.min(valid)), maximum=float(np.max(valid)),
            mean=float(np.mean(valid)), standard_deviation=float(np.std(valid)),
            median=median,
            NMAD=float(1.4826 * np.median(np.abs(valid - median))),
            P01=float(np.percentile(valid, 1)), P05=float(np.percentile(valid, 5)),
            P95=float(np.percentile(valid, 95)), P99=float(np.percentile(valid, 99)),
        )
    else:
        result.update({key: np.nan for key in (
            "minimum", "maximum", "mean", "standard_deviation", "median", "NMAD",
            "P01", "P05", "P95", "P99",
        )})
    return result


def collect_values(dataset: h5py.Dataset, chunk_size: int = 262_144) -> np.ndarray:
    """Read a 1D HDF5 field in bounded slices using the existing audit contract."""
    if dataset.ndim != 1:
        raise ValueError(f"expected one-dimensional dataset, got {dataset.shape}")
    result = np.empty(dataset.shape[0], dtype=float)
    for start in range(0, len(result), chunk_size):
        stop = min(start + chunk_size, len(result))
        result[start:stop] = read_values(dataset, slice(start, stop))
    return result


def field_inventory(inventory: dict, paths: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for name in AUDIT_FIELDS:
        path = paths.get(name)
        item = inventory["objects"].get("/" + path, {}) if path else {}
        attrs = item.get("attributes", {})
        rows.append({
            "field": name, "path": "/" + path if path else "NOT_PRESENT",
            "presence_status": "PRESENT" if path else "NOT_PRESENT",
            "shape": item.get("shape"), "dtype": item.get("dtype"),
            "chunks": item.get("chunks"), "storage_fillvalue": item.get("hdf5_storage_fillvalue"),
            "units_attribute": attrs.get("units", "UNKNOWN"),
            "attributes_json": json.dumps(attrs, sort_keys=True, default=json_value),
            "declared_missing_values": json.dumps(
                {k: attrs[k] for k in ("_FillValue", "missing_value") if k in attrs},
                default=json_value,
            ),
        })
    return rows


def up_normal(lon_deg: Any, lat_deg: Any) -> np.ndarray:
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    return np.column_stack((np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon), np.sin(lat)))


def height_sensitivity(
    excess_range: Any, lon_deg: Any, lat_deg: Any, azimuth: Any, elevation: Any,
) -> np.ndarray:
    """Height change for removing a positive sensor-to-target excess range.

    Units and application are intentionally not inferred. ``excess_range`` must
    be interpreted in the same linear-distance unit as the desired output.
    """
    delta = np.asarray(excess_range, dtype=float)
    direction = beam(np.asarray(lon_deg), np.asarray(lat_deg),
                     np.asarray(azimuth), np.asarray(elevation))
    up = up_normal(lon_deg, lat_deg)
    return -delta * np.sum(direction * up, axis=1)


def bilinear_grid(grid: np.ndarray, lon_deg: Any, lat_deg: Any,
                  spacing_deg: float = NGA_STEP_DEG) -> np.ndarray:
    """Bilinear lookup on a north-to-south, west-to-east global grid."""
    values = np.asarray(grid)
    lon, lat = np.broadcast_arrays(np.asarray(lon_deg, float), np.asarray(lat_deg, float))
    if values.ndim != 2 or spacing_deg <= 0:
        raise ValueError("grid must be 2D and spacing positive")
    row = (90.0 - lat) / spacing_deg
    col = np.mod(lon, 360.0) / spacing_deg
    valid = np.isfinite(row) & np.isfinite(col) & (row >= 0) & (row <= values.shape[0] - 1)
    r0 = np.clip(np.floor(row).astype(int), 0, values.shape[0] - 1)
    r1 = np.minimum(r0 + 1, values.shape[0] - 1)
    c0 = np.floor(col).astype(int) % values.shape[1]
    c1 = (c0 + 1) % values.shape[1]
    fr, fc = row - r0, col - np.floor(col)
    v00, v01 = values[r0, c0], values[r0, c1]
    v10, v11 = values[r1, c0], values[r1, c1]
    out = ((1 - fr) * ((1 - fc) * v00 + fc * v01)
           + fr * ((1 - fc) * v10 + fc * v11))
    return np.where(valid, out, np.nan)


def tide_free_to_mean_formula(latitude_deg: Any) -> np.ndarray:
    """Numerical formula used for the H5 comparison; not a provenance claim."""
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=float))
    return 0.1287 - 0.3848 * np.sin(latitude) ** 2


def geoid_model_comparison(h5_geoid: Any, egm2008: Any,
                           geoid18: Any) -> dict[str, dict[str, float | int]]:
    fields = [np.asarray(x, dtype=float) for x in (h5_geoid, egm2008, geoid18)]
    if not (fields[0].shape == fields[1].shape == fields[2].shape):
        raise ValueError("H5, EGM2008, and GEOID18 comparisons require common coordinates")
    return {
        "H5 minus EGM2008": distribution(fields[0] - fields[1]),
        "H5 minus GEOID18": distribution(fields[0] - fields[2]),
        "EGM2008 minus GEOID18": distribution(fields[1] - fields[2]),
    }


def select_independent_common_test(population: pd.DataFrame) -> pd.DataFrame:
    """Freeze existing independent-test rows with common candidate support."""
    if "research_role" not in population or "common_support_all_candidates" not in population:
        raise ValueError("population lacks frozen role/common-support columns")
    role = population["research_role"].astype(str)
    common = population["common_support_all_candidates"].fillna(False).astype(bool)
    selected = population.loc[(role == "independent_test") & common].copy()
    if selected["pulse_index"].duplicated().any():
        raise ValueError("independent-test pulse IDs are not unique")
    return selected.sort_values("pulse_index").reset_index(drop=True)


def load_egm2008_grid(archive: Path = NGA_ZIP) -> np.ndarray:
    if sha256(archive) != NGA_SHA256:
        raise ValueError("NGA EGM2008 archive SHA256 does not match the recorded source")
    grid = np.empty((NGA_ROWS, NGA_COLS), dtype="<f4")
    row_bytes = NGA_COLS * np.dtype("<f4").itemsize
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        if NGA_MEMBER not in names:
            raise ValueError(f"EGM2008 grid member missing: {NGA_MEMBER}")
        with zf.open(NGA_MEMBER) as stream:
            for row in range(NGA_ROWS):
                lead = stream.read(4)
                if len(lead) != 4 or struct.unpack("<i", lead)[0] != row_bytes:
                    raise ValueError(f"bad EGM2008 sequential record at row {row}")
                raw = stream.read(row_bytes)
                tail = stream.read(4)
                if len(raw) != row_bytes or tail != lead:
                    raise ValueError(f"truncated EGM2008 sequential record at row {row}")
                grid[row] = np.frombuffer(raw, dtype="<f4", count=NGA_COLS)
            if stream.read(1):
                raise ValueError("unexpected bytes after EGM2008 grid")
    return grid


def validate_egm2008_interpolation(grid: np.ndarray, archive: Path = NGA_ZIP) -> dict[str, Any]:
    """Check the grid interpolation against NGA's bundled OUTPUT3.DAT examples."""
    with zipfile.ZipFile(archive) as zf:
        lines = zf.read("OUTPUT3.DAT").decode("ascii").splitlines()
    reference = np.asarray([[float(value) for value in line.split()]
                            for line in lines if line.strip()], dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("unexpected NGA OUTPUT3.DAT sample layout")
    lat, lon, expected = reference.T
    computed = bilinear_grid(grid, lon, lat)
    differences = computed - expected
    maximum = float(np.nanmax(np.abs(differences)))
    if maximum > 0.01001:
        raise ValueError(f"NGA interpolation sample differs by {maximum:.6f} m")
    return {"sample_count": int(len(expected)), "max_abs_difference_m": maximum,
            "differences_m": differences.tolist(), "tolerance_m": 0.01001,
            "source_member": "OUTPUT3.DAT", "result": "PASS"}


def candidate_evidence_status(applied_status: str, physical_basis: bool) -> str:
    """Do not promote a sensitivity because of residual fit."""
    if applied_status not in VALID_APPLIED_STATUS:
        raise ValueError(f"invalid applied status: {applied_status}")
    if not physical_basis:
        return "UNAVAILABLE"
    if applied_status == "CONFIRMED_NOT_APPLIED":
        return "VERIFIED"
    if applied_status in {"CONFIRMED_APPLIED", "NOT_APPLICABLE"}:
        return "REJECTED"
    return "PHYSICAL_SENSITIVITY_ONLY"


def add_sensitivity(baseline_residual: Any, delta_height: Any) -> np.ndarray:
    baseline = np.asarray(baseline_residual, dtype=float)
    delta = np.asarray(delta_height, dtype=float)
    if baseline.shape != delta.shape:
        raise ValueError("baseline and sensitivity lengths differ")
    return baseline.copy() + delta


def residual_metrics(residual: Any) -> dict[str, float | int]:
    raw = np.asarray(residual, dtype=float).ravel()
    values = raw[np.isfinite(raw)]
    n_input = int(raw.size)
    n_nonfinite = n_input - int(values.size)
    if not len(values):
        return {
            "n": 0, "n_input": n_input, "n_nonfinite": n_nonfinite,
            "mean_bias_m": np.nan, "median_bias_m": np.nan, "RMSE_m": np.nan,
            "cRMSE_m": np.nan, "NMAD_m": np.nan, "cRMSE_NMAD_ratio": np.nan,
            "median_absolute_residual_m": np.nan, "MAE_m": np.nan,
            "median_centered_RMS_m": np.nan,
            "residual_P05_m": np.nan, "residual_P25_m": np.nan,
            "residual_P50_m": np.nan, "residual_P75_m": np.nan,
            "residual_P95_m": np.nan, "absolute_P90_m": np.nan,
            "absolute_P95_m": np.nan, "bias_squared_fraction_of_MSE": np.nan,
            "dispersion_squared_fraction_of_MSE": np.nan,
        }
    mean = float(values.mean())
    median = float(np.median(values))
    rmse = float(np.sqrt(np.mean(values ** 2)))
    centered_rms = float(np.sqrt(np.mean((values - mean) ** 2)))
    nmad = float(1.4826 * np.median(np.abs(values - median)))
    bias_fraction = mean ** 2 / rmse ** 2 if rmse > 0 else np.nan
    dispersion_fraction = centered_rms ** 2 / rmse ** 2 if rmse > 0 else np.nan
    return {
        "n": int(len(values)), "n_input": n_input, "n_nonfinite": n_nonfinite,
        "mean_bias_m": mean,
        "median_bias_m": median,
        "RMSE_m": rmse, "cRMSE_m": centered_rms, "NMAD_m": nmad,
        "cRMSE_NMAD_ratio": centered_rms / nmad if nmad > 0 else np.nan,
        "median_absolute_residual_m": float(np.median(np.abs(values))),
        "MAE_m": float(np.mean(np.abs(values))),
        "median_centered_RMS_m": float(np.sqrt(np.mean((values - median) ** 2))),
        "residual_P05_m": float(np.percentile(values, 5)),
        "residual_P25_m": float(np.percentile(values, 25)),
        "residual_P50_m": median,
        "residual_P75_m": float(np.percentile(values, 75)),
        "residual_P95_m": float(np.percentile(values, 95)),
        "absolute_P90_m": float(np.percentile(np.abs(values), 90)),
        "absolute_P95_m": float(np.percentile(np.abs(values), 95)),
        "bias_squared_fraction_of_MSE": bias_fraction,
        "dispersion_squared_fraction_of_MSE": dispersion_fraction,
    }


def compare_common_pulses(pulse_ids: Any, scenarios: dict[str, Any]) -> pd.DataFrame:
    ids = np.asarray(pulse_ids)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("pulse IDs must be unique one-dimensional identifiers")
    rows = []
    for label, residual in scenarios.items():
        values = np.asarray(residual, dtype=float)
        if values.shape != ids.shape:
            raise ValueError(f"scenario {label} does not use the common pulse IDs")
        if not np.isfinite(values).all():
            raise ValueError(f"scenario {label} has missing values on the frozen common pulse IDs")
        rows.append({"hypothesis": label, **residual_metrics(values)})
    return pd.DataFrame(rows)


def statistics_row(granule: str, field: str, scope: str, values: Any,
                   group_id: str = "") -> dict[str, Any]:
    return {"granule_id": granule, "field": field, "scope": scope,
            "group_id": group_id, **distribution(values)}


def _subsample_rows(frame: pd.DataFrame, field_values: dict[str, np.ndarray],
                    granule: str, common_ids: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(field_values["refh"])
    snr = field_values["refh_snr"]
    lon, lat = field_values["refh_longitude"], field_values["refh_latitude"]
    track = field_values["track_num"]
    correction_names = [name for name in CORRECTION_FIELDS if name in field_values]
    masks: dict[str, np.ndarray] = {
        "full_granule": np.ones(n, dtype=bool),
        "snr_ge_3": np.isfinite(snr) & (snr >= 3),
        "snr_ge_5": np.isfinite(snr) & (snr >= 5),
    }
    fixed_sets = {
        "existing_ground_target_population": frame["pulse_index"].to_numpy(dtype=np.int64),
        "independent_test_eligible": frame.attrs["test_ids"],
        "independent_test_c7_common_support": common_ids,
    }
    for scope, ids in fixed_sets.items():
        m = np.zeros(n, dtype=bool)
        m[ids] = True
        masks[scope] = m
    for field in correction_names:
        data = field_values[field]
        for scope, mask in masks.items():
            rows.append(statistics_row(granule, field, scope, data[mask]))
        good_track = np.isfinite(track)
        for group in np.unique(track[good_track]):
            mask = good_track & (track == group)
            rows.append(statistics_row(granule, field, "track", data[mask], str(int(group))))
        good_xy = np.isfinite(lon) & np.isfinite(lat)
        if np.any(good_xy):
            x0, x1 = np.nanmin(lon[good_xy]), np.nanmax(lon[good_xy])
            y0, y1 = np.nanmin(lat[good_xy]), np.nanmax(lat[good_xy])
            xbin = np.full(n, -1, dtype=int)
            ybin = np.full(n, -1, dtype=int)
            xbin[good_xy] = np.clip(((lon[good_xy] - x0) / max(x1 - x0, 1e-12) * 4).astype(int), 0, 3)
            ybin[good_xy] = np.clip(((lat[good_xy] - y0) / max(y1 - y0, 1e-12) * 4).astype(int), 0, 3)
            for y in range(4):
                for x in range(4):
                    mask = good_xy & (xbin == x) & (ybin == y)
                    if np.any(mask):
                        rows.append(statistics_row(granule, field, "spatial_4x4", data[mask], f"x{x}_y{y}"))
    return rows


def _load_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    common = pd.read_parquet(TARGET_COMMON)
    test = pd.read_parquet(TARGET_TEST)
    common = common.copy()
    test = test.copy()
    return common, test


def _matrix_rows() -> list[dict[str, Any]]:
    details = {
        "refh_neutat_delay_total": (
            "Neutral-atmospheric delay associated with Refh ranging", "UNKNOWN",
            "Range-path effect; dataset units and sign are not declared", "UNKNOWN"),
        "refh_neutat_delay_derivative": (
            "Neutral-atmospheric delay derivative associated with Refh", "UNKNOWN",
            "Range-path rate/derivative; units and use are not declared", "UNKNOWN"),
        "rwstart_neutat_delay_total": (
            "Neutral-atmospheric delay at range-window start", "UNKNOWN",
            "Range-window endpoint; units and use in Refh processing are not declared", "UNKNOWN"),
        "rwstart_neutat_delay_derivative": (
            "Neutral-atmospheric delay derivative at range-window start", "UNKNOWN",
            "Range-window endpoint derivative; units and use are not declared", "UNKNOWN"),
        "rwstop_neutat_delay_total": (
            "Neutral-atmospheric delay at range-window stop", "UNKNOWN",
            "Range-window endpoint; units and use in Refh processing are not declared", "UNKNOWN"),
        "rwstop_neutat_delay_derivative": (
            "Neutral-atmospheric delay derivative at range-window stop", "UNKNOWN",
            "Range-window endpoint derivative; units and use are not declared", "UNKNOWN"),
        "range_bias_correction": (
            "Range-bias correction field", "UNKNOWN",
            "Instrument range calibration; units/sign not declared", "UNKNOWN"),
        "geoid": (
            "Geoid undulation field; numerically compared with official models", "NUMERICALLY_SUPPORTED",
            "Potential ellipsoid-to-geoid height conversion; applied stage unknown", "UNKNOWN"),
        "geoid_free2mean": (
            "Free-to-mean tide field; numerical latitude formula checked", "NUMERICALLY_SUPPORTED",
            "Permanent tide-system conversion; field sign/application stage unknown", "UNKNOWN"),
        "tide_earth": ("Earth tide field", "UNKNOWN", "Tide displacement/correction", "UNKNOWN"),
        "tide_earth_free2mean": ("Earth tide free-to-mean field", "UNKNOWN", "Tide convention", "UNKNOWN"),
        "tide_load": ("Load tide field", "UNKNOWN", "Tide displacement/correction", "UNKNOWN"),
        "tide_pole": ("Pole tide field", "UNKNOWN", "Tide displacement/correction", "UNKNOWN"),
        "tide_ocean": ("Ocean tide field", "UNKNOWN", "Tide displacement/correction", "UNKNOWN"),
        "tide_ocean_pole": ("Ocean pole tide field", "UNKNOWN", "Tide displacement/correction", "UNKNOWN"),
        "dac": ("Dynamic atmospheric correction field", "UNKNOWN", "Atmospheric/ocean loading correction", "UNKNOWN"),
    }
    rows = []
    for field, (definition, evidence, target, status) in details.items():
        rows.append({
            "correction": field, "definition": definition, "source": "H5 field; no field-level source metadata",
            "units": "UNKNOWN", "sign_convention": "UNKNOWN", "physical_target": target,
            "processing_stage": "NOT_IDENTIFIABLE_FROM_AVAILABLE_L1B_METADATA",
            "applied_to_refh": "UNKNOWN", "applied_status": status,
            "numerical_evidence_level": evidence,
            "evidence": ("H5 field exists, but dataset/root metadata do not state a per-field applied flag, "
                         "unit, sign convention, or processor stage."),
        })
    for row in rows:
        if row["applied_status"] not in VALID_APPLIED_STATUS:
            raise ValueError("invalid correction application status")
    return rows


def _evidence_rows() -> list[dict[str, Any]]:
    return [
        {"document_title": "CASALS L1B Waveforms Tutorial", "url": "https://book.cryointhecloud.com/l1b-waveforms-tutorial/",
         "version_date": "2026-02-10", "section_page": "L1B fields / Refh and correction datasets",
         "supported_claim": "Refh is described as a maximum-amplitude waveform-bin height in WGS84 ellipsoidal coordinates; the tutorial says some correction datasets are marked NOT APPLIED but does not map that flag to each field in these H5 files.",
         "casals_specific": True, "evidence_limit": "Does not establish source realization/epoch or this product's per-field applied status."},
        {"document_title": "CASALS H5 root and dataset metadata", "url": "local H5 inputs; hashes in h5_processing_metadata.json",
         "version_date": "file-specific", "section_page": "root attributes and audited dataset attributes",
         "supported_claim": "Records present fields, attributes, storage fill values, and missing units/status metadata for the two exact granules.",
         "casals_specific": True, "evidence_limit": "Metadata absence cannot prove a correction was not applied."},
        {"document_title": "NGA EGM2008 official model page and 2.5-minute grid", "url": "https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84",
         "version_date": "EGM2008; archive checksum recorded", "section_page": "EGM2008 2.5-minute global geoid height grid; WGS84 Tide-Free",
         "supported_claim": "Defines the official comparison model and its grid/tide convention; the supplied official grid is sampled at the H5 comparison coordinates.",
         "casals_specific": False, "evidence_limit": "Numerical agreement identifies a model pattern, not the CASALS processor's source declaration."},
        {"document_title": "NOAA GEOID18 technical details", "url": "https://geodesy.noaa.gov/GEOID/GEOID18/geoid18_tech_details.shtml",
         "version_date": "GEOID18", "section_page": "Purpose and reference frame",
         "supported_claim": "GEOID18 provides geoid heights intended for NAVD88 with NAD83(2011) epoch 2010.0 ellipsoid heights.",
         "casals_specific": False, "evidence_limit": "Does not verify the CASALS source frame."},
        {"document_title": "IERS Conventions (2010)", "url": "https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36.pdf?__blob=publicationFile&v=1",
         "version_date": "2010", "section_page": "Permanent tide conventions", "supported_claim": "Provides the geodetic distinction among tide-free, zero-tide, and mean-tide systems.",
         "casals_specific": False, "evidence_limit": "The matching latitude formula does not identify the H5 field's documented convention."},
        {"document_title": "CASALS bin-georeferencing audit", "url": "research/geolocation/bin_georeferencing_audit/",
         "version_date": "current repository audit outputs", "section_page": "full-granule beam/range-window audit",
         "supported_claim": "Provides existing H5 geometry, range-window, and beam-direction diagnostics reused here.",
         "casals_specific": True, "evidence_limit": "Internal geometric closure cannot independently reveal whether a stored correction was applied."},
    ]


def _correlation_rows(granule: str, sample: pd.DataFrame, arrays: dict[str, np.ndarray],
                      frame: dict[str, Any]) -> list[dict[str, Any]]:
    from scipy.stats import spearmanr

    rows = []
    ids = sample["pulse_index"].to_numpy(dtype=np.int64)
    residual = sample["C7_residual_m"].to_numpy(dtype=float)
    lon, lat = arrays["refh_longitude"][ids], arrays["refh_latitude"][ids]
    delay = arrays["refh_neutat_delay_total"][ids]
    range_bias = arrays["range_bias_correction"][ids]
    az, el = arrays["local_beam_azimuth"][ids], arrays["local_beam_elevation"][ids]
    beam_dir = beam(lon, lat, az, el)
    incidence = np.rad2deg(np.arccos(np.clip(-np.sum(beam_dir * up_normal(lon, lat), axis=1), -1, 1)))
    c7_shift = frame["c7_height_m"] - arrays["refh"][ids]
    geoid18 = -c7_shift
    geoid_diff = arrays["geoid"][ids] - geoid18
    for label, values in (("neutral_delay_field_numeric", delay),
                          ("range_bias_field_numeric", range_bias),
                          ("beam_incidence_deg", incidence),
                          ("H5_geoid_minus_GEOID18_m", geoid_diff)):
        good = np.isfinite(residual) & np.isfinite(values)
        corr = float(spearmanr(values[good], residual[good]).statistic) if good.sum() > 2 and np.ptp(values[good]) else np.nan
        rows.append({"granule_id": granule, "context_variable": label,
                     "method": "Spearman; descriptive only", "n": int(good.sum()),
                     "correlation": corr, "interpretation": "association only; no causal/applied-status inference"})
    for field in ("track_num", "sweep_num"):
        grouping = arrays[field][ids]
        temp = pd.DataFrame({"group": grouping, "residual": residual}).dropna()
        for group, part in temp.groupby("group", sort=True):
            rows.append({"granule_id": granule, "context_variable": field,
                         "method": "descriptive group", "group_id": group,
                         "n": len(part), "median_residual_m": float(part.residual.median()),
                         "mean_residual_m": float(part.residual.mean()),
                         "interpretation": "descriptive only; fixed independent-test support"})
    return rows


def _run_frame_audit(granule: str, test_common: pd.DataFrame,
                     arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    datadir.set_data_dir(str(PROJ_DATA.resolve()))
    source = CRS.from_epsg(4979)
    target = CRS.from_user_input("EPSG:6347+5703")
    # The frozen project pipeline uses longitude/latitude order explicitly.
    group = TransformerGroup(source, target, always_xy=True, allow_ballpark=False)
    if len(group.transformers) != 1:
        raise RuntimeError(f"Expected one selected C7 transformer, found {len(group.transformers)}")
    c7 = group.transformers[0]
    frame_group = TransformerGroup(source, CRS.from_epsg(6319), always_xy=True, allow_ballpark=False)
    if not frame_group.transformers:
        raise RuntimeError("No WGS84 to NAD83(2011) 3D operation available")
    frame_transformer = frame_group.transformers[0]
    ids = test_common["pulse_index"].to_numpy(dtype=np.int64)
    lon, lat, h = arrays["refh_longitude"][ids], arrays["refh_latitude"][ids], arrays["refh"][ids]
    ex, ny, c7h = c7.transform(lon, lat, h)
    fx, fy, fh = frame_transformer.transform(lon, lat, h)
    stored = test_common
    diffs = {
        "easting_difference_m": np.asarray(ex) - stored["c7_easting_m_epsg6347"].to_numpy(float),
        "northing_difference_m": np.asarray(ny) - stored["c7_northing_m_epsg6347"].to_numpy(float),
        "height_difference_m": np.asarray(c7h) - stored["C7_height_m"].to_numpy(float),
    }
    if any(np.nanmax(np.abs(x)) > 1e-6 for x in diffs.values()):
        raise RuntimeError("Reconstructed selected C7 operation does not match saved C7 coordinates")
    c7_shift = np.asarray(c7h) - h
    frame_shift = np.asarray(fh) - h
    geoid_component = c7_shift - frame_shift
    rows = []
    for component, values in (("C7 total vertical shift", c7_shift),
                              ("WGS84 ensemble to NAD83(2011) frame step", frame_shift),
                              ("GEOID18 vertical-grid component", geoid_component)):
        component_stats = distribution(values)
        rows.append({"granule_id": granule, "component": component,
                     "operation_description": c7.description,
                     "frame_operation_description": frame_transformer.description,
                     "frame_operation_definition": frame_transformer.definition,
                     "operation_accuracy_m": c7.accuracy,
                     "median_contribution_m": component_stats["median"],
                     "mean_contribution_m": component_stats["mean"],
                     "minimum_contribution_m": component_stats["minimum"],
                     "maximum_contribution_m": component_stats["maximum"],
                     "P05_contribution_m": component_stats["P05"],
                     "P95_contribution_m": component_stats["P95"],
                     "component_stats_json": json.dumps(component_stats),
                     "sample_n": len(values), "c7_matches_saved_within_1e-6_m": True})
    detail = {
        "selected_operation": c7.description, "accuracy_m": c7.accuracy,
        "definition": c7.definition, "source_crs": "EPSG:4979 (WGS84 3D ensemble; assumed)",
        "target_crs": "EPSG:6347 + EPSG:5703", "required_local_grid": "us_noaa_g2018u0.tif",
        "frame_operation": frame_transformer.description,
        "frame_operation_accuracy_m": frame_transformer.accuracy,
        "frame_operation_definition": frame_transformer.definition,
        "frame_numeric_shift_stats": distribution(frame_shift),
        "c7_total_shift_stats": distribution(c7_shift),
        "geoid_component_stats": distribution(geoid_component),
        "stored_c7_parity_max_abs_m": {k: float(np.nanmax(np.abs(v))) for k, v in diffs.items()},
        "source_realization": None, "source_coordinate_epoch": None,
        "interpretation": "C7 numerically applies the GEOID18 vertical grid. The available frame step is a 2 m-accuracy no-op; the unknown CASALS realization/epoch prevents a verified physical source-frame claim.",
    }
    return pd.DataFrame(rows), detail, {
        "c7_height_m": np.asarray(c7h), "frame_height_m": np.asarray(fh),
        "c7_shift_m": c7_shift, "frame_shift_m": frame_shift,
        "geoid18_undulation_m": -geoid_component,
    }


def _make_figures(stats: pd.DataFrame, hyp: pd.DataFrame, geoid: pd.DataFrame,
                  sensitivity: pd.DataFrame, frame_rows: pd.DataFrame,
                  output: Path) -> None:
    figdir = output / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    colors = {GRANULES[0]: "#3569a8", GRANULES[1]: "#e47c2f"}

    full = stats[stats.scope == "full_granule"]
    panels = [
        ["refh_neutat_delay_total", "range_bias_correction"],
        ["geoid"],
        ["geoid_free2mean", "tide_earth", "tide_earth_free2mean", "tide_load",
         "tide_pole", "tide_ocean", "tide_ocean_pole", "dac"],
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, fields in zip(axes, panels):
        positions = np.arange(len(fields))
        width = 0.34
        for j, granule in enumerate(GRANULES):
            part = full[(full.granule_id == granule) & full.field.isin(fields)].set_index("field").reindex(fields)
            xs = positions + (j - .5) * width
            ax.bar(xs, part["median"], width, label=granule.split("_")[2], color=colors[granule])
            for x, low, high in zip(xs, part.P05, part.P95):
                if np.isfinite(low) and np.isfinite(high):
                    ax.vlines(x, low, high, color=colors[granule], linewidth=1.2)
        ax.axhline(0, color="0.35", linewidth=.8)
        ax.set_xticks(positions, fields, rotation=35, ha="right")
        ax.set_ylabel("Stored numeric value")
    axes[0].set_title("Range and atmosphere")
    axes[1].set_title("Geoid")
    axes[2].set_title("Tides and DAC")
    axes[0].legend(title="Granule date", fontsize=8)
    fig.suptitle("H5 correction-field medians and P05-P95; units are not declared", y=1.02)
    fig.tight_layout(); fig.savefig(figdir / "correction_magnitudes.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.0))
    for x, granule in enumerate(GRANULES):
        row = sensitivity.loc[(sensitivity.granule_id == granule)
                              & (sensitivity.scope == "independent_test_common_support")
                              & (sensitivity.field == "refh_neutat_delay_total")].iloc[0]
        median = row["median_height_effect_if_input_is_m"]
        lower = median - row["P05_height_effect_if_input_is_m"]
        upper = row["P95_height_effect_if_input_is_m"] - median
        ax.errorbar(x, median, yerr=[[lower], [upper]], fmt="o", capsize=5,
                    markersize=7, color=colors[granule], label=granule.split("_")[2])
    ax.axhline(0, color="0.3", linewidth=.8)
    ax.set_xticks(range(len(GRANULES)), ["2024-11-12", "2024-11-18"])
    ax.set_ylabel("Height change if stored delay is metres of excess slant range (m)")
    ax.set_title("Geometry sensitivity on fixed independent-test targets (P05-P95)")
    ax.legend(title="Granule")
    fig.tight_layout(); fig.savefig(figdir / "range_height_sensitivity.png", dpi=170); plt.close(fig)

    model_rows = geoid[(geoid["scope"] == "geoid18_common_sample") & geoid["comparison"].isin(
        ["H5 minus EGM2008", "H5 minus GEOID18", "EGM2008 minus GEOID18"])]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = model_rows.granule_id.str.extract(r"(202411\d{2})")[0] + "\n" + model_rows.comparison
    short_labels = labels.str.replace("20241112", "Nov 12", regex=False).str.replace(
        "20241118", "Nov 18", regex=False).str.replace("H5 minus EGM2008", "H5 − EGM08", regex=False).str.replace(
        "H5 minus GEOID18", "H5 − GEOID18", regex=False).str.replace(
        "EGM2008 minus GEOID18", "EGM08 − GEOID18", regex=False)
    x_model = np.arange(len(model_rows))
    model_values = model_rows.median_difference_m.to_numpy(float)
    ax.bar(x_model, model_values, color="#627d98")
    ax.axhline(0, color="0.3", linewidth=.8)
    for xpos, value in zip(x_model, model_values):
        if abs(value) < .01:
            ax.scatter([xpos], [value], color="#c44536", marker="D", s=26, zorder=4)
            ax.annotate(f"{value:+.4f} m", (xpos, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)
        else:
            ax.annotate(f"{value:+.3f} m", (xpos, value), xytext=(0, -4),
                        textcoords="offset points", ha="center", va="top", fontsize=8, color="white")
    ax.set_xticks(x_model, short_labels, rotation=25, ha="right")
    ax.set_ylabel("Median model difference (m)")
    ax.set_title("H5 geoid field against official global and regional models")
    ax.set_ylim(min(model_values) - .12, .12)
    fig.tight_layout(); fig.savefig(figdir / "geoid_comparison.png", dpi=170); plt.close(fig)

    order = ["C7", "C1", "C3", "H1 atmospheric sensitivity", "H2 range-bias sensitivity"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(order)); width = .34
    for j, granule in enumerate(GRANULES):
        part = hyp[(hyp.granule_id == granule) & (hyp.population == "snr_ge_5")].set_index("hypothesis").reindex(order)
        ax.bar(x + (j - .5) * width, part.median_bias_m, width,
               label=granule.split("_")[2], color=colors[granule])
    ax.axhline(0, color="0.3", linewidth=.8)
    ax.set_xticks(x, ["C7 baseline", "C1", "C3", "H1\natmosphere", "H2\nrange bias"])
    ax.set_ylabel("Median residual on common independent-test targets (m)")
    ax.set_title("SNR >= 5 common targets; H1/H2 are sensitivities, not corrections")
    ax.legend(title="Granule date")
    fig.tight_layout(); fig.savefig(figdir / "c1_c3_c7_bias_comparison.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    frame = frame_rows[frame_rows.component.isin(("WGS84 ensemble to NAD83(2011) frame step",
                                                   "GEOID18 vertical-grid component"))]
    for j, granule in enumerate(GRANULES):
        part = frame[frame.granule_id == granule]
        vals = [json.loads(v)["median"] for v in part.component_stats_json]
        positions = np.arange(len(vals)) + j * .36
        ax.bar(positions, vals, .34,
               label=granule.split("_")[2], color=colors[granule])
        if vals and abs(vals[0]) < 1e-9:
            ax.annotate("0.000 m", (positions[0], 0), xytext=(0, 7 + 10 * j),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=colors[granule])
    ax.set_xticks([.18, 1.18], ["WGS84 to NAD83(2011)\nframe step", "GEOID18 grid\nvertical step"])
    ax.set_ylabel("Median vertical contribution to C7 (m)")
    ax.set_title("Selected C7 operation decomposition")
    ax.legend(title="Granule date")
    fig.tight_layout(); fig.savefig(figdir / "frame_decomposition.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    point_specs = [
        ("C7", "C7 / H2", "o", (5, -12)),
        ("C1", "C1", "s", (-2, 12)),
        ("C3", "C3", "^", (4, -14)),
        ("H1 atmospheric sensitivity", "H1", "D", (5, 8)),
    ]
    for ax, granule in zip(axes, GRANULES):
        part = hyp[(hyp.granule_id == granule) & (hyp.population == "snr_ge_5")].set_index("hypothesis")
        for hypothesis, label, marker, offset in point_specs:
            row = part.loc[hypothesis]
            ax.scatter(row.RMSE_m, row.median_bias_m, marker=marker, s=42,
                       color=colors[granule], edgecolor="white", linewidth=.5, zorder=3)
            if hypothesis == "C7" and granule == GRANULES[0]:
                offset = (-48, -12)
            ax.annotate(label, (row.RMSE_m, row.median_bias_m), xytext=offset,
                        textcoords="offset points", fontsize=8,
                        arrowprops={"arrowstyle": "-", "lw": .55, "color": "0.4"})
        ax.axhline(0, color="0.3", linewidth=.8)
        ax.set_title(f"{granule.split('_')[2]} | n={int(part.loc['C7', 'n'])}")
        ax.set_xlim(.15, 2.05)
        ax.set_xlabel("RMSE (m)")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Median residual (m)")
    axes[0].text(.02, .02, "H2 overlaps C7 because stored range bias is zero.",
                 transform=axes[0].transAxes, fontsize=7, va="bottom")
    fig.suptitle("Residual and RMSE trade-offs do not establish attribution", y=1.02)
    fig.tight_layout(); fig.savefig(figdir / "bias_attribution.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def run_audit(output: Path = OUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    field_inventory_rows: list[dict[str, Any]] = []
    statistic_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"granules": {}, "scope": "read-only correction audit"}
    atmospheric_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    frame_rows_all: list[pd.DataFrame] = []
    hypothesis_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    geoid_rows: list[dict[str, Any]] = []
    geoid_model_evidence: dict[str, Any] = {}
    common_table, test_table = _load_targets()
    geoid_sample = pd.read_csv(GEOID18_SAMPLE)
    egm_grid = load_egm2008_grid()
    egm_sample_validation = validate_egm2008_interpolation(egm_grid)
    time_trend_rows: list[dict[str, Any]] = []

    for granule in GRANULES:
        path = H5_DIR / f"{granule}{H5_SUFFIX}"
        with h5py.File(path, "r") as h5:
            inv, paths = h5_inventory(h5)
            field_inventory_rows.extend({"granule_id": granule, **row}
                                        for row in field_inventory(inv, paths))
            fields: dict[str, np.ndarray] = {}
            n = int(h5.attrs["n_pulses"])
            for name in AUDIT_FIELDS:
                if name not in paths:
                    continue
                ds = h5[paths[name]]
                if ds.ndim == 1 and ds.shape[0] == n and np.issubdtype(ds.dtype, np.number):
                    fields[name] = collect_values(ds)
            if len(fields.get("refh", [])) != n:
                raise ValueError(f"Refh pulse count does not match metadata for {granule}")
            root_attrs = {k: json_value(v) for k, v in h5.attrs.items()}
            metadata["granules"][granule] = {
                "input_path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                "file_size_bytes": path.stat().st_size, "pulse_count": n,
                "global_attributes": root_attrs,
                "audited_dataset_attributes": {
                    row["field"]: json.loads(row["attributes_json"])
                    for row in field_inventory(inv, paths)
                },
                "dataset_presence": {row["field"]: row["presence_status"]
                                     for row in field_inventory(inv, paths)},
                "field_units_are_source_declared": False,
                "interpretation": "HDF5 storage fill values are recorded separately; only _FillValue/missing_value attrs are treated as missing-value declarations.",
            }

        # Full, SNR-stratified, frozen target, per-track, and 4x4 spatial stats.
        common_g = common_table[common_table.granule_id == granule]
        test_g = test_table[test_table.granule_id == granule]
        common_ids = test_g.loc[test_g.common_support_all_candidates.fillna(False), "pulse_index"].to_numpy(np.int64)
        if len(np.unique(common_ids)) != len(common_ids):
            raise ValueError("independent-test common-support pulse IDs are not unique")
        if len(common_g) != 0:
            common_ids_609 = common_g.pulse_index.to_numpy(np.int64)
        else:
            common_ids_609 = np.array([], dtype=np.int64)
        if common_ids_609.size and np.any((common_ids_609 < 0) | (common_ids_609 >= n)):
            raise ValueError("frozen ground-target population has pulse indices outside the H5")
        # The 609-row table is used for its exact frozen support population.
        pop_frame = common_g.copy()
        pop_frame.attrs["test_ids"] = test_g.pulse_index.to_numpy(np.int64)
        stats_source = pop_frame if len(pop_frame) else test_g
        statistic_rows.extend(_subsample_rows(stats_source, fields, granule, common_ids))
        snr_all = fields["refh_snr"]
        sensitivity_masks = {
            "full_granule": np.ones(n, dtype=bool),
            "snr_ge_3": np.isfinite(snr_all) & (snr_all >= 3),
            "snr_ge_5": np.isfinite(snr_all) & (snr_all >= 5),
        }
        geometry_effects: dict[str, np.ndarray] = {}
        for field in ("refh_neutat_delay_total", "range_bias_correction"):
            geometry_effects[field] = height_sensitivity(
                fields[field], fields["refh_longitude"], fields["refh_latitude"],
                fields["local_beam_azimuth"], fields["local_beam_elevation"],
            )
            factor = float(metadata["granules"][granule]["global_attributes"].get("ns_to_meters", np.nan))
            for scope, mask in sensitivity_masks.items():
                values = geometry_effects[field][mask]
                sensitivity_rows.append({
                    "granule_id": granule, "field": field, "scope": scope,
                    "population": "H5 rows selected by the named fixed mask", "n": int(mask.sum()),
                    "height_effect_if_input_is_m": distribution(values),
                    "median_height_effect_if_input_is_m": float(np.nanmedian(values)),
                    "minimum_height_effect_if_input_is_m": float(np.nanmin(values)),
                    "maximum_height_effect_if_input_is_m": float(np.nanmax(values)),
                    "P05_height_effect_if_input_is_m": float(np.nanpercentile(values, 5)),
                    "P95_height_effect_if_input_is_m": float(np.nanpercentile(values, 95)),
                    "height_effect_if_input_is_ns": distribution(values * factor) if field.startswith("refh_neutat") else "NOT_APPLICABLE",
                    "unit_status": "UNKNOWN", "applied_status": "UNKNOWN",
                    "candidate_status": candidate_evidence_status("UNKNOWN", True),
                    "sign_convention_assumption": "positive value means excess sensor-to-target slant range to remove",
                })
        if "delta_time" in fields:
            time = fields["delta_time"]
            valid_time = np.flatnonzero(np.isfinite(time))
            ordered_bins = np.array_split(valid_time[np.argsort(time[valid_time])], 16)
            for field in CORRECTION_FIELDS:
                if field not in fields:
                    continue
                for index, group_ids in enumerate(ordered_bins):
                    if not len(group_ids):
                        continue
                    d = distribution(fields[field][group_ids])
                    time_trend_rows.append({
                        "granule_id": granule, "field": field,
                        "time_bin": f"q{index:02d}", "n": len(group_ids),
                        "delta_time_start_numeric": float(np.nanmin(time[group_ids])),
                        "delta_time_end_numeric": float(np.nanmax(time[group_ids])),
                        "median_numeric": d["median"], "minimum_numeric": d["minimum"],
                        "maximum_numeric": d["maximum"],
                        "unit_status": "UNKNOWN; delta_time and correction field units are not declared",
                    })

        snr_check = fields["refh_snr"][test_g.pulse_index.to_numpy(np.int64)]
        if not np.allclose(snr_check, test_g.refh_snr.to_numpy(float), atol=1e-8, rtol=0, equal_nan=True):
            raise RuntimeError("saved target SNR does not match exact H5 pulse indices")

        # Target populations use the fixed parquet IDs; do not choose new points.
        sample = select_independent_common_test(test_g)
        ids = sample.pulse_index.to_numpy(np.int64)
        if len(ids) == 0:
            raise RuntimeError(f"no frozen common independent-test target rows for {granule}")
        h1_m = geometry_effects["refh_neutat_delay_total"][ids]
        h2_m = geometry_effects["range_bias_correction"][ids]
        factor = float(metadata["granules"][granule]["global_attributes"].get("ns_to_meters", np.nan))
        for field, values in (("refh_neutat_delay_total", h1_m),
                              ("range_bias_correction", h2_m)):
            sensitivity_rows.append({
                "granule_id": granule, "field": field,
                "scope": "independent_test_common_support",
                "population": "fixed independent-test common C0-C7 support", "n": len(ids),
                "height_effect_if_input_is_m": distribution(values),
                "median_height_effect_if_input_is_m": float(np.nanmedian(values)),
                "minimum_height_effect_if_input_is_m": float(np.nanmin(values)),
                "maximum_height_effect_if_input_is_m": float(np.nanmax(values)),
                "P05_height_effect_if_input_is_m": float(np.nanpercentile(values, 5)),
                "P95_height_effect_if_input_is_m": float(np.nanpercentile(values, 95)),
                "height_effect_if_input_is_ns": distribution(values * factor) if field.startswith("refh_neutat") else "NOT_APPLICABLE",
                "unit_status": "UNKNOWN", "applied_status": "UNKNOWN",
                "candidate_status": candidate_evidence_status("UNKNOWN", True),
                "sign_convention_assumption": "positive value means excess sensor-to-target slant range to remove",
            })
        atmospheric_rows.append({
            "granule_id": granule, "population": "fixed independent-test common C0-C7 support",
            "n": len(ids), "refh_delay_stats_json": json.dumps(distribution(fields["refh_neutat_delay_total"])),
            "rwstart_delay_stats_json": json.dumps(distribution(fields.get("rwstart_neutat_delay_total", np.full(n, np.nan)))),
            "rwstop_delay_stats_json": json.dumps(distribution(fields.get("rwstop_neutat_delay_total", np.full(n, np.nan)))),
            "derivative_stats_json": json.dumps(distribution(fields.get("refh_neutat_delay_derivative", np.full(n, np.nan)))),
            "unit_status": "UNKNOWN; H5 contains no units attribute",
            "application_status": "UNKNOWN; field presence is not proof of application state",
            "physical_height_effect_full_if_values_are_m": json.dumps(distribution(geometry_effects["refh_neutat_delay_total"])),
            "physical_height_effect_full_if_values_are_ns": json.dumps(distribution(
                geometry_effects["refh_neutat_delay_total"] * float(metadata["granules"][granule]["global_attributes"].get("ns_to_meters", np.nan)))),
            "physical_height_effect_test_common_if_values_are_m": json.dumps(distribution(h1_m)),
            "physical_height_effect_test_common_median_if_values_are_m": float(np.nanmedian(h1_m)),
            "sign_interpretation": "positive excess range removed moves target toward sensor; with downward beam this raises ellipsoidal height",
            "geometry_source": "existing bin audit beam() helper; measured local beam fields interpreted empirically as radians",
            "double_correction_risk": "YES; applied status unknown; sensitivity must not be promoted to corrected Refh",
        })
        # Existing window/timing fields: measured values only, no bin correction fitted.
        for name in ("bin_size", "refh_bounce_time_offset", "rwstart_bounce_time_offset",
                     "rwstop_bounce_time_offset", "delta_time", "rwstart", "rwstop", "refh"):
            if name in fields:
                timing_rows.append({"granule_id": granule, "field": name,
                                    "interpretation": "stored H5 numeric values; units not assumed",
                                    **distribution(fields[name])})
        if all(name in fields for name in ("rwstart_neutat_delay_total", "rwstop_neutat_delay_total",
                                            "refh_neutat_delay_total", "rwstart_bounce_time_offset",
                                            "rwstop_bounce_time_offset", "refh_bounce_time_offset")):
            dt = fields["rwstop_bounce_time_offset"] - fields["rwstart_bounce_time_offset"]
            with np.errstate(divide="ignore", invalid="ignore"):
                fraction = (fields["refh_bounce_time_offset"] - fields["rwstart_bounce_time_offset"]) / dt
                interpolated = fields["rwstart_neutat_delay_total"] + fraction * (
                    fields["rwstop_neutat_delay_total"] - fields["rwstart_neutat_delay_total"])
            timing_rows.append({"granule_id": granule, "field": "refh_neutat_minus_endpoint_linear_interpolation",
                                "interpretation": "numerical continuity only; no processor formula inferred",
                                **distribution(fields["refh_neutat_delay_total"] - interpolated)})

        # Exact same independent-test IDs for C7, C1/C3 and physical sensitivities.
        scenarios = {
            "C7": sample.C7_residual_m.to_numpy(float),
            "C1": sample.C1_residual_m.to_numpy(float),
            "C3": sample.C3_residual_m.to_numpy(float),
            "H1 atmospheric sensitivity": add_sensitivity(sample.C7_residual_m.to_numpy(float), h1_m),
            "H2 range-bias sensitivity": add_sensitivity(sample.C7_residual_m.to_numpy(float), h2_m),
        }
        snr_common = fields["refh_snr"][ids]
        populations = {
            "all_common_support": np.ones(len(ids), dtype=bool),
            "snr_ge_3": np.isfinite(snr_common) & (snr_common >= 3),
            "snr_ge_5": np.isfinite(snr_common) & (snr_common >= 5),
        }
        for population, mask in populations.items():
            population_ids = ids[mask]
            subset_scenarios = {name: values[mask] for name, values in scenarios.items()}
            hyp = compare_common_pulses(population_ids, subset_scenarios)
            for row in hyp.to_dict(orient="records"):
                label = row["hypothesis"]
                status = ("PROVISIONAL_NOT_DATUM_VERIFIED" if label == "C7" else
                          "PHYSICAL_SENSITIVITY_ONLY" if label.startswith("H") else
                          "NUMERICAL_COMPARISON_ONLY")
                candidate_status = ("DOCUMENTED_ALTERNATIVE" if label == "C7" else
                                    "PHYSICAL_SENSITIVITY_ONLY")
                hypothesis_rows.append({"granule_id": granule, "hypothesis": label,
                                        "evidence_status": status, "population": population,
                                        "candidate_status": candidate_status,
                                        "pulse_id_sha256": hashlib.sha256(population_ids.tobytes()).hexdigest(),
                                        "same_pulse_ids": True, "C8_used": False, **row})
                attribution_rows.append({"granule_id": granule, "hypothesis": label,
                                         "evidence_status": status, "candidate_status": candidate_status,
                                         "population": population, **row,
                                         "conclusion": ("C7 baseline" if label == "C7" else
                                             "numerical comparison; physical source frame or applied status unresolved" if label in {"C1", "C3"} else
                                             "sensitivity only; lower residual statistics do not establish an unapplied correction")})
            # Requested frame and combined-chain hypotheses remain explicit N/A
            # rows because their required source-processing evidence is absent.
            empty_metrics = residual_metrics([])
            for label, candidate_status, note in (
                ("H3 verified source-frame alternative", "UNAVAILABLE",
                 "Source realization/epoch are absent; no verified frame alternative can be constructed."),
                ("H5 physically justified combined chain", "UNAVAILABLE",
                 "Per-field applied status, units, and sign conventions are unavailable; no combined chain is identifiable."),
            ):
                row = {"hypothesis": label, **empty_metrics}
                hypothesis_rows.append({"granule_id": granule, "evidence_status": "UNAVAILABLE",
                                        "population": population, "candidate_status": candidate_status,
                                        "pulse_id_sha256": hashlib.sha256(population_ids.tobytes()).hexdigest(),
                                        "same_pulse_ids": "NOT_EVALUATED", "C8_used": False, **row})
                attribution_rows.append({"granule_id": granule, "evidence_status": "UNAVAILABLE",
                                         "candidate_status": candidate_status, "population": population,
                                         **row, "conclusion": note})
            # H4 reports the already-computed C3 free-to-mean sensitivity explicitly.
            c3_row = hyp.loc[hyp.hypothesis == "C3"].iloc[0].to_dict()
            h4_row = {**c3_row, "hypothesis": "H4 geoid/tide convention alternative (C3)"}
            hypothesis_rows.append({"granule_id": granule, "evidence_status": "PHYSICAL_SENSITIVITY_ONLY",
                                    "population": population, "candidate_status": "PHYSICAL_SENSITIVITY_ONLY",
                                    "pulse_id_sha256": hashlib.sha256(population_ids.tobytes()).hexdigest(),
                                    "same_pulse_ids": True, "C8_used": False, **h4_row})
            attribution_rows.append({"granule_id": granule, "evidence_status": "PHYSICAL_SENSITIVITY_ONLY",
                                     "candidate_status": "PHYSICAL_SENSITIVITY_ONLY", "population": population,
                                     **h4_row,
                                     "conclusion": "C3 numeric alternative only; H5 free-to-mean formula matches by latitude, while its CASALS tide-system meaning and applied state remain undocumented."})

        # Reconstruct selected project C7 on the fixed target sample and decompose it.
        frame_df, frame_detail, frame_values = _run_frame_audit(granule, sample, fields)
        frame_rows_all.append(frame_df)
        # For associations use the same fixed sample with per-pulse frame-derived geoid18.
        association_rows.extend(_correlation_rows(granule, sample, fields, frame_values))

        # Geoid comparison uses the exact common 0.1%-trajectory sample already used
        # by the project GEOID18 audit, evaluated against the official NGA EGM grid.
        gm = geoid_sample.loc[geoid_sample.granule_id == granule].copy()
        sample_ids = gm.pulse_index.to_numpy(np.int64)
        if sample_ids.size and np.any((sample_ids < 0) | (sample_ids >= n)):
            raise ValueError("GEOID18 sample pulse IDs fall outside H5")
        h5_geoid = fields["geoid"][sample_ids]
        if not np.allclose(h5_geoid, gm.h5_geoid_field_m.to_numpy(float), atol=2e-5, rtol=0):
            raise RuntimeError("existing GEOID18 sample's H5 geoid differs at its recorded pulse IDs")
        egm = bilinear_grid(egm_grid, gm.longitude_deg_assumed.to_numpy(float),
                            gm.latitude_deg_assumed.to_numpy(float))
        noaa = gm.noaa_geoid18_undulation_m.to_numpy(float)
        pair_distributions = geoid_model_comparison(h5_geoid, egm, noaa)
        pairs = {name: h5_geoid - egm if name == "H5 minus EGM2008" else
                 h5_geoid - noaa if name == "H5 minus GEOID18" else egm - noaa
                 for name in pair_distributions}
        for name, diff in pairs.items():
            d = pair_distributions[name]
            geoid_rows.append({"granule_id": granule, "scope": "geoid18_common_sample",
                               "comparison": name, "n": d["n_valid"],
                               "mean_difference_m": d["mean"], "median_difference_m": d["median"],
                               "P01_difference_m": d["P01"], "P05_difference_m": d["P05"],
                               "P95_difference_m": d["P95"], "P99_difference_m": d["P99"],
                               "minimum_difference_m": d["minimum"], "maximum_difference_m": d["maximum"],
                               "coordinate_source": "existing datum_comparison/geoid18_sample_comparison.csv",
                               "H5_units_status": "UNKNOWN", "EGM_units": "metres",
                               "GEOID18_units": "metres"})
        formula_ref = tide_free_to_mean_formula(gm.latitude_deg_assumed.to_numpy(float))
        formula_difference = float(np.nanmedian(np.abs(fields["geoid_free2mean"][sample_ids] - formula_ref)))
        geoid_model_evidence[granule] = {
            "common_sample_n": int(len(gm)), "coordinate_provenance": str(GEOID18_SAMPLE.relative_to(ROOT)),
            "h5_egm2008_difference": distribution(pairs["H5 minus EGM2008"]),
            "h5_geoid18_difference": distribution(pairs["H5 minus GEOID18"]),
            "egm2008_geoid18_difference": distribution(pairs["EGM2008 minus GEOID18"]),
            "model_match_status": "NUMERICALLY_SUPPORTED" if np.nanpercentile(np.abs(pairs["H5 minus EGM2008"]), 95) < .02 else "UNKNOWN",
            "interpretation": "Model agreement, if within grid interpolation tolerance, is numerical identification only; H5 lacks source-model metadata.",
            "c1_c7_median_difference_m": float(np.nanmedian(sample.C1_residual_m - sample.C7_residual_m)),
            "geoid_free2mean_formula": "0.1287 - 0.3848*sin(latitude)^2 (numeric metres formula only)",
            "free2mean_formula_median_abs_difference": formula_difference,
        }
        # Preserve C7 operation detail in JSON for each exact granule.
        metadata["granules"][granule]["c7_operation_audit"] = frame_detail

    # Global correction matrix and evidence registry.
    pd.DataFrame(field_inventory_rows).to_csv(output / "correction_field_inventory.csv", index=False)
    pd.DataFrame(statistic_rows).to_csv(output / "correction_statistics.csv", index=False)
    pd.DataFrame(_matrix_rows()).to_csv(output / "correction_application_matrix.csv", index=False)
    pd.DataFrame(_evidence_rows()).to_csv(output / "correction_evidence_registry.csv", index=False)
    (output / "h5_processing_metadata.json").write_text(json.dumps(metadata, indent=2, default=json_value), encoding="utf-8")
    pd.DataFrame(atmospheric_rows).to_csv(output / "atmospheric_delay_diagnostics.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(output / "range_to_height_sensitivity.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output / "range_timing_audit.csv", index=False)
    geoid_frame = pd.DataFrame(geoid_rows)
    geoid_frame.to_csv(output / "geoid_model_comparison.csv", index=False)
    (output / "correction_time_trends.csv").write_text(pd.DataFrame(time_trend_rows).to_csv(index=False), encoding="utf-8")
    (output / "geoid_model_evidence.json").write_text(json.dumps({
        "source_archive": str(NGA_ZIP.relative_to(ROOT)), "source_url": "https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84",
        "archive_sha256": NGA_SHA256, "grid_member": NGA_MEMBER,
        "grid_shape": [NGA_ROWS, NGA_COLS], "spacing_arc_minutes": 2.5,
        "tide_system": "WGS84 Tide-Free (NGA archive documentation)",
        "sample_interpolation_method": "bilinear on 2.5 arc-minute global grid; Fortran sequential-record markers validated",
        "official_sample_interpolation_validation": egm_sample_validation,
        "per_granule": geoid_model_evidence,
    }, indent=2, default=json_value), encoding="utf-8")
    frames = pd.concat(frame_rows_all, ignore_index=True)
    frames.to_csv(output / "reference_frame_operation_audit.csv", index=False)
    hypotheses = pd.DataFrame(hypothesis_rows)
    hypotheses.to_csv(output / "physical_hypothesis_comparison.csv", index=False)
    pd.DataFrame(attribution_rows).to_csv(output / "bias_attribution_summary.csv", index=False)
    pd.DataFrame(association_rows).to_csv(output / "residual_context_associations.csv", index=False)

    stats_frame = pd.DataFrame(statistic_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    _make_figures(stats_frame, hypotheses, geoid_frame, sensitivity_frame, frames, output)

    c8_path = OUT.parent / "visual_revision/ground_c8_calibration_support_gate.csv"
    c8_gate = pd.read_csv(c8_path).to_dict(orient="records") if c8_path.exists() else []
    summary = {
        "scope": "read-only correction audit; no C0-C8, datum_comparison, ground_aware, revised, production, H5, or LAZ data changed",
        "granules": metadata["granules"],
        "correction_application_statuses": {row["correction"]: row["applied_status"] for row in _matrix_rows()},
        "c8_support_gate": c8_gate,
        "egm2008_evidence": geoid_model_evidence,
        "frame_operation": {g: metadata["granules"][g]["c7_operation_audit"] for g in GRANULES},
        "common_independent_test_metrics": hypothesis_rows,
        "limitations": [
            "No field-level H5 units, sign conventions, applied flags, CASALS processor version, source realization, or coordinate epoch are present.",
            "Atmospheric and range height effects are conditional physical sensitivities; unknown applied state creates double-correction risk.",
            "C7's numerical no-op frame step cannot verify the physical source realization; the selected operation's stated accuracy is 2.015 m overall.",
            "Ground-target results apply only to the frozen supported independent-test subset and do not certify absolute accuracy.",
        ],
    }
    (output / "correction_audit_summary.json").write_text(json.dumps(summary, indent=2, default=json_value), encoding="utf-8")
    artifact_files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "audit_manifest.json")
    manifest = {
        "generated_from": "research/reference/refh_correction_audit.py",
        "source_h5_sha256": {g: metadata["granules"][g]["sha256"] for g in GRANULES},
        "protected_reference_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in
            (TARGET_COMMON, TARGET_TEST, GEOID18_SAMPLE,
             OUT.parent / "datum_comparison/selected_vertical_configuration.json",
             c8_path) if p.exists()
        },
        "artifacts": {str(p.relative_to(output)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                      for p in artifact_files},
        "NGA_archive_sha256": sha256(NGA_ZIP),
    }
    (output / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    summary = run_audit(args.output)
    print(json.dumps({"output": str(args.output), "granules": list(summary["granules"]),
                      "hypotheses": len(summary["common_independent_test_metrics"])}, indent=2))


if __name__ == "__main__":
    main()
