"""Evidence-bounded CASALS vertical-bias attribution helpers and report builder.

The module deliberately keeps numeric sensitivities separate from processing
evidence. It does not alter the production processor or historical audit data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "outputs/notebooks/10_3dep_elevation_validation/correction_audit"
GROUND_DIR = ROOT / "outputs/notebooks/10_3dep_elevation_validation/ground_aware"
DEFAULT_OUTPUT = ROOT / "outputs/notebooks/10_3dep_elevation_validation/final_bias_attribution"
GRANULES = (
    "casals_l1b_20241112T165718_001_02",
    "casals_l1b_20241118T171757_001_02",
)
CASALS_BUCKET_URL = "https://casals-data.s3.us-west-2.amazonaws.com"
ARCO_ROOT = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_DOC = "https://github.com/google-research/arco-era5/blob/main/README.md"
C = 299_792_458.0
NS_TO_ROUNDTRIP_RANGE_M = C / 2.0 * 1e-9
NS_TO_ONEWAY_PATH_M = C * 1e-9
EARTH_G = 9.80665
NADIR_WAVELENGTH_UM = 1.040  # nominal airborne CASALS wavelength; not per-pulse H5 metadata


def range_effect_from_delay(value: Any, convention: str) -> np.ndarray:
    """Convert candidate stored delay conventions to one-way range effect (m)."""
    x = np.asarray(value, dtype=float)
    if convention in {"m_one_way_range", "m_one_way_optical_path"}:
        return x.copy()
    if convention == "ns_two_way_time":
        return x * NS_TO_ROUNDTRIP_RANGE_M
    if convention == "ns_one_way_time":
        return x * NS_TO_ONEWAY_PATH_M
    raise ValueError(f"unsupported delay convention: {convention}")


def removed_excess_height_effect(excess_range_m: Any, beam_elevation_rad: Any) -> np.ndarray:
    """Height change when positive excess range is removed from a downward beam."""
    distance, elevation = np.broadcast_arrays(
        np.asarray(excess_range_m, dtype=float), np.asarray(beam_elevation_rad, dtype=float)
    )
    if np.any((elevation <= 0) | (elevation > np.pi / 2)):
        raise ValueError("beam elevation must be in (0, pi/2] radians above the horizon")
    return distance * np.sin(elevation)


def target_xyz_from_range(sensor_xyz_m: Any, beam_unit_from_sensor: Any,
                         geometric_range_m: Any) -> np.ndarray:
    """Reconstruct Cartesian target position from independent range geometry."""
    sensor = np.asarray(sensor_xyz_m, dtype=float)
    beam = np.asarray(beam_unit_from_sensor, dtype=float)
    distance = np.asarray(geometric_range_m, dtype=float)
    if sensor.shape[-1:] != (3,) or beam.shape[-1:] != (3,):
        raise ValueError("sensor and beam vectors must have a final dimension of 3")
    sensor, beam = np.broadcast_arrays(sensor, beam)
    norm = np.linalg.norm(beam, axis=-1)
    if np.any(~np.isfinite(sensor)) or np.any(~np.isfinite(beam)) or np.any(norm == 0):
        raise ValueError("sensor and beam vectors must be finite; beam direction cannot be zero")
    if np.any(~np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError("geometric range must be finite and nonnegative")
    unit = beam / norm[..., None]
    return sensor + unit * distance[..., None]


def source_frame_status(operation_is_noop: bool, source_realization: str | None,
                        coordinate_epoch: str | None) -> str:
    """A numeric no-op cannot establish a trajectory's physical realization."""
    if source_realization and coordinate_epoch:
        return "SOURCE_FRAME_PROVENANCE_AVAILABLE"
    if operation_is_noop:
        return "UNRESOLVED_SOURCE_FRAME_NUMERIC_NOOP"
    return "UNRESOLVED_SOURCE_FRAME"


def compare_refh_application_state(
    processed_range_m: float,
    independent_raw_range_m: float | None,
    atmospheric_excess_m: float | None,
    *,
    calibration_bias_m: float | None = None,
    tolerance_m: float = 0.01,
) -> dict[str, Any]:
    """Compare applied/unapplied range models only when raw inputs are independent.

    ``calibration_bias_m`` is signed as an additive term in the raw measured
    range. Without independent raw ranging or a known calibration term, status
    remains unidentifiable even when processed Refh and a delay field exist.
    """
    vals = (processed_range_m, independent_raw_range_m, atmospheric_excess_m,
            calibration_bias_m)
    if any(v is None or not np.isfinite(v) for v in vals):
        return {"status": "APPLICATION_STATE_NOT_IDENTIFIABLE_FROM_AVAILABLE_INPUTS",
                "unapplied_error_m": None, "applied_error_m": None}
    raw = float(independent_raw_range_m) - float(calibration_bias_m)
    unapplied_error = float(processed_range_m) - raw
    applied_error = float(processed_range_m) - (raw - float(atmospheric_excess_m))
    if abs(unapplied_error) <= tolerance_m and abs(applied_error) > tolerance_m:
        status = "CONFIRMED_NOT_APPLIED"
    elif abs(applied_error) <= tolerance_m and abs(unapplied_error) > tolerance_m:
        status = "CONFIRMED_APPLIED"
    else:
        status = "APPLICATION_STATE_NOT_IDENTIFIABLE_FROM_AVAILABLE_INPUTS"
    return {"status": status, "unapplied_error_m": unapplied_error,
            "applied_error_m": applied_error}


def eligible_independent_test(population: pd.DataFrame, *, snr_min: float = 5.0) -> pd.DataFrame:
    """Select frozen eligible, common-support independent-test rows without residuals."""
    required = {"research_role", "is_ground_target_eligible", "common_support_all_candidates",
                "refh_snr", "pulse_index"}
    missing = required - set(population.columns)
    if missing:
        raise ValueError(f"population missing selection fields: {sorted(missing)}")
    selected = population.loc[
        population.research_role.astype(str).eq("independent_test")
        & population.is_ground_target_eligible.fillna(False).astype(bool)
        & population.common_support_all_candidates.fillna(False).astype(bool)
        & (pd.to_numeric(population.refh_snr, errors="coerce") >= snr_min)
    ].copy()
    if selected.pulse_index.duplicated().any():
        raise ValueError("selected pulse IDs are not unique")
    return selected.sort_values([c for c in ("granule_id", "pulse_index") if c in selected]).reset_index(drop=True)


def select_prospective_granules(metadata_rows: list[dict[str, Any]],
                                existing_dates: set[str]) -> list[dict[str, Any]]:
    """Apply metadata-only prospective gates before any residual inspection."""
    accepted = []
    for row in metadata_rows:
        acq_date = str(row.get("acquisition_date", ""))
        has_scene = bool(row.get("scene_evidence", False))
        has_ground = bool(row.get("eligible_ground_site_evidence", False))
        if acq_date in existing_dates:
            reason = "NO_NEW_ACQUISITION_DATE"
        elif not has_scene:
            reason = "SCENE_EVIDENCE_NOT_REVIEWED"
        elif not has_ground:
            reason = "NO_QUALIFIED_STABLE_GROUND_SITE"
        else:
            reason = "PROSPECTIVE_METADATA_ELIGIBLE"
            accepted.append(row)
        row["prospective_selection_status"] = reason
    return accepted


def _ciddor_group_refractivity(lam_um: float, temp_c: Any, pressure_hpa: Any,
                               vapor_pressure_pa: Any, co2_ppm: float = 420.0) -> np.ndarray:
    """Ciddor group refractivity (n_g-1), from the Ciddor 1996 formulation.

    This mirrors the published group-refractivity equations used by GeodePy;
    temperature is Celsius, pressure is hPa, vapor partial pressure is Pa.
    """
    tc, p_hpa, pv_pa = np.broadcast_arrays(np.asarray(temp_c, float),
                                            np.asarray(pressure_hpa, float),
                                            np.asarray(vapor_pressure_pa, float))
    p = p_hpa * 100.0
    pv = pv_pa
    tk = tc + 273.15
    sigma2 = (1.0 / lam_um) ** 2
    k0, k1, k2, k3 = 238.0185, 5_792_105.0, 57.362, 167_917.0
    w0, w1, w2, w3 = 295.235, 2.6422, -0.032380, 0.004028
    alpha, beta, gamma = 1.00062, 3.14e-8, 5.6e-7
    a0, a1, a2 = 1.58123e-6, -2.9331e-8, 1.1043e-10
    b0, b1, c0, c1 = 5.707e-6, -2.051e-8, 1.9898e-4, -2.376e-6
    dlc, e0 = 1.83e-11, -0.765e-8
    cf, gas_r, molar_w = 1.022, 8.314510, 0.018015
    t_std_air, p_std_air = 288.15, 101325.0
    t_std_wv, p_std_wv = 293.15, 1333.0

    ngas = k1 * ((k0 + sigma2) / (k0 - sigma2) ** 2) + k3 * ((k2 + sigma2) / (k2 - sigma2) ** 2)
    ngas *= 1.0 + 0.534e-6 * (co2_ppm - 450.0)
    ngws = cf * (w0 + 3 * w1 * sigma2 + 5 * w2 * sigma2**2 + 7 * w3 * sigma2**3)
    molar_d = (28.9635 + 12.011e-6 * (co2_ppm - 400.0)) * 1e-3
    z_std_air = 1.0 - (p_std_air / t_std_air) * (a0 + a1 * 15.0 + a2 * 15.0**2) + (p_std_air / t_std_air) ** 2 * dlc
    rho_std_air = p_std_air * molar_d / (z_std_air * gas_r * t_std_air)
    z_std_wv = 1.0 - (p_std_wv / t_std_wv) * (
        a0 + a1 * 20.0 + a2 * 20.0**2 + (b0 + b1 * 20.0) + (c0 + c1 * 20.0)
    ) + (p_std_wv / t_std_wv) ** 2 * (dlc + e0)
    rho_std_wv = p_std_wv * molar_w / (z_std_wv * gas_r * t_std_wv)
    enhance = alpha + beta * p + gamma * tc**2
    x_v = enhance * pv / p
    z = 1.0 - (p / tk) * (
        a0 + a1 * tc + a2 * tc**2 + (b0 + b1 * tc) * x_v
        + (c0 + c1 * tc) * x_v**2
    ) + (p / tk) ** 2 * (dlc + e0 * x_v**2)
    rho_air = p * molar_d * (1.0 - x_v) / (z * gas_r * tk)
    rho_wv = p * molar_w * x_v / (z * gas_r * tk)
    return ((rho_air / rho_std_air) * ngas + (rho_wv / rho_std_wv) * ngws) * 1e-8


def _vapor_pressure_from_specific_humidity(q: Any, pressure_pa: Any,
                                           temp_k: Any, co2_ppm: float = 420.0) -> np.ndarray:
    q, pressure, temp_c = np.broadcast_arrays(np.asarray(q, float), np.asarray(pressure_pa, float),
                                               np.asarray(temp_k, float) - 273.15)
    molar_d = (28.9635 + 12.011e-6 * (co2_ppm - 400.0)) * 1e-3
    molar_w = 0.018015
    x_v = (q / molar_w) / ((q / molar_w) + ((1.0 - q) / molar_d))
    enhance = 1.00062 + 3.14e-8 * pressure + 5.6e-7 * temp_c**2
    return x_v * pressure / enhance


def integrate_atmospheric_profile(level_hpa: Any, temperature_k: Any, specific_humidity: Any,
                                  geopotential_m2s2: Any, surface_pressure_pa: float,
                                  ground_height_m: float, instrument_height_m: float,
                                  beam_elevation_rad: float, *, wavelength_um: float = NADIR_WAVELENGTH_UM,
                                  co2_ppm: float = 420.0) -> dict[str, float]:
    """Integrate the Ciddor group refractivity from surface to aircraft.

    Pressure-level ERA5 fields are filtered to levels above the model surface.
    Temperature, humidity and log pressure are linearly interpolated in
    geopotential height to the ground and instrument endpoints. A plane-parallel
    slant mapping 1/sin(elevation) is appropriate here only because the audited
    CASALS beams are near nadir; it is retained as a stated approximation.
    """
    plev, temp, q, geop = np.broadcast_arrays(np.asarray(level_hpa, float),
                                               np.asarray(temperature_k, float),
                                               np.asarray(specific_humidity, float),
                                               np.asarray(geopotential_m2s2, float))
    z = geop / EARTH_G
    keep = (plev * 100.0 < surface_pressure_pa) & np.isfinite(z) & np.isfinite(temp) & np.isfinite(q)
    plev, temp, q, z = plev[keep], temp[keep], q[keep], z[keep]
    order = np.argsort(z)
    plev, temp, q, z = plev[order], temp[order], q[order], z[order]
    if len(z) < 3 or not ground_height_m < instrument_height_m:
        raise ValueError("profile lacks enough valid levels or invalid endpoint order")
    if ground_height_m < z[0] - 1500 or instrument_height_m > z[-1] + 1500:
        raise ValueError("profile does not bracket the surface-to-instrument path")

    def interp(y: np.ndarray, height: float) -> float:
        return float(np.interp(height, z, y))

    inside = (z > ground_height_m) & (z < instrument_height_m)
    heights = np.concatenate(([ground_height_m], z[inside], [instrument_height_m]))
    temps = np.interp(heights, z, temp)
    qs = np.clip(np.interp(heights, z, q), 0.0, 0.05)
    logp = np.interp(heights, z, np.log(plev * 100.0))
    pressures = np.exp(logp)
    # Ground pressure comes from ERA5 surface pressure; the profile pressure at
    # the endpoint is replaced while T/q use the lowest valid pressure levels.
    pressures[0] = surface_pressure_pa
    pv = _vapor_pressure_from_specific_humidity(qs, pressures, temps, co2_ppm)
    n_excess = _ciddor_group_refractivity(wavelength_um, temps - 273.15,
                                         pressures / 100.0, pv, co2_ppm)
    zenith_path = float(np.sum(np.diff(heights) * (n_excess[:-1] + n_excess[1:]) * 0.5))
    slant_path = zenith_path / math.sin(beam_elevation_rad)
    surface_index = float(n_excess[0])
    return {
        "zenith_one_way_group_excess_path_m": zenith_path,
        "slant_one_way_group_excess_path_m": slant_path,
        "surface_group_refractivity_m_per_m": surface_index,
        "instrument_group_refractivity_m_per_m": float(n_excess[-1]),
        "path_height_m": float(instrument_height_m - ground_height_m),
        "wavelength_um": float(wavelength_um),
    }


def _retry_session() -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retries = Retry(total=7, connect=7, read=7, status=7, backoff_factor=0.7,
                    status_forcelist=(429, 500, 502, 503, 504), allowed_methods={"GET", "HEAD"})
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _download_object(url: str, *, block_bytes: int = 16 * 1024 * 1024) -> tuple[bytes, dict[str, str]]:
    """Read a public GCS object using bounded HTTP ranges and record provenance."""
    session = _retry_session()
    head = session.head(url, timeout=(20, 90))
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    content = bytearray()
    for start in range(0, size, block_bytes):
        end = min(start + block_bytes, size) - 1
        response = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 180))
        if response.status_code != 206:
            raise RuntimeError(f"range fetch failed ({response.status_code}) for {url}")
        if len(response.content) != end - start + 1:
            raise RuntimeError(f"short range fetch for {url} at {start}-{end}")
        content.extend(response.content)
    return bytes(content), {
        "url": url, "size_bytes": str(size),
        "etag": head.headers.get("ETag", ""),
        "generation": head.headers.get("x-goog-generation", ""),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def fetch_arco_profile(time_index: int, latitude_deg: float, longitude_deg: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch one bilinearly sampled ERA5 pressure-level profile from public ARCO."""
    import numcodecs  # optional runtime helper; only needed for profile data retrieval

    metadata_url = f"{ARCO_ROOT}/.zmetadata"
    metadata_bytes, metadata_source = _download_object(metadata_url)
    consolidated = json.loads(metadata_bytes)
    meta = consolidated["metadata"]

    def decode_chunk(variable: str, key: str) -> tuple[np.ndarray, dict[str, str]]:
        url = f"{ARCO_ROOT}/{variable}/{key}"
        raw, source = _download_object(url)
        arr_meta = meta[f"{variable}/.zarray"]
        decoded = numcodecs.get_codec(arr_meta["compressor"]).decode(raw)
        array = np.frombuffer(decoded, dtype=np.dtype(arr_meta["dtype"]))
        array = array.reshape(arr_meta["chunks"], order=arr_meta["order"])
        if array.ndim >= 3 and array.shape[0] == 1:
            array = array[0]
        return array, source

    levels, level_source = decode_chunk("level", "0")
    latitudes, lat_source = decode_chunk("latitude", "0")
    longitudes, lon_source = decode_chunk("longitude", "0")
    # ERA5 latitude is north-to-south; longitude is eastward 0..359.75 degrees.
    lon = longitude_deg % 360.0
    r = (90.0 - latitude_deg) / 0.25
    c = lon / 0.25
    r0, c0 = int(math.floor(r)), int(math.floor(c)) % 1440
    r1, c1 = min(r0 + 1, 720), (c0 + 1) % 1440
    fr, fc = r - r0, c - math.floor(c)

    def sample(arr: np.ndarray, has_level: bool) -> np.ndarray:
        if has_level:
            corners = np.stack((arr[:, r0, c0], arr[:, r0, c1],
                                arr[:, r1, c0], arr[:, r1, c1]), axis=-1)
        else:
            corners = np.asarray([arr[r0, c0], arr[r0, c1], arr[r1, c0], arr[r1, c1]])
        return ((1-fr) * ((1-fc) * corners[..., 0] + fc * corners[..., 1])
                + fr * ((1-fc) * corners[..., 2] + fc * corners[..., 3]))

    profile: dict[str, Any] = {
        "level_hpa": np.asarray(levels, float),
        "latitude_grid": np.asarray(latitudes, float),
        "longitude_grid": np.asarray(longitudes, float),
    }
    sources = [metadata_source, level_source, lat_source, lon_source]
    for variable, has_level in (("temperature", True), ("specific_humidity", True),
                                ("geopotential", True), ("surface_pressure", False)):
        suffix = ".0.0.0" if has_level else ".0.0"
        array, source = decode_chunk(variable, f"{time_index}{suffix}")
        profile[variable] = sample(array, has_level)
        sources.append(source)
        del array
    profile["surface_pressure"] = float(profile["surface_pressure"])
    profile["profile_time_index"] = int(time_index)
    profile["latitude_deg"] = float(latitude_deg)
    profile["longitude_deg"] = float(longitude_deg)
    profile["grid_cells"] = {"latitude_index_pair": [r0, r1], "longitude_index_pair": [c0, c1],
                              "latitude_weights": [1-fr, fr], "longitude_weights": [1-fc, fc]}
    return profile, {"dataset_metadata": metadata_source, "chunks": sources}


def _list_casals_l1b_inventory() -> list[dict[str, Any]]:
    """List S3 metadata only; never downloads object content."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retries = Retry(total=6, connect=6, read=6, status=6, backoff_factor=0.5,
                    status_forcelist=(429, 500, 502, 503, 504), allowed_methods={"GET"})
    session.mount("https://", HTTPAdapter(max_retries=retries))
    params: dict[str, str] = {"list-type": "2", "prefix": "lidar/", "max-keys": "1000"}
    all_objects: list[dict[str, Any]] = []
    while True:
        response = session.get(CASALS_BUCKET_URL, params=params, timeout=(20, 90))
        response.raise_for_status()
        root = ET.fromstring(response.content)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for item in root.findall("s3:Contents", ns):
            key = item.findtext("s3:Key", default="", namespaces=ns)
            if key.lower().endswith(".h5"):
                all_objects.append({
                    "object_key": key,
                    "size_bytes": int(item.findtext("s3:Size", default="0", namespaces=ns)),
                    "last_modified_utc": item.findtext("s3:LastModified", default="", namespaces=ns),
                    "etag": item.findtext("s3:ETag", default="", namespaces=ns).strip('"'),
                })
        token = root.findtext("s3:NextContinuationToken", default="", namespaces=ns)
        if not token:
            break
        params["continuation-token"] = token
    return all_objects


def _save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
                    encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_artifact_hash_manifest(output: Path) -> pd.DataFrame:
    manifest = output / "artifact_sha256_manifest.csv"
    rows = [
        {"relative_path": path.relative_to(output).as_posix(),
         "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != manifest
    ]
    result = pd.DataFrame(rows)
    result.to_csv(manifest, index=False)
    return result


def _decode_h5_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _h5_field_map(handle: h5py.File) -> dict[str, h5py.Dataset]:
    result: dict[str, h5py.Dataset] = {}
    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            key = name.rsplit("/", 1)[-1]
            if key in result:
                raise ValueError(f"ambiguous H5 field basename: {key}")
            result[key] = obj
    handle.visititems(visit)
    return result


def _repo_file_presence(name: str) -> list[str]:
    data_root = ROOT / "data"
    return [str(path.relative_to(ROOT)) for path in data_root.rglob(name) if path.is_file()]


def _profile_timestamp_index(h5_attrs: dict[str, str]) -> tuple[str, int]:
    timestamp = datetime.fromisoformat(h5_attrs["start_utca"].replace("Z", "+00:00"))
    nearest = timestamp.replace(minute=0, second=0, microsecond=0)
    if (timestamp - nearest).total_seconds() >= 1800:
        nearest += timedelta(hours=1)
    delta_hours = (nearest - datetime(1940, 1, 1, tzinfo=timezone.utc)).total_seconds() / 3600.0
    return nearest.isoformat().replace("+00:00", "Z"), int(round(delta_hours))


def _metric_row(group: pd.DataFrame, scenario: str, residual_col: str) -> dict[str, Any]:
    from research.reference.refh_correction_audit import residual_metrics
    values = group[residual_col].to_numpy(dtype=float)
    result = residual_metrics(values)
    valid = values[np.isfinite(values)]
    result["coverage_within_1m"] = float(np.mean(np.abs(valid) <= 1.0)) if len(valid) else np.nan
    result["hypothesis"] = scenario
    result["pulse_id_sha256"] = hashlib.sha256(
        np.asarray(group["pulse_index"], dtype="<i8").tobytes()
    ).hexdigest()
    return result


def _make_figures(output: Path, profile_frames: dict[str, pd.DataFrame],
                  independent_rows: list[dict[str, Any]], residuals: pd.DataFrame,
                  decision: dict[str, Any]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {GRANULES[0]: "#2C6E9B", GRANULES[1]: "#C05A32"}

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    for granule, frame in profile_frames.items():
        ax.plot(frame["group_refractivity_m_per_m"] * 1e6, frame["height_m"],
                marker="o", ms=3.5, lw=1.8, color=colors[granule],
                label="Nov 12, 2024" if "20241112" in granule else "Nov 18, 2024")
    ax.set(xlabel="Ciddor group refractivity (µm/m)", ylabel="ERA5 geopotential height (m)",
           title="Independent ERA5 pressure-level profiles")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.22)
    fig.text(0.02, 0.02,
             "0.25-degree hourly reanalysis; nominal 1040 nm. H5 and 3DEP vertical reference mismatch remains.\n"
             "This profile checks atmospheric scale; it does not establish CASALS correction application.",
             fontsize=8, color="#444444")
    fig.savefig(figures / "atmospheric_physics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.axis("off")
    ax.set_title("Raw Refh reconstruction: unavailable from supplied inputs", loc="left", fontsize=14)
    paragraphs = [
        "Model A | Raw TX/RX timing + independent trajectory + calibration produce the no-atmosphere Refh model.",
        "Model B | The same raw inputs plus the documented atmospheric range correction produce the corrected Refh model.",
        "Inputs absent locally: referenced L1A HDF5, matched GPS/geolocation products, processor log/config, original time-zero and clock calibration, and independent range calibration.",
        "The processed Refh and stored neutral-delay field are shared descendants. Comparing them cannot identify whether the processor applied the correction.",
        "Status: APPLICATION_STATE_NOT_IDENTIFIABLE_FROM_AVAILABLE_INPUTS.",
    ]
    content = "\n\n".join(textwrap.fill(p, width=82) for p in paragraphs)
    ax.text(0.04, 0.91, content, va="top", ha="left", fontsize=9.3, linespacing=1.35,
            family="DejaVu Sans", transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.02, 0.08), 0.96, 0.84, fill=False, lw=1.2,
                                edgecolor="#9B3B2A", transform=ax.transAxes))
    fig.savefig(figures / "refh_range_reconstruction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    scenario_order = ["C7_baseline", "H1_stored_delay_as_m", "H1_ERA5_profile_sensitivity"]
    labels = {"C7_baseline": "C7 baseline", "H1_stored_delay_as_m": "H1 stored field as m",
              "H1_ERA5_profile_sensitivity": "ERA5 profile sensitivity"}
    granules = list(residuals["granule_id"].drop_duplicates())
    xbase = np.arange(len(granules), dtype=float)
    width = 0.24
    for j, scenario in enumerate(scenario_order):
        medians, lows, highs = [], [], []
        col = {"C7_baseline": "C7_residual_m", "H1_stored_delay_as_m": "H1_stored_as_m_residual_m",
               "H1_ERA5_profile_sensitivity": "H1_era5_profile_residual_m"}[scenario]
        for granule in granules:
            vals = residuals.loc[residuals.granule_id.eq(granule), col].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            median = float(np.median(vals))
            medians.append(median)
            lows.append(float(np.percentile(vals, 25)))
            highs.append(float(np.percentile(vals, 75)))
        xpos = xbase + (j - 1) * width
        ax.errorbar(xpos, medians, yerr=[np.asarray(medians)-lows, np.asarray(highs)-medians],
                    fmt="o", capsize=3, lw=1.2, color=["#455A64", "#B24A3B", "#2C6E9B"][j],
                    label=labels[scenario])
    ax.axhline(0, color="#777777", lw=0.9)
    ax.set_xticks(xbase, ["Nov 12, 2024", "Nov 18, 2024"][:len(granules)])
    ax.set_ylabel("Median C7-to-3DEP residual (m); bars show pulse IQR")
    ax.set_title("Frozen SNR >= 5 independent-test pulses; conditional sensitivities")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.19)
    fig.text(0.01, 0.02,
             "One historical road site per date. H1 is not verified and these residuals do not certify absolute accuracy.",
             fontsize=8, color="#444444")
    fig.savefig(figures / "cross_granule_bias.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    plot_rows = []
    for granule in granules:
        for scenario, col in (("C7", "C7_residual_m"), ("H1 stored=m", "H1_stored_as_m_residual_m"),
                              ("H1 ERA5", "H1_era5_profile_residual_m")):
            vals = residuals.loc[residuals.granule_id.eq(granule), col].to_numpy(float)
            from research.reference.refh_correction_audit import residual_metrics
            metrics = residual_metrics(vals)
            plot_rows.append((granule, scenario, metrics["RMSE_m"], metrics["median_bias_m"]))
    markers = {"C7": "o", "H1 stored=m": "s", "H1 ERA5": "^"}
    short_date = lambda g: "Nov 12" if "20241112" in g else "Nov 18"
    label_offsets = {
        ("Nov 12", "C7"): (-30, 10), ("Nov 12", "H1 stored=m"): (9, -16),
        ("Nov 12", "H1 ERA5"): (9, 10), ("Nov 18", "C7"): (9, 9),
        ("Nov 18", "H1 stored=m"): (9, -16), ("Nov 18", "H1 ERA5"): (9, 10),
    }
    for scenario in ("C7", "H1 stored=m", "H1 ERA5"):
        rows = [r for r in plot_rows if r[1] == scenario]
        ax.scatter([r[2] for r in rows], [r[3] for r in rows], s=70,
                   marker=markers[scenario], label=scenario,
                   color="#455A64" if scenario == "C7" else ("#B24A3B" if scenario == "H1 stored=m" else "#2C6E9B"))
        for g, _, rmse, bias in rows:
            date_label = short_date(g)
            short_label = "N12" if date_label == "Nov 12" else "N18"
            ax.annotate(short_label, (rmse, bias),
                        xytext=label_offsets[(date_label, scenario)], textcoords="offset points", fontsize=7)
    ax.set(xlabel="RMSE (m)", ylabel="Median bias (m)",
           title="Residual fit is descriptive; it does not establish processing state")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(figures / "model_discrimination.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    ax.axis("off")
    ax.set_title("Bias budget: no additive attribution is supported", loc="left", fontsize=14)
    paragraphs = [
        "Observed C7 mean residuals: -1.82 m (Nov 12) and -1.57 m (Nov 18); medians: -1.84 m and -1.55 m.",
        "Geoid: H5 resembles EGM2008 numerically; application stage is unknown.",
        "Frame: the selected WGS84 ensemble to NAD83(2011) operation is a numeric no-op; source realization and epoch are unknown.",
        "Atmosphere: ERA5 supports a meter-scale magnitude; H5 units and application status are unknown.",
        "Calibration: stored range_bias_correction is zero; an independent calibration record is absent.",
        "Attribution: components remain jointly unidentified. Do not sum candidates or assign a remainder.",
        "3DEP local-plane NMAD and pulse spread describe sample variability, not absolute vertical accuracy.",
    ]
    content = "\n\n".join(textwrap.fill(p, width=78) for p in paragraphs)
    ax.text(0.035, 0.84, content, va="top", ha="left", fontsize=8.4,
            linespacing=1.18, transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.015, 0.08), 0.97, 0.82, fill=False, lw=1.1,
                                edgecolor="#8B6B27", transform=ax.transAxes))
    fig.savefig(figures / "residual_error_budget.png", dpi=180)
    plt.close(fig)


def build_final_attribution(output: Path = DEFAULT_OUTPUT, *, fetch_atmosphere: bool = True) -> dict[str, Any]:
    """Build the final evidence package from frozen local audits plus public ERA5 metadata/data."""
    from research.reference.refh_correction_audit import (
        add_sensitivity, compare_common_pulses, height_sensitivity, residual_metrics,
    )

    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    audit_stats = pd.read_csv(AUDIT_DIR / "correction_statistics.csv")
    correction_matrix = pd.read_csv(AUDIT_DIR / "correction_application_matrix.csv")
    frame_audit = pd.read_csv(AUDIT_DIR / "reference_frame_operation_audit.csv")
    geoid_audit = pd.read_csv(AUDIT_DIR / "geoid_model_comparison.csv")
    historical = pd.read_parquet(GROUND_DIR / "ground_target_test_population.parquet")
    selected = eligible_independent_test(historical, snr_min=5.0)
    if not selected.groupby("granule_id").size().to_dict() == {GRANULES[0]: 14, GRANULES[1]: 70}:
        raise ValueError("frozen SNR>=5 independent-test selection differs from audit baseline")

    geometry_cache: dict[str, dict[str, np.ndarray]] = {}
    geometry_fields = ("refh_neutat_delay_total", "instrument_longitude", "instrument_latitude",
                       "instrument_altitude", "local_beam_azimuth", "local_beam_elevation")
    for granule in GRANULES:
        ids = selected.loc[selected.granule_id.eq(granule), "pulse_index"].to_numpy(dtype=np.int64)
        h5_path = ROOT / "data/raw/casals_l1b" / f"{granule}.h5"
        with h5py.File(h5_path, "r") as h5:
            fields = _h5_field_map(h5)
            values_by_field: dict[str, np.ndarray] = {}
            for field in geometry_fields:
                ds = fields[field]
                values = np.asarray(ds[ids], dtype=float)
                for attr in ("_FillValue", "missing_value"):
                    if attr in ds.attrs:
                        values[np.isin(values, np.asarray(ds.attrs[attr]).ravel())] = np.nan
                values_by_field[field] = values * float(ds.attrs.get("scale_factor", 1.0)) + float(ds.attrs.get("add_offset", 0.0))
        geometry_cache[granule] = values_by_field

    def stat(granule: str, field: str) -> float:
        if granule in geometry_cache and field in geometry_cache[granule] and field != "refh_neutat_delay_total":
            return float(np.nanmedian(geometry_cache[granule][field]))
        row = audit_stats.loc[(audit_stats.granule_id == granule)
                              & (audit_stats.field == field)
                              & (audit_stats.scope == "full_granule")]
        if len(row) != 1:
            raise ValueError(f"missing unique full-granule statistic: {granule}/{field}")
        return float(row.iloc[0]["median"])

    # Processor and field evidence. Linked intermediate products are checked by exact basename.
    source_rows: list[dict[str, Any]] = [
        {"evidence_id": "NASA_CASALS_DESIGN_2024", "source_type": "official NASA NTRS paper",
         "source": "https://ntrs.nasa.gov/citations/20240001267", "finding": "CASALS paper describes a nominal 1040-nm airborne/mission laser design.",
         "status": "NOMINAL_WAVELENGTH_ONLY", "relevance": "Used as a stated model wavelength; not per-pulse H5 metadata or L1B processing evidence."},
        {"evidence_id": "CRYOCLOUD_L1B_TUTORIAL", "source_type": "public CASALS L1B data tutorial",
         "source": "https://book.cryointhecloud.com/l1b-waveforms-tutorial/", "finding": "Documents public L1B H5 access and Refh as max-amplitude waveform bin.",
         "status": "PRODUCT_FIELD_DESCRIPTION_ONLY", "relevance": "Does not document neutral-delay application, units, sign, or source frame."},
        {"evidence_id": "ARCO_ERA5", "source_type": "public reanalysis source",
         "source": ARCO_DOC, "finding": "Hourly, 0.25-degree ERA5 pressure-level fields used for independent atmospheric scale check.",
         "status": "INDEPENDENT_MODEL_SENSITIVITY", "relevance": "Not CASALS processor evidence."},
    ]
    h5_attrs_by_granule: dict[str, dict[str, str]] = {}
    for granule in GRANULES:
        path = ROOT / "data/raw/casals_l1b" / f"{granule}.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as h5:
            attrs = {key: _decode_h5_attr(value) for key, value in h5.attrs.items()}
            fields = _h5_field_map(h5)
            delay_ds = fields["refh_neutat_delay_total"]
            derivative_ds = fields["refh_neutat_delay_derivative"]
            root_refs = {k: attrs.get(k, "") for k in ("l1a_file", "ard_file", "geoloc_file", "gps_file", "tdms_file")}
            missing_refs = {k: v for k, v in root_refs.items() if v and not _repo_file_presence(v)}
            h5_attrs_by_granule[granule] = attrs
            source_rows.append({
                "evidence_id": f"LOCAL_H5_{granule}", "source_type": "local L1B HDF5",
                "source": str(path.relative_to(ROOT)),
                "finding": (f"Root links L1A/ARD/geolocation/GPS/TDMS names; local exact-name matches missing: "
                            f"{json.dumps(missing_refs, sort_keys=True)}. Neutral-delay field has attrs "
                            f"{json.dumps({k: _decode_h5_attr(v) for k, v in delay_ds.attrs.items()}, sort_keys=True)}; "
                            f"derivative attrs {json.dumps({k: _decode_h5_attr(v) for k, v in derivative_ds.attrs.items()}, sort_keys=True)}."),
                "status": "APPLICATION_STATE_NOT_IDENTIFIABLE_FROM_AVAILABLE_INPUTS",
                "relevance": "No field-level units, sign, applied flag, or processor version/configuration in these H5 files.",
            })

    inventory = _list_casals_l1b_inventory()
    inventory_date = datetime.now(timezone.utc).date().isoformat()
    date_re = __import__("re").compile(r"casals_l1b_(\d{8})T")
    known_dates = {"2024-11-12", "2024-11-18"}
    manifest_rows = []
    for obj in inventory:
        m = date_re.search(Path(obj["object_key"]).name)
        acquisition_date = (f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}" if m else "UNKNOWN")
        status = "NO_NEW_ACQUISITION_DATE" if acquisition_date in known_dates else "NOT_SCENE_REVIEWED"
        manifest_rows.append({
            **obj, "inventory_date_utc": inventory_date, "acquisition_date": acquisition_date,
            "product_level": "L1B_H5", "content_downloaded": False,
            "prospective_selection_status": status,
            "exclusion_reason": ("Only the two historical acquisition dates occur in the public L1B H5 inventory; "
                                 "same-date granules do not satisfy independent cross-date validation." if status == "NO_NEW_ACQUISITION_DATE"
                                 else "No unseen acquisition date with reviewed scene and eligible ground site."),
        })
    pd.DataFrame(manifest_rows).to_csv(output / "validation_granule_manifest.csv", index=False)

    era5_results: dict[str, dict[str, Any]] = {}
    profiles: dict[str, pd.DataFrame] = {}
    era5_sources: dict[str, Any] = {}
    if fetch_atmosphere:
        for granule in GRANULES:
            attrs = h5_attrs_by_granule[granule]
            acq_hour, index = _profile_timestamp_index(attrs)
            lon, lat = stat(granule, "instrument_longitude"), stat(granule, "instrument_latitude")
            profile, provenance = fetch_arco_profile(index, lat, lon)
            matching = selected[selected.granule_id.eq(granule)]
            ground_z = float(matching.reference_height_local_plane_m.median())
            instrument_z = stat(granule, "instrument_altitude")
            elevation = stat(granule, "local_beam_elevation")
            result = integrate_atmospheric_profile(
                profile["level_hpa"], profile["temperature"], profile["specific_humidity"],
                profile["geopotential"], profile["surface_pressure"], ground_z,
                instrument_z, elevation, wavelength_um=NADIR_WAVELENGTH_UM,
            )
            result.update({"granule_id": granule, "profile_time_utc": acq_hour,
                           "profile_time_index": index, "instrument_latitude_deg": lat,
                           "instrument_longitude_deg": lon, "ground_height_navd88_m": ground_z,
                           "instrument_altitude_h5_m": instrument_z,
                           "beam_elevation_rad": elevation,
                           "surface_pressure_pa": profile["surface_pressure"],
                           "grid_cells_json": json.dumps(profile["grid_cells"], sort_keys=True)})
            era5_results[granule] = result
            era5_sources[granule] = provenance
            prof_height = profile["geopotential"] / EARTH_G
            pv = _vapor_pressure_from_specific_humidity(
                profile["specific_humidity"], profile["level_hpa"] * 100,
                profile["temperature"],
            )
            group_n = _ciddor_group_refractivity(
                NADIR_WAVELENGTH_UM, profile["temperature"] - 273.15,
                profile["level_hpa"], pv,
            )
            profiles[granule] = pd.DataFrame({
                "granule_id": granule, "profile_time_utc": acq_hour,
                "latitude_deg": lat, "longitude_deg": lon,
                "pressure_level_hpa": profile["level_hpa"], "height_m": prof_height,
                "temperature_k": profile["temperature"], "specific_humidity_kg_per_kg": profile["specific_humidity"],
                "geopotential_m2_per_s2": profile["geopotential"],
                "surface_pressure_pa": profile["surface_pressure"],
                "group_refractivity_m_per_m": group_n,
            })
    profile_frame = pd.concat(profiles.values(), ignore_index=True) if profiles else pd.DataFrame()
    if not profile_frame.empty:
        profile_frame.to_csv(output / "era5_profile_inputs.csv", index=False)

    # Unit and path-magnitude comparison. All candidates remain conditional because the H5 contract is absent.
    unit_rows: list[dict[str, Any]] = []
    independent_rows: list[dict[str, Any]] = []
    if fetch_atmosphere:
        for granule, result in era5_results.items():
            stored = stat(granule, "refh_neutat_delay_total")
            derivative = stat(granule, "refh_neutat_delay_derivative")
            for convention in ("m_one_way_range", "ns_two_way_time", "ns_one_way_time"):
                converted = float(range_effect_from_delay(stored, convention))
                independent_rows.append({
                    "granule_id": granule, "profile_time_utc": result["profile_time_utc"],
                    "stored_field_median_numeric": stored, "stored_field_units": "UNDECLARED",
                    "candidate_convention": convention, "candidate_range_effect_m": converted,
                    "ERA5_slant_one_way_group_excess_path_m": result["slant_one_way_group_excess_path_m"],
                    "difference_candidate_minus_ERA5_m": converted - result["slant_one_way_group_excess_path_m"],
                    "candidate_over_ERA5_ratio": converted / result["slant_one_way_group_excess_path_m"],
                    "comparison_status": "PHYSICAL_SCALE_COMPARISON_ONLY_NOT_UNIT_OR_APPLICATION_PROOF",
                    "model_assumptions": "ERA5 hourly 0.25-degree pressure levels; Ciddor group index; nominal 1040 nm; plane-parallel slant factor.",
                })
            independent_rows.append({
                "granule_id": granule, "profile_time_utc": result["profile_time_utc"],
                "stored_field_median_numeric": derivative, "stored_field_units": "UNDECLARED",
                "candidate_convention": "refh_neutat_delay_derivative_if_m_per_m",
                "candidate_range_effect_m": np.nan,
                "ERA5_slant_one_way_group_excess_path_m": result["surface_group_refractivity_m_per_m"],
                "difference_candidate_minus_ERA5_m": derivative - result["surface_group_refractivity_m_per_m"],
                "candidate_over_ERA5_ratio": derivative / result["surface_group_refractivity_m_per_m"],
                "comparison_status": "NUMERIC_DERIVATIVE_SCALE_COMPARISON_ONLY",
                "model_assumptions": "Surface group refractivity is the local d(path-excess)/d(target-height) scale for a near-nadir path; derivative semantics remain undocumented.",
            })
            for convention in ("m_one_way_range", "ns_two_way_time", "ns_one_way_time"):
                converted = float(range_effect_from_delay(stored, convention))
                unit_rows.append({
                    "granule_id": granule, "candidate_convention": convention,
                    "stored_median": stored, "converted_range_effect_m": converted,
                    "ERA5_predicted_slant_path_m": result["slant_one_way_group_excess_path_m"],
                    "difference_m": converted - result["slant_one_way_group_excess_path_m"],
                    "ratio_to_ERA5": converted / result["slant_one_way_group_excess_path_m"],
                    "unit_status": "UNKNOWN", "application_status": "UNKNOWN",
                    "double_correction_risk": "YES_IF_STORED_REFH_ALREADY_INCLUDES_FIELD",
                })
            unit_rows.append({
                "granule_id": granule, "candidate_convention": "derivative_m_per_m_if_delay_is_m",
                "stored_median": derivative,
                "converted_range_effect_m": np.nan,
                "ERA5_predicted_slant_path_m": result["surface_group_refractivity_m_per_m"],
                "difference_m": derivative - result["surface_group_refractivity_m_per_m"],
                "ratio_to_ERA5": derivative / result["surface_group_refractivity_m_per_m"],
                "unit_status": "UNKNOWN; numerical slope is compatible with an m/m interpretation",
                "application_status": "UNKNOWN",
                "double_correction_risk": "YES_IF_STORED_REFH_ALREADY_INCLUDES_FIELD",
            })
    pd.DataFrame(unit_rows).to_csv(output / "atmospheric_unit_validation.csv", index=False)
    pd.DataFrame(independent_rows).to_csv(output / "independent_atmospheric_comparison.csv", index=False)

    # Same frozen pulse IDs for all candidate height sensitivities.
    residual_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []
    site_features: list[dict[str, Any]] = []
    for granule in GRANULES:
        subset = selected[selected.granule_id.eq(granule)].copy()
        ids = subset.pulse_index.to_numpy(dtype=np.int64)
        h5_path = ROOT / "data/raw/casals_l1b" / f"{granule}.h5"
        h5_fields = geometry_cache[granule]
        delay = h5_fields["refh_neutat_delay_total"]
        elevation = h5_fields["local_beam_elevation"]
        azimuth = h5_fields["local_beam_azimuth"]
        d_m = range_effect_from_delay(delay, "m_one_way_range")
        d_ns_tw = range_effect_from_delay(delay, "ns_two_way_time")
        d_ns_ow = range_effect_from_delay(delay, "ns_one_way_time")
        h_m = height_sensitivity(d_m, h5_fields["instrument_longitude"], h5_fields["instrument_latitude"], azimuth, elevation)
        h_ns_tw = height_sensitivity(d_ns_tw, h5_fields["instrument_longitude"], h5_fields["instrument_latitude"], azimuth, elevation)
        h_ns_ow = height_sensitivity(d_ns_ow, h5_fields["instrument_longitude"], h5_fields["instrument_latitude"], azimuth, elevation)
        if fetch_atmosphere:
            d_era5 = era5_results[granule]["zenith_one_way_group_excess_path_m"] / np.sin(elevation)
            h_era5 = height_sensitivity(d_era5, h5_fields["instrument_longitude"], h5_fields["instrument_latitude"], azimuth, elevation)
        else:
            h_era5 = np.full(len(subset), np.nan)
        data = subset[["pulse_index", "granule_id", "site_id", "roi_id", "track_num", "refh_snr",
                       "reference_height_local_plane_m", "C7_residual_m", "longitude_deg", "latitude_deg"]].copy()
        data["stored_delay_numeric"] = delay
        data["stored_delay_units"] = "UNDECLARED"
        data["stored_derivative_granule_median"] = stat(granule, "refh_neutat_delay_derivative")
        data["height_effect_if_m"] = h_m
        data["height_effect_if_ns_two_way"] = h_ns_tw
        data["height_effect_if_ns_one_way"] = h_ns_ow
        data["height_effect_ERA5_profile"] = h_era5
        data["H1_stored_as_m_residual_m"] = add_sensitivity(data.C7_residual_m.to_numpy(), h_m)
        data["H1_stored_as_ns_two_way_residual_m"] = add_sensitivity(data.C7_residual_m.to_numpy(), h_ns_tw)
        data["H1_stored_as_ns_one_way_residual_m"] = add_sensitivity(data.C7_residual_m.to_numpy(), h_ns_ow)
        data["H1_era5_profile_residual_m"] = add_sensitivity(data.C7_residual_m.to_numpy(), h_era5)
        data["candidate_status"] = "PHYSICAL_SENSITIVITY_ONLY; APPLICATION_STATE_UNKNOWN"
        residual_frames.append(data)
        for site_id, site_group in data.groupby("site_id", sort=True):
            for hypothesis, col in (
                ("C7_baseline", "C7_residual_m"),
                ("H1_stored_delay_as_m", "H1_stored_as_m_residual_m"),
                ("H1_stored_delay_as_ns_two_way", "H1_stored_as_ns_two_way_residual_m"),
                ("H1_stored_delay_as_ns_one_way", "H1_stored_as_ns_one_way_residual_m"),
                ("H1_ERA5_profile_sensitivity", "H1_era5_profile_residual_m"),
            ):
                row = _metric_row(site_group, hypothesis, col)
                metric_rows.append({"granule_id": granule, "site_id": site_id, "group_type": "site",
                                    **row, "evidence_status": ("PROVISIONAL_C7_BASELINE" if hypothesis == "C7_baseline" else "CONDITIONAL_SENSITIVITY_ONLY"),
                                    "coverage_rule": "fraction |residual| <= 1 m"})
            for track_id, track_group in site_group.groupby("track_num", sort=True):
                for hypothesis, col in (
                    ("C7_baseline", "C7_residual_m"),
                    ("H1_stored_delay_as_m", "H1_stored_as_m_residual_m"),
                    ("H1_ERA5_profile_sensitivity", "H1_era5_profile_residual_m"),
                ):
                    row = _metric_row(track_group, hypothesis, col)
                    metric_rows.append({"granule_id": granule, "site_id": site_id,
                                        "track_num": int(track_id), "group_type": "track_block",
                                        **row, "evidence_status": ("PROVISIONAL_C7_BASELINE" if hypothesis == "C7_baseline" else "CONDITIONAL_SENSITIVITY_ONLY"),
                                        "coverage_rule": "fraction |residual| <= 1 m"})
            ground_status = str(site_group["ground_context_status"].iloc[0]) if "ground_context_status" in site_group else "UNKNOWN"
            if ground_status == "nan":
                ground_status = "UNKNOWN_IN_FROZEN_TABLE"
            historical_site = historical.loc[(historical.granule_id.eq(granule))
                                             & historical.site_id.eq(site_id)
                                             & historical.research_role.astype(str).eq("independent_test")
                                             & historical.is_ground_target_eligible.fillna(False).astype(bool)]
            common_site = historical_site.loc[historical_site.common_support_all_candidates.fillna(False).astype(bool)]
            c7_site_metrics = residual_metrics(site_group.C7_residual_m)
            h1m_site_metrics = residual_metrics(site_group.H1_stored_as_m_residual_m)
            era5_site_metrics = residual_metrics(site_group.H1_era5_profile_residual_m)
            population_rows.append({
                "granule_id": granule, "acquisition_date": h5_attrs_by_granule[granule].get("start_utca", "")[:10],
                "site_id": site_id, "site_role": "historical_independent_test; not prospective",
                "independent_site_count": 1, "ground_context_status": ground_status,
                "ground_eligible_independent_test_pulses": int(len(historical_site)),
                "common_support_pulses": int(len(common_site)),
                "valid_snr_ge_5_pulses": int(len(site_group)),
                "eligible_common_snr_ge_5_pulses": int(len(site_group)),
                "distinct_track_blocks": int(site_group.track_num.nunique()),
                "spatial_block_definition": "distinct CASALS track number within the frozen road ROI",
                "median_stored_delay_numeric": float(np.median(delay)),
                "full_granule_delay_median_numeric": stat(granule, "refh_neutat_delay_total"),
                "ERA5_slant_group_excess_path_m": (era5_results[granule]["slant_one_way_group_excess_path_m"] if fetch_atmosphere else np.nan),
                "C7_mean_bias_m": c7_site_metrics["mean_bias_m"],
                "C7_median_bias_m": c7_site_metrics["median_bias_m"],
                "C7_RMSE_m": c7_site_metrics["RMSE_m"],
                "C7_NMAD_m": c7_site_metrics["NMAD_m"],
                "H1_stored_as_m_mean_bias_m": h1m_site_metrics["mean_bias_m"],
                "H1_stored_as_m_median_bias_m": h1m_site_metrics["median_bias_m"],
                "H1_stored_as_m_RMSE_m": h1m_site_metrics["RMSE_m"],
                "H1_stored_as_m_NMAD_m": h1m_site_metrics["NMAD_m"],
                "H1_ERA5_mean_bias_m": era5_site_metrics["mean_bias_m"],
                "H1_ERA5_median_bias_m": era5_site_metrics["median_bias_m"],
                "H1_ERA5_RMSE_m": era5_site_metrics["RMSE_m"],
                "H1_ERA5_NMAD_m": era5_site_metrics["NMAD_m"],
                "selection_status": "FROZEN_INDEPENDENT_TEST_COMMON_SUPPORT_SNR_GE_5",
            })
            coords = site_group.sort_values("pulse_index")[["longitude_deg", "latitude_deg"]].drop_duplicates()
            if len(coords) >= 2:
                points = coords.iloc[[0, len(coords)//2, -1]][["longitude_deg", "latitude_deg"]].values.tolist()
            elif len(coords) == 1:
                points = [coords.iloc[0].tolist(), coords.iloc[0].tolist()]
            else:
                points = []
            site_features.append({
                "type": "Feature", "geometry": {"type": "LineString", "coordinates": points},
                "properties": {"site_id": site_id, "granule_id": granule,
                               "role": "historical_independent_test_site",
                               "prospective_eligibility": "NOT_PROSPECTIVE; no unseen acquisition date",
                               "geometry_note": "Representative line through frozen supported pulses; not a site boundary."},
            })

    residual_df = pd.concat(residual_frames, ignore_index=True)
    residual_df.to_parquet(output / "physical_hypothesis_residuals.parquet", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "physical_hypothesis_metrics.csv", index=False)
    pd.DataFrame(population_rows).to_csv(output / "validation_population_summary.csv", index=False)
    (output / "validation_site_manifest.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": site_features,
                    "properties": {"prospective_new_sites": 0,
                                   "reason": "Public L1B inventory contains only the two historical acquisition dates; no new-date site can be qualified."}},
                   indent=2), encoding="utf-8")

    # Site/date change is descriptive only; no pulse-level causal regression is used.
    change_rows = []
    compare_metrics = (
        ("Stored H5 neutral delay numeric median", "stored_delay_numeric", lambda g: float(g.stored_delay_numeric.median())),
        ("Full-granule H5 neutral delay median", "full_granule_delay", lambda g: stat(str(g.granule_id.iloc[0]), "refh_neutat_delay_total")),
        ("ERA5 one-way group excess slant path", "era5", lambda g: float(era5_results[g.granule_id.iloc[0]]["slant_one_way_group_excess_path_m"]) if fetch_atmosphere else np.nan),
        ("C7 mean bias", "C7_mean", lambda g: residual_metrics(g.C7_residual_m)["mean_bias_m"]),
        ("C7 median residual", "C7", lambda g: residual_metrics(g.C7_residual_m)["median_bias_m"]),
        ("H1 residual if stored field is meters", "H1_m", lambda g: residual_metrics(g.H1_stored_as_m_residual_m)["median_bias_m"]),
        ("Residual if independent ERA5 path were added", "H1_ERA5", lambda g: residual_metrics(g.H1_era5_profile_residual_m)["median_bias_m"]),
    )
    summary_by_granule = {granule: residual_df.loc[residual_df.granule_id.eq(granule)] for granule in GRANULES}
    for label, key, getter in compare_metrics:
        nov12 = getter(summary_by_granule[GRANULES[0]])
        nov18 = getter(summary_by_granule[GRANULES[1]])
        change_rows.append({
            "metric": label, "nov12_value": nov12, "nov18_value": nov18,
            "nov18_minus_nov12": nov18 - nov12 if np.isfinite(nov12) and np.isfinite(nov18) else np.nan,
            "unit_or_interpretation": "m under candidate conventions; H5 field units undeclared" if key != "C7" else "m, observed sample residual",
            "independent_units": "2 historical sites/granules",
            "causal_status": "INSUFFICIENT_INDEPENDENT_UNITS_FOR_ATTRIBUTION",
            "note": "Different dates, sites, scenes, flight conditions and unresolved frame/calibration; descriptive comparison only.",
        })
    pd.DataFrame(change_rows).to_csv(output / "residual_vs_delay_analysis.csv", index=False)

    # Preserve the exact current C7 operation record and separate numeric model checks from applied status.
    frame_by_granule = {
        granule: frame_audit.loc[frame_audit.granule_id.eq(granule)].to_dict(orient="records")
        for granule in GRANULES
    }
    geoid_by_granule = {
        granule: geoid_audit.loc[geoid_audit.granule_id.eq(granule)].to_dict(orient="records")
        for granule in GRANULES
    }
    frame_evidence = {
        "status": "UNRESOLVED_SOURCE_FRAME",
        "source_frame": "WGS84 ensemble assumed by existing C7 operation; actual GNSS/INS realization and coordinate epoch not established",
        "target_frame": "NAD83(2011) / NAVD88 GEOID18 candidate output",
        "numeric_operation_records": frame_by_granule,
        "geoid_model_comparisons": geoid_by_granule,
        "interpretation": "The selected WGS84-ensemble to NAD83(2011) operation is a numerical no-op. This does not establish that the input trajectory is WGS84 ensemble or quantify its true source-frame offset. H5 geoid values numerically agree with EGM2008 within the sampled comparison, but H5 attributes do not prove geoid model or application stage.",
        "source_realization_epoch_needed": ["GNSS/INS processing software and configuration", "trajectory datum realization", "coordinate reference epoch", "L1B processor CRS declaration"],
    }
    _save_json(output / "source_frame_evidence.json", frame_evidence)

    source_rows.extend([
        {"evidence_id": "PUBLIC_CASALS_BUCKET_LIST", "source_type": "anonymous S3 metadata inventory",
         "source": "https://casals-data.s3.us-west-2.amazonaws.com/?list-type=2&prefix=lidar/",
         "finding": f"{len(inventory)} public H5 objects listed under lidar/; names are L1B and dates are {sorted({r['acquisition_date'] for r in manifest_rows})}. No content downloaded.",
         "status": "NO_CROSS_DATE_PROSPECTIVE_GRANULE", "relevance": "New-acquisition-date validation cannot be selected from the accessible inventory."},
        {"evidence_id": "CASALS_PROCESSOR_SEARCH", "source_type": "public-source search",
         "source": "NASA NTRS / CASALS public documentation and linked CryoCloud tutorial; checked 2026-10-13",
         "finding": "No publicly linked CASALS L1B source, configuration, geolocation ATBD, or processing log was found that states refh_neutat_delay_total units or application to Refh.",
         "status": "NO_DIRECT_PROCESSOR_EVIDENCE_FOUND", "relevance": "Absence of public evidence is not evidence that no correction was applied."},
    ])
    for _, row in correction_matrix.iterrows():
        source_rows.append({
            "evidence_id": f"H5_FIELD_{row.get('correction', 'unknown')}",
            "source_type": "existing read-only correction audit",
            "source": str((AUDIT_DIR / "correction_application_matrix.csv").relative_to(ROOT)),
            "finding": row.get("evidence", ""),
            "status": row.get("applied_status", "UNKNOWN"),
            "relevance": (f"field={row.get('correction')}; units={row.get('units')}; sign={row.get('sign_convention')}; "
                          f"stage={row.get('processing_stage')}; numeric evidence={row.get('numerical_evidence_level')}"),
        })
    for granule in GRANULES:
        for row in geoid_by_granule[granule]:
            source_rows.append({
                "evidence_id": f"GEOID_MODEL_COMPARISON_{granule}_{row.get('comparison', 'unknown')}",
                "source_type": "existing read-only geoid model comparison",
                "source": str((AUDIT_DIR / "geoid_model_comparison.csv").relative_to(ROOT)),
                "finding": f"{row.get('comparison')}: n={row.get('n')}, median={row.get('median_difference_m')} m, units: {row.get('H5_units_status', 'H5 unknown')}",
                "status": "NUMERICALLY_SUPPORTED_ONLY",
                "relevance": "Numerical agreement does not identify which model or stage generated the H5 value.",
            })
    pd.DataFrame(source_rows).to_csv(output / "source_processing_evidence.csv", index=False)

    range_rows = []
    for granule in GRANULES:
        for model in ("A_no_atmospheric_correction", "B_atmospheric_correction"):
            range_rows.append({
                "granule_id": granule, "model": model,
                "status": "UNAVAILABLE", "independent_raw_range_reconstruction": False,
                "stored_refh_available": True, "stored_delay_available": True,
                "missing_indispensable_inputs": "matched L1A/raw TX-RX samples; original time zero and clock calibration; independent GPS/INS/geolocation trajectory; processor range equation and calibration record",
                "reason": "Processed Refh and the stored delay are shared L1B descendants; no independent raw range inputs exist locally, so their agreement cannot identify application state.",
            })
    pd.DataFrame(range_rows).to_csv(output / "range_reconstruction_comparison.csv", index=False)

    # Bias budget does not force component estimates to add to the observed residual.
    bias_rows = []
    for granule in GRANULES:
        g = residual_df[residual_df.granule_id.eq(granule)]
        c7m = residual_metrics(g.C7_residual_m)
        h1m = residual_metrics(g.H1_stored_as_m_residual_m)
        geo = geoid_by_granule[granule]
        geoid_med = next((row.get("median_difference_m") for row in geo
                          if row.get("comparison") == "H5 minus GEOID18"), None)
        bias_rows.extend([
            {"granule_id": granule, "component": "Observed C7 versus local 3DEP plane", "estimated_effect_m": c7m["median_bias_m"],
             "effect_variation_m": c7m["NMAD_m"], "evidence_source": "frozen independent-test common-support SNR>=5 pulses",
             "applied_status": "C7 operation provisional; source frame unresolved", "verification_status": "OBSERVED_SAMPLE_RESIDUAL",
             "uncertainty": "3DEP/local plane, pulse correlation, return-ground match and frame components not fully represented by pulse NMAD",
             "identifiable": "sample residual only; not a unique physical component"},
            {"granule_id": granule, "component": "Atmospheric field as meters (conditional sensitivity)", "estimated_effect_m": float(g.height_effect_if_m.median()),
             "effect_variation_m": float(1.4826 * np.median(np.abs(g.height_effect_if_m - g.height_effect_if_m.median()))),
             "evidence_source": "H5 numeric delay plus real H5 beam geometry", "applied_status": "UNKNOWN",
             "verification_status": "PHYSICAL_SENSITIVITY_ONLY; ERA5 magnitude comparison does not establish units or application",
             "uncertainty": "units/sign/processor state absent; double-correction risk remains", "identifiable": "candidate effect only"},
            {"granule_id": granule, "component": "Geoid/reference surface", "estimated_effect_m": None,
             "effect_variation_m": None, "evidence_source": f"existing geoid model comparison; H5 minus GEOID18 median={geoid_med}",
             "applied_status": "UNKNOWN", "verification_status": "NUMERIC_MODEL_COMPARISON_ONLY",
             "uncertainty": "model identity and geoid application stage not declared in H5", "identifiable": "not as a residual contribution"},
            {"granule_id": granule, "component": "Coordinate frame", "estimated_effect_m": None,
             "effect_variation_m": None, "evidence_source": "existing reference_frame_operation_audit.csv; selected conversion numerical no-op",
             "applied_status": "UNRESOLVED_SOURCE_FRAME", "verification_status": "OPERATION_NUMERICALLY_REPRODUCED_ONLY",
             "uncertainty": "trajectory realization/epoch absent; operation metadata accuracy 2.015 m is not a correction estimate",
             "identifiable": "no"},
            {"granule_id": granule, "component": "Instrument range calibration", "estimated_effect_m": None,
             "effect_variation_m": None, "evidence_source": "H5 range_bias_correction field is zero; no independent calibration record",
             "applied_status": "UNKNOWN", "verification_status": "NOT_IDENTIFIABLE",
             "uncertainty": "zero stored field does not prove zero instrument calibration bias", "identifiable": "no"},
            {"granule_id": granule, "component": "Joint unidentified remainder", "estimated_effect_m": None,
             "effect_variation_m": None, "evidence_source": "processing, frame, calibration and new-date validation evidence gaps",
             "applied_status": "NOT_SEPARABLE", "verification_status": "UNIDENTIFIED",
             "uncertainty": "cannot allocate the observed residual among atmosphere, frame, calibration, reference and surface terms",
             "identifiable": "joint component only; numeric allocation prohibited"},
            {"granule_id": granule, "component": "H1 field-as-m residual (conditional)", "estimated_effect_m": h1m["median_bias_m"],
             "effect_variation_m": h1m["NMAD_m"], "evidence_source": "same frozen pulses as C7; hypothetical atmospheric field correction",
             "applied_status": "UNKNOWN", "verification_status": "NOT_A_VERIFIED_CORRECTED_HEIGHT",
             "uncertainty": "sensitivity only; no independent application evidence", "identifiable": "conditional residual only"},
        ])
    pd.DataFrame(bias_rows).to_csv(output / "bias_component_budget.csv", index=False)

    delay_medians = {g: stat(g, "refh_neutat_delay_total") for g in GRANULES}
    c7_metrics_12 = residual_metrics(residual_df[residual_df.granule_id.eq(GRANULES[0])].C7_residual_m)
    c7_metrics_18 = residual_metrics(residual_df[residual_df.granule_id.eq(GRANULES[1])].C7_residual_m)
    c7_mean_delta = c7_metrics_18["mean_bias_m"] - c7_metrics_12["mean_bias_m"]
    c7_median_delta = c7_metrics_18["median_bias_m"] - c7_metrics_12["median_bias_m"]
    delay_delta = delay_medians[GRANULES[1]] - delay_medians[GRANULES[0]]
    decision = {
        "decision": "NO — not identifiable from available evidence",
        "classification": "A3_PROVEN_NON_IDENTIFIABILITY_FROM_AVAILABLE_INPUTS",
        "core_question": "Was refh_neutat_delay_total already applied when CASALS generated Refh?",
        "application_status": "APPLICATION_STATE_NOT_IDENTIFIABLE_FROM_AVAILABLE_INPUTS",
        "verified_cause_for_approximately_minus_1_6m_bias": False,
        "verified_numeric_components": [
            "C7 selected frame operation is a numerical no-op with reported operation accuracy metadata 2.015 m; the true source frame remains unknown.",
            "H5 geoid values numerically agree with EGM2008 on the sampled coordinates; that does not prove product lineage or applied stage.",
            "ERA5/Ciddor profile integration provides an independent atmospheric magnitude and derivative-scale comparison; it does not prove H5 units or Refh application.",
        ],
        "historical_delay_difference_m_if_field_is_metres": delay_delta,
        "historical_C7_mean_bias_difference_Nov18_minus_Nov12_m": c7_mean_delta,
        "historical_C7_median_bias_difference_Nov18_minus_Nov12_m": c7_median_delta,
        "cross_granule_interpretation": "Under a meter interpretation, delay changes by about -0.41 m while the mean C7 road bias changes by about +0.25 m. The directions differ, but site/date and flight conditions are confounded, so this is not a causal test.",
        "absolute_NAVD88_accuracy_claim_supported": False,
        "independent_new_date_validation": "UNAVAILABLE; 273 public L1B H5 objects occur only on 2024-11-12 and 2024-11-18; no new-date scene/site was available to qualify.",
        "remaining_bias": "Observed C7 mean biases are about -1.82 m and -1.57 m (medians -1.84 m and -1.55 m) in the frozen SNR>=5 common samples. H1 sensitivities are conditional and cannot be called corrected Refh.",
        "required_CASALS_team_inputs": [
            "Matched original L1A/raw TX and RX waveform or ranging samples for representative pulse IDs",
            "Pulse launch time, receive sample time zero, sample interval, clock/timing calibration, bounce-time and Refh-bin semantics",
            "L1B processor repository/version/commit, configuration, exact range equation and correction-stage log for these files",
            "Neutral-delay source, physical quantity, units, sign, one-way/two-way convention, and whether it is applied to Refh",
            "Independent GPS/INS trajectory solution, source realization, coordinate epoch and vertical reference",
            "Independent range-calibration measurements and calibration constants",
            "Additional acquisition dates with reviewed stable ground scenes and reference-data metadata",
        ],
        "limitations": [
            "ERA5 is hourly and 0.25 degree; surface-to-aircraft integration uses a near-nadir plane-parallel mapping and nominal 1040 nm.",
            "ERA5 geopotential height and H5/3DEP height references are not proven to share a vertical datum; this affects path height by a small but unresolved amount.",
            "Only one historical road site per acquisition date is represented in the frozen SNR>=5 common sample.",
            "No pulses are statistically independent just because many pulse records exist; track/site/date are the relevant grouping levels.",
            "A lower RMSE under a candidate sensitivity is not evidence of processor behavior or cause.",
        ],
    }
    _save_json(output / "final_attribution_decision.json", decision)

    # Provenance links remote immutable object generations/hashes to the small derived profiles.
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": _sha256(Path(__file__)),
        "frozen_inputs": {
            str((AUDIT_DIR / name).relative_to(ROOT)): _sha256(AUDIT_DIR / name)
            for name in ("correction_statistics.csv", "correction_application_matrix.csv",
                         "reference_frame_operation_audit.csv", "geoid_model_comparison.csv")
        },
        "selected_historical_population": str((GROUND_DIR / "ground_target_test_population.parquet").relative_to(ROOT)),
        "selected_pulses_by_granule": selected.groupby("granule_id").size().to_dict(),
        "casals_s3_inventory": {"bucket": "casals-data", "prefix": "lidar/", "metadata_only": True,
                                "h5_objects": len(inventory), "inventory_date_utc": inventory_date},
        "era5_arco_sources": era5_sources,
        "profile_method": "Ciddor group refractivity integrated in geopotential height; surface pressure from ERA5; bilinear horizontal interpolation; nearest UTC hour; 1/sin(elevation) slant mapping",
        "no_local_dependency_changes": True,
    }
    _save_json(output / "run_provenance.json", provenance)

    _make_figures(output, profiles, independent_rows, residual_df, decision)

    report = _render_report(decision, source_rows, unit_rows, population_rows, bias_rows,
                            range_rows, len(inventory), era5_results, metric_rows)
    (output / "final_attribution_report.md").write_text(report, encoding="utf-8")
    _write_artifact_hash_manifest(output)
    return {"output": str(output), "decision": decision["decision"],
            "inventory_h5_count": len(inventory),
            "selected_pulses": selected.groupby("granule_id").size().to_dict(),
            "atmospheric_profiles": era5_results}


def _render_report(decision: dict[str, Any], source_rows: list[dict[str, Any]],
                   unit_rows: list[dict[str, Any]], population_rows: list[dict[str, Any]],
                   bias_rows: list[dict[str, Any]], range_rows: list[dict[str, Any]],
                   inventory_count: int, era5_results: dict[str, dict[str, Any]],
                   metric_rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Final CASALS L1B Vertical Bias Attribution",
        "",
        f"**{decision['decision']}**",
        "",
        "## Finding",
        "",
        "The approximately -1.6 m C7-to-3DEP road residual is not physically identified by available evidence. The strongest valid conclusion is A3: the processed L1B files lack the independent ranging and processor provenance needed to determine whether neutral delay was applied when Refh was generated. A numerically improved residual under an atmospheric sensitivity does not replace that evidence.",
        "",
        "## Processing evidence",
        "",
        "| Correction | Applied status | Units | Independent evidence | Conclusion |",
        "|---|---|---|---|---|",
        "| Neutral atmosphere | Unknown | Undeclared in H5 | ERA5/Ciddor supports a physical scale only; no processor source/log | Double-correction risk remains |",
        "| Instrument range bias | Unknown | Undeclared; stored field is zero | No independent calibration record | Zero field does not prove zero bias |",
        "| Geoid | Unknown application stage | H5 units/lineage not declared | H5 values numerically resemble EGM2008; GEOID18 comparison is recorded | Numerical model agreement only |",
        "| Source frame | Unresolved | Not applicable | C7 selected WGS84-ensemble to NAD83(2011) operation is a no-op; source realization/epoch absent | Strict frame conversion not verified |",
        "| Other geophysical corrections | Unknown | Undeclared | H5 fields and existing audit only | Application cannot be reconstructed |",
        "",
        "The H5 root references L1A, ARD, geolocation, GPS and TDMS files, but the matched L1A/ARD/geolocation/GPS products and exact referenced TDMS files are absent locally. The public source search found the NASA design paper and a public L1B waveform tutorial, but no CASALS L1B processor code, configuration, geolocation ATBD or processing log that documents the neutral-delay calculation or application. The NASA paper's 1040 nm is used only as a nominal wavelength.",
        "",
        "Physical-model references: [NASA CASALS design paper](https://ntrs.nasa.gov/citations/20240001267), [public CASALS L1B waveform tutorial](https://book.cryointhecloud.com/l1b-waveforms-tutorial/), [official ERA5 pressure-level dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=access), [ARCO-ERA5 data documentation](https://github.com/google-research/arco-era5/blob/main/README.md), [Ciddor 1996](https://opg.optica.org/ao/abstract.cfm?URI=ao-35-9-1566), and [Mendes et al. 2004 optical group-delay model](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2004GL020308).",
        "",
        "## Independent atmospheric comparison",
        "",
        "ERA5 pressure-level temperature, specific humidity, geopotential and surface pressure were bilinearly sampled at each H5 median instrument location and nearest UTC hour. Ciddor group refractivity was integrated from the historical local 3DEP plane to H5 instrument altitude. The resulting path is a one-way optical group excess path; for reciprocal two-way ranging its numerical value equals the one-way geometric-range effect after multiplying round-trip time by c/2. Beam slant mapping uses 1/sin(elevation), valid here as a near-nadir approximation.",
        "",
        "| Granule | ERA5 slant excess path (m) | H5 field median | H5 derivative median | Status |",
        "|---|---:|---:|---:|---|",
    ]
    audit = pd.read_csv(AUDIT_DIR / "correction_statistics.csv")
    def val(g: str, field: str) -> float:
        row = audit[(audit.granule_id == g) & (audit.field == field) & (audit.scope == "full_granule")]
        return float(row.iloc[0]["median"])
    for granule, result in era5_results.items():
        lines.append(f"| {granule} | {result['slant_one_way_group_excess_path_m']:.3f} | {val(granule, 'refh_neutat_delay_total'):.3f} | {val(granule, 'refh_neutat_delay_derivative'):.6f} | Magnitude comparison only; H5 units/application unknown |")
    lines.extend([
        "",
        "The stored derivative values near 0.000278 numerically match the ground-level group-refractivity scale under an m/m interpretation. That supports a plausible physical scale; it does not prove the derivative definition or total-field unit. The meter candidate is much closer to the independent profile scale than either nanosecond conversion, but model resolution, vertical-frame mismatch, source wavelength detail and unobserved atmospheric variability prevent this from resolving the H5 unit contract.",
        "",
        "## Conditional historical validation",
        "",
        "The public S3 metadata inventory contains **" + str(inventory_count) + " L1B H5 objects**, all on 2024-11-12 or 2024-11-18. No object content was downloaded for this inventory. Thus no unseen acquisition date is available for prospective ground validation. The two existing sites are retained as historical checks only.",
        "",
        "| Date | Site | Independent sites | Track blocks | Valid pulses | H5 delay median (site sample / full granule; units unknown) | ERA5 path (m) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in population_rows:
        lines.append(f"| {row['acquisition_date']} | {row['site_id']} | 1 | {row['distinct_track_blocks']} | {row['eligible_common_snr_ge_5_pulses']} | {row['median_stored_delay_numeric']:.3f} / {row['full_granule_delay_median_numeric']:.3f} | {row['ERA5_slant_group_excess_path_m']:.3f} |")
    lines.extend([
        "",
        "| Date | Site | Hypothesis | Mean bias (m) | Median bias (m) | RMSE (m) | NMAD (m) | Evidence status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ])
    shown_hypotheses = {"C7_baseline", "H1_stored_delay_as_m", "H1_ERA5_profile_sensitivity"}
    for row in metric_rows:
        if row["group_type"] == "site" and row["hypothesis"] in shown_hypotheses:
            date = "2024-11-12" if "20241112" in row["granule_id"] else "2024-11-18"
            label = {"C7_baseline": "C7 provisional baseline",
                     "H1_stored_delay_as_m": "H1: H5 field treated as meters",
                     "H1_ERA5_profile_sensitivity": "ERA5 profile sensitivity"}[row["hypothesis"]]
            lines.append(f"| {date} | {row['site_id']} | {label} | {row['mean_bias_m']:.3f} | {row['median_bias_m']:.3f} | {row['RMSE_m']:.3f} | {row['NMAD_m']:.3f} | {row['evidence_status']} |")
    lines.extend([
        "",
        "C7 and each candidate sensitivity were evaluated on the same frozen independent-test pulse IDs, filtered to common C0-C7 support and SNR >= 5. The field-as-meter and ERA5 residuals are sensitivities, not corrected Refh. Pulse spread is descriptive; track, site and acquisition date are the relevant independent grouping levels.",
        "",
        "## Physical hypotheses and cross-date result",
        "",
        "The full-granule H5 delay medians differ by about **" + f"{decision['historical_delay_difference_m_if_field_is_metres']:.3f} m" + "** under an unverified meter interpretation. The C7 mean road bias changes by **" + f"{decision['historical_C7_mean_bias_difference_Nov18_minus_Nov12_m']:.3f} m" + "** (the median changes by **" + f"{decision['historical_C7_median_bias_difference_Nov18_minus_Nov12_m']:.3f} m" + "**). These are different sites, land areas, flights and surface contexts; two units cannot identify the atmospheric prediction or a constant calibration/frame term. No cross-site causal regression is warranted.",
        "",
        "H1's prior reduction in RMSE is consistent with a plausible atmospheric scale but is not proof the correction was missing. The actual processing state could be applied, unapplied, partly applied, or based on a different correction definition. Applying the field again without a processor record could double-correct Refh.",
        "",
        "## Bias budget",
        "",
        "| Component | Contribution | Verified? | Uncertainty / remaining issue |",
        "|---|---:|---|---|",
    ])
    for component in ("Observed C7 versus local 3DEP plane", "Atmospheric field as meters (conditional sensitivity)",
                      "Geoid/reference surface", "Coordinate frame", "Instrument range calibration", "Joint unidentified remainder"):
        rows = [r for r in bias_rows if r["component"] == component]
        c = rows[0]
        value = "not separable" if c["estimated_effect_m"] is None else f"{float(c['estimated_effect_m']):.3f} m (Nov 12); {float(rows[1]['estimated_effect_m']):.3f} m (Nov 18)"
        verified = "No" if "OBSERVED" not in c["verification_status"] else "Observed sample residual only"
        lines.append(f"| {component} | {value} | {verified} | {c['uncertainty']} |")
    lines.extend([
        "",
        "No component values are forced to sum to the observed residual. The unexplained amount is joint and unallocated because atmosphere, frame, calibration, reference datum and return-to-ground effects are not separately identified.",
        "",
        "## Direct answers",
        "",
        "1. **Was H1 a genuinely missing atmospheric correction?** Unknown. Its improved fit is a numerical match, not application evidence.",
        "2. **Why does the delay differ by about 0.41 m while C7 road bias differs less?** Different dates/sites and flight/surface conditions are confounded; only two historical units exist, and C7 bias change does not track the stored delay change in a way that identifies cause.",
        "3. **Are neutral-delay units and applied status known?** No. The meter interpretation has the strongest profile-scale support; units and status remain undeclared, with double-correction risk.",
        "4. **Is the source-frame transformation established?** No. The selected numeric operation is a no-op; GNSS/INS realization and coordinate epoch are missing.",
        "5. **Is remaining bias stable on independent validation?** There is no new-date independent validation. The two frozen historical road sites show C7 mean biases near -1.82 m and -1.57 m (medians -1.84 m and -1.55 m); this is not a stability claim.",
        "6. **Can a corrected Refh be used for absolute NAVD88/3DEP comparison?** No. No correction is verified as applied or unapplied, and frame lineage remains unresolved.",
        "7. **Are CASALS team inputs still needed?** Yes. Required items are listed in `final_attribution_decision.json`, including matched raw L1A/TX/RX/timing, processor version/config/log, range and calibration definitions, trajectory realization/epoch, and further acquisition dates.",
        "",
        "## Artifacts and reproducibility",
        "",
        "See `source_processing_evidence.csv`, `source_frame_evidence.json`, `range_reconstruction_comparison.csv`, `atmospheric_unit_validation.csv`, `independent_atmospheric_comparison.csv`, `validation_granule_manifest.csv`, `validation_site_manifest.geojson`, `validation_population_summary.csv`, `physical_hypothesis_metrics.csv`, `physical_hypothesis_residuals.parquet`, `residual_vs_delay_analysis.csv`, `bias_component_budget.csv`, `final_attribution_decision.json`, `run_provenance.json` and `artifact_sha256_manifest.csv`.",
        "",
        "Figures: [ERA5 profiles](figures/atmospheric_physics.png), [raw Refh reconstruction evidence](figures/refh_range_reconstruction.png), [frozen cross-granule residuals](figures/cross_granule_bias.png), [conditional model comparison](figures/model_discrimination.png), and [unallocated bias budget](figures/residual_error_budget.png).",
        "",
        "Rebuild from the repository root with `python -m research.reference.final_bias_validation --output outputs/notebooks/10_3dep_elevation_validation/final_bias_attribution`. This reads public S3 object metadata and ERA5 profile chunks; it does not download L1B object contents.",
        "",
        "Inputs in production, historical `datum_comparison/`, `ground_aware/`, `correction_audit/`, L1B H5 and LAZ folders were read only.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-era5", action="store_true", help="build evidence package without remote ERA5 profile data")
    args = parser.parse_args()
    print(json.dumps(build_final_attribution(args.output, fetch_atmosphere=not args.skip_era5), indent=2))


if __name__ == "__main__":
    main()

