"""Independent numerical checks of the audited CASALS GEOID18 operation.

This module compares PROJ/C7, an independent rasterio interpolation, and the
NOAA NGS/VDatum services on fixed CASALS H5 pulses. The pulse sample uses only
H5 coordinates, Refh finiteness, and Refh SNR; no 3DEP value or residual enters
sample selection or any conversion parameter.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import pyproj
from pyproj import datadir, network
import requests

from research.reference import vertical_alignment as physical
from research.reference import vertical_datum_comparison as datum


NGS_GEOID_URL = "https://geodesy.noaa.gov/api/geoid/ght"
VDATUM_URL = "https://vdatum.noaa.gov/vdatumweb/api/convert"
NGS_DOC_URL = "https://geodesy.noaa.gov/web_services/geoid.shtml"
VDATUM_DOC_URL = "https://vdatum.noaa.gov/docs/services.html"
COORDINATE_ASSUMPTION = "Assumed NAD83(2011) coordinates for numerical verification only"
MODEL_TOLERANCE_M = 0.01
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_SECONDS = 15
REQUEST_PAUSE_SECONDS = 0.15

# Frozen from deterministic farthest-point sampling over each complete H5
# geolocation stream, restricted only by finite Refh/coordinates and SNR >= 5.
# The exact IDs are asserted again at runtime so source changes cannot silently
# move this independent software-verification sample.
FIXED_PULSE_IDS = {
    "casals_l1b_20241112T165718_001_02": (
        15805, 370402, 473119, 776982, 812128, 1206973, 1241411, 1787391,
        1872322, 2245951, 2338046, 2378154, 2420003, 2543554, 2767167,
        2991327, 3281448, 3378144, 3504073, 3592799,
    ),
    "casals_l1b_20241118T171757_001_02": (
        0, 404702, 728063, 1149695, 1284801, 1713151, 2099156, 2392064,
        2617600, 2718640, 2802144, 2841215, 2905343, 3033120, 3107327,
        3211520, 3239327, 3313599, 3455999, 3604224,
    ),
}
REGION_BY_GRANULE = {
    "casals_l1b_20241112T165718_001_02": "chesapeak_delaware",
    "casals_l1b_20241118T171757_001_02": "contiguous",
}


class CrosscheckResponseError(ValueError):
    """A service response is not a valid result for the requested conversion."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CrosscheckResponseError(f"{field} is not numeric") from exc
    if not math.isfinite(number):
        raise CrosscheckResponseError(f"{field} is not finite")
    return number


def parse_ngs_geoid_response(payload: Mapping[str, Any]) -> float:
    """Return NGS GEOID18 undulation in metres after identity/unit checks."""
    if not isinstance(payload, Mapping):
        raise CrosscheckResponseError("NGS response is not a JSON object")
    if payload.get("errorCode") is not None:
        raise CrosscheckResponseError(f"NGS errorCode={payload.get('errorCode')}")
    if str(payload.get("geoidModel", "")).strip().upper() != "GEOID18":
        raise CrosscheckResponseError(
            f"NGS returned model {payload.get('geoidModel')!r}, expected GEOID18"
        )
    if str(payload.get("units", "")).strip().lower() not in {"m", "meter", "meters"}:
        raise CrosscheckResponseError(
            f"NGS returned units {payload.get('units')!r}, expected metres"
        )
    return _number(payload.get("geoidHeight"), "geoidHeight")


def build_ngs_params(longitude_deg: float, latitude_deg: float) -> dict[str, str]:
    """Build the documented single-point GEOID18 request, with west longitude."""
    if longitude_deg >= 0:
        raise ValueError("CASALS study points must use negative west longitude")
    return {
        "lat": format(float(latitude_deg), ".17g"),
        "lon": format(float(longitude_deg), ".17g"),
        "model": "14",
    }


def build_vdatum_params(
    longitude_deg: float,
    latitude_deg: float,
    ellipsoidal_height_m: float,
    region: str,
) -> dict[str, str]:
    """Documented same-frame NAD83(2011) ellipsoid-to-NAVD88 request."""
    if longitude_deg >= 0:
        raise ValueError("CASALS study points must use negative west longitude")
    if region not in {"contiguous", "chesapeak_delaware"}:
        raise ValueError(f"Unsupported documented VDatum region: {region}")
    return {
        "s_x": format(float(longitude_deg), ".17g"),
        "s_y": format(float(latitude_deg), ".17g"),
        "s_z": format(float(ellipsoidal_height_m), ".17g"),
        "region": region,
        "s_h_frame": "NAD83_2011",
        "s_coor": "geo",
        "s_v_frame": "NAD83_2011",
        "s_v_unit": "m",
        "s_v_elevation": "height",
        "s_v_geoid": "geoid18",
        "t_h_frame": "NAD83_2011",
        "t_coor": "geo",
        "t_v_frame": "NAVD88",
        "t_v_unit": "m",
        "t_v_elevation": "height",
        "t_v_geoid": "geoid18",
        "epoch_in": "2010.0",
        "epoch_out": "2010.0",
    }


def parse_vdatum_response(
    payload: Mapping[str, Any], expected_params: Mapping[str, str]
) -> dict[str, float | str | None]:
    """Validate the echoed VDatum route and return target NAVD88 height."""
    if not isinstance(payload, Mapping):
        raise CrosscheckResponseError("VDatum response is not a JSON object")
    if payload.get("errorCode") is not None:
        raise CrosscheckResponseError(
            f"VDatum errorCode={payload.get('errorCode')}: {payload.get('message', '')}"
        )
    text_fields = (
        "s_h_frame", "s_coor", "s_v_frame", "s_v_unit", "s_v_elevation",
        "s_v_geoid", "t_h_frame", "t_coor", "t_v_frame", "t_v_unit",
        "t_v_elevation", "t_v_geoid",
    )
    for field in text_fields:
        if str(payload.get(field, "")).strip().lower() != str(expected_params[field]).lower():
            raise CrosscheckResponseError(
                f"VDatum {field}={payload.get(field)!r}, expected {expected_params[field]!r}"
            )
    if str(payload.get("region", "")).strip().lower() != expected_params["region"].lower():
        raise CrosscheckResponseError(
            f"VDatum region={payload.get('region')!r}, expected {expected_params['region']!r}"
        )
    for field in ("epoch_in", "epoch_out"):
        if abs(_number(payload.get(field), field) - float(expected_params[field])) > 1e-9:
            raise CrosscheckResponseError(
                f"VDatum {field}={payload.get(field)!r}, expected {expected_params[field]!r}"
            )
    for field, request_field in (("s_x", "s_x"), ("s_y", "s_y"), ("s_z", "s_z")):
        actual = _number(payload.get(field), field)
        expected = float(expected_params[request_field])
        if abs(actual - expected) > max(1e-9, abs(expected) * 1e-12):
            raise CrosscheckResponseError(f"VDatum echoed {field} differs from the request")
    return {
        "target_height_m": _number(payload.get("t_z"), "t_z"),
        "target_longitude_deg": _number(payload.get("t_x"), "t_x"),
        "target_latitude_deg": _number(payload.get("t_y"), "t_y"),
        "uncertainty_m": _number(payload["uncertainty"], "uncertainty")
        if payload.get("uncertainty") not in (None, "") else None,
        "target_vertical_frame": str(payload["t_v_frame"]),
        "target_vertical_unit": str(payload["t_v_unit"]),
        "epoch_in": _number(payload["epoch_in"], "epoch_in"),
        "epoch_out": _number(payload["epoch_out"], "epoch_out"),
    }


def _farthest_spatial_positions(longitude: np.ndarray, latitude: np.ndarray, count: int) -> np.ndarray:
    """Deterministic max-min spread in a local, cosine-corrected lon/lat plane."""
    lon = np.asarray(longitude, dtype=np.float64).reshape(-1)
    lat = np.asarray(latitude, dtype=np.float64).reshape(-1)
    if lon.shape != lat.shape or len(lon) < count or not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError("Spatial sample inputs must be equal-sized, finite arrays with enough points")
    x = lon * np.cos(np.deg2rad(float(np.median(lat))))
    xy = np.column_stack((x, lat))
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = hi - lo
    span[span == 0.0] = 1.0
    xy = (xy - lo) / span
    center = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    first = int(np.argmin(np.sum((xy - center) ** 2, axis=1)))
    chosen = [first]
    nearest = np.sum((xy - xy[first]) ** 2, axis=1)
    while len(chosen) < count:
        candidate = int(np.argmax(nearest))
        chosen.append(candidate)
        distance = np.sum((xy - xy[candidate]) ** 2, axis=1)
        nearest = np.minimum(nearest, distance)
    return np.asarray(chosen, dtype=np.int64)


def _fixed_sample_points() -> pd.DataFrame:
    """Recompute and verify the frozen 20-per-granule H5-only sample."""
    rows: list[dict[str, Any]] = []
    for spec in datum.GRANULES:
        if spec.granule_id not in FIXED_PULSE_IDS:
            raise ValueError(f"No frozen crosscheck IDs for {spec.granule_id}")
        with h5py.File(spec.h5_path, "r") as h5:
            lon = np.asarray(h5["refh_longitude"], dtype=np.float64)
            lat = np.asarray(h5["refh_latitude"], dtype=np.float64)
            height = np.asarray(h5["refh"], dtype=np.float64)
            snr = np.asarray(h5["refh_snr"], dtype=np.float64)
        valid = (np.isfinite(lon) & np.isfinite(lat) & np.isfinite(height) & np.isfinite(snr)
                 & (snr >= 5.0) & (lon >= -130.0) & (lon <= -60.0)
                 & (lat >= 24.0) & (lat <= 58.0))
        eligible_ids = np.flatnonzero(valid)
        selected_positions = _farthest_spatial_positions(lon[eligible_ids], lat[eligible_ids], 20)
        selection_ids = eligible_ids[selected_positions]
        if tuple(sorted(map(int, selection_ids))) != FIXED_PULSE_IDS[spec.granule_id]:
            raise ValueError(
                f"Frozen sample pulse IDs changed for {spec.granule_id}; source H5 or selection inputs differ"
            )
        ranks = {int(pulse_id): rank for rank, pulse_id in enumerate(selection_ids, start=1)}
        for pulse_id in sorted(selection_ids):
            i = int(pulse_id)
            rows.append({
                "granule_id": spec.granule_id,
                "pulse_id": i,
                "pulse_index": i,
                "selection_rank": ranks[i],
                "longitude_deg": float(lon[i]),
                "latitude_deg": float(lat[i]),
                "refh_ellipsoidal_m": float(height[i]),
                "refh_snr": float(snr[i]),
                "sample_selection": "farthest spatial sample over complete H5; finite values; Refh SNR >= 5",
                "coordinate_assumption": COORDINATE_ASSUMPTION,
                "vdatum_region": REGION_BY_GRANULE[spec.granule_id],
            })
        del lon, lat, height, snr, valid, eligible_ids
    return pd.DataFrame(rows).sort_values(["granule_id", "pulse_index"]).reset_index(drop=True)


def _cache_key(provider: str, url: str, params: Mapping[str, str]) -> str:
    text = json.dumps({"provider": provider, "url": url, "params": dict(sorted(params.items()))},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def _load_response_cache(runs_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    cache: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for path in sorted(runs_root.glob("*/geodetic_crosscheck/service_requests_responses.json")):
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for record in content.get("calls", []):
            if record.get("status") != "success" or not record.get("valid"):
                continue
            provider = record.get("provider")
            try:
                if provider == "NGS Geoid Height Service":
                    parse_ngs_geoid_response(record.get("response_json"))
                elif provider == "NOAA VDatum Full API":
                    parse_vdatum_response(record.get("response_json"), record["request_params"])
                else:
                    continue
            except (CrosscheckResponseError, KeyError, TypeError):
                continue
            key = record.get("cache_key")
            if key and key not in cache:
                cache[key] = record
                sources[key] = str(path)
    return cache, sources


def _query_service(
    *,
    provider: str,
    url: str,
    params: dict[str, str],
    parser: Any,
    session: requests.Session,
    cache: Mapping[str, dict[str, Any]],
    cache_sources: Mapping[str, str],
) -> dict[str, Any]:
    key = _cache_key(provider, url, params)
    if key in cache:
        record = dict(cache[key])
        record["cache_reused"] = True
        record["cache_source_file"] = cache_sources.get(key)
        record["reused_at_utc"] = _utc_now()
        return record

    attempts: list[dict[str, Any]] = []
    parsed: Any = None
    error_text: str | None = None
    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        requested_at = _utc_now()
        attempt: dict[str, Any] = {
            "attempt": attempt_number,
            "requested_at_utc": requested_at,
            "service_url": url,
            "request_params": params,
        }
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            attempt["request_url"] = response.url
            attempt["http_status"] = int(response.status_code)
            attempt["response_text"] = response.text
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError):
                payload = None
            attempt["response_json"] = payload
            if response.status_code != 200:
                error_text = f"HTTP {response.status_code}"
                attempt["error"] = error_text
                attempts.append(attempt)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt_number < MAX_ATTEMPTS:
                        time.sleep(0.4 * attempt_number)
                        continue
                break
            try:
                parsed = parser(payload)
            except CrosscheckResponseError as exc:
                error_text = str(exc)
                attempt["error"] = error_text
                attempts.append(attempt)
                # An API-level/semantic error is point-specific; don't repeat it.
                break
            attempts.append(attempt)
            return {
                "provider": provider,
                "service_url": url,
                "request_params": params,
                "request_url": response.url,
                "requested_at_utc": requested_at,
                "http_status": int(response.status_code),
                "response_text": response.text,
                "response_json": payload,
                "parsed": parsed,
                "attempts": attempts,
                "attempt_count": attempt_number,
                "cache_key": key,
                "cache_reused": False,
                "status": "success",
                "valid": True,
            }
        except requests.RequestException as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            attempt["error"] = error_text
            attempts.append(attempt)
            if attempt_number < MAX_ATTEMPTS:
                time.sleep(0.4 * attempt_number)
                continue
            break

    return {
        "provider": provider,
        "service_url": url,
        "request_params": params,
        "requested_at_utc": attempts[-1]["requested_at_utc"] if attempts else _utc_now(),
        "request_url": attempts[-1].get("request_url") if attempts else None,
        "http_status": attempts[-1].get("http_status") if attempts else None,
        "response_text": attempts[-1].get("response_text") if attempts else None,
        "response_json": attempts[-1].get("response_json") if attempts else None,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "cache_key": key,
        "cache_reused": False,
        "status": "failed",
        "valid": False,
        "error": error_text or "No valid service response",
    }


def _difference_stats(values: Any) -> dict[str, Any]:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = raw[np.isfinite(raw)]
    if not len(valid):
        return {"n": 0, "mean_m": None, "median_m": None, "rmse_m": None, "max_abs_m": None}
    return {
        "n": int(len(valid)),
        "mean_m": float(np.mean(valid)),
        "median_m": float(np.median(valid)),
        "rmse_m": float(np.sqrt(np.mean(valid**2))),
        "max_abs_m": float(np.max(np.abs(valid))),
    }


def _frame_status_from_operation(audit: Mapping[str, Any]) -> dict[str, Any]:
    frame = audit["frame"]
    return {
        "source_frame": frame["source_frame"],
        "source_realization": frame["source_realization"],
        "source_coordinate_epoch": frame["source_coordinate_epoch"],
        "target_frame": frame["target_frame"],
        "required_target_coordinate_epoch": frame["required_target_coordinate_epoch"],
        "target_coordinate_epoch_achieved": frame["target_coordinate_epoch_achieved"],
        "actual_frame_operation_is_noop": frame["is_noop"],
        "actual_frame_operation_is_ballpark": frame["is_ballpark"],
        "frame_delta_longitude_deg_max_abs": frame["delta_longitude_deg_max_abs"],
        "frame_delta_latitude_deg_max_abs": frame["delta_latitude_deg_max_abs"],
        "frame_delta_ellipsoidal_height_m_max_abs": frame["delta_ellipsoidal_height_m_max_abs"],
    }


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_json_safe(obj), indent=2, allow_nan=False), encoding="utf-8")


def run_geodetic_crosscheck(output: str | Path, *, project_root: str | Path | None = None) -> dict[str, Any]:
    """Run fixed-sample PROJ/raster/NGS and same-frame PROJ/VDatum checks.

    The function intentionally has no reference-height or residual arguments.
    NOAA requests are serial, with three bounded attempts for transport and
    retryable HTTP errors. Invalid point results are retained as failures.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(project_root).resolve() if project_root is not None else datum.PROJECT_ROOT
    grid = datum.OUTPUT_ROOT / "proj_data" / datum.GEOID18_GRID
    if physical.sha256(grid) != physical.GRID_SHA256:
        raise ValueError("The audited GEOID18 grid SHA256 changed")

    samples = _fixed_sample_points()
    sample_file = output / "sample_points.csv"
    samples.to_csv(sample_file, index=False, float_format="%.15g")
    runs_root = root / "outputs/notebooks/geolocation/vertical_datum_alignment/runs"
    cache, cache_sources = _load_response_cache(runs_root)

    previous_data_dir = datadir.get_data_dir()
    previous_network = network.is_network_enabled()
    operation_audit_path = output.parent / "operation_audit.json"
    prior_operation_audit = (
        json.loads(operation_audit_path.read_text(encoding="utf-8"))
        if operation_audit_path.is_file() else {}
    )
    datadir.set_data_dir(str(grid.parent))
    network.set_network_enabled(False)
    geoid_rows: list[dict[str, Any]] = []
    vdatum_rows: list[dict[str, Any]] = []
    call_records: list[dict[str, Any]] = []
    operation_records: dict[str, Any] = {}
    try:
        with requests.Session() as session:
            for spec in datum.GRANULES:
                group = samples.loc[samples.granule_id.eq(spec.granule_id)].sort_values("pulse_index")
                lon = group.longitude_deg.to_numpy(dtype=np.float64)
                lat = group.latitude_deg.to_numpy(dtype=np.float64)
                height = group.refh_ellipsoidal_m.to_numpy(dtype=np.float64)
                _, _, h_proj, operation = datum.transform_c7(lon, lat, height, return_audit=True)
                h_proj = np.asarray(h_proj, dtype=np.float64)
                n_proj = height - h_proj
                operation_records[spec.granule_id] = {
                    "proj_version": pyproj.__version__,
                    "proj_runtime_version": pyproj.proj_version_str,
                    "grid_name": grid.name,
                    "grid_sha256": physical.GRID_SHA256,
                    "assumed_coordinates": COORDINATE_ASSUMPTION,
                    "operation_definition": operation["selected_operation"]["definition"],
                    "operation_description": operation["selected_operation"]["description"],
                    "operation_accuracy_metadata_m": operation["selected_accuracy_m"],
                    "frame_operation": None,
                    "C7_operation_audit": operation,
                }
                # Preserve the separate source-frame operation already audited by
                # the current C7 run. No extra placeholder or correction is passed.
                existing = prior_operation_audit.get("granules", {}).get(spec.granule_id, {})
                if isinstance(existing.get("frame"), Mapping):
                    operation_records[spec.granule_id]["frame_operation"] = _frame_status_from_operation(
                        {"frame": existing["frame"]}
                    )
                    operation_records[spec.granule_id]["frame_operation_detail"] = existing["frame"]
                # grid_point_check reuses the already audited, independent four-cell
                # raster interpolation and asserts exact sign/height consistency.
                raster_checks = [
                    physical.grid_point_check(
                        grid, float(x), float(y), float(z), float(hp),
                        projection_tolerance_m=None,
                    )
                    for x, y, z, hp in zip(lon, lat, height, h_proj)
                ]
                n_raster = np.asarray([row["N_GEOID18_m"] for row in raster_checks], dtype=np.float64)
                group_rows = group.to_dict("records")
                for i, sample in enumerate(group_rows):
                    common = {
                        "granule_id": spec.granule_id,
                        "pulse_index": int(sample["pulse_index"]),
                        "selection_rank": int(sample["selection_rank"]),
                        "longitude_deg": float(lon[i]),
                        "latitude_deg": float(lat[i]),
                        "refh_ellipsoidal_m": float(height[i]),
                        "coordinate_assumption": COORDINATE_ASSUMPTION,
                    }
                    ngs_params = build_ngs_params(lon[i], lat[i])
                    ngs = _query_service(
                        provider="NGS Geoid Height Service", url=NGS_GEOID_URL,
                        params=ngs_params, parser=parse_ngs_geoid_response,
                        session=session, cache=cache, cache_sources=cache_sources,
                    )
                    ngs["granule_id"] = spec.granule_id
                    ngs["pulse_index"] = int(sample["pulse_index"])
                    call_records.append(ngs)
                    n_ngs = float(ngs["parsed"]) if ngs["status"] == "success" else np.nan
                    geoid_rows.append({
                        **common,
                        "N_PROJ_m": float(n_proj[i]),
                        "N_RASTER_m": float(n_raster[i]),
                        "N_NGS_m": n_ngs,
                        "PROJ_minus_NGS_m": float(n_proj[i] - n_ngs) if np.isfinite(n_ngs) else np.nan,
                        "Raster_minus_NGS_m": float(n_raster[i] - n_ngs) if np.isfinite(n_ngs) else np.nan,
                        "PROJ_minus_Raster_m": float(n_proj[i] - n_raster[i]),
                        "ngs_status": ngs["status"],
                        "ngs_error": ngs.get("error"),
                        "ngs_cache_reused": bool(ngs.get("cache_reused", False)),
                    })

                    vdatum_params = build_vdatum_params(
                        lon[i], lat[i], height[i], REGION_BY_GRANULE[spec.granule_id]
                    )
                    vdatum = _query_service(
                        provider="NOAA VDatum Full API", url=VDATUM_URL,
                        params=vdatum_params,
                        parser=lambda payload, expected=vdatum_params: parse_vdatum_response(payload, expected),
                        session=session, cache=cache, cache_sources=cache_sources,
                    )
                    vdatum["granule_id"] = spec.granule_id
                    vdatum["pulse_index"] = int(sample["pulse_index"])
                    call_records.append(vdatum)
                    h_vdatum = (float(vdatum["parsed"]["target_height_m"])
                                if vdatum["status"] == "success" else np.nan)
                    h_ngs = float(height[i] - n_ngs) if np.isfinite(n_ngs) else np.nan
                    vdatum_rows.append({
                        **common,
                        "H_PROJ_m": float(h_proj[i]),
                        "H_NGS_m": h_ngs,
                        "H_VDATUM_m": h_vdatum,
                        "PROJ_minus_NGS_height_m": float(h_proj[i] - h_ngs) if np.isfinite(h_ngs) else np.nan,
                        "VDatum_minus_PROJ_m": float(h_vdatum - h_proj[i]) if np.isfinite(h_vdatum) else np.nan,
                        "PROJ_minus_VDatum_m": float(h_proj[i] - h_vdatum) if np.isfinite(h_vdatum) else np.nan,
                        "VDatum_minus_NGS_height_m": float(h_vdatum - h_ngs)
                        if np.isfinite(h_vdatum) and np.isfinite(h_ngs) else np.nan,
                        "vdatum_status": vdatum["status"],
                        "vdatum_error": vdatum.get("error"),
                        "vdatum_cache_reused": bool(vdatum.get("cache_reused", False)),
                        "model_epoch_assumption": "NAD83_2011 ellipsoid to NAVD88/GEOID18; assumed position epochs 2010.0 in and out",
                    })
                    time.sleep(REQUEST_PAUSE_SECONDS)
    finally:
        datadir.set_data_dir(previous_data_dir)
        network.set_network_enabled(previous_network)

    geoid = pd.DataFrame(geoid_rows)
    vdatum_frame = pd.DataFrame(vdatum_rows)
    geoid.to_csv(output / "geoid18_software_comparison.csv", index=False, float_format="%.12g")
    vdatum_frame.to_csv(output / "vdatum_transform_comparison.csv", index=False, float_format="%.12g")

    geoid_summaries: list[dict[str, Any]] = []
    vdatum_summaries: list[dict[str, Any]] = []
    for spec in datum.GRANULES:
        gm = geoid.loc[geoid.granule_id.eq(spec.granule_id)]
        vm = vdatum_frame.loc[vdatum_frame.granule_id.eq(spec.granule_id)]
        local_pass = all(
            _difference_stats(gm[col])["n"] == 20
            and _difference_stats(gm[col])["max_abs_m"] <= MODEL_TOLERANCE_M
            for col in ("PROJ_minus_NGS_m", "Raster_minus_NGS_m", "PROJ_minus_Raster_m")
        )
        ngs_success = int(gm.ngs_status.eq("success").sum())
        geoid_status = ("PASSED" if ngs_success == 20 and local_pass
                        else "INCOMPLETE" if ngs_success < 20 else "FAILED")
        geoid_summaries.append({
            "granule_id": spec.granule_id,
            "verification_points": int(len(gm)),
            "successful_NGS_calls": ngs_success,
            **{f"PROJ_vs_NGS_{key}": value for key, value in
               _difference_stats(gm.PROJ_minus_NGS_m).items() if key != "n"},
            **{f"Raster_vs_NGS_{key}": value for key, value in
               _difference_stats(gm.Raster_minus_NGS_m).items() if key != "n"},
            **{f"PROJ_vs_Raster_{key}": value for key, value in
               _difference_stats(gm.PROJ_minus_Raster_m).items() if key != "n"},
            "tolerance_m": MODEL_TOLERANCE_M,
            "status": geoid_status,
        })
        v_success = int(vm.vdatum_status.eq("success").sum())
        vstats = _difference_stats(vm.PROJ_minus_VDatum_m)
        v_status = ("PASSED" if v_success == 20 and vstats["n"] == 20
                    and vstats["max_abs_m"] <= MODEL_TOLERANCE_M
                    else "INCOMPLETE" if v_success < 20 else "FAILED")
        vdatum_summaries.append({
            "granule_id": spec.granule_id,
            "successful_VDatum_calls": v_success,
            "PROJ_minus_VDatum_mean_m": vstats["mean_m"],
            "PROJ_minus_VDatum_median_m": vstats["median_m"],
            "PROJ_minus_VDatum_RMSE_m": vstats["rmse_m"],
            "PROJ_minus_VDatum_max_abs_m": vstats["max_abs_m"],
            "VDatum_minus_NGS_max_abs_m": _difference_stats(vm.VDatum_minus_NGS_height_m)["max_abs_m"],
            "model": "NAD83(2011) ellipsoid to NAVD88 using GEOID18",
            "epoch_assumption": "2010.0 input and output for same-frame numerical verification only",
            "tolerance_m": MODEL_TOLERANCE_M,
            "status": v_status,
        })
    geoid_summary = pd.DataFrame(geoid_summaries)
    vdatum_summary = pd.DataFrame(vdatum_summaries)
    geoid_complete = bool(geoid_summary.status.eq("PASSED").all())
    vdatum_complete = bool(vdatum_summary.status.eq("PASSED").all())
    if geoid_complete and vdatum_complete:
        software_status = "PASSED"
    elif geoid_summary.status.eq("FAILED").any() or vdatum_summary.status.eq("FAILED").any():
        software_status = "FAILED"
    else:
        software_status = "INCOMPLETE"

    summary = {
        "created_utc": _utc_now(),
        "software_numerical_verification": software_status,
        "geoid18_model_verification": geoid_summaries,
        "vdatum_same_frame_verification": vdatum_summaries,
        "source_frame_transformation_verification": "UNRESOLVED",
        "strict_CASALS_3DEP_vertical_closure": "NOT ACHIEVED",
        "coordinate_assumption": COORDINATE_ASSUMPTION,
        "sample_selection": {
            "points_per_granule": 20,
            "rule": "Deterministic max-min spatial sample from complete H5 lon/lat arrays; finite Refh/coordinates; SNR >= 5; selected pulse IDs are frozen and asserted",
            "uses_3DEP_or_residual": False,
            "pulse_hashes": {
                gid: physical.pulse_hash(group.pulse_index.to_numpy(dtype=np.int64))
                for gid, group in samples.groupby("granule_id", sort=True)
            },
        },
        "model_tolerance_m": MODEL_TOLERANCE_M,
        "tolerance_basis": "Both NOAA services returned metre heights rounded to millimetres in this run; the 0.01 m acceptance threshold allows that output rounding and independent raster interpolation arithmetic.",
        "ngs_service": {"url": NGS_GEOID_URL, "documentation": NGS_DOC_URL,
                        "model_id": 14, "expected_model": "GEOID18", "units": "m"},
        "vdatum_service": {"url": VDATUM_URL, "documentation": VDATUM_DOC_URL,
                           "source_frame": "NAD83_2011 ellipsoid", "target_frame": "NAVD88",
                           "geoid_model": "geoid18", "coordinate_system": "geographic",
                           "height_units": "m", "input_epoch": 2010.0, "output_epoch": 2010.0,
                           "service_version": None,
                           "service_version_note": "Full API response does not expose a software version field"},
        "proj_environment": {"pyproj_version": pyproj.__version__, "proj_version": pyproj.proj_version_str,
                             "grid_name": datum.GEOID18_GRID, "grid_sha256": physical.GRID_SHA256,
                             "operation_by_granule": operation_records},
        "cache": {"prior_valid_response_count": len(cache),
                  "reused_call_count": int(sum(bool(row.get("cache_reused")) for row in call_records))},
    }
    input_manifest_path = output.parent / "input_sha256.csv"
    run_inputs = (
        pd.read_csv(input_manifest_path).to_dict(orient="records")
        if input_manifest_path.is_file() else []
    )
    code_paths = {
        "geodetic_crosscheck.py": Path(__file__).resolve(),
        "vertical_alignment.py": Path(physical.__file__).resolve(),
        "vertical_datum_comparison.py": Path(datum.__file__).resolve(),
    }
    notebook_path = root / "notebooks/L1B/geolocation/vertical_datum_alignment.ipynb"
    code_sha256 = {
        name: {"path": str(path.relative_to(root)), "sha256": physical.sha256(path)}
        for name, path in code_paths.items()
    }
    if notebook_path.is_file():
        notebook_json = json.loads(notebook_path.read_text(encoding="utf-8"))
        notebook_source = {
            "metadata": notebook_json.get("metadata", {}),
            "cells": [
                {"cell_type": cell["cell_type"], "source": cell.get("source", [])}
                for cell in notebook_json.get("cells", [])
            ],
        }
        notebook_source_bytes = json.dumps(
            notebook_source, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        code_sha256["vertical_datum_alignment.ipynb source"] = {
            "path": str(notebook_path.relative_to(root)),
            "sha256": hashlib.sha256(notebook_source_bytes).hexdigest(),
            "hash_scope": "metadata and cell source only; execution outputs excluded",
        }
    summary["source_provenance"] = {
        "run_alignment_input_sha256_manifest": run_inputs,
        "code_sha256": code_sha256,
        "input_manifest_available": bool(run_inputs),
    }
    _write_json(output / "crosscheck_summary.json", summary)
    _write_json(output / "service_requests_responses.json", {
        "created_utc": summary["created_utc"],
        "sources": {"NGS": NGS_GEOID_URL, "VDatum": VDATUM_URL,
                    "NGS_documentation": NGS_DOC_URL, "VDatum_documentation": VDATUM_DOC_URL},
        "call_count": len(call_records),
        "calls": call_records,
    })
    figures = plot_software_differences(geoid, vdatum_frame, output / "software_differences.png")
    interpretation = _interpretation(summary)
    return {
        "sample_points": samples,
        "geoid_comparison": geoid,
        "vdatum_comparison": vdatum_frame,
        "geoid_summary": geoid_summary,
        "vdatum_summary": vdatum_summary,
        "summary": summary,
        "interpretation": interpretation,
        "figure": figures,
        "output": output,
    }


def plot_software_differences(geoid: pd.DataFrame, vdatum: pd.DataFrame, output: Path):
    """Plot observed API/local software differences only, grouped by granule."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
    categories = (
        ("PROJ - NGS", "PROJ_minus_NGS_m", "#2c7fb8", geoid),
        ("Raster - NGS", "Raster_minus_NGS_m", "#41ab5d", geoid),
        ("PROJ - VDatum", "PROJ_minus_VDatum_m", "#f28e2b", vdatum),
    )
    for ax, spec in zip(axes, datum.GRANULES):
        for x, (label, column, color, frame) in enumerate(categories):
            values = frame.loc[frame.granule_id.eq(spec.granule_id), column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                spread = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.zeros(1)
                ax.scatter(x + spread, values, s=22, color=color, alpha=0.8,
                           label=label if ax is axes[0] else None)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xticks(range(3), [item[0] for item in categories], rotation=18, ha="right")
        ax.set_title(spec.granule_id.split("_")[2][:8])
        ax.grid(axis="y", alpha=0.22)
        ax.set_xlabel("Software comparison")
    axes[0].set_ylabel("Height difference (m)")
    fig.suptitle("Independent GEOID18 software differences | 20 fixed H5 pulses per granule")
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def _interpretation(summary: Mapping[str, Any]) -> str:
    geoid = summary["geoid18_model_verification"]
    vdatum = summary["vdatum_same_frame_verification"]
    geo_lines = []
    for row in geoid:
        geo_lines.append(
            f"- {row['granule_id'].split('_')[2][:8]}: {row['verification_points']} fixed pulses; "
            f"PROJ–NGS max |ΔN| {row['PROJ_vs_NGS_max_abs_m']!s} m; "
            f"raster–NGS max |ΔN| {row['Raster_vs_NGS_max_abs_m']!s} m; "
            f"PROJ–raster max |ΔN| {row['PROJ_vs_Raster_max_abs_m']!s} m; "
            f"NGS calls {row['successful_NGS_calls']}/20 ({row['status']})."
        )
    v_lines = []
    for row in vdatum:
        v_lines.append(
            f"- {row['granule_id'].split('_')[2][:8]}: VDatum calls {row['successful_VDatum_calls']}/20; "
            f"PROJ–VDatum mean {row['PROJ_minus_VDatum_mean_m']!s} m, "
            f"RMSE {row['PROJ_minus_VDatum_RMSE_m']!s} m, "
            f"maximum absolute difference {row['PROJ_minus_VDatum_max_abs_m']!s} m "
            f"({row['status']}; {row['model']}; epoch 2010.0 in/out)."
        )
    return "\n".join([
        "**Does PROJ compute GEOID18 correctly?** Under the explicitly assumed NAD83(2011) coordinates, the numerical result is compared against both an independent rasterio interpolation and NOAA NGS GEOID18 service output:",
        *geo_lines,
        "",
        "**Does NOAA VDatum agree under the same frame assumption?**",
        *v_lines,
        "",
        "**Can software differences explain a CASALS–3DEP residual?** These comparisons quantify numerical model and conversion implementations only. They do not use 3DEP values to choose samples, methods, or parameters. The independent 3DEP residual is reported separately in the following evaluation section.",
        "",
        "**What remains unresolved?** The API comparison assigns the same numeric coordinates to NAD83(2011), epoch 2010.0 for verification. CASALS WGS84 realization, coordinate epoch, and any upstream frame conversion remain unknown; the installed WGS84-ensemble-to-NAD83(2011) step remains a no-op. The software agreement therefore does not prove a CASALS source-frame conversion or strict vertical closure. No additional physical correction is applied.",
        "",
        f"**Classification:** Software numerical verification **{summary['software_numerical_verification']}**; source-frame transformation verification **UNRESOLVED**; strict CASALS–3DEP vertical closure **NOT ACHIEVED**.",
    ])
