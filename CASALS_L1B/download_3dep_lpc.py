# -*- coding: utf-8 -*-
"""
Robustly download USGS 3DEP lidar point-cloud clips covering a CASALS L1B H5 footprint.

This script is designed for the current CASALS workflow:
  1. Read the CASALS H5 georeferenced reference-return footprint from
     refh_longitude/refh_latitude.
  2. Query the USGS 3DEP Elevation Index Lidar Point Cloud layer.
  3. Prefer current 3DEP-compliant LPC workunits over legacy/non-compliant workunits.
  4. Resolve public Entwine Point Tile (EPT) resources.
  5. Extract full-density EPT-derived LAZ/LAS clips with PDAL.
  6. Write explicit manifests, sidecars, and success/skipped/failure accounting.

Important semantics
-------------------
- The output 3DEP clips are EPT-derived clips, not archival copies of original USGS
  source LAZ tiles.
- The WGS84 bbox is used for index lookup only. Extraction uses a local projected
  polygon so the clip footprint remains geometrically controlled.
- CASALS refh is WGS84 ellipsoidal height. 3DEP LPC normally uses its own horizontal
  and vertical CRS/geoid. This script preserves/records the 3DEP project metadata;
  it does not transform vertical datums.

Dependencies
------------
conda install -c conda-forge h5py numpy pandas pyproj requests pdal python-pdal tqdm

Run
---
python download_3dep_lpc.py

Configuration is defined in main(); no argparse is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlparse

import h5py
import numpy as np
import pandas as pd
import requests
from pyproj import CRS, Transformer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass(frozen=True)
class Config:
    h5_path: Path
    out_root: Path
    point_cloud_dir: Path
    clip_buffer_m: float = 30.0
    output_format: str = "laz"  # "laz" or "las"
    dry_run: bool = False
    overwrite: bool = False

    # 3DEP LPC index and EPT root.
    lpc_index_endpoint: str = "https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/8/query"
    ept_bucket_root: str = "https://usgs-lidar-public.s3.amazonaws.com"

    # HTTP behavior.
    request_timeout_connect_s: int = 15
    request_timeout_read_s: int = 180
    user_agent: str = "CASALS-3DEP-EPT-H5-Footprint-Clipper/2.0"
    result_record_count: int = 2000

    # Footprint CRS used to build the extraction polygon.
    clip_crs_mode: str = "nad83_utm"  # "nad83_utm" or "wgs84_utm"
    clip_crs_epsg_override: Optional[int] = None

    # Workunit selection.
    # Current default: use preferred 3DEP-compliant workunits if available, and
    # record old legacy/non-compliant workunits as skipped instead of hard failures.
    prefer_3dep_meets_workunits: bool = True
    try_nonpreferred_if_no_preferred: bool = True
    skip_unresolved_nonpreferred_as_skipped: bool = True

    # Clip planning.
    # all_usable_ept: extract every usable preferred EPT resource.
    # best_one: extract only the highest-ranked usable EPT resource.
    clip_selection_policy: str = "all_usable_ept"  # "all_usable_ept" or "best_one"

    # Output horizontal CRS.
    # lpc_horiz_crs: reproject XY to the LPC index horizontal CRS if parseable.
    # clip_crs: reproject XY to the clip polygon CRS.
    # ept_native: keep EPT's native horizontal CRS.
    output_crs_mode: str = "lpc_horiz_crs"  # "lpc_horiz_crs", "clip_crs", "ept_native"

    # EPT/Writer options.
    allow_missing_ept_crs: bool = False
    las_minor_version: int = 4
    fallback_las_dataformat_id: int = 6
    fallback_las_scale_xyz_m: float = 0.01
    las_offset: str = "auto"

    # Verification.
    require_at_least_one_available_clip: bool = True
    run_pdal_info_postcheck: bool = True


@dataclass(frozen=True)
class H5Footprint:
    source_h5: str
    n_records: int
    n_valid_lonlat: int
    longitude_min: float
    longitude_max: float
    latitude_min: float
    latitude_max: float
    longitude_p50: float
    latitude_p50: float
    clip_crs: str
    clip_crs_name: str
    center_x_clip_crs: float
    center_y_clip_crs: float
    local_xmin_m: float
    local_ymin_m: float
    local_xmax_m: float
    local_ymax_m: float
    width_m: float
    height_m: float
    clip_buffer_m: float
    clip_polygon_wkt: str
    clip_local_bbox: str
    query_bbox_wgs84: str
    start_utca: str
    end_utca: str
    n_pulses_attr: Any
    n_sweeps_attr: Any
    n_tracks_attr: Any


@dataclass(frozen=True)
class WorkunitRef:
    source_index: int
    workunit: str
    workunit_id: str
    project: str
    project_id: str
    collect_start: str
    collect_end: str
    ql: str
    spec: str
    p_method: str
    dem_gsd_meters: str
    horiz_crs: str
    vert_crs: str
    geoid: str
    lpc_pub_date: str
    lpc_category: str
    lpc_reason: str
    lpc_link: str
    sourcedem_link: str
    metadata_link: str


@dataclass(frozen=True)
class WorkunitStatus:
    source_index: int
    workunit: str
    project: str
    lpc_category: str
    lpc_reason: str
    ql: str
    collect_start: str
    collect_end: str
    horiz_crs: str
    vert_crs: str
    geoid: str
    status: str
    message: str


@dataclass(frozen=True)
class QueryFailure:
    stage: str
    target: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class EptResource:
    source_index: int
    ept_prefix: str
    ept_url: str
    bounds_json: str
    srs_json: str
    schema_json: str
    ept_srs_wkt: str
    ept_srs_user_input: str
    las_dataformat_id: int
    las_scale_x: float
    las_scale_y: float
    las_scale_z: float


@dataclass(frozen=True)
class ClipPlan:
    clip_id: str
    ept_url: str
    ept_prefix: str
    output_path: str
    sidecar_path: str
    clip_crs: str
    clip_polygon_wkt: str
    clip_local_bbox: str
    query_bbox_wgs84: str
    ept_srs_user_input: str
    ept_srs_wkt: str
    output_crs_mode: str
    output_crs_user_input: str
    output_srs_wkt: str
    horizontal_reprojection_applied: bool
    vertical_datum_transform_applied: bool
    las_storage_crs_source: str
    horiz_crs: str
    vert_crs: str
    geoid: str
    workunit: str
    project: str
    ql: str
    collect_start: str
    collect_end: str
    lpc_category: str
    lpc_reason: str
    metadata_link: str
    las_minor_version: int
    las_dataformat_id: int
    las_scale_x: float
    las_scale_y: float
    las_scale_z: float
    output_format: str


@dataclass(frozen=True)
class ClipResult:
    clip_id: str
    ept_url: str
    output_path: str
    sidecar_path: str
    status: str
    point_count: int
    bytes_written: int
    pdal_info_status: str = "not_run"
    pdal_info_point_count: int = -1
    error_message: str = ""


# =============================================================================
# Generic utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def maybe_progress(iterable, total: int | None = None, desc: str = ""):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        if desc:
            print(f"{desc}: tqdm unavailable; continuing without progress bar.")
        return iterable


def sanitize_name(value: Any, max_len: int = 120) -> str:
    text = str(value).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("._-")
    return (cleaned or "unnamed")[:max_len]


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha1_text(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def bbox_to_string(bbox: tuple[float, float, float, float]) -> str:
    return ",".join(f"{v:.8f}" for v in bbox)


def parse_bbox_string(bbox_str: str) -> tuple[float, float, float, float]:
    parts = [float(v) for v in str(bbox_str).split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected four bbox values, got {bbox_str!r}")
    return parts[0], parts[1], parts[2], parts[3]


def bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def transform_bbox_between_crs(
    bbox: tuple[float, float, float, float], src_crs: str | CRS, dst_crs: str | CRS
) -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xmin, ymin, xmax, ymax = bbox
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    coords = [transformer.transform(x, y) for x, y in corners]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return min(xs), min(ys), max(xs), max(ys)


def epoch_millis_to_utc_date(value: Any) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        ms = int(float(value))
        return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000.0))
    except Exception:
        return ""


def normalize_h5_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# =============================================================================
# HTTP / services
# =============================================================================

def build_session(cfg: Config) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update({"User-Agent": cfg.user_agent})
    return sess


def fetch_json(session: requests.Session, cfg: Config, url: str, params: dict[str, Any] | None = None) -> Any:
    response = session.get(
        url,
        params=params,
        timeout=(cfg.request_timeout_connect_s, cfg.request_timeout_read_s),
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    if isinstance(payload, dict) and payload.get("errors"):
        raise RuntimeError(f"API errors: {payload['errors']}")
    return payload


def fetch_optional_json(session: requests.Session, cfg: Config, url: str) -> Any | None:
    response = session.get(url, timeout=(cfg.request_timeout_connect_s, cfg.request_timeout_read_s))
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


# =============================================================================
# H5 footprint and CRS
# =============================================================================

def utm_epsg_from_lonlat(lon: float, lat: float, mode: str) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    if mode == "wgs84_utm":
        return 32600 + zone if lat >= 0 else 32700 + zone
    if mode == "nad83_utm":
        if lat < 0:
            raise ValueError("NAD83 UTM EPSG 269xx is valid for northern UTM zones only.")
        return 26900 + zone
    raise ValueError(f"Unsupported clip_crs_mode={mode!r}")


def require_h5_1d(h5: h5py.File, name: str) -> np.ndarray:
    if name not in h5:
        raise KeyError(f"Required HDF5 dataset not found: {name}")
    arr = np.asarray(h5[name][...])
    if arr.ndim != 1:
        raise ValueError(f"Dataset {name!r} must be 1D, got shape {arr.shape}")
    return arr


def compute_h5_footprint(cfg: Config) -> H5Footprint:
    if not cfg.h5_path.exists():
        raise FileNotFoundError(f"Input H5 file not found: {cfg.h5_path}")

    with h5py.File(cfg.h5_path, "r") as h5:
        attrs = {k: normalize_h5_attr(v) for k, v in h5.attrs.items()}
        lon = require_h5_1d(h5, "refh_longitude").astype(np.float64)
        lat = require_h5_1d(h5, "refh_latitude").astype(np.float64)

    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (lon >= -180.0)
        & (lon <= 180.0)
        & (lat >= -90.0)
        & (lat <= 90.0)
    )
    if int(np.sum(valid)) == 0:
        raise RuntimeError("No valid refh_longitude/refh_latitude records found in the H5 file.")

    lon_v = lon[valid]
    lat_v = lat[valid]
    lon_min, lon_max = float(np.nanmin(lon_v)), float(np.nanmax(lon_v))
    lat_min, lat_max = float(np.nanmin(lat_v)), float(np.nanmax(lat_v))
    lon_med, lat_med = float(np.nanmedian(lon_v)), float(np.nanmedian(lat_v))

    if cfg.clip_crs_epsg_override is not None:
        clip_epsg = int(cfg.clip_crs_epsg_override)
    else:
        clip_epsg = utm_epsg_from_lonlat(lon_med, lat_med, cfg.clip_crs_mode)
    clip_crs = CRS.from_epsg(clip_epsg)
    clip_crs_str = f"EPSG:{clip_epsg}"

    to_local = Transformer.from_crs("EPSG:4326", clip_crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(clip_crs, "EPSG:4326", always_xy=True)

    x, y = to_local.transform(lon_v, lat_v)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite_xy = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite_xy)) == 0:
        raise RuntimeError("H5 lon/lat could not be transformed to the local clip CRS.")

    xmin = float(np.nanmin(x[finite_xy]) - cfg.clip_buffer_m)
    xmax = float(np.nanmax(x[finite_xy]) + cfg.clip_buffer_m)
    ymin = float(np.nanmin(y[finite_xy]) - cfg.clip_buffer_m)
    ymax = float(np.nanmax(y[finite_xy]) + cfg.clip_buffer_m)
    cx, cy = to_local.transform(lon_med, lat_med)

    local_ring = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    polygon_wkt = "POLYGON ((" + ", ".join(f"{px:.3f} {py:.3f}" for px, py in local_ring) + "))"

    lonlat_corners = [to_wgs84.transform(px, py) for px, py in local_ring[:-1]]
    corner_lons = [p[0] for p in lonlat_corners]
    corner_lats = [p[1] for p in lonlat_corners]
    query_bbox_wgs84 = bbox_to_string((min(corner_lons), min(corner_lats), max(corner_lons), max(corner_lats)))

    return H5Footprint(
        source_h5=str(cfg.h5_path.resolve()),
        n_records=int(lon.size),
        n_valid_lonlat=int(np.sum(valid)),
        longitude_min=lon_min,
        longitude_max=lon_max,
        latitude_min=lat_min,
        latitude_max=lat_max,
        longitude_p50=lon_med,
        latitude_p50=lat_med,
        clip_crs=clip_crs_str,
        clip_crs_name=clip_crs.name,
        center_x_clip_crs=float(cx),
        center_y_clip_crs=float(cy),
        local_xmin_m=xmin,
        local_ymin_m=ymin,
        local_xmax_m=xmax,
        local_ymax_m=ymax,
        width_m=float(xmax - xmin),
        height_m=float(ymax - ymin),
        clip_buffer_m=float(cfg.clip_buffer_m),
        clip_polygon_wkt=polygon_wkt,
        clip_local_bbox=bbox_to_string((xmin, ymin, xmax, ymax)),
        query_bbox_wgs84=query_bbox_wgs84,
        start_utca=str(attrs.get("start_utca", "")),
        end_utca=str(attrs.get("end_utca", "")),
        n_pulses_attr=attrs.get("n_pulses", ""),
        n_sweeps_attr=attrs.get("n_sweeps", ""),
        n_tracks_attr=attrs.get("n_tracks", ""),
    )


# =============================================================================
# LPC index / EPT resolution
# =============================================================================

def lpc_out_fields() -> str:
    return ",".join(
        [
            "workunit", "workunit_id", "project", "project_id", "collect_start", "collect_end",
            "ql", "spec", "p_method", "dem_gsd_meters", "horiz_crs", "vert_crs", "geoid",
            "lpc_pub_date", "lpc_category", "lpc_reason", "lpc_link", "sourcedem_link", "metadata_link",
        ]
    )


def item_text(attrs: dict[str, Any], key: str) -> str:
    value = attrs.get(key)
    return "" if value is None else str(value).strip()


def query_lpc_workunits(session: requests.Session, cfg: Config, fp: H5Footprint) -> tuple[list[WorkunitRef], list[QueryFailure]]:
    params_base = {
        "f": "pjson",
        "geometry": fp.query_bbox_wgs84,
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "false",
        "outFields": lpc_out_fields(),
        "resultRecordCount": int(cfg.result_record_count),
    }

    refs: list[WorkunitRef] = []
    failures: list[QueryFailure] = []
    seen: set[tuple[str, str, str]] = set()
    offset = 0

    try:
        while True:
            payload = fetch_json(session, cfg, cfg.lpc_index_endpoint, params={**params_base, "resultOffset": offset})
            features = payload.get("features", []) if isinstance(payload, dict) else []
            for feature in features:
                attrs = feature.get("attributes", {}) if isinstance(feature, dict) else {}
                workunit = item_text(attrs, "workunit")
                project = item_text(attrs, "project")
                lpc_link = item_text(attrs, "lpc_link")
                key = (workunit, project, lpc_link)
                if not (workunit or project or lpc_link) or key in seen:
                    continue
                refs.append(
                    WorkunitRef(
                        source_index=len(refs),
                        workunit=workunit,
                        workunit_id=item_text(attrs, "workunit_id"),
                        project=project,
                        project_id=item_text(attrs, "project_id"),
                        collect_start=epoch_millis_to_utc_date(attrs.get("collect_start")),
                        collect_end=epoch_millis_to_utc_date(attrs.get("collect_end")),
                        ql=item_text(attrs, "ql"),
                        spec=item_text(attrs, "spec"),
                        p_method=item_text(attrs, "p_method"),
                        dem_gsd_meters=item_text(attrs, "dem_gsd_meters"),
                        horiz_crs=item_text(attrs, "horiz_crs"),
                        vert_crs=item_text(attrs, "vert_crs"),
                        geoid=item_text(attrs, "geoid"),
                        lpc_pub_date=epoch_millis_to_utc_date(attrs.get("lpc_pub_date")),
                        lpc_category=item_text(attrs, "lpc_category"),
                        lpc_reason=item_text(attrs, "lpc_reason"),
                        lpc_link=lpc_link,
                        sourcedem_link=item_text(attrs, "sourcedem_link"),
                        metadata_link=item_text(attrs, "metadata_link"),
                    )
                )
                seen.add(key)
            exceeded = bool(payload.get("exceededTransferLimit")) if isinstance(payload, dict) else False
            if not exceeded or not features:
                break
            offset += len(features)
    except Exception as exc:
        failures.append(QueryFailure("lpc_index_query", cfg.lpc_index_endpoint, type(exc).__name__, str(exc)))

    return refs, failures


def ql_rank(ql: str) -> int:
    text = str(ql).upper()
    for n in range(0, 10):
        if f"QL {n}" in text or text == f"QL{n}":
            return n
    return 99


def parse_date_rank(date_text: str) -> int:
    # Higher is newer. Empty dates rank lowest.
    try:
        if not date_text:
            return 0
        return int(str(date_text).replace("-", "")[:8])
    except Exception:
        return 0


def is_preferred_lpc(ref: WorkunitRef) -> bool:
    category = ref.lpc_category.lower()
    reason = ref.lpc_reason.lower()
    ql = ref.ql.lower()
    legacy = "legacy" in ref.project.lower() or "legacy" in ref.workunit.lower() or "legacy" in ref.lpc_link.lower()
    does_not_meet = "does not meet" in category or "predates" in reason
    is_meets = "meets" in category and "does not meet" not in category
    return bool(is_meets and not does_not_meet and not legacy and ql != "other")


def workunit_sort_key(ref: WorkunitRef) -> tuple[int, int, int, int]:
    # Lower tuple sorts first. Prefer 3DEP compliant, better QL, finer DEM GSD, newer collection.
    preferred = 0 if is_preferred_lpc(ref) else 1
    qrank = ql_rank(ref.ql)
    try:
        gsd = int(round(float(ref.dem_gsd_meters) * 100))
    except Exception:
        gsd = 999999
    newer = -parse_date_rank(ref.collect_end or ref.lpc_pub_date)
    return preferred, qrank, gsd, newer


def select_workunits_for_resolution(cfg: Config, refs: list[WorkunitRef]) -> tuple[list[WorkunitRef], list[WorkunitStatus]]:
    statuses: list[WorkunitStatus] = []
    if not refs:
        return [], statuses

    sorted_refs = sorted(refs, key=workunit_sort_key)
    preferred_refs = [r for r in sorted_refs if is_preferred_lpc(r)]

    if cfg.prefer_3dep_meets_workunits and preferred_refs:
        selected = preferred_refs
        selected_ids = {r.source_index for r in selected}
        for r in sorted_refs:
            if r.source_index in selected_ids:
                statuses.append(workunit_status(r, "selected_for_ept_resolution", "Preferred 3DEP-compliant LPC workunit."))
            else:
                statuses.append(workunit_status(r, "skipped_lower_priority", "Skipped because at least one preferred 3DEP-compliant LPC workunit covers the footprint."))
        return selected, statuses

    if cfg.try_nonpreferred_if_no_preferred:
        for r in sorted_refs:
            statuses.append(workunit_status(r, "selected_for_ept_resolution", "No preferred workunit available; trying this workunit as fallback."))
        return sorted_refs, statuses

    for r in sorted_refs:
        statuses.append(workunit_status(r, "skipped_nonpreferred", "Non-preferred LPC workunit and try_nonpreferred_if_no_preferred=False."))
    return [], statuses


def workunit_status(ref: WorkunitRef, status: str, message: str) -> WorkunitStatus:
    return WorkunitStatus(
        source_index=ref.source_index,
        workunit=ref.workunit,
        project=ref.project,
        lpc_category=ref.lpc_category,
        lpc_reason=ref.lpc_reason,
        ql=ref.ql,
        collect_start=ref.collect_start,
        collect_end=ref.collect_end,
        horiz_crs=ref.horiz_crs,
        vert_crs=ref.vert_crs,
        geoid=ref.geoid,
        status=status,
        message=message,
    )


def ept_candidate_prefixes(ref: WorkunitRef) -> list[str]:
    candidates = [
        ref.workunit,
        ref.project,
        Path(urlparse(ref.lpc_link).path).name,
        Path(urlparse(ref.lpc_link).path).parent.name,
    ]
    out: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def ept_crs_from_srs(srs: Any) -> CRS | None:
    if not isinstance(srs, dict):
        return None
    # Prefer explicit WKT because some EPT srs objects include horizontal="3857" but no separate
    # vertical CRS; WKT usually captures the intended native horizontal CRS.
    for key in ("wkt", "compoundwkt", "prettywkt"):
        value = srs.get(key)
        if isinstance(value, str) and value.strip():
            try:
                return CRS.from_wkt(value)
            except Exception:
                pass
    authority, code = srs.get("authority"), srs.get("code")
    if authority and code:
        try:
            return CRS.from_user_input(f"{authority}:{code}")
        except Exception:
            pass
    for key in ("horizontal", "authority"):
        value = srs.get(key)
        if isinstance(value, str) and value.strip():
            try:
                # Bare horizontal code in USGS EPT often means EPSG code.
                return CRS.from_user_input(f"EPSG:{value.strip()}" if value.strip().isdigit() else value.strip())
            except Exception:
                pass
    return None


def ept_srs_user_input(srs: Any) -> str:
    if not isinstance(srs, dict):
        return ""
    authority, code = srs.get("authority"), srs.get("code")
    if authority and code:
        return f"{authority}:{code}"
    horizontal = srs.get("horizontal")
    if horizontal:
        text = str(horizontal).strip()
        return f"EPSG:{text}" if text.isdigit() else text
    crs = ept_crs_from_srs(srs)
    return crs.to_string() if crs is not None else ""


def ept_bounds_xy(bounds: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bounds, list) or len(bounds) < 6:
        return None
    try:
        xmin, ymin, xmax, ymax = float(bounds[0]), float(bounds[1]), float(bounds[3]), float(bounds[4])
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        return xmin, ymin, xmax, ymax
    except Exception:
        return None


def schema_by_name(schema: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(schema, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for dim in schema:
        if isinstance(dim, dict) and dim.get("name"):
            out[str(dim["name"]).lower()] = dim
    return out


def schema_scale(dims: dict[str, dict[str, Any]], name: str, fallback: float) -> float:
    dim = dims.get(name.lower())
    if not dim:
        return fallback
    try:
        scale = float(dim.get("scale"))
        return scale if scale > 0 else fallback
    except Exception:
        return fallback


def infer_las_writer(schema: Any, cfg: Config) -> tuple[int, float, float, float]:
    dims = schema_by_name(schema)
    has_rgb = all(name in dims for name in ("red", "green", "blue"))
    has_nir = any(name in dims for name in ("infrared", "nir"))
    dataformat_id = 8 if has_rgb and has_nir else 7 if has_rgb else cfg.fallback_las_dataformat_id
    fallback = float(cfg.fallback_las_scale_xyz_m)
    return (
        dataformat_id,
        schema_scale(dims, "X", fallback),
        schema_scale(dims, "Y", fallback),
        schema_scale(dims, "Z", fallback),
    )


def resolve_ept_resource(
    session: requests.Session, cfg: Config, ref: WorkunitRef
) -> tuple[EptResource | None, list[QueryFailure]]:
    failures: list[QueryFailure] = []
    for prefix in ept_candidate_prefixes(ref):
        ept_url = f"{cfg.ept_bucket_root}/{quote(prefix)}/ept.json"
        try:
            payload = fetch_optional_json(session, cfg, ept_url)
            if payload is None:
                continue
            if not isinstance(payload, dict):
                raise RuntimeError("EPT metadata was not a JSON object.")
            if "bounds" not in payload:
                raise RuntimeError("EPT metadata did not contain a bounds field.")
            crs = ept_crs_from_srs(payload.get("srs"))
            ept_srs_wkt = crs.to_wkt() if crs is not None else ""
            dataformat_id, sx, sy, sz = infer_las_writer(payload.get("schema", []), cfg)
            return (
                EptResource(
                    source_index=ref.source_index,
                    ept_prefix=prefix,
                    ept_url=ept_url,
                    bounds_json=json.dumps(payload.get("bounds", [])),
                    srs_json=json.dumps(payload.get("srs", {})),
                    schema_json=json.dumps(payload.get("schema", [])),
                    ept_srs_wkt=ept_srs_wkt,
                    ept_srs_user_input=ept_srs_user_input(payload.get("srs")),
                    las_dataformat_id=dataformat_id,
                    las_scale_x=sx,
                    las_scale_y=sy,
                    las_scale_z=sz,
                ),
                failures,
            )
        except Exception as exc:
            failures.append(QueryFailure("ept_resolve_attempt", ept_url, type(exc).__name__, str(exc)))
    failures.append(
        QueryFailure(
            "ept_resolve",
            "candidate_prefixes=" + ";".join(ept_candidate_prefixes(ref)),
            "NoPublicEpt",
            "No usable public EPT dataset was found for this selected LPC workunit.",
        )
    )
    return None, failures


def ept_covers_footprint(ept: EptResource, fp: H5Footprint) -> tuple[bool, str]:
    bounds_xy = ept_bounds_xy(json.loads(ept.bounds_json))
    if bounds_xy is None:
        return False, "EPT metadata has no valid 3D bounds array."
    ept_crs = ept_crs_from_srs(json.loads(ept.srs_json))
    if ept_crs is None:
        return True, "EPT CRS could not be parsed; coverage accepted but not pre-verified."
    try:
        local_bbox = parse_bbox_string(fp.clip_local_bbox)
        bbox_in_ept = transform_bbox_between_crs(local_bbox, fp.clip_crs, ept_crs)
        if bbox_intersects(bounds_xy, bbox_in_ept):
            return True, "EPT bounds intersect the CASALS footprint clip."
        return False, "EPT bounds do not intersect the CASALS footprint clip."
    except Exception as exc:
        return True, f"EPT coverage transform failed; accepted but not pre-verified: {type(exc).__name__}: {exc}"


# =============================================================================
# Planning and extraction
# =============================================================================

def resolve_lpc_horizontal_crs(ref: WorkunitRef) -> CRS | None:
    text = str(ref.horiz_crs).strip()
    if not text:
        return None
    try:
        if text.isdigit():
            return CRS.from_epsg(int(text))
        return CRS.from_user_input(text)
    except Exception:
        return None


def desired_output_crs(cfg: Config, fp: H5Footprint, ref: WorkunitRef, ept: EptResource) -> tuple[str, str, str]:
    """Return (user_input, wkt, source_label)."""
    if cfg.output_crs_mode == "lpc_horiz_crs":
        crs = resolve_lpc_horizontal_crs(ref)
        if crs is not None:
            code = crs.to_authority()
            user_input = f"{code[0]}:{code[1]}" if code else crs.to_string()
            return user_input, crs.to_wkt(), "lpc_index_horiz_crs"
        # Fallback to clip CRS if LPC CRS is not parseable.
        crs = CRS.from_user_input(fp.clip_crs)
        return fp.clip_crs, crs.to_wkt(), "clip_crs_fallback_for_unparseable_lpc_horiz_crs"
    if cfg.output_crs_mode == "clip_crs":
        crs = CRS.from_user_input(fp.clip_crs)
        return fp.clip_crs, crs.to_wkt(), "clip_crs"
    if cfg.output_crs_mode == "ept_native":
        if ept.ept_srs_wkt:
            return ept.ept_srs_user_input or "ept_native", ept.ept_srs_wkt, "ept_native_srs"
        return "", "", "missing_ept_native_srs"
    raise ValueError(f"Unsupported output_crs_mode={cfg.output_crs_mode!r}")


def output_differs_from_ept(ept: EptResource, output_srs_wkt: str) -> bool:
    if not ept.ept_srs_wkt or not output_srs_wkt:
        return False
    try:
        return not CRS.from_wkt(ept.ept_srs_wkt).equals(CRS.from_wkt(output_srs_wkt))
    except Exception:
        return ept.ept_srs_wkt != output_srs_wkt


def build_clip_id(fp: H5Footprint, ref: WorkunitRef, ept: EptResource, cfg: Config, output_crs_user_input: str) -> str:
    return sha1_text(
        stable_json(
            {
                "source_h5": fp.source_h5,
                "clip_polygon_wkt": fp.clip_polygon_wkt,
                "clip_crs": fp.clip_crs,
                "clip_buffer_m": fp.clip_buffer_m,
                "query_bbox_wgs84": fp.query_bbox_wgs84,
                "ept_url": ept.ept_url,
                "ept_srs_user_input": ept.ept_srs_user_input,
                "output_crs_mode": cfg.output_crs_mode,
                "output_crs_user_input": output_crs_user_input,
                "horiz_crs": ref.horiz_crs,
                "vert_crs": ref.vert_crs,
                "geoid": ref.geoid,
                "output_format": cfg.output_format,
                "las_dataformat_id": ept.las_dataformat_id,
                "las_scale_x": ept.las_scale_x,
                "las_scale_y": ept.las_scale_y,
                "las_scale_z": ept.las_scale_z,
            }
        ),
        n=12,
    )


def build_clip_plan(cfg: Config, fp: H5Footprint, ref: WorkunitRef, ept: EptResource) -> ClipPlan:
    output_crs_user_input, output_srs_wkt, storage_source = desired_output_crs(cfg, fp, ref, ept)
    do_reproject = output_differs_from_ept(ept, output_srs_wkt)
    clip_id = build_clip_id(fp, ref, ept, cfg, output_crs_user_input)
    safe_prefix = sanitize_name(ept.ept_prefix, max_len=90)
    safe_h5 = sanitize_name(Path(fp.source_h5).stem, max_len=90)
    crs_tag = sanitize_name(output_crs_user_input.replace(":", ""), max_len=30) or "native"
    output_name = f"{safe_h5}_{safe_prefix}_{crs_tag}_{clip_id}.{cfg.output_format}"
    output_path = cfg.point_cloud_dir / output_name
    sidecar_path = cfg.out_root / "clip_sidecars" / f"{output_name}.json"
    return ClipPlan(
        clip_id=clip_id,
        ept_url=ept.ept_url,
        ept_prefix=ept.ept_prefix,
        output_path=str(output_path),
        sidecar_path=str(sidecar_path),
        clip_crs=fp.clip_crs,
        clip_polygon_wkt=fp.clip_polygon_wkt,
        clip_local_bbox=fp.clip_local_bbox,
        query_bbox_wgs84=fp.query_bbox_wgs84,
        ept_srs_user_input=ept.ept_srs_user_input,
        ept_srs_wkt=ept.ept_srs_wkt,
        output_crs_mode=cfg.output_crs_mode,
        output_crs_user_input=output_crs_user_input,
        output_srs_wkt=output_srs_wkt,
        horizontal_reprojection_applied=do_reproject,
        vertical_datum_transform_applied=False,
        las_storage_crs_source=storage_source,
        horiz_crs=ref.horiz_crs,
        vert_crs=ref.vert_crs,
        geoid=ref.geoid,
        workunit=ref.workunit,
        project=ref.project,
        ql=ref.ql,
        collect_start=ref.collect_start,
        collect_end=ref.collect_end,
        lpc_category=ref.lpc_category,
        lpc_reason=ref.lpc_reason,
        metadata_link=ref.metadata_link,
        las_minor_version=int(cfg.las_minor_version),
        las_dataformat_id=int(ept.las_dataformat_id),
        las_scale_x=float(ept.las_scale_x),
        las_scale_y=float(ept.las_scale_y),
        las_scale_z=float(ept.las_scale_z),
        output_format=cfg.output_format,
    )


def sidecar_payload(cfg: Config, fp: H5Footprint, plan: ClipPlan, result: ClipResult | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "script": "download_3dep_lpc.py",
        "script_semantics": "3DEP full_density_ept_clip_for_casals_h5_footprint_not_original_source_tile",
        "scientific_notes": [
            "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
            "refh is WGS84 ellipsoidal height unless otherwise documented.",
            "This is not an official multi-return point cloud.",
            "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            "3DEP clips here are EPT-derived clips, not archival source LAZ copies, and no vertical datum transform is applied.",
        ],
        "source_h5": fp.source_h5,
        "casals_footprint": asdict(fp),
        "clip_plan": asdict(plan),
        "config_subset": {
            "clip_buffer_m": cfg.clip_buffer_m,
            "output_format": cfg.output_format,
            "clip_crs_mode": cfg.clip_crs_mode,
            "clip_crs_epsg_override": cfg.clip_crs_epsg_override,
            "prefer_3dep_meets_workunits": cfg.prefer_3dep_meets_workunits,
            "try_nonpreferred_if_no_preferred": cfg.try_nonpreferred_if_no_preferred,
            "clip_selection_policy": cfg.clip_selection_policy,
            "output_crs_mode": cfg.output_crs_mode,
            "allow_missing_ept_crs": cfg.allow_missing_ept_crs,
            "vertical_datum_transform_applied": False,
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if result is not None:
        payload["clip_result"] = asdict(result)
    return payload


def existing_clip_is_valid(plan: ClipPlan) -> tuple[bool, int, int, str]:
    out = Path(plan.output_path)
    sidecar = Path(plan.sidecar_path)
    if not out.exists() or out.stat().st_size <= 0:
        return False, 0, -1, "clip file missing or empty"
    if not sidecar.exists():
        return False, int(out.stat().st_size), -1, "sidecar missing"
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        stored_plan = meta.get("clip_plan", {})
        required = [
            "clip_id",
            "ept_url",
            "clip_polygon_wkt",
            "clip_crs",
            "output_crs_user_input",
            "output_format",
            "las_dataformat_id",
        ]
        current = asdict(plan)
        for key in required:
            if stored_plan.get(key) != current.get(key):
                return False, int(out.stat().st_size), -1, f"sidecar mismatch for {key}"
        point_count = int(meta.get("clip_result", {}).get("point_count", -1))
        return True, int(out.stat().st_size), point_count, "existing clip matches"
    except Exception as exc:
        return False, int(out.stat().st_size), -1, f"sidecar unreadable: {type(exc).__name__}: {exc}"


def get_pdal_module():
    try:
        import pdal  # type: ignore
        return pdal
    except Exception as exc:
        raise RuntimeError(
            "PDAL python bindings are not installed. Install with: conda install -c conda-forge pdal python-pdal"
        ) from exc


def run_pdal_info_summary(path: Path) -> tuple[str, int, str]:
    try:
        proc = subprocess.run(
            ["pdal", "info", "--summary", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
        if proc.returncode != 0:
            return "failed", -1, proc.stderr.strip() or proc.stdout.strip()
        payload = json.loads(proc.stdout)
        # PDAL summary JSON has had several layouts across versions. Try common fields.
        point_count = -1
        if isinstance(payload, dict):
            if "summary" in payload and isinstance(payload["summary"], dict):
                point_count = int(payload["summary"].get("num_points", payload["summary"].get("points", -1)))
            if point_count < 0:
                point_count = int(payload.get("num_points", payload.get("points", -1)))
        return "ok", point_count, ""
    except FileNotFoundError:
        return "skipped", -1, "pdal executable not found on PATH"
    except Exception as exc:
        return "failed", -1, f"{type(exc).__name__}: {exc}"


def extract_one_clip(cfg: Config, fp: H5Footprint, plan: ClipPlan) -> ClipResult:
    out = Path(plan.output_path)
    sidecar = Path(plan.sidecar_path)
    ensure_dir(out.parent)

    if not cfg.overwrite:
        valid, nbytes, point_count, _ = existing_clip_is_valid(plan)
        if valid:
            pdal_status, pdal_count, pdal_msg = ("not_run", -1, "")
            if cfg.run_pdal_info_postcheck:
                pdal_status, pdal_count, pdal_msg = run_pdal_info_summary(out)
            return ClipResult(
                clip_id=plan.clip_id,
                ept_url=plan.ept_url,
                output_path=str(out),
                sidecar_path=str(sidecar),
                status="already_exists_sidecar_matched",
                point_count=point_count,
                bytes_written=nbytes,
                pdal_info_status=pdal_status,
                pdal_info_point_count=pdal_count,
                error_message=pdal_msg,
            )

    if cfg.dry_run:
        result = ClipResult(
            clip_id=plan.clip_id,
            ept_url=plan.ept_url,
            output_path=str(out),
            sidecar_path=str(sidecar),
            status="planned_dry_run",
            point_count=0,
            bytes_written=0,
        )
        write_json(sidecar, sidecar_payload(cfg, fp, plan, result))
        return result

    pdal = get_pdal_module()
    tmp_out = out.parent / f"{out.stem}.part{out.suffix}"
    tmp_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    polygon_clause = f"{plan.clip_polygon_wkt}/{plan.clip_crs}"

    writer: dict[str, Any] = {
        "type": "writers.las",
        "filename": str(tmp_out),
        "minor_version": plan.las_minor_version,
        "dataformat_id": plan.las_dataformat_id,
        "extra_dims": "all",
        "scale_x": plan.las_scale_x,
        "scale_y": plan.las_scale_y,
        "scale_z": plan.las_scale_z,
        "offset_x": cfg.las_offset,
        "offset_y": cfg.las_offset,
        "offset_z": cfg.las_offset,
        "compression": cfg.output_format == "laz",
    }
    if plan.output_srs_wkt:
        writer["a_srs"] = plan.output_srs_wkt

    pipeline_spec: list[dict[str, Any]] = [
        {"type": "readers.ept", "filename": plan.ept_url, "polygon": polygon_clause},
    ]
    if plan.horizontal_reprojection_applied:
        pipeline_spec.append({"type": "filters.reprojection", "out_srs": plan.output_srs_wkt or plan.output_crs_user_input})
    pipeline_spec.append(writer)

    try:
        if tmp_out.exists():
            tmp_out.unlink()
        if tmp_sidecar.exists():
            tmp_sidecar.unlink()
        pipeline = pdal.Pipeline(json.dumps(pipeline_spec))
        point_count = int(pipeline.execute())
        if point_count <= 0:
            raise RuntimeError("PDAL EPT clip returned zero points.")
        if not tmp_out.exists() or tmp_out.stat().st_size <= 0:
            raise RuntimeError("PDAL pipeline completed, but output clip is missing or empty.")
        tmp_bytes = int(tmp_out.stat().st_size)
        pdal_status, pdal_count, pdal_msg = ("not_run", -1, "")
        # Postcheck after final rename, not on the .part file.
        result = ClipResult(
            clip_id=plan.clip_id,
            ept_url=plan.ept_url,
            output_path=str(out),
            sidecar_path=str(sidecar),
            status="extracted",
            point_count=point_count,
            bytes_written=tmp_bytes,
            pdal_info_status=pdal_status,
            pdal_info_point_count=pdal_count,
            error_message="",
        )
        write_json(tmp_sidecar, sidecar_payload(cfg, fp, plan, result))
        tmp_out.replace(out)
        tmp_sidecar.replace(sidecar)
        if cfg.run_pdal_info_postcheck:
            pdal_status, pdal_count, pdal_msg = run_pdal_info_summary(out)
            result = ClipResult(**{**asdict(result), "bytes_written": int(out.stat().st_size), "pdal_info_status": pdal_status, "pdal_info_point_count": pdal_count, "error_message": pdal_msg})
            write_json(sidecar, sidecar_payload(cfg, fp, plan, result))
        return result
    except Exception as exc:
        try:
            if tmp_out.exists():
                tmp_out.unlink()
            if tmp_sidecar.exists():
                tmp_sidecar.unlink()
        except Exception:
            pass
        return ClipResult(
            clip_id=plan.clip_id,
            ept_url=plan.ept_url,
            output_path=str(out),
            sidecar_path=str(sidecar),
            status="failed",
            point_count=0,
            bytes_written=0,
            error_message=f"{type(exc).__name__}: {exc}",
        )


# =============================================================================
# Main workflow
# =============================================================================

def validate_config(cfg: Config) -> None:
    if cfg.output_format not in {"laz", "las"}:
        raise ValueError("output_format must be 'laz' or 'las'.")
    if cfg.clip_selection_policy not in {"all_usable_ept", "best_one"}:
        raise ValueError("clip_selection_policy must be 'all_usable_ept' or 'best_one'.")
    if cfg.output_crs_mode not in {"lpc_horiz_crs", "clip_crs", "ept_native"}:
        raise ValueError("output_crs_mode must be 'lpc_horiz_crs', 'clip_crs', or 'ept_native'.")
    if cfg.clip_buffer_m < 0:
        raise ValueError("clip_buffer_m must be >= 0.")
    if cfg.result_record_count <= 0:
        raise ValueError("result_record_count must be > 0.")
    if not cfg.dry_run:
        get_pdal_module()


def run(cfg: Config) -> None:
    validate_config(cfg)
    ensure_dir(cfg.out_root / "manifests")
    ensure_dir(cfg.out_root / "clip_sidecars")
    ensure_dir(cfg.point_cloud_dir)
    session = build_session(cfg)

    print("=" * 80)
    print("3DEP LPC download for CASALS L1B H5 footprint")
    print("=" * 80)
    print(f"H5: {cfg.h5_path.resolve()}")
    print(f"Output root: {cfg.out_root.resolve()}")
    print(f"Point-cloud directory: {cfg.point_cloud_dir.resolve()}")
    print(f"Dry run: {cfg.dry_run}")
    print(f"Output CRS mode: {cfg.output_crs_mode}")
    print()

    fp = compute_h5_footprint(cfg)
    write_json(cfg.out_root / "manifests" / "casals_h5_footprint.json", asdict(fp))

    print("CASALS H5 footprint:")
    print(f"  valid lon/lat records: {fp.n_valid_lonlat:,} / {fp.n_records:,}")
    print(f"  lon range: {fp.longitude_min:.8f}, {fp.longitude_max:.8f}")
    print(f"  lat range: {fp.latitude_min:.8f}, {fp.latitude_max:.8f}")
    print(f"  clip CRS: {fp.clip_crs} ({fp.clip_crs_name})")
    print(f"  local clip size: {fp.width_m:.2f} m x {fp.height_m:.2f} m")
    print(f"  WGS84 query bbox: {fp.query_bbox_wgs84}")
    print()

    print("Querying USGS 3DEP Elevation Index Lidar Point Cloud layer...")
    refs, hard_failures = query_lpc_workunits(session, cfg, fp)
    refs = sorted(refs, key=workunit_sort_key)
    write_csv(cfg.out_root / "manifests" / "lpc_workunits.csv", [asdict(r) for r in refs], list(WorkunitRef.__dataclass_fields__.keys()))
    print(f"  LPC workunits intersecting query bbox: {len(refs)}")

    selected_refs, workunit_statuses = select_workunits_for_resolution(cfg, refs)
    print(f"  Selected workunits for EPT resolution: {len(selected_refs)}")
    print(f"  Skipped/lower-priority workunits: {max(0, len(refs) - len(selected_refs))}")

    print("Resolving public EPT resources...")
    ept_by_source: dict[int, EptResource] = {}
    soft_resolution_notes: list[QueryFailure] = []
    for ref in maybe_progress(selected_refs, total=len(selected_refs), desc="EPT resolve"):
        ept, ept_failures = resolve_ept_resource(session, cfg, ref)
        if ept is None:
            # Treat unresolved non-preferred/fallback records as hard only when no preferred selected, or if
            # they are selected and no EPT can be found. These are real failures for selected workunits.
            hard_failures.extend(ept_failures)
            workunit_statuses.append(workunit_status(ref, "selected_but_no_public_ept", "; ".join(f.error_message for f in ept_failures[-1:])))
        else:
            ept_by_source[ref.source_index] = ept
            workunit_statuses.append(workunit_status(ref, "public_ept_resolved", ept.ept_url))

    ept_rows = [asdict(ept) for ept in ept_by_source.values()]
    write_csv(cfg.out_root / "manifests" / "resolved_ept_resources.csv", ept_rows, list(EptResource.__dataclass_fields__.keys()))
    print(f"  Resolved EPT resources: {len(ept_by_source)}")

    print("Building clip plans...")
    clip_plans: list[ClipPlan] = []
    for ref in selected_refs:
        ept = ept_by_source.get(ref.source_index)
        if ept is None:
            continue
        if not ept.ept_srs_wkt and not cfg.allow_missing_ept_crs:
            hard_failures.append(
                QueryFailure(
                    "clip_plan",
                    ept.ept_url,
                    "MissingEptCrs",
                    "EPT metadata has no parseable CRS; skipped because allow_missing_ept_crs=False.",
                )
            )
            continue
        ok, note = ept_covers_footprint(ept, fp)
        if not ok:
            hard_failures.append(QueryFailure("coverage_check", ept.ept_url, "CoverageError", note))
            continue
        clip_plans.append(build_clip_plan(cfg, fp, ref, ept))

    # De-duplicate by EPT URL and clip ID.
    unique: dict[tuple[str, str], ClipPlan] = {}
    for plan in clip_plans:
        unique.setdefault((plan.ept_url, plan.clip_id), plan)
    clip_plans = sorted(unique.values(), key=lambda p: (p.ept_prefix, p.clip_id))

    if cfg.clip_selection_policy == "best_one" and len(clip_plans) > 1:
        # Select newest/best using the source workunit attributes already embedded in the plan.
        clip_plans = clip_plans[:1]

    write_csv(cfg.out_root / "manifests" / "clip_plan.csv", [asdict(p) for p in clip_plans], list(ClipPlan.__dataclass_fields__.keys()))
    write_csv(cfg.out_root / "manifests" / "workunit_status.csv", [asdict(s) for s in workunit_statuses], list(WorkunitStatus.__dataclass_fields__.keys()))
    write_csv(cfg.out_root / "manifests" / "failed_queries.csv", [asdict(f) for f in hard_failures], list(QueryFailure.__dataclass_fields__.keys()))
    print(f"  Planned clips: {len(clip_plans)}")
    print(f"  Hard query/resolve/coverage failures recorded: {len(hard_failures)}")
    print()

    if cfg.dry_run:
        print("DRY RUN ONLY: no PDAL extraction will be executed.")
        print("Set dry_run = False in main() after inspecting manifests/clip_plan.csv.")
        print()

    results: list[ClipResult] = []
    for plan in maybe_progress(clip_plans, total=len(clip_plans), desc="EPT clips"):
        result = extract_one_clip(cfg, fp, plan)
        results.append(result)

    write_csv(cfg.out_root / "manifests" / "clip_results.csv", [asdict(r) for r in results], list(ClipResult.__dataclass_fields__.keys()))

    success_statuses = {"extracted", "already_exists_sidecar_matched", "planned_dry_run"}
    status_counts: dict[str, int] = {}
    total_bytes = 0
    total_points = 0
    available = 0
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        if result.status in success_statuses:
            available += 1
            total_bytes += int(result.bytes_written)
            total_points += int(result.point_count)

    run_success = bool((not hard_failures) and (not cfg.require_at_least_one_available_clip or available > 0) and all(r.status in success_statuses for r in results))
    summary = {
        "script": "download_3dep_lpc.py",
        "script_semantics": "3DEP full_density_ept_clip_for_casals_h5_footprint_not_original_source_tile",
        "scientific_notes": [
            "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
            "refh is WGS84 ellipsoidal height unless otherwise documented.",
            "This is not an official multi-return point cloud.",
            "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            "3DEP clips are EPT-derived clips, not archival source LAZ copies; horizontal or vertical datum differences remain documented rather than transformed here.",
        ],
        "source_h5": str(cfg.h5_path.resolve()),
        "out_root": str(cfg.out_root.resolve()),
        "dry_run": cfg.dry_run,
        "run_success": run_success,
        "casals_h5_footprint": asdict(fp),
        "lpc_workunit_count": len(refs),
        "selected_workunit_count": len(selected_refs),
        "skipped_or_lower_priority_workunit_count": max(0, len(refs) - len(selected_refs)),
        "resolved_ept_count": len(ept_by_source),
        "clip_plan_count": len(clip_plans),
        "hard_failure_count": len(hard_failures),
        "clip_status_counts": status_counts,
        "available_clip_count": available,
        "reported_point_count_sum": total_points,
        "available_clip_bytes": total_bytes,
        "available_clip_gb": total_bytes / (1024 ** 3),
        "output_crs_mode": cfg.output_crs_mode,
        "vertical_datum_transform_applied": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(cfg.out_root / "manifests" / "summary.json", summary)

    print("Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    if not run_success:
        raise RuntimeError("3DEP LPC run did not complete successfully. Check manifests/failed_queries.csv and clip_results.csv.")


def main() -> None:
    cfg = Config(
        h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
        out_root=Path(r"./outputs/download_3dep_lpc"),
        point_cloud_dir=Path(r"./point_cloud_data/download_3dep_lpc"),
        clip_buffer_m=30.0,
        output_format="laz",
        dry_run=False,
        overwrite=False,
        clip_crs_mode="nad83_utm",
        clip_crs_epsg_override=None,
        prefer_3dep_meets_workunits=True,
        try_nonpreferred_if_no_preferred=True,
        clip_selection_policy="all_usable_ept",
        output_crs_mode="lpc_horiz_crs",
        allow_missing_ept_crs=False,
        run_pdal_info_postcheck=True,
        require_at_least_one_available_clip=True,
    )
    run(cfg)


if __name__ == "__main__":
    main()
