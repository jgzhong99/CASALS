r"""Detect the UTM zone for a CASALS-style HDF5 granule.

Strategy:
1. Scan scalar HDF5 attributes for explicit EPSG / UTM text.
2. If no explicit CRS is found, infer UTM from median lon/lat.

Example:
    python CASALS_L1B/detect_h5_utm_zone.py ^
        --h5 CASALS_L1B\casals_h5_downloads\casals_l1b_20241118T171757_001_02.h5
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np

try:
    from pyproj import CRS
except Exception:
    CRS = None


EPSG_RE = re.compile(r"EPSG[:\s]*([0-9]{4,6})", re.IGNORECASE)
UTM_RE = re.compile(r"UTM(?:\s+ZONE)?[:\s]*([0-9]{1,2})([NS])?", re.IGNORECASE)


@dataclass
class UtmResult:
    source: str
    zone: int
    hemisphere: str
    epsg: int
    crs_name: Optional[str]
    lon_median: Optional[float] = None
    lat_median: Optional[float] = None
    detail: Optional[str] = None


def iter_scalar_attr_strings(h5: h5py.File) -> Iterable[tuple[str, str]]:
    for key, value in h5.attrs.items():
        text = scalar_attr_to_text(value)
        if text is not None:
            yield f"/@{key}", text


def scalar_attr_to_text(value: object) -> Optional[str]:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if np.isscalar(value):
        return str(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return None


def epsg_to_utm(epsg: int) -> Optional[tuple[int, str]]:
    if 32601 <= epsg <= 32660:
        return epsg - 32600, "N"
    if 32701 <= epsg <= 32760:
        return epsg - 32700, "S"
    return None


def detect_from_attrs(h5: h5py.File) -> Optional[UtmResult]:
    for path, text in iter_scalar_attr_strings(h5):
        match = EPSG_RE.search(text)
        if match:
            epsg = int(match.group(1))
            utm = epsg_to_utm(epsg)
            if utm is not None:
                zone, hemisphere = utm
                return UtmResult(
                    source="attribute_epsg",
                    zone=zone,
                    hemisphere=hemisphere,
                    epsg=epsg,
                    crs_name=crs_name_from_epsg(epsg),
                    detail=f"{path}={text}",
                )

        match = UTM_RE.search(text)
        if match:
            zone = int(match.group(1))
            hemisphere = (match.group(2) or "N").upper()
            epsg = (32600 if hemisphere == "N" else 32700) + zone
            return UtmResult(
                source="attribute_utm",
                zone=zone,
                hemisphere=hemisphere,
                epsg=epsg,
                crs_name=crs_name_from_epsg(epsg),
                detail=f"{path}={text}",
            )
    return None


def crs_name_from_epsg(epsg: int) -> Optional[str]:
    if CRS is None:
        return None
    try:
        return CRS.from_epsg(epsg).name
    except Exception:
        return None


def read_first_1d(h5: h5py.File, candidates: list[str]) -> tuple[np.ndarray, str]:
    for name in candidates:
        if name in h5:
            arr = np.asarray(h5[name][...], dtype=np.float64)
            if arr.ndim == 1:
                return arr, name
    raise KeyError(f"Could not find a 1D dataset in candidates: {candidates}")


def infer_from_lonlat(h5: h5py.File) -> UtmResult:
    lon, lon_name = read_first_1d(h5, [
        "refh_longitude",
        "rwstart_longitude",
        "rwstop_longitude",
        "instrument_longitude",
    ])
    lat, lat_name = read_first_1d(h5, [
        "refh_latitude",
        "rwstart_latitude",
        "rwstop_latitude",
        "instrument_latitude",
    ])

    mask = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(mask):
        raise ValueError(f"No finite lon/lat values found in {lon_name} and {lat_name}.")

    lon_med = float(np.nanmedian(lon[mask]))
    lat_med = float(np.nanmedian(lat[mask]))
    if not (-180.0 <= lon_med <= 180.0 and -90.0 <= lat_med <= 90.0):
        raise ValueError(f"Invalid median lon/lat: {lon_med}, {lat_med}")

    zone = int(math.floor((lon_med + 180.0) / 6.0) + 1)
    zone = max(1, min(zone, 60))
    hemisphere = "N" if lat_med >= 0.0 else "S"
    epsg = (32600 if hemisphere == "N" else 32700) + zone
    return UtmResult(
        source="median_lonlat",
        zone=zone,
        hemisphere=hemisphere,
        epsg=epsg,
        crs_name=crs_name_from_epsg(epsg),
        lon_median=lon_med,
        lat_median=lat_med,
        detail=f"lon={lon_name}, lat={lat_name}",
    )


def detect_utm_zone(h5_path: Path) -> UtmResult:
    with h5py.File(h5_path, "r") as h5:
        from_attrs = detect_from_attrs(h5)
        if from_attrs is not None:
            return from_attrs
        return infer_from_lonlat(h5)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect UTM zone for a CASALS HDF5 file.")
    parser.add_argument("--h5", required=True, type=Path, help="Path to the input .h5 file.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = detect_utm_zone(args.h5)

    print(f"H5: {args.h5}")
    print(f"Detection source: {result.source}")
    print(f"UTM zone: {result.zone}{result.hemisphere}")
    print(f"EPSG: {result.epsg}")
    if result.crs_name:
        print(f"CRS name: {result.crs_name}")
    if result.lon_median is not None and result.lat_median is not None:
        print(f"Median lon/lat: {result.lon_median:.12f}, {result.lat_median:.12f}")
    if result.detail:
        print(f"Detail: {result.detail}")


if __name__ == "__main__":
    main()
