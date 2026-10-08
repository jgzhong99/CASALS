"""Horizontal coordinate helpers for CASALS L1B workflows.

These routines handle horizontal CRSs only. They do not perform or imply a
vertical datum or reference-frame transformation for CASALS ``refh`` heights.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pyproj import CRS, Transformer


def infer_wgs84_utm_epsg(lon: np.ndarray, lat: np.ndarray) -> int:
    lon_median = float(np.nanmedian(lon))
    lat_median = float(np.nanmedian(lat))
    if not (-180.0 <= lon_median <= 180.0 and -90.0 <= lat_median <= 90.0):
        raise ValueError(
            f"Invalid median lon/lat for UTM inference: {lon_median}, {lat_median}"
        )
    zone = int(math.floor((lon_median + 180.0) / 6.0) + 1)
    zone = max(1, min(zone, 60))
    return 32600 + zone if lat_median >= 0.0 else 32700 + zone


def transform_lonlat_to_projected(
    lon: np.ndarray,
    lat: np.ndarray,
    epsg: int,
) -> tuple[np.ndarray, np.ndarray, CRS]:
    crs = CRS.from_epsg(int(epsg))
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), crs


def transform_xy(
    x: np.ndarray,
    y: np.ndarray,
    source_crs: Any,
    target_crs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(
        CRS.from_user_input(source_crs), CRS.from_user_input(target_crs), always_xy=True
    )
    out_x, out_y = transformer.transform(x, y)
    return np.asarray(out_x, dtype=np.float64), np.asarray(out_y, dtype=np.float64)


def horizontal_crs_only(crs: CRS) -> CRS:
    if crs.is_compound:
        for sub_crs in crs.sub_crs_list:
            if not sub_crs.is_vertical:
                return CRS.from_user_input(sub_crs)
        raise RuntimeError(f"Could not identify horizontal CRS from compound CRS: {crs}")
    if crs.is_vertical:
        raise RuntimeError(f"Reference CRS is vertical-only and unusable for XY projection: {crs}")
    return crs
