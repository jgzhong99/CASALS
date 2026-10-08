"""Shared grid and GeoTIFF helpers used by refh surface workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from pyproj import CRS
from scipy import ndimage

try:
    import rasterio
except Exception as exc:  # pragma: no cover
    rasterio = None
    _RASTERIO_IMPORT_ERROR = exc
else:
    _RASTERIO_IMPORT_ERROR = None


def disk_structure(radius_cells: int) -> np.ndarray:
    radius_cells = int(max(1, radius_cells))
    yy, xx = np.ogrid[-radius_cells: radius_cells + 1, -radius_cells: radius_cells + 1]
    return (xx * xx + yy * yy) <= radius_cells * radius_cells

def fill_nearest_within_mask(
    values: np.ndarray,
    support_mask: np.ndarray,
    resolution: float,
    max_distance_m: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    out = values.astype(np.float32).copy()
    valid = np.isfinite(out) & support_mask
    target = support_mask & ~valid

    distance_cells, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    distance_m = distance_cells.astype(np.float32) * float(resolution)

    if max_distance_m is None:
        fill_mask = target
    else:
        fill_mask = target & (distance_m <= float(max_distance_m))

    out[fill_mask] = out[indices[0][fill_mask], indices[1][fill_mask]]
    out[~support_mask] = np.nan
    return out, distance_m

def write_float_geotiff(path: Path, arr: np.ndarray, crs: CRS, transform: Any, nodata: float) -> None:
    if rasterio is None:
        raise RuntimeError(f"rasterio import failed: {_RASTERIO_IMPORT_ERROR}")
    out = np.asarray(arr, dtype=np.float32)
    out = np.where(np.isfinite(out), out, nodata).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(out, 1)

def write_uint8_geotiff(path: Path, arr: np.ndarray, crs: CRS, transform: Any, nodata: int = 0) -> None:
    if rasterio is None:
        raise RuntimeError(f"rasterio import failed: {_RASTERIO_IMPORT_ERROR}")
    out = np.asarray(arr, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(out, 1)

def robust_normalize(values: np.ndarray, lo_p: float = 2.0, hi_p: float = 98.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.zeros_like(values, dtype=np.float64)
    lo = float(np.nanpercentile(values[finite], lo_p))
    hi = float(np.nanpercentile(values[finite], hi_p))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    out = (values - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)
