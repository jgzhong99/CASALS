"""Create a support-limited CASALS L1B refh-based surface DSM from selected points.

Scientific meaning:
    Each selected point is one CASALS L1B geolocated max-Rx-bin / refh
    reference-return point.

Outputs:
    A filled refh-based surface DSM, a raw strict companion DSM, and optional
    selected LAS plus summary rasters.

This script does not:
    - create a ground-classified DEM,
    - infer multi-return structure,
    - extend the surface beyond a conservative support mask.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
from pyproj import CRS, Transformer

try:
    import rasterio
    from rasterio.transform import from_origin
except Exception as exc:  # pragma: no cover
    rasterio = None
    from_origin = None
    _RASTERIO_IMPORT_ERROR = exc
else:
    _RASTERIO_IMPORT_ERROR = None

try:
    import laspy
except Exception as exc:  # pragma: no cover
    laspy = None
    _LASPY_IMPORT_ERROR = exc
else:
    _LASPY_IMPORT_ERROR = None

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass
class Config:
    h5_path: Path
    point_cloud_dir: Path
    out_dir: Path

    # Main quality threshold. In the tested CASALS file, refh_snr = refh_amp/refh_thres,
    # and good_snr is equivalent to refh_snr >= 5.0. Lower thresholds keep more surface.
    snr_threshold: float = 3.0

    # DSM grid spacing in output horizontal CRS units, usually meters.
    dsm_resolution_m: float = 1.0

    # Horizontal output CRS. If None, infer WGS84 UTM EPSG from median lon/lat.
    output_epsg_override: Optional[int] = None

    # Add a small XY buffer to the raster extent so boundary points are not clipped.
    extent_buffer_m: float = 0.0

    # Optional additional filters. Leave None to disable.
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    amp_min: Optional[float] = None
    amp_min_percentile: Optional[float] = None

    # Optional robust vertical clipping based on selected points. Useful for suppressing
    # extreme refh spikes before DSM max aggregation. Leave None for strict SNR-only DSM.
    selected_z_percentile_clip: Optional[Tuple[float, float]] = None

    # Cell policy for raw DSM. DSM normally uses max Z per cell. If min_points_per_cell > 1,
    # cells with fewer selected points become NODATA, making the raw DSM stricter.
    aggregation: str = "max"  # currently: "max", "mean", "median", "p95"
    min_points_per_cell: int = 1

    # Conservative support mask and fill controls.
    write_raw_strict_dsm: bool = True
    write_support_mask_raster: bool = True
    write_fill_source_raster: bool = True
    support_buffer_m: float = 0.0
    support_closing_m: Optional[float] = None
    support_fill_holes: bool = True
    fill_internal_holes: bool = True
    fill_method: str = "idw"
    idw_radius_m: Optional[float] = None
    idw_k: int = 12
    idw_power: float = 2.0
    idw_min_neighbors: int = 3
    max_internal_fill_distance_m: Optional[float] = None
    idw_chunk_size: int = 250_000

    # Output toggles.
    write_selected_las: bool = True
    write_count_raster: bool = True
    write_snr_max_raster: bool = True
    write_snr_mean_raster: bool = False
    write_preview_png: bool = True

    # LAS writing settings.
    las_scale_m: float = 0.001

    # NODATA value for float rasters.
    nodata_float: float = -9999.0

    # Reproducibility.
    random_seed: int = 42


@dataclass
class PointData:
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray
    snr: np.ndarray
    amp: np.ndarray
    thres: np.ndarray
    good_snr: np.ndarray
    track_num: Optional[np.ndarray]
    sweep_num: Optional[np.ndarray]
    pulse_index: np.ndarray
    attrs: Dict[str, Any]


@dataclass
class ResolvedFillParams:
    support_buffer_m: float
    support_closing_m: float
    support_fill_holes: bool
    fill_internal_holes: bool
    fill_method: str
    idw_radius_m: float
    idw_k: int
    idw_power: float
    idw_min_neighbors: int
    max_internal_fill_distance_m: float
    idw_chunk_size: int


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def find_dataset(h5: h5py.File, name: str) -> Optional[h5py.Dataset]:
    """Return dataset by exact root name or by recursive basename match."""
    if name in h5 and isinstance(h5[name], h5py.Dataset):
        return h5[name]

    matches = []

    def visitor(path: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and path.split("/")[-1] == name:
            matches.append(path)

    h5.visititems(visitor)
    if len(matches) == 1:
        return h5[matches[0]]
    if len(matches) > 1:
        raise ValueError(f"Multiple H5 datasets named {name!r}: {matches}")
    return None


def require_dataset(h5: h5py.File, name: str) -> h5py.Dataset:
    ds = find_dataset(h5, name)
    if ds is None:
        raise KeyError(f"Required dataset {name!r} was not found in H5 file.")
    return ds


def read_optional_dataset(h5: h5py.File, name: str, n_expected: int) -> Optional[np.ndarray]:
    ds = find_dataset(h5, name)
    if ds is None:
        return None
    arr = ds[...]
    arr = np.asarray(arr).reshape(-1)
    if arr.size != n_expected:
        warnings.warn(f"Optional dataset {name!r} has size {arr.size}, expected {n_expected}; ignored.")
        return None
    return arr


def read_point_data(h5_path: Path) -> PointData:
    with h5py.File(h5_path, "r") as h5:
        lon = np.asarray(require_dataset(h5, "refh_longitude")[...], dtype=np.float64).reshape(-1)
        lat = np.asarray(require_dataset(h5, "refh_latitude")[...], dtype=np.float64).reshape(-1)
        z = np.asarray(require_dataset(h5, "refh")[...], dtype=np.float64).reshape(-1)
        amp = np.asarray(require_dataset(h5, "refh_amp")[...], dtype=np.float64).reshape(-1)

        snr_ds = find_dataset(h5, "refh_snr")
        thres_ds = find_dataset(h5, "refh_thres")
        if thres_ds is not None:
            thres = np.asarray(thres_ds[...], dtype=np.float64).reshape(-1)
        else:
            thres = np.full(lon.shape, np.nan, dtype=np.float64)

        if snr_ds is not None:
            snr = np.asarray(snr_ds[...], dtype=np.float64).reshape(-1)
        else:
            if thres_ds is None:
                raise KeyError("Neither refh_snr nor refh_thres was found; cannot compute SNR.")
            snr = np.divide(amp, thres, out=np.full_like(amp, np.nan), where=(thres != 0))

        good_snr_arr = read_optional_dataset(h5, "good_snr", lon.size)
        if good_snr_arr is None:
            good_snr = snr >= 5.0
        else:
            good_snr = good_snr_arr.astype(bool)

        track_num = read_optional_dataset(h5, "track_num", lon.size)
        sweep_num = read_optional_dataset(h5, "sweep_num", lon.size)

        sizes = {"lon": lon.size, "lat": lat.size, "z": z.size, "snr": snr.size, "amp": amp.size, "thres": thres.size}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"Required datasets do not have matching sizes: {sizes}")

        attrs = {}
        for key, value in h5.attrs.items():
            try:
                if isinstance(value, bytes):
                    attrs[key] = value.decode("utf-8", errors="replace")
                elif isinstance(value, np.ndarray):
                    attrs[key] = value.tolist()
                else:
                    attrs[key] = value.item() if hasattr(value, "item") else value
            except Exception:
                attrs[key] = str(value)

    return PointData(
        lon=lon,
        lat=lat,
        z=z,
        snr=snr,
        amp=amp,
        thres=thres,
        good_snr=good_snr,
        track_num=track_num,
        sweep_num=sweep_num,
        pulse_index=np.arange(lon.size, dtype=np.uint32),
        attrs=attrs,
    )


def infer_wgs84_utm_epsg(lon: np.ndarray, lat: np.ndarray) -> int:
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    zone = int(math.floor((lon0 + 180.0) / 6.0) + 1)
    if not (1 <= zone <= 60):
        raise ValueError(f"Cannot infer UTM zone from median longitude {lon0}")
    return (32600 if lat0 >= 0 else 32700) + zone


def transform_lonlat_to_projected(lon: np.ndarray, lat: np.ndarray, epsg: int) -> Tuple[np.ndarray, np.ndarray, CRS]:
    out_crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(CRS.from_epsg(4326), out_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), out_crs


def build_valid_mask(pd: PointData, cfg: Config) -> Tuple[np.ndarray, Dict[str, Any]]:
    base = (
        np.isfinite(pd.lon)
        & np.isfinite(pd.lat)
        & np.isfinite(pd.z)
        & np.isfinite(pd.snr)
        & np.isfinite(pd.amp)
        & (pd.lon >= -180.0)
        & (pd.lon <= 180.0)
        & (pd.lat >= -90.0)
        & (pd.lat <= 90.0)
    )

    mask = base.copy()
    reasons = {"n_total": int(pd.lon.size), "n_base_valid": int(base.sum())}

    if cfg.z_min is not None:
        mask &= pd.z >= cfg.z_min
    if cfg.z_max is not None:
        mask &= pd.z <= cfg.z_max
    reasons["n_after_optional_z_limits"] = int(mask.sum())

    if cfg.amp_min is not None:
        mask &= pd.amp >= cfg.amp_min
    if cfg.amp_min_percentile is not None:
        amp_ref = pd.amp[mask & np.isfinite(pd.amp)]
        if amp_ref.size == 0:
            raise ValueError("No valid amplitude values available for amp_min_percentile.")
        amp_thr = float(np.nanpercentile(amp_ref, cfg.amp_min_percentile))
        mask &= pd.amp >= amp_thr
        reasons["amp_min_percentile"] = cfg.amp_min_percentile
        reasons["amp_min_from_percentile"] = amp_thr
    reasons["n_after_amp_filters"] = int(mask.sum())

    mask &= pd.snr >= cfg.snr_threshold
    reasons["snr_threshold"] = cfg.snr_threshold
    reasons["n_after_snr_threshold"] = int(mask.sum())

    if cfg.selected_z_percentile_clip is not None:
        lo_p, hi_p = cfg.selected_z_percentile_clip
        z_ref = pd.z[mask & np.isfinite(pd.z)]
        if z_ref.size == 0:
            raise ValueError("No selected points available for selected_z_percentile_clip.")
        z_lo = float(np.nanpercentile(z_ref, lo_p))
        z_hi = float(np.nanpercentile(z_ref, hi_p))
        mask &= (pd.z >= z_lo) & (pd.z <= z_hi)
        reasons["selected_z_percentile_clip"] = [lo_p, hi_p]
        reasons["selected_z_clip_values"] = [z_lo, z_hi]
    reasons["n_selected_surface_points"] = int(mask.sum())

    if not np.any(mask):
        raise ValueError("No points survived filtering. Lower snr_threshold or relax other filters.")
    return mask, reasons


def resolve_fill_params(cfg: Config) -> ResolvedFillParams:
    support_closing_m = float(cfg.support_closing_m) if cfg.support_closing_m is not None else 2.0 * float(cfg.dsm_resolution_m)
    idw_radius_m = float(cfg.idw_radius_m) if cfg.idw_radius_m is not None else max(3.0 * float(cfg.dsm_resolution_m), support_closing_m + float(cfg.dsm_resolution_m))
    max_internal_fill_distance_m = (
        float(cfg.max_internal_fill_distance_m)
        if cfg.max_internal_fill_distance_m is not None
        else 4.0 * float(cfg.dsm_resolution_m)
    )
    return ResolvedFillParams(
        support_buffer_m=float(cfg.support_buffer_m),
        support_closing_m=support_closing_m,
        support_fill_holes=bool(cfg.support_fill_holes),
        fill_internal_holes=bool(cfg.fill_internal_holes),
        fill_method=str(cfg.fill_method).lower(),
        idw_radius_m=idw_radius_m,
        idw_k=max(1, int(cfg.idw_k)),
        idw_power=float(cfg.idw_power),
        idw_min_neighbors=max(1, int(cfg.idw_min_neighbors)),
        max_internal_fill_distance_m=max_internal_fill_distance_m,
        idw_chunk_size=max(10_000, int(cfg.idw_chunk_size)),
    )


def compute_grid_geometry(x: np.ndarray, y: np.ndarray, resolution: float, buffer_m: float) -> Dict[str, Any]:
    xmin = float(np.nanmin(x) - buffer_m)
    xmax = float(np.nanmax(x) + buffer_m)
    ymin = float(np.nanmin(y) - buffer_m)
    ymax = float(np.nanmax(y) + buffer_m)

    width = int(math.ceil((xmax - xmin) / resolution))
    height = int(math.ceil((ymax - ymin) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid grid shape width={width}, height={height}")

    xmax_snapped = xmin + width * resolution
    ymin_snapped = ymax - height * resolution
    transform = from_origin(xmin, ymax, resolution, resolution)

    return {
        "xmin": xmin,
        "xmax": xmax_snapped,
        "ymin": ymin_snapped,
        "ymax": ymax,
        "width": width,
        "height": height,
        "resolution": resolution,
        "transform": transform,
    }


def xy_to_row_col(x: np.ndarray, y: np.ndarray, grid: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    col = np.floor((x - float(grid["xmin"])) / float(grid["resolution"])).astype(np.int64)
    row = np.floor((float(grid["ymax"]) - y) / float(grid["resolution"])).astype(np.int64)
    inside = (row >= 0) & (row < int(grid["height"])) & (col >= 0) & (col < int(grid["width"]))
    return row, col, inside


def aggregate_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    snr: np.ndarray,
    grid: Dict[str, Any],
    cfg: Config,
) -> Dict[str, np.ndarray]:
    width = int(grid["width"])
    height = int(grid["height"])

    row, col, inside = xy_to_row_col(x, y, grid)
    if not np.all(inside):
        row = row[inside]
        col = col[inside]
        z = z[inside]
        snr = snr[inside]

    flat = row * width + col
    n_cells = height * width

    count = np.bincount(flat, minlength=n_cells).astype(np.uint32).reshape(height, width)

    if cfg.aggregation.lower() == "max":
        dsm_flat = np.full(n_cells, -np.inf, dtype=np.float64)
        np.maximum.at(dsm_flat, flat, z)
        dsm = dsm_flat.reshape(height, width)
    elif cfg.aggregation.lower() == "mean":
        z_sum = np.bincount(flat, weights=z, minlength=n_cells).reshape(height, width)
        dsm = np.divide(z_sum, count, out=np.full((height, width), np.nan), where=(count > 0))
    elif cfg.aggregation.lower() in {"median", "p95"}:
        dsm_flat = np.full(n_cells, np.nan, dtype=np.float64)
        order = np.argsort(flat)
        flat_sorted = flat[order]
        z_sorted = z[order]
        starts = np.r_[0, np.flatnonzero(np.diff(flat_sorted)) + 1]
        ends = np.r_[starts[1:], flat_sorted.size]
        q = 50 if cfg.aggregation.lower() == "median" else 95
        for s, e in zip(starts, ends):
            dsm_flat[flat_sorted[s]] = np.nanpercentile(z_sorted[s:e], q)
        dsm = dsm_flat.reshape(height, width)
    else:
        raise ValueError(f"Unsupported aggregation={cfg.aggregation!r}; use max, mean, median, or p95.")

    snr_max_flat = np.full(n_cells, -np.inf, dtype=np.float64)
    np.maximum.at(snr_max_flat, flat, snr)
    snr_max = snr_max_flat.reshape(height, width)

    snr_sum = np.bincount(flat, weights=snr, minlength=n_cells).reshape(height, width)
    snr_mean = np.divide(snr_sum, count, out=np.full((height, width), np.nan), where=(count > 0))

    valid_cell = count >= int(cfg.min_points_per_cell)
    dsm = dsm.astype(np.float64)
    dsm[~valid_cell] = np.nan
    snr_max[~valid_cell] = np.nan
    snr_mean[~valid_cell] = np.nan

    dsm[~np.isfinite(dsm)] = np.nan
    snr_max[~np.isfinite(snr_max)] = np.nan

    return {
        "dsm": dsm.astype(np.float32),
        "count": count,
        "snr_max": snr_max.astype(np.float32),
        "snr_mean": snr_mean.astype(np.float32),
    }


def disk_structure(radius_cells: int) -> np.ndarray:
    radius_cells = int(max(1, radius_cells))
    yy, xx = np.ogrid[-radius_cells: radius_cells + 1, -radius_cells: radius_cells + 1]
    return (xx * xx + yy * yy) <= radius_cells * radius_cells


def make_support_mask(observed_mask: np.ndarray, grid: Dict[str, Any], fill_cfg: ResolvedFillParams) -> np.ndarray:
    mask = observed_mask.astype(bool).copy()
    res = float(grid["resolution"])

    buffer_cells = int(math.ceil(fill_cfg.support_buffer_m / res))
    if buffer_cells > 0:
        mask = ndimage.binary_dilation(mask, structure=disk_structure(buffer_cells))

    closing_cells = int(math.ceil(fill_cfg.support_closing_m / res))
    if closing_cells > 0:
        mask = ndimage.binary_closing(mask, structure=disk_structure(closing_cells))

    if fill_cfg.support_fill_holes:
        mask = ndimage.binary_fill_holes(mask)

    # Never let morphology drop observed strict cells; support is conservative but inclusive.
    mask |= observed_mask.astype(bool)
    return mask.astype(bool)


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


def idw_fill_dsm(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    raw_dsm: np.ndarray,
    support_mask: np.ndarray,
    grid: Dict[str, Any],
    fill_cfg: ResolvedFillParams,
) -> Dict[str, np.ndarray]:
    fill_method = fill_cfg.fill_method
    if fill_method not in {"idw"}:
        raise ValueError(f"Unsupported fill_method={fill_method!r}; use 'idw'.")

    out = raw_dsm.astype(np.float32).copy()
    source = np.zeros(raw_dsm.shape, dtype=np.uint8)
    source[np.isfinite(raw_dsm) & support_mask] = 1
    nearest_distance = np.full(raw_dsm.shape, np.nan, dtype=np.float32)

    tree = cKDTree(np.column_stack((x, y)))
    rows, cols = np.nonzero(support_mask & ~np.isfinite(raw_dsm))
    if rows.size > 0:
        qx = float(grid["xmin"]) + (cols.astype(np.float64) + 0.5) * float(grid["resolution"])
        qy = float(grid["ymax"]) - (rows.astype(np.float64) + 0.5) * float(grid["resolution"])
        queries = np.column_stack((qx, qy))
        eps = 1e-6
        k = fill_cfg.idw_k

        for start in range(0, queries.shape[0], fill_cfg.idw_chunk_size):
            end = min(start + fill_cfg.idw_chunk_size, queries.shape[0])
            q = queries[start:end]
            d, idx = tree.query(
                q,
                k=k,
                distance_upper_bound=fill_cfg.idw_radius_m,
                workers=-1,
            )
            if k == 1:
                d = d[:, None]
                idx = idx[:, None]

            valid = np.isfinite(d) & (idx >= 0) & (idx < x.size)
            n_valid = valid.sum(axis=1)
            nearest = np.min(np.where(valid, d, np.inf), axis=1)

            vals = np.full(q.shape[0], np.nan, dtype=np.float64)
            ok = n_valid >= fill_cfg.idw_min_neighbors
            if np.any(ok):
                vv = valid[ok]
                dd = d[ok]
                ii = idx[ok]
                ii_safe = np.where(vv, ii, 0)
                weights = np.where(vv, 1.0 / np.maximum(dd, eps) ** fill_cfg.idw_power, 0.0)
                z_nei = np.where(vv, z[ii_safe], 0.0)
                denom = np.sum(weights, axis=1)
                vals[ok] = np.divide(
                    np.sum(weights * z_nei, axis=1),
                    denom,
                    out=np.full(np.sum(ok), np.nan, dtype=np.float64),
                    where=denom > 0,
                )

            rr = rows[start:end]
            cc = cols[start:end]
            ok_rows = rr[ok]
            ok_cols = cc[ok]
            out[ok_rows, ok_cols] = vals[ok].astype(np.float32)
            source[ok_rows, ok_cols] = 2
            nearest_distance[rr, cc] = np.where(np.isfinite(nearest), nearest, np.nan).astype(np.float32)

    if fill_cfg.fill_internal_holes:
        filled, fill_dist = fill_nearest_within_mask(
            out,
            support_mask,
            float(grid["resolution"]),
            fill_cfg.max_internal_fill_distance_m,
        )
        fill_mask = support_mask & ~np.isfinite(out) & np.isfinite(filled)
        out = filled
        source[fill_mask] = 3
        nearest_distance[fill_mask] = fill_dist[fill_mask]

    out[~support_mask] = np.nan
    nearest_distance[~support_mask] = np.nan
    return {
        "filled_dsm": out.astype(np.float32),
        "fill_source": source,
        "nearest_distance_m": nearest_distance.astype(np.float32),
    }


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


def write_uint32_geotiff(path: Path, arr: np.ndarray, crs: CRS, transform: Any) -> None:
    if rasterio is None:
        raise RuntimeError(f"rasterio import failed: {_RASTERIO_IMPORT_ERROR}")
    out = np.asarray(arr, dtype=np.uint32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype="uint32",
        crs=crs,
        transform=transform,
        nodata=0,
        compress="deflate",
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


def write_selected_las(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pd: PointData,
    mask: np.ndarray,
    out_crs: CRS,
    cfg: Config,
) -> None:
    if laspy is None:
        raise RuntimeError(f"laspy import failed: {_LASPY_IMPORT_ERROR}")

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([cfg.las_scale_m, cfg.las_scale_m, cfg.las_scale_m], dtype=np.float64)
    header.offsets = np.array([
        math.floor(float(np.nanmin(x))),
        math.floor(float(np.nanmin(y))),
        math.floor(float(np.nanmin(z))),
    ], dtype=np.float64)
    header.add_crs(out_crs)

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.intensity = np.clip(np.rint(pd.amp[mask]), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    las.classification = np.ones(x.size, dtype=np.uint8)

    t = robust_normalize(pd.snr[mask], lo_p=2, hi_p=98)
    r = np.clip(2.0 * t - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * t - 1.0), 0.0, 1.0)
    b = np.clip(1.5 - 2.0 * t, 0.0, 1.0)
    las.red = np.rint(r * 65535).astype(np.uint16)
    las.green = np.rint(g * 65535).astype(np.uint16)
    las.blue = np.rint(b * 65535).astype(np.uint16)

    extra_dims = [
        laspy.ExtraBytesParams(name="longitude", type=np.float64, description="refh_lon_deg"),
        laspy.ExtraBytesParams(name="latitude", type=np.float64, description="refh_lat_deg"),
        laspy.ExtraBytesParams(name="refh_snr", type=np.float64, description="refh_snr"),
        laspy.ExtraBytesParams(name="refh_amp", type=np.float64, description="refh_amp"),
        laspy.ExtraBytesParams(name="refh_thres", type=np.float64, description="refh_thres"),
        laspy.ExtraBytesParams(name="good_snr", type=np.uint8, description="good_snr_flag"),
        laspy.ExtraBytesParams(name="pulse_index", type=np.uint32, description="pulse_index"),
    ]
    if pd.track_num is not None:
        extra_dims.append(laspy.ExtraBytesParams(name="track_num", type=np.uint16, description="track_num"))
    if pd.sweep_num is not None:
        extra_dims.append(laspy.ExtraBytesParams(name="sweep_num", type=np.uint32, description="sweep_num"))
    las.add_extra_dims(extra_dims)

    las["longitude"] = pd.lon[mask]
    las["latitude"] = pd.lat[mask]
    las["refh_snr"] = pd.snr[mask]
    las["refh_amp"] = pd.amp[mask]
    las["refh_thres"] = pd.thres[mask]
    las["good_snr"] = pd.good_snr[mask].astype(np.uint8)
    las["pulse_index"] = pd.pulse_index[mask]
    if pd.track_num is not None:
        las["track_num"] = pd.track_num[mask].astype(np.uint16)
    if pd.sweep_num is not None:
        las["sweep_num"] = pd.sweep_num[mask].astype(np.uint32)

    las.write(path)


def write_previews(out_dir: Path, x: np.ndarray, y: np.ndarray, snr: np.ndarray, grids: Dict[str, np.ndarray], cfg: Config) -> None:
    if plt is None:
        warnings.warn(f"matplotlib import failed; previews not written: {_MATPLOTLIB_IMPORT_ERROR}")
        return

    rng = np.random.default_rng(cfg.random_seed)
    max_pts = min(250_000, x.size)
    idx = rng.choice(x.size, size=max_pts, replace=False) if x.size > max_pts else np.arange(x.size)

    fig, ax = plt.subplots(figsize=(7, 8))
    sc = ax.scatter(
        x[idx],
        y[idx],
        c=snr[idx],
        s=0.4,
        cmap="viridis",
        vmin=cfg.snr_threshold,
        vmax=max(6.0, np.nanpercentile(snr, 98)),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Projected X (m)")
    ax.set_ylabel("Projected Y (m)")
    ax.set_title(f"Selected CASALS refh surface points, SNR >= {cfg.snr_threshold}")
    fig.colorbar(sc, ax=ax, label="refh_snr")
    fig.tight_layout()
    fig.savefig(out_dir / "selected_surface_points_snr_xy.png", dpi=220)
    plt.close(fig)

    filled_dsm = grids["filled_dsm"]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(np.where(np.isfinite(filled_dsm), filled_dsm, np.nan), cmap="terrain")
    ax.set_title(f"Filled DSM ({cfg.aggregation}, {cfg.dsm_resolution_m:g} m, SNR >= {cfg.snr_threshold})")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, label="refh elevation, WGS84 ellipsoidal height (m)")
    fig.tight_layout()
    fig.savefig(out_dir / "filled_dsm_preview.png", dpi=220)
    plt.close(fig)

    strict_dsm = grids["strict_dsm"]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(np.where(np.isfinite(strict_dsm), strict_dsm, np.nan), cmap="terrain")
    ax.set_title(f"Raw strict DSM ({cfg.aggregation}, {cfg.dsm_resolution_m:g} m, SNR >= {cfg.snr_threshold})")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, label="refh elevation, WGS84 ellipsoidal height (m)")
    fig.tight_layout()
    fig.savefig(out_dir / "strict_dsm_preview.png", dpi=220)
    plt.close(fig)

    count = grids["count"]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(np.where(count > 0, count, np.nan), cmap="magma")
    ax.set_title("Selected point count per DSM cell")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, label="point count")
    fig.tight_layout()
    fig.savefig(out_dir / "selected_point_count_grid.png", dpi=220)
    plt.close(fig)

    support = grids["support_mask"].astype(np.uint8)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(support, cmap="gray", vmin=0, vmax=1)
    ax.set_title("DSM support mask")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, label="0 outside, 1 inside")
    fig.tight_layout()
    fig.savefig(out_dir / "support_mask_preview.png", dpi=220)
    plt.close(fig)

    fill_source = grids["fill_source"]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(fill_source, cmap="viridis", vmin=0, vmax=3)
    ax.set_title("DSM fill source")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], label="0 outside, 1 observed, 2 IDW, 3 nearest")
    fig.tight_layout()
    fig.savefig(out_dir / "fill_source_preview.png", dpi=220)
    plt.close(fig)


def summarize_array(values: np.ndarray, percentiles: Iterable[float] = (0, 1, 2, 5, 50, 95, 98, 99, 100)) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": int(values.size), "n_finite": 0}
    return {
        "n": int(values.size),
        "n_finite": int(finite.size),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "mean": float(np.nanmean(finite)),
        "std": float(np.nanstd(finite)),
        "percentiles": {str(p): float(np.nanpercentile(finite, p)) for p in percentiles},
    }


def main() -> None:
    # -------------------------------------------------------------------------
    # USER SETTINGS: edit here.
    # -------------------------------------------------------------------------
    cfg = Config(
        h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
        point_cloud_dir=Path(r"./point_cloud_data/make_refh_dsm"),
        out_dir=Path(r"./outputs/make_refh_dsm"),

        # Main threshold. Try 2.0, 3.0, 4.0, 4.5, 5.0.
        # In the tested file, 5.0 is exactly good_snr=True and keeps only ~1.19%.
        snr_threshold=5.0,

        # DSM grid size in meters.
        dsm_resolution_m=5.0,

        # Usually leave None; this infers WGS84 / UTM from H5 lon/lat. For this file it should be EPSG:32618.
        output_epsg_override=None,

        # Optional stricter filters. Keep None unless you are debugging obvious artifacts.
        z_min=None,
        z_max=None,
        amp_min=None,
        amp_min_percentile=None,
        selected_z_percentile_clip=None,

        # DSM policy: raw strict refh DSM uses max by default.
        aggregation="max",
        min_points_per_cell=1,

        # Conservative support-limited fill defaults.
        write_raw_strict_dsm=True,
        write_support_mask_raster=True,
        write_fill_source_raster=True,
        support_buffer_m=0.0,
        support_closing_m=None,
        support_fill_holes=True,
        fill_internal_holes=True,
        fill_method="idw",
        idw_radius_m=None,
        idw_k=12,
        idw_power=2.0,
        idw_min_neighbors=3,
        max_internal_fill_distance_m=None,

        write_selected_las=True,
        write_count_raster=True,
        write_snr_max_raster=True,
        write_snr_mean_raster=False,
        write_preview_png=True,
    )
    # -------------------------------------------------------------------------

    cfg.point_cloud_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    fill_cfg = resolve_fill_params(cfg)

    print("=" * 88)
    print("CASALS refh SNR-selected surface DSM")
    print("=" * 88)
    print(f"H5: {cfg.h5_path.resolve()}")
    print(f"Point-cloud directory: {cfg.point_cloud_dir.resolve()}")
    print(f"Output: {cfg.out_dir.resolve()}")
    print(f"SNR threshold: {cfg.snr_threshold}")
    print(f"DSM resolution: {cfg.dsm_resolution_m} m")
    print("Scientific caveat: output DSM is max-Rx-bin/refh surface DSM, not ground DEM.")

    pd = read_point_data(cfg.h5_path)
    mask, mask_info = build_valid_mask(pd, cfg)

    epsg = cfg.output_epsg_override or infer_wgs84_utm_epsg(pd.lon[mask], pd.lat[mask])
    x_all, y_all, out_crs = transform_lonlat_to_projected(pd.lon, pd.lat, epsg)

    x = x_all[mask]
    y = y_all[mask]
    z = pd.z[mask]
    snr = pd.snr[mask]

    grid = compute_grid_geometry(x, y, cfg.dsm_resolution_m, cfg.extent_buffer_m)
    raw_grids = aggregate_to_grid(x, y, z, snr, grid, cfg)
    strict_dsm = raw_grids["dsm"]
    observed_mask = np.isfinite(strict_dsm)
    support_mask = make_support_mask(observed_mask, grid, fill_cfg)
    fill_results = idw_fill_dsm(x, y, z, strict_dsm, support_mask, grid, fill_cfg)

    grids = {
        "strict_dsm": strict_dsm,
        "filled_dsm": fill_results["filled_dsm"],
        "count": raw_grids["count"],
        "snr_max": raw_grids["snr_max"],
        "snr_mean": raw_grids["snr_mean"],
        "support_mask": support_mask,
        "fill_source": fill_results["fill_source"],
        "nearest_distance_m": fill_results["nearest_distance_m"],
    }

    stem = cfg.h5_path.stem
    thr_tag = f"snr{cfg.snr_threshold:g}".replace(".", "p")
    res_tag = f"{cfg.dsm_resolution_m:g}m".replace(".", "p")

    # Keep the canonical DSM filename for the filled product to minimize downstream breakage.
    filled_dsm_path = cfg.out_dir / f"{stem}_refh_surface_dsm_{thr_tag}_{res_tag}_epsg{epsg}.tif"
    strict_dsm_path = cfg.out_dir / f"{stem}_refh_surface_dsm_strict_{thr_tag}_{res_tag}_epsg{epsg}.tif"
    support_path = cfg.out_dir / f"{stem}_refh_surface_support_mask_{thr_tag}_{res_tag}_epsg{epsg}.tif"
    fill_source_path = cfg.out_dir / f"{stem}_refh_surface_fill_source_{thr_tag}_{res_tag}_epsg{epsg}.tif"
    count_path = cfg.out_dir / f"{stem}_refh_surface_count_{thr_tag}_{res_tag}_epsg{epsg}.tif"
    snr_max_path = cfg.out_dir / f"{stem}_refh_surface_snr_max_{thr_tag}_{res_tag}_epsg{epsg}.tif"
    snr_mean_path = cfg.out_dir / f"{stem}_refh_surface_snr_mean_{thr_tag}_{res_tag}_epsg{epsg}.tif"

    write_float_geotiff(filled_dsm_path, grids["filled_dsm"], out_crs, grid["transform"], cfg.nodata_float)
    if cfg.write_raw_strict_dsm:
        write_float_geotiff(strict_dsm_path, grids["strict_dsm"], out_crs, grid["transform"], cfg.nodata_float)
    if cfg.write_support_mask_raster:
        write_uint8_geotiff(support_path, grids["support_mask"].astype(np.uint8), out_crs, grid["transform"], nodata=0)
    if cfg.write_fill_source_raster:
        write_uint8_geotiff(fill_source_path, grids["fill_source"], out_crs, grid["transform"], nodata=0)
    if cfg.write_count_raster:
        write_uint32_geotiff(count_path, grids["count"], out_crs, grid["transform"])
    if cfg.write_snr_max_raster:
        write_float_geotiff(snr_max_path, grids["snr_max"], out_crs, grid["transform"], cfg.nodata_float)
    if cfg.write_snr_mean_raster:
        write_float_geotiff(snr_mean_path, grids["snr_mean"], out_crs, grid["transform"], cfg.nodata_float)

    las_path = None
    if cfg.write_selected_las:
        las_path = cfg.point_cloud_dir / f"{stem}_selected_refh_surface_{thr_tag}_epsg{epsg}.las"
        write_selected_las(las_path, x, y, z, pd, mask, out_crs, cfg)

    if cfg.write_preview_png:
        write_previews(cfg.out_dir, x, y, snr, grids, cfg)

    strict_valid = np.isfinite(grids["strict_dsm"])
    filled_valid = np.isfinite(grids["filled_dsm"])
    idw_fill_mask = grids["fill_source"] == 2
    nearest_fill_mask = grids["fill_source"] == 3
    support_valid = grids["support_mask"]
    outside_support = ~support_valid
    observed_abs_diff = np.abs(grids["filled_dsm"][strict_valid] - grids["strict_dsm"][strict_valid])
    observed_cells_preserved = bool(np.all(np.isfinite(grids["filled_dsm"][strict_valid])))
    strict_cells_match = bool(observed_cells_preserved and np.nanmax(observed_abs_diff, initial=0.0) <= 1e-6)

    metadata = {
        "script": "make_refh_dsm.py",
        "script_semantics": "support_limited_CASALS_L1B_refh_surface_DSM_with_raw_strict_companion",
        "scientific_notes": [
            "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
            "refh is WGS84 ellipsoidal height unless otherwise documented.",
            "This is not an official multi-return point cloud.",
            "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            "The filled DSM is interpolated only inside a conservative support mask derived from observed DSM cells.",
            "The raw strict DSM is retained as an audit product.",
        ],
        "config": asdict(cfg),
        "resolved_fill_parameters": asdict(fill_cfg),
        "source_h5": str(cfg.h5_path),
        "source_attrs_subset": pd.attrs,
        "output_crs": out_crs.to_string(),
        "output_crs_wkt": out_crs.to_wkt(),
        "mask_info": mask_info,
        "selected_fraction_of_total": float(mask.sum() / pd.lon.size),
        "selected_point_summaries": {
            "x": summarize_array(x),
            "y": summarize_array(y),
            "z_refh": summarize_array(z),
            "snr": summarize_array(snr),
            "amp": summarize_array(pd.amp[mask]),
            "thres": summarize_array(pd.thres[mask]),
        },
        "grid": {
            k: (_json_safe(v) if k != "transform" else tuple(grid["transform"])) for k, v in grid.items()
        },
        "dsm_summary": {
            "aggregation": cfg.aggregation,
            "min_points_per_cell": cfg.min_points_per_cell,
            "strict_valid_cells": int(strict_valid.sum()),
            "filled_valid_cells": int(filled_valid.sum()),
            "support_cells": int(support_valid.sum()),
            "outside_support_cells": int(outside_support.sum()),
            "idw_filled_cells": int(idw_fill_mask.sum()),
            "nearest_filled_cells": int(nearest_fill_mask.sum()),
            "strict_valid_cell_fraction": float(strict_valid.sum() / grids["strict_dsm"].size),
            "filled_valid_cell_fraction": float(filled_valid.sum() / grids["filled_dsm"].size),
            "support_cell_fraction": float(support_valid.sum() / support_valid.size),
            "strict_dsm_values": summarize_array(grids["strict_dsm"][strict_valid]),
            "filled_dsm_values": summarize_array(grids["filled_dsm"][filled_valid]),
            "point_count_per_nonempty_cell": summarize_array(grids["count"][grids["count"] > 0]),
            "fill_distance_m_for_nearest_fallback": summarize_array(grids["nearest_distance_m"][nearest_fill_mask]),
            "strict_cells_match_filled_on_observed": strict_cells_match,
            "all_observed_cells_retained_in_filled_dsm": observed_cells_preserved,
            "max_abs_diff_on_observed_cells_m": float(np.nanmax(observed_abs_diff, initial=0.0)),
        },
        "outputs": {
            "filled_dsm_tif": str(filled_dsm_path),
            "strict_dsm_tif": str(strict_dsm_path) if cfg.write_raw_strict_dsm else None,
            "support_mask_tif": str(support_path) if cfg.write_support_mask_raster else None,
            "fill_source_tif": str(fill_source_path) if cfg.write_fill_source_raster else None,
            "count_tif": str(count_path) if cfg.write_count_raster else None,
            "snr_max_tif": str(snr_max_path) if cfg.write_snr_max_raster else None,
            "snr_mean_tif": str(snr_mean_path) if cfg.write_snr_mean_raster else None,
            "selected_las": str(las_path) if las_path is not None else None,
            "preview_pngs": [
                str(cfg.out_dir / "selected_surface_points_snr_xy.png"),
                str(cfg.out_dir / "filled_dsm_preview.png"),
                str(cfg.out_dir / "strict_dsm_preview.png"),
                str(cfg.out_dir / "selected_point_count_grid.png"),
                str(cfg.out_dir / "support_mask_preview.png"),
                str(cfg.out_dir / "fill_source_preview.png"),
            ] if cfg.write_preview_png else [],
        },
    }

    metadata_path = cfg.out_dir / f"{stem}_refh_surface_dsm_{thr_tag}_{res_tag}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2)

    print("\nDone.")
    print(json.dumps(_json_safe({
        "selected_points": int(mask.sum()),
        "selected_fraction": float(mask.sum() / pd.lon.size),
        "output_crs": out_crs.to_string(),
        "dsm_shape": [int(grids["filled_dsm"].shape[0]), int(grids["filled_dsm"].shape[1])],
        "strict_valid_dsm_cells": int(strict_valid.sum()),
        "support_cells": int(support_valid.sum()),
        "idw_filled_cells": int(idw_fill_mask.sum()),
        "nearest_filled_cells": int(nearest_fill_mask.sum()),
        "filled_dsm_tif": str(filled_dsm_path),
        "strict_dsm_tif": str(strict_dsm_path) if cfg.write_raw_strict_dsm else None,
        "selected_las": str(las_path) if las_path is not None else None,
        "metadata": str(metadata_path),
    }), indent=2))


if __name__ == "__main__":
    main()
