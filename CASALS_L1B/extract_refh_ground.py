"""Extract tentative CASALS refh ground candidates and a derived IDW DTM.

Scientific meaning:
    Input points are CASALS L1B geolocated max-Rx-bin / refh reference-return
    points. Ground labels here are tentative derived candidates only.

Outputs:
    Classified high-SNR LAS, ground-only LAS, DTM GeoTIFF, support/source masks,
    metadata JSON, and one QA preview.

This script does not:
    - produce an official ground DEM,
    - replace formal terrain validation,
    - change CASALS vertical datum semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import json
import math
import warnings

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

from scipy.spatial import cKDTree
from scipy import ndimage


@dataclass
class Config:
    h5_path: Path
    point_cloud_dir: Path
    out_dir: Path

    # High-confidence SNR threshold. For the tested CASALS granule, good_snr == refh_snr >= 5.0.
    snr_threshold: float = 5.0

    # Horizontal output CRS. If None, infer WGS84 UTM EPSG from median lon/lat.
    output_epsg_override: Optional[int] = None

    # Preliminary ground-candidate classification.
    prelim_grid_resolution_m: float = 10.0
    prelim_low_percentile: float = 10.0
    prelim_support_buffer_m: float = 35.0
    prelim_support_closing_m: float = 30.0
    # Progressive morphological filtering on the preliminary low surface.
    # These parameters intentionally mirror the earlier casals_highsnr_ground_morphology.py workflow.
    use_progressive_morphology: bool = True
    morphology_window_sizes_m: Tuple[float, ...] = (30.0, 50.0, 80.0, 120.0, 180.0)
    custom_height_thresholds_m: Optional[Tuple[float, ...]] = None
    base_threshold_m: float = 1.0
    slope_threshold: float = 0.06
    max_threshold_m: float = 8.0
    final_median_smooth_size_cells: int = 3

    ground_above_prelim_tolerance_m: float = 2.5
    ground_below_prelim_tolerance_m: float = 3.0

    # Final DTM raster.
    dtm_resolution_m: float = 10.0
    extent_buffer_m: float = 0.0
    nodata_float: float = -9999.0

    # Ground support mask for final DTM. This controls where internal holes may be filled.
    support_buffer_m: float = 35.0
    support_closing_m: float = 30.0
    support_fill_holes: bool = True

    # IDW interpolation from ground candidates.
    idw_radius_m: float = 45.0
    idw_k: int = 12
    idw_power: float = 2.0
    idw_min_neighbors: int = 3
    idw_chunk_size: int = 250_000

    # Internal hole filling. Filled only inside the support mask.
    fill_internal_holes: bool = True
    max_internal_fill_distance_m: Optional[float] = 60.0  # None means fill all holes inside support mask.

    # Preliminary low-surface smoothing before progressive morphology.
    prelim_smoothing_sigma_cells: float = 0.8

    # Optional smoothing, done by normalized Gaussian convolution inside support mask.
    smoothing_sigma_cells: float = 0.7

    # Mesh output from final filled/smoothed DTM.
    write_mesh: bool = False
    mesh_format: str = "ply"  # currently PLY only
    mesh_stride: int = 1
    max_mesh_vertices: int = 1_000_000
    mesh_max_edge_z_diff_m: Optional[float] = 40.0
    mesh_use_projected_coordinates: bool = True

    # LAS and diagnostics.
    write_classified_highsnr_las: bool = True
    write_ground_only_las: bool = True
    write_support_mask_tif: bool = True
    write_source_mask_tif: bool = True
    write_distance_tif: bool = True
    write_morphology_debug_tifs: bool = False
    write_preview_png: bool = True

    # LAS writing settings.
    las_scale_m: float = 0.001

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
class GridDef:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    width: int
    height: int
    resolution: float
    transform: Any


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
    """Return dataset by exact root name or recursive basename match."""
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
    arr = np.asarray(ds[...]).reshape(-1)
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

        thres_ds = find_dataset(h5, "refh_thres")
        if thres_ds is not None:
            thres = np.asarray(thres_ds[...], dtype=np.float64).reshape(-1)
        else:
            thres = np.full(lon.shape, np.nan, dtype=np.float64)

        snr_ds = find_dataset(h5, "refh_snr")
        if snr_ds is not None:
            snr = np.asarray(snr_ds[...], dtype=np.float64).reshape(-1)
        else:
            if thres_ds is None:
                raise KeyError("Neither refh_snr nor refh_thres was found; cannot compute SNR.")
            snr = np.divide(amp, thres, out=np.full_like(amp, np.nan), where=(thres != 0))

        good_arr = read_optional_dataset(h5, "good_snr", lon.size)
        good_snr = good_arr.astype(bool) if good_arr is not None else (snr >= 5.0)

        track_num = read_optional_dataset(h5, "track_num", lon.size)
        sweep_num = read_optional_dataset(h5, "sweep_num", lon.size)

        sizes = {
            "lon": lon.size,
            "lat": lat.size,
            "z": z.size,
            "snr": snr.size,
            "amp": amp.size,
            "thres": thres.size,
            "good_snr": good_snr.size,
        }
        if len(set(sizes.values())) != 1:
            raise ValueError(f"Required datasets do not have matching sizes: {sizes}")

        attrs: Dict[str, Any] = {}
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


def valid_base_mask(pd: PointData) -> np.ndarray:
    return (
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


def make_grid_from_points(x: np.ndarray, y: np.ndarray, resolution: float, buffer_m: float = 0.0) -> GridDef:
    xmin = float(np.nanmin(x) - buffer_m)
    xmax_raw = float(np.nanmax(x) + buffer_m)
    ymin_raw = float(np.nanmin(y) - buffer_m)
    ymax = float(np.nanmax(y) + buffer_m)

    width = int(math.ceil((xmax_raw - xmin) / resolution))
    height = int(math.ceil((ymax - ymin_raw) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid grid shape width={width}, height={height}")

    xmax = xmin + width * resolution
    ymin = ymax - height * resolution
    transform = from_origin(xmin, ymax, resolution, resolution)
    return GridDef(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width=width, height=height, resolution=resolution, transform=transform)


def xy_to_row_col(x: np.ndarray, y: np.ndarray, grid: GridDef) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    col = np.floor((x - grid.xmin) / grid.resolution).astype(np.int64)
    row = np.floor((grid.ymax - y) / grid.resolution).astype(np.int64)
    inside = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
    return row, col, inside


def disk_structure(radius_cells: int) -> np.ndarray:
    radius_cells = int(max(1, radius_cells))
    yy, xx = np.ogrid[-radius_cells: radius_cells + 1, -radius_cells: radius_cells + 1]
    return (xx * xx + yy * yy) <= radius_cells * radius_cells


def rasterize_percentile(x: np.ndarray, y: np.ndarray, z: np.ndarray, grid: GridDef, percentile: float) -> Tuple[np.ndarray, np.ndarray]:
    row, col, inside = xy_to_row_col(x, y, grid)
    row = row[inside]
    col = col[inside]
    z = z[inside]
    flat = row * grid.width + col
    n_cells = grid.height * grid.width

    out = np.full(n_cells, np.nan, dtype=np.float32)
    count = np.bincount(flat, minlength=n_cells).astype(np.uint32)

    if flat.size > 0:
        order = np.argsort(flat)
        flat_sorted = flat[order]
        z_sorted = z[order]
        starts = np.r_[0, np.flatnonzero(np.diff(flat_sorted)) + 1]
        ends = np.r_[starts[1:], flat_sorted.size]
        for s, e in zip(starts, ends):
            out[flat_sorted[s]] = np.float32(np.nanpercentile(z_sorted[s:e], percentile))

    return out.reshape(grid.height, grid.width), count.reshape(grid.height, grid.width)


def make_support_mask_from_points(x: np.ndarray, y: np.ndarray, grid: GridDef, buffer_m: float, closing_m: float, fill_holes: bool) -> np.ndarray:
    row, col, inside = xy_to_row_col(x, y, grid)
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    mask[row[inside], col[inside]] = True

    buffer_cells = int(math.ceil(buffer_m / grid.resolution))
    if buffer_cells > 0:
        mask = ndimage.binary_dilation(mask, structure=disk_structure(buffer_cells))

    closing_cells = int(math.ceil(closing_m / grid.resolution))
    if closing_cells > 0:
        st = disk_structure(closing_cells)
        mask = ndimage.binary_closing(mask, structure=st)

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)

    return mask.astype(bool)


def fill_nearest_within_mask(values: np.ndarray, support_mask: np.ndarray, resolution: float, max_distance_m: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Fill NaN cells inside support_mask from nearest valid cell. Returns filled array and distance raster."""
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


def normalized_gaussian_smooth(values: np.ndarray, support_mask: np.ndarray, sigma_cells: float) -> np.ndarray:
    if sigma_cells is None or sigma_cells <= 0:
        return values.astype(np.float32)
    valid = np.isfinite(values) & support_mask
    val0 = np.where(valid, values, 0.0).astype(np.float64)
    w0 = valid.astype(np.float64)
    num = ndimage.gaussian_filter(val0, sigma=float(sigma_cells), mode="nearest")
    den = ndimage.gaussian_filter(w0, sigma=float(sigma_cells), mode="nearest")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=(den > 1e-8))
    out[~support_mask] = np.nan
    return out.astype(np.float32)




def odd_window_cells(window_m: float, res: float) -> int:
    """Convert a metric morphology window to an odd number of raster cells."""
    n = int(round(float(window_m) / float(res)))
    n = max(3, n)
    if n % 2 == 0:
        n += 1
    return n


def height_threshold_for_window(window_m: float, cfg: Config, idx: int) -> float:
    """Progressive morphology height threshold for one opening stage."""
    if cfg.custom_height_thresholds_m is not None:
        if idx >= len(cfg.custom_height_thresholds_m):
            raise ValueError("custom_height_thresholds_m length must match morphology_window_sizes_m.")
        return float(cfg.custom_height_thresholds_m[idx])
    return float(min(cfg.max_threshold_m, cfg.base_threshold_m + cfg.slope_threshold * float(window_m)))


def progressive_morphological_opening_surface(surface: np.ndarray, support: np.ndarray, cfg: Config, res: float) -> Tuple[np.ndarray, list[Dict[str, Any]]]:
    """Apply progressive grayscale morphological opening to a filled low-surface raster.

    This reproduces the earlier morphology-only workflow before the final IDW/mesh step.
    The surface is assumed to be filled inside support; outside support stays NaN.
    """
    current = surface.astype(np.float64).copy()
    summaries: list[Dict[str, Any]] = []

    if not cfg.use_progressive_morphology:
        current[~support] = np.nan
        return current.astype(np.float32), summaries

    # ndimage grayscale morphology cannot handle NaN as terrain values, so fill outside
    # support with nearest in-support values for stable edge behavior, then restore NaN outside.
    valid = np.isfinite(current) & support
    if valid.sum() == 0:
        raise ValueError("No valid cells are available for progressive morphology.")
    filled_current, _ = fill_nearest_within_mask(current.astype(np.float32), support, res, max_distance_m=None)
    current = filled_current.astype(np.float64)

    for idx, window_m in enumerate(cfg.morphology_window_sizes_m):
        win = odd_window_cells(window_m, res)
        threshold = height_threshold_for_window(window_m, cfg, idx)

        eroded = ndimage.grey_erosion(current, size=(win, win), mode="nearest")
        opened = ndimage.grey_dilation(eroded, size=(win, win), mode="nearest")
        diff = current - opened
        elevated = support & (diff > threshold)

        before = current.copy()
        current = np.where(elevated, opened, current)
        current[~support] = np.nan

        abs_change = np.abs(current - before)
        summaries.append({
            "stage": int(idx + 1),
            "window_m": float(window_m),
            "window_cells": int(win),
            "height_threshold_m": float(threshold),
            "n_cells_adjusted": int(np.nansum(elevated)),
            "fraction_support_cells_adjusted": float(np.nansum(elevated) / max(1, int(np.sum(support)))),
            "median_abs_change_m": float(np.nanmedian(abs_change[support])),
            "p95_abs_change_m": float(np.nanpercentile(abs_change[support], 95)),
        })

    if cfg.final_median_smooth_size_cells and cfg.final_median_smooth_size_cells > 1:
        size = int(cfg.final_median_smooth_size_cells)
        if size % 2 == 0:
            size += 1
        filled_current, _ = fill_nearest_within_mask(current.astype(np.float32), support, res, max_distance_m=None)
        smoothed = ndimage.median_filter(filled_current, size=(size, size), mode="nearest")
        smoothed[~support] = np.nan
        abs_change = np.abs(smoothed - current)
        summaries.append({
            "stage": "final_median_smooth",
            "window_cells": int(size),
            "n_support_cells": int(np.sum(support)),
            "median_abs_change_m": float(np.nanmedian(abs_change[support])),
            "p95_abs_change_m": float(np.nanpercentile(abs_change[support], 95)),
        })
        current = smoothed.astype(np.float64)

    current[~support] = np.nan
    return current.astype(np.float32), summaries

def build_preliminary_ground_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, cfg: Config) -> Tuple[np.ndarray, GridDef, np.ndarray, np.ndarray, list[Dict[str, Any]], np.ndarray]:
    """Build a morphology-filtered preliminary ground surface from high-SNR refh points.

    Returns:
        prelim_ground: morphology-filtered tentative ground raster
        grid: grid definition
        support: preliminary support mask
        count: count of high-SNR points per seed cell
        morph_summaries: per-stage PMF metadata
        seed: initial low-percentile seed raster before filling/morphology
    """
    grid = make_grid_from_points(x, y, cfg.prelim_grid_resolution_m, buffer_m=cfg.extent_buffer_m)
    seed, count = rasterize_percentile(x, y, z, grid, cfg.prelim_low_percentile)
    support = make_support_mask_from_points(
        x, y, grid,
        buffer_m=cfg.prelim_support_buffer_m,
        closing_m=cfg.prelim_support_closing_m,
        fill_holes=True,
    )
    filled, _ = fill_nearest_within_mask(seed, support, grid.resolution, max_distance_m=None)
    low_smooth = normalized_gaussian_smooth(filled, support, cfg.prelim_smoothing_sigma_cells)
    prelim_ground, morph_summaries = progressive_morphological_opening_surface(low_smooth, support, cfg, grid.resolution)
    prelim_ground[~support] = np.nan
    return prelim_ground, grid, support, count, morph_summaries, seed

def sample_grid_nearest(x: np.ndarray, y: np.ndarray, grid_values: np.ndarray, grid: GridDef) -> np.ndarray:
    row, col, inside = xy_to_row_col(x, y, grid)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    good = inside
    out[good] = grid_values[row[good], col[good]]
    return out


def classify_highsnr_points_against_prelim(x: np.ndarray, y: np.ndarray, z: np.ndarray, prelim: np.ndarray, grid: GridDef, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    ground_est = sample_grid_nearest(x, y, prelim, grid)
    residual = z - ground_est
    cls = np.ones(x.size, dtype=np.uint8)
    cls[np.isfinite(residual) & (residual < -float(cfg.ground_below_prelim_tolerance_m))] = 7
    cls[np.isfinite(residual) & (residual >= -float(cfg.ground_below_prelim_tolerance_m)) & (residual <= float(cfg.ground_above_prelim_tolerance_m))] = 2
    cls[np.isfinite(residual) & (residual > float(cfg.ground_above_prelim_tolerance_m))] = 1
    return cls, residual


def idw_interpolate_ground(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid: GridDef,
    support_mask: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IDW interpolate ground points to grid cells inside support_mask.

    Returns initial_idw, source_mask, nearest_distance_m.
      source_mask: 0 outside support, 1 IDW interpolated, 2 internal nearest-fill.

    Important SciPy behavior: cKDTree.query marks missing neighbors with infinite
    distance and index == tree.n. We must never index z with those sentinel indices.
    """
    if x.size == 0:
        raise ValueError("No ground candidate points were supplied to IDW interpolation.")

    tree = cKDTree(np.column_stack((x, y)))

    rows, cols = np.nonzero(support_mask)
    qx = grid.xmin + (cols.astype(np.float64) + 0.5) * grid.resolution
    qy = grid.ymax - (rows.astype(np.float64) + 0.5) * grid.resolution
    queries = np.column_stack((qx, qy))

    out = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    source = np.zeros((grid.height, grid.width), dtype=np.uint8)
    nearest_dist = np.full((grid.height, grid.width), np.nan, dtype=np.float32)

    k = max(1, int(cfg.idw_k))
    chunk_size = max(10_000, int(cfg.idw_chunk_size))
    eps = 1e-6

    for start in range(0, queries.shape[0], chunk_size):
        end = min(start + chunk_size, queries.shape[0])
        q = queries[start:end]
        d, idx = tree.query(
            q,
            k=k,
            distance_upper_bound=float(cfg.idw_radius_m),
            workers=-1,
        )
        if k == 1:
            d = d[:, None]
            idx = idx[:, None]

        # Missing neighbors from cKDTree.query have idx == x.size and d == inf.
        valid = np.isfinite(d) & (idx >= 0) & (idx < x.size)
        n_valid = valid.sum(axis=1)
        nearest = np.min(np.where(valid, d, np.inf), axis=1)

        vals = np.full(q.shape[0], np.nan, dtype=np.float64)
        ok = n_valid >= int(cfg.idw_min_neighbors)
        if np.any(ok):
            vv = valid[ok]
            dd = d[ok]
            ii = idx[ok].copy()

            # Replace invalid sentinel indices before indexing z. Their weights are zero,
            # so the replacement value does not affect the weighted average.
            ii_safe = np.where(vv, ii, 0)
            weights = np.where(vv, 1.0 / np.maximum(dd, eps) ** float(cfg.idw_power), 0.0)
            z_nei = z[ii_safe]
            z_nei = np.where(vv, z_nei, 0.0)
            denom = np.sum(weights, axis=1)
            vals[ok] = np.divide(
                np.sum(weights * z_nei, axis=1),
                denom,
                out=np.full(np.sum(ok), np.nan, dtype=np.float64),
                where=denom > 0,
            )

        rr = rows[start:end]
        cc = cols[start:end]
        out[rr[ok], cc[ok]] = vals[ok].astype(np.float32)
        source[rr[ok], cc[ok]] = 1
        nearest_dist[rr, cc] = nearest.astype(np.float32)

    if cfg.fill_internal_holes:
        filled, fill_dist = fill_nearest_within_mask(out, support_mask, grid.resolution, cfg.max_internal_fill_distance_m)
        fill_mask = support_mask & ~np.isfinite(out) & np.isfinite(filled)
        out = filled
        source[fill_mask] = 2
        nearest_dist[fill_mask] = fill_dist[fill_mask].astype(np.float32)

    out[~support_mask] = np.nan
    nearest_dist[~support_mask] = np.nan
    return out, source, nearest_dist

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


def write_classified_las(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pd: PointData,
    mask: np.ndarray,
    classification: np.ndarray,
    residual: np.ndarray,
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
    las.classification = classification.astype(np.uint8)

    # Color by class: ground green, elevated red, low outlier blue.
    r = np.zeros(x.size, dtype=np.uint16)
    g = np.zeros(x.size, dtype=np.uint16)
    b = np.zeros(x.size, dtype=np.uint16)
    r[classification == 1] = 65535
    g[classification == 2] = 65535
    b[classification == 7] = 65535
    # fallback gray for unknown
    other = ~np.isin(classification, [1, 2, 7])
    r[other] = g[other] = b[other] = 32768
    las.red = r
    las.green = g
    las.blue = b

    extra_dims = [
        laspy.ExtraBytesParams(name="longitude", type=np.float64, description="refh_lon_deg"),
        laspy.ExtraBytesParams(name="latitude", type=np.float64, description="refh_lat_deg"),
        laspy.ExtraBytesParams(name="refh_snr", type=np.float64, description="refh_snr"),
        laspy.ExtraBytesParams(name="refh_amp", type=np.float64, description="refh_amp"),
        laspy.ExtraBytesParams(name="refh_thres", type=np.float64, description="refh_thres"),
        laspy.ExtraBytesParams(name="ground_resid", type=np.float64, description="z_minus_prelim"),
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
    las["ground_resid"] = residual
    las["pulse_index"] = pd.pulse_index[mask]
    if pd.track_num is not None:
        las["track_num"] = pd.track_num[mask].astype(np.uint16)
    if pd.sweep_num is not None:
        las["sweep_num"] = pd.sweep_num[mask].astype(np.uint32)

    las.write(path)


def write_ascii_ply_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment CASALS tentative ground DTM mesh; CRS recorded in metadata JSON\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def create_mesh_from_raster(
    dtm: np.ndarray,
    grid: GridDef,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, int]:
    valid = np.isfinite(dtm)
    requested_stride = max(1, int(cfg.mesh_stride))
    estimated_vertices = int(valid[::requested_stride, ::requested_stride].sum())
    stride = requested_stride
    if estimated_vertices > int(cfg.max_mesh_vertices):
        stride = int(math.ceil(requested_stride * math.sqrt(estimated_vertices / float(cfg.max_mesh_vertices))))
        stride = max(1, stride)
        warnings.warn(
            f"Mesh would contain about {estimated_vertices:,} vertices at stride={requested_stride}. "
            f"Automatically increasing stride to {stride}."
        )

    dtm_s = dtm[::stride, ::stride]
    valid_s = np.isfinite(dtm_s)
    h, w = dtm_s.shape

    vertex_map = -np.ones((h, w), dtype=np.int64)
    rr, cc = np.nonzero(valid_s)
    vertex_map[rr, cc] = np.arange(rr.size, dtype=np.int64)

    x = grid.xmin + ((cc * stride).astype(np.float64) + 0.5) * grid.resolution
    y = grid.ymax - ((rr * stride).astype(np.float64) + 0.5) * grid.resolution
    z = dtm_s[rr, cc].astype(np.float64)
    vertices = np.column_stack((x, y, z))

    faces = []
    max_dz = cfg.mesh_max_edge_z_diff_m
    for r in range(h - 1):
        for c in range(w - 1):
            ids = [vertex_map[r, c], vertex_map[r, c + 1], vertex_map[r + 1, c], vertex_map[r + 1, c + 1]]
            if min(ids) < 0:
                continue
            z00 = dtm_s[r, c]
            z01 = dtm_s[r, c + 1]
            z10 = dtm_s[r + 1, c]
            z11 = dtm_s[r + 1, c + 1]
            if max_dz is not None:
                if (np.nanmax([z00, z01, z10, z11]) - np.nanmin([z00, z01, z10, z11])) > float(max_dz):
                    continue
            v00, v01, v10, v11 = ids
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    return vertices, np.asarray(faces, dtype=np.int32), stride


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> str:
    try:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()
        ok = o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)
        if not ok:
            raise RuntimeError("open3d.io.write_triangle_mesh returned False")
        return "open3d_binary_ply"
    except Exception as exc:
        warnings.warn(f"Open3D mesh export failed or unavailable; writing ASCII PLY fallback. Reason: {exc}")
        write_ascii_ply_mesh(path, vertices, faces)
        return "ascii_ply_fallback"


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


def write_previews(out_dir: Path, xg: np.ndarray, yg: np.ndarray, zg: np.ndarray, dtm: np.ndarray, source: np.ndarray, residual: np.ndarray, cfg: Config) -> None:
    if plt is None:
        warnings.warn(f"matplotlib import failed; previews not written: {_MATPLOTLIB_IMPORT_ERROR}")
        return

    rng = np.random.default_rng(cfg.random_seed)
    max_pts = min(200_000, xg.size)
    idx = rng.choice(xg.size, size=max_pts, replace=False) if xg.size > max_pts else np.arange(xg.size)
    r = residual[np.isfinite(residual)]
    lo, hi = np.nanpercentile(r, [0.5, 99.5]) if r.size else (-5, 5)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im = axes[0, 0].imshow(np.where(np.isfinite(dtm), dtm, np.nan), cmap="terrain")
    axes[0, 0].set_title("Tentative ground DTM")
    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")
    fig.colorbar(im, ax=axes[0, 0], label="CASALS refh height (m)")

    im = axes[0, 1].imshow(source, cmap="viridis", vmin=0, vmax=2)
    axes[0, 1].set_title("Source mask")
    axes[0, 1].set_xlabel("Column")
    axes[0, 1].set_ylabel("Row")
    fig.colorbar(im, ax=axes[0, 1], label="0 outside, 1 IDW, 2 fill")

    sc = axes[1, 0].scatter(xg[idx], yg[idx], c=zg[idx], s=0.8, cmap="terrain", linewidths=0)
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].set_title("Ground candidates")
    axes[1, 0].set_xlabel("Projected X (m)")
    axes[1, 0].set_ylabel("Projected Y (m)")
    fig.colorbar(sc, ax=axes[1, 0], label="refh height (m)")

    axes[1, 1].hist(r[(r >= lo) & (r <= hi)], bins=120, color="0.35")
    axes[1, 1].axvline(cfg.ground_above_prelim_tolerance_m, color="r", linestyle="--", label="above tolerance")
    axes[1, 1].axvline(-cfg.ground_below_prelim_tolerance_m, color="b", linestyle="--", label="below tolerance")
    axes[1, 1].set_title("Residual to preliminary ground")
    axes[1, 1].set_xlabel("z - preliminary ground (m)")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "tentative_ground_qa_preview.png", dpi=220)
    plt.close(fig)


def main() -> None:
    # -------------------------------------------------------------------------
    # USER SETTINGS: edit here. No argparse.
    # -------------------------------------------------------------------------
    cfg = Config(
        h5_path=Path(r"./casals_h5_downloads/casals_l1b_20241112T165718_001_02.h5"),
        point_cloud_dir=Path(r"./point_cloud_data/extract_refh_ground"),
        out_dir=Path(r"./outputs/extract_refh_ground"),

        # Use high-confidence CASALS reference returns. In the tested granule, this equals good_snr=True.
        snr_threshold=5.0,

        # Preliminary ground-point classification.
        prelim_grid_resolution_m=1.0,
        prelim_low_percentile=10.0,
        prelim_support_buffer_m=35.0,
        prelim_support_closing_m=30.0,
        prelim_smoothing_sigma_cells=0.8,
        ground_above_prelim_tolerance_m=2.5,
        ground_below_prelim_tolerance_m=3.0,

        # Final continuous DTM.
        dtm_resolution_m=5.0,
        support_buffer_m=35.0,
        support_closing_m=30.0,
        support_fill_holes=True,
        idw_radius_m=45.0,
        idw_k=12,
        idw_power=2.0,
        idw_min_neighbors=3,
        fill_internal_holes=True,
        max_internal_fill_distance_m=60.0,
        smoothing_sigma_cells=0.7,

        # Mesh export from the final DTM.
        write_mesh=False,
        mesh_stride=1,
        max_mesh_vertices=1_000_000,
        mesh_max_edge_z_diff_m=40.0,

        # Extra products for QA. Keep these True during development.
        write_classified_highsnr_las=True,
        write_ground_only_las=True,
        write_support_mask_tif=True,
        write_source_mask_tif=True,
        write_distance_tif=True,
        write_preview_png=True,
    )
    # -------------------------------------------------------------------------

    if rasterio is None:
        raise RuntimeError(f"rasterio import failed: {_RASTERIO_IMPORT_ERROR}")

    cfg.point_cloud_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("CASALS high-SNR morphology ground extraction + IDW surface + mesh")
    print("=" * 96)
    print(f"H5: {cfg.h5_path.resolve()}")
    print(f"Point-cloud directory: {cfg.point_cloud_dir.resolve()}")
    print(f"Output: {cfg.out_dir.resolve()}")
    print(f"SNR threshold: {cfg.snr_threshold}")
    print("Scientific caveat: output is tentative CASALS-derived ground candidate surface, not final DEM.")

    pd = read_point_data(cfg.h5_path)
    base = valid_base_mask(pd)
    high = base & (pd.snr >= float(cfg.snr_threshold))
    if not np.any(high):
        raise ValueError("No points survived SNR threshold. Lower cfg.snr_threshold.")

    epsg = cfg.output_epsg_override or infer_wgs84_utm_epsg(pd.lon[base], pd.lat[base])
    x_all, y_all, out_crs = transform_lonlat_to_projected(pd.lon, pd.lat, epsg)

    xh = x_all[high]
    yh = y_all[high]
    zh = pd.z[high]

    print(f"Total records: {pd.lon.size:,}")
    print(f"Base valid records: {int(base.sum()):,}")
    print(f"High-SNR records: {int(high.sum()):,} ({high.sum()/pd.lon.size:.6%} of total)")
    print(f"Output CRS: {out_crs.to_string()}")

    # 1. Preliminary low surface from high-SNR points.
    prelim, prelim_grid, prelim_support, prelim_count, morphology_stage_summaries, prelim_seed = build_preliminary_ground_surface(xh, yh, zh, cfg)

    # 2. Classify high-SNR points using residuals to the preliminary low/smoothed surface.
    cls_high, residual_high = classify_highsnr_points_against_prelim(xh, yh, zh, prelim, prelim_grid, cfg)
    ground_local = cls_high == 2
    if ground_local.sum() < 10:
        raise ValueError("Too few ground candidate points after residual filtering. Relax tolerances.")

    xg = xh[ground_local]
    yg = yh[ground_local]
    zg = zh[ground_local]

    print(f"Ground candidate points: {int(ground_local.sum()):,} / {xh.size:,}")
    unique_cls, unique_cnt = np.unique(cls_high, return_counts=True)
    print("High-SNR classification counts:", {int(c): int(n) for c, n in zip(unique_cls, unique_cnt)})

    # 3. Final ground support mask and IDW DTM.
    final_grid = make_grid_from_points(xg, yg, cfg.dtm_resolution_m, buffer_m=cfg.extent_buffer_m)
    support_mask = make_support_mask_from_points(
        xg,
        yg,
        final_grid,
        buffer_m=cfg.support_buffer_m,
        closing_m=cfg.support_closing_m,
        fill_holes=cfg.support_fill_holes,
    )

    dtm_idw, source_mask, nearest_distance = idw_interpolate_ground(xg, yg, zg, final_grid, support_mask, cfg)
    dtm_final = normalized_gaussian_smooth(dtm_idw, support_mask & np.isfinite(dtm_idw), cfg.smoothing_sigma_cells)
    dtm_final[~support_mask] = np.nan

    # Preserve source mask after smoothing.
    valid_dtm = np.isfinite(dtm_final)
    source_mask[(source_mask == 0) & valid_dtm & support_mask] = 2
    source_mask[~support_mask] = 0

    # 4. Output files.
    stem = cfg.h5_path.stem
    thr_tag = f"snr{cfg.snr_threshold:g}".replace(".", "p")
    res_tag = f"{cfg.dtm_resolution_m:g}m".replace(".", "p")
    base_name = f"{stem}_tentative_ground_idw_filled_{thr_tag}_{res_tag}_epsg{epsg}"

    dtm_path = cfg.out_dir / f"{base_name}.tif"
    morphology_path = cfg.out_dir / f"{base_name}_morphology_prelim_surface.tif"
    morphology_seed_path = cfg.out_dir / f"{base_name}_morphology_seed_surface.tif"
    morphology_support_path = cfg.out_dir / f"{base_name}_morphology_support_mask.tif"
    support_path = cfg.out_dir / f"{base_name}_support_mask.tif"
    source_path = cfg.out_dir / f"{base_name}_source_mask.tif"
    distance_path = cfg.out_dir / f"{base_name}_nearest_ground_distance_m.tif"
    classified_las_path = cfg.point_cloud_dir / f"{base_name}_classified_highsnr.las"
    ground_las_path = cfg.point_cloud_dir / f"{base_name}_ground_candidates_only.las"
    mesh_path = cfg.point_cloud_dir / f"{base_name}_mesh.ply"
    metadata_path = cfg.out_dir / f"{base_name}_metadata.json"

    write_float_geotiff(dtm_path, dtm_final, out_crs, final_grid.transform, cfg.nodata_float)
    if cfg.write_morphology_debug_tifs:
        write_float_geotiff(morphology_path, prelim, out_crs, prelim_grid.transform, cfg.nodata_float)
        write_float_geotiff(morphology_seed_path, prelim_seed, out_crs, prelim_grid.transform, cfg.nodata_float)
        write_uint8_geotiff(morphology_support_path, prelim_support.astype(np.uint8), out_crs, prelim_grid.transform, nodata=0)
    if cfg.write_support_mask_tif:
        write_uint8_geotiff(support_path, support_mask.astype(np.uint8), out_crs, final_grid.transform, nodata=0)
    if cfg.write_source_mask_tif:
        write_uint8_geotiff(source_path, source_mask, out_crs, final_grid.transform, nodata=0)
    if cfg.write_distance_tif:
        write_float_geotiff(distance_path, nearest_distance, out_crs, final_grid.transform, cfg.nodata_float)

    if cfg.write_classified_highsnr_las:
        write_classified_las(classified_las_path, xh, yh, zh, pd, high, cls_high, residual_high, out_crs, cfg)
    if cfg.write_ground_only_las:
        # Build a mask over original records for selected high-SNR ground candidates.
        high_indices = np.flatnonzero(high)
        ground_indices = high_indices[ground_local]
        ground_mask_global = np.zeros(pd.lon.size, dtype=bool)
        ground_mask_global[ground_indices] = True
        residual_ground = residual_high[ground_local]
        write_classified_las(
            ground_las_path,
            x_all[ground_mask_global],
            y_all[ground_mask_global],
            pd.z[ground_mask_global],
            pd,
            ground_mask_global,
            np.full(ground_mask_global.sum(), 2, dtype=np.uint8),
            residual_ground,
            out_crs,
            cfg,
        )

    mesh_info: Dict[str, Any] = {"written": False}
    if cfg.write_mesh:
        vertices, faces, mesh_stride_used = create_mesh_from_raster(dtm_final, final_grid, cfg)
        if vertices.size == 0 or faces.size == 0:
            warnings.warn("No mesh written because DTM had insufficient valid connected cells.")
            mesh_info = {"written": False, "reason": "insufficient_valid_cells"}
        else:
            writer_used = write_mesh(mesh_path, vertices, faces)
            mesh_info = {
                "written": True,
                "path": str(mesh_path),
                "writer": writer_used,
                "mesh_stride_used": int(mesh_stride_used),
                "n_vertices": int(vertices.shape[0]),
                "n_faces": int(faces.shape[0]),
                "crs_note": "PLY does not formally store EPSG CRS; coordinates are in output_crs and CRS is recorded in metadata.",
            }

    if cfg.write_preview_png:
        write_previews(cfg.out_dir, xg, yg, zg, dtm_final, source_mask, residual_high, cfg)

    valid_cells = int(np.isfinite(dtm_final).sum())
    total_cells = int(dtm_final.size)

    metadata = {
        "script": "extract_refh_ground.py",
        "script_semantics": "CASALS_high_SNR_refh_tentative_ground_candidates_and_IDW_DTM",
        "scientific_notes": [
            "Each point is one CASALS L1B max-Rx-bin/refh reference-return point.",
            "refh is WGS84 ellipsoidal height unless otherwise documented.",
            "This is not an official multi-return point cloud.",
            "This is not a ground-classified point cloud unless explicitly marked as tentative derived product.",
            "This output is a derived tentative ground candidate surface, not an official ground DEM.",
        ],
        "config": asdict(cfg),
        "source_h5": str(cfg.h5_path),
        "source_attrs_subset": pd.attrs,
        "output_crs": out_crs.to_string(),
        "output_crs_wkt": out_crs.to_wkt(),
        "counts": {
            "n_total_records": int(pd.lon.size),
            "n_base_valid": int(base.sum()),
            "n_high_snr": int(high.sum()),
            "n_ground_candidates": int(ground_local.sum()),
            "n_elevated_or_non_ground_candidates": int((cls_high == 1).sum()),
            "n_low_outliers": int((cls_high == 7).sum()),
        },
        "summaries": {
            "high_snr_z": summarize_array(zh),
            "high_snr_residual_to_prelim": summarize_array(residual_high),
            "ground_candidate_z": summarize_array(zg),
            "dtm_values": summarize_array(dtm_final[np.isfinite(dtm_final)]),
            "nearest_ground_distance_inside_support": summarize_array(nearest_distance[support_mask]),
        },
        "preliminary_grid": {
            "width": prelim_grid.width,
            "height": prelim_grid.height,
            "resolution": prelim_grid.resolution,
            "valid_seed_cells": int(np.isfinite(prelim).sum()),
            "support_cells": int(prelim_support.sum()),
        },
        "final_grid": {
            "xmin": final_grid.xmin,
            "xmax": final_grid.xmax,
            "ymin": final_grid.ymin,
            "ymax": final_grid.ymax,
            "width": final_grid.width,
            "height": final_grid.height,
            "resolution": final_grid.resolution,
            "transform": tuple(final_grid.transform),
            "support_cells": int(support_mask.sum()),
            "valid_dtm_cells": valid_cells,
            "total_cells": total_cells,
            "valid_cell_fraction": float(valid_cells / total_cells),
            "source_mask_codes": {
                "0": "outside_support_or_nodata",
                "1": "IDW_interpolated_from_ground_candidates",
                "2": "internal_hole_filled_from_nearest_valid_DTM_cell",
            },
        },
        "mesh": mesh_info,
        "outputs": {
            "dtm_tif": str(dtm_path),
            "morphology_prelim_surface_tif": str(morphology_path) if cfg.write_morphology_debug_tifs else None,
            "morphology_seed_surface_tif": str(morphology_seed_path) if cfg.write_morphology_debug_tifs else None,
            "morphology_support_mask_tif": str(morphology_support_path) if cfg.write_morphology_debug_tifs else None,
            "support_mask_tif": str(support_path) if cfg.write_support_mask_tif else None,
            "source_mask_tif": str(source_path) if cfg.write_source_mask_tif else None,
            "nearest_ground_distance_tif": str(distance_path) if cfg.write_distance_tif else None,
            "classified_highsnr_las": str(classified_las_path) if cfg.write_classified_highsnr_las else None,
            "ground_candidates_las": str(ground_las_path) if cfg.write_ground_only_las else None,
            "mesh_ply": str(mesh_path) if mesh_info.get("written") else None,
            "qa_preview_png": str(cfg.out_dir / "tentative_ground_qa_preview.png") if cfg.write_preview_png else None,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2)

    print("\nDone.")
    print(json.dumps(_json_safe({
        "dtm_tif": str(dtm_path),
        "mesh_ply": str(mesh_path) if mesh_info.get("written") else None,
        "metadata": str(metadata_path),
        "ground_candidates": int(ground_local.sum()),
        "valid_dtm_cells": valid_cells,
        "valid_dtm_cell_fraction": float(valid_cells / total_cells),
        "mesh": mesh_info,
    }), indent=2))


if __name__ == "__main__":
    main()
