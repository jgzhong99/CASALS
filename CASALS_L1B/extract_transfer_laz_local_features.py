"""
Extract local 3D neighborhood features for transferred CASALS LAZ products.

This script reads one or more transfer LAZ files, computes notebook-aligned local
features from spherical XYZ neighborhoods, and writes sibling LAZ files with the
new feature dimensions appended as extra dims. The original transfer LAZ files are
not modified.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import laspy
import numpy as np
import pandas as pd
from laspy import ExtraBytesParams
from scipy.spatial import cKDTree


FEATURE_DIM_SPECS = {
    "local_neighbor_count": np.uint32,
    "point_density_pts_m3": np.float32,
    "roughness_m": np.float32,
    "local_z_span_m": np.float32,
    "local_z_nmad_m": np.float32,
    "local_slope_deg": np.float32,
    "sphericity": np.float32,
    "linearity": np.float32,
    "planarity": np.float32,
    "anisotropy": np.float32,
    "omnivariance": np.float32,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def safe_json_dump(payload: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("finite_mask requires at least one array")
    mask = np.ones(np.asarray(arrays[0]).shape[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(np.asarray(arr))
    return mask


def robust_nmad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    median = np.median(arr)
    return float(1.4826 * np.median(np.abs(arr - median)))


def maybe_quantiles(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None, None
    return float(np.percentile(arr, 5)), float(np.median(arr)), float(np.percentile(arr, 95))


def compute_local_neighbor_features(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    n_points = np.asarray(x).shape[0]
    out = {
        "local_neighbor_count": np.zeros(n_points, dtype=np.uint32),
        "point_density_pts_m3": np.full(n_points, np.nan, dtype=np.float64),
        "roughness_m": np.full(n_points, np.nan, dtype=np.float64),
        "local_z_span_m": np.full(n_points, np.nan, dtype=np.float64),
        "local_z_nmad_m": np.full(n_points, np.nan, dtype=np.float64),
        "local_slope_deg": np.full(n_points, np.nan, dtype=np.float64),
        "sphericity": np.full(n_points, np.nan, dtype=np.float64),
        "linearity": np.full(n_points, np.nan, dtype=np.float64),
        "planarity": np.full(n_points, np.nan, dtype=np.float64),
        "anisotropy": np.full(n_points, np.nan, dtype=np.float64),
        "omnivariance": np.full(n_points, np.nan, dtype=np.float64),
    }

    valid_xyz = finite_mask(x, y, z)
    if not np.any(valid_xyz):
        return out

    radius_m = float(config["LOCAL_FEATURE_RADIUS_M"])
    max_neighbors = max(
        int(config["LOCAL_FEATURE_MAX_NEIGHBORS"]),
        int(config["LOCAL_FEATURE_MIN_NEIGHBORS"]),
        1,
    )
    min_neighbors = int(config["LOCAL_FEATURE_MIN_NEIGHBORS"])
    chunk_size = max(1, int(config["LOCAL_FEATURE_QUERY_CHUNK_SIZE"]))
    sphere_volume_m3 = float((4.0 / 3.0) * np.pi * radius_m**3)

    xyz_valid = np.column_stack((
        np.asarray(x, dtype=np.float64)[valid_xyz],
        np.asarray(y, dtype=np.float64)[valid_xyz],
        np.asarray(z, dtype=np.float64)[valid_xyz],
    ))
    valid_indices = np.flatnonzero(valid_xyz)
    tree = cKDTree(xyz_valid)

    for start in range(0, xyz_valid.shape[0], chunk_size):
        stop = min(start + chunk_size, xyz_valid.shape[0])
        query_xyz = xyz_valid[start:stop]
        print(f"Feature chunk: {stop:,}/{xyz_valid.shape[0]:,}", flush=True)
        try:
            neighbor_count = np.asarray(
                tree.query_ball_point(query_xyz, r=radius_m, return_length=True, workers=-1),
                dtype=np.uint32,
            )
        except TypeError:
            neighbor_count = np.asarray(
                [len(ids) for ids in tree.query_ball_point(query_xyz, r=radius_m, workers=-1)],
                dtype=np.uint32,
            )

        distances, neighbor_idx = tree.query(
            query_xyz,
            k=max_neighbors,
            distance_upper_bound=radius_m,
            workers=-1,
        )
        if max_neighbors == 1:
            distances = distances[:, None]
            neighbor_idx = neighbor_idx[:, None]
        valid_neighbor_mask = np.isfinite(distances) & (neighbor_idx >= 0) & (neighbor_idx < xyz_valid.shape[0])

        point_ids = valid_indices[start:stop]
        out["local_neighbor_count"][point_ids] = neighbor_count
        out["point_density_pts_m3"][point_ids] = neighbor_count.astype(np.float64) / sphere_volume_m3

        for row in range(query_xyz.shape[0]):
            if int(neighbor_count[row]) < min_neighbors:
                continue
            row_mask = valid_neighbor_mask[row]
            if not np.any(row_mask):
                continue

            neighborhood_xyz = xyz_valid[neighbor_idx[row, row_mask]]
            if neighborhood_xyz.shape[0] < min_neighbors:
                continue

            neighborhood_z = neighborhood_xyz[:, 2]
            point_id = point_ids[row]
            out["local_z_span_m"][point_id] = float(np.nanmax(neighborhood_z) - np.nanmin(neighborhood_z))
            out["local_z_nmad_m"][point_id] = robust_nmad(neighborhood_z)

            centroid = neighborhood_xyz.mean(axis=0)
            centered = neighborhood_xyz - centroid
            try:
                eigvals, eigvecs = np.linalg.eigh((centered.T @ centered) / float(neighborhood_xyz.shape[0]))
            except np.linalg.LinAlgError:
                continue
            eigvals = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None)
            l1, l2, l3 = eigvals[::-1]
            if not np.isfinite(l1) or l1 <= 0.0:
                continue

            out["sphericity"][point_id] = float(l3 / l1)
            out["linearity"][point_id] = float((l1 - l2) / l1)
            out["planarity"][point_id] = float((l2 - l3) / l1)
            out["anisotropy"][point_id] = float((l1 - l3) / l1)
            out["omnivariance"][point_id] = float(np.cbrt(max(l1 * l2 * l3, 0.0)))

            normal = eigvecs[:, int(np.argmin(eigvals))]
            normal_norm = float(np.linalg.norm(normal))
            if not np.isfinite(normal_norm) or normal_norm <= 0.0:
                continue
            normal = normal / normal_norm
            center_xyz = query_xyz[row]
            out["roughness_m"][point_id] = float(abs((center_xyz - centroid) @ normal))
            out["local_slope_deg"][point_id] = float(
                np.degrees(np.arctan2(np.linalg.norm(normal[:2]), abs(normal[2])))
            )

    return out


def add_missing_feature_dims(las: laspy.LasData) -> None:
    existing = {dim.name for dim in las.point_format.dimensions}
    new_dims = []
    for name, dtype in FEATURE_DIM_SPECS.items():
        if name not in existing:
            new_dims.append(ExtraBytesParams(name=name, type=dtype, description=name[:32]))
    if new_dims:
        las.add_extra_dims(new_dims)


def cast_feature_array(name: str, values: np.ndarray) -> np.ndarray:
    dtype = FEATURE_DIM_SPECS[name]
    if dtype == np.uint32:
        return np.asarray(values, dtype=np.uint32)
    return np.asarray(values, dtype=np.float32)


def build_feature_summary_table(features: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in [
        "point_density_pts_m3",
        "roughness_m",
        "local_z_span_m",
        "local_z_nmad_m",
        "local_slope_deg",
        "sphericity",
        "linearity",
        "planarity",
        "anisotropy",
        "omnivariance",
    ]:
        arr = np.asarray(features[name], dtype=np.float64)
        valid_fraction = float(np.isfinite(arr).mean())
        p05, median, p95 = maybe_quantiles(arr)
        rows.append({
            "feature": name,
            "valid_fraction": valid_fraction,
            "p05": p05,
            "median": median,
            "p95": p95,
        })
    return pd.DataFrame(rows)


def process_one_laz(input_laz_path: Path, config: Dict[str, Any]) -> None:
    output_laz_path = input_laz_path.with_name(f"{input_laz_path.stem}_local_features.laz")
    metadata_path = input_laz_path.with_name(f"{input_laz_path.stem}_local_features_metadata.json")

    metadata: Dict[str, Any] = {
        "status": "running",
        "runtime_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_laz_path": str(input_laz_path),
        "output_laz_path": str(output_laz_path),
        "metadata_json_path": str(metadata_path),
        "config": json_safe(config),
        "warnings": [],
    }

    try:
        print(f"Processing transfer LAZ: {input_laz_path.stem}", flush=True)
        las = laspy.read(str(input_laz_path))
        crs = las.header.parse_crs()
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        z = np.asarray(las.z, dtype=np.float64)

        features = compute_local_neighbor_features(x, y, z, config)
        add_missing_feature_dims(las)
        for name, values in features.items():
            las[name] = cast_feature_array(name, values)

        las.header.generating_software = "extract_transfer_laz_local_features.py"
        las.write(str(output_laz_path))

        summary_df = build_feature_summary_table(features)
        print(pd.DataFrame([{
            "point_count": int(las.header.point_count),
            "point_format": int(las.header.point_format.id),
            "crs": crs.to_string() if crs else None,
        }]).to_string(index=False), flush=True)
        print(summary_df.to_string(index=False), flush=True)
        print(f"Wrote: {output_laz_path}", flush=True)

        metadata.update({
            "status": "success",
            "point_count": int(las.header.point_count),
            "point_format": int(las.header.point_format.id),
            "crs": crs.to_string() if crs else None,
            "written_feature_dims": list(FEATURE_DIM_SPECS),
            "feature_summary": summary_df.to_dict(orient="records"),
        })
    except Exception as exc:
        metadata.update({
            "status": "failed",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        })
        print(f"[ERROR] {input_laz_path.stem}: {exc}", flush=True)
    finally:
        safe_json_dump(metadata, metadata_path)


def main() -> None:
    INPUT_LAZ_PATHS = [
        Path("CASALS_L1B/outputs/transfer_3dep_labels_to_casals/casals_l1b_20241112T165718_001_02.laz"),
        Path("CASALS_L1B/outputs/transfer_3dep_labels_to_casals/casals_l1b_20241118T171757_001_02.laz"),
    ]

    CONFIG: Dict[str, Any] = {
        # Density audit on the full transfer clouds showed 5 m already gives
        # high min-neighbor coverage without broadening neighborhoods too much.
        "LOCAL_FEATURE_RADIUS_M": 5.0,
        "LOCAL_FEATURE_MAX_NEIGHBORS": 24,
        "LOCAL_FEATURE_MIN_NEIGHBORS": 6,
        "LOCAL_FEATURE_QUERY_CHUNK_SIZE": 100_000,
    }

    for input_laz_path in INPUT_LAZ_PATHS:
        process_one_laz(Path(input_laz_path), CONFIG)


if __name__ == "__main__":
    main()
