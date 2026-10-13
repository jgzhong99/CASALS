"""Bounded research helpers for empirical CASALS–3DEP comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.transformer import TransformerGroup
from scipy.spatial import cKDTree


def horizontal_transform_audit(
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    target_crs: Any = "EPSG:6347",
    area_of_interest: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Transform nominal H5 longitude/latitude in 2D and record the operation."""

    lon = np.asarray(longitude, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)
    if lon.shape != lat.shape or not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError("Longitude and latitude must be same-shaped finite arrays.")
    source = CRS.from_epsg(4326)
    target = CRS.from_user_input(target_crs)
    aoi = AreaOfInterest(*area_of_interest) if area_of_interest is not None else None
    group = TransformerGroup(
        source,
        target,
        always_xy=True,
        area_of_interest=aoi,
        allow_ballpark=True,
    )
    if not group.transformers:
        raise RuntimeError("No usable 2D horizontal transformation is available.")
    transformer = group.transformers[0]
    x, y = transformer.transform(lon, lat, errcheck=True)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise RuntimeError("Horizontal transformation returned non-finite coordinates.")
    accuracy = float(transformer.accuracy)
    return x, y, {
        "source_crs_assumption": "EPSG:4326 nominal H5 longitude/latitude; realization and epoch unresolved",
        "target_crs": target.to_string(),
        "selected_operation": transformer.description,
        "operation_definition": transformer.definition,
        "operation_accuracy_m": accuracy if accuracy >= 0 else None,
        "ballpark_operation": "ballpark" in transformer.description.lower(),
        "preferred_operation_available": bool(group.best_available),
        "unavailable_operation_count": len(group.unavailable_operations),
        "coordinate_status": "2D exploratory horizontal transformation; no vertical operation applied",
        "area_of_interest_lonlat": list(area_of_interest) if area_of_interest is not None else None,
    }


def class2_ground_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
) -> np.ndarray:
    """Return finite XYZ rows whose LAS classification is ASPRS class 2."""

    arrays = [np.asarray(v) for v in (x, y, z, classification)]
    if any(a.ndim != 1 for a in arrays) or len({a.size for a in arrays}) != 1:
        raise ValueError("x, y, z, and classification must be same-length 1D arrays.")
    xyz = np.column_stack([arrays[0], arrays[1], arrays[2]]).astype(np.float64, copy=False)
    keep = (arrays[3] == 2) & np.all(np.isfinite(xyz), axis=1)
    return xyz[keep]


def build_class2_reference_grid(
    laz_path: str | Path,
    *,
    resolution_m: float = 0.75,
    validation_stride: int = 10_000,
    chunk_size: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Stream class-2 LAS points into the reference grid and a held-out sample.

    Every ``validation_stride``-th LAS record is excluded from the grid. Those
    original points can then independently check local interpolation without
    fitting a second surface for every CASALS pulse.
    """

    import laspy

    if resolution_m <= 0 or validation_stride < 2 or chunk_size < 1:
        raise ValueError("Resolution, validation stride, and chunk size must be positive.")
    path = Path(laz_path)
    with laspy.open(path) as reader:
        header = reader.header
        crs = header.parse_crs()
        if crs is None or crs.to_epsg() != 6347:
            raise ValueError(f"Expected horizontal EPSG:6347 LAZ coordinates, got {crs}.")
        min_x, min_y = map(float, header.mins[:2])
        max_x, max_y = map(float, header.maxs[:2])
        x0 = np.floor(min_x / resolution_m) * resolution_m
        y0 = np.floor(min_y / resolution_m) * resolution_m
        nx = int(np.ceil((max_x - x0) / resolution_m)) + 1
        ny = int(np.ceil((max_y - y0) / resolution_m)) + 1
        z_sum = np.zeros(nx * ny, dtype=np.float64)
        counts = np.zeros(nx * ny, dtype=np.int32)
        validation_parts = []
        class2_n = 0
        point_offset = 0
        for chunk in reader.chunk_iterator(chunk_size):
            size = len(chunk)
            record_index = point_offset + np.arange(size, dtype=np.int64)
            point_offset += size
            keep = np.asarray(chunk.classification) == 2
            if not keep.any():
                continue
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            finite = keep & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            validation = finite & (record_index % validation_stride == 0)
            if validation.any():
                validation_parts.append(np.column_stack((x[validation], y[validation], z[validation])))
            train = finite & ~validation
            if not train.any():
                continue
            ix = np.floor((x[train] - x0) / resolution_m).astype(np.int64)
            iy = np.floor((y[train] - y0) / resolution_m).astype(np.int64)
            inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            flat = iy[inside] * nx + ix[inside]
            np.add.at(z_sum, flat, z[train][inside])
            np.add.at(counts, flat, 1)
            class2_n += int(inside.sum())

    occupied = np.flatnonzero(counts)
    if not occupied.size:
        raise ValueError(f"No finite ASPRS class-2 points were found in {path}.")
    iy, ix = np.divmod(occupied, nx)
    ground_xy = np.column_stack((x0 + (ix + 0.5) * resolution_m, y0 + (iy + 0.5) * resolution_m))
    ground_z = z_sum[occupied] / counts[occupied]
    validation_xyz = (
        np.concatenate(validation_parts, axis=0)
        if validation_parts
        else np.empty((0, 3), dtype=np.float64)
    )
    metadata = {
        "source_laz": path.name,
        "source_laz_size_bytes": int(path.stat().st_size),
        "las_point_count": int(header.point_count),
        "class2_training_point_count": class2_n,
        "held_out_class2_point_count": int(len(validation_xyz)),
        "grid_cell_count": int(len(ground_z)),
        "grid_resolution_m": float(resolution_m),
        "validation_stride_las_records": int(validation_stride),
        "crs": crs.to_string(),
        "bounds_m": {
            "min_x": float(header.mins[0]), "min_y": float(header.mins[1]), "min_z": float(header.mins[2]),
            "max_x": float(header.maxs[0]), "max_y": float(header.maxs[1]), "max_z": float(header.maxs[2]),
        },
        "validation_rule": "ASPRS class-2 points at deterministic LAS record indices divisible by validation_stride are excluded from the grid.",
    }
    return ground_xy, ground_z, validation_xyz, metadata


def load_matched_elevation_data(
    h5_path: str | Path,
    laz_path: str | Path,
    result_dir: str | Path,
) -> dict[str, Any]:
    """Load and verify the preserved full-granule match table against its inputs."""

    import h5py
    import laspy
    import pandas as pd

    h5_path, laz_path, result_dir = Path(h5_path), Path(laz_path), Path(result_dir)
    audit = json.loads((result_dir / "reference_frame_audit.json").read_text(encoding="utf-8"))
    if h5_path.stat().st_size != audit.get("source_h5_size_bytes"):
        raise RuntimeError("The current H5 size does not match the preserved elevation-result audit.")
    if laz_path.stat().st_size != audit.get("source_laz_size_bytes"):
        raise RuntimeError("The current LAZ size does not match the preserved elevation-result audit.")
    with h5py.File(h5_path, "r") as h5:
        attrs = {str(key): str(value) for key, value in h5.attrs.items()}
        if attrs != audit.get("source_h5_attributes"):
            raise RuntimeError("The current H5 attributes differ from the preserved elevation-result audit.")
        scalar = {
            name: np.asarray(h5[name])
            for name in ("refh_longitude", "refh_latitude", "refh", "refh_snr", "sweep_num", "track_num", "good_snr")
        }
    n = len(scalar["refh"])
    if any(len(values) != n for values in scalar.values()):
        raise RuntimeError("The current H5 scalar fields have inconsistent lengths.")
    if not np.array_equal(scalar["good_snr"], scalar["refh_snr"] >= 5.0):
        raise RuntimeError("The current H5 good_snr field differs from refh_snr >= 5.")

    with laspy.open(laz_path) as reader:
        header = reader.header
        grid_meta = audit["reference_grid"]
        crs = header.parse_crs()
        if crs is None or crs.to_epsg() != 6347:
            raise RuntimeError("The current LAZ does not retain the audited EPSG:6347 horizontal CRS.")
        if int(header.point_count) != int(grid_meta["las_point_count"]):
            raise RuntimeError("The current LAZ point count differs from the preserved elevation-result audit.")
        if int(header.point_count) != int(audit["historical_provenance_evidence"]["clip_result_point_count"]):
            raise RuntimeError("The current LAZ point count no longer matches its clipping sidecar.")
        if header.parse_crs().to_epsg() != int(grid_meta["crs"].split(":")[-1]):
            raise RuntimeError("The current LAZ CRS disagrees with the preserved grid metadata.")

    columns = [
        "pulse_index", "sweep_num", "track_num", "longitude_deg", "latitude_deg",
        "casals_refh_m", "refh_snr", "easting_m_epsg6347", "northing_m_epsg6347",
        "reference_height_local_plane_m", "reference_height_local_median_m",
        "raw_difference_plane_m", "raw_difference_local_median_m", "neighbor_count_within_3m",
        "robust_neighbor_count", "nearest_ground_distance_m", "local_slope_deg",
        "plane_residual_nmad_m", "ground_roughness_nmad_m", "surface_acceptable", "spatial_split",
    ]
    matched = pd.read_parquet(result_dir / "matched_casals_3dep_points.parquet", columns=columns)
    ids = matched["pulse_index"].to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or (ids.size and (ids.min() < 0 or ids.max() >= n)):
        raise RuntimeError("The preserved match table has invalid or duplicate H5 pulse indices.")
    if not np.array_equal(matched["sweep_num"].to_numpy(), scalar["sweep_num"][ids]):
        raise RuntimeError("Sweep IDs in the preserved match table do not match the current H5.")
    if not np.array_equal(matched["track_num"].to_numpy(), scalar["track_num"][ids]):
        raise RuntimeError("Track IDs in the preserved match table do not match the current H5.")
    if not np.allclose(matched["casals_refh_m"], scalar["refh"][ids], rtol=0.0, atol=1e-9):
        raise RuntimeError("Refh values in the preserved match table do not match the current H5.")
    if not np.allclose(matched["refh_snr"], scalar["refh_snr"][ids], rtol=0.0, atol=1e-12):
        raise RuntimeError("SNR values in the preserved match table do not match the current H5.")
    if not np.allclose(
        matched["raw_difference_plane_m"],
        matched["casals_refh_m"] - matched["reference_height_local_plane_m"],
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError("Stored raw differences do not reproduce from Refh minus the local plane.")

    lon, lat = scalar["refh_longitude"], scalar["refh_latitude"]
    geo_valid = np.isfinite(lon) & np.isfinite(lat)
    x = np.full(n, np.nan); y = np.full(n, np.nan)
    if geo_valid.any():
        x[geo_valid], y[geo_valid], transform = horizontal_transform_audit(
            lon[geo_valid], lat[geo_valid], target_crs="EPSG:6347"
        )
    else:
        raise RuntimeError("The current H5 has no finite nominal pulse coordinates.")
    if not np.allclose(matched["easting_m_epsg6347"], x[ids], rtol=0.0, atol=0.02):
        raise RuntimeError("Projected eastings do not reproduce from current H5 coordinates.")
    if not np.allclose(matched["northing_m_epsg6347"], y[ids], rtol=0.0, atol=0.02):
        raise RuntimeError("Projected northings do not reproduce from current H5 coordinates.")

    return {
        "granule_id": h5_path.stem,
        "matched": matched,
        "raw_metrics_source": pd.read_csv(result_dir / "elevation_raw_difference_metrics.csv"),
        "holdout_metrics_source": pd.read_csv(result_dir / "elevation_empirical_holdout_metrics.csv"),
        "support_source": pd.read_csv(result_dir / "reference_ground_support.csv"),
        "audit": audit,
        "longitude": lon,
        "latitude": lat,
        "refh": scalar["refh"],
        "snr": scalar["refh_snr"],
        "sweep": scalar["sweep_num"],
        "track": scalar["track_num"],
        "x": x,
        "y": y,
        "h5_size_bytes": int(h5_path.stat().st_size),
        "laz_size_bytes": int(laz_path.stat().st_size),
        "horizontal_transform": transform,
    }


def summarize_matched_elevation_data(data: dict[str, Any]) -> dict[str, Any]:
    """Recompute supported raw and spatial-holdout metrics from the preserved matches."""

    import pandas as pd

    matched = data["matched"]
    raw_source = data["raw_metrics_source"]
    holdout_source = data["holdout_metrics_source"]
    labels = {"all_valid_refh": "All valid", "snr_ge_3": "SNR ≥ 3", "snr_ge_5": "SNR ≥ 5"}
    raw_values: dict[str, np.ndarray] = {}
    holdout_values: dict[str, np.ndarray] = {}
    raw_rows = []
    holdout_rows = []
    test_rows = None

    for key, label in labels.items():
        base = matched["surface_acceptable"].to_numpy(dtype=bool) & np.isfinite(matched["raw_difference_plane_m"])
        snr = matched["refh_snr"].to_numpy(dtype=np.float64)
        if key == "snr_ge_3":
            base &= snr >= 3.0
        elif key == "snr_ge_5":
            base &= snr >= 5.0
        raw = matched.loc[base, "raw_difference_plane_m"].to_numpy(dtype=np.float64)
        raw_values[label] = raw
        metrics = elevation_metrics(raw)
        source_row = raw_source[
            (raw_source.population == "whole_granule")
            & (raw_source.surface_quality == "surface_acceptable")
            & (raw_source.snr_stratum == key)
        ]
        if len(source_row) != 1:
            raise RuntimeError(f"Expected one preserved raw-metric row for {data['granule_id']} / {key}.")
        source_row = source_row.iloc[0]
        if int(source_row.n) != metrics["n"]:
            raise RuntimeError(f"Raw sample count changed for {data['granule_id']} / {key}.")
        for metric in ("bias_m", "median_m", "std_m", "rmse_m", "mae_m", "nmad_m", "p05_m", "p95_m"):
            if not np.isclose(float(source_row[metric]), float(metrics[metric]), rtol=0.0, atol=1e-8):
                raise RuntimeError(f"Raw {metric} no longer matches the preserved output for {data['granule_id']} / {key}.")
        raw_rows.append({
            "granule_id": data["granule_id"], "snr_stratum": key, "population": "all valid matched H5 pulses",
            "n_denominator": int(source_row.n_denominator), "n_supported_acceptable": int(metrics["n"]),
            "coverage": float(source_row.coverage), "difference_definition": source_row.difference_definition,
            "surface_method": source_row.surface_method, **metrics,
        })

        calibration = (
            matched["surface_acceptable"].to_numpy(dtype=bool)
            & (matched["spatial_split"].astype(str).to_numpy() == "calibration")
            & np.isfinite(matched["raw_difference_plane_m"])
        )
        test = (
            matched["surface_acceptable"].to_numpy(dtype=bool)
            & (matched["spatial_split"].astype(str).to_numpy() == "independent_test")
            & np.isfinite(matched["raw_difference_plane_m"])
        )
        if key == "snr_ge_3":
            calibration &= snr >= 3.0
            test &= snr >= 3.0
        elif key == "snr_ge_5":
            calibration &= snr >= 5.0
            test &= snr >= 5.0
        source_holdout = holdout_source[holdout_source.snr_stratum == key]
        if len(source_holdout) != 1:
            raise RuntimeError(f"Expected one preserved holdout-metric row for {data['granule_id']} / {key}.")
        source_holdout = source_holdout.iloc[0]
        metrics = empirical_alignment_metrics(
            matched["raw_difference_plane_m"].to_numpy(dtype=np.float64),
            calibration,
            test,
            test_denominator=int(source_holdout.test_denominator_n),
        )
        if int(metrics["calibration_n"]) != int(source_holdout.calibration_n) or int(metrics["test_n"]) != int(source_holdout.test_n):
            raise RuntimeError(f"Spatial holdout counts changed for {data['granule_id']} / {key}.")
        for metric in ("empirical_vertical_offset_m", "bias_m", "median_m", "rmse_m", "mae_m", "nmad_m", "absolute_p95_m"):
            if not np.isclose(float(source_holdout[metric]), float(metrics[metric]), rtol=0.0, atol=1e-8):
                raise RuntimeError(f"Holdout {metric} no longer matches the preserved output for {data['granule_id']} / {key}.")
        residual = matched.loc[test, "raw_difference_plane_m"].to_numpy(dtype=np.float64) - metrics["empirical_vertical_offset_m"]
        holdout_values[label] = residual
        holdout_rows.append({
            **metrics,
            "granule_id": data["granule_id"], "snr_stratum": key,
            "calibration_site_id": source_holdout.calibration_site_id,
            "independent_test_site_id": source_holdout.independent_test_site_id,
            "split_rule": source_holdout.split_rule,
            "surface_acceptance": source_holdout.surface_acceptance,
        })
        if key == "all_valid_refh":
            test_rows = matched.loc[test].copy()
            test_rows["holdout_residual_all_valid_m"] = residual

    if test_rows is None:
        raise RuntimeError("Independent-test rows are missing.")
    return {
        "raw_values": raw_values,
        "holdout_values": holdout_values,
        "raw_metrics": pd.DataFrame(raw_rows),
        "holdout_metrics": pd.DataFrame(holdout_rows),
        "test_rows": test_rows,
    }


def independent_ground_interpolation_check(
    ground_xy: np.ndarray,
    ground_z: np.ndarray,
    validation_xyz: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """Score a small, held-out sample of actual 3DEP class-2 points."""

    import pandas as pd

    xyz = np.asarray(validation_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("validation_xyz must have shape (n, 3).")
    if not len(xyz):
        raise ValueError("The LAZ produced no deterministic held-out class-2 points.")
    surface = interpolate_ground_surface(xyz[:, :2], ground_xy, ground_z, chunk_size=5_000)
    rows = pd.DataFrame({
        "reference_x_m": xyz[:, 0], "reference_y_m": xyz[:, 1], "held_out_ground_z_m": xyz[:, 2],
        **surface,
    })
    for name, column in (
        ("plane", "reference_height_plane_m"),
        ("local_median", "reference_height_median_m"),
    ):
        rows[f"{name}_prediction_error_m"] = rows[column] - rows["held_out_ground_z_m"]
    plane = rows["plane_prediction_error_m"].to_numpy(dtype=np.float64)
    plane = plane[np.isfinite(plane)]
    median = rows["local_median_prediction_error_m"].to_numpy(dtype=np.float64)
    median = median[np.isfinite(median)]
    summary = {
        "validation_n": int(len(rows)),
        "plane_supported_n": int(np.isfinite(rows.reference_height_plane_m).sum()),
        "plane_acceptable_n": int(rows.surface_acceptable.sum()),
        "unsupported_n": int((rows.surface_status == "insufficient_neighbors").sum()),
        "plane_error": elevation_metrics(plane),
        "local_median_error": elevation_metrics(median),
        "interpretation": "Held-out 3DEP class-2 point interpolation check only; not CASALS accuracy or vertical-frame validation.",
    }
    return rows, summary


def select_elevation_cases(
    matched: Any,
    *,
    calibration_offset_m: float,
    gross_error_threshold_m: float = 2.0,
) -> tuple[Any, list[dict[str, Any]]]:
    """Choose reproducible independent-test waveform cases without residual tuning."""

    import pandas as pd

    test = matched[
        (matched.spatial_split.astype(str) == "independent_test")
        & matched.surface_acceptable.astype(bool)
        & np.isfinite(matched.raw_difference_plane_m)
    ].copy()
    test["all_valid_holdout_residual_m"] = test.raw_difference_plane_m - float(calibration_offset_m)
    cases: list[dict[str, Any]] = []
    specifications = [
        ("High SNR, low error", (test.refh_snr >= 5.0) & (test.all_valid_holdout_residual_m.abs() <= 1.0),
         "Among independent-test pulses with SNR >= 5 and |all-valid aligned residual| <= 1 m, nearest zero residual; tie by pulse index."),
        ("Moderate SNR, low error", (test.refh_snr >= 3.0) & (test.refh_snr < 5.0) & (test.all_valid_holdout_residual_m.abs() <= 1.0),
         "Among independent-test pulses with 3 <= SNR < 5 and |all-valid aligned residual| <= 1 m, nearest zero residual; tie by pulse index."),
    ]
    for role, mask, rule in specifications:
        pool = test.loc[mask].sort_values(["all_valid_holdout_residual_m", "pulse_index"], key=lambda s: s.abs() if s.name == "all_valid_holdout_residual_m" else s)
        if len(pool):
            cases.append({"selection_role": role, "selection_status": "selected", "selection_rule": rule, "row": pool.iloc[0]})
        else:
            cases.append({"selection_role": role, "selection_status": "unavailable", "selection_rule": rule, "row": None})

    gross = test[test.all_valid_holdout_residual_m.abs() > gross_error_threshold_m].copy()
    gross = gross.assign(abs_residual=gross.all_valid_holdout_residual_m.abs()).sort_values(
        ["abs_residual", "pulse_index"], ascending=[False, True]
    )
    chosen = []
    for case_number in (1, 2):
        rule = (
            f"Gross-error case {case_number}: independent-test |all-valid aligned residual| > "
            f"{gross_error_threshold_m:g} m; choose largest absolute residual, with the second case preferring a distinct SNR band or >=25 m separation."
        )
        candidate = None
        for _, row in gross.iterrows():
            if int(row.pulse_index) in chosen:
                continue
            if case_number == 2 and chosen:
                first = gross[gross.pulse_index == chosen[0]].iloc[0]
                distinct_band = (float(first.refh_snr) >= 3) != (float(row.refh_snr) >= 3)
                separation = float(np.hypot(first.easting_m_epsg6347 - row.easting_m_epsg6347,
                                            first.northing_m_epsg6347 - row.northing_m_epsg6347))
                if not distinct_band and separation < 25.0:
                    continue
            candidate = row
            break
        if candidate is None and case_number == 2:
            remaining = gross[~gross.pulse_index.isin(chosen)]
            if len(remaining):
                candidate = remaining.iloc[0]
        if candidate is not None:
            chosen.append(int(candidate.pulse_index))
            cases.append({"selection_role": f"Gross error {case_number}", "selection_status": "selected",
                          "selection_rule": rule, "row": candidate})
        else:
            cases.append({"selection_role": f"Gross error {case_number}", "selection_status": "unavailable",
                          "selection_rule": rule, "row": None})
    return test, cases


def build_ground_profile(matched: Any, *, calibration_offset_m: float, minimum_pulses: int = 25) -> tuple[Any, int]:
    """Choose a supported test track with the most SNR >= 3 observations.

    Ties use measured path length, then row count. Every supported observation
    on the selected track remains in the profile, including low-SNR outliers.
    """

    frame = matched[
        (matched.spatial_split.astype(str) == "independent_test")
        & matched.surface_acceptable.astype(bool)
        & np.isfinite(matched.reference_height_local_plane_m)
    ].copy()
    options = []
    for track, group in frame.groupby("track_num", sort=True):
        ordered = group.sort_values(["sweep_num", "pulse_index"]).copy()
        if len(ordered) < minimum_pulses:
            continue
        dx = np.diff(ordered.easting_m_epsg6347.to_numpy(dtype=np.float64))
        dy = np.diff(ordered.northing_m_epsg6347.to_numpy(dtype=np.float64))
        distance = np.concatenate(([0.0], np.cumsum(np.hypot(dx, dy))))
        good_snr_n = int((ordered.refh_snr >= 3.0).sum())
        options.append((good_snr_n, float(distance[-1]), len(ordered), int(track), ordered, distance))
    if not options:
        raise ValueError(f"No independent-test track has at least {minimum_pulses} supported pulses.")
    _, _, _, track, profile, distance = max(
        options, key=lambda item: (item[0], item[1], item[2], -item[3]),
    )
    profile["distance_m"] = distance
    profile["casals_refh_m"] = profile["casals_refh_m"].to_numpy(dtype=np.float64)
    profile["empirically_aligned_refh_m"] = profile["casals_refh_m"] - float(calibration_offset_m)
    return profile, track


def interpolate_ground_surface(
    query_xy: np.ndarray,
    ground_xy: np.ndarray,
    ground_z: np.ndarray,
    *,
    search_radius_m: float = 3.0,
    min_neighbors: int = 6,
    max_neighbors: int = 32,
    max_slope_deg: float = 5.0,
    max_plane_residual_nmad_m: float = 0.20,
    chunk_size: int = 20_000,
    ground_tree: cKDTree | None = None,
) -> dict[str, np.ndarray]:
    """Fit robust local planes without extrapolating beyond ground-point support.

    The independent local-median surface uses the same retained class-2
    neighbors. ``plane_fit_valid`` records geometrically supported fits;
    ``surface_acceptable`` also applies the slope and roughness limits.
    """

    q = np.asarray(query_xy, dtype=np.float64)
    xy = np.asarray(ground_xy, dtype=np.float64)
    z = np.asarray(ground_z, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 2 or not np.all(np.isfinite(q)):
        raise ValueError("query_xy must be a finite (n, 2) array.")
    if xy.ndim != 2 or xy.shape[1] != 2 or z.ndim != 1 or len(xy) != len(z):
        raise ValueError("ground_xy and ground_z must have shapes (m, 2) and (m,).")
    if not len(xy) or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(z)):
        raise ValueError("Ground points must be non-empty and finite.")
    if search_radius_m <= 0 or min_neighbors < 3 or max_neighbors < min_neighbors:
        raise ValueError("Invalid radius or neighborhood size.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    tree = ground_tree if ground_tree is not None else cKDTree(xy)
    n = len(q)
    out: dict[str, np.ndarray] = {
        "reference_height_plane_m": np.full(n, np.nan),
        "reference_height_median_m": np.full(n, np.nan),
        "neighbor_count": np.zeros(n, dtype=np.int16),
        "robust_neighbor_count": np.zeros(n, dtype=np.int16),
        "nearest_ground_distance_m": np.full(n, np.nan),
        "local_slope_deg": np.full(n, np.nan),
        "plane_residual_nmad_m": np.full(n, np.nan),
        "ground_roughness_nmad_m": np.full(n, np.nan),
        "plane_condition_number": np.full(n, np.nan),
    }
    out["plane_fit_valid"] = np.zeros(n, dtype=bool)
    out["surface_acceptable"] = np.zeros(n, dtype=bool)
    out["surface_status"] = np.full(n, "insufficient_neighbors", dtype="U32")

    k = min(max_neighbors, len(xy))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        qpart = q[start:stop]
        distances, indices = tree.query(
            qpart, k=k, distance_upper_bound=search_radius_m, workers=1
        )
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        valid = np.isfinite(distances) & (indices < len(xy))
        counts = valid.sum(axis=1)
        out["neighbor_count"][start:stop] = counts.astype(np.int16)
        nearest = np.where(valid.any(axis=1), np.min(np.where(valid, distances, np.inf), axis=1), np.nan)
        out["nearest_ground_distance_m"][start:stop] = nearest
        rows = np.flatnonzero(counts >= min_neighbors)
        if not rows.size:
            continue

        local_q = qpart[rows]
        local_valid = valid[rows]
        local_indices = np.minimum(indices[rows], len(xy) - 1)
        local_xy = xy[local_indices]
        local_z = z[local_indices]
        dx = local_xy[:, :, 0] - local_q[:, None, 0]
        dy = local_xy[:, :, 1] - local_q[:, None, 1]
        design = np.stack((dx, dy, np.ones_like(dx)), axis=2)
        z_masked = np.where(local_valid, local_z, np.nan)
        z_base = np.nanmedian(z_masked, axis=1)
        z_rel = np.where(local_valid, local_z - z_base[:, None], 0.0)

        def solve(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            weight = mask.astype(np.float64)
            gram = np.einsum("nki,nkj->nij", design, design * weight[:, :, None])
            rhs = np.einsum("nki,nk->ni", design, z_rel * weight)
            eigen = np.linalg.eigvalsh(gram)
            condition = np.divide(
                eigen[:, -1], eigen[:, 0], out=np.full(len(eigen), np.inf), where=eigen[:, 0] > 1e-12
            )
            enough = (mask.sum(axis=1) >= min_neighbors) & np.isfinite(condition) & (condition <= 1e8)
            beta = np.full((len(mask), 3), np.nan)
            if enough.any():
                beta[enough] = np.linalg.solve(gram[enough], rhs[enough, :, None])[:, :, 0]
            return beta, condition, enough

        beta, condition, fit_ok = solve(local_valid)
        predicted = np.einsum("nki,ni->nk", design, np.nan_to_num(beta))
        residual = z_rel - predicted
        residual = np.where(local_valid & fit_ok[:, None], residual, np.nan)
        residual_center = np.full(len(rows), np.nan)
        initial_nmad = np.full(len(rows), np.nan)
        if fit_ok.any():
            residual_center[fit_ok] = np.median(residual[fit_ok], axis=1)
            initial_nmad[fit_ok] = 1.4826 * np.median(
                np.abs(residual[fit_ok] - residual_center[fit_ok, None]), axis=1
            )
        cutoff = np.maximum(3.0 * np.nan_to_num(initial_nmad, nan=0.0), 0.02)
        inliers = local_valid & (np.abs(residual - residual_center[:, None]) <= cutoff[:, None])
        beta2, condition2, fit_ok2 = solve(inliers)
        final_ok = fit_ok & fit_ok2
        if not final_ok.any():
            out["surface_status"][start + rows] = "ill_conditioned_plane"
            continue

        pred2 = np.einsum("nki,ni->nk", design, np.nan_to_num(beta2))
        resid2 = z_rel - pred2
        resid2 = np.where(inliers & final_ok[:, None], resid2, np.nan)
        resid_center2 = np.full(len(rows), np.nan)
        residual_nmad = np.full(len(rows), np.nan)
        roughness = np.full(len(rows), np.nan)
        median_height = np.full(len(rows), np.nan)
        if final_ok.any():
            resid_center2[final_ok] = np.nanmedian(resid2[final_ok], axis=1)
            residual_nmad[final_ok] = 1.4826 * np.nanmedian(
                np.abs(resid2[final_ok] - resid_center2[final_ok, None]), axis=1
            )
            roughness[final_ok] = 1.4826 * np.nanmedian(
                np.abs(z_masked[final_ok] - np.nanmedian(z_masked[final_ok], axis=1)[:, None]), axis=1
            )
            median_height[final_ok] = np.nanmedian(
                np.where(inliers[final_ok], local_z[final_ok], np.nan), axis=1
            )
        slope = np.degrees(np.arctan(np.hypot(beta2[:, 0], beta2[:, 1])))
        plane_height = z_base + beta2[:, 2] + resid_center2

        absolute_rows = start + rows
        valid_rows = absolute_rows[final_ok]
        out["reference_height_plane_m"][valid_rows] = plane_height[final_ok]
        out["reference_height_median_m"][valid_rows] = median_height[final_ok]
        out["robust_neighbor_count"][valid_rows] = inliers.sum(axis=1)[final_ok].astype(np.int16)
        out["local_slope_deg"][valid_rows] = slope[final_ok]
        out["plane_residual_nmad_m"][valid_rows] = residual_nmad[final_ok]
        out["ground_roughness_nmad_m"][valid_rows] = roughness[final_ok]
        out["plane_condition_number"][valid_rows] = condition2[final_ok]
        out["plane_fit_valid"][valid_rows] = True
        acceptable = (
            (slope[final_ok] <= max_slope_deg)
            & (residual_nmad[final_ok] <= max_plane_residual_nmad_m)
        )
        out["surface_acceptable"][valid_rows] = acceptable
        out["surface_status"][valid_rows] = np.where(
            slope[final_ok] > max_slope_deg,
            "steep_local_surface",
            np.where(residual_nmad[final_ok] > max_plane_residual_nmad_m, "rough_local_surface", "matched"),
        )
        failed_rows = absolute_rows[~final_ok]
        out["surface_status"][failed_rows] = "ill_conditioned_plane"

    return out


def elevation_metrics(values: np.ndarray) -> dict[str, float | int | None]:
    """Summarize signed elevation differences; these metrics make no accuracy claim."""

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not x.size:
        return {
            "n": 0,
            "bias_m": None,
            "median_m": None,
            "std_m": None,
            "rmse_m": None,
            "mae_m": None,
            "nmad_m": None,
            "p05_m": None,
            "p50_m": None,
            "p95_m": None,
            "absolute_p90_m": None,
            "absolute_p95_m": None,
        }
    median = float(np.median(x))
    absolute = np.abs(x)
    return {
        "n": int(x.size),
        "bias_m": float(np.mean(x)),
        "median_m": median,
        "std_m": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "rmse_m": float(np.sqrt(np.mean(np.square(x)))),
        "mae_m": float(np.mean(absolute)),
        "nmad_m": float(1.4826 * np.median(np.abs(x - median))),
        "p05_m": float(np.percentile(x, 5)),
        "p50_m": median,
        "p95_m": float(np.percentile(x, 95)),
        "absolute_p90_m": float(np.percentile(absolute, 90)),
        "absolute_p95_m": float(np.percentile(absolute, 95)),
    }


def empirical_alignment_metrics(
    differences_m: np.ndarray,
    calibration_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    test_denominator: int | None = None,
) -> dict[str, Any]:
    """Estimate a median offset on calibration points and score disjoint test points."""

    d = np.asarray(differences_m, dtype=np.float64).reshape(-1)
    calibration = np.array(calibration_mask, dtype=bool, copy=True).reshape(-1)
    test = np.array(test_mask, dtype=bool, copy=True).reshape(-1)
    if d.shape != calibration.shape or d.shape != test.shape:
        raise ValueError("Differences and split masks must have the same length.")
    if np.any(calibration & test):
        raise ValueError("Calibration and test samples must be disjoint.")
    calibration &= np.isfinite(d)
    test &= np.isfinite(d)
    if not calibration.any() or not test.any():
        raise ValueError("Both calibration and independent test samples must be non-empty.")
    offset = float(np.median(d[calibration]))
    residual = d[test] - offset
    summary = elevation_metrics(residual)
    denom = int(test_denominator) if test_denominator is not None else int(test.sum())
    summary.update({
        "empirical_vertical_offset_m": offset,
        "calibration_n": int(calibration.sum()),
        "test_n": int(test.sum()),
        "test_denominator_n": denom,
        "test_coverage": float(test.sum() / denom) if denom else None,
        "evaluation_semantics": "spatial holdout agreement after empirical median alignment; not strict absolute accuracy",
    })
    return summary


def spatial_site_masks(
    site_ids: np.ndarray,
    calibration_site_ids: set[str] | tuple[str, ...] | list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Split by predeclared spatial site identity, independently of residuals."""

    sites = np.asarray(site_ids, dtype=object).reshape(-1)
    if any(value is None or str(value) == "" for value in sites):
        raise ValueError("Every evaluation point must have a spatial site identity.")
    calibration_sites = {str(value) for value in calibration_site_ids}
    if not calibration_sites:
        raise ValueError("At least one calibration site must be declared.")
    calibration = np.isin(sites.astype(str), list(calibration_sites))
    test = ~calibration
    if not calibration.any() or not test.any():
        raise ValueError("Calibration and test sites must both be represented.")
    return calibration, test


def snr_strata(snr: np.ndarray) -> dict[str, np.ndarray]:
    """Return the complete valid population and nested SNR sensitivity masks."""

    value = np.asarray(snr, dtype=np.float64).reshape(-1)
    valid = np.isfinite(value)
    return {
        "all_valid_refh": valid,
        "snr_ge_3": valid & (value >= 3.0),
        "snr_ge_5": valid & (value >= 5.0),
    }


def ground_target_c8_support_gate(
    common_sample: Any,
    calibration_audit: dict[str, Any],
    *,
    minimum_pulses: int = 30,
    minimum_spatial_blocks: int = 3,
    minimum_snr: float = 5.0,
) -> Any:
    """Audit archived road-target C8 fits using common-support high-SNR pulses."""

    import pandas as pd

    required = {"granule_id", "research_role", "refh_snr", "axis_distance_m"}
    if not required.issubset(common_sample.columns):
        raise ValueError(f"C8 support sample is missing columns: {sorted(required - set(common_sample.columns))}")
    rows = []
    for granule_id, audit in calibration_audit["granules"].items():
        calibration = common_sample.loc[
            (common_sample.granule_id.astype(str) == str(granule_id))
            & common_sample.research_role.eq("calibration_offset")
        ].copy()
        if len(calibration) != int(audit["calibration_common_n"]):
            raise ValueError(f"C8 calibration support count differs from its audit: {granule_id}")
        high_quality = calibration.refh_snr.ge(minimum_snr) & calibration.refh_snr.notna()
        if len(calibration) and calibration.axis_distance_m.nunique() > 1:
            edges = np.linspace(calibration.axis_distance_m.min(), calibration.axis_distance_m.max(), minimum_spatial_blocks + 1)
            block_ids = np.digitize(calibration.axis_distance_m.to_numpy(), edges[1:-1])
            supported_blocks = int(np.unique(block_ids[high_quality.to_numpy()]).size)
        else:
            supported_blocks = int(high_quality.any())
        support_passes = int(high_quality.sum()) >= minimum_pulses and supported_blocks >= minimum_spatial_blocks
        rows.append({
            "granule_id": str(granule_id),
            "archived_calibration_common_n": int(len(calibration)),
            "high_quality_common_n_snr_ge_5": int(high_quality.sum()),
            "spatial_blocks_with_high_quality_support": supported_blocks,
            "minimum_common_pulses": int(minimum_pulses),
            "minimum_spatial_blocks": int(minimum_spatial_blocks),
            "archived_offset_m_failure_diagnostic_only": float(audit["calibration_offset_m"]),
            "status": "SUPPORTED" if support_passes else "INSUFFICIENT_CALIBRATION_SUPPORT",
        })
    return pd.DataFrame(rows)


def load_verified_ground_aware_outputs(project_root: str | Path) -> dict[str, Any]:
    """Verify source and artifact identities, then load saved ground-aware tables read-only."""

    import json

    import numpy as np
    import pandas as pd

    from research.l1b_validation import sha256_file

    root = Path(project_root)
    visual = root / "outputs/notebooks/10_3dep_elevation_validation/visual_revision"
    datum = root / "outputs/notebooks/10_3dep_elevation_validation/datum_comparison"
    output = root / "outputs/notebooks/10_3dep_elevation_validation/ground_aware"
    source_manifest = json.loads((visual / "input_manifest.json").read_text(encoding="utf-8"))
    ground_provenance = json.loads((output / "ground_aware_provenance.json").read_text(encoding="utf-8"))
    landcover_manifest = json.loads((
        root / "outputs/notebooks/11_landcover_lidar_response/visual_revision/input_manifest.json"
    ).read_text(encoding="utf-8"))

    for stem, expected in source_manifest["inputs"].items():
        source_h5 = root / expected["h5_path"]
        source_laz = root / expected["laz_path"]
        h5_sha = sha256_file(source_h5)
        laz_sha = sha256_file(source_laz)
        ground_input = ground_provenance["inputs"][stem]
        landcover_input = landcover_manifest["h5_inputs"][stem]
        if (source_h5.stat().st_size != expected["h5_size_bytes"]
                or h5_sha != expected["h5_sha256"]
                or h5_sha != ground_input["h5_sha256"]
                or h5_sha != landcover_input["sha256"]):
            raise RuntimeError(f"H5 identity differs from saved analyses: {stem}")
        if (source_laz.stat().st_size != expected["laz_size_bytes"]
                or laz_sha != expected["laz_sha256"]
                or laz_sha != ground_input["laz_sha256"]):
            raise RuntimeError(f"3DEP LAZ identity differs from saved analysis: {stem}")

    artifact_manifest = pd.read_csv(output / "ground_aware_artifact_manifest.csv")
    for artifact in artifact_manifest.itertuples(index=False):
        artifact_path = output / artifact.path
        if (not artifact_path.is_file()
                or artifact_path.stat().st_size != artifact.size_bytes
                or sha256_file(artifact_path) != artifact.sha256):
            raise RuntimeError(f"Saved ground-aware artifact failed its hash check: {artifact.path}")

    context = pd.read_csv(output / "ground_context_population_summary.csv")
    interpolation_samples = pd.read_csv(output / "independent_class2_interpolation_check.csv")
    interpolation_rows = []
    for granule_id, sample in interpolation_samples.groupby("granule_id", sort=True):
        accepted = sample.loc[sample.surface_acceptable]
        interpolation_rows.append({
            "granule_id": granule_id,
            "held-out class-2 points": len(sample),
            "plane-fit valid": int(sample.plane_fit_valid.sum()),
            "quality accepted": len(accepted),
            "plane RMSE (m)": float(np.sqrt(np.mean(accepted.plane_prediction_error_m ** 2))) if len(accepted) else np.nan,
            "local-median RMSE (m)": float(np.sqrt(np.mean(accepted.median_prediction_error_m ** 2))) if len(accepted) else np.nan,
        })

    calibration_audit = json.loads((output / "ground_empirical_alignment_audit.json").read_text(encoding="utf-8"))
    c8_gate = ground_target_c8_support_gate(
        pd.read_parquet(output / "ground_c0_c7_common_sample.parquet"), calibration_audit,
    )
    c8_gate_path = visual / "ground_c8_calibration_support_gate.csv"
    c8_gate.to_csv(c8_gate_path, index=False)
    return {
        "summary": json.loads((output / "ground_aware_summary.json").read_text(encoding="utf-8")),
        "split_manifest": pd.read_csv(output / "ground_spatial_split_manifest.csv"),
        "candidate_metrics": pd.read_csv(output / "ground_conditional_candidate_metrics.csv"),
        "test_metrics": pd.read_csv(output / "ground_conditional_test_metrics.csv"),
        "descriptive_metrics": pd.read_csv(output / "descriptive_reference_return_to_ground_metrics.csv"),
        "context_population_summary": context,
        "interpolation_summary": pd.DataFrame(interpolation_rows),
        "c8_support_gate": c8_gate,
        "artifact_count": int(len(artifact_manifest)),
        "output_root": output,
        "visual_revision_root": visual,
        "datum_output_root": datum,
        "selected_configuration": json.loads((datum / "selected_vertical_configuration.json").read_text(encoding="utf-8")),
    }
