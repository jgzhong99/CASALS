"""Conditional CASALS source-frame hypotheses; no residual-derived corrections."""
import time
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.enums import TransformDirection
from pyproj.exceptions import CRSError, ProjError
from pyproj.transformer import TransformerGroup

from research.reference import vertical_datum_comparison as datum
from research.elevation_comparison import interpolate_ground_surface
from research.reference.refh_correction_audit import residual_metrics

TARGET_EPOCH = 2010.0
TARGET_EPSG = 6319
# EPSG geocentric CRS paired with each supported explicit source realization.
# WGS84(G2296) is routed by the EPSG database through ITRF2020, so the
# independent XYZ checkpoint remains deliberately limited to ITRF2014.
GEOCENTRIC_EPSG = {7912: 7789, 9989: 9988}
STRATA = {"all_valid_refh": None, "snr_ge_3": 3.0, "snr_ge_5": 5.0}


class FrameOperationUnavailable(RuntimeError):
    """The required installed EPSG frame operation cannot be selected."""

# H0 is the unchanged physical-only C7 result. H4 deliberately changes only the
# coordinate time supplied to PROJ; it is a parameter-time sensitivity, not propagation.
HYPOTHESES = {
    "H1_WGS84_G2296_obs_epoch": {
        "source_epsg": 10605, "evaluation": "observation", "coordinate_time": "observation",
        "kind": "conditional_frame_hypothesis",
    },
    "H2_ITRF2014_obs_epoch": {
        "source_epsg": 7912, "evaluation": "observation", "coordinate_time": "observation",
        "kind": "conditional_frame_hypothesis",
    },
    "H3_ITRF2020_obs_epoch": {
        "source_epsg": 9989, "evaluation": "observation", "coordinate_time": "observation",
        "kind": "conditional_frame_hypothesis",
    },
    "H4_ITRF2014_parameter_time_sensitivity_2010": {
        "source_epsg": 7912, "evaluation": TARGET_EPOCH, "coordinate_time": TARGET_EPOCH,
        "kind": "parameter_time_sensitivity_not_epoch_propagation",
    },
}


def observation_epoch(granule_id: str) -> float:
    """Decimal year of the granule timestamp; this is not proof of coordinate epoch."""
    t = datetime.strptime(granule_id[11:26], "%Y%m%dT%H%M%S")
    y0, y1 = datetime(t.year, 1, 1), datetime(t.year + 1, 1, 1)
    return t.year + (t - y0) / (y1 - y0)


def crs_audit(epsg: int) -> dict:
    """Read the installed EPSG CRS definition without substituting another CRS."""
    crs = CRS.from_epsg(epsg)
    datum_json = crs.datum.to_json_dict()
    ellipsoid = crs.ellipsoid
    return {
        "epsg": int(epsg),
        "name": crs.name,
        "dimension": len(crs.axis_info),
        "axis": [{"name": a.name, "direction": a.direction, "unit": a.unit_name} for a in crs.axis_info],
        "ellipsoid": ellipsoid.name,
        "semi_major_axis_m": float(ellipsoid.semi_major_metre),
        "inverse_flattening": float(ellipsoid.inverse_flattening),
        "reference_frame_name": crs.datum.name,
        "datum_type": datum_json.get("type"),
        "is_dynamic": datum_json.get("type") == "DynamicGeodeticReferenceFrame",
        "frame_reference_epoch": datum_json.get("frame_reference_epoch"),
        "crs_projjson": crs.to_json_dict(),
    }


def frame_transformer(source_epsg: int, longitude_deg, latitude_deg):
    """Return only the best available explicit 3D non-ballpark EPSG operation."""
    source = CRS.from_epsg(source_epsg)
    target = CRS.from_epsg(TARGET_EPSG)
    if len(source.axis_info) != 3 or len(target.axis_info) != 3:
        raise RuntimeError(f"EPSG:{source_epsg}->EPSG:{TARGET_EPSG} requires explicit 3D CRSs")
    lon = np.asarray(longitude_deg, dtype=float).reshape(-1)
    lat = np.asarray(latitude_deg, dtype=float).reshape(-1)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not valid.any():
        raise ValueError("No finite coordinates are available to define an operation area")
    aoi = AreaOfInterest(float(lon[valid].min()), float(lat[valid].min()),
                         float(lon[valid].max()), float(lat[valid].max()))
    group = TransformerGroup(source, target, always_xy=True, area_of_interest=aoi,
                             allow_ballpark=False)
    if not group.best_available:
        unavailable = [op.name for op in group.unavailable_operations]
        raise FrameOperationUnavailable(f"Best EPSG operation is unavailable; no fallback used: {unavailable}")
    if not group.transformers:
        raise FrameOperationUnavailable(f"No non-ballpark EPSG:{source_epsg}->EPSG:{TARGET_EPSG} operation is available")
    transformer = group.transformers[0]
    if getattr(transformer, "is_ballpark", False) or "proj=noop" in transformer.definition.lower():
        raise FrameOperationUnavailable(f"EPSG:{source_epsg}->EPSG:{TARGET_EPSG} resolved to a null or ballpark operation")
    return transformer


def _area_contains(area, longitude_deg, latitude_deg) -> bool:
    if area is None:
        return False
    west, south, east, north = area.bounds
    lon = np.asarray(longitude_deg, dtype=float)
    lat = np.asarray(latitude_deg, dtype=float)
    lon_ok = ((lon >= west) | (lon <= east)) if west > east else ((lon >= west) & (lon <= east))
    return bool((lon_ok & (lat >= south) & (lat <= north)).all())


def operation_audit(transformer, source_epsg: int, longitude_deg, latitude_deg) -> dict:
    """Record the selected database operation, complete Helmert values and area."""
    operation = transformer.to_json_dict()
    steps = operation.get("steps", [])
    helmerts = [step for step in steps if step.get("type") == "Transformation"]
    time_dependent = [step for step in helmerts
                      if "time-dependent" in step.get("method", {}).get("name", "").lower()]
    if not helmerts or len(time_dependent) != 1:
        raise RuntimeError(
            f"Expected an explicit operation chain with one time-dependent frame step, "
            f"found {len(helmerts)} frame steps and {len(time_dependent)} time-dependent steps"
        )
    helmert = time_dependent[0]
    method_name = str(helmert.get("method", {}).get("name", ""))
    if "coordinate frame" in method_name.lower():
        rotation_convention = "coordinate_frame"
    elif "position vector" in method_name.lower():
        rotation_convention = "position_vector"
    else:
        raise RuntimeError(f"Unrecognized Helmert rotation convention: {method_name!r}")
    parameter_names = {row.get("name") for row in helmert.get("parameters", [])}
    required = {
        "X-axis translation", "Y-axis translation", "Z-axis translation",
        "X-axis rotation", "Y-axis rotation", "Z-axis rotation", "Scale difference",
        "Rate of change of X-axis translation", "Rate of change of Y-axis translation",
        "Rate of change of Z-axis translation", "Rate of change of X-axis rotation",
        "Rate of change of Y-axis rotation", "Rate of change of Z-axis rotation",
        "Rate of change of Scale difference", "Parameter reference epoch",
    }
    if not required <= parameter_names:
        raise RuntimeError(f"EPSG operation is not the required 14-parameter time-dependent Helmert: {helmert}")
    parameter_epoch = next(row["value"] for row in helmert["parameters"]
                           if row["name"] == "Parameter reference epoch")
    area = transformer.area_of_use
    return {
        "source_epsg": int(source_epsg), "target_epsg": TARGET_EPSG,
        "operation_name": transformer.description,
        "operation_accuracy_metadata_m": transformer.accuracy,
        "operation_definition": transformer.definition,
        "operation_authority": helmert.get("id"),
        "operation_method": helmert.get("method"),
        "rotation_convention": rotation_convention,
        "transformation_steps": [
            {"name": step.get("name"), "authority_id": step.get("id"),
             "method": step.get("method"), "parameters": step.get("parameters")}
            for step in helmerts
        ],
        "helmert_parameters": helmert.get("parameters"),
        "helmert_parameter_reference_epoch": float(parameter_epoch),
        "area_of_use": {"name": area.name, "bounds": list(area.bounds)} if area else None,
        "area_contains_input_extent": _area_contains(area, longitude_deg, latitude_deg),
        "best_available": True,
        "is_ballpark": False,
        "is_noop": False,
        "operation_projjson": operation,
    }


def apply_frame(source_epsg: int, evaluation_epoch: float, longitude_deg, latitude_deg,
                ellipsoidal_height_m, *, coordinate_time: float | None = None,
                transformer=None):
    """Apply frame-only Helmert at evaluation time and retain PROJ's output time."""
    lon = np.asarray(longitude_deg, dtype=float).reshape(-1)
    lat = np.asarray(latitude_deg, dtype=float).reshape(-1)
    height = np.asarray(ellipsoidal_height_m, dtype=float).reshape(-1)
    if len({len(lon), len(lat), len(height)}) != 1 or not len(lon):
        raise ValueError("Longitude, latitude, and height arrays must have one equal non-empty shape")
    if not all(np.isfinite(v).all() for v in (lon, lat, height)):
        raise ValueError("Frame transformation inputs must be finite")
    transformer = transformer or frame_transformer(source_epsg, lon, lat)
    time_in = float(evaluation_epoch if coordinate_time is None else coordinate_time)
    input_time = np.full(len(lon), time_in, dtype=float)
    flon, flat, fh, time_out = transformer.transform(lon, lat, height, input_time, errcheck=True)
    flon, flat, fh, time_out = (np.asarray(v, dtype=float) for v in (flon, flat, fh, time_out))
    if not all(np.isfinite(v).all() for v in (flon, flat, fh, time_out)):
        raise RuntimeError("PROJ returned a non-finite frame result")
    if not np.array_equal(time_out, input_time):
        raise RuntimeError("The frame Helmert changed coordinate time; this path must not imply epoch propagation")
    blon, blat, bh, back_time = transformer.transform(
        flon, flat, fh, time_out, direction=TransformDirection.INVERSE, errcheck=True
    )
    geod = CRS.from_epsg(4326).get_geod()
    _, _, horizontal = geod.inv(lon, lat, np.asarray(blon), np.asarray(blat))
    roundtrip_h = np.asarray(bh, dtype=float) - height
    audit = operation_audit(transformer, source_epsg, lon, lat)
    audit.update({
        "evaluation_epoch": float(evaluation_epoch),
        "coordinate_time_in": time_in,
        "coordinate_time_out": float(time_out[0]),
        "coordinate_time_preserved": True,
        "coordinate_epoch_propagation_applied": False,
        "target_coordinate_epoch_required": TARGET_EPOCH,
        "target_coordinate_epoch_achieved": False,
        "roundtrip_max_abs_height_m": float(np.max(np.abs(roundtrip_h))),
        "roundtrip_max_horizontal_m": float(np.max(np.asarray(horizontal))),
        "roundtrip_time_preserved": bool(np.array_equal(back_time, time_out)),
    })
    if audit["roundtrip_max_abs_height_m"] > 1e-6 or audit["roundtrip_max_horizontal_m"] > 1e-6:
        raise RuntimeError("Frame transformation inverse round trip exceeded 1 micrometre")
    return flon, flat, fh, audit


def xyz_checkpoint(source_epsg: int, longitude_deg: float, latitude_deg: float,
                   ellipsoidal_height_m: float, coordinate_time: float) -> dict:
    """Rebuild one geographic checkpoint through an independent geocentric XYZ path."""
    if source_epsg not in GEOCENTRIC_EPSG:
        raise RuntimeError(f"No explicit geocentric checkpoint CRS is registered for EPSG:{source_epsg}")
    source = CRS.from_epsg(source_epsg)
    target = CRS.from_epsg(TARGET_EPSG)
    source_xyz = CRS.from_epsg(GEOCENTRIC_EPSG[source_epsg])
    target_xyz = CRS.from_epsg(6317)  # NAD83(2011) geocentric, EPSG database counterpart of 6319.
    if not all(axis.unit_name.lower() in {"metre", "meter"}
               for axis in source_xyz.axis_info + target_xyz.axis_info):
        raise RuntimeError("Geocentric checkpoint axes are not all in metres")
    to_xyz = Transformer.from_crs(source, source_xyz, always_xy=True)
    from_xyz = Transformer.from_crs(target_xyz, target, always_xy=True)
    x, y, z = to_xyz.transform(longitude_deg, latitude_deg, ellipsoidal_height_m, errcheck=True)
    group = TransformerGroup(source_xyz, target_xyz, always_xy=True,
        area_of_interest=AreaOfInterest(longitude_deg - 0.01, latitude_deg - 0.01,
                                        longitude_deg + 0.01, latitude_deg + 0.01),
        allow_ballpark=False)
    if not group.best_available or not group.transformers:
        raise RuntimeError("Independent geocentric Helmert operation is unavailable")
    transformed_xyz = group.transformers[0].transform(x, y, z, coordinate_time, errcheck=True)
    out_lon, out_lat, out_h = from_xyz.transform(*transformed_xyz[:3], errcheck=True)
    geod = CRS.from_epsg(4326).get_geod()
    _, _, horizontal = geod.inv(longitude_deg, latitude_deg, out_lon, out_lat)
    return {
        "source_epsg": source_epsg, "target_epsg": TARGET_EPSG,
        "input_geographic_lon_lat_h": [float(longitude_deg), float(latitude_deg), float(ellipsoidal_height_m)],
        "input_geocentric_xyz_m": [float(x), float(y), float(z)],
        "input_coordinate_time": float(coordinate_time),
        "output_geocentric_xyz_m": [float(v) for v in transformed_xyz[:3]],
        "output_coordinate_time": float(transformed_xyz[3]),
        "output_geographic_lon_lat_h": [float(out_lon), float(out_lat), float(out_h)],
        "horizontal_difference_from_input_m": float(horizontal),
        "ellipsoidal_height_difference_from_input_m": float(out_h - ellipsoidal_height_m),
        "geocentric_operation_definition": group.transformers[0].definition,
        "xyz_units": "metre",
    }


def _scenario_metrics(frame: pd.DataFrame, valid: np.ndarray, *, granule_id: str,
                      hypothesis: str, population: str, role_mask: np.ndarray) -> tuple[list[dict], list[dict]]:
    coverage_rows = []
    snr = frame.refh_snr.to_numpy(float)
    for stratum, threshold in STRATA.items():
        eligible = role_mask.copy()
        if threshold is not None:
            eligible &= snr >= threshold
        supported = eligible & valid
        ids = frame.loc[supported, "pulse_index"].to_numpy(np.int64)
        coverage_rows.append({
            "granule_id": granule_id, "hypothesis": hypothesis, "population": population,
            "stratum": stratum, "eligible_n": int(eligible.sum()), "n": int(supported.sum()),
            "coverage": float(supported.sum() / eligible.sum()) if eligible.any() else 0.0,
            "valid_pulse_sha256": _pulse_hash(ids),
        })
    return [], coverage_rows


def _pulse_hash(ids) -> str:
    import hashlib
    return hashlib.sha256(np.sort(np.asarray(ids, dtype="<i8")).tobytes()).hexdigest()


def _h0_operation_record(R: dict, granule_id: str) -> dict:
    prior = R["operations"]["granules"][granule_id]
    frame = prior.get("frame", {})
    return {
        "granule_id": granule_id, "hypothesis": "H0_noop_C7_baseline", "status": "baseline_preserved",
        "source_epsg": 4979, "source_crs_assumption": "WGS84 ensemble treated as already NAD83(2011), epoch 2010.0",
        "frame_operation": frame.get("definition"),
        "frame_operation_description": frame.get("description"),
        "frame_operation_is_noop": frame.get("is_noop"),
        "coordinate_epoch_propagation_applied": False,
        "target_coordinate_epoch_achieved": False,
        "geoid_operation": prior.get("combined_operation"),
        "geoid_operation_grids": prior.get("selected_grids"),
        "geoid_grid_sha256": datum.GEOID18_GRID_SHA256,
        "reason": "Preserved physical-only C7 baseline; source frame remains unknown.",
    }


def run_frame_hypotheses(R: dict, output: str | Path) -> dict:
    """Requery real class-2 surfaces and compare H0-H4 on both own and common samples."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pop = R["population"].copy()
    method_frames: dict[tuple[str, str], pd.DataFrame] = {}
    method_valid: dict[tuple[str, str], np.ndarray] = {}
    operation_rows: list[dict] = []
    crs_records: list[dict] = []
    shift_rows: list[dict] = []
    decomposition_rows: list[dict] = []
    common_metric_rows: list[dict] = []
    coverage_rows: list[dict] = []
    common_pulse_rows: list[dict] = []
    own_pulse_rows: list[dict] = []

    for spec in datum.GRANULES:
        g = pop.loc[pop.granule_id.eq(spec.granule_id)].sort_values("pulse_index").reset_index(drop=True)
        if g.empty:
            raise ValueError(f"Frozen population has no rows for {spec.granule_id}")
        lon, lat, h = (g[c].to_numpy(float) for c in
                       ("longitude_deg", "latitude_deg", "casals_refh_m"))
        base_H = g.GEOID18_height_m.to_numpy(float)
        base_ref = g.reference_height_m.to_numpy(float)
        base_x = g.c7_easting_m_epsg6347.to_numpy(float)
        base_y = g.c7_northing_m_epsg6347.to_numpy(float)
        base_valid = (g.c7_surface_acceptable.to_numpy(bool) & np.isfinite(base_H)
                      & np.isfinite(base_ref) & np.isfinite(base_x) & np.isfinite(base_y))
        h0 = pd.DataFrame({
            "pulse_index": g.pulse_index.to_numpy(np.int64), "longitude_deg": lon,
            "latitude_deg": lat, "ellipsoidal_height_m": h, "navd88_diagnostic_m": base_H,
            "query_x_m": base_x, "query_y_m": base_y, "reference_height_m": base_ref,
        })
        h0["residual_m"] = h0.navd88_diagnostic_m - h0.reference_height_m
        method_frames[(spec.granule_id, "H0_noop_C7_baseline")] = h0
        method_valid[(spec.granule_id, "H0_noop_C7_baseline")] = base_valid
        operation_rows.append(_h0_operation_record(R, spec.granule_id))

        observation = observation_epoch(spec.granule_id)
        for name, rule in HYPOTHESES.items():
            epsg = rule["source_epsg"]
            crs_row = {"granule_id": spec.granule_id, "hypothesis": name,
                       "source_coordinate_epoch_verified": False,
                       "source_coordinate_epoch_assumption": (
                           "flight timestamp used as coordinate time, conditional only"
                           if rule["coordinate_time"] == "observation" else
                           "unknown; 2010.0 is supplied only to evaluate parameter-time sensitivity"),
                       "observation_epoch_decimal_year": observation,
                       "evaluation_epoch": observation if rule["evaluation"] == "observation" else float(rule["evaluation"]),
                       "coordinate_time_in": observation if rule["coordinate_time"] == "observation" else float(rule["coordinate_time"]),
                       "required_target_coordinate_epoch": TARGET_EPOCH,
                       "coordinate_epoch_propagation_applied": False,
                       "target_coordinate_epoch_achieved": False,
                       "velocity_or_deformation_model": None,
                       "kind": rule["kind"]}
            try:
                crs_row.update(crs_audit(epsg))
                transformer = frame_transformer(epsg, lon, lat)
                evaluation = crs_row["evaluation_epoch"]
                coordinate_time = crs_row["coordinate_time_in"]
                flon, flat, fh, audit = apply_frame(
                    epsg, evaluation, lon, lat, h, coordinate_time=coordinate_time,
                    transformer=transformer)
            except (CRSError, ProjError, FrameOperationUnavailable) as exc:
                # An absent/unsupported installed operation is recorded; no alternate frame is substituted.
                crs_row.update({"operation_status": "unavailable", "unavailable_reason": f"{type(exc).__name__}: {exc}"})
                crs_records.append(crs_row)
                operation_rows.append({"granule_id": spec.granule_id, "hypothesis": name,
                                       "source_epsg": epsg, "status": "unavailable",
                                       "reason": f"{type(exc).__name__}: {exc}"})
                continue
            # Fail hard on operation-definition, grid, interpolation, or numeric errors.
            if not audit["area_contains_input_extent"]:
                raise RuntimeError(f"Selected EPSG:{epsg} operation does not cover {spec.granule_id}")
            if audit["helmert_parameter_reference_epoch"] != 2010.0:
                raise RuntimeError(f"Unexpected Helmert parameter reference epoch for EPSG:{epsg}")
            if name == "H2_ITRF2014_obs_epoch":
                audit["geocentric_xyz_checkpoint"] = xyz_checkpoint(
                    epsg, float(lon[0]), float(lat[0]), float(h[0]), float(coordinate_time))
                checkpoint = audit["geocentric_xyz_checkpoint"]
                geod = CRS.from_epsg(4326).get_geod()
                _, _, checkpoint_distance = geod.inv(
                    float(flon[0]), float(flat[0]), checkpoint["output_geographic_lon_lat_h"][0],
                    checkpoint["output_geographic_lon_lat_h"][1])
                if abs(float(checkpoint_distance)) > 1e-5 or abs(
                        checkpoint["output_geographic_lon_lat_h"][2] - float(fh[0])) > 1e-5:
                    raise RuntimeError("Independent geocentric XYZ checkpoint does not match the 3D route")
            operation_rows.append({**audit, "granule_id": spec.granule_id, "hypothesis": name,
                                   "source_coordinate_epoch_assumption": crs_row["source_coordinate_epoch_assumption"],
                                   "source_coordinate_epoch_verified": False,
                                   "frame_transformation_output_coordinate_time": float(audit["coordinate_time_out"]),
                                   "velocity_or_deformation_model": None,
                                   "status": "conditional_operation_available_coordinate_epoch_not_propagated",
                                   "source_crs": crs_row})
            crs_row.update({
                "operation_status": "available", "operation_name": audit["operation_name"],
                "operation_definition": audit["operation_definition"],
                "helmert_parameter_reference_epoch": audit["helmert_parameter_reference_epoch"],
                "area_of_use": audit["area_of_use"],
                "area_contains_input_extent": audit["area_contains_input_extent"],
                "operation_authority": audit["operation_authority"],
            })
            crs_records.append(crs_row)
            x, y, H, geoid_audit = datum.transform_c7(flon, flat, fh, source_epsg=TARGET_EPSG,
                                                      return_audit=True)
            if geoid_audit["selected_operation_grid_constraint"] != datum.GEOID18_GRID:
                raise RuntimeError("NAD83(2011) vertical operation did not require GEOID18")
            if geoid_audit["selected_grid_sha256"] != datum.GEOID18_GRID_SHA256:
                raise RuntimeError("NAD83(2011) vertical operation did not use the pinned GEOID18 grid")
            query = np.column_stack([x, y])
            xy, z, grid_meta = datum.build_local_class2_query_grid(spec.laz_path, query)
            surface = interpolate_ground_surface(query, xy, z)
            ref = np.asarray(surface["reference_height_plane_m"], dtype=float)
            valid = (np.asarray(surface["surface_acceptable"], dtype=bool) & np.isfinite(ref)
                     & np.isfinite(H) & np.isfinite(x) & np.isfinite(y))
            frame = pd.DataFrame({
                "pulse_index": g.pulse_index.to_numpy(np.int64), "longitude_deg": flon,
                "latitude_deg": flat, "ellipsoidal_height_m": fh,
                "navd88_diagnostic_m": np.asarray(H, dtype=float),
                "query_x_m": np.asarray(x, dtype=float), "query_y_m": np.asarray(y, dtype=float),
                "reference_height_m": ref,
            })
            frame["residual_m"] = frame.navd88_diagnostic_m - frame.reference_height_m
            method_frames[(spec.granule_id, name)] = frame
            method_valid[(spec.granule_id, name)] = valid
            geoid_audit.update({"grid_sha256": datum.GEOID18_GRID_SHA256,
                                "grid_path": str(geoid_audit["selected_grid_path"])})
            operation_rows[-1]["geoid_operation"] = geoid_audit
            operation_rows[-1]["surface_grid_metadata"] = grid_meta

        scenarios = [name for gid, name in method_frames if gid == spec.granule_id]
        populations = sorted(g.population.dropna().unique().tolist())
        role = g.research_role.eq("independent_test").to_numpy()
        for scenario in scenarios:
            valid = method_valid[(spec.granule_id, scenario)]
            for population in populations:
                role_mask = role & g.population.eq(population).to_numpy()
                _, rows = _scenario_metrics(g, valid, granule_id=spec.granule_id,
                                            hypothesis=scenario, population=population,
                                            role_mask=role_mask)
                coverage_rows.extend(rows)
                for row in rows:
                    threshold = STRATA[row["stratum"]]
                    elig = role_mask.copy()
                    if threshold is not None:
                        elig &= g.refh_snr.to_numpy(float) >= threshold
                    ids = g.loc[elig & valid, "pulse_index"].to_numpy(np.int64)
                    own_pulse_rows.extend({"granule_id": spec.granule_id, "hypothesis": scenario,
                                           "population": population, "stratum": row["stratum"],
                                           "pulse_index": int(p)} for p in ids)

        # Common-pulse metrics use the same exact pulse IDs for every available method.
        for population in populations:
            role_mask = role & g.population.eq(population).to_numpy()
            for stratum, threshold in STRATA.items():
                eligible = role_mask.copy()
                if threshold is not None:
                    eligible &= g.refh_snr.to_numpy(float) >= threshold
                scenario_ids = []
                for scenario in scenarios:
                    valid = method_valid[(spec.granule_id, scenario)]
                    scenario_ids.append(set(g.loc[eligible & valid, "pulse_index"].astype(np.int64)))
                common_ids = sorted(set.intersection(*scenario_ids)) if scenario_ids else []
                common_set = set(common_ids)
                common_hash = _pulse_hash(common_ids)
                common_pulse_rows.extend({"granule_id": spec.granule_id, "population": population,
                                          "stratum": stratum, "pulse_index": int(p),
                                          "common_pulse_sha256": common_hash} for p in common_ids)
                common_n = len(common_ids)
                for scenario in scenarios:
                    source = method_frames[(spec.granule_id, scenario)]
                    sample = source.loc[source.pulse_index.isin(common_set)]
                    vals = sample.residual_m.to_numpy(float)
                    common_metric_rows.append({
                        "granule_id": spec.granule_id, "hypothesis": scenario,
                        "population": population, "stratum": stratum,
                        "eligible_n": int(eligible.sum()), "n": common_n,
                        "coverage": float(common_n / eligible.sum()) if eligible.any() else 0.0,
                        "pulse_sha256": common_hash, **residual_metrics(vals),
                    })

        # Decomposition and shift summaries are paired against H0 on identical pulses.
        h0 = method_frames[(spec.granule_id, "H0_noop_C7_baseline")]
        for scenario in scenarios:
            if scenario == "H0_noop_C7_baseline":
                continue
            f = method_frames[(spec.granule_id, scenario)]
            valid = method_valid[(spec.granule_id, "H0_noop_C7_baseline")] & method_valid[(spec.granule_id, scenario)]
            for population in populations:
                role_mask = role & g.population.eq(population).to_numpy()
                indices = np.flatnonzero(valid & role_mask)
                a = h0.iloc[indices].reset_index(drop=True)
                b = f.iloc[indices].reset_index(drop=True)
                delta_h = b.ellipsoidal_height_m.to_numpy(float) - a.ellipsoidal_height_m.to_numpy(float)
                delta_H = b.navd88_diagnostic_m.to_numpy(float) - a.navd88_diagnostic_m.to_numpy(float)
                delta_reference = b.reference_height_m.to_numpy(float) - a.reference_height_m.to_numpy(float)
                delta_residual = b.residual_m.to_numpy(float) - a.residual_m.to_numpy(float)
                if not np.allclose(delta_residual, delta_H - delta_reference, rtol=0, atol=1e-10):
                    raise AssertionError("Residual-change decomposition does not close")
                for j, row in enumerate(a.itertuples()):
                    decomposition_rows.append({
                        "granule_id": spec.granule_id, "population": population,
                        "hypothesis": scenario, "pulse_index": int(row.pulse_index),
                        "delta_h_ellipsoidal_m": float(delta_h[j]),
                        "delta_H_NAVD88_diagnostic_m": float(delta_H[j]),
                        "delta_reference_height_m": float(delta_reference[j]),
                        "delta_residual_m": float(delta_residual[j]),
                    })
                shift = np.hypot(b.query_x_m.to_numpy(float) - a.query_x_m.to_numpy(float),
                                 b.query_y_m.to_numpy(float) - a.query_y_m.to_numpy(float))
                shift_rows.append({
                    "granule_id": spec.granule_id, "hypothesis": scenario,
                    "population": population, "n_paired": len(a),
                    "pulse_sha256": _pulse_hash(a.pulse_index),
                    "median_delta_h_ellipsoidal_m": float(np.median(delta_h)) if len(a) else np.nan,
                    "median_delta_H_NAVD88_diagnostic_m": float(np.median(delta_H)) if len(a) else np.nan,
                    "median_horizontal_query_displacement_m": float(np.median(shift)) if len(a) else np.nan,
                    "median_delta_reference_height_m": float(np.median(delta_reference)) if len(a) else np.nan,
                    "median_delta_residual_m": float(np.median(delta_residual)) if len(a) else np.nan,
                })

    pulse_frames = []
    for spec in datum.GRANULES:
        g = pop.loc[pop.granule_id.eq(spec.granule_id)].sort_values("pulse_index").reset_index(drop=True)
        for (gid, name), fr in method_frames.items():
            if gid != spec.granule_id:
                continue
            pulse_frames.append(pd.DataFrame({
                "granule_id": gid, "hypothesis": name, "pulse_index": fr.pulse_index.to_numpy(np.int64),
                "population": g.population.to_numpy(), "research_role": g.research_role.to_numpy(),
                "refh_snr": g.refh_snr.to_numpy(float), "supported": method_valid[(gid, name)],
                "residual": fr.residual_m.to_numpy(float)}))
    pulse_residuals = pd.concat(pulse_frames, ignore_index=True)

    result = {
        "metrics": pd.DataFrame(common_metric_rows),
        "coverage": pd.DataFrame(coverage_rows),
        "shifts": pd.DataFrame(shift_rows),
        "operations": pd.DataFrame(operation_rows),
        "source_crs": pd.DataFrame(crs_records),
        "common_pulses": pd.DataFrame(common_pulse_rows),
        "population_pulses": pd.DataFrame(own_pulse_rows),
        "decomposition": pd.DataFrame(decomposition_rows),
        "available_hypotheses": sorted({name for _, name in method_frames}),
        "pulse_residuals": pulse_residuals,
        "output": output,
    }
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output / f"frame_hypothesis_{key}.csv", index=False)
    (output / "frame_hypothesis_operations.json").write_text(
        json.dumps(datum._json_value(operation_rows), indent=2, allow_nan=False), encoding="utf-8")
    (output / "frame_hypothesis_manifest.json").write_text(json.dumps({
        "target_frame": "NAD83(2011)", "target_coordinate_epoch_required": TARGET_EPOCH,
        "coordinate_epoch_propagation_applied_for_H1_H4": False,
        "H4_is_valid_epoch_propagation": False,
        "grid_name": datum.GEOID18_GRID, "grid_sha256": datum.GEOID18_GRID_SHA256,
        "available_hypotheses": result["available_hypotheses"],
        "common_pulse_sets": result["metrics"][["granule_id", "population", "stratum", "n", "pulse_sha256"]]
            .drop_duplicates().to_dict(orient="records"),
    }, indent=2, allow_nan=False), encoding="utf-8")
    return result


def block_bootstrap_bias(pulses, *, hypothesis, snr_min=5., block=5000, n_boot=2000, seed=20261015):
    """Moving-block bootstrap of mean bias (blocks of consecutive pulse_index; spatial autocorrelation).

    Percentile 95% interval for the mean CASALS-3DEP residual, per granule, independent field test."""
    rng = np.random.default_rng(seed)
    rows = []
    sel = pulses.loc[pulses.hypothesis.eq(hypothesis) & pulses.population.eq("field")
                     & pulses.research_role.eq("independent_test") & pulses.supported
                     & pulses.refh_snr.ge(snr_min)]
    for gid, g in sel.groupby("granule_id", sort=True):
        key = (g.pulse_index.to_numpy() // block).astype(int)
        ids = np.unique(key)
        sums = np.array([g.residual.to_numpy()[key == k].sum() for k in ids])
        cnts = np.array([(key == k).sum() for k in ids])
        draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
        means = sums[draws].sum(1) / cnts[draws].sum(1)
        rows.append(dict(granule_id=gid, hypothesis=hypothesis, n=len(g), n_blocks=len(ids),
                         mean_bias_m=float(g.residual.mean()),
                         ci95_low_m=float(np.percentile(means, 2.5)), ci95_high_m=float(np.percentile(means, 97.5))))
    return pd.DataFrame(rows)


def vdatum_itrf_check(output, *, hypothesis="H2_ITRF2014_obs_epoch"):
    """Independent software check of the frame step: NOAA VDatum (HTDP-based) ITRF2014 -> NAD83(2011)
    @2010.0 -> NAVD88 (GEOID18) on the frozen 20 pulses per granule, versus the PROJ pipeline used here."""
    import json
    import requests
    from research.reference import geodetic_crosscheck as gc
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pts = gc._fixed_sample_points()
    session, calls, rows = requests.Session(), [], []
    for gid, g in pts.groupby("granule_id", sort=True):
        epoch = round(observation_epoch(gid), 4)
        lon, lat, h = (g[c].to_numpy(float) for c in ("longitude_deg", "latitude_deg", "refh_ellipsoidal_m"))
        flon, flat, fh, _ = apply_frame(HYPOTHESES[hypothesis]["source_epsg"], epoch, lon, lat, h, coordinate_time=epoch)
        _, _, H_proj, _ = datum.transform_c7(flon, flat, fh, source_epsg=6319)
        for (_, p), Hp, dh in zip(g.iterrows(), H_proj, fh - h):
            params = gc.build_vdatum_params(p.longitude_deg, p.latitude_deg, p.refh_ellipsoidal_m, p.vdatum_region)
            params.update(s_h_frame="ITRF2014", s_v_frame="ITRF2014", epoch_in=format(epoch, ".4f"), epoch_out="2010.0")
            res = gc._query_service(provider="NOAA VDatum Full API", url=gc.VDATUM_URL, params=params,
                                    parser=lambda payload, e=params: gc.parse_vdatum_response(payload, e),
                                    session=session, cache={}, cache_sources={})
            calls.append(res)
            Hv = float(res["parsed"]["target_height_m"]) if res["status"] == "success" else np.nan
            rows.append(dict(granule_id=gid, pulse_index=int(p.pulse_index), epoch=epoch,
                             H_PROJ_m=float(Hp), H_VDatum_m=Hv, PROJ_minus_VDatum_m=float(Hp) - Hv,
                             status=res["status"]))
            time.sleep(gc.REQUEST_PAUSE_SECONDS)
    df = pd.DataFrame(rows)
    (output / "vdatum_itrf2014_calls.json").write_text(json.dumps(gc._json_safe(dict(calls=calls)), indent=1), encoding="utf-8")
    df.to_csv(output / "vdatum_itrf2014_comparison.csv", index=False)
    ok = df.status.eq("success")
    summ = df.loc[ok].groupby("granule_id").PROJ_minus_VDatum_m.agg(
        successful="count", mean_m="mean", median_m="median", max_abs_m=lambda s: s.abs().max()).reset_index()
    summ["requested"] = df.groupby("granule_id").size().values
    return dict(comparison=df, summary=summ, output=output)
