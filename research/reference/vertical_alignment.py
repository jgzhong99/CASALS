"""Physical height comparison only; 3DEP is never an input to height corrections."""
from pathlib import Path
import hashlib
import json

import h5py
import numpy as np
import pandas as pd
from pyproj import CRS, datadir, network
from pyproj.transformer import TransformerGroup

from research.reference import vertical_datum_comparison as datum
from research.elevation_comparison import interpolate_ground_surface
from research.reference.refh_correction_audit import (
    bilinear_grid, collect_values, load_egm2008_grid, residual_metrics, sha256,
    validate_egm2008_interpolation,
)

HISTORY = datum.OUTPUT_ROOT.parent
METHODS = ("Native", "EGM2008", "GEOID18")
STRATA = {"all_valid_refh": None, "snr_ge_3": 3., "snr_ge_5": 5.}
GRID_SHA256 = "fa9a407ac7ee3f5a3694008e4bcd09ce9cc250452f0c3b11700a4960340abce2"
CORRECTION_FIELDS = (
    "refh", "geoid", "geoid_free2mean", "refh_neutat_delay_total",
    "refh_neutat_delay_derivative", "range_bias_correction", "tide_earth",
    "tide_earth_free2mean", "tide_load", "tide_ocean", "tide_ocean_pole", "tide_pole",
)
SOURCES = {
    "meeting": "docs/meeting/20261015_140029.txt",
    "GEOID18": "https://geodesy.noaa.gov/GEOID/GEOID18/geoid18_tech_details.shtml",
    "HTDP": "https://geodesy.noaa.gov/TOOLS/Htdp/HTDP-user-guide.pdf",
    "NASA_design": "https://ntrs.nasa.gov/api/citations/20240001267/downloads/Conference%20Paper.pdf",
}


def pulse_hash(ids):
    return hashlib.sha256(np.sort(np.asarray(ids, dtype="<i8")).tobytes()).hexdigest()


def operation_evidence(transformer):
    """An available non-ballpark operation may still be a null frame shift."""
    ops = list(transformer.operations)
    return dict(definition=transformer.definition, description=transformer.description,
                accuracy_metadata_m=transformer.accuracy,
                is_noop="proj=noop" in transformer.definition,
                is_ballpark=any(op.has_ballpark_transformation for op in ops)
                or "ballpark" in transformer.description.lower(),
                parameters=[dict(operation=op.name, name=p.name, value=p.value,
                                 unit=p.unit_name) for op in ops for p in op.params])


def physical_status(frame):
    verified = (bool(frame.get("source_realization"))
                and frame.get("source_coordinate_epoch") is not None
                and frame.get("target_coordinate_epoch_achieved") == 2010.0
                and not frame["is_noop"] and not frame["is_ballpark"]
                and frame.get("processing_application_verified") is True)
    return "verified physical transformation" if verified else "provisional physical transformation"


def transform_source_heights(longitude_deg, latitude_deg, refh_m, geoid_m):
    """CASALS arrays and official geodetic models only; no reference/fit parameters.

    No additional stored correction is applied: its sign and applied state are
    unverified. The source is the WGS84 ensemble, without an invented realization.
    """
    lon, lat, h, geoid = [np.asarray(v, dtype=float) for v in
                         (longitude_deg, latitude_deg, refh_m, geoid_m)]
    if lon.ndim != 1 or any(v.shape != lon.shape for v in (lat, h, geoid)):
        raise ValueError("Source coordinates, height, and geoid must have the same 1D shape")
    if not all(np.isfinite(v).all() for v in (lon, lat, h, geoid)):
        raise ValueError("Non-finite source input on frozen pulse population")
    x, y, height, audit = datum.transform_c7(lon, lat, h, return_audit=True)
    group = TransformerGroup(CRS.from_epsg(4979), CRS.from_epsg(6319),
                             always_xy=True, allow_ballpark=False)
    frame_transformer = datum.choose_available_non_ballpark_operation(group)
    flon, flat, fh = [np.asarray(v) for v in frame_transformer.transform(lon, lat, h)]
    frame = operation_evidence(frame_transformer)
    frame.update(source_frame="WGS84 ensemble (EPSG:4979)", source_realization=None,
                 source_coordinate_epoch=None, target_frame="NAD83(2011) (EPSG:6319)",
                 required_target_coordinate_epoch=2010.0, target_coordinate_epoch_achieved=None,
                 source_ellipsoid="WGS84", target_ellipsoid="GRS80",
                 processing_application_verified=False,
                 time_propagation_applied=False, parameter_source="Installed PROJ/EPSG operation",
                 delta_longitude_deg_max_abs=float(np.max(np.abs(flon-lon))),
                 delta_latitude_deg_max_abs=float(np.max(np.abs(flat-lat))),
                 delta_ellipsoidal_height_m_median=float(np.median(fh-h)),
                 delta_ellipsoidal_height_m_max_abs=float(np.max(np.abs(fh-h))))
    combined = audit["selected_operation"]
    audit = dict(frame=frame, combined_operation=combined["definition"],
                 combined_description=combined["description"],
                 combined_accuracy_metadata_m=audit["selected_accuracy_m"],
                 combined_is_ballpark=any(op.get("has_ballpark_transformation", False)
                                         for op in combined["operation_json"].get("steps", [])),
                 selected_grids=audit["selected_operation_grids"],
                 status=physical_status(frame), strict_closure=False)
    values = dict(Native_height_m=h.copy(), EGM2008_height_m=h-geoid,
                  GEOID18_height_m=np.asarray(height), c7_easting_m_epsg6347=np.asarray(x),
                  c7_northing_m_epsg6347=np.asarray(y), frame_longitude_deg=flon,
                  frame_latitude_deg=flat, frame_ellipsoidal_height_m=fh,
                  frame_delta_longitude_deg=flon-lon, frame_delta_latitude_deg=flat-lat,
                  frame_delta_ellipsoidal_height_m=fh-h)
    return values, audit


def evaluate_heights(frame, reference_height_m):
    """Reference changes affect residuals only, never the stored transformed heights."""
    reference = np.asarray(reference_height_m, dtype=float)
    if reference.shape != (len(frame),):
        raise ValueError("Reference and source pulse populations differ")
    out = frame.copy()
    out["reference_height_m"] = reference
    for method in METHODS:
        out[f"{method}_residual_m"] = out[f"{method}_height_m"].to_numpy(float)-reference
    return out


def summarize(frame, *, population):
    rows = []
    for stratum, threshold in STRATA.items():
        eligible = pd.Series(True, index=frame.index) if threshold is None else frame.refh_snr.ge(threshold)
        group = frame.loc[eligible & frame.common_support_all_candidates]
        for method in METHODS:
            values = group[f"{method}_residual_m"].to_numpy(float)
            assert np.isfinite(values).all()
            rows.append(dict(granule_id=frame.granule_id.iloc[0], population=population,
                             stratum=stratum, method=method, eligible_n=int(eligible.sum()),
                             coverage=len(group)/eligible.sum() if eligible.any() else 0.,
                             pulse_sha256=pulse_hash(group.pulse_index),
                             status="provisional physical transformation" if method == "GEOID18"
                             else "native ellipsoidal height" if method == "Native"
                             else "EGM2008-referenced height; not NAVD88",
                             **residual_metrics(values)))
    return pd.DataFrame(rows)


def audit_source_metadata():
    """Reread H5 scalar fields and TDMS metadata, without reading TDMS waveforms."""
    from nptdms import TdmsFile
    evidence, rows = {}, []
    for spec in datum.GRANULES:
        with h5py.File(spec.h5_path, "r") as h5:
            attrs = {k: str(v) for k, v in h5.attrs.items()}
            upstream = {k: dict(basename=attrs[k], local_matches=[str(p.relative_to(datum.PROJECT_ROOT))
                        for p in (datum.PROJECT_ROOT/"data").rglob(attrs[k])])
                        for k in ("gps_file", "geoloc_file", "l1a_file", "ard_file", "tdms_file")}
            evidence[spec.granule_id] = dict(root_attributes=attrs, upstream=upstream,
                source_realization=None, coordinate_epoch=None, GNSS_INS_frame_conversion="UNKNOWN",
                navigation_to_Refh_frame_chain="UNKNOWN",
                time_origin_interpretation="2018-01-01 time counter origin, not a coordinate epoch")
            for name in CORRECTION_FIELDS:
                ds = h5[name]
                v = collect_values(ds)
                description = "UNKNOWN: no per-field official processing contract found"
                action, acts_on, expected = "not applied", "UNKNOWN", "UNKNOWN"
                sign, applied, unit = "UNKNOWN", "UNKNOWN", "UNKNOWN"
                if name == "refh":
                    description, unit = "WGS84 ellipsoidal reference-return height (meeting)", "m (product convention; H5 undeclared)"
                    action, acts_on, expected, applied = "input height", "ellipsoidal height", "identity", "input; not an extra correction"
                elif name == "geoid":
                    description, unit = "EGM2008 tide-free geoid height (Jeff Lee, meeting)", "m (NGA numerical validation)"
                    action, acts_on, sign, expected = "EGM2008 diagnostic only", "geoid height", "H_EGM = h - N_EGM", "-geoid, not NAVD88"
                elif name == "geoid_free2mean":
                    description, acts_on = "Mean-tide geoid conversion provided (meeting); exact field contract unresolved", "geoid height; complete height chain unknown"
                    expected = "centimetre-scale formula match only; not applied"
                elif name == "refh_neutat_delay_total":
                    acts_on = "neutral atmosphere optical path/range/time: exact quantity UNKNOWN"
                    expected = "1-1.6 numeric units; cannot treat as vertical height or apply twice"
                elif name == "refh_neutat_delay_derivative":
                    acts_on, expected = "delay derivative; independent/dependent units UNKNOWN", "~0.00028 numeric slope, not an application proof"
                elif name == "range_bias_correction":
                    acts_on, expected = "range (field name only; official definition unconfirmed)", "stored zero; placeholder versus applied zero unresolved"
                else:
                    acts_on = "solid-Earth/load/ocean/pole/permanent-tide term: range/coordinate/height stage UNKNOWN"
                    expected = "cannot infer a vertical sign or sum terms from names"
                rows.append(dict(granule_id=spec.granule_id, field=name,
                    official_definition=description, units=unit, H5_units_attribute=str(ds.attrs.get("units", "UNDECLARED")),
                    acts_on=acts_on, sign_convention=sign, applied_to_refh=applied,
                    expected_height_effect=expected, current_action=action,
                    eligible_for_additional_correction=False, n_valid=int(np.isfinite(v).sum()),
                    median_numeric=float(np.nanmedian(v)), minimum_numeric=float(np.nanmin(v)),
                    maximum_numeric=float(np.nanmax(v)),
                    attributes={k: str(value) for k, value in ds.attrs.items()}))
    tdms = []
    for path in sorted((datum.PROJECT_ROOT/"data/raw/tdms").glob("*.tdms")):
        data = TdmsFile.read_metadata(path)
        tdms.append(dict(file=path.name, bytes=path.stat().st_size,
            metadata_prefix_sha256=hashlib.sha256(path.open("rb").read(16384)).hexdigest(),
            file_properties={k: str(v) for k, v in data.properties.items()},
            groups=[dict(name=g.name, properties={k: str(v) for k, v in g.properties.items()},
                    channel_count=len(g.channels()), channel_properties=[dict(name=c.name,
                    properties={k: str(v) for k, v in c.properties.items()}) for c in g.channels()])
                    for g in data.groups()]))
    return pd.DataFrame(rows), dict(H5=evidence, TDMS=tdms, sources=SOURCES,
        HTDP_assessment="Official time-dependent method investigated; no transformation run because source realization and coordinate epoch are unknown. WGS84 transforms are approximate; HTDP must not transform between realizations within a NAD83 plate frame.",
        NASA_design_assessment="POS 510 hardware described; no per-granule GNSS solution frame/epoch or correction applied-state contract established")


def grid_point_check(grid_path, lon, lat, h, proj_height):
    """Independent four-cell raster interpolation; does not call PROJ to sample N."""
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(grid_path) as ds:
        assert ds.crs.to_epsg() == 6319
        glon = lon % 360
        col, row = (~ds.transform)*(glon, lat)
        col, row = col-.5, row-.5
        c, r = int(np.floor(col)), int(np.floor(row))
        if not (0 <= c < ds.width-1 and 0 <= r < ds.height-1):
            raise ValueError("Single-point check is outside the audited GEOID18 grid")
        cells = ds.read(1, window=Window(c, r, 2, 2))
        fc, fr = col-c, row-r
        n = float((1-fr)*((1-fc)*cells[0, 0]+fc*cells[0, 1])
                  + fr*((1-fc)*cells[1, 0]+fc*cells[1, 1]))
        # This audited GeoTIFF stores signed geoid undulation N; C7 uses inverse vgridshift.
        navd88 = float(h-n)
        np.testing.assert_allclose(navd88, proj_height, rtol=0, atol=1e-7)
        return dict(longitude_deg=lon, latitude_deg=lat, assumed_NAD83_ellipsoidal_h_m=h,
                    N_GEOID18_m=n, independently_computed_H_m=navd88,
                    PROJ_H_m=float(proj_height), difference_m=navd88-float(proj_height),
                    formula="H = h - N; subtract signed grid undulation through inverse vgridshift",
                    grid_crs=ds.crs.to_string(), target_epsg=ds.tags()["target_crs_epsg_code"],
                    bounds=list(ds.bounds), area_of_use=ds.tags()["area_of_use"],
                    source_frame_status="assumed NAD83 only; CASALS frame not verified")


def run_alignment(output):
    """Physical transformations precede independent 3DEP evaluation; no fitting path."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    grid = datum.OUTPUT_ROOT/"proj_data"/datum.GEOID18_GRID
    assert sha256(grid) == GRID_SHA256
    network.set_network_enabled(False)
    datadir.set_data_dir(str(grid.parent))
    corrections, source_evidence = audit_source_metadata()
    corrections.to_csv(output/"physical_correction_audit.csv", index=False)
    (output/"source_frame_audit.json").write_text(json.dumps(source_evidence, indent=2), encoding="utf-8")
    split_path = datum.OUTPUT_ROOT/"selection_split_manifest.csv"
    assert sha256(split_path) == "cd714dd3057193bc9f510dfa34adfe19b2c28db628f2b93f41525a8a30f8b702"
    split = pd.read_csv(split_path)
    baseline = pd.read_parquet(datum.OUTPUT_ROOT/"vertical_candidate_common_sample.parquet")
    reference_columns = ["granule_id", "pulse_index", "research_role", "casals_refh_m", "longitude_deg", "latitude_deg",
                         "refh_snr", "easting_m_epsg6347", "northing_m_epsg6347", "surface_acceptable",
                         "reference_height_local_plane_m", "common_support_all_candidates", "C7_height_m",
                         "c7_reference_height_local_plane_m"]
    road = pd.read_parquet(HISTORY/"ground_aware/ground_target_test_population.parquet", columns=reference_columns)
    geoid_sample = pd.read_csv(datum.OUTPUT_ROOT/"geoid18_sample_comparison.csv")
    geoid_prior = pd.read_csv(HISTORY/"correction_audit/geoid_model_comparison.csv")
    egm = load_egm2008_grid()
    frames, metrics, parity, geoid_rows, operation_rows = [], [], [], [], []
    operations = dict(grid_name=grid.name, grid_sha256=GRID_SHA256,
                      NGA_check=validate_egm2008_interpolation(egm), granules={})
    for spec in datum.GRANULES:
        print(f"Processing {spec.granule_id}: physical heights, then class-2 reference", flush=True)
        roles = split.loc[split.granule_id.eq(spec.granule_id), ["pulse_index", "research_role"]]
        cols = ["pulse_index", "sweep_num", "track_num", "longitude_deg", "latitude_deg", "casals_refh_m",
                "refh_snr", "easting_m_epsg6347", "northing_m_epsg6347", "reference_height_local_plane_m", "surface_acceptable"]
        field = pd.read_parquet(spec.revised_dir/"matched_casals_3dep_points.parquet", columns=cols)
        field = field.merge(roles, on="pulse_index", how="inner", validate="one_to_one")
        assert len(field) == len(roles)
        field["granule_id"], field["population"] = spec.granule_id, "field"
        rr = road.loc[road.granule_id.eq(spec.granule_id)].copy()
        rr["population"] = "road"
        fresh_parts = []
        with h5py.File(spec.h5_path, "r") as h5:
            for source in (field, rr):
                source = source.sort_values("pulse_index").reset_index(drop=True)
                ids = source.pulse_index.to_numpy(np.int64)
                for name, col in (("refh", "casals_refh_m"), ("refh_longitude", "longitude_deg"),
                                  ("refh_latitude", "latitude_deg"), ("refh_snr", "refh_snr")):
                    np.testing.assert_allclose(h5[name][ids], source[col], rtol=0, atol=1e-10)
                # Only these four H5 inputs reach the physical transformation boundary.
                heights, audit = transform_source_heights(h5["refh_longitude"][ids], h5["refh_latitude"][ids],
                                                         h5["refh"][ids], h5["geoid"][ids])
                fresh_parts.append(source.assign(**heights))
            sample = geoid_sample.loc[geoid_sample.granule_id.eq(spec.granule_id)].sort_values("pulse_index")
            sid = sample.pulse_index.to_numpy(np.int64)
            delta = np.asarray(h5["geoid"][sid])-bilinear_grid(egm, h5["refh_longitude"][sid], h5["refh_latitude"][sid])
            old = geoid_prior.loc[geoid_prior.granule_id.eq(spec.granule_id) & geoid_prior.comparison.eq("H5 minus EGM2008")].iloc[0]
            np.testing.assert_allclose(np.median(delta), old.median_difference_m, atol=1e-12, rtol=0)
            geoid_rows.append(dict(granule_id=spec.granule_id, n=len(delta),
                                   H5_minus_NGA_EGM2008_median_m=float(np.median(delta)), max_abs_m=float(np.max(np.abs(delta)))))
        fresh = pd.concat(fresh_parts, ignore_index=True)
        query = fresh[["c7_easting_m_epsg6347", "c7_northing_m_epsg6347"]].to_numpy()
        xy, z, grid_meta = datum.build_local_class2_query_grid(spec.laz_path, query)
        surface = interpolate_ground_surface(query, xy, z)
        fresh["c7_reference_height_local_plane_m"] = surface["reference_height_plane_m"]
        fresh["c7_surface_acceptable"] = surface["surface_acceptable"]
        fresh["common_support_all_candidates"] = (fresh.surface_acceptable & fresh.c7_surface_acceptable
            & np.isfinite(fresh.reference_height_local_plane_m) & np.isfinite(fresh.c7_reference_height_local_plane_m))
        for name in ("local_slope_deg", "neighbor_count", "plane_residual_nmad_m"):
            fresh[f"c7_{name}"] = surface[name]
        fresh = evaluate_heights(fresh, fresh.c7_reference_height_local_plane_m)
        for pop, old in (("field", baseline.loc[baseline.granule_id.eq(spec.granule_id)]),
                         ("road", rr.loc[rr.common_support_all_candidates])):
            new = fresh.loc[fresh.population.eq(pop) & fresh.common_support_all_candidates].sort_values("pulse_index")
            old = old.sort_values("pulse_index")
            np.testing.assert_array_equal(new.pulse_index, old.pulse_index)
            for new_col, old_col in (("GEOID18_height_m", "C7_height_m"),
                                     ("c7_reference_height_local_plane_m", "c7_reference_height_local_plane_m")):
                np.testing.assert_allclose(new[new_col], old[old_col], atol=1e-8, rtol=0)
            parity.append(dict(granule_id=spec.granule_id, population=pop, n=len(new),
                               pulse_sha256=pulse_hash(new.pulse_index),
                               max_height_difference_m=float(np.max(np.abs(new.GEOID18_height_m.to_numpy()-old.C7_height_m.to_numpy()))),
                               max_reference_difference_m=float(np.max(np.abs(new.reference_height_m.to_numpy()-old.c7_reference_height_local_plane_m.to_numpy())))))
            metrics.append(summarize(fresh.loc[fresh.population.eq(pop) & fresh.research_role.eq("independent_test")], population=pop))
        first = fresh.iloc[0]
        single = grid_point_check(grid, first.frame_longitude_deg, first.frame_latitude_deg,
                                  first.frame_ellipsoidal_height_m, first.GEOID18_height_m)
        operations["granules"][spec.granule_id] = dict(**audit, reference_grid=grid_meta, independent_grid_point=single,
            median_total_vertical_change_m=float(np.median(fresh.GEOID18_height_m-fresh.Native_height_m)),
            maximum_total_vertical_change_m=float(np.max(fresh.GEOID18_height_m-fresh.Native_height_m)),
            minimum_total_vertical_change_m=float(np.min(fresh.GEOID18_height_m-fresh.Native_height_m)))
        operation_rows.append(dict(granule_id=spec.granule_id, **audit["frame"]))
        frames.append(fresh)
    result = dict(population=pd.concat(frames, ignore_index=True), metrics=pd.concat(metrics, ignore_index=True),
                  parity=pd.DataFrame(parity), geoid=pd.DataFrame(geoid_rows), frame=pd.DataFrame(operation_rows),
                  corrections=corrections, operations=operations)
    prior = pd.read_csv(datum.OUTPUT_ROOT/"final_elevation_metrics.csv")
    for row in result["metrics"].query("population == 'field' and method == 'GEOID18'").itertuples():
        old = prior.loc[prior.granule_id.eq(row.granule_id) & prior.configuration_id.eq("C7") & prior.snr_stratum.eq(row.stratum)].iloc[0]
        assert row.n == old.n
        for new_col, old_col in (("mean_bias_m", "bias_m"), ("RMSE_m", "rmse_m"), ("NMAD_m", "nmad_m")):
            np.testing.assert_allclose(getattr(row, new_col), old[old_col], atol=1e-8, rtol=0)
    result["final_summary"] = result["metrics"].query("population == 'field' and stratum == 'snr_ge_5' and method == 'GEOID18'").copy()
    result["final_summary"]["strict_closure"] = False
    # Drop legacy candidate heights from the saved population; no stored correction enters the result.
    result["population"] = result["population"].drop(columns=["C7_height_m"], errors="ignore")
    result["population"].to_parquet(output/"population.parquet", index=False)
    for key in ("metrics", "parity", "geoid", "frame", "final_summary"):
        result[key].to_csv(output/f"{key}.csv", index=False)
    (output/"operation_audit.json").write_text(json.dumps(operations, indent=2), encoding="utf-8")
    inputs = [split_path, grid, datum.OUTPUT_ROOT/"vertical_candidate_common_sample.parquet",
              datum.OUTPUT_ROOT/"final_elevation_metrics.csv", HISTORY/"ground_aware/ground_target_test_population.parquet",
              datum.OUTPUT_ROOT/"geoid18_sample_comparison.csv", HISTORY/"correction_audit/geoid_model_comparison.csv",
              HISTORY/"correction_audit/reference_data/egm2008_2p5min_nga.zip", datum.PROJECT_ROOT/SOURCES["meeting"],
              Path(__file__), Path(datum.__file__), datum.PROJECT_ROOT/"research/elevation_comparison.py",
              datum.PROJECT_ROOT/"research/reference/refh_correction_audit.py"]
    for spec in datum.GRANULES:
        inputs.extend([spec.h5_path, spec.laz_path, spec.revised_dir/"matched_casals_3dep_points.parquet"])
    pd.DataFrame([dict(path=p.relative_to(datum.PROJECT_ROOT).as_posix(), bytes=p.stat().st_size,
                       sha256=sha256(p)) for p in inputs]).to_csv(output/"input_sha256.csv", index=False)
    result["output"] = output
    return result


def plot_results(result):
    """Only uncorrected physical height differences; full high-quality test populations."""
    import matplotlib.pyplot as plt
    figures = []
    pop = result["population"]
    test = pop.loc[pop.population.eq("field") & pop.research_role.eq("independent_test")
                   & pop.common_support_all_candidates & pop.refh_snr.ge(5)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    colors = {"Native": "#1f77b4", "EGM2008": "#ff7f0e", "GEOID18": "#2ca02c"}
    for row, (gid, group) in zip(axes, test.groupby("granule_id", sort=True)):
        for ax, methods in zip(row, [METHODS, ("EGM2008", "GEOID18")]):
            for method in methods:
                v = np.sort(group[f"{method}_residual_m"])
                ax.plot(v, np.arange(1, len(v)+1)/len(v), label=method, color=colors[method])
            view = "full view" if len(methods) == 3 else "converted heights"
            ax.set(title=f"{gid[11:19]} | n={len(group):,} | {view}", xlabel="CASALS - 3DEP (m)", ylabel="Cumulative fraction")
            ax.axvline(0, color="grey", ls=":")
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle("Independent field test | SNR >= 5 | entire residual distribution")
    figures.append(("residual_distributions.png", fig))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    lo, hi = test.GEOID18_residual_m.min(), test.GEOID18_residual_m.max()
    for ax, (gid, group) in zip(axes, test.groupby("granule_id", sort=True)):
        x, y = group.c7_easting_m_epsg6347, group.c7_northing_m_epsg6347
        points = ax.scatter(x-x.min(), y-y.min(), c=group.GEOID18_residual_m, s=7,
                            cmap="viridis", vmin=lo, vmax=hi, rasterized=True)
        ax.set(title=f"{gid[11:19]} | n={len(group):,}", xlabel="Easting from test minimum (m)", ylabel="Northing from test minimum (m)", aspect="equal")
    fig.colorbar(points, ax=axes, label="Provisional GEOID18 - 3DEP (m)")
    fig.suptitle("Provisional GEOID18 spatial residuals | SNR >= 5 | full color range")
    figures.append(("geoid18_spatial_residuals.png", fig))
    for name, fig in figures:
        fig.savefig(result["output"]/name, dpi=160)
    return figures
