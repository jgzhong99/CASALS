"""Meeting-informed research orchestration; production and historical outputs are read-only."""
from pathlib import Path
import hashlib
import json

import h5py
import numpy as np
import pandas as pd
from pyproj import datadir, network

from research.reference import vertical_datum_comparison as datum
from research.reference.refh_correction_audit import (
    bilinear_grid, load_egm2008_grid, residual_metrics, sha256,
    validate_egm2008_interpolation,
)

HISTORY = datum.OUTPUT_ROOT.parent
METHODS = {"Native": "C0", "EGM2008": "C1", "Tide minus (sensitivity)": "C3",
           "Tide plus (sensitivity)": "C4", "GEOID18": "C7"}
STRATA = {"all_valid_refh": -np.inf, "snr_ge_3": 3., "snr_ge_5": 5.}


def pulse_hash(ids):
    return hashlib.sha256(np.sort(np.asarray(ids, dtype="<i8")).tobytes()).hexdigest()


def summarize(frame, *, population):
    """One fixed support mask and denominator for every method; never trim residuals."""
    rows = []
    for stratum, threshold in STRATA.items():
        eligible = frame.refh_snr.ge(threshold)
        group = frame.loc[eligible & frame.common_support_all_candidates]
        for method in METHODS:
            values = group[f"{method}_residual_m"].to_numpy(float)
            assert np.isfinite(values).all()
            rows.append(dict(granule_id=frame.granule_id.iloc[0], population=population,
                             stratum=stratum, method=method, eligible_n=int(eligible.sum()),
                             coverage=len(group)/eligible.sum() if eligible.any() else 0.,
                             pulse_sha256=pulse_hash(group.pulse_index),
                             **residual_metrics(values)))
    return pd.DataFrame(rows)


def empirical_alignment(frame):
    """Fit only the preserved calibration_offset half; evaluate a separate field site."""
    rows, residuals = [], []
    for stratum, threshold in STRATA.items():
        good = frame.common_support_all_candidates & frame.refh_snr.ge(threshold)
        cal = frame.loc[good & frame.research_role.eq("calibration_offset")]
        test = frame.loc[good & frame.research_role.eq("independent_test")]
        assert len(cal) and len(test)
        assert set(cal.pulse_index).isdisjoint(test.pulse_index)
        # Convex bounding boxes must be separated as well as pulse identities.
        xy = ["easting_m_epsg6347", "northing_m_epsg6347"]
        assert ((cal[xy].max() < test[xy].min()) | (test[xy].max() < cal[xy].min())).any()
        for method in ("Native", "EGM2008", "GEOID18"):
            offset = float(cal[f"{method}_residual_m"].median())
            before = test[f"{method}_residual_m"].to_numpy(float)
            after = before - offset
            for state, values in (("before", before), ("aligned", after)):
                rows.append(dict(granule_id=frame.granule_id.iloc[0], stratum=stratum,
                                 method=method, state=state, offset_m=offset,
                                 calibration_n=len(cal), test_n=len(test),
                                 calibration_pulse_sha256=pulse_hash(cal.pulse_index),
                                 test_pulse_sha256=pulse_hash(test.pulse_index),
                                 **residual_metrics(values)))
            r = test[["granule_id", "pulse_index", "track_num", "easting_m_epsg6347",
                      "northing_m_epsg6347", "c7_reference_height_local_plane_m"]].copy()
            r["stratum"], r["method"], r["offset_m"] = stratum, method, offset
            r["before_residual_m"], r["aligned_residual_m"] = before, after
            residuals.append(r)
            np.testing.assert_allclose(residual_metrics(before)["cRMSE_m"],
                                       residual_metrics(after)["cRMSE_m"], atol=1e-10)
    return pd.DataFrame(rows), pd.concat(residuals, ignore_index=True)


def run_alignment(output):
    """Reread both H5/LAZ pairs and reproduce C7 on the preserved field and road IDs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    network.set_network_enabled(False)
    grid = datum.OUTPUT_ROOT / "proj_data" / datum.GEOID18_GRID
    assert sha256(grid) == "fa9a407ac7ee3f5a3694008e4bcd09ce9cc250452f0c3b11700a4960340abce2"
    datadir.set_data_dir(str(grid.parent))
    split_path = datum.OUTPUT_ROOT / "selection_split_manifest.csv"
    assert sha256(split_path) == "cd714dd3057193bc9f510dfa34adfe19b2c28db628f2b93f41525a8a30f8b702"
    split = pd.read_csv(split_path)
    baseline = pd.read_parquet(datum.OUTPUT_ROOT / "vertical_candidate_common_sample.parquet")
    road = pd.read_parquet(HISTORY / "ground_aware/ground_target_test_population.parquet")
    geoid_sample = pd.read_csv(datum.OUTPUT_ROOT / "geoid18_sample_comparison.csv")
    geoid_prior = pd.read_csv(HISTORY / "correction_audit/geoid_model_comparison.csv")
    egm = load_egm2008_grid()
    nga_check = validate_egm2008_interpolation(egm)
    frames, metrics, alignments, aligned_rows, parity, geoid_rows, sites = [], [], [], [], [], [], []
    operations = {}
    for spec in datum.GRANULES:
        print(f"Processing {spec.granule_id}: current H5 and class-2 LAZ", flush=True)
        field = datum._load_matched_roles(spec, split).sort_values("pulse_index").reset_index(drop=True)
        field["granule_id"], field["population"] = spec.granule_id, "field"
        rr = road.loc[road.granule_id.eq(spec.granule_id)].sort_values("pulse_index").copy()
        rr["population"] = "road"
        with h5py.File(spec.h5_path, "r") as h5:
            ids = rr.pulse_index.to_numpy(np.int64)
            for name, col in (("refh", "casals_refh_m"), ("refh_longitude", "longitude_deg"),
                              ("refh_latitude", "latitude_deg"), ("refh_snr", "refh_snr")):
                np.testing.assert_allclose(h5[name][ids], rr[col], rtol=0, atol=1e-10)
            candidates, _ = datum.build_height_candidates(h5["refh"][ids], h5["geoid"][ids], h5["geoid_free2mean"][ids])
            for key, values in candidates.items():
                rr[f"{key}_height_m"] = values
            sample = geoid_sample.loc[geoid_sample.granule_id.eq(spec.granule_id)].sort_values("pulse_index")
            sid = sample.pulse_index.to_numpy(np.int64)
            h5_geoid = np.asarray(h5["geoid"][sid], float)
            np.testing.assert_allclose(h5_geoid, sample.h5_geoid_field_m, atol=1e-12, rtol=0)
            delta = h5_geoid - bilinear_grid(egm, h5["refh_longitude"][sid], h5["refh_latitude"][sid])
            old = geoid_prior.loc[geoid_prior.granule_id.eq(spec.granule_id) & geoid_prior.comparison.eq("H5 minus EGM2008")].iloc[0]
            np.testing.assert_allclose(np.median(delta), old.median_difference_m, atol=1e-12, rtol=0)
            geoid_rows.append(dict(granule_id=spec.granule_id, n=len(delta),
                                   H5_minus_NGA_EGM2008_median_m=float(np.median(delta)),
                                   max_abs_m=float(np.max(np.abs(delta)))))
        # Reuse the audited class-2 grid/interpolator once for the union of fixed queries.
        source = pd.concat([field, rr], ignore_index=True)
        fresh, audit = datum._surface_query(spec, source)
        operations[spec.granule_id] = {"operation": audit["transform"]["selected_operation"]["definition"],
                                      "accuracy_metadata_m": audit["transform"]["selected_accuracy_m"],
                                      "reference_grid": audit["reference_grid"]}
        # C7 XY is identical here, but the historical local-query and full-grid planes
        # can differ at cell boundaries. Use the recomputed C7 plane for every method.
        np.testing.assert_allclose(fresh.c7_easting_m_epsg6347, fresh.easting_m_epsg6347, atol=1e-8, rtol=0)
        np.testing.assert_allclose(fresh.c7_northing_m_epsg6347, fresh.northing_m_epsg6347, atol=1e-8, rtol=0)
        keep = fresh.common_support_all_candidates
        operations[spec.granule_id]["max_C7_vs_original_plane_difference_m"] = float(np.max(np.abs(
            fresh.loc[keep, "c7_reference_height_local_plane_m"] -
            fresh.loc[keep, "reference_height_local_plane_m"])))
        for pop, old in (("field", baseline.loc[baseline.granule_id.eq(spec.granule_id)]),
                         ("road", rr.loc[rr.common_support_all_candidates])):
            new = fresh.loc[fresh.population.eq(pop) & keep].sort_values("pulse_index")
            old = old.sort_values("pulse_index")
            np.testing.assert_array_equal(new.pulse_index, old.pulse_index)
            errors = {}
            for col in ("C0_height_m", "C1_height_m", "C3_height_m", "C4_height_m", "C7_height_m", "c7_reference_height_local_plane_m"):
                np.testing.assert_allclose(new[col], old[col], atol=1e-8, rtol=0)
                errors[col] = float(np.max(np.abs(new[col].to_numpy()-old[col].to_numpy())))
            parity.append(dict(granule_id=spec.granule_id, population=pop, n=len(new),
                               pulse_sha256=pulse_hash(new.pulse_index), max_abs_m=max(errors.values())))
        for method, code in METHODS.items():
            fresh[f"{method}_residual_m"] = fresh[f"{code}_height_m"] - fresh.c7_reference_height_local_plane_m
        for pop in ("field", "road"):
            group = fresh.loc[fresh.population.eq(pop)].copy()
            metrics.append(summarize(group.loc[group.research_role.eq("independent_test")], population=pop))
        ff = fresh.loc[fresh.population.eq("field")]
        am, ar = empirical_alignment(ff)
        alignments.append(am)
        aligned_rows.append(ar)
        sm = pd.read_csv(spec.revised_dir / "spatial_calibration_test_manifest.csv")
        sites.append(sm.loc[sm.split.isin(["calibration", "independent_test"])])
        frames.append(fresh)
    result = dict(population=pd.concat(frames, ignore_index=True), metrics=pd.concat(metrics, ignore_index=True),
                  alignment=pd.concat(alignments, ignore_index=True), aligned=pd.concat(aligned_rows, ignore_index=True),
                  parity=pd.DataFrame(parity), geoid=pd.DataFrame(geoid_rows), sites=pd.concat(sites, ignore_index=True))
    # Explicit metric parity on identical historical field test IDs, in addition to XYZ parity.
    prior = pd.read_csv(datum.OUTPUT_ROOT / "final_elevation_metrics.csv")
    for row in result["metrics"].query("population == 'field' and method == 'GEOID18'").itertuples():
        old = prior.loc[prior.granule_id.eq(row.granule_id) & prior.configuration_id.eq("C7") & prior.snr_stratum.eq(row.stratum)].iloc[0]
        assert row.n == old.n
        for new_col, old_col in (("mean_bias_m", "bias_m"), ("RMSE_m", "rmse_m"), ("NMAD_m", "nmad_m")):
            np.testing.assert_allclose(getattr(row, new_col), old[old_col], atol=1e-8, rtol=0)
    for key, table in result.items():
        if key in ("population", "aligned"):
            table.to_parquet(output/f"{key}.parquet", index=False)
        else:
            table.to_csv(output/f"{key}.csv", index=False)
    (output/"operation_audit.json").write_text(json.dumps(dict(operations=operations, NGA_check=nga_check), indent=2), encoding="utf-8")
    inputs = [split_path, grid, datum.OUTPUT_ROOT / "vertical_candidate_common_sample.parquet",
              datum.OUTPUT_ROOT / "final_elevation_metrics.csv",
              HISTORY / "ground_aware/ground_target_test_population.parquet",
              datum.OUTPUT_ROOT / "geoid18_sample_comparison.csv",
              HISTORY / "correction_audit/geoid_model_comparison.csv",
              HISTORY / "correction_audit/reference_data/egm2008_2p5min_nga.zip",
              HISTORY / "datum_comparison/3dep_z_provenance_audit.json",
              HISTORY / "final_bias_attribution/physical_hypothesis_metrics.csv",
              datum.PROJECT_ROOT / "docs/meeting/20261015_140029.txt",
              Path(__file__), Path(datum.__file__),
              datum.PROJECT_ROOT / "research/elevation_comparison.py"]
    for spec in datum.GRANULES:
        inputs.extend([spec.h5_path, spec.laz_path,
                       spec.revised_dir / "matched_casals_3dep_points.parquet",
                       spec.revised_dir / "spatial_calibration_test_manifest.csv"])
    pd.DataFrame([dict(path=p.relative_to(datum.PROJECT_ROOT).as_posix(),
                       bytes=p.stat().st_size, sha256=sha256(p)) for p in inputs]).to_csv(
                           output / "input_sha256.csv", index=False)
    result["output"] = output
    return result


def plot_results(result):
    """Full empirical distributions; maps and profiles use all SNR>=5 supported field test rows."""
    import matplotlib.pyplot as plt
    figures = []
    pop = result["population"]
    test = pop.loc[pop.population.eq("field") & pop.research_role.eq("independent_test") &
                   pop.common_support_all_candidates & pop.refh_snr.ge(5)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    colors = {"Native": "#1f77b4", "EGM2008": "#ff7f0e", "GEOID18": "#2ca02c"}
    for row, (gid, group) in zip(axes, test.groupby("granule_id", sort=True)):
        for ax, methods in zip(row, [("Native", "EGM2008", "GEOID18"), ("EGM2008", "GEOID18")]):
            for method in methods:
                v = np.sort(group[f"{method}_residual_m"])
                ax.plot(v, np.arange(1, len(v)+1)/len(v), label=method, color=colors[method])
            view = "full view" if len(methods) == 3 else "converted heights"
            ax.set(title=f"{gid[11:19]} | n={len(group):,} | {view}", xlabel="CASALS − 3DEP (m)", ylabel="Cumulative fraction")
            ax.axvline(0, color="grey", ls=":")
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle("Independent field test | SNR ≥ 5 | entire residual distribution")
    figures.append(("residual_distributions.png", fig))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    lo, hi = test["GEOID18_residual_m"].min(), test["GEOID18_residual_m"].max()
    for ax, (gid, group) in zip(axes, test.groupby("granule_id", sort=True)):
        x, y = group.c7_easting_m_epsg6347, group.c7_northing_m_epsg6347
        points = ax.scatter(x-x.min(), y-y.min(), c=group.GEOID18_residual_m, s=7,
                            cmap="viridis", vmin=lo, vmax=hi, rasterized=True)
        ax.set(title=f"{gid[11:19]} | n={len(group):,}", xlabel="Easting from test minimum (m)", ylabel="Northing from test minimum (m)", aspect="equal")
    fig.colorbar(points, ax=axes, label="GEOID18 − 3DEP (m)")
    fig.suptitle("Unaligned GEOID18 spatial residuals | SNR ≥ 5 | full color range")
    figures.append(("geoid18_spatial_residuals.png", fig))
    ar = result["aligned"].query("method == 'GEOID18' and stratum == 'snr_ge_5'")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (gid, group) in zip(axes, ar.groupby("granule_id", sort=True)):
        xy = group[["easting_m_epsg6347", "northing_m_epsg6347"]].to_numpy()
        centered = xy-xy.mean(axis=0)
        axis = np.linalg.svd(centered, full_matrices=False)[2][0]
        distance = centered@axis
        ax.scatter(distance, group.before_residual_m, s=5, alpha=.35, label="GEOID18")
        ax.scatter(distance, group.aligned_residual_m, s=5, alpha=.35, label="GEOID18 − calibration offset")
        ax.axhline(0, color="grey", ls=":")
        ax.set(title=f"{gid[11:19]} | n={len(group):,}", xlabel="Test ROI principal-axis distance (m)", ylabel="Height minus 3DEP (m)")
        ax.legend(fontsize=8)
    fig.suptitle("Held-out field height differences before/after empirical alignment | SNR ≥ 5")
    figures.append(("alignment_profile.png", fig))
    for name, fig in figures:
        fig.savefig(result["output"]/name, dpi=160)
    return figures
