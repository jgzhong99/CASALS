p='research/reference/frame_hypotheses.py'
s=open(p,encoding='utf8').read()
s=s.replace("    metric_rows, shift_rows, op_rows = [], [], []\n","    metric_rows, shift_rows, op_rows, pulse_rows = [], [], [], []\n")
s=s.replace("            for pop_name in (\"field\", \"road\"):",
"""            pulse_rows.append(f.assign(hypothesis=name, supported=ok)[
                ["granule_id", "hypothesis", "pulse_index", "population", "research_role", "refh_snr",
                 "supported", "H", "ref", "residual"]])
            for pop_name in (\"field\", \"road\"):""")
s=s.replace("    out = dict(metrics=pd.DataFrame(metric_rows), shifts=pd.DataFrame(shift_rows), operations=pd.DataFrame(op_rows))\n    for k, v in out.items():",
"""    out = dict(metrics=pd.DataFrame(metric_rows), shifts=pd.DataFrame(shift_rows), operations=pd.DataFrame(op_rows))
    pulses = pd.concat(pulse_rows, ignore_index=True)
    pulses.to_parquet(output / "frame_hypothesis_pulses.parquet", index=False)
    for k, v in out.items():""")
s=s.replace("    out[\"output\"] = output\n    return out\n","    out[\"pulses\"] = pulses\n    out[\"output\"] = output\n    return out\n")
s+='''

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
        flon, flat, fh, _ = apply_frame(HYPOTHESES[hypothesis][0], epoch, lon, lat, h)
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
                             VDatum_height_shift_m=Hv - (p.refh_ellipsoidal_m - float(
                                 Hp - fh[list(g.pulse_index).index(p.pulse_index)] + h[list(g.pulse_index).index(p.pulse_index)])) if False else np.nan,
                             status=res["status"]))
            time.sleep(gc.REQUEST_PAUSE_SECONDS)
    df = pd.DataFrame(rows).drop(columns="VDatum_height_shift_m")
    (output / "vdatum_itrf2014_calls.json").write_text(json.dumps(gc._json_safe(dict(calls=calls)), indent=1), encoding="utf-8")
    df.to_csv(output / "vdatum_itrf2014_comparison.csv", index=False)
    ok = df.status.eq("success")
    summ = df.loc[ok].groupby("granule_id").PROJ_minus_VDatum_m.agg(
        successful="count", mean_m="mean", median_m="median", max_abs_m=lambda s: s.abs().max()).reset_index()
    summ["requested"] = df.groupby("granule_id").size().values
    return dict(comparison=df, summary=summ, output=output)
'''
s=s.replace("from datetime import datetime\n","import time\nfrom datetime import datetime\n")
open(p,'w',encoding='utf8').write(s)
