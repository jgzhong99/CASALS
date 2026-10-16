p='research/reference/frame_hypotheses.py'
s=open(p,encoding='utf8').read()
# 1. per-pulse residual table inside run_frame_hypotheses
marker="    result = {\n        \"metrics\": pd.DataFrame(common_metric_rows),"
assert marker in s
add='''    pulse_frames = []
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

'''
s=s.replace(marker,add+marker)
s=s.replace('        "available_hypotheses": sorted({name for _, name in method_frames}),\n        "output": output,',
            '        "available_hypotheses": sorted({name for _, name in method_frames}),\n        "pulse_residuals": pulse_residuals,\n        "output": output,')
# 2. vdatum function fix
s=s.replace('flon, flat, fh, _ = apply_frame(HYPOTHESES[hypothesis][0], epoch, lon, lat, h)',
            'flon, flat, fh, _ = apply_frame(HYPOTHESES[hypothesis]["source_epsg"], epoch, lon, lat, h, coordinate_time=epoch)')
open(p,'w',encoding='utf8').write(s)

import json
q=r'notebooks/L1B/geolocation/vertical_datum_alignment.ipynb'
nb=json.load(open(q,encoding='utf8'))
def G(c): return ''.join(c['source'])
P='H2_ITRF2014_obs_epoch'
for c in nb['cells']:
    t=G(c)
    n=t
    n=n.replace("F['pulses']","F['pulse_residuals']").replace("pl = F['pulse_residuals']; base_pop = R['population']","pl = F['pulse_residuals']")
    n=n.replace('H1_WGS84_G1762_obs_epoch',P)
    if c['cell_type']=='code' and 'block_bootstrap_bias(' in n:
        n=n.replace("for h in F['pulse_residuals'].hypothesis.unique()","for h in F['pulse_residuals'].hypothesis.unique() if not h.startswith('H0')")
        # H0 CDF from the same table
        a=n.index("for ax, (gid, g) in zip(axes, base_pop"); b=n.index("    ax.axvline(0")
        n=n[:a]+'''for ax, gid in zip(axes, sorted(pl.granule_id.unique())):
    for hyp, label, color in (('H0_noop_C7_baseline', 'H0 no frame change', 'tab:red'), (PRIMARY, 'H2 ITRF2014 -> NAD83(2011)', 'tab:green')):
        g1 = pl.loc[pl.granule_id.eq(gid) & pl.hypothesis.eq(hyp) & pl.population.eq('field')
                    & pl.research_role.eq('independent_test') & pl.supported & pl.refh_snr.ge(5)]
        v = np.sort(g1.residual); ax.plot(v, np.arange(1, len(v)+1)/len(v), label=label, color=color)
'''+n[b:]
        n=n.replace("hyp_all = F['metrics']","PRIMARY = '%s'\nhyp_all = F['metrics']"%P)
        n=n.replace("H1-H4","H2 primary; H1-H4")
        n=n.replace("Table 6b-2 - Road reviewed-interior check (separate population), H1","Table 6b-2 - Road reviewed-interior check (separate population), H2")
    n=n.replace("if not h.startswith('H0')","if not h.startswith('H0')")
    if c['cell_type']=='code' and 'final = ci.query' in n:
        n=n.replace("ci.query(\"hypothesis != 'H0'\")","ci").replace("H1 (WGS84 G1762 -> NAD83(2011) @2010.0 -> GEOID18)","H2 (ITRF2014 at flight time -> NAD83(2011) -> GEOID18; frame epoch not propagated)").replace("H1-H4 range","H2-H4 and H1 range").replace("bias_range_H1_to_H4_m","bias_range_hypotheses_m")
    if c['cell_type']=='markdown' and n.startswith('### 6b'):
        n=n.replace("H1-H4 are reported as a range; H0 (no frame change) is shown only as the comparison.","H2 (ITRF2014) is shown as the primary hypothesis because it is the only one with an independent geocentric XYZ checkpoint, HTDP check and VDatum check, not because of its residual; H1, H3 and H4 are reported alongside as a range and H0 (no frame change) as the comparison.")
    if c['cell_type']=='markdown' and n.startswith('### 6c'):
        n=n.replace("the H1 bias","the H2 bias")
    if c['cell_type']=='markdown' and n.startswith('## 3c'):
        n=n.replace("converts the frozen 20 pulses per granule from ITRF2014 at the flight epoch to NAD83(2011) epoch 2010.0 and NAVD88 (GEOID18). Its NAVD88 height is compared with the H2 PROJ pipeline above.","converts the frozen 20 pulses per granule from ITRF2014 at the flight epoch to NAD83(2011) epoch 2010.0 (HTDP propagation included) and NAVD88 (GEOID18). Its NAVD88 height is compared with the H2 PROJ pipeline above, which applies the frame Helmert only and does not propagate to 2010.0; the difference therefore includes the small propagation term quantified by the HTDP checkpoint in Section 3.")
    if n!=t: c['source']=n.splitlines(True)
json.dump(nb,open(q,'w',encoding='utf8'),indent=1,ensure_ascii=False)
