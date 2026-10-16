import json, uuid
p=r'notebooks/L1B/geolocation/vertical_datum_alignment.ipynb'
nb=json.load(open(p,encoding='utf8'))
G=lambda c:''.join(c['source'])
def put(c,s): c['source']=s.splitlines(True)
def cell(t,src):
    d={'cell_type':t,'id':uuid.uuid4().hex[:8],'metadata':{},'source':src.splitlines(True)}
    if t=='code': d.update(execution_count=None,outputs=[])
    return d
cells=nb['cells']
def idx(prefix): return [k for k,c in enumerate(cells) if c['cell_type']=='markdown' and G(c).startswith(prefix)][0]
# warnings
c1=[c for c in cells if c['cell_type']=='code' and 'run_alignment(RUN_DIR)' in G(c)][0]
put(c1,"import warnings\nwarnings.filterwarnings('ignore', message='Best transformation is not available', category=UserWarning)\n"+G(c1))
# section 5 wording
i5=idx('## 5.')
put(cells[i5],G(cells[i5]).replace('## 5. Independent geodetic software verification','## 5. Numerical verification of the GEOID18 implementation (same-frame)')
 +'\n\n**Scope:** because the input is the assumed NAD83(2011) coordinate, this section verifies the GEOID18 grid implementation only. NGS, VDatum, PROJ and the raster check all read the same GEOID18 model, so their agreement shows implementation consistency, not independent validation of the frame step. The frame step is verified in Section 3c.')
c5=[c for c in cells if c['cell_type']=='code' and 'scale_ratio' in G(c)][0]
s=G(c5)
a=s.index("    max_bias_m = float"); b=s.index("else:\n    conclusion")
s=s[:a]+'''    conclusion = (f"Across both granules, the largest sampled software difference is {software_max_m:.6f} m, so the GEOID18 "
                  "numerical implementation cannot account for any metre-level CASALS-3DEP residual. "
                  "This same-frame agreement does not validate the WGS84-to-NAD83(2011) frame conversion; see Section 3c.")
'''+s[b:]
put(c5,s)
# 3c after 3b code cell
i4=idx('## 4.')
md3c='''## 3c. Independent verification of the frame step (NOAA VDatum)

NOAA VDatum (HTDP-based, an implementation independent of PROJ/EPSG) converts the frozen 20 pulses per granule from ITRF2014 at the flight epoch to NAD83(2011) epoch 2010.0 and NAVD88 (GEOID18). Its NAVD88 height is compared with the H2 PROJ pipeline above. VDatum uncertainty is reported by the service (about 0.06 m for the vertical datum conversion) and rounds heights to 1 mm. Failed requests are kept as failures.'''
code3c='''V = vdatum_itrf_check(RUN_DIR / 'vdatum_itrf2014')
display(Markdown('**Table 3c-1 - H2 (ITRF2014 @ flight epoch) NAVD88 height: PROJ minus VDatum; metres**'))
display(V['summary'].round(4))'''
cells[i4:i4]=[cell('markdown',md3c),cell('code',code3c)]
# import
c3b=[c for c in cells if c['cell_type']=='code' and 'run_frame_hypotheses' in G(c)][0]
put(c3b,G(c3b).replace('import run_frame_hypotheses','import run_frame_hypotheses, block_bootstrap_bias, vdatum_itrf_check'))
# section 6 additions after sec6 table cell (the one with metric_columns) -> insert before '## 7.'
i7=idx('## 7.')
md6='''### 6b. Primary result under the rigorous frame transformation

H1-H4 are reported as a range; H0 (no frame change) is shown only as the comparison. Intervals are 95% moving-block bootstrap intervals (blocks of 5,000 consecutive pulse indices) for the mean bias, which accounts for along-track correlation. They cover sampling variability only, not the unresolved frame assumption or systematic terms. All values are evaluation only; no hypothesis is selected by its residual.'''
code6='''hyp_all = F['metrics']
primary = hyp_all.query("population == 'field'")
cols = ['granule_id', 'stratum', 'hypothesis', 'n', 'coverage', 'mean_bias_m', 'median_bias_m', 'RMSE_m', 'cRMSE_m', 'NMAD_m', 'absolute_P95_m']
display(Markdown('**Table 6b-1 - Field independent test, all SNR strata, H1-H4**'))
display(primary[cols].round(4).reset_index(drop=True))
display(Markdown('**Table 6b-2 - Road reviewed-interior check (separate population), H1**'))
display(hyp_all.query("population == 'road' and hypothesis == 'H1_WGS84_G1762_obs_epoch'")[cols].round(4).reset_index(drop=True))
ci = pd.concat([block_bootstrap_bias(F['pulses'], hypothesis=h) for h in F['pulses'].hypothesis.unique()])
display(Markdown('**Table 6b-3 - Mean bias with 95% block-bootstrap interval, field, SNR >= 5**'))
display(ci.round(4).reset_index(drop=True))
import matplotlib.pyplot as plt
pl = F['pulses']; base_pop = R['population']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
for ax, (gid, g) in zip(axes, base_pop.groupby('granule_id', sort=True)):
    g0 = g.loc[g.population.eq('field') & g.research_role.eq('independent_test') & g.common_support_all_candidates & g.refh_snr.ge(5)]
    v = np.sort(g0.GEOID18_residual_m); ax.plot(v, np.arange(1, len(v)+1)/len(v), label='H0 no frame change', color='tab:red')
    g1 = pl.loc[pl.granule_id.eq(gid) & pl.hypothesis.eq('H1_WGS84_G1762_obs_epoch') & pl.population.eq('field')
                & pl.research_role.eq('independent_test') & pl.supported & pl.refh_snr.ge(5)]
    v = np.sort(g1.residual); ax.plot(v, np.arange(1, len(v)+1)/len(v), label='H1 WGS84 G1762 -> NAD83(2011)', color='tab:green')
    ax.axvline(0, color='grey', ls=':'); ax.grid(alpha=.2); ax.legend()
    ax.set(title=f"{gid[11:19]} | SNR >= 5", xlabel='CASALS - 3DEP (m)', ylabel='Cumulative fraction')
fig.savefig(F['output'] / 'h0_vs_h1_residual_cdf.png', dpi=160); plt.show()'''
md6c='''### 6c. Diagnostic: remaining bias versus stored tide terms (not applied)

The two granules have different `tide_earth` medians. The table shows what the H1 bias would be if that field were added to or subtracted from Refh. This is a **diagnostic hypothesis test that is not applied**: the field's sign, units and applied state are unverified, and two granules cannot validate a relationship. It only identifies what additional granules and processor documentation should test.'''
code6c='''te = audit.query("field == 'tide_earth'").set_index('granule_id').median_numeric
h1 = ci.query("hypothesis == 'H1_WGS84_G1762_obs_epoch'").set_index('granule_id').mean_bias_m
diag = pd.DataFrame({'tide_earth_median': te, 'H1_bias_as_is': h1, 'if_tide_added': h1 + te, 'if_tide_subtracted': h1 - te})
display(diag.round(4))
display(Markdown('Granule-to-granule bias difference: as is %.3f m; tide added %.3f m; tide subtracted %.3f m. Two granules only: **not applied, not a finding**.' % tuple(abs(float(diag[c].iloc[0]-diag[c].iloc[1])) for c in ['H1_bias_as_is','if_tide_added','if_tide_subtracted'])))'''
cells[i7:i7]=[cell('markdown',md6),cell('code',code6),cell('markdown',md6c),cell('code',code6c)]
# make sure numpy imported before: cell 'import numpy as np' is in section 5 -> fine
# final summary cell: replace with H1-primary
cf=[c for c in cells if c['cell_type']=='code' and 'final_columns' in G(c)][0]
put(cf,'''final = ci.query("hypothesis == 'H1_WGS84_G1762_obs_epoch'").merge(
    F['metrics'].query("population == 'field' and stratum == 'snr_ge_5' and hypothesis == 'H1_WGS84_G1762_obs_epoch'")[['granule_id', 'coverage', 'RMSE_m', 'cRMSE_m', 'NMAD_m', 'absolute_P95_m']], on='granule_id')
rng = ci.query("hypothesis != 'H0'").groupby('granule_id').mean_bias_m.agg(['min', 'max'])
final['bias_range_H1_to_H4_m'] = [f"{rng.loc[g,'min']:+.3f} to {rng.loc[g,'max']:+.3f}" for g in final.granule_id]
final['H0_bias_m'] = R['final_summary'].set_index('granule_id').mean_bias_m.reindex(final.granule_id).to_numpy()
final['strict_closure'] = False
display(final.round(4))
for row in final.itertuples():
    display(Markdown(f"**{row.granule_id[11:19]}:** H1 (WGS84 G1762 -> NAD83(2011) @2010.0 -> GEOID18), n={row.n:,}; signed bias {row.mean_bias_m:+.4f} m (95% block CI {row.ci95_low_m:+.3f} to {row.ci95_high_m:+.3f}), H1-H4 range {row.bias_range_H1_to_H4_m} m, RMSE {row.RMSE_m:.4f} m, centered RMSE {row.cRMSE_m:.4f} m, NMAD {row.NMAD_m:.4f} m. H0 (no frame change) bias was {row.H0_bias_m:+.4f} m. Strict closure: **not achieved**; source frame unverified; no residual-derived correction applied."))''')
# fix ci defined in code6 before final: ok. hypothesis filter 'H0' prefix: ci has no H0 rows
# header & section 4/1 text
c0=cells[0]; put(c0,G(c0).replace('| Coordinate epoch | Unknown | 2010.0 for GEOID18 input | No verified source-to-target propagation |','| Coordinate epoch | Unknown (flight date used in hypotheses) | 2010.0 for GEOID18 input | Explicit hypotheses H1-H4 in Section 3b; not verified |'))
i4=idx('## 4.'); put(cells[i4],G(cells[i4]).replace('| EGM2008 |','| EGM2008 |'))
json.dump(nb,open(p,'w',encoding='utf8'),indent=1,ensure_ascii=False)
