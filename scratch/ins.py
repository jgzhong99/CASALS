import json, uuid
p=r'notebooks/L1B/geolocation/vertical_datum_alignment.ipynb'
nb=json.load(open(p,encoding='utf8'))
md='''## 3b. Explicit frame hypotheses (replacing the no-op as the working assumption)

The PROJ result above treats the *WGS84 ensemble* and NAD83(2011) as identical (2 m stated accuracy). Physically they are different 3-D frames: NAD83(2011) is plate-fixed at epoch 2010.0, whereas WGS84/ITRF coordinates observed in 2024 are in a global frame at the observation epoch. Each plausible source frame is therefore transformed rigorously: ellipsoidal (lon, lat, h) -> geocentric XYZ -> time-dependent 14-parameter Helmert at the flight epoch -> NAD83(2011) epoch 2010.0 (GRS80) -> GEOID18. The parameters are the installed EPSG/PROJ published operations (NGS-derived); none is estimated from 3DEP. The horizontal shift also moves the 3DEP query position, so the reference is re-queried at the transformed coordinates. Each transformation is checked by a 3-D inverse round trip (< 1 micrometre).

| Hypothesis | Source frame | Source epoch |
|---|---|---|
| H0 | Already NAD83(2011) (the previous no-op) | n/a |
| H1 | WGS84 (G1762) | flight date |
| H2 | ITRF2014 | flight date |
| H3 | ITRF2020 | flight date |
| H4 | ITRF2014 | 2010.0 (epoch sensitivity) |

These are **mutually exclusive assumptions about unrecorded metadata, not corrections**. The notebook does not choose among them using residuals. WGS84 G2139 (the operational realization in late 2024) is absent from the installed EPSG database; G1762 and ITRF2014/2020 agree with it at the cm level in height. The conclusions that follow are conditional on the source being an ITRF-aligned (non-NAD83) solution, which only the GNSS/INS processing record can confirm.'''
code='''from research.reference.frame_hypotheses import run_frame_hypotheses
F = run_frame_hypotheses(R, RUN_DIR / 'frame_hypotheses')
display(Markdown('**Table 3b-1 - Frame-induced coordinate changes relative to the no-op (medians; metres)**'))
display(F['shifts'].round(4))
base = R['metrics'].query("population == 'field' and stratum == 'snr_ge_5' and method == 'GEOID18'").assign(hypothesis='H0_noop_NAD83_2011')
hyp = F['metrics'].query("population == 'field' and stratum == 'snr_ge_5'")
cols = ['granule_id', 'hypothesis', 'n', 'coverage', 'mean_bias_m', 'median_bias_m', 'RMSE_m', 'cRMSE_m', 'NMAD_m']
table = pd.concat([base[cols], hyp[cols]]).sort_values(['granule_id', 'hypothesis'])
display(Markdown('**Table 3b-2 - Independent 3DEP field test, SNR >= 5, by frame hypothesis (evaluation only)**'))
display(table.round(4).reset_index(drop=True))
spread = table.groupby('granule_id').mean_bias_m.agg(lambda s: s.drop(s.index[[]]).max() - s.min())
H1234 = table[~table.hypothesis.str.startswith('H0')]
display(Markdown('Bias spread among H1-H4 per granule (m): ' + ', '.join(f"{g[11:19]} {v:.4f}" for g, v in H1234.groupby('granule_id').mean_bias_m.agg(lambda s: s.max() - s.min()).items())))'''
def cell(t,src):
    d={'cell_type':t,'id':uuid.uuid4().hex[:8],'metadata':{},'source':src.splitlines(True)}
    if t=='code': d.update(execution_count=None,outputs=[])
    return d
i=[k for k,c in enumerate(nb['cells']) if c['cell_type']=='markdown' and ''.join(c['source']).startswith('## 4.')][0]
nb['cells'][i:i]=[cell('markdown',md),cell('code',code)]
json.dump(nb,open(p,'w',encoding='utf8'),indent=1,ensure_ascii=False)
print(i)
