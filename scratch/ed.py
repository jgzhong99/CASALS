import json
p=r'notebooks/L1B/geolocation/vertical_datum_alignment.ipynb'
nb=json.load(open(p,encoding='utf8'))
def get(c): return ''.join(c['source'])
def put(c,s): c['source']=s.splitlines(True)
for c in nb['cells']:
    s=get(c)
    if c['cell_type']=='code' and 'spread = table.groupby' in s:
        put(c,'\n'.join(l for l in s.split('\n') if not l.startswith('spread = '))); 
    if c['cell_type']!='markdown': continue
    s=s.replace('CASALS?3DEP','CASALS-3DEP')
    if s.startswith('## 3. Reference-frame'):
        s=s.replace('No HTDP or Helmert correction is run because CASALS source realization and coordinate epoch remain unidentified. Flight date alone does not establish either input condition.',
         'The no-op above is therefore retained only as hypothesis H0 (the data are already NAD83(2011)). Section 3b evaluates the physically rigorous alternatives explicitly. Flight date alone does not establish the source realization, so the choice among hypotheses remains an open metadata question. HTDP is not used because it cannot transform between realizations within one plate frame; the installed EPSG/PROJ 14-parameter operations implement the same global-frame (ITRF -> NAD83(2011) @2010.0) Helmert.')
    if s.startswith('## 4. Physical'):
        s=s.replace('| GEOID18 | Existing C7 XYZ operation | **Provisional physical transformation**, because its source-frame step is unresolved |',
         '| GEOID18 | Existing C7 XYZ operation (H0: frame step is a no-op) | **Provisional**: valid only if CASALS is already NAD83(2011); see Section 3b for H1-H4 |')
    if s.startswith('## 7. Closure'):
        s='''## 7. Closure assessment

**The dominant error is the omitted frame transformation, not the geoid.** Treating the WGS84/ITRF source as NAD83(2011) (H0) leaves a -1.65 to -1.67 m bias. Rigorous time-dependent transformation to NAD83(2011) @2010.0 (H1-H4) raises ellipsoidal heights by about +1.30 to +1.37 m (the Helmert translation plus rotation in Maryland, 0.5-1.0 m horizontal shift) and reduces the independent-test bias to roughly -0.28 to -0.37 m. The result is insensitive to the realization (G1762, ITRF2014, ITRF2020 agree within 0.01 m in bias) and to the epoch assumption (2010.0 vs flight date changes it by about 0.01 m), because the shift is dominated by the static frame offset. The frame shift moves the 3DEP query by about 1 m but changes the sampled reference by only about 0.001-0.003 m on these low-slope fields.

**Strict closure is still not claimed.** (1) The GNSS/INS source realization and epoch are unrecorded, so H1-H4 are conditional assumptions; if the navigation solution is already NAD83(2011) (H0), the large bias remains unexplained. (2) A residual bias of about -0.29 m (20241118) and -0.37 m (20241112) remains, and the 0.08 m difference between granules is larger than the spread among H1-H4. Candidate contributors are the unverified Refh range/atmosphere/tide processing state, 3DEP vertical accuracy and vegetation, none of which is corrected here. No residual-derived correction is applied. Residual-minimizing selection of a hypothesis is deliberately not done.

Required evidence: per-granule GNSS/INS realization and epoch, and the processor's Refh time-to-range equation with field-level units, signs and applied/not-applied records.'''
    put(c,s)
# also update final markdown print cell text
for c in nb['cells']:
    if c['cell_type']=='code' and 'final_columns' in get(c):
        put(c,get(c).replace('provisional GEOID18,','provisional GEOID18 under H0 (no frame change; see Section 3b for H1-H4),'))
json.dump(nb,open(p,'w',encoding='utf8'),indent=1,ensure_ascii=False)
