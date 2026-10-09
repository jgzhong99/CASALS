# 3DEP reference frames and pseudo-reference agreement

CASALS Refh longitude/latitude are treated as WGS84, and `refh` as WGS84
ellipsoidal height (the [CryoCloud L1B tutorial](https://book.cryointhecloud.com/l1b-waveforms-tutorial/)
convention). The original H5 does not fully identify the realization, coordinate
epoch, or processor corrections. This is a pending verification assumption,
not an independently certified source frame.

## Evidence and conversion

| Local granule | LAS horizontal CRS | Source vertical evidence | Geoid evidence |
| --- | --- | --- | --- |
| 2024-11-12 | EPSG:6347, NAD83(2011) / UTM 18N | EPSG:5703, NAVD88 height, in MD_Southeast_1_2019 work-unit fields copied into the download sidecar | GEOID18 in the same metadata |
| 2024-11-18 | EPSG:6347, NAD83(2011) / UTM 18N | EPSG:5703, NAVD88 height, in NC_HurricaneFlorence_9_2020 work-unit fields copied into the download sidecar | GEOID18 in the same metadata |

Evidence paths and metadata links are recorded separately for each granule in
`outputs/reference_crs_validation/*_summary.json`. Both LAS headers are **2D**;
EPSG:6347 alone says nothing about Z. Live USGS EPT JSON for both resources
advertises EPSG:3857 without a vertical component. Source NAVD88/GEOID18
therefore comes from recorded work-unit metadata, not the EPT projection or
project name. Missing vertical evidence stays `vertical_crs_unknown`.
Conflicting vertical claims or a sidecar/header horizontal discrepancy are rejected.

Historical clips used `filters.reprojection` with only `out_srs`. Its name does
not restrict it to XY: [PDAL reprojection](https://pdal.io/en/2.9.1/stages/filters.reprojection.html)
can transform Z when the CRS operation involves height. The historical
`vertical_datum_transform_applied=false` flag was hard-coded, without an actual
operation/grid/source-to-output Z audit. Both advertised CRSs here are 2D,
which suggests Z passthrough, but does not establish the historical processing
with the exact old software. Source vertical metadata is confirmed; historical
clip processing remains unverified.

Future download pipelines explicitly use 2D source/output CRSs, copy source Z
to a temporary dimension, and [restore Z](https://pdal.io/en/2.9.1/stages/filters.ferry.html)
before writing. The temporary dimension is excluded from LAS extras. Sidecars
record the pipeline, PDAL metadata, source/output WKT, work-unit vertical/geoid
claims, and measured restored Z differences before LAS quantization. This is
horizontal reprojection with source Z preserved, not a full XYZ datum conversion.
No existing download was rerun or replaced.

Complete XYZ conversion reuses the CRS diagnostics in
`research/reference/diagnose_3dep_offsets.py`: select the first audited
`TransformerGroup` operation with `always_xy=True`, `allow_ballpark=False`;
require complete, verified source/target frames and the best operation to be
available. Apply that same transformer, record its PROJ pipeline, selected
and unavailable alternative grids, accuracy metadata, and observed Z changes.
Unavailable alternatives alone do not invalidate an available best operation.
[PROJ operation selection](https://proj.org/en/stable/operations/operations_computation.html)
is not equivalent to choosing a formula by residual fit. Unchanged Z can be
legitimate between verified frames with the same vertical datum; a 2D CRS
cannot establish this.

Strict XYZ conversion is currently blocked by CASALS source-frame confirmation,
historical clip Z audit, and missing necessary PROJ grids. Local AOI queries
report `best_available=False`; unavailable grid alternatives include
`us_noaa_g1999u08.tif`, GEOID18 `us_noaa_g2018u0.tif`, and NADCON grids.
This list is not a prescription to install every alternative: re-audit the
operation after confirming the source realization and intended datum path.
No grids were downloaded automatically.

## Transfer and evaluation

There are two transfer semantics:

- `verified_crs` is the default. Both complete frames must be confirmed; an
  unavailable required operation/grid stops matching. It does not fit dz.
- `empirical_diagnostic` must be explicit. Fit the existing constant ground dz,
  mark `empirical_alignment=true`, and produce exploratory pseudo-labels.
  Fit residuals are fitting diagnostics only. This smoke reports no independent
  elevation residual evaluation. Any future elevation evaluation must use
  spatial regions excluded from fitting.

Transfer status codes are unchanged: 0 far/no match, 1 strict, 2 weak,
3 ambiguous, 4 nonfinite. Far and ambiguous outputs are class 1; nonfinite
coordinates retain independent invalidity evidence and may be class 7.
Nearest distance, vote ratio and neighbor fields remain available. All kNN
coordinates are either in verified complete frames or explicitly empirical
alignment space; exploratory matching does not certify datum agreement.

Evaluation verifies the original H5 path in a LAS provenance VLR, original
record indices, and **all** matched original longitude, latitude, and Refh
coordinates (not empirical output Z). Sparse indices are supported. Missing or
different provenance/coordinates reject alignment; old products without this
evidence need regeneration or an explicit provenance audit. Nonfinite/withheld
references and statuses 0/3/4 are excluded. Default primary results use status 1;
a second result includes statuses 1 and 2. Optional evaluation filters can only
restrict these populations. Missing support produces `evaluation unavailable`,
a reason and counts, without precision/recall/F1 values.

Reports include total and aligned points, strict and strict+weak valid reference
counts, status counts and coverage. All metrics mean **pseudo-reference
agreement**, including the internal `accuracy` fraction. In this baseline,
vegetation/building and other non-ground/non-noise classes map to unclassified.
No new vegetation/building classifier is introduced. Missing DTM alone gives
class 1 with `unclassified_missing_dtm`; other classification rules are preserved.

## Real-data smoke and reproduction

```powershell
python -m pytest -q
python -m compileall -q casals_l1b research/reference tools/download_3dep_lpc.py tests
python research/reference/validate_reference_smoke.py
```

The tested interpreter is the local `map` conda environment. The smoke compares
frozen commit `8acc3e2` with the new implementation on the same 64 middle sweeps
(16,384 Refh records) per granule. It uses the existing `height_only` baseline
and identical CASALS-derived DTM on both sides, not a full production rerun.
3DEP is streamed to select the buffered local footprint: 28,914 points for
11/12 and 209,304 for 11/18, with no selection cap reached. Streaming spatial
selection scans the existing LAZ records; it does not classify the full granule.
Original H5/LAZ and historical products remain read-only. A temporary LAS round
trip verifies original coordinate/provenance identity and is removed afterwards.
Only one compact summary JSON and evaluation CSV per granule are retained.

| Granule | Population | n | Agreement | Macro F1 (fixed 3 classes) | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 11/12 | old all matched | 16,384 | 0.968140 | 0.666876 | 0.977337 |
| 11/12 | new strict | 10,175 | 0.994201 | 0.413171 | 0.993752 |
| 11/12 | new strict+weak | 10,194 | 0.992447 | 0.404141 | 0.992763 |
| 11/18 | old all matched | 16,384 | 0.675293 | 0.631390 | 0.654479 |
| 11/18 | new strict | 4,773 | 0.866331 | 0.543728 | 0.880981 |
| 11/18 | new strict+weak | 8,558 | 0.913531 | 0.545567 | 0.925827 |

Old all-matched metrics included 6,085 / 6,249 unsupported points labeled noise
and 105 / 1,577 ambiguous points forced into dominant classes. New metrics
exclude those populations. Strict coverage is 62.10% / 29.13%; strict+weak
coverage is 62.22% / 52.23%. The changed denominators/classes explain metric
changes; they are not evidence of improved independent accuracy. Fixed-label
macro F1 includes absent reference classes; CSVs also retain present-label F1.
There were no missing-DTM points in these particular smoke samples; the changed
DTM behavior is separately tested. Empirical dz was 37.6984 / 38.0734 m, fitted
on 314 / 246 inliers, and is not a geoid/datum determination.

Validation: 43 tests passed, including missing vertical CRS, full XYZ/vertical
unit conversion, missing required operation/grid, unsupported/ambiguous labels,
invalid reference exclusion, mismatched H5/coordinates, and no-reference Refh
classification. The new PDAL pipeline construction is tested, but PDAL is not
installed in the tested environment, so its end-to-end download execution has
not been verified. No new downloads or independent ground-truth checks were run.
