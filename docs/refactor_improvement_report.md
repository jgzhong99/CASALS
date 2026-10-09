# CASALS Third-Round Refactor Report

Date: 2026-10-09  
Starting revision: `master` at `08a5afe2585ccf934749585a40575c473a735ff9`  
Scope: simplify the research CLI and shared code, separate prediction from
evaluation, add peak geolocation and validation, rebuild the four active
notebooks, and document verified behavior. No commit was created.

## 1. Problems addressed

- The command-line entry point had a wide flat surface and stale command
  spellings. It now exposes three groups and their current leaf commands.
- DSM and tentative DTM duplicated H5/CRS reading and summary code. Those
  common operations now live in `casals_l1b/refh.py`; their distinct masks,
  grids, interpolation, fill limits, and output semantics remain in their
  workflow modules.
- Prediction previously required evaluation inputs. `classify_refh()` now
  predicts from the CASALS H5 alone; `evaluate_classification()` separately
  aligns predictions and reference labels and calculates metrics.
- The peak detector lacked a supported direct API, and detected components
  had no implemented path to coordinate candidates. `extract_components()`
  and `locate_peaks()`/`validate_candidate_table()` now provide that path
  without running the detector again during geolocation.
- The active notebooks were replaced with four independent real-data
  workflows that use package APIs, save explanatory outputs, and state the
  scientific limits.
- Two generic LAS/LAZ viewers duplicated each other. The bounded,
  class-filterable Qt viewer remains alongside the H5-specific refh viewer.

## 2. Files removed or consolidated

- Deleted tracked `backup.bat`; it was an obsolete workstation backup script.
- Deleted `tools/view_lpc_open3d.py`; it duplicated generic LAS/LAZ viewing.
  `tools/view_lpc_qt_classes.py` remains the bounded class-filterable viewer,
  and `tools/view_refh_points.py` remains the refh-specific H5 viewer.
- Consolidated duplicate refh data readers and array-statistics helpers from
  `refh_dsm.py` and `refh_ground.py` into `refh.py`. DSM and DTM algorithms
  remain separate.
- Moved prediction/evaluation alignment and metrics into the new
  `casals_l1b/evaluation.py`; the parameter research script now imports the
  public evaluation module and invokes the nested CLI.
- Preserved all 21 archived research notebooks and their saved outputs under
  `research/archived_notebooks/`. Moved the four superseded synthetic root
  demos into `research/archived_notebooks/superseded_root_demos/`, retaining
  their saved outputs. Exactly four root notebooks remain active; they are the
  real-data workflows listed in section 7.
- `config/local/` and `.compile_tmp/` remain ignored and untracked. No local
  H5, TDMS, reference point cloud, or generated validation output was added to
  Git.

## 3. Supported CLI

Run with `python -m casals_l1b` (or the installed `casals` entry point):

| Group | Commands | Purpose |
| --- | --- | --- |
| `refh` | `export`, `filter`, `dsm`, `ground`, `classify`, `evaluate` | Official refh products, derived surfaces, prediction, and independent evaluation |
| `peaks` | `extract`, `locate`, `validate` | Extract waveform components, geolocate an existing component table, and validate candidate output |
| `reference` | `download`, `diagnose`, `transfer` | Acquire 3DEP clips, diagnose offsets, and transfer experimental pseudo-labels |

The old flat command aliases are not routed. Viewer, animation, and parameter
scan tools remain standalone utilities.

## 4. Prediction and evaluation APIs

- `classify_refh(h5_path, config, reference_laz_path=None)` runs prediction
  with only the input H5 and classifier configuration. It writes predicted
  LAS/LAZ fields including `point_index`, predicted class, reason, class
  counts, and run metadata. Without a reference, metadata records
  `evaluation.status = not_run`; no evaluation metrics are invented.
- `evaluate_classification(prediction, reference, ...)` evaluates an existing
  prediction against separately supplied reference labels using explicit
  point alignment. It reports confusion counts and accuracy, macro-F1,
  weighted-F1, and per-class metrics with population counts.
- `casals_l1b/classification_cli.py` uses the evaluator only when an optional
  reference is supplied. Parameter research imports from
  `casals_l1b.evaluation`, not CLI internals.

## 5. Geolocation methods and real-data results

`locate_peaks(h5_path, component_table, ...)` consumes an existing component
table and checks each pulse index against the H5 sweep and track. It writes
`candidate_points.parquet`, `closure_residuals.csv`,
`validation_summary.csv`, and `run_metadata.json`. Invalid or missing geometry
keeps an explicit status and missing coordinates.

Two experimental methods are implemented:

1. **Segment interpolation:** convert the H5 `rwstart` and `rwstop` geodetic
   endpoints to ECEF and linearly interpolate by
   `peak_bin / (number_of_RX_bins - 1)`.
2. **Beam range:** form the audited down-looking ENU unit vector (azimuth
   clockwise from north; elevation above horizon, reversed toward ground),
   rotate it to ECEF, and offset the official refh coordinate by
   `(peak_bin - raw_argmax_bin) * bin_size`. This is explicitly
   refh-anchored.

On the real 2024-11-18 H5, sweeps 5000–5002, the component table contained
3,064 rows from 768 pulses. `peaks locate` produced 3,064 finite candidates
for each method; `peaks validate` passed pulse identity checks with zero
sweep/track mismatches. Raw-argmax refh closure was available for 766 pulses
(two pulses had no usable component row): 3D median 0.01659 m, P90 0.01764 m,
P95 0.02860 m; horizontal median 0.00066 m, P95 0.00112 m; vertical median
-0.01652 m. Among 3,064 paired component candidates, beam/segment ECEF
separation had median 0.05746 m, P90 0.20781 m, P95 0.23645 m.

These are internal consistency diagnostics. The closure uses raw waveform
argmax against the H5 refh fields; the beam method uses refh as its anchor.
Neither demonstrates independent absolute accuracy or validates all secondary
returns as ground coordinates.

## 6. Real-data numerical comparisons

All comparisons below use local CASALS inputs and outputs. They do not change
detector thresholds or choose a preferred model.

- **Shared refh reader:** compared with the pre-refactor DSM and ground
  readers on both local 3,604,480-record H5 files. Longitude, latitude,
  height, SNR, amplitude, threshold, quality mask, track, sweep, pulse index,
  attributes, and summaries were exactly equal.
- **Surface workflows:** on matched inputs, direct package API outputs
  reproduced all six notebook DSM/DTM rasters exactly: pixels, CRS, affine
  transform, and nodata. The 2024-11-18 DSM selected 1,952,951 points and
  produced a 2 m, 730 x 1,282 EPSG:32618 grid; 41,503 cells were strictly
  observed and 4,967 were support-limited IDW fills. The 2024-11-12 tentative
  DTM used 41,075 ground candidates, 1,656 non-ground points, and 9 low
  outliers; 18,508 cells were valid. These are refh-derived products, not an
  independent terrain validation.
- **Classification same-CRS parity:** with the archived transferred reference
  and matching EPSG:6347 CRS, the refactored run exactly matched the existing
  validation output for XYZ, predicted class, point index, reason, HAG,
  validity/evaluation fields, parameters, and evaluation metrics.
- **Separate evaluation CLI:** evaluating the H5-only prediction against the
  current transferred pseudo-reference aligned all 3,604,480 points and
  returned accuracy 0.937062, macro-F1 0.867278, weighted-F1 0.933758. This
  population includes 1,996,088 far-no-match pseudo-label assignments. For
  the 1,608,392 supported pseudo-reference points, accuracy was 0.886491,
  macro-F1 0.553444, and weighted-F1 0.881867. These values describe
  agreement with transferred 3DEP labels, not independently surveyed truth.
- **Peak detector:** current map environment results for sweeps 5000–5002
  were 3,064 components. The historical record was 3,068; for the known
  near-tie at sweep 5000 / track 214, the historical bin was 1694 and the
  current result is 1693. The earlier NumPy/SciPy environment is unavailable,
  so this environment-sensitive discrepancy remains unresolved and was not
  tuned away.

The no-reference prediction uses the H5's native EPSG:32618 coordinates. A
separate same-CRS parity run above uses EPSG:6347 on both sides. Geometry/HAG
from unlike-CRS runs is not used as an algorithm parity claim.

## 7. Active notebooks and figures

All four notebooks were executed from clean kernels against local real data;
each is self-contained and calls package workflows rather than relying on
another notebook's state.

| Notebook | Contents and principal saved outputs |
| --- | --- |
| `01_refh_quality.ipynb` | Refh quality counts/distributions, filtered point map, DSM support/filled-surface maps, tentative DTM and ground-candidate diagnostics |
| `02_refh_classification.ipynb` | H5-only prediction and separate pseudo-reference evaluation, prediction/label maps, confusion and population summaries |
| `03_waveform_analysis.ipynb` | Real RX waveform matrices, component examples, peak/SNR summaries, and sweep-level detector counts |
| `04_peak_geolocation.ipynb` | Real segment/beam candidate examples and maps, status counts, closure residual distributions, and method-discrepancy plots |

Notebook 01 uses already prepared local refh point/raster products where
available and prints the package commands needed to create missing products.
It does not depend on another active notebook.

## 8. Validation record

Final checks were run after the current source and test edits:

- `python -m pytest -q` — **PASS**, 18 tests.
- `python -m compileall -q casals_l1b research tools tests` — **PASS**.
- Root help, all three group helps, and all 12 leaf command helps — **PASS**.
- Four clean-kernel notebook executions — **PASS**; outputs are saved in the
  notebooks. Refh API raster parity and classification same-CRS parity —
  **PASS**.
- Real `peaks extract`, `peaks locate`, and `peaks validate` workflows —
  **PASS** for sweeps 5000–5002.
- `git diff --check` and local-only/tracked-path scans — **PASS**.

## 9. Limits and items not verified

- Geolocation methods are research candidates, not an official CASALS
  multi-return product. No independent control points or field survey were
  used; absolute geolocation accuracy is **NOT VERIFIED**.
- Refh closure is an internal raw-argmax comparison. Beam range is
  refh-anchored, so closure is not independent validation.
- Transferred 3DEP classes are pseudo-reference labels. Far-no-match
  assignments and transfer uncertainty limit interpretation of evaluation
  scores; the numbers above are not accuracy certification.
- The peak near-tie / component-count difference is **NOT RESOLVED** across
  historical and current scientific-library environments.
- Interactive Qt viewer behavior and a live 3DEP download were not exercised
  in this validation pass. Synthetic geometry unit tests and real offline
  workflow executions were run.
- Refh quality masks and tentative DTM conventions remain workflow outputs;
  the terrain interpretation and external absolute accuracy are **NOT
  VERIFIED**.
