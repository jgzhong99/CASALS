# CASALS to 3DEP: physical vertical transformation and closure

Updated 2026-10-15. The [current notebook](../notebooks/L1B/geolocation/vertical_datum_alignment.ipynb) preserves the physical-only H0 baseline and adds conditional, explicitly 3D reference-frame hypotheses, fixed-point NGS HTDP verification, and independent evaluation. `empirical_alignment()` and its calls, residual-derived height parameters, associated result files, charts, and before/after tables have been removed from the active helper, notebook, and tests. There is no disabled fitting option. The [original audit notebook](../notebooks/L1B/geolocation/elevation_reference_audit.ipynb), historical `vertical_datum_comparison.py` including its historical C8 experiment, and all existing run directories remain unchanged.

**Strict physical transformation and CASALS–3DEP height closure have not been achieved.** The available C7 calculation is a reproducible **provisional physical transformation**. No newly verified frame or ranging correction was found, and no additional atmospheric, range-bias, or tide term is applied.

## Evidence and actual call chain

The notebook calls `research/reference/vertical_alignment.py:run_alignment()`:

1. `audit_source_metadata()` rereads root/field attributes and the twelve scalar correction fields from both current H5 files, plus metadata only from both local TDMS files.
2. `transform_source_heights(longitude_deg, latitude_deg, refh_m, geoid_m)` accepts only these four CASALS arrays. It calls the preserved `vertical_datum_comparison.transform_c7()` and audits the separate EPSG:4979 to EPSG:6319 operation. It has no reference-height, residual, fitted-parameter, or optional-correction input.
3. The existing `build_local_class2_query_grid()` and `interpolate_ground_surface()` rebuild the independently queried 3DEP surface. Only `evaluate_heights()` then calculates `height - reference`. Reference values cannot feed back into transformed heights.
4. `summarize()` reports the complete, common pulse populations and signed residual metrics. `plot_results()` shows only the physical height residual distributions and uncorrected spatial residuals.

The active helper never calls the historical candidate builder, selector, empirical functions, or C8. Frozen legacy split labels remain pulse-membership evidence, not instructions to fit parameters. Historical baseline heights and references are consulted only for post-computation numeric parity.

## Source-frame investigation

The [2026-10-15 meeting](meeting/20261015_140029.txt), especially Jeff Lee's statements at lines 195–213, identifies WGS84, EGM2008 tide-free, and product mean-tide conversion fields. The supplied meeting convention identifies Refh as ellipsoidal height. It does not establish a specific WGS84 realization, ITRF/IGS mapping, coordinate epoch, intermediate navigation frame, complete Refh formula, or field application state.

Both H5 root attribute sets were reread. They contain observation times, `delta_time`, a `casals_epoch_utc_a` time origin of 2018-01-01, and names of upstream files. A time-counter origin and a flight timestamp do not establish the navigation coordinate epoch. The exact upstream records are absent from the local `data/` tree:

| Date | GPS source named in H5 | Geolocation source named in H5 | TDMS source named in H5 |
|---|---|---|---|
| 2024-11-12 | `CASALS_GPS_2024-11-12_10-28-49.csv` | `casals_geolocation_20241112T165718_001_01.h5` | `casals_12-11-2024_11-58-21_22.tdms` |
| 2024-11-18 | `CASALS_GPS_2024-11-18_09-43-48.csv` | `casals_geolocation_20241118T171757_001_02.h5` | `casals_18-11-2024_12-18-37_136.tdms` |

The named L1A/ARD records are also absent. The two available TDMS files (`13-34-53_158-003` and `13-37-23_164-002`) have scan/timing properties and 256 channels each, with empty channel property sets; they provide no GNSS reference-frame or coordinate-epoch contract and do not match the H5 source names. Only metadata was read. The known partial-tail warning for the second TDMS does not supply missing navigation provenance.

The [NASA CASALS design paper](https://ntrs.nasa.gov/api/citations/20240001267/downloads/Conference%20Paper.pdf), page 3, describes POS 510 GNSS/INS hardware. This does not establish a per-granule solution frame, epoch, or applied correction record. The linked CryoCloud tutorial could not be retrieved in this investigation; its earlier descriptions remain historical evidence, not new field-level confirmation.

The [NGS HTDP guide](https://geodesy.noaa.gov/TOOLS/Htdp/HTDP-user-guide.pdf), sections 8?9 and Tables 4?5, identifies official global-frame operations and separate coordinate-epoch propagation. A bounded fixed-point check runs HTDP 3.6.0 for H5 pulse 0 in each granule under explicit WGS84(G2296), ITRF2014, and ITRF2020 assumptions; the ITRF2014 case is also propagated to NAD83(2011), epoch 2010.0. This is conditional model verification, not evidence assigning any realization or epoch to CASALS. NGS cautions that WGS84 transformations are approximate; HTDP must not transform between realizations within one NAD83 plate frame (use NCAT for those).

## Actual C7 operation and missing frame step

The required physical chain is source ellipsoidal XYZ ? verified source-frame transformation ? coordinate propagation to NAD83(2011), epoch 2010.0 ? GEOID18 ? NAVD88:

```text
H_NAVD88 = h_NAD83(2011),2010.0 - N_GEOID18
```

[NOAA explicitly requires that frame and epoch for GEOID18 input](https://geodesy.noaa.gov/GEOID/GEOID18/geoid18_tech_details.shtml). The current C7 operation is:

```text
proj=pipeline
step proj=unitconvert xy_in=deg xy_out=rad
step inv proj=vgridshift grids=us_noaa_g2018u0.tif multiplier=1
step proj=utm zone=18 ellps=GRS80
```

The preserved H0 EPSG:4979 WGS84 ensemble ? EPSG:6319 NAD83(2011) frame operation is `proj=noop ellps=GRS80`, the inverse of ?NAD83(2011) to WGS 84 (1)?. It is **non-ballpark but null**, with three zero translation parameters and 2.0 m stated accuracy. Longitude, latitude, and ellipsoidal height changes are all numerically zero. No time propagation is performed. The combined operation?s 2.015 m accuracy is metadata, not an observed error or guarantee. WGS84 and GRS80 ellipsoids are recorded, but this no-op is not evidence of a resolved physical frame transformation. New H1-H4 routes are isolated conditional diagnostics described below; they leave this H0 C7 baseline intact.

The H0 result remains provisional because its source realization/coordinate epoch, target epoch achievement, and Refh processing application are unverified. The new H1-H4 experiments are recorded as conditional reference-frame hypotheses and do not promote H0 or establish the source metadata.

## GEOID18 grid and independent sign check

The previously audited grid is reused, without replacement:

- Name: `us_noaa_g2018u0.tif`.
- SHA-256: `fa9a407ac7ee3f5a3694008e4bcd09ce9cc250452f0c3b11700a4960340abce2`.
- Grid CRS: EPSG:6319; target: EPSG:5703; applicable area: CONUS.
- The signed grid undulation N is subtracted through the inverse vertical grid operation. A negative local N raises the numerical orthometric height relative to h.

`grid_point_check()` independently reads four GeoTIFF cells and bilinearly interpolates N at one real source location per granule, then checks `h - N` against PROJ to 1e-7 m. It does not use PROJ to obtain the comparison N. `operation_audit.json` stores each point's longitude, latitude, input ellipsoidal height, signed N, independently calculated H, PROJ H, difference, grid extent, and the source-frame caveat. This proves numerical direction and sign only; the unverified CASALS frame does not become NAD83(2011) merely because interpolation succeeds.

No EGM2008–GEOID18 difference is turned into an extra correction.

## Refh processing audit

`physical_correction_audit.csv` records, separately for both granules, official definition or unresolved status, units and H5 units attributes, affected quantity, sign, applied state, expected height effect, current action, valid count, median, minimum, maximum, and source attributes for all twelve requested fields.

| Field or group | Independent evidence | Current action |
|---|---|---|
| `refh` | WGS84 ellipsoidal reference-return height convention | Preserve as input |
| `geoid` | Meeting-confirmed EGM2008 tide-free; NGA numeric audit | Subtract only for the separate EGM2008-referenced diagnostic |
| `geoid_free2mean` | Mean-tide geoid correction exists; numeric formula match; complete sign/stage unknown | Not applied |
| `refh_neutat_delay_total` | Full-granule numeric medians 1.593985 / 1.180952; units, range equation, sign and applied status unconfirmed | Not applied |
| `refh_neutat_delay_derivative` | Approximately 0.00028 numeric derivative; field contract unconfirmed | Not applied |
| `range_bias_correction` | Stored zero; calibration versus placeholder/applied zero unresolved | Not applied |
| `tide_earth`, `tide_earth_free2mean`, `tide_load`, `tide_ocean`, `tide_ocean_pole`, `tide_pole` | No verified field-specific units/sign/application chain | Not applied |

Atmospheric optical-path excess, one-/two-way time, slant-range correction, and vertical height correction are not interchangeable. If an omitted positive excess slant range were independently confirmed, the first-order height effect would be `-delta_range * dot(sensor_to_target_beam, local_up)`. Here neither the ranging equation nor the unit/sign/application premise is confirmed, so that hypothetical expression is not executed. The existing ERA5 analysis supports only a physical scale, not omitted correction status. Matching the residual magnitude does not justify another application.

## Independent 3DEP evaluation

The two audited references remain `MD_Southeast_1_2019` / `MD_Southeast_2019_D20` and `NC_HurricaneFlorence_9_2020` / `NC_HurricaneFlorence_2020_D20`. The existing source-product metadata and sampled Z preservation evidence remain in the [historical report](casals_l1b_validation_report.md). The Maryland generic XML GEOID12B conflict is retained; final product evidence identifies NAVD88/GEOID18.

The original class-2 interpolation is reused: 0.75 m cells, deterministic every-10,000th-LAZ-record reference holdout, robust local plane, at least six neighbors within 3 m, slope no greater than 5 degrees, and plane residual NMAD no greater than 0.20 m. This is reference-surface fitting, not fitting a CASALS height correction. Fixed pulse membership and quality gates are unchanged, all methods use the same recomputed C7 reference heights, and no CASALS residual is removed by size.

Field test ROIs are the main high-quality reporting population, with seasonal-cover uncertainty retained because historical NAIP predates CASALS by 529 / 887 days. Reviewed road cores are a separate conditional likely-ground check; neither SNR nor geographic membership proves a ground return. Legacy non-test field samples are used for numeric baseline reproduction only and never estimate a parameter.

For SNR >= 5, residual is transformed CASALS minus 3DEP; all heights/metrics are metres:

| Date | Method | Matched / eligible | Coverage | Mean bias | Median bias | RMSE | Centered RMSE | NMAD | Absolute P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024-11-12 | Native | 445 / 478 | 93.10% | -37.5729 | -37.5633 | 37.5730 | 0.1005 | 0.0876 | 37.7529 |
| 2024-11-12 | EGM2008 | 445 / 478 | 93.10% | -0.7349 | -0.7252 | 0.7417 | 0.1005 | 0.0877 | 0.9151 |
| 2024-11-12 | Provisional GEOID18 | 445 / 478 | 93.10% | -1.6705 | -1.6609 | 1.6735 | 0.1005 | 0.0877 | 1.8507 |
| 2024-11-18 | Native | 6,615 / 6,615 | 100% | -38.0164 | -38.0134 | 38.0165 | 0.0893 | 0.0874 | 38.1651 |
| 2024-11-18 | EGM2008 | 6,615 / 6,615 | 100% | -0.6208 | -0.6178 | 0.6272 | 0.0892 | 0.0874 | 0.7695 |
| 2024-11-18 | Provisional GEOID18 | 6,615 / 6,615 | 100% | -1.6486 | -1.6456 | 1.6510 | 0.0892 | 0.0874 | 1.7972 |

The geoid changes mostly affect bias, while centered RMSE and NMAD barely change. EGM2008 is a different height reference from NAVD88 and is not selected because it gives lower RMSE. Signed bias and uncentered RMSE remain the headline quantities; centered dispersion is not absolute accuracy.

All-valid and SNR >= 3 tails are retained in the notebook and `metrics.csv`. C7 SNR >= 3 field RMSE remains 8.6462 / 1.7257 m (4,274 / 8,623 pulses); all-valid RMSE is 83.8434 / 19.0600 m (14,885 / 8,924 pulses). The separate high-SNR road check has 14 / 70 pulses, bias -1.8234 / -1.5713 m, RMSE 1.8262 / 1.5752 m, and NMAD 0.0456 / 0.1115 m.

## Closure conclusion

1. The active experiment contains no empirical height-offset estimation or application.
2. Conditional official EPSG/PROJ frame operations and a two-point HTDP epoch-2010 propagation are implemented and numerically checked; the actual CASALS GNSS/INS realization and coordinate epoch remain **unidentified**, so no route can be selected as the true physical transformation. H0 remains the preserved no-op baseline.
3. Neutral atmosphere and other ranging/tide correction applied states remain unresolved. None is reapplied.
4. The executed physical calculations include the EGM2008 diagnostic, the preserved H0 GEOID18 operation, and conditional H1-H4 frame-plus-GEOID18 diagnostics. No additional range, atmospheric, or tide correction is verified or applied.
5. The high-quality signed GEOID18 biases remain approximately -1.67 / -1.65 m. Strict closure is **not achieved**.
6. Missing evidence is per-granule GNSS/INS realization and coordinate epoch; trajectory/intermediate-frame and epoch propagation records; and the exact Refh range equation with per-field units, signs, application stages and applied flags. No residual fit replaces these records.

## Reproduction and validation

Run all cells of the current notebook with the existing `casals-showcase` kernel, from the repository root or notebook directory. `nptdms` is the already-declared notebook dependency and is required for the metadata audit. Every run gets a new directory under `outputs/notebooks/geolocation/vertical_datum_alignment/runs/<run-id>/`.

Outputs: `population.parquet`, `metrics.csv`, `geoid.csv`, `parity.csv`, `frame.csv`, `operation_audit.json`, `source_frame_audit.json`, `physical_correction_audit.csv`, `input_sha256.csv`, `final_summary.csv`, and two actual-data figures. No empirical alignment result files or charts are generated. Every comparison includes matched counts, eligible denominators, common pulse hashes, signed mean/median bias, RMSE, centered RMSE, NMAD and absolute P95.

Focused tests cover the four-argument physical boundary, lack of fitting/correction entry points, changing 3DEP reference heights without changing CASALS heights, actual H5/geoid/C7 parity for both dates, independent grid sign checks, no-op/ballpark detection, withholding verified status on incomplete evidence, retained residual tails and the bias/RMSE decomposition. Full notebook execution, parity, preservation and visual verification records are recorded with the completed run below.

Earlier physical-only run retained as the H0 baseline:  [20261015T214817464342Z](../outputs/notebooks/geolocation/vertical_datum_alignment/runs/20261015T214817464342Z/). All seven code cells executed in a fresh kernel without errors. Two PNG figures were embedded and visually inspected. Its parity, preservation, and validation records remain historical evidence; the current frame/epoch execution below supersedes its notebook validation status while preserving that baseline.

## Independent geodetic software verification

The current notebook adds a fixed-sample software comparison using the existing C7 operation, a separate rasterio interpolation, the [NGS Geoid Height Service](https://geodesy.noaa.gov/web_services/geoid.shtml), and the [NOAA VDatum Full API](https://vdatum.noaa.gov/docs/services.html). The NGS request explicitly uses `model=14` (GEOID18) and negative west longitude. The VDatum request explicitly declares geographic coordinates, NAD83(2011) ellipsoidal input, NAVD88/GEOID18 output, metre height units, the documented service region, and 2010.0 input/output epochs.

The 20 pulse IDs per granule are frozen and asserted by the helper. They were selected by deterministic max-min spatial sampling over each complete H5 stream, using finite Refh and coordinates and SNR >= 5 only. The manifest includes granule, pulse ID/index, spatial rank, coordinates, Refh, SNR, and the explicit numerical-only assumption `Assumed NAD83(2011) coordinates for numerical verification only`. No 3DEP height or residual participates in this selection or in choosing conversion parameters. Every comparison uses the same 20 coordinates per granule. The local C7 operation remains the pinned GEOID18 grid (`us_noaa_g2018u0.tif`, SHA-256 `fa9a407ac7ee3f5a3694008e4bcd09ce9cc250452f0c3b11700a4960340abce2`).

The APIs returned 40/40 valid NGS GEOID18 results and 40/40 valid VDatum results. Their height outputs were rounded to millimetres in the recorded responses; the numerical comparison uses a 0.01 m tolerance. Statistics below are metres; PROJ-NGS and raster-NGS refer to geoid undulation differences, while PROJ-VDatum compares NAVD88 heights.

| Granule | Points / NGS calls | PROJ-NGS mean / RMSE / max abs | Raster-NGS mean / RMSE / max abs | PROJ-raster max abs | Result |
|---|---:|---:|---:|---:|---|
| 2024-11-12 | 20 / 20 | -0.000082 / 0.000279 / 0.000609 | -0.000082 / 0.000278 / 0.000610 | 0.000004 | PASSED |
| 2024-11-18 | 20 / 20 | -0.000081 / 0.000264 / 0.000450 | -0.000082 / 0.000264 / 0.000448 | 0.000005 | PASSED |

| Granule | VDatum calls | PROJ-VDatum mean | RMSE | Max abs | VDatum-NGS max abs | Result |
|---|---:|---:|---:|---:|---:|---|
| 2024-11-12 | 20 / 20 | -0.000011 | 0.000515 | 0.001101 | 0.000497 | PASSED |
| 2024-11-18 | 20 / 20 | -0.000003 | 0.000337 | 0.000625 | 0.000722 | PASSED |

The largest observed difference among the sampled software outputs is 0.001101 m. The unchanged independent C7 field-test residuals at SNR >= 5 are:

| Granule | Matched / eligible | Bias | RMSE | NMAD |
|---|---:|---:|---:|---:|
| 2024-11-12 | 445 / 478 | -1.670506 m | 1.673526 m | 0.087692 m |
| 2024-11-18 | 6,615 / 6,615 | -1.648579 m | 1.650993 m | 0.087435 m |

The largest software difference is about 1,500 times smaller than the remaining bias, so the GEOID18 numerical implementation does not explain the metre-level CASALS-3DEP residual. PROJ and the independent raster agree within 5 micrometres; PROJ and VDatum agree within 1.2 millimetres. This is evidence of numerical consistency under the stated coordinate assumption, not proof that the unknown CASALS source frame was transformed correctly.

The classifications in the preceding 2026-10-15 H0-only section describe its same-frame software check. The current conditional frame and epoch checks, and their limits, are reported in the following section; neither set of numerical checks establishes the CASALS source realization or strict physical closure.

The complete current notebook run is [20261015T230358378823Z](../outputs/notebooks/geolocation/vertical_datum_alignment/runs/20261015T230358378823Z/). All eight code cells executed without errors. Its `geodetic_crosscheck/` directory contains the fixed sample manifest, per-point comparison CSVs, raw NOAA request/response log, summary, and software-difference plot. Successful responses from the actual serial 80-call service check are retained and reused in this complete notebook execution. `crosscheck_summary.json` also records the raw-input manifest, source-code hashes, and a notebook-source hash. The current C7 metric table matches the preceding completed run exactly. A post-run SHA-256 check against the saved protected-file manifest verified all 1,376 listed historical inputs and results unchanged; see [preservation_verification.csv](../outputs/notebooks/geolocation/vertical_datum_alignment/runs/20261015T230358378823Z/preservation_verification.csv).

## Conditional 3D frame and coordinate-epoch results

Current real-data notebook run: [`20261017T015011000000Z`](../outputs/notebooks/geolocation/vertical_datum_alignment/runs/20261017T015011000000Z/). It uses both current H5/LAZ granules and preserves the physical-only H0 result. No residual is used to select an operation or estimate a correction.

### Operation audit and same-epoch verification

All source hypotheses use explicit 3D geographic CRSs: WGS84(G2296) EPSG:10605, ITRF2014 EPSG:7912, and ITRF2020 EPSG:9989. Their installed frame reference epochs are respectively 2024, 2010, and 2015; the selected time-dependent Helmert parameter reference epoch is 2010.0. These are three distinct metadata/time concepts. The operation audit records the full EPSG chain, parameters/rates, coordinate-frame rotation convention, authority, area of use, input/output coordinate time, and inverse round trip. EPSG:10605 is routed through ITRF2020 by EPSG operation 10608, followed by the time-dependent ITRF2020-to-NAD83(2011) operation 10334. ITRF2014 uses operation 8970; ITRF2020 uses 10334. The G2296, ITRF2014, and ITRF2020 operations were available, non-ballpark, explicit, and covered both granule extents. An independent ITRF2014 geocentric XYZ construction agreed with the 3D geographic route within 10 micrometres.

PROJ and NGS HTDP 3.6.0 were compared at the same input and output observation epoch for the actual H5 pulse 0 in each granule. HTDP writes ellipsoidal height to millimetres, so the height differences include that output rounding.

| Source hypothesis | Maximum horizontal difference, two points | Maximum height difference, two points | Result |
|---|---:|---:|---|
| WGS84(G2296) | 0.000006 m | 0.000403 m | Numerically consistent at the same epoch |
| ITRF2014 | 0.000005 m | 0.000208 m | Numerically consistent at the same epoch |
| ITRF2020 | 0.000005 m | 0.000363 m | Numerically consistent at the same epoch |

### Separate coordinate-epoch propagation

For two fixed pulse-0 points only, HTDP applied the ITRF2014-assumption frame route to NAD83(2011) at the observation epoch, then its modelled coordinate propagation to epoch 2010.0. The B-to-C changes are not inferred from H4 and are not a whole-granule propagation result.

| Date | B-to-C horizontal displacement | B-to-C ellipsoidal-height change |
|---|---:|---:|
| 2024-11-12 | 0.032809 m | +0.013 m |
| 2024-11-18 | 0.031209 m | +0.012 m |

The target-height values are reported at millimetre precision by HTDP. The flight timestamps and input-frame assumptions remain conditional because neither H5 provides the navigation coordinate epoch. Thus the model operation is verified for these checkpoints, while a verified CASALS epoch transformation is not established. H4 only supplies 2010.0 as the Helmert parameter-evaluation time for the unchanged input coordinates; its label explicitly says it is not epoch propagation.

### GEOID18 checks at the propagated checkpoints

The two HTDP outputs were passed through the NAD83(2011) C7 route and independently checked with four-cell rasterio interpolation, the NGS GEOID18 service, and NOAA VDatum. Both NGS and both VDatum responses validated. The numerical tolerance is 0.01 m. The largest PROJ-to-NGS geoid difference was 0.000174 m, the largest PROJ-to-raster height difference was 0.00000102 m, and the largest PROJ-to-VDatum NAVD88-height difference was 0.000174 m. These results use the pinned `us_noaa_g2018u0.tif`, SHA-256 `fa9a407ac7ee3f5a3694008e4bcd09ce9cc250452f0c3b11700a4960340abce2`. The final run reuses the validated responses logged with the checkpoint; it makes no broad network sample for this additional check.

### Common-pulse 3DEP results

Primary comparison uses the independent field test population at SNR >= 5. Each row below uses the exact intersection of valid pulses across H0 and every available H1-H4 route. Bias is transformed CASALS minus the independently re-queried 3DEP plane, in metres.

| Date | Hypothesis | Common n | Mean bias | RMSE | NMAD |
|---|---|---:|---:|---:|---:|
| 2024-11-12 | H0 no-op C7 baseline | 427 | -1.6713 | 1.6743 | 0.0896 |
| 2024-11-12 | H1 WGS84(G2296), observation-time assumption | 427 | -0.3681 | 0.3820 | 0.0933 |
| 2024-11-12 | H2 ITRF2014, observation-time assumption | 427 | -0.3687 | 0.3826 | 0.0933 |
| 2024-11-12 | H3 ITRF2020, observation-time assumption | 427 | -0.3682 | 0.3820 | 0.0933 |
| 2024-11-12 | H4 ITRF2014, parameter-time sensitivity at 2010 | 427 | -0.3567 | 0.3709 | 0.0918 |
| 2024-11-18 | H0 no-op C7 baseline | 6,615 | -1.6486 | 1.6510 | 0.0874 |
| 2024-11-18 | H1 WGS84(G2296), observation-time assumption | 6,615 | -0.2893 | 0.3028 | 0.0879 |
| 2024-11-18 | H2 ITRF2014, observation-time assumption | 6,615 | -0.2899 | 0.3033 | 0.0879 |
| 2024-11-18 | H3 ITRF2020, observation-time assumption | 6,615 | -0.2893 | 0.3028 | 0.0879 |
| 2024-11-18 | H4 ITRF2014, parameter-time sensitivity at 2010 | 6,615 | -0.2821 | 0.2960 | 0.0876 |

The exact shared high-SNR field pulse sets are recorded in `frame_hypothesis_common_pulses.csv`: 427 pulses for 2024-11-12 (SHA-256 `cacd54d3c66f382c533089c7c1e72569cac94bb35f65043a67ef46e1e83b7e3a`) and 6,615 for 2024-11-18 (SHA-256 `c876199660cd2607c333d6745898998cd12e0672e24d1aa44b6641bacb61dcc2`). Exact IDs are saved alongside those hashes. The separate own-coverage table retains method-specific support: for 2024-11-12 H0 has 445/478 pulses, H1/H2/H3 each 435/478, and H4 437/478; the common intersection is 427/478. For 2024-11-18 every method has 6,615/6,615. The all-valid and SNR >= 3 strata and road population are also saved.

On the common high-SNR field test sets, the 2024-11-12 minus 2024-11-18 mean-bias difference is -0.022677 m for H0, and -0.078826, -0.078851, -0.078828, and -0.074513 m for H1-H4 respectively. This is a date-level difference in two separate independent-test populations, not a pulse-paired temporal change.

### Paired frame-effect decomposition

For every paired valid field/road pulse, the saved decomposition verifies `delta_residual = delta_H - delta_reference` within 1e-10 m. Medians below are for the independent-test field population, across its SNR strata; query displacement is horizontal.

| Date | Route | Median delta h | Median delta H | Median query displacement | Median delta 3DEP | Median delta residual |
|---|---|---:|---:|---:|---:|---:|
| 2024-11-12 | H2 ITRF2014 at observation time | +1.304344 m | +1.304356 m | 1.179960 m | +0.003752 m | +1.300602 m |
| 2024-11-12 | H4 parameter-time sensitivity at 2010 | +1.315557 m | +1.315567 m | 1.000213 m | +0.002953 m | +1.312610 m |
| 2024-11-18 | H2 ITRF2014 at observation time | +1.360686 m | +1.360709 m | 1.126996 m | +0.002361 m | +1.358348 m |
| 2024-11-18 | H4 parameter-time sensitivity at 2010 | +1.371134 m | +1.371151 m | 0.950382 m | +0.005012 m | +1.366139 m |

H1 and H3 produce nearly the same changes as H2 (within about 0.0007 m in median ellipsoidal-height change). The transformed routes reduce the absolute common-sample bias by about 1.30 to 1.37 m under the ITRF-aligned assumptions. Most of the residual change is in transformed height; the re-queried low-slope 3DEP field-plane median changes are a few millimetres. These residual improvements make an omitted frame conversion a plausible major contributor if the source was global-frame aligned. They do not establish that it was.

### Current evidence classification

- **Numerical transformation verified:** yes, for the explicit installed EPSG operations and the fixed HTDP/XYZ checkpoints; the GEOID18 route also agrees with independent raster, NGS, and VDatum checks.
- **Source-frame hypothesis plausible:** yes. H1-H3 are coherent global-frame assumptions, but the H5 and available source records do not identify the actual GNSS/INS realization.
- **Coordinate-epoch transformation verified for CASALS:** no. HTDP demonstrated the modelled 2010.0 step for two conditional fixed points; the source coordinate epochs and the per-pulse source trajectory provenance are missing.
- **Strict physical CASALS-3DEP closure achieved:** no. The conditional high-SNR biases remain about -0.28 to -0.37 m, and Refh ranging/correction application and independent-reference uncertainty remain unresolved. No fitted bias, residual-selected route, or additional physical correction is applied.

## Signed H0/H2 residual visualizations

The completed notebook run is [20261017T015011000000Z](../outputs/notebooks/geolocation/vertical_datum_alignment/runs/20261017T015011000000Z/). The existing `residual_distributions.png` and `geoid18_spatial_residuals.png` remain the H0 physical-baseline figures in the run root. New comparison figures are saved under `frame_hypotheses/`: `vertical_residual_histograms.png`, `h0_vs_h2_residual_cdf.png`, and `h2_spatial_residuals.png`.

The H0/H2 figures use the exact all-method intersection saved in `frame_hypothesis_common_pulses.csv`, restricted to field independent-test pulses with SNR >= 5. The counts and hashes are 427 / `cacd54d3c66f382c533089c7c1e72569cac94bb35f65043a67ef46e1e83b7e3a` for 2024-11-12 and 6,615 / `c876199660cd2607c333d6745898998cd12e0672e24d1aa44b6641bacb61dcc2` for 2024-11-18. The plot helper checks those IDs against `F['metrics']`, recomputes the displayed statistics, and retains each signed residual without centering or outlier removal. Histograms use density normalization and one Freedman-Diaconis bin definition per method across both dates. The spatial panels use H2's own transformed query coordinates in EPSG:6347, equal aspect, and one symmetric color range that includes all plotted residuals.
