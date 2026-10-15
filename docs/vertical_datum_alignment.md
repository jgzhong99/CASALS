# CASALS to 3DEP: physical vertical transformation and closure

Updated 2026-10-15. The [current notebook](../notebooks/L1B/geolocation/vertical_datum_alignment.ipynb) now performs physical height conversion and independent evaluation only. `empirical_alignment()` and its calls, residual-derived height parameters, associated result files, charts, and before/after tables have been removed from the active helper, notebook, and tests. There is no disabled fitting option. The [original audit notebook](../notebooks/L1B/geolocation/elevation_reference_audit.ipynb), historical `vertical_datum_comparison.py` including its historical C8 experiment, and all existing run directories remain unchanged.

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

The [NGS HTDP guide](https://geodesy.noaa.gov/TOOLS/Htdp/HTDP-user-guide.pdf), sections 8–9 and Tables 4–5, was investigated as the official time-dependent route. Identified frames and epochs are prerequisites. NGS cautions that WGS84 transformations are approximate; HTDP must not transform between realizations within one NAD83 plate frame (use NCAT for those). Consequently no realization, epoch, velocity, Helmert parameter set, or HTDP result has been assigned to CASALS by assumption. Exact source-frame provenance is needed before implementing that route.

## Actual C7 operation and missing frame step

The required physical chain is source ellipsoidal XYZ → verified source-frame/epoch transformation → NAD83(2011), epoch 2010.0 ellipsoidal height → GEOID18 → NAVD88:

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

The actual EPSG:4979 WGS84 ensemble → EPSG:6319 NAD83(2011) frame operation is `proj=noop ellps=GRS80`, the inverse of “NAD83(2011) to WGS 84 (1)”. It is **non-ballpark but null**, with three zero translation parameters and 2.0 m stated accuracy. Longitude, latitude, and ellipsoidal height changes are all numerically zero. No time propagation is performed. The combined operation's 2.015 m accuracy is metadata, not an observed error or guarantee. WGS84 and GRS80 ellipsoids are recorded, but this no-op is not evidence of a resolved physical frame transformation.

The current result is always provisional because source realization/epoch, target epoch achievement, and Refh processing application are unverified. The status gate cannot promote a no-op, ballpark operation, or incomplete evidence to verified status.

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
2. A true CASALS WGS84-realization/epoch → NAD83(2011), epoch 2010.0 transformation has **not** been identified or implemented; the actual frame step remains a recorded no-op.
3. Neutral atmosphere and other ranging/tide correction applied states remain unresolved. None is reapplied.
4. The executed physical operations are EGM2008 diagnostic subtraction and the audited GEOID18 inverse grid operation plus UTM projection under the documented provisional source assumption. No newly verified physical correction is available.
5. The high-quality signed GEOID18 biases remain approximately -1.67 / -1.65 m. Strict closure is **not achieved**.
6. Missing evidence is per-granule GNSS/INS realization and coordinate epoch; trajectory/intermediate-frame and epoch propagation records; and the exact Refh range equation with per-field units, signs, application stages and applied flags. No residual fit replaces these records.

## Reproduction and validation

Run all cells of the current notebook with the existing `casals-showcase` kernel, from the repository root or notebook directory. `nptdms` is the already-declared notebook dependency and is required for the metadata audit. Every run gets a new directory under `outputs/notebooks/geolocation/vertical_datum_alignment/runs/<run-id>/`.

Outputs: `population.parquet`, `metrics.csv`, `geoid.csv`, `parity.csv`, `frame.csv`, `operation_audit.json`, `source_frame_audit.json`, `physical_correction_audit.csv`, `input_sha256.csv`, `final_summary.csv`, and two actual-data figures. No empirical alignment result files or charts are generated. Every comparison includes matched counts, eligible denominators, common pulse hashes, signed mean/median bias, RMSE, centered RMSE, NMAD and absolute P95.

Focused tests cover the four-argument physical boundary, lack of fitting/correction entry points, changing 3DEP reference heights without changing CASALS heights, actual H5/geoid/C7 parity for both dates, independent grid sign checks, no-op/ballpark detection, withholding verified status on incomplete evidence, retained residual tails and the bias/RMSE decomposition. Full notebook execution, parity, preservation and visual verification records are recorded with the completed run below.

Completed physical-only run: [20261015T214817464342Z](../outputs/notebooks/geolocation/vertical_datum_alignment/runs/20261015T214817464342Z/). All seven code cells executed in a fresh kernel without errors. Two PNG figures are embedded and visually inspected. All 88 focused tests passed. C7 heights and independently rebuilt reference planes match the historical baseline exactly on 32,859 / 16,064 field pulses and 147 / 165 road test pulses; main test counts remain 445 / 6,615. Both independent four-cell grid checks agree with PROJ within 2.2e-14 m. The pre/post SHA-256 check preserved all 1376 files (36449452738 bytes), including all pre-existing output files, raw H5/LAZ, other notebooks and unchanged historical datum/correction modules. See `execution_verification.json`, `acceptance_audit.json`, `preservation_verification.csv`, and `artifact_sha256.csv` in this run. No strict physical closure is claimed.
