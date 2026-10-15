# CASALS to 3DEP vertical alignment, 2026-10-15

The [executed notebook](../notebooks/L1B/geolocation/vertical_datum_alignment.ipynb) rereads both current H5 granules and streams both audited LAZ references. The [original audit notebook](../notebooks/L1B/geolocation/elevation_reference_audit.ipynb) was renamed without changing its bytes, cells, or saved outputs. Historical output directories remain in place.

## Physical interpretation

[Jeff Lee's meeting statements](meeting/20261015_140029.txt) confirm WGS84, ellipsoidal Refh, EGM2008 tide-free, and available mean-tide corrections. They do not confirm the full processing formula, realization/epoch, all correction signs, or applied status. EGM2008 subtraction is now supported by official model information as well as the prior numerical audit. It produces EGM2008-referenced heights; it does not produce NAVD88 heights.

The preferred target-compatible **provisional** geodetic method remains the original GEOID18/C7 operation because the audited 3DEP products specify NAVD88/GEOID18. Its selection does not depend on which method has the smallest RMSE. The current PROJ pipeline is:

```text
proj=pipeline
step proj=unitconvert xy_in=deg xy_out=rad
step inv proj=vgridshift grids=us_noaa_g2018u0.tif multiplier=1
step proj=utm zone=18 ellps=GRS80
```

This is the available EPSG:4979 WGS84 ensemble to EPSG:6347 + EPSG:5703 operation, with the audited grid SHA-256 `fa9a407ac7ee3f5a3694008e4bcd09ce9cc250452f0c3b11700a4960340abce2`. Its WGS84 to NAD83(2011) frame component is numerically a no-op with 2 m stated accuracy; the overall 2.015 m accuracy is operation metadata. [NOAA specifies NAD83(2011), epoch 2010.0 ellipsoid heights for GEOID18](https://geodesy.noaa.gov/GEOID/GEOID18/geoid18_tech_details.shtml). CASALS realization and coordinate epoch remain unknown, so this available operation is not a verified realization/epoch transformation or an absolute accuracy certification.

No artificial correction is formed from EGM2008 minus GEOID18. `refh - geoid - geoid_free2mean` and its opposite-sign alternative remain explicitly labeled sensitivities. Providing mean-tide fields does not identify the complete ellipsoid-height/geoid-height tide conversion or establish which sign/stage applies to Refh. Atmospheric sensitivities remain separate historical evidence and do not enter transformed heights.

## Fixed sample and reference

References are the audited `MD_Southeast_1_2019` / `MD_Southeast_2019_D20` and `NC_HurricaneFlorence_9_2020` / `NC_HurricaneFlorence_2020_D20` clips. Source metadata, the Maryland XML geoid conflict, and sampled source-to-clip Z preservation remain documented in the [historical report](casals_l1b_validation_report.md).

The existing 0.75 m class-2 grid and robust local-plane interpolation are reused, including the every-10,000th-record holdout, 3 m search, minimum six neighbors, maximum 5 degree slope, and maximum 0.20 m plane residual NMAD. All methods use the same pulses, support mask, and freshly recomputed C7 reference plane. C7 XY is numerically identical to the original XY here. The original full-grid and C7 local-query planes differ by up to 1.51 mm / 1.21 mm; using one C7 plane removes that small reference difference from the new paired comparison. C7 itself exactly reproduces historical supported pulse IDs, heights, and reference planes.

The primary relative comparison uses predeclared field test sites. Calibration uses only the preserved `calibration_offset` spatial half, never the test site or former candidate-selection half. November 12 uses `open_field_12_01` (east field) to `open_field_12_02` (southeast field), with site centers 760.148 m apart. November 18 uses `open_field_01` to `open_field_02`, 858.649 m apart. Disjoint pulse identities and separated spatial bounding boxes are checked. Historical NAIP predates CASALS by 529 / 887 days: these fields have uncertain seasonal cover and are not ground-return truth. Reviewed road interiors are reported separately, with no empirical fit because their high-SNR calibration support fails the existing gate.

## High-quality independent field test

All rows below use SNR >= 5, with no CASALS residual trimming. Residual is CASALS height minus 3DEP. Coverage is matched count divided by all test-ROI pulses in that SNR stratum. All metric units are metres.

| Date | Method | Matched / eligible | Coverage | Mean bias | Median bias | RMSE | Centered RMSE | NMAD | Absolute P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024-11-12 | Native | 445 / 478 | 93.10% | -37.5729 | -37.5633 | 37.5730 | 0.1005 | 0.0876 | 37.7529 |
| 2024-11-12 | EGM2008 | 445 / 478 | 93.10% | -0.7349 | -0.7252 | 0.7417 | 0.1005 | 0.0877 | 0.9151 |
| 2024-11-12 | GEOID18 | 445 / 478 | 93.10% | -1.6705 | -1.6609 | 1.6735 | 0.1005 | 0.0877 | 1.8507 |
| 2024-11-18 | Native | 6,615 / 6,615 | 100% | -38.0164 | -38.0134 | 38.0165 | 0.0893 | 0.0874 | 38.1651 |
| 2024-11-18 | EGM2008 | 6,615 / 6,615 | 100% | -0.6208 | -0.6178 | 0.6272 | 0.0892 | 0.0874 | 0.7695 |
| 2024-11-18 | GEOID18 | 6,615 / 6,615 | 100% | -1.6486 | -1.6456 | 1.6510 | 0.0892 | 0.0874 | 1.7972 |

The geodetic height changes mainly shift the residual center. Centered RMSE and NMAD change negligibly. EGM2008 has a smaller numerical RMSE against 3DEP here, but that does not make its vertical reference NAVD88. The GEOID18 comparison retains a substantial negative systematic bias. The centimeter-scale tide sensitivities do not explain this bias.

Lower-SNR tails remain in all saved metrics:

| Date | C7 test stratum | Matched / eligible | Mean bias | RMSE | Centered RMSE | NMAD |
|---|---|---:|---:|---:|---:|---:|
| 2024-11-12 | SNR >= 3 | 4,274 / 4,852 | -1.4866 | 8.6462 | 8.5174 | 0.1173 |
| 2024-11-12 | All valid | 14,885 / 17,171 | +22.4477 | 83.8434 | 80.7825 | 0.2957 |
| 2024-11-18 | SNR >= 3 | 8,623 / 8,623 | -1.6614 | 1.7257 | 0.4664 | 0.0919 |
| 2024-11-18 | All valid | 8,924 / 8,924 | -0.0651 | 19.0600 | 19.0599 | 0.0948 |

The reviewed road high-SNR C7 check reproduces 14 / 70 test pulses, with mean bias -1.8234 / -1.5713 m and RMSE 1.8262 / 1.5752 m. It is a conditional reference-return-to-ground comparison. It is not pooled with the field test or treated as independent ground-return classification.

## Separate empirical transfer

`offset = median(transformed height - 3DEP)` is fitted in the calibration ROI, then `height_aligned = height_transformed - offset` is applied to the independent field test. The following rows use C7 and SNR >= 5. Offsets are fitted separately by granule and stratum; there is no cross-date or test-data fit.

| Date | Calibration n | Test n | Calibration offset | Test bias before | Test bias after | RMSE before | RMSE after | NMAD before / after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024-11-12 | 94 | 445 | -1.7302 | -1.6705 | +0.0597 | 1.6735 | 0.1169 | 0.0877 / 0.0877 |
| 2024-11-18 | 3,101 | 6,615 | -1.5917 | -1.6486 | -0.0568 | 1.6510 | 0.1058 | 0.0874 / 0.0874 |

A spatially transferable constant explains much of the high-SNR field disagreement, but leaves about +6 cm / -6 cm mean residual. Centered RMSE remains 0.1005 / 0.0892 m exactly under constant subtraction. This is a relative site-transfer diagnostic, not NAVD88 conversion, causal attribution, or independent absolute accuracy. The notebook also reports Native and EGM2008 empirical transfer and every SNR stratum.

## Reproduction and validation

Open the new notebook in the existing `casals-showcase` kernel and run all cells, from the repository root or notebook directory. It calls `research/reference/vertical_alignment.py`; each execution creates a distinct directory under `outputs/notebooks/geolocation/vertical_datum_alignment/runs/`. No historical notebook execution is required. The original audit was not rerun.

Each completed run contains `population.parquet` (all fixed queried pulses and validity flags), `metrics.csv`, `alignment.csv`, `aligned.parquet`, `sites.csv`, `parity.csv`, `geoid.csv`, `operation_audit.json`, `input_sha256.csv`, `final_summary.csv`, and the three actual-data figures. The metrics and alignment tables include pulse-set SHA-256 hashes. The run directory linked below also stores execution and preservation verification.

The NGA EGM2008 check reuses all 1,200 historical coordinates per granule and reproduces prior H5-minus-NGA medians +0.000438 / -0.002074 m to 1e-12 m. NGA's six bundled reference points pass the existing 1 cm interpolation tolerance (maximum 0.009916 m). The C7 check covers all 32,859 / 16,064 supported field pulses, plus 147 / 165 road test pulses, with zero height/reference-plane difference. All six independent field test C7 metric rows reproduce the baseline to 1e-8 m.

Focused validation passed 79 tests, including calibration/test leakage, offset sign, geometry separation, common denominators, retained extreme residuals, and the reused datum/interpolation/correction/ERA5 diagnostics. There is no new physical correction that demonstrably eliminates the residual bias under the available processing information.

Completed run: [20261015T202527182325Z](../outputs/notebooks/geolocation/vertical_datum_alignment/runs/20261015T202527182325Z/). All eight code cells completed without errors; the three actual-data PNGs are embedded in the notebook and were visually inspected. SHA-256 verification preserved all 204 protected files (29755166180 bytes): 182 historical files, two raw H5 files, two LAZ references, 17 production Python files, and the byte-identical renamed audit notebook. See `execution_verification.json`, `preservation_verification.csv`, and `artifact_sha256.csv` in this run.
