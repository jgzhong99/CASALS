# CASALS L1B Elevation and Waveform Response Validation

This report preserves the original numbered notebook labels and output paths used by those experiments. Current active notebook entry points are listed in [notebooks/README.md](../notebooks/README.md); historical result paths below remain unchanged for provenance.

**Status (2026-10-13):** A GEOID18-consistent 3D CASALS-to-3DEP comparison and a full-granule Refh correction audit are complete for two granules. The selected transformation remains provisional because the CASALS source realization, coordinate epoch, and Refh processing conventions are undocumented. Stored correction fields support numerical model matches and conditional sensitivities, but do not establish applied status. Empirical alignment and land-cover waveform results remain descriptive.

## 1. Scientific questions

1. How do native `Refh`, H5 scalar corrections, a GEOID18-consistent 3D transformation, and a separate empirical alignment compare with the documented 3DEP reference on spatially held-out pulses?
2. Within the two granules, how do H5 Refh SNR, native amplitude, provisional candidate-screen rate, and candidate-waveform morphology vary across historically image-interpreted ROIs and matched track pairs?

The vertical-frame comparison, earlier empirical holdout, and waveform analysis answer different questions. The available 3D path follows the documented 3DEP GEOID18 reference, but its input-frame assumption is unresolved, so its residuals are not certified absolute accuracy. The waveform comparison describes these sampled sites and acquisitions; it does not establish a general land-cover effect.

## 2. Data and methods

The study uses two CASALS L1B H5 granules from 2024-11-12 and 2024-11-18 (3,604,480 pulses each), their paired 3DEP LAZ files, the existing land-cover ROI manifest, and actual NAIP imagery. The NAIP metadata dates are 2023-06-02 for the November 12 granule and 2022-06-15 for November 18. These NAIP images predate the corresponding CASALS acquisitions by 529 and 887 days, respectively. The imagery is historical context, not a contemporaneous ground label.

For elevation comparison, ASPRS class-2 points were streamed into a 0.75 m grid and locally interpolated. Test surfaces required at least six class-2 neighbors within 3 m, slope no greater than 5 degrees, and local plane residual NMAD no greater than 0.20 m. A deterministic sample of every 10,000th LAZ record was withheld from the grid for an independent interpolation check. The check produced plane RMSEs of 0.047 m (385 supported points; 361 acceptable) for November 12 and 0.050 m (6,376 supported; 6,115 acceptable) for November 18. This checks the local interpolation against held-out 3DEP points only; it does not measure CASALS accuracy.

In the initial native-height comparison, H5 longitude/latitude was transformed to EPSG:6347 for horizontal matching, with no vertical transformation. The difference is defined as CASALS H5 `Refh` minus local 3DEP plane height. Raw metrics retain all supported and acceptable observations. Separate empirical offsets are the median raw differences at a predeclared open-field calibration site; residual metrics are computed at a different open-field test site after subtracting that offset. SNR strata are applied consistently to the raw and holdout populations.

The archived clip sidecars link each LAZ to the current H5 basename and point count. They identify `MD_Southeast_1_2019` / `MD_Southeast_2019_D20` for November 12 and `NC_HurricaneFlorence_9_2020` / `NC_HurricaneFlorence_2020_D20` for November 18. Reprojected clip bounding boxes agree with the LAZ headers within 0.605 m and 0.200 m, respectively. The sidecars record horizontal output EPSG:6347 and EPSG:5703/GEOID18 vertical context, but their `vertical_datum_transform_applied=false` flag alone does not prove point-level Z preservation. The new point-identity sample below independently checks Z values against small source EPT nodes.

For waveform response, current H5 RX waveforms, `bg_mean`, `refh_snr`, and `refh_amp` were reread and features recomputed for 128,427 ROI pulse centers. Previous candidate features and flags were not reused as current results. A pulse-by-pulse comparison of previous and recomputed flags found zero gains, losses, or changes across those same pulse IDs. The provisional candidate screen requires a local peak to pass the height, prominence, and width criteria together. A previously computed full-window threshold was reused after checking configuration and current-data consistency; it was estimated from 6,144 real standardized background-tail windows per granule: 10.885 sigma on November 12 and 10.195 sigma on November 18. Simulated null-bootstrap false-alarm rates were 0.0058 and 0.0029. These are noise-only checks under the bootstrap assumptions, not calibrated detection probabilities. Morphology is summarized only for waveforms that pass the candidate screen.

## 3. CASALS-3DEP findings

### Native Refh and earlier full-calibration empirical holdout

The LAZ files contain 9,081,045 points (November 12; 4,646,895 class-2) and 117,530,910 points (November 18; 63,226,508 class-2). Locally supported and acceptable surfaces matched 3,369,562 of 3,604,480 H5 pulses (93.5%) on November 12 and 3,457,681 (95.9%) on November 18.

The raw difference distribution is broad and tail-sensitive. The central values cluster near -38 m, but the vertical-frame audit does not establish that this is a datum offset.

| Granule | SNR stratum | Matched n | Mean difference (m) | Median (m) | RMSE (m) | NMAD (m) |
|---|---|---:|---:|---:|---:|---:|
| 2024-11-12 | All valid | 3,369,562 | -12.164 | -37.571 | 93.741 | 43.237 |
| 2024-11-12 | SNR ≥ 3 | 482,173 | -36.750 | -37.661 | 40.505 | 0.145 |
| 2024-11-12 | SNR ≥ 5 | 39,811 | -37.648 | -37.658 | 37.666 | 0.120 |
| 2024-11-18 | All valid | 3,457,681 | -31.297 | -37.962 | 45.273 | 0.227 |
| 2024-11-18 | SNR ≥ 3 | 2,600,908 | -35.970 | -37.996 | 36.568 | 0.137 |
| 2024-11-18 | SNR ≥ 5 | 1,703,603 | -37.711 | -38.017 | 37.761 | 0.099 |

The separate-site holdout comparison quantifies agreement after each calibration-site median offset is removed. It remains a relative diagnostic, not absolute accuracy.

| Granule | Calibration site → independent test site | SNR stratum | Test n / eligible denominator | Empirical offset (m) | Test mean residual (m) | RMSE (m) | NMAD (m) | Absolute P95 (m) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2024-11-12 | East field → southeast field | All valid | 14,885 / 17,171 | -37.663 | 24.208 | 84.332 | 0.295 | 209.176 |
| 2024-11-12 | East field → southeast field | SNR ≥ 3 | 4,274 / 4,852 | -37.659 | 0.271 | 8.522 | 0.117 | 0.278 |
| 2024-11-12 | East field → southeast field | SNR ≥ 5 | 445 / 478 | -37.659 | 0.086 | 0.132 | 0.088 | 0.230 |
| 2024-11-18 | Field site 01 → field site 02 | All valid | 8,924 / 8,924 | -37.980 | 1.547 | 19.123 | 0.095 | 0.243 |
| 2024-11-18 | Field site 01 → field site 02 | SNR ≥ 3 | 8,623 / 8,623 | -37.981 | -0.048 | 0.469 | 0.092 | 0.212 |
| 2024-11-18 | Field site 01 → field site 02 | SNR ≥ 5 | 6,615 / 6,615 | -37.979 | -0.037 | 0.097 | 0.087 | 0.190 |

### Vertical-frame audit and provisional candidate selection

The H5 files contain no units, vertical CRS, realization, coordinate epoch, or field-level correction semantics for `refh`, `geoid`, `geoid_free2mean`, and related tide/range fields. A CryoCloud L1B tutorial describes `refh` as WGS84 ellipsoidal height, but that convention does not identify the realization, epoch, or CASALS processing details. The H5 `geoid_free2mean` values numerically follow `0.1287 − 0.3848 sin²(latitude)` within 1.4×10⁻⁶ m on November 12 and 6.4×10⁻⁶ m on November 18; this is a formula match, not confirmation of the H5 field's meaning. H5 `geoid` medians differ from sampled NOAA GEOID18 undulations by −0.936 m and −1.030 m, respectively, so the field is not treated as GEOID18 without metadata.

| Granule | H5 geoid median (m) | H5 free-to-mean median (m) | Sampled NOAA GEOID18 N median (m) | H5 geoid − GEOID18 N median (m) |
|---|---:|---:|---:|---:|
| 2024-11-12 | −36.832462 | −0.019859 | −35.896918 | −0.935584 |
| 2024-11-18 | −37.431847 | −0.005459 | −36.401594 | −1.030103 |

The product summaries and final LPC reports identify the common target as NAD83(2011)/UTM18N (EPSG:6347) plus NAVD88 height (EPSG:5703), using GEOID18. The Maryland vendor XML has a conflicting generic `altdatum` entry naming GEOID12B; its abstract, product summary, and final LPC report identify GEOID18, and the final report's tile WKT lists EPSG:5703. This conflict is retained in the source audit. The North Carolina work-unit XML and final report identify GEOID18 without that conflict.

| Granule | Work unit / project | Source metadata target | Identity-matched source EPT points | Maximum absolute ΔZ (m) | Maximum ΔXY (m) |
|---|---|---|---:|---:|---:|
| 2024-11-12 | `MD_Southeast_1_2019` / `MD_Southeast_2019_D20` | NAVD88 / GEOID18; Maryland XML conflict noted above | 11,871 | 0.005 | 0.0071 |
| 2024-11-18 | `NC_HurricaneFlorence_9_2020` / `NC_HurricaneFlorence_2020_D20` | NAVD88 / GEOID18 | 11,775 | 0.005 | 0.0064 |

The point check used exact GPS-time bit patterns with return/source attributes to match a small sample from source EPT nodes to each current clipped LAZ. All sampled Z values agree within the 0.01 m LAS quantization tolerance. The EPT declaration itself provides horizontal EPSG:3857 but no vertical CRS; the source product metadata supplies the vertical interpretation. This verifies Z passthrough for the sampled identities, not every LAZ record.

C7 applies a real 3D `TransformerGroup` operation from the assumed EPSG:4979 WGS84 3D ensemble to EPSG:6347 + EPSG:5703. Ballpark operations are prohibited. PROJ listed 14 available operations for November 12 and 13 for November 18, with no unavailable operations in the active environment. The two highest-ranked alternatives report 1.14 m operation accuracy: one uses GEOID99/NADCON5, the other GEOID03/NADCON5. Because the 3DEP products specify GEOID18, C7 uses the available operation that requires `us_noaa_g2018u0.tif` (ranked third; accuracy 2.015 m). That grid was stored under the experiment output with SHA-256 `fa9a407ac7ee3f5a3694008e4bcd09ce9cc250452f0c3b11700a4960340abce2`. The 2.015 m figure is PROJ's operation accuracy metadata, not an observed error or a guarantee for these CASALS heights. Its inverse round-trip error is at most 2.5×10⁻⁹ m horizontally and 3.2×10⁻¹⁴ m vertically on the 512-point check; these values verify numerical reversibility only. The transformed XY equals the existing EPSG:6347 projection in this sample, and the 3DEP plane is still re-queried at C7 XY.

The candidate split is frozen from geometry before candidate-height metrics: each preserved calibration ROI is divided into 8,987 offset-fit and 8,987 candidate-selection pulses on November 12, and 3,570 + 3,570 on November 18. The independent-test ROIs remain 17,171 and 8,924 pulses. The split-manifest SHA-256 is `cd714dd3057193bc9f510dfa34adfe19b2c28db628f2b93f41525a8a30f8b702`. Candidate selection uses only common pulse IDs with acceptable original and C7-requeried 3DEP support; no independent-test metrics enter selection.

| Candidate | Height formula | Interpretation |
|---|---|---|
| C0 | `Refh` | Native baseline; source vertical frame unresolved |
| C1 | `Refh − geoid` | H5 scalar hypothesis |
| C2 | `Refh + geoid` | H5 sign-sensitivity diagnostic |
| C3 | `Refh − geoid − geoid_free2mean` | H5 scalar sensitivity |
| C4 | `Refh − geoid + geoid_free2mean` | H5 scalar sensitivity |
| C5 | `Refh + geoid − geoid_free2mean` | H5 sign-sensitivity diagnostic |
| C6 | `Refh + geoid + geoid_free2mean` | H5 sign-sensitivity diagnostic |
| C7 | GEOID18-constrained PROJ XYZ transform | Provisional geodetic path to the documented 3DEP target |
| C8 (full-granule empirical) | `Refh − calibration-only median(C0 residual)` | Separate empirical alignment; never a datum candidate |

On the SNR ≥ 3 candidate-selection pulses, C3 had the smallest numerical RMSE in both granules, while C1 and C4 were within centimeters. C7 was retained because its operation is tied to the documented target geoid, not because numerical fit proves its datum. The table reports uncorrected residuals on the common candidate-selection IDs; NMAD reflects the central distribution while RMSE also reflects tails.

| Candidate | 2024-11-12 bias / RMSE / NMAD (m) | 2024-11-18 bias / RMSE / NMAD (m) |
|---|---:|---:|
| C0 | −37.704 / 37.925 / 0.114 | −37.981 / 37.984 / 0.094 |
| C1 | −0.875 / 4.177 / 0.114 | −0.566 / 0.753 / 0.093 |
| C2 | −74.534 / 74.646 / 0.114 | −75.395 / 75.397 / 0.094 |
| C3 | −0.855 / 4.172 / 0.114 | −0.561 / 0.749 / 0.093 |
| C4 | −0.894 / 4.181 / 0.114 | −0.572 / 0.757 / 0.093 |
| C5 | −74.514 / 74.626 / 0.114 | −75.390 / 75.391 / 0.094 |
| C6 | −74.554 / 74.665 / 0.114 | −75.401 / 75.402 / 0.094 |
| C7 | −1.810 / 4.467 / 0.114 | −1.596 / 1.671 / 0.093 |

The selected configuration is C7, with status `PROVISIONAL_NOT_DATUM_VERIFIED`. Its median vertical correction to `Refh` is +35.894 m on November 12 and +36.368 m on November 18. The missing CASALS realization and coordinate epoch prevent a verified geodetic source-to-target claim, even though the target product, selected GEOID18 grid, and sampled Z passthrough are evidenced.

Here, C8 denotes the earlier whole-granule open-field empirical holdout. The separate road-target C8 fit below uses different calibration rows and fails its support gate; its saved values are retained only as failure diagnostics.

On the independent-test pulses, the SNR ≥ 5 C7 comparison has a −1.67 m bias and 1.67 m RMSE on November 12 (445 / 478 eligible, 93.1% support) and a −1.65 m bias and 1.65 m RMSE on November 18 (6,615 / 6,615). The central NMAD is about 0.088 m for both. Native C0 remains about −37.6 to −38.0 m from the target. C8 remains a separate site-calibrated baseline: its current SNR ≥ 5 RMSE is 0.113 m and 0.098 m, compared with the previous full-calibration values of 0.132 m and 0.097 m. The differences are −0.019 m and +0.001 m. This is a descriptive comparison: the current C8 offset uses half the calibration ROI, whereas the earlier analysis used the full calibration ROI, and the common-support rules differ.

The SNR ≥ 3 and all-valid test tails remain visible in the saved metrics and figures. C7 SNR ≥ 3 RMSE is 8.65 m on November 12 and 1.73 m on November 18; the all-valid RMSE is 83.84 m and 19.06 m. These high values arise from retained large residuals, many at low SNR. C7's lower central NMAD does not certify absolute vertical accuracy.

### Conditional road-target evaluation

This supplemental analysis evaluates whether existing matched CASALS pulse centers fall in a reviewed likely-ground target area and have nearby 3DEP class-2 support. It reuses the existing matched rows, applies C7 only as the already selected provisional coordinate transformation, and re-queries the local class-2 surface at transformed coordinates. It does not classify the origin of an individual CASALS `Refh` return. The reference is a local plane fit to 3DEP LAZ class-2 points, not an official DEM raster. All residuals below are `C7 transformed Refh − re-queried local plane`; they are conditional comparisons, not certified elevation accuracy.

The context population contains 118,220 matched pulse centers: 61,498 in non-ground contexts, 53,742 in uncertain contexts, and 2,980 in road-core target areas. The two reviewed road polygons contain 3,820 November 12 and 1,040 November 18 pulse centers. Inward edge trims of 2.0 m and 0.5 m, respectively, leave 2,375 and 605 geometry-only target-eligible centers; the remaining 1,445 and 435 centers stay uncertain because they fall in the road boundary band. Canopy, water, open-field, and other non-road contexts remain descriptive context or uncertain cases rather than presumed ground returns. Every context row records `return_origin_status=unresolved_for_CASALS_Refh`.

| Acquisition | NAIP acquisition date (age at CASALS) | Road pulse centers | Inward trim | Ground-target-eligible centers | Boundary centers retained as uncertain |
|---|---:|---:|---:|---:|---:|
| 2024-11-12 | 2023-06-02 (529 days) | 3,820 | 2.0 m | 2,375 | 1,445 |
| 2024-11-18 | 2022-06-15 (887 days) | 1,040 | 0.5 m | 605 | 435 |

| Acquisition | H5 pulses | Existing matched reference rows | Original road-polygon centers | Interior target | Boundary uncertain | Original class-2 support rejected | Original support accepted in guard gap | Additional C7/candidate support rejection | Final common C0–C7 support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024-11-12 | 3,604,480 | 3,559,190 | 3,820 | 2,375 | 1,445 | 1,962 | 24 | 0 | 389 |
| 2024-11-18 | 3,604,480 | 3,598,281 | 1,040 | 605 | 435 | 352 | 33 | 0 | 220 |

The 24 and 33 originally supported guard-gap points are excluded by the frozen spatial split, not by C7. Among the remaining non-guard target points, every original-surface-acceptable point also passes the C7 and common-candidate support checks (389/389 and 220/220). The main loss is the original local-surface support gate; the SNR-specific test denominators below show the further signal-quality stratification without treating SNR as ground classification.

NAIP is historical scene context and is not a contemporaneous ground label. Geometry-only road roles were frozen before any candidate residuals: November 12 has 165 calibration, 918 candidate-selection, 161 guard-gap, and 1,131 independent-test pulses; November 18 has 62, 204, 114, and 225. The split uses a two-meter total guard gap at each one-third axis boundary. The segments are portions of one road per granule, so the test portions are spatially separated but are not independent sites. C0–C7 common support includes 389 pulses on November 12 and 220 on November 18. At SNR ≥ 3, common support covers 54/453 (11.9%) and 138/195 (70.8%) of test target pulses; at SNR ≥ 5 it covers 14/84 (16.7%) and 70/117 (59.8%). These low and date-dependent support fractions constrain interpretation.

| Granule | SNR stratum | C7 common-support test n / eligible target denominator | RMSE (m) | NMAD (m) | P95 absolute difference (m) | Maximum absolute difference (m) |
|---|---|---:|---:|---:|---:|---:|
| 2024-11-12 | All valid Refh | 147 / 1,131 | 97.810 | 0.318 | 213.815 | 232.336 |
| 2024-11-12 | SNR ≥ 3 | 54 / 453 | 11.533 | 0.126 | 2.062 | 83.690 |
| 2024-11-18 | All valid Refh | 165 / 225 | 20.570 | 0.114 | 2.108 | 136.375 |
| 2024-11-18 | SNR ≥ 3 | 138 / 195 | 1.596 | 0.110 | 1.797 | 2.062 |

For the SNR ≥ 3 common-support test population, candidate metrics are:
The archived C8 rows below are retained only as failure diagnostics because the road-target calibration-support gate is not met.

| Granule | Candidate | n | Bias (m) | RMSE (m) | NMAD (m) |
|---|---|---:|---:|---:|---:|
| 2024-11-12 | C0 | 54 | −36.149 | 37.943 | 0.127 |
| 2024-11-12 | C1 | 54 | 0.686 | 11.551 | 0.126 |
| 2024-11-12 | C3 | 54 | 0.706 | 11.552 | 0.126 |
| 2024-11-12 | C7 | 54 | −0.250 | 11.533 | 0.126 |
| 2024-11-12 | C8 | 54 | −160.222 | 160.636 | 0.127 |
| 2024-11-18 | C0 | 138 | −37.964 | 37.965 | 0.110 |
| 2024-11-18 | C1 | 138 | −0.563 | 0.575 | 0.110 |
| 2024-11-18 | C3 | 138 | −0.558 | 0.569 | 0.110 |
| 2024-11-18 | C7 | 138 | −1.591 | 1.596 | 0.110 |
| 2024-11-18 | C8 | 138 | 0.139 | 0.180 | 0.110 |

At SNR ≥ 5, C7 bias/RMSE/NMAD is −1.823/1.826/0.046 m on November 12 (n=14) and −1.571/1.575/0.111 m on November 18 (n=70). The small November 12 sample and low coverage limit that comparison. The complete C0–C8 table for all-valid, SNR ≥ 3, and SNR ≥ 5 populations is saved in `ground_conditional_test_metrics.csv`.

The central distributions are tight compared with the tail-sensitive RMSE, especially on November 12. The retained November 12 maximum is a +83.690 m difference in the SNR ≥ 3 subset; in the all-valid subset, the maximum absolute difference is 232.336 m. On November 18 the all-valid maximum is 136.375 m. These large cases remain in all summaries. Their local class-2 neighborhoods meet the configured support and roughness gates (the reviewed examples have 32 nearby class-2 points and slope below 5°), but this does not resolve nominal footprint mismatch, surface/return mixture, or the actual origin of `Refh`. The >2 m review list is a display aid, not an exclusion rule.

Crosswalking the preserved historical SNR ≥ 3 empirical-holdout large-difference lists against the reviewed context layer places all 20 November 12 cases in `open_field_12_02` and the one November 18 case in `open_field_02`; all 21 are `uncertain`, and none is in a `non_ground_context` polygon. The old cases therefore are not confirmed canopy returns. The field ROIs remain uncertain because the older NAIP imagery cannot establish crop or grass height at the 2024 acquisition. The pulse-level crosswalk and original case-table hashes are recorded under the new ground-aware output.

A companion descriptive pass applies the existing provisional C7 transform to all 118,220 reviewed context pulses and queries the already-built class-2 training grid; it does not reread the LAZ. C7 XY matches the existing EPSG:6347 coordinates in this sample. The local surface passes its quality gate for 40,931/70,932 November 12 and 40,642/47,288 November 18 context pulses. These context-level differences still do not establish return origin or canopy height. Selected non-road ROIs show the support and spread:

The C8 column is retained only to show the unsupported calibration failure; its values are not candidate corrections.

| Acquisition / ROI | Context pulses / C7-supported | C0 median (m) | C7 median / RMSE / NMAD (m) | C8 median (m) |
|---|---:|---:|---:|---:|
| 2024-11-12 `open_field_12_01` | 17,974 / 17,974 | −37.663 | −1.769 / 62.044 / 0.208 | −161.736 |
| 2024-11-12 `open_field_12_02` | 17,171 / 14,885 | −37.564 | −1.661 / 83.843 / 0.296 | −161.637 |
| 2024-11-12 `tree_canopy_12_01` | 31,314 / 6,944 | −26.972 | +4.240 / 105.751 / 94.048 | −151.045 |
| 2024-11-12 `water_pond_12_01` | 653 / 0 | −38.168 | no supported C7 surface | −162.242 |
| 2024-11-18 `open_field_01` | 7,140 / 7,140 | −37.980 | −1.595 / 15.583 / 0.096 | +0.123 |
| 2024-11-18 `tree_canopy_01` | 3,503 / 3,257 | −25.180 | +11.101 / 32.226 / 2.415 | +12.923 |
| 2024-11-18 `tree_canopy_02` | 4,329 / 4,329 | −26.075 | +10.324 / 40.512 / 1.960 | +12.028 |
| 2024-11-18 `tree_canopy_18_03` | 22,352 / 16,523 | −37.896 | −1.549 / 21.928 / 0.241 | +0.208 |

C0 uses the original-XY class-2 plane; C7 uses the transformed-height value and transformed-XY plane, only where its surface is accepted. The archived C8 values use unsupported road-calibration offsets and are failure diagnostics only; they are not transferable land-cover corrections. The differences among forest ROIs, including the edge-mixed `tree_canopy_18_03`, do not support a general canopy-return or land-cover effect. Full per-ROI metrics, pulse-level values, and transform/support audit are saved in [`descriptive_reference_return_to_ground_metrics.csv`](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/descriptive_reference_return_to_ground_metrics.csv), [`descriptive_reference_return_to_ground_population.parquet`](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/descriptive_reference_return_to_ground_population.parquet), and [`descriptive_context_c7_query_audit.json`](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/descriptive_context_c7_query_audit.json).

The archived road-target C8 fits fail the predeclared support gate: at least 30 SNR >= 5 common-support calibration pulses distributed across at least three spatial blocks. Blocks are equal thirds of the frozen calibration segment. No road-target C8 correction is applied or interpreted; the offsets and test metrics are retained to show the failure mode.

| Granule | Archived calibration common n | SNR >= 5 common n | Blocks with SNR >= 5 support | Archived offset (m; diagnostic only) | Gate status |
|---|---:|---:|---:|---:|---|
| 2024-11-12 | 94 | 0 | 0 | +124.073 | Insufficient calibration support |
| 2024-11-18 | 1 | 0 | 0 | -38.103 | Insufficient calibration support |

The saved SNR >= 5 test values for this road-target fit (n=14 and n=70) are failure diagnostics, not evidence of generalization. The machine-readable gate is [`ground_c8_calibration_support_gate.csv`](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/ground_c8_calibration_support_gate.csv).

The separate held-out class-2 interpolation check is very small near the reviewed contexts: November 12 has 3 points (2 plane-fit valid, 1 passing surface quality), with 0.0138 m plane error and 0.0175 m local-median error on the single accepted point; November 18 has 6 points (5 valid and accepted), with 0.0199 m plane RMSE and 0.0330 m local-median RMSE. This checks local interpolation against a few withheld reference points only. It does not validate CASALS heights or the vertical frame.

The new figures show the reviewed context and road cores on NAIP, class-2 neighborhoods, C0–C7 comparison, full residual tails, four fixed waveform examples, and test-track elevation profiles: [scene overview](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/figures/ground_scene_overview.png), [reference support](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/figures/reference_ground_context.png), [candidate comparison](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/figures/ground_candidate_comparison.png), [residual distributions](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/figures/ground_residual_distributions.png), [waveform cases](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/figures/ground_vs_nonground_cases.png), and [elevation profile](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/figures/ground_elevation_profile.png). See the [ground-aware result folder](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/), including the [context manifest](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/ground_context_manifest.csv), [population summary](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/ground_context_population_summary.csv), [frozen spatial split](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/ground_spatial_split_manifest.csv), [test metrics](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/ground_conditional_test_metrics.csv), [ground-target large-difference review list](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/large_ground_target_residuals_for_review.csv), [historical large-difference context crosswalk](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/historical_large_difference_context_crosswalk.csv), and [class-2 interpolation check](../outputs/notebooks/10_3dep_elevation_validation/ground_aware/independent_class2_interpolation_check.csv). The per-file SHA-256 manifest is stored in the same folder.

### CASALS Refh Processing Corrections and Residual Vertical Bias

Both source H5 files were read across all 3,604,480 pulses using chunked field reads. The audit reused the existing bin-georeferencing beam helper, the saved GEOID18 sample comparison, and the frozen ground-target tables. The CASALS L1B tutorial describes `Refh` as the maximum-amplitude waveform-bin height in WGS84 ellipsoidal coordinates, and notes that some correction datasets can be marked not applied; it does not provide per-field application flags for these two files. The H5 fields and root metadata contain no per-field units, sign conventions, source model declarations, applied flags, source-frame realization, or coordinate epoch. [CASALS L1B tutorial](https://book.cryointhecloud.com/l1b-waveforms-tutorial/)

#### Measured H5 fields and application status

The following are measured full-granule numeric values, with all 3,604,480 samples valid under declared HDF5 missing-value attributes. Units remain unknown where the H5 does not declare them.

| Field | 2024-11-12 median [min, max] | 2024-11-18 median [min, max] | H5 units attribute |
| --- | --- | --- | --- |
| Refh neutral delay total | 1.593985 [1.524848, 1.645126] | 1.180952 [1.110669, 1.283471] | UNKNOWN |
| Refh neutral delay derivative | 0.000278 [0.000271, 0.000282] | 0.000278 [0.000271, 0.000282] | UNKNOWN |
| Range-window start neutral delay | 1.527246 [1.524806, 1.532220] | 1.113098 [1.110623, 1.171893] | UNKNOWN |
| Range-window start neutral-delay derivative | 0.000271 [0.000271, 0.000272] | 0.000271 [0.000271, 0.000272] | UNKNOWN |
| Range-window stop neutral delay | 1.640370 [1.637892, 1.645459] | 1.226318 [1.223732, 1.285335] | UNKNOWN |
| Range-window stop neutral-delay derivative | 0.000282 [0.000282, 0.000282] | 0.000282 [0.000281, 0.000283] | UNKNOWN |
| Range-bias correction | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | UNKNOWN |
| H5 geoid | -36.832462 [-36.846004, -36.819122] | -37.431847 [-37.448933, -37.382866] | UNKNOWN |
| Geoid free-to-mean field | -0.019859 [-0.019936, -0.019788] | -0.005459 [-0.005510, -0.005432] | UNKNOWN |
| Earth tide | -0.015728 [-0.016005, -0.015476] | -0.075044 [-0.075115, -0.074997] | UNKNOWN |
| Earth tide free-to-mean | -0.009538 [-0.009574, -0.009505] | -0.002770 [-0.002794, -0.002757] | UNKNOWN |
| Load tide | 0.006266 [0.006228, 0.006313] | 0.001652 [0.001618, 0.001679] | UNKNOWN |
| Pole tide | -0.004368 [-0.004368, -0.004367] | -0.004380 [-0.004381, -0.004380] | UNKNOWN |
| Ocean tide | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | UNKNOWN |
| Ocean pole tide | 0.000348 [0.000347, 0.000348] | 0.000319 [0.000318, 0.000319] | UNKNOWN |
| Dynamic atmospheric correction | -0.127479 [-0.128610, -0.126058] | -0.041126 [-0.043564, -0.037339] | UNKNOWN |


`range_bias_correction` is exactly zero for every pulse in both granules. Its HDF5 storage fill value is also zero, but there is no `_FillValue` or `missing_value` attribute; zero cannot be distinguished from a placeholder or zero calibration using this metadata alone. The per-field application matrix therefore reports `UNKNOWN` for neutral delay, range bias, geoid, free-to-mean, other tides, and DAC. The geoid fields are `NUMERICALLY_SUPPORTED` as model/formula matches, not confirmed as applied or unapplied. The full application matrix and evidence registry are in [correction_application_matrix.csv](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/correction_application_matrix.csv) and [correction_evidence_registry.csv](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/correction_evidence_registry.csv).

Track, spatial, and time-bin summaries are retained for every correction field. For example, the track medians of the neutral-delay field span 1.579112 to 1.594533 on November 12 and 1.178568 to 1.181056 on November 18. The range-bias field remains zero in every track, 4x4 region, and time bin. The full group-level rows are in [correction_statistics.csv](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/correction_statistics.csv) and [correction_time_trends.csv](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/correction_time_trends.csv).

#### Atmospheric and range sensitivity

Using the measured beam geometry and existing audit convention (downward sensor-to-target vector; local angles empirically interpreted in radians), removing a positive excess slant range raises the target height. If `refh_neutat_delay_total` is interpreted as metres of excess geometric range and is not already applied, the full-granule median height sensitivities are 1.591955 m (P05-P95 1.533928 to 1.628034) and 1.179909 m (P05-P95 1.173675 to 1.180569). If the numeric values instead use nanoseconds and the root `ns_to_meters` factor is appropriate, the corresponding medians are 0.238628 m and 0.176864 m. Both unit scenarios are conditional; no H5 or processor evidence selects between them.

On the frozen common independent-test support (147 and 165 targets), the metre-assumption median sensitivities are 1.592204 m and 1.180472 m. This sign can move the negative C7 median residual toward zero. The `ns_to_meters` scenario is much smaller. Since applied status is unknown, reapplying the field carries a double-correction risk and neither scenario is a corrected Refh result.

The range-bias height sensitivity is zero because the stored field is zero; its status remains unknown. Stored `refh_neutat_delay_total` is numerically close to linear interpolation between the stored range-window endpoint delays: the full-granule median residual is -0.000527 and -0.000529 in undeclared field units. This is internal continuity within the same L1B product, not proof that the delay was applied. The existing bin audit favors its H2 bin-center convention empirically, while its physical range model remains unavailable because `bin_size` units and processor semantics are undocumented; no half-bin or timing value was fitted to the 3DEP residuals. [Bin-georeferencing audit findings](../research/geolocation/bin_georeferencing_audit/findings.md)

Descriptive Spearman associations on the fixed common independent-test targets are:

| Granule | Context variable | n | Spearman correlation with C7 residual |
| --- | --- | --- | --- |
| 2024-11-12 | neutral_delay_field_numeric | 147 | -0.869 |
| 2024-11-12 | range_bias_field_numeric | 147 | — |
| 2024-11-12 | beam_incidence_deg | 147 | 0.502 |
| 2024-11-12 | H5_geoid_minus_GEOID18_m | 147 | 0.790 |
| 2024-11-18 | neutral_delay_field_numeric | 165 | -0.635 |
| 2024-11-18 | range_bias_field_numeric | 165 | — |
| 2024-11-18 | beam_incidence_deg | 165 | 0.394 |
| 2024-11-18 | H5_geoid_minus_GEOID18_m | 165 | 0.473 |


The range-bias association is undefined because the field is constant. These correlations, especially the neutral-delay and beam-incidence patterns, are descriptive and may reflect location/track covariation; they do not establish a correction's application state or causal attribution.

#### H5 geoid, EGM2008, GEOID18, and tide convention

At the 1,200 common coordinates per granule already used in the saved GEOID18 audit, the H5 `geoid` field agrees with the official NGA EGM2008 2.5-minute WGS84 Tide-Free grid to a median of +0.000438 m and -0.002074 m. The same sample shows the following model differences:

| Granule | n | H5 - EGM2008 median (m) | H5 - GEOID18 median (m) | EGM2008 - GEOID18 median (m) |
| --- | --- | --- | --- | --- |
| 2024-11-12 | 1200 | 0.000438 | -0.935584 | -0.935980 |
| 2024-11-18 | 1200 | -0.002074 | -1.030103 | -1.028116 |


The archive checksum is `0f65f16e6fd3f89a6b8022d7a89375d0c29fb275a551927175669bb610904cd0`. The grid reader passed NGA's six bundled `OUTPUT3.DAT` sample points with maximum absolute difference 0.009916 m (the bundled interpolation reference is stated to agree within 1 cm). This strongly supports a numerical EGM2008 match for the H5 field, while its H5 source-model declaration remains absent. NGA describes EGM2008 as the WGS84 Tide-Free model; NOAA describes GEOID18 for NAVD88 use with NAD83(2011), epoch 2010.0 ellipsoid heights. [NGA EGM2008](https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84), [NOAA GEOID18 technical details](https://geodesy.noaa.gov/GEOID/GEOID18/geoid18_tech_details.shtml)

On the high-SNR common targets, C1's median residual is higher than C7 by +0.935569 m and +1.028278 m, matching the opposite sign of the H5-minus-GEOID18 differences. This accounts numerically for almost all of the roughly 1 m C1/C7 separation. It does not establish that CASALS generated `Refh` using EGM2008. The H5 `geoid_free2mean` field matches `0.1287 - 0.3848 sin(latitude)^2` to median absolute differences of 6.73e-07 m and 9.70e-08 m. The corresponding C3-minus-C1 median residual shifts are +0.01985 m and +0.00548 m. These are numerical formula/convention matches; the exact H5 tide-system meaning and application stage remain unknown. IERS documents the distinct permanent-tide systems, but cannot identify the H5 field's convention without CASALS metadata. [IERS Conventions (2010)](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36.pdf?__blob=publicationFile&v=1)

Other stored tide fields have full-granule medians of -0.015728/-0.075044 (`tide_earth`), -0.009538/-0.002770 (`tide_earth_free2mean`), +0.006266/+0.001652 (`tide_load`), -0.004368/-0.004380 (`tide_pole`), zero (`tide_ocean`), +0.000348/+0.000319 (`tide_ocean_pole`), and -0.127479/-0.041126 (`dac`) for November 12/18. Their units, signs, and applied states are unknown, so they were not summed or added as a combined correction.

#### C7 operation and common-sample bias comparison

The selected C7 pipeline is `proj=pipeline step proj=unitconvert xy_in=deg xy_out=rad step inv proj=vgridshift grids=us_noaa_g2018u0.tif multiplier=1 step proj=utm zone=18 ellps=GRS80`. It exactly reproduces the saved C7 coordinates on all 312 common independent-test rows. The WGS84 3D ensemble to NAD83(2011) 3D component resolves numerically to `proj=noop ellps=GRS80` (2.0 m stated operation accuracy); the selected combined operation reports 2.015 m accuracy. The observed C7 vertical shifts are:

| Granule | C7 vertical shift median (m) | WGS84-NAD83(2011) step median (m) | Fixed test rows | Selected operation accuracy (m) |
| --- | --- | --- | --- | --- |
| 2024-11-12 | 35.898712 | 0.000000 | 147 | 2.015 |
| 2024-11-18 | 36.372920 | 0.000000 | 165 | 2.015 |


Thus the selected C7 operation's numeric vertical change is GEOID18; the separate frame step contributes zero in the reconstructed operation. Since the CASALS source realization and coordinate epoch are absent, this is not evidence that a zero-shift WGS84-ensemble assumption is physically correct for these measurements.

The next table uses the same frozen independent-test pulse IDs within each SNR stratum. Residuals are candidate height minus the saved local 3DEP ground plane. H1 assumes the stored neutral-delay values are metres of positive excess slant range; H2 uses the stored range-bias value with the same sign convention. C1/C3 and H1/H2 are sensitivity comparisons, not verified corrected heights.

| Granule | Hypothesis | Evidence status | n | Mean bias (m) | Median bias (m) | RMSE (m) | NMAD (m) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-11-12 | C7 | DOCUMENTED_ALTERNATIVE | 14 | -1.8234 | -1.8377 | 1.8262 | 0.0456 |
| 2024-11-12 | C1 | PHYSICAL_SENSITIVITY_ONLY | 14 | -0.8878 | -0.9021 | 0.8936 | 0.0456 |
| 2024-11-12 | C3 | PHYSICAL_SENSITIVITY_ONLY | 14 | -0.8680 | -0.8822 | 0.8739 | 0.0456 |
| 2024-11-12 | H1 atmospheric sensitivity | PHYSICAL_SENSITIVITY_ONLY | 14 | -0.2312 | -0.2454 | 0.2525 | 0.0456 |
| 2024-11-12 | H2 range-bias sensitivity | PHYSICAL_SENSITIVITY_ONLY | 14 | -1.8234 | -1.8377 | 1.8262 | 0.0456 |
| 2024-11-12 | H3 verified source-frame alternative | UNAVAILABLE | 0 | — | — | — | — |
| 2024-11-12 | H5 physically justified combined chain | UNAVAILABLE | 0 | — | — | — | — |
| 2024-11-12 | H4 geoid/tide convention alternative (C3) | PHYSICAL_SENSITIVITY_ONLY | 14 | -0.8680 | -0.8822 | 0.8739 | 0.0456 |
| 2024-11-18 | C7 | DOCUMENTED_ALTERNATIVE | 70 | -1.5713 | -1.5494 | 1.5752 | 0.1115 |
| 2024-11-18 | C1 | PHYSICAL_SENSITIVITY_ONLY | 70 | -0.5430 | -0.5211 | 0.5541 | 0.1115 |
| 2024-11-18 | C3 | PHYSICAL_SENSITIVITY_ONLY | 70 | -0.5375 | -0.5156 | 0.5488 | 0.1115 |
| 2024-11-18 | H1 atmospheric sensitivity | PHYSICAL_SENSITIVITY_ONLY | 70 | -0.3908 | -0.3689 | 0.4061 | 0.1115 |
| 2024-11-18 | H2 range-bias sensitivity | PHYSICAL_SENSITIVITY_ONLY | 70 | -1.5713 | -1.5494 | 1.5752 | 0.1115 |
| 2024-11-18 | H3 verified source-frame alternative | UNAVAILABLE | 0 | — | — | — | — |
| 2024-11-18 | H5 physically justified combined chain | UNAVAILABLE | 0 | — | — | — | — |
| 2024-11-18 | H4 geoid/tide convention alternative (C3) | PHYSICAL_SENSITIVITY_ONLY | 70 | -0.5375 | -0.5156 | 0.5488 | 0.1115 |


For the wider common-supported set, C7 RMSE/NMAD are 97.810/0.318 m and 20.570/0.114 m. In the SNR>=3 stratum they are 11.533/0.126 m and 1.596/0.110 m. These large RMSE/NMAD gaps show the gross-error tail remains in the target population; the SNR filter is a reporting stratum, not proof of ground return. The SNR>=5 support is only 14 targets on November 12 and 70 on November 18.

| Granule | Population | Hypothesis | n | Mean bias (m) | Median bias (m) | RMSE (m) | NMAD (m) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-11-12 | all_common_support | C7 | 147 | 30.9070 | -1.7951 | 97.8103 | 0.3180 |
| 2024-11-12 | all_common_support | H1 atmospheric sensitivity | 147 | 32.4902 | -0.2029 | 98.2979 | 0.3180 |
| 2024-11-12 | snr_ge_3 | C7 | 54 | -0.2499 | -1.8268 | 11.5334 | 0.1265 |
| 2024-11-12 | snr_ge_3 | H1 atmospheric sensitivity | 54 | 1.3419 | -0.2346 | 11.6053 | 0.1265 |
| 2024-11-18 | all_common_support | C7 | 165 | 1.8476 | -1.5769 | 20.5699 | 0.1136 |
| 2024-11-18 | all_common_support | H1 atmospheric sensitivity | 165 | 3.0271 | -0.3964 | 20.7036 | 0.1136 |
| 2024-11-18 | snr_ge_3 | C7 | 138 | -1.5915 | -1.5771 | 1.5956 | 0.1103 |
| 2024-11-18 | snr_ge_3 | H1 atmospheric sensitivity | 138 | -0.4110 | -0.3966 | 0.4266 | 0.1103 |


The H1 metre-assumption sensitivity reduces the high-SNR RMSE to 0.252 m and 0.406 m, respectively, while the nanosecond interpretation changes the candidate by only about 0.239 m and 0.177 m in median height. The better numerical fit cannot establish that the delay was omitted from L1B. H2 is unchanged from C7. H3 (verified source-frame alternative) and H5 (physically justified combined chain) are explicitly `UNAVAILABLE`; H4 duplicates the C3 geoid/tide sensitivity. The existing road-target C8 support gate remains `INSUFFICIENT_CALIBRATION_SUPPORT` for both granules, with zero high-quality SNR>=5 calibration support; C8 is excluded from attribution.

#### Findings, limits, and required processing evidence

The available evidence identifies three separate facts: (1) the stored delay has a positive height effect under a physically plausible excess-range interpretation; (2) the H5 geoid is numerically almost identical to NGA EGM2008 and differs from GEOID18 by about 1 m; and (3) the selected C7 pipeline applies GEOID18 but no numeric WGS84-to-NAD83(2011) frame shift. It does not identify whether the neutral delay, range bias, or other H5 fields were already applied when Refh was written. The remaining negative C7 residual on high-SNR supported targets is therefore not uniquely attributable, and double-correction risk remains.

A conclusive CASALS processing record needs: the exact L1B Refh bin/time-to-range formula; field-by-field units, signs, fill-value semantics, and applied/not-applied flags; the neutral-delay conversion (including optical-path versus one-way/two-way range and application stage); range-bias calibration values and zero-fill convention; the geoid model/version and tide system and whether it contributes to Refh or is exposed as a diagnostic; the GNSS/INS reference realization and coordinate epoch; and the per-file processor version/configuration or L1A-to-L1B provenance log.

The reproducible tables, six figures, H5 attributes, and source hashes are under [correction_audit](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/); the [summary](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/correction_audit_summary.json), [field inventory](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/correction_field_inventory.csv), [same-sample metrics](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/physical_hypothesis_comparison.csv), and [audit manifest](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/audit_manifest.json) record provenance and denominators.

#### Bias and dispersion on the frozen common-support test

This analysis reuses the 312 independent-test pulses with common candidate support inside the reviewed road targets. Every scenario uses the same pulse IDs within each stratum: November 12 has 147 all-support, 54 SNR ≥ 3, and 14 SNR ≥ 5 pulses; November 18 has 165, 138, and 70. No residual-based observations are removed. The pulse-ID hashes and exact metric parity against the prior C7/C1/C3/H1 tables are recorded in [`residual_diagnostics_manifest.json`](../outputs/notebooks/10_3dep_elevation_validation/residual_diagnostics/residual_diagnostics_manifest.json).

The table below reports the high-SNR sample. `cRMSE` is population-centered RMS dispersion. Its decomposition with mean bias is exact: RMSE² = bias² + cRMSE². The fractions describe squared-error arithmetic and do not assign causal error sources.

| Acquisition | Scenario | n | Mean bias (m) | RMSE (m) | cRMSE (m) | NMAD (m) | Bias² / RMSE² |
|---|---|---:|---:|---:|---:|---:|---:|
| 2024-11-12 | C7 baseline | 14 | -1.823 | 1.826 | 0.101 | 0.046 | 0.9969 |
| 2024-11-12 | H1 stored delay as meters | 14 | -0.231 | 0.252 | 0.101 | 0.046 | 0.8387 |
| 2024-11-12 | H1 ERA5 profile sensitivity | 14 | -0.217 | 0.240 | 0.101 | 0.046 | 0.8213 |
| 2024-11-18 | C7 baseline | 70 | -1.571 | 1.575 | 0.111 | 0.111 | 0.9951 |
| 2024-11-18 | H1 stored delay as meters | 70 | -0.391 | 0.406 | 0.110 | 0.111 | 0.9260 |
| 2024-11-18 | H1 ERA5 profile sensitivity | 70 | -0.428 | 0.442 | 0.111 | 0.111 | 0.9376 |

The conditional H1 scenarios shift the residual center much more than its spread, as expected for a nearly constant height term. Their application status remains unknown; they are sensitivities and do not create corrected Refh heights. The SNR ≥ 5 results show relative agreement within these selected road samples, not absolute elevation accuracy. In particular, cRMSE near 0.1 m is not a 0.1 m absolute-accuracy claim.

| Acquisition | C7 stratum | n | Mean bias (m) | RMSE (m) | cRMSE (m) | NMAD (m) | Bias² / RMSE² |
|---|---|---:|---:|---:|---:|---:|---:|
| 2024-11-12 | All common support | 147 | +30.907 | 97.810 | 92.799 | 0.318 | 0.0998 |
| 2024-11-12 | SNR ≥ 3 | 54 | -0.250 | 11.533 | 11.531 | 0.126 | 0.0005 |
| 2024-11-12 | SNR ≥ 5 | 14 | -1.823 | 1.826 | 0.101 | 0.046 | 0.9969 |
| 2024-11-18 | All common support | 165 | +1.848 | 20.570 | 20.487 | 0.114 | 0.0081 |
| 2024-11-18 | SNR ≥ 3 | 138 | -1.591 | 1.596 | 0.114 | 0.110 | 0.9949 |
| 2024-11-18 | SNR ≥ 5 | 70 | -1.571 | 1.575 | 0.111 | 0.111 | 0.9951 |

This separates the two patterns hidden by RMSE alone. On November 12, a single +83.69 m SNR ≥ 3 residual makes cRMSE 11.53 m while NMAD remains 0.126 m; the all-common tail is larger still. On November 18, the SNR ≥ 3 and SNR ≥ 5 RMSE is mostly the negative mean offset, while centered dispersion is about 0.11 m. These are Refh-to-local-ground differences with unresolved return origin, so extreme values are not confirmed ground-elevation errors.

For the November 12 SNR ≥ 5 sample, three pulses account for 85.9% of C7 centered squared error (37.7%, 27.3%, and 20.9%). Each had 32 nearby class-2 points and acceptable local support; the reviewed top cases had slopes of about 2.3–4.5° and plane-residual NMAD of 0.10–0.15 m. Leave-one-out values are influence diagnostics only; all 14 pulses remain included. On November 18, the 70 SNR ≥ 5 pulses span 57 tracks (44 singleton and 13 doubleton groups). No track has three selected pulses, so per-track dispersion ratios are withheld. Four equal-width road-axis blocks retain 7–23 points each; these are spatial blocks along the reviewed road, not aircraft along-track distance. Predictor summaries and all block/track rows are descriptive because each acquisition has one reviewed road site; no confidence intervals or per-track calibration are reported.

The five CSV tables and four saved figures are in [`residual_diagnostics/`](../outputs/notebooks/10_3dep_elevation_validation/residual_diagnostics/): [metric comparison](../outputs/notebooks/10_3dep_elevation_validation/residual_diagnostics/residual_metrics_comparison.csv), [bias decomposition](../outputs/notebooks/10_3dep_elevation_validation/residual_diagnostics/residual_bias_decomposition.csv), [pulse distributions](../outputs/notebooks/10_3dep_elevation_validation/residual_diagnostics/residual_distribution_diagnostics.csv), [track/spatial summaries](../outputs/notebooks/10_3dep_elevation_validation/residual_diagnostics/per_track_spatial_summary.csv), [all 14 high-influence pulses](../outputs/notebooks/10_3dep_elevation_validation/residual_diagnostics/high_influence_pulse_cases.csv), and [diagnostic figures](../outputs/notebooks/10_3dep_elevation_validation/residual_diagnostics/).

#### Final vertical-bias attribution and external profile check

**NO - not identifiable from available evidence (A3).** The approximate -1.6 m C7-to-3DEP road residual has no verified physical attribution. H5 does not declare neutral-delay units, sign, or whether the field was applied to Refh. Matched raw ranging, trajectory, calibration, processor configuration, and per-file application records are absent. Treating the field as meters improves the historical residual fit, but that does not show the correction was omitted; applying it again could double-correct Refh.

An independent ERA5/Ciddor group-refractivity profile checks whether the measured field is on a plausible physical scale. ERA5 pressure-level temperature, humidity, geopotential and surface pressure were sampled at the H5 median instrument location and nearest UTC hour, then integrated from the local 3DEP plane to instrument altitude with a near-nadir slant approximation and nominal 1040 nm. The H5 vertical references are not proven to match the ERA5/3DEP integration frame.

| Acquisition | Full-granule H5 delay median (units undeclared) | ERA5 slant group excess path | H5 / ERA5 ratio under meter convention | Derivative comparison |
|---|---:|---:|---:|---|
| 2024-11-12 | 1.594 m | 1.609 m | 0.991 | H5 0.000278; ERA5 surface 0.000286 |
| 2024-11-18 | 1.181 m | 1.167 m | 1.012 | H5 0.000278; ERA5 surface 0.000275 |

The meter candidate is much closer to the independent profile scale than the one-way or two-way nanosecond conversions. This is physical-scale evidence only: it does not verify the field's unit, sign, derivative semantics, or processor application state.

| Date / historical site | Common SNR >= 5 pulses | C7 mean / median bias | C7 RMSE / NMAD | H1-as-meter RMSE | ERA5 sensitivity RMSE |
|---|---:|---:|---:|---:|---:|
| 2024-11-12 / county-road segment | 14 | -1.823 / -1.838 m | 1.826 / 0.046 m | 0.252 m | 0.240 m |
| 2024-11-18 / farm-road segment | 70 | -1.571 / -1.549 m | 1.575 / 0.111 m | 0.406 m | 0.442 m |

These same-pulse results are frozen historical checks from one road site per date, not absolute-accuracy certification. From November 12 to November 18, the full-granule delay median changes by about -0.413 m under the unverified meter convention, while the SNR ≥ 5 C7 mean and median biases change by +0.252 m and +0.288 m. A larger positive omitted delay on November 12 would predict a more negative residual, so the direction is consistent with the first-order missed-delay hypothesis. Different sites, flight conditions, and surface contexts, plus the imperfect magnitude match, prevent causal attribution. The public metadata inventory lists 273 L1B H5 files on those same two dates; it provides no unseen date for prospective validation.

The remaining C7 mean residuals are approximately -1.82 m and -1.57 m in the two high-SNR samples. Geoid application, source frame, and range calibration cannot be separated from the joint residual with available evidence. No additive component budget is supported, and no corrected Refh is qualified for absolute NAVD88/3DEP use. Required CASALS team inputs and the direct answers to the seven attribution questions are recorded in the [final attribution report](../outputs/notebooks/10_3dep_elevation_validation/final_bias_attribution/final_attribution_report.md); the complete tables and provenance are in the [final bias attribution package](../outputs/notebooks/10_3dep_elevation_validation/final_bias_attribution/), with a [package SHA-256 manifest](../outputs/notebooks/10_3dep_elevation_validation/final_bias_attribution/artifact_sha256_manifest.csv).

The independent atmospheric calculation follows [ERA5 pressure-level data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=access), the [ARCO-ERA5 public data layout](https://github.com/google-research/arco-era5/blob/main/README.md), and the [Ciddor refractivity formulation](https://opg.optica.org/ao/abstract.cfm?URI=ao-35-9-1566). Rebuild the package from the repository root with `python -m research.reference.final_bias_validation --output outputs/notebooks/10_3dep_elevation_validation/final_bias_attribution`.

## 4. Gross-error behavior

The central and tail metrics tell different stories. For the November 12 SNR ≥ 3 holdout, 20 of 4,274 reference-return-to-ground residuals exceed 2 m (0.47%); none of the 445 SNR ≥ 5 residuals exceed 2 m. For November 18, one of 8,623 SNR ≥ 3 residuals exceeds 2 m (0.012%), and none of the 6,615 SNR ≥ 5 residuals do. In the all-valid populations, the corresponding exceedance counts are 5,646/14,885 for November 12 and 150/8,924 for November 18. Thus the lower-SNR and all-valid RMSE values are dominated by long tails even where the central residual distributions are tight. These are large differences between CASALS reference-return height and the local 3DEP class-2 surface; they do not establish that a CASALS pulse is a ground return or that a difference is a confirmed ground-elevation error. Large cases remain listed with pulse, sweep, track, coordinates, SNR, and local-reference support. The spatial residual map caps its diverging color scale at the absolute 97th percentile to show central patterns; every difference above 2 m remains outlined and listed in the linked table.

## 5. Land-cover waveform response

Eleven of the 12 manifest ROIs contain pulse centers: nine primary polygons across eight sites and two exploratory November 12 polygons. The November 18 water ROI has zero centers and remains listed as excluded. All 128,427 available waveforms passed input validity checks and remain in ROI denominators. Amplitude is native H5 counts, not reflectance. Candidate fractions are screen rates, not return-detection probabilities.

| Granule / ROI | Tier | Pulses | Candidate-screen rate | Median Refh SNR | Median amplitude (counts) | Candidate width (bins) | Candidate energy (counts·bin) |
|---|---|---:|---:|---:|---:|---:|---:|
| Nov 12 open field 1 | Primary | 17,974 | 29.1% | 2.38 | 656 | 9.38 | 9,227 |
| Nov 12 open field 2 | Primary | 17,171 | 25.5% | 2.26 | 665 | 8.04 | 9,468 |
| Nov 12 forest | Primary | 32,014 | 4.0% | 1.91 | 495 | 7.86 | 7,607 |
| Nov 12 road | Exploratory | 3,820 | 26.9% | 2.25 | 588 | 7.83 | 9,173 |
| Nov 12 pond | Exploratory | 10,160 | 5.6% | 1.95 | 479 | 8.13 | 7,504 |
| Nov 18 open field 1 | Primary | 7,140 | 98.2% | 7.12 | 1,799 | 8.39 | 16,451 |
| Nov 18 open field 2 | Primary | 8,924 | 97.1% | 6.07 | 1,992 | 7.99 | 17,090 |
| Nov 18 road | Primary | 1,040 | 91.8% | 6.34 | 1,690 | 8.04 | 15,349 |
| Nov 18 south forest 1 | Primary | 3,503 | 73.8% | 3.49 | 864 | 13.51 | 15,374 |
| Nov 18 south forest 2 | Primary | 4,329 | 65.2% | 3.25 | 864 | 11.54 | 13,804 |
| Nov 18 northwest woodlot | Primary | 22,352 | 75.9% | 4.48 | 1,467 | 9.18 | 18,187 |

The two adjacent southern forest polygons on November 18 are one site, not independent replicates. The northwest woodlot pulse footprint appears to follow the crop-canopy edge in the 2022 NAIP view, so interpret that sample as edge-mixed rather than forest interior. Candidate morphology values use candidate waveforms only; no width or energy is assigned to screened-out pulses. The two acquisition dates differ markedly in SNR, amplitude, and candidate rate, so cross-date differences are confounded with acquisition and instrument conditions.

Matched-track comparisons retain common track IDs within each granule and combine adjacent ROIs from the same site. Differences are open-field minus comparison-site medians across the shared tracks; sweep/time separation is descriptive.

| Granule | Open field vs. site | Shared tracks | Δ median SNR | Δ median amplitude (counts) | Δ candidate rate | Median sweep / time separation |
|---|---|---:|---:|---:|---:|---:|
| Nov 12 | East field vs. forest | 0 | — | — | — | — |
| Nov 12 | Southeast field vs. forest | 149 | +0.530 | +252.5 | +0.171 | 2,443.5 / 4.344 s |
| Nov 18 | Field 1 vs. road | 240 | +0.637 | +19.5 | +0.000 | 923 / 1.641 s |
| Nov 18 | Field 1 vs. woodlot | 256 | +2.251 | +163.8 | +0.159 | 1,593 / 2.832 s |
| Nov 18 | Field 1 vs. south forest | 83 | +4.710 | +1,184.5 | +0.256 | 2,921 / 5.193 s |
| Nov 18 | Field 2 vs. road | 240 | -0.559 | +176.8 | +0.000 | 438.5 / 0.780 s |
| Nov 18 | Field 2 vs. woodlot | 245 | +1.137 | +302.5 | +0.151 | 231 / 0.411 s |
| Nov 18 | Field 2 vs. south forest | 83 | +3.271 | +1,261.5 | +0.263 | 4,295.5 / 7.636 s |

These are within-granule, site-specific descriptions. The 2024-11-12 east-field and forest ROIs have no shared tracks; only one November 12 field–forest pair has common support. Track IDs are not physical beam identities or independent replicates. The sample of sites is too small to support a general land-cover effect, significance claim, or classification accuracy.

## 6. Representative examples and artifacts

Notebook 10's earlier real-pulse boards pair local 3DEP support, raw RX waveforms, stored heights, residuals, and fixed pulse/track/sweep identifiers. The new four-case board shows the minimum and maximum C7 residuals from common-support road test pulses, the largest supported canopy-context difference, and a supported uncertain road-boundary pulse nearest the fixed target edge. Each case includes its NAIP patch, nearby class-2 grid support, raw waveform, pulse/track/sweep identity, SNR, and separately labeled C0, provisional C7, and empirical C8 differences where available. Selection is deterministic and does not use waveform shape or alter evaluation membership. The new road profile uses geometry-derived distance and retains the full test support and long tails. Notebook 11's gallery pairs actual NAIP patches with reread H5 waveforms, background mean, raw maximum, any qualified candidate peak, SNR, and fixed pulse identifiers. NAIP overlays use nominal H5 centers; image-to-footprint registration is not calibrated.

- Current entry points: [elevation reference audit](../notebooks/L1B/geolocation/elevation_reference_audit.ipynb) and [land-cover response](../notebooks/L1B/applications/landcover_response.ipynb).
- [Current datum-comparison summary](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/datum_comparison_summary.json), [selected configuration](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/selected_vertical_configuration.json), [candidate metrics](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/vertical_candidate_metrics.csv), [final elevation metrics](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/final_elevation_metrics.csv), [H5 height audit](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/casals_height_metadata_audit.json), [3DEP Z provenance audit](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/3dep_z_provenance_audit.json), and [artifact SHA-256 manifest](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/datum_comparison_artifact_manifest.csv)
- Datum-comparison figures: [reference-frame overview](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/reference_frame_overview.png), [geoid comparison](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/geoid_correction_comparison.png), [candidate bias and RMSE](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/candidate_bias_rmse_comparison.png), [candidate residuals](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/candidate_residual_distributions.png), [selected vs. baseline](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/selected_vs_baseline.png), [spatial residual maps](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/final_residual_spatial_maps.png), [error by SNR](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/final_error_snr.png), and [ground profile and waveforms](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/final_ground_profile.png)
- [Notebook 10 November 12 error board](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-B_error_board_casals_l1b_20241112T165718_001_02.png), [November 18 error board](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-B_error_board_casals_l1b_20241118T171757_001_02.png), [November 12 spatial board](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-C_spatial_snr_error_casals_l1b_20241112T165718_001_02.png), [November 18 spatial board](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-C_spatial_snr_error_casals_l1b_20241118T171757_001_02.png), [November 12 cases](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-D_real_pulse_cases_casals_l1b_20241112T165718_001_02.png), [November 18 cases](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-D_real_pulse_cases_casals_l1b_20241118T171757_001_02.png), [November 12 ground profile](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-E_ground_profile_casals_l1b_20241112T165718_001_02_track_24.png), [November 18 ground profile](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-E_ground_profile_casals_l1b_20241118T171757_001_02_track_115.png), [raw metrics](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/raw_elevation_metrics.csv), [holdout metrics](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/empirical_holdout_metrics.csv), and [November 12 gross-case table](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/casals_l1b_20241112T165718_001_02_all_gross_errors_snr_ge_3.csv) / [November 18 gross-case table](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/casals_l1b_20241118T171757_001_02_all_gross_errors_snr_ge_3.csv).
- [Notebook 11 November 12 NAIP context](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-A_naip_roi_context_casals_l1b_20241112T165718_001_02.png), [November 18 NAIP context](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-A_naip_roi_context_casals_l1b_20241118T171757_001_02.png), [waveform gallery](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-B_real_waveform_gallery.png), [ROI signal board](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-C_roi_signal_summary.png), [candidate morphology](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-D_candidate_waveform_morphology.png), [matched-track board](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-E_primary_matched_tracks.png), and [ROI response table](../outputs/notebooks/11_landcover_lidar_response/visual_revision/roi_response_summary.csv).
- [Reference-frame audit](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/reference_frame_audit.json), [Notebook 10 input hashes](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/input_manifest.json), [Notebook 11 input hashes](../outputs/notebooks/11_landcover_lidar_response/visual_revision/input_manifest.json), [track-pair summary](../outputs/notebooks/11_landcover_lidar_response/visual_revision/matched_track_site_pair_summary.csv), [full matched-track rows](../outputs/notebooks/11_landcover_lidar_response/visual_revision/matched_track_comparisons.csv), [representative waveform IDs](../outputs/notebooks/11_landcover_lidar_response/visual_revision/representative_waveform_manifest.csv), and [road-target C8 support gate](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/ground_c8_calibration_support_gate.csv).
- [Correction-audit summary](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/correction_audit_summary.json), [application matrix](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/correction_application_matrix.csv), [same-pulse hypothesis metrics](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/physical_hypothesis_comparison.csv), [geoid-model evidence](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/geoid_model_evidence.json), and [audit hashes](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/audit_manifest.json).
- Correction-audit figures: [stored correction fields](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/figures/correction_magnitudes.png), [range-to-height sensitivity](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/figures/range_height_sensitivity.png), [geoid-model comparison](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/figures/geoid_comparison.png), [common-target bias comparison](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/figures/c1_c3_c7_bias_comparison.png), [C7 frame decomposition](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/figures/frame_decomposition.png), and [bias attribution sensitivities](../outputs/notebooks/10_3dep_elevation_validation/correction_audit/figures/bias_attribution.png).

## 7. Limitations and conclusions

The paired LAZ headers identify horizontal EPSG:6347 but carry no vertical CRS axis. Source work-unit metadata identifies NAVD88/GEOID18, and the source-to-clip identity sample verifies Z passthrough within LAS quantization for the sampled points. A complete no-ballpark XYZ operation using the documented GEOID18 grid is available. The CASALS input frame remains assumed because its realization, coordinate epoch, and height-correction conventions are absent from the H5. C7 is therefore provisional: the approximately −1.65 m SNR ≥ 5 residual bias is a comparison result, not a datum diagnosis or absolute-accuracy certification. C8 and the earlier approximately −38 m empirical offsets remain empirical alignment values, never geodetic corrections.

Other limits are the two-granule scope, local interpolation and support criteria, operation accuracy metadata of 2.015 m, possible horizontal coordinate/footprint mismatch, an incomplete audit of every clipped LAS point, older NAIP dates, provisional noise-bootstrap screen, pulse/track dependence, small site count, and absence of radiometric calibration. The SNR ≥ 3 and all-valid C7 RMSEs are dominated by retained low-SNR tails. `refh_amp` is not reflectance, candidate fractions are not detection probability, and the available RX data do not establish official multi-return counts or canopy height. The findings support follow-up source-frame and provenance work; they do not support a general land-cover response claim or a certified CASALS absolute-elevation accuracy claim.

### Execution and preservation record

Notebook 10 and 11 were executed from fresh `casals-showcase` kernels with no cell errors (7 and 5 code cells executed). Notebook 10 verified current H5/LAZ hashes and all 26 cached ground-aware artifacts, loaded those results read-only, and wrote the C8 support-gate CSV under `visual_revision/`. Notebook 11 recomputed 128,427 ROI waveforms from the current H5 inputs and refreshed its `visual_revision/` tables and figures; candidate counts matched the prior summary with zero flag transitions across 11 ROIs. The four residual-diagnostic figures and refreshed Notebook 10/11 boards were opened and visually inspected. The residual manifest records source/code/output hashes, and all 24 legacy metric and pulse-ID parity checks passed. The full test suite passed (187 tests); compileall and `git diff --check` passed. The six correction-audit figures were also visually inspected for axes, outlier visibility, case labels, legends, and completeness. NAIP overlays show nominal H5 centers, but image-to-footprint registration remains uncalibrated. The audit read and hashed all 3,604,480 pulses in each source H5; it did not alter Refh or apply a candidate correction. Historical `revised/`, `datum_comparison/`, and `ground_aware/` files, source H5/LAZ files, production code, and notebooks 00-09 were not modified. Preflight hashes and sizes matched for all 96 protected reference and historical-result files; source H5 hashes are recorded separately in the correction-audit manifest.

### 2026-10-15 meeting-informed follow-up

The historical findings above retain their original evidence date. Jeff Lee subsequently confirmed WGS84, ellipsoidal Refh, EGM2008 tide-free, and available mean-tide correction fields. This does not confirm the full processing formula, realization/epoch, field signs, or applied status. The [new alignment experiment](vertical_datum_alignment.md) and [executed notebook](../notebooks/L1B/geolocation/vertical_datum_alignment.ipynb) supersede model-identity uncertainty while preserving those unresolved processing questions. All historical result folders remain unchanged.

### Physical-only closure follow-up (2026-10-15)

The active [vertical datum notebook](../notebooks/L1B/geolocation/vertical_datum_alignment.ipynb) now removes all empirical height fitting and application. It retains only Native, EGM2008-referenced, and provisional GEOID18 physical heights with independent 3DEP evaluation. The recorded source-frame operation is a non-ballpark no-op; source realization/epoch and correction applied states remain unverified. Strict physical closure is not achieved. Earlier empirical results above and all historical run directories are preserved as historical evidence, not part of the current method. See the [current physical closure report](vertical_datum_alignment.md).
