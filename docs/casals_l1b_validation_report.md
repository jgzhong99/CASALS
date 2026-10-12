# CASALS L1B Elevation and Waveform Response Validation

**Status (2026-10-12):** Raw CASALS?3DEP comparison and spatially held-out empirical alignment are complete for two granules. The vertical reference frame is unresolved, so strict absolute elevation accuracy is unavailable. Land-cover waveform results are descriptive research screens.

## Scientific questions

1. How do stored CASALS `Refh` heights compare with a locally interpolated 3DEP class-2 ground surface, and how much agreement remains on a separate open-field site after removing a calibration-site median offset?
2. Within the two granules, how do H5 Refh SNR, native amplitude, provisional candidate-screen rate, and candidate-waveform morphology vary across historically image-interpreted ROIs and matched track pairs?

The two analyses answer different questions. A numerical CASALS?3DEP difference can be computed, but its absolute vertical meaning cannot be confirmed. The waveform comparison describes these sampled sites and acquisitions; it does not establish a general land-cover effect.

## Data and methods

The study uses two CASALS L1B H5 granules from 2024-11-12 and 2024-11-18 (3,604,480 pulses each), their paired 3DEP LAZ files, the existing land-cover ROI manifest, and actual NAIP imagery. The NAIP metadata dates are 2023-06-02 for the November 12 granule and 2022-06-15 for November 18?529 and 887 days before the corresponding CASALS acquisitions. The imagery is historical context, not a contemporaneous ground label.

For elevation comparison, ASPRS class-2 points were streamed into a 0.75 m grid and locally interpolated. Test surfaces required at least six class-2 neighbors within 3 m, slope no greater than 5 degrees, and local plane residual NMAD no greater than 0.20 m. A deterministic sample of every 10,000th LAZ record was withheld from the grid for an independent interpolation check. The check produced plane RMSEs of 0.047 m (385 supported points; 361 acceptable) for November 12 and 0.050 m (6,376 supported; 6,115 acceptable) for November 18. This checks the local interpolation against held-out 3DEP points only; it does not measure CASALS accuracy.

H5 longitude/latitude was transformed to EPSG:6347 for horizontal matching, with no vertical transformation. The difference is defined as CASALS H5 `Refh` minus local 3DEP plane height. Raw metrics retain all supported and acceptable observations. Separate empirical offsets are the median raw differences at a predeclared open-field calibration site; residual metrics are computed at a different open-field test site after subtracting that offset. SNR strata are applied consistently to the raw and holdout populations.

The archived clip sidecars link each LAZ to the current H5 basename and point count. They identify `MD_Southeast_1_2019` / `MD_Southeast_2019_D20` for November 12 and `NC_HurricaneFlorence_9_2020` / `NC_HurricaneFlorence_2020_D20` for November 18. Reprojected clip bounding boxes agree with the LAZ headers within 0.605 m and 0.200 m, respectively. The sidecars record horizontal output EPSG:6347, EPSG:5703/GEOID18 vertical context, and `vertical_datum_transform_applied=false`; these checks establish file and horizontal-footprint linkage, not point-level Z preservation.

For waveform response, current H5 RX waveforms, `bg_mean`, `refh_snr`, and `refh_amp` were reread and features recomputed for 128,427 ROI pulse centers. Previous candidate features and flags were not reused as current results. A pulse-by-pulse comparison of previous and recomputed flags found zero gains, losses, or changes across those same pulse IDs. The provisional candidate screen requires a local peak to pass the height, prominence, and width criteria together. A previously computed full-window threshold was reused after checking configuration and current-data consistency; it was estimated from 6,144 real standardized background-tail windows per granule: 10.885 sigma on November 12 and 10.195 sigma on November 18. Simulated null-bootstrap false-alarm rates were 0.0058 and 0.0029. These are noise-only checks under the bootstrap assumptions, not calibrated detection probabilities. Morphology is summarized only for waveforms that pass the candidate screen.

## CASALS?3DEP findings

The LAZ files contain 9,081,045 points (November 12; 4,646,895 class-2) and 117,530,910 points (November 18; 63,226,508 class-2). Locally supported and acceptable surfaces matched 3,369,562 of 3,604,480 H5 pulses (93.5%) on November 12 and 3,457,681 (95.9%) on November 18.

The raw difference distribution is broad and tail-sensitive. The central values cluster near -38 m, but the vertical-frame audit does not establish that this is a datum offset.

| Granule | SNR stratum | Matched n | Mean difference (m) | Median (m) | RMSE (m) | NMAD (m) |
|---|---|---:|---:|---:|---:|---:|
| 2024-11-12 | All valid | 3,369,562 | -12.164 | -37.571 | 93.741 | 43.237 |
| 2024-11-12 | SNR ? 3 | 482,173 | -36.750 | -37.661 | 40.505 | 0.145 |
| 2024-11-12 | SNR ? 5 | 39,811 | -37.648 | -37.658 | 37.666 | 0.120 |
| 2024-11-18 | All valid | 3,457,681 | -31.297 | -37.962 | 45.273 | 0.227 |
| 2024-11-18 | SNR ? 3 | 2,600,908 | -35.970 | -37.996 | 36.568 | 0.137 |
| 2024-11-18 | SNR ? 5 | 1,703,603 | -37.711 | -38.017 | 37.761 | 0.099 |

The separate-site holdout comparison quantifies agreement after each calibration-site median offset is removed. It remains a relative diagnostic, not absolute accuracy.

| Granule | Calibration site ? test site | SNR stratum | Test n / eligible denominator | Empirical offset (m) | Test mean residual (m) | RMSE (m) | NMAD (m) | Absolute P95 (m) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2024-11-12 | East field ? southeast field | All valid | 14,885 / 17,171 | -37.663 | 24.208 | 84.332 | 0.295 | 209.176 |
| 2024-11-12 | East field ? southeast field | SNR ? 3 | 4,274 / 4,852 | -37.659 | 0.271 | 8.522 | 0.117 | 0.278 |
| 2024-11-12 | East field ? southeast field | SNR ? 5 | 445 / 478 | -37.659 | 0.086 | 0.132 | 0.088 | 0.230 |
| 2024-11-18 | Field site 01 ? field site 02 | All valid | 8,924 / 8,924 | -37.980 | 1.547 | 19.123 | 0.095 | 0.243 |
| 2024-11-18 | Field site 01 ? field site 02 | SNR ? 3 | 8,623 / 8,623 | -37.981 | -0.048 | 0.469 | 0.092 | 0.212 |
| 2024-11-18 | Field site 01 ? field site 02 | SNR ? 5 | 6,615 / 6,615 | -37.979 | -0.037 | 0.097 | 0.087 | 0.190 |

## Gross-error behavior

The central and tail metrics tell different stories. For the November 12 SNR ? 3 holdout, 20 of 4,274 residuals exceed 2 m (0.47%); none of the 445 SNR ? 5 residuals exceed 2 m. For November 18, one of 8,623 SNR ? 3 residuals exceeds 2 m (0.012%), and none of the 6,615 SNR ? 5 residuals do. In the all-valid populations, the corresponding exceedance counts are 5,646/14,885 for November 12 and 150/8,924 for November 18. Thus the lower-SNR and all-valid RMSE values are dominated by long tails even where the central residual distributions are tight. Gross cases are retained and listed with pulse, sweep, track, coordinates, SNR, and local-reference support; they were not removed to improve the metrics.

## Land-cover waveform response

Eleven of the 12 manifest ROIs contain pulse centers: nine primary polygons across eight sites and two exploratory November 12 polygons. The November 18 water ROI has zero centers and remains listed as excluded. All 128,427 available waveforms passed input validity checks and remain in ROI denominators. Amplitude is native H5 counts, not reflectance. Candidate fractions are screen rates, not return-detection probabilities.

| Granule / ROI | Tier | Pulses | Candidate-screen rate | Median Refh SNR | Median amplitude (counts) | Candidate width (bins) | Candidate energy (counts?bin) |
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

The two adjacent southern forest polygons on November 18 are one site, not independent replicates. Candidate morphology values use candidate waveforms only; no width or energy is assigned to screened-out pulses. The two acquisition dates differ markedly in SNR, amplitude, and candidate rate, so cross-date differences are confounded with acquisition and instrument conditions.

Matched-track comparisons retain common track IDs within each granule and combine adjacent ROIs from the same site. Differences are open-field minus comparison-site medians across the shared tracks; sweep/time separation is descriptive.

| Granule | Open field vs. site | Shared tracks | ? median SNR | ? median amplitude (counts) | ? candidate rate | Median sweep / time separation |
|---|---|---:|---:|---:|---:|---:|
| Nov 12 | East field vs. forest | 0 | ? | ? | ? | ? |
| Nov 12 | Southeast field vs. forest | 149 | +0.530 | +252.5 | +0.171 | 2,443.5 / 4.344 s |
| Nov 18 | Field 1 vs. road | 240 | +0.637 | +19.5 | +0.000 | 923 / 1.641 s |
| Nov 18 | Field 1 vs. woodlot | 256 | +2.251 | +163.8 | +0.159 | 1,593 / 2.832 s |
| Nov 18 | Field 1 vs. south forest | 83 | +4.710 | +1,184.5 | +0.256 | 2,921 / 5.193 s |
| Nov 18 | Field 2 vs. road | 240 | -0.559 | +176.8 | +0.000 | 438.5 / 0.780 s |
| Nov 18 | Field 2 vs. woodlot | 245 | +1.137 | +302.5 | +0.151 | 231 / 0.411 s |
| Nov 18 | Field 2 vs. south forest | 83 | +3.271 | +1,261.5 | +0.263 | 4,295.5 / 7.636 s |

These are within-granule, site-specific descriptions. The 2024-11-12 east-field and forest ROIs have no shared tracks; only one November 12 field?forest pair has common support. Track IDs are not physical beam identities or independent replicates. The sample of sites is too small to support a general land-cover effect, significance claim, or classification accuracy.

## Representative examples and artifacts

Notebook 10's real-pulse case boards pair local 3DEP support, the raw RX waveform, stored heights, residual, and pulse/track/sweep identifiers. They include high- and moderate-SNR cases plus two retained gross cases. The ground-profile figures use measured horizontal coordinates and preserve raw and empirically offset Refh heights. Notebook 11's gallery pairs actual NAIP patches with reread H5 waveforms, background mean, raw maximum, any qualified candidate peak, SNR, and fixed pulse identifiers. NAIP overlays use nominal H5 centers; image-to-footprint registration is not calibrated.

- [Notebook 10](../notebooks/10_3dep_elevation_validation.ipynb) and [Notebook 11](../notebooks/11_landcover_lidar_response.ipynb)
- [Notebook 10 November 12 error board](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-B_error_board_casals_l1b_20241112T165718_001_02.png), [November 18 error board](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-B_error_board_casals_l1b_20241118T171757_001_02.png), [November 12 spatial board](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-C_spatial_snr_error_casals_l1b_20241112T165718_001_02.png), [November 18 spatial board](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-C_spatial_snr_error_casals_l1b_20241118T171757_001_02.png), [November 12 cases](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-D_real_pulse_cases_casals_l1b_20241112T165718_001_02.png), [November 18 cases](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-D_real_pulse_cases_casals_l1b_20241118T171757_001_02.png), [November 12 ground profile](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-E_ground_profile_casals_l1b_20241112T165718_001_02_track_0.png), [November 18 ground profile](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-E_ground_profile_casals_l1b_20241118T171757_001_02_track_167.png), [raw metrics](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/raw_elevation_metrics.csv), [holdout metrics](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/empirical_holdout_metrics.csv), and [November 12 gross-case table](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/casals_l1b_20241112T165718_001_02_all_gross_errors_snr_ge_3.csv) / [November 18 gross-case table](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/casals_l1b_20241118T171757_001_02_all_gross_errors_snr_ge_3.csv).
- [Notebook 11 NAIP context](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-A_naip_roi_context_casals_l1b_20241118T171757_001_02.png), [waveform gallery](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-B_real_waveform_gallery.png), [ROI signal board](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-C_roi_signal_summary.png), [candidate morphology](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-D_candidate_waveform_morphology.png), [matched-track board](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-E_primary_matched_tracks.png), and [ROI response table](../outputs/notebooks/11_landcover_lidar_response/visual_revision/roi_response_summary.csv).
- [Reference-frame audit](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/reference_frame_audit.json), [Notebook 10 input hashes](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/input_manifest.json), [Notebook 11 input hashes](../outputs/notebooks/11_landcover_lidar_response/visual_revision/input_manifest.json), [track-pair summary](../outputs/notebooks/11_landcover_lidar_response/visual_revision/matched_track_site_pair_summary.csv), and [representative waveform IDs](../outputs/notebooks/11_landcover_lidar_response/visual_revision/representative_waveform_manifest.csv).

## Limitations and conclusions

The paired LAZ headers identify horizontal EPSG:6347 but carry no vertical CRS axis. Historical sidecars report EPSG:5703/GEOID18 context and `vertical_datum_transform_applied=false`; this does not prove that output Z was preserved from the source EPT. CASALS H5 metadata does not establish the Refh vertical datum realization or coordinate epoch. A complete, verified XYZ transformation is unavailable. Consequently, the approximately -38 m raw median differences and empirical offsets are reported as observed values; no offset is called a datum correction and no RMSE or residual is certified as absolute accuracy.

Other limits are the two-granule scope, local interpolation and support criteria, possible horizontal coordinate/footprint mismatch, older NAIP dates, provisional noise-bootstrap screen, pulse/track dependence, small site count, and absence of radiometric calibration. `refh_amp` is not reflectance, candidate fractions are not detection probability, and the available RX data do not establish official multi-return counts or canopy height. The findings support follow-up sampling and frame/provenance work; they do not support a general land-cover response claim or a CASALS absolute-elevation accuracy claim.

## Execution and preservation record

Both revised notebooks were executed from clean `casals-showcase` kernels. `nbformat.validate` passed for both; all 9 code cells completed without notebook error outputs. The full suite passed (118 tests); `compileall` and `git diff --check` passed. Recomputed SHA-256 values for both original H5 and both LAZ inputs match the linked manifests. New analysis artifacts are confined to each notebook's `visual_revision/` output folder. The prior `revised/` outputs, original H5/LAZ inputs, production pipeline, and notebooks 00?09 were not edited by this analysis. The final figures were visually inspected.
