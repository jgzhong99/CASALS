# CASALS L1B Elevation and Waveform Response Validation

**Status (2026-10-12):** A GEOID18-consistent 3D CASALS-to-3DEP comparison is complete for two granules. The selected transformation remains provisional because the CASALS source realization, coordinate epoch, and Refh processing conventions are undocumented. The empirical alignment and land-cover waveform results remain descriptive.

## Scientific questions

1. How do native `Refh`, H5 scalar corrections, a GEOID18-consistent 3D transformation, and a separate empirical alignment compare with the documented 3DEP reference on spatially held-out pulses?
2. Within the two granules, how do H5 Refh SNR, native amplitude, provisional candidate-screen rate, and candidate-waveform morphology vary across historically image-interpreted ROIs and matched track pairs?

The vertical-frame comparison, earlier empirical holdout, and waveform analysis answer different questions. The available 3D path follows the documented 3DEP GEOID18 reference, but its input-frame assumption is unresolved, so its residuals are not certified absolute accuracy. The waveform comparison describes these sampled sites and acquisitions; it does not establish a general land-cover effect.

## Data and methods

The study uses two CASALS L1B H5 granules from 2024-11-12 and 2024-11-18 (3,604,480 pulses each), their paired 3DEP LAZ files, the existing land-cover ROI manifest, and actual NAIP imagery. The NAIP metadata dates are 2023-06-02 for the November 12 granule and 2022-06-15 for November 18. These NAIP images predate the corresponding CASALS acquisitions by 529 and 887 days, respectively. The imagery is historical context, not a contemporaneous ground label.

For elevation comparison, ASPRS class-2 points were streamed into a 0.75 m grid and locally interpolated. Test surfaces required at least six class-2 neighbors within 3 m, slope no greater than 5 degrees, and local plane residual NMAD no greater than 0.20 m. A deterministic sample of every 10,000th LAZ record was withheld from the grid for an independent interpolation check. The check produced plane RMSEs of 0.047 m (385 supported points; 361 acceptable) for November 12 and 0.050 m (6,376 supported; 6,115 acceptable) for November 18. This checks the local interpolation against held-out 3DEP points only; it does not measure CASALS accuracy.

In the initial native-height comparison, H5 longitude/latitude was transformed to EPSG:6347 for horizontal matching, with no vertical transformation. The difference is defined as CASALS H5 `Refh` minus local 3DEP plane height. Raw metrics retain all supported and acceptable observations. Separate empirical offsets are the median raw differences at a predeclared open-field calibration site; residual metrics are computed at a different open-field test site after subtracting that offset. SNR strata are applied consistently to the raw and holdout populations.

The archived clip sidecars link each LAZ to the current H5 basename and point count. They identify `MD_Southeast_1_2019` / `MD_Southeast_2019_D20` for November 12 and `NC_HurricaneFlorence_9_2020` / `NC_HurricaneFlorence_2020_D20` for November 18. Reprojected clip bounding boxes agree with the LAZ headers within 0.605 m and 0.200 m, respectively. The sidecars record horizontal output EPSG:6347 and EPSG:5703/GEOID18 vertical context, but their `vertical_datum_transform_applied=false` flag alone does not prove point-level Z preservation. The new point-identity sample below independently checks Z values against small source EPT nodes.

For waveform response, current H5 RX waveforms, `bg_mean`, `refh_snr`, and `refh_amp` were reread and features recomputed for 128,427 ROI pulse centers. Previous candidate features and flags were not reused as current results. A pulse-by-pulse comparison of previous and recomputed flags found zero gains, losses, or changes across those same pulse IDs. The provisional candidate screen requires a local peak to pass the height, prominence, and width criteria together. A previously computed full-window threshold was reused after checking configuration and current-data consistency; it was estimated from 6,144 real standardized background-tail windows per granule: 10.885 sigma on November 12 and 10.195 sigma on November 18. Simulated null-bootstrap false-alarm rates were 0.0058 and 0.0029. These are noise-only checks under the bootstrap assumptions, not calibrated detection probabilities. Morphology is summarized only for waveforms that pass the candidate screen.

## Native-Refh baseline and earlier full-calibration empirical holdout

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

## Gross-error behavior

The central and tail metrics tell different stories. For the November 12 SNR ≥ 3 holdout, 20 of 4,274 residuals exceed 2 m (0.47%); none of the 445 SNR ≥ 5 residuals exceed 2 m. For November 18, one of 8,623 SNR ≥ 3 residuals exceeds 2 m (0.012%), and none of the 6,615 SNR ≥ 5 residuals do. In the all-valid populations, the corresponding exceedance counts are 5,646/14,885 for November 12 and 150/8,924 for November 18. Thus the lower-SNR and all-valid RMSE values are dominated by long tails even where the central residual distributions are tight. Gross cases are retained and listed with pulse, sweep, track, coordinates, SNR, and local-reference support; they were not removed to improve the metrics. The spatial residual map caps its diverging color scale at the absolute 97th percentile to show central patterns; every residual above 2 m remains outlined and listed in the linked table.

## Vertical-datum audit and selected provisional path

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
| C8 | `Refh − calibration-only median(C0 residual)` | Separate empirical alignment; never a datum candidate |

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

On the independent-test pulses, the SNR ≥ 5 C7 comparison has a −1.67 m bias and 1.67 m RMSE on November 12 (445 / 478 eligible, 93.1% support) and a −1.65 m bias and 1.65 m RMSE on November 18 (6,615 / 6,615). The central NMAD is about 0.088 m for both. Native C0 remains about −37.6 to −38.0 m from the target. C8 remains a separate site-calibrated baseline: its current SNR ≥ 5 RMSE is 0.113 m and 0.098 m, compared with the previous full-calibration values of 0.132 m and 0.097 m. The differences are −0.019 m and +0.001 m. This is a descriptive comparison: the current C8 offset uses half the calibration ROI, whereas the earlier analysis used the full calibration ROI, and the common-support rules differ.

The SNR ≥ 3 and all-valid test tails remain visible in the saved metrics and figures. C7 SNR ≥ 3 RMSE is 8.65 m on November 12 and 1.73 m on November 18; the all-valid RMSE is 83.84 m and 19.06 m. These high values arise from low-SNR gross errors and are not removed. C7's lower central NMAD does not certify absolute vertical accuracy.

## Land-cover waveform response

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

## Representative examples and artifacts

Notebook 10's earlier real-pulse boards pair local 3DEP support, raw RX waveforms, stored heights, residuals, and fixed pulse/track/sweep identifiers. The new datum-comparison profile uses the independent-test track with the most SNR ≥ 3 supported pulses and plots those pulses against the re-queried 3DEP local plane; the error maps and SNR figure retain the full test support and gross tails. Notebook 11's gallery pairs actual NAIP patches with reread H5 waveforms, background mean, raw maximum, any qualified candidate peak, SNR, and fixed pulse identifiers. NAIP overlays use nominal H5 centers; image-to-footprint registration is not calibrated.

- [Notebook 10](../notebooks/10_3dep_elevation_validation.ipynb) and [Notebook 11](../notebooks/11_landcover_lidar_response.ipynb)
- [Current datum-comparison summary](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/datum_comparison_summary.json), [selected configuration](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/selected_vertical_configuration.json), [candidate metrics](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/vertical_candidate_metrics.csv), [final elevation metrics](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/final_elevation_metrics.csv), [H5 height audit](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/casals_height_metadata_audit.json), [3DEP Z provenance audit](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/3dep_z_provenance_audit.json), and [artifact SHA-256 manifest](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/datum_comparison_artifact_manifest.csv)
- Datum-comparison figures: [reference-frame overview](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/reference_frame_overview.png), [geoid comparison](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/geoid_correction_comparison.png), [candidate bias and RMSE](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/candidate_bias_rmse_comparison.png), [candidate residuals](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/candidate_residual_distributions.png), [selected vs. baseline](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/selected_vs_baseline.png), [spatial residual maps](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/final_residual_spatial_maps.png), [error by SNR](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/final_error_snr.png), and [ground profile and waveforms](../outputs/notebooks/10_3dep_elevation_validation/datum_comparison/figures/final_ground_profile.png)
- [Notebook 10 November 12 error board](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-B_error_board_casals_l1b_20241112T165718_001_02.png), [November 18 error board](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-B_error_board_casals_l1b_20241118T171757_001_02.png), [November 12 spatial board](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-C_spatial_snr_error_casals_l1b_20241112T165718_001_02.png), [November 18 spatial board](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-C_spatial_snr_error_casals_l1b_20241118T171757_001_02.png), [November 12 cases](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-D_real_pulse_cases_casals_l1b_20241112T165718_001_02.png), [November 18 cases](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-D_real_pulse_cases_casals_l1b_20241118T171757_001_02.png), [November 12 ground profile](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-E_ground_profile_casals_l1b_20241112T165718_001_02_track_24.png), [November 18 ground profile](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/10-E_ground_profile_casals_l1b_20241118T171757_001_02_track_115.png), [raw metrics](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/raw_elevation_metrics.csv), [holdout metrics](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/empirical_holdout_metrics.csv), and [November 12 gross-case table](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/casals_l1b_20241112T165718_001_02_all_gross_errors_snr_ge_3.csv) / [November 18 gross-case table](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/casals_l1b_20241118T171757_001_02_all_gross_errors_snr_ge_3.csv).
- [Notebook 11 November 12 NAIP context](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-A_naip_roi_context_casals_l1b_20241112T165718_001_02.png), [November 18 NAIP context](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-A_naip_roi_context_casals_l1b_20241118T171757_001_02.png), [waveform gallery](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-B_real_waveform_gallery.png), [ROI signal board](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-C_roi_signal_summary.png), [candidate morphology](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-D_candidate_waveform_morphology.png), [matched-track board](../outputs/notebooks/11_landcover_lidar_response/visual_revision/11-E_primary_matched_tracks.png), and [ROI response table](../outputs/notebooks/11_landcover_lidar_response/visual_revision/roi_response_summary.csv).
- [Reference-frame audit](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/reference_frame_audit.json), [Notebook 10 input hashes](../outputs/notebooks/10_3dep_elevation_validation/visual_revision/input_manifest.json), [Notebook 11 input hashes](../outputs/notebooks/11_landcover_lidar_response/visual_revision/input_manifest.json), [track-pair summary](../outputs/notebooks/11_landcover_lidar_response/visual_revision/matched_track_site_pair_summary.csv), and [representative waveform IDs](../outputs/notebooks/11_landcover_lidar_response/visual_revision/representative_waveform_manifest.csv).

## Limitations and conclusions

The paired LAZ headers identify horizontal EPSG:6347 but carry no vertical CRS axis. Source work-unit metadata identifies NAVD88/GEOID18, and the source-to-clip identity sample verifies Z passthrough within LAS quantization for the sampled points. A complete no-ballpark XYZ operation using the documented GEOID18 grid is available. The CASALS input frame remains assumed because its realization, coordinate epoch, and height-correction conventions are absent from the H5. C7 is therefore provisional: the approximately −1.65 m SNR ≥ 5 residual bias is a comparison result, not a datum diagnosis or absolute-accuracy certification. C8 and the earlier approximately −38 m empirical offsets remain empirical alignment values, never geodetic corrections.

Other limits are the two-granule scope, local interpolation and support criteria, operation accuracy metadata of 2.015 m, possible horizontal coordinate/footprint mismatch, an incomplete audit of every clipped LAS point, older NAIP dates, provisional noise-bootstrap screen, pulse/track dependence, small site count, and absence of radiometric calibration. The SNR ≥ 3 and all-valid C7 RMSEs are dominated by retained low-SNR tails. `refh_amp` is not reflectance, candidate fractions are not detection probability, and the available RX data do not establish official multi-return counts or canopy height. The findings support follow-up source-frame and provenance work; they do not support a general land-cover response claim or a certified CASALS absolute-elevation accuracy claim.

## Execution and preservation record

Notebook 10 was executed from a clean `casals-showcase` kernel after the GEOID18 operation was pinned, and its embedded figures and tables were visually inspected. Notebook 11 and both notebooks' existing `visual_revision/` folders were left untouched by this vertical-datum task. Current comparison outputs, downloaded public metadata, sampled EPT nodes, and figures are confined to `outputs/notebooks/10_3dep_elevation_validation/datum_comparison/`; the artifact manifest records file sizes and SHA-256 hashes. The original H5/LAZ files, production pipeline, and notebooks 00–09 were not edited by this task.
