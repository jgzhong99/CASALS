# CASALS Notebook Scientific Review

## Scope and baseline

Reviewed and revised active notebooks `00`–`09` on `master`, starting from commit `26421667aebd49c975785ea4cfd8a349fd8bb661`. Changes are limited to the ten active notebooks, three research helpers, their focused tests, and the README/documentation index. No production classifier, waveform detector, or geolocation implementation was changed. Research outputs are isolated under each notebook's `scientific_review/` directory.

Every notebook ran independently from a clean kernel on real input data where applicable. The table records the question, substantive repair, fresh result, figure output, and scientific boundary.

## Notebook review and execution

| Notebook | Main change and evidence from this run | Execution | Scientific status |
| --- | --- | --- | --- |
| `00_s3_data_inventory` | Kept complete paginated `ListObjectsV2` inventory; made `HeadObject` opt-in; matched local H5 by full filename and `Path.stat()`; corrected the Level-2 wording. Live scan: 27 pages, 26,418 objects, 3,432,133,618,446 bytes. | PASS; 1 figure | Counts and bytes describe listed metadata. No object body was downloaded and no full-object contents were verified. |
| `01_refh_quality` | Added dynamic population checks, class identity, CRS/transform/resolution checks, nodata/NaN/Inf/support masks, shared DSM ranges, and DTM support counts. `good_snr=True`: 42,740 of 3,604,480; filtering removed 0 of those rows. Strict DSM: 41,503 supported cells; filled DSM: 46,470; DTM: 18,508/18,508 supported cells. Projection round-trip maximum error: 2.37e-9 m. | PASS; 5 figures | Refh is the official maximum-RX-bin point. Filled surfaces and DTM remain support-limited/tentative and are not new observations. |
| `02_refh_classification` | Separated H5-only prediction from 3DEP evaluation; validated source H5, LAS EPSG:6347, full `point_index` coverage, row identity, and transfer statuses; added fixed-class confusion/F1 summaries and explicit denominators. Strict status 1: n=1,461,182 (40.54% of CASALS rows), accuracy 0.927969, fixed-class macro-F1 0.586718, weighted-F1 0.926050. Status 1+2: n=1,489,955 (41.34%), accuracy 0.922489, macro-F1 0.586738, weighted-F1 0.922636. | PASS; 3 figures | 3DEP transferred labels are pseudo-reference, not ground truth or independent accuracy. |
| `03_waveform_analysis` | Recorded the effective detector configuration and checked sweep/track/pulse identity against H5, including raw argmax and adjacent-bin context. Three sweeps yielded 768 pulse summaries, 3,064 components, and 256 tracks per sweep. Selected pulse 1,280,218 (sweep 5000, track 214): SNR 6.814, raw argmax bin 1694; adjacent bin 1693 differs by 10 counts, within one estimated background sigma. | PASS; 5 figures | Detector components describe waveform structure under the saved settings; adjacent-bin sensitivity and morphology do not establish physical returns. |
| `04_peak_geolocation` | Added audited research H2 bin-center mapping alongside production H1 and retained the legacy Refh-anchored beam diagnostic. H1/H2 raw-argmax closure uses 766 pulses: median 16.591 mm / 0.526 mm, respectively; p95 28.597 mm / 0.537 mm. For 3,064 paired component rows, H1/H2 median/p95 separation is 18.217/68.545 mm; H2/beam is 38.804/173.271 mm; H1/beam is 57.463/236.448 mm. | PASS; 7 figures | H2 is an internal research mapping comparison, not a production change or independent absolute-accuracy test. H1 remains the production mapping. |
| `05_forest_transect_showcase` | Validated classifier metadata against the actual input H5 and preserved full point-index/row identity checks; showed six AOI candidates, a fixed transect, strict/filled support, and saved noise reasons. Transect: 36,763 points; strict DSM support 96.026%, filled DSM and DTM support 100%. Noise labels: 16,947 (46.10%); 1,057 high-SNR points with HAG 5–25 m occur in 14 ten-metre bins. | PASS; 12 figures | 2022 NAIP supports “probable forest canopy,” not exact 2024 footprint truth. Noise reasons reflect classifier rules, not independently confirmed physical causes. |
| `06_waveform_gallery` | Verified all six waveform records against the H5 sweep/track grid and tied each case to dated local NAIP context. A/B remain probable canopy, C probable field, D probable road/mixed verge, and E/F unknown. Added bounded retries for transient NAIP 502/503/504 responses. | PASS; 2 figures; all six cases rendered | Illustrative cases, not a representative land-cover sample. Imagery is dated 2022-06-15 and does not provide 2024 ground truth. |
| `07_georeferencing_geometry` | Kept actual stored instrument, RWSTART/RWSTOP, and Refh fields; made ENU axes metric and separated vertical display compression. Selected pulse 2,128,614: H1 closure 13.119 mm and H2 closure 0.537 mm. | PASS; 3 figures | H2 closure is internal consistency. Vertical convention/epoch semantics remain unconfirmed; aircraft glyph is schematic. |
| `08_waveform_decomposition` | Used fresh Notebook 06/09 metadata; compared the full waveform window with detector/noise-supported windows; counted model parameters and reported BIC/RMSE within each window only. Six pulses fit; 0 fit failures; median RMSE 99.04 digital counts; K=0–5 each occurred once; three cases changed model order across windows and five reached the candidate limit. | PASS; 3 figures | Gaussian terms are descriptive candidates, not validated physical returns. BIC values from different windows are not directly comparable; candidate-limit cases may be underfit. |
| `09_nadir_window_reflection` | Screened a bounded block using each bin's finite-sweep denominator; empty candidate intervals now preserve the input waveform and produce an explicit no-op mitigation result. Block: 90 sweeps × 256 tracks (23,040 pulses), bins 0–600. No recurring band; maximum fixed-bin occupancy 3.33%; 36 pulses had an early sample above 8σ. Strongest event: pulse 3,077,720, bin 415, 11.41σ, unconfirmed. | PASS; 7 figures | One isolated block cannot establish or exclude a reflection mechanism. Thresholds are heuristic; no bins were masked and no template was subtracted. |

## Material before/after changes

| Area | Before | After | Interpretation |
| --- | --- | --- | --- |
| S3 access in 00 | Before: default execution issued 26,418 HeadObject requests; the two historical H5 sizes were assigned to opposite acquisition dates; the L2 conclusion overclaimed absence. | After: default execution issued 0 HeadObject requests; local stat sizes (14,039,701,685 and 14,082,437,919 bytes) match the exact S3 keys and corrected date mapping; conclusion is limited to explicit Level-2 markers in this prefix. | Inventory remained 26,418 objects and 3,432,133,618,446 listed bytes. Matching name and size do not establish download provenance or verify object contents. |
| Evaluation in `02` | The main supported-label score combined transfer statuses 1–3: n=1,608,392, accuracy 0.886491, macro-F1 0.553444. | Primary score now uses strict status 1 only (n=1,461,182); status 1+2 is separately reported. Fixed labels 1/2/7 and each population's denominator are explicit. | These scores use different pseudo-reference populations and a fixed-class summary; they are not a like-for-like classifier performance improvement. |
| Geolocation in `04` | H1 segment mapping was compared with the legacy beam mapping; paired H1/beam median/p95 separation was 57.463/236.448 mm. | Audited H2 is added to the same pulse/bin population; H2/beam median/p95 separation is 38.804/173.271 mm, while H1 and the beam comparison are retained. | The smaller internal discrepancy does not show that either method is closer to true secondary-return position. |
| Reflection screen in `09` | The empty candidate result was not fully tied to a valid-sweep denominator and explicit no-op contract. | Per-bin denominators use finite sweeps; empty intervals produce zero flagged bins and preserve components (6 before/after for the selected comparison), with template subtraction skipped. | The measured null result is retained; no attenuation or masking was introduced to force a candidate. |

Other changes improve source identity, support masks, plots, and explanatory wording without changing scientific algorithms or manufacturing a better metric. The full 48 inline PNG outputs (counts shown above) were generated during the clean-kernel runs and visually checked. Plot support masks preserve gaps; no unsupported raster cells were filled for display. The two prior Matplotlib backend warnings in notebooks 01 and 04 were fixed by selecting the inline backend after imports that configure Agg; both now render actual inline figures without those warnings.

### Principal figure groups

- 00: storage and product-level totals, acquisition dates, and S3 directory structure.
- 01: Refh footprint and signal distributions, product comparison, and strict/filled DSM plus DTM support.
- 02: projected prediction/reference maps, fixed-label confusion matrix, and representative pseudo-reference errors.
- 03: real RX matrix, selected pulse with detector components, and component/rejection diagnostics.
- 04: raw-argmax closure, paired H1/H2/beam differences, candidate cloud, and selected-pulse geometry.
- 05: dated NAIP and HAG context, support-aware surface and transect profiles, comparison boards, and noise reasons.
- 06: six raw waveforms alongside independently dated spatial context.
- 07: scene geometry, one-pulse ENU geometry, and the H1/H2 bin-mapping explanation.
- 08: full-window and detector-supported Gaussian fits, residuals, and fit/model-order summaries.
- 09: early-bin occupancy and candidate bands, selected pulse context, and the explicit no-op mitigation check.

Notebook images are embedded from the same clean-kernel runs; research artifacts are written below the corresponding outputs/notebooks/<notebook>/scientific_review/ directories.

## Validation and preservation

- Clean-kernel execution: all ten notebooks `00`–`09` passed; no stored exception/error outputs. Notebook 00 used the live public S3 listing. Notebooks 01–09 used the explicitly configured real inputs and isolated output paths.
- Tests: `python -m pytest -q -p no:cacheprovider` — **65 passed**.
- Compilation: `python -m compileall -q casals_l1b research tests` — **passed**.
- Git whitespace validation: `git diff --check` — **passed**.
- Preservation check: all 209 baseline output files remain present with matching sizes and SHA-256 hashes. All 13 protected input files retain baseline size/timestamp metadata; protected H5/3DEP files with baseline hashes also match.
- Scope check: production modules, archived notebooks, backup/history, and source data are unchanged. Newly generated research artifacts are under their corresponding notebook `scientific_review/` directories.

Run tests and compile checks from the repository root using the existing `map` environment:

```powershell
& 'C:\Users\zhong267\AppData\Local\miniconda3\envs\map\python.exe' -m pytest -q -p no:cacheprovider
& 'C:\Users\zhong267\AppData\Local\miniconda3\envs\map\python.exe' -m compileall -q casals_l1b research tests
```

## Remaining scientific limits

- Notebook 02 evaluates against transferred 3DEP pseudo-labels. Status filtering, alignment, and confusion accounting are validated; independent truth is unavailable.
- H1/H2 closure compares mappings to the same official Refh anchor. It does not certify secondary-return geolocation or replace production H1.
- Notebook 05/06 land-cover readings rely on historical imagery, tentative DTM/HAG, and selected examples; labels remain probable or unknown.
- Notebook 08 model fits summarize waveform shapes; physical interpretation, model adequacy beyond the tested windows, and independent validation remain open.
- Notebook 09 found no recurring band in one bounded block. The isolated event remains unexplained, and the exploratory thresholds do not justify production filtering or mitigation.

