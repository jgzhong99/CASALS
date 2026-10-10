# CASALS showcase final execution and review

Executed and reviewed on 2026-10-10. These artifacts illustrate real L1B data and tentative research products, not official CASALS L2 products.

## Scope and inputs

This round started from a clean Git worktree. Modified files: notebooks 05, 06 and 07; `research/showcase.py`; `tests/test_showcase.py`; this report; `README.md`; `docs/experiments.md`. No new branch, dependency, classifier, detector, production geolocation algorithm or framework was introduced. Before-comparison copies of all three showcase output trees are retained at `.compile_tmp/before10/`.

Both raw H5s were screened: `data/raw/casals_l1b/casals_l1b_20241112T165718_001_02.h5` and `casals_l1b_20241118T171757_001_02.h5`. Each has 3,604,480 pulses, 256 tracks and 2,728 RX samples. Final presentation cases use November 18. Only the six selected gallery RX/TX records and one selected geometry RX record were loaded; no full waveform copy or new full point cloud was generated.

Existing workflow metadata `outputs` supplied the DSM, DTM, masks and classified LAZ paths. November 18 DSM is the existing 2 m Notebook 01 product. November 18 tentative DTM is the existing 10 m showcase-derived product from the earlier implementation, not regenerated this round. November 12 uses its existing 5 m Notebook 01 DTM and earlier showcase DSM. Classification metadata is explicitly pinned to `outputs/classification/full_20261009/<granule>/<granule>_run_metadata.json`. The detector config/component tables are pinned to `outputs/notebooks/03_waveform_analysis/peaks/<granule>/run_metadata.json`.

`find_products()` now validates the actual recorded H5 path (`source_h5`, `input_h5`, `inputs.h5_path` or `config.h5_path`), supports an explicit metadata path, returns that selected metadata and rejects ambiguous automatic matches. Filename or directory stem is insufficient. `load_classes()` additionally reuses the evaluation module's CASALS VLR reader, validates LAZ source H5, rejects duplicate/out-of-domain indices, requires requested coverage and compares original longitude/latitude/Refh identity (1e-7 degrees / 1 mm tolerances) for the requested subset. No accuracy evaluation or relabelling is implied.

Historical RGB evidence comes from the [USGS NAIP ImageServer](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer): 2022-06-15 for November 18 and 2023-06-02 for November 12. Existing imagery crops were reused. This round additionally saved small local candidate/final-case crops and 1 m overview exports for both footprints; the initial 0.5 m full-extent November 12 overview exceeded the service image-size limit and was replaced with an explicitly coarser 1 m overview. Native catalog resolution is 0.6 m / 0.3 m, respectively. Adjacent JSON retains catalog dates/tile CRS, locked raster IDs, requested and returned EPSG:32618 extent and retrieval metadata. No fitted shift, independent footprint calibration or 3DEP height overlay was used.

## Notebook 05: retained transect, substantive noise diagnosis

The historical image has recognizable tree crowns along the elevated sampled strip. The AOI remains **probable forest canopy**, never confirmed 2024 forest. The fixed line remains appropriate for surface/classification discussion; lower Refh zones are not automatically open ground.

| Parameter | Final real-data result |
| --- | --- |
| AOI, EPSG:32618 | E 347970–348130 m; N 4005970–4006130 m |
| A / B | (347975, 4006048) / (348125, 4006048) m |
| Length / buffer | 150 m / half-width 5 m |
| Full Refh population | 36,763; class 1: 13,525; class 2: 6,291; class 7: 16,947 |
| DSM / DTM | 2 m / 10 m, independently sampled with each affine/CRS |
| Profile samples | 151 at 1 m spacing including endpoint |
| Strict / filled DSM support | 145/151 (96.03%) / 151/151 |
| Tentative DTM support | 151/151 |
| Filled-only samples | 6/151 (3.97%) |

The 46.10% noise labels have saved reasons:

| Saved reason | Count | Fraction of noise |
| --- | ---: | ---: |
| scanline HAG outlier | 13,697 | 80.82% |
| above classifier maximum HAG | 1,802 | 10.63% |
| below classifier ground | 1,362 | 8.04% |
| processed-point low density | 86 | 0.51% |

These reasons describe rule assignment, not independently proven physical causes. The classifier's local ground differs from the showcase DTM; reason categories need not exactly match showcase-HAG thresholds. Noise height spans −186.58 to +221.78 m (median −20.78 m); showcase-DTM HAG spans −153.18 to +255.21 m (median 12.62 m). Median SNR is 3.04; 15,890/16,947 (93.76%) have SNR <5. Stored instrument/window geometry is finite for every noise point; finite geometry does not prove valid scattering.

**1,057 noise-labelled points have SNR >=5 and HAG 5–25 m**, all with the saved scanline-outlier reason, across 14 ten-metre distance bins. Their structured overlap with plausible canopy-range returns and historical tree cover supports a possible classification limitation, not confirmed physical canopy returns. They are now highlighted with open rust-coloured circles in the main profile. Noise spans 73 of 76 sampled tracks and almost the whole line (E 347981.48–348124.95 m); it is not confined to one track. Highest noise fraction is track 8: 139/238 (58.4%); track 55 has 343/637 (53.8%). Full per-track counts are saved.

Deleting noise materially affects interpretation: in SNR >=5 / HAG 0–25 m strata, a ten-metre bin's 95th-percentile upper elevation changes by up to **3.170 m**. Unrestricted P95/P99 differences can reach 102.13/204.09 m because extreme heights dominate, so those are not canopy-height changes. No points or class labels were removed from CSV/statistics. Deterministic scatter thinning only improves display density; the existing full-height inset retains all records.

Blue strict DSM, dashed orange filled DSM and green tentative DTM remain distinct, with 10 m DTM resolution stated. Nodata/support gaps remain NaN. Added `transect_noise_diagnostic.png`, `noise_diagnostic.json`, `transect_track_noise.csv` and `transect_upper_structure.csv`; expanded point CSV and selection metadata record the checks. This board is ready for a research meeting **with the classification limitation stated**.

## Notebook 06: spatial diversity without unsupported labels

The scalar search used supported |HAG| <1 m, SNR >=5 and one highest-SNR candidate per 50 m grid cell: 183 candidate cells for November 12 and 147 for November 18. Both NAIP coverage overviews and six November 18 local crops were visually reviewed. Candidate lists and `land_cover_search_metadata.json` retain the criteria and decisions. November 18 offers a clear field interior and road corridor, so no mixed-granule gallery was necessary.

Replaced old unknown C (1280702) and D (1280420) with open-field pulse 2827710 and road-context pulse 3082120. A/B/E/F and detector thresholds are unchanged. The final composition is two probable canopy, one probable open field, one probable road/mixed-verge and two unknown signal-condition cases.

| Case | Pulse index | Sweep / track | Morphology | SNR | Land cover | Evidence / confidence |
| --- | ---: | --- | --- | ---: | --- | --- |
| A | 2128614 | 8314 / 55 | Multiple detected components | 9.065 | probable tree canopy | probable; NAIP 2022-06-15 |
| B | 2085162 | 8145 / 81 | Single prominent component | 8.490 | probable tree canopy | probable; NAIP 2022-06-15 |
| C | 2827710 | 11045 / 245 | Single prominent component | 16.669 | probable open field | probable; NAIP 2022-06-15 |
| D | 3082120 | 12039 / 68 | Single prominent component | 12.181 | probable road / mixed verge | probable; NAIP 2022-06-15 |
| E | 1280230 | 5000 / 55 | Broadest selected main component (28.7 bins) | 1.813 | unknown | unknown; NAIP 2022-06-15 |
| F | 1280439 | 5001 / 189 | Low SNR / ambiguous | 1.351 | unknown | unknown; NAIP 2022-06-15 |

C is well inside a historically open cultivated field, with no nearby visible crowns/buildings in the final crop. Its HAG is −0.00493 m; 966 neighbors within 5 m have SNR >=5 and an elevation P10–P90 interval of -33.357 to -33.121 m. Current class 2 is supporting context only. Label: **probable open field**, not confirmed bare ground.

D is nearer the visible road centre than rejected verge candidates 3076303 and 3083622. HAG is +0.37256 m and SNR 12.18, with current class 2. The historical road is narrow and exact footprint registration/size is unvalidated, so label **probable road / mixed verge** does not establish pavement material or a pure hard-surface return. No confirmed building/hard-material example is claimed. E/F remain unknown despite suggestive local tree cover because their low-quality signals cannot be assigned reliably to those structures. A/B are class 7; their probable canopy labels derive from independent imagery, illustrating why labels cannot be inferred from classification.

All six raw RX/TX records were read using the verified sweep/track grid and inferred record axis. Raw argmax and inverse-H2 nearest-bin association pass for all six. E/F exactly reproduce existing detector component bins and raw argmax; A–D are newly detected bounded records under the same saved config, with no pre-existing component row to compare. That distinction is retained; rerunning a detector is not an independent accuracy test. Raw RX is plotted, while existing preprocessing is used only by detection, with no hidden smoothing or normalization. Case E width 28.662 bins remains below the unchanged 30-bin broad flag threshold. Final pulse IDs/positions, evidence dates/confidence, config, components and local checks are saved in CSV/JSON. The gallery and dated evidence board are ready for research slides with probable/unknown labels retained.

## Notebook 07: clearer geometry, unchanged computation

Selected pulse **2128614**, sweep **8314**, track **55**, raw RX argmax **1612**. Coordinates use stored H5 geodetic positions through EPSG:4979 → EPSG:4978 → one local ENU frame; ellipsoidal interpretation remains a working research convention with unconfirmed realization/epoch and correction semantics.

| Stored object | Longitude | Latitude | Source height (m) |
| --- | ---: | ---: | ---: |
| Instrument | −76.687374729520 | 36.187444789359 | 5487.095991454 |
| RWSTART | −76.689713512340 | 36.187342636431 | 221.920886559 |
| RWSTOP | −76.689895115341 | 36.187334702453 | −186.555454367 |
| Official Refh | −76.689820853919 | 36.187337946865 | −19.526905371 |

| Check | Final value |
| --- | ---: |
| Instrument–Refh | 5511.033519661 m |
| RX window length | 408.803786438 m |
| H1 internal closure | 13.118993 mm |
| H2 internal closure | 0.536639 mm |
| ECEF/ENU round trip | 0 m at stored floating precision |
| Instrument distance to RW infinite line | 0.003094610 m |

No numeric regression versus baseline. H1/H2 reuse the audited pure function with independent formula asserts. H2 is **Research-supported bin-center mapping**, not official corrected geolocation or independent absolute accuracy.

The scene retains sweeps 8284–8343, one actual median-time record per sweep sorted by time, 129 displayed Refh, four representative observation rays and 117 supported terrain samples. Added a metre-scale, time-ordered trajectory inset with a real direction arrow and selected window endpoints. Full-altitude views prominently state **vertical dimension compressed ×0.025 for visualization**. Surface/window views use equal metre scale; the separate closure view uses centimetres relative to Refh, with distinct symbols.

The only measured airborne anchor is instrument position. Aircraft strokes are schematic there; no separate aircraft GNSS point, lever arm or calibrated attitude is invented. Instrument-to-Refh observation line and receive-window segment are distinct. RWSTART/RWSTOP remain receive-window edges, not surface intersections. The 2D briefing figure now shows both ECEF affine formulas, P0/P1, N/b and the raw maximum waveform marker, in a defined vertical projection with saved off-plane residuals.

Both HTMLs embed Plotly JS, contain zero external script-source tags and have verified serialized ENU axes, manual aspect ratio, camera and traces. **Browser/GUI interaction is untested**: the available computer-use runtime returned no apps/browsers. Camera interaction, legend rendering and live clipping therefore cannot be called PASS. Static geometry figures were inspected and are ready for research slides; HTML live use remains unverified.

## Execution, tests and protection

Existing conda `map`, Python 3.12.13; no dependency installation. Three independent fresh-kernel executions from repository root completed with **6 / 6 / 8 code cells**, zero notebook error outputs. Windows Jupyter/ZeroMQ warnings did not prevent execution. All nine main PNGs (including the new diagnostic), historical search/evidence images and final gallery were actually opened and reviewed; a clipped 3D title and inset/label overlap were corrected and re-executed. Main PNGs are approximately 220 dpi; the 16:9 boards are presentation artifacts.

- `python -m pytest -q`: **52 passed**, up from 47. New tests cover metadata ambiguity/explicit selection, wrong-H5 products, duplicate classified indices, missing requested pulses, wrong LAZ H5 provenance and successful reordered mapping with original-coordinate identity. Earlier transect, CRS/nodata/support, reordered waveform record and H1/H2/ECEF/ENU tests remain.
- `python -m compileall -q casals_l1b research tests`: **PASS**.
- Protected before/after size/mtime inventory: **310 files, zero missing, zero changes**, covering original data/production outputs, notebooks 01–04 and package source. This is not a new full-content hash audit. `backup/outputs/` was absent at the start and remains absent; nothing was deleted or moved. `casals_l1b/geolocation.py` is unchanged.
- Only own showcase outputs were regenerated. Existing production surfaces/LAZ and H5 were read, not written. Before comparisons and existing source/data are preserved.

Machine-readable execution/HTML/PNG/protection record: `outputs/notebooks/05_forest_transect_showcase/casals_l1b_20241118T171757_001_02/final_refinement_validation.json`. Selection, noise, final case and geometry metadata remain beside each notebook's figures.

## Final figures and meeting verdict

| Figure | Repository-relative file | Content | Slides |
| --- | --- | --- | --- |
| 05-A | `outputs/notebooks/05_forest_transect_showcase/casals_l1b_20241118T171757_001_02/aoi_overview.png` | Historical NAIP, HAG spatial view and fixed A–B | Yes, research briefing |
| 05-B | `outputs/notebooks/05_forest_transect_showcase/casals_l1b_20241118T171757_001_02/transect_profile.png` | All classes, high-SNR noise, strict/filled DSM and tentative DTM | Yes, research briefing |
| 05-C | `outputs/notebooks/05_forest_transect_showcase/casals_l1b_20241118T171757_001_02/transect_comparison_board.png` | 16:9 spatial/profile board; recommended | Yes, research briefing |
| 05-D | `outputs/notebooks/05_forest_transect_showcase/casals_l1b_20241118T171757_001_02/transect_noise_diagnostic.png` | Relative HAG and saved noise reasons | Yes, research briefing |
| 06-A | `outputs/notebooks/06_waveform_gallery/casals_l1b_20241118T171757_001_02/waveform_gallery.png` | Six actual RX waveforms; separate morphology/land-cover labels | Yes, research briefing |
| 06-B | `outputs/notebooks/06_waveform_gallery/casals_l1b_20241118T171757_001_02/waveform_gallery_evidence.png` | Dated NAIP contexts and actual Refh centres | Yes, research briefing |
| 07-A | `outputs/notebooks/07_georeferencing_geometry/casals_l1b_20241118T171757_001_02/scene_geometry_3d.png` | Instrument trajectory/direction, rays and tentative terrain | Yes, research briefing |
| 07-B | `outputs/notebooks/07_georeferencing_geometry/casals_l1b_20241118T171757_001_02/single_pulse_geometry_3d.png` | Instrument/window/Refh and separate centimetre closure view | Yes, research briefing |
| 07-C | `outputs/notebooks/07_georeferencing_geometry/casals_l1b_20241118T171757_001_02/bin_georeferencing_explanation.png` | Defined vertical projection, raw RX and ECEF H1/H2 equations; recommended | Yes, research briefing |

Standalone `scene_geometry_3d.html` and `single_pulse_geometry_3d.html` are in the same 07 granule directory. Live browser rendering is not verified.

**05: ready** for a probable-canopy/Refh-surface research briefing, including the noise-rule limitation. **06: ready** for illustrative waveform and conservative spatial interpretation, with road/verge mixing and E/F ambiguity retained. **07 static figures: ready** for actual instrument/window geometry and internal bin closure; **HTML interaction: not yet verified**. Remaining scientific limits are historical imagery/footprint registration, tentative ground/HAG, component physical attribution and incomplete official georeferencing semantics. None is converted into a confirmed land-cover or accuracy claim.

Re-execution: start Jupyter at repository root with the existing `casals-showcase` kernel, or execute each notebook independently with `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=casals-showcase --ExecutePreprocessor.timeout=1800`. All source paths/configuration are repository-relative. Changing a pulse/AOI requires fresh visual review; unreviewed cases retain unknown labels. Missing default feature outputs are extracted only for the fixed small subset into the showcase directory; missing surfaces use the existing formal APIs only in that directory.
