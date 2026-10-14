# Research Experiments and Notebooks

This is the formal catalog for the 13 active notebooks. The navigation guide is [notebooks/README.md](../notebooks/README.md). All notebooks run from the repository root. New generated artifacts use `outputs/notebooks/<topic>/<notebook>/<dataset>/`; numbered output folders referenced by prior experiments remain historical inputs and are not moved or overwritten.

## Active notebook catalog

| Notebook | Research question | Input data | Method | Output | Validation status |
| --- | --- | --- | --- | --- | --- |
| [`data/s3_inventory`](../notebooks/data/s3_inventory.ipynb) | What metadata are present under the public CASALS S3 prefix? | Public S3 object listing; optional explicit keys | Paginated listing with optional read-access checks | Inline inventory tables | Previously executed and reviewed; current source is statically validated after migration. |
| [`data/tdms_audit`](../notebooks/data/tdms_audit.ipynb) | What acquisition, Step/Sweep, wavelength, and raw-waveform evidence is stored in one L0 TDMS? | One explicitly selected file under `data/raw/tdms/` | Metadata-only inventory, channel-layout audit, evidence grading, and bounded row slices | `outputs/notebooks/data/tdms_audit/<tdms-stem>/` | Executed on casals_18-11-2024_13-34-53_158-003.tdms (8 code cells, no errors); 32x8 layout complete, wavelength grade C, three rows read by slices. |
| [`refh/quality_and_surfaces`](../notebooks/refh/quality_and_surfaces.ipynb) | What do official Refh quality fields, filtering, support-limited DSMs, and tentative DTMs show? | Two explicit L1B H5 files and read-only historical products | Refh quality summaries, product identity checks, and surface review | `outputs/notebooks/refh/quality_and_surfaces/<granule-stem>/` | Executed on real data after migration (5 code cells, no errors); generated outputs use the topic path. |
| [`refh/classification`](../notebooks/refh/classification.ipynb) | How does the H5-only classifier compare with aligned 3DEP pseudo-labels? | L1B H5, explicit historical transfer, and 3DEP source metadata | Prediction and status-stratified pseudo-reference evaluation | `outputs/notebooks/refh/classification/<granule-stem>/` | Previous real-data results remain historical inputs; not rerun after migration. Static notebook, import, and link checks pass. |
| [`refh/forest_transect`](../notebooks/refh/forest_transect.ipynb) | What observed and support-limited Refh surfaces occur across a historically tree-covered AOI? | L1B H5, existing surface products, and dated NAIP imagery | AOI screening, fixed transects, and surface/profile comparison | `outputs/notebooks/refh/forest_transect/<granule-stem>/` | Migration smoke reached local inputs and AOI screening; the upstream USGS NAIP request returned HTTP 400 before completion. Prior results remain unchanged. |
| [`waveforms/waveform_analysis`](../notebooks/waveforms/waveform_analysis.ipynb) | What do real TX/RX waveforms, detector components, sweep/track structure, and failure cases show? | Explicit L1B H5 | Component extraction, bounded waveform examples, and quality associations | `outputs/notebooks/waveforms/waveform_analysis/<granule-stem>/` | Executed on real data after migration (6 code cells, no errors); generated outputs use the topic path. |
| [`waveforms/waveform_gallery`](../notebooks/waveforms/waveform_gallery.ipynb) | How do selected waveform shapes differ from their separate spatial land-cover evidence? | L1B H5, saved detector features, and dated NAIP assets | Selected pulse plots with spatial context | `outputs/notebooks/waveforms/waveform_gallery/<granule-stem>/` | Previous execution and visual review are documented; not rerun after migration. Static notebook, import, and link checks pass. |
| [`waveforms/waveform_decomposition`](../notebooks/waveforms/waveform_decomposition.ipynb) | How do exploratory Gaussian fits describe six selected RX waveforms? | L1B H5 and preserved selected waveform/reflection records | Bounded fit, residual, and model-order sensitivity comparisons | `outputs/notebooks/waveforms/waveform_decomposition/<granule-stem>/` | Previous execution is documented; all six default fits remain boundary-limited. Not rerun after migration. |
| [`waveforms/nadir_window_reflection`](../notebooks/waveforms/nadir_window_reflection.ipynb) | Is a bounded early-bin signature recurring in selected real-H5 sweep blocks? | L1B H5 and read-only waveform-gallery metadata | Fixed-bin screening, synthetic controls, and selected-row diagnostics | `outputs/notebooks/waveforms/nadir_window_reflection/<granule-stem>/` | Prior bounded run is documented; anomalies remain unattributed and no mitigation is applied to real RX. Not rerun after migration. |
| [`geolocation/peak_geolocation`](../notebooks/geolocation/peak_geolocation.ipynb) | How do production H1 and audited H2 map raw-argmax and detector bins? | Explicit L1B H5 and component records | Same-pulse H1/H2 closure and candidate comparison | `outputs/notebooks/geolocation/peak_geolocation/<granule-stem>/` | Previous real-data execution is documented; not rerun after migration. Static notebook, import, and link checks pass. |
| [`geolocation/beam_geometry`](../notebooks/geolocation/beam_geometry.ipynb) | How do stored instrument, window, and Refh positions relate in ECEF and local ENU? | L1B H5, selected waveform cases, and existing surfaces | Field inventory and H1/H2 geometry plots | `outputs/notebooks/geolocation/beam_geometry/<granule-stem>/` | Executed on real data after migration (8 code cells, no errors); generated outputs use the topic path. |
| [`geolocation/elevation_reference_validation`](../notebooks/geolocation/elevation_reference_validation.ipynb) | Do two CASALS/3DEP pairs pass reference-frame and Z-processing gates? | H5/LAZ pairs and frozen historical validation artifacts | Source-frame audit and read-only residual/bias report review | Existing historical validation outputs under `outputs/notebooks/10_3dep_elevation_validation/` | Previously executed; historical inputs remain at their original paths. Not rerun after migration. |
| [`applications/landcover_response`](../notebooks/applications/landcover_response.ipynb) | How do H5 signal and waveform features vary across reviewed land-cover ROIs? | Two L1B H5 granules, ROI manifest, historical NAIP, and membership inputs | Full-ROI response summaries and matched-track sensitivity analysis | `outputs/notebooks/applications/landcover_response/multi_granule_full_roi/` | Prior full-granule run is documented; historical images and memberships remain read-only inputs. Not rerun after migration. |

The former numbering remains in historical execution reports and output directories to preserve provenance. Those labels are not current notebook entry points. Previous clean-kernel execution and science review evidence is in [notebook_scientific_review.md](notebook_scientific_review.md) and [casals_l1b_validation_report.md](casals_l1b_validation_report.md).

## Refh quality, surfaces, and classification

- Core implementation: `casals_l1b/refh.py`, `refh_export.py`, `refh_filter.py`, `refh_dsm.py`, `refh_ground.py`, `classification.py`, and `evaluation.py`.
- Classifier parameter research: `research/classification/debug_classifier_params.py` and `research/classification/explore_3dep_like_rules.py`.
- The strict DSM is the observed refh surface; fill values remain support-limited. The DTM and ground candidate classes are tentative CASALS-derived products.
- H5-only prediction and pseudo-reference evaluation are separate. Transferred 3DEP labels are not independent ground truth.

## Waveform components and candidate geolocation

- `casals_l1b/waveform.py` contains the shared waveform preprocessing and detector. `casals_l1b/peaks_cli.py` provides bounded H5 feature extraction.
- `casals_l1b/geolocation.py` maps existing component rows to segment and refh-anchored beam candidates; it does not run the detector.
- `research/geolocation/geolocate_sweep_bins_from_beam_angle_rules.py` preserves the broader historical angle-rule study. `docs/casals_l1b_pulse_waveform_geolocation_notes.md` retains the explanatory geometry note.
- Internal refh closure and segment/beam agreement do not independently certify secondary-return locations. The current real-data measurements and caveats are reported in the improvement report.

## 3DEP comparison

- Formal commands: `python -m casals_l1b reference download`, `reference diagnose`, and `reference transfer`.
- Current classification comparison: `notebooks/refh/classification.ipynb`.
- Historical offset and transfer notebooks remain in `research/archived_notebooks/refh/`.
- Empirical height alignment is not a datum transformation. The transfer status and confidence fields define the pseudo-label population and must be reported with any comparison.

## Preserved research records

The 21 former topic notebooks and four superseded synthetic root demos remain under `research/archived_notebooks/` with their saved outputs. The active collection contains 13 topic-organized notebooks. Historical evidence is inventoried in [refactor_baseline.md](refactor_baseline.md); implementation history is recorded in [refactor_inventory.md](refactor_inventory.md).

Presentations, notes, PDFs, and source references remain under `docs/presentations/` and `docs/references/`. See [references/README.md](references/README.md).

## Elevation reference gate and land-cover response

- Notebook 10 records the H5/3DEP reference evidence, AOI-specific PROJ operation/grid availability, and blockers. It does not compute height residuals while the source frames and historical Z handling remain unresolved.
- Notebook 11 uses every nominal pulse center inside the reviewed NAIP ROIs from both existing H5 granules. It preserves `good_snr`, fixed SNR thresholds, invalid/no-candidate denominators, and same-track sensitivity rows separately. Three-sigma waveform screening is descriptive, not an official return count.
- Research helper and focused tests: `research/l1b_validation.py` and `tests/test_l1b_validation.py`. The existing 3DEP and land-cover results remain at `outputs/notebooks/10_3dep_elevation_validation/` and `outputs/notebooks/11_landcover_lidar_response/` as historical evidence and read-only inputs. New execution output uses the topic-organized destinations documented above. See [casals_l1b_validation_report.md](casals_l1b_validation_report.md) for evidence, counts and limitations.
