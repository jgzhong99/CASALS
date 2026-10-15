# Refactor Inventory

Updated 2026-10-15. The path inventory below records the earlier two-research-line layout pass; the third round kept the established directories and simplified the interfaces and duplicated refh code within them. The old paths below are from the layout-pass base commit `e1acecdefa393952526ad810908387c2adb041f4`. The 2026-10-09 report is a historical snapshot; the current active notebook count is 13, and `backup.bat` is currently tracked and present.

## Python code

| Former path | Current path | Role |
| --- | --- | --- |
| `scripts/export_refh_las.py` | `casals_l1b/refh_export.py` | Official refh export CLI; shared refh implementation is in `casals_l1b/refh.py`. |
| `scripts/filter_refh_points.py` | `casals_l1b/refh_filter.py` | Official refh filtering CLI. |
| `scripts/extract_waveform_features.py` | `casals_l1b/peaks_cli.py` | Waveform component and pulse/sweep diagnostics CLI. |
| `scripts/make_refh_dsm.py` | `casals_l1b/refh_dsm.py` | Refh DSM workflow. |
| `scripts/extract_refh_ground.py` | `casals_l1b/refh_ground.py` | Tentative refh ground and DTM workflow. |
| `scripts/classify_and_evaluate_refh.py` | `casals_l1b/classification_cli.py` plus `casals_l1b/evaluation.py` | H5-only prediction with optional reference comparison; evaluation is also independently callable. |
| `scripts/diagnose_3dep_offsets.py` | `research/reference/diagnose_3dep_offsets.py` | Reference-frame and vertical-difference research. |
| `scripts/transfer_3dep_labels_to_casals.py` | `research/reference/transfer_3dep_labels_to_casals.py` | Experimental 3DEP pseudo-label transfer. |
| `scripts/extract_transfer_laz_local_features.py` | `research/reference/extract_transfer_laz_local_features.py` | Transferred-reference feature analysis. |
| `scripts/summarize_refh_error_distributions.py` | `research/refh/summarize_refh_error_distributions.py` | Refh distribution diagnostics. |
| `experiments/classification/debug_classifier_params.py` | `research/classification/debug_classifier_params.py` | Classifier parameter exploration. |
| `experiments/classification/explore_3dep_like_rules.py` | `research/classification/explore_3dep_like_rules.py` | Experimental rule exploration. |
| `experiments/geolocation/geolocate_sweep_bins_from_beam_angle_rules.py` | `research/geolocation/geolocate_sweep_bins_from_beam_angle_rules.py` | Candidate secondary-peak geolocation research. |
| `scripts/animate_pushbroom.py` | `tools/animate_pushbroom.py` | Visualization utility. |
| `scripts/animate_rx_waveforms.py` | `tools/animate_rx_waveforms.py` | Visualization utility. |
| `scripts/detect_h5_utm_zone.py` | `tools/detect_h5_utm_zone.py` | CRS inspection utility. |
| `scripts/download_3dep_lpc.py` | `tools/download_3dep_lpc.py` | 3DEP clip download utility. |
| `scripts/view_lpc_qt_classes.py` | `tools/view_lpc_qt_classes.py` | Classification viewer. |
| `scripts/view_refh_points.py` | `tools/view_refh_points.py` | Official refh point viewer. |

The top-level `scripts/` and `experiments/` packages were removed after migration. The unified entry point exposes only `refh`, `peaks`, and `reference`; the installed console entry point is `casals`.

## Third-round implementation

- `casals_l1b/__main__.py` routes nested group commands directly to their existing implementations. Viewer, animation, and parameter-scan helpers remain standalone.
- `casals_l1b/refh.py` now supplies one unfiltered surface-data reader for DSM and tentative DTM inputs, plus shared refh and surface statistics. DSM and DTM retain their distinct masks, grid semantics, interpolation, and output rules.
- `refh_export.py`, `refh_filter.py`, `refh_dsm.py`, and `refh_ground.py` expose direct callable workflow functions. Their `main()` functions parse CLI arguments and invoke those APIs.
- `casals_l1b/evaluation.py` owns alignment and metrics independently of `classification_cli.py`. H5-only prediction writes `evaluation.status = not_run` when no reference is supplied.
- `casals_l1b/peaks_cli.py` exposes `extract_components()` using the existing detector and bounded H5 reads. `casals_l1b/geolocation.py` maps those records to experimental segment and refh-anchored beam candidates and writes a Parquet candidate table, closure diagnostics, validation summary, and run metadata.
- `tools/view_lpc_open3d.py` was removed as a duplicate generic LAS/LAZ viewer. `tools/view_lpc_qt_classes.py` remains the bounded, class-filterable LAS/LAZ viewer; `tools/view_refh_points.py` remains the H5-specific refh quality viewer.
- The 2026-10-09 report recorded `backup.bat` as deleted; a current checkout check finds it tracked and present. Local `config/local/` and `.compile_tmp/` remain ignored.

## Notebooks

The earlier layout-pass review covered [Refh quality and surfaces](../notebooks/L1B/refh/quality_and_surfaces.ipynb), [Refh classification](../notebooks/L1B/refh/classification.ipynb), [waveform analysis](../notebooks/L1B/waveforms/waveform_analysis.ipynb), and [peak geolocation](../notebooks/L1B/geolocation/peak_geolocation.ipynb). At the time, these used the numbered paths `notebooks/01_refh_quality.ipynb`, `notebooks/02_refh_classification.ipynb`, `notebooks/03_waveform_analysis.ipynb`, and `notebooks/04_peak_geolocation.ipynb`. Their saved outputs were regenerated from local real data and checked after execution; evidence and limits are in [refactor_improvement_report.md](refactor_improvement_report.md).

All 21 former research notebooks and the geolocation archive notes are preserved under `research/archived_notebooks/` with their original filenames and saved cell outputs:

- `introduction/`: `h5_schema_and_georeferencing_audit.ipynb`, `l1b_structure_and_refh.ipynb`, `single_h5_beam_reader.ipynb`.
- `refh/`: `3dep_offset_diagnosis.ipynb`, `3dep_pseudolabel_transfer_qc.ipynb`, `labeled_pointcloud_by_class_reader.ipynb`, `refh_quality_and_dsm.ipynb`.
- `waveform/`: `labeled_pointcloud_peak_ratio_diagnostics.ipynb`, `raw_waveform_noise_peak_diagnostics.ipynb`, `single_sweep_raw_waveform_viewer.ipynb`, `sweep_tx_rx_matrix.ipynb`, `sweep_waveform_matrix.ipynb`, `waveform_feature_extraction.ipynb`, `waveform_features_vs_refh_quality.ipynb`.
- `geolocation/`: `beam_geometry_and_bin_georeference_theory.ipynb`, `beam_range_peak_geolocation_validation.ipynb`, `geolocation_rule_audit_revised.ipynb`, `geolocation_rule_audit_revised_v2.ipynb`, `range_window_bin_mapping_hypothesis.ipynb`, `refh_forward_reconstruction_validation.ipynb`, `sweep_peak_geolocation_validation.ipynb`, plus `ARCHIVE_NOTES.md`.

The four superseded synthetic root demos were moved to
`research/archived_notebooks/superseded_root_demos/`, retaining their saved
outputs and original filenames: `01_refh_data_contract.ipynb`,
`02_waveform_components.ipynb`, `03_beam_geolocation_candidates.ipynb`, and
`04_refh_classification_and_reference.ipynb`. There are 25 archived notebooks
in total. Four notebooks were active at the time of the 2026-10-09 report; the
current active set has 13 notebooks, including `notebooks/L0/data/tdms_audit.ipynb`.

## 2026-10-14 active notebook topic migration

The 12 formerly numbered active notebooks were grouped by topic on 2026-10-14.
The new TDMS audit is the additional thirteenth active notebook. Current
product-level paths are listed alongside that earlier topic layout.

| Former active notebook | Topic path at 2026-10-14 | Current product-level path |
| --- | --- |
| `00_s3_data_inventory.ipynb` | `notebooks/data/s3_inventory.ipynb` | `notebooks/catalog/s3_inventory.ipynb` |
| `01_refh_quality.ipynb` | `notebooks/refh/quality_and_surfaces.ipynb` | `notebooks/L1B/refh/quality_and_surfaces.ipynb` |
| `02_refh_classification.ipynb` | `notebooks/refh/classification.ipynb` | `notebooks/L1B/refh/classification.ipynb` |
| `03_waveform_analysis.ipynb` | `notebooks/waveforms/waveform_analysis.ipynb` | `notebooks/L1B/waveforms/waveform_analysis.ipynb` |
| `04_peak_geolocation.ipynb` | `notebooks/geolocation/peak_geolocation.ipynb` | `notebooks/L1B/geolocation/peak_geolocation.ipynb` |
| `05_forest_transect_showcase.ipynb` | `notebooks/refh/forest_transect.ipynb` | `notebooks/L1B/refh/forest_transect.ipynb` |
| `06_waveform_gallery.ipynb` | `notebooks/waveforms/waveform_gallery.ipynb` | `notebooks/L1B/waveforms/waveform_gallery.ipynb` |
| `07_georeferencing_geometry.ipynb` | `notebooks/geolocation/beam_geometry.ipynb` | `notebooks/L1B/geolocation/beam_geometry.ipynb` |
| `08_waveform_decomposition.ipynb` | `notebooks/waveforms/waveform_decomposition.ipynb` | `notebooks/L1B/waveforms/waveform_decomposition.ipynb` |
| `09_nadir_window_reflection.ipynb` | `notebooks/waveforms/nadir_window_reflection.ipynb` | `notebooks/L1B/waveforms/nadir_window_reflection.ipynb` |
| `10_3dep_elevation_validation.ipynb` | `notebooks/geolocation/elevation_reference_validation.ipynb` | `notebooks/L1B/geolocation/elevation_reference_validation.ipynb` |
| `11_landcover_lidar_response.ipynb` | `notebooks/applications/landcover_response.ipynb` | `notebooks/L1B/applications/landcover_response.ipynb` |

The old numbered notebook source paths are not active duplicates. References
to numbered output directories were audited by role: current notebooks use
them as read-only historical products, pseudo-reference inputs, selected-case
metadata, or preserved validation evidence. The related `research/reference/`
scripts also read and write the existing 3DEP attribution chain under
`outputs/notebooks/10_3dep_elevation_validation/`; its reports and downstream
checks depend on those artifact paths. Those references remain in place to
preserve provenance. New notebook runs retain the topic-based output paths in
`notebooks/README.md`; the elevation-validation notebook is explicitly
read-only over its historical result tree. Historical reports and catalogs
retain their original artifact paths.

The active output-path review also corrected two helper-backed destinations:
waveform analysis now passes the topic/notebook root to the feature extractor,
which adds the granule name once, and peak geolocation writes its extracted
tables into the granule output root without a duplicate granule subfolder. The
waveform-gallery fallback now checks the extractor's actual
`feature_subset/<granule>/run_metadata.json` path. Existing historical result
folders were not changed.

Specific historical path consumers retained after review:

- `01_refh_quality/` supplies the Refh quality notebook's previous LAS and
  raster products; `02_refh_classification/reference_transfer/` supplies the
  classification notebook's existing pseudo-reference.
- `03_waveform_analysis/method_improvement/peaks/` supplies existing waveform
  feature metadata. `06_waveform_gallery/method_improvement/` supplies
  selected-case metadata and imagery to beam geometry, waveform decomposition,
  nadir reflection, and land-cover response. `09_nadir_window_reflection/`
  supplies selected reflection records to waveform decomposition.
- `10_3dep_elevation_validation/` and `11_landcover_lidar_response/` contain
  frozen 3DEP attribution and ROI evidence read by the elevation-validation and
  land-cover notebooks. `research/reference/` scripts form a dependent
  read/write chain within the 10 directory; reports cite those same artifacts.
  They remain at their established paths so reruns and saved analyses stay
  linked.
- Historical report links and `data/reference/landcover_rois/roi_manifest.geojson`
  retain their original output references. These identify evidence sources;
  they are not current notebook entry points or destinations for the
  representative validation runs.

## Local-only and preserved material

- The 2026-10-09 report recorded `backup.bat` as removed; the current checkout has it tracked and present. That report also records removal of empty `.agents/`, `casals_l1b/docs/`, and former script/experiment shells, plus the redundant Open3D LAS/LAZ viewer. Original and large data were preserved.
- `.compile_tmp/` and `backup/` are ignored and untracked. Historical output trees are collected under `backup/outputs/`; `.compile_tmp/` remains local scratch, including `casals_gui_3d_surface.html`.
- `config/local/casals_gui_settings.json` remains on disk but is untracked so workstation-specific settings are not published.
- Original H5, TDMS, 3DEP reference inputs, `Archive/`, archived notebooks, and baseline outputs remain preserved; prior output trees are under `backup/outputs/`.
