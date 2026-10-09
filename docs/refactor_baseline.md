# CASALS Refactor Baseline

This inventory records the repository and local research assets before the
refactor. Files below `data/`, `outputs/`, `tdms/`, `Archive/`, and the old
`CASALS_L1B` data folders are local and ignored by Git unless stated otherwise.
On 2026-10-09, historical root output trees were moved intact from `outputs/`
to `backup/outputs/`; paths recorded inside historical metadata still identify
where each run was originally written.

## Source baseline

- Base branch: `master`, equal to `origin/master` after `git fetch origin`.
- Base commit: `2f38f9a712328d7bde095fecd83efa8b49e73925` (`2026-10-08`, `update`).
- Work branch: `refactor/reorganize-casals`.
- Git working tree was clean before branch creation.
- Git tree: 129 tracked files, including 34 Python files, 20 notebooks, 4
  presentations, and 30 tracked `.pyc` files under the GUI `__pycache__`.
- There is no root README, package metadata, or test suite at baseline. The only
  tracked dependency file is `requirements-qt-pyvista.txt`.

## Local data and result inventory

| Current location | Contents | Files | Bytes | Treatment |
| --- | --- | ---: | ---: | --- |
| `CASALS_L1B/casals_h5_downloads/` | Two original CASALS L1B H5 files (14,082,437,919 and 14,039,701,685 bytes) | 2 | 28,122,139,604 | Preserve as raw inputs; do not rewrite. |
| `tdms/` | Two original TDMS files, indexes, and local viewer support files | 9 | 37,904,264,158 | Preserve; move only the source TDMS assets. |
| `CASALS_L1B/point_cloud_data/` | 13 LAS/LAZ files, including downloaded 3DEP clips and reusable refh/classification products | 13 | 1,817,693,012 | Preserve; distinguish 3DEP references from generated derived products. |
| `CASALS_L1B/outputs/` | Existing workflow outputs, including tables, metadata, rasters, LAS/LAZ, and videos | 180 | 9,212,210,800 | Preserve until corresponding workflows have been rerun and compared. |
| `CASALS_L1B/outputs/` | Existing workflow outputs, including tables, metadata, rasters, LAS/LAZ, and videos | 180 | 9,212,210,800 | Preserve until corresponding workflows have been rerun and compared. |
| `outputs/` | Older root-level reports, metadata, tables, and visualization legend | 11 | 164,928 | Preserve as baseline evidence. |
| `CASALS_L1B/beam_geolocation_rule_audit_outputs_revised/` | Geolocation audit tables and point-cloud exports | 18 | 35,846,036 | Preserve as research evidence. |
| `CASALS_L1B/notebooks/casals_beam_range_validation_outputs/` | Range/beam validation CSV results | 24 | 4,163,849 | Preserve with its notebook. |
| `CASALS_L1B/notebooks/casals_h5_schema_audit_outputs/` | H5 schema audit CSV results | 8 | 73,651 | Preserve with its notebook. |
| `Archive/` | Ignored legacy references, TDMS research code/notebook, and viewer materials | 20 | 56,868,370 | Preserve all unique materials; do not treat this as disposable output. |

The point-cloud inventory includes the 3DEP clip products under
`point_cloud_data/download_3dep_lpc/`. Those are external reference inputs even
though they were generated as clips. The other LAS/LAZ files are reusable
derived products and are not raw CASALS input data.

## Existing scientific results

The successful classification run metadata and evaluation summaries are in
`CASALS_L1B/outputs/classify_and_evaluate_casals_refh/`. The recorded
`rule_combined_v1` baseline used `GROUND_SNR_MIN=5`, `GRID_RES_M=15`,
`MIN_POINTS_PER_CELL=1`, `GROUND_CELL_PERCENTILE=2`, `DTM_IDW_K=12`, and
`DTM_IDW_POWER=2`; the metadata JSON retains the full configuration.

| H5 stem | Evaluated points | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| `casals_l1b_20241112T165718_001_02` | 1,450,551 | 0.894911657708 | 0.519858130794 | 0.880365535787 |
| `casals_l1b_20241118T171757_001_02` | 2,653,617 | 0.919614247271 | 0.565101082036 | 0.912379963679 |

The source files are `CASALS_L1B/outputs/classify_and_evaluate_casals_refh/`
`casals_l1b_*_evaluation_summary.json` and `*_run_metadata.json`.

Other important local result sets include:

- `CASALS_L1B/outputs/debug_casals_refh_classifier_params_refined/` (best
  configuration, top configurations, run metadata, and failed-run records).
- `CASALS_L1B/outputs/diagnose_3dep_offsets_outputs/` and
  `outputs/diagnose_3dep_offsets/` (reference-frame configuration, matched
  samples, summaries, and reports). Some root-level run records refer to the
  previous `Research/CASALS` location and show missing-input failures; retain
  those records as historical evidence.
- `CASALS_L1B/outputs/sweep_peak_georef_rigorous_validation/`,
  `CASALS_L1B/beam_geolocation_rule_audit_outputs_revised/`, and the notebook
  output folders listed above.
- `CASALS_L1B/outputs/animate_pushbroom/` and
  `CASALS_L1B/outputs/animate_rx_waveforms/`, which contain generated videos
  and their parameter metadata.

All 20 tracked notebooks were parsed for their topic, code cells, paths, and
saved outputs. Both geolocation rule-audit notebook revisions and the separate
beam/range validation notebooks contain distinct saved results; all are being
retained pending a source-by-source comparison. No notebook has been merged or
stripped of output at baseline.

## Local environment baseline

The available interpreter is Python 3.14.6. It has NumPy, pandas, SciPy,
pyproj, rasterio, scikit-learn, Matplotlib, OpenCV, and pytest. It does not have
h5py, laspy, pyarrow, Open3D, PDAL, GDAL bindings, PyQt5, PyVista, PyVistaQt,
pyqtgraph, or nptdms. Therefore the scientific workflows and GUI cannot be
validated in this interpreter until their required packages are installed.

The GUI settings file at the repository root contains a machine-specific TDMS
path and window state. It is preserved locally and will be moved out of Git
tracking during the refactor.
