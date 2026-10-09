# Research experiments and notebooks

This index identifies the preserved research tracks. Experimental results and notebook outputs are evidence of prior runs, not automatic claims of independent accuracy.

## Waveform diagnostics

- `casals_l1b/waveform.py` provides shared waveform record indexing, preprocessing, candidate-component detection, peak features, quality diagnostics, continuity features, and range-window checks.
- The formal feature command is `python -m casals_l1b peaks --h5 <file.h5>`.
- Historical waveform notebooks and saved results: `research/archived_notebooks/waveform/`.
- Current runnable notebook: `notebooks/02_waveform_components.ipynb`.
- Extra peaks are waveform-derived diagnostics, not official geolocated multi-returns.

## Beam and range geolocation

- `research/geolocation/geolocate_sweep_bins_from_beam_angle_rules.py` explores sweep/bin location rules.
- `research/archived_notebooks/geolocation/` preserves beam geometry theory, range/peak validation, rule audits, forward reconstruction, sweep-peak validation, and the separately preserved range-window/bin-mapping hypothesis notebook.
- Current runnable notebook: `notebooks/03_beam_geolocation_candidates.ipynb`.
- Earlier geolocation result tables are retained under `outputs/baseline_pre_refactor/`.
- The two revised geolocation-rule-audit notebooks were kept separate because their code and findings differ materially; neither was merged into the other.

## 3DEP comparison and pseudo-label transfer

- Commands: `download-3dep`, `diagnose-3dep`, and `transfer-3dep` through `python -m casals_l1b`.
- Research notebooks: `research/archived_notebooks/refh/3dep_offset_diagnosis.ipynb` and `research/archived_notebooks/refh/3dep_pseudolabel_transfer_qc.ipynb`.
- Current classification notebook: `notebooks/04_refh_classification_and_reference.ipynb`.
- 3DEP clips are external comparison data. Horizontal CRS and vertical reference-frame differences must be considered explicitly; empirical offsets are not datum transformations.
- Transferred labels remain pseudo-reference labels and are not an independent accuracy certification.

## Refh filtering, surfaces, and classification

- Formal processing commands: `refh-filter`, `refh-dsm`, `refh-ground`, and `classify-refh` through `python -m casals_l1b`.
- Research parameter exploration: `research/classification/debug_classifier_params.py` and `research/classification/explore_3dep_like_rules.py`.
- Historical notebooks and diagnostics: `research/archived_notebooks/refh/` and `research/archived_notebooks/waveform/`, plus preserved output metadata under `outputs/baseline_pre_refactor/`.
- The DSM is a refh surface; the extracted ground product is tentative. Neither is an official ground DEM.

## Preserved source materials

The four current root notebooks are `01_refh_data_contract.ipynb`, `02_waveform_components.ipynb`, `03_beam_geolocation_candidates.ipynb`, and `04_refh_classification_and_reference.ipynb`. They use package APIs and small or synthetic demonstrations; the first notebook reports local H5 schema only unless full loading is explicitly enabled.

Presentations, a waveform/geolocation note, and related source references remain under `docs/presentations/` and `docs/references/`. See [references/README.md](references/README.md). The project-level archival directory remains local and untracked.
