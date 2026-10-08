# Research experiments and notebooks

This index identifies the preserved research tracks. Experimental results and notebook outputs are evidence of prior runs, not automatic claims of independent accuracy.

## Waveform diagnostics

- `casals_l1b/waveform.py` provides shared waveform record indexing, preprocessing, candidate-component detection, peak features, quality diagnostics, continuity features, and range-window checks.
- `experiments/waveform/` is reserved for standalone waveform experiments; current waveform investigation remains in the notebooks below and in the formal feature extractor.
- Notebooks: `notebooks/waveform/` and `notebooks/introduction/`.
- Extra peaks are waveform-derived diagnostics, not official geolocated multi-returns.

## Beam and range geolocation

- `experiments/geolocation/geolocate_sweep_bins_from_beam_angle_rules.py` explores sweep/bin location rules.
- `notebooks/geolocation/` contains beam geometry theory, range/peak validation, rule audits, forward reconstruction, sweep-peak validation, and the separately preserved range-window/bin-mapping hypothesis notebook.
- Earlier geolocation result tables are retained under `outputs/baseline_pre_refactor/`.
- The two revised geolocation-rule-audit notebooks were kept separate because their code and findings differ materially; neither was merged into the other.

## 3DEP comparison and pseudo-label transfer

- Formal scripts: `scripts/download_3dep_lpc.py`, `scripts/diagnose_3dep_offsets.py`, and `scripts/transfer_3dep_labels_to_casals.py`.
- Research notebooks: `notebooks/refh/3dep_offset_diagnosis.ipynb` and `notebooks/refh/3dep_pseudolabel_transfer_qc.ipynb`.
- 3DEP clips are external comparison data. Horizontal CRS and vertical reference-frame differences must be considered explicitly; empirical offsets are not datum transformations.
- Transferred labels remain pseudo-reference labels and are not an independent accuracy certification.

## Refh filtering, surfaces, and classification

- Formal processing: `scripts/filter_refh_points.py`, `scripts/make_refh_dsm.py`, `scripts/extract_refh_ground.py`, and `scripts/classify_and_evaluate_refh.py`.
- Research parameter exploration: `experiments/classification/debug_classifier_params.py` and `experiments/classification/explore_3dep_like_rules.py`.
- Research notebooks and historical diagnostics: `notebooks/refh/`, `notebooks/waveform/`, plus preserved output metadata under `outputs/baseline_pre_refactor/`.
- The DSM is a refh surface; the extracted ground product is tentative. Neither is an official ground DEM.

## Preserved source materials

Presentations, a waveform/geolocation note, and related source references remain under `docs/presentations/` and `docs/references/`. See [references/README.md](references/README.md). The project-level archival directory remains local and untracked.
