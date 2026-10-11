# Research Experiments and Notebooks

This index covers the ten active notebooks and distinguishes their bounded evidence from preserved historical work.

## Active notebooks

| Notebook | Research question |
| --- | --- |
| `notebooks/00_s3_data_inventory.ipynb` | What metadata are present under the public CASALS S3 prefix, with paginated listing and optional explicit-key read checks? |
| `notebooks/01_refh_quality.ipynb` | What do official refh quality fields, filtering, support-limited DSM products, and tentative DTM look like on real CASALS inputs? |
| `notebooks/02_refh_classification.ipynb` | What does the H5-only classifier predict, and how does it compare with explicitly aligned 3DEP pseudo-labels under distinct transfer-status populations? |
| `notebooks/03_waveform_analysis.ipynb` | What do real RX/TX waveforms, detected components, sweep/track structure, quality associations, and failure cases show? |
| `notebooks/04_peak_geolocation.ipynb` | How do production H1 and audited H2 map the same raw-argmax and detector bins, with the legacy Refh-anchored beam retained as a diagnostic? |
| `notebooks/05_forest_transect_showcase.ipynb` | What observed and support-limited Refh surfaces occur across a historically tree-covered AOI, with tentative DTM context? |
| `notebooks/06_waveform_gallery.ipynb` | How do selected real waveform morphologies differ, separately from their spatial land-cover evidence? |
| `notebooks/07_georeferencing_geometry.ipynb` | How do stored instrument/window/Refh positions and audited H1/H2 mappings relate in ECEF and local ENU? |
| `notebooks/08_waveform_decomposition.ipynb` | How do exploratory Gaussian component fits describe six selected real RX waveforms relative to existing detector markers and Refh context? |
| `notebooks/09_nadir_window_reflection.ipynb` | Does a bounded real-H5 sweep block contain a recurring early-bin signature consistent with the proposed nadir-window reflection hypothesis, and what does cautious screening or mitigation show? |

Each notebook names its local input and output paths. It uses package APIs for scientific processing and raises an actionable error if required data or prior workflow products are missing. The full collection is numbered 00 through 09. Clean-kernel execution and scientific review evidence are recorded in [notebook_scientific_review.md](notebook_scientific_review.md).

Notebooks 05 through 07 run independently from the repository root. Small numerical/input helpers are in `research/showcase.py`; scientific processing reuses the package and existing bin audit. PNG/CSV/JSON and self-contained Plotly HTML outputs are generated below each notebook's `outputs/notebooks/<notebook>/scientific_review/<granule-stem>/` directory. Execution, numerical closures, visual review, and preservation checks are recorded in [notebook_scientific_review.md](notebook_scientific_review.md).

Notebooks 08 and 09 also run from the repository root. They reuse selected real pulses and the road-context anchor from Notebook 06; Notebook 08 reads the fresh screening event from Notebook 09. Their research helpers are `research/waveform_decomposition.py` and `research/nadir_window_reflection.py`; outputs are isolated below each notebook's `scientific_review/<granule-stem>/` directory. Gaussian terms remain waveform candidates, and the null screen covers one block only. See [notebook_scientific_review.md](notebook_scientific_review.md). Neither notebook changes production detection or geolocation behavior.

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
- Current classification comparison: `notebooks/02_refh_classification.ipynb`.
- Historical offset and transfer notebooks remain in `research/archived_notebooks/refh/`.
- Empirical height alignment is not a datum transformation. The transfer status and confidence fields define the pseudo-label population and must be reported with any comparison.

## Preserved research records

The 21 former topic notebooks and four superseded synthetic root demos remain under `research/archived_notebooks/` with their saved outputs. The active collection contains ten notebooks numbered 00 through 09. Historical evidence is inventoried in [refactor_baseline.md](refactor_baseline.md); implementation history is recorded in [refactor_inventory.md](refactor_inventory.md).

Presentations, notes, PDFs, and source references remain under `docs/presentations/` and `docs/references/`. See [references/README.md](references/README.md).
