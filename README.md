# CASALS Research Workflows

This repository contains two CASALS L1B research lines: refh quality, surfaces, classification, and 3DEP comparison; and waveform multi-peak detection with experimental 3D candidate geolocation.

Each pulse has one official geolocated `refh` point associated with the maximum-amplitude RX bin. Other detected waveform components are diagnostic features. Candidate positions for those components are experimental and are not official multi-return coordinates. CASALS `refh` height is treated as WGS84 ellipsoidal height unless source metadata says otherwise. An empirical 3DEP height offset is not a vertical datum transformation.

## Repository layout

```text
casals_l1b/       Shared H5, refh, waveform, classification, and geolocation code
research/         Reference comparison and parameter/geolocation research
tools/            Standalone viewers, animations, and data utilities
notebooks/        Four current real-data research notebooks
casals_gui_app/   Qt TDMS viewer implementation; casals_gui.py is its launcher
data/             Local raw and reference data (ignored by Git)
outputs/          Local workflow results (ignored by Git)
docs/             Workflow, data, experiment, and refactor notes
tests/            Small numerical and interface regression tests
```

See [data layout](docs/data_layout.md), [workflow reference](docs/workflow_reference.md), and [research index](docs/experiments.md). The third-round implementation and validation record is [refactor_improvement_report.md](docs/refactor_improvement_report.md).

## Environment setup

Install the project in a supported Python environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

Install optional GUI and visualization dependencies only when needed:

```powershell
python -m pip install -e ".[gui,visualization]"
```

The 3DEP clipping workflow also requires PDAL and its Python bindings. Install those from a compatible conda-forge environment.

## Core CLI

Run commands from the repository root. The formal CLI exposes three groups: `refh`, `peaks`, and `reference`.

```powershell
python -m casals_l1b refh export --h5 data/raw/casals_l1b/<granule>.h5
python -m casals_l1b refh filter --h5 data/raw/casals_l1b/<granule>.h5
python -m casals_l1b refh dsm --h5 data/raw/casals_l1b/<granule>.h5
python -m casals_l1b refh ground --h5 data/raw/casals_l1b/<granule>.h5
python -m casals_l1b refh classify --h5 data/raw/casals_l1b/<granule>.h5
python -m casals_l1b refh evaluate --prediction <classified.laz> --reference <labels.laz>

python -m casals_l1b peaks extract --h5 data/raw/casals_l1b/<granule>.h5
python -m casals_l1b peaks locate --h5 <granule.h5> --components <component_table.parquet> --method both
python -m casals_l1b peaks validate --h5 <granule.h5> --candidates <candidate_points.parquet>

python -m casals_l1b reference download --h5 data/raw/casals_l1b/<granule>.h5
python -m casals_l1b reference diagnose --h5 <granule.h5> --reference <3dep_clip.laz>
python -m casals_l1b reference transfer --casals-h5 <granule.h5> --dep3-las <3dep_clip.laz>
```

Classification runs with H5 and classifier parameters alone. Evaluation is a separate step; `refh classify --reference` is available when an explicitly paired pseudo-reference evaluation is desired. Transferred 3DEP labels remain pseudo-reference labels and do not establish independent accuracy. `python -m casals_l1b --help` and each nested `--help` show the current arguments. The installed `casals` command uses the same groups.

The principal APIs can also be called directly: `export_refh`, `filter_refh`, `make_refh_dsm`, `make_refh_ground`, `classify_refh`, `evaluate_classification`, `extract_components`, `locate_peaks`, and `validate_candidate_table`.

## Current notebooks

The four notebooks are [refh quality and products](notebooks/01_refh_quality.ipynb), [classification and 3DEP pseudo-reference](notebooks/02_refh_classification.ipynb), [waveform analysis](notebooks/03_waveform_analysis.ipynb), and [peak geolocation](notebooks/04_peak_geolocation.ipynb). They use explicit local inputs, call package functions, and include saved real-data outputs. Start Jupyter from the repository root. The 21 earlier research notebooks remain under `research/archived_notebooks/`; four superseded synthetic root demos were also moved there with saved outputs, leaving exactly four active notebooks under `notebooks/`.

## Viewers

- `tools/view_lpc_qt_classes.py` opens generic classified LAS/LAZ files, supports class visibility toggles, and limits display through chunked reservoir sampling.
- `tools/view_refh_points.py` reads CASALS H5 directly to display refh quality fields and optional classification overlays.

The Qt workflow viewer is `python casals_gui.py`. Local GUI settings live in `config/local/` and are not tracked.

## Tests

```powershell
python -m pytest -q -p no:cacheprovider
```

The tests check shared helpers and interfaces on small fixtures. Real-data execution and scientific interpretation are documented separately in the refactor improvement report.
