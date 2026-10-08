# CASALS Research Workflows

This repository contains analysis tools and research records for CASALS Level-1B waveform data. The workflows read the official CASALS reference-height (`refh`) product, inspect waveform structure and geolocation, compare selected CASALS products with 3DEP lidar, and make clearly labeled derived surfaces and classifications.

CASALS L1B is treated here as a geolocated waveform product. Each pulse has one official geolocated `refh` point associated with the maximum-amplitude receiver-waveform bin. Additional waveform peaks are diagnostic features; they are not official geolocated returns. `refh` is treated as WGS84 ellipsoidal height unless source metadata says otherwise. Horizontal projection and vertical reference-frame interpretation are documented separately; an empirical height offset is not a vertical datum transformation.

## Repository layout

```text
casals_l1b/       Shared H5, coordinate, waveform, noise, raster, and classification code
scripts/          Formal processing entry points and standalone utilities
experiments/      Classification, geolocation, and waveform research code
notebooks/        Preserved notebooks grouped by topic
casals_gui_app/   Qt TDMS viewer implementation; casals_gui.py is its launcher
data/raw/         User-provided CASALS H5 and TDMS inputs (local, ignored by Git)
data/reference/   External reference inputs, including 3DEP clips (local, ignored)
data/derived/     Reusable derived data, if promoted from a workflow output
outputs/          Generated workflow results and preserved pre-refactor results
docs/             Workflow, data-layout, experiment, and source-reference notes
tests/            Small synthetic tests for shared numerical behavior
```

See [docs/data_layout.md](docs/data_layout.md), [docs/workflow_reference.md](docs/workflow_reference.md), and [docs/experiments.md](docs/experiments.md) for details.
The [refactor report](docs/refactor_report.md) records the migration, preserved baselines, and real-data validation results.

## Environment setup

Use a supported Python environment and install the project in editable mode:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

Install optional GUI and visualization dependencies only when those features are needed:

```powershell
python -m pip install -e ".[gui,visualization]"
```

The 3DEP clipping workflow also calls PDAL. Install PDAL and its Python bindings from a compatible conda-forge environment; they are not included in the pip extras. GDAL command-line and OSR cross-checks are optional and are reported as unavailable when not installed.

## Prepare local data

Place CASALS `.h5` files under `data/raw/casals_l1b/`, TDMS files and sidecars under `data/raw/tdms/`, and external 3DEP LAS/LAZ references under `data/reference/3dep/`. These directories are local data, are excluded from Git, and are not modified by package installation. The current workstation copy was moved from its previous project-local folders without rewriting file contents.

Workflow outputs default to `outputs/<workflow>/`. Existing pre-refactor outputs are retained under `outputs/baseline_pre_refactor/` for comparison. Do not use them as newly generated results; their metadata records their original paths and run configuration.

## Core workflow

The principal workflow uses a CASALS H5 input:

```powershell
python -m scripts.export_refh_las --h5 data/raw/casals_l1b/<granule>.h5
python -m scripts.filter_refh_points --h5 data/raw/casals_l1b/<granule>.h5
python -m scripts.extract_waveform_features --h5 data/raw/casals_l1b/<granule>.h5
python -m scripts.make_refh_dsm --h5 data/raw/casals_l1b/<granule>.h5
python -m scripts.extract_refh_ground --h5 data/raw/casals_l1b/<granule>.h5
```

Each command accepts `--help`. `--output-dir` selects the workflow output directory; the point-cloud workflows keep LAS/LAZ files inside that directory. JSON configuration overrides are supported by the export, filter, DSM, and ground commands while keeping input/output paths on the command line.

3DEP comparison is a separate research workflow. First download or provide a reference clip, then run `scripts/diagnose_3dep_offsets.py` and/or `scripts/transfer_3dep_labels_to_casals.py` with the H5 and reference LAS/LAZ paths. Transferred labels are pseudo-reference labels for evaluation; they do not certify independent classification accuracy. Run `scripts/classify_and_evaluate_refh.py` only with an explicit H5/reference pair.

## Notebooks

Notebooks retain their outputs as historical research records. Run from the repository root or from the notebook directory after installing the editable package. Update the input path variables in a notebook to point into `data/raw/` or `data/reference/`; do not treat embedded output paths or figures as a current rerun. Topic folders are `notebooks/introduction/`, `waveform/`, `geolocation/`, and `refh/`.

## GUI

The supported desktop viewer is the PyQt5 launcher:

```powershell
python casals_gui.py
```

The GUI reads local settings from `config/local/casals_gui_settings.json`. A sanitized example is provided at `config/casals_gui_settings.example.json`. Install the `gui` extra and the TDMS reader dependency for data browsing; the `visualization` extra enables optional 3D display components.

## Tests

Run the synthetic regression tests with:

```powershell
python -m pytest
```

These tests check shared helpers and deterministic scientific behavior on small arrays. They do not replace real-data workflow validation or independent scientific review.

## Research references

The preserved presentations and source materials are indexed in [docs/references/README.md](docs/references/README.md). The prior workflow notes remain in [docs/workflow_reference.md](docs/workflow_reference.md).
