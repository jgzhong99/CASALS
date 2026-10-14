# CASALS Research Workflows

This repository contains CASALS L0 TDMS acquisition audits and L1B research on refh quality, surfaces, classification, 3DEP comparison, waveforms, and experimental 3D candidate geolocation.

Each pulse has one official geolocated `refh` point associated with the maximum-amplitude RX bin. Other detected waveform components are diagnostic features. Candidate positions for those components are experimental and are not official multi-return coordinates. CASALS `refh` height is treated as WGS84 ellipsoidal height unless source metadata says otherwise. An empirical 3DEP height offset is not a vertical datum transformation.

## Repository layout

```text
casals_l1b/       Shared H5, refh, waveform, classification, and geolocation code
research/         Reference comparison and parameter/geolocation research
tools/            Standalone viewers, animations, and data utilities
notebooks/        Thirteen active research notebooks, organized by topic
casals_tdms_viewer/ Qt TDMS viewer implementation; casals_gui.py is its launcher
data/             Local raw and reference data (ignored by Git)
outputs/          Local workflow results (ignored by Git)
backup/           Archived results from earlier code (ignored by Git)
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

The 13 active notebooks are organized by research topic. [notebooks/README.md](notebooks/README.md) is the navigation guide and [docs/experiments.md](docs/experiments.md) is the formal research catalog. Start Jupyter from the repository root. The L0 TDMS audit reads one explicitly selected file, distinguishes wavelength evidence from channel indices, samples waveform rows without loading full channels, and checks local L1B metadata references without reading H5 payloads. Only complete, unit-bearing wavelength mappings are plotted; lookup tables and formulas also need explicit channel-order and applicability evidence. Install `nptdms` with the `notebooks` optional dependency; the Qt viewer is not required.

Existing notebook output directories remain intact as historical results and inputs. New notebook runs write to `outputs/notebooks/<topic>/<notebook>/<dataset>/`. The 21 earlier research notebooks and four superseded synthetic root demos remain under `research/archived_notebooks/` with their saved outputs.

The active classification notebook keeps strict status 1 separate from status 1+2 pseudo-reference coverage. Peak geolocation retains production H1 and audited H2 comparisons without changing production mapping. The full historical review evidence is in [notebook scientific review](docs/notebook_scientific_review.md) and the [L1B validation report](docs/casals_l1b_validation_report.md).

The forest transect and waveform gallery remain exploratory displays based on existing products, selected real waveforms, dated NAIP crops, and audited H1/H2 mapping. Historical imagery supports probable canopy labels, not exact footprint truth. Waveform decomposition and nadir-window reflection report bounded diagnostics and unresolved cases; they do not change production detection or establish physical returns. See the [showcase report](docs/showcase_execution_report.md) and [waveform/reflection report](docs/waveform_reflection_execution_report.md).

## Viewers

- `tools/view_lpc_qt_classes.py` opens generic classified LAS/LAZ files, supports class visibility toggles, and limits display through chunked reservoir sampling.
- `tools/view_refh_points.py` reads CASALS H5 directly to display refh quality fields and optional classification overlays.

The Qt workflow viewer is `python casals_gui.py`. Local GUI settings live in `config/local/` and are not tracked.

## Tests

```powershell
python -m pytest -q -p no:cacheprovider
```

The tests check shared helpers and interfaces on small fixtures. Real-data execution and scientific interpretation are documented separately in the refactor improvement report.
