# CASALS Research Workflows

This repository contains two CASALS L1B research lines: refh quality, surfaces, classification, and 3DEP comparison; and waveform multi-peak detection with experimental 3D candidate geolocation.

Each pulse has one official geolocated `refh` point associated with the maximum-amplitude RX bin. Other detected waveform components are diagnostic features. Candidate positions for those components are experimental and are not official multi-return coordinates. CASALS `refh` height is treated as WGS84 ellipsoidal height unless source metadata says otherwise. An empirical 3DEP height offset is not a vertical datum transformation.

## Repository layout

```text
casals_l1b/       Shared H5, refh, waveform, classification, and geolocation code
research/         Reference comparison and parameter/geolocation research
tools/            Standalone viewers, animations, and data utilities
notebooks/        Ten current real-data research notebooks
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

Notebook 00 uses paginated public S3 listing with no per-object `HeadObject` calls by default. Notebook 02 evaluates only strict transfer status 1 as its primary pseudo-reference population and reports status 1+2 separately. Notebook 04 compares production H1 with audited H2 on the same raw-argmax and detected-component bins while retaining the legacy Refh-anchored beam diagnostic. The full review and measured execution status are in [notebook scientific review](docs/notebook_scientific_review.md).

The active collection contains ten notebooks numbered 00 through 09. The first five are [S3 inventory](notebooks/00_s3_data_inventory.ipynb), [refh quality and products](notebooks/01_refh_quality.ipynb), [classification and 3DEP pseudo-reference](notebooks/02_refh_classification.ipynb), [waveform analysis](notebooks/03_waveform_analysis.ipynb), and [peak geolocation](notebooks/04_peak_geolocation.ipynb). They use explicit inputs and package functions. Start Jupyter from the repository root. The 21 earlier research notebooks and four superseded synthetic root demos remain under `research/archived_notebooks/` with their saved outputs.

Three independently runnable research showcases are [forest transect](notebooks/05_forest_transect_showcase.ipynb), [waveform gallery](notebooks/06_waveform_gallery.ipynb), and [georeferencing geometry](notebooks/07_georeferencing_geometry.ipynb). They use real L1B H5, existing products located through workflow metadata, dated NAIP crops, and the audited research H1/H2 mapping. Review-generated artifacts are written below each notebook's `outputs/notebooks/<notebook>/scientific_review/` directory; pre-existing products used as inputs remain read-only. Missing surfaces are generated through formal APIs without duplicating full point clouds. Historical imagery supports probable canopy labels, not exact 2024 footprint truth. These are exploratory demonstrations, not official L2 products. See [the executed showcase report](docs/showcase_execution_report.md).

The [waveform decomposition notebook](notebooks/08_waveform_decomposition.ipynb) compares exploratory Gaussian fits against existing detector markers on six real pulses. The [nadir-window reflection notebook](notebooks/09_nadir_window_reflection.ipynb) screens one bounded real-H5 block for recurring early-bin energy and documents an unconfirmed isolated transient. Helpers are research-only in `research/waveform_decomposition.py` and `research/nadir_window_reflection.py`; outputs are under each notebook's `scientific_review/<granule-stem>/` folder. Notebook 08 reports full-window and detector-supported fit-window sensitivity; Notebook 09 uses per-bin finite-sweep denominators and preserves the no-candidate/no-op result. Neither analysis changes production detection or confirms physical returns or a window reflection. See [the waveform and reflection execution report](docs/waveform_reflection_execution_report.md).

## Viewers

- `tools/view_lpc_qt_classes.py` opens generic classified LAS/LAZ files, supports class visibility toggles, and limits display through chunked reservoir sampling.
- `tools/view_refh_points.py` reads CASALS H5 directly to display refh quality fields and optional classification overlays.

The Qt workflow viewer is `python casals_gui.py`. Local GUI settings live in `config/local/` and are not tracked.

## Tests

```powershell
python -m pytest -q -p no:cacheprovider
```

The tests check shared helpers and interfaces on small fixtures. Real-data execution and scientific interpretation are documented separately in the refactor improvement report.
