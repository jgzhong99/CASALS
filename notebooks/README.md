# CASALS Research Notebooks

The active collection is grouped first by product level and then by research topic. `L0/` contains raw TDMS acquisition analysis, `L1B/` contains analyses based on geolocated H5 granules, and `catalog/` contains the public S3 inventory that spans product levels. Start Jupyter from the CASALS repository root so `data/`, `outputs/`, and the project modules resolve consistently. Install notebook dependencies with:

```powershell
python -m pip install -e ".[dev,notebooks]"
```

New generated results retain the established `outputs/notebooks/<topic>/<notebook>/<dataset>/` paths. Existing numbered result folders are preserved at their original paths when a notebook reads them as historical evidence. Moving notebooks by product level does not move or copy result folders.

## Cross-product catalog

| Notebook | Purpose | Main inputs | Generated output |
| --- | --- | --- | --- |
| [S3 inventory](catalog/s3_inventory.ipynb) | Inventory public CASALS S3 object metadata with paginated listing and optional explicit-key checks. | Public S3 listing; optional object keys. | Inline tables; no local artifact by default. |

## Level 0

| Notebook | Purpose | Main inputs | Generated output |
| --- | --- | --- | --- |
| [TDMS audit](L0/data/tdms_audit.ipynb) | Audit file/group/channel metadata, Step/Sweep layout, wavelength evidence, sampling assumptions, selected raw bytes, and L1B `/dac` associations. | One explicitly selected `.tdms` file and the two local L1B H5 metadata/index/navigation fields; no L1B waveform arrays. | `outputs/notebooks/data/tdms_audit/<tdms-stem>/runs/<run-id>/` with raw-integrity, DAC, inventory, and figure artifacts. |

The TDMS notebook lists candidate files and requires `TDMS_PATH` when more than one file exists. It excludes `.tdms_index` sidecars and never substitutes synthetic data. Configure `TDMS_PATH`, optional `OUTPUT_ROOT`, optional `ROW_INDICES`, and `N_EXAMPLE_CHANNELS` in its settings cell. `nptdms` is included in the `notebooks` extra; no GUI packages are required. Wavelength evidence is reported as A/B/C/D; only complete A/B/C mappings with explicit units may be plotted in physical units. Lookup tables and formulas also require explicit channel order and applicability. Grade D scan indices do not produce wavelength values. Selected raw storage chunks are compared to channel slices, a raw-only reader, and direct bytes; they are called acquisition rows only where file metadata and channel lengths support that interpretation. The notebook summarizes `/dac` against each L1B file's own sweep, track, time, beam, and position arrays, without converting it to wavelength or linking mismatched TDMS sources.

## Level 1B

### Refh

| Notebook | Purpose | Main inputs | Generated output |
| --- | --- | --- | --- |
| [Quality and surfaces](L1B/refh/quality_and_surfaces.ipynb) | Review official Refh quality, filtering, support-limited DSMs, and tentative DTMs. | Two explicit L1B H5 granules and existing read-only products. | `outputs/notebooks/refh/quality_and_surfaces/<granule-stem>/`. |
| [Classification](L1B/refh/classification.ipynb) | Compare H5-only class predictions with aligned 3DEP pseudo-reference labels. | L1B H5, preserved transfer products, and 3DEP source metadata. | `outputs/notebooks/refh/classification/<granule-stem>/`. |
| [Forest transect](L1B/refh/forest_transect.ipynb) | Display observed/support-limited Refh surfaces and selected profiles across historical canopy context. | L1B H5, existing surface products, and dated NAIP imagery. | `outputs/notebooks/refh/forest_transect/<granule-stem>/`. |

### Waveforms

| Notebook | Purpose | Main inputs | Generated output |
| --- | --- | --- | --- |
| [Waveform analysis](L1B/waveforms/waveform_analysis.ipynb) | Inspect real TX/RX waveforms, detector components, sweep/track structure, and failure cases. | Explicit L1B H5. | `outputs/notebooks/waveforms/waveform_analysis/<granule-stem>/`. |
| [Waveform gallery](L1B/waveforms/waveform_gallery.ipynb) | Compare selected real waveform shapes with separate spatial land-cover evidence. | L1B H5, historical detector features, and dated NAIP assets. | `outputs/notebooks/waveforms/waveform_gallery/<granule-stem>/`. |
| [Waveform decomposition](L1B/waveforms/waveform_decomposition.ipynb) | Compare bounded Gaussian fit diagnostics with existing detector markers and Refh context. | L1B H5 and preserved selected waveform records. | `outputs/notebooks/waveforms/waveform_decomposition/<granule-stem>/`. |
| [Nadir-window reflection](L1B/waveforms/nadir_window_reflection.ipynb) | Screen selected real-H5 sweep blocks for recurring early-bin anomalies. | L1B H5 and historical waveform-gallery metadata. | `outputs/notebooks/waveforms/nadir_window_reflection/<granule-stem>/`. |

### Geolocation

| Notebook | Purpose | Main inputs | Generated output |
| --- | --- | --- | --- |
| [Peak geolocation](L1B/geolocation/peak_geolocation.ipynb) | Compare production H1 and audited H2 mappings for the same raw-argmax and detector bins. | Explicit L1B H5 and component records. | `outputs/notebooks/geolocation/peak_geolocation/<granule-stem>/`. |
| [Beam geometry](L1B/geolocation/beam_geometry.ipynb) | Relate stored instrument/window/Refh positions and H1/H2 mappings in ECEF and ENU. | L1B H5, selected waveform cases, and existing surfaces. | `outputs/notebooks/geolocation/beam_geometry/<granule-stem>/`. |
| [Elevation reference audit](L1B/geolocation/elevation_reference_audit.ipynb) | Review strict H5/3DEP frame and height-processing gates and remaining blockers. | H5/LAZ pairs and preserved historical validation artifacts. | Reads existing `outputs/notebooks/10_3dep_elevation_validation/` results. |
| [Vertical datum alignment](L1B/geolocation/vertical_datum_alignment.ipynb) | Meeting-informed same-pulse EGM2008 / GEOID18 comparison, tide sensitivities, and separate-site empirical alignment. | Both current H5/LAZ pairs; fixed historical ROI splits; audited PROJ grid. | New timestamped runs under `outputs/notebooks/geolocation/vertical_datum_alignment/runs/`. |

### Applications

| Notebook | Purpose | Main inputs | Generated output |
| --- | --- | --- | --- |
| [Land-cover response](L1B/applications/landcover_response.ipynb) | Summarize H5 signal and waveform features across reviewed field, road, canopy, and water ROIs. | Two L1B H5 granules, ROI manifest, historical NAIP assets, and preserved membership inputs. | `outputs/notebooks/applications/landcover_response/multi_granule_full_roi/`. |

See the [formal experiment catalog](../docs/experiments.md) for research questions, methods, and validation status. Historical execution reports retain their original notebook labels and output paths for traceability.
