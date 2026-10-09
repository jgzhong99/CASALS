# CASALS L1B Workflow Reference

## Scientific interpretation

CASALS L1B is treated as a geolocated waveform product, not a traditional discrete-return point cloud. Each pulse has one official geolocated `refh` point, defined by `refh_longitude`, `refh_latitude`, and `refh`; `refh` corresponds to the receiver waveform maximum-amplitude bin. Unless source metadata documents otherwise, `refh` is interpreted as WGS84 ellipsoidal height.

Secondary peaks found in a sampled waveform are diagnostic features. They are not official geolocated returns and do not create a formal multi-return point cloud. Horizontal projection is separate from vertical reference-frame interpretation. The scripts do not silently apply a vertical datum correction.

## Formal workflows

Run commands from the repository root after installing the package. Set `--h5`, `--reference`, or other input paths explicitly when required. Each script provides `--help`.

| Step | Entry point | Purpose |
| --- | --- | --- |
| 1 | `python -m casals_l1b refh-export --h5 <file.h5>` | Export an unclassified Level-A refh LAS. |
| 2 | `python -m casals_l1b refh-filter --h5 <file.h5>` | Write raw, noise-labeled, and clean refh LAS products. |
| 3 | `python -m casals_l1b peaks --h5 <file.h5>` | Derive waveform component and pulse/sweep diagnostics. |
| 4 | `python -m casals_l1b refh-dsm --h5 <file.h5>` | Build a support-limited refh surface DSM with a raw strict companion. |
| 5 | `python -m casals_l1b refh-ground --h5 <file.h5>` | Derive tentative ground candidates and an interpolated DTM. |
| 6 | `python -m casals_l1b download-3dep --h5 <file.h5>` | Query the USGS 3DEP LPC index and clip usable EPT resources. Requires PDAL. |
| 7 | `python -m casals_l1b diagnose-3dep --h5 <file.h5> --reference <clip.laz>` | Diagnose CASALS/3DEP vertical and reference-frame differences without modifying CASALS points. |
| 8 | `python -m casals_l1b transfer-3dep --casals-h5 <file.h5> --dep3-las <clip.laz>` | Transfer 3DEP labels for pseudo-reference analysis. |
| 9 | `python -m casals_l1b classify-refh --h5 <file.h5> --reference <labels.laz>` | Run the configured deterministic refh classifier and evaluation. |

Per-input outputs live under `outputs/{refh,classification,peaks,reference}/<input-stem>/<step>/`. LAS/LAZ products stay beside their workflow metadata and diagnostics. The 3DEP downloader writes manifests under `outputs/reference/<h5-stem>/download/`; downloaded reference clips belong in `data/reference/3dep/`. Research figures and reports use the same domain roots where an input file identifies the result; multi-input aggregates may use a shared explicitly named directory.

## Waveform feature extraction

The feature extractor reads large `rx_waveform` arrays in bounded slices. It does not assume every dataset has a complete rectangular sweep/track layout. Its outputs are component-level, pulse-level, and sweep-level summaries, with optional diagnostic figures.

Interpret the output conservatively:

- Candidate component count is not a true return count.
- Prominent secondary components are waveform features, not official returns.
- Range-window checks and tentative height bookkeeping do not geolocate additional returns.
- Waveform features can support refh quality diagnosis or downstream experiments, but they do not change the official refh definition.

The current examples are `notebooks/01_refh_data_contract.ipynb` and `notebooks/02_waveform_components.ipynb`. Earlier introduction and waveform notebooks, including the range-window/bin-mapping hypothesis, are preserved under `research/archived_notebooks/`.

## Refh products

- `refh-export` writes unclassified Level-A refh records; initial LAS classification is `1`.
- `refh-filter` assigns likely noise class `7` and records its noise reason codes. It preserves the original threshold and class semantics.
- `refh-dsm` writes a support-limited filled DSM and a strict observed-cell companion. Fill products are restricted to the configured support mask; they are not a classified ground DEM.
- `refh-ground` writes a tentative derived ground-candidate product and DTM, not an official ground DEM.
- `classify-refh` uses transferred 3DEP labels only as pseudo-reference evaluation labels. Its metrics are not independent accuracy estimates.

## 3DEP comparisons

The downloader writes EPT-derived clips, not archival copies of source USGS LAZ tiles. It preserves project metadata and does not transform vertical datum. Offset diagnosis keeps geodetic reference-frame checks in that workflow; they are not folded into general coordinate helpers. Pseudo-label transfer outputs remain analysis products.

## Historical research records

Notebook sources and saved outputs are preserved. Prior generated products were moved beneath `outputs/baseline_pre_refactor/` for comparison and are not post-refactor results. For the original HEAD, major configuration, metrics, and output inventory, see [refactor_baseline.md](refactor_baseline.md). The experiment index is [experiments.md](experiments.md).
