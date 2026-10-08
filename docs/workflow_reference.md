# CASALS L1B Workflow Reference

## Scientific interpretation

CASALS L1B is treated as a geolocated waveform product, not a traditional discrete-return point cloud. Each pulse has one official geolocated `refh` point, defined by `refh_longitude`, `refh_latitude`, and `refh`; `refh` corresponds to the receiver waveform maximum-amplitude bin. Unless source metadata documents otherwise, `refh` is interpreted as WGS84 ellipsoidal height.

Secondary peaks found in a sampled waveform are diagnostic features. They are not official geolocated returns and do not create a formal multi-return point cloud. Horizontal projection is separate from vertical reference-frame interpretation. The scripts do not silently apply a vertical datum correction.

## Formal workflows

Run commands from the repository root after installing the package. Set `--h5`, `--reference`, or other input paths explicitly when required. Each script provides `--help`.

| Step | Entry point | Purpose |
| --- | --- | --- |
| 1 | `python -m scripts.export_refh_las --h5 <file.h5>` | Export an unclassified Level-A refh LAS. |
| 2 | `python -m scripts.filter_refh_points --h5 <file.h5>` | Write raw, noise-labeled, and clean refh LAS products. |
| 3 | `python -m scripts.extract_waveform_features --h5 <file.h5>` | Derive waveform component and pulse/sweep diagnostics. |
| 4 | `python -m scripts.make_refh_dsm --h5 <file.h5>` | Build a support-limited refh surface DSM with a raw strict companion. |
| 5 | `python -m scripts.extract_refh_ground --h5 <file.h5>` | Derive tentative ground candidates and an interpolated DTM. |
| 6 | `python -m scripts.download_3dep_lpc --h5 <file.h5>` | Query the USGS 3DEP LPC index and clip usable EPT resources. Requires PDAL. |
| 7 | `python -m scripts.diagnose_3dep_offsets --h5 <file.h5> --reference <clip.laz>` | Diagnose CASALS/3DEP vertical and reference-frame differences without modifying CASALS points. |
| 8 | `python -m scripts.transfer_3dep_labels_to_casals --casals-h5 <file.h5> --dep3-las <clip.laz>` | Transfer 3DEP labels for pseudo-reference analysis. |
| 9 | `python -m scripts.classify_and_evaluate_refh --h5 <file.h5> --reference <labels.laz>` | Run the configured deterministic refh classifier and evaluation. |

The normal outputs live under `outputs/<workflow>/`. An output LAS/LAZ is kept beside its workflow metadata and diagnostics rather than in a separate project-wide point-cloud folder. The 3DEP download workflow writes its manifests under `outputs/download_3dep_lpc/`; downloaded reference clips belong in `data/reference/3dep/`.

## Waveform feature extraction

The feature extractor reads large `rx_waveform` arrays in bounded slices. It does not assume every dataset has a complete rectangular sweep/track layout. Its outputs are component-level, pulse-level, and sweep-level summaries, with optional diagnostic figures.

Interpret the output conservatively:

- Candidate component count is not a true return count.
- Prominent secondary components are waveform features, not official returns.
- Range-window checks and tentative height bookkeeping do not geolocate additional returns.
- Waveform features can support refh quality diagnosis or downstream experiments, but they do not change the official refh definition.

Relevant notebooks are in `notebooks/introduction/` and `notebooks/waveform/`; the independent range-window/bin-mapping hypothesis is preserved at `notebooks/geolocation/range_window_bin_mapping_hypothesis.ipynb`.

## Refh products

- `export_refh_las.py` writes unclassified Level-A refh records; initial LAS classification is `1`.
- `filter_refh_points.py` assigns likely noise class `7` and records its noise reason codes. It preserves the original threshold and class semantics.
- `make_refh_dsm.py` writes a support-limited filled DSM and a strict observed-cell companion. Fill products are restricted to the configured support mask; they are not a classified ground DEM.
- `extract_refh_ground.py` writes a tentative derived ground-candidate product and DTM, not an official ground DEM.
- `classify_and_evaluate_refh.py` uses transferred 3DEP labels only as pseudo-reference evaluation labels. Its metrics are not independent accuracy estimates.

## 3DEP comparisons

The downloader writes EPT-derived clips, not archival copies of source USGS LAZ tiles. It preserves project metadata and does not transform vertical datum. Offset diagnosis keeps geodetic reference-frame checks in that workflow; they are not folded into general coordinate helpers. Pseudo-label transfer outputs remain analysis products.

## Historical research records

Notebook sources and saved outputs are preserved. Prior generated products were moved beneath `outputs/baseline_pre_refactor/` for comparison and are not post-refactor results. For the original HEAD, major configuration, metrics, and output inventory, see [refactor_baseline.md](refactor_baseline.md). The experiment index is [experiments.md](experiments.md).
