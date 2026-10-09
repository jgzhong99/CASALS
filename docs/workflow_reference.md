# CASALS L1B Workflow Reference

## Scientific interpretation

CASALS L1B is a geolocated waveform product. Each pulse has one official `refh` point defined by `refh_longitude`, `refh_latitude`, and `refh`; `refh` is associated with the RX waveform maximum-amplitude bin. Secondary detected components are not official returns. The geolocation module can create experimental candidate coordinates for them, but these are not a validated multi-return point cloud.

Horizontal projection and vertical reference-frame interpretation are separate. `refh` is treated as WGS84 ellipsoidal height unless source metadata documents otherwise. Empirical CASALS-to-3DEP height alignment is not a geoid or datum transformation.

## Formal command groups

Run commands from the repository root. Outputs default beneath `outputs/<workflow>/<input-stem>/<step>/`; use `--output-dir` to choose another location.

| Group | Command | Purpose |
| --- | --- | --- |
| `refh` | `python -m casals_l1b refh export --h5 <file.h5>` | Export official unclassified Level-A refh LAS. |
| `refh` | `python -m casals_l1b refh filter --h5 <file.h5>` | Write raw, noise-labeled, and clean refh products. |
| `refh` | `python -m casals_l1b refh dsm --h5 <file.h5>` | Build a support-limited refh surface DSM and strict observed-cell companion. |
| `refh` | `python -m casals_l1b refh ground --h5 <file.h5>` | Derive tentative ground candidates and an interpolated DTM. |
| `refh` | `python -m casals_l1b refh classify --h5 <file.h5>` | Predict classes using CASALS H5 and configured classifier rules. |
| `refh` | `python -m casals_l1b refh evaluate --prediction <pred.laz> --reference <ref.laz>` | Evaluate an existing prediction against an explicitly aligned reference. |
| `peaks` | `python -m casals_l1b peaks extract --h5 <file.h5>` | Extract bounded-slice waveform component and pulse/sweep diagnostics. |
| `peaks` | `python -m casals_l1b peaks locate --h5 <file.h5> --components <components.parquet> --method both` | Map existing components to experimental segment and/or beam candidates. |
| `peaks` | `python -m casals_l1b peaks validate --h5 <file.h5> --candidates <candidates.parquet>` | Check candidate schema and H5 pulse identity. |
| `reference` | `python -m casals_l1b reference download --h5 <file.h5>` | Query the USGS 3DEP index and clip usable EPT resources; requires PDAL. |
| `reference` | `python -m casals_l1b reference diagnose --h5 <file.h5> --reference <clip.laz>` | Diagnose CASALS/3DEP frame and height differences. |
| `reference` | `python -m casals_l1b reference transfer --casals-h5 <file.h5> --dep3-las <clip.laz>` | Transfer labels for pseudo-reference analysis. |

`python -m casals_l1b --help` lists only these three top-level groups. Each nested command supports `--help`. Standalone viewer, animation, and parameter-exploration tools remain under `tools/` and `research/` and are not registered in the main CLI.

## Refh products

- `refh export` writes each valid official max-RX-bin `refh` point as LAS class 1.
- `refh filter` preserves raw points, records likely noise as class 7 with a `noise_reason` code, and writes a clean product under the configured thresholds.
- `refh dsm` writes an interpolated surface only within its support mask and retains the strict observed-cell DSM for audit. It is not a ground DEM.
- `refh ground` uses a distinct tentative ground-candidate algorithm and writes a derived DTM. It is not an official or independently validated terrain product.
- `refh classify` does not require a reference file. It writes class predictions, reason codes, point indices, counts, and run metadata. If no reference is supplied, evaluation metadata reports `not_run` and no accuracy metrics are generated.
- `refh evaluate` is separate from prediction. In the optional combined classify path, the reference is passed explicitly after prediction. 3DEP-transferred labels are pseudo-reference labels; the reported metrics do not certify independent accuracy.

## Waveform extraction and geolocation

`peaks extract` uses the package's existing detector with bounded H5 reads. The component table retains the existing detector schema and indices. Candidate component count is not a return count, and feature extraction does not geolocate additional returns.

`peaks locate` reads the component table and H5 pulse records. It checks `pulse_index` against `sweep_num` and `track_num`, then writes `candidate_points.parquet`, `validation_summary.csv`, `closure_residuals.csv`, and `run_metadata.json`.

- The segment method interpolates WGS84 ECEF `rwstart` to `rwstop` by `peak_bin / (n_rx_bins - 1)`. Raw RX argmax closure is an internal consistency check and does not fit model parameters.
- The beam method uses the documented local-angle convention and an official `refh` anchor with H5 `bin_size`. Its candidates and closure are refh-anchored by construction.
- Coordinate validity records that finite modeled coordinates were produced. It does not establish that a secondary component is a physical return or that its absolute location is correct.
- Segment/beam agreement is internal model agreement. Neither method is an independent reference for the other.

The measured real-data residuals, model comparison, and unresolved assumptions are in [the improvement report](refactor_improvement_report.md).

## Current notebooks and historical records

The current runnable notebooks are:

1. `notebooks/01_refh_quality.ipynb`
2. `notebooks/02_refh_classification.ipynb`
3. `notebooks/03_waveform_analysis.ipynb`
4. `notebooks/04_peak_geolocation.ipynb`

They use explicit local data paths and package APIs. Missing inputs produce an actionable error rather than a synthetic fallback. The 21 earlier notebooks and four superseded synthetic root demos, with their saved outputs, remain under `research/archived_notebooks/`.

## 3DEP comparisons

Downloaded clips are EPT-derived products, not archival copies of source USGS tiles. The label-transfer workflow defaults to `verified_crs` and requires confirmed complete source/target frames plus an available audited PROJ operation. Explicit `--alignment-mode empirical_diagnostic` permits exploratory ground dz alignment, marked `empirical_alignment`; it is not a datum transformation. Far/ambiguous points remain unclassified. Evaluation checks original H5 and coordinate identity, reports strict and strict+weak pseudo-reference agreement with coverage, and excludes unsupported points. Missing DTM alone leaves classifier points unclassified. See [3dep_reference.md](3dep_reference.md) for evidence, commands, limitations and smoke results.

## Research records

The earlier project history and baseline inventory remain in [refactor_baseline.md](refactor_baseline.md). The active implementation and verification evidence are in [refactor_improvement_report.md](refactor_improvement_report.md).
