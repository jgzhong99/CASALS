# Data and output layout

Local inputs and generated results are excluded from Git. They remain on the workstation and are not bundled into the source repository.

| Path | Contents | Origin and handling |
| --- | --- | --- |
| `data/raw/casals_l1b/` | Original CASALS L1B H5 granules | User-provided source data. Preserve as read-only inputs. |
| `data/raw/tdms/` | TDMS recordings, index sidecars, viewer assets, and notes | User-provided acquisition data and related assets. Preserve as inputs/reference material. |
| `data/reference/3dep/` | Downloaded 3DEP LAS/LAZ clips | External reference data. Preserve their original files and source metadata. |
| `data/derived/` | Reusable derived data promoted from an output run | Keep only products intentionally reused as inputs; record their source workflow. |
| `outputs/refh/<h5-stem>/` | Official refh export/filter LAS, DSM, and tentative ground products | Regenerable workflow outputs, separated by input granule and task. |
| `outputs/peaks/<h5-stem>/` | Waveform component, pulse, sweep, and diagnostic outputs | Regenerable waveform-derived diagnostics; extra peaks are not official geolocated returns. |
| `outputs/classification/<h5-stem>/` | Per-granule classifications, comparisons, and run metadata | Regenerable results; 3DEP labels remain pseudo-reference. Aggregate tables stay at the classification root. |
| `outputs/reference/<h5-stem>/` | 3DEP comparison and label-transfer outputs | Regenerable comparison products with source/reference metadata. |
| `outputs/baseline_pre_refactor/` | Original output folders retained for comparison | Historical evidence. Do not mix these files with post-refactor runs. |
| `config/local/` | User-specific GUI settings | Local configuration; ignored by Git. |

The baseline inventory recorded two H5 files (28.1 GB total), TDMS assets (37.9 GB), two external 3DEP clips (644 MB), and the prior generated output tree (about 9.2 GB plus point-cloud products). The exact file counts and pre-refactor classification metrics are in [refactor_baseline.md](refactor_baseline.md). Data relocation used file moves on the same volume; file contents were not rewritten.

`outputs/` and local data are not committed. A historical output can be removed only after the corresponding workflow has been rerun and the required comparison has been recorded. The 21 former topic-folder notebooks, including their saved outputs and figures, are preserved byte-for-byte under `research/archived_notebooks/`. Four current notebooks live directly under `notebooks/`.
