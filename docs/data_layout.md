# Data and output layout

Local inputs and generated results are excluded from Git. They remain on the workstation and are not bundled into the source repository.

| Path | Contents | Origin and handling |
| --- | --- | --- |
| `data/raw/casals_l1b/` | Original CASALS L1B H5 granules | User-provided source data. Preserve as read-only inputs. |
| `data/raw/tdms/` | TDMS recordings, index sidecars, viewer assets, and notes | User-provided acquisition data and related assets. Preserve as inputs/reference material. |
| `data/reference/3dep/` | Downloaded 3DEP LAS/LAZ clips | External reference data. Preserve their original files and source metadata. |
| `data/derived/` | Reusable derived data promoted from an output run | Keep only products intentionally reused as inputs; record their source workflow. |
| `outputs/<workflow>/` | New LAS/LAZ, tables, figures, rasters, videos, and run metadata | Regenerable outputs; each workflow keeps its products together. |
| `outputs/baseline_pre_refactor/` | Original output folders retained for comparison | Historical evidence. Do not mix these files with post-refactor runs. |
| `config/local/` | User-specific GUI settings | Local configuration; ignored by Git. |

The baseline inventory recorded two H5 files (28.1 GB total), TDMS assets (37.9 GB), two external 3DEP clips (644 MB), and the prior generated output tree (about 9.2 GB plus point-cloud products). The exact file counts and pre-refactor classification metrics are in [refactor_baseline.md](refactor_baseline.md). Data relocation used file moves on the same volume; file contents were not rewritten.

`outputs/` and local data are not committed. A historical output can be removed only after the corresponding workflow has been rerun and the required comparison has been recorded. Unique notebook results and research evidence must remain available even when ordinary derived outputs are regenerated.
