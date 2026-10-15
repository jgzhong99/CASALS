# CASALS TDMS Wavelength and Metadata Audit

Audit date: 2026-10-14  
Scope: every `.tdms` file under `data/raw/tdms/`, both local L1B H5 files, the available viewer configuration, and relevant local design material. Raw TDMS and H5 inputs were read only. The notebook was run separately on both TDMS files; the batch command independently inventoried both.

## Follow-up evidence (2026-10-14)

A second pass checked storage-chunk boundaries, selected raw values, `.tdms_index` metadata prefixes, and quantitative `/dac` associations. The fresh batch artifacts are in `outputs/notebooks/data/tdms_audit/batch_20261014_raw_integrity_final_01/`. The selected-file notebook completed without cell errors for the 13:34 TDMS in `outputs/notebooks/data/tdms_audit/casals_18-11-2024_13-34-53_158-003/runs/20261014_final_audit_01/` and for the 13:37 TDMS in `outputs/notebooks/data/tdms_audit/casals_18-11-2024_13-37-23_164-002/runs/20261014_partial_tail_review_01/`. The earlier directories listed below remain preserved.

### Raw-byte boundaries and selected value checks

Both files have one declared segment whose next-segment boundary equals file size, and both `.tdms_index` files match the main file's metadata prefix byte-for-byte except for the `TDSm` versus `TDSh` tag. `incomplete_final_segment=False` describes the segment boundary; it does not imply that the raw payload ends on a whole channel chunk.

| TDMS file | Raw data bytes | Storage chunk accounting | Bounded checks | Row interpretation |
| --- | ---: | --- | --- | --- |
| `casals_18-11-2024_13-34-53_158-003.tdms` | 20,300,431,360 | 14,080 complete chunks x 2,816 int16 samples/channel | Chunks 0, 7,040, and 14,079; all 256 channels; 2,162,688 values compared across indexed slices, raw-only `nptdms`, and direct bytes. Every checked value matched. | `Sweeps/File=14,080`, 2,816 samples/channel, and raw-byte layout agree structurally. This supports 14,080 candidate row intervals, but does not certify every sample or acquisition event. |
| `casals_18-11-2024_13-37-23_164-002.tdms` | 17,582,522,368 | 12,194 complete chunks x 2,816 samples/channel, followed by a 1,310,720-byte tail = 2,560 samples/channel across 256 channels | Chunks 0, 1, 6,097, 12,191, 12,192, 12,193, plus the partial tail; all 256 channels; 4,980,736 values compared across indexed slices, raw-only `nptdms`, and direct bytes. Every checked value matched. | Channel-name structure is complete, but 34,340,864 samples/channel is not divisible by `Sweeps/File=14,080`. Candidate acquisition rows and `valid_prefix_rows` remain unresolved; 2,816 is only the observed TDMS chunk size. |

For the 13:37 file, `nptdms` reports 256 channel-status records with `expected_length=2,816` and `read_length=2,560`, a 256-value final-chunk shortfall per channel (65,536 summed). The selected raw tail is consistent with the recorded channel values and direct raw bytes. The raw area is structurally bounded and sampled consistently; the available files do not show whether acquisition ended normally or was interrupted, and this check does not rule out acquisition-level dropouts. No complete raw-file byte-by-byte checksum comparison was performed.

### Quantitative L1B `/dac` association

The notebook read the complete `/dac` arrays and the associated `/track_num`, `/sweep_num`, `/delta_time`, beam-angle, and instrument-position arrays from both local L1B H5 files. It did not load the full H5 files or TX/RX waveform datasets. `/dac` has 3,604,480 `float64` entries in each file, but only a dimension-list HDF5 attribute; it has no wavelength unit or calibration field. Values exactly round-trip through `float32`, which describes storage precision only.

Each H5 has track IDs 0-255 with 14,080 samples per track, sweep IDs 0-14,079 with 256 samples per sweep, exactly one record per track/sweep pair, and strictly increasing `delta_time`. The first track IDs in file order are `0, 8, 16, ... 120`. These validate the H5's internal index grid; they do not pair it with either local TDMS.

| L1B H5 | `/dac` range; unique values | Pearson r: track / sweep / time | r: beam azimuth / elevation | r: instrument longitude / latitude | Sweep-mean trend |
| --- | --- | --- | --- | --- | --- |
| `casals_l1b_20241112T165718_001_02.h5` | -0.1286101 to -0.1260585; 168,475 | -0.277 / 0.516 / 0.516 | 0.557 / 0.893 | 0.528 / 0.516 | Slope 6.44e-8 per sweep; R2 0.288; 53.1% of adjacent sweep-mean steps increase. |
| `casals_l1b_20241118T171757_001_02.h5` | -0.0435643 to -0.0373387; 1,401,176 | 0.008 / 0.985 / 0.985 | 0.789 / -0.822 | -0.985 / -0.987 | Slope 4.16e-7 per sweep; R2 0.971; 99.9% of adjacent sweep-mean steps increase. |

The Nov 12 within-track standard deviation has median 0.0004874, comparable to its 0.0004851 range of track means. For Nov 18 the median within-track standard deviation is 0.0017139, much larger than the 0.0000477 range of track means. These observations show that `/dac` varies across sweep/time, with different descriptive patterns between the two files; they do not identify DAC units, wavelength, or causal control behavior. Beam-angle and position correlations are also descriptive and use HDF5 values with no unit attributes.

Neither H5's root `tdms_file` basename matches either local TDMS file. No correlation was computed between local TDMS values and L1B arrays. Both local TDMS files remain wavelength Grade D; neither L1B `/dac` array changes that evidence. No physical wavelength map is available or recoverable from the inspected TDMS metadata or these uncalibrated H5 DAC values.

## Findings

Neither local TDMS supports a physical wavelength mapping. Both receive grade **D: scan indices only**. The 13:34 file supports bounded waveform samples and a complete 32-step by 8-sweep channel layout. The 13:37 file has the same complete Step/Sweep name set, but its channel read-status records and channel lengths do not support a safe row reshape. Neither L1B H5 filename reference matches a local TDMS file, and local L1A H5 files are **unavailable / not inspected**.

No `wavelength_nm` mapping was emitted. A channel count of 256, scan index, campaign design wavelength, nominal tuning range, or an unannotated DAC field does not establish per-channel wavelength values.

## TDMS files, segments, and channel layout

The audit found exactly two `.tdms` files; `.tdms_index` files were treated as optional metadata indexes, not as separate data files. `nptdms` 1.12.1 read metadata from both indexes. It reported one segment in each file and `incomplete_final_segment=False` for both. The inventory reads the segment count from `nptdms._reader._segments`, and records that the library presents merged final properties. Since each file has one segment, the properties below describe its only observed segment.

| TDMS file | File size | Root `name` property | Groups / channels | Channel dtype and length | Step/Sweep set | Row layout |
| --- | ---: | --- | --- | --- | --- | --- |
| `casals_18-11-2024_13-34-53_158-003.tdms` | 20,300,447,744 bytes | `casals_18-11-2024_13-34-53_158` | 1 / 256 | `int16`, 39,649,280 values per channel | Complete: all 256 `Sweep0Step0` … `Sweep7Step31` combinations; no missing, duplicate, extra, or unrecognized names. File order is sweep-major then step. | `39,649,280 = 14,080 × 2,816`; `Sweeps/File` gives a divisible row reshape. |
| `casals_18-11-2024_13-37-23_164-002.tdms` | 17,582,538,752 bytes | `casals_18-11-2024_13-37-23_164` | 1 / 256 | `int16`, 34,340,864 values per channel | Complete name set; no missing, duplicate, extra, or unrecognized names. File order is sweep-major then step. | `34,340,864` is not divisible by declared `Sweeps/File=14,080`; `samples_per_row` is unresolved and waveform row sampling was skipped. |

Both files have the same 12 integer-valued `CASALS` group properties and no channel-level properties:

| Property | Observed value in both files |
| --- | ---: |
| `TxWinWidth(tick)` | 10 |
| `RxWinWidth(tick)` | 341 |
| `# WVLSteps/Sweep` | 32 |
| `# Sweeps/ShiftCycle` | 8 |
| `SOAtoStartPulseDelay(tick)` | 80 |
| `SOAtoMZMDelay(tick)` | 14 |
| `RxADCDelay(tick)` | 41 |
| `SweepToFirstSOADelay(tick)` | 26 |
| `ErrorBits(U16)` | 0 |
| `SOADelay(tick)` | 0 |
| `Sweeps/File` | 14,080 |
| `Sweeps/Frame` | 100 |

`Sweeps/Frame=100` does not divide 14,080. Its frame-boundary meaning is unresolved, so the audit does not treat this as a channel-layout failure. The indexed channel set and the file/viewer ordering are separate facts: the viewer candidate index `step × 8 + sweep` agrees with the viewer's step-major software reshape, but it is not a verified physical footprint order or an L1B `track_num` mapping.

### 13:37 TDMS channel read status

The second file has 256 `ChannelSegmentStatus` entries. Every entry reports `expected_length=2,816` and `read_length=2,560`, a difference of 256 values per channel (65,536 values summed over 256 channels). `nptdms` also warned that the 17,582,522,368-byte data block is not a multiple of its 1,441,792-byte chunk size and that it would attempt the last chunk. Its file-level `incomplete_final_segment` flag nevertheless remains `False`. The audit preserves both observations and does not infer which physical row or acquisition event the shortfall represents.

The 13:34 file has no `channel_statuses` collection from `nptdms`; that is recorded as unavailable, not as a proof that every raw value is valid. Its file-level `incomplete_final_segment` flag is `False`.

## Wavelength evidence rules and result

The audit searches property names, string values, JSON-serialized small numeric arrays, and local configuration values. It emits a `wavelength_nm` value only when every expected Step/Sweep channel is covered by one complete, unit-bearing source with explicit ordering or formula semantics:

| Grade | Required evidence |
| --- | --- |
| A | One consistent, unit-bearing physical wavelength property for every channel. |
| B | A complete unit-bearing lookup table for all channels and an explicit table order that defines Step/Sweep indexing. |
| C | Unit-bearing start, step increment, and sweep offset; an explicit supported Step/Sweep formula; and explicit applicability across each ShiftCycle. |
| D | Scan indices only, incomplete/unordered candidates, absent wavelength evidence, or conflicting mappings. No physical values are emitted. |

Both TDMS inventories contain only scalar file/group properties and empty channel property sets. The sole wavelength-related property candidate is `# WVLSteps/Sweep=32`; it counts scan steps and has no wavelength unit. There is no per-channel wavelength field, complete calibration table, or parameterized wavelength formula. Therefore both files are grade D and the mapping output is empty.

The local `data/raw/tdms/CASALS_viewer_parameter.txt` describes itself as a viewer parameter file with default fields set to zero. It includes channel-selection indices 0 and 255, not physical wavelengths. The viewer `.ini` and `.aliases` files are LabVIEW VI Server configuration. The available viewer ZIP contains an executable, aliases, and an `.ini`; it has no source code or wavelength lookup table. None of those files supplies a calibrated map.

The local proposal draft `Archive/000 proposal/CASALS 20241216Rev.pdf` describes nominal campaign design values, including 1040 nm and 256 tuned wavelengths. These describe instrument/campaign context, not an individual calibration record for either local TDMS file. NASA's 2024 CASALS paper likewise reports 256 contiguous airborne ground spots and a 1040 nm system with a 1.152 MHz tuning rate; its design summary does not provide a per-file Step/Sweep wavelength table. A separate NASA design paper discusses a 1040 nm-centered, 30 nm tuning design and 256 airborne footprints. Neither range was linearly interpolated into channel values.

## Bounded waveform evidence

The original row-oriented view of the 13:34 file read rows 0, 7,040, and 14,079 using channel slices across all 256 channels. That sample contained 2,162,688 `int16` values (4,325,376 bytes). Every sampled value was finite; the sampled minimum and maximum were −32,730 and 32,735 raw counts. The follow-up pass separately checked the selected storage chunks listed above. Finite values do not establish a valid return because no invalid-sample code is documented in the inspected metadata. A bounded viewer/software parity check passed after the viewer's configured cut; it verifies software reshaping only.

No row-index waveform slices were read from the 13:37 file because its row/sample shape is unresolved and every channel has a segment-status shortfall. The later bounded storage-chunk and tail checks are summarized above; no full-file waveform scan was performed.

The executed notebook saved the complete property inventory, layout, grade-D wavelength evidence, per-channel sample statistics, and these figures under:

`outputs/notebooks/data/tdms_audit/casals_18-11-2024_13-34-53_158-003/runs/20261014_wavelength_audit_05/`

The channel-layout, raw waveform example, percentile-zoom example, and 3-row overview figures are `channel_layout.png`, `waveform_examples.png`, `waveform_examples_zoom.png`, and `waveform_overview.png` in that directory. The first three are available as software/index evidence; none establishes physical footprint identity.

## L1B and L1A cross-level evidence

Both local L1B H5 files were opened read-only with `h5py`. The recursive inventory recorded root/group/dataset attributes, object names, shapes, and dtypes without reading dataset payloads. Each H5 contains 69 HDF5 objects (the root and 68 datasets) and 97 attributes.

| Local L1B H5 | Root `tdms_file` reference | Exact local TDMS match | Root `l1a_file` reference | Exact local L1A match |
| --- | --- | --- | --- | --- |
| `casals_l1b_20241112T165718_001_02.h5` | `casals_12-11-2024_11-58-21_22.tdms` | No | `casals_l1a_20241112T165718_001_01.h5` | No |
| `casals_l1b_20241118T171757_001_02.h5` | `casals_18-11-2024_12-18-37_136.tdms` | No | `casals_l1a_20241118T171757_001_02.h5` | No |

The local TDMS basenames are the 13:34 and 13:37 files above. The 18 November L1B reference therefore does not match either local TDMS basename, and the November 12 reference has no local TDMS counterpart. No local L1A H5 was found under `data/raw/`; status: **unavailable / not inspected**.

The recursively reviewed H5 metadata includes root counts (`n_sweeps=14,080`, `n_tracks=256`) and datasets such as `/track_num` and `/sweep_num` (3,604,480 `int64` entries each), `/tracks` (256 entries), and `/sweeps` (14,080 entries). `/dac` is a 3,604,480-entry `float64` dataset with only an HDF5 dimension-list reference attribute; it has no wavelength unit or calibration annotation. It remains an unclassified candidate field. Matching counts do not repair the absent source-file correspondence or establish wavelength/track identity.

The full H5 metadata record is in `cross_level_evidence.json` beside the notebook outputs. It contains every root and dataset attribute, dataset shape/dtype, metadata candidate, and exact-reference result for both files.

## External sources

The [NI TDMS file-format reference](https://www.ni.com/en/support/documentation/supplemental/07/tdms-file-format-internal-structure.html) describes the file/group/channel metadata hierarchy and TDMS segments. It is used here to distinguish segment metadata from raw waveform blocks, not to infer CASALS calibration.

NASA's [2024 CASALS IGARSS paper](https://ntrs.nasa.gov/citations/20240001267) describes the 256-spot airborne design, 1040 nm laser, tuning rate, and detector architecture. NASA's [2023 CASALS airborne-demonstration paper](https://ntrs.nasa.gov/citations/20230015620) describes design tuning range and airborne/space footprint counts. These are design sources; neither provides a per-flight calibration for the two audited files.

The NI format page is dated 2024-11-12; NASA NTRS record 20240001267 was acquired 2024-01-29, and record 20230015620 was acquired 2023-10-30. All three sources were accessed 2026-10-14. The design statements above are external context; all file counts, property values, filename comparisons, and sample statistics in this report come from the local artifacts listed here.

## Reproduction and validation

Batch-audit both local TDMS files into a new output directory:

```powershell
python -m research.tdms_audit --all --data-dir data/raw/tdms --output-dir outputs/notebooks/data/tdms_audit/<new-batch-dir>
```

Run the selected-file notebook into a new directory; keep both the input file and source notebook unchanged:

```powershell
$env:CASALS_TDMS_PATH = 'data/raw/tdms/casals_18-11-2024_13-34-53_158-003.tdms'
$env:CASALS_TDMS_OUTPUT_ROOT = 'outputs/notebooks/data/tdms_audit/casals_18-11-2024_13-34-53_158-003/runs/<new-run-dir>'
python -m jupyter nbconvert --to notebook --execute notebooks/data/tdms_audit.ipynb --output tdms_audit.executed.ipynb --output-dir $env:CASALS_TDMS_OUTPUT_ROOT --ExecutePreprocessor.cwd=. --ExecutePreprocessor.timeout=600
```

The earlier metadata-only batch remains preserved in `outputs/notebooks/data/tdms_audit/batch_20261014_channel_status/`. The current raw-integrity batch and both selected-file notebook runs are listed in the follow-up evidence above. Final validation passed: `python -m pytest -q -p no:cacheprovider` reported 236 passed; `python -m compileall -q casals_l1b research tools tests` passed; all 13 active notebooks passed nbformat validation, IPython-aware code-cell compilation, and import resolution; current notebook links in the README and experiment catalog resolve.
