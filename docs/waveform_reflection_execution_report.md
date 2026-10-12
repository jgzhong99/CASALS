# Waveform Decomposition and Nadir-Window Hypothesis Review

**Execution date:** 2026-10-11
**Inputs:** November 18 and November 12, 2024 CASALS L1B H5 granules
**Outputs:** `outputs/notebooks/08_waveform_decomposition/method_improvement/` and `outputs/notebooks/09_nadir_window_reflection/method_improvement/`

These are research-only waveform morphology and anomaly-screening studies. They do not change the production detector, geolocation mapping, stored RX arrays, or official `Refh`. The proposed optical-window reflection is one hypothesis among several; it is not treated as a documented CASALS signature or as a confirmed cause.

## Notebook 08: Gaussian morphology fits

The notebook reads six selected real TX/RX records from the November 18 granule, verifies each pulse through the H5 sweep/track grid, and fits the raw RX counts. The model has a fitted baseline and nonnegative Gaussian components. K=0 is included as the baseline-only candidate. Every candidate order within a comparison uses the same finite samples; full-window and signal-supported windows are compared separately. RMSE for both fits is evaluated on the same 2,728-bin waveform. BIC uses `n·ln(RSS/n) + p·ln(n)` with `p = 2 + 3K`.

| Case | Pulse (sweep / track) | SNR | Full-window K | Supported-window K | Full-waveform RMSE (counts) |
| --- | --- | ---: | ---: | ---: | ---: |
| A | 2,128,614 (8314 / 55) | 9.06 | 5 | 5 | 86.32 |
| B | 2,085,162 (8145 / 81) | 8.49 | 5 | 1 | 101.98 |
| C | 2,827,710 (11045 / 245) | 16.67 | 5 | 1 | 80.02 |
| D | 3,082,120 (12039 / 68) | 12.18 | 5 | 1 | 97.48 |
| E | 1,280,439 (5001 / 189) | 1.35 | 1 | 1 | 106.86 |
| F | 3,077,720 (12022 / 194) | 4.03 | 1 | 1 | 99.66 |

All six fits completed; no optimizer failures occurred. Median full-waveform RMSE was 98.57 counts. For the high-SNR cases A–D, the selected order followed the configured Kmax in each 3/5/7 sensitivity run. Under the default Kmax=5, A–D selected K=5; the low-SNR K=1 cap applied to E and F. Every selected fit also touched at least one parameter boundary. B, C, and D changed from K=5 in the full window to K=1 in the signal-supported window. The BIC margin flag means only that the best fit separated from other orders inside the tested candidate set. Together, cap hits, boundary hits, and window sensitivity leave the real-data model order unresolved. Gaussian components are not physical-return counts.

The fit diagnostics retain failed candidates, optimizer status, parameter-boundary distances, component overlap, candidate/seed/search caps, full and supported sample counts, residual tails/skew/kurtosis/adjacent correlation, and common-set RMSE. Four deterministic synthetic controls exercise noise-only, single-component, separated-two-component, and overlapping-component behavior. These controls check numerical behavior only.

The H5 stores 80 TX counts and 2,728 RX counts per record. It contains no separate TX/RX sample-time coordinates, sample intervals, shared time origin, relative-delay calibration, waveform scaling, or channel-specific receiver response. The reported TX/RX positions are zero-based array indices only. `delta_time` is per-pulse metadata; `rwstart`/`rwstop` are not mapped to waveform bins here. Root nanosecond-to-distance constants do not supply the missing timing contract. TX is plotted separately, and TX-informed fitting and physical-range conversion remain unavailable.

## Notebook 09: fixed-bin baseline and all-bin Stage A survey

### Preserved November 18 baseline

The original selection remains sweeps 11994–12083, stop-exclusive: 90 sweeps × 256 tracks = 23,040 pulses, or 0.639% of the granule. The preserved early gate contains 601 samples (zero-based bins 0–600). Each exact track/bin occupancy uses its own finite-sweep denominator. With the original heuristic gates (20% occupancy, 5σ median excess, score 0.45), no candidate band passes. Maximum fixed-bin occupancy is 3/90 = 3.33%; 36 pulses have an early sample above 8σ. The predeclared 10%, 20%, and 30% occupancy sensitivity checks also return zero candidate bands.

Pulse 3,077,720 remains the strongest isolated baseline event: sweep 12022, track 194, bin 415, 11.408σ; its raw maximum and `Refh`-associated bin are both 415. Stored H5 `Refh` is 183.09 m, compared with a same-track block median of −33.17 m and P10–P90 of −33.35 to −33.05 m. This is a large local H5 outlier, but its cause is unconfirmed. Stored `rwstart`, `rwstop`, bounce-time offsets, beam angles, and `delta_time` are reported without assigning unsupported units or converting bin position to time or range.

### Supplemental 180-sweep context sensitivity

The selected November 18 block was expanded symmetrically by 45 sweeps on each side to `[11949, 12129)` (180 sweeps; 46,080 pulses across 256 tracks). This is a supplemental sensitivity context. The original `[11994, 12084)` 90-sweep fixed-bin screen and event selection remain unchanged.

Using the same candidate gates (occupancy at least 20%, median excess at least 5 sigma, and ghost score at least 0.45), no candidate band passed in the 180-sweep block. Maximum fixed-bin occupancy was 4/180 (2.22%; track 17, bin 566). The selected screen-event bin (track 194, bin 415) exceeded the threshold in 1/180 valid sweeps (0.56%). These recurrence summaries describe this bounded block only; they do not identify the transient's physical source or establish a reflection signature.

Figure 09-B2 is one large heatmap with a single plotting area: three vertically stacked track blocks, each covering the same 180 sweeps. Rows show signed `RX - H5 bg_mean` counts over early RX bins; the blocks cover the unconfirmed screen-event track and two local comparison tracks. Dashed bounds show the original 90-sweep screen, while the selected sweep and event bin are marked for reference. The figure is `track_sweep_context_180.png`. Supporting outputs are `extended_180_sweep_track_bin_scores.csv`, `extended_180_sweep_candidate_bands.csv`, and `extended_180_sweep_summary.json`.

### Prespecified 16-sweep follow-up blocks

The sensitive Stage A scan checks every RX bin for peaks above 8σ with at least 4σ prominence. Three starts (1408, 5632, 9856) were applied to each November granule before inspecting results. The existing 90-sweep baseline was scanned with the same event rule. Every follow-up block had 4,096 pulses and a 100% finite waveform-sample fraction.

| Granule | Sweeps (start inclusive, stop exclusive) | Stage A peak events | Records with event | Exact track/bin rows recurring in ≥20% of sweeps |
| --- | --- | ---: | ---: | ---: |
| Nov 12 | 1408–1423 | 205 | 202 | 4 |
| Nov 12 | 5632–5647 | 1,210 | 1,199 | 102 |
| Nov 12 | 9856–9871 | 1,176 | 1,166 | 90 |
| Nov 18 baseline | 11994–12083 | 20,296 | 20,162 | 0 |
| Nov 18 | 1408–1423 | 3,803 | 3,466 | 293 |
| Nov 18 | 5632–5647 | 2,650 | 2,403 | 107 |
| Nov 18 | 9856–9871 | 2,566 | 2,305 | 118 |

These counts use a sensitive any-bin peak screen and are not reflection detections. Exact-bin recurrence alone does not establish shared physical delay because the H5 does not map RX indices to TX-relative time and may have scene, receiver, or channel effects. The baseline fixed-bin null remains unchanged.

### Competing hypotheses and limits

The notebook keeps three alternatives explicit: H1 direct optical-window reflection, H2 receiver transient/recovery/ringing, and H3 delayed internal ghost or cross-talk. The [NASA CASALS instrument paper (NTRS 20230015620)](https://ntrs.nasa.gov/citations/20230015620) supports the aircraft, APD-array, multiplexing, digitizer, and FPGA-timing architecture context. It does not document a nadir-window reflection signature. The H5 metadata inspected here does not identify such a signature either.

Seven deterministic synthetic scenarios cover noise-only, moving surface, fixed-bin event, fixed-delay drift, intermittent event, surface overlap, and broad/ringing-like morphology. Recall, localization, masking, retained energy, and argmax changes apply only to those synthetic records. Undefined synthetic metrics are encoded as JSON `null`; no real-data sensitivity or false-positive rate is inferred.

No real band was supported, so mitigation remained a true no-op: zero bins were flagged, raw RX was preserved, the selected event's detector count remained 6 before/after, and template subtraction was skipped. No RX mask, subtraction, or production correction was applied.

## Execution, validation, and preservation

- Notebooks 08 and 09 executed in clean kernels after Notebook 06 produced its case table and Notebook 09 wrote its pulse handoff. All code cells completed with no stored error outputs.
- Full suite: `python -m pytest -q -p no:cacheprovider` — **73 passed**.
- Compilation: `python -m compileall -q casals_l1b research tests` — **passed**.
- Notebook validation: all ten notebooks `00`–`09` have valid structure, executed code-cell counts, and no error outputs.
- New artifacts are isolated below the two `method_improvement/<granule-stem>/` paths named above. Earlier outputs and source H5 data were not overwritten.
- The helpers are confined to `research/waveform_decomposition.py` and `research/nadir_window_reflection.py`; the two focused test files are `tests/test_waveform_decomposition.py` and `tests/test_nadir_window_reflection.py`. Production waveform detection, classification, and geolocation modules were unchanged.

The complete notebook-by-notebook review and preservation audit are in [notebook_scientific_review.md](notebook_scientific_review.md). These outputs are exploratory artifacts, not official L2 products or independent accuracy certification.


Final visual pass: Notebook 08 was re-executed after adding vertical separation between the TX, RX and residual panels. Notebook 09 was re-executed after removing numeric ticks from the conceptual timing diagram and separating labels in the optical-path schematic. The corrected PNGs were opened at full size; the timing offsets remain expressly qualitative and uncalibrated. The final preservation audit matched all 550 pre-edit output hashes (5,652,750,027 bytes), all four protected H5/3DEP LAZ hashes, and all 13 protected metadata records; see `.compile_tmp/notebook_audit/final_preservation_hash_audit.json`.


Residual-model check: across the six 2,728-bin real fits, H5-derived noise sigma spans 80.74?107.33 counts. Full-fit residual RMSE is 0.964?1.234 times that sigma; background-only residual RMSE is 0.912?0.987 times sigma. Lag-1 residual autocorrelation is 0.913?0.951 for every case. The pulse-3,077,720 fit has residual skewness 2.13 and excess kurtosis 10.78. The residuals therefore show structured dependence and a heavy-tailed case; RSS-based BIC is a bounded working ranking, not a validated independent-Gaussian likelihood or a physical component-count test.
