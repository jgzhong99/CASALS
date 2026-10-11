# Waveform Decomposition and Nadir-Window Reflection: Execution Report

## Scope and input

Both notebooks were executed against the real November 18, 2024 granule `data/raw/casals_l1b/casals_l1b_20241118T171757_001_02.h5` (3,604,480 pulses, 14,080 sweeps, 256 tracks, and 2,728 RX bins). The RX reads are bounded: Notebook 08 loads only six selected pulses, while Notebook 09 reads a 90-sweep block (sweeps 11994-12083 across 256 tracks, 23,040 pulses, or 0.639% of the granule). The selected block is centered near the previously reviewed road-context anchor at sweep 12039 / track 68. Pulse identities are mapped through the H5 sweep/track index grid, not inferred from flat record order.

Notebook 09 tests the briefing's proposed mechanism: a small fraction of pulses may reflect internally from the protective optical window in the nadir path, re-enter the receiver, and form a strong non-geophysical return. A stable early bin, abnormal amplitude, weak scene dependence, and track-localized recurrence are treated as a hypothesis to test, not as known truth. Refh is only a limited scene-dependence proxy.

## Notebook 08: waveform decomposition

The notebook retains and plots each raw RX waveform. Baselines/noise use H5 `bg_mean` and `bg_std`; the current Notebook 06 detector configuration and marker locations initialize/anchor the comparison. The exploratory fit is a sum of Gaussian terms selected by BIC over the available candidates. Detector smoothing is not substituted for the raw samples being fit. At Refh SNR below 5, the search is capped at one component; no detector candidates yields a no-detectable-peak result. A fit order is called stable only within the tested candidate set when its BIC separation passes the stated rule. Components remain waveform-derived candidates, not validated physical returns.

| Case | Pulse (sweep / track) | Context | SNR | Chosen K | Fit review |
| --- | --- | --- | ---: | ---: | --- |
| A | 2128614 (8314 / 55) | Probable tree canopy, multiple detector markers | 9.06 | 5 | Stable within tested set; at available candidate limit |
| B | 2085162 (8145 / 81) | Probable tree canopy, one dominant return | 8.49 | 3 | Stable within tested set; at available candidate limit |
| C | 2827710 (11045 / 245) | Probable open field | 16.67 | 2 | Stable within tested set; at available candidate limit |
| D | 3082120 (12039 / 68) | Probable road / mixed verge | 12.18 | 4 | Stable within tested set; at available candidate limit |
| E | 1280439 (5001 / 189) | Low-SNR ambiguous waveform | 1.35 | 0 | No detectable peak; no resolved component |
| F | 3077720 (12022 / 194) | Isolated early transient from Notebook 09; unconfirmed | 4.03 | 1 | Unstable order; low-SNR one-component cap |

All six fits completed without a fit failure. The chosen-K distribution is one case each at K=0 through K=5; one fitted order is unstable (F), one case has no resolved component (E), and five cases reach the available candidate limit. Median raw-count RMSE is 99.04 counts. The limit cases do not establish that the selected K is globally optimal; additional candidates were not evaluated. Case F corresponds to the strongest early transient found in the Notebook 09 block, but neither that event nor its fitted component is confirmed as a window reflection. The gallery is suitable for a waveform-morphology discussion with these caveats visible. It does not demonstrate a confirmed multi-return or window-reflection result.

Outputs are in `outputs/notebooks/08_waveform_decomposition/casals_l1b_20241118T171757_001_02/`, including the six-panel gallery, individual decomposition/residual plots, `waveform_decomposition_examples.csv`, `waveform_fit_metrics.csv`, case JSON, and summary JSON.

## Notebook 09: nadir-window reflection screen

For each track/bin in the selected block, the screening score combines fixed-bin occupancy above 5 H5 background standard deviations, median exceedance amplitude, an early-bin prior, and a weak-scene-association proxy. A band is retained only when occupancy is at least 20%, median exceedance is at least 5 sigma, and score is at least 0.45. The early gate covers bins 0-600. These are transparent screening thresholds, not calibrated sensor limits.

No track/bin band passed all gates. Maximum fixed-bin occupancy was 3.33%; 36 of 23,040 pulses had at least one early-bin sample above 8 sigma. The strongest isolated event is pulse 3077720, sweep 12022 / track 194, at zero-based bin 415 with 11.41 sigma excess; its raw-maximum and Refh-associated bins are both 415. Its H5 Refh is 183.09 m, versus a same-track block median of -33.17 m and P10-P90 of -33.35 to -33.05 m. This is an isolated local H5 outlier whose cause is unresolved; there is no independent scene truth here to attribute it to the optical window. No candidate band, affected track range, or range interval is supported by this sample. No range conversion is claimed for the isolated bin.

The three mitigation strategies were compared without inventing a correction:

| Strategy | Result in this block |
| --- | --- |
| A: flag only | No supported interval to tag; zero bins flagged and no downstream filtering applied. This remains the recommended policy after human review if a band is supported. |
| B: mask bins | No supported interval to mask; detector count stayed 6 to 6 and the low-SNR-capped Gaussian fit stayed K=1 to K=1. This no-op is not mitigation efficacy evidence. |
| C: template subtraction | Skipped because no recurring band supported a template. Template subtraction remains exploratory and is not a production correction. |

The range-by-track heatmap, track-by-sweep examples, same-sweep raw early-window comparison for tracks 194/68/55 (`representative_waveforms.png`), candidate CSV, early-transient review, and mitigation board are in `outputs/notebooks/09_nadir_window_reflection/casals_l1b_20241118T171757_001_02/`. The control tracks provide local comparison only, not scene truth. The candidate CSV contains its schema/header and no candidate rows. The board shows the three strategy statuses and the unchanged before/after comparison.

The selected 90-sweep block is only 0.639% of this granule, and only one date was screened. The null result cannot rule out a reflection elsewhere in the granule or on another date. The isolated event does not establish fixed-bin recurrence, weak scene dependence, or track-localized reflection behavior. Notebook 09 is suitable for presenting the screening method and honest null result, not as evidence that the optical-window signature has been identified.

## Execution and preservation

Both notebooks were executed in clean kernels in dependency order (09, then 08) using the local `map` Python environment and the repository's notebook execution runner. The updated figures were visually inspected, including the overview, range/track and track/sweep views, mitigation board, six-case gallery, fit-quality summary, and the individual case F figure. The track/sweep legend was given an opaque light background for legibility. Neither execution reported cell errors.

Validation completed:

- `python -m pytest -q -p no:cacheprovider`: 58 passed.
- `python -m compileall -q casals_l1b research tests`: passed.
- The source H5 and all production geolocation, classification, and detector behavior were left unchanged. New helpers are research-only in `research/waveform_decomposition.py` and `research/nadir_window_reflection.py`; tests are in `tests/test_waveform_decomposition.py` and `tests/test_nadir_window_reflection.py`.

The outputs are local research artifacts under the ignored `outputs/notebooks/` tree, not official L2 products. The material scientific limitation is that this real-data screen found no clear recurring ghost signature; Notebook 08 therefore includes the strongest unconfirmed transient rather than labeling it a reflection-affected case.
