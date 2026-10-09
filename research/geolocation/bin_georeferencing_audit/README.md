# CASALS L1B bin georeferencing audit

This experiment reads the original H5 files independently of production
`casals_l1b/geolocation.py`. It tests window geometry, raw RX maxima, competing
index conventions, stored timing, beam directions, and real non-reference bins.
It never applies geoid, atmosphere, tide, or range bias corrections again.

## Run

The tested existing environment is `C:/Users/zhong267/AppData/Local/miniconda3/envs/map/python.exe`.
The default/base Python on this computer lacks h5py. Activate `map` first:

```powershell
conda activate map
python research/geolocation/bin_georeferencing_audit/run_audit.py --mode smoke --output-root outputs/research/bin_georeferencing_audit/smoke_new
python research/geolocation/bin_georeferencing_audit/run_audit.py --mode full --output-root outputs/research/bin_georeferencing_audit/full_new
python -m research.geolocation.bin_georeferencing_audit.complete_report --output-root outputs/research/bin_georeferencing_audit/full_new
python -m pytest -q
python -m compileall -q research/geolocation/bin_georeferencing_audit tests
```

Use a fresh output root: existing granule directories cause an error rather than
overwrite historical results. `--h5 path1.h5 path2.h5` selects explicit originals.
`--chunk-size` defaults to 14,080 records, matching the record dimension of the
actual RX storage chunks (14,080 × 11). Memory is bounded by that batch and a few
NumPy arrays, not the entire waveform. Input SHA256 is checked before and after.
Both actual waveform shapes are discovered, including their record axis; RX bin
coordinates must be zero-based contiguous. Missing essential endpoints fail.
Missing optional fields remain NaN and are counted. Explicit fill values are
masked before scale/offset; HDF5's default storage fill zero is not treated as
missing. Units and correction application are not guessed from variable names.

Smoke selects sweep IDs 0, 5000, 7040, 14079 after reading real sweep numbers;
it does not assume a sweep × track reshape. Statistics cover those selected
pulses only. Full visits every pulse, writes zstd Parquet diagnostics in batches,
then reads individual scalar columns for exact quantiles and NMAD. No waveform
is persisted. Sweep/track/block/elevation/SNR summaries retain counts and P95.

## Predeclared analysis choices

- H1: `t=b/(N-1)`; H2: `t=(b+0.5)/N`; H3: reversed H1.
- H4 empirical: `tau(b)=tau_start+(b+0.5)*5e-10` in stored offset units.
  The increment was identified in the first November 12 block, then frozen.
  It is **not** an official seconds or raw round-trip timing contract.
- H5: Refh-anchored, historical radian/north-clockwise/horizon elevation,
  negated ENU at Refh, with window length/N spacing. This choice explicitly
  follows the H2 edge interpretation; additional N−1 spacing comparison is
  described in findings. Its Refh closure is zero by definition.
- H6: UNAVAILABLE as a physical range model because bin_size units/semantics
  are undocumented. Numerical height-step comparisons are still reported.
- High-quality subset: valid geometry and raw RX; SNR≥10; unique maximum;
  maximum exceeds second largest stored amplitude by >1 ADU. Empty subsets
  remain empty; the threshold is never optimized against closure.
- Near maxima: all bins within 1 stored ADU; adjacent amplitudes and exact tie
  count retained. Raw argmax chooses the first exact maximum; no smoothing.
- Validation: 512 contiguous sweeps/block; block modulo four equals three is
  validation. No formula or residual-dependent filter is fitted on these blocks.
  Both dates were used historically, so November 18 is replication, not blind.
- Uncertainty: 2,000 seeded bootstrap draws of block medians, reporting the
  median of block medians. These intervals are not absolute accuracy bounds.
- Examples: seed 20261009, stratified raw candidates plus extreme residuals
  and the sweep 5000/track 214 case. Raw local peaks use prominence
  max(5 ADU, 15% waveform span), spacing≥12 bins; widths≥20 label wide peaks.
  These are morphology labels, not ground/canopy classifications.

## Function contract

`bin_to_xyz(start_xyz, stop_xyz, bins, n_bins, convention="H2")` accepts ECEF
meters with final dimension 3 and broadcastable bins. `n_bins` is an integer≥2;
bins are finite real zero-based indices in `[0,N−1]`. Fractional bins use the
same affine formula, without rounding. It returns ECEF meters; convert with
`Transformer.from_crs(4978,4979,always_xy=True)` for longitude, latitude and
ellipsoidal height. Invalid bins, nonfinite/zero-length endpoints, incompatible
shapes, and unknown conventions raise `ValueError`. It does not clip,
extrapolate, detect peaks, or assert that a sampled bin is a physical return.

For H2, window edges are at mathematical indices −0.5 and N−0.5; the legal
first/last sample centers do not coincide with these edges. Those edge indices
are outside the public function's legal sample-index domain.

## Evidence

`findings.md` is the scientific decision, including unresolved field semantics,
separate granule values, anomalies, external reference limitations, and production
recommendation. `analysis.ipynb` reads the new audit artifacts and displays the
verified figures; it never substitutes archived results for a real-data run.
The official tutorial is [CASALS L1B Waveform Data Tutorial](https://book.cryointhecloud.com/l1b-waveforms-tutorial/).

The supplement rereads original geometry, compares both beam spacings and
stored-value sensitivity, and audits atmospheric correction nonlinearity without
reapplying corrections. `bin_agreement_with_counts.csv` includes the whole
population and screening rejection counts; `n_total` in conditional metric
summaries denotes that subset's size. `field_explanations.csv` separates official
definitions from the new empirical relationships. Reviewed morphology figures
use stronger raw contrast (25% prominence, median+45% span, 20-bin separation)
and retain concurrent morphology/SNR labels, without changing mapping candidates
or closure statistics. Top five stored-SNR pulses augment the seeded examples.

External LAZ headers were inspected in `external_reference_inventory.json`.
Their stored horizontal CRS alone does not establish a vertical datum, epoch,
or matching physical surface. No external accuracy statistic is manufactured.
