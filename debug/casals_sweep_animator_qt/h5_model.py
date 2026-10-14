"""HDF5 data model, sweep indexing, and caching for CASALS 3D Sweep Animator.

Scientific boundaries:
- Reads original CASALS L1B HDF5 in read-only mode.
- Respects official refh reference-height points.
- Implements deterministic decimation for historical 3D display (budget ~200k-250k points).
- On-demand waveform fetching with bounded LRU cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import collections
import h5py
import numpy as np

from casals_l1b.h5 import find_dataset, read_attrs_subset, require_dataset
from casals_l1b.waveform import (
    SweepTrackIndex,
    build_record_index_grid,
    infer_waveform_record_axis,
)
from .geometry import (
    LocalENU,
    compute_trajectory_headings,
    estimate_cross_track_direction,
    geodetic_to_ecef,
    project_cross_track,
)

REQUIRED_DATASETS = [
    "sweep_num",
    "track_num",
    "refh_longitude",
    "refh_latitude",
    "refh",
    "instrument_longitude",
    "instrument_latitude",
    "instrument_altitude",
    "rx_waveform",
    "rwstart_longitude",
    "rwstart_latitude",
    "rwstart",
    "rwstop_longitude",
    "rwstop_latitude",
    "rwstop",
]

OPTIONAL_DATASETS = [
    "refh_snr",
    "refh_amp",
    "bg_mean",
    "bg_std",
    "delta_time",
    "good_snr",
]


@dataclass
class SweepData:
    """All geometry, telemetry, and waveform data for a single sweep."""

    sweep_id: int
    sweep_index: int
    total_sweeps: int

    # Pulse metadata in this sweep
    pulse_indices: np.ndarray
    track_nums: np.ndarray
    track_present: np.ndarray
    valid_pulse_mask: np.ndarray

    # Aircraft telemetry for this sweep
    instrument_enu: np.ndarray
    heading_deg: float

    # Official Refh points (un-decimated for current sweep)
    refh_lon: np.ndarray
    refh_lat: np.ndarray
    refh_height: np.ndarray
    refh_enu: np.ndarray
    refh_snr: Optional[np.ndarray]
    refh_amp: Optional[np.ndarray]

    # Range window endpoints
    rwstart_enu: np.ndarray
    rwstop_enu: np.ndarray
    rwstart_ecef: np.ndarray
    rwstop_ecef: np.ndarray
    rwstart_height: np.ndarray
    rwstop_height: np.ndarray

    # Instrument position for each valid pulse
    pulse_instrument_enu: np.ndarray

    # Signal diagnostics
    bg_mean: Optional[np.ndarray]
    bg_std: Optional[np.ndarray]
    delta_time: Optional[np.ndarray]
    good_snr: Optional[np.ndarray]

    # Waveform matrix [n_tracks, n_rx_bins]
    waveform_matrix: np.ndarray
    n_rx_bins: int
    n_tracks: int

    # Cross-track geometry
    can_use_meters: bool
    cross_track_reason: str
    cross_track_unit_vector: Optional[np.ndarray]
    cross_track_center: Optional[np.ndarray]
    cross_track_refh_m: Optional[np.ndarray]


class H5Model:
    """Manages an open CASALS L1B HDF5 dataset, indexing, and on-demand reading."""

    def __init__(
        self,
        file_path: str | Path,
        history_budget: int = 220_000,
        waveform_block_sweeps: Optional[int] = None,
        waveform_cache_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        self.path = Path(file_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"CASALS H5 file not found: {self.path}")

        self.history_budget = int(history_budget)
        self.waveform_cache_limit_bytes = max(0, int(waveform_cache_bytes))
        self._waveform_blocks: collections.OrderedDict[int, tuple[np.ndarray, np.ndarray]] = collections.OrderedDict()
        self.waveform_cache_size_bytes = 0
        self.waveform_cache_hits = 0
        self.waveform_cache_misses = 0

        with h5py.File(self.path, "r") as h5:
            self._validate_datasets(h5)
            self.attrs = read_attrs_subset(h5)
            self.index_info: SweepTrackIndex = build_record_index_grid(h5)
            if self.index_info.duplicate_sweep_track_cells > 0:
                raise ValueError(
                    f"H5 file contains {self.index_info.duplicate_sweep_track_cells} duplicate "
                    "(sweep_num, track_num) cells. Aborting to avoid corrupt animation."
                )

            rx_ds = require_dataset(h5, "rx_waveform")
            self.record_axis = infer_waveform_record_axis(rx_ds, self.index_info.n_records)
            self.n_rx_bins = int(rx_ds.shape[1] if self.record_axis == 0 else rx_ds.shape[0])
            self.n_records = self.index_info.n_records
            self.n_tracks = self.index_info.n_tracks
            chunk_records = rx_ds.chunks[self.record_axis] if rx_ds.chunks else 0
            chunk_sweeps = chunk_records // self.n_tracks if chunk_records and chunk_records % self.n_tracks == 0 else 0
            self.waveform_block_sweeps = max(
                1,
                int(waveform_block_sweeps or chunk_sweeps or 16),
            )

            # Read base 1D fields into compact memory
            self._load_base_fields(h5)

        # Build valid sweep list
        self._build_sweep_index()

        # Compute fixed ENU coordinate system
        self._setup_enu()

        # Build deterministic decimated historical cloud
        self._build_historical_decimation()

        # Compute global color statistics
        self._compute_global_statistics()

    def _validate_datasets(self, h5: h5py.File) -> None:
        missing = [name for name in REQUIRED_DATASETS if find_dataset(h5, name) is None]
        if missing:
            raise KeyError(f"Required datasets missing from H5 file: {missing}")

    def _load_base_fields(self, h5: h5py.File) -> None:
        """Load essential 1D columns into memory."""
        self.sweep_num = np.asarray(require_dataset(h5, "sweep_num")[...], dtype=np.int64).reshape(-1)
        self.track_num = np.asarray(require_dataset(h5, "track_num")[...], dtype=np.int64).reshape(-1)

        self.refh_lon = np.asarray(require_dataset(h5, "refh_longitude")[...], dtype=np.float64).reshape(-1)
        self.refh_lat = np.asarray(require_dataset(h5, "refh_latitude")[...], dtype=np.float64).reshape(-1)
        self.refh_height = np.asarray(require_dataset(h5, "refh")[...], dtype=np.float64).reshape(-1)

        self.inst_lon = np.asarray(require_dataset(h5, "instrument_longitude")[...], dtype=np.float64).reshape(-1)
        self.inst_lat = np.asarray(require_dataset(h5, "instrument_latitude")[...], dtype=np.float64).reshape(-1)
        self.inst_alt = np.asarray(require_dataset(h5, "instrument_altitude")[...], dtype=np.float64).reshape(-1)

        self.rwstart_lon = np.asarray(require_dataset(h5, "rwstart_longitude")[...], dtype=np.float64).reshape(-1)
        self.rwstart_lat = np.asarray(require_dataset(h5, "rwstart_latitude")[...], dtype=np.float64).reshape(-1)
        self.rwstart_h = np.asarray(require_dataset(h5, "rwstart")[...], dtype=np.float64).reshape(-1)

        self.rwstop_lon = np.asarray(require_dataset(h5, "rwstop_longitude")[...], dtype=np.float64).reshape(-1)
        self.rwstop_lat = np.asarray(require_dataset(h5, "rwstop_latitude")[...], dtype=np.float64).reshape(-1)
        self.rwstop_h = np.asarray(require_dataset(h5, "rwstop")[...], dtype=np.float64).reshape(-1)

        # Optional fields
        self.refh_snr = self._read_opt(h5, "refh_snr", dtype=np.float32)
        self.refh_amp = self._read_opt(h5, "refh_amp", dtype=np.float32)
        self.bg_mean = self._read_opt(h5, "bg_mean", dtype=np.float32)
        self.bg_std = self._read_opt(h5, "bg_std", dtype=np.float32)
        self.delta_time = self._read_opt(h5, "delta_time", dtype=np.float64)
        self.good_snr = self._read_opt(h5, "good_snr", dtype=np.bool_)

    def _read_opt(self, h5: h5py.File, name: str, dtype: Any) -> Optional[np.ndarray]:
        ds = find_dataset(h5, name)
        if ds is None:
            return None
        arr = np.asarray(ds[...], dtype=dtype).reshape(-1)
        if arr.size != self.n_records:
            return None
        return arr

    def _build_sweep_index(self) -> None:
        """Identify unique sweeps that actually exist in the file."""
        grid_sweeps = np.arange(self.index_info.n_sweeps, dtype=np.int64)
        # A sweep is valid if at least one track cell is >= 0
        valid_mask = (self.index_info.record_index_grid >= 0).any(axis=1)
        self.valid_sweep_ids = grid_sweeps[valid_mask]
        self.n_valid_sweeps = int(len(self.valid_sweep_ids))

        # Map sweep ID to valid sweep index (0 to n_valid_sweeps - 1)
        self.sweep_id_to_idx = {int(sid): idx for idx, sid in enumerate(self.valid_sweep_ids)}

    def _setup_enu(self) -> None:
        """Establish the single, fixed local ENU origin for the entire file."""
        valid_refh = (
            np.isfinite(self.refh_lon)
            & np.isfinite(self.refh_lat)
            & np.isfinite(self.refh_height)
        )
        if not np.any(valid_refh):
            raise ValueError("No finite Refh points found in H5 to establish local ENU origin.")

        self.origin_lon = float(np.median(self.refh_lon[valid_refh]))
        self.origin_lat = float(np.median(self.refh_lat[valid_refh]))
        self.origin_height = float(np.median(self.refh_height[valid_refh]))

        self.enu = LocalENU(self.origin_lon, self.origin_lat, self.origin_height)

        # Convert all valid Refh points to ENU
        self.refh_enu = np.full((self.n_records, 3), np.nan, dtype=np.float64)
        self.refh_enu[valid_refh] = self.enu.geodetic_to_enu(
            self.refh_lon[valid_refh],
            self.refh_lat[valid_refh],
            self.refh_height[valid_refh],
        )

        # Precompute instrument position per valid sweep
        self.sweep_instrument_enu = np.full((self.n_valid_sweeps, 3), np.nan, dtype=np.float64)
        for s_idx, sid in enumerate(self.valid_sweep_ids):
            pulses = self.index_info.record_index_grid[sid]
            valid_p = pulses[pulses >= 0]
            if len(valid_p):
                # Median instrument position for this sweep
                lons = self.inst_lon[valid_p]
                lats = self.inst_lat[valid_p]
                alts = self.inst_alt[valid_p]
                fin = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(alts)
                if np.any(fin):
                    med_lon = np.median(lons[fin])
                    med_lat = np.median(lats[fin])
                    med_alt = np.median(alts[fin])
                    self.sweep_instrument_enu[s_idx] = self.enu.geodetic_to_enu(med_lon, med_lat, med_alt)

        # Forward fill any NaN instrument positions if present
        for i in range(1, self.n_valid_sweeps):
            if np.isnan(self.sweep_instrument_enu[i, 0]):
                self.sweep_instrument_enu[i] = self.sweep_instrument_enu[i - 1]
        for i in range(self.n_valid_sweeps - 2, -1, -1):
            if np.isnan(self.sweep_instrument_enu[i, 0]):
                self.sweep_instrument_enu[i] = self.sweep_instrument_enu[i + 1]

        # Trajectory headings
        self.sweep_headings = compute_trajectory_headings(self.sweep_instrument_enu)

    def _build_historical_decimation(self) -> None:
        """Create a globally bounded, stratified Refh sample with sweep prefixes."""
        valid_refh = np.isfinite(self.refh_enu[:, 0])
        total_valid = int(np.sum(valid_refh))
        budget = min(total_valid, max(0, self.history_budget))
        counts = np.zeros(self.n_valid_sweeps, dtype=np.int64)
        sweep_records: list[np.ndarray] = []
        for k, sid in enumerate(self.valid_sweep_ids):
            pulse_ids = self.index_info.record_index_grid[sid]
            pulse_ids = pulse_ids[pulse_ids >= 0]
            pulse_ids = pulse_ids[valid_refh[pulse_ids]]
            sweep_records.append(pulse_ids)
            counts[k] = pulse_ids.size

        if total_valid and budget:
            exact = counts.astype(np.float64) * (float(budget) / total_valid)
            quotas = np.floor(exact).astype(np.int64)
            remainder = budget - int(quotas.sum())
            if remainder:
                order = np.argsort(-(exact - quotas), kind="stable")
                quotas[order[:remainder]] += 1
        else:
            quotas = np.zeros(self.n_valid_sweeps, dtype=np.int64)

        decimated_enu_list: List[np.ndarray] = []
        decimated_snr_list: List[np.ndarray] = []
        decimated_height_list: List[np.ndarray] = []
        decimated_amp_list: List[np.ndarray] = []
        decimated_sweep_list: List[np.ndarray] = []

        offsets = [0]
        accum = 0
        for k, valid_p in enumerate(sweep_records):
            n_sample = int(quotas[k])
            if n_sample:
                # Golden-ratio staggered phase across sweeps to distribute across-track pulses
                # uniformly over all tracks, eliminating artificial grooving/streak lines.
                phase = (k * 0.618033988749895) % 1.0
                sample_idx = (np.floor((np.arange(n_sample) + phase) * len(valid_p) / n_sample).astype(np.int64)) % len(valid_p)
                sampled_p = valid_p[sample_idx]
                decimated_enu_list.append(self.refh_enu[sampled_p])
                if self.refh_snr is not None:
                    decimated_snr_list.append(self.refh_snr[sampled_p])
                decimated_height_list.append(self.refh_height[sampled_p])
                if self.refh_amp is not None:
                    decimated_amp_list.append(self.refh_amp[sampled_p])
                decimated_sweep_list.append(np.full(n_sample, k, dtype=np.int32))
                accum += len(sampled_p)
            offsets.append(accum)

        self.decimated_offsets = np.asarray(offsets, dtype=np.int64)
        if decimated_enu_list:
            self.decimated_enu = np.vstack(decimated_enu_list).astype(np.float32, copy=False)
            self.decimated_height = np.concatenate(decimated_height_list).astype(np.float32, copy=False)
            self.decimated_snr = (
                np.concatenate(decimated_snr_list).astype(np.float32, copy=False)
                if decimated_snr_list
                else None
            )
            self.decimated_amp = (
                np.concatenate(decimated_amp_list).astype(np.float32, copy=False)
                if decimated_amp_list
                else None
            )
            self.decimated_sweep_index = np.concatenate(decimated_sweep_list)
        else:
            self.decimated_enu = np.empty((0, 3), dtype=np.float32)
            self.decimated_height = np.empty(0, dtype=np.float32)
            self.decimated_snr = None
            self.decimated_amp = None
            self.decimated_sweep_index = np.empty(0, dtype=np.int32)

        self.total_decimated_points = int(self.decimated_enu.shape[0])

    def _compute_global_statistics(self) -> None:
        """Estimate stable full-file percentiles for colormaps so they don't flicker per frame."""
        sample_step = max(1, self.n_records // 50_000)
        h_sample = self.refh_height[::sample_step]
        h_fin = h_sample[np.isfinite(h_sample)]
        if len(h_fin):
            self.height_vmin = float(np.percentile(h_fin, 1.0))
            self.height_vmax = float(np.percentile(h_fin, 99.0))
            if self.height_vmax <= self.height_vmin:
                self.height_vmax = self.height_vmin + 10.0
        else:
            self.height_vmin, self.height_vmax = 0.0, 100.0

        if self.refh_snr is not None:
            snr_sample = self.refh_snr[::sample_step]
            snr_fin = snr_sample[np.isfinite(snr_sample)]
            if len(snr_fin):
                self.snr_vmin = float(np.percentile(snr_fin, 2.0))
                self.snr_vmax = float(np.percentile(snr_fin, 98.0))
            else:
                self.snr_vmin, self.snr_vmax = 0.0, 50.0
        else:
            self.snr_vmin, self.snr_vmax = 0.0, 50.0

        if self.refh_amp is not None:
            amp_sample = self.refh_amp[::sample_step]
            amp_fin = amp_sample[np.isfinite(amp_sample)]
            if len(amp_fin):
                self.amp_vmin = float(np.percentile(amp_fin, 2.0))
                self.amp_vmax = float(np.percentile(amp_fin, 98.0))
            else:
                self.amp_vmin, self.amp_vmax = 0.0, 1000.0
        else:
            self.amp_vmin, self.amp_vmax = 0.0, 1000.0

        range_heights = np.concatenate([self.rwstart_h, self.rwstop_h])
        finite_range = range_heights[np.isfinite(range_heights)]
        if finite_range.size:
            self.height_range = (
                float(np.percentile(finite_range, 1.0)),
                float(np.percentile(finite_range, 99.0)),
            )
            if self.height_range[1] <= self.height_range[0]:
                self.height_range = (self.height_range[0] - 1.0, self.height_range[0] + 1.0)
        else:
            self.height_range = (float(self.height_vmin), float(self.height_vmax))

    def get_history_slice(
        self, sweep_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Return deterministic historical points strictly up to the given sweep index.

        Returns (enu, height, snr, amp).
        """
        k = max(0, min(self.n_valid_sweeps - 1, int(sweep_idx)))
        end_offset = int(self.decimated_offsets[k + 1])
        return (
            self.decimated_enu[:end_offset],
            self.decimated_height[:end_offset],
            self.decimated_snr[:end_offset] if self.decimated_snr is not None else None,
            self.decimated_amp[:end_offset] if self.decimated_amp is not None else None,
        )

    def _read_waveform_block(self, h5: h5py.File, block_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Read one sweep block, using a contiguous slab only when record IDs prove it safe."""
        block_start = block_index * self.waveform_block_sweeps
        block_stop = min(self.n_valid_sweeps, block_start + self.waveform_block_sweeps)
        sweep_ids = self.valid_sweep_ids[block_start:block_stop]
        record_grid = self.index_info.record_index_grid[sweep_ids]
        record_ids = np.sort(record_grid[record_grid >= 0])
        if record_ids.size == 0:
            return record_ids, np.empty((0, self.n_rx_bins), dtype=np.float32)

        first = int(record_ids[0])
        last = int(record_ids[-1]) + 1
        rx_ds = require_dataset(h5, "rx_waveform")
        if record_ids.size == last - first:
            if self.record_axis == 0:
                waveforms = np.asarray(rx_ds[first:last, :])
            else:
                waveforms = np.asarray(rx_ds[:, first:last]).T
            if not np.array_equal(record_ids, np.arange(first, last, dtype=np.int64)):
                raise AssertionError("contiguous waveform block record check failed")
        else:
            if self.record_axis == 0:
                waveforms = np.asarray(rx_ds[record_ids, :])
            else:
                waveforms = np.asarray(rx_ds[:, record_ids]).T

        nbytes = int(record_ids.nbytes + waveforms.nbytes)
        if self.waveform_cache_limit_bytes and nbytes <= self.waveform_cache_limit_bytes:
            while self._waveform_blocks and self.waveform_cache_size_bytes + nbytes > self.waveform_cache_limit_bytes:
                _, (old_ids, old_values) = self._waveform_blocks.popitem(last=False)
                self.waveform_cache_size_bytes -= int(old_ids.nbytes + old_values.nbytes)
            self._waveform_blocks[block_index] = (record_ids, waveforms)
            self.waveform_cache_size_bytes += nbytes
        return record_ids, waveforms

    def clear_waveform_cache(self) -> None:
        self._waveform_blocks.clear()
        self.waveform_cache_size_bytes = 0

    def waveform_cache_stats(self) -> dict[str, int]:
        return {
            "hits": int(self.waveform_cache_hits),
            "misses": int(self.waveform_cache_misses),
            "bytes": int(self.waveform_cache_size_bytes),
            "limit_bytes": int(self.waveform_cache_limit_bytes),
        }

    def read_sweep_data(self, h5: h5py.File, sweep_idx: int) -> SweepData:
        """Read a full-track aligned sweep using the bounded waveform block cache."""
        k = max(0, min(self.n_valid_sweeps - 1, int(sweep_idx)))
        sid = int(self.valid_sweep_ids[k])

        pulse_indices = self.index_info.record_index_grid[sid].copy()
        track_present = pulse_indices >= 0
        track_nums = np.arange(self.n_tracks, dtype=np.int64)

        def gather(values: Optional[np.ndarray], *, dtype: Any = np.float64, fill: Any = np.nan) -> Optional[np.ndarray]:
            if values is None:
                return None
            out = np.full(self.n_tracks, fill, dtype=dtype)
            out[track_present] = values[pulse_indices[track_present]]
            return out

        refh_lon = gather(self.refh_lon)
        refh_lat = gather(self.refh_lat)
        refh_height = gather(self.refh_height)
        refh_snr = gather(self.refh_snr, dtype=np.float32)
        refh_amp = gather(self.refh_amp, dtype=np.float32)
        bg_mean = gather(self.bg_mean, dtype=np.float32)
        bg_std = gather(self.bg_std, dtype=np.float32)
        delta_time = gather(self.delta_time)
        good_snr = gather(self.good_snr, dtype=np.bool_, fill=False)

        instrument_lon = gather(self.inst_lon)
        instrument_lat = gather(self.inst_lat)
        instrument_alt = gather(self.inst_alt)
        pulse_inst_enu = self.enu.geodetic_to_enu(instrument_lon, instrument_lat, instrument_alt)
        refh_enu = self.enu.geodetic_to_enu(refh_lon, refh_lat, refh_height)

        rwstart_lon = gather(self.rwstart_lon)
        rwstart_lat = gather(self.rwstart_lat)
        rwstart_height = gather(self.rwstart_h)
        rwstop_lon = gather(self.rwstop_lon)
        rwstop_lat = gather(self.rwstop_lat)
        rwstop_height = gather(self.rwstop_h)
        rwstart_ecef = geodetic_to_ecef(rwstart_lon, rwstart_lat, rwstart_height)
        rwstop_ecef = geodetic_to_ecef(rwstop_lon, rwstop_lat, rwstop_height)
        rwstart_enu = self.enu.ecef_to_enu(rwstart_ecef)
        rwstop_enu = self.enu.ecef_to_enu(rwstop_ecef)

        block_index = k // self.waveform_block_sweeps
        cached = self._waveform_blocks.pop(block_index, None)
        if cached is None:
            self.waveform_cache_misses += 1
            block_ids, block_waveforms = self._read_waveform_block(h5, block_index)
        else:
            self.waveform_cache_hits += 1
            block_ids, block_waveforms = cached
            self._waveform_blocks[block_index] = cached
        valid_tracks = np.flatnonzero(track_present)
        waveform_matrix = np.full((self.n_tracks, self.n_rx_bins), np.nan, dtype=np.float32)
        if valid_tracks.size:
            rows = np.searchsorted(block_ids, pulse_indices[valid_tracks])
            if np.any(rows >= block_ids.size) or not np.array_equal(block_ids[rows], pulse_indices[valid_tracks]):
                raise ValueError(f"Waveform block does not contain every pulse in sweep {sid}.")
            waveform_matrix[valid_tracks] = block_waveforms[rows].astype(np.float32, copy=False)

        # Sweep instrument position and heading
        instrument_enu = self.sweep_instrument_enu[k]
        heading_deg = float(self.sweep_headings[k])

        # Cross-track geometry estimation
        valid_refh_mask = track_present & np.isfinite(refh_enu).all(axis=1)
        if np.any(valid_refh_mask):
            v, center, can_meters, reason = estimate_cross_track_direction(
                refh_enu[valid_refh_mask, :2], track_nums[valid_refh_mask],
            )
            if can_meters and v is not None and center is not None:
                cross_track_refh_m = project_cross_track(refh_enu[:, :2], v, center)
            else:
                cross_track_refh_m = None
        else:
            v, center, can_meters, reason = None, None, False, "No valid Refh points."
            cross_track_refh_m = None

        return SweepData(
            sweep_id=sid,
            sweep_index=k,
            total_sweeps=self.n_valid_sweeps,
            pulse_indices=pulse_indices,
            track_nums=track_nums,
            track_present=track_present,
            valid_pulse_mask=valid_refh_mask,
            instrument_enu=instrument_enu,
            heading_deg=heading_deg,
            refh_lon=refh_lon,
            refh_lat=refh_lat,
            refh_height=refh_height,
            refh_enu=refh_enu,
            refh_snr=refh_snr,
            refh_amp=refh_amp,
            rwstart_enu=rwstart_enu,
            rwstop_enu=rwstop_enu,
            rwstart_ecef=rwstart_ecef,
            rwstop_ecef=rwstop_ecef,
            rwstart_height=rwstart_height,
            rwstop_height=rwstop_height,
            pulse_instrument_enu=pulse_inst_enu,
            bg_mean=bg_mean,
            bg_std=bg_std,
            delta_time=delta_time,
            good_snr=good_snr,
            waveform_matrix=waveform_matrix,
            n_rx_bins=self.n_rx_bins,
            n_tracks=self.n_tracks,
            can_use_meters=can_meters,
            cross_track_reason=reason,
            cross_track_unit_vector=v,
            cross_track_center=center,
            cross_track_refh_m=cross_track_refh_m,
        )

