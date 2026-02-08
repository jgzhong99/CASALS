"""Notebook-aligned TDMS parser for the CASALS GUI.

Implements exactly the same extraction path as `read_tdms.ipynb`:
1) parse channels named `Sweep{s}Step{st}`
2) build row cube `(sweep, step, samples)`
3) apply `TX_SAMPLES_CUT`
4) flatten to Ordering B: step-major -> sweep-minor
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TDMS_IMPORT_ERROR: Exception | None = None
try:
    from nptdms import TdmsFile
except Exception as exc:  # pragma: no cover - runtime dependency
    TdmsFile = Any  # type: ignore[assignment]
    TDMS_IMPORT_ERROR = exc


def tdms_available() -> bool:
    return TDMS_IMPORT_ERROR is None


@dataclass(frozen=True)
class TdmsMeta:
    path: Path
    group_name: str
    rows_per_file: int
    sweeps_per_cycle: int
    steps_per_sweep: int
    footprints_per_row: int
    samples_per_row: int
    samples_per_row_rx: int
    channels: int
    tx_samples_cut: tuple[int, int]
    dr_m_per_sample: float


class CasalsTdmsProcessor:
    """Minimal TDMS processor used by the GUI."""

    DEFAULT_GROUP = "CASALS"
    CHANNEL_PATTERN = re.compile(r"^Sweep(\d+)Step(\d+)$")
    EXPECTED_FOOTPRINTS_PER_ROW = 256

    def __init__(self, tx_samples_cut: tuple[int, int] = (8, 68)) -> None:
        self.tdms: TdmsFile | None = None
        self.group = None
        self.channel_index: dict[tuple[int, int], object] = {}

        self.wvl_steps_per_sweep = 32
        self.sweeps_per_cycle = 8
        self.rows_per_file = 0
        self.samples_per_row = 0
        self.samples_per_row_rx = 0
        self.footprints_per_row = self.EXPECTED_FOOTPRINTS_PER_ROW
        self.tx_samples_cut = tx_samples_cut

        c = 299_792_458.0
        fs = 1e9
        self.dr = 0.5 * c / fs

        self._cache_row_id: int | None = None
        self._cache_row: np.ndarray | None = None

    def close(self) -> None:
        if self.tdms is not None:
            try:
                self.tdms.close()
            except Exception:
                pass
        self.tdms = None
        self.group = None
        self.channel_index = {}
        self.rows_per_file = 0
        self.samples_per_row = 0
        self.samples_per_row_rx = 0
        self._cache_row_id = None
        self._cache_row = None

    def _validate_tx_cut(self) -> tuple[int, int]:
        head_cut = int(self.tx_samples_cut[0])
        tail_cut = int(self.tx_samples_cut[1])
        if head_cut < 0 or tail_cut < 0:
            raise ValueError(f"TX_SAMPLES_CUT must be non-negative, got {self.tx_samples_cut}")
        if self.samples_per_row <= 0:
            raise ValueError("samples_per_row is not initialized.")
        if head_cut + tail_cut >= self.samples_per_row:
            raise ValueError(
                f"Invalid TX_SAMPLES_CUT={self.tx_samples_cut}; "
                f"need head+tail < samples_per_row={self.samples_per_row}"
            )
        return head_cut, tail_cut

    def _require_loaded(self) -> None:
        if self.tdms is None or self.group is None:
            raise RuntimeError("No TDMS file loaded.")

    def load(self, path: str | Path) -> TdmsMeta:
        if not tdms_available():
            raise RuntimeError(
                "TDMS support is unavailable because 'nptdms' is not installed. "
                "Install dependency with: pip install nptdms"
            )

        tdms_path = Path(path).expanduser()
        if not tdms_path.exists():
            raise FileNotFoundError(f"TDMS file does not exist: {tdms_path}")

        self.close()
        self.tdms = TdmsFile.open(tdms_path)

        group_names = [g.name for g in self.tdms.groups()]
        if not group_names:
            raise ValueError("No groups found in TDMS file.")

        group_name = self.DEFAULT_GROUP if self.DEFAULT_GROUP in group_names else group_names[0]
        self.group = self.tdms[group_name]

        self.wvl_steps_per_sweep = int(self.group.properties.get("# WVLSteps/Sweep", 32))
        self.sweeps_per_cycle = int(self.group.properties.get("# Sweeps/ShiftCycle", 8))
        self.rows_per_file = int(self.group.properties.get("Sweeps/File", 0))
        self.footprints_per_row = self.wvl_steps_per_sweep * self.sweeps_per_cycle
        if self.footprints_per_row != self.EXPECTED_FOOTPRINTS_PER_ROW:
            raise ValueError(
                f"Invalid footprint count per row: {self.footprints_per_row}. "
                f"Expected exactly {self.EXPECTED_FOOTPRINTS_PER_ROW} "
                "(track index must be 0..255)."
            )
        if self.rows_per_file <= 0:
            raise ValueError("Group property 'Sweeps/File' is missing or <= 0.")

        self.channel_index = {}
        for channel in self.group.channels():
            match = self.CHANNEL_PATTERN.match(channel.name)
            if match is None:
                continue
            sweep = int(match.group(1))
            step = int(match.group(2))
            self.channel_index[(sweep, step)] = channel

        if not self.channel_index:
            raise ValueError("No channels matched pattern Sweep{s}Step{st}.")

        expected_keys = {
            (sweep, step)
            for sweep in range(self.sweeps_per_cycle)
            for step in range(self.wvl_steps_per_sweep)
        }
        missing_keys = sorted(expected_keys - set(self.channel_index))
        if missing_keys:
            preview = ", ".join(f"Sweep{s}Step{st}" for s, st in missing_keys[:8])
            if len(missing_keys) > 8:
                preview += ", ..."
            raise ValueError(
                f"Missing {len(missing_keys)} expected channels; "
                f"first missing: {preview}"
            )

        ref_channel = self.channel_index.get((0, 0))
        if ref_channel is None:
            ref_channel = next(iter(self.channel_index.values()))
        channel_length = len(ref_channel)
        if channel_length % self.rows_per_file != 0:
            raise ValueError(
                f"Channel length {channel_length} is not divisible by rows={self.rows_per_file}; "
                "reshape would be ambiguous."
            )
        for (sweep, step), channel in self.channel_index.items():
            this_length = len(channel)
            if this_length != channel_length:
                raise ValueError(
                    "Inconsistent channel lengths detected: "
                    f"Sweep{sweep}Step{step} has {this_length}, expected {channel_length}."
                )

        self.samples_per_row = channel_length // self.rows_per_file
        head_cut, tail_cut = self._validate_tx_cut()
        self.samples_per_row_rx = self.samples_per_row - head_cut - tail_cut
        self._cache_row_id = None
        self._cache_row = None

        return TdmsMeta(
            path=tdms_path,
            group_name=group_name,
            rows_per_file=self.rows_per_file,
            sweeps_per_cycle=self.sweeps_per_cycle,
            steps_per_sweep=self.wvl_steps_per_sweep,
            footprints_per_row=self.footprints_per_row,
            samples_per_row=self.samples_per_row,
            samples_per_row_rx=self.samples_per_row_rx,
            channels=len(self.channel_index),
            tx_samples_cut=(head_cut, tail_cut),
            dr_m_per_sample=self.dr,
        )

    def extract_vis_swath_row(self, row_id: int) -> np.ndarray:
        """Return one row in notebook Ordering B with shape `(256, samples_rx)`."""
        self._require_loaded()
        if row_id < 0 or row_id >= self.rows_per_file:
            raise ValueError(f"row_id out of range: {row_id} (expected 0..{self.rows_per_file - 1})")

        if self._cache_row_id == row_id and self._cache_row is not None:
            return self._cache_row

        head_cut, tail_cut = self._validate_tx_cut()
        row_start = row_id * self.samples_per_row
        row_end = row_start + self.samples_per_row

        row_cube = np.empty(
            (self.sweeps_per_cycle, self.wvl_steps_per_sweep, self.samples_per_row),
            dtype=np.int16,
        )

        for sweep in range(self.sweeps_per_cycle):
            for step in range(self.wvl_steps_per_sweep):
                channel = self.channel_index[(sweep, step)]

                segment = np.asarray(channel[row_start:row_end], dtype=np.int16)
                if segment.size != self.samples_per_row:
                    raise ValueError(
                        f"Unexpected waveform length for row={row_id}, sweep={sweep}, step={step}: "
                        f"{segment.size} (expected {self.samples_per_row})"
                    )
                row_cube[sweep, step, :] = segment

        end_idx = -tail_cut if tail_cut > 0 else None
        row_cube_rx = row_cube[:, :, head_cut:end_idx]
        row = row_cube_rx.transpose(1, 0, 2).reshape(self.footprints_per_row, row_cube_rx.shape[-1])
        if row.shape[0] != self.EXPECTED_FOOTPRINTS_PER_ROW:
            raise ValueError(
                f"Unexpected row footprint dimension: {row.shape[0]} "
                f"(expected {self.EXPECTED_FOOTPRINTS_PER_ROW})."
            )
        self._cache_row_id = row_id
        self._cache_row = row
        return row
