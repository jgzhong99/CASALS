"""Single-thread HDF5 reader with a thread-safe latest-request mailbox."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Optional

import h5py
import numpy as np
from PyQt5 import QtCore

from .cross_section import CrossSectionGrid, compute_waveform_cross_section
from .h5_model import H5Model, SweepData


@dataclass(frozen=True)
class FrameParams:
    mode_2d: str = "minus_bg"
    view_2d: str = "curtain"
    mapping_convention: str = "H2"
    n_cols: int = 384
    n_rows: int = 512
    asinh_scale: float = 100.0
    fixed_height_range: Optional[tuple[float, float]] = None
    fixed_x_range: Optional[tuple[float, float]] = None


@dataclass(frozen=True)
class FrameRequest:
    file_generation: int
    request_id: int
    sweep_index: int
    params: FrameParams


@dataclass
class SweepFrame:
    sweep_data: SweepData
    grid: CrossSectionGrid
    request_id: int
    file_generation: int
    sweep_index: int
    history_visible_count: int
    block_read_ms: float
    cross_section_ms: float


class H5Worker(QtCore.QThread):
    """Owns the only persistent H5 handle and discards superseded frames."""

    file_loaded = QtCore.pyqtSignal(int, object)
    frame_ready = QtCore.pyqtSignal(object)
    load_progress = QtCore.pyqtSignal(int, int, str)
    error_occurred = QtCore.pyqtSignal(int, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._condition = threading.Condition()
        self._pending_load: Optional[tuple[str, int]] = None
        self._pending_request: Optional[FrameRequest] = None
        self._requested_generation = 0
        self._latest_request_id = 0
        self._stopped = False
        self._model: Optional[H5Model] = None
        self._h5_file: Optional[h5py.File] = None

    def submit_load(self, file_path: str, generation: int) -> None:
        with self._condition:
            self._requested_generation = int(generation)
            self._latest_request_id += 1
            self._pending_load = (str(file_path), int(generation))
            self._pending_request = None
            self._condition.notify()

    def submit_request(self, request: FrameRequest) -> None:
        with self._condition:
            if self._stopped or request.file_generation != self._requested_generation:
                return
            self._latest_request_id = int(request.request_id)
            self._pending_request = request
            self._condition.notify()

    def _request_is_current(self, request: FrameRequest) -> bool:
        with self._condition:
            return (
                not self._stopped
                and request.file_generation == self._requested_generation
                and request.request_id == self._latest_request_id
            )

    def _close_file(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
        self._model = None

    def _load(self, path: str, generation: int) -> None:
        self._close_file()
        try:
            self.load_progress.emit(generation, 10, f"Opening {Path(path).name}…")
            model = H5Model(path)
            if generation != self._requested_generation:
                return
            self.load_progress.emit(generation, 75, "Opening the worker-owned waveform reader…")
            h5_file = h5py.File(model.path, "r", rdcc_nbytes=8 * 1024 * 1024)
            if generation != self._requested_generation:
                h5_file.close()
                return
            self._h5_file = h5_file
            self._model = model
            self.load_progress.emit(generation, 100, f"Loaded {model.n_valid_sweeps:,} sweeps.")
            self.file_loaded.emit(generation, model)
        except Exception as exc:
            if generation == self._requested_generation:
                self.error_occurred.emit(generation, f"Failed to load H5 file: {exc}")

    def _prepare_frame(self, request: FrameRequest) -> None:
        model, h5_file = self._model, self._h5_file
        if model is None or h5_file is None or not self._request_is_current(request):
            return
        try:
            started = time.perf_counter()
            sweep = model.read_sweep_data(h5_file, request.sweep_index)
            block_read_ms = (time.perf_counter() - started) * 1000.0
            # A replaced request skips expensive 2D work, but the just-read block stays cached.
            if not self._request_is_current(request):
                return
            started = time.perf_counter()
            params = request.params
            grid = compute_waveform_cross_section(
                sweep,
                mode=params.mode_2d,
                view=params.view_2d,
                mapping_convention=params.mapping_convention,
                n_cols=params.n_cols,
                n_rows=params.n_rows,
                asinh_scale=params.asinh_scale,
                fixed_height_range=params.fixed_height_range,
                fixed_x_range=params.fixed_x_range,
            )
            cross_section_ms = (time.perf_counter() - started) * 1000.0
            if not self._request_is_current(request):
                return
            frame = SweepFrame(
                sweep_data=sweep,
                grid=grid,
                request_id=request.request_id,
                file_generation=request.file_generation,
                sweep_index=request.sweep_index,
                history_visible_count=int(model.decimated_offsets[request.sweep_index + 1]),
                block_read_ms=block_read_ms,
                cross_section_ms=cross_section_ms,
            )
            self.frame_ready.emit(frame)
        except Exception as exc:
            if self._request_is_current(request):
                self.error_occurred.emit(request.file_generation, f"Error preparing sweep {request.sweep_index}: {exc}")

    def run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._stopped and self._pending_load is None and self._pending_request is None:
                        self._condition.wait()
                    if self._stopped:
                        break
                    load = self._pending_load
                    if load is not None:
                        self._pending_load = None
                        self._pending_request = None
                    request = None if load is not None else self._pending_request
                    if load is None:
                        self._pending_request = None
                if load is not None:
                    self._load(*load)
                elif request is not None:
                    self._prepare_frame(request)
        finally:
            self._close_file()

    def shutdown(self) -> None:
        with self._condition:
            self._stopped = True
            self._pending_load = None
            self._pending_request = None
            self._condition.notify_all()
        if self.isRunning() and QtCore.QThread.currentThread() is not self:
            self.wait()

