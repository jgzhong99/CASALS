"""Asynchronous background worker for non-blocking HDF5 reads and grid generation.

Scientific boundaries & requirements:
- HDF5 file handle is strictly owned by the worker thread.
- Returns pure NumPy data structures (SweepFrame), does not manipulate Qt GUI items directly.
- Handles pending-latest request semantics so rapid slider scrubbing does not queue outdated sweeps.
- Supports clean thread termination and resource cleanup on file switch / application exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
from PyQt5 import QtCore

from .cross_section import CrossSectionGrid, compute_waveform_cross_section
from .h5_model import H5Model, SweepData


@dataclass
class FrameParams:
    """Rendering parameters needed by the worker to prepare a SweepFrame."""

    mode_2d: str = "minus_bg"
    x_axis_mode: str = "meters"
    mapping_convention: str = "H2"
    point_color_mode: str = "height"
    n_cols: int = 384
    n_rows: int = 512
    asinh_scale: float = 100.0
    fixed_height_range: Optional[tuple[float, float]] = None


@dataclass
class SweepFrame:
    """Pre-computed frame package emitted by worker to GUI thread."""

    sweep_data: SweepData
    history_enu: np.ndarray
    history_values: Optional[np.ndarray]
    grid: CrossSectionGrid
    request_id: int
    all_trajectory_enu: np.ndarray


class H5Worker(QtCore.QObject):
    """Background worker for loading H5 files and computing sweep frames asynchronously."""

    # Signals to GUI thread
    file_loaded = QtCore.pyqtSignal(object)  # Emits H5Model
    frame_ready = QtCore.pyqtSignal(object)  # Emits SweepFrame
    load_progress = QtCore.pyqtSignal(int, str)  # progress %, status message
    error_occurred = QtCore.pyqtSignal(str)  # error message

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[H5Model] = None
        self._h5_file: Optional[h5py.File] = None
        self._pending_request: Optional[tuple[int, FrameParams, int]] = None
        self._is_busy = False
        self._is_stopped = False
        self._lock = QtCore.QMutex()

    @QtCore.pyqtSlot(str)
    def load_file(self, file_path: str) -> None:
        """Load and initialize H5 model in background thread."""
        self._lock.lock()
        try:
            self._close_current()
            self.load_progress.emit(10, f"Opening {Path(file_path).name}...")

            # Initialize model (reads metadata, builds grid, ENU, decimation)
            model = H5Model(file_path)
            self.load_progress.emit(70, "Opening persistent H5 reader handle...")

            # Open worker's private H5 handle for subsequent waveform reading
            self._h5_file = h5py.File(model.path, "r")
            self.model = model

            self.load_progress.emit(100, f"Loaded {model.n_valid_sweeps:,} sweeps.")
            self.file_loaded.emit(model)
        except Exception as exc:
            self.error_occurred.emit(f"Failed to load H5 file: {exc}")
        finally:
            self._lock.unlock()

    @QtCore.pyqtSlot(int, object, int)
    def request_sweep(self, sweep_idx: int, params: FrameParams, request_id: int) -> None:
        """Enqueue a sweep read request.

        If a request is already being processed, sets this as the latest pending request,
        discarding any previous intermediate skipped requests.
        """
        self._lock.lock()
        self._pending_request = (sweep_idx, params, request_id)
        if self._is_busy:
            self._lock.unlock()
            return
        self._is_busy = True
        self._lock.unlock()

        # Trigger processing loop
        self._process_requests()

    def _process_requests(self) -> None:
        """Loop through pending requests, always picking the latest available."""
        while not self._is_stopped:
            self._lock.lock()
            if self._pending_request is None:
                self._is_busy = False
                self._lock.unlock()
                return

            sweep_idx, params, request_id = self._pending_request
            self._pending_request = None
            model = self.model
            h5 = self._h5_file
            self._lock.unlock()

            if model is None or h5 is None:
                continue

            try:
                # 1. Read single sweep data and waveforms
                sweep_data = model.read_sweep_data(h5, sweep_idx)

                # 2. Get deterministic historical Refh slice
                hist_enu, hist_h, hist_snr, hist_amp = model.get_history_slice(sweep_idx)

                # Select values for history coloring
                if params.point_color_mode == "snr":
                    hist_val = hist_snr
                elif params.point_color_mode == "amp":
                    hist_val = hist_amp
                else:  # 'height' or 'uniform'
                    hist_val = hist_h

                # 3. Compute 2D cross-section grid
                grid = compute_waveform_cross_section(
                    sweep=sweep_data,
                    mode=params.mode_2d,
                    x_axis_mode=params.x_axis_mode,
                    mapping_convention=params.mapping_convention,
                    n_cols=params.n_cols,
                    n_rows=params.n_rows,
                    asinh_scale=params.asinh_scale,
                    fixed_height_range=params.fixed_height_range,
                )

                # 4. Check if request was superseded while processing
                self._lock.lock()
                is_stale = (
                    self._pending_request is not None
                    and self._pending_request[2] > request_id
                )
                self._lock.unlock()

                if not is_stale and not self._is_stopped:
                    frame = SweepFrame(
                        sweep_data=sweep_data,
                        history_enu=hist_enu,
                        history_values=hist_val,
                        grid=grid,
                        request_id=request_id,
                        all_trajectory_enu=model.sweep_instrument_enu,
                    )
                    self.frame_ready.emit(frame)
            except Exception as exc:
                if not self._is_stopped:
                    self.error_occurred.emit(f"Error reading sweep {sweep_idx}: {exc}")

    def _close_current(self) -> None:
        """Close existing H5 file handle and clear state."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None
        self.model = None
        self._pending_request = None

    @QtCore.pyqtSlot()
    def stop(self) -> None:
        """Cleanly stop worker loop and release file handles."""
        self._is_stopped = True
        self._lock.lock()
        self._close_current()
        self._lock.unlock()

