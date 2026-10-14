"""GUI-thread playback scheduling and request identity checks."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

from PyQt5 import QtCore

from .h5_model import H5Model
from .worker import FrameParams, FrameRequest, H5Worker, SweepFrame


class FrameController(QtCore.QObject):
    model_loaded = QtCore.pyqtSignal(object, int)
    frame_ready = QtCore.pyqtSignal(object)
    load_progress = QtCore.pyqtSignal(int, str)
    error_occurred = QtCore.pyqtSignal(str)
    position_changed = QtCore.pyqtSignal(int)
    playback_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.worker = H5Worker()
        self.worker.file_loaded.connect(self._on_file_loaded)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.load_progress.connect(self._on_load_progress)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

        self.file_generation = 0
        self.request_id = 0
        self.model: Optional[H5Model] = None
        self.current_sweep_idx = 0
        self.current_params = FrameParams()
        self.requested_fps = 5.0
        self.sweep_step = 1
        self.is_playing = False
        self._waiting = False
        self._request_started = 0.0
        self._timer_armed_at = 0.0
        self._timer_interval_ms = 0
        self._timer_lateness_ms = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance_playback)

    def load_file(self, path: str | Path) -> int:
        self.pause()
        self.file_generation += 1
        self.request_id += 1
        self.model = None
        self._waiting = False
        self._timer.stop()
        self.current_sweep_idx = 0
        self.worker.submit_load(str(path), self.file_generation)
        return self.file_generation

    def request_frame(self, sweep_index: int, params: Optional[FrameParams] = None) -> None:
        if self.model is None:
            return
        self._timer.stop()
        self.current_sweep_idx = max(0, min(int(sweep_index), self.model.n_valid_sweeps - 1))
        self.current_params = params or self.current_params
        self.request_id += 1
        self._waiting = True
        self._request_started = time.perf_counter()
        self.position_changed.emit(self.current_sweep_idx)
        self.worker.submit_request(FrameRequest(
            file_generation=self.file_generation,
            request_id=self.request_id,
            sweep_index=self.current_sweep_idx,
            params=self.current_params,
        ))

    def update_params(self, params: FrameParams) -> None:
        self.request_frame(self.current_sweep_idx, params)

    def play(self, fps: Optional[float] = None, step: Optional[int] = None) -> None:
        if self.model is None:
            return
        if fps is not None:
            self.requested_fps = max(0.1, float(fps))
        if step is not None:
            self.sweep_step = max(1, int(step))
        self.is_playing = True
        self.playback_changed.emit(True)
        if not self._waiting:
            self._schedule_next()

    def pause(self) -> None:
        self._timer.stop()
        if self.is_playing:
            self.is_playing = False
            self.playback_changed.emit(False)

    def set_playback(self, fps: float, step: int) -> None:
        self.requested_fps = max(0.1, float(fps))
        self.sweep_step = max(1, int(step))

    def _schedule_next(self) -> None:
        if not self.is_playing:
            return
        period_ms = 1000.0 / self.requested_fps
        elapsed_ms = (time.perf_counter() - self._request_started) * 1000.0
        cadence_margin_ms = min(5.0, period_ms * 0.05)
        remaining_ms = max(0.0, period_ms - elapsed_ms - self._timer_lateness_ms - cadence_margin_ms)
        self._timer_interval_ms = max(0, int(round(remaining_ms)))
        self._timer_armed_at = time.perf_counter()
        self._timer.start(self._timer_interval_ms)

    def _advance_playback(self) -> None:
        if not self.is_playing or self.model is None:
            return
        if self._timer_armed_at:
            actual_wait_ms = (time.perf_counter() - self._timer_armed_at) * 1000.0
            observed_lateness_ms = max(0.0, actual_wait_ms - self._timer_interval_ms)
            # Compensate for this Windows event-loop timer's measured wake-up delay.
            self._timer_lateness_ms = min(20.0, 0.75 * self._timer_lateness_ms + 0.25 * observed_lateness_ms)
            self._timer_armed_at = 0.0
        next_idx = self.current_sweep_idx + self.sweep_step
        if next_idx >= self.model.n_valid_sweeps:
            next_idx = 0
        self.request_frame(next_idx)

    @QtCore.pyqtSlot(int, object)
    def _on_file_loaded(self, generation: int, model: H5Model) -> None:
        if generation != self.file_generation:
            return
        self.model = model
        self.current_sweep_idx = 0
        self.model_loaded.emit(model, generation)
        self.request_frame(0)

    @QtCore.pyqtSlot(int, int, str)
    def _on_load_progress(self, generation: int, percent: int, message: str) -> None:
        if generation == self.file_generation:
            self.load_progress.emit(percent, message)

    @QtCore.pyqtSlot(int, str)
    def _on_error(self, generation: int, message: str) -> None:
        if generation == self.file_generation:
            self._waiting = False
            self.error_occurred.emit(message)

    @QtCore.pyqtSlot(object)
    def _on_frame_ready(self, frame: SweepFrame) -> None:
        if (
            frame.file_generation != self.file_generation
            or frame.request_id != self.request_id
            or frame.sweep_index != self.current_sweep_idx
        ):
            return
        self._waiting = False
        self.frame_ready.emit(frame)
        if self.is_playing:
            self._schedule_next()

    def shutdown(self) -> None:
        self.pause()
        self.worker.shutdown()

