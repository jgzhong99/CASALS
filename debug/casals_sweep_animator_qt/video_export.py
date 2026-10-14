"""Main-thread frame composition and background MP4 encoding."""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Full, Queue
import json
import os
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .h5_model import H5Model, SweepData


def composite_frame_image(
    scene_image: QtGui.QImage | QtGui.QPixmap,
    panel_image: QtGui.QImage | QtGui.QPixmap,
    sweep_data: SweepData,
    model: H5Model,
    mapping_convention: str,
    *,
    output_size: tuple[int, int] = (1920, 1080),
) -> QtGui.QImage:
    """Compose rendered 3D and 2D views in memory on the GUI thread."""
    scene_image = scene_image.toImage() if isinstance(scene_image, QtGui.QPixmap) else scene_image
    panel_image = panel_image.toImage() if isinstance(panel_image, QtGui.QPixmap) else panel_image
    width, height = map(int, output_size)
    width -= width % 2
    height -= height % 2
    header_h = max(56, int(height * 0.065))
    view_h = height - header_h
    scene_h = int(view_h * 0.60)
    panel_h = view_h - scene_h
    output = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
    output.fill(QtGui.QColor("#10161d"))

    valid = sweep_data.valid_pulse_mask & np.isfinite(sweep_data.refh_height)
    if np.any(valid):
        lo, hi = np.nanpercentile(sweep_data.refh_height[valid], [0, 100])
        height_text = f"Refh height {lo:.1f} to {hi:.1f} m"
    else:
        height_text = "Refh unavailable"

    painter = QtGui.QPainter(output)
    painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
    painter.fillRect(0, 0, width, header_h, QtGui.QColor("#171f29"))
    painter.setPen(QtGui.QColor("#f0f4f8"))
    painter.setFont(QtGui.QFont("Segoe UI", max(11, int(height * 0.015)), QtGui.QFont.DemiBold))
    painter.drawText(16, int(header_h * 0.40),
                     f"CASALS Sweep Viewer | Sweep {sweep_data.sweep_id} ({sweep_data.sweep_index + 1}/{sweep_data.total_sweeps})")
    painter.setPen(QtGui.QColor("#afbdcb"))
    painter.setFont(QtGui.QFont("Segoe UI", max(8, int(height * 0.009))))
    painter.drawText(16, int(header_h * 0.77),
                     f"{model.path.name} | {int(np.sum(valid))}/{sweep_data.n_tracks} valid Refh | "
                     f"Mapping {mapping_convention} | Heading {sweep_data.heading_deg:.1f} deg (schematic) | {height_text}")
    painter.drawImage(QtCore.QRect(0, header_h, width, scene_h), scene_image)
    painter.drawImage(QtCore.QRect(0, header_h + scene_h, width, panel_h), panel_image)
    painter.end()
    return output


def qimage_to_rgb(image: QtGui.QImage, *, output_size: tuple[int, int]) -> np.ndarray:
    width, height = map(int, output_size)
    width -= width % 2
    height -= height % 2
    scaled = image.scaled(width, height, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
    rgb = scaled.convertToFormat(QtGui.QImage.Format_RGB888)
    stride = rgb.bytesPerLine()
    ptr = rgb.bits()
    ptr.setsize(rgb.sizeInBytes())
    return np.frombuffer(ptr, dtype=np.uint8).reshape(height, stride)[:, : width * 3].reshape(height, width, 3).copy()


class VideoExportDialog(QtWidgets.QDialog):
    def __init__(self, model: H5Model, current_sweep_id: int, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Export sweep animation")
        self.resize(540, 300)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        low, high = int(model.valid_sweep_ids[0]), int(model.valid_sweep_ids[-1])
        self.spin_start = QtWidgets.QSpinBox()
        self.spin_start.setRange(low, high)
        self.spin_start.setValue(max(low, current_sweep_id - 25))
        self.spin_end = QtWidgets.QSpinBox()
        self.spin_end.setRange(low, high)
        self.spin_end.setValue(min(high, self.spin_start.value() + 49))
        self.spin_step = QtWidgets.QSpinBox()
        self.spin_step.setRange(1, 1000)
        self.spin_step.setValue(1)
        self.combo_fps = QtWidgets.QComboBox()
        self.combo_fps.addItems(["1", "2", "5", "10", "15", "20", "30"])
        self.combo_fps.setCurrentText("10")
        default_out = Path("outputs") / "casals_sweep_animator_qt" / "videos" / f"{model.path.stem}_animation.mp4"
        self.line_path = QtWidgets.QLineEdit(str(default_out.resolve()))
        self.btn_browse = QtWidgets.QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_path)
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.line_path, 1)
        path_row.addWidget(self.btn_browse)
        form.addRow("Start sweep ID", self.spin_start)
        form.addRow("End sweep ID", self.spin_end)
        form.addRow("Step", self.spin_step)
        form.addRow("Frames per second", self.combo_fps)
        form.addRow("Output", path_row)
        layout.addLayout(form)

        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        for control in (self.spin_start, self.spin_end, self.spin_step):
            control.valueChanged.connect(self._update_summary)
        self.combo_fps.currentTextChanged.connect(self._update_summary)
        self.line_path.textChanged.connect(self._update_summary)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        self.btn_export = buttons.addButton("Export", QtWidgets.QDialogButtonBox.AcceptRole)
        self.btn_export.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_summary()

    def _browse_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save MP4", self.line_path.text(), "MP4 video (*.mp4)")
        if path:
            self.line_path.setText(path)

    def _update_summary(self, *_args) -> None:
        start, end, step = self.spin_start.value(), self.spin_end.value(), max(1, self.spin_step.value())
        valid = [int(sid) for sid in self.model.valid_sweep_ids
                 if start <= int(sid) <= end and (int(sid) - start) % step == 0]
        if end < start or not valid:
            self.summary.setText("Select a range that contains at least one valid sweep.")
            self.btn_export.setEnabled(False)
        else:
            fps = float(self.combo_fps.currentText())
            self.summary.setText(f"{len(valid)} frames; {len(valid) / fps:.1f} seconds at {fps:g} FPS; 1920 x 1080 output.")
            self.btn_export.setEnabled(bool(self.line_path.text().strip()))

    def get_settings(self) -> dict[str, Any]:
        return {
            "start_sweep": self.spin_start.value(),
            "end_sweep": self.spin_end.value(),
            "step": self.spin_step.value(),
            "fps": float(self.combo_fps.currentText()),
            "output_path": self.line_path.text(),
        }


class VideoEncoder(QtCore.QThread):
    """Encode RGB frames in a worker thread; this class never accesses widgets."""

    progress = QtCore.pyqtSignal(int, int, str)
    export_done = QtCore.pyqtSignal(str, str, int)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal(int)

    def __init__(self, output_path: str | Path, fps: float, frame_count: int, metadata: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.output_path = Path(output_path).resolve()
        self.fps = float(fps)
        self.expected_frames = int(frame_count)
        self.metadata = dict(metadata)
        self.frames: Queue[np.ndarray] = Queue(maxsize=2)
        self._cancel = threading.Event()
        self._finish = threading.Event()
        self._written = 0

    def offer_frame(self, frame: np.ndarray) -> bool:
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Video frames must be uint8 RGB arrays.")
        try:
            self.frames.put_nowait(array)
            return True
        except Full:
            return False

    def finish_input(self) -> None:
        self._finish.set()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        writer = None
        temp_path = self.output_path.with_name(f".{self.output_path.stem}.{threading.get_ident()}.tmp.mp4")
        temp_meta: Optional[Path] = None
        verified = False
        try:
            if self.expected_frames <= 0:
                raise ValueError("No frames selected for video export.")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            width, height = 1920, 1080
            writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height), True)
            if not writer.isOpened():
                raise OSError(f"OpenCV could not open an MP4 encoder at {self.output_path}")
            start_time = time.perf_counter()
            while not self._cancel.is_set():
                try:
                    frame = self.frames.get(timeout=0.1)
                except Empty:
                    if self._finish.is_set():
                        break
                    continue
                if frame.shape != (height, width, 3):
                    raise ValueError(f"Video frame has shape {frame.shape}; expected {(height, width, 3)}")
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self._written += 1
                self.progress.emit(self._written, self.expected_frames, f"Encoded frame {self._written}/{self.expected_frames}")
            writer.release()
            writer = None
            if self._cancel.is_set():
                self.cancelled.emit(self._written)
                return
            if self._written != self.expected_frames:
                raise RuntimeError(f"Encoded {self._written} frames; expected {self.expected_frames}")

            capture = cv2.VideoCapture(str(temp_path))
            if not capture.isOpened():
                raise RuntimeError("MP4 was written but could not be reopened for validation")
            decoded = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded += 1
            width_out = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_out = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture.release()
            if decoded != self.expected_frames or (width_out, height_out) != (width, height):
                raise RuntimeError(f"MP4 verification failed: decoded {decoded} frames at {width_out} x {height_out}")

            meta_path = self.output_path.with_suffix(".json")
            metadata = {
                **self.metadata,
                "video_file": self.output_path.name,
                "frame_count": self._written,
                "decoded_frame_count": decoded,
                "fps": self.fps,
                "output_dimensions": [width, height],
                "encode_seconds": round(time.perf_counter() - start_time, 3),
                "verified": True,
            }
            temp_meta = meta_path.with_name(f".{meta_path.name}.{threading.get_ident()}.tmp")
            temp_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            os.replace(temp_path, self.output_path)
            os.replace(temp_meta, meta_path)
            verified = True
            self.export_done.emit(str(self.output_path), str(meta_path), decoded)
        except Exception as exc:
            self.failed.emit(f"Video export failed: {exc}")
        finally:
            if writer is not None:
                writer.release()
            if not verified:
                for path in (temp_path, temp_meta):
                    if path is not None:
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            pass

