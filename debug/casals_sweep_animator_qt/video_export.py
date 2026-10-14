"""Video and screenshot export tools for CASALS 3D Sweep Animator.

Scientific boundaries & requirements:
- Exports synchronized 3D + 2D MP4 animation and PNG screenshots.
- Never loads entire H5 waveform dataset into memory; reads sweep-by-sweep.
- Writes companion JSON metadata sidecar documenting all rendering parameters.
- Allows user to specify sweep range, step, and FPS with frame count pre-estimation.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time
import cv2
import h5py
import matplotlib
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .cross_section import compute_waveform_cross_section
from .h5_model import H5Model, SweepData


def capture_widget_pixmap(widget: QtWidgets.QWidget) -> QtGui.QPixmap:
    """Grab a crisp pixmap from a Qt widget (including OpenGL views)."""
    pixmap = widget.grab()
    return pixmap


def composite_screenshot(
    scene3d_pixmap: QtGui.QPixmap,
    panel2d_pixmap: QtGui.QPixmap,
    sweep_data: SweepData,
    model: H5Model,
    mapping_convention: str,
    output_path: str | Path,
) -> Path:
    """Composite 3D and 2D views with scientific title and telemetry banner into a PNG."""
    p3d = scene3d_pixmap
    p2d = panel2d_pixmap

    # Target width is max of both widgets
    target_w = max(p3d.width(), p2d.width(), 1200)
    # Header height 70px
    header_h = 70

    # Scale pixmaps to target width while keeping aspect ratio
    p3d_scaled = p3d.scaledToWidth(target_w, QtCore.Qt.SmoothTransformation)
    p2d_scaled = p2d.scaledToWidth(target_w, QtCore.Qt.SmoothTransformation)

    total_h = header_h + p3d_scaled.height() + p2d_scaled.height()

    composite = QtGui.QImage(target_w, total_h, QtGui.QImage.Format_ARGB32)
    composite.fill(QtGui.QColor(15, 18, 24))

    painter = QtGui.QPainter(composite)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

    # Draw header banner
    banner_rect = QtCore.QRect(0, 0, target_w, header_h)
    painter.fillRect(banner_rect, QtGui.QColor(22, 28, 38))

    # Title text
    font_title = QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold)
    painter.setFont(font_title)
    painter.setPen(QtGui.QColor(240, 244, 250))
    painter.drawText(
        16,
        28,
        f"CASALS 3D Sweep Animator | Sweep {sweep_data.sweep_id} ({sweep_data.sweep_index + 1}/{sweep_data.total_sweeps})",
    )

    # Subtitle / Telemetry text
    font_sub = QtGui.QFont("Segoe UI", 9)
    painter.setFont(font_sub)
    painter.setPen(QtGui.QColor(160, 175, 195))

    valid_refh = int(np.sum(sweep_data.valid_pulse_mask))
    h_min = float(np.nanmin(sweep_data.refh_height[sweep_data.valid_pulse_mask])) if valid_refh else np.nan
    h_max = float(np.nanmax(sweep_data.refh_height[sweep_data.valid_pulse_mask])) if valid_refh else np.nan
    sub_text = (
        f"File: {model.path.name} | Valid Refh: {valid_refh}/{sweep_data.n_tracks} | "
        f"Refh Height: [{h_min:.1f}, {h_max:.1f}] m | Mapping: {mapping_convention} | "
        f"Aircraft Heading: {sweep_data.heading_deg:.1f}° (schematic)"
    )
    painter.drawText(16, 52, sub_text)

    # Draw 3D view
    y_offset = header_h
    painter.drawPixmap(0, y_offset, p3d_scaled)

    # Draw 2D view
    y_offset += p3d_scaled.height()
    painter.drawPixmap(0, y_offset, p2d_scaled)

    painter.end()

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    composite.save(str(out), "PNG", quality=95)
    return out


class VideoExportWorker(QtCore.QThread):
    """Background worker for frame-by-frame rendering and MP4 encoding."""

    progress = QtCore.pyqtSignal(int, int, str)  # curr_frame, total_frames, status_msg
    finished = QtCore.pyqtSignal(str, str)  # video_path, metadata_path
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        model: H5Model,
        start_sweep: int,
        end_sweep: int,
        step: int,
        fps: float,
        output_path: str | Path,
        render_callback: Any,  # Callable[[int], QImage] invoked on main thread or pre-captured
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.model = model
        self.start_sweep = int(start_sweep)
        self.end_sweep = int(end_sweep)
        self.step = max(1, int(step))
        self.fps = float(fps)
        self.output_path = Path(output_path).resolve()
        self.render_callback = render_callback
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            sweeps = np.arange(self.start_sweep, self.end_sweep + 1, self.step, dtype=np.int64)
            # Filter sweeps that exist in model
            valid_sweeps = [s for s in sweeps if s in self.model.sweep_id_to_idx]
            total_frames = len(valid_sweeps)
            if total_frames == 0:
                self.error.emit("No valid sweeps selected in the chosen range.")
                return

            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # First frame determines video dimensions
            first_sweep_id = valid_sweeps[0]
            first_img: QtGui.QImage = self.render_callback(first_sweep_id)
            if first_img.isNull():
                self.error.emit("Failed to capture first frame for video.")
                return

            w = first_img.width()
            h = first_img.height()
            w -= w % 2
            h -= h % 2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (w, h), True)
            if not writer.isOpened():
                self.error.emit(f"OpenCV failed to open VideoWriter at {self.output_path}")
                return

            start_time = time.monotonic()

            for idx, sid in enumerate(valid_sweeps):
                if self._is_cancelled:
                    writer.release()
                    self.error.emit("Video export cancelled by user.")
                    return

                qimg: QtGui.QImage = self.render_callback(sid)
                if qimg.width() != w or qimg.height() != h:
                    qimg = qimg.scaled(w, h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)

                # Convert QImage to BGR NumPy array
                qimg = qimg.convertToFormat(QtGui.QImage.Format_RGB888)
                ptr = qimg.bits()
                ptr.setsize(h * w * 3)
                arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                writer.write(bgr)
                self.progress.emit(idx + 1, total_frames, f"Writing sweep {sid} ({idx+1}/{total_frames})...")

            writer.release()
            elapsed = time.monotonic() - start_time

            # Write JSON sidecar metadata
            meta_path = self.output_path.with_suffix(".json")
            metadata = {
                "source_h5": self.model.path.name,
                "video_file": self.output_path.name,
                "start_sweep": int(valid_sweeps[0]),
                "end_sweep": int(valid_sweeps[-1]),
                "sweep_step": self.step,
                "total_frames": total_frames,
                "fps": self.fps,
                "resolution": [w, h],
                "elapsed_seconds": round(elapsed, 2),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "scientific_notes": [
                    "Visualizes CASALS L1B LiDAR acquisition process sweep by sweep.",
                    "Upper view shows schematic aircraft along real instrument trajectory, beams, and accumulating Refh swath.",
                    "Lower view shows synchronized 2D vertical waveform cross-section with official Refh and reconstructed argmax overlays.",
                    "Historical Refh cloud uses deterministic decimation (~200k points display budget).",
                ],
            }
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            self.finished.emit(str(self.output_path), str(meta_path))
        except Exception as exc:
            self.error.emit(f"Video export failed: {exc}")


class VideoExportDialog(QtWidgets.QDialog):
    """Interactive dialog to configure and execute MP4 video export."""

    def __init__(self, model: H5Model, current_sweep_id: int, parent=None) -> None:
        super().__init__(parent=parent)
        self.model = model
        self.setWindowTitle("Export MP4 Sweep Animation")
        self.resize(460, 360)
        self._build_ui(current_sweep_id)

    def _build_ui(self, current_sweep_id: int) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()

        # Start sweep
        self.spin_start = QtWidgets.QSpinBox()
        self.spin_start.setRange(int(self.model.valid_sweep_ids[0]), int(self.model.valid_sweep_ids[-1]))
        self.spin_start.setValue(max(int(self.model.valid_sweep_ids[0]), current_sweep_id - 50))
        self.spin_start.valueChanged.connect(self._update_estimates)
        form.addRow("Start Sweep ID:", self.spin_start)

        # End sweep
        self.spin_end = QtWidgets.QSpinBox()
        self.spin_end.setRange(int(self.model.valid_sweep_ids[0]), int(self.model.valid_sweep_ids[-1]))
        self.spin_end.setValue(min(int(self.model.valid_sweep_ids[-1]), self.spin_start.value() + 150))
        self.spin_end.valueChanged.connect(self._update_estimates)
        form.addRow("End Sweep ID:", self.spin_end)

        # Step
        self.spin_step = QtWidgets.QSpinBox()
        self.spin_step.setRange(1, 100)
        self.spin_step.setValue(1)
        self.spin_step.valueChanged.connect(self._update_estimates)
        form.addRow("Sweep Step:", self.spin_step)

        # FPS
        self.combo_fps = QtWidgets.QComboBox()
        self.combo_fps.addItems(["5", "10", "15", "20", "30"])
        self.combo_fps.setCurrentText("15")
        self.combo_fps.currentTextChanged.connect(self._update_estimates)
        form.addRow("Playback FPS:", self.combo_fps)

        # Output path
        path_layout = QtWidgets.QHBoxLayout()
        default_out = Path("outputs") / "videos" / f"{self.model.path.stem}_animation.mp4"
        self.line_path = QtWidgets.QLineEdit(str(default_out.resolve()))
        self.btn_browse = QtWidgets.QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_path)
        path_layout.addWidget(self.line_path)
        path_layout.addWidget(self.btn_browse)
        form.addRow("Output File:", path_layout)

        layout.addLayout(form)

        # Estimate info label
        self.label_estimate = QtWidgets.QLabel()
        self.label_estimate.setStyleSheet("color: #4da6ff; font-weight: bold; margin: 10px 0;")
        layout.addWidget(self.label_estimate)
        self._update_estimates()

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.label_status = QtWidgets.QLabel()
        self.label_status.setStyleSheet("color: #a0aec0;")
        layout.addWidget(self.label_status)

        # Buttons
        btn_box = QtWidgets.QDialogButtonBox()
        self.btn_export = btn_box.addButton("Export Video", QtWidgets.QDialogButtonBox.AcceptRole)
        self.btn_cancel = btn_box.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.btn_export.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        layout.addWidget(btn_box)

    def _browse_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save MP4 Video", self.line_path.text(), "MP4 Video (*.mp4)"
        )
        if path:
            self.line_path.setText(path)

    def _update_estimates(self) -> None:
        start = self.spin_start.value()
        end = self.spin_end.value()
        step = max(1, self.spin_step.value())
        fps = float(self.combo_fps.currentText() or 15)

        if end < start:
            self.label_estimate.setText("Invalid range: End Sweep is less than Start Sweep.")
            self.btn_export.setEnabled(False)
            return

        frames = (end - start) // step + 1
        duration_s = frames / fps
        self.label_estimate.setText(
            f"Estimated Frames: {frames:,} | Video Duration: {duration_s:.1f} s @ {fps:g} FPS"
        )
        self.btn_export.setEnabled(True)

    def get_settings(self) -> Dict[str, Any]:
        return {
            "start_sweep": self.spin_start.value(),
            "end_sweep": self.spin_end.value(),
            "step": self.spin_step.value(),
            "fps": float(self.combo_fps.currentText()),
            "output_path": self.line_path.text(),
        }

