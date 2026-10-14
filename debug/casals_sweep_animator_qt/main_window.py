"""CASALS 3D sweep and waveform cross-section viewer."""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .frame_controller import FrameController
from .h5_model import H5Model
from .panel2d import Panel2DWidget
from .scene3d import Scene3DWidget
from .video_export import (
    VideoEncoder,
    VideoExportDialog,
    composite_frame_image,
    qimage_to_rgb,
)
from .worker import FrameParams, SweepFrame


class MainWindow(QtWidgets.QMainWindow):
    """Interactive GUI. HDF5 preparation is delegated to FrameController."""

    def __init__(self, initial_h5: Optional[str | Path] = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CASALS Sweep Viewer")
        self.resize(1600, 960)
        self.setMinimumSize(1120, 720)

        self.model: Optional[H5Model] = None
        self.current_frame: Optional[SweepFrame] = None
        self.current_sweep_idx = 0
        self._closing = False
        self._loading = False
        self._export_encoder: Optional[VideoEncoder] = None
        self._export_indices: list[int] = []
        self._export_cursor = 0
        self._export_pending_frame: Optional[np.ndarray] = None
        self._export_old_sweep = 0
        self._export_was_playing = False
        self._export_params = FrameParams()
        self._export_settings: dict = {}
        self._export_camera_state: Optional[tuple] = None
        self._restore_export_state = False
        self._export_active = False
        self._last_frame_monotonic = 0.0
        self._recent_frame_ms: list[float] = []

        self.controller = FrameController(self)
        self.controller.model_loaded.connect(self._on_model_loaded)
        self.controller.frame_ready.connect(self._on_frame_ready)
        self.controller.load_progress.connect(self._on_load_progress)
        self.controller.error_occurred.connect(self._on_error)
        self.controller.position_changed.connect(self._on_position_changed)
        self.controller.playback_changed.connect(self._on_playback_changed)

        self._build_ui()
        self._apply_theme()
        self._set_model_enabled(False)
        if initial_h5 is not None:
            QtCore.QTimer.singleShot(100, lambda: self.load_h5(str(initial_h5)))

    def _build_ui(self) -> None:
        self._build_toolbar()
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.control_scroll = QtWidgets.QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.control_scroll.setFixedWidth(540)
        self.control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(3, 3, 3, 3)
        control_layout.setSpacing(8)
        self._build_playback_controls(control_layout)
        self._build_scene_controls(control_layout)
        self._build_cross_section_controls(control_layout)
        self.control_scroll.setWidget(self.control_panel)
        root.addWidget(self.control_scroll)

        views = QtWidgets.QWidget()
        views_layout = QtWidgets.QVBoxLayout(views)
        views_layout.setContentsMargins(0, 0, 0, 0)
        views_layout.setSpacing(6)
        self.view_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.scene3d = Scene3DWidget()
        self.panel2d = Panel2DWidget()
        self.view_splitter.addWidget(self.scene3d)
        self.view_splitter.addWidget(self.panel2d)
        self.view_splitter.setStretchFactor(0, 6)
        self.view_splitter.setStretchFactor(1, 4)
        self.view_splitter.setSizes([560, 360])
        views_layout.addWidget(self.view_splitter, 1)
        self._build_timeline(views_layout)
        root.addWidget(views, 1)

        self.statusBar().showMessage("Open a CASALS L1B HDF5 file to begin.")
        self._seek_timer = QtCore.QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.timeout.connect(lambda: self._request_sweep(self.slider_timeline.value()))
        self._export_retry_timer = QtCore.QTimer(self)
        self._export_retry_timer.setSingleShot(True)
        self._export_retry_timer.timeout.connect(self._retry_export_enqueue)

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("File")
        toolbar.setMovable(False)
        self.action_open = toolbar.addAction("Open H5")
        self.action_open.triggered.connect(self._open_file_dialog)
        self.action_screenshot = toolbar.addAction("Save PNG")
        self.action_screenshot.triggered.connect(self._save_screenshot)
        self.action_video = toolbar.addAction("Export MP4")
        self.action_video.triggered.connect(self._export_video)
        toolbar.addSeparator()
        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #9fb0c2; padding: 0 8px;")
        toolbar.addWidget(self.file_label)
        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setStyleSheet("color: #7dd3c7; padding: 0 8px;")
        toolbar.addWidget(self.progress_label)
        self.action_cancel_export = toolbar.addAction("Cancel export")
        self.action_cancel_export.triggered.connect(self._cancel_video_export)
        self.action_cancel_export.setVisible(False)

    def _group(self, title: str, parent_layout: QtWidgets.QVBoxLayout) -> QtWidgets.QVBoxLayout:
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(9, 9, 9, 9)
        layout.setSpacing(5)
        parent_layout.addWidget(group)
        return layout

    def _add_combo(self, layout: QtWidgets.QFormLayout, label: str, items: list[tuple[str, object]]) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        for text, value in items:
            combo.addItem(text, value)
        layout.addRow(label, combo)
        return combo

    def _build_playback_controls(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        layout = self._group("Playback", parent_layout)
        row = QtWidgets.QHBoxLayout()
        self.btn_first = QtWidgets.QPushButton("|<")
        self.btn_prev = QtWidgets.QPushButton("<")
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_next = QtWidgets.QPushButton(">")
        self.btn_last = QtWidgets.QPushButton(">|")
        for button in (self.btn_first, self.btn_prev, self.btn_play, self.btn_next, self.btn_last):
            button.setFixedWidth(40)
            row.addWidget(button)
        self.btn_play.setFixedWidth(56)
        layout.addLayout(row)
        form = QtWidgets.QFormLayout()
        self.spin_fps = QtWidgets.QDoubleSpinBox()
        self.spin_fps.setRange(1.0, 30.0)
        self.spin_fps.setDecimals(0)
        self.spin_fps.setSingleStep(1.0)
        self.spin_fps.setValue(10.0)
        self.spin_step = QtWidgets.QSpinBox()
        self.spin_step.setRange(1, 1000)
        self.spin_step.setValue(1)
        form.addRow("Playback FPS", self.spin_fps)
        form.addRow("Sweep step", self.spin_step)
        layout.addLayout(form)
        self.sweep_label = QtWidgets.QLabel("Sweep - / -")
        self.sweep_label.setStyleSheet("font-weight: 600; color: #d9e4ef;")
        layout.addWidget(self.sweep_label)

        self.btn_first.clicked.connect(lambda: self._request_sweep(0))
        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_next.clicked.connect(self._on_next)
        self.btn_last.clicked.connect(lambda: self._request_sweep(self.model.n_valid_sweeps - 1) if self.model else None)
        self.btn_play.clicked.connect(self._toggle_play)
        self.spin_fps.valueChanged.connect(self._update_playback_settings)
        self.spin_step.valueChanged.connect(self._update_playback_settings)

    def _build_scene_controls(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        layout = self._group("3D Scene", parent_layout)
        self.layer_aircraft = QtWidgets.QCheckBox("Aircraft (schematic)")
        self.layer_trajectory = QtWidgets.QCheckBox("Trajectory")
        self.layer_history = QtWidgets.QCheckBox("History Refh")
        self.layer_current = QtWidgets.QCheckBox("Current Refh")
        for checkbox, checked in ((self.layer_aircraft, True), (self.layer_trajectory, True),
                                  (self.layer_history, True), (self.layer_current, True)):
            checkbox.setChecked(checked)
            layout.addWidget(checkbox)
            checkbox.toggled.connect(self._update_layers)

        form = QtWidgets.QFormLayout()
        self.combo_beams = self._add_combo(form, "Refh beams", [
            ("Off", "none"), ("Every 16 tracks", "16"), ("Every 8 tracks", "8"), ("All tracks", "all")
        ])
        self.combo_beams.setCurrentIndex(1)
        self.combo_camera = self._add_combo(form, "Camera", [
            ("Follow sweep", "follow"), ("Overview", "overview"), ("Free camera", "free")
        ])
        self.combo_z_scale = self._add_combo(form, "Z scale (Vertical)", [
            ("0.1x (Max compressed)", "0.1"),
            ("0.2x", "0.2"),
            ("0.25x (Default balanced)", "0.25"),
            ("0.5x", "0.5"),
            ("1.0x (True metric)", "1.0"),
            ("2.0x (Exaggerated)", "2.0"),
        ])
        self.combo_z_scale.setCurrentIndex(2)
        self.combo_point_color = self._add_combo(form, "History color", [
            ("Height", "height"), ("SNR", "snr"), ("Amplitude", "amp"), ("Uniform", "uniform")
        ])
        self.combo_point_cmap = self._add_combo(form, "Height colors", [(n.title(), n) for n in ("viridis", "plasma", "inferno", "turbo")])
        layout.addLayout(form)
        self.btn_reset_camera = QtWidgets.QPushButton("Reset camera (Center on point cloud)")
        layout.addWidget(self.btn_reset_camera)
        self.combo_beams.currentIndexChanged.connect(self._update_layers)
        self.combo_camera.currentIndexChanged.connect(self._on_camera_changed)
        self.combo_z_scale.currentIndexChanged.connect(self._on_z_scale_changed)
        self.combo_point_color.currentIndexChanged.connect(self._on_point_color_changed)
        self.combo_point_cmap.currentIndexChanged.connect(lambda: self.scene3d.set_point_colormap(self.combo_point_cmap.currentData()))
        self.btn_reset_camera.clicked.connect(self._reset_camera)

    def _build_cross_section_controls(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        layout = self._group("2D Cross-Section", parent_layout)
        form = QtWidgets.QFormLayout()
        self.combo_2d_view = self._add_combo(form, "View", [
            ("Waveform Curtain", "curtain"), ("Georeferenced Cross-Section", "georeferenced")
        ])
        self.combo_2d_mode = self._add_combo(form, "Response", [
            ("Net RX (minus bg)", "minus_bg"),
            ("Raw RX amplitude", "raw"),
            ("Asinh net RX", "asinh"),
            ("Standardized RX", "standardized"),
            ("Official Refh points", "refh_only"),
        ])
        self.combo_mapping = self._add_combo(form, "Range mapping", [("H2 (bin centers)", "H2"), ("H1 (endpoint centers)", "H1")])
        self.combo_2d_cmap = self._add_combo(form, "Color map", [(n.title(), n) for n in ("viridis", "plasma", "inferno", "turbo", "gray")])
        self.combo_range_mode = self._add_combo(form, "Color range", [("Stable file range", "stable"), ("Auto per frame", "local")])
        layout.addLayout(form)
        self.layer_refh_overlay = QtWidgets.QCheckBox("Show official Refh points")
        self.layer_refh_overlay.setChecked(True)
        self.layer_argmax_overlay = QtWidgets.QCheckBox("Show reconstructed RX argmax")
        self.layer_argmax_overlay.setChecked(False)
        layout.addWidget(self.layer_refh_overlay)
        layout.addWidget(self.layer_argmax_overlay)

        self.combo_2d_view.currentIndexChanged.connect(self._request_current_frame)
        self.combo_2d_mode.currentIndexChanged.connect(self._request_current_frame)
        self.combo_mapping.currentIndexChanged.connect(self._request_current_frame)
        self.combo_2d_cmap.currentIndexChanged.connect(lambda: self.panel2d.set_colormap(self.combo_2d_cmap.currentData()))
        self.combo_range_mode.currentIndexChanged.connect(self._on_color_range_mode)
        self.layer_refh_overlay.toggled.connect(self._update_overlay_visibility)
        self.layer_argmax_overlay.toggled.connect(self._update_overlay_visibility)

    def _build_timeline(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        bar = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(5, 3, 5, 2)
        layout.setSpacing(8)
        self.slider_timeline = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_timeline.setRange(0, 0)
        self.slider_timeline.setTracking(True)
        self.index_spin = QtWidgets.QSpinBox()
        self.index_spin.setRange(0, 0)
        self.index_spin.setPrefix("Index ")
        layout.addWidget(QtWidgets.QLabel("Timeline"))
        layout.addWidget(self.slider_timeline, 1)
        layout.addWidget(self.index_spin)
        parent_layout.addWidget(bar)
        self.slider_timeline.valueChanged.connect(self._on_slider_changed)
        self.slider_timeline.sliderReleased.connect(lambda: self._request_sweep(self.slider_timeline.value()))
        self.index_spin.valueChanged.connect(self._request_sweep)

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #12171e; color: #d6dee8; font-family: 'Segoe UI', Arial; font-size: 9pt; }
            QToolBar { background: #19212b; border: 0; spacing: 8px; padding: 5px; }
            QGroupBox { font-weight: 600; border: 1px solid #303d4c; border-radius: 6px; margin-top: 8px; padding-top: 8px; background: #171e27; }
            QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; color: #89c2ff; }
            QPushButton { background: #26364a; border: 1px solid #3d526c; border-radius: 4px; padding: 5px 8px; }
            QPushButton:hover { background: #304761; }
            QPushButton:disabled { color: #657383; background: #1b222b; }
            QComboBox, QSpinBox, QDoubleSpinBox { background: #202a36; border: 1px solid #374657; border-radius: 3px; padding: 3px 5px; min-height: 22px; }
            QScrollArea { border: 0; }
            QStatusBar { background: #19212b; color: #9fb0c2; }
            QSplitter::handle { background: #222c38; }
        """)

    def _set_model_enabled(self, enabled: bool) -> None:
        self.control_panel.setEnabled(enabled)
        self.slider_timeline.setEnabled(enabled)
        self.index_spin.setEnabled(enabled)
        self.action_screenshot.setEnabled(enabled)
        self.action_video.setEnabled(enabled)
        self.btn_play.setEnabled(enabled)
        for button in (self.btn_first, self.btn_prev, self.btn_next, self.btn_last):
            button.setEnabled(enabled)

    def _open_file_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CASALS L1B HDF5", "", "HDF5 files (*.h5 *.hdf5);;All files (*)")
        if path:
            self.load_h5(path)

    def load_h5(self, path: str) -> None:
        if self._export_active:
            self._cancel_video_export()
        self._loading = True
        self._set_model_enabled(False)
        self.file_label.setText(Path(path).name)
        self.statusBar().showMessage(f"Opening {path} ...")
        self.controller.load_file(path)

    @QtCore.pyqtSlot(object, int)
    def _on_model_loaded(self, model: H5Model, _generation: int) -> None:
        self.model = model
        self.current_sweep_idx = 0
        self._loading = False
        self.slider_timeline.blockSignals(True)
        self.index_spin.blockSignals(True)
        self.slider_timeline.setRange(0, model.n_valid_sweeps - 1)
        self.index_spin.setRange(0, model.n_valid_sweeps - 1)
        self.slider_timeline.setValue(0)
        self.index_spin.setValue(0)
        self.slider_timeline.blockSignals(False)
        self.index_spin.blockSignals(False)

        history_start = time.perf_counter()
        model_history = model.decimated_enu
        self.scene3d.set_history_data(
            positions=model_history,
            sweep_indices=model.decimated_sweep_index,
            prefix_count=model.decimated_offsets,
            heights=model.decimated_height,
            snr=model.decimated_snr,
            amplitude=model.decimated_amp,
            ranges={"height": (model.height_vmin, model.height_vmax),
                    "snr": (model.snr_vmin, model.snr_vmax),
                    "amp": (model.amp_vmin, model.amp_vmax)},
            trajectory=model.sweep_instrument_enu.astype(np.float32),
        )
        history_upload_ms = (time.perf_counter() - history_start) * 1000.0
        self.panel2d.set_color_ranges({
            "raw": (0.0, max(1.0, model.amp_vmax * 2.0)),
            "minus_bg": (0.0, max(1.0, model.amp_vmax)),
            "asinh": (0.0, 5.0),
            "standardized": (0.0, 10.0),
        })
        self._set_model_enabled(True)
        self.progress_label.clear()
        self.statusBar().showMessage(
            f"Loaded {model.n_valid_sweeps:,} sweeps, {model.n_tracks:,} tracks, "
            f"{model.n_rx_bins:,} RX bins. History: {model.total_decimated_points:,} points "
            f"({history_upload_ms:.0f} ms CPU preparation / buffer upload call)."
        )
        self._request_current_frame()

    @QtCore.pyqtSlot(int, str)
    def _on_load_progress(self, percent: int, message: str) -> None:
        self.progress_label.setText(f"{percent}% - {message}")

    @QtCore.pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self.progress_label.clear()
        if self._export_active:
            self._on_export_failed(f"Video export stopped: {message}")
            return
        self.statusBar().showMessage(message, 15000)
        QtWidgets.QMessageBox.critical(self, "CASALS Sweep Viewer", message)

    def _frame_params(self) -> FrameParams:
        height_range = self.model.height_range if self.model is not None else None
        return FrameParams(
            mode_2d=self.combo_2d_mode.currentData(),
            view_2d=self.combo_2d_view.currentData(),
            mapping_convention=self.combo_mapping.currentData(),
            n_cols=384,
            n_rows=512,
            asinh_scale=100.0,
            fixed_height_range=height_range,
        )

    def _request_current_frame(self, *_args) -> None:
        if self.model is None or self._loading or self._export_active:
            return
        self._request_sweep(self.current_sweep_idx, self._frame_params())

    def _request_sweep(self, index: int, params: Optional[FrameParams] = None) -> None:
        if self.model is None or self._export_active:
            return
        index = max(0, min(int(index), self.model.n_valid_sweeps - 1))
        self._seek_timer.stop()
        self.controller.request_frame(index, params or self._frame_params())

    @QtCore.pyqtSlot(int)
    def _on_position_changed(self, index: int) -> None:
        self.current_sweep_idx = index
        self.slider_timeline.blockSignals(True)
        self.index_spin.blockSignals(True)
        self.slider_timeline.setValue(index)
        self.index_spin.setValue(index)
        self.slider_timeline.blockSignals(False)
        self.index_spin.blockSignals(False)
        if self.model is not None:
            sweep_id = int(self.model.valid_sweep_ids[index])
            self.sweep_label.setText(f"Sweep ID {sweep_id:,}   |   {index + 1:,} / {self.model.n_valid_sweeps:,}")

    def _on_slider_changed(self, index: int) -> None:
        if self.model is None or self._export_active:
            return
        self.current_sweep_idx = int(index)
        self.index_spin.blockSignals(True)
        self.index_spin.setValue(index)
        self.index_spin.blockSignals(False)
        self._seek_timer.start(30)

    @QtCore.pyqtSlot(object)
    def _on_frame_ready(self, frame: SweepFrame) -> None:
        if self._closing:
            return
        self.current_frame = frame
        self.current_sweep_idx = frame.sweep_index
        self.scene3d.update_frame(frame.sweep_data, frame.history_visible_count)
        self.panel2d.update_cross_section(frame.grid)
        self._on_position_changed(frame.sweep_index)
        now = time.monotonic()
        if self._last_frame_monotonic:
            self._recent_frame_ms.append((now - self._last_frame_monotonic) * 1000.0)
            self._recent_frame_ms = self._recent_frame_ms[-30:]
        self._last_frame_monotonic = now
        fps = 1000.0 / float(np.mean(self._recent_frame_ms)) if self._recent_frame_ms else 0.0
        status = (f"Sweep {frame.sweep_data.sweep_id:,} | valid Refh "
                  f"{int(frame.sweep_data.valid_pulse_mask.sum())}/{frame.sweep_data.n_tracks} | "
                  f"waveform block {frame.block_read_ms:.1f} ms | 2D {frame.cross_section_ms:.1f} ms | "
                  f"GUI {fps:.1f} FPS")
        if frame.grid.closure_median_3d_m is not None:
            status += f" | H{frame.grid.mapping_convention} argmax-Refh median {frame.grid.closure_median_3d_m:.3f} m"
        self.statusBar().showMessage(status)
        if self._restore_export_state and frame.sweep_index == self._export_old_sweep:
            self._restore_export_state = False
            if self._export_camera_state is not None:
                mode, center, azimuth, elevation, scale_factor = self._export_camera_state
                self.scene3d.camera_mode = mode
                self.scene3d.camera.center = center
                self.scene3d.camera.azimuth = azimuth
                self.scene3d.camera.elevation = elevation
                self.scene3d.camera.scale_factor = scale_factor
                self.scene3d.canvas.update()
            if self._export_was_playing:
                self.controller.play(self.spin_fps.value(), self.spin_step.value())
        if self._export_active:
            QtCore.QTimer.singleShot(80, self._capture_export_frame)

    def _on_prev(self) -> None:
        self._request_sweep(self.current_sweep_idx - 1)

    def _on_next(self) -> None:
        self._request_sweep(self.current_sweep_idx + 1)

    def _toggle_play(self) -> None:
        if self.controller.is_playing:
            self.controller.pause()
        else:
            self.controller.play(self.spin_fps.value(), self.spin_step.value())

    def _update_playback_settings(self, *_args) -> None:
        self.controller.set_playback(self.spin_fps.value(), self.spin_step.value())

    @QtCore.pyqtSlot(bool)
    def _on_playback_changed(self, playing: bool) -> None:
        self.btn_play.setText("Pause" if playing else "Play")

    def _update_layers(self, *_args) -> None:
        self.scene3d.set_layers(
            aircraft=self.layer_aircraft.isChecked(),
            trajectory=self.layer_trajectory.isChecked(),
            history=self.layer_history.isChecked(),
            current=self.layer_current.isChecked(),
            beams=self.combo_beams.currentData(),
        )

    def _on_camera_changed(self, *_args) -> None:
        self.scene3d.set_camera_mode(self.combo_camera.currentData())

    def _on_z_scale_changed(self, *_args) -> None:
        try:
            val = float(self.combo_z_scale.currentData())
        except (ValueError, TypeError):
            val = 0.25
        self.scene3d.set_z_scale(val)

    def _on_point_color_changed(self, *_args) -> None:
        self.scene3d.set_point_color_mode(self.combo_point_color.currentData())

    def _reset_camera(self) -> None:
        self.scene3d.set_camera_mode(self.combo_camera.currentData())
        self.scene3d.reset_view()

    def _on_color_range_mode(self, *_args) -> None:
        self.panel2d.set_color_range_mode(self.combo_range_mode.currentData())
        if self.current_frame is not None:
            self.panel2d.update_cross_section(self.current_frame.grid)

    def _update_overlay_visibility(self, *_args) -> None:
        self.panel2d.show_refh_overlay = self.layer_refh_overlay.isChecked()
        self.panel2d.show_argmax_overlay = self.layer_argmax_overlay.isChecked()
        if self.current_frame is not None:
            self.panel2d.update_cross_section(self.current_frame.grid)

    def _file_identity(self) -> dict:
        assert self.model is not None
        stat = self.model.path.stat()
        return {
            "path": str(self.model.path),
            "file_name": self.model.path.name,
            "size_bytes": stat.st_size,
            "modified_time_ns": stat.st_mtime_ns,
        }

    def _display_settings(self) -> dict:
        return {
            "response_2d": self.combo_2d_mode.currentData(),
            "view_2d": self.combo_2d_view.currentData(),
            "mapping_convention": self.combo_mapping.currentData(),
            "color_range_mode": self.combo_range_mode.currentData(),
            "color_map_2d": self.combo_2d_cmap.currentData(),
            "history_color_mode": self.combo_point_color.currentData(),
            "history_height_color_map": self.combo_point_cmap.currentData(),
            "beam_density": self.combo_beams.currentData(),
            "aircraft_visible": self.layer_aircraft.isChecked(),
            "trajectory_visible": self.layer_trajectory.isChecked(),
            "history_visible": self.layer_history.isChecked(),
            "current_refh_visible": self.layer_current.isChecked(),
            "argmax_overlay": self.layer_argmax_overlay.isChecked(),
            "refh_overlay": self.layer_refh_overlay.isChecked(),
            "coordinate_system": "local ENU",
            "coordinate_unit": "m",
            "axis_scale_xyz": [1.0, 1.0, 1.0],
            "axis_origin_enu_m": self.scene3d._axis_origin_enu.tolist(),
            "axis_reference_length_m": float(self.scene3d._axis_reference_m),
            "camera_mode": self.scene3d.camera_mode,
            "aircraft_orientation": "schematic; heading derived from horizontal trajectory",
            "refh_height_datum": "WGS84 ellipsoidal height (EPSG:4979)",
            "beam_bin_geometry": "research visualization using H1/H2 range-window mapping",
        }

    def _display_metadata(self, sweep: SweepFrame) -> dict:
        camera = self.scene3d.camera
        return {
            "input_file": self._file_identity(),
            "sweep_id": sweep.sweep_data.sweep_id,
            "sweep_index": sweep.sweep_index,
            "total_sweeps": sweep.sweep_data.total_sweeps,
            "track_count": sweep.sweep_data.n_tracks,
            "valid_refh_count": int(sweep.sweep_data.valid_pulse_mask.sum()),
            **self._display_settings(),
            "camera_center": list(camera.center),
            "camera_azimuth_deg": float(camera.azimuth),
            "camera_elevation_deg": float(camera.elevation),
            "camera_scale_factor": float(camera.scale_factor),
        }

    def _save_screenshot(self) -> None:
        if self.model is None or self.current_frame is None:
            return
        default = Path("outputs") / "casals_sweep_animator_qt" / "screenshots" / f"{self.model.path.stem}_sweep_{self.current_frame.sweep_data.sweep_id}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save composed screenshot", str(default.resolve()), "PNG image (*.png)")
        if not path:
            return
        try:
            image = self._composed_image(self.current_frame)
            output = Path(path).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            if not image.save(str(output), "PNG"):
                raise OSError(f"Could not write screenshot to {output}")
            output.with_suffix(".json").write_text(json.dumps({
                **self._display_metadata(self.current_frame),
                "image_file": output.name,
                "output_dimensions": [image.width(), image.height()],
            }, indent=2), encoding="utf-8")
            self.statusBar().showMessage(f"Saved {output}", 8000)
        except Exception as exc:
            self._show_error(f"Screenshot failed: {exc}")

    def _composed_image(self, frame: SweepFrame) -> QtGui.QImage:
        scene_image = self.scene3d.capture_framebuffer()
        panel_image = self.panel2d.grab().toImage()
        return composite_frame_image(
            scene_image,
            panel_image,
            frame.sweep_data,
            self.model,
            frame.grid.mapping_convention,
        )

    def _export_video(self) -> None:
        if self.model is None or self.current_frame is None or self._export_active:
            return
        dialog = VideoExportDialog(self.model, self.current_frame.sweep_data.sweep_id, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        settings = dialog.get_settings()
        ids = self.model.valid_sweep_ids
        self._export_indices = [
            i for i, sid in enumerate(ids)
            if settings["start_sweep"] <= int(sid) <= settings["end_sweep"]
            and (int(sid) - settings["start_sweep"]) % settings["step"] == 0
        ]
        if not self._export_indices:
            self._show_error("The selected range contains no valid sweeps.")
            return
        self._export_old_sweep = self.current_sweep_idx
        self._export_was_playing = self.controller.is_playing
        camera = self.scene3d.camera
        self._export_camera_state = (
            self.scene3d.camera_mode,
            tuple(camera.center),
            float(camera.azimuth),
            float(camera.elevation),
            float(camera.scale_factor),
        )
        self.controller.pause()
        self._export_params = self._frame_params()
        self._export_settings = settings
        metadata = {
            "input_file": self._file_identity(),
            "start_sweep_id": int(ids[self._export_indices[0]]),
            "end_sweep_id": int(ids[self._export_indices[-1]]),
            "step_sweep_id": settings["step"],
            "sweep_ids": [int(ids[i]) for i in self._export_indices],
            "fps": settings["fps"],
            "output_dimensions": [1920, 1080],
            "display_settings": self._display_settings(),
            "camera_behavior": "Follow mode recenters on instrument and current swath; camera movement is included in each rendered frame." if self.scene3d.camera_mode == "follow" else "Camera mode remains fixed during export.",
        }
        self._export_encoder = VideoEncoder(
            settings["output_path"], settings["fps"], len(self._export_indices), metadata, self
        )
        self._export_encoder.progress.connect(self._on_export_progress)
        self._export_encoder.export_done.connect(self._on_export_done)
        self._export_encoder.failed.connect(self._on_export_failed)
        self._export_encoder.cancelled.connect(self._on_export_cancelled)
        self._export_active = True
        self._export_cursor = 0
        self._export_pending_frame = None
        self.control_panel.setEnabled(False)
        self.slider_timeline.setEnabled(False)
        self.index_spin.setEnabled(False)
        self.action_open.setEnabled(False)
        self.action_screenshot.setEnabled(False)
        self.action_video.setEnabled(False)
        self.action_cancel_export.setVisible(True)
        self.progress_label.setText(f"Starting export: {len(self._export_indices)} frames")
        self._export_encoder.start()
        self.controller.request_frame(self._export_indices[0], self._export_params)

    def _capture_export_frame(self) -> None:
        if not self._export_active or self._export_pending_frame is not None or self.current_frame is None:
            return
        try:
            if self._export_cursor == 0 and self._export_encoder is not None:
                camera = self.scene3d.camera
                self._export_encoder.metadata["camera_start_sweep_id"] = self.current_frame.sweep_data.sweep_id
                self._export_encoder.metadata["camera_start_state"] = {
                    "center": list(camera.center),
                    "azimuth_deg": float(camera.azimuth),
                    "elevation_deg": float(camera.elevation),
                    "scale_factor": float(camera.scale_factor),
                }
            image = self._composed_image(self.current_frame)
            self._export_pending_frame = qimage_to_rgb(image, output_size=(1920, 1080))
            self._retry_export_enqueue()
        except Exception as exc:
            self._on_export_failed(f"Could not capture export frame: {exc}")

    def _retry_export_enqueue(self) -> None:
        if not self._export_active or self._export_pending_frame is None or self._export_encoder is None:
            return
        if not self._export_encoder.offer_frame(self._export_pending_frame):
            self._export_retry_timer.start(20)
            return
        self._export_pending_frame = None
        self._export_cursor += 1
        if self._export_cursor >= len(self._export_indices):
            self._export_encoder.finish_input()
            self.progress_label.setText("Encoding and validating MP4...")
            return
        self.controller.request_frame(self._export_indices[self._export_cursor], self._export_params)

    @QtCore.pyqtSlot(int, int, str)
    def _on_export_progress(self, current: int, total: int, message: str) -> None:
        self.progress_label.setText(f"{message} ({100 * current // max(1, total)}%)")

    @QtCore.pyqtSlot(str, str, int)
    def _on_export_done(self, video_path: str, metadata_path: str, decoded: int) -> None:
        self._finish_export_ui()
        self.statusBar().showMessage(f"Export verified: {decoded} frames - {video_path} (metadata: {metadata_path})", 15000)

    @QtCore.pyqtSlot(str)
    def _on_export_failed(self, message: str) -> None:
        self._finish_export_ui()
        self._show_error(message)

    @QtCore.pyqtSlot(int)
    def _on_export_cancelled(self, count: int) -> None:
        self._finish_export_ui(restore=True)
        self.statusBar().showMessage(f"Video export cancelled after {count} encoded frames.", 10000)

    def _cancel_video_export(self) -> None:
        if self._export_encoder is not None:
            self._export_encoder.cancel()
        self._export_pending_frame = None
        self._finish_export_ui(restore=False)
        self.statusBar().showMessage("Cancelling video export...", 8000)

    def _finish_export_ui(self, *, restore: bool = True) -> None:
        self._export_retry_timer.stop()
        self._export_active = False
        self.control_panel.setEnabled(self.model is not None)
        self.slider_timeline.setEnabled(self.model is not None)
        self.index_spin.setEnabled(self.model is not None)
        self.action_open.setEnabled(True)
        self.action_screenshot.setEnabled(self.model is not None)
        self.action_video.setEnabled(self.model is not None)
        self.action_cancel_export.setVisible(False)
        self.progress_label.clear()
        encoder, self._export_encoder = self._export_encoder, None
        if encoder is not None and encoder.isRunning():
            encoder.cancel()
            encoder.wait(3000)
        if restore and self.model is not None:
            self._restore_export_state = True
            self._request_sweep(self._export_old_sweep)

    def _show_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 15000)
        QtWidgets.QMessageBox.critical(self, "CASALS Sweep Viewer", message)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._closing = True
        if self._export_encoder is not None:
            self._export_encoder.cancel()
            self._export_encoder.wait(5000)
        self.controller.shutdown()
        event.accept()

