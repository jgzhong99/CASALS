"""Main Qt application window and playback controller for CASALS 3D Sweep Animator.

Scientific boundaries & requirements:
- Synchronized 3D OpenGL visualization and 2D vertical waveform cross-section.
- Responsive, non-blocking asynchronous playback with pending-latest request management.
- Dynamic camera controls (Overview vs Follow Aircraft).
- Clear scientific disclosures for schematic aircraft orientation and research bin mapping.
- PNG screenshot and MP4 video export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import time
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .cross_section import CrossSectionGrid
from .h5_model import H5Model, SweepData
from .panel2d import Panel2DWidget
from .scene3d import Scene3DWidget
from .video_export import (
    VideoExportDialog,
    VideoExportWorker,
    capture_widget_pixmap,
    composite_screenshot,
)
from .worker import FrameParams, H5Worker, SweepFrame


class MainWindow(QtWidgets.QMainWindow):
    """Main application window for CASALS 3D Sweep Animator."""

    def __init__(self, initial_h5: Optional[str | Path] = None, parent=None) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle("CASALS 3D Sweep Animator & Waveform Cross-Section Viewer")
        self.resize(1600, 960)

        self.model: Optional[H5Model] = None
        self.current_sweep_idx = 0
        self.current_frame: Optional[SweepFrame] = None
        self._request_id = 0
        self._is_playing = False
        self._step_size = 1
        self._requested_fps = 5.0

        # Frame timing diagnostics
        self._last_frame_time = time.monotonic()
        self._fps_history = []

        # Setup Thread and Worker
        self._init_worker()

        # Build UI layout
        self._build_ui()

        # Setup Playback Timer
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        # Apply dark scientific stylesheet
        self._apply_theme()

        # Load initial file if provided
        if initial_h5 is not None:
            QtCore.QTimer.singleShot(100, lambda: self.load_h5(str(initial_h5)))

    def _init_worker(self) -> None:
        self.worker_thread = QtCore.QThread()
        self.worker = H5Worker()
        self.worker.moveToThread(self.worker_thread)

        # Worker signals
        self.worker.file_loaded.connect(self._on_file_loaded)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.load_progress.connect(self._on_load_progress)
        self.worker.error_occurred.connect(self._on_worker_error)

        self.worker_thread.start()

    def _apply_theme(self) -> None:
        style = """
        QMainWindow {
            background-color: #12161c;
            color: #d1d7e0;
        }
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 9pt;
            color: #d1d7e0;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #283241;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 10px;
            background-color: #181e27;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            color: #60a5fa;
        }
        QPushButton {
            background-color: #243042;
            border: 1px solid #3b4d66;
            border-radius: 3px;
            padding: 4px 8px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #2d3d54;
            border-color: #60a5fa;
        }
        QPushButton:pressed {
            background-color: #1a2330;
        }
        QPushButton:disabled {
            background-color: #161c24;
            color: #556274;
            border-color: #232b36;
        }
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
            background-color: #1a222c;
            border: 1px solid #2f3e50;
            border-radius: 3px;
            padding: 2px 4px;
        }
        QComboBox:hover, QSpinBox:hover {
            border-color: #4b6b94;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #243040;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: #2563eb;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #60a5fa;
            width: 14px;
            margin-top: -4px;
            margin-bottom: -4px;
            border-radius: 7px;
        }
        QStatusBar {
            background-color: #0f1318;
            border-top: 1px solid #222a36;
            color: #94a3b8;
        }
        QScrollBar:vertical {
            background: #141a22;
            width: 10px;
        }
        QScrollBar::handle:vertical {
            background: #283547;
            border-radius: 5px;
        }
        """
        self.setStyleSheet(style)

    def _build_ui(self) -> None:
        # Central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # 1. Left Control Panel
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(310)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        control_widget = QtWidgets.QWidget()
        self.control_layout = QtWidgets.QVBoxLayout(control_widget)
        self.control_layout.setContentsMargins(4, 4, 8, 4)
        self.control_layout.setSpacing(8)

        self._build_file_group()
        self._build_playback_group()
        self._build_3d_group()
        self._build_2d_group()
        self._build_export_group()

        self.control_layout.addStretch()
        scroll_area.setWidget(control_widget)
        main_layout.addWidget(scroll_area)

        # 2. Right Visualization Splitter (3D + 2D)
        self.view_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.view_splitter.setChildrenCollapsible(False)

        # 3D View
        self.scene3d = Scene3DWidget(parent=self)
        self.view_splitter.addWidget(self.scene3d)

        # 2D Panel
        self.panel2d = Panel2DWidget(parent=self)
        self.view_splitter.addWidget(self.panel2d)

        # Initial splitter ratios: ~65% for 3D, ~35% for 2D
        self.view_splitter.setSizes([600, 320])
        main_layout.addWidget(self.view_splitter, 1)

        # 3. Status Bar
        self.status_bar = self.statusBar()
        self.status_label_main = QtWidgets.QLabel("Ready. Open a CASALS L1B HDF5 file to begin.")
        self.status_label_closure = QtWidgets.QLabel("")
        self.status_label_fps = QtWidgets.QLabel("FPS: --")
        self.status_label_closure.setStyleSheet("color: #38bdf8; margin-right: 15px;")
        self.status_label_fps.setStyleSheet("color: #a3e635; margin-right: 15px;")

        self.status_bar.addWidget(self.status_label_main, 1)
        self.status_bar.addPermanentWidget(self.status_label_closure)
        self.status_bar.addPermanentWidget(self.status_label_fps)

    def _build_file_group(self) -> None:
        group = QtWidgets.QGroupBox("H5 File")
        layout = QtWidgets.QVBoxLayout(group)

        self.btn_open_file = QtWidgets.QPushButton("Open CASALS H5...")
        self.btn_open_file.clicked.connect(self._browse_h5)
        layout.addWidget(self.btn_open_file)

        self.lbl_filename = QtWidgets.QLabel("No file loaded")
        self.lbl_filename.setWordWrap(True)
        self.lbl_filename.setStyleSheet("color: #94a3b8; font-size: 8pt;")
        layout.addWidget(self.lbl_filename)

        self.lbl_summary = QtWidgets.QLabel("Pulses: -- | Sweeps: -- | Tracks: --")
        self.lbl_summary.setStyleSheet("color: #cbd5e1; font-size: 8pt;")
        layout.addWidget(self.lbl_summary)

        self.control_layout.addWidget(group)

    def _build_playback_group(self) -> None:
        group = QtWidgets.QGroupBox("Playback & Timeline")
        layout = QtWidgets.QVBoxLayout(group)

        # Play / Pause / Step buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_restart = QtWidgets.QPushButton("⏮")
        self.btn_prev = QtWidgets.QPushButton("◀")
        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_next = QtWidgets.QPushButton("▶")
        self.btn_restart.clicked.connect(self._on_restart)
        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_next.clicked.connect(self._on_next)

        btn_layout.addWidget(self.btn_restart)
        btn_layout.addWidget(self.btn_prev)
        btn_layout.addWidget(self.btn_play, 2)
        btn_layout.addWidget(self.btn_next)
        layout.addLayout(btn_layout)

        # Timeline Slider
        slider_layout = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.slider)
        layout.addLayout(slider_layout)

        # Current Sweep ID & Jump
        jump_layout = QtWidgets.QHBoxLayout()
        jump_layout.addWidget(QtWidgets.QLabel("Sweep:"))
        self.spin_sweep = QtWidgets.QSpinBox()
        self.spin_sweep.setEnabled(False)
        self.spin_sweep.valueChanged.connect(self._on_spin_sweep_changed)
        jump_layout.addWidget(self.spin_sweep)

        self.lbl_sweep_total = QtWidgets.QLabel("/ --")
        jump_layout.addWidget(self.lbl_sweep_total)
        layout.addLayout(jump_layout)

        # FPS & Step
        params_layout = QtWidgets.QHBoxLayout()
        params_layout.addWidget(QtWidgets.QLabel("FPS:"))
        self.combo_fps = QtWidgets.QComboBox()
        self.combo_fps.addItems(["1", "2", "5", "10", "20", "30"])
        self.combo_fps.setCurrentText("5")
        self.combo_fps.currentTextChanged.connect(self._on_fps_changed)
        params_layout.addWidget(self.combo_fps)

        params_layout.addWidget(QtWidgets.QLabel("Step:"))
        self.combo_step = QtWidgets.QComboBox()
        self.combo_step.addItems(["1", "2", "5", "10", "20"])
        self.combo_step.setCurrentText("1")
        self.combo_step.currentTextChanged.connect(self._on_step_changed)
        params_layout.addWidget(self.combo_step)
        layout.addLayout(params_layout)

        self.control_layout.addWidget(group)

    def _build_3d_group(self) -> None:
        group = QtWidgets.QGroupBox("3D Scene Options")
        layout = QtWidgets.QVBoxLayout(group)

        # Display Toggles
        toggles_grid = QtWidgets.QGridLayout()
        self.chk_aircraft = QtWidgets.QCheckBox("Aircraft")
        self.chk_aircraft.setChecked(True)
        self.chk_aircraft.toggled.connect(self._on_3d_param_changed)

        self.chk_trajectory = QtWidgets.QCheckBox("Trajectory")
        self.chk_trajectory.setChecked(True)
        self.chk_trajectory.toggled.connect(self._on_3d_param_changed)

        self.chk_history = QtWidgets.QCheckBox("History Refh")
        self.chk_history.setChecked(True)
        self.chk_history.toggled.connect(self._on_3d_param_changed)

        self.chk_current = QtWidgets.QCheckBox("Current Refh")
        self.chk_current.setChecked(True)
        self.chk_current.toggled.connect(self._on_3d_param_changed)

        toggles_grid.addWidget(self.chk_aircraft, 0, 0)
        toggles_grid.addWidget(self.chk_trajectory, 0, 1)
        toggles_grid.addWidget(self.chk_history, 1, 0)
        toggles_grid.addWidget(self.chk_current, 1, 1)
        layout.addLayout(toggles_grid)

        # Beams
        beam_layout = QtWidgets.QHBoxLayout()
        beam_layout.addWidget(QtWidgets.QLabel("Beams:"))
        self.combo_beams = QtWidgets.QComboBox()
        self.combo_beams.addItems(["None", "16 tracks", "8 tracks", "All tracks"])
        self.combo_beams.setCurrentText("16 tracks")
        self.combo_beams.currentTextChanged.connect(self._on_3d_param_changed)
        beam_layout.addWidget(self.combo_beams)
        layout.addLayout(beam_layout)

        # Vertical Scale
        vscale_layout = QtWidgets.QHBoxLayout()
        vscale_layout.addWidget(QtWidgets.QLabel("Z-Scale:"))
        self.combo_vscale = QtWidgets.QComboBox()
        self.combo_vscale.addItems(["1.0 (True)", "0.1", "0.05", "0.025"])
        self.combo_vscale.setCurrentText("0.05")
        self.combo_vscale.currentTextChanged.connect(self._on_vscale_changed)
        vscale_layout.addWidget(self.combo_vscale)
        layout.addLayout(vscale_layout)

        # Camera Mode & Reset View
        cam_layout = QtWidgets.QHBoxLayout()
        self.combo_cam = QtWidgets.QComboBox()
        self.combo_cam.addItems(["Overview", "Follow Aircraft"])
        self.combo_cam.currentTextChanged.connect(self._on_cam_changed)
        cam_layout.addWidget(self.combo_cam)

        self.btn_reset_cam = QtWidgets.QPushButton("Reset View")
        self.btn_reset_cam.clicked.connect(self._on_reset_cam)
        cam_layout.addWidget(self.btn_reset_cam)
        layout.addLayout(cam_layout)

        self.control_layout.addWidget(group)

    def _build_2d_group(self) -> None:
        group = QtWidgets.QGroupBox("2D Cross-Section Options")
        layout = QtWidgets.QVBoxLayout(group)

        # Mode
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        self.combo_2d_mode = QtWidgets.QComboBox()
        self.combo_2d_mode.addItems(
            [
                "Minus bg_mean (Net RX)",
                "Raw RX amplitude",
                "Asinh-scaled Net RX",
                "Standardized RX (σ)",
                "Refh Spatial Section",
            ]
        )
        self.combo_2d_mode.currentTextChanged.connect(self._on_2d_param_changed)
        mode_layout.addWidget(self.combo_2d_mode)
        layout.addLayout(mode_layout)

        # X-Axis & Mapping Convention
        geom_layout = QtWidgets.QHBoxLayout()
        geom_layout.addWidget(QtWidgets.QLabel("X-Axis:"))
        self.combo_xaxis = QtWidgets.QComboBox()
        self.combo_xaxis.addItems(["Meters", "Track Index"])
        self.combo_xaxis.currentTextChanged.connect(self._on_2d_param_changed)
        geom_layout.addWidget(self.combo_xaxis)

        geom_layout.addWidget(QtWidgets.QLabel("Map:"))
        self.combo_mapping = QtWidgets.QComboBox()
        self.combo_mapping.addItems(["H2", "H1"])
        self.combo_mapping.currentTextChanged.connect(self._on_2d_param_changed)
        geom_layout.addWidget(self.combo_mapping)
        layout.addLayout(geom_layout)

        # Colormap & Range
        cmap_layout = QtWidgets.QHBoxLayout()
        cmap_layout.addWidget(QtWidgets.QLabel("Color:"))
        self.combo_cmap = QtWidgets.QComboBox()
        self.combo_cmap.addItems(["turbo", "viridis", "inferno", "gray"])
        self.combo_cmap.currentTextChanged.connect(self._on_cmap_changed)
        cmap_layout.addWidget(self.combo_cmap)

        self.combo_crange = QtWidgets.QComboBox()
        self.combo_crange.addItems(["Fixed Range", "Auto Frame"])
        self.combo_crange.currentTextChanged.connect(self._on_2d_param_changed)
        cmap_layout.addWidget(self.combo_crange)
        layout.addLayout(cmap_layout)

        # Overlays
        chk_layout = QtWidgets.QHBoxLayout()
        self.chk_overlay_refh = QtWidgets.QCheckBox("Refh Overlay")
        self.chk_overlay_refh.setChecked(True)
        self.chk_overlay_refh.toggled.connect(self._on_2d_param_changed)

        self.chk_overlay_argmax = QtWidgets.QCheckBox("Argmax Overlay")
        self.chk_overlay_argmax.setChecked(True)
        self.chk_overlay_argmax.toggled.connect(self._on_2d_param_changed)

        chk_layout.addWidget(self.chk_overlay_refh)
        chk_layout.addWidget(self.chk_overlay_argmax)
        layout.addLayout(chk_layout)

        self.control_layout.addWidget(group)

    def _build_export_group(self) -> None:
        group = QtWidgets.QGroupBox("Export")
        layout = QtWidgets.QVBoxLayout(group)

        self.btn_screenshot = QtWidgets.QPushButton("Save Screenshot (PNG)...")
        self.btn_screenshot.clicked.connect(self._on_save_screenshot)
        layout.addWidget(self.btn_screenshot)

        self.btn_export_mp4 = QtWidgets.QPushButton("Export MP4 Video...")
        self.btn_export_mp4.clicked.connect(self._on_export_mp4)
        layout.addWidget(self.btn_export_mp4)

        self.control_layout.addWidget(group)

    # -------------------------------------------------------------------------
    # Actions & Slots
    # -------------------------------------------------------------------------

    def _browse_h5(self) -> None:
        default_dir = Path("data/raw/casals_l1b")
        if not default_dir.exists():
            default_dir = Path(".")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open CASALS L1B HDF5", str(default_dir.resolve()), "HDF5 Files (*.h5 *.hdf5)"
        )
        if path:
            self.load_h5(path)

    def load_h5(self, path: str) -> None:
        self._pause()
        self.lbl_filename.setText(Path(path).name)
        self.status_label_main.setText(f"Loading {Path(path).name} in background...")
        self.btn_open_file.setEnabled(False)
        self.slider.setEnabled(False)
        self.spin_sweep.setEnabled(False)

        # Worker does all loading in background thread
        QtCore.QMetaObject.invokeMethod(self.worker, "load_file", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, path))

    @QtCore.pyqtSlot(int, str)
    def _on_load_progress(self, percent: int, msg: str) -> None:
        self.status_label_main.setText(f"Loading ({percent}%): {msg}")

    @QtCore.pyqtSlot(object)
    def _on_file_loaded(self, model: H5Model) -> None:
        self.model = model
        self.btn_open_file.setEnabled(True)
        self.lbl_filename.setText(model.path.name)
        self.lbl_summary.setText(
            f"Pulses: {model.n_records:,} | Sweeps: {model.n_valid_sweeps:,} | Tracks: {model.n_tracks}"
        )

        # Setup slider and controls
        self.slider.setRange(0, model.n_valid_sweeps - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)

        self.spin_sweep.setRange(int(model.valid_sweep_ids[0]), int(model.valid_sweep_ids[-1]))
        self.spin_sweep.setValue(int(model.valid_sweep_ids[0]))
        self.spin_sweep.setEnabled(True)

        self.lbl_sweep_total.setText(f"/ {int(model.valid_sweep_ids[-1])}")

        # Set stable color range for 3D and 2D
        self.scene3d.set_color_ranges(model.height_vmin, model.height_vmax)
        self.panel2d.set_fixed_range(0.0, 500.0)

        # Reset camera to view initial scene
        inst0 = model.sweep_instrument_enu[0]
        self.scene3d.reset_view(center=inst0, distance=1800.0)

        # Request initial sweep
        self.current_sweep_idx = 0
        self._request_current_frame()

    @QtCore.pyqtSlot(str)
    def _on_worker_error(self, err_msg: str) -> None:
        self.btn_open_file.setEnabled(True)
        self.status_label_main.setText(f"Error: {err_msg}")
        QtWidgets.QMessageBox.critical(self, "Error", err_msg)

    def _get_frame_params(self) -> FrameParams:
        mode_text = self.combo_2d_mode.currentText()
        if "Minus" in mode_text:
            m2d = "minus_bg"
        elif "Raw" in mode_text:
            m2d = "raw"
        elif "Asinh" in mode_text:
            m2d = "asinh"
        elif "Standardized" in mode_text:
            m2d = "standardized"
        else:
            m2d = "refh_only"

        x_mode = "meters" if self.combo_xaxis.currentText() == "Meters" else "track"
        mapping = self.combo_mapping.currentText()

        return FrameParams(
            mode_2d=m2d,
            x_axis_mode=x_mode,
            mapping_convention=mapping,
            point_color_mode=self.scene3d.point_color_mode,
            n_cols=384,
            n_rows=512,
            asinh_scale=100.0,
        )

    def _request_current_frame(self) -> None:
        if self.model is None:
            return
        self._request_id += 1
        params = self._get_frame_params()
        QtCore.QMetaObject.invokeMethod(
            self.worker,
            "request_sweep",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, self.current_sweep_idx),
            QtCore.Q_ARG(object, params),
            QtCore.Q_ARG(int, self._request_id),
        )

    @QtCore.pyqtSlot(object)
    def _on_frame_ready(self, frame: SweepFrame) -> None:
        # Check if frame matches current sweep index
        if frame.sweep_data.sweep_index != self.current_sweep_idx:
            return

        self.current_frame = frame
        sw = frame.sweep_data

        # Measure FPS
        now = time.monotonic()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            actual_fps = 1.0 / dt
            self._fps_history.append(actual_fps)
            if len(self._fps_history) > 10:
                self._fps_history.pop(0)
            avg_fps = sum(self._fps_history) / len(self._fps_history)
            self.status_label_fps.setText(f"Actual: {avg_fps:.1f} FPS | Req: {self._requested_fps:g} FPS")

        # 1. Update 3D Scene
        density_text = self.combo_beams.currentText()
        if "16" in density_text:
            self.scene3d.beam_density = "16"
        elif "8" in density_text:
            self.scene3d.beam_density = "8"
        elif "All" in density_text:
            self.scene3d.beam_density = "all"
        else:
            self.scene3d.beam_density = "none"

        self.scene3d.show_aircraft = self.chk_aircraft.isChecked()
        self.scene3d.show_trajectory = self.chk_trajectory.isChecked()
        self.scene3d.show_history_refh = self.chk_history.isChecked()
        self.scene3d.show_current_refh = self.chk_current.isChecked()
        self.scene3d.show_beams = self.scene3d.beam_density != "none"

        self.scene3d.update_frame(
            sweep=sw,
            history_enu=frame.history_enu,
            history_values=frame.history_values,
            all_trajectory_enu=frame.all_trajectory_enu,
        )

        # 2. Update 2D Panel
        self.panel2d.show_refh_overlay = self.chk_overlay_refh.isChecked()
        self.panel2d.show_argmax_overlay = self.chk_overlay_argmax.isChecked()
        self.panel2d.color_range_mode = "fixed" if self.combo_crange.currentText() == "Fixed Range" else "auto"
        self.panel2d.update_cross_section(frame.grid)

        # 3. Update Status Bar
        valid_count = int(np.sum(sw.valid_pulse_mask))
        snr_median = float(np.nanmedian(sw.refh_snr)) if sw.refh_snr is not None and valid_count > 0 else np.nan
        h_min = float(np.nanmin(sw.refh_height[sw.valid_pulse_mask])) if valid_count > 0 else np.nan
        h_max = float(np.nanmax(sw.refh_height[sw.valid_pulse_mask])) if valid_count > 0 else np.nan

        alt = float(sw.pulse_instrument_enu[0, 2]) if len(sw.pulse_instrument_enu) else np.nan
        status_text = (
            f"Sweep: {sw.sweep_id} ({sw.sweep_index + 1}/{sw.total_sweeps}) | "
            f"Valid Refh: {valid_count}/{sw.n_tracks} | "
            f"Median SNR: {snr_median:.1f} | "
            f"Refh Height: [{h_min:.1f}, {h_max:.1f}] m | "
            f"Aircraft Orientation: schematic, heading {sw.heading_deg:.1f}°"
        )
        self.status_label_main.setText(status_text)

        if frame.grid.closure_median_3d_m is not None:
            c3d = frame.grid.closure_median_3d_m
            cv = frame.grid.closure_median_vertical_m
            self.status_label_closure.setText(f"Closure: 3D {c3d*1000:.1f} mm, V {cv*1000:.1f} mm")
        else:
            self.status_label_closure.setText("")

    # -------------------------------------------------------------------------
    # Playback Controls
    # -------------------------------------------------------------------------

    def _toggle_play(self) -> None:
        if self._is_playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        if self.model is None:
            return
        self._is_playing = True
        self.btn_play.setText("⏸ Pause")
        interval = max(20, int(round(1000.0 / self._requested_fps)))
        self._play_timer.start(interval)

    def _pause(self) -> None:
        self._is_playing = False
        self.btn_play.setText("▶ Play")
        self._play_timer.stop()

    def _on_play_tick(self) -> None:
        if self.model is None:
            return
        next_idx = self.current_sweep_idx + self._step_size
        if next_idx >= self.model.n_valid_sweeps:
            next_idx = 0  # Loop back
        self.current_sweep_idx = next_idx
        self.slider.blockSignals(True)
        self.spin_sweep.blockSignals(True)
        self.slider.setValue(next_idx)
        self.spin_sweep.setValue(int(self.model.valid_sweep_ids[next_idx]))
        self.slider.blockSignals(False)
        self.spin_sweep.blockSignals(False)
        self._request_current_frame()

    def _on_slider_changed(self, value: int) -> None:
        if self.model is None:
            return
        self.current_sweep_idx = value
        self.spin_sweep.blockSignals(True)
        self.spin_sweep.setValue(int(self.model.valid_sweep_ids[value]))
        self.spin_sweep.blockSignals(False)
        self._request_current_frame()

    def _on_spin_sweep_changed(self, value: int) -> None:
        if self.model is None:
            return
        if value in self.model.sweep_id_to_idx:
            idx = self.model.sweep_id_to_idx[value]
            self.current_sweep_idx = idx
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
            self._request_current_frame()

    def _on_restart(self) -> None:
        if self.model is None:
            return
        self._pause()
        self.current_sweep_idx = 0
        self.slider.setValue(0)

    def _on_prev(self) -> None:
        if self.model is None:
            return
        self._pause()
        self.current_sweep_idx = max(0, self.current_sweep_idx - self._step_size)
        self.slider.setValue(self.current_sweep_idx)

    def _on_next(self) -> None:
        if self.model is None:
            return
        self._pause()
        self.current_sweep_idx = min(self.model.n_valid_sweeps - 1, self.current_sweep_idx + self._step_size)
        self.slider.setValue(self.current_sweep_idx)

    def _on_fps_changed(self, text: str) -> None:
        try:
            self._requested_fps = float(text)
            if self._is_playing:
                interval = max(20, int(round(1000.0 / self._requested_fps)))
                self._play_timer.setInterval(interval)
        except Exception:
            pass

    def _on_step_changed(self, text: str) -> None:
        try:
            self._step_size = max(1, int(text))
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Visualization Param Changes
    # -------------------------------------------------------------------------

    def _on_vscale_changed(self, text: str) -> None:
        scale_val = 0.05
        if "1.0" in text:
            scale_val = 1.0
        elif "0.1" in text:
            scale_val = 0.1
        elif "0.05" in text:
            scale_val = 0.05
        elif "0.025" in text:
            scale_val = 0.025
        self.scene3d.set_vertical_scale(scale_val)
        if self.current_frame is not None:
            self._on_frame_ready(self.current_frame)

    def _on_cam_changed(self, text: str) -> None:
        self.scene3d.camera_mode = "follow" if "Follow" in text else "overview"
        if self.current_frame is not None:
            self._on_frame_ready(self.current_frame)

    def _on_reset_cam(self) -> None:
        if self.model is not None:
            inst = self.model.sweep_instrument_enu[self.current_sweep_idx]
            self.scene3d.reset_view(center=inst, distance=1800.0)

    def _on_cmap_changed(self, text: str) -> None:
        self.scene3d.point_colormap = text
        self.panel2d.set_colormap(text)
        if self.current_frame is not None:
            self._on_frame_ready(self.current_frame)

    def _on_3d_param_changed(self) -> None:
        if self.current_frame is not None:
            self._on_frame_ready(self.current_frame)

    def _on_2d_param_changed(self) -> None:
        self._request_current_frame()

    # -------------------------------------------------------------------------
    # Screenshot and Video Export
    # -------------------------------------------------------------------------

    def _on_save_screenshot(self) -> None:
        if self.model is None or self.current_frame is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please load a file first.")
            return

        default_out = Path("outputs") / "screenshots" / f"{self.model.path.stem}_sweep_{self.current_frame.sweep_data.sweep_id}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Screenshot", str(default_out.resolve()), "PNG Images (*.png)"
        )
        if not path:
            return

        p3d = capture_widget_pixmap(self.scene3d)
        p2d = capture_widget_pixmap(self.panel2d)
        mapping = self.combo_mapping.currentText()

        saved = composite_screenshot(
            scene3d_pixmap=p3d,
            panel2d_pixmap=p2d,
            sweep_data=self.current_frame.sweep_data,
            model=self.model,
            mapping_convention=mapping,
            output_path=path,
        )
        QtWidgets.QMessageBox.information(self, "Screenshot Saved", f"Saved composite screenshot to:\n{saved}")

    def _on_export_mp4(self) -> None:
        if self.model is None or self.current_frame is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please load a file first.")
            return

        self._pause()
        curr_sid = self.current_frame.sweep_data.sweep_id
        dlg = VideoExportDialog(self.model, curr_sid, parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        settings = dlg.get_settings()

        # Render capture callback: sets sweep, waits for ready, grabs pixmap
        # For video export, we create an offscreen composite function
        def render_sweep_image(sweep_id: int) -> QtGui.QImage:
            s_idx = self.model.sweep_id_to_idx[sweep_id]
            # Synchronously read single sweep data
            with h5py.File(self.model.path, "r") as h5:
                sweep_data = self.model.read_sweep_data(h5, s_idx)
            hist_enu, hist_h, hist_snr, hist_amp = self.model.get_history_slice(s_idx)
            params = self._get_frame_params()
            grid = compute_waveform_cross_section(
                sweep=sweep_data,
                mode=params.mode_2d,
                x_axis_mode=params.x_axis_mode,
                mapping_convention=params.mapping_convention,
                n_cols=params.n_cols,
                n_rows=params.n_rows,
                asinh_scale=params.asinh_scale,
            )
            # Update GUI widgets directly
            self.scene3d.update_frame(
                sweep=sweep_data,
                history_enu=hist_enu,
                history_values=hist_h,
                all_trajectory_enu=self.model.sweep_instrument_enu,
            )
            self.panel2d.update_cross_section(grid)
            QtWidgets.QApplication.processEvents()

            p3d = capture_widget_pixmap(self.scene3d)
            p2d = capture_widget_pixmap(self.panel2d)
            temp_path = Path(".compile_tmp") / "export_frame.png"
            composite_path = composite_screenshot(
                scene3d_pixmap=p3d,
                panel2d_pixmap=p2d,
                sweep_data=sweep_data,
                model=self.model,
                mapping_convention=params.mapping_convention,
                output_path=temp_path,
            )
            return QtGui.QImage(str(composite_path))

        progress_dialog = QtWidgets.QProgressDialog("Exporting MP4 video...", "Cancel", 0, 100, self)
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)

        exporter = VideoExportWorker(
            model=self.model,
            start_sweep=settings["start_sweep"],
            end_sweep=settings["end_sweep"],
            step=settings["step"],
            fps=settings["fps"],
            output_path=settings["output_path"],
            render_callback=render_sweep_image,
            parent=self,
        )

        def on_prog(curr, tot, msg):
            progress_dialog.setMaximum(tot)
            progress_dialog.setValue(curr)
            progress_dialog.setLabelText(msg)

        def on_fin(vid_p, meta_p):
            progress_dialog.close()
            QtWidgets.QMessageBox.information(
                self, "Export Complete", f"Successfully exported video to:\n{vid_p}\n\nMetadata sidecar:\n{meta_p}"
            )

        def on_err(err):
            progress_dialog.close()
            QtWidgets.QMessageBox.critical(self, "Export Failed", err)

        exporter.progress.connect(on_prog)
        exporter.finished.connect(on_fin)
        exporter.error.connect(on_err)
        progress_dialog.canceled.connect(exporter.cancel)

        exporter.start()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._pause()
        self.worker.stop()
        self.worker_thread.quit()
        self.worker_thread.wait(2000)
        super().closeEvent(event)

