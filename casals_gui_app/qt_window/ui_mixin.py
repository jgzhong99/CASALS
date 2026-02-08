"""UI construction and style helpers for the Qt CASALS main window."""

from __future__ import annotations

from .runtime import (
    Figure,
    FigureCanvas,
    NavigationToolbar,
    PYQTGRAPH_IMPORT_ERROR,
    QtCore,
    QtGui,
    QtInteractor,
    QtWidgets,
    pg,
)


class QtMainWindowUiMixin:
    def _build_ui(self) -> None:
        root = QtWidgets.QWidget(self)
        self.setCentralWidget(root)
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        file_group = QtWidgets.QGroupBox("TDMS File")
        file_layout = QtWidgets.QGridLayout(file_group)
        self.file_path_edit = QtWidgets.QLineEdit()
        choose_btn = QtWidgets.QPushButton("Choose...")
        load_btn = QtWidgets.QPushButton("Load")
        self.file_info_label = QtWidgets.QLabel("No file loaded.")
        self.file_info_label.setWordWrap(True)
        file_layout.addWidget(self.file_path_edit, 0, 0)
        file_layout.addWidget(choose_btn, 0, 1)
        file_layout.addWidget(load_btn, 0, 2)
        file_layout.addWidget(self.file_info_label, 1, 0, 1, 3)
        file_layout.setColumnStretch(0, 1)
        root_layout.addWidget(file_group)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root_layout.addWidget(self.main_splitter, 1)

        self.controls_scroll = QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        controls_host = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QVBoxLayout(controls_host)
        self.controls_layout.setAlignment(QtCore.Qt.AlignTop)
        self.controls_scroll.setWidget(controls_host)
        self.controls_scroll.setMinimumWidth(self.CONTROL_PANEL_MIN_WIDTH)
        self.main_splitter.addWidget(self.controls_scroll)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        self.main_splitter.addWidget(right)
        self.main_splitter.setStretchFactor(1, 1)

        self.view_stack = QtWidgets.QStackedWidget()
        right_layout.addWidget(self.view_stack, 1)

        self._build_controls()
        self._build_2d_view()
        self._build_3d_view()
        self._apply_ui_scale()

        self.status_label = QtWidgets.QLabel("Load a TDMS file to start.")
        right_layout.addWidget(self.status_label)

        self.summary_text = QtWidgets.QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(200)
        right_layout.addWidget(self.summary_text)
        two_d_backend = "PyQtGraph (interactive)" if self._interactive_2d_enabled else "Matplotlib"
        optional_note = ""
        if not self._interactive_2d_enabled and PYQTGRAPH_IMPORT_ERROR is not None:
            optional_note = "\n5) Optional interactive 2D: pip install pyqtgraph"
        self._set_summary(
            "Notebook flow:\n"
            "1) Extract row cube (sweep, step, samples)\n"
            "2) Apply TX_SAMPLES_CUT = [8, 68]\n"
            "3) Flatten to Ordering B (step-major -> sweep-minor)\n"
            f"4) 2D in {two_d_backend}, 3D in embedded PyVistaQt"
            f"{optional_note}"
        )

        choose_btn.clicked.connect(self.choose_file)
        load_btn.clicked.connect(self.load_file)

        self._draw_placeholder("Load a TDMS file, then click Render.")
        self.view_stack.setCurrentIndex(0)

    def _build_controls(self) -> None:
        row_group = QtWidgets.QGroupBox("Row")
        row_layout = QtWidgets.QGridLayout(row_group)
        row_layout.addWidget(QtWidgets.QLabel("Row ID"), 0, 0)
        self.row_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.row_slider.setRange(0, 0)
        self.row_spin = QtWidgets.QSpinBox()
        self.row_spin.setRange(0, 0)
        self.prev_btn = QtWidgets.QPushButton("Prev")
        self.next_btn = QtWidgets.QPushButton("Next")
        row_layout.addWidget(self.row_slider, 1, 0, 1, 3)
        row_layout.addWidget(self.row_spin, 2, 0)
        row_layout.addWidget(self.prev_btn, 2, 1)
        row_layout.addWidget(self.next_btn, 2, 2)
        self.play_btn = QtWidgets.QPushButton("Play")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        row_layout.addWidget(self.play_btn, 3, 0)
        row_layout.addWidget(self.pause_btn, 3, 1)
        row_layout.addWidget(self.stop_btn, 3, 2)
        row_layout.addWidget(QtWidgets.QLabel("2D speed"), 4, 0)
        self.playback_speed_spin = QtWidgets.QDoubleSpinBox()
        self.playback_speed_spin.setRange(self.MIN_PLAYBACK_ROWS_PER_SEC, self.MAX_PLAYBACK_ROWS_PER_SEC)
        self.playback_speed_spin.setSingleStep(0.5)
        self.playback_speed_spin.setDecimals(1)
        self.playback_speed_spin.setSuffix(" row/s")
        self.playback_speed_spin.setValue(self.DEFAULT_PLAYBACK_ROWS_PER_SEC)
        row_layout.addWidget(self.playback_speed_spin, 4, 1, 1, 2)
        self.controls_layout.addWidget(row_group)

        self.display_group = QtWidgets.QGroupBox("Display")
        display_layout = QtWidgets.QGridLayout(self.display_group)
        self.mode_value_label = QtWidgets.QLabel("2D")
        self.mode_toggle_btn = QtWidgets.QPushButton("Switch to 3D")
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(list(self.CMAPS))
        self.symmetric_check = QtWidgets.QCheckBox("Symmetric around 0")
        self.symmetric_check.setChecked(True)
        self.force_zero_center_check = QtWidgets.QCheckBox("Force 0 at colorbar center")
        self.force_zero_center_check.setChecked(False)
        self.auto_render_check = QtWidgets.QCheckBox("Auto render")
        self.auto_render_check.setChecked(True)
        display_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
        display_layout.addWidget(self.mode_value_label, 0, 1)
        display_layout.addWidget(self.mode_toggle_btn, 1, 0, 1, 2)
        display_layout.addWidget(QtWidgets.QLabel("Colormap"), 2, 0)
        display_layout.addWidget(self.cmap_combo, 2, 1)
        display_layout.addWidget(self.symmetric_check, 3, 0, 1, 2)
        display_layout.addWidget(self.force_zero_center_check, 4, 0, 1, 2)
        display_layout.addWidget(self.auto_render_check, 5, 0, 1, 2)
        self.controls_layout.addWidget(self.display_group)

        self.three_d_sampling_group = QtWidgets.QGroupBox("3D Sampling")
        ds_layout = QtWidgets.QGridLayout(self.three_d_sampling_group)
        self.ds_sample_spin = QtWidgets.QSpinBox()
        self.ds_sample_spin.setRange(1, 64)
        self.ds_sample_spin.setValue(2)
        self.ds_fp_spin = QtWidgets.QSpinBox()
        self.ds_fp_spin.setRange(1, 16)
        self.ds_fp_spin.setValue(1)
        self.keep_view_check = QtWidgets.QCheckBox("Keep mouse view")
        self.keep_view_check.setChecked(True)
        self.fast_3d_check = QtWidgets.QCheckBox("Fast 3D (auto downsample)")
        self.fast_3d_check.setChecked(True)
        self.interactive_lod_check = QtWidgets.QCheckBox("Interactive LOD boost")
        self.interactive_lod_check.setChecked(True)
        self.show_colorbar_check = QtWidgets.QCheckBox("Show 3D colorbar")
        self.show_colorbar_check.setChecked(False)
        ds_layout.addWidget(QtWidgets.QLabel("Sample ds"), 0, 0)
        ds_layout.addWidget(self.ds_sample_spin, 0, 1)
        ds_layout.addWidget(QtWidgets.QLabel("Footprint ds"), 1, 0)
        ds_layout.addWidget(self.ds_fp_spin, 1, 1)
        ds_layout.addWidget(self.keep_view_check, 2, 0, 1, 2)
        ds_layout.addWidget(self.fast_3d_check, 3, 0, 1, 2)
        ds_layout.addWidget(self.interactive_lod_check, 4, 0, 1, 2)
        ds_layout.addWidget(self.show_colorbar_check, 5, 0, 1, 2)
        self.controls_layout.addWidget(self.three_d_sampling_group)

        self.three_d_view_group = QtWidgets.QGroupBox("3D View")
        view_layout = QtWidgets.QGridLayout(self.three_d_view_group)
        self.elev_spin = QtWidgets.QDoubleSpinBox()
        self.elev_spin.setRange(-90.0, 90.0)
        self.elev_spin.setSingleStep(1.0)
        self.elev_spin.setValue(self.DEFAULT_ELEV)
        self.azim_spin = QtWidgets.QDoubleSpinBox()
        self.azim_spin.setRange(-180.0, 180.0)
        self.azim_spin.setSingleStep(1.0)
        self.azim_spin.setValue(self.DEFAULT_AZIM)
        self.x_scale_spin = QtWidgets.QDoubleSpinBox()
        self.x_scale_spin.setRange(self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        self.x_scale_spin.setSingleStep(0.05)
        self.x_scale_spin.setValue(self.DEFAULT_X_VIS_SCALE)
        self.y_scale_spin = QtWidgets.QDoubleSpinBox()
        self.y_scale_spin.setRange(self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        self.y_scale_spin.setSingleStep(0.05)
        self.y_scale_spin.setValue(self.DEFAULT_Y_VIS_SCALE)
        self.z_scale_spin = QtWidgets.QDoubleSpinBox()
        self.z_scale_spin.setRange(self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        self.z_scale_spin.setSingleStep(0.05)
        self.z_scale_spin.setValue(self.DEFAULT_Z_SCALE)
        reset_view_btn = QtWidgets.QPushButton("Reset 3D View")
        view_layout.addWidget(QtWidgets.QLabel("Elev"), 0, 0)
        view_layout.addWidget(self.elev_spin, 0, 1)
        view_layout.addWidget(QtWidgets.QLabel("Azim"), 1, 0)
        view_layout.addWidget(self.azim_spin, 1, 1)
        view_layout.addWidget(QtWidgets.QLabel("X vis scale"), 2, 0)
        view_layout.addWidget(self.x_scale_spin, 2, 1)
        view_layout.addWidget(QtWidgets.QLabel("Y vis scale"), 3, 0)
        view_layout.addWidget(self.y_scale_spin, 3, 1)
        view_layout.addWidget(QtWidgets.QLabel("Z vis scale"), 4, 0)
        view_layout.addWidget(self.z_scale_spin, 4, 1)
        view_layout.addWidget(reset_view_btn, 5, 0, 1, 2)
        self.controls_layout.addWidget(self.three_d_view_group)

        self.clip_group = QtWidgets.QGroupBox("General - Percentile Clip")
        clip_layout = QtWidgets.QGridLayout(self.clip_group)
        self.clip_low_spin = QtWidgets.QDoubleSpinBox()
        self.clip_low_spin.setRange(0.0, 100.0)
        self.clip_low_spin.setSingleStep(0.1)
        self.clip_low_spin.setValue(self.DEFAULT_PERCENTILE_CLIP[0])
        self.clip_high_spin = QtWidgets.QDoubleSpinBox()
        self.clip_high_spin.setRange(0.0, 100.0)
        self.clip_high_spin.setSingleStep(0.1)
        self.clip_high_spin.setValue(self.DEFAULT_PERCENTILE_CLIP[1])
        clip_layout.addWidget(QtWidgets.QLabel("Low (%)"), 0, 0)
        clip_layout.addWidget(self.clip_low_spin, 0, 1)
        clip_layout.addWidget(QtWidgets.QLabel("High (%)"), 1, 0)
        clip_layout.addWidget(self.clip_high_spin, 1, 1)
        self.controls_layout.addWidget(self.clip_group)

        self.ui_group = QtWidgets.QGroupBox("General - UI")
        ui_layout = QtWidgets.QGridLayout(self.ui_group)
        self.ui_control_scale_spin = QtWidgets.QSpinBox()
        self.ui_control_scale_spin.setRange(
            int(self.MIN_UI_CONTROL_SCALE * 100.0),
            int(self.MAX_UI_CONTROL_SCALE * 100.0),
        )
        self.ui_control_scale_spin.setSingleStep(5)
        self.ui_control_scale_spin.setSuffix(" %")
        startup_scale_pct = self._to_int(
            getattr(self, "_startup_ui_control_scale_percent", self.DEFAULT_UI_CONTROL_SCALE_PERCENT),
            self.DEFAULT_UI_CONTROL_SCALE_PERCENT,
        )
        startup_scale_pct = int(
            round(
                self._clamp(
                    float(startup_scale_pct),
                    self.MIN_UI_CONTROL_SCALE * 100.0,
                    self.MAX_UI_CONTROL_SCALE * 100.0,
                )
            )
        )
        self.ui_control_scale_spin.setValue(startup_scale_pct)

        self.ui_font_spin = QtWidgets.QDoubleSpinBox()
        self.ui_font_spin.setRange(self.MIN_UI_FONT_PT, self.MAX_UI_FONT_PT)
        self.ui_font_spin.setSingleStep(0.5)
        self.ui_font_spin.setDecimals(1)
        self.ui_font_spin.setSuffix(" pt")
        startup_font_pt = self._to_float(
            getattr(self, "_startup_ui_font_pt", self._default_ui_font_pt),
            self._default_ui_font_pt,
        )
        startup_font_pt = self._clamp(float(startup_font_pt), self.MIN_UI_FONT_PT, self.MAX_UI_FONT_PT)
        self.ui_font_spin.setValue(startup_font_pt)

        reset_ui_btn = QtWidgets.QPushButton("Reset UI Scale")

        ui_layout.addWidget(QtWidgets.QLabel("Control size"), 0, 0)
        ui_layout.addWidget(self.ui_control_scale_spin, 0, 1)
        ui_layout.addWidget(QtWidgets.QLabel("Font size"), 1, 0)
        ui_layout.addWidget(self.ui_font_spin, 1, 1)
        ui_layout.addWidget(reset_ui_btn, 2, 0, 1, 2)
        self.controls_layout.addWidget(self.ui_group)

        action_row = QtWidgets.QWidget()
        action_layout = QtWidgets.QVBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        render_btn = QtWidgets.QPushButton("Render")
        save_btn = QtWidgets.QPushButton("Save PNG")
        action_layout.addWidget(render_btn)
        action_layout.addWidget(save_btn)
        self.controls_layout.addWidget(action_row)
        self.controls_layout.addStretch(1)

        self.row_slider.valueChanged.connect(self._on_row_slider_changed)
        self.row_slider.sliderReleased.connect(self._maybe_render)
        self.row_spin.valueChanged.connect(self._on_row_spin_changed)
        self.prev_btn.clicked.connect(lambda: self._shift_row(-1))
        self.next_btn.clicked.connect(lambda: self._shift_row(1))
        self.play_btn.clicked.connect(self._start_2d_playback)
        self.pause_btn.clicked.connect(self._pause_2d_playback)
        self.stop_btn.clicked.connect(self._stop_2d_playback)
        self.playback_speed_spin.valueChanged.connect(self._on_playback_speed_changed)

        self.mode_toggle_btn.clicked.connect(self._toggle_mode)
        self.cmap_combo.currentTextChanged.connect(self._maybe_render)
        self.symmetric_check.toggled.connect(self._maybe_render)
        self.force_zero_center_check.toggled.connect(self._maybe_render)
        self.auto_render_check.toggled.connect(self._on_auto_render_toggled)

        self.ds_sample_spin.valueChanged.connect(self._maybe_render)
        self.ds_fp_spin.valueChanged.connect(self._maybe_render)
        self.keep_view_check.toggled.connect(self._on_keep_view_toggled)
        self.fast_3d_check.toggled.connect(self._maybe_render)
        self.interactive_lod_check.toggled.connect(self._maybe_render)
        self.show_colorbar_check.toggled.connect(self._maybe_render)

        self.elev_spin.valueChanged.connect(self._on_view_controls_changed)
        self.azim_spin.valueChanged.connect(self._on_view_controls_changed)
        self.x_scale_spin.valueChanged.connect(self._on_view_controls_changed)
        self.y_scale_spin.valueChanged.connect(self._on_view_controls_changed)
        self.z_scale_spin.valueChanged.connect(self._on_view_controls_changed)
        reset_view_btn.clicked.connect(self._reset_3d_view)

        self.clip_low_spin.valueChanged.connect(self._maybe_render)
        self.clip_high_spin.valueChanged.connect(self._maybe_render)

        self.ui_control_scale_spin.valueChanged.connect(self._on_ui_scale_changed)
        self.ui_font_spin.valueChanged.connect(self._on_ui_scale_changed)
        reset_ui_btn.clicked.connect(self._reset_ui_scale)

        render_btn.clicked.connect(lambda: self.render_current_row(show_errors=True))
        save_btn.clicked.connect(self.save_png)

        self._on_playback_speed_changed(float(self.playback_speed_spin.value()))
        self._update_playback_controls()
        self._apply_ui_scale()
        self._apply_mode_visibility()

    def _build_2d_view(self) -> None:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        if self._interactive_2d_enabled:
            plot_col = QtWidgets.QWidget()
            plot_col_layout = QtWidgets.QVBoxLayout(plot_col)
            plot_col_layout.setContentsMargins(0, 0, 0, 0)
            plot_col_layout.setSpacing(6)

            self.pg_plot = pg.PlotWidget()
            self.pg_plot.setBackground("white")
            self.pg_plot.showGrid(x=True, y=True, alpha=0.16)
            self.pg_plot.setLabel("bottom", "Footprint index (Ordering B: step-major -> sweep-minor)")
            self.pg_plot.setLabel("left", "Range (m)")
            self.pg_plot.setTitle("CASALS row heatmap")

            self.pg_image_item = pg.ImageItem(axisOrder="row-major")
            self.pg_plot.addItem(self.pg_image_item)
            plot_col_layout.addWidget(self.pg_plot, 1)

            self.pg_colorbar_view = pg.GraphicsLayoutWidget()
            self.pg_colorbar_view.setBackground("white")
            self.pg_colorbar_view.setMinimumHeight(64)
            self.pg_colorbar_view.setMaximumHeight(96)
            self.pg_colorbar = pg.ColorBarItem(
                values=(0.0, 1.0),
                colorMap=self._pg_colormap_from_cmap(self._get_cmap()),
                label="Amplitude (signed ADC counts)",
                interactive=False,
                orientation="horizontal",
                pen=pg.mkPen("#2a2f36"),
            )
            self.pg_colorbar_view.addItem(self.pg_colorbar, row=0, col=0)
            self.pg_colorbar.setImageItem(self.pg_image_item)
            self.pg_colorbar.setPreferredHeight(18)
            self.pg_colorbar.setMinimumHeight(16)
            try:
                self.pg_colorbar.axis.setStyle(showValues=True, autoExpandTextSpace=True, tickTextOffset=8)
            except Exception:
                pass
            plot_col_layout.addWidget(self.pg_colorbar_view, 0)

            view_box = self.pg_plot.getViewBox()
            view_box.setMouseEnabled(x=True, y=True)
            view_box.invertY(True)

            layout.addWidget(plot_col, 1)
        else:
            self.figure = Figure(figsize=(8.6, 6.4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            toolbar = NavigationToolbar(self.canvas, page)
            layout.addWidget(toolbar)
            layout.addWidget(self.canvas, 1)

        self.view_stack.addWidget(page)

    def _build_3d_view(self) -> None:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.pv_view = QtInteractor(page)
        self._last_cube_axes_actor = None
        self.pv_view.set_background("white")
        self.pv_view.add_axes(interactive=False)
        self._update_3d_axis_fonts()
        try:
            self.pv_view.enable_trackball_style()
        except Exception:
            pass
        # QtInteractor is already a QWidget; add it directly for reliable mouse events.
        layout.addWidget(self.pv_view, 1)
        self.view_stack.addWidget(page)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_summary(self, text: str) -> None:
        self.summary_text.setPlainText(text)

    def _current_ui_control_scale(self) -> float:
        if not hasattr(self, "ui_control_scale_spin"):
            return 1.0
        raw = float(self.ui_control_scale_spin.value()) / 100.0
        return self._clamp(raw, self.MIN_UI_CONTROL_SCALE, self.MAX_UI_CONTROL_SCALE)

    def _visual_font_pt(self) -> float:
        if not hasattr(self, "ui_font_spin"):
            return self._default_ui_font_pt
        return self._clamp(float(self.ui_font_spin.value()), self.MIN_UI_FONT_PT, self.MAX_UI_FONT_PT)

    def _visual_tick_font(self) -> QtGui.QFont:
        font = QtGui.QFont(QtWidgets.QApplication.font())
        font.setPointSizeF(max(6.0, self._visual_font_pt() * 0.90))
        return font

    def _update_2d_axis_fonts(self) -> None:
        font_pt = self._visual_font_pt()
        tick_font = self._visual_tick_font()

        if self._interactive_2d_enabled and self.pg_plot is not None:
            try:
                self.pg_plot.getAxis("bottom").setLabel(
                    "Footprint index (Ordering B: step-major -> sweep-minor)",
                    **{"font-size": f"{font_pt:.1f}pt"},
                )
                self.pg_plot.getAxis("left").setLabel("Range (m)", **{"font-size": f"{font_pt:.1f}pt"})
                self.pg_plot.getAxis("bottom").setStyle(tickFont=tick_font)
                self.pg_plot.getAxis("left").setStyle(tickFont=tick_font)
            except Exception:
                pass
            if self.pg_colorbar is not None:
                axis = self.pg_colorbar.axis
                colorbar_tick_font = QtGui.QFont(tick_font)
                colorbar_tick_font.setPointSizeF(max(8.5, font_pt * 1.10))
                try:
                    axis.setStyle(
                        showValues=True,
                        autoExpandTextSpace=True,
                        tickTextOffset=max(8, int(round(font_pt * 0.9))),
                    )
                except Exception:
                    pass
                try:
                    axis.setStyle(tickFont=colorbar_tick_font)
                except Exception:
                    try:
                        axis.setTickFont(colorbar_tick_font)
                    except Exception:
                        pass
                try:
                    axis.setTextPen(pg.mkPen("#2a2f36"))
                except Exception:
                    pass
                try:
                    axis.setPen(pg.mkPen("#2a2f36"))
                except Exception:
                    pass
                try:
                    axis.setHeight(int(max(32, round(font_pt * 2.9))))
                except Exception:
                    pass
                try:
                    axis.setLabel(
                        "Amplitude (signed ADC counts)",
                        **{"font-size": f"{max(9.0, font_pt * 1.12):.1f}pt"},
                    )
                except Exception:
                    pass
                try:
                    bar_height = int(max(16, round(font_pt * 1.35)))
                    self.pg_colorbar.setPreferredHeight(bar_height)
                    self.pg_colorbar.setMinimumHeight(max(14, bar_height - 2))
                except Exception:
                    pass
                try:
                    if self.pg_colorbar_view is not None:
                        holder_height = int(max(64, round(font_pt * 5.2)))
                        self.pg_colorbar_view.setMinimumHeight(holder_height)
                        self.pg_colorbar_view.setMaximumHeight(holder_height + 16)
                except Exception:
                    pass
            return

        if self.figure is None or self.canvas is None:
            return
        tick_size = max(6.0, font_pt * 0.90)
        title_size = font_pt * 1.05
        for ax in self.figure.axes:
            try:
                ax.tick_params(axis="both", labelsize=tick_size)
            except Exception:
                pass
            try:
                ax.xaxis.label.set_size(font_pt)
                ax.yaxis.label.set_size(font_pt)
                ax.title.set_fontsize(title_size)
            except Exception:
                pass
        self.canvas.draw_idle()

    def _update_3d_axis_fonts(
        self,
        axes_ranges: tuple[float, float, float, float, float, float] | None = None,
        scene_bounds: tuple[float, float, float, float, float, float] | None = None,
        render_now: bool = True,
    ) -> tuple[bool, bool]:
        if not hasattr(self, "pv_view"):
            return False, False
        if axes_ranges is not None:
            self._last_3d_axes_ranges = axes_ranges
        if scene_bounds is not None:
            self._last_3d_scene_bounds = scene_bounds

        font_size = max(8, int(round(self._visual_font_pt() * 1.10)))
        width_px = max(480, int(self.pv_view.width()))
        height_px = max(320, int(self.pv_view.height()))
        tick_px = max(58, int(round(font_size * 3.0)))
        n_xlabels = int(self._clamp(float(width_px) / float(tick_px), 5.0, 10.0))
        n_ylabels = int(self._clamp(float(height_px) / float(tick_px), 5.0, 10.0))
        n_zlabels = int(self._clamp(float(height_px) / float(tick_px * 1.2), 4.0, 8.0))
        x_fmt = "%.2f"
        y_fmt = "%.0f"
        z_fmt = "%.1f"
        if self._last_3d_axes_ranges is not None:
            x0, x1, y0, y1, z0, z1 = self._last_3d_axes_ranges
            x_span = abs(float(x1) - float(x0))
            z_span = abs(float(z1) - float(z0))
            if abs(float(y1) - float(y0)) <= 260.0:
                n_ylabels = min(n_ylabels, 9)
            if x_span < 10.0:
                x_fmt = "%.3f"
            elif x_span >= 200.0:
                x_fmt = "%.1f"
            if z_span < 20.0:
                z_fmt = "%.2f"
            elif z_span >= 200.0:
                z_fmt = "%.0f"

        kwargs = {
            "grid": False,
            "location": "outer",
            "all_edges": False,
            "use_2d": False,
            "xtitle": "Range (m)",
            "ytitle": "Track index",
            "ztitle": "Amplitude",
            "font_size": font_size,
            "n_xlabels": n_xlabels,
            "n_ylabels": n_ylabels,
            "n_zlabels": n_zlabels,
            "minor_ticks": False,
            "ticks": "outside",
        }
        if self._last_3d_axes_ranges is not None:
            kwargs["axes_ranges"] = self._last_3d_axes_ranges
        if self._last_3d_scene_bounds is not None:
            kwargs["bounds"] = self._last_3d_scene_bounds

        cube_axes_actor = None
        fallback_drop_sets = (
            (),
            ("ticks",),
            ("minor_ticks", "ticks"),
            ("bounds",),
            ("ticks", "bounds"),
            ("minor_ticks", "ticks", "bounds"),
            ("n_xlabels", "n_ylabels", "n_zlabels", "minor_ticks", "ticks"),
            ("use_2d", "n_xlabels", "n_ylabels", "n_zlabels", "minor_ticks", "ticks"),
            ("font_size", "use_2d", "n_xlabels", "n_ylabels", "n_zlabels", "minor_ticks", "ticks"),
            (
                "axes_ranges",
                "bounds",
                "font_size",
                "use_2d",
                "n_xlabels",
                "n_ylabels",
                "n_zlabels",
                "minor_ticks",
                "ticks",
            ),
        )
        used_call_kwargs = None
        for drop_keys in fallback_drop_sets:
            call_kwargs = dict(kwargs)
            for key in drop_keys:
                call_kwargs.pop(key, None)
            try:
                cube_axes_actor = self.pv_view.show_bounds(**call_kwargs)
                used_call_kwargs = call_kwargs
                break
            except Exception:
                cube_axes_actor = None
        if cube_axes_actor is None:
            self._last_cube_axes_actor = None
            return False, False
        self._last_cube_axes_actor = cube_axes_actor

        try:
            cube_axes_actor.SetXLabelFormat(x_fmt)
            cube_axes_actor.SetYLabelFormat(y_fmt)
            cube_axes_actor.SetZLabelFormat(z_fmt)
        except Exception:
            pass

        if render_now:
            try:
                self.pv_view.render()
            except Exception:
                pass
        used_axes_ranges = bool(used_call_kwargs is not None and "axes_ranges" in used_call_kwargs)
        used_axes_ranges = self._enforce_cube_axes_ranges_and_bounds(cube_axes_actor) or used_axes_ranges
        if used_axes_ranges and self._last_3d_axes_ranges is not None:
            used_axes_ranges = self._cube_axes_ranges_match(cube_axes_actor, self._last_3d_axes_ranges)
        used_bounds = bool(used_call_kwargs is not None and "bounds" in used_call_kwargs)
        return used_axes_ranges, used_bounds

    def _enforce_cube_axes_ranges_and_bounds(self, cube_axes_actor) -> bool:
        use_2d_mode = getattr(cube_axes_actor, "SetUse2DMode", None)
        if callable(use_2d_mode):
            try:
                use_2d_mode(False)
            except Exception:
                pass

        bounds = self._last_3d_scene_bounds
        if bounds is not None:
            try:
                cube_axes_actor.SetBounds(*[float(v) for v in bounds])
            except Exception:
                try:
                    cube_axes_actor.SetBounds([float(v) for v in bounds])
                except Exception:
                    pass

        ranges = self._last_3d_axes_ranges
        if ranges is None:
            return False
        x0, x1, y0, y1, z0, z1 = (float(v) for v in ranges)
        success = False
        for method_name, pair in (
            ("SetXAxisRange", (x0, x1)),
            ("SetYAxisRange", (y0, y1)),
            ("SetZAxisRange", (z0, z1)),
        ):
            method = getattr(cube_axes_actor, method_name, None)
            if not callable(method):
                continue
            try:
                method(float(pair[0]), float(pair[1]))
                success = True
            except Exception:
                try:
                    method([float(pair[0]), float(pair[1])])
                    success = True
                except Exception:
                    pass

        if not success:
            return False
        try:
            cube_axes_actor.Modified()
        except Exception:
            pass
        return self._cube_axes_ranges_match(cube_axes_actor, ranges)

    @staticmethod
    def _actor_axis_range_pair(cube_axes_actor, getter_name: str) -> tuple[float, float] | None:
        getter = getattr(cube_axes_actor, getter_name, None)
        if not callable(getter):
            return None
        try:
            values = getter()
        except Exception:
            return None
        try:
            if values is None or len(values) < 2:
                return None
            lo = float(values[0])
            hi = float(values[1])
            return lo, hi
        except Exception:
            return None

    def _cube_axes_ranges_match(
        self,
        cube_axes_actor,
        expected_axes_ranges: tuple[float, float, float, float, float, float],
    ) -> bool:
        getters_sets = (
            ("GetXAxisRange", "GetYAxisRange", "GetZAxisRange"),
            ("GetXRange", "GetYRange", "GetZRange"),
        )
        actual = None
        for gx, gy, gz in getters_sets:
            xr = self._actor_axis_range_pair(cube_axes_actor, gx)
            yr = self._actor_axis_range_pair(cube_axes_actor, gy)
            zr = self._actor_axis_range_pair(cube_axes_actor, gz)
            if xr is not None and yr is not None and zr is not None:
                actual = (xr[0], xr[1], yr[0], yr[1], zr[0], zr[1])
                break
        if actual is None:
            return False

        expected = tuple(float(v) for v in expected_axes_ranges)
        for a, e in zip(actual, expected):
            tol = max(1e-3, abs(e) * 5e-4)
            if abs(float(a) - float(e)) > tol:
                return False
        return True

    def _apply_ui_scale(self) -> None:
        if not hasattr(self, "ui_control_scale_spin") or not hasattr(self, "ui_font_spin"):
            return

        scale = self._current_ui_control_scale()
        font_pt = self._visual_font_pt()

        base_font = QtWidgets.QApplication.font()
        font = QtGui.QFont(base_font)
        font.setPointSizeF(font_pt)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setFont(font)
        self.setFont(font)

        input_h = int(round(24 * scale))
        pad_v = int(round(3 * scale))
        pad_h = int(round(7 * scale))
        indicator = int(round(14 * scale))
        button_min_w = int(round(72 * scale))
        row_btn_min_w = int(round(56 * scale))
        group_title_pad = int(round(4 * scale))
        group_title_left = int(round(8 * scale))

        self.setStyleSheet(
            f"""
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton {{
                min-height: {input_h}px;
                padding: {pad_v}px {pad_h}px;
            }}
            QPushButton {{
                min-width: {button_min_w}px;
            }}
            QCheckBox::indicator {{
                width: {indicator}px;
                height: {indicator}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {group_title_left}px;
                padding: 0 {group_title_pad}px 0 {group_title_pad}px;
            }}
            """
        )

        prev_next_width = max(row_btn_min_w, int(round(52 * scale)))
        if hasattr(self, "prev_btn"):
            self.prev_btn.setMinimumWidth(prev_next_width)
        if hasattr(self, "next_btn"):
            self.next_btn.setMinimumWidth(prev_next_width)
        if hasattr(self, "play_btn"):
            self.play_btn.setMinimumWidth(prev_next_width)
        if hasattr(self, "pause_btn"):
            self.pause_btn.setMinimumWidth(prev_next_width)
        if hasattr(self, "stop_btn"):
            self.stop_btn.setMinimumWidth(prev_next_width)

        self._update_2d_axis_fonts()
        self._update_3d_axis_fonts()
        self._apply_resolution_adaptive_layout(initial=False, force=True)

    def _on_ui_scale_changed(self, _value) -> None:
        self._apply_ui_scale()
        if self._layout_initialized:
            self._save_settings()

    def _reset_ui_scale(self) -> None:
        self.ui_control_scale_spin.blockSignals(True)
        self.ui_control_scale_spin.setValue(self.DEFAULT_UI_CONTROL_SCALE_PERCENT)
        self.ui_control_scale_spin.blockSignals(False)
        self.ui_font_spin.blockSignals(True)
        self.ui_font_spin.setValue(self._default_ui_font_pt)
        self.ui_font_spin.blockSignals(False)
        self._apply_ui_scale()
        if self._layout_initialized:
            self._save_settings()
