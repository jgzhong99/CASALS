"""Data/render/event logic for the Qt CASALS main window."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..tdms_processor import TDMS_IMPORT_ERROR, tdms_available
from .runtime import LinearSegmentedColormap, QtCore, QtWidgets, mpl_cm, pv


class QtMainWindowRenderMixin:
    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _sanitize_row(self) -> int:
        max_row = max(0, self.processor.rows_per_file - 1)
        row = int(self.row_spin.value())
        row = int(self._clamp(float(row), 0.0, float(max_row)))
        if row != self.row_spin.value():
            self.row_spin.blockSignals(True)
            self.row_spin.setValue(row)
            self.row_spin.blockSignals(False)
        if row != self.row_slider.value():
            self.row_slider.blockSignals(True)
            self.row_slider.setValue(row)
            self.row_slider.blockSignals(False)
        return row

    def _sanitize_percentiles(self) -> tuple[float, float]:
        low = float(self.clip_low_spin.value())
        high = float(self.clip_high_spin.value())
        low = self._clamp(low, 0.0, 99.9)
        high = self._clamp(high, 0.1, 100.0)
        if low >= high:
            high = min(100.0, low + 0.1)
            if low >= high:
                low = max(0.0, high - 0.1)

        self.clip_low_spin.blockSignals(True)
        self.clip_low_spin.setValue(round(low, 2))
        self.clip_low_spin.blockSignals(False)
        self.clip_high_spin.blockSignals(True)
        self.clip_high_spin.setValue(round(high, 2))
        self.clip_high_spin.blockSignals(False)
        return low, high

    def _sanitize_downsample(self) -> tuple[int, int]:
        ds_sample = max(1, min(64, int(self.ds_sample_spin.value())))
        ds_fp = max(1, min(16, int(self.ds_fp_spin.value())))
        if ds_sample != self.ds_sample_spin.value():
            self.ds_sample_spin.blockSignals(True)
            self.ds_sample_spin.setValue(ds_sample)
            self.ds_sample_spin.blockSignals(False)
        if ds_fp != self.ds_fp_spin.value():
            self.ds_fp_spin.blockSignals(True)
            self.ds_fp_spin.setValue(ds_fp)
            self.ds_fp_spin.blockSignals(False)
        return ds_sample, ds_fp

    @staticmethod
    def _downsample_indices(length: int, step: int) -> np.ndarray:
        length = int(length)
        if length <= 0:
            return np.zeros((0,), dtype=np.int32)
        step = max(1, int(step))
        indices = np.arange(0, length, step, dtype=np.int32)
        last_idx = length - 1
        if indices.size == 0 or int(indices[-1]) != last_idx:
            indices = np.concatenate([indices, np.array([last_idx], dtype=np.int32)])
        return indices

    def _effective_3d_downsample(
        self,
        n_fp: int,
        n_samples: int,
        target_fp: int,
        target_samples: int,
    ) -> tuple[int, int]:
        ds_sample, ds_fp = self._sanitize_downsample()
        if not self.fast_3d_check.isChecked():
            return ds_sample, ds_fp

        min_ds_fp = max(1, int(np.ceil(float(n_fp) / float(target_fp))))
        min_ds_sample = max(1, int(np.ceil(float(n_samples) / float(target_samples))))
        return max(ds_sample, min_ds_sample), max(ds_fp, min_ds_fp)

    def _sanitize_3d_view(self) -> tuple[float, float, float, float, float]:
        elev = self._clamp(float(self.elev_spin.value()), -90.0, 90.0)
        azim = self._clamp(float(self.azim_spin.value()), -180.0, 180.0)
        x_scale = self._clamp(float(self.x_scale_spin.value()), self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        y_scale = self._clamp(float(self.y_scale_spin.value()), self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        z_scale = self._clamp(float(self.z_scale_spin.value()), self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)

        self.elev_spin.blockSignals(True)
        self.elev_spin.setValue(round(elev, 2))
        self.elev_spin.blockSignals(False)
        self.azim_spin.blockSignals(True)
        self.azim_spin.setValue(round(azim, 2))
        self.azim_spin.blockSignals(False)
        self.x_scale_spin.blockSignals(True)
        self.x_scale_spin.setValue(round(x_scale, 3))
        self.x_scale_spin.blockSignals(False)
        self.y_scale_spin.blockSignals(True)
        self.y_scale_spin.setValue(round(y_scale, 3))
        self.y_scale_spin.blockSignals(False)
        self.z_scale_spin.blockSignals(True)
        self.z_scale_spin.setValue(round(z_scale, 3))
        self.z_scale_spin.blockSignals(False)
        return elev, azim, x_scale, y_scale, z_scale

    def _get_cmap(self) -> str:
        cmap = str(self.cmap_combo.currentText())
        if cmap not in self.CMAPS:
            cmap = self.CMAPS[0]
        return cmap

    def _resolve_mpl_cmap(self, cmap_name: str | None = None):
        name = str(cmap_name) if cmap_name is not None else self._get_cmap()
        if name not in self.CMAPS:
            name = self.CMAPS[0]
        if name == self.CUSTOM_CMAP_GREEN_BLACK_BLUE:
            if LinearSegmentedColormap is not None:
                return LinearSegmentedColormap.from_list(
                    self.CUSTOM_CMAP_GREEN_BLACK_BLUE,
                    ["#00ff00", "#000000", "#0000ff"],
                    N=256,
                )
            return mpl_cm.get_cmap("RdBu_r", 256)
        return mpl_cm.get_cmap(name, 256)

    def _lookup_table_from_cmap(self, cmap_name: str) -> np.ndarray:
        cmap = self._resolve_mpl_cmap(cmap_name)
        lut = np.clip(cmap(np.linspace(0.0, 1.0, 256)) * 255.0, 0.0, 255.0).astype(np.uint8)
        return lut

    def _draw_placeholder(self, text: str) -> None:
        if self._interactive_2d_enabled and self.pg_plot is not None and self.pg_image_item is not None:
            self.pg_image_item.setImage(
                np.zeros((2, 2), dtype=np.float32),
                autoLevels=False,
                levels=(0.0, 1.0),
            )
            self.pg_image_item.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
            self.pg_image_item.setLookupTable(self._lookup_table_from_cmap(self._get_cmap()))
            title_pt = max(8.0, self._visual_font_pt() * 1.05)
            self.pg_plot.setTitle(f"<span style='font-size:{title_pt:.1f}pt'>{text}</span>")
            view_box = self.pg_plot.getViewBox()
            try:
                view_box.setAspectLocked(False)
            except Exception:
                pass
            view_box.setRange(xRange=(0.0, 1.0), yRange=(1.0, 0.0), padding=0.0)
            self._update_2d_axis_fonts()
            return

        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=self._visual_font_pt())
        ax.set_axis_off()
        self.canvas.draw_idle()

    def choose_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose TDMS file",
            str(Path.cwd()),
            "TDMS files (*.tdms);;All files (*.*)",
        )
        if path:
            self.file_path_edit.setText(path)

    def load_file(self) -> None:
        if not tdms_available():
            QtWidgets.QMessageBox.critical(
                self,
                "nptdms missing",
                "Cannot load TDMS because nptdms is not installed.\n"
                f"Import error: {TDMS_IMPORT_ERROR}\n\n"
                "Install dependency with:\n"
                "pip install nptdms",
            )
            return

        path_text = self.file_path_edit.text().strip()
        if not path_text:
            QtWidgets.QMessageBox.warning(self, "Missing file", "Please choose a TDMS file first.")
            return

        try:
            self.meta = self.processor.load(path_text)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", f"Failed to load TDMS file:\n{exc}")
            return

        max_row = max(0, self.meta.rows_per_file - 1)
        target_row = max(0, min(max_row, int(self._startup_row_id)))
        self.row_slider.blockSignals(True)
        self.row_slider.setRange(0, max_row)
        self.row_slider.setValue(target_row)
        self.row_slider.blockSignals(False)
        self.row_spin.blockSignals(True)
        self.row_spin.setRange(0, max_row)
        self.row_spin.setValue(target_row)
        self.row_spin.blockSignals(False)
        self._last_camera_position = None
        self._last_3d_axes_ranges = None

        self.file_info_label.setText(
            f"Loaded: {self.meta.path.name} | Group: {self.meta.group_name} | "
            f"Rows: {self.meta.rows_per_file} | Footprints/row: {self.meta.footprints_per_row} | "
            f"Samples/row: {self.meta.samples_per_row} | Rx samples: {self.meta.samples_per_row_rx} | "
            f"Channels: {self.meta.channels}"
        )
        self._set_status("File loaded.")
        self.render_current_row(show_errors=True)

    def _on_auto_render_toggled(self, checked: bool) -> None:
        if checked:
            self._maybe_render()
        else:
            self._render_timer.stop()

    def _apply_mode_visibility(self) -> None:
        mode = self._view_mode
        if hasattr(self, "three_d_sampling_group"):
            self.three_d_sampling_group.setVisible(mode == "3D")
        if hasattr(self, "three_d_view_group"):
            self.three_d_view_group.setVisible(mode == "3D")
        if hasattr(self, "clip_group"):
            self.clip_group.setVisible(True)
        if hasattr(self, "ui_group"):
            self.ui_group.setVisible(True)
        if hasattr(self, "mode_value_label"):
            self.mode_value_label.setText(mode)
        if hasattr(self, "mode_toggle_btn"):
            self.mode_toggle_btn.setText("Switch to 3D" if mode == "2D" else "Switch to 2D")
        if hasattr(self, "view_stack") and self.view_stack.count() >= 2:
            self.view_stack.setCurrentIndex(1 if mode == "3D" else 0)

    def _set_view_mode(self, mode: str, trigger_render: bool = True) -> None:
        mode_norm = str(mode).upper()
        if mode_norm not in {"2D", "3D"}:
            mode_norm = "2D"
        self._view_mode = mode_norm
        self._apply_mode_visibility()
        self._apply_resolution_adaptive_layout(initial=False, force=True)

        if not trigger_render:
            return
        if not self.auto_render_check.isChecked():
            self._set_status(f"Mode switched to {mode_norm}. Auto render is off, click Render.")
            return
        self._maybe_render()

    def _toggle_mode(self) -> None:
        next_mode = "3D" if self._view_mode == "2D" else "2D"
        self._set_view_mode(next_mode, trigger_render=True)

    def _on_row_slider_changed(self, value: int) -> None:
        if self.row_spin.value() == value:
            return
        self.row_spin.blockSignals(True)
        self.row_spin.setValue(value)
        self.row_spin.blockSignals(False)

    def _on_row_spin_changed(self, value: int) -> None:
        if self.row_slider.value() != value:
            self.row_slider.blockSignals(True)
            self.row_slider.setValue(value)
            self.row_slider.blockSignals(False)
        self._maybe_render()

    def _shift_row(self, delta: int) -> None:
        self.row_spin.setValue(self.row_spin.value() + int(delta))
        self._sanitize_row()
        self._maybe_render()

    def _on_view_controls_changed(self, _value: float) -> None:
        if self._is_syncing_camera:
            return
        self._sanitize_3d_view()
        self._maybe_render()

    def _maybe_render(self) -> None:
        if not self.auto_render_check.isChecked():
            return
        self._render_timer.stop()
        self._render_timer.start(self.RENDER_DEBOUNCE_MS)

    def _clear_render_timer(self) -> None:
        self._render_timer.stop()

    def _display_range(self, row_data: np.ndarray) -> tuple[float, float]:
        low, high = self._sanitize_percentiles()
        vmin, vmax = np.percentile(row_data, [low, high])
        vmin = float(vmin)
        vmax = float(vmax)

        force_zero_center = bool(self.force_zero_center_check.isChecked()) if hasattr(self, "force_zero_center_check") else False
        if self.symmetric_check.isChecked() or force_zero_center:
            magnitude = max(abs(vmin), abs(vmax))
            if np.isclose(magnitude, 0.0):
                magnitude = 1.0
            return -magnitude, magnitude

        if np.isclose(vmin, vmax):
            return vmin, vmin + 1.0
        return vmin, vmax

    def _draw_2d(self, row_data: np.ndarray, row_id: int, vmin: float, vmax: float) -> None:
        n_fp, n_samples = row_data.shape
        y_max = n_samples * self.processor.dr
        font_pt = self._visual_font_pt()
        title_pt = max(8.0, font_pt * 1.05)
        tick_pt = max(6.0, font_pt * 0.90)

        if self._interactive_2d_enabled and self.pg_plot is not None and self.pg_image_item is not None:
            image_data = row_data.T.astype(np.float32, copy=False)
            self.pg_image_item.setImage(
                image_data,
                autoLevels=False,
                levels=(float(vmin), float(vmax)),
            )
            self.pg_image_item.setLookupTable(self._lookup_table_from_cmap(self._get_cmap()))

            x_max = max(1.0, float(n_fp - 1))
            y_max_safe = max(float(y_max), 1e-6)
            self.pg_image_item.setRect(QtCore.QRectF(0.0, 0.0, x_max, y_max_safe))

            self.pg_plot.setTitle(
                f"<span style='font-size:{title_pt:.1f}pt'>CASALS row {row_id} heatmap (interactive 2D)</span>"
            )
            self._update_2d_axis_fonts()
            view_box = self.pg_plot.getViewBox()
            view_box.setLimits(xMin=0.0, xMax=x_max, yMin=0.0, yMax=y_max_safe)
            try:
                view_box.setAspectLocked(False)
            except Exception:
                pass
            view_box.setRange(xRange=(0.0, x_max), yRange=(y_max_safe, 0.0), padding=0.02)
            return

        self.figure.clf()
        ax = self.figure.add_subplot(111)
        image = ax.imshow(
            row_data.T,
            aspect="auto",
            origin="lower",
            extent=[0, n_fp - 1, 0.0, y_max],
            cmap=self._resolve_mpl_cmap(self._get_cmap()),
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.invert_yaxis()
        ax.set_xlabel("Footprint index (Ordering B: step-major -> sweep-minor)", fontsize=font_pt)
        ax.set_ylabel("Range (m)", fontsize=font_pt)
        ax.set_title(f"CASALS row {row_id} heatmap", fontsize=title_pt)
        ax.tick_params(axis="both", labelsize=tick_pt)
        cbar = self.figure.colorbar(image, ax=ax, pad=0.02)
        cbar.set_label("Amplitude (signed ADC counts)", fontsize=font_pt)
        cbar.ax.tick_params(labelsize=tick_pt)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    @staticmethod
    def _camera_from_angles(
        center: tuple[float, float, float],
        radius: float,
        elev: float,
        azim: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        el = np.deg2rad(float(elev))
        az = np.deg2rad(float(azim))
        dist = max(5.0, float(radius) * 2.8)

        cx, cy, cz = center
        px = cx + dist * np.cos(el) * np.cos(az)
        py = cy + dist * np.cos(el) * np.sin(az)
        pz = cz + dist * np.sin(el)
        return (px, py, pz), (cx, cy, cz), (0.0, 0.0, 1.0)

    @staticmethod
    def _angles_from_camera(camera_position) -> tuple[float, float]:
        pos, focal, _viewup = camera_position
        dx = float(pos[0] - focal[0])
        dy = float(pos[1] - focal[1])
        dz = float(pos[2] - focal[2])
        radius = max(1e-6, float(np.sqrt(dx * dx + dy * dy + dz * dz)))
        elev = float(np.degrees(np.arcsin(np.clip(dz / radius, -1.0, 1.0))))
        azim = float(np.degrees(np.arctan2(dy, dx)))
        return elev, azim

    def _draw_3d(
        self,
        row_data: np.ndarray,
        row_id: int,
        vmin: float,
        vmax: float,
        force_interactive_lod: bool = False,
    ) -> tuple[tuple[int, int], int, int]:
        if force_interactive_lod and self.interactive_lod_check.isChecked():
            target_fp = self.INTERACT_MAX_3D_FP
            target_samples = self.INTERACT_MAX_3D_SAMPLES
        else:
            target_fp = self.MAX_3D_FP
            target_samples = self.MAX_3D_SAMPLES

        ds_sample, ds_fp = self._effective_3d_downsample(
            n_fp=row_data.shape[0],
            n_samples=row_data.shape[1],
            target_fp=target_fp,
            target_samples=target_samples,
        )
        elev, azim, x_vis_scale, y_vis_scale, z_vis_scale = self._sanitize_3d_view()

        fp_idx = self._downsample_indices(row_data.shape[0], ds_fp)
        sample_idx = self._downsample_indices(row_data.shape[1], ds_sample)
        z_raw = row_data[np.ix_(fp_idx, sample_idx)].astype(np.float32, copy=False)
        z_color = np.clip(z_raw, vmin, vmax)

        sample_axis = sample_idx.astype(np.float32, copy=False) * float(self.processor.dr)
        fp_axis = fp_idx.astype(np.float32, copy=False)
        x_grid, y_grid = np.meshgrid(sample_axis, fp_axis)

        x_span = float(sample_axis[-1] - sample_axis[0]) if sample_axis.size > 1 else 1.0
        y_span = float(fp_axis[-1] - fp_axis[0]) if fp_axis.size > 1 else 1.0
        z_span = float(np.max(z_color) - np.min(z_color)) if z_color.size > 1 else 1.0
        z_span_geom = float(np.max(z_raw) - np.min(z_raw)) if z_raw.size > 1 else 1.0
        x_span = max(1e-6, x_span)
        y_span = max(1e-6, y_span)
        z_span = max(1e-6, z_span)
        z_span_geom = max(1e-6, z_span_geom)

        y_auto_scale = self._clamp(x_span / y_span, self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        z_auto_scale = self._clamp(x_span / z_span, self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        x_total_scale = self._clamp(float(x_vis_scale), self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        y_total_scale = self._clamp(y_auto_scale * float(y_vis_scale), self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        z_total_scale = self._clamp(z_auto_scale * float(z_vis_scale), self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        self._last_axis_scale = (x_total_scale, y_total_scale, z_total_scale)
        self._update_3d_axis_fonts(
            axes_ranges=(
                0.0,
                float(max(0, row_data.shape[1] - 1)) * float(self.processor.dr),
                0.0,
                float(max(0, int(self.processor.footprints_per_row) - 1)),
                float(vmin),
                float(vmax),
            )
        )

        grid = pv.StructuredGrid(x_grid, y_grid, z_raw)
        grid.point_data["amplitude"] = z_color.ravel(order="F")

        if self._pv_mesh_actor is not None:
            self.pv_view.remove_actor(self._pv_mesh_actor, render=False)
            self._pv_mesh_actor = None
        try:
            self.pv_view.remove_scalar_bar(render=False)
        except Exception:
            pass

        axis_font_size = max(8, int(round(self._visual_font_pt() * 1.10)))
        hud_font_size = max(9, int(round(self._visual_font_pt() * 0.95)))
        show_bar = bool(self.show_colorbar_check.isChecked() and not force_interactive_lod)
        self._pv_mesh_actor = self.pv_view.add_mesh(
            grid,
            scalars="amplitude",
            cmap=self._resolve_mpl_cmap(self._get_cmap()),
            clim=(float(vmin), float(vmax)),
            show_edges=False,
            smooth_shading=True,
            lighting=True,
            ambient=0.55,
            diffuse=0.60,
            specular=0.10,
            show_scalar_bar=show_bar,
            scalar_bar_args={
                "title": "Amplitude",
                "fmt": "%.1f",
                "title_font_size": axis_font_size,
                "label_font_size": axis_font_size,
            },
            name="casals_surface",
            render=False,
        )
        try:
            self._pv_mesh_actor.SetScale(
                float(self._last_axis_scale[0]),
                float(self._last_axis_scale[1]),
                float(self._last_axis_scale[2]),
            )
        except Exception:
            try:
                self._pv_mesh_actor.scale = self._last_axis_scale
            except Exception:
                pass

        if self.keep_view_check.isChecked() and self._last_camera_position is not None:
            self.pv_view.camera_position = self._last_camera_position
        else:
            cx = float(np.mean(sample_axis)) if sample_axis.size else 0.0
            cy = float(np.mean(fp_axis)) if fp_axis.size else 0.0
            cz = float(np.mean(z_raw)) if z_raw.size else 0.0
            radius = max(
                x_span * float(self._last_axis_scale[0]),
                y_span * float(self._last_axis_scale[1]),
                z_span_geom * float(self._last_axis_scale[2]),
                1.0,
            )
            self.pv_view.camera_position = self._camera_from_angles(
                center=(cx, cy, cz),
                radius=radius,
                elev=elev,
                azim=azim,
            )

        try:
            self.pv_view.add_text(
                f"CASALS 3D row surface | row={row_id} | shape={z_raw.shape[0]}x{z_raw.shape[1]}",
                position="upper_left",
                font_size=hud_font_size,
                name="hud",
            )
        except Exception:
            pass

        self.pv_view.reset_camera_clipping_range()
        self.pv_view.render()
        self._last_camera_position = self.pv_view.camera_position
        cur_elev, cur_azim = self._angles_from_camera(self._last_camera_position)
        if self.keep_view_check.isChecked():
            self._is_syncing_camera = True
            self.elev_spin.blockSignals(True)
            self.elev_spin.setValue(round(cur_elev, 2))
            self.elev_spin.blockSignals(False)
            self.azim_spin.blockSignals(True)
            self.azim_spin.setValue(round(cur_azim, 2))
            self.azim_spin.blockSignals(False)
            self._is_syncing_camera = False

        return z_raw.shape, ds_fp, ds_sample

    def _sync_camera_from_pyvista(self) -> None:
        if self.view_stack.currentIndex() != 1:
            return
        if not self.keep_view_check.isChecked():
            return
        camera_position = self.pv_view.camera_position
        if camera_position is None:
            return

        self._last_camera_position = camera_position
        elev, azim = self._angles_from_camera(camera_position)
        if self._is_syncing_camera:
            return

        if (
            abs(elev - float(self.elev_spin.value())) < 0.2
            and abs(azim - float(self.azim_spin.value())) < 0.2
        ):
            return

        self._is_syncing_camera = True
        self.elev_spin.blockSignals(True)
        self.elev_spin.setValue(round(elev, 2))
        self.elev_spin.blockSignals(False)
        self.azim_spin.blockSignals(True)
        self.azim_spin.setValue(round(azim, 2))
        self.azim_spin.blockSignals(False)
        self._is_syncing_camera = False

    def _reset_3d_view(self) -> None:
        self.elev_spin.setValue(self.DEFAULT_ELEV)
        self.azim_spin.setValue(self.DEFAULT_AZIM)
        self.x_scale_spin.setValue(self.DEFAULT_X_VIS_SCALE)
        self.y_scale_spin.setValue(self.DEFAULT_Y_VIS_SCALE)
        self.z_scale_spin.setValue(self.DEFAULT_Z_SCALE)
        self._last_camera_position = None
        self._last_3d_axes_ranges = None
        self._maybe_render()

    def render_current_row(self, show_errors: bool = True, force_interactive_lod: bool = False) -> None:
        self._clear_render_timer()
        if self.processor.tdms is None or self.meta is None:
            if show_errors:
                QtWidgets.QMessageBox.warning(self, "No data", "Please load a TDMS file first.")
            return

        row_id = self._sanitize_row()
        try:
            row_data = self.processor.extract_vis_swath_row(row_id).astype(np.float32, copy=False)
            expected_fp = int(self.processor.footprints_per_row)
            if row_data.shape[0] != expected_fp:
                raise ValueError(
                    f"Invalid row footprint dimension: {row_data.shape[0]} "
                    f"(expected {expected_fp})."
                )
            vmin, vmax = self._display_range(row_data)
        except Exception as exc:
            if show_errors:
                QtWidgets.QMessageBox.critical(self, "Extraction failed", f"Failed to extract row:\n{exc}")
            return

        mode = self._view_mode
        if mode not in {"2D", "3D"}:
            mode = "2D"
            self._set_view_mode(mode, trigger_render=False)

        try:
            if mode == "3D":
                self.view_stack.setCurrentIndex(1)
                ds_shape, ds_fp, ds_sample = self._draw_3d(
                    row_data,
                    row_id,
                    vmin,
                    vmax,
                    force_interactive_lod=force_interactive_lod,
                )
            else:
                self.view_stack.setCurrentIndex(0)
                ds_shape = None
                ds_fp = None
                ds_sample = None
                self._draw_2d(row_data, row_id, vmin, vmax)
        except Exception as exc:
            self._set_status(f"Render failed: {exc}")
            self._set_summary(f"Render failed.\n\nDetails:\n{exc}")
            if show_errors:
                QtWidgets.QMessageBox.critical(self, "Render failed", f"Failed to render plot:\n{exc}")
            return

        if mode == "3D" and ds_shape is not None:
            phase = "preview" if force_interactive_lod else "surface"
            sx, sy, sz = self._last_axis_scale
            self._set_status(
                f"3D pyvista {phase} | row={row_id} | input={row_data.shape[0]}x{row_data.shape[1]} | "
                f"ds={ds_shape[0]}x{ds_shape[1]} (fp_step={ds_fp}, samp_step={ds_sample}) | "
                f"display=[{vmin:.1f}, {vmax:.1f}] | visual-scale=({sx:.2f}, {sy:.2f}, {sz:.2f})"
            )
        else:
            self._set_status(
                f"2D rendered | row={row_id} | shape={row_data.shape[0]}x{row_data.shape[1]} | "
                f"display=[{vmin:.1f}, {vmax:.1f}]"
            )

        self._set_summary(
            f"Notebook-aligned extraction:\n"
            f"- 2D backend: {'PyQtGraph (interactive)' if self._interactive_2d_enabled else 'Matplotlib'}\n"
            f"- row_id: {row_id}\n"
            f"- raw cube: (sweep={self.processor.sweeps_per_cycle}, step={self.processor.wvl_steps_per_sweep}, "
            f"samples={self.processor.samples_per_row})\n"
            f"- TX_SAMPLES_CUT: {self.processor.tx_samples_cut}\n"
            f"- flattening: Ordering B (step-major -> sweep-minor)\n"
            f"- matrix shape: {row_data.shape[0]} x {row_data.shape[1]} (footprint x sample)\n"
            f"- mode: {mode} (embedded Qt widget)\n"
            f"- 3D backend: PyVistaQt (VTK)\n"
            f"- fast 3D: {self.fast_3d_check.isChecked()} | target <= {self.MAX_3D_FP}x{self.MAX_3D_SAMPLES}\n"
            f"- interactive LOD: {self.interactive_lod_check.isChecked()} | preview <= {self.INTERACT_MAX_3D_FP}x{self.INTERACT_MAX_3D_SAMPLES}\n"
            f"- 3D colorbar: {self.show_colorbar_check.isChecked()}\n"
            f"- user axis visual scale (x,y,z): ({self.x_scale_spin.value():.3f}, {self.y_scale_spin.value():.3f}, {self.z_scale_spin.value():.3f})\n"
            f"- axis visual scale only (x,y,z): ({self._last_axis_scale[0]:.2f}, {self._last_axis_scale[1]:.2f}, {self._last_axis_scale[2]:.2f})\n"
            f"- percentile clip: [{self.clip_low_spin.value():.2f}, {self.clip_high_spin.value():.2f}]%\n"
            f"- symmetric display: {self.symmetric_check.isChecked()}\n"
            f"- force 0 at colorbar center: {self.force_zero_center_check.isChecked()}\n"
            f"- dr = {self.processor.dr:.6f} m/sample"
        )

    def save_png(self) -> None:
        mode = self._view_mode.lower()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save plot as PNG",
            str(Path.cwd() / f"casals_row_{self.row_spin.value():04d}_{mode}.png"),
            "PNG (*.png)",
        )
        if not path:
            return
        try:
            if self._view_mode == "3D":
                self.pv_view.screenshot(path)
            else:
                if self._interactive_2d_enabled and self.pg_plot is not None:
                    ok = self.pg_plot.grab().save(path, "PNG")
                    if not ok:
                        raise RuntimeError("Qt failed to save interactive 2D snapshot.")
                else:
                    self.figure.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", f"Failed to save PNG:\n{exc}")

    def _cleanup(self) -> None:
        self._clear_render_timer()
        self._camera_poll_timer.stop()
        try:
            self.processor.close()
        except Exception:
            pass
        try:
            self.pv_view.close()
        except Exception:
            pass

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._layout_initialized:
            self._layout_initialized = True

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._layout_initialized and self._auto_layout_enabled:
            self._apply_resolution_adaptive_layout(initial=False)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_settings()
        self._cleanup()
        super().closeEvent(event)
