"""Data/render/event logic for the Qt CASALS main window."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..tdms_processor import TDMS_IMPORT_ERROR, tdms_available
from .runtime import LinearSegmentedColormap, QtCore, QtWidgets, mpl_cm, pg, pv


class QtMainWindowRenderMixin:
    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _clamp_axis_range(raw_range, bound_lo: float, bound_hi: float) -> tuple[float, float]:
        lo_bound = float(min(bound_lo, bound_hi))
        hi_bound = float(max(bound_lo, bound_hi))
        min_span = max(1e-6, (hi_bound - lo_bound) * 1e-6)
        try:
            a = float(raw_range[0])
            b = float(raw_range[1])
        except Exception:
            return lo_bound, hi_bound

        descending = a > b
        lo = min(a, b)
        hi = max(a, b)
        if (hi - lo) < min_span:
            center = 0.5 * (lo + hi)
            lo = center - 0.5 * min_span
            hi = center + 0.5 * min_span

        if lo < lo_bound:
            shift = lo_bound - lo
            lo += shift
            hi += shift
        if hi > hi_bound:
            shift = hi - hi_bound
            lo -= shift
            hi -= shift

        lo = max(lo_bound, lo)
        hi = min(hi_bound, hi)
        if (hi - lo) < min_span:
            lo = lo_bound
            hi = hi_bound

        return (hi, lo) if descending else (lo, hi)

    def _track_index_axis_range(self, n_fp: int | None = None) -> tuple[float, float]:
        if n_fp is None:
            if self.meta is not None:
                try:
                    n_fp = int(getattr(self.meta, "footprints_per_row"))
                except Exception:
                    n_fp = None
            if n_fp is None:
                try:
                    n_fp = int(getattr(self.processor, "footprints_per_row"))
                except Exception:
                    n_fp = None
        if n_fp is None or n_fp <= 0:
            n_fp = int(getattr(self.processor, "EXPECTED_FOOTPRINTS_PER_ROW", 256))
        track_max = float(max(0, n_fp - 1))
        return 0.0, track_max

    def _normalize_axes_ranges(
        self, axes_ranges: tuple[float, float, float, float, float, float]
    ) -> tuple[float, float, float, float, float, float]:
        x0, x1, y0, y1, z0, z1 = (float(v) for v in axes_ranges)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        if z1 < z0:
            z0, z1 = z1, z0
        return x0, x1, y0, y1, z0, z1

    def _axis_origin_from_axes_ranges(self) -> tuple[float, float, float] | None:
        if self._last_3d_axes_ranges is None:
            return None
        x0, _x1, y0, _y1, z0, _z1 = (float(v) for v in self._last_3d_axes_ranges)
        return x0, y0, z0

    @staticmethod
    def _float_pair_from_values(values) -> tuple[float, float] | None:
        try:
            if values is None or len(values) < 2:
                return None
            return float(values[0]), float(values[1])
        except Exception:
            return None

    def _debug_3d_axes_enabled(self) -> bool:
        raw = str(os.getenv("CASALS_DEBUG_3D_AXES", "")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _tuple_list(values, ndigits: int = 6):
        if values is None:
            return None
        try:
            return [round(float(v), ndigits) for v in values]
        except Exception:
            return values

    def _log_3d_axes_debug(
        self,
        *,
        row_id: int,
        force_interactive_lod: bool,
        input_shape: tuple[int, int],
        ds_shape: tuple[int, int],
        ds_fp: int,
        ds_sample: int,
        axes_ranges: tuple[float, float, float, float, float, float],
        scene_bounds: tuple[float, float, float, float, float, float],
        used_axes_ranges: bool,
        used_bounds: bool,
    ) -> None:
        if not self._debug_3d_axes_enabled():
            return

        actor_bounds = None
        if self._pv_mesh_actor is not None:
            getter = getattr(self._pv_mesh_actor, "GetBounds", None)
            if callable(getter):
                actor_bounds = self._tuple_list(getter())

        plotter_bounds = self._tuple_list(getattr(self.pv_view, "bounds", None))
        cube_actor = getattr(self, "_last_cube_axes_actor", None)
        cube_x = None
        cube_y = None
        cube_z = None
        if cube_actor is not None:
            range_getters = (
                ("GetXAxisRange", "GetYAxisRange", "GetZAxisRange"),
                ("GetXRange", "GetYRange", "GetZRange"),
            )
            for gx, gy, gz in range_getters:
                gx_fn = getattr(cube_actor, gx, None)
                gy_fn = getattr(cube_actor, gy, None)
                gz_fn = getattr(cube_actor, gz, None)
                if not (callable(gx_fn) and callable(gy_fn) and callable(gz_fn)):
                    continue
                cube_x = self._float_pair_from_values(gx_fn())
                cube_y = self._float_pair_from_values(gy_fn())
                cube_z = self._float_pair_from_values(gz_fn())
                if cube_x is not None and cube_y is not None and cube_z is not None:
                    break

        payload = {
            "row_id": int(row_id),
            "preview_lod": bool(force_interactive_lod),
            "input_shape_fp_sample": [int(input_shape[0]), int(input_shape[1])],
            "ds_shape_fp_sample": [int(ds_shape[0]), int(ds_shape[1])],
            "downsample_steps": {"fp_step": int(ds_fp), "sample_step": int(ds_sample)},
            "expected_axes_ranges": self._tuple_list(axes_ranges),
            "expected_scene_bounds": self._tuple_list(scene_bounds),
            "axis_scale_xyz": self._tuple_list(self._last_axis_scale),
            "show_bounds_used_axes_ranges": bool(used_axes_ranges),
            "show_bounds_used_bounds": bool(used_bounds),
            "mesh_actor_bounds": actor_bounds,
            "plotter_bounds": plotter_bounds,
            "cube_axes_x_range": self._tuple_list(cube_x),
            "cube_axes_y_range": self._tuple_list(cube_y),
            "cube_axes_z_range": self._tuple_list(cube_z),
        }
        print("[CASALS 3D AXIS DEBUG] " + json.dumps(payload, ensure_ascii=False))

    def _on_render_timer_timeout(self) -> None:
        force_preview = bool(getattr(self, "_interactive_lod_pending", False))
        self._interactive_lod_pending = False
        self.render_current_row(show_errors=False, force_interactive_lod=force_preview)

    def _on_full_res_render_timeout(self) -> None:
        if self._view_mode != "3D":
            return
        if not self.auto_render_check.isChecked():
            return
        if self.processor.tdms is None or self.meta is None:
            return
        self.render_current_row(show_errors=False, force_interactive_lod=False)

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

    def _pg_colormap_from_cmap(self, cmap_name: str):
        cmap = self._resolve_mpl_cmap(cmap_name)
        stops = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        colors = np.clip(cmap(stops) * 255.0, 0.0, 255.0).astype(np.uint8)
        return pg.ColorMap(stops, colors)

    def _draw_placeholder(self, text: str) -> None:
        if self._interactive_2d_enabled and self.pg_plot is not None and self.pg_image_item is not None:
            self.pg_image_item.setImage(
                np.zeros((2, 2), dtype=np.float32),
                autoLevels=False,
                levels=(0.0, 1.0),
            )
            self.pg_image_item.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
            self.pg_image_item.setLookupTable(self._lookup_table_from_cmap(self._get_cmap()))
            if self.pg_colorbar is not None:
                self.pg_colorbar.setColorMap(self._pg_colormap_from_cmap(self._get_cmap()))
                self.pg_colorbar.setLevels((0.0, 1.0))
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
        if self._playback_timer.isActive() or self._playback_paused:
            self._stop_2d_playback(set_status=False)
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
        self._has_rendered_3d_scene = False
        self._last_3d_axes_ranges = None
        self._last_3d_scene_bounds = None
        self._update_camera_polling_state()

        self.file_info_label.setText(
            f"Loaded: {self.meta.path.name} | Group: {self.meta.group_name} | "
            f"Rows: {self.meta.rows_per_file} | Footprints/row: {self.meta.footprints_per_row} | "
            f"Samples/row: {self.meta.samples_per_row} | Rx samples: {self.meta.samples_per_row_rx} | "
            f"Channels: {self.meta.channels}"
        )
        self._set_status("File loaded.")
        self.render_current_row(show_errors=True)
        self._playback_paused = False
        self._update_playback_controls()

    def _on_auto_render_toggled(self, checked: bool) -> None:
        if checked:
            self._maybe_render()
        else:
            self._clear_render_timer()

    def _on_keep_view_toggled(self, _checked: bool) -> None:
        self._update_camera_polling_state()
        self._maybe_render()

    def _update_camera_polling_state(self) -> None:
        if not hasattr(self, "_camera_poll_timer"):
            return
        in_3d = getattr(self, "_view_mode", "2D") == "3D"
        has_scene = bool(getattr(self, "_has_rendered_3d_scene", False))
        keep_view = bool(hasattr(self, "keep_view_check") and self.keep_view_check.isChecked())
        should_poll = in_3d and has_scene and keep_view
        if should_poll:
            if not self._camera_poll_timer.isActive():
                self._camera_poll_timer.start()
        else:
            if self._camera_poll_timer.isActive():
                self._camera_poll_timer.stop()

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
        self._update_playback_controls()

    def _set_view_mode(self, mode: str, trigger_render: bool = True) -> None:
        mode_norm = str(mode).upper()
        if mode_norm not in {"2D", "3D"}:
            mode_norm = "2D"
        if mode_norm != "2D" and (self._playback_timer.isActive() or self._playback_paused):
            self._stop_2d_playback(set_status=False)
        self._view_mode = mode_norm
        self._apply_mode_visibility()
        self._apply_resolution_adaptive_layout(initial=False, force=True)
        self._update_camera_polling_state()

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
        if self._playback_timer.isActive():
            self._pause_2d_playback()
        if self.row_spin.value() == value:
            return
        self.row_spin.blockSignals(True)
        self.row_spin.setValue(value)
        self.row_spin.blockSignals(False)

    def _on_row_spin_changed(self, value: int) -> None:
        if self._playback_timer.isActive():
            self._pause_2d_playback()
        if self.row_slider.value() != value:
            self.row_slider.blockSignals(True)
            self.row_slider.setValue(value)
            self.row_slider.blockSignals(False)
        self._maybe_render()

    def _shift_row(self, delta: int) -> None:
        if self._playback_timer.isActive():
            self._pause_2d_playback()
        self.row_spin.setValue(self.row_spin.value() + int(delta))
        self._sanitize_row()
        self._maybe_render()

    def _on_view_controls_changed(self, _value: float) -> None:
        if self._is_syncing_camera:
            return
        self._sanitize_3d_view()
        if self._view_mode != "3D" or self._pv_mesh_actor is None or not self._has_rendered_3d_scene:
            self._maybe_render()
            return

        sender = self.sender()
        if sender in (self.elev_spin, self.azim_spin):
            if self._apply_camera_from_controls():
                return
            self._maybe_render()
            return

        if sender in (self.x_scale_spin, self.y_scale_spin, self.z_scale_spin):
            if self._apply_axis_scale_from_controls():
                return
            self._maybe_render()
            return

        self._maybe_render()

    def _scene_center_spans_from_axes(self) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        if self._last_3d_axes_ranges is None:
            return None
        x0, x1, y0, y1, z0, z1 = (float(v) for v in self._last_3d_axes_ranges)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        cz = 0.5 * (z0 + z1)
        x_span = max(1e-6, x1 - x0)
        y_span = max(1e-6, y1 - y0)
        z_span = max(1e-6, z1 - z0)
        return (cx, cy, cz), (x_span, y_span, z_span)

    def _apply_camera_from_controls(self) -> bool:
        scene = self._scene_center_spans_from_axes()
        if scene is None:
            return False
        center, spans = scene
        elev, azim, _x, _y, _z = self._sanitize_3d_view()
        sx, sy, sz = self._last_axis_scale
        radius = max(
            spans[0] * float(sx),
            spans[1] * float(sy),
            spans[2] * float(sz),
            1.0,
        )
        self.pv_view.camera_position = self._camera_from_angles(
            center=center,
            radius=radius,
            elev=elev,
            azim=azim,
        )
        self.pv_view.reset_camera_clipping_range()
        self.pv_view.render()
        self._last_camera_position = self.pv_view.camera_position
        self._has_rendered_3d_scene = True
        self._update_camera_polling_state()
        return True

    def _apply_axis_scale_from_controls(self) -> bool:
        if self._pv_mesh_actor is None:
            return False
        scene = self._scene_center_spans_from_axes()
        if scene is None:
            return False
        _center, spans = scene
        axis_origin = self._axis_origin_from_axes_ranges()
        if axis_origin is None:
            return False
        _elev, _azim, x_vis_scale, y_vis_scale, z_vis_scale = self._sanitize_3d_view()
        x_span, y_span, z_span = spans
        y_auto_scale = self._clamp(x_span / y_span, self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        z_auto_scale = self._clamp(x_span / z_span, self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        x_total_scale = self._clamp(float(x_vis_scale), self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        y_total_scale = self._clamp(y_auto_scale * float(y_vis_scale), self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        z_total_scale = self._clamp(z_auto_scale * float(z_vis_scale), self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        self._last_axis_scale = (x_total_scale, y_total_scale, z_total_scale)

        try:
            self._pv_mesh_actor.SetOrigin(
                float(axis_origin[0]),
                float(axis_origin[1]),
                float(axis_origin[2]),
            )
        except Exception:
            pass
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
                return False

        if self._last_3d_axes_ranges is not None:
            self._last_3d_axes_ranges = self._normalize_axes_ranges(self._last_3d_axes_ranges)
        self._last_3d_scene_bounds = self._last_3d_axes_ranges
        self._update_3d_axis_fonts(
            axes_ranges=self._last_3d_axes_ranges,
            scene_bounds=self._last_3d_scene_bounds,
            render_now=False,
        )
        self.pv_view.reset_camera_clipping_range()
        self.pv_view.render()
        self._last_camera_position = self.pv_view.camera_position
        self._has_rendered_3d_scene = True
        self._update_camera_polling_state()
        return True

    def _maybe_render(self) -> None:
        if not self.auto_render_check.isChecked():
            return
        preview_3d = self._view_mode == "3D" and self.interactive_lod_check.isChecked()
        self._render_timer.stop()
        self._full_res_render_timer.stop()
        self._interactive_lod_pending = preview_3d
        self._render_timer.start(self.RENDER_DEBOUNCE_MS)
        if preview_3d:
            self._full_res_render_timer.start(self.RENDER_DEBOUNCE_MS + self.INTERACTIVE_LOD_SETTLE_MS)

    def _clear_render_timer(self, include_full_res: bool = True) -> None:
        self._render_timer.stop()
        self._interactive_lod_pending = False
        if include_full_res:
            self._full_res_render_timer.stop()

    def _playback_rows_per_sec(self) -> float:
        if not hasattr(self, "playback_speed_spin"):
            return float(self.DEFAULT_PLAYBACK_ROWS_PER_SEC)
        value = float(self.playback_speed_spin.value())
        clamped = float(
            self._clamp(
                value,
                float(self.MIN_PLAYBACK_ROWS_PER_SEC),
                float(self.MAX_PLAYBACK_ROWS_PER_SEC),
            )
        )
        if not np.isclose(value, clamped):
            self.playback_speed_spin.blockSignals(True)
            self.playback_speed_spin.setValue(clamped)
            self.playback_speed_spin.blockSignals(False)
        return clamped

    def _playback_interval_ms(self) -> int:
        rows_per_sec = self._playback_rows_per_sec()
        return max(16, int(round(1000.0 / max(rows_per_sec, 1e-6))))

    def _update_playback_controls(self) -> None:
        if not all(hasattr(self, name) for name in ("play_btn", "pause_btn", "stop_btn")):
            return
        has_data = self.meta is not None and self.processor.tdms is not None
        is_2d_mode = self._view_mode == "2D"
        is_running = self._playback_timer.isActive()
        self.play_btn.setEnabled(has_data and is_2d_mode and not is_running)
        self.pause_btn.setEnabled(is_running)
        self.stop_btn.setEnabled(is_running or self._playback_paused)

    def _on_playback_speed_changed(self, _value: float) -> None:
        interval_ms = self._playback_interval_ms()
        self._playback_timer.setInterval(interval_ms)
        if self._playback_timer.isActive():
            self._playback_timer.start(interval_ms)
        if self._layout_initialized:
            self._save_settings()

    def _start_2d_playback(self, *_args) -> None:
        if self.processor.tdms is None or self.meta is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Please load a TDMS file before playback.")
            self._update_playback_controls()
            return
        if self._view_mode != "2D":
            self._set_view_mode("2D", trigger_render=False)

        interval_ms = self._playback_interval_ms()
        self._playback_paused = False
        self._playback_timer.start(interval_ms)
        self._update_playback_controls()
        self._set_status(f"2D playback running | {self._playback_rows_per_sec():.1f} row/s")

    def _pause_2d_playback(self, *_args) -> None:
        if not self._playback_timer.isActive():
            return
        self._playback_timer.stop()
        self._playback_paused = True
        self._update_playback_controls()
        self._set_status("2D playback paused.")

    def _stop_2d_playback(self, *_args, set_status: bool = True) -> None:
        was_active = self._playback_timer.isActive()
        was_paused = self._playback_paused
        self._playback_timer.stop()
        self._playback_paused = False
        self._update_playback_controls()
        if set_status and (was_active or was_paused):
            self._set_status("2D playback stopped.")

    def _on_playback_tick(self) -> None:
        if self.processor.tdms is None or self.meta is None:
            self._stop_2d_playback(set_status=False)
            return
        if self._view_mode != "2D":
            self._stop_2d_playback(set_status=False)
            return

        max_row = int(self.row_spin.maximum())
        next_row = int(self.row_spin.value()) + 1
        if next_row > max_row:
            self._stop_2d_playback(set_status=False)
            self._set_status("2D playback finished at last row.")
            return

        self.row_spin.blockSignals(True)
        self.row_spin.setValue(next_row)
        self.row_spin.blockSignals(False)
        self.row_slider.blockSignals(True)
        self.row_slider.setValue(next_row)
        self.row_slider.blockSignals(False)
        self._preserve_2d_view_on_next_draw = True
        try:
            self.render_current_row(show_errors=False)
        finally:
            self._preserve_2d_view_on_next_draw = False
        self._set_status(
            f"2D playback | row={next_row}/{max_row} | {self._playback_rows_per_sec():.1f} row/s"
        )

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
            view_box = self.pg_plot.getViewBox()
            keep_view = bool(self._preserve_2d_view_on_next_draw)
            previous_view_range = None
            if keep_view:
                try:
                    previous_view_range = view_box.viewRange()
                except Exception:
                    previous_view_range = None

            image_data = row_data.T.astype(np.float32, copy=False)
            self.pg_image_item.setImage(
                image_data,
                autoLevels=False,
                levels=(float(vmin), float(vmax)),
            )
            cmap_name = self._get_cmap()
            self.pg_image_item.setLookupTable(self._lookup_table_from_cmap(cmap_name))
            if self.pg_colorbar is not None:
                self.pg_colorbar.setColorMap(self._pg_colormap_from_cmap(cmap_name))
                self.pg_colorbar.setLevels((float(vmin), float(vmax)))

            x_max = max(1.0, float(n_fp - 1))
            y_max_safe = max(float(y_max), 1e-6)
            self.pg_image_item.setRect(QtCore.QRectF(0.0, 0.0, x_max, y_max_safe))

            self.pg_plot.setTitle(
                f"<span style='font-size:{title_pt:.1f}pt'>CASALS row {row_id} heatmap (interactive 2D)</span>"
            )
            self._update_2d_axis_fonts()
            view_box.setLimits(xMin=0.0, xMax=x_max, yMin=0.0, yMax=y_max_safe)
            try:
                view_box.setAspectLocked(False)
            except Exception:
                pass

            if (
                keep_view
                and previous_view_range is not None
                and len(previous_view_range) == 2
            ):
                x_range = self._clamp_axis_range(previous_view_range[0], 0.0, x_max)
                y_range = self._clamp_axis_range(previous_view_range[1], 0.0, y_max_safe)
                view_box.setRange(xRange=x_range, yRange=y_range, padding=0.0)
            else:
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
        x_axis_min = 0.0
        x_axis_max = float(max(0, row_data.shape[1] - 1)) * float(self.processor.dr)
        y_axis_min, y_axis_max = self._track_index_axis_range(n_fp=row_data.shape[0])
        z_axis_min = float(np.min(z_raw)) if z_raw.size else 0.0
        z_axis_max = float(np.max(z_raw)) if z_raw.size else 1.0
        if np.isclose(z_axis_min, z_axis_max):
            z_axis_min -= 0.5
            z_axis_max += 0.5
        cx = 0.5 * (x_axis_min + x_axis_max)
        cy = 0.5 * (y_axis_min + y_axis_max)
        cz = 0.5 * (z_axis_min + z_axis_max)
        axis_origin = (x_axis_min, y_axis_min, z_axis_min)

        x_span = x_axis_max - x_axis_min
        y_span = y_axis_max - y_axis_min
        z_span_geom = z_axis_max - z_axis_min
        x_span = max(1e-6, x_span)
        y_span = max(1e-6, y_span)
        z_span_geom = max(1e-6, z_span_geom)

        y_auto_scale = self._clamp(x_span / y_span, self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        z_auto_scale = self._clamp(x_span / z_span_geom, self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        x_total_scale = self._clamp(float(x_vis_scale), self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        y_total_scale = self._clamp(y_auto_scale * float(y_vis_scale), self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        z_total_scale = self._clamp(z_auto_scale * float(z_vis_scale), self.MIN_AXIS_SCALE, self.MAX_AXIS_SCALE)
        self._last_axis_scale = (x_total_scale, y_total_scale, z_total_scale)
        axes_ranges = self._normalize_axes_ranges(
            (
                x_axis_min,
                x_axis_max,
                y_axis_min,
                y_axis_max,
                z_axis_min,
                z_axis_max,
            )
        )
        # Keep cube-axis numeric values in physical units from the current row data.
        # Visual stretching is applied on the mesh actor transform only.
        scene_bounds = axes_ranges

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
            self._pv_mesh_actor.SetOrigin(
                float(axis_origin[0]),
                float(axis_origin[1]),
                float(axis_origin[2]),
            )
        except Exception:
            pass
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

        used_axes_ranges, used_bounds = self._update_3d_axis_fonts(
            axes_ranges=axes_ranges,
            scene_bounds=scene_bounds,
            render_now=False,
        )
        if not used_axes_ranges and not np.allclose(self._last_axis_scale, (1.0, 1.0, 1.0), atol=1e-6):
            try:
                self._pv_mesh_actor.SetScale(1.0, 1.0, 1.0)
            except Exception:
                try:
                    self._pv_mesh_actor.scale = (1.0, 1.0, 1.0)
                except Exception:
                    pass
            self._last_axis_scale = (1.0, 1.0, 1.0)
            self._last_3d_axes_ranges = None
            self._last_3d_scene_bounds = None
            self._update_3d_axis_fonts(render_now=False)

        if (
            self.keep_view_check.isChecked()
            and self._has_rendered_3d_scene
            and self._last_camera_position is not None
        ):
            self.pv_view.camera_position = self._last_camera_position
        else:
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
        self._has_rendered_3d_scene = True
        self._update_camera_polling_state()
        self._log_3d_axes_debug(
            row_id=row_id,
            force_interactive_lod=force_interactive_lod,
            input_shape=row_data.shape,
            ds_shape=z_raw.shape,
            ds_fp=ds_fp,
            ds_sample=ds_sample,
            axes_ranges=axes_ranges,
            scene_bounds=scene_bounds,
            used_axes_ranges=used_axes_ranges,
            used_bounds=used_bounds,
        )
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
        if not self._has_rendered_3d_scene:
            return
        if self._pv_mesh_actor is None:
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
        self._has_rendered_3d_scene = False
        self._last_3d_axes_ranges = None
        self._last_3d_scene_bounds = None
        self._update_camera_polling_state()
        self._maybe_render()

    def render_current_row(self, show_errors: bool = True, force_interactive_lod: bool = False) -> None:
        self._clear_render_timer(include_full_res=not force_interactive_lod)
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
        self._playback_timer.stop()
        self._playback_paused = False
        self._camera_poll_timer.stop()
        self._has_rendered_3d_scene = False
        self._last_3d_scene_bounds = None
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
