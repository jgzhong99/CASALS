"""Settings and adaptive layout helpers for the Qt CASALS main window."""

from __future__ import annotations

import json

from .runtime import QtCore, QtWidgets


class QtMainWindowSettingsMixin:
    @staticmethod
    def _to_int(value, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _to_float(value, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _load_settings(self) -> dict:
        if not self._settings_path.exists():
            return {}
        try:
            payload = json.loads(self._settings_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _restore_settings(self) -> bool:
        payload = self._settings_cache if isinstance(self._settings_cache, dict) else {}
        if not payload:
            return False

        tdms_path = payload.get("tdms_path")
        if isinstance(tdms_path, str):
            self.file_path_edit.setText(tdms_path)

        mode = str(payload.get("view_mode", "2D")).upper()
        self._set_view_mode(mode, trigger_render=False)

        cmap = str(payload.get("cmap", self.CMAPS[0]))
        if cmap in self.CMAPS:
            self.cmap_combo.setCurrentText(cmap)
        self.reverse_cmap_check.setChecked(bool(payload.get("reverse_cmap", False)))

        self.symmetric_check.setChecked(bool(payload.get("symmetric", True)))
        self.force_zero_center_check.setChecked(bool(payload.get("force_zero_center", False)))
        self.auto_render_check.setChecked(bool(payload.get("auto_update", True)))
        self.keep_view_check.setChecked(bool(payload.get("keep_view", True)))
        self.fast_3d_check.setChecked(bool(payload.get("fast_3d", True)))
        self.interactive_lod_check.setChecked(bool(payload.get("interactive_lod", True)))
        self.show_colorbar_check.setChecked(bool(payload.get("show_3d_colorbar", False)))
        reverse_track_axis = payload.get("reverse_track_axis", False)
        # Legacy configs had this option defaulting to True. If no schema marker exists,
        # treat it as legacy and reset to the new default (ascending 0 -> 255).
        if "reverse_track_axis_schema" not in payload:
            reverse_track_axis = False
        self.reverse_track_axis_check.setChecked(bool(reverse_track_axis))

        self.clip_low_spin.setValue(
            self._to_float(payload.get("clip_low", self.DEFAULT_PERCENTILE_CLIP[0]), self.DEFAULT_PERCENTILE_CLIP[0])
        )
        self.clip_high_spin.setValue(
            self._to_float(payload.get("clip_high", self.DEFAULT_PERCENTILE_CLIP[1]), self.DEFAULT_PERCENTILE_CLIP[1])
        )
        self.ds_sample_spin.setValue(self._to_int(payload.get("ds_sample", 2), 2))
        self.ds_fp_spin.setValue(self._to_int(payload.get("ds_fp", 1), 1))
        self.auto_target_fp_spin.setValue(
            self._to_int(payload.get("auto_target_fp", self.MAX_3D_FP), self.MAX_3D_FP)
        )
        self.auto_target_sample_spin.setValue(
            self._to_int(payload.get("auto_target_samples", self.MAX_3D_SAMPLES), self.MAX_3D_SAMPLES)
        )
        self.preview_target_fp_spin.setValue(
            self._to_int(payload.get("preview_target_fp", self.INTERACT_MAX_3D_FP), self.INTERACT_MAX_3D_FP)
        )
        self.preview_target_sample_spin.setValue(
            self._to_int(
                payload.get("preview_target_samples", self.INTERACT_MAX_3D_SAMPLES),
                self.INTERACT_MAX_3D_SAMPLES,
            )
        )
        self.elev_spin.setValue(self._to_float(payload.get("elev", self.DEFAULT_ELEV), self.DEFAULT_ELEV))
        self.azim_spin.setValue(self._to_float(payload.get("azim", self.DEFAULT_AZIM), self.DEFAULT_AZIM))
        self.x_scale_spin.setValue(
            self._to_float(payload.get("x_vis_scale", payload.get("aspect_x", self.DEFAULT_X_VIS_SCALE)), self.DEFAULT_X_VIS_SCALE)
        )
        self.y_scale_spin.setValue(
            self._to_float(payload.get("y_vis_scale", payload.get("aspect_y", self.DEFAULT_Y_VIS_SCALE)), self.DEFAULT_Y_VIS_SCALE)
        )
        self.z_scale_spin.setValue(self._to_float(payload.get("z_scale", self.DEFAULT_Z_SCALE), self.DEFAULT_Z_SCALE))
        self.ui_control_scale_spin.setValue(
            self._to_int(payload.get("ui_control_scale_pct", self.DEFAULT_UI_CONTROL_SCALE_PERCENT), self.DEFAULT_UI_CONTROL_SCALE_PERCENT)
        )
        self.ui_font_spin.setValue(
            self._to_float(payload.get("ui_font_pt", self._default_ui_font_pt), self._default_ui_font_pt)
        )
        self.playback_speed_spin.setValue(
            self._to_float(
                payload.get("playback_rows_per_sec", self.DEFAULT_PLAYBACK_ROWS_PER_SEC),
                self.DEFAULT_PLAYBACK_ROWS_PER_SEC,
            )
        )
        self._apply_ui_scale()

        self._startup_row_id = self._to_int(payload.get("row_id", 0), 0)
        self._sanitize_percentiles()
        self._sanitize_downsample()
        self._sanitize_3d_view()

        restored_layout = False
        geometry_b64 = payload.get("qt_window_geometry")
        if isinstance(geometry_b64, str) and geometry_b64.strip():
            try:
                geometry = QtCore.QByteArray.fromBase64(geometry_b64.encode("ascii"))
                restored_layout = bool(self.restoreGeometry(geometry))
            except Exception:
                restored_layout = False

        if not restored_layout:
            size = payload.get("qt_window_size")
            if isinstance(size, list) and len(size) == 2:
                width = max(480, self._to_int(size[0], self.PREFERRED_WINDOW_WIDTH))
                height = max(360, self._to_int(size[1], self.PREFERRED_WINDOW_HEIGHT))
                self.resize(width, height)
                restored_layout = True

        splitter_sizes = payload.get("qt_main_splitter_sizes")
        if isinstance(splitter_sizes, list) and len(splitter_sizes) >= 2:
            left = max(120, self._to_int(splitter_sizes[0], 320))
            right = max(320, self._to_int(splitter_sizes[1], 980))
            self.main_splitter.setSizes([left, right])

        return restored_layout

    def _save_settings(self) -> None:
        payload = dict(self._settings_cache) if isinstance(self._settings_cache, dict) else {}

        payload.update(
            {
                "tdms_path": self.file_path_edit.text().strip(),
                "row_id": int(self.row_spin.value()),
                "clip_low": float(self.clip_low_spin.value()),
                "clip_high": float(self.clip_high_spin.value()),
                "cmap": self._get_cmap(),
                "reverse_cmap": bool(self.reverse_cmap_check.isChecked()),
                "symmetric": bool(self.symmetric_check.isChecked()),
                "force_zero_center": bool(self.force_zero_center_check.isChecked()),
                "auto_update": bool(self.auto_render_check.isChecked()),
                "keep_view": bool(self.keep_view_check.isChecked()),
                "view_mode": str(self._view_mode),
                "elev": float(self.elev_spin.value()),
                "azim": float(self.azim_spin.value()),
                "x_vis_scale": float(self.x_scale_spin.value()),
                "y_vis_scale": float(self.y_scale_spin.value()),
                "z_scale": float(self.z_scale_spin.value()),
                "ds_sample": int(self.ds_sample_spin.value()),
                "ds_fp": int(self.ds_fp_spin.value()),
                "auto_target_fp": int(self.auto_target_fp_spin.value()),
                "auto_target_samples": int(self.auto_target_sample_spin.value()),
                "preview_target_fp": int(self.preview_target_fp_spin.value()),
                "preview_target_samples": int(self.preview_target_sample_spin.value()),
                "fast_3d": bool(self.fast_3d_check.isChecked()),
                "interactive_lod": bool(self.interactive_lod_check.isChecked()),
                "show_3d_colorbar": bool(self.show_colorbar_check.isChecked()),
                "reverse_track_axis": bool(self.reverse_track_axis_check.isChecked()),
                "reverse_track_axis_schema": 1,
                "ui_control_scale_pct": int(self.ui_control_scale_spin.value()),
                "ui_font_pt": float(self.ui_font_spin.value()),
                "playback_rows_per_sec": float(self.playback_speed_spin.value()),
                "qt_auto_layout": bool(self._auto_layout_enabled),
                "qt_window_size": [int(self.width()), int(self.height())],
                "qt_main_splitter_sizes": [int(v) for v in self.main_splitter.sizes()],
            }
        )

        try:
            geometry_b64 = bytes(self.saveGeometry().toBase64()).decode("ascii")
            payload["qt_window_geometry"] = geometry_b64
        except Exception:
            pass

        if (
            isinstance(self._settings_cache, dict)
            and payload == self._settings_cache
            and self._settings_path.exists()
        ):
            return

        try:
            tmp_path = self._settings_path.with_suffix(self._settings_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self._settings_path)
            self._settings_cache = payload
        except Exception as exc:
            self._set_status(f"Failed to save settings: {exc}")

    def _screen_available_size(self) -> tuple[int, int]:
        screen = None
        try:
            handle = self.windowHandle()
            if handle is not None:
                screen = handle.screen()
        except Exception:
            screen = None
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return self.PREFERRED_WINDOW_WIDTH, self.PREFERRED_WINDOW_HEIGHT
        geometry = screen.availableGeometry()
        return int(geometry.width()), int(geometry.height())

    def _apply_resolution_adaptive_layout(self, initial: bool, force: bool = False) -> None:
        if self._layout_busy:
            return
        if not force and not initial and not self._auto_layout_enabled:
            return
        if not hasattr(self, "controls_scroll") or not hasattr(self, "main_splitter"):
            return
        if not hasattr(self, "summary_text"):
            return

        self._layout_busy = True
        try:
            screen_w, screen_h = self._screen_available_size()
            eff_min_w = max(480, min(self.MIN_WINDOW_WIDTH, max(480, screen_w - 40)))
            eff_min_h = max(360, min(self.MIN_WINDOW_HEIGHT, max(360, screen_h - 40)))
            self.setMinimumSize(eff_min_w, eff_min_h)

            if initial:
                target_w = int(min(self.PREFERRED_WINDOW_WIDTH, screen_w * self.WINDOW_SCREEN_FRACTION))
                target_h = int(min(self.PREFERRED_WINDOW_HEIGHT, screen_h * self.WINDOW_SCREEN_FRACTION))
                target_w = max(eff_min_w, target_w)
                target_h = max(eff_min_h, target_h)
                self.resize(target_w, target_h)
                layout_w = target_w
                layout_h = target_h
            else:
                layout_w = max(eff_min_w, int(self.width()))
                layout_h = max(eff_min_h, int(self.height()))

            ctrl_w = int(
                self._clamp(
                    layout_w * self.CONTROL_PANEL_WIDTH_RATIO,
                    self.CONTROL_PANEL_MIN_WIDTH,
                    self.CONTROL_PANEL_MAX_WIDTH,
                )
            )
            self.controls_scroll.setMinimumWidth(self.CONTROL_PANEL_MIN_WIDTH)

            summary_h = int(
                self._clamp(
                    layout_h * self.SUMMARY_HEIGHT_RATIO,
                    self.SUMMARY_MIN_HEIGHT,
                    self.SUMMARY_MAX_HEIGHT,
                )
            )
            self.summary_text.setMaximumHeight(summary_h)

            if initial:
                self.main_splitter.setSizes([ctrl_w, max(1, layout_w - ctrl_w)])
        finally:
            self._layout_busy = False
