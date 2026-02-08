"""Modularized Qt main window implementation for the CASALS GUI."""

from __future__ import annotations

from pathlib import Path

from .qt_window.render_mixin import QtMainWindowRenderMixin
from .qt_window.runtime import (
    MATPLOTLIB_IMPORT_ERROR,
    PYVISTA_IMPORT_ERROR,
    QT_IMPORT_ERROR,
    BaseMainWindow,
    QtCore,
    QtWidgets,
    pg,
)
from .qt_window.settings_mixin import QtMainWindowSettingsMixin
from .qt_window.ui_mixin import QtMainWindowUiMixin
from .tdms_processor import TDMS_IMPORT_ERROR, CasalsTdmsProcessor, TdmsMeta, tdms_available


class CASALSQtMainWindow(
    QtMainWindowUiMixin,
    QtMainWindowSettingsMixin,
    QtMainWindowRenderMixin,
    BaseMainWindow,
):
    SETTINGS_FILENAME = "casals_gui_settings.json"
    DEFAULT_PERCENTILE_CLIP = (0.5, 99.5)
    DEFAULT_ELEV = 32.0
    DEFAULT_AZIM = -58.0
    DEFAULT_X_VIS_SCALE = 1.0
    DEFAULT_Y_VIS_SCALE = 1.0
    DEFAULT_Z_SCALE = 1.0
    MAX_3D_FP = 128
    MAX_3D_SAMPLES = 320
    INTERACT_MAX_3D_FP = 72
    INTERACT_MAX_3D_SAMPLES = 180
    RENDER_DEBOUNCE_MS = 80
    PREFERRED_WINDOW_WIDTH = 1540
    PREFERRED_WINDOW_HEIGHT = 960
    MIN_WINDOW_WIDTH = 920
    MIN_WINDOW_HEIGHT = 620
    WINDOW_SCREEN_FRACTION = 0.94
    CONTROL_PANEL_WIDTH_RATIO = 0.24
    CONTROL_PANEL_MIN_WIDTH = 260
    CONTROL_PANEL_MAX_WIDTH = 420
    SUMMARY_HEIGHT_RATIO = 0.22
    SUMMARY_MIN_HEIGHT = 140
    SUMMARY_MAX_HEIGHT = 280
    AUTO_LAYOUT_DEFAULT = False
    MIN_UI_CONTROL_SCALE = 0.70
    MAX_UI_CONTROL_SCALE = 1.80
    DEFAULT_UI_CONTROL_SCALE_PERCENT = 100
    MIN_UI_FONT_PT = 7.0
    MAX_UI_FONT_PT = 24.0
    MIN_AXIS_SCALE = 0.05
    MAX_AXIS_SCALE = 20.0
    CUSTOM_CMAP_GREEN_BLACK_BLUE = "green_black_blue"
    CMAPS = (
        CUSTOM_CMAP_GREEN_BLACK_BLUE,
        "RdBu_r",
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "gray",
    )

    def __init__(self) -> None:
        if QT_IMPORT_ERROR is not None:
            raise RuntimeError(
                "PyQt5 is required for the Qt GUI.\n"
                "Install with: pip install PyQt5\n"
                f"Import error: {QT_IMPORT_ERROR}"
            )
        if MATPLOTLIB_IMPORT_ERROR is not None:
            raise RuntimeError(
                "matplotlib is required for 2D plotting.\n"
                "Install with: pip install matplotlib\n"
                f"Import error: {MATPLOTLIB_IMPORT_ERROR}"
            )
        if PYVISTA_IMPORT_ERROR is not None:
            raise RuntimeError(
                "pyvista + pyvistaqt are required for true embedded 3D.\n"
                "Install with: pip install pyvista pyvistaqt\n"
                f"Import error: {PYVISTA_IMPORT_ERROR}"
            )

        super().__init__()
        self.setWindowTitle("CASALS TDMS Viewer (Qt + PyVistaQt)")
        self._settings_path = Path.cwd() / self.SETTINGS_FILENAME
        self._settings_cache = self._load_settings()
        self._auto_layout_enabled = bool(
            self._settings_cache.get("qt_auto_layout", self.AUTO_LAYOUT_DEFAULT)
        )
        self._startup_row_id = self._to_int(self._settings_cache.get("row_id", 0), 0)
        self._view_mode = "2D"

        self.processor = CasalsTdmsProcessor()
        self.meta: TdmsMeta | None = None

        self._last_camera_position = None
        self._is_syncing_camera = False
        self._pv_mesh_actor = None
        self._last_axis_scale = (1.0, 1.0, 1.0)
        self._last_3d_axes_ranges = None
        self._layout_initialized = False
        self._layout_busy = False
        self._interactive_2d_enabled = pg is not None
        self.figure = None
        self.canvas = None
        self.pg_plot = None
        self.pg_image_item = None
        self._default_ui_font_pt = self._to_float(QtWidgets.QApplication.font().pointSizeF(), 10.0)
        if self._default_ui_font_pt <= 0.0:
            self._default_ui_font_pt = 10.0

        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(lambda: self.render_current_row(show_errors=False))

        self._camera_poll_timer = QtCore.QTimer(self)
        self._camera_poll_timer.setInterval(120)
        self._camera_poll_timer.timeout.connect(self._sync_camera_from_pyvista)
        self._camera_poll_timer.start()

        self._build_ui()
        restored_layout = self._restore_settings()
        if not restored_layout:
            self._apply_resolution_adaptive_layout(initial=True)

        if not tdms_available():
            self._set_status("nptdms is missing. Install with: pip install nptdms")
            self._set_summary(
                "Cannot load TDMS because nptdms is not installed.\n"
                f"Import error: {TDMS_IMPORT_ERROR}\n\n"
                "Install dependency with:\n"
                "pip install nptdms"
            )

__all__ = ["CASALSQtMainWindow"]
