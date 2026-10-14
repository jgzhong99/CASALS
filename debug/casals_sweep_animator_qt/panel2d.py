"""PyQtGraph renderer for the synchronized RX waveform cross-section."""

from __future__ import annotations

from typing import Optional
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from .cross_section import CrossSectionGrid


class Panel2DWidget(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.show_refh_overlay = True
        self.show_argmax_overlay = False
        self.colormap_name = "viridis"
        self.color_range_mode = "stable"
        self.auto_percentiles = (1.0, 99.5)
        self.color_ranges = {
            "raw": (0.0, 2000.0),
            "minus_bg": (0.0, 500.0),
            "asinh": (0.0, 5.0),
            "standardized": (0.0, 10.0),
        }
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.message_label = QtWidgets.QLabel("")
        self.message_label.setStyleSheet("background: #151c24; color: #e9b86e; padding: 3px 8px;")
        self.message_label.setVisible(False)
        layout.addWidget(self.message_label)

        self.glw = pg.GraphicsLayoutWidget(parent=self)
        self.glw.setBackground((18, 23, 30))
        layout.addWidget(self.glw, 1)
        self.plot: pg.PlotItem = self.glw.addPlot(row=0, col=0)
        self.plot.showGrid(x=True, y=True, alpha=0.20)
        self.plot.setLabel("left", "WGS84 ellipsoidal height", units="m")
        self.image_item = pg.ImageItem(axisOrder="col-major")
        self.plot.addItem(self.image_item)

        self.colorbar = pg.ColorBarItem(values=self.color_ranges["minus_bg"], colorMap=self.colormap_name, width=18)
        self.colorbar.setImageItem(self.image_item)
        self.glw.addItem(self.colorbar, row=0, col=1)

        self.refh_overlay = pg.PlotDataItem(
            pen=None,
            symbol="o",
            symbolSize=3.5,
            symbolPen=pg.mkPen(color=(245, 247, 250, 220), width=0.8),
            symbolBrush=pg.mkBrush(245, 247, 250, 220),
        )
        self.argmax_overlay = pg.PlotDataItem(
            pen=None,
            symbol="t",
            symbolSize=5,
            symbolPen=pg.mkPen(color=(16, 20, 25, 230), width=0.8),
            symbolBrush=pg.mkBrush(240, 145, 55, 230),
        )
        self.plot.addItem(self.refh_overlay)
        self.plot.addItem(self.argmax_overlay)

    def set_colormap(self, name: str) -> None:
        self.colormap_name = str(name).lower()
        self.colorbar.setColorMap(self.colormap_name)

    def set_color_ranges(self, ranges: dict[str, tuple[float, float]]) -> None:
        self.color_ranges.update(ranges)

    def set_color_range_mode(self, mode: str) -> None:
        if mode not in {"stable", "local"}:
            raise ValueError("color range mode must be stable or local")
        self.color_range_mode = mode

    def _levels(self, grid: CrossSectionGrid, finite: np.ndarray) -> tuple[float, float]:
        stable = self.color_ranges.get(grid.mode.lower(), self.color_ranges["minus_bg"])
        if self.color_range_mode == "local" and finite.size:
            lo, hi = np.percentile(finite, self.auto_percentiles)
            lo, hi = float(lo), float(hi)
        else:
            lo, hi = map(float, stable)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        return lo, hi

    def update_cross_section(self, grid: CrossSectionGrid) -> None:
        if grid.view == "curtain":
            self.plot.setLabel("bottom", "Track index")
            view_name = "Waveform Curtain"
        else:
            self.plot.setLabel("bottom", "Cross-track distance", units="m")
            view_name = "Georeferenced Cross-Section"

        self.plot.setLabel("left", "WGS84 ellipsoidal height", units="m")
        title = f"<b>{view_name}</b>  ·  {grid.color_title} ({grid.color_unit})  ·  Mapping {grid.mapping_convention}"
        if not grid.available:
            title = f"<b>{view_name}</b>  ·  <span style='color:#e9b86e'>Unavailable: {grid.message}</span>"
        self.plot.setTitle(title, color="#e4eaf0", size="10pt")
        self.message_label.setText(grid.message)
        self.message_label.setVisible(bool(grid.message) and not grid.available)

        finite = grid.grid_data[np.isfinite(grid.grid_data)]
        levels = self._levels(grid, finite)
        self.image_item.setImage(grid.grid_data, autoLevels=False, levels=levels)
        self.image_item.setRect(QtCore.QRectF(
            grid.x_min,
            grid.h_min,
            grid.x_max - grid.x_min,
            grid.h_max - grid.h_min,
        ))
        self.colorbar.setLevels(levels)
        self.colorbar.setLabel("right", f"{grid.color_title} ({grid.color_unit})")

        valid_refh = grid.refh_valid & np.isfinite(grid.refh_x) & np.isfinite(grid.refh_h)
        self.refh_overlay.setData(x=grid.refh_x[valid_refh], y=grid.refh_h[valid_refh])
        self.refh_overlay.setVisible(bool(self.show_refh_overlay and np.any(valid_refh)))

        if self.show_argmax_overlay and grid.argmax_x is not None and grid.argmax_h is not None:
            valid = np.isfinite(grid.argmax_x) & np.isfinite(grid.argmax_h)
            self.argmax_overlay.setData(x=grid.argmax_x[valid], y=grid.argmax_h[valid])
            self.argmax_overlay.setVisible(bool(np.any(valid)))
        else:
            self.argmax_overlay.setVisible(False)

