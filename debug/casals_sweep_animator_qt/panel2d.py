"""Two-dimensional cross-section visualization panel for CASALS 3D Sweep Animator.

Scientific boundaries & requirements:
- Displays physical (cross-track, height) or (track, height) waveform heatmaps.
- Shows official Refh overlay on top of reconstructed waveform response.
- Shows reconstructed argmax positions with distinct symbol for closure comparison.
- Provides scientific colormaps (Viridis, Inferno, Turbo, Grayscale) and units.
- Supports stable color range across playback to prevent visual flickering.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .cross_section import CrossSectionGrid


class Panel2DWidget(QtWidgets.QWidget):
    """2D panel rendering the synchronized waveform vertical cross-section and overlays."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)

        # Settings
        self.show_refh_overlay = True
        self.show_argmax_overlay = True
        self.colormap_name = "turbo"
        self.color_range_mode = "fixed"  # 'fixed' or 'auto'
        self.fixed_vmin = 0.0
        self.fixed_vmax = 500.0
        self.auto_percentiles = (1.0, 99.5)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # pyqtgraph GraphicsLayoutWidget
        self.glw = pg.GraphicsLayoutWidget(parent=self)
        self.glw.setBackground((20, 24, 30))
        layout.addWidget(self.glw)

        # Plot item
        self.plot: pg.PlotItem = self.glw.addPlot(row=0, col=0)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "WGS84 Ellipsoidal Height", units="m")
        self.plot.setLabel("bottom", "Cross-track Distance", units="m")

        # Heatmap image item (axisOrder='col-major' means data[x, y])
        self.image_item = pg.ImageItem(axisOrder="col-major")
        self.plot.addItem(self.image_item)

        # ColorBarItem
        self.colorbar = pg.ColorBarItem(
            values=(self.fixed_vmin, self.fixed_vmax),
            colorMap=self.colormap_name,
            width=16,
            interactive=True,
        )
        self.colorbar.setImageItem(self.image_item)
        self.glw.addItem(self.colorbar, row=0, col=1)

        # Overlays
        # 1. Official Refh overlay (white circles with black contour)
        self.refh_overlay = pg.PlotDataItem(
            pen=pg.mkPen(color=(255, 255, 255, 200), width=1.5),
            symbol="o",
            symbolSize=4,
            symbolPen=pg.mkPen(color=(0, 0, 0, 220), width=1.0),
            symbolBrush=pg.mkBrush(255, 255, 255, 240),
            name="Official Refh",
        )
        self.plot.addItem(self.refh_overlay)

        # 2. Reconstructed argmax overlay (bright orange triangles)
        self.argmax_overlay = pg.PlotDataItem(
            pen=None,
            symbol="t",
            symbolSize=5,
            symbolPen=pg.mkPen(color=(0, 0, 0, 200), width=1.0),
            symbolBrush=pg.mkBrush(255, 140, 20, 240),
            name="Reconstructed Argmax",
        )
        self.plot.addItem(self.argmax_overlay)

        # Legend
        self.legend = self.plot.addLegend(offset=(10, 10))
        self.legend.addItem(self.refh_overlay, "Official Refh")
        self.legend.addItem(self.argmax_overlay, "Reconstructed Argmax (H1/H2)")

    def set_colormap(self, name: str) -> None:
        """Update colormap."""
        self.colormap_name = str(name).lower()
        try:
            self.colorbar.setColorMap(self.colormap_name)
        except Exception:
            self.colorbar.setColorMap("turbo")

    def set_fixed_range(self, vmin: float, vmax: float) -> None:
        """Set the stable fixed color range."""
        self.fixed_vmin = float(vmin)
        self.fixed_vmax = float(vmax)
        if self.fixed_vmax <= self.fixed_vmin:
            self.fixed_vmax = self.fixed_vmin + 1.0

    def update_cross_section(self, grid: CrossSectionGrid) -> None:
        """Render the 2D cross-section grid and overlays."""
        # Update Axis Labels
        if grid.x_axis_mode == "meters":
            self.plot.setLabel("bottom", "Cross-track Distance", units="m")
        else:
            self.plot.setLabel("bottom", "Track Index", units="")

        # Title
        title_text = (
            f"<b>2D Waveform Cross-Section</b> | {grid.color_title} ({grid.color_unit}) "
            f"| Mapping: <b>{grid.mapping_convention}</b>"
        )
        self.plot.setTitle(title_text, color="#e0e4eb", size="10pt")

        # Determine Levels
        finite_vals = grid.grid_data[np.isfinite(grid.grid_data)]
        if self.color_range_mode == "auto" and len(finite_vals) > 0:
            lo_p, hi_p = self.auto_percentiles
            vmin = float(np.percentile(finite_vals, lo_p))
            vmax = float(np.percentile(finite_vals, hi_p))
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin = self.fixed_vmin
            vmax = self.fixed_vmax

        # Update Image Data & Geometry
        self.image_item.setImage(grid.grid_data, autoLevels=False, levels=(vmin, vmax))
        rect = QtCore.QRectF(
            grid.x_min,
            grid.h_min,
            grid.x_max - grid.x_min,
            grid.h_max - grid.h_min,
        )
        self.image_item.setRect(rect)
        self.colorbar.setLevels((vmin, vmax))

        # Update Official Refh Overlay
        if self.show_refh_overlay and len(grid.refh_x) > 0:
            valid = grid.refh_valid & np.isfinite(grid.refh_x) & np.isfinite(grid.refh_h)
            if np.any(valid):
                # Sort by X for clean polyline connection
                order = np.argsort(grid.refh_x[valid])
                x_sorted = grid.refh_x[valid][order]
                h_sorted = grid.refh_h[valid][order]
                self.refh_overlay.setData(x=x_sorted, y=h_sorted)
                self.refh_overlay.setVisible(True)
            else:
                self.refh_overlay.setVisible(False)
        else:
            self.refh_overlay.setVisible(False)

        # Update Reconstructed Argmax Overlay
        if self.show_argmax_overlay and grid.argmax_x is not None and grid.argmax_h is not None:
            valid_arg = np.isfinite(grid.argmax_x) & np.isfinite(grid.argmax_h)
            if np.any(valid_arg):
                self.argmax_overlay.setData(x=grid.argmax_x[valid_arg], y=grid.argmax_h[valid_arg])
                self.argmax_overlay.setVisible(True)
            else:
                self.argmax_overlay.setVisible(False)
        else:
            self.argmax_overlay.setVisible(False)

