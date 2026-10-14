"""Interactive 3D spatial renderer for the CASALS sweep animator (Pure Qt, Zero OpenGL).

Renders true 3D point clouds, schematic aircraft, flight trajectory, measurement beams,
and 3D coordinate axes using a high-performance CPU-projected 3D camera pipeline with
interactive mouse orbit, pan, and zoom.

Completely free of OpenGL, QOpenGLWidget, and graphics driver state pollution.
"""

from __future__ import annotations

from typing import Optional, Sequence
import time
import matplotlib
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .h5_model import SweepData


def _surface_mesh(sections: Sequence[tuple[float, float, float, float]], segments: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth closed fuselage or engine from elliptical cross sections."""
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    vertices = []
    for y, rx, rz, zc in sections:
        vertices.extend(np.column_stack([rx * np.cos(theta), np.full(segments, y), zc + rz * np.sin(theta)]))
    faces = []
    for ring in range(len(sections) - 1):
        a = ring * segments
        b = (ring + 1) * segments
        for j in range(segments):
            k = (j + 1) % segments
            faces.extend([[a + j, b + j, b + k], [a + j, b + k, a + k]])
    faces.extend([[0, j + 1, j] for j in range(segments - 1)])
    last = (len(sections) - 1) * segments
    faces.extend([[last, last + j, last + j + 1] for j in range(1, segments - 1)])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def _wing_mesh(span: float, chord_root: float, chord_tip: float, z_center: float, sweep_back: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a mirrored thick wing from spanwise airfoil stations."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    stations = [0.45, span * 0.55, span]
    thick = [0.16, 0.10, 0.035]
    for sign in (1.0, -1.0):
        base = len(vertices)
        for distance, thickness in zip(stations, thick):
            frac = distance / span
            chord = chord_root + (chord_tip - chord_root) * frac
            leading_y = 1.1 - sweep_back * frac
            x = sign * distance
            vertices.extend([
                [x, leading_y, z_center + thickness],
                [x, leading_y - chord, z_center + thickness * 0.65],
                [x, leading_y - chord, z_center - thickness * 0.65],
                [x, leading_y, z_center - thickness],
            ])
        for station in range(len(stations) - 1):
            a = base + station * 4
            b = a + 4
            faces.extend([
                [a, b, b + 1], [a, b + 1, a + 1],
                [a + 3, a + 2, b + 2], [a + 3, b + 2, b + 3],
                [a, a + 1, a + 2], [a, a + 2, a + 3],
                [b, b + 3, b + 2], [b, b + 2, b + 1],
                [a, a + 3, b + 3], [a, b + 3, b],
                [a + 1, b + 1, b + 2], [a + 1, b + 2, a + 2],
            ])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def _fin_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray([
        [-0.20, -6.8, 0.45], [0.20, -6.8, 0.45], [-0.18, -9.4, 0.45], [0.18, -9.4, 0.45],
        [-0.12, -7.4, 3.3], [0.12, -7.4, 3.3], [-0.10, -9.2, 3.3], [0.10, -9.2, 3.3],
    ], dtype=np.float32)
    faces = np.asarray([
        [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5],
        [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],
        [0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6],
    ], dtype=np.uint32)
    return vertices, faces


def _ellipsoid(center: tuple[float, float, float], radii: tuple[float, float, float], n_lat: int = 8, n_lon: int = 12) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    for i in range(n_lat + 1):
        phi = -0.5 * np.pi + np.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * np.pi * j / n_lon
            vertices.append([
                center[0] + radii[0] * np.cos(phi) * np.cos(theta),
                center[1] + radii[1] * np.cos(phi) * np.sin(theta),
                center[2] + radii[2] * np.sin(phi),
            ])
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + (j + 1) % n_lon
            d = (i + 1) * n_lon + j
            faces.extend([[a, b, c], [a, c, d]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def create_aircraft_mesh_parts() -> dict[str, tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]]:
    """Return smooth fuselage, thick lifting surfaces, canopy, and engine meshes."""
    return {
        "Fuselage": (_surface_mesh([
            (10.2, 0.08, 0.08, 0.05), (8.5, 0.42, 0.42, 0.0), (6.0, 0.72, 0.72, 0.0),
            (2.5, 0.92, 0.88, 0.0), (-2.0, 0.92, 0.84, 0.0), (-6.5, 0.58, 0.52, 0.08),
            (-9.5, 0.24, 0.28, 0.18), (-10.2, 0.04, 0.05, 0.2),
        ]), (0.83, 0.86, 0.89, 1.0)),
        "Main wings": (_wing_mesh(12.0, 4.6, 1.1, -0.15, 3.1), (0.70, 0.75, 0.81, 1.0)),
        "Tailplane": (_wing_mesh(4.8, 2.2, 0.7, 0.18, 0.7), (0.76, 0.80, 0.85, 1.0)),
        "Fin": (_fin_mesh(), (0.88, 0.40, 0.20, 1.0)),
        "Canopy": (_ellipsoid((0.0, 5.9, 0.54), (0.50, 1.25, 0.50)), (0.12, 0.20, 0.27, 1.0)),
        "Left engine": (_surface_mesh([(-1.6, 0.35, 0.34, -0.15), (-0.6, 0.44, 0.40, -0.15), (1.2, 0.42, 0.38, -0.15), (1.6, 0.2, 0.2, -0.15)]), (0.53, 0.58, 0.64, 1.0)),
        "Right engine": (_surface_mesh([(-1.6, 0.35, 0.34, -0.15), (-0.6, 0.44, 0.40, -0.15), (1.2, 0.42, 0.38, -0.15), (1.6, 0.2, 0.2, -0.15)]), (0.53, 0.58, 0.64, 1.0)),
    }


def aircraft_transform_matrix(
    instrument_enu: np.ndarray,
    heading_deg: float,
    model_scale: float = 1.0,
    display_exaggeration: float = 1.0,
) -> np.ndarray:
    """Return a uniformly scaled row-vector transform for the aircraft model."""
    psi = np.deg2rad(heading_deg)
    c, s = np.cos(psi), np.sin(psi)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32) * model_scale * display_exaggeration
    matrix[3, :3] = instrument_enu
    return matrix


def enu_axis_segments(origin: Sequence[float], length_m: float) -> np.ndarray:
    """Return equal-length positive X/Y/Z axes anchored at an ENU point."""
    anchor = np.asarray(origin, dtype=np.float64)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("axis origin must be a finite ENU 3-vector")
    length = float(length_m)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("axis length must be finite and positive")
    endpoints = anchor + np.eye(3, dtype=np.float64) * length
    return np.stack((np.broadcast_to(anchor, (3, 3)), endpoints), axis=1).reshape(6, 3).astype(np.float32)


def build_view_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix from ENU world coordinates to camera coordinates.

    Camera frame: +X points right, +Y points up on screen, +Z points towards viewer.
    """
    azim = np.deg2rad(float(azimuth_deg))
    elev = np.deg2rad(float(elevation_deg))

    # Camera forward vector (pointing from camera to target)
    cx = np.cos(elev) * np.sin(azim)
    cy = np.cos(elev) * np.cos(azim)
    cz = np.sin(elev)
    forward = -np.array([cx, cy, cz], dtype=np.float32)

    # World up (+Z)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    norm_r = np.linalg.norm(right)
    if norm_r < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right /= norm_r

    up = np.cross(right, forward)
    norm_u = np.linalg.norm(up)
    if norm_u > 1e-6:
        up /= norm_u

    return np.vstack([right, up, -forward])


class _EventHook:
    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, cb) -> None:
        if cb not in self._callbacks:
            self._callbacks.append(cb)

    def disconnect(self, cb) -> None:
        if cb in self._callbacks:
            self._callbacks.remove(cb)

    def emit(self, event=None) -> None:
        for cb in list(self._callbacks):
            try:
                cb(event)
            except Exception:
                pass


class _CanvasEvents:
    def __init__(self) -> None:
        self.draw = _EventHook()


class CanvasProxy:
    def __init__(self, widget: QtWidgets.QWidget, events: _CanvasEvents) -> None:
        self.view = widget
        self.native = widget
        self.events = events

    def update(self) -> None:
        self.view.update()


class CameraProxy:
    """Exposes 3D camera properties (center, azimuth, elevation, scale_factor)."""

    def __init__(self, scene: Scene3DWidget) -> None:
        self._scene = scene

    @property
    def center(self) -> tuple[float, float, float]:
        return (float(self._scene._center[0]), float(self._scene._center[1]), float(self._scene._center[2]))

    @center.setter
    def center(self, val: Sequence[float]) -> None:
        self._scene._center = np.asarray(val, dtype=np.float64)
        self._scene._reproject_all()

    @property
    def azimuth(self) -> float:
        return float(self._scene.azimuth)

    @azimuth.setter
    def azimuth(self, val: float) -> None:
        self._scene.azimuth = float(val) % 360.0
        self._scene._reproject_all()

    @property
    def elevation(self) -> float:
        return float(self._scene.elevation)

    @elevation.setter
    def elevation(self, val: float) -> None:
        self._scene.elevation = float(np.clip(val, -89.0, 89.0))
        self._scene._reproject_all()

    @property
    def scale_factor(self) -> float:
        return float(self._scene.scale_factor)

    @scale_factor.setter
    def scale_factor(self, val: float) -> None:
        self._scene.scale_factor = float(max(10.0, val))
        self._scene._reproject_all()

    @property
    def distance(self) -> float:
        return float(self.scale_factor)

    @distance.setter
    def distance(self, val: float) -> None:
        self.scale_factor = float(val)


class Interactive3DPlotWidget(pg.PlotWidget):
    """PyQtGraph 2D plot widget extended with intuitive 3D mouse orbit, pan, and zoom."""

    def __init__(self, scene: Scene3DWidget, parent=None) -> None:
        super().__init__(parent=parent)
        self._scene = scene
        self.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
        self.getPlotItem().getViewBox().setMenuEnabled(False)
        self._drag_start: Optional[QtCore.QPoint] = None
        self._drag_mode: Optional[str] = None

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag_start = ev.pos()
            self._drag_mode = "orbit"
        elif ev.button() in (QtCore.Qt.RightButton, QtCore.Qt.MiddleButton):
            self._drag_start = ev.pos()
            self._drag_mode = "pan"
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._drag_start is not None and self._drag_mode == "orbit":
            diff = ev.pos() - self._drag_start
            self._drag_start = ev.pos()
            self._scene.azimuth = (self._scene.azimuth - diff.x() * 0.4) % 360.0
            self._scene.elevation = float(np.clip(self._scene.elevation + diff.y() * 0.4, -89.0, 89.0))
            self._scene.camera_mode = "free"
            self._scene._reproject_all()
        elif self._drag_start is not None and self._drag_mode == "pan":
            diff = ev.pos() - self._drag_start
            self._drag_start = ev.pos()
            R = self._scene._current_R
            scale_m = self._scene.scale_factor * 2.0 / max(1.0, float(self.width()))
            dx_m = -float(diff.x()) * scale_m
            dy_m = float(diff.y()) * scale_m
            pan_world = R[0] * dx_m + R[1] * dy_m
            self._scene._center += pan_world.astype(np.float64)
            self._scene.camera_mode = "free"
            self._scene._reproject_all()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        self._drag_start = None
        self._drag_mode = None
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        self._scene.reset_view()
        ev.accept()

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        delta = ev.angleDelta().y()
        factor = 0.85 if delta > 0 else 1.18
        self._scene.scale_factor = float(np.clip(self._scene.scale_factor * factor, 20.0, 100000.0))
        self._scene._reproject_all()
        ev.accept()


class Scene3DWidget(QtWidgets.QWidget):
    """Pure Qt interactive 3D spatial viewer for CASALS sweeps (Zero OpenGL)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.aircraft_model_scale = 1.0
        self.aircraft_display_exaggeration = 3.0
        self.camera_mode = "follow"
        self.beam_density = "16"
        self.point_color_mode = "height"
        self.point_colormap = "viridis"
        self.show_aircraft = True
        self.show_trajectory = True
        self.show_history_refh = True
        self.show_current_refh = True
        self.show_beams = True
        self.visible_history_count = 0
        self.last_history_upload_ms = 0.0
        self.last_history_upload_bytes = 0

        # Camera & spatial projection state
        self.azimuth: float = 42.0
        self.elevation: float = 28.0
        self.scale_factor: float = 1200.0
        self.z_scale: float = 0.25
        self._center = np.zeros(3, dtype=np.float64)
        self._current_R = build_view_matrix(self.azimuth, self.elevation)

        self._history_sources: dict[str, Optional[np.ndarray]] = {}
        self._history_positions = np.empty((0, 3), dtype=np.float32)
        self._history_brushes: Optional[list[QtGui.QBrush]] = None
        self._history_ranges = {"height": (0.0, 1000.0), "snr": (0.0, 50.0), "amp": (0.0, 1000.0)}
        self._history_prefix: Optional[np.ndarray] = None
        self._trajectory = np.empty((0, 3), dtype=np.float32)
        self._overview_center = np.zeros(3, dtype=np.float64)
        self._overview_span = 1200.0
        self._axis_origin_enu = np.zeros(3, dtype=np.float64)
        self._axis_reference_m = 100.0
        self._last_sweep: Optional[SweepData] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top overlay banner
        self.overlay = QtWidgets.QLabel("3D View (No OpenGL): Left Drag to Orbit · Right Drag to Pan · Scroll to Zoom", self)
        self.overlay.setStyleSheet("background: rgba(12, 17, 23, 210); color: #c6d2df; padding: 4px 8px; font-size: 9pt;")
        self.overlay.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.overlay, 0)

        # 3D interactive plot widget
        self.plot_widget = Interactive3DPlotWidget(self, parent=self)
        self.plot_widget.setBackground("#11171e")
        self.plot = self.plot_widget.getPlotItem()
        self.plot.setAspectLocked(True, ratio=1.0)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setLabel("bottom", "3D Camera X", units="m", color="#9fb0c2")
        self.plot.setLabel("left", "3D Camera Y", units="m", color="#9fb0c2")
        layout.addWidget(self.plot_widget, 1)

        # Event hook and proxies
        self._events = _CanvasEvents()
        self.canvas = CanvasProxy(self.plot_widget, self._events)
        self.camera = CameraProxy(self)
        self.view = self.plot_widget

        # 1. 3D Trajectory items
        self.trajectory_full_item = pg.PlotDataItem(
            pen=pg.mkPen(color=(50, 85, 120, 160), width=1.5, style=QtCore.Qt.DashLine)
        )
        self.plot.addItem(self.trajectory_full_item)

        self.trajectory_past_item = pg.PlotDataItem(
            pen=pg.mkPen(color=(70, 200, 240, 220), width=2.5)
        )
        self.plot.addItem(self.trajectory_past_item)

        # 2. 3D Measurement beam lines
        self.beam_lines_item = pg.PlotDataItem(
            pen=pg.mkPen(color=(60, 220, 240, 180), width=1.2)
        )
        self.beam_lines_item.last_upload_ms = 0.0
        self.plot.addItem(self.beam_lines_item)

        # 3. 3D Ground point clouds
        self.history_scatter = pg.ScatterPlotItem(
            size=3,
            pen=None,
            pxMode=True,
        )
        self.plot.addItem(self.history_scatter)

        self.current_scatter = pg.ScatterPlotItem(
            size=6,
            pen=pg.mkPen(255, 255, 255, 220, width=1),
            brush=pg.mkBrush(255, 210, 50, 240),
            pxMode=True,
        )
        self.current_scatter.last_upload_ms = 0.0
        self.plot.addItem(self.current_scatter)

        # 4. 3D Aircraft wireframe & center marker
        self._build_base_aircraft_wireframe()
        self.aircraft_wire_item = pg.PlotDataItem(
            pen=pg.mkPen(color=(255, 225, 80, 240), width=1.8)
        )
        self.plot.addItem(self.aircraft_wire_item)

        self.aircraft_marker = pg.ScatterPlotItem(
            size=9,
            symbol="star",
            pen=pg.mkPen(255, 255, 255, 240, width=1),
            brush=pg.mkBrush(255, 230, 80, 255),
            pxMode=True,
        )
        self.plot.addItem(self.aircraft_marker)

        # 5. 3D Tripod axes (Red=East, Green=North, Blue=Up)
        self.axis_east = pg.PlotDataItem(pen=pg.mkPen(color=(240, 70, 70, 240), width=2.5))
        self.axis_north = pg.PlotDataItem(pen=pg.mkPen(color=(60, 220, 90, 240), width=2.5))
        self.axis_up = pg.PlotDataItem(pen=pg.mkPen(color=(60, 150, 245, 240), width=2.5))
        self.plot.addItem(self.axis_east)
        self.plot.addItem(self.axis_north)
        self.plot.addItem(self.axis_up)

        # Compatibility aliases
        self.current_visual = self.current_scatter
        self.beam_visual = self.beam_lines_item
        self.history_visual = self.history_scatter
        self.trajectory_visual = self.trajectory_past_item
        self.axis_visual = self.axis_east
        self.aircraft_visuals = [self.aircraft_wire_item, self.aircraft_marker]

        self._apply_visibility()

    def _build_base_aircraft_wireframe(self) -> None:
        parts = create_aircraft_mesh_parts()
        verts_list, edges_list = [], []
        v_offset = 0
        for name, (mesh, _color) in parts.items():
            v, f = mesh
            if name.endswith("engine"):
                side = -2.25 if name.startswith("Left") else 2.25
                v = v.copy()
                v[:, 0] += side
            verts_list.append(v)
            for face in f:
                edges_list.extend([
                    [face[0] + v_offset, face[1] + v_offset],
                    [face[1] + v_offset, face[2] + v_offset],
                    [face[2] + v_offset, face[0] + v_offset]
                ])
            v_offset += len(v)
        self._base_aircraft_verts = np.vstack(verts_list).astype(np.float32)
        self._aircraft_edges = np.array(edges_list, dtype=np.int32)

    def _compute_history_colors(self) -> None:
        if len(self._history_positions) == 0:
            self._history_brushes = []
            return

        values = self._history_sources.get(self.point_color_mode)
        if self.point_color_mode == "uniform" or values is None:
            self._history_brushes = None
            return

        lo, hi = self._history_ranges.get(self.point_color_mode, (0.0, 1.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0

        val_arr = np.asarray(values, dtype=np.float32)
        norm = np.clip((val_arr - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        norm[~np.isfinite(norm)] = 0.0

        cmap_name = self.point_colormap if self.point_color_mode == "height" else {"snr": "cividis", "amp": "magma"}.get(self.point_color_mode, "viridis")
        try:
            cmap = matplotlib.colormaps[cmap_name]
        except KeyError:
            cmap = matplotlib.colormaps["viridis"]

        rgba_uint8 = (cmap(norm) * 255).astype(np.uint8)
        rgba_uint8[:, 3] = 160
        self._history_brushes = np.array([pg.mkBrush(r, g, b, a) for r, g, b, a in rgba_uint8], dtype=object)

    def set_history_data(
        self,
        positions: np.ndarray,
        sweep_indices: np.ndarray,
        prefix_count: np.ndarray,
        heights: np.ndarray,
        snr: Optional[np.ndarray],
        amplitude: Optional[np.ndarray],
        ranges: dict[str, tuple[float, float]],
        trajectory: np.ndarray,
    ) -> None:
        t0 = time.perf_counter()
        self._history_sources = {"height": heights, "snr": snr, "amp": amplitude}
        self._history_positions = np.ascontiguousarray(positions, dtype=np.float32)
        self._history_ranges.update(ranges)
        self._history_prefix = prefix_count
        self._trajectory = np.ascontiguousarray(trajectory, dtype=np.float32)

        self._compute_history_colors()

        self.last_history_upload_ms = (time.perf_counter() - t0) * 1000.0
        self.last_history_upload_bytes = int(self._history_positions.nbytes)

        self._set_overview_bounds()
        self._reproject_all()

    def set_point_color_mode(self, mode: str) -> None:
        mode = mode.lower()
        if mode not in {"height", "snr", "amp", "uniform"}:
            raise ValueError(f"unknown point color mode: {mode}")
        self.point_color_mode = mode
        self._recolor_history()

    def _recolor_history(self) -> None:
        t0 = time.perf_counter()
        self._compute_history_colors()
        self.last_history_upload_ms = (time.perf_counter() - t0) * 1000.0
        self._reproject_all()

    def set_point_colormap(self, name: str) -> None:
        self.point_colormap = str(name).lower()
        if self.point_color_mode == "height":
            self._recolor_history()

    def set_camera_mode(self, mode: str) -> None:
        mode = mode.lower()
        if mode not in {"follow", "overview", "free"}:
            raise ValueError("camera mode must be follow, overview, or free")
        self.camera_mode = mode
        if mode == "overview":
            self._set_overview_camera()
        elif mode == "follow" and self._last_sweep is not None:
            self._update_follow_camera(self._last_sweep)

    def set_z_scale(self, scale: float) -> None:
        self.z_scale = float(max(0.01, min(10.0, scale)))
        self._set_overview_bounds()
        if self.camera_mode == "follow" and self._last_sweep is not None:
            self._update_follow_camera(self._last_sweep)
        elif self.camera_mode == "overview":
            self._set_overview_camera()
        else:
            self._reproject_all()

    def _set_overview_bounds(self) -> None:
        if not self._history_positions.size:
            if len(self._trajectory):
                finite = self._trajectory[np.isfinite(self._trajectory).all(axis=1)]
                if len(finite):
                    min_xyz = finite.min(axis=0).astype(np.float64)
                    max_xyz = finite.max(axis=0).astype(np.float64)
                    self._overview_center = 0.5 * (min_xyz + max_xyz)
                    self._overview_center[2] *= self.z_scale
                    self._overview_span = max(float(np.max(max_xyz[:2] - min_xyz[:2])), 500.0)
            return

        finite = self._history_positions[np.isfinite(self._history_positions).all(axis=1)]
        if not len(finite):
            return
        # Center directly on ground point cloud
        min_xyz = finite.min(axis=0).astype(np.float64)
        max_xyz = finite.max(axis=0).astype(np.float64)
        center_raw = 0.5 * (min_xyz + max_xyz)
        self._overview_center = np.array([center_raw[0], center_raw[1], center_raw[2] * self.z_scale], dtype=np.float64)
        self._overview_span = max(float(np.max(max_xyz[:2] - min_xyz[:2])), 500.0)

    def _set_overview_camera(self) -> None:
        self._center = self._overview_center.copy()
        self.scale_factor = self._overview_span * 0.75
        self.azimuth = 42.0
        self.elevation = 28.0
        self._reproject_all()

    def reset_view(self, center: Optional[Sequence[float]] = None, distance: Optional[float] = None) -> None:
        if center is None:
            self._set_overview_camera()
            return
        c = np.asarray(center, dtype=np.float64).copy()
        if len(c) == 3:
            c[2] *= self.z_scale
        self._center = c
        self.scale_factor = float(distance or max(self._overview_span * 0.5, 300.0))
        self.azimuth = 42.0
        self.elevation = 28.0
        self._reproject_all()

    def _apply_visibility(self) -> None:
        self.history_scatter.setVisible(self.show_history_refh)
        self.trajectory_full_item.setVisible(self.show_trajectory)
        self.trajectory_past_item.setVisible(self.show_trajectory)
        self.beam_lines_item.setVisible(self.show_beams and self.beam_density != "none")
        self.current_scatter.setVisible(self.show_current_refh)
        self.aircraft_wire_item.setVisible(self.show_aircraft)
        self.aircraft_marker.setVisible(self.show_aircraft)

    def _update_follow_camera(self, sweep: SweepData) -> None:
        valid = sweep.valid_pulse_mask & np.isfinite(sweep.refh_enu).all(axis=1)
        if np.any(valid):
            swath = np.nanmean(sweep.refh_enu[valid], axis=0)
        else:
            swath = np.array([sweep.instrument_enu[0], sweep.instrument_enu[1], 0.0])
        # Camera center is strictly anchored on the ground point cloud
        self._center = np.array([swath[0], swath[1], swath[2] * self.z_scale], dtype=np.float64)

        inst_z = sweep.instrument_enu[2] * self.z_scale
        ground_z = swath[2] * self.z_scale
        vert_span = abs(inst_z - ground_z)
        self.scale_factor = max(vert_span * 0.75, 400.0)
        self._reproject_all()

    def _reproject_all(self) -> None:
        """Project all 3D scene elements through the 3D orbital camera matrix."""
        R = build_view_matrix(self.azimuth, self.elevation)
        self._current_R = R
        center = self._center
        sz = float(self.z_scale)

        # 1. 3D Ground point cloud
        count = min(self.visible_history_count, len(self._history_positions))
        if self.show_history_refh and count > 0:
            target_max = 35000
            if count > target_max:
                sub_idx = np.linspace(0, count - 1, target_max, dtype=np.int32)
            else:
                sub_idx = slice(0, count)
            pts = self._history_positions[sub_idx].copy()
            pts[:, 2] *= sz
            pts_cam = (pts - center) @ R.T
            if self._history_brushes is not None:
                brushes = self._history_brushes[sub_idx]
                self.history_scatter.setData(x=pts_cam[:, 0], y=pts_cam[:, 1], brush=brushes)
            else:
                self.history_scatter.setData(x=pts_cam[:, 0], y=pts_cam[:, 1], brush=pg.mkBrush(90, 170, 220, 140))
        else:
            self.history_scatter.clear()

        # 2. Current sweep points
        if self._last_sweep is not None and self.show_current_refh:
            sw = self._last_sweep
            valid = sw.valid_pulse_mask & np.isfinite(sw.refh_enu).all(axis=1)
            curr_pts = sw.refh_enu[valid].copy()
            if len(curr_pts) > 0:
                curr_pts[:, 2] *= sz
                curr_cam = (curr_pts - center) @ R.T
                self.current_scatter.setData(x=curr_cam[:, 0], y=curr_cam[:, 1])
                self.current_scatter.setVisible(True)
            else:
                self.current_scatter.clear()
                self.current_scatter.setVisible(False)
        else:
            self.current_scatter.clear()
            self.current_scatter.setVisible(False)

        # 3. 3D Measurement beams
        if self._last_sweep is not None and self.show_beams and self.beam_density != "none":
            sw = self._last_sweep
            valid = sw.valid_pulse_mask & np.isfinite(sw.refh_enu).all(axis=1)
            density = {"16": 16, "8": 8, "all": 1}.get(self.beam_density, 0)
            picks = np.flatnonzero(valid)
            if density > 1:
                picks = picks[::density]
            elif density == 0:
                picks = np.empty(0, dtype=np.int64)

            if len(picks) > 0:
                inst_pts = sw.pulse_instrument_enu[picks].copy()
                inst_pts[:, 2] *= sz
                refh_pts = sw.refh_enu[picks].copy()
                refh_pts[:, 2] *= sz
                p_inst = (inst_pts - center) @ R.T
                p_refh = (refh_pts - center) @ R.T
                beam_x = np.empty(len(picks) * 3, dtype=np.float32)
                beam_y = np.empty(len(picks) * 3, dtype=np.float32)
                beam_x[0::3] = p_inst[:, 0]
                beam_x[1::3] = p_refh[:, 0]
                beam_x[2::3] = np.nan
                beam_y[0::3] = p_inst[:, 1]
                beam_y[1::3] = p_refh[:, 1]
                beam_y[2::3] = np.nan
                self.beam_lines_item.setData(beam_x, beam_y)
                self.beam_lines_item.setVisible(True)
            else:
                self.beam_lines_item.setData([], [])
                self.beam_lines_item.setVisible(False)
        else:
            self.beam_lines_item.setData([], [])
            self.beam_lines_item.setVisible(False)

        # 4. 3D Trajectory
        if self.show_trajectory and len(self._trajectory) > 0:
            k = self._last_sweep.sweep_index if self._last_sweep is not None else len(self._trajectory) - 1
            past_pts = self._trajectory[:k + 1].copy()
            past_pts[:, 2] *= sz
            past_cam = (past_pts - center) @ R.T
            self.trajectory_past_item.setData(past_cam[:, 0], past_cam[:, 1])
            self.trajectory_past_item.setVisible(True)
        else:
            self.trajectory_past_item.setVisible(False)

        # 5. 3D Aircraft
        if self._last_sweep is not None and self.show_aircraft:
            sw = self._last_sweep
            inst_scaled = sw.instrument_enu.copy()
            inst_scaled[2] *= sz
            M = aircraft_transform_matrix(inst_scaled, sw.heading_deg, self.aircraft_model_scale, self.aircraft_display_exaggeration)
            world_verts = self._base_aircraft_verts @ M[:3, :3] + M[3, :3]
            cam_verts = (world_verts - center) @ R.T

            edges = self._aircraft_edges
            wire_x = np.empty(len(edges) * 3, dtype=np.float32)
            wire_y = np.empty(len(edges) * 3, dtype=np.float32)
            wire_x[0::3] = cam_verts[edges[:, 0], 0]
            wire_x[1::3] = cam_verts[edges[:, 1], 0]
            wire_x[2::3] = np.nan
            wire_y[0::3] = cam_verts[edges[:, 0], 1]
            wire_y[1::3] = cam_verts[edges[:, 1], 1]
            wire_y[2::3] = np.nan
            self.aircraft_wire_item.setData(wire_x, wire_y)
            self.aircraft_wire_item.setVisible(True)

            craft_pos = (inst_scaled - center) @ R.T
            self.aircraft_marker.setData(x=[craft_pos[0]], y=[craft_pos[1]])
            self.aircraft_marker.setVisible(True)
        else:
            self.aircraft_wire_item.setData([], [])
            self.aircraft_wire_item.setVisible(False)
            self.aircraft_marker.clear()
            self.aircraft_marker.setVisible(False)

        # 6. 3D Tripod Axes (Red=+East, Green=+North, Blue=+Up)
        axis_len = max(50.0, self.scale_factor * 0.15)
        self._axis_reference_m = axis_len
        self._axis_origin_enu = center.copy()
        tripod_enu = np.array([
            [0, 0, 0], [axis_len, 0, 0],
            [0, 0, 0], [0, axis_len, 0],
            [0, 0, 0], [0, 0, axis_len * sz],
        ], dtype=np.float32)
        tripod_cam = tripod_enu @ R.T
        # Place axes at offset anchor
        anchor_x = -self.scale_factor * 0.8
        anchor_y = -self.scale_factor * 0.8
        self.axis_east.setData(tripod_cam[0:2, 0] + anchor_x, tripod_cam[0:2, 1] + anchor_y)
        self.axis_north.setData(tripod_cam[2:4, 0] + anchor_x, tripod_cam[2:4, 1] + anchor_y)
        self.axis_up.setData(tripod_cam[4:6, 0] + anchor_x, tripod_cam[4:6, 1] + anchor_y)

        # 7. Update view range
        s = self.scale_factor
        self.plot.setXRange(-s, s, padding=0.0)
        self.plot.setYRange(-s, s, padding=0.0)

        self.overlay.setText(
            f"3D View (No OpenGL): Azimuth {self.azimuth:.0f}° · Elevation {self.elevation:.0f}° · "
            f"Z-Scale {self.z_scale:.2f}x · Span {self.scale_factor:.0f} m · [Left: Orbit, Right: Pan, Scroll: Zoom, DblClick: Reset]"
        )

        self._events.draw.emit()
        self.plot_widget.update()

    def update_frame(self, sweep: SweepData, history_visible_count: int) -> None:
        self._last_sweep = sweep
        self.visible_history_count = int(history_visible_count)

        if self.camera_mode == "follow":
            self._update_follow_camera(sweep)
        else:
            self._reproject_all()

    def set_layers(self, *, aircraft: bool, trajectory: bool, history: bool, current: bool, beams: str) -> None:
        self.show_aircraft = bool(aircraft)
        self.show_trajectory = bool(trajectory)
        self.show_history_refh = bool(history)
        self.show_current_refh = bool(current)
        self.beam_density = str(beams).lower()
        self.show_beams = self.beam_density != "none"
        self._apply_visibility()
        self._reproject_all()

    def capture_framebuffer(self) -> QtGui.QImage:
        """Capture rendered 3D view without OpenGL."""
        return self.plot_widget.grab().toImage()
