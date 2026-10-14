"""Three-dimensional OpenGL visualization scene for CASALS 3D Sweep Animator.

Scientific boundaries & requirements:
- Local ENU display coordinates with configurable vertical scaling (e.g. 1.0, 0.1, 0.05, 0.025).
- Schematic aircraft model with fuselage, nose, wings, and stabilizers; oriented by trajectory heading.
- Ground Refh points are NOT meshed into a false continuous terrain surface.
- Beams are discrete instrument-to-Refh indicator lines, not physical ray-tracing.
- Camera supports Rotate, Pan, Zoom, Reset view, and Follow aircraft mode.
"""

from __future__ import annotations

from typing import Optional, Tuple
import matplotlib
import numpy as np
from PyQt5 import QtCore, QtGui
import pyqtgraph.opengl as gl

from .h5_model import SweepData


def create_schematic_aircraft_mesh(scale: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate vertices, faces, and colors for a schematic twin-wing aircraft.

    Components:
    - Fuselage: tapered aerodynamic body
    - Nose: nose cone pointing forward (+Y in aircraft body frame)
    - Wings: left and right swept main wings
    - Horizontal stabilizer: left and right tail wings
    - Vertical stabilizer: dorsal fin (+Z in aircraft body frame)

    Returns
    -------
    vertices : np.ndarray, shape (V, 3)
    faces : np.ndarray, shape (F, 3)
    face_colors : np.ndarray, shape (F, 4)
    """
    s = float(scale)

    # Key body-frame dimensions (in meters before scale multiplier)
    # Nose tip at y = +11, tail tip at y = -11
    # Right wingtip at x = +13, left wingtip at x = -13
    raw_vertices = [
        # 0: Nose tip
        [0.0, 11.0, 0.0],
        # 1-4: Cockpit/fuselage ring (front)
        [1.1, 7.0, 0.0],
        [0.0, 7.0, 1.2],
        [-1.1, 7.0, 0.0],
        [0.0, 7.0, -0.9],
        # 5-8: Mid fuselage ring (wings root)
        [1.3, 0.0, 0.0],
        [0.0, 0.0, 1.3],
        [-1.3, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        # 9-12: Aft fuselage ring (tail root)
        [0.7, -7.0, 0.0],
        [0.0, -7.0, 0.8],
        [-0.7, -7.0, 0.0],
        [0.0, -7.0, -0.6],
        # 13: Tail cone tip
        [0.0, -11.0, 0.3],
        # 14: Right wing tip
        [13.0, -2.0, 0.4],
        # 15: Right wing trailing edge root
        [1.3, -3.5, 0.0],
        # 16: Left wing tip
        [-13.0, -2.0, 0.4],
        # 17: Left wing trailing edge root
        [-1.3, -3.5, 0.0],
        # 18: Right tail wing tip
        [4.5, -9.5, 0.3],
        # 19: Left tail wing tip
        [-4.5, -9.5, 0.3],
        # 20: Vertical stabilizer top tip
        [0.0, -9.5, 3.8],
        # 21: Vertical stabilizer base front
        [0.0, -6.5, 0.8],
    ]

    verts = np.array(raw_vertices, dtype=np.float32) * s

    # Triangular faces (indices into verts)
    raw_faces = [
        # Nose cone triangles
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        # Forward fuselage
        [1, 5, 2],
        [5, 6, 2],
        [2, 6, 3],
        [6, 7, 3],
        [3, 7, 4],
        [7, 8, 4],
        [4, 8, 1],
        [8, 5, 1],
        # Aft fuselage
        [5, 9, 6],
        [9, 10, 6],
        [6, 10, 7],
        [10, 11, 7],
        [7, 11, 8],
        [11, 12, 8],
        [8, 12, 5],
        [12, 9, 5],
        # Tail cone
        [9, 13, 10],
        [10, 13, 11],
        [11, 13, 12],
        [12, 13, 9],
        # Right wing (upper & lower)
        [1, 14, 5],
        [5, 14, 15],
        [1, 5, 14],
        [5, 15, 14],
        # Left wing (upper & lower)
        [3, 7, 16],
        [7, 17, 16],
        [3, 16, 7],
        [7, 16, 17],
        # Horizontal stabilizer (tail wings)
        [9, 18, 13],
        [9, 13, 18],
        [11, 13, 19],
        [11, 19, 13],
        # Vertical stabilizer (fin)
        [21, 20, 10],
        [10, 20, 13],
        [21, 10, 20],
        [10, 13, 20],
    ]

    faces = np.array(raw_faces, dtype=np.int32)

    # Clean color palette for aircraft: light aircraft white/gray with contrast
    colors = np.full((len(faces), 4), [0.88, 0.90, 0.94, 0.95], dtype=np.float32)
    # Give wings slightly distinct shade
    colors[24:32] = [0.75, 0.82, 0.90, 0.95]
    # Tail fin highlight
    colors[36:40] = [0.95, 0.45, 0.20, 0.95]

    return verts, faces, colors


class Scene3DWidget(gl.GLViewWidget):
    """Interactive OpenGL 3D view for trajectory, aircraft, beams, and Refh points."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setBackgroundColor(QtGui.QColor(18, 22, 28))

        # Visual settings
        self.vertical_scale = 0.05
        self.aircraft_scale = 2.5
        self.camera_mode = "overview"  # 'overview' or 'follow'
        self.beam_density = "16"  # 'none', '16', '8', 'all'
        self.point_color_mode = "height"  # 'height', 'snr', 'amp', 'uniform'
        self.point_colormap = "turbo"
        self.snr_min_filter = 0.0
        self.show_aircraft = True
        self.show_trajectory = True
        self.show_history_refh = True
        self.show_current_refh = True
        self.show_beams = True

        # Stable color normalization range (set from H5Model)
        self.color_vmin = 2000.0
        self.color_vmax = 2400.0

        # Create persistent GL items
        self._init_gl_items()

        # Camera setup
        self.setCameraPosition(distance=1500, elevation=35, azimuth=45)
        self._last_aircraft_enu = np.zeros(3)

    def _init_gl_items(self) -> None:
        """Create and add persistent GL items to the scene."""
        # 1. Coordinate axes at origin (East=Red, North=Green, Up=Blue)
        self.axes_item = gl.GLAxisItem()
        self.axes_item.setSize(x=150, y=150, z=150 * self.vertical_scale)
        self.addItem(self.axes_item)

        # 2. Schematic aircraft mesh
        self._base_aircraft_verts, self._aircraft_faces, self._aircraft_colors = create_schematic_aircraft_mesh(
            scale=self.aircraft_scale
        )
        self.aircraft_mesh_data = gl.MeshData(
            vertexes=self._base_aircraft_verts,
            faces=self._aircraft_faces,
            faceColors=self._aircraft_colors,
        )
        self.aircraft_item = gl.GLMeshItem(
            meshdata=self.aircraft_mesh_data,
            smooth=False,
            drawEdges=True,
            edgeColor=(0.2, 0.3, 0.4, 0.8),
        )
        self.addItem(self.aircraft_item)

        # 3. Trajectory lines
        self.trajectory_past = gl.GLLinePlotItem(
            pos=np.zeros((1, 3)),
            color=(0.1, 0.85, 0.95, 0.8),
            width=2.0,
            antialias=True,
            mode="line_strip",
        )
        self.addItem(self.trajectory_past)

        self.trajectory_curr_marker = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=(1.0, 0.9, 0.2, 1.0),
            size=8.0,
        )
        self.addItem(self.trajectory_curr_marker)

        # 4. Historical Refh point cloud
        self.history_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3)),
            color=np.zeros((0, 4)),
            size=2.0,
            pxMode=True,
        )
        self.addItem(self.history_scatter)

        # 5. Current sweep Refh points (highlighted)
        self.current_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3)),
            color=np.zeros((0, 4)),
            size=6.0,
            pxMode=True,
        )
        self.addItem(self.current_scatter)

        # 6. Current sweep measurement beam connections
        self.beam_lines = gl.GLLinePlotItem(
            pos=np.zeros((0, 3)),
            color=(0.25, 0.80, 1.0, 0.45),
            width=1.2,
            antialias=True,
            mode="lines",
        )
        self.addItem(self.beam_lines)

    def set_vertical_scale(self, scale: float) -> None:
        """Update vertical display scaling factor."""
        self.vertical_scale = float(scale)
        self.axes_item.setSize(x=150, y=150, z=max(10.0, 150.0 * self.vertical_scale))

    def set_color_ranges(self, vmin: float, vmax: float) -> None:
        """Update global normalization limits for point cloud coloring."""
        self.color_vmin = float(vmin)
        self.color_vmax = float(vmax)
        if self.color_vmax <= self.color_vmin:
            self.color_vmax = self.color_vmin + 1.0

    def reset_view(self, center: Optional[Tuple[float, float, float]] = None, distance: float = 2000.0) -> None:
        """Reset camera to overview perspective."""
        if center is not None:
            c_pos = QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2]) * self.vertical_scale)
            self.opts["center"] = c_pos
        self.setCameraPosition(distance=distance, elevation=35, azimuth=-60)
        self.update()

    def _map_values_to_rgba(
        self,
        values: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Map 1D values to RGBA float32 array in [0, 1] using stable range."""
        n = len(values)
        if n == 0:
            return np.empty((0, 4), dtype=np.float32)

        if self.point_color_mode == "uniform":
            return np.full((n, 4), [0.35, 0.70, 0.95, alpha], dtype=np.float32)

        norm = (values - self.color_vmin) / (self.color_vmax - self.color_vmin)
        norm = np.clip(norm, 0.0, 1.0)
        norm[~np.isfinite(norm)] = 0.0

        try:
            cmap = matplotlib.colormaps[self.point_colormap]
        except Exception:
            cmap = matplotlib.colormaps["turbo"]

        rgba = cmap(norm).astype(np.float32)
        rgba[:, 3] = float(alpha)
        return rgba

    def update_frame(
        self,
        sweep: SweepData,
        history_enu: np.ndarray,
        history_values: Optional[np.ndarray],
        all_trajectory_enu: np.ndarray,
    ) -> None:
        """Update all 3D items for the current sweep."""
        z_scale = self.vertical_scale
        k = sweep.sweep_index

        # 1. Update Aircraft Position and Heading
        inst_enu = sweep.instrument_enu
        heading_deg = sweep.heading_deg
        self._last_aircraft_enu = inst_enu.copy()

        if self.show_aircraft:
            # Rotate body vertices by heading around ENU Up axis (+Z)
            # Aviation heading psi: 0=North (+Y), 90=East (+X)
            # East = x*cos(psi) + y*sin(psi), North = -x*sin(psi) + y*cos(psi)
            psi = np.deg2rad(heading_deg)
            cos_p, sin_p = np.cos(psi), np.sin(psi)

            vb = self._base_aircraft_verts
            rot_e = vb[:, 0] * cos_p + vb[:, 1] * sin_p
            rot_n = -vb[:, 0] * sin_p + vb[:, 1] * cos_p
            rot_u = vb[:, 2] * z_scale

            world_verts = np.column_stack(
                [
                    rot_e + inst_enu[0],
                    rot_n + inst_enu[1],
                    rot_u + inst_enu[2] * z_scale,
                ]
            ).astype(np.float32)

            self.aircraft_mesh_data.setVertexes(world_verts)
            self.aircraft_item.setVisible(True)
        else:
            self.aircraft_item.setVisible(False)

        # 2. Update Trajectory
        if self.show_trajectory and len(all_trajectory_enu) > 0:
            past_enu = all_trajectory_enu[: k + 1].copy()
            past_enu[:, 2] *= z_scale
            self.trajectory_past.setData(pos=past_enu)
            self.trajectory_past.setVisible(True)

            curr_pos = np.array([[inst_enu[0], inst_enu[1], inst_enu[2] * z_scale]], dtype=np.float32)
            self.trajectory_curr_marker.setData(pos=curr_pos)
            self.trajectory_curr_marker.setVisible(True)
        else:
            self.trajectory_past.setVisible(False)
            self.trajectory_curr_marker.setVisible(False)

        # 3. Update Historical Refh Cloud
        if self.show_history_refh and len(history_enu) > 0:
            hist_pos = history_enu.copy()
            hist_pos[:, 2] *= z_scale

            # Filter SNR if requested
            if self.snr_min_filter > 0 and history_values is not None:
                mask = history_values >= self.snr_min_filter
                hist_pos = hist_pos[mask]
                val_sub = history_values[mask]
            else:
                val_sub = history_values if history_values is not None else hist_pos[:, 2] / z_scale

            colors = self._map_values_to_rgba(val_sub, alpha=0.45)
            self.history_scatter.setData(pos=hist_pos, color=colors)
            self.history_scatter.setVisible(True)
        else:
            self.history_scatter.setData(pos=np.zeros((0, 3)), color=np.zeros((0, 4)))
            self.history_scatter.setVisible(False)

        # 4. Update Current Sweep Refh Points (Highlighted)
        if self.show_current_refh and len(sweep.refh_enu) > 0:
            curr_pos = sweep.refh_enu.copy()
            valid = sweep.valid_pulse_mask
            if np.any(valid):
                curr_pos = curr_pos[valid]
                curr_pos[:, 2] *= z_scale

                # Highlight color: bright yellow-orange with high alpha
                curr_colors = np.full((len(curr_pos), 4), [1.0, 0.95, 0.15, 1.0], dtype=np.float32)
                self.current_scatter.setData(pos=curr_pos, color=curr_colors)
                self.current_scatter.setVisible(True)
            else:
                self.current_scatter.setVisible(False)
        else:
            self.current_scatter.setVisible(False)

        # 5. Update Current Sweep Beams
        density = self.beam_density.lower()
        if self.show_beams and density != "none" and len(sweep.refh_enu) > 0:
            valid_p = np.flatnonzero(sweep.valid_pulse_mask)
            if len(valid_p) > 0:
                if density == "16":
                    step = 16
                elif density == "8":
                    step = 8
                else:  # 'all'
                    step = 1
                sampled_p = valid_p[::step]

                p_inst = sweep.pulse_instrument_enu[sampled_p].copy()
                p_refh = sweep.refh_enu[sampled_p].copy()

                p_inst[:, 2] *= z_scale
                p_refh[:, 2] *= z_scale

                # Interleave inst and refh for mode='lines' (line segments)
                line_pts = np.empty((len(sampled_p) * 2, 3), dtype=np.float32)
                line_pts[0::2] = p_inst
                line_pts[1::2] = p_refh

                self.beam_lines.setData(pos=line_pts)
                self.beam_lines.setVisible(True)
            else:
                self.beam_lines.setVisible(False)
        else:
            self.beam_lines.setVisible(False)

        # 6. Follow Aircraft Camera
        if self.camera_mode == "follow":
            target = QtGui.QVector3D(float(inst_enu[0]), float(inst_enu[1]), float(inst_enu[2] * z_scale))
            self.opts["center"] = target
            self.update()

