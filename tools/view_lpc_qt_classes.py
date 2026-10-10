"""Qt LAS/LAZ point-cloud classification viewer.

Purpose
-------
Open a classified LAS/LAZ point cloud, display sampled points in a Qt-native
OpenGL widget, and toggle visibility by LAS classification code.

Features
--------
- Modern, clean dark theme with resizable split pane layout.
- Real-time classification visibility filtering with search/filter box and quick actions (All, None, Invert).
- Double-click solo mode (isolate a single class with one click).
- Interactive customizable class colors via QColorDialog, with instant OpenGL updates without reloading.
- Multiple palette presets (ASPRS Standard, High Contrast, Vivid Neon, Soft Pastel) and reset button.
- Elevation colormapping mode (Turbo, Viridis, Plasma, Terrain) for continuous vertical profiling.
- Scene guides: Toggleable Ground Grid (G), Coordinate Axes (X), and 3D Bounding Box (B).
- Camera view orientation presets: Top (XY), Front (XZ), Side (YZ), 3D Isometric, and Fit/Reset.
- Drag-and-drop support: drop any .las or .laz file into the window to load it immediately.
- Point size, opacity (alpha blending), and vertical Z-scale exaggeration sliders with two-way sync.
- 3D viewport background color selector (Dark Slate, Pure Black, Deep Navy, Neutral Gray, Light/White, Custom).
- Viewport snapshot export (PNG/JPG) for documentation and presentations.
- Rich file metadata, spatial bounding box extents, and CRS details with one-click copy to clipboard.
- Asynchronous background file loading with progress indicator.

Recommended installation
------------------------
conda install -c conda-forge numpy laspy lazrs pyqtgraph pyopengl pyqt

Run
---
python tools/view_lpc_qt_classes.py
python tools/view_lpc_qt_classes.py path/to/cloud.laz --max-display-points 800000
"""

from __future__ import annotations

import argparse
import colorsys
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import laspy
except Exception as exc:  # pragma: no cover
    raise ImportError("laspy is required. Install with: conda install -c conda-forge laspy lazrs") from exc

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "pyqtgraph + a Qt binding are required. Install with: "
        "conda install -c conda-forge pyqtgraph pyopengl pyqt"
    ) from exc

Signal = getattr(QtCore, "Signal", None) or getattr(QtCore, "pyqtSignal")


@dataclass(frozen=True)
class ViewerConfig:
    max_display_points: int = 800_000
    random_seed: int = 42
    read_chunk_size: int = 2_000_000
    center_xy_for_view: bool = True
    center_z_for_view: bool = True
    z_scale_for_view: float = 1.0
    point_size_px: float = 2.0
    point_opacity: float = 1.0


@dataclass(frozen=True)
class SampledPointCloud:
    xyz: np.ndarray  # sampled original coordinates, shape=(n, 3), float64
    classification: np.ndarray  # sampled class codes, shape=(n,), uint8


@dataclass(frozen=True)
class LasViewSummary:
    input_path: Path
    point_count_total: int
    point_count_valid_xyz: int
    point_count_sampled: int
    las_version: str
    point_format_id: int
    classification_counts: dict[int, int]
    sampled_classification_counts: dict[int, int]
    unknown_class_codes: list[int]
    crs_text: str
    crs_epsg: Optional[int]
    crs_name: str
    sampled_cloud: SampledPointCloud
    bounds_xyz: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )
    file_size_bytes: int = 0


CLASS_LABEL_MAP: dict[int, str] = {
    0: "Never classified / Unknown",
    1: "Unassigned / Processed",
    2: "Ground / Bare earth",
    3: "Low vegetation",
    4: "Medium vegetation",
    5: "High vegetation",
    6: "Building",
    7: "Low noise / Outlier",
    8: "Model key / Reserved",
    9: "Water",
    10: "Rail",
    11: "Road surface",
    12: "Overlap / Reserved",
    13: "Wire - Guard",
    14: "Wire - Conductor",
    15: "Transmission tower",
    16: "Wire - Connector",
    17: "Bridge deck",
    18: "High noise",
    19: "Overhead structure",
    20: "Ignored ground",
    21: "Snow",
    22: "Temporal exclusion",
}

# Standard ASPRS palette: Earthy browns for ground, greens for vegetation, orange for buildings, cyan for water.
CLASS_COLOR_MAP: dict[int, tuple[float, float, float]] = {
    0: (0.40, 0.40, 0.40),
    1: (0.42, 0.68, 0.38),
    2: (0.58, 0.46, 0.33),
    3: (0.68, 0.85, 0.46),
    4: (0.35, 0.72, 0.30),
    5: (0.12, 0.46, 0.14),
    6: (0.86, 0.44, 0.24),
    7: (0.95, 0.20, 0.18),
    8: (0.75, 0.75, 0.25),
    9: (0.10, 0.72, 0.90),
    10: (0.45, 0.35, 0.30),
    11: (0.30, 0.30, 0.35),
    12: (0.80, 0.65, 0.20),
    13: (0.90, 0.85, 0.20),
    14: (0.95, 0.65, 0.15),
    15: (0.60, 0.60, 0.70),
    16: (0.85, 0.75, 0.30),
    17: (0.62, 0.38, 0.26),
    18: (1.00, 0.08, 0.15),
    19: (0.55, 0.50, 0.65),
    20: (0.72, 0.65, 0.50),
    21: (0.95, 0.96, 1.00),
    22: (0.88, 0.20, 0.75),
}

# High Contrast categorical palette
HIGH_CONTRAST_COLOR_MAP: dict[int, tuple[float, float, float]] = {
    0: (0.55, 0.55, 0.55),
    1: (0.12, 0.53, 0.90),
    2: (0.89, 0.10, 0.11),
    3: (0.20, 0.63, 0.17),
    4: (1.00, 0.50, 0.00),
    5: (0.42, 0.24, 0.60),
    6: (0.65, 0.34, 0.16),
    7: (0.97, 0.51, 0.75),
    9: (0.00, 0.80, 0.80),
    17: (0.70, 0.70, 0.10),
    18: (0.90, 0.10, 0.30),
    20: (0.60, 0.60, 0.30),
    21: (0.95, 0.95, 1.00),
    22: (0.50, 0.10, 0.80),
}

# Vivid Neon palette (high luminosity, excels on dark viewports)
VIVID_NEON_COLOR_MAP: dict[int, tuple[float, float, float]] = {
    0: (0.50, 0.50, 0.55),
    1: (0.00, 0.95, 0.60),
    2: (1.00, 0.75, 0.00),
    3: (0.45, 1.00, 0.10),
    4: (0.00, 0.90, 0.35),
    5: (0.00, 0.70, 0.25),
    6: (1.00, 0.35, 0.10),
    7: (1.00, 0.05, 0.25),
    9: (0.00, 0.85, 1.00),
    17: (0.90, 0.55, 0.25),
    18: (1.00, 0.00, 0.60),
    20: (0.85, 0.80, 0.50),
    21: (1.00, 1.00, 1.00),
    22: (0.75, 0.20, 1.00),
}

# Soft Pastel palette
PASTEL_COLOR_MAP: dict[int, tuple[float, float, float]] = {
    0: (0.65, 0.65, 0.65),
    1: (0.60, 0.82, 0.68),
    2: (0.82, 0.72, 0.58),
    3: (0.72, 0.88, 0.65),
    4: (0.52, 0.78, 0.60),
    5: (0.38, 0.62, 0.45),
    6: (0.88, 0.62, 0.52),
    7: (0.92, 0.52, 0.52),
    9: (0.55, 0.78, 0.92),
    17: (0.75, 0.62, 0.50),
    18: (0.92, 0.45, 0.45),
    20: (0.82, 0.78, 0.68),
    21: (0.96, 0.96, 1.00),
    22: (0.85, 0.62, 0.82),
}

PALETTES: dict[str, dict[int, tuple[float, float, float]]] = {
    "ASPRS Standard": CLASS_COLOR_MAP,
    "High Contrast": HIGH_CONTRAST_COLOR_MAP,
    "Vivid Neon": VIVID_NEON_COLOR_MAP,
    "Soft Pastel": PASTEL_COLOR_MAP,
}

FALLBACK_CLASS_COLOR = (0.55, 0.50, 0.65)


def generate_distinct_color(code: int) -> tuple[float, float, float]:
    """Generate a reproducible, vibrant color for non-standard classification codes."""
    hue = (int(code) * 0.618033988749895) % 1.0
    sat = 0.72 + (int(code) % 3) * 0.08
    val = 0.85 + (int(code) % 2) * 0.10
    return colorsys.hsv_to_rgb(hue, min(1.0, sat), min(1.0, val))


def color_to_css(color: tuple[float, float, float]) -> str:
    r = int(np.clip(round(color[0] * 255.0), 0, 255))
    g = int(np.clip(round(color[1] * 255.0), 0, 255))
    b = int(np.clip(round(color[2] * 255.0), 0, 255))
    return f"rgb({r}, {g}, {b})"


def color_to_hex(color: tuple[float, float, float]) -> str:
    r = int(np.clip(round(color[0] * 255.0), 0, 255))
    g = int(np.clip(round(color[1] * 255.0), 0, 255))
    b = int(np.clip(round(color[2] * 255.0), 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def get_elevation_colormap_colors(
    norm_z: np.ndarray, colormap_name: str, alpha: float = 1.0
) -> np.ndarray:
    """Map normalized [0, 1] values to RGBA float32 using matplotlib or mathematical fallbacks."""
    name = colormap_name.lower().strip()
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(name)
        rgba = cmap(norm_z).astype(np.float32)
        rgba[:, 3] = float(alpha)
        return rgba
    except Exception:
        # Fallback pseudo-spectral gradient
        v = np.clip(norm_z, 0.0, 1.0)
        r = np.clip(np.sin(v * np.pi - np.pi / 2.0) * 0.5 + 0.5, 0.0, 1.0)
        g = np.clip(np.sin(v * np.pi) * 0.85 + 0.15, 0.0, 1.0)
        b = np.clip(np.cos(v * np.pi) * 0.5 + 0.5, 0.0, 1.0)
        a = np.full_like(v, float(alpha), dtype=np.float32)
        return np.column_stack((r, g, b, a)).astype(np.float32)


class LasLoadWorker(QtCore.QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path, cfg: ViewerConfig) -> None:
        super().__init__()
        self.path = path
        self.cfg = cfg

    def run(self) -> None:
        try:
            summary = sampled_points_from_las(self.path, self.cfg)
            self.finished.emit(summary)
        except Exception as exc:  # pragma: no cover - GUI error path
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def validate_input_path(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Input LAS/LAZ file does not exist: {path}")
    if path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError(f"Input file must end with .las or .laz: {path}")
    return path


def safe_parse_crs(header) -> tuple[str, Optional[int], str]:
    try:
        crs = header.parse_crs()
    except Exception:
        crs = None

    if crs is None:
        return "CRS unavailable", None, ""

    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None

    try:
        text = crs.to_string()
    except Exception:
        text = "CRS parsed but to_string() failed"

    try:
        name = crs.name or ""
    except Exception:
        name = ""

    return text, epsg, name


def update_classification_counts(counts: dict[int, int], cls: np.ndarray) -> None:
    unique, count = np.unique(cls, return_counts=True)
    for key, value in zip(unique.tolist(), count.tolist()):
        counts[int(key)] = counts.get(int(key), 0) + int(value)


def merge_reservoir(
    keys_a: np.ndarray,
    points_a: np.ndarray,
    cls_a: np.ndarray,
    keys_b: np.ndarray,
    points_b: np.ndarray,
    cls_b: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep a fixed-size uniform random sample using random keys."""
    if keys_a.size == 0:
        merged_keys = keys_b
        merged_points = points_b
        merged_cls = cls_b
    elif keys_b.size == 0:
        merged_keys = keys_a
        merged_points = points_a
        merged_cls = cls_a
    else:
        merged_keys = np.concatenate([keys_a, keys_b])
        merged_points = np.vstack([points_a, points_b])
        merged_cls = np.concatenate([cls_a, cls_b])

    if merged_keys.size <= max_points:
        return merged_keys, merged_points, merged_cls

    keep = np.argpartition(merged_keys, -max_points)[-max_points:]
    return merged_keys[keep], merged_points[keep], merged_cls[keep]


def sampled_points_from_las(path: Path, cfg: ViewerConfig) -> LasViewSummary:
    path = validate_input_path(path)
    if cfg.max_display_points <= 0:
        raise ValueError("max_display_points must be > 0.")
    if cfg.read_chunk_size <= 0:
        raise ValueError("read_chunk_size must be > 0.")

    classification_counts: dict[int, int] = {}
    sampled_keys = np.empty(0, dtype=np.float64)
    sampled_points = np.empty((0, 3), dtype=np.float64)
    sampled_cls = np.empty(0, dtype=np.uint8)
    total_points = 0
    valid_xyz_points = 0
    rng = np.random.default_rng(cfg.random_seed)

    try:
        reader = laspy.open(path)
    except Exception as exc:
        msg = f"Failed to open LAS/LAZ file: {path}\n{type(exc).__name__}: {exc}"
        if path.suffix.lower() == ".laz":
            msg += "\nLAZ reading may require lazrs: conda install -c conda-forge lazrs"
        raise RuntimeError(msg) from exc

    with reader:
        header = reader.header
        crs_text, crs_epsg, crs_name = safe_parse_crs(header)
        las_version = str(header.version)
        point_format_id = int(header.point_format.id)
        file_size_bytes = path.stat().st_size if path.exists() else 0

        bounds_xyz = (
            (float(header.x_min), float(header.x_max)),
            (float(header.y_min), float(header.y_max)),
            (float(header.z_min), float(header.z_max)),
        )

        for chunk in reader.chunk_iterator(cfg.read_chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            try:
                cls = np.asarray(chunk.classification, dtype=np.uint8)
            except Exception as exc:
                raise RuntimeError("LAS classification field could not be read.") from exc

            total_points += int(x.size)
            update_classification_counts(classification_counts, cls)

            valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            n_valid = int(np.sum(valid_mask))
            valid_xyz_points += n_valid
            if n_valid == 0:
                continue

            xyz = np.column_stack((x[valid_mask], y[valid_mask], z[valid_mask]))
            cls_valid = cls[valid_mask]
            keys = rng.random(n_valid)
            sampled_keys, sampled_points, sampled_cls = merge_reservoir(
                sampled_keys,
                sampled_points,
                sampled_cls,
                keys,
                xyz,
                cls_valid,
                cfg.max_display_points,
            )

    if total_points <= 0:
        raise RuntimeError(f"Input LAS/LAZ file has no points: {path}")
    if valid_xyz_points <= 0 or sampled_points.size == 0:
        raise RuntimeError(f"Input LAS/LAZ file has no finite XYZ points: {path}")

    order = np.argsort(sampled_keys)
    sampled_points = sampled_points[order]
    sampled_cls = sampled_cls[order]

    sampled_classification_counts: dict[int, int] = {}
    update_classification_counts(sampled_classification_counts, sampled_cls)

    known_codes = set(CLASS_COLOR_MAP)
    unknown_codes = sorted(code for code in classification_counts if code not in known_codes)

    return LasViewSummary(
        input_path=path,
        point_count_total=total_points,
        point_count_valid_xyz=valid_xyz_points,
        point_count_sampled=int(sampled_points.shape[0]),
        las_version=las_version,
        point_format_id=point_format_id,
        classification_counts=dict(sorted(classification_counts.items())),
        sampled_classification_counts=dict(sorted(sampled_classification_counts.items())),
        unknown_class_codes=unknown_codes,
        crs_text=crs_text,
        crs_epsg=crs_epsg,
        crs_name=crs_name,
        sampled_cloud=SampledPointCloud(
            xyz=sampled_points.astype(np.float64, copy=False),
            classification=sampled_cls.astype(np.uint8, copy=False),
        ),
        bounds_xyz=bounds_xyz,
        file_size_bytes=file_size_bytes,
    )


def centered_points_for_view(
    xyz: np.ndarray, cfg: ViewerConfig
) -> tuple[np.ndarray, tuple[float, float, float]]:
    x0 = float(np.nanmedian(xyz[:, 0])) if cfg.center_xy_for_view else 0.0
    y0 = float(np.nanmedian(xyz[:, 1])) if cfg.center_xy_for_view else 0.0
    z0 = float(np.nanmedian(xyz[:, 2])) if cfg.center_z_for_view else 0.0
    centered = np.column_stack(
        (
            xyz[:, 0] - x0,
            xyz[:, 1] - y0,
            (xyz[:, 2] - z0) * float(cfg.z_scale_for_view),
        )
    ).astype(np.float32)
    return centered, (x0, y0, z0)


def _nice_grid_spacing(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 10.0
    exponent = np.floor(np.log10(value))
    base = value / (10.0**exponent)
    if base <= 1.5:
        nice = 1.0
    elif base <= 3.0:
        nice = 2.0
    elif base <= 7.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10.0**exponent))


DARK_STYLESHEET = """
QMainWindow {
    background-color: #17181c;
}
QWidget {
    color: #e2e4ea;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
}
QSplitter::handle {
    background-color: #262832;
    width: 4px;
}
QSplitter::handle:hover {
    background-color: #3b82f6;
}
QTabWidget::pane {
    border: 1px solid #282b36;
    background-color: #1c1e24;
    border-radius: 6px;
    padding: 2px;
}
QTabBar::tab {
    background-color: #17181c;
    color: #9499a8;
    padding: 7px 14px;
    margin-right: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    border: 1px solid transparent;
    font-weight: 500;
}
QTabBar::tab:selected {
    background-color: #1c1e24;
    color: #ffffff;
    border: 1px solid #282b36;
    border-bottom: 1px solid #1c1e24;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background-color: #22242c;
    color: #d1d5db;
}
QGroupBox {
    background-color: #1c1e24;
    border: 1px solid #282b36;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 600;
    color: #93c5fd;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
}
QPushButton {
    background-color: #272a35;
    color: #f3f4f6;
    border: 1px solid #363b4a;
    border-radius: 4px;
    padding: 5px 10px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #323746;
    border-color: #4b5267;
}
QPushButton:pressed {
    background-color: #1c1e24;
}
QPushButton:disabled {
    background-color: #1a1b20;
    color: #555b6a;
    border-color: #22252c;
}
QPushButton#primaryButton {
    background-color: #2563eb;
    border-color: #3b82f6;
    color: #ffffff;
    font-weight: bold;
}
QPushButton#primaryButton:hover {
    background-color: #1d4ed8;
    border-color: #60a5fa;
}
QPushButton#secondaryButton {
    background-color: #1e293b;
    border-color: #334155;
    color: #e2e8f0;
}
QPushButton#secondaryButton:hover {
    background-color: #334155;
}
QTableWidget {
    background-color: #16171d;
    alternate-background-color: #1c1e24;
    border: 1px solid #282b36;
    border-radius: 5px;
    gridline-color: #22242e;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QHeaderView::section {
    background-color: #22252e;
    color: #9ba1b0;
    font-weight: 600;
    border: none;
    border-bottom: 1px solid #2e323e;
    border-right: 1px solid #282b36;
    padding: 5px 6px;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #15161b;
    color: #f3f4f6;
    border: 1px solid #282b36;
    border-radius: 4px;
    padding: 3px 6px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #3b82f6;
}
QComboBox QAbstractItemView {
    background-color: #1c1e24;
    color: #e2e4ea;
    border: 1px solid #282b36;
    selection-background-color: #2563eb;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #282b36;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #3b82f6;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #60a5fa;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #93c5fd;
}
QCheckBox {
    spacing: 5px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #363b4a;
    border-radius: 3px;
    background-color: #16171d;
}
QCheckBox::indicator:checked {
    background-color: #2563eb;
    border-color: #3b82f6;
}
QCheckBox::indicator:hover {
    border-color: #60a5fa;
}
QStatusBar {
    background-color: #131418;
    color: #9499a8;
    border-top: 1px solid #22242c;
}
QProgressBar {
    border: 1px solid #282b36;
    border-radius: 3px;
    background-color: #15161b;
    text-align: center;
    color: #ffffff;
    max-height: 14px;
}
QProgressBar::chunk {
    background-color: #3b82f6;
}
"""


class CustomGLViewWidget(gl.GLViewWidget):
    """GLViewWidget with interactive camera tracking signal."""

    camera_changed = Signal()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(ev)
        self.camera_changed.emit()

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        super().wheelEvent(ev)
        self.camera_changed.emit()


class ClassificationPointCloudViewer(QtWidgets.QMainWindow):
    def __init__(self, initial_path: Optional[Path] = None, cfg: Optional[ViewerConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ViewerConfig()
        self.summary: Optional[LasViewSummary] = None
        self.centered_points: Optional[np.ndarray] = None
        self.colors_rgba: Optional[np.ndarray] = None
        self.classification: Optional[np.ndarray] = None
        self.center_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Dynamic color states
        self.class_colors: dict[int, tuple[float, float, float]] = dict(CLASS_COLOR_MAP)
        self.class_checkboxes: dict[int, QtWidgets.QCheckBox] = {}
        self.class_color_buttons: dict[int, QtWidgets.QPushButton] = {}

        # View and scene states
        self.current_color_mode: str = "Classification"
        self.grid_on_ground: bool = True

        # Async worker
        self.load_thread: Optional[QtCore.QThread] = None
        self.load_worker: Optional[LasLoadWorker] = None

        self.setWindowTitle("LAS/LAZ Classification Point Cloud Viewer")
        self.resize(1520, 940)
        self.setAcceptDrops(True)
        self.setStyleSheet(DARK_STYLESHEET)

        self._build_ui()
        self._setup_shortcuts()

        if initial_path is not None:
            QtCore.QTimer.singleShot(50, lambda: self.load_las_file(Path(initial_path)))

    def _build_ui(self) -> None:
        pg.setConfigOptions(antialias=False)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        # Main horizontal splitter
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
        root_layout.addWidget(self.splitter)

        # Left panel (Sidebar)
        left_panel = QtWidgets.QWidget()
        left_panel.setMinimumWidth(430)
        left_panel.setMaximumWidth(680)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        # Header row: Open & Reload buttons
        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(6)
        self.open_button = QtWidgets.QPushButton("📂 Open LAS / LAZ")
        self.open_button.setObjectName("primaryButton")
        self.open_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.open_button.clicked.connect(self.choose_file)

        self.reload_button = QtWidgets.QPushButton("🔄 Reload")
        self.reload_button.setObjectName("secondaryButton")
        self.reload_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.reload_button.clicked.connect(self.reload_current_file)
        self.reload_button.setEnabled(False)

        header_row.addWidget(self.open_button, stretch=3)
        header_row.addWidget(self.reload_button, stretch=1)
        left_layout.addLayout(header_row)

        # File banner card
        self.file_card = QtWidgets.QFrame()
        self.file_card.setStyleSheet(
            "QFrame { background-color: #1a1c22; border: 1px solid #282b36; border-radius: 6px; padding: 6px; }"
        )
        file_card_layout = QtWidgets.QVBoxLayout(self.file_card)
        file_card_layout.setContentsMargins(4, 2, 4, 2)
        file_card_layout.setSpacing(2)

        self.file_title_label = QtWidgets.QLabel("No file loaded")
        self.file_title_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #ffffff;")
        self.file_subtitle_label = QtWidgets.QLabel("Drag and drop a .las/.laz file here or click Open")
        self.file_subtitle_label.setStyleSheet("color: #8b92a4; font-size: 11px;")
        file_card_layout.addWidget(self.file_title_label)
        file_card_layout.addWidget(self.file_subtitle_label)
        left_layout.addWidget(self.file_card)

        # Quick Sliders Group (Point Size, Opacity, Z-Scale)
        sliders_group = QtWidgets.QGroupBox("Quick Display Adjustments")
        sliders_layout = QtWidgets.QGridLayout(sliders_group)
        sliders_layout.setContentsMargins(8, 8, 8, 8)
        sliders_layout.setSpacing(6)

        # 1. Point Size
        sliders_layout.addWidget(QtWidgets.QLabel("Point Size:"), 0, 0)
        self.point_size_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.point_size_slider.setRange(5, 200)  # 0.5 to 20.0 px
        self.point_size_slider.setValue(int(round(self.cfg.point_size_px * 10)))
        self.point_size_spin = QtWidgets.QDoubleSpinBox()
        self.point_size_spin.setRange(0.5, 20.0)
        self.point_size_spin.setSingleStep(0.5)
        self.point_size_spin.setValue(self.cfg.point_size_px)
        self.point_size_spin.setSuffix(" px")
        self.point_size_slider.valueChanged.connect(self._on_point_size_slider_moved)
        self.point_size_spin.valueChanged.connect(self._on_point_size_spin_changed)
        sliders_layout.addWidget(self.point_size_slider, 0, 1)
        sliders_layout.addWidget(self.point_size_spin, 0, 2)

        # 2. Point Opacity
        sliders_layout.addWidget(QtWidgets.QLabel("Opacity:"), 1, 0)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(int(round(self.cfg.point_opacity * 100)))
        self.opacity_spin = QtWidgets.QSpinBox()
        self.opacity_spin.setRange(10, 100)
        self.opacity_spin.setSingleStep(5)
        self.opacity_spin.setValue(int(round(self.cfg.point_opacity * 100)))
        self.opacity_spin.setSuffix(" %")
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_moved)
        self.opacity_spin.valueChanged.connect(self._on_opacity_spin_changed)
        sliders_layout.addWidget(self.opacity_slider, 1, 1)
        sliders_layout.addWidget(self.opacity_spin, 1, 2)

        # 3. Z Scale
        sliders_layout.addWidget(QtWidgets.QLabel("Z Scale:"), 2, 0)
        self.z_scale_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.z_scale_slider.setRange(1, 100)  # 0.1x to 10.0x
        self.z_scale_slider.setValue(int(round(self.cfg.z_scale_for_view * 10)))
        self.z_scale_spin = QtWidgets.QDoubleSpinBox()
        self.z_scale_spin.setRange(0.05, 50.0)
        self.z_scale_spin.setDecimals(2)
        self.z_scale_spin.setSingleStep(0.25)
        self.z_scale_spin.setValue(self.cfg.z_scale_for_view)
        self.z_scale_spin.setSuffix(" x")
        self.z_scale_slider.valueChanged.connect(self._on_z_scale_slider_moved)
        self.z_scale_spin.valueChanged.connect(self._on_z_scale_spin_changed)
        sliders_layout.addWidget(self.z_scale_slider, 2, 1)
        sliders_layout.addWidget(self.z_scale_spin, 2, 2)

        left_layout.addWidget(sliders_group)

        # Tabs: Classes & Colors, Render & Scene, File & Metadata
        self.tabs = QtWidgets.QTabWidget()
        left_layout.addWidget(self.tabs, stretch=1)

        # ====================================================================
        # Tab 1: Classes & Colors
        # ====================================================================
        classes_tab = QtWidgets.QWidget()
        classes_layout = QtWidgets.QVBoxLayout(classes_tab)
        classes_layout.setContentsMargins(4, 6, 4, 4)
        classes_layout.setSpacing(6)

        # Search bar & selection buttons
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.setSpacing(4)
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("🔍 Filter class by name or ID...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._filter_class_table)
        filter_row.addWidget(self.search_input, stretch=3)

        self.btn_select_all = QtWidgets.QPushButton("All")
        self.btn_select_all.setToolTip("Select all classes (Ctrl+A)")
        self.btn_select_all.clicked.connect(lambda: self.set_all_classes_checked(True))
        self.btn_clear_all = QtWidgets.QPushButton("None")
        self.btn_clear_all.setToolTip("Deselect all classes (Ctrl+D)")
        self.btn_clear_all.clicked.connect(lambda: self.set_all_classes_checked(False))
        self.btn_invert = QtWidgets.QPushButton("Invert")
        self.btn_invert.setToolTip("Invert selection (Ctrl+I)")
        self.btn_invert.clicked.connect(self.invert_class_selection)

        filter_row.addWidget(self.btn_select_all)
        filter_row.addWidget(self.btn_clear_all)
        filter_row.addWidget(self.btn_invert)
        classes_layout.addLayout(filter_row)

        # Class table
        self.class_table = QtWidgets.QTableWidget(0, 6)
        self.class_table.setHorizontalHeaderLabels(["Show", "Color", "ID", "Classification Label", "Sampled", "Total"])
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.class_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.class_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.class_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.class_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.class_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.class_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.class_table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.class_table.cellDoubleClicked.connect(self._on_table_double_clicked)
        self.class_table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.class_table.customContextMenuRequested.connect(self._on_table_context_menu)
        classes_layout.addWidget(self.class_table, stretch=1)

        # Palette Presets Row
        palette_row = QtWidgets.QHBoxLayout()
        palette_row.setSpacing(6)
        palette_row.addWidget(QtWidgets.QLabel("Palette:"))
        self.palette_combo = QtWidgets.QComboBox()
        self.palette_combo.addItems(list(PALETTES.keys()))
        self.palette_combo.currentTextChanged.connect(self.apply_palette_preset)
        palette_row.addWidget(self.palette_combo, stretch=2)

        self.btn_reset_colors = QtWidgets.QPushButton("🔄 Reset Colors")
        self.btn_reset_colors.setToolTip("Reset all class colors back to ASPRS standard defaults")
        self.btn_reset_colors.clicked.connect(self.reset_all_class_colors)
        palette_row.addWidget(self.btn_reset_colors, stretch=1)
        classes_layout.addLayout(palette_row)

        self.tabs.addTab(classes_tab, "🏷️ Classes")

        # ====================================================================
        # Tab 2: Render & Scene Controls
        # ====================================================================
        render_tab = QtWidgets.QWidget()
        render_layout = QtWidgets.QVBoxLayout(render_tab)
        render_layout.setContentsMargins(6, 6, 6, 6)
        render_layout.setSpacing(8)

        # Color Mode Group
        color_mode_group = QtWidgets.QGroupBox("Point Cloud Color Mode")
        color_mode_layout = QtWidgets.QVBoxLayout(color_mode_group)
        self.color_mode_combo = QtWidgets.QComboBox()
        self.color_mode_combo.addItems(
            [
                "Classification (Custom / Editable)",
                "Elevation: Turbo (Spectral)",
                "Elevation: Viridis (Perceptual)",
                "Elevation: Plasma (Warm Purple/Yellow)",
                "Elevation: Terrain (Earth Tones)",
            ]
        )
        self.color_mode_combo.currentIndexChanged.connect(self._on_color_mode_changed)
        color_mode_layout.addWidget(self.color_mode_combo)
        render_layout.addWidget(color_mode_group)

        # Scene Guides Group
        guides_group = QtWidgets.QGroupBox("Scene Guides & Reference")
        guides_layout = QtWidgets.QGridLayout(guides_group)
        guides_layout.setSpacing(6)

        self.cb_grid = QtWidgets.QCheckBox("Show Ground Grid [G]")
        self.cb_grid.setChecked(True)
        self.cb_grid.toggled.connect(self.set_grid_visible)
        guides_layout.addWidget(self.cb_grid, 0, 0)

        self.cb_axis = QtWidgets.QCheckBox("Show Coordinate Axes [X]")
        self.cb_axis.setChecked(True)
        self.cb_axis.toggled.connect(self.set_axis_visible)
        guides_layout.addWidget(self.cb_axis, 0, 1)

        self.cb_bbox = QtWidgets.QCheckBox("Show Bounding Box [B]")
        self.cb_bbox.setChecked(False)
        self.cb_bbox.toggled.connect(self.set_bbox_visible)
        guides_layout.addWidget(self.cb_bbox, 1, 0)

        self.cb_grid_ground = QtWidgets.QCheckBox("Place Grid at Base (Z-min)")
        self.cb_grid_ground.setChecked(True)
        self.cb_grid_ground.toggled.connect(self._on_grid_ground_toggled)
        guides_layout.addWidget(self.cb_grid_ground, 1, 1)

        render_layout.addWidget(guides_group)

        # Sampling & Performance Group
        perf_group = QtWidgets.QGroupBox("Sampling & Resolution")
        perf_layout = QtWidgets.QGridLayout(perf_group)
        perf_layout.addWidget(QtWidgets.QLabel("Max Display Points:"), 0, 0)
        self.max_points_spin = QtWidgets.QSpinBox()
        self.max_points_spin.setRange(10_000, 20_000_000)
        self.max_points_spin.setSingleStep(50_000)
        self.max_points_spin.setValue(self.cfg.max_display_points)
        perf_layout.addWidget(self.max_points_spin, 0, 1)

        self.btn_apply_reload = QtWidgets.QPushButton("Reload with Max Points")
        self.btn_apply_reload.setObjectName("secondaryButton")
        self.btn_apply_reload.clicked.connect(self.reload_current_file)
        perf_layout.addWidget(self.btn_apply_reload, 1, 0, 1, 2)
        render_layout.addWidget(perf_group)

        render_layout.addStretch(1)
        self.tabs.addTab(render_tab, "⚙️ Render & Scene")

        # ====================================================================
        # Tab 3: File & Metadata Info
        # ====================================================================
        info_tab = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout(info_tab)
        info_layout.setContentsMargins(6, 6, 6, 6)
        info_layout.setSpacing(6)

        self.info_text = QtWidgets.QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlaceholderText("Open a classified LAS/LAZ file to inspect header and CRS details.")
        info_layout.addWidget(self.info_text, stretch=1)

        info_btn_row = QtWidgets.QHBoxLayout()
        self.btn_copy_info = QtWidgets.QPushButton("📋 Copy Metadata")
        self.btn_copy_info.clicked.connect(self.copy_metadata_to_clipboard)
        info_btn_row.addWidget(self.btn_copy_info)
        info_layout.addLayout(info_btn_row)

        self.tabs.addTab(info_tab, "ℹ️ Metadata")

        # Add sidebar to splitter
        self.splitter.addWidget(left_panel)

        # ====================================================================
        # Right Panel: 3D Viewport with Top Navigation Toolbar
        # ====================================================================
        view_container = QtWidgets.QWidget()
        view_layout = QtWidgets.QVBoxLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)

        # 3D Viewport Top Toolbar
        viewport_toolbar = QtWidgets.QFrame()
        viewport_toolbar.setStyleSheet(
            "QFrame { background-color: #17181d; border-bottom: 1px solid #282b36; padding: 3px; }"
        )
        tb_layout = QtWidgets.QHBoxLayout(viewport_toolbar)
        tb_layout.setContentsMargins(6, 3, 6, 3)
        tb_layout.setSpacing(6)

        # Camera preset buttons
        self.btn_reset_view = QtWidgets.QPushButton("🎯 Fit / Reset [R]")
        self.btn_reset_view.setToolTip("Fit camera to point cloud extents [R]")
        self.btn_reset_view.clicked.connect(self.reset_view)
        tb_layout.addWidget(self.btn_reset_view)

        self.btn_view_top = QtWidgets.QPushButton("🗺️ Top (XY) [1]")
        self.btn_view_top.setToolTip("Top orthographic map view [1]")
        self.btn_view_top.clicked.connect(self.set_view_top)
        tb_layout.addWidget(self.btn_view_top)

        self.btn_view_front = QtWidgets.QPushButton("📐 Front (XZ) [2]")
        self.btn_view_front.setToolTip("Front elevation cross-section [2]")
        self.btn_view_front.clicked.connect(self.set_view_front)
        tb_layout.addWidget(self.btn_view_front)

        self.btn_view_side = QtWidgets.QPushButton("📏 Side (YZ) [3]")
        self.btn_view_side.setToolTip("Side elevation profile [3]")
        self.btn_view_side.clicked.connect(self.set_view_side)
        tb_layout.addWidget(self.btn_view_side)

        self.btn_view_iso = QtWidgets.QPushButton("🌐 3D [4]")
        self.btn_view_iso.setToolTip("3D perspective view [4]")
        self.btn_view_iso.clicked.connect(self.set_view_iso)
        tb_layout.addWidget(self.btn_view_iso)

        tb_layout.addSpacing(10)

        # Background color selector
        tb_layout.addWidget(QtWidgets.QLabel("BG:"))
        self.bg_combo = QtWidgets.QComboBox()
        self.bg_combo.addItems(["Dark Slate", "Pure Black", "Deep Navy", "Neutral Gray", "Light / White", "Custom..."])
        self.bg_combo.currentTextChanged.connect(self._on_bg_color_selected)
        tb_layout.addWidget(self.bg_combo)

        tb_layout.addStretch(1)

        # Export Snapshot button
        self.btn_snapshot = QtWidgets.QPushButton("📸 Snapshot [Ctrl+S]")
        self.btn_snapshot.setToolTip("Export 3D viewport image to PNG file (Ctrl+S)")
        self.btn_snapshot.clicked.connect(self.export_screenshot)
        tb_layout.addWidget(self.btn_snapshot)

        view_layout.addWidget(viewport_toolbar)

        # 3D Viewport Widget
        self.view = CustomGLViewWidget()
        self.view.setBackgroundColor(QtGui.QColor("#16181d"))
        self.view.opts["distance"] = 200.0
        self.view.opts["elevation"] = 35.0
        self.view.opts["azimuth"] = -45.0
        self.view.camera_changed.connect(self._update_camera_status)

        # 3D Scatter plot item
        self.scatter = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=np.empty((0, 4), dtype=np.float32),
            size=float(self.cfg.point_size_px),
            pxMode=True,
            glOptions="opaque",
        )
        self.view.addItem(self.scatter)

        # Coordinate axes
        self.axis = gl.GLAxisItem()
        self.axis.setSize(x=50.0, y=50.0, z=25.0)
        self.view.addItem(self.axis)

        # Ground grid
        self.grid = gl.GLGridItem()
        self.grid.setSize(x=100.0, y=100.0)
        self.grid.setSpacing(x=10.0, y=10.0)
        self.grid.setColor(QtGui.QColor(95, 115, 140, 100))
        self.view.addItem(self.grid)

        # Bounding box wireframe
        self.bbox = gl.GLBoxItem()
        self.bbox.setColor((100, 180, 255, 120))
        self.bbox.setVisible(False)
        self.view.addItem(self.bbox)

        view_layout.addWidget(self.view, stretch=1)
        self.splitter.addWidget(view_container)

        # Default splitter proportion (sidebar 480px, viewport remaining)
        self.splitter.setSizes([480, 1040])

        # Status Bar
        self.status_bar = self.statusBar()
        self.camera_label = QtWidgets.QLabel("Camera: Dist 200.0m | Elev 35.0° | Azim -45.0°")
        self.camera_label.setStyleSheet("color: #8b92a4; padding-right: 12px;")
        self.points_label = QtWidgets.QLabel("Points: 0 visible / 0 sampled")
        self.points_label.setStyleSheet("color: #93c5fd; padding-right: 12px; font-weight: 500;")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setMaximumWidth(140)
        self.progress_bar.setVisible(False)

        self.status_bar.addWidget(self.progress_bar)
        self.status_bar.addPermanentWidget(self.camera_label)
        self.status_bar.addPermanentWidget(self.points_label)
        self.status_bar.showMessage("Ready. Drop or open a classified LAS/LAZ file to begin.")

    def _setup_shortcuts(self) -> None:
        """Register application-wide keyboard shortcuts."""
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, self.choose_file)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, self.reload_current_file)
        QtGui.QShortcut(QtGui.QKeySequence("F5"), self, self.reload_current_file)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, self.export_screenshot)
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, self.reset_view)
        QtGui.QShortcut(QtGui.QKeySequence("G"), self, self.toggle_grid)
        QtGui.QShortcut(QtGui.QKeySequence("X"), self, self.toggle_axis)
        QtGui.QShortcut(QtGui.QKeySequence("B"), self, self.toggle_bbox)
        QtGui.QShortcut(QtGui.QKeySequence("1"), self, self.set_view_top)
        QtGui.QShortcut(QtGui.QKeySequence("2"), self, self.set_view_front)
        QtGui.QShortcut(QtGui.QKeySequence("3"), self, self.set_view_side)
        QtGui.QShortcut(QtGui.QKeySequence("4"), self, self.set_view_iso)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+A"), self, lambda: self.set_all_classes_checked(True))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+D"), self, lambda: self.set_all_classes_checked(False))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+I"), self, self.invert_class_selection)

    # ========================================================================
    # Drag and Drop
    # ========================================================================
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".las", ".laz")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".las", ".laz")):
                event.acceptProposedAction()
                self.load_las_file(Path(file_path))
                return

    # ========================================================================
    # File Loading
    # ========================================================================
    def choose_file(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open classified LAS/LAZ point cloud",
            str(Path.cwd()),
            "Point clouds (*.las *.laz);;All files (*.*)",
        )
        if file_path:
            self.load_las_file(Path(file_path))

    def current_config_from_ui(self) -> ViewerConfig:
        return ViewerConfig(
            max_display_points=int(self.max_points_spin.value()),
            random_seed=self.cfg.random_seed,
            read_chunk_size=self.cfg.read_chunk_size,
            center_xy_for_view=self.cfg.center_xy_for_view,
            center_z_for_view=self.cfg.center_z_for_view,
            z_scale_for_view=float(self.z_scale_spin.value()),
            point_size_px=float(self.point_size_spin.value()),
            point_opacity=float(self.opacity_spin.value()) / 100.0,
        )

    def load_las_file(self, path: Path) -> None:
        if self.load_thread is not None:
            QtWidgets.QMessageBox.warning(self, "Busy", "A point cloud is currently loading. Please wait.")
            return

        try:
            path = validate_input_path(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid file", str(exc))
            return

        self.cfg = self.current_config_from_ui()
        self.open_button.setEnabled(False)
        self.reload_button.setEnabled(False)
        self.btn_apply_reload.setEnabled(False)
        self.progress_bar.setVisible(True)

        self.file_title_label.setText(path.name)
        self.file_subtitle_label.setText(f"Loading {path.parent} ...")
        self.statusBar().showMessage(f"Streaming and sampling {path.name} ...")

        thread = QtCore.QThread(self)
        worker = LasLoadWorker(path, self.cfg)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.on_load_finished)
        worker.failed.connect(self.on_load_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.on_load_thread_done)
        self.load_thread = thread
        self.load_worker = worker
        thread.start()

    def on_load_thread_done(self) -> None:
        self.load_thread = None
        self.load_worker = None
        self.open_button.setEnabled(True)
        self.reload_button.setEnabled(True)
        self.btn_apply_reload.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_load_failed(self, message: str) -> None:
        self.statusBar().showMessage("Failed to load point cloud.")
        self.file_subtitle_label.setText("Load error. See details in Metadata tab.")
        self.info_text.setPlainText(f"Error loading file:\n{message}")
        QtWidgets.QMessageBox.critical(self, "Loading failed", message)

    def on_load_finished(self, summary: LasViewSummary) -> None:
        self.summary = summary
        self.classification = summary.sampled_cloud.classification

        # Initialize colors for all classes found
        self._init_class_colors(summary)

        # Center coordinates
        self.centered_points, self.center_xyz = centered_points_for_view(
            summary.sampled_cloud.xyz, self.cfg
        )

        # Populate tables and UI
        self.file_title_label.setText(summary.input_path.name)
        file_size_mb = summary.file_size_bytes / (1024 * 1024)
        pct_sampled = (
            (summary.point_count_sampled / summary.point_count_total * 100.0)
            if summary.point_count_total > 0
            else 0.0
        )
        self.file_subtitle_label.setText(
            f"{summary.point_count_total:,} points ({file_size_mb:.1f} MB) | "
            f"{summary.point_count_sampled:,} displayed ({pct_sampled:.1f}%)"
        )

        self.populate_class_table(summary)
        self._recompute_all_colors()
        self.update_visible_classes()
        self.reset_view()
        self.update_info_metadata()
        self.statusBar().showMessage(f"Loaded {summary.input_path.name} successfully.")

    def reload_current_file(self) -> None:
        if self.summary is None:
            return
        self.load_las_file(self.summary.input_path)

    # ========================================================================
    # Color Management & Palettes
    # ========================================================================
    def _init_class_colors(self, summary: LasViewSummary) -> None:
        """Initialize color mappings for all classes in this cloud."""
        self.class_colors = dict(CLASS_COLOR_MAP)
        for code in summary.classification_counts.keys():
            if code not in self.class_colors:
                self.class_colors[code] = generate_distinct_color(code)

    def _recompute_all_colors(self) -> None:
        """Recompute full RGBA array based on current color mode, palettes, and opacity."""
        if self.classification is None or self.centered_points is None:
            return

        alpha = float(self.opacity_spin.value()) / 100.0

        if self.current_color_mode == "Classification":
            colors = np.empty((self.classification.size, 4), dtype=np.float32)
            fallback = np.asarray((*FALLBACK_CLASS_COLOR, alpha), dtype=np.float32)
            colors[:] = fallback
            for code, color in self.class_colors.items():
                mask = self.classification == code
                if np.any(mask):
                    colors[mask] = np.asarray((*color, alpha), dtype=np.float32)
            self.colors_rgba = colors
        else:
            # Elevation colormap
            cmap_name = self.current_color_mode.split(":")[-1].split("(")[0].strip()
            z = self.centered_points[:, 2]
            z_min = float(np.nanmin(z))
            z_max = float(np.nanmax(z))
            span = max(z_max - z_min, 1e-6)
            norm_z = np.clip((z - z_min) / span, 0.0, 1.0)
            self.colors_rgba = get_elevation_colormap_colors(norm_z, cmap_name, alpha)

    def set_class_color(self, code: int, new_rgb: tuple[float, float, float]) -> None:
        """Set custom color for a single class and immediately update display."""
        self.class_colors[code] = new_rgb

        # Update button swatch
        btn = self.class_color_buttons.get(code)
        if btn is not None:
            btn.setStyleSheet(
                f"background-color: {color_to_css(new_rgb)}; "
                "border: 1px solid #778; border-radius: 4px; margin: 2px;"
            )

        # Update OpenGL scatter buffer
        if (
            self.colors_rgba is not None
            and self.classification is not None
            and self.current_color_mode == "Classification"
        ):
            mask = self.classification == code
            if np.any(mask):
                alpha = float(self.opacity_spin.value()) / 100.0
                self.colors_rgba[mask, 0] = new_rgb[0]
                self.colors_rgba[mask, 1] = new_rgb[1]
                self.colors_rgba[mask, 2] = new_rgb[2]
                self.colors_rgba[mask, 3] = alpha
                self.update_visible_classes()

    def choose_class_color(self, code: int) -> None:
        """Open QColorDialog to pick a custom color for a class."""
        current_rgb = self.class_colors.get(code, FALLBACK_CLASS_COLOR)
        init_qcolor = QtGui.QColor.fromRgbF(*current_rgb)
        label = CLASS_LABEL_MAP.get(code, "Custom Class")

        dialog = QtWidgets.QColorDialog(init_qcolor, self)
        dialog.setWindowTitle(f"Choose Color for Class {code}: {label}")
        if dialog.exec():
            col = dialog.selectedColor()
            if col.isValid():
                new_rgb = (col.redF(), col.greenF(), col.blueF())
                self.set_class_color(code, new_rgb)

    def apply_palette_preset(self, palette_name: str) -> None:
        """Apply a preset categorical palette to all classes."""
        if not self.class_checkboxes:
            return
        palette = PALETTES.get(palette_name, CLASS_COLOR_MAP)
        for code in self.class_checkboxes.keys():
            if code in palette:
                rgb = palette[code]
            else:
                rgb = generate_distinct_color(code)
            self.class_colors[code] = rgb
            btn = self.class_color_buttons.get(code)
            if btn is not None:
                btn.setStyleSheet(
                    f"background-color: {color_to_css(rgb)}; "
                    "border: 1px solid #778; border-radius: 4px; margin: 2px;"
                )
        self._recompute_all_colors()
        self.update_visible_classes()

    def reset_all_class_colors(self) -> None:
        """Reset all colors to default ASPRS standard."""
        self.palette_combo.blockSignals(True)
        self.palette_combo.setCurrentText("ASPRS Standard")
        self.palette_combo.blockSignals(False)
        self.apply_palette_preset("ASPRS Standard")

    def _on_color_mode_changed(self, index: int) -> None:
        mode_text = self.color_mode_combo.currentText()
        if "Classification" in mode_text:
            self.current_color_mode = "Classification"
        else:
            self.current_color_mode = mode_text
        self._recompute_all_colors()
        self.update_visible_classes()

    # ========================================================================
    # Class Table & Filtering
    # ========================================================================
    def populate_class_table(self, summary: LasViewSummary) -> None:
        self.class_table.blockSignals(True)
        self.class_table.setRowCount(0)
        self.class_checkboxes.clear()
        self.class_color_buttons.clear()

        total_pts = summary.point_count_total
        total_smp = summary.point_count_sampled

        for row, code in enumerate(summary.classification_counts.keys()):
            self.class_table.insertRow(row)

            # Col 0: Checkbox
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(True)
            checkbox.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            checkbox.stateChanged.connect(self.update_visible_classes)
            cb_container = QtWidgets.QWidget()
            cb_layout = QtWidgets.QHBoxLayout(cb_container)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            cb_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            cb_layout.addWidget(checkbox)
            self.class_table.setCellWidget(row, 0, cb_container)
            self.class_checkboxes[int(code)] = checkbox

            # Col 1: Clickable color swatch button
            rgb = self.class_colors.get(code, FALLBACK_CLASS_COLOR)
            color_btn = QtWidgets.QPushButton()
            color_btn.setFixedSize(36, 20)
            color_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            color_btn.setToolTip(f"Click to change color for class {code}")
            color_btn.setStyleSheet(
                f"background-color: {color_to_css(rgb)}; "
                "border: 1px solid #778; border-radius: 4px; margin: 2px;"
            )
            color_btn.clicked.connect(lambda checked=False, c=code: self.choose_class_color(c))
            self.class_table.setCellWidget(row, 1, color_btn)
            self.class_color_buttons[int(code)] = color_btn

            # Col 2: Class ID
            id_item = QtWidgets.QTableWidgetItem(str(code))
            id_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.class_table.setItem(row, 2, id_item)

            # Col 3: Label
            label_text = CLASS_LABEL_MAP.get(code, f"User Defined ({code})")
            label_item = QtWidgets.QTableWidgetItem(label_text)
            self.class_table.setItem(row, 3, label_item)

            # Col 4: Sampled Points
            smp_n = summary.sampled_classification_counts.get(code, 0)
            smp_pct = (smp_n / total_smp * 100.0) if total_smp > 0 else 0.0
            smp_item = QtWidgets.QTableWidgetItem(f"{smp_n:,} ({smp_pct:.1f}%)")
            smp_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self.class_table.setItem(row, 4, smp_item)

            # Col 5: Total Points
            tot_n = summary.classification_counts.get(code, 0)
            tot_pct = (tot_n / total_pts * 100.0) if total_pts > 0 else 0.0
            tot_item = QtWidgets.QTableWidgetItem(f"{tot_n:,} ({tot_pct:.1f}%)")
            tot_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self.class_table.setItem(row, 5, tot_item)

        self.class_table.blockSignals(False)
        self.class_table.resizeRowsToContents()

    def _filter_class_table(self, text: str) -> None:
        """Filter table rows by search query."""
        query = text.strip().lower()
        for row in range(self.class_table.rowCount()):
            if not query:
                self.class_table.setRowHidden(row, False)
                continue
            code_item = self.class_table.item(row, 2)
            label_item = self.class_table.item(row, 3)
            code_str = code_item.text().lower() if code_item else ""
            label_str = label_item.text().lower() if label_item else ""
            matched = (query in code_str) or (query in label_str)
            self.class_table.setRowHidden(row, not matched)

    def _on_table_double_clicked(self, row: int, col: int) -> None:
        """Double-clicking a row solos that classification code."""
        code_item = self.class_table.item(row, 2)
        if not code_item:
            return
        code = int(code_item.text())
        self.solo_class(code)

    def _on_table_context_menu(self, pos: QtCore.QPoint) -> None:
        """Show context menu on right-click."""
        row = self.class_table.rowAt(pos.y())
        if row < 0:
            return
        code_item = self.class_table.item(row, 2)
        if not code_item:
            return
        code = int(code_item.text())
        label = CLASS_LABEL_MAP.get(code, f"Class {code}")

        menu = QtWidgets.QMenu(self)
        action_color = menu.addAction(f"🎨 Change Color for {code}...")
        action_reset = menu.addAction("🔄 Reset to Default Color")
        menu.addSeparator()
        action_solo = menu.addAction(f"👁️ Solo Class {code} ({label})")
        action_hide = menu.addAction(f"🚫 Hide Class {code}")
        menu.addSeparator()
        action_all = menu.addAction("👁️‍🗨️ Select All Classes")
        action_none = menu.addAction("❌ Deselect All Classes")
        action_inv = menu.addAction("🔀 Invert Selection")

        chosen = menu.exec(self.class_table.viewport().mapToGlobal(pos))
        if chosen == action_color:
            self.choose_class_color(code)
        elif chosen == action_reset:
            default_rgb = CLASS_COLOR_MAP.get(code, generate_distinct_color(code))
            self.set_class_color(code, default_rgb)
        elif chosen == action_solo:
            self.solo_class(code)
        elif chosen == action_hide:
            if code in self.class_checkboxes:
                self.class_checkboxes[code].setChecked(False)
        elif chosen == action_all:
            self.set_all_classes_checked(True)
        elif chosen == action_none:
            self.set_all_classes_checked(False)
        elif chosen == action_inv:
            self.invert_class_selection()

    def solo_class(self, target_code: int) -> None:
        """Solo target class: show only this class, or restore all if already soloed."""
        selected = self.selected_class_codes()
        if selected == [target_code]:
            # Restore all classes
            self.set_all_classes_checked(True)
            return

        for code, checkbox in self.class_checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(code == target_code)
            checkbox.blockSignals(False)
        self.update_visible_classes()

    def selected_class_codes(self) -> list[int]:
        return [code for code, cb in self.class_checkboxes.items() if cb.isChecked()]

    def set_all_classes_checked(self, checked: bool) -> None:
        if not self.class_checkboxes:
            return
        for cb in self.class_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self.update_visible_classes()

    def invert_class_selection(self) -> None:
        if not self.class_checkboxes:
            return
        for cb in self.class_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(not cb.isChecked())
            cb.blockSignals(False)
        self.update_visible_classes()

    # ========================================================================
    # Sliders & View Parameter Adjustments
    # ========================================================================
    def _on_point_size_slider_moved(self, value: int) -> None:
        self.point_size_spin.blockSignals(True)
        self.point_size_spin.setValue(value / 10.0)
        self.point_size_spin.blockSignals(False)
        self.update_visible_classes()

    def _on_point_size_spin_changed(self, value: float) -> None:
        self.point_size_slider.blockSignals(True)
        self.point_size_slider.setValue(int(round(value * 10)))
        self.point_size_slider.blockSignals(False)
        self.update_visible_classes()

    def _on_opacity_slider_moved(self, value: int) -> None:
        self.opacity_spin.blockSignals(True)
        self.opacity_spin.setValue(value)
        self.opacity_spin.blockSignals(False)
        self._recompute_all_colors()
        self.update_visible_classes()

    def _on_opacity_spin_changed(self, value: int) -> None:
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(value)
        self.opacity_slider.blockSignals(False)
        self._recompute_all_colors()
        self.update_visible_classes()

    def _on_z_scale_slider_moved(self, value: int) -> None:
        self.z_scale_spin.blockSignals(True)
        self.z_scale_spin.setValue(value / 10.0)
        self.z_scale_spin.blockSignals(False)
        self.apply_z_scale()

    def _on_z_scale_spin_changed(self, value: float) -> None:
        self.z_scale_slider.blockSignals(True)
        self.z_scale_slider.setValue(int(round(value * 10)))
        self.z_scale_slider.blockSignals(False)
        self.apply_z_scale()

    def apply_z_scale(self) -> None:
        if self.summary is None:
            return
        self.cfg = self.current_config_from_ui()
        self.centered_points, self.center_xyz = centered_points_for_view(
            self.summary.sampled_cloud.xyz, self.cfg
        )
        self._update_scene_guides()
        if self.current_color_mode != "Classification":
            self._recompute_all_colors()
        self.update_visible_classes()

    # ========================================================================
    # 3D Scene Guides (Grid, Axis, BBox)
    # ========================================================================
    def set_grid_visible(self, visible: bool) -> None:
        self.grid.setVisible(visible)
        self.cb_grid.blockSignals(True)
        self.cb_grid.setChecked(visible)
        self.cb_grid.blockSignals(False)

    def toggle_grid(self) -> None:
        self.set_grid_visible(not self.grid.visible())

    def set_axis_visible(self, visible: bool) -> None:
        self.axis.setVisible(visible)
        self.cb_axis.blockSignals(True)
        self.cb_axis.setChecked(visible)
        self.cb_axis.blockSignals(False)

    def toggle_axis(self) -> None:
        self.set_axis_visible(not self.axis.visible())

    def set_bbox_visible(self, visible: bool) -> None:
        self.bbox.setVisible(visible)
        self.cb_bbox.blockSignals(True)
        self.cb_bbox.setChecked(visible)
        self.cb_bbox.blockSignals(False)

    def toggle_bbox(self) -> None:
        self.set_bbox_visible(not self.bbox.visible())

    def _on_grid_ground_toggled(self, checked: bool) -> None:
        self.grid_on_ground = checked
        self._update_scene_guides()

    def _update_scene_guides(self) -> None:
        if self.centered_points is None or self.centered_points.size == 0:
            return

        xyz = self.centered_points
        min_x = float(np.nanmin(xyz[:, 0]))
        max_x = float(np.nanmax(xyz[:, 0]))
        min_y = float(np.nanmin(xyz[:, 1]))
        max_y = float(np.nanmax(xyz[:, 1]))
        min_z = float(np.nanmin(xyz[:, 2]))
        max_z = float(np.nanmax(xyz[:, 2]))

        dx = max(max_x - min_x, 1.0)
        dy = max(max_y - min_y, 1.0)
        dz = max(max_z - min_z, 1.0)

        # Update bounding box wireframe
        self.bbox.setSize(x=dx, y=dy, z=dz)
        self.bbox.resetTransform()
        self.bbox.translate(min_x, min_y, min_z)

        # Update ground grid size and position
        grid_dim = float(max(20.0, max(dx, dy) * 1.15))
        spacing = _nice_grid_spacing(grid_dim / 10.0)
        self.grid.setSize(x=grid_dim, y=grid_dim)
        self.grid.setSpacing(x=spacing, y=spacing)
        self.grid.resetTransform()
        if self.grid_on_ground:
            self.grid.translate(0.0, 0.0, min_z - 0.1)
        else:
            self.grid.translate(0.0, 0.0, 0.0)

        # Update coordinate axes proportional to data
        axis_len = float(max(10.0, max(dx, dy) * 0.15))
        self.axis.setSize(x=axis_len, y=axis_len, z=max(axis_len * 0.5, dz * 0.25))

    # ========================================================================
    # Camera Presets & View Controls
    # ========================================================================
    def reset_view(self) -> None:
        """Fit camera to view the point cloud."""
        if self.centered_points is None or self.centered_points.size == 0:
            self.view.setCameraPosition(
                pos=pg.Vector(0, 0, 0), distance=200.0, elevation=35.0, azimuth=-45.0
            )
            return

        xyz = self.centered_points
        xy_extent = np.nanmax(xyz[:, :2], axis=0) - np.nanmin(xyz[:, :2], axis=0)
        z_extent = float(np.nanmax(xyz[:, 2]) - np.nanmin(xyz[:, 2]))
        distance = float(max(30.0, float(np.nanmax(xy_extent)), z_extent) * 1.8)

        self.view.setCameraPosition(
            pos=pg.Vector(0, 0, 0), distance=distance, elevation=35.0, azimuth=-45.0
        )
        self._update_scene_guides()
        self._update_camera_status()

    def set_view_top(self) -> None:
        """Top-down bird's-eye view (XY plane)."""
        self.view.setCameraPosition(elevation=89.9, azimuth=-90.0)
        self._update_camera_status()

    def set_view_front(self) -> None:
        """Front elevation profile view (XZ plane)."""
        self.view.setCameraPosition(elevation=0.0, azimuth=-90.0)
        self._update_camera_status()

    def set_view_side(self) -> None:
        """Side elevation profile view (YZ plane)."""
        self.view.setCameraPosition(elevation=0.0, azimuth=0.0)
        self._update_camera_status()

    def set_view_iso(self) -> None:
        """Perspective 3D view."""
        self.view.setCameraPosition(elevation=35.0, azimuth=-45.0)
        self._update_camera_status()

    def _update_camera_status(self) -> None:
        opts = self.view.opts
        dist = float(opts.get("distance", 0.0))
        elev = float(opts.get("elevation", 0.0))
        azim = float(opts.get("azimuth", 0.0))
        self.camera_label.setText(
            f"Camera: Dist {dist:.1f}m | Elev {elev:.1f}° | Azim {azim:.1f}°"
        )

    # ========================================================================
    # Viewport Background & Rendering
    # ========================================================================
    def _on_bg_color_selected(self, choice: str) -> None:
        if choice == "Dark Slate":
            bg = QtGui.QColor("#16181d")
        elif choice == "Pure Black":
            bg = QtGui.QColor("#000000")
        elif choice == "Deep Navy":
            bg = QtGui.QColor("#0b0f19")
        elif choice == "Neutral Gray":
            bg = QtGui.QColor("#2d3139")
        elif choice == "Light / White":
            bg = QtGui.QColor("#f4f5f8")
        elif choice == "Custom...":
            col = QtWidgets.QColorDialog.getColor(
                QtGui.QColor("#16181d"), self, "Select Viewport Background Color"
            )
            if col.isValid():
                bg = col
            else:
                return
        else:
            bg = QtGui.QColor("#16181d")

        self.view.setBackgroundColor(bg)

        # Adapt grid and bounding box contrast based on background luminance
        lum = (0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()) / 255.0
        if lum > 0.5:
            self.grid.setColor(QtGui.QColor(140, 140, 140, 160))
            self.bbox.setColor((40, 80, 140, 180))
        else:
            self.grid.setColor(QtGui.QColor(95, 115, 140, 100))
            self.bbox.setColor((100, 180, 255, 120))

    def update_visible_classes(self) -> None:
        """Update visible points in GLScatterPlotItem."""
        if (
            self.summary is None
            or self.centered_points is None
            or self.colors_rgba is None
            or self.classification is None
        ):
            return

        selected = self.selected_class_codes()
        if selected:
            mask = np.isin(self.classification, np.asarray(selected, dtype=np.uint8))
            pos = self.centered_points[mask]
            color = self.colors_rgba[mask]
        else:
            pos = np.empty((0, 3), dtype=np.float32)
            color = np.empty((0, 4), dtype=np.float32)

        alpha = float(self.opacity_spin.value()) / 100.0
        self.scatter.setGLOptions("opaque" if alpha >= 0.99 else "translucent")
        self.scatter.setData(
            pos=pos,
            color=color,
            size=float(self.point_size_spin.value()),
            pxMode=True,
        )

        total_smp = self.summary.point_count_sampled
        vis_n = int(pos.shape[0])
        vis_pct = (vis_n / total_smp * 100.0) if total_smp > 0 else 0.0
        self.points_label.setText(
            f"Visible: {vis_n:,} / {total_smp:,} ({vis_pct:.1f}%)"
        )
        self.statusBar().showMessage(
            f"Showing {vis_n:,} points across {len(selected)} active classes."
        )

    # ========================================================================
    # Snapshot / Screenshot Export
    # ========================================================================
    def export_screenshot(self) -> None:
        default_name = f"pointcloud_{self.summary.input_path.stem if self.summary else 'view'}.png"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save 3D View Screenshot",
            str(Path.cwd() / default_name),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )
        if not file_path:
            return

        image = self.view.readQImage()
        if image.isNull():
            QtWidgets.QMessageBox.warning(self, "Export Failed", "Could not capture 3D viewport image.")
            return

        if image.save(file_path):
            self.statusBar().showMessage(f"Screenshot saved to: {Path(file_path).name}", 4000)
        else:
            QtWidgets.QMessageBox.critical(self, "Export Failed", f"Failed to save screenshot to {file_path}")

    # ========================================================================
    # Metadata & Extents
    # ========================================================================
    def update_info_metadata(self) -> None:
        if self.summary is None:
            return
        s = self.summary
        cx, cy, cz = self.center_xyz
        bx, by, bz = s.bounds_xyz
        dx = bx[1] - bx[0]
        dy = by[1] - by[0]
        dz = bz[1] - bz[0]
        file_mb = s.file_size_bytes / (1024 * 1024)

        lines = [
            "============================================================",
            "FILE & HEADER INFORMATION",
            "============================================================",
            f"File Path:         {s.input_path}",
            f"File Size:         {file_mb:.2f} MB ({s.file_size_bytes:,} bytes)",
            f"LAS Version:       {s.las_version}",
            f"Point Format ID:   {s.point_format_id}",
            "",
            "============================================================",
            "POINT COUNTS & SAMPLING",
            "============================================================",
            f"Total Points:      {s.point_count_total:,}",
            f"Finite XYZ Points: {s.point_count_valid_xyz:,}",
            f"Sampled Display:   {s.point_count_sampled:,} ({(s.point_count_sampled / max(s.point_count_total, 1) * 100):.2f}%)",
            "",
            "============================================================",
            "SPATIAL EXTENTS & COORDINATES",
            "============================================================",
            f"X Extent:          [{bx[0]:.3f}, {bx[1]:.3f}]  (Span: {dx:.2f} m)",
            f"Y Extent:          [{by[0]:.3f}, {by[1]:.3f}]  (Span: {dy:.2f} m)",
            f"Z Extent:          [{bz[0]:.3f}, {bz[1]:.3f}]  (Span: {dz:.2f} m)",
            f"Display Center:    X={cx:.3f}, Y={cy:.3f}, Z={cz:.3f}",
            f"Current Z Scale:   {self.z_scale_spin.value():.2f}x",
            "",
            "============================================================",
            "COORDINATE REFERENCE SYSTEM (CRS)",
            "============================================================",
            f"CRS Name:          {s.crs_name or 'Not specified'}",
            f"EPSG Code:         {s.crs_epsg if s.crs_epsg is not None else 'None'}",
            f"WKT / CRS Text:    {s.crs_text}",
            "",
            "============================================================",
            "CLASSIFICATION SUMMARY",
            "============================================================",
        ]
        for code, count in s.classification_counts.items():
            smp = s.sampled_classification_counts.get(code, 0)
            lbl = CLASS_LABEL_MAP.get(code, "Unmapped class")
            lines.append(f"  Class {code:2d} ({lbl:<26}): {count:>10,} total | {smp:>8,} sampled")

        if s.unknown_class_codes:
            lines.append(
                f"\nNon-standard Class Codes Detected: {', '.join(map(str, s.unknown_class_codes))}"
            )

        self.info_text.setPlainText("\n".join(lines))

    def copy_metadata_to_clipboard(self) -> None:
        text = self.info_text.toPlainText()
        if text:
            QtWidgets.QApplication.clipboard().setText(text)
            self.statusBar().showMessage("Metadata copied to clipboard!", 3000)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qt LAS/LAZ classification point-cloud viewer.")
    parser.add_argument("input", nargs="?", type=Path, help="Optional LAS/LAZ file to open at startup.")
    parser.add_argument("--max-display-points", type=int, default=800_000, help="Maximum sampled points for display.")
    parser.add_argument("--read-chunk-size", type=int, default=2_000_000, help="LAS/LAZ read chunk size.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for display sampling.")
    parser.add_argument("--z-scale", type=float, default=1.0, help="Initial vertical scale for display.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Initial point size in screen pixels.")
    parser.add_argument("--opacity", type=float, default=1.0, help="Initial point opacity (0.1 to 1.0).")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = ViewerConfig(
        max_display_points=args.max_display_points,
        random_seed=args.random_seed,
        read_chunk_size=args.read_chunk_size,
        z_scale_for_view=args.z_scale,
        point_size_px=args.point_size,
        point_opacity=args.opacity,
    )

    app = QtWidgets.QApplication(sys.argv)
    win = ClassificationPointCloudViewer(initial_path=args.input, cfg=cfg)
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
