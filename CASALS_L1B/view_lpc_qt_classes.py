"""Qt LAS/LAZ point-cloud classification viewer.

Purpose
-------
Open a classified LAS/LAZ point cloud, display sampled points in a Qt-native
OpenGL widget, and toggle visibility by LAS classification code.

This tool is intended for interactive inspection only. It does not modify the
source LAS/LAZ file and does not write derivative point-cloud products.

Recommended installation
------------------------
conda install -c conda-forge numpy laspy lazrs pyqtgraph pyopengl pyqt

Run
---
python view_lpc_qt_classes.py
python view_lpc_qt_classes.py path/to/cloud.laz --max-display-points 800000
"""

from __future__ import annotations

import argparse
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


CLASS_LABEL_MAP: dict[int, str] = {
    0: "never classified / unknown",
    1: "processed / unclassified",
    2: "ground / bare earth",
    3: "low vegetation",
    4: "medium vegetation",
    5: "high vegetation",
    6: "building",
    7: "low noise",
    9: "water",
    17: "bridge deck",
    18: "high noise",
    20: "ignored ground",
    21: "snow",
    22: "temporal exclusion",
}

CLASS_COLOR_MAP: dict[int, tuple[float, float, float]] = {
    0: (0.20, 0.20, 0.20),
    1: (0.35, 0.52, 0.72),
    2: (0.15, 0.75, 0.20),
    3: (0.55, 0.90, 0.35),
    4: (0.25, 0.70, 0.25),
    5: (0.05, 0.45, 0.10),
    6: (0.78, 0.48, 0.18),
    7: (0.92, 0.15, 0.10),
    9: (0.05, 0.78, 0.88),
    17: (0.62, 0.32, 0.82),
    18: (1.00, 0.05, 0.05),
    20: (0.98, 0.82, 0.10),
    21: (1.00, 1.00, 1.00),
    22: (0.92, 0.12, 0.75),
}

FALLBACK_CLASS_COLOR = (0.52, 0.47, 0.62)


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
    )


def class_color_rgb(code: int) -> tuple[float, float, float]:
    return CLASS_COLOR_MAP.get(int(code), FALLBACK_CLASS_COLOR)


def classification_colors_rgba(classification: np.ndarray) -> np.ndarray:
    cls = np.asarray(classification, dtype=np.uint8)
    colors = np.empty((cls.size, 4), dtype=np.float32)
    fallback = np.asarray((*FALLBACK_CLASS_COLOR, 1.0), dtype=np.float32)
    colors[:] = fallback
    for code, color in CLASS_COLOR_MAP.items():
        colors[cls == code] = np.asarray((*color, 1.0), dtype=np.float32)
    return colors


def centered_points_for_view(xyz: np.ndarray, cfg: ViewerConfig) -> tuple[np.ndarray, tuple[float, float, float]]:
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


def color_to_css(color: tuple[float, float, float]) -> str:
    r = int(round(color[0] * 255.0))
    g = int(round(color[1] * 255.0))
    b = int(round(color[2] * 255.0))
    return f"rgb({r}, {g}, {b})"


class ClassificationPointCloudViewer(QtWidgets.QMainWindow):
    def __init__(self, initial_path: Optional[Path] = None, cfg: Optional[ViewerConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ViewerConfig()
        self.summary: Optional[LasViewSummary] = None
        self.centered_points: Optional[np.ndarray] = None
        self.colors_rgba: Optional[np.ndarray] = None
        self.classification: Optional[np.ndarray] = None
        self.class_checkboxes: dict[int, QtWidgets.QCheckBox] = {}
        self.load_thread: Optional[QtCore.QThread] = None
        self.load_worker: Optional[LasLoadWorker] = None

        self.setWindowTitle("LAS/LAZ Classification Viewer - Qt")
        self.resize(1500, 950)
        self._build_ui()

        if initial_path is not None:
            QtCore.QTimer.singleShot(0, lambda: self.load_las_file(Path(initial_path)))

    def _build_ui(self) -> None:
        pg.setConfigOptions(antialias=False)

        main = QtWidgets.QWidget(self)
        self.setCentralWidget(main)
        root_layout = QtWidgets.QHBoxLayout(main)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        left_panel = QtWidgets.QWidget(main)
        left_panel.setMinimumWidth(390)
        left_panel.setMaximumWidth(500)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        self.open_button = QtWidgets.QPushButton("Open LAS/LAZ")
        self.open_button.clicked.connect(self.choose_file)
        left_layout.addWidget(self.open_button)

        controls_group = QtWidgets.QGroupBox("Display controls")
        controls_layout = QtWidgets.QGridLayout(controls_group)

        controls_layout.addWidget(QtWidgets.QLabel("Max sampled points"), 0, 0)
        self.max_points_spin = QtWidgets.QSpinBox()
        self.max_points_spin.setRange(10_000, 20_000_000)
        self.max_points_spin.setSingleStep(50_000)
        self.max_points_spin.setValue(self.cfg.max_display_points)
        controls_layout.addWidget(self.max_points_spin, 0, 1)

        controls_layout.addWidget(QtWidgets.QLabel("Point size (px)"), 1, 0)
        self.point_size_spin = QtWidgets.QDoubleSpinBox()
        self.point_size_spin.setRange(0.5, 20.0)
        self.point_size_spin.setSingleStep(0.5)
        self.point_size_spin.setValue(self.cfg.point_size_px)
        self.point_size_spin.valueChanged.connect(self.update_visible_classes)
        controls_layout.addWidget(self.point_size_spin, 1, 1)

        controls_layout.addWidget(QtWidgets.QLabel("Z scale"), 2, 0)
        self.z_scale_spin = QtWidgets.QDoubleSpinBox()
        self.z_scale_spin.setRange(0.05, 50.0)
        self.z_scale_spin.setDecimals(2)
        self.z_scale_spin.setSingleStep(0.25)
        self.z_scale_spin.setValue(self.cfg.z_scale_for_view)
        self.z_scale_spin.valueChanged.connect(self.apply_z_scale)
        controls_layout.addWidget(self.z_scale_spin, 2, 1)

        self.reload_button = QtWidgets.QPushButton("Reload with max points")
        self.reload_button.clicked.connect(self.reload_current_file)
        controls_layout.addWidget(self.reload_button, 3, 0, 1, 2)

        left_layout.addWidget(controls_group)

        button_row = QtWidgets.QHBoxLayout()
        self.select_all_button = QtWidgets.QPushButton("Select all")
        self.clear_all_button = QtWidgets.QPushButton("Clear all")
        self.reset_view_button = QtWidgets.QPushButton("Reset view")
        self.select_all_button.clicked.connect(lambda: self.set_all_classes_checked(True))
        self.clear_all_button.clicked.connect(lambda: self.set_all_classes_checked(False))
        self.reset_view_button.clicked.connect(self.reset_view)
        button_row.addWidget(self.select_all_button)
        button_row.addWidget(self.clear_all_button)
        button_row.addWidget(self.reset_view_button)
        left_layout.addLayout(button_row)

        self.class_table = QtWidgets.QTableWidget(0, 6)
        self.class_table.setHorizontalHeaderLabels(["Visible", "Color", "Class", "Label", "Total", "Sampled"])
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.class_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.class_table.horizontalHeader().setStretchLastSection(False)
        self.class_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.class_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.class_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.class_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.class_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.class_table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        left_layout.addWidget(self.class_table, stretch=1)

        self.info_text = QtWidgets.QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumBlockCount(200)
        self.info_text.setPlaceholderText("Open a classified LAS/LAZ file to inspect class counts.")
        self.info_text.setMinimumHeight(160)
        left_layout.addWidget(self.info_text)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("k")
        self.view.opts["distance"] = 200.0
        self.view.opts["elevation"] = 30.0
        self.view.opts["azimuth"] = -45.0

        self.scatter = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=np.empty((0, 4), dtype=np.float32),
            size=float(self.cfg.point_size_px),
            pxMode=True,
            glOptions="opaque",
        )
        self.view.addItem(self.scatter)

        self.axis = gl.GLAxisItem()
        self.axis.setSize(x=50.0, y=50.0, z=25.0)
        self.view.addItem(self.axis)

        self.grid = gl.GLGridItem()
        self.grid.setSize(x=100.0, y=100.0)
        self.grid.setSpacing(x=10.0, y=10.0)
        self.view.addItem(self.grid)

        root_layout.addWidget(left_panel)
        root_layout.addWidget(self.view, stretch=1)

        self.statusBar().showMessage("Ready")

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
        )

    def load_las_file(self, path: Path) -> None:
        if self.load_thread is not None:
            QtWidgets.QMessageBox.warning(self, "Busy", "A point cloud is still loading.")
            return

        try:
            path = validate_input_path(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid input", str(exc))
            return

        self.cfg = self.current_config_from_ui()
        self.open_button.setEnabled(False)
        self.reload_button.setEnabled(False)
        self.statusBar().showMessage(f"Loading {path.name} ...")
        self.info_text.setPlainText(f"Loading:\n{path}\n\nThis may take time for large LAZ files.")

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

    def on_load_failed(self, message: str) -> None:
        self.statusBar().showMessage("Loading failed")
        self.info_text.setPlainText(message)
        QtWidgets.QMessageBox.critical(self, "Loading failed", message)

    def on_load_finished(self, summary: LasViewSummary) -> None:
        self.summary = summary
        self.classification = summary.sampled_cloud.classification
        self.colors_rgba = classification_colors_rgba(self.classification)
        self.centered_points, center_xyz = centered_points_for_view(summary.sampled_cloud.xyz, self.cfg)

        self.populate_class_table(summary)
        self.update_visible_classes()
        self.reset_view()
        self.update_info_text(center_xyz)
        self.statusBar().showMessage(f"Loaded {summary.input_path.name}")

    def reload_current_file(self) -> None:
        if self.summary is None:
            return
        self.load_las_file(self.summary.input_path)

    def populate_class_table(self, summary: LasViewSummary) -> None:
        self.class_table.blockSignals(True)
        self.class_table.setRowCount(0)
        self.class_checkboxes.clear()

        for row, code in enumerate(summary.classification_counts.keys()):
            self.class_table.insertRow(row)

            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_visible_classes)
            checkbox_container = QtWidgets.QWidget()
            checkbox_layout = QtWidgets.QHBoxLayout(checkbox_container)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            checkbox_layout.addWidget(checkbox)
            self.class_table.setCellWidget(row, 0, checkbox_container)
            self.class_checkboxes[int(code)] = checkbox

            swatch = QtWidgets.QLabel()
            swatch.setMinimumSize(28, 18)
            swatch.setStyleSheet(
                f"background-color: {color_to_css(class_color_rgb(code))}; "
                "border: 1px solid #555;"
            )
            self.class_table.setCellWidget(row, 1, swatch)

            self.class_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(code)))
            self.class_table.setItem(row, 3, QtWidgets.QTableWidgetItem(CLASS_LABEL_MAP.get(code, "unmapped class")))
            self.class_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{summary.classification_counts[code]:,}"))
            sampled_n = summary.sampled_classification_counts.get(code, 0)
            self.class_table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{sampled_n:,}"))

        self.class_table.blockSignals(False)
        self.class_table.resizeRowsToContents()

    def selected_class_codes(self) -> list[int]:
        return [code for code, checkbox in self.class_checkboxes.items() if checkbox.isChecked()]

    def set_all_classes_checked(self, checked: bool) -> None:
        if not self.class_checkboxes:
            return
        for checkbox in self.class_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)
        self.update_visible_classes()

    def apply_z_scale(self) -> None:
        if self.summary is None:
            return
        self.cfg = self.current_config_from_ui()
        self.centered_points, center_xyz = centered_points_for_view(self.summary.sampled_cloud.xyz, self.cfg)
        self.update_visible_classes()
        self.update_info_text(center_xyz)

    def update_visible_classes(self) -> None:
        if self.summary is None or self.centered_points is None or self.colors_rgba is None or self.classification is None:
            return

        selected = self.selected_class_codes()
        if selected:
            mask = np.isin(self.classification, np.asarray(selected, dtype=np.uint8))
            pos = self.centered_points[mask]
            color = self.colors_rgba[mask]
        else:
            pos = np.empty((0, 3), dtype=np.float32)
            color = np.empty((0, 4), dtype=np.float32)

        self.scatter.setData(
            pos=pos,
            color=color,
            size=float(self.point_size_spin.value()),
            pxMode=True,
        )
        self.statusBar().showMessage(
            f"Visible sampled points: {pos.shape[0]:,} / {self.summary.point_count_sampled:,}"
        )

    def update_info_text(self, center_xyz: tuple[float, float, float]) -> None:
        if self.summary is None:
            return
        summary = self.summary
        lines = [
            f"Input: {summary.input_path}",
            f"LAS version: {summary.las_version}",
            f"Point format ID: {summary.point_format_id}",
            f"Total points: {summary.point_count_total:,}",
            f"Finite XYZ points: {summary.point_count_valid_xyz:,}",
            f"Sampled display points: {summary.point_count_sampled:,}",
            f"View center: x={center_xyz[0]:.3f}, y={center_xyz[1]:.3f}, z={center_xyz[2]:.3f}",
            f"Z scale: {self.z_scale_spin.value():.2f}",
            f"CRS: {summary.crs_text}",
        ]
        if summary.crs_epsg is not None:
            lines.append(f"CRS EPSG: {summary.crs_epsg}")
        if summary.crs_name:
            lines.append(f"CRS name: {summary.crs_name}")
        if summary.unknown_class_codes:
            lines.append("Unknown class codes using fallback color: " + ", ".join(map(str, summary.unknown_class_codes)))
        self.info_text.setPlainText("\n".join(lines))

    def reset_view(self) -> None:
        if self.centered_points is None or self.centered_points.size == 0:
            self.view.setCameraPosition(distance=200.0, elevation=30.0, azimuth=-45.0)
            return

        xyz = self.centered_points
        xy_extent = np.nanmax(xyz[:, :2], axis=0) - np.nanmin(xyz[:, :2], axis=0)
        z_extent = float(np.nanmax(xyz[:, 2]) - np.nanmin(xyz[:, 2]))
        distance = float(max(30.0, np.nanmax(xy_extent), z_extent) * 1.8)
        self.view.setCameraPosition(distance=distance, elevation=35.0, azimuth=-45.0)

        grid_size = float(max(20.0, np.nanmax(xy_extent)))
        spacing = _nice_grid_spacing(grid_size / 10.0)
        self.grid.setSize(x=grid_size, y=grid_size)
        self.grid.setSpacing(x=spacing, y=spacing)
        axis_size = float(max(10.0, grid_size * 0.15))
        self.axis.setSize(x=axis_size, y=axis_size, z=max(axis_size * 0.5, z_extent * 0.25))


def _nice_grid_spacing(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 10.0
    exponent = np.floor(np.log10(value))
    base = value / (10.0 ** exponent)
    if base <= 1.5:
        nice = 1.0
    elif base <= 3.0:
        nice = 2.0
    elif base <= 7.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10.0 ** exponent))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qt LAS/LAZ classification point-cloud viewer.")
    parser.add_argument("input", nargs="?", type=Path, help="Optional LAS/LAZ file to open at startup.")
    parser.add_argument("--max-display-points", type=int, default=800_000, help="Maximum sampled points for display.")
    parser.add_argument("--read-chunk-size", type=int, default=2_000_000, help="LAS/LAZ read chunk size.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for display sampling.")
    parser.add_argument("--z-scale", type=float, default=1.0, help="Initial vertical scale for display.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Initial point size in screen pixels.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = ViewerConfig(
        max_display_points=args.max_display_points,
        random_seed=args.random_seed,
        read_chunk_size=args.read_chunk_size,
        z_scale_for_view=args.z_scale,
        point_size_px=args.point_size,
    )

    app = QtWidgets.QApplication(sys.argv)
    win = ClassificationPointCloudViewer(initial_path=args.input, cfg=cfg)
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
