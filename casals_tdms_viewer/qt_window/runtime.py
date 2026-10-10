"""Runtime dependency imports for the Qt CASALS main window."""

from __future__ import annotations

QT_IMPORT_ERROR: Exception | None = None
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover - runtime dependency
    QtCore = None  # type: ignore[assignment]
    QtGui = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    QT_IMPORT_ERROR = exc

MATPLOTLIB_IMPORT_ERROR: Exception | None = None
try:
    from matplotlib import cm as mpl_cm
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure
except Exception as exc:  # pragma: no cover - runtime dependency
    mpl_cm = None  # type: ignore[assignment]
    FigureCanvas = None  # type: ignore[assignment]
    NavigationToolbar = None  # type: ignore[assignment]
    LinearSegmentedColormap = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]
    MATPLOTLIB_IMPORT_ERROR = exc

PYVISTA_IMPORT_ERROR: Exception | None = None
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception as exc:  # pragma: no cover - runtime dependency
    pv = None  # type: ignore[assignment]
    QtInteractor = None  # type: ignore[assignment]
    PYVISTA_IMPORT_ERROR = exc

PYQTGRAPH_IMPORT_ERROR: Exception | None = None
try:
    import pyqtgraph as pg
except Exception as exc:  # pragma: no cover - runtime dependency
    pg = None  # type: ignore[assignment]
    PYQTGRAPH_IMPORT_ERROR = exc

BaseMainWindow = QtWidgets.QMainWindow if QtWidgets is not None else object

__all__ = [
    "QT_IMPORT_ERROR",
    "MATPLOTLIB_IMPORT_ERROR",
    "PYVISTA_IMPORT_ERROR",
    "PYQTGRAPH_IMPORT_ERROR",
    "QtCore",
    "QtGui",
    "QtWidgets",
    "Figure",
    "FigureCanvas",
    "NavigationToolbar",
    "LinearSegmentedColormap",
    "mpl_cm",
    "pv",
    "QtInteractor",
    "pg",
    "BaseMainWindow",
]
