"""Entry point for CASALS Sweep Animator.

Run with:
    python -m debug.casals_sweep_animator_qt [path/to/casals_l1b.h5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from PyQt5 import QtCore, QtWidgets

from .main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casals_sweep_animator_qt",
        description="CASALS Sweep Animator and Waveform Cross-Section Viewer",
    )
    parser.add_argument(
        "h5",
        nargs="?",
        type=Path,
        default=None,
        help="Optional CASALS L1B HDF5 file to open upon launch.",
    )
    args = parser.parse_args(argv)

    # Enable high-DPI scaling and rounding policy
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(QtCore.Qt, "HighDpiScaleFactorRoundingPolicy"):
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("CASALS Sweep Animator")

    initial_h5 = str(args.h5.resolve()) if args.h5 is not None else None
    window = MainWindow(initial_h5=initial_h5)
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
