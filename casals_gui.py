"""CASALS GUI launcher (Qt + PyVistaQt)."""

import sys


def _configure_high_dpi(QtCore, QtWidgets) -> None:
    qt = QtCore.Qt
    app_cls = QtWidgets.QApplication

    for attr_name in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        attr = getattr(qt, attr_name, None)
        if attr is not None:
            app_cls.setAttribute(attr, True)

    policy_enum = getattr(qt, "HighDpiScaleFactorRoundingPolicy", None)
    set_policy = getattr(app_cls, "setHighDpiScaleFactorRoundingPolicy", None)
    if policy_enum is not None and set_policy is not None:
        set_policy(policy_enum.PassThrough)


def main() -> int:
    try:
        from PyQt5 import QtCore, QtWidgets
    except ImportError:
        sys.stderr.write(
            "[CASALS GUI startup error]\n"
            "PyQt5 is required for the GUI.\n"
            "Install with:\n"
            "  pip install PyQt5\n"
        )
        return 1

    try:
        from casals_gui_app.qt_main_window import CASALSQtMainWindow
        _configure_high_dpi(QtCore, QtWidgets)
        app = QtWidgets.QApplication(sys.argv)
        window = CASALSQtMainWindow()
        window.show()
        return int(app.exec_())
    except Exception as exc:
        sys.stderr.write(f"[CASALS GUI startup error]\n{exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
