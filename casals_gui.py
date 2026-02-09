"""CASALS GUI launcher (Qt + PyVistaQt)."""

import sys


def _write_startup_error(message: str) -> None:
    """Emit a standardized startup error to stderr."""
    sys.stderr.write(f"[CASALS GUI startup error]\n{message}\n")


def _configure_high_dpi(QtCore, QtWidgets) -> None:
    """Configure high DPI settings for the application."""
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


def _run_event_loop(app) -> int:
    """Run Qt event loop across supported Qt bindings."""
    app_exec = getattr(app, "exec", None) or getattr(app, "exec_", None)
    if app_exec is None:
        raise RuntimeError("QApplication is missing an event loop runner (exec/exec_).")
    return int(app_exec())


def main() -> int:
    """Main entry point for the CASALS GUI application."""
    try:
        from PyQt5 import QtCore, QtWidgets
    except ImportError:
        _write_startup_error("PyQt5 is required for the GUI.\nInstall with:\n  pip install PyQt5")
        return 1

    try:
        _configure_high_dpi(QtCore, QtWidgets)
        from casals_gui_app.qt_main_window import CASALSQtMainWindow

        app = QtWidgets.QApplication(sys.argv)
        window = CASALSQtMainWindow()
        window.show()
        return _run_event_loop(app)
    except Exception as exc:
        _write_startup_error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
