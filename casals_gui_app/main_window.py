"""Compatibility entry for the CASALS GUI main window.

The primary GUI is now Qt + PyVistaQt (`CASALSQtMainWindow`).
This module keeps the historical import path working:

    from casals_gui_app.main_window import CASALSMainWindow
"""

from __future__ import annotations

import sys

from .qt_main_window import CASALSQtMainWindow
from .qt_window.runtime import QtWidgets


class CASALSMainWindow(CASALSQtMainWindow):
    """Backward-compatible entry that bridges legacy Tk-style startup.

    Legacy callers may still do:
        root = tk.Tk()
        CASALSMainWindow(root)
        root.mainloop()

    When a Tk root is provided, this compatibility layer patches that root so
    `root.mainloop()` runs the Qt event loop and `root.quit()/destroy()` map
    to Qt shutdown.
    """

    def __init__(self, *_legacy_args, **_legacy_kwargs) -> None:
        legacy_root = _legacy_kwargs.pop("root", None)
        if legacy_root is None and _legacy_args:
            legacy_root = _legacy_args[0]

        self._compat_qt_loop_started = False
        self._compat_qt_app = self._ensure_qapplication()
        super().__init__()
        if legacy_root is not None:
            self._bridge_legacy_tk_root(legacy_root)
            self.show()

    @staticmethod
    def _ensure_qapplication():
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        return app

    def _bridge_legacy_tk_root(self, legacy_root) -> None:
        withdraw = getattr(legacy_root, "withdraw", None)
        if callable(withdraw):
            try:
                withdraw()
            except Exception:
                pass

        def _qt_mainloop(*_args, **_kwargs):
            if self._compat_qt_loop_started:
                return 0
            self._compat_qt_loop_started = True
            self.show()
            return int(self._compat_qt_app.exec_())

        def _qt_quit(*_args, **_kwargs):
            self._compat_qt_app.quit()

        def _qt_destroy(*_args, **_kwargs):
            try:
                self.close()
            finally:
                self._compat_qt_app.quit()

        if callable(getattr(legacy_root, "mainloop", None)):
            try:
                legacy_root.mainloop = _qt_mainloop
            except Exception:
                pass
        if callable(getattr(legacy_root, "quit", None)):
            try:
                legacy_root.quit = _qt_quit
            except Exception:
                pass
        if callable(getattr(legacy_root, "destroy", None)):
            try:
                legacy_root.destroy = _qt_destroy
            except Exception:
                pass


__all__ = ["CASALSMainWindow", "CASALSQtMainWindow"]
