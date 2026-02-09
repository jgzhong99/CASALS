"""Main window exports for the CASALS GUI."""

from __future__ import annotations

from .qt_main_window import CASALSQtMainWindow

# Keep legacy class name as a direct alias while removing Tk compatibility code.
CASALSMainWindow = CASALSQtMainWindow


__all__ = ["CASALSMainWindow", "CASALSQtMainWindow"]
