from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

_QT_DLL_DIR_HANDLES: list[object] = []


def _safe_existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    return candidate if candidate.exists() else None


def _bootstrap_qt_environment() -> None:
    pyside6 = importlib.import_module("PySide6")
    if not getattr(pyside6, "__file__", None):
        raise RuntimeError("PySide6 is installed but does not expose a file path.")

    pyside_root = Path(pyside6.__file__).resolve().parent
    plugins_dir = pyside_root / "plugins"
    platforms_dir = plugins_dir / "platforms"

    current_path = os.environ.get("PATH", "")
    path_parts = [part for part in current_path.split(os.pathsep) if part]
    root_str = str(pyside_root)
    if root_str not in path_parts:
        os.environ["PATH"] = root_str + (os.pathsep + current_path if current_path else "")

    if hasattr(os, "add_dll_directory"):
        _QT_DLL_DIR_HANDLES.append(os.add_dll_directory(root_str))

    if _safe_existing_path(os.environ.get("QT_PLUGIN_PATH")) is None and plugins_dir.exists():
        os.environ["QT_PLUGIN_PATH"] = str(plugins_dir)
    if _safe_existing_path(os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")) is None and platforms_dir.exists():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)


def main() -> None:
    project_dir = Path(__file__).resolve().parent / "CASALS_L1B"
    sys.path.insert(0, str(project_dir))
    os.chdir(project_dir)
    _bootstrap_qt_environment()
    from casals_refh_filter_gui import main as app_main

    app_main()


if __name__ == "__main__":
    main()
