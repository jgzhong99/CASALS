"""CASALS 3D Sweep Animator and Waveform Cross-Section Viewer.

A standalone Qt application for dynamic visualization of airborne CASALS L1B LiDAR
swaths, aircraft trajectory, beam geometry, and synchronized 2D waveform cross-sections.
"""

from __future__ import annotations

from .geometry import LocalENU, bin_to_xyz, geodetic_to_ecef, ecef_to_geodetic
from .h5_model import H5Model, SweepData
from .main_window import MainWindow

__all__ = [
    "MainWindow",
    "H5Model",
    "SweepData",
    "LocalENU",
    "bin_to_xyz",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
]

