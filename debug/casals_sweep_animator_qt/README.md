# CASALS Sweep Viewer

An interactive, read-only viewer for CASALS L1B HDF5 data. VisPy renders the 3D scene, PyQtGraph renders the synchronized 2D cross-section, and PyQt5 provides the interface.

## Install and launch

From the repository root, install the GUI dependencies and start the viewer:

```powershell
python -m pip install -e ".[gui]"
python -m debug.casals_sweep_animator_qt
python -m debug.casals_sweep_animator_qt "data/raw/casals_l1b/casals_l1b_20241112T165718_001_02.h5"
```

The first command installs `vispy`, `PyQt5`, `pyqtgraph`, and the repository's GUI dependencies. A working desktop OpenGL driver is required for the 3D canvas.

## Controls

- Playback: play or pause, step forward or back, choose a playback rate and step size, or drag the timeline. The timeline index and source sweep ID are shown separately.
- 3D scene: toggle aircraft, trajectory, history and current Refh points, and beam density. Change history color and use Follow, Overview, or Free camera. The local ENU axes use equal scale in all directions: one meter has the same world-space length on X, Y, and Z.
- 2D cross-section: switch between **Waveform Curtain** (columns preserve the complete track index) and **Georeferenced Cross-Section** (horizontal axis in meters). Select raw RX, background-subtracted RX, asinh, standardized RX, or Refh-only display; choose H1 or H2 range mapping; show or hide official Refh and reconstructed argmax overlays.
- Export: save a composed PNG with a JSON description, or render a selected sweep range to MP4 with a JSON sidecar.

Standardized RX is marked unavailable when required background mean or standard deviation data is absent or invalid. It is never replaced with raw RX without an explicit message. The georeferenced view also reports when track geometry does not support a reliable meter axis.

## Data and rendering behavior

- The worker thread owns the read-only HDF5 handle. It keeps one pending latest frame request and drops results whose request ID or file generation is stale.
- Waveforms are read in chunk-aligned sweep blocks when the record-index grid proves the records are contiguous. The block LRU is limited to 128 MiB by default. Opening a file indexes metadata and decimates history; it does not load all waveform samples.
- History points use a deterministic, time-ordered sample bounded by 220,000 points. Every sweep keeps an explicit row for every track; missing records stay missing through the 2D view.
- Range-bin coordinates use the unchanged H1/H2 fractions along the ECEF endpoints. The 2D height grid interpolates the endpoint WGS84 ellipsoidal heights at those same fractions; real-data validation checks it against direct ECEF conversion with a 1 mm limit. The 3D geometry and RX-argmax positions remain in ECEF-derived coordinates. Official Refh values stay separate from RX argmax reconstruction.
- 3D positions and axes remain in local ENU meters with a 1:1:1 axis scale; the viewer applies no vertical exaggeration.
- Vertical display scaling applies to 3D only. Refh and bin heights remain in source physical units in the 2D panel.
- The aircraft is a schematic marker with heading derived from the horizontal track. H1/H2 closure against Refh is an internal visualization diagnostic, not independent accuracy validation.
- Video frames are captured and composed on the GUI thread. A two-frame queue feeds an encoder thread that only handles RGB arrays. The completed MP4 is reopened and decoded before the video and metadata sidecar are moved into place.

## Validation

Run the unit and GUI-offscreen tests, then run the real-data check from a visible Windows desktop session:

```powershell
python -m pytest -q
python -m compileall -q debug/casals_sweep_animator_qt tests
git diff --check
python -m debug.casals_sweep_animator_qt.validate_real_data
```

The real-data check uses the local November 12, 2024 CASALS file by default. It checks the live OpenGL renderer, captures early/middle/late Follow views plus Overview and Curtain examples, measures a 55-frame visible GUI playback sequence without a screenshot on each frame, checks H1/H2 height-grid parity against direct ECEF conversion, and exports a 50-frame MP4 through the viewer's production GUI path. PNGs, sidecars, the MP4, `performance.json`, and `validation_report.md` are written under ignored `outputs/casals_sweep_animator_qt/final/`.

Performance depends on the input file, storage cache, display scale, and graphics driver. The validation report records the actual result; the viewer makes no fixed FPS claim.
