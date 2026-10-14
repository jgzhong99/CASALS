"""Real-data validation and performance benchmark on local CASALS L1B files.

Verifies:
- Early, mid, and late sweep reads and geometry on real 14GB H5 files.
- Refh and argmax closure for H1 and H2.
- Meter-based cross-track geometry.
- Benchmark timings: initialization time, frame prep time, memory usage.
- Generates composite screenshot PNGs and a test MP4 export with metadata sidecar.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import psutil

# Ensure casals root is in sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5 import QtCore, QtGui, QtWidgets

from debug.casals_sweep_animator_qt.h5_model import H5Model, SweepData
from debug.casals_sweep_animator_qt.cross_section import compute_waveform_cross_section
from debug.casals_sweep_animator_qt.geometry import LocalENU
from debug.casals_sweep_animator_qt.scene3d import Scene3DWidget
from debug.casals_sweep_animator_qt.panel2d import Panel2DWidget
from debug.casals_sweep_animator_qt.video_export import (
    capture_widget_pixmap,
    composite_screenshot,
    VideoExportWorker,
)


def get_process_memory_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024.0 * 1024.0)


def run_validation(h5_path: Path) -> dict:
    print("=" * 80)
    print(f"Validating real H5: {h5_path.name}")
    print("=" * 80)

    mem_start = get_process_memory_mb()
    t0 = time.perf_counter()

    # 1. Initialization
    model = H5Model(h5_path, history_budget=220_000)
    t_init = time.perf_counter() - t0
    mem_loaded = get_process_memory_mb()

    print(f"Initialization time: {t_init:.2f} s")
    print(f"Memory RSS after indexing and decimation: {mem_loaded:.1f} MB (Delta: {mem_loaded - mem_start:.1f} MB)")
    print(f"Valid sweeps: {model.n_valid_sweeps:,} (IDs {model.valid_sweep_ids[0]}..{model.valid_sweep_ids[-1]})")
    print(f"Tracks: {model.n_tracks}, RX bins: {model.n_rx_bins}")
    print(f"Decimated history points: {model.total_decimated_points:,} (stride: {model.decimation_stride})")
    print(f"Origin: lon={model.origin_lon:.6f}, lat={model.origin_lat:.6f}, h={model.origin_height:.2f} m")

    results = {
        "file": h5_path.name,
        "init_time_s": t_init,
        "mem_loaded_mb": mem_loaded,
        "n_sweeps": model.n_valid_sweeps,
        "n_tracks": model.n_tracks,
        "n_rx_bins": model.n_rx_bins,
        "decimated_points": model.total_decimated_points,
        "sweeps_evaluated": [],
    }

    # 2. Check early, mid, late sweeps
    test_sweep_ids = [
        int(model.valid_sweep_ids[100]),
        int(model.valid_sweep_ids[model.n_valid_sweeps // 2]),
        int(model.valid_sweep_ids[-100]),
    ]

    with h5py.File(h5_path, "r") as h5:
        for sid in test_sweep_ids:
            s_idx = model.sweep_id_to_idx[sid]

            t_read0 = time.perf_counter()
            sw = model.read_sweep_data(h5, s_idx)
            t_read = time.perf_counter() - t_read0

            t_cs0 = time.perf_counter()
            grid_h2 = compute_waveform_cross_section(
                sw, mode="minus_bg", x_axis_mode="meters", mapping_convention="H2"
            )
            t_cs = time.perf_counter() - t_cs0

            grid_h1 = compute_waveform_cross_section(
                sw, mode="minus_bg", x_axis_mode="meters", mapping_convention="H1"
            )

            valid_refh = int(np.sum(sw.valid_pulse_mask))
            h_range = [
                float(np.nanmin(sw.refh_height[sw.valid_pulse_mask])),
                float(np.nanmax(sw.refh_height[sw.valid_pulse_mask])),
            ] if valid_refh else [np.nan, np.nan]
            med_snr = float(np.nanmedian(sw.refh_snr)) if sw.refh_snr is not None else np.nan

            print(f"\n--- Sweep {sid} (Index {s_idx}) ---")
            print(f"  Valid Refh: {valid_refh}/{sw.n_tracks} | Median SNR: {med_snr:.1f}")
            print(f"  Refh Height: [{h_range[0]:.2f}, {h_range[1]:.2f}] m (Span: {h_range[1]-h_range[0]:.2f} m)")
            print(f"  Aircraft ENU: [{sw.instrument_enu[0]:.1f}, {sw.instrument_enu[1]:.1f}, {sw.instrument_enu[2]:.1f}] m")
            print(f"  Heading: {sw.heading_deg:.1f}°")
            print(f"  Cross-track meters status: {sw.can_use_meters} ({sw.cross_track_reason})")
            print(f"  Closure H2: 3D {grid_h2.closure_median_3d_m*1000:.2f} mm | Vert {grid_h2.closure_median_vertical_m*1000:.2f} mm")
            print(f"  Closure H1: 3D {grid_h1.closure_median_3d_m*1000:.2f} mm | Vert {grid_h1.closure_median_vertical_m*1000:.2f} mm")
            print(f"  Timings: Waveform read {t_read*1000:.2f} ms, Grid generation {t_cs*1000:.2f} ms (Total prep: {(t_read+t_cs)*1000:.2f} ms)")

            results["sweeps_evaluated"].append({
                "sweep_id": sid,
                "sweep_idx": s_idx,
                "valid_refh": valid_refh,
                "median_snr": med_snr,
                "refh_height_min": h_range[0],
                "refh_height_max": h_range[1],
                "closure_h2_3d_mm": grid_h2.closure_median_3d_m * 1000 if grid_h2.closure_median_3d_m is not None else None,
                "closure_h1_3d_mm": grid_h1.closure_median_3d_m * 1000 if grid_h1.closure_median_3d_m is not None else None,
                "prep_time_ms": (t_read + t_cs) * 1000,
            })

    # 3. Continuous 150-sweep sequence benchmark
    bench_start = 5000
    bench_count = 150
    print(f"\nBenchmarking continuous {bench_count} sweeps (sweeps {bench_start}..{bench_start+bench_count})...")
    times = []
    with h5py.File(h5_path, "r") as h5:
        for sid in range(bench_start, bench_start + bench_count):
            if sid not in model.sweep_id_to_idx:
                continue
            s_idx = model.sweep_id_to_idx[sid]
            t_s = time.perf_counter()
            sw = model.read_sweep_data(h5, s_idx)
            _ = compute_waveform_cross_section(sw, mode="minus_bg", x_axis_mode="meters", mapping_convention="H2")
            times.append(time.perf_counter() - t_s)

    avg_prep_ms = np.mean(times) * 1000
    p95_prep_ms = np.percentile(times, 95) * 1000
    max_fps_calc = 1.0 / np.mean(times)
    print(f"Continuous prep time: Mean {avg_prep_ms:.2f} ms, P95 {p95_prep_ms:.2f} ms")
    print(f"Max throughput capability: ~{max_fps_calc:.1f} FPS (without render bottleneck)")

    results["bench_continuous_mean_ms"] = avg_prep_ms
    results["bench_continuous_p95_ms"] = p95_prep_ms
    results["bench_max_throughput_fps"] = max_fps_calc

    return results, model


def render_and_save_visualizations(model: H5Model, sweeps_to_capture: list[int]):
    """Launch offscreen Qt widgets, render representative sweeps, and save screenshots + MP4 clip."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    out_screenshots = ROOT / "outputs" / "screenshots"
    out_screenshots.mkdir(parents=True, exist_ok=True)
    out_videos = ROOT / "outputs" / "videos"
    out_videos.mkdir(parents=True, exist_ok=True)

    # Instantiate widgets
    scene3d = Scene3DWidget()
    scene3d.resize(1200, 600)
    panel2d = Panel2DWidget()
    panel2d.resize(1200, 350)

    scene3d.show()
    panel2d.show()
    app.processEvents()

    scene3d.set_vertical_scale(0.05)
    scene3d.set_color_ranges(model.height_vmin, model.height_vmax)
    panel2d.set_fixed_range(0.0, 500.0)

    screenshots_saved = []

    with h5py.File(model.path, "r") as h5:
        for sid in sweeps_to_capture:
            if sid not in model.sweep_id_to_idx:
                continue
            s_idx = model.sweep_id_to_idx[sid]
            sw = model.read_sweep_data(h5, s_idx)
            hist_enu, hist_h, _, _ = model.get_history_slice(s_idx)

            grid = compute_waveform_cross_section(
                sw, mode="minus_bg", x_axis_mode="meters", mapping_convention="H2"
            )

            # Center camera around aircraft
            scene3d.reset_view(center=sw.instrument_enu, distance=1800.0)

            scene3d.update_frame(
                sweep=sw,
                history_enu=hist_enu,
                history_values=hist_h,
                all_trajectory_enu=model.sweep_instrument_enu,
            )
            panel2d.update_cross_section(grid)
            app.processEvents()
            time.sleep(0.05)
            app.processEvents()

            p3d = capture_widget_pixmap(scene3d)
            p2d = capture_widget_pixmap(panel2d)

            png_path = out_screenshots / f"{model.path.stem}_sweep_{sid}.png"
            saved = composite_screenshot(
                scene3d_pixmap=p3d,
                panel2d_pixmap=p2d,
                sweep_data=sw,
                model=model,
                mapping_convention="H2",
                output_path=png_path,
            )
            print(f"Saved representative screenshot: {saved.name}")
            screenshots_saved.append(saved)

    # Export a 50-frame MP4 animation clip (e.g. sweep 5000 to 5050)
    print("\nExporting representative 50-frame MP4 video clip...")
    mp4_path = out_videos / f"{model.path.stem}_sweeps_5000_5050.mp4"

    def render_frame_for_video(sid: int) -> QtGui.QImage:
        s_idx = model.sweep_id_to_idx[sid]
        with h5py.File(model.path, "r") as h5:
            sw = model.read_sweep_data(h5, s_idx)
        hist_enu, hist_h, _, _ = model.get_history_slice(s_idx)
        grid = compute_waveform_cross_section(
            sw, mode="minus_bg", x_axis_mode="meters", mapping_convention="H2"
        )
        scene3d.update_frame(
            sweep=sw,
            history_enu=hist_enu,
            history_values=hist_h,
            all_trajectory_enu=model.sweep_instrument_enu,
        )
        panel2d.update_cross_section(grid)
        app.processEvents()

        p3d = capture_widget_pixmap(scene3d)
        p2d = capture_widget_pixmap(panel2d)
        tmp = ROOT / ".compile_tmp" / "temp_video_frame.png"
        comp = composite_screenshot(
            scene3d_pixmap=p3d,
            panel2d_pixmap=p2d,
            sweep_data=sw,
            model=model,
            mapping_convention="H2",
            output_path=tmp,
        )
        return QtGui.QImage(str(comp))

    worker = VideoExportWorker(
        model=model,
        start_sweep=5000,
        end_sweep=5050,
        step=1,
        fps=15.0,
        output_path=mp4_path,
        render_callback=render_frame_for_video,
    )
    worker.run()
    print(f"Saved MP4 video animation: {mp4_path.name}")
    print(f"Saved sidecar JSON: {mp4_path.with_suffix('.json').name}")

    scene3d.close()
    panel2d.close()

    return screenshots_saved, mp4_path


if __name__ == "__main__":
    h5_nov12 = ROOT / "data" / "raw" / "casals_l1b" / "casals_l1b_20241112T165718_001_02.h5"
    if not h5_nov12.exists():
        print(f"Error: {h5_nov12} does not exist.")
        sys.exit(1)

    results, model = run_validation(h5_nov12)
    screenshots, mp4 = render_and_save_visualizations(model, [200, 5000, 12000])

    print("\nValidation and benchmark completed successfully!")

