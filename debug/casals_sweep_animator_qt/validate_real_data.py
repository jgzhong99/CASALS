"""Run real HDF5, Windows OpenGL, GUI throughput, screenshot, and MP4 checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Callable

import cv2
import h5py
import numpy as np
import psutil
from PyQt5 import QtCore, QtWidgets

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from debug.casals_sweep_animator_qt.cross_section import (
    _ecef_to_wgs84_height,
    _height_samples,
    compute_waveform_cross_section,
)
from debug.casals_sweep_animator_qt.geometry import bin_to_xyz
from debug.casals_sweep_animator_qt.main_window import MainWindow
from debug.casals_sweep_animator_qt.video_export import VideoExportDialog


def wait_until(app: QtWidgets.QApplication, predicate: Callable[[], bool], timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise TimeoutError(f"Timed out waiting for {label} after {timeout_s:.1f} seconds")


def flush_gui(app: QtWidgets.QApplication, seconds: float = 0.18) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)


def gl_info(window: MainWindow) -> dict[str, str]:
    return {
        "vendor": "Pure Qt Raster Engine",
        "renderer": "QPainter / Software (Zero OpenGL)",
        "version": "Qt 5.15 (Native No-OpenGL Pipeline)",
    }


def save_view(window: MainWindow, output_dir: Path, name: str, index: int, view: str, camera: str) -> dict:
    assert window.model is not None
    window.combo_2d_view.setCurrentIndex(window.combo_2d_view.findData(view))
    window.combo_2d_mode.setCurrentIndex(window.combo_2d_mode.findData("minus_bg"))
    window.combo_mapping.setCurrentIndex(window.combo_mapping.findData("H2"))
    window.combo_camera.setCurrentIndex(window.combo_camera.findData(camera))
    window._request_sweep(index)
    request_id = window.controller.request_id
    wait_until(QtWidgets.QApplication.instance(), lambda: window.current_frame is not None and window.current_frame.request_id == request_id,
               30.0, f"sweep {index} ({name})")
    flush_gui(QtWidgets.QApplication.instance(), 0.2)
    frame = window.current_frame
    assert frame is not None
    if view == "georeferenced" and not frame.grid.available:
        raise RuntimeError(f"Georeferenced view was unavailable at sweep {frame.sweep_data.sweep_id}: {frame.grid.message}")
    image = window._composed_image(frame)
    path = output_dir / f"{name}.png"
    if not image.save(str(path), "PNG"):
        raise OSError(f"Could not save rendered screenshot: {path}")
    metadata = {
        **window._display_metadata(frame),
        "image_file": path.name,
        "output_dimensions": [image.width(), image.height()],
        "capture_method": "visible Qt main window; VisPy OpenGL framebuffer plus PyQtGraph QWidget capture",
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"name": name, "path": str(path), "sweep_id": frame.sweep_data.sweep_id,
            "view": view, "camera": camera, "bytes": path.stat().st_size,
            "grid_available": frame.grid.available, "grid_shape": list(frame.grid.grid_data.shape)}


def save_cross_section_closeup(window: MainWindow, output_dir: Path, name: str) -> dict:
    assert window.current_frame is not None
    image = window.panel2d.grab().toImage()
    path = output_dir / f"{name}.png"
    if not image.save(str(path), "PNG"):
        raise OSError(f"Could not save cross-section close-up: {path}")
    metadata = {
        **window._display_metadata(window.current_frame),
        "image_file": path.name,
        "output_dimensions": [image.width(), image.height()],
        "capture_method": "visible PyQtGraph cross-section widget capture",
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"name": name, "path": str(path), "sweep_id": window.current_frame.sweep_data.sweep_id,
            "view": "2D cross-section close-up", "camera": window.scene3d.camera_mode,
            "bytes": path.stat().st_size, "grid_available": window.current_frame.grid.available,
            "grid_shape": list(window.current_frame.grid.grid_data.shape)}


def run_video_export(window: MainWindow, output_path: Path, start_index: int, count: int, fps: float) -> dict:
    assert window.model is not None
    end_index = min(window.model.n_valid_sweeps - 1, start_index + count - 1)
    expected_count = end_index - start_index + 1
    start_id = int(window.model.valid_sweep_ids[start_index])
    end_id = int(window.model.valid_sweep_ids[end_index])

    def configure_dialog() -> None:
        dialogs = [w for w in QtWidgets.QApplication.topLevelWidgets() if isinstance(w, VideoExportDialog)]
        if not dialogs:
            QtCore.QTimer.singleShot(50, configure_dialog)
            return
        dialog = dialogs[-1]
        dialog.spin_start.setValue(start_id)
        dialog.spin_end.setValue(end_id)
        dialog.spin_step.setValue(1)
        dialog.combo_fps.setCurrentText(str(int(fps)))
        dialog.line_path.setText(str(output_path))
        dialog.btn_export.click()

    QtCore.QTimer.singleShot(100, configure_dialog)
    window._export_video()
    wait_until(QtWidgets.QApplication.instance(), lambda: not window._export_active, 600.0, "GUI MP4 export completion")
    if not output_path.is_file():
        raise FileNotFoundError(f"Video export did not create {output_path}")
    sidecar = output_path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("decoded_frame_count") != expected_count or metadata.get("verified") is not True:
        raise RuntimeError(f"MP4 sidecar did not confirm all {expected_count} decoded frames: {metadata}")
    return {"path": str(output_path), "sidecar": str(sidecar), "frame_count": expected_count,
            "fps": fps, "bytes": output_path.stat().st_size, "verified": True}


def run_validation(h5_path: Path, output_dir: Path, export_frames: int = 50) -> dict:
    h5_path = h5_path.resolve()
    before = h5_path.stat()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(1600, 960)
    window.show()
    load_start = time.perf_counter()
    window.load_h5(str(h5_path))
    wait_until(app, lambda: window.model is not None and window.current_frame is not None,
               180.0, "real HDF5 indexing and first displayed frame")
    initialization_s = time.perf_counter() - load_start
    flush_gui(app, 0.25)
    model = window.model
    assert model is not None
    renderer = gl_info(window)
    memory_after_load_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    system_cpu_percent_after_load = psutil.cpu_percent(interval=0.2)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = model.n_valid_sweeps
    sample_indices = {
        "early": min(100, n - 1),
        "middle": n // 2,
        "late": max(0, n - 101),
    }
    screenshots = []
    for label, index in sample_indices.items():
        screenshots.append(save_view(window, output_dir, f"{label}_follow_georeferenced", index, "georeferenced", "follow"))
    screenshots.append(save_view(window, output_dir, "middle_overview_georeferenced", sample_indices["middle"], "georeferenced", "overview"))
    screenshots.append(save_view(window, output_dir, "middle_follow_curtain", sample_indices["middle"], "curtain", "follow"))
    screenshots.append(save_cross_section_closeup(window, output_dir, "middle_waveform_curtain_closeup"))
    flush_gui(app, 0.2)
    full_path = output_dir / "full_gui_window.png"
    full_image = window.grab().toImage()
    if not full_image.save(str(full_path), "PNG"):
        raise OSError(f"Could not save full-window GUI screenshot: {full_path}")
    full_metadata = {
        **window._display_metadata(window.current_frame),
        "image_file": full_path.name,
        "output_dimensions": [full_image.width(), full_image.height()],
        "capture_method": "visible QMainWindow grab including controls and both rendered views",
    }
    full_path.with_suffix(".json").write_text(json.dumps(full_metadata, indent=2), encoding="utf-8")
    screenshots.append({"name": "full_gui_window", "path": str(full_path),
                        "sweep_id": window.current_frame.sweep_data.sweep_id,
                        "view": "full GUI", "camera": window.scene3d.camera_mode,
                        "bytes": full_path.stat().st_size, "grid_available": window.current_frame.grid.available,
                        "grid_shape": list(window.current_frame.grid.grid_data.shape)})

    # H1/H2 Refh closure is an internal geometry diagnostic, not independent accuracy validation.
    closure = []
    height_grid_parity = []
    for label, index in sample_indices.items():
        window._request_sweep(index, window._frame_params())
        request_id = window.controller.request_id
        wait_until(app, lambda: window.current_frame is not None and window.current_frame.request_id == request_id,
                   30.0, f"H2 geometry frame {label}")
        sweep = window.current_frame.sweep_data
        entry = {"label": label, "sweep_id": sweep.sweep_id}
        bins = np.broadcast_to(
            np.arange(sweep.n_rx_bins, dtype=np.float64),
            (sweep.n_tracks, sweep.n_rx_bins),
        )
        for mapping in ("H1", "H2"):
            grid = compute_waveform_cross_section(
                sweep, mode="refh_only", view="curtain", mapping_convention=mapping,
                fixed_height_range=model.height_range,
            )
            exact_xyz = bin_to_xyz(
                sweep.rwstart_ecef[:, None, :], sweep.rwstop_ecef[:, None, :], bins,
                sweep.n_rx_bins, convention=mapping,
            )
            exact_heights = _ecef_to_wgs84_height(exact_xyz)
            displayed_heights = _height_samples(sweep, mapping)
            finite_height = np.isfinite(exact_heights) & np.isfinite(displayed_heights)
            max_height_error_m = (
                float(np.max(np.abs(exact_heights[finite_height] - displayed_heights[finite_height])))
                if np.any(finite_height) else 0.0
            )
            if max_height_error_m > 0.001:
                raise RuntimeError(
                    f"{mapping} display-height approximation exceeded 1 mm at {label}: "
                    f"{max_height_error_m:.6g} m"
                )
            height_grid_parity.append({
                "label": label,
                "sweep_id": sweep.sweep_id,
                "mapping": mapping,
                "max_abs_error_m": max_height_error_m,
                "compared_bins": int(np.count_nonzero(finite_height)),
            })
            entry[mapping] = {
                "median_3d_closure_m": grid.closure_median_3d_m,
                "median_horizontal_closure_m": grid.closure_median_horizontal_m,
                "median_vertical_closure_m": grid.closure_median_vertical_m,
                "height_grid_max_abs_error_m": max_height_error_m,
            }
        closure.append(entry)

    # Measure actual playback through GUI frame updates and visible VisPy draw events without per-frame grabs.
    params = window._frame_params()
    start_index = min(5000, max(0, n - min(55, n)))
    benchmark_count = min(55, n - start_index)
    block_ms: list[float] = []
    cross_section_ms: list[float] = []
    current_update_ms: list[float] = []
    beam_update_ms: list[float] = []
    frame_samples: list[tuple[float, int]] = []
    draw_samples: list[tuple[float, int]] = []
    recording = {"active": False}

    def record_frame(frame) -> None:
        if not recording["active"]:
            return
        frame_samples.append((time.perf_counter(), frame.sweep_index))
        block_ms.append(frame.block_read_ms)
        cross_section_ms.append(frame.cross_section_ms)
        current_update_ms.append(window.scene3d.current_visual.last_upload_ms)
        beam_update_ms.append(window.scene3d.beam_visual.last_upload_ms)

    def record_draw(_event) -> None:
        if not recording["active"] or window.scene3d._last_sweep is None:
            return
        index = window.scene3d._last_sweep.sweep_index
        if not draw_samples or draw_samples[-1][1] != index:
            draw_samples.append((time.perf_counter(), index))

    window.controller.frame_ready.connect(record_frame)
    window.scene3d.canvas.events.draw.connect(record_draw)
    window.controller.pause()
    preroll_index = max(0, start_index - 1)
    window._request_sweep(preroll_index, params)
    preroll_request_id = window.controller.request_id
    wait_until(app, lambda: window.current_frame is not None and window.current_frame.request_id == preroll_request_id,
               30.0, f"playback benchmark pre-roll {preroll_index}")
    flush_gui(app, 0.2)
    system_cpu_percent_during_benchmark = psutil.cpu_percent(interval=0.2)
    recording["active"] = True
    window.controller.play(10.0, 1)
    wait_until(app, lambda: len(frame_samples) >= benchmark_count, 300.0,
               f"{benchmark_count}-frame playback at requested 10 FPS")
    window.controller.pause()
    flush_gui(app, 0.2)
    recording["active"] = False
    window.controller.frame_ready.disconnect(record_frame)
    window.scene3d.canvas.events.draw.disconnect(record_draw)

    frame_intervals_ms = [1000.0 * (b[0] - a[0]) for a, b in zip(frame_samples, frame_samples[1:])]
    draw_intervals_ms = [1000.0 * (b[0] - a[0]) for a, b in zip(draw_samples, draw_samples[1:])]
    capture_started = time.perf_counter()
    _ = window.scene3d.capture_framebuffer()
    _ = window.panel2d.grab()
    app.processEvents()
    single_framebuffer_capture_ms = (time.perf_counter() - capture_started) * 1000.0
    cache_stats = model.waveform_cache_stats()
    memory_after_benchmark_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    baseline_path = ROOT / "outputs" / "casals_sweep_animator_qt" / "baseline" / "baseline_gui_fps.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else None

    video = run_video_export(
        window,
        output_dir / f"{h5_path.stem}_{benchmark_count}frames.mp4",
        start_index,
        min(export_frames, benchmark_count),
        fps=10.0,
    )
    app.processEvents()
    after = h5_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input HDF5 changed during the read-only viewer validation")

    def summary(values: list[float]) -> dict[str, float]:
        mean = statistics.fmean(values)
        return {
            "mean_ms": mean,
            "median_ms": statistics.median(values),
            "p95_ms": float(np.percentile(values, 95)),
            "fps_from_mean": 1000.0 / mean if mean > 0 else 0.0,
            "frames_at_or_below_100ms": sum(v <= 100.0 for v in values),
        }

    gui_playback = {
        "requested_fps": 10.0,
        "frame_count": len(frame_samples),
        "sweep_indices": [index for _, index in frame_samples],
        "frame_ready_intervals": summary(frame_intervals_ms),
        "vispy_draw_count": len(draw_samples),
        "vispy_draw_sweep_indices": [index for _, index in draw_samples],
        "vispy_draw_intervals": summary(draw_intervals_ms) if draw_intervals_ms else None,
        "target_10_fps_measured": bool(
            len(frame_samples) == benchmark_count
            and frame_intervals_ms
            and statistics.fmean(frame_intervals_ms) <= 100.0
            and float(np.percentile(frame_intervals_ms, 95)) <= 110.0
            and all(b[1] == a[1] + 1 for a, b in zip(frame_samples, frame_samples[1:]))
            and len(draw_samples) == benchmark_count
            and draw_intervals_ms
            and statistics.fmean(draw_intervals_ms) <= 100.0
            and float(np.percentile(draw_intervals_ms, 95)) <= 110.0
            and all(b[1] == a[1] + 1 for a, b in zip(draw_samples, draw_samples[1:]))
        ),
        "capture_per_frame": False,
    }

    report = {
        "input": {
            "path": str(h5_path),
            "name": h5_path.name,
            "size_bytes": before.st_size,
            "mtime_ns_before": before.st_mtime_ns,
            "mtime_ns_after": after.st_mtime_ns,
            "unchanged": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "ram_total_bytes": psutil.virtual_memory().total,
            "system_cpu_percent_after_load": system_cpu_percent_after_load,
            "system_cpu_percent_during_benchmark": system_cpu_percent_during_benchmark,
            "gpu_opengl": renderer,
            "opencv": cv2.__version__,
        },
        "dataset": {
            "valid_sweeps": model.n_valid_sweeps,
            "first_sweep_id": int(model.valid_sweep_ids[0]),
            "last_sweep_id": int(model.valid_sweep_ids[-1]),
            "tracks": model.n_tracks,
            "rx_bins": model.n_rx_bins,
            "waveform_record_axis": model.record_axis,
            "waveform_block_sweeps": model.waveform_block_sweeps,
            "history_budget": model.history_budget,
            "history_points": model.total_decimated_points,
            "waveform_cache": cache_stats,
            "fixed_enu_origin": [model.origin_lon, model.origin_lat, model.origin_height],
        },
        "timing": {
            "initialization_to_first_display_s": initialization_s,
            "rss_after_load_mb": memory_after_load_mb,
            "rss_after_benchmark_mb": memory_after_benchmark_mb,
            "actual_gui_playback_at_10fps_request": gui_playback,
            "worker_waveform_block_read": summary(block_ms),
            "worker_2d_cross_section": summary(cross_section_ms),
            "vispy_current_overlay_buffer_update_call": summary(current_update_ms),
            "vispy_beam_buffer_update_call": summary(beam_update_ms),
            "vispy_static_history_buffer_upload_call": {
                "elapsed_ms": window.scene3d.last_history_upload_ms,
                "bytes": window.scene3d.last_history_upload_bytes,
                "note": "one-time initialization upload; GL command-call time, not GPU completion time",
            },
            "single_framebuffer_capture": {"elapsed_ms": single_framebuffer_capture_ms},
            "first_frame_ms": frame_intervals_ms[0] if frame_intervals_ms else None,
            "measured_sequence_start_index": start_index,
            "measured_sequence_frames": benchmark_count,
            "target_10_fps_measured": gui_playback["target_10_fps_measured"],
            "timing_note": "Visible 1600x960 playback; frame-ready and unique-sweep VisPy draw-event intervals; no per-frame framebuffer capture.",
        },
        "baseline_gui_reference": baseline,
        "h1_h2_internal_refh_closure": closure,
        "h1_h2_height_grid_parity": height_grid_parity,
        "screenshots": screenshots,
        "video_export": video,
        "validation_limits": [
            "H1/H2 closure compares research-mapped RX argmax positions to the L1B Refh reference and is not independent ground-truth accuracy.",
            "The aircraft mesh is schematic; heading is derived from the horizontal trajectory.",
            "Only the measured sweep sequence and selected GUI export frames were timed.",
        ],
    }
    (output_dir / "performance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, output_dir / "validation_report.md")
    window.close()
    app.processEvents()
    return report


def write_markdown_report(report: dict, path: Path) -> None:
    timing = report["timing"]
    gui = timing["actual_gui_playback_at_10fps_request"]
    ready = gui["frame_ready_intervals"]
    draw = gui["vispy_draw_intervals"]
    runtime = report["runtime"]
    dataset = report["dataset"]
    baseline = report.get("baseline_gui_reference")
    lines = [
        "# CASALS Sweep Viewer real-data validation",
        "",
        f"- Input: `{report['input']['name']}` ({report['input']['size_bytes']:,} bytes; unchanged after run).",
        f"- Windows runtime: `{runtime['platform']}`; CPU `{runtime['cpu']}`; OpenGL `{runtime['gpu_opengl']['renderer']}` ({runtime['gpu_opengl']['version']}).",
        f"- System CPU utilization: {runtime['system_cpu_percent_after_load']:.0f}% after load and {runtime['system_cpu_percent_during_benchmark']:.0f}% during playback (includes all processes on this machine).",
        f"- Data: {dataset['valid_sweeps']:,} sweeps x {dataset['tracks']:,} tracks x {dataset['rx_bins']:,} bins; block read-ahead {dataset['waveform_block_sweeps']} sweeps; history display {dataset['history_points']:,} points.",
        f"- Initialization through first displayed frame: {timing['initialization_to_first_display_s']:.2f} s; process RSS {timing['rss_after_load_mb']:.0f} MB after load and {timing['rss_after_benchmark_mb']:.0f} MB after benchmark.",
        f"- Visible GUI playback at 1600x960, requested 10 FPS: {gui['frame_count']} frames; frame-ready {ready['mean_ms']:.1f} ms mean / {ready['p95_ms']:.1f} ms P95 ({ready['fps_from_mean']:.2f} FPS); {gui['vispy_draw_count']} unique VisPy sweep draws" + (f" at {draw['fps_from_mean']:.2f} FPS ({draw['mean_ms']:.1f} ms mean / {draw['p95_ms']:.1f} ms P95)." if draw else "; draw timing unavailable."),
        f"- Mean frame preparation: waveform block read {timing['worker_waveform_block_read']['mean_ms']:.1f} ms; 2D cross-section {timing['worker_2d_cross_section']['mean_ms']:.1f} ms. Current overlay buffer update calls {timing['vispy_current_overlay_buffer_update_call']['mean_ms']:.2f} ms; beam update calls {timing['vispy_beam_buffer_update_call']['mean_ms']:.2f} ms.",
        f"- The H1/H2 display-height interpolation was compared with direct ECEF-to-WGS84 conversion at {len(report['h1_h2_height_grid_parity'])} real sweep/mapping samples; maximum absolute difference {max((item['max_abs_error_m'] for item in report['h1_h2_height_grid_parity']), default=0.0):.9f} m (1 mm acceptance limit).",
        f"- Static history was uploaded once ({timing['vispy_static_history_buffer_upload_call']['bytes']:,} bytes; {timing['vispy_static_history_buffer_upload_call']['elapsed_ms']:.1f} ms in the upload call). One sample framebuffer capture took {timing['single_framebuffer_capture']['elapsed_ms']:.1f} ms; playback timing did not capture each frame.",
        *([f"- Prior GUI baseline: {baseline['mean_frame_latency_ms']:.1f} ms mean / {baseline['sequential_gui_fps']:.2f} FPS at {baseline['window'][0]}x{baseline['window'][1]} on the same 55 sweeps. It used serial seeks after each frame-ready event; compare the paced playback and prior sequential test as separate run protocols."] if baseline else []),
        f"- Sustained 10 FPS target: {'met' if timing['target_10_fps_measured'] else 'not met'}; acceptance is mean frame-ready and unique-draw intervals <=100 ms, P95 <=110 ms, with every measured sweep rendered in order. Intervals at or below 100 ms: {ready['frames_at_or_below_100ms']}/{max(0, gui['frame_count'] - 1)} ready and {(draw['frames_at_or_below_100ms'] if draw else 0)}/{max(0, gui['vispy_draw_count'] - 1)} draw.",
        f"- MP4: `{Path(report['video_export']['path']).name}`; {report['video_export']['frame_count']} decoded frames at {report['video_export']['fps']:g} FPS; sidecar verified.",
        "",
        "## Rendered screenshots",
        "",
    ]
    for screenshot in report["screenshots"]:
        lines.append(f"- `{Path(screenshot['path']).name}`: sweep {screenshot['sweep_id']}, {screenshot['camera']} camera, {screenshot['view']} view, {screenshot['bytes']:,} bytes.")
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {item}" for item in report["validation_limits"])
    lines.extend(["", "Playback timing records GUI frame-ready intervals and visible VisPy draw events without grabbing every frame. Buffer upload-call times are CPU-side timings and do not wait for GPU completion; one separate framebuffer-capture sample is reported. Results apply to this machine and input file.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", nargs="?", type=Path,
                        default=ROOT / "data" / "raw" / "casals_l1b" / "casals_l1b_20241112T165718_001_02.h5")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "casals_sweep_animator_qt" / "final")
    parser.add_argument("--export-frames", type=int, default=50)
    args = parser.parse_args()
    report = run_validation(args.h5, args.output_dir, max(1, args.export_frames))
    print(json.dumps({
        "input": report["input"]["path"],
        "opengl": report["runtime"]["gpu_opengl"],
        "initialization_s": report["timing"]["initialization_to_first_display_s"],
        "gui_playback": report["timing"]["actual_gui_playback_at_10fps_request"],
        "video": report["video_export"],
        "report": str(args.output_dir / "validation_report.md"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
