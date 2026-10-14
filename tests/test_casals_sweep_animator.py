"""Automated tests for CASALS 3D Sweep Animator and Waveform Cross-Section Viewer.

Covers:
11.1 Sweep indexing (rectangular grid, missing tracks, non-zero start, non-contiguous, shuffled, duplicate check, record axes)
11.2 Geometry (WGS84 -> ECEF -> ENU, origin, East/North/Up, H1/H2 bin mapping, tilted beam, invalid endpoints, degenerate cross-track)
11.3 Heatmap & Cross-section (synthetic waveform, background subtraction, standardized response, height direction, X units, overlays)
11.4 Playback consistency (causality of history points, backward seek, random seek vs sequential, caching)
11.5 GUI smoke test (window launch, loading, control interactions, clean handle close)
"""

from __future__ import annotations

from pathlib import Path
import os
import threading
import time
import h5py
import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets

from debug.casals_sweep_animator_qt.geometry import (
    LocalENU,
    bin_to_xyz,
    compute_trajectory_headings,
    ecef_to_geodetic,
    estimate_cross_track_direction,
    geodetic_to_ecef,
    project_cross_track,
)
from debug.casals_sweep_animator_qt.h5_model import H5Model, SweepData
from debug.casals_sweep_animator_qt.cross_section import (
    CrossSectionGrid,
    _ecef_to_wgs84_height,
    _fill_curtain_row,
    _height_samples,
    compute_waveform_cross_section,
)
from debug.casals_sweep_animator_qt.worker import FrameParams
from debug.casals_sweep_animator_qt.scene3d import aircraft_transform_matrix, enu_axis_segments


def make_synthetic_casals_h5(
    file_path: Path,
    n_sweeps: int = 4,
    n_tracks: int = 8,
    n_bins: int = 64,
    record_axis: int = 0,
    missing_cells: bool = False,
    duplicate_cells: bool = False,
    non_zero_start: bool = False,
    non_contiguous: bool = False,
    shuffled: bool = False,
    missing_bg_fields: bool = False,
) -> Path:
    """Create a minimal synthetic CASALS L1B HDF5 file for testing."""
    sweep_start = 10 if non_zero_start else 0
    sweep_step = 2 if non_contiguous else 1
    sweep_list = [sweep_start + i * sweep_step for i in range(n_sweeps)]

    sweeps = []
    tracks = []
    for s in sweep_list:
        for t in range(n_tracks):
            if missing_cells and s == sweep_list[0] and t == 2:
                continue  # skip track 2 in first sweep
            sweeps.append(s)
            tracks.append(t)

    if duplicate_cells:
        # duplicate the first sweep/track cell
        sweeps.append(sweeps[0])
        tracks.append(tracks[0])

    sweeps = np.array(sweeps, dtype=np.int64)
    tracks = np.array(tracks, dtype=np.int64)
    n_pulses = len(sweeps)

    if shuffled:
        rng = np.random.default_rng(1234)
        perm = rng.permutation(n_pulses)
        sweeps = sweeps[perm]
        tracks = tracks[perm]

    # Geodetic coordinates around Purdue Airport (lat ~40.41, lon ~-86.93)
    base_lon = -86.93
    base_lat = 40.41
    ground_h = 190.0
    inst_alt = 2200.0

    # Cross-track offset along East (dx per track)
    dx_track = 15.0  # meters per track
    dlon_deg = dx_track / (111320.0 * np.cos(np.deg2rad(base_lat)))

    refh_lons = base_lon + (tracks - n_tracks / 2.0) * dlon_deg
    refh_lats = base_lat + (sweeps - sweep_start) * 0.0001
    refh_heights = ground_h + np.sin(tracks * 0.5) * 5.0

    # Instrument trajectory moving North
    inst_lons = base_lon * np.ones(n_pulses, dtype=np.float64)
    inst_lats = base_lat + (sweeps - sweep_start) * 0.0001
    inst_alts = inst_alt * np.ones(n_pulses, dtype=np.float64)

    # Range window: centered around Refh
    # rwstart is above Refh, rwstop is below Refh
    rwstart_h = refh_heights + 150.0
    rwstop_h = refh_heights - 150.0

    # Tilted beam: beam angle ~ 5 degrees off nadir along cross-track
    rwstart_lons = refh_lons - 0.00005
    rwstart_lats = refh_lats
    rwstop_lons = refh_lons + 0.00005
    rwstop_lats = refh_lats

    # Synthetic waveform: background ~ 50, peak at bin 32 with amplitude 800
    waveforms = np.full((n_pulses, n_bins), 50.0, dtype=np.int16)
    for i in range(n_pulses):
        peak_b = 32
        for b in range(n_bins):
            val = 50.0 + 800.0 * np.exp(-0.5 * ((b - peak_b) / 3.0) ** 2)
            waveforms[i, b] = int(round(val))

    with h5py.File(file_path, "w") as h5:
        h5.attrs["n_pulses"] = n_pulses
        h5.attrs["n_sweeps"] = n_sweeps
        h5.attrs["n_tracks"] = n_tracks
        h5.attrs["n_rx_bins"] = n_bins

        h5.create_dataset("sweep_num", data=sweeps)
        h5.create_dataset("track_num", data=tracks)

        h5.create_dataset("refh_longitude", data=refh_lons)
        h5.create_dataset("refh_latitude", data=refh_lats)
        h5.create_dataset("refh", data=refh_heights)

        h5.create_dataset("instrument_longitude", data=inst_lons)
        h5.create_dataset("instrument_latitude", data=inst_lats)
        h5.create_dataset("instrument_altitude", data=inst_alts)

        h5.create_dataset("rwstart_longitude", data=rwstart_lons)
        h5.create_dataset("rwstart_latitude", data=rwstart_lats)
        h5.create_dataset("rwstart", data=rwstart_h)

        h5.create_dataset("rwstop_longitude", data=rwstop_lons)
        h5.create_dataset("rwstop_latitude", data=rwstop_lats)
        h5.create_dataset("rwstop", data=rwstop_h)

        if record_axis == 0:
            h5.create_dataset("rx_waveform", data=waveforms)
        else:
            h5.create_dataset("rx_waveform", data=waveforms.T)

        if not missing_bg_fields:
            h5.create_dataset("refh_snr", data=np.full(n_pulses, 25.0, dtype=np.float32))
            h5.create_dataset("refh_amp", data=np.full(n_pulses, 850, dtype=np.int16))
            h5.create_dataset("good_snr", data=np.ones(n_pulses, dtype=bool))
            h5.create_dataset("bg_mean", data=np.full(n_pulses, 50.0, dtype=np.float32))
            h5.create_dataset("bg_std", data=np.full(n_pulses, 5.0, dtype=np.float32))
            h5.create_dataset("delta_time", data=np.linspace(100.0, 105.0, n_pulses, dtype=np.float64))

    return file_path


# =============================================================================
# 11.1 Sweep Indexing Tests
# =============================================================================

class TestSweepIndexing:
    def test_rectangular_grid(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "rect.h5", n_sweeps=4, n_tracks=8)
        model = H5Model(h5_path)
        assert model.n_valid_sweeps == 4
        assert model.n_tracks == 8
        assert model.n_records == 32
        assert model.index_info.complete_rectangular_grid is True
        assert len(model.valid_sweep_ids) == 4

    def test_missing_tracks(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "missing.h5", n_sweeps=4, n_tracks=8, missing_cells=True)
        model = H5Model(h5_path)
        assert model.n_valid_sweeps == 4
        assert model.index_info.missing_sweep_track_cells == 1
        assert model.index_info.complete_rectangular_grid is False
        with h5py.File(h5_path, "r") as h5:
            sw = model.read_sweep_data(h5, 0)
        np.testing.assert_array_equal(sw.track_nums, np.arange(8))
        assert sw.track_present.shape == (8,)
        assert not sw.track_present[2]
        assert sw.pulse_indices[2] == -1
        assert np.isnan(sw.waveform_matrix[2]).all()
        assert np.isnan(sw.refh_enu[2]).all()
        assert sw.waveform_matrix.shape == (8, 64)

    def test_non_zero_start_and_non_contiguous_sweeps(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(
            tmp_path / "non_zero.h5",
            n_sweeps=4,
            n_tracks=8,
            non_zero_start=True,
            non_contiguous=True,
        )
        model = H5Model(h5_path)
        # Sweeps should be [10, 12, 14, 16]
        assert list(model.valid_sweep_ids) == [10, 12, 14, 16]
        assert model.n_valid_sweeps == 4

    def test_shuffled_pulse_order(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "shuffled.h5", n_sweeps=4, n_tracks=8, shuffled=True)
        model = H5Model(h5_path)
        assert model.n_valid_sweeps == 4
        with h5py.File(h5_path, "r") as h5:
            # Check sweep 0 data
            sw = model.read_sweep_data(h5, 0)
            assert len(sw.track_nums) == 8
            # Waveform maximum should be at bin 32
            assert np.all(np.argmax(sw.waveform_matrix, axis=1) == 32)

    def test_duplicate_sweep_track_detected(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "dup.h5", n_sweeps=4, n_tracks=8, duplicate_cells=True)
        with pytest.raises(ValueError, match="duplicate"):
            H5Model(h5_path)

    def test_record_axis_1(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "axis1.h5", n_sweeps=3, n_tracks=6, record_axis=1)
        model = H5Model(h5_path)
        assert model.record_axis == 1
        with h5py.File(h5_path, "r") as h5:
            sw = model.read_sweep_data(h5, 0)
            assert sw.waveform_matrix.shape == (6, 64)


# =============================================================================
# 11.2 Geometry Tests
# =============================================================================

class TestGeometry:
    def test_enu_axis_segments_have_equal_metric_length(self):
        origin = np.asarray([120.0, -45.0, 8.0])
        length_m = 25.0
        segments = enu_axis_segments(origin, length_m)
        np.testing.assert_allclose(segments[::2], np.broadcast_to(origin, (3, 3)), atol=1e-6)
        vectors = segments[1::2] - segments[::2]
        np.testing.assert_allclose(vectors, np.eye(3) * length_m, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [length_m] * 3, atol=1e-6)

    def test_height_only_ecef_conversion_matches_full_wgs84_conversion(self):
        lon = np.linspace(-150.0, 150.0, 1000)
        lat = np.linspace(-80.0, 80.0, 1000)
        height = np.linspace(-500.0, 9000.0, 1000)
        xyz = geodetic_to_ecef(lon, lat, height)
        np.testing.assert_allclose(_ecef_to_wgs84_height(xyz), ecef_to_geodetic(xyz)[:, 2], rtol=0.0, atol=1e-6)

    def test_aircraft_transform_uses_enu_translation_and_heading(self):
        from vispy.visuals.transforms import MatrixTransform

        matrix = aircraft_transform_matrix(np.asarray([100.0, 200.0, 3000.0]), 90.0)
        mapped = MatrixTransform(matrix).map(np.asarray([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 10.0, 0.0, 1.0],
        ], dtype=np.float32))
        np.testing.assert_allclose(mapped[0], [100.0, 200.0, 3000.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(mapped[1], [110.0, 200.0, 3000.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(matrix[:3, :3], axis=0), [1.0, 1.0, 1.0], atol=1e-6)

    def test_curtain_interpolation_fast_path_preserves_increasing_and_decreasing_height(self):
        response = np.square(np.linspace(-1.0, 1.0, 256))
        grid = np.linspace(-0.9, 0.9, 128)
        for heights in (np.linspace(-1.0, 1.0, 256), np.linspace(1.0, -1.0, 256)):
            actual = np.full(grid.shape, np.nan, dtype=np.float32)
            _fill_curtain_row(actual, grid, heights, response)
            expected = np.interp(grid, heights if heights[0] < heights[-1] else heights[::-1],
                                 response if heights[0] < heights[-1] else response[::-1])
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)

    def test_geodetic_ecef_roundtrip(self):
        lon = np.array([-86.93, 0.0, 120.5])
        lat = np.array([40.41, 0.0, -33.8])
        h = np.array([190.0, 0.0, 50.0])

        xyz = geodetic_to_ecef(lon, lat, h)
        assert xyz.shape == (3, 3)

        geo = ecef_to_geodetic(xyz)
        np.testing.assert_allclose(geo[:, 0], lon, atol=1e-8)
        np.testing.assert_allclose(geo[:, 1], lat, atol=1e-8)
        np.testing.assert_allclose(geo[:, 2], h, atol=1e-4)

    def test_local_enu_origin_and_directions(self):
        origin_lon = -86.93
        origin_lat = 40.41
        origin_h = 190.0
        enu = LocalENU(origin_lon, origin_lat, origin_h)

        # Origin in ENU must be [0, 0, 0]
        orig_enu = enu.geodetic_to_enu(origin_lon, origin_lat, origin_h)
        np.testing.assert_allclose(orig_enu, [0.0, 0.0, 0.0], atol=1e-5)

        # Roundtrip ENU <-> ECEF
        test_pts = np.array([[100.0, -50.0, 20.0], [0.0, 0.0, 0.0]])
        ecef = enu.enu_to_ecef(test_pts)
        enu_back = enu.ecef_to_enu(ecef)
        np.testing.assert_allclose(enu_back, test_pts, atol=1e-5)

        # Move 1000m Up in ENU
        pt_up = enu.enu_to_geodetic(np.array([0.0, 0.0, 1000.0]))
        np.testing.assert_allclose(pt_up[2], origin_h + 1000.0, atol=0.01)

    def test_h1_and_h2_bin_mapping(self):
        p0 = np.array([0.0, 0.0, 2000.0])
        p1 = np.array([100.0, 0.0, 1600.0])
        n_bins = 200

        # H1: b=0 -> p0, b=N-1 -> p1
        pos_b0_h1 = bin_to_xyz(p0, p1, 0, n_bins, convention="H1")
        pos_bN_h1 = bin_to_xyz(p0, p1, n_bins - 1, n_bins, convention="H1")
        np.testing.assert_allclose(pos_b0_h1, p0, atol=1e-6)
        np.testing.assert_allclose(pos_bN_h1, p1, atol=1e-6)

        # H2: b=0 -> p0 + 0.5/N*(p1-p0), b=N-1 -> p0 + (N-0.5)/N*(p1-p0)
        pos_b0_h2 = bin_to_xyz(p0, p1, 0, n_bins, convention="H2")
        pos_bN_h2 = bin_to_xyz(p0, p1, n_bins - 1, n_bins, convention="H2")
        expected_b0 = p0 + 0.5 / n_bins * (p1 - p0)
        expected_bN = p0 + (n_bins - 0.5) / n_bins * (p1 - p0)
        np.testing.assert_allclose(pos_b0_h2, expected_b0, atol=1e-6)
        np.testing.assert_allclose(pos_bN_h2, expected_bN, atol=1e-6)

    def test_tilted_beam_causes_horizontal_displacement(self):
        p0 = np.array([0.0, 0.0, 2000.0])
        p1 = np.array([50.0, 10.0, 1600.0])
        n_bins = 100

        pos_b0 = bin_to_xyz(p0, p1, 0, n_bins, convention="H2")
        pos_b50 = bin_to_xyz(p0, p1, 50, n_bins, convention="H2")

        # Horizontal positions must differ
        assert pos_b0[0] != pos_b50[0]
        assert pos_b0[1] != pos_b50[1]

    def test_trajectory_headings(self):
        # Aircraft moving pure North: dy > 0, dx = 0 -> Heading = 0 deg
        traj_north = np.array([[0.0, 0.0, 2000.0], [0.0, 100.0, 2000.0], [0.0, 200.0, 2000.0]])
        headings = compute_trajectory_headings(traj_north)
        np.testing.assert_allclose(headings, [0.0, 0.0, 0.0], atol=1e-3)

        # Aircraft moving pure East: dx > 0, dy = 0 -> Heading = 90 deg
        traj_east = np.array([[0.0, 0.0, 2000.0], [100.0, 0.0, 2000.0], [200.0, 0.0, 2000.0]])
        headings_e = compute_trajectory_headings(traj_east)
        np.testing.assert_allclose(headings_e, [90.0, 90.0, 90.0], atol=1e-3)

    def test_cross_track_estimation(self):
        # 16 tracks spaced along East (X-axis in ENU)
        tracks = np.arange(16)
        xy = np.column_stack([tracks * 10.0, np.zeros(16)])
        v, center, ok, reason = estimate_cross_track_direction(xy, tracks)
        assert ok is True
        # v must point along East: [1, 0]
        np.testing.assert_allclose(v, [1.0, 0.0], atol=1e-3)

        # Degenerate case: all points identical
        xy_deg = np.zeros((16, 2))
        _, _, ok_deg, _ = estimate_cross_track_direction(xy_deg, tracks)
        assert ok_deg is False


# =============================================================================
# 11.3 Heatmap & Cross-Section Tests
# =============================================================================

class TestCrossSectionHeatmap:
    def test_h1_h2_height_grid_matches_ecef_bin_geometry(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "height_grid.h5", n_sweeps=3, n_tracks=8, n_bins=64)
        model = H5Model(h5_path)
        with h5py.File(h5_path, "r") as h5:
            sweep = model.read_sweep_data(h5, 0)
        bins = np.broadcast_to(np.arange(sweep.n_rx_bins), (sweep.n_tracks, sweep.n_rx_bins))
        for mapping in ("H1", "H2"):
            xyz = bin_to_xyz(sweep.rwstart_ecef[:, None, :], sweep.rwstop_ecef[:, None, :],
                             bins, sweep.n_rx_bins, convention=mapping)
            expected = ecef_to_geodetic(xyz.reshape(-1, 3))[:, 2].reshape(sweep.n_tracks, sweep.n_rx_bins)
            actual = _height_samples(sweep, mapping)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-5, equal_nan=True)

    def test_cross_section_modes(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "cs.h5", n_sweeps=3, n_tracks=8, n_bins=64)
        model = H5Model(h5_path)
        with h5py.File(h5_path, "r") as h5:
            sw = model.read_sweep_data(h5, 0)

        # Mode A: Raw
        grid_raw = compute_waveform_cross_section(sw, mode="raw", view="curtain", mapping_convention="H2")
        assert grid_raw.grid_data.shape == (8, 512)
        assert grid_raw.color_title == "Raw RX"

        # Mode A: Minus BG
        grid_mbg = compute_waveform_cross_section(sw, mode="minus_bg", view="curtain", mapping_convention="H2")
        assert grid_mbg.color_title == "Net RX"
        assert np.isfinite(grid_mbg.grid_data).any()

        # Mode B: Standardized
        grid_std = compute_waveform_cross_section(sw, mode="standardized", view="curtain", mapping_convention="H2")
        assert "Standardized" in grid_std.color_title
        fin = grid_std.grid_data[np.isfinite(grid_std.grid_data)]
        assert len(fin) > 0

        # Mode C: Refh only
        grid_refh = compute_waveform_cross_section(sw, mode="refh_only", view="curtain", mapping_convention="H2")
        assert np.all(np.isnan(grid_refh.grid_data))
        assert len(grid_refh.refh_x) == 8

        grid_geo = compute_waveform_cross_section(sw, mode="raw", view="georeferenced", mapping_convention="H2", n_cols=64, n_rows=48)
        assert grid_geo.grid_data.shape == (64, 48)
        assert not grid_geo.available
        assert "track span" in grid_geo.message

        geo_path = make_synthetic_casals_h5(tmp_path / "cs_geo.h5", n_sweeps=3, n_tracks=16, n_bins=64)
        geo_model = H5Model(geo_path)
        with h5py.File(geo_path, "r") as h5:
            geo_sweep = geo_model.read_sweep_data(h5, 0)
        grid_geo = compute_waveform_cross_section(geo_sweep, mode="raw", view="georeferenced", n_cols=64, n_rows=48)
        assert grid_geo.available
        assert grid_geo.grid_data.shape == (64, 48)
        assert np.isfinite(grid_geo.grid_data).any()

    def test_missing_background_fields_fallback(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "no_bg.h5", n_sweeps=3, n_tracks=8, missing_bg_fields=True)
        model = H5Model(h5_path)
        with h5py.File(h5_path, "r") as h5:
            sw = model.read_sweep_data(h5, 0)
        grid = compute_waveform_cross_section(sw, mode="standardized")
        assert not grid.available
        assert "missing" in grid.message
        assert np.isnan(grid.grid_data).all()

    def test_refh_overlay_matches_official_values(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "overlay.h5", n_sweeps=3, n_tracks=8)
        model = H5Model(h5_path)
        with h5py.File(h5_path, "r") as h5:
            sw = model.read_sweep_data(h5, 0)
        grid = compute_waveform_cross_section(sw, mode="raw", view="curtain")
        # Curtain columns preserve their full track indices.
        np.testing.assert_array_equal(grid.refh_x, sw.track_nums)
        np.testing.assert_array_equal(grid.refh_h, sw.refh_height)

    def test_argmax_height_comes_from_ecef_bin_geometry(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "ecef_height.h5", n_sweeps=2, n_tracks=16, n_bins=64)
        model = H5Model(h5_path)
        with h5py.File(h5_path, "r") as h5:
            sw = model.read_sweep_data(h5, 0)
        grid = compute_waveform_cross_section(
            sw, mode="raw", view="georeferenced", mapping_convention="H2", n_cols=64, n_rows=48
        )
        assert grid.available
        bins = np.argmax(sw.waveform_matrix, axis=1)
        xyz = bin_to_xyz(sw.rwstart_ecef, sw.rwstop_ecef, bins, sw.n_rx_bins, convention="H2")
        expected_height = ecef_to_geodetic(xyz)[:, 2]
        np.testing.assert_allclose(grid.argmax_h, expected_height, rtol=0.0, atol=1e-7)


# =============================================================================
# 11.4 Playback Data Consistency Tests
# =============================================================================

class TestPlaybackConsistency:
    def test_causality_and_backward_seek(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "play.h5", n_sweeps=6, n_tracks=8)
        model = H5Model(h5_path, history_budget=50)

        # Advance to sweep 3
        h3_enu, h3_h, _, _ = model.get_history_slice(3)
        n3 = len(h3_enu)

        # Advance to sweep 5
        h5_enu, h5_h, _, _ = model.get_history_slice(5)
        n5 = len(h5_enu)
        assert n5 >= n3

        # Seek backward to sweep 3: future points must disappear
        h3_back, _, _, _ = model.get_history_slice(3)
        assert len(h3_back) == n3
        np.testing.assert_array_equal(h3_back, h3_enu)

    def test_random_seek_identical_to_sequential(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "seek.h5", n_sweeps=6, n_tracks=8)
        model = H5Model(h5_path, history_budget=100)

        # Sequential
        seq_history = [model.get_history_slice(i)[0] for i in range(6)]

        # Random seek
        rng = np.random.default_rng(42)
        order = rng.permutation(6)
        for i in order:
            h_rand = model.get_history_slice(i)[0]
            np.testing.assert_array_equal(h_rand, seq_history[i])

    def test_waveform_cache_consistency(self, tmp_path: Path):
        h5_path = make_synthetic_casals_h5(tmp_path / "cache.h5", n_sweeps=4, n_tracks=8)
        model = H5Model(h5_path, waveform_block_sweeps=2, waveform_cache_bytes=100_000)
        with h5py.File(h5_path, "r") as h5:
            first = model.read_sweep_data(h5, 0)
            second = model.read_sweep_data(h5, 1)
            third = model.read_sweep_data(h5, 2)
        np.testing.assert_array_equal(first.waveform_matrix, second.waveform_matrix)
        stats = model.waveform_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert 0 < stats["bytes"] <= stats["limit_bytes"]
        assert np.all(np.argmax(third.waveform_matrix, axis=1) == 32)


class TestFrameWorker:
    @staticmethod
    def _wait(app, predicate, timeout=8.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            app.processEvents()
            if predicate():
                return True
            time.sleep(0.005)
        return predicate()

    def test_latest_request_and_file_generation_are_enforced(self, tmp_path: Path, qapp, monkeypatch):
        from debug.casals_sweep_animator_qt import worker as worker_module
        from debug.casals_sweep_animator_qt.frame_controller import FrameController

        first_path = make_synthetic_casals_h5(tmp_path / "worker_first.h5", n_sweeps=5, n_tracks=12, n_bins=32)
        second_path = make_synthetic_casals_h5(tmp_path / "worker_second.h5", n_sweeps=5, n_tracks=12, n_bins=32)
        controller = FrameController()
        assert controller._timer.timerType() == QtCore.Qt.PreciseTimer
        loaded = []
        frames = []
        controller.model_loaded.connect(lambda model, generation: loaded.append((model, generation)))
        controller.frame_ready.connect(frames.append)
        release = threading.Event()

        try:
            gen1 = controller.load_file(str(first_path))
            assert self._wait(qapp, lambda: bool(frames))
            assert frames[-1].file_generation == gen1

            original = worker_module.compute_waveform_cross_section
            entered = threading.Event()
            block_next = {"value": True}

            def delayed_cross_section(*args, **kwargs):
                sweep = args[0]
                if block_next["value"] and sweep.sweep_index == 1:
                    block_next["value"] = False
                    entered.set()
                    assert release.wait(4.0)
                return original(*args, **kwargs)

            monkeypatch.setattr(worker_module, "compute_waveform_cross_section", delayed_cross_section)
            frames.clear()
            controller.request_frame(1)
            assert entered.wait(4.0)
            gen2 = controller.load_file(str(second_path))
            release.set()
            assert self._wait(qapp, lambda: any(generation == gen2 for _, generation in loaded))
            assert self._wait(qapp, lambda: frames and frames[-1].file_generation == gen2)
            assert all(frame.file_generation == gen2 for frame in frames)

            entered.clear()
            release.clear()
            block_next["value"] = True
            delivered_before = len(frames)
            controller.request_frame(1)
            assert entered.wait(4.0)
            final_index = controller.model.n_valid_sweeps - 1
            for index in range(2, final_index + 1):
                controller.request_frame(index)
            final_request_id = controller.request_id
            release.set()
            assert self._wait(qapp, lambda: len(frames) > delivered_before and frames[-1].request_id == final_request_id)
            assert frames[-1].sweep_index == final_index
            assert frames[-1].file_generation == gen2
            assert all(frame.request_id == final_request_id for frame in frames[delivered_before:])
        finally:
            release.set()
            controller.shutdown()


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


# =============================================================================
# 11.5 GUI Smoke Tests
# =============================================================================

class TestGUISmoke:
    def test_mainwindow_offscreen_launch_and_load(self, tmp_path: Path, qapp):
        import time
        h5_path = make_synthetic_casals_h5(tmp_path / "gui_smoke.h5", n_sweeps=4, n_tracks=8)
        from debug.casals_sweep_animator_qt.main_window import MainWindow

        win = MainWindow(initial_h5=str(h5_path))

        # Wait for file_loaded signal or model to become populated
        t0 = time.time()
        while (win.model is None or win.current_frame is None) and (time.time() - t0 < 5.0):
            QtWidgets.QApplication.processEvents()
            time.sleep(0.05)

        assert win.model is not None
        assert win.model.n_valid_sweeps == 4
        assert win.current_frame is not None

        # Test controls exist and work
        win._on_next()
        assert win.current_sweep_idx == 1

        win._on_prev()
        assert win.current_sweep_idx == 0

        # Test mode switching does not crash
        win.combo_2d_mode.setCurrentText("Raw RX amplitude")
        win.combo_2d_view.setCurrentText("Georeferenced Cross-Section")
        win.combo_mapping.setCurrentText("H1")
        QtWidgets.QApplication.processEvents()

        # Test close
        win.close()

