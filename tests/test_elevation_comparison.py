import warnings

import numpy as np
import pytest

from research.elevation_comparison import (
    build_class2_reference_grid,
    class2_ground_points,
    elevation_metrics,
    empirical_alignment_metrics,
    horizontal_transform_audit,
    interpolate_ground_surface,
    snr_strata,
    spatial_site_masks,
)
from research.l1b_validation import reference_readiness


def test_horizontal_transform_is_explicitly_2d():
    x, y, audit = horizontal_transform_audit(np.array([-75.0]), np.array([38.0]))
    assert 400_000 < x[0] < 600_000
    assert 4_000_000 < y[0] < 4_300_000
    assert audit["target_crs"] == "EPSG:6347"
    assert audit["coordinate_status"].startswith("2D exploratory")


def test_ground_filter_keeps_only_finite_class_two():
    xyz = class2_ground_points(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([5.0, 6.0, np.nan, 8.0]),
        np.array([2, 1, 2, 2]),
    )
    np.testing.assert_array_equal(xyz, [[0.0, 0.0, 5.0], [3.0, 3.0, 8.0]])


def test_class2_reference_grid_keeps_record_stride_out_of_training(tmp_path):
    import laspy
    from pyproj import CRS

    path = tmp_path / "small_reference.las"
    las = laspy.create(point_format=3, file_version="1.2")
    las.header.add_crs(CRS.from_epsg(6347))
    x = 500_000.0 + np.tile(np.arange(5, dtype=float), 4)
    y = 4_200_000.0 + np.repeat(np.arange(4, dtype=float), 5)
    z = 100.0 + 0.1 * (x - 500_000.0) + 0.2 * (y - 4_200_000.0)
    las.x, las.y, las.z = x, y, z
    las.classification = np.where(np.arange(len(x)) % 7 == 0, 1, 2)
    las.write(path)

    ground_xy, ground_z, held_out, metadata = build_class2_reference_grid(
        path, resolution_m=0.5, validation_stride=5, chunk_size=8
    )

    assert metadata["las_point_count"] == 20
    assert len(held_out) == 3
    assert metadata["class2_training_point_count"] + len(held_out) == 17
    assert len(ground_xy) == len(ground_z) > 0


def test_robust_local_plane_recovers_flat_and_sloped_surfaces_with_outlier():
    gx, gy = np.meshgrid(np.arange(-4.0, 4.1, 0.5), np.arange(-4.0, 4.1, 0.5))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    z = 12.0 + 0.025 * xy[:, 0] - 0.018 * xy[:, 1]
    z[np.argmin(np.sum(xy * xy, axis=1))] += 20.0
    query = np.array([[0.0, 0.0], [1.25, -0.75]])
    result = interpolate_ground_surface(query, xy, z)
    expected = 12.0 + 0.025 * query[:, 0] - 0.018 * query[:, 1]
    np.testing.assert_allclose(result["reference_height_plane_m"], expected, atol=0.03)
    assert result["surface_acceptable"].all()
    assert (result["robust_neighbor_count"] >= 6).all()
    assert np.nanmax(result["local_slope_deg"]) < 5.0


def test_unsupported_sparse_surface_is_not_extrapolated():
    xy = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
    result = interpolate_ground_surface(np.array([[0.0, 0.0], [10.0, 10.0]]), xy, np.ones(4))
    assert not result["plane_fit_valid"].any()
    assert np.isnan(result["reference_height_plane_m"]).all()
    assert (result["surface_status"] == "insufficient_neighbors").all()


def test_ill_conditioned_neighborhood_does_not_emit_all_nan_median_warning():
    x = np.linspace(-1.0, 1.0, 9)
    xy = np.column_stack([x, np.zeros_like(x)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = interpolate_ground_surface(np.array([[0.0, 0.0]]), xy, 5.0 + x)

    assert result["surface_status"][0] == "ill_conditioned_plane"
    assert not any("All-NaN slice encountered" in str(item.message) for item in caught)


def test_raw_metrics_and_empirical_alignment_recover_known_38m_offset():
    rng = np.random.default_rng(104)
    error = rng.normal(0.0, 0.20, 4000)
    sites = np.array(["cal"] * 2800 + ["test"] * 1200, dtype=object)
    difference = 38.0 + error
    calibration, test = spatial_site_masks(sites, ["cal"])
    result = empirical_alignment_metrics(difference, calibration, test)
    raw = elevation_metrics(difference)
    assert abs(raw["bias_m"] - 38.0) < 0.02
    assert abs(result["empirical_vertical_offset_m"] - 38.0) < 0.02
    assert abs(result["bias_m"]) < 0.03
    assert abs(result["rmse_m"] - 0.20) < 0.02
    assert result["evaluation_semantics"].startswith("spatial holdout agreement")
    assert reference_readiness({})["status"] == "BLOCKED"
    assert raw["rmse_m"] > 37.0  # remains a raw difference, not absolute accuracy


def test_test_values_never_affect_calibration_offset():
    difference = np.array([38.0, 39.0, 200.0, -100.0])
    sites = np.array(["cal", "cal", "test", "test"], dtype=object)
    calibration, test = spatial_site_masks(sites, ["cal"])
    a = empirical_alignment_metrics(difference, calibration, test)["empirical_vertical_offset_m"]
    difference[test] = 1e6
    b = empirical_alignment_metrics(difference, calibration, test)["empirical_vertical_offset_m"]
    assert a == b == 38.5


def test_spatially_varying_offset_remains_in_holdout_residuals():
    difference = np.array([38.0, 38.1, 39.0, 39.1])
    sites = np.array(["cal", "cal", "test", "test"], dtype=object)
    calibration, test = spatial_site_masks(sites, ["cal"])
    result = empirical_alignment_metrics(difference, calibration, test)
    assert result["empirical_vertical_offset_m"] == pytest.approx(38.05)
    assert result["bias_m"] == pytest.approx(1.00)
    assert result["rmse_m"] == pytest.approx(1.0012, abs=0.002)


def test_empirical_alignment_accepts_read_only_split_masks():
    difference = np.array([2.0, 2.2, 3.0, 3.2])
    calibration = np.array([True, True, False, False])
    test = np.array([False, False, True, True])
    calibration.setflags(write=False)
    test.setflags(write=False)

    result = empirical_alignment_metrics(difference, calibration, test)

    assert result["empirical_vertical_offset_m"] == pytest.approx(2.1)
    assert result["rmse_m"] == pytest.approx(np.sqrt(1.01))


def test_spatial_split_is_disjoint_and_depends_only_on_sites():
    sites = np.array(["field_a", "field_a", "field_b", "road"])
    calibration, test = spatial_site_masks(sites, ["field_a"])
    assert not np.any(calibration & test)
    np.testing.assert_array_equal(calibration, [True, True, False, False])
    np.testing.assert_array_equal(test, [False, False, True, True])
    with pytest.raises(ValueError, match="both be represented"):
        spatial_site_masks(sites, ["absent"])


def test_snr_sensitivity_masks_are_nested_and_keep_all_valid():
    masks = snr_strata(np.array([np.nan, 0.5, 2.9, 3.0, 4.9, 5.0]))
    assert masks["all_valid_refh"].sum() == 5
    assert masks["snr_ge_3"].sum() == 3
    assert masks["snr_ge_5"].sum() == 1
    assert np.all(masks["snr_ge_5"] <= masks["snr_ge_3"])
