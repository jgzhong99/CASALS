import h5py
import numpy as np
import pytest
from pyproj import CRS

from casals_l1b.classification import (
    DEFAULT_CONFIG,
    classify_points_baseline,
    read_casals_h5_refh_points,
)
from casals_l1b.geo import (
    horizontal_crs_only,
    infer_wgs84_utm_epsg,
    transform_xy,
)
from casals_l1b.h5 import find_dataset, read_attrs_subset, read_optional_array, require_dataset
from casals_l1b.noise import (
    NoiseConfig,
    REASON_LOW_SNR_AND_LOW_AMP,
    REASON_LOW_SNR_HARD,
    label_noise_arrays,
)
from casals_l1b.raster import disk_structure, fill_nearest_within_mask, robust_normalize
from casals_l1b.waveform import detect_candidate_components_1d, safe_argmax_bin


def test_h5_helpers_find_required_and_optional_datasets(tmp_path):
    path = tmp_path / "small.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("records/value", data=np.array([1, 2, 3], dtype=np.int16))
        h5.attrs["start_utca"] = b"2024-11-12T00:00:00Z"

    with h5py.File(path, "r") as h5:
        values = require_dataset(h5, "records/value")
        assert np.array_equal(values[...], [1, 2, 3])
        assert find_dataset(h5, "value").name == "/records/value"
        assert read_optional_array(h5, "value", 3).tolist() == [1, 2, 3]
        with pytest.raises(ValueError, match="expected 4"):
            read_optional_array(h5, "value", 4, ignore_size_mismatch=False)
        assert read_attrs_subset(h5)["start_utca"] == "2024-11-12T00:00:00Z"


def test_h5_basename_lookup_rejects_ambiguous_matches(tmp_path):
    path = tmp_path / "ambiguous.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("first/value", data=[1])
        h5.create_dataset("second/value", data=[2])

    with h5py.File(path, "r") as h5, pytest.raises(ValueError, match="Multiple datasets"):
        find_dataset(h5, "value")


def test_classifier_h5_reader_uses_shared_h5_and_crs_helpers(tmp_path):
    path = tmp_path / "points.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("refh_longitude", data=[-86.9, -86.8])
        h5.create_dataset("refh_latitude", data=[40.4, 40.5])
        h5.create_dataset("refh", data=[200.0, 201.0])
        h5.create_dataset("refh_amp", data=[10.0, 20.0])
        h5.create_dataset("bg_mean", data=[2.0, 2.0])
        h5.create_dataset("bg_std", data=[2.0, 4.0])
        h5.attrs["projected_crs"] = "EPSG:32616"

    points = read_casals_h5_refh_points(path)
    assert points["fields"]["refh_snr"].tolist() == [4.0, 4.5]
    assert points["derived_fields"] == ["refh_snr"]
    assert points["detected_attr_crs"] == CRS.from_epsg(32616)
    assert points["point_index"].tolist() == [0, 1]


def test_geo_inference_projection_and_horizontal_crs():
    lon = np.array([-86.9, -86.8])
    lat = np.array([40.4, 40.5])
    assert infer_wgs84_utm_epsg(lon, lat) == 32616
    assert infer_wgs84_utm_epsg(np.array([151.0]), np.array([-33.0])) == 32756

    x, y = transform_xy(lon, lat, 4326, CRS.from_epsg(32616))
    lon_roundtrip, lat_roundtrip = transform_xy(x, y, 32616, 4326)
    assert np.allclose(lon_roundtrip, lon, atol=1e-7)
    assert np.allclose(lat_roundtrip, lat, atol=1e-7)
    assert horizontal_crs_only(CRS.from_epsg(32616)) == CRS.from_epsg(32616)


def test_waveform_argmax_and_candidate_detection():
    assert safe_argmax_bin([np.nan, 2.0, 1.0]) == (1.0, 2.0)
    assert np.isnan(safe_argmax_bin([np.nan, np.nan])[0])

    waveform = np.array([0, 0, 0, 1, 4, 1, 0, 0, 3, 0], dtype=float)
    components = detect_candidate_components_1d(
        waveform,
        bg_mean=0.0,
        bg_std=1.0,
        refh_thres=None,
        config={
            "background_mode": "raw",
            "min_height_sigma": 2.0,
            "min_prominence_sigma": 1.0,
            "min_width_bins": 0.3,
            "min_distance_bins": 2,
        },
    )
    assert [item["peak_bin"] for item in components] == [4, 8]
    assert all(item["amplitude_raw"] == item["amplitude_processed"] for item in components)


def test_noise_labels_keep_existing_reason_bits_and_thresholds():
    result = label_noise_arrays(
        z_refh=np.array([100.0, 100.1, 101.0, 100.0]),
        refh_snr=np.array([1.0, 1.7, 2.0, 3.0]),
        refh_amp=np.array([10.0, 20.0, 50.0, 100.0]),
        easting=np.arange(4, dtype=float),
        northing=np.zeros(4),
        cfg=NoiseConfig(
            amp_low_percentile=50.0,
            use_refh_threshold_if_available=False,
            use_error_fields_if_available=False,
            use_global_z_percentile_guard=False,
            use_local_height_filter=False,
        ),
    )
    assert result.reason_code[0] & REASON_LOW_SNR_HARD
    assert result.reason_code[0] & REASON_LOW_SNR_AND_LOW_AMP
    assert result.reason_code[1] & REASON_LOW_SNR_AND_LOW_AMP
    assert result.keep_mask.tolist() == [False, False, True, True]


def test_raster_helpers_preserve_support_mask_and_nodata():
    assert disk_structure(1).astype(int).tolist() == [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
    normalized = robust_normalize(np.array([0.0, 5.0, 10.0, np.nan]), 0.0, 100.0)
    assert np.allclose(normalized[:3], [0.0, 0.5, 1.0])
    assert np.isnan(normalized[3])

    values = np.array([[1.0, np.nan, 99.0], [np.nan, np.nan, 9.0]])
    support = np.array([[True, True, False], [True, True, True]])
    filled, distance = fill_nearest_within_mask(
        values, support, resolution=1.0, max_distance_m=1.0
    )
    assert filled[0, 1] == 1.0
    assert filled[1, 0] == 1.0
    assert np.isnan(filled[0, 2])
    assert distance[0, 1] == 1.0


def test_classification_core_preserves_baseline_class_and_reason_mapping():
    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "CLASSIFIER_MODE": "height_only",
            "GROUND_RESID_TOL_M": 0.2,
            "NOISE_HAG_MAX_M": 3.0,
            "BELOW_GROUND_NOISE_MIN_DEPTH_M": 2.0,
            "USE_NEAR_GROUND_GUARD": False,
            "USE_SIGNAL_DENSITY_NOISE": False,
            "USE_DTM_SUPPORT_CONFIDENCE": False,
            "USE_SCANLINE_OUTLIER_NOISE": False,
            "USE_BELOW_GROUND_REFINEMENT": False,
        }
    )
    result = classify_points_baseline(
        z=np.array([10.0, 10.1, 11.5, 14.0, 7.0, 10.0]),
        local_ground_z_m=np.full(6, 10.0),
        dtm_sample_valid=np.array([True, True, True, True, True, False]),
        point_density_pts_m3=np.full(6, np.nan),
        config=config,
    )
    assert result["pred_class_baseline"].tolist() == [2, 2, 1, 7, 7, 1]
    assert result["classification_reason"].tolist() == [1, 1, 5, 4, 3, 2]
    assert np.allclose(
        result["height_above_ground_m"][:5], [0.0, 0.1, 1.5, 4.0, -3.0]
    )
