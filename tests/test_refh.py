import h5py
import laspy
import numpy as np
import pytest

from casals_l1b.refh import (
    ProjectedRefh,
    RefhSelection,
    read_refh_points,
    write_refh_las,
)


def _make_refh_h5(path):
    with h5py.File(path, "w") as h5:
        h5.attrs["n_pulses"] = 4
        h5.attrs["source"] = b"synthetic"
        h5.create_dataset("refh_longitude", data=[-72.0, -72.1, -72.2, -72.3])
        h5.create_dataset("refh_latitude", data=[42.0, 42.1, 42.2, 42.3])
        h5.create_dataset("refh", data=[100.0, 101.0, 102.0, 103.0])
        h5.create_dataset("refh_amp", data=[10.0, 20.0, 30.0, 40.0])
        h5.create_dataset("refh_snr", data=[2.0, 1.0, 3.0, 4.0])
        h5.create_dataset("good_snr", data=[1, 0, 1, 1])
        h5.create_dataset("track_num", data=[1, 1, 2, 2])
        h5.create_dataset("sweep_num", data=[1, 2, 1, 2])
        h5.create_dataset("delta_time", data=[0.1, 0.2, 0.3, 0.4])
        h5.create_dataset("refh_error", data=np.zeros((2, 2)))
        h5.create_dataset("nested/unused", data=[7])


def test_read_refh_points_preserves_order_and_skips_malformed_optional(tmp_path):
    path = tmp_path / "synthetic.h5"
    _make_refh_h5(path)

    with pytest.warns(UserWarning, match="Optional dataset 'refh_error'"):
        data = read_refh_points(path)

    assert data.n_input_records == 4
    assert data.n_valid_records == 4
    assert data.pulse_index.tolist() == [0, 1, 2, 3]
    assert data.attrs["source"] == "synthetic"
    assert data.optional["delta_time"].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert "refh_error" not in data.optional

    selected = read_refh_points(
        path,
        selection=RefhSelection(
            filter_good_snr_only=True,
            refh_snr_min=3.0,
            track_range=(2, 2),
            sweep_range=(2, 2),
        ),
        optional_fields=("delta_time",),
    )
    assert selected.pulse_index.tolist() == [3]
    assert selected.track_num.tolist() == [2]
    assert selected.sweep_num.tolist() == [2]


def test_read_refh_points_keeps_root_only_contract_and_strict_optional_mode(tmp_path):
    path = tmp_path / "synthetic.h5"
    _make_refh_h5(path)
    with h5py.File(path, "a") as h5:
        refh = h5["refh"][...]
        del h5["refh"]
        h5.create_dataset("nested/refh", data=refh)

    with pytest.raises(KeyError, match="root dataset 'refh'"):
        read_refh_points(path)

    _make_refh_h5(path)
    with pytest.raises(ValueError, match="expected to be 1D"):
        read_refh_points(path, optional_fields=("refh_error",), ignore_bad_optional_fields=False)


def test_write_refh_las_preserves_selected_coordinates_and_attributes(tmp_path):
    source = tmp_path / "synthetic.h5"
    output = tmp_path / "points.las"
    _make_refh_h5(source)
    data = read_refh_points(source, optional_fields=("delta_time",))
    projected = ProjectedRefh(
        easting=np.array([500000.0, 500001.0, 500002.0, 500003.0]),
        northing=np.array([4600000.0, 4600001.0, 4600002.0, 4600003.0]),
        utm_epsg=32618,
        utm_crs_name="WGS 84 / UTM zone 18N",
        projection_check={},
    )

    info = write_refh_las(
        output,
        data,
        projected,
        mask=np.array([False, True, False, True]),
        include_optional_fields=True,
    )

    assert info["n_points"] == 2
    with laspy.open(output) as reader:
        las = reader.read()
    assert np.array_equal(las.x, [500001.0, 500003.0])
    assert np.array_equal(las.z, [101.0, 103.0])
    assert np.array_equal(las.pulse_index, [1, 3])
    assert np.array_equal(las.delta_time, [0.2, 0.4])
    assert np.array_equal(las.classification, [1, 1])
    assert reader.header.parse_crs().to_epsg() == 32618
