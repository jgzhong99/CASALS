import numpy as np
import pandas as pd
import h5py

from casals_l1b.geolocation import (
    BEAM_METHOD,
    SEGMENT_METHOD,
    beam_direction_enu,
    ecef_to_geodetic,
    geodetic_to_ecef,
    interpolate_segment,
    locate_peaks,
    validate_candidate_table,
)


def test_coordinate_round_trip_segment_and_beam_convention():
    geo = np.array([[-75.25, 38.4, 123.0], [12.0, -43.0, -10.0]])
    ecef = geodetic_to_ecef(geo[:, 0], geo[:, 1], geo[:, 2])
    np.testing.assert_allclose(ecef_to_geodetic(ecef), geo, atol=1e-8)
    np.testing.assert_allclose(interpolate_segment([0, 0, 0], [10, 20, 30], 0.25), [2.5, 5, 7.5])
    np.testing.assert_allclose(beam_direction_enu(0.0, 0.0), [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(beam_direction_enu(np.pi / 2, 0.0), [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(beam_direction_enu(0.0, np.pi / 2), [0.0, 0.0, -1.0], atol=1e-12)


def test_real_schema_candidate_output_and_invalid_bin_status(tmp_path):
    h5_path = tmp_path / "small.h5"
    start = np.array([-75.0, 38.0, 100.0])
    stop = np.array([-74.9999, 38.0001, 0.0])
    start_xyz = geodetic_to_ecef(*start)
    stop_xyz = geodetic_to_ecef(*stop)
    ref_geo = ecef_to_geodetic(interpolate_segment(start_xyz, stop_xyz, 0.5))
    with h5py.File(h5_path, "w") as h5:
        h5["sweep_num"] = np.array([7, 7])
        h5["track_num"] = np.array([2, 3])
        h5["rx_waveform"] = np.array([[0, 1, 9, 1, 0], [0, 1, 9, 1, 0]], dtype=np.int16)
        for prefix, geo in (("rwstart", start), ("rwstop", stop)):
            h5[f"{prefix}_longitude"] = np.array([geo[0], geo[0]])
            h5[f"{prefix}_latitude"] = np.array([geo[1], geo[1]])
            h5[prefix] = np.array([geo[2], geo[2]])
        for name, value in {
            "refh_longitude": ref_geo[0], "refh_latitude": ref_geo[1], "refh": ref_geo[2],
            "bin_size": np.linalg.norm(stop_xyz - start_xyz) / 4,
            "local_beam_azimuth": 0.2, "local_beam_elevation": 1.4,
            "instrument_longitude": -75.0, "instrument_latitude": 38.0, "instrument_altitude": 1000.0,
        }.items():
            h5[name] = np.full(2, value)

    components = pd.DataFrame({
        "pulse_index": [0, 1], "sweep_num": [7, 7], "track_num": [2, 3],
        "component_rank": [1, 2], "peak_bin": [2.0, 6.0], "amplitude_raw": [10.0, 4.0],
        "prominence": [8.0, 2.0], "is_main_component": [True, False],
        "is_valid_secondary_candidate": [False, True],
    })
    components_path = tmp_path / "components.parquet"
    components.to_parquet(components_path, index=False)
    outputs = locate_peaks(h5_path, components_path, tmp_path / "located", method="both")
    candidates = pd.read_parquet(outputs["candidate_points"])
    assert set(candidates["geolocation_method"]) == {SEGMENT_METHOD, BEAM_METHOD}
    valid = candidates[candidates["pulse_index"] == 0]
    assert valid["geometry_valid"].all()
    assert (valid["geometry_status"] == "ok").all()
    assert candidates.loc[candidates["pulse_index"] == 1, "geometry_status"].eq("invalid_peak_bin").all()
    assert candidates.loc[candidates["pulse_index"] == 1, ["candidate_x", "candidate_y", "candidate_z"]].isna().all().all()
    closure = pd.read_csv(outputs["validation_summary"])
    closure = closure[closure["scope"] == "refh_closure"]
    assert closure["3d_median"].iloc[0] < 1e-5
    validation = validate_candidate_table(h5_path, outputs["candidate_points"], tmp_path / "validated")
    assert pd.read_csv(validation["validation_summary"])["status"].eq("pass").all()
