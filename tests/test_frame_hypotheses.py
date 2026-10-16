"""Focused checks for explicit 3D frame routes and epoch semantics."""

import numpy as np
import pandas as pd
import pytest
from pyproj import datadir, network

from research.reference import frame_hypotheses as frames
from research.reference import vertical_alignment as physical
from research.reference import vertical_datum_comparison as datum


CHECKPOINTS = {
    10605: (-75.28082029258174, 38.4036548426112, -12.16775691627296, 2024.865318192167),
    7912: (-75.28082029258174, 38.4036548426112, -12.16775691627296, 2024.865318192167),
    9989: (-75.28082029258174, 38.4036548426112, -12.16775691627296, 2024.865318192167),
}


@pytest.fixture
def geoid18_environment():
    old_dir, old_network = datadir.get_data_dir(), network.is_network_enabled()
    grid = datum.OUTPUT_ROOT / "proj_data" / datum.GEOID18_GRID
    assert physical.sha256(grid) == datum.GEOID18_GRID_SHA256
    datadir.set_data_dir(str(grid.parent))
    network.set_network_enabled(False)
    try:
        yield grid
    finally:
        datadir.set_data_dir(old_dir)
        network.set_network_enabled(old_network)


def test_hypotheses_use_explicit_3d_epsg_sources_and_separate_h4_sensitivity():
    assert {name: rule["source_epsg"] for name, rule in frames.HYPOTHESES.items()} == {
        "H1_WGS84_G2296_obs_epoch": 10605,
        "H2_ITRF2014_obs_epoch": 7912,
        "H3_ITRF2020_obs_epoch": 9989,
        "H4_ITRF2014_parameter_time_sensitivity_2010": 7912,
    }
    h4 = frames.HYPOTHESES["H4_ITRF2014_parameter_time_sensitivity_2010"]
    assert h4["kind"] == "parameter_time_sensitivity_not_epoch_propagation"
    assert h4["evaluation"] == h4["coordinate_time"] == 2010.0
    assert frames.TARGET_EPSG == 6319 and frames.TARGET_EPOCH == 2010.0


@pytest.mark.parametrize("epsg,reference_epoch", [(10605, 2024.0), (7912, 2010.0), (9989, 2015.0)])
def test_source_crs_audit_reports_dynamic_3d_identity(epsg, reference_epoch):
    audit = frames.crs_audit(epsg)
    assert audit["dimension"] == 3
    assert audit["is_dynamic"] is True
    assert audit["frame_reference_epoch"] == reference_epoch
    assert audit["ellipsoid"]
    assert audit["reference_frame_name"]


@pytest.mark.parametrize("epsg", [10605, 7912, 9989])
def test_selected_database_operations_are_time_dependent_and_audited(epsg):
    lon, lat, height, epoch = CHECKPOINTS[epsg]
    transformer = frames.frame_transformer(epsg, [lon], [lat])
    audit = frames.operation_audit(transformer, epsg, [lon], [lat])
    assert audit["best_available"] and not audit["is_ballpark"] and not audit["is_noop"]
    assert audit["source_epsg"] == epsg and audit["target_epsg"] == 6319
    assert audit["area_contains_input_extent"]
    assert audit["helmert_parameter_reference_epoch"] == 2010.0
    assert "Time-dependent" in audit["operation_method"]["name"]
    assert audit["rotation_convention"] == "coordinate_frame"
    names = {row["name"] for row in audit["helmert_parameters"]}
    assert "X-axis translation" in names
    assert "Rate of change of X-axis translation" in names
    assert "Parameter reference epoch" in names
    assert audit["operation_authority"]
    if epsg == 10605:
        assert len(audit["transformation_steps"]) == 2
        assert [step["authority_id"]["code"] for step in audit["transformation_steps"]] == [10608, 10334]


@pytest.mark.parametrize("epsg", [10605, 7912, 9989])
def test_frame_only_transform_preserves_the_coordinate_time(epsg):
    lon, lat, height, epoch = CHECKPOINTS[epsg]
    out_lon, out_lat, out_h, audit = frames.apply_frame(
        epsg, epoch, [lon], [lat], [height], coordinate_time=epoch
    )
    assert np.isfinite([out_lon[0], out_lat[0], out_h[0]]).all()
    assert audit["coordinate_time_in"] == audit["coordinate_time_out"] == epoch
    assert audit["coordinate_time_preserved"]
    assert audit["coordinate_epoch_propagation_applied"] is False
    assert audit["target_coordinate_epoch_achieved"] is False
    assert audit["roundtrip_max_horizontal_m"] < 1e-6
    assert audit["roundtrip_max_abs_height_m"] < 1e-6


def test_missing_best_operation_is_reported_without_fallback(monkeypatch):
    class MissingGroup:
        best_available = False
        transformers = []
        unavailable_operations = [type("Operation", (), {"name": "unavailable official operation"})()]

    monkeypatch.setattr(frames, "TransformerGroup", lambda *args, **kwargs: MissingGroup())
    with pytest.raises(frames.FrameOperationUnavailable, match="no fallback used"):
        frames.frame_transformer(7912, [-75.0], [38.0])


def test_independent_geocentric_xyz_checkpoint_matches_geographic_route():
    lon, lat, height, epoch = CHECKPOINTS[7912]
    geographic = frames.apply_frame(7912, epoch, [lon], [lat], [height], coordinate_time=epoch)
    xyz = frames.xyz_checkpoint(7912, lon, lat, height, epoch)
    geod = frames.CRS.from_epsg(4326).get_geod()
    out = xyz["output_geographic_lon_lat_h"]
    _, _, distance = geod.inv(float(geographic[0][0]), float(geographic[1][0]), out[0], out[1])
    assert xyz["xyz_units"] == "metre"
    assert abs(distance) < 1e-5
    assert abs(float(geographic[2][0]) - out[2]) < 1e-5
    assert xyz["output_coordinate_time"] == epoch


def test_nad83_c7_route_requires_and_records_pinned_geoid18(geoid18_environment):
    lon, lat, height, _ = (*CHECKPOINTS[7912][:3], 2010.0)
    _, _, out_h, audit = datum.transform_c7([lon], [lat], [height], source_epsg=6319, return_audit=True)
    assert np.isfinite(out_h).all()
    assert audit["selected_operation_grid_constraint"] == datum.GEOID18_GRID
    assert audit["selected_grid_sha256"] == datum.GEOID18_GRID_SHA256
    assert audit["selected_grid_path"] == str(geoid18_environment)


def test_coverage_report_keeps_method_specific_pulse_hashes():
    frame = pd.DataFrame({"pulse_index": [10, 11, 12, 13], "refh_snr": [2.0, 4.0, 6.0, 8.0]})
    valid = np.array([True, False, True, True])
    coverage = frames._scenario_metrics(
        frame, valid, granule_id="test", hypothesis="H1", population="field",
        role_mask=np.array([True, True, True, True]),
    )[1]
    by_stratum = {row["stratum"]: row for row in coverage}
    assert by_stratum["all_valid_refh"]["eligible_n"] == 4
    assert by_stratum["all_valid_refh"]["n"] == 3
    assert by_stratum["snr_ge_5"]["eligible_n"] == 2
    assert by_stratum["snr_ge_5"]["n"] == 2
    assert len(by_stratum["all_valid_refh"]["valid_pulse_sha256"]) == 64
