"""Small, offline checks for NOAA geodetic service request and response handling."""

import pytest
import requests

from research.reference import geodetic_crosscheck as crosscheck


def _vdatum_fixture(params):
    return {
        "region": params["region"].upper(),
        "s_h_frame": params["s_h_frame"],
        "s_coor": params["s_coor"],
        "s_v_frame": params["s_v_frame"],
        "s_v_unit": params["s_v_unit"],
        "s_v_elevation": params["s_v_elevation"],
        "s_v_geoid": params["s_v_geoid"],
        "s_x": params["s_x"],
        "s_y": params["s_y"],
        "s_z": params["s_z"],
        "t_h_frame": params["t_h_frame"],
        "t_coor": params["t_coor"],
        "t_v_frame": params["t_v_frame"],
        "t_v_unit": params["t_v_unit"],
        "t_v_elevation": params["t_v_elevation"],
        "t_v_geoid": params["t_v_geoid"],
        "epoch_in": params["epoch_in"],
        "epoch_out": params["epoch_out"],
        "t_x": params["s_x"],
        "t_y": params["s_y"],
        "t_z": "12.345",
        "uncertainty": "0.06",
    }


def test_ngs_request_uses_explicit_model_and_negative_west_longitude():
    params = crosscheck.build_ngs_params(-76.5, 36.25)
    assert params == {"lat": "36.25", "lon": "-76.5", "model": "14"}
    with pytest.raises(ValueError, match="negative west longitude"):
        crosscheck.build_ngs_params(76.5, 36.25)


def test_ngs_parser_checks_model_units_and_preserves_signed_height():
    assert crosscheck.parse_ngs_geoid_response({
        "geoidModel": "GEOID18", "units": "m", "geoidHeight": -35.897,
    }) == -35.897
    with pytest.raises(crosscheck.CrosscheckResponseError, match="expected GEOID18"):
        crosscheck.parse_ngs_geoid_response({
            "geoidModel": "GEOID12B", "units": "m", "geoidHeight": -35.897,
        })
    with pytest.raises(crosscheck.CrosscheckResponseError, match="expected metres"):
        crosscheck.parse_ngs_geoid_response({
            "geoidModel": "GEOID18", "units": "ft", "geoidHeight": -117.77,
        })


def test_vdatum_request_declares_same_frame_vertical_route_and_epochs():
    params = crosscheck.build_vdatum_params(-75.2, 38.4, -25.5, "chesapeak_delaware")
    assert params["s_v_frame"] == "NAD83_2011"
    assert params["s_v_geoid"] == "geoid18"
    assert params["t_v_frame"] == "NAVD88"
    assert params["t_v_geoid"] == "geoid18"
    assert params["s_v_unit"] == params["t_v_unit"] == "m"
    assert params["epoch_in"] == params["epoch_out"] == "2010.0"
    assert params["region"] == "chesapeak_delaware"
    with pytest.raises(ValueError, match="Unsupported documented VDatum region"):
        crosscheck.build_vdatum_params(-75.2, 38.4, -25.5, "guess")


def test_vdatum_parser_checks_route_units_epochs_and_height_sign():
    params = crosscheck.build_vdatum_params(-75.2, 38.4, -25.5, "chesapeak_delaware")
    payload = _vdatum_fixture(params)
    parsed = crosscheck.parse_vdatum_response(payload, params)
    assert parsed["target_height_m"] == 12.345
    assert parsed["epoch_in"] == parsed["epoch_out"] == 2010.0
    assert parsed["uncertainty_m"] == 0.06

    for field, value in (("s_v_frame", "WGS84_G2296"), ("t_v_unit", "ft"),
                         ("t_v_geoid", "geoid12b"), ("epoch_out", "0.0")):
        bad = dict(payload, **{field: value})
        with pytest.raises(crosscheck.CrosscheckResponseError):
            crosscheck.parse_vdatum_response(bad, params)
    with pytest.raises(crosscheck.CrosscheckResponseError, match="errorCode=412"):
        crosscheck.parse_vdatum_response({"errorCode": 412, "message": "unsupported"}, params)


def test_spatial_sample_is_deterministic_and_spread_over_the_extent():
    lon = [-76.0, -75.9, -75.8, -75.7, -75.6, -75.5]
    lat = [36.0, 36.1, 36.2, 36.3, 36.4, 36.5]
    first = crosscheck._farthest_spatial_positions(lon, lat, 4)
    second = crosscheck._farthest_spatial_positions(lon, lat, 4)
    assert first.tolist() == second.tolist()
    assert len(set(first.tolist())) == 4
    assert min(first) == 0 and max(first) == 5


class _Response:
    status_code = 200
    url = "https://example.test/service?x=1"
    text = '{"geoidModel":"GEOID18","units":"m","geoidHeight":-35.897}'

    @staticmethod
    def json():
        return {"geoidModel": "GEOID18", "units": "m", "geoidHeight": -35.897}


class _Session:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def get(self, url, *, params, timeout):
        self.calls += 1
        assert timeout == crosscheck.REQUEST_TIMEOUT_SECONDS
        if self.error:
            raise self.error
        return self.result


def test_successful_request_raw_response_can_be_reused_without_another_call():
    params = {"lat": "38.4", "lon": "-75.2", "model": "14"}
    session = _Session(result=_Response())
    first = crosscheck._query_service(
        provider="NGS Geoid Height Service", url=crosscheck.NGS_GEOID_URL,
        params=params, parser=crosscheck.parse_ngs_geoid_response,
        session=session, cache={}, cache_sources={},
    )
    second = crosscheck._query_service(
        provider="NGS Geoid Height Service", url=crosscheck.NGS_GEOID_URL,
        params=params, parser=crosscheck.parse_ngs_geoid_response,
        session=session, cache={first["cache_key"]: first},
        cache_sources={first["cache_key"]: "previous-run.json"},
    )
    assert first["status"] == second["status"] == "success"
    assert first["response_text"] == _Response.text
    assert second["cache_reused"] is True
    assert second["cache_source_file"] == "previous-run.json"
    assert session.calls == 1


def test_transport_failure_uses_only_bounded_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(crosscheck.time, "sleep", sleeps.append)
    session = _Session(error=requests.Timeout("offline"))
    record = crosscheck._query_service(
        provider="NGS Geoid Height Service", url=crosscheck.NGS_GEOID_URL,
        params={"lat": "38", "lon": "-75", "model": "14"},
        parser=crosscheck.parse_ngs_geoid_response, session=session, cache={}, cache_sources={},
    )
    assert record["status"] == "failed"
    assert record["attempt_count"] == crosscheck.MAX_ATTEMPTS == 3
    assert session.calls == 3
    assert len(sleeps) == 2
