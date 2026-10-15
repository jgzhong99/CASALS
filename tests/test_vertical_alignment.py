"""Physical-only boundaries, independent-reference invariance, and real-model parity."""
import ast
import inspect
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from pyproj import datadir, network

from research.reference import vertical_alignment as physical


@pytest.fixture
def model_environment():
    old, online = datadir.get_data_dir(), network.is_network_enabled()
    grid = physical.datum.OUTPUT_ROOT/"proj_data"/physical.datum.GEOID18_GRID
    assert physical.sha256(grid) == physical.GRID_SHA256
    datadir.set_data_dir(str(grid.parent))
    network.set_network_enabled(False)
    yield grid
    datadir.set_data_dir(old)
    network.set_network_enabled(online)


def test_transform_boundary_has_no_reference_correction_or_fit_arguments():
    assert list(inspect.signature(physical.transform_source_heights).parameters) == [
        "longitude_deg", "latitude_deg", "refh_m", "geoid_m"]
    tree = ast.parse(inspect.getsource(physical))
    assert not hasattr(physical, "empirical_alignment")
    # No candidate builder, fitting, legacy result loader, or hidden correction option.
    forbidden = {"empirical_alignment", "build_height_candidates", "_load_matched_roles", "_surface_query",
                 "polyfit", "curve_fit", "least_squares", "lstsq", "fit", "alignment", "aligned", "offset_m"}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not (names | attrs) & forbidden
    source = inspect.getsource(physical.transform_source_heights)
    assert not any(term in source for term in ("read_parquet", "read_csv", "residual_m", "calibration", "refh_neutat"))
    assert physical.METHODS == ("Native", "EGM2008", "GEOID18")


@pytest.mark.parametrize("index", [0, 1])
def test_real_heights_baseline_geoid_and_single_point(model_environment, index):
    spec = physical.datum.GRANULES[index]
    saved = pd.read_parquet(physical.datum.OUTPUT_ROOT/"vertical_candidate_common_sample.parquet")
    saved = saved.loc[saved.granule_id.eq(spec.granule_id)].sort_values("pulse_index").iloc[::128]
    ids = saved.pulse_index.to_numpy(np.int64)
    with h5py.File(spec.h5_path, "r") as h5:
        args = [h5[k][ids] for k in ("refh_longitude", "refh_latitude", "refh", "geoid")]
    heights, audit = physical.transform_source_heights(*args)
    np.testing.assert_allclose(heights["EGM2008_height_m"], saved.C1_height_m, atol=1e-10, rtol=0)
    np.testing.assert_allclose(heights["GEOID18_height_m"], saved.C7_height_m, atol=1e-10, rtol=0)
    assert audit["frame"]["is_noop"] and not audit["frame"]["is_ballpark"]
    assert not audit["combined_is_ballpark"]
    assert audit["frame"]["source_realization"] is None
    assert audit["frame"]["source_coordinate_epoch"] is None
    assert audit["status"] == "provisional physical transformation"
    assert not audit["strict_closure"]
    np.testing.assert_array_equal(heights["frame_delta_ellipsoidal_height_m"], 0)
    check = physical.grid_point_check(model_environment, args[0][0], args[1][0], args[2][0],
                                     heights["GEOID18_height_m"][0])
    assert abs(check["difference_m"]) < 1e-7
    assert check["N_GEOID18_m"] < 0
    assert check["PROJ_H_m"] > args[2][0]
    source = pd.DataFrame(heights).assign(granule_id=spec.granule_id, pulse_index=ids,
        refh_snr=6., common_support_all_candidates=True)
    before = physical.evaluate_heights(source, saved.c7_reference_height_local_plane_m.to_numpy())
    after = physical.evaluate_heights(source, saved.c7_reference_height_local_plane_m.to_numpy()+10.)
    for method in physical.METHODS:
        np.testing.assert_array_equal(before[f"{method}_height_m"], after[f"{method}_height_m"])
        np.testing.assert_allclose(after[f"{method}_residual_m"], before[f"{method}_residual_m"]-10.)
    # Reference values cannot feed back through evaluation into any physical source output.
    pd.testing.assert_frame_equal(source, pd.DataFrame(heights).assign(granule_id=spec.granule_id,
        pulse_index=ids, refh_snr=6., common_support_all_candidates=True))


@pytest.mark.parametrize("missing", ["source_realization", "source_coordinate_epoch",
                                     "target_coordinate_epoch_achieved", "processing_application_verified"])
def test_missing_physical_evidence_never_verified(missing):
    evidence = dict(source_realization="explicit test evidence", source_coordinate_epoch=2024.,
                    target_coordinate_epoch_achieved=2010., processing_application_verified=True,
                    is_noop=False, is_ballpark=False)
    evidence[missing] = None
    assert physical.physical_status(evidence) == "provisional physical transformation"


@pytest.mark.parametrize("invalid", ["is_noop", "is_ballpark"])
def test_null_or_ballpark_never_verified(invalid):
    evidence = dict(source_realization="explicit test evidence", source_coordinate_epoch=2024.,
                    target_coordinate_epoch_achieved=2010., processing_application_verified=True,
                    is_noop=False, is_ballpark=False)
    evidence[invalid] = True
    assert physical.physical_status(evidence) == "provisional physical transformation"


def test_unknown_corrections_have_no_entry_to_height_function():
    signature = inspect.signature(physical.transform_source_heights)
    with pytest.raises(TypeError):
        signature.bind([0.], [0.], [0.], [0.], refh_neutat_delay_total=[1.6])
    with pytest.raises(TypeError):
        signature.bind([0.], [0.], [0.], [0.], fitted_parameters={"shift": 1.6})


def test_metrics_preserve_signed_bias_tail_and_common_population():
    frame = pd.DataFrame(dict(granule_id=["test"]*4, pulse_index=[1, 2, 3, 4],
                              common_support_all_candidates=[False, True, True, True], refh_snr=[6.]*4))
    for method in physical.METHODS:
        frame[f"{method}_height_m"] = [0., -2., -4., 1000.]
    metrics = physical.summarize(physical.evaluate_heights(frame, np.zeros(4)), population="test")
    assert set(metrics.n) == {3} and set(metrics.coverage) == {.75}
    assert metrics.pulse_sha256.nunique() == 1
    assert metrics.RMSE_m.min() > 500
    np.testing.assert_allclose(metrics.mean_bias_m, (1000-2-4)/3)
    np.testing.assert_allclose(metrics.RMSE_m**2, metrics.mean_bias_m**2+metrics.cRMSE_m**2)


def test_no_empirical_output_paths_in_active_helper():
    source = Path(physical.__file__).read_text(encoding="utf-8")
    for path in ("alignment.csv", "aligned.parquet", "alignment_profile.png"):
        assert path not in source
