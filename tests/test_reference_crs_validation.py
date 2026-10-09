import numpy as np
import pytest
from pyproj import CRS
from casals_l1b.classification import DEFAULT_CONFIG, classify_points_baseline
from casals_l1b.evaluation import align_prediction_to_reference, evaluate_classification
from research.reference import diagnose_3dep_offsets as crs
from research.reference.transfer_3dep_labels_to_casals import CONFIG, transfer_labels_to_casals


def test_horizontal_header_does_not_define_vertical():
    _, vertical, frame = crs.resolve_reference_frame(CRS.from_epsg(6347))
    assert vertical is None
    assert frame.reference_frame_status == 'vertical_crs_unknown'
    with pytest.raises(ValueError, match='vertical_crs_unknown'):
        crs.transform_verified_xyz([[0, 0, 0]], 4979, 6347, source_verified=True, target_verified=True)


def test_complete_xyz_operation_and_missing_required_grid(monkeypatch):
    xyz = np.array([[-75., 38., 100.]])
    output, audit = crs.transform_verified_xyz(xyz, 4979, 4978, source_verified=True, target_verified=True)
    assert np.linalg.norm(output[0]) > 6e6
    assert abs(audit['z_change_median_m']) > 1e6
    assert audit['pipeline']
    with pytest.raises(ValueError, match='verification required'):
        crs.transform_verified_xyz(xyz, 4979, 4978, source_verified=False, target_verified=True)
    class Missing:
        best_available = False
        transformers = []
        unavailable_operations = []
    monkeypatch.setattr(crs, 'build_transformer_group', lambda *args: Missing())
    with pytest.raises(RuntimeError, match='missing grids'):
        crs.transform_verified_xyz(xyz, 4979, 'EPSG:6347+5703', source_verified=True, target_verified=True)


def test_far_ambiguous_and_nonfinite_transfer():
    config = {**CONFIG, 'label_knn': 2, 'label_min_neighbors': 2}
    result = transfer_labels_to_casals(np.array([[100, 0, 0], [0, 0, 0], [np.nan, 0, 0]]),
                                     np.array([[0, 0, 0], [.1, 0, 0]]), np.array([2, 6]), config)
    assert result['transfer_status'].tolist() == [0, 3, 4]
    assert result['classification'].tolist() == [1, 1, 7]


def test_invalid_reference_population_excluded():
    metrics = evaluate_classification(np.array([2, 1, 7, 7, 7]), np.array([2, 1, 7, 7, 7]),
              np.ones(5), np.ones(5), np.array([1, 2, 0, 3, 4]), None, None, DEFAULT_CONFIG)
    assert metrics['subset_results']['strict']['n_points'] == 1
    assert metrics['subset_results']['strict_plus_weak']['n_points'] == 2
    assert metrics['subset_results']['strict']['support_7'] == 0
    assert metrics['evaluation_summary_rows'][0]['coverage'] == .2


@pytest.mark.parametrize('mismatch', ['source_h5', 'longitude', 'refh_original_m'])
def test_same_index_wrong_identity_rejected(mismatch):
    prediction = dict(source_h5='a.h5', point_index=np.array([10]), longitude=np.array([-75.]),
                      latitude=np.array([38.]), refh_original_m=np.array([100.]))
    reference = {**prediction, 'classification': np.array([2])}
    reference[mismatch] = 'b.h5' if mismatch == 'source_h5' else np.array([0.])
    with pytest.raises(ValueError, match='provenance|identity'):
        align_prediction_to_reference(prediction, reference, DEFAULT_CONFIG)


def test_missing_dtm_alone_is_unclassified():
    result = classify_points_baseline(np.array([100.]), np.array([np.nan]), np.array([0]),
               np.array([1.]), {**DEFAULT_CONFIG, 'CLASSIFIER_MODE': 'height_only'})
    assert result['pred_class_baseline'].tolist() == [1]
    assert result['classification_reason'].tolist() == [2]


def test_download_pipeline_preserves_z_without_duplicate_output_dimension(tmp_path):
    import json
    from dataclasses import fields
    from tools.download_3dep_lpc import Config, ClipPlan, build_clip_pipeline
    values = {f.name: "" for f in fields(ClipPlan)}
    values.update(clip_crs="EPSG:6347", clip_polygon_wkt="POLYGON ((0 0, 1 0, 1 1, 0 0))",
                  ept_url="https://example.invalid/ept.json", ept_srs_user_input="EPSG:3857",
                  output_crs_user_input="EPSG:6347", horizontal_reprojection_applied=True,
                  ept_schema_json=json.dumps([{"name":"X","type":"signed","size":4},
                                             {"name":"Extra","type":"floating","size":8}]))
    pipeline = build_clip_pipeline(Config(tmp_path/'input.h5', tmp_path, tmp_path), ClipPlan(**values), tmp_path/'test.laz')
    assert pipeline[1]['dimensions'] == 'Z=>CASALSOriginalZ'
    assert pipeline[3]['dimensions'] == 'CASALSOriginalZ=>Z'
    assert len(CRS(pipeline[2]['in_srs']).axis_info) == 2
    assert len(CRS(pipeline[2]['out_srs']).axis_info) == 2
    assert pipeline[-1]['extra_dims'] == 'Extra=double'


def test_known_vertical_units_xyz_conversion():
    output, audit = crs.transform_verified_xyz([[500000, 4250000, 100]], 'EPSG:6347+5703', 'EPSG:6347+6360', source_verified=True, target_verified=True)
    np.testing.assert_allclose(output[0], [500000, 4250000, 328.08333333333], atol=1e-7)
    assert audit['selected_grids'] == []
