import numpy as np

from casals_l1b.classification import DEFAULT_CONFIG
from casals_l1b.evaluation import (
    align_prediction_to_reference,
    evaluate_classification,
    map_reference_labels_to_baseline_classes,
)


def test_explicit_partial_point_index_alignment_and_metrics():
    prediction = {
        "source_h5": "example.h5",
        "refh_original_m": np.arange(4, dtype=float),
        "point_index": np.arange(4),
        "x": np.arange(4, dtype=float),
        "y": np.zeros(4),
        "longitude": np.arange(4, dtype=float),
        "latitude": np.zeros(4),
    }
    reference = {"classification": np.array([2, 18]), "point_index": np.array([1, 3])}
    reference.update(source_h5="example.h5", longitude=np.array([1., 3.]), latitude=np.zeros(2), refh_original_m=np.array([1., 3.]))
    alignment = align_prediction_to_reference(prediction, reference, DEFAULT_CONFIG)
    np.testing.assert_array_equal(alignment["eval_match_valid"], [0, 1, 0, 1])
    truth = map_reference_labels_to_baseline_classes(alignment["eval_gt_class_raw"])
    np.testing.assert_array_equal(truth, [1, 2, 1, 7])

    metrics = evaluate_classification(
        pred_class_baseline=np.array([1, 2, 7, 1], dtype=np.uint8),
        eval_gt_class=np.array([1, 2, 7, 1], dtype=np.uint8),
        eval_match_valid=np.ones(4, dtype=np.uint8),
        dtm_sample_valid=np.ones(4, dtype=np.uint8),
        reference_transfer_status=np.ones(4, dtype=np.uint8),
        reference_nearest3dep_dist_m=None,
        reference_class_vote_ratio=None,
        config=DEFAULT_CONFIG,
    )
    primary = metrics["primary_metrics"]
    assert primary["accuracy"] == 1.0
    assert primary["macro_f1"] == 1.0
    assert primary["weighted_f1"] == 1.0
    assert metrics["subset_results"]["strict"]["n_points"] == 4


def test_no_reference_population_has_no_metrics():
    metrics = evaluate_classification(
        pred_class_baseline=np.array([1, 2], dtype=np.uint8),
        eval_gt_class=np.zeros(2, dtype=np.uint8),
        eval_match_valid=np.zeros(2, dtype=np.uint8),
        dtm_sample_valid=np.ones(2, dtype=np.uint8),
        reference_transfer_status=None,
        reference_nearest3dep_dist_m=None,
        reference_class_vote_ratio=None,
        config=DEFAULT_CONFIG,
    )
    assert metrics["primary_metrics"] == {}
    assert len(metrics["evaluation_summary_rows"]) == 2
    assert metrics["subset_results"]["strict"]["status"] == "evaluation unavailable"


def test_strict_and_strict_plus_weak_exclude_other_transfer_statuses():
    config = {
        **DEFAULT_CONFIG,
        "EVAL_REQUIRE_TRANSFER_STATUS": [1, 2],
        "EVAL_REQUIRE_VALID_DTM": False,
        "EVAL_IGNORE_REFERENCE_NOISE": False,
    }
    status = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    result = evaluate_classification(
        pred_class_baseline=np.array([7, 1, 1, 2, 7], dtype=np.uint8),
        eval_gt_class=np.array([7, 1, 2, 2, 7], dtype=np.uint8),
        eval_match_valid=np.ones(5, dtype=bool),
        dtm_sample_valid=np.ones(5, dtype=bool),
        reference_transfer_status=status,
        reference_nearest3dep_dist_m=np.ones(5),
        reference_class_vote_ratio=np.ones(5),
        config=config,
    )

    strict = result["subset_results"]["strict"]
    strict_weak = result["subset_results"]["strict_plus_weak"]
    assert strict["n_points"] == 1
    assert strict_weak["n_points"] == 2
    assert np.asarray(strict["confusion_matrix"]).sum() == strict["n_points"]
    assert np.asarray(strict_weak["confusion_matrix"]).sum() == strict_weak["n_points"]
