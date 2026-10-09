import numpy as np

from casals_l1b.classification import DEFAULT_CONFIG
from casals_l1b.evaluation import (
    align_prediction_to_reference,
    evaluate_classification,
    map_reference_labels_to_baseline_classes,
)


def test_explicit_partial_point_index_alignment_and_metrics():
    prediction = {
        "point_index": np.arange(4),
        "x": np.arange(4, dtype=float),
        "y": np.zeros(4),
        "longitude": np.arange(4, dtype=float),
        "latitude": np.zeros(4),
    }
    reference = {"classification": np.array([2, 18]), "point_index": np.array([1, 3])}
    alignment = align_prediction_to_reference(prediction, reference, DEFAULT_CONFIG)
    np.testing.assert_array_equal(alignment["eval_match_valid"], [0, 1, 0, 1])
    truth = map_reference_labels_to_baseline_classes(alignment["eval_gt_class_raw"])
    np.testing.assert_array_equal(truth, [1, 2, 1, 7])

    metrics = evaluate_classification(
        pred_class_baseline=np.array([1, 2, 7, 1], dtype=np.uint8),
        eval_gt_class=np.array([1, 2, 7, 1], dtype=np.uint8),
        eval_match_valid=np.ones(4, dtype=np.uint8),
        dtm_sample_valid=np.ones(4, dtype=np.uint8),
        reference_transfer_status=None,
        reference_nearest3dep_dist_m=None,
        reference_class_vote_ratio=None,
        config=DEFAULT_CONFIG,
    )
    primary = metrics["primary_metrics"]
    assert primary["accuracy"] == 1.0
    assert primary["macro_f1"] == 1.0
    assert primary["weighted_f1"] == 1.0
    assert metrics["subset_results"]["all_matched"]["n_points"] == 4


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
    assert metrics["evaluation_summary_rows"] == []
    assert metrics["subset_results"]["all_matched"]["status"] == "no_points_after_filtering"
