import json

import h5py
import laspy
import numpy as np

from casals_l1b.classification import DEFAULT_CONFIG
from casals_l1b.classification_cli import classify_refh, build_output_paths


def test_classify_refh_writes_prediction_without_evaluation(tmp_path):
    h5_path = tmp_path / "small_refh.h5"
    n = 24
    with h5py.File(h5_path, "w") as h5:
        h5["refh_longitude"] = -75.0 + np.arange(n) * 0.00002
        h5["refh_latitude"] = 38.0 + np.arange(n) * 0.00002
        h5["refh"] = 100.0 + np.tile(np.linspace(0.0, 3.0, 6), 4)
        h5["refh_amp"] = np.full(n, 500.0)
        h5["refh_snr"] = np.full(n, 6.0)
        h5["good_snr"] = np.ones(n, dtype=np.uint8)
        h5["track_num"] = np.tile(np.arange(6, dtype=np.uint16), 4)
        h5["sweep_num"] = np.repeat(np.arange(4, dtype=np.uint32), 6)

    config = dict(DEFAULT_CONFIG)
    config.update(
        OUTPUT_ROOT=tmp_path / "classification",
        CLASSIFIER_MODE="height_only",
        COMPUTE_SCANLINE_FEATURES=False,
        USE_SCANLINE_OUTLIER_NOISE=False,
        WRITE_DIAGNOSTIC_PNG=False,
        WRITE_ERROR_FEATURE_SUMMARY=False,
    )

    result = classify_refh(h5_path, config)

    assert result["status"] == "success"
    output_paths = build_output_paths(config["OUTPUT_ROOT"], h5_path.stem)
    assert output_paths["classified_laz"].is_file()
    metadata = json.loads(output_paths["run_metadata_json"].read_text(encoding="utf-8"))
    assert metadata["evaluation"] == {
        "status": "not_run",
        "reason": "no_reference_provided",
    }
    assert "accuracy" not in metadata["evaluation"]

    las = laspy.read(output_paths["classified_laz"])
    np.testing.assert_array_equal(las.point_index, np.arange(n, dtype=np.uint32))
    assert len(las.classification) == n
    assert len(las.classification_reason) == n
    assert "eval_match_valid" not in las.point_format.extra_dimension_names
