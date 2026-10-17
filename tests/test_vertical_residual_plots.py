"""Checks for identity-matched signed H0/H2 visualization inputs and artifacts."""
from __future__ import annotations

import hashlib

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from research.reference.refh_correction_audit import residual_metrics
from research.reference import vertical_residual_plots as plots


GRANULES = ("casals_l1b_20241112T165718_001_02", "casals_l1b_20241118T165718_001_02")
IDS = np.asarray([103, 109, 117, 128, 142], dtype=np.int64)


def _hash(ids):
    return hashlib.sha256(np.sort(np.asarray(ids, dtype="<i8")).tobytes()).hexdigest()


def make_result(tmp_path):
    common_rows, metric_rows, pulse_rows = [], [], []
    for date_index, granule_id in enumerate(GRANULES):
        pulse_sha256 = _hash(IDS)
        common_rows.extend({"granule_id": granule_id, "population": "field", "stratum": "snr_ge_5",
                            "pulse_index": int(pulse_id), "common_pulse_sha256": pulse_sha256}
                           for pulse_id in IDS)
        for hypothesis, residuals, coordinate_base in (
            (plots.H0, np.asarray([-2.4, -2.0, -1.7, -1.2, 0.35]) - 0.02 * date_index, 100.0),
            (plots.H2, np.asarray([-0.9, -0.55, -0.31, 0.1, 1.15]) - 0.08 * date_index, 1000.0),
        ):
            metric_rows.append({"granule_id": granule_id, "hypothesis": hypothesis,
                                "population": "field", "stratum": "snr_ge_5", "n": len(IDS),
                                "pulse_sha256": pulse_sha256, **residual_metrics(residuals)})
            for j, (pulse_id, residual) in enumerate(zip(IDS, residuals)):
                pulse_rows.append({
                    "granule_id": granule_id, "hypothesis": hypothesis, "pulse_index": pulse_id,
                    "population": "field", "research_role": "independent_test", "refh_snr": 6.0 + j,
                    "supported": True, "residual": residual,
                    "query_x_m": coordinate_base + j * 5.0 + date_index * 100.0,
                    "query_y_m": coordinate_base * 2.0 + j * 7.0 + date_index * 100.0,
                    "query_crs_epsg": plots.HORIZONTAL_EPSG,
                })
    return {"common_pulses": pd.DataFrame(common_rows), "metrics": pd.DataFrame(metric_rows),
            "pulse_residuals": pd.DataFrame(pulse_rows), "output": tmp_path}


def test_prepare_uses_exact_common_ids_and_recomputes_saved_metrics(tmp_path):
    result = make_result(tmp_path)
    samples, metadata = plots.prepare_common_h0_h2(result)
    for granule_id in GRANULES:
        h0 = samples[(granule_id, plots.H0)]
        h2 = samples[(granule_id, plots.H2)]
        np.testing.assert_array_equal(h0.pulse_index.to_numpy(), IDS)
        np.testing.assert_array_equal(h2.pulse_index.to_numpy(), IDS)
        assert len(h0) == len(h2) == metadata[granule_id]["metrics"][plots.H0]["n"] == len(IDS)
        assert metadata[granule_id]["pulse_sha256"] == _hash(IDS)
        assert metadata[granule_id]["metrics"][plots.H2]["RMSE_m"] == pytest.approx(
            residual_metrics(h2.residual)["RMSE_m"]
        )


def test_duplicate_common_ids_are_rejected(tmp_path):
    result = make_result(tmp_path)
    result["common_pulses"] = pd.concat([result["common_pulses"], result["common_pulses"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate common pulse IDs"):
        plots.prepare_common_h0_h2(result)


def test_saved_metrics_must_match_plotted_sample(tmp_path):
    result = make_result(tmp_path)
    result["metrics"].loc[0, "RMSE_m"] += 0.01
    with pytest.raises(ValueError, match="does not match saved metrics"):
        plots.prepare_common_h0_h2(result)


@pytest.mark.parametrize("column,value,error", [
    ("residual", np.nan, "Non-finite H0_noop_C7_baseline residuals"),
    ("refh_snr", 4.9, "violate the SNR >= 5 filter"),
    ("research_role", "calibration_offset", "does not contain every common pulse"),
    ("query_crs_epsg", 4326, "Query coordinates must be EPSG:6347"),
])
def test_plot_inputs_reject_invalid_pulse_rows(tmp_path, column, value, error):
    result = make_result(tmp_path)
    result["pulse_residuals"].loc[0, column] = value
    with pytest.raises(ValueError, match=error):
        plots.prepare_common_h0_h2(result)


def test_histogram_edges_are_method_specific_shared_and_cover_every_value(tmp_path):
    samples, _ = plots.prepare_common_h0_h2(make_result(tmp_path))
    fig, edges = plots.plot_histograms(samples)
    try:
        assert set(edges) == set(plots.METHODS)
        for method_index, hypothesis in enumerate(plots.METHODS):
            combined = np.concatenate([samples[(gid, hypothesis)].residual.to_numpy() for gid in GRANULES])
            expected_edges = plots.histogram_edges(combined)
            np.testing.assert_array_equal(edges[hypothesis], expected_edges)
            for granule_id in GRANULES:
                residual = samples[(granule_id, hypothesis)].residual.to_numpy()
                counts, _ = np.histogram(residual, bins=edges[hypothesis])
                assert counts.sum() == len(residual)
                assert edges[hypothesis][0] <= residual.min() <= residual.max() <= edges[hypothesis][-1]
            row_limits = [ax.get_xlim() for ax in fig.axes[2 * method_index:2 * method_index + 2]]
            assert all(lo <= 0.0 <= hi for lo, hi in row_limits)
            np.testing.assert_allclose(row_limits[0], row_limits[1])
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_cdf_uses_raw_signed_residuals_without_centering(tmp_path):
    samples, _ = plots.prepare_common_h0_h2(make_result(tmp_path))
    fig = plots.plot_cdf(samples)
    try:
        expected = np.sort(samples[(GRANULES[0], plots.H0)].residual.to_numpy())
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), expected)
        assert not np.isclose(expected.mean(), 0.0)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_spatial_map_uses_h2_query_coordinates(tmp_path):
    samples, _ = plots.prepare_common_h0_h2(make_result(tmp_path))
    fig, color_range = plots.plot_h2_spatial(samples)
    try:
        h2 = samples[(GRANULES[0], plots.H2)]
        h0 = samples[(GRANULES[0], plots.H0)]
        np.testing.assert_array_equal(fig.axes[0].collections[0].get_offsets(),
                                      h2[["query_x_m", "query_y_m"]].to_numpy())
        assert not np.array_equal(h2.query_x_m.to_numpy(), h0.query_x_m.to_numpy())
        assert len(fig.axes[0].get_xticks()) <= 6
        assert color_range[0] < 0 < color_range[1]
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_plot_files_are_created_without_replacing_historical_outputs(tmp_path):
    preserved = {
        "residual_distributions.png": b"baseline-cdf",
        "geoid18_spatial_residuals.png": b"baseline-map",
        "h0_vs_h1_residual_cdf.png": b"historical-cdf-name",
    }
    for name, content in preserved.items():
        (tmp_path / name).write_bytes(content)
    paths = plots.save_vertical_residual_plots(make_result(tmp_path))
    assert {key: path.name for key, path in paths.items()} == plots.OUTPUT_NAMES
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    for name, content in preserved.items():
        assert (tmp_path / name).read_bytes() == content


def test_plot_saver_refuses_to_replace_a_current_plot(tmp_path):
    result = make_result(tmp_path)
    target = tmp_path / plots.OUTPUT_NAMES["cdf"]
    target.write_bytes(b"protected")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        plots.save_vertical_residual_plots(result)
    assert target.read_bytes() == b"protected"
