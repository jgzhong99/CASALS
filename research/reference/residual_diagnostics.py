"""Same-pulse bias and dispersion diagnostics for the frozen road test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.reference.final_bias_validation import range_effect_from_delay
from research.reference.refh_correction_audit import (
    add_sensitivity,
    height_sensitivity,
    residual_metrics,
    select_independent_common_test,
)


GRANULES = (
    "casals_l1b_20241112T165718_001_02",
    "casals_l1b_20241118T171757_001_02",
)
STRATA = ("all_common_support", "snr_ge_3", "snr_ge_5")
SCENARIOS = (
    "C7",
    "C1",
    "C3",
    "H1_stored_as_m",
    "H1_ERA5_profile",
    "H1_stored_as_ns_two_way",
    "H1_stored_as_ns_one_way",
)
SCENARIO_LABELS = {
    "C7": "C7 baseline",
    "C1": "C1",
    "C3": "C3",
    "H1_stored_as_m": "H1 stored value as m",
    "H1_ERA5_profile": "H1 ERA5 profile",
    "H1_stored_as_ns_two_way": "H1 ns, two-way",
    "H1_stored_as_ns_one_way": "H1 ns, one-way",
}
H5_FIELDS = (
    "refh_neutat_delay_total",
    "instrument_longitude",
    "instrument_latitude",
    "local_beam_azimuth",
    "local_beam_elevation",
    "refh_snr",
    "track_num",
    "sweep_num",
)
EXPECTED_N = {
    GRANULES[0]: {"all_common_support": 147, "snr_ge_3": 54, "snr_ge_5": 14},
    GRANULES[1]: {"all_common_support": 165, "snr_ge_3": 138, "snr_ge_5": 70},
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pulse_id_sha256(pulse_ids: Any) -> str:
    ids = np.asarray(pulse_ids, dtype="<i8")
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("pulse IDs must be unique one-dimensional identifiers")
    return hashlib.sha256(ids.tobytes()).hexdigest()


def stratum_masks(snr: Any) -> dict[str, np.ndarray]:
    values = np.asarray(snr, dtype=float)
    if values.ndim != 1:
        raise ValueError("SNR values must be one-dimensional")
    return {
        "all_common_support": np.ones(values.size, dtype=bool),
        "snr_ge_3": np.isfinite(values) & (values >= 3.0),
        "snr_ge_5": np.isfinite(values) & (values >= 5.0),
    }


def confidence_interval_status(independent_units: int) -> str:
    if independent_units < 3:
        return "NOT_ESTIMATED: fewer than 3 independent sites"
    return "NOT_ESTIMATED: descriptive analysis; no interval procedure specified"


def _read_h5_fields(path: Path, pulse_ids: np.ndarray) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        for name in H5_FIELDS:
            if name not in h5:
                raise KeyError(f"H5 field not found: {name}")
            dataset = h5[name]
            values = np.asarray(dataset[pulse_ids], dtype=float).reshape(-1)
            for attr in ("_FillValue", "missing_value"):
                if attr in dataset.attrs:
                    values[np.isin(values, np.asarray(dataset.attrs[attr], dtype=float).ravel())] = np.nan
            values = values * float(dataset.attrs.get("scale_factor", 1.0))
            values = values + float(dataset.attrs.get("add_offset", 0.0))
            output[name] = values
    return output


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    earth_radius_m = 6_371_008.8
    lon1, lat1, lon2, lat2 = np.deg2rad([lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * earth_radius_m * np.arcsin(np.sqrt(a)))


def _selected_pulse_distance(frame: pd.DataFrame) -> np.ndarray:
    distance = pd.Series(0.0, index=frame.index, dtype=float)
    for _, group in frame.groupby("track_num", sort=False):
        ordered = group.sort_values(["sweep_num", "pulse_index"])
        previous: tuple[float, float] | None = None
        cumulative = 0.0
        for index, row in ordered.iterrows():
            current = (float(row.longitude_deg), float(row.latitude_deg))
            if previous is not None:
                cumulative += _haversine_m(*previous, *current)
            distance.loc[index] = cumulative
            previous = current
    return distance.to_numpy(float)


def _metric_row(
    frame: pd.DataFrame,
    scenario: str,
    stratum: str,
    values: np.ndarray,
    *,
    status: str,
) -> dict[str, Any]:
    metrics = residual_metrics(values)
    if metrics["n_nonfinite"] or metrics["n"] != len(frame):
        raise ValueError(f"{scenario}/{stratum} has missing values on frozen support")
    ids = frame.pulse_index.to_numpy(dtype="<i8")
    return {
        "granule_id": str(frame.granule_id.iloc[0]),
        "stratum": stratum,
        "hypothesis": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "candidate_status": status,
        "residual_semantics": (
            "saved candidate residual" if scenario in {"C7", "C1", "C3"}
            else "C7 residual plus conditional sensitivity; not corrected Refh"
        ),
        "same_pulse_ids_across_scenarios": True,
        "pulse_id_sha256": pulse_id_sha256(ids),
        "confidence_interval_status": confidence_interval_status(1),
        **metrics,
    }


def _scenario_status(scenario: str) -> str:
    if scenario == "C7":
        return "PROVISIONAL_NOT_DATUM_VERIFIED"
    if scenario in {"C1", "C3"}:
        return "NUMERICAL_COMPARISON_ONLY"
    if scenario == "H1_ERA5_profile":
        return "CONDITIONAL_SENSITIVITY_ONLY; ERA5 is an independent model profile"
    return "PHYSICAL_SENSITIVITY_ONLY; H5 units and applied status unknown"


def _make_bias_decomposition(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "granule_id", "stratum", "hypothesis", "scenario_label", "candidate_status",
        "n", "pulse_id_sha256", "mean_bias_m", "cRMSE_m", "RMSE_m",
        "bias_squared_m2", "dispersion_squared_m2", "MSE_m2",
        "bias_squared_fraction_of_MSE", "dispersion_squared_fraction_of_MSE",
        "decomposition_identity_status", "interpretation",
    ]
    rows = []
    for row in metrics.itertuples(index=False):
        bias2 = row.mean_bias_m ** 2
        dispersion2 = row.cRMSE_m ** 2
        mse = row.RMSE_m ** 2
        exact = np.isclose(bias2 + dispersion2, mse, rtol=1e-12, atol=1e-12)
        rows.append({
            "granule_id": row.granule_id,
            "stratum": row.stratum,
            "hypothesis": row.hypothesis,
            "scenario_label": row.scenario_label,
            "candidate_status": row.candidate_status,
            "n": row.n,
            "pulse_id_sha256": row.pulse_id_sha256,
            "mean_bias_m": row.mean_bias_m,
            "cRMSE_m": row.cRMSE_m,
            "RMSE_m": row.RMSE_m,
            "bias_squared_m2": bias2,
            "dispersion_squared_m2": dispersion2,
            "MSE_m2": mse,
            "bias_squared_fraction_of_MSE": row.bias_squared_fraction_of_MSE,
            "dispersion_squared_fraction_of_MSE": row.dispersion_squared_fraction_of_MSE,
            "decomposition_identity_status": "PASS" if exact else "FAIL",
            "interpretation": "Squared-error decomposition; fractions are not causal error-source shares.",
        })
    return pd.DataFrame(rows, columns=columns)


def _pulse_distribution_rows(
    frame: pd.DataFrame,
    stratum: str,
    scenario_values: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario, values in scenario_values.items():
        mean = float(np.mean(values))
        centered = values - mean
        centered_square = centered ** 2
        total_square = float(centered_square.sum())
        for position, (_, pulse) in enumerate(frame.iterrows()):
            rows.append({
                "granule_id": pulse.granule_id,
                "stratum": stratum,
                "hypothesis": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "pulse_index": int(pulse.pulse_index),
                "track_num": pulse.track_num,
                "sweep_num": pulse.sweep_num,
                "longitude_deg": pulse.longitude_deg,
                "latitude_deg": pulse.latitude_deg,
                "refh_snr": pulse.refh_snr,
                "reference_height_local_plane_m": pulse.reference_height_local_plane_m,
                "local_slope_deg": pulse.c7_local_slope_deg,
                "road_axis_distance_m": pulse.axis_distance_m,
                "residual_m": float(values[position]),
                "mean_centered_residual_m": float(centered[position]),
                "absolute_residual_m": float(abs(values[position])),
                "centered_squared_error_m2": float(centered_square[position]),
                "centered_squared_error_share": (
                    float(centered_square[position] / total_square) if total_square > 0 else np.nan
                ),
                "pulse_id_sha256": pulse_id_sha256(frame.pulse_index.to_numpy(dtype="<i8")),
                "centering_note": "Descriptive mean-centering only; original residual is unchanged.",
            })
    return rows


def _spatial_summary_rows(
    frame: pd.DataFrame,
    stratum: str,
    scenario_values: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    track = pd.to_numeric(frame.track_num, errors="coerce")
    axis = pd.to_numeric(frame.axis_distance_m, errors="coerce")
    lo, hi = float(axis.min()), float(axis.max())
    edges = np.linspace(lo, hi + np.finfo(float).eps * max(1.0, abs(hi)), 5)
    block = pd.cut(axis, bins=edges, labels=["road_axis_1", "road_axis_2", "road_axis_3", "road_axis_4"],
                   include_lowest=True, duplicates="drop")
    predictors = {
        "selected_pulse_cumulative_distance_within_track_m": _selected_pulse_distance(frame),
        "reference_elevation_m": pd.to_numeric(frame.reference_height_local_plane_m, errors="coerce").to_numpy(float),
        "local_slope_deg": pd.to_numeric(frame.c7_local_slope_deg, errors="coerce").to_numpy(float),
        "refh_snr": pd.to_numeric(frame.refh_snr, errors="coerce").to_numpy(float),
        "road_axis_distance_m": axis.to_numpy(float),
    }

    for scenario, values in scenario_values.items():
        for group_type, group_ids, min_n in (
            ("track", track, 3),
            ("road_axis_block", block.astype(str), 5),
        ):
            for group_id in pd.unique(group_ids):
                mask = (group_ids == group_id) if group_type == "track" else (group_ids.astype(str) == str(group_id))
                n = int(mask.sum())
                if not n:
                    continue
                group_values = values[mask]
                metrics = residual_metrics(group_values)
                sufficient = n >= min_n
                rows.append({
                    "granule_id": frame.granule_id.iloc[0], "stratum": stratum,
                    "hypothesis": scenario, "scenario_label": SCENARIO_LABELS[scenario],
                    "spatial_unit_type": group_type, "spatial_unit_id": str(group_id),
                    "n": n, "mean_bias_m": metrics["mean_bias_m"],
                    "median_bias_m": metrics["median_bias_m"],
                    "RMSE_m": metrics["RMSE_m"] if sufficient else np.nan,
                    "cRMSE_m": metrics["cRMSE_m"] if sufficient else np.nan,
                    "NMAD_m": metrics["NMAD_m"] if sufficient else np.nan,
                    "cRMSE_NMAD_ratio": metrics["cRMSE_NMAD_ratio"] if sufficient else np.nan,
                    "dispersion_status": "REPORTED" if sufficient else f"WITHHELD: n<{min_n}",
                    "confidence_interval_status": confidence_interval_status(1),
                    "grouping_note": (
                        "Existing track IDs; rows are clustered within one reviewed road site."
                        if group_type == "track" else
                        "Four equal-width bins along the reviewed road axis; not aircraft along-track distance."
                    ),
                })

        for predictor_name, predictor in predictors.items():
            finite = np.isfinite(values) & np.isfinite(predictor)
            x, y = predictor[finite], values[finite]
            if len(x) < 5:
                rho, status = np.nan, "WITHHELD: fewer than 5 finite pairs"
            elif np.unique(x).size < 2 or np.unique(y).size < 2:
                rho, status = np.nan, "WITHHELD: no rank variation"
            else:
                rho = float(spearmanr(x, y).statistic)
                status = "DESCRIPTIVE_ONLY; clustered pulses, no significance claim"
            rows.append({
                "granule_id": frame.granule_id.iloc[0], "stratum": stratum,
                "hypothesis": scenario, "scenario_label": SCENARIO_LABELS[scenario],
                "spatial_unit_type": "predictor_relationship", "spatial_unit_id": predictor_name,
                "n": int(len(x)), "n_excluded_missing_pair": int(len(frame) - len(x)),
                "spearman_rho": rho, "dispersion_status": status,
                "confidence_interval_status": confidence_interval_status(1),
                "grouping_note": (
                    "Selected-pulse cumulative geodesic distance within each H5 track, ordered by sweep and pulse; "
                    "sparse selected points make this a descriptive path proxy."
                    if predictor_name == "selected_pulse_cumulative_distance_within_track_m" else
                    "Descriptive association across selected pulses; track/site clustering is retained in interpretation."
                ),
            })
    return rows


def _leave_one_out_delta(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full = residual_metrics(values)
    d_crmse = np.empty(values.size, dtype=float)
    d_rmse = np.empty(values.size, dtype=float)
    for index in range(values.size):
        loo = residual_metrics(np.delete(values, index))
        d_crmse[index] = loo["cRMSE_m"] - full["cRMSE_m"]
        d_rmse[index] = loo["RMSE_m"] - full["RMSE_m"]
    return d_crmse, d_rmse


def _high_influence_table(frame: pd.DataFrame, scenario_values: dict[str, np.ndarray]) -> pd.DataFrame:
    if len(frame) != 14:
        raise ValueError(f"November 12 SNR>=5 diagnostic requires 14 frozen pulses, got {len(frame)}")
    c7 = scenario_values["C7"]
    h1 = scenario_values["H1_stored_as_m"]
    c7_centered = c7 - c7.mean()
    h1_centered = h1 - h1.mean()
    c7_square, h1_square = c7_centered ** 2, h1_centered ** 2
    loo_c7_crmse, loo_c7_rmse = _leave_one_out_delta(c7)
    loo_h1_crmse, loo_h1_rmse = _leave_one_out_delta(h1)
    result = frame[[
        "granule_id", "pulse_index", "track_num", "sweep_num", "longitude_deg", "latitude_deg",
        "refh_snr", "reference_height_local_plane_m", "surface_acceptable",
        "c7_surface_acceptable", "c7_plane_fit_valid", "c7_neighbor_count",
        "c7_nearest_ground_distance_m", "c7_local_slope_deg", "c7_plane_residual_nmad_m",
        "c7_ground_roughness_nmad_m", "axis_distance_m",
    ]].copy()
    result["C7_residual_m"] = c7
    result["H1_stored_as_m_residual_m"] = h1
    result["H1_ERA5_profile_residual_m"] = scenario_values["H1_ERA5_profile"]
    result["C7_mean_centered_residual_m"] = c7_centered
    result["H1_stored_as_m_mean_centered_residual_m"] = h1_centered
    result["stored_delay_numeric"] = frame["stored_delay_numeric"].to_numpy(float)
    result["predicted_height_effect_if_delay_is_m"] = frame["height_effect_if_m"].to_numpy(float)
    result["predicted_ERA5_height_effect_m"] = frame["height_effect_ERA5_profile"].to_numpy(float)
    result["C7_centered_squared_error_share"] = c7_square / c7_square.sum()
    result["H1_centered_squared_error_share"] = h1_square / h1_square.sum()
    result["C7_absolute_residual_rank"] = pd.Series(np.abs(c7)).rank(ascending=False, method="min").astype(int)
    result["C7_signed_residual_rank_low_to_high"] = pd.Series(c7).rank(ascending=True, method="min").astype(int)
    result["LOO_delta_C7_cRMSE_m"] = loo_c7_crmse
    result["LOO_delta_C7_RMSE_m"] = loo_c7_rmse
    result["LOO_delta_H1_cRMSE_m"] = loo_h1_crmse
    result["LOO_delta_H1_RMSE_m"] = loo_h1_rmse
    result["deletion_performed"] = False
    result["influence_interpretation"] = "Ranking and leave-one-out diagnostics only; all 14 observations remain included."
    return result.sort_values("C7_centered_squared_error_share", ascending=False).reset_index(drop=True)


def _plot_outputs(metrics: pd.DataFrame, distributions: pd.DataFrame, spatial: pd.DataFrame,
                  influence: pd.DataFrame, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    scenarios = list(SCENARIOS)
    palette = dict(zip(scenarios, plt.get_cmap("tab10").colors[:len(scenarios)]))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    metric_specs = (("mean_bias_m", "Mean bias (m)", False),
                    ("RMSE_m", "RMSE (m; log scale)", True),
                    ("NMAD_m", "NMAD (m; log scale)", True))
    for row, granule in enumerate(GRANULES):
        date = "2024-11-12" if "20241112" in granule else "2024-11-18"
        for col, (metric, title, log_scale) in enumerate(metric_specs):
            ax = axes[row, col]
            subset = metrics.loc[metrics.granule_id.eq(granule)]
            for scenario in scenarios:
                values = []
                for stratum in STRATA:
                    match = subset.loc[subset.stratum.eq(stratum) & subset.hypothesis.eq(scenario), metric]
                    values.append(float(match.iloc[0]))
                ax.plot(range(3), values, marker="o", linewidth=1.5, label=SCENARIO_LABELS[scenario],
                        color=palette[scenario])
            ax.set_xticks(range(3), ["All", "SNR ≥ 3", "SNR ≥ 5"])
            ax.set_title(f"{date}: {title}")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.25)
            if log_scale:
                ax.set_yscale("log")
            ax.axhline(0.0, color="black", linewidth=0.6) if not log_scale else None
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Same-pulse residual metrics by fixed support stratum", fontsize=15)
    path = output / "bias_rmse_nmad_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False, constrained_layout=True)
    for ax, granule in zip(axes, GRANULES):
        date = "2024-11-12" if "20241112" in granule else "2024-11-18"
        rows = distributions.loc[distributions.granule_id.eq(granule)
                                 & distributions.stratum.eq("snr_ge_5")]
        data = [rows.loc[rows.hypothesis.eq(s), "mean_centered_residual_m"].to_numpy(float)
                for s in scenarios]
        ax.boxplot(data, tick_labels=[SCENARIO_LABELS[s] for s in scenarios], showfliers=False,
                   patch_artist=True, medianprops={"color": "black"})
        for x, vals in enumerate(data, start=1):
            jitter = np.linspace(-0.10, 0.10, len(vals)) if len(vals) > 1 else np.zeros(len(vals))
            ax.scatter(x + jitter, vals, s=13, alpha=0.32, color=palette[scenarios[x - 1]], zorder=3)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{date}: SNR ≥ 5; n={int(rows.loc[rows.hypothesis.eq('C7'), 'pulse_index'].nunique())}")
        ax.set_ylabel("Mean-centered residual (m)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Mean-centering is descriptive; full residual samples are shown", fontsize=14)
    path = output / "centered_residual_distributions.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    top = influence.sort_values("C7_centered_squared_error_share", ascending=False)
    axes[0].bar(top.pulse_index.astype(str), top.C7_centered_squared_error_share * 100,
                color="#3975a5", label="C7 centered squared-error share")
    axes[0].plot(top.pulse_index.astype(str), top.H1_centered_squared_error_share * 100,
                 marker="o", color="#d47b35", label="H1 meter sensitivity share")
    axes[0].set_ylabel("Share of centered squared error (%)")
    axes[0].set_title("2024-11-12 SNR ≥ 5: all 14 pulses ranked by influence")
    axes[0].tick_params(axis="x", rotation=50, labelsize=8)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    high = spatial.loc[
        spatial.granule_id.eq(GRANULES[1])
        & spatial.stratum.eq("snr_ge_5")
        & spatial.spatial_unit_type.eq("predictor_relationship")
    ]
    dist = distributions.loc[distributions.granule_id.eq(GRANULES[1])
                             & distributions.stratum.eq("snr_ge_5")]
    for scenario in ("C7", "H1_stored_as_m"):
        vals = dist.loc[dist.hypothesis.eq(scenario)]
        axes[1].scatter(vals.road_axis_distance_m, vals.residual_m, s=20, alpha=0.65,
                         label=SCENARIO_LABELS[scenario], color=palette[scenario])
    axes[1].set_xlabel("Reviewed road-axis distance (m)")
    axes[1].set_ylabel("Original / sensitivity residual (m)")
    axes[1].set_title("2024-11-18 SNR ≥ 5: residuals along the road axis")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.suptitle("Influence and spatial profile; road-axis distance is not aircraft along-track distance")
    path = output / "residual_influence_and_spatial_profile.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    nov18 = spatial.loc[
        spatial.granule_id.eq(GRANULES[1]) & spatial.stratum.eq("snr_ge_5")
        & spatial.spatial_unit_type.eq("track")
    ]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    for scenario in ("C7", "H1_stored_as_m"):
        rows = nov18.loc[nov18.hypothesis.eq(scenario)].copy()
        rows["track_order"] = pd.to_numeric(rows.spatial_unit_id, errors="coerce")
        rows = rows.sort_values("track_order")
        axes[0].scatter(rows.track_order, rows.median_bias_m, s=20, alpha=0.7,
                        label=SCENARIO_LABELS[scenario], color=palette[scenario])
    axes[0].set_ylabel("Per-track median residual (m)")
    axes[0].set_title("2024-11-18 SNR ≥ 5: track medians (descriptive; singleton tracks retained)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    adequate = nov18.loc[nov18.n.ge(3)]
    for scenario in ("C7", "H1_stored_as_m"):
        rows = adequate.loc[adequate.hypothesis.eq(scenario)].copy()
        rows["track_order"] = pd.to_numeric(rows.spatial_unit_id, errors="coerce")
        rows = rows.sort_values("track_order")
        if len(rows):
            axes[1].scatter(rows.track_order, rows.cRMSE_NMAD_ratio, s=35,
                            label=SCENARIO_LABELS[scenario], color=palette[scenario])
    if adequate.empty:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No track has n ≥ 3; track dispersion ratios are withheld.",
                     ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_ylabel("Per-track cRMSE / NMAD")
    axes[1].set_xlabel("H5 track number")
    axes[1].set_title("Dispersion is reported only for tracks with at least three selected pulses")
    axes[1].grid(alpha=0.25)
    if not adequate.empty:
        axes[1].legend(frameon=False)
    fig.suptitle("Per-track comparison; tracks are nested within one site per date")
    path = output / "per_track_residual_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def build_residual_diagnostics(project_root: str | Path,
                               output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_root).resolve() if output_root else (
        root / "outputs/notebooks/10_3dep_elevation_validation/residual_diagnostics"
    )
    population_path = root / "outputs/notebooks/10_3dep_elevation_validation/ground_aware/ground_target_test_population.parquet"
    saved_residual_path = root / "outputs/notebooks/10_3dep_elevation_validation/final_bias_attribution/physical_hypothesis_residuals.parquet"
    legacy_path = root / "outputs/notebooks/10_3dep_elevation_validation/correction_audit/physical_hypothesis_comparison.csv"
    source_manifest_path = root / "outputs/notebooks/10_3dep_elevation_validation/visual_revision/input_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    population = pd.read_parquet(population_path)
    selected = select_independent_common_test(population)
    if not selected.is_ground_target_eligible.fillna(False).astype(bool).all():
        raise ValueError("common-support selection unexpectedly includes non-eligible road targets")
    if set(selected.granule_id.unique()) != set(GRANULES) or len(selected) != 312:
        raise ValueError("frozen common independent-test population differs from the saved 312-pulse baseline")
    saved_residuals = pd.read_parquet(saved_residual_path)
    legacy = pd.read_csv(legacy_path)
    output.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    high_influence: pd.DataFrame | None = None
    parity_rows = []
    granule_effects: dict[str, float] = {}

    for granule in GRANULES:
        base = selected.loc[selected.granule_id.eq(granule)].sort_values("pulse_index").reset_index(drop=True).copy()
        ids = base.pulse_index.to_numpy(dtype=np.int64)
        h5_path = root / "data/raw/casals_l1b" / f"{granule}.h5"
        manifest_h5 = source_manifest["inputs"][granule]
        if Path(root / manifest_h5["h5_path"]).resolve() != h5_path.resolve():
            raise ValueError(f"H5 source path differs from the verified visual-revision manifest for {granule}")
        if h5_path.stat().st_size != int(manifest_h5["h5_size_bytes"]):
            raise ValueError(f"H5 source size differs from the verified visual-revision manifest for {granule}")
        fields = _read_h5_fields(h5_path, ids)
        snr = pd.to_numeric(base.refh_snr, errors="coerce").to_numpy(float)
        if not np.allclose(fields["refh_snr"], snr, rtol=0, atol=1e-7, equal_nan=True):
            raise ValueError(f"H5 SNR no longer matches the frozen population for {granule}")
        for field, column in (("track_num", "track_num"), ("sweep_num", "sweep_num")):
            if not np.array_equal(fields[field], pd.to_numeric(base[column], errors="coerce").to_numpy(float)):
                raise ValueError(f"H5 {field} no longer matches the frozen population for {granule}")

        effect_m = height_sensitivity(
            fields["refh_neutat_delay_total"], fields["instrument_longitude"],
            fields["instrument_latitude"], fields["local_beam_azimuth"], fields["local_beam_elevation"],
        )
        effect_ns_two_way = height_sensitivity(
            range_effect_from_delay(fields["refh_neutat_delay_total"], "ns_two_way_time"),
            fields["instrument_longitude"], fields["instrument_latitude"],
            fields["local_beam_azimuth"], fields["local_beam_elevation"],
        )
        effect_ns_one_way = height_sensitivity(
            range_effect_from_delay(fields["refh_neutat_delay_total"], "ns_one_way_time"),
            fields["instrument_longitude"], fields["instrument_latitude"],
            fields["local_beam_azimuth"], fields["local_beam_elevation"],
        )
        if not (np.isfinite(effect_m).all() and np.isfinite(effect_ns_two_way).all()
                and np.isfinite(effect_ns_one_way).all()):
            raise ValueError(f"non-finite physical sensitivity on frozen support for {granule}")

        saved = saved_residuals.loc[saved_residuals.granule_id.eq(granule)].sort_values("pulse_index")
        saved_ids = saved.pulse_index.to_numpy(dtype=np.int64)
        expected_high_ids = base.loc[snr >= 5, "pulse_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(saved_ids, expected_high_ids):
            raise ValueError(f"saved final-attribution IDs differ from fixed SNR>=5 IDs for {granule}")
        era5_effects = (saved.H1_era5_profile_residual_m.to_numpy(float)
                        - saved.C7_residual_m.to_numpy(float))
        if not np.isfinite(era5_effects).all() or float(np.ptp(era5_effects)) >= 1e-10:
            raise ValueError(f"saved ERA5 effect is not constant within {granule}")
        era5_effect = float(era5_effects[0])
        granule_effects[granule] = era5_effect
        if not np.allclose(
            height_sensitivity(
                fields["refh_neutat_delay_total"][snr >= 5], fields["instrument_longitude"][snr >= 5],
                fields["instrument_latitude"][snr >= 5], fields["local_beam_azimuth"][snr >= 5],
                fields["local_beam_elevation"][snr >= 5],
            ), saved.height_effect_if_m.to_numpy(float), rtol=0, atol=1e-10,
        ):
            raise ValueError(f"stored-meter effect does not match the frozen prior output for {granule}")
        if not np.allclose(
            base.loc[snr >= 5, "C7_residual_m"].to_numpy(float) + effect_m[snr >= 5],
            saved.H1_stored_as_m_residual_m.to_numpy(float), rtol=0, atol=1e-10,
        ):
            raise ValueError(f"stored-meter residual does not match the frozen prior output for {granule}")
        for column, effect in (
            ("H1_stored_as_ns_two_way_residual_m", effect_ns_two_way),
            ("H1_stored_as_ns_one_way_residual_m", effect_ns_one_way),
        ):
            if not np.allclose(
                base.loc[snr >= 5, "C7_residual_m"].to_numpy(float) + effect[snr >= 5],
                saved[column].to_numpy(float), rtol=0, atol=1e-10,
            ):
                raise ValueError(f"{column} does not match the frozen prior output for {granule}")
        if not np.allclose(
            saved.C7_residual_m.to_numpy(float) + era5_effect,
            saved.H1_era5_profile_residual_m.to_numpy(float), rtol=0, atol=1e-10,
        ):
            raise ValueError(f"ERA5 sensitivity does not reproduce the frozen prior output for {granule}")

        base["stored_delay_numeric"] = fields["refh_neutat_delay_total"]
        base["height_effect_if_m"] = effect_m
        base["height_effect_if_ns_two_way"] = effect_ns_two_way
        base["height_effect_if_ns_one_way"] = effect_ns_one_way
        base["height_effect_ERA5_profile"] = era5_effect
        scenarios = {
            "C7": base.C7_residual_m.to_numpy(float).copy(),
            "C1": base.C1_residual_m.to_numpy(float).copy(),
            "C3": base.C3_residual_m.to_numpy(float).copy(),
            "H1_stored_as_m": add_sensitivity(base.C7_residual_m.to_numpy(float), effect_m),
            "H1_ERA5_profile": add_sensitivity(base.C7_residual_m.to_numpy(float), np.full(len(base), era5_effect)),
            "H1_stored_as_ns_two_way": add_sensitivity(base.C7_residual_m.to_numpy(float), effect_ns_two_way),
            "H1_stored_as_ns_one_way": add_sensitivity(base.C7_residual_m.to_numpy(float), effect_ns_one_way),
        }
        masks = stratum_masks(snr)
        for stratum in STRATA:
            mask = masks[stratum]
            group = base.loc[mask].reset_index(drop=True)
            group_values = {name: values[mask].copy() for name, values in scenarios.items()}
            if len(group) != EXPECTED_N[granule][stratum]:
                raise ValueError(f"{granule}/{stratum} count differs from frozen baseline")
            group_ids = group.pulse_index.to_numpy(dtype="<i8")
            group_hash = pulse_id_sha256(group_ids)
            if any(len(values) != len(group_ids) for values in group_values.values()):
                raise ValueError(f"scenario pulse counts differ for {granule}/{stratum}")
            for scenario, values in group_values.items():
                row = _metric_row(group, scenario, stratum, values, status=_scenario_status(scenario))
                if row["pulse_id_sha256"] != group_hash:
                    raise AssertionError("scenario pulse ID hashes differ")
                metric_rows.append(row)
                distribution_rows.extend(_pulse_distribution_rows(group, stratum, {scenario: values}))
            spatial_rows.extend(_spatial_summary_rows(group, stratum, group_values))

            old_names = {
                "C7": "C7", "C1": "C1", "C3": "C3",
                "H1_stored_as_m": "H1 atmospheric sensitivity",
            }
            old_ids = legacy.loc[
                legacy.granule_id.eq(granule) & legacy.population.eq(stratum), "pulse_id_sha256"
            ].drop_duplicates()
            if len(old_ids) != 1 or old_ids.iloc[0] != group_hash:
                raise ValueError(f"prior saved {stratum} pulse IDs differ for {granule}")
            for scenario, old_name in old_names.items():
                old = legacy.loc[
                    legacy.granule_id.eq(granule) & legacy.population.eq(stratum)
                    & legacy.hypothesis.eq(old_name)
                ]
                current = next(row for row in metric_rows if row["granule_id"] == granule
                               and row["stratum"] == stratum and row["hypothesis"] == scenario)
                if len(old) != 1:
                    raise ValueError(f"prior saved metrics missing {granule}/{stratum}/{old_name}")
                comparisons = {key: (float(old.iloc[0][old_key]), float(current[key])) for key, old_key in (
                    ("mean_bias_m", "mean_bias_m"), ("median_bias_m", "median_bias_m"),
                    ("RMSE_m", "RMSE_m"), ("NMAD_m", "NMAD_m"), ("absolute_P95_m", "absolute_P95_m"),
                )}
                maximum = max(abs(a - b) for a, b in comparisons.values())
                if maximum > 1e-10:
                    raise ValueError(f"legacy metric parity failed for {granule}/{scenario}: {maximum:g}")
                parity_rows.append({"granule_id": granule, "stratum": stratum, "hypothesis": scenario,
                                    "pulse_id_sha256_matches": True, "maximum_metric_difference_m": maximum,
                                    "parity_status": "PASS"})

        if granule == GRANULES[0]:
            high = base.loc[snr >= 5].reset_index(drop=True)
            high_scenarios = {name: values[snr >= 5] for name, values in scenarios.items()}
            high_influence = _high_influence_table(high, high_scenarios)

    metrics = pd.DataFrame(metric_rows)
    distributions = pd.DataFrame(distribution_rows)
    spatial = pd.DataFrame(spatial_rows)
    decomposition = _make_bias_decomposition(metrics)
    if not decomposition.decomposition_identity_status.eq("PASS").all():
        raise AssertionError("bias/dispersion decomposition identity failed")
    if high_influence is None or len(high_influence) != 14:
        raise AssertionError("November 12 high-influence table was not built")

    csv_files = {
        "residual_metrics_comparison.csv": metrics,
        "residual_bias_decomposition.csv": decomposition,
        "residual_distribution_diagnostics.csv": distributions,
        "per_track_spatial_summary.csv": spatial,
        "high_influence_pulse_cases.csv": high_influence,
    }
    for name, frame in csv_files.items():
        frame.to_csv(output / name, index=False)
    figures = _plot_outputs(metrics, distributions, spatial, high_influence, output)
    output_files = sorted([output / name for name in csv_files] + figures)
    manifest = {
        "analysis": "CASALS ground-target residual bias and dispersion diagnostics",
        "analysis_code": {
            "path": str(Path(__file__).resolve().relative_to(root)),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "support_rule": "independent_test AND existing common_support_all_candidates; no residual-based filtering",
        "selected_rows": {granule: int((selected.granule_id == granule).sum()) for granule in GRANULES},
        "inputs": {
            str(path.relative_to(root)): {"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in (population_path, saved_residual_path, legacy_path, source_manifest_path)
        },
        "source_h5_identity_from_verified_visual_revision_manifest": {
            granule: {
                "path": source_manifest["inputs"][granule]["h5_path"],
                "size_bytes": source_manifest["inputs"][granule]["h5_size_bytes"],
                "sha256": source_manifest["inputs"][granule]["h5_sha256"],
            }
            for granule in GRANULES
        },
        "stratum_counts": {granule: EXPECTED_N[granule] for granule in GRANULES},
        "h1_status": "conditional sensitivity only; H5 units and applied status unknown",
        "era5_granule_effect_m": granule_effects,
        "confidence_intervals": "not estimated; one reviewed road site per granule",
        "legacy_parity": parity_rows,
        "files": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in output_files
        ],
    }
    (output / "residual_diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_root": output,
        "metrics": metrics,
        "decomposition": decomposition,
        "distributions": distributions,
        "spatial": spatial,
        "high_influence": high_influence,
        "legacy_parity": pd.DataFrame(parity_rows),
        "figures": figures,
        "manifest": manifest,
    }
