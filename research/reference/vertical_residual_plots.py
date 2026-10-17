"""Plot signed H0/H2 residuals on the saved all-method common pulse sets."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from research.reference.refh_correction_audit import residual_metrics


H0 = "H0_noop_C7_baseline"
H2 = "H2_ITRF2014_obs_epoch"
METHODS = (H0, H2)
HORIZONTAL_EPSG = 6347
STAT_COLUMNS = ("mean_bias_m", "median_bias_m", "RMSE_m", "cRMSE_m", "NMAD_m", "absolute_P95_m")
OUTPUT_NAMES = {
    "histograms": "vertical_residual_histograms.png",
    "cdf": "h0_vs_h2_residual_cdf.png",
    "spatial": "h2_spatial_residuals.png",
}
COLORS = {H0: "#b2182b", H2: "#2166ac"}
LABELS = {H0: "H0 no-op C7 baseline", H2: "H2 ITRF2014 at observation time"}


def _pulse_hash(ids) -> str:
    values = np.sort(np.asarray(ids, dtype="<i8").reshape(-1))
    return hashlib.sha256(values.tobytes()).hexdigest()


def prepare_common_h0_h2(result: dict) -> tuple[dict[tuple[str, str], pd.DataFrame], dict[str, dict]]:
    """Validate and return H0/H2 rows for the persisted field independent-test intersection."""
    metrics = result["metrics"]
    common = result["common_pulses"]
    pulses = result["pulse_residuals"]
    required_common = {"granule_id", "population", "stratum", "pulse_index", "common_pulse_sha256"}
    required_metrics = {"granule_id", "hypothesis", "population", "stratum", "n", "pulse_sha256", *STAT_COLUMNS}
    required_pulses = {"granule_id", "hypothesis", "pulse_index", "population", "research_role",
                       "refh_snr", "supported", "residual", "query_x_m", "query_y_m", "query_crs_epsg"}
    for label, frame, required in (("common_pulses", common, required_common),
                                   ("metrics", metrics, required_metrics),
                                   ("pulse_residuals", pulses, required_pulses)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")

    common = common.loc[common.population.eq("field") & common.stratum.eq("snr_ge_5")].copy()
    granules = sorted(common.granule_id.astype(str).unique())
    if len(granules) != 2:
        raise ValueError(f"Expected the two current granules in the common-pulse table, found {granules}")

    samples: dict[tuple[str, str], pd.DataFrame] = {}
    metadata: dict[str, dict] = {}
    for granule_id in granules:
        group = common.loc[common.granule_id.eq(granule_id)].copy()
        ids_raw = group.pulse_index.to_numpy()
        ids = ids_raw.astype(np.int64)
        if not np.array_equal(ids_raw, ids):
            raise ValueError(f"Non-integer pulse IDs in common set for {granule_id}")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Duplicate common pulse IDs for {granule_id}")
        if len(group.common_pulse_sha256.unique()) != 1:
            raise ValueError(f"Common pulse rows have inconsistent hashes for {granule_id}")
        pulse_sha256 = _pulse_hash(ids)
        recorded_hash = str(group.common_pulse_sha256.iloc[0])
        if pulse_sha256 != recorded_hash:
            raise ValueError(f"Common pulse ID hash mismatch for {granule_id}")

        method_metrics = {}
        for hypothesis in METHODS:
            metric_rows = metrics.loc[
                metrics.granule_id.eq(granule_id) & metrics.hypothesis.eq(hypothesis)
                & metrics.population.eq("field") & metrics.stratum.eq("snr_ge_5")
            ]
            if len(metric_rows) != 1:
                raise ValueError(f"Expected one saved {hypothesis} metric row for {granule_id}; got {len(metric_rows)}")
            metric = metric_rows.iloc[0]
            if int(metric.n) != len(ids) or str(metric.pulse_sha256) != pulse_sha256:
                raise ValueError(f"Saved {hypothesis} count/hash does not match common IDs for {granule_id}")

            rows = pulses.loc[
                pulses.granule_id.eq(granule_id) & pulses.hypothesis.eq(hypothesis)
                & pulses.population.eq("field") & pulses.research_role.eq("independent_test")
            ].copy()
            if rows.pulse_index.duplicated().any():
                raise ValueError(f"Duplicate {hypothesis} pulse rows for {granule_id}")
            rows = rows.loc[rows.pulse_index.isin(ids)].copy()
            if len(rows) != len(ids) or set(rows.pulse_index.astype(np.int64)) != set(ids):
                raise ValueError(f"{hypothesis} does not contain every common pulse exactly once for {granule_id}")
            rows = rows.sort_values("pulse_index").reset_index(drop=True)
            if not np.array_equal(rows.pulse_index.to_numpy(np.int64), np.sort(ids)):
                raise ValueError(f"{hypothesis} pulse order differs from the common ID set for {granule_id}")
            snr = rows.refh_snr.to_numpy(float)
            residual = rows.residual.to_numpy(float)
            if not np.isfinite(snr).all() or np.any(snr < 5):
                raise ValueError(f"{hypothesis} common pulses violate the SNR >= 5 filter for {granule_id}")
            if not rows.supported.astype(bool).all():
                raise ValueError(f"Unsupported {hypothesis} rows appear in the common set for {granule_id}")
            if not np.isfinite(residual).all():
                raise ValueError(f"Non-finite {hypothesis} residuals appear in the common set for {granule_id}")
            if not np.isfinite(rows[["query_x_m", "query_y_m"]].to_numpy(float)).all():
                raise ValueError(f"Non-finite query coordinates appear in the common set for {granule_id}")
            if not rows.query_crs_epsg.eq(HORIZONTAL_EPSG).all():
                raise ValueError(f"Query coordinates must be EPSG:{HORIZONTAL_EPSG} for {granule_id}")

            computed = residual_metrics(residual)
            if int(computed["n"]) != int(metric.n):
                raise ValueError(f"Recomputed {hypothesis} n does not match saved metrics for {granule_id}")
            for column in STAT_COLUMNS:
                if not np.isclose(float(computed[column]), float(metric[column]), rtol=0.0, atol=1e-10):
                    raise ValueError(f"Recomputed {hypothesis} {column} does not match saved metrics for {granule_id}")
            method_metrics[hypothesis] = {"n": int(metric.n),
                                          **{column: float(metric[column]) for column in STAT_COLUMNS}}
            samples[(granule_id, hypothesis)] = rows

        h0_ids = samples[(granule_id, H0)].pulse_index.to_numpy(np.int64)
        h2_ids = samples[(granule_id, H2)].pulse_index.to_numpy(np.int64)
        if not np.array_equal(h0_ids, h2_ids):
            raise ValueError(f"H0/H2 plotted pulse IDs differ for {granule_id}")
        metadata[granule_id] = {"pulse_ids": np.sort(ids), "pulse_sha256": pulse_sha256,
                                "metrics": method_metrics}
    return samples, metadata


def histogram_edges(values, *, min_bin_width_m: float = 0.02) -> np.ndarray:
    """Build fixed-width Freedman-Diaconis edges that include the complete support."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Histogram input must be a non-empty finite residual array")
    if min_bin_width_m <= 0:
        raise ValueError("Minimum histogram bin width must be positive")
    q25, q75 = np.quantile(values, [0.25, 0.75])
    fd_width = 2.0 * float(q75 - q25) / (len(values) ** (1.0 / 3.0))
    width = max(fd_width, float(min_bin_width_m))
    lo, hi = float(values.min()), float(values.max())
    start = np.floor(lo / width) * width
    stop = np.ceil(hi / width) * width
    if stop <= start:
        stop = start + width
    count = max(1, int(np.ceil((stop - start) / width)))
    edges = start + np.arange(count + 1, dtype=float) * width
    while edges[-1] < hi:
        edges = np.append(edges, edges[-1] + width)
    if edges[0] > lo:
        edges = np.insert(edges, 0, edges[0] - width)
    return edges


def plot_histograms(samples: dict[tuple[str, str], pd.DataFrame]):
    """Create density histograms with one reproducible bin definition per method."""
    import matplotlib.pyplot as plt

    granules = sorted({key[0] for key in samples})
    if len(granules) != 2:
        raise ValueError(f"Expected two granules for the histogram layout, found {granules}")
    edges = {}
    for hypothesis in METHODS:
        combined = np.concatenate([samples[(gid, hypothesis)].residual.to_numpy(float) for gid in granules])
        edges[hypothesis] = histogram_edges(combined)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="row", sharey="row", constrained_layout=True)
    for row, hypothesis in enumerate(METHODS):
        for col, granule_id in enumerate(granules):
            ax = axes[row, col]
            frame = samples[(granule_id, hypothesis)]
            values = frame.residual.to_numpy(float)
            stats = residual_metrics(values)
            ax.hist(values, bins=edges[hypothesis], density=True, color=COLORS[hypothesis], alpha=0.38,
                    edgecolor=COLORS[hypothesis], linewidth=0.55)
            ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.1, label="Zero")
            ax.axvline(stats["mean_bias_m"], color="#d95f02", linestyle="-", linewidth=1.2, label="Mean")
            ax.axvline(stats["median_bias_m"], color="#1b9e77", linestyle=":", linewidth=1.4, label="Median")
            ax.set_title(_date_label(granule_id))
            ax.grid(axis="y", alpha=0.22)
            ax.text(0.02, 0.97,
                    f"n = {stats['n']:,}\nmean bias = {stats['mean_bias_m']:+.3f} m\n"
                    f"RMSE = {stats['RMSE_m']:.3f} m\nNMAD = {stats['NMAD_m']:.3f} m",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none"})
            if col == 0:
                ax.set_ylabel(f"{LABELS[hypothesis]}\nDensity (m$^{{-1}}$)")
            if row == 1:
                ax.set_xlabel("Signed residual, CASALS − 3DEP (m)")
            xlo = min(0.0, float(edges[hypothesis][0]))
            xhi = max(0.0, float(edges[hypothesis][-1]))
            pad = max((xhi - xlo) * 0.02, 0.01)
            ax.set_xlim(xlo - pad, xhi + pad)
            ax.legend(loc="upper right", fontsize=8, frameon=False)
            if values.min() < ax.get_xlim()[0] or values.max() > ax.get_xlim()[1]:
                raise AssertionError("Histogram x-limits clipped residual support")
    fig.suptitle("Signed vertical residuals on identical all-method common pulses | field independent test | SNR ≥ 5")
    return fig, edges


def plot_cdf(samples: dict[tuple[str, str], pd.DataFrame]):
    """Create empirical CDFs from uncentered signed residuals on the plotted common IDs."""
    import matplotlib.pyplot as plt

    granules = sorted({key[0] for key in samples})
    if len(granules) != 2:
        raise ValueError(f"Expected two granules for the CDF layout, found {granules}")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), constrained_layout=True)
    for ax, granule_id in zip(axes, granules):
        values_all = []
        for hypothesis in METHODS:
            values = np.sort(samples[(granule_id, hypothesis)].residual.to_numpy(float))
            values_all.append(values)
            ax.plot(values, np.arange(1, len(values) + 1, dtype=float) / len(values),
                    label=LABELS[hypothesis], color=COLORS[hypothesis], linewidth=1.6)
        lo = min(0.0, *(float(v.min()) for v in values_all))
        hi = max(0.0, *(float(v.max()) for v in values_all))
        pad = max((hi - lo) * 0.025, 0.01)
        ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0, label="Zero residual")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set(title=f"{_date_label(granule_id)} | n={len(values_all[0]):,}",
               xlabel="Signed residual, CASALS − 3DEP (m)", ylabel="Cumulative fraction")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("H0 and H2 empirical CDFs | identical all-method common pulses | no centering")
    return fig


def plot_h2_spatial(samples: dict[tuple[str, str], pd.DataFrame]):
    """Map H2 signed residuals at H2's own EPSG:6347 3DEP query coordinates."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator

    granules = sorted({key[0] for key in samples})
    if len(granules) != 2:
        raise ValueError(f"Expected two granules for the spatial map layout, found {granules}")
    h2_frames = [samples[(gid, H2)] for gid in granules]
    all_residuals = np.concatenate([frame.residual.to_numpy(float) for frame in h2_frames])
    limit = float(np.max(np.abs(all_residuals)))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1e-12
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    scatter = None
    for ax, granule_id, frame in zip(axes, granules, h2_frames):
        if not frame.query_crs_epsg.eq(HORIZONTAL_EPSG).all():
            raise ValueError(f"H2 query coordinates are not EPSG:{HORIZONTAL_EPSG} for {granule_id}")
        x = frame.query_x_m.to_numpy(float)
        y = frame.query_y_m.to_numpy(float)
        values = frame.residual.to_numpy(float)
        scatter = ax.scatter(x, y, c=values, s=8, cmap="RdBu_r", norm=norm,
                             rasterized=True, linewidths=0)
        stats = residual_metrics(values)
        ax.text(0.02, 0.98,
                f"n = {stats['n']:,}\nmean bias = {stats['mean_bias_m']:+.3f} m\nRMSE = {stats['RMSE_m']:.3f} m",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none"})
        ax.set(title=_date_label(granule_id), xlabel="H2 transformed easting, EPSG:6347 (m)",
               ylabel="H2 transformed northing, EPSG:6347 (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.15)
    fig.colorbar(scatter, ax=axes, label="H2 signed residual, CASALS − 3DEP (m)")
    fig.suptitle("H2 spatial residuals on identical all-method common pulses | full shared color range")
    return fig, (-limit, limit)


def _date_label(granule_id: str) -> str:
    import re
    from datetime import datetime

    match = re.search(r"20\d{6}", str(granule_id))
    return datetime.strptime(match.group(0), "%Y%m%d").strftime("%Y-%m-%d") if match else str(granule_id)


def save_vertical_residual_plots(result: dict, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Validate, render, and save new artifacts without replacing existing files."""
    import matplotlib.pyplot as plt

    samples, _ = prepare_common_h0_h2(result)
    output = Path(output_dir if output_dir is not None else result["output"])
    output.mkdir(parents=True, exist_ok=True)
    paths = {key: output / name for key, name in OUTPUT_NAMES.items()}
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to replace existing plot artifacts: {existing}")

    figures = []
    try:
        hist, _ = plot_histograms(samples)
        cdf = plot_cdf(samples)
        spatial, _ = plot_h2_spatial(samples)
        figures.extend([hist, cdf, spatial])
        hist.savefig(paths["histograms"], dpi=180, bbox_inches="tight", facecolor="white")
        cdf.savefig(paths["cdf"], dpi=180, bbox_inches="tight", facecolor="white")
        spatial.savefig(paths["spatial"], dpi=180, bbox_inches="tight", facecolor="white")
    finally:
        for figure in figures:
            plt.close(figure)
    return paths
