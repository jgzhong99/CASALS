"""Focused visualizations for the CASALS elevation and waveform notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from rasterio.windows import Window, bounds as window_bounds
from scipy.spatial import cKDTree


_COLORS = {
    "All valid": "#536878",
    "SNR ≥ 3": "#247ba0",
    "SNR ≥ 5": "#d17a22",
    "open_field": "#2a9d8f",
    "tree_canopy": "#557153",
    "road_surface": "#b56576",
    "water": "#277da1",
}


def save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _spaced_indices(size: int, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, limit, dtype=np.int64)


def plot_reference_coverage(views: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Show class-2 grid coverage, H5 matching status, and fixed ROI roles."""

    fig, axes = plt.subplots(1, len(views), figsize=(8.0 * len(views), 7.2), squeeze=False)
    for ax, view in zip(axes[0], views):
        ground = np.asarray(view["ground_xy"], dtype=np.float64)
        selected = _spaced_indices(len(ground), 45_000)
        if selected.size:
            ax.scatter(ground[selected, 0], ground[selected, 1], s=0.25, color="#b9c0c6", alpha=0.6,
                       rasterized=True, label=f"3DEP class-2 grid ({len(selected):,} shown)")
        x, y = np.asarray(view["x"]), np.asarray(view["y"])
        valid = np.asarray(view["valid"], dtype=bool)
        for key, label, color, marker, limit in (
            ("unsupported", "No valid local plane", "#ca6b45", ".", 18_000),
            ("flagged", "Plane supported, quality flagged", "#e0a528", "x", 12_000),
            ("acceptable", "Acceptable local plane", "#286f9b", ".", 24_000),
        ):
            mask = valid & np.asarray(view[key], dtype=bool)
            ids = np.flatnonzero(mask)
            if ids.size:
                take = _spaced_indices(ids.size, limit)
                use = ids[take]
                ax.scatter(x[use], y[use], s=2.2 if marker == "." else 7.0, c=color, marker=marker,
                           linewidths=0.35, alpha=0.74, rasterized=True,
                           label=f"{label} ({ids.size:,}; {len(use):,} shown)")
        for roi in view["rois"]:
            xy = np.asarray(roi["xy"], dtype=np.float64)
            ax.plot(xy[:, 0], xy[:, 1], color=roi["color"], lw=1.6, ls=roi["linestyle"])
            center = xy[:-1].mean(axis=0) if len(xy) > 1 else xy.mean(axis=0)
            ax.annotate(f"{roi['role']}\n{roi['roi_id']}", center, fontsize=7,
                        color=roi["color"], ha="center", va="center",
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"})
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(title=f"{view['date']} | {view['granule_id']}", xlabel="Easting (m, EPSG:6347)",
               ylabel="Northing (m, EPSG:6347)")
        ax.legend(loc="best", fontsize=7, framealpha=0.9)
        ax.grid(alpha=0.15)
    fig.suptitle("CASALS footprint and 3DEP ground-reference coverage\nDisplay samples are deterministic; match counts use every H5 record",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_path)


def elevation_exceedance_table(
    granule_id: str,
    residuals: dict[str, np.ndarray],
    thresholds_m: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0),
) -> pd.DataFrame:
    rows = []
    for label, values in residuals.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        for threshold in thresholds_m:
            count = int(np.count_nonzero(np.abs(finite) > threshold))
            rows.append({
                "granule_id": granule_id,
                "population": "independent open-field test site",
                "snr_stratum": label,
                "threshold_m": float(threshold),
                "sample_n": int(finite.size),
                "exceedance_n": count,
                "exceedance_probability": float(count / finite.size) if finite.size else np.nan,
                "residual_semantics": "empirically aligned spatial-holdout agreement; not absolute error",
            })
    return pd.DataFrame(rows)


def plot_elevation_error_board(
    *,
    granule_id: str,
    date: str,
    raw_by_stratum: dict[str, np.ndarray],
    holdout_by_stratum: dict[str, np.ndarray],
    raw_metrics: dict[str, dict[str, Any]],
    holdout_metrics: dict[str, dict[str, Any]],
    exceedance: pd.DataFrame,
    output_path: str | Path,
    zoom_m: float = 1.0,
) -> Path:
    """Compare raw differences, full holdout tails, central values, and exceedance."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    labels = list(raw_by_stratum)
    all_raw = np.concatenate([np.asarray(raw_by_stratum[label]) for label in labels])
    finite_raw = all_raw[np.isfinite(all_raw)]
    lo, hi = float(finite_raw.min()), float(finite_raw.max())
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    bins = np.linspace(lo, hi, 180)
    ax = axes[0, 0]
    details = []
    for label in labels:
        values = np.asarray(raw_by_stratum[label], dtype=np.float64)
        values = values[np.isfinite(values)]
        color = _COLORS[label]
        ax.hist(values, bins=bins, density=True, histtype="step", lw=1.7, color=color, label=f"{label} (n={len(values):,})")
        metrics = raw_metrics[label]
        ax.axvline(metrics["median_m"], color=color, ls="-", lw=1.0, alpha=0.8)
        ax.axvline(metrics["bias_m"], color=color, ls=":", lw=1.0, alpha=0.8)
        details.append(
            f"{label}: median {metrics['median_m']:.2f}, mean {metrics['bias_m']:.2f}, "
            f"RMSE {metrics['rmse_m']:.2f}, NMAD {metrics['nmad_m']:.2f} m"
        )
    ax.set_yscale("log")
    ax.set(title="1 | Raw Refh − 3DEP plane differences; full range", xlabel="Raw difference (m; frame mismatch may remain)", ylabel="Density (log scale)")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.02, "Solid line: median   Dotted line: mean\n" + "\n".join(details), transform=ax.transAxes,
            fontsize=7.3, va="bottom", bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none"})

    all_holdout = np.concatenate([np.asarray(holdout_by_stratum[label]) for label in labels])
    finite_holdout = all_holdout[np.isfinite(all_holdout)]
    hmin, hmax = float(finite_holdout.min()), float(finite_holdout.max())
    if hmin == hmax:
        hmin, hmax = hmin - 0.1, hmax + 0.1
    hbins = np.linspace(hmin, hmax, 160)
    ax = axes[0, 1]
    for label in labels:
        values = np.asarray(holdout_by_stratum[label], dtype=np.float64)
        values = values[np.isfinite(values)]
        ax.hist(values, bins=hbins, histtype="step", lw=1.8, color=_COLORS[label],
                label=f"{label} (n={len(values):,})")
    ax.set_yscale("log")
    ax.axvline(0, color="#333333", ls="--", lw=1)
    ax.set(title="2 | Spatial holdout residuals; full range", xlabel="Residual after calibration-site median offset (m)", ylabel="Pulse count (log scale)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for label in labels:
        values = np.asarray(holdout_by_stratum[label], dtype=np.float64)
        values = values[np.isfinite(values)]
        inside = values[np.abs(values) <= zoom_m]
        outside_n = int(values.size - inside.size)
        ax.hist(inside, bins=80, range=(-zoom_m, zoom_m), histtype="step", lw=1.8,
                color=_COLORS[label], label=f"{label}: in view {len(inside):,}; outside {outside_n:,} ({outside_n/max(1,len(values)):.2%})")
    ax.axvline(0, color="#333333", ls="--", lw=1)
    ax.set(title=f"3 | Central residuals (±{zoom_m:g} m)", xlabel="Empirically aligned holdout residual (m)", ylabel="Pulse count")
    ax.legend(fontsize=7.6)

    ax = axes[1, 1]
    thresholds = np.sort(exceedance["threshold_m"].unique())
    for label in labels:
        part = exceedance[exceedance.snr_stratum == label].sort_values("threshold_m")
        probs = part["exceedance_probability"].to_numpy(dtype=np.float64)
        plotted = np.where(probs > 0, probs, np.nan)
        ax.plot(part["threshold_m"], plotted, marker="o", ms=3.5, lw=1.6, color=_COLORS[label], label=label)
        for xval, yval, count in zip(part["threshold_m"], probs, part["exceedance_n"]):
            if yval == 0:
                ax.annotate("0", (xval, 0.008), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6, color=_COLORS[label])
    ax.set_xscale("log")
    ax.set_yscale("log")
    positive = exceedance.loc[exceedance.exceedance_probability > 0, "exceedance_probability"]
    ymin = min(0.00005, float(positive.min()) / 2) if len(positive) else 0.00005
    ax.set_ylim(ymin, 1.1)
    ax.set_xticks(thresholds, labels=[f"{t:g}" for t in thresholds])
    ax.set(title="4 | Absolute residual exceedance", xlabel="Threshold t (m)", ylabel="P(|residual| > t)")
    ax.grid(which="both", alpha=0.2)
    ax.legend(fontsize=8)
    ax.text(0.98, 0.02, "Zero counts are labeled 0; they are not plotted at the detection floor.", transform=ax.transAxes,
            fontsize=7, ha="right", va="bottom")

    foot = "Metrics use all supported, acceptable points; display panels retain long tails. Empirical offsets describe relative holdout agreement, not absolute elevation accuracy."
    fig.suptitle(f"CASALS–3DEP elevation comparison | {date} | {granule_id}\n{foot}", fontsize=13, fontweight="bold")
    return save_figure(fig, output_path)


def plot_spatial_snr_board(
    *,
    granule_id: str,
    date: str,
    test_rows: pd.DataFrame,
    residual_m: np.ndarray,
    gross_error_threshold_m: float,
    output_path: str | Path,
) -> Path:
    """Keep every test-site gross error visible beside SNR and residual magnitude."""

    frame = test_rows.reset_index(drop=True).copy()
    residual = np.asarray(residual_m, dtype=np.float64)
    if len(frame) != len(residual):
        raise ValueError("Residuals and spatial test rows must have matching lengths.")
    valid = np.isfinite(residual) & np.isfinite(frame.easting_m_epsg6347) & np.isfinite(frame.northing_m_epsg6347)
    frame = frame.loc[valid].reset_index(drop=True)
    residual = residual[valid]
    gross = np.abs(residual) > gross_error_threshold_m
    limit = max(float(np.percentile(np.abs(residual), 97)) if residual.size else 0.1, 0.1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    sc = axes[0].scatter(frame.easting_m_epsg6347, frame.northing_m_epsg6347, c=residual,
                         cmap="RdBu_r", norm=norm, s=7, alpha=0.75, rasterized=True)
    axes[0].scatter(frame.loc[gross, "easting_m_epsg6347"], frame.loc[gross, "northing_m_epsg6347"],
                    marker="^", s=50, facecolors="none", edgecolors="#111111", linewidths=1.2,
                    label=f"All |residual| > {gross_error_threshold_m:g} m ({int(gross.sum())})", zorder=5)
    fig.colorbar(sc, ax=axes[0], label="Empirically aligned residual (m)")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].set(title="Signed residual by location", xlabel="Easting (m, EPSG:6347)", ylabel="Northing (m, EPSG:6347)")
    sc2 = axes[1].scatter(frame.easting_m_epsg6347, frame.northing_m_epsg6347,
                          c=frame.refh_snr, cmap="viridis", s=7, alpha=0.75, rasterized=True)
    axes[1].scatter(frame.loc[gross, "easting_m_epsg6347"], frame.loc[gross, "northing_m_epsg6347"],
                    marker="^", s=50, facecolors="none", edgecolors="#111111", linewidths=1.2, zorder=5)
    fig.colorbar(sc2, ax=axes[1], label="H5 Refh SNR")
    axes[1].set(title="SNR at the same positions", xlabel="Easting (m, EPSG:6347)", ylabel="Northing (m, EPSG:6347)")
    for ax in axes[:2]:
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.15)

    axes[2].scatter(frame.refh_snr, np.abs(residual), c=np.where(gross, "#b23a48", "#527a8a"),
                    s=np.where(gross, 26, 8), alpha=0.58, linewidths=0, rasterized=True)
    axes[2].axhline(gross_error_threshold_m, color="#b23a48", ls="--", lw=1.2,
                    label=f"Display threshold {gross_error_threshold_m:g} m")
    axes[2].set_yscale("symlog", linthresh=0.1)
    if residual.size:
        axes[2].set_ylim(0, float(np.max(np.abs(residual))) * 1.05)
    axes[2].set(title=f"Error versus SNR (n={len(frame):,}; gross={int(gross.sum()):,})",
                xlabel="H5 Refh SNR", ylabel="Absolute empirical residual (m; symlog)")
    axes[2].legend(fontsize=7)
    axes[2].text(0.02, 0.98, "All gross points shown; pulse IDs are listed in the linked CSV.",
                 transform=axes[2].transAxes, ha="left", va="top", fontsize=6.5,
                 bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    fig.suptitle(f"Independent test site | {date} | {granule_id}\nAll supported, acceptable SNR ≥ 3 test observations are shown; gross-error threshold only highlights points.",
                 fontsize=12.5, fontweight="bold")
    return save_figure(fig, output_path)


def plot_elevation_case_board(
    *,
    h5_path: str | Path,
    cases: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Display traceable H5 waveforms, local 3DEP support, and case metrics."""

    import h5py

    if not cases:
        raise ValueError("No supported independent-test cases were selected.")
    fig, axes = plt.subplots(len(cases), 3, figsize=(15, max(3.1 * len(cases), 5)), squeeze=False,
                             constrained_layout=True)
    with h5py.File(h5_path, "r") as h5:
        for row_axes, case in zip(axes, cases):
            row = case["row"]
            pulse = int(row.pulse_index)
            xy = np.array([row.easting_m_epsg6347, row.northing_m_epsg6347], dtype=np.float64)
            support = case["ground_points"]
            support_xy = support[["easting_m", "northing_m"]].to_numpy(dtype=np.float64)
            support_z = support["height_m"].to_numpy(dtype=np.float64)
            ax = row_axes[0]
            if len(support):
                sc = ax.scatter(support_xy[:, 0], support_xy[:, 1], c=support_z, cmap="terrain", s=8,
                                alpha=0.75, rasterized=True)
                fig.colorbar(sc, ax=ax, label="3DEP grid height (m; local structure)")
            nearby = case["nearby_casals"]
            if len(nearby):
                ax.scatter(nearby.easting_m_epsg6347, nearby.northing_m_epsg6347,
                           s=8, c="#5b6770", alpha=0.4, label="Nearby CASALS pulses")
            ax.scatter([xy[0]], [xy[1]], marker="*", s=150, c="#d1495b", edgecolors="white",
                       linewidths=0.8, label="Selected pulse", zorder=5)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set(title="A | Local support", xlabel="Easting (m)", ylabel="Northing (m)")
            ax.legend(fontsize=6.5, loc="best")

            waveform = np.asarray(h5["rx_waveform"][pulse, :], dtype=np.float64)
            bg = float(h5["bg_mean"][pulse])
            raw_max_bin = int(np.nanargmax(waveform))
            ax = row_axes[1]
            ax.plot(np.arange(len(waveform)), waveform, color="#24566b", lw=0.65, label="Original RX")
            ax.axhline(bg, color="#777777", ls="--", lw=1, label="H5 bg_mean")
            ax.axvline(raw_max_bin, color="#d1495b", ls=":", lw=1.2, label="Raw maximum")
            ax.set(title=f"B | RX waveform; Refh amplitude {int(h5['refh_amp'][pulse]):,} counts; "
                         f"SNR {float(h5['refh_snr'][pulse]):.2f}",
                   xlabel="RX bin", ylabel="Raw RX amplitude (counts)")
            ax.text(0.99, 0.02, "No official Refh RX-bin field is stored in this H5.",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})
            ax.legend(fontsize=6.5, loc="best")

            ax = row_axes[2]
            refh = float(row.casals_refh_m)
            plane = float(row.reference_height_local_plane_m)
            ax.scatter([0, 1], [refh, plane], s=75, c=["#d1495b", "#457b9d"], zorder=3)
            ax.set_xticks([0, 1], ["CASALS Refh", "3DEP local plane"])
            low, high = min(refh, plane), max(refh, plane)
            pad = max((high - low) * 0.3, 0.5)
            ax.set_ylim(low - pad, high + pad)
            ax.grid(axis="y", alpha=0.25)
            ax.set(title="C | Stored heights; reference frames unresolved", ylabel="Recorded height (m)")
            text = (
                f"Raw difference: {row.raw_difference_plane_m:+.3f} m\n"
                f"All-valid calibrated residual: {case['residual_m']:+.3f} m\n"
                f"SNR {row.refh_snr:.2f} | slope {row.local_slope_deg:.2f}° | roughness "
                f"{row.ground_roughness_nmad_m:.3f} m | neighbors {int(row.neighbor_count_within_3m)}\n"
                f"{case['selection_role']} | pulse {pulse} | track {int(row.track_num)} | sweep {int(row.sweep_num)}"
            )
            ax.text(0.02, 0.02, text, transform=ax.transAxes, va="bottom", fontsize=7.2,
                    bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"})
    fig.suptitle(f"Real CASALS–3DEP test-site examples | {Path(h5_path).stem}\n"
                 "Empirical residuals are relative diagnostics; unresolved gross errors remain unexplained.",
                 fontsize=12, fontweight="bold")
    return save_figure(fig, output_path)


def plot_ground_profile(profile: pd.DataFrame, *, granule_id: str, track: int, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    low_quality = profile.refh_snr.to_numpy(dtype=float) < 3.0
    ax.plot(profile.distance_m, profile.reference_height_local_plane_m, color="#e09f3e", lw=1.2,
            label="3DEP local ground plane")
    ax.plot(profile.distance_m, profile.empirically_aligned_refh_m, color="#277da1", lw=1.1,
            label="CASALS Refh after calibration-site offset")
    ax.scatter(profile.distance_m[~low_quality], profile.empirically_aligned_refh_m[~low_quality],
               s=15, c="#277da1", label="CASALS SNR ≥ 3")
    ax.scatter(profile.distance_m[low_quality], profile.empirically_aligned_refh_m[low_quality],
               s=18, c="#b23a48", marker="x", label="CASALS SNR < 3")
    ax.plot(profile.distance_m, profile.casals_refh_m, color="#8d5a97", lw=0.8, alpha=0.55,
            label="Raw CASALS Refh (offset retained)")
    ax.set(title=f"Ground profile by measured horizontal distance | {granule_id} | track {track}",
           xlabel="Cumulative horizontal distance from H5 projected coordinates (m)",
           ylabel="Stored height (m; vertical frames unresolved)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    return save_figure(fig, output_path)


def _check_naip_raster(ds: rasterio.io.DatasetReader) -> None:
    t = ds.transform
    if ds.crs is None or not np.isclose(t.b, 0.0) or not np.isclose(t.d, 0.0) or t.a <= 0 or t.e >= 0:
        raise ValueError("NAIP imagery must have a projected, north-up affine transform for marker overlays.")


def _show_naip_patch(ax: plt.Axes, ds: rasterio.io.DatasetReader, x: float, y: float, *, half_width_m: float) -> None:
    pixel_x, pixel_y = abs(ds.transform.a), abs(ds.transform.e)
    row, col = ds.index(x, y)
    width = max(1, int(np.ceil(2 * half_width_m / pixel_x)))
    height = max(1, int(np.ceil(2 * half_width_m / pixel_y)))
    window = Window(col - width // 2, row - height // 2, width, height)
    window = window.intersection(Window(0, 0, ds.width, ds.height))
    if window.width < 1 or window.height < 1:
        ax.text(0.5, 0.5, "Outside NAIP raster bounds", transform=ax.transAxes, ha="center", va="center")
        return
    image = np.moveaxis(ds.read([1, 2, 3], window=window), 0, -1)
    left, bottom, right, top = window_bounds(window, ds.transform)
    ax.imshow(image, extent=(left, right, bottom, top), origin="upper")
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)


def plot_naip_roi_board(
    *,
    granule_id: str,
    h5_date: str,
    expected_naip_date: str,
    image_path: str | Path,
    metadata_path: str | Path,
    roi_features: list[dict[str, Any]],
    roi_audit: pd.DataFrame,
    membership: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Render each granule's actual NAIP overview plus checkable ROI insets."""

    import json
    from pyproj import Transformer

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    dates = metadata.get("acquisition_dates", [])
    if dates != [expected_naip_date]:
        raise ValueError(f"NAIP metadata date mismatch: {dates} != {expected_naip_date}.")
    props = [feature["properties"] for feature in roi_features]
    audit = roi_audit.set_index("roi_id")
    counts = membership.groupby("roi_id").size().to_dict()
    with rasterio.open(image_path) as ds:
        _check_naip_raster(ds)
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        image = np.moveaxis(ds.read([1, 2, 3]), 0, -1)
        extent = (ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top)
        fig = plt.figure(figsize=(17, 10), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, width_ratios=(1.3, 1, 1))
        overview = fig.add_subplot(grid[:, 0])
        overview.imshow(image, extent=extent, origin="upper")
        tier_color = {"primary": "#00a88f", "exploratory": "#efaa36", "excluded": "#bb5b63"}
        for feature in roi_features:
            p = feature["properties"]
            rid = p["roi_id"]
            coords = np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float64)
            px, py = transformer.transform(coords[:, 0], coords[:, 1])
            tier = str(audit.loc[rid, "analysis_tier"])
            color = tier_color.get(tier, "#bb5b63")
            overview.plot(px, py, color=color, lw=1.5 if tier == "primary" else 1.1,
                          ls="-" if tier == "primary" else "--")
            overview.annotate(f"{rid}\n{tier}; n={counts.get(rid,0):,}",
                              (float(np.mean(px[:-1])), float(np.mean(py[:-1]))),
                              fontsize=6.5, color="white", ha="center", va="center",
                              bbox={"boxstyle": "round,pad=0.2", "facecolor": "#20262b", "alpha": 0.78, "edgecolor": "none"})
        for rid, group in membership.groupby("roi_id", sort=True):
            ids = _spaced_indices(len(group), 1200)
            sample = group.iloc[ids]
            px, py = transformer.transform(sample.longitude_deg.to_numpy(), sample.latitude_deg.to_numpy())
            tier = str(audit.loc[rid, "analysis_tier"])
            overview.scatter(px, py, s=1.0, color=tier_color.get(tier, "#bb5b63"), alpha=0.5, rasterized=True)
        overview.set(xlabel=f"Easting (m, {ds.crs.to_string()})", ylabel="Northing (m)",
                     title="Granule context | ROI boundaries and nominal H5 centers")
        overview.xaxis.set_major_locator(MaxNLocator(nbins=5))
        overview.yaxis.set_major_locator(MaxNLocator(nbins=6))
        overview.set_aspect("equal", adjustable="datalim")
        overview.legend(handles=[
            plt.Line2D([0], [0], color=tier_color["primary"], label="Primary"),
            plt.Line2D([0], [0], color=tier_color["exploratory"], ls="--", label="Exploratory"),
            plt.Line2D([0], [0], color=tier_color["excluded"], ls="--", label="Excluded"),
        ], loc="best", fontsize=8)

        representative = {}
        for feature in roi_features:
            p = feature["properties"]
            cls = str(p.get("class_label", ""))
            rid = str(p.get("roi_id", ""))
            if counts.get(rid, 0) == 0 or cls not in {"open_field", "tree_canopy", "road_surface", "water"}:
                continue
            current = representative.get(cls)
            if current is None or counts.get(rid, 0) > counts.get(current["properties"]["roi_id"], 0):
                representative[cls] = feature
        class_order = ["open_field", "road_surface", "tree_canopy", "water"]
        for slot, cls in enumerate(class_order):
            ax = fig.add_subplot(grid[slot // 2, 1 + slot % 2])
            feature = representative.get(cls)
            if feature is None:
                ax.axis("off")
                ax.text(0.5, 0.5, f"No pulse-supported {cls.replace('_',' ')} ROI", transform=ax.transAxes,
                        ha="center", va="center")
                continue
            p = feature["properties"]
            rid = str(p["roi_id"])
            group = membership[membership.roi_id == rid]
            coords = np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float64)
            px, py = transformer.transform(coords[:, 0], coords[:, 1])
            cx, cy = float(np.mean(px[:-1])), float(np.mean(py[:-1]))
            half_width = {"open_field": 110.0, "road_surface": 45.0, "tree_canopy": 75.0, "water": 60.0}[cls]
            _show_naip_patch(ax, ds, cx, cy, half_width_m=half_width)
            ax.plot(px, py, color="#f3b61f", lw=2.0, label="ROI boundary")
            take = _spaced_indices(len(group), 2500)
            sample = group.iloc[take]
            sx, sy = transformer.transform(sample.longitude_deg.to_numpy(), sample.latitude_deg.to_numpy())
            ax.scatter(sx, sy, s=5, c="#00d5bd", edgecolors="#123c40", linewidths=0.15,
                       alpha=0.75, label=f"Nominal H5 centers ({len(group):,})")
            tier = str(audit.loc[rid, "analysis_tier"])
            ax.set(title=f"{cls.replace('_',' ').title()} | {rid} | {tier}", xlabel="Easting (m)", ylabel="Northing (m)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="best")
        fig.suptitle(f"Real NAIP context | {granule_id}\nNAIP {expected_naip_date} | CASALS H5 {h5_date}; image-to-footprint registration is uncalibrated",
                     fontsize=12.5, fontweight="bold")
    return save_figure(fig, output_path)


def plot_waveform_gallery(
    *,
    cases: pd.DataFrame,
    h5_paths: dict[str, str | Path],
    naip_paths: dict[str, tuple[str | Path, str | Path]],
    output_path: str | Path,
) -> Path:
    """Pair actual NAIP patches with raw selected waveforms and fixed IDs."""

    import h5py
    import json
    from pyproj import Transformer

    if cases.empty:
        raise ValueError("No waveform cases were available for the gallery.")
    fig, axes = plt.subplots(len(cases), 2, figsize=(16, max(3.0 * len(cases), 7)), squeeze=False,
                             constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.7]})
    for (_, case), (image_ax, wave_ax) in zip(cases.iterrows(), axes):
        stem = str(case.granule_id)
        image_path, metadata_path = naip_paths[stem]
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        with rasterio.open(image_path) as ds:
            _check_naip_raster(ds)
            transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            x, y = transformer.transform(float(case.longitude_deg), float(case.latitude_deg))
            class_widths = {"open_field": 110.0, "road_surface": 40.0, "tree_canopy": 70.0, "water": 55.0}
            _show_naip_patch(image_ax, ds, x, y, half_width_m=class_widths.get(str(case.class_label), 75.0))
            image_ax.scatter([x], [y], marker="*", s=145, c="#ffcc33", edgecolors="#202020", linewidths=0.8,
                             label="Nominal pulse center", zorder=5)
            coords = np.asarray(case.roi_coordinates, dtype=np.float64)
            px, py = transformer.transform(coords[:, 0], coords[:, 1])
            image_ax.plot(px, py, color="#00d5bd", lw=1.4, label="ROI boundary")
            image_ax.set(title=f"NAIP {metadata['acquisition_dates'][0]} | {case.roi_id} ({case.analysis_tier})",
                         xlabel="Easting (m)", ylabel="Northing (m)")
            image_ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            image_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            image_ax.tick_params(labelsize=6.5)
            image_ax.title.set_fontsize(8)
            image_ax.legend(fontsize=6, loc="best")

        with h5py.File(h5_paths[stem], "r") as h5:
            pulse = int(case.pulse_index)
            wave = np.asarray(h5["rx_waveform"][pulse, :], dtype=np.float64)
            bg = float(h5["bg_mean"][pulse])
            raw_max = int(np.nanargmax(wave))
        bins = np.arange(len(wave))
        wave_ax.plot(bins, wave, color="#284b63", lw=0.7, label="Raw RX counts")
        wave_ax.axhline(bg, color="#757575", ls="--", lw=1.0, label="H5 background mean")
        wave_ax.axvline(raw_max, color="#ca3c4e", ls=":", lw=1.2, label="Raw maximum")
        if bool(case.candidate_signal_detected) and pd.notna(case.candidate_peak_bin):
            wave_ax.axvline(int(case.candidate_peak_bin), color="#2a9d8f", ls="-.", lw=1.0, label="Qualified candidate peak")
        wave_ax.set(title=f"{case.selection_role} | SNR {case.refh_snr:.2f} | T{int(case.track_num)} S{int(case.sweep_num)} P{int(case.pulse_index)}",
                    xlabel="RX bin", ylabel="Raw RX amplitude (counts)")
        wave_ax.title.set_fontsize(8)
        wave_ax.tick_params(labelsize=7)
        wave_ax.legend(fontsize=6.8, ncol=2, loc="best")
    fig.suptitle("Representative real RX responses paired with NAIP context\n"
                 "Selection uses fixed within-ROI SNR quantiles; the H5 stores no official Refh RX-bin field.",
                 fontsize=12.5, fontweight="bold")
    return save_figure(fig, output_path)


def plot_landcover_signal_board(roi_summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Keep granules separate and show ROI-level SNR, screen rate, and amplitude."""

    granules = list(dict.fromkeys(roi_summary.granule_id.astype(str)))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False, constrained_layout=True)
    metrics = [
        ("refh_snr_median", "refh_snr_p10", "refh_snr_p90", "Refh SNR (dimensionless)", "Median Refh SNR by ROI"),
        ("candidate_signal_rate_all_roi_pulses", None, None, "Qualified candidate fraction", "Candidate-screen fraction by ROI"),
        ("refh_amp_counts_median", "refh_amp_counts_p10", "refh_amp_counts_p90", "Native Refh amplitude (counts)", "Refh amplitude by ROI"),
    ]
    for row, stem in enumerate(granules[:2]):
        part = roi_summary[roi_summary.granule_id == stem].sort_values(["class_label", "site_id", "roi_id"])
        x = np.arange(len(part))
        colors = [_COLORS.get(str(value), "#687782") for value in part.class_label]
        site_counts = part.groupby("class_label").site_id.nunique().to_dict()
        for col, (center_col, low_col, high_col, ylabel, title) in enumerate(metrics):
            ax = axes[row, col]
            center = part[center_col].to_numpy(dtype=float)
            if low_col:
                low = part[low_col].to_numpy(dtype=float)
                high = part[high_col].to_numpy(dtype=float)
                ax.vlines(x, low, high, color=colors, lw=2.4, alpha=0.8)
            if center_col == "candidate_signal_rate_all_roi_pulses":
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
            ax.scatter(x, center, c=colors, s=42, edgecolors="white", linewidths=0.6, zorder=3)
            if col == 1:
                for xpos, (_, rec) in zip(x, part.iterrows()):
                    ax.annotate(f"{int(rec.n_candidate_signals):,}/{int(rec.n_roi_pulses):,}",
                                (xpos, float(rec[center_col])), xytext=(0, 5), textcoords="offset points",
                                ha="center", fontsize=6)
            labels = [f"{r.roi_id}\n{r.site_id}\nn={int(r.n_roi_pulses):,}" for r in part.itertuples()]
            ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=6.5)
            ax.set(title=title, ylabel=ylabel)
            ax.grid(axis="y", alpha=0.2)
        class_sites = ", ".join(f"{key}: {value} site(s)" for key, value in sorted(site_counts.items()))
        axes[row, 0].text(0.01, 0.98, f"{stem}\n{class_sites}", transform=axes[row, 0].transAxes,
                          va="top", fontsize=6.8, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    for row in range(len(granules), 2):
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("ROI-level signal differences by granule\n"
                 "Each ROI/site remains visible; candidate fractions are research screening rates, not calibrated detection probabilities.",
                 fontsize=12.5, fontweight="bold")
    return save_figure(fig, output_path)


def plot_waveform_morphology(roi_features: pd.DataFrame, output_path: str | Path) -> Path:
    """Compare candidate-only widths and native relative support energy."""

    selected = roi_features[roi_features.candidate_signal_detected.astype(bool)].copy()
    granules = list(dict.fromkeys(roi_features.granule_id.astype(str)))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), squeeze=False, constrained_layout=True)
    metrics = [
        ("width_halfmax_bins", "Candidate peak width (bins)"),
        ("dominant_support_energy_counts_bin", "Dominant support energy (counts·bin)"),
        ("asymmetry_skewness", "Support asymmetry (skewness; optional)"),
    ]
    for row, stem in enumerate(granules[:2]):
        source = roi_features[roi_features.granule_id == stem].sort_values(["class_label", "site_id", "roi_id"])
        labels, colors, groups = [], [], []
        for rec in source.drop_duplicates("roi_id").itertuples():
            group = selected[(selected.granule_id == stem) & (selected.roi_id == rec.roi_id)]
            values = group
            labels.append(f"{rec.roi_id}\n{rec.site_id}\nn={len(group):,}")
            colors.append(_COLORS.get(str(rec.class_label), "#687782"))
            groups.append(values)
        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            data = [group[metric].dropna().to_numpy(dtype=float) for group in groups if metric in group]
            positions = [i + 1 for i, group in enumerate(groups) if metric in group]
            nonempty = [(x, values) for x, values in zip(positions, data) if values.size]
            if nonempty:
                positions, data = zip(*nonempty)
                boxes = ax.boxplot(data, positions=positions, widths=0.55, showfliers=False,
                                   patch_artist=True, medianprops={"color": "#202020", "lw": 1.2})
                for box, pos in zip(boxes["boxes"], positions):
                    box.set_facecolor(colors[pos - 1]); box.set_alpha(0.72)
                ax.set_xticks(range(1, len(labels) + 1), labels, rotation=30, ha="right", fontsize=6.3)
            else:
                ax.text(0.5, 0.5, "No qualified candidate waveforms", transform=ax.transAxes, ha="center")
            ax.set(title=ylabel, ylabel=ylabel)
            ax.grid(axis="y", alpha=0.2)
        axes[row, 0].text(0.01, 0.98, stem, transform=axes[row, 0].transAxes,
                          va="top", fontsize=7, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    for row in range(len(granules), 2):
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Waveform morphology among qualified candidate signals only\n"
                 "Width remains in bins; energy remains native relative counts·bin. These are not all-pulse distributions.",
                 fontsize=12.5, fontweight="bold")
    return save_figure(fig, output_path)


def plot_matched_track_board(pairs: pd.DataFrame, output_path: str | Path) -> Path:
    """Show within-granule, site-level contrasts on common track IDs only."""

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)
    metrics = [
        ("snr_difference_a_minus_b", "SNR difference (open field − comparison)"),
        ("amplitude_difference_a_minus_b", "Amplitude difference (counts)"),
        ("candidate_rate_difference_a_minus_b", "Candidate-rate difference (fraction)"),
    ]
    if pairs.empty:
        for ax in axes:
            ax.axis("off")
            ax.text(0.5, 0.5, "No shared track IDs for primary open-field comparisons in this granule.",
                    transform=ax.transAxes, ha="center", va="center", wrap=True)
        fig.suptitle("Matched-track comparisons | no common support", fontsize=12, fontweight="bold")
        return save_figure(fig, output_path)
    category_column = "comparison_short" if "comparison_short" in pairs else "comparison"
    categories = list(dict.fromkeys(pairs[category_column].astype(str)))
    for ax, (metric, ylabel) in zip(axes, metrics):
        if metric not in pairs:
            ax.axis("off")
            continue
        for pos, category in enumerate(categories):
            part = pairs[pairs[category_column].astype(str) == category]
            jitter = np.linspace(-0.12, 0.12, len(part)) if len(part) > 1 else np.array([0.0])
            colors = ["#277da1" if str(value) == "tree_canopy" else "#bb6b4b"
                      for value in part.get("class_b", pd.Series("", index=part.index))]
            ax.scatter(pos + jitter, part[metric], c=colors, s=27, alpha=0.75, edgecolors="white", linewidths=0.4)
            if len(part):
                median = float(np.nanmedian(part[metric]))
                ax.plot([pos - 0.22, pos + 0.22], [median, median], color="#202020", lw=1.2)
        ax.axhline(0, color="#333333", ls="--", lw=1)
        ax.set_xticks(range(len(categories)), categories, rotation=0, ha="center", fontsize=7)
        ax.set(ylabel=ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Matched-track differences by independent site pair\n"
                 "Descriptions only; adjacent ROIs from one site are combined before matching.",
                 fontsize=12, fontweight="bold")
    return save_figure(fig, output_path)
