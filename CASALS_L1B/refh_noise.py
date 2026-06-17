"""Shared CASALS L1B refh noise/outlier labeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class NoiseConfig:
    use_snr_amp_noise_label: bool = True
    snr_hard_min: float = 1.5
    snr_soft_min: float = 1.8
    amp_low_percentile: float = 5.0
    low_signal_requires_low_amp_for_soft_snr: bool = True
    use_refh_threshold_if_available: bool = True
    refh_threshold_margin_min: float = 0.0
    use_error_fields_if_available: bool = True
    max_refh_error_m: Optional[float] = None
    max_refh_horizontal_error_deg: Optional[float] = None
    use_global_z_percentile_guard: bool = False
    z_low_percentile: float = 0.05
    z_high_percentile: float = 99.95
    use_local_height_filter: bool = True
    local_grid_cell_size_m: float = 5.0
    local_min_points_per_cell: int = 12
    local_abs_residual_threshold_m: float = 25.0
    local_mad_multiplier: float = 8.0
    local_min_sigma_m: float = 0.75
    use_open3d_statistical_outlier: bool = False
    open3d_sor_nb_neighbors: int = 20
    open3d_sor_std_ratio: float = 2.5
    open3d_sor_max_points: int = 750_000
    open3d_sor_seed: int = 42
    allow_sampled_sor_for_labeling: bool = False


@dataclass
class NoiseResult:
    noise_mask: np.ndarray
    keep_mask: np.ndarray
    reason_code: np.ndarray
    local_median_z: np.ndarray
    local_mad_z: np.ndarray
    local_z_residual: np.ndarray
    thresholds: dict[str, Any]
    counts: dict[str, Any]


REASON_LOW_SNR_HARD = np.uint16(1 << 0)
REASON_LOW_SNR_AND_LOW_AMP = np.uint16(1 << 1)
REASON_BELOW_REFH_THRESHOLD = np.uint16(1 << 2)
REASON_LOCAL_HEIGHT_OUTLIER = np.uint16(1 << 3)
REASON_OPEN3D_SOR_OUTLIER = np.uint16(1 << 4)
REASON_GLOBAL_Z_PERCENTILE = np.uint16(1 << 5)
REASON_REFH_ERROR = np.uint16(1 << 6)
REASON_REFH_HORIZ_ERROR = np.uint16(1 << 7)

REASON_NAMES = {
    int(REASON_LOW_SNR_HARD): "low_snr_hard",
    int(REASON_LOW_SNR_AND_LOW_AMP): "low_snr_and_low_amp",
    int(REASON_BELOW_REFH_THRESHOLD): "below_refh_threshold",
    int(REASON_LOCAL_HEIGHT_OUTLIER): "local_height_outlier",
    int(REASON_OPEN3D_SOR_OUTLIER): "open3d_sor_outlier",
    int(REASON_GLOBAL_Z_PERCENTILE): "global_z_percentile_guard",
    int(REASON_REFH_ERROR): "refh_error_exceeds_threshold",
    int(REASON_REFH_HORIZ_ERROR): "refh_horizontal_error_exceeds_threshold",
}


def add_reason(reason_code: np.ndarray, mask: np.ndarray, reason_bit: np.uint16) -> None:
    reason_code[mask] = np.bitwise_or(reason_code[mask], reason_bit)


def compute_cell_median_mad(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell_size_m: float,
    min_points_per_cell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be > 0")
    if min_points_per_cell <= 0:
        raise ValueError("min_points_per_cell must be > 0")
    if z.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, {
            "cell_size_m": float(cell_size_m),
            "min_points_per_cell": int(min_points_per_cell),
            "n_cells_total": 0,
            "n_cells_with_stats": 0,
            "median_points_per_cell": None,
            "p98_points_per_cell": None,
            "xmin": None,
            "ymin": None,
            "nx": 0,
        }

    xmin = float(np.nanmin(x))
    ymin = float(np.nanmin(y))
    ix = np.floor((x - xmin) / cell_size_m).astype(np.int64)
    iy = np.floor((y - ymin) / cell_size_m).astype(np.int64)
    nx = int(ix.max() + 1)
    key = iy * np.int64(nx) + ix
    unique_key, inv = np.unique(key, return_inverse=True)
    n_groups = len(unique_key)
    counts = np.bincount(inv, minlength=n_groups)

    cell_median = np.full(n_groups, np.nan, dtype=np.float64)
    cell_mad = np.full(n_groups, np.nan, dtype=np.float64)

    order = np.argsort(inv, kind="mergesort")
    z_sorted = z[order]
    starts = np.r_[0, np.cumsum(counts[:-1])]

    for g in range(n_groups):
        c = int(counts[g])
        if c < min_points_per_cell:
            continue
        segment = z_sorted[starts[g] : starts[g] + c]
        med = float(np.nanmedian(segment))
        mad = float(np.nanmedian(np.abs(segment - med)))
        cell_median[g] = med
        cell_mad[g] = mad

    local_median = cell_median[inv]
    local_mad = cell_mad[inv]
    local_residual = z - local_median

    info = {
        "cell_size_m": float(cell_size_m),
        "min_points_per_cell": int(min_points_per_cell),
        "n_cells_total": int(n_groups),
        "n_cells_with_stats": int(np.sum(np.isfinite(cell_median))),
        "median_points_per_cell": float(np.median(counts)) if len(counts) else None,
        "p98_points_per_cell": float(np.percentile(counts, 98)) if len(counts) else None,
        "xmin": xmin,
        "ymin": ymin,
        "nx": nx,
    }
    return local_median, local_mad, local_residual, info


def open3d_statistical_outlier_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cfg: NoiseConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(x)
    outlier = np.zeros(n, dtype=bool)
    info: dict[str, Any] = {
        "enabled": bool(cfg.use_open3d_statistical_outlier),
        "applied": False,
        "n_input": int(n),
        "reason": "not_enabled",
    }
    if not cfg.use_open3d_statistical_outlier:
        return outlier, info

    if n > cfg.open3d_sor_max_points and not cfg.allow_sampled_sor_for_labeling:
        info.update({
            "applied": False,
            "reason": "skipped_too_many_points",
            "max_points": int(cfg.open3d_sor_max_points),
        })
        return outlier, info

    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        info.update({"applied": False, "reason": f"open3d_import_failed: {type(exc).__name__}: {exc}"})
        return outlier, info

    if n > cfg.open3d_sor_max_points:
        rng = np.random.default_rng(cfg.open3d_sor_seed)
        idx = rng.choice(n, size=int(cfg.open3d_sor_max_points), replace=False)
        info["labeling_mode"] = "sampled_subset_only"
    else:
        idx = np.arange(n)
        info["labeling_mode"] = "all_points"

    pts = np.column_stack([
        x[idx] - float(np.nanmedian(x[idx])),
        y[idx] - float(np.nanmedian(y[idx])),
        z[idx] - float(np.nanmedian(z[idx])),
    ]).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    _, ind_inlier = pcd.remove_statistical_outlier(
        nb_neighbors=int(cfg.open3d_sor_nb_neighbors),
        std_ratio=float(cfg.open3d_sor_std_ratio),
    )
    inlier_subset = np.zeros(len(idx), dtype=bool)
    inlier_subset[np.asarray(ind_inlier, dtype=np.int64)] = True
    outlier[idx] = ~inlier_subset
    info.update({
        "applied": True,
        "reason": "ok",
        "n_processed": int(len(idx)),
        "n_outlier": int(np.sum(outlier)),
        "nb_neighbors": int(cfg.open3d_sor_nb_neighbors),
        "std_ratio": float(cfg.open3d_sor_std_ratio),
    })
    return outlier, info


def label_noise_arrays(
    *,
    z_refh: np.ndarray,
    refh_snr: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    cfg: NoiseConfig,
    refh_amp: Optional[np.ndarray] = None,
    optional: Optional[dict[str, np.ndarray]] = None,
) -> NoiseResult:
    z_refh = np.asarray(z_refh, dtype=np.float64)
    refh_snr = np.asarray(refh_snr, dtype=np.float64)
    easting = np.asarray(easting, dtype=np.float64)
    northing = np.asarray(northing, dtype=np.float64)
    refh_amp = None if refh_amp is None else np.asarray(refh_amp, dtype=np.float64)
    optional = optional or {}

    n = len(z_refh)
    reason_code = np.zeros(n, dtype=np.uint16)
    thresholds: dict[str, Any] = {}

    if cfg.use_snr_amp_noise_label:
        thresholds.update({
            "snr_hard_min": float(cfg.snr_hard_min),
            "snr_soft_min": float(cfg.snr_soft_min),
            "amp_low_percentile": float(cfg.amp_low_percentile),
        })
        low_snr_hard = refh_snr < cfg.snr_hard_min
        add_reason(reason_code, low_snr_hard, REASON_LOW_SNR_HARD)
        if refh_amp is not None:
            amp_threshold = float(np.nanpercentile(refh_amp, cfg.amp_low_percentile))
            thresholds["amp_low_threshold"] = amp_threshold
            low_amp = refh_amp < amp_threshold
            low_snr_soft = refh_snr < cfg.snr_soft_min
            if cfg.low_signal_requires_low_amp_for_soft_snr:
                add_reason(reason_code, low_snr_soft & low_amp, REASON_LOW_SNR_AND_LOW_AMP)
            else:
                add_reason(reason_code, low_snr_soft, REASON_LOW_SNR_AND_LOW_AMP)
        else:
            thresholds["amp_available"] = False

    if cfg.use_refh_threshold_if_available and refh_amp is not None and "refh_thres" in optional:
        thres = np.asarray(optional["refh_thres"], dtype=np.float64)
        valid = np.isfinite(thres)
        below = valid & ((refh_amp - thres) <= float(cfg.refh_threshold_margin_min))
        add_reason(reason_code, below, REASON_BELOW_REFH_THRESHOLD)
        thresholds.update({
            "refh_thres_available": True,
            "refh_threshold_margin_min": float(cfg.refh_threshold_margin_min),
            "n_below_refh_threshold": int(np.sum(below)),
        })
    else:
        thresholds["refh_thres_available"] = "refh_thres" in optional

    if cfg.use_error_fields_if_available:
        if cfg.max_refh_error_m is not None and "refh_error" in optional:
            err = np.asarray(optional["refh_error"], dtype=np.float64)
            bad = np.isfinite(err) & (err > float(cfg.max_refh_error_m))
            add_reason(reason_code, bad, REASON_REFH_ERROR)
            thresholds["max_refh_error_m"] = float(cfg.max_refh_error_m)
        if cfg.max_refh_horizontal_error_deg is not None:
            bad_h = np.zeros(n, dtype=bool)
            for name in ("refh_longitude_error", "refh_latitude_error"):
                if name in optional:
                    err = np.asarray(optional[name], dtype=np.float64)
                    bad_h |= np.isfinite(err) & (np.abs(err) > float(cfg.max_refh_horizontal_error_deg))
            add_reason(reason_code, bad_h, REASON_REFH_HORIZ_ERROR)
            thresholds["max_refh_horizontal_error_deg"] = float(cfg.max_refh_horizontal_error_deg)

    if cfg.use_global_z_percentile_guard:
        zlo, zhi = np.nanpercentile(z_refh, [cfg.z_low_percentile, cfg.z_high_percentile])
        bad = (z_refh < zlo) | (z_refh > zhi)
        add_reason(reason_code, bad, REASON_GLOBAL_Z_PERCENTILE)
        thresholds.update({
            "z_low_percentile": float(cfg.z_low_percentile),
            "z_high_percentile": float(cfg.z_high_percentile),
            "z_low_threshold_m": float(zlo),
            "z_high_threshold_m": float(zhi),
        })

    local_median = np.full(n, np.nan, dtype=np.float64)
    local_mad = np.full(n, np.nan, dtype=np.float64)
    local_residual = np.full(n, np.nan, dtype=np.float64)
    if cfg.use_local_height_filter:
        local_median, local_mad, local_residual, local_info = compute_cell_median_mad(
            easting,
            northing,
            z_refh,
            cell_size_m=cfg.local_grid_cell_size_m,
            min_points_per_cell=cfg.local_min_points_per_cell,
        )
        sigma = np.maximum(local_mad * 1.4826, float(cfg.local_min_sigma_m))
        adaptive_threshold = np.maximum(
            float(cfg.local_abs_residual_threshold_m),
            float(cfg.local_mad_multiplier) * sigma,
        )
        bad = np.isfinite(local_residual) & (np.abs(local_residual) > adaptive_threshold)
        add_reason(reason_code, bad, REASON_LOCAL_HEIGHT_OUTLIER)
        thresholds["local_height_filter"] = {
            **local_info,
            "local_abs_residual_threshold_m": float(cfg.local_abs_residual_threshold_m),
            "local_mad_multiplier": float(cfg.local_mad_multiplier),
            "local_min_sigma_m": float(cfg.local_min_sigma_m),
            "n_local_height_outlier": int(np.sum(bad)),
        }

    sor_mask, sor_info = open3d_statistical_outlier_mask(easting, northing, z_refh, cfg)
    if np.any(sor_mask):
        add_reason(reason_code, sor_mask, REASON_OPEN3D_SOR_OUTLIER)
    thresholds["open3d_statistical_outlier"] = sor_info

    noise_mask = reason_code != 0
    keep_mask = ~noise_mask

    counts_by_reason: dict[str, int] = {}
    for bit, name in REASON_NAMES.items():
        counts_by_reason[name] = int(np.sum((reason_code & np.uint16(bit)) != 0))

    counts = {
        "n_total_valid_input_points": int(n),
        "n_keep": int(np.sum(keep_mask)),
        "n_noise_labeled": int(np.sum(noise_mask)),
        "keep_fraction": float(np.mean(keep_mask)) if n else 0.0,
        "noise_fraction": float(np.mean(noise_mask)) if n else 0.0,
        "counts_by_reason": counts_by_reason,
    }
    return NoiseResult(
        noise_mask=noise_mask,
        keep_mask=keep_mask,
        reason_code=reason_code,
        local_median_z=local_median,
        local_mad_z=local_mad,
        local_z_residual=local_residual,
        thresholds=thresholds,
        counts=counts,
    )


def label_noise(data: Any, proj: Any, cfg: NoiseConfig) -> NoiseResult:
    return label_noise_arrays(
        z_refh=data.z_refh,
        refh_amp=getattr(data, "refh_amp", None),
        refh_snr=data.refh_snr,
        easting=proj.easting,
        northing=proj.northing,
        optional=getattr(data, "optional", {}),
        cfg=cfg,
    )
