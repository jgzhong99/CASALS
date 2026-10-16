"""Auditable CASALS-to-3DEP vertical-datum candidate evaluation.

This module keeps geodetic evidence, uncorrected candidate residuals, and
empirical alignment separate. Expensive point-cloud work is limited to the
existing clipped LAZ files and small, traceable EPT samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/notebooks/10_3dep_elevation_validation/datum_comparison"
REVISED_ROOT = PROJECT_ROOT / "outputs/notebooks/10_3dep_elevation_validation/revised"
H5_ROOT = PROJECT_ROOT / "data/raw/casals_l1b"
LAZ_ROOT = PROJECT_ROOT / "data/reference/3dep"


@dataclass(frozen=True)
class GranuleSpec:
    granule_id: str
    h5_name: str
    laz_name: str
    work_unit: str
    project: str
    ept_prefix: str
    ept_url: str
    pointcloud_xml: str
    final_report: str
    ept_points: int

    @property
    def h5_path(self) -> Path:
        return H5_ROOT / self.h5_name

    @property
    def laz_path(self) -> Path:
        return LAZ_ROOT / self.laz_name

    @property
    def revised_dir(self) -> Path:
        return REVISED_ROOT / self.granule_id


GRANULES = (
    GranuleSpec(
        "casals_l1b_20241112T165718_001_02",
        "casals_l1b_20241112T165718_001_02.h5",
        "casals_l1b_20241112T165718_001_02_MD_Southeast_1_2019_EPSG6347_39a068a77804.laz",
        "MD_Southeast_1_2019",
        "MD_Southeast_2019_D20",
        "MD_Southeast_1_2019",
        "https://usgs-lidar-public.s3.amazonaws.com/MD_Southeast_1_2019/ept.json",
        "MD_Southeast_1_2019_pointcloud.xml",
        "MD_Southeast_1_2019_final_lpc_report.txt",
        14_910_331_922,
    ),
    GranuleSpec(
        "casals_l1b_20241118T171757_001_02",
        "casals_l1b_20241118T171757_001_02.h5",
        "casals_l1b_20241118T171757_001_02_NC_HurricaneFlorence_9_2020_EPSG6347_f50533a04725.laz",
        "NC_HurricaneFlorence_9_2020",
        "NC_HurricaneFlorence_2020_D20",
        "NC_HurricaneFlorence_9_2020",
        "https://usgs-lidar-public.s3.amazonaws.com/NC_HurricaneFlorence_9_2020/ept.json",
        "NC_HurricaneFlorence_9_2020_pointcloud.xml",
        "NC_HurricaneFlorence_9_2020_final_lpc_report.txt",
        180_495_136_950,
    ),
)


H5_HEIGHT_FIELDS = (
    "refh", "refh_latitude", "refh_longitude", "geoid", "geoid_free2mean",
    "refh_error", "refh_snr", "good_snr", "tide_earth", "tide_earth_free2mean",
    "tide_load", "tide_ocean", "tide_ocean_pole", "tide_pole",
    "range_bias_correction", "refh_neutat_delay_total", "refh_neutat_delay_derivative",
)

TARGET_HORIZONTAL_EPSG = 6347
TARGET_VERTICAL_EPSG = 5703
TARGET_COMPOUND = "EPSG:6347 + EPSG:5703"
GEOID18_GRID = "us_noaa_g2018u0.tif"
GEOID18_GRID_URL = f"https://cdn.proj.org/{GEOID18_GRID}"
GEOID18_GRID_SHA256 = "fa9a407ac7ee3f5a3694008e4bcd09ce9cc250452f0c3b11700a4960340abce2"

CANDIDATE_LABELS = {
    "C0": "Native Refh",
    "C1": "Refh − H5 geoid",
    "C2": "Refh + H5 geoid",
    "C3": "Refh − H5 geoid − free-to-mean",
    "C4": "Refh − H5 geoid + free-to-mean",
    "C5": "Refh + H5 geoid − free-to-mean",
    "C6": "Refh + H5 geoid + free-to-mean",
    "C7": "PROJ XYZ → NAD83(2011) / UTM18N + NAVD88 (GEOID18)",
    "C8": "Empirical Refh alignment (calibration only)",
}


def _json_value(value: Any) -> Any:
    """Convert numpy/HDF5 values into compact JSON-safe values."""
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OUS":
            return [_json_value(v) for v in value.tolist()]
        if value.size <= 64:
            return [_json_value(v) for v in value.tolist()]
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_geoid18_grid() -> Path:
    """Require the pinned local NOAA GEOID18 grid used by every C7 route."""
    from pyproj import datadir

    data_dirs = [Path(value) for value in str(datadir.get_data_dir()).split(os.pathsep) if value]
    grid = next((directory / GEOID18_GRID for directory in data_dirs
                 if (directory / GEOID18_GRID).is_file()), None)
    if grid is None:
        raise FileNotFoundError(
            f"Required local GEOID18 grid {GEOID18_GRID} is unavailable; configure the pinned grid first."
        )
    actual = _sha256(grid)
    if actual != GEOID18_GRID_SHA256:
        raise ValueError(
            f"GEOID18 grid SHA256 mismatch for {grid}: {actual} != {GEOID18_GRID_SHA256}"
        )
    return grid


def _write_artifact_manifest(output_dir: Path) -> Path:
    """Hash experiment outputs after the final summary has been written."""
    manifest_path = output_dir / "datum_comparison_artifact_manifest.csv"
    rows = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file() and p != manifest_path):
        rows.append({
            "relative_path": path.relative_to(output_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _canonical_hash(obj: Any) -> str:
    encoded = json.dumps(_json_value(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summary(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    n = int(array.size)
    if not finite.size:
        return {"n": n, "n_finite": 0, "finite_fraction": 0.0, "min": None, "p01": None,
                "median": None, "mean": None, "p99": None, "max": None, "std": None}
    q = np.quantile(finite, [0.01, 0.5, 0.99])
    return {
        "n": n, "n_finite": int(finite.size), "finite_fraction": float(finite.size / max(n, 1)),
        "min": float(np.min(finite)), "p01": float(q[0]), "median": float(q[1]),
        "mean": float(np.mean(finite)), "p99": float(q[2]), "max": float(np.max(finite)),
        "std": float(np.std(finite, ddof=0)),
    }


def _plain_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in attrs.items():
        if str(key) == "DIMENSION_LIST":
            continue
        result[str(key)] = _json_value(value)
    return result


def audit_casals_height_fields(output_dir: Path = OUTPUT_ROOT) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit actual H5 attributes and values; absent units/frames remain absent."""
    import h5py

    granule_rows: dict[str, Any] = {}
    field_rows: list[dict[str, Any]] = []
    for spec in GRANULES:
        with h5py.File(spec.h5_path, "r") as h5:
            root_attrs = _plain_attrs(h5.attrs)
            data: dict[str, np.ndarray] = {}
            fields: dict[str, Any] = {}
            for name in H5_HEIGHT_FIELDS:
                if name not in h5:
                    fields[name] = {"present": False, "definition_status": "missing"}
                    continue
                ds = h5[name]
                values = np.asarray(ds)
                data[name] = values
                # HDF5 dimension references are not physical metadata; all readable
                # user attributes are retained verbatim for the audit.
                attrs = _plain_attrs(ds.attrs)
                stats = _summary(values)
                fields[name] = {
                    "present": True,
                    "shape": list(ds.shape),
                    "dtype": str(ds.dtype),
                    "attributes": attrs,
                    "units": attrs.get("units", attrs.get("unit")),
                    "stats": stats,
                    "definition_status": "not defined by H5 field attributes",
                }
                field_rows.append({
                    "granule_id": spec.granule_id,
                    "field": name,
                    "present": True,
                    "dtype": str(ds.dtype),
                    "n": int(values.size),
                    "n_finite": int(np.isfinite(values).sum()) if values.dtype.kind in "fiu" else None,
                    "finite_fraction": float(np.isfinite(values).mean()) if values.dtype.kind in "fiu" and values.size else None,
                    "min": stats["min"] if values.dtype.kind in "fiu" else None,
                    "p01": stats["p01"] if values.dtype.kind in "fiu" else None,
                    "median": stats["median"] if values.dtype.kind in "fiu" else None,
                    "p99": stats["p99"] if values.dtype.kind in "fiu" else None,
                    "max": stats["max"] if values.dtype.kind in "fiu" else None,
                    "units": attrs.get("units", attrs.get("unit")),
                    "definition_source": "H5 dataset attributes; no explicit units/definition when blank",
                    "evidence_status": "H5 values and attributes audited; semantics unresolved unless documented externally",
                })
            lat = data.get("refh_latitude", np.array([], dtype=np.float64))
            free = data.get("geoid_free2mean", np.array([], dtype=np.float64))
            free_check: dict[str, Any] = {"available": False}
            if lat.size and free.size and lat.size == free.size:
                expected = 0.1287 - 0.3848 * np.sin(np.deg2rad(lat.astype(np.float64))) ** 2
                delta = free.astype(np.float64) - expected
                free_check = {
                    "available": True,
                    "comparison_formula": "0.1287 - 0.3848 sin(latitude)^2 m (NASA ICESat-2 ATL03 ATBD reference only)",
                    "observed_minus_formula_m": _summary(delta),
                    "interpretation": "Numerical correspondence only; does not establish CASALS field semantics or units.",
                }
            granule_rows[spec.granule_id] = {
                "source_h5": spec.h5_name,
                "source_size_bytes": int(spec.h5_path.stat().st_size),
                "source_sha256": _sha256(spec.h5_path),
                "root_attributes": root_attrs,
                "field_audit": fields,
                "reference_frame_conclusion": {
                    "refh": "CryoCloud L1B tutorial describes refh as WGS84 ellipsoidal height; this H5 has no local units, 3D CRS, realization, coordinate epoch, or processing-convention attributes.",
                    "geoid": "The H5 has a field named geoid but no model, ellipsoid, sign, units, or application status metadata.",
                    "geoid_free2mean": "The H5 has a field named geoid_free2mean but no physical definition, sign, units, or application status metadata.",
                    "tides_and_corrections": "Related fields were inventoried and summarized; H5 attributes do not establish their units or whether they are already included in refh.",
                    "coordinate_epoch": "Unresolved; acquisition time is not a coordinate epoch.",
                    "status": "ASSUMED_SOURCE_FRAME",
                },
                "free_to_mean_formula_comparison": free_check,
            }
    audit = {
        "title": "CASALS L1B height-field metadata audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition_sources": {
            "refh_convention": "https://book.cryointhecloud.com/l1b-waveforms-tutorial/",
            "free_to_mean_comparison": "https://icesat-2.gsfc.nasa.gov/sites/default/files/page_files/ICESat2_ATL03_ATBD_r005.pdf",
            "scope_note": "External conventions are not substituted for missing granule-specific H5 metadata.",
        },
        "granules": granule_rows,
    }
    _write_json(output_dir / "casals_height_metadata_audit.json", audit)
    fields_frame = pd.DataFrame(field_rows)
    fields_frame.to_csv(output_dir / "casals_height_fields_summary.csv", index=False)
    return audit, fields_frame


def freeze_selection_split(output_dir: Path = OUTPUT_ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Freeze geometry-only roles before loading or computing candidate residuals."""
    source_col_rows = []
    manifest_rows = []
    for spec in GRANULES:
        source = pd.read_parquet(
            spec.revised_dir / "matched_casals_3dep_points.parquet",
            columns=["pulse_index", "easting_m_epsg6347", "northing_m_epsg6347", "spatial_split"],
        )
        source_col_rows.append({"granule_id": spec.granule_id, "matched_rows": int(len(source))})
        calibration = source.loc[source.spatial_split.astype(str) == "calibration"].copy()
        independent = source.loc[source.spatial_split.astype(str) == "independent_test"].copy()
        if len(calibration) < 4 or independent.empty:
            raise ValueError(f"Frozen spatial calibration/test split is incomplete for {spec.granule_id}.")
        # The existing test ROI remains untouched. The existing calibration ROI is
        # split deterministically by projected geometry, then pulse index as a tie-break.
        calibration = split_calibration_geometry(calibration)
        independent["research_role"] = "independent_test"
        combined = pd.concat([calibration, independent], ignore_index=True)
        roles = combined["research_role"].astype(str)
        split = combined["spatial_split"].astype(str)
        if not ((roles == "independent_test") == (split == "independent_test")).all():
            raise RuntimeError("The frozen independent-test geometry differs from the preserved split.")
        roi_manifest = pd.read_csv(spec.revised_dir / "spatial_calibration_test_manifest.csv")
        roi_rows = roi_manifest[roi_manifest.split.isin(["calibration", "independent_test"])]
        geometry = "; ".join(
            f"{r.roi_id} ({r.site_id}; {r.split}; {r.geometry})" for r in roi_rows.itertuples(index=False)
        )
        for row in combined.itertuples(index=False):
            manifest_rows.append({
                "granule_id": spec.granule_id,
                "pulse_index": int(row.pulse_index),
                "research_role": str(row.research_role),
                "original_spatial_split": str(row.spatial_split),
                "easting_m_epsg6347": float(row.easting_m_epsg6347),
                "northing_m_epsg6347": float(row.northing_m_epsg6347),
                "fixed_before_candidate_results": True,
                "membership_basis": "preserved manual open-field ROI membership; calibration ROI bisected by sorted EPSG:6347 easting, northing, pulse_index",
                "source_roi_geometry": geometry,
            })
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["granule_id", "research_role", "pulse_index"], kind="mergesort"
    ).reset_index(drop=True)
    if manifest.duplicated(["granule_id", "pulse_index"]).any():
        raise RuntimeError("A pulse was assigned more than one frozen research role.")
    role_counts = manifest.groupby(["granule_id", "research_role"], sort=True).size().to_dict()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Geometry-only partition; this function reads no height, residual, SNR, or candidate columns.",
        "existing_split_source": "revised/*/spatial_calibration_test_manifest.csv and spatial_split labels in matched_casals_3dep_points.parquet",
        "role_counts": {f"{k[0]}::{k[1]}": int(v) for k, v in role_counts.items()},
        "source_matched_rows": source_col_rows,
        "independent_test_rule": "Retain the pre-existing spatially separate independent_test ROI unchanged.",
        "candidate_selection_rule": "Within the pre-existing calibration ROI, sort by EPSG:6347 easting, northing, pulse_index; first floor(n/2) pulses calibrate only the empirical-offset baseline, remaining pulses select geodetic candidates.",
        "test_data_used_for_selection": False,
    }
    out_csv = output_dir / "selection_split_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_csv, index=False)
    summary["manifest_sha256"] = _sha256(out_csv)
    summary["manifest_content_sha256"] = _canonical_hash(manifest.to_dict(orient="records"))
    _write_json(output_dir / "selection_split_manifest_audit.json", summary)
    return manifest, summary


def split_calibration_geometry(calibration: pd.DataFrame) -> pd.DataFrame:
    """Split a preserved calibration ROI into non-overlapping spatial halves."""
    needed = {"pulse_index", "easting_m_epsg6347", "northing_m_epsg6347"}
    missing = needed - set(calibration.columns)
    if missing:
        raise ValueError(f"Calibration geometry is missing columns: {sorted(missing)}")
    if calibration.pulse_index.astype(np.int64).duplicated().any():
        raise ValueError("Calibration pulse indices must be unique before partitioning.")
    result = calibration.sort_values(
        ["easting_m_epsg6347", "northing_m_epsg6347", "pulse_index"], kind="mergesort"
    ).reset_index(drop=True)
    middle = len(result) // 2
    result["research_role"] = "calibration_offset"
    result.loc[middle:, "research_role"] = "candidate_selection"
    return result


def build_height_candidates(
    refh_m: Sequence[float] | np.ndarray,
    geoid_m: Sequence[float] | np.ndarray | None,
    free_to_mean_m: Sequence[float] | np.ndarray | None,
    *,
    units: str = "m",
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Create C0-C6 exactly as written, preserving non-finite source values."""
    if units.strip().lower() not in {"m", "meter", "meters", "metre", "metres"}:
        raise ValueError(f"Candidate formula inputs must be in metres; got units={units!r}.")
    refh = np.asarray(refh_m, dtype=np.float64).reshape(-1)
    candidates = {"C0": refh.copy()}
    warnings: list[str] = []
    if geoid_m is None:
        warnings.append("H5 geoid is missing; only C0 can be computed.")
        return candidates, warnings
    geoid = np.asarray(geoid_m, dtype=np.float64).reshape(-1)
    if len(refh) != len(geoid):
        raise ValueError("Refh and H5 geoid lengths differ.")
    candidates.update({"C1": refh - geoid, "C2": refh + geoid})
    if free_to_mean_m is None:
        warnings.append("H5 geoid_free2mean is missing; C3-C6 cannot be computed.")
        return candidates, warnings
    free = np.asarray(free_to_mean_m, dtype=np.float64).reshape(-1)
    if len(refh) != len(free):
        raise ValueError("Refh and geoid_free2mean lengths differ.")
    candidates.update({
        "C3": refh - geoid - free,
        "C4": refh - geoid + free,
        "C5": refh + geoid - free,
        "C6": refh + geoid + free,
    })
    return candidates, warnings


def common_support_mask(
    candidate_values: Mapping[str, np.ndarray],
    original_surface_acceptable: Sequence[bool] | np.ndarray,
    c7_surface_acceptable: Sequence[bool] | np.ndarray,
    original_reference_m: Sequence[float] | np.ndarray,
    c7_reference_m: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return one candidate-neutral support mask for a common pulse population."""
    arrays = [np.asarray(v, dtype=np.float64).reshape(-1) for v in candidate_values.values()]
    arrays.extend([
        np.asarray(original_surface_acceptable, dtype=bool).reshape(-1),
        np.asarray(c7_surface_acceptable, dtype=bool).reshape(-1),
        np.asarray(original_reference_m, dtype=np.float64).reshape(-1),
        np.asarray(c7_reference_m, dtype=np.float64).reshape(-1),
    ])
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError("All candidate/support arrays must share the same pulse order and length.")
    common = np.asarray(original_surface_acceptable, dtype=bool) & np.asarray(c7_surface_acceptable, dtype=bool)
    common &= np.isfinite(original_reference_m) & np.isfinite(c7_reference_m)
    for values in candidate_values.values():
        common &= np.isfinite(values)
    return common


def distribution_metrics(residual_m: Sequence[float] | np.ndarray) -> dict[str, Any]:
    values = np.asarray(residual_m, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"n": 0, "bias_m": None, "median_m": None, "rmse_m": None, "mae_m": None,
                "nmad_m": None, "abs_p90_m": None, "abs_p95_m": None, "gross_error_rate_gt_2m": None}
    med = float(np.median(values))
    return {
        "n": int(values.size),
        "bias_m": float(np.mean(values)),
        "median_m": med,
        "rmse_m": float(np.sqrt(np.mean(values**2))),
        "mae_m": float(np.mean(np.abs(values))),
        "nmad_m": float(1.4826 * np.median(np.abs(values - med))),
        "abs_p90_m": float(np.quantile(np.abs(values), 0.90)),
        "abs_p95_m": float(np.quantile(np.abs(values), 0.95)),
        "gross_error_rate_gt_2m": float(np.mean(np.abs(values) > 2.0)),
    }


def select_vertical_configuration(
    selection_metrics: pd.DataFrame,
    candidate_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select by evidence first; selection metrics may only be from candidate_selection."""
    if not selection_metrics.empty and "research_role" in selection_metrics:
        roles = set(selection_metrics["research_role"].astype(str))
        if roles - {"candidate_selection"}:
            raise ValueError(f"Configuration selection received forbidden populations: {sorted(roles)}")
    if "candidate_id" in selection_metrics:
        candidates_with_metrics = set(selection_metrics["candidate_id"].astype(str))
    else:
        candidates_with_metrics = set()
    fully_verified = [
        candidate for candidate, evidence in candidate_evidence.items()
        if candidate in candidates_with_metrics
        and evidence.get("evidence_status") == "VERIFIED_GEODETIC"
        and evidence.get("operation_complete", True)
        and evidence.get("source_frame_verified", True)
    ]
    if fully_verified:
        chosen = max(fully_verified, key=lambda candidate: (
            int(candidate_evidence[candidate].get("evidence_rank", 0)),
            -float(selection_metrics.loc[selection_metrics.candidate_id == candidate, "rmse_m"].mean()),
        ))
        status = "VERIFIED_GEODETIC"
        reason = "Selected from candidates with complete source/target frame and operation evidence; numerical metrics are secondary diagnostics."
    else:
        eligible = [
            c for c, e in candidate_evidence.items()
            if c in candidates_with_metrics and e.get("operation_complete", True)
            and e.get("provisional_eligible", False)
        ]
        if not eligible:
            raise ValueError("No evidence-supported or provisional candidate is available for research selection.")
        chosen = max(eligible, key=lambda candidate: (
            int(candidate_evidence[candidate].get("evidence_rank", 0)),
            -float(selection_metrics.loc[selection_metrics.candidate_id == candidate, "rmse_m"].mean()),
        ))
        status = "PROVISIONAL_NOT_DATUM_VERIFIED"
        reason = "No candidate has verified CASALS source realization and coordinate epoch; the highest-ranked physically plausible available path is retained provisionally, with selection-site numerical agreement as a check rather than datum proof."
    rows = selection_metrics[selection_metrics.candidate_id.astype(str) == chosen] if "candidate_id" in selection_metrics else pd.DataFrame()
    return {
        "candidate_id": chosen,
        "candidate_label": CANDIDATE_LABELS.get(chosen, chosen),
        "evidence_status": status,
        "selection_reason": reason,
        "selection_population": "candidate_selection only; independent_test metrics are not an input",
        "selection_metrics": rows.to_dict(orient="records"),
        "test_metrics_used": False,
    }


def configure_private_proj_data(output_dir: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Install only the required public GEOID18 grid in an isolated PROJ directory."""
    import pyproj
    from pyproj import datadir
    import requests

    proj_dir = output_dir / "proj_data"
    proj_dir.mkdir(parents=True, exist_ok=True)
    installed_dir = Path(datadir.get_data_dir())
    proj_db = installed_dir / "proj.db"
    if not proj_db.exists():
        raise FileNotFoundError(f"PROJ database not found in {installed_dir}.")
    private_db = proj_dir / "proj.db"
    if not private_db.exists() or _sha256(private_db) != _sha256(proj_db):
        shutil.copy2(proj_db, private_db)
    grid_path = proj_dir / GEOID18_GRID
    if not grid_path.exists() or grid_path.stat().st_size < 1_000_000:
        response = requests.get(GEOID18_GRID_URL, timeout=120)
        response.raise_for_status()
        grid_path.write_bytes(response.content)
    if _sha256(grid_path) != GEOID18_GRID_SHA256:
        raise ValueError(f"Downloaded GEOID18 grid SHA256 mismatch: {grid_path}")
    os.environ["PROJ_NETWORK"] = "OFF"
    os.environ["PROJ_DATA"] = str(proj_dir)
    os.environ["PROJ_LIB"] = str(proj_dir)
    datadir.set_data_dir(str(proj_dir))
    grid_manifest = {
        "name": GEOID18_GRID,
        "url": GEOID18_GRID_URL,
        "source": "NOAA GEOID18 model distributed by the official PROJ CDN",
        "source_model_url": "https://geodesy.noaa.gov/GEOID/GEOID18/",
        "license": "U.S. Government data; public domain. NOAA NGS source; cite NOAA NGS GEOID18.",
        "local_path": str(grid_path.relative_to(output_dir)),
        "size_bytes": int(grid_path.stat().st_size),
        "sha256": _sha256(grid_path),
        "available": True,
        "used_by_selected_operation": None,
    }
    _write_json(output_dir / "proj_grid_manifest.json", {"grids": [grid_manifest]})
    environment = {
        "python": os.sys.version,
        "pyproj_version": pyproj.__version__,
        "proj_version": pyproj.proj_version_str,
        "installed_proj_data_dir": str(installed_dir),
        "user_proj_data_dir": str(datadir.get_user_data_dir()),
        "private_proj_data_dir": str(proj_dir),
        "private_proj_db_sha256": _sha256(private_db),
        "proj_network": "OFF",
        "operation_selection": "TransformerGroup; always_xy=True; allow_ballpark=False; AOI restricted to CASALS granule locations; C7 additionally requires NOAA GEOID18 grid to match product metadata",
        "target_horizontal_crs": "EPSG:6347",
        "target_vertical_crs": "EPSG:5703",
        "target_compound_crs": TARGET_COMPOUND,
        "source_crs_assumption": "EPSG:4979 (WGS 84 3D ensemble), pending CASALS realization and coordinate epoch evidence",
    }
    _write_json(output_dir / "proj_environment_audit.json", environment)
    return {"environment": environment, "grid": grid_manifest}


def build_target_compound_crs():
    from pyproj import CRS
    from pyproj.crs import CompoundCRS

    horizontal = CRS.from_epsg(TARGET_HORIZONTAL_EPSG)
    vertical = CRS.from_epsg(TARGET_VERTICAL_EPSG)
    target = CompoundCRS(name="NAD83(2011) / UTM zone 18N + NAVD88 height (GEOID18)", components=[horizontal, vertical])
    return CRS.from_wkt(target.to_wkt())


def validate_xyz_crs(source_crs: Any, target_crs: Any) -> None:
    """Reject 2D-only CRSs when a complete XYZ transformation is requested."""
    source_axes = list(getattr(source_crs, "axis_info", []))
    target_axes = list(getattr(target_crs, "axis_info", []))
    if len(source_axes) < 3:
        raise ValueError("Source CRS must explicitly contain three axes for XYZ conversion.")
    if len(target_axes) < 3 or not bool(getattr(target_crs, "is_compound", False)):
        raise ValueError("Target CRS must be a compound horizontal-plus-vertical CRS with three axes.")
    for label, axes in (("source", source_axes), ("target", target_axes)):
        if not all(str(getattr(axis, "unit_name", "")).lower() in {"metre", "meter", "metres", "meters"}
                   for axis in axes[-1:]):
            raise ValueError(f"{label.title()} vertical axis must be in metres.")


def source_frame_evidence_status(realization: str | None, coordinate_epoch: float | None) -> str:
    """A named CRS assumption is not source-frame verification."""
    return "VERIFIED_SOURCE_FRAME" if realization and coordinate_epoch is not None else "ASSUMED_SOURCE_FRAME"


def choose_available_non_ballpark_operation(
    group: Any,
    *,
    required_grid_name: str | None = None,
) -> Any:
    """Choose a ranked available operation, optionally requiring a named grid."""
    transformers = list(getattr(group, "transformers", []))
    if not transformers:
        raise RuntimeError("No available non-ballpark transformation operation exists.")
    for transformer in transformers:
        if bool(getattr(transformer, "is_ballpark", False)):
            continue
        grid_rows, grids_available = selected_operation_grid_report(transformer)
        grid_names = {str(row["name"]) for row in grid_rows}
        if required_grid_name is not None and required_grid_name not in grid_names:
            continue
        if not grids_available:
            continue
        return transformer
    if required_grid_name is not None:
        raise RuntimeError(
            f"No available non-ballpark operation uses required grid {required_grid_name}."
        )
    if all(bool(getattr(transformer, "is_ballpark", False)) for transformer in transformers):
        raise RuntimeError("Ballpark coordinate operations are prohibited.")
    raise RuntimeError("No available non-ballpark operation has all required grids available.")


def selected_operation_grid_report(transformer: Any) -> tuple[list[dict[str, Any]], bool]:
    """Report grids used by one available operation and whether all are local."""
    rows = []
    for operation in getattr(transformer, "operations", []):
        for grid in getattr(operation, "grids", []):
            rows.append({
                "operation": getattr(operation, "name", None),
                "name": getattr(grid, "short_name", None),
                "url": getattr(grid, "url", None),
                "available": bool(getattr(grid, "available", False)),
                "open_license": getattr(grid, "open_license", None),
            })
    return rows, all(row["available"] for row in rows)


def require_selected_operation_grids(transformer: Any) -> list[dict[str, Any]]:
    rows, available = selected_operation_grid_report(transformer)
    if not available:
        missing = [row["name"] for row in rows if not row["available"]]
        raise RuntimeError(f"Selected operation has unavailable required grids: {missing}")
    return rows


def xyz_round_trip_metrics(
    transformer: Any,
    source_lon: Sequence[float] | np.ndarray,
    source_lat: Sequence[float] | np.ndarray,
    source_z: Sequence[float] | np.ndarray,
    target_x: Sequence[float] | np.ndarray,
    target_y: Sequence[float] | np.ndarray,
    target_z: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Measure inverse XYZ round-trip error in horizontal metres and vertical metres."""
    from pyproj import CRS
    from pyproj.enums import TransformDirection

    lon = np.asarray(source_lon, dtype=np.float64)
    lat = np.asarray(source_lat, dtype=np.float64)
    z = np.asarray(source_z, dtype=np.float64)
    x = np.asarray(target_x, dtype=np.float64)
    y = np.asarray(target_y, dtype=np.float64)
    tz = np.asarray(target_z, dtype=np.float64)
    back_lon, back_lat, back_z = transformer.transform(
        x, y, tz, direction=TransformDirection.INVERSE, errcheck=False
    )
    back_lon = np.asarray(back_lon, dtype=np.float64)
    back_lat = np.asarray(back_lat, dtype=np.float64)
    back_z = np.asarray(back_z, dtype=np.float64)
    geod = CRS.from_epsg(4326).get_geod()
    _, _, distance_m = geod.inv(lon, lat, back_lon, back_lat)
    finite = np.isfinite(back_lon) & np.isfinite(back_lat) & np.isfinite(back_z)
    return {
        "sample_count": int(len(lon)),
        "horizontal_error_m": _summary(np.asarray(distance_m)[finite]),
        "vertical_error_m": _summary((back_z - z)[finite]),
        "finite": bool(finite.all()),
    }


def _transformer_group(lon: np.ndarray, lat: np.ndarray, *, source_epsg: int = 4979):
    from pyproj import CRS
    from pyproj.aoi import AreaOfInterest
    from pyproj.transformer import TransformerGroup

    valid = np.isfinite(lon) & np.isfinite(lat)
    if not valid.any():
        raise ValueError("No finite longitude/latitude values are available for the AOI.")
    aoi = AreaOfInterest(
        west_lon_degree=float(np.min(lon[valid])),
        south_lat_degree=float(np.min(lat[valid])),
        east_lon_degree=float(np.max(lon[valid])),
        north_lat_degree=float(np.max(lat[valid])),
    )
    source_crs = CRS.from_epsg(source_epsg)
    target_crs = build_target_compound_crs()
    validate_xyz_crs(source_crs, target_crs)
    group = TransformerGroup(
        source_crs, target_crs,
        always_xy=True, area_of_interest=aoi, allow_ballpark=False,
    )
    return group, aoi


def _grid_audit_rows(group: Any) -> list[dict[str, Any]]:
    rows = []
    for operation in getattr(group, "unavailable_operations", []):
        grids = getattr(operation, "grids", [])
        if not grids:
            rows.append({"operation": getattr(operation, "name", None), "grid": None, "available": False})
        for grid in grids:
            rows.append({
                "operation": getattr(operation, "name", None),
                "grid": getattr(grid, "short_name", None),
                "url": getattr(grid, "url", None),
                "available": bool(getattr(grid, "available", False)),
            })
    return rows


def _transform_audit(group: Any, transformer: Any, source_epsg: int, aoi: Any) -> dict[str, Any]:
    operations = []
    selected_index = next(
        (i for i, candidate in enumerate(group.transformers) if candidate is transformer), None
    )
    for index, tr in enumerate(group.transformers):
        grid_rows, _ = selected_operation_grid_report(tr)
        operations.append({
            "ranked_index": index,
            "name": tr.name,
            "description": tr.description,
            "accuracy_m": tr.accuracy,
            "definition": tr.definition,
            "required_grids": grid_rows,
            "area_of_use": str(tr.area_of_use) if tr.area_of_use else None,
            "operation_json": tr.to_json_dict(),
        })
    return {
        "source_crs": f"EPSG:{source_epsg}",
        "target_crs": TARGET_COMPOUND,
        "aoi_lonlat": {"west": aoi.west_lon_degree, "south": aoi.south_lat_degree,
                        "east": aoi.east_lon_degree, "north": aoi.north_lat_degree},
        "always_xy": True,
        "allow_ballpark": False,
        "best_available": bool(group.best_available),
        "transformer_count": len(group.transformers),
        "unavailable_operations_count": len(group.unavailable_operations),
        "unavailable_operation_grids": _grid_audit_rows(group),
        "selected_operation": operations[selected_index] if selected_index is not None else None,
        "selected_operation_ranked_index": selected_index,
        "all_available_operations": operations,
        "selected_accuracy_m": transformer.accuracy if transformer else None,
        "frame_status": "ASSUMED_SOURCE_FRAME" if source_epsg == 4979 else "KNOWN_NAD83_2011_SOURCE_FOR_GEOID_GRID_LOOKUP",
    }


def transform_c7(
    longitude_deg: Sequence[float] | np.ndarray,
    latitude_deg: Sequence[float] | np.ndarray,
    refh_m: Sequence[float] | np.ndarray,
    *,
    source_epsg: int = 4979,
    return_audit: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    """Transform real XYZ values and verify finite output and inverse round trip."""
    from pyproj import CRS, Transformer
    from pyproj.enums import TransformDirection

    lon = np.asarray(longitude_deg, dtype=np.float64).reshape(-1)
    lat = np.asarray(latitude_deg, dtype=np.float64).reshape(-1)
    z = np.asarray(refh_m, dtype=np.float64).reshape(-1)
    if len({len(lon), len(lat), len(z)}) != 1:
        raise ValueError("Longitude, latitude, and height arrays must have equal length.")
    grid_path = require_geoid18_grid()
    group, aoi = _transformer_group(lon, lat, source_epsg=source_epsg)
    if not group.transformers:
        raise RuntimeError("No non-ballpark C7 transformer is available for this area.")
    # The comparison target's product metadata specifies NAVD88/GEOID18 for
    # every source CRS. Never let the frame-specific path choose another geoid.
    transformer = choose_available_non_ballpark_operation(
        group, required_grid_name=GEOID18_GRID
    )
    used_grids = require_selected_operation_grids(transformer)
    if not any(row.get("name") == GEOID18_GRID and row.get("available") for row in used_grids):
        raise RuntimeError(f"Selected C7 operation does not use required grid {GEOID18_GRID}.")
    x, y, out_z = transformer.transform(lon, lat, z, errcheck=False)
    x, y, out_z = (np.asarray(v, dtype=np.float64) for v in (x, y, out_z))
    finite_input = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(z)
    finite_output = np.isfinite(x) & np.isfinite(y) & np.isfinite(out_z)
    valid = finite_input & finite_output
    audit = _transform_audit(group, transformer, source_epsg, aoi)
    audit["input_count"] = int(len(lon))
    audit["finite_input_count"] = int(finite_input.sum())
    audit["finite_output_count"] = int(finite_output.sum())
    audit["finite_transform_fraction"] = float(valid.sum() / max(int(finite_input.sum()), 1))
    audit["selected_operation_grids"] = used_grids
    audit["selected_operation_status"] = "available_non_ballpark_required_GEOID18_grid"
    audit["selected_operation_grid_constraint"] = GEOID18_GRID
    audit["selected_grid_path"] = str(grid_path)
    audit["selected_grid_sha256"] = GEOID18_GRID_SHA256
    audit["vertical_change_m"] = _summary(out_z[valid] - z[valid])

    sample = np.flatnonzero(valid)
    if sample.size > 512:
        sample = sample[np.linspace(0, sample.size - 1, 512).astype(int)]
    if len(sample):
        audit["round_trip"] = xyz_round_trip_metrics(
            transformer, lon[sample], lat[sample], z[sample], x[sample], y[sample], out_z[sample]
        )
    else:
        audit["round_trip"] = {"sample_count": 0, "finite": False}

    # Compare C7 target XY to the existing pure horizontal re-projection so the
    # changed 3DEP query coordinate is explicitly measured.
    horizontal = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(TARGET_HORIZONTAL_EPSG), always_xy=True)
    hx, hy = horizontal.transform(lon, lat, errcheck=False)
    shift = np.hypot(x - np.asarray(hx), y - np.asarray(hy))
    audit["xy_shift_vs_existing_epsg6347_m"] = _summary(shift[valid])
    if return_audit:
        return x, y, out_z, audit
    return x, y, out_z, None


def sample_geoid18_undulation(
    longitude_deg: Sequence[float] | np.ndarray,
    latitude_deg: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample NOAA GEOID18 through the NAD83(2011) ellipsoid-to-NAVD88 path."""
    require_geoid18_grid()
    lon = np.asarray(longitude_deg, dtype=np.float64).reshape(-1)
    lat = np.asarray(latitude_deg, dtype=np.float64).reshape(-1)
    group, aoi = _transformer_group(lon, lat, source_epsg=6319)
    if not group.transformers:
        raise RuntimeError("GEOID18 sample operation is unavailable in the private PROJ environment.")
    transformer = choose_available_non_ballpark_operation(group, required_grid_name=GEOID18_GRID)
    used_grids = require_selected_operation_grids(transformer)
    _, _, orthometric_from_zero = transformer.transform(lon, lat, np.zeros_like(lon), errcheck=False)
    undulation = -np.asarray(orthometric_from_zero, dtype=np.float64)
    audit = _transform_audit(group, transformer, 6319, aoi)
    audit["selected_operation_grids"] = used_grids
    audit["grid_undulation_sign"] = "N_GEOID18 = - transformed NAVD88 height for input NAD83(2011) ellipsoidal h=0 m"
    audit["undulation_m"] = _summary(undulation)
    return undulation, audit


def _ept_node_bounds(node_id: str, root_bounds: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    depth, ix, iy, iz = (int(part) for part in node_id.split("-"))
    min_x, min_y, min_z, max_x, max_y, max_z = map(float, root_bounds)
    scale = 2**depth
    dx, dy, dz = (max_x - min_x) / scale, (max_y - min_y) / scale, (max_z - min_z) / scale
    return (
        min_x + ix * dx, min_y + iy * dy, min_z + iz * dz,
        min_x + (ix + 1) * dx, min_y + (iy + 1) * dy, min_z + (iz + 1) * dz,
    )


def _intersects_3d(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] <= b[3] and a[3] >= b[0] and a[1] <= b[4] and a[4] >= b[1] and a[2] <= b[5] and a[5] >= b[2]


def _point_identity_keys(points: Any) -> list[tuple[int, ...] | None]:
    """Stable LAS identity from GPS time and recorded return/source attributes."""
    names = {str(n).lower() for n in points.point_format.dimension_names}
    if not {"gps_time", "point_source_id", "return_number", "number_of_returns"}.issubset(names):
        raise ValueError("EPT/clipped point format lacks required source identity dimensions.")
    gps = np.asarray(points.gps_time, dtype=np.float64)
    gps_bits = gps.view(np.uint64)
    arrays = [
        gps_bits,
        np.asarray(points.point_source_id, dtype=np.uint64),
        np.asarray(points.return_number, dtype=np.uint64),
        np.asarray(points.number_of_returns, dtype=np.uint64),
        np.asarray(points.intensity, dtype=np.uint64),
        np.asarray(points.classification, dtype=np.uint64),
    ]
    result: list[tuple[int, ...] | None] = []
    for row in zip(*arrays):
        if not np.isfinite(gps[len(result)]):
            result.append(None)
        else:
            result.append(tuple(int(value) for value in row))
    return result


def _read_ept_source_sample(
    spec: GranuleSpec,
    *,
    max_nodes: int = 6,
    sample_limit: int = 12_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch a few local EPT nodes; never requests the full EPT point cloud."""
    import laspy
    import requests
    from pyproj import Transformer
    from collections import deque

    base_url = spec.ept_url.rsplit("/", 1)[0]
    response = requests.get(spec.ept_url, timeout=60)
    response.raise_for_status()
    ept = response.json()
    if int(ept.get("points", -1)) != spec.ept_points:
        raise RuntimeError(f"EPT point count changed for {spec.work_unit}.")
    if str(ept.get("srs", {}).get("horizontal")) != "3857":
        raise RuntimeError(f"Expected horizontal EPSG:3857 EPT source for {spec.work_unit}.")

    with laspy.open(spec.laz_path) as reader:
        header = reader.header
        if header.parse_crs() is None or header.parse_crs().to_epsg() != 6347:
            raise RuntimeError(f"Clipped LAS horizontal CRS is not EPSG:6347: {spec.laz_path.name}")
        clip_bounds = [
            float(header.mins[0]), float(header.mins[1]), float(header.mins[2]),
            float(header.maxs[0]), float(header.maxs[1]), float(header.maxs[2]),
        ]
        clip_count = int(header.point_count)
        clip_scales = np.asarray(header.scales, dtype=np.float64)
    to_3857 = Transformer.from_crs(6347, 3857, always_xy=True)
    corners = [
        to_3857.transform(x, y)
        for x in (clip_bounds[0], clip_bounds[3])
        for y in (clip_bounds[1], clip_bounds[4])
    ]
    ept_query_bounds = (
        min(x for x, _ in corners) - 3.0,
        min(y for _, y in corners) - 3.0,
        float(clip_bounds[2]) - 5.0,
        max(x for x, _ in corners) + 3.0,
        max(y for _, y in corners) + 3.0,
        float(clip_bounds[5]) + 5.0,
    )

    # EPT's bounds are the index cube; boundsConforming is only the observed
    # data extent. Retain hashes for every hierarchy response used to locate nodes.
    hierarchy_hashes: dict[str, str] = {}
    page_queue = deque(["0-0-0-0"])
    visited: set[str] = set()
    node_candidates: list[tuple[str, int]] = []
    max_depth = 9
    while page_queue:
        page_id = page_queue.popleft()
        if page_id in visited:
            continue
        visited.add(page_id)
        page_url = f"{base_url}/ept-hierarchy/{page_id}.json"
        page_response = requests.get(page_url, timeout=60)
        page_response.raise_for_status()
        hierarchy_hashes[page_url] = hashlib.sha256(page_response.content).hexdigest()
        page = page_response.json()
        for node_id, count in page.items():
            bounds = _ept_node_bounds(node_id, ept["bounds"])
            if not _intersects_3d(bounds, ept_query_bounds):
                continue
            depth = int(node_id.split("-")[0])
            if int(count) == -1:
                if depth <= max_depth:
                    page_queue.append(node_id)
            elif int(count) > 0 and 7 <= depth <= max_depth:
                node_candidates.append((node_id, int(count)))
    if not node_candidates:
        raise RuntimeError(f"No small EPT nodes intersect the clipped LAZ bounds for {spec.work_unit}.")

    # Use the largest fine nodes first, but cap downloads to a tiny subset of the
    # 15-180 billion point source. Deterministic node and point ordering is retained.
    node_candidates.sort(key=lambda pair: (pair[1], pair[0]), reverse=True)
    from collections import Counter

    source_candidates: list[dict[str, Any]] = []
    node_sources: list[dict[str, Any]] = []
    downloaded_nodes = 0
    max_candidate_points = sample_limit * 8
    total_source_points_read = 0
    for node_id, declared_count in node_candidates[:max_nodes]:
        node_url = f"{base_url}/ept-data/{node_id}.laz"
        node_response = requests.get(node_url, timeout=180)
        node_response.raise_for_status()
        node_sha = hashlib.sha256(node_response.content).hexdigest()
        node_sources.append({
            "node_id": node_id,
            "declared_point_count": declared_count,
            "url": node_url,
            "download_size_bytes": len(node_response.content),
            "sha256": node_sha,
        })
        downloaded_nodes += 1
        temp_path = OUTPUT_ROOT / "source_metadata" / f"_sample_{spec.work_unit}_{node_id.replace('-', '_')}.laz"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(node_response.content)
        try:
            points = laspy.read(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
        total_source_points_read += len(points.points)
        src_x = np.asarray(points.x, dtype=np.float64)
        src_y = np.asarray(points.y, dtype=np.float64)
        src_z = np.asarray(points.z, dtype=np.float64)
        source_gps = np.asarray(points.gps_time, dtype=np.float64)
        source_id = np.asarray(points.point_source_id, dtype=np.uint64)
        source_return = np.asarray(points.return_number, dtype=np.uint64)
        source_nreturns = np.asarray(points.number_of_returns, dtype=np.uint64)
        source_intensity = np.asarray(points.intensity, dtype=np.uint64)
        source_class = np.asarray(points.classification, dtype=np.uint64)
        tx, ty = Transformer.from_crs(3857, 6347, always_xy=True).transform(src_x, src_y, errcheck=False)
        tx, ty = np.asarray(tx), np.asarray(ty)
        clip_match_bbox = (
            (tx >= clip_bounds[0] - 3.0) & (tx <= clip_bounds[3] + 3.0)
            & (ty >= clip_bounds[1] - 3.0) & (ty <= clip_bounds[4] + 3.0)
            & (src_z >= clip_bounds[2] - 5.0) & (src_z <= clip_bounds[5] + 5.0)
        )
        local_indices = np.flatnonzero(clip_match_bbox)
        if not local_indices.size:
            continue
        keys = _point_identity_keys(points)
        for local_idx in local_indices:
            key = keys[int(local_idx)]
            if key is None:
                continue
            source_candidates.append({
                "identity_key": key,
                "source_node_id": node_id,
                "source_node_index": int(local_idx),
                "gps_time": float(source_gps[local_idx]),
                "point_source_id": int(source_id[local_idx]),
                "return_number": int(source_return[local_idx]),
                "number_of_returns": int(source_nreturns[local_idx]),
                "intensity": int(source_intensity[local_idx]),
                "classification": int(source_class[local_idx]),
                "source_x_3857_m": float(src_x[local_idx]),
                "source_y_3857_m": float(src_y[local_idx]),
                "source_z_m": float(src_z[local_idx]),
                "source_x_epsg6347_m": float(tx[local_idx]),
                "source_y_epsg6347_m": float(ty[local_idx]),
            })
        if len(source_candidates) >= max_candidate_points:
            break
    if not source_candidates:
        raise RuntimeError(f"Fetched EPT nodes contained no finite points within the clipped LAZ bounding box for {spec.work_unit}.")

    # Remove ambiguous duplicate identities in the source sample, then take a
    # deterministic evenly spaced subset so the local LAS scan remains bounded.
    source_counts = Counter(row["identity_key"] for row in source_candidates)
    unique_rows = [row for row in source_candidates if source_counts[row["identity_key"]] == 1]
    if len(unique_rows) > sample_limit:
        selected = np.linspace(0, len(unique_rows) - 1, sample_limit).astype(np.int64)
        unique_rows = [unique_rows[int(i)] for i in selected]
    if not unique_rows:
        raise RuntimeError(f"No unique source point identities found in the EPT sample for {spec.work_unit}.")
    identity_to_source = {row["identity_key"]: row for row in unique_rows}
    source_gps_bits = np.asarray([key[0] for key in identity_to_source], dtype=np.uint64)
    source_xy = np.asarray([[r["source_x_epsg6347_m"], r["source_y_epsg6347_m"]] for r in unique_rows])
    local_xy_bounds = (
        float(source_xy[:, 0].min() - 2.0), float(source_xy[:, 1].min() - 2.0),
        float(source_xy[:, 0].max() + 2.0), float(source_xy[:, 1].max() + 2.0),
    )
    matches: dict[tuple[int, ...], list[dict[str, Any]]] = {key: [] for key in identity_to_source}
    local_record_offset = 0
    local_scan_count = 0
    with laspy.open(spec.laz_path) as reader:
        for chunk in reader.chunk_iterator(1_000_000):
            size = len(chunk)
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            classification = np.asarray(chunk.classification, dtype=np.uint8)
            intensity = np.asarray(chunk.intensity, dtype=np.uint16)
            in_bbox = (
                (x >= local_xy_bounds[0]) & (x <= local_xy_bounds[2])
                & (y >= local_xy_bounds[1]) & (y <= local_xy_bounds[3])
            )
            gps_bits = np.asarray(chunk.gps_time, dtype=np.float64).view(np.uint64)
            ids = np.flatnonzero(in_bbox & np.isin(gps_bits, source_gps_bits, assume_unique=False))
            local_scan_count += int(len(ids))
            if ids.size:
                subset = chunk[ids]
                keys = _point_identity_keys(subset)
                for local_subset_idx, idx in enumerate(ids):
                    key = keys[local_subset_idx]
                    if key in matches:
                        matches[key].append({
                            "clip_las_point_index": int(local_record_offset + idx),
                            "clip_x_m": float(x[idx]),
                            "clip_y_m": float(y[idx]),
                            "clip_z_m": float(z[idx]),
                            "clip_classification": int(classification[idx]),
                            "clip_intensity": int(intensity[idx]),
                        })
            local_record_offset += size
    sample_rows: list[dict[str, Any]] = []
    duplicate_local = 0
    for key, source in identity_to_source.items():
        found = matches[key]
        if len(found) != 1:
            if len(found) > 1:
                duplicate_local += 1
            continue
        clip = found[0]
        sample_rows.append({
            "granule_id": spec.granule_id,
            "work_unit": spec.work_unit,
            "source_ept_node_id": source["source_node_id"],
            "source_ept_point_index": source["source_node_index"],
            "clip_las_point_index": clip["clip_las_point_index"],
            "gps_time": source["gps_time"],
            "point_source_id": source["point_source_id"],
            "return_number": source["return_number"],
            "number_of_returns": source["number_of_returns"],
            "intensity_source": source["intensity"],
            "intensity_clip": clip["clip_intensity"],
            "classification_source": source["classification"],
            "classification_clip": clip["clip_classification"],
            "source_x_epsg6347_m": source["source_x_epsg6347_m"],
            "source_y_epsg6347_m": source["source_y_epsg6347_m"],
            "clip_x_m": clip["clip_x_m"],
            "clip_y_m": clip["clip_y_m"],
            "source_z_m": source["source_z_m"],
            "clip_z_m": clip["clip_z_m"],
            "delta_z_m": clip["clip_z_m"] - source["source_z_m"],
            "delta_xy_m": math.hypot(clip["clip_x_m"] - source["source_x_epsg6347_m"],
                                      clip["clip_y_m"] - source["source_y_epsg6347_m"]),
        })
    report = {
        "work_unit": spec.work_unit,
        "ept_url": spec.ept_url,
        "ept_point_count": int(ept["points"]),
        "ept_horizontal_crs": "EPSG:3857",
        "ept_vertical_crs": None,
        "ept_schema": ept.get("schema"),
        "ept_hierarchy_pages_read": int(len(hierarchy_hashes)),
        "ept_hierarchy_sha256": hierarchy_hashes,
        "candidate_nodes_found": int(len(node_candidates)),
        "candidate_nodes_downloaded": int(downloaded_nodes),
        "source_node_manifest": node_sources,
        "source_points_read_from_nodes": int(total_source_points_read),
        "source_points_in_clipped_bbox": int(len(source_candidates)),
        "source_unique_identity_candidates": int(len(unique_rows)),
        "source_duplicate_identity_count": int(len(source_candidates) - len({r['identity_key'] for r in source_candidates})),
        "clip_las_point_count": clip_count,
        "clip_las_scales_m": clip_scales.tolist(),
        "clip_las_local_points_scanned_in_bbox": int(local_scan_count),
        "identity_match_count": int(len(sample_rows)),
        "ambiguous_local_identity_count": int(duplicate_local),
        "source_identity_sample_limit": int(sample_limit),
        "point_identity_fields": ["GpsTime bit pattern", "PointSourceId", "ReturnNumber", "NumberOfReturns", "Intensity", "Classification"],
        "coordinate_tolerance_xy_m": 2.0,
        "z_tolerance_m": float(clip_scales[2] / 2.0 + 0.005 + 1e-9),
        "sample_rule": "Evenly spaced unique source point identities among the first six largest intersecting EPT nodes; matched against current clipped LAZ by LAS identity, then XY/Z compared.",
    }
    if sample_rows:
        dz = np.asarray([row["delta_z_m"] for row in sample_rows], dtype=np.float64)
        dxy = np.asarray([row["delta_xy_m"] for row in sample_rows], dtype=np.float64)
        report["delta_z_m"] = _summary(dz)
        report["delta_xy_m"] = _summary(dxy)
        report["all_z_within_quantization_tolerance"] = bool(np.all(np.abs(dz) <= report["z_tolerance_m"] + 1e-12))
        report["all_xy_within_linkage_tolerance"] = bool(np.all(dxy <= report["coordinate_tolerance_xy_m"]))
        report["point_level_z_status"] = (
            "VERIFIED_SAMPLE_PASSTHROUGH" if len(sample_rows) >= 100 and report["all_z_within_quantization_tolerance"]
            else "INCONCLUSIVE_SAMPLE"
        )
    else:
        report.update({"delta_z_m": _summary([]), "delta_xy_m": _summary([]),
                       "all_z_within_quantization_tolerance": False,
                       "all_xy_within_linkage_tolerance": False,
                       "point_level_z_status": "NO_UNAMBIGUOUS_POINT_MATCHES"})
    return sample_rows, report


def audit_3dep_z_provenance(output_dir: Path = OUTPUT_ROOT) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reconcile product metadata and sampled source-to-clip point identities."""
    import re

    metadata_dir = output_dir / "source_metadata"
    source_manifest_path = metadata_dir / "source_metadata_manifest.json"
    downloaded_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    metadata_docs = {row["name"]: row for row in downloaded_manifest}
    granules: dict[str, Any] = {}
    all_samples: list[dict[str, Any]] = []
    for spec in GRANULES:
        xml_path = metadata_dir / spec.pointcloud_xml
        report_path = metadata_dir / spec.final_report
        xml = xml_path.read_text(encoding="utf-8", errors="replace")
        final_report = report_path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
        _, z_audit = _read_ept_source_sample(spec)
        all_samples.extend(_)
        # Keep the contradictory legacy metadata field visible instead of silently
        # reconciling it away. The project abstract and final point-cloud report are
        # the controlling product evidence for these staged EPT work units.
        alt_datum_values = re.findall(r"<altdatum>(.*?)</altdatum>", xml, flags=re.IGNORECASE | re.DOTALL)
        vertical_wkt = re.findall(r"VERT_CS\[\s*\"([^\"]+)\"", final_report, flags=re.IGNORECASE)
        final_vertical_refs = sorted(set(re.findall(r"NAVD88 height - Geoid18 \([^\)]+\)", final_report, flags=re.IGNORECASE)))
        xml_claims_navd_geoid18 = bool(re.search(r"NAVD88.{0,40}GEOID18|GEOID18.{0,40}NAVD88", xml, flags=re.IGNORECASE | re.DOTALL))
        xml_claims_nad83_2011_epoch = bool(re.search(r"NAD83\s*\(2011\).{0,80}epoch\s*2010\.0|epoch\s*2010\.0.{0,80}NAD83\s*\(2011\)", xml, flags=re.IGNORECASE | re.DOTALL))
        final_claims_geoID18 = bool(final_vertical_refs or re.search(r"NAVD88 height - Geoid18", final_report, flags=re.IGNORECASE))
        granules[spec.granule_id] = {
            "work_unit": spec.work_unit,
            "project": spec.project,
            "source_ept_url": spec.ept_url,
            "source_metadata_files": [
                {"name": spec.pointcloud_xml, **{k: v for k, v in metadata_docs[spec.pointcloud_xml].items() if k != "name"}},
                {"name": spec.final_report, **{k: v for k, v in metadata_docs[spec.final_report].items() if k != "name"}},
            ],
            "work_unit_and_product_evidence": {
                "horizontal_crs": "NAD83(2011) / UTM zone 18N, EPSG:6347",
                "vertical_crs": "NAVD88 height, EPSG:5703",
                "geoid_model": "GEOID18",
                "horizontal_coordinate_epoch": "2010.0 per source project XML where explicitly stated",
                "xml_geoid18_claim_in_abstract": xml_claims_navd_geoid18,
                "xml_nad83_2011_epoch_claim": xml_claims_nad83_2011_epoch,
                "xml_altdatum_fields": alt_datum_values,
                "final_report_vertical_wkt_names": vertical_wkt,
                "final_report_navd88_geoid18_claim": final_claims_geoID18,
                "metadata_conflict": any("GEOID12B" in v.upper() for v in alt_datum_values) and final_claims_geoID18,
                "metadata_reconciliation": "The XML abstract and work-unit/product summary plus the final LPC report identify NAVD88/GEOID18; the MD vendor XML's generic altdatum field says GEOID12B and is retained as a source-metadata conflict. The final LPC WKT lists EPSG:5703 for all reported tiles.",
            },
            "clipped_laz": {
                "path": str(spec.laz_path.relative_to(PROJECT_ROOT)),
                "size_bytes": int(spec.laz_path.stat().st_size),
                "sha256": _sha256(spec.laz_path),
                "horizontal_crs": "EPSG:6347; LAS header contains no vertical axis",
                "source_workunit_id_from_preserved_audit": json.loads((spec.revised_dir / "reference_frame_audit.json").read_text(encoding="utf-8"))["historical_provenance_evidence"]["workunit"],
                "historical_vertical_datum_transform_applied_flag": False,
                "flag_interpretation": "Historical sidecar flag is not used as point-level proof; it was set by configuration.",
            },
            "point_identity_z_check": z_audit,
        }
    samples = pd.DataFrame(all_samples)
    samples.to_csv(output_dir / "3dep_source_clip_z_check.csv", index=False)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata_manifest": str(source_manifest_path.relative_to(PROJECT_ROOT)),
        "method": "Read USGS EPT metadata and a capped subset of local EPT LAZ nodes; match exact LAS identity attributes before comparing XY and Z. No full source point cloud was downloaded.",
        "source_z_semantics": "Work-unit/product evidence: NAVD88 (EPSG:5703), GEOID18. EPT JSON itself identifies only horizontal EPSG:3857.",
        "clipped_z_semantics": "The clipped LAS header contains only horizontal EPSG:6347; sampled source-to-clip Z agreement tests whether historical clipping changed source Z.",
        "granules": granules,
        "overall_status": "VERIFIED_SAMPLE_PASSTHROUGH" if all(g["point_identity_z_check"]["point_level_z_status"] == "VERIFIED_SAMPLE_PASSTHROUGH" for g in granules.values()) else "PARTIAL_OR_INCONCLUSIVE",
        "scope_limit": "Point-level result is a traceable sample from each clipped product; it does not inspect every LAS record. Source vertical semantics follow the source product metadata, not the EPT projection declaration.",
    }
    _write_json(output_dir / "3dep_z_provenance_audit.json", report)
    return report, samples


def _sample_geoid_comparison(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare H5 geoid fields with NOAA GEOID18 at reproducible real locations."""
    import h5py

    rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for spec in GRANULES:
        with h5py.File(spec.h5_path, "r") as h5:
            n = int(h5["refh"].shape[0])
            ids = np.linspace(0, n - 1, min(1200, n), dtype=np.int64)
            lon = np.asarray(h5["refh_longitude"][ids], dtype=np.float64)
            lat = np.asarray(h5["refh_latitude"][ids], dtype=np.float64)
            refh = np.asarray(h5["refh"][ids], dtype=np.float64)
            h5_geoid = np.asarray(h5["geoid"][ids], dtype=np.float64) if "geoid" in h5 else np.full(len(ids), np.nan)
            free = np.asarray(h5["geoid_free2mean"][ids], dtype=np.float64) if "geoid_free2mean" in h5 else np.full(len(ids), np.nan)
        valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(refh)
        lon, lat, refh, h5_geoid, free, ids = (v[valid] for v in (lon, lat, refh, h5_geoid, free, ids))
        geoid18, geoid_audit = sample_geoid18_undulation(lon, lat)
        _, _, c7_height, c7_audit = transform_c7(lon, lat, refh, return_audit=True)
        audits[spec.granule_id] = {
            "sample_count": int(len(ids)),
            "sample_rule": "1,200 evenly spaced H5 record indices, then finite lon/lat/Refh filter",
            "geoid18_operation": geoid_audit,
            "c7_operation": c7_audit,
            "interpretation": "H5 geoid and NOAA GEOID18 are compared numerically at the same nominal coordinates. H5 realization/epoch and geoid field semantics remain unverified, so their difference is not interpreted as model error.",
        }
        for i in range(len(ids)):
            rows.append({
                "granule_id": spec.granule_id,
                "pulse_index": int(ids[i]),
                "longitude_deg_assumed": float(lon[i]),
                "latitude_deg_assumed": float(lat[i]),
                "refh_h5_m": float(refh[i]),
                "h5_geoid_field_m": float(h5_geoid[i]),
                "noaa_geoid18_undulation_m": float(geoid18[i]),
                "h5_geoid_minus_noaa_geoid18_m": float(h5_geoid[i] - geoid18[i]),
                "geoid_free2mean_h5_field_m": float(free[i]),
                "h5_geoid_correction_to_height_m": float(-h5_geoid[i]),
                "c7_proj_vertical_correction_m": float(c7_height[i] - refh[i]),
                "c7_minus_refh_minus_h5_geoid_m": float(c7_height[i] - (refh[i] - h5_geoid[i])),
                "c7_finite": bool(np.isfinite(c7_height[i]) and np.isfinite(geoid18[i])),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "geoid18_sample_comparison.csv", index=False)
    _write_json(output_dir / "geoid18_sample_comparison_audit.json", audits)
    return frame, audits


def _candidate_evidence(
    transform_available: bool,
    z_source_verified: bool,
) -> dict[str, dict[str, Any]]:
    return {
        "C0": {"evidence_status": "BASELINE_ONLY", "evidence_rank": 0, "provisional_eligible": True,
               "operation_complete": True, "source_frame_verified": False,
               "reason": "Stored Refh, no vertical frame evidence in the H5."},
        "C1": {"evidence_status": "DIAGNOSTIC_ONLY", "evidence_rank": 3, "provisional_eligible": True,
               "operation_complete": True, "source_frame_verified": False,
               "reason": "Minus H5 geoid is the sign-plausible scalar hypothesis, but field model/frame/units/application status are undocumented."},
        "C2": {"evidence_status": "DIAGNOSTIC_ONLY", "evidence_rank": 0, "provisional_eligible": True,
               "operation_complete": True, "source_frame_verified": False,
               "reason": "Plus-geoid sign alternative retained as a diagnostic."},
        "C3": {"evidence_status": "DIAGNOSTIC_ONLY", "evidence_rank": 2, "provisional_eligible": True,
               "operation_complete": True, "source_frame_verified": False,
               "reason": "Minus-geoid plus free-to-mean sensitivity; H5 semantics are undocumented."},
        "C4": {"evidence_status": "DIAGNOSTIC_ONLY", "evidence_rank": 1, "provisional_eligible": True,
               "operation_complete": True, "source_frame_verified": False,
               "reason": "Minus-geoid opposite free-to-mean sensitivity; H5 semantics are undocumented."},
        "C5": {"evidence_status": "DIAGNOSTIC_ONLY", "evidence_rank": 0, "provisional_eligible": True,
               "operation_complete": True, "source_frame_verified": False,
               "reason": "Plus-geoid free-to-mean sensitivity; retained as a diagnostic."},
        "C6": {"evidence_status": "DIAGNOSTIC_ONLY", "evidence_rank": 0, "provisional_eligible": True,
               "operation_complete": True, "source_frame_verified": False,
               "reason": "Plus-geoid free-to-mean sensitivity; retained as a diagnostic."},
        "C7": {"evidence_status": "PROVISIONAL_NOT_DATUM_VERIFIED", "evidence_rank": 100,
               "provisional_eligible": bool(transform_available and z_source_verified),
               "operation_complete": bool(transform_available), "source_frame_verified": False,
               "reason": "Complete no-ballpark PROJ/GEOID18 path is available and matches the 3DEP target evidence, but CASALS realization and coordinate epoch are unresolved."},
    }


def _metric_rows_for_partition(
    frame: pd.DataFrame,
    *,
    partition: str,
    stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    role_frame = frame.loc[frame.research_role.astype(str) == partition].copy()
    if role_frame.empty:
        return rows
    common = role_frame.common_support_all_candidates.to_numpy(dtype=bool)
    strata = {
        "all_valid_refh": np.ones(len(role_frame), dtype=bool),
        "snr_ge_3": role_frame.refh_snr.to_numpy(dtype=np.float64) >= 3.0,
        "snr_ge_5": role_frame.refh_snr.to_numpy(dtype=np.float64) >= 5.0,
    }
    for snr_stratum, snr_mask in strata.items():
        denominator = int(snr_mask.sum())
        keep = common & snr_mask
        for candidate_id in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            candidate_col = f"{candidate_id}_height_m"
            ref_col = "c7_reference_height_local_plane_m" if candidate_id == "C7" else "reference_height_local_plane_m"
            if candidate_col not in role_frame or ref_col not in role_frame:
                continue
            values = role_frame[candidate_col].to_numpy(dtype=np.float64)
            reference = role_frame[ref_col].to_numpy(dtype=np.float64)
            residual = values[keep] - reference[keep]
            metrics = distribution_metrics(residual)
            rows.append({
                "granule_id": str(role_frame.granule_id.iloc[0]),
                "candidate_id": candidate_id,
                "candidate_label": CANDIDATE_LABELS[candidate_id],
                "research_role": partition,
                "population_stage": stage,
                "snr_stratum": snr_stratum,
                "difference_definition": "uncorrected candidate height minus local 3DEP plane; C7 uses surface queried at transformed XY; C0-C6 use original EPSG:6347 query XY",
                "n_denominator_role_and_snr": denominator,
                "n_common_surface_support": int(keep.sum()),
                "coverage": float(keep.sum() / denominator) if denominator else 0.0,
                **metrics,
            })
    return rows


def _common_sample_columns(frame: pd.DataFrame) -> list[str]:
    base = [
        "granule_id", "pulse_index", "research_role", "spatial_split", "sweep_num", "track_num",
        "longitude_deg", "latitude_deg", "refh_snr", "easting_m_epsg6347", "northing_m_epsg6347",
        "surface_acceptable", "reference_height_local_plane_m", "reference_height_local_median_m",
        "neighbor_count_within_3m", "local_slope_deg", "ground_roughness_nmad_m",
        "geoid_h5_m", "geoid_free2mean_h5_m", "c7_easting_m_epsg6347", "c7_northing_m_epsg6347",
        "c7_height_navd88_geoid18_m", "c7_reference_height_local_plane_m",
        "c7_reference_height_local_median_m", "c7_surface_acceptable", "c7_neighbor_count",
        "c7_xy_shift_m", "common_support_all_candidates",
    ]
    return [column for column in base if column in frame.columns] + [
        f"{c}_height_m" for c in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7") if f"{c}_height_m" in frame.columns
    ]


def _candidate_manifest(
    *,
    coordinate_operations: Mapping[str, Any],
    split_audit: Mapping[str, Any],
    z_audit: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    rows = [
        {"candidate_id": "C0", "formula": "Refh", "target": "none; native stored values", "selection_class": "baseline", "evidence_status": "unresolved source vertical frame"},
        {"candidate_id": "C1", "formula": "Refh - geoid", "target": "H5 geoid field candidate", "selection_class": "diagnostic", "evidence_status": "H5 geoid model/units/frame/application unresolved"},
        {"candidate_id": "C2", "formula": "Refh + geoid", "target": "H5 geoid field candidate", "selection_class": "diagnostic", "evidence_status": "sign sensitivity only"},
        {"candidate_id": "C3", "formula": "Refh - geoid - geoid_free2mean", "target": "H5 fields", "selection_class": "diagnostic", "evidence_status": "both H5 field semantics unresolved"},
        {"candidate_id": "C4", "formula": "Refh - geoid + geoid_free2mean", "target": "H5 fields", "selection_class": "diagnostic", "evidence_status": "both H5 field semantics unresolved"},
        {"candidate_id": "C5", "formula": "Refh + geoid - geoid_free2mean", "target": "H5 fields", "selection_class": "diagnostic", "evidence_status": "both H5 field semantics unresolved"},
        {"candidate_id": "C6", "formula": "Refh + geoid + geoid_free2mean", "target": "H5 fields", "selection_class": "diagnostic", "evidence_status": "both H5 field semantics unresolved"},
        {"candidate_id": "C7", "formula": "TransformerGroup(EPSG:4979 assumption) → EPSG:6347 + EPSG:5703 using NOAA GEOID18 grid", "target": TARGET_COMPOUND, "selection_class": "provisional geodetic path", "evidence_status": "target and operation evidenced; CASALS realization/coordinate epoch unresolved"},
        {"candidate_id": "C8", "formula": "Refh - median(C0 residual on calibration_offset)", "target": "empirical relative alignment", "selection_class": "separate empirical baseline; never competes for datum selection", "evidence_status": "not a datum transformation"},
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_definitions": rows,
        "target_reference": {
            "horizontal_crs": "EPSG:6347 NAD83(2011) / UTM zone 18N",
            "vertical_crs": "EPSG:5703 NAVD88 height",
            "geoid": "GEOID18",
            "source_evidence": "USGS work-unit/project metadata and final LPC reports; source point-level identity/Z sample is in 3dep_z_provenance_audit.json.",
        },
        "source_reference": {
            "assumed_input_crs_for_C7": "EPSG:4979 WGS 84 3D ensemble",
            "source_realization": "not identified by the CASALS H5",
            "coordinate_epoch": None,
            "processing_corrections": "not fully identified by the CASALS H5",
            "status": "ASSUMED_SOURCE_FRAME",
        },
        "comparison_population": {
            "common_pulse_identity": "original H5 pulse_index",
            "support": "same frozen pulses with acceptable original and C7-requeried 0.75 m class-2 local ground plane; candidate formulas must be finite",
            "selection_split_manifest_sha256": split_audit.get("manifest_sha256"),
            "test_used_for_candidate_selection": False,
        },
        "3dep_z_provenance_status": z_audit.get("overall_status"),
        "coordinate_operations_file": "coordinate_operations.json",
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    _write_json(output_dir / "vertical_candidate_manifest.json", manifest)
    return manifest


def _set_missing_c7(frame: pd.DataFrame, reason: str) -> pd.DataFrame:
    out = frame.copy()
    for column in (
        "c7_easting_m_epsg6347", "c7_northing_m_epsg6347", "c7_height_navd88_geoid18_m",
        "C7_height_m", "c7_reference_height_local_plane_m", "c7_reference_height_local_median_m",
        "c7_xy_shift_m", "c7_nearest_ground_distance_m", "c7_local_slope_deg",
        "c7_plane_residual_nmad_m", "c7_ground_roughness_nmad_m",
    ):
        out[column] = np.nan
    for column in ("c7_surface_acceptable", "c7_plane_fit_valid"):
        out[column] = False
    out["c7_neighbor_count"] = 0
    candidate_values = {
        key: out[f"{key}_height_m"].to_numpy(dtype=np.float64)
        for key in ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
        if f"{key}_height_m" in out
    }
    common = out.surface_acceptable.to_numpy(dtype=bool) & np.isfinite(
        out.reference_height_local_plane_m.to_numpy(dtype=np.float64)
    )
    for values in candidate_values.values():
        common &= np.isfinite(values)
    out["common_support_all_candidates"] = common
    out["c7_unavailable_reason"] = reason
    return out


def _freeze_selected_configuration(
    selection: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    selection_metrics: pd.DataFrame,
    split_audit: Mapping[str, Any],
    coordinate_operations: Mapping[str, Any],
    z_audit: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    candidate = str(selection["candidate_id"])
    operation_complete = bool(evidence.get(candidate, {}).get("operation_complete", False))
    grid_info = json.loads((output_dir / "proj_grid_manifest.json").read_text(encoding="utf-8")) if (output_dir / "proj_grid_manifest.json").exists() else {"grids": []}
    config = {
        "candidate_id": candidate,
        "candidate_label": CANDIDATE_LABELS[candidate],
        "status": selection["evidence_status"],
        "source_crs": "EPSG:4979 WGS 84 3D ensemble (assumed; not verified by CASALS H5)" if candidate == "C7" else "CASALS H5 source frame unresolved",
        "source_realization": None,
        "source_coordinate_epoch": None,
        "source_frame_status": "assumed_source_frame" if candidate == "C7" else "unresolved_source_frame",
        "target_crs": TARGET_COMPOUND if candidate == "C7" else "candidate-specific H5 formula; no complete CRS claim",
        "target_horizontal": "EPSG:6347 NAD83(2011) / UTM zone 18N",
        "target_vertical": "EPSG:5703 NAVD88 height, GEOID18",
        "operation_complete": operation_complete,
        "operation_by_granule": {
            granule_id: details.get("c7", {}).get("selected_operation")
            for granule_id, details in coordinate_operations.items()
        } if candidate == "C7" else {},
        "required_grids": [
            grid for grid in grid_info.get("grids", [])
            if candidate == "C7" and grid.get("available") and grid.get("name") == GEOID18_GRID
        ],
        "selection_population": "candidate_selection spatial half of each preserved calibration ROI; only common pulse IDs with original and C7-requeried acceptable ground support",
        "selection_metrics_primary": selection["selection_metrics"],
        "selection_reason": selection["selection_reason"],
        "evidence_level": selection["evidence_status"],
        "3dep_z_provenance_status": z_audit.get("overall_status", "not verified"),
        "split_manifest_sha256": split_audit.get("manifest_sha256"),
        "test_metrics_used": False,
        "remaining_uncertainty": [
            "CASALS H5 does not identify the source geodetic realization or coordinate epoch.",
            "CASALS refh processing corrections and H5 geoid/free-to-mean field definitions are not documented in the H5.",
            "NOAA GEOID18 is defined for NAD83(2011) epoch 2010.0; no source coordinate-epoch transformation can be asserted for CASALS.",
            "3DEP clipped-Z evidence is a point-identity sample, not an audit of every clipped LAS point.",
        ],
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    config["configuration_sha256"] = _canonical_hash(config)
    _write_json(output_dir / "selected_vertical_configuration.json", config)
    return config


def _load_previous_empirical_metrics(spec: GranuleSpec) -> pd.DataFrame:
    path = spec.revised_dir / "elevation_empirical_holdout_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_final_results(
    frames: Sequence[pd.DataFrame],
    selected: Mapping[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate candidates on test only after selected config has been written."""
    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    residual_frames: list[pd.DataFrame] = []
    empirical_audit: dict[str, Any] = {
        "method": "C8 = native Refh minus the median native-Refh-minus-original-3DEP-plane residual in the frozen calibration_offset half; separate empirical baseline, not a datum conversion.",
        "metrics_use_independent_test_only": True,
        "granules": {},
    }
    strata = {
        "all_valid_refh": np.ones,
        "snr_ge_3": lambda v: v >= 3.0,
        "snr_ge_5": lambda v: v >= 5.0,
    }
    for frame in frames:
        spec = next(s for s in GRANULES if s.granule_id == str(frame.granule_id.iloc[0]))
        frame = frame.copy()
        candidate_rows.extend(_metric_rows_for_partition(frame, partition="candidate_selection", stage="selection_metrics_frozen_before_test"))
        candidate_rows.extend(_metric_rows_for_partition(frame, partition="calibration_offset", stage="postselection_diagnostic_only"))
        candidate_rows.extend(_metric_rows_for_partition(frame, partition="independent_test", stage="postselection_test_diagnostic"))

        common = frame.common_support_all_candidates.to_numpy(dtype=bool)
        calibration = frame.research_role.astype(str).to_numpy() == "calibration_offset"
        test = frame.research_role.astype(str).to_numpy() == "independent_test"
        snr = frame.refh_snr.to_numpy(dtype=np.float64)
        raw_cal_residual = frame.C0_height_m.to_numpy(dtype=np.float64) - frame.reference_height_local_plane_m.to_numpy(dtype=np.float64)
        test_frame = frame.loc[test & common].copy()
        for candidate_id in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"):
            candidate_col = f"{candidate_id}_height_m"
            ref_col = "c7_reference_height_local_plane_m" if candidate_id == "C7" else "reference_height_local_plane_m"
            if candidate_col in test_frame and ref_col in test_frame:
                test_frame[f"{candidate_id}_uncorrected_residual_m"] = (
                    test_frame[candidate_col].to_numpy(dtype=np.float64)
                    - test_frame[ref_col].to_numpy(dtype=np.float64)
                )
        empirical_offsets = {}
        for stratum, selector in strata.items():
            snr_mask = selector(snr) if stratum != "all_valid_refh" else np.ones(len(frame), dtype=bool)
            fit_mask = calibration & common & snr_mask & np.isfinite(raw_cal_residual)
            test_mask = test & common & snr_mask
            if not fit_mask.any():
                offset = None
            else:
                offset = float(np.median(raw_cal_residual[fit_mask]))
            empirical_offsets[stratum] = {
                "calibration_offset_m": offset,
                "calibration_n": int(fit_mask.sum()),
                "test_n": int(test_mask.sum()),
                "fit_rule": "median uncorrected C0 residual on calibration_offset half, common C0-C7 support, within the same SNR stratum",
            }
            if offset is not None:
                test_frame[f"C8_empirical_residual_{stratum}_m"] = (
                    test_frame.C0_height_m.to_numpy(dtype=np.float64)
                    - test_frame.reference_height_local_plane_m.to_numpy(dtype=np.float64)
                    - offset
                )
            else:
                test_frame[f"C8_empirical_residual_{stratum}_m"] = np.nan

            stratum_mask_full = test_mask
            denominator = int((test & snr_mask).sum())
            support_n = int(stratum_mask_full.sum())
            selected_col = f"{selected['candidate_id']}_uncorrected_residual_m"
            if selected_col in test_frame:
                selected_values = test_frame[selected_col].to_numpy(dtype=np.float64)
                selected_metrics = distribution_metrics(selected_values[selector(test_frame.refh_snr.to_numpy(dtype=np.float64))] if stratum != "all_valid_refh" else selected_values)
                final_rows.append({
                    "granule_id": spec.granule_id, "configuration_id": selected["candidate_id"],
                    "configuration_label": selected["candidate_label"], "configuration_status": selected["status"],
                    "snr_stratum": stratum, "n_denominator_independent_test": denominator,
                    "n_common_support": selected_metrics["n"],
                    "coverage": float(selected_metrics["n"] / denominator) if denominator else 0.0,
                    "difference_definition": "selected configuration minus 3DEP local plane; C7 reference is queried at C7 transformed XY",
                    **selected_metrics,
                })
            baseline_values = test_frame.C0_uncorrected_residual_m.to_numpy(dtype=np.float64)
            baseline_metrics = distribution_metrics(baseline_values[selector(test_frame.refh_snr.to_numpy(dtype=np.float64))] if stratum != "all_valid_refh" else baseline_values)
            final_rows.append({
                "granule_id": spec.granule_id, "configuration_id": "C0", "configuration_label": CANDIDATE_LABELS["C0"],
                "configuration_status": "native baseline", "snr_stratum": stratum,
                "n_denominator_independent_test": denominator, "n_common_support": baseline_metrics["n"],
                "coverage": float(baseline_metrics["n"] / denominator) if denominator else 0.0,
                "difference_definition": "native Refh minus original-XY 3DEP local plane",
                **baseline_metrics,
            })
            c8_values = test_frame[f"C8_empirical_residual_{stratum}_m"].to_numpy(dtype=np.float64)
            if stratum != "all_valid_refh":
                c8_snr_mask = selector(test_frame.refh_snr.to_numpy(dtype=np.float64))
                c8_values = c8_values[c8_snr_mask]
            c8_metrics = distribution_metrics(c8_values)
            final_rows.append({
                "granule_id": spec.granule_id, "configuration_id": "C8", "configuration_label": CANDIDATE_LABELS["C8"],
                "configuration_status": "EMPIRICAL_ONLY", "snr_stratum": stratum,
                "n_denominator_independent_test": denominator, "n_common_support": c8_metrics["n"],
                "coverage": float(c8_metrics["n"] / denominator) if denominator else 0.0,
                "difference_definition": "C0 residual minus calibration-only C0 median; not a datum conversion",
                **c8_metrics,
            })
        empirical_audit["granules"][spec.granule_id] = empirical_offsets
        test_frame["selected_configuration_id"] = str(selected["candidate_id"])
        test_frame["selected_configuration_status"] = str(selected["status"])
        test_frame["selected_height_m"] = test_frame[f"{selected['candidate_id']}_height_m"]
        selected_ref_col = "c7_reference_height_local_plane_m" if selected["candidate_id"] == "C7" else "reference_height_local_plane_m"
        test_frame["selected_reference_height_m"] = test_frame[selected_ref_col]
        test_frame["selected_residual_m"] = test_frame["selected_height_m"] - test_frame["selected_reference_height_m"]
        residual_frames.append(test_frame)

    candidate_metrics = pd.DataFrame(candidate_rows)
    final_metrics = pd.DataFrame(final_rows)
    residuals = pd.concat(residual_frames, ignore_index=True)
    # Historical empirical metrics are read only now, after the selected config
    # and its selection hash are already on disk.
    historical_rows = []
    for spec in GRANULES:
        old = _load_previous_empirical_metrics(spec)
        if not old.empty:
            for row in old.itertuples(index=False):
                historical_rows.append({
                    "granule_id": spec.granule_id,
                    "snr_stratum": str(row.snr_stratum),
                    "previous_full_calibration_rmse_m": float(row.rmse_m),
                    "previous_test_n": int(row.test_n),
                })
    historical = pd.DataFrame(historical_rows)
    if not historical.empty:
        final_metrics = final_metrics.merge(historical, on=["granule_id", "snr_stratum"], how="left")
        selected_rows = final_metrics.configuration_id == str(selected["candidate_id"])
        c8_rows = final_metrics.configuration_id == "C8"
        final_metrics["selected_rmse_minus_previous_empirical_m"] = np.where(
            selected_rows,
            final_metrics.rmse_m - final_metrics.previous_full_calibration_rmse_m,
            np.nan,
        )
        final_metrics["c8_rmse_minus_previous_empirical_m"] = np.where(
            c8_rows,
            final_metrics.rmse_m - final_metrics.previous_full_calibration_rmse_m,
            np.nan,
        )
        final_metrics["historical_comparison_note"] = (
            "Previous RMSE is joined by SNR stratum and used the full calibration ROI with original-XY support. Current metrics use the common C0-C7 support; C7 also queries the target surface at transformed XY. RMSE differences are descriptive, not a paired method effect."
        )
    candidate_metrics.to_csv(output_dir / "vertical_candidate_metrics.csv", index=False)
    final_metrics.to_csv(output_dir / "final_elevation_metrics.csv", index=False)
    residuals.to_parquet(output_dir / "final_elevation_residuals.parquet", index=False)
    _write_json(output_dir / "empirical_alignment_audit.json", empirical_audit)
    return candidate_metrics, final_metrics, residuals, empirical_audit


def run_datum_comparison(output_dir: Path = OUTPUT_ROOT, *, make_figures: bool = True) -> dict[str, Any]:
    """Run the bounded real-data comparison and freeze selection before test metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest, split_audit = freeze_selection_split(output_dir)
    h5_audit, h5_fields = audit_casals_height_fields(output_dir)
    proj_setup_error = None
    try:
        proj_environment = configure_private_proj_data(output_dir)
    except Exception as exc:
        proj_environment = {"environment": {}, "grid": {}}
        proj_setup_error = f"{type(exc).__name__}: {exc}"
        _write_json(output_dir / "proj_environment_audit.json", {"status": "GRID_SETUP_FAILED", "error": proj_setup_error})
    try:
        z_audit, z_samples = audit_3dep_z_provenance(output_dir)
    except Exception as exc:
        z_audit = {
            "overall_status": "POINT_LEVEL_AUDIT_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "scope_limit": "Metadata evidence is retained in source_metadata/; point-level Z preservation was not verified.",
        }
        z_samples = pd.DataFrame()
        _write_json(output_dir / "3dep_z_provenance_audit.json", z_audit)
        z_samples.to_csv(output_dir / "3dep_source_clip_z_check.csv", index=False)
    geoid_samples = pd.DataFrame()
    geoid_audits: dict[str, Any] = {}
    if not proj_setup_error:
        try:
            geoid_samples, geoid_audits = _sample_geoid_comparison(output_dir)
        except Exception as exc:
            geoid_audits = {"status": "GEOID18_SAMPLE_FAILED", "error": f"{type(exc).__name__}: {exc}"}
            _write_json(output_dir / "geoid18_sample_comparison_audit.json", geoid_audits)
            geoid_samples.to_csv(output_dir / "geoid18_sample_comparison.csv", index=False)

    frames: list[pd.DataFrame] = []
    coordinate_operations: dict[str, Any] = {}
    c7_available_by_granule: dict[str, bool] = {}
    for spec in GRANULES:
        frame = _load_matched_roles(spec, split_manifest)
        frame["granule_id"] = spec.granule_id
        try:
            frame, query_audit = _surface_query(spec, frame)
            coordinate_operations[spec.granule_id] = {"c7": query_audit["transform"], "reference_grid": query_audit["reference_grid"]}
            c7_available_by_granule[spec.granule_id] = bool(
                query_audit["transform"].get("transformer_count", 0) > 0
                and query_audit["transform"].get("finite_output_count", 0) == len(frame)
                and frame.c7_surface_acceptable.any()
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            frame = _set_missing_c7(frame, reason)
            coordinate_operations[spec.granule_id] = {"c7_status": "failed_or_unavailable", "error": reason}
            c7_available_by_granule[spec.granule_id] = False
        frames.append(frame)
    if geoid_audits:
        coordinate_operations["geoid18_grid_samples"] = geoid_audits
    _write_json(output_dir / "coordinate_operations.json", coordinate_operations)

    selection_rows: list[dict[str, Any]] = []
    common_selection_ids: dict[str, list[int]] = {}
    for frame in frames:
        selection_frame = frame.loc[
            (frame.research_role.astype(str) == "candidate_selection")
            & frame.common_support_all_candidates.astype(bool)
        ]
        common_selection_ids[str(frame.granule_id.iloc[0])] = selection_frame.pulse_index.astype(int).tolist()
        selection_rows.extend(_metric_rows_for_partition(
            frame, partition="candidate_selection", stage="preselection_candidate_selection_only"
        ))
    selection_metrics = pd.DataFrame(selection_rows)
    z_verified = z_audit.get("overall_status") == "VERIFIED_SAMPLE_PASSTHROUGH"
    transform_available = all(c7_available_by_granule.values()) and not proj_setup_error
    evidence = _candidate_evidence(transform_available, z_verified)
    selection = select_vertical_configuration(selection_metrics, evidence)
    selection["common_candidate_selection_pulse_counts"] = {k: len(v) for k, v in common_selection_ids.items()}
    selection["candidate_selection_pulse_content_sha256"] = _canonical_hash(common_selection_ids)
    selection["split_manifest_sha256"] = split_audit["manifest_sha256"]
    selection["candidate_evidence"] = evidence
    selected_config = _freeze_selected_configuration(
        selection, evidence, selection_metrics, split_audit, coordinate_operations, z_audit, output_dir
    )

    grid_manifest_path = output_dir / "proj_grid_manifest.json"
    if grid_manifest_path.exists():
        grid_manifest = json.loads(grid_manifest_path.read_text(encoding="utf-8"))
        for grid in grid_manifest.get("grids", []):
            grid["used_by_selected_operation"] = bool(selected_config["candidate_id"] == "C7" and transform_available and grid["name"] == GEOID18_GRID)
            if grid["used_by_selected_operation"]:
                grid["selected_configuration_id"] = selected_config["candidate_id"]
        _write_json(grid_manifest_path, grid_manifest)
    candidate_manifest = _candidate_manifest(
        coordinate_operations=coordinate_operations, split_audit=split_audit,
        z_audit=z_audit, output_dir=output_dir,
    )

    # Test residuals are first formed after selected_vertical_configuration.json
    # has been written. The selector received only candidate_selection metrics.
    candidate_metrics, final_metrics, residuals, empirical_audit = _write_final_results(frames, selected_config, output_dir)
    common_parts = []
    for frame in frames:
        common_parts.append(frame.loc[frame.common_support_all_candidates.astype(bool), _common_sample_columns(frame)].copy())
    common_samples = pd.concat(common_parts, ignore_index=True)
    common_samples["selected_configuration_id"] = selected_config["candidate_id"]
    common_samples["selected_configuration_status"] = selected_config["status"]
    common_samples.to_parquet(output_dir / "vertical_candidate_common_sample.parquet", index=False)
    run_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_configuration": selected_config,
        "h5_metadata_status": "ASSUMED_SOURCE_FRAME",
        "3dep_z_provenance_status": z_audit.get("overall_status"),
        "proj_setup_error": proj_setup_error,
        "candidate_selection_test_metrics_used": False,
        "selection_metric_rows": int(len(selection_metrics)),
        "candidate_metric_rows": int(len(candidate_metrics)),
        "candidate_common_sample_rows": int(len(common_samples)),
        "final_independent_test_rows": int(len(residuals)),
        "candidate_selection_common_ids": {k: len(v) for k, v in common_selection_ids.items()},
        "independent_test_common_ids": {
            str(frame.granule_id.iloc[0]): int(((frame.research_role.astype(str) == "independent_test") & frame.common_support_all_candidates.astype(bool)).sum())
            for frame in frames
        },
        "output_files": [
            "casals_height_metadata_audit.json", "casals_height_fields_summary.csv",
            "3dep_z_provenance_audit.json", "3dep_source_clip_z_check.csv",
            "proj_environment_audit.json", "proj_grid_manifest.json", "coordinate_operations.json",
            "selection_split_manifest.csv", "selection_split_manifest_audit.json",
            "vertical_candidate_manifest.json", "vertical_candidate_metrics.csv",
            "vertical_candidate_common_sample.parquet", "selected_vertical_configuration.json",
            "final_elevation_metrics.csv", "final_elevation_residuals.parquet",
            "empirical_alignment_audit.json",
        ],
    }
    _write_json(output_dir / "datum_comparison_summary.json", run_summary)
    if make_figures:
        build_datum_figures(output_dir=output_dir, geoid_samples=geoid_samples,
                            candidate_metrics=candidate_metrics, final_metrics=final_metrics,
                            residuals=residuals, selection=selected_config)
        run_summary["figures"] = sorted(p.name for p in (output_dir / "figures").glob("*.png"))
    artifact_paths = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {
            "datum_comparison_summary.json", "datum_comparison_artifact_manifest.csv"
        }
    )
    run_summary["output_files"] = artifact_paths + [
        "datum_comparison_summary.json", "datum_comparison_artifact_manifest.csv"
    ]
    _write_json(output_dir / "datum_comparison_summary.json", run_summary)
    _write_artifact_manifest(output_dir)
    return {
        "summary": run_summary,
        "h5_audit": h5_audit,
        "h5_fields": h5_fields,
        "z_audit": z_audit,
        "z_samples": z_samples,
        "proj_environment": proj_environment,
        "geoid_samples": geoid_samples,
        "coordinate_operations": coordinate_operations,
        "candidate_manifest": candidate_manifest,
        "candidate_metrics": candidate_metrics,
        "final_metrics": final_metrics,
        "residuals": residuals,
        "common_samples": common_samples,
        "selected_configuration": selected_config,
        "empirical_alignment_audit": empirical_audit,
    }


def _save_plot(fig: Any, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build_datum_figures(
    *,
    output_dir: Path = OUTPUT_ROOT,
    geoid_samples: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    residuals: pd.DataFrame,
    selection: Mapping[str, Any],
) -> list[Path]:
    """Create the report/notebook figures from saved, traceable tables."""
    import h5py
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    colors = {"C0": "#606c76", "C1": "#277da1", "C2": "#d1495b", "C3": "#43aa8b",
              "C4": "#f8961e", "C5": "#9b5de5", "C6": "#bc6c25", "C7": "#003f5c", "C8": "#777777"}
    granule_order = [s.granule_id for s in GRANULES]

    # Figure 1: frame paths and the independent evidence status.
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                               facecolor=color, edgecolor="#36454f", linewidth=1.0)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=8, wrap=True)

    box(0.35, 2.25, 2.5, 1.6, "CASALS L1B H5", "Refh + geoid fields\nRefh convention: WGS84 ellipsoidal\nrealization / epoch unresolved", "#e7eef2")
    box(4.1, 4.25, 3.15, 1.25, "C0–C6 | Scalar diagnostics", "Refh; ± geoid; ± free-to-mean\nH5 semantics are not documented", "#f4eee3")
    box(4.1, 2.4, 3.15, 1.3, "C7 | PROJ XYZ path", "Assumed EPSG:4979 → EPSG:6347 + 5703\nNOAA GEOID18 grid; no ballpark", "#dcecf4")
    box(4.1, 0.5, 3.15, 1.2, "C8 | Empirical baseline", "Subtract calibration median\nNot a datum conversion", "#eeeeee")
    box(9.3, 2.25, 3.15, 1.6, "3DEP reference", "NAD83(2011) / UTM18N\nNAVD88 height / GEOID18\nSampled clip Z preserves source Z", "#e7f1e7")
    for start, end in [((2.85, 3.2), (4.1, 4.85)), ((2.85, 3.05), (4.1, 3.05)), ((2.85, 2.7), (4.1, 1.1)), ((7.25, 3.05), (9.3, 3.05))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#36454f"))
    ax.text(6.55, 0.08, f"Selected: {selection['candidate_id']} — {selection['status']}. Numerical fit does not verify the CASALS frame.",
            ha="center", fontsize=9, weight="bold", color="#8a3b12")
    ax.set_title("Reference-frame evidence and comparison paths", fontsize=14, weight="bold", pad=10)
    produced.append(_save_plot(fig, fig_dir / "reference_frame_overview.png"))

    # Figure 2: observed H5 and GEOID18 sample values, kept as a numerical
    # comparison rather than interpreted as model error.
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for col, granule in enumerate(granule_order):
        sub = geoid_samples.loc[geoid_samples.granule_id == granule] if not geoid_samples.empty else pd.DataFrame()
        label = "11/12" if "20241112" in granule else "11/18"
        if sub.empty:
            for ax in axes[:, col]:
                ax.text(0.5, 0.5, "GEOID18 sample unavailable", ha="center", va="center", transform=ax.transAxes)
            continue
        axes[0, col].hist(sub.h5_geoid_field_m, bins=35, alpha=0.65, color="#43aa8b", label="H5 geoid")
        axes[0, col].hist(sub.noaa_geoid18_undulation_m, bins=35, alpha=0.65, color="#003f5c", label="NOAA GEOID18 N")
        axes[0, col].set(title=f"{label} | undulation samples", xlabel="Geoid undulation (m)", ylabel="Count")
        axes[0, col].legend(fontsize=8)
        differences = sub.h5_geoid_minus_noaa_geoid18_m.to_numpy(dtype=np.float64)
        difference_median = float(np.nanmedian(differences))
        half_span = max(0.003, 3.0 * float(np.nanstd(differences)))
        axes[1, col].hist(differences, bins=35, color="#f8961e", alpha=0.8)
        axes[1, col].axvline(difference_median, color="#333333", lw=1.2, ls="--",
                             label=f"Median {difference_median:.3f} m")
        axes[1, col].set_xlim(difference_median - half_span, difference_median + half_span)
        axes[1, col].set(title=f"{label} | H5 − sampled GEOID18 N", xlabel="Difference (m; zoomed to observed cluster)", ylabel="Count")
        axes[1, col].legend(fontsize=8)
    formula_ax = axes[0, 2]
    for granule, color in zip(granule_order, ("#277da1", "#d1495b")):
        sub = geoid_samples.loc[geoid_samples.granule_id == granule] if not geoid_samples.empty else pd.DataFrame()
        if not sub.empty:
            latitude_rad = np.deg2rad(sub.latitude_deg_assumed.to_numpy(dtype=np.float64))
            expected = 0.1287 - 0.3848 * np.sin(latitude_rad) ** 2
            residual_mm = (sub.geoid_free2mean_h5_field_m.to_numpy(dtype=np.float64) - expected) * 1000.0
            short = "11/12" if "20241112" in granule else "11/18"
            formula_ax.hist(residual_mm, bins=35, alpha=0.5, color=color, label=short)
    formula_ax.axvline(0.0, color="#333333", lw=1)
    formula_ax.set(title="H5 free-to-mean − formula", xlabel="Difference (mm)", ylabel="Count")
    formula_ax.legend(title="Granule", fontsize=8)
    formula_ax.text(0.03, 0.03, "Numerical correspondence only; H5 semantics remain undocumented.",
                    transform=formula_ax.transAxes, fontsize=7.5, va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb"})
    # Compare the complete PROJ vertical correction with the H5 geoid scalar.
    corr_ax = axes[1, 2]
    corr_ax.clear()
    for granule, color in zip(granule_order, ("#277da1", "#d1495b")):
        sub = geoid_samples.loc[geoid_samples.granule_id == granule] if not geoid_samples.empty else pd.DataFrame()
        if not sub.empty:
            short = "11/12" if "20241112" in granule else "11/18"
            corr_ax.hist(sub.c7_proj_vertical_correction_m, bins=35, alpha=0.45, color=color, label=f"C7 {short}")
            corr_ax.hist(-sub.h5_geoid_field_m, bins=35, histtype="step", color=color, lw=1.2, label=f"−H5 geoid {short}")
    corr_ax.set(title="C7 vs scalar correction", xlabel="Vertical correction to Refh (m)", ylabel="Count")
    corr_ax.legend(fontsize=7)
    fig.suptitle("H5 geoid and free-to-mean fields versus GEOID18 samples", fontsize=14, weight="bold")
    produced.append(_save_plot(fig, fig_dir / "geoid_correction_comparison.png"))

    # Figure 3: candidate metrics on selection-site SNR>=3 data only.
    selection_metrics = candidate_metrics.loc[
        (candidate_metrics.research_role == "candidate_selection")
        & (candidate_metrics.snr_stratum == "snr_ge_3")
    ]
    candidates = [f"C{i}" for i in range(8)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    width = 0.36
    xloc = np.arange(len(candidates))
    for gi, granule in enumerate(granule_order):
        sub = selection_metrics.loc[selection_metrics.granule_id == granule].set_index("candidate_id")
        vals_bias = [float(sub.loc[c, "bias_m"]) if c in sub.index and pd.notna(sub.loc[c, "bias_m"]) else np.nan for c in candidates]
        vals_rmse = [float(sub.loc[c, "rmse_m"]) if c in sub.index and pd.notna(sub.loc[c, "rmse_m"]) else np.nan for c in candidates]
        label = "11/12" if "20241112" in granule else "11/18"
        axes[0].bar(xloc + (gi - 0.5) * width, vals_bias, width, label=label,
                    color=("#277da1", "#d1495b")[gi], alpha=0.86)
        axes[1].bar(xloc + (gi - 0.5) * width, vals_rmse, width, label=label,
                    color=("#277da1", "#d1495b")[gi], alpha=0.86)
    for ax in axes:
        ax.set_xticks(xloc, candidates)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(title="Granule")
    axes[0].axhline(0, color="#333333", lw=0.8)
    axes[0].set(ylabel="Uncorrected residual bias (m)", title="Bias")
    axes[1].set(ylabel="Uncorrected RMSE (m)", title="RMSE")
    fig.suptitle("Candidate comparison on frozen candidate-selection pulses (SNR ≥ 3)", fontsize=13.5, weight="bold")
    produced.append(_save_plot(fig, fig_dir / "candidate_bias_rmse_comparison.png"))

    # Figure 4: family-centered core windows preserve readable distributions.
    # Every sample remains in the denominator and out-of-window counts are
    # printed in the legend so the full tails remain visible in the figure.
    common_path = output_dir / "vertical_candidate_common_sample.parquet"
    common_samples = pd.read_parquet(common_path) if common_path.exists() else pd.DataFrame()
    families = [("Native baseline", ["C0"]), ("Minus-geoid candidates", ["C1", "C3", "C4", "C7"]),
                ("Plus-geoid diagnostics", ["C2", "C5", "C6"])]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for row_idx, granule in enumerate(granule_order):
        sub = common_samples.loc[
            (common_samples.granule_id == granule)
            & (common_samples.research_role == "candidate_selection")
            & (common_samples.refh_snr >= 3.0)
        ]
        for col, (family, cands) in enumerate(families):
            ax = axes[row_idx, col]
            candidate_residuals = {}
            for candidate_id in cands:
                height_col = f"{candidate_id}_height_m"
                ref_col = "c7_reference_height_local_plane_m" if candidate_id == "C7" else "reference_height_local_plane_m"
                if height_col in sub and ref_col in sub:
                    residual = sub[height_col].to_numpy(dtype=np.float64) - sub[ref_col].to_numpy(dtype=np.float64)
                    residual = residual[np.isfinite(residual)]
                    candidate_residuals[candidate_id] = residual
            pooled = np.concatenate(list(candidate_residuals.values())) if candidate_residuals else np.array([])
            center = float(np.median(pooled)) if pooled.size else 0.0
            half_span = 5.0
            bins = np.linspace(center - half_span, center + half_span, 71)
            for candidate_id, residual in candidate_residuals.items():
                visible = residual[(residual >= center - half_span) & (residual <= center + half_span)]
                outside = int(len(residual) - len(visible))
                if visible.size:
                    ax.hist(visible, bins=bins, histtype="step", lw=1.5, color=colors[candidate_id],
                            label=f"{candidate_id} (n={len(residual):,}; outside view={outside:,})")
                else:
                    ax.plot([], [], color=colors[candidate_id],
                            label=f"{candidate_id} (n={len(residual):,}; outside view={outside:,})")
            if center - half_span <= 0.0 <= center + half_span:
                ax.axvline(0, color="#333333", lw=0.8, label="zero residual")
            ax.axvline(center, color="#777777", lw=0.8, ls=":")
            ax.set_xlim(center - half_span, center + half_span)
            ax.set(title=f"{'11/12' if '20241112' in granule else '11/18'} | {family}",
                   xlabel="Uncorrected candidate − reference (m; ±5 m around family median)", ylabel="Count")
            ax.legend(fontsize=6.5)
    fig.suptitle("Candidate residual cores on common pulses; all tail counts retained in legends", fontsize=13.5, weight="bold")
    produced.append(_save_plot(fig, fig_dir / "candidate_residual_distributions.png"))

    # Figure 5: frozen independent test—selected, native baseline, and separate C8.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    primary = final_metrics.loc[final_metrics.snr_stratum == "snr_ge_3"]
    configurations = [str(selection["candidate_id"]), "C0", "C8"]
    labels = ["Selected", "Native Refh", "C8 empirical"]
    metric_names = ["bias_m", "rmse_m", "nmad_m"]
    xloc = np.arange(len(metric_names)); width = 0.25
    for gi, granule in enumerate(granule_order):
        sub = primary.loc[primary.granule_id == granule].set_index("configuration_id")
        for ci, cid in enumerate(configurations):
            if cid not in sub.index:
                continue
            vals = [float(sub.loc[cid, m]) if pd.notna(sub.loc[cid, m]) else np.nan for m in metric_names]
            axes[0].bar(xloc + (ci - 1) * width + gi * 0.03, vals, width / 2,
                        color=colors.get(cid, "#999999"), label=f"{'11/12' if '20241112' in granule else '11/18'} {labels[ci]}")
    axes[0].set_xticks(xloc, ["Bias", "RMSE", "NMAD"])
    axes[0].set_ylabel("Residual metric (m)"); axes[0].set_title("Independent-test SNR ≥ 3")
    axes[0].axhline(0, color="#333333", lw=0.7); axes[0].grid(axis="y", alpha=0.2); axes[0].legend(fontsize=7, ncol=2)
    thresholds = np.linspace(0, 5, 101)
    for gi, granule in enumerate(granule_order):
        sub = residuals.loc[residuals.granule_id == granule]
        sub = sub.loc[sub.refh_snr >= 3.0]
        for cid, col, name in [(str(selection["candidate_id"]), colors.get(str(selection["candidate_id"]), "#003f5c"), "Selected"),
                               ("C0", colors["C0"], "Native Refh"), ("C8", colors["C8"], "C8 empirical")]:
            rcol = "selected_residual_m" if cid == str(selection["candidate_id"]) else (
                "C0_uncorrected_residual_m" if cid == "C0" else "C8_empirical_residual_snr_ge_3_m"
            )
            if rcol not in sub:
                continue
            values = np.abs(sub[rcol].to_numpy(dtype=np.float64))
            values = np.sort(values[np.isfinite(values)])
            if not len(values):
                continue
            exceed = np.searchsorted(values, thresholds, side="right")
            frac = 1.0 - exceed / len(values)
            axes[1].plot(thresholds, frac, color=col, linestyle=("-", "--")[gi],
                         label=f"{'11/12' if '20241112' in granule else '11/18'} {name}")
    axes[1].set(xlabel="Absolute residual threshold (m)", ylabel="Fraction exceeding threshold", title="Same-pulse exceedance curves")
    axes[1].set_ylim(0, 1.02); axes[1].grid(alpha=0.2); axes[1].legend(fontsize=7)
    fig.suptitle("Selected configuration versus native baseline and empirical alignment", fontsize=13.5, weight="bold")
    produced.append(_save_plot(fig, fig_dir / "selected_vs_baseline.png"))

    # Figure 6: all common independent-test pulses remain on the map; colors are
    # saturated at ±2 m for readability and gross points are outlined, not dropped.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, granule in zip(axes, granule_order):
        sub = residuals.loc[residuals.granule_id == granule]
        x = sub.c7_easting_m_epsg6347.to_numpy(dtype=np.float64) if selection["candidate_id"] == "C7" else sub.easting_m_epsg6347.to_numpy(dtype=np.float64)
        y = sub.c7_northing_m_epsg6347.to_numpy(dtype=np.float64) if selection["candidate_id"] == "C7" else sub.northing_m_epsg6347.to_numpy(dtype=np.float64)
        r = sub.selected_residual_m.to_numpy(dtype=np.float64)
        sc = ax.scatter(x, y, c=r, s=8, cmap="coolwarm", vmin=-2.0, vmax=2.0, rasterized=True, linewidths=0)
        gross = np.isfinite(r) & (np.abs(r) > 2.0)
        if gross.any():
            ax.scatter(x[gross], y[gross], facecolors="none", edgecolors="#111111", s=22, linewidths=0.4,
                       label=f"|residual| > 2 m ({int(gross.sum()):,})")
            ax.legend(fontsize=7, loc="best")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(title=f"{'11/12' if '20241112' in granule else '11/18'} independent test | n={len(sub):,}",
               xlabel="Easting (m)", ylabel="Northing (m)")
        fig.colorbar(sc, ax=ax, label="Selected residual (m; colors capped at ±2 m)")
    fig.suptitle("Independent-test spatial residuals; gross-error points retained", fontsize=13.5, weight="bold")
    produced.append(_save_plot(fig, fig_dir / "final_residual_spatial_maps.png"))

    # Figure 7: residual versus measured waveform SNR and absolute-error ECDF.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for col, granule in enumerate(granule_order):
        sub = residuals.loc[residuals.granule_id == granule]
        snr = sub.refh_snr.to_numpy(dtype=np.float64)
        err = sub.selected_residual_m.to_numpy(dtype=np.float64)
        ax = axes[0, col]
        ax.scatter(snr, err, c=np.abs(err), s=7, cmap="magma", alpha=0.55, rasterized=True)
        ax.axhline(0, color="#333333", lw=0.8); ax.axhline(2, color="#777777", ls="--", lw=0.8); ax.axhline(-2, color="#777777", ls="--", lw=0.8)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set(title=f"{'11/12' if '20241112' in granule else '11/18'} | all common test pulses (n={len(sub):,})",
               xlabel="H5 refh_snr", ylabel="Selected residual (m; symlog)")
        ax.grid(alpha=0.2)
        ax = axes[1, col]
        for mask, label, color in [(np.ones(len(sub), bool), "all valid Refh", "#777777"),
                                   (snr >= 3.0, "SNR ≥ 3", "#277da1"), (snr >= 5.0, "SNR ≥ 5", "#d1495b")]:
            vals = np.sort(np.abs(err[mask & np.isfinite(err)]))
            if len(vals):
                ax.step(vals, np.arange(1, len(vals) + 1) / len(vals), where="post", label=f"{label} (n={len(vals):,})", color=color)
        ax.set(xlabel="Absolute selected residual (m)", ylabel="Empirical CDF", title="Error body and tail")
        ax.set_xlim(left=0); ax.grid(alpha=0.2); ax.legend(fontsize=7)
    fig.suptitle("Independent-test error by H5 SNR; no gross-error filtering", fontsize=13.5, weight="bold")
    produced.append(_save_plot(fig, fig_dir / "final_error_snr.png"))

    # Figure 8: a traceable independent-test ground profile plus two raw RX
    # waveforms selected by SNR, never by residual size.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
    for spec in GRANULES:
        sub = common_samples.loc[
            (common_samples.granule_id == spec.granule_id)
            & (common_samples.research_role == "independent_test")
            & common_samples.common_support_all_candidates.astype(bool)
        ].copy()
        if sub.empty:
            continue
        track_counts = sub.assign(_supported_snr=(sub.refh_snr >= 3.0)).groupby("track_num")._supported_snr.sum()
        track = int(track_counts.sort_values(ascending=False).index[0])
        profile = sub.loc[
            (sub.track_num.astype(int) == track) & (sub.refh_snr >= 3.0)
        ].sort_values(["sweep_num", "pulse_index"]).copy()
        px = profile.c7_easting_m_epsg6347.to_numpy(dtype=np.float64) if selection["candidate_id"] == "C7" else profile.easting_m_epsg6347.to_numpy(dtype=np.float64)
        py = profile.c7_northing_m_epsg6347.to_numpy(dtype=np.float64) if selection["candidate_id"] == "C7" else profile.northing_m_epsg6347.to_numpy(dtype=np.float64)
        profile["distance_m"] = np.r_[0.0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
        reference_col = "c7_reference_height_local_plane_m" if selection["candidate_id"] == "C7" else "reference_height_local_plane_m"
        height_col = f"{selection['candidate_id']}_height_m"
        if len(granule_order) == 2:
            ax = axes[granule_order.index(spec.granule_id)]
        else:
            ax = axes[0]
        ax.plot(profile.distance_m, profile[reference_col], color="#e09f3e", lw=1.1, label="3DEP class-2 local plane")
        ax.plot(profile.distance_m, profile[height_col], color="#003f5c", lw=1.0, label=f"{selection['candidate_id']} CASALS height")
        ax.scatter(profile.distance_m, profile[height_col], c=np.where(profile.refh_snr >= 3.0, "#277da1", "#b23a48"), s=8, alpha=0.6)
        ax.set(title=f"{'11/12' if '20241112' in spec.granule_id else '11/18'} test track {track} | SNR ≥ 3 (n={len(profile):,})",
               xlabel="Cumulative transformed-XY distance (m)", ylabel="Height (m)")
        ax.grid(alpha=0.2); ax.legend(fontsize=7)
        high_pool = profile.loc[profile.refh_snr >= 5.0]
        moderate_pool = profile.loc[(profile.refh_snr >= 3.0) & (profile.refh_snr < 5.0)]
        low_pool = profile.loc[profile.refh_snr < 3.0]
        chosen_waveforms = []
        if not high_pool.empty:
            chosen_waveforms.append(("High SNR", high_pool.sort_values(["refh_snr", "pulse_index"], ascending=[False, True]).iloc[0]))
        if not moderate_pool.empty:
            chosen_waveforms.append(("Moderate SNR", moderate_pool.assign(_d=(moderate_pool.refh_snr - 4.0).abs()).sort_values(["_d", "pulse_index"]).iloc[0]))
        elif not low_pool.empty:
            chosen_waveforms.append(("Low SNR", low_pool.sort_values(["refh_snr", "pulse_index"]).iloc[0]))
        for waveform_label, row in chosen_waveforms:
            # Use the second row of axes for two granule waveforms by default.
            wave_ax = axes[2] if len(chosen_waveforms) == 1 else axes[2]
            pulse = int(row.pulse_index)
            with h5py.File(spec.h5_path, "r") as h5:
                wave = np.asarray(h5["rx_waveform"][pulse, :], dtype=np.float64)
                bg = float(h5["bg_mean"][pulse])
            wave_ax.plot(np.arange(len(wave)), wave, lw=0.75,
                         label=f"{'11/12' if '20241112' in spec.granule_id else '11/18'} {waveform_label}: pulse {pulse}, SNR {row.refh_snr:.2f}")
            wave_ax.axhline(bg, color="#888888", ls=":", lw=0.6)
    axes[2].set(title="Representative raw RX waveforms", xlabel="RX bin (not an official Refh bin)", ylabel="RX amplitude (counts)")
    axes[2].legend(fontsize=7); axes[2].grid(alpha=0.2)
    fig.suptitle(f"Real independent-test profiles and pulses | {selection['candidate_id']} ({selection['status']})\nTrack/waveform selection uses support and SNR, not residual magnitude.", fontsize=13.2, weight="bold")
    produced.append(_save_plot(fig, fig_dir / "final_ground_profile.png"))
    return produced




def _load_matched_roles(spec: GranuleSpec, split_manifest: pd.DataFrame) -> pd.DataFrame:
    """Load only frozen ROI pulse rows and verify their H5 pulse values."""
    import h5py

    roles = split_manifest.loc[split_manifest.granule_id == spec.granule_id].copy()
    wanted = set(roles.pulse_index.astype(np.int64))
    cols = [
        "pulse_index", "sweep_num", "track_num", "longitude_deg", "latitude_deg",
        "casals_refh_m", "refh_snr", "easting_m_epsg6347", "northing_m_epsg6347",
        "reference_height_local_plane_m", "reference_height_local_median_m",
        "raw_difference_plane_m", "neighbor_count_within_3m", "robust_neighbor_count",
        "nearest_ground_distance_m", "local_slope_deg", "plane_residual_nmad_m",
        "ground_roughness_nmad_m", "surface_acceptable", "spatial_split",
    ]
    matched = pd.read_parquet(spec.revised_dir / "matched_casals_3dep_points.parquet", columns=cols)
    matched = matched.loc[matched.pulse_index.astype(np.int64).isin(wanted)].copy()
    matched["pulse_index"] = matched.pulse_index.astype(np.int64)
    matched = matched.merge(
        roles[["pulse_index", "research_role", "fixed_before_candidate_results"]],
        on="pulse_index", how="inner", validate="one_to_one",
    )
    if len(matched) != len(roles):
        missing = sorted(wanted - set(matched.pulse_index.astype(np.int64)))[:10]
        raise RuntimeError(f"Frozen pulse identities do not all exist in {spec.granule_id}; examples={missing}.")
    ids = matched.pulse_index.to_numpy(dtype=np.int64)
    with h5py.File(spec.h5_path, "r") as h5:
        refh = np.asarray(h5["refh"][ids], dtype=np.float64)
        geoid = np.asarray(h5["geoid"][ids], dtype=np.float64) if "geoid" in h5 else None
        free = np.asarray(h5["geoid_free2mean"][ids], dtype=np.float64) if "geoid_free2mean" in h5 else None
        lon = np.asarray(h5["refh_longitude"][ids], dtype=np.float64)
        lat = np.asarray(h5["refh_latitude"][ids], dtype=np.float64)
        snr = np.asarray(h5["refh_snr"][ids], dtype=np.float64)
        if not np.allclose(matched.casals_refh_m, refh, rtol=0.0, atol=1e-9):
            raise RuntimeError(f"Refh mismatch against current H5 pulse indices for {spec.granule_id}.")
        if not np.allclose(matched.longitude_deg, lon, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"Longitude mismatch against current H5 pulse indices for {spec.granule_id}.")
        if not np.allclose(matched.latitude_deg, lat, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"Latitude mismatch against current H5 pulse indices for {spec.granule_id}.")
        if not np.allclose(matched.refh_snr, snr, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"SNR mismatch against current H5 pulse indices for {spec.granule_id}.")
    candidate_map, warnings = build_height_candidates(refh, geoid, free)
    for candidate_id, values in candidate_map.items():
        matched[f"{candidate_id}_height_m"] = values
    matched["geoid_h5_m"] = geoid if geoid is not None else np.nan
    matched["geoid_free2mean_h5_m"] = free if free is not None else np.nan
    matched["longitude_deg"] = lon
    matched["latitude_deg"] = lat
    matched["refh_snr"] = snr
    matched["candidate_warnings"] = "; ".join(warnings)
    return matched


def build_local_class2_query_grid(
    laz_path: Path,
    query_xy: np.ndarray,
    *,
    resolution_m: float = 0.75,
    validation_stride: int = 10_000,
    chunk_size: int = 1_000_000,
    neighbor_radius_m: float = 3.6,
    return_validation_points: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    """Stream class-2 LAZ records and grid only cells near fixed C7 queries.

    The cell origin, record-index validation holdout, and resolution match the
    existing full reference-grid builder. The retained point set is local to the
    frozen ROI queries, avoiding a second multi-million-cell grid.
    """
    import laspy
    from scipy.spatial import cKDTree

    query = np.asarray(query_xy, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 2 or not len(query) or not np.all(np.isfinite(query)):
        raise ValueError("query_xy must be a non-empty finite (n, 2) array.")
    if resolution_m <= 0 or validation_stride < 2 or chunk_size < 1 or neighbor_radius_m <= 0:
        raise ValueError("Invalid query-grid parameters.")
    query_tree = cKDTree(query)
    xs, ys, zs = [], [], []
    validation_xs, validation_ys, validation_zs, validation_indices = [], [], [], []
    training_n = validation_n = scan_n = 0
    with laspy.open(laz_path) as reader:
        header = reader.header
        crs = header.parse_crs()
        if crs is None or crs.to_epsg() != TARGET_HORIZONTAL_EPSG:
            raise ValueError(f"Expected horizontal EPSG:{TARGET_HORIZONTAL_EPSG} coordinates, got {crs}.")
        x0 = float(np.floor(float(header.mins[0]) / resolution_m) * resolution_m)
        y0 = float(np.floor(float(header.mins[1]) / resolution_m) * resolution_m)
        point_offset = 0
        for chunk in reader.chunk_iterator(chunk_size):
            size = len(chunk)
            record_index = point_offset + np.arange(size, dtype=np.int64)
            point_offset += size
            scan_n += size
            classification = np.asarray(chunk.classification)
            keep = classification == 2
            if not keep.any():
                continue
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            keep &= np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            candidate_idx = np.flatnonzero(keep)
            if not candidate_idx.size:
                continue
            point_xy = np.column_stack((x[candidate_idx], y[candidate_idx]))
            distance, _ = query_tree.query(point_xy, k=1, distance_upper_bound=neighbor_radius_m, workers=1)
            near = np.isfinite(distance)
            if not near.any():
                continue
            candidate_idx = candidate_idx[near]
            is_validation = (record_index[candidate_idx] % validation_stride) == 0
            validation_n += int(is_validation.sum())
            if return_validation_points and is_validation.any():
                validation_idx = candidate_idx[is_validation]
                validation_xs.append(x[validation_idx])
                validation_ys.append(y[validation_idx])
                validation_zs.append(z[validation_idx])
                validation_indices.append(record_index[validation_idx])
            candidate_idx = candidate_idx[~is_validation]
            if candidate_idx.size:
                xs.append(x[candidate_idx]); ys.append(y[candidate_idx]); zs.append(z[candidate_idx])
                training_n += int(candidate_idx.size)
    if not training_n:
        raise RuntimeError(f"No class-2 reference points found near C7 queries in {laz_path.name}.")
    x = np.concatenate(xs); y = np.concatenate(ys); z = np.concatenate(zs)
    ix = np.floor((x - x0) / resolution_m).astype(np.int64)
    iy = np.floor((y - y0) / resolution_m).astype(np.int64)
    key = np.column_stack((ix, iy))
    unique_key, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=z, minlength=len(unique_key))
    ground_xy = np.column_stack((x0 + (unique_key[:, 0] + 0.5) * resolution_m,
                                 y0 + (unique_key[:, 1] + 0.5) * resolution_m))
    ground_z = sums / counts
    metadata = {
        "source_laz": laz_path.name,
        "source_laz_size_bytes": int(laz_path.stat().st_size),
        "las_point_count_streamed": int(scan_n),
        "class2_training_points_near_queries": int(training_n),
        "held_out_class2_validation_points_near_queries": int(validation_n),
        "grid_cell_count_near_queries": int(len(ground_z)),
        "grid_resolution_m": float(resolution_m),
        "validation_stride_las_records": int(validation_stride),
        "query_search_radius_m": float(neighbor_radius_m),
        "crs": crs.to_string(),
        "held_out_validation_points_returned": bool(return_validation_points),
        "rule": "Same 0.75 m cell origin and class-2 record-index holdout as existing reference builder; retain only ground points within 3.6 m of any frozen transformed query.",
    }
    if not return_validation_points:
        return ground_xy, ground_z, metadata
    held_out = pd.DataFrame({
        "easting_m_epsg6347": np.concatenate(validation_xs) if validation_xs else np.array([], dtype=np.float64),
        "northing_m_epsg6347": np.concatenate(validation_ys) if validation_ys else np.array([], dtype=np.float64),
        "elevation_m": np.concatenate(validation_zs) if validation_zs else np.array([], dtype=np.float64),
        "source_las_record_index": np.concatenate(validation_indices) if validation_indices else np.array([], dtype=np.int64),
    })
    metadata["held_out_validation_point_count"] = int(len(held_out))
    return ground_xy, ground_z, metadata, held_out


def _surface_query(
    spec: GranuleSpec,
    frame: pd.DataFrame,
    *,
    return_validation_points: bool = False,
    validation_query_xy: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]] | tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    """Requery the same 3DEP class-2 reference at C7-transformed XY."""
    from research.elevation_comparison import interpolate_ground_surface

    x, y, z, transform_audit = transform_c7(
        frame.longitude_deg.to_numpy(dtype=np.float64),
        frame.latitude_deg.to_numpy(dtype=np.float64),
        frame.casals_refh_m.to_numpy(dtype=np.float64),
        return_audit=True,
    )
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    query = np.column_stack([x, y])
    grid_query = query
    if validation_query_xy is not None:
        extra_query = np.asarray(validation_query_xy, dtype=np.float64)
        if extra_query.ndim != 2 or extra_query.shape[1] != 2 or not np.all(np.isfinite(extra_query)):
            raise ValueError("validation_query_xy must be a finite (n, 2) array.")
        if len(extra_query):
            grid_query = np.vstack([query, extra_query])
    grid_result = build_local_class2_query_grid(
        spec.laz_path, grid_query, return_validation_points=return_validation_points,
    )
    if return_validation_points:
        ground_xy, ground_z, grid_meta, held_out = grid_result
    else:
        ground_xy, ground_z, grid_meta = grid_result
    surface = interpolate_ground_surface(query, ground_xy, ground_z)
    out = frame.copy()
    out["c7_easting_m_epsg6347"] = x
    out["c7_northing_m_epsg6347"] = y
    out["c7_height_navd88_geoid18_m"] = z
    out["C7_height_m"] = z
    out["c7_reference_height_local_plane_m"] = surface["reference_height_plane_m"]
    out["c7_reference_height_local_median_m"] = surface["reference_height_median_m"]
    out["c7_surface_acceptable"] = surface["surface_acceptable"]
    out["c7_plane_fit_valid"] = surface["plane_fit_valid"]
    out["c7_neighbor_count"] = surface["neighbor_count"]
    out["c7_nearest_ground_distance_m"] = surface["nearest_ground_distance_m"]
    out["c7_local_slope_deg"] = surface["local_slope_deg"]
    out["c7_plane_residual_nmad_m"] = surface["plane_residual_nmad_m"]
    out["c7_ground_roughness_nmad_m"] = surface["ground_roughness_nmad_m"]
    out["c7_xy_shift_m"] = np.hypot(
        out.c7_easting_m_epsg6347.to_numpy() - out.easting_m_epsg6347.to_numpy(),
        out.c7_northing_m_epsg6347.to_numpy() - out.northing_m_epsg6347.to_numpy(),
    )
    common = common_support_mask(
        {k: out[f"{k}_height_m"].to_numpy(dtype=np.float64) for k in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7") if f"{k}_height_m" in out},
        out.surface_acceptable.to_numpy(dtype=bool),
        out.c7_surface_acceptable.to_numpy(dtype=bool),
        out.reference_height_local_plane_m.to_numpy(dtype=np.float64),
        out.c7_reference_height_local_plane_m.to_numpy(dtype=np.float64),
    )
    out["common_support_all_candidates"] = common
    grid_meta["c7_query_count"] = int(len(out))
    grid_meta["additional_validation_query_count"] = int(max(0, len(grid_query) - len(query)))
    grid_meta["c7_surface_acceptable_count"] = int(out.c7_surface_acceptable.sum())
    grid_meta["common_all_candidate_support_count"] = int(common.sum())
    audit = {"transform": transform_audit, "reference_grid": grid_meta}
    if return_validation_points:
        return out, audit, held_out, ground_xy, ground_z
    return out, audit


