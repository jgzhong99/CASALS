"""Explicit WGS84/ITRF -> NAD83(2011) frame hypotheses for CASALS heights.

The CASALS H5 files do not declare the GNSS/INS realization or coordinate epoch.
Instead of silently using the PROJ no-op (WGS84 ensemble -> NAD83(2011)), each
physically plausible source frame is transformed with the published
time-dependent 3-D Helmert (installed EPSG/PROJ operation), evaluated at the
flight (observation) epoch, to NAD83(2011) epoch 2010.0, followed by GEOID18.
Nothing here is estimated from 3DEP; 3DEP only evaluates the result afterwards.
The hypotheses are mutually exclusive *assumptions*, not corrections.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from pyproj.transformer import TransformerGroup

from research.reference import vertical_datum_comparison as datum
from research.elevation_comparison import interpolate_ground_surface
from research.reference.refh_correction_audit import residual_metrics

STRATA = {"all_valid_refh": None, "snr_ge_3": 3., "snr_ge_5": 5.}
TARGET_EPOCH = 2010.0  # GEOID18 input requirement (NAD83(2011) epoch 2010.0)

# source EPSG (3-D geographic), label, epoch rule
HYPOTHESES = {
    "H1_WGS84_G1762_obs_epoch": (9057, "obs"),  # EPSG defines WGS84 realizations as 2-D CRS; h passes through
    "H2_ITRF2014_obs_epoch": (7912, "obs"),
    "H3_ITRF2020_obs_epoch": (9989, "obs"),
    "H4_ITRF2014_epoch_2010": (7912, 2010.0),
}


def observation_epoch(granule_id: str) -> float:
    """Decimal-year of the granule start (id: casals_l1b_YYYYMMDDThhmmss_...)."""
    t = datetime.strptime(granule_id[11:26], "%Y%m%dT%H%M%S")
    y0, y1 = datetime(t.year, 1, 1), datetime(t.year + 1, 1, 1)
    return t.year + (t - y0) / (y1 - y0)


def frame_transformer(source_epsg: int):
    """Available, non-ballpark, non-noop operation source CRS -> NAD83(2011) (EPSG:6319)."""
    group = TransformerGroup(CRS.from_epsg(source_epsg), CRS.from_epsg(6319),
                             always_xy=True, allow_ballpark=False)
    t = datum.choose_available_non_ballpark_operation(group)
    if "proj=noop" in t.definition:
        raise RuntimeError(f"EPSG:{source_epsg}->6319 resolved to a no-op")
    return t


def apply_frame(source_epsg, epoch, lon, lat, h):
    t = frame_transformer(source_epsg)
    ep = np.full(len(lon), float(epoch))
    flon, flat, fh, _ = t.transform(lon, lat, h, ep)
    # strict 3-D inverse round trip (same epoch)
    blon, blat, bh, _ = t.transform(flon, flat, fh, ep, direction="INVERSE")
    rt_h = float(np.max(np.abs(bh - h)))
    rt_xy = float(np.max(np.hypot((blon - lon) * 111e3 * np.cos(np.radians(lat)), (blat - lat) * 111e3)))
    assert rt_h < 1e-6 and rt_xy < 1e-6, (rt_h, rt_xy)
    info = dict(operation=t.description, definition=t.definition, accuracy_metadata_m=t.accuracy,
                roundtrip_max_height_m=rt_h, roundtrip_max_horizontal_m=rt_xy)
    return np.asarray(flon), np.asarray(flat), np.asarray(fh), info


def _metrics(frame, residual_col, support):
    rows = []
    for stratum, thr in STRATA.items():
        elig = pd.Series(True, index=frame.index) if thr is None else frame.refh_snr.ge(thr)
        grp = frame.loc[elig & support]
        v = grp[residual_col].to_numpy(float)
        rows.append(dict(stratum=stratum, eligible_n=int(elig.sum()),
                         coverage=len(grp) / elig.sum() if elig.any() else 0., **residual_metrics(v)))
    return rows


def run_frame_hypotheses(R, output):
    """Evaluate each hypothesis on the frozen R['population'] pulses (field + road)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pop = R["population"]
    metric_rows, shift_rows, op_rows = [], [], []
    for spec in datum.GRANULES:
        g = pop.loc[pop.granule_id.eq(spec.granule_id)].reset_index(drop=True)
        lon, lat, h = (g[c].to_numpy(float) for c in ("longitude_deg", "latitude_deg", "casals_refh_m"))
        base_H = g["GEOID18_height_m"].to_numpy(float)
        base_xy = g[["c7_easting_m_epsg6347", "c7_northing_m_epsg6347"]].to_numpy(float)
        base_ref = g["reference_height_m"].to_numpy(float)
        for name, (epsg, ep) in HYPOTHESES.items():
            epoch = observation_epoch(spec.granule_id) if ep == "obs" else ep
            flon, flat, fh, info = apply_frame(epsg, epoch, lon, lat, h)
            # NAD83(2011) ellipsoidal h (epoch 2010.0) -> NAVD88 via GEOID18; UTM 18N (EPSG:6347)
            x, y, H, _ = datum.transform_c7(flon, flat, fh, source_epsg=6319)
            query = np.column_stack([x, y])
            xy, z, _ = datum.build_local_class2_query_grid(spec.laz_path, query)
            surf = interpolate_ground_surface(query, xy, z)
            ref = np.asarray(surf["reference_height_plane_m"], float)
            ok = (g.common_support_all_candidates.to_numpy(bool) & np.asarray(surf["surface_acceptable"], bool)
                  & np.isfinite(ref) & np.isfinite(H))
            f = g.assign(H=H, ref=ref, residual=H - ref)
            cos = np.cos(np.radians(lat))
            shift_rows.append(dict(
                granule_id=spec.granule_id, hypothesis=name, epoch=epoch,
                delta_h_ellipsoidal_median_m=float(np.median(fh - h)),
                delta_h_ellipsoidal_min_m=float(np.min(fh - h)), delta_h_ellipsoidal_max_m=float(np.max(fh - h)),
                delta_east_median_m=float(np.median((flon - lon) * 111320 * cos)),
                delta_north_median_m=float(np.median((flat - lat) * 110574)),
                delta_H_navd88_vs_noop_median_m=float(np.median(H - base_H)),
                query_xy_shift_median_m=float(np.median(np.hypot(*(query - base_xy).T))),
                reference_height_change_median_m=float(np.nanmedian(ref - base_ref))))
            op_rows.append(dict(granule_id=spec.granule_id, hypothesis=name, epoch=epoch, **info))
            for pop_name in ("field", "road"):
                sel = f.population.eq(pop_name) & f.research_role.eq("independent_test")
                for r in _metrics(f.loc[sel], "residual", pd.Series(ok, index=f.index).loc[sel]):
                    metric_rows.append(dict(granule_id=spec.granule_id, hypothesis=name, population=pop_name, **r))
    out = dict(metrics=pd.DataFrame(metric_rows), shifts=pd.DataFrame(shift_rows), operations=pd.DataFrame(op_rows))
    for k, v in out.items():
        v.to_csv(output / f"frame_hypothesis_{k}.csv", index=False)
    out["output"] = output
    return out
