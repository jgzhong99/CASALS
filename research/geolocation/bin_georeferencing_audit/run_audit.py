"""Independent, read-only L1B bin audit. No production geolocation imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Geod, Transformer
from scipy.signal import find_peaks, peak_widths

TO_XYZ = Transformer.from_crs(4979, 4978, always_xy=True)
TO_GEO = Transformer.from_crs(4978, 4979, always_xy=True)
GEOD = Geod(ellps="WGS84")
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "outputs/research/bin_georeferencing_audit"
TUTORIAL = "https://book.cryointhecloud.com/l1b-waveforms-tutorial/"
# Frozen after first 14,080 records of November 12; not an official unit contract.
EMPIRICAL_BOUNCE_STEP = 5e-10
SEED = 20261009
FIELDS = ["delta_time", "sweep_num", "track_num", "bin_size", "refh_snr",
          "good_snr", "bg_std", "bg_mean", "local_beam_azimuth", "local_beam_elevation",
          "instrument_longitude", "instrument_latitude", "instrument_altitude",
          "range_bias_correction"] + [f"{prefix}{suffix}" for prefix in
          ("rwstart", "rwstop", "refh") for suffix in
          ("", "_longitude", "_latitude", "_bounce_time_offset", "_neutat_delay_total",
           "_neutat_delay_derivative")]


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_value), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(h5):
    objects, names = {}, {}
    def visit(name, obj):
        entry = {"type": "dataset" if isinstance(obj, h5py.Dataset) else "group",
                 "attributes": {k: json_value(v) for k, v in obj.attrs.items()}}
        if isinstance(obj, h5py.Dataset):
            entry.update(shape=list(obj.shape), dtype=str(obj.dtype), chunks=obj.chunks,
                         compression=obj.compression, scaleoffset=obj.scaleoffset,
                         hdf5_storage_fillvalue=json_value(obj.fillvalue))
            names.setdefault(name.split("/")[-1], []).append(name)
        objects["/" + name] = entry
    h5.visititems(visit)
    paths = {}
    for name, matches in names.items():
        if len(matches) != 1:
            raise ValueError(f"Ambiguous field {name}: {matches}")
        paths[name] = matches[0]
    return {"global_attributes": {k: json_value(v) for k, v in h5.attrs.items()},
            "objects": objects}, paths


def read_values(ds, selection):
    raw = np.asarray(ds[selection], dtype=float)
    # HDF5 default storage fill is often zero and is NOT a missing-data contract.
    for key in ("_FillValue", "missing_value"):
        if key in ds.attrs:
            raw[np.isin(raw, np.asarray(ds.attrs[key]).ravel())] = np.nan
    return raw * float(ds.attrs.get("scale_factor", 1)) + float(ds.attrs.get("add_offset", 0))


def bin_to_xyz(start_xyz, stop_xyz, bins, n_bins, *, convention="H2"):
    """ECEF meters -> ECEF meters, broadcast (...,3) endpoints and (...) bins.

    H1 centers span endpoints; H2 centers inside window edges (recommended only
    for audited granules); H3 reverses H1. Domain is [0,N-1], fractional allowed.
    Invalid finite-domain input raises ValueError; never clip or extrapolate.
    """
    if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    a, z, b = np.asarray(start_xyz, float), np.asarray(stop_xyz, float), np.asarray(bins, float)
    if a.ndim < 1 or z.ndim < 1 or a.shape[-1] != 3 or z.shape[-1] != 3:
        raise ValueError("endpoints must have final dimension 3")
    a, z = np.broadcast_arrays(a, z)
    if not np.isfinite(a).all() or not np.isfinite(z).all() or np.any(np.linalg.norm(z-a, axis=-1) == 0):
        raise ValueError("nonfinite or zero-length endpoints")
    if not np.isfinite(b).all() or np.any((b < 0) | (b > n_bins-1)):
        raise ValueError("bin outside [0,N-1] or nonfinite")
    if convention == "H1":
        t = b / (n_bins-1)
    elif convention == "H2":
        t = (b+0.5) / n_bins
    elif convention == "H3":
        t = 1-b / (n_bins-1)
    else:
        raise ValueError("unknown convention")
    return a + t[..., None] * (z-a)


def xyz(lon, lat, height):
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(height) & (abs(lon) <= 180) & (abs(lat) <= 90)
    out = np.column_stack(TO_XYZ.transform(lon, lat, height))
    out[~valid] = np.nan
    return out


def beam(lon, lat, az, el, *, degrees=False):
    """Historical candidate: north-clockwise, horizon elevation, negated ENU."""
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    if degrees:
        az, el = np.deg2rad(az), np.deg2rad(el)
    east = np.column_stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)))
    north = np.column_stack((-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)))
    up = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
    return -(np.cos(el)*np.sin(az))[:, None]*east - (np.cos(el)*np.cos(az))[:, None]*north - np.sin(el)[:, None]*up


def angle(a, b):
    # atan2 is stable for the tiny residuals here; acos loses precision near 1.
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a*b, axis=1)))


def diagnostics(f, raw, ids, n_bins, attrs):
    n = len(ids)
    points = {p: xyz(f[p+"_longitude"], f[p+"_latitude"], f[p]) for p in ("rwstart", "rwstop", "refh")}
    a, z, r = (points[p] for p in ("rwstart", "rwstop", "refh"))
    sensor = xyz(f["instrument_longitude"], f["instrument_latitude"], f["instrument_altitude"])
    v = z-a
    length = np.linalg.norm(v, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        direction = v / length[:, None]
        t = np.sum((r-a)*v, axis=1) / length**2
        ir = r-sensor
        ir /= np.linalg.norm(ir, axis=1)[:, None]
    bdir = beam(f["refh_longitude"], f["refh_latitude"], f["local_beam_azimuth"], f["local_beam_elevation"])
    idir = beam(f["instrument_longitude"], f["instrument_latitude"], f["local_beam_azimuth"], f["local_beam_elevation"])
    finite_rx = np.isfinite(raw).all(axis=1)
    safe = np.where(np.isfinite(raw), raw, -np.inf)
    b = safe.argmax(axis=1)
    amplitude = safe[np.arange(n), b]
    second = np.partition(safe, -2, axis=1)[:, -2]
    near_count = np.sum(safe >= amplitude[:, None]-1, axis=1)
    ties = np.sum(safe == amplitude[:, None], axis=1)
    geom_valid = np.isfinite(a).all(1) & np.isfinite(z).all(1) & np.isfinite(r).all(1) & (length > 0)
    valid = geom_valid & finite_rx
    # Fixed before full execution, no closure-dependent screening.
    high = valid & (f["refh_snr"] >= 10) & (ties == 1) & (amplitude-second > 1)
    dt = f["rwstop_bounce_time_offset"]-f["rwstart_bounce_time_offset"]
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (f["refh_bounce_time_offset"]-f["rwstart_bounce_time_offset"])/dt
        h4 = (b+0.5)*EMPIRICAL_BOUNCE_STEP/dt
    out = {"pulse_index": ids, "sweep_num": f["sweep_num"].astype(np.int64),
           "track_num": f["track_num"].astype(np.int64), "geometry_valid": geom_valid,
           "rx_valid": finite_rx, "valid": valid, "high_quality": high,
           "raw_argmax_bin": b, "raw_max": amplitude, "second_max": second,
           "tie_count": ties, "near_max_count_1adu": near_count,
           "previous_bin_amplitude": safe[np.arange(n), np.maximum(0,b-1)],
           "next_bin_amplitude": safe[np.arange(n), np.minimum(n_bins-1,b+1)],
           "refh_snr": f["refh_snr"], "beam_elevation": f["local_beam_elevation"],
           "segment_length_m": length, "t_geom": t,
           "refh_line_distance_m": np.linalg.norm(r-(a+t[:,None]*v), axis=1),
           "refh_segment_distance_m": np.linalg.norm(r-(a+np.clip(t,0,1)[:,None]*v), axis=1),
           "outside_segment": (t<0)|(t>1),
           "bin_delta_H1": t*(n_bins-1)-b, "bin_delta_H2": t*n_bins-.5-b,
           "time_bin_delta_H1": tt*(n_bins-1)-b, "time_bin_delta_H2": tt*n_bins-.5-b,
           "t_time_minus_geom": tt-t, "bounce_span": dt,
           "bounce_step_per_bin": dt/n_bins,
           "segment_spacing_m_per_bin": length/(n_bins-1),
           "edge_spacing_m_per_bin": length/n_bins, "stored_bin_size": f["bin_size"],
           "spacing_difference": length/(n_bins-1)-f["bin_size"],
           "spacing_ratio": length/(n_bins-1)/f["bin_size"],
           "spacing_relative_error": (length/(n_bins-1)-f["bin_size"])/f["bin_size"],
           "stored_minus_height_step": f["bin_size"]-(f["rwstart"]-f["rwstop"])/n_bins,
           "stored_minus_sin_elevation_step": f["bin_size"]-length/n_bins*np.sin(f["local_beam_elevation"]),
           "RW_vs_IR_deg": angle(direction, ir), "RW_vs_beam_refh_deg": angle(direction,bdir),
           "RW_vs_beam_instrument_deg": angle(direction,idir), "IR_vs_beam_deg": angle(ir,idir),
           "RW_vs_beam_degrees_candidate_deg": angle(direction,beam(f["refh_longitude"],f["refh_latitude"],f["local_beam_azimuth"],f["local_beam_elevation"],degrees=True)),
           "RW_vs_encoded_positive_deg": angle(direction,-bdir),
           "instrument_to_refh_m": np.linalg.norm(r-sensor,axis=1),
           "instrument_to_rwstart_m": np.linalg.norm(a-sensor,axis=1),
           "instrument_to_rwstop_m": np.linalg.norm(z-sensor,axis=1),
           "instrument_projection_t": np.sum((sensor-a)*v,axis=1)/length**2,
           "rw_order_reversed": np.linalg.norm(z-sensor,axis=1)<np.linalg.norm(a-sensor,axis=1),
           "range_bias_correction": f["range_bias_correction"],
           "neutat_delay_difference": f["rwstop_neutat_delay_total"]-f["rwstart_neutat_delay_total"]}
    for p, point in (("rwstart",a),("rwstop",z)):
        delta = point-sensor
        out[p+"_instrument_beam_line_distance_m"] = np.linalg.norm(delta-np.sum(delta*idir,axis=1)[:,None]*idir,axis=1)
    # Units unknown: these columns are explicitly conditional on offsets being seconds.
    c = attrs.get("speed_of_light", np.nan)
    out["conditional_span_times_c_m"] = dt*c
    out["conditional_span_times_c_half_m"] = dt*c/2
    out["conditional_c_half_minus_length_m"] = dt*c/2-length
    models = {"H1": b/(n_bins-1), "H2": (b+.5)/n_bins, "H3": 1-b/(n_bins-1), "H4_empirical": h4}
    for model, fraction in models.items():
        pred = a+fraction[:,None]*v
        diff = pred-r
        geo = np.column_stack(TO_GEO.transform(*pred.T))
        out[model+"_3d_m"] = np.linalg.norm(diff,axis=1)
        out[model+"_along_m"] = np.sum(diff*direction,axis=1)
        out[model+"_perp_m"] = np.linalg.norm(diff-out[model+"_along_m"][:,None]*direction,axis=1)
        out[model+"_height_m"] = geo[:,2]-f["refh"]
        out[model+"_horizontal_m"] = GEOD.inv(f["refh_longitude"],f["refh_latitude"],geo[:,0],geo[:,1])[2]
    # H5 closure is tautological; non-reference endpoint/center discrepancies are useful.
    out["H5_3d_m"] = np.where(valid,0.,np.nan)
    for endpoint, tb in (("first",np.zeros(n)),("last",np.full(n,n_bins-1))):
        seg = a+((tb+.5)/n_bins)[:,None]*v
        anchored = r+((tb-b)*length/n_bins)[:,None]*bdir
        out["H5_vs_H2_"+endpoint+"_m"] = np.linalg.norm(anchored-seg,axis=1)
    # Detection of abrupt changes is across the same track, handled after streaming.
    for j, axis in enumerate("xyz"):
        out["RW_direction_"+axis] = direction[:,j]
    for key, value in out.items():
        if key.endswith(("_m", "_deg")) or key.startswith(("bin_delta", "time_bin_delta")):
            out[key] = np.where(valid,value,np.nan)
    return pd.DataFrame(out)


def statistics(values, mask=None):
    values = np.asarray(values, dtype=float)
    total = len(values) if mask is None else int(np.sum(mask))
    x = values if mask is None else values[mask]
    x = x[np.isfinite(x)]
    row = {"n_total": total, "n_valid": len(x), "n_rejected": total-len(x),
           "valid_fraction": len(x)/total if total else np.nan}
    if len(x):
        med = np.median(x)
        row.update(median=med, NMAD=1.4826*np.median(abs(x-med)),
                   P90=np.percentile(x,90), P95=np.percentile(x,95),
                   P99=np.percentile(x,99), max=x.max(), min=x.min(), mean=x.mean())
    return row


def field_contract(inv, paths):
    rows = []
    for name in sorted(set(FIELDS+["rx_waveform","tx_waveform","rx_bins","tx_bins"]) | set(paths)):
        path = paths.get(name)
        attrs = inv["objects"]["/"+path]["attributes"] if path else {}
        documented = name in {"refh","delta_time","sweep_num","track_num","rx_waveform","tx_waveform","rx_bins","tx_bins"}
        rows.append({"field": name, "path": "/"+path if path else "MISSING",
                     "shape": str(inv["objects"]["/"+path].get("shape")) if path else "",
                     "units": attrs.get("units","WGS84 ellipsoidal m" if name=="refh" else "UNKNOWN"),
                     "definition_evidence": "Documented" if documented or attrs.get("description") else "Unknown",
                     "source": TUTORIAL if documented else "H5 attributes (no semantic attributes in audited inputs)",
                     "description": attrs.get("description",attrs.get("long_name","")),
                     "scale": attrs.get("scale_factor",1), "offset": attrs.get("add_offset",0),
                     "explicit_missing_value": str(attrs.get("_FillValue",attrs.get("missing_value","NONE"))),
                     "correction_applied": "UNKNOWN" if any(s in name for s in ("correction","delay","geoid","tide","dac")) else "N/A",
                     "empirical_note": "see findings.md; no unit inferred from name alone"})
    return pd.DataFrame(rows)


def summarize(outdir):
    parquet = outdir/"pulse_diagnostics.parquet"
    pf = pq.ParquetFile(parquet)
    masks = pq.read_table(parquet,columns=["valid","high_quality"]).to_pandas()
    rows = []
    for name in pf.schema_arrow.names:
        if name in {"pulse_index","sweep_num","track_num"}:
            continue
        x = pq.read_table(parquet,columns=[name])[name].to_numpy()
        for subset, mask in (("all",None),("high_quality",masks.high_quality.to_numpy())):
            rows.append({"metric": name, "subset": subset, **statistics(x,mask)})
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir/"all_metrics_summary.csv",index=False)
    categories = {"geometry_summary.csv": ("segment","line_distance","t_geom","outside","instrument_to","rw_order"),
                  "bin_mapping_comparison.csv": ("bin_delta","tie_count","near_max","raw_argmax"),
                  "refh_closure_summary.csv": ("H1_","H2_","H3_","H4_","H5_3d"),
                  "timing_spacing_comparison.csv": ("spacing","stored","bounce","time_bin","t_time","conditional","neutat","range_bias"),
                  "beam_geometry_summary.csv": ("_deg","beam_line","H5_vs")}
    for filename, patterns in categories.items():
        summary[summary.metric.map(lambda s:any(p in s for p in patterns))].to_csv(outdir/filename,index=False)
    cols = ["pulse_index","sweep_num","track_num","valid","high_quality","refh_snr","beam_elevation",
            "H1_3d_m","H2_3d_m","bin_delta_H1","bin_delta_H2","time_bin_delta_H2","refh_line_distance_m",
            "RW_vs_beam_refh_deg","RW_vs_beam_instrument_deg","segment_length_m","RW_direction_x","RW_direction_y","RW_direction_z"]
    df = pq.read_table(parquet,columns=cols).to_pandas()
    df["sweep_block"] = df.sweep_num//512
    df["split"] = np.where(df.sweep_block%4==3,"validation_contiguous_block","exploration")
    df["beam_group"] = pd.cut(df.beam_elevation,[-np.inf,1.3,1.45,1.55,np.inf]).astype(str)
    df["snr_group"] = pd.cut(df.refh_snr,[-np.inf,3,10,30,np.inf]).astype(str)
    metrics = ["H1_3d_m","H2_3d_m","bin_delta_H1","bin_delta_H2","refh_line_distance_m","RW_vs_beam_refh_deg"]
    for group in ("sweep_num","track_num","sweep_block","split","beam_group","snr_group"):
        grouped = df.groupby(group,observed=True)[metrics].agg(["count","median","max"])
        grouped.columns = ["_".join(c) for c in grouped.columns]
        for m in metrics:
            grouped[m+"_P95"] = df.groupby(group,observed=True)[m].quantile(.95)
        grouped["n_total"] = df.groupby(group,observed=True).size()
        grouped["n_valid"] = df.groupby(group,observed=True).valid.sum()
        grouped.to_csv(outdir/(group+"_summary.csv"))
    agreement = []
    for subset, mask in (("all",df.valid), ("high_quality",df.high_quality)):
        for model in ("H1","H2"):
            x = df.loc[mask,"bin_delta_"+model].dropna().to_numpy()
            agreement.append({"subset":subset,"model":model,"n_total":len(df),"n_used":len(x),
                              "n_rejected":len(df)-len(x),"nearest_bin_agreement":np.mean(abs(x)<.5) if len(x) else np.nan,
                              "within_half_bin":np.mean(abs(x)<=.5) if len(x) else np.nan,"within_one_bin":np.mean(abs(x)<=1) if len(x) else np.nan,
                              **statistics(abs(x)),"signed_bias":np.median(x) if len(x) else np.nan})
    pd.DataFrame(agreement).to_csv(outdir/"bin_agreement.csv",index=False)
    # Correlation-aware uncertainty: bootstrap medians of 512-sweep block medians.
    block = df.groupby("sweep_block")[metrics].median()
    rng = np.random.default_rng(SEED)
    bootstrap = []
    for m in metrics:
        x = block[m].dropna().to_numpy()
        boot = np.median(rng.choice(x,(2000,len(x)),replace=True),axis=1)
        bootstrap.append({"metric":m,"n_blocks":len(x),"estimand":"median of 512-sweep block medians",
                          "median":np.median(x),"CI_low":np.percentile(boot,2.5),"CI_high":np.percentile(boot,97.5)})
    pd.DataFrame(bootstrap).to_csv(outdir/"block_uncertainty.csv",index=False)
    ordered = df.sort_values(["track_num","sweep_num"])
    ordered["segment_length_jump_m"] = ordered.groupby("track_num").segment_length_m.diff().abs()
    prev = ordered.groupby("track_num")[["RW_direction_x","RW_direction_y","RW_direction_z"]].shift().to_numpy()
    ordered["direction_jump_deg"] = angle(ordered[["RW_direction_x","RW_direction_y","RW_direction_z"]].to_numpy(),prev)
    pd.DataFrame([{"metric":m,**statistics(ordered[m].to_numpy())} for m in
                  ["segment_length_jump_m","direction_jump_deg"]]).to_csv(outdir/'track_continuity_summary.csv',index=False)
    worst = []
    for m in ("H2_3d_m","refh_line_distance_m","RW_vs_beam_refh_deg","segment_length_jump_m","direction_jump_deg"):
        top = ordered.nlargest(10,m).copy()
        top["selection_reason"] = "top10_"+m
        worst.append(top)
    anomalies = pd.concat(worst,ignore_index=True)
    anomalies.to_csv(outdir/"anomalous_pulses.csv",index=False)
    return summary, df, anomalies


def representative_examples(h5,paths,outdir,df,anomalies,n_bins,record_axis):
    rng = np.random.default_rng(SEED)
    # Stratify candidates by block, SNR and angle; bounded waveform search only.
    ids = set(anomalies.pulse_index.astype(int))
    for _, group in df.groupby(["sweep_block","snr_group","beam_group"],observed=True):
        ids.update(rng.choice(group.pulse_index.to_numpy(),min(2,len(group)),replace=False).tolist())
    historical = df[(df.sweep_num==5000)&(df.track_num==214)]
    ids.update(historical.pulse_index.astype(int))
    representatives, category_count, hist = [], {}, []
    figs = outdir/"figures"
    figs.mkdir(exist_ok=True)
    ds = h5[paths["rx_waveform"]]
    for i in sorted(ids):
        wave = read_values(ds,(i,slice(None)) if record_axis==0 else (slice(None),i))
        if not np.isfinite(wave).all():
            continue
        spread = np.ptp(wave)
        peaks,_ = find_peaks(wave,prominence=max(5,.15*spread),distance=12)
        category = "single" if len(peaks)==1 else "double" if len(peaks)==2 else "multiple" if len(peaks)>2 else "weak"
        if len(peaks) and np.max(peak_widths(wave,peaks)[0])>=20:
            category = "wide"
        b = int(wave.argmax())
        row = df[df.pulse_index==i].iloc[0]
        if row.refh_snr < 3: category="low_snr"
        if b<20 or b>n_bins-21: category="boundary"
        if np.sum(wave>=wave[b]-1)>1: category="near_tie"
        is_hist = i in historical.pulse_index.values
        is_worst = i in anomalies.pulse_index.values
        if is_hist:
            hist.append({"pulse_index":i,"sweep_num":int(row.sweep_num),"track_num":int(row.track_num),
                         "argmax_bin":b,"amp_1693":wave[1693] if n_bins>1693 else None,
                         "amp_1694":wave[1694] if n_bins>1694 else None,
                         "max_amplitude":wave[b],"tie_count":int(np.sum(wave==wave[b]))})
        if category_count.get(category,0)>=2 and not is_hist and not is_worst:
            continue
        category_count[category] = category_count.get(category,0)+1
        f = {name: np.array([read_values(h5[paths[name]],i)]) if name in paths else np.array([np.nan]) for name in FIELDS}
        a = xyz(f['rwstart_longitude'],f['rwstart_latitude'],f['rwstart'])[0]
        z = xyz(f['rwstop_longitude'],f['rwstop_latitude'],f['rwstop'])[0]
        r = xyz(f['refh_longitude'],f['refh_latitude'],f['refh'])[0]
        if not np.isfinite([a,z,r]).all() or np.linalg.norm(z-a)==0:
            continue
        bd = beam(f['refh_longitude'],f['refh_latitude'],f['local_beam_azimuth'],f['local_beam_elevation'])[0]
        test_bins = np.unique(np.r_[0,n_bins-1, b, np.clip(b+.4,0,n_bins-1),n_bins*.25,n_bins*.75,peaks])
        xyz2 = bin_to_xyz(a,z,test_bins,n_bins)
        geo = np.column_stack(TO_GEO.transform(*xyz2.T))
        xyz5 = r+(test_bins-b)[:,None]*np.linalg.norm(z-a)/n_bins*bd
        along = np.sum((xyz2-a)*(z-a)/np.linalg.norm(z-a),axis=1)
        continuity = np.max(abs(np.diff(along)/np.diff(test_bins)-np.linalg.norm(z-a)/n_bins)) if len(test_bins)>1 else 0
        for j,tb in enumerate(test_bins):
            representatives.append({"pulse_index":i,"sweep_num":int(row.sweep_num),"track_num":int(row.track_num),
                "category":category,"snr":row.refh_snr,"selection_seed":SEED,"bin":tb,"raw_argmax":b,
                "is_detected_raw_peak":bool(tb in peaks),"x":xyz2[j,0],"y":xyz2[j,1],"z":xyz2[j,2],
                "longitude":geo[j,0],"latitude":geo[j,1],"ellipsoidal_height":geo[j,2],
                "H5_minus_H2_m":np.linalg.norm(xyz5[j]-xyz2[j]),"continuity_step_error_m":continuity,
                "finite":bool(np.isfinite(xyz2[j]).all()),"physical_return_verified":False})
        # Plot examples once per category, plus worst closure and historical pulse.
        if category_count[category]<=1 or is_hist or i==int(anomalies.iloc[0].pulse_index):
            fig = plt.figure(figsize=(11,4.2))
            ax = fig.add_subplot(121)
            ax.plot(wave,lw=.7); ax.scatter(peaks,wave[peaks],s=18,c='orange')
            ax.axvline(b,color='crimson',ls='--',label='raw argmax'); ax.legend()
            ax.set(xlabel='Zero-based RX bin',ylabel='Stored raw amplitude',title=f'{category}: pulse {i}, sweep {int(row.sweep_num)}, track {int(row.track_num)}')
            ax2=fig.add_subplot(122,projection='3d')
            delta=xyz2-r
            ax2.plot(delta[:,0],delta[:,1],delta[:,2],'.-',label='H2 bin positions')
            ax2.scatter([0],[0],[0],c='crimson',label='Official Refh')
            for j in np.flatnonzero(np.isin(test_bins,peaks)):
                ax2.text(*delta[j],f' b={test_bins[j]:g}',fontsize=7)
            ax2.set(xlabel='ECEF ΔX (m)',ylabel='ΔY (m)',zlabel='ΔZ (m)',title='Relative to Refh; sampled bins are not certified returns')
            ax2.legend(fontsize=7);fig.tight_layout();fig.savefig(figs/f'pulse_{i}_{category}.png',dpi=160);plt.close(fig)
    pd.DataFrame(representatives).to_csv(outdir/"representative_pulses.csv",index=False)
    pd.DataFrame(hist).to_csv(outdir/"historical_near_tie_case.csv",index=False)
    save_json(outdir/"sample_selection.json",{"seed":SEED,"candidate_count":len(ids),"categories":category_count,
        "rule":"two seeded pulses per 512-sweep block/SNR/elevation stratum plus worst anomalies and historical case; raw scipy peaks prominence max(5,15% waveform span), distance 12; widths >=20 bins; classes are waveform morphology, not landcover",
        "missing_categories": sorted(set(['single','double','multiple','wide','weak','boundary','low_snr','near_tie'])-set(category_count))})


def diagnostic_figures(outdir,df):
    parquet=outdir/"pulse_diagnostics.parquet"
    cols=["refh_line_distance_m","refh_segment_distance_m","bin_delta_H1","bin_delta_H2","H1_3d_m","H2_3d_m","H3_3d_m",
          "segment_spacing_m_per_bin","edge_spacing_m_per_bin","stored_bin_size","t_geom","t_time_minus_geom","RW_vs_beam_refh_deg","RW_vs_beam_instrument_deg"]
    # Deterministic plotting sample is explicitly labeled; statistics use every pulse.
    data=pq.read_table(parquet,columns=cols).to_pandas().iloc[::max(1,len(df)//20000)]
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    def ecdf(ax,columns):
        for name in columns:
            x=np.sort(data[name].dropna());ax.plot(x,np.arange(1,len(x)+1)/len(x),label=name)
        ax.legend(fontsize=7);ax.set_ylabel('Empirical CDF (plot sample)');ax.grid(alpha=.2)
    ecdf(axes[0,0],["refh_line_distance_m","refh_segment_distance_m"]);axes[0,0].set_xlabel('Distance (m)')
    ecdf(axes[0,1],["bin_delta_H1","bin_delta_H2"]);axes[0,1].set_xlabel('Geometry bin minus raw argmax (bins)')
    ecdf(axes[0,2],["H1_3d_m","H2_3d_m"]);axes[0,2].set_xlabel('Refh closure (m)')
    axes[1,0].scatter(data.stored_bin_size,data.edge_spacing_m_per_bin,s=2,alpha=.2)
    axes[1,0].set(xlabel='Stored bin_size (unit undocumented)',ylabel='ECEF edge spacing (m/bin)')
    axes[1,1].scatter(data.t_geom,data.t_time_minus_geom,s=2,alpha=.2)
    axes[1,1].set(xlabel='t_geom',ylabel='t_time - t_geom')
    ecdf(axes[1,2],["RW_vs_beam_refh_deg","RW_vs_beam_instrument_deg"]);axes[1,2].set_xlabel('Angle residual (degrees)')
    fig.suptitle(outdir.name+' — full statistics, deterministic plotting sample');fig.tight_layout()
    fig.savefig(outdir/'figures/geometry_timing_beam.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,group in zip(axes,['sweep_num','track_num']):
        table=pd.read_csv(outdir/(group+'_summary.csv'))
        ax.plot(table[group],table.H1_3d_m_median,label='H1 median')
        ax.plot(table[group],table.H2_3d_m_P95,label='H2 P95');ax.set(xlabel=group,ylabel='Closure (m)');ax.legend()
    fig.tight_layout();fig.savefig(outdir/'figures/residual_by_sweep_track.png',dpi=160);plt.close(fig)


def audit_file(path,outroot,mode,chunk_size,smoke_sweeps):
    outdir=outroot/path.stem
    outdir.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();before=path.stat()
    print(f'{path.name}: hashing original input',flush=True)
    source_hash=sha256(path)
    with h5py.File(path,'r') as h5:
        inv,paths=inventory(h5);save_json(outdir/'metadata_inventory.json',inv)
        contract=field_contract(inv,paths);contract.to_csv(outdir/'field_contract.csv',index=False)
        contract.to_csv(outdir/'h5_field_contract.csv',index=False)
        required=['rx_waveform','delta_time','sweep_num','track_num']+[p+s for p in ['rwstart','rwstop','refh'] for s in ['', '_longitude','_latitude']]
        missing=[name for name in required if name not in paths]
        if missing: raise ValueError(f'Missing essential fields: {missing}')
        ds=h5[paths['rx_waveform']];n=h5[paths['delta_time']].shape[0]
        axes=[j for j,size in enumerate(ds.shape) if size==n]
        if ds.ndim!=2 or len(axes)!=1: raise ValueError('Ambiguous waveform record axis')
        axis=axes[0];n_bins=ds.shape[1-axis]
        if 'rx_bins' in paths and not np.array_equal(h5[paths['rx_bins']][:],np.arange(n_bins)):
            raise ValueError('RX bin coordinate is not zero-based contiguous; requires explicit contract')
        writer=None;count=0;invalid_counts={};field_invalid={name:0 for name in FIELDS}
        try:
            for lo in range(0,n,chunk_size):
                hi=min(n,lo+chunk_size);selection=slice(lo,hi)
                sweeps=read_values(h5[paths['sweep_num']],selection)
                keep=np.ones(hi-lo,dtype=bool) if mode=='full' else np.isin(sweeps,smoke_sweeps)
                if not keep.any():continue
                f={}
                for name in FIELDS:
                    f[name]=read_values(h5[paths[name]],selection)[keep] if name in paths else np.full(keep.sum(),np.nan)
                    field_invalid[name]+=int((~np.isfinite(f[name])).sum())
                raw=read_values(ds,(selection,slice(None)) if axis==0 else (slice(None),selection))
                if axis==1:raw=raw.T
                raw=raw[keep]
                d=diagnostics(f,raw,np.arange(lo,hi)[keep],n_bins,inv['global_attributes'])
                for reason,mask in {'nonfinite_or_out_of_bounds_geometry':~d.geometry_valid,
                                    'invalid_rx':~d.rx_valid,'outside_segment':d.outside_segment,
                                    'line_distance_gt_2mm':d.refh_line_distance_m>.002,
                                    'H2_closure_gt_10cm':d.H2_3d_m>.1,
                                    'zero_length_segment':d.segment_length_m==0,
                                    'rw_order_reversed':d.rw_order_reversed,
                                    'instrument_not_before_start':d.instrument_projection_t>=0}.items():
                    invalid_counts[reason]=invalid_counts.get(reason,0)+int(mask.sum())
                for prefix in ['rwstart','rwstop','refh','instrument']:
                    lon=f[prefix+'_longitude'];lat=f[prefix+'_latitude']
                    for reason,mask in {prefix+'_longitude_out_of_bounds':abs(lon)>180,
                                        prefix+'_latitude_out_of_bounds':abs(lat)>90,
                                        prefix+'_nonfinite_lonlat':~(np.isfinite(lon)&np.isfinite(lat))}.items():
                        invalid_counts[reason]=invalid_counts.get(reason,0)+int(mask.sum())
                table=pa.Table.from_pandas(d,preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(outdir/'pulse_diagnostics.parquet',table.schema,compression='zstd')
                writer.write_table(table);count+=len(d)
                if lo//chunk_size%16==0:print(f'  {hi:,}/{n:,} records scanned; {count:,} audited',flush=True)
        finally:
            if writer:writer.close()
        if count==0:raise ValueError('No pulses selected')
        print('  exact full-population summaries and representative waveforms',flush=True)
        summary,df,anomalies=summarize(outdir)
        representative_examples(h5,paths,outdir,df,anomalies,n_bins,axis)
        diagnostic_figures(outdir,df)
    after=path.stat()
    print('  verifying original SHA256 unchanged',flush=True)
    after_hash=sha256(path)
    if source_hash!=after_hash:raise RuntimeError('Source H5 hash changed')
    meta={'source_h5':str(path.resolve()),'sha256_before':source_hash,'sha256_after':after_hash,
          'source_unchanged':source_hash==after_hash,'size_bytes':before.st_size,'mtime_ns_before':before.st_mtime_ns,
          'mtime_ns_after':after.st_mtime_ns,'mode':mode,'n_input_pulses':n,'n_audited':count,'n_rx_bins':n_bins,
          'rx_record_axis':axis,'chunk_size':chunk_size,'rx_storage_chunks':inv['objects']['/'+paths['rx_waveform']]['chunks'],
          'full_traversal':mode=='full' and count==n,'invalid_counts_overlap':invalid_counts,
          'invalid_field_counts':field_invalid,'missing_fields':[name for name in FIELDS if name not in paths],
          'elapsed_seconds':time.monotonic()-start,'python':sys.executable,'platform':platform.platform(),
          'versions':{m.__name__:m.__version__ for m in [h5py,np,pd,pa]},
          'script_sha256':sha256(Path(__file__)),'utc_finished':datetime.now(timezone.utc).isoformat(),
          'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'seed':SEED,'high_quality_rule':'valid geometry/RX; refh_snr>=10; unique max; max-second>1 stored ADU',
          'block_split':'512 contiguous sweeps per block; block%4==3 validation; other blocks exploration',
          'H4_status':'EMPIRICALLY INFERRED: tau=start+(b+0.5)*5e-10 in stored offset units; no documented seconds contract',
          'H5_status':'Refh anchored, edge-derived spacing, closure zero by construction',
          'H6_status':'UNAVAILABLE: bin_size units/range meaning undocumented',
          'corrections':'No geoid, tide, atmosphere, range bias reapplied. Applied status unknown.',
          'tutorial_url':TUTORIAL,'independence':'Both historical granules previously explored; second date is cross-date replication, not blind validation.'}
    save_json(outdir/'run_metadata.json',meta)
    print(f'  completed in {meta["elapsed_seconds"]:.1f}s',flush=True)
    return outdir


def compare(directories,outroot):
    dest=outroot/'comparison';dest.mkdir(exist_ok=True)
    allrows=[];modelrows=[]
    for directory in directories:
        summary=pd.read_csv(directory/'all_metrics_summary.csv');summary.insert(0,'granule',directory.name);allrows.append(summary)
        for model in ['H1','H2','H3','H4_empirical','H5','H6']:
            row={'granule':directory.name,'model':model,'status':'UNAVAILABLE' if model=='H6' else 'EVALUATED',
                 'refh_independent':model not in ['H5','H6'],'coordinate_reference':'EPSG:4978 m; EPSG:4979 ellipsoidal h',
                 'limitations':'closure != absolute accuracy; H4 offset unit empirical; H5 anchored tautology; H6 unit unknown'}
            q=summary[(summary.metric==model+'_3d_m')&(summary.subset=='all')]
            if len(q):row.update(q.iloc[0][['n_total','n_valid','valid_fraction','median','P95','max']].to_dict())
            modelrows.append(row)
    combined=pd.concat(allrows,ignore_index=True);combined.to_csv(dest/'cross_granule_summary.csv',index=False)
    pd.DataFrame(modelrows).to_csv(dest/'model_comparison.csv',index=False)
    pd.DataFrame([
        {'evidence':'raw H5 full traversal','status':'PASS' if all(json.loads((d/'run_metadata.json').read_text())['full_traversal'] for d in directories) else 'SMOKE ONLY'},
        {'evidence':'Refh closure','status':'computed independently of Refh anchor for H1-H4'},
        {'evidence':'time offsets / bin centers','status':'empirical; offsets lack units'},
        {'evidence':'beam convention','status':'empirical; radian/north-clockwise/negated ENU at Refh'},
        {'evidence':'official bin processor confirmation','status':'NOT VERIFIED'},
        {'evidence':'external absolute accuracy','status':'NOT VERIFIED; 3DEP datum/epoch/surface correspondence not established'}
    ]).to_csv(dest/'evidence_matrix.csv',index=False)
    (dest/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric in zip(axes,['H1_3d_m','H2_3d_m','RW_vs_beam_refh_deg']):
        q=combined[(combined.metric==metric)&(combined.subset=='all')]
        ax.bar(np.arange(len(q)),q['P95']);ax.set_xticks(np.arange(len(q)),[s[11:19] for s in q.granule])
        ax.set(title=metric,ylabel='P95 (m or degrees)')
    fig.tight_layout();fig.savefig(dest/'figures/cross_granule.png',dpi=160);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5',type=Path,nargs='+',default=sorted((ROOT/'data/raw/casals_l1b').glob('*.h5')))
    parser.add_argument('--mode',choices=['smoke','full'],default='smoke')
    parser.add_argument('--chunk-size',type=int,default=14080)
    parser.add_argument('--smoke-sweeps',type=int,nargs='+',default=[0,5000,7040,14079])
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args(argv)
    if args.chunk_size<1:parser.error('chunk-size must be positive')
    if not args.h5 or any(not p.is_file() for p in args.h5):parser.error('NOT VERIFIED: original H5 unavailable')
    directories=[audit_file(p,args.output_root,args.mode,args.chunk_size,args.smoke_sweeps) for p in args.h5]
    compare(directories,args.output_root)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
