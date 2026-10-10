"""Small numerical and input helpers for the three L1B showcase notebooks.

No production behavior is changed. Raster heights retain their source datum.
"""
from pathlib import Path
import json
import numpy as np
import rasterio
from pyproj import Transformer
from casals_l1b.refh import read_refh_points, project_refh_points
from research.geolocation.bin_georeferencing_audit.run_audit import bin_to_xyz


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf-8')


def relative(path):
    return str(Path(path).resolve().relative_to(Path.cwd().resolve())).replace('\\', '/')


def find_products(h5_path, key):
    """Use metadata outputs, checking source identity and existing files first."""
    stem = Path(h5_path).stem
    for p in sorted(Path('outputs').rglob('*metadata.json')):
        d = json.loads(p.read_text(encoding='utf-8'))
        source = d.get('source_h5', d.get('input_h5', d.get('config', {}).get('h5_path', '')))
        outputs = d.get('outputs', {})
        if key not in outputs or not outputs[key]:
            continue
        if stem not in str(source) and stem not in str(p):
            continue
        if Path(outputs[key]).exists():
            return {k: Path(v) for k, v in outputs.items() if isinstance(v, str) and Path(v).exists()}, p
    raise FileNotFoundError(f'No existing metadata output {key} for {stem}')


def ensure_surfaces(h5_path, output_dir):
    """Generate only missing formal surfaces; never duplicate full point clouds."""
    try:
        dsm, dm = find_products(h5_path, 'strict_dsm_tif')
    except FileNotFoundError:
        from casals_l1b.refh_dsm import Config, make_refh_dsm
        out = Path(output_dir) / 'derived_dsm'
        make_refh_dsm(Config(Path(h5_path), out / 'point_clouds', out,
                             snr_threshold=4.5, dsm_resolution_m=2., write_selected_las=False))
        dsm, dm = find_products(h5_path, 'strict_dsm_tif')
    try:
        dtm, tm = find_products(h5_path, 'dtm_tif')
    except FileNotFoundError:
        from casals_l1b.refh_ground import Config, make_refh_ground
        out = Path(output_dir) / 'derived_dtm'
        make_refh_ground(Config(Path(h5_path), out / 'point_clouds', out,
                                write_classified_highsnr_las=False, write_ground_only_las=False))
        dtm, tm = find_products(h5_path, 'dtm_tif')
    return dsm, dtm, [relative(dm), relative(tm)]


def sample_raster(path, xy, source_crs, support_path=None):
    """Nearest cell sampling in each raster's CRS; nodata/outside => NaN."""
    xy = np.asarray(xy, float)
    with rasterio.open(path) as ds:
        x, y = Transformer.from_crs(source_crs, ds.crs, always_xy=True).transform(*xy.T)
        row, col = rasterio.transform.rowcol(ds.transform, x, y)
        row, col = np.asarray(row), np.asarray(col)
        inside = (row >= 0) & (row < ds.height) & (col >= 0) & (col < ds.width)
        arr = ds.read(1, masked=True).filled(np.nan).astype(float) if ds.dtypes[0].startswith('float') else ds.read(1).astype(float)
        values = np.full(len(xy), np.nan)
        values[inside] = arr[row[inside], col[inside]]
        if ds.nodata is not None:
            values[values == ds.nodata] = np.nan
    valid = np.isfinite(values)
    if support_path is not None:
        support, sv = sample_raster(support_path, xy, source_crs)
        valid &= sv & (support > 0)
    values[~valid] = np.nan
    return values, valid


def transect_coordinates(xy, a, b):
    a, b, xy = np.asarray(a, float), np.asarray(b, float), np.asarray(xy, float)
    length = np.linalg.norm(b-a)
    if not np.isfinite(length) or length == 0:
        raise ValueError('Transect needs distinct finite endpoints')
    u = (b-a)/length
    delta = xy-a
    return delta @ u, np.abs(delta[:, 0]*u[1]-delta[:, 1]*u[0])


def enu_frame(lon, lat, height):
    origin = np.array(Transformer.from_crs(4979, 4978, always_xy=True).transform(lon, lat, height))
    lo, la = np.deg2rad([lon, lat])
    rotation = np.array([[-np.sin(lo), np.cos(lo), 0],
                         [-np.sin(la)*np.cos(lo), -np.sin(la)*np.sin(lo), np.cos(la)],
                         [np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)]])
    return origin, rotation


def to_enu(xyz, origin, rotation):
    return (np.asarray(xyz)-origin) @ rotation.T


def from_enu(enu, origin, rotation):
    return np.asarray(enu) @ rotation + origin


def load_points(h5_path):
    p = read_refh_points(Path(h5_path), optional_fields=())
    q = project_refh_points(p)
    return p, np.column_stack([q.easting, q.northing]), q.utm_epsg


def load_classes(h5_path, pulses):
    import laspy
    products, metadata = find_products(h5_path, 'classified_laz')
    classes = np.zeros(int(pulses.max())+1, dtype=np.uint8)
    with laspy.open(products['classified_laz']) as src:
        for chunk in src.chunk_iterator(500_000):
            ids = np.asarray(chunk.point_index, int)
            if (ids < 0).any() or (ids >= len(classes)).any():
                raise ValueError('Classification point_index outside source pulse domain')
            classes[ids] = np.asarray(chunk.classification)
    if not np.isin(classes[pulses], [1,2,7]).all():
        raise ValueError('Existing classification does not cover every requested pulse with class 1/2/7')
    return classes[pulses], relative(metadata)


NAIP_SERVICE = 'https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer'


def fetch_naip(bounds, crs, out):
    """Small georeferenced RGB export, pinned to intersecting primary rasters.

    Archive catalog attributes and returned extent; export CRS is independent
    of original NAD83 tile CRS. No registration adjustment is applied.
    """
    import requests
    from datetime import datetime, timezone
    out = Path(out)
    provenance = out.with_suffix('.json')
    if out.exists() and provenance.exists():
        return json.loads(provenance.read_text())
    x0, y0, x1, y1 = map(float, bounds)
    geometry = dict(xmin=x0, ymin=y0, xmax=x1, ymax=y1, spatialReference={'wkid':int(crs)})
    query = requests.get(NAIP_SERVICE+'/query', params=dict(f='json', geometry=json.dumps(geometry),
                         geometryType='esriGeometryEnvelope', spatialRel='esriSpatialRelIntersects',
                         outFields='*', returnGeometry='true', where='Category=1'), timeout=60)
    query.raise_for_status()
    catalog = query.json()
    if 'error' in catalog or not catalog.get('features'):
        raise RuntimeError(f'NAIP catalog query failed: {catalog}')
    ids = [f['attributes']['OBJECTID'] for f in catalog['features']]
    request = dict(f='json', bbox=','.join(map(str,bounds)), bboxSR=int(crs), imageSR=int(crs),
                   size=f'{int(np.ceil((x1-x0)/.5))},{int(np.ceil((y1-y0)/.5))}',
                   format='tiff', pixelType='U8', bandIds='0,1,2', interpolation='RSP_NearestNeighbor',
                   mosaicRule=json.dumps({'mosaicMethod':'esriMosaicLockRaster','lockRasterIds':ids}))
    response = requests.get(NAIP_SERVICE+'/exportImage', params=request, timeout=90)
    response.raise_for_status()
    export = response.json()
    if 'href' not in export:
        raise RuntimeError(f'NAIP export failed: {export}')
    image = requests.get(export['href'], timeout=90)
    image.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(image.content)
    with rasterio.open(out) as ds:
        if ds.crs.to_epsg() != int(crs) or ds.count < 3:
            raise ValueError('NAIP export CRS/bands invalid')
        actual = list(ds.bounds)
    dates = sorted(set(datetime.fromtimestamp(f['attributes']['acquisition_date']/1000, timezone.utc).date().isoformat() for f in catalog['features']))
    data = dict(service=NAIP_SERVICE, acquisition_dates=dates, catalog=catalog, export=export,
                request=request, exported_crs=f'EPSG:{crs}', exported_bounds=actual,
                retrieval_utc=datetime.now(timezone.utc).isoformat(),
                registration='Service georeferencing; no fitted shift; absolute footprint alignment not independently validated')
    save_json(provenance, data)
    return data


def show_ortho(ax, path):
    with rasterio.open(path) as ds:
        rgb = np.moveaxis(ds.read([1,2,3]),0,-1)
        ax.imshow(rgb, extent=[ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top])
    ax.set_aspect('equal')
    ax.ticklabel_format(useOffset=False, style='plain')


def screen_granule(h5_path, out, window_m=100, grid_step_m=2):
    """Deterministic full-scalar screening; no waveform reads."""
    import pandas as pd
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dsm, dtm, metadata = ensure_surfaces(h5_path, out)
    p, xy, crs = load_points(h5_path)
    classes, cm = load_classes(h5_path, p.pulse_index)
    ground, valid = sample_raster(dtm['dtm_tif'], xy, crs, dtm['support_mask_tif'])
    hag = p.z_refh-ground
    frame = pd.DataFrame({'gx':np.floor(xy[:,0]/window_m).astype(int),
                          'gy':np.floor(xy[:,1]/window_m).astype(int), 'valid':valid,
                          'high5':valid&(hag>5)&(p.refh_snr>=5),
                          'high10':valid&(hag>10)&(p.refh_snr>=5), 'noise':classes==7,
                          'hag':np.where(valid&(p.refh_snr>=5),hag,np.nan)})
    rows = []
    for (gx,gy), g in frame.groupby(['gx','gy']):
        x,y = np.meshgrid(np.arange(gx*window_m+grid_step_m/2,(gx+1)*window_m,grid_step_m),
                           np.arange(gy*window_m+grid_step_m/2,(gy+1)*window_m,grid_step_m))
        grid = np.column_stack([x.ravel(),y.ravel()])
        _, dv = sample_raster(dsm['strict_dsm_tif'],grid,crs,dsm['support_mask_tif'])
        _, tv = sample_raster(dtm['dtm_tif'],grid,crs,dtm['support_mask_tif'])
        rows.append(dict(granule=Path(h5_path).stem,gx=int(gx),gy=int(gy),point_count=len(g),
                         density_pts_m2=len(g)/window_m**2,
                         high5_fraction=float(g.high5.sum()/max(g.valid.sum(),1)),
                         high10_fraction=float(g.high10.sum()/max(g.valid.sum(),1)),
                         dtm_coverage=float(tv.mean()),strict_dsm_coverage=float(dv.mean()),
                         noise_fraction=float(g.noise.mean()),hag_p90=float(g.hag.quantile(.9)),
                         hag_p10=float(g.hag.quantile(.1))))
    df = pd.DataFrame(rows)
    df['score'] = df.high10_fraction*df.dtm_coverage*df.strict_dsm_coverage*(1-df.high5_fraction).clip(.05,1)
    df = df.sort_values(['score','gx','gy'],ascending=[False,True,True])
    df.to_csv(out/'aoi_candidates.csv',index=False)
    save_json(out/'screening_manifest.json',dict(h5=relative(h5_path), products_metadata=metadata,
                classification_metadata=cm, crs=crs, window_m=window_m,grid_step_m=grid_step_m,
                lonlat_bounds=[p.lon.min(),p.lat.min(),p.lon.max(),p.lat.max()],
                point_denominator='Finite Refh; HAG fractions among DTM-supported points; numerator also SNR>=5',
                score='high10 * DTM coverage * strict DSM coverage * max(1-high5,0.05)'))
    return df
