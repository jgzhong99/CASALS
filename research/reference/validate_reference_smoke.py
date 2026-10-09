"""Small real-data reference smoke; run from repository root.

Reads 64 middle sweeps per local H5 and streams corresponding LAZ clips to
select their buffered XY neighbourhood. Original inputs remain read-only.
Writes only one compact CRS/status JSON and evaluation CSV per granule.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, h5py, laspy, requests
from pyproj import CRS
sys.path.insert(0,str(Path.cwd()))
from research.reference import transfer_3dep_labels_to_casals as tr
from research.reference import diagnose_3dep_offsets as cr
from casals_l1b import classification as cl
from casals_l1b import evaluation as ev

# Frozen pre-change baseline; load in memory without writing historical code/files.
BASELINE_COMMIT = "8acc3e2"
def oldmodule(name, source_path):
    import subprocess, types
    source = subprocess.run(["git", "show", f"{BASELINE_COMMIT}:{source_path}"], check=True, capture_output=True, text=True, encoding="utf-8").stdout
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, source_path, "exec"), module.__dict__)
    return module
oldtr = oldmodule('old_transfer', 'research/reference/transfer_3dep_labels_to_casals.py')
oldev = oldmodule('casals_l1b._old_evaluation', 'casals_l1b/evaluation.py')
oldcl = oldmodule('casals_l1b._old_classification', 'casals_l1b/classification.py')
out=Path('outputs/reference_crs_validation');out.mkdir(parents=True,exist_ok=True)
side_root=Path('Archive/backup/outputs/baseline_pre_refactor/l1b_outputs/download_3dep_lpc/clip_sidecars')
for h5path in sorted(Path('data/raw/casals_l1b').glob('*.h5')):
    stem=h5path.stem
    laz=next(Path('data/reference/3dep').glob(stem+'*.laz'))
    side=side_root/(laz.name+'.json')
    data=json.loads(side.read_text(encoding='utf8'));plan=data['clip_plan']
    with h5py.File(h5path) as h:
        n = h[tr.find_dataset_path(h, ["refh"])].size
    # Predeclared middle 64 sweeps; original record indices preserved.
    start=(n//2//256)*256
    cas=tr.read_casals_h5(h5path,slice(start,start+16384))
    with laspy.open(laz) as reader: headercrs=reader.header.parse_crs()
    horiz,vert,frame=cr.resolve_reference_frame(headercrs,side)
    xyz=tr.project_casals_to_target(cas,horiz)
    bounds=(xyz[:,0].min()-6,xyz[:,1].min()-6,xyz[:,0].max()+6,xyz[:,1].max()+6)
    dep=tr.read_3dep_las(laz, bounds, max_points=1000000)
    print(stem,'selected 3DEP',len(dep['xyz']),flush=True)
    flags=tr.build_quality_flags(cas,xyz,tr.CONFIG['min_ground_snr'])
    dz,inliers,fit=tr.estimate_ground_vertical_shift(cas,xyz,dep['xyz'],dep['classification'],tr.CONFIG,flags)
    aligned=xyz.copy();aligned[:,2]+=dz
    before=oldtr.transfer_labels_to_casals(aligned,dep['xyz'],dep['classification'],oldtr.CONFIG)
    after=tr.transfer_labels_to_casals(aligned,dep['xyz'],dep['classification'],tr.CONFIG)
    cfg={**cl.DEFAULT_CONFIG,'CLASSIFIER_MODE':'height_only'}
    grid=cl.build_ground_grid(xyz[:,0],xyz[:,1],cas['z'],cas['fields']['refh_snr'],cfg)
    ground,valid=cl.sample_ground_grid_idw(xyz[:,0],xyz[:,1],grid,cfg)
    args=(cas['z'],ground,valid,np.full(len(xyz),np.nan),cfg)
    oldpred=oldcl.classify_points_baseline(*args)['pred_class_baseline']; pred=cl.classify_points_baseline(*args)['pred_class_baseline']
    metrics=[]
    for label,mod,truth,prediction in [('old_all_matched',oldev,before,oldpred),('new',ev,after,pred)]:
        m=mod.evaluate_classification(prediction,ev.map_reference_labels_to_baseline_classes(truth['classification']),np.ones(len(xyz)),valid,truth['transfer_status'],truth['nearest_3dep_distance_m'],truth['class_vote_ratio'],cfg)
        for row in m['evaluation_summary_rows']:
            metrics.append({ 'version':label,**row})
    pd.DataFrame(metrics).drop(columns=['match_status_counts'],errors='ignore').assign(metric_semantics='exploratory pseudo-reference agreement', independent_accuracy_claim=False).to_csv(out/(stem+'_evaluation.csv'),index=False)
    # Exercise LAS round trip identity contract, keeping transient point products out of deliverables.
    cas['alignment_mode']='empirical_diagnostic'
    import tempfile
    tempdir = tempfile.TemporaryDirectory()
    temp = Path(tempdir.name) / (stem + ".laz")
    tr.write_labeled_las(temp,cas,xyz,aligned,flags,inliers,after,horiz,dz)
    ref=ev.read_reference_labels(temp)
    identity=ev.align_prediction_to_reference(dict(source_h5=str(h5path.resolve()),point_index=cas['point_index'],longitude=cas['lon'],latitude=cas['lat'],refh_original_m=cas['z']),ref,cfg)
    assert np.count_nonzero(identity['eval_match_valid'])==len(xyz)
    tempdir.cleanup()
    response = requests.get(plan["ept_url"], timeout=30)
    response.raise_for_status()
    ept = response.json()
    srs=ept.get('srs',{})
    boundsll=(float(cas['lon'].min()),float(cas['lat'].min()),float(cas['lon'].max()),float(cas['lat'].max()))
    group=cr.build_transformer_group(CRS.from_epsg(4979),cr.build_target_compound_crs(horiz,vert),boundsll)
    missing=cr.collect_missing_grids(group)
    summary=dict(granule=stem,source_h5=str(h5path.resolve()),source_3dep=str(laz.resolve()),smoke_only=True,baseline_commit=BASELINE_COMMIT,
      original_indices=[int(start),int(start+16384)],n_casals=len(xyz),reference_selection=dep['header_summary'],
      reference_frame=cr.asdict(frame),las_horizontal=headercrs.to_string(),workunit=plan['workunit'],
      workunit_horizontal=plan['horiz_crs'],workunit_vertical=plan['vert_crs'],geoid=plan['geoid'],
      evidence_sidecar=str(side),metadata_link=plan['metadata_link'],ept_url=plan['ept_url'],ept_srs={key:srs.get(key) for key in ('authority','horizontal','vertical')},
      historical_pdal_processing='old pipeline used filters.reprojection out_srs only; both advertised EPT and output CRS are 2D. No recorded operation or measured source/output Z check; historical no-vertical-transform flag is not proof.',
      casals_frame={'horizontal':'WGS84','vertical':'WGS84 ellipsoidal height','status':'documented convention; source realization/epoch and processing confirmation unresolved'},
      strict_xyz_ready=False,strict_blockers=['CASALS source frame remains pending verification', 'historical clip Z handling has no measured processing audit']+(['required best PROJ operation unavailable'] if not group.best_available else []),
      proj_best_available=group.best_available,proj_missing_grids=sorted({g["grid_name"] for g in missing if g.get("grid_name") and not g.get("available")}),proj_available_operations=[t.description for t in group.transformers],
      alignment_mode='empirical_diagnostic',empirical_alignment=True,dz_m=float(dz),dz_fit_inlier_count=int(inliers.sum()),fit_residuals='fit diagnostics only; not independent accuracy; no holdout elevation evaluation performed',
      transfer_status_counts={tr.STATUS_NAMES[int(v)]:int(c) for v,c in zip(*np.unique(after['transfer_status'],return_counts=True))},
      old_class_counts=tr.collect_class_counts(before['classification']),new_class_counts=tr.collect_class_counts(after['classification']),
      classifier='unchanged height_only baseline on identical local CASALS DTM; no vegetation/building model',
      dtm_missing_count=int(np.count_nonzero(valid==0)),metric_semantics='exploratory pseudo-reference agreement',independent_accuracy_claim=False,identity_roundtrip='all original indices/lon/lat/refh and H5 path verified')
    (out/(stem+'_summary.json')).write_text(json.dumps(summary,indent=2,default=str),encoding='utf8')
    print(stem,'finished',[(r['version'],r['subset_name'],r['n_points'],r.get('accuracy')) for r in metrics],flush=True)

