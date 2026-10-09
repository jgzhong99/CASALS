"""Supplement fresh audit with N-1 beam spacing and evidence-based report.

Reads the NEW pulse diagnostics and original geometry (no archived outputs).
Does not rerun or replace raw maxima, or modify the audit's predeclared filters.
"""
import argparse
import json
from pathlib import Path
from scipy.signal import find_peaks, peak_widths

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from research.geolocation.bin_georeferencing_audit.run_audit import (
    ROOT, inventory, read_values, xyz, beam, bin_to_xyz, statistics, sha256,
)


def supplement(directory):
    meta=json.loads((directory/'run_metadata.json').read_text())
    if not meta['full_traversal']:raise ValueError('NOT VERIFIED: full original audit required')
    target=directory/'beam_spacing_sensitivity.parquet'
    if target.exists():raise FileExistsError(target)
    writer=None
    with h5py.File(meta['source_h5'],'r') as h:
        _,paths=inventory(h)
        cols=['pulse_index','raw_argmax_bin','valid','high_quality','H2_along_m','H2_perp_m']
        try:
            for batch in pq.ParquetFile(directory/'pulse_diagnostics.parquet').iter_batches(batch_size=meta['chunk_size'],columns=cols):
                d=batch.to_pandas();lo=int(d.pulse_index.iloc[0]);hi=int(d.pulse_index.iloc[-1])+1
                if not np.array_equal(d.pulse_index,np.arange(lo,hi)):raise ValueError('Unexpected record layout')
                get=lambda name:read_values(h[paths[name]],slice(lo,hi))
                a=xyz(get('rwstart_longitude'),get('rwstart_latitude'),get('rwstart'))
                z=xyz(get('rwstop_longitude'),get('rwstop_latitude'),get('rwstop'))
                r=xyz(get('refh_longitude'),get('refh_latitude'),get('refh'))
                bd=beam(get('refh_longitude'),get('refh_latitude'),get('local_beam_azimuth'),get('local_beam_elevation'))
                length=np.linalg.norm(z-a,axis=1);n=meta['n_rx_bins'];b=d.raw_argmax_bin.to_numpy()
                out={'pulse_index':d.pulse_index.to_numpy(),'valid':d.valid.to_numpy(),'high_quality':d.high_quality.to_numpy()}
                t=(b+.5)/n
                out['H2_along_m']=d.H2_along_m.to_numpy()
                out['H2_perp_m']=d.H2_perp_m.to_numpy()
                out['t_times_one_minus_t']=t*(1-t)
                if all(k in paths for k in ['refh_neutat_delay_total','rwstart_neutat_delay_total','rwstop_neutat_delay_total']):
                    out['neutat_ref_minus_linear']=get('refh_neutat_delay_total')-(get('rwstart_neutat_delay_total')+t*(get('rwstop_neutat_delay_total')-get('rwstart_neutat_delay_total')))
                for label,tb in [('first',0),('last',n-1)]:
                    seg=a+(tb+.5)/n*(z-a)
                    anchor=r+(tb-b)[:,None]*length[:,None]/(n-1)*bd
                    out['H5_Nminus1_vs_H2_'+label+'_m']=np.where(d.valid,np.linalg.norm(anchor-seg,axis=1),np.nan)
                    # Unknown unit: numerical sensitivity only, not physical H6 validation.
                    numeric=r+(tb-b)[:,None]*get('bin_size')[:,None]*bd
                    out['stored_numeric_vs_H2_'+label+'_m']=np.where(d.valid,np.linalg.norm(numeric-seg,axis=1),np.nan)
                table=pa.Table.from_pydict(out)
                if writer is None:writer=pq.ParquetWriter(target,table.schema,compression='zstd')
                writer.write_table(table)
        finally:
            if writer:writer.close()
        reps=pd.read_csv(directory/'representative_pulses.csv');rows=[]
        for i,g in reps.groupby('pulse_index'):
            get=lambda name:np.array([read_values(h[paths[name]],int(i))])
            a=xyz(get('rwstart_longitude'),get('rwstart_latitude'),get('rwstart'))[0]
            z=xyz(get('rwstop_longitude'),get('rwstop_latitude'),get('rwstop'))[0]
            r=xyz(get('refh_longitude'),get('refh_latitude'),get('refh'))[0]
            bd=beam(get('refh_longitude'),get('refh_latitude'),get('local_beam_azimuth'),get('local_beam_elevation'))[0]
            b=g['bin'].to_numpy();br=g.raw_argmax.iloc[0];n=meta['n_rx_bins'];length=np.linalg.norm(z-a)
            h2=bin_to_xyz(a,z,b,n)
            models={'H1':bin_to_xyz(a,z,b,n,convention='H1'),
                    'H3':bin_to_xyz(a,z,b,n,convention='H3'),
                    'H5_Nminus1':r+(b-br)[:,None]*length/(n-1)*bd,
                    'H5_edge':r+(b-br)[:,None]*length/n*bd,
                    'stored_numeric_UNDOCUMENTED':r+(b-br)[:,None]*get('bin_size')[0]*bd}
            for method,p in models.items():
                for j in range(len(g)):
                    rows.append({'pulse_index':int(i),'sweep_num':int(g.sweep_num.iloc[j]),'track_num':int(g.track_num.iloc[j]),
                                 'bin':b[j],'raw_argmax':br,'is_detected_raw_peak':g.is_detected_raw_peak.iloc[j],
                                 'method':method,'x':p[j,0],'y':p[j,1],'z':p[j,2],
                                 'difference_from_H2_m':np.linalg.norm(p[j]-h2[j]),
                                 'model_status':'numerical_sensitivity_only' if method=='stored_numeric_UNDOCUMENTED' else 'experimental_geometry'})
        pd.DataFrame(rows).to_csv(directory/'non_refh_model_comparison.csv',index=False)
        # Review raw morphology with stronger contrast, independently of closure.
        reviewed=[];coverage=set()
        snr_table=pq.read_table(directory/'pulse_diagnostics.parquet',columns=['pulse_index','refh_snr']).to_pandas()
        candidate_ids=set(reps.pulse_index.astype(int)) | set(snr_table.nlargest(5,'refh_snr').pulse_index.astype(int))
        for i in sorted(candidate_ids):
            wave=read_values(h[paths['rx_waveform']],(int(i),slice(None)))
            peaks,properties=find_peaks(wave,prominence=max(5,.25*np.ptp(wave)),height=np.median(wave)+.45*np.ptp(wave),distance=20)
            widths=peak_widths(wave,peaks)[0] if len(peaks) else np.array([])
            kind='single' if len(peaks)==1 else 'double' if len(peaks)==2 else 'multiple' if len(peaks)>2 else 'weak'
            labels={kind}
            if len(widths) and max(widths)>=20:labels.add('wide')
            snr=float(read_values(h[paths['refh_snr']],int(i)));br=int(wave.argmax())
            if snr<3:labels.update(['weak','low_snr'])
            if snr>=10:labels.add('high_snr')
            if br<20 or br>meta['n_rx_bins']-21:labels.add('boundary')
            if np.sum(wave>=wave[br]-1)>1:labels.add('near_tie')
            if not (labels-coverage):continue
            coverage.update(labels)
            get=lambda name:np.array([read_values(h[paths[name]],int(i))])
            a=xyz(get('rwstart_longitude'),get('rwstart_latitude'),get('rwstart'))[0]
            z=xyz(get('rwstop_longitude'),get('rwstop_latitude'),get('rwstop'))[0]
            pp=bin_to_xyz(a,z,peaks,meta['n_rx_bins']) if len(peaks) else np.empty((0,3))
            r=xyz(get('refh_longitude'),get('refh_latitude'),get('refh'))[0]
            fig=plt.figure(figsize=(11,4.8));ax=fig.add_subplot(121);ax.plot(wave,lw=.65)
            ax.scatter(peaks,wave[peaks],s=22,color='darkorange');ax.axvline(br,color='crimson',ls='--',label='raw argmax');ax.legend()
            # Label at most four strongest peaks to keep the geometry readable.
            chosen=np.argsort(wave[peaks])[-4:] if len(peaks) else []
            for j in chosen:ax.annotate(str(peaks[j]),(peaks[j],wave[peaks[j]]),xytext=(2,5),textcoords='offset points',fontsize=8)
            ax.set(xlabel='Zero-based RX bin',ylabel='Raw stored amplitude',title=f'Pulse {int(i)}; '+', '.join(sorted(labels)))
            ax2=fig.add_subplot(122,projection='3d');line=np.array([a-r,z-r]);ax2.plot(*line.T,label='RW edges')
            if len(peaks):ax2.scatter(*(pp-r).T,color='darkorange',label='Raw prominent samples')
            ax2.scatter([0],[0],[0],color='crimson',label='Official Refh')
            # Bin numbers are labeled on the waveform; avoid overlapping 3D text.
            center=np.mean(line,axis=0);radius=np.ptp(line,axis=0).max()*.55
            ax2.set_box_aspect((1,1,1))
            ax2.set_xlim(center[0]-radius,center[0]+radius)
            ax2.set_ylim(center[1]-radius,center[1]+radius)
            ax2.set_zlim(center[2]-radius,center[2]+radius)
            from matplotlib.ticker import MaxNLocator
            for axis in [ax2.xaxis,ax2.yaxis,ax2.zaxis]:axis.set_major_locator(MaxNLocator(3))
            ax2.set(xlabel='ECEF ΔX (m)',ylabel='ΔY (m)',zlabel='ΔZ (m)',title='H2 sample positions; physical surface unverified')
            ax2.legend(fontsize=7);fig.tight_layout();fig.savefig(directory/f'figures/reviewed_pulse_{int(i)}_{kind}.png',dpi=160);plt.close(fig)
            reviewed.append({'pulse_index':int(i),'sweep_num':int(read_values(h[paths['sweep_num']],int(i))),'track_num':int(read_values(h[paths['track_num']],int(i))),
                             'labels':','.join(sorted(labels)),'raw_peaks':','.join(map(str,peaks)),
                             'raw_argmax':br,'snr':snr,'n_raw_peaks':len(peaks),
                             'selection':'original seeded examples plus top5 stored SNR; raw prominence>=25% span, height>=median+45% span, separation>=20; labels independent of closure'})
        pd.DataFrame(reviewed).to_csv(directory/'reviewed_raw_examples.csv',index=False)
        print('  reviewed raw morphology coverage:',sorted(coverage),flush=True)
    masks=pq.read_table(target,columns=['high_quality']).to_pandas().high_quality.to_numpy()
    rows=[]
    for name in pq.ParquetFile(target).schema_arrow.names[3:]:
        values=pq.read_table(target,columns=[name])[name].to_numpy()
        for subset,mask in [('all',None),('high_quality',masks)]:
            rows.append({'metric':name,'subset':subset,**statistics(values,mask)})
    pd.DataFrame(rows).to_csv(directory/'beam_spacing_sensitivity_summary.csv',index=False)
    q=pq.read_table(target,columns=['H2_along_m','t_times_one_minus_t','neutat_ref_minus_linear']).to_pandas()
    pd.DataFrame([{'x':'H2_along_m','y':k,'pearson_r':q.H2_along_m.corr(q[k]),
                   'interpretation':'exploratory correlation; not causal attribution or correction reapplication'}
                  for k in ['t_times_one_minus_t','neutat_ref_minus_linear']]).to_csv(directory/'residual_correlations.csv',index=False)
    # Explicit-argmax subset, separate from SNR-filtered high quality.
    raw=pq.read_table(directory/'pulse_diagnostics.parquet',columns=['valid','tie_count','raw_max','second_max','bin_delta_H1','bin_delta_H2']).to_pandas()
    mask=raw.valid & (raw.tie_count==1) & (raw.raw_max-raw.second_max>1)
    pd.DataFrame([{'method':m,'n_total':len(raw),'n_used':int(mask.sum()),'n_rejected':int((~mask).sum()),
                   'within_half_bin':float((abs(raw.loc[mask,'bin_delta_'+m])<=.5).mean()),
                   'within_one_bin':float((abs(raw.loc[mask,'bin_delta_'+m])<=1).mean()),
                   **statistics(raw['bin_delta_'+m].to_numpy(),mask.to_numpy())} for m in ['H1','H2']]).to_csv(directory/'unambiguous_argmax_agreement.csv',index=False)
    save={'source_h5':meta['source_h5'],'original_audit_script_sha256':meta['script_sha256'],
          'supplement_script_sha256':sha256(Path(__file__)),'n_audited':meta['n_audited'],
          'meaning':'H5 uses length/(N-1); H5_edge uses length/N; stored_numeric has unknown units and is not an available physical H6 model',
          'raw_source':'raw maxima from fresh full original-H5 audit; geometry newly reread from original H5'}
    (directory/'supplement_metadata.json').write_text(json.dumps(save,indent=2),encoding='utf-8')
    finish_artifacts(directory)


def finish_artifacts(directory):
    """Exact screening counts, correction diagnostics, and field interpretation table."""
    meta=json.loads((directory/'run_metadata.json').read_text())
    parquet=directory/'pulse_diagnostics.parquet'
    raw=pq.read_table(parquet,columns=['valid','high_quality','bin_delta_H1','bin_delta_H2']).to_pandas()
    rows=[]
    for subset,mask in [('all',raw.valid.to_numpy()),('high_quality',raw.high_quality.to_numpy())]:
        for method in ['H1','H2']:
            x=raw['bin_delta_'+method].to_numpy()[mask]
            x=x[np.isfinite(x)]
            rows.append({'subset':subset,'model':method,**statistics(abs(x)),
                         'population_total':len(raw),'n_used':len(x),'screening_rejected':len(raw)-len(x),
                         'within_half_bin':np.mean(abs(x)<=.5) if len(x) else np.nan,
                         'within_one_bin':np.mean(abs(x)<=1) if len(x) else np.nan,
                         'signed_bias':np.median(x) if len(x) else np.nan})
    pd.DataFrame(rows).to_csv(directory/'bin_agreement_with_counts.csv',index=False)
    q=pq.read_table(directory/'beam_spacing_sensitivity.parquet',columns=['H2_along_m','neutat_ref_minus_linear']).to_pandas()
    difference=q.H2_along_m-q.neutat_ref_minus_linear
    pd.DataFrame([{'metric':'H2_along_minus_neutat_linear_residual','unit':'numerically compared stored correction to meters; correction unit undocumented',
                   **statistics(difference.to_numpy())},
                  {'metric':'abs_H2_along_minus_neutat_linear_residual',**statistics(abs(difference.to_numpy()))}]).to_csv(directory/'correction_nonlinearity_diagnostic.csv',index=False)
    pd.DataFrame([
        {'field':'refh','definition':'height of RX maximum-amplitude bin; WGS84 ellipsoidal meters','evidence':'Documented','source':meta['tutorial_url']},
        {'field':'rwstart/rwstop triplets','definition':'3D window edges consistent with half-bin centered samples; heights treated as WGS84 ellipsoidal','evidence':'Empirically inferred','source':'new full H5 geometry/timing/closure; semantic attrs absent'},
        {'field':'bin_size','definition':'numerically equals (rwstart height-rwstop height)/N; does not equal along-beam spacing','evidence':'Empirically inferred','source':'stored_minus_height_step全量统计'},
        {'field':'bounce_time_offsets','definition':'center bin satisfies N*(ref-start)/(stop-start)-0.5=raw bin; sample increment5e-10 stored units','evidence':'Empirically inferred','source':'full time_bin_delta_H2 / bounce_step_per_bin'},
        {'field':'local beam angles','definition':'radians; north-clockwise; elevation from horizon; encoded toward sensor; negated at Refh ENU','evidence':'Empirically inferred','source':'full few-convention beam comparison; historical candidate only'},
        {'field':'range_bias / neutat / geoid / tides / dac','definition':'units and applied status not formally established; no corrections reapplied','evidence':'Unknown','source':'H5 lacks semantic attrs; processor not available'},
        {'field':'coordinate epoch / instrument pose time / lever arm','definition':'not established','evidence':'Unknown','source':'no complete corresponding processor/GPS/pose intermediates available'}
    ]).to_csv(directory/'field_explanations.csv',index=False)


def report(root):
    dirs=sorted(p for p in root.iterdir() if (p/'run_metadata.json').exists())
    tables={p.name:pd.read_csv(p/'all_metrics_summary.csv') for p in dirs}
    def row(p,metric,subset='all'):
        q=tables[p.name];return q[(q.metric==metric)&(q.subset==subset)].iloc[0]
    def table(metrics):
        return pd.DataFrame([{'metric':m,'stat':s,**{p.name[11:19]:row(p,m)[s] for p in dirs}} for m,s in metrics]).to_markdown(index=False,floatfmt='.9g')
    numeric=table([(m,s) for m in ['refh_line_distance_m','refh_segment_distance_m','H1_3d_m','H2_3d_m','H3_3d_m','bin_delta_H1','bin_delta_H2',
                   'edge_spacing_m_per_bin','segment_spacing_m_per_bin','stored_bin_size','stored_minus_height_step','t_time_minus_geom',
                   'time_bin_delta_H2','bounce_step_per_bin','RW_vs_beam_refh_deg','RW_vs_beam_instrument_deg','RW_vs_IR_deg','conditional_c_half_minus_length_m']
                  for s in ['median','P95','max']])
    hashes=[];counts=[];agreement=[];history=[];anomaly=[];examples=[];unknown=[]
    for p in dirs:
        m=json.loads((p/'run_metadata.json').read_text());hashes.append(f"- `{p.name}.h5`: `{m['sha256_before']}`; before/after SHA256 identical.")
        counts.append({'date':p.name[11:19],'pulses':m['n_audited'],'RX_bins':m['n_rx_bins'],'valid':round(row(p,'valid')['mean']*m['n_audited']),
                       'high_quality':round(row(p,'high_quality')['mean']*m['n_audited']),
                       'outside_segment':m['invalid_counts_overlap']['outside_segment'],
                       'H2_gt_10cm':m['invalid_counts_overlap']['H2_closure_gt_10cm']})
        q=pd.read_csv(p/'bin_agreement_with_counts.csv');q.insert(0,'date',p.name[11:19]);agreement.append(q)
        q=pd.read_csv(p/'historical_near_tie_case.csv');q.insert(0,'date',p.name[11:19]);history.append(q)
        q=pd.read_csv(p/'anomalous_pulses.csv');q=q[q.selection_reason=='top10_H2_3d_m'].head(3)
        q=q[['pulse_index','sweep_num','track_num','H2_3d_m','bin_delta_H2','refh_line_distance_m']];q.insert(0,'date',p.name[11:19]);anomaly.append(q)
        reviewed=pd.read_csv(p/'reviewed_raw_examples.csv')
        proposed=reviewed[(reviewed.n_raw_peaks>=2)&(reviewed.snr>=3)].sort_values('n_raw_peaks')
        q=pd.read_csv(p/'representative_pulses.csv');q=q[q.is_detected_raw_peak & (q['bin']!=q.raw_argmax)]
        if len(proposed) and proposed.pulse_index.iloc[0] in q.pulse_index.values:
            choice=proposed.iloc[0]
            raw_peaks=np.array([int(b) for b in str(choice.raw_peaks).split(',')])
            q=q[(q.pulse_index==choice.pulse_index)&q['bin'].isin(raw_peaks)]
        if len(q):
            pid=q.pulse_index.iloc[0];q=q[q.pulse_index==pid].head(5)
            q=q[['pulse_index','sweep_num','track_num','category','bin','raw_argmax','longitude','latitude','ellipsoidal_height','H5_minus_H2_m']]
            q.insert(0,'date',p.name[11:19]);examples.append(q)
        coverage=set(','.join(reviewed.labels).split(','))
        missing=sorted(set(['single','double','multiple','wide','weak','boundary','low_snr','high_snr','near_tie'])-coverage)
        unknown.append(f"- {p.name}: missing fields {m['missing_fields']}; invalid/overlapping-reason counts in `run_metadata.json`; reviewed raw morphology missing categories {missing}. 初轮sample_selection.json中的类别因互斥优先标签可能掩盖形态，补充reviewed_raw_examples.csv采用并行标签，单独记录更强对比选择规则。")
    correlations=pd.concat([pd.read_csv(p/'residual_correlations.csv').assign(date=p.name[11:19]) for p in dirs])
    correction=pd.concat([pd.read_csv(p/'correction_nonlinearity_diagnostic.csv').assign(date=p.name[11:19]) for p in dirs])
    models=pd.read_csv(root/'comparison/model_comparison.csv')
    formulas={'H1':'P0+b/(N-1)*v','H2':'P0+(b+.5)/N*v','H3':'P0+(1-b/(N-1))*v',
              'H4_empirical':'P0+((b+.5)*5e-10/(tau_stop-tau_start))*v',
              'H5':'Pr+(b-br)*length/N*d_beam','H6':'Pr+(b-br)*stored_bin_size*d_beam (unit unknown)'}
    models['formula']=models.model.map(formulas)
    models['required_inputs']=models.model.map(lambda m:'RW triplets,N' if m in ['H1','H2','H3'] else 'RW triplets,N,offset span,inferred sampling' if m=='H4_empirical' else 'Refh triplet,raw br,beam angles,Refh local basis,spacing')
    models['recommendation']=models.model.map({'H1':'not preferred: center convention mismatch','H2':'research recommended, official semantics pending',
        'H3':'reject reversed mapping','H4_empirical':'timing diagnostic only','H5':'model consistency only; tautological closure','H6':'unavailable physical range model'})
    extras=[]
    for p in dirs:
        extras.append({'granule':p.name,'model':'H5_Nminus1','status':'EVALUATED','refh_independent':False,
                       'coordinate_reference':'EPSG:4978 meters','n_total':json.loads((p/'run_metadata.json').read_text())['n_audited'],
                       'n_valid':row(p,'H5_3d_m')['n_valid'],'valid_fraction':row(p,'H5_3d_m')['valid_fraction'],
                       'median':0,'P95':0,'max':0,'formula':'Pr+(b-br)*length/(N-1)*d_beam',
                       'required_inputs':'Refh triplet,raw br,beam angles,Refh local basis,RW length,N',
                       'limitations':'zero closure by definition; see beam_spacing_sensitivity_summary.csv',
                       'recommendation':'model consistency only'})
    pd.concat([models,pd.DataFrame(extras)],ignore_index=True).to_csv(root/'comparison/candidate_contract.csv',index=False)
    # Gate Level 2 conservatively; a large unexplained geometry bias cannot pass.
    meets=all(row(p,'H2_3d_m')['P95']<=.1 and row(p,'refh_line_distance_m')['P95']<=.002 for p in dirs)
    level='Level 1 — Internally consistent'
    content=f'''# CASALS L1B 任意 RX bin georeferencing：严格验证报告

日期：2026-10-09。结论：**{level}**。推荐研究用 H2：

`P(b) = P_RWSTART + ((b + 0.5) / N) * (P_RWSTOP - P_RWSTART)`。

两份完整原始 H5 重新遍历；没有使用历史 CSV/Notebook 结果替代本轮数据。
该判断不认证任意次峰的官方级或绝对三维精度，也不合并到 production。
Level 2 数值目标的部分几何项满足状态：`{meets}`；字段语义、修正应用及高质量子集的证据门槛仍须逐项看下文。
H2 是当前两个产品的经验上最可靠样本中心映射；不能把 RWSTART/RWSTOP 说成首末样本中心。
H1也可能通过给定的宽松closure／half-bin阈值；阈值不是模型唯一性证明。
选择H2依据连续的bin-center时间关系和亚毫米残差解释，而非为H1或H2调整验收条件。

## 数据与复现证据

{pd.DataFrame(counts).to_markdown(index=False)}

{chr(10).join(hashes)}

RX 实际为 int16、record axis=0、zero-based rx_bins；storage chunks=(14080,11)，逐批14080 pulse读取。
每份 pulse_diagnostics.parquet 保留全部 pulse，包括无效状态；无大规模 waveform 输出。
每个标量的 quantile/NMAD 精确取自全量诊断，并非随机 sample。图中 CDF/scatter 为确定性稀疏绘图样本，统计表为全量。
筛选原因重叠计数，不能将相加当作唯一 invalid 总数。有效几何与原始波形状态分别保存。
两份历史数据都已被早期研究接触，不是盲测；本次固定 H1/H2/H3 与少量历史支持 beam convention。
H4 的5e-10增量只从11月12日首14080条识别，之后冻结。没有 shift/scale 调参。
连续512-sweep blocks中 block%4==3 是验证区，其余为探索区；split_summary.csv逐项报告。
block_uncertainty.csv提供2000次固定seed block-bootstrap的“block median的median”区间，不能视为独立空间真值误差区间。

## 全量数值对照

单位：距离m，角度degree，bin delta为bin；offset单位未正式确认。P95对signed量是signed percentile；absolute bin统计另见bin_agreement.csv。

{numeric}

## 1–5：端点、映射和整数／fractional bin

1. RWSTART/RWSTOP 实际有height、lon、lat三元组，因此可作为3D位置进行研究计算；H5没有这些字段的语义attrs。
   `refh` 的WGS84椭球高由[NASA/CryoCloud官方教程](https://book.cryointhecloud.com/l1b-waveforms-tutorial/)支持。
   RW height和instrument altitude采用同一椭球约定是经验支持的工作假设，官方逐字段frame/epoch仍未知。
2. 与bin 0和N−1的关系：数值证据支持RW是window edges，sample centers位于 `(b+0.5)/N`。
   H1的首末bin-center假设产生随bin变化的半bin形态偏差，不能解释成已确认的官方约定。
   用zero-based数组索引b；若输入一基编号j，则b=j−1，不能直接代入j。
3. 当前推荐H2；逆向H3被闭合结果否定。H4经验时间候选与H2近乎等价，不是另一份完全独立定位模型。
4. 任意整数bin使用H2 affine ECEF式，再从EPSG:4978转换EPSG:4979得到lon/lat/ellipsoidal h。
5. Fractional bin直接使用相同连续式，不round、不clip。法律域0≤b≤N−1；数学window edges是−0.5和N−0.5，公开函数拒绝域外edge索引。
   真实非参考样本检验端点、argmax附近+.4bin、远离Refh的quartile bins和raw局部峰；代表样本CSV保存finite/连续性step残差。

## 6–8：bin_size、时间、beam

6. stored bin_size没有units/description。当前最显著新证据是它与 `(rwstart-rwstop)/N` 的高度步长相等，而与ECEF沿束步长不同。
   `stored_minus_sin_elevation_step`提供另一几何关系检验。不能直接当沿束m/bin使用，亦不能将数值相等升级为正式字段定义。
   本轮定位优先使用window edge length/N；length/(N−1)是H1的spacing，必须分开。
7. 时间比例 `(tau_refh-tau_start)/(tau_stop-tau_start)` 与几何参数比较，反推center bin使用 `N*t_time−0.5`。
   window offset span/N 的5e-10增量由全量比较检验；绝对offset是约e-5量级，不足以决定clock、双程/单程、sign和时刻含义。
   global `sec_to_meters=c/2`、`ns_to_meters` 是记录的处理常数，但没有直接把offset字段绑定seconds的说明。
   conditional `span*c`/`span*c/2`诊断明确标为假设单位秒；segment的光速差异不能武断归因某一已应用修正。
   在“offset单位秒”的条件下，实测span*c接近完整RW长度，而span*c/2约为一半：

{table([('segment_length_m','median'),('bounce_span','median'),('conditional_span_times_c_m','median'),('conditional_span_times_c_half_m','median'),('conditional_c_half_minus_length_m','median')])}

   因此这些bounce offset的数值行为不支持直接作为raw双程传播时延再乘c/2。
   5e-10若为单程bounce increment，对应raw双程1ns采样与global ns_to_meters常数相容；
   这仍是处理后字段关系的经验解释，clock/time-zero及processing convention须CASALS确认。
8. radian、north-clockwise、horizon elevation、negated ENU在Refh local basis与RW方向相符。
   同一规则的instrument basis、错误degree解释和反向sign也有全量对照；未做无边界angle搜索。
   RW/IR/beam方向、instrument beam line端点perpendicular distance均已计算。
   非零instrument差异可能包含local basis、时刻、frame或lever arm，不能全部归罪beam角。

## 候选方法及证据独立性

|候选|所需字段／数学定义|判定与限制|
|---|---|---|
|H1|RW三元组、N；P0+b/(N−1)*v|闭合厘米级；center约定不匹配当前时间/bin关系|
|H2|RW三元组、N；P0+(b+.5)/N*v|研究推荐；无Refh锚定；仍须官方字段定义和修正模型确认|
|H3|RW三元组、N；P0+(1−b/(N−1))*v|FAIL：反向索引不支持|
|H4 official|sample timing定义和offset单位|UNAVAILABLE：缺明确sample timing／单位契约|
|H4 empirical|RW、offsets；tau_start+(b+.5)*5e−10|经验等价H2；时间关系支持内部一致性，不是独立绝对精度|
|H5 N−1 spacing|Refh三元组、raw br、beam角和local basis；Pr+(b−br)*length/(N−1)*d|锚定闭合恒为0；作为模型比较，不能定位精度验收|
|H5 edge spacing|同上，length/N|另一spacing版本，与H2的真实非refh一致性报告|
|H6 stored spacing|同上，stored bin_size|物理range model UNAVAILABLE；数值敏感性计算仅为诊断，units/height-vs-range未正式确认|

两种H5 spacing的全量first/last sample discrepancy见beam_spacing_sensitivity_summary.csv。
各模型的真实secondary／fractional比较见non_refh_model_comparison.csv，保存XYZ和相对H2差异。
H5在br处closure=0是定义；H5/segment agreement也不是外部精度。
所有candidate ECEF单位m，geodetic是EPSG:4979下椭球高；endpoint输入unit/frame仍受上述契约限制。

## 9–10：厘米偏差、近并列和异常

H1相对H2的parameter差为 `(b−(N−1)/2)/(N*(N−1))`，所以存在bin位置依赖的最多约半bin的沿束差。
这一差异无需人为平移即可解释主要厘米级H1残差；remaining H2 residual不得自动归因已知校正。
补充读取原始neutat_delay_total，计算 `delay_refh−[delay_start+t_center*(delay_stop−delay_start)]`。
它与signed H2沿束残差的数值关系如下，未将任何修正重新应用于坐标：

{correlations[['date','x','y','pearson_r']].to_markdown(index=False,floatfmt='.9g')}

{correction[correction.metric=='abs_H2_along_minus_neutat_linear_residual'][['date','median','P95','max']].to_markdown(index=False,floatfmt='.9g')}

这一近似等量的数值证据支持“不同range处大气修正非线性影响endpoint interpolation residual”的解释，
比单独相关性更强，但修正字段单位／applied状态和官方处理过程仍未知，不能据此私自修正任意bin或宣称处理器parity。
H2 affine给出endpoint模型坐标，未捕捉逐range非线性；当前对Refh实测的亚毫米差不可自动外推为所有bin的误差上界。
原始max规则与同值／近值计数保留，未固定任何bin改善closure。

历史案例在两份新读原始数据中的实际值：

{pd.concat(history).to_markdown(index=False)}

int16存储值的numpy argmax在float64转换后完全精确；当前记录不能据此证明浮点环境导致1693/1694逆转。
若旧实验使用smoothing、fitting、不同产品版本、不同pulse索引或其他waveform表示，它的近并列结论不能直接转移到当前raw H5。
归档v2中beam winner在Refh basis、radian、north-clockwise、negated elevation方向的假设可复核；
归档的b/(N−1)不能作为“唯一正确约定”延续，当前增加H2后具有更强内部支持。

最大H2残差具体记录（不从总体删去）：

{pd.concat(anomaly).to_markdown(index=False,floatfmt='.9g')}

每份anomalous_pulses.csv同时包括line距离、beam角、同track的segment长度／方向跳变前十条。
异常的threshold计数、geometry/RX rejection、out-of-bounds和instrument order见run_metadata.json；
同track连续性表不把不同track的交错record当作同一扫描线跳变。
高质量阈值SNR≥10、unique maximum、top-two差>1 ADU，未依赖闭合误差调参；为空时不补造样本。

{pd.concat(agreement)[['date','subset','model','population_total','n_used','screening_rejected','within_half_bin','within_one_bin','signed_bias']].to_markdown(index=False,floatfmt='.9g')}

另提供不要求SNR阈值的明确argmax子集unambiguous_argmax_agreement.csv，保留unfiltered总体。

## 11：真实非Refh示例

{pd.concat(examples).to_markdown(index=False,floatfmt='.9g') if examples else 'NOT VERIFIED: no usable raw secondary morphology example'}

代表pulse具有真实raw waveform局部峰，图示waveform和对应相对Refh ECEF3D。
这些未确认峰来自冠层、地面或建筑；raw局部峰和sample bin不能直接称独立物理scatterer。
宽峰／低SNR／边界／近并列等类别只按raw形态和stored SNR标注；缺失类别明确记录，不fabricate。
{chr(10).join(unknown)}

## 12–13：能主张的精度与官方问题

能主张的是本轮两个granule内部的Refh closure／时间／beam数值一致性及研究函数的连续性。
不能把这些差异当绝对空间精度、secondary return uncertainty、官方处理器parity或ground accuracy。
**保守判定Level 1**：几何数值证据强，但H2没有处理实测存在的逐range非线性修正；
尚不能建立非Refh bin处的修正、有效时间模型或误差界，offset物理语义和RW高度frame仍未充分约束。
这一判断不单纯因为缺官方确认（官方确认另属Level 3）；问题是当前affine任意bin模型的重要物理假设仍未闭合。
不得只因漂亮closure或达到诊断阈值提升到Level 2/3/4。强经验支持与官方确认在证据矩阵分开。
现有3DEP两份LAZ已读取header（external_reference_inventory.json）：horizontal NAD83(2011)/UTM18N；
未在header建立vertical datum、coordinate epoch、准确采集时间和共位置表面。文件creation date是派生文件生成日期，不能替代采集日期。
因此没有直接比较CASALS椭球高与未知3DEP高程；external absolute accuracy为NOT VERIFIED。
H5列出L1A/ARD/GPS/geolocation输入名称，但当前目录没有对应完整中间产品；TDMS并非同一L1B定位处理器的可直接替代真值。

必须向CASALS确认：RW代表sample edges还是centers；offset单位／clock／bounce定义；sample time zero／间隔；
bin_size是否明确为height increment；RW/instrument/refh reference frame、epoch和vertical datum；
beam angle方向和local origin；range_bias/neutat/geoid/tides/dac是否applied；
不同range位置的校正是否非线性，以及H2是否与官方processor逐bin等价。

## bin_to_xyz函数完整契约与production建议

输入：start_xyz/stop_xyz为broadcastable (...,3) ECEF meter、有限且非零segment；
bins为broadcastable实数数组，0≤b≤N−1；n_bins是integer≥2；convention H1/H2/H3（默认H2）。
输出：同broadcast形状的ECEF XYZ meter。无效／域外／unknown convention抛ValueError，不extrapolate／clip。
fractional直接affine；输入应是该pulse的正确window和N，不内置pulse顺序假设。
本函数不执行peak detection、不重施任何correction、不认证sample是物理return。
**建议暂不纳入production**：先取得正式字段／processor确认，并处理异常与非参考range校正模型。
本轮只新增研究和测试，production geolocation、detection/classification/Refh及原始H5不变。

## 验收与执行命令

A（数据真实性）全量两文件、bounded reads和before/after hashes已记录；B（字段契约）未知明确保留；
C/D（几何／mapping）line vs segment、raw argmax和三个index候选均实算；
E（timing／beam）数值检查完成但official semantics NOT VERIFIED；
F（nonrefh）integer/fractional及真实raw示例完成，external accuracy NOT VERIFIED；
G（软件）pytest／compileall结果见validation.json。研究执行完成不代表科学的所有证据门槛PASS。

```powershell
conda activate map
python research/geolocation/bin_georeferencing_audit/run_audit.py --mode smoke --output-root outputs/research/bin_georeferencing_audit/smoke_new
python research/geolocation/bin_georeferencing_audit/run_audit.py --mode full --output-root outputs/research/bin_georeferencing_audit/full_new
python -m research.geolocation.bin_georeferencing_audit.complete_report --output-root outputs/research/bin_georeferencing_audit/full_new
python -m pytest -q
python -m compileall -q research/geolocation/bin_georeferencing_audit tests
```

analysis.ipynb的可见AUDIT配置须指向新运行目录；执行notebook只读取本轮已验证输出。
源文档：仓库docs/casals_l1b_pulse_waveform_geolocation_notes.md、两份archived geolocation notebooks和历史beam audit script仅作待复核资料；
官方教程检索日期2026-10-09，直接URL打开出现404，但网页检索索引可读取内容。不能宣称获取了官方完整processor说明。
'''
    (ROOT/'research/geolocation/bin_georeferencing_audit/findings.md').write_text(content,encoding='utf-8')
    print('findings.md generated',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    dirs=sorted(p for p in args.output_root.iterdir() if (p/'run_metadata.json').exists())
    if len(dirs)!=2:raise ValueError('NOT VERIFIED: two full original-H5 audits required')
    for d in dirs:
        print('Supplement:',d.name,flush=True);supplement(d)
    report(args.output_root)


if __name__=='__main__':main()
