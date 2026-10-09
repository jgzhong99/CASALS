# CASALS L1B 任意 RX bin georeferencing：严格验证报告

日期：2026-10-09。结论：**Level 1 — Internally consistent**。推荐研究用 H2：

`P(b) = P_RWSTART + ((b + 0.5) / N) * (P_RWSTOP - P_RWSTART)`。

两份完整原始 H5 重新遍历；没有使用历史 CSV/Notebook 结果替代本轮数据。
该判断不认证任意次峰的官方级或绝对三维精度，也不合并到 production。
Level 2 数值目标的部分几何项满足状态：`True`；字段语义、修正应用及高质量子集的证据门槛仍须逐项看下文。
H2 是当前两个产品的经验上最可靠样本中心映射；不能把 RWSTART/RWSTOP 说成首末样本中心。
H1也可能通过给定的宽松closure／half-bin阈值；阈值不是模型唯一性证明。
选择H2依据连续的bin-center时间关系和亚毫米残差解释，而非为H1或H2调整验收条件。

## 数据与复现证据

|     date |   pulses |   RX_bins |   valid |   high_quality |   outside_segment |   H2_gt_10cm |
|---------:|---------:|----------:|--------:|---------------:|------------------:|-------------:|
| 20241112 |  3604480 |      2728 | 3604480 |             16 |                 0 |            0 |
| 20241118 |  3604480 |      2728 | 3604480 |         178079 |                 0 |            0 |

- `casals_l1b_20241112T165718_001_02.h5`: `637ba73ca9966b86d9243987916c0e60b8fec76c8690d49b7048468ed308cb70`; before/after SHA256 identical.
- `casals_l1b_20241118T171757_001_02.h5`: `dac3d5ed0116083f98944e8d6b0e86f6d812f0f5e4f196cc010add5e382c149a`; before/after SHA256 identical.

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

| metric                            | stat   |          20241112 |          20241118 |
|:----------------------------------|:-------|------------------:|------------------:|
| refh_line_distance_m              | median |    4.41273723e-06 |    4.38646231e-06 |
| refh_line_distance_m              | P95    |    1.10612866e-05 |    1.06818305e-05 |
| refh_line_distance_m              | max    |    1.14534634e-05 |    1.10972438e-05 |
| refh_segment_distance_m           | median |    4.41273723e-06 |    4.38646231e-06 |
| refh_segment_distance_m           | P95    |    1.10612866e-05 |    1.06818305e-05 |
| refh_segment_distance_m           | max    |    1.14534634e-05 |    1.10972438e-05 |
| H1_3d_m                           | median |    0.0163167764   |    0.0154884148   |
| H1_3d_m                           | P95    |    0.0687403545   |    0.0283593388   |
| H1_3d_m                           | max    |    0.0749279418   |    0.0749278182   |
| H2_3d_m                           | median |    0.000526645195 |    0.000529168867 |
| H2_3d_m                           | P95    |    0.00055075425  |    0.000543255296 |
| H2_3d_m                           | max    |    0.000555187167 |    0.000555210761 |
| H3_3d_m                           | median |   91.5785795      |   87.3818573      |
| H3_3d_m                           | P95    |  374.855521       |  156.927244       |
| H3_3d_m                           | max    |  408.728985       |  408.729082       |
| bin_delta_H1                      | median |   -0.093759945    |   -0.102215888    |
| bin_delta_H1                      | P95    |    0.419539779    |   -0.0500412125   |
| bin_delta_H1                      | max    |    0.499820454    |    0.499819421    |
| bin_delta_H2                      | median |    0.00351429739  |    0.00353104538  |
| bin_delta_H2                      | P95    |    0.00367513032  |    0.00362509582  |
| bin_delta_H2                      | max    |    0.00370407001  |    0.0037048113   |
| edge_spacing_m_per_bin            | median |    0.149854754    |    0.149854747    |
| edge_spacing_m_per_bin            | P95    |    0.149854778    |    0.149854811    |
| edge_spacing_m_per_bin            | max    |    0.14985481     |    0.149854903    |
| segment_spacing_m_per_bin         | median |    0.149909707    |    0.149909699    |
| segment_spacing_m_per_bin         | P95    |    0.14990973     |    0.149909763    |
| segment_spacing_m_per_bin         | max    |    0.149909763    |    0.149909855    |
| stored_bin_size                   | median |    0.149623549    |    0.149722498    |
| stored_bin_size                   | P95    |    0.149718638    |    0.149738916    |
| stored_bin_size                   | max    |    0.149725631    |    0.149743129    |
| stored_minus_height_step          | median |    0              |    0              |
| stored_minus_height_step          | P95    |    0              |    0              |
| stored_minus_height_step          | max    |    0              |    0              |
| t_time_minus_geom                 | median |   -1.28823218e-06 |   -1.29437147e-06 |
| t_time_minus_geom                 | P95    |   -2.15521744e-07 |   -1.1261083e-06  |
| t_time_minus_geom                 | max    |    1.17631049e-09 |    4.85856104e-10 |
| time_bin_delta_H2                 | median |    1.98951966e-12 |    9.09494702e-13 |
| time_bin_delta_H2                 | P95    |    1.04591891e-11 |    7.27595761e-12 |
| time_bin_delta_H2                 | max    |    1.86446414e-11 |    1.18234311e-11 |
| bounce_step_per_bin               | median |    5e-10          |    5e-10          |
| bounce_step_per_bin               | P95    |    5e-10          |    5e-10          |
| bounce_step_per_bin               | max    |    5e-10          |    5e-10          |
| RW_vs_beam_refh_deg               | median |    0.000133355519 |    0.000108665412 |
| RW_vs_beam_refh_deg               | P95    |    0.000212080328 |    0.000681132783 |
| RW_vs_beam_refh_deg               | max    |    0.000338077941 |    0.00113309767  |
| RW_vs_beam_instrument_deg         | median |    0.00416810767  |    0.00197220458  |
| RW_vs_beam_instrument_deg         | P95    |    0.00582069248  |    0.0148988698   |
| RW_vs_beam_instrument_deg         | max    |    0.0067734716   |    0.015531542    |
| RW_vs_IR_deg                      | median |    4.79393728e-06 |    3.41066434e-05 |
| RW_vs_IR_deg                      | P95    |    7.20054851e-06 |    0.000227826592 |
| RW_vs_IR_deg                      | max    |    9.42069909e-06 |    0.000246698325 |
| conditional_c_half_minus_length_m | median | -204.345314       | -204.345293       |
| conditional_c_half_minus_length_m | P95    | -204.345243       | -204.345216       |
| conditional_c_half_minus_length_m | max    | -204.345182       | -204.345013       |

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

| metric                            | stat   |     20241112 |     20241118 |
|:----------------------------------|:-------|-------------:|-------------:|
| segment_length_m                  | median |  408.80377   |  408.803749  |
| bounce_span                       | median |    1.364e-06 |    1.364e-06 |
| conditional_span_times_c_m        | median |  408.916913  |  408.916913  |
| conditional_span_times_c_half_m   | median |  204.458456  |  204.458456  |
| conditional_c_half_minus_length_m | median | -204.345314  | -204.345293  |

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

|     date | x          | y                       |    pearson_r |
|---------:|:-----------|:------------------------|-------------:|
| 20241112 | H2_along_m | t_times_one_minus_t     | -0.999988317 |
| 20241112 | H2_along_m | neutat_ref_minus_linear |  0.999998692 |
| 20241118 | H2_along_m | t_times_one_minus_t     | -0.993332876 |
| 20241118 | H2_along_m | neutat_ref_minus_linear |  0.999978493 |

|     date |         median |            P95 |            max |
|---------:|---------------:|---------------:|---------------:|
| 20241112 | 9.61088841e-08 | 5.62631958e-07 | 9.98901491e-07 |
| 20241118 | 2.10009912e-07 | 8.87995455e-07 | 1.84756508e-06 |

这一近似等量的数值证据支持“不同range处大气修正非线性影响endpoint interpolation residual”的解释，
比单独相关性更强，但修正字段单位／applied状态和官方处理过程仍未知，不能据此私自修正任意bin或宣称处理器parity。
H2 affine给出endpoint模型坐标，未捕捉逐range非线性；当前对Refh实测的亚毫米差不可自动外推为所有bin的误差上界。
原始max规则与同值／近值计数保留，未固定任何bin改善closure。

历史案例在两份新读原始数据中的实际值：

|     date |   pulse_index |   sweep_num |   track_num |   argmax_bin |   amp_1693 |   amp_1694 |   max_amplitude |   tie_count |
|---------:|--------------:|------------:|------------:|-------------:|-----------:|-----------:|----------------:|------------:|
| 20241112 |       1280218 |        5000 |         214 |          346 |        -42 |         -5 |             451 |           1 |
| 20241118 |       1280218 |        5000 |         214 |         1694 |       1850 |       1860 |            1860 |           1 |

int16存储值的numpy argmax在float64转换后完全精确；当前记录不能据此证明浮点环境导致1693/1694逆转。
若旧实验使用smoothing、fitting、不同产品版本、不同pulse索引或其他waveform表示，它的近并列结论不能直接转移到当前raw H5。
归档v2中beam winner在Refh basis、radian、north-clockwise、negated elevation方向的假设可复核；
归档的b/(N−1)不能作为“唯一正确约定”延续，当前增加H2后具有更强内部支持。

最大H2残差具体记录（不从总体删去）：

|     date |   pulse_index |   sweep_num |   track_num |        H2_3d_m |   bin_delta_H2 |   refh_line_distance_m |
|---------:|--------------:|------------:|------------:|---------------:|---------------:|-----------------------:|
| 20241112 |        563521 |        2201 |          10 | 0.000555187167 |  0.00370407001 |         1.12695827e-05 |
| 20241112 |        614944 |        2402 |           1 | 0.000555177403 |  0.00370402563 |         1.11432444e-05 |
| 20241112 |        610466 |        2384 |          21 | 0.000555173018 |  0.0037039906  |         1.11786181e-05 |
| 20241118 |        158829 |         620 |         107 | 0.000555210761 |  0.0037048113  |         5.51666213e-06 |
| 20241118 |        177005 |         691 |         107 | 0.000555206107 |  0.00370478761 |         5.41543937e-06 |
| 20241118 |        182167 |         711 |         188 | 0.000555174817 |  0.00370456772 |         5.54935961e-06 |

每份anomalous_pulses.csv同时包括line距离、beam角、同track的segment长度／方向跳变前十条。
异常的threshold计数、geometry/RX rejection、out-of-bounds和instrument order见run_metadata.json；
同track连续性表不把不同track的交错record当作同一扫描线跳变。
高质量阈值SNR≥10、unique maximum、top-two差>1 ADU，未依赖闭合误差调参；为空时不补造样本。

|     date | subset       | model   |   population_total |   n_used |   screening_rejected |   within_half_bin |   within_one_bin |    signed_bias |
|---------:|:-------------|:--------|-------------------:|---------:|---------------------:|------------------:|-----------------:|---------------:|
| 20241112 | all          | H1      |            3604480 |  3604480 |                    0 |                 1 |                1 | -0.093759945   |
| 20241112 | all          | H2      |            3604480 |  3604480 |                    0 |                 1 |                1 |  0.00351429739 |
| 20241112 | high_quality | H1      |            3604480 |       16 |              3604464 |                 1 |                1 | -0.102956471   |
| 20241112 | high_quality | H2      |            3604480 |       16 |              3604464 |                 1 |                1 |  0.00353309347 |
| 20241118 | all          | H1      |            3604480 |  3604480 |                    0 |                 1 |                1 | -0.102215888   |
| 20241118 | all          | H2      |            3604480 |  3604480 |                    0 |                 1 |                1 |  0.00353104538 |
| 20241118 | high_quality | H1      |            3604480 |   178079 |              3426401 |                 1 |                1 | -0.111864754   |
| 20241118 | high_quality | H2      |            3604480 |   178079 |              3426401 |                 1 |                1 |  0.00349987478 |

另提供不要求SNR阈值的明确argmax子集unambiguous_argmax_agreement.csv，保留unfiltered总体。

## 11：真实非Refh示例

|     date |   pulse_index |   sweep_num |   track_num | category   |   bin |   raw_argmax |   longitude |   latitude |   ellipsoidal_height |   H5_minus_H2_m |
|---------:|--------------:|------------:|------------:|:-----------|------:|-------------:|------------:|-----------:|---------------------:|----------------:|
| 20241112 |        102901 |         401 |         175 | multiple   |  2588 |         1560 | -75.280252  | 38.4041177 |          -166.668649 |  0.000735515267 |
| 20241118 |         11085 |          43 |         106 | low_snr    |    37 |         1363 | -76.6772818 | 36.1923731 |           207.154766 |  0.000639512357 |
| 20241118 |         11085 |          43 |         106 | low_snr    |    72 |         1363 | -76.6772843 | 36.1923732 |           201.914519 |  0.000635466253 |
| 20241118 |         11085 |          43 |         106 | low_snr    |    90 |         1363 | -76.6772856 | 36.1923733 |           199.219535 |  0.000633416658 |
| 20241118 |         11085 |          43 |         106 | low_snr    |   111 |         1363 | -76.677287  | 36.1923734 |           196.075387 |  0.000631053135 |
| 20241118 |         11085 |          43 |         106 | low_snr    |   139 |         1363 | -76.677289  | 36.1923734 |           191.883189 |  0.000627948284 |

代表pulse具有真实raw waveform局部峰，图示waveform和对应相对Refh ECEF3D。
这些未确认峰来自冠层、地面或建筑；raw局部峰和sample bin不能直接称独立物理scatterer。
宽峰／低SNR／边界／近并列等类别只按raw形态和stored SNR标注；缺失类别明确记录，不fabricate。
- casals_l1b_20241112T165718_001_02: missing fields []; invalid/overlapping-reason counts in `run_metadata.json`; reviewed raw morphology missing categories []. 初轮sample_selection.json中的类别因互斥优先标签可能掩盖形态，补充reviewed_raw_examples.csv采用并行标签，单独记录更强对比选择规则。
- casals_l1b_20241118T171757_001_02: missing fields []; invalid/overlapping-reason counts in `run_metadata.json`; reviewed raw morphology missing categories []. 初轮sample_selection.json中的类别因互斥优先标签可能掩盖形态，补充reviewed_raw_examples.csv采用并行标签，单独记录更强对比选择规则。

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
