# CASALS L1B 高程验证与地物波形响应报告

日期：2026-10-12  
数据范围：2024-11-12 与 2024-11-18 两个 CASALS L1B granule，以及现有的两个 3DEP LAZ 和现有 NAIP 影像。  
状态：3DEP 绝对高程验证 **BLOCKED**；地物波形响应 **DIAGNOSTIC_ONLY**。

> Updated 2026-10-12: Sections 1-8 retain the earlier audit snapshot. Section 9 reports the revised Notebook 10/11 execution and supersedes earlier statements that no raw or empirical elevation comparison was performed. Strict absolute accuracy remains unavailable.

## 1. 研究问题与结论

本次工作分别审计 CASALS Refh 与 3DEP 的高程参考框架，并比较历史 NAIP 图像解释 ROI 内的 CASALS 接收幅度、SNR 与波形形态。

- 目前**不能**对 CASALS 做可信的绝对高程验证：CASALS 坐标参考框架和 Refh 高程语义未证实；3DEP LAZ 缺少垂直 CRS 与历史 Z 处理记录；所需的非 ballpark 三维坐标操作及网格不可用。因此没有计算 CASALS–3DEP 高差、偏差、RMSE、NMAD、置信区间或相对高程误差。
- 地物响应结果保留了两个 granule 中 11 个有脉冲中心的 ROI 的全部 **128,427** 条波形。现有 H5 的 `good_snr` 标志在两个完整 granule 的 **7,208,960** 条记录上都严格等于 `refh_snr >= 5`。这只是数据字段与阈值的关系，不是独立验证的回波检出率。
- 对全 2,728 个 RX bin 取最大值的 3σ 筛查在本次 128,427 条 ROI 波形中全部通过，因而不能区分可检出回波和噪声极值。下文将其仅作为描述性筛查，不称为检出结果。

## 2. Study A：CASALS–3DEP 参考框架审计

| Granule | CASALS H5 证据 | 3DEP LAZ 头信息 | 非 ballpark 三维操作 | 状态 |
|---|---|---|---:|---|
| 2024-11-12 | 3,604,480 脉冲；有 GPS/UTC epoch 属性，但没有可确认的水平 datum realization、坐标历元或 Refh 垂直参考语义 | 水平 CRS 为 EPSG:6347（NAD83(2011) / UTM 18N）；头信息没有垂直轴。Z scale 为 0.01 m，offset 为 118.855 m；这不证明 Z datum 或历史变换 | 0 个可用；14 个候选操作因网格缺失而不可用 | **BLOCKED** |
| 2024-11-18 | 3,604,480 脉冲；同样缺少完整的 CASALS 水平/垂直框架证据 | 水平 CRS 为 EPSG:6347；头信息没有垂直轴。Z scale 为 0.01 m，offset 为 687.165 m；这不证明 Z datum 或历史变换 | 0 个可用；13 个候选操作因网格缺失而不可用 | **BLOCKED** |

H5 中的 epoch 属性包括 `casals_epoch_gps_sec=1198800018.0`、GPS 周/秒以及 `casals_epoch_utc_a=2018-01-01T00:00:00Z`。这些字段没有说明经纬度采用的 datum realization，也不能据此认定坐标历元或 `refh` 是椭球高还是正高。相关 H5 数据集没有提供足以补齐这些语义的 CRS 元数据。

两份 LAZ 均未找到关联 sidecar。当前文件名和 LAZ 头只能确认水平 CRS 与存储 Z 的 scale/offset，不能确认源 3DEP 高程基准、裁剪前后 Z 是否经过改正，或历史裁剪所用的完整 source/work-unit。LAZ 的 Z 范围是头信息范围，不是筛选后的地面高程范围。

Notebook 10 在显式的候选源假设下检查到目标复合 CRS `EPSG:6347+5703`（NAD83(2011) 平面坐标加 NAVD88 米制高程），但将 `always_xy=True`、`allow_ballpark=False`，并使用每个 AOI。两个 AOI 都没有可用的非 ballpark transformer，且没有应用任何转换。候选操作缺少的网格包含 `us_noaa_g2018u0.tif`、`us_noaa_nadcon5_nad83_2007_nad83_2011_conus.tif`、geoid 网格及 Maryland/North Carolina 区域网格；每个 AOI 的完整清单、缺失操作和来源记录见 [reference_frame_audit.json](../outputs/notebooks/10_3dep_elevation_validation/reference_frame_audit.json)。

每个 granule 的 blocker 为：CASALS 水平框架未证实、CASALS 垂直框架未证实、3DEP 垂直框架未证实、历史 Z 处理未证实、原始来源/work-unit provenance 不完整，以及所需三维转换不可用。由于严格门槛未通过，本次没有重建 3DEP ground surface、生成地面样本或支持域、筛选 stable-ground 样本，亦没有计算绝对或相对高程误差。**没有用经验 dz 平移来打开门槛。**

解锁此项验证需要：CASALS 处理器版本及 Refh 水平/垂直定义和改正记录；3DEP 原始 tile/work-unit、垂直 CRS、裁剪历史及 Z 转换记录；AOI 所需的完整 PROJ/NADCON/Geoid 网格。获得并核验这些证据后，才能按独立地面条件重建参考面并处理 3DEP 与 2024 CASALS 之间的时间差。

## 3. Study B：ROI 和样本定义

Notebook 11 使用本机已有的 NAIP 2022-06-15 影像，以图像内容手工勾画开放地表、道路、林冠与水体候选 ROI。影像比 CASALS 采集早约 2.4 年；标签是**概率性历史图像解释**，不是 2024 现场标签。ROI 成员资格仅按 H5 Refh 经度/纬度的名义中心投影到 EPSG:32618 判断，没有影像配准、实测 footprint 或独立位置校正。没有用 HAG、波形、幅度、SNR 或分类结果来画标签。

清单共 12 个 ROI：11 个 ROI 有名义 H5 中心并纳入全波形处理，共 128,427 个脉冲中心；另一个 2024-11-18 水塘候选 ROI 为 0 个中心，保留在清单并标为排除。2024-11-18 有三个林冠 ROI，但其中两个属于同一南部林地，因此只有两个林地 site；不把 ROI 内大量脉冲当成独立重复。

## 4. Study B：字段定义与全量质量审计

- `refh_amp`：H5 中与官方 Refh 位置关联的幅度字段，按 counts 报告；不是经辐射定标的反射率。
- `refh_snr`：H5 Refh SNR 字段；另报告固定 `>=1.5`、`>=1.8`、`>=3` 和 `>=5` 阈值比例。
- `good_snr`：原样保留的 H5 布尔字段。分块核验完整 H5 后，两份 granule 均有 0 个缺失 SNR、0 个与 `refh_snr >= 5` 不符的记录。见 [h5_good_snr_consistency_audit.csv](../outputs/notebooks/11_landcover_lidar_response/multi_granule_full_roi/h5_good_snr_consistency_audit.csv)。因此在当前文件中它不是独立于 5 阈值 SNR 的验证标签。
- 波形最大值筛查：`max((RX - bg_mean) / bg_std)`，跨全部 2,728 bins；3/4/5/6/8σ 阈值只作敏感性计数。3σ 规则不能称为官方回波检出规则。
- 宽度：最大正峰的半高连续宽度，限于最高峰所在的 3σ 连通支撑区域，单位 bins。
- 主支撑能量：只对最高峰所在的 3σ 连通分量求和 `max(RX-bg_mean, 0)`，单位 counts·bin；其他分离分量保存在逐波形记录中。该量不是辐射能量或反射率。

所有 128,427 条选入 ROI 的波形记录和 background 数值均有效，均保留在分母中；没有发生波形或 background 无效记录。所有波形均通过 3σ 全窗最大值筛查，所以该规则没有提供无检出样本。`good_snr` / SNR 阈值计数仍按完整 ROI 分母报告。逐波形特征、原始样本清单、排除记录、每 ROI/类别/track 汇总均在 [Notebook 11 输出目录](../outputs/notebooks/11_landcover_lidar_response/multi_granule_full_roi/)。

## 5. Study B：ROI 级描述性结果

下表的 `n` 是 ROI 内全部名义 H5 脉冲中心数；`good_snr` 率在当前 H5 中等于 `refh_snr >= 5` 率。能量列是主支撑相对计数-bin 指标。

| 日期 | ROI | n | `good_snr` | Refh SNR 中位数 | Refh 幅度中位数 (counts) | 宽度中位数 (bins) | 主支撑能量中位数 (counts·bin) |
|---|---|---:|---:|---:|---:|---:|---:|
| 11-12 | open_field_12_01 | 17,974 | 1.68% | 2.38 | 656 | 7 | 4,571 |
| 11-12 | open_field_12_02 | 17,171 | 2.78% | 2.26 | 665 | 6 | 4,129 |
| 11-12 | road_core_12_01 | 3,820 | 4.35% | 2.25 | 588 | 6 | 3,472 |
| 11-12 | tree_canopy_12_01 | 32,014 | 0.19% | 1.91 | 495 | 6 | 2,473 |
| 11-12 | water_pond_12_01 | 10,160 | 0.24% | 1.95 | 479 | 6 | 2,523 |
| 11-18 | open_field_01 | 7,140 | 87.18% | 7.12 | 1,799 | 8 | 15,036 |
| 11-18 | open_field_02 | 8,924 | 74.13% | 6.07 | 1,992 | 7 | 15,327 |
| 11-18 | road_core_01 | 1,040 | 70.87% | 6.34 | 1,690 | 7 | 13,427 |
| 11-18 | tree_canopy_01 | 3,503 | 14.05% | 3.49 | 864 | 11 | 11,045 |
| 11-18 | tree_canopy_02 | 4,329 | 10.28% | 3.25 | 864 | 9 | 9,115 |
| 11-18 | tree_canopy_18_03 | 22,352 | 42.14% | 4.48 | 1,467 | 8 | 14,218 |

2024-11-12 的两个开放地表 ROI 合计 35,145 脉冲，`good_snr` 为 780/35,145（2.22%）；该日期的道路为 166/3,820（4.35%）、单个林地 ROI 为 62/32,014（0.19%）、单个水体 ROI 为 24/10,160（0.24%）。2024-11-18 的两个开放地表 ROI 合计 12,840/16,064（79.93%），道路为 737/1,040（70.87%），三个林冠 ROI 合计 10,357/30,184（34.31%）。林冠 ROI 之间的比例从 10.28% 到 42.14%，显示 site 差异不能由 pooled pulse 数消除。

两日期的 H5 `good_snr` 全 granule 比例分别为 42,740/3,604,480（1.19%）与 1,752,317/3,604,480（48.62%），并且各自都严格等于 `refh_snr >= 5`。日期、granule、track 覆盖和测量条件明显不同；跨日期的幅度/SNR 差异不能归因于地物。

## 6. Track 敏感性与解释边界

在 2024-11-12 的纳入 ROI 之间没有共同 track ID。2024-11-18 有 23 个共同 track（56–78）；匹配子集每个 ROI 每个共同 track 保留一个样本。该小型 same-track 对照中，两个南部林冠 ROI 的 `good_snr` 比例分别为 17.4% 和 8.7%，西北独立林地 ROI 为 91.3%，而两个开放地表 ROI 为 87.0%。结果说明林地 site 选择和轨迹覆盖会改变汇总，不能支持普遍地物效应。track 编号不用于推断 wavelength 或 beam identity。

NAIP 标签的时间差、中心位置与真实光斑 footprint 不确定、少量独立 site、脉冲空间相关、两日期的观测条件和 track 覆盖，以及缺少 radiometric calibration，都是主要限制。现有结果适合检查相对接收信号特征和提出后续采样问题；不支持校准反射率、地物因果效应、分类精度、physical return 数量或 canopy height 结论。没有进行 Gaussian decomposition。

## 7. 交付文件与图件

- [Notebook 10：3DEP 高程参考门槛](../notebooks/10_3dep_elevation_validation.ipynb)，审计结果：[reference_frame_audit.json](../outputs/notebooks/10_3dep_elevation_validation/reference_frame_audit.json)。图件 `reference_frame_gate.png` 只显示 BLOCKED 门槛，不伪造残差图。
- [Notebook 11：地物波形响应](../notebooks/11_landcover_lidar_response.ipynb)，全部表格和图件位于 `outputs/notebooks/11_landcover_lidar_response/multi_granule_full_roi/`；ROI 清单为 [roi_manifest.geojson](../data/reference/landcover_rois/roi_manifest.geojson)。
- 09-B2 保持为一张大图：180-sweep `RdBu_r` signed heatmap，以 0 为中心：[track_sweep_context_180.png](../outputs/notebooks/09_nadir_window_reflection/method_improvement/casals_l1b_20241118T171757_001_02/track_sweep_context_180.png)。
- 研究 helper 与小型测试：[research/l1b_validation.py](../research/l1b_validation.py)、[tests/test_l1b_validation.py](../tests/test_l1b_validation.py)。

## 8. 验证记录

Notebook 10 和 11 均在 `casals-showcase` 干净内核下从仓库根目录独立执行成功；两个 notebook 都没有保存的执行错误。全套 pytest 为 **93 passed**；`compileall` 和 `git diff --check` 均通过。原始两份 H5 与两份 3DEP LAZ 的 SHA-256 均与任务前保存的基线一致。测试中的 WGS84 经纬高到地心 XYZ 转换及逆转换只验证合成坐标的代码路径，不构成当前 CASALS–3DEP 框架证据。

## 9. Revised Notebook 10 and 11 results (2026-10-12)

This section records the fresh-kernel execution of Notebooks 10 and 11. New outputs are isolated under `outputs/notebooks/10_3dep_elevation_validation/revised/` and `outputs/notebooks/11_landcover_lidar_response/revised/`. Earlier output folders were preserved.

### Study A: actual CASALS-3DEP height comparison

Both 3DEP LAZ files were read and their ASPRS class-2 points were streamed into the local ground surfaces. November 12 contained 4,646,895 class-2 points in a 9,081,045-point LAZ; November 18 contained 63,226,508 class-2 points in a 117,530,910-point LAZ. Supported, locally acceptable surfaces matched 3,369,562 of 3,604,480 H5 pulses (93.5%) on November 12 and 3,457,681 (95.9%) on November 18.

The archived sidecars identify the November 12 source as workunit `MD_Southeast_1_2019` / project `MD_Southeast_2019_D20`, and November 18 as `NC_HurricaneFlorence_9_2020` / `NC_HurricaneFlorence_2020_D20`. They record horizontal output EPSG:6347, `vert_crs=5703`, GEOID18 context, and `vertical_datum_transform_applied=false`. The sidecar H5 basenames and clip point counts match the current inputs and LAZ headers. Reprojecting the recorded EPSG:26918 clip boxes to EPSG:6347 reproduces the LAZ header XY bounds within 0.605 m and 0.200 m, respectively. This confirms horizontal file/footprint linkage only; it does not prove point-level Z preservation. The LAZ headers have no vertical CRS axis, and the H5 metadata still does not establish the Refh vertical frame or coordinate epoch.

Raw differences retain all matched observations and are defined as CASALS H5 Refh minus the local 3DEP class-2 plane height. Vertical reference mismatch may remain.

| Granule | Matched / valid H5 pulses | Coverage | Mean (m) | Median (m) | SD (m) | RMSE (m) | MAE (m) | NMAD (m) | P05 to P95 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2024-11-12 | 3,369,562 / 3,604,480 | 93.5% | -12.164 | -37.571 | 92.949 | 93.741 | 75.181 | 43.237 | -166.311 to 173.579 |
| 2024-11-18 | 3,457,681 / 3,604,480 | 95.9% | -31.297 | -37.962 | 32.713 | 45.273 | 38.923 | 0.227 | -38.240 to -15.186 |

The large raw tails are preserved in these metrics; no evaluation point was selected or removed by its residual. The empirical offsets were recalculated from the current, predeclared calibration sites. Each calibration site is spatially separate from its test site. Values below describe holdout agreement after subtracting the calibration median; they are not absolute-accuracy estimates.

| Granule | SNR stratum | Calibration n | Test n / denominator | Offset (m) | Test bias (m) | RMSE (m) | MAE (m) | NMAD (m) | Abs. P95 (m) | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024-11-12 | All valid | 17,974 | 14,885 / 17,171 | -37.663 | 24.208 | 84.332 | 45.213 | 0.295 | 209.175 | 86.7% |
| 2024-11-12 | SNR >= 3 | 5,203 | 4,274 / 4,852 | -37.659 | 0.271 | 8.522 | 0.606 | 0.117 | 0.278 | 88.1% |
| 2024-11-12 | SNR >= 5 | 302 | 445 / 478 | -37.659 | 0.086 | 0.132 | 0.112 | 0.088 | 0.230 | 93.1% |
| 2024-11-18 | All valid | 7,140 | 8,924 / 8,924 | -37.980 | 1.547 | 19.123 | 2.227 | 0.095 | 0.243 | 100.0% |
| 2024-11-18 | SNR >= 3 | 7,015 | 8,623 / 8,623 | -37.981 | -0.048 | 0.469 | 0.087 | 0.092 | 0.212 | 100.0% |
| 2024-11-18 | SNR >= 5 | 6,225 | 6,615 / 6,615 | -37.979 | -0.037 | 0.097 | 0.076 | 0.087 | 0.190 | 100.0% |

The SNR strata expose substantial residual tails, especially in the November 12 data: its all-valid and SNR >= 3 RMSE values are much larger than the corresponding NMAD and central percentile range. The November 18 all-valid RMSE is also tail-sensitive. The stricter SNR >= 5 subsets are smaller and show tighter empirical holdout agreement. Across-track raw-difference summaries are provided as a diagnostic; tracks are nested observations and do not establish independent accuracy.

`reference_frame_status=UNRESOLVED`, `raw_height_comparison_status=COMPLETED`, `empirical_alignment_status=COMPLETED`, and `strict_absolute_accuracy_status=UNAVAILABLE` for both granules. No empirical offset is presented as a datum transformation, and no metric in these tables is certified absolute elevation accuracy.

Key Notebook 10 outputs: [November 12 raw difference distribution](../outputs/notebooks/10_3dep_elevation_validation/revised/casals_l1b_20241112T165718_001_02/raw_difference_distribution.png), [November 12 holdout map](../outputs/notebooks/10_3dep_elevation_validation/revised/casals_l1b_20241112T165718_001_02/empirical_holdout_spatial_map.png), [November 18 holdout map](../outputs/notebooks/10_3dep_elevation_validation/revised/casals_l1b_20241118T171757_001_02/empirical_holdout_spatial_map.png), and [November 12 audit JSON](../outputs/notebooks/10_3dep_elevation_validation/revised/casals_l1b_20241112T165718_001_02/reference_frame_audit.json).

### Study B: land-cover waveform response

The image dates were re-read from the NAIP metadata: 2023-06-02 for the November 12 granule and 2022-06-15 for November 18. The imagery predates CASALS acquisition by 529 and 887 days, respectively. It remains historical image context, not a contemporaneous field label. Of 12 manifest ROIs, 11 contained H5 pulse centers and 9 were primary ROI polygons spanning 8 independent sites; two November 12 ROIs remain exploratory, and the zero-pulse November 18 water ROI remains excluded.

All 128,427 ROI pulse waveforms were valid and retained in the denominators. The provisional candidate screen uses a full-record maximum threshold estimated from 6,144 real standardized background-tail windows per granule: 10.885 sigma on November 12 and 10.195 sigma on November 18. Synthetic-null false-alarm rates were 0.0058 and 0.0029. These are noise-only controls under the tail-window bootstrap assumptions; `detection_probability_calibrated=false` and candidate rates are not official return-detection probabilities.

| Granule / ROI | Pulse n | Candidate rate | Median Refh SNR | Median Refh amplitude (counts) |
|---|---:|---:|---:|---:|
| 2024-11-12 open field 01 | 17,974 | 29.1% | 2.38 | 656 |
| 2024-11-12 open field 02 | 17,171 | 25.5% | 2.26 | 665 |
| 2024-11-12 forest site | 32,014 | 4.0% | 1.91 | 495 |
| 2024-11-12 exploratory road | 3,820 | 26.9% | 2.25 | 588 |
| 2024-11-12 exploratory water | 10,160 | 5.6% | 1.95 | 479 |
| 2024-11-18 open field 01 | 7,140 | 98.2% | 7.12 | 1,799 |
| 2024-11-18 open field 02 | 8,924 | 97.1% | 6.07 | 1,992 |
| 2024-11-18 road | 1,040 | 91.8% | 6.34 | 1,690 |
| 2024-11-18 canopy polygon 01 | 3,503 | 73.8% | 3.49 | 864 |
| 2024-11-18 canopy polygon 02 | 4,329 | 65.2% | 3.25 | 864 |
| 2024-11-18 separate woodlot site | 22,352 | 75.9% | 4.48 | 1,467 |

The adjacent November 18 canopy polygons count as one site (2 ROIs, 7,832 pulses), not two independent sites. H5 `good_snr` equals `refh_snr >= 5` over all 7,208,960 granule records; this is field consistency, not independent return validation. `refh_amp` remains native digital counts and is not reflectance.

The conservative intersection across every primary ROI contains no track in November 12 and tracks 56-78 (23 tracks) in November 18. Pairwise intersections remain in `common_track_coverage.csv`; the primary matched-track figure and the new all-primary tables use the 23-track intersection. Within that shared-track subset, open-field 01 minus canopy polygon 01/02 had median SNR differences of +5.259/+5.384 and median amplitude differences of +1,354/+1,375 counts. These directional within-granule contrasts are descriptive: tracks are nested, the two canopy polygons are one site, and the small number of sites cannot support a general land-cover effect or significance claim. The much higher candidate rates, SNR, and amplitude in November 18 than November 12 also leave acquisition and instrument conditions confounded with class.

Key Notebook 11 outputs: [ROI audit CSV](../outputs/notebooks/11_landcover_lidar_response/revised/roi_audit.csv), [ROI response summary](../outputs/notebooks/11_landcover_lidar_response/revised/roi_response_summary.csv), [all-primary shared tracks](../outputs/notebooks/11_landcover_lidar_response/revised/all_primary_roi_common_tracks.csv), [matched-track figure](../outputs/notebooks/11_landcover_lidar_response/revised/matched_track_landcover_comparison.png), [November 18 NAIP ROI board](../outputs/notebooks/11_landcover_lidar_response/revised/naip_roi_board_casals_l1b_20241118T171757_001_02.png), and [summary JSON](../outputs/notebooks/11_landcover_lidar_response/revised/landcover_response_summary.json).

### Execution and interpretation status

Both revised notebooks completed from a clean `casals-showcase` kernel without cell errors. Notebook 10 now performs actual raw and spatially held-out empirical comparisons for both granules; Notebook 11 performs the noise-control, ROI, site, and matched-track summaries described above. The strict vertical-frame audit remains unresolved, and the land-cover analysis remains descriptive. `python -m pytest -q -p no:cacheprovider` passed with 109 tests; `python -m compileall -q casals_l1b research tests` passed; and `git diff --check` passed (Git emitted only an LF-to-CRLF advisory for `research/l1b_validation.py`). Both notebooks were freshly executed after the final figure edits with no cell errors. This work did not modify raw H5/LAZ inputs or production algorithms.
