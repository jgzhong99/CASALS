# CASALS 仓库重构报告

## 基线与工作分支

- 初始 `HEAD`：`2f38f9a712328d7bde095fecd83efa8b49e73925`，对应已同步的 `master`。
- 工作分支：`refactor/reorganize-casals`。
- 重构前 Git 树：129 个跟踪文件，其中 34 个 Python 文件、20 个 Notebook、4 个演示文稿和 30 个 GUI `.pyc` 文件。基线数据清单与旧分类指标见 [refactor_baseline.md](refactor_baseline.md)。
- 未提交任何 commit 或 push。

## 最终目录职责

```text
casals_l1b/       共用 H5、坐标、波形、噪声、栅格和分类逻辑
scripts/          正式数据处理入口与独立工具
experiments/      分类和地理定位研究代码
notebooks/        按 introduction、waveform、geolocation、refh 分类的研究记录
casals_gui.py     Qt 查看器启动器
casals_gui_app/   Qt/TDMS GUI 实现
data/raw/         本地 CASALS H5 与 TDMS 原始输入
data/reference/   本地 3DEP 参考片段
data/derived/     可明确追溯来源后复用的派生数据位置
outputs/          新工作流结果及独立保存的重构前基线
docs/             数据布局、工作流、实验、基线、报告和资料索引
tests/            共用科学 helper 的小型合成测试
```

## 代码整理与删除

- 提取共享模块：`casals_l1b.h5`、`geo`、`waveform`、`noise`、`raster` 和 `classification`。正式脚本和研究脚本使用这些公共实现；分类核心不再影子复制 CRS/H5 搜索函数。
- 保留确实有不同 fallback 和错误语义的 3DEP H5/CRS 查找逻辑；没有为了统一而改变其处理策略。
- 正式脚本移至 `scripts/`；分类、地理定位实验移至 `experiments/`。`pyproject.toml` 提供核心依赖、`dev`/`notebooks`/`gui`/`visualization` extras 和 9 个 console entrypoint。
- `--sweep-end` 的帮助文字明确为 inclusive，和原有 `np.arange(start, end + 1)` 及基线选择一致；没有改变选择范围算法。
- 删除的无效或冗余内容：旧波形入口 `CASALS_L1B/detect_waveform_components.py`（仅兼容转发）、未使用的 `casals_gui_app/main_window.py` 别名、失效的 `run_casals_refh_filter_gui.py`、重复且过时的 `requirements-qt-pyvista.txt`、有自动 `git add/commit/push` 副作用的 `backup.bat`，以及 30 个已跟踪 `.pyc` 缓存文件。波形模块中经全仓搜索确认无人调用的旧别名也已移除。
- 旧 GUI 设置移到忽略的 `config/local/casals_gui_settings.json`；仓库提供不含机器路径的 `config/casals_gui_settings.example.json`。
- 实际运行发现 `diagnose_3dep_offsets.py` 使用 Pandas `to_markdown()`，但环境依赖未声明 `tabulate`。已将 `tabulate>=0.9` 加入核心依赖并重跑诊断成功。

## Notebook 迁移

20 个跟踪 Notebook 按主题移动或重命名；**没有合并 Notebook**。对每个旧/新 Notebook 的代码单元逐项比较，执行计数和保存的输出完全相同（20/20）。

| 原路径（均位于 `CASALS_L1B/notebooks/`） | 新路径 |
| --- | --- |
| `01_l1b_structure_and_refh.ipynb` | `notebooks/introduction/l1b_structure_and_refh.ipynb` |
| `casals_l1b_h5_schema_georef_audit.ipynb` | `notebooks/introduction/h5_schema_and_georeferencing_audit.ipynb` |
| `casals_l1b_single_h5_beam_reader.ipynb` | `notebooks/introduction/single_h5_beam_reader.ipynb` |
| `03_l1b_beam_geometry_and_bin_georeference_theory.ipynb` | `notebooks/geolocation/beam_geometry_and_bin_georeference_theory.ipynb` |
| `casals_l1b_beam_range_peak_geolocation_validation.ipynb` | `notebooks/geolocation/beam_range_peak_geolocation_validation.ipynb` |
| `casals_l1b_geolocation_rule_audit_revised.ipynb` | `notebooks/geolocation/geolocation_rule_audit_revised.ipynb` |
| `casals_l1b_geolocation_rule_audit_revised_v2.ipynb` | `notebooks/geolocation/geolocation_rule_audit_revised_v2.ipynb` |
| `03b_l1b_refh_forward_reconstruction_validation.ipynb` | `notebooks/geolocation/refh_forward_reconstruction_validation.ipynb` |
| `casals_l1b_sweep_peak_geolocation_rigorous_validation.ipynb` | `notebooks/geolocation/sweep_peak_geolocation_validation.ipynb` |
| `03_3dep_offset_diagnosis.ipynb` | `notebooks/refh/3dep_offset_diagnosis.ipynb` |
| `04_3dep_pseudolabel_transfer_qc.ipynb` | `notebooks/refh/3dep_pseudolabel_transfer_qc.ipynb` |
| `05_labeled_casals_pointcloud_by_class_reader.ipynb` | `notebooks/refh/labeled_pointcloud_by_class_reader.ipynb` |
| `02_refh_quality_and_dsm.ipynb` | `notebooks/refh/refh_quality_and_dsm.ipynb` |
| `05b_labeled_casals_pointcloud_waveform_peak_ratio_diagnostics.ipynb` | `notebooks/waveform/labeled_pointcloud_peak_ratio_diagnostics.ipynb` |
| `06_l1b_raw_waveform_noise_peak_diagnostics.ipynb` | `notebooks/waveform/raw_waveform_noise_peak_diagnostics.ipynb` |
| `01c_l1b_single_sweep_raw_waveform_viewer.ipynb` | `notebooks/waveform/single_sweep_raw_waveform_viewer.ipynb` |
| `01b_l1b_sweep_tx_rx_matrix.ipynb` | `notebooks/waveform/sweep_tx_rx_matrix.ipynb` |
| `01b_l1b_sweep_waveform_matrix.ipynb` | `notebooks/waveform/sweep_waveform_matrix.ipynb` |
| `02_l1b_waveform_feature_extraction.ipynb` | `notebooks/waveform/waveform_feature_extraction.ipynb` |
| `03_l1b_waveform_features_vs_refh_quality.ipynb` | `notebooks/waveform/waveform_features_vs_refh_quality.ipynb` |

另外保留了独立的 `notebooks/geolocation/range_window_bin_mapping_hypothesis.ipynb`；它与旧 geolocation 审计方法不同，单独保存。所有当前 Notebook 均通过 JSON 解析和代码单元语法检查（22 个文件、213 个代码单元、0 个语法错误）。

## 数据与资料处置

- 两个原始 CASALS H5 移到 `data/raw/casals_l1b/`，共 `28,122,139,604` 字节；两个 TDMS 文件、索引和 viewer 附件移到 `data/raw/tdms/`，基线清单共 9 个文件、`37,904,264,158` 字节。
- 两个外部 3DEP 片段移到 `data/reference/3dep/`：`48,263,708` 与 `595,959,424` 字节。原始 H5、TDMS 和 3DEP 文件通过移动归位，没有改写文件内容。
- 旧 `CASALS_L1B/outputs`、根级输出、点云和 Notebook 结果移到 `outputs/baseline_pre_refactor/`，用于对照。旧分类、geolocation、动画、滤波和其他实验记录都保留。
- 演示文稿移至 `docs/presentations/`；来源说明、PDF、图和网址材料移至 `docs/references/`。根目录的本地 `Archive/` 保留。
- 跟踪源码和资料没有因环境限制而跳过迁移。未跟踪的 `.compile_tmp/` 保持原样，没有纳入项目树；其中仍有一个 5,745,400 字节 HTML 和一个 622 字节 `.pyc`。`Archive/read_tdms.ipynb` 也仍是本地归档文件。

## 测试和运行结果

测试环境为项目 `.venv`（Python 3.14.6；NumPy 2.5.3、SciPy 1.18.1）。结果如下：

- `python -m pytest -q`：8 passed。
- `compileall`：`casals_l1b/`、`scripts/`、`experiments/`、GUI、启动器和测试文件全部通过。
- `git diff --check HEAD`：通过；Git 只提示工作树 LF/CRLF 规范化信息，没有空白错误。
- 20 个正式脚本/实验模块的 `--help` 均成功；9 个安装后的 console entrypoint 的 `--help` 均成功。
- 共用模块导入成功。
- Qt 启动器在 `QT_QPA_PLATFORM=offscreen` 下创建窗口并运行事件循环，退出码 0。
- 真实 TDMS 浏览未完成。首个 20.3 GB TDMS 文件带 Windows `RecallOnDataAccess` 属性，GUI 载入无可见进度且可能触发按需召回整份文件；为避免隐式读取/下载整个大文件而中止。GUI 启动已验证，不能据此声称实际 TDMS 数据浏览通过。
- 3DEP 下载器依赖 PDAL；当前未安装 PDAL，因此没有联网下载新片段。已用现存 3DEP clip 验证诊断和标签转移。

### 真实数据重放和基线比较

| 工作流 | 输入与结果 | 与归档基线比较 |
| --- | --- | --- |
| `export_refh_las` | 2024-11-12 H5；3,604,480 个 refh 点 | 点记录全部维度、顺序、头部 CRS/scale/offset 和文件大小均相同。 |
| `filter_refh_points` | 2024-11-12；SNR 筛选后 42,740 点，noise 0 点 | raw、noise-labeled、clean LAS 的所有维度逐点一致；计数与 reason code 计数相同。 |
| `extract_refh_ground` | 2024-11-12；41,075 个 ground 候选、1,656 个非 ground 候选、9 个低异常点 | 两个 LAS 的所有维度相同；4 张 GeoTIFF 的像素、shape、transform、CRS 和 nodata 全部相同（487×118）。 |
| `make_refh_dsm` | 2024-11-18；1,952,951 个选中点，EPSG:32618，栅格 730×1282 | 6 张 GeoTIFF 像素与空间元数据完全相同；严格有效格 41,503、支持格 46,470、IDW 填格 4,967；选中点 LAS 逐点相同。 |
| `transfer_3dep_labels_to_casals` | 2024-11-12 H5 + 9,081,045 点 3DEP clip | 输出类别及转移状态计数与旧 LAZ 完全一致：class 1/2/7 分别 395,125/1,213,267/1,996,088；strict/weak/ambiguous/far 分别 1,461,182/28,773/118,437/1,996,088。连续维度仅 `x_original_m`、`y_original_m` 和 `nearest3dep_dist_m` 有末位差异，最大分别为 `5.82e-11 m`、`2.79e-9 m`、`5.96e-8 m`；`dz=37.6581362552 m` 与类别保持一致。 |
| `diagnose_3dep_offsets` | 2024-11-12 H5 + 上述较小 clip；42,740 个高 SNR 点，28,039 个匹配地面样本 | 完整生成 36 行敏感性结果。流程没有验证垂直基准转换，所以 `final_reference_frame_accuracy_summary` 正确记录为 `not_computed`。旧诊断记录对应 2024-11-18，不是同一输入，不作数值等价声明。 |
| `classify_and_evaluate_refh` | 两个 H5 与保留的旧转移 LAZ；另用新 2024-11-12 转移 LAZ 做端到端复核 | 分类标签、point index、reason code、分类摘要和评价表完全相同。分类 HAG 辅助维度有不超过 `5.09e-9 m` 的差异；2024-11-18 不超过 `5.39e-12 m`。新转移 LAZ 上 2024-11-12 的评价行也与基线完全相同。指标基于 3DEP 转移伪参考，不是独立精度认证。 |
| `extract_waveform_features` | 2024-11-18 H5，sweep 5000–5002，768 个 pulse、3,064 个组件 | 检测核心函数与旧源码 AST 完全一致，但一处主峰不同：sweep 5000 / track 214 从旧结果 bin 1693 变为新结果 bin 1694，导致 raw amplitude 差 10。其余 prominence 的最大差 `1.82e-12`；该 pulse 的附加高度变化约 `0.1498 m`。当前平滑值在两个相邻 bin 仅差约 `2.3e-13`，显示这是近似并列峰；基线没有记录 NumPy/SciPy 版本，因此无法确定确切触发因素。没有改检测算法来掩盖差异。 |

分类基线主要指标仍是伪参考评价结果：2024-11-12 高置信子集 accuracy / macro-F1 / weighted-F1 为 `0.894912 / 0.519858 / 0.880366`；2024-11-18 为 `0.919614 / 0.565101 / 0.912380`。两个文件的基线评价表逐项相等。

## 输出保留情况与复现命令

本次没有物理删除旧 outputs。旧输出都已隔离到 `outputs/baseline_pre_refactor/`；重放结果写入被 Git 忽略的 `outputs/refactor_validation/`。这样既能核对已有结果，也不会把历史研究证据误作新结果。新默认运行仍写入 `outputs/<workflow>/`。

从仓库根目录运行正式工作流的示例：

```powershell
python -m scripts.export_refh_las --h5 data/raw/casals_l1b/<granule>.h5
python -m scripts.filter_refh_points --h5 data/raw/casals_l1b/<granule>.h5
python -m scripts.extract_waveform_features --h5 data/raw/casals_l1b/<granule>.h5
python -m scripts.make_refh_dsm --h5 data/raw/casals_l1b/<granule>.h5
python -m scripts.extract_refh_ground --h5 data/raw/casals_l1b/<granule>.h5
python -m scripts.transfer_3dep_labels_to_casals --casals-h5 <granule.h5> --dep3-las <reference.laz>
python -m scripts.classify_and_evaluate_refh --h5 <granule.h5> --reference <transferred.laz>
```

完整工作流参数和科学解释见 [workflow_reference.md](workflow_reference.md)；用户输入与可再生输出的边界见 [data_layout.md](data_layout.md)。

## 尚存限制

1. 波形主峰有一处 1-bin 差异；缺少旧环境版本信息，根因无法从现有记录确认。
2. 117,530,910 点的 2024-11-18 3DEP clip 没有在当前仅约 4.4 GB 可用内存下重跑转移；分类重放使用已保留的旧转移文件。小 clip 的 2024-11-12 转移和诊断已实际运行。
3. TDMS 文件按需召回，未完成真实 GUI 数据浏览；PDAL 未安装，未运行在线 clip 下载。
4. 基线文件记录了配置和结果，但未记录波形运行的 NumPy/SciPy 版本。旧诊断元数据指向 `anaconda3/envs/py11`，该环境在当前机器不存在；近似并列峰的精确旧环境复现因此受限。
