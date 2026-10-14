# CASALS 3D Sweep Animator and Waveform Cross-Section Viewer

**CASALS 3D Sweep Animator** 是一个面向机载激光雷达 CASALS L1B 数据的独立、交互式 PyQt5/OpenGL 科学可视化与研究诊断软件。

该软件基于单个真实 CASALS L1B HDF5 文件，动态展示机载激光雷达的观测过程，实现了三维测量几何（飞机姿态示意、真实航迹、观测光束与逐步累积的 Refh 测量条带）与二维垂直剖面（RX 波形物理高程映射热力图、标准化响应热力图、官方 Refh 剖面）的完全时间轴同步。

---

## 1. 软件核心功能

1. **三维观测几何展示**：
   - 绘制基于真实 `instrument_longitude / latitude / altitude` 的机载轨迹（已过轨迹、当前位置标记）；
   - 展示程序生成的示意三维飞机模型（包含机身、机头、机翼、水平尾翼和垂尾）；
   - 飞机航向由相邻 sweep 的水平航迹自动推算；
   - 绘制当前 sweep 代表性观测光束（连接每个脉冲各自的 instrument position 与官方 Refh 点）；
   - 随着 sweep 推进，历史 Refh 点云持续累积，逐渐形成整条测量条带。

2. **二维垂直剖面同步视图**：
   - **Mode A: Raw / Background-subtracted RX heatmap**：支持原始幅度、扣除背景均值（Net RX）以及 Asinh 尺度显示变换；
   - **Mode B: Standardized RX response**：基于脉冲背景标准差的归一化超额响应 $Z = (\text{RX} - \text{bg\_mean}) / \text{bg\_std}$；
   - **Mode C: Refh spatial section**：直接绘制官方 Refh 沿剖面的高程与分布；
   - **Refh & Argmax Overlays**：在二维剖面上清晰叠加当前 sweep 官方 Refh 高程，并支持叠加根据 H1/H2 重建的波形最大峰高程进行几何闭合比对。

3. **完整交互与播放控制**：
   - 播放、暂停、步进、步退、滑块自由拖动跳转、播放速度（1, 2, 5, 10, 20, 30 FPS）与步长调整；
   - 三维视角支持自由旋转、平移、缩放、重置视角，以及跟随飞机模式（Follow Aircraft）；
   - 支持三维垂直显示缩放（1.0, 0.1, 0.05, 0.025），且垂直缩放严格限制于三维视觉显示，不影响二维剖面的真实物理高程。

4. **高性能按需读取与异步架构**：
   - 文件打开时仅加载元数据、sweep/track 索引网格和轻量 1D 几何字段，整个 14 GB 的波形数据集绝不一次性载入内存；
   - 历史点云采用确定性降采样策略（显示预算约 200,000–250,000 点），通过前缀偏移量切片实现 $O(1)$ 更新，时间轴任意跳转瞬间响应；
   - 独立的 `H5Worker` 后台线程管理私有 HDF5 文件句柄，支持最新请求优先机制（pending-latest），快速拖动滑块不阻塞 UI。

5. **导出能力**：
   - 一键导出当前 3D + 2D 联合视图为高清 PNG 截图（带参数与遥测横幅）；
   - 支持指定 sweep 范围、步长和帧率，导出 MP4 测量过程动画，并自动生成同名 JSON 元数据 sidecar。

---

## 2. 科学与几何定义

### 2.1 原始 Refh 与研究用 Waveform-Bin 映射的界限

- **官方 Refh**：CASALS L1B 每个脉冲提供一条接收波形 `rx_waveform` 和一个官方地理定位参考点 `(refh_longitude, refh_latitude, refh)`。`refh` 对应接收波形的最大幅值 bin（argmax），在 WGS84 椭球高约定下给出。**不能认为它必然是地面回波**（可能来自植被冠层、建筑物或水面）。
- **研究用 Waveform-Bin 映射**：L1B 文件并未提供官方的多回波或全波形离散点云三维坐标。本软件中将非最大幅值 RX bin 映射到三维空间或二维剖面，属于**研究诊断性几何重建**，供科学分析与可视化使用，绝非经过外部第三方真值独立检校的官方多回波产品。

### 2.2 局部 ENU 坐标系与 WGS84 椭球高

- 源坐标定义：
  - 经纬度：WGS84 地理坐标系（EPSG:4979）
  - 高程：WGS84 椭球高（Ellipsoidal height, meters）
  - 地心空间直角坐标：EPSG:4978 ECEF
- **固定 ENU 原点**：全文件打开后，自动选取全文件所有有效 Refh 点的中位数经度、纬度与椭球高作为**唯一固定原点**。
  - X 轴：Local East（正东向，米）
  - Y 轴：Local North（正北向，米）
  - Z 轴：Local Up（局部天向，米）
- 原点在整场回放和跳转中固定不变。3D 场景中的坐标基于 ENU，2D 剖面纵轴则标定为真实的 WGS84 绝对椭球高（米）。

### 2.3 飞机姿态与航向说明

- 飞机的位置基于当前 sweep 各脉冲记录的 `instrument_longitude`, `instrument_latitude`, `instrument_altitude`；
- 三维飞机模型为程序生成的专业双发科研飞机示意模型；
- 飞机航向（Heading）由轨迹在局部水平投影（East, North）的中心差分自动推算，水平保持姿态，不虚构未经 GNSS/INS 记录的机身 roll/pitch；
- 状态信息明确提示：`Aircraft orientation: schematic, heading derived from trajectory`。

### 2.4 H1 与 H2 距离窗 Bin 映射几何

本软件提供两种 range-window 空间映射模型：

- **H2 模型（推荐，内部一致性最佳）**：
  $$P(b) = P_0 + \frac{b + 0.5}{N} (P_1 - P_0)$$
  其中 $P_0$ 为 `rwstart` 的 ECEF 坐标，$P_1$ 为 `rwstop` 的 ECEF 坐标，$N$ 为 RX bin 总数（通常 2,728），$b \in [0, N-1]$ 为 0-based bin 索引。该模型将 range window 端点视作采样窗口边缘，各 bin 中心位于 $(b + 0.5)/N$。在真实数据验证中，H2 对应官方 Refh 的三维闭合差中位数小于 1 毫米（约 0.5 mm）。
- **H1 模型（对比模型）**：
  $$P(b) = P_0 + \frac{b}{N - 1} (P_1 - P_0)$$
  该模型将端点视作首末采样中心。在真实数据中，H1 与官方 Refh 存在约 10–25 毫米的残差偏差。

> **注意**：倾斜激光束中，不同 bin 在空间上具有不同的水平位置与高程。软件对每个脉冲沿倾斜直线计算三维位置，绝不采用粗暴固定水平经纬度只改变高度的简化做法。

### 2.5 横向剖面轴（Cross-track Axis）定义

提供两种横轴显示模式：

1. **Cross-track distance (meters)**：
   - 提取当前 sweep 有效 Refh 点的局部 ENU 水平位置；
   - 通过 PCA 拟合主横向方向向量 $\vec{v}$，根据 `track_num` 的增序确定正方向；
   - 以当前 sweep 中心为原点，将每个波形 bin 的三维空间位置投影到该横向轴上；
   - 若横向跨度小于 5 米或与 track 相关性过低（退化情况），软件自动回退到 Track Index 模式并提示原因。
2. **Track Index**：直接以 track 编号（0 到 255）作为横轴。

---

## 3. 安装与运行方法

### 3.1 运行环境

软件基于 Python 3.10+，推荐使用本仓库现有的 `[gui]` 依赖环境：

- `PyQt5`
- `pyqtgraph` (含 `pyqtgraph.opengl`)
- `PyOpenGL`
- `numpy`
- `h5py`
- `matplotlib`
- `opencv-python` (用于视频编码)

### 3.2 启动程序

从仓库根目录直接运行启动命令：

```bash
# 启动空窗口并通过界面打开文件
python -m debug.casals_sweep_animator_qt

# 或者直接在命令行指定要打开的 CASALS L1B H5 文件
python -m debug.casals_sweep_animator_qt "data/raw/casals_l1b/casals_l1b_20241112T165718_001_02.h5"
```

---

## 4. 界面使用与参数说明

### 4.1 控制面板（左侧）

- **File**：
  - `Open CASALS H5...`：选择并打开本地 L1B 文件，后台线程异步解析；
  - 显示总脉冲数、有效 sweep 数、track 数与当前状态。
- **Playback & Timeline**：
  - `⏮`：重置到首个 sweep；
  - `◀ / ▶`：向前/向后单帧步进；
  - `▶ Play / ⏸ Pause`：启动或暂停自动播放；
  - `Timeline Slider`：快速拖动时间轴，支持任意跳转；
  - `Sweep` 选框：直接输入有效 sweep ID 跳转；
  - `FPS`：演示播放速度（1, 2, 5, 10, 20, 30）；
  - `Step`：步长（每帧跳过若干 sweep，用于快进）。
- **3D Scene Options**：
  - `Aircraft / Trajectory / History Refh / Current Refh`：独立图层显示开关；
  - `Beams`：光束密度选择（`None`, `16 tracks`, `8 tracks`, `All tracks`）；
  - `Z-Scale`：垂直显示缩放（`1.0`, `0.1`, `0.05`, `0.025`），初始推荐 0.05；
  - `Camera Mode`：`Overview`（固定全局鸟瞰）或 `Follow Aircraft`（视角跟随飞机前进）；
  - `Reset View`：恢复默认斜俯视角。
- **2D Cross-Section Options**：
  - `Mode`：
    - `Minus bg_mean (Net RX)`：扣除背景的净接收响应（推荐）；
    - `Raw RX amplitude`：原始 ADU 幅值；
    - `Asinh-scaled Net RX`：对数类显示变换，适合观察微弱次回波；
    - `Standardized RX (σ)`：标准化超额响应；
    - `Refh Spatial Section`：仅观察官方 Refh 点的二维高程分布。
  - `X-Axis`：`Meters`（物理横向距离）或 `Track Index`；
  - `Map`：选择 `H2` 或 `H1` 波形距离窗映射公式；
  - `Color`：选择色彩映射表（`turbo`, `viridis`, `inferno`, `gray`）；
  - `Range`：`Fixed Range`（基于全文件采样统计的稳定色标，防闪烁）或 `Auto Frame`；
  - `Refh Overlay / Argmax Overlay`：官方参考点（白圈连线）与重建峰值（橙色三角）的叠显开关。
- **Export**：
  - `Save Screenshot (PNG)...`：导出当前帧的高清排版图片；
  - `Export MP4 Video...`：打开视频导出对话框，设置起止 sweep、步长与帧率，批量渲染保存 MP4 动画及同名 `.json` 元数据。

---

## 5. 性能与内存指标

在 14.04 GB 完整 CASALS L1B 数据文件（14,080 sweeps，3,604,480 pulses，2,728 RX bins）上的实际测定性能：

| 性能指标 | 实测数值 | 说明 |
| :--- | :--- | :--- |
| **初始化时间** | ~4.7 秒 | 完成全文件索引、有效 sweep 检测、固定 ENU 原点计算与历史点云降采样 |
| **进程常驻内存 (RSS)** | ~868 MB | 内存增长受控，绝不一次性缓存几 GB 的全量波形 |
| **历史点云显示规模** | 225,280 点 | 严格控制在预设的 200k–250k 预算内，且保持因果时间顺序 |
| **单 Sweep 波形按需读取** | ~340–370 毫秒 | 仅从磁盘读取当前 sweep 对应的 256 行波形 |
| **2D 网格化生成耗时** | ~39–46 毫秒 | 70 万个数据点瞬时完成空间投影与规则网格统计 |
| **单帧总准备耗时** | ~380–415 毫秒 | 后台线程执行，鼠标与主界面 100% 流畅响应 |
| **最大连续播放吞吐** | ~2.5 FPS | 播放定时器按请求帧率运行，处理不及时时自动丢弃旧任务，无队列积压 |

---

## 6. 当前限制与说明

1. **不可做地面网格化假设**：Refh 点是最大幅值回波参考点，可能位于森林冠层内部、地表、建筑或水面，软件坚持将其显示为离散观测点云，不得连接成伪地面 TIN 网格；
2. **H2 映射的科学定位**：H2 映射在数值上实现了微米/毫米级的官方 Refh 闭合一致性，但其本质是探测器时间窗边界的几何连续假设，尚未经独立地面实测点验证，不可作为经检校的官方级绝对坐标；
3. **视频导出的渲染捕获**：MP4 导出采用主界面各组件的高清离屏同步渲染，在导出大范围帧数前请确认起止范围，避免耗时过长。

