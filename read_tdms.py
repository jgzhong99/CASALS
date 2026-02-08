# %% [markdown]
# # CASALS TDMS (Waveform LiDAR) — notebook quickstart
# 
# 目标：让初学者在 10–15 分钟内建立对 **CASALS pushbroom waveform LiDAR** 与 **TDMS 文件组织方式** 的正确心智模型，并能用 Python：
# 
# 1) 读出 TDMS 的关键配置（32 steps/sweep、8 sweeps/row、256 footprints/row、rows/file）；  
# 2) 从 **一个 channel** 切出 **某一次推进（某一 row）** 的单条波形；  
# 3) 从 **一次推进（一个 row）** 组装出整条 swath 的 **256 条波形热力图**（并使用正确的跨轨排序：**step-major → sweep-minor**）。
# 
# ---
# 
# 核心心智模型（先记住这三句话）：
# 
# - **Pushbroom**：沿航向每推进一次，产生一个 **row（cross-track scan）**。  
# - **Row = full swath**：每个 row 同时包含 **256 个 footprint 波形**（256 个并行 tracks）。  
# - **TDMS 存储**：这 256 个 footprint 被存成 **256 个 channels（Sweep{j}Step{k}）**；每个 channel 是把所有 rows 的波形首尾相接成的 1D 长数组。
# 

# %%
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import plotly.graph_objects as go

from nptdms import TdmsFile

plt.rcParams["font.size"] = 16

# %%

tdms_path = "./tdms/casals_18-11-2024_13-37-23_164-002.tdms"

tdms_path = "./tdms/casals_18-11-2024_13-34-53_158-003.tdms"

# %%
c = 299_792_458.0
Fs = 1e9          # sampling rate (Hz)
dt = 1.0 / Fs     # seconds per sample
dr = 0.5 * c * dt # meters per sample  (~0.149896 m)

# %% [markdown]
# ## Import TDMS and inspect acquisition configuration
# 
# 输入：`tdms_path`（TDMS 文件路径）
# 
# 输出（你应该看到并理解的关键点）：
# - 组属性（group properties）给出 pushbroom 扫描结构：  
#   `#WVLSteps/Sweep=32`, `#Sweeps/ShiftCycle=8` → **256 footprints/row**  
#   `Sweeps/File=14080` → **14080 rows/file（推进次数）**
# - 通道命名：256 个 channel 形如 `Sweep{j}Step{k}`，它们对应 8 sweeps × 32 steps 的组合。  
# - 后续我们会基于这些属性，把 TDMS 恢复成一个逻辑上的三维/四维张量：  
#   `waveform[row, footprint, sample]`（或显式分解为 `waveform[row, step, sweep, sample]`）
# 

# %% [markdown]
# ## CASALS scan geometry (from the project docs)
# 
# 你需要把下面四个数字当作“系统级常识”，它们解释了为什么数据会长成 256 个 channels + pushbroom rows 的形态：
# 
# - Wavelength-tuning 在跨轨方向指向，覆盖约 **0.75°**（≈0.013 rad）全角。  
# - **One tuning sweep → 32 steps**：每个 sweep 目标 32 个等间距 footprint 位置。  
# - **8 sweeps fill in the swath → 256 footprints**：8 次 sweep 交错填充跨轨，形成 256 个重叠 footprint（256 parallel tracks）。  
# - 在 5 km 距离条件下：  
#   cross-track spacing ≈ **0.24 m**（≈FWHM），形成约 **65 m** swath；  
#   along-track spacing ≈ **0.17 m**（80 m/s 时），这就是 pushbroom 推进方向的采样间距。
# 
# > 这些数值的意义：CASALS 更像“每一行 256 像素、每像素是一条波形”的推扫系统，而不是传统 LAS/LAZ 点云输出。
# 

# %%

# ----------------------------
# 0) Open TDMS (streaming-friendly)
# ----------------------------
tdms_path = Path(tdms_path)  # reuse your tdms_path variable or set it to the .tdms file
tdms = TdmsFile.open(tdms_path)

print(f"File: {tdms_path}")

# ----------------------------
# 1) TDMS data model refresher (file → group → channel)
# ----------------------------
print("\n[TDMS hierarchy]")
print("  TDMS is organized as: File → Groups → Channels.")
print("  CASALS stores LiDAR measurements as waveform samples in channels; acquisition metadata are group properties.")

# ----------------------------
# 2) Inspect file / group properties (acquisition geometry is here)
# ----------------------------
print("\n[File properties]")
for k, v in tdms.properties.items():
    print(f"  {k}: {v}")

GROUP = "CASALS"
g = tdms[GROUP]

print(f"\n[Group: {GROUP}]")
print("  Group properties encode acquisition timing/scan configuration (e.g., steps per sweep, sweeps per cycle, window widths).")
for k, v in g.properties.items():
    print(f"  {k}: {v}")

# Pull key configuration (use fallbacks if a key is absent)
WVL_STEPS_PER_SWEEP = int(g.properties.get("# WVLSteps/Sweep", 32))
SWEEPS_PER_CYCLE    = int(g.properties.get("# Sweeps/ShiftCycle", 8))
ROWS_PER_FILE       = int(g.properties.get("Sweeps/File", 0))  # you observed 14080

print("\n[Derived swath layout]")
print(f"  WVL steps per sweep      = {WVL_STEPS_PER_SWEEP}  (discrete pointing positions within one sweep)")
print(f"  Sweeps per shift cycle   = {SWEEPS_PER_CYCLE}     (number of sweeps composing one cross-track row)")
print(f"  Footprints per row       = {WVL_STEPS_PER_SWEEP * SWEEPS_PER_CYCLE}  (= channels across swath; expected 256)")
print(f"  Rows per file            = {ROWS_PER_FILE}        (cross-track scans along-track; expected 14080 for this file)")

# ----------------------------
# 3) Enumerate channels and parse naming convention
# ----------------------------
chs = list(g.channels())
print("\n[Channels]")
print(f"  Total channels in group = {len(chs)}")
print("  Expected channels ≈ footprints per row (256). Your file shows 256 channels, consistent with 8×32.")

pat = re.compile(r"^Sweep(\d+)Step(\d+)$")
idx_map = {}  # (sweep, step) -> channel object
bad_names = []

for ch in chs:
    m = pat.match(ch.name)
    if not m:
        bad_names.append(ch.name)
        continue
    s, st = int(m.group(1)), int(m.group(2))
    idx_map[(s, st)] = ch

print("\n[Channel naming model]")
print("  CASALS channels are named Sweep{j}Step{k}.")
print("  Interpretation: each channel corresponds to ONE fixed cross-track footprint index (j,k) across the swath.")
print(f"  Parsed Sweep/Step channels = {len(idx_map)}")

if bad_names:
    print("  Note: some channels did not match Sweep{j}Step{k}:")
    for nm in bad_names[:10]:
        print(f"    - {nm}")
    if len(bad_names) > 10:
        print(f"    ... ({len(bad_names)-10} more)")


# %%
chk_group = "CASALS"
chk_channel = "Sweep0Step0"
chk_ch = tdms[chk_group][chk_channel]

# 常见的 waveform timing 键名（有些来自 LabVIEW 写 TDMS 的习惯）
cand_keys = [
    "wf_increment", "wf_offset", "wf_start_offset", "wf_start_time",
    "SamplingRate", "SampleRate", "dt", "TimeIncrement", "XIncrement"
]

print("[Channel properties check]")
print(f"  group/channel = {chk_group}/{chk_channel}")
chk_ch.properties

# %% [markdown]
# ## Infer waveform layout from TDMS channel length
# 
# CASALS 的 TDMS **不是点云**，而是波形阵列。你需要建立下面这个严格对应关系：
# 
# - **Swath 宽度（footprints/row）**：  
#   `footprints_per_row = (#WVLSteps/Sweep) × (#Sweeps/ShiftCycle) = 32 × 8 = 256`
# 
# - **Pushbroom 推进次数（rows/file）**：  
#   `rows = Sweeps/File`（你这份文件是 14080）
# 
# - **TDMS 的存储方式（关键）**：  
#   每个 channel（例如 `Sweep0Step0`）并不是“一个波形”，而是把所有 rows 的波形 **首尾相接** 存成一条 1D 数组：  
#   `channel_length = rows × samples_per_row`
# 
# 因此：
# 
# - `samples_per_row = channel_length / rows`  
#   你已经推导出 `samples_per_row = 2816`，这表示：**每个 footprint、每一次推进（row）都有一条长度为 2816 的回波波形**。
# 
# 再强调一次常见误区：
# 
# - `channel_length (39,649,280)` 是 **ADC 采样点总数**（一个 channel 的 1D 长度），不是“footprint 次数”。  
# - 真正的 “footprint 波形条数” 是：`rows × 256 = 14,080 × 256 = 3,604,480` 条波形（每条 2816 点）。
# 

# %%
EXAMPLE_CH = "Sweep0Step0"
ch0 = g[EXAMPLE_CH]

print(f"\n[Example channel: {EXAMPLE_CH}]")
print("  Meaning: ONE fixed footprint (one Sweep/Step) across the whole file, stored as a flattened 1D stream.")
print(f"  dtype = {ch0.data_type}")
print(f"  channel_length = {len(ch0):,} samples (flattened across all rows)")

if ROWS_PER_FILE <= 0:
    raise ValueError("Group property 'Sweeps/File' is missing or 0; cannot infer rows.")

if len(ch0) % ROWS_PER_FILE != 0:
    raise ValueError(f"Channel length {len(ch0)} is not divisible by rows={ROWS_PER_FILE}; reshape would be ambiguous.")

SAMPLES_PER_ROW = len(ch0) // ROWS_PER_FILE
FOOTPRINTS_PER_ROW = WVL_STEPS_PER_SWEEP * SWEEPS_PER_CYCLE
TOTAL_FOOTPRINT_WAVEFORMS = ROWS_PER_FILE * FOOTPRINTS_PER_ROW

print("\n[Derived tensor sizes]")
print(f"  rows_per_file        = {ROWS_PER_FILE:,}  (pushbroom advances / cross-track scans)")
print(f"  footprints_per_row   = {FOOTPRINTS_PER_ROW}  (= {WVL_STEPS_PER_SWEEP} steps × {SWEEPS_PER_CYCLE} sweeps)")
print(f"  samples_per_row      = {SAMPLES_PER_ROW:,}  (ADC samples per waveform)")
print(f"  total waveforms/file = rows × footprints = {TOTAL_FOOTPRINT_WAVEFORMS:,}")

print("\n[Reshape rule for ONE channel]")
print("  Each channel is stored as: [row0 waveform][row1 waveform]...[rowN waveform]")
print("  So: channel_length = rows_per_file × samples_per_row")
print(f"  Check: {len(ch0):,} = {ROWS_PER_FILE:,} × {SAMPLES_PER_ROW:,}")

# Optional: compare against tick windows (sanity/diagnostic only)
tx_tick = g.properties.get("TxWinWidth(tick)", None)
rx_tick = g.properties.get("RxWinWidth(tick)", None)
if tx_tick is not None and rx_tick is not None:
    tx_tick, rx_tick = int(tx_tick), int(rx_tick)
    print("\n[Window diagnostics (tick vs samples)]")
    print(f"  TxWinWidth(tick) = {tx_tick}, RxWinWidth(tick) = {rx_tick}, sum = {tx_tick+rx_tick} ticks")
    print("  samples_per_row / (Tx+Rx ticks) often reveals oversampling or padding/alignment.")
    print(f"  samples_per_row / ticks_sum ≈ {SAMPLES_PER_ROW/(tx_tick+rx_tick):.3f}")

print("\n[Memory note]")
print("  Reading ch[:] loads the entire channel into RAM (~39.6M int16 ≈ 75 MB for a single channel).")
print("  Prefer slicing ch[a:b] to read one row waveform or a small block of rows.")

# %%
# --- Example A: read ONE row waveform from ONE channel ---
demo_row_id = 100
demo_a = demo_row_id * SAMPLES_PER_ROW
demo_b = demo_a + SAMPLES_PER_ROW
demo_wave = np.asarray(ch0[demo_a:demo_b], dtype=np.int16)

print("\n[Example A: one row × one channel]")
print(f"  row_id      = {demo_row_id}")
print(f"  slice       = [{demo_a}:{demo_b}] in the flattened channel")
print(f"  wave.shape  = {demo_wave.shape} (one waveform)")
print(f"  min/max     = {demo_wave.min()} / {demo_wave.max()}  (signed ADC counts)")

# %%
# --- Example B: read a small block of rows from ONE channel ---
demo_row0, demo_n = 100, 10
demo_a = demo_row0 * SAMPLES_PER_ROW
demo_b = (demo_row0 + demo_n) * SAMPLES_PER_ROW
demo_block = np.asarray(ch0[demo_a:demo_b], dtype=np.int16).reshape(demo_n, SAMPLES_PER_ROW)

print("\n[Example B: (n_rows × samples) block from one channel]")
print(f"  rows [{demo_row0}..{demo_row0+demo_n-1}] → demo_block.shape = {demo_block.shape}")
print("  Interpretation: each row in demo_block is one waveform at a fixed footprint.")

# %% [markdown]
# ## Extract one pushbroom advance (one row) across the full swath (256 channels)
# 
# 一次“推进”（one pushbroom advance）在 CASALS 数据里对应一个 **row / cross-track scan**。该 row 的最自然可视化单位是：
# 
# - **256 个 footprint（256 条波形）**：整条 swath  
# - 每条波形长度：`samples_per_row = 2816`
# 
# 关键点：**跨轨物理排序（cross-track ordering）**
# 
# 文档描述的是：一个 tuning sweep 在 swath 上采 32 个等间距位置（32 steps），8 个 sweeps 交错填充形成 256 footprints。  
# 因此，要把 y 轴解释为“跨轨从一侧到另一侧”，更合理的排列通常是：
# 
# - **step-major → sweep-minor**：  
#   `Sweep0Step0, Sweep1Step0, …, Sweep7Step0, then Step1…`
# 
# 下面我们会同时生成两种排序（A/B），但把 **Ordering B（step-major）** 作为默认的 cross-track 显示方式。
# 

# %% [markdown]
# （提示）你之前观察到：Ordering B 会让“地面/强反射层”呈现更连续的斜线结构，而 Ordering A 会把这种结构打散成锯齿/蛇形。这正是跨轨排序正确与否最直观的判据。

# %%
PERCENTILE_CLIP = [0.5, 99.5]

# %%
import numpy as np

# --- choose which pushbroom advance (row) to visualize ---
vis_row_id = 100  # any integer in [0, ROWS_PER_FILE-1]

print("[Row selection]")
print(f"  vis_row_id = {vis_row_id}")
print("  Meaning: one pushbroom advance / one cross-track scan (row).")
print(f"  This row contains {FOOTPRINTS_PER_ROW} footprint waveforms (256 channels).")

# --- extract one row across ALL 256 channels into a canonical 3D cube ---
# raw cube shape: (sweep, step, samples)
vis_row_cube_sweep_step = np.empty((SWEEPS_PER_CYCLE, WVL_STEPS_PER_SWEEP, SAMPLES_PER_ROW), dtype=np.int16)

vis_a = vis_row_id * SAMPLES_PER_ROW
vis_b = vis_a + SAMPLES_PER_ROW

vis_missing = []
for s in range(SWEEPS_PER_CYCLE):
    for st in range(WVL_STEPS_PER_SWEEP):
        ch = idx_map.get((s, st), None)
        if ch is None:
            vis_missing.append((s, st))
            vis_row_cube_sweep_step[s, st, :] = 0
        else:
            vis_row_cube_sweep_step[s, st, :] = np.asarray(ch[vis_a:vis_b], dtype=np.int16)

print("\n[Extraction result]")
print(f"  vis_row_cube_sweep_step.shape = {vis_row_cube_sweep_step.shape}  (sweep, step, samples)")
if vis_missing:
    print(f"  WARNING: missing {len(vis_missing)} channels; first few: {vis_missing[:10]}")
else:
    print("  OK: all 256 channels found.")


# %%
# ---------------------------------------------------------
# Apply Tx/start-pulse cut (remove transmit-reference window)
# Basis: user guide describes a start-pulse window (~80 ns) and a 1 GHz digitizer (~1 ns/sample),
# so removing ~80 samples isolates the return window for range interpretation.
# ---------------------------------------------------------
TX_SAMPLES_CUT = [8, 68]

print("\n[Tx cut]")
print(f"  Removing first {TX_SAMPLES_CUT} samples as Tx/start-pulse reference segment (~80 ns @ 1 GHz).")
print("  Remaining samples are treated as the return window (Rx) for subsequent plots/analysis.")

vis_row_cube_sweep_step_rx = vis_row_cube_sweep_step[:, :, TX_SAMPLES_CUT[0]:-TX_SAMPLES_CUT[1]]  # (sweep, step, samples_rx)
SAMPLES_PER_ROW_RX = vis_row_cube_sweep_step_rx.shape[-1]

print(f"  vis_row_cube_sweep_step_rx.shape = {vis_row_cube_sweep_step_rx.shape}  (sweep, step, samples_rx)")
print(f"  samples_rx = {SAMPLES_PER_ROW_RX} (from original {SAMPLES_PER_ROW})")

# --- build two 2D layouts for plotting (after Tx cut) ---
# Ordering A: sweep-major → step-minor  (Sweep0Step0..Sweep0Step31, Sweep1Step0..)
vis_swath_A = vis_row_cube_sweep_step_rx.reshape(SWEEPS_PER_CYCLE * WVL_STEPS_PER_SWEEP, SAMPLES_PER_ROW_RX)

# Ordering B: step-major → sweep-minor  (Sweep0Step0, Sweep1Step0, ... Sweep7Step0, then Step1 ...)
vis_swath_B = vis_row_cube_sweep_step_rx.transpose(1, 0, 2).reshape(WVL_STEPS_PER_SWEEP * SWEEPS_PER_CYCLE, SAMPLES_PER_ROW_RX)

print("\n[Ordering summary (after Tx cut)]")
print(f"  vis_swath_A.shape = {vis_swath_A.shape}  (index=0..255, sweep-major, Rx-only)")
print(f"  vis_swath_B.shape = {vis_swath_B.shape}  (index=0..255, step-major, Rx-only)  <-- recommended")

# Choose B as the default for subsequent plots/analysis (keep name vis_swath_row as requested)
vis_swath_row = vis_swath_B
print("\n[Default]")
print("  Using vis_swath_row = Ordering B (step-major → sweep-minor), with Tx cut applied (Rx-only).")
print(f"  min/max in this row = {vis_swath_row.min()} / {vis_swath_row.max()}  (signed ADC counts)")

# %% [markdown]
# ### Heatmap of one row (256 footprints × waveform samples) — **step-major cross-track ordering**
# 
# How to read this figure:
# 
# - **Y-axis (0–255)**: cross-track footprint index in **step-major → sweep-minor** order.  
#   Every **8 indices** correspond to one **Step** (because there are 8 sweeps per step).  
#   So boundaries at `y = 8, 16, 24, ...` separate `Step0, Step1, ..., Step31`.
# 
# - **X-axis (0–2815)**: waveform sample index inside the acquisition window (`samples_per_row = 2816`).  
#   (At this stage it is sample index, not meters; time/range calibration can be added later.)
# 
# - **Color**: raw ADC amplitude (signed int16 counts).  
#   Negative values are normal for raw digitizer counts (baseline/offset); they are not “negative photons”.
# 
# Why this heatmap matters:
# 
# - This is the most direct “single-row view” for CASALS pushbroom waveform LiDAR: **one row = full swath (256 waveforms)**.  
# - Continuous surfaces (e.g., ground) often appear as laterally coherent features (lines/bands) across the footprint index.
# 

# %%
import numpy as np
import plotly.graph_objects as go

# --- build axes for the heatmap ---
n_fp = vis_swath_row.shape[0]      # 256
n_s  = vis_swath_row.shape[1]      # samples in (trimmed) window

x_fp = np.arange(n_fp)             # footprint index (0..255)
y_idx = np.arange(n_s)             # sample index within your plotted window
y_rng = y_idx * dr                 # range in meters (relative)

hm_lo, hm_hi = np.percentile(vis_swath_row, PERCENTILE_CLIP)

m = max(abs(hm_lo), abs(hm_hi))
zmin, zmax = -m, m

colorscale = [
    [0.00, "#0000ff"],
    [0.49, "#000000"],
    [0.50, "#000000"],
    [0.51, "#000000"],
    [1.00, "#00ff00"],
]

fig = go.Figure(
    data=go.Heatmap(
        z=vis_swath_row.T,   # shape (n_s, n_fp)
        x=x_fp,              # x matches columns (footprints)
        y=y_rng,             # y matches rows (range in meters)
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        colorbar=dict(
            title=f"Raw ADC amplitude<br>(signed int16 counts;<br>clipped {PERCENTILE_CLIP[0]}–{PERCENTILE_CLIP[1]}%)"
        ),
        customdata=y_idx,    # keep the original sample index for hover
        hovertemplate=(
            "footprint=%{x}<br>"
            "range=%{y:.3f} m<br>"
            "sample_idx=%{customdata}<br>"
            "amp=%{z}<extra></extra>"
        ),
    )
)

fig.update_layout(
    width=900,
    height=1000,
    title="CASALS: One pushbroom advance (one row) — heatmap with range axis (Fs=1 GHz assumed)",
    xaxis=dict(title="Cross-track footprint index (0 … 255)  [step-major, sweep-minor]", constrain="domain"),
    yaxis=dict(title="Range (m) relative to plotted window start", autorange="reversed"),
    annotations=[dict(
        x=0.01, y=0.99, xref="paper", yref="paper",
        text=("Each vertical column = one footprint waveform (plotted as a heatmap)<br>"
              "Ordering B: step-major → sweep-minor (8 sweeps per step)<br>"
              f"Range axis uses dr=c/(2·Fs)={dr:.6f} m/sample (Fs=1 GHz assumed)<br>"
              f"Color scale: {PERCENTILE_CLIP[0]}–{PERCENTILE_CLIP[1]}% clipping; raw signed ADC counts"),
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.85)", bordercolor="black", borderwidth=1
    )]
)

fig.show()


# %%
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import binary_opening

hm_lo, hm_hi = np.percentile(vis_swath_row, PERCENTILE_CLIP)
amplitude_threshold = 300

# 对称色标范围，保证 0 在中间
m = max(abs(hm_lo), abs(hm_hi))
zmin, zmax = -m, m

colorscale = [
    [0.00, "#0000ff"],
    [0.49, "#000000"],
    [0.50, "#000000"],
    [0.51, "#000000"],
    [1.00, "#00ff00"],
]

Z = vis_swath_row.T.astype(np.float32)  # (samples, footprints)

# 1) 二值化 mask（固定阈值）
mask = np.abs(Z) >= amplitude_threshold

# 2) 开运算（保留你的参数）
structure = np.ones((3, 1), dtype=bool)
mask_open = binary_opening(mask, structure=structure, iterations=3)

# 3) 乘 mask（gating）
Z_gate = Z.copy()
Z_gate[~mask_open] = 0

print("[Gated heatmap]")
print(f"  amplitude_threshold = {amplitude_threshold} on |amp|")
print(f"  mask_open true ratio = {mask_open.mean():.4f}")
print("  display: Z outside mask_open is set to NaN (not drawn)")

fig = go.Figure(
    data=go.Heatmap(
        z=Z_gate,
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        colorbar=dict(
            title=(f"Raw ADC amplitude<br>(signed counts;<br>"
                   f"clipped {PERCENTILE_CLIP[0]}–{PERCENTILE_CLIP[1]}%)")
        ),
        customdata=Z,  # keep raw amplitude for hover
        hovertemplate="fp=%{x}<br>samp=%{y}<br>raw_amp=%{customdata}<br>shown_amp=%{z}<extra></extra>",
    )
)

fig.update_layout(
    width=900,
    height=1000,
    title=f"CASALS: Gated heatmap (Z masked by opened mask, |amp| ≥ {amplitude_threshold})",
    xaxis=dict(title="Cross-track footprint index (0 … 255)  [step-major, sweep-minor]", constrain="domain"),
    yaxis=dict(title="Waveform sample index within window (0 … N-1)", autorange="reversed"),
)

fig.show()


# %% [markdown]
# **Interactive 3D view**: one row as a waveform surface
# 
# We visualize a single pushbroom row as a 3D surface:
# 
# - **X**: waveform sample index (within the receive window)
# - **Y**: cross-track footprint index (0–255; Ordering B)
# - **Z / color**: raw ADC amplitude (signed int16 counts)
# 
# Because 256×2816 points can be heavy for interactive rendering, we downsample along the sample axis.
# Hover on the surface to read (sample, footprint, amplitude).
# 

# %%
# --- use existing vis_swath_row: shape (256, samples_rx), Ordering B ---
# 3D surface gets heavy quickly; use adaptive downsampling to avoid UI freeze.
P3_USER_DS_SAMPLE = 2
P3_USER_DS_FP = 1
P3_MAX_FP = 96
P3_MAX_SAMPLES = 240

p3_auto_ds_fp = max(1, int(np.ceil(vis_swath_row.shape[0] / P3_MAX_FP)))
p3_auto_ds_sample = max(1, int(np.ceil(vis_swath_row.shape[1] / P3_MAX_SAMPLES)))

# Final downsample = max(user setting, auto safety setting)
p3_ds_fp = max(P3_USER_DS_FP, p3_auto_ds_fp)
p3_ds_sample = max(P3_USER_DS_SAMPLE, p3_auto_ds_sample)

p3_row = vis_swath_row[::p3_ds_fp, ::p3_ds_sample].astype(np.float32)  # (n_fp, n_samp)
p3_fp = np.arange(vis_swath_row.shape[0], dtype=np.int32)[::p3_ds_fp]  # footprint index
p3_samp = np.arange(vis_swath_row.shape[1], dtype=np.int32)[::p3_ds_sample]  # sample index
p3_range_m = p3_samp.astype(np.float32) * dr

print("[3D input grid]")
print(f"  original shape  = {vis_swath_row.shape}  (footprints, samples)")
print(f"  downsampled     = {p3_row.shape}        (footprints, samples)")
print(f"  ds factors      = fp_step={p3_ds_fp}, samp_step={p3_ds_sample}")
print(f"  total vertices  = {p3_row.size:,}")
print("  note            = adaptive downsampling is enabled to keep 3D interaction responsive.")


# %%
# Robust symmetric clipping around zero for a stable diverging colormap.
p3_lo, p3_hi = np.percentile(p3_row, PERCENTILE_CLIP)
p3_abs = max(abs(float(p3_lo)), abs(float(p3_hi)))
if np.isclose(p3_abs, 0.0):
    p3_abs = 1.0
p3_row_clip = np.clip(p3_row, -p3_abs, p3_abs)

fig = go.Figure(
    data=[
        go.Surface(
            x=p3_range_m,  # 1D range axis (meters)
            y=p3_fp,       # 1D footprint axis
            z=p3_row_clip,
            colorscale="RdBu_r",
            cmin=-p3_abs,
            cmax=p3_abs,
            cmid=0.0,
            colorbar=dict(title=f"ADC counts<br>(symmetric clip +/-{p3_abs:.1f})"),
            lighting=dict(ambient=0.55, diffuse=0.60, specular=0.10, roughness=0.85, fresnel=0.03),
            lightposition=dict(x=120, y=-80, z=180),
            hovertemplate="range=%{x:.3f} m<br>footprint=%{y}<br>amp=%{z:.1f}<extra></extra>",
        )
    ]
)

fig.update_layout(
    title=(
        "CASALS 3D row surface (Ordering B: step-major -> sweep-minor) "
        f"| ds=({p3_ds_fp}, {p3_ds_sample})"
    ),
    scene=dict(
        xaxis=dict(title="Range (m)"),
        yaxis=dict(title="Footprint index (0..255)"),
        zaxis=dict(title="Amplitude (signed ADC counts)"),
        dragmode="turntable",
        camera=dict(eye=dict(x=1.8, y=-1.6, z=0.9)),
        aspectmode="manual",
        aspectratio=dict(x=2.4, y=1.2, z=0.7),
    ),
    template="plotly_white",
    height=760,
    uirevision="casals-row-3d",
    margin=dict(l=20, r=20, t=70, b=20),
)

fig.show()

# %%
# Extract maximum amplitude of each track index (footprint)
# vis_swath_row shape: (256, samples_rx) — each row is one footprint waveform
MAX_CLIP = 99.5
max_amp_per_track = np.percentile(vis_swath_row, MAX_CLIP, axis=1)  # shape (256,)

print("[Maximum amplitude per track]")
print(f"  vis_swath_row.shape = {vis_swath_row.shape}  (256 footprints, {vis_swath_row.shape[1]} samples)")
print(f"  max_amp_per_track.shape = {max_amp_per_track.shape}")
print(f"  max_amp range = {max_amp_per_track.min():.1f} .. {max_amp_per_track.max():.1f}  (ADC counts)")
print(f"  mean max_amp = {max_amp_per_track.mean():.1f}")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.arange(256),
        y=max_amp_per_track,
        mode="lines+markers",
        marker=dict(size=3),
        name="Max amplitude",
        hovertemplate="footprint=%{x}<br>max_amp=%{y:.1f}<extra></extra>",
    )
)

fig.update_layout(
    title=f"Maximum amplitude per track (row={vis_row_id}, step-major ordering)",
    xaxis_title="Cross-track footprint index (0 … 255)",
    yaxis_title="Max |amplitude| (ADC counts)",
    height=400,
)

fig.show()

# %% [markdown]
# ### Plot a single waveform (one row × one channel) — interactive
# 
# 目的：把 “热力图中的某一条横线” 单独展开成 1D 曲线，便于理解波形的峰值/噪声底。
# 
# 知识点（务必理解）：
# - 你画的是 **raw signed ADC counts**（int16），因此出现负值完全正常：它表示相对电子基线的偏移，不是“负光子”。  
# - 一条曲线对应：固定 footprint（一个 `Sweep{j}Step{k}` channel）在某一次推进（row）上的 **2816 点波形**。  
# - 如果要把 x 轴变成 ns 或 meters，需要采样间隔（digitizer 采样率或 TDMS timing properties）。
# 

# %%
# Interactive plot (Plotly): hover readout + zoom/pan
# If Plotly is not installed:  !pip install plotly

import numpy as np
import plotly.graph_objects as go

pw_row_id = 100
sweep_id = 7
step_id = 7
pw_group = "CASALS"
pw_channel = f"Sweep{sweep_id}Step{step_id}"   # change to any existing channel name

pw_ch = tdms[pw_group][pw_channel]
pw_a = pw_row_id * SAMPLES_PER_ROW
pw_b = pw_a + SAMPLES_PER_ROW

# raw waveform (full 2816 samples)
pw_wave_raw = np.asarray(pw_ch[pw_a:pw_b], dtype=np.int16)
pw_x_raw = np.arange(pw_wave_raw.size)

pw_rx_wave = pw_wave_raw[TX_SAMPLES_CUT[0] : pw_wave_raw.size - TX_SAMPLES_CUT[1]]
pw_tx_wave = pw_wave_raw[pw_wave_raw.size - TX_SAMPLES_CUT[1] : pw_wave_raw.size]

# build x-axes
# x_trim: 0..(N_trim-1)  (clean for viewing)
# x_abs:  original sample index in the 0..2815 coordinate system
pw_x_trim = np.arange(pw_rx_wave.size)
pw_x_abs  = np.arange(TX_SAMPLES_CUT[0], pw_wave_raw.size - TX_SAMPLES_CUT[1])

print("[Single waveform]")
print(f"  row_id      = {pw_row_id}")
print(f"  channel     = {pw_group}/{pw_channel}")
print(f"  raw length  = {pw_wave_raw.size} samples (expected {SAMPLES_PER_ROW})")
print(f"  trim rule   = drop head {TX_SAMPLES_CUT[0]}, drop tail {TX_SAMPLES_CUT[1]}")
print(f"  trimmed N   = {pw_rx_wave.size} samples")
print(f"  raw min/max = {pw_wave_raw.min()} / {pw_wave_raw.max()} (signed ADC counts)")
print(f"  trim min/max= {pw_rx_wave.min()} / {pw_rx_wave.max()} (signed ADC counts)")
print(f"  kept abs idx range = [{pw_x_abs[0]} .. {pw_x_abs[-1]}] in original 0..{pw_wave_raw.size-1}")
print(f"  tx segment length = {pw_tx_wave.size} samples")

# %%
# Plotly interactive line
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=pw_x_trim*dr,
        y=pw_rx_wave,
        mode="lines",
        name=f"{pw_channel} (trimmed)",
        customdata=pw_x_abs,
        hovertemplate="trim_idx=%{x}<br>abs_idx=%{customdata}<br>amp=%{y}<extra></extra>",
    )
)

fig.update_layout(
    title=f"CASALS single waveform (interactive, head/tail trimmed) — row={pw_row_id}, channel={pw_channel}",
    xaxis_title="Trimmed sample index (after head/tail cut)",
    yaxis_title="Raw ADC amplitude (signed int16 counts)",
    height=420
)

fig.add_annotation(
    x=0.01, y=0.99, xref="paper", yref="paper",
    text=("Head/tail trimming removes boundary/padding/Tx-related segments.\n"
          "Hover shows both trimmed index and original absolute sample index."),
    showarrow=False, align="left",
    bgcolor="white", opacity=0.85
)

fig.show()


# %%
# TX x-axis: absolute sample index in original 0..SAMPLES_PER_ROW-1
pw_tx_abs_idx = np.arange(
    pw_wave_raw.size - TX_SAMPLES_CUT[1],
    pw_wave_raw.size
)

fig_tx = go.Figure()
fig_tx.add_trace(
    go.Scatter(
        x=pw_tx_abs_idx,
        y=pw_tx_wave,
        mode="lines",
        name=f"{pw_channel} (TX segment)",
        hovertemplate=(
            "abs_idx=%{x}<br>"
            "amp=%{y}<extra></extra>"
        ),
    )
)

fig_tx.update_layout(
    title=f"CASALS TX segment (interactive) — row={pw_row_id}, channel={pw_channel}",
    xaxis_title="Absolute sample index (original waveform)",
    yaxis_title="Raw ADC amplitude (signed int16 counts)",
    height=300
)

fig_tx.show()


# %% [markdown]
# ## TEST

# %%
import numpy as np

def extract_vis_swath_row(vis_row_id: int, verbose: bool = False) -> np.ndarray:
    """
    Input:
      vis_row_id: one pushbroom advance index in [0, ROWS_PER_FILE-1]

    Output:
      vis_swath_row: shape (256, samples_rx)
        - Ordering B (recommended): step-major → sweep-minor
        - Head/tail trimming applied using global TX_SAMPLES_CUT = [head, tail]
    """

    # --- basic checks using your existing globals ---
    if vis_row_id < 0 or vis_row_id >= ROWS_PER_FILE:
        raise ValueError(f"vis_row_id out of range: {vis_row_id} (expected 0..{ROWS_PER_FILE-1})")

    if (not isinstance(TX_SAMPLES_CUT, (list, tuple))) or len(TX_SAMPLES_CUT) != 2:
        raise ValueError("TX_SAMPLES_CUT must be a list/tuple of length 2, e.g., [head_cut, tail_cut]")

    head_cut = int(TX_SAMPLES_CUT[0])
    tail_cut = int(TX_SAMPLES_CUT[1])

    if head_cut < 0 or tail_cut < 0 or (head_cut + tail_cut) >= SAMPLES_PER_ROW:
        raise ValueError(
            f"Invalid TX_SAMPLES_CUT={TX_SAMPLES_CUT}: require head_cut>=0, tail_cut>=0, head_cut+tail_cut < {SAMPLES_PER_ROW}"
        )

    if verbose:
        print("[Row selection]")
        print(f"  vis_row_id = {vis_row_id}")
        print("  Meaning: one pushbroom advance / one cross-track scan (row).")
        print(f"  This row contains {FOOTPRINTS_PER_ROW} footprint waveforms (256 channels).")

    # --- slice in the flattened 1D channel stream ---
    vis_a = vis_row_id * SAMPLES_PER_ROW
    vis_b = vis_a + SAMPLES_PER_ROW

    # --- extract row cube: (sweep, step, samples) ---
    vis_row_cube_sweep_step = np.empty(
        (SWEEPS_PER_CYCLE, WVL_STEPS_PER_SWEEP, SAMPLES_PER_ROW),
        dtype=np.int16
    )

    vis_missing = []
    for s in range(SWEEPS_PER_CYCLE):
        for st in range(WVL_STEPS_PER_SWEEP):
            ch = idx_map.get((s, st), None)
            if ch is None:
                vis_missing.append((s, st))
                vis_row_cube_sweep_step[s, st, :] = 0
            else:
                vis_row_cube_sweep_step[s, st, :] = np.asarray(ch[vis_a:vis_b], dtype=np.int16)

    if verbose:
        print("\n[Extraction result]")
        print(f"  vis_row_cube_sweep_step.shape = {vis_row_cube_sweep_step.shape}  (sweep, step, samples)")
        
        if vis_missing:
            print(f"  WARNING: missing {len(vis_missing)} channels; first few: {vis_missing[:10]}")
        else:
            print("  OK: all 256 channels found.")

        # --- apply head/tail cut (your exact logic) ---
        print("\n[Tx cut]")
        print(f"  TX_SAMPLES_CUT = {TX_SAMPLES_CUT}  (cut head, cut tail)")
        print(f"  Keep samples: [{head_cut} : {SAMPLES_PER_ROW - tail_cut}] in original 0..{SAMPLES_PER_ROW-1}")

    end_idx = -tail_cut if tail_cut > 0 else None  # avoid ':-0' -> empty
    vis_row_cube_sweep_step_rx = vis_row_cube_sweep_step[:, :, head_cut:end_idx]
    SAMPLES_PER_ROW_RX = vis_row_cube_sweep_step_rx.shape[-1]

    # --- Ordering B: step-major → sweep-minor ---
    vis_swath_row = (
        vis_row_cube_sweep_step_rx
        .transpose(1, 0, 2)  # (step, sweep, samples_rx)
        .reshape(WVL_STEPS_PER_SWEEP * SWEEPS_PER_CYCLE, SAMPLES_PER_ROW_RX)  # (256, samples_rx)
    )

    if verbose:
        print(f"  vis_row_cube_sweep_step_rx.shape = {vis_row_cube_sweep_step_rx.shape}  (sweep, step, samples_rx)")
        print(f"  samples_rx = {SAMPLES_PER_ROW_RX} (from original {SAMPLES_PER_ROW})")
        print("\n[Default]")
        print("  Using Ordering B (step-major → sweep-minor).")
        print(f"  vis_swath_row.shape = {vis_swath_row.shape}  (256 footprints, samples_rx)")
        print(f"  min/max in this row = {vis_swath_row.min()} / {vis_swath_row.max()}  (signed ADC counts)")

    return vis_swath_row


# %% [markdown]
# * Test

# %%
vis_row_id = 2
vis_swath_row = extract_vis_swath_row(vis_row_id)
vlo, vhi = np.percentile(np.concatenate([row_flat_sweep_major.ravel(), row_flat_step_major.ravel()]), PERCENTILE_CLIP)
vis_swath_row_clipped = np.clip(vis_swath_row, vlo, vhi)

mask = vis_swath_row >= 200
mask_open = binary_opening(mask, structure=np.ones((2, 1), dtype=bool), iterations=1)

plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.imshow(vis_swath_row_clipped.T, aspect="auto", origin="lower")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(mask_open.T, aspect="auto", origin="lower")
plt.show()

# %%
#from tqdm import tqdm
#pc_chunks = []  # each chunk: [x_fp, y_row, z_range_or_sample, intensity]
#
#for vis_row_id in tqdm(range(ROWS_PER_FILE)):
#    vis_swath_row = extract_vis_swath_row(vis_row_id)  # (256, n_samples)
#
#    # per-row clipping (stable visualization / intensity)
#    vlo, vhi = np.percentile(vis_swath_row, PERCENTILE_CLIP)
#    vis_swath_row_clipped = np.clip(vis_swath_row, vlo, vhi).astype(np.float32)
#
#    # valid-point mask (fixed threshold) + opening (your params)
#    mask = vis_swath_row >= 400
#    mask_open = binary_opening(mask, structure=np.ones((3, 1), dtype=bool), iterations=3)
#
#    # indices of valid points: (fp_idx, samp_idx)
#    ij = np.argwhere(mask_open)
#    if ij.size == 0:
#        continue
#
#    fp_idx = ij[:, 0].astype(np.int32)
#    samp_idx = ij[:, 1].astype(np.int32)
#
#    # coordinates (choose a consistent convention)
#    x_fp = fp_idx.astype(np.float32)                           # cross-track footprint index
#    y_row = np.full(fp_idx.shape[0], vis_row_id, np.float32)   # along-track row index
#
#    # z: use range (m) if dr exists; otherwise sample index
#    if "dr" in globals():
#        z = (samp_idx.astype(np.float32) * float(dr))
#    else:
#        z = samp_idx.astype(np.float32)
#
#    intensity = vis_swath_row_clipped[fp_idx, samp_idx]
#
#    pc_chunks.append(np.stack([x_fp, y_row, z, intensity], axis=1))
#
#point_cloud = np.concatenate(pc_chunks, axis=0) if pc_chunks else np.empty((0, 4), dtype=np.float32)


# %%
#from pathlib import Path
#import numpy as np
#
#ply_path = Path("casals_rowmask_pointcloud.ply")
#
#pc = point_cloud.astype(np.float32)  # shape (N, 4): x y z intensity
#n = pc.shape[0]
#
#print("[Save PLY]")
#print(f"  points = {n}")
#print(f"  path   = {ply_path.resolve()}")
#
#with open(ply_path, "w", newline="\n") as f:
#    f.write("ply\n")
#    f.write("format ascii 1.0\n")
#    f.write(f"element vertex {n}\n")
#    f.write("property float x\n")
#    f.write("property float y\n")
#    f.write("property float z\n")
#    f.write("property float intensity\n")
#    f.write("end_header\n")
#    np.savetxt(f, pc, fmt="%.6f %.6f %.6f %.6f")
#
#print("  done.")



