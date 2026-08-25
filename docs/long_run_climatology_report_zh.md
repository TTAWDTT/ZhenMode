# 长时间积分与气候态对比 — 阶段性报告

> 2026-08-24 · Pam (pam-mt5l9102)
>
> 依据 `docs/long_run_climatology_plan_zh.md` 执行。本报告记录截至当前的进展，随迭代更新。
>
> **2026-08-25 更新**：阶段 2 (365d) PASS、阶段 3 正式对比完成、阶段 4 归档。稳定性弧线闭合。

---

## 执行进度

| 阶段 | 内容 | 状态 | 结果 |
|------|------|------|------|
| 1 | 90 天季尺度稳定性 | ✅ 完成 | **PASS** |
| 2 | 365 天年尺度 + 季节循环风 | ✅ 完成 | **PASS**（16 格/3d sponge）|
| 3 | 气候态统计对比 | ✅ 完成 | A 类 A1/A2 PASS（corr 0.974/0.997）；A3 Sverdrup FAIL（无西边界层，物理预期）|
| 4 | 报告归档 | ✅ 完成 | 本文档 |

---

## 阶段 1：90 天积分 — PASS ✅

**配置**：JAX, 128×128×14, dt=150s, WOA2023 分层初场, NCEP 2023-01 月均风, Haney SST 恢复 τ=5d, nu_bi=1e12, 对流调整 on。发散看门狗 |eta|>3m。

**结果**（`results/long_run_s1_90d.npz`, `logs/long_run_s1_90d.log`）：

| 指标 | 值 | 判据 | 结果 |
|------|-----|------|------|
| NaN/Inf | 0 全程 | 无 | ✅ |
| max\|u\| peak | 1.685 m/s | <10 | ✅ |
| max\|u\| final | 1.426 m/s | — | 稳定 |
| monotonic_drift | False | False | ✅ |
| amplitude_bounded | True | True | ✅ |
| max\|T\| final | 24.685°C | <35.88 | ✅ |
| wall time | 3001s (0.83h) | — | — |

**关键观察**：
- max|u| 全程 0.76~1.69 m/s 波动，**无单调增长**——无能量泄漏/慢漂移
- max|T| 24~26.6°C 振荡，末四分之一(26.28) < 前半段(26.60)——**收敛不累积**
- SSH_std 升至 0.523 后稳定，KE 在 3.3e3~1.16e4 振荡无趋势——**统计稳态**
- 之前担心的 90 天速度爆炸（~2.5 个月 i≈126-127）**未复发**——dt=150 + nu_bi=1e12 + Haney 恢复压住
- drift_above_init=True（末段比初始高 ~2.4°C）是 informationonly：受热海洋合理平衡到高于初始 SST，符合"收敛而非锚定"口径

**结论**：模式稳定跑过一个季节并达到统计稳态。按门控进入阶段 2。

---

## 阶段 3：气候态统计对比 vs WOA2023 — A 类双 PASS ✅

用阶段 2 的 365d npz 做正式气候态对比（`src/bench_climatology_compare.py`），稳态窗口取末 90 天（day 275-365，10 快照）。**核心纪律：统计特征对比，非逐点场相关**——T3-1 教训（逐点 SLA corr 对涡主导观测 = 结构错配，误导性 FAIL）硬编码进脚本，排除项显式列出且脚本拒绝计算。

`results/climatology_s2_365d_sponge16/`（图 + `climatology_compare.npz`）, `logs/climatology_s2_365d_sponge16.log`。

### A 类（大尺度分量，主判据）

| 指标 | 模式 vs WOA | 判据 | 结果 |
|------|-------------|------|------|
| **A1 纬向平均 SST(y)** | corr=**0.974**, RMSE=1.089°C | corr>0.3, RMSE<2 | ✅ PASS |
| **A2 SST 大尺度型态**（>2°平滑去均值）| corr=**0.997**, RMSE=0.729°C | corr>0.3, RMSE<2 | ✅ PASS |
| A3 Sverdrup 平衡 | corr=0.007, mag_ratio=4.58 | corr>0.3 | ❌ FAIL（物理预期，非缺陷）|

### A3 Sverdrup 平衡 — FAIL（物理预期，已诊断）

A3 是原 deferred 的唯一 A 类判据，本次用 3D 快照（snap_00028..00037, day 275-365）正式计算。Sverdrup 线性稳态内部平衡：`β·V = curl(τ)/ρ₀`，V = ∫v·dz 深度积分经向输运。

- **LHS（模式 V）**：从 3D V 快照深度积分（layer-center 约定 `0.5*(v[k]+v[k+1])*dz_wet`，部分底 cell 按海底深度掩膜），末 90 天时间平均。
- **RHS（Sverdrup 预测）**：重建 2023 季节风场，同窗口时间平均 τ，`curl = ∂τ_y/∂x − ∂τ_x/∂y`，除以 ρ₀β。
- **对比**：内部域（排除 16 格 sponge 带 + 陆地）纬向平均 V(y) 去均值 pattern corr = **0.007**（目标 >0.3），2D 内部 |Vmod|/|Vsve| std 比 = **4.58**。

**FAIL 的物理诊断（非模式缺陷）**：Sverdrup 是线性稳态内部理论，**要求西边界层（WBL）闭合风驱环流**。本区域模式用双周期 + sponge 域（无 WBL），深度积分输运不受 Sverdrup 关系约束——纬向平均 V(y) 是一个平滑的南向正压模态（std 0.6 m²/s），与 curl 驱动的 Sverdrup 预测（std 2.1 m²/s，含强局地再分析 curl 极值）**不相关**。2D 场上模式输运方差是 Sverdrup 预测的 ~4.6×，正压环流模态主导。**这个 FAIL 量化了"无西边界层"的局限性，正是已记录的 sponge 区域模式的残余限制**——与 T3-1 教训一致：失败必须被诊断理解，而非粉饰为 PASS。

**A3 的意义**：A1/A2 PASS 证明模式大尺度 SST 气候态与真实 WOA 高度一致；A3 FAIL 诚实地界定了这个 skill 的边界——热力学气候态成立，但深度积分的动力学输运不满足 Sverdrup（因无 WBL）。要 A3 PASS 需真正的开边界/西边界层处理，超出当前 sponge 工程补丁范围。

A1/A2 相较 90d 预演（corr 0.956/0.990）**进一步收敛**——更长积分让模式 SST 气候态更接近真实 WOA 气候态。pattern corr 0.997 几乎完美匹配大尺度结构。

### B 类（动力合理性，信息性）

| 指标 | 值 | 解读 |
|------|-----|------|
| B1 SST 方差 | mean 0.276°C², max 2.75°C² | 有活跃变率，非死态 |
| B2 SSH 谱斜率 | -5.72 | 地转湍流预期 -3~-5，偏陡（强阻尼 nu_bi=1e12 + sponge 边缘耗散小尺度特征）|
| B3 KE 漂移 | +21.61% | 稳态窗口内缓升，\|drift\|<50%，大致稳定——即阶段 2 监控的 KE 漂移，确认为适度且线性 |

### 排除项（设计上不计算，T3-1 教训）

- ❌ 逐点 SLA/SSH 空间相关（结构错配）
- ❌ 中尺度涡逐个匹配（相位不可预测）
- ❌ 中尺度 SLA 方差绝对值（分辨率限制，H1-H4 已证伪）

### 90d 预演（历史，已被 365d 正式版取代）

阶段 2 跑前用阶段 1 的 90d npz 预演验证脚本正确性（末 45 天窗口）：A1 corr=0.956/RMSE=1.305，A2 corr=0.990/RMSE=0.892，B 类 slope=-5.76/KE drift=-19%。倾向正确，脚本无误。365d 正式版结果更优。

**关键结论**：模式气候态的大尺度 SST 结构与真实 WOA 气候态高度一致（corr 0.997）。这与 T3-1 的"FAIL"形成对比——**用统计标尺（而非逐点相关），模式展现出真实的气候态 skill**。验证了 repositioning 备忘的核心判断：模式的物理正确性在大尺度分量上是成立的。

---

## 阶段 2：365 天年尺度 — PASS ✅

**配置**：同阶段 1，但风场改为 **12 个月季节循环**（NCEP 2023 全年），季节风用 `interp_seasonal_wind()` 在月边界做 5 天窗口插值（`--wind-blend-days 5`），snap 10 天，`--save-3d`（逐快照流式写盘，供阶段 3 Sverdrup）。

### 首次运行：day 20–50 静默崩溃（已修复）

首次 stage-2 跑在 day 20–50 间静默退出（`EXIT=1`，无 traceback、无 NaN、无 npz）。崩溃点不固定（day40–50 / day20–25 两次）→ 非确定性，非数值爆炸。

**根因**：旧版 `run_long_integration.py` 为季节风预编译了 **12 个独立 JIT step 闭包**（每月一个 `make_solver`，各自把当月风场烘焙进 XLA 图作常量），单步 445 MB → 12 步 941 MB+，长时间运行内存累积触发进程级崩溃（系统 31.4 GB 总内存，free 仅 3.3 GB）。阶段 1 用单步（固定 1 月风）跑 90 天稳定，将崩溃隔离到 12-闭包设计而非数值。

**修复**（commit `9113d46`，已合并 main）：forcing 从 JIT 闭包常量改为 step 运行时参数。
- `jax_solver.py`：新增 `JaxForcing` namedtuple（5 个 forcing 叶子）；`_step_impl(state,p,forcing=None)` 给 forcing 时 `p._replace` 换叶子→动态 JIT 输入，其余常量折叠，内部物理函数全不变；`step(state,forcing=None)` 单参数=烘焙 forcing（所有旧 caller 不变），双参数=单图+forcing 作数据；`make_forcing(grid,...)` 构造 JaxForcing。
- `run_long_integration.py`：seasonal 路径 build **1 个** solver + 12 个 JaxForcing 数据对象，每步传当月 forcing。单编译图，无内存倍增。
- **物理等价验证**：动态 vs 静态路径 3 步随机 forcing，max diff 4e-14（浮点 round-off）。pytest 55 passed / 0 failed。15 天 seasonal smoke 跑通，数值与旧多图路径一致（day15 max|u|=1.407, max|T|=26.181）。

### 稳定性弧线：200d PASS → 365d FAIL(day340) → 365d PASS

修复内存崩溃后，跑长积分暴露出**与风场无关的物理稳定性问题**，经三轮诊断—修复迭代闭合：

| 运行 | sponge 配置 | 结果 | 关键诊断 |
|------|------------|------|----------|
| 200d 阶梯风 | 无 | day 150 爆炸 | 风场阶跃是触发，非根因 |
| 200d 风场插值 | 无 | day 140 爆炸（更早）| 证明风场非根因 |
| 200d + 8 格 sponge | 8 格/τ=5d | **PASS（day 140 hold）** | 加 sponge 有效 |
| 365d + 8 格 sponge | 8 格/τ=5d | **day 340 爆炸** | sponge 太窄，能量堆积点迁移到 y=8-15 |
| 365d + 16 格 sponge | 16 格/τ=3d | **PASS（全程 365d）** | 加宽 sponge 吸收全年涡能循环 |

**根因诊断（双周期边界条件）**：求解器用 `jnp.fft.fftfreq(nx/ny)`（`jax_solver.py:131-132`）做水平导数→FFT 隐含**双周期边界条件**。区域海洋模式中，风驱流在隐式周期边界处堆积，N/S 边缘 KE 累积到内部的 ~8.7×，最终爆炸。该问题**与风场无关**（阶梯风与插值风都在同一窗口爆炸）。

**Sponge 层（Rayleigh damping，标准 OGcm 手段）**，commit `50de4cc`：
- 构造余弦锥削阻尼场 `sponge_rate(nx,ny,1)`，N/S 边缘最大、内部为 0。
- 三个注入点：(1) 动量倾向 `dudt -= r*u`；(2) tracer 倾向 `dTdt += r*(T_clim - T)`；(3) **自由表面步** `eta/ubt/vbt *= exp(-r*dt_half)`——关键，因为精确 SW 求解对 k=0 恒等（守恒 eta），无耗散通道；不加则 eta 爬到 1.19 后爆炸。

**day-340 崩溃二次诊断**：8 格 sponge 在其带内有效（N 边 y=0-7 平坦 ~920），但**不稳定迁移到带外 y=8-15**：KE 312→542→2371（day 320→330 跳 4.4×），day 340 NaN。内部（y=56-71）保持平静（161）。典型的"sponge 太窄"失效模式。加宽到 16 格 + τ 缩到 3d 后，迁移点落入带内被吸收，全程通过。

**教训（day-270 误判）**：曾基于 70 天指数倍增时间拟合（113-270d doubling）判定 KE 漂移"非爆炸前兆"。但 day 280 后增长**加速**（超线性），指数模型低估了超线性尾。机制诊断（sponge 偏小）正确，"让它跑完"的判断错误。**教训：漂移增长率本身在上升时，重权对待；指数倍增时间模型低估超线性尾。**

### 最终运行（365d, 16 格/3d sponge）— PASS

`results/long_run_s2_365d_sponge16_3d.npz`, `logs/long_run_s2_365d_sponge16_3d.log`，wall 4.15h。

| 指标 | 值 | 判据 | 结果 |
|------|-----|------|------|
| diverged_at | None | — | ✅ |
| max\|u\| peak | 1.350 m/s | <10 | ✅ |
| max\|u\| final | 1.337 m/s | — | 稳定 |
| max\|T\| final | 24.733°C | <35.88 | ✅ |
| drift above init | False | — | ✅ |
| monotonic_drift | False | False | ✅ |
| amplitude_bounded | True | True | ✅ |
| NaN | 0 全程 | 0 | ✅ |

**关键观察**：
- **KE 线性**（非超线性）：6.58e3 (day10) → 1.02e4 (day365)，+55% over full year，无超线性尾——与 8 格 run 的 day-340 爆炸形成决定性对比。
- **max|u| 全程 1.32-1.35**，远低于 10.0 bound，平坦无爬升（8 格 run 崩溃前在爬）。
- **max|eta| 末值 1.124m**（阈值 3.0），线性缓升 +0.02/10d——sponge 有效压制了 eta 蔓延。
- day 340（8 格 run 的崩溃点）本 run 持 max_u 1.348 / KE 9.37e3 / 0 NaN。

**结论**：16 格/3d sponge 吸收了 8 格 sponge 无法承受的全年涡能循环。稳定性弧线闭合。

---

## 工具产出

| 文件 | 用途 |
|------|------|
| `src/run_long_integration.py` | 阶段 1/2 驱动器（JAX，看门狗，预注册判据，--save-3d, --seasonal-wind, --sponge-days/cells, --wind-blend-days）|
| `src/jax_solver.py` | 核心求解器（sponge 层三注入点，动态 forcing JaxForcing）|
| `src/bench_climatology_compare.py` | 阶段 3 气候态对比（统计特征，非逐点；A/B 类 + 排除项）|
| `results/long_run_s1_90d.npz` | 阶段 1 快照 |
| `results/long_run_s2_365d_sponge16_3d.npz` | 阶段 2 最终快照（PASS）|
| `results/long_run_s2_365d_sponge16_3d/` | 阶段 2 3D 流式快照（供 Sverdrup）|
| `results/climatology_s1/` | 阶段 3 预演图 + 数据 |
| `results/climatology_s2_365d_sponge16/` | 阶段 3 正式图 + 数据 |

## 环境

- 框架：JAX 0.11.1 (CPU backend)，主力 `jax_solver.py`
- Python：`C:\Python314\python.exe` + `PYTHONPATH=C:\Users\zhen.luo\Python\Python314\site-packages`
- 速度：~74.6ms/步，实时比 ~4000×；365d wall ~4.15h
- GPU：不可用（JAX 无 Windows CUDA wheel；jax-GPU 需 WSL2 路径，Erin 7× 验证）。全程 CPU。

---

## 总结

四阶段门控完成：

1. **阶段 1（90d 季尺度）PASS** — 模式跑过一个季节并达统计稳态。
2. **阶段 2（365d 年尺度 + 季节风）PASS** — 经 3 轮诊断—修复迭代闭合稳定性弧线：双周期边界条件根因 → sponge 层（8 格部分修复）→ day-340 二次诊断（sponge 太窄，能量堆积点迁移到 y=8-15）→ 16 格/3d sponge 吸收全年涡能循环。全程 0 NaN，max|u| 1.35，KE 线性无超线性尾。
3. **阶段 3（气候态对比）A1/A2 PASS, A3 FAIL（已诊断）** — 模式气候态大尺度 SST 与 WOA2023 高度一致：A1 纬向平均 SST corr=0.974/RMSE=1.089，A2 大尺度型态 corr=0.997/RMSE=0.729。A3 Sverdrup FAIL（corr=0.007）已诊断：无西边界层的区域模式不满足 Sverdrup 线性稳态平衡，正压模态主导深度积分输运（方差比 4.58×）——量化了 sponge 域的动力学局限，非模式缺陷。用统计标尺（非逐点相关），模式在热力学气候态上展现真实 skill，动力学输运的 Sverdrup skill 受限于边界处理。
4. **阶段 4（报告）** — 本文档归档。

模式在区域尺度、CPU/JAX 下能稳定跑过完整年循环并复现真实海洋气候态的大尺度热力学结构。残余局限：B2 谱斜率偏陡（强阻尼+sponge 耗散小尺度）、A3 Sverdrup 受无 WBL 限制、sponge 是区域周期 BC 的工程性补丁（非真开边界）。要 A3 PASS 需真正的开边界/西边界层处理。
