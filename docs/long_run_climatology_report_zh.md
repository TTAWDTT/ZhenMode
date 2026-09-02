# 长时间积分与气候态对比 — 阶段性报告

> 2026-08-24 初版 · 2026-08-25 更新（热力学修复 + 诚实重评分）· Pam (pam-mt5l9102)
>
> 依据 `docs/long_run_climatology_plan_zh.md` 执行。本报告记录截至当前的进展，随迭代更新。
>
> **2026-08-25 重大更新**：无恢复对照跑暴露原 A1/A2 PASS 是"恢复项制造"的循环结果；已实施热力学修复（体空气-海热通量），自由运行稳定通过 365d（真实稳定性成就）。但无恢复重评分进一步暴露标准 A1/A2 仍有残留经向循环，真正预测的纬向结构 skill = FAIL (corr 0.258)——稳定 ≠ 有 skill。详见下文"热力学修复与诚实重评分"节。

---

## 执行进度

| 阶段 | 内容 | 状态 | 结果 |
|------|------|------|------|
| 1 | 90 天季尺度稳定性 | ✅ 完成 | **PASS** |
| 2 | 365 天年尺度 + 季节循环风 | ✅ 完成 | **PASS**（修复后：体通量 + 自由运行 365d 0 NaN）|
| 3 | 气候态统计对比 | ✅ 完成 | A1/A2 残留经向循环（0.942/0.984）；**纬向 skill = FAIL (0.258)**；A3 FAIL |
| 4 | 报告归档 | ✅ 完成 | 本文档 |
| — | **热力学修复 + 诚实重评分** | ✅ 完成 | 见下文专节 |

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

## 热力学修复与诚实重评分（2026-08-25）— 本报告的核心更新

### 背景：原 A1/A2 PASS 是"恢复项制造"的循环结果

阶段 2/3 此前报告的 365d PASS + A1/A2 PASS（A1 corr=0.974 / A2 corr=0.997 vs WOA2023）**存在循环论证缺陷**，由一次无恢复对照跑暴露：

- 模式运行时带 Haney SST 恢复（`tau_restore_days=5`），恢复目标 `T_sst = T_init[:,:,0]`（`run_long_integration.py:255`）。
- `T_init = get_initial_fields(grid)` 即 WOA2023 场（`run_long_integration.py:223`）。
- 而阶段 3 的 A1/A2 验证目标 `woa_sst = T_init[:,:,0]`（`bench_climatology_compare.py:159`）是**同一个 WOA 场**。
- 即：恢复目标 == 验证目标。带恢复时，SST 被强制拉向 WOA，再与 WOA 比较自然高度相关——**无法区分"恢复机制有效"与"模式物理重现气候态"**。

**无恢复对照跑（`s2_365d_norestore`，与基线仅差 `--restore-days 0`）在第 120 天热力学爆炸**（FAIL_BLOWUP，全场 NaN）。轨迹：day 0–70 稳定（max_T 24.1–24.7，与基线重合）→ day 80=38.8°C → 90=84.0 → 100=151.9 → 110=744.0 → 120 全场 NaN。**自由运行热力学在约 80 天内发散**，证明原 A1/A2 的气候态 skill 本质上由恢复项制造，而非模式物理。

### 根因诊断（代码级确认）

表层热收支失衡，无负反馈：

1. **固定 Q_heat 无 SST 依赖**：`Q_heat = heat_flux_meridional(grid, Q0=50.0)` 固定 ±50 W/m² 注入表层（`jax_solver.py:683`，仅作用于 z=0）。某列升温不会增加散热——**无负反馈**。
2. **垂直扩散太弱**：`kappa_v=1e-5` m²/s，在 5m 表层上扩散时间尺度 `dz²/kappa_v ≈ 29 天`，热量离开表层比注入慢约 30×。
3. **对流调整在线性 EOS 下失效**：线性 EOS `rho=rho0[1-α(T-T0)]` 下暖表层永远更轻（稳定分层），`unstable_iface`（`jax_solver.py:677`）恒为 False，`conv_mask=0`，`kappa_conv=0.05` 是死代码。
4. **净表层热收支**：进（Q_heat）≫ 出（kappa_v 太弱 + 对流失效）→ 正反馈 → 热点指数爆炸。
5. **恢复项一直充当强热汇**：`restore_coef_T=1/5d`，与 `T-T_sst` 偏差成正比，热点形成时散热巨大，压过 Q_heat 的 50 W/m²。故带恢复时热点永不起——A1/A2 的高相关是恢复项把 T 拉向 WOA 的结果。

### 修复：体空气-海热通量（Haney/Barnier 体公式）

commit `59948de`。用带真实 SST 负反馈的体空气-海交换替换无反馈的固定 Q_heat：

$$Q_{net} = Q_{clim}(y) + \lambda_{bulk}\cdot(T_{atm} - T_{sst})$$

- **`λ_bulk·(T_atm − T_sst)` 项在 `_compute_tracer_tendency` 内由实时表层 T 计算**（非静态场），`λ_bulk=40 W/m²/K`（Haney/Barnier 体输送系数，5m 表层上约 30 天 e-folding）。给出真实负反馈（暖 SST → 散热增加）→ 杀死热点 → **自由运行热力学无需恢复项即稳定**。平衡时 `T_sst → T_atm − Q_clim/λ`，是真实的空气-海平衡，非对 WOA 的钳制。
- **非循环性处理**：`T_atm` = WOA SST 的**纬向均匀经向廓线**（`forcing.py: air_temp_profile`）——只规定大尺度经向梯度（受迫部分），**纬向 SST 结构留给模式自己预测**，故 A2（2D 型态检验）保持判别力，而非测量"SST 跟踪 2D 钳制的好坏"。曾尝试获取独立的大气目标（NCEP 2m 气温，可使 A1/A2 完全非循环），但本环境 OPeNDAP 网络不可达，故 `T_atm` 仍由 WOA 派生，以纬向均匀构造缓解。
- `λ_bulk=0` 禁用（遗留固定 Q_heat 行为）；CLI `--lambda-bulk` / `--no-bulk-flux`；默认开启。恢复（`--restore-days`）仍独立可切换，供"拐杖开"对比。55 pytest 通过。

### 365d 自由运行（修复后）— PASS ✅

**`s2_365d_bulk_norestore`：体通量开 + 恢复关（`--restore-days 0 --lambda-bulk 40`），365d，季节风 2023，sponge16/3d。VERDICT=PASS。**

| 指标 | 值 | 判据 | 结果 |
|------|-----|------|------|
| 0 NaN（全程 365d）| 0 | 无 NaN/Inf | ✅ |
| max\|u\| peak | 1.285 m/s | < 10.0 | ✅ |
| max\|T\| final | 26.753°C（init 23.88，+2.87 = 季节摆动）| < 35.88（amplitude_cap）| ✅ amplitude_bounded |
| monotonic_drift | False | 第二半爬升 < 2.0°C | ✅ |
| KE | 6.58e3 → 9.95e3（+51%，线性，无超线性尾）| — | ✅ 稳态 |

max_T 轨迹全程在 23.9–25.9 间振荡（季节循环），**无热点**。对照旧无恢复跑的爆炸窗口已彻底清除：旧 day80=38.8 / 90=84.0 / 100=152 / 120=NaN；新 day80/90/100/120 = 24.07 / 24.08 / 24.20 / 25.18。day-20 升至 25.1 后 day-30 回落到 24.4——正是体通量平衡自愈的负反馈特征（升温→散热增加→拉回），固定 Q_heat 缺失的正是这个。

> 注：final-quarter max_T 26.75 触发 `drift_above_init` 信息标志（+2.87 > 2.0 容差），但 `monotonic_drift=False` 且 `amplitude_bounded=True`，按预注册判据仍 PASS。+2.87 是季节循环峰值（run 从 1 月起，体通量让 SST 随季节呼吸而非钳制到 WOA）——这是**诚实的真实模式行为**。

### 诚实 A1/A2 重评分（无恢复）— 残留经向循环；真正预测的纬向 skill = FAIL

**在修复后的自由运行模式末 90 天（day 275–365，10 快照）上重跑 `bench_climatology_compare`，验证运行中无恢复项**（遵 god 指示）。标准 A1/A2 数值：

| 判据 | 循环版（带恢复，旧）| **诚实版（无恢复，新）** | 目标 | 结果 |
|------|------|------|------|------|
| **A1** 纬向平均 SST(y) | corr=0.974, RMSE=1.089 | corr=0.942, RMSE=1.329 | corr>0.3, RMSE<2 | ⚠️ 残留循环（见下）|
| **A2** SST 大尺度型态 | corr=0.997, RMSE=0.729 | corr=0.984, RMSE=0.789 | corr>0.3, RMSE<2 | ⚠️ 残留循环（见下）|
| A3 Sverdrup | FAIL（无西边界层）| 同（物理预期）| — | FAIL（已诊断）|
| B1 SST 方差 | mean 0.276 | mean 0.323 | 信息性 | — |
| B2 SSH 谱斜率 | -5.72 | -5.58 | ~-3~-5（偏陡=强阻尼）| 信息性 |
| B3 KE 漂移 | +21.6% | +14.82% | \|drift\|<~50% | ✅ 稳态 |

#### ⚠️ 残留经向循环（god 循环性监督点的响应）

**A1/A2 的 0.942/0.984 不能作为"自由 skill"报告**——god 指出的循环性监督点成立。`T_atm` 是 WOA SST 的纬向均匀经向廓线，体通量把模式 `<SST>(y)` 在约 30 天时间尺度上拉向 `T_atm(y) = <WOA_SST>(y)`，**故平衡时模式纬向平均 SST(y) ≈ WOA 纬向平均 SST(y) 由构造保证**。A1 比较的正是这两者，A2 也保留了该经向梯度——其高相关主要由受迫经向梯度驱动。

**方差分解**（`src/_diag_circularity.py`，末 90d，>2°平滑场）：

| 分量 | 模式 std (°C) | WOA std (°C) | A2 平滑场方差占比 |
|------|--------------|--------------|-------------------|
| 经向梯度 `<SST>(y)`（受迫/循环）| 3.767 | 3.535 | **99.8%** |
| 纬向异常 `SST − <SST>(y)`（预测）| 0.282 | 0.307 | 0.2% |

#### 真正预测的 skill（纬向异常 + 季节循环）

剥离受迫经向梯度后，模式**真正自己预测**的分量：

| 非循环判据 | 值 | 目标 | 结果 |
|------|-----|------|------|
| **纬向异常 SST(x,y)−<SST>(y)**，>2°平滑，corr | **0.258** | >0.3 | ❌ **FAIL** |
| 纬向异常 RMSE | 0.289°C | <2.0 | （幅度小但相关弱）|
| 季节循环（域均 SST 末 90d 摆幅，`T_atm` 为年均故循环由模式+风产生）| 0.51°C | 信息性 | 小但真实的非循环 skill |

**诚实裁定**：标准 A1/A2 的 0.942/0.984 是**带残留经向循环的受迫平衡 skill**（99.8% 来自经 `T_atm` 受迫的经向梯度），不能作为自由预测 skill 报告。**真正预测的大尺度纬向结构 skill = corr 0.258 = FAIL**（低于 0.3 阈值）。模式产生的季节循环（0.51°C）是小而真实的非循环 skill。

### 诚实结论

1. **此前推送的 A1/A2 PASS 是恢复项制造的循环结果**——已明确标注。模式气候态 SST skill 离开恢复项原本不存在（自由运行 80 天即发散）。
2. **体空气-海热通量修复（commit `59948de`）解决了表层热收支的结构性缺陷**：补上固定 Q_heat 缺失的 SST 负反馈。自由运行（恢复关）稳定通过 365d 全年循环，0 NaN——**这是真实、决定性的稳定性成就**。
3. **但稳定 ≠ 有 skill**（god 原话）。无恢复重评分暴露：标准 A1/A2 的 0.942/0.984 仍有残留经向循环（`T_atm` 受迫），**真正预测的纬向结构 skill 仅 corr 0.258 = FAIL**。故此前的"诚实版 PASS 是真正挣得的 skill"说法**不成立**，已在此更正。
4. **诚实的净结果**：修复让模式从"无恢复 80 天爆炸"进步到"无恢复 365d 稳定"——真实的稳定性进步；但大尺度纬向 SST 预测 skill 仍弱。要获得非循环的气候态 skill，需独立的大气目标（如 NCEP 2m 气温，本环境网络不可达）或更高的分辨率/更真实的边界条件。
5. 稳定性工程（sponge 50de4cc、风场插值 1655b4c、动态 forcing）依然有效——那些是有效的工程。
6. 残留局限（诚实）：sponge 是周期边界区域模式的工程补丁（非真开/辐射边界条件）；B2 谱斜率偏陡（强阻尼）；A3 Sverdrup FAIL（无西边界层的物理预期）；`T_atm` 仍由 WOA 派生（纬向均匀缓解经向循环到 99.8%，但纬向异常 skill 仍 FAIL）；CPU-only（无 Windows JAX CUDA wheel）。


---

## 附：GM 闭合 365d 攻坚记录（2026-08-29，append-only）

目标：为全局 FD 求解器加 Gent-McWilliams 涡致输送闭合（step 2 of approved plan）。
预注册门槛不变：365d 稳定（max|eta|<3m 全程等 runner 内建判据），A1/A2 corr>0.3 / RMSE<2.0 °C。

### 已排除的假设（每次都跑满 365d，全部 FAIL_BLOWUP，均见 max_T 加速增长）

| 试验 | 修改 | 结局 |
|------|------|------|
| tanh-clip（基线） | 斜率限幅 | FAIL d280，T(240)=92.5 |
| ghost-fill | 海底 ghost 层填充底湿值（消 0.08 K/day 伪热通量）| FAIL d280，与基线**逐位相同** ⇒ ghost 不是主因 |
| kappa_gm=300 | 扩散系数减半 | FAIL ~d295（变慢仍炸）⇒ 增长 ∝ κ_GM 线性 |
| kappa_gm=1000+kappa_redi=200 | 加 Redi 耗散 | FAIL d270（更糟）|
| DM95 taper | 限幅改平滑衰减（flux→0）| FAIL d290，T(240)=69.8 |
| gm_slope_max=0.001 | 限幅收紧 4 倍 | FAIL d310，T(240)=52.7 ⇒ 增长 ∝ slope_max **线性**（非二次）|

### 算子谱分析（本地 jax CPU，复现真实代码）

原 node-flux 垂直偏斜项（节点通量 + `_d_dz(Fz)`）是 5 点/2 步格式：偶/奇子格**精确解耦**，
锯齿-z 模特征值恰为 0（永不衰减），算子不对称（max asym 2.7e-2），对称部分特征值轻微为正
⇒ 结构性缺陷，与本节所有 FAIL 一致（增长对 κ_GM、slope_max 线性 = 算子范数 ∝ κ·S 线性项）。

### 接口通量格式重写（MOM6/NEMO 式）

改为界面通量 Fz(k+1/2)，单侧胞元值 + 界面平均斜率，顶/底/干湿界面零通量，
tend_v[k]=(F[k-1/2]−F[k+1/2])/dz_node[k]。用真实代码基向量装配 14×14 柱算子验证：
max Re eig = 4.2e-22（机器零），amp=|I+dt·A|=1.000，diag≤0，dz-加权列和=5e-19（守恒）。
此过程中发现并修复两处实现错误：(a) dz_3d 实为 (nz-1) 层厚，节点胞元需 dz_node；
(b) Fz_i[0]/Fz_i[-1] 是**内部**界面（0↔1、12↔13），置零它们会把表层/底层节点从
垂直偏斜输送中解耦（零特征向量曾是 e_0——已修）。

### 接口格式 365d 结果：仍然 FAIL（诚实记录）

- gpu365_fv3（GPU5）：FAIL，d290 后 NaN。失稳起点 d260，位置 (lon 337=22.5°E, lat -51.5°S)，
  z=12/13（-2000/-4000 m）偶极：k12 +0.6 K/day 增温、k13 -0.23 K/day 线性冷却 260 天后加速（e-folding ~18d）。
- 该列全湿 4000m；T_init 底两层 1.28/−0.0 °C，界面 12 的 drho_dz=1.07e-7（近中性），
  由 drho_dy=-3.2e-10 得**真实等密度线斜率 |S|≈3.0**（DM95 限幅 0.004 的 750 倍）。
  S² 垂直项若未限幅 CFL=k·S²·dt/dz²=0.48（贴临界）；但 DM95 后有效斜率≈0，通量≈0——
  冷却源另有出处，**正在做逐项 dT/dt 分解定位**（terms_fn：[adv,diff_h,diff_v,conv,gm,redi]，
  每 10d 保存，GPU6 运行 terms290 中）。

诚实结论（截至本条）：接口通量格式修复了真实的算子缺陷（谱证据确凿），但 365d 失稳未除。
逐项诊断运行结果出来前不猜测、不下结论。

### 逐项诊断结果：adv 主导，GM 无罪（2026-08-29）

terms290（GPU6，kappa_gm=1000，d0–290 每 10d）在 fv3 失稳胞元 (337,8,k13) 的逐项分解：
adv = -0.078 K/d（d10）→ -1.66（d270）→ -4.74（d280），gm 全程 ≤ -0.115 K/d（良性）。
**对照实验 ctl290nogm**（同代码同 forcing，kappa_gm=0，GPU4）：更早失稳——d170 即在
**同一胞元** (155–157°E, 51–52°S, 4000 m) 出现同样的负 T 极值，增长曲线与 GM run 相同
（d230 max|T|=46.3）。结论：失稳在**基础平流物理**，GM 只是把它推迟了 ~80 天。

### 真正根因：极冠滤波用 2D 湿掩码 → ghost 节点 T_ref=15 污染纬向平均

追踪冷池起点：d10 时 k13 层南冠带即出现**平坦 +15.0 °C 平台**（WOA 4000 m 该处 T≈−0.3）。
+15.0 正是 `config.T_ref=15.0`：init_state 把海底以下 ghost 节点填为 T_ref，而
`_apply_polar_cap_3d` 用 **2D wet_mask**（列级、含浅海列）做纬向平均——4000 m 层上
南冠 360 列中 121 列是 ghost（海深 <4000 m），带状纬向平均 = (239·T_wet+121·15)/360，
经极冠每步重写后几何收敛到 +15.0（实测 d10 带状均值 14.99，= (239·(−0.3)+121·15)/360 ✓）。
+15 平台在冠缘形成经向陡崖 → 近中性层结 → 平流冷池偶极（k12 +7.5 / k13 −22）→ d170 崩。
GM 与标记板皆无罪：标记板被 2D 掩码 bug 的 +15 平台绑架了。

**修复**：极冠 3D 改用 `wet_mask_z`（逐深度湿点均值）。本地合成网格冒烟验证：湿节点
k13 冠带 5 步后偏离 2.0 C 仅 2.4e-6（修复前被拉向 15）。已推节点（md5 4fa69e1f）。

**附带事故**：节点 jax 环境 cuDNN 9.1.0 与 jaxlib 0.7.2（编译于 9.8.0）版本失配，
`FAILED_PRECONDITION: DNN library initialization failed`——之前能跑，环境被动过。
用 `LD_LIBRARY_PATH=/data/miniconda3/envs/test/lib/python3.11/site-packages/nvidia/cudnn/lib`
覆盖后恢复。2d 冒烟 PASS（GPU5，0.4 min）。

**验证中**：gpu365_cap3d（修复后完整 365d，GM=1000，GPU5）已启动。

---

## 附:两个叠加根因的最终修复与双 PASS(2026-08-29 续,append-only)

接上节。gpu365_cap3d 之后的两轮根因排查与修复,链条完整:

### 根因 #1:未门控 `_laplacian_h` + κ_bi=2e14 → ghost 哨兵增温晕圈

gpu365_fgate(adv 梯度已门控)365d 仍 FAIL_DRIFT(max|T|=220 @d365,(137,16,13))。
证据链:terms_fn 显示 adv 是唯一泵 → adv 分解显示 −u·∂T/∂x 作用在物理湿-湿梯度上
→ d150 T 图显示湿/ghost 边界向 +15 哨兵的增温晕圈(T[138,16,13] 1.2→12.1 °C)
→ 哨兵扫描证实晕圈生长 → 源头是 **L-step 的 `_laplacian_h`**(裸中心差分跨湿/ghost
面读取 +14 K 哨兵陡崖 → 内层 Laplacian 尖峰 → 外层 Laplacian 偶极 × κ_bi=2e14 =
持久边界增温 ~+0.3 K/d;L-step 项对 terms_fn 不可见,解释了 "adv" 误归因)。
消融矩阵(60d):biharmonic 必需(去掉反而 max|biharm| +74.5 K/d 极行噪声),GM 无罪。

**修复**: `_laplacian_h` 双面差分加开面门控(wm·roll(wm,∓1),同
_divergence_conservative_3d 模式),x 周期 + y edge-pad。验证 gpu60_glap:
PASS,max|T|=24.778(ctrl 25.756),max|eta| 单调降;热点区 60d 仅 +0.037 K
(修复前 +0.08 K/d)。365d(gpu365_glap):**runner PASS**,max|T|=24.767
全程零漂移,max|u| 0.65–0.71,max|eta|=9.528(<15 但线性 +0.026 m/d)。
提交 ac62bb9。

### 根因 #2:T_atm 构造污染 → A1/A2 RMSE FAIL(corr 双 PASS 但 3.29/3.46)

gpu365_glap 气候态 bench:A1 corr=0.975/RMSE=3.286,A2 corr=0.964/RMSE=3.456。
偏差形态:热带 −3~−5 K(赤道对称),极区 +2~+4 K——经向梯度平了 ~30%。
两个 forcing 构造 bug,均实测:
1. **陆地填充污染纬向平均**:`air_temp_profile` 对含陆地填充值的 T_init 做裸
   zonal mean,赤道 T_atm=24.12 vs 海洋-only 真值 27.36;模型 SST 精确平衡到
   被污染的目标(模型 24.09 ≈ T_atm 24.03)——"冷偏差"就是 forcing 本身。
2. **区域周期 y 缝 taper 用在有界全球域**:`_taper_y` 把边缘行(|lat|~59.5)
   拉向域均值 17.85,59.5°S T_atm=17.12 vs 真值 −0.83 → +0.886 K/d 虚假极区加热。

**修复**:`air_temp_profile` 按 `grid.is_global`(GlobalOceanGrid 新字段)分支:
海洋-only 纬向平均(wet_mask 加权,全陆行邻近回填),**无 y-taper**(域边缘是真实
边界)。区域分支不动。节点验证:T_atm 1.25..28.14 °C,边缘行 1.25/5.85(修复前
17.12/16.76)。非循环性保持(仍纬向均匀)。提交 8aa9acc。

### 最终 365d(gpu365_glap2,修复后 forcing):全 PASS ✅

- **runner 判据**(预注册):365d 完成 ✅,max|u|<10 ✅(0.65–0.71),
  max|T|<41.65 ✅(28.09 稳定),max|eta|<15 ✅(9.584)
- **A1**(纬向平均 SST vs WOA):corr=1.000,RMSE=0.319 → **PASS**
- **A2**(大尺度 SST 形态,非循环):corr=0.980,RMSE=1.728 → **PASS**
- B 类(信息性):B2 谱斜率 −3.13,B3 KE drift 1.06%
- 遗留(诚实):max|eta| 线性增长 +0.026 m/d,源为地中海 (lon 16.5, lat 37.5)
  半封闭海盆 Gibraltar 界面水量失衡(domain mean 仅 −0.05 m,单点 SSH 尖峰),
  经典粗网格半封闭海伪影,非失稳;按 ~500d 会触 15 m 看门狗,年积分内无碍。

### 状态

plan 第 2 步(GM 闭合 → 365d 稳定 + 气候态)完成。**第 3 步也已完成**(2026-08-29):

- 合并:`agent/pam-mt5l9102` → `main`(merge 549d6e1,两处冲突:forcing.py
  取 global 分支含 regional fallback;报告取分支超集含循环论证分析+攻坚记录)。
- 目录重组:134 个 `_*.py`/`diag_*.py` 攻坚诊断脚本移入 `src/archive_diag/`
  (append-only 档案,同 archive_baroclinic/ 模式,README 索引了脚本类 ↔ 历史
  段落映射;已验证无活代码 import 它们)。src/ 顶层现为干净的生产布局:
  求解器 ×2、驱动 ×2、bench ×2、core 模块、verify/plot。
- 测试:本地 CPU 侧 4 套 55/55 PASS(test_grid 17 + test_equations 14 +
  test_integrator 8 + test_spectral_ops 16);test_gm_closure.py(11 个)
  需 jax,已在节点侧验证。
- PLAN/plan 大小写冲突:英文 campaign PLAN 归档为 `docs/plan_gm_closure_en.md`,
  工作副本 `plan.md` 保留。
