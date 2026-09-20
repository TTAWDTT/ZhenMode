# Ocean Solver — 时间线总结

> **📦 已归档（Archived）** — 2026-09-20 仓库整理时移入 `docs/archive/`。
>
> 本文属于**已退役的区域谱模式（regional spectral solver）**时期的工作、
> 过程性工作日志，或已被后续文档取代的早期版本。保留它只是为了留存历史推理链，
> **不代表当前主线**。
>
> 当前主线是**全球有限差分模式**（`src/jax_solver_global.py`，见
> [`docs/solver_technical_report_zh.md`](../solver_technical_report_zh.md) 与
> [`docs/decisions.md`](../decisions.md)）。文档索引见 [`docs/README.md`](../README.md)。


> 谱方法静力原始方程海洋模式 · 41 次提交 · 2026-08-19 ~ 2026-08-23
>
> 本文件按时间顺序梳理项目从第一个 commit 到当前的全部进展。
> 每条记录标注：**日期时间** · **commit** · **提交信息** · **做了什么 / 为什么 / 影响**。

---

## 项目一句话定位

从零构建一个**可验证、可追踪**的谱方法海洋环流模式，用真实数据检验其模拟能力边界。

- **方程**：静力原始方程（HPE）—— 水平动量 + 连续性 + 静水压力 + 状态方程 + 示踪物
- **数值**：伪谱方法（FFT 水平导数，无限阶精度）+ 非均匀 z-level 垂直差分
- **时间**：IMEX Strang 分裂（线性项精确矩阵指数 + 非线性项 forward-backward RK2）
- **加速**：JAX JIT，整个 step 编译为一个 XLA 计算图，比 numpy 快约 12×
- **域**：西北太平洋副热带（35°N, 150°E），128×128（0.1°, dx≈9.8 km），垂直 14 层

---

## 阶段 A：基础求解器（2026-08-19）

### 2026-08-19 15:22 · `9dd0a66` · feat: initial commit
**初始版本：numpy 谱方法求解器 + ETOPO 地形 + CDO 重映射。**
确立了项目骨架：`config.py`（frozen dataclass 配置）、`grid.py`（读 ETOPO2022 bathymetry 生成 OceanGrid）、`spectral_ops.py`（FFT 导数/拉普拉斯/去混淆）、`integrator.py`（numpy IMEX Strang）。域选在西北太平洋副热带开阔海域（陆地占比低，适合周期谱边界）。这是全部后续工作的地基。

### 2026-08-19 17:21 · `b74d394` · perf: vectorize loops + multi-threaded scipy.fft backend
**向量化 + 多线程 FFT 后端。**
消除网格点级 Python 循环，改为整场向量化运算（`jnp.fft.fft` 一次变换整条轴）。启用 scipy.fft 多线程后端。这是为 JAX 移植铺路的性能基线。

### 2026-08-19 18:17 · `c9984ad` · feat: JAX JIT solver — 12.4x speedup
**JAX JIT 移植：12.4× 加速（numpy 280 ms/步 → JAX 74.6 ms/步），最大差异 2.8e-13。**
确立了核心架构：整个 `step` 函数编译为一个 XLA 计算图，所有静态参数（波数、衰减因子、Coriolis 旋转角、浅水波频率、z 网格差分系数、强迫场）在 `_compute_params` 中**一次性预计算**并被 JIT 闭包捕获为常量，运行时零 Python 开销。`make_solver` 返回 `(step_fn, init_state, diagnostics)` 三元组。numpy 与 JAX 结果一致到浮点精度（2.8e-13），证明移植正确。

---

## 阶段 B：物理完善与稳定性修复（2026-08-20 ~ 2026-08-21）

### 2026-08-20 13:19 · `effe5ce` · fix(free-surface): use H_sw=sum(dz) as effective SW depth
**修正自由表面等效水深 H_sw = sum(dz)。**
此前自由表面浅水波方程的等效水深与正压速度提取、波频率投影不一致，导致质量/动量不守恒。改为 `H_sw = sum(dz)`（z 网格垂直跨度），使正压速度提取、波频率 $\omega=\sqrt{gHk^2}$、投影回 3D 场三者自洽。k=0 模恒等（平均质量动量守恒）。

### 2026-08-20 16:23 · `855d98c` · feat(physics): UNESCO EOS, Smagorinsky, quadratic bottom friction, 2D forcing
**物理参数化大扩充。**
- UNESCO 1980 非线性状态方程（`eos_type='unesco'`），适用于精确密度计算
- Smagorinsky 非线性亚网格闭合（`smag_cs`，默认 0 关闭）
- 二次底摩擦（`bottom_friction='quadratic'`, `cd=0.0025`）
- 2D 空间变化的海表强迫（风应力 + 热通量场，而非标量）

### 2026-08-20 21:41 · `bf0134d` · fix(stability): semi-implicit baroclinic PGF + barotropic wind
**半隐式斜压 PGF + 正压风应力进自由表面步。**
修复自由表面步长稳定性：把斜压压力梯度力的正压分量和正压风应力作为强迫放进精确浅水波求解器（`_free_surface_step` 的 particular solution），而非留在显式步。避免 Strang 分裂在风强迫下的共振失稳。

### 2026-08-21 12:43 · `d2a25ef` · fix(advection): rewrite to 3D advective form
**🔧 第一次爆炸修复：重写平流为 3D advective form。**
**根因**：通量形式平流 $-\nabla_h\cdot(u\mathbf{u}_h)$ 展开后多出 $-T\cdot\nabla_h\cdot\mathbf{u}_h$ 项。在 2D 浅水里水平散度=0 故无碍，但 3D HPE 中水平散度≠0（由连续性方程与 $w$ 平衡），这个多出的项成为**非物理正源**——在辐聚区正比于温度本身，形成指数增长正反馈，数十步即 NaN。
**修复**：改用 advective form $-(u\partial_x T + v\partial_y T + w\partial_z T)$ 并加入完整 3D 平流（含垂直项），从根源消除虚假源。$w$ 由连续性方程从海底向上积分诊断。

### 2026-08-21 13:02 · `b101818` · feat(data): WOA2023 climatology reader
**WOA2023 气候态温盐读取器与插值器。**
`woa_data.py::get_initial_fields(grid)` 返回 (T_init, S_init) 的 (nx,ny,nz) 实测气候态场。从此初场可从理想均匀场切换为真实层结（22°C 表深层温差），为真实数据对比铺路，但也直接触发了下一次爆炸。

### 2026-08-21 17:04 · `e8ae3cc` · feat(stability): CFL scan under WOA stratification
**WOA 层结下 CFL 扫描 + 内波稳定限诊断。**
系统扫描 dt=10~900s（各积分 2 小时）：dt≤450 STABLE，dt=600 起 DEGRADED。发现外层重力波 CFL（~46s）**不是**稳定限——forward-backward RK2 半隐式处理了外层模；实际稳定限由内部斜压波决定，可持续到 ~450-600s。这一诊断纠正了"dt=300 爆炸是裸 CFL 越界"的误判。

### 2026-08-21 17:48 · `33fb58c` · fix(stability): scale-selective biharmonic viscosity/diffusivity
**🔧 第二次爆炸修复：scale-selective 双调和黏性/扩散。**
**根因**：WOA 真实层结 22°C 温差产生强斜压 PGF，与显式 RK2 耦合产生网格尺度不稳定，dt=300s 约 30-40 步发散。
**修复**：加双调和算子 $\nabla^4$（$\propto k^4$），谱空间衰减 `exp(-nu_bi*k^4*dt)`。它**尺度选择性**极强——网格尺度模被强力压制，环流尺度几乎不受影响。`nu_bi=1e12` 标定后，三种配置（无强迫/风+热/仅风）dt=300s 均稳定 100 步，温度无漂移。

### 2026-08-21 17:51 · `cc03fb5` · feat(visual): field maps, cross-sections, time series, animation
**可视化工具：场图、剖面、时间序列、动画。**
`plot_fields.py` / `plot_animation.py`。用于后续诊断结果的可视化呈现。

### 2026-08-21 21:26 · `dc61ca4` · feat(pytorch): port to PyTorch + JAX-equivalence benchmark
**PyTorch 移植 + JAX 等价性基准。**
`torch_solver.py` 独立实现一套，与 JAX 版交叉验证。多框架等价性是求解器正确性的额外保证。

### 2026-08-21 21:39 · `fa9822b` · perf(pytorch): CSE horizontal laplacian
**PyTorch 版 CSE（公共子表达式消除）优化水平 Laplacian。**
在 `momentum.py` / `tracers.py` 残差中复用已算的 Laplacian，减少重复 FFT。

---

## 阶段 C：月尺度稳定 + 可视化验证（2026-08-22 凌晨）

### 2026-08-22 00:58 · `17d1b57` · feat(verify): real-WOA stability diagnostics + summary PPT
**真实 WOA 层结稳定性诊断 + 总结 PPT。**
确认 biharmonic 修复后真实层结下短积分稳定。生成 `ocean_solver_summary.pptx` / `ocean_solver_short.pptx`。

### 2026-08-22 01:00 · `23aa58d` · docs(report): mark 5.3 real-stratification instability resolved
**文档：标记 5.3 实测层结不稳定已由 biharmonic 修复解决。**
更新 `report.md` 第五节，记录第二次爆炸的根因与修复，状态表更新为"WOA 实测 + 带强迫 + dt=300s ✅ 稳定"。

### 2026-08-22 01:05 · `980e0ca` · feat(verify): multi-day forced WOA stability + no-T-drift
**多日强迫 WOA 稳定性 + 无漂移检查。**
`verify_real_run.py` 短期烟雾测试，确认多日强迫下温度不漂移。

### 2026-08-22 01:44 · `eaf9a5b` · feat(diag): locate month-scale T-drift hot spot
**月尺度 T 漂移热点定位 + --days 参数。**
`diag_month_drift.py` 把 30 天积分的温度漂移定位到**西北角（NW corner, i=0..1, j=115..124, k=0 海表）**——一处上暖下冷的准稳定柱，比海表冷却还强的边界"热泵"。加 `--days` 参数支持任意积分时长。

### 2026-08-22 01:47 · `50c137f` · feat(forcing): real NCEP/NCAR R1 10m wind stress loaders
**NCEP/NCAR R1 真实风应力读取器（免认证）。**
`wind_reanalysis.py` 从 NOAA PSL 下载 NCEP/NCAR R1 10m 再分析风，转为风应力场。支持月均与日均气候态。免认证 urllib + 本地 netCDF4（OPeNDAP 在 Windows 上 `OSError(-75)`）。

---

## 阶段 D：月尺度稳定修复 + 验证阶梯（2026-08-22 上午~下午）

### 2026-08-22 11:25 · `318fdc8` · fix(forcing): stabilize month-scale run via taper + convection + SST restoring
**🔧 第三次漂移修复：taper + 对流 + SST 恢复稳定月尺度积分。**
**根因**：偏微分域 x/y 双向周期（伪谱 FFT），但强迫 y 方向非周期（Stommel 风应力与经向热通量在南北边取相反非零值）→ 周期谱缝处阶梯不连续 → 网格尺度强迫能量钉死在 NW 角 → 边界热泵 → 温度单调爬升至 43.78°C。
**修复**（三个物理上正交的环节组合）：
1. **强迫渐变**（`_taper_y`）：y 边 8 格内升余弦渐变到零，消除周期缝阶梯不连续
2. **对流调整**（`kappa_conv`）：静力不稳定柱整柱垂直混合（仅在边界伪源清除后才生效）
3. **SST 恢复**（Haney, τ=5d）：海气负反馈锚定稳定平衡

判定口径：受热海洋本就平衡到高于初始 SST 几度，故判据是"**收敛而非锚定**"——`monotonic_drift=False` + `amplitude_bounded=True` + 无 NaN + max|u|<10。

### 2026-08-22 11:25 · `4e377bf` · Merge branch 'a/month-scale-pressure'
合并月尺度压力分支。

### 2026-08-22 11:31 · `54cedb4` · feat(forcing): add y-edge taper to real reanalysis wind + --real-wind flag
**真实再分析风加 y 边缘 taper + --real-wind 月尺度标志。**
把 taper 修复也应用到真实 NCEP 风场（不只理想化 Stommel）。加 `--real-wind` CLI 标志。

### 2026-08-22 11:49 · `6d286b2` · docs(README): document month-scale stable forced runs
**README 文档化月尺度稳定强迫运行。**
记录理想化风与真实风两种 30 天稳定配置，以及"收敛不累积"的通过判据。

### 2026-08-22 13:25 · `b9af9fc` · feat(diag): add blowup-localization tool for 90d velocity explosion
**90 天速度爆炸定位工具。**
`diag_localize_blowup.py`。定位 90 天积分在 ~2.5 个月、i≈126-127/~30°N 处的真实内部斜压不稳定（已排除是谱缝伪影），属长时域稳定性问题，与精度阶梯分开追踪。

### 2026-08-22 14:45 · `fd19b03` · feat(bench): add Tier-1 accuracy suite; all 5 checks PASS
**⭐ Tier-1 精度套件：5 项全 PASS。**
`bench_accuracy.py` 实现验证阶梯第一层：
- T1a 谱导数收敛（1e-17）
- T1b 地转平衡（3.4e-5 m/s²）
- T1c Rossby 波西传相速（1.44%）
- T1d 重力波 c=√(gH)（0.54%，需切到无黏物理）
- T1e Ekman/Sverdrup（全部）

### 2026-08-22 17:31 · `7af55fc` · feat(bench): add Tier-2/3 verification benchmarks
**Tier-2/3 验证基准。**
- T2-1 惯性振荡（`bench_t2_inertial.py`）
- T2-2 地转调整（`bench_t2_geostrophic.py`）
- T3-1 真实数据 SLA 对比（`bench_t3_realdata.py`）

### 2026-08-22 18:26 · `5c44e59` · docs(verification): T2-2 geostrophic PASS; T3-1 real-data FAIL
**T2-2 地转调整 PASS；T3-1 真实数据 FAIL + 诊断。**
- T2-1 惯性振荡：频率误差 0.07%，振幅漂移 3.5e-4 → PASS
- T2-2 地转调整：活跃区 corr=0.9984，残差比 0.052 → PASS（需 H_sw=200m + 边缘海绵）
- **T3-1 真实数据 SLA**：corr=-0.386, RMSE 0.26m, 模型 SSH std 4mm vs 观测 26cm → **FAIL**
  - 诊断：模型正压 SSH 仅 4mm 但大尺度符号正确（Sverdrup 平衡）；观测 SLA 被斜压中尺度涡主导（26cm），正压模型原理上无法产生；负相关是噪声主导非环流反了；**预注册证据线未为凑 PASS 改标尺（守住 R1/R4 红线）**

### 2026-08-22 19:11 · `e0eb16e` · docs(verification): add Chinese verification memo
**中文验证备忘。**
`verification_memo_zh.md`，把验证阶梯结果用中文记录。

---

## 阶段 E：斜压激活诊断实验（2026-08-22 晚 ~ 2026-08-23）

T3-1 失败后，路线图判断问题是"分层未激活/参数阻尼"，而非缺少分层结构。设计四个假设（H1-H4）逐一测试，**全部证伪**。

### 2026-08-22 20:12 · `d894497` · feat(bench): add Step 1 baroclinic activation diagnostic + roadmap
**Step 1 斜压激活诊断 + 路线图。**
`bench_baroclinic_step1.py` + `baroclinic_activation_roadmap_zh.md`。
关键事实（代码读证）：模型本就是 baroclinic HPE 谱模型（`pressure.py` 有 baroclinic 项 ∫ρ'g dz'，`tracers.py` T/S 独立预报）；T3-1 只有 4mm 是因为 `bench_t3_realdata.py` 未传 T_init/S_init → 默认均匀 T/S → ρ'≡0 → 退化为正压。激活基础设施（WOA 读取、`init_state(T_init,S_init)`、Haney 恢复）均已存在。

### 2026-08-22 21:19 · `78f8e9a` · feat(bench): Step 2 viscosity scan + spatial-structure diagnostic
**Step 2 黏性扫描 + 空间结构诊断。**
`bench_baroclinic_step2.py`。扫描 nu_bi，测空间结构。

### 2026-08-22 21:37 · `1c627d9` · feat(bench): Step 5 real-data mesoscale band-pass SLA comparison
**Step 5 真实数据中尺度带通 SLA 对比。**
`bench_baroclinic_step5.py`。

### 2026-08-22 21:45 · `858e03c` · feat(bench): Step 5 spatial-mismatch diagnostic
**Step 5 空间失配诊断：径向谱、带通相关、相干/相位。**
`bench_baroclinic_step5_diag.py`。发现模型 SSH 在 ~460km 以下基本无中尺度功率，能量困在 550-777km 环流尺度。

### 2026-08-22 21:56 · `3d679c1` · feat(bench): add spin-up evolution diagnostic
**spin-up 演化诊断（测试斜压不稳定是否随时间发展）。**
`bench_baroclinic_spin_evo.py`。分层(WOA)初场 + Haney 30d 恢复 + 真实风，跑 90 天，每 10 天采样 SSH_std / SSH_max / 50-400km 谱带功率占比（eddy_frac）。
- dt=300 首跑 day50 后 NaN，day10-50 的 eddy_frac 0.10%→0.23% 实为逼近失稳的数值伪影。

### 2026-08-22 22:47 · `e4177b9` · fix(bench): add divergence watchdog + dt option
**发散看门狗 + dt 选项。**
加 `ETA_BLOWUP_M=3.0` 看门狗，失稳时记 DIVERGED 而非误导性 NaN 表。加 `--dt`。dt=150 下 90 天稳定。

### 2026-08-22 23:58 · `2584f9d` · feat(bench): add initial perturbation seeding
**初始扰动播种。**
`bandlimited_noise_2d` + `inject_perturbation`：50-400km 带限 T/S 扰动（表层增强，垂直按 nz/4 衰减）注入初场，测试"缺初始噪声"假设。

### 2026-08-23 00:01 · `ef92149` · docs(bench): record H1 falsification and perturbation experiment pivot
**H1 证伪 + 扰动实验转折。**
- **H1（spin-up 长度不足）证伪**：dt=150、90 天积分全有限，SSH_std 0.4833→0.5171 慢升，**eddy_frac(50-400km) 全程 ~0.0018%-0.0031%（≈0）**。模型处于正压大尺度平衡，90 天内不发展斜压不稳定。spin-up 长度不是阻碍。

### 2026-08-23 11:04 · `aae0c26` · feat(bench): add daily-climatology wind forcing option
**日常气候态风强迫选项。**
`daily_wind_forcing`：用 NCEP 日均气候态风替代月均风（31 个日步函数），测试 H4（风缺高频变率）。

### 2026-08-23 13:57 · `137d3cd` · docs: add comprehensive progress report
**综合进度报告（中文，全项目时间线）。**
`progress_report_zh.md`。汇总全部进展。记录 H2/H3/H4 证伪：
- **H2（缺初始扰动）证伪**：0.5°C 中尺度扰动 10 天内被 nu_bi=1e12 阻尼到 ≈0
- **H3（黏性过强）证伪**：nu_bi 在 [3e11, 5e11] 无稳定涡窗口（3e11 day70 发散）
- **H4（风缺高频变率）证伪**：日均风 eddy_frac 仍 0.0000%

---

## 最终结论

### 求解器层面：可信 ✅
- 算子精度：谱导数 1e-17，地转平衡 3.4e-5，Rossby 波 1.44%，重力波 0.54%
- 理想化动力学：惯性振荡 0.07%，地转调整 corr 0.9984
- 稳定性：三次爆炸/漂移均已定位根因并修复，月尺度积分在合理参数下稳定

### 物理能力边界：分辨率限制 ⚠️
- 128×128（dx≈9km）在西北太平洋**无法解析第一斜压变形半径**（该海域 ~30-50km）
- 模型能量困在环流尺度（550-777km），中尺度带（50-400km）功率基本为零
- 即使注入扰动种子、降黏性到稳定极限、用日均风强迫，斜压不稳定仍不发展
- **这是分辨率/配置限制，不是求解器 bug**

### 真实数据对比：如实失败 📊
- T3-1 corr=-0.386，模型 SSH std 4mm vs 观测 SLA std 26cm
- 模型对其能表现的分量（大尺度风驱正压海面）给出符号正确的响应
- 失败于其原理上不能表现的分量（斜压中尺度涡）
- 预注册证据线未移动（R1/R4 红线守住）

### 四假设汇总
| 假设 | 内容 | 判定 |
|------|------|------|
| H1 | spin-up 不够长 | **证伪**（90 天 eddy_frac ≈ 0） |
| H2 | 缺初始扰动 | **证伪**（0.5°C 种子 10 天内被阻尼） |
| H3 | 黏性过强 | **证伪**（[3e11, 5e11] 无涡窗口） |
| H4 | 风强迫缺高频变率 | **证伪**（日均风 eddy_frac 仍 0） |

---

## 下一步选项（Route B）

| 路线 | 内容 | 预期 | 成本 |
|------|------|------|------|
| **B-1** | 谱证据 + 诚实报告 | 用径向谱/能量谱佐证"分辨率不足导致涡缺失"，定位贡献为"谱方法海洋模式的验证与能力边界" | 低（已有数据） |
| **B-2** | 网格加密到 256×256 | dx≈4.5km 接近变形半径，可能激发斜压涡 | 高（JAX 重编译 + 计算量 4×） |
| **B-3** | 重新定位贡献 | 转向大尺度动力学（风驱环流、Sverdrup 平衡、地转调整），不做中尺度涡 | 中（需重新设计实验） |

---

## 三次稳定性危机速查

| # | 现象 | 根因 | 修复 | commit |
|---|------|------|------|--------|
| 1 | 小扰动数十步 NaN | 通量形式平流缺垂直项 → 虚假源 $-T\cdot\nabla_h\cdot\mathbf{u}_h$ 正反馈 | 改 3D advective form | `d2a25ef` |
| 2 | WOA 层结 30-40 步 NaN | 强斜压 PGF 与显式 RK2 耦合不稳 | scale-selective biharmonic (nu_bi=1e12, ∝k⁴) | `33fb58c` |
| 3 | 月尺度温度爬升至 43.78°C | 周期谱缝非周期强迫 → NW 角热泵 | y 边缘 taper + 对流调整 + SST 恢复(τ=5d) | `318fdc8` |
