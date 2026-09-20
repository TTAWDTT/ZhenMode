# Baroclinic 激活路线图（针对真实数据 SLA 对比）

> **📦 已归档（Archived）** — 2026-09-20 仓库整理时移入 `docs/archive/`。
>
> 本文属于**已退役的区域谱模式（regional spectral solver）**时期的工作、
> 过程性工作日志，或已被后续文档取代的早期版本。保留它只是为了留存历史推理链，
> **不代表当前主线**。
>
> 当前主线是**全球有限差分模式**（`src/jax_solver_global.py`，见
> [`docs/solver_technical_report_zh.md`](../solver_technical_report_zh.md) 与
> [`docs/decisions.md`](../decisions.md)）。文档索引见 [`docs/README.md`](../README.md)。


> ⚠️ **方向已搁置（2026-08-23）**：本项目目的已重新定位为"稳定快速的谱方法海洋模式"，
> 不再以"点对点逼近真实观测"为成败判据。斜压激活（追涡）方向作为开发主线搁置——
> 四假设（H1-H4）证伪的结论（分辨率不足以解析变形半径）依然正确，但追涡属于另一条路
> （网格加密到 256×256），是新项目而非当前模式的补丁。相关实验脚本原移至
> `src/archive_baroclinic/`（该目录已于 2026-09-19 整理时清理）。
> 详见 `../repositioning_memo_zh.md`。
> 下方历史内容保留作档案，不改动。

> 状态：2026-08-22 由 synergy-max 起草。目标：把当前退化的 barotropic 行为升级为激活的分层
> (baroclinic) 水柱，从而让真实数据 SLA 对比有一个可辩护的“solid 指标”。
> 本文档是路线图，不是实现；涉及改动均在测试层/脚本层，不触碰生产默认 `PhysicsConfig`。

## 一、关键事实（代码读证，不是猜测）

1. **这个模型本来就是 baroclinic 静力原始方程谱模型**，不是简单的 barotropic 模型：
   - `src/pressure.py`：`compute_hydrostatic_pressure` 计算 p = barotropic 项 (ρ₀gη) +
     **baroclinic 项 (∫ρ′g dz′ 沿垂直网格累积梯形积分)** —— 注释明确写 "baroclinic contribution"。
   - `src/eos.py`：线性 EOS（默认）与 UNESCO 1980 非线性 EOS；`density_anomaly(T,S)` 返回 ρ′。
   - `src/tracers.py`：T、S 是独立 prognostic 变量，有完整平流 + 扩散 + 海表热通量方程。
   - `src/integrator.py`：Strang 分裂 IMEX；`_explicit_full_step` 注释说明 forward–backward
     RK2 耦合正是为了稳定**内重力波（imaginary eigenvalues）**——这是 baroclinic-aware 设计。
   - `src/jax_solver.py`：`JaxState = (u,v,T,S,eta)`，5 个 prognostics 齐全。

2. **T3-1 为什么只有 ~4mm 的 barotropic 风生 SSH 效应？** 因为
   `bench_t3_realdata.py` 调用 `make_solver(...)` 后未传 `T_init/S_init`，
   `_init_state(grid, physics, T_init=None, S_init=None)` 走默认分支：
   T ≡ `physics.T_ref` (15.0)、S ≡ `physics.S_ref` (35.0) 处处均匀 → ρ′ ≡ 0 →
   **baroclinic 压力项恒等于零** → 模型退化为 barotropic，再加上 `nu_bi=1e12`
   强阻尼网格尺度涡。这是**“分层未激活/参数阻尼”问题，不是缺少分层结构问题**。

3. **激活所需的全部基础设施已经存在且可用**：
   - `src/woa_data.py::get_initial_fields(grid)` 返回 (T_init, S_init) 的 (nx,ny,nz)；
     `WOA_DIR` 硬编码 = `data\woa`，且 `woa23_decav_t00_01.nc`、`woa23_decav_s00_01.nc`
     **在磁盘上确实存在**（另有 04 月变版本）。
   - `make_solver(...)` 返回的 `init_state(T_init, S_init)` 直接接受 WOA 分层初场
     （docstring: "If T_init/S_init are provided (e.g., from WOA climatology)"）。
   - `make_solver(T_sst=..., tau_restore_days=...)` 支持 Haney 海表温度恢复
     （Joseph 层上层），防止风生下涌累积暖水导致的虚假月尺度热失控。

## 二、路线图（按优先级）

### Step 0：验证前提（已完成，绿色）
- WOA 文件存在：`woa23_decav_t00_01.nc` / `woa23_decav_s00_01.nc` ✓
- `init_state` 支持分层初场 ✓  ·  `T_sst/tau_restore` 存在 ✓ · 工作树干净 ✓

### Step 1：激活 baroclinic 初场（最小改动，先跑通分层）
- 在 bench 脚本里加：
  ```python
  from src.woa_data import get_initial_fields
  T_init, S_init = get_initial_fields(grid)          # (nx,ny,nz)
  solver = make_solver(grid, physics, dt, forcing=wind, ...)
  state  = solver.init_state(T_init=T_init, S_init=S_init)
  ```
- 验证信号：`ρ′ = density_anomaly(T,S)` 不再处处为 0；水平密度梯度产生非零 baroclinic PGF；
  积分短期内能维持层结（不立刻崩溃）。
- **注意 NaN**：`woa_data.py` 主块专门打印 `T_init NaN count` —— WOA 在极浅/陆架可能插值出 NaN，
  需要 mask/填值（如用最近有效或海表参考值）。

### Step 2：重新标定粘性（关键，否则分层没用）
- `nu_bi = kappa_bi = 1e12` 是为**未分层**（均匀 T/S）压涡而标定的。
  在真实层结下这些值会过度阻尼 mesoscale 涡，SLA 方差依旧上不去。
- 做法：分层后测一组 `nu_bi`（如 1e10–1e12）＋ 水平 Laplacian `nu_h`（100→ 调），
  目标是让涡在不失稳的前提下能发展出 EKE / SSH 方差。
- 在测试/脚本层用 `dataclasses.replace(physics, nu_bi=...)` 覆盖，**不动生产默认**。

### Step 3：启用/评估海表温度恢复（防止热失控）
- 风生下涌会把暖水带到表层累积，无热量锚时会有虚假月尺度暖化（代码注释已警告）。
- 评估 `T_sst = WOA 海表 T`、`tau_restore_days ≈ 10–60` 一档，看是否让 T 场稳定。
- 这是**要不要、给多大**的决策点，先跑出来看再定，不预设。

### Step 4：短诊断积分，判断涡是否出现
- 判别信号（涡确实出现）：
  - SSH 标准差显著 > 现有 ~4mm；
  - EKE（涡动能）随时间不衰减到 0；
  - 流场出现 mesoscale 涡结构（可出图）。
- 若失败：调 Step 2/3 或增大风应力；若仍无涡，说明 dx≈9km / 128×128 不足以解析该区域
  主导涡尺度，路线需降级为“动力层结 + 统计/能量谱佐证”（路线B），提前告知，不硬撑 PASS。

### Step 5：真实数据 SLA 对比（baroclinic 版）
- 复用 `bench_t3_realdata.py` 的 SLA 缓存与 0.5° 平滑、corr 指标，
  只是初场/层结变了 → 重算 corr / RMSE / 模型SSH方差。
- 诚实预期：**不保证成功**。dx≈9km 与 0.5° 平滑尺度、以及“仅风应力(＋可选热通量)驱动”
  可能仍限制与真实 SLA 的匹配度。这步把一次“决定性物理失败”转成“可测试、有希望的问题”。

## 三、验证阶梯（承接已完成的调研）

| 层级 | 内容 | 对应本模型 |
|---|---|---|
| L0 数值 | MMS 指数收敛 + 时间积分阶 | 待补（可作为 Step 4 之前的独立 sanity） |
| L1 代码验证 | Stommel/Munk 解析解对比 | 待补（barotropic 解析解，快速校验 wind-driven gyre） |
| L2 理想物理 | Rossby / Kelvin / 海山 / 双环流统计平衡 | 待补 |
| L3 观测对比 | SLA 与实测相关（baroclinic） | **Step 5（本文档主线）** |
| L4 统计/谱 | KE 谱 ~k⁻³ 地转湍流 | 待补（baroclinic 后补，佐证动力合理性） |

> 说明：L3（真实数据 SLA）是用户最关心的“solid 指标”来源。但要让它 credible，
> 建议在推进 L3 前补 L1 的 Stommel 解析对比 + L0 的数值验证，否则 reviewer 会问
> “你的动力学本身可信吗”。Step 4 正好可顺带产出一个。

## 四、成功 / 失败判据（预先声明，防止事后挪移）

- **成功（可辩护）**：分层激活后模型 SSH 标准差显著提升（如量级进入 cm 级）、
  出现有结构的 mesoscale 涡、与真实 SLA 的 corr 改善且**在 3 独立设置（不同 nu_bi / 恢复时间窗）
  下稳定** —— 证明提升不是单个超参的偶然。
- **部分成功**：涡出现但 SLA corr 仍低 → 如实报告分辨率/强迫限制，作为动力层结可行性证据，
  不冒充观测验证（路线B 接手统计/谱指标）。
- **失败**：分层仍无涡 / corr 不改善 → 不挪移 bar，如实记录为“该配置无法解析主导涡”的结论。

## 五、边界与红线

- **不触碰**生产 `PhysicsConfig`（nu_bi=1e12 等）默认值；只在测试/脚本层 `dataclasses.replace` 覆盖。
- **不挪移已注册的证据 bar**（R1/R4），T3-1 的 FAIL 如实保留，路线图是一次**新实验**不是重写历史。
- 每个实验注册到 `research_experiment`，走双门（inspector 代码门 + auditor 红线门）。
- 遵循 append-only 研究轨迹。

## 六、执行更新（spin-up 诊断，2026-08-23）

> 本节为 append-only 追加，不改动上面历史结论。记录对“分层为何没起涡”的诊断进展。

### 6.1 已做：spin-up 长度诊断（H1）
- 新增 `src/bench_baroclinic_spin_evo.py`：分层(W斜)初场 + Haney 30d 海表恢复 +
  真实 2023-01 风，跑 90 天，每 10 天采样 SSH_std / SSH_max / 50-400km 谱带功率占比。
- dt=300 首跑：day50 后数值发散(NaN)，day10-50 “eddy_frac 0.10%→0.23%” 实为**逼近失稳
  前的数值伪影**，不是真实不稳定性。
- **修数值上限**：加发散 watchdog + `--dt`，dt=150（90 天稳定，SSH_max 全程 0.87-0.91m）。
- **H1 证伪（决定性）**：dt=150、90 天积分全有限、SSH_std 0.4833→0.5171（慢升），
  **eddy_frac(50-400km) 全程 ~0.0018%-0.0031%（≈0）**。子 meso 20-99km 0.000%。
  → 模型处于**正压大尺度平衡**，90 天内不发展斜压不稳定。**spin-up 长度不是阻碍**。

### 6.2 下一步（当前实验 #1：初始扰动播种）
- 已加 `bandlimited_noise_2d` + `inject_perturbation`（commit 2584f9d）：把 50-400km
  带限 m尺 T/S 扰动(表增强，垂直按 nz/4 衰减, S=0.1×T)注入初场。
- 冒烟验证：4d、amp0.5°C 跑通(exit0, wall164s)，扰动打印正常，eddy_frac 4d 仍 0.0001。
- **90 天 perturbed 跑（spin_evo_90d_pert05, amp0.5°C, seed7）已后台启动(~59min)**：
  - 若 eddy_frac 上升→播种有效，H1 是红鲱鱼，阻碍是“缺初始噪声”→ 接日风(Step: #3)或降粘性。
  - 若 eddy_frac 仍≈0→ `nu_bi=1e12` 过度耗散把种子压掉 → 实验 #2：在稳定包络内降 `nu_bi`
    （如 3e11/5e11，已知 <1e12 需配 Haney 恢复 + dt 保持稳定）。

### 6.3 与上面路线图的关系
- 上面的 Step 1（激活分层，已过）/ Step 2（粘性重标定）/ Step 3（海表恢复，已启 30d）仍适用；
  本次新增一个**更靠前的判别实验**（#1 播种），比 Step 2 更便宜、能先区分“缺种子”vs“过度耗散”。
- 红线不变：不挪 R1/R4，T3-1 的 FAIL 如实保留；生产 `PhysicsConfig` 默认值不动；结果在
  `results/spin_evo_90d_pert05.npz` + `logs/spin_evo_90d_pert05.log`。
