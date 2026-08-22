# Baroclinic 激活路线图（针对真实数据 SLA 对比）

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
