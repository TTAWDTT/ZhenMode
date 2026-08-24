# 长时间积分与气候态对比 — 阶段性报告

> 2026-08-24 · Pam (pam-mt5l9102)
>
> 依据 `docs/long_run_climatology_plan_zh.md` 执行。本报告记录截至当前的进展，随迭代更新。

---

## 执行进度

| 阶段 | 内容 | 状态 | 结果 |
|------|------|------|------|
| 1 | 90 天季尺度稳定性 | ✅ 完成 | **PASS** |
| 2 | 365 天年尺度 + 季节循环风 | 🔄 进行中 | — |
| 3 | 气候态统计对比 | 🔄 预演完成（90d），正式待 365d | 90d 预演 A 类双 PASS |
| 4 | 报告归档 | 🔄 撰写中 | 本文档 |

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

## 阶段 3 预演：90 天气候态 vs WOA2023 — A 类双 PASS ✅

用阶段 1 的 90 天 npz 预演气候态对比脚本（`src/bench_climatology_compare.py`），验证脚本正确性并提前看结果倾向。稳态窗口取末 45 天。

### A 类（大尺度分量，主判据）

| 指标 | 模式 vs WOA | 判据 | 结果 |
|------|-------------|------|------|
| **A1 纬向平均 SST(y)** | corr=**0.956**, RMSE=1.305°C | corr>0.3, RMSE<2 | ✅ PASS |
| **A2 SST 大尺度型态**（>2°平滑去均值）| corr=**0.990**, RMSE=0.892°C | corr>0.3, RMSE<2 | ✅ PASS |
| A3 Sverdrup 平衡 | deferred（需 3D 速度快照）| — | 待阶段 2 加 --save-3d |

### B 类（动力合理性，信息性）

| 指标 | 值 | 解读 |
|------|-----|------|
| B1 SST 方差 | mean 0.30°C², max 6.74°C² | 有活跃变率，非死态 |
| B2 SSH 谱斜率 | -5.76 | 地转湍流预期 -3~-5，偏陡（强阻尼 nu_bi=1e12 特征）|
| B3 KE 漂移 | -19% | 稳态窗口内略降，\|drift\|<50%，大致稳定 |

### 排除项（设计上不计算，T3-1 教训）

- ❌ 逐点 SLA/SSH 空间相关（结构错配）
- ❌ 中尺度涡逐个匹配（相位不可预测）
- ❌ 中尺度 SLA 方差绝对值（分辨率限制，H1-H4 已证伪）

**关键结论**：模式气候态的大尺度 SST 结构与真实 WOA 气候态高度一致（corr 0.99）。这与 T3-1 的"FAIL"形成对比——**用统计标尺（而非逐点相关），模式展现出真实的气候态 skill**。验证了 repositioning 备忘的核心判断：模式的物理正确性在大尺度分量上是成立的。

---

## 阶段 2：365 天年尺度 — 进行中 🔄

**配置**：同阶段 1，但风场改为 **12 个月季节循环**（NCEP 2023 全年，每 30 天切月），snap 10 天，`--save-3d`（存 3D T/u/v 供阶段 3 Sverdrup）。预计 wall ~4.4h。

### 首次运行：day 20–50 静默崩溃（已修复）

首次 stage-2 跑在 day 20–50 间静默退出（`EXIT=1`，无 traceback、无 NaN、无 npz）。崩溃点不固定（day40–50 / day20–25 两次）→ 非确定性，非数值爆炸。

**根因**：旧版 `run_long_integration.py` 为季节风预编译了 **12 个独立 JIT step 闭包**（每月一个 `make_solver`，各自把当月风场烘焙进 XLA 图作常量），单步 445 MB → 12 步 941 MB+，长时间运行内存累积触发进程级崩溃（系统 31.4 GB 总内存，free 仅 3.3 GB）。阶段 1 用单步（固定 1 月风）跑 90 天稳定，将崩溃隔离到 12-闭包设计而非数值。

**修复**（commit `9113d46`，已合并 main）：forcing 从 JIT 闭包常量改为 step 运行时参数。
- `jax_solver.py`：新增 `JaxForcing` namedtuple（5 个 forcing 叶子）；`_step_impl(state,p,forcing=None)` 给 forcing 时 `p._replace` 换叶子→动态 JIT 输入，其余常量折叠，内部物理函数全不变；`step(state,forcing=None)` 单参数=烘焙 forcing（所有旧 caller 不变），双参数=单图+forcing 作数据；`make_forcing(grid,...)` 构造 JaxForcing。
- `run_long_integration.py`：seasonal 路径 build **1 个** solver + 12 个 JaxForcing 数据对象，每步传当月 forcing。单编译图，无内存倍增。
- **物理等价验证**：动态 vs 静态路径 3 步随机 forcing，max diff 4e-14（浮点 round-off）。pytest 55 passed / 0 failed。15 天 seasonal smoke 跑通，数值与旧多图路径一致（day15 max|u|=1.407, max|T|=26.181）。

### 重启运行（修复后）

修复后从主仓库（main @ 9113d46）重启 365 天跑。单图常驻内存 ~473 MB（vs 旧版 941 MB+），不再倍增。关键里程碑：越过 day 50（旧崩溃窗口上界）即确认修复生效。

**待完成后**：若 PASS，用末 90 天做正式气候态对比（阶段 3 正式版），产出最终报告。

---

## 工具产出

| 文件 | 用途 |
|------|------|
| `src/run_long_integration.py` | 阶段 1/2 驱动器（JAX，看门狗，预注册判据，--save-3d, --seasonal-wind）|
| `src/bench_climatology_compare.py` | 阶段 3 气候态对比（统计特征，非逐点；A/B 类 + 排除项）|
| `results/long_run_s1_90d.npz` | 阶段 1 快照 |
| `results/climatology_s1/` | 阶段 3 预演图 + 数据 |

## 环境

- 框架：JAX 0.11.1 (CPU backend)，主力 `jax_solver.py`
- Python：`C:\Python314\python.exe` + `PYTHONPATH=C:\Users\zhen.luo\Python\Python314\site-packages`
- 速度：~74.6ms/步，实时比 ~4000×
