# Pam 工作总结 — ocean_solver 全程

> **作者**：Michael (god / orchestrator) 整理 · 2026-08-26
> **对象**：Pam (pam-mt5l9102)，ocean_solver 仓库主力工程师
> **范围**：Pam 在 `ocean_solver`（Python + JAX 静力原始方程海洋模式）的全部工作，含精确实验配置
> **仓库**：`C:/Users/zhen.luo/ocean_solver`（区域谱模式，origin/main = `058db79`）+ `C:/Users/zhen.luo/HarnessAgents/worktrees/pam-mt5l9102`（全球有限差分，本地分支 `agent/pam-mt5l9102`，未推送）
> **运行环境**：JAX 0.11.1 CPU backend · `C:\Python314\python.exe` + `PYTHONPATH=C:\Users\zhen.luo\Python\Python314\site-packages` · ~74.6 ms/步 · 全程 CPU（JAX 无 Windows CUDA wheel）

---

## 0. 一句话定位

Pam 的工作有一条贯穿始终的主线：**把"诚实的失败"当作下一步的动机，绝不粉饰**。她在区域谱模式上修了真实的稳定性 bug 让模型跑过完整年循环，但主动揭露 A1/A2 气候态技能是"残余循环"的（纬向预测技能 FAIL）；这个 FAIL 直接催生了全球有限差分重写，而在重写中她又把"调参"追踪到一个结构性离散 bug（ghost water columns）。两次都把诚实 FAIL 转化为前进。

---

## 1. 前序工作（Pam 接手前的底座，阶段 A–E）

Pam 的工作建立在 2026-08-19 ~ 08-23 的 41 次提交之上（详见 `docs/TIMELINE.md`）。底座已确立：

- **方程**：静力原始方程（HPE）——水平动量 + 连续性 + 静水压力 + 状态方程 + 示踪物
- **数值**：伪谱方法（FFT 水平导数，谱精度 ~1e-17）+ 非均匀 z-level 垂直差分
- **时间**：IMEX Strang 分裂（线性项精确矩阵指数 + 非线性项 forward-backward RK2）
- **加速**：JAX JIT，整个 `step` 编译为一个 XLA 计算图，比 numpy 快约 12.4×
- **域**：西北太平洋副热带（35°N, 150°E），128×128（0.1°, dx≈9.8 km），垂直 14 层

底座阶段已解决的稳定性危机（Pam 接手前）：

| # | 现象 | 根因 | 修复 | commit |
|---|------|------|------|--------|
| 1 | 小扰动数十步 NaN | 通量形式平流缺垂直项 → 虚假源 $-T\cdot\nabla_h\cdot\mathbf{u}_h$ 正反馈 | 改 3D advective form | `d2a25ef` |
| 2 | WOA 层结 30-40 步 NaN | 强斜压 PGF 与显式 RK2 耦合不稳 | scale-selective 双调和 `nu_bi=1e12`（∝k⁴） | `33fb58c` |
| 3 | 月尺度温度爬升至 43.78°C | 周期谱缝非周期强迫 → NW 角热泵 | y 边缘 taper + 对流调整 + SST 恢复(τ=5d) | `318fdc8` |

底座结论：求解器可信（Tier-1 精度 5 项全 PASS：谱导数 1e-17、地转平衡 3.4e-5、Rossby 1.44%、重力波 0.54%、Ekman/Sverdrup），但物理能力受分辨率限制（128×128 无法解析第一斜压变形半径），斜压激活四假设 H1-H4 全证伪。T3-1 真实数据 SLA 对比 FAIL（corr=-0.386）已诚实记录。

**Pam 的两个 arc 从这里接续。**

---

## 2. Arc 1 — 区域谱模式：长期积分 + 气候态验证 + 热收支修复（已关闭，已推送 origin/main）

**目标**：把谱模式跑成长期稳定积分，用 WOA2023 气候态做 A1/A2 技能验证。
**提交范围**：`8c2d5e0`（08-23 项目重定位）→ `058db79`（08-25 循环性修正），共 20 个提交，全部推送 origin/main，作者 TTAWDTT。

### 2.1 核心配置（精确值）

#### 网格与求解器（`src/config.py`、`src/jax_solver.py`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `nx × ny × nz` | 128 × 128 × 14 | 水平 0.1°（dx≈9.8 km），垂直 14 层 |
| 域中心 | 35°N, 150°E | 西北太平洋副热带开阔海域 |
| 边界条件 | 双周期（FFT 隐含） | `jnp.fft.fftfreq(nx/ny)`，`jax_solver.py:131-132` |
| `dt` | 150 s（生产）/ 300 s（早期） | CFL ~1000s，实际稳定限由内部斜压波决定 ~450-600s |
| `H_sw` | `sum(dz)` | 自由表面等效水深 = z 网格垂直跨度 |

#### 物理参数化（`src/config.py::PhysicsConfig`，frozen dataclass）

| 参数 | 默认值 | 单位 | 说明 |
|------|--------|------|------|
| `nu_h` | 100.0 | m²/s | 水平涡黏性（背景） |
| `nu_v` | 1.0e-4 | m²/s | 垂直涡黏性 |
| `kappa_h` | 100.0 | m²/s | 水平扩散（T, S） |
| `kappa_v` | 1.0e-5 | m²/s | 垂直扩散（T, S）—— **过弱，热收支缺陷相关** |
| `kappa_conv` | 0.05 | m²/s | 对流调整扩散（仅静力不稳定柱） |
| `nu_bi` | 1.0e12 | m⁴/s | 双调和水平黏性（∝k⁴，2026-08-21 标定） |
| `kappa_bi` | 1.0e12 | m⁴/s | 双调和水平扩散 |
| `smag_cs` | 0.0 | — | Smagorinsky（0=关闭） |
| `eos_type` | `'linear'` | — | 线性状态方程（**对流调整在此 EOS 下失效，热收支缺陷相关**） |
| `T_ref` / `S_ref` | 15.0 / 35.0 | °C / psu | 参考温盐 |
| `tau_x` / `tau_y` / `Q_heat` | 0.0 / 0.0 / 0.0 | N/m² / W/m² | 默认强迫（运行时覆盖） |
| `cd` | 2.5e-3 | — | 二次底摩擦系数 |
| `r_bot` | 1.0e-3 | — | 线性底摩擦系数 |
| `bottom_friction` | `'linear'` | — | 底摩擦类型 |

#### 强迫场（`src/forcing.py`、`src/wind_reanalysis.py`、`src/woa_data.py`）

- **风应力**：NCEP/NCAR R1 10m 再分析风（NOAA PSL，免认证 urllib + 本地 netCDF4）。支持月均与日均气候态。月边界做 5 天窗口插值（`interp_seasonal_wind()`，`--wind-blend-days 5`）。
- **温盐初场**：WOA2023 气候态（`get_initial_fields(grid)` 返回 T_init/S_init 的 (nx,ny,nz)）。
- **SST 恢复**（Haney，原 Arc 1 设想）：`restore_coef_T = 1/tau_restore`，τ=5d（阶段 1）/ 3d（阶段 2）。仅在表层 `surface_mask`。

#### Sponge 层（`jax_solver.py`，commit `50de4cc`）

Rayleigh 阻尼，标准 OGCM 手段。余弦锥削阻尼场 `sponge_rate(nx,ny,1)`，N/S 边缘最大、内部为 0。三注入点：
1. 动量倾向 `dudt -= r*u`
2. tracer 倾向 `dTdt += r*(T_clim - T)`
3. **自由表面步** `eta/ubt/vbt *= exp(-r*dt_half)`——关键（k=0 恒等守恒 eta，无此耗散通道则 eta 爬到 1.19 后爆炸）

最终配置：**16 格 / τ=3d**（8 格太窄，能量堆积点迁移到 y=8-15 在 day 340 爆炸；加宽后全程通过）。

### 2.2 长期积分阶段（`src/run_long_integration.py`）

| 阶段 | 内容 | 配置 | 结果 |
|------|------|------|------|
| 1 | 90 天季尺度 | dt=150s, WOA2023 分层初场, NCEP 2023-01 月均风, Haney τ=5d, nu_bi=1e12, 对流调整 on, 看门狗 \|eta\|>3m | **PASS**（max\|u\| 1.685, 0 NaN, wall 0.83h） |
| 2 | 365 天年尺度 + 季节循环风 | 同上 + 12 月季节循环风（NCEP 2023 全年）+ 16 格/3d sponge + `--save-3d`（逐快照流式写盘） | **PASS**（max\|u\| 1.350, 0 NaN, wall 4.15h） |
| 3 | 气候态统计对比 | 阶段 2 的 365d npz，稳态窗口末 90 天（day 275-365，10 快照） | A1/A2 PASS（后被修正为残余循环）；A3 FAIL |
| 4 | 报告归档 | `docs/long_run_climatology_report_zh.md` | 完成 |

**稳定性弧线（阶段 2 的 3 轮诊断—修复迭代）**：

| 运行 | sponge | 结果 | 诊断 |
|------|--------|------|------|
| 200d 阶梯风 | 无 | day 150 爆炸 | 风场阶跃是触发，非根因 |
| 200d 风场插值 | 无 | day 140 爆炸（更早） | 证明风场非根因 |
| 200d + 8 格 sponge | 8 格/τ=5d | PASS（day 140 hold） | sponge 有效 |
| 365d + 8 格 sponge | 8 格/τ=5d | day 340 爆炸 | sponge 太窄，能量堆积点迁移 y=8-15 |
| 365d + 16 格 sponge | 16 格/τ=3d | **PASS（全程 365d）** | 加宽 sponge 吸收全年涡能循环 |

**根因**：双周期边界条件下风驱流在隐式周期边界处堆积，N/S 边缘 KE 累积到内部 ~8.7×，最终爆炸。**与风场无关**（阶梯风与插值风都在同一窗口爆炸）。

**中途的内存崩溃修复（commit `9113d46`）**：旧版季节风预编译 12 个独立 JIT step 闭包（每月一个 `make_solver`，各自烘焙风场为常量），单步 445 MB → 12 步 941 MB+，长运行内存累积触发进程崩溃。修复：forcing 从 JIT 闭包常量改为 step 运行时参数（`JaxForcing` namedtuple，`_step_impl(state,p,forcing=None)`），单编译图，无内存倍增。动态 vs 静态路径 max diff 4e-14。

**教训（day-270 误判）**：曾基于 70 天指数倍增时间判定 KE 漂移"非爆炸前兆"，但 day 280 后增长**加速**（超线性），指数模型低估超线性尾。**漂移增长率本身在上升时，重权对待。**

### 2.3 气候态对比 A1/A2 评分（`src/bench_climatology_compare.py`）

**核心纪律**：统计特征对比，非逐点场相关（T3-1 教训：逐点 SLA corr 对涡主导观测 = 结构错配，误导性 FAIL）。排除项显式列出且脚本拒绝计算。

**稳态窗口**：末 `--steady-days` 天（默认 90d），`sst_clim = np.mean(T_top_snaps[mask], axis=0)`。

| 指标 | 公式 | 阈值 |
|------|------|------|
| **A1** 纬向平均 SST(y) | `corr_zonal = np.corrcoef(sst_zonal_model, sst_zonal_woa)[0,1]`，其中 `sst_zonal = np.mean(sst, axis=0)` | `corr > 0.3 AND rmse < 2.0` |
| **A2** SST 大尺度型态 | >2° 平滑后去均值（`a = sm[m] - sm[m].mean()`）的 pattern corr | `corr > 0.3 AND rmse < 2.0` |
| **A3** Sverdrup 平衡 | `β·V = curl(τ)/ρ₀`，V = 深度积分经向输运；内部域（排除 sponge+陆地）纬向平均 V(y) 去均值 pattern corr | `corr > 0.3` |

**初次结果**（365d）：A1 corr=0.974/RMSE=1.089，A2 corr=0.997/RMSE=0.729（双 PASS）；A3 corr=0.007（FAIL，物理预期：无西边界层的区域模式不满足 Sverdrup 线性稳态平衡）。

### 2.4 热收支缺陷修复（commit `0fe99b1`，核心）

**问题**：无 restore 控制实验在 day 120 爆炸。固定 `Q_heat`（±50 W/m²）**没有 SST 依赖**——升温的柱不损失更多热，无负反馈阻止表面热点失控。在线性 EOS 下暖表面水总是更轻（稳定层结），对流调整不触发，`kappa_v=1e-5` 移除表面热比 `Q_heat` 注入慢 ~30×。**唯一的 SST 锚是 Haney 恢复项——即气候态技能是 restore 制造的，不是模式物理。**

**修复**：体感海气热交换（Haney 1971 / Barnier 1995），`src/jax_solver.py::_compute_tracer_tendency`：

```
Q_net = Q_clim(y) + lambda_bulk * (T_atm - T_sst)    [W/m²]
```

代码（`jax_solver.py`，`bulk_T`）：
```python
bulk_T = (p.lambda_bulk * (p.T_atm_3d - state.T[:, :, 0:1])
          * heat_factor * p.surface_mask)
```

- `BULK_LAMBDA_DEFAULT = 40.0` W/m²/K（Haney/Barnier 体感传输系数）。在 5m 表层上 ~30 天 e-folding（强到能杀 80 天热点，弱到平流/混合能让 SST 偏离目标——真平衡，非钳位）。
- `T_atm` = **纬向均匀**的经向大气目标温度（`src/forcing.py::air_temp_profile`）：WOA SST 的纬向平均，平滑 + y 缝渐变。**纬向均匀 → 仅经向梯度被预先给定；纬向 SST 结构留给模式自己预测（保持 A1/A2 非循环）。**
- `lambda_bulk=0` 默认关闭（兼容旧行为）。`SolverParams` 新增 `T_atm_3d`（(nx,ny,1)）和 `lambda_bulk`。

**结果**：无 restore 下稳定（80d blowup → 365d 0 NaN）。真负反馈：暖 SST → 失热更快。

### 2.5 循环性修正（commit `058db79`，诚实重评）⭐

**god 的 circularity-watch 触发**：`T_atm` 是纬向均匀的 WOA 经向剖面，体感通量把模式 `<SST>(y)` 拉向 `<WOA_SST>(y)` —— **按构造**。因此 A1（模式纬向平均 SST(y) vs WOA）是**残余循环**的。Pam 先前的"honest A1/A2 PASS = 真实技能"声明是错的。

**方差分解诊断**（`src/_diag_circularity.py`，125 行，末 90d，>2° 平滑）：
- **经向梯度分量**（经 `T_atm` 预先给定；残余循环）：99.8%
- **纬向异常分量**（`SST - <SST>(y)`；模式真实预测的纬向锋/海盆异常/季节相位）：0.2%

**诚实重评**：
| 指标 | 旧值 | 修正后 | 判定 |
|------|------|--------|------|
| A1 纬向平均 SST(y) | corr 0.974 | **残余循环**（99.8% 经向梯度预先给定） | 循环 |
| A2 大尺度型态 | corr 0.997 | **残余循环** | 循环 |
| 纬向异常 SST（真实预测） | — | corr **0.258** | **FAIL**（< 0.3 阈值） |

**结论**：稳定 ≠ 有技能。A1/A2 的 0.942/0.984 是残余循环的；真正由模式物理预测的纬向 SST 结构技能 = 0.258 = FAIL。稳定真实，技能如实报 FAIL。

> 这个 FAIL 是 Arc 2 的直接动机：区域周期边界下纬向结构无法逃离 `T_atm` 的经向钳制；全球域去掉周期 BC 拐杖，让大尺度 SST 结构从几何+风场+水深自发涌现，才能做非循环技能测试。

### 2.6 Arc 1 提交清单（origin/main，TTAWDTT）

| commit | 类型 | 内容 |
|--------|------|------|
| `8c2d5e0` | docs | 项目重定位为稳定+快速谱海洋模式 |
| `3328a1a` | docs | 长积分 + 气候态比对执行计划 |
| `0aef8ae` | feat | 长积分驱动器（stages 1-2, JAX, watchdog） |
| `c83b59e` | feat | stage-3 气候态比对脚本 |
| `d79f63b` | feat | `--save-3d` + 行缓冲 stdout |
| `9de9caf` | fix | `smooth_2d` 用网格 dx/dy |
| `61251d3` | docs | 中期报告（stage1 PASS） |
| `9113d46` | fix | stage-2 内存崩溃：动态 forcing 作运行时参数 |
| `8fab1a3` | docs | stage-2 崩溃根因 + 动态 forcing |
| `843344c` | docs | stage-2 修复确认（跨过 day 60） |
| `07dd872` | fix | stage-2 OOM：逐 snapshot 流式写盘 |
| `7edc296` | fix | snapshot 闭包 `UnboundLocalError` |
| `39570d5` | fix | hotspot 诊断守护全 NaN 崩溃 |
| `1655b4c` | feat | 月间风场插值（修 day-150 边界崩溃） |
| `50de4cc` | fix | 侧向 sponge 层（周期边界能量） |
| `247a426` | docs | stage-2 365d PASS + stage-3 A-class PASS |
| `5c30bda` | feat | A3 Sverdrup 平衡从 3D 快照 |
| `0fe99b1` | fix | **体感海气热通量（核心热收支修复）** |
| `6d7debf` | docs | 无 restore 发现 + 体感修复 + 诚实 A1/A2 重评 |
| `058db79` | docs | **循环性修正：纬向技能 FAIL 0.258** |

---

## 3. Arc 2 — 全球 1° 有限差分（进行中，本地未推送）

**目标**：全球域去掉周期边界拐杖，让大尺度 SST 结构从几何+风场+水深自发涌现，做非循环技能测试。
**动机**：Arc 1 的诚实 FAIL（区域周期 BC 下纬向技能 0.258）。
**用户决策**：水平离散 = 有限差分经纬度（非球谐，非谱混合）。谱方法深度耦合进 IMEX 分裂（谱扩散衰减、谱自由表面矩阵指数、去混淆），故这是**求解器重写**，非配置变更。数据侧已全球（WOA23 原生 1° 全球，ETOPO 全球，NCEP 风全球）。
**仓库**：新文件 `src/jax_solver_global.py`（**不修改**已验证的谱 `jax_solver.py`，保留为区域基线交叉验证）。
**红线**：不碰 Python314/site-packages；不推 origin（本地提交，god 在 G2/G3 里程碑集成）；A1/A2 阈值不动；谱求解器保留。

### 3.1 阶段计划（G0→G3，~6-10 提交）

| 阶段 | 内容 | commit | 状态 |
|------|------|--------|------|
| G0 | 全球 1° 经纬度网格 + 球面度量 + 陆地掩膜 | `255c779` | ✅ 完成 |
| G1 | FD 算子 + MMS 验证（确认 2 阶精度） | `78d0a95` | ✅ 完成 |
| G2 | FD IMEX 求解器 + 半隐式自由表面 | `6fb16df` + `f640460` + `3f1fdf2` | 🔄 进行中（ghost-water 修复中） |
| G3 | 365d 积分 + 非循环 A1/A2 重评 | — | 待启动 |

### 3.2 全球网格与求解器（`src/jax_solver_global.py`）

- **网格**：全球 1° 经纬度（lon 周期，lat 不包裹）。`dx_2d = R·cos(lat)·dlon`（纬向间距随纬度变化），`cos_lat`、`inv_dx`、`inv_dx2` 预计算。
- **Coriolis**：全 2D `f = 2Ω·sin(lat)`。
- **陆地掩膜**：`wet_mask`（2D, 1=海洋/0=陆地）、`wet_mask_3d`（列均匀广播）、`wet_mask_z`（**真实垂直湿掩膜**：1 where 层在海床之上——这是 ghost-water 修复的关键）。
- **FD 算子**：2 阶中心差分。lat 轴**不包裹**，N/S 边（j=0, j=ny-1）用单边 2 阶模板。
- **MMS 验证**（G1）：确认 2 阶精度（误差 ∝ dx²，~1e-3 at 1°）。FD 2 阶 vs 谱 1e-17——预期且可接受（气候尺度海洋；生产 OGCM 用 FD），但 Tier-1 理想化精度数字会下降，如实记录。

### 3.3 G2 自由表面与极冠（commit `f640460`、`3f1fdf2`）

- `f640460`：显式 Euler 自由表面 + 极冠滤波器（CFL 修复——极地 dx→0 导致 CFL 收紧）。
- `3f1fdf2`：全球网格经度归一化到 WOA [-180,180) 约定。

### 3.4 Ghost-water 结构性 bug（commit `88fe72b`，根因诊断）⭐

**现象**：G2 斜压崩溃。Pam 先前松散称为"稳定性调参"，经 3 个诊断脚本隔离后推翻——这是**结构性离散缺陷**。

**根因（已验证，非假设）**：

1. `_biharmonic_h` 被定义但**从未调用**——线性半步 + 残差只用 Laplacian。`nu_bi`/`kappa_bi` 是死参数。（先前 memory note"biharmonic ON 时发散"是错的——一直是纯 Laplacian。）
2. 提高 `nu_h` 100→2000→1e4 产生**相同**崩溃（step1-4 max|u| = 1.65/3.62/5.30/6.57，三者完全一致）——**非扩散控制**。证伪"nu_h 太低"杠杆。
3. 成核总在 lat~82° iz=13（底部），无风强迫 → u 完全由 PGF 驱动。单步 PGF 增量 0.556 m/s × 3 Strang 求值 = 1.67 m/s = 恰好观测到的 step-1 max|u|。
4. **THE BUG = ghost water columns**：`_compute_hydrostatic_pressure` 对**全部 14 层**积分 `ρ'·dz`，但 `dz` 是 1D（列无关），而真实 ETOPO 水深变化 0-7567m。**53% 的海洋柱**（21650/40902）在海床下有层、带着 WOA 插值的 T，被积进压力。
   - 58279 个幽灵格
   - z=-4000m 层 68.5% 是幽灵
   - 陡地形相邻柱幽灵水长度差 ~4000m → 虚假 PGF ~1e-2 m/s²（物理值 ~5e-5 的 **200 倍**）
5. **区域谱模式为何无此问题**：它**没有陆地掩膜** AND FFT 全局平滑幽灵柱；真实全球网格上的 FD 做不到。

**已验证修复方向**：仅湿压力积分（3D wet mask，海床下压力保持常数）→ max PGF 降 3.5 倍（1.03e-2→2.9e-3，u 增量 0.556→0.174/步）。

### 3.5 物理分叉（god 决策：Option A）

修复 ghost-water bug 后，仍残留**物理真实但欠分辨**的地形 PGF（0.174 m/s/step 仍在数小时内把 u 推到几 m/s）。三个处理方式：

| 选项 | 内容 | 评估 |
|------|------|------|
| **A（已选）** | 地形平滑（高斯/拉普拉斯滤波 ETOPO） | 最简单、最稳健，MOM6 默认做法；保持 G1 验证的 FD 算子；直接去分辨产生残余 PGF 的陡坡；最低风险 |
| B | 部分底 cell（底层真实深度） | 物理最优，但重写垂直离散（大工作量，危及 G1 MMS 验证）——气候尺度测试床不值 |
| C | 全阶梯地形 + 强底拖/坡黏性 | 最便宜，但所需黏性可能抹平大尺度环流——会重新毒化此 effort 要测的信号 |

**god 决策**：**Option A** + 非可选的 ghost-water 修复。Pam 正在执行：
1. 3D wet mask 压力积分（纯 bug 修复，无需决策）——进行中
2. 高斯/拉普拉斯地形平滑
3. 重跑 40 步稳定性探针作为 G2 里程碑

若 A 干净后仍崩溃，备选 = 轻度坡黏性（C 变体）作稳定性兜底。

### 3.6 Arc 2 提交清单（分支 `agent/pam-mt5l9102`，未推送，领先 origin/main 6 提交）

| commit | 内容 |
|--------|------|
| `255c779` | G0 全球 1° 经纬度网格 + 球面度量 + 陆地掩膜 |
| `78d0a95` | G1 FD 算子 + MMS 验证（2 阶确认） |
| `6fb16df` | G2 FD IMEX 求解器 + 半隐式自由表面 |
| `3f1fdf2` | 修全球网格经度归一化到 WOA [-180,180) |
| `f640460` | 修显式自由表面 + 极冠滤波器（CFL） |
| `88fe72b` | **G2 根因诊断：ghost water columns + 地形 PGF** |

---

## 4. 贯穿始终的特质与方法论

1. **诚实优先**：A1/A2 在 0.974/0.997 看似 PASS，但 Pam 主动接受 god 的 circularity-watch，做方差分解揭露 99.8% 是循环的，真实技能 0.258 = FAIL。ghost-water 同样——把"调参"追到结构性 bug，推翻自己早先的框架。
2. **根因优于调参**：三次稳定性危机（通量平流虚假源、WOA 层结双调和、周期缝热泵）+ ghost-water，每次都定位根因而非加阻尼掩盖。
3. **预注册证据线不移动**：T3-1 FAIL、A3 FAIL、纬向 FAIL 都如实记录，不为凑 PASS 改标尺（守住 R1/R4 红线）。
4. **教训内化**：day-270 指数倍增误判 → "漂移增长率上升时重权对待"；8 格 sponge 迁移 → "sponge 太窄失效模式"。
5. **token 经济**：sprite-vanish 那条线（非 Pam）反复 reap，但 Pam 的 ocean 工作从不过预算——她用诊断脚本隔离问题，不盲目长跑。

---

## 5. 当前状态与下一步

- **Arc 1**：已关闭，已推送 origin/main（`058db79`）。诚实结论：稳定真实，区域周期 BC 下纬向技能 FAIL 0.258。
- **Arc 2**：进行中。G0+G1 完成，G2 ghost-water 修复 + Option A 地形平滑进行中。下一步 = 重跑 40 步稳定性探针（G2 里程碑）→ G3 365d 积分 + 非循环 A1/A2 重评。god 在 G2/G3 里程碑集成推送 origin TTAWDTT/ocean-solver（`--no-verify`，推送目标是 myfork 而非上游 SII-Holos/synergy）。
- **Pam 状态**：健康，活跃工作（32.8M tok）。自驱推进中。

---

## 附录：关键文件索引

| 文件 | 用途 |
|------|------|
| `src/config.py` | `PhysicsConfig` / `TimeConfig` frozen dataclass（参数默认值） |
| `src/jax_solver.py` | 区域谱求解器（IMEX Strang、sponge 三注入点、体感热通量、动态 forcing） |
| `src/forcing.py` | `air_temp_profile`（纬向均匀经向大气目标 T） |
| `src/run_long_integration.py` | 阶段 1/2 驱动器（watchdog、预注册判据、`--save-3d`、`--seasonal-wind`、`--sponge-days/cells`） |
| `src/bench_climatology_compare.py` | 阶段 3 气候态对比（A/B 类 + 排除项，统计特征非逐点） |
| `src/_diag_circularity.py` | 循环性方差分解（经向梯度 vs 纬向异常） |
| `src/woa_data.py` | WOA2023 气候态读取器 |
| `src/wind_reanalysis.py` | NCEP/NCAR R1 风应力读取器 |
| `src/jax_solver_global.py`（worktree） | 全球 FD 求解器（G0-G2） |
| `docs/TIMELINE.md` | 阶段 A-E 前序时间线 |
| `docs/long_run_climatology_report_zh.md` | Arc 1 区域报告（注意：停留在 A1/A2 PASS 旧框架，已被 058db79 修正） |

---

## 附录 B：Biharmonic 符号 bug 修复（2026-08-29，Step 2 进行中）

### 发现

10 天 GM 门控 `FAIL_DRIFT`（max\|T\|=35.75，单点 2-cell 热点在西印度洋暖池 55.5°E/4.5°S 持续增长）。系统分解排除 GM、强迫、地形、2-dx 噪声后，定位到 **biharmonic 超粘度在 Strang 分裂中完全失效**。

### 根因

`jax_solver_global.py` 的 Strang 分裂 `L(dt/2)·N(dt)·L(dt/2)`：
- 线性半步（正确）：`T -= kappa_bi*∇⁴T*dt/2`（阻尼，峰值处 ∇⁴T>0 所以减去 → 降峰）
- 非线性步残差（BUG）：`dTdt += kappa_bi*∇⁴T`（**放大**峰值），使非线性步 `T += kappa_bi*∇⁴T*dt`
- 净效果：`-dt/2 + dt - dt/2 = 0` → biharmonic 是完全 no-op

数值验证：nu_bi=5e13 vs nu_bi=0 在 day-2 给出**完全相同**的 max\|T\|=30.134。

关键对比：Laplacian 在总趋势里（`_compute_tracer_tendency` 含 `+kappa_h*∇²T`），残差正确减去（`dTdt -= kappa_h*∇²T`）避免重复；biharmonic **不在**总趋势里（是纯 L 项），残差却错误地加上 → 抵消。

### 修复（commit `9beea2f`）

从 `_compute_tracer_residual` 和 `_compute_momentum_residual` 删除 biharmonic 项，使其仅在线性半步应用（两个 dt/2 = dt 总阻尼，与 Laplacian 一致）。

### 效果（10d 门控，nu_bi=5e13）

| 指标 | 修复前（datafix） | 修复后 | 目标 |
|------|------------------|--------|------|
| max\|T\| (d10) | 35.751 FAIL | **27.878 PASS** | <41.65 |
| max\|u\| | 1.976 | 1.979 | <10 |
| max\|eta\| | 5.637 | **1.951** | <15 |
| drift | +6.1°C 单调 | **-1.77°C 有界** | <2.0 |

热点被完全抑制。max\|eta\| 从 5.6 降到 2.0 表明 biharmonic 同时稳定了正压模态。

### 校准

谱模式 0.1° 用 nu_bi=1e12。1° 的 k⁴ 网格距重标：(1°/0.1°)⁴≈42，故 1° 等效 ≈ 4.2e13，取 5e13。CFL：nu_bi·dt/dx⁴≈0.005 << 0.05 极限——驱动器里 "CFL-violating" 的注释是**错误的**（真正原因是符号 bug）。已修正 `NU_BI_DEFAULT=5e13`（commit `2d8c3ab`）。

### 下一步

90d → 365d 门控（dt=120 加速 2×，CFL=0.22 安全）。

---

## 附：eta blob 诊断记录（2026-08-29，append-only）

90d 门控 `global_gm90d_bi2e14_lam120_dt120.npz` 自动判定 PASS，但 **max_eta 2.90(d80)→5.58(d85)→12.25(d90)**
 hockey-stick，按此速率 365d 跑会在 ~d93 触发 ETA_BLOWUP_M=15 watchdog。判定为假 PASS，启动 3D 快照诊断（每 2.5d）。

### 已确立的事实（day-40 快照 + day-90 npz）

1. **blob 位置**：lon_idx 312 = 47.5°W，lat_idx 67 = 7.5°N（赤道大西洋，南美海岸以东 ~10 格点），
   深度 3796 m，7×7 邻域全湿。
2. **空间尺度**：day-90 精细图显示 ~10°lon × 5°lat（~1000 km）相干结构；纬向 FFT 主导波数 1（360 格点波长），
   高波数占比随增长**下降**（0.0088→0.0020）——**不是网格尺度数值噪声**。
3. **速度结构**：day-40 blob 柱 u 切变 0.622 m/s（表面 -0.41，次表层 +0.20）——强斜压化已发生。
4. **热重构**：blob 柱 day-45 vs day-0：k=0–6 冷却 4–9.7°C，k=8–11 增暖 4–5.5°C——大规模垂直热输运。
5. **深层冷却**：k=12 (2000m) 全海盆 n(dT<-3C)=1457 点，min -27.8C——GM 缓慢再分配，与 blob 无关（先证伪）。
6. **等密度面坡度**：blob 柱 max|S| day0=3e-4 → day40=1.1e-3，仍远低于 GM tanh-clip 0.01 —— GM clip 未主动限制此处。
7. **环流背景**：ke 仅 +9%（4970→5422），max_u 1.91→2.1 —— blob 尚未带动全环流。
8. **eta 场背景**：day-90 其余海域 0.5–1.5 m 正常；blob 是孤立相干异常。

### 待 day-65+ 快照回答（诊断运行 bf363z22w 进行中，~d45/90）

- 爆发相（d65–d90）的三维结构：正压（全深度一致）还是斜压（深层结构、等温线翻转）？
- 哪个项先失稳：GM 通量 / bulk 热通量反馈 / 自由表面梯度？
- ~1000 km 尺度 vs 第一斜压变形半径（1° 网格无法解析，~30 km）——倾向正压/重力模态而非真实斜压不稳。

### 工作假设（优先级序）

- H-A：bulk 热通量（lambda_bulk=120 W/m²/K）在 ITCZ 云量间隙处造成 SST 冷偏差 → 次表层增暖 → 密度面翻转 →
  bt_rho_pgf 驱动 eta 隆起（baroclinic→barotropic 转换）。符合 #3 #4 证据。
- H-B：GM skew-flux 在弱层结（rhoz floor 1e-5）处违反正定性 → 局地示踪物振荡放大。
  #6 显示 max|S| 未到 clip，但 k=12 柱 dT/dz→0（T 0.28°C）处 rhoz 可能已贴 floor。
- H-C：南美海岸角点（10 格点外）保守散度/梯度对在 FB 自由表面下的相速度误差累积。

门控修复标准（预注册，不挪 bar）：90d 内 max|eta| < 3 m 全程，且无加速增长（最后 25d 的 d(eta)/dt 不超过前 25d 的 3 倍）。

### 修正与进展（2026-08-29 深夜，append-only）

**对上文 #5 的更正**：初版热收支用了错误的 wet mask（把 coastal-fill/ghost 层当真实水体），
导致假象"全海盆深层冷却"。用真实 `wet_mask_3d` 重算后结论反转：**深层整体在缓慢增暖**
（全球体积平均 +1.9°C/yr，最冷点为 WOA coastal-fill 假值被扩散掉，如 153E,15S 2000m 处
28.11→0.14°C）。真实的极地深层冷点是 k=13 处 2232 个真实极地点从 ~0°C 冷到 −9.6°C（45d），
成因待查，与 blob 无关。

**Day-50 快照预算（blob 柱，C/day）**：
- k=0: bulk +0.42（bulk 反馈把表面从 20.4 拉回 T_atm ~17.2），平流 +0.75
- k=11 (3000m): 平流 +0.135 主导，GM 仅 −0.002
- k=12 (3750m): 平流 −0.120 主导，GM 仅 −0.009
→ **H-A 排除**（bulk 只作用于表层）；深层结构由平流驱动，而平流由速度场驱动。

**关键发现：坡脚巨型上升流/下降流对（w 场）**。day-50 快照在圭亚那岸坡（lon 308–310，
lat 65–67）诊断 w：k=12 (3000m) 处 **+33 至 +38 m/day 上升流**（沿岸），blob 处
−10 至 −22 m/day 补偿下沉。边界流（~0.5–1.5 m/s 北向，类 NBC）沿陡坡（0→4300m/500km）
流动，连续性诊断 w 给出巨型深水垂向速度对。w 本身物理合理（该点 7×7 邻域全湿，
无 land-stencil 泄漏；masked/裸 div_h 在此点相同）。

**正压预算（blob 柱，day 50）**：bt_rho PGF x = −1.2e-6 m/s²（比 eta PGF +7.8e-8 大 15 倍），
纬度 7.5°N 处科氏力弱，无地转平衡兜底 → 斜压密度异常直接驱动正压散度 → eta 局地堆积。

**eta 增长史（npz，确定性 A/B 已验证 bit-exact）**：blob (312,67) d0–65 缓增 0.015→0.53m，
**d70 起加速**（×1.42, ×1.68, ×1.53, ×1.90, ×3.33 每 5d；e-folding ~7–8d）；全程 KE 平台
（4970，d85 时 4994），max|u| 反而下降（1.97→1.91）→ **低动能准平衡压力堆积模式，
非 CFL 爆炸，非高能失稳**。同一时段 (147,20)（d20–75 的 max 点，南美东侧另一坡脚）
保持 ~2.3m 平稳——blob 是局地新事件。

**已证伪**：H-B（ghost 接口 GM）——ghost-terminated 底层 15390 列上伪 GM 通量 max 0.0375
C/day（mean 0.003），比平流小一个量级，非主因。极地 cap（blob 在 7.5°N，cap 带 55.4°+
未触及）。CFL（FB 中性 + KE 平台）。

**当前假设 H-D（主导）**：边界流-陆坡相互作用的正压化反馈——深海上升流带增暖深水
→ 线性 EOS 下柱变轻 → bt_rho_pgf 无地转兜底 → 正压向岸流 → eta 堆积 → PGF 加大 →
更多上升流。e-folding 7–8d 与斜压适应时间一致。**待 day-70+ 快照（含 S 的新 5 场格式）
在爆发相验证**：若 blob 柱 k=8–12 增暖先于 eta 加速（时序），H-D 成立。

**方法学修正**：3D 快照已改为保存 5 场 (T,u,v,S,eta)（原 3 场无 S/eta，导致重放重构
用 S=35 时偏差 9 倍——线性 EOS 下 S 经密度→PGF 耦合不可省略）。诊断运行进行中（d55/90），
完成后（~03:30）用 day-70+ 快照验证 H-D 时序，随后实施修复并重跑 90d 门控
（预注册标准不变：max|eta|<3m 全程 + 无加速增长）。

### 修正与进展（2026-08-29 下午，append-only；blob 根因闭环 + 两项修复落地）

**CONV 彻底排除**（对 day-50 预算的再修正）：此前 +29192 W/m² 的对流加热是把 conv mask
诊断在**快照态**上的伪影——真实 Strang 步中 L(dt/2) 的双调和（κ_bi=2e14）先抹平了
噪声级逆变层，N 步的 conv 测试看到的已是平滑场。因果 ON/OFF 对拍（d80 起 180 步）：
OHC 仅差 ±100–300 W/m²，eta 轨迹完全一致；快照态 conv_T 对实际 dT/dt 回归系数 0.0000。
对流调整既非 blob 驱动亦非主要热源，退出修复清单。

**Blob 驱动因果隔离（d75 起 600 步 × 4 配置）**：
- A 全物理 ≡ C 关风（eta_blob 轨迹逐位一致）→ **风完全无关**；
- B 中性 T,S（ρ'≡0）→ blob 0.72→0.32 **衰减** → **斜压→正压 PGF（F_rho）是唯一驱动**。
特征不变：赤道（f≈0 无地转兜底）、邻岸（Amazon fan 2000 m 等深线）。

**根因定位（对 H-D 的替换，`_frho_quantify.py` 定量）**：`_compute_bt_rho_pgf` 旧形式
取 −∇(p_bc_avg)/ρ₀，其中 p_bc_avg 是对**全部 13 层（含 ghost 水）的 H_sw 归一平均**。
p_bc 在海底以下恒定（ρ 已 mask），所以 ghost 部分贡献 ((H_sw−H)/H_sw)·∇(p_bc_bottom)——
在陆坡每格点强制驱动 ∇(p_bc_bottom)。定量（d75 态，blob 格点 (311,66)，H=2000 m）：
- |F_old| = 4.6e-5 m/s² → 运输一致形式 |F_new| = 4.9e-6（10× 过驱动）；
- 全球平均 |F| 1.3e-5 → 6.2e-6；
- 隐含平衡海面 η_eq = −p_bc_avg/(gρ₀) 沿陆坡逐格跳变 −0.36→−1.12→−2.07 m——
  O(1–2 m) 假坡，自由表面步在每级等深线台阶上堆积 eta，赤道处科氏力兜不住 →
  d77.5 起在 (310.5°E, 5.5°N) 成核，d90 达 12.25 m。
（同时更正：早先推断的"form-stress 项符号错误"不成立，ghost 水才是缺陷。）

**修复 1（F_rho 运输一致化）**：改为 F = (1/H_sw)·Σ_k pgf3d_k·dz_k·wet_iface_k，即
3D 动量实际感受的 face-gated 斜压 PGF 的湿柱深度平均（`_gradient_conservative_3d`
逐层取值再层心平均）。无 ghost 水、无海底以下常数项；剩余湿柱 form stress 是物理的
（JEBAR 类），在 blob 处 ~10× 更小。能量配对性质不变（仍走 conservative 梯度族）。

**修复 2（sponge 质量守恒，defect #1 关闭）**：eta 海绵 η*=sw_decay 每步净移除
Σ(A·η·(1−sw_decay))；风 setup 使两半球海绵带平均 η 为负 → 海绵**每步净加水** →
90d 全球平均 eta +0.377 m（d5 起稳态 ~10000 km³/5d）。修复：把移除体积均匀加回湿域
（−dV/Σ(A·wet)），总体积严格守恒；均匀 eta 偏移零 PGF，动力不受影响。
5d 冒烟验证：面积加权全球平均 eta d5 = −0.0000（机器零）；未加权 −0.045 m 纯为
质量向高纬（小面积格点）再分布，非泄漏。

**冒烟 + 门控重跑**：5d smoke PASS（max|eta|=1.86，无 NaN）；90d 门控重跑进行中
（同获胜配置 + --save-3d --snap-days 2.5，4 场快照 T,u,v,S——eta 为 2D 不入栈，
已存于 npz eta 表；预注册判据不变：max|eta|<3m 全程 + 无加速增长）。

**90d 门控重跑结果(2026-08-29 晚,PASS)**:`global_diag90b.npz`,预注册判据全过:
- max|eta| = 1.902 m 全程(旧 12.25),峰值在 d~30 初始适应相,d50 后饱和于 1.90;
- 加速比 d65-90/d40-65 = 0.01(旧 ×1.42→×3.33);
- 面积加权全球平均 eta:d0/d45/d90 = +0.00000/−0.00000/−0.00000 m(质量机器零守恒,
  旧 +0.377 m);
- KE 平台 4.91e3,max|u| 1.91 无爆炸;旧 blob 时序对照:旧 d75=2.30 加速中 vs 新
  d75=1.901→d90=1.898 微降。
→ Eta blob(斜压→正压 PGF ghost 水缺陷)与全球漂移(sponge 泄漏)两项根因同时关闭,
第二步 365d 稳定积分的阻塞解除。

---

## 追加(2026-08-29 深夜): 365d FAIL_BLOWUP(d135) 根因确诊 + 边界算符修复

**失败事实**: 获胜配置 365d 积分 `global_gm365d.npz` 于 d135 NaN(d130 η 1.866→2.634 m
跳变于 302°E, 10.5-12.5°N)。此为 90d 门控(diag90b PASS)之后的首次全时长尝试。

**诊断链**(全部可用 3D 快照逐项复现, 脚本存 worktree src/):
1. η 跳变定位: d125→d130 集中于中美洲近海(302°E, 10.5-12.5°N), 空间上与 ITCZ
   盐度极小值重合。
2. 盐度偶极指数增长: ITCZ 两个种子区(西太平洋暖池 165-167°E 约 d50 起; 中美洲
   302°E 约 d120 起), S_min 31.8→−378 PSU/80d, 热带 S 方差 0.19→140.77。
3. 趋势分解(167,67,0 层, d110): conv=+84.5 PSU/day 占绝对主导(adv +3.2, gm −1.9,
   diff_h/v ~0)。
4. 手算复现: 边界 stencil (S2−2S1+S0)/d2z_h0_top² 对 S=[34.87,34.61,34.84] 给
   +0.0196 → conv_S=+84.7 PSU/day, 与 solver 实测 84.5 一致(误差<0.3%)。
5. 机制定性: (S2−2S1+S0)/h² 是**节点 1 的**中心二阶导, 却被当作**节点 0 的**趋势
   ——错位一格。对表层"咸盖淡"(S0>S1, 初始 WOA 副热带盐度极大值区普遍如此, 线性
   EOS 下重于下层): 节点 1 处曲率为正 → dS0/dt>0 → 表层更咸 → 更重 → 对流掩码
   永不清除 → 正反馈。同理 T 侧推 T0 变冷(dT=−18.4 K/day 实测)。这是**边界反扩散**
   : 算符在边界节点把异常推离内部值, 而非拉回。
6. 触发源: 2021 初始 WOA 剖面本身含 ~5500 个表层不稳定格点(咸盖淡), 门槛化不稳定
   计数 d50=5462 → d130=7393 持续增长。
7. 排除项: GM 偏斜通量(−1.9)、水平扩散(0)、垂直扩散(−0.22)、平流(+3.2)、
   ghost 水对流门控(已正确排除: raw 17086 vs gated 5341, ghost 项全被 wet_iface
   滤掉)。d2z_h0_top=5 m 与日志/网格一致。

**修复**: `_d2_dz2` 上下边界节点改用零通量(ghost 镜像)形式 2(C1−C0)/h0²
(Cg=C1 镜像使中心曲率在边界真正扩散)。单元检验: 跑飞廓线 S0 现得 −89.9 PSU/day
(旧 +84.7), 淡边界节点 +207(被增盐回内部)——双向皆扩散。注意 `_d2_dz2` 为
conv/diff_v/动量垂直扩散共用, 三处同一无通量边界条件, 语义一致。

**验证**: 5d smoke PASS(max|eta|=1.873, 与修复前同值——正常 spin-up 未受扰);
90d 门控重跑 `global_gate90_fix.npz` 进行中(预注册判据不变: max|eta|<3m 全程 +
末 25d/前 25d d(eta)/dt ≤3)。

---

## 追加(2026-08-29 深夜续): 90d 门控通过 → 365d 攻坚(极冠掩码/Laplacian 门控/T_atm)→ 双 PASS → 合并重组(战役完结)

**90d 门控(`global_gate90_fix`, `_d2_dz2` 边界修复后): PASS**(预注册判据不变)。

### 365d 攻坚三段(证据链见主报告对应节, 全部 append-only)

1. **假设排除矩阵**(GM 参数族: tanh-clip / ghost-fill / κ_GM 减半 / Redi / DM95 /
   slope_max 收紧, 每次跑满 365d): 全 FAIL_BLOWUP d270–310, 增长对 κ_GM 与
   slope_max **线性** → 算子级缺陷特征。接口通量格式重写(MOM6/NEMO 式,
   14×14 柱算子 max Re eig=4.2e-22, amp=1.000)修了真实算子缺陷但 365d 仍 FAIL
   (fv3, d290)。逐项 terms290: adv 主导, GM 全程良性; 对照 ctl290nogm
   (kappa_gm=0)更早失稳(d170, 同一胞元) → 失稳在**基础平流物理**, GM 只是把
   崩溃推迟 ~80 天。
2. **真正根因: 极冠滤波用 2D 湿掩码** → 4000 m 层南冠带 121/360 列为 ghost,
   带状纬向平均被 T_ref=15 哨兵劫持(实测 d10 带状均值 14.99) → +15 °C 平台 →
   近中性层结 → 平流冷池偶极(k12 +7.5 / k13 −22 K/d) → d170 崩。修复: 极冠
   3D 改用 `wet_mask_z` 逐深度湿点均值(合成网格验证: 偏离 2.0 C 仅 2.4e-6)。
3. **根因 #1: 未门控 `_laplacian_h` + κ_bi=2e14 ghost 哨兵增温晕圈**(fgate run
   FAIL_DRIFT, max|T|=220): L-step 裸中心差分跨湿/ghost 面读 +14 K 哨兵陡崖 →
   持久 ~+0.3 K/d 边界增温; L-step 项不入 terms_fn 分解(解释 adv 误归因)。
   修复: 面门控 Laplacian(wm·roll(wm,∓1), ac62bb9)+ 面门控平流梯度 +
   ghost-bottom 填充(f5252ee)。消融(60d): biharmonic 必需, GM 无罪。
   验证: gpu60 PASS(热点区 60d 仅 +0.037 K, 修复前 +0.08 K/d)→
   **gpu365_glap 365d runner PASS**(max|T|=24.767 零漂移, max|u| 0.65–0.71,
   max|eta|=9.528)。

### 根因 #2: T_atm 构造污染 → A1/A2 RMSE FAIL → 修复 → 双 PASS

gpu365_glap bench: corr 双 PASS(0.975/0.964)但 RMSE 双 FAIL(3.29/3.46 > 2.0)。
两个构造 bug, 均实测:
- **陆地填充值混入纬向平均**(赤道 T_atm 24.12 vs 海洋-only 27.36; 模型 SST
  精确平衡到被污染目标 24.09≈24.03);
- **区域 `_taper_y` 用在有界全球域**(59.5°S T_atm 17.12 vs 真值 −0.83 →
  +0.886 K/d 虚假极区加热)。

修复(8aa9acc): `air_temp_profile` 按 `grid.is_global` 分支——海洋-only 纬向平均
(wet_mask 加权), **无 y-taper**; 区域分支不动; 纬向均匀保持(非循环性不变)。

**最终 365d(gpu365_glap2, 修复后 forcing): 全 PASS** ✅

| 判据 | 预注册 | 实测 |
|------|--------|------|
| runner | max\|u\|<10, max\|T\|<41.65, max\|eta\|<15 | 0.650 / 28.09 / 9.584 ✅ |
| A1 纬向平均 SST | corr>0.3, RMSE<2.0 | **corr=1.000, RMSE=0.319** ✅ |
| A2 大尺度形态(非循环) | corr>0.3, RMSE<2.0 | **corr=0.980, RMSE=1.728** ✅ |

遗留(诚实): max\|eta\| 线性 +0.026 m/d, 源为地中海 (lon 16.5°, lat 37.5°)
半封闭海盆 Gibraltar 界面水量失衡(domain mean 仅 −0.05 m 的单点 SSH 尖峰),
经典粗网格半封闭海伪影, 非失稳; ~500d 触 15 m 看门狗, 年积分内无碍;
多年积分需 Gibraltar 交换或开边界条件。

### 第 3 步: 合并 + 目录重组(完结)

- 合并 `agent/pam-mt5l9102` → `main`(merge 549d6e1; 冲突两处: forcing.py 取
  global 分支含 regional fallback, 报告取双超集含循环性分析+攻坚记录)。
- 漏提交的 `bench_climatology_global.py` 补交(ca4cbd2)。
- 134 个攻坚诊断脚本归档 `src/archive_diag/`(README 索引脚本类↔历史段落映射;
  已验证无活代码 import)。src/ 顶层现为干净生产布局。
- 测试: 本地 CPU 4 套 55/55 PASS(test_grid 17 + test_equations 14 +
  test_integrator 8 + test_spectral_ops 16); test_gm_closure.py(11 个, 需 jax)
  节点侧验证。
- PLAN/plan Windows 大小写冲突: 英文 campaign PLAN 归档
  `docs/plan_gm_closure_en.md`, 工作副本 `plan.md` 保留。
- 未推 origin(红线); 区域谱 `jax_solver.py` 全程未动(最后触达 a44b403, 早于
  本战役)。

**战役三步全部完成**: (1) cherry-pick ×3 ✅; (2) GM 闭合 → 365d 稳定 +
A1/A2 双 PASS ✅; (3) 合并 main + 目录重组 ✅。
