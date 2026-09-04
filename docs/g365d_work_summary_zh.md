# 全球 1° 365 天积分工作总结 — 从全球 FD 重写到真实季节性强迫生产运行

> **日期**：2026-08-25 ~ 2026-09-04
> **范围**：从项目**初次切换到 365 天、1° 全球积分**（2026-08-25 G0 提交）起，全球有限差分模式（`jax_solver_global.py`）的全部工作：G0–G3 建设门、GM 闭合攻坚、双 PASS 气候态验证、Med eta 修复、动态季节性强迫路径、012 GPU 环境搭建与 365 天生产运行启动。
> **前置文档**：`docs/pam_work_summary_zh.md`（Pam 全程总结，含区域谱模式 arc）、`docs/progress_report_zh.md`（08-19~08-23 底座阶段）、`docs/long_run_climatology_report_zh.md`（区域 365d + GM 攻坚 append-only 报告）、`docs/repositioning_memo_zh.md`（项目定位校正）
> **本文取代**：`docs/g365d_work_summary_zh.md` 的窄版覆盖（原版本只写了 09-03~09-04 一段）。

---

## 0. 一句话定位

这条弧线的起点是一个**诚实 FAIL**：区域谱模式的 365 天积分稳定（free-run 0 NaN），但气候态验证暴露 A1/A2 是"恢复项制造"的循环结果，真正预测的纬向结构 skill = corr 0.258 < 0.3（FAIL）。诊断确认循环来自**周期边界 + `T_atm` 钳制经向梯度**的实验设计，不是可以调参修掉的缺陷——需要**真实海陆边界（有西边界层结构）、更大尺度、非循环的气候态检验**。于是 08-25 当天立项全球 FD 重写（G0），9 天后（09-03）拿到 **365d 稳定 + A1/A2 corr 1.000/0.980、RMSE 0.319/1.728 双 PASS**，09-04 把最后一块短板（季节循环风应力）补上并启动 012 GPU 生产运行。整个过程的关键词：**预注册判据不挪 bar、诚实 FAIL 驱动下一步、每个根因都有量化证据链**。

---

## 1. 起点：区域模式的循环性暴露（2026-08-23 ~ 08-25）

### 1.1 区域 365d 的成就与天花板

区域谱模式 arc（详见 pam_work_summary）在 08-24~08-25 完成：365d 季节风积分 PASS（16 格/τ=3d sponge，wall 4.15h，CPU 74.6ms/步），A1 corr=0.974 / A2 corr=0.997 vs WOA2023。表面上是 PASS，但随后一轮对照跑揭示问题：

- **无恢复对照**（只改 `--restore-days 0`）在第 120 天热力学爆炸（FAIL_BLOWUP）——证明 A1/A2 的高相关本质由 SST 恢复项（τ=5d，强热汇）制造；
- **体热通量修复**（Haney/Barnier，commit `0fe99b1`）补上 SST 负反馈后，free-run（恢复关）稳定通过 365d 全年循环，0 NaN——真实的稳定性成就；
- 但 **诚实重评分**（commit `058db79`，god 循环性监督触发）暴露：free-run 的 A1/A2 = 0.942/0.984，分解显示平滑场方差 **99.8% 是经向梯度**——而经向梯度正是 `T_atm`（纬向均匀 WOA 廓线）强迫规定的。**真正预测的纬向异常结构 skill = corr 0.258 = FAIL**。

### 1.2 为什么指向全球 FD 重写

循环性的结构性来源：区域模式双周期边界（FFT 隐含）+ sponge 补丁，纬向结构没有真实的海陆对比可以预测；且 `T_atm` 钳制经向梯度是热力学上不可回避的（海洋模式必须有 SST 负反馈才稳定）。出路是**换一个 skill 有物理来源可挣的实验设计**：全球 1° 网格（真实海陆、真实西边界流区、赤道波导），A2 检验 2D 大尺度型态——模式必须自己预测纬向异常（暖池、冷舌、西边界），这才是不循环的判据。

方向由用户拍板，G0-G3 分阶段计划（PLAN.md）当日启动。

---

## 2. G0–G3：全球 FD 求解器从零到长积分（08-25 ~ 08-29）

### 2.1 四道建设门

| 门 | 日期 | 内容 | commit |
|----|------|------|--------|
| **G0** | 08-25 | 360×170 1° lat-lon 网格 + 球面度量（dx=R·cos(lat)·dlon 随纬度变化）+ ETOPO 0.1° 块平均海陆掩码（66.8% 海洋，深度 0-7567m） | `255c779` |
| **G1** | 08-25 | FD 算子族 + MMS 制造解验证：2 阶收敛确认（dx 减半误差比 4.84，理论 ~4）；laplacian(线性场)=6.8e-7；诚实基线：1° 下 FD 误差 ~1e-3（对比谱 1e-17，这是 FD 的预期代价） | `78d0a95` |
| **G2** | 08-25 | 完整 IMEX 时间积分：Strang 分裂 L(dt/2)→N(dt)→L(dt/2)；半隐式自由表面（Crank-Nicolson 浅水 + 8 次 Jacobi Helmholtz，解除外重力波 CFL ~560s）；前向-后向 RK2 非线性步；全流程海陆掩码 | `6fb16df` |
| **G3** | 08-26 | 长积分驱动 `run_long_integration_global.py`（沿用区域 driver 的监控/快照/判据框架）+ `wind_reanalysis` 双线性插值 `np.ix_` 修复（区域 128×128 方阵掩盖的 nx≠ny 广播 bug） | `8ea657b` |

### 2.2 G2/G3 稳定性攻坚（~20 个 diag + fix 提交）

切换真实 ETOPO + WOA 初值后接连暴露六个结构性缺陷，每个都有独立诊断链（根因脚本均入 `src/archive_diag/` 档案）：

1. **幽灵水柱伪 PGF**（08-26）：海底以下节点被填 T_ref=15，密度 PGF 平均含 ghost 贡献，陆架断裂处过强迫 O(3e-5 m/s²)。修复：3D 湿掩码 + 传输一致 barotropic rho-PGF（wet 列平均）+ min_depth 地板 + 地形平滑（`431e126`/`88fe72b`/`b84fc34`，后者 `a636c5b` 终版）。
2. **N/S 边界单侧差分爆炸**（08-26）：截断纬度 ±60°（消 1/cos² 度量奇异）+ 镜像 ghost 胞无通量墙（∂u/∂n=0）+ 边界行法向速度归零（`eb54d82`）。
3. **2-dx 网格尺度 w 噪声**（08-27）：诊断 w 出现网格尺度锯齿 → 全部非线性平流乘积做 dealias（`b6e6e6d`/`84e0e02`）。
4. **极帽硬边不连续**（08-27）：zonally-uniform 极帽与自由行之间 6.7°C 悬崖被 _d_dy 指数放大（每步翻倍，step 12 NaN）→ cos² 锥形混合（MOM6 式，`48c82fd`）。
5. **自由表面质量泄漏 + 伴随不匹配**（08-27）：质量漏 → 通量形式 barotropic 散度；FB 步 div/grad 伴随不匹配 → consistent-triple cap + 伴随一致 eta 梯度（`6ee26cf`/`520472b`）。
6. **day-45 赤道大西洋 barotropic 爆炸**（08-29）：两个裸中心差分 PGF 非伴随、在海岸线注入能量，赤道（f→0）无法束缚 → FIX #5a/5b：`_gradient_conservative`（2D/3D 掩码伴随 PGF）。消融证据：修复前 eta 5.36(d45)→17.90(d49) 爆炸；仅 #5a 推迟到 d50-54；#5a+#5b 单调下降 PASS（`3461e82`）。

另有独立热模式（Solomon Sea T-runaway）在 08-28 由 **10d gate 修复套件**解决（`4c19a07`：海岸线 T 悬崖的 pre/post 掩码保持、对流调整 wet_mask_z 门控、半隐式 barotropic Coriolis——后者正是生产配置 eta 漂移的真正元凶，6 天 e-fold → 15m by d10）。

**10d gate PASS 后**（08-28）生产配置定型：lat ±60°、no-flux 墙、polcap 2 行 + cos² 锥形、nu_h=5e6、nu_bi=2e14（后调）、bulk λ=40、真实 2023-01 NCEP 风、dt=60s。

### 2.3 中途的物理性诊断（诚实记录）

- **ITCZ 盐度失控**（09-01）：365d 在 d135 NaN。根因链完整：`_d2_dz2` 顶/底边界节点误用 k=1 的曲率（一格错位的**反扩散**边界），线性 EOS 下 WOA "咸盖淡"剖面被对流调整**放大**而非清除（conv_S 最高 +84.5 PSU/day，手算 84.7 复现）→ ITCZ 盐度偶极 → rho PGF → eta 跳变 → NaN。修复：ghost-mirror 边界（2(C1-C0)/h0² 双向真扩散），共享算子让 conv/diff_v/动量垂直扩散同时获得无通量边界（`09fcc92`）。90d gate 复跑 PASS（max|eta|=1.904）。
- **Amazon 扇赤道 eta 团块**（08-31，`a636c5b`）：barotropic rho-PGF 平均含 ghost 层贡献 + sponge 体积不守恒（+0.377 m/90d 全球漂移）。修复：wet 列传输一致形式 + sponge 移除体积全局均匀补水。90d gate PASS（max|eta|=1.902，全球平均 eta 机器零）。

---

## 3. GM 闭合 365d 攻坚（08-29，append-only 完整记录在长积分报告）

目标：给全球 FD 加 Gent-McWilliams 涡致输送闭合。**六轮 365d 全程 FAIL_BLOWUP 排除假设**（tanh-clip、ghost-fill、κ_GM=300、+Redi、DM95 taper、slope_max 收紧 4 倍），量化锁定增长 ∝ κ_GM 线性、∝ slope_max 线性。

三条独立证据链逐层推进：

1. **算子谱分析**（本地 jax CPU 复现真实代码）：node-flux 垂直偏斜项是 5 点/2 步格式，偶/奇 z 子格精确解耦，锯齿模特征值恰为 0（永不衰减），算子不对称（max asym 2.7e-2）——结构性缺陷坐实。
2. **接口通量格式重写**（MOM6/NEMO 式）：14×14 柱算子验证 max Re eig = 4.2e-22（机器零），守恒 dz-加权列和 5e-19。但 365d 仍 FAIL（诚实记录）。
3. **逐项 dT/dt 分解**（terms_fn，GPU 并行两跑）：adv 主导（d280 达 -4.74 K/d），GM 全程 ≤ 0.115 K/d 良性；对照实验（κ_GM=0）**更早失稳**且同一胞元——失稳在基础平流物理，GM 只推迟 80 天。

**真正根因（一石三鸟的发现）**：极冠滤波用 **2D 湿掩码**做纬向平均——4000m 层上南冠 121/360 列是 ghost（海深 <4000m，被填 T_ref=+15），带状均值被几何拖向 +15.0°C（实测 d10 = 14.99，与 (239·(-0.3)+121·15)/360 精确吻合）。+15 平台在冠缘形成陡崖 → 近中性层结 → 平流冷池偶极。修复：3D 极冠改 `wet_mask_z` 逐深度湿点均值（`f5252ee` 附带 ghost-bottom fill）。

**第二个叠加根因**（gpu365_fgate 暴露）：`_laplacian_h` 未门控——裸中心差分跨湿/ghost 面读取 +14K 哨兵陡崖，× κ_bi=2e14 = 持久边界增温晕圈（+0.3 K/d），且 L-step 项对 terms_fn 不可见（解释了此前 "adv" 的误归因）。修复：双面差分开面门控（`ac62bb9`）。消融矩阵确认 biharmonic 必需、GM 无罪。

---

## 4. 最终 365d + A1/A2 双 PASS（09-03，merge `549d6e1`）

`gpu365_glap2`（修复后完整 365d，GM=1000）：

| 判据（预注册，不挪 bar） | 结果 | 判定 |
|------|------|------|
| 365d 完成，0 NaN | ✅ | PASS |
| max\|u\| < 10 m/s | 0.65–0.71 | PASS |
| max\|T\| 稳定（cap 41.65°C） | 28.09 稳定 | PASS |
| max\|eta\| < 15 m | 9.58 | PASS |
| **A1** 纬向平均 SST vs WOA（corr>0.3, RMSE<2） | corr=1.000, RMSE=0.319 | **PASS** |
| **A2** 大尺度 SST 型态，>2°平滑去均值（非循环） | corr=0.980, RMSE=1.728 | **PASS** |
| B 类（信息性） | B2 谱斜率 −3.13，B3 KE drift 1.06% | — |

双 PASS 之前还有一次**forcing 侧的诚实 FAIL**：gpu365_glap 的 A1/A2 corr PASS 但 RMSE 3.29/3.46 FAIL，偏差形态（热带 −3~−5K 对称、极区 +2~+4K）指向两个 forcing 构造 bug，均实测坐实：① `air_temp_profile` 对含陆地填充的 T_init 做裸纬向平均（赤道 T_atm 24.12 vs 海洋-only 27.36——"冷偏差"就是 forcing 本身，模型 SST 精确平衡到污染目标 24.09）；② 区域周期 y 缝 taper 被误用在有界全球域（59.5°S T_atm 17.12 vs 真值 −0.83 → +0.886 K/d 虚假极区加热）。修复：`grid.is_global` 分支 + 海洋-only 加权纬向平均 + 无 y-taper（`8aa9acc`）。**非循环性保持**：T_atm 仍纬向均匀，只规定经向梯度，纬向结构留给模式预测——这正是 A2 能 PASS 的前提。

**merge 与重组**：`agent/pam-mt5l9102` → main（`549d6e1`，136 文件 +12144 行）；134 个 `_*.py`/`diag_*.py` 攻坚脚本移入 `src/archive_diag/`（append-only，README 索引脚本类↔历史段落映射）；测试 4 套 55/55 本地 PASS + test_gm_closure 11 个节点验证；英文 PLAN 归档 `docs/plan_gm_closure_en.md`。

**遗留（诚实）**：max|eta| 线性 +0.026 m/d，源为地中海（16.5°E, 37.5°N）半封闭海盆——Gibraltar 14km 在 1° 网格是次网格，一格直布罗陀无法支撑双层交换，残余 PGF 失配驱动伪净流出。年积分内无碍，~500d 会触 15m 看门狗。

---

## 5. Med eta 修复：实现 → 网关封锁 → 真实验证闭环（09-03 ~ 09-04）

### 5.1 修复设计（`1bbba9a`）

MOM 族标准做法：Med 盒 [−6,42]°E × [30,46.5]°N 内 Rayleigh 松弛 η→0（τ=30d，cos-taper 边缘），移除体积**全局质量守恒补水**（零 PGF，动力学不受影响）。零速率路径 bit-exact 验证。

### 5.2 osm 网关封锁与 012 迁移

验证 bump 测试最初被 osm 网关（k8s-node3-gpu）连续封锁（"LICENSE_OUT_OF_LIMIT" 拥塞横幅）阻断，commit 里如实记了 "execution blocked by gateway outage"。改走 yundun 堡垒机 → **k8s-sh-azn-gpu-012**（8× L20X GPU）。

### 5.3 真实 ETOPO bump 验证 PASS

合成平底地形上 bump 衰减单调振荡（1.0→−0.354m），诊断为平底波畸变（无海岸线反射）判据失效——改推**真实 ETOPO**（10.4MB npz twin，2321 块 base64，~2h15m，md5 `0f7081a2`）后重跑：

- 注入 1m SSH 扰动（Med 盒内），800 步后进入 ±0.002m 平稳衰减，与 exp(−rate·t) 预期一致；
- `gm_eta` 全程精确守恒（全局补水路径正确）；A_ocean=3.2437e14 m² 与真实海陆分布吻合。

Plan A 从"已实现未验证"变为"已验证"，`1bbba9a` 的缺口关闭。

---

## 6. 动态强迫路径 + 季节性风应力（09-04，本弧线最后的架构关隘）

### 6.1 为什么季节循环此前一直缺席

所有历史长积分（区域与全球）风应力都是单月快照。区域模式的教训：12 闭包 × 12 月快照 = 12× XLA 图内存 → OOM 崩溃（365d 区域 arc 的死因）。FD 侧被迫同样推迟（G3 commit 原注释 "seasonal cycling deferred"）。**全球模式的季节循环是物理上必须补的**：A2 判据要在大尺度型态上成立，KE/SSH 的年循环信号是强迫真实生效的直接证据。

### 6.2 step_dyn 设计（`bd76d5f`）

`FDPhysParams` 是 namedtuple，强迫字段烘在 JIT 闭包里。改动核心：

```python
@jax.jit
def step_dyn(state, tau_x, tau_y, q_heat):
    return _step_impl(state, params._replace(
        tau_x_2d=tau_x, tau_y_2d=tau_y, Q_heat_2d=q_heat))
```

强迫作为运行时 traced 参数进图，**12 个月快照共享同一张 XLA 图**——12× 内存问题从架构上消除。默认路径（`dynamic_forcing=False`）返回 arity 与数值均不变，archive 的 88 个诊断脚本不受影响。

**等价性验证（预注册精神）**：同初值同随机风，烘焙 vs 动态各 50 步，最大差异 du=6.9e-18 / dT=0.0 / deta=4.3e-19——XLA fp64 1-ulp 舍入级，判据完全不受影响。

### 6.3 驱动集成（`3f5ed80`/`d8a7bf1`）

- `--seasonal-wind` 用 `step_dyn` + 12 个月 NCEP R1 2023 快照 + 月界 5 天线性混合（沿用区域模式验证过的做法，消除月界阶跃）；
- 新增 `--init-from <npz>`：预计算 WOA 初始场直接加载（见 §7.2）；顺手修复 argparse 插入时丢失的 `--save-3d` 条目。

### 6.4 冒烟验证链（每步数值可对）

小网格（lat_max=15）0.5 天 seasonal+eta_relax 全路径 PASS → **全生产网格 360×120×14** 0.5 天 CPU PASS（max|u|=0.709, max|T|=29.565, max|eta|=3.144）→ GPU 上**逐位一致**（0.709/29.565/3.144）——fp64 数值跨设备可复现。冻结判据一行未动。

---

## 7. 离线环境搭建：012 节点全链路（09-03 ~ 09-04）

### 7.1 连接与运行环境

| 项 | 内容 |
|----|------|
| 堡垒机 | yundun.insightst.com:60022，交互式菜单 PTY → `:` + 节点号 → root shell；`stty -echo` 关回显 |
| 命令通道 | PTY 延迟大且会串流 → marker 协议（`echo S{i}; cmd; echo E{i}`，读到 E{i} 才返回）——早期 push 校验 FAIL 的根因 |
| 容器 | `jaxtest2`（weatherllm:v1.0.0-py311-dev-mlflow），`-v /data:/data -w /data/tmp/ocean/src` |
| 目录 | `/data/tmp/ocean/{src,data/{woa,wind},results,logs}`，全部离线自足 |
| 权限坑 | 容器 uid=1000(trainer) vs 宿主 root 目录 → `chmod -R 777 /data/tmp/ocean`（bump.log 写入失败首次暴露） |

### 7.2 文件传输：为什么是 base64-over-PTY

用户问过为什么不用 rsync/scp——结论：yundun 是审计堡垒机，只给交互式菜单 PTY，不暴露节点 sshd/SFTP，禁端口转发（-L/-R/-D/ProxyJump），012 在 VPC 里只通 aliyun 镜像源、无法反连开隧道。PTY 键入是唯一通路：6000 字符/块、marker 协议、`/bin/rm -f`（root 的 rm 别名 rm -i 会挂起交互）、逐文件 md5。本弧线累计推送：全部源文件 + ETOPO 10.4MB（~2h15m）+ 12 月风场 3.5MB（~36min）+ init 场 3.2MB（~40min），全部 md5 OK。

### 7.3 WOA 20MB → 3.2MB（确定性预计算）

012 无外网，WOA2023 npz twin（19.9MB）按慢信道要 ~4.5h。观察 `get_initial_fields(grid)` 是**纯确定性 scipy 插值**，本地有同一 WOA npz + 同一 ETOPO（md5 已核对），于是在本地对生产网格精确参数（lat_max=60, ny=120, smooth=30, min_depth=100）预计算 `init_fields_g360x120.npz`（3.2MB，T [−1.90, 29.65]°C，S [5.98, 40.59] PSU，高值在红海/波斯湾 1° 格点，物理合理）。4.5h → ~40min，012 不再需要 WOA 依赖。风场不需要此步骤（12 月快照 cache npz 本来就独立，3.5MB）。

### 7.4 GPU 栈修复（8× L20X 全部可见）

容器内 jax/jaxlib 名义均 0.4.35，实际 **jaxlib 0.4.34（CPU wheel）+ jax-cuda12-plugin 0.4.35** 错配（早前 pip 解析残留）。修复链：卸 jax 全家 → `jax[cuda12]==0.4.35`（拉全 nvidia-cu12 依赖栈）→ `--no-deps jaxlib==0.4.35` 补齐。仍失败：`nvidia/cuda_nvcc/` 目录在但**无 `__init__.py`** → namespace 包 `__file__=None` → jax 0.4.35 的 `_try_cuda_nvcc_import` 捕获 ImportError 不捕获 TypeError。一行修复 `touch __init__.py` → `jax.devices()` 返回 8× CudaDevice。

3 天测速：4320 步 1.4 分钟（含 ~0.7 分钟编译）→ 稳态 ~90 步/s → **365 天 ≈ 100 分钟纯算**。

---

## 8. 生产运行（365 天，012 GPU，当前状态）

**启动**：2026-09-04 01:00（012 本地），容器内 nohup，PID 6211。

```
python run_long_integration_global.py \
  --days 365 --dt 60 \
  --seasonal-wind --wind-year 2023 --wind-blend-days 5 \
  --eta-relax-days 30 --eta-relax-box -6 42 30 46.5 \
  --tag g365d_012 --out-dir /data/tmp/ocean/results --log-dir /data/tmp/ocean/logs \
  --init-from /data/tmp/ocean/data/init_fields_g360x120.npz
```

| 项 | 值 |
|----|-----|
| 网格 | 360×120×14, lat ±60°, dx_eq≈111km, 海洋 71.6% |
| 地形 | 真实 ETOPO 2022（30 passes 平滑, min_depth 100m） |
| 初始场 | WOA2023 预计算 npz（T max 29.65°C） |
| 风应力 | NCEP/NCAR R1 2023 年 12 月快照，动态传入，月界 5 天线性混合 |
| 热通量 | bulk λ=40 W/m²/K，T_atm=海洋-only 纬向 WOA SST 廓线（非循环） |
| eta_relax | τ=30d, Med 盒, 质量守恒（`1bbba9a`，bump 已验证） |
| 次网格 | GM κ=1000（skew-flux 形式）、nu_h=5e6、nu_bi=2e14 |
| 判据 | max\|u\|<10；漂移容差 2.0°C；\|eta\|<15 看门狗；**不事后挪动** |

**写作时的中期状态**（d290 实测读数）：max|u|=0.728, max|T|=28.00, max|eta|=1.38, **0 NaN**，KE 呈现明显月际振荡（季节循环生效的直接证据），eta_relax 把地中海 SSH 压在 1.4m（对比无修复的 9.58m @ d365 线性增长）。预计 ~2 小时完成。

**本段验证链**：动态 forcing 等价性（1-ulp）→ 小网格冒烟 → 全网格冒烟（CPU）→ GPU 冒烟（逐位一致）→ 3 天测速 → 365 天启动。生产运行没有引入新的未验证环节。

---

## 9. 运行完成后的验收清单（预注册）

> **状态（2026-09-04）：全部完成，记录如下。**

1. **读 VERDICT**（PASS / FAIL_BLOWUP / FAIL_DRIFT），诚实记录，不挪 bar；
   → **VERDICT: PASS**（525600 步，0 NaN，max_u_peak 1.418，末帧 max|u|=0.651 / max|T|=27.972 / max|eta|=1.697，wall 32.3 min）。
2. **画图**（用户明确要求的交付物）：max|u|/max|T|/max|eta|/SSH_std/KE 五联时序；T_top 初始化 vs 365 天对比；eta 末帧平面图（重点 Med 盒）；月均风应力矢量图（证明季节循环进去了）；
   → `results/acceptance_g365d_012/`（fig_A~fig_D，生成脚本 `make_acceptance_figs.py` 已入库）。
3. **季节性检查**：KE/SSH_std 时序的 12 个月周期信号；
   → KE 年周期 R²=0.403（+半年项 0.770），SSH_std 年周期 R²=0.768（+半年项 0.919）；强迫侧 NW Pac 盒 |tau| 季节摆幅 560%。
4. 结果 npz 拉回本地（base64 慢信道，压缩后 ~5-15MB）或 012 上画图取回 PNG；
   → `global_g365d_012.npz`（20.2MB，gzip 后 20.16MB）已按 base64-over-PTY 分块拉回本地，gz 与 npz 两级 md5（`6743233c` / `53876042`）均与 012 端一致。
5. 验收通过后把 eta_relax 验证结论与季节性运行写进 `docs/report.md` 主线。
   → `docs/report.md` §6（commit `fd77f4b`），含官方 A1/A2 评分：**OVERALL PASS**（A1 corr 0.993 / RMSE 1.080，A2 corr 0.977 / RMSE 1.881，B2 谱斜率 −3.02）。

---

## 附 A：这条弧线的完整提交清单（G0 → 生产启动）

### 建设与稳定性（08-25 ~ 08-29）
| commit | 内容 |
|--------|------|
| `255c779` | G0 全球 1° 网格 + 球面度量 + 海陆掩码 |
| `78d0a95` | G1 FD 算子 + MMS 验证（2 阶收敛确认） |
| `6fb16df` | G2 FD IMEX 求解器 + 半隐式自由表面 |
| `f640460`~`3461e82` | G2/G3 稳定性攻坚：极帽滤波、no-flux 墙、dealias、保守通量形式、Sielecki 耦合、伴随 PGF（#5a/5b）、10d gate 套件（`4c19a07`） |
| `4c19a07` | 生产 FD 求解器稳定化 — 10d gate PASS |
| `a636c5b` | 传输一致 rho-PGF + 质量守恒 sponge（Amazon 团块修复） |
| `09fcc92` | `_d2_dz2` 无通量边界（ITCZ 盐度失控修复） |

### GM 攻坚与双 PASS（08-29 ~ 09-03）
| commit | 内容 |
|--------|------|
| `8031b21` | GM skew-flux 形式 + no-flux Fz BC（step-12 爆裂修复） |
| `9beea2f`/`2d8c3ab`/`b6e3033` | biharmonic 默认值启用与调参 |
| `f5252ee` | face-gated 梯度 + ghost-bottom fill（2D 湿掩码极冠根因修复） |
| `ac62bb9` | face-gated `_laplacian_h`（κ_bi 幽灵哨兵增温晕圈修复） |
| `8aa9acc` | 海洋-only T_atm 廓线 + 全球域去 y-taper（RMSE FAIL 根因） |
| `549d6e1` | **merge：365d 稳定 + A1/A2 双 PASS**（136 文件，+12144 行） |
| `c53d173`/`58e572e`/`459f8f4`/`09a4dba`/`c2afacc` | 报告、计划归档、134 诊断脚本入库 |

### 热力学收尾与生产准备（09-03 ~ 09-04）
| commit | 内容 |
|--------|------|
| `1bbba9a` | Med-region eta Rayleigh 松弛（半封闭海 SSH 伪影修复） |
| `bd76d5f` | **动态强迫路径 step_dyn**（季节性风的架构前提） |
| `3f5ed80` | `--init-from` 预计算初始场加载 |
| `d8a7bf1` | argparse 修复（--save-3d 条目） |
| `433ffaa` | 本文档窄版（本版为全弧线扩展） |
| （运行中） | `g365d_012` 365 天生产积分，012 GPU |

## 附 B：方法学红线（全程遵守）

- **预注册判据不挪 bar**：MAX_U_BOUND=10 / DRIFT_TOL_C=2.0 / AMPLITUDE_CAP_C=12 / ETA_BLOWUP_M=15 冻结；A1/A2 corr>0.3 / RMSE<2.0 预注册；
- **诚实 FAIL 驱动下一步**：区域循环性 FAIL → 全球重写；GM 六轮 FAIL → 谱分析 → 真根因；RMSE FAIL → forcing 构造 bug 实测；
- **不碰 Python314/site-packages**（仅 PYTHONPATH 方式运行）；区域谱求解器（main 上的 jax_solver.py）不动，保留为验证基线；
- **默认路径不变性**：动态 forcing 等价性 1-ulp、eta_relax 零速率 bit-exact、`is_global` 分支不动区域 forcing——每次改动都有等价性验证；
- 根因脚本全部 append-only 入库（`src/archive_diag/` 135 个），可复现审计。
