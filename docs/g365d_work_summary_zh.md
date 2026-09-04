# 全球 1° 365 天积分工作总结 — Med eta 修复验证 + 真实季节性风应力首跑

> **日期**：2026-09-03 ~ 09-04
> **范围**：全球有限差分模式（`jax_solver_global.py`）365 天积分的三项工作——① Med eta 漂移修复（Plan A）的实现与预注册 bump 验证；② 动态强迫（runtime forcing）路径，使 FD 求解器首次支持真实季节性风应力；③ 在 k8s-sh-azn-gpu-012（8× L20X GPU）上完成全部环境搭建、数据推送与生产运行启动。
> **前置**：`docs/pam_work_summary_zh.md`（区域谱 + 全球 FD 的前序工作）、commit `1bbba9a`（eta_relax 实现）、`88dfc35`（Pam campaign 总结）

---

## 0. 一句话定位

这一段工作的主线是**把全球 1° FD 模式从"单月固定风应力的数百天探针"推进到"真实季节性强迫的整年积分"**。三个技术关隘：季节性风应力需要动态强迫路径（旧架构把强迫烘进 JIT 闭包，12 个月快照会 12× 显存——正是压垮区域 365d 的失败模式）；离线节点没有网络，20MB 的 WOA 初始场要靠"本地预计算 + 3.2MB npz 单文件"绕过；堡垒机只给一个交互式 PTY，所有文件传输都走 base64-over-PTY 慢信道。到本文写作时，365 天生产运行已在 GPU 上启动，预计约 2 小时完成。

---

## 1. 背景：这一步要验证什么

全球 FD 重写（Pam arc 2）的 G2/G3 门已通过：no-flux N/S 墙 + 极帽滤波 + 双调和粘性让固定风强迫下 200~1000 天稳定。但此前所有长积分有一个共同短板——**风应力用的是单月（2023-01）NCEP 快照，没有季节循环**。区域模式的教训（12 闭包 × 12 月 = 12× XLA 图内存，OOM 崩溃）使季节循环在 FD 侧被明确推迟为"future refinement"（`run_long_integration_global.py` 中原注释）。

同时，commit `1bbba9a` 的 Med eta 漂移修复（半封闭海 SSH Rayleigh 松弛，τ=30d，质量守恒全局补水）**实现已提交但从未执行验证**——当时 osm 网关封锁，bump 测试只跑了合成平底地形（单调振荡，无法定论）。本段工作把这两个缺口一起补上。

---

## 2. Med eta 修复的预注册 bump 验证（012，真实 ETOPO）

### 2.1 为什么换到 012

osm 网关（k8s-node3-gpu）连续封锁（"LICENSE_OUT_OF_LIMIT" 拥塞横幅，Host ID 直选也只回显输入）。012（k8s-sh-azn-gpu-012，8× L20X）经 yundun 堡垒机可通，于是全链路迁移。

### 2.2 环境搭建（一次性成本，全部留档可复用）

| 步骤 | 内容 | 坑 |
|------|------|-----|
| 堡垒机连接 | yundun.insightst.com:60022 → 菜单选节点 → root shell；`stty -echo` 关回显 | 拥塞时重试；connect_node 加了 90s/30s 超时 + RuntimeError |
| 命令通道 | PTY 延迟大且会串流，marker 协议：`echo S{i}; {cmd}; echo E{i}`，读到 E{i} 才返回 | 直接读 prompt 会截断，是早期 push 校验 FAIL 的根因 |
| 运行容器 | `jaxtest2`（weatherllm:v1.0.0-py311-dev-mlflow），`-v /data:/data -w /data/tmp/ocean/src` | 首个容器无 /data 挂载已弃用 |
| JAX | 0.4.35，先 CPU 验证；GPU 需修包（见 §3.2） | orphan nvidia/cuda_nvcc 目录导致 `import jax` TypeError |
| 权限 | 容器 uid=1000(trainer) vs 宿主 root 目录：`chmod -R 777 /data/tmp/ocean` | bump.log 写入失败首次暴露 |
| /data 目录 | `/data/tmp/ocean/{src,data/{woa,wind},results,logs}` | 全部离线自足，无外网依赖 |

### 2.3 真实地形推送与 bump 结果

合成平底地形上 bump 衰减呈单调振荡（1.0 → -0.354 m），诊断为平底波畸变（无海岸线反射），判据失效。改推真实 ETOPO（10.4MB npz twin，2321 块 base64，~2h15m，md5 `0f7081a2` 校验）后重跑：

- **bump 测试 PASS**：注入 1m SSH 扰动（16.5°E, 37.5°N，Med 盒内），800 步后进入 ±0.002 m 平稳衰减；
- **质量守恒**：`gm_eta` 全程精确守恒（全局补水路径正确）；
- **A_ocean = 3.2437e+14 m²**，与真实海陆分布吻合。

`1bbba9a` 的"执行被网关阻断"缺口就此关闭——Plan A 从"已实现未验证"变为"已验证"。

---

## 3. 动态强迫路径（本段唯一的求解器代码改动）

### 3.1 设计

`FDPhysParams` 是 namedtuple，强迫字段（`tau_x_2d, tau_y_2d, Q_heat_2d`）烘在闭包里。改动（commit `bd76d5f`）：

```python
# make_solver_global(..., dynamic_forcing=True) 时额外返回:
@jax.jit
def step_dyn(state, tau_x, tau_y, q_heat):
    return _step_impl(state, params._replace(
        tau_x_2d=tau_x, tau_y_2d=tau_y, Q_heat_2d=q_heat))
```

强迫作为**运行时 traced 参数**进图，12 个月快照共享同一张 XLA 图——12× 内存问题从架构上消除。默认路径（`dynamic_forcing=False`）返回 arity 与数值均不变（archive 的 88 个诊断脚本不受影响）。

**等价性验证**（预注册精神：改动不能改变默认路径物理）：同一初始条件、同一随机风场，烘焙路径 vs 动态路径各 50 步，最大差异 `du=6.9e-18, dT=0.0, deta=4.3e-19`——XLA float64 1-ulp 舍入级（烘焙常量预计算 `1/(ρ₀H)` vs 运行时除法），dT 严格为 0。判据（max|u|<10 等）完全不受影响。

### 3.2 驱动集成（`run_long_integration_global.py`）

- `--seasonal-wind` 不再静默降级为固定 1 月风，改用 `step_dyn` + `interp_seasonal_wind` 的 5 天线性月界混合（消除月界阶跃，沿用区域模式验证过的做法）；
- seasonal 模式下烘焙占位强迫为零，Q_heat 运行时传入；
- 新增 `--init-from <npz>`（commit `3f5ed80`）：直接加载预计算 T_init/S_init，跳过 woa_data。

### 3.3 冒烟验证（WSL 本地，jax 0.11.1 CPU）

- 小网格（lat_max=15, ny=30）0.5 天：seasonal + eta_relax 全路径 PASS；
- **全生产网格（360×120×14, lat ±60°）0.5 天**：PASS，max|u|=0.709, max|T|=29.565, max|eta|=3.144；
- 预注册判据框架原样保留（MAX_U_BOUND=10 / DRIFT_TOL_C=2.0 / AMPLITUDE_CAP_C=12 / ETA_BLOWUP_M=15，一行未动）。

---

## 4. 离线数据策略：WOA 20MB → 3.2MB

012 只通 aliyun VPC 镜像源，无外网。WOA2023 npz twin（t00_01 11.2MB + s00_01 8.7MB）按 base64 慢信道要 ~4.5 小时。观察：`get_initial_fields(grid)` 是**纯确定性 scipy 插值**，本地 WSL 有 scipy + 同一 WOA npz + 同一 ETOPO（md5 已核对），于是在本地对**生产网格精确参数**（lat_max=60, ny=120, smooth_passes=30, min_depth=100）预计算：

- 输出 `init_fields_g360x120.npz`（3.2MB，T/S/wet_mask/lat/lon/z），md5 `33bb07778e343999e1e0dda4bfb3e343`；
- T 范围 [-1.90, 29.65]°C，S [5.98, 40.59] PSU（高值在红海/波斯湾 1° 格点，物理合理）；
- 传输时间 4.5h → ~40min，且 012 上不再需要 netCDF4/WOA 依赖。

风场侧不需要此步骤：`load_monthly_wind` 本来就优先读 cache npz，直接推 12 个月 `monthly_mean_900..911.npz`（2023 年，共 3.5MB）。

---

## 5. 012 GPU 栈修复（8× L20X 可用）

容器里 pip list 显示 jax/jaxlib 均 0.4.35，但 jax 实际 fallback CPU，`import jax` 报 `cuda_nvcc.__file__ = None → pathlib TypeError`。诊断链条：

1. 实际状态是 **jaxlib 0.4.34（CPU wheel）+ jax-cuda12-plugin 0.4.35** 的错配（早前 pip 解析残留）；
2. pinned 重装 `jax[cuda12]==0.4.35` 联合解析报 ResolutionImpossible → 拆开装：卸 jax 全家 → 装 `jax[cuda12]==0.4.35`（拉全 nvidia-cu12 依赖栈）→ `--no-deps jaxlib==0.4.35` 补齐；
3. `from nvidia import cuda_nvcc` 仍返回 `__file__=None`：`nvidia/cuda_nvcc/` 目录在（bin/include/nvvm 都在）但**没有 `__init__.py`**，Python 把它解析为 namespace 包，`__file__` 恒为 None，而 jax 0.4.35 的 `_try_cuda_nvcc_import` 没料到这种情况（捕获 ImportError 不捕获 TypeError）；
4. **一行修复**：`touch nvidia/cuda_nvcc/__init__.py` → `jax.devices()` 返回全部 8× CudaDevice。

GPU 冒烟（全网格 0.5 天，seasonal）PASS，且诊断值与本地 CPU **逐位一致**（0.709 / 29.565 / 3.144）——fp64 数值跨设备可复现。3 天测速：4320 步 1.4 分钟（含 ~0.7 分钟编译）→ 稳态 ~90 步/s → 365 天约 100 分钟纯算。

---

## 6. 文件传输信道（为什么是 base64-over-PTY）

用户问过为什么不用 rsync/scp——记录一下结论：yundun 是审计堡垒机，连接只给交互式菜单 PTY，不暴露节点 sshd/SFTP，禁端口转发（-L/-R/-D/ProxyJump），012 本身在 VPC 里只通 aliyun 镜像源、无法反连开隧道。所以 PTY 键入信道是唯一通路，代价 6000 字符/块、~3.5 分钟/MB。可靠性靠：marker 协议（读到 E{i} 才算完成）、`/bin/rm -f`（root 的 rm 别名 rm -i 会挂起）、逐文件 md5 校验。本段累计推送：4 个源文件 + `_test_eta_bump.py`（前段）+ 7 个源文件 + ETOPO 10.4MB（~2h15m）+ 12 月风场 3.5MB（~36min）+ init 场 3.2MB（~40min），全部 md5 OK。

---

## 7. 生产运行（365 天，当前状态）

**启动时间**：2026-09-04 01:00（012 本地时间），容器内 nohup，PID 6211。

```
python run_long_integration_global.py \
  --days 365 --dt 60 \
  --seasonal-wind --wind-year 2023 --wind-blend-days 5 \
  --eta-relax-days 30 --eta-relax-box -6 42 30 46.5 \
  --tag g365d_012 --out-dir /data/tmp/ocean/results --log-dir /data/tmp/ocean/logs \
  --init-from /data/tmp/ocean/data/init_fields_g360x120.npz
```

**配置与冻结判据**：

| 项 | 值 |
|----|-----|
| 网格 | 360×120×14, lat ±60°, dx_eq≈111 km, 海洋 71.6% |
| 地形 | 真实 ETOPO 2022（30 passes 平滑, min_depth 100m） |
| 初始场 | WOA2023 预计算（T max 29.65°C） |
| 风应力 | NCEP/NCAR R1 2023 年 12 月快照，动态传入，月界 5 天线性混合 |
| 热通量 | bulk λ=40 W/m²/K，T_atm=纬向 WOA SST 廓线（非循环） |
| eta_relax | τ=30d, Med 盒 [−6,42]°E × [30,46.5]°N, 1° 缓冲（质量守恒） |
| 次网格 | 无 GM/Redi（kappa=0），nu_h=5e6, nu_bi=2e14 |
| 判据 | max|u|<10 m/s；漂移容差 2.0°C；|eta|<15 m 看门狗；判据**不事后挪动** |

**预期产出**：`results/global_g365d_012.npz`（days/max_u/max_T/max_eta/ssh_std/ke 时序 + 每 10 天 eta/T_top 快照 + 判定 VERDICT），365 天预计 ~2 小时。

**本段验证链**：动态 forcing 等价性（1-ulp）→ 小网格冒烟 → 全网格冒烟（CPU，值一致）→ GPU 冒烟（值一致）→ 3 天测速 → 365 天启动。每一步的数值都对得上，生产运行没有引入新的未验证环节。

---

## 8. 下一步（运行完成后）

1. **验收判据**：读 VERDICT（PASS / FAIL_BLOWUP / FAIL_DRIFT）；诚实记录，不挪 bar。
2. **画图**（用户明确要求）：max|u|/max|T|/max|eta|/SSH_std/KE 五联时序图；T_top 初始化 vs 365 天对比图；eta 末帧平面图（重点看 Med 盒：eta_relax 是否压住半封闭海 SSH 漂移）；月均风应力玫瑰/矢量图（证明季节循环真的进去了）。
3. **季节性检查**：KE / SSH_std 时序应有 12 个月周期信号——这是"季节性风应力真正生效"的直观证据。
4. 结果 npz 拉回本地（同样走 base64 慢信道，压缩后估计 ~5-15MB）或直接在 012 上画图取回 PNG。
5. 验收通过后把 `eta_relax` 验证结论写进 `docs/report.md` 主线。

---

## 附：本段提交清单

| commit | 内容 |
|--------|------|
| `1bbba9a` | eta_relax 实现（Med-region η Rayleigh 松弛，前段） |
| `bd76d5f` | **动态强迫路径 step_dyn**（seasonal wind 的架构前提） |
| `3f5ed80` | `--init-from` 预计算初始场加载 |
| `d8a7bf1` | argparse 修复（--save-3d 条目在插入时丢失） |
| （运行中） | `g365d_012` 365 天生产积分，012 GPU |
