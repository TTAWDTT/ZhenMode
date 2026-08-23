# archive_baroclinic/ — 已搁置的斜压激活实验脚本

> 状态：**方向已搁置**（2026-08-23）。本目录的脚本保留作历史档案，不再作为开发主线。
> 详见 `docs/repositioning_memo_zh.md`。

## 这里有什么

这 6 个脚本是 2026-08-22 ~ 2026-08-23 期间，为"激活斜压不稳定、让模型产生中尺度涡以匹配真实 SLA"这条路线写的诊断实验：

| 脚本 | 对应假设 | 做什么 |
|------|---------|--------|
| `bench_baroclinic_step1.py` | — | 斜压激活诊断：验证 WOA 分层初场下 ρ'≠0、baroclinic PGF 非零 |
| `bench_baroclinic_step2.py` | H3 | 黏性扫描 + 空间结构诊断 |
| `bench_baroclinic_step5.py` | — | 真实数据中尺度带通 SLA 对比 |
| `bench_baroclinic_step5_diag.py` | — | 径向谱 / 带通相关 / 相干相位，发现 <460km 无中尺度功率 |
| `bench_baroclinic_spin_evo.py` | H1/H2/H4 | 90 天 spin-up 演化 + 扰动播种 + 日风选项 |
| `bench_baroclinic_diag.py` | — | 辅助诊断 |

## 为什么搁置

四个假设（H1 spin-up 长度 / H2 缺初始扰动 / H3 黏性过强 / H4 风缺高频变率）**全部证伪**——90 天积分 `eddy_frac` 全程 ≈0，根因是 128×128 / dx≈9km 的网格无法解析该海域 ~30-50km 第一斜压变形半径。

但更根本的原因是**项目目的的重新定位**：本模式的目标是"**稳定积分、速度快、物理正确的谱方法海洋模式**"，而非"点对点逼近真实观测"。在这个定位下，中尺度涡出不出来不是缺陷——很多经典物理模式（Stommel/Munk 解析理论、QG 模式）有意不含涡，照样是成功的大尺度动力学模型。追涡是另一条路（网格加密到 256×256），那是**新项目**，不是当前模式的补丁。

## 它们还有价值吗

有。这些脚本：

- 忠实记录了"为什么不能在当前分辨率下追涡"的完整证据链（H1-H4 证伪），避免后人重复走这条路
- `bench_baroclinic_step5_diag.py` 的径向谱/带通相关工具，在将来做能谱分析时可复用
- `bench_baroclinic_spin_evo.py` 的发散看门狗（`ETA_BLOWUP_M`）是稳定积分的好工具，可移植

## 注意

- 这些脚本**不**被任何主流程 import，移到此目录不影响核心求解器
- 脚本里的相对 import 路径（`from jax_solver import ...`）在此目录下可能需要调整才能直接运行
- 若日后要重启斜压方向，从 `docs/baroclinic_activation_roadmap_zh.md` 接续
