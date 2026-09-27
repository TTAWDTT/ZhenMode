# 工业级海洋模式差距矩阵

> 2026-09-27。这里区分三类证据：`direct rerun`、`public output`、`design/literature reference`。
> 只有 MOM6 0.5 度切片目前有 direct rerun；其他模式先作为设计和协议参照。

## 当前直接对照

| 模型 | 对照级别 | 已完成 | 关键数字 |
|---|---|---|---|
| MOM6 | direct rerun | 30d wind-only | ocean_solver A2 `0.866 C`; MOM6 `1.096 C` |
| MOM6 | direct rerun | 30d prescribed restore | ocean_solver A2 `0.865 C`; MOM6 `1.115 C` |
| MOM6 | direct rerun | 365d prescribed restore | ocean_solver A2 `2.245 C`; MOM6 本地重跑中 |
| MOM6 | direct rerun | 365d Stage-F exact dynamic bulk v10 | ocean_solver annual 3D global RMSE `1.547 C`; MOM6 v10 running with temp+salt |
| MOM6 | direct rerun | 30d Stage-F exact dynamic bulk | ocean_solver A2 `1.692 C`; MOM6 `1.782 C` |
| ocean_solver | internal | 30d Stage-F prescribed proxy | FAIL_DRIFT; A2 `5.629 C`; max T `51.35 C` |
| MOM6 | direct rerun | 30d Stage-F prescribed proxy | 稳定性警告; 因 C 盘满无空间评分 |
| MOM6 | direct rerun | 30d Stage-F 3D temperature | corrected: ocean_solver `0.861 C`; MOM6 `1.036 C`; surface favors ocean_solver, NA favors MOM6 |
| ocean_solver | internal | 365d Stage-F MLD20 | A2 `1.406 C`; NA RMSE `1.302 C`; heat drift `-0.276%` |
| ocean_solver | internal | 365d fixed 100m 40--60N MLD | A2 1.343 C; NA RMSE 1.451 C; near-wall 1.209 C; not promoted |
| ocean_solver | internal | 30d Stage-I ice/mixed-layer | A2 `1.231 C`; NA RMSE `1.316 C` |
| ocean_solver | internal | 365d Stage-F dynamic ice, no MLD | A2 `1.213 C`; NA RMSE `0.983 C`; not promoted |
| ocean_solver | internal | 365d Stage-F dynamic ice, 3D | global 3D RMSE `1.543 C`; MLD `288.7 m`; not promoted |

这些数字只代表固定 0.5 度切片，不是 OMIP、预报系统或全球气候模式级结论。

## 差距矩阵

| 方向 | 工业级常见做法 | ocean_solver 当前状态 | 主要差距 | 下一步证据 |
|---|---|---|---|---|
| 动力核 | 有限体积/有限差分、保守算子、成熟压力梯度与垂直坐标 | 0.5 度浅水+tracer 原型，已有 split、FCT、保守约束 | 非 OMIP 级物理核 | 365d matched bulk 对照 |
| 平流输运 | 可调 tracer advection、monotonic/FCT、上游稳定性 | 已加 FCT/TVD，30d/365d 稳定 | 缺 3D FCT 与更多 tracer | 单独 tracer advection benchmark |
| 海气通量 | 完整 bulk：T/S/radiation/precip/evap/runoff | 目前只有 wind + dynamic Haney/Bulk heat proxy | 缺湿度和辐射通量 | 补齐 Stage-F forcing 清单 |
| 海冰 | 成熟热力学+动力学海冰模式 | 最小 ice thickness/growth/melt/insulation/brine loop | 不是海冰模式 | Stage-I 协议已建，但需动态 bulk 365d |
| 混合层 | KPP/参数化边界层 | 简化混合层与 stratification MLD 诊断 | 需要标准混合层误差指标 | MLD bias/RMSE 对照 |
| 水团/环流 | 长期 spinup 和 WOA/Argo 水团检验 | 只有 SST 为主，短积分 | 未验证三维环流 | 增加 temperature/salinity 剖面误差 |
| 可扩展性 | MPI/并行域分解，部分 GPU/加速器 | JAX 可选 scan/compile，单机为主 | 缺多节点 scaling | 补 wall-time 和 scaling test |
| 可维护性 | 分模块、可测、可复现实验目录 | 有 manifest/scorer/gate/protocol | 尚未成完整用户系统 | 保持 benchmark contract 优先 |
| 科学声明 | OMIP/CORE-II/再分析类公开协议 | 本项目自设 0.5 度切片 | 不能直接宣称全球性能 | 后续接入 OMIP 子集 |

## 当前判断

1. **小切片上已经能跑通工业级对照协议**，但还不能宣称超过工业级模式。
2. **Stage-F 是下一个门槛**：只有通过完整 bulk forcing 365d，才比 30d restore 更有说服力。
3. **最小 ice/mixed-layer 闭环已经稳定**，但仍要和 industrial 模式在同一 forcing 下对比。
4. **修正后的 30d 3D 对照**：ocean_solver 全域 RMSE `0.861 C`，MOM6 `1.036 C`；MOM6 在 40--60N 与近壁略好。此前“深层暖偏差”主要是 3D scorer 把海底以下 ghost layers 计入了。
5. **年度动态海冰 3D gate 已失败**：3D RMSE 基本不变（1.543 vs 1.547 C），MLD 反而恶化到 288.7 m，因此不能作为年度基线。
6. **要超过工业级模式，优先级不是继续调参**，而是：完整 bulk forcing → 365d 对照 → 三维误差 → 并行/速度 → 扩展到更多物理过程。









