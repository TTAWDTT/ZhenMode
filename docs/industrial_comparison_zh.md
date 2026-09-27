# 工业级对照简报（中文）

> 本页只做结论摘要；原始数字和运行命令在
> [research/experiments/industrial_comparison_045](../research/experiments/industrial_comparison_045/)。

## 当前证据等级

| 证据类型 | 结论 |
|---|---|
| direct rerun | 目前只有 MOM6 0.5 度同切片 |
| public output | 未接入 |
| literature | 只作为设计参考，不进主表 |

## 已完成对照

| 对照 | ocean_solver | MOM6 | 状态 |
|---|---:|---:|---|
| 30d wind-only A2 | 0.866 C | 1.096 C | 完成 |
| 30d prescribed restore A2 | 0.865 C | 1.115 C | 完成 |
| 30d Stage-F A2 | 1.692 C | 1.782 C | 完成 |
| 30d Stage-F global 3D RMSE | 0.861 C | 1.036 C | 完成 |
| 30d band ice + fixed MLD A2 | 1.562 C | 1.782 C | 30d 通过 |
| 30d band ice + fixed MLD 3D RMSE | 0.777 C | 1.036 C | 30d 通过 |
| 365d Stage-F global 3D RMSE | 1.547 C | running | 年度对照未完成 |

## 当前内部诊断基线

- 365d Stage-F no-ice control：global A2 1.297 C
- annual 3D RMSE：1.547 C
- 40--60N 3D RMSE：1.276 C
- global MLD bias：+139.3 m

年度误差主要来自上层海洋和 North Atlantic；不再用单一全局标量继续调参。

## sea-ice / mixed-layer 最小闭环

已有：

- 显式冰厚状态
- 生长/融化潜热
- 冰绝缘热通量
- 卤水盐通量
- 混合层热容
- 密度阈值 MLD 诊断
- 稳定性与气候评分

但只作为 thermodynamic proxy，不是完整海冰模式。

## 下一步

1. 等 MOM6 annual Stage-F v12 完成 final-90d 3D/MLD 评分。
2. 用 --require-3d 的年度 gate 决定是否接受或拒绝候选。
3. 评分后再启动 pre-registered cooling_ice 或第二个工业对照。

## 权威文件

- Protocol：[protocol_v2.md](../research/experiments/industrial_comparison_045/protocol_v2.md)
- Gap matrix：[industrial_gap_matrix.md](../research/experiments/industrial_comparison_045/industrial_gap_matrix.md)
- Direct table：[current_direct_comparison_table.md](../research/experiments/industrial_comparison_045/current_direct_comparison_table.md)
- Stage-I status：[stage_i_minimal_closed_loop_status.md](../research/experiments/industrial_comparison_045/stage_i_minimal_closed_loop_status.md)
- Next closure：[next_closure_pre_registration.md](../research/experiments/industrial_comparison_045/next_closure_pre_registration.md)
