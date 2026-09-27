# 工业级对标现状

更新时间：2026-09-27 16:10

## 1. 直接对照

最硬证据仍是 MOM6 0.5 度同切片直接对照。

| 对照 | ocean_solver | MOM6 | 状态 |
|---|---:|---:|---|
| 30d wind-only | A2 `0.866 C` | A2 `1.096 C` | 完成 |
| 30d prescribed restore | A2 `0.865 C` | A2 `1.115 C` | 完成 |
| 30d Stage-F | A2 `1.692 C` | A2 `1.782 C` | 完成 |
| 30d Stage-F 3D | `0.861 C` | `1.036 C` | 完成 |
| 30d band-ice + fixed-MLD | A2 `1.562 C`；3D `0.777 C` | A2 `1.782 C`；3D `1.036 C` | 30d 通过 |
| 365d Stage-F 3D | no-ice `1.547 C` | MOM6 v11 running | 年度对照未完成 |

## 2. 年度内部候选

band-ice + fixed-MLD 的有效年度 v2 检查已完成，结论是**拒绝**：

| 指标 | no-ice control | combined probe |
|---|---:|---:|
| global A2 | `1.204 C` | `1.246 C` |
| NA RMSE | `0.975 C` | `1.495 C` |
| near-wall RMSE | `1.055 C` | `1.260 C` |
| global 3D RMSE | `1.547 C` | `1.559 C` |
| NA 3D RMSE | `1.276 C` | `1.626 C` |
| near-wall 3D RMSE | `1.124 C` | `1.387 C` |
| MLD bias | `+139.3 m` | `+121.2 m` |
| 40--60N MLD bias | `+224.6 m` | `+17.2 m` |

MLD 明显改善，但年度温度和三维误差变差，所以不能作为 baseline。

## 3. 当前判断

1. 当前内部 baseline 仍是 **365d Stage-F no-ice control**。
2. 固定深度 + 北向动态海冰这一族已被有效年度 gate 拒绝。
3. 已定位 v2 候选失败来源：40--60N 上层海洋的晚季增暖。
4. 后续物理改进只能是季节依赖 / 冰态依赖，而不是全局标量。

## 4. 当前运行

| run | status |
|---|---|
| MOM6 annual Stage-F v11 | running |
| ocean_solver band-ice + fixed-MLD annual v2 | completed; rejected |

