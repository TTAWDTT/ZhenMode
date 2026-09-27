# 工业级对标现状

更新时间：2026-09-27 13:50

## 1. 直接对照

最硬证据仍是 MOM6 0.5 度同切片直接对照。

| 对照 | ocean_solver | MOM6 | 状态 |
|---|---:|---:|---|
| 30d wind-only | A2 `0.866 C` | A2 `1.096 C` | 完成 |
| 30d prescribed restore | A2 `0.865 C` | A2 `1.115 C` | 完成 |
| 30d Stage-F | A2 `1.692 C` | A2 `1.782 C` | 完成 |
| 30d Stage-F 3D | `0.861 C` | `1.036 C` | 完成 |
| 365d Stage-F 3D | `1.547 C` | MOM6 v11 running | 年度对照未完成 |

MOM6 在 30d North Atlantic 三维误差略好。年度三维对照是当前主要门槛。

## 2. 最强内部候选

新的 30d 组合闭包是：

- 动态海冰只在 `40--65N` 生效；
- `100m` 混合层热容只在 `40--60N` 生效；
- 其他区域保持原 Stage-F 控制。

| 指标 | no-ice control | combined probe |
|---|---:|---:|
| global A2 | `1.692 C` | `1.562 C` |
| NA RMSE | `2.088 C` | `0.848 C` |
| near-wall RMSE | `1.958 C` | `0.534 C` |
| global 3D RMSE | `0.861 C` | `0.777 C` |
| NA 3D RMSE | `1.175 C` | `0.580 C` |
| near-wall 3D RMSE | `1.125 C` | `0.311 C` |
| MLD bias | `+36.7 m` | `+16.3 m` |

它通过了 30d 的全部预注册门槛，但还不能升为年度 baseline：对应的 365d
final-90d 检查正在跑。

## 3. 当前运行

| run | status |
|---|---|
| MOM6 annual Stage-F v11 | running |
| ocean_solver band-ice + fixed-MLD annual v1 | running |

## 4. 下一步

1. 等 MOM6 v11 完成，补齐年度 3D 温度和 MLD 对照。
2. 等 ocean_solver band-ice/fixed-MLD 年度检查完成。
3. 只有通过 final-90d 3D/MLD gate 才考虑固化为新 baseline。
4. 不再继续全局标量调参。
