# 工业级对标现状

更新时间：2026-09-27

## 1. 和工业级模式的差距

目前最硬的证据是 **MOM6 0.5° 同切片直接对照**，其他模式仍只是协议/架构参照。

| 对照 | ocean_solver | MOM6 | 状态 |
|---|---:|---:|---|
| 30d wind-only | A2 `0.866 C` | A2 `1.096 C` | 完成 |
| 30d prescribed restore | A2 `0.865 C` | A2 `1.115 C` | 完成 |
| 30d Stage-F | A2 `1.692 C` | A2 `1.782 C` | 完成 |
| 30d Stage-F 3D | `0.861 C` | `1.036 C` | 完成 |
| 365d Stage-F 3D | `1.547 C` | running | 年度对照未完成 |

结论：小切片上 ocean_solver 已经不是“玩具”，但还不能宣称超过工业级模式。
MOM6 在 30d 的 North Atlantic 三维误差略好；年度三维对照是当前关键门槛。

## 2. 标准化 benchmark protocol

协议文件：`research/experiments/industrial_comparison_045/protocol_v2.md`

已固定：

- 同网格、同 bathymetry、同 WOA 初始场、同 NCEP forcing；
- 30d 稳定性门 -> 365d 气候门；
- final-10d / final-90d 评分窗口；
- 全球 A2、NA 40--60N、near-wall、heat/salt drift、stability；
- 3D 必须应用 bathymetric vertical mask；
- 外部模型缺项必须标 `not_comparable`。

## 3. sea-ice / mixed-layer 最小闭环

见 `stage_i_minimal_closed_loop_status.md`。

已经闭环：

- ice thickness / extent；
- growth / melt latent heat；
- ice insulation；
- brine salt flux；
- mixed-layer heat capacity；
- density-threshold MLD；
- stability + climate score。

还未闭环：

- 没有冰动力学；
- 没有完整湿度/辐射/降水/径流；
- 没有匹配的 MOM6 Stage-I；
- 最新的 40--60N 固定 100m MLD 年度门还没完成。

## 4. 当前运行

| run | purpose |
|---|---|
| MOM6 annual Stage-F v8 | 完成年度三维工业对照 |
| ocean_solver annual fixed 100m MLD, 40--60N | 固定深度 MLD 的年度门 |

## 5. 下一步

1. 等 MOM6 v8 完成，先补齐年度三维对照。
2. 等 fixed 100m 40--60N 年度门完成；只有超过年度 no-ice control 才讨论 promotion。
3. 如果 fixed-depth MLD 通过，再做同一个区域的 dynamic-ice + MLD 年度组合。
4. 才考虑第二个工业模式对照；不要在自设指标上继续盲调。
