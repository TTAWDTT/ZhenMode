# 工业级对标现状

更新时间：2026-09-27 19:55

## 1. 直接对照

最硬证据仍是 MOM6 0.5 度同切片直接对照。

| 对照 | ocean_solver | MOM6 | 状态 |
|---|---:|---:|---|
| 30d wind-only | A2 0.866 C | A2 1.096 C | 完成 |
| 30d prescribed restore | A2 0.865 C | A2 1.115 C | 完成 |
| 30d Stage-F | A2 1.692 C | A2 1.782 C | 完成 |
| 30d Stage-F 3D | 0.861 C | 1.036 C | 完成 |
| 30d band-ice + fixed-MLD | A2 1.562 C；3D 0.777 C | A2 1.782 C；3D 1.036 C | 30d 通过 |
| 365d Stage-F 3D | no-ice 1.547 C | v12 running，约 day 104/365 | 年度对照未完成 |

## 2. 年度内部候选

| candidate | 结论 |
|---|---|
| band-ice + fixed-MLD v2 | rejected；global A2 1.246 C，NA 1.495 C，3D 1.559 C |
| band-ice + fixed-MLD + cooling gate v2 | rejected；global A2 1.315 C，NA 1.947 C，3D 1.574 C |

两个方案都能改善部分 MLD 或 near-wall bias，但年度 global/NA/3D 误差变差，
所以只能保留为 diagnostic。

## 3. 下一步

1. 等 MOM6 annual Stage-F v12 完成 final-90d 3D/MLD 评分。
2. 完成评分前，不再启动新的内部 upper-ocean closure。
3. 评分后若继续做闭合，先试 cooling_ice：只在「大气冷却 SST」或「有动态海冰」
   的格点上启用 40--60N 的 100m 混合层。这是一个已预注册的
   seasonal/ice-state dependent closure，不是新的全局标量调参。

## 4. 当前运行

| run | status |
|---|---|
| MOM6 annual Stage-F v12 | running，约 day 104/365 |
| ocean_solver band-ice + fixed-MLD annual v2 | completed; rejected |
| ocean_solver cooling-gate annual v2 | completed; rejected |
