# 工业级对标现状

更新时间：2026-09-27 17:50

## 1. 直接对照

最硬证据仍是 MOM6 0.5 度同切片直接对照。

| 对照 | ocean_solver | MOM6 | 状态 |
|---|---:|---:|---|
| 30d wind-only | A2 0.866 C | A2 1.096 C | 完成 |
| 30d prescribed restore | A2 0.865 C | A2 1.115 C | 完成 |
| 30d Stage-F | A2 1.692 C | A2 1.782 C | 完成 |
| 30d Stage-F 3D | 0.861 C | 1.036 C | 完成 |
| 30d band-ice + fixed-MLD | A2 1.562 C；3D 0.777 C | A2 1.782 C；3D 1.036 C | 30d 通过 |
| 365d Stage-F 3D | no-ice 1.547 C | v12 running | 年度对照未完成 |

## 2. 年度内部候选

| candidate | 结论 |
|---|---|
| band-ice + fixed-MLD v2 | rejected；global A2 1.246 C，NA 1.495 C |
| band-ice + fixed-MLD + cooling gate v2 | rejected；global A2 1.315 C，NA 1.947 C |

两个方案都能改善部分 MLD 或 near-wall bias，但年度 global/NA/3D 误差变差，
所以只能保留为 diagnostic。

## 3. 当前判断

1. 当前内部 baseline 仍是 **365d Stage-F no-ice control**。
2. 固定深度 + 北向动态海冰这一族已被有效年度 gate 拒绝。
3. 在 MOM6 年度 final-90d 3D/MLD 对照完成前，暂停新的内部 upper-ocean closure 调参。
4. 下一步先完成外部 direct comparison，再决定是否继续做物理闭合。

## 4. 当前运行

| run | status |
|---|---|
| MOM6 annual Stage-F v12 | running |
| ocean_solver band-ice + fixed-MLD annual v2 | completed; rejected |
| ocean_solver cooling-gate annual v2 | completed; rejected |
