# 工业级对标现状

更新时间：2026-09-27 14:35

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

第一个 band-ice + fixed-MLD 年度检查**无效**：它误用了另一个 bathymetry，
ocean fraction 是 `90.8%`，而 30d probe 和年度 no-ice control 都是 `67.9%`。
所以那些很差的 v1 数字不能作为物理结论。已删除 v1 原始输出，并用相同
bathymetry 启动 v2。现在等 v2 的 final-90d 3D/MLD gate。

## 3. 当前判断

1. 当前内部 baseline 仍是 **365d Stage-F no-ice control**。
2. 不把 30d 候选升为 baseline，也不根据无效的 v1 判它失败。
3. 等 MOM6 v11 年度 3D 温度/MLD 完成，再决定是否做第二个工业模式。
4. 后续物理改进只能是季节依赖 / 冰态依赖，而不是全局标量。

## 4. 当前运行

| run | status |
|---|---|
| MOM6 annual Stage-F v11 | running |
| ocean_solver band-ice + fixed-MLD annual v2 | running |
