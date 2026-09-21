# Budget diagnostics 结果

## 结论
新的 budget 层已经接入 run driver。  
在 1° / 365d 的 centered 和 FCT/TVD 对照里：

- 总体积严格不变。
- 全球热含量漂移约 0.7%。
- 全球盐含量漂移约 0.0007%。
- 两种 transport 方案的预算几乎完全一致。

这说明当前 1° 上的气候误差不是 tracer transport / heat-salt drift 主导。  
下一步更适合去看垂直坐标 / 地形结构 / 大尺度闭合，而不是继续换 transport。

## 365d 对照

| 方案 | mean T | mean S | heat rel drift | salt rel drift | total volume |
|---|---:|---:|---:|---:|---:|
| centered | 4.31018 | 34.71790 | 0.7050% | 7.06e-6 | 1.203616e18 m^3 |
| FCT/TVD | 4.31021 | 34.71789 | 0.7058% | 6.83e-6 | 1.203616e18 m^3 |

## 结果文件

- `results/diagnostics_budget/global_centered_1deg_365d_budget.npz`
- `results/diagnostics_budget/global_fct_1deg_365d_budget.npz`

新增诊断数组：

- `mean_T`
- `mean_S`
- `mean_T_top`
- `mean_S_top`
- `heat_content_J`
- `salt_content_kg`
- `total_volume_m3`
- `mean_depth_m`
