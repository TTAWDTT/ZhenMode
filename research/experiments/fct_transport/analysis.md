# FCT / TVD 对照实验分析

## 结论
在 1° 全球网格、真实 ETOPO 地形、365 天积分下，三种水平 tracer transport 都能稳定通过：

- centered：精度好，但海面高度略高。
- monotone：更平滑，海面高度更低，但精度略低。
- `--fct-adv`（TVD/MUSCL）：接近 centered 的精度，`max|eta|` 比 centered 略低，
  计算成本与 centered / monotone 相近。

因此我把 `--fct-adv` 继续保留为非默认选项，暂不改默认。  
它更像一个稳健的“备份方案”：在 1° 已经很稳，但在更高分辨率或复杂海岸附近
可能比 centered 更安全。

## 30 天对照

| 方案 | max\|T\| (d30) | max\|u\| peak | max\|eta\| (d30) | mean T_top drift |
|---|---:|---:|---:|---:|
| centered | 27.894 | 0.928 | 1.378 | -1.407 |
| monotone | 27.802 | 0.928 | 1.375 | -1.404 |
| FCT/TVD | 27.849 | 0.928 | 1.378 | -1.407 |

## 365 天对照

| 方案 | max\|T\| (d365) | max\|u\| peak | max\|eta\| (d365) | mean T_top drift |
|---|---:|---:|---:|---:|
| centered | 27.755 | 1.693 | 1.359 | -1.550 |
| monotone | 27.751 | 1.693 | 1.341 | -1.543 |
| FCT/TVD | 27.753 | 1.693 | 1.357 | -1.549 |

## 气候态打分（365d, 最后 90d 平均）

| 方案 | A1 corr | A1 RMSE | A2 corr | A2 RMSE | 总判定 |
|---|---:|---:|---:|---:|---|
| centered | 0.997 | 1.012 | 0.974 | 2.115 | FAIL |
| monotone | 0.997 | 1.000 | 0.974 | 2.115 | FAIL |
| FCT/TVD | 0.997 | 1.013 | 0.974 | 2.115 | FAIL |

A2 都卡在同一个地方，说明**问题不在 transport 方案本身**。  
下一步不是继续折腾 transport，而是去查垂直坐标 / 地形 / 诊断 / 更大尺度闭合。

## 结果文件

- `results/fct_transport/global_fct_tvd_1deg_30d.npz`
- `results/fct_transport/global_fct_tvd_1deg_365d.npz`
- `results/fct_transport/global_monotone_1deg_365d.npz`
- `results/fct_transport/clim_centered/`
- `results/fct_transport/clim_monotone/`
- `results/fct_transport/clim_fct/`
