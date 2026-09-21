# A2 误差空间分布初判

用 `results/fct_transport/clim_fct/climatology_compare_g.npz` 看了
`model - WOA` 的平滑 SST 误差。  
结论：transport 不是主导，当前误差主要集中在**高纬度 / 陆地附近**。

## 按纬度带

| 纬度带 | mean error | RMSE | max abs error |
|---|---:|---:|---:|
| 40N..60N | +2.71 C | 6.49 | 15.82 |
| 20N..40N | -4.25 C | 6.08 | 14.54 |
| 0..20N | -3.63 C | 6.08 | 14.59 |
| 20S..0 | -2.73 C | 5.20 | 14.06 |
| 40S..20S | -1.01 C | 2.95 | 13.30 |
| 60S..40S | -0.36 C | 1.97 | 7.14 |

## 最大误差位置

最大的几组误差都在 **59.5N、58.5N、57.5N** 一带，接近闭合的北边界。  
这提示下一步应该先查：

- 高纬边界 / polar cap 处理；
- 陆地附近的 bathymetry / coast mask；
- surface bulk heat flux 的纬向目标是否合理；
- 以及是否缺少海冰 / 冷池类约束。

## 这不改变 budget 结论
体积严格守恒、heat/salt drift 很小，所以 transport 层不是当前气候误差的主要杠杆。
