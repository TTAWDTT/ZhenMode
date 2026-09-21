# 实验进展报告 — 真实大气强迫带来 A2 通过

**日期：** 2026-09-21  
**阶段：** surface forcing attribution and first passing annual climate run

## 结论一句话
不要继续调 transport 或只盯极区误差；当前最有效的改进是把表面强迫从“纬向均匀 WOA SST 目标”换成真实空间变化的 NCEP 2m 气温。这个 opt-in 改动让 365d A2 RMSE 从 2.115 C 降到 1.883 C，首次同时通过 A1/A2。

## 这次做了什么
1. 对已有 365d 气候态做了空间归因：
   - 闭合南北墙：约 5.6% A2 squared error；
   - 海岸：约 31.6%；
   - 深水开阔大洋：约 62.6%。
2. 证明最大误差点在高纬，但它们数量少，不主导全球分数。
3. 发现 `model - WOA` 与 `T_atm - WOA` 相关约 0.83，`R^2` 约 0.69，指向大气目标太平滑。
4. 跑了 `bulk-lambda-mult = 0.25 / 0.5 / 2.0`，只有 2x 改善 A2 约 4%，低于 5% 门槛，所以不改默认。
5. 新增 `--real-air-temp`，用 NCEP R1 年均 2m 气温作为 bulk target。
6. 365d real-air 实验整体 PASS：
   - A1 RMSE：1.466 C
   - A2 RMSE：1.883 C
   - 相比 baseline 改善：11.0%
   - 热漂移：0.651%
   - 盐漂移：7.37e-6

## 为什么这重要
成熟模式如 MOM6、NEMO、ROMS、HYCOM、MPAS-Ocean 通常使用空间变化的大气状态，而不是只用一个纬向剖面。  
这次的改进正对应调研里的这条核心经验。

## 下一步
1. 用同样配置复跑或确认 real-air 结果。
2. 把 2m 气温扩展到 monthly-varying 动态强迫。
3. 再回头处理海岸/垂直混合，而不是先重构网格。
4. 暂时不要把 real-air target 直接设为默认；等确认后再更新 baseline。

## 相关文件
- `research/experiments/polar_boundary_attribution/`
- `research/experiments/forcing_sensitivity/`
- `research/experiments/real_air_temp/`
- `research/findings.md`
- `research/research-log.md`
