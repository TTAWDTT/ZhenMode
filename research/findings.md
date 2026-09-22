# 调研结论

## 研究问题
目前广泛使用的海洋环流模式里，哪些数值方法、物理参数化、软件组织方式最能支撑“可靠、可扩展、科学上可信”？  
`ocean_solver` 下一步应该优先借鉴哪些？

## 当前理解
最强的可迁移信号不是一个特别炫的方案，而是一整套工程体系：

- 保守运输
- 严格的湿/干与开/闭面控制
- 泛化垂直坐标
- 模块化参数化
- 一等公民的诊断与预算检查

这些模式在网格和坐标系选择上差异很大，但在上述原则上是收敛的。

## 已调研模式

| 模式 | 网格 / 垂直坐标 | 时间积分 | 运输 / 闭合方案 | 价值 |
|---|---|---|---|---|
| MOM6 | 结构网格，泛化垂直坐标 | split explicit + ALE | GM/Redi、MEKE、KPP/ePBL 等 | 现代全球 OGCM 中最直接相关 |
| MITgcm | 曲线坐标 / 有限体积 | pressure method + 隐式 / 自由表面变体 | 包架构丰富，支持 adjoint | 数值通用性和生态最好 |
| NEMO | 正交曲线坐标 | split-explicit free surface | CEN/FCT/MUSCL/UBS/QUICKEST；TKE/OSM/GLS | 成熟欧洲社区模式 |
| ROMS | 曲线坐标 + sigma | split explicit，FB AB3-AM4 或 LF-AM3 | MPDATA/TVD；KPP/MY2.5/GLS | 沿海 / 区域模式稳健性强 |
| POP2 / CESM | 结构 lat-lon，z | 隐式自由表面 barotropic solve | GM/Redi、KPP、harmonic/biharmonic | 成熟 CESM 海洋分量，现正被 MOM6 接替 |
| MPAS-Ocean | 非结构 Voronoi | split explicit | ALE、monotone transport、GM/Redi、Leith、CVMix | 可变分辨率、可扩展 |
| FESOM2 | 非结构三角网格 | split explicit | 全 3D FCT、隐式垂直平流稳定化、等中性扩散 | 扩展性和运输稳健性突出 |
| ICON-O | icosahedral-triangular C-grid | 显式为主 + 局部隐式 | GM、谐波/双谐波、Smagorinsky/Leith、TKE、IDEMIX | 现代 mimetic 离散，支持 GPU |
| HYCOM | 混合 isopycnal/z/sigma | hybrid-coordinate treatment | KPP、Kraus-Turner、Mellor-Yamada、PWP | 业务化预报系统，带 DA |
| FVCOM | 非结构三角网格 | 显式 / 半隐式 | MPDATA/TVD、湿干处理、嵌套 | 沿海 / 河口生产级模式 |

## 数值共性

成熟海洋模式的稳健性来自组合拳：

1. 保守的有限体积 / 有限元算子
2. 明确的隐式或 split-explicit 自由表面处理
3. 保守 / 单调的 tracer transport
4. 严格的湿/干与开/闭面 mask
5. 灵活的垂直坐标处理
6. 模块化闭合方案
7. 独立的诊断与预算层

## 迁移启示

- **运输稳健性不只是“中心差分 vs 单调”。**  
  FESOM2 和 NEMO 都常用低阶兜底 + 高阶基础方案，再配 limiter。  
  FESOM2 对全 3D 通量做 limiter，特别值得借鉴，因为近岸的垂直平流常常是实际瓶颈。

- **垂直坐标柔性在最强模式里非常常见。**  
  MOM6、MPAS、FESOM2、ICON-O、HYCOM 都支持泛化垂直坐标或 ALE/remap。  
  这是处理混合层、地形和窄海的成熟路径，不需要让 transport 代码绑死在单一 z 网格上。

- **闭合方案是共性的，不是特殊的。**  
  KPP、GM/Redi、harmonic/biharmonic、Smagorinsky/Leith、TKE、IDEMIX、内潮混合反复出现。  
  真正重要的是干净的闭合接口，而不是一次性塞进更多方案。

- **边界纪律是隐藏地基。**  
  无通量墙、湿/干 mask、开/闭面、halo exchange、partial cells 都是核心不变量。  
  这正是 `ocean_solver` 已经多次踩坑的地方，必须继续当作一等公民处理。

- **诊断不是可选的。**  
  MITgcm、MOM6、NEMO、ICON、MPAS、POP、FESOM 都把 diagnostics 和 dynamical core 分开。  
  这样才可以在不碰核心求解器的情况下测试 budgets、能量、运输和参数化。

## 对 ocean_solver 的建议

### 近期优先
1. **真实大气强迫**  
   已加入 `--real-air-temp`，用 NCEP R1 年均 2m 气温替代“纬向均匀 WOA SST 目标”。  
   这是 MOM6 / NEMO / ROMS / HYCOM 一类业务模式的共同做法：先有空间变化的大气状态，再谈更复杂的闭合。

2. **回到海岸误差和垂直混合**  
   monthly forcing 已经实现：稳定 PASS，但没有超过 annual real air。  
   这说明当前下一杠杆更可能在近岸 mask / 浅水 / 垂直混合。

3. **把 wet/dry、open/closed face contract 显式化**  
   把它做成可复用的测试不变量，而不是散落在不同算子里的隐含约定。

4. **继续保留 budget / 误差归因层**  
   不要看到最大误差点就调参；先看 squared error 的空间归属。

### 中期
- 从 conservative remap 或 z-star 开始引入垂直坐标柔性，不要直接重写成 unstructured mesh。
- 可以考虑对极端垂直 Courant 数做隐式垂直平流兜底。

### 后期
- 如果需要多 GPU，先考虑 JAX sharding / `pjit`，不要立刻跳到 unstructured mesh。
- 只有当结构网格上的 transport、mask、诊断体系足够成熟，再考虑非结构网格。

## 暂不建议做
- 现在就转向 unstructured mesh。
- 现在就加 nonhydrostatic 动力。
- 在 transport / diagnostics 还不完善前继续堆 closure。

## 开放问题
- 最小的 ALE / z-star remap 层是什么，才能帮 `ocean_solver` 而不破坏当前 1° baseline？
- 哪种 tracer transport limiter 在 JAX 里最容易正确实现？
- 现有 wet/dry face contract 能否被改造成可复用的测试不变量？
- 当前性能里，多少来自 JAX whole-step compilation？如果把代码拆得更模块化，会损失多少？

## 最新归因

用已有 365d 气候态做了空间归因：

- 闭合南北墙附近只占 A2 squared error 的约 **5.6%**。
- 海岸格点约 **31.6%**。
- 深水开阔大洋约 **62.6%**。
- 最大误差确实在高纬海岸，但数量太少，不足以解释全球 A2 失败。
- `model - WOA` 与 `T_atm - WOA` 的相关系数约 **0.83**，`R^2` 约 **0.69**。

这说明当时真正的大问题不是 transport，也不是极区墙，而是**表面大气强迫太理想化**。

随后做了 λ 扫描：

| bulk lambda | A1 RMSE | A2 RMSE |
|---|---:|---:|
| 0.25x | 3.104 C | 3.387 C |
| 0.5x | 1.835 C | 2.482 C |
| 1.0x baseline | 1.013 C | 2.115 C |
| 2.0x | 0.504 C | 2.031 C |

2.0x 只改善 A2 约 4%，低于预注册的 5% 门槛，所以不改默认值。

接着加入 NCEP R1 年均 2m 气温作为 opt-in 大气目标：

| run | A1 corr / RMSE | A2 corr / RMSE | 总判定 |
|---|---:|---:|---|
| zonal WOA SST baseline | 0.997 / 1.013 C | 0.974 / 2.115 C | FAIL |
| annual NCEP 2m air | 0.997 / 1.466 C | 0.988 / 1.883 C | PASS |

这是当前第一个 365d A1/A2 同时通过的气候态实验。

## 当前进展
- 已完成主流海洋模式的横向调研。
- 已加入 `--fct-adv`，实现紧凑 TVD/MUSCL flux-limited horizontal transport。
- `tests/test_fct_advection.py` 已覆盖 uniform conservation 和 boundedness。
- 已完成 30d / 365d 的 centered、monotone、FCT 对照实验。
- 初步结论：FCT 与 centered 差异很小，`max|eta|` 略低，成本相近；暂不设为默认。
- 气候态 A2 的 RMSE 在三种方案里几乎相同，说明 transport 不是当前气候误差的主要瓶颈。
- 新增 heat / salt / volume budget 诊断；365d 显示体积严格守恒，热含量漂移约 0.7%，盐含量漂移约 0.0007%，centered 和 FCT 几乎一致。
- 完成 polar / boundary 归因：闭合边界不是主要 SSE 来源；深水开阔大洋和海岸更重要。
- 完成 bulk lambda 0.25 / 0.5 / 2.0 对照；均稳定，但没有达到 A2 改善 5% 的门槛。
- 新增 `--real-air-temp` 和 NCEP R1 2m 气温缓存；`tests/test_air_reanalysis.py` 通过，全量测试 151 passed。
- 新增 365d annual-mean NCEP 2m air 实验；A2 RMSE 从 2.115 C 降到 1.883 C，首次整体 PASS。
- 复跑 annual real air：A2 RMSE 1.886 C，确认稳定。
- 新增 opt-in monthly NCEP 2m air forcing；365d 稳定 PASS，但 A2 RMSE 1.990 C，未超过 annual real air，因此 annual real air 仍是更好的简单 baseline。
- 完成 annual real air 的 coastal / vertical 归因：raw 全球 RMSE 1.665 C，平均 bias -1.211 C；深水占 raw SSE 73.9%，海岸占 17.2%，但海岸 RMSE 2.05 C。
- WOA 表面-50m 层结越强，模式 SST 越偏冷；这指向垂直混合 / mixed layer 作为下一个实验方向。
- 完成 vertical mixing 敏感性：`kappa_v=1e-6` 且 `kappa_conv=0.01` 的组合把 A2 RMSE 降到 1.848 C，复跑 1.844 C，均 PASS。
- 但最强层结冷 bias 仅改善约 0.10 C，低于 0.2 C 的预注册门槛，所以垂直混合是 suggestive 而非 decisive。
- 区域审计显示降低垂直混合使全球 RMSE 从 1.665 C 降到 1.625 C，海岸改善 6.46%，强层结分位改善 5.98%，但 81% 格点整体变暖，说明这不是高度定向的修复。
- 当前最大的剩余误差集中在 `300..360E / 40..60N`。后续审计确认该区域主要是 cold bias，而不是早期粗看得到的 warm bias。

## 北边界 / 极帽诊断（2026-09-21）

在降低垂直混合和 annual real air 基础上，把闭合北墙从 60N 扩到 65N：

| run | A1 RMSE | A2 RMSE | 北大西洋 A2 bias | 北大西洋 A2 RMSE |
|---|---:|---:|---:|---:|
| 60N candidate | 1.420 C | 1.844 C | -2.044 C | 2.811 C |
| 65N | 1.457 C | 1.770 C | -1.703 C | 2.177 C |
| 65N repeat | 1.463 C | 1.773 C | -1.700 C | 2.175 C |
| wider polar cap | 1.435 C | 1.964 C | -2.572 C | 3.592 C |

结论：

- 60N 闭合墙是北大西洋冷偏差的重要来源；65N 扩域使 40..60N A2 RMSE 稳定降低约 22.6%。
- 全球 A2 只改善 3.85..4.01%，低于 5% 门槛，所以先作为 diagnostic candidate，不直接改成 production default。
- 加宽极帽使目标区更冷、全球 A2 更差，不是当前补救方向。
- 90d no-cap probe 稳定，但没有 365d 证据，暂不替代 65N 诊断基线。
- 剩余高纬冷偏差指向缺失的高纬表面物理；下一步先诊断 surface heat flux / sea-ice proxy，而不是继续调 transport。

## 65N 表面热通量敏感性（2026-09-21）

在 65N 候选基线上只改 `lambda_bulk`：

| run | A1 RMSE | A2 RMSE | 北大西洋 A2 bias | 北大西洋 A2 RMSE |
|---|---:|---:|---:|---:|
| lambda20 | 2.189 C | 2.362 C | -2.583 C | 3.128 C |
| lambda40 | 1.463 C | 1.773 C | -1.700 C | 2.175 C |
| lambda80 | 1.130 C | 1.530 C | -1.052 C | 1.572 C |

结论：

- 表面热交换时间尺度是当前残余冷偏差的一阶控制因子。
- `lambda80` 使全球 A2 改善 13.7%，北大西洋 RMSE 改善 27.7%，并消除该区域冻结格点。
- 但它只是标量恢复强度候选，不是机制修复，也不直接改 production default。
- λ80 后残余冷误差仍集中在最强正热输入分位，说明下一个问题更可能在海洋热输送 / 高纬水团结构，而不是 sea-ice flux cap。
- 下一步固定 `lambda_bulk=80`，诊断 GM / bolus heat transport 和 mixed-layer closure。

## GM 热输送与混合层诊断（2026-09-21）

固定 `65N + lambda_bulk=80 + reduced vertical mixing` 后先扫 GM：

| run | A2 RMSE | 北大西洋 A2 bias | 北大西洋 A2 RMSE |
|---|---:|---:|---:|
| GM0 | 1.419 C | -0.788 C | 1.338 C |
| GM500 | 1.449 C | -0.883 C | 1.417 C |
| GM1000 | 1.530 C | -1.052 C | 1.572 C |
| GM3000 | 2.089 C | -1.793 C | 2.346 C |

结论是原 GM 强度偏大；`GM500` 是保留物理闭包时更好的诊断候选。

随后固定 `GM500` 检查混合层/对流闭包：

| run | A2 RMSE | 北大西洋 A2 bias | 北大西洋 A2 RMSE |
|---|---:|---:|---:|
| column conv, kconv=0.01 | 1.449 C | -0.883 C | 1.417 C |
| localized conv | 1.410 C | -0.713 C | 1.306 C |
| localized conv repeat | 1.410 C | -0.713 C | 1.306 C |
| kconv=0.05 | 1.446 C | -0.888 C | 1.422 C |
| kconv=0.002 | 1.455 C | -0.867 C | 1.402 C |

局地化对流把全球 A2 再改善 2.71%，北大西洋 RMSE 再改善 7.84%，且复跑几乎完全一致。相比之下，继续改 `kappa_conv` 标量收益很小。当前最佳诊断候选是：

`65N + lambda_bulk=80 + kappa_gm=500 + localized convective adjustment + annual real air`

这一步是闭包结构改进，但仍先保持 opt-in，不等同 production default。下一步应跑 3D heat-tendency decomposition，把残余误差拆成 advection、convection、GM、diffusion 和 surface flux。

## 3D 热倾向分解（2026-09-21）

用当前最佳候选 `65N + lambda80 + GM500 + localized convection + annual real air` 跑 365d，最后 90d 做深度分项归因。全球 A2 RMSE 为 `1.412 C`。

在北大西洋 `40..60N` 表层：

| 项 | mean K/day | mean share |
|---|---:|---:|
| convection | +0.231 | 0.465 |
| surface bulk flux | +0.179 | 0.360 |
| advection | -0.069 | 0.139 |
| GM | +0.017 | 0.034 |
| diffusion | ~0 | ~0 |

近墙 `55..60N` 更强：convection `+0.432 K/day`，bulk flux `+0.202 K/day`，advection `-0.112 K/day`。

关键解释：

- 表面热通量不是残余冷偏差的原因；冷格点反而收到更多 bulk 增暖。
- 扩散几乎可以忽略。
- 平流是明确的表层冷却项； colder cells 的平流冷却更强，相关系数约 0.43，近墙约 0.51。
- 对流在表层是增暖、在次表层是冷却，说明它在把热往下搬，而不是直接制造表层冷偏差。
- 因此下一步不是继续调 scalar，而是检查水平热输送和边界流结构。

## 水平黏性与 SSS restoring（2026-09-22）

在当前最佳候选上继续排除两个候选机制：

| 项 | 全球 A2 RMSE | 北大西洋 A2 RMSE |
|---|---:|---:|
| baseline nu_h=5e6 | 1.412 C | 1.313 C |
| nu_h=2.5e6 | 1.415 C | 1.341 C |
| nu_h=1e6 | 1.422 C | 1.375 C |
| SSS restoring 30d | 1.409 C | 1.308 C |
| SSS restoring 90d | 1.409 C | 1.305 C |

结论：

- 降低水平黏性增强了流速，但没有改善冷偏差，反而略差。
- SSS restoring 只有极小改善，不是主要控制因子。
- 剩余误差更可能受水平热输送、边界流几何或分辨率限制。

## 分辨率诊断（2026-09-22）

0.8° 运行显著优于 1° 候选：

| run | 全球 A2 RMSE | 北大西洋 A2 RMSE |
|---|---:|---:|
| 1.0° | 1.412 C | 1.313 C |
| 0.8° | 1.360 C | 1.252 C |

0.8° 使全球 A2 改善约 3.6%，北大西洋 RMSE 改善约 4.7%，近墙 RMSE 从 1.666C 降到 1.421C。0.5° 在当前候选物理下 10 天内 NaN，暂不可用。

0.7° 进一步优于 0.8°：全球 A2 `1.337C`，北大西洋 RMSE `1.206C`，近墙 RMSE `1.215C`。相对 1°，全球 A2 改善 5.30%，北大西洋 RMSE 改善 8.14%。因此 0.7° 是当前新的首选诊断分辨率。

0.65° 也 NaN，因此 0.7° 是当前最细的稳定诊断分辨率。

在 0.7° 上关闭 GM 进一步改善：全球 A2 `1.336C`，北大西洋 RMSE `1.043C`。当前新的最佳诊断候选是 `65N + lambda80 + GM0 + localized convection + annual real air at 0.7°`。

## Candidate baseline freeze

The `0.7-degree + GM0 + localized convection + lambda80 + annual real air`
configuration is now locked as `candidate_65n_07_gm0` in
`scripts/run_candidate_baseline.sh`. It is the new diagnostic baseline for
follow-on comparisons; the old 1-degree run remains the historical default.

## 近墙温度梯度诊断（2026-09-22）

在 0.7° 基线上比较 `GM0` 和 `GM500` 最后 90 天的 SST 梯度、速度-梯度对齐度和完整求解器平流项：

| run | 区域 | SST bias | SST RMSE | 平均流速 | 保存的平流项 |
|---|---|---:|---:|---:|---:|
| GM0 | 40..60N | -0.832 C | 1.052 C | 0.0814 m/s | -0.0572 K/day |
| GM500 | 40..60N | -0.911 C | 1.138 C | 0.0817 m/s | -0.0720 K/day |
| GM0 | 55..60N 近墙 | -1.081 C | 1.186 C | 0.0416 m/s | -0.0377 K/day |
| GM500 | 55..60N 近墙 | -1.146 C | 1.246 C | 0.0418 m/s | -0.0752 K/day |

结论：

- GM500 在近墙把完整平流冷却加强约 `0.038 K/day`，这是它区域误差更大的直接来源。
- 但两个 run 的近墙平均温度梯度和流速几乎相同，说明这不是边界流整体变强/变弱的问题。
- 最冷的近墙格点集中在 `300..302E / 57..59N` 和 `313..315E / 59.45N`；部分冷点平均流速只有 `0.006..0.010 m/s`。
- 因此这不是简单的“边界流速错了”，更像局地墙体几何、 bathymetry/land mask 或垂直热再分配的问题。
- 下一步先对最冷近墙格点做表层到 200 m 的热倾向分解，再看是否需要改北边界条件或高纬表面物理。

## 最冷近墙格点的垂直归因（2026-09-22）

对 GM0 和 GM500 最后 90 天各取 100 个最冷近墙格点，分解表面和次表层热倾向。表面节点（z=0）结果是：

| run | 组 | advection | convection | surface bulk |
|---|---|---:|---:|---:|
| GM0 | 全部近墙 | -0.038 K/day | +0.455 K/day | +0.134 K/day |
| GM0 | 100 个最冷格点 | -0.058 K/day | +0.612 K/day | +0.096 K/day |
| GM500 | 全部近墙 | -0.075 K/day | +0.281 K/day | +0.156 K/day |
| GM500 | 100 个最冷格点 | -0.187 K/day | +0.251 K/day | +0.222 K/day |

结论：

- GM500 在最冷格点造成明显更强的局地平流冷却；这解释了关 GM 的区域收益。
- 但 GM0 的最冷格点平均表面总趋势仍是增暖，所以冷偏差不是由这些格点当前的局地净冷却项直接造成的。
- 50..200 m 的对流/平流都很小，说明“垂直热搬走过强”不是第一嫌疑。
- 最冷格点是混合群体：一部分贴岸，另一部分在 59..60N 弱通风区。因此下一步要把贴岸几何和高纬通风分开看。

## 贴岸 vs 高纬通风诊断（2026-09-22）

把近墙区域按离陆距离分成贴岸（<=3 格）和内部（>3 格）后比较：

| run | 区域 | cells | SST bias | SST RMSE | 表面净趋势 |
|---|---|---:|---:|---:|---:|
| GM0 | 全部近墙 | 548 | -1.081 C | 1.186 C | +0.552 K/day |
| GM0 | 贴岸 <=3 格 | 122 | -1.453 C | 1.539 C | +0.778 K/day |
| GM0 | 内部 >3 格 | 426 | -0.976 C | 1.065 C | +0.488 K/day |
| GM0 | 59..60N 内部 | 56 | -1.197 C | 1.246 C | +0.471 K/day |
| GM500 | 内部 >3 格 | 426 | -1.061 C | 1.151 C | +0.377 K/day |

结论：

- 贴岸格点仍是更强的误差源，RMSE 明显高于内部。
- 剔除贴岸格点后，`59..60N` 内部冷区仍然存在，但强度和数量都小于贴岸组。
- GM500 会让这个内部区更冷，主要又是平流冷却加强。
- 因此下一步先做贴岸 mask/ventilation 诊断；高纬 ice-proxy 只作为后续窄范围实验。

## 近墙贴岸/水深归因（2026-09-22）

在 GM0 最后 90 天的近墙 `55..60N` 区域，按离陆距离分箱：

| group | cells | RMSE | SSE share | cold>1C share | mean depth |
|---|---:|---:|---:|---:|---:|
| land 0..3 cells | 122 | 1.539 C | 0.371 | 0.836 | 1155 m |
| land 4..7 cells | 238 | 1.252 C | 0.482 | 0.706 | 2506 m |
| land >=8 cells | 188 | 0.770 C | 0.146 | 0.218 | 2334 m |

结论：

- 冷偏差主要来自贴岸和过渡带（离陆 0..7 格），不是仅浅水。
- `0..3` 格带贡献 37% 的近墙 SSE，`4..7` 格带贡献 48%。
- 如果把 `0..3` 格带误差清零，GM0 近墙 RMSE 从 `1.186C` 降到 `0.940C`。
- 贴岸组平均深度 `1155m`，说明这不只是浅水掩膜问题。
- 高纬内部冷区仍然存在，但相对贴岸/过渡带是次要问题。
