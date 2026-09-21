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
1. **保守 / 单调 / 有界的 tracer transport**  
   已加入 `--fct-adv`：一个紧凑的 TVD/MUSCL flux-limited horizontal tracer transport。  
   还不是完整的 Zalesak 3D FCT，但已经比裸 centered transport 更稳健。

2. **诊断 / budget 层**  
   在加更多 closure 前，应先有 heat / salt / mass / energy residual 的标准诊断。

3. **把 wet/dry、open/closed face contract 显式化**  
   把它做成可复用的测试不变量，而不是散落在不同算子里的隐含约定。

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

## 当前进展
- 已完成主流海洋模式的横向调研。
- 已加入 `--fct-adv`，实现紧凑 TVD/MUSCL flux-limited horizontal transport。
- `tests/test_fct_advection.py` 已覆盖 uniform conservation 和 boundedness。
- 已完成 30d / 365d 的 centered、monotone、FCT 对照实验。
- 初步结论：FCT 与 centered 差异很小，`max|eta|` 略低，成本相近；暂不设为默认。
- 气候态 A2 的 RMSE 在三种方案里几乎相同，说明 transport 不是当前气候误差的主要瓶颈。
- 新增 heat / salt / volume budget 诊断；365d 显示体积严格守恒，热含量漂移约 0.7%，盐含量漂移约 0.0007%，centered 和 FCT 几乎一致。
