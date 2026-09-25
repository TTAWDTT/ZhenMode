# Ocean Solver 工业级模式对标报告

日期：2026-09-25  
状态：阶段性结论  
对象：当前 `candidate_65n_045_icefloor` baseline 与成熟海洋模式的能力/架构对比

---

## 1. 一句话结论

`ocean_solver` **不是工业级海洋模式**，而是一个结构化、可复现、便于快速做物理/数值 A/B 实验的研究原型。  
它的价值在于“快速试验想法”，不在于替代 MOM6、NEMO、MITgcm、FESOM2、ICON-Ocean 这类成熟模式。

当前最合理的定位是：

> 一个小型、单 GPU、JAX 化的研究沙盒，用来快速检验闭合方案、输运格式和诊断思想；  
> 不是业务化、多 decade、多圈层耦合的工业级模式。

---

## 2. 本报告的对比范围

这份报告做的是 **能力/架构/验证体系对标**，不是“同一硬件、同一强迫、同一网格”的同场竞技 benchmark。

原因很直接：

- 我们的模型目前主要验证 365d SST 气候态；
- 成熟模式通常运行 multi-decade、OMIP/CORE-II/CMIP 类实验；
- 它们包含更完整海冰、耦合器、同化和生态/化学模块；
- 直接把我们的 A2 SST RMSE 与它们的论文指标相减是不公平的。

所以本报告回答三个问题：

1. `ocean_solver` 现在处于什么位置？
2. 成熟模式强在哪里？
3. 如果继续推进，哪些方向最值得借鉴？

---

## 3. 当前 baseline 的事实

`candidate_65n_045_icefloor` 是当前 production-like candidate：

- 0.45° 水平分辨率
- 365d 积分
- `dt=1800s`
- annual real 2m air forcing + seasonal NCEP wind
- WOA 初始场
- ETOPO bathymetry
- FCT/TVD horizontal tracer transport
- localized convection
- GM0
- ice-air floor proxy
- 单 GPU JAX 运行

关键指标：

| 指标 | 数值 |
|---|---:|
| global A2 RMSE | 1.1126 C |
| North Atlantic 40–60N A2 RMSE | 1.0096 C |
| near-wall raw bias | -0.9121 C |
| below-freezing SST cells | 0 |
| 365d wall time | 约 54 min |
| reproducibility | 已复现 |

诊断上界：

| rung | global A2 | NA A2 | near-wall bias |
|---|---:|---:|---:|
| ice floor only | 1.113 C | 1.010 C | -0.912 C |
| + coastal restore tau=10d | 1.060 C | 0.978 C | -0.834 C |
| + coastal restore tau=3d | 1.003 C | 0.939 C | -0.711 C |

这些 coastal restore 结果只能作为误差归因上界，不能当作纯动力 production default。

---

## 4. 与成熟模式的能力对比

下表是粗粒度能力对比：

| 维度 | `ocean_solver` 当前 | 成熟海洋模式 |
|---|---|---|
| 定位 | 研究原型 / A/B 实验沙盒 | 业务、气候、耦合系统中的成熟组件 |
| 水平网格 | structured lat-lon finite difference | structured finite-volume、curvilinear、unstructured mesh、icosahedral mesh 均有 |
| 垂直坐标 | z-level，14 层 | z、z-star、sigma、isopycnal、hybrid、ALE 等更成熟 |
| 时间步进 | split-explicit + RK2/JAX | split-explicit、pressure method、implicit/free-surface variant 更成熟 |
| 示踪输运 | FCT/TVD，主要用于稳定性实验 | FCT、MUSCL、TVD、PPM、QUICKEST 等方案更多，且长期验证 |
| 闭合 | GM/Redi、localized convection、简单 bulk flux | KPP/ePBL、MEKE、backscatter、TKE、IDEMIX、Smagorinsky/Leith 等更完整 |
| 海冰 | ice-air floor proxy | 通常有完整 sea-ice model 和耦合器 |
| 强迫 | NCEP R1 wind + annual 2m air + WOA + ETOPO | OMIP/CORE-II/JRA55-do/ERA5 等标准化强迫更完整 |
| 同化 | 无 | ROMS、HYCOM、FVCOM、部分业务系统有成熟 DA |
| 并行 | 单 GPU，JAX/XLA | MPI/OpenMP、block-structured、unstructured mesh scaling 更成熟 |
| 耦合 | 无完整 coupler | CESM、EC-Earth、ICON、FESOM、NEMO 等有成熟 coupler |
| I/O | 简单 NPZ/诊断输出 | NetCDF/ParallelIO、XIOS、标准化 diagnostic suite |
| 验证 | SST/区域 RMSE、稳定性、budget | OMIP/CORE-II、AMOC、MHT、sea-ice extent、MLD、heat content 等多维验证 |
| 社区生态 | 单项目、小代码库、便于实验 | 数十年维护、大量用户、文档、培训、回归测试 |

这张表的核心不是“我们输得很惨”，而是说明：  
**工业级模式的价值不只在动力学核心，还在完整物理、耦合、验证、软件生态和长期维护。**

---

## 5. 主要成熟模式的强项

### MOM6

MOM6 的强项是：

- generalized vertical coordinate / ALE
- split-explicit time stepping
- modular closure stack
- strong diagnostics
- 在 CESM/NOAA/GFDL 生态中长期使用

对我们最有价值的借鉴：

1. 垂直坐标和 remap 的模块化设计；
2. diagnostics 与 dynamical core 的清晰分离；
3. closure interface 的可插拔性。

### NEMO

NEMO 的强项是：

- operational / climate 双用途
- C-grid finite-difference/finite-volume hybrid
- 多种 tracer advection 方案
- 生态、海冰、耦合等模块成熟

对我们最有价值的借鉴：

1. 多套 tracer advection 方案并存；
2. FCT/MUSCL/TVD 的工程实现；
3. operational 级别的 diagnostic suite。

### MITgcm

MITgcm 的强项是：

- 非常灵活的 finite-volume framework
- hydrostatic / nonhydrostatic 可选
- adjoint 和 inverse modeling 生态强
- package architecture 清晰

对我们最有价值的借鉴：

1. package 化架构；
2. adjoint/parameter estimation 的可能性；
3. 更通用的方程组组织方式。

### FESOM2

FESOM2 的强项是：

- unstructured triangular mesh
- finite-volume discretization
- full 3D FCT limiter
- variable resolution
- 强扩展性

对我们最有价值的借鉴：

1. full 3D FCT limiter；
2. extreme vertical advection 的隐式/稳定化处理；
3. mesh-aware transport 的工程思路。

### ICON-Ocean

ICON-Ocean 的强项是：

- icosahedral / triangular C-grid
- mimetic finite-volume
- local refinement
- GPU support
- coupled climate system

对我们最有价值的借鉴：

1. mimetic / structure-preserving discretization；
2. local refinement 的可行路径；
3. modern coupled model architecture。

### ROMS / FVCOM

ROMS 和 FVCOM 的强项是：

- coastal / regional robustness
- terrain-following or unstructured mesh
- wet/dry, open boundary, nesting
- data assimilation

对我们最有价值的借鉴：

1. coastal boundary discipline；
2. open boundary condition 设计；
3. coastal heat budget 的诊断方法。

---

## 6. 当前模型真正有价值的地方

`ocean_solver` 不应该被评价为“有没有追上 MOM6/NEMO”。  
更有意义的问题是：它能否快速回答某些具体研究问题？

当前确实有价值的地方：

- 代码小，实验改动快；
- JAX/GPU 能快速做 A/B 测试；
- 已有 reproducible candidate；
- 已有 baseline、repeat、metrics、stop-node 记录；
- 已做过 transport、forcing、coastal、ice proxy、GM/Redi、resolution 等系统消融；
- 能快速测试一个闭合是否值得进入更大模型。

所以它的合理角色是：

> **research sandbox for closure/transport experiments**, not production ocean model.

---

## 7. 与工业级模式的主要差距

按影响排序：

### 1. 物理完整度

我们缺：

- 完整海冰
- 完整 mixed-layer model
- realistic surface flux formulation
- rivers / runoff
- tides
- biogeochemistry
- full coupling

### 2. 数值体系

我们缺：

- generalized vertical coordinate
- ALE / remap
- full 3D FCT limiter
- 更成熟的 pressure/free-surface treatment
- partial bottom cells
- more robust boundary-layer treatment

### 3. 验证体系

我们缺：

- OMIP/CORE-II style benchmark
- multi-decade climate metrics
- AMOC / MHT / heat content / sea ice / MLD
- standardized forcing datasets
- published model-vs-observation protocol

### 4. 工程生态

我们缺：

- MPI / multi-GPU scaling
- parallel I/O
- coupler
- standardized config system
- release engineering
- large community test suite

---

## 8. 如果继续推进，优先级建议

不建议目标是“追平工业级模式”。  
建议目标是“把研究原型的强项用对地方”。

如果继续做，优先级建议如下：

### P0：先做外部对标 benchmark，而不是继续调参

建立一个小型 but standardized protocol：

- fixed forcing
- fixed initial condition
- fixed bathymetry
- fixed metrics
- fixed reporting format

哪怕不运行 MOM6/NEMO，也要先让我们的结果能被外部读者理解。

### P1：补完整 sea-ice / mixed-layer 最小闭环

当前 ice-air floor 只是 proxy。  
下一步值得做的是：

- simple thermodynamic sea-ice model
- mixed-layer depth diagnostic
- brine rejection / salt flux proxy
- surface heat flux formulation 升级

这是对 SST / polar / NA 误差最直接的物理改进。

### P2：做 generalized vertical coordinate / ALE 的第一步

不一定马上重写 solver。  
可以先做：

- partial bottom cells
- z-star prototype
- conservative remapping
- topography-pressure diagnostics

这会提高对 narrow seas、coastal topography、deep circulation 的描述能力。

### P3：把 FCT/TVD 从“experimental flag”升级为“validated transport”

当前 FCT 是稳定性工具，但还没有形成完整的 benchmark：

- advection unit tests
- rotation test
- coastal tracer test
- long-run budget closure
- monotonicity check
- energy/heat/salt residual report

### P4：建立 external-model-informed diagnostic suite

不必先跑 MOM6/NEMO。  
可以先把它们的 diagnostic 思想搬过来：

- AMOC
- MHT
- MLD
- sea-ice extent
- heat content
- regional SST/SSS bias
- circulation strength
- tracer budget residuals

---

## 9. 不建议现在做的事情

不建议现在追这些：

1. 直接写成 unstructured mesh
2. 立刻做 nonhydrostatic dynamics
3. 立刻做 full biogeochemistry
4. 立刻做 full coupled climate model
5. 追求和 MOM6/NEMO/FESOM2 同级别的工程生态

这些会消耗大量时间，而且不会立刻改善当前研究问题的判断力。

---

## 10. 总体评价

`ocean_solver` 是一个合格的研究原型，但不是工业级模式。

它现在最有价值的地方不是“比 MOM6/NEMO/FESOM2 更强”，而是：

- 小
- 快
- 可复现
- 适合做闭合实验
- 能快速判断一个物理/数值想法是否值得迁移到更大模型

所以下一步应该停止“无尽头调参”，转而做两件事：

1. **对外报告清楚它是什么、不是什么。**
2. **用一个标准化 benchmark protocol 把它的结果放进更大的坐标系。**

这比继续内部迭代更有价值。
