# main 与修复分支：客观对照

Lady，2026-09-30。本轮仅分支整理、源码/保存证据复核和同测试回归；
未开启Goal、未push、未新增CPU/GPU数值实验或生产积分。

## 版本与总判断

- `main` / 已查验的GitHub `main`：`bb7ba235eed5706b0adf86472e95b46bc4c021b1`。
- `ttawdtt/core-repair-review`的被测代码：`066f827bbfba67f6e3e6d119057e5bb833ac2f30`。
- 全部130个本地提交保留在分支，不squash、不改写；本地main对齐origin/main。
  本报告的后续文档提交不计入被测代码版本或下表差异统计。

**局部正确性与拒绝错误证据的能力确有进步，但整模式资格和交付效率不足。
不能称为完成去冗余、实现可靠百年积分或达到工业级。**
目前既不能以候选失败否定旧生产方法，也不能以算子通过证明候选优于main。

## 本轮相同测试的直接比较

将两个提交的`src/`、`tests/`和`pyproject.toml`各自导出到独立快照。
仅在main测试目录放入分支的三份补充回归文件，main源码未改。
八份实际执行的测试文件在两份快照中SHA256逐一相同；另记录导入模块的
绝对路径，确认两次运行分别加载各自快照而非本机editable安装的另一版本。
环境均为原conda、原生Windows CPU、Python 3.12/JAX 0.11.2；没有GPU实验。

| 同一测试组 | main | 分支 | 可以推出什么 |
| --- | --- | --- | --- |
| 未改动的5份旧测试：黏性CFL、垂向边界、冰盐、混合层冰、manifest | 18通过 | 18通过 | 这些旧控制未退化，不是全部旧轨迹相容性 |
| 评分门禁 | 4通过、14失败 | 18通过 | 缺预算、Inf、偏差穿零、覆盖/评分域/参照身份不一致的拒绝更可靠 |
| 面积加权、湿点缺值、常量相关系数3个评分控制 | 1通过、2失败 | 3通过 | 评分定义及覆盖声明改善，不是模式气候精度提高 |
| 表面热、沿岸扩散、相变/融冰盐、完整步源计入及局地导数 | 4通过、9失败 | 13通过 | 指定控制上的离散预算和源处理修复 |
| 合计 | 27通过、25失败 | 52通过 | 抽样局部证据；不是全库或工业验收 |

25个失败里23个是断言不满足，2个是main缺少新报告字段
`qualification_scope` / `coverage_complete`的KeyError；后两项单列为接口改进，
不能说成两个数值bug。参数化失败也不能当作25个独立根因。
这组测试特意针对已有修复和旧兼容控制，并非随机样本或完整旧套件；
未以main缺少新候选模块制造导入失败来充当优越性证据。

最直观的物理例子：同一20m混合层、5m首节点、100W/m²热输入，
main按节点库存只计入25W/m²，分支满足100W/m²；完整海冰步的旧热源
重复计入也在相同控制上被检出并修复。两个结论只覆盖测试声明的状态/合同。
面积评分控制中main偏差为1.5、分支为1.0，是评分权重的改变，
不能当作SST真的改善0.5°C；新旧指标定义不可混用。

原始快照、模块路径、日志、JUnit保存在本地忽略目录
`results/branch_comparison_20260930/`；本轮ruff与diff-check通过。
没有把这些短测试的耗时作为吞吐或GPU加速比。

### 重放清单

每份快照内用同一conda执行`python -m pytest -q --tb=short`，参数为：

```text
tests/test_nu_nsub_cfl.py
tests/test_vertical_bc.py
tests/test_ice_salt_flux.py
tests/test_benchmark_manifest.py
tests/test_mixed_layer_ice.py
tests/test_benchmark_gate.py
tests/test_benchmark_metrics.py::test_snapshot_scores_wet_area_not_number_of_cells
tests/test_benchmark_metrics.py::test_wet_missing_value_is_not_removed_to_improve_score
tests/test_benchmark_metrics.py::test_constant_patterns_do_not_get_a_perfect_roundoff_correlation
tests/test_surface_energy.py::test_surface_heat_input_matches_column_budget
tests/test_surface_energy.py::test_local_horizontal_diffusion_conserves_heat
tests/test_surface_energy.py::test_ice_closure_conserves_water_ice_enthalpy
tests/test_surface_energy.py::test_complete_melt_salt_flux_is_bounded_by_available_ice
tests/test_surface_energy.py::test_full_step_applies_surface_heat_only_once_with_ice
tests/test_surface_energy.py::test_local_diffusion_budget_with_meridional_gradients_and_land
tests/test_surface_energy.py::test_surface_operator_is_differentiable_away_from_phase_boundary
```

快照zip通过`git archive <被测提交> src tests pyproject.toml`生成；main中的
三份补充文件是`test_benchmark_gate.py`、`test_benchmark_metrics.py`和
`test_surface_energy.py`，均取被测分支提交。未选择的测试不计入通过数量。
本轮main/branch JUnit SHA256分别为
`67ca9fa1488ff1ed4963ef33b4593aa393bfe239857c136a6a3439ab791cd4e8` /
`a5d896318e9bb8fc5660de9f2e59eb56ee45f3e1be4e81d9c7fe9a396835b4dd`。

## 保存积分与候选证据：进步的上限

以下是历史冻结版本的结果，本轮只读；不是066f827重新完成了积分。
main与最终分支尚无同输入、同物理、同精度、同时间窗的完整长期对照，
因此不能计算可靠积分年数、气候误差、预报技巧或吞吐的改善比例。

| 项目 | 已有证据 | 尚未完成 / 不能推出 |
| --- | --- | --- |
| 默认标量扩散 | 同网格算子净/绝对预算的沿岸旧值约−6.90e−6，新值约5.27e−18；空间制造解收敛比4.116/4.054 | 是算子舍入级预算，不是完整模式热盐闭合或气候误差 |
| 原FD显式材料候选 | 修复后2°、固定1月真实输入、600/300s配置分别完成30天及移动库存审计 | 不是main的同配置长期对照；不是季节、百年或全域资格 |
| 1°材料候选 | 600/300s配置在第176/353次尝试因128子步容量拒绝 | 256容量的单独诊断下一步仍出现负顶厚度，未解决几何范围问题 |
| completed-bed弱式研究 | 指定压力精度/瞬时功、静止、内容/源及完整1°规定tracer阶段控制通过；流式装配解决对应显存失败 | 是新局部状态合同，尚未接完整海洋动力步或生产工厂；不是main直接A/B |
| 弱式时间候选 | 平滑Fourier控制约二阶、CPU/GPU一致 | 移动有符号0.773/1.156、脉冲0.623/0.858，均未达到原1.9门槛；根因尚未隔离 |
| 可微 / 重启 | 局地导数、受控回滚和特定重启控制有正证据 | 真实对流跨门控梯度失败、真实30天跨进程原字节门槛失败继续保留 |

证据位置：

- [默认扩散前后算子对照](../research/experiments/conservative_tracer_diffusion/review.md)。
- [实施记录](legacy_core_repair_status_zh.md)：§21–23物理/分辨率/负厚度，
  §27–28跨进程字节差，§35压力精度，§36容量，§37时间失败。
- 本机原件：`results/legacy_repair/m4_spatial_repair_matrix_C_20260930T005600Z/combined_validation_final.json`、
  `results/legacy_repair/m4_thin_top_C_20260930T074500Z/validation_final.json`、
  `results/legacy_repair/m2_rstar_weak_time_C_20260930/saved_time_verification.json`。

源hash“符合当前”的历史旗标仅指验收时冻结源；后续清理不自动升级证据。
多数大数组和部分重放脚本位于忽略的results目录，保存于本机不等于
他人仅clone分支即可复现全部实验，这是交付可复现性仍欠缺的一部分。

## 复杂度与推进效率：没有达成原始删减目标

对被测两个提交的完整差异是224个文件、增加31,633行、删除644行，
净增30,989行。这里只是Git行数，不是缺陷数、有效代码量或开发时间。

| 范围 | main | 分支 |
| --- | ---: | ---: |
| src跟踪文件 | 24 | 35 |
| 主求解器行数 | 2,599 | 3,184 |
| tests跟踪文件 | 25 | 74 |
| docs跟踪文件 | 21 | 28 |
| research跟踪文件 | 320 | 447 |

差异中src净增3,804行、tests净增9,944行、docs净增3,917行、research净增
12,974行；130条提交里85条以research/docs开头。测试/独立反证是有价值的，
但这些统计加上生产候选未整体验收，不能支持“代码更精简、生产主线已完成”。
当前CLI仍调用原FD工厂，不导入材料/FV/C-grid/r-star候选；工厂几何和过程
默认仍是legacy。默认未被新研究替换，不意味着相对main所有数值实现未改。

**主要推进问题是研究路径扩张快于整体验收闭环。** 已修局部问题是真实收益，
失败隔离也有价值；但新模块、新状态合同与大量阶段记录增加理解/维护成本，
却没有交付新的合格1°完整积分路径。最近弱式工作仍是局部模块，不能用
其中的绿测试回应用户对完整模式的诉求。没有证据证明全分支适合直接合并main。

## 后续止损边界（建议，不是自动执行授权）

1. 保持main为可回退基线；生产必要修复与未验收研究分开评审，不整包合并130条。
2. 下一项只定位原移动限幅失败：保持原域/初值/门槛，最小阶段轨迹区分
   时变支持、界/限幅切换与阶段一致性；根因及反证明确前不加新核心。
3. 后续每项必须给出同一失败案例的前后结果和旧路径回归；算子、完整步、
   真实积分、气候/预报评分分别结账，不再用提交数或测试数代替交付。
4. 同条件生产A/B、长期与独立评分未完成之前，不承诺超越工业模式；
   保留所有失败原件，不放宽门槛、换域/初值或把负证据当冗余删除。

本轮没有执行上述数值诊断或扩展计划，等待用户下一条安排。

## 二次审查：生产缺口与时间候选的理论前提

依据用户后续“再仔细审查+调研”请求开展；仍不改数值内核、不做新的
CPU/GPU海洋积分、不push。核对FD主步/工厂、CLI判定/强迫/重启、严格
验证脚本、弱式Euler/Heun/限幅和已有时间探针，并重新查阅下列一手资料。
不是声称本轮重新逐行审查了全部224个差异文件。

### 已复现的四类工程缺口

使用已有生产重启测试夹具，仅将动力推进换成指定状态的桩；不下载数据、
不调用动力时间步。可重放代码是
`research/reviews/test_driver_contract_review.py`，本机结果为
`scratch/branch_audit_20260930/review_contract_final.log/.xml`。

| 优先级 | 代码位置 | 复现结果与影响 |
| --- | --- | --- |
| P1 | `src/run_long_integration_global.py:1250` | 检查在整个snapshot批次之后；默认间隔10天。两步桩中第1步u=11、第2步u=0，仍PASS且peak=0；第1步T为NaN、第2步恢复有限，也PASS。无法证明每步均未越界或标定首失败步。 |
| P1 | `src/run_long_integration_global.py:1181` | maxu只取u，不看v。u=0、v恒为11m/s，8个桩步后PASS且max_u_peak=0。原字段确实只声明max|u|，这里指出水平速度监测合同不完整，不伪称违反原仅u统计定义。 |
| P2 | `src/jax_solver_global.py:2626`、`src/run_long_integration_global.py:620` | 工厂接受dt=0/-60；CLI接受days=-1，未推进任何步却输出PASS及days=[0]。restart单独校验正dt，但普通工厂/CLI没有统一入口检查。 |
| P2 | `src/run_long_integration_global.py:1299`、`:1402` | u恒为11的桩确实得到FAIL_BLOWUP，但main正常返回；模块入口只是main()，安装entry point也获得None，CLI失败不会转成非零进程状态。只读退出码的调度器可误报成功。 |

直接重放命令：

```bash
python -m pytest research/reviews/test_driver_contract_review.py -q -s --tb=short
```

7个反例断言当前均失败，这是缺口见证，不是声称修复后的绿回归；该文件
不在默认tests目录内，不把它混入已有通过数量。应在实际修复时提升到正式
回归套件。瞬时异常的桩只检验监测逻辑，未证明真实求解器会从NaN恢复。
上述判定模式在远端main也存在，不能说是130条提交新引入的数值退化，
也不能据此宣布用户历史百年轨迹无效；需要原轨迹证据另行核验。

最重要的工程断层：`scripts/verify_debug_integration.py:57`已有逐步六字段、
u/v双分量、首拒绝与批内峰值监测，生产CLI却没有复用这些性质。
修复应复用状态监测逻辑而非引入新动力内核；不能让安装后的driver反向
依赖测试或scripts路径。GPU批监测应尽量在设备端完成，不每步拷回六个大场。
补全判定须版本化输出语义，保留旧报告，不倒改旧PASS或降低原10m/s、15m线。

另有只读确认的强迫来源风险：CLI真实空气温度抓取失败时会打印警告并
退回WOA纬向温度（`:865`、`:880`），NPZ config却仅保留requested
`real_air_temp*`开关（`:1347`），未保存实际`T_atm_source`或强迫数组身份。
开启严格restart时合同冻结实际强迫，因此不指控该checkpoint错配；普通
输出的来源可追溯性仍不足。需区分requested与applied，并让资格实验明确
禁止替代强迫，而不是抓取失败后改变实验。该项未新增网络复现实验。

### 调研新收获：Heun组合的被组合对象未必是纯Euler步

`weak_bounded.py:50`先形成依赖duration的末态几何；`:65–81`用末态一致
质量逆、新质量边项、低阶预测值和节点界进行成对修正，再重新编码内容。
`weak_time.py:35–41`把这个完整更新做两次后与初态各取一半。
所以，仅有正确的第二阶段时间/状态和凸组合，还不能套用普通RK2阶数证明。

在状态/时间局部足够光滑时，设被组合的映射为
`E_h(t,x)=x+hF(t,x)+h²G(t,x)+O(h³)`。直接展开可得

```text
0.5*(x + E_h(t+h, E_h(t,x)))
  = x + hF + h²*(0.5*(F_t + F_x F) + G) + O(h³).
```

相对目标ODE的二阶精确展开，额外G不会被这个组合消掉；若G不为零，
就不能直接证明二阶。若支持/限幅在该点不光滑，这个Taylor论证本身也
不能直接使用。这是新增的理论风险定位，**不是已经测得G非零或失败根因**。

[Kuzmin等的原论文§3/Remark 4](https://arxiv.org/html/2009.01133v2#S3)
区分步长无关的半离散限幅与依赖低阶预测/步长的FCT，后者不能直接当成
同一半离散RHS；其保界结果也有CFL和阶段条件。本项目可以研究全离散
FCT的时间一致性，但不能由SSP名称跳过阶数验证。这不是要求改成该论文
的新方法，也不采用其放松局地界的选项。

现有`local_euler_defect.py:25`仅在原始一个状态、未调用实际
`_state_fields`倍率的冻结速度上比较E(h)与E(h/2)；它在那点的仿射结果
不能排除实际失败轨迹中其他状态的G或分支变化。最新失败仍为原有符号/
脉冲案例及原1.9门槛，不换域、初值、参考或丢弃不好的节点。

下一次若获准做数值诊断，最小项是在**原失败轨迹**的少量锚点，用真实
场回调记录同起点E(h)-2E(h/2)+x，并拆成high-content与limited-content；
同步记支持、界的极值来源、逐边限幅分支。先区分底层有限步映射的额外
截断项与支持/限幅切换；均不足以解释时才继续查整个阶段耦合。
支持/限幅改变、G与阶段一致性可以并存，不强行判为单一根因。本轮未执行。

### 工业资料反映的路线取舍

- [MITgcm自由面与tracer守恒文档](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html)
  保留线性自由面，并要求变厚度下tracer、连续方程使用一致几何/输运；
  r-star针对表层接近消失有合理动机，不是长期模拟的必选标签。
- [MOM6快慢模态耦合](https://mom6.readthedocs.io/en/main/api/generated/pages/Barotropic_Baroclinic_Coupling.html)
  要求层输运匹配正压时间平均输运。因此原M2一致性问题值得修，
  不能反过来推出必须换水平网格或全部状态结构。
- [MITgcm FD/FV比较](https://mitgcm.readthedocs.io/en/latest/algorithm/finitevol-meth.html)
  说明特定内部二阶中心离散可等价；二者名称不是工业资格的优劣判断。
- 2026年8月的[FESOM2-JAX预印本](https://arxiv.org/html/2608.01546v1)
  是更近的可微海洋参照：作者报告逐核迁移核验、同配置1958–2019回算，
  并验证避开切换点的梯度及短窗完整积分。其单设备字节复现与多GPU舍入
  一致性分开声明。这里借鉴验证组织，不复制模型、不以作者吞吐作本机
  实测，也不把短窗导数外推为几十年可用梯度；尚未本地复现该预印本。

二次审查的建议比“继续追新坐标”更具体：先补生产入口的监测、输入/输出
判定及实际强迫身份，让已有严格测试与生产执行真正接通；时间研究保持
窄范围，先证明被组合映射的前提。存在必要工程工作，但目前仍未完成，
新增7个反例也不是新的积分资格。

## 生产调用链、修复依赖与成熟模式验收实践复核

Lady，2026-09-30。此次以`cfb5bb6`为源码基线，只读核对默认调用链、
现有回归和保存结果；未新开CPU/GPU实验、改内核、切分支、提交或push。
范围包括driver建模/推进/判定/重启，FD的L/N/L、快慢耦合、扩散/源处理，
库存与阶段账本，研究Euler/Heun及CI。不是全库无缺陷的背书。

### 当前真正运行的主线与取舍

生产入口是`run_long_integration_global.main -> make_solver_global -> _step_impl`。
CLI在`src/run_long_integration_global.py:920`没有传入新几何/过程方案参数，
因此沿工厂默认`column_geometry='legacy'`、`match_barotropic_transport=False`、
`process_time_scheme='legacy'`。用户可显式打开mode-split；是否split与是否
采用新候选是两个不同维度。材料顶层、FV/C-grid模块及研究弱式组合不因
被安装、导入测试或具有局地通过项而获得生产资格。

| 现有成果 | 实际实现/依赖 | 判断与边界 |
| --- | --- | --- |
| 表面热源、海冰焓/融冰盐修复`cf097d8` | `_surface_heat_weights`、tracer源、`_dynamic_ice_closure`及完整步源分配；`tests/test_surface_energy.py` | 在旧几何也生效，应保留。需一起保留源分配与冰闭合，不能只摘一个权重函数；热/相变控制不是现实海冰技能验收。 |
| 守恒水平扩散`36a1da4` | `_horizontal_diffusion_flux`同时服务tendency、L半步、N残差扣除与诊断；`tests/test_horizontal_tracer_diffusion.py` | 应保留完整调用闭环。只改tendency会让L/N抵消关系不一致；局地守恒/耗散/空间精度不等于整模式闭合。 |
| 湿面伴随投影`cce8f87`及后续细化 | `_column_divergence`、3D梯度、参数字段、预条件/实际输运残差；`tests/test_column_projection.py` | 有明确离散依据，但只在开启project_adv_vel时使用。移植不能遗漏几何算子、有效参数和实际残差控制。 |
| 严格重启`e47010e` | `restart_contract.py`加driver恢复历史/输出身份、有效参数/实际强迫/源码指纹；`tests/test_production_restart.py` | 应保留工程成果，不能把新模块孤立摘出就声称CLI修好了。小网格连续/两次重启控制与真实30天跨进程字节失败仍分别记账。 |
| 阶段账本 | `stage_budgets.py`直接依赖FD私有步与面输运算子；`tests/test_stage_budgets.py` | 有用但非独立可搬运的旁路；生产snapshot库存不能冒充它的源积分账本。 |
| nodal/material/r-star候选 | 新质量/输运/时间方案及对应网格、状态、审计合同 | 保留代码和失败证据，不作为本轮生产交付前提；弱式时间未达标，不推广。 |

`diagnostics.py:66`仍用静态参考节点体积，driver在`:1202`调用默认库存。
该量不包含eta引起的移动库存和冰潜热，也没有累计外部输入。它可以描述
参考库存变化，却不能单独回答“热盐预算是否相对实际源闭合”。阶段
`decomposition_residual`的近零也只说明账目分解一致，不能代替独立源闭合。

旧生产链还保留值得针对性修复、而非以“旧方法全错”处理的问题：
`_compute_tracer_tendency`在legacy且conv_nsub>1时同时除κ和子步时长
（`:1526`）；改变子步数会改变有效对流强度。L半步对整个速度旋转
（`:1947`），快模态又演化均值Coriolis（`:1811`）。两者已在候选过程
方案分开处理，但不能把候选控制通过说成旧生产默认已修复，或由此
推断它们是某次历史发散的根因。应各自用固定物理系数的最小控制验收。

### “生产级”应落到哪些可核验要求

以下是借鉴成熟模式/业务系统的项目验收框架，不是存在一张通用工业
认证，也不规定所有限幅场必须满足同一个阶数。已登记的1.9时间门槛
仍不改；其失败不能被其他控制、CPU/GPU一致或稳定长跑抵消。

| 验收层 | 一手实践依据 | 本项目尚需补齐 |
| --- | --- | --- |
| 同平台基线回归、重启 | [MITgcm贡献/测试文档](https://mitgcm.readthedocs.io/en/latest/contributing/contributing.html#required-testing-for-mitgcm-code-contributors)要求先测未修改master、再用同选项测分支；另有2+2重启检查及不同运行环境日测 | 已有52项同源控制只是抽样。应建立冻结输入的完整步参考与连续/分段控制；CPU/GPU数值一致和同环境字节重启分开验收。 |
| 连续方程与tracer耦合 | [MOM6快慢耦合](https://mom6.readthedocs.io/en/main/api/generated/pages/Barotropic_Baroclinic_Coupling.html)使用正压时间平均输运；[MITgcm自由面](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html)保留线性方案，非线性方案要求一致厚度/输运 | 有必要检验实际面通量/连续性/常量场保持；并不由此要求把原FD全面换成FV或r-star。 |
| 长期模拟与气候评价 | [OMIP实验/诊断协议](https://gmd.copernicus.org/articles/9/3231/2016/)规定大气强迫实验与海洋/海冰诊断 | 需冻结强迫、区域、恢复项、日历、评分窗及独立参考；长期漂移、源闭合、空间/季节偏差、环流等分别报告。百年无NaN不是气候态精度。 |
| 预报验证 | [OceanPredict业务验证综述](https://sp.copernicus.org/articles/5-opsr/16/2025/)与[Class 4观测空间比较](https://sp.copernicus.org/articles/5-opsr/17/2025/) | 需多起报日、明确预报时效、观测时空配对与覆盖；和持续性/气候基线比较。现有内部SST评分没有提供这一资格。 |
| 可微与性能 | [MITgcm梯度检验](https://mitgcm.readthedocs.io/en/latest/autodiff/autodiff.html)以及已有本地局地FD/JVP控制 | 局地导数、完整短窗目标梯度、切换点适用范围分别声明；同物理/分辨率/精度下测完整步吞吐、显存、JIT与I/O，不以单算子/GPU支持声明工业优势。 |

当前`.github/workflows/ci.yml`只有CPU合成地形回归与MMS，
`pyproject.toml`默认tests目录不包含研究时间验收程序。CI全绿与第37节
时间资格FAIL可以同时成立；需要不同的自动化验收层和明确的promotion
条件，不能删失败案例制造一张全绿报表。

### 收敛到一个迭代顺序

1. **保留现分支作为工作基础，main作为冻结基线。** 已生效修复有跨函数/
   driver依赖，未经上述闭环移植控制就建议“从main重来”依据不足。
   不整支合并；后续可把已验收的生产修复作为独立小补丁交付。
2. **先接通生产可靠性判定。** 提升已有7个工程反例为正式回归，修逐步
   六字段/双速度监测、输入校验、失败退出码、requested/applied强迫身份。
   复用已存在的监测性质，不改变动力离散，不每步拷回全部GPU场。
3. **再逐项修旧链的物理一致性。** 每个修复必须对应独立源/算子控制、
   实际L/N/L完整步及冻结生产基线；不顺手变更坐标、网格、强迫或参数。
4. **候选研究维持隔离。** 若下一轮授权时间诊断，只在原失败轨迹锚点
   分离有限步映射、支持/限幅切换与阶段一致性；根因未证实前不扩大模块。
5. **最后才扩大真实积分和外部资格。** 工程/完整步/重启控制通过后按
   固定配置推进长期、气候及预报评价；百年与超工业级均保持未完成。

本轮新增交付是调用链/依赖/验收取舍，不是新的数值通过项；只更新本报告，
不把研究进度当作生产进度，也不废置已验证的修复和反例。
