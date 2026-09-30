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
