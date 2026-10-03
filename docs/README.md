# 文档索引 / Documentation Index

本轮工程的入口是 [ZhenMode 架构](research_engineering_architecture_zh.md)、
[生产职责](production_architecture_zh.md)、[实验运行](experiments_zh.md)、
[统一评价](evaluation_zh.md)、[MOM6 接入](baselines_zh.md)、
[验证报告](research_engineering_validation_zh.md) 和
[历史生产证据](../archive/evidence/README.md)。下面保留原技术说明和研究过程索引；
旧修复计划、候选失败与验收路线图不替代上述默认生产边界或原始历史成果。

本目录分成两块：**当前主线**（描述今天还在跑的全球有限差分模式）与
**`archive/`**（区域谱模式时期的工作日志与已被取代的早期文档）。
主线文档是权威的；归档文档只用来追溯历史推理，**不要照着它配置运行**。

代码里出现 `D<n>` 标记时，索引的是 [`decisions.md`](decisions.md) 的第 n 条。

## 先读哪几份

1. [项目入口与代码主线](../README.md)：CLI → 初始化/强迫 → FD完整步 → 输出/重启。
2. [修复实施记录](legacy_core_repair_status_zh.md)：区分生产、显式候选与失败证据；§36–37是最新弱式容量/时间结果。
3. [节点r-star实验索引](../research/experiments/material_rstar_coordinates/README.md)：按依赖链找代码、协议和审计，不把旧控制当并行生产方案。
4. [决策日志](decisions.md)：按源码中的D编号查理由；完整方案/长期验收按需查阅。

[`research/plan.md`](../research/plan.md)是旧研究日志，不是当前执行计划。
已保存结果以当时的源码快照/提交和输入hash为准；后续清理不会重标失败或
使旧结果自动成为当前源码的资格证明。路线图不是自动运行授权。

---

## 当前主线 / Current

| 文档 | 作用 |
| --- | --- |
| [`branch_review_zh.md`](branch_review_zh.md) | main与本地修复分支的同测试对照、已获收益、未过资格及复杂度成本；不以绿测试替代完整模式交付。 |
| [`decisions.md`](decisions.md) | **决策日志（D1–D46）**。为什么求解器长这样：诊断出的失效、定案的测量、被否掉的方案。代码里只留一两行不变量，按 D 编号指回这里。 |
| [`debug_validation_zh.md`](debug_validation_zh.md) | 本轮混合层、海冰、输运、重启、评分与安装修复；回归和积分的证据，以及尚未验证的边界。 |
| [`method_choice_review_zh.md`](method_choice_review_zh.md) | 原方法可用性与替换动机的独立复核；区分必要一致性修复、可选架构迁移及新原型自身问题。 |
| [`legacy_core_repair_plan_zh.md`](legacy_core_repair_plan_zh.md) | 原核心优先的修复方案：3–7 工作日规划、工作包、验收矩阵、止损与新核心替换门槛；已开始执行。 |
| [`legacy_core_repair_status_zh.md`](legacy_core_repair_status_zh.md) | 实施与验收记录：冻结 R、原路径兼容性、M1 几何候选、真实地形对照及保留的失败；不将稳定性等同物理预算达标。 |
| [`gpu_runtime_zh.md`](gpu_runtime_zh.md) | 本机 Linux/WSL CUDA 环境、同参数 CPU/GPU 重放合同及未通过/未验证边界；不将后端可用等同工业性能达标。 |
| [`industrial_alignment_roadmap_zh.md`](industrial_alignment_roadmap_zh.md) | 持续对齐并超越成熟模式的完整验收矩阵、文献依据、逐轮研究/实现规则与未完成项。 |
| [`solver_technical_report_zh.md`](solver_technical_report_zh.md) | 求解器技术报告：方程、离散、时间积分、模块结构。 |
| [`deep-heat-poisoning-root-cause.md`](deep-heat-poisoning-root-cause.md) | 深海增温（deep-heat poisoning）根因：列热收支泄漏的两个独立缺陷（含 Defect 5：GM 与 Redi 是同一算子的双计）。 |
| [`resolution_cfl_limits.md`](resolution_cfl_limits.md) | 分辨率标度：实测 CFL 上限与 `dt_bt`/`nu_h`/`nu_bi` 的 `dx^1/2/4` 自动缩放。 |
| [`hydrostatic_primitive_equations.md`](hydrostatic_primitive_equations.md) | 静力原始方程组的中文讲解（物理背景，与实现无关）。 |
| [`repositioning_memo_zh.md`](repositioning_memo_zh.md) | 项目定位声明：从"逼近真实观测"回到"稳定快速的海洋模式"。 |
| [`spinup_plan_zh.md`](spinup_plan_zh.md) | 平衡态加速自旋升方案（把模式推到准平衡气候态）。 |

## 已归档 / Archived

`archive/` 下的文档分两类：

- **区域谱模式时期（2026-08-19 ~ 08-23）的底座记录**：
  `TIMELINE.md`、`progress_report_zh.md`、`report.md`、`verification_ladder.md`、
  `verification_memo_zh.md`、`zhihu_spectral_method.md`。
- **过程性工作日志 / 已被取代的方案**：
  `pam_work_summary_zh.md`、`g365d_work_summary_zh.md`、
  `long_run_climatology_plan_zh.md`、`long_run_climatology_report_zh.md`、
  `plan_gm_closure_en.md`、`baroclinic_activation_roadmap_zh.md`。

对应的退役代码在 [`../archive/regional/`](../archive/regional/README.md)。


## Scratch

`scratch/` is intentionally ignored. It holds legacy probe outputs,
downloaded reference pages, old presentations, and other local artifacts
that should never be confused with mainline source or docs.
