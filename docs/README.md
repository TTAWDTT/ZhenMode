# 文档索引 / Documentation Index

本目录分成两块：**当前主线**（描述今天还在跑的全球有限差分模式）与
**`archive/`**（区域谱模式时期的工作日志与已被取代的早期文档）。
主线文档是权威的；归档文档只用来追溯历史推理，**不要照着它配置运行**。

代码里出现 `D<n>` 标记时，索引的是 [`decisions.md`](decisions.md) 的第 n 条。

---

## 当前主线 / Current

| 文档 | 作用 |
| --- | --- |
| [`decisions.md`](decisions.md) | **决策日志（D1–D41）**。为什么求解器长这样：诊断出的失效、定案的测量、被否掉的方案。代码里只留一两行不变量，按 D 编号指回这里。 |
| [`debug_validation_zh.md`](debug_validation_zh.md) | 本轮混合层、海冰、输运、重启、评分与安装修复；回归和积分的证据，以及尚未验证的边界。 |
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
