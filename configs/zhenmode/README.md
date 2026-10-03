# 正式方法预设

默认方法是已有 ZhenMode 全球有限差分生产主线。以下是现有脚本参数的可展开记录，不是新推荐或已重新获得历史成绩的配置。`source` 固定原脚本路径和来源提交；当前源码身份另由每次 run 冻结。

| Preset ID | 参数来源 | 对应 case |
| --- | --- | --- |
| `global-045deg-seasonal` | `scripts/run_candidate_baseline_045_icefloor.sh` | `global-045deg-seasonal-365d` |
| `global-050deg-wind-only` | `scripts/run_industrial_comparison_050_wind_only.sh` | `global-050deg-wind-only-30d` |
| `global-050deg-restoring` | `scripts/run_industrial_comparison_050_restore_30d.sh` | `global-050deg-restoring-30d` |
| `synthetic-smoke` | 原受控生产 driver fixture 的运行方式；网格与输入明确重新定义 | `synthetic-production-smoke-80s` |

脚本中 `candidate` 一词不自动意味着新的动力核心。前三项由实际调用和参数核对，均走既有全球 FD 主线，不使用移动材料库存候选。

每个预设可有一个 `parent`；子预设 `options` 只能添加父级未定义选项。修改已有选项必须使用带 `before`、`after`、`factor`、`reason` 的 `overrides`，拒绝循环继承与重复覆盖。实验中的覆盖只放在 experiment `changes`。展开会记录 CLI 所有默认值、分辨率自动缩放、有效快步时长、物理参数和环境 CG 默认，避免依赖隐含参数。
