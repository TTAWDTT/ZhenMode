# 试验索引

| 实验 ID | Case / parent preset | 类别 | 声明改动 | 当前状态 |
| --- | --- | --- | --- | --- |
| `global-045deg-seasonal-365d-gpu-baseline` | 0.45° 365 天 / `global-045deg-seasonal-checkpoints` | baseline | 继承既有配方，仅预设增加10天 checkpoint | 需固定实际输入并显式 `--backend cuda`；状态以独立 manifest 为准 |
| `global-045deg-seasonal-365d-baseline` | 0.45° 365 天 / `global-045deg-seasonal` | baseline | 无 | 已展开；当前长运行未执行 |
| `global-050deg-wind-only-30d-baseline` | 0.5° 仅风 / `global-050deg-wind-only` | baseline | 无 | 已展开；当前长运行未执行 |
| `global-050deg-restoring-30d-baseline` | 0.5° 恢复 / `global-050deg-restoring` | baseline | 无 | 已展开；当前长运行未执行 |
| `synthetic-production-smoke-80s-baseline` | 合成 80 秒 / `synthetic-smoke` | baseline | 无 | 数值复跑由本次交付验证报告记录 |
| `global-045deg-kv-half-20261003` | 0.45° 365 天 / `global-045deg-seasonal` | single_factor | `kappa_v: 1e-6 → 5e-7 m2/s` | proposed；没有优越性结论 |
| `global-045deg-kv-sweep-20261003` | 同上一 case / preset | sweep | `kappa_v: 5e-7, 1e-6, 2e-6 m2/s` | dry-run；不自动启动 |

每次实际执行在 `outputs/<UTC timestamp>-<config hash>-<unique suffix>/` 独立保存最终配置、manifest、过程日志、模型输出及实际物理网格。`zhenmode runs list` 读取这些 manifest，可同时看见失败、待运行、积分完成和评价验收。

历史结果留在原历史证据路径和 archive 索引，不用这份新实验索引重新命名或升级协议。特别是 0.91364293°C raw 与 A2 1.11257074°C 的旧记录保持旧评分身份。新面积加权 v2 评价必须通过自己的协议文件执行。
