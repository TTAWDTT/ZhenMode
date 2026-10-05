# 配置总索引

正式方法唯一明确：既有 ZhenMode 全球有限差分生产主线。这里保存 ZhenMode 预设和两套 MOM6 原生输入接入预设，全部指向现有文件。它们是可审阅的运行配方或接入契约，不是本轮推荐参数，也不是已经完成长期气候验收的配置。

| 方法 / preset ID | 真实预设文件 | 共同 case | 实验或接入关系 | 来源与验证身份 |
| --- | --- | --- | --- | --- |
| ZhenMode / `global-045deg-seasonal-checkpoints` | [严格重启点配方](zhenmode/presets/global-045deg-seasonal-checkpoints.yaml) | 同一 365 天全球问题 | [显式 GPU 基线](../experiments/zhenmode/global-045deg/global-045deg-seasonal-gpu-baseline.yaml) | 继承既有季节强迫配方，仅增加 10 天 checkpoint；实际状态由 run manifest 记录 |
| ZhenMode / `global-045deg-seasonal` | [0.45° 季节强迫](zhenmode/presets/global-045deg-seasonal.yaml) | [365 天全球问题](../cases/global-045deg-seasonal-365d.yaml) | [基线](../experiments/zhenmode/global-045deg/global-045deg-seasonal-baseline.yaml)、[垂向扩散单因素](../experiments/zhenmode/global-045deg/20261003-reduce-vertical-mixing.yaml)、[三点扫参](../experiments/zhenmode/global-045deg/sweep-vertical-mixing.yaml) | 参数来自 `run_candidate_baseline_045_icefloor.sh` 的生产 FD 调用；当前已展开，365 天未重新积分 |
| ZhenMode / `global-050deg-wind-only` | [0.5° 仅风](zhenmode/presets/global-050deg-wind-only.yaml) | [共同仅风 30 天](../cases/global-050deg-wind-only-30d.yaml) | [FD 基线](../experiments/zhenmode/global-050deg/global-050deg-wind-only-baseline.yaml)，对应下方 MOM6 仅风接入 | 参数来自既有 `run_industrial_comparison_050_wind_only.sh`；当前已展开，30 天未重新积分 |
| ZhenMode / `global-050deg-restoring` | [0.5° 温盐恢复](zhenmode/presets/global-050deg-restoring.yaml) | [共同恢复 30 天](../cases/global-050deg-restoring-30d.yaml) | [FD 基线](../experiments/zhenmode/global-050deg/global-050deg-restoring-baseline.yaml)，对应下方 MOM6 恢复接入 | 参数来自既有 `run_industrial_comparison_050_restore_30d.sh`；当前已展开，30 天未重新积分 |
| ZhenMode / `synthetic-smoke` | [有界生产工程见证](zhenmode/presets/synthetic-smoke.yaml) | [全湿 8×8×4、80 秒](../cases/synthetic-production-smoke-80s.yaml) | [生产链基线与评价](../experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml) | 输入服务明确生成；使用既有受控 driver 方式调用同一生产工厂，运行、输出和评价由每次 run manifest 记录；没有海洋效果资格 |
| MOM6 / `mom6-global-050deg-wind-only-30d` | [0.5° 仅风原生接入](mom6/presets/global-050deg-wind-only-30d.yaml) | [与 FD 相同的仅风 case](../cases/global-050deg-wind-only-30d.yaml) | `baseline mom6 prepare-native --preset` 引用此文件，核查 `--case`、原生输入目录和共同 reference；不经过 ZhenMode experiment resolver | 已有 720×260×14 全球对照条件整理成严格接入声明；读取实际原生 DT/DTBT 等，不猜历史时间步；本轮未重新运行 30 天 |
| MOM6 / `mom6-global-050deg-restoring-30d` | [0.5° 恢复原生接入](mom6/presets/global-050deg-restoring-30d.yaml) | [与 FD 相同的恢复 case](../cases/global-050deg-restoring-30d.yaml) | 同一 `prepare-native` 链，另核恢复目标文件及实际层热容量；对应上方 FD 恢复基线 | 已知原生恢复参数来自历史 30 天记录；缺失他机完整输入未重建，本轮未重新运行 30 天；完整物理与垂向等价尚未核验 |

前三套 FD 参数来源提交均为 `25258950905f9d1aa84509c4c99ebad9ef33ba2b`，原脚本路径和进一步历史来源直接保存在各 YAML 的 `source`。原文件名包含 `candidate` 不代表移动材料库存方法：上述预设经调用核对仍属于全球 FD 生产谱系。材料候选第 353→354 步失败不能否定既有百年稳定记录。

0.45° 预设与历史低 RMSE 记录保持来源关系，但新面积加权 v2 评价不会重新授予旧湿格等权 raw/A2 成绩；历史报告提交也不等于完整执行源码身份。外部全球输入在 case 中有版本、路径与待核验 SHA256 状态，不能把尚未恢复的数据声明为历史原始缓存。

两套 MOM6 预设共享 FD 的 case 标签，同时显式声明 native C-grid、14 层 Z-star ALE 和原生参数读取。上游及依赖固定于 [MOM6 pins](../src/zhenmode/baselines/mom6/pins.json)，缓存与编译产物放在隔离目录。`prepare-native` 必须核对真实深度、湿掩膜、初始温盐、输入文件与执行来源；风变换、恢复物理或垂向等价未完成时，比较为受限，不能公平排名。已实测的 [官方 tc1 小例](../src/zhenmode/baselines/mom6/cases/tc1.json) 是另一个问题，不能替这两套全球接入证明 30 天成绩。

ZhenMode 预设继承、实验变化、配置展开和独立 run 记录见 [实验说明](../docs/experiments_zh.md)。实际运行与评分状态从 run manifest 读取，不由预设存在或命令退出成功推断。

## 继承与覆盖

每个预设可有一个 `parent`；子预设 `options` 只能添加父级未定义选项。修改已有选项必须使用带 `before`、`after`、`factor`、`reason` 的 `overrides`，拒绝循环继承与重复覆盖。实验中的覆盖只放在 experiment `changes`。展开会记录 CLI 所有默认值、分辨率自动缩放、有效快步时长、物理参数和环境 CG 默认，避免依赖隐含参数。
