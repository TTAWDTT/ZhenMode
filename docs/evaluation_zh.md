# 统一评价：运行完成、指标验收与公平比较分开

正式生产方法是既有全球有限差分谱系。百年稳定轨迹和低 RMSE 历史证据保留原身份；新的工程整理没有重新授予这些历史成绩，也没有把材料库存候选第353步失败当成生产故障。历史记录见 `archive` 的证据索引及 `docs/legacy_core_repair_status_zh.md` §23。

## 一个评分入口，一份指标实现

`zhenmode evaluate score` 连接结果读取、严格协议检查、已有 `metrics.score_npz` / `external.score_external_field`、可比性校验与报告。`evaluation/` 不含第二份评分实现。解析与制造解验证继续使用 `validation.mms`；独立驻波 oracle 继续在其原研究合同目录，正式包不反向导入研究代码。

```sh
zhenmode evaluate score --input outputs/RUN/model/global_run.npz --protocol evaluation/protocols/production-smoke-v1.json --run-manifest outputs/RUN/manifest.json --out-dir outputs/RUN/evaluation
```

运行 manifest 必须有稳定 `run_id`、`case_id`、`method`、`config_hash`、`execution_status=completed`，以及实际模型产物的 `result_path/result_sha256` 或 `outputs` 的 `model_result` 项。随便选择另一 NPZ 配一个 completed 标签会被拒绝。失败和未完成执行保持在运行索引，不补成指标通过。

## 协议冻结与空间、时间含义

协议 JSON v1 拒绝未知和重复字段，温度须为 `degC`，时间窗口为有限非负模式日且起点小于终点。窗口终点必须与完整结果末日一致；数据须覆盖窗口、时间严格递增。报告同时记录请求窗口和实际选中的保存时刻，不能把缺失预热期或错误窗口静默替代成全程。

新 SST 评分明确为 `area_weighted_angular_box_v2`：湿区面积权重、双方共同的经纬角度窗口（各半宽2°）、同一个参考湿区。保存记录按算术平均计时，不是假定已有 time bounds 的时间积分。中心推算面积注明 `inferred_center_edges`；不能声称已独立核实任意原生网格几何。

原实现的 `T_init` 是保存的初始表层温度。即使它来自 WOA，也不是默认认证的独立气候观测。protocol 的参考版本与字段 hash 分别记录；模板 hash 可以留 null，但报告必须绑定实际参考字段。MOM6 外部评分严格核原生中心、湿掩膜、Celsius 单位和实际 geometry/reference hash；没有通用重网格。

`protocol_file_sha256` 冻结实际文件字节，`protocol_content_sha256` 冻结规范 JSON 内容，二者分开。兼容字段 `protocol_sha256` 延续历史文件字节含义。相同 ID 的文件被换掉也会拒绝。未来重网格、时间 bounds 加权或独立观测验证应定义独立协议并增加对应校验，不可偷换当前实现。

## 三类结果

| 类别 | 当前报告 | 限制 |
|---|---|---|
| 数值 | 执行/异常退出、watchdog、finite coverage、内容首末漂移；MOM6 tc1 原生 CFL/截断及完成情况 | 端点内容变化不是独立完整收支残差；小例不替代长期漂移 |
| 效果 | 明确口径的 raw/A2/区域 SST 指标；由已有评分器生成 | 初值参照不证明独立气候精度；初态 MLD 标 `initial_T_S` |
| 成本 | compile、integration、IO、end-to-end 分项、硬件、精度与 ranks | 未实测分项为 null，不能用另一方端到端替纯积分；不由总耗时倒推分项 |

`completed` 仅说明执行完成。所有预设协议目前没有新推荐的气候阈值，故 `acceptance=not_declared`；已有 watchdog PASS 保留在数值字段。预注册阈值存在时，实际有限指标和完整覆盖也须满足才通过；缺值失败。

## 可比性与来源

统一 `compare` 要求共同 case、实际海深/湿区网格身份、有效物理参数化、原始数据、协议内容、实际窗口、参考场与指标定义一致。`dt/dtype/子步` 等数值方案改动在 `change_categories` 明示，可以评价误差变化；混合/强迫等物理改动须保留受限比较身份。结果可以并列阅读，工具不会跨协议生成排名。

生产来源检查由完整注册包、冻结文件 hash、真实 `sys.modules` 核心执行清单和结果内嵌来源共同约束。两份门面文件不够；缺失注册包、真实动力/时间积分文件或已知版本时标 limited，并拒绝公平比较。源版本的注册表 hash 不认识时，工具保留诊断但不自授完整认证。

报告区分运行时 `execution_source_identity` 和本次评价的 `scoring_source_identity`。后者重新核当前安装包全部 Python 文件，不能把旧运行 hash 当作当前评分源码。进程收据仍需日志复核；这些工程 hash 不证明作者报告的未归档运行代码完全身份，也不把过去报告提交当作全部执行源码。

成本比较独立检查测量分项、硬件、精度、ranks 和计时 scope。即使条件相同，也只给比较资格；正式加速结论还要同等误差下实测误差—成本曲线。本实现明确 `ranking=not_performed`、`equal_error_speedup=not_established`。

## 历史证据

```sh
zhenmode evaluate import-historical --input research/experiments/ice_proxy_045/benchmark_365d_repeat.json --out outputs/ice-proxy-historical.json
```

导入保存原路径、原字节 hash、原指标和旧协议，原文件不变。bb7ba23 快照的365日重复 PASS、评分280–365日、raw约0.91364293°C、A2约1.11257074°C 属旧湿格等权协议，不与新面积 v2 混排。长期 PASS 支持历史数值稳定，部分 OHC/AMOC 警告仍保留；不扩大成所有气候指标达标或同一源码从零连续325年。

## 有界验证

`tests/evaluation` 覆盖错误单位、未知/重复字段、窗口、非 binary/NaN 掩膜、NaN 峰值、参考 hash、借用结果、来源门面/缺项、协议变更及不兼容比较。`tests/baselines` 验转换/原生产物 tamper、原生单位、记录身份和 POSIX 监控异常清理；这些夹具不冒充真实模型运行。实际执行与结果参见此次重构验证报告，跳过/未运行和失败单列。
