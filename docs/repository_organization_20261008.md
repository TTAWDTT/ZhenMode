# 仓库整理记录：2026-10-08

本轮基于已合并主线 `c682e80254171b6cda33e0196c1a51a57e389542`，分支 `ttawdtt/repository-organization`。目标是让后续对照和 bench 有明确归属，同时保持已有数值方法、正式默认值和评分口径。

## 核查与取舍

遍历正式包、测试、维护脚本、配置和文档，核查静态导入、动态命令、打包数据及源码身份。正式模型已有清楚的动力／物理／时间积分分区，保留该结构；不按行数机械拆分文件。最近增加的离线准备和外部海冰接入混在实验执行目录，按实际职责归位。

相同职责的源码哈希采集与文件身份校验集中到 `provenance/sources.py`。13 处等价源码哈希推导改用一个不缓存结果的实现，各阶段重新核验的次数保持。`checked_file` 从 WOA 转换器迁到来源校验模块，原有大小、SHA256 和失败条件保持。

独立解析解、NumPy／Fortran 参考、故障负例均保留。全函数扫描发现模型状态身份与历史回归采集器存在相同计算，但后者需要观察原版本，故保留其独立实现；不同 JRA 制造资料的测试夹具也保留各自资料语义，不为了消除表面相似而合并。

## 文件迁移

| 原位置 | 新位置 |
| --- | --- |
| `src/zhenmode/execution/initialization.py` | `src/zhenmode/preparation/woa.py` |
| `src/zhenmode/execution/native_initialization.py` | `src/zhenmode/preparation/native_initial.py` |
| `src/zhenmode/execution/native_geometry.py` | `src/zhenmode/preparation/coast.py` |
| `src/zhenmode/execution/native_bottom.py` | `src/zhenmode/preparation/bottom.py` |
| `src/zhenmode/execution/native_fd.py` | `src/zhenmode/preparation/fd.py` |
| `src/zhenmode/execution/datasets.py` | `src/zhenmode/preparation/acquisition.py` |
| `src/zhenmode/execution/preparation.py` | `src/zhenmode/preparation/forcing.py` |
| `src/zhenmode/execution/native_sis2.py` | `src/zhenmode/coupling/sis2.py` |
| `src/zhenmode/execution/native_ice_case.py` | `src/zhenmode/coupling/geometry.py` |
| `src/zhenmode/execution/native_run.py` | `src/zhenmode/execution/wind_run.py` |
| `src/zhenmode/execution/sis2_bridge.f90` | `src/zhenmode/coupling/sis2_bridge.f90` |
| `tests/data/test_native_sis2.py` | `tests/coupling/test_sis2.py` |
| `tests/data/test_native_wind_run.py` | `tests/runtime/test_native_wind_run.py` |
| `tests/validation/test_benchmark_gate.py` | `tests/evaluation/test_benchmark_gate.py` |
| `tests/validation/test_benchmark_manifest.py` | `tests/evaluation/test_benchmark_manifest.py` |
| `tests/validation/test_benchmark_metrics.py` | `tests/evaluation/test_benchmark_metrics.py` |
| `tests/validation/test_benchmark_table.py` | `tests/evaluation/test_benchmark_table.py` |
| `tests/validation/test_score_external_model.py` | `tests/evaluation/test_score_external_model.py` |
| `tests/validation/test_summarize_ice_diagnostics.py` | `tests/evaluation/test_summarize_ice_diagnostics.py` |

删除 `tests/validation/__init__.py`，其测试归入已有 `tests/evaluation`。新增 `preparation`、`coupling` 包标记和耦合测试包标记，不安装旧模块转接层。没有删除原始数据、研究原型或历史产物。

构建清理同时排除已退休的 Python 模块和 `zhenmode` 包内 Fortran 驱动，避免旧 build 缓存把迁移前的驱动混入 wheel；保留无关缓存，并继续检查暂存目录不能越出受管 build 或覆盖源码。

## 调用与接口

CLI 命令、参数和退出状态保持原样，直接调用新所属模块。源码身份登记覆盖全部新路径；严格 checkpoint 按实际文件核验，旧版本 checkpoint 使用其原 Git 版本，不伪造旧哈希。逐文件职责和链路见[准备与耦合导读](preparation_coupling_zh.md)，正式模型导读已补齐 contacts、SIS2、TEOS 和海岸模块。

## 验证

| 检查 | 结果与范围 |
| --- | --- |
| 文件字节核对 | 模型的 60 个 Python 文件、评价和配置定义的 36 个文件、迁移的 Fortran 驱动均保持原字节 |
| 迁移实现 AST 核对 | 10 个迁移实现仅在导入、共同源码哈希调用及共同文件校验名称上变化；数值计算保持 |
| 测试收集 | 原 1324 项全部保留，按迁移表逐项对应；新增 6 项来源校验控制，总计 1330 项 |
| 工作流、评价、架构与耦合测试 | 244 passed、5 skipped；Windows 本地范围，跳过不能当作通过 |
| 资料、原生运行和新来源控制 | 首次 172 passed、1 failed；失败是故障注入测试残留旧导入，修正后其完整 16 项强迫准备测试通过；没有改动断言或阈值 |
| 生产连续／分段重启 | 重构前、后各自连续 8 步与 2+2+4 步恢复逐字节一致；合成 CPU 8×8×4、80 秒 |
| 重构前后输出 | 87 个保存字段及 6 个原始状态数组逐字节一致；唯一变化字段是如实记录新路径与哈希的 `source_identity_json` |
| 独立 MMS | 全部通过，纬向导数加密误差比 4.30 |
| wheel | 107 个 Python／Fortran 文件与源码逐字节一致，退休路径缺席 |
| 仓库外安装与运行 | 新环境安装 wheel，共用既有第三方依赖；从外部临时目录、隔离 Python 模式验证导入、帮助、合同、配置展开，正式合成算例运行和评价完成、8 个接受步、PASS |
| Ruff 与差异检查 | 通过 |

本地数值验证由单 CPU、180 秒、4 GiB 运行器约束。安装检查明确复用固定第三方依赖，不宣称重新安装或验证了所有可选后端。完整 Linux CI 的实际结果随 PR 记录。

机器可读迁移表、冻结源码、导入图、测试收集对应、字节／AST核对、重启前后、wheel 和安装运行收据保存在本地 `outputs/repository-organization-20261008/`，不提交大产物。首次失败与修正后检查分别保留。上述检查验证整理及既有小算例，不产生新的全球气候或性能优越性结论。
