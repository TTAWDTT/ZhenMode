# 当前工程验证报告

本报告描述删除转接层后的版本。完整机器记录、实际源码和产物 hash 见 [validation.json](research_engineering_validation.json)；
删除与迁移依据见 [清理清单](cleanup_manifest.json)。第一阶段报告原字节在 [独立证据目录](../archive/evidence/reorganization/phase1/README.md)，不改写成当前结果。

## 已完成的独立本地检查

每次数值调用限制单 CPU、180 秒、4 GiB，记录整个 Job Object 的峰值内存与退出状态。
环境为 Windows、Python 3.12.10、JAX/JAXlib 0.11.2、NumPy 2.5.3；依赖版本在 [CPU 验证文件](../requirements/validation-cpu-py312.txt)。

| 范围 | 结果 | 结论边界 |
| --- | --- | --- |
| 测试收集 | 2101 项；main 的 2008 项和第一阶段的 2101 项均逐节点保留 | 参数 ID 保留；没有因删除或迁移漏收集 |
| 工程、实验、评价合同 | 231 passed，279 subtests passed | 单位、继承、消融、来源、协议、窗口、网格、产物与负例检查 |
| 来源、打包、MMS CLI 与既有评分测试 | 112 passed | 当前实际文件校验、错误退出与旧评分语义 |
| 运行、重启与 MOM6 适配合同 | 73 passed，1 skipped | Windows 下跳过 POSIX 假 MPI 进程合同；未启动 MOM6 |
| 单文件类型/来源合同 | 最终 11 passed | 显式生产精度、pickle、pytree、真实模块 tamper/缺失拒绝 |
| 独立热源预算及扩散故障注入 | 2 passed，25 deselected | 独立输入积分及错误源识别；没有回填残差自证 |
| MMS | PASS，二阶收敛检查通过 | 算子小问题，不是全球气候验收 |
| fp64 / fp32 全状态记录 | 各 275 项，274 项逐字节一致 | 唯一差异为预先声明的 pickle 模块身份变更；离线检查其字段相同 |
| 输出及两次中断重启 | 344 项与第一阶段冻结记录逐字节一致 | 8 个接受步；连续/重启记录一致，保留实际输入校验值 |
| 干净 wheel 安装 | 仓库外新环境，106 个源码文件，0 个旧别名 | 无 research，实际安装文件与当前源码逐字节一致，无临时 `src` 插入 |
| sdist → wheel | 构建成功，106 个源码文件与直接 wheel 同字节 | 检查缓存不会重新引入已删除模块 |
| 历史与迁移校验 | 5 份原 Git 证据、83 个删除来源、38 个迁移文件、6 份报告通过 | 不把历史报告当作当前复跑 |
| 展开与 dry-run | 全球预设、单因素变化、三点 sweep 通过；dry-run 为 proposed | 没有启动全球试验或扫参 |
| Ruff | PASS | 自动 CI 工程检查 |

源码 AST 对比保留 491 个现存正式/研究函数定义，6 个变更限于命令编排和来源定位/收集。
数值工具、动力、物理、时间积分、工厂和独立测试 support 中没有公式变更。
测试合同中的变更及其目的逐项保存在机器报告，未删除断言以制造通过。

## 仓库外完整小运行

新建环境只安装正式 wheel 与锁定依赖，从仓库外执行：

```bash
zhenmode experiment run experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml --root <仓库绝对路径> --outputs outputs/smoke --evaluate
zhenmode runs list --outputs outputs/smoke
```

实际 run ID：`20261003T082557671805Z-20d0205b8a52-9e412432`。
使用正式生产工厂和完整时间步执行合成 80 秒问题，运行与评价均 completed；验收状态为 `not_declared`，不冒充指标达标。
实际安装源码的 106 项身份与当前源码一致，机器报告保存 manifest、输出、重启和报告的校验值。
产物位于本机独立环境 `C:/Users/86153/.codex/tmp/zhenmode-clean-v3-20261003/outputs/smoke`，没有提交大数组。

## 保留的失败与修复

删除旧路径后，首次相关测试有 12 个旧路径/类型身份断言失败，第二次剩余 1 个旧空目录造成的 namespace 失败；
修改合同及清除本 worktree 可重建的空缓存后，相关集合最终 231 项通过。
单独运行类型测试暴露了隐含的双精度初始化依赖，改为显式导入生产后端后 11 项通过。
一次安装复查发现 8 个仅含包说明的 initializer 被非预期换行转换；恢复已执行源码字节后，仓库外完整安装复查通过。
这些失败、退出码、资源 receipt 和后续通过均保留在机器报告，没有归为成功或抹掉。

## 可复现命令与未运行项

```powershell
python scripts/run_bounded_research_tests.py tests/infrastructure tests/fd/test_production_ownership.py tests/experiments tests/evaluation -q
python scripts/run_bounded_research_tests.py tests/infrastructure/test_source_layout.py tests/runtime/test_runtime_contracts.py tests/runtime/test_restart_trajectory.py tests/baselines -q
python scripts/run_bounded_research_tests.py --module ocean_solver mms
python scripts/run_bounded_research_tests.py --script scripts/capture_modularization_state.py --source-root src --output <新的fp64记录.npz>
python scripts/run_bounded_research_tests.py --script scripts/capture_driver_modularization.py --source-root src --output <新的驱动记录.npz> --wind-jit
python scripts/verify_historical_evidence.py --cleanup
```

记录捕获器拒绝覆盖已有全状态记录；复跑比较应保留每次的新输出及来源 sidecar。
历史基准与原工具可从第一阶段提交和 main 冻结；不能让改过的工具同时产生参考与候选后称为独立对比。
原始数值参考数组的路径与 SHA 保存在机器报告，本机 `logs/reorganization` 保留冻结产物。

此前自动启动的 CI `37104386941`、`37104383953` 已确认 cancelled。
当前自动 CI 只运行 Ruff，全套 2101 项与 MMS 需要人工验收后手动选择 `full_validation`。
本轮没有重跑 MOM6 构建/积分、全球 30/365 天、GPU 或百年气候。
第一阶段 MOM6 缓存小例及其构建身份限制见原报告，不作为本轮新运行。

工程回归不建立新的长期气候精度、工业级资格或同等误差速度优势。
旧 raw/A2 保持旧协议；面积 v2、共享网格要求和成本口径没有在本轮清理中改动。
Git 换行过滤可能使 Git blob 与实际 checkout 字节不同，执行来源始终校验实际文件；历史报告同时保留来源提交和原字节身份。
