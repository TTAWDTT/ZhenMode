# 清理交付与接口边界

本轮继续整理第一阶段提交 `0fcd7c00dae610c2cac8aa1656ecc87fc231b29a`，目标是让当前工程直接表达生产方法的职责。
正式默认方法、数值公式、参数默认值、接受/拒绝步及评分语义保持。具体文件、来源和校验值见 [逐文件清单](cleanup_manifest.json)。

## 删除与迁移

| 处理 | 数量 | 依据 |
| --- | --- | --- |
| 删除生产 bare-module 转接文件 | 31 | 当前调用者、CLI 和打包均已改用所属模块 |
| 删除研究 bare-module 转接文件 | 9 | 研究包与测试直接使用 `zhenmode_research.candidates` |
| 删除包内别名、汇总门面、类型名伪装辅助及旧直接启动文件 | 42 | 没有独立数值实现；不再承担当前入口或序列化兼容 |
| 删除 `scripts/run_tests.py` | 1 | 仅打印标题并启动 pytest，使用 pytest 或有界 runner 即可 |
| 归档 campaign shell | 21 | 按原提交保留字节与配方来源，不宣传当前可直接运行 |
| 归档 controlled-window 工具 | 11 | 它们执行并校验指定历史源码包，不能静默改接当前模块 |
| 研究验证工具移入 `research/tools` | 6 | 压力/势能控制和质量/成本研究工具不属于默认生产流程 |
| 删除重复的第一阶段报告及旧导航 | 6 | 当前报告与导航已覆盖使用入口，旧版本可从 Git 获取 |

删除文件可由来源提交恢复；没有删除原始输入、未提交实验产物或仅存结果。
历史工具有明确索引；第一阶段重复档案和完整机器报告在后续清理中删除。
包内旧 `fd`/`data` 转接目录已消失，正式 wheel 从第一阶段 177 个 Python 文件收敛为 106 个实际模块。

## 保留的接口

保留 `ocean_solver` 包名、`ocean-solver` 生产命令和统一 `zhenmode` 命令。
模型内部直接调用 `model.factory`、`timestepping`、`dynamics`、`physics` 和它们的叶模块；没有新的汇总门面。
IO 网格装配仍承担实际的读取与纯几何组装职责，`RunServices` 仍承担运行服务绑定职责。

Python 调用者应使用实际 owner，例如：

```python
from ocean_solver.model.factory import make_solver_global
from ocean_solver.state.types import JaxStateG
from ocean_solver.io.grid import make_global_grid
```

旧 bare imports 与旧 `ocean_solver.fd.*` / `data.*` 路径不再支持。
状态类型与网格类型恢复实际模块名，旧 bare-module pickle 需要原提交环境；不设置 `sys.modules` 别名或伪装 `__module__`。
状态字段、defaults、pytree 和数值数组不变。严格重启仍要求实际源码 hash 一致，历史 checkpoint 应用原源码重放。
迁移记录中的旧路径只是档案标签，不能让新字节成为旧运行的源码身份。

## 审阅与验证

[当前验证报告](research_engineering_validation_zh.md)分别列出已复跑、失败后修复、跳过和未运行项。
完整测试集合没有少收集；独立参考计算和故障注入仍在测试中，历史参考函数保留原字节。

```bash
python scripts/verify_historical_evidence.py --cleanup
python -m ruff check .
```

第一条检查原历史 Git 证据、删除文件来源及保留的历史工具字节，不启动数值积分。
自动 CI 只运行 Ruff；全套 suite 与 MMS 在人工审阅后手动选择 `full_validation` 执行。
本轮没有启动 MOM6 构建、全球长积分、GPU 或百年重跑。
