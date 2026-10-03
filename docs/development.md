# 开发与测试

正式包和研究包独立安装。执行完整开发测试时：

```sh
python -m pip install -e ".[dev]"
python -m pip install -e ./research
python -m ruff check .
python scripts/make_synthetic_bathymetry.py
python -m pytest tests/ -q
python -m ocean_solver mms
```

测试使用当前 checkout。`tests/fd`、`tests/runtime`、`tests/data` 对应正式模型；`tests/baselines`、`tests/evaluation`、`tests/experiments` 对应接入契约；研究测试使用独立原型和参考计算。必要参考解与负例不能改成对同一实现的回调验证。

Windows 本地有界调用使用单 CPU、180 秒、4 GiB：

```powershell
python scripts/run_bounded_research_tests.py tests/infrastructure tests/experiments tests/evaluation -q
python scripts/run_bounded_research_tests.py --module ocean_solver mms
```

真实数据前后回归可使用 `scripts/capture_global_regression.py` 和 `scripts/compare_global_regression.py`。计划、输入、来源、数值记录与报告保存在各次运行目录；输出拒绝覆盖。全局强迫可显式选择缓存目录，读取器在调用时解析该目录。

自动 CI 运行 Ruff，完整测试和 MMS 通过 `workflow_dispatch` 的 `full_validation` 手动启动。测试通过说明对应契约有效；长期气候效果和同等误差下的成本优势需要独立实验。

内部调用直接引用所属模块：网格类型与垂向厚度在 `geometry.types`，纯网格操作在 `geometry.mesh`，评分及报告在 `evaluation`。`runtime.entry` 只保留启动入口；需要控制依赖的运行使用 `runtime.application.default_services()` 和 `dataclasses.replace` 注入 `RunServices`，不从 CLI 模块导入内部函数。旧区域 SSH/SLA 下载器及历史气候评分脚本已删除；重现其历史操作应检出相应 Git 提交。
