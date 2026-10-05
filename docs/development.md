# 开发与测试

需要 Python 3.12+。仅安装正式包即可收集和执行仓库测试：

```sh
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -m ruff check .
python scripts/make_synthetic_bathymetry.py
python -m pytest tests/ -q
python -m zhenmode mms
```

`requirements.txt` 固定 CPU 开发验证环境；`pyproject.toml` 定义可安装包的依赖和入口。GPU 环境见 [GPU 运行](gpu_runtime_zh.md)。

`tests/fd`、`tests/runtime`、`tests/data` 对应模型与输入；`tests/baselines`、`tests/evaluation`、`tests/experiments` 对应接入、评分与实验契约。`test_optin_*` 覆盖正式包提供的可选接口，不代表默认生产方案。独立解析解、NumPy 参考计算和故障负例保留在测试中，不能改为同一实现的回调自证。研究专属测试位于本地研究工作区，不参与正式收集。

Windows 本地有界调用使用单 CPU、180 秒、4 GiB，可按明确范围拆分：

```powershell
python scripts/run_bounded_tests.py tests/infrastructure tests/experiments tests/evaluation -q
python scripts/run_bounded_tests.py --module zhenmode mms
```

全球回归使用 `scripts/capture_global_regression.py` 和 `scripts/compare_global_regression.py`，完整输入、源码身份、数值记录和结果保存在独立运行目录，拒绝覆盖。重启小例使用 `scripts/verify_production_restart_cpu.py --output <新目录>`。超时或资源不足属于失败或未完成，不能推断数值通过。
全球采集器默认 `--layout product`；历史 `ocean_solver` 版本使用 `--layout canonical` 或 `legacy`，并必须指定独立冻结的 `--source-root`。它只观测原版本，不安装旧包转接层。

自动 CI 只运行 Ruff。完整测试和 MMS 通过 `workflow_dispatch` 的 `full_validation` 人工启动。数值测试通过不等于长期气候效果或公平加速结论。

内部调用直接引用所属模块；结构见 [方法与架构](production_architecture_zh.md)。旧源码的运行和严格 checkpoint 需检出对应 Git 提交，不通过路径别名或替换 hash 绕过身份检查。

直接模型运行拒绝已存在的 `global_<tag>.npz`，请为新运行选择独立 `--tag` 或 `--out-dir`。受中断的严格 checkpoint 续跑仍核对原配置与源码；已完成或已保存失败结果的目录不会被覆盖。
最终结果先写入同目录临时文件并 fsync，再以原子硬链接发布，拒绝覆盖竞争写入；checkpoint 则原子替换。文件系统须支持同目录硬链接；不支持时明确失败。发布前写入中断不会留下残缺的最终结果，异常退出可能留有独立临时文件，不阻止同名 checkpoint 续跑。
