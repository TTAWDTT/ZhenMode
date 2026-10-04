# ZhenMode

ZhenMode 是基于 JAX 的全球有限差分海洋模式，原名 ocean-solver。默认方法采用静力原始方程、线性自由面和快慢步协调；Python 包名为 `ocean_solver`。

既有生产谱系有长期稳定和低 RMSE 的历史成果，见 [长期运行来源](https://github.com/TTAWDTT/ZhenMode/commit/ad45c2b9673236c2a666c37a11439f1dd513abf9) 和 [365 天重复运行来源](https://github.com/TTAWDTT/ZhenMode/blob/bb7ba235eed5706b0adf86472e95b46bc4c021b1/research/experiments/ice_proxy_045/benchmark_365d_repeat_manifest.json)。这些记录有各自源码、配置和评分口径，不能视为当前版本已复现全部成绩。

## 安装与运行

需要 Python 3.12+；固定 CPU 开发环境可先安装根目录 `requirements.txt`：

```sh
python -m pip install ".[dev]"
zhenmode --help
ocean-solver --help
zhenmode experiment validate experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml
zhenmode experiment run experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml --evaluate --outputs outputs/smoke
zhenmode runs list --outputs outputs/smoke
```

合成 80 秒算例使用正式工厂、生产时间步、输出和重启，验证运行链。每次运行保存独立 ID、最终配置、源码/数据/环境身份与评价结果；执行完成和指标达标分别记录。真实全球运行需要 ETOPO、WOA 和所选强迫数据，见 [输入与 case](cases/README.md)。

安装 wheel 后可从仓库外运行；实验命令加 `--root <仓库绝对路径>` 以定位配置。也可使用 `python -m ocean_solver`。

## 配置与实验

[正式预设](configs/README.md)包含三套全球配方及一个合成配方。实验引用共同 case 和 parent preset，只声明变化：

```sh
zhenmode experiment expand experiments/zhenmode/global-045deg/global-045deg-seasonal-baseline.yaml
zhenmode sweep expand experiments/zhenmode/global-045deg/sweep-vertical-mixing.yaml
```

扫参默认仅展开与估算资源。字段、单位、继承、消融和运行状态见 [实验说明](docs/experiments_zh.md)。

## 评价与对照

`zhenmode evaluate` 校验结果、协议、来源和可比性，生成评分报告。旧湿格等权 raw/A2 与面积加权 v2 分别标识；当前外部适配要求共享网格。误差和成本比较须使用相同物理问题、参考场、评分窗口及计时口径，见 [评价说明](docs/evaluation_zh.md)。

`zhenmode baseline mom6` 提供 `doctor → fetch → build → prepare → run → convert → evaluate`，固定上游依赖并使用隔离缓存，见 [MOM6 使用说明](docs/baselines_zh.md)。

## 开发

| 目录 | 用途 |
| --- | --- |
| `src/ocean_solver/` | 正式模型、实验执行、对照接入与统一评价 |
| `cases/`, `configs/`, `experiments/`, `evaluation/` | 问题定义、预设、实验与评价协议 |
| `tests/`, `scripts/`, `docs/` | 测试、维护工具、方法和使用说明 |
| `data/`, `outputs/`, `results/`, `logs/` | 本地输入与运行产物，默认忽略 |

参见 [方法与架构](docs/production_architecture_zh.md)、[开发与测试](docs/development.md)。研究原型、私人监控和运行产物仅保留在本地，不参与安装或正式测试。
