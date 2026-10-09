# ZhenMode

ZhenMode 是基于 JAX 的全球有限差分海洋模式，原名 ocean-solver。默认方法采用静力原始方程、线性自由面和快慢步协调。Python 产品包为 `zhenmode`，正式模式位于 `zhenmode.model`。

既有生产谱系有长期稳定和低 RMSE 的历史成果，见 [长期运行来源](https://github.com/TTAWDTT/ZhenMode/commit/ad45c2b9673236c2a666c37a11439f1dd513abf9) 和 [365 天重复运行来源](https://github.com/TTAWDTT/ZhenMode/blob/bb7ba235eed5706b0adf86472e95b46bc4c021b1/research/experiments/ice_proxy_045/benchmark_365d_repeat_manifest.json)。这些记录有各自源码、配置和评分口径，不能视为当前版本已复现全部成绩。

## 安装与运行

需要 Python 3.12+；固定 CPU 开发环境可先安装根目录 `requirements.txt`：
从旧安装升级时，先卸载 `ocean-solver` 发行包再安装本项目，或使用新虚拟环境。

```sh
python -m pip install ".[dev]"
zhenmode --help
ocean-solver --help
zhenmode experiment validate experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml
zhenmode experiment run experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml --evaluate --outputs outputs/smoke
zhenmode runs list --outputs outputs/smoke
```

合成 80 秒算例使用正式工厂、生产时间步、输出和重启，验证运行链。每次运行保存独立 ID、最终配置、源码/数据/环境身份与评价结果；执行完成和指标达标分别记录。真实全球运行需要 ETOPO、WOA 和所选强迫数据，见 [输入与 case](cases/README.md)。

安装 wheel 后可从仓库外运行；实验命令加 `--root <仓库绝对路径>` 以定位配置。也可使用 `python -m zhenmode`。

## 配置与实验

[正式预设](configs/README.md)包含三套全球配方及一个合成配方。实验引用共同 case 和 parent preset，只声明变化：

```sh
zhenmode experiment expand experiments/zhenmode/global-045deg/global-045deg-seasonal-baseline.yaml
zhenmode sweep expand experiments/zhenmode/global-045deg/sweep-vertical-mixing.yaml
```

扫参默认仅展开与估算资源。GPU 实验需显式 `--backend cuda`，先固定实际数据并核对预算，见 [GPU 运行](docs/gpu_runtime_zh.md)。字段、单位、继承、消融和运行状态见 [实验说明](docs/experiments_zh.md)。

## 评价与对照

`zhenmode evaluate` 校验结果、协议、来源和可比性，生成评分报告。旧湿格等权 raw/A2 与面积加权 v2 分别标识；当前外部适配要求共享网格。误差和成本比较须使用相同物理问题、参考场、评分窗口及计时口径，见 [评价说明](docs/evaluation_zh.md)。

`zhenmode baseline mom6` 提供 `doctor → fetch → build → prepare → run → convert → evaluate`，固定上游依赖并使用隔离缓存，见 [MOM6 使用说明](docs/baselines_zh.md)。

当前推进顺序是整理代码、逐项接入对照和标准 bench，再依据量化结果改良方法。每项比较先固定共同 case、适用范围和评价协议，显式记录物理与离散差异；解析算例和受限物理实验可以分别评价对应的数值能力。

[OMIP-2 参照 benchmark 规范](docs/benchmark_spec_zh.md)定义全球海洋与海冰气候实验的共同机制、观测评价和成本口径；[机制符合表](docs/benchmark_compliance_zh.md)记录该范围的基础与缺口。可安装命令支持合同冻结与计划展开，完整执行与气候评价资格仍需单独验证。以下命令只检查或规划，不启动积分：

```sh
zhenmode benchmark describe
zhenmode benchmark plan --profile integration-6h --method zhenmode --dt-seconds 600
zhenmode baseline mom6 omip2-plan --profile integration-6h
```

`zhenmode baseline oceananigans` 提供固定原生 CUDA 对照的准备和校验。驻波的正式三方入口、协议和适用范围见[三方驻波比较](docs/standing_wave_zh.md)，九组真实运行见[实测结果](docs/standing_wave_results_zh.md)。

驻波的[空间／时间控制](docs/wave_space_time_zh.md)、[旋转调整与分层热成风](docs/channel_dynamics_zh.md)及[真实来源风应力短窗](docs/real_wind_control_zh.md)分别检验对应机制。新增 30 个原生运行、对称外模的改动和适用范围见[机制对照实测](docs/dynamics_bench_results_zh.md)。

## 开发

| 目录 | 用途 |
| --- | --- |
| `src/zhenmode/model/` | 正式模式：组装、输入、动力与物理过程、积分、诊断、输出和重启 |
| `src/zhenmode/baselines/` | MOM6、Oceananigans 原生适配与固定版本；上游源码和构建产物使用独立缓存 |
| `src/zhenmode/preparation/` | 离线资料获取、强迫映射、海岸与温盐初态准备；不推进海洋 |
| `src/zhenmode/coupling/` | 原生 SIS2 客户端、Fortran 驱动及共享网格准备 |
| `src/zhenmode/execution/` | 配置展开、实验执行、扫参、运行身份和状态管理 |
| `src/zhenmode/evaluation/` | 唯一评分实现、可比性检查和报告 |
| `cases/`, `configs/`, `experiments/`, `protocols/` | 用户编辑的问题、方法预设、试验定义和评价规则 |
| `tests/`, `scripts/`, `docs/` | 测试、维护工具、方法和使用说明 |
| `data/`, `outputs/`, `results/`, `logs/` | 本地输入与运行产物，默认忽略 |

模型内部按 `solver`（求解）、`inputs`（输入）、`runtime`（运行）、`io`（产物与重启）、`diagnostics`（诊断）组织；配置与解析验证分别为 `config.py`、`verification.py`。每个文件的职责与运行链路见 [模型导读](src/zhenmode/model/README.md)。

资料准备和耦合模块的逐文件职责见 [准备与耦合导读](docs/preparation_coupling_zh.md)。参见 [方法与架构](docs/production_architecture_zh.md)、[开发与测试](docs/development.md)。研究原型、私人监控和运行产物仅保留在本地，不参与安装或正式测试。

模型不依赖对照、实验管理或评分模块。`ocean-solver` 命令继续直接运行同一个正式模式；Python 导入改用 `zhenmode.model.*`，不保留旧包转接层。旧源码生成的严格 checkpoint 和 Python pickle 需在其原 Git 版本读取，不能通过本次路径迁移绕过源码身份校验。
