# ZhenMode

ZhenMode 是基于 JAX 的全球经度周期、纬度截断有限差分海洋模式，原名 ocean-solver。默认方法是既有 FD 生产主线，使用线性自由面、快慢步协调及既有物理过程；包名仍为 `ocean_solver`。

历史有约 194.7 年、192–274 年阶段的 PASS，以及从约 273 年 checkpoint 续跑到累计约 325 年的记录。旧协议下 365 天重复运行保存了 raw SST RMSE 约 0.91364°C、A2 约 1.11257°C。这些是原生产谱系的历史成果，仍有 OHC/AMOC 等警告；不是同一源码从零连续 325 年，也不表示当前版本已复现全部成绩。原材料、字节校验、旧评分口径和来源边界见 [历史证据索引](archive/evidence/README.md)。

## 安装与一次完整的小运行

使用 Python 3.12+。正式安装不包含未采用的研究方法：

```bash
python -m pip install ".[dev]"
zhenmode --help
ocean-solver --help
zhenmode experiment validate experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml
zhenmode experiment run experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml --evaluate --outputs outputs/smoke
zhenmode runs list --outputs outputs/smoke
```

同一入口也可用 `python -m ocean_solver`。合成 80 秒算例使用正式工厂、完整生产时间步、输出和版本化重启，输入由安装包生成；它验证工程链，不代表真实海洋效果。每次运行创建独立 run ID、最终生效配置、源码/数据/环境身份、日志、输出及评价报告。运行完成、指标验收和可比性分别记录。托管本地运行限制为单 CPU、180 秒、4 GiB；超时保留失败。

从仓库外安装 wheel 后仍能运行上述命令，只需给实验命令传 `--root <仓库绝对路径>`。配置和协议在仓库中，数值实现不依赖仓库 `src`、tests 或 research 的临时搜索路径。正式全球运行需要 ETOPO、WOA 和所选 NCEP 数据，见 [case 与输入引用](cases/README.md)；没有所需文件会明确拒绝。

## 方法、配置和新实验

| 路径 | 职责 |
| --- | --- |
| `src/ocean_solver/{model,runtime,config,state,geometry}` | 模型装配、运行、定义与几何 |
| `numerics`, `dynamics`, `physics`, `timestepping` | 数值基础、动力过程、参数化、时间协调 |
| `forcing`, `io`, `diagnostics`, `audit`, `provenance` | 强迫与文件读取、输出/重启、诊断和实际来源 |
| `cases/`, `configs/<method>/presets/`, `experiments/` | 共同物理问题、多套来源明确的预设、试验/消融/扫参 |
| `evaluation/`, `ocean_solver.evaluation` | 版本化协议及统一评价编排；算法仅在 `validation/benchmarks` |
| `baselines/`, `ocean_solver.baselines`, `interop/mom6` | 外部模式的固定源码/构建/输入/运行/转换接入 |
| `research/`, `archive/` | 单独安装的未采用方法、独立 oracle、失败材料及历史索引 |
| `tests/`, `scripts/`, `docs/` | 对应合同的测试、维护/历史复现入口和说明 |
| `data/`, `outputs/` | 本地输入缓存与独立运行产物，默认不提交大文件 |

已有三套真实全球配方及一个合成预设，见 [预设索引](configs/README.md)。全球配方来自既有生产脚本，不是本轮新推荐参数或复跑成绩。实验引用 case 和 parent preset，只声明必要变化；完整 CLI 默认和分辨率缩放在运行前展开。未知字段、单位错误、重复/歧义覆盖、循环继承和不成立的单因素消融被拒绝。

```bash
zhenmode experiment expand experiments/zhenmode/global-045deg/global-045deg-seasonal-baseline.yaml
zhenmode sweep expand experiments/zhenmode/global-045deg/sweep-vertical-mixing.yaml
```

扫参只展开和估算资源，不启动大量运行。新实验的字段、状态及例子见 [实验说明](docs/experiments_zh.md)。`zhenmode model` 与原 `ocean-solver` 使用相同生产参数；内部调用直接使用所属模块，旧 FD/data 路径和 bare-module 转接入口已删除。

## 评价与 MOM6

`zhenmode evaluate --help` 提供结果校验、评分、报告和可比性检查。现有 SST 评分参照保存的初始表层场，不能称为独立观测验证。旧湿格等权 raw/A2 和面积加权 v2 保持不同身份，禁止混排；当前外部接入要求共享网格，没有隐含重网格。编译、纯积分、IO 和端到端成本分别表示，缺项不猜测，不据此宣称同等误差下的加速。

MOM6 在仓库外固定源码、FMS/CVMix/GSW 依赖和隔离缓存，提供 `doctor → fetch → build → prepare → run → convert → evaluate`。Linux/WSL 复用已构建缓存的小例：

```bash
zhenmode baseline mom6 doctor --cache /root/.local/share/ocean-solver-industrial-reference
zhenmode baseline mom6 prepare --cache /root/.local/share/ocean-solver-industrial-reference
```

随后对 `prepare` 返回的独立目录执行 `run --run-dir`、`convert --run-dir` 和 `evaluate --run-dir`。已有 tc1 实测链走通，但与全球 FD 问题不同，明确不可用于全球排名。共同 30 天问题的 native 输入接入、强迫转换和受限比较要求见 [MOM6 说明](docs/baselines_zh.md)；本轮没有重新构建 MOM6 或启动 30 天全球积分。

## 验证与历史复现

开发全套测试另外安装研究 distribution，正式方法不反向依赖它：

```bash
python -m pip install -e ./research
python -m ruff check .
python -m pytest tests/experiments tests/evaluation tests/baselines -q
python scripts/verify_historical_evidence.py
```

本地数值验证使用有界 runner；Windows 示例：

```powershell
python scripts/run_bounded_research_tests.py --module ocean_solver mms
```

自动 CI 只检查 Ruff；全套测试、合成浴深和 MMS 需在人工验收后手动选择 `full_validation` 启动。独立数值 oracle 与负例保留。冻结集合、生产 fp32/fp64 数值对比、输出/重启、真实 MOM6 小例、外部目录干净安装及失败/跳过记录见 [本轮验证报告](docs/research_engineering_validation_zh.md)。工程回归或 CI 通过不等于长期气候资格、工业级资格或速度优势。

后续在原资源上限内完成真实数据的 1° 全球 12 小时前后对照及 checkpoint 续跑，数值逐字节一致；0.45° 首次编译超时及单独的缓存接线修复也保留在 [全球回归报告](docs/global_regression_validation_zh.md)。这没有重新授予历史全年 RMSE 或百年稳定性结论。

[目标架构](docs/research_engineering_architecture_zh.md)、[生产职责](docs/production_architecture_zh.md)、[迁移记录](docs/research_engineering_layout.json)、[全部文档](docs/README.md)提供进一步说明。材料库存、FV/C-grid 和 r-star 留在研究区；候选 353→354 步失败不能否定原生产历史，PR29 不属于本轮方法或交付。
