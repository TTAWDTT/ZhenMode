# regional — 区域谱方法求解器（已退役）

**归档日期**: 2026-09-19 · **最后活跃**: 2026-08-25
**位置**: 2026-09-20 从 `src/archive_regional/` 移到 `archive/regional/`，
让 `src/` 只保留当前主线。

## 这是什么

本目录是项目**早期**的区域海洋模式：在 NW Pacific 子域上解静力原始方程，
水平方向用**伪谱法**（FFT 导数 + 谱 Poisson 求解），垂直有限差分，
时间步用 IMEX Strang 分裂（线性项走矩阵指数，非线性项显式）。

**它已被全局有限差分求解器取代**，不再作为开发主线：

| | 区域谱（本目录） | 全局 FD（当前主线） |
|---|---|---|
| 求解器 | `jax_solver.py` | `../../src/jax_solver_global.py` |
| 驱动 | `run_long_integration.py` | `../../src/run_long_integration_global.py` |
| 网格 | 128×128 区域平面 | 全球经纬网格、可调分辨率 |
| 水平算子 | FFT 伪谱 | 守恒型有限差分 |
| 域 | lon 143.6–156.3E, lat 28.6–41.3N | 全球 |

重新定位的决策记录见 `../../docs/repositioning_memo_zh.md`。

## 目录内容

### 核心 9 模块
- `jax_solver.py` — JAX JIT 求解器（区域线主入口）
- `integrator.py` — numpy IMEX Strang 分裂积分器
- `momentum.py` — 动量方程 RHS
- `tracers.py` — 温度/盐度输运
- `pressure.py` — 压力梯度与静力平衡
- `spectral_ops.py` — FFT 算子（导数、Poisson）
- `state.py` — `ModelState` 数据类与初始化
- `torch_solver.py` — PyTorch 后端求解器（性能对比用）
- `eos.py` — 线性 + UNESCO 非线性状态方程

### 驱动 / 基准 / 验证
- `run_long_integration.py`、`diagnose_stability.py`、`run_1day.py`、`run_wind_test.py`
- `bench_*.py` — 精度/性能对比（含 `bench_jax_torch.py`、`bench_climatology_compare.py`）
- `compare_jax_numpy.py`、`test_wave_speed.py`、`test_forcing.py`
- `verify_real_run.py`、`verify_real_run_long.py`、`verify_real_wind.py`、`verify_stability.py`
- `stability_scan.py`、`plot_fields.py`、`plot_animation.py`、`plot_bench.py`
- `_test_eta_bump.py`、`_test_eta_relax.py`

### 生成脚本
- `make_ppt.py`、`make_ppt_short.py` — 生成项目汇报 PPT

### 测试（`tests/`）
- `tests/test_grid.py`、`test_integrator.py`、`test_spectral_ops.py`、`test_equations.py`

这些测试不在 `tests/` 下，因此不会被 `pytest tests/`（`pyproject.toml` 的
`testpaths`）收集。如需运行（需同时把 `src/` 加进路径）：

```bash
python -m pytest archive/regional/tests/ -o "pythonpath=src archive/regional"
```

## 注意：本目录当前**不可直接运行**

- 活代码**不**依赖本目录，本仓库的测试与 CI 也不导入它。
- 2026-09-20 清理主线死代码时，`src/config.py` 与 `src/grid.py` 里只被本目录
  使用的区域辅助设施被删除：`GridConfig`、`TimeConfig`、`DEFAULT_CONFIG`、
  `make_grid`、`OceanGrid`。因此本目录的脚本 `from config import GridConfig`
  一类导入会失败——它们是历史记录，不是可运行代码。要复活任何脚本，先把它
  依赖的区域配置/网格生成器从 git 历史（`git log -- src/config.py`）里取回。
- 脚本内的导入（`from config import ...`）按仓库根为 cwd 解析：
  ```bash
  cd <repo-root> && PYTHONPATH=src python archive/regional/<script>.py
  ```
