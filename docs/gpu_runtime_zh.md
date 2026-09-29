# GPU 运行环境与验证边界

Lady，2026-09-29。这里记录实际设备/环境，不代替数值或工业模式验收。

## 已确认的运行环境

- 本机 RTX 4060 Laptop GPU，8188 MiB，驱动 616.56；WSL2 Ubuntu 24.04。
- 独立 Linux Python 3.12.3：JAX/jaxlib/CUDA13 plugin 0.11.2，NumPy 2.5.3，
  SciPy 1.18.1，netCDF4 1.7.4。具体 NVIDIA 库版本另存于本机 `WSL_GPU_pip_freeze.txt`。
- 实际环境数据在 D 盘项目的 `.venv/gpu-wsl/`，已被既有 `.venv/` 规则忽略。
  原入口 `/root/.venvs/ocean-solver-gpu` 为指向该目录的符号链接，以保留绝对 shebang。
  该入口只属于当前本机 WSL 用户，不能作为其他机器的通用路径。
- 原 Windows conda `ocean-solver` 未修改。原生 Windows 的 JAX 是 CPU 后端，
  不是本次 CUDA 运行环境；不安装或替换 NVIDIA 驱动。

[JAX 官方安装文档](https://docs.jax.dev/en/latest/installation.html)将原生 Windows
NVIDIA 后端标为不支持，WSL2 标为实验性。CUDA13 的 Linux 驱动要求至少 580；
当前设备已通过真实 CUDA float64 运算检查，未禁用版本约束，未回退 CPU。
这只证明后端可用，不证明海洋模式精度、吞吐或长期可靠性。

## 可复用配置

仓库的 [`environment-gpu.yml`](../environment-gpu.yml)用于 **Linux/WSL 中的 conda**：

```bash
conda env create -f environment-gpu.yml
conda activate ocean-solver-gpu
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -c 'import jax; print(jax.devices())'
```

本机实际通过隔离 venv 安装，而不是执行过上述 conda 文件；不能混报两种环境。
若 Ubuntu 的系统 Python 缺少 ensurepip，使用合适的 Linux conda 或配置 venv 依赖，
不要把空路径传给 `python -m venv`，也不要在仓库根创建系统依赖目录。
本轮 PowerShell/WSL 引号传递曾产生空目标，已验证并清理仅由该操作生成的目录；
后续安装/迁移通过保存的 Linux 脚本执行。失败日志也保留，不隐去环境问题。

CUDA 测试必须明确设置 `JAX_PLATFORMS=cuda`；失败时停下处理，不能靠 CPU 回退
宣称 GPU 通过。为了避免占满共享显存，本机验证设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
不使用 `JAX_SKIP_CUDA_CONSTRAINTS_CHECK`，也不删除其他用户的缓存或停止 WSL。

本机已有环境可在 **WSL/Linux shell** 中验证原 FD 候选：

```bash
cd /mnt/d/Github/ocean-solver
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/gpu-wsl/bin/python -m pytest tests/test_legacy_subcycled_rk.py -q
```

这不是 Windows PowerShell 下直接执行 ELF Python 的命令。
`src/run_long_integration_global.py` 的生产默认仍为旧方案；GPU 后端可用不自动切换
数值方案，也不使仅在 API/smoke 中选择的新候选成为正式生产路径。

## 本次验收范围

合同/环境/冻结源码及报告位于本机
`results/legacy_repair/m3_rk_C_20260929T062246Z/`，这些数据没有进入 Git。

1. 强制 CUDA 的 float64 运算及设备/库版本清单。
2. 原 FD 的独立非线性过程、实际子步账本、干节点/常量、单次源与三步 AD 检查。
3. 同一 WSL 环境、同一 WOA/NCEP 固定月输入、相同参数的 CPU/CUDA 一天重放。
   float64 六字段门槛预注册为 `max_abs_error <= 1e-9*(1+reference_max_abs)`。
4. 初态及参数数组须逐位相同；保存实际账本，而不只比较“两个都没爆炸”。

初次验证未验收 float32、多 GPU、百年 AD、气候/预报评分或工业模式优势。
时间记录区分编译与执行；初次报告不作公平加速比声明。先验证相容性，再预注册
相同环境、同精度、预热/同步、重复采样和硬件占用条件的独立吞吐实验。
库存/时间阶数的既有未通过项仍见[原核心验收记录](legacy_core_repair_status_zh.md)。

实际 23 项 CUDA 局部测试已通过，两后端的一天重放均接受 144 步。
直接分别构造参数时有两个派生常量的末位差异，严格逐位参数门槛失败；保留原报告，
不放宽门槛。改用同一冻结有效常量后，实际 GPU 初态/参数回读逐位相同，
六字段末态差异最多 `2.985e-13`（各字段单位分别核对），通过原定相容限。
这不是修正积分末态；也不改变两后端仍未通过的真实物理库存结论。
