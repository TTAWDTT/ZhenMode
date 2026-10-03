# GPU 环境

`environment-gpu.yml` 定义 Linux/WSL 的 GPU 开发环境：

```sh
conda env create -f environment-gpu.yml
conda activate ocean-solver-gpu
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python -c 'import jax; print(jax.devices())'
```

运行时显式选择后端并记录设备、精度和依赖版本。CUDA 设备检查仅验证环境；性能实验还需记录编译、预热、同步、积分和 IO 的计时范围。

模型运行入口与 CPU 相同，见 [README](../README.md)。数值验证和资源约束见 [开发说明](development.md)。
