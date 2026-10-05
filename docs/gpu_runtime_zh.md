# GPU 环境

`environment-gpu.yml` 定义 Linux/WSL 的 GPU 开发环境：

```sh
conda env create -f environment-gpu.yml
conda activate ocean-solver-gpu
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python -c 'import jax; print(jax.devices())'
```

运行时显式选择后端并记录设备、精度和依赖版本。CUDA 设备检查仅验证环境；性能实验还需记录编译、预热、同步、积分和 IO 的计时范围。

托管实验使用 `zhenmode experiment run <实验.yaml> --backend cuda --evaluate --outputs <新结果目录>`。先用同样参数加 `--dry-run` 核对配置与资源；CUDA 资源声明可以超出 CPU 默认上限，但不得超过单 CPU、3 小时、8192 MiB 主机 RSS。详细计时与采样限制见 [实验说明](experiments_zh.md)。真实输入的路径、版本、校验值须先固定；仓库预设和数据引用不代表已经完成当前年积分。

独立 GPU 编译可能因自动调优选择不同的浮点归约顺序，见 [OpenXLA 的确定性说明](https://openxla.org/xla/determinism)。需要核查复现时，可显式用 `XLA_FLAGS=--xla_gpu_autotune_level=0` 执行两次独立短窗，比较原始状态及记录；该环境选项记录在每次 manifest 中，程序不自动添加。关闭调优可能改变吞吐和浮点结果，需以实测为准，不能凭一次小例宣称所有 GPU 运行逐位确定，也不能与默认编译环境的时间混作同一测速条件。

模型运行入口与 CPU 相同，见 [README](../README.md)。数值验证和资源约束见 [开发说明](development.md)。
