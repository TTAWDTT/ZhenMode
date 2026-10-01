# Ocean coarse 独立诊断静态资格（2026-10-01 UTC）

本轮仅静态核查，积分步数0，未导入 JAX/求解器、未编译 step、未 push。原 MOM coarse 能量 FAIL 不变。本次准备不是成对性能比较，不认证工业资格。

## 合法目的及范围

独立检查既有 ocean 生产算子在同一 standing-wave 物理合同下的原生离散能量、相位、边界和守恒行为，以定位其自身离散机制。即使 MOM 未通过也可形成这项 diagnostic；结果不得用于“相同或更好质量且显著更快”的成对声明。

网格64×8×4，dt100 s，320步，一周期32000 s；保存 t0及每1000 s完整瞬时状态，共33份。平底H100 m，g9.81，rho1025，f0，T15/S35，A0.01 m；无物理强迫、扩散、拖曳、GM/Redi、冰或海绵。ocean 使用 nodal dual、点采样、collocated 速度；MOM 使用 C-grid 和 cell-mean eta。这些数值/采样差异必须保留，不能称算子相同。

`run_ocean.py` 明确调用既有完整 `make_solver_global` 和 `_step_impl`，保留原 FFT 2/3 zonal 与五点 meridional filter、legacy forward-backward surface 等合同设置。不会用简化替代算子。

## 已通过的静态检查

- 已保存 NPZ 每个数组及成员集合逐项精确匹配 `native_arrays(0.)`；128022 bytes，SHA256 `43ffb2c82f30638ac0b79b9199459c6fc1bb0922a76fee84b5fad7df8a0160f7`。
- 既有合同与仓库 coarse-contract 文件逐字节一致。
- `jax_solver_global.py` SHA256 `0ee5874cef57a318e534edb9b99690c797a4ecb1be95c4f0283f957e32ff4169`，匹配既有纯边界掩膜见证；见证仅覆盖最终掩膜，不证明完整 step 的 y 不变性。
- 原生输出保存 u/v/T/S/eta/ice、静态原厚度及明确标为 derived 的 eta-adjusted 几何厚度；后者不是原生移动层输运状态。每份有 hash 与工程 gate，拒绝后不保存成成功状态。
- 每步 gate 检查六字段有限性、正厚度、冻结体积/示踪物/v/ice门槛；原评分函数及阈值保持。

## 运行前仍须落实

单CPU亲和、单进程、CPU JAX、专用非root账户；guard聚合4 GiB、每项600 s、合计128 MiB输出。现有两项MOM对照后根目录约102 MB；按33份未压缩原生小状态估计 ocean 原生输出约4 MB，另加export/日志，先留10 MiB额度足够的可能性较高，但实际仍由共享根目录guard限制。

没有现成 ocean 粗网格实测时间，不能借 MOM 约几秒来估计 JAX。建议只预留600 s总墙钟，其中首次导入/JIT编译包含在内；编译若超过内存/时间限即停止，不增配、不循环重试、不另跑warm-up。4 GiB地址空间限制可能先阻止JAX运行；因此目前状态为 **static-ready / runtime-unqualified**。

下列为最小后续命令模板，路径变量由执行环境绑定，不含公开的私有绝对路径；当前未执行：

```sh
"$PYTHON" "$REPO/research/experiments/industrial_flat_f0/run_guard.py" \
  --run-directory "$RUNROOT/ocean-coarse-diagnostic" \
  --wall-seconds 600 --output-budget-directory "$RUNROOT" \
  --output-limit-bytes 134217728 -- \
  "$PYTHON" "$REPO/research/experiments/industrial_flat_f0/run_ocean.py" \
  --run-directory "$RUNROOT/ocean-coarse-diagnostic" \
  --input-file "$RUNROOT/frozen-inputs/ocean_native_inputs.npz"
```

上述命令需由专用非root账户运行。完成后才进行只读 export/原评分，并用独立记录明确 diagnostic；不覆盖原基线、不混入成对 speedup。运行前确认账户无任务、源/input hash未变、CPU资源与预算可用。
