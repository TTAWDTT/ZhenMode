# 驻波空间／时间分离协议

2026-10-09：扩展正式驻波入口，保留 v0/v1 原协议、指标和阈值。新的
`standing-wave-space-time-v1` 只接受下列五个组合；ny=8、nz=4、物理尺寸、初态、
模式方案和输出时刻均沿用已冻结的三方驻波定义。

| 控制轴 | nx | dt（秒） |
| --- | --- | --- |
| 空间，固定 dt=25 | 64、128、256 | 25 |
| 时间，固定 nx=256 | 256 | 100、50、25 |

两轴共享 nx=256、dt=25 的一次运行，三种模式共 15 个原生运行。每个 GPU 工作进程
要求真实 CUDA，最多一核 CPU、4 GiB 主机 RSS、600 秒；运行逐个执行。
新配置先测短窗口，记录编译／准备及稳定窗口积分成本，再决定正式运行。

```sh
zhenmode evaluate wave freeze --study 64 25 --out NEW_CONTRACT.json
zhenmode benchmark standing-wave --study 64 25 --output NEW_RUN \
  --mom-executable PINNED_MOM6 --mom-source PINNED_SOURCE \
  --oceananigans-cache VERIFIED_CACHE --julia ISOLATED_JULIA \
  --source-revision ZHENMODE_COMMIT
zhenmode evaluate wave-study --runs NX64_DT25 NX128_DT25 NX256_DT25 NX256_DT50 NX256_DT100 \
  --output NEW_STUDY.json
```

分析器重新读取、校验和评分实际输出，核对保存的输出哈希和指标、完整 5×3 计数，
要求共同包字节及各模式的源码／可执行程序身份一致。输出两条控制轴上的原有指标，
以及投影海面和深度平均速度得到的模态相位、振幅。
相位及对数振幅对无量纲时间 t/P 作线性最小二乘；频率偏差为斜率/(2π)−1，
振幅趋势为每周期对数变化。两种拟合均保存最大残差；趋势不代表瞬时误差，
也不作为完整模式阶数或严格可加的空间／时间误差分解。

真实运行前冻结的控制为：独立闭式波在五个组合上通过；预置 ±1% 频率偏差、
每周期 0.02／0.05 对数衰减回收误差不超过 2e−14；不支持的组合、布尔数值和
篡改的步长拒绝。旧评分的独立参考生成器继续使用，不调用被测解析函数。

工具选择：已查 BootLoops 工具索引，没有海洋模式运行器；本仓库已具备三方驱动、
资源监督、原生数组校验和评分，故只扩展其冻结配置并增加模态分析，复用全部驱动。
本轮数据用于误差诊断和方法选择；用于选择的值不再作为后续独立认证数据。

## 新外模研究

`--method symmetric-external-mode` 将相同五个组合冻结为独立的
`standing-wave-space-time-method-v1`，只改变 ZhenMode 外模选择。原研究的默认定义、
三方完整性要求和报告保持。单模式分析必须显式选择 `--models zhenmode`，仍要求五个
组合全部存在、同一方法和同一源码／程序身份；报告记录实际模型列表与五个运行，
缺少对照组不能被视为完整三方研究。不同方法的目录不得混进一个研究。

```sh
zhenmode evaluate wave freeze --study 64 25 --method symmetric-external-mode --out NEW.json
zhenmode benchmark standing-wave --study 64 25 --method symmetric-external-mode \
  --models zhenmode --output NEW_RUN --source-revision ZHENMODE_COMMIT
zhenmode evaluate wave-study --models zhenmode \
  --runs NX64_DT25 NX128_DT25 NX256_DT25 NX256_DT50 NX256_DT100 --output NEW_STUDY.json
```

研究中的振幅固定为 0.01 m，`--study` 与 `--half` 同用会拒绝。与旧研究比较时分别
保留各自的源码和数据身份，旧对照不计作新源码的运行，也不用于跨硬件速度排名。
