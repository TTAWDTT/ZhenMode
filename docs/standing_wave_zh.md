# 三方驻波比较

这个 bench 比较同一个平底、无旋转、均匀温盐的自由表面驻波问题。ZhenMode 调用既有完整 FD 工厂和时间步，MOM6 调用固定原生可执行文件，Oceananigans 调用固定版本的 `HydrostaticFreeSurfaceModel`；评分在各自原生位置进行。

## 运行

在 Linux/WSL 中准备 Julia 1.10+ 和真实 CUDA，使用独立项目缓存。Julia 启动器可以通过 `--julia` 指定；本机 BootLoops 环境使用其已安装隔离启动器。包服务可通过标准 `JULIA_PKG_SERVER` 环境变量选择；本机使用 `https://us-east.pkg.julialang.org`。模型版本会严格核对，拒绝静默降级。

```sh
zhenmode baseline oceananigans prepare --cache "$HOME/.cache/zhenmode/oceananigans/0.113.5" --julia julia
zhenmode baseline oceananigans doctor --cache "$HOME/.cache/zhenmode/oceananigans/0.113.5" --julia julia
zhenmode benchmark standing-wave --case coarse --output NEW_OUTPUT \
  --mom-executable PINNED_MOM6_EXECUTABLE --mom-source PINNED_MOM6_CHECKOUT \
  --oceananigans-cache "$HOME/.cache/zhenmode/oceananigans/0.113.5" --julia julia \
  --source-revision ZHENMODE_COMMIT
```

`--case medium`、`fine` 分别加密水平网格并减半时间步。`--models` 可选择一个或多个模式作范围明确的复跑；默认运行全部三种，始终逐个执行。安装包在仓库外运行时显式给出源码提交；实际执行包字节另有完整记录，不把报告提交代替源码身份。输出目录必须是新目录，失败也保留已有状态和日志。

## 问题和差异

共同条件：H=100 m，g=9.81 m/s²，rho0=1025 kg/m³，T=15°C，S=35 psu，A=0.01 m，f=0；x 周期、y 封闭自由滑移。一个周期 32000 秒，保存 0 至 32000 秒、间隔 1000 秒的 33 个瞬时状态。风、热盐源、恢复、底拖曳、显式混合、GM/Redi、冰、海绵和极区平均均关闭，动量、连续性与标量输运保留。

| 模式 | 位置和几何 | 已冻结方案 |
| --- | --- | --- |
| ZhenMode | FD 点值；固定 nodal dual 容量；输出 h 加入显式诊断的表面位移 | 已有完整 legacy 步，mode_split=false，conservative_kv=true |
| MOM6 | C 网格；eta 初始化为精确格均值；四个原生移动层，无 ALE 重映射 | f49a000；SPLIT=true、SPLIT_RK2B=false；声明的零非绝热源问题采用 ADIABATIC=true |
| Oceananigans | C 网格；eta 精确格均值；z-star 按原生网格实际层厚输出 | 0.113.5、作者提交1e8587b；原生 SplitRungeKutta3、SplitExplicitFreeSurface，请求8个子步；保留作者默认时间平均核，并记录实际子步设置 |

Oceananigans 的原生线性浮力使用相对于15°C/35 psu的 T/S 异常，以匹配共同密度参考；另推进两个非零物理温盐标量见证，保持常量输运检查有意义。它因此推进四个标量；该额外工作明确记录，不据此进行速度排名。选择原生 RK3 的依据是[固定作者 z-star 守恒测试](https://github.com/CliMA/Oceananigans.jl/blob/1e8587b17171b0bba5bc6728118c3dbbd3c8acf6/test/vertical_coordinate/conservation_explicit.jl#L11)对 AB2 守恒性的说明。

四个参考层容量为 [100/6,100/3,100/3,100/6] m。初始表面位移在 FD 诊断及 MOM 中作用于顶层，在 Oceananigans 中通过原生 z-star 均匀伸缩全部层；协议 v1 明确这项离散差异。MOM 和 Oceananigans 的 eta 按格均值与对应解析格积分比较，FD eta 按点值比较，u 按原生位置比较。没有隐含重网格。

## 评分和结果

本轮九组真实运行与适用范围见[实测结果](standing_wave_results_zh.md)。

```sh
zhenmode evaluate wave freeze --case coarse --out CONTRACT.json
zhenmode evaluate wave score --contract CONTRACT.json --output MODEL/output.npz --report NEW_SCORE.json
# 原 v0 历史数据可使用其原合同，或显式冻结 v0：
zhenmode evaluate wave freeze --case coarse --schema standing-wave-v0 --out LEGACY_CONTRACT.json
```

v0 和 v1 共用唯一的评分实现。既有固定尺度 eta／深度平均 u 的时间空间 RMS、模态相位与振幅、动势能变化、体积和常量 T/S 指标保持。速度误差指标是深度平均速度，不等于三维速度误差。失败的工程阈值原样保留，运行完成与阈值通过分别记录。

解析参考来自线性小振幅波，三个原生模式执行完整非线性动力学；A/H=1e-4，对应有限振幅误差底。此次属于明确范围的数值比较，所有报告保持 `industrial_qualified=false`、`performance_comparison=false`。既有 MOM6 缓存二进制的身份可核对，但历史源到二进制完整构建收据缺失，该限制随报告保存。

## 代码职责

- `benchmarks/standing_wave.py`：共同物理定义、版本化合同及数值方案声明。
- `evaluation/standing_wave.py`：唯一解析参考、原生几何／输出校验及评分；不调用模式。
- `execution/standing_wave.py`：串行资源监督、来源和状态记录、共同结果序列化。
- `execution/wave_zhenmode.py`：既有 FD 核心的 case 组装和原生输出。
- `baselines/mom6/standing_wave.py`：MOM 输入、实际运行、已生效参数核查及诊断转换。
- `baselines/oceananigans/`：隔离 Julia 环境、固定源码、原生驱动与输出转换。

正式模型不依赖这些实验／对照模块。测试参考数组在 `tests/support/standing_wave.py` 独立生成；评分器不得调用被测模式或用自己的解析函数生成测试答案。原型原文及历史证据保留在本地冻结归档；新入口不依赖研究目录。
