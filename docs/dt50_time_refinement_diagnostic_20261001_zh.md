# 固定网格 dt50 成对时间细化 diagnostic（2026-10-01 UTC）

## 运行前预注册

目的仅为观察时间误差/耗散随dt缩小的变化，不是等误差性能比较或工业模式验收。原dt100 MOM能量FAIL及原2%冻结线保持。64×8×4、32000s、相同物理与初态、33个相同输出时刻；dt50、640步。MOM保留BE0.6、BEBT0.1、DT_BT_FILTER−0.25，不使用FILTER0、不调BE。两模型各最多一次，不fine、不自动修参重跑。

独立协议 `research/experiments/industrial_flat_f0/dt50_protocol.json`，运行前SHA256：`e60127b32a2c9a2c7fc75dd9a83b6c3925f08b2b479c12fe48eb17fd01be6432`。独立contract SHA256：`4b0789552bae0d70fd9d631412c0d8535bbed5ad41195a8561f7341b7e840d87`。协议绑定原合同/轨迹/score、输入、二进制及源码hash，运行前提交且要求干净tree。协议及合同不被后续结果改写。

MOM请求参数diff仅DT/DT_THERM/DT_FORCING/DTBT 100→50；预注册派生DT_TRACER_ADVECT 100→50，实际resolved必须恰好这些差异。预计DTBT50、dt_filt6.25 s、nstep1、nfilter1，每次btstep两快步；归一velocity/eta权重8/9、1/9，transport/acceleration0.9、0.1，与dt100权重相同，仅物理时间跨度减半。

单CPU/单rank、聚合4GiB，每个600s、全批1200s、新产物根合计128MiB。模型阶段由既有guard采样监督，后处理按阶段检查总时限/体量/聚合RSS，控制器本身也限CPU和地址空间；不是连续监督后处理。成功记录在预算复核后发布，时间/体量为发布前快照。

Ocean每步六字段有限、正derived层厚及冻结守恒gate；MOM保持原native MAXTRUNC=0，读所有完整保存状态验证有限和正层厚。MOM此处不声明逐快步持久化或新增内部检查。观测到非有限/非正层厚或资源失败后停止，不运行后续模型。最终score继续报告所有原指标及失败项，不掩盖失败。

## 评分和可审源码边界

原A的 `standing_wave_v0/score.py` SHA256 `29e97ecb6f18d79cc67cffe09b3ec51563e76819027fbb6a73474f7a40bc291b` 保持不变，原入口仍拒绝dt50。独立 `diagnostic_scoring.py` 加载另一个模块命名空间，仅替换该命名空间的合同准入为唯一预注册dt50差异；数组/身份/采样检查、指标公式和阈值仍为原函数。负例验证阈值/选项改动被拒绝、源hash变化被拒绝、原模块不受修改。相关91测试通过，独立静态代码门通过。诊断返回的screen结果不转成正式基线通过。

公开 `run_ocean.py` 记录首次lower/compile、首次已编译执行、其余执行、I/O、gate与初始化/总墙钟；没有额外warm-up轨迹。MOM无JIT，使用原生Initialization/Main loop及checkpoint/diagnostics子时钟；无法拆出对应首次/warm部分，不伪造同口径数值，不将不同I/O口径的积分时间相比。

Ocean每步T/S gate为相对各参考值的误差；原最终score的T/S指标是绝对最大偏差，两者都保持，不能互相替代。本例参考T15、S35。

补记上一轮计时入口只提供hash的审查缺口：现在公开 `docs/diagnostic_sources/ocean_dt100_timed_entry.py.txt` 全文，仅将唯一私有ROOT路径行替换为相对定位；原实际入口SHA256 `bd080d2b8f7fd7a6636e6508a3ae767bf7dc2eeabe734a93a60e54b4cd2c3b30`，脱敏文本SHA256 `76d248b7f01b1b7e598c2d715ee2e7d1b27de1855ea4673119c0c338925867a0`。文本用于审查而不是新的运行入口；两者路径变化明确，不声称字节相同。本轮使用已公开的新入口。
