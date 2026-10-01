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

## 完成结果（2026-10-01 UTC）

预注册提交 `2a98697b231f4fe17aba6ab5158a5e7fe4bf8acd` 在积分前完成。一次预启动Git检查因Windows worktree元数据格式无法被WSL默认路径解析而退出，未创建模型启动标记、无模型guard/积分。随后只做Git路径映射与公开CSV的既有LF字节规范化，未改源码/协议/hash/科学参数；Windows和WSL均确认干净工作树后启动。MOM与ocean实际各一次，均640步/33个完整状态、exit0、无stop reason且后代清理确认。没有数值重跑。

MOM实际resolved diff恰好DT、DT_THERM、DT_FORCING、DTBT、DT_TRACER_ADVECT 100→50；BE/BEBT/filter等其余参数不变。有效DTBT50、滤波长度6.25 s；权重与nstep/nfilter由实际参数和pinned源码推得，与预注册一致，不声称另外捕获了运行中的权重数组。BE predictor时段由60 s减为30 s。ocean实际dt50、steps640，640次原相对工程gate通过。

| 指标 | MOM dt100 原FAIL | MOM dt50 diagnostic | ocean dt100 diagnostic | ocean dt50 diagnostic |
|---|---:|---:|---:|---:|
| eta error | 0.005826980423 | 0.003227181623 | 0.005148928620 | 0.003982881954 |
| u error | 0.003155499231 | 0.001711246893 | 0.002958724069 | 0.002963631815 |
| 最大相位误差 rad | 0.009446376374 | 0.005689631983 | 0.012597128981 | 0.010300152005 |
| 最大振幅误差 | 0.013475163010 | 0.006866557587 | 0.002459404879 | 0.001227455226 |
| 最大绝对能量偏差 | 0.026510930141 | 0.013350180987 | 0.004924858537 | 0.002456417206 |
| 末时有符号能量变化 | −0.021679708130 | −0.010899599016 | −0.000098626236 | −0.000049406180 |
| 体积相对误差 | 2.22045e−16 | 2.22045e−16 | 2.22045e−16 | 2.22045e−16 |
| T最大绝对偏差 | 6.57252e−14 | 1.20792e−13 | 0 | 0 |
| S最大绝对偏差 | 1.77636e−13 | 2.84217e−13 | 0 | 0 |
| v/U | 0 | 0 | 0 | 0 |

两项dt50 diagnostic的全部原阈值screen通过，原dt100 MOM FAIL仍保留。33状态无非有限值；MOM最小保存层厚16.65668272273631 m，ocean最小16.656678712104615 m。MOM原生En最大偏差0.013350180987082072，末时变化−0.010899599016218842，与scorer一致。ocean能量来自原生字段/声明的derived几何诊断，不是假称原生En。各33时刻K/PE/E、模态系数、振幅与相位见对应公开CSV；所有指标及hash见scalar JSON。

## 资源与计时实测（2026-10-01 UTC）

全批含分析23.95193 s；完成发布前产物56870283 bytes，128MiB限内。单CPU、聚合4GiB限内，模型guard持续采样/后处理阶段复核。完成记录是发布前快照，不把它作为持续测量。

| 计时/资源 | MOM dt50 | ocean dt50 |
|---|---:|---:|
| guard总墙钟 s | 8.427486599 | 13.884083602 |
| 聚合采样峰值RSS bytes | 145907712 | 905641984 |
| 初始化 s | 0.034499 | 11.502760649 |
| cold lower/compile s | 不适用，无JIT | 3.621205309 |
| 首次已编译执行 s | 无独立时钟 | 0.004722700 |
| 其余已编译执行累计 s | 无独立时钟 | 1.202090823 |
| 积分时钟 s | Main loop含I/O 7.882114 | 同步执行1.206813523 |
| 输出I/O s | checkpoint子时钟0.035549，diagnostics子时钟0.052390 | 0.059217755 |
| 工程gate累计 s | 原生检查无独立时钟 | 0.086035842 |

MOM checkpoint33次、diagnostics framework4480次；子时钟有层级，不能相加为全I/O或与ocean文件写时钟等同。ocean初始化包含compile/t0 I/O，cold指新进程首次lower/compile，没有额外warm-up；积分等于first+remaining。两个模型的不同计时口径不可用来宣称等误差speedup。

## 解释与下一步边界（2026-10-01 UTC）

最大能量偏差dt50/dt100比：MOM0.503572712、ocean0.498779242；振幅误差也近减半，支持本例有随dt近线性缩小的时间相关能量扰动/耗散。仅两个dt不能确立通用收敛阶，也未隔离全部机制。MOM没有用FILTER0或调整BE。

ocean的u error几乎不变且略增，phase改善有限。全湿、平底、均匀度量时，`jax_solver_global.py:204–228,374–419` 的平均面通量差/伴随梯度可约化为中央差分，空间符号sin(k dx)/dx给一周期色散估计0.0100883261 rad，接近dt50 phase最大误差0.0103001520 rad。这只是空间误差主导的候选解释，未对完整legacy半步/耦合/滤波作严格证明；下一项最小动作是对既有33时刻与实际完整离散更新做独立符号/色散审查，先不再扫dt。

同dt50下ocean能量/振幅误差更小，但eta/u/phase误差比MOM更大。因此不能宣称全指标相同或更好质量，更不能据此宣称正式工业目标已达成。本实验仍是一周期理想诊断，无真实强迫/海岸/地形/长期稳定性证据。后续空间细化或正式配对性能实验另行设计；本轮没有fine或额外积分。

## 有效半步及能量符号补充（2026-10-01 UTC）

ocean `mode_split=False`，`jax_solver_global.py:2438–2455` 每全步两次linear half-step，`:1958`各调用free-surface：有效时段由dt100时的50 s减为dt50时的25 s，不是独立快模subcycle；中间nonlinear全步由100 s减为50 s。`:995–1020` 的非线性乘积dealias保持原FFT mask及meridional `[1,4,6,4,1]/16` 权重，不随dt重调。

能量最大绝对偏差的符号必须区分：MOM dt50最差29000 s为 **−1.3350181%**，ocean dt50最差4000 s为 **+0.2456417%**。ocean末时仅−0.0049406%，故其max|E/E0−1|减半主要表明能量波动幅度缩小，不能单凭最大值把它解释为净耗散减半。原33时刻CSV保留完整有符号曲线供独立判断。
