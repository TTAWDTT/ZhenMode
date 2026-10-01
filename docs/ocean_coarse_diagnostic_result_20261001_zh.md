# Ocean coarse 单次独立诊断结果（2026-10-01 UTC）

获准的一次运行正常完成320步，33个完整瞬时状态。没有修参重跑，原MOM coarse FAIL及冻结合同保持。这是 diagnostic 的工程筛查结果，不是成对性能验收、工业质量认证或 speedup 证据。

生产 `src/` 与冻结源码base `8ad3552ff651cf45dfec7ae9a119ac75eef28890` 没有diff。隔离入口仅增加阶段/计时与导入路径记录，独立静态审查通过；原生产源码、输入、physics、step及gate未改变。

| 冻结指标 | 本次结果 |
|---|---:|
| eta error | 0.0051489286195498 |
| u error | 0.002958724068759961 |
| 最大相位误差 rad | 0.01259712898118875 |
| 最大振幅误差 | 0.002459404879480598 |
| 最大绝对能量偏差 | 0.004924858537119103 |
| 体积相对误差 | 2.220446049250313e−16 |
| T / S 最大偏差 | 0 / 0 |
| v/U | 0 |

所有冻结指标在本次 diagnostic 工程screen内通过。最大能量偏差出现在4000 s；初始E=25195174085.693478 J，末时有符号相对变化−0.00009862623648115854。33时刻的K、自由面PE、E、模态系数、振幅比与相位误差见同目录 `ocean_coarse_energy_modal_20261001.csv`。能量来自冻结scorer对原生字段和明确声明的derived geometric quadrature计算；ocean没有此次可用的原生总能量诊断，不称其为原生En。静态原厚度和derived厚度区别保留。

完成状态意味着320次工程gate全部通过；33保存状态无非有限值，最小保存层厚16.656678712104615 m。各步检查非有限、非正层厚、守恒和合同限制，失败即抛错，本次无失败阶段。没有宣称逐步持久化完整状态或通用物理正确性。

## 资源和计时口径

单rank/CPU、CPU JAX、专用非root账户。guard exit0，无stop reason，后代清理确认。

- guard总墙钟14.070851685 s；child总墙钟13.111308829 s。
- 初始化12.444996332 s，**其中包含**cold lower/compile 3.663682707 s和t0输出。
- 积分实际同步执行累计0.577763141 s。
- 输出I/O累计0.054816873 s，包含t0 gate；后续工程gate累计0.039727369 s。
- cold lower/compile计时包含一次阶段记录写入。因此上述类别有包含关系，不能直接相加或与MOM含checkpoint的Main loop比值解释速度。
- 聚合采样峰值RSS835219456 bytes，4 GiB限内；guard输出根峰值105044529 bytes。本次运行前根目录101946608 bytes，新增约3.10 MB，低于16 MiB增量限；合计128 MiB限内。资源是采样上限证据，可能在两次采样间短暂超出。

## 身份与可复核限制

完整小型scalar summary见 `ocean_coarse_diagnostic_scalar_20261001.json`，含冻结合同、输入、源模块、隔离计时入口、guard、receipt和snapshot index的SHA256。公开CSV只含33行标量，没有原始数组、私有路径、凭据或原用户目录。

输入SHA256 `43ffb2c82f30638ac0b79b9199459c6fc1bb0922a76fee84b5fad7df8a0160f7`；生产源模块SHA256 `0ee5874cef57a318e534edb9b99690c797a4ecb1be95c4f0283f957e32ff4169`；33时刻CSV SHA256 `a2e8a8e25310daa08ddfb91c4910833679f18413d1022ed32ca2e2171785022d`。

ocean点采样与MOM cell mean、collocated与C-grid、静态nodal与moving层数值含义不同。MOM既有原FAIL仍是FAIL，FILTER=0仅是单变量diagnostic。此结果可供独立审查各自离散行为，不能直接推出工业数值模式相同或更好质量同时显著更快。
