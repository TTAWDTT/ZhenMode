# 标准 benchmark 的温盐与密度

当前生产默认仍使用[线性状态方程](../src/zhenmode/model/solver/physics/eos.py)。[TEOS-10 组件](../src/zhenmode/model/solver/physics/teos10.py)已接入显式可选的 `PhysicsConfig(thermodynamics='teos10_reference')` FD 路径：CT/SR状态、明确的固定海压力、非线性密度、共同压力对流、中性梯度、正确表面SST及CP0库存。当前接线限无冰组件，普通生产CLI和默认预设未切换；完整原生case、海冰、混合与气候资格仍待取得。

## 三种量不能混用

| 输入 | 新组件要求 | 当前资料或生产语义 |
| --- | --- | --- |
| 盐度 | Absolute Salinity，SA，g/kg | WOA 为 PSS-78 实用盐度 SP。SR=(35.16504/35)SP 是参考盐度；SR 不包含地理盐度异常，不能仅改名成精确 SA |
| 温度 | Conservative Temperature，CT，ITS-90 °C | WOA 温度是原位温度；当前生产 T 未实施 TEOS 转换。位温、原位温度和 CT 不能直接互换 |
| 压力 | sea pressure，dbar，扣除大气压 | 固定 MOM6 密度接口接受 Pa 并乘 10⁻⁴；直接将 Pa 传给新组件是单位错误 |

本项目候选 MOM6 commit `d3a3b4c92dcff5afb8ba2e41426e6251f7abca42` 的 [MOM_EOS_TEOS10](https://github.com/NOAA-GFDL/MOM6/blob/d3a3b4c92dcff5afb8ba2e41426e6251f7abca42/src/equation_of_state/MOM_EOS_TEOS10.F90)使用 SA/CT/p 密度接口；[convert_temp_salt_for_TEOS10](https://github.com/NOAA-GFDL/MOM6/blob/d3a3b4c92dcff5afb8ba2e41426e6251f7abca42/src/equation_of_state/MOM_EOS.F90)实际按固定因子换算盐度，再将位温转 CT。因此其盐度初始化采用 SR≈SA 近似，而非地理 SA 转换。当前候选配置仍选 WRIGHT，不能据源码能力宣称已启用 TEOS-10。

精确地理盐度转换参见 [GSW SA_from_SP](https://www.teos-10.org/pubs/gsw/html/gsw_SA_from_SP.html)；原位温度转 CT 参见 [GSW CT_from_t](https://www.teos-10.org/pubs/gsw/html/gsw_CT_from_t.html)。后续原生初始化须冻结转换、压力和海陆处理，使两边近似明确可核查。

## 组件及独立验证

密度按 Roquet 的 75 项比容多项式取倒数。代码把数学系数按压力、温度和变换盐度的幂组织，用通用 Horner 求值；其数学系数来自固定 GSW-Fortran `29e64d652786e1d076a05128c920f394202bfe10` 的[系数表](https://github.com/TEOS-10/GSW-Fortran/blob/29e64d652786e1d076a05128c920f394202bfe10/modules/gsw_mod_specvol_coefficients.f90)。JAX 自动微分产生偏导，独立参考为该版本未经改写的 Fortran 解析导数。

复算命令：

```sh
python scripts/check_teos10.py --source PINNED_GSW_ROOT --mom-source PINNED_MOM_EOS_TEOS10.F90 --output NEW_OUTPUT
```

在 Linux 外层施加单 CPU、180 秒、4 GiB 限制；不直接以无资源限制的 shell 执行数值命令。检查器核对全部八个实际 GSW 文件及 MOM 文件的 SHA256，字节原样复制到新工作目录；编译原始 GSW 密度、导数及盐度例程，并执行提取的实际 MOM6 密度函数。空 receiver 类型只提供未使用的 `this` 参数，此检查不运行完整 MOM EOS 模块或海洋积分。编译失败、中断及源码不符保留失败收据，已有输出目录拒绝覆盖。

制造检查覆盖 228 个状态，包含六个[官方 density 示例](https://www.teos-10.org/pubs/gsw/html/gsw_rho.html)。数值范围 SA=0…42、CT=−2…40、p=0…8000；Host 预检另允许 CT 至 −3，拒绝非有限、masked、空数据、错误形状和超范围值。该矩形范围是数值检查范围，**不是海洋学 funnel 的证明**。函数本身为无裁剪 JAX kernel，调用者须在 tracing 前执行 `validate_state`；预检不能仅凭数值辨认 SP 被误标为 SA。

2026-10-06 独立安装、仓库外复算：

| 精度 | 密度 RMSE，kg/m³ | 最大密度绝对误差，kg/m³ | 验证容差 |
| --- | ---: | ---: | ---: |
| fp64 | 1.52×10⁻¹³ | 4.55×10⁻¹³ | 2×10⁻¹⁰ |
| fp32 | 6.26×10⁻⁵ | 1.68×10⁻⁴ | 5×10⁻⁴ |

三项密度导数及 Pa/dbar 换算通过。GSW 压力导数为每 Pa，本组件为每 dbar，参考值乘 10⁴后比较。fp32 导数最大绝对误差约 3.01×10⁻⁶，容差 10⁻⁵；fp64 容差 2×10⁻¹⁰。复算同时提取实际 MOM 的换算常量声明，约 6.82 秒、峰值 RSS 214 MiB，属于组件验证成本，不是模式积分速度。

18 个 Fortran 生成参考状态及驱动保留在[参考 JSON](../tests/support/teos10_reference.json)和[驱动](../tests/support/teos10_reference.f90)，日常[测试](../tests/fd/test_teos10.py)核对其身份。完整源码、二进制、228 组输出、编译命令和资源收据放在本地 outputs，避免将编译缓存提交。GSW 参考来源请引用 McDougall and Barker (2011), *Getting started with TEOS-10 and the Gibbs Seawater Oceanographic Toolbox*；[官方工具说明](https://www.teos-10.org/software.htm)列明引用方式。

## 必须整体接线的使用者

| 链路 | 需要完成的工作 |
| --- | --- |
| 初始化与恢复目标 | 把 WOA 原位温度/SP 按冻结近似转成实际状态变量，记录原始及转换后库存；月盐目标使用同一语义 |
| 压力梯度 | `dynamics/pressure.py` 明确 EOS 取样的海压力及参考密度，不用层号冒充压力 |
| 对流 | `physics/vertical.py` 两个水团在同一界面压力比较；分别使用原位压力会混入压缩效应 |
| 涡旋斜率 | `physics/isopycnal.py` 使用当地压力下的温盐密度梯度，排除压力压缩项，再验证离散作用 |
| 工厂诊断与 MLD | `factory.py`、`diagnostics/mixed_layer.py` 同步变量语义与参考压力；不能继续在线性诊断中读取 CT/SA |
| 表面、海冰与输出 | 表面 bulk 使用正确的 SST；热源/焓与 CT 热容量、冻结温度、盐源和观测温度转换同步处理 |

共同压力负例：SA=35.16504、上层 CT=10、下层 CT=11 时，500 dbar 下密度差约 +0.187 kg/m³，表示不稳定；若错误地分别用 0/1000 dbar，则差约 −4.258，方向反转。中性梯度检查也排除了均匀温盐柱的压力压缩项。

尚未验证多项式相对精确 Gibbs EOS 的 funnel 精度、原生初态转换、海冰热力学、完整积分与观测评分。组件结果不授予 B08 完整符合、全球 case 执行或长期气候资格。

## CT/SR 参考变体的实际链路

原生字段完成其声明的映射后，显式调用 `inputs.initial_conditions.convert_teos_reference_fields(t_insitu, SP, pressure_dbar)` 返回 CT 和 SR。函数不猜压力、不补缺失、不读文件；真实原生初始化的来源、映射、ghost处理与库存收据尚待接入。`factory.make_solver_global(..., eos_pressure_dbar=full_grid_pressure)` 要求压力形状与状态相同、单调、0…8000 dbar、表面为零；它是固定参考压力，不随η变化，不冒充动态全压力。

`init_state` 必须收到完整 CT/SR 数组；空初态、Kelvin值、错误压力、无效ghost参考值和旧冰／bulk接口拒绝。压力、对流、涡旋斜率及factory诊断统一使用新变量含义。非线性变化属于显式机制变体，没有藏进默认数值路径。

在线无冰交换先将表面CT转为零压力位温／原位SST，再调用既有bulk。热源按CP0更新CT；分步及端点库存使用相同的已声明潜在焓定义，并以独立边界通量测输入。盐度为SR≈SA，恢复目标也须为SR；通量接口不会替调用者猜测输入是不是SP。表面冻结门槛使用盐度相关的GSW CT-freezing多项式，p=0、溶解空气饱和度=0；它只拒绝需要冰的组件步骤，没有补上冰动力或冰焓方程，也没有证明与MOM的实际冻结配置一致。

MLD可明确选择 `thermodynamics='teos10_reference'`，采用零压力潜在密度；缺失层保留NaN，错误Kelvin/盐度值和Infinity拒绝。快照库存同样须声明该类型，使用CP0并拒绝冲突的热容量。普通CLI尚无此变体的原生数据接入，不能将未转换的旧T/S直接交给它运行。

复算温度组件时，在前述Fortran命令追加 `--temperatures`。它核对22个实际GSW文件，比较原位／位温／CT的五种转换、解析导数和表面冻结CT；分别在fp64/fp32下要求绝对差≤10⁻¹⁰/10⁻⁴，并用改错0.01°C的参考值测试比较器。公开系数是两条路线的最深共享层，因此这验证转换实现／迭代／单位接线的一致性，不能认证共享方程或物理真值。JAX迭代数与阈值在温度参考生成前固定，未从新参考值拟合参数。GSW与Gibbs密度的六个官方样本差约−5.14×10⁻⁴…6.22×10⁻⁵ kg/m³；宽矩形所有样本最大差约0.112，不能把这张矩形表称为funnel有效域证明。

独立温度生成值及驱动在[参考JSON](../tests/support/temperature_reference.json)和[驱动](../tests/support/temperature_reference.f90)。原GSW代码未修改；JAX采用通用多项式求值及固定五次Newton迭代，没有CPU回调或新增GSW运行依赖。BootLoops工具索引及外部引擎表未提供海水热力学求值器；已有线性EOS也不提供转换，因此扩展本项目现有热力学模块，并将来源、失败控制和范围写在这里。
