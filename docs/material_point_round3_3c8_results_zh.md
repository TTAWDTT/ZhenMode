# 点值示踪量修复及限幅区间检查（3c8）

本阶段原预声明检查已完成；完整物理与全球SOTA资格仍OPEN。本报告绑定生产源码 3c8e86bef2ee1fbfd50cca4343d6fc61694d9edc，配置 material_point_v4 / integrated_reference_v1 / 三次压力 / 原容量柱均值。主网格开发对照分数为0.005388509039568577，温度体积及时间加权绝对RMS为7.334242326890638e-9°C；前轮v3分别为0.0524039393437639与8.530755561576719e-7°C。这里参照线性解析内部波，而原生模型积分完整非线性方程。ZhenMode观测物理节点，MOM6/Oceananigans采用各自原生层平均/速度位置；eta/u/v按原振幅或速度尺度归一。旧波已参与开发，三方是有限111配置选择后的冻结对照，MOM6历史完整构建及动态库身份仍有缺口。因此仅能报告指定观测与固定门槛下的开发排名，不能宣称全球SOTA或各baseline全局最优。

## 最终结果

六组内部波原生对照全部通过原误差、库存和稳定screen，且分数均低于两份冻结baseline。分数为五项误差除以各自原预声明门槛后的最大值，越小越好；同组比较。primary为8×32×16、horizontal-coarse为8×16×16、vertical-fine为8×32×32；main/half为0.01/0.005m幅值。

| 网格/幅值 | 新3c8 | 前轮v3 | 冻结MOM6 | 冻结Oceananigans |
|---|---:|---:|---:|---:|
| primary/main | 0.005388509040 | 0.05240393934 | 0.03669933448 | 0.1228595011 |
| primary/half | 0.002701289300 | 0.05219678373 | 0.03639019552 | 0.1228830919 |
| horizontal-coarse/main | 0.005109790701 | 0.05207124196 | 0.08869339763 | 0.1365438423 |
| horizontal-coarse/half | 0.002597085100 | 0.05188571543 | 0.08856800027 | 0.1365657872 |
| vertical-fine/main | 0.005413792384 | 0.01347960395 | 0.03017267819 | 0.04403194603 |
| vertical-fine/half | 0.002707164476 | 0.01263850656 | 0.02979265288 | 0.04403889090 |

十条原生均已完成且通过原科学screen。两个零扰动的eta/u/v/T/S/h漂移均0。热风温度RMS3.279198835499751e-5°C、盐度RMS6.815990080079416e-6psu，T/S相对库存误差3.1252778143198157e-12/2.220446049250313e-16；盐度较前轮略增但原门槛仍通过，不称所有指标都下降。主旧默认回放17数组精确一致。独立十条已保存数据读回核对通过：完整133源身份、参数/输入输出哈希/时间轴、原误差与库存、实际容量和动态h原公式均符合原合同。十条五个完整场T/S/eta/u/v对冻结362数值完全相同（maxabs=0，原物理单位），这是描述性差分，不是另设等价选择门槛。热风T库存独立重算为3.125197529148943e-12，发布原值3.1252778143198157e-12保留，微小算术差在原核对合同内；不覆盖原报告值。

## 本次变化与依据

水平柱示踪量乘积矩与三次垂向点原函数共同构造目标RHS；零质量面不分配水平修正，物理边界和停滞面固定，连接区间用实际容量分配兼容残差。一个全域alpha同时缩放水平与垂向守恒面修正。完整−T Rmass/h保持，非零常量不被人为清零。支持范围是全湿均匀物理表底、闭通道和受限离散组合；没有任意地形、全三维能量、完整RK2正性或气候积分资格。

容量修补47的实际顶容量与直接质量面整柱和已经接入。独立温标控制却发现旧band基准逐柱系数因床面约1e-18的Euler差从1变0，完整RHS C→K差2.8097102655739095e-8°C/s超过1e-13°C/s。新积分路径用局部浓度差构造端点/邻层通量增量，保留真实质量残差与旧默认API。原函数应用改为同一四点区间积分加reverse-cumsum，O(N)替换密集矩阵应用；数学多项式控制与实际容量检查独立核对，浮点求和顺序改变，不主张逐位相同。

362的新独立二维非仿射控制三档误差及加密比通过，却三档feasible=false、alpha=0，故其科学资格为OPEN，误差只证明回退基准。已接触数据的定位记录显示原区间lo=2.386066257516362e-14、hi=-1.1093846257164162e-14，而原基准实际包络违规数量为0、最小Euler值11.77372。3c8只在基准增量余量坐标中求解同一包络：余量的literal0对应原包络明确包含基准。没有epsilon padding或门槛放宽；绝对下界仍作交集，负基准/空区间仍可失败。编译融合是原因推断，未形式验证LLVM。

## 验证范围与独立性

141项本地受影响控制及全库ruff通过。完整功能CI38286199660实际1721项pytest通过、6710warnings、1081.12秒，MMS ALL PASS。独立核对完整133文件的Git、wheel、build-input、installed、计划与加载路径前后相同；实际顶容量/直接质量和在16、32节点精确相同，三次多项式控制最大相对差≤4.48e-16。已失败47温标输入作为DEVELOPMENT回归，新源最大差≤3.67e-18°C/s，小于原1e-13°C/s；温度常量RHS最大≤3.92e-19°C/s，小于原1e-17°C/s。六个编译区间不变量覆盖正下界alpha、零/非零修正空floor及半alpha上界。

新的独立1D在freeze后预声明35/69/137节点及新剖面，单次有效全验证通过：归一加权L1为1.625210824e-4/1.178462770e-4/7.167341328e-5，加密比1.37909/1.64421。常量RHS和最终漂移、端点漂移均0；最大预算残差5.833e-16低于原逐网格预算门槛（约5.258e-12），反质量通量负例拒绝。每条轨迹最小alpha为0；这是共享生产primitive/projection/blend与JAX的补偿核心检查，不能证明全程修正生效或实际二维水平owner/完整模型资格。

新的二维正控制在源码freeze后独立选择不同流函数/剖面/网格，并在数值比较前声明单调垂向结构及每网格alpha>0.5，以验证修正分支实际生效。它检验生产二维水平owner瞬时RHS；仍共享PDE、factory几何与JAX，不能替代时间积分或压力资格。这项新正控制单次完整验证通过：26×27/52×53/104×105网格alpha均1、feasible均true，以解析RHS的容量加权L1归一后，瞬时RHS L1误差为0.004820824590/0.002342416778/0.001161087610，加密比2.05806/2.01743；温度常量RHS最大≤8.95e-22°C/s，平均柱库存Euler残差≤6.55e-15°C·m小于原逐网格约1.025e-9°C·m门槛，反质量负例与零质量面/闭边界精确检查通过。声明与数据没有在观察后调整；这是特意设计的active正控制，不能代替所有场的资格。

## 保留的失败与可恢复记录

454原十条、独立1D33/65/129、完整CI1715PASS均保持原身份；其理想顶积分行与实际容量11/50ULP差已另源修复。47温标失败与0..7完成、8准备中断、9未执行保持；细网格积分147.4586秒的回退记录保持，362为9.39248秒只说明同设备跨源恢复，没有同期三方速度比较。362二维false可行性失败与新1D29/57/113、1720pytest/MMS及十条数据一致性PASS各自保留，后者不能覆盖前者。

前轮原1D21/41/81常量控制失败保持OPEN；mean1e主分数0.05492232392019455变差，2..9未执行，完整CI原5FAIL/1687PASS身份registry缺模块另补记录；arithb8额外动态h逐位门槛FAIL4.44e-16m及错误扩批中止记录不重标PASS。最初454 h检查因空间轴未reshape产生false，按原交换格式reshape另留通过记录；没有在failed exit下扩该批。所有阈值、旧原21/111/32及冻结MOM/OA结果均不覆盖。

输出根目录 D:/Github/ocean-solver/outputs/vertical-mean-round3-20261012/，本源子目录material-point-v4-marginfix。声明、测时pilot、单次完整控制、逐行追加journal及采样RSS记录均保存。每原生run为1CPU/600秒/4GiB aggregate RSS采样（非硬cgroup）；控制1CPU/180秒/4GiB采样。JAX GPU allocator fraction .40不是全设备内存硬上限。原生测时只用于有限后续计算规划。用户授权持续按证据研究，单批完成不自行暂停；[草稿PR69](https://github.com/TTAWDTT/ZhenMode/pull/69)不合并。


[原生声明](data/material_point_round3_3c8_native_declaration.json)、[十条终态](data/material_point_round3_3c8_native10.json)、[六组三方分数](data/material_point_round3_3c8_comparison.json)、[独立原生读回](data/material_point_round3_3c8_native_readback_result.json)、[完整CI终态](data/material_point_round3_3c8_ci_terminal.json)、[CI独立核对](data/material_point_round3_3c8_ci_readback.json)、[同生产提交完整CI](https://github.com/TTAWDTT/ZhenMode/actions/runs/38286199660)。

[独立源码数学声明](data/material_point_round3_3c8_source_declaration.json)、[源码数学结果](data/material_point_round3_3c8_source_math.json)、[新1D声明](data/material_point_round3_3c8_1d_declaration.json)、[新1D结果](data/material_point_round3_3c8_1d_result.json)、[新2D声明](data/material_point_round3_3c8_2d_declaration.json)、[新2D结果](data/material_point_round3_3c8_2d_result.json)、[旧362二维真实失败](data/material_point_round3_3c8_previous362_2d_failure.json)、[已接触数据余量诊断](data/material_point_round3_3c8_margin_diagnostic.json)。

本源十条总驱动墙钟342.90790797秒、最大采样RSS2267553792字节。细网格真实integration_s为9.391196672999996秒；驱动墙钟34.668503894秒，不混用指标。原projection242.68秒来自两尺寸实测；后续个别行更慢的真实时间原样保留，无速度排名。下一有限证据阶段先核剩余动量误差中的垂向速度位置与非线性参照差异，不复扫参数或删物理项追分数。
