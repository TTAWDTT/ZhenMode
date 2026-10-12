# 点值示踪量与水平柱矩一致性（开发选项）

前轮最优配置主网格分数0.0524039393437639，MOM为0.036699334480504484，不能称SOTA。旧所有内部波、热风、1D数据属于开发。旧新增dynamic-h逐位相同门槛依然FAIL/OPEN；它不能阻止另一个明确改变物理算子的假设，但不能被新声明覆盖或重标PASS。

季度相位RHS定位显示16/32/64节点温度误差的柱平均下界分别占L2的86.7%/93.0%/95.3%。在保持原h容量的前提下，仅修改垂向通量不能去掉这个柱误差。新的material_point_v4将水平整柱示踪量矩与垂向物理点积分共同修正。没有拟合参数，没有改网格、容量、dt、EOS或评分阈值。

I使用四点三次区间积分的逆柱原函数，内部点和床面具有三次精确性；顶行明确覆为原h容量，保留真实表面质量通量，不宣称顶行三次精确。对于水平不变的仿射q(d)，柱面通量为q_top ΣFm+Σh q_d I(Fm/h)。相对原水平柱通量之差按|Fm|分配，零质量面不获得修正。一般曲线/水平变化仍有离散误差。

点值目标使用同一实际质量面与Fz，完整保留−T Rmass/h。垂向anti-flux在零质量面、表面及床面为零，物理端点在连接允许时保留目标；每个连接区间的兼容残差按容量最小范数分配。这不保留v3的相邻节点pins，不保证各子区间点目标全部闭合。所有水平和垂向anti-flux使用一个全域alpha，共享面不使用不同柱系数。alpha约束相对原band-v3基准的Euler包络和表面特征增量，盐度可附绝对零下界；空可行区间显式false并退回原基准，不能宣称它修复已经违规的基准。温度不附0°C下界。

本选项仅限全湿均匀物理表底、至少六点、unsplit symmetric闭边界、四阶压力连续性、centered_second示踪量、nodal_capacity柱均值和integrated_reference_v1算术。没有完整三维能量adjoint、完整RK2正性、TVD或全球气候资格。原默认数值路径不变。

原生pilot固定material_point_v4+cubic_uniform_v1 pressure、旧nodal_capacity均值；压力初始加速度诊断支持这个组合，失败的cubic_consistent_v1均值不参与。首先精确回放旧band/梯形主算例，再运行新主pilot：原科学屏障全部通过、温度误差不高于旧最佳8.530755561576719e-7C且归一分数严格下降，才扩余批。新物理算子可能改变eta和动态h，原静态dz_node/网格/合同不变且输出h必须按原公式从dz_node与eta回读。旧额外dynamic-h算术等价gate独立保持FAIL。每run1CPU/600秒/4GiB采样RSS；pilot前ETA未知，后续依据本机实测规划。

Implementation refinement before native: failed affine control retained in controls-attempt03.log and affine-debug01. Endpoint recomputation created a roundoff-only outward ~1e-23 increment, zeroing global alpha. v4 now structurally preserves accepted baseline endpoint rates by antiF1=-Danti0 and antiF_N-1=Dantilast, when adjacentmassfaces connected. Interior point target unchanged; stagnant barrier priority unchanged. All original controls and thresholds unchanged.

实际容量修补：独立454源码审查发现ideal uniform primitive顶行与factory真实np.linspace中点容量相差11ULP/50ULP（最大约1.11e-14m）。454原科学结果/1715完整pytest与MMS/独立10native与新1D均保留，不改源码身份。新的修补只将factoryprimitive顶行覆为实际dz_node，同时horizontal point integral顶项直接使用sum(actual mass faces)，避免mass/h再乘h的舍入；内部primitive与point/projection/blend算法不变。新局部exactcapacity/actualmass controls、完整新CI、独立新包identity以及同10合同的新源码numericalequivalence另记录，不使用旧dynamic-h extra exact门槛假装PASS。

增量修补：47新独立非恒定温标控制失败2.80971e-8>原1e-13，尽管同源码1716pytest/MMS全部通过。诊断定位未改的band基准逐柱限幅在摄氏/开尔文由1变0，触发点为床面近零修正；外层全域alpha均1、插值repin不改善。新integrated_reference路径传入真实upwind面浓度，端点及邻层修正使用质量通量乘局部浓度差，避免相等乘积相减的舍入；基准Euler倾向也从完整积分分子装配，保留显式reference mass residual。旧legacy路径/API默认不改变。初137控制含已接触失败输入的两个模块回归通过，不将它再叫保留盲检数据。

计算改写：47细网格实际积分147.4586秒，454为9.5262秒，原pilot时间投影已撤销。新点原函数应用相同四点局部积分权重并反向累积，工作量随节点线性增长；实际顶行容量和predivision质量和/物理Fz0仍保留，内部与床面多项式门槛1e-14不变。140受影响控制通过；未来新本机pilot再决定时长，不先声称速度改善。新功能提交与完整CI/隔离native/source审查另绑定，47温标FAIL及0..7完成/8准备中断/9未执行始终保留。

新增量/前缀计算方案的独立静态审查接受上述代数范围；真实rawflux=F*upwind时增量恒等式成立，generic调用不使用这个假设。这个阶段改变了核心源码，旧1D未改源码的桥不再成立；必须用新源身份独立核对/新的未使用1D与实际2D数据验证。原输出或旧CI都不能替代新功能的完整CI。

区间余量修补：362源的1720pytest/MMS、十条原生和新1D通过，但未接触的实际二维RHS三档全部可行性false/alpha0，原声明allEulerintervalsfeasible门槛OPEN。误差/refinement通过来自fallback基准，不能当active-v4证书。仅对已经接触的失败24x23做定位，原基准全部满足原bounds且T_Euler>11.77，原求解alpha区间lo2.386e-14>hi−1.109e-14；数学等价的余量表达式结构性包含原先明确允许的零修正，得到lo0/hi−0，feasibletrue/alpha0，不添加epsilon或改变真实包络。新jointScalarBlend按LM=min(currentmin−q−low,allowance−low,0)、UM=max(currentmax−q−low,allowance−low,0)解相同约束，绝对floor交集、positive-lower-alpha与空区间拒绝仍保留。141本地控制和已接触场开发回归通过，独立实数代数审查接受；原3622DOPEN和所有旧失败保留，须新源码/真正CI/新的未用active非affine二维profile验证。


## 动量点速度可选分支（主试点排名未改善）

新增显式momentum_vertical_velocity_scheme，默认dual_face_v1保留旧−Fz上侧控制体面取样；point_primitive_v1用同一O(N)三次原函数得到物理节点的向上z速度，表面−Fz0/床面0。示踪量与连续性仍使用原Fz face owner。该选项独立于vertical_tracer_scheme，不自动改变旧v4对照；目前限定float64、原容量、fullywet uniform physicalendpoint≥6、integrated_reference、closed fourth-order unsplit legacy且无projectAdv通道。新选项在development合同非默认时显式序列化，旧默认合同digest保持。

依据是制造流场只读诊断16/32/64：原face取样误差约随节点加密减半，节点原函数速度误差约1.00356e-11m/s且物理床面严格0；同momentum owner/dealias仍有水平滤波误差约1.53849e-11m/s²。这证明局部位置一致性问题，不证明旧主eta误差成因、三维能量或完整时间资格。生产3c8有限已验收记录与旧所有失败均独立保留；ece37c7已完成冻结、1737pytest/MMS软件CI及独立瞬时控制，但主原生综合分数严格改善门槛FAIL，余批2..9未执行；当前最佳仍3c8，详见[momentum报告](momentum_point_ece_results_zh.md)。

新增参数字段改变参数fingerprint；旧默认数值回放检查不等于旧checkpoint参数元数据兼容。显式point选项也改变factory同owner的w诊断，原Fz/tracer闭合未变。

严格zero控制第一次对名义u=.01均匀流要求w逐位0而失败，日志保留。独立诊断确认实际Nm最大3.20731e-21 m/s、Fz/旧w/新w最大3.33992e-20m/s，所以该输入不满足实际离散散度为0的前提。没有改物理代码或阈值；真zero操作数u=v=0保持原逐位0检查，另加非零均匀流实际表面残差−Fz0与床面0的精确归属控制。名义连续场常量流不承诺离散w逐位零，完整质量残差不剪除。
