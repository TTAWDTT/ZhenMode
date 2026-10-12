# 点值示踪量与水平柱矩一致性（开发选项）

前轮最优配置主网格分数0.0524039393437639，MOM为0.036699334480504484，不能称SOTA。旧所有内部波、热风、1D数据属于开发。旧新增dynamic-h逐位相同门槛依然FAIL/OPEN；它不能阻止另一个明确改变物理算子的假设，但不能被新声明覆盖或重标PASS。

季度相位RHS定位显示16/32/64节点温度误差的柱平均下界分别占L2的86.7%/93.0%/95.3%。在保持原h容量的前提下，仅修改垂向通量不能去掉这个柱误差。新的material_point_v4将水平整柱示踪量矩与垂向物理点积分共同修正。没有拟合参数，没有改网格、容量、dt、EOS或评分阈值。

I使用四点三次区间积分的逆柱原函数，内部点和床面具有三次精确性；顶行明确覆为原h容量，保留真实表面质量通量，不宣称顶行三次精确。对于水平不变的仿射q(d)，柱面通量为q_top ΣFm+Σh q_d I(Fm/h)。相对原水平柱通量之差按|Fm|分配，零质量面不获得修正。一般曲线/水平变化仍有离散误差。

点值目标使用同一实际质量面与Fz，完整保留−T Rmass/h。垂向anti-flux在零质量面、表面及床面为零，物理端点在连接允许时保留目标；每个连接区间的兼容残差按容量最小范数分配。这不保留v3的相邻节点pins，不保证各子区间点目标全部闭合。所有水平和垂向anti-flux使用一个全域alpha，共享面不使用不同柱系数。alpha约束相对原band-v3基准的Euler包络和表面特征增量，盐度可附绝对零下界；空可行区间显式false并退回原基准，不能宣称它修复已经违规的基准。温度不附0°C下界。

本选项仅限全湿均匀物理表底、至少六点、unsplit symmetric闭边界、四阶压力连续性、centered_second示踪量、nodal_capacity柱均值和integrated_reference_v1算术。没有完整三维能量adjoint、完整RK2正性、TVD或全球气候资格。原默认数值路径不变。

原生pilot固定material_point_v4+cubic_uniform_v1 pressure、旧nodal_capacity均值；压力初始加速度诊断支持这个组合，失败的cubic_consistent_v1均值不参与。首先精确回放旧band/梯形主算例，再运行新主pilot：原科学屏障全部通过、温度误差不高于旧最佳8.530755561576719e-7C且归一分数严格下降，才扩余批。新物理算子可能改变eta和动态h，原静态dz_node/网格/合同不变且输出h必须按原公式从dz_node与eta回读。旧额外dynamic-h算术等价gate独立保持FAIL。每run1CPU/600秒/4GiB采样RSS；pilot前ETA未知，后续依据本机实测规划。

Implementation refinement before native: failed affine control retained in controls-attempt03.log and affine-debug01. Endpoint recomputation created a roundoff-only outward ~1e-23 increment, zeroing global alpha. v4 now structurally preserves accepted baseline endpoint rates by antiF1=-Danti0 and antiF_N-1=Dantilast, when adjacentmassfaces connected. Interior point target unchanged; stagnant barrier priority unchanged. All original controls and thresholds unchanged.
