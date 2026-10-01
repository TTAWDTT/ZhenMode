# FD静态桥：两种库存合同（实施前声明）

只读单列/全湿nodal_dual_v1；显式输入depth/h0/eta/FD[T,S,u,v]、Tref/Sref。
核验h0为节点dual厚度，接口[eta,-节点中点,...,-depth_last]。
拒绝其他几何、非正顶体、eta>0导致节点插值外推、非法温盐或目标域不同。

合同A：水=sum(h0)+eta，热盐=sum((h0+eta*top)*FD[T,S])。
原momentum线性质量采用参考h0，保留rho0*sum(h0*FD[u,v])；不冒称
material动量与reference动量相同。候选N的T/S取h*点值，u/v取h0*点值。
在旧dual控制体上视为均值，界限保持P1物理交叠remap到正厚目标带。
选择A，因为不向真实失败前状态凭空补库存；无法逆回原FD点值。

合同B：同域FD节点分段线性插值的几何积分。eta<=0且底覆盖时不外推。
B与A通常不同，分别报告，绝不补差额使两者同时“守恒”。新桥压力
来自候选均值的共同有界P1；原压力参照是原节点EOS梯形积分加rho0*g*eta，
在共同深度线性插值压力。报告差值不保证两者相同。

解析曲率节点9/17/33/65点，冻结二次T=20+.1z+.001z²和三次
T=20+.1z+.00001z³，域[-40,0]，S35。mass-lumped与几何积分差及压力
误差预期约二阶；不是有限曲率精确零门槛。非均匀节点、2.5m原顶体
eta=-2.49正厚、温盐边界限幅、库存守恒/拒绝独立测试。
CLI消费调用方构造的规范npz，不自动读取历史checkpoint/forcing、不运行
时间步。历史source与输入hash仍由nagi调用者验证；本云端无私有原数组。
qualification=false，FD信息损失/真实湿床/动力耦合继续阻断生产接入。

## 调用与限制
`python research/experiments/fd_static_bridge/bridge.py canonical_column.npz summary.json`
输入严格keys：depth(m正向下，首0)、h0(m)、eta(m标量)、values(N×4，顺序T/S/u/v)、target(下降物理z接口)、Tref/Sref(标量)、source_sha(40hex历史source声明)。输出仅无数组库存/压力差标量与输入SHA；文件已存在则拒绝覆盖。
源声明不是核验：输出historical_source_independently_verified=false，调用方仍需冻结源码/输入审计。规范单列需由nagi已核验原件抽取，不能假称本云端已读取私有状态。
原压力对照为EOS梯形节点压力的深度插值；未调用生产PGF或证明实际湿面力等价。新桥重构为算法级同P1，不是PR9代码调用复用，不证明温盐动力耦合。
同分辨率桥的维数未必降低，但没有声明逆算；减少目标层数或limiter非线性会丢失点值信息。桥只保留指定4库存，不能据此保留剖面、压力或动能；压力差完整报告而不补残差。正eta缺表面节点覆盖拒绝，部分湿底列和legacy非dual几何拒绝。
合同冻结是仓库协议声明，不作为外部可验证预注册证明。
