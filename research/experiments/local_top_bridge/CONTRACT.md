# 隔离局部顶部桥冻结合同
基于PR11 b33f8ea；固定b=-22.5m、原三体接口[eta,-2.5,-10,-22.5]，
要求原top weights[2.5,7.5,12.5]且前三湿，原节点覆盖带下界。
每柱eta<=0且eta>-2.5（原入态正厚），不外推、借深层或clip。
原热盐=h*T/S；参考动量=rho0*h0*u/v；水=sum(h)。目标三个正厚体
比例[1/9,1/3,5/9]，固定同柱底/顶。均值保持物理界限P1只用带内三个体；
共享重构解析交叠积分，pressure亦用其T/S。候选速度Nu/(rho0*h*)明确
不是原参考质量速度。原reference K、原actual-thickness K、新actual K分别报。
库存比较gamma512*(sum(abs(oldN))+sum(abs(newN)))逐字段，不加单位混合floor。
几何分辨gamma8端点规模、库存均值gamma1100尺度；拒绝原库存深拷贝不改。

原EOS/压力按原节点masked梯形。顶部共同深度比，带底DeltaP显式；
下方仅p_join=p_new(b)+原[p(d)-p(b)]，深层原压力增量数组不修改。
单柱深层node/原热盐/参考动量与原压力增量字节一致；邻柱域
[b,min(eta_i,eta_j)]，更高表层仍本柱保留不丢弃。共同下方最多-400m，
若节点覆盖不足不外推。邻柱带底差DeltaPi-DeltaPj/距离影响显式输出。
西干列拒绝，不伪造湿面。压力差只量化，无生产通过阈值；qualificationfalse。
协议声明不是外部预注册证明，无生产步/伪逆/工业资格。

## CLI交付接口
`python research/experiments/local_top_bridge/replay.py canonical_columns.npz summary.json`
NPZ严格keys：
- 每柱数组depth/reference_weights/wet_mask为(C,N)，values为(C,N,4)，
  eta/terrain_depth/discrete_bottom为(C,)。保留完整原节点与原weights，不截底。
- 共享scalar Tref/Sref/alpha/beta/rho0/gravity/source_sha/terrain_sha；EOS需历史
  核验值（beta7.6e-4），SHA分别40/64hex。身份仅声明、输出未独立核验标记。
- distances_m为(C,C)非负有限矩阵；i<j正距离才比较，0表示不请求该对。
  地理距离须调用方核验，不把默认1000m冒称实际距离。没有label依赖；
  输出column索引由调用方映射中心/南/东/北，干列报告拒绝且不生成比较对。

单次不可变bytes加载和hash，只输出无原数组的静态标量。每柱accepted仅指
协议和库存检查，不是压力/物理正确pass。输出库存residual和逐字段gamma界，
水残差、EOS身份、各K、速度放大、带底和扫描压力差、邻柱梯度影响。
拒绝输出只保留原因，函数返回原输入深快照。输出文件已存在则拒绝覆盖。

nagi需独立匹配历史source与输入hash，再构造此规范包；本云端没有真实
四柱原数组。真实中心eta只能作为接口形状合成值，不冒称完成真实回放。
FD点值不可逆、参考/实际动量质量语义、压力代价、底部未采样信息仍未关闭。
