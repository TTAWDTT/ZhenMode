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

## 阻断修订：有限派生量、动能语义与外部几何绑定

输入有限不等于计算有限。column在所有计算中以浮点overflow/invalid/divide
raise执行，并检查audit输出、旧/新库存、residual/预算、重构积分、压力、
深层库存及全诊断finite；任一不有限明确拒绝并返回调用前深快照。
压力/pair也拒绝非有限计算。JSON先以allow_nan=false序列化再创建输出，
不会因NaN留下成功产物。

全部库存按单位水平面积：热盐m*K/m*psu，动量kg/(m*s)，水m；所有K为J/m²。
original_reference_mass_K =rho0/2 sum(h0*u_original²)；
original_velocity_actual_mass_K =rho0/2 sum(h_actual*u_original²)；
reference_momentum_on_original_actual_mass_K =sum(Nu²/(2*rho0*h_actual))；
candidate_actual_mass_K=sum(Nu_new²/(2*rho0*h_new))。
先前original_actual_mass_K名字错误，实际是第三项，现已更名并补第二项。

新的CLI命令必需`--geometry-report nagi_geometry_binding.json`。
绑定JSON schema_version=1、full_geometry_passed=true、input_sha256(完整canonical
NPZ快照)、reference_weights_sha256(np数组C-order tobytes)、source_sha、terrain_sha。
CLI对这五个字段匹配，记录整个报告bytes的SHA256；只绑定已获外部审查的
全几何断言，不自行执行全几何检查。报告应由nagi既有全几何审查结果生成，
不能因本局部组件成功就生成full_geometry_passed=true。篡改深层weights使
旧报告身份不符而拒绝。裸column函数的accepted_scope明确仅局部顶部，
即使保深层字节也不能证明深层几何正确；输出全几何未在本机重验标记。

旧不带报告的CLI调用不再可用。qualification=false，完整动力资格未新增。
