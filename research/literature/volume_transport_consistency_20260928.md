# 移动体积与正压/温盐输运时层：下一次物理改动前的依据

查阅日期：2026-09-28。当前核对内核 `02a385b`；本文是实现前分析，
不是新的物理方案已通过。完整验收仍由工业级路线约束。

## 一手来源

1. [MOM6 Barotropic-Baroclinic Coupling](https://mom6.readthedocs.io/en/main/api/generated/pages/Barotropic_Baroclinic_Coupling.html)：
   斜压示踪剂速度不是正压最终端点速度，而包含子步时间平均输运。
   层输运之和需等于推动海面的同一个平均输运；压力/速度末态修正
   不能自动建立这个等式。本文不直接移植其 ALE/PPM Newton 算法，
   因为本项目现在没有那些层厚状态、界面或 PPM 连续求解器。
2. [MITgcm nonlinear free surface](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html)：
   相同的厚度与速度时层必须出现在 h 和 h*C 更新中。守恒取决于
   实际更新，而非事后库存加一个 eta*C。该文还区分压力海面和用于
   连续方程的厚度状态，并将其写入重启；滤波、未收敛压力解与多时层
   都可能使两者不同。线性自由面近似不是天然“不守恒”，但不能把
   其被忽略的移动库存与实际海洋体积混为一谈。
3. [MITgcm vertical grid](https://mitgcm.readthedocs.io/en/latest/algorithm/vert-grid.html)：
   层中心与界面中心表示可选，但必须明确控制体、界面与测点位置；
   节点数量等于 nz 不能推导出 nz 个与真实地形相符的有限体积层。

## 当前代码的可核对事实

- `make_fd_params` 的 dz_node 顶/底取完整相邻节点间距，内部取相邻
  间距的一半和。全湿列有 sum(dz_node)=H_sw+(dz_first+dz_last)/2，
  不等于 H_sw；掩膜后也不自动等于真实地形深度。
- `_barotropic_velocity` 对节点速度做梯形积分再除 H_sw，对应的
  端点权重是相邻间距的一半；温盐 Fz 用的是 dz_node 完整端点权重。
  即使全湿，垂向剪切也会使两种“列输运”不同；不能全归因于海岸。
- `_free_surface_step_fd` 用固定 H_sw 与二维 surface-wet divergence。
  `_vertical_transport_iface` 用逐层三维 both-wet divergence；部分
  深度台阶的面开放与度量也不同。D34 修复的是列投影内部一致性，
  没有把正压连续方程同步改成同一个输运。
- `_step_impl` 先 L/N/L，再执行正压子步，只把最终正压速度差广播
  回三维场；没有把子步平均面输运返回给已经完成的温盐 RK。
- `_explicit_full_step` stage-1 用旧速度，stage-2 可投影为零列散度。
  这不等于两个阶段都使用海面更新所需的正压平均输运。真正有海面
  位移时，任意要求顶面固定坐标输运为零也不是物理闭合条件。
- 极盖在完整步末尾进一步改速度/温盐。MOM6/MITgcm 的输运一致性
  条件意味着必须计入此阶段，不能只验证滤波前的压力投影。

## 研究约束

下一阶段先用全湿剪切、台阶和实际子步构建独立恒等式反证，分开
度量垂向体积、湿面开放、时间平均与极盖的影响。几何/时层设计必须
同时说明热盐水、源项、动量/压力、冰库和重启语义。若只加入一个
eta*C 诊断、清零顶面、回补全局均值或增大扩散就“闭合”，应判定
不是根因修复。生产门槛需要新协议，不在尚未做完整步检查时宣布
Jacobi/600 或任何投影已经解决百年可靠性。
