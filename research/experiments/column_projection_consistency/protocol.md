# 列投影必须约束真正的温盐顶面输运

日期：2026-09-28；起点 `ab56075`。完整工业级目标与路线保持不变。
先验证/修复实际湿面投影，再继续移动体积和正压时间平均输运耦合。
本协议只承诺验证所述根因，不把局部投影改正确冒充整模式守恒。

## 研究依据与待反证假设

- [MOM6 正压/斜压耦合](https://mom6.readthedocs.io/en/main/api/generated/pages/Barotropic_Baroclinic_Coupling.html)：温盐层输运之和须与自由面连续方程的时间平均输运匹配。
- [MITgcm 非线性自由面](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html)：体积和 h*C 更新必须使用匹配的厚度、速度时层，诊断库存补项不能代替实现。
- [MITgcm 垂向网格](https://mitgcm.readthedocs.io/en/latest/algorithm/vert-grid.html)：节点、层中心、界面及其体积定义不能混用。
- [JAX CG](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html)：输入算子须 Hermitian positive definite；导数使用隐式线性解，也依赖解收敛。当前投影有常量/网格空模，只对相容右端在非零子空间讨论正定性。

代码检查发现 `_column_divergence` 对每层调用二维 surface-wet divergence，
即使先把干底速度乘零，wet/dry 内部面仍错误开放。实际温盐/w 诊断使用
三维 both-wet face gates。投影矩阵还用二维梯度向下广播，与真正列散度
的湿体积加权负伴随未必相符。

H-GATE：真实顶面 `Fz[...,0]` 与投影的列散度在干底台阶不相等，导致
解完错误 Poisson 问题仍有实际顶面输运。全湿是控制，不能只测全湿。
H-ADJOINT：改用真实三维湿面列散度后，压力修正也须使用其一致的
三维湿面梯度；只替换散度、保留二维梯度的修正不是足够的 CG 修复。

## 预注册门槛

1. ±65°、三维干底台阶/陆地与全湿，三个随机种子：列散度必须逐格
   等于实际底向上连续方程的顶面；相对最大实际列绝对输运 <= 5e-13。
2. 独立面积/节点体积内积：B 与 G3 满足负伴随，Poisson -A*B*G3
   对称且半正定；误差 <= 5e-13 相对绝对内积尺度。
3. 小网格、float64、最多 1000 CG 次，实际顶面 L2 残余相对起点
   <= 1e-9；湿节点动能不增加。干层速度哨兵不改变湿层投影。
4. 零速度/零右端返回有限零解；投影保持常量温盐的局地 tendency 为
   舍入尺度零；JVP 与独立中心有限差分在相容问题上吻合。
5. 单测先复现失败，再最小范围修复；不清零真实顶面、不改温盐、不
   扣全局平均、不改外部源表、不把改投影当成移动体积的实现。
6. 全量测试、ruff、MMS。真实 ETOPO、合成强迫、2°、float64、
   dt600/dt_bt50、kappa_bi2e14：基线及动态冰各一日，记录实际阶段预算
   与顶面输运、默认 150 次 CG 的实际收敛表现及源码哈希。
   同时记录失败与旧 ab56075 的差别，不预设预算缺口一定消失。

## 仍必须解决的完整目标

该投影仅约束显式 stage-2 速度，不能自动解决 stage-1、正压平均输运、
节点几何/浅海部分单元、完整 h*C/海冰盐水库存。对流系数缩放嫌疑须
单独反证。真实强迫百年、独立气候/预报、GPU/分布式、全模式伴随与
生产持久预算/重启仍按完整路线验收。
