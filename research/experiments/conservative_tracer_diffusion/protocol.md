# 默认标量水平扩散守恒：预注册协议

日期：2026-09-28；起点：`e7de24e`；执行前提交协议。

## 调研结论

- [MOM6 预算文档](https://mom6-analysiscookbook.readthedocs.io/en/latest/notebooks/Closing_tracer_budgets.html)以层厚乘浓度的广延量闭合各过程。
- [MOM6 扩散文档](https://mom6.readthedocs.io/en/main/api/generated/pages/Horizontal_Diffusion.html)通过相邻水柱间面通量更新示踪剂；其等中性扩散不能直接等同本项目背景水平 Laplacian。
- [MITgcm 离散算法](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html)说明边界几何、有限体积和自由面/示踪剂离散兼容性。
- [NEMO 引擎说明](https://nemo-ocean.eu/framework/components/engines/)明确海洋/海冰质量、盐和能量一致性目标。

据此采用局部面交换、不做全球温度/盐度后验扣均值。首先修复默认
标量背景扩散及标量超黏性，不冒充完成所有动量或中性混合的守恒。

## 假设与复现

H-DIFF：无沿岸增强时，当前 `kappa_h*_laplacian_h` 的展开球面度量修正
和 `kappa_bi*_biharmonic_h` 不严格按 `A*dz_node*wet` 望远镜相消。
`_d_dy` 的度量修正还会读到关闭面的陆地/干底节点。

探索性基线已保存于本地
`results/industrial_alignment/default_diffusion_baseline.json`，不用于调阈值。
24×32×4、纬度 ±30°、种子 2718 的随机示踪剂：

| 工况 | 常系数旧算子净/绝对预算 | 原面通量净/绝对预算 | 旧超黏性净/绝对预算 |
| --- | ---: | ---: | ---: |
| 全湿 | 1.516e-9 | 4.714e-18 | 2.375e-9 |
| 陆地与干底阶跃 | -6.898e-6 | 5.273e-18 | -1.970e-6 |

随机场的全局 W 数字不是当前真实气候态的漂移估计；上述比值只是
离散守恒反证。测得原型能守恒也不证明极值、精度或整模式可靠。

## 修改方案

1. 全部标量水平 Laplacian 使用 `D_k(C)=div(k_face*grad_face(C))`。
   相邻湿面使用同一平均系数；经向周期、南北封闭，纬向采用面 cos。
2. 常系数标量 biharmonic 使用 `-kappa_bi*D_1(D_1(C))`。
   D1 对体积加权内积自伴、非正，故其负平方耗散方差且守恒。
3. 背景/沿岸、完整 tendency、线性半步、残差相减、诊断使用一致算子。
   不留“默认非守恒、开沿岸才守恒”的兼容旁路。
4. 动量 `_laplacian_h`/`_biharmonic_h` 本次不改：矢量球面耗散与能量
   需要独立研究，不能以标量预算证明替换其物理。

## 冻结验收与反证条件

- float64、全湿/陆地/干底、低纬/高纬、至少三个随机种子：
  `|sum(V*tendency)| <= 5e-13*sum(V*|tendency|)`。
- 常数场精确零；干节点/关闭边界值的改变不得影响湿域 tendency。
- 面 Laplacian 的 `<C,D(C)>_V` 非正且双线性互易；biharmonic
  的 `<C,-D²(C)>_V` 非正。闭合不允许靠补偿后验均值实现。
- 背景线性扩散在 CFL 安全步下不造新极值；**biharmonic 不承诺单调**。
- 完整扩散半步对 T/S 分别满足离散预算，关闭的过程不引入其他源。
- 光滑制造解在 24/48/96 纬向网格上收敛比至少 3.5（内部区间），
  保留单独墙面/岸线断言，不能用制造解裁剪掩盖墙面守恒。
- 算子方向导数与有限差分相符；完整已有测试、lint、MMS 通过。
- 真实 ETOPO、合成强迫四组至少 1 天通过；之后延长并与冻结基线对照。
  它不是百年或真实气候技巧证明。

若非正性、精度、整步预算或稳定性失败，返回推导/文献修正方案，
不得放宽阈值或重开旧默认路径来制造 PASS。

## 后续，不计为本实验完成

真实强迫长期预算、动量能量耗散、GM/Redi、极盖/滤波账本、自由面与
节点体积统一，以及与成熟模式的完整公平比较，仍属于持续目标。
