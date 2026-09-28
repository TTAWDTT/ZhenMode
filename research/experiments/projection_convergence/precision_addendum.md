# float32 完整轨迹超标后的反证分解

日期：2026-09-28；当前数值内核 `02a385b`。不改变原协议的 5e-5
原生面积 L2 门槛。冻结右端全部通过不等于完整步通过；实际基线
一日最坏比值 5.4293e-5 已超标，应明确判 FAIL。

## 依据

- [JAX CG](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html)：
  停止范数针对线性矩阵残差，不能替代实际速度修正的原生输运残差。
- [JAX 数值精度](https://docs.jax.dev/en/latest/faq.html#why-is-my-jit-compiled-function-returning-a-different-result-than-my-op-by-op-function)：
  低精度计算、融合与消去会影响结果；必须测量实际计算，不能把
  float32 账本缺口自动解释为“可忽略舍入”。
- [MITgcm 非线性自由面](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html)：
  未收敛解、滤波与多时层都可能使连续厚度不一致，完整物理闭合另验。

## 注册的只读实验

1. 同 2°/真实地形/合成强迫、基线和动态冰、Jacobi/600、float32
   再演化一日，在实际 stage-2 逐步捕获 before/after 速度。计数所有
   原生残差 >5e-5 的步骤，保存最坏右端、步骤编号和 SHA256。
2. 对同一个最坏 float32 预测速度，分别比较：原生 float32 解；
   参数/速度转 float64 后的高精度解；高精度解最终量化回 float32。
   对每个输出分别用原生 float32 和参考 float64 算术计算同一个
   湿面列散度，均按固定初始 float32 预测输运面积范数归一化。
3. 保存每种真实残差、float32 与参考输运的差别、dtype 和源哈希。
   若高精度求解后最终量化仍超门槛，则单独增加 cap/收紧 CG 不足；
   若原生输运运算本身是主因，不能只把诊断改为 float64，因实际
   温盐的 Fz 也使用该运算。任何改变数值实现须另注册后验证。
4. 实验只读捕获，不改强迫、温盐、物理算子、门槛或源表。投影
   精度解释不代替移动体积/时层修复、百年、气候及整模式梯度验收。
