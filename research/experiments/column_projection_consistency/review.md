# 真正湿面列投影：根因已修，生产收敛仍不充分

日期：2026-09-28；预注册协议 `11b5609`，冻结旧内核 `ab56075`。
本轮改变实际 stage-2 投影，不是只读账本；完整体积/时层耦合、百年及
工业级验收未完成。

## 复现与修复

旧列散度对每层调用二维 surface-wet divergence，乘零干底速度并不会
关闭 wet/dry 内部面，因而不等于温盐真正的三维 both-wet face transport。
新约束直接累加 `_divergence_h*dz_node`。

压力修正与 Poisson 同时使用一致的三维湿面梯度 G3，而非二维梯度
向下广播。矩阵取 `-A*B*G3`、右端同号取负，在相容非零子空间为对称
正算子；常量、陆地及网格空模仍存在，不能称无空模严格正定。
旧 `tol=0` 可能在已收敛后继续放大舍入/空模误差；停止阈值改为
`max(1e-12, 32*eps(dtype))`，迭代 cap 不是收敛保证。
不清零真实顶面、不改温盐、外部源表或新增全局均值补偿。

最初 **7 failed、7 passed**；同一批 14 项修复后全部通过。另加
float32 与实际阶段只读监测，现有 16 项直接回归覆盖：顶面等式、
负伴随/对称/耗散、动能非增加、干层哨兵、零右端、常量保持及 JVP
中心有限差分。全量 **330 passed，23 warnings**；ruff、diff-check、
MMS ALL PASS，经向收敛比 4.30。未宣称整个模式伴随正确。

## 冻结内核对照

```bash
python research/experiments/column_projection_consistency/analyze_projection.py
```

±65°、24x32x4、float64、三随机种子：

| 掩膜/内核 | 列约束相对真实顶面最大误差 | 150 cap 后真实面积 L2 残差比 | 1000 cap 表现 |
| --- | ---: | ---: | --- |
| 全湿旧版 | 1.20e-16 至 2.89e-16 | 3.17e-14 至 1.44e-12 | 两例动能增长，一例非有限 |
| 台阶旧版 | 0.312 至 0.696 | 0.0589 至 0.0716 | 残差增大至起点 2.64 至 6.34 倍 |
| 全湿修复 | 1.20e-16 至 2.89e-16 | 7.10e-13 至 1.03e-12 | 在容差停止，不过度迭代 |
| 台阶修复 | 1.20e-16 至 2.89e-16 | 8.99e-11 至 2.09e-10 | 8.67e-13 至 9.10e-13 |

修复湿动能比约 0.819 至 0.839，未增加。这些是局部随机测试，不是
气候精度、生产吞吐或所有地形的收敛证明。原始数据保存在本地
`results/industrial_alignment/column_projection_comparison.json`，旧非有限
结果保留为 false/null，未填零。

## 真实地形一日测试

```bash
python scripts/verify_debug_integration.py --days 1 --cases baseline ice --dtype float64 --kappa-bi 2e14 --audit-budget --out results/industrial_alignment/column_projection_1d.json
python research/experiments/nonlinear_process_budgets/analyze_attribution.py --input results/industrial_alignment/column_projection_1d.json --out results/industrial_alignment/column_projection_attribution.json
```

真实 ETOPO、合成初值/强迫、2°、180x65x14、±65°，80 次平滑、500 m
最小深度，dt600/dt_bt50；基线和动态冰各 144 步有限稳定。峰值速度
约 0.6043 m/s、海面 0.9834 m，冰组最大保存冰厚 0.02412 m。
本轮命令未指定 `OCEAN_PAV_NITER`；当前环境检查未设置该开关，代码
默认 cap 为 150。但原运行 JSON 没有保存环境开关，后续必须纳入
机器可读来源，不能仅依赖事后环境和文档保证重演。

| 一日累计/实际监测 | 基线 | 混合层 + 冰 |
| --- | ---: | ---: |
| 固定节点焓预算残差 (J) | -2.028493073e21 | -2.043024553e21 |
| 相比旧投影残差绝对值减少 | 2.600% | 2.519% |
| 真正顶面平流热输运 (J) | -2.028493073e21 | -2.043024553e21 |
| 残差加线性化海面库存 (J) | +2.051110754e21 | +2.082575700e21 |
| 实际 stage-2 累计面积 L2 残差比 | 0.014638 | 0.014628 |

最后一行是 `sqrt(sum_step A*Fz_after^2 / sum_step A*Fz_before^2)`，
不是逐步最大值或热预算残差。默认 cap 在这个真实地形仍不充分；
小网格 1e-9 门槛不能外推生产收敛。只修湿面也不能解决 stage-1 及
自由面/温盐时间耦合，因此没有 conservation PASS。

分析执行时逐项验证了原运行源码哈希。内核 SHA256：
`34904c8b2e0bd54a645914f4fd71e5770d8bca31423764ff0920f962d6c8578a`；
账本：`fc0eb7fdb7c3d90194ab3c862030675500660c4d6fe0f9018574042401caf233`。
原始 JSON 仅存本地，短期代理残差不能外推真实海洋百年漏损。

补充用同元组执行 `--dtype float32`，输出
`results/industrial_alignment/column_projection_float32_1d.json`；两组各
144 步仍稳定，运行源码哈希亦逐项匹配。这不修改 float64 预注册门槛，
也不把两种精度曲线相近冒充库存可靠。

| float32 一日实际量 | 基线 | 混合层 + 冰 |
| --- | ---: | ---: |
| 投影累计面积 L2 残差比 | 0.015873 | 0.015908 |
| 海面位移体积残差 (m³) | -3.501705e8 | -3.306940e8 |
| 非线性记账热残差 (J) | -1.876769e18 | -1.878697e18 |
| 未解释名义水盐变化 (kg) | 0 | +4.719193e10 |

float64 的位移体积残差约 -0.43/-0.47 m³；冰组盐残差约 2.635e10 kg。
精度敏感性真实存在：账本使用 float64 累计不能消除 float32 预报状态
更新的误差。上述非线性记账残差不是自动确认的物理漏损或已解释外源，
还需逐阶段舍入与几何兼容性研究，不用巨大参考库存归一化后称零。

## 下一项必要工作

真实地形迭代/预条件敏感性须先注册；保存迭代配置与实际残差，再让
正压时间平均输运、单元体积和 h*C 在实际 RK/子步时层匹配，同时
量化混合精度的实际库存误差。节点/
浅海部分单元、完整冰盐水、对流 kappa/n 嫌疑、持久预算/重启、真实
强迫百年、独立气候预报、GPU/分布式和全模式伴随仍属完整验收范围。
