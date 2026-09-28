# 投影收敛：真实完整步 float64 通过，float32 超标，物理预算未闭合

日期：2026-09-28；预注册协议 `0bdc47d`，冻结轨迹内核 `cce8f87`。
完整工业级目标不变，物理体积/时层、百年及气候验收未完成。

## 实现与独立验证

cap、有效 rtol、none/jacobi 通过工厂与两条 CLI 显式传入，环境兼容
开关只在工厂构建时解析一次。显式 cap 优先，无效参数构建时报错。
生产 config 与 smoke 每个 case 保存有效值及 cap 来源，不依赖事后
猜测 trace 发生时的环境。保留旧 none/150 默认，但没有把它标成达标。

原生 -A*B*G3 的精确 Jacobi 对角以逐单位基矩阵独立验证；湿台阶、
陆地、干/孤立行、正向/JVP/VJP、体积内积和零右端均测试。预条件
只改变 Krylov 求解，不改物理算子、边界、温盐或源表。故意 cap=1
被实际阶段监测检出。新增 API 门槛最初因缺少实现出现导入失败，
实现后与既有投影测试共 36 项通过。全量 350 passed、23 个既有
warnings；ruff、diff-check、MMS ALL PASS，经向收敛比仍为 4.30。

账本另存单步最大相对残差；`accumulate_budget` 对此使用 max，而
库存与平方范数求和。累计 L2 小不等于每一步都收敛。零输运/未启用
投影不能被分析脚本解释为一次成功的非零右端投影。

## 固定右端 96 次比较

```bash
python research/experiments/projection_convergence/analyze_convergence.py
```

真实 ETOPO、合成初值/强迫、2°、180x66x14、中心±65°、80 次平滑、500m
最小深度、dt600/dt_bt50、kappa_bi2e14。冻结旧内核/none150 演化基线
与动态冰，float64/32，在第 1、72、144 步取同一个真实 stage-2 预测
速度；共 12 份原始 NPZ，输入 SHA256 保存在原始 JSON。首步只读
捕获与普通冻结内核的六个状态数组最大差全部为零。

每种精度/配置共有六个固定右端；“通过”同时要求实际原生面积 L2
残差、有限及湿动能门槛，非 CG 自报成功。

| 精度 | 预条件/cap | 通过数 | 最坏原生残差 | 编译后 CPU 中位秒/次 |
| --- | --- | ---: | ---: | ---: |
| float64 | none/150 | 0/6 | 2.387e-2 | 0.214 |
| float64 | none/300 | 0/6 | 1.733e-4 | 0.425 |
| float64 | none/600 | 6/6 | 1.147e-12 | 0.735 |
| float64 | jacobi/150 | 0/6 | 6.341e-3 | 0.236 |
| float64 | jacobi/300 | 4/6 | 1.316e-9 | 0.449 |
| float64 | jacobi/600 | 6/6 | 1.067e-12 | 0.508 |
| float32 | none/150 | 0/6 | 2.439e-2 | 0.109 |
| float32 | none/300 | 0/6 | 2.074e-4 | 0.219 |
| float32 | none/600 | 6/6 | 4.102e-5 | 0.249 |
| float32 | jacobi/150 | 0/6 | 6.341e-3 | 0.109 |
| float32 | jacobi/300 | 6/6 | 3.165e-5 | 0.176 |
| float32 | jacobi/600 | 6/6 | 3.165e-5 | 0.171 |

cap1200 也通过但残差未继续明显降低：600 已达到停止容差。float64
Jacobi/300 的两例失败保留，未把 1e-9 门槛调宽。选择 Jacobi/600
进入完整步复核；约 31% CPU 求解耗时降低只针对上述固定右端，同
准确度比较，不是整体吞吐或 GPU 性能。f32 实际修正残差高于 rtol
也不能忽略：CG 矩阵停止范数与实际速度修正/面积范数并非同一个量。

原始 `results/industrial_alignment/projection_convergence.json` 全 96 次
保留且来源 SHA256 在分析时逐项匹配。之后仅将相同的预条件乘法
lambda 改为 def 并补 API docstring 以通过 lint；完整积分将保存新的
运行源码哈希，不假称旧 JSON 的哈希等于随后编辑过的源码。

## 完整积分：稳定不是所有门槛都通过

```bash
python scripts/verify_debug_integration.py --days 1 --cases baseline ice --dtype float64 --kappa-bi 2e14 --audit-budget --projection-niter 600 --projection-preconditioner jacobi --out results/industrial_alignment/projection_jacobi_float64_1d.json
python research/experiments/nonlinear_process_budgets/analyze_attribution.py --input results/industrial_alignment/projection_jacobi_float64_1d.json --out results/industrial_alignment/projection_jacobi_float64_attribution.json
```

另以 float32 执行相同矩阵，文件名对应 `projection_jacobi_float32_*`。
运行数值内核 `02a385b`；有效 cap=600、rtol=1e-12/3.814697e-6、
Jacobi、来源 explicit 全部入 JSON。四组各 144 步稳定，峰值速度约
0.6043m/s、海面 0.9834m；两份归因 JSON 和源码哈希逐项核对。

| 精度/组 | 累计原生 L2 比 | 最坏单步比 | 投影门槛 | 固定节点热预算残差 J | eta 体积残差 m3 |
| --- | ---: | ---: | --- | ---: | ---: |
| float64 基线 | 9.926e-13 | 1.089e-12 | PASS | -2.027929320e21 | -0.428 |
| float64 冰 | 9.925e-13 | 1.092e-12 | PASS | -2.042456600e21 | -0.438 |
| float32 基线 | 3.604e-5 | 5.429e-5 | FAIL | -2.026096289e21 | -3.334e8 |
| float32 冰 | 3.611e-5 | 5.411e-5 | FAIL | -2.040734395e21 | -3.417e8 |

原协议每一步上限 float64=1e-9、float32=5e-5 不变。float32 平均比
通过但最坏步骤超标，不能按四组稳定的 `stability_pass` 宣称投影
全通过。固定右端、局部梯度、真实完整步的验证范围不能混用。

float64 相比此前 none/150 的约 1.46% 累计残差，投影现在收敛；
但热缺口仅减少约 0.028%，仍主要等于 stage-1/2 的真实顶面输运。
加线性化海面库存依然留下约 +2.052e21/+2.083e21J；没有物理守恒
PASS。float32 非线性记账热残差约 -1.90e18J，冰组未解释水盐变化
约 4.61e10kg，也不能被“投影稳定”或 float64 累计抹去。

内核 SHA256：`6043b25ad4628581610665c697620b878e82f30844522640d55e64700a4f0e9f`；
账本：`6ee824fa59f985ece377631918c6575792bfc1af11e1ccf4ecff432ad6bb1729`。

float32 精度反证另按[追加协议](precision_addendum.md)处理。多输出
捕获最初引起约一 ULP 温度差而被严格校验拒绝，保留这个反证；
快照分析不会被称为原批量轨迹的逐位重演，也不取代原超标证据。

## 精度反证：不是 float32 末态量化的不可逾越下限

```bash
python research/experiments/projection_convergence/analyze_precision.py
```

追加协议提交 `e72a3ed`、方法澄清 `2e74ba7` 先于重跑。原账本
单步图推进的两条轨迹各 144 步，基线 18 步、冰组 17 步超过 5e-5；
最坏分别在第 13/9 步。它们不是原 fori_loop 批量图的逐位重演，
最坏 5.482e-5/5.520e-5 不替代原批量的 5.429e-5/5.411e-5。
捕获图返回状态不用于推进；逐步与参考状态对比仍有温度约一 ULP、
冰组盐度约一 ULP 差，原始报告明确标 false，不制造零扰动证据。

固定最坏捕获右端，用同一真实网格重建 float64 一致度量求解，
然后把校正速度量化回 float32。以下原生输运均为相同算子的独立
op-by-op 探针，不是完整融合图的逐位重演；输入和输出 NPZ、步号及
SHA256 均保存于 `projection_precision.json`，来源哈希已逐项核对。

| 固定快照 | 原 float32 校正的原生残差 | float64 解量化回 float32 的原生残差 | 一致 float64 解/输运残差 |
| --- | ---: | ---: | ---: |
| 基线第 13 步 | 5.482e-5 | 3.034e-7 | 9.641e-13 |
| 冰组第 9 步 | 5.520e-5 | 2.130e-7 | 9.159e-13 |

同一已量化校正速度，参考 float64 算术与原生 float32 输运的差别
约 2e-7 至 3e-7；网格度量量化约 2e-8，比原来约 5.5e-5 小得多。
这反证“只能提高整个状态精度才可能过线”：主要问题在当前 float32
投影求解/修正路径，而不是单独的速度末态量化或输运算术。尚未
分开证明是停止范数、递推残差还是修正消去导致；不能直接把 CG
返回称作收敛。下一改动须研究实际残差控制/迭代改进或混合精度
求解并重新注册，而不是降低门槛或仅把诊断换成 float64。

全模式 float32 库存、真正体积与时层缺口不因这两个固定右端复核
而消失。本轮没有把高精度探针静默写入生产数值内核。

## 仍未获证的要求

网格/分辨率泛化、失败闭合的生产收敛门槛、真正
单元体积与 h*C/正压时间平均输运、混合精度、对流系数、完整物理、
真实强迫百年、独立气候/预报、GPU/分布式及全模式梯度仍需验收。

2026-09-29范围勘误：此前文字把ny误写为65；原JSON和12份NPZ
均为66。中心筛选abs(lat)<=65°在2°格网上含两端中心，整格边缘
到±66°。旧协议的65行文字不删改，实际只验证了66行夹具；
这不是一次新的65行实验。详见[几何复核](../volume_transport_consistency/review.md)。
