# D1 finite operator diagnosis — executed 2026-10-13 Asia/Shanghai

**FINITE STENCIL DEVELOPMENT PASS；physical/eta attribution OPEN。** 唯一预声明四实例全部完成，108/108 正向有限比较通过，41/41 comparator-group 的 +1 植入错误捕获，另两种 Ny 的显式错误符号均捕获。未修改生产、native package、参数或阈值；没有重跑、扩 grid、时间演化或拟合。此结果不能给冻结70的其他科学、CI或native资格补章。

## 原始记录

- [声明](DECLARATION.md)、[根静态确认与最终hash](root-static-confirmation.json)、[执行hash/环境/CPU](execution-receipt.json)。实现及预算在数值执行前冻结。
- [按阶段资源账](scoreboard.jsonl)、[实测 timing 决策](timing-decision.json)，各阶段 stdout 和0.2s采样RSS见 `truth.log`、`timing.log`、`complete.log` 与各 `*-resources.jsonl`。三个阶段退出码均0，无丢弃运行。
- [完整摘要](complete/summary.json)、[逐项 gate 与植入错误](complete/gates.jsonl)、[MP truth birth/contact记录](truth/truth.json)。对每个比较都保存 max_abs、声明scale和预算；不能仅凭此摘要重建精度声明。
- 四组完整数组：`complete/case01_ny8_phase0.5235987755982988.npz`、`case02_ny8_phase1.047197551196598.npz`、`case03_ny16_phase0.5235987755982988.npz`、`case04_ny16_phase1.047197551196598.npz`；同名 JSON 保存分墙行/垂向区域的 signed_mean、max_abs。`controls_ny8.npz` 和 `controls_ny16.npz` 保留 sine/cos/sin2 及 pressure/continuity controls。

## 版本与资源的实际证据

`source-before.json/source-after.json`：installed 的133个 Python 文件在 before/after 均逐项与 wheel-identity、build-input、冻结 Git blob一致；146个完整 build-input key（含13非Python）在两次 frozen Git archive均一致。wheel SHA一致。不是当前可变worktree package import。`modules-before.json/modules-after.json` 记录35个实际 loaded zhenmode模块的 `__file__` 和SHA，均落于冻结 installed目录，before/after不变。冻结身份 `70d408583aa99db63e49d958d89dcaccad273f0e`。结束检查工作树 `git status --short` 仍为空。

CPU affinity=[0]，CPU backend，线程环境1。三个阶段实际总墙钟15.593388335s；truth阶段 supervisor耗时0.208069746s，timing阶段结束时累计7.308461690s，complete结束时累计15.593388335s。采样process-tree RSS峰735141888B（约701.1MiB），未触发180s/4GiB限制。Windows Git archive用于出处核对；RSS采样范围是WSL进程树，未计Windows Git进程本身，不把这份采样账说成全主机内存证书。

Timing-only 未读连续Nu/Nv oracle，未执行有限数值比较。测得 identity/import1.831745084s、factory初始化1.857270880s、首helper组合2.090395135s、同组合warm0.043076558s。预声明投影6.856325049s，小于剩余172.688195043s，才启动一次complete。实际complete阶段含final出处核对约8.285s；其数值摘要在6.686841208s写出。投影不是实测耗时，不能混为同一数。

## 有限问题得到的答案

1. **u derivative 的 wall差按预言出现。** 原始产品的 edge-u 与 odd-u 反事实差满足 south `v₀u₀/h`、north `−v_last u_last/h`、其他行0。四实例 `primary_raw_wall_delta` 最大 residual依次为5.169878828e−26、3.877409121e−26、2.261821987e−26、2.584939414e−26，声明预算均5.684341886e−14（原始两component组保守scale=2）。这验证的是固定field的有限索引公式，不是一般u应设odd的证明。
2. **filter的 edge/odd差按预言出现。** 单独sin2产品 controls分别验证 south `[(9q₀+q₁)/16,q₀/8]`、north `[q_last/8,(9q_last+q_penult)/16]`，并对完整finite-index filter比对；cos的width2 edge与strict-even差也被捕获。这没有把梯度误差或vertical错误归入filter。
3. **pressure/continuity保持自己的paired role。** cos标量的gradient为 `−K4 sin`，normal-sine的divergence为 `+K4 cos`；mass actual faces/dispatch同单独channel算子一致，两墙闭通量控制通过。所有正向gate中最大 residual为5.273559367e−15（Ny16 continuity_sin_symbol），对应声明预算5.456968211e−12。两Ny的 `sum(cos*Dsin + sin*Gcos)` residual分别1.776356839e−15、0，对应预算7.137595208e−13、1.428560480e−12。此有限work恒等不证明非线性whole-model稳定性。
4. **analytic w 与 native w 被分开验证。** 每个w输入下的完整native momentum tendency分别对独立scalar-index raw+filter期望通过；保存 native helper水平/垂向products、独立raw、filter后expected、signed residual。nativew−analyticw是另一描述量，不用它解释u ghost缺陷，也不从其有差异推断某一方案错误。

finite native reference最多只确认预声明舍入预算一致性。约1e−8量级force对应128eps预算含max(1,...) floor（约1e−14），**不能把极小实测残差升级成许多有效数字的独立native科学认证**。Native与float64 scalar oracle共享同一field样本，且scalar预言读取的是公开stencil定义；独立性仅针对数组索引/ghost闭合实现，非物理route完全独立。

## MP记录的强度

suite mpmath生成而非managed环境临时安装。80dps的Ny8/16有限identity最大residual为2.9518e−80、4.0060e−80；160dps分别2.1338e−160、3.3786e−160。每Ny对应28/52个identity，声明gate为10^(−dps+10)。比较native时把160dps值转换float64，仍按finite64预算判断，未声称160digit native一致。

MP与scalar-index来自同一作者并共享数学sin/cos、ghost定义和oracle代码；不同算术提高了实现检查强度，却不是新的独立物理证明。没有Arb ball enclosure，没有数字精度认证，truth birth记录明确该限制及本次contact。旧pilot/root error数据未进入任何reference或选择。

## 连续场误差：仅描述，不给PASS

下表直接读取同名case JSON；范围是south第0行、垂向内部节点，或距墙≥3行的水平内区且垂向内部。固定nz=9；不能拟合分辨率阶数或把垂向/滤波截断看成wall误差。

| Ny / phase | south0 Nu连续误差max_abs | 水平内区Nu误差max_abs | south0 Nv误差max_abs | 水平内区Nv误差max_abs |
|---|---:|---:|---:|---:|
| 8 / π/6 | 1.054611893e−10 | 7.549188244e−11 | 4.358525865e−10 | 4.358525865e−10 |
| 8 / π/3 | 1.054611893e−10 | 7.549188244e−11 | 1.307557759e−9 | 1.307557759e−9 |
| 16 / π/6 | 8.271270821e−11 | 6.555781333e−11 | 7.528797534e−11 | 3.784982117e−10 |
| 16 / π/3 | 8.271270821e−11 | 6.555781333e−11 | 2.258639260e−10 | 1.135494635e−9 |

native dual-face w在south首行底节点相对analytic-w的描述差：Ny8两相9.621342906e−6、1.666465475e−5；Ny16两相9.769721361e−6、1.692165377e−5。原静态阅读已说明旧w采样在physical bed以上；这里仍不据此作整体误差归因或更改w owner。

## 限制与停止位置

已将静态“此已制造field存在离散延拓差”的假设推进为一个冻结field/算子集合的DEVELOPMENT复现；没有给出保存eta误差的因果链，也没有选择通用切向parity、更改符号或开新grid。无黏不可穿透与另加黏性free-slip条件的区别仍如STATIC_REVIEW所述。原先70完整SCI/native余项和CI状态各自pending，不能用D1的software gate代替。此唯一D1已完成，数值调用停止。
