# B：capacity / negative-top 的最小复现与诊断交付

**结论：复现并补齐拒绝原因诊断；没有修复真实 1° 的几何限制，也没有恢复真实完整积分。**
新增的是读取既有结果的离线诊断，不是另一个求解器候选。原数值源码、生产驱动、
benchmark gate、1.9 时间阶门槛和所有数值拒绝条件保持不变。

## 基线与文件边界

- 云端路径 `/workspace/ocean-solver`；唯一仓库 `TTAWDTT/ocean-solver`。
- 显式 fetch 后核验 `origin/ttawdtt/core-repair-review` 为
  `82e1ca4a2158d4c0ed20f91c04e80cdf43f8a36b`，从此建立 `codex/ocean-numerical-repair`。
  此 checkout 默认 fetch refspec 只抓 main，故明确 fetch core 引用，没有替换基线。
- 基线树不存在 `AGENTS.md` / `.agents/skills`；环境 `/workspace/.agents` 为空。
  已读交付执行记录、原核心修复状态和 Space 中的 A/B/C 边界。
- 仅新增 `research/experiments/material_failure_replay/`、
  `tests/test_material_failure_replay.py` 及本文；没有写跨包共享模块。

## 哪个问题真正阻塞完整步

原状态记录 §22–23 区分两个依次发生的问题：128 计算容量首先拒绝；另行登记
256 容量后接受一小步，随后 `h_top = 2.5 + eta` 为负。历史两档尝试的整柱仍约
747.5 m，故不是整柱干涸。云端没有这些原始数组，历史数字在此仅为引用，
不作为重新执行证据。

当前代码中的因果链可直接核对：

1. `material_thickness` 将全部自由面变化加在首节点。
2. `_nonlinear_subcycle_plan` 取起点/终点最小厚度，采用所有湿界面可能对流的安全上界。
   正厚度趋零时，所需子步数增长；越过零时，增加容量不可能使厚度重新为正。
3. 不支持的计划执行零个 tracer 子步，但 `attempted_state.eta` 仍来自动力预测。
   因此拒绝态的库存实现残差也可能很大。该残差必须保留，但不能被误认为
   先发生了已接受态的库存泄漏。返回态仍由既有总门禁逐字节回滚。

生产 CLI 调用原 FD 工厂，没有调用 `material_top`；弱式 r-star/moving-limiter
研究路径也不被生产 CLI 或材料工厂导入。三条路径的通过/失败资格不得相互挪用。

## 固定的合成最小复现

8×8×4、50 m 全湿柱，`z=[0,-5,-20,-50]`，float64，dt/dt_bt=10/5 s；
均匀 T=15、S=35，eta=-2.49988 m，u 为幅度 1 m/s 的确定性经度索引正弦场、v=0，
`kappa_conv=0.0015`，其余扩散/拖曳为零，禁用极盖，全部 factory 参数写入输入包。
这是专门构造的阈值穿越案例，没有随机采样，不是从真实 1° 状态裁出的子域。
最初探索用较薄初态直接触发负顶厚；随后选择上述初态以固定展示先容量、后几何的序列。
未将这些合成参数替换原 1° 配置。

| 独立调用 | 原始 valid | 非线性要求/激活 | 最薄湿节点 m | 同柱总厚 m |
| --- | --- | --- | --- | --- |
| 原容量128，初态 | false | 161 / 0 | +3.841197895e-5 | 47.500038412 |
| 仅容量256，同一初态 | true | 161 / 161 | +3.841197895e-5 | 47.500038412 |
| 容量256，从该接受态下一步 | false | 1326 / 0 | -4.317595002e-5 | 47.499956824 |

三次均运行原材料候选的完整调用，实际是**合成候选完整调用1次接受、2次拒绝**，
真实 ocean steps 接受数仍0。原128轨迹在首次拒绝停止；256 是独立反事实诊断，
不是悄悄扩容后宣称原合同通过。额外回归将下一步容量增至2048，容量条件满足，
仍因几何拒绝。两处拒绝的返回六字段与各自输入逐字节相等。

接受步原联合CFL约0.4617402≤0.5，局地库存舍入比约0.00189585≤1；下一拒绝步
速度峰值约1 m/s，最薄节点位于 `[0,0,0]`。几何另用 NumPy 从原始 eta、湿掩膜
及参考宽度重构，不调用 `material_thickness`。没有夹厚度、温盐或扣除残差。

## 最小诊断修补与使用

`audit.diagnose(result, initial, params, maximum)`只读取既有 `MaterialTopResult`，
保留其 `valid`、每项原检查、失败项、未裁剪子步数、独立起止几何、最小节点索引、
整柱厚度和回滚字节证据。诊断顺序先非有限值、整柱/节点几何、再容量与其他拒绝。
这是一种原因分类优先级，不冒称检测到了算子执行时序上的首个失败。
缺少必要检查字段直接报错，不将缺值当零。拒绝步预算明确不是接受步增量。

```sh
JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python research/experiments/material_failure_replay/audit.py --output /tmp/material_replay_new
python -m pytest tests/test_material_failure_replay.py -q
```

输出目录必须新建。复现成功时 CLI **退出1**，因为仍存在物理拒绝；若未复现预定
序列则退出2。报告另列 `synthetic_reproduction_matched=true`、
`qualification_passed=false`、`real_1degree_replayed=false`，不把负对照测试绿解释为资格绿。

证据在 `research/experiments/material_failure_replay/evidence/`，四个小型 npz 保存全部
输入/工厂参数、各次起点、尝试态、返回态、原 checks 和预算；不含需 pickle 的对象。
`config.json` 补充 None 参数及容量/门槛，`report.json` 保存逐文件 SHA256、设备和成本。
报告的 `checkout_head_at_execution` 是运行时基线，不声称新增诊断已经存在于该提交；
实际执行脚本及测试用其独立源码 hash 标识。

- 配置 SHA256：`f0587dc35f99fff20af3ef5d2de1f0fb27ec6202e1a6d9a47b0cbd26fd9b41d8`
- 合成全部输入 SHA256：`6276df46004104e7a99ed7b2f81c6e866e2643671c1e93dec06bf740118ea2e7`
- 原始包 hashes 逐项核对，并全部以 `allow_pickle=False` 读取成功。
- Python3.12.14、JAX0.4.38、NumPy2.5.3、CPU float64；不同于历史本地 JAX0.11.2，
  不宣称跨环境字节一致。正式复现含编译13.05秒、进程峰值RSS 532636 KiB。
  该计时与相关回归重叠，不作速度比较。
- 环境5 CPU、约17 GiB可用内存；只安装必要CPU测试依赖（轮子约171 MiB），
  没有付费API、GPU、海洋数据下载或长积分。实际账单不可见，不猜货币费用。

## 本轮测试

- 新增拒绝诊断回归9项通过，18.78秒；含2048容量反事实及六字段原字节回滚。
- 原材料、实际几何子步、checkpoint分块和节点动量度量回归61项通过，186.43秒。
- 仓库ruff、显式审计脚本ruff与diff空白检查通过。日志和JUnit随证据包保存。
- 没有运行全库测试或真实积分。70项回归通过不是70项物理资格通过。
- 与指定基线比较，`src/`、生产脚本、打包配置和CI文件diff均为空。

## 未完成项与下一步

真正的数值修复尚未交付。仅修改厚度分母或换成 r-star 会同时改变库存、源归一化、
相对输运、压力/扩散度量与重启语义；本轮不以局部正厚度代替这些联合条件。
也没有以修改对流调度、减小原dt、增加容量或调大黏性来绕过原失败。

首要缺口是原1°两档的最后接受态、首拒绝尝试态、网格/参数、强迫时钟和冻结来源
hash，尤其是 §23 的 thin-top 包及其输入。仅有文档或另一路指定 tracer 步不能补足。
这些原件未在本 checkout 的 data/ 或 tracked results 中找到；不得用同名合成数据顶替。

原 moving-limiter 0.62–1.16 未达1.9，根因未隔离；本轮没有改其控制或重跑完整细化
矩阵。它不是当前材料工厂首拒绝的调用路径。恢复联合几何/输运接口之前，不启动
另一批局部候选。真实1°、7/30天、跨进程重启、冰、全步梯度及公平MOM6质量/速度
验收均保持未验证/原失败状态；本PR不得据此晋升生产。
