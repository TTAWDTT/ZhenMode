# C 工业质量与速度联合验收合同

本包基于 `82e1ca4a2158d4c0ed20f91c04e80cdf43f8a36b`，分支
`codex/ocean-quality-speed-gate`，仅新增 `scripts/quality_speed/`、对应测试及本说明/证据。
不修改 solver、生产驱动、旧 benchmark 模块、科学门槛或生产默认。
此 checkout 和该提交未提供 AGENTS.md 或 .agents/skills；已读取 S0/S1 执行记录、
legacy 状态及生产冻结协议。原 §37 两个移动限幅时间阶失败及 1° 失败继续有效。

本次交付是独立验收工具与可复现负对照，**没有新的工业或完整物理资格结果**。
旧 SST 内部门禁的 PASS 不转换成此合同的 PASS；旧 payload 缺字段将 INCOMPLETE。

## 执行与接口

无需第三方依赖运行门禁和合同测试：

```bash
python -m unittest discover -s tests -p test_quality_speed_contract.py -v
python scripts/quality_speed/gate.py --control evidence/control.json \
  --candidate evidence/candidate.json --policy evidence/policy.json \
  --policy-sha256 PRE_REGISTERED_CANONICAL_SHA256 --out new-verdict.json
```

`--out` 使用排他创建，已有证据不会覆盖。JSON 重复键、NaN/Inf 拒绝。
门禁顺序：

1. 验证预登记 policy 的 canonical SHA256（`contract.digest`）、必填字段、完整积分和附件字节。
2. 比较双方及 policy 的 `pair_contract`；不同硬件、精度、网格、初边值、forcing、闭合、
   评分窗口、输出节奏、时长、dt/dt_bt 均 NOT_COMPARABLE，不能拿不可比当通过。
   双方 source SHA、config SHA 和算法也必须匹配 policy 的 `runs` 登记。
3. 两方均满足绝对科学门槛，且候选每项不劣于对照，才计算速度。
   必须包含 `heat_budget`、`salt_budget`、`volume_budget`、`solution_error`、
   `convergence_order`；可登记更多三维温盐、SSH、海冰、混合层和区域指标。
4. 至少三个一一配对重复，纯积分与总 wall 的**每个配对比值**均达到预登记
   `minimum_speedup > 1`。报告中位数、最小值、最大值和全部比值；不是置信区间。
   不硬定 2×，也不允许只快某个 kernel 就通过。

退出码：PASS=0；QUALITY_FAIL/SPEED_FAIL=1；INCOMPLETE=2；NOT_COMPARABLE=3。
在质量失败、缺证据或不可比时 `speed=null`。`synthetic_control` 即使 PASS，
`industrial_qualified` 仍为 false；工业标签只表示所提交登记合同满足，不表示百年、
气候或预报资格。

## Manifest 与冻结方式

schema_version 为严格整数 1（拒绝 true/1.0）。完整机器字段见 `contract.run_errors` 和测试中的具名夹具。
夹具是人为的单元测试数据，绝不是 MOM6 结果或科学阈值来源。

- `source_sha` 为实际 40 位 Git SHA，`source_clean=true`；`source_tree_sha256`
  指向包含实际参与源码逐文件 SHA256 的冻结清单；`config_sha256` 指向完整配置。
- `hardware` 必须含 device、cpu、accelerator、runtime、threads、ranks、memory_limit_bytes；
  environment、command、precision、algorithm 不得空。精度应包括状态/累加精度，runtime 应含
  编译器、JAX/XLA 或 MPI/编译选项；GPU型号/数目/显存、CPU亲和及线程设置写入实际文件。
- grid/initial/forcing/boundary/physics/scoring/output 的 `_sha256` 各指向实际文件或
  冻结文件清单，包含层数/坐标/湿区/体积权重、时间插值/日历、区域/时间窗口及输出策略。
  `manifest.attach_files` 流式计算选定文件 hash；CLI 验证所有引用的附件存在且未变。
  相对路径须留在 manifest 所在根目录内，不能逃逸或引用未附带的 hash。
- requested/accepted/attempted 步数必须全等，accepted×dt 必须等于 duration；
  verdict=PASS、coverage_complete=true、not_comparable=false 都须显式给出，不能从缺失推断。
  not_comparable 必须为 JSON 布尔值；任一方 true 则 NOT_COMPARABLE，缺失/错误类型则
  INCOMPLETE。顶层字段采用 RUN_FIELDS 白名单（含 artifacts），不支持的 status/pass/
  comparable 等别名及拼写错误不会被静默忽略，而是 INCOMPLETE；不能用额外字段覆盖 verdict。
  这是对尚未验收的 schema 1 的收紧；旧 manifest 需补显式声明，不自动迁移。
  pair_contract 用 canonical JSON 身份比较，避免 true==1，并区分 1 与 1.0。
- quality 的每项有非负 value、unit、definition_sha256、evidence_sha256。
  库存指标必须是**扣除实际外源后的绝对残差**，不是巨大初始库存归一化的百分比。
  如热残差 `abs(ΔH−∫Qdt)/(湿表面积×时长)` W/m²；盐/水量使用登记的 kg/s、m³/s。
  残差保留带符号原始账本，实际移动库存与固定参考库存不可混用。
- 误差需同域面积/体积权重和独立参照；不能把 WOA 初值自动称作独立验证。
  收敛需至少三档步长/网格、误差范数、细参照充分性及原冻结失败案例原件。
  现有 §37 的 1.9 时间阶与参照差 10% 要求不能降低；缺完整模式收敛即缺证据。
- policy 显式给每项 category（conservation/error/convergence）、direction（max/min）、
  limit、unit、definition_sha256，以及完整 pair_contract 和双方 runs。
  **本包没有批准新的物理容差或速度目标**；须协调者从既有冻结协议及选定场景登记、审查、
  保存 hash 后再运行。不能按已跑结果改 policy 后传入新 hash 规避预登记。

门禁校验提交的汇总和附件身份，不能证明汇总真实来自那些附件，不能认证伪造的 source、
计时或工业标签。独立 scorer/账本复算与运行见证仍是验收责任；此包不伪造自动认证。
不支持以不同 dt 自动比 time-to-solution；这需要另登记精度等价合同，当前明确不可比。

## 分项计时与最小监测对照

`timing.measure_lifecycle` 提供有同步边界的 callback 接口：setup → cold_compile
（含 trace/lower）→ warmup → integration → diagnostics_io → total。
同步回调必须等全部状态与诊断叶子完成；异步入队时间不能当执行时间。
暖机后必须重置测量初态；每次冷计时使用新进程和禁用/隔离持久编译缓存。
积分包含真实生产监测，I/O 包含实际输出写完；total 含全部分项及调度余量。
纯积分和 total 均必须大于零；每个 trial 的 phase_status 必须显式列出五个分项及 total 的状态。
measured 必须为正时长；not_applicable 必须恰为零且有非空文本 reason。
冷编译无需运行时编译（AOT/解释执行）的环境可声明 not_applicable 并说明原因，
不能靠省略字段得到零；integration 和 total 永远不能豁免。报告另区分进程启动墙钟，避免把 import、初始化隐去。
峰值内存必须是实测值及方法，或显式 unavailable+原因，不填假零。

```bash
python scripts/quality_speed/monitor_probe.py --out monitor-new.json
```

该入口只用 CPU、8×8×4 合成非均匀温度、8 个真实完整 solver 步，2 步暖机、
3 对交替顺序的新进程；每个子进程 60 秒超时，总共最多 6 个串行子进程。
计时比较既有完整步与同一步加既有 `classify_state`/主机峰值/首次拒绝检查，
双方测量末态全部同步并比较六字段字节。无监测路径只存在于此诊断脚本，
它**不能**满足生产门禁的 monitoring scope，未接入或替换生产驱动。
包括分项计时、实际 NPZ 写盘和父进程 wall；并不包含完整生产快照/重启策略，
因此结果只代表此合成控制下的监测开销。缺依赖/超时/状态不一致明确非零退出。
原交付因缺依赖未运行；续作实测状态见下文，不能把标准库计时单测当 JAX 实测。

## 工业配对场景选择与缺口

以下为建议先后顺序，尚未冻结新数值配置，不自动启动下载或积分：

| 场景 | 公平配对条件与用途 | 当前缺口/结论 |
| --- | --- | --- |
| 同源生产监测开销 | 上述 CPU 小控制、同初态同完整步，定位 host 同步及监测代价 | 原交付缺依赖；续作已安装 CPU 依赖，实测状态见下文 |
| MOM6 同物理短期海盆 | 冻结相同盆地/底形/有效体积、线性 EOS、封闭壁、无冰风驱/表面通量；同资源先完成 1 天预算，再独立登记 7/30 天 | 无 MOM6 可执行文件/版本编译见证、共同网格映射、成对输入和逐步账本；不能导入旧历史分数充数 |
| 真实 ETOPO/WOA/NCEP 2°配对 | 先保留 core 已登记 180×66×14、±66°边界及原 dt/闭合；对齐 MOM6 有效体积、表面源、calendar、恢复和输出，误差按共同物理评分域 | 本云 checkout 无 data；本地历史数据/hash清单不等于实际数据到位；尚无同版本同机重复总wall |
| 真实 1°及海冰生产场景 | 待 B 修复原失败并重跑原冻结案例、A 接受步源账本/严格重启闭环后，完整质量门通过再计时 | 原 negative-top/容量与时间阶失败仍保留；缺公平 MOM6 冰/混合层/边界闭合，当前不具验收条件 |

优先选择 MOM6 作为首个外部配对对象，是因为仓库已有 MOM6 输入/输出适配历史；
不是认定历史配置已公平。参考方案为项目内部技术选择，无新外部模式性能事实。
跨硬件结果另列成本/资源吞吐，不标纯算法加速。长期气候、百年、预报仍需独立资格矩阵。

## 首批资源和验收证据（历史，f95f739）

云环境 cgroup：4 CPU、16 GiB；默认 Python 缺 JAX、pytest、netCDF4，存在 NumPy。
未下载科学数据、未安装 JAX、未调用收费 API/GPU、未跑物理积分。仅安装小型 ruff 工具
到 `/tmp/ocean-c-lint` 进行静态检查；普通用户安装路径只读失败，未提权或换环境。
`quality_speed_evidence/` 保存实际单测日志、lint、blocked probe 与源/配置身份。
blocked probe 的 source_clean=false 如实记录当时尚未提交的新增工具，源码逐文件 hash
仍可核查；不伪称已经测量。全库回归和监测脚本的 JAX 运行未验证，远端 CI 另看 PR。

## B 复审修复（2026-09-30 续作）

新容器已恢复 shell，在同一工作分支从远端 f95f739 追加提交；core 基线仍为
82e1ca4a2158d4c0ed20f91c04e80cdf43f8a36b。检查未发现 AGENTS.md / .agents/skills，
/workspace/.agents 为空；已读 S0/S1 与冻结协议，未发现并发写进程。

B 的 not_comparable=true 绕过在原 head 复现为 PASS。修复后双方显式声明不可比
均停止于 NOT_COMPARABLE；无 speed、pass=false、industrial_qualified=false。
状态别名、布尔/非布尔混用、缺字段、schema 版本与冻结硬件类型均有回归，CLI 仍验证附件字节。
相同新增回归修复前 32 测试/70 子用例失败，修复后 32 测试通过；不是 70 个独立测试。
实际日志见 quality_speed_evidence/review_{before,after,ruff}.log。
新合同及既有 benchmark gate/manifest 定向 pytest：53 passed、240 subtests passed（4.03秒），
日志 review_targeted.log。

4 CPU quota / 16 GiB；官方 PyPI 项目声明依赖装入 /tmp/ocean-c-venv，
CPU JAX 0.11.2、NumPy 2.5.3、netCDF4 1.7.4、pytest 9.1.1。
仅小网格有界测试，无科学数据下载、模型、收费 API、外部 GPU 或长积分。
B PR3 @ 38765d32 的只读报告/测试审查保留：原 1° 接受步为0、moving-limiter 1.9仍失败；
未重跑 B 测试或修改其文件。独立复审与真实 MOM6 配对/物理资格仍未完成。

续作监测对照：monitor_probe_review.json 状态 MEASURED，6 个新进程（3对）末态六字段
字节一致，8×8×4、每次8步、2步暖机、每子进程60秒上限，外层180秒上限。
source_sha=f95f739、source_clean=false 如实记录修复尚未提交时的 checkout，
逐文件 source_tree hash 冻结实际代码；不能把 source_sha 单独当成全部运行代码身份。
本对照仅验证该合成小网格的监测开销，不是生产监测全输出、更不是工业配对资格。
真实数据、MOM6 与原失败场景仍未验证；旧 BLOCKED 文件作为历史原件保留。

A 复审续修：读取 PR4 固定 head 的 c_peer_review.txt（当时 PR2 评论时间线仅含 B，
故引用仓库内原始审查），在 cec135c 复现未解释 cold_compile=0 返回 PASS。
新增 phase_status、零时长理由与正负 AOT 对照。37项最终合同测试及CLI状态见
phase_review_after.log；首次新增36项修复前26子用例失败（phase_review_before.log），
随后另补1项零时长理由类型回归。timing helper 给实际测量阶段明确 measured 状态。
监测 probe 的历史计时原件不改写；该 probe 本身并非可提交工业 gate 的 manifest。
