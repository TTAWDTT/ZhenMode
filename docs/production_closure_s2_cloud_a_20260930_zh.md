# 范围 A 生产闭环云端交付记录

本批为 S2 的首个可审查工程交付，不是 S2/G2 整体放行，也不是工业资格。
实际目录 `/workspace/ocean-solver`，远端仅 `TTAWDTT/ocean-solver`。
fetch 后核验基线 `82e1ca4a2158d4c0ed20f91c04e80cdf43f8a36b`，从此建立
`codex/ocean-production-closure`；draft PR 目标 `ttawdtt/core-repair-review`，不向 main 合并。
实现提交 `c4df6039e94dd3b3efcae02eaa86f86f0559dcbd`。
指定提交与工作区没有 `AGENTS.md` 或 `.agents/skills`；已读取 S0/S1 执行记录、
S2 方案、冻结协议相关边界和历史 §27–28。其他仓库未操作。

## 实际改动

仅生产驱动、新生产测试与一个有界见证脚本；`stage_budgets.py`、
`restart_contract.py`、数值核心与 benchmark gate 均保持基线内容。
没有改黏性、步长建议、默认数值方案或任一原物理门槛。

- `--budget-audit` 显式启用严格接受步账本。普通步保持原计算图；同输入另算已有
  `make_budget_step`，逐步比较六字段 shape、dtype 与原始字节。相同且状态/账本
  有限、原监测接受后才累计。最大值字段取最大，其余沿用既有累计合同。
  审计不一致返回 `FAIL_AUDIT_IDENTITY`、代码7；账本非有限为 `FAIL_LEDGER`、代码2，
  CLI 两者退出1。以上是失败码，不是退出码。默认不开启账本，不声称默认已闭合。
- 拒绝步原始账本单独放在 `rejected_*.npz`；代码7另保存审计六字段。
  拒绝态 `resumable=False`，不会污染接受累计或覆盖最后有效 checkpoint。
- 诊断 schema 4；复用已有严格 checkpoint/原子写入，新增接受累计、快照时累计
  历史、全步 eta 峰值。恢复时验证键、形状、最后历史与累计字节一致。
  强迫时间由绝对接受步×dt恢复；记录 elapsed seconds 和360天相位。
  旧版本缺新源/生产合同时拒绝，不声称迁移续跑。
- 默认强迫加载错误直接失败；仅显式 `--allow-forcing-fallback` 允许探索回退。
  `--strict-forcing` 在加载前要求选定本地地形、初值及强迫缓存都存在，禁止回退，
  不下载缺项。记录 requested/applied、回退原因、实际选定文件 SHA256、计算 dtype
  数组身份、年/月、单位、插值、平滑、冰下限、恢复参数及360天重复历。
  严格来源不等于数据产品质量或真实历史/Gregorian资格。
- 每次正常完成/数值拒绝报告含源码文件hash、Git HEAD、有效参数/网格身份、
  运行设备、JAX/NumPy/Python版本、有关环境旗标及六字段终态原始字节身份。
  Git HEAD与文件hash分别记录，不把未提交时的运行冒称等于HEAD。

## 账本解释

`ledger_source_inputs` 是已有阶段审计实际注入的规定热、bulk/coastal热、温盐恢复、
brine、sponge及海冰大气/盐源。库存仍为固定参考节点海水显热减海冰潜热、
名义水盐质量及分开的 eta 排水体积；`surface_displacement_tracer_change` 是所选
线性化项，**不是完整实际变体积库存**。

`advection_boundary_changes` 保持内部参考域输运诊断名称和解释，不映射成
`netboundaryflux`。`budget_residual` 是观测变化减实际源，保留有符号和逐步绝对
累计量，另给湿面积×接受秒数的 W/m²。残差既不扣回状态，也不冒充源。
`physical_budget_closed=False`、外部净边界通量 `not_measured` 明示未验收。
现有稳定性 PASS 仍仅表示原稳定性/漂移合同，不代表物理预算闭合。

## 测试与重启见证

证据位于 `results/production_delivery/s2_cloud_a/`。修后定向回归 **141通过、0失败、512.41秒**，范围为生产账本/重启/监测、
通用重启轨迹、阶段账本与诊断六个测试文件；全仓Ruff通过，未在本地跑全库。最终结果以
`reviewed_targeted.log/.xml` 和 `cross_process_reviewed/report.json` 为准。
源码/数据/配置身份见各进程 `*_identity.json`；其 grid/forcing/effective_params
逐数组含 dtype、shape、SHA256，原始终态另存 `*_endpoint.npz`。

四个独立进程使用已有8×8×4全湿合成夹具、T=17°C、S=35、月序列指定风应力、
外步10s/正压5s、scan和JIT风；连续8步与2+2+4步均推进80秒。
两种结果包括六字段原始终态、账本、完整历史、源身份、时间/输出计数均字节一致。
四份 StableHLO hash 相同；这仅是当前 CPU 图见证，不是旧 CUDA 可执行的比较。
`verify_saved_arrays.py` 不导入求解器/重启代码，独立复核18份文件checksum、六字段
原始终态及24份源码与实现提交相同；`independent_verification.json` 保存结果。
有效配置 SHA256 为 `1d342168957f075c242a1b2ea230130f035bf914df84b585c82ae6ecfb5fa22c`，
合成强迫/初值身份 SHA256 为 `20bfdf52e32671017802e9b648393e7162f5cdb39cd5be48c508bde7281a56c5`。
这些是排序规范JSON身份的hash，逐数组原始字节hash另在进程合同中。
最终四进程墙钟约66.92秒（与定向回归并发），不作公平吞吐或加速比。

新测试覆盖非均匀规定热的独立面积×时间源公式、独立NumPy终点参考热库存、
拒绝步不累计、审计字节失败、非有限单步/累计账本、两次重启历史、eta快照间峰值、
显式空气回退、缺输入及非有限输入拒绝、安装布局源码身份、离线地形路径选择。
既有 stage-budget 回归覆盖其他已定义源/冰审计；不是新增生产实海冰资格。

## 保留的失败与互审

1. 原生产监测/重启86项通过；新增测试初轮2通过1失败：把逐步残差之和与两个
   约1e18的累计量相减要求1J绝对误差，实测差768J。保留 `first_heat_probe.npz`。
   新代数比较改用操作数尺度的8eps舍入界；实际残差/模型/原物理门槛不变。
2. `cross_process/last.log` 的初次进程探针为 `restart contract mismatch: sources`：
   本任务在最后进程启动前补改了报告源码，被严格合同正确拒绝。不是历史分歧复现。
   后续冻结版本 `cross_process_frozen` 全字段通过；不覆盖首次记录。
3. 首次扩大回归在互审前源码发现非有限初态报告失败；记录于 `final_targeted.log/.xml`。
   因缺陷已修且新版套件启动，旧套件收到中断，不作为全套通过证据。
4. 独立只读代理互审发现并复核修复四项：wheel布局空hash、地形双文件/无netCDF
   选择错误、非有限初态被身份编码器拒绝、maximum隐藏单步负无穷。
   只读复核提交c4df603确认四项已解决；平铺测试只是布局模拟，未声称实装wheel验证。
   另对C PR #2精确head `f95f7391c304d82aed37ce6303159ede087ab1f0` 做只读互审，
   `c_peer_review.txt` 保存纯Python复现：原policy hash正确拒绝改门槛，外部新policy
   审批仍是设计边界；未说明原因的cold_compile=0会PASS，是文档校验缺口，交C处理。
   无A/C文件冲突，未修改C源码。跨包集成及联合回归仍由协调者完成，不merge。

## 未完成与成本

原真实2°/30天跨进程CUDA字节FAIL原样保留；云端无原始数据/检查点，未重跑。
不能用上述短CPU控制消除它，根因仍未隔离。本批未进行1°真实积分、MOM6对照、
GPU、7/30天、季节/百年或独立气候/预报资格。
S2分海盆/深度账本、实际变体积物理库存、真实外部netboundaryflux、完整能量分析、
任意配置审计字节相同与低开销审计路径仍待后续；G2不整体通过。

资源预检可见5 CPU、17GiB内存；仅CPU小网格和有界测试。安装临时CPU依赖环境
约631MiB磁盘，CPU JAX wheel约85.7MiB，未下载科学数据，未用付费API/外部GPU。
审计模式会运行普通与审计两步并做主机字节核对，明确增加成本；不是性能优化完成。
GitHub CI单独核验，远端CI绿不能替代上述缺失的科学资格。
