# 最终独立文案与阅读框架审查

结论：**准出，当前文本零项未解决 finding**。此结论仅允许按现有范围报告结果，不改变原生排名、物理资格或任何失败门槛。

被审公共报告：`D:/CodexWorktrees/thermal-full-scan/ocean-solver/docs/momentum_point_ece_results_zh.md`。最终 SHA256：`4cb36d7ba288a6fb91590ca2e68ce8bf95c6f1b60979b375cd7455ca05298429`。生产源码身份为 `ece37c7b7ad383314e4dcfd982641d9ca541e1c3`，最近完整验收配置为 `3c8e86bef2ee1fbfd50cca4343d6fc61694d9edc`。

审查者是新上下文非作者，未编写科学文案、未修改生产源码、未执行原生积分或证明/参考数值重算。已完整阅读安装的 BootLoops `prose-lint` 与 `referee-sim` 技能；审查交付位置遵从本任务指定的 `momentum-point-v1/final-frame-review.md`。执行了数值/来源检查、逐段意义与语法检查、独立的 colleague/read-aloud 检查、整篇顺序阅读及最后的开头冷读。实际 Markdown 使用 Python Markdown tables 扩展转换为 HTML，并从 HTML 读取文本、表格与链接；这验证 Markdown 结构和文本完整性，不声称浏览器视觉排版检查。

## 最小修正及复查

初始报告 SHA256 为 `5eee204eedfe4c34da264739491f53b6bda638f5b02301c588cfdcfb077a34be`。初始 L30 的“新的未用真实_compute_momentum_tendency入口瞬时RHS剖面”有 F1/F7 修饰关系歧义，可能读成没有使用真实入口。声明 `/physical_sources` 明确实际调用该入口。这是措辞缺陷，不是源码或结果矛盾。

根代理自行修改正文。首次仅换句首形成“入口计算（…）在freeze后正式预声明…”的双谓语直接拼接；审查者指出 F1 句子拼接缺口，根代理随后将完整首句修为：此前未测的剖面先在 freeze 后正式声明网格，并通过真实入口计算，零物理源条件和输出与平流的差紧邻入口说明。最终 L30 语法完整、修饰关系明确、数值和范围未变。审查者未直接改正文；最后审查的是根代理修改后的完整句子及其 HTML 文本。

最初渲染检查完成文件和哈希写入后，向 Windows GBK 控制台打印 Unicode 减号时发生 `UnicodeEncodeError`。采用 `python -X utf8` 重复纯文件渲染检查后正常退出；该设置错误不涉及科学计算。初始渲染和检查收据保存在 `final-frame-rendered-initial.html/.txt` 与 `final-frame-check-receipt-initial.json`，最终收据为 `final-frame-check-receipt.json`。

## 数值、范围及来源

| 检查对象 | 读取证据 | 结论 |
|---|---|---|
| 主试点五项误差、综合分数及百分比 | `docs/data/momentum_point_ece_primary_vs3c8.json` 与 pilot decision | 两表指标一致；0.005416191620723861 高于 0.005388509039568577，约增加 0.514%；温度/u/v 描述百分比一致。严格改善 FAIL，不推广。 |
| 原生执行范围 | pilot decision、native pilot audit | 实际仅索引 0/1，2..9 未跑；读回是保存数组审查，未冒充新积分。ece 无热风/null/六组完整验收。 |
| 原生权重与差分 | native pilot audit 的 rows/trajectory_difference_to3c8 | eta 相对振幅、速度相对尺度、T/S 绝对单位和五场 maxabs 差一致；描述性差分没有新等价门槛。 |
| 旧最佳与 baseline | 3c8 公共报告开头及六组表 | 六组属于 3c8；111 有限选配及不同观测位置紧邻首次 baseline 比较；MOM6 历史构建/库身份缺口保留，无全局最优/SOTA 资格。 |
| 制造流场与滤波限制 | `momentum-node-location-3c8/result.json` rows | 16/32/64 旧 w、点 w、16 节点实际 adv_u 数字相符；既有 DEVELOPMENT 身份、水平滤波残差和非因果限定保留。 |
| 本地控制和原零断言失败 | 两个 controls 日志、失败 attempt01 日志、constant diagnostic、strictzero fixture record | 148 PASS + 9 PASS = 157；原 1 FAIL/7 PASS、Nm/Fz 数字、非零名义均匀流不满足实际离散零散度相符，原失败未重标。 |
| 旧 scalar 资格 | source math 的 source/scalar_core_bridge、prior_scalar_records | 133 Python 文件单位明确；column/vertical 未改、transport 仅指定函数 AST 改动；旧 scalar1D/主动2D 明确 REUSED，而非新 blind。 |
| 真实动量入口控制 | new momentum declaration/result | 36×19/72×37/144×73、归一 L1 与比值相符；首次出现就给 f=0/eta=0/常量 T/S/零源；≤5.17e-26 m/s² 是实际入口与平流的有限差。L1=1 是代数推论，反号负例才是实际运行负例。 |
| ece 完整 CI 与旧跳测 | 原始 `ci-ece-full-attempt02.log` L107/L276/L841/L864、两份 docs CI JSON | 同源 1737 PASS、6710 warnings、1589.16 s、MMS ALL PASS；第一次 lint-only/test SKIPPED 保留，不用其 success 冒充完整验收。 |
| 模态与双幅值 | eta modal result | mode2 平方误差占比相符；Parseval ≤2.23e-16 保守包住 2.220446049250313e-16；main/half 来自 3c8 三对，ece 仅 main；与二次响应相容，不称因果识别或物理能量。 |
| 代数等级首次定义 | target spec、proof verdict snapshot | VERIFIED-CLOSED 在首次使用就限定为 Targets I/II 的程序级核对等级，排除 formal/refereed/native 能量资格；原快照的 numerical pending 与后续有限计算 chronology 明确相容。 |
| 有限参考与残差表 | reference spec v2、reference result、numeric audit、raw readback | n32/n64 eta/theta 收敛及表中全部 RMS/比例相符；eta 米制与首表归一指标明确区分，T 使用实际 h×area/时间权重；不换原评分。 |
| 独立性最弱共享层 | numeric audit/result/raw readback 与正文 L59/L61 | 新作者/新代码、实增广 expm 与 Hermitian 卷積演化分开，但共享物理合同、投影对象、binary64；仅 n16/41 时刻独立积分，非新 holdout。 |
| n64 边界独立范围 | reference result 的 nchecks 与 raw readback saved_grid_gate_readback | producer 通过原门槛；n64 压力/垂向速度历史未保存，独立审查仅实现表达式；正文没有把源码核查写成独立数值重算。 |

报告直接链接的 **19 个本地文件全部存在**。参考 export manifest 的 **14 份公开副本全部与原始输出逐字节相同**，且 SHA256 和字节长度同时符合 manifest。最终收据记录每条链接、每份副本与最终正文哈希。外部 CI URL 的 run ID 与原始本地终态和日志一致；本审查没有声称重新查询当前远端状态。

## 读者与标准异议

每个读者群体分别审查正确性与新意。这里的“回答位置”指该读者实际会看到的位置；不是由审查者拼接多处证据后才成立的答案。

| 读者 | 标准异议 | 轴 | 回答位置 | 严重性 | 处理 |
|---|---|---|---|---|---|
| 原生海洋模型用户 | 这个选项是否已能取代现有最佳并适用于一般海洋模拟？ | 正确性 | 标题和开头先给 FAIL/不推广；L18 缺失的热风/null/六组与 L22 限制通道 | 无未解决项 | 保持局部资格与原生采用范围。 |
| 原生海洋模型用户 | 比既有 3c8 配置实际改善了什么？ | 新意 | 开头与主表承认综合分数更差；L22 给显式点节点速度行为 | 无未解决项 | 仅报告新增选项和有限控制，不称排名进步。 |
| 数值分析人员 | 四倍误差比是否被误写成一般时空阶数或完整系统收敛？ | 正确性 | L30 给固定三档、分量归一、零源入口；末句排除压力/完整时间/能量 | 无未解决项 | 已回答固定瞬时组件范围。 |
| 数值分析人员 | Galerkin/eigenphase 与独立 expm 是否只是复用同一代码？ | 新意 | L59 首次数值独立声明即给 n16/41 和演化算法差异；L61 共享层 | 无未解决项 | 算法独立与科学独立数据分开，未提出新通用算法优先权。 |
| 内部波与自由表面理论人员 | 小二阶残差是否证明全部原生误差由物理非线性生成？ | 正确性 | 二阶节首段先给归因 OPEN；定义段列出移动体积/运动学/滤波/FD/分裂排除项；表前定义残差 | 无未解决项 | 只支持规定二阶贡献相容。 |
| 内部波与自由表面理论人员 | 常规二阶自平流和 Galerkin skew 关系被称为新发现了吗？ | 新意 | L42 称预先固定的两个代数命题，定义核对等级，无首次发现/新理论宣称 | 无未解决项 | 没有借程序等级暗示理论首创。 |
| 代数/形式化证明读者 | VERIFIED-CLOSED 是否涵盖 C2 表压强迫及原生非线性能量？ | 正确性 | L42 在该词首次出现时仅限定 I/II、非 formal/refereed/native；spec v2 明确 C2 单独来源 | 无未解决项 | 两命题核对与参考实现/模型资格分开。 |
| 代数/形式化证明读者 | 多路线核对是否被当成新的数学定理或 formal certificate？ | 新意 | L42 原定义与快照等级范围 | 无未解决项 | 只报告有限工作流等级，不宣称新的证明系统结果。 |
| MOM6/Oceananigans 和既有 3c8 使用者 | 不同原生观测位置、有限配置和身份缺口是否被隐藏以宣称优越性？ | 正确性 | 开头与 L16 首次具体 baseline 比较立即给观测约定、有限111、MOM6身份缺口 | 无未解决项 | 只比较冻结面板的主组，未转移六组成绩。 |
| MOM6/Oceananigans 和既有 3c8 使用者 | incumbent 已有相似/更好的方法，候选凭什么称 SOTA？ | 新意 | 标题和开头承认不改善既有最佳并排除全球SOTA/全局最优 | 无未解决项 | 本报告没有需要借文献扫描支持的优先权或普适超越声明。 |
| 可复现性审查者 | 跳测CI、旧失败和 producer 自报边界是否冒充完整独立 PASS？ | 正确性 | L26 历史FAIL、L34 SKIPPED、L59 n64仅源审均在所属结果处明确 | 无未解决项 | 原失败与部分验证不被新的PASS覆盖。 |
| 可复现性审查者 | 已接触输入、REUSED 控制或保存数组回放是不是新盲验证？ | 新意 | L24/L28/L36/L61 分别给 contacted/REUSED/旧数据/共享层限定 | 无未解决项 | lineage 的最弱共享层和新旧证据分别声明。 |
| 一般技术读者 | 只读标题、开头和表格，会不会以为候选已获排名或非线性真值认证？ | 正确性 | 标题/开头直给主分数失败；二阶节首段、表前单位与末段共同给有限定义 | 无未解决项 | 冷读不会得到相反的采用/因果结论。 |
| 一般技术读者 | 具体新结果是什么，哪些仍未做？ | 新意 | 标题/开头、L30、二阶节，以及 L61 下一scalar诊断尚未纳入 | 无未解决项 | 有限新增控制与规定参考一致性清晰，未来工作未写成已通过。 |

## 最后冷读与准出范围

最后只读标题、开头、两张表的引导说明及末段：主分数仍 FAIL，索引 2..9 未跑，3c8 仍为该冻结开发面板的最近完整验收最佳。二阶代数身份、有限参考排除项、表格单位、n16 独立检查和 n64 源审范围均可直接定位；物理唯一归因仍 OPEN，没有 new holdout、global SOTA、新 score 或完整非线性真值声明。

逐段意义、语法、编号单位、必要性措辞、重复限定和 rendered comment/swallow 检查在最终哈希上零项未解决 finding。重复的范围限定各自约束不同结果，适用 prose-lint B15 的保留边界，不能为缩短文字删除。技术记录中的软件术语指真实接口、CI 与文件，不作为科学发现的装饰词。该审查准出仅对上述最终哈希有效；后续科学内容或关键限定修改须重新审查。

## 文档同步与原始代码文本存档补充

主报告准出后，根代理更新 `docs/material_point_transport_v4_zh.md` 的 L28 标题和 L32 最后一句。已阅读该两处实际 Git diff 及相邻限定：标题改为“主试点排名未改善”，正文绑定 ece37c7 的冻结、1737 pytest/MMS 软件 CI 与独立瞬时控制，随后立即给主综合分数 FAIL、2..9 未执行及当前有限已验收最佳仍为 3c8，并链接到主报告。内容与本审查已核证据一致，没有新增科学结论。该方法文档 SHA256 为 `c072f11a8ac8d5519667376e730a95ad8cbcadf19d7fffff4b80f2081db04c09`；这里只准出该两处同步 diff，不扩称整份方法文档的重新科学审计。

三份公开原始科学 evaluator 文件改以 `.py.txt` 存档，export manifest 仅更改 published 路径：`momentum_point_ece_reference_evaluator.py.txt`（9331 字节）、`momentum_point_ece_reference_numeric_audit_evaluator.py.txt`（6610 字节）、`momentum_point_ece_reference_raw_readback_evaluator.py.txt`（7709 字节）。原始 `.py` 来源路径及全部源哈希保留。改名后的 manifest SHA256 为 `82d56d8c75a81a73bfef759b592d0a9ad8764297f920f0c6a28864e7a16e13ac`。再次执行纯文件解析/渲染与 hash 检查：19 条主报告本地链接全部存在，14 份 manifest 副本仍与原始输出逐字节、哈希及长度完全相符；主报告没有直接引用这三份旧脚本路径。改名不改变计算内容或证据。新 manifest 收据记入 `final-frame-check-receipt.json`，改名前的最终正文收据另存 `final-frame-check-receipt-before-script-archive.json`。

最终主报告哈希仍为 `4cb36d7ba288a6fb91590ca2e68ce8bf95c6f1b60979b375cd7455ca05298429`。上述文档同步和文本存档路径改名在本次限定审查内亦准出，零项未解决 finding。根代理报告后续全库 ruff PASS；本审查未重新运行 ruff，也不把该软件结果作为任何科学归因或原生资格证据。
