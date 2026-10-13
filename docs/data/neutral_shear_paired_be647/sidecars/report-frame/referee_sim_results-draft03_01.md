# Draft03 独立事实与受众审阅

审阅者：Lady / neutral_paired_report_frame。作者上下文隔离：本审阅者未编写 producer、科学 core、外层 wrapper 或报告。完整读取 BootLoops prose-lint 与 referee-sim 后，分别作数值/诚实核对、逐句读稿、实际渲染文本检查和受众两轴反驳审。作者英文稿未修改，也未自行翻译。

唯一审阅对象为 `results-draft03.md`，SHA256 `ad31d12bcd682764a6f6fd3463cdb1b95c34a31462e76efc89a6c618a53e34e2`。这是三个主网格已完成、控制与 stage summary 未纳入的固定版本；后来的 worker 进展不改变此版本。结论：未发现两张历史比较表的指标抄写错误；发现四项 MEDIUM 的范围、定义或显著性问题。不是英文零 finding，也不是中文最终稿审阅结论。

## 数字、单位与来源

`draft03-number-check01.json` 保存六份原 science JSON 的真实 SHA、48 项字面十进制比较和来源 token：主比较表 36 项 RMS 加第二表 12 项 normalized-L1，全部匹配。只比较记录中的字符串数值，没有从数组算任何指标，也没有算误差比、阶数或物理量。

| 原保存记录 | SHA256 | 所用对象 |
| --- | --- | --- |
| independent-neutral-shear-native/00-main/science-verdict01.json | 147163432215ed57ab88037b1974f09c65dbf859a2f2d43ec3cb436c737da837 | 历史粗网格 |
| independent-neutral-shear-native/01-main/science-verdict01.json | 17949d807a332500e848e2040bb38bc906954a78cb32878f4966e8bb58ebe89d | 历史中网格 |
| independent-neutral-shear-native/02-main/science-verdict01.json | 1c5e1a10d2b2ee18f49e8a170fd1ca6b68448211fa7ee70300e750399fd30cbe | 历史细网格 |
| neutral-shear-paired-be647-attempt02/00-main/science-verdict01.json | abec6f2aa16308cd012e1a64686241ae824670fe0bef5cad2a1db3d7d15027bb | 新粗网格 |
| neutral-shear-paired-be647-attempt02/01-main/science-verdict01.json | bb4d6ffcf6347a42486e798b9960702827c10fcc752edf1c5223adeef7b782d8 | 新中网格 |
| neutral-shear-paired-be647-attempt02/02-main/science-verdict01.json | e28264427788a7bcfb5515e882cbe2941e36c29b27c3be7573b4b9bdf6deacdb | 新细网格 |

报告表中的 T、S、u/v、eta、rho 单位分别为 °C、psu、m/s、m、kg/m³，与读取的观察器定义一致。rho 是源密度接口核对后的总密度误差；Phi 对应的源诊断是 anomaly potential 乘 rho0，不能解释成新验证的总静水压。

读取新旧 `review/audit_arrays.py` 与 `review/oracle.py` 的指标定义，而未导入执行。每个保存时刻，T/S/u/v/rho 的空间权重为实际 `h*area`，eta 为 `area`。RMS 为各输出空间 MSE 的梯形时间平均再开根号。T normalized-L1 的分母是各时刻的 `sum(weight*abs(truth['Q']))`；S 分母是该值乘 `alpha_T/beta_S`，之后作梯形时间平均。它们是无量纲值。**L1 不是直接除以固定 0.08 °C 或其对应盐度振幅；振幅与 RMS 阈值来源是另一约定。** 不得在中文改写中混用。

读取两阶段声明和 case 0 `actual-params.json`：历史源为 `70d408583aa99db63e49d958d89dcaccad273f0e`，历史实际 transport 为 `centered_second`；新源为 `be647d70fc7394f3014dca8bc776e207d4f02ae3`，新实际 transport 为 `paired_product_fourth`。表格基线是该保存历史实现，不能升级成未做的全球竞争方案比较。

已读的原记录支持：三个主 case science PASS；T/S 的保存比较值下降，u/v/eta/rho 三网格均升高；case 2 本身 finest-L1 PASS 与未纳入本稿的相邻网格 summary 是不同对象。失败 predecessor 的整数 constant fixture 被 float64 guard 拒绝，attempt02 仅修两处 fixture dtype；失败记录没有被 PASS 替代。176 analytic checks 属于旧阶段，bridge 的 `REUSED_FROM_PRIOR_ANALYTIC_CHECK`、`reuse=true`、`new_observer_calls=0` 不构成新科学调用。既有 CI terminal 记录含 be647、1799 pytest、MMS/Ruff，只是复用的软件资格；本审没有重复 source/CI qualification。

## 具体 findings 与最小替换

没有对作者文件应用以下替换；均为供 root 下一稿使用的建议。行号以固定英文稿为准。

| ID | 严重度、规则 | 精确原句/位置 | 问题 | 建议替换或插入 |
| --- | --- | --- | --- | --- |
| F1 | MEDIUM；referee-sim unread answer / abstract firewall，prose-lint B16 | 第3行：`Evidence remains contacted DEVELOPMENT.` | 内部标签没有解释何种数据已被接触；static native capacity、eta-cap discrepancy 与连续非线性动量加速度0到30/32/154行才出现。摘要只读者无法知道最重要的物理限制。 | 将该句替换为：`These are previously examined development profiles. The evidence covers fixed physical-z tracer points and steady geostrophic balance with static native capacity and a separately reported eta-cap transport/stock discrepancy. The continuous nonlinear momentum acceleration is zero; holdout and general moving-surface qualification remain untested.` 下一中文稿在首段用自然中文表述同一边界。 |
| F2 | MEDIUM；headline-number units，prose-lint N11/N12 | 第5行：`The new T_RMS is \`0.00004407068051203133\`, below the old saved value \`0.00037185579607669175\`.`；第52/107/130行表头只有 RMS / normalized L1 | 首个主数值缺 °C；表格没有时空平均与 L1 分母，读者可误认为终态 RMS 或固定振幅归一化。body的“actual area/nodal capacity”没有补齐这些定义。 | 数值句替换为：`The recorded whole-trajectory T_RMS is \`0.00004407068051203133\` degrees C, below the saved historical value \`0.00037185579607669175\` degrees C.` 在第一比较表之前插入：`At each stored time, T, S, u, v and rho errors use area times the observer's nodal capacity as weights; eta uses area alone. RMS is the square root of the trapezoidal time average of these spatial mean-square errors. The dimensionless T/S normalized L1 values use the weighted absolute analytic spice as the denominator at each time, with alpha_T/beta_S included for S, followed by a trapezoidal time average.` |
| F3 | MEDIUM；incumbent novelty / baseline identity，prose-lint N13 | 第105行首句：`The following table directly transcribes the three new main-case receipts and the three saved historical source70 JSON receipts.`；第107行 `Saved source70` | source70 仅是内名；表前没有旧源全identity、实际 centered_second、旧JSON入口。读者无法就地判断改进针对哪个已有实现。 | 将首句替换为：`The historical baseline is source \`70d408583aa99db63e49d958d89dcaccad273f0e\` with recorded \`centered_second\` transport; the new receipts use source \`be647d70fc7394f3014dca8bc776e207d4f02ae3\` with \`paired_product_fourth\`. The table transcribes the three saved science JSON receipts from each stage.` 表头就地标 transport；附近链接六份JSON或记录manifest中的精确入口。保留medium/fine仅描述性、coarse原严格门及无新ranking限制。 |
| F4 | MEDIUM；independence prominence，prose-lint E3/B16 | 第7行首句：`The independent terminal review is [PILOT_TERMINAL_REVIEW01.json](review/PILOT_TERMINAL_REVIEW01.json), SHA256 \`c0d5951bc49358540c1ac956f7cca66bf9570badaedcb76e2896eb1554a5af86\`, with [readable scope and limits](review/PILOT_TERMINAL_REVIEW01.md).` | 第7行说明records-only，但“相对producer独立”与“Lady拥有outer bindings，未fresh自审wrapper”直到156行才解释。不能让早期independent被读成所有科学/执行层均接受新的独立审计。 | 首句用：`A reviewer independent of the producer checked the terminal record identities in [PILOT_TERMINAL_REVIEW01.json](review/PILOT_TERMINAL_REVIEW01.json), SHA256 \`c0d5951bc49358540c1ac956f7cca66bf9570badaedcb76e2896eb1554a5af86\`.` 同段插入：`The reviewer owns the outer observer bindings, so those bindings have not received a fresh independent implementation audit.` 保留同段原有未解码、未执行observer说明及readable scope链接。 |

这些新增措辞只解释原证据范围，不引入新数值、物理结果或待做数值任务。F2 是重要防错：中文不能把 normalized-L1 分母替换成振幅。

## 受众与 strongest standard objections

受众由借用的方法、比较对象和可能据结果行动者列出。每组有正确性与新意两轴；表中的“已有答案”有具体位置，“无追加finding”不表示泛化被证实。

| 受众及为何必须列入 | 轴 | 专业读者最先提出的强标准反驳 | 答案位置或缺口 | 严重度 / 建议状态 |
| --- | --- | --- | --- | --- |
| 数值PDE/收敛读者；使用网格细化、RMS和L1 | correctness | “RMS究竟是终态、输出均值还是先均方再时间积分？L1除以哪个量？同一输出采样和物理单位吗？没有定义便不能比较误差或称收敛。” | 32行只给空间权重；34/172–180行明确summary pending，避免伪convergence，但RMS时间平均/L1分母缺正文。定义可由现存code核实。 | MEDIUM；F2待root |
| 同上 | novelty | “三网格T/S变小不能证明完整模型四阶，时间误差和其余场误差也未分离。你们的推进到底是存储基线比较还是普遍阶数结论？” | 105行medium/fine描述性；128行其余误差上升；186行排除完整四阶、三维一致性和ranking。该强反驳在表后与结论已有明确答案，但开头物理范围需F1。 | F1 MEDIUM prominence；不追加阶数/新数字 |
| 海洋动力学/地转平衡读者；借用thermal-wind、hydrostatic、EOS | correctness | “非线性方程在一个连续非线性加速度为0的特殊地转态上跑完，不能检验一般accelerating flow；freeze_adv_vel=false并不改变真解的特殊性。” | 30与154行正确区分原生速度演化和连续解析加速度0；开头缺限制。Phi与pressure诊断在154行范围正确。 | MEDIUM；F1待root |
| 同上 | novelty | “静态地转精确态和compensated tracer是标准控制。这里新增的是哪个具体已知实现的误差下降，而非一种新的海洋动力学普遍结果？” | 61/105行仅coarse原严格比较；128行具体方向；186行范围限制。source70实际centered_second需就地命名。 | MEDIUM；F3待root |
| 守恒/自由表面读者；使用nodal inventory与eta容量 | correctness | “把eta加入observer顶cap的inventory近零，并不能证明源静态容量覆盖移动全柱输运/库存；未表示cap项在哪里，真实物理柱与diagnostic是否混淆？” | 32行区分源static与observereta；65/94/103行保存cap discrepancy；154/188行无moving-surface/full-column推论。开头未告知。 | MEDIUM；F1待root；无新cap阈值 |
| 同上 | novelty | “这类固定H离散库存测试是标准守恒检查；不能把零volume drift当新增moving-free-surface守恒能力。相比既有方法改进什么？” | 65、154、186/188行明确cap描述性、无新general资格；实际推进仅历史point tracer误差比较。守恒强反驳已被这些段落回答。 | 无独立新finding；F1负责提前边界 |
| 示踪输运方案使用者/旧centered_second维护者；incumbent比较对象 | correctness | “新源与旧源的点/均值、权重、边界/步长和spice归一约定是否相同？改用强结构profile是否只改善T/S，同时损害其他场？” | 新旧声明geometry/physics/mesh、observer定义读取一致；128行诚实给其余RMS上升；方法定义与baseline未就地明示。 | MEDIUM；F2/F3待root |
| 同上 | novelty | “paired_product_fourth名字不等于新的普遍高阶算法；历史自有二阶实现不是现有最佳方案。到底只超过保存source70，还是宣称领域改进？” | 105行saved比较和仅coarse原门；186行无ranking，结论限定contacted profile。baseline应在表边写centered_second/source全SHA。 | MEDIUM；F3待root；不要求新benchmark |
| 软件与复现审阅者；借CI、wheel、哈希和independent observer | correctness | “相同源码身份和1799软件测试不能代替独立物理观测；terminal review只查records、Lady还拥有binding时，哪些层真正独立？失败fixture和REUSED检查有无冒充新PASS？” | 7行records-only；24行既有CI；38–44行失败与176 REUSED清晰；156行自身wrapper未fresh独立。后三点答案成立，independence重要边界出现过晚。 | MEDIUM；F4待root；不重做source/CI |
| 同上 | novelty | “增加了一批身份check并非新增科学实验；bridge资源不是再执行176检查。报告是否把复用当新科学成果？” | 42/44行明确原176执行一次、bridge metadata-only、new_observer_calls=0；84/101行明确records-only；读者在reuse章节可直接找到答案。 | 无追加finding；不得改称fresh analytic |
| 工程实践者/一般科学读者；可能据标题摘要采用选项 | correctness | “三个主case PASS是否等于五case和整个stage通过？结构剖面可否直接当holdout、非静态表面和实际海洋流验证？” | 第3行和159/172–180行明确control/summary pending，因此五case误读已防止；contacted/static/acceleration限制只在后文，仍可能被当应用资格。 | MEDIUM；F1待root；pending保持 |
| 同上 | novelty | “为什么值得改变默认设置？这些数值不是独立泛化、普遍改进或与MOM/OA的新增比较，是否有足够理由采取行动？” | 5/128行T/S改善且其他误差上升；186行默认未改、选项与legacy distinct且无ranking；足以回答本稿仅条件性记录，不能推出采取默认切换。 | 无追加finding；最终开头需F1/F3直陈范围 |

## 渲染和冷摘要读法

使用现存 Pandoc 对固定英文稿生成 `results-draft03-render01.html`（HTML5、MathML），然后从实际输出的 body 提取 `results-draft03-rendered-text01.txt`。不是从原Markdown再次导出plain text冒充渲染。读过输出的开头、两比较表、资源段、边界/结论和路径表，并在输出作lowercase-after-period、必要性、反向误差方向、四阶/holdout术语及断句检查：未见吞句或丢失表值。HTML没有浏览器视觉审或截图；本轮检查范围为实际生成的文本与标记。

最后冷读标题和开头前三段：three-main与pending状态明确，粗网格T/S降且其他误差升明确；原生/终态与records-only未混作新计算。开头仍会让摘要读者漏掉固定H/eta-cap与连续非线性加速度0，并不了解contacted DEVELOPMENT和independence的确切范围，故F1/F4成立。首个主T_RMS没有单位，故F2成立。标题没有宣称普遍四阶或全stage PASS。

只读了现存代码/JSON/文本及文本文件哈希；没有解码NPZ、读取物理数组、调用模型/oracle/observer、重算物理指标、预测、ratio、CFL或运行任何sourcecontrol/CI。没有改科学门、source或作者报告。下一中文全文及controls/summary/packet加入之后，仍需独立全文cold review和cold摘要审；本次英文48值相符与四finding不得替代该项。
