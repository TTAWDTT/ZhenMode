# 中文候选稿独立冷读：恢复附录之前

Lady / neutral_paired_report_frame 未参与 producer、科学核心、外层 observer binding 或作者报告的编写。本轮继续使用已完整读取的 BootLoops prose-lint 和 referee-sim，独立完整读中文正文，分开执行诚实/数字、两轮prose、受众两轴反驳与最后的冷摘要读法。本审阅文件不是作者报告，不修改任何科学或作者文件。

初始中文版本 SHA256 为 `0993aad8a85747e8104cd2437fbcf7069d30937c0033c390774c6c2169b63001`，实际 `report-render02.html` 有5个MathML元素，其中4个display。本审提出两组LOW后，由root更正权重/任意场符号并删除两处结构标语。更正后独立保存的候选稿为 `zh-candidate-pre-restore01.md`，SHA256 `006d7eaace99beea221973851347466f2f9f0de9b5e50d2af1229bb7e2f1b8f4`，对应 `report-render03.html`。新增任意场集合的inline math使后者变为6个MathML元素、4个display，数量变化合理。

结论限于恢复附录之前：已核对的185个表格/日志/资源/原ratio值均与现存原JSON/JSONL吻合，18个public records链接均存在且原字节SHA匹配；英文稿四个MEDIUM的范围/定义问题在中文稿已回答。修正后第二轮prose另有一项LOW句法对象混用，已告知root。不能把本记录称为最终中文零finding或独立恢复验收。

## 记录与实际渲染

`ZH_PRE_RESTORE_NUMBER_LINK_RENDER01.json` 保存185项检查的打印值、原记录十进制值、原JSON/JSONL文件SHA，以及18项实际链接目标和原字节SHA。包括五case的6场RMS、4场normalized-RMS，三主L1，4个原summary ratio，库存漂移，case4最大绝对null变化，最小h/T/S及cap输运，保存旧基线RMS/L1，实际资源、原预测、observer/adapter、summary及cap代理值。比较使用标准库Decimal，仅核字面记录；未从数组算误差，也未从L1算ratio。

五case原science记录PASS和最终stage的 `PASS_FIXED_CONTACTED_DEVELOPMENT_CASES`、indices `[0,1,2,3,4]` 与现存stage/three-grid JSON一致。全文不再把仍pending的英文draft03状态带入此中文阶段；原stage的四项L1 ratio已保存为15.196520189951752、15.361537985783285、15.195729801902853、15.667690874637437，报告只抄录，未拟合阶数。case3是zero-spice有流，case4为zero-shear-null，表中的null最大绝对变化与RMS明确分开。

记录审阅数量473、1220、1223、2506与四份原terminal JSON的checks数组吻合，对应四SHA与报告相符。这是metadata/check记录数量，正文238行没有将其当作科学实验次数或全部独立断言。源码、wheel、软件CI与176解析复用按原范围沿用；没有重新执行它们。manifest实际SHA为 `476ddf2848d6a6ab341d6b7f40223f287a3ad19f3a1ba4533c9fe96bcca48796`，其已有记录支持841选定文件、673309247原bytes、33分卷及673508371拼接ZIP bytes；整体SHA79747029…的叙述是打包记录，不是本审重新哈希分卷或解压所得。

读取 actual Pandoc HTML02和HTML03。display势、容量权重、MSE/RMS、L1式的MathML语义均与原观察器定义一致；HTML03中有mfrac/msqrt和正确的W、chi。检查了实际body文本、表格顺序、上下文、标点及lowercase-after-period，没有发现渲染吞句或丢值。纯文本提取会把分数/根号展平，因此数学结论依据MathML结构和TeX annotation核对，未将展平文字当公式视觉结果。没有浏览器截图/像素审核。

首个失败的render01原记录应继续保留为FAILED display-marker adapter。此处只确认修正后的HTML02/03，不把先前$标记问题称作科学数组或指标失败。原raw packet与科学记录没有因渲染修复改写。

## Findings沿革

| ID | 规则、严重度 | 精确问题 | 当前答案或最小修正 | 状态 |
| --- | --- | --- | --- | --- |
| 英文F1 | MEDIUM；scope prominence / abstract firewall | contacted DEVELOPMENT、static-H/eta-cap、连续非线性动量加速度0后置 | 中文3行直接写已接触、静态H、eta顶盖差异、解析非线性加速度0及一般流/holdout/全四阶/ranking边界；13/21/26–30行给具体定义 | 中文已解决；原英文finding保留 |
| 英文F2 | MEDIUM；headline units与normalization convention | 首T_RMS无单位，时空平均/L1分母未就地说明 | 中文5行°C；9行及25–47行给空间MSE、eta面积权、梯形时间平均再sqrt；L1各时刻除加权abs(Q)，S另乘alpha/beta，避免固定振幅误定义 | 核心中文定义已解决；本轮另有下列句法LOW |
| 英文F3 | MEDIUM；baseline identity | source70表未定义实际实现 | 中文5与144行给旧70全revision与centered_second、新be全revision与paired_product_fourth；164–171行精确JSON入口 | 已解决 |
| 英文F4 | MEDIUM；independence prominence | producer-relative独立与自有wrapper限制在末尾 | 中文7行明示Lady相对于producer独立、Lady拥有binding、共享物理族且未fresh自审 | 已解决 |
| 中文F5 | LOW；prose-lint N11 | 初始版13行f为Coriolis，30–35行复用f为任意场；21行w是垂直速度，23–45行又作容量权重 | root将任意场改为chi、M_chi/RMS_chi，权重统一W；实际f和物理w保留。新稿与MathML03已确认 | 006d版已解决 |
| 中文F6 | LOW；prose-lint B8/F5 | 初始版9行`误差定义先于结果解释。`与256行`可使用的最终科学表述是：`只是结构宣告 | root删除两个前缀；006d版及实际render03中正文直接给定义/结论 | 006d版已解决 |
| 中文F7 | LOW；prose-lint F1/F4、N11对象一致性 | 006d版9行：`L1 分母不是固定的 0.08 °C 振幅，也不是先把各时刻 RMS 做算术平均。` 后半句沿用“L1分母”主体，却接RMS运算方法，混用两个对象 | 最小删除`，也不是先把各时刻 RMS 做算术平均`。同段前两句已准确区分RMS先平均MSE再sqrt与L1的spice分母，不需重述 | 已告知root；本记录截止尚未核修后SHA |

prose第一轮读源和实际HTML02，对象/术语/单位/语气检查产生F5/F6；root修后第二轮连续读006d和实际HTML03，产生F7。这是明确的多轮审，不是把初始0数值错当0prose finding。新文字须再由本非作者上下文核对，最终附录加入后继续读exactSHA全文。

## 受众反驳表

列6组必需受众，每组correctness与novelty分开。每个无新finding的行都列出强标准反驳和读者会遇到的明确答案；结论不是泛化认证。

| 受众与借用依据 | 轴 | Strongest standard objection | 读者路径上的答案 | 本轮处置 |
| --- | --- | --- | --- | --- |
| 数值PDE/收敛读者；借网格细化、加权误差 | correctness | “终态RMS还是时间平均MSE开根？spice分母是振幅还是每时刻空间abs量？如果平均约定不同，误差数值和汇总比不可比。” | 9行及25–47行先于结果表给完整约定；原JSON185项相符；null最大绝对变化121行另定义 | 科学定义已答；F7句法待核修 |
| 同上 | novelty | “T/S三个网格和约15的L1比不能证明完整模型四阶；时间/其他分量未分离。你们到底建立哪个命题？” | 标题为固定案例结果；3/5行排除完整四阶；107行ratio非RMS比或拟合阶；256行只称原门通过 | 无追加finding；无需新拟合/数字 |
| 海洋地转/EOS/静水动力学；借用thermal-wind与补偿tracer | correctness | “连续非线性加速度为0的地转态不验证一般非线性动量；未冻结原生速度也不让真解变一般。压力是不是异常势而非总静水压？” | 3行范围在标题后；21行速度/加速度区别及异常势*rho0；37行rho总密度接口误差 | 已答，无新finding |
| 同上 | novelty | “解析地转态和compensated tracer是标准验证对象；较已有方法的新内容在哪里，而不是新的动力学普遍性？” | 5行给保存旧centered_second及新paired_product_fourth的具体比较，仍非外holdout/新物理族；7行说明共享问题族 | 已答，推进限保存案例 |
| 守恒/自由表面读者；借nodal容量、库存门 | correctness | “observer将eta加到cap不等于源移动全柱输运；零离散漂移能否遮盖未表示的cap transport/stock？” | 3行先说静态H/cap差异；25–30行分源与观察容量；64/111–140行给库存定义及cap描述性值 | 已答，无新阈值/全柱资格 |
| 同上 | novelty | “固定H离散inventory是标准一致性控制，不是新moving-free-surface守恒方法；你们是否把库存近零包装为新能力？” | 64行不称移动全柱；140行不因小漂移消除差异；3行无general surface资格 | 已答，无新finding |
| 示踪scheme incumbent/旧centered_second维护者；比较基线 | correctness | “同一problem、同一点值/尺度/步长才可比；新T/S下降是否掩盖动量/表面/密度误差增大？” | 5与144行旧新来源和方法就地；5/152–160行给四类误差上升及原阈值；9/13/47行定义采样/尺度 | 已答，表值已逐项检查 |
| 同上 | novelty | “自有保存二阶实现不是整个领域现有最佳scheme；paired名称不能证明一般高阶优势。比较对象到底是谁？” | 5与142–173行精确旧实现+JSON；3/144行无ranking/SOTA、medium/fine描述性；软件默认不变 | 已答，不要求外部新benchmark |
| 软件/复现审阅者；借CI、loaded路径、records-only独立审 | correctness | “1799软件test/133文件/35路径只能说明身份；Lady拥有outerbinding则哪些层独立？失败intfixture是否改成PASS？176是否复制冒充重算？” | 7行相对producer独立且自身binding未fresh；205–211行真实失败、dtype修正、176 REUSED零observer调用；227–238行records边界 | 已答，无新source/CI资格 |
| 同上 | novelty | “重复metadata检查不是新科学实验；打包成功也不等于别人可恢复。你们是否混同这些结果？” | 238行checks不是实验次数；242行明确独立恢复待单独记录；250行快照边界与晚写附加物 | 已答；restore结论不由本审填写 |
| 实践者/一般科学读者；可能据报告采用可选方案 | correctness | “五cases通过究竟覆盖generic ocean flow还是已挑选剖面？全case状态是否与science/summary凭据一致？” | 第3行完整scope限定；7行共享族、无另造参考；stage原JSON绑定indices0–4与原summary；控制有流/null区分68行 | 已答；原英文pending不是新中文状态 |
| 同上 | novelty | “为什么可以据此更改默认或宣称领先？三网格比较和两个控制不提供独立泛化与外部排名。” | 3/5行排除holdout/SOTA/ranking、默认未改；256行只称固定范围原门通过 | 已答，条件性结果未扩大 |

## 最后冷摘要与限制

最后只读标题、3/5/7/9行：读者即遇五固定case、contacted development、static-H/eta-cap、连续解析非线性加速度0、无generic/holdout/四阶/ranking；随后可见粗T_RMS单位/旧新方案、其他误差增加、默认未变、producer-relative独立与自有binding未fresh。范围已在会形成印象的位置，英文四项MEDIUM在这里没有残留。第9行F7对象混用仍为可核的LOW修文项；不因此否认前两句和正文公式正确。

仅对报告/原JSON与JSONL/已生成HTML及小public records做身份和文本核对。未decode NPZ、导入科学code、启动模型/oracle/observer，未计算任何物理指标/ratio/CFL/新预测，也未重复source/CI qualification或独立archive restore。pack/restore属于其他审阅者；本版公开归档仍按报告文字记为“独立恢复待记录”。待最终restore附录与确切稿SHA到达后，再核末段变更和完整冷摘要，不把006d版的结论外推到最终文件。
