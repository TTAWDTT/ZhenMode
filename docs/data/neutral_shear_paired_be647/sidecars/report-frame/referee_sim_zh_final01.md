# 最终中文报告：独立全文与冷摘要审阅

审阅者为 Lady / neutral_paired_report_frame，未编写 producer、科学核心、外层 binding 或作者报告。完成 BootLoops prose-lint 两轮逐句/实际输出审阅、referee-sim 六组受众的 correctness/novelty 两轴反驳，之后对最终全文和摘要再冷读。结论只绑定以下确切文件：

| 对象 | SHA256 |
| --- | --- |
| docs/neutral_shear_paired_be647_results_zh.md | 7d0f3ee390ec7fc87d0272c62de4231fc5067524d7326cc3012b0b062089b3a9 |
| publication-preparation/report-render04.html | 1a8ca635567fdb59122d214e02f82e393aa351b8dc0619c52db489745e08ba0f |

本次检查未发现尚未解决的事实、单位、来源、范围或prose问题。状态为 `PASS_FINAL_EXACT_REPORT_FRAME_RECORD_REVIEW`；这表示上述读稿和记录核对通过，不增加科学资格。若报告或HTML字节改变，不能沿用该绑定作为新版本终审。

## 数字、链接与恢复附录

`FRAME_REVIEW_ZH_FINAL01.json` 保存最终绑定和限制。前轮 `ZH_PRE_RESTORE_NUMBER_LINK_RENDER01.json` 已逐项核对185个原JSON/JSONL结果表、资源、保存ratio和cap值，没有不匹配。最终稿与该已查候选稿只有第9、242、244行不同：第9行删除一句错误主体衔接的RMS说明，第242行加入实际恢复终态，第244行由计划提供改为已提供；科学数字正文和表格保持原字节内容。本终审没有重复科学数字计算，也没有重算ratio、阶数或预测。

最终18个records链接与前轮列表一致，实际公开树内目标均存在，逐文件SHA仍与原保存JSON相同。manifest、selection及恢复脚本实际存在且SHA与原恢复记录相符；清单的33个分卷文件均存在。本审没有自行重哈希所有分卷或再次恢复841文件，完整归档恢复证据来自以下原记录：

| 原来源 | SHA256 |
| --- | --- |
| INDEPENDENT_RESTORE_REVIEW01.json | c67f91869e3fd5cac2fc047b7aa4da857c6563610b12fc3c11d7a4a62c5e2205 |
| independent-restore01-resources.json | de6598c327c7c0a16842c2539c70c2dcad2ddf8a15916a4c8c95c23d0baf8cc8 |
| INDEPENDENT_RESTORE_VERDICT01.json | 10a09a0a64849524de545b42c8646264cbf6e95685c3853b2d3dee90404d1ffb |

附录的841选定文件、33分卷、133 Python文件、五case、205逐时输出、23桥接原件、失败前身覆盖以及before/after/restored相同均由这些元数据支持。父exit0、stop reason null、实际68.40982751099999 s及采样子进程组峰值46821376 bytes与原父资源完全一致。此处的“一次”由原reader_invocations=1支持；归档SHA、manifest及selection身份也一致。正文明确该检查证明字节保留，未把它当新增科学验证、整个工作目录快照或硬内存峰值资格。

恢复命令的两个参数与便携脚本实际argparse定义一致，脚本要求输出及joined ZIP是新路径；未执行该命令。晚写报告、审阅、恢复/打包本身与缓存的排除范围被保留，最终报告不会假装自身属于841原件。独立恢复完成后，全文不再保留“独立恢复仍待”的旧结论。

## Findings全部保留并核修

| 原finding | 严重度、规则 | 最终答案/修正位置 | 结论 |
| --- | --- | --- | --- |
| 英文F1：物理与contacted范围后置 | MEDIUM；referee-sim abstract firewall | 中文首段给已接触DEVELOPMENT、static H/eta-cap、连续解析非线性动量加速度0、无general moving surface/accelerating flow/全四阶/holdout/ranking | 已解决 |
| 英文F2：单位与RMS/L1约定缺失 | MEDIUM；headline units、N11/N12 | 首个T_RMS带°C；开头及公式给空间MSE、eta面积权、梯形时间平均再sqrt；L1是每时刻加权abs(Q)分母，S另乘alpha/beta | 已解决 |
| 英文F3：source70内名缺实际基线 | MEDIUM；incumbent identity、N13 | 开头与比较表附近给旧70完整revision/centered_second、新be完整revision/paired_product_fourth及原JSON链接；中/细比较仍描述性 | 已解决 |
| 英文F4：independence限定后置 | MEDIUM；E3/B16、prominence | 首段后直接写Lady相对producer独立、Lady自有binding没有fresh自审、共享既定物理族/参考 | 已解决 |
| 中文F5：f/w符号碰撞 | LOW；N11 | f保留Coriolis；任意场改chi，M_chi/RMS_chi；物理w保留，容量权重统一W。HTML04同步 | 已解决 |
| 中文F6：两处结构标语 | LOW；B8/F5 | 删除“误差定义先于结果解释。”与“可使用的最终科学表述是：”，正文直接陈述定义和结论 | 已解决 |
| 中文F7：L1分母主体接RMS平均方法 | LOW；F1/F4、对象一致性 | 第9行只保留“L1分母不是固定0.08°C振幅”，删除衔接错误的后半句；该段前文和RMS公式仍清楚 | 已解决；确切diff与HTML04已核 |

初始中文和修后版本的发现分别存于 `referee_sim_zh_pre_restore01.md`；英文发现仍在原draft03审阅中。没有覆盖早期发现或把它们改写为从未发生。首个render01的display-marker adapter失败继续保留；本审的最终metadata脚本首读曾因Windows默认GBK读取UTF-8 review JSON失败，已在 `FRAME_CHECK_ADAPTER_FAILURE01.json` 保留并修为明确UTF-8。该失败没有产出PASS、触碰作者/科学/packet输入或重复科学调用。

## 最强标准受众反驳及最终答案

每组由实际借用的方法、结果或行动者产生；没有把“我们知道作者想说什么”当作答案。无未解决finding的理由是强反驳在读者会看见的位置得到明确、有限的回答。

| 受众 | 轴 | Strongest standard objection | 最终答案位于实际阅读路径何处 | 当前结果 |
| --- | --- | --- | --- | --- |
| 数值PDE与收敛读者 | correctness | “终态RMS、平均RMS和时间平均MSE开根不同；L1固定振幅与每时刻weighted abs(Q)也不同。定义不清便无法比较。” | 开头第9行先于任何结果表，25–47行给公式；eta面积权、S分母换算和时间平均明确 | 已答；无残留finding |
| 同上 | novelty | “三网格约15的L1比不证明完整四阶；时间误差与其他场未分离。标题是否夸成普遍高阶资格？” | 标题仅固定案例结果；首段/5行无全模型四阶；107行原ratio非拟合阶；末句仅原门通过 | 已答，无新order/rate主张 |
| 地转、EOS和静水海洋动力学 | correctness | “连续非线性加速度0的地转态不检验一般动量；原生速度演化不改变真解特殊性。pressure是否混成总静水压？” | 首段先给加速度0；21行解释freeze_adv_vel=false与解析特殊性，说明异常势*rho0；37行density对象明确 | 已答；special profile仍special |
| 同上 | novelty | “标准解析地转态和compensated tracer并非新海洋动力学结果。增加什么，而非再演示已知平衡？” | 首段/5行把推进限具体旧centered_second保存比较；7行共享物理族而无新独立参考；历史表给原JSON | 已答；没有普遍动力学新意包装 |
| 守恒与自由表面读者 | correctness | “observer把eta加到cap不让源具有移动全柱输运；零diagnostic drift是否遮盖cap错误？” | 首段static-H/eta-cap；25–30行分源与观察；64/111–140行库存定义、cap描述值及无一般全柱资格 | 已答，cap没有被近零漂移抹除 |
| 同上 | novelty | “固定H离散inventory是标准一致性检查，不是新的moving-surface守恒方法。” | 首段无general surface资格；64和140行明确库存检查/描述诊断；最终结论保留容量差异 | 已答；未扩大为新一般守恒能力 |
| centered_second incumbent与输运scheme使用者 | correctness | “比较必须同problem、点/均值与尺度；新T/S下降不能掩盖其他场升高。” | 首段后5行先写四类RMS略高仍达原门；9/13/47行定义采样/尺度；144–173行同族及原JSON | 已答，方向和单位均可核 |
| 同上 | novelty | “自有历史二阶实现不是领域当前最佳scheme；paired名称不等于普遍高阶领先。” | 5/144行完整source/actual scheme；3/144行无SOTA/ranking、无外部baseline升级；默认未变 | 已答，无外部排名或泛化主张 |
| 软件与复现审阅者 | correctness | “CI、源码SHA和loaded paths仅证明身份；Lady自有binding仍非fresh实现审；失败fixture和REUSED176不能冒充新PASS。” | 7行independence限定在第一次出现；205–211行失败、dtype与176 REUSED/零observer调用；227–238行身份/科学分开 | 已答；原失败/复用范围保留 |
| 同上 | novelty | “metadata checks不是实验；打包成功不等于独立可恢复；恢复成功也不证科学正确。” | 238行counts限定；242行实际一次独立恢复原SHA与资源；同段明确byte preservation非科学验证，250行选定快照范围 | 已答，新附录也未混级 |
| 实践者和一般科学读者 | correctness | “五PASS究竟是选定开发剖面还是generic flow？全stage是否真的有原summary和controls，而非summary由表值推断？” | 首段固定/已接触范围；68行有流/真null区别；107行原summary与stage精确JSON、indices0–4；原status可核 | 已答，全stage在有限范围成立 |
| 同上 | novelty | “这些数据不支持默认切换、外holdout或新外部竞赛排名。报告是否诱导实际依赖未经覆盖的泛化？” | 首段无holdout/SOTA/general资格，5行默认未改，最后一句仅fixed development原门通过 | 已答；行动范围没有扩大 |

## 实际输出与最后冷摘要

HTML04保留4个display MathML、2个inline MathML，权重W与chi符号一致；原压力势、观察容量、MSE/RMS和L1式中的fraction/sqrt结构与原定义相符。实际body开头、所有表、变化段、归档/恢复说明及结尾已读，未见吞句、孤立lowercase尾、缺值或原来两个标语。数学公式通过实际MathML语义和TeX annotation核对；展平文本不用于冒充公式的视觉准确性。没有浏览器像素/截图检查。

最后只读标题与开头四段：五固定case通过、已接触开发资料、static-H/eta-cap、连续解析非线性加速度0首先可见；粗T_RMS的°C、旧新实际transport、其他误差升高、默认不变、producer-relative独立、共享物理族和准确RMS/L1定义也在前文。读者不需要拼接远处限制来纠正标题/摘要印象。新恢复附录没有把有限物理命题扩成一般资格。

这轮未编写/修改作者报告；所有修文由root完成并由非作者核对。仅查现存文本/JSON/JSONL、MathML及小公开记录的身份，未decode NPZ、导入科学code、启动model/oracle/observer、重算物理指标/ratio/CFL/预测、重跑source/CI qualification、打包或restore。本报告终审只覆盖明确列出的检查；七项原finding已核解决，不构成对范围之外命题的认证。
