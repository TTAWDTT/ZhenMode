# 温盐中性剪切阶段：非作者文档与原字节包装准出

**最终判定：零残余 findings，接受本报告明确限定的五案例阶段陈述及选定原字节数据包。** 本准出不建立新物理族、独立holdout、一般非线性动量、真实上盖/全水柱守恒、全模型四阶或全球SOTA资格；原best70面板分数与旧Ft严格FAIL保持。

最终公共文件：`D:/CodexWorktrees/thermal-full-scan/ocean-solver/docs/neutral_shear_70_results_zh.md`，SHA256 `0aa4277f03e5e00e523b28c861b1c81489ebb83c467d614584dd0fb2ed1b8335`。审查者未撰写或修改该公共报告、生产代码、科学observer或阈值。此前单独准备并经root封门执行了去重payload比较器；这是独立包装位比较，不是物理oracle或科学gate复算。科学结果仅从已有终态、journal及汇总读取。

本次应用已完整读取的BootLoops prose-lint/referee-sim，分开检查数字/单位、逐句表达及读者形成的框架；最后单独冷读标题、摘要、表格、资格边界及末段。唯一实际执行是已授权的五案例重复payload比较、一次archive复原、只读SHA/ZIP/源文件核对及Pandoc文档渲染，没有native、JAX/model、科学真值/评分、CI或reference求解。

## 已闭合的文字与渲染意见

| 问题 | 规则及范围 | 作者修订/最后读回 |
|---|---|---|
| 摘要“完整原生步进器有限验证”可能被读成全水柱或一般动量资格 | referee-sim摘要/最先阅读位置；MEDIUM | 首段现在明确点值温盐输运与平衡、固定H诊断库存，并排除真实上盖/全水柱及一般非线性动量。闭合。 |
| B中0.15未给温度单位；constant-RHS及null门未按字段说明单位 | prose-lint N3；LOW | B=gα×0.15℃；T/S RHS分别℃/s、PSU/s；null逐η/h、u/v、T、S给m、m/s、℃、PSU。与冻结合同一致。闭合。 |
| 首次实际Pandoc2.12输出丢失公式Phi/eta/tfrac，且误作superscript | prose-lint F9/F12渲染完整性；blocking格式缺陷 | 首次HTML/plain/AST与[final-INITIAL_RENDER_DEFECT.json](final-INITIAL_RENDER_DEFECT.json)保留。root仅将公式block分隔符改成字面$$，公式内容不变；精确撤销两delimiter变更即恢复原报告SHA81ec847bca72e15e5d9f2a51a25eab6df513f933680f59f4c7ad3bdb111597ef。新HTML有一个真正Math块，公式源逐字符/归一空白核对一致；plain通过仅保留Math源的Lua过滤器保留完整TeX。闭合。 |

修订由作者完成，最后检查由不同的本审查者执行。归档中的results-draft.md保留包装时原身份，SHA `c182662b2ceb2b805daec1457b403efd042b29dfa8c17d15b562e8cd827da0f4`，公开报告明确说明其非最终文案；该差异不被错误当成archive篡改或数据更换。

## 受众、正确性和新颖性框架

| 受众/借用依据 | 最强标准反对意见 | 轴 | 阅读位置及结论 |
|---|---|---|---|
| 海洋连续动力学与实际模型使用者：完整native及地转/输运解析解 | 特殊u(y,z)、v=w=0且动量advection连续为零，不能代表一般非线性动量；正eta而节点止于0又遗漏上盖。 | 正确性 | 摘要即限定点值/平衡及固定H库存，资格边界给遗漏上盖量和非线性动量排除，方法说明连续初值未改为离散精确平衡。已回答。 |
| 同上 | 热风/neutral-spice族早已用于开发，新的profile不是新物理族或未知holdout；没有模型改进/三方胜出。 | 新颖性 | 摘要第二段明确已接触族、原成绩不重排、未运行MOM/OA，best70及FtFAIL保持。只补充一个锁定profile的有限证据。已回答。 |
| 数值分析与收敛社区：三网格L1及fourth梯度/压力选项 | 同时加密x/y/z、dt固定的四倍误差下降，不能分离每个算子阶数或证明完整时空四阶。 | 正确性 | 两表之间的第一次ratio陈述紧接“有限点值误差趋势接近二阶”，明确不能证明个别算子或完整四阶；不把floor1.25门说成一般阶数认证。已回答。 |
| 同上 | 这只是原70既有算法在一个制造解上的验证，并非新算法或新精度理论。 | 新颖性 | 标题绑定冻结70，摘要声明无生产源码修改，水平centered_second仍在且末段因果诊断未完成。没有首创、算法优先权或general-order claim。已回答。 |
| 数值守恒/自由表面社区：diagnostic h、库存及正海面 | h顶层加eta和漂移很小不等于原静态质量面表示真实上盖，也不能证明动态体积能量或物理全水柱守恒。 | 正确性 | 摘要先排除；observer方法区分实际诊断h与静态源容量；资格段给上盖transport与T库存proxy未修，库存措辞始终为diagnostic。已回答。 |
| 同上 | 五案例库存PASS不是对原守恒算法的数学推广或新物理机制发现。 | 新颖性 | 只陈述声明误差/库存/控制门通过，shared physics/coordinate与特殊连续解限定就地存在；后续因果分解明确不属于当前结论。已回答。 |
| 观测量/MOM6/Oceananigans incumbent读者：未来baseline与采样比较 | 物理点观测不能等同full-cell或layer means；需要x/z均值及y积分，并且没有本阶段baseline源构建比较。 | 正确性 | 摘要不运行baselines；资格段明确观测位置与未来所需积分，旧70/Ft报告链接存在并保留旧谱系限制。已回答。 |
| 同上 | 新stage没有对incumbents或旧面板产生排名优势，不能从小RMS偷换SOTA。 | 新颖性 | 第二摘要保留旧best70分数和FtFAIL，明确无globalSOTA或baseline全局最优。已回答。 |
| 源码/实验可重复性读者：133源码、35路径、callable/runtime与新producer | 包源码相等不能自动保证所有历史实际执行路径；新capture也不能填旧native空缺。producer是否替代了原step？ | 正确性 | 方法说明新profile通过factory/init._replace再原step；各case实际35paths/callable/runtime与before/after133记录独立封存。资格段明确新capture不追溯修旧缺口，L duration的source推导范围在原receipt/独立review保存。已回答。 |
| 同上 | 独立编码仍共享物理方程/常数，old70CI也不等于新增独立物理oracle或新CI执行。 | 新颖性 | 摘要明确observer不变holdout；资格段说明共享层、仅引用原同sourceCI1750/MMS、没有文档功能CI重跑。已回答。 |
| 软件/资源与证据审计者：初始化、timing、五scientific audits、Ruff/CI历史 | initialize、12步timing和进程exit0不能充当完整native或science；观察数不能冒充断言数；RSS sampling不是硬cgroup。 | 正确性 | 方法分开176控制、初始化四hash、科学观察220/220/222/177/183与断言212/212/214/171/177；表列41/4800，资源段标采样终止、初始化/timing/audit不计入原生表。独立终态逐项一致。已回答。 |
| 同上 | 更多检查和单机速度记录不是物理资格扩张或三方性能优势。 | 新颖性 | 资源段明确没有三方性能比较，实际全run唯一，当前有限阶段与来源局限不因检查数量改变。已回答。 |
| 数据归档/外部重放使用者：大fine NPZ、12parts、omissions、restore | 去掉outputs和派生NPZ会不会丢失fields/time？能否字节复原全部选入文件？另机绝对路径/解释器又如何复核？ | 正确性 | 数据段明确205×10+time位核对、native-extra static保留、426selected而非allfiles；具体restore要求校验每part/whole/member。最后段区分字节恢复、payload再生和历史provenance重放。一次实际restore及独立source/member核对PASS。已回答。 |
| 同上 | 归档/去重只是证据包装，不是新增scientific PASS，不能再生NPZ后冒用旧container SHA。 | 新颖性 | 数据段不保证重复容器字节同一，省略清单给原SHA/原因；另机及新container要求另记谱系。后包装准出与原科学证据分离。已回答。 |
| 一般科研/摘要读者 | 首屏高精度数字会不会把有限特殊场包装成完整海洋准确性，旧失败藏到body？ | 正确性 | 冷读标题+两摘要可同时遇到有限五case、点值、fixedH、非一般动量、已接触族、旧best/FtFAIL与非SOTA，核心排除不靠全文拼接。已回答。 |
| 同上 | 实际新增成果是否只是换名重用热风数据或还没完成的next-order诊断？ | 新颖性 | profile/初值由新producer预声明构造，原源不变；下一阶段明确未建立完整因果归因且被package排除，完整现阶段五case与待研究问题分开。已回答。 |

该表覆盖报告借用的PDE/analytic解、数值norm/收敛、守恒、baseline权威、source/CI、资源及数据重放，不以“有PASS词”代替强标准反对意见。最后摘要冷读未形成更大的物理或排名结论。

## 数字、单位和状态核对

所有RMS/L1、ratio、stock/null、wall/RSS和观察/断言数量均读取原five-case-summary01、各science-verdict/journal、three-grid-verdict01及原资源终态；没有重算科学门。两表打印值是各原值的有限显示精度，headline fine T/S与表和原case2一致。T/S单位℃/PSU、velocity m/s、eta/h m、density total kg/m³、P_prime anomaly Pa，主L1以spice signal归一且zero-spice无该分母。

main ratio T3.917484291191368/3.9699801977709477、S3.917335558324281/3.9698081197639508与独立summary实际PASS一致。五science有效审计0failed；初始化/source及旧pre-execution纠正/receipt-capture竞态是保留的准备过程，并未重标成执行science。source/frame均不将新native的35实际路径改造为旧70历史路径证书。

主网格upper-cap transport6.885974213414107e-8/6.930522393877726e-8/6.941674363056637e-8 m²/s和background proxy3.750000000000001e-6℃·m保留。这些是持续存在的模型/采样上盖缺口，不是随加密应趋零的main T/S截断误差，因此与点值误差减小不矛盾。观察者和归档源共同限制未被新数字掩盖。

constant-RHS只适用于孤立rest scalar source，null零剪切的流体tiny roundoff未被写成所有位模式为零。205快照payload比较是exact logical element bits，包含signed-zero bit；它不授予physical correctness，也不把per-output NPZ容器与native容器的文件字节说成相同。

## 实际归档复原与source133验收

按root明确授权只调用一次公开restore脚本，实际exit0恢复426原件到新目录final-restored。随后独立[final-verify-archive.py](final-verify-archive.py)只读核对原源、archive和已复原文件，结果[final-ARCHIVE_VERIFICATION.json](final-ARCHIVE_VERIFICATION.json)：

- Manifest SHA256 `214de8a81c41dcadd6bccbb00c16480d9594fbd7993116ff965c50db51086a3c`。
- 12parts依序串流的全体235311286 bytes，SHA256 `caea40c0a881d5be064ef6f861d0ed430e97fc0dc4da10e3338b6feaf14dc405`，与原selected-originals.zip一致；每part各自长度/SHA通过，单part≤20MiB。
- 426个member全部ZIP_STORED，archive name集合=manifest集合=restore实际文件集合；每member archive内容、原source文件、restore文件长度及SHA完全一致。
- 215个省略原文件SHA与原源匹配，205为原输出容器，10为adapter派生容器；它们未混入selected或restore集合。缓存、next-order及final-后包装材料未被伪称已包含。
- 实际原wheel archive SHA `7dbf38429a48355d2d10fe775c3bfd705d78497f19c9b832e4acddd6a5a22da2`保持；恢复installed133个Python文件完整registry与wheel内容、wheel-identity及build-input映射逐项一致，源revision70d一致。这里只复核源字节，不重跑任何source scientific gate或旧CI。

5case×41=205snapshot全部十个动态keys（Fz/S/T/eta/ice/pressure_anomaly_Pa/rho/u/v/w_native）和time已在root封门后唯一执行[verify_output_payloads.py](verify_output_payloads.py)逐shape/dtype/uint8 C元素bits匹配；7负例触发，实际13.109916738s、267RSS samples、peak308350976B、CPU1/AS4GiB。原结果SHA `fb55034adcd7178efbbe6057a67e2d6ab3937453379362da41bce94e86807200`与archive manifest绑定。该比较未再次执行。

native额外time/x/y/z/dz_node/area保留在complete archive，time实际int64未cast。复原包能取回这些原字节；严格历史路径/解释器身份、外机portable arithmetic和派生NPZ新container谱系仍按公开报告限定。

## 最后Pandoc、links及字节保护

实际Pandoc2.12新版本生成[final-report-VERIFIED.html](final-report-VERIFIED.html)及[final-report-VERIFIED.txt](final-report-VERIFIED.txt)。一个display Math的TeX source完整；23个普通正文段逐段在HTML中一致，共24p/两表。全部文字连续读回、首句/末句和摘要专项读回无吞句、来源单位丢失或新的范围扩大；小写孤片扫描无候选。plain的verbatim Math过滤器只是保留显示数学源，不作任何数学运算或修改报告。

[final-RENDER_VERIFICATION.json](final-RENDER_VERIFICATION.json)保存5个源链接/HTML锚点一致性：4个本地目标存在，1个旧CI URL的run ID与已存同源旧CI事实一致，本次没有实时重查CI。公开14个data文件（12parts+manifest+restore）和公共报告共15项Git text:unset，保护原字节。最终审查文件另附、不在已冻结426members中。

## 最终准出对象

批准的报告身份为本记录首行的0aa4277f完整SHA；批准的原资料集合为manifest214de8a8和archive caea40c0完整SHA以及其12个明确parts。初始坏render、旧报告81ec847、归档草稿c182662、先前准备记录均保留历史身份，不以最终状态覆盖。没有扩大检查到下一阶段因果数据；没有科学/native/CI重算、source或阈值变更，本审查者未commit/push。
