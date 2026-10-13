# 动量节点速度候选：局部控制通过，主分数未改善

生产候选 ece37c7b7ad383314e4dcfd982641d9ca541e1c3 新增显式 momentum_vertical_velocity_scheme='point_primitive_v1'。原 'dual_face_v1' 默认保留。主原生pilot通过原误差、库存和稳定screen，但综合分数0.005416191620723861高于最近冻结3c8的0.005388509039568577，差约0.514%。预声明严格改善门槛失败，check_pilot明确exit1，余批索引2..9全部未执行（包括细网格测时5），候选不推广为最佳配置。当前完整验收的最佳仍是3c8配置与其六组冻结三方对照；该范围不能证明全球SOTA或baseline全局最优。

## 主试点分项

| 误差 | 冻结3c8 | 动量候选ece | 单位/约定 |
|---|---:|---:|---|
| eta | 0.0005388509039568577 | 0.0005416191620723861 | 相对原eta振幅 |
| u | 0.00003198853242502769 | 0.00003187644386925723 | 相对原速度尺度 |
| v | 0.000004440723369690159 | 0.000003095817074544062 | 相对原速度尺度 |
| T | 7.334242326890638e-9 | 7.242707124017776e-9 | °C绝对RMS |
| S | 0 | 0 | psu绝对RMS |
| 分数 | 0.005388509039568577 | 0.005416191620723861 | 原五项误差/各自门槛之最大值 |

温度下降约1.248%，u下降约0.350%，v下降约30.286%；海面误差略升，仍决定综合分数。两源都是完整非线性原生轨迹，相对于同一线性解析内部波参照；温盐采用动态体积/时间权重，eta使用面积/时间权重，ZhenMode采用物理节点观测。候选仅主组在指定节点/原生层平均及速度位置观测约定下比有限111配置选择后的冻结MOM6/Oceananigans低；MOM6历史完整构建/动态库身份仍有缺口，不能称各baseline全局最优。未执行其它组，不能将3c8六组领先转移成ece六组实测。原0旧band回放17数组完全相同。

独立2条已保存native数据读回通过133个Python源文件身份、参数/输入输出哈希、时间轴、实际容量及h=actualcapacity+topeta、原误差/库存和负控制。候选对3c8全轨迹maxabs差为T5.4222457635e-9°C、S0psu、eta2.5538390108e-11m、u2.8605570593e-9m/s、v2.2414059397e-9m/s；这是描述性差分，没有另设等价门槛。未重复native索引。没有这版的热风、null或全部六组原生验收。

## 修改和独立控制

旧动量w来自上侧控制体面−Fz[..., :-1]，其最后一项在床面上方。新显式选项用同一三次O(N)原函数求物理节点向上z速度，表面−Fz0、床面0；tracer/连续性仍用原Fz face-flux owner，保持完整质量残差。选项独立于vertical_tracer_scheme，有效非material tracer组合也受测；只限定float64、fullywet uniform physicalendpoints≥6、原容量、integrated_reference、closed centered_fourth unsplit legacy/no projectAdv通道。旧默认合同digest不加字段，新选项非默认时显式序列化。它还改变同owner w诊断，追加params字段可能改变旧checkpoint指纹；旧默认数值回放不是旧restart元数据兼容认证。

制造流场16/32/64节点的旧w误差3.260596e-7/1.616128e-7/8.047698e-8m/s约随加密减半；外部同原函数点w误差约1.00356e-11m/s，床面严格0。相同真实momentum owner及水平滤波下，16节点adv_u误差3.24880e-10→1.53848e-11m/s²；后者存在约同量级水平滤波误差，不随垂向加密消失。诊断属于DEVELOPMENT，原数据已接触；不证明主eta因果或三维能量。

148项原控制/首批新控制通过，补充9项解析多项式/zero/非material/真实scalar残差隔离控制通过，合计157项本地控制及ruff通过。首补充控制曾1FAIL/7PASS：名义u=.01均匀流被要求逐位w0，但实际离散Nm最大3.2073109600282284e-21m/s，Fz、旧w、新w最大均3.339920932414013e-20m/s，该输入不满足实际离散散度为0。原文件和失败日志保持；truezero操作数u=v=0保留同一严格0断言，另加非零常量流实际−Fz0/床面0精确归属检查。没有改物理代码、epsilon裁剪或放宽阈值；名义非零均匀流逐位w0的旧断言仍为历史FAIL。

冻结后独立source/math控制通过完整133 Git/archive/build/wheel/installed/plan/loadedpaths前后身份，实际容量/默认/表底/多项式、已失败47输入作为DEVELOPMENT温标回归、常量/反质量负例和库存控制。column/vertical字节不变，transport函数AST只有_compute_vertical_velocity改变；旧3c8 scalar1D和主动2D资格按此范围明确REUSED，不称新blind或证明完整新耦合模型。

此前未测的瞬时RHS剖面在freeze后正式预声明36×19/72×37/144×73网格，并通过真实_compute_momentum_tendency入口计算（f=0、eta=0、T/S常量、源/扩散/阻力系数均0；入口在此退化为平流，与平流差≤5.17e-26m/s²）：u归一加权L1为0.003414027926/0.000841133793/0.000208710134、加密比4.05884/4.03015；v为0.015759486034/0.003959729229/0.000991177665、加密比3.97994/3.99497。以各分量解析RHS容量加权L1归一。常量u rate严格0，实际point分支与旧分支各网格不同，表面/床面/闭面/容量精确检查与反符号w负例通过；有物理源的组合未认证。零候选归一L1=1是同一非零解析范数的代数推论，另有实际运行的反符号负例，不能写成独立第二次零候选运行。此检查只认证新瞬时动量组件，不代替压力、完整时间、能量或native排名失败。

## 软件与下一证据阶段

首CI38292930386为lint-only：手动入口遗漏full_validation=true，test被默认SKIPPED，原日志和记录保留，不算完整pytest/MMS。已显式true派发同源完整CI38293318389，已核实终态SUCCESS：同源码实际1737项pytest通过、6710warnings、1589.16秒，MMS ALL PASS；[同源完整CI](https://github.com/TTAWDTT/ZhenMode/actions/runs/38293318389)及[终态记录](data/momentum_point_ece_ci_terminal.json)保存。

新的只读eta空间分解使用保存的3c8六组及ece主组，不新增native，不改原分数/阈值。主3c8eta平方误差的空间mode2占99.789986%，ece为99.792078%；完整离散cos基的Parseval差≤2.23e-16。先planted正控制及错误相位负控制通过，再按测时单次有限分解。双幅值关系来自3c8的三组main/half配对，ece只有main数据；二次模态与这些配对关系只支持“与二次响应相容”，数值非线性算子误差也可有相同特征。随后完成的有限二阶参考计算见下节；既有线性评分不变，不删除非线性追分数。

原3c8十条/新1D/主动2D/1721pytest与MMS完整记录、原47/362/mean/arith/原1D失败均保持。[3c8完整已验收报告](material_point_round3_3c8_results_zh.md)。本源完整输出与可恢复HANDOFF在D:/Github/ocean-solver/outputs/vertical-mean-round3-20261012/。PR69保持DRAFT，不merge。

## 有限二阶参考诊断

BootLoops prove-protocol核对了预先固定的两个代数命题：给定一阶内部波场的二阶自平流恒等式，以及截断Legendre Galerkin线性参考的能量加权反对称关系。其`VERIFIED-CLOSED`是这两个命题的程序级核对等级，包含不同推导路线、连续两次新上下文反驳检查及非作者重建；它不表示形式化证明、审稿认证或原生模型能量资格。最终代数记录保留了当时数值参考尚未完成的状态；下面的结果是随后完成的有限数值计算，物理归因仍OPEN。[代数命题](data/momentum_point_ece_reference_proof_target_spec.md)、[原判定快照](data/momentum_point_ece_reference_proof_verdict_snapshot.json)、[参考定义v2](data/momentum_point_ece_reference_spec_v2.md)均保留原身份。

参考采用冻结主组合同的参数与原41个保存时刻，固定H、线性连续性、零初始二阶状态，在16/32/64个Legendre基函数上演化规定的二阶平流及表面压力强迫，不拟合原生eta。该定义排除移动体积、非线性运动学eta·v项、原生滤波、有限差分壁面和过程/时间分裂。一次预声明全计算通过既定数值检查：n32/n64的eta、物理节点theta mode2相对RMS差分别为1.8758322128692746e-16、5.522953216595545e-16，均小于原1e-7门槛；最细截断的温度、压力及垂向速度边界相对残差也通过原1e-7门槛。正控制及错误相位负控制通过，计算未修改生产源码或重复原生积分。

下表使用n64参考与保存的主组非线性轨迹作描述性比较。eta采用原面积/时间权重；T采用原实际h×area/时间权重，T二阶参照同时包含theta20和空间mode2。残差定义为“原生减原线性参照误差，再减该有限二阶参照”；残差比例为残差RMS/原误差RMS。eta原误差在此以m列出，与前表按原海面振幅归一的eta指标具有不同单位，不替换前表分数。

| 指标 | 冻结3c8主组 | ece主组 | 单位/约定 |
|---|---:|---:|---|
| eta原误差RMS | 6.860584702972653e-10 | 6.895829831340842e-10 | m |
| eta二阶参考RMS | 7.051645033463736e-10 | 7.051645033463736e-10 | m |
| eta剩余残差RMS | 3.81414859376025e-11 | 3.541752548321291e-11 | m |
| eta残差/原误差 | 0.05559509515431827 | 0.051360788113192515 | 无量纲 |
| T原误差RMS | 7.334242326890638e-9 | 7.242707124017776e-9 | °C |
| T二阶参考RMS | 6.799621461113437e-9 | 6.799621461113437e-9 | °C |
| T剩余残差RMS | 2.808595478271924e-9 | 2.5598884918289927e-9 | °C |
| T残差/原误差 | 0.382942825324212 | 0.3534436016803804 | 无量纲 |

另一个非作者数值核对从定义重建实矩阵，使用SciPy增广矩阵指数，在n16及全部41个保存时刻检查，未复用参考实现的Hermitian特征分解/时间卷积代码。eta与物理节点theta mode2相对RMS差为1.2767860426403096e-14及6.930205540858173e-13，均通过预声明1e-7门槛；错误旋转、1%扰动及反号比较均被拒绝。独立原始数组读回还重现了上表所有标量及n32/n64收敛值。n64压力/垂向速度系数历史没有存入参考NPZ，因此这部分边界数值由原计算报告，独立核对只检查实现表达式；没有将源码核对说成独立重算。

代数与两种有限数值方法共享规定的物理合同、投影参考及binary64算术；保存的原生数据已接触，不是新的独立预测集。小残差支持该规定二阶贡献与数据相容，不能证明唯一物理原因、完整非线性真值或新排名。ece原线性综合分数仍高于3c8，严格改善仍FAIL；下一项scalar梯度诊断尚未纳入本报告，也未声明通过。[参考数值结果](data/momentum_point_ece_reference_result.json)、[独立数值审查](data/momentum_point_ece_reference_numeric_audit.md)、[独立比较结果](data/momentum_point_ece_reference_numeric_audit_result.json)、[原始数组读回](data/momentum_point_ece_reference_numeric_audit_raw_readback.json)提供具体记录。


[主试点分项](data/momentum_point_ece_primary_vs3c8.json)、[试点决策](data/momentum_point_ece_pilot_decision.json)、[独立原生读回](data/momentum_point_ece_native_pilot_audit.json)、[源码数学控制](data/momentum_point_ece_source_math.json)、[动量入口声明](data/momentum_point_ece_new_momentum_declaration.json)、[动量入口控制](data/momentum_point_ece_new_momentum_result.json)、[历史严格零断言](data/momentum_point_ece_strictzero_fixture_record.json)、[已跳测CI记录](data/momentum_point_ece_ci_attempt01_skipped.json)、[空间模态结果](data/momentum_point_ece_eta_modal_result.json)、[二阶参考导出来源及哈希](data/momentum_point_ece_reference_export_manifest.json)、[独立最终报告审查](data/momentum_point_ece_final_frame_review.md)。

