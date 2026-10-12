# 可选四阶点值梯度：固定面板六组分数改善

冻结源码70d408583aa99db63e49d958d89dcaccad273f0e新增显式`material_point_gradient_scheme='fourth_uniform_v1'`，默认`minmod_v1`保持。该配置完成本阶段预声明的有限独立控制、十条原生检查及同源完整CI；六组内部波综合分数均低于旧3c8及两份冻结baseline，进入既有固定开发面板的当前最佳配置。比较沿用有限111配置选择，节点与原生层平均及速度位置观测约定不同，MOM6历史完整构建/动态库身份仍有缺口；结果不证明各baseline全局最优。主分数仅改善约0.0126331%，不是完整物理或全球SOTA资格。原生历史loaded Python路径未保存，实际动量选项的历史取值仍由冻结源码默认推导；这些来源限制随结果保留。

## 原生结果

主试点0/1通过原误差、库存和稳定检查，旧默认回放17数组精确一致；严格改善检查实际exit0。随后按独立控制、细网格测时和两尺寸计划完成余批，十条原生记录均通过原科学检查。分数为五项误差除以各自原门槛后的最大值，同组比较，越小越好。

| 指标 | 冻结3c8主组 | 70d主组 | 单位/约定 |
|---|---:|---:|---|
| 综合分数 | 0.005388509039568577 | 0.0053878283060362665 | 原归一误差最大值 |
| eta | 0.0005388509039568577 | 0.0005387828306036267 | 相对原海面振幅 |
| u | 0.00003198853242502769 | 0.000031921402933548914 | 相对原速度尺度 |
| v | 0.000004440723369690159 | 0.000004082506786096297 | 相对原速度尺度 |
| T | 7.334242326890638e-9 | 7.088858339175954e-9 | °C绝对RMS |
| S | 0 | 0 | psu绝对RMS |

主分数改善约0.0126331%，T下降约3.34573%，u下降约0.209855%，v下降约8.06663%；海面仍决定综合分数。参照仍是已用于开发的线性解析内部波，原生积分仍保留完整非线性方程。eta采用原面积/时间权重，温盐采用原实际动态体积/时间权重，没有以二阶参考取代评分、删除物理非线性或重挑阈值。

六组按原观测约定比较：ZhenMode为物理节点，MOM6/Oceananigans为各自原生层平均/速度位置。两份baseline是有限111配置选择后的冻结输出，本阶段复用原记录，未重跑；MOM6历史完整构建及动态库身份缺口仍在。下面仅为指定观测、原门槛和有限配置面板内的开发排名，不能证明各baseline全局最优或全球SOTA。primary为8×32×16、horizontal-coarse为8×16×16、vertical-fine为8×32×32；main/half幅值为0.01/0.005m。

| 网格/幅值 | 70d | 冻结3c8 | 冻结MOM6 | 冻结Oceananigans |
|---|---:|---:|---:|---:|
| primary/main | 0.0053878283060362665 | 0.005388509039568577 | 0.036699334480504484 | 0.12285950114586117 |
| primary/half | 0.0027004846904144834 | 0.0027012892996825487 | 0.03639019551928815 | 0.12288309188059326 |
| horizontal-coarse/main | 0.005106513974973408 | 0.0051097907013557325 | 0.08869339763321504 | 0.13654384226339566 |
| horizontal-coarse/half | 0.00259374876381348 | 0.002597085100116228 | 0.08856800026656829 | 0.13656578723702445 |
| vertical-fine/main | 0.00541372630872662 | 0.005413792384326902 | 0.030172678193749452 | 0.04403194602861599 |
| vertical-fine/half | 0.0027070690927476562 | 0.0027071644764075703 | 0.029792652883607818 | 0.04403889090009892 |

候选与默认两个零扰动案例的eta/u/v/T/S/h漂移、原误差及库存漂移均严格0。热风T、S绝对RMS为3.2750856066568785e-5°C、6.80128585725626e-6psu；T库存相对误差3.1254998589247407e-12，比旧3c8的3.1252778143198157e-12增加2.220446049250313e-16，仍小于原1e-8门槛。volume与S库存误差均为2.220446049250313e-16。热风通过原检查，但不能称每项指标都改善；它没有原生发布的综合分数。逐组五项误差、库存及原始输出哈希见[十条终态记录](data/material_point_gradient_70_native10_summary.json)，原值不被独立重算的末位差覆盖。

## 机制与修改范围

原点值目标和水平柱矩使用minmod梯度，在已制造的单调背景加正负扰动场中产生奇振幅响应偏差。首次外部诊断漏乘水平通量的层容量，向实际owner传入错误单位，独立审查拒绝；原脚本、数组、结果和失败记录保留。修正的attempt02使用实际`fx*h/fy*h`入口，在同16/32节点场一次有限评估；独立原始数组重组、25项精确Fraction导数矩及133文件来源核对通过限定范围审查。这些已接触数据帮助选择候选，不是新生产源码的独立资格点。

修正诊断的最终奇振幅超量RMS由1.4306452958623352e-13/6.918557374060449e-14°C/s降至约2e-23°C/s的浮点取消量级；全部16次调用alpha为1且feasible为true，八组low_rate数组在两个选择器间逐位相同。约1e-23残差的末位没有精度认证，32节点偶振幅误差还略增，不能据此宣称所有误差或全时间稳定性改善。首次读回的固定1e-25关联算术检查失败也保留；attempt02另行声明存储算术逐位重现及epsilon导出的关联差预算，只涉及事后读回，不改变科学门槛。[诊断单位失败](data/material_point_gradient_70_mechanism_attempt01_attempt01_owner_unit_failure.json)、[修正诊断结果](data/material_point_gradient_70_mechanism_attempt02_result.json)及[限定审查](data/material_point_gradient_70_mechanism_attempt02_review_VERDICT.json)分别保存。

生产修改只作用于material_point_v4的点值目标及水平柱矩两个直接梯度使用处。五点四阶导数以局部浓度差构造，常量导数严格0；旧limited MUSCL基准、实际Fz质量残差、物理端点/停滞面和共享Euler系数不变。支持范围沿用全湿、均匀、物理表底至少6节点、float64、受限闭通道及既有积分/连续性组合，未扩到部分湿或非均匀网格。动量路径沿用3c8的dual_face_v1，未使用ece排名失败的point-momentum组合。新增参数可能影响旧checkpoint指纹，默认数值回放不认证旧restart元数据兼容。

## 本地软件与完整CI

170项本地控制集合至少各有一次PASS，不能写成单轮170PASS。首轮168PASS/2FAIL为新测试harness错误（NumPy广播JAX tracer、测试模块互导）；原失败脚本与日志保留。改为JAX广播并将原408字符fixture函数原样移到tests/support后，受影响48项通过，耗时38.14秒。生产科学源码未因这两项失败改变，原门槛保持；全库ruff通过。冻结前独立静态/API/支持范围/基准隔离审查接受限定修改。

同源码完整CI38306749152明确`full_validation=true`，实际1750项pytest通过、6710条warnings、1159.33秒，MMS ALL PASS，ruff PASS。实际日志含70d完整提交身份及测试终态；首次日志下载网络超时仅作为读取错误保存，随后只读重试成功，不增加测试运行次数。[完整CI终态](data/material_point_gradient_70_full_ci_terminal.json)、[实际原日志](data/material_point_gradient_70_ci_raw_full_attempt02.log)与[同源CI页面](https://github.com/TTAWDTT/ZhenMode/actions/runs/38306749152)可核对。早期声明里的pending保留为当时状态，由后续终态记录补足。

## 独立科学控制

独立来源核对比较了146个Git/archive构建文件以及133个wheel/installed Python文件，逐字节一致；每个科学检查在实际隔离package加载前后核对来源。source/math在预先选择的新非仿射场和primary32×16、fine128×64几何上通过一次有限检查：常量RHS最大低于6e-19°C/s（原门槛1e-17），温标差低于6.06e-18°C/s（原门槛1e-13）；degree0..3原函数的内部/床面相对差≤4.48e-16，degree0..4梯度绝对差≤1.50e-15°C/m（原门槛1e-14）。实际顶容量/直接质量面和及基准选择器逐位核对通过；顶行仍按实际梯形容量保留，不把内部三次精度移称为顶行精度。六个dyadic区间正负控制、停滞面固定及库存检查通过。

新1D剖面在源码冻结后先声明37/73/145节点和解析特征线，再做独立测时与一次有限Heun演化。归一容量加权L1采用初始连续剖面范围作归一；加密比5.173123637454697/3.986240785337165超过原1.25门槛，最细误差小于原0.05。常量RHS、最终漂移和端点漂移均严格0，库存残差绝对值≤4.01e-16，低于原逐网格约6.18e-12预算。三个网格minAlpha均为0、allfeasible为true；这是带局部水平补偿的点柱核心检查，没有证明四阶修正全程生效或真实二维水平owner的时间资格。

| 新1D节点数 | 归一容量加权L1 | minAlpha | 范围 |
|---:|---:|---:|---|
| 37 | 3.461446310717418e-5 | 0 | 核心演化 |
| 73 | 6.6912112551412625e-6 | 0 | 核心演化 |
| 145 | 1.678576788375142e-6 | 0 | 核心演化 |

新实际ACTIVE2D控制在冻结后预声明不同quartic流函数/示踪量及30×31/60×61/120×121网格，并要求每个网格alpha>0.5。实际标量owner使用层积分水平通量及生产连续性Fz；以离散采样解析RHS的容量加权L1归一，瞬时误差和分支状态如下。加密比4.005915840671503/4.000409972412763通过原1.25门槛，最细误差小于原0.05；每网格alpha=1、feasible=true，明确验证修正分支生效。常量RHS≤1.26e-21°C/s、Euler库存残差≤1.74e-15，各自低于原1e-17及逐网格约8.59e-10门槛；反质量负例、端点/停滞面、闭面/容量及实际owner与核心数组同值检查通过。这是float64瞬时正控制，不覆盖全时间、压力或任意场。

| 新二维Ny×Nz | 归一容量加权rate L1 | alpha | feasible |
|---|---:|---:|---|
| 30×31 | 0.0005129695628111374 | 1 | true |
| 60×61 | 0.00012805300540840852 | 1 | true |
| 120×121 | 3.200997055088732e-5 | 1 | true |

随后保存数组读回重表达解析特征线与quartic多项式，重现误差及哈希；这是事后存储/解析一致性检查，不增加未用资格点。独立推导仍共享PDE约定、几何、生产primitive/projection/blend及JAX/NumPy，两个新控制的有限PASS不等于完整物理证明。profile、参数、幅值和门槛没有观察后调整；这些控制现已接触，不可给据其结果修改的后续源码重复认证。

## 十条数据读回与来源限制

独立原始数组读回不调用project score函数。保存的0/1先核对通过，随后追加此前未读的5/2/3/4/6/7/8/9，未重复原生积分或覆盖首两条核对。十条均通过来源/合同/输入输出哈希、时间轴、原始native与发布状态同值、原容量及动态诊断`h=dz_node+顶层eta`、原库存/误差和负控制检查。两条null的六字段严格零及热风原解析参照由独立算术重现，热风T库存重算与发布值相同。

独立主组综合分数0.005387828306036287与发布值存在末位差，保留两者，均严格低于同式重算3c8；细网格T重算6.837669991371231e-9°C及1.7202503736858847e-9°C也与发布值略异，在原核对合同内，不覆盖原值或另加等价阈值。逐组全轨迹差分只是描述量，不能当作截断误差界。原生生产元数据没有保存当时全部loaded Python路径；dual_face_v1由冻结factory默认和worker的合同关键字调用推导，不是单独capture的实际参数。独立控制记录的实际加载路径不能补造原生历史证据。

十条驱动墙钟合计337.83427460599995秒，最大采样RSS2313125888字节；细网格真实integration_s为10.288129616000049秒，驱动墙钟37.08547325699999秒。这里只报告同设备资源实测，没有同期三方速度比较。原生限制为1CPU/600秒/4GiB聚合RSS采样，独立控制为1CPU/180秒/4GiB采样；RSS监督及JAX allocator fraction不是全主机/全设备硬内存上限。

## 保留记录与后续问题

原1D、47/362、mean/arith失败、ece严格分数FAIL、原非零均匀流严格零断言FAIL及首次lint跳测保持原身份。首次owner单位诊断失败、首次关联算术读回失败、新测试harness失败、原生身份检查的WSL Git路径失败及后续修复均保留；早期pending快照只由后续声明和终态追加补足。原冻结21/111/32/MOM/OA记录未重跑，旧波、热风及制造场仍为development。

后续只读D1已完成108项有限stencil软件比较，但它不增加70d科学资格或未接触预测点。保存eta误差的因果归因和通用u墙面延拓选择仍未解决，D1也不能推出一般u应采用odd延拓；见[有限D1报告](D:/Github/ocean-solver/outputs/vertical-mean-round3-20261012/momentum-wall-static-70/diagnostic-D1/REPORT.md)。研究继续核对墙面延拓潜在问题，不因阶段通过扩大物理结论。原[3c8报告](material_point_round3_3c8_results_zh.md)及[ece报告](momentum_point_ece_results_zh.md)保留历史结论；PR69保持草稿，不merge。

[原生声明](data/material_point_gradient_70_declared_plan.json)、[六组冻结比较](data/material_point_gradient_70_six_group_comparison.json)、[独立科学审查](data/material_point_gradient_70_science_REVIEW.md)、[有限科学判定](data/material_point_gradient_70_science_FINAL_VERDICT.json)、[十条独立读回](data/material_point_gradient_70_native_audit_all10_report.md)、[十条读回终态](data/material_point_gradient_70_native_all10_consolidated_result.json)提供具体记录。完整输出根目录为D:/Github/ocean-solver/outputs/vertical-mean-round3-20261012/；本阶段目录为material-point-gradient-v1。


[来源及逐文件哈希](data/material_point_gradient_70_export_manifest.json)、[构建输入](data/material_point_gradient_70_build_input.json)、[wheel身份](data/material_point_gradient_70_wheel_identity.json)、[独立源身份](data/material_point_gradient_70_science_source_identity_result.json)、[源码数学声明](data/material_point_gradient_70_science_source_math_declaration.json)、[源码数学结果](data/material_point_gradient_70_science_source_math_result.json)、[新1D声明](data/material_point_gradient_70_science_new_1d_declaration.json)、[新1D结果](data/material_point_gradient_70_science_new_1d_result.json)、[新ACTIVE2D声明](data/material_point_gradient_70_science_new_2d_declaration.json)、[新ACTIVE2D结果](data/material_point_gradient_70_science_new_2d_result.json)、[科学保存数组读回](data/material_point_gradient_70_science_saved_readback.json)。

[本地控制集合终态](data/material_point_gradient_70_local_control_final.json)、[首轮harness失败](data/material_point_gradient_70_attempt01_control_failure.json)、[原生读回首次身份检查失败](data/material_point_gradient_70_native_attempt01_failure.json)、[原生读回追加声明](data/material_point_gradient_70_native_append_declaration.json)、[原生读回追加结果](data/material_point_gradient_70_native_append_audit_summary.json)、[冻结源码存档](data/material_point_gradient_70_science_frozen_source.tar)、[控制实现存档](data/material_point_gradient_70_science_check_new_2d.py.txt)与其它实现存档均保持原字节。

