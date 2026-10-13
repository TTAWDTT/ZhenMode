# 切向动量 Gram 滤波：有限验证与原生晋级失败

候选生产 dabd48abd898c20e0471a5ab9fc18e9b61cfa96d 仅为显式可选实验；当前冻结开发面板最佳仍为 70d408583aa99db63e49d958d89dcaccad273f0e。原生主 pilot 的分数 0.005418658568218491 高于 70d 的 0.0053878283060362665，严格晋级门失败（+0.572220576%）。综合分数为 η、u、v、T、S 五项误差各除以其原门槛后取最大值；库存、体积及null稳定屏障另作硬门。原科学屏障通过不替代该失败；索引 2..9（含细网格 5）全部未执行。原生晋级评分使用原 first-linear 参考、原阈值和旧冻结物理定义；这不是全球 SOTA 或完整物理资格。MOM/OA 不复算，其历史构建与采样限制沿用 70d 报告。

显式 momentum_tangent_filter_scheme 默认 binomial_v1，可选 gram_fourth_v1 只替换 adv_u 的末次滤波。原法向 binomial、GM、标量、Fz、pressure、dual_face_v1 w 与时间积分保持各自算子。新路径仅限预声明 float64 完整湿均匀 Cartesian 闭合通道和 70d 特定组合；旧默认17数组的形状、dtype及数值精确一致，未另验证 signed-zero 位模式，但新增参数字段改变重启元数据指纹，未声称旧 checkpoint 元数据兼容。

Ft=I−RᵀR/16（R为纬向无周期绕回的二阶差分矩阵，n≥4）的固定均匀矩阵推导与有限实例核对确认固定均匀权重下的常量、仿射、均值与倾向 L2 范数性质；不证明动态体积动能、完整非线性时间稳定性、正性、TVD 或整模型精度。e0 会出现 −1/16 与 TV 从 1 增至 17/16；纬向 Nyquist 不保证完全消去。Fn 理论备选未在本候选采用或比较。两 fresh skeptics 的论证与独立 spec-only 重建支持限定 algebra grade；第二 skeptic 的 child101 exact PASS/exit0 被数学接受，但末行 CR 外层 launcherFAIL 原记录完整保留，未称 clean 软件执行。

只读 D1 和外部 Ft-only 原型给出了修改机制证据。外部 Ny12/24 的 analytic-w 切向瞬时 RMS 从 3.441087017974906e−11/9.364309953801162e−12 降为 1.0510264130749754e−11/3.525966951984318e−12 m/s²。两网格法向逐位不变，固定权重 norm/sum 通过；一次非作者 147 项独立数组/矩阵/解析 truth/源码核对通过。actual dual-face w 的定位误差仍在，解析-w 的 v 输出未归档，所以其 reported v RMS 未计独立数据验收。瞬时改善不能代替全时段结果。

新 191 个唯一本地控制均至少一次通过：n4 测时 1 PASS/3.19s，其余 190 PASS/1 deselected/100.72s；没有一次 191-test run 的说法。全库 Ruff 通过，实际五生产文件、测试及方法页独立静态审查通过并封 SHA。完整133 package Python 文件的新 wheel SHA 3cabb4b324ff012611c440df1065fc2e230c0366dd0f9431c4a8c59e27a4523b；独立 postfreeze 强身份检查301项通过（144个捕获src路径/完整133 Python文件），其旧 scalar/gauge/mass 控制仅由 unchanged AST 桥接标为 REUSED。新三网格、正负振幅瞬时制造验证135项/六案例通过：两分量finest归一化L1为约0.00279619/0.00456101，逐相邻refinement约3.925–3.992，原 full-owner1e−17 实际0；各grid/sign的切向RMS严格改善、法向uint64逐位相等、固定norm/sum均通过。旧w定位误差未修、uz/vz严格0使w不进入此制造truth。新真正完整CI38318954281实际1771 pytest通过、6710 warnings、1571.69秒，MMS五检查ALL PASS与Ruff通过，非作者原log核对同dabd功能源。

首原生启动在导入前失败：root只向driver的sys.path加入包路径，子进程没有收到 PYTHONPATH，新 wave_zhenmode 无法导入。原 attempt01 failed0/log/resources/declaredplan 保留，无初始化数组。另 native-attempt02 只补显式 immutable-installed PYTHONPATH，十合同和133source SHA与原声明完全相同；没有覆盖失败或更换门槛。

第二次有效启动索引0/1各一次 completed：旧默认17数组 exact，原科学屏障通过。新 primary η/u/v均为按原幅值/速度尺度归一的误差，T/S为°C/psu绝对RMS。η误差 0.0005418658568218491、u3.249236765953524e−5、v4.09669695296176e−6、T7.090249656218243e−9°C、S0；对应70为0.0005387828306036267/3.1921402933548914e−5/4.082506786096297e−6/7.088858339175954e−9°C。体积与T/S库存相对误差均2.220446049250313e−16，原门槛通过。pilot checker actual exit1/allowfalse，禁止后续扩批、系数扫或失败候选组合调参。

来源限制：历史native worker的实际已加载Python文件路径没有捕获，momentum dual-face 默认仅由冻结factory及worker kwargs推导，不能追溯改成运行时实存。旧内部波、热风和本轮制造场现在都是 development；共享 PDE/JAX/helper、节点观察与层平均差异、线性参考及旧 baseline 构建谱系均记录，不能称独立全物理 holdout。源码、新瞬时science、完整CI与保存原生数据独立核对均已终态；文档准出由非作者另审，不由写者自证。当前判定是 native ranking FAIL，实验不晋级，不合并 PR69。

源码审查首次Linux无法解析Windows managedworktree gitdir、第二次Linux观察docs换行差异的两条失败保留，后以实际host-native Windows HEAD/clean回执作平台状态适配，原144/133逐字节门不变。旧70的146 build-input项实为144 src + README.md + pyproject.toml；后两项仅在本轮后验补核为REUSED，新build捕获仍144。包装脚本执行路径未捕获，不作额外来源证书。独立科学脚本还保留suite Python缺JAX启动失败与错将未选Fn常量答案用于原Bn的positivecontrol失败；后以已验证项目解释器和严格Bn=(I+L/4)^2给出的[3/8,7/8,7/8,3/8]修正已知fixture，制造场、方案和原门槛不变。原静态漏检回执及失败脚本保留，没有重标原执行PASS。保存原生数据审查另保留两处selector元数据比较对象过窄的辅助harness失败，仅恢复尚未完成row1，完成row0不重复。

下一只读机制核对在取数前单独声明，复用已验收固定v2 n64二阶参考，不新solve、不fit、不改原评分。非作者64项一次算术/接口审查确认：canonical η 残差RMS由70的3.507874737402458e−11降为dab的3.302236045265057e−11m（下降约5.8622%）；T残差由2.00454574553482e−9升为2.0094616138549377e−9°C（约0.245236%）。原 first-linear η RMS仍由6.859718001251586e−10升为6.898970719130102e−10m。这里只表明两个有限参考的排序不同，不能以该差异认定真实海洋误差更低、唯一归因、晋级或扩批。η用area，T用逐时actual dynamic h×area，各自空间归一化后作时间梯形积分；reference明确REUSED/sharedphysics/fixedH/linearcontinuity，不是真fullphysics/holdout。author代码缺单独constant1 RMS断言由非作者已知答案补核，原代码未重跑或改写。

可复算材料包括原生两个有效run的native/interchange数组、旧v3回放原数组、六个新制造场的全部原始数组、完整失败脚本/日志/资源、固定算法和数学控制、source/wheel身份及真正CI原log。原字节哈希清单将在准出时封存；source生产版本可由Git提交dabd48恢复。旧selector70 raw用于latestbaseline，只读参考沿用已发布v2范围，不形成另一套晋级评分。

材料链接：

- [保存原生与完整CI独立审查](data/momentum_tangent_dabd_stage_review__native-pilot-audit__AUDIT_REPORT.md)；[新瞬时资格终态](data/momentum_tangent_dabd_stage_review__FINAL_REVIEW.json)。
- [严格pilot决策](data/momentum_tangent_dabd_stage_native-attempt02__pilot-decision.json)；[primary对70比较](data/momentum_tangent_dabd_stage_primary-vs-frozen70.json)；[旧默认回放原数组](data/momentum_tangent_dabd_baseline_original_v3_primary_output.npz)。
- [新六制造场完整原数组](data/momentum_tangent_dabd_stage_review__new-momentum-arrays.npz)；[候选有效原生输出](data/momentum_tangent_dabd_stage_native-attempt02__01-candidate-primary-main__zhenmode__output.npz)；[候选native内部数组](data/momentum_tangent_dabd_stage_native-attempt02__01-candidate-primary-main__zhenmode__native.npz)。
- [限定数学workflow结论](data/momentum_tangent_dabd_proof_FINAL_VERDICT.json)；[S2数学接受/外层FAIL区分](data/momentum_tangent_dabd_proof_skeptic2-admissibility.json)；[外部原型独立原始核对](data/momentum_tangent_dabd_prototype_independent-data-audit__FINAL_RULING.json)。

- [固定二阶参考新核对声明](data/momentum_tangent_dabd_canonical_declaration.json)；[原读回结果](data/momentum_tangent_dabd_canonical_result.json)；[非作者64项核对](data/momentum_tangent_dabd_canonical_review__FINAL_RULING.json)；[复用的E64/theta64原数据](data/momentum_tangent_dabd_reference_reused_reference_arrays.npz)。

[完整原字节导出清单](data/momentum_tangent_dabd_export_manifest.json)包含294项（另计本manifest），原输入、数组、脚本和失败记录均可按source/relative/destination映射恢复。算法和物理定义不变；其他机器的宿主路径/解释器适配需另记录。
