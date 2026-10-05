# ZhenMode／MOM6 机制符合表

对应[规范 zhenmode-omip2-physical-v1](benchmark_spec_zh.md)。核查日期2026-10-05。下方原有现状表仍以main合并PR38后的2d17198619d48618c758354a268e3ba425ee4e47为生产默认基线；MOM6旧接入为f49a00096df607b48354603e2398e14e189fd62e的solo-driver/tc1与两个0.5°30日case。新增组件与耦合候选分别登记，不把它们算成旧年度配置已启用的能力。

## 初始实现快照（PR40）

| 范围 | 已实现及验证 | 仍不具备的资格 |
| --- | --- | --- |
| 共同合同 | `zhenmode benchmark freeze/check-contract/plan`；拒绝被改写后重新哈希的合同；短窗/六循环同物理hash；真实Gregorian时长 | 数据、原生方案、观测协议尚未完整绑定，所有计划 execution_ready=false |
| B01 | [JRA读取](../src/zhenmode/model/inputs/forcing/jra55.py)：11种字段、实际文件hash、SI/10m/日历/grid/bounds校验、瞬时插值与均通量区间积分 | 仅接受已准备的原生矩形网格；不下载、不重网格；真实资料/转换权重尚未冻结 |
| B02–B03 | [在线bulk](../src/zhenmode/model/solver/physics/air_sea.py)，独立作者Fortran八工况参考和符号/反馈/强风负例 | 开水面组件，未证明完整冰海大气交换或气候效果 |
| B04–B06 | 净短/长波、雨/径流/蒸发虚拟盐、50m/年piston恢复；[显式FD接线](../src/zhenmode/model/inputs/forcing/online.py)的小网格实际调用 | 不含穿透/雪/陆冰/冰；恢复目标仍由调用者提供，不冒充已绑定月WOA |
| B12 | [独立端点预算](../src/zhenmode/model/diagnostics/surface_budget.py)：体积测库存、面积积分外源，破坏热源的负例产生非零残差 | 仅水热/虚拟盐表面组件；没有完整水冰质量/焓/盐和长期漂移验收 |
| MOM6+SIS2 | [新候选pins](../src/zhenmode/baselines/mom6/omip2-pins.json)冻结官方examples及9个组件、2个嵌套依赖和5份配置实际字节；可计划、检查完整源码并准备已知配置改动 | 未获取完整新源码/输入、未构建执行；缺口不会因配置准备成功而消失 |

MOM6候选为官方 `OM4_025.JRA`，examples commit `9b856f1d620f79e8035f96deb8004d544af77af3`，不是已匹配的基准。审读其实际coupler的 `ncar_ocean_fluxes`：中性阻力为LY2004表达式、迭代两次；不含LY2009强风修正，气温/饱和湿度约定也需对齐。其原配置采用julian日历、0.1667m/日盐恢复、5PSU截断和全球淡水修正，EOS为WRIGHT，风bicubic而通量bilinear映射。`omip2-prepare`目前只更正日历/窗口、piston、盐差物理范围截断和全球修正；不会偷偷把旧盐目标、EOS或映射宣称为新合同。完整源码变更及数据绑定后才能构建验收。

独立参考来自[JRA55-do作者例程](https://github.com/HiroyukiTsujino/JRA55-do/blob/30c8e1a84386c1db8a436d980826b884b2075089/anl/diagflux/src/bulk-ncar.F90)，通过[本项目小驱动](../tests/support/ncar_reference.f90)编译并保存[参考及来源hash](../tests/support/ncar_reference.json)。只对系数和饱和湿度做对照；作者例程使用输入θ算密度，不能据此宣称本项目全部通量逐位相同。其0.4常量默认单精度后提升，系数对照容限1e−7有明确来源。没有复制上游整套实现进入正式包。

已运行：合同6项、强迫读取24项、通量4项、表面接线/预算/精度3项、MOM候选3项；连同源码身份与结构检查共78项通过。另58项既有表面受热、混合层、评价负例和生产重启检查通过。采用单CPU、每次180秒、4GiB上限；独立Fortran参考用Linux单CPU/180秒/4GiB虚拟内存限制编译/执行。测试收集从1014增至1054，新增40项，原测试无遗漏。干净wheel安装后在仓库外执行了合同和MOM候选计划命令。NetCDF4/NumPy 2.5的数组shape弃用警告仍存在。数值验证为制造小数据和8×8×4 FD组件步骤。MOM6五份上游配置的字节校验与改动展开已实际执行；完整源码checkout准备、MOM6构建、真实全球短窗、GPU、长期积分及新观测评分均未运行；这里的通过不是工业资格或加速结论。

## 候选源码与输入准备的进一步实施

完整固定源码已获取并完成Linux原生Git核验；准备输出包含独立暂存的LY2009/Gill无冰交换适配，上游cache未改。实际MOM6原始/适配例程分别编译并使用真实FMS常数模块，与此前作者Fortran参考的8个样本比较：适配系数最大相对误差约2.75×10⁻⁸，通过1e−7容差；原始LY2004约0.715，作为规则差异控制。完整surface_flux模块链接真实FMS，经初始化后执行8个工况，饱和湿度、感热、蒸发、应力和出射长波也通过独立系数参考与解析空气性质检查。该比较不含SIS2潜热能量、吸收辐射或海冰，既不是MOM6整体精度缺陷，也不是海洋RMSE或完整通量同义证明。

可安装入口新增原始JRA首年资料的发布方SHA256核验、失败记录、排他写锁与HTTP续传，以及每段单CPU/180秒/4GiB监督的MOM6+SIS2构建。构建从固定提交的Git archive生成干净源码树，排除cache中的未跟踪构建控制；实际源码、执行权限、链接、生成物和二进制依赖分别绑定。当前完成了含LY2009/Gill及SIS2→MOM实际潜热传递适配的耦合构建。编译通过不授予标准实验执行资格。

`SIS_fast_thermo.F90`无冰潜热的温度依赖、MOM边界的独立潜热能量字段及两侧冰—海底部热记录已显式适配。实际源码表达式编译检查覆盖凝结、单位缩放、干格及冰底部其他热项；这不是完整耦合步骤或大气—冰顶部库存闭合证明。顶部仍有固定潜热记录待对齐。原始FMS2的`time_interp_external2.F90`采用线性时间插值，并仅在特定属性下读取`average_T1/T2`，CF bounds存在不足以证明均通量保持。

新增[时间适配](../src/zhenmode/baselines/mom6/omip2_time.py)只对显式标记的均通量选择覆盖完整计算区间的记录，普通输入保持原有线性插值。两个实际coupler调用者的固定字节绑定步末雨雪/辐射和步首径流/陆冰。时间格式适配保留真实日期与通量值，转换为FMS支持的Gregorian/1958年参考时刻，并声明unlimited记录与原生bounds。完整实际FMS模块与真实库的制造输入检查已通过：六小时累计能量4.32 MJ/m²；跨界、越界、单记录日均越界、单位不一致及旧负时刻编码五项控制被拒绝。真实首窗11字段、6时刻共66项独立检查也通过，比较三个格点与整个矩形场求和。真实检查约9.6秒、峰值约226MiB，单CPU/180秒/4GiB。两项检查均未运行耦合海洋；完整case的输入表、实际时钟、累计预算和运行收据仍待绑定。含读取补丁的新构建另记状态，不能沿用先前二进制身份。

含读取补丁的新耦合构建已完成：前两段达到墙时上限，第三段约75.5秒完成；每段记录实际源码未变化、产物及动态库身份。最终二进制SHA256为`f3162132520d4d072c87323432ffb792e840854e41e64f3c4f4b3224f39915bc`，资格仍为build-only。Windows相关基础设施及输入合同114项通过、1项POSIX专属跳过；WSL干净wheel、仓库外执行相关输入/接入/资源合同46项通过。测试收集1090→1099，原测试无遗漏。没有把整个收集集合写成全部执行通过。

补充核对SIS的实际`ice_model.F90`导出：内部`IOF%flux_lh_ocn_top`经`US%QRZ_T_to_W_m2`成为公开`Ice%flux_lh`，故耦合边界已经是W/m²。扩展编译检查执行该语句和新耦合传递调用，再进入实际MOM消费表达式；采用不同的SIS/MOM非单位缩放，并通过凝结与独立SI期望值。单格恒等交换只供应存储，仍未验证全局重分配或耦合步骤。原生面积另按经纬bounds与6371km半径独立核对，拒绝正值但错误的面积；Ctrl-C记录failed。对应负例与输入/接入合同20项通过、1项平台跳过。

相邻年度收据可重复接入：逐文件身份、资料角色、grid/bounds与许可核验后合并时间索引，按来源流式读取。制造的1958→1959窗口证明天气跨年插值与均通量区间积分，重叠、断档和缺下一年端点被拒绝；实际重复参数CLI也执行过。没有把该制造窗口算成年度积分。相关基础设施/输入/接入合同129项通过、1项平台跳过；干净wheel、仓库外真实首六小时重新准备约101秒、峰值266MiB，11个生成的字段文件与修正前字节一致，原生area最大相对差约1.2×10⁻¹⁴。真实跨年原始资料尚未获取或执行。

输入review与时间适配合并后，Windows相关合同139项通过、1项平台跳过，WSL干净安装相关合同54项通过；收集1107项，原1090项无遗漏。真实首窗的实际FMS检查进一步覆盖全部4276800个值，与独立CF参考逐值相同，约8.4秒、峰值226MiB；仍不是海洋积分或观测评分。

真实1958年JRA55-do v1.4.0的11份原始年度文件已核对发布方SHA256；六小时窗口在显式1°全域矩形网格完成转换及读取器核验。天气状态采用周期双线性映射，辐射/雨雪采用球面面积映射，径流/陆冰按作者明确的原始整格面积转成kg/s再路由。未解析小岛通过原地形格内陆地比例提供沿岸接收点；非零排水超过500km拒绝。独立质量积分采用fp64，避免原始fp32求和造成假残差。源文件、面积、权重、路由、实际执行模块和生成物均有收据；失败尝试保留本地。该准备尚未接入MOM6运行，tripolar映射与公共观测评分仍缺失。

WOA13v2年温盐及首窗所需月盐资料已取得，记录官方URL、CF元数据和本地SHA256；没有取得发布方校验值，不宣称发布方字节认证。现有ETOPO缓存保留CDO预处理历史，未独立复核原始NOAA文件。二者仍需原生初始化/恢复接线。

两边实际无冰表面组件与固定独立参考的8工况对照已通过：感热RMSE约6.92×10⁻⁸ W/m²，蒸发约8.10×10⁻¹⁴ kg/m²/s，应力约5.52×10⁻¹¹ Pa（fp64）。这些是组件误差，不是海洋观测分数。完整海冰、非线性EOS、风/浮力混合、顶部预算、原生全球输入和观测评价未补齐；真实全球积分、新观测评分、GPU与长期实验均未运行。默认生产预设没有切换。

## 生产默认与旧对照接入的现状

2026-10-06新增[非线性热力学组件](thermodynamics_zh.md)：JAX 75项密度与导数对未经改写的固定GSW Fortran检查228状态，密度RMSE约fp64 1.52×10⁻¹³、fp32 6.26×10⁻⁵ kg/m³；执行实际MOM密度函数的Pa换算，并保留共同压力及中性梯度负例。Windows组件/基础设施104项通过，仓库外干净wheel组件16项通过，收集1107→1123。它尚未接入生产变量、初始化、表面热量和所有密度使用者；未核多项式相对精确Gibbs EOS的funnel精度。B08完整符合仍未取得，不把组件成绩写成全球观测分数。

状态含义：**符合**=有对应范围的证据；**部分**=已有基础但不满足完整要求；**缺失**=当前实现/配置缺少所需链条；**待核**=有接口或上游例子，但没有选定配置、构建与实际运行证据。源码能力、当前接入与执行资格分别写，不能用一个勾概括。

## 表面与内部机制

| 规范项 | ZhenMode现状、证据 | MOM6上游能力／当前接入 | 下一份验收材料 |
| --- | --- | --- | --- |
| B01 高频大气 | **缺失**：年度使用NCEP月均风、年均空气；无湿度/气压/辐射全链。[读取器](../src/zhenmode/model/inputs/forcing/reanalysis.py)有月空气选项，也不等于JRA高频支持 | **待核／缺失**：需选定完整大气强迫driver。现有[导出器](../src/zhenmode/baselines/mom6/forcing.py)只覆盖风、空气和感热代理 | v1.4.0实际文件清单；时间bounds、单位/高度、跨年及海陆/向量转换正负例 |
| B02 相对风/NCAR应力 | **部分**：固定Cd=1.3e−3计算应力，无当前海流反馈；[seasonal](../src/zhenmode/model/inputs/forcing/seasonal.py)用360日/五日月界混合 | **部分／缺失**：pinned solo支持规定应力/A→C转换；不等于已接NCAR在线相对风 | 固定天气与各海流的算法参考；α=1、换算/平均顺序与更新相位 |
| B03 感热/潜热/蒸发 | **缺失**：年度Qideal+80(Tair−SST)有实时反馈，但不是完整bulk。[组装](../src/zhenmode/model/inputs/forcing/bundle.py)、[tendency](../src/zhenmode/model/solver/dynamics/tendencies.py) | **待核／缺失**：solo file可输入通量及live-SST恢复；solo atmos_ocean_fluxes为dummy，不提供完整大气交换 | 选定coupler/强迫组件commit；温湿风/海温变化的独立通量参考及实际耦合日志 |
| B04 辐射/短波穿透 | **缺失**：年度仅理想纬向Q。已有[受热权重](../src/zhenmode/model/solver/physics/surface.py)，不等于短波吸收 | **部分／待核**：solo file读短/长波、热源等；完整出射长波、反照率及沉积配置未运行核验 | 分项辐射及列沉积积分、符号控制、方案与参数冻结 |
| B05 淡水/盐 | **缺失于年度**：无所需雨/雪/蒸发/径流链；已有盐恢复/可选冰盐源不是全部淡水预算 | **部分／待核**：solo有雨/雪/蒸发/径流输入字段，但当前全球case未形成完整JRA输入及水盐预算证明 | 水/盐方程、质量或虚拟盐语义、原生源项与全局预算 |
| B06 恢复约定 | **部分**：有SSS恢复选项，年度关闭；现热式不满足新规范。[factory](../src/zhenmode/model/solver/factory.py)；冰空气floor不适用于新profile | **部分／待核**：有T/S恢复系数；没有本规范WOA上10m目标、50m/年及冰区掩膜的已执行配置 | 生效参数及单位转换；恢复盐源独立输出；确认无直接SST恢复 |
| B07 海冰 | **部分**：有可选[最小闭环](../src/zhenmode/model/solver/physics/surface.py)，年度未开启；源码明确没有冰动力、夹卷或分层热力学 | **待核／缺失**：官方有MOM6+SIS2例子；仓库当前pins/build没有固定SIS2/coupler执行链 | 完整组件pin、初冰及雪处理、冰水质量/焓/盐控制和季节诊断 |
| B08 非线性热力学 | **缺失**：[EOS](../src/zhenmode/model/solver/physics/eos.py)明确仅线性；[MLD诊断](../src/zhenmode/model/diagnostics/mixed_layer.py)也用线性密度 | **已有源码能力／待核配置**：pinned MOM_EOS支持TEOS10等；当前缓存r8不能据源码选择列表证明实际EOS/变量语义 | 温盐压转换、独立GSW样本、热容量和冻结温度语义；实际resolved EOS |
| B09 风/浮力驱动混合层 | **部分**：[vertical](../src/zhenmode/model/solver/physics/vertical.py)有背景扩散及局地对流；受热深度选项/密度MLD不等于完整湍流闭合 | **待核／待核**：CVMix依赖已pin；未选择/验证本规范的混合层配置，不凭链接库存在算通过 | 风驱动、冷却/稳定层结、夹卷与列热盐预算过程例 |
| B10 海域/初态/输运 | **部分**：已有全球FD输运与重启。年度±65°墙、min_depth500、14节点离散不代表规范全域。[输入准备](../src/zhenmode/model/inputs/prepare.py)、[几何](../src/zhenmode/model/solver/geometry/fd_metrics.py) | **部分／缺失接入**：当前prepare-native只接受720×260的两个30日case；不能容纳新全域/native网格协议 | 实际网格bounds/面积/体积、海峡与浅海、初态转换/库存、离散误差及重启 |
| B11 涡旋/水平/底部闭合 | **部分**：年度GM/Redi均关闭。[isopycnal](../src/zhenmode/model/solver/physics/isopycnal.py)现共用skew算子，注释说明两者系数会叠加；名字不证明两个独立机制 | **待核／待核**：未逐项审读完整原生闭合，也没有冻结本目标配置 | 按分辨率选方案；GM与Redi的独立过程作用、参数和实际开关 |
| B12 独立预算 | **部分**：已有[ledger](../src/zhenmode/model/diagnostics/budgets.py)记录source/stage/residual；年度未开启预算审计，端点OHC/盐变化不是闭合证明 | **部分／待核**：tc1报告首末内容/CFL，adapter明确不是独立预算闭合；完整预算诊断未接入 | 原生库存与独立外源/边界测量、未回填残差、负例及预算收敛 |

B10的固定参考层厚/线性自由面本身不是不符合行业机制的证据；需核相应库存和外源关系。旧材料库存候选353失败不用于判定本表，也不指定它为升级路线。

## 执行、评价和来源

| 合同项 | 当前基础／限制 | 本规范资格 |
| --- | --- | --- |
| 输入来源 | PR38接通精确实际路径、全文件hash与严格输入；新JRA/观测/权重未获取冻结 | **部分**；新profile不能运行 |
| MOM6 source→binary | [pins](../src/zhenmode/baselines/mom6/pins.json)固定MOM6/FMS/CVMix/GSW。[adapter](../src/zhenmode/baselines/mom6/adapter.py)要求build_manifest；旧缓存缺绑定收据 | **待核**；binary hash+clean source不替代构建证明 |
| 运行器 | ZhenMode CPU/CUDA监督与独立run ID已接；MOM6当前1CPU/1rank/180s/4GiB/128MiB输出 | **部分**；后者不能直接承载完整profile，超额另需计划和授权 |
| 月均/3D诊断 | 年度输出十日瞬时SST且save_3d=False。现[output](../src/zhenmode/model/io/output.py)有可选字段 | **缺失目标输出**；严格月bounds/3D/通量积分未接 |
| 观测评分 | [protocols](../src/zhenmode/evaluation/protocols.py)当前只接受保存初始SST、area-v2、保存记录等权、无重网格 | **缺失新协议**；现有scorer不得重标工业气候评分 |
| MOM6原生转换 | [external](../src/zhenmode/evaluation/external.py)要求参考同网格中心/湿区并取T_init；非通用重网格 | **缺失公共观测评价链** |
| MLD/海冰/环流 | 有局部诊断/最小冰报告；没有规范下完整时间序列与统一观测处理 | **部分**；不以初态或零冰覆盖充当运行技能 |
| 采样合同 | 当前时间轴检查覆盖和末日；独立制造控制显示不同保存频率可得到不同area-v2分数 | **待扩展**；新协议必须核完整bounds/缺月/权重，旧实现语义保持 |
| 成本/误差 | PR38分compile/integration/IO/E2E；既有compare不自动授予equal-error speedup | **部分**；forcing分项、设备峰值、新profile误差—成本曲线待实现 |
| 气候效果 | 既有长期稳定与低RMSE属于历史谱系；PR38年度为简化配置/初始参考。MOM6新profile未跑 | **未运行**；两边均无本规范下的工业气候验收 |

总体：ZhenMode已具备正式FD生产与工程运行基础；MOM6已具备固定源码与小例接入基础。**两边当前都不具备本规范完整执行/评价资格。**上游MOM6完整配置的存在不能认证当前solo缓存，更不能证明新profile成绩。

## 证据定位与阅读边界

- 项目源码：上表相对链接对应2d17198；年度具体选项见[正式预设](../configs/zhenmode/presets/global-045deg-seasonal.yaml)。[PR38](https://github.com/TTAWDTT/ZhenMode/pull/38)保留作者运行记录；本轮没有独立重跑年度或严格尾段重启。
- 固定MOM6：[solo surface](https://github.com/NOAA-GFDL/MOM6/blob/f49a00096df607b48354603e2398e14e189fd62e/config_src/drivers/solo_driver/MOM_surface_forcing.F90)的set_forcing、file buoyancy、get_file_time_level及参数；[solo dummy](https://github.com/NOAA-GFDL/MOM6/blob/f49a00096df607b48354603e2398e14e189fd62e/config_src/drivers/solo_driver/atmos_ocean_fluxes.F90)全文；[EOS](https://github.com/NOAA-GFDL/MOM6/blob/f49a00096df607b48354603e2398e14e189fd62e/src/equation_of_state/MOM_EOS.F90)的EOS_init。源码审读不是实际调用证明。
- [官方MOM6-examples](https://github.com/NOAA-GFDL/MOM6-examples)目录/README证实有ice_ocean_SIS2配置；该可变上游没有在本任务中锁定运行配置。SIS2、coupler及bulk实现的具体tuple仍是待核项。
- [OMIP-2 §2及Appendix C](https://gmd.copernicus.org/articles/13/3643/2020/)、[JRA55-do2018 Table1](https://www.coaps.fsu.edu/pub/eric/papers_html/Tsujino_et_al_18.pdf)、[GSW密度](https://www.teos-10.org/pubs/gsw/html/gsw_rho.html)作为机制参照；没有声称所有论文/模式源码全文均已读。
- 文献要求、项目选择与源码事实分开；没有借新增库/接口给任何模式补写PASS。当前build、数据获取、数值容差及效果阈值未冻结的项继续显式待核。

## 实施顺序与审阅决定

| 优先级 | 交付 | 验收范围 |
| --- | --- | --- |
| P0 | 审阅规范项目选择：资料版本、α、湿空气公式、SSS恢复/冰区、初态与评价身份 | 形成冻结v1及数据/验收清单；不先调算法或重跑简化年度 |
| P1 | JRA读取/时间语义与NCAR表面交换；同步选定MOM6完整强迫/海冰driver与依赖tuple | 两实现对独立固定海态参考，温湿风/当前SST负例与累计输入 |
| P2 | 淡水、非线性热力学、混合层、完整冰机制及预算逐项接通 | 独立过程例与实际库存/外源；保持既有生产基线独立身份 |
| P3 | native输出、公共观测/重网格新协议、机制资格检查与成本分项 | 单一可安装evaluation实现；拒绝缺字段、错误单位/窗口/mask/协议 |
| P4 | 同物理机制短窗→年→完整循环，按资源审批扩大 | 执行完成、机制符合、效果与成本分别授予；独立复跑/作者/历史/失败分列 |

这些是下一阶段工程任务，不是本次已经实现的功能。改变EOS、表面通量、混合或库存的工作分别列行为影响与前后证据；不会藏在目录重构中，也不以回填残差形成自证。
