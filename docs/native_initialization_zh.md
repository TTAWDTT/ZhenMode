# 原生海岸、水柱与温盐准备

`prepare-native-geometry` 生成全球 1°、360×180 的整格海陆表示；`prepare-native-initial` 在固定垂向部分单元上准备 WOA 点值；`complete-native-bottom` 按逐柱区域资料审查补齐水柱底段。三条命令保存文件身份、处理方法和参考库存。FD 输入桥及风驱动短步进已接入；完整 MOM6 层映射、FD 度量的物理精度、动力敏感性及完整机制运行仍须分别验证。

## 海岸与海深

海岸输入为 [GSHHG 2.3.7](https://www.soest.hawaii.edu/pwessel/gshhg/) high 二进制文件及其发布者校验收据。解析采用原始多边形层级和 Greenwich 展开规则；湖泊不计入海洋，南极采用冰前缘边界，冰架腔体未表示。点采样给出的海面比例与连通图具有分辨率误差，不能作为精确多边形面积。

海深输入沿用身份已记录的 ETOPO2022 缓存。本地文件含 CDO 双线性预处理，`original_NOAA_bytes_reverified=false` 保持原值。源经度从 0° 开始；位于经度接缝的源单元分别映射到两侧，纬向权重按球面面积计算。每个原生海格采用负高程海洋支撑的条件面积均值海深，保存非负海洋高程与无海深支撑的冲突。

初始整格海洋表示包含海面占比至少一半的格点，并保留有海面支撑的旧海格。通道修正使用对齐原生边界的 0.1° 海面采样图，先最小化新增整格数，再最小化球面路径长度。每条新增路线保存端点、经过的原生格及海面边界见证；候选格须有实际负高程支撑。明确指定的区域另采用 0.01° 采样。整格表示会合并同格内不同海面片段，产物同时记录缺少采样边界见证的整格开口数。

几何策略文件须明确列出区域细化及资料冲突处理，例如：

```json
{
  "policy": "binary_coast_channels_v1",
  "reference_dry_exclusions": [
    {"lon_lat": [337.5, 81.5], "basis": "retain_parent_dry_no_negative_marine_relief"}
  ],
  "regional_refinements": [
    {"id": "magellan", "lon_bounds_deg": [286, 292], "lat_bounds_deg": [-56, -50], "seed_lon_lat": [289.5, -53.5]}
  ]
}
```

`reference_dry_exclusions` 只允许保留原本无水、海岸新判为海洋而无负高程海深支撑的具名格。它不能删除已有水柱，也不能作为一般缺测处理。格陵兰附近 22.5°W、81.5°N 的批准选择仍保留地理未解决标记；取得可信区域海深后需重新审查。其他未列明冲突阻止几何准备成功。

`geometry.json` 与 NPZ 保存来源、采样支撑、球面面积、海深、通道路线和海陆差异。面积及体积分别记录多表示和少表示量，避免净误差掩盖相互抵消。只有整格图的海洋分量连通后，几何准备命令才返回成功；该结果的地理、动力和气候资格保持未取得。

## 点值与补底

原生点值使用已转换的 WOA13v2 PT/SR，保留配对原值。同层缺测沿湿连通海区补值；5500 m 以下沿用已经准备好的最深 WOA 平面。固定部分单元截到准备后的海深，单节点浅海保留该几何中的水深。参考库存采用点值乘参考容量的求积；MOM6 的有限体积层平均需要另做垂向映射。

补底审查文件选择 `regional_profiles_then_same_column_zero_gradient_v1`，绑定 `native_initialization.json` 的 SHA256、证据文件及每根待补水柱的决定。文件必须恰好覆盖所有可补底水柱。已有值、表层缺测、内部缺测及无锚点水柱不会被自动替换。所有科学资料判断由具名审查文件传入，命令核对其完整性和身份；文件身份本身不授予区域资料科学资格。

每根水柱的记录形式如下。`regional_profile_then_fallback` 可给部分底段提供区域 PT/SR，剩余底段按补底前同柱最深已解决点补齐；新增区域赋值点不作该次延拓的供体。`no_qualified_regional_profile` 则需要记录未选区域资料的原因及证据。

```json
{
  "native_flat_index": 2,
  "decision": "regional_profile_then_fallback",
  "reason": "区域资料及未选样本见对应审查文件",
  "evidence_index": 0,
  "regional_points": [
    {"node_index": 1, "ptemp": 8.0, "sr": 36.0, "source_trace": "制造示例；实际值须指向区域来源和映射记录"}
  ]
}
```

父审查对象还须含 `policy`、`native_receipt_sha256`、`evidence` 和 `columns`。证据条目含 `path`、`bytes`、`sha256`；路径相对原审查文件或采用绝对路径。完成产物原样保存审查文件，并在收据保存原位置，便于解析其来源。

输出保留初始缺测掩膜，并分别记录区域赋值、同柱延拓、原生供体层、垂向距离、供体的原始锚点及其水平补值来源。新增赋值点的输出 CT 由 PT/SR 计算；全部既有 PT、CT、SR 和 SP 数值保持原字节。输入或软件在准备期间改变时，产物写失败收据；已有输出目录拒绝覆盖。

## 敏感性与范围

另一个算术方案使用原水柱最深两个已解决点线性延拓 PT/SR。每个候选记录两个供体、距离、对应体积及 CT/SR 库存差异；没有第二锚点或超出数值检查范围的候选分别保留状态，禁止裁剪到有效范围。比较使用两方案均有有效值的同一底段支撑。动力敏感性仍须通过实际运行和分项预算验证。

准备批次 `omip2-20261007` 的几何记录 `native-coast-prepared-03/geometry.json` 列出 43,006 个海格和 35 个新增通道格，参考体积为 1.3402308523909478×10¹⁸ m³。原生点值准备剩余 147 个底段缺测节点，分布于 116 根水柱，占体积约 0.01989%，结果见 `native-coast-points-01/native_initialization.json`。独立结构检查器确认 552,217 个已有温盐点在补底后保持原值，结果见 `native-bottom-independent-01.json`。这些数量属于指定资料、节点和策略组合，修改输入须生成新记录。

该记录的区域审查采用官方 WOA13v2 0.25° 年客观分析场。源样本限同一原生格，须在两端深度上温盐配对、GSHHG 判为海面，并在最近 ETOPO 采样点具有所需深度及单一深度连通分量支撑。逐样本先按原声明的固定 Boussinesq 压力转 PT/SR，再按深度插值、以球面面积归一化。ETOPO 采样连通性仍有误差，压力也保留原近似定义。HTTP 字节段及 ETag 已保存，完整原文件尚未全量校验，来源身份限这些已取得字节与生成记录。

`quarter-regional-review-01/regional-review.json` 的区域规则选择了 125 个节点，其余 22 个采用同柱补底。其中 8 个节点的候选区域样本出现多个深度分量，审查文件记录了未自动混合的决定。`native-bottom-completed-02/native_initialization.json` 的两锚点线性方案在 142 个节点有效，4 个超出数值检查范围，1 个缺第二锚点；有效比较中的最大局部 PT、SR 差异约为 8.69°C 和 10.05 g/kg。局部差异和未定水域需要动力比较，全球体积占比较小不能代替该检查。

## 复跑与检查

在已安装产品环境中执行下列准备入口；Windows 数值命令须使用项目的单 CPU、180 秒、4 GiB 监督器，Linux 使用等价外部资源限制。下列 `SOURCE`、`PARENT_GEOMETRY`、`SHORELINE_RECEIPT`、`POLICY`、`NODES`、`BOTTOM_REVIEW` 均指实际文件或目录，输出必须采用新目录。

```text
python scripts/run_bounded_tests.py --module zhenmode benchmark prepare-native-geometry --parent-geometry PARENT_GEOMETRY --shoreline-acquisition SHORELINE_RECEIPT --policy POLICY --output GEOMETRY_OUTPUT
python scripts/run_bounded_tests.py --module zhenmode benchmark prepare-native-initial --source-prepared SOURCE --geometry GEOMETRY_OUTPUT --nodes-file NODES --output POINTS_OUTPUT
python scripts/run_bounded_tests.py --module zhenmode benchmark complete-native-bottom --native-prepared POINTS_OUTPUT --review-file BOTTOM_REVIEW --output COMPLETED_OUTPUT
python scripts/run_bounded_tests.py --script scripts/check_native_bottom_preparation.py --native-prepared POINTS_OUTPUT --completed COMPLETED_OUTPUT --output NEW_CHECK_JSON
```

未解决的海洋连通或温盐支撑令 CLI 返回 3。`complete_wet_support=true` 仅表示全部湿节点已赋值，`native_initialization_ready`、`mom_layer_initialization_ready` 和 `execution_ready` 继续为 false。当前参考节点及球面面积也须通过实际求解器接口验证，不能直接据有限温盐数组宣布完整 case 可运行。

独立结构检查器不导入产品 helper，重新推导固定容量、逐柱寻找供体，并用 `math.fsum` 计算参考库存。检查器检出改变已有值、遗漏湿点、错误供体三个植入错误；它共享输入资料、NetCDF/NumPy 读取层和声明常数，不验证区域地理或热力学转换的科学独立性。数据测试另覆盖制造已知海面、正高程冲突、不能删除已有水柱、接缝、通道路线、区域优先、非法补值、源身份改变及越界敏感性。

## FD 输入桥

`prepare-fd-initial` 将完整固定部分单元点值导出为正式 FD 输入文件。`load_fd_native_inputs` 返回实际 `GlobalOceanGrid`、CT、SR 和全网格 dbar 压力，供调用者传入 `make_solver_global` 与 `init_state`。调用者仍须选择 `fixed_partial_v1`、`teos10_reference` 及自己的机制配置；数据入口不切换生产默认，也不启动积分。

策略文件须明确选择有效球面积度量、固定 Boussinesq 参考压力及无水节点的数值占位值：

```json
{
  "schema_version": 1,
  "horizontal_metric": "geographic_area_v1",
  "pressure_definition": "fixed_boussinesq_reference_v1",
  "inactive_ct_deg_c": 15.0,
  "inactive_sr_g_kg": 35.0
}
```

入口保留全部湿点 CT/SR 字节，只在无水节点填入策略指定的有限占位值。压力为全节点 `rho0*g*depth/10000`，使用项目常数 1025 kg/m³ 和 9.81 m/s²；每个被评价节点均须满足组件数值检查范围，包括无水参考节点。入口拒绝超范围压力，保持输入节点，禁止裁剪或自动补值。探索用 8000/9000 m 干节点因此不能直接传入；7950 m 干节点替换方案的原型工厂接纳记录见 `outputs/omip2-20261007/fd-native-entrance-02/receipt.json`，其积分步数为0、完整case资格为false。方案保留了实际湿节点，无水参考模板改变仍需动力检验；该原型记录不构成本节正式入口的运行验收。

原生球面积 `A` 进入 `dx=A/dy`、`dy=R*dlat`，并保留中心 `cos(lat)`。有效度量满足既有面联系的 `dx=L*cos(lat)` 与 `A=dx*dy` 兼容规则；输出记录它与历史沿经度方向的宽度的差异。物理面宽、向量度量及动力精度须另行验证，输入接受不能替代这些检查。

准备命令保存 `fd-native-inputs.npz` 和 `fd_initialization.json`，保留原生初态时间及 WOA 气候时间定义。读取时再次核对文件身份、容量、球面积、有效度量、压力定义、数组形状和无水占位值。即使人工重绑了文件摘要，定义不符的压力、容量或广播型示踪物仍会被拒绝；准备过程中输入变化则保留失败收据。

```text
python scripts/run_bounded_tests.py --module zhenmode benchmark prepare-fd-initial --native-prepared COMPLETED_OUTPUT --policy FD_POLICY --output NEW_FD_INPUT_DIRECTORY
```

该文件格式只取得数据准备状态。产物保留 `completed_timesteps=0`、`initialization_entry_verified=false`、`execution_ready=false`；实际工厂读取、机制运行及重启／评价仍要由各自执行记录确认。
## 真实 GPU 风驱动运行与续跑

`zhenmode benchmark run-fd-wind` 将准备好的固定部分单元 CT/SR 初态和原生 JRA 文件送进
实际 FD 积分器，保存六个状态字段、绝对步数和逐步预算。它是明确命名的风驱动组件运行：
当前海温和海流参与风应力计算，热、雨雪、蒸发、径流、陆冰排水、盐恢复和海冰尚未施加。
读取全部天气字段也不会把这些过程计作已启用，输出保持完整 case 资格为 false。

在 Linux/WSL 的 CUDA 环境中执行：

```bash
zhenmode benchmark run-fd-wind \
  --native-prepared FD_INPUTS --forcing-manifest JRA_DIR/forcing.json \
  --start 1958-01-01T00:00:00 --dt-seconds 0.1 --steps 4 --output RUN_A
zhenmode benchmark run-fd-wind \
  --native-prepared FD_INPUTS --forcing-manifest JRA_DIR/forcing.json \
  --start 1958-01-01T00:00:00 --dt-seconds 0.1 --steps 2 \
  --resume RUN_A/checkpoint.npz --output RUN_B
```

第二条命令从第 4 步继续两步，天气取样使用原始起点加绝对步数，不重置到首个时刻。
检查点复用已有严格格式，绑定几何、有效参数、输入、产品源码和运行环境；改变 dt、
参数或来源会拒绝接续。已有输出目录会拒绝覆盖。资源监督使用单 GPU、一个主机 CPU、
8192 MiB 主机 RSS 上限，短窗默认墙时 360 秒，可显式声明不超过 10800 秒。
RSS 为进程组采样监督，GPU 使用 0.40 分配器比例，两者均不冒称设备或 cgroup 硬配额。

`run.json` 保存启用范围、模拟时间、逐步读取／应力／核心及预算计时；`budgets.json`
保留实际残差，进程完成不等于守恒或稳定性通过。极区滤波按现有每步固定比例应用，
应与物理混合区分。`--match-transport` 可显式启用已有输运修正候选，默认行为保持。

跨进程 GPU 重算可能因编译自动调优出现浮点差异。核查可复现性时，可按 GPU 运行说明
使用 `XLA_FLAGS=--xla_gpu_autotune_level=0`；本轮同配置的真实连续 6 步与 4+2 步续跑，
52 份状态／累计预算／预算历史数组的 dtype、shape 和完整 C-order 字节相同（含正负零）。
比较记录为 `D:/Github/ocean-solver/outputs/fd-real-step-20261008/gpu-restart-bit-comparison-det-01.json`。
该结果限于此次版本、设备、环境和短窗。
可将 `JAX_COMPILATION_CACHE_DIR` 指向本任务自己的缓存目录，复用编译结果；冷编译与
缓存启动须分开计时。真实气候验证、完整冰海交换和正式误差—成本比较仍需后续运行。


### 原生 SIS2 与 GPU 海洋交换

`NativeSIS2` 在一个持续运行的 CPU 进程中调用固定构建的 SIS2；FD 海洋步进和表面库存更新在 GPU 上执行。海冰厚度类别、雪、热力学剖面和动力学状态保留在 SIS2 内部，完整重启交给原生 `ice_model_restart`。`JaxStateG.ice` 仍属于原有简化海冰路线，不能替代 SIS2 的状态；此耦合路线关闭简化海冰闭合。

可安装的连接入口是 `zhenmode benchmark compile-sis2-bridge --coupled-build BUILD --output OUTPUT`。它先核对已有构建的源树和中间文件记录，再把随包安装的 `sis2_bridge.f90` 链接到原生对象和库，并保存链接命令、驱动源码、可执行文件及依赖身份。新驱动只安排调用和交换数据；SIS2 的热力学、动力学和输运仍由原生模块计算。`NativeSIS2.exchange` 接收海洋表层的 A 网格速度、PT（K）、SR（g/kg）、海面高度和 frazil 能量；推进时还要求各海冰类别的完整大气通量，以及显式的径流、陆冰和相应热通量。

海水表层采用参考单元厚度加海面位移计算质量、盐和热库存，参考垂向几何保持固定。过冷 CT 先升至 TEOS10 表面冻结点，正的能量缺额传给 SIS2；未使用的 frazil 能量缺额返回海水，并从海水焓中扣除。原生盐通量的单位为 kg/m²/s，正方向离开海水；SIS2 的长波通量正方向进入海水，感热和潜热正方向离开海水。质量交换携带的原生焓、雪和陆冰融化潜热分别接入海洋；液态降水采用 0°C 液态参考焓。海水库存包含 `eta * surface_tracer`。可选 `surface_pressure_pa` 同时进入三维压力梯度和外模态，接收 SIS2 的大气／海冰表面负载。该字段省略时沿用原有入口。

STEP 回复绑定原生 `interval_seconds`。表面交换使用该时长；可选 `duration` 用于核对。时长不一致或湿表层容量耗尽会被拒绝：在 JIT 动态输入下返回不可接受的非有限温盐状态，由运行器的有限性检查阻止接受。当前表面负载也可传给 `diagnostics(state, surface_pressure_pa=...)`，使压力诊断与步进采用同一负载。桥接编译前后核对原构建记录中的 MPI 包装器和 Fortran 编译器身份。

五步人工输入试验使用 4×4×4 FD 网格、固定垂向部分单元和 TEOS10 CT/SR，海冰使用同水平位置的 4×4 原生网格。双方在该试验中采用 SIS2 给出的单元面积。前两步冷却并降雪，后三步升温；试验还施加非均匀 frazil、非零应力、径流和陆冰。RTX 4060 Laptop 执行海洋主积分，SIS2 在一个 CPU 上运行；每步 1 秒，共推进 5 秒。各步分别保存交换源、海洋核心库存变化、两者合计与外部输入的差，以及两个组件的时间。热、盐差值包含海洋核心的库存变化，不能将交换算子的恒等式当成整个模式的守恒结论。

相关记录在 `outputs/fd-real-step-20261008/sis2-native-probe-01`：`case-gpu-coupled-05/result.json` 保存上述实际耦合运行，`case-gpu-coupled-installed-01/result.json` 保存仓库外 wheel 安装后的重复运行。`native-full-restart-comparison-02.json` 核对原生 SIS2 连续三步与两步后保存、恢复再一步：本配置重启文件中的全部 45 个 NetCDF 变量均具有相同形状、类型和数据字节。这项比较只覆盖海冰重启；GPU 海洋与 SIS2 共同时间的联合重启还需要单独验证。独立表面库存／压力平衡控制通过 6 项测试，另一个已有测试核对带预算记录的步进与实际核心解一致；一次扩大到全部阶段预算参数组合的运行在 180 秒上限中止，其记录保留。

上述人工输入试验未使用真实 JRA 全局强迫。短波仅在表层沉积，穿透分布、实际全球海岸与极区几何、混合配置及成对长期运行仍需要接通和验证。记录中的 `full_case_qualification` 保持 `false`，小网格时间不能用于宣称全球加速或工业级资格。


### 全球 SIS2 几何与原生 FMS 交换

`zhenmode benchmark prepare-sis2-case --native-prepared FD_INPUT --output CASE` 写出 SIS2 可读取的海深和显式 supergrid。海陆掩膜及湿格海深来自已准备的 FD 输入，子格面积按球面矩形积分，并由原生 mosaic 读取器合并。此次全球读取中，43,006 个湿格、湿格海深和经纬位置均与 FD 相同；经纬坐标差为零、湿格海深数据字节相同，面积最大相对差为 1.2104989141785517e-14。解析 `spherical` 配置的中点面积有约 12.7 ppm 差异，因此该入口采用文件网格。准备记录的 `native_read_verified=false` 表示单次准备命令没有运行原生读取；实际读取另存 `native-read.json`。

`NativeSIS2.bulk` 在实际 SIS2 表层温度、流速和粗糙度上调用固定原生 FMS 模块，输入气象状态的高度为 10 m。开水面的九项输出已与此前通过作者系数检查的 JAX 组件对照；比较采用同一组原生表层温度和流速，包含 SIS2 的 A→C→A 重构。原生粗糙度会在快速海冰更新后变化，下一次交换使用更新的值；海面／海冰不施加地形粗糙度倍率。此对照覆盖开水面 bulk，海冰系数尚未独立验证。

原始 JRA55-do 年度径流、陆冰和原始放流格点面积重新核对后，两个淡水字段按已批准的新海岸重路由，原始与映射质量通量一致。其余九个气象文件按 SHA-256 核对后复用，水平位置、单元面积及权重不变。重新路由的真实陆冰输入涉及 394 个湿格；真实降雪涉及 11,313 个湿格。记录在 `outputs/fd-real-step-20261008/jra-revised-coast-01`。

实际全球短窗已使用真实 WOA CT/SR、上述几何和重新路由的 JRA 输入：原生 SIS2 完成 1 秒快速与慢速更新，GPU 海洋完成十个 0.1 秒步进，两者共同推进至 1 秒。交换包含热、雨雪、蒸发、径流和陆冰；海洋主积分保留在 RTX 4060 Laptop GPU，SIS2 在 CPU 上运行。记录为 `sis2-native-probe-01/case-global-jra-gpu-coupled-01/result.json` 和 `ocean-exchange-budget.json`。计入海面位移的海洋库存与实际施加量之间仍有热、盐差值：分别约 -2.747e9 J 和 -7.342e3 kg；水差为零。这些差值包含核心步进，已单独保存，不能视为全部来源闭合。

该真实运行保留组件范围：短波在表面沉积、GM/Redi 和盐恢复关闭，辐射沿用既有 MOM 候选设置。大步长耗散、联合重启、完整过程配置及长期评价还在实现和验证；`full_case_qualification=false`。原生读取、bulk 对照和真实 1 秒运行分别支持对应结论。
