# 开发与测试

需要 Python 3.12+。仅安装正式包即可收集和执行仓库测试：

```sh
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -m ruff check .
python scripts/make_synthetic_bathymetry.py
python -m pytest tests/ -q
python -m zhenmode mms
```

`requirements.txt` 固定 CPU 开发验证环境；`pyproject.toml` 定义可安装包的依赖和入口。GPU 环境见 [GPU 运行](gpu_runtime_zh.md)。

`tests/fd`、`tests/runtime`、`tests/data` 对应模型、运行与输入；`tests/coupling` 对应外部组件交换；`tests/baselines`、`tests/evaluation`、`tests/experiments` 对应对照、评分与实验契约。旧 `tests/validation` 的评分测试已归入 `tests/evaluation`，保留原断言和独立参考。`test_optin_*` 覆盖正式包提供的可选接口，不代表默认生产方案。独立解析解、NumPy 参考计算和故障负例保留在测试中，不能改为同一实现的回调自证。研究专属测试位于本地研究工作区，不参与正式收集。

Windows 本地有界调用使用单 CPU、180 秒、4 GiB，可按明确范围拆分：

```powershell
python scripts/run_bounded_tests.py tests/infrastructure tests/experiments tests/evaluation -q
python scripts/run_bounded_tests.py --module zhenmode mms
```

全球回归使用 `scripts/capture_global_regression.py` 和 `scripts/compare_global_regression.py`，完整输入、源码身份、数值记录和结果保存在独立运行目录，拒绝覆盖。重启小例使用 `scripts/verify_production_restart_cpu.py --output <新目录>`。超时或资源不足属于失败或未完成，不能推断数值通过。
全球采集器默认 `--layout product`；历史 `ocean_solver` 版本使用 `--layout canonical` 或 `legacy`，并必须指定独立冻结的 `--source-root`。它只观测原版本，不安装旧包转接层。

自动 CI 只运行 Ruff。完整测试和 MMS 通过 `workflow_dispatch` 的 `full_validation` 人工启动。数值测试通过不等于长期气候效果或公平加速结论。

海气交换新增组件沿用上述有界运行器，见[benchmark规范](benchmark_spec_zh.md)。独立参考驱动在`tests/support/ncar_reference.f90`，数值和来源hash在同名JSON；复算时从JSON的固定作者commit获取原始`bulk-ncar.F90`到忽略的工作目录，先核验SHA256，再以`gfortran -cpp -O0 -o reference tests/support/ncar_reference.f90 WORK/bulk-ncar.F90`编译，不定义`OGCM_LYCOEF`或`OGCM_CALHEIGHT`。输出列为θ/qair/Ts/qs/wind/Cd/Ch/Ce/参考潜热。参考只检验系数和饱和湿度，不能证明整个新通量与作者实现完全同义。该小参考的编译不等于MOM6完整构建已通过。

MOM6系数对照在Linux中使用`python scripts/check_mom6_coefficients.py --source PINNED_COUPLER/surface_flux.F90 --reference tests/support/ncar_reference.json --fms-include BUILD/build/fms --output NEW_OUTPUT`，外层须施加单CPU、180秒和4GiB上限。它使用实际FMS常数模块，编译固定源码与变换后的源码，不调用ZhenMode系数实现生成期望值；原始LY2004是规则差异控制，不能把差异解释成MOM6整体模式误差。完整无冰模块检查使用`python scripts/check_mom6_surface_exchange.py --source PINNED_COUPLER/surface_flux.F90 --reference tests/support/ncar_reference.json --fms-build BUILD/build/fms --output ANOTHER_NEW_OUTPUT`，链接真实FMS并执行初始化与实际surface_flux入口。报告包含源码、独立参考、驱动、依赖与二进制hash；SIS2潜热和海冰不在该检查范围内。

原始强迫通过`zhenmode benchmark fetch-jra --catalog ESGF_COMPLETE_FILE_RESPONSE.json --destination LOCAL_DATA --year 1958`获取。catalog须包含全部响应记录；冻结source_id、20190429发布、11个变量及发布方SHA256。中断保留partial和failed收据；续传核查HTTP区间，最终仍核对完整文件SHA256，已有损坏输入不覆盖。每次调用保存独立append-only尝试日志，当前状态文件只是索引。完成下载仅代表原始字节可用，还需原生空间映射、时间窗口和机制核验。原始JRA许可及引用要求保留在输入资料中，不能用其他版本替代。

原生矩形窗口：`zhenmode benchmark prepare-forcing --acquisition LOCAL_DATA/acquisition-1958.json --grid NATIVE_GEOMETRY.npz --runoff-area ORIGINAL_AREA.json --output NEW_INPUT --start 1958-01-01T00:00:00 --end 1958-01-01T06:00:00`。外层仍须单CPU/180秒/4GiB限制。网格声明`lon/lat/lon_bounds/lat_bounds/area/wet_mask`，可加原地形推导的`land_fraction`；面积引用声明path/variable/bytes/sha256。天气双线性、均通量球面面积、排水沿岸kg/s映射分别记录，不提供tripolar映射或缺测归一化。原始面积按作者说明采用整格面积，不能直接照抄有误的CMOR `cell_measures`标签。输出不授予海洋执行资格；制造数据仍为manufactured，错误单位、许可缺失、远距离非零排水和改写原始字节拒绝。
窗口末端超出该年最后一条天气记录时，重复`--acquisition LOCAL_NEXT_YEAR/acquisition-1959.json`接入下一年的插值端点。每份收据仍须完整核验v1.4.0的11份原始文件；时间记录显式合并排序，不在内存拼接完整三维场。源grid、bounds、许可和资料角色须一致，重叠、断档和缺端点拒绝；逐记录保留来源hash/索引。跨年制造窗口仅证明插值与区间积分，不算年度模式运行。原生area独立按bounds和6371km半径核对，Ctrl-C保存failed收据。

潜热表达式检查：`python scripts/check_mom6_latent_energy.py --examples PINNED_CHECKOUT --fms-build BUILD/build/fms --output NEW_ENERGY_CHECK`。编译实际暂存的SIS2/MOM/耦合库存源码片段，独立SI期望值不由被测实现生成；此检查不执行完整耦合或大气—冰顶部预算。
该检查还核对固定`SIS2/src/ice_model.F90`并执行其实际内部能量→W/m²导出语句及新耦合传递调用，采用不同的SIS/MOM单位缩放。单格交换仅提供恒等存储，不验证全球重分配；不能把`Ice%flux_lh`再除一次潜热换算因子，它在公开边界已经是W/m²。

MOM6时间格式适配：`zhenmode baseline mom6 omip2-time-inputs --prepared-input NEW_INPUT --output NEW_FMS_INPUT --dt-atmos 3600 --dt-cpld 3600`。输入须是上述空间准备的完整收据。生成Gregorian、1958年参考时刻、unlimited时间维及明确的FMS区间属性，保留实际日期和物理通量值。雨雪/辐射按大气步末读取，径流/陆冰按耦合步首读取；这些时相来自固定coupler源码，不是可随意互换的标签。步长须整除相应原始区间，输出目录拒绝覆盖。

准备前还核对窗口内所有均通量源边界与实际调用步长的相位：只核对窗口长度不够；例如每小时步长的00:30–05:30会跨越03:00源边界而被拒绝。01:00–05:00可在源区间内开始且不跨界，允许准备。数值复制采用至多8MiB多轴块，连大单记录也分块；完整年度的墙时仍未验证，不因内存改为流式就承诺180秒内完成。每份文件复制前后重新哈希，并核对inode、大小、mtime/ctime；源文件变动或Ctrl-C保留failed收据。

实际FMS时间检查：`python scripts/check_mom6_time_inputs.py --source PINNED_FMS/time_interp/time_interp_external2.F90 --fms-build BUILD/build/fms --fms-include PINNED_FMS/include --output NEW_TIME_CHECK`。在Linux中由外层施加单CPU/180秒/4GiB限制；编译完整、固定来源的FMS读取模块并链接真实FMS库。独立制造值检查区间均值、普通线性插值、单记录日均及六小时累计能量；跨界、越界、错误单位和旧负时刻编码均须拒绝。1958年资料不能直接以1970年参考时刻的负增量交给该FMS版本。此检查不运行耦合海洋，也不授予完整case资格；实际case还须绑定相同的时钟、输入表及构建收据。

为同一命令追加`--original-prepared NEW_INPUT --native-prepared NEW_FMS_INPUT`并选择另一新输出目录，可独立比较真实1958年首六小时、每小时的11字段。先核对两层收据及每个实际文件；参考直接从适配前CF时刻/bounds和值计算，检查所有格点，同时保留三个样本与全场求和，不调用被测读取器生成期望值。实际Fortran输出为native-real64列优先stream，检查器核对完整长度、逐值误差及输出hash。输入是制造数据时仍标记manufactured；该小检查固定首窗和一小时步长，不能冒充任意年度或完整耦合验证。

内部调用直接引用所属模块；结构见 [方法与架构](production_architecture_zh.md)和[准备与耦合导读](preparation_coupling_zh.md)。离线准备使用 `zhenmode.preparation.*`，原生 SIS2 使用 `zhenmode.coupling.sis2`，有界风驱动运行使用 `zhenmode.execution.wind_run`；现有 CLI 命令与参数保持。旧源码的运行和严格 checkpoint 需检出对应 Git 提交，不通过路径别名或替换 hash 绕过身份检查。

非线性热力学组件和固定 GSW/MOM6 参考检查见[温盐与密度](thermodynamics_zh.md)。日常测试用 `python scripts/run_bounded_tests.py tests/fd/test_teos10.py -q`；完整 Fortran 复算在 Linux 外层使用同样资源上限。组件尚未切换生产状态变量或 EOS。

CT/SR参考变体的转换与FD接线检查使用`python scripts/run_bounded_tests.py tests/fd/test_temperature_conversions.py tests/fd/test_thermodynamic_coupling.py tests/fd/test_online_surface.py tests/fd/test_teos10.py -q`。实际GSW复算追加`--temperatures`，核对全部22份原始例程、CP0及实际包源码，输出精度、来源、失败控制和范围。该变体目前只由明确的Python API接入；普通CLI及默认预设仍用线性状态方程。不要以无冰组件的运行代替完整协议短窗。

WOA13v2原始网格转换使用 `zhenmode benchmark prepare-initial-source --acquisition WOA/acquisition.json --pressure-reference PRESSURE/pressure.json --output NEW_OUTPUT`。外层施加单CPU、180秒、4GiB上限；逐层读取官方年度 `t_an/s_an`，保留成对原始缺失掩膜，生成MOM初始化需要的位温 `ptemp` 及参考变体的 `ct/sr`。PSS-78单位 `1` 保留数值，不当成kg/kg缩放。温盐必须使用相同年气候time、0至12月climatology bounds；保留年零的原始气候月份编码，不把它转成1958年天气时钟。

压力JSON字段精确为 `path/variable/bytes/sha256/units/definition`，units为dbar；所引NetCDF有与原WOA相同depth/lat坐标及二维 `p[depth,lat]`，有限、单调、表面0、0至8000dbar。压力定义必须显式给出；例如固定rho0*g*depth/10000是Boussinesq参考近似，不能写成精确地理压力。准备器核实际输入及完整执行包身份，收据和输入在执行结束后再次核验；中断或变化保存failed，已有目录拒绝覆盖。

独立源检查使用 `python scripts/check_woa_initial_source.py --prepared NEW_OUTPUT --acquisition WOA/acquisition.json --pressure-reference PRESSURE/pressure.json --oracle TEMPERATURE_ORACLE --output NEW_CHECK`，在Linux外层施加相同资源上限。oracle为前述固定原始GSW `--temperatures` 检查输出目录：核其源码/驱动/二进制，再逐层核所有掩膜，预先固定种子每层至多8个样本，经原始Fortran另算SP→SR及PT/CT，并要求改错0.01°C的参考触发失败。此检查验证转换/写出，不认证共享数学系数或压力生产者；样本检查不等于每个温度数值均已与GSW比较。

这些输出仍是原始网格源产品，`native_initialization_ready=false`、`execution_ready=false`。原生重映射、湿柱支持、补值/外推、浅海/底部、完整水柱体积和初始库存需单独准备验证。MOM所读盐度为原SP，参考变体采用SR≈SA；不能将同一文件改标签来混用。

原生点场准备命令是 `zhenmode benchmark prepare-native-initial --source-prepared WOA_CONVERSION --geometry RECTANGULAR_GEOMETRY --nodes-file NODES.json --output NEW_NATIVE`。`NODES.json` 仅含 `z_nodes_m` 数组，负值向下；几何目录含身份匹配的 `geometry.json/grid.npz/bathymetry.npz`，面积另从球面bounds核对。当前入口要求与WOA相同水平坐标，只进行经度周期重排、垂向点值线性插值、同层湿域平滑补值及最深源以下零梯度延拓。PT/SR先映射，之后由PT/SR生成等价CT及供MOM读取的SP；不将CT改标签当位温。初始化时刻为1958-01-01，源年度气候time另存收据。

显式 FD 固定部分单元候选的接触、压力、容量和局部检查范围见
[固定参考部分单元](fixed_partial_cells_zh.md)。局部算子核验不授予完整 case 资格。

输出含三维固定参考厚度、PT/CT/SR/SP点场、原始成对支持／水平补值／深层延拓／未解决掩膜、最近原始锚点及湿图路径距离、每层补值线性系统和未知位置、源码／输入身份和已解决支持上的库存。最近锚点不是平滑值的唯一供体，实际平滑值由完整边界锚点和保存的系统定义。无锚点分量保留缺失并返回退出码3；相应库存明确是部分库存，不能当成全球初态。MOM仍须独立完成原生层重映射及库存检查；FD运行仍须统一水平面积度量、补全跨节点参考面联系与压力平衡，并完成全域稳定性和过程资格。即使点场全部有限，收据的完整模式执行资格仍为false。ETOPO的既有CDO来源状态随收据保留；负高程不能代替独立海陆分类。

直接模型运行拒绝已存在的 `global_<tag>.npz`，请为新运行选择独立 `--tag` 或 `--out-dir`。受中断的严格 checkpoint 续跑仍核对原配置与源码；已完成或已保存失败结果的目录不会被覆盖。
最终结果先写入同目录临时文件并 fsync，再以原子硬链接发布，拒绝覆盖竞争写入；checkpoint 则原子替换。文件系统须支持同目录硬链接；不支持时明确失败。发布前写入中断不会留下残缺的最终结果，异常退出可能留有独立临时文件，不阻止同名 checkpoint 续跑。
