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

`tests/fd`、`tests/runtime`、`tests/data` 对应模型与输入；`tests/baselines`、`tests/evaluation`、`tests/experiments` 对应接入、评分与实验契约。`test_optin_*` 覆盖正式包提供的可选接口，不代表默认生产方案。独立解析解、NumPy 参考计算和故障负例保留在测试中，不能改为同一实现的回调自证。研究专属测试位于本地研究工作区，不参与正式收集。

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

实际FMS时间检查：`python scripts/check_mom6_time_inputs.py --source PINNED_FMS/time_interp/time_interp_external2.F90 --fms-build BUILD/build/fms --fms-include PINNED_FMS/include --output NEW_TIME_CHECK`。在Linux中由外层施加单CPU/180秒/4GiB限制；编译完整、固定来源的FMS读取模块并链接真实FMS库。独立制造值检查区间均值、普通线性插值、单记录日均及六小时累计能量；跨界、越界、错误单位和旧负时刻编码均须拒绝。1958年资料不能直接以1970年参考时刻的负增量交给该FMS版本。此检查不运行耦合海洋，也不授予完整case资格；实际case还须绑定相同的时钟、输入表及构建收据。

为同一命令追加`--original-prepared NEW_INPUT --native-prepared NEW_FMS_INPUT`并选择另一新输出目录，可独立比较真实1958年首六小时、每小时的11字段。先核对两层收据及每个实际文件；参考直接从适配前CF时刻/bounds和值计算，检查三个格点与整个矩形场求和，不调用被测读取器生成期望值。输入是制造数据时仍标记manufactured；该小检查固定首窗和一小时步长，不能冒充任意年度或完整耦合验证。

内部调用直接引用所属模块；结构见 [方法与架构](production_architecture_zh.md)。旧源码的运行和严格 checkpoint 需检出对应 Git 提交，不通过路径别名或替换 hash 绕过身份检查。

直接模型运行拒绝已存在的 `global_<tag>.npz`，请为新运行选择独立 `--tag` 或 `--out-dir`。受中断的严格 checkpoint 续跑仍核对原配置与源码；已完成或已保存失败结果的目录不会被覆盖。
最终结果先写入同目录临时文件并 fsync，再以原子硬链接发布，拒绝覆盖竞争写入；checkpoint 则原子替换。文件系统须支持同目录硬链接；不支持时明确失败。发布前写入中断不会留下残缺的最终结果，异常退出可能留有独立临时文件，不阻止同名 checkpoint 续跑。
