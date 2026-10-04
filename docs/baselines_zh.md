# MOM6：固定来源、隔离构建、真实运行、转换与报告

MOM6 是外部对照，生产 FD 不依赖其代码。`ocean_solver.baselines.mom6` 只做环境检查、固定源码获取、build cache、原生输入准备、有界运行与产物绑定。评分复用 `evaluation`，不复制评分器、不发展材料库存候选或新的动力核心。

## 固定版本与环境

| 项 | 固定身份 |
|---|---|
| MOM6 | `f49a00096df607b48354603e2398e14e189fd62e` |
| FMS 2023.03 | `527a42f3ff36d75aac68b65757b35f73a74146bb` |
| CVMix | `fce422195a0c58f15a55946b5ed517ba4365e232` |
| GSW | `29e64d652786e1d076a05128c920f394202bfe10` |

这些版本来自既有构建记录，这些身份保存在仓库 pins 文件中。`MOM6-examples a5ebafe...` 保留为历史 reference；tc1 使用 MOM6 本身 `.testing/tc1`，不依赖 examples 来生成另一套源码。源码抓取、构建路径在仓库外，以 commit和cache明确身份。

`doctor/prepare` 记录源码、子模块、构建配置、实际编译器及依赖库身份。缓存路径由调用者显式指定。

无完整 `build_manifest.json` 或其绑定不一致时 `build_provenance=unverified`。干净 pinned source + 一份 binary hash 不能证明该 binary由它构建。新的 build 保存实际命令、flags、source与dependency身份；环境变化会失去已记录构建资格。工具不安装全局软件、不改系统安全或凭据。

## 完整 tc1 小例

先在 Linux/WSL 的 Python3.12+ 环境安装本项目。需要 git、CMake、make、autoreconf、MPI Fortran 和 NetCDF-Fortran；缺项会明确报错。

```sh
zhenmode baseline mom6 doctor --cache "$HOME/.cache/zhenmode/mom6/f49a000"
# 全新缓存需要获取并构建；请先确认资源范围：
zhenmode baseline mom6 fetch --cache "$HOME/.cache/zhenmode/mom6/f49a000"
zhenmode baseline mom6 build --cache "$HOME/.cache/zhenmode/mom6/f49a000" --wall-seconds 1800
# prepare 返回独立 run_dir，可复用已核验的缓存：
zhenmode baseline mom6 prepare --cache "$HOME/.cache/zhenmode/mom6/f49a000"
zhenmode baseline mom6 run --run-dir RUN_DIR --wall-seconds 180
zhenmode baseline mom6 convert --run-dir RUN_DIR
zhenmode baseline mom6 evaluate --run-dir RUN_DIR
```

prepare逐字节复制4个官方参数文件，记录hash；独立 run ID含 UTC时间、配置hash和随机后缀，重复不覆盖。tc1保持上游 Mercator网格、10×8×8、dt900s、0.25日/24步、原生0/0.125/0.25日诊断；不伪称与全球物理问题相同。

run强制1个CPU affinity、1个直接MPI初始化rank、180s墙时、4GiB地址空间/进程组RSS及128MiB输出守卫；退出、超时、异常清理均保存failed，清理子进程组。完成时立即冻结原生 `.nc` hash；第一次convert也不能接受被替换的输出。执行每次只一次，重跑必须新prepare。转换检查原生统计单位/时序，报告 CFL、velocity截断、完成和有限性、内容首末漂移；不会从此小例生成SST RMSE或工业排名。

## 两个共同全球 case 的原生接入

`configs/mom6/presets/global-050deg-wind-only-30d.yaml` 与 `global-050deg-restoring-30d.yaml` 引用同名 `cases/`，不是独立复制问题。原始资料来自已有30日 wind-only/restoring记录；这些接入预设不声称他机原生MOM_input已恢复，不猜历史dt、不推荐新参数。`required_inputs` 文件名是明确的接入约定，不同的真实原生布局应另存预设。

```sh
zhenmode baseline mom6 prepare-native \
  --cache "$HOME/.cache/zhenmode/mom6/f49a000" \
  --root REPO \
  --case REPO/cases/global-050deg-wind-only-30d.yaml \
  --preset REPO/configs/mom6/presets/global-050deg-wind-only-30d.yaml \
  --native-dir EXISTING_NATIVE_INPUT_DIRECTORY \
  --reference-npz SHARED_INITIAL_AND_ACTUAL_GRID_REFERENCE.npz
```

输入目录应有4个native launch文件与INPUT/。共同 reference 必须保存 `T_init/S_init/depth/wet_mask/lat/lon`，使用生产实际处理后的深度/掩膜；仅同一 raw ETOPO版本不够，smooth80/min_depth500等处理也改变真实物理网格。worker另保存 actual-physical-grid侧车，不能拿原始地形替代。未备齐文件即拒绝，不下载或默默换成别种强迫。

preflight读取本版本solo driver真实role→file/variable参数，检查其实际相对INPUT路径、变量、单位、原生维度/坐标、海深/湿区和初始T/S逐值一致。固定核心声明包含720×260×14、球面范围、风文件/无浮力或规定恢复等。实际DT/DTBT/DT_THERM记录，未指定的物理保留原生resolved params和完整输入hash；不能由少量选项相同推论完整物理等价。风变换、垂向抽样、热盐恢复容量及完整参数化仍为受限比较，`effective_physics_sha256=null` 明确阻止公平排名。

准备成功后使用同一run/convert/evaluate链。默认180s守卫不足完成既有约50min的4-rank 30日全球例时会记录failed/未完成；扩大 MPI、墙时或启动长算例前应先明确资源范围。需要长积分的资源计划、外部运行器和完整日志，必须在后续明确批准范围中实现与核查。

非tc1转换注册实际 `prog.nc/ocean_geometry.nc`，不隐含重网格。评价统一核实际reference/geometry绑定和协议两种hash，复用外部共享评分；原生中心或湿区不一致拒绝。历史 30 日报告保留原协议和来源身份，不自动升级为 v2 或当前源码的独立复跑。

## 强迫导出

```sh
zhenmode baseline mom6 export-forcing --kind wind --bathy data/ETOPO_2022_v1_r3600x1800_surface.nc.npz --out outputs/mom6-inputs/wind.nc
zhenmode baseline mom6 export-forcing --kind air-temperature --bathy data/ETOPO_2022_v1_r3600x1800_surface.nc.npz --out outputs/mom6-inputs/air.nc
zhenmode baseline mom6 export-forcing --kind sensible-heat --bathy data/ETOPO_2022_v1_r3600x1800_surface.nc.npz --out outputs/mom6-inputs/sensible.nc
```

三种导出共用 `baselines.forcing`，保留已有 A-grid 转置、fp32 字段及每月 `16 + 30*m` 天的 Julian 时间映射。浴深和输出路径必须显式提供，已有输出拒绝覆盖；WOA 和空气缓存按正式读取器配置。sensible-heat 是 `λ × (空气温度 − 固定初始 SST)` 的代理，不是运行时空气海洋耦合，也不自动满足公平比较的输入合同。原 `interop.mom6.*` 三个脚本入口已退役。
