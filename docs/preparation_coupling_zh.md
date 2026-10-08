# 输入准备与原生耦合导读

离线资料准备、模型运行时输入和外部组件耦合分别承担不同工作。它们共用文件身份与必要的网格／热力学工具，数值公式和生效配置由实际调用方决定。

## 文件职责

| 文件 | 职责 | 现有入口 |
| --- | --- | --- |
| `preparation/acquisition.py` | 固定 JRA 发布资料的选择、下载、续传及逐次收据 | `benchmark fetch-jra` |
| `preparation/forcing.py` | 将已核验 JRA 窗口映射到矩形网格，处理天气、均通量和沿岸排水 | `benchmark prepare-forcing` |
| `preparation/woa.py` | 原始年度 WOA 成对温盐及压力核验、PT/CT/SR 转换，保留缺测 | `benchmark prepare-initial-source` |
| `preparation/coast.py` | 按显式政策组合海岸、海深和连通证据，准备整格海陆几何 | `benchmark prepare-native-geometry` |
| `preparation/native_initial.py` | 准备固定参考水柱点场、同层补值和已批准深层延拓，保存支持与库存 | `benchmark prepare-native-initial` |
| `preparation/bottom.py` | 按逐项审查记录补齐底段温盐，保存来源和影响 | `benchmark complete-native-bottom` |
| `preparation/fd.py` | 写出并读取 FD 固定部分单元的 CT/SR、压力、面积和容量输入 | `benchmark prepare-fd-initial` |
| `coupling/geometry.py` | 准备与 FD 输入匹配的 SIS2 supergrid、海深和显式耦合配置 | `benchmark prepare-sis2-case` |
| `coupling/sis2.py` | 核验并链接固定原生对象，管理持久 SIS2 进程和精确交换格式 | `benchmark compile-sis2-bridge`、`NativeSIS2` Python API |
| `coupling/sis2_bridge.f90` | 组织原生 SIS2 调用和二进制字段交换，不重写 SIS2 物理 | 由链接命令使用随 wheel 安装的驱动 |
| `execution/wind_run.py` | 启动有界 CUDA 风驱动组件，保存预算、接受步和严格重启 | `benchmark run-fd-wind` |

表中 Python 文件均位于 `src/zhenmode/` 下。命令前缀为 `zhenmode`；模块直接从所属路径导入，没有旧 `execution.native_*` 转接层。MOM6 专用 tripolar 网格和 FMS 时间格式仍由 `baselines/mom6` 管理。

`provenance/sources.py` 统一核验所引文件和采集完整 Python 源码哈希。每次采集都重新读取实际文件，不缓存结果；准备结束及发布时的再次校验保留。原生 Fortran 驱动、对象、库和可执行文件继续由 SIS2 链接收据独立核验。

## 实际链路

- 原始 JRA 获取 → 窗口与空间准备 → `model.inputs.forcing.jra55` 运行时采样 → 明确选择的运行入口。
- 原始 WOA 转换 + 海岸几何 → 原生水柱初态 → 审查补底 → FD 输入产品 → 实际 FD 工厂。
- 固定 MOM6/SIS2/FMS 构建 + FD 输入 → SIS2 共享网格准备与驱动链接 → 持久 `NativeSIS2` 进程 → `model.solver.physics.sis2` 在 GPU 上施加海洋交换。

准备成功表示对应资料产品满足其校验要求；风驱动组件仅施加风应力。完整冰海运行的过程选择、共同物理条件和比较范围仍需由具体 case 与运行记录说明。

## 迁移与复现

此次调整保持现有 CLI 参数、数值公式、补值政策和评分语义。迁移前的严格 checkpoint 仍须在原 Git 版本读取；新源码身份使用实际新路径，不改写历史收据，也不绕过来源校验。本地 `research/`、原始数据和历史产物保留原字节，不参与本次迁移或正式安装。

迁移清单和验证范围见[整理记录](repository_organization_20261008.md)。
