# Case、预设、实验、扫参与运行记录

工程入口围绕唯一的既有全球 FD 生产方法。实验管理包 `zhenmode.execution` 只负责配置展开、输入接线和运行记录，数值执行仍进入正式 `runtime.entry → application → model.factory → timestepping`。MOM6 使用独立 baseline 构建与执行契约，但共享物理问题必须引用同一 case，并核验实际输入和网格。

安装后，从仓库根目录可执行：

```powershell
zhenmode experiment validate experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml
zhenmode experiment expand experiments/zhenmode/global-045deg/global-045deg-seasonal-baseline.yaml --output expanded-global045.json
zhenmode sweep expand experiments/zhenmode/global-045deg/sweep-vertical-mixing.yaml --output sweep-plan.json
zhenmode experiment run experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml --dry-run --outputs outputs/proposed
zhenmode runs list --outputs outputs/proposed
```

`expand --output` 和 sweep 输出是 create-only；已经存在的文件会被拒绝。`run --dry-run` 冻结一个 `proposed` run，不积分。扫参仅展开，不提供自动批量启动命令。资源估计给出网格格点数、积分步数、格点步数和状态数组存储下界；未有实测校准的运行时间显式为 `null`。

有界数值见证命令如下，实际执行状态和成绩以交付验证报告为准：

```powershell
zhenmode experiment run experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml --evaluate --outputs outputs/smoke
```

托管本地运行强制 1 CPU，180 秒墙钟和 4096 MiB 上限。Windows 使用 Job Object 限制 CPU affinity 和内存，并在关闭 job 时终止子进程；Linux 使用 CPU affinity 和 address-space limit。不支持可靠限制的平台会明确拒绝。大型真实网格和长积分应先形成独立资源计划；目前的全球预设只验证展开，不自动重跑 365 天或百年。

仓库外使用相同安装命令，传 `--root <仓库绝对路径>`，配置路径可以相对 root；`--outputs` 相对当前工作目录或使用绝对路径。worker 在每次 run 的独立目录启动，不依赖研究目录、测试 fixture 或 `src` 临时路径。

每个 YAML 声明 `schema_version: 1`，拒绝未知字段与重复 key。数量写为 `{value: 1800, unit: s}`，要求规范单位精确匹配，不隐含换算。case 数量使用 `s`、`deg`、`m`、`degC`、`psu`、`N/m2`；method 选项使用生产参数相应的 `m2/s`、`m4/s`、`W/m2/K`、`day` 等。duration、输出和 checkpoint 必须与 `dt` 严格对齐。最后不足一个输出间隔的终点也列入最终输出时刻，和生产积分一致。

case 固定共同物理条件和数据引用；preset 保存方法特有离散配置；experiment 引用 case、parent preset，包含稳定 ID、中文名称、目的、method、variant、变化清单和评价协议。单因素消融 `single_factor` 只允许一个明确的生产参数变化，`before` 必须与完整 parent 展开值一致。多参数使用 `combination`；普通调参使用 `tuning`。同一参数多次覆盖、静默覆盖父预设、未声明变化、循环继承及和 case 不一致的协议身份全部拒绝。

config hash 取最终有效物理和方法配置，不取实验中文标题、目的、实验 ID 或解析后的 provenance 机器绝对路径。case 中声明的 `data[].path` 及数据版本、预期校验值参与 hash；如果在 case 中显式更换为另一条绝对缓存路径，hash 也会改变。相同最终配置的多次运行仍有不同 run ID，互不覆盖。`expanded.json` 同时保留解析后的全部生产 CLI 默认、分辨率自动缩放、实际快步时长、参数化、共同 case、数据引用、配置来源和协议内容。

manifest 的执行状态为 `proposed / running / completed / failed`。一次积分完成但生产诊断出现漂移 FAIL，可以是 `completed`，同时 `production_gate` 为失败；异常退出、超时和未完成积分为 `failed`。`acceptance` 与 `comparability` 由评价流程单独写入，不能从进程退出码推导技巧达标。

`run --evaluate` 对积分完成的 run 使用冻结协议直接生成 `run/evaluation/report.json`、`metrics.json`、`report.md` 并写回 manifest。未完成运行不评分；评分拒绝写 `evaluation.status: failed`，保留 `execution_status: completed` 并让 CLI 返回非零。协议未声明指标门槛时验收状态为 `not_declared`，不等于 passed。

每次 run 固定安装包全部 `.py` 来源校验值及 Git 提交/工作区状态；worker 运行前验证整包身份，运行后列出实际加载文件及字节校验值。外部数据按实际 loader 角色、路径和冻结内容核对，运行后再次核对；风月份角色互换不能通过路径集合的等价检查。输出 NPZ 和 `actual-physical-grid.npz` 都绑定实际 SHA256；后者保存真实经纬度、深度、湿掩膜及几何量，供共享输入协议核查。

当前输入引用中尚未取得原始外部缓存的预期 SHA256，明确标为 `pending`。运行时可以固定已在本地的实际缓存并保存所观察内容，但这不会把它自动变成原历史运行数据。历史报告提交、完整执行源码身份和当前独立复跑仍是不同证据。旧 raw/A2 协议身份不变，也不与新面积加权 v2 成绩混排。

合成 80 秒问题只验证真实生产组装和记录契约，没有海洋效果资格。其全湿 8×8×4 网格来自安装包输入服务，不引用 tests 或 research；经向间距 45°、纬向间距约 8.5714°，实际网格参数从组装阶段注入并记录。评价不可将它与真实全球积分或 MOM6 tc1 排名。

验证命令：

```powershell
python -m pytest tests/experiments -q
```

这组测试不运行 JAX 数值，覆盖未知字段、重复 key、错误单位、循环继承、歧义覆盖、单因素/组合定义、case/协议负例、sweep 只展开、不覆盖 run、真实实现来源覆盖、失败状态保持和执行完成不等于验收。`test_evaluation_bridge.py` 直接串联配置展开、run manifest 与真实评分器，用明确声明的数组 fixture 核对协议文件字节及规范内容身份，并拒绝冻结后改变协议；它不是模式数值参考解。
