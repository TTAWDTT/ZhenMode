# 真实全球回归验证（2026-10-03）

本次验证使用原工作区的实际 ETOPO NetCDF、WOA2023 温盐 NetCDF、
2023 年十二个月风缓存和年度空气温度缓存。不是 worktree 的合成浴深。
实际数据 SHA256、完整展开参数、执行源码和资源 receipt 见
[机器报告](global_regression_validation.json)。数据身份是当前本地观察值，
不能据此补全原历史运行缺失的数据身份。

整理前参考为 `25258950905f9d1aa84509c4c99ebad9ef33ba2b`，从其 Git archive
单独冻结全部 `src`，逐文件核对 Git blob；未编辑历史源码。
当前正式包为 `9cb3c63907f83097b6e7826ff41ea0887277a700` 的安装 wheel，
106 个实际源码文件与 checkout 逐字节一致，无 research 或 editable 安装。
两侧在独立进程中使用同一个 Python/JAX/NumPy 环境、单 CPU 和相同输入。
旧 reader 显式传入缓存目录，适配及 observer 身份单独记录。
机器报告中重复的源码清单通过 `source_inventory_sha256` 引用
`shared_source_inventories`，保留每个实际文件 SHA；数值数据不归一化。

## 0.45° 原配方的失败与资源边界

| 调用 | 结果 | 墙钟 | 峰值 job private |
| --- | --- | --- | --- |
| 当前 `2069035` 完整入口 | 进入积分前缓存接线失败 | 96.782 s | 约 0.98 GiB |
| 修复后的完整入口 | 初态/强迫完成；首次编译超时，0 个接受步 | 180.188 s | 约 2.02 GiB |
| 整理前独立初态准备 | completed，未积分 | 92.937 s | 约 0.98 GiB |
| 当前独立初态准备 | completed，未积分 | 96.844 s | 约 0.98 GiB |
| 整理前从冻结初态积分 | 首次编译超时，0 个接受步 | 180.312 s | 约 3.69 GiB |

800×288×14 网格的温度、盐度、经纬度、层位和湿掩膜六组初始数组逐字节一致。
没有完成 0.45° 积分对照，不能宣称其轨迹已复现或数值不稳定。
用户选择保持单 CPU / 180 s / 4 GiB，并改做 1° 全球回归；未启动更大预算。

发现并单独提交的缓存接线修复见
[修复说明](external_cache_binding_fix_zh.md)。40 项相关测试通过；用 Git 中原 reader
函数定义执行相同选择缓存测试时，三个负例均失败，确认测试能够检测原缺陷。
修复不改变数值公式或默认物理参数。

## 1° 诊断配方

经度全球周期、纬度 ±65° 截断，实际 360×130×14，fp32，dt=1800 s。
沿用原配方的 FCT、局地对流、投影、季节风、年度空气温度及冰点下限等物理选项。
分辨率改为 1° 后按生产规则显式展开：`dt_bt=150 s`、12 个快步、
`nu_bi=2e14 m4/s`；原显式 `nu_h=2e6 m2/s` 保持。
它是用户选择的工程诊断配方，不冒充历史 0.45° 的 RMSE 复现或新推荐。

两侧分别使用自己的原生网格/WOA 插值实现准备初态，耗时 94.047 / 89.016 s。
六组数组逐字节相同后才选取同一冻结初态做积分。
这将准备与积分拆为清晰的有界阶段，没有改变输入算法，也没有隐含重网格。

两小时四步对照已通过：初始及四个后续时刻的六个全状态变量一致，
独立 NPZ 读取核对最终状态、生产诊断和 checkpoint 共 99 项数值记录逐字节相同。
参考 / 当前墙钟为 77.281 / 75.469 s，峰值 job private 约 1.22 GiB。
第一步含延迟编译；后续单步约 3.2 s。这些是资源记录，不是公平速度排名。

日志收尾曾暴露 observer 在计算日志 SHA 后追加摘要的问题。
原 `completed.json` 不改字节，新增 `completed-final.json`：独立验证原 SHA
对应日志前缀，尾部恰为已记录的结束摘要，只更新日志身份；数值文件未改动。
后续 observer 在索引产物前关闭 Tee。原执行 observer 字节在机器报告保留。
比较器拒绝原校验失败；最终核对通过后还用副本做负例：文件改动、以及更新文件
身份后仅改变一个浮点位的温度，均被拒绝。

## 12 小时及全球重启

在两小时对照通过及单步成本明确后，将同一诊断配方扩展为 24 步、12 小时，
每三小时输出和保存 checkpoint；每次调用仍使用原资源上限。

| 范围 | 结果 | 墙钟 | 峰值 job private |
| --- | --- | --- | --- |
| 整理前连续 24 步 | completed，生产短窗门槛 PASS | 141.203 s | 约 1.22 GiB |
| 当前连续 24 步 | completed，生产短窗门槛 PASS | 141.031 s | 约 1.22 GiB |
| 当前从第 18 步续跑到第 24 步 | completed，6 个新增接受步 | 80.687 s | 约 1.24 GiB |
| 前后数值核对 | 25 个时刻 × 6 个状态变量一致，99 项数值记录逐字节一致 | 0.906 s | — |
| 连续/重启核对 | 七个重合时刻全状态一致，最终及完整诊断历史 71 项逐字节一致 | 0.703 s | — |

重启加载的是当前版本自己的第 18 步 checkpoint，来源、参数、网格、强迫及
文件内容严格校验；没有加载旧源码 checkpoint 绕过来源合同。
连续/重启比较包括全部生产 NPZ 字段及源码元数据，没有任何归一化。
12 小时运行的日志收尾校验直接通过，不需要两小时阶段的独立收尾修正。

当前短窗的后续单步中位数约 3.182 s。仅作本机 1° CPU 资源规划的线性外推，
纯积分一年约 15.5 小时，百年约 64.5 天；不包含准备、编译、IO 或长程状态变化，
不适用于 0.45°、GPU，也不是全年/百年实测。没有据此启动这些任务。

## 运行命令与证据边界

每次调用外层使用 `scripts/run_bounded_research_tests.py`，1 CPU / 180 s / 4 GiB。
本机独立产物目录为
`C:/Users/86153/.codex/tmp/zhenmode-global-regression-20261003T085444256440Z`。
其 `plan-1deg.json` 是初态准备方案，`plan-staged-1deg.json` 是两小时方案，
`plan-staged-1deg-12h.json` 是随后较长窗口方案；全部参数和校验值也保存在机器报告。
`plan-restart-1deg-12h.json` 冻结了实际当前 checkpoint 的 SHA。
用 `$probe` 指向该目录，`$python` 指向其 `environment/Scripts/python.exe`：

```powershell
& $python scripts/run_bounded_research_tests.py --script scripts/capture_global_regression.py --plan "$probe/plan-1deg.json" --output <新的初态目录> --revision <源码提交> --layout canonical --prepare-only
& $python scripts/run_bounded_research_tests.py --script scripts/capture_global_regression.py --plan "$probe/plan-staged-1deg.json" --output <新的积分目录> --revision <源码提交> --layout canonical
& $python scripts/run_bounded_research_tests.py --script scripts/compare_global_regression.py <参考目录> <当前目录> --output <新的比较报告.json>
```

历史一侧另外传 `--layout legacy --source-root <独立冻结的src>`。
所有输出目录/报告 create-only，不覆盖前次运行。首次两小时运行使用独立完成的
日志收尾 receipt，比较时显式传 `--receipt-file completed-final.json`。
新机器需取得匹配 SHA 的数据、重新冻结 source 和计划中的绝对路径，不能把本机路径
当作公共数据版本；不得跨源码直接加载旧 checkpoint 绕过严格身份。

此验证不建立长期气候稳定性、280–365 天 SST RMSE、工业级资格或加速结论。
没有启动 GPU、MOM6、全年、百年或全套 CI；旧 raw/A2 与 v2 协议不变。
