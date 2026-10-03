# 2026-10-03 研究工程重构验收

本轮完成生产职责拆分、研究 distribution 隔离、共同 case、多套预设、实验/扫参/run 索引、统一评价及 MOM6 接入。验证支持既有小算例行为保持和工程链可运行；没有重新授予历史气候成绩、工业资格或速度优势。机器记录见 [validation.json](research_engineering_validation.json)，逐文件保全见 [preservation.json](research_engineering_preservation.json)，迁移与工程例外见 [layout.json](research_engineering_layout.json) 和 [production_migrations.json](production_migrations.json)。

## 基线、环境和来源

初始用户 checkout 是干净的 `ttawdtt/core-repair-review`，HEAD `82e1ca4a2158d4c0ed20f91c04e80cdf43f8a36b`。实际远端 main 为 `25258950905f9d1aa84509c4c99ebad9ef33ba2b`。在独立 managed worktree、分支 `ttawdtt/zhenmode-research-engineering` 实施，没有重置或覆盖原 checkout。旧 origin 名仍可用，但读写明确使用 canonical ZhenMode 仓库。

重构前通过 Git archive 冻结完整 main、原 2008 个收集节点、代表性非零 8×4×6 配置的 275 项全状态记录，以及生产驱动 8 个接受步、两次中断恢复的 344 项记录。基线和候选使用同一验证工具，分别加载各自独立源码；这里的前后回归不是独立科学 oracle。制造解、独立预设热源预算和错误注入负例承担独立正确性检查，未改成残差回填。

Windows：Python 3.12.10，JAX/JAXlib 0.11.2，NumPy 2.5.3，SciPy 1.18.1，AMD Ryzen 7 8845H。依赖版本见 [CPU 验证锁定文件](../requirements/validation-cpu-py312.txt)。MOM6 使用 Ubuntu 24.04 WSL、既有 fp64 二进制和隔离缓存。没有启动 GPU。

每次数值调用一 CPU、180 秒、4 GiB；Windows Job Object 约束整个子进程树，receipt 保存墙时、RSS、private bytes 和退出状态。MOM6 一 MPI rank，另限 128 MiB 日志/输出；启动前已报告复用缓存小例的资源计划。没有放宽范围或重跑百年。

## 实测结果

| 范围 | 结果 | 实际边界 |
| --- | --- | --- |
| 测试 collection | 2101 节点，原 2008 全保留，新增 93 | 保留完整参数后缀，显式迁移表核对，不以总数替代匹配 |
| fp64 / fp32 生产记录 | 各 275 项逐字节一致 | 既有 legacy 时间方案、非零小网格和过程；不穷举全部 opt-in 配置 |
| JIT 驱动、输出、重启 | 344 项一致；连续与两次中断恢复一致 | 相同合成输入缓存；源码身份另存，不伪装成旧源码 hash |
| 工程合同组 | 236 passed、1 Windows skip | 实验、评价、MOM6、infra、生产职责；后补构建载荷 5 passed |
| POSIX 清理负例 | WSL 1 passed | mock 监控异常验证 kill/wait；没有运行模型 |
| 数据读取 | 116 passed、98 skipped | 当时缺浴深；随后合成浴深相关网格/输入组 155 passed |
| 既有评分回归 | 70 passed | raw/A2/面积口径各自保留，不混排 |
| 运行错误拒绝 | 33 passed | 参数、单位、缺输入及运行合同 |
| 独立预算负例 | 2 passed | 预设热源独立参考及内部扩散源污染能被检出 |
| MMS | operator 和二阶收敛 gate PASS | 独立解析检查；收敛实际误差/阈值保留日志 |
| wheel 外部目录 | 177 源文件逐字节核对；31 同对象桥、pickle/pytree、162 来源文件通过 | 完整载荷精确匹配，研究包及 9 研究桥均不存在 |
| 正式 wheel 全流程 | 外部目录 8 步 PASS，输出/重启/评分 completed | `acceptance=not_declared`；初始场参照 SST，不是气候观测精度认证 |
| MOM6 tc1 | 准备→运行→转换→报告通过，24 步、0.25 模式日 | 小原生稳定性报告；与全球 FD 不同问题，不能排名 |
| 历史保全 | 672 原文件、477 指定证据核对；5 历史 Git blob + 谱系通过 | 两导航更新和一研究 import 重接明确记录，原证据结论不重写 |
| lint / diff | ruff、Git whitespace 检查通过 | 完整远端 CI 状态独立查看，不能从本地分组声称全套通过 |

最终正式安装运行 ID：`20261003T063424577671Z-20d0205b8a52-31c8db26`。其配置 hash 为 `20d0205b8a52bb6a2c226454445c22b636442827bd292a51441d9d7d03881430`，完整生效配置、结果 hash、实际网格、执行源码/数据、环境及评分来源分别保存。端到端约 8.625 s；编译/纯积分/IO 未单独测量，报告 null。相同配置的重复运行另有不同 ID，不覆盖；资格比较不生成排名，成本分项不全时 `cost_comparable=false`。

最终 MOM6 ID：`20261003T062202Z-beb53b99eb44-c7277267`。本地原生端到端约 1.27938 s，聚合峰值约 58.7 MB；有限、完成、正质量、CFL 与无速度截断均通过。质量端点变化约 −1.7136e−10%，热约 0.00079956%，盐约 1.8566e−14%；这些是原生短例诊断，不是独立完整收支闭合。二进制 SHA256 `35743c5cc1068b5d19f7ae36d358a8e83d3da43c12429211a61e545c9f4caac9`，实际依赖库、Makefile、工具链和上游所有 tracked 源均记录。缓存没有可验证的本轮完整 build receipt，明确 `build_provenance=unverified`，没有补造构建日志。

## 遇到的失败与修正

1. 首次 editable collection 的 runner 没处理 `.pth`，产生 79 个收集错误。修正 venv 路径初始化及 scripts namespace 冲突后，全部原节点保留。
2. 初始工程合同组有 8 项路径/来源映射与别名问题，逐项修正，最终组 236 passed。旧断言的数值含义和独立参考保留。
3. 第一次真实生产运行 ID `20261003T060503623149Z-20d0205b8a52-b23a3a06` 完成且生产 gate PASS，但评分失败：协议文件字节 hash 与规范内容 hash 混用。保存 execution completed / evaluation failed；区分两种 hash 后新运行通过，不重写旧状态。
4. 驱动前后第一次有四项元数据差异：候选测试缓存新增了合成浴深，基线没有，来源 hash 正确反映差异。保存失败比较，给旧源码独立缓存复制同字节输入后重跑，344 项全部一致。没有把校验值归一化掉。
5. 第一轮 wheel 使用旧 setuptools `build/lib`，带入已迁出的候选。先前只检查应有文件的安装验证不足；完整载荷检测拒绝了它。新增构建暂存区清理（只接受本仓库受管 `build/` 内目录，只触及生成 Python 载荷、拒绝与源目录重叠）和五个检查，包括对研究/历史/tests 路径的拒绝，并核 sdist→wheel。新环境验证无多余文件、无研究桥，再完成真实外部目录运行。早期 wheel 的运行不作为正式包隔离通过证据。

所有失败、proposed、completed 及未验收项保留身份。没有数值超时被改写为 PASS。上述工程故障不解释成原生产核心失败。

## 复跑命令和产物

从原提交可恢复冻结基线，勿覆盖现有 checkout：

```sh
git archive 25258950905f9d1aa84509c4c99ebad9ef33ba2b -o BASELINE.zip
python scripts/capture_test_collection.py compare BASELINE_COLLECTION.json CURRENT_COLLECTION.json --layout docs/research_engineering_layout.json
python scripts/compare_modularization_captures.py BASELINE.npz CURRENT.npz --reference-source BASELINE/src --candidate-source src
```

解压到独立目录后，Windows 按次串行运行下列命令；针对旧源码需把其 `src/compat` 放在本次进程的 PYTHONPATH 首位，使用原 bare bridges，不能加载当前别名。数值数据缓存也要一致。下面文件名是参数示例，实际本轮清单/hash 在机器报告内。

```powershell
python scripts/run_bounded_research_tests.py --module scripts.capture_modularization_state --source-root src --output STATE_FP64.npz
python scripts/run_bounded_research_tests.py --module scripts.capture_modularization_state --source-root src --output STATE_FP32.npz --dtype float32
python scripts/run_bounded_research_tests.py --module scripts.capture_driver_modularization --source-root src --output DRIVER.npz --wind-jit
python scripts/run_bounded_research_tests.py --module scripts.capture_test_collection capture --output COLLECTION.json
python scripts/run_bounded_research_tests.py tests/experiments tests/evaluation tests/baselines tests/infrastructure tests/fd/test_production_ownership.py
python scripts/run_bounded_research_tests.py --module ocean_solver mms
```

独立 venv 只安装正式 wheel，不装 research；从仓库外目录运行：

```sh
python REPO/scripts/run_bounded_research_tests.py --script REPO/scripts/validate_installed_modularization.py --source-root REPO/src
python -m ocean_solver experiment run REPO/experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml --root REPO --outputs outputs --evaluate
python -m ocean_solver runs list --outputs outputs
```

MOM6 按 [完整说明](baselines_zh.md) 对 prepare 返回的新目录依次 run/convert/evaluate，不重复覆盖已有原生运行。README 中安装、帮助、validate、expand、sweep、runs、MMS、历史校验均实际执行；全球 expand 与三点 sweep 只展开，未积分。原始日志、NPZ、运行目录和缓存留在忽略的本地工作目录，机器报告保存路径与字节 hash；不提交 ETOPO/WOA、上游源码或大产物。新 clone 可用上述独立 Git 基线与合成输入重建回归，原历史成绩则按证据索引检出其来源。

## 未运行与剩余限制

完整 2101 项 suite 由远端 CI 执行，本地没有把它作为单个超 180 s 的任务启动。CI 状态以当前 PR checks 为准；分组通过不冒充全套通过。

本轮没有重建 MOM6、运行全球30天/驻波动力积分、全球365天、百年积分或 GPU。固定 fetch/build 链及错误处理已接线，从零构建尚未独立验收；原生全球输入接入不认证完整风、垂向采样和物理等价。不存在隐含重网格，也没有等误差成本曲线或加速结论。

真实全球数据在配置中是版本/路径引用，未提供确定旧数据字节时 expected SHA 留空并标未认证；运行时冻结实际所读文件，不能据此把当前数据认定为历史原件。没有猜测算法或推荐参数替用户作研究决策。

旧 campaign shell 保留 21 份，因为它们仍有历史复現/动态环境用途；其中旧 launch 路径在当前树未必存在，不宣传可直接运行。来源提交、引用与待判断理由列在 preservation。历史复现检出原 commit；本轮正式入口已有新的文档化链，薄兼容门面不保存第二份算法。
