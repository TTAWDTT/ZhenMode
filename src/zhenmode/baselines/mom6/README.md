# MOM6 接入

本目录集中保存 `adapter.py`（获取、构建、准备、运行、转换）、`forcing.py`（共同强迫导出）、`pins.json`（固定来源）和 `cases/tc1.json`（内置算例）。MOM6/FMS 源码、编译产物及大输出放仓库外 cache。先执行 `zhenmode baseline mom6 doctor --cache CACHE`。完整说明见 [MOM6 接入](../../../../docs/baselines_zh.md)。

版本和算例 JSON 随 wheel 安装，接入程序直接读取它们，不需要仓库顶层的另一份副本。Linux/WSL、已安装本项目的 Python 环境：

```sh
zhenmode baseline mom6 fetch --cache "$HOME/.cache/zhenmode/mom6/f49a000"
zhenmode baseline mom6 build --cache "$HOME/.cache/zhenmode/mom6/f49a000" --wall-seconds 1800
zhenmode baseline mom6 prepare --cache "$HOME/.cache/zhenmode/mom6/f49a000"
# prepare 返回唯一 RUN_DIR；每次重复重新 prepare。
zhenmode baseline mom6 run --run-dir RUN_DIR --wall-seconds 180
zhenmode baseline mom6 convert --run-dir RUN_DIR
zhenmode baseline mom6 evaluate --run-dir RUN_DIR
```

`fetch/build` 不默认执行；构建前先报告工具链和资源。已有 cache 只检查，不自动覆盖。无可核查 source→binary 构建收据时标 unverified，即使当前源码干净且 executable hash已记录。

tc1 是10×8×8、24步、0.25日的原生执行 smoke，不与全球 case 比气候指标或速度。两个共同全球 case 的 MOM6 原生接入预设在 `configs/mom6/presets/`；先备齐实际原生输入、共同初态/海深 reference，再用 `prepare-native`。缺少历史他机输入不虚构恢复，完整物理核验未完成时保持 limited。

`omip2.py` 与 `omip2-pins.json` 对应独立的 MOM6+SIS2 候选，保留旧solo接入身份。计划不启动构建。准备命令在Linux/WSL核验全部组件、配置字节，展开日历/窗口/恢复改动，并在新目录生成LY2009/Gill无冰交换适配；上游cache不改写。适配覆盖系数、饱和湿度、空气性质、10m位温与出射长波，不代表SIS2潜热、冰、EOS、映射或预算已经符合，见[机制符合表](../../../../docs/benchmark_compliance_zh.md)。

```sh
zhenmode baseline mom6 omip2-plan --profile integration-6h
zhenmode baseline mom6 omip2-prepare --examples-dir PINNED_CHECKOUT --output NEW_PREPARATION --profile integration-6h
# 先报告构建资源；每次单CPU、180秒、4GiB主机RSS监督。相同目录可分段续构建。
zhenmode baseline mom6 omip2-build --preparation NEW_PREPARATION --output NEW_BUILD --wall-seconds 180
```

`coupled_sources.py`只将固定提交的构建控制、组件及嵌套依赖从Git archive暂存到独立目录，排除cache中的未跟踪配置；派生补丁逐字节核验。SIS2无冰潜热按当前SST形成实际能量，经独立字段传入MOM，不再由水质量乘固定潜热重建；冰—海底部热记录同步使用实际能量。顶部库存、海冰机制及均通量时间读取仍需核验，这些适配是独立候选，不修改旧solo接入或ZhenMode默认公式。

构建返回completed、failed或incomplete；超时退出码为2，保留中间产物。每段核验实际上游/补丁源码、执行权限、链接、工具、编译参数和上段中间文件快照，拒绝未知可编译文件、被改写的缓存及不同来源复用目录。实际源码预期从已核原始pin派生，不能通过同时改写暂存源码和清单绕过校验。完成后再次核源码并记录二进制及动态库hash。flags通过环境传递，保留autoconf探测的MPI、Cray pointer等必需参数。主机RSS每0.2秒监督，不是硬cgroup上限；175秒发送终止信号，180秒结束宽限清理。构建成功仍保持execution_ready=false；不启动模型，也不授予气候或评分资格。区间均通量读取、原生机制配置、输入和诊断接线通过后再准备运行目录。
