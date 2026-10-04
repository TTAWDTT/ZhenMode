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
