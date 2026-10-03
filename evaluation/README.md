# 评价协议与报告

评分实现在可安装包 `ocean_solver.validation.benchmarks`；本目录仅存协议。`ocean_solver.evaluation` 负责校验、接线、来源和报告，不复制 RMSE、平滑、漂移或门槛算法。

```sh
zhenmode evaluate score \
  --input outputs/RUN/model/global_run.npz \
  --protocol evaluation/protocols/production-smoke-v1.json \
  --run-manifest outputs/RUN/manifest.json \
  --out-dir outputs/RUN/evaluation
zhenmode evaluate compare outputs/RUN_A/evaluation/report.json outputs/RUN_B/evaluation/report.json --out outputs/comparison.json
```

`score` 只读模式结果，写独立的 `metrics.json`、`report.json`、`report.md`，并更新运行 manifest 的 `acceptance`、`comparability` 和报告路径/hash。已有报告拒绝覆盖；复评分使用新目录。

协议分别绑定 case、参考身份、掩膜、面积权重、保存记录的算术时间平均和评分窗口。不执行隐含重网格。结果和实际 geometry/reference 必须来自已完成运行的 hash 绑定产物；生产来源必须覆盖完整注册包及实际核心执行文件，缺项保留 limited 身份。

比较拒绝不同 case、实际物理网格、有效物理、数据、协议、窗口、参考场和覆盖。数值时间步/精度变化可以比较误差，成本比较另须相同计时范围和明确硬件/精度；没有误差—成本曲线时不生成正式加速结论，也不自动排名。

旧证据可以原样导入：

```sh
zhenmode evaluate import-historical --input research/experiments/ice_proxy_045/benchmark_365d_repeat.json --out outputs/historical-evidence.json
```

缺少 `metric_definition` 的历史报告保持 `legacy_equal_cell_index_box_v1`，不自动成为 v2。详细口径与边界见 [统一评价说明](../docs/evaluation_zh.md)。
