# 历史生产证据索引

`index.json` 指向原始 Git 对象并保存原字节 SHA-256。这里不复制旧实验数据、不改旧结论、不把报告提交冒充完整运行源码身份。

- `ad45c2b` 的提交说明记载约 194.7 模式年，`_spinC_from30_ana.json` 保存 PASS 和 no-blowup；OHC/AMOC 警告仍需阅读。
- `4dbd14f` 的 `_ana_tmp/spinC_slope005x_analysis.json` 保存约 192–274 年阶段的 PASS，仍有 AMOC 警告。
- `docs/deep-heat-poisoning-root-cause.md` 记载从约 273 年 checkpoint 继续 52 年，到累计约 325 年。不是同一源码从零连续 325 年。
- `bb7ba23` 的 365 天重复 manifest 保存旧评分窗口 280–365 天、raw 0.913642930844384°C、A2 1.112570737767704°C。参照是保存的 WOA 初始表层场，raw 为湿格等权。禁止与 `area_weighted_angular_box_v2` 混排。
- `2375166` 是 `9944bbd` 的祖先，原生产入口长期沿 `run_long_integration_global → make_solver_global → _step_impl` 演进。`9944bbd` 是后出的 manifest 工具提交，不能据此断言它就是完整执行源码。

上述证据支持既有生产谱系的历史长期数值稳定和特定协议下低 RMSE；当前版本的完整历史复现、全部气候指标、工业级资格和速度优势须另行验证。候选 `h_top=2.5+η` 的 353→354 步失败不否定这些生产历史；参见原 `docs/legacy_core_repair_status_zh.md` §23。

核验与导出原始材料：

```bash
python scripts/verify_historical_evidence.py
python scripts/verify_historical_evidence.py --export outputs/history-replay
```

导出目录必须不存在。脚本只导出证据字节和索引，不声称已完成数值复跑。数值历史复现应使用原提交及相应外部数据、checkpoint、环境；缺项列为缺失，不用当前实现补出旧成绩。
