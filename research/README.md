# 研究代码

研究包保存未采用的 FV/C-grid、材料库存和 r-star 原型。正式 `ocean_solver` 包不导入它。

```sh
python -m pip install -e ./research
```

| 路径 | 用途 |
| --- | --- |
| `src/zhenmode_research/` | 可安装的候选实现 |
| `experiments/` | 当前测试使用的研究组件、独立参考计算、方法协议和输入夹具 |
| `tools/diagnostics/` | 仍被研究测试使用的压力/势能诊断 |
| `literature/` | 方法对照笔记 |

相应测试在 `tests/candidates` 和 `tests/research`。保留的方法协议描述公式和适用条件；运行报告、日志和结果输出保存在本地运行目录。

生产实验使用顶层 `experiments/` 与 `zhenmode experiment`。研究候选不自动成为正式方法，也不凭候选失败否定生产谱系的历史成果。
