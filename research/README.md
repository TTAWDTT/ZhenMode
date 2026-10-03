# 研究区

材料库存、FV/C-grid 和 r-star 是未采用的研究路线。正式包不导入本目录，不通过它们替代既有生产主线。

需要研究测试时单独安装：

```bash
python -m pip install -e ./research
```

研究代码使用 `zhenmode_research.candidates` 下的实际模块，不再安装历史 bare-module 别名。

| 区域 | 内容 |
| --- | --- |
| `src/zhenmode_research` | 独立安装的候选实现及其源码身份 |
| `experiments` | 历史实验、协议、独立参考计算与失败记录 |
| `tools/diagnostics` | 研究用瞬时压力/势能合同 |
| `tools/quality_speed` | 研究用注册验收、产物校验和生命周期计时工具 |
| `reviews` | 原独立审阅记录 |

从 [r-star 实验索引](experiments/material_rstar_coordinates/README.md)、[重启回放](experiments/material_restart_replay/README.md) 查具体研究。
原 controlled-window 工具移到 [历史工具区](../archive/tools/README.md)，它们校验并执行指定历史源码包。

生产运行使用 `zhenmode model`；生产实验与研究原型的执行状态、协议及结论各自记录。
候选 353→354 步失败不否定历史百年稳定成果；PR29 不在本次交付中。
