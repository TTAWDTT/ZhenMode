# 0.5° wind-only MOM6 对照进展

当前已把 MOM6 从“初始化失败”推进到“动力积分可用”。

## 已完成

- 修复 stack overflow
- 关闭不必要的初始场输出
- 改成共享 0.5° 全球网格：720x260，纬度 -65..65
- 改成 14 层 Z* ALE 垂直坐标
- 使用共享 WOA T/S 和 ETOPO 派生地形
- 使用 ocean_solver 同款 2023 月度 NCEP 风应力文件
- MOM6 1 天 smoke 稳定，约 5.5 分钟/天
- ocean_solver 同类 wind-only 30 天已跑完，3.4 分钟

## 目前能说的

ocean_solver 30d wind-only：
- global A2 RMSE：0.866 C
- global raw bias/RMSE：-0.058 / 0.566 C
- NA 40-60N RMSE：0.675 C
- near-wall bias：-0.144 C
- 30 天稳定 PASS

MOM6 30d 也跑完了，4 进程用时 49.6 分钟。

## 同输入 wind-only 30d 对比

| 模型 | days | global A2 | global raw bias/RMSE | NA 40-60N RMSE | near-wall bias/RMSE | wall time |
|---|---:|---:|---:|---:|---:|---:|
| ocean_solver | 30 | 0.8658 C | -0.0580 / 0.5660 C | 0.6751 C | -0.1442 / 0.2106 C | 3.4 min |
| MOM6 | 30 | 1.0961 C | -0.4070 / 1.1827 C | 0.6814 C | +0.0567 / 0.2434 C | 49.6 min |

这是第一张真正的同输入对照表。结论要克制：ocean_solver 在这个 30d wind-only
控制下 global A2 更好、NA RMSE 基本相当；但缺热/盐通量，所以还不是气候态比较，
也不能说“打败工业级模式”。

## 下一步

1. 用同协议做 matched bulk heat/salt forcing 或 SST restoring。
2. 再跑 30d 稳定性检查，然后到 365d。
3. 把 heat/salt drift、MLD、wall time 和物理设置一起写进 manifest。
