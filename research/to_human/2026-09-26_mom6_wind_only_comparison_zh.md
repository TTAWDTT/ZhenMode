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

MOM6 30d 还在跑。跑完后才做第一张真正的同输入对照表。

## 还不能说的

现在没有热/盐通量，所以这不是完整气候态比较。不能说“打败工业级模式”。
下一步是先完成 30 天同风场对照，再加 matched bulk forcing 或 SST restoring。
