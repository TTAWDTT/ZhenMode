# 旋转调整与分层热成风比较

2026-10-09：新增两类有解析参照的机制实验，沿用已固定的 ZhenMode、MOM6、
Oceananigans 三个原生模式和统一原生输出格式。它们是明确记录的理想化变体，
不宣称逐项复现上游完整算例，也不授予全球气候或工业运行资格。

## 旋转地转调整

采用经典旋转浅水方程的闭通道余弦初态。MOM6 的
[adjustment 初始化模块](https://github.com/NOAA-GFDL/MOM6/blob/f49a00096df607b48354603e2398e14e189fd62e/src/user/adjustment_initialization.F90)
提供该测试家族的上游来源；本变体采用均匀水柱海面扰动，不复制其分层盐度锋面。

H=100 m，A=0.01 m，f=π/32000 s⁻¹，Ly=32000√981/√3 m，Lx=100 km。
x 周期、y 封闭；初始速度为零，η=A cos(πy/Ly)。温盐保持 15°C/35 psu，
dt=50 s，积分 32000 s、每 1000 s 输出。nx=8，ny=64/128/256，nz=4。

令 k=π/Ly、ω²=f²+gHk²、r=f²/ω²=1/4，线性参考为：

- η=A cos(ky)[r+(1−r)cos(ωt)]。
- v=(Agk/ω)sin(ky)sin(ωt)。
- u=(fAgk/ω²)sin(ky)[1−cos(ωt)]。

参考同时包含平衡分量与惯性重力波；模式仍推进自身完整非线性方程。
FD 的边缘中心行法向速度被既有壁面算子清零，C 网格的壁面速度位于边界面；
原生位置与该离散差异保留，不能通过移动坐标或删去误差行获得较好分数。

## 分层热成风与温盐输运

采用固定 MOM6
[baroclinic-zone 初态](https://github.com/NOAA-GFDL/MOM6/blob/f49a00096df607b48354603e2398e14e189fd62e/src/user/baroclinic_zone_initialization.F90)
中的线性垂向分层和截断正弦锋面，旋转到封闭的 y 方向。
增加保持底部压力恒定的海面／速度，以及密度中性的温盐波。
该实验检验斜压压力平衡、垂向剪切与非均匀三维标量输运，不检验不稳定急流的湍流统计。

域为 100×100 km，H=100 m，f=1e−4 s⁻¹，32×32×16；垂向参考容量采用
等间距 FD 节点的双单元。dt=50 s，积分 86400 s、每 2700 s 输出。
Γ=0.02 °C/m，Δ(y)=0.5 sin[(π/2)clip((y−50 km)/25 km,−1,1)] °C。
背景 T=15+Γz+Δ(y)，S=35；沿 x 叠加 0.1°C 的温盐波，盐度扰动为
(α/β)倍温度扰动，α=2e−4 °C⁻¹、β=7.6e−4 psu⁻¹，因而线性密度扰动抵消。

海面取小根，满足 η−αΓη²/2−αΔ(η+H)=0；
u=−gαΔ′(y)(z+H)/f，v=0。平衡背景不随时间变化；中性波沿该剪切流平移。
输入生成采用稳定二次根和 12 点积分；评价另用 Newton 解柱压力方程及
32 点横向积分、解析垂向 sinc 积分，不读取被测温盐来生成答案。
参考的 16→32 点积分差异在预置控制中要求 ≤5e−13。

FD 使用固定参考节点；MOM6 与 Oceananigans 使用各自真实层厚推导的层中心和层平均。
原生离散单元内采用平坦层面，标量在 x/y/层体积上取均值，速度保留原生水平位置。
这套表示差异是实验声明的一部分。四个过程都关闭：外部强迫、显式混合、恢复和冰；
动量、连续性、科氏力、密度压力和温盐输运保留。

## 运行和评分

```sh
zhenmode benchmark channel --case geostrophic-adjustment --freeze NEW_CONTRACT.json
zhenmode benchmark channel --case thermal-wind --output NEW_RUN \
  --mom-executable PINNED_MOM6 --mom-source PINNED_SOURCE \
  --oceananigans-cache VERIFIED_CACHE --julia ISOLATED_JULIA \
  --source-revision ZHENMODE_COMMIT
zhenmode evaluate channel --contract NEW_RUN/contract.json \
  --output NEW_RUN/zhenmode/output.npz --report NEW_SCORE.json
```

共同验证／序列化／资源监督只保留一份，分别在 `evaluation/native_channel.py`
与 `execution/native_channel.py`。案例物理和输入在 `benchmarks/channel_dynamics.py`；
独立参考与评分在 `evaluation/channel_dynamics.py`；模式适配器不调用评价参考。

记录海面与三维 u/v 的时空 RMS、温盐绝对 RMS（°C/psu），以及体积和温盐库存
相对初值的最大变化。两类实验的海面 RMS 都除以预先固定的 0.01 m，
热成风的该尺度不按实际海面幅度重新调整。
速度按预先定义的物理尺度归一化；旋转实验按 A√(g/H)，热成风按最大理论表层流速。
三维误差使用各时刻原生体积权重与梯形时间权重，海面使用面积权重。
粗／中／细旋转的海面、u、v 相对 RMS 筛选为 10%/5%/2%；
热成风三项相对 RMS 均为 10%。体积为 1e−10、温盐库存为 1e−8。
旋转的常量温盐 RMS 为 1e−10；热成风温盐 RMS 上限为各中性波幅的 10%。
工程筛选与实际执行完成分别记录，失败指标保留。

真实运行前控制包括三种原生布局上的独立旋转闭式波、NPZ 实际读写、速度反号、
错误时钟／f、锋面固定点、密度中性抵消、底部柱压力方程和积分加密。
这些机制测试结果用于方法选择，不可再作为该选择的独立认证数据。
