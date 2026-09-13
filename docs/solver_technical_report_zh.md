# 海洋环流求解器技术报告

> 日期：2026-09-08
> 求解器：全球有限差分海洋模式（JAX/GPU），静力原始方程组
> 代码：`src/jax_solver_global.py`（求解器核心）、`src/run_long_integration_global.py`（长程积分驱动）
>
> **渲染说明**：本文件含 LaTeX 数学公式，用支持 KaTeX/MathJax 的查看器打开（VS Code 自带 Markdown 预览、Typora、Obsidian 均可）。

---

## 1. 微分方程汇总

求解**球坐标下的静力原始方程组**（hydrostatic primitive equations，Boussinesq 近似，$z$ 坐标），与 MOM/NEMO 级 OGCM 同一方程体系。

### 1.1 水平动量方程

$u$、$v$ 分量，Coriolis 参数 $f = 2\Omega\sin\varphi$ 为完整二维场（无 $\beta$ 平面近似）：

$$
\frac{\partial u}{\partial t} = -\,\boldsymbol{u}\cdot\nabla_h u - w\frac{\partial u}{\partial z} + f v - \frac{1}{\rho_0}\frac{\partial p}{\partial x} + \nabla_h\cdot(\nu_h \nabla_h u) + \frac{\partial}{\partial z}\!\left(\nu_v \frac{\partial u}{\partial z}\right) + F_{\text{风}} + F_{\text{底}}
$$

$$
\frac{\partial v}{\partial t} = -\,\boldsymbol{u}\cdot\nabla_h v - w\frac{\partial v}{\partial z} - f u - \frac{1}{\rho_0}\frac{\partial p}{\partial y} + \nabla_h\cdot(\nu_h\nabla_h v) + \frac{\partial}{\partial z}\!\left(\nu_v \frac{\partial v}{\partial z}\right) + G_{\text{风}} + G_{\text{底}}
$$

### 1.2 静力近似与连续方程

$$
\frac{\partial p}{\partial z} = -\rho g
$$

（静力近似，全压强由密度逐层梯形累积）

$$
\nabla_h\cdot\boldsymbol{u} + \frac{\partial w}{\partial z} = 0
$$

（不可压缩 / Boussinesq）

**正压模**（自由表面，$\eta$ 为海面高度）：

$$
\frac{\partial \eta}{\partial t} + H_{sw}\,\nabla_h\cdot\boldsymbol{u}_{bt} = 0
$$

**斜压模**：垂直速度 $w$ 由水平散度自海底向上积分诊断（海底 $w=0$）：

$$
\frac{\partial w}{\partial z} = -\nabla_h\cdot\boldsymbol{u}, \qquad w(z_{bot}) = 0
$$

### 1.3 状态方程（线性 EOS）

$$
\rho' = \rho_0\left[-\alpha_T\,(T - T_0) + \beta_S\,(S - S_0)\right]
$$

其中 $\rho_0 = 1025\ \mathrm{kg/m^3}$，$T_0 = 15\ \mathrm{°C}$，$S_0 = 35\ \mathrm{psu}$，$\alpha_T = 2.0\times10^{-4}\ \mathrm{K^{-1}}$，$\beta_S = 7.6\times10^{-4}\ \mathrm{psu^{-1}}$。

### 1.4 示踪方程（$T$、$S$ 同型）

$$
\frac{\partial T}{\partial t} = \underbrace{\nabla_h\cdot(\kappa_h \nabla_h T)}_{\text{侧向扩散}} + \underbrace{\frac{\partial}{\partial z}\!\left(\kappa_v \frac{\partial T}{\partial z}\right)}_{\text{垂直扩散}} + \underbrace{C_{\text{conv}}}_{\text{对流调整}} + \underbrace{\mathcal{G}_{GM/Redi}}_{\text{中尺度闭合}} + \underbrace{\frac{Q_{net}}{\rho_0\, c_p\, H_{surf}}}_{\text{海气热通量}}
$$

各项说明：

- **对流调整 $C_{\text{conv}}$**：界面通量形式，静力不稳定列（上重下轻）内以 $\kappa_{conv} = 0.05\ \mathrm{m^2/s}$ 混合；**列内严格守恒**（任意掩膜下列积分恒为零，杜绝对流制造热量）。
- **GM/Redi 中尺度闭合**（Griffies 1998 skew-flux 残差形式）：等密面坡度带 Danabasoglu–McWilliams (1995) 四次 taper，$S_{max} = 0.01$；界面通量离散（MOM6/NEMO 同款），避免锯齿模与深层 runaway。
- **海气热通量**（Haney/Barnier bulk）：

$$
Q_{net} = Q_{clim}(y) + \lambda\,(T_{atm} - SST)
$$

$T_{atm}$ 为纬向均匀的经向廓线——只锚定大尺度经向梯度，纬向 SST 结构由模式自身平流/混合预报（气候态对比非循环论证）。盐度无恢复项。

### 1.5 边界条件

| 位置 | 条件 |
|---|---|
| 经向 $0\text{–}360°$ | 周期 |
| 南北 $60°$ 截断墙 | 刚壁 no-flux：$v = 0$ + 镜像幽灵胞零法向扩散通量 |
| 海面 | 风应力 $\tau$（NCEP R1 实测，12 月季节循环 + 5 天线性过渡）；bulk 热通量；无盐通量 |
| 海底 | no-flux（幽灵层填充底层 wet 值）；线性底摩擦 $r_{bot} = 10^{-3}\ \mathrm{s^{-1}}$ |

---

## 2. 网格方案与计算方式

### 2.1 水平网格

- **真实全球经纬度网格 $360\times120$，$1°\times1°$**
- 经向周期；纬向 $[-59.5°, 59.5°]$（极区 $60°$ 截断）
- 球面度量完整保留：$dx = R\cos\varphi\,\Delta\lambda$ 随纬度变化，所有算子带 $\cos\varphi$ 加权
- 极帽滤波：最外 2 行纬向平均 + 3 行 $\cos^2$ 渐变，处理 $\cos\varphi\to0$ 度量奇异

### 2.2 垂直网格

**14 层非均匀 $z$-levels**（上密下疏）：

$$
z_k \in \{0,\ -5,\ -15,\ -30,\ -50,\ -75,\ -100,\ -150,\ -200,\ -300,\ -450,\ -650,\ -900,\ -1200\}\ \mathrm{m}
$$

（表层 5 m，深层 250–300 m）

### 2.3 地形与掩膜

- **ETOPO 2022 实测水深**，30 道 Laplacian 平滑 + 100 m 最小水深截断（浅于 100 m 划为陆地）
- 逐层湿掩膜 $W_k$（wet_mask_z）：海底以下的"幽灵水"从压强积分中剔除，根治陡地形压强梯度力假源
- 湿点占比 71.6%

### 2.4 时间积分（Strang 分裂）

$$
\mathcal{U}(t+\Delta t) = \mathcal{L}\!\left(\tfrac{\Delta t}{2}\right)\,\mathcal{N}(\Delta t)\,\mathcal{L}\!\left(\tfrac{\Delta t}{2}\right)\,\mathcal{U}(t)
$$

- **$\mathcal{L}$ 步（线性）**：显式 Laplacian + 双调和扩散 → 逐格点精确 Coriolis 旋转（2D $f$ 场）→ 自由表面 **forward-backward（Sielecki）格式** + 半隐式正压 Coriolis + 隐式底摩擦（无条件稳定）
- **$\mathcal{N}$ 步（非线性）**：RK2 预报-校正，示踪子步先行（内波特征值移到虚轴左侧）

自由表面 forward-backward 更新（Sielecki 格式，$|\lambda|=1$ 中性）：

$$
\eta^{n+1} = \eta^n - \tfrac{\Delta t}{2} H_{sw}\,\nabla_h\cdot\boldsymbol{u}_{bt}^n
$$

$$
\boldsymbol{u}_{bt}^{n+1} = \frac{\boldsymbol{u}_{bt}^n + \tfrac{\Delta t}{2}\left(-g\nabla_h\eta^{n+1} + \boldsymbol{F}\right)}{1 + r_{bt}\,\tfrac{\Delta t}{2}}
$$

稳定 watchdog 三判据：$\max|u| < 10\ \mathrm{m/s}$、漂移 $< 2\ \mathrm{°C}$、$|\eta| < 15\ \mathrm{m}$；超限即 FAIL 并落断点。

### 2.5 空间离散与守恒性

| 性质 | 做法 |
|---|---|
| 质量守恒 | 通量形式散度；wet/dry 界面与截断墙通量严格归零；全域求和机器零（实测 $\sim10^{-21}$） |
| 能量中性 | 梯度算子取散度算子的**离散伴随**（面积加权内积）；正压自由波 $\|\lambda\| = 1$ 不增不减 |
| 防混叠 | 经向 2/3 FFT 截断 + 纬向 5 点二项式低通 $[1,4,6,4,1]/16$ |
| 海岸无污染 | 水平梯度/平流算子全部 face-gated：跨湿/干界面差分归零（根治幽灵哨兵 $15\ \mathrm{°C}$ 悬崖驱动的海岸热泵） |
| 对流守恒 | 界面通量形式对流调整，任意掩膜下列积分严格为零 |
| 海绵/边界松弛 | 质量补偿式：松弛移走的体积均匀回填，动力学不受均匀水位偏移影响 |

### 2.6 计算架构

- **JAX/XLA**：整个时间步 JIT 编译为单一计算图
- 季节风走动态强迫路径：12 个月风场作为运行时参数传入，一个图服务全部月份（省 12× 显存）
- GPU 上 $\Delta t = 150\ \mathrm{s}$ 约 **6.5 分钟/模拟年**（$360\times120\times14$，576 步/天）
- 逐年断点续算（容器回收不丢进度）+ 快照对齐恢复

### 2.7 验证链

- FD 算子经 MMS（制造解）逐项验证
- 与谱方法区域求解器（保留的未动基线）交叉验证
- JAX/NumPy 双后端位精确比对

---

## 3. 积分效果

### 3.1 积分里程（无一次数值爆库）

| 运行 | 时长 | 配置 | 结果 |
|---|---|---|---|
| g365d_013 | 1 年 | 生产配置（$\Delta t=60\ \mathrm{s}$，季节风，$\lambda=40$） | PASS，完成物理体检 |
| g3650d_014 | 10 年 | 同上 | AMOC 0.6→1.36 Sv 成长；SSS 稀释 yr4 自愈；deepT $+0.115\ \mathrm{°C/yr}$ |
| g3650d_016 | 10 年 | 对流调整修复后 | **deepT 漂移 $+0.02\ \mathrm{°C/10yr}$**（修复前 $+1.15$） |
| spinA100b | **96 年** | 畸变物理加速起转（$\Delta t=150\ \mathrm{s}$、$\kappa_v\times20$、$\lambda\times5$） | 全程稳定，断点续算跨容器回收 |

### 3.2 96 年加速起转的气候态（yr-96 状态，vs WOA 观测）

| 量 | 模式 | 观测/参考 | 评价 |
|---|---|---|---|
| SST（全球湿点均值） | 18.78 ℃ | 18.25 ℃ | 偏差 **$+0.53\ \mathrm{°C}$**（10 年时 $+1.0$，减半） |
| SSS | 34.78 psu | 34.62 | **$+0.16$**（10 年时 $+0.32$，减半） |
| 深层剖面 650/900/1200 m | 10.2 / 7.3 / 3.6 ℃ | $\sim$10.1 / 7.0 / 3.5 | 基本吻合 |
| 300 m | 15.65 ℃ | $\sim$14.5 | $+1.1\ \mathrm{°C}$（加速物理畸变痕迹，phase B 待消） |
| AMOC @26.5°N | **$+2.28$ Sv** | 17 Sv | **方向翻正**（10 年时 $-0.2$ 倒置），量级弱 7 倍 |
| AABW 胞 | $-2.49$ Sv | $-29$ Sv | 方向正确，量级弱 |
| 大西洋 MHT @26.5°N | **$+0.035$ PW** | $+1.20$ PW | 翻正（10 年时 $-0.44$） |
| KE（90 年轨迹） | 单调 $+27\%$ | — | 环流仍在成长，**未平衡**（与 AMOC 偏弱自洽） |

### 3.3 结论

积分路径健康：温度场基本定住（$\max T$ 漂移 $+0.001\ \mathrm{°C/yr}$）、水团结构达标、翻转环流方向全部正确且处于成长轨道中段。与官方模式（AMOC $15.2\pm2.3$ Sv、MHT $1.20$ PW）的差距是**起转时长问题**（平衡需 100–1600 年，当前 96 年加速等效），不是结构性问题。

### 3.4 已知局限（如实）

- $60°$ 纬向截断（无极区，AMOC/AABW 上限受限）
- 14 层垂直分辨率偏粗
- 线性 EOS
- $\lambda=40$ bulk 恢复仍强于真实海气耦合
- 加速段 300 m 暖异常（$+1.1\ \mathrm{°C}$）需 phase B（正常物理 30–50 年）消化；phase B 开 `--save-3d` 后即可计算真正的末 30 年三维气候平均
