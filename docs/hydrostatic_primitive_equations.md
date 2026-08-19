# 静力原始方程组（Hydrostatic Primitive Equations）

## 0. 什么是"原始方程组"？

想象站在海边看海水。海水在动——有水平的流动（洋流），有上下翻涌，有温度变化，有盐度变化。**原始方程组**就是一组数学公式，用来描述"海水在每一点、每一时刻的速度、温度、盐度、压力是多少"。

它叫"原始"（primitive），是因为它几乎没有做简化——保留了最原始的物理过程。这跟那些做了大量简化的模型（比如只算两层水、或者只算温度不管流速的模型）形成对比。

海洋里一小块水会经历这些事：

1. **被水流推着走**（平流 / advection）——周围的水把它冲到别处
2. **被地球自转甩偏**（Coriolis 力 / 科氏力）——地球在转，所以水流不是走直线，会被偏转。北半球往右偏，南半球往左偏
3. **被压力差推着走**（压力梯度力）——水从压力高的地方往压力低的地方流
4. **被风吹**（风应力）——海面的风把表层水拖着走
5. **被海底摩擦拖住**（底摩擦）——靠近海底的水流变慢
6. **自己也在扩散**（湍流混合 / 扩散）——搅动让快的和慢的、热的和冷的混在一起
7. **被太阳晒热**（热通量）——海面吸收太阳辐射

原始方程组就是把这些过程**全部写成方程**。

---

## 1. "静力"是什么意思？

### 1.1 静力平衡—— 最关键的一个简化

真实情况下，海水在垂直方向也满足牛顿第二定律：垂直加速度 = 垂直方向的各种力。

但海洋有个特点：**垂直方向的运动远比水平方向慢**。海水主要在大范围水平流动，垂直方向几乎是在"静止"状态——重力把水往下压，压力从下往上托住它，两者刚好平衡。

我们称该平衡为**静力平衡**（hydrostatic balance）：

$$
\frac{\partial p}{\partial z} = -\rho\, g
$$

其中 $p$ 是压力，$\rho$ 是密度，$g$ 是重力加速度，$z$ 是垂直坐标（向上为正）。

**物理含义**：深度越深，压力越大，而且压力增加的速率正好等于单位体积水的重量 $\rho g$。

做了这个简化之后，**垂直方向的动量方程就可以不要了**——不需要算垂直加速度，直接用"压力 = 上方所有水的重量"来算压力。垂直速度 $w$ 也不再是独立演化的变量，而是用"流进来的水必须等于流出去的水"（质量守恒）反推出来。


### 1.2 其他核心假设

除了静力平衡，这套方程组还做了以下近似：

- **Boussinesq 近似**：密度变化很小，在惯性项（加速度、平流）中用常数参考密度 $\rho_0$ 代替真实密度 $\rho$；只在压力梯度力和浮力中保留密度变化
- **不可压缩**：海水密度不随压力变化，速度场无散度 $\nabla \cdot \mathbf{u} = 0$
- **f 平面 / β 平面**：Coriolis 参数 $f$ 在整个域内线性变化：$f = f_0 + \beta(y - y_0)$
- **传统近似**：只保留 Coriolis 力的局部垂直分量，忽略水平分量

### 1.3 在海洋模型谱系中的位置

这套方程在海洋模型谱系中的位置：

- **更简化的**：浅水方程（Shallow Water Equations）、准地转方程（Quasi-Geostrophic）——计算更快但丢失更多物理过程
- **本模型**：静力原始方程（Hydrostatic Primitive Equations）——MOM6、NEMO 等主流海洋大模式的核心选择，在精度和计算量之间取得平衡
- **更精确的**：非静力方程（Non-hydrostatic Equations）——保留完整垂直动量方程，适合小尺度高精度模拟（如 MITgcm 的非静力模式）

---

## 2. 完整方程组

### 2.1 水平动量方程（u、v）

东西方向流速 $u$ 和南北方向流速 $v$ 的演化方程：

$$
\boxed{
\frac{\partial u}{\partial t}
= -\nabla_h \cdot (u\,\mathbf{u}_h)
+ f\,v
- \frac{1}{\rho_0}\frac{\partial p}{\partial x}
+ \nu_h \nabla_h^2 u
+ \nu_v \frac{\partial^2 u}{\partial z^2}
+ \frac{\tau_x}{\rho_0 \,\Delta z_{\text{top}}}\,\delta_{\text{top}}
- r_{\text{bot}}\, u_{\text{bot}}
}
$$

$$
\boxed{
\frac{\partial v}{\partial t}
= -\nabla_h \cdot (v\,\mathbf{u}_h)
- f\,u
- \frac{1}{\rho_0}\frac{\partial p}{\partial y}
+ \nu_h \nabla_h^2 v
+ \nu_v \frac{\partial^2 v}{\partial z^2}
+ \frac{\tau_y}{\rho_0 \,\Delta z_{\text{top}}}\,\delta_{\text{top}}
- r_{\text{bot}}\, v_{\text{bot}}
}
$$

其中 $\mathbf{u}_h = (u, v)$ 是水平速度，$\nabla_h = (\partial/\partial x,\;\partial/\partial y)$ 是水平梯度算子。

**逐项物理含义：**

- **平流** $-\nabla_h \cdot (u\,\mathbf{u}_h)$ — 水流自身的运动把动量搬运到别处
- **Coriolis** $+f\,v$（u 方程）/ $-f\,u$（v 方程） — 地球自转使运动偏转
- **压力梯度** $-\frac{1}{\rho_0}\frac{\partial p}{\partial x}$ — 压力差驱动水流
- **水平扩散** $\nu_h \nabla_h^2 u$ — 湍流水平混合，平滑速度场
- **垂直扩散** $\nu_v \frac{\partial^2 u}{\partial z^2}$ — 湍流垂直混合
- **风应力** $\frac{\tau_x}{\rho_0 \Delta z_{\text{top}}}\delta_{\text{top}}$ — 海面风把动量传给表层水（仅最顶层）
- **底摩擦** $-r_{\text{bot}}\, u_{\text{bot}}$ — 海底摩擦使底层水流减速（仅最底层）

其中 $\delta_{\text{top}}$ 表示该项只在表层生效，$u_{\text{bot}}$ 是最深层的流速。

**Coriolis 参数：**

$$
f = f_0 + \beta\,(y - y_0) = 2\Omega\sin\varphi_0 + \frac{2\Omega\cos\varphi_0}{R_E}(y - y_0)
$$

其中 $\Omega = 7.2921 \times 10^{-5}\;\text{rad/s}$ 是地球自转角速度，$R_E = 6371\;\text{km}$ 是地球半径，$\varphi_0$ 是参考纬度。

---

### 2.2 连续性方程（诊断垂直速度 $w$）

由不可压缩假设 $\nabla \cdot \mathbf{u} = 0$：

$$
\boxed{
\frac{\partial w}{\partial z} = -\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right)
}
$$

**物理含义**：水不可压缩。如果某个地方水在水平方向散开了（流出去的多），那垂直方向一定有水补充进来；反之亦然。

**求解方式**：从海底向上积分，底部边界条件为刚性盖 $w = 0$：

$$
w(z_k) = w(z_{k+1}) - \int_{z_k}^{z_{k+1}}\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right)dz'
$$

离散化（梯形法则，逐层向上）：

$$
w_k = w_{k+1} - \frac{1}{2}\left[(\nabla_h \cdot \mathbf{u}_h)_k + (\nabla_h \cdot \mathbf{u}_h)_{k+1}\right] \Delta z_k
$$

> **注意**：$w$ 是**诊断量**，不是预报量。它不参与时间积分，而是在每步结束后根据当前 $u, v$ 重算。

---

### 2.3 静水压力方程

由静力平衡 $\partial p/\partial z = -\rho g$ 从自由表面 $z = \eta$ 向下积分：

$$
\boxed{
p(z) = \underbrace{\rho_0\, g\, \eta}_{\text{正压项（海面高度）}} + \underbrace{g\int_{z}^{0} \rho'(z')\, dz'}_{\text{斜压项（密度积分）}}
}
$$

其中 $\rho' = \rho - \rho_0$ 是**密度异常**（Boussinesq 近似下只保留密度偏差部分）。

**物理含义**：
- **正压项** $\rho_0 g \eta$：海面高度 $\eta$ 造成的水柱重量，所有深度相同。
- **斜压项** $g\int_z^0 \rho' dz'$：从深度 $z$ 到海面之间，密度异常（冷水更重、热水更轻）造成的额外压力。

**水平压力梯度力**（驱动动量方程）：

$$
-\frac{1}{\rho_0}\frac{\partial p}{\partial x} = -g\frac{\partial \eta}{\partial x} - \frac{g}{\rho_0}\frac{\partial}{\partial x}\int_z^0 \rho'\, dz'
$$

$$
-\frac{1}{\rho_0}\frac{\partial p}{\partial y} = -g\frac{\partial \eta}{\partial y} - \frac{g}{\rho_0}\frac{\partial}{\partial y}\int_z^0 \rho'\, dz'
$$

---

### 2.4 状态方程（EOS）

线性化 Boussinesq 状态方程：

$$
\boxed{
\rho = \rho_0\left[1 - \alpha_T(T - T_{\text{ref}}) + \beta_S(S - S_{\text{ref}})\right]
}
$$

**物理含义**：
- $\alpha_T \approx 2.0 \times 10^{-4}\;\text{K}^{-1}$：热膨胀系数——**越热越轻**
- $\beta_S \approx 7.6 \times 10^{-4}\;\text{psu}^{-1}$：盐收缩系数——**越咸越重**
- $T_{\text{ref}}$、$S_{\text{ref}}$：参考温盐（通常取 $15°\text{C}$、$35\;\text{psu}$）
- $\rho_0 \approx 1025\;\text{kg/m}^3$：参考密度

密度异常（用于压力计算）：

$$
\rho' = \rho - \rho_0 = \rho_0\left[-\alpha_T(T - T_{\text{ref}}) + \beta_S(S - S_{\text{ref}})\right]
$$

浮力：

$$
b = -g\frac{\rho'}{\rho_0} = g\left[\alpha_T(T - T_{\text{ref}}) - \beta_S(S - S_{\text{ref}})\right]
$$

---

### 2.5 示踪物方程（温度 T、盐度 S）

温度和盐度是被动示踪物，跟随水流搬运，同时自身扩散：

$$
\boxed{
\frac{\partial T}{\partial t}
= -\nabla_h \cdot (T\,\mathbf{u}_h)
+ \kappa_h \nabla_h^2 T
+ \kappa_v \frac{\partial^2 T}{\partial z^2}
+ \frac{Q_{\text{heat}}}{\rho_0\, C_p\, \Delta z_{\text{top}}}\,\delta_{\text{top}}
}
$$

$$
\boxed{
\frac{\partial S}{\partial t}
= -\nabla_h \cdot (S\,\mathbf{u}_h)
+ \kappa_h \nabla_h^2 S
+ \kappa_v \frac{\partial^2 S}{\partial z^2}
}
$$

**逐项物理含义：**

- **平流** $-\nabla_h \cdot (T\,\mathbf{u}_h)$ — 温度/盐度被水流搬运
- **水平扩散** $\kappa_h \nabla_h^2 T$ — 湍流水平混合
- **垂直扩散** $\kappa_v \frac{\partial^2 T}{\partial z^2}$ — 湍流垂直混合
- **表面热通量** $\frac{Q_{\text{heat}}}{\rho_0 C_p \Delta z_{\text{top}}}\delta_{\text{top}}$ — 海面太阳辐射加热（仅表层）

其中 $C_p \approx 3992\;\text{J/(kg·°C)}$ 是海水比热容。盐度方程没有表面源项（不考虑蒸发/降水时）。

---

## 3. 时间积分方法

### 3.1 IMEX Strang 分裂法

方程组可以写成算子分裂形式：

$$
\frac{d\mathbf{U}}{dt} = \underbrace{\mathcal{L}(\mathbf{U})}_{\text{线性（精确求解）}} + \underbrace{\mathcal{N}(\mathbf{U})}_{\text{非线性（显式求解）}}
$$

**线性部分 $\mathcal{L}$**（有精确解析解，无条件稳定）：
- 水平扩散：$\nu_h \nabla_h^2 u$、$\kappa_h \nabla_h^2 T$（谱矩阵指数）
- f 平面 Coriolis 旋转：$f_0$ 部分（精确旋转矩阵）

**非线性部分 $\mathcal{N}$**（显式 Euler）：
- 平流（通量形式，去混淆）
- 压力梯度力
- 垂直扩散（非均匀 z 网格有限差分）
- β 平面 Coriolis 修正：$(f - f_0)$ 部分
- 表面强迫（风应力、热通量）
- 线性底摩擦

**Strang 分裂**（时间对称，二阶精度）：

$$
\boxed{
\mathbf{U}(t + \Delta t) = e^{\mathcal{L}\,\Delta t/2}\;\left[\mathbf{U}(t) + \mathcal{N}(\mathbf{U}^*)\,\Delta t\right]\;e^{\mathcal{L}\,\Delta t/2}
}
$$

分三步执行：

- **1. 线性半步** $e^{\mathcal{L}\,\Delta t/2}$ — 水平扩散（谱衰减）+ f₀ Coriolis 旋转
- **2. 非线性整步** $\mathbf{U}^* + \mathcal{N}(\mathbf{U}^*)\,\Delta t$ — 所有非线性项显式 Euler
- **3. 线性半步** $e^{\mathcal{L}\,\Delta t/2}$ — 同步骤 1

### 3.2 避免重复计算

非线性整步中先计算**全部**倾向（包含线性部分），然后**减去**已由线性半步精确处理的项：

$$
\left.\frac{\partial u}{\partial t}\right|_{\text{显式}} = \left.\frac{\partial u}{\partial t}\right|_{\text{全部}} - \nu_h \nabla_h^2 u - f_0\,v
$$

这样显式部分只保留真正的非线性项和 β 平面修正 $(f - f_0)$。

### 3.3 f 平面 Coriolis 精确旋转

线性半步中的 Coriolis 旋转有精确解：

$$
\begin{cases}
\displaystyle\frac{du}{dt} = +f_0\, v \\
\displaystyle\frac{dv}{dt} = -f_0\, u
\end{cases}
\quad\Longrightarrow\quad
\begin{pmatrix} u \\ v \end{pmatrix}_{t+\Delta t}
=
\begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} u \\ v \end{pmatrix}_t
$$

其中 $\theta = f_0 \,\Delta t$。北半球（$f_0 > 0$）中，向北的速度（$v > 0$）旋转为向东（$u > 0$），物理方向正确。

### 3.4 诊断量更新

每步结束后重算诊断量：

$$
\rho \leftarrow \text{EOS}(T, S), \quad
p \leftarrow \text{hydrostatic}(\rho', \eta), \quad
w \leftarrow \text{continuity}(u, v)
$$

---

## 4. 方程组总结

- **水平流速 $u, v$**（动量方程/牛二，**预报**）：$\partial u/\partial t$ = 平流 + Coriolis + 压力梯度 + 扩散 + 风 − 摩擦
- **垂直流速 $w$**（连续性方程，**诊断**）：$\partial w/\partial z = -(\partial u/\partial x + \partial v/\partial y)$
- **压力 $p$**（静水压力，**诊断**）：$p = \rho_0 g \eta + g\int_z^0 \rho' dz'$
- **密度 $\rho$**（状态方程，**诊断**）：$\rho = \rho_0[1 - \alpha_T \Delta T + \beta_S \Delta S]$
- **温度 $T$**（示踪物方程，**预报**）：$\partial T/\partial t$ = 平流 + 扩散 + 表面加热
- **盐度 $S$**（示踪物方程，**预报**）：$\partial S/\partial t$ = 平流 + 扩散

**静力原始方程 = 水平动量方程（牛二）+ 连续性方程（质量守恒）+ 静水压力（垂直平衡）+ 状态方程（密度）+ 示踪物方程（温盐守恒）**

核心简化就一个字：**"静"**——垂直方向不加速，压力等于上方水重，垂直速度靠质量守恒反推。这样省掉了最贵的垂直动量方程，计算量大减，精度对大尺度海洋环流来说足够了。