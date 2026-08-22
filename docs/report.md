# 海洋模式技术报告

> 静力原始方程 · 谱方法 · JAX 加速 · 数值稳定性

---

## 一、使用的方程

### 1.1 方程组概述

本模型求解的是**静力原始方程**（Hydrostatic Primitive Equations）。它是海洋大尺度环流模拟的标准方程组，被 MOM6、NEMO 等主流海洋模式采用。核心思想是：在垂直方向假设静力平衡（压力等于上方水柱重量），从而省去最昂贵的垂直动量方程，只保留水平动量方程、连续性方程、静水压力方程、状态方程和示踪物方程。

**关键近似：**

| 近似 | 含义 |
|------|------|
| **静力平衡** | $\partial p/\partial z = -\rho g$，垂直方向不加速，压力由上方水重决定 |
| **Boussinesq 近似** | 惯性项中用参考密度 $\rho_0$ 代替真实密度，只在压力梯度力和浮力中保留密度变化 |
| **不可压缩** | 速度场无散度 $\nabla \cdot \mathbf{u} = 0$ |
| **f/β 平面** | Coriolis 参数 $f = f_0 + \beta(y - y_0)$ 在域内线性变化 |

### 1.2 水平动量方程

$$
\frac{\partial u}{\partial t}
= -\left(u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + w\frac{\partial u}{\partial z}\right)
+ f\,v
- \frac{1}{\rho_0}\frac{\partial p}{\partial x}
+ \nu_h \nabla_h^2 u
+ \nu_v \frac{\partial^2 u}{\partial z^2}
+ \frac{\tau_x}{\rho_0 \,\Delta z_{\text{top}}}\,\delta_{\text{top}}
- r_{\text{bot}}\, u_{\text{bot}}
$$

$$
\frac{\partial v}{\partial t}
= -\left(u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + w\frac{\partial v}{\partial z}\right)
- f\,u
- \frac{1}{\rho_0}\frac{\partial p}{\partial y}
+ \nu_h \nabla_h^2 v
+ \nu_v \frac{\partial^2 v}{\partial z^2}
+ \frac{\tau_y}{\rho_0 \,\Delta z_{\text{top}}}\,\delta_{\text{top}}
- r_{\text{bot}}\, v_{\text{bot}}
$$

各项含义：平流（非线性搬运）→ Coriolis 力（地球自转偏转）→ 压力梯度力（压力差驱动）→ 水平扩散（湍流混合）→ 垂直扩散 → 风应力（表层）→ 底摩擦（底层）。

> **注意：** 平流项使用**平流形式**（advective form）而非通量形式（flux form），原因详见第五节。

### 1.3 连续性方程（诊断垂直速度 $w$）

由不可压缩假设：

$$
\frac{\partial w}{\partial z} = -\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right)
$$

$w$ 不是独立预报变量，而是从海底（$w = 0$）向上逐层积分诊断出来。

### 1.4 静水压力方程

$$
p(z) = \underbrace{\rho_0\, g\, \eta}_{\text{正压项（海面高度）}} + \underbrace{g\int_{z}^{0} \rho'(z')\, dz'}_{\text{斜压项（密度积分）}}
$$

压力梯度力分解为正压部分（由海面高度 $\eta$ 驱动，所有深度相同）和斜压部分（由密度异常 $\rho' = \rho - \rho_0$ 的深度积分驱动，随深度变化）。

### 1.5 状态方程

支持两种状态方程：

**线性 EOS（默认）：**

$$
\rho = \rho_0\left[1 - \alpha_T(T - T_{\text{ref}}) + \beta_S(S - S_{\text{ref}})\right]
$$

其中 $\alpha_T = 2.0 \times 10^{-4}\;\text{K}^{-1}$（热膨胀系数），$\beta_S = 7.6 \times 10^{-4}\;\text{psu}^{-1}$（盐收缩系数），$T_{\text{ref}} = 15°\text{C}$，$S_{\text{ref}} = 35\;\text{psu}$，$\rho_0 = 1025\;\text{kg/m}^3$。

**UNESCO 1980 非线性 EOS（可选）：**

$$
\rho(T,S) = \rho_{\text{SMOW}}(T) + B(T)\,S + C(T)\,S^{3/2} + D\,S^2
$$

其中 $\rho_{\text{SMOW}}(T)$ 是标准平均海水密度（5 阶多项式），$B(T)$、$C(T)$ 分别为 4 阶和 2 阶多项式系数。适用于需要精确密度计算的情景。

### 1.6 示踪物方程（温度 $T$、盐度 $S$）

$$
\frac{\partial T}{\partial t}
= -\left(u\frac{\partial T}{\partial x} + v\frac{\partial T}{\partial y} + w\frac{\partial T}{\partial z}\right)
+ \kappa_h \nabla_h^2 T
+ \kappa_v \frac{\partial^2 T}{\partial z^2}
+ \frac{Q_{\text{heat}}}{\rho_0\, C_p\, \Delta z_{\text{top}}}\,\delta_{\text{top}}
$$

$$
\frac{\partial S}{\partial t}
= -\left(u\frac{\partial S}{\partial x} + v\frac{\partial S}{\partial y} + w\frac{\partial S}{\partial z}\right)
+ \kappa_h \nabla_h^2 S
+ \kappa_v \frac{\partial^2 S}{\partial z^2}
$$

### 1.7 预报量与诊断量

| 变量 | 角色 | 方程 |
|------|------|------|
| $u, v$（水平流速） | **预报** | 水平动量方程 |
| $T, S$（温盐） | **预报** | 示踪物方程 |
| $\eta$（海面高度） | **预报** | 自由表面方程（浅水波） |
| $w$（垂直流速） | **诊断** | 连续性方程 |
| $p$（压力） | **诊断** | 静水压力方程 |
| $\rho$（密度） | **诊断** | 状态方程 |

---

## 二、网格方法

### 2.1 网格配置

模型采用 **Arakawa A 网格**（所有变量定义在同一点），水平网格为 $128 \times 128$，垂直 14 层。

| 参数 | 值 |
|------|-----|
| 水平分辨率 | $0.1° \times 0.1°$ |
| 域中心 | $35°\text{N}, 150°\text{E}$（西北太平洋副热带海域） |
| 经度范围 | $[143.6°\text{E},\; 156.3°\text{E}]$ |
| 纬度范围 | $[28.6°\text{N},\; 41.3°\text{N}]$ |
| $dx$（纬度中心处东西间距） | $\approx 9{,}840\;\text{m}$ |
| $dy$（南北间距） | $\approx 11{,}111\;\text{m}$ |

水平网格间距由地球半径和纬度计算：

$$
dx = R_E \cos(\varphi_0) \cdot \Delta\lambda_{\text{rad}}, \qquad dy = R_E \cdot \Delta\varphi_{\text{rad}}
$$

### 2.2 垂直网格

采用非均匀 z-level 网格（z 向下为负，海面 $z=0$）：

```
z = [0, -5, -15, -30, -50, -75, -100, -150, -200, -300, -500, -1000, -2000, -4000]
```

对应的层厚：

```
dz = [5, 10, 15, 20, 25, 25, 50, 50, 100, 200, 500, 1000, 2000]  (m)
```

**设计原则：** 上层密集（5m、10m、15m……），深层稀疏（500m、1000m、2000m）。这是因为海洋上层斜压过程活跃——温度、盐度的剧烈变化主要发生在温跃层（约 100–1000m），需要高分辨率；而深层变化缓慢，可以放粗网格以节省计算量。

### 2.3 地形与掩膜

海深数据来自 **ETOPO2022** 全球地形数据集（$0.1°$ 分辨率）。读取目标区域的子集，生成：

- `depth`：海洋深度场（正值 = 海洋深度，0 = 陆地）
- `ocean_mask` / `land_mask`：海洋/陆地点的布尔掩膜

平均海洋深度从地形数据中提取，用于自由表面浅水波方程中的等效水深 $H_{\text{sw}}$。

### 2.4 边界条件

谱方法要求**周期性边界条件**——这是使用 FFT 的隐含前提。$128 \times 128$ 的网格在 x 和 y 方向都被假定为周期性的。这意味着：

- 模式域必须选在开阔海域（本模型选取西北太平洋副热带海域，陆地占比低）
- 边界处的流场会被"绕回"到对侧边界（周期性环绕）
- 对于区域海洋模拟，这是一种近似；对于全球海洋模拟，则是精确的

垂直方向边界条件：
- 海面：风应力注入（表层），热通量注入（表层），自由表面 $\eta$
- 海底：刚性边界 $w = 0$，线性底摩擦 $-r_{\text{bot}}\,u_{\text{bot}}$

---

## 三、谱方法与向量运算

### 3.1 为什么用谱方法？

偏微分方程中有大量空间导数：$\partial u/\partial x$、$\partial^2 u/\partial z^2$、$\nabla^2 u$、$\partial p/\partial x$…… 

有限差分方法用"看邻居"来近似求导：

$$
\frac{\partial u}{\partial x}\bigg|_{x_i} \approx \frac{u_{i+1} - u_{i-1}}{2\Delta x}
$$

每个点的导数依赖邻居值，引入了空间耦合，内存访问模式差，精度受网格限制。

**谱方法的核心洞察：** 在傅里叶空间里，求导变成乘法。

$$
\frac{\partial u}{\partial x} \quad \longleftrightarrow \quad ik \cdot \hat{u}
$$

其中 $\hat{u} = \mathcal{F}(u)$ 是傅里叶系数，$k$ 是波数。每个频率分量独立地乘以 $ik$，**完全解耦**，无需读取邻居。

### 3.2 水平导数——谱方法实现

所有水平导数通过 FFT 实现：

**一阶导数：**

$$
\frac{\partial u}{\partial x} = \text{IFFT}\left(ik_x \cdot \text{FFT}(u)\right)
$$

```python
def _d_dx(u, p):
    u_hat = jnp.fft.fft(u, axis=0)           # FFT along x
    return jnp.real(jnp.fft.ifft(1j * p.kx * u_hat, axis=0))  # multiply by ik, IFFT back
```

**二阶导数 / 拉普拉斯算子：**

$$
\nabla_h^2 u = \text{IFFT2}\left(-k^2 \cdot \text{FFT2}(u)\right), \qquad k^2 = k_x^2 + k_y^2
$$

**水平散度：**

$$
\nabla_h \cdot \mathbf{u}_h = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}
$$

谱方法的优势在于：
1. **精度无限阶**——在可分辨的波数范围内，导数是精确的（不是近似）
2. **无数值扩散**——不像有限差分会引入截断误差
3. **天然适合周期边界**——FFT 假设的就是周期信号

### 3.3 垂直导数——非均匀有限差分

垂直方向不能用谱方法（非周期边界、非均匀网格），改用有限差分。

**一阶导数（内部，中心差分）：**

$$
\left.\frac{\partial u}{\partial z}\right|_k = \frac{u_{k+1} - u_{k-1}}{\Delta z_{\text{up}} + \Delta z_{\text{dn}}}
$$

其中 $\Delta z_{\text{up}} = |z_{k-1} - z_k|$，$\Delta z_{\text{dn}} = |z_k - z_{k+1}|$。由于网格非均匀，分母是上下两层厚度之和。

**一阶导数（边界，单侧差分）：**

$$
\left.\frac{\partial u}{\partial z}\right|_{\text{top}} = \frac{u_1 - u_0}{\Delta z_{\text{top}}}, \qquad
\left.\frac{\partial u}{\partial z}\right|_{\text{bot}} = \frac{u_{-1} - u_{-2}}{\Delta z_{\text{bot}}}
$$

**二阶导数（内部，非均匀格式）：**

$$
\left.\frac{\partial^2 u}{\partial z^2}\right|_k = \frac{h_p\, u_{k-1} - (h_m + h_p)\, u_k + h_m\, u_{k+1}}{\frac{1}{2} h_m h_p (h_m + h_p)}
$$

其中 $h_m = |z_{k-1} - z_k|$，$h_p = |z_k - z_{k+1}|$。这是非均匀网格上的标准二阶精度格式。

### 3.4 去混淆（Dealiasing）

非线性乘积（如 $u \cdot \partial u/\partial x$）在物理空间逐元素相乘时，会产生**混叠误差**（aliasing error）——高波数能量被"折叠"回低波数，形成虚假的数值噪声。

**原理：** 两个截断到波数 $K$ 的函数相乘，乘积的最高波数是 $2K$。但 FFT 网格只能表示到 $K$，超出部分被折叠回来（混叠）。解决方法是 **2/3 规则**：保留最低 2/3 的波数，截断最高 1/3。

实现方式：

```python
def _dealias_h(field, p):
    field_hat = jnp.fft.fft2(field, axes=(0, 1))   # FFT to spectral space
    field_hat = field_hat * p.dealias_2d             # zero out upper 1/3 wavenumbers
    return jnp.real(jnp.fft.ifft2(field_hat, axes=(0, 1)))  # IFFT back
```

其中 `dealias_2d` 是预计算的掩膜：波数 $|k| < N/3$ 的模式保留（乘 1），其余置零。

所有非线性平流项的输出在返回前都经过去混淆处理：

```python
adv_u = -(u * du_dx + v * du_dy + w * du_dz)
adv_u = _dealias_h(adv_u, p)   # 去混淆
```

### 3.5 向量化运算

整个求解器不使用任何 Python 循环来遍历网格点。所有运算都是**整个场同时计算**的向量化操作：

| 运算 | 实现方式 |
|------|---------|
| 水平导数 | `jnp.fft.fft` / `jnp.fft.ifft`（整个轴一次性变换） |
| 垂直导数 | 数组切片 + 逐元素运算（`u[..., 2:] - u[..., :-2]`） |
| 非线性乘积 | 逐元素乘法（`u * du_dx`） |
| 累积积分（压力、w） | `jnp.cumsum`（沿垂直轴） |
| 波数乘法 | 预计算的 `kx`、`ky`、`k2` 数组广播到 3D |

波数向量预计算并 reshape 为可广播的 3D 形状：

```
kx: (nx, 1, 1)     → 可与 (nx, ny, nz) 场逐元素相乘
ky: (1, ny, 1)
k2: (nx, ny, 1)
```

这样 `1j * p.kx * u_hat` 就是一次逐元素乘法，同时处理所有 $(y, z)$ 点。

---

## 四、JAX 加速

### 4.1 为什么用 JAX？

原始 numpy 版本的单步耗时约 372 ms（$128 \times 128 \times 14$ 网格），模拟一天的海洋演化需要 144 步（$\Delta t = 600\;\text{s}$），总耗时超过 50 秒。如果要做长期模拟或参数扫描，这个速度不可接受。

JAX 提供了三个关键加速手段：

1. **JIT 编译**（`@jax.jit`）：XLA 编译器将 Python 函数编译为优化的机器码
2. **算子融合**：XLA 将多个逐元素操作合并为单次内存遍历
3. **常量折叠**：所有静态参数在编译时固化为 XLA 常量

### 4.2 JIT 编译策略

整个时间步进函数被编译为一个 XLA 计算图：

```python
params = _compute_params(grid, physics, dt, forcing)  # 预计算所有静态参数

@jax.jit
def step(state):
    return _step_impl(state, params)   # params 被捕获为闭包常量
```

**关键设计：** 所有不随时间变化的量——波数 $k_x, k_y$、扩散衰减因子 $e^{-\nu k^2 \Delta t/2}$、Coriolis 旋转角、浅水波频率 $\omega = \sqrt{gHk^2}$、z 网格差分系数、风应力场——都在 `_compute_params` 中**一次性预计算**，然后被 JIT 闭包捕获。XLA 将它们视为编译时常量，进行常量折叠和死代码消除。

这意味着每一步只需要：
- 输入：当前状态 $(u, v, T, S, \eta)$ —— 5 个数组
- 输出：下一步状态 —— 5 个数组
- 中间不产生 Python 开销

### 4.3 预计算的衰减因子

线性部分的扩散和浅水波演化有精确解析解，预计算为衰减/旋转因子：

**水平扩散（矩阵指数）：**

$$
u(t + \Delta t/2) = \text{IFFT}\left(e^{-\nu_h k^2 \Delta t/2} \cdot \text{FFT}(u)\right)
$$

衰减因子 `decay_u = exp(-nu_h * k2 * dt/2)` 在编译时固化为常量，运行时只需一次 FFT + 逐元素乘法 + IFFT。

**Coriolis 精确旋转：**

$$
\begin{pmatrix} u \\ v \end{pmatrix}_{t+\Delta t/2}
=
\begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} u \\ v \end{pmatrix}_t, \qquad \theta = f_0 \Delta t/2
$$

**自由表面浅水波（精确矩阵指数）：**

自由表面的演化等价于线性浅水方程，每个波数独立地以频率 $\omega = \sqrt{gHk^2}$ 振荡。用 Rodrigues 公式精确求解：

$$
\eta_{\text{new}} = \cos(\omega \Delta t/2)\,\eta - H \frac{\sin(\omega \Delta t/2)}{\omega}\,(ik_x \bar{u} + ik_y \bar{v})
$$

`sin`、`cos`、`sin/ω` 全部预计算。

### 4.4 时间积分方案

采用 **Strang 分裂 IMEX** 方案：

$$
\mathbf{U}(t + \Delta t) = e^{\mathcal{L}\,\Delta t/2}\;\left[\mathbf{U}(t) + \mathcal{N}(\mathbf{U}^*)\,\Delta t\right]\;e^{\mathcal{L}\,\Delta t/2}
$$

分三步执行：

1. **线性半步** $e^{\mathcal{L}\,\Delta t/2}$：水平扩散（谱衰减）+ f₀ Coriolis 旋转 + 自由表面浅水波（精确矩阵指数）
2. **非线性整步** $\mathcal{N}(\Delta t)$：平流 + 斜压压力梯度 + 垂直扩散 + β 平面修正 + 表面强迫 + 底摩擦
3. **线性半步** $e^{\mathcal{L}\,\Delta t/2}$：同步骤 1

非线性整步使用**前向-后向 RK2**（Forward-Backward RK2）格式：

```
预测：
  T_pred = T + dt * R_T(state)          [前向：用旧速度更新温度]
  u_pred = u + dt * R_u(state_T_pred)   [后向：用新温度算压力梯度更新速度]
校正：
  T_new = T + 0.5*dt * (R_T(state) + R_T(state_pred))
  u_new = u + 0.5*dt * (R_u(state) + R_u(state_T_new))
```

这种耦合方式解决了斜压 PGF-示踪物耦合产生的内重力波不稳定问题（详见第五节）。

### 4.5 性能数据

在 CPU（x86-64，`jax_enable_x64=True`）上的基准测试结果（`src/bench_compare.py`，小扰动无强迫配置、$\Delta t = 300\;\text{s}$、20 步、多次计时取稳态）：

| 指标 | 数值 |
|------|------|
| 网格规模 | $128 \times 128 \times 14 \approx 23$ 万点 |
| JIT 编译 + 首步 | $\approx 0.92\;\text{s}$ |
| 稳态每步耗时（JAX） | $\approx 74.6\;\text{ms}$ |
| 稳态每步耗时（numpy） | $\approx 280.7\;\text{ms}$ |
| JAX 加速比 | $\approx 3.76\times$ |
| 模拟时间/步 | $300\;\text{s}$ |
| 实测模拟实时比 | numpy $\sim 1069\times$，JAX $\sim 4019\times$（模拟 1 天 JAX 需 $\sim 21.5\;\text{s}$） |

**波动传播校验**：实测外重力波速 $c_{\text{meas}} = 186.46\;\text{m/s}$ 对理论值 $c_{\text{theory}} = \sqrt{gH} = 198.09\;\text{m/s}$，误差 $\approx 5.87\%$；总质量漂移 $7.11\times 10^{-15}$（浮点精度内，PASS）。

**时间步收敛测试**（以 $\Delta t = 10\;\text{s}$ 为参考，$t = 1800\;\text{s}$）：

| $\Delta t$（s） | $\max\|\Delta T\|$（°C） | $\max\|\Delta u\|$（m/s） | $\max\|\Delta \eta\|$（m） |
|------|------|------|------|
| 30 | $2.37\times 10^{-11}$ | $1.46\times 10^{-7}$ | $7.73\times 10^{-7}$ |
| 60 | $5.97\times 10^{-11}$ | $3.58\times 10^{-7}$ | $1.82\times 10^{-6}$ |
| 120 | $1.34\times 10^{-10}$ | $7.92\times 10^{-7}$ | $3.66\times 10^{-6}$ |
| 300 | $3.79\times 10^{-10}$ | $2.28\times 10^{-6}$ | $8.25\times 10^{-6}$ |

经验收敛阶约 $\mathcal{O}(\Delta t^{1.20})$，低于理论 $\mathcal{O}(\Delta t^2)$（受 Strang 分裂与显式前向-后向步的截断误差影响）。


### 4.6 与 numpy 版本的对比
| 特性 | numpy 版本 | JAX 版本 |
|------|-----------|----------|
| 每步耗时（稳态） | $\approx 280.7\;\text{ms}$ | $\approx 74.6\;\text{ms}$ |
| JAX 加速比 | $1\times$（基准） | $\approx 3.76\times$ |
| FFT 调用次数 | 每个导数 2 次（FFT + IFFT） | 相同，但被 XLA 融合优化 |
| 内存分配 | 每个中间结果分配新数组 | XLA 算子融合，减少中间分配 |
| Python 开销 | 每步有 Python 解释器开销 | 编译后零 Python 开销 |
| GPU 支持 | 不支持 | 改一行配置即可（`jax.devices('gpu')`） |

> **有效性说明：** 该速度对比仅在稳定的小扰动·无强迫配置下成立。numpy 版本仍使用通量形式 + 仅水平平流的旧公式（存在 5.2 节的虚假源项问题），在真实强迫下层流会溢出；因此对逐场精度的对比仅在小扰动/无强迫算例上有意义。

---

## 五、梯度爆炸问题

### 5.1 问题概述

在开发过程中，模型遇到了**数值爆炸**（blow-up）问题——模拟若干步后，温度、速度等场迅速增长到无穷大（NaN）。这个问题出现了两次，原因不同。

### 5.2 第一次爆炸：通量形式的虚假源项（已解决）

#### 什么时候出现

在模型初期版本中，使用**通量形式**（flux form）的平流项，并且**只做水平平流，不包含垂直平流**。即使初始场只是 $0.01°\text{C}$ 量级的小扰动（均匀温度场加随机噪声），模拟数十步后温度就发散到 NaN。

#### 为什么会出现

通量形式的平流写为：

$$
\frac{\partial T}{\partial t}\bigg|_{\text{flux}} = -\frac{\partial(uT)}{\partial x} - \frac{\partial(vT)}{\partial y}
$$

展开：

$$
-\frac{\partial(uT)}{\partial x} - \frac{\partial(vT)}{\partial y}
= -\left(u\frac{\partial T}{\partial x} + v\frac{\partial T}{\partial y}\right) - T\underbrace{\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right)}_{\nabla_h \cdot \mathbf{u}_h}
$$

即通量形式 = 平流形式 + $T \cdot \nabla_h \cdot \mathbf{u}_h$。

**关键矛盾：** 在二维浅水模型中，水平散度 $\nabla_h \cdot \mathbf{u}_h = 0$（水平不可压缩），所以多出的项为零，两种形式等价。但在三维静力模型中，水平散度**不为零**——它由连续性方程 $\partial w/\partial z = -\nabla_h \cdot \mathbf{u}_h$ 与垂直速度平衡。

如果只做水平通量形式平流、忽略垂直平流，就多出了一个非物理的源项：

$$
\text{虚假源项} = -T \cdot \nabla_h \cdot \mathbf{u}_h
$$

这个项正比于温度本身和水平散度。在辐聚区（$\nabla_h \cdot \mathbf{u}_h < 0$），它表现为一个**正温度源**——温度越高，源越强，温度进一步升高——形成正反馈回路，导致指数增长。

同样的问题也出现在动量方程中：通量形式的水平动量平流多出了 $u \cdot \nabla_h \cdot \mathbf{u}_h$ 的虚假源项。

#### 修复方法

改用**平流形式**（advective form），并加入完整的三维平流（包含垂直方向）：

$$
\frac{\partial T}{\partial t}\bigg|_{\text{adv}} = -\left(u\frac{\partial T}{\partial x} + v\frac{\partial T}{\partial y} + w\frac{\partial T}{\partial z}\right)
$$

平流形式不包含 $T \cdot \nabla_h \cdot \mathbf{u}_h$ 项，从根源上消除了虚假源。垂直速度 $w$ 由连续性方程诊断得出，从海底向上积分。

修复后，小扰动模拟（5 种测试场景）全部稳定通过。

### 5.3 第二次爆炸：真实层结下的大时间步不稳定（已解决）

#### 什么时候出现

将初始场从理想化的小扰动切换为 **WOA2023 实测气候态温盐数据**后，出现了新的不稳定。

WOA2023 数据提供了西北太平洋区域的实测温度和盐度三维场：

| 参数 | 值 |
|------|-----|
| T 范围 | $1.45°\text{C}$（深层）$\sim 23.88°\text{C}$（表层） |
| S 范围 | $33.27 \sim 34.92\;\text{psu}$ |
| 垂直层结 | 表层到深层温差约 $22°\text{C}$ |

使用 $\Delta t = 300\;\text{s}$ 积分时，约 **30–40 步**后温度场发散为 NaN。表现为：

- `max|T|` 在前 10 步正常（$\sim 24°\text{C}$），到第 20 步迅速增长到 $\sim 3.6$（单位异常），第 30 步达到 $\sim 3.5 \times 10^3$，第 40 步变为 $-\infty$（NaN）
- `max|u|` 从 0 增长到 $\sim 0.4\;\text{m/s}$（第 10 步）到 $\sim 3.6\;\text{m/s}$（第 20 步）后发散
- 无强迫、有强迫（风应力 + 热通量）、仅风应力三种测试都在相近步数爆炸

将时间步缩小到 $\Delta t = 30\;\text{s}$ 时，300 步内保持稳定，但温度缓慢漂移（$23.88°\text{C} \to 29.5°\text{C}$），说明存在缓慢的能量错误累积。

#### 为什么会出现

经排查，这个问题**不是**前一次的虚假源项问题——平流形式已修复，且去混淆（2/3 规则）已加入但未能解决。当前的分析指向以下几个可能原因：

**（1）强斜压梯度 → 大压力梯度力**

WOA 实测数据在 14 层垂直网格上有 $22°\text{C}$ 的温差，表层 5m 内的温度梯度极为陡峭。密度异常通过状态方程产生强烈的斜压压力梯度力。这个压力梯度力驱动斜压流速，斜压流速改变温度场分布，新的温度场又产生新的压力梯度力——形成一个非线性耦合回路。在大时间步（$\Delta t = 300\;\text{s}$）下，显式 RK2 格式可能无法稳定地处理这种快速耦合。

**（2）垂直分辨率不足**

最上层仅 5m 厚，但其下是 10m、15m、20m……在 $22°\text{C}$ 温差跨越温跃层时，5m 层与 10m 层之间的温度梯度远大于深层。非均匀垂直有限差分在强梯度处的截断误差更大，可能引入数值噪声。

**（3）前向-后向 RK2 的显式稳定性限制**

非线性整步使用显式 RK2，其稳定域有限。虽然前向-后后耦合解决了内重力波的纯虚特征值问题（将特征值从虚轴移到左半平面），但对于大振幅、高波数的斜压模态，特征值的实部可能仍超出 RK2 的稳定域。

**（4）线性 CFL 判据通过但非线性不稳定**

线性 CFL 数 $\text{CFL} = u \cdot \Delta t / \Delta x \approx 0.003$（$u = 1\;\text{m/s}$，$\Delta t = 300\;\text{s}$，$\Delta x \approx 9840\;\text{m}$），远低于 1。但这只约束了线性平流稳定性。真实的非线性项（平流 × 压力梯度 × 浮力耦合）可以有更严格的隐式稳定性约束，尤其是在强层结下。

#### 当前状态（更新）

| 配置 | 稳定性 |
|------|--------|
| 小扰动 + $\Delta t = 300\;\text{s}$ | ✅ 稳定 |
| WOA 实测 + 无强迫 + $\Delta t = 300\;\text{s}$ | ✅ 稳定（扫描实验，见 5.4） |
| WOA 实测 + 无强迫 + $\Delta t = 600\;\text{s}$ | ⚠️ $>10\;\text{m/s}$（DEGRADED） |
| WOA 实测 + 带强迫 + $\Delta t = 300\;\text{s}$ | ✅ 稳定（`33fb58c` biharmonic 修复后，100 步通过） |
| WOA 实测 + 无强迫 + $\Delta t = 300\;\text{s}$ | ✅ 稳定（150 步，biverify 实测） |
| WOA 实测数据 + $\Delta t = 30\;\text{s}$ | ✅ 稳定（修复后温度漂移消除，见 5.4 修正） |

---

### 5.4 稳定性域扫描：内部波形稳定限（新增实测）

用 `src/stability_scan.py` 在**真实 WOA 层结、无强迫**配置下做了一套系统扫描（$\Delta t = 10 \sim 900\;\text{s}$，各积分 2 小时），以区分"外层重力波 CFL"与"内部斜压波实际稳定限"：

| dt(s) | 结果 | max\|u\|（m/s） |
|-------|-------|----------------|
| 10–450 | STABLE | 2.68 → 3.17 |
| 600 | DEGRADED | 1.13e+01 |
| 900 | DEGRADED | 2.25e+01 |

- 外层模波速 $c_{\text{ext}} = \sqrt{gH}=198\;\text{m/s}$，其 CFL $\Delta t \lesssim 46\;\text{s}$。
- 但前向-后向 RK2 通过 forward-backward 半隐式处理外层重力 + 科氏项，**外层模并非稳定限**。
- 实际稳定限由**内部（斜压）波**决定，可持续到 $\Delta t \approx 450\text{–}600\;\text{s}$，特征流速约 $2.7\text{–}3.2\;\text{m/s}$（量级合理的斜压调整流）。
- 首次越界在 $\Delta t = 600\;\text{s}$（max|u| 升到 >10 m/s）。

**结论修正**：单纯斜压调整在 $\Delta t = 300\;\text{s}$ 下是稳定的，5.3 所述 dt=300 爆炸主要源于**强迫项**（风应力 + 热通量）在实况层结下的耦合不稳定与温度漂移，而非裸 CFL 越界。这一强迫耦合不稳定已由 **scale-selective biharmonic 黏性/扩散修复（`33fb58c`）** 消除：在 WOA 实况初始场 + 三种配置（无强迫、Stommel 风应力 + 经向热通量强制加热、仅风应力）下，$\Delta t = 300\;\text{s}$ 均稳定积分 100 步（约 8.3 小时模拟），max|u| 增长至约 $3.0\;\text{m/s}$（合理的斜压调整流），温度保持在 $24.4\text{–}24.6°\text{C}$ 无漂移（证据：`logs/verify_real_run_biverify.log`，ALL TESTS PASSED）。

---

### 5.5 月尺度强迫漂移：周期谱缝上的非周期强迫 → NW 角边界热泵（已解决）

在 5.2/5.3 的尺度选择性 biharmonic 修复稳定了 $dt=300\;\text{s}$ 短积分之后，把积分拉长到**月尺度（30 天，约 8640 步）**在"真实 WOA 初始场 + Stommel 风应力 + 经向热通量强迫"下又重新出现温度漂移（`verify_30d_conv_restore30.log`：max|T| 由 $23.88°\text{C}$ 单调爬升至 $43.78°\text{C}$）。

**真正根因**：偏微分域在 $x$ 与 $y$ 两个方向都是**周期性的**（伪谱 FFT），而强迫剖面在 $y$ 方向**非周期**（Stommel 风应力与经向热通量在南北两条边取相反的非零值）。这在**周期谱缝**处制造了一个阶梯不连续，把虚假的**网格尺度强迫能量钉死在西北角（NW corner，$i=0..1,\;j=115..124$）**，形成一处比海表冷却更强的**边界"热泵"**。热点定位脚本（`diag_month_drift.py`）确认热点位于 $k=0$（海表）且 $\text{Tcol\_top}\sim22.9°\text{C}$ / $\text{Tcol\_bot}\sim1.49°\text{C}$，上暖下冷的准稳定柱。

**组合修复**（三个物理上正交的环节）：

1. **强迫渐变**（`forcing.py` 新增 `_taper_y`）：对两 $y$ 边在 `taper_cells=8` 格内把非周期强迫以升余弦渐变到零，消除周期缝上的阶梯不连续（NW 角热泵随之消失）。
2. **对流调整**（`jax_solver.py` 的 `kappa_conv` 垂直扩散）：对静力不稳定柱（列内任一界面不稳定）做整柱垂直混合。这一项**只有在边界伪源被清除后才生效**——仅渐变无对流仍漂移（10 天 max|T| $28.5°\text{C}$），渐变+对流即稳定（10 天 max|T| $23.5°\text{C}$）。
3. **海表温盐恢复（Haney, $\tau=5$ 天）**（`--restore-days=5` 锚定到初始上表层 $T$）：固定经向 $Q$ 模式会持续把热泵进内部下沉辐合点（Stommel 回归支，$i\approx53,\;j\approx20,\;k=0$）。没有海气负反馈时热量在月中段累积（无恢复 30 天峰值 $35.15°\text{C}/28.35°\text{C}$）；SST 恢复把运行收敛到**稳定平衡**。

**判定口径（经用户确认）**：受热受迫海洋本就会平衡到高于初始 WOA 海表温度的状态（此处抬升约 $3\text{–}4°\text{C}$）。因此月尺度"无漂移"的科学判据是**收敛**而非锚定初始 SST：

- `monotonic_drift=False`——末四分之一段 max|T| 不再相对前半段持续攀升（真正"累积"的信号）；
- `amplitude_bounded`——末四分之一段 max|T| 低于（初始最大 + $12°\text{C}$ 绝对上限），仅真正的网格/耦合失控会触顶；
- 无 NaN/Inf，max|u| < $10\;\text{m/s}$。

**逐档验证结果**（30 天实测，新的强制平衡判据对全部历史日志的判别一致）：

| 配置 | max|T| 前半段 | 末四分之一段 | monotonic | 幅值有界 | 判定 |
|------|------------|------------|-----------|---------|------|
| 无恢复（taper+conv） | 27.28 | 35.15 | True | True | **FAIL** |
| 无渐变（conv+restore30） | 30.89 | 44.17 | True | False | **FAIL** |
| restore-days=10 | 27.46 | 28.33 | False | True | **PASS** |
| **restore-days=5（工作配置）** | 26.44 | 27.35 | False | True | **PASS** |

结论：**force taper + 对流调整 + SST 恢复（$\tau=5\;\text{d}$）** 组合修正在真实 WOA 初始场下把 30 天强迫积分稳定到 `restore-days=5` 的受迫平衡（max|T| $\approx26.6°\text{C}$，max|u|$\approx1.9\;\text{m/s}$，无 NaN）。`restore-days=5` 即为当前工作配置（证据：`logs/verify_30d_taper_conv_restore5.log`，PASS）。

## 附录：模型参数汇总

| 类别 | 参数 | 值 |
|------|------|-----|
| 物理 | $\rho_0$ | $1025\;\text{kg/m}^3$ |
| | $\alpha_T$ | $2.0 \times 10^{-4}\;\text{K}^{-1}$ |
| | $\beta_S$ | $7.6 \times 10^{-4}\;\text{psu}^{-1}$ |
| | $C_p$ | $3992\;\text{J/(kg·°C)}$ |
| | $g$ | $9.81\;\text{m/s}^2$ |
| 湍流 | $\nu_h$（水平黏性） | $100\;\text{m}^2/\text{s}$ |
| | $\nu_v$（垂直黏性） | $1.0 \times 10^{-4}\;\text{m}^2/\text{s}$ |
| | $\kappa_h$（水平扩散） | $100\;\text{m}^2/\text{s}$ |
| | $\kappa_v$（垂直扩散） | $1.0 \times 10^{-5}\;\text{m}^2/\text{s}$ |
| 底摩擦 | $r_{\text{bot}}$（线性） | $1.0 \times 10^{-3}\;\text{s}^{-1}$ |
| | $C_d$（二次） | $2.5 \times 10^{-3}$ |
| 网格 | 水平 | $128 \times 128$（$0.1°$） |
| | 垂直 | 14 层 z-level（非均匀） |
| 时间 | $\Delta t$ | $300\;\text{s}$（理想化）/ $30\;\text{s}$（实测数据） |
| | 总时长 | $86{,}400\;\text{s}$（1 天默认） |
