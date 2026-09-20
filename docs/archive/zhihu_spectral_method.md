# 为什么求导可以快 20 倍？——谱方法：把偏微分方程变成矩阵乘法

> **📦 已归档（Archived）** — 2026-09-20 仓库整理时移入 `docs/archive/`。
>
> 本文属于**已退役的区域谱模式（regional spectral solver）**时期的工作、
> 过程性工作日志，或已被后续文档取代的早期版本。保留它只是为了留存历史推理链，
> **不代表当前主线**。
>
> 当前主线是**全球有限差分模式**（`src/jax_solver_global.py`，见
> [`docs/solver_technical_report_zh.md`](../solver_technical_report_zh.md) 与
> [`docs/decisions.md`](../decisions.md)）。文档索引见 [`docs/README.md`](../README.md)。


---

## 一、问题：求导为什么慢？

假设要用计算机模拟海洋流动。这需要求解一组偏微分方程（静力原始方程）：

$$
\frac{\partial u}{\partial t} = -\underbrace{\left(u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}\right)}_{\text{平流}} + \underbrace{f v}_{\text{科氏力}} - \underbrace{\frac{1}{\rho_0}\frac{\partial p}{\partial x}}_{\text{压力梯度}} + \underbrace{\nu \nabla^2 u}_{\text{扩散}} + \cdots
$$

这个方程里有大量的空间导数：$\partial u/\partial x$、$\partial u/\partial y$、$\nabla^2 u$、$\partial p/\partial x$……

计算机不能求解析导数，只能做**离散近似**。最经典的方法是有限差分：

$$
\frac{\partial u}{\partial x}\bigg|_{x_i} \approx \frac{u_{i+1} - u_{i-1}}{2\Delta x}
$$

看起来很简单。但问题藏在细节里。

### 有限差分的困境

假设你的计算网格是 $128 \times 128 \times 14$（水平 128×128，垂直 14 层），总共约 23 万个网格点。

每个点求一次 $\partial u / \partial x$，需要读取左右邻居的值。这意味着：

```
物理空间求导的本质：每个点的结果依赖邻居 → 空间耦合

点 i 的导数 = (u[i+1] - u[i-1]) / (2*dx)
           ↑          ↑
        需要读取      需要读取
```

在二维平面上，每个点需要读取 4 个邻居（上下左右）。如果用高阶格式（5 点、7 点 stencil），邻居更多。

这带来三个问题：

| 问题 | 说明 |
|------|------|
| **内存访问模式差** | 每个点要读不相邻的内存位置，CPU 缓存命中率低 |
| **无法完全并行** | 虽然各点计算独立，但邻居读取引入了数据依赖 |
| **精度依赖网格密度** | 二阶差分误差 $O(\Delta x^2)$，要达到高精度必须加密网格，计算量爆炸 |

以 $128 \times 128 \times 14$ 的网格为例，用 numpy 实现的有限差分版本，**单步耗时约 372 ms**。这太慢了。

**有没有一种方法，让求导变成每个点独立计算，不依赖邻居？**

---

## 二、关键洞察：傅里叶空间里，求导就是乘法

这是整个谱方法的核心。一句话讲完：

$$
\text{物理空间求导 } \frac{\partial u}{\partial x} \quad \longleftrightarrow \quad \text{傅里叶空间乘法 } ik \cdot \hat{u}
$$

其中 $\hat{u} = \mathcal{F}(u)$ 是 $u$ 的傅里叶变换，$k$ 是波数（频率），$i$ 是虚数单位。

### 为什么？

回忆傅里叶变换的定义。任何周期函数 $u(x)$ 可以分解为正弦波的叠加：

$$
u(x) = \sum_k \hat{u}(k) \, e^{ikx}
$$

其中 $\hat{u}(k)$ 是波数为 $k$ 的分量振幅。对两边求导：

$$
\frac{\partial u}{\partial x} = \sum_k \hat{u}(k) \cdot \frac{\partial}{\partial x} e^{ikx} = \sum_k \hat{u}(k) \cdot ik \cdot e^{ikx}
$$

**求导只作用于 $e^{ikx}$，结果就是乘一个 $ik$。** 每个傅里叶模式独立地被乘以 $ik$，模式之间没有任何耦合。

### 这意味着什么？

在物理空间，求导是"看邻居"（有限差分 stencil）：

```
物理空间：∂u/∂x[i] = (u[i+1] - u[i-1]) / (2*dx)
                      ↑          ↑
                   读取邻居    读取邻居   ← 空间耦合
```

在傅里叶空间，求导是"每个频率乘一个系数"：

```
傅里叶空间：∂û/∂x[k] = ik * û[k]
                       ↑     ↑
                     乘常数  自己   ← 完全独立！
```

**每个点只用自己的值乘一个常数，不需要任何邻居。** 这就是"对角矩阵"——所有频率分量解耦，求导变成逐元素乘法。

二阶导数同理：

$$
\frac{\partial^2 u}{\partial x^2} \longleftrightarrow (ik)^2 \hat{u} = -k^2 \hat{u}
$$

拉普拉斯算子 $\nabla^2 = \partial^2/\partial x^2 + \partial^2/\partial y^2$ 变成：

$$
\nabla^2 u \longleftrightarrow -(k_x^2 + k_y^2) \hat{u} = -k^2 \hat{u}
$$

**一次乘法搞定，不需要 stencil，不需要邻居读取。**

### 完整流程

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  物理空间 u  │ FFT │  傅里叶空间 û │ ×ik │  ik · û (逐元素) │ IFFT│  物理空间 ∂u/∂x│
│  (N 个点)   │ ──→ │  (N 个频率)   │ ──→ │  (N 个独立乘法)  │ ──→ │  (N 个点)     │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
```

整个过程只有三步：**FFT → 乘法 → IFFT**。

其中 FFT 的复杂度是 $O(N \log N)$，而有限差分是 $O(N)$。看起来 FFT 更慢？但关键在于：

- FFT 的 $O(N \log N)$ 是**高度优化的批量运算**，CPU/GPU 都有硬件级实现
- 有限差分的 $O(N)$ 虽然线性，但**内存访问模式差**，实际性能远低于理论
- 谱方法的精度是**谱收敛**（指数收敛），比有限差分的 $O(\Delta x^2)$ 快太多——同样的精度，谱方法可以用更少的网格点

---

## 三、非线性问题：伪谱方法

如果方程只有线性项（扩散、科氏力），那在傅里叶空间一切都很美好——全是逐元素乘法。

但流体方程有**非线性项**——平流项 $u \cdot \nabla u$。两个场相乘在傅里叶空间是卷积，$O(N^2)$ 复杂度，比回到物理空间做乘法还慢。

解决方案就是**伪谱方法**（Pseudospectral Method）：

```
非线性项的计算流程：

1. u, v 在物理空间                    ← 已知 u(x), v(x)
2. u*v 在物理空间相乘                  ← 逐点乘法，O(N)
3. FFT(u*v) 进入傅里叶空间             ← O(N log N)
4. 乘 ik 求导 + 乘 2/3 去混叠掩码       ← 逐元素乘法，O(N)
5. IFFT 回到物理空间                   ← O(N log N)
```

**非线性乘积在物理空间做（简单），导数在傅里叶空间做（高效），两者通过 FFT 桥接。**

### 去混叠（Dealising）：为什么需要 2/3 规则

物理空间的乘法 $u \cdot v$ 会产生新的高频分量（两个频率 $k_1, k_2$ 相乘产生 $k_1+k_2$ 和 $k_1-k_2$）。如果这些高频分量超过了网格能表示的最大波数，就会"折叠"回低频，污染计算结果——这叫**混叠误差**。

解决方法很简单粗暴：**直接砍掉高 1/3 的频率**。

```
2/3 去混叠规则：
  保留 |k| < (2/3) * k_max 的频率
  将 |k| > (2/3) * k_max 的频率置零

  在代码中就是一个逐元素掩码乘法：
  û_dealiased = û * mask    ← mask 是 0 或 1，O(N) 逐元素操作
```

这个掩码可以和求导算子 $ik$ **合并**——两者都是傅里叶空间的逐元素乘法，可以一次完成：

$$
\text{IFFT}\Big(\underbrace{\text{mask} \cdot ik}_{\text{一次乘法}} \cdot \hat{u}\Big)
$$

这里有一个关键的工程优化：**在傅里叶空间把去混叠和求导链式组合，省掉约 50% 的 FFT 调用。**

---

## 四、从数学到矩阵并行

现在把上面的数学翻译成代码。以一维导数为例：

```python
import numpy as np

def d_dx(u, dx):
    """谱方法求导：∂u/∂x = IFFT(ik · FFT(u))"""
    nx = len(u)
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)   # 波数 [1/m]
    u_hat = np.fft.fft(u)                       # FFT: 物理空间 → 频率空间
    du_hat = 1j * k * u_hat                     # 逐元素乘法（求导）
    return np.real(np.fft.ifft(du_hat))          # IFFT: 频率空间 → 物理空间
```

核心就 4 行。注意第三行 `1j * k * u_hat`——这是**逐元素乘法**，每个频率分量独立计算，没有任何数据依赖。

### 在二维、三维上

二维情况完全一样，只是 FFT 变成二维 FFT：

```python
def d_dx_2d(u, dx):
    """二维谱方法求 ∂u/∂x"""
    nx, ny = u.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    kx = kx.reshape(nx, 1)              # 广播到 (nx, ny)
    u_hat = np.fft.fft2(u)               # 二维 FFT
    return np.real(np.fft.ifft2(1j * kx * u_hat))
```

拉普拉斯算子 $\nabla^2$：

```python
def laplacian_2d(u, dx, dy):
    """谱方法求拉普拉斯：∇²u = IFFT(-(kx² + ky²) · FFT(u))"""
    kx = 2 * np.pi * np.fft.fftfreq(u.shape[0], d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(u.shape[1], d=dy)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k2 = kx**2 + ky**2
    u_hat = np.fft.fft2(u)
    return np.real(np.fft.ifft2(-k2 * u_hat))
```

注意一个重要的事实：**波数 $k_x, k_y, k^2$ 都是预计算的常数数组**。在时间积分循环中，它们不变——每次求导只是用同一个常数数组乘以当前的 $\hat{u}$。

这就是你说的"转为线性运算"——偏微分方程中的所有线性算子（导数、拉普拉斯、扩散）在傅里叶空间都是**对角矩阵**，即逐元素乘法。

### 为什么这天然适合并行

| 操作 | 物理空间（有限差分） | 傅里叶空间（谱方法） |
|------|-------------------|-------------------|
| 求导 | 读取邻居，stencil 计算 | 逐元素乘常数 $ik$ |
| 拉普拉斯 | 读取 4-8 个邻居 | 逐元素乘常数 $-k^2$ |
| 扩散时间步 | 隐式求解线性方程组 | 逐元素乘 $\exp(-\nu k^2 \Delta t)$ |
| 数据依赖 | 有（邻居读取） | 无（每点独立） |
| 并行度 | 受内存访问限制 | **完美并行**（逐元素） |

**谱方法把空间耦合的偏微分方程算子，变成了完全解耦的逐元素运算。而逐元素运算正是 GPU 最擅长的事——数万个核心同时执行相同的乘法。**

---

## 五、和神经网络执行模型的类比

这里是本文最核心的类比。

### 神经网络的前向传播

一个全连接层：

$$
\mathbf{y} = \sigma(\mathbf{W} \cdot \mathbf{x} + \mathbf{b})
$$

- 输入向量 $\mathbf{x}$，乘权重矩阵 $\mathbf{W}$，加偏置 $\mathbf{b}$，过激活函数 $\sigma$
- GPU 用 cuBLAS 做矩阵乘法，数万核心并行

### 谱方法的一步时间积分

$$
\hat{u}(t+\Delta t) = \underbrace{e^{-\nu k^2 \Delta t}}_{\text{预计算常数}} \cdot \hat{u}(t) + \Delta t \cdot \underbrace{\text{IFFT}(ik \cdot \text{FFT}(u \cdot u))}_{\text{非线性项}}
$$

- 傅里叶系数 $\hat{u}$，乘预计算常数 $e^{-\nu k^2 \Delta t}$，加非线性项
- GPU 用 cuFFT 做 FFT，用逐元素乘法做导数

**结构完全一样：线性变换（矩阵乘 / 逐元素乘）+ 非线性变换（激活函数 / FFT 桥接的物理空间乘积）。**

这意味着谱方法的每一步时间积分，可以像神经网络的前向传播一样被编译和优化。

### JAX 编译器的角色

以 JAX（Google 的高性能数值计算框架）为例。JAX 的核心能力是 `@jax.jit`——通过 XLA 编译器把 Python 函数编译成优化的机器码。

```
没有 JIT（numpy 版本）:
  每步执行约 30 个独立的 Python 函数调用
  每个调用：Python 解释器 → 类型检查 → 数组分配 → 执行 → 返回
  开销巨大

有 JIT（JAX 版本）:
  整个 step() 函数被编译成一个 XLA 计算图
  XLA 做三件事:
    1. 内核融合 — 30 个操作合并成 1 个 GPU 内核
    2. 常量折叠 — 波数 k、衰减因子 exp(-νk²Δt) 编入计算图
    3. 内存优化 — 中间结果留在寄存器，不分配临时数组
```

具体对比：

```python
# numpy 版本：每个操作是独立的 Python 调用
uu = u * u                          # 分配数组 → 执行 → 返回
uu_hat = fft2(uu)                   # 分配数组 → 执行 → 返回
uu_hat *= dealias_mask              # 分配数组 → 执行 → 返回
du = ifft2(1j * kx * uu_hat)        # 分配数组 → 执行 → 返回
# ... 重复 30 次，每次都有 Python 开销 + 内存分配

# JAX 版本：整个时间步编译成一个内核
@jax.jit
def step(state):
    # XLA 编译器看到的是整个计算图：
    # FFT → 乘 ik → 乘 mask → IFFT → 加 → 乘 dt → ...
    # 融合成一个内核，零 Python 开销，零中间分配
    return _step_impl(state, params)
```

### 实测加速比

在一个海洋求解器上（128×128×14 网格，静力原始方程）的实测数据：

| 版本 | 每步耗时 | 加速比 | 优化手段 |
|------|---------|--------|---------|
| numpy 循环版 | 372 ms | 1.0× | 基线 |
| numpy 向量化 + scipy FFT | 230 ms | 1.6× | 向量化 + 多线程 FFT |
| **JAX JIT（CPU）** | **18.7 ms** | **19.9×** | XLA 内核融合 + 常量折叠 |
| JAX GPU（预估） | ~1-2 ms | ~200-370× | 数千核心并行 FFT |

正确性验证：JAX 与 numpy 结果逐步对比，最大误差 $2.8 \times 10^{-13}$（float64 精度极限），算法完全等价。

---

## 六、扩散方程：一个更优雅的例子

线性扩散方程是展示谱方法优势的最佳案例：

$$
\frac{\partial u}{\partial t} = \nu \nabla^2 u
$$

### 有限差分做法

用隐式格式（Crank-Nicolson）：

$$
\frac{u^{n+1} - u^n}{\Delta t} = \frac{\nu}{2}\left(\nabla^2 u^{n+1} + \nabla^2 u^n\right)
$$

整理得：

$$
\left(I - \frac{\nu \Delta t}{2} \nabla^2\right) u^{n+1} = \left(I + \frac{\nu \Delta t}{2} \nabla^2\right) u^n
$$

这是一个线性方程组 $\mathbf{A} \mathbf{u}^{n+1} = \mathbf{b}$。在二维网格上，$\mathbf{A}$ 是一个稀疏矩阵，但仍然需要求解线性方程组——通常用共轭梯度法，复杂度 $O(N^{1.5})$ 左右。

### 谱方法做法

在傅里叶空间，$\nabla^2$ 变成 $-k^2$，方程变成每个频率独立的常微分方程：

$$
\frac{d\hat{u}}{dt} = -\nu k^2 \hat{u}
$$

解是：

$$
\hat{u}(t+\Delta t) = e^{-\nu k^2 \Delta t} \cdot \hat{u}(t)
$$

**一个乘法。** 不需要解线性方程组，不需要迭代。而且：

- 衰减因子 $e^{-\nu k^2 \Delta t}$ 是预计算的常数
- 对任意大的 $\Delta t$ 都稳定（$e^{-\nu k^2 \Delta t}$ 永远在 0 到 1 之间）
- 高频分量（大 $k$）衰减更快——物理上正确（小尺度扩散更快）

代码只有三行：

```python
def diffusion_step(u, nu, dx, dy, dt):
    k2 = precomputed_k2   # 预计算的 kx² + ky²
    decay = np.exp(-nu * k2 * dt)  # 预计算，循环外只算一次
    u_hat = np.fft.fft2(u)
    return np.real(np.fft.ifft2(decay * u_hat))
```

**有限差分需要解稀疏线性方程组；谱方法只需要一次乘法。** 这就是"把偏微分方程变成矩阵乘法"的含义——不是比喻，是字面意义上的。

---

## 七、总结：三层加速

回到开头的技术路线：

```
第一层：谱方法（数学层面）
  物理空间导数 → 傅里叶空间逐元素乘法
  消除了空间耦合，每个频率分量独立计算

第二层：张量运算（编译层面）
  所有物理算子统一表达为张量运算（FFT + 逐元素乘加）
  XLA 编译器做内核融合、常量折叠、内存优化
  消除了 Python 开销和中间数组分配

第三层：GPU 并行（硬件层面）
  逐元素运算是 GPU 的完美场景
  数千核心同时执行相同的乘法
  FFT 有 cuFFT 硬件加速
```

核心思想可以用一句话概括：

**傅里叶变换把偏微分方程中的空间导数变成逐元素乘法，而逐元素乘法正是 GPU 和现代编译器最擅长优化的运算。谱方法天然地把物理方程映射到了与神经网络相同的执行模型上。**

这不是巧合。神经网络做的是"线性变换 + 非线性激活"的交替，而伪谱方法做的是"逐元素线性乘法 + FFT 桥接的物理空间非线性乘积"的交替。两者的计算结构高度同构——这就是为什么同一套 GPU 加速基础设施（cuBLAS、cuFFT、XLA/TorchInductor）能同时服务深度学习和科学计算。

---

## 附录：关键概念速查

| 物理空间 | 傅里叶空间 | 关系 |
|---------|-----------|------|
| $u(x)$ | $\hat{u}(k)$ | 傅里叶变换对 |
| $\partial u/\partial x$ | $ik \cdot \hat{u}$ | 求导 → 乘法 |
| $\nabla^2 u$ | $-k^2 \cdot \hat{u}$ | 拉普拉斯 → 乘法 |
| $u \cdot v$ | $\hat{u} * \hat{v}$（卷积） | 乘积 → 卷积（所以回物理空间做） |
| $e^{\nu \nabla^2 \Delta t} u$ | $e^{-\nu k^2 \Delta t} \cdot \hat{u}$ | 矩阵指数 → 逐元素指数 |

| 术语 | 含义 |
|------|------|
| 波数 $k$ | 空间频率，单位 $1/\text{m}$。$k$ 越大 = 波长越短 = 变化越快 |
| 去混叠 | 砍掉高 1/3 频率，防止非线性乘积产生的高频分量污染低频 |
| 伪谱方法 | 非线性项在物理空间做，导数在傅里叶空间做，用 FFT 桥接 |
| 内核融合 | 编译器把多个逐元素操作合并成一个内核，减少内存读写 |
| XLA | Accelerated Linear Algebra，Google 的线性代数编译器 |

---

*本文的 benchmark 数据来自一个用 JAX 实现的海洋静力原始方程求解器。谱方法是计算物理中的经典方法，广泛应用于海洋模型、大气模型和湍流模拟中。*
