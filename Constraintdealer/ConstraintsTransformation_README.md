# 约束转化方法论：从约束优化到黑箱标量代价

## 目录

1. [问题与设计目标](#1-问题与设计目标)
2. [数学基础](#2-数学基础)
3. [五种约束类型](#3-五种约束类型)
4. [优先级嵌套系统](#4-优先级嵌套系统)
5. [硬约束 vs 软约束 vs 可调约束](#5-硬约束-vs-软约束-vs-可调约束)
6. [通用 M 层框架](#6-通用-m-层框架)
7. [对数变换：消灭数值爆炸](#7-对数变换消灭数值爆炸)
8. [深层嵌套与 δ 的标度律](#8-深层嵌套与-δ-的标度律)
9. [完整操作手册](#9-完整操作手册)
10. [代码索引](#10-代码索引)

---

## 1. 问题与设计目标

### 1.1 我们要解决什么问题

考虑最一般的约束优化问题：

$$
\begin{aligned}
\min_{x \in \mathbb{R}^d} &\quad f(x) \\
\text{s.t.} &\quad g_i(x, \xi) \le 0, \quad i = 1,\dots,M
\end{aligned}
$$

其中约束 $g_i$ 可能涉及不确定性 $\xi$，并且约束之间有**字典序优先级**
（某些约束比另一些更重要，必须优先满足）。

MMOG-IGO 求解器只需要一个**标量代价** $\mathcal{L}(x)$ 来对候选解做
`argsort` 排序。我们需要一个转化方案，满足：

1. **保序**：$x$ 越好 → $\mathcal{L}(x)$ 越小
2. **严格优先级**：任何对高层约束的违反，必须比所有低层代价更大
3. **float32 数值稳定**：全动态范围内可区分（~7 位有效数字）
4. **黑箱友好**：不需要知道 $f_{\max}$，$f$ 可达 $10^8$ 甚至为负

### 1.2 两个核心原则

**原则一：对数变换一切。**

$$
\boxed{\mathcal{T}(x) = \text{sign}(x) \cdot \log(1 + |x|)}
$$

每个原始值——目标函数 $f(x)$、约束函数 $g_i(x)$、分位数、最坏情形最大值——
在进入饱和函数之前，都必须经过 $\mathcal{T}$。这彻底消灭了数值爆炸。

**原则二：选择你的优先级编码方式。** 每层独立选择：

| 模式 | 机制 | 行为 |
|------|------|------|
| **硬约束** | `jnp.where(g>0, T(g)+δ, 内层)` | 任何违反 > 任何内层代价 |
| **软约束** | `T(g) + 内层` | 违反与内层代价加性竞争 |
| **可调约束** | `δ·σ(β·T(g)) + 内层` | β 控制软硬程度 |

---

## 2. 数学基础

### 2.1 饱和函数 $\sigma_k$

$$
\boxed{\sigma_k(x) = \frac{kx}{\sqrt{1 + (kx)^2}}}
$$

**基本性质：**

1. **奇函数**：$\sigma_k(-x) = -\sigma_k(x)$。这是处理负值代价的数学基础——
   正负对称，整个变换链保持奇性。

2. **值域**：严格在 $(-1, 1)$ 内，与 $k$ 无关。

3. **$k$ 控制"拐点"**：
   $$
   \sigma_k(1/k) = \frac{1}{\sqrt{2}} \approx 0.707
   $$
   - $|x| \ll 1/k$：$\sigma_k(x) \approx kx$（近似线性 — 区分度好）
   - $|x| \gg 1/k$：$\sigma_k(x) \to \text{sign}(x)$（饱和 — 区分度差）

4. **导数**：
   $$
   \sigma_k'(x) = \frac{k}{(1 + (kx)^2)^{3/2}} > 0
   $$
   单调递增，最大斜率在 $x=0$ 处为 $k$。

**为什么选这个函数？** 对比常见的 sigmoid、tanh、arctan：

| 函数 | 奇函数 | 显式反函数 | 嵌套有闭式 | 计算成本 |
|------|--------|-----------|-----------|---------|
| $\sigma_k$ | ✓ | ✗ | ✓ ($1/\sqrt{n+1}$) | 低 |
| tanh | ✓ | ✓ | ✗ | 中 |
| arctan | ✓ | ✓ | ✗ | 高 |
| sigmoid | ✗ (不对称) | ✓ | ✗ | 中 |

$\sigma_k$ 的核心优势是**嵌套有精确闭式解**（见 §8），这对深层嵌套的
$\delta$ 选择至关重要。

### 2.2 对数变换 $\mathcal{T}$

$$
\boxed{\mathcal{T}(x) = \text{sign}(x) \cdot \log(1 + |x|)}
$$

**基本性质：**

1. **奇函数**：$\mathcal{T}(-x) = -\mathcal{T}(x)$

2. **零点保序**：$\mathcal{T}(0) = 0$，$\mathcal{T}'(0) = 1$。在小值区域，
   $\mathcal{T}$ 近似恒等变换，保持对微小差异的敏感度。

3. **大值压缩**：
   - $x = 10^2 \to \mathcal{T} \approx 4.62$
   - $x = 10^5 \to \mathcal{T} \approx 11.51$
   - $x = 10^8 \to \mathcal{T} \approx 18.42$
   
   8 个数量级被压缩到 $[0, 18]$ 区间。

4. **导数的物理意义**：
   $$
   \mathcal{T}'(x) = \frac{1}{1 + |x|}
   $$
   在 $x=0$ 处导数为 1（不损失小值灵敏度），在 $x=10^8$ 处导数为 $10^{-8}$
   （极端压缩大值）。

**为什么约束也需要 $\mathcal{T}$？** 如果约束函数 $g(x)$ 本身可以达到
$10^6$（例如距离惩罚在长时域上累加），那么：

- 不做 $\mathcal{T}$：$g = 10^6$，$\sigma(10^6) \approx 0.999999$
- 做 $\mathcal{T}$：$\mathcal{T}(10^6) \approx 13.8$，$\sigma(13.8) \approx 0.997$

虽然 $\sigma$ 输出都接近 1，但在对数变换后，$g=10^5$ 和 $g=10^6$ 在
$\sigma$ 输入端的差异是 $11.5 \to 13.8$（差 2.3），而不是 $10^5 \to 10^6$
（在 float32 下 $\sigma$ 输出完全无法区分）。

**逻辑：** $\mathcal{T}$ 必须在**判断违反 / 加 $\delta$ 之前**应用，
这样数值在进入嵌套级联之前已被压缩到可控范围。

### 2.3 嵌套级联的数学结构

单层 $\sigma$ 将 $(-\infty, \infty)$ 映射到 $(-1, 1)$。多层嵌套时：

$$
\begin{aligned}
\sigma_1^{(1)}(\infty) &= 1 \\
\sigma_1^{(2)}(\infty) &= \sigma_1(1) = 1/\sqrt{2} \approx 0.707 \\
\sigma_1^{(3)}(\infty) &= \sigma_1(1/\sqrt{2}) = 1/\sqrt{3} \approx 0.577 \\
&\;\;\vdots \\
\sigma_1^{(n)}(\infty) &= 1/\sqrt{n} \quad \text{（对 } n \ge 1 \text{ 精确成立）}
\end{aligned}
$$

**证明（归纳法）：** 假设 $\sigma_1^{(n)}(1) = 1/\sqrt{n+1}$。
$$
\sigma_1^{(n+1)}(1) = \sigma_1(1/\sqrt{n+1}) = \frac{1/\sqrt{n+1}}{\sqrt{1 + 1/(n+1)}} = \frac{1/\sqrt{n+1}}{\sqrt{(n+2)/(n+1)}} = \frac{1}{\sqrt{n+2}}
$$
证毕。

这个恒等式是整套 $\delta$ 选值理论的基石——它精确给出了任意深度嵌套下
内层输出的最大界。

---

## 3. 五种约束类型

### 总览

| # | 类型 | 数学形式 | g_raw 计算 | 示例 |
|---|------|---------|-----------|------|
| 0 | 无约束 | $\min f(x)$ | — | 纯目标优化 |
| 1 | **确定性** | $g(x) \le 0$ | $g(x)$ 直接 | 物理边界、避障 |
| 2 | **机会约束** | $P(g(x,\xi) \le 0) \ge 1-\alpha$ | $Q_{1-\alpha}(g(x,\xi))$ | 感知噪声下的安全 |
| 3 | **鲁棒** | $g(x,\xi) \le 0\;\forall\xi\in\Xi$ | $\max_{\xi\in\Xi} g(x,\xi)$ | 最坏情形保证 |
| 4 | **分布鲁棒** | $\inf_{P\in\mathcal{P}} P(g \le 0) \ge 1-\alpha$ | $\max_{P\in\mathcal{P}} Q_{1-\alpha}$ | 分布偏移 |

### 3.1 确定性约束

**数学：** $g(x) \le 0$。不涉及任何不确定性。

**g_raw 计算：** 直接使用 $g(x)$。正 = 违反，负 = 满足。

**单边：** `g_raw = g(x) - c`
**双边 band：** `g_raw = max(c_lo - g(x), g(x) - c_hi)`

```python
Deterministic(lambda x, ctx: x[0] + x[1] + 4,  # x1+x2 >= -4
              mode='hard', priority=1)
```

### 3.2 机会约束（概率约束）

**数学：** $P(g(x,\xi) \le 0) \ge 1-\alpha$。$\xi$ 的分布**已知**。

**核心等价关系（分位数法）：**

$$
P(g(x,\xi) \le 0) \ge 1-\alpha \quad\Longleftrightarrow\quad Q_{1-\alpha}\big(g(x,\xi)\big) \le 0
$$

其中 $Q_q$ 是 Monte Carlo 样本的第 $q$ 分位数。

**上界版本**（$g \le c$）：`g_raw = Q_{1-α}(g(x,ξ))`
**下界版本**（$g \ge c$）：`g_raw = -Q_α(g(x,ξ))`
**环形版本**：两边同时检查。

**选择 $M$（MC 样本数）：** $M \ge 10/\alpha$。例如 $\alpha=0.1$ 时 $M \ge 100$。

```python
Chance(lambda x, xi, ctx: jnp.linalg.norm(x + xi) - safe_dist,
       noise_fn=lambda key, shape: jax.random.normal(key, shape),
       alpha=0.1,          # 90% 概率满足
       mode='hard', priority=1)
```

### 3.3 鲁棒约束（最坏情形）

**数学：** $g(x,\xi) \le 0$ 对**所有** $\xi \in \Xi$ 成立。不假设分布，
只假设集合成员。

**核心等价关系：**

$$
g(x,\xi) \le 0 \;\; \forall \xi \in \Xi \quad\Longleftrightarrow\quad
\max_{\xi \in \Xi} \; g(x,\xi) \le 0
$$

**实现：** `lax.scan` 在离散化的不确定性集上取最大值。

**关键优势：** 支持**非凸、不连通**的不确定性集（例如 $\Xi = [-3,-2] \cup [1,2]$），
这是传统 LMI（线性矩阵不等式）方法无法处理的。

```python
Robust(lambda x, xi, ctx: (x[0]+xi)**2 + x[1]**2 - 10,
       uncertainty_set=jnp.concatenate([
           jnp.linspace(-3, -2, 20),
           jnp.linspace( 1,  2, 20),
       ]),
       mode='hard', priority=1)
```

### 3.4 分布鲁棒约束

**数学：** $\inf_{P \in \mathcal{P}} P(g(x,\xi) \le 0) \ge 1-\alpha$。

**两层最坏情形：**
1. 对模糊集 $\mathcal{P}$ 中每个候选分布 $P_k$，计算 $(1-\alpha)$-分位数
2. 取所有候选分布中的**最大**分位数

$$
g_{\text{raw}} = \max_{P_k \in \mathcal{P}} \; Q_{1-\alpha}^{P_k}\big(g(x,\xi)\big)
$$

**常见模糊集：**

| 类型 | $\mathcal{P}$ | 描述 |
|------|--------------|------|
| 矩约束 | $\{P : |\mathbb{E}[\xi]-\hat\mu| \le \varepsilon\}$ | 均值/方差有界 |
| Wasserstein 球 | $\{P : W_p(P, \hat P_n) \le \varepsilon\}$ | 分布距离有界 |
| $\phi$-散度 | $\{P : D_\phi(P \| \hat P_n) \le \varepsilon\}$ | KL/χ² 球 |
| 离散混合 | $\{P = \sum w_k \delta_{\xi_k}\}$ | 几组可能分布的组合 |

**位置：** 介于机会约束（已知分布）和鲁棒约束（无分布假设）之间——
比机会约束保守，比鲁棒约束精细。

---

## 4. 优先级嵌套系统

### 4.1 嵌套公式

对于 $M$ 个约束，优先级从高到低排列（1 = 最高），嵌套从内到外构建：

$$
\boxed{\mathcal{L}_0 = \sigma_{k_{\text{inner}}}\!\big(\mathcal{T}(f(x))\big)}
$$

$$
\boxed{\mathcal{L}_i = \begin{cases}
\sigma_1\!\big(\texttt{if } g_i > 0 \text{ then } \mathcal{T}(g_i) + \delta_i \text{ else } \mathcal{L}_{i-1}\big) & \text{硬约束} \\[8pt]
\sigma_1\!\big(\mathcal{T}(g_i) + \mathcal{L}_{i-1}\big) & \text{软约束} \\[8pt]
\sigma_1\!\big(\delta_i^{\text{soft}} \cdot \sigma(\beta_i \cdot \mathcal{T}(g_i)) + \mathcal{L}_{i-1}\big) & \text{可调约束}
\end{cases}}
$$

$$
\boxed{\mathcal{L}(x) = \mathcal{L}_M}
$$

其中 $i=1$ 是最内层约束（最低优先级），$i=M$ 是最外层（最高优先级）。

### 4.2 $\delta$ 选值理论

对于硬约束层，$\delta_i$ 必须保证：违反该层的最小代价 > 所有内层内容的最大可能输出。

**情况一：** 该硬约束层内部有 $d$ 层 $\sigma$ 包裹（全部满足时）。
内层最大输出 $= \sigma_1^{(d+1)}(1) = 1/\sqrt{d+2}$。

需要 $\sigma_1(\delta_i) > 1/\sqrt{d+2}$，即 $\delta_i > 1/\sqrt{d+1}$。

| 内层 $\sigma$ 数量 $d$ | 内层最大输出 | $\delta$ 下界 | 推荐 $\delta$ |
|----------------------|------------|-------------|-------------|
| 1 (紧邻目标) | $1/\sqrt{3} \approx 0.577$ | $1/\sqrt{2} \approx 0.707$ | 3.0 |
| 2 | $1/\sqrt{4} = 0.5$ | $1/\sqrt{3} \approx 0.577$ | 3.0 |
| 3 | $1/\sqrt{5} \approx 0.447$ | $1/\sqrt{4} = 0.5$ | 1.5 或 3.0 |
| $\ge 4$ | $< 0.408$ | $< 0.5$ | 1.5（外层）或 3.0（内层） |

**标准值（通用，不需调节）：**
- 最外层硬约束：$\delta = 1.5$（$\sigma_1(1.5) \approx 0.832$）
- 内层硬约束：$\delta = 3.0$（$\sigma_1(3.0) \approx 0.949$）

**情况二：** 该硬约束层内部有软约束层。软约束的 $\mathcal{T}(g_{\max})$ 可能达到
$\sim 18$（若 $g_{\max} \sim 10^8$），但经过一层 $\sigma$ 后压缩到
$\sigma_1(18) \approx 0.999$。再经外层 $\sigma_1$ 后输出约 $0.707$。
所以只要软约束在硬约束**内部**，$\delta=1.5$ 仍然安全。

**结论：只要硬约束在外面、软约束在里面，$\delta = 1.5/3.0$ 是通用常数。**

### 4.3 三层示例输出范围

以混合系统 MPC 为例：

```
L1 (硬, 静态障碍):    [0.832, 1.000)    ← 任何碰撞 > 任何无碰撞
L2 (硬, 行人安全):    [0.688, 0.707)    ← 任何行人违反 > 任何纯跟踪
L3 (目标函数):        [0.000, 0.577)    ← 无违规区域
```

边距：L1–L2 ≈ 0.125，L2–L3 ≈ 0.111。在 float32 精度下（$\varepsilon \approx 6\times 10^{-8}$），
分别对应 ~200 万和 ~180 万个可区分的值——远超每轮排序所需的 ~80 个样本。

---

## 5. 硬约束 vs 软约束 vs 可调约束

### 5.1 三种模式对比

| 模式 | 机制 | 违反最小代价 | 可否妥协 | 典型场景 |
|------|------|------------|---------|---------|
| **Hard** | `jnp.where(g>0, T(g)+δ, inner)` | σ₁(δ) | **否** | 碰撞、物理极限、法规 |
| **Tunable** | `δ_soft·σ(β·T(g)) + inner` | 连续渐变 | β 参数控制 | 需要调软硬的约束 |
| **Soft** | `T(g) + inner` | 无下限 | **是** | 舒适、效率、偏好 |

### 5.2 硬约束 (Hard)

```python
sigma_k(
    jnp.where(g_raw > 0,          # 任何违反 → 走惩罚分支
              log_transform(g_raw) + delta,   # T(g) + δ
              inner)                         # 满足 → 走内层
)
```

**行为：** 输出范围**不相交**。任何违反（哪怕 $10^{-6}$）的代价 > 任何内层最大值。
求解器必须严格按优先级满足约束。

### 5.3 软约束 (Soft)

```python
sigma_k(
    log_transform(g_raw) + inner   # 始终加性，无分支
)
```

**行为：** 约束和目标**始终加性竞争**。小违反 + 好目标可能赢过无违反 + 烂目标。

**数值演示：**
```
Tiny viol (g=0.001) + good tracking (f=1):     cost ≈ 0.070
No viol (g=0) + terrible tracking (f=1000):    cost ≈ 0.494
→ 小违反获胜！求解器可以"挤过去"。
```

**两种变体：**

| 变体 | 公式 | 效果 |
|------|------|------|
| 完整版 | `T(g) + inner` | 违反惩罚，满足**奖励** (T(g)<0 降低成本) |
| 仅惩罚 | `max(0, T(g)) + inner` | 只惩罚违反，不奖励满足 |

### 5.4 可调约束 (Tunable)

$$
\boxed{\text{contribution} = \delta_{\text{soft}} \cdot \sigma\!\big(\beta \cdot \mathcal{T}(g)\big)}
$$

**$\beta$ 控制过渡的锐度，$\delta_{\text{soft}}$ 控制最大贡献幅度。**

| $\beta$ | 半惩罚点 $g \approx$ | 行为 |
|---------|---------------------|------|
| 0.1 | ~320 | 非常软，缓慢渐变 |
| 1.0 | ~0.78 | 适中 |
| 10 | ~0.06 | 较硬，微小违反触发大惩罚 |
| 100 | ~0.006 | 接近硬约束 |

**$\beta$ 选值公式：**
$$
\beta \approx \frac{0.58}{\log(1 + g_{\text{accept}})}
$$
其中 $g_{\text{accept}}$ 是你认为"可接受"的最大违反量。

**重要：** 即使 $\beta \to \infty$，可调约束也**不会**产生像硬约束那样的
不相交输出区间。$\sigma(\delta + \text{inner})$ 中，一个好的内层值仍然
可以补偿惩罚。真正的严格优先级需要 `jnp.where` 分支。

---

## 6. 通用 M 层框架

### 6.1 每层独立选择类型 + 模式

任意 $M$ 个约束，第 $i$ 层声明：

- **类型**：`Deterministic` / `Chance` / `Robust` / `DRO`
- **模式**：`'hard'` / `'soft'` / `'tunable'`
- **优先级**：整数，1 = 最高（最外层）

```python
constraints = [
    Deterministic(g1, mode='hard',   priority=1, delta=1.5),    # 最高优先级
    Chance(g2, ...,  mode='tunable', priority=2, beta=5.0),     # 中等
    Robust(g3, ...,   mode='soft',   priority=3),               # 较低
    # ... 任意 M 层 ...
]
cost_fn = build(my_obj, constraints)
```

### 6.2 设计规则

```
OUTERMOST (最高优先级，最重要)
    ↑
    ├── HARD  层放这里 (安全关键)
    │      δ = 1.5 (最外层) / 3.0 (内层)
    │
    ├── TUNABLE 层 (中等重要，可调软硬)
    │      δ_soft + β 参数
    │
    ├── SOFT 层 (偏好，可妥协)
    │      无 δ，加性竞争
    ↓
INNERMOST (目标函数，k=0.1)
```

**黄金规则：**

1. **硬在外，软在内。** 硬约束包裹软约束，确保安全永远优先。
2. **同类型可合并。** 两个同优先级的软约束可加在同一层：
   $\mathcal{T}(g_a) + \mathcal{T}(g_b)$。
3. **$\delta$ 值是通用的。** 对数变换后不需按问题调整。

### 6.3 四层混合示例

```python
constraints = [
    # L1 (最外层, 最高优先级): 静态障碍 — 硬
    Deterministic(lambda x, ctx: compute_static_violation(x),
                  mode='hard', priority=1, delta=1.5),
    # L2: 舒适区域 — 软 (可妥协)
    Deterministic(lambda x, ctx: compute_comfort_deviation(x),
                  mode='soft', priority=2),
    # L3: 行人安全 — 硬 (安全关键)
    Chance(lambda x, xi, ctx: compute_pedestrian_violation(x, xi),
           noise_fn=pedestrian_noise_fn, alpha=0.1,
           mode='hard', priority=3, delta=3.0),
]
cost_fn = build(lambda x, ctx: compute_objective(x), constraints)
```

**输出范围：**

| 场景 | 范围 |
|------|------|
| L1 违反 (撞墙) | $[0.832, 1.0)$ |
| L3 违反 (太近行人) | $[0.688, 0.707)$ |
| L2 活跃, L1/L3 满足 | L2 与目标加性竞争 |
| 全部满足, $f \ge 0$ | $[0, 0.577)$ |

---

## 7. 对数变换：消灭数值爆炸

### 7.1 不做变换会怎样

假设 $f$ 可达 $10^8$，直接进入三重 $\sigma$：

$$
\sigma_1(\sigma_1(\sigma_1(10^8))) = 0.577350269\ldots
$$
$$
\sigma_1(\sigma_1(\sigma_1(10^7))) = 0.577350268\ldots
$$

差异：$\sim 10^{-9}$。在 float32 下（$\varepsilon \approx 6\times 10^{-8}$），
**零个可区分的值** → `argsort` 变成随机 → 求解器退化 → NaN。

### 7.2 做了变换后

| $f$ | $\mathcal{T}(f)$ | 三重 $\sigma$ 后 |
|-----|-----------------|-----------------|
| 0 | 0 | 0 |
| 1 | 0.693 | 0.069 |
| 100 | 4.615 | 0.360 |
| $10^4$ | 9.210 | 0.489 |
| $10^6$ | 13.82 | 0.533 |
| $10^8$ | 18.42 | 0.551 |

每个相邻量级之间都有 $\sim 10^{-3}$ 到 $\sim 10^{-2}$ 的差异，
对应的 float32 可区分数值 > $10^4$。

### 7.3 可区分值统计（float32, $k_{\text{inner}}=0.1$）

| $f$ 区间 | 输出宽度 | 可区分数值 |
|----------|---------|----------|
| $[10^{-2}, 1]$ | 0.068 | ~1,130,000 |
| $[1, 10^3]$ | 0.374 | ~6,240,000 |
| $[10^3, 10^5]$ | 0.073 | ~1,220,000 |
| $[10^5, 10^7]$ | 0.027 | ~457,000 |
| $[10^7, 10^8]$ | 0.007 | ~123,000 |

即使在最苛刻的 $[10^7, 10^8]$ 区间，仍有 ~123,000 个可区分的值——
超过求解器每轮排序需求的 1000 倍以上。

### 7.4 $k_{\text{inner}}$ 的选择

最内层 $\sigma$ 离原始 $f$ 最近，值域最宽。拐点在输入 $= 1/k$ 处。

| $k$ | 拐点输入 | 拐点 $f$ | 推荐 $f_{\max}$ |
|-----|---------|----------|---------------|
| 1.0 | 1 | ~1.7 | $\sim 10^2$ |
| 0.5 | 2 | ~6.4 | $\sim 10^3$ |
| 0.2 | 5 | ~150 | $\sim 10^6$ |
| **0.1** | **10** | **~2.2×10⁴** | **$\sim 10^8$ (默认)** |
| 0.05 | 20 | ~4.8×10⁸ | $\sim 10^{12}$ |

**选值口诀：** 选 $k$ 使得 $\log(1 + f_{\max}) \lesssim 2/k$。

外层约束的 $k=1$ 通常足够，因为对数变换后违反值的量级适中
（$\log(1+v_{\max}) \sim 3$–$5$）。

---

## 8. 深层嵌套与 $\delta$ 的标度律

### 8.1 闭式恒等式

$$
\boxed{\sigma_1^{(n)}(1) = \frac{1}{\sqrt{n+1}} \quad \text{（精确）}}
$$

证明见 §2.3。这意味着内层输出随深度**缩小**：

| $n$ | $\sigma_1^{(n)}(1)$ |
|-----|---------------------|
| 1 | 0.707 |
| 2 | 0.577 |
| 5 | 0.408 |
| 10 | 0.302 |
| 50 | 0.140 |
| 100 | 0.100 |

### 8.2 $\delta$ 随深度变化

外层硬约束需要的 $\delta$：
$$
\sigma_1(\delta) > \sigma_1^{(n+1)}(1) \;\Longrightarrow\; \delta > 1/\sqrt{n}
$$

**越深越容易分离，$\delta$ 可以越小。** $\delta=1.5$ 和 $\delta=3.0$ 是
保守 overshoot，对任意深度都安全。

### 8.3 真正的危险：软约束直接贴在硬约束里面

如果软约束层与硬约束层之间**没有 $\sigma$ 分隔**（0 层 $\sigma$），
则 $\mathcal{T}(g_{\max}) \approx 18.4$ 直接进入硬约束的比较，
需要 $\delta \sim 19$ 才能维持优先级——不现实。

**只要有一层 $\sigma$ 隔开：** $\sigma_1(18.4) \approx 0.999$，
问题消失。这再次强化了设计规则：**每个约束层都被自己的 $\sigma$ 包裹，
硬约束放在软约束外面。**

---

## 9. 完整操作手册

### Step 1 — 列出所有约束，分配优先级，选择模式

| 约束 | 类型 | 优先级 | 模式 |
|------|------|--------|------|
| 避障 | 确定性 | 1 (最高) | Hard |
| 行人安全 | 机会约束 | 2 | Tunable |
| 舒适 | 确定性 | 3 | Soft |

**规则：** 硬在外，软在内。

### Step 2 — 为每个约束写 g_raw 计算函数

按 §3 中的模板。每个返回**原始值**（正 = 违反，负 = 满足）。
**不要**做 `max(0, ...)` 后处理——让 $\mathcal{T}$ 处理符号。

### Step 3 — 选 $k_{\text{inner}}$

$f$ 可到 $10^8$ → $k_{\text{inner}} = 0.1$（默认）。
$f$ 已知有界且小 → $k_{\text{inner}} = 1.0$。

### Step 4 — 从内到外构建嵌套

```python
from Constraintdealer.Constran import *

constraints = [
    Deterministic(g1, mode='hard',   priority=1, delta=1.5),
    Chance(g2, ...,  mode='tunable', priority=2, delta_soft=2.0, beta=5.0),
    Deterministic(g3, mode='soft',   priority=3),
]
cost_fn = build(my_obj, constraints)
```

### Step 5 — 赋值 $\delta$（仅硬约束层）

- 最外层硬约束：$\delta = 1.5$
- 内层硬约束：$\delta = 3.0$
- 或使用 `autodelta()` 自动赋值

软约束和可调约束不需要 $\delta$。

### Step 6 — 验证

```python
result = quick_check(cost_fn, [
    x_feasible, x_soft_viol, x_hard_viol
])
print(result['ok'])  # True = 健康
```

---

## 10. 代码索引

| 约束类型 | 函数 | 文件 |
|---------|------|------|
| 无约束 | `cost_cos_cos`, `cost_quadratic1` | `Constraints.py` |
| 确定性 (半平面) | `cost_quadratic_half_plane` | `Constraints.py` |
| 确定性 (band) | `cost_quadratic_linear_constraint` | `Constraints.py` |
| 确定性 (环形) | `cost_circle_constraint` | `Constraints.py` |
| 2层确定性嵌套 | `cost_sat_hierarchical` | `Constraints.py` |
| 3层确定性嵌套 | `cost_sat_hierarchical3` | `Constraints.py` |
| 机会约束 (高斯) | `cost_sat_hierarchical4` | `Constraints.py` |
| 机会约束 (离散) | `cost_sat_hierarchical5` | `Constraints.py` |
| 机会约束 (双峰环形) | `cost_sat_hierarchical6` | `Constraints.py` |
| 机会约束 (冲突嵌套) | `cost_sat_hierarchical7-8` | `Constraints.py` |
| 机会约束 (重尾风险) | `cost_sat_hierarchical9-10` | `Constraints.py` |
| 鲁棒 (非凸集) | `cost_robust1` | `RobustConstraints.py` |
| 鲁棒 (2层嵌套) | `cost_robust2` | `RobustConstraints.py` |
| 生产级三级 MPC | — | `Hybridsystemtest.py`, `Hybrid_test_README.md` |
| 翻译器 API | `build()`, `Deterministic`, `Chance`, `Robust`, `DRO` | `Constran.py` |

---

## 附录：快速参考卡片

```python
# ─── 核心函数 ───
@jax.jit
def sigma_k(x, k=1.0):
    kx = k * x
    return kx / jnp.sqrt(1.0 + kx**2)

@jax.jit
def log_transform(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

# ─── 通用常数 ───
K_INNER = 0.1       # 目标层 σ 的拐点
DELTA_OUTER = 1.5   # 最外层硬约束 δ
DELTA_INNER = 3.0   # 内层硬约束 δ

# ─── 常用 Violation 模式 ───
# 确定性:                g_raw = g(x)
# 确定性 band:           g_raw = max(c_lo-g(x), g(x)-c_hi)
# 机会约束上界:           g_raw = quantile(samples, 1-α)
# 机会约束下界:           g_raw = -quantile(samples, α)
# 鲁棒:                  g_raw = lax.scan(max, -inf, xi_grid)
# 分布鲁棒:              g_raw = max over P of quantile under P

# ─── 嵌套模板 ───
cost_fn = build(
    objective_fn,
    [
        Deterministic(g1, mode='hard',   priority=1, delta=1.5),
        Chance(g2, ...,  mode='tunable', priority=2, delta_soft=2.0, beta=5.0),
        Robust(g3, ...,  mode='soft',    priority=3),
    ],
    k_inner=0.1,
)
```
