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

**原则一：多段 α 变换（T_alpha）。**

$$
\boxed{\mathcal{T}_\alpha(x) = \text{sign}(x) \cdot T_{\text{target}}(|x|)}
$$

$T_{\text{target}}$ 通过预设的分段表在对数空间插值。小 $|x|$ 时 $T_{\text{target}}$ 为常数
（地板），大 $|x|$ 时恢复对数行为。地板消灭了小违反的数值盲区——
$g=10^{-10}$ 的微小违反立刻被拉升到 $\mathcal{T} \approx 0.7$，求解器立即可感知。
详见 [§7](#7-多段-α-变换与预设表)。

纯 `log_transform`（$T(x)=\text{sign}(x)\log(1+|x|)$）仍可作为 `transform='log'` 使用。

**原则二：Tunable 连续谱。** 所有约束都是加性的，只有两种模式：

| 模式 | 机制 | β 控制什么 |
|------|------|-----------|
| **Tunable** | $\delta \cdot \sigma(\beta \cdot \mathcal{T}(g)) + \text{inner}$ | β: 0.1→软偏好, 1→标准, 100→硬, 1e7→纯硬 |
| **Soft** | $\mathcal{T}(g) + \text{inner}$ | 无参数, 最简 |

`mode='hard'` 自动映射为 `Tunable + β=1e7`。

## 2. 数学基础

### 2.1 饱和函数 $\sigma_k$

$$
\boxed{\sigma_k(x) = \frac{kx}{\sqrt{1 + (kx)^2}}}
$$

**基本性质：**

1. **奇函数**：$\sigma_k(-x) = -\sigma_k(x)$。
2. **值域**：严格在 $(-1, 1)$ 内，与 $k$ 无关。
3. **$k$ 控制"拐点"**：$\sigma_k(1/k) = 1/\sqrt{2} \approx 0.707$。
4. **嵌套有精确闭式**：$\sigma_1^{(n)}(1) = 1/\sqrt{n+1}$（归纳法可证）。

### 2.2 多段 α 变换 $\mathcal{T}_\alpha$

#### 原始想法

$$
\boxed{\mathcal{T}(x) = \text{sign}(x) \cdot \log(1 + \alpha(|x|) \cdot |x|)}
$$

其中 $\alpha(|x|)$ 不是常数——它随 $|x|$ 的量级变化：

- $|x|$ 微小（如 $10^{-8}$）：$\alpha$ 巨大（如 $10^6$）→ $\log(1+10^6 \cdot 10^{-8}) \approx 0.01$，**放大微小值**
- $|x|$ 中等：$\alpha$ 适度 → 平滑过渡
- $|x|$ 巨大（如 $10^8$）：$\alpha \approx 1/|x|$ → $\log(1+x) \approx \log|x|$，**恢复标准对数**

这比固定 $\alpha$ 的 log 变换强在：小 $g$ 不再被"压扁到零"，而是在进入 $\sigma$ 之前就被 $\alpha$ 放大到可感知的量级。

#### 怎么变成结点表的

直接存 $\alpha(|x|)$ 的函数形式不灵活。我们把 $(|x|, \alpha)$ 转化为 $(g, T)$ 的对应关系：

$$
T = \log(1 + \alpha \cdot g) \quad\Longleftrightarrow\quad \alpha = \frac{e^T - 1}{g}
$$

存 $(g_i, T_i)$ 比存 $(g_i, \alpha_i)$ 更直观——$T=0.7$ 直接告诉你"$\sigma(0.7) \approx 0.57$"，不需要心算 log。

**以 `'standard'` 预设为例，等价于如下的 $\alpha$ 变化：**

| $g$ | $T$ | 反算 $\alpha$ | 直观含义 |
|-----|-----|-------------|---------|
| $10^{-6}$ | 0.7 | $\approx 10^6$ | 微小违反被放大百万倍 |
| $10^{-4}$ | 0.8 | $\approx 1.2\times 10^4$ | |
| $0.01$ | 0.9 | $\approx 1.5\times 10^2$ | |
| $0.1$ | 1.0 | $\approx 17$ | |
| $1$ | 1.5 | $\approx 3.5$ | |
| $10$ | 2.5 | $\approx 1.1$ | α 接近 1，趋近 log |
| $100$ | 4.0 | $\approx 0.5$ | |
| $10^4$ | 7.0 | $\approx 0.1$ | |
| $10^6$ | 10.0 | $\approx 0.02$ | α 很小，≈ log(|x|) |

**核心机制：** $g=10^{-6}$ 时 $\alpha \approx 10^6$，把违反放大百万倍再取 log——求解器立刻感知。$g$ 越大 $\alpha$ 越小，最终退化为标准对数行为。

#### 实现：log-线性插值

结点表存的是 $(\log g_i, T_i)$。对任意输入 $|x|$：

1. 取 $\log|x|$
2. 在 $\log g_i$ 中找到所在区间
3. 线性插值出 $T$
4. $\mathcal{T}_\alpha(x) = \text{sign}(x) \cdot T$

等价于**在 $\log g$ 坐标上对 $T$ 做分段线性插值**，无需显式计算 $\alpha$ 或 $\log$。

#### "地板"是什么

任何 $|x|$ 小于表中第一个 $g_i$ → 落在第一个结点左边 → 插值出 $T=T_1$（常数）。

$g=10^{-10}$ 和 $g=10^{-7}$ 输出**相同**的 $T=0.7$——这就是"地板"。它保证了任何非零违反都能被感知（$\sigma(0.7) \approx 0.57 \gg 0$），消灭了 $\log(1+x) \approx x \approx 0$ 造成的盲区。

#### 三档分辨率标定

**第一个 knot 的 $g$ 值 = 该模式的"分辨率"。** 小于此值的违反全在地板区——求解器几乎无感。

| 模式 | 分辨率 | 地板 T | 语义 |
|------|--------|--------|------|
| **Soft** | $10^{-2}$ | 0.03 | $g<0.01$ 几乎无感，目标和约束平等竞争 |
| **Tunable** | $10^{-4}$ | 0.15 | $g<10^{-4}$ 微感，σ 压缩兜底 |
| **Hard** | $10^{-6}$ | 0.50 | $g<10^{-6}$ 立即感知，任何违反 > 任何满足 |

```
       g=0   1e-6  1e-4  1e-3  0.01   0.1    1     10
Soft:  .044  .050  .050  .050  .050  .084  .237  .478
Tun:   .044  .053  .053  .059  .068  .085  .125  .175
Hard:  .044  .129  .129  .175  .225  .277  .312  .322
```

- **Soft** 在 $g<0.01$ 区间完全平坦——$g=0$ 和 $g=0.005$ 代价相同。$g=0.1$ 才开始抬头。
- **Tunable** 在 $g<10^{-4}$ 平坦，$g=0.01$ 时有明显惩罚（0.068 vs 0.044），但有 σ 压缩。
- **Hard** 在 $g<10^{-6}$ 平坦，$g=10^{-6}$ 就跳到 0.129——任何微小违反代价都远超无违反最大值（0.133 vs 0.129...实际在最坏目标下 0.215 > 0.133，分离保证）。

**调分辨率：改第一个 knot 的 $g$ 值。** 调地板高度：改第一个 $T$ 值。
详见 [§2.3 怎么调整容忍度](#23-三档默认标定与时域累加)。

### 2.3 三档默认标定与时域累加

三种模式对应三套结点表，按**每步违反的容忍度**标定。注意：你写的 $g$ 是整个时域上的**总和** $g_{\text{total}} = \sum_t \max(0, \text{pen}_t)$，
所以时域越长累加越大，代价自动递增。

#### 容忍度速查

| 模式 | 每步 pen=0.01, H=1 | H=10 | H=50 | H=200 | 语义 |
|------|-------------------|------|------|-------|------|
| **Soft** | +0.03（轻触） | +0.08 | +0.14 | +0.30 | 累加敏感，目标和约束平等竞争 |
| **Tunable** | +0.04（轻微） | +0.05 | +0.07 | +0.11 | σ 压缩，边际递减，累积不如 Soft 敏感 |
| **Hard** | +0.23（重罚） | +0.25 | +0.26 | +0.27 | 一步即罚，再多也差不多 |

```
Soft:    g=0→0.044  g=0.01→0.074  g=0.1→0.123  g=1→0.273  g=10→0.478
Tunable: g=0→0.044  g=0.01→0.082  g=0.1→0.096  g=1→0.138  g=10→0.175
Hard:    g=0→0.044  g=0.01→0.271  g=0.1→0.293  g=1→0.316  g=10→0.322
```

**Hard 保证：** 最小违反（$g=10^{-6}$）代价 $0.238$ > 最大无违反代价（$f=10^6$ 时 $0.133$）。任何违反 > 任何满足。  
**Soft 竞争：** $g=0.01$ + $f=0.01$（$0.044$）< $g=0$ + $f=100$（$0.057$）。微小违反 + 好目标胜出。

#### 怎么调整容忍度

**如果不满意默认行为，改结点表。** 三种模式各有一张表：

```python
# Soft 表: g 从 1e-4 到 1e6, T 从 0.02 到 12
# 调 "g=0.01 时多敏感" → 改第三个 T 值 (当前 0.15)
#   → 改为 0.05: 更不敏感, 小违反几乎无感
#   → 改为 0.30: 更敏感, 小违反就明显惩罚

# Tunable 表: 同理, 改对应 g 的 T 值
#   还可以调 tune_preset 的 (β, δ_soft)

# Hard 表: 改地板 (第一个 T 值, 当前 0.6)
#   → 改为 1.0: 小违反代价更大
#   → 改为 0.3: 退化到接近 Tunable
```

**如果觉得累加太敏感**（200 步轻度擦边不应超过 1 步重度碰撞）：
- 改用 Tunable（σ 压缩让边际递减）
- 或在 `g_fn` 里做 `sum(T_alpha(pen_t))` 而非 `T_alpha(sum(pen_t))`
- 或降低 Soft 表在 $g \in [1, 100]$ 区间的 T 值

**如果觉得累加不够敏感**（10 步擦边就应重罚）：
- 改用 Hard
- 或提高 Soft/Tunable 的 T 值

### 2.4 嵌套级联的数学结构

单层 $\sigma$ 将 $(-\infty, \infty)$ 映射到 $(-1, 1)$。多层嵌套时：

$$
\sigma_1^{(n)}(1) = 1/\sqrt{n+1} \quad \text{（精确）}
$$

**证明（归纳法）：** $\sigma_1^{(n+1)}(1) = \sigma_1(1/\sqrt{n+1}) = 1/\sqrt{n+2}$。

这是整套 $\delta$ 选值理论的基石。

---

## 3. 五种约束类型

| # | 类型 | 数学形式 | g_raw 计算 |
|---|------|---------|-----------|
| 0 | 无约束 | $\min f(x)$ | — |
| 1 | **确定性** | $g(x) \le 0$ | $g(x)$ 直接 |
| 2 | **机会约束** | $P(g(x,\xi) \le 0) \ge 1-\alpha$ | $Q_{1-\alpha}(g(x,\xi))$ MC 分位数 |
| 3 | **鲁棒** | $g(x,\xi) \le 0\;\forall\xi\in\Xi$ | $\max_{\xi\in\Xi} g(x,\xi)$ lax.scan |
| 4 | **分布鲁棒** | $\inf_{P\in\mathcal{P}} P(g \le 0) \ge 1-\alpha$ | $\max_{P\in\mathcal{P}} Q_{1-\alpha}$ |

（各类型详细说明和代码模板见 [ConstranUser_README.md](ConstranUser_README.md)）

---

## 4. 优先级嵌套系统

### 4.1 嵌套公式

对于 $M$ 个约束，嵌套从内到外构建。每层都是**加性的**——没有 `jnp.where`：

$$
\boxed{\mathcal{L}_0 = \sigma_{k_{\text{inner}}}\!\big(\mathcal{T}_\alpha^{\text{obj}}(f(x))\big)}
$$

$$
\boxed{\mathcal{L}_i = \begin{cases}
\sigma_1\!\big(\delta_i \cdot \sigma(\beta_i \cdot \mathcal{T}_\alpha^i(g_i)) + \mathcal{L}_{i-1}\big) & \text{Tunable} \\[8pt]
\sigma_1\!\big(\mathcal{T}_\alpha^i(g_i) + \mathcal{L}_{i-1}\big) & \text{Soft}
\end{cases}}
$$

$$
\boxed{\mathcal{L}(x) = \mathcal{L}_M}
$$

其中 $i=1$ 是最内层约束，$i=M$ 是最外层。每层独立选择：
- **变换表** `transform`：$\mathcal{T}_\alpha^i$（tight/standard/sharp/wide/log）
- **Tunable 参数**：$(\beta_i, \delta_i)$，从软到硬的连续谱

### 4.2 Tunable 连续谱：从软到硬

**$\beta$ 是唯一的关键参数。** 所有约束都是 `contrib + inner` 的加性形式：

| β | 行为 | 违反 $g=0.001$ | 违反 $g=100$ | 预设名 |
|---|------|---------------|-------------|--------|
| 0.1 | 极软，大违反才触发 | 几乎无感 | 温和 | `'mild'` |
| 0.5 | 标准软 | 轻微 | 明显 | `'standard'` |
| 1.0 | 适中 | 可感 | 强 | `'firm'` |
| 5.0 | 较硬 | 立刻触发 | 满罚 | `'nearhard'` |
| 100 | 硬 | 即触即满 | 封顶 | — |
| $10^7$ | 纯硬（≈ 旧 Hard） | 和 $g=100$ 几乎一样 | 封顶 | `mode='hard'` |

**β ≥ 100 时过渡宽度 < $10^{-7}$，float32 不可分辨——等价于旧版的 Hard 模式。**
`mode='hard'` 自动映射为 `Tunable + β=1e7`，无需手写 β。

**$\delta$ 控制最大贡献幅度：** 内层内容 σ 后约 $[0, 0.7]$。$\delta=1\sim3$ 可与目标同级竞争或压倒。

**Tunable 预设套餐：**

| 预设 | β | δ | 适用 |
|------|---|---|------|
| `'mild'` | 0.1 | 1.0 | 舒适/效率偏好 |
| `'standard'` | 0.5 | 1.0 | 标准软约束 |
| `'firm'` | 1.0 | 2.0 | 重要偏好 |
| `'strong'` | 2.0 | 2.0 | 较强约束 |
| `'nearhard'` | 5.0 | 3.0 | 近似硬约束 |

### 4.3 位置 = 天然权重

即使 δ 和 β 相同，外层也比内层更有影响力——因为外层贡献不经内层 σ 压缩：

```
同样 g=100 的违反：
  L1 (最外层, 0层σ压缩): 贡献 ≈ 4.6 (原封不动)
  L2 (中层,   1层σ压缩): 贡献 ≈ 0.98
  L3 (最内层, 2层σ压缩): 贡献 ≈ 0.70
```

嵌套本身就提供了分层。Tunable 的 δ 和 β 在这个基础上做微调。

### 4.4 三层示例

```python
cost_fn = build(
    objective_fn,
    [
        Deterministic(static_viol, mode='hard', priority=1,    # → Tunable β=1e7
                      delta=1.5, transform='sharp'),
        Chance(ped_viol, mode='tunable', priority=2,
               tune_preset='firm', transform='standard'),
        Deterministic(comfort, mode='soft', priority=3,
                      transform='wide'),
    ],
    k_inner=0.1, obj_transform='standard',
)
```

没有 `jnp.where`，所有层都是连续可微的。

---

## 5. 通用 M 层框架

### 5.1 每层独立配置四张表

| 配置项 | 控制什么 | 预设 |
|--------|---------|------|
| `transform` | 违反感知基线 | tight/standard/sharp/wide/log |
| `tune_preset` 或 `(beta, delta_soft)` | 软硬程度 + 影响力 | mild/standard/firm/strong/nearhard |
| `mode` | Tunable 或 Soft | 'tunable'（默认）, 'soft', 或 'hard'（→ Tunable β=1e7） |
| `priority` | 嵌套顺序 | 1=最高（最外层） |

### 5.2 设计规则

```
OUTERMOST (最高优先级)
    ↑
    ├── 大 β (≥100) — 硬约束, 安全关键
    ├── 中 β (1~10)  — 重要但可调
    ├── 小 β (0.1~1) — 软偏好, 可妥协
    ├── Soft         — 最简, 无参数
    ↓
INNERMOST (目标函数, k=0.1)
```

---

## 6. 深层嵌套与 $\delta$

$\sigma_1^{(n)}(1) = 1/\sqrt{n+1}$ 的闭式保证了：**越深越容易分离，$\delta$ 可以越小。**

T_alpha 的地板进一步降低了 $\delta$ 需求：

| transform | T(0⁺) | 建议 δ (外层) | 建议 δ (内层) |
|-----------|-------|-------------|-------------|
| sharp | 1.0 | 0.1~0.3 | 0.3~0.5 |
| standard | 0.7 | 0.3~0.5 | 0.5~0.7 |
| tight | 0.3 | 0.6~0.8 | 0.8~1.0 |

**$\delta$ 太大** → 求解器不敢靠近约束边界，解太保守。
**$\delta$ 太小** → 层级分离脆弱。

---

## 7. 完整操作手册

**Step 1** — 列出约束，分配优先级，选择变换表和 Tunable 参数。

**Step 2** — 写 g_raw 计算函数（正值=违反，负值=满足）。

**Step 3** — 选 $k_{\text{inner}}$（默认 0.1）和 `obj_transform`（默认 'standard'）。

**Step 4** — 从内到外构建：

```python
from Constraintdealer.Constran import *

cost_fn = build(
    objective_fn,
    [
        Deterministic(g1, mode='hard', priority=1,
                      delta=1.5, transform='sharp'),
        Deterministic(g2, mode='tunable', priority=2,
                      tune_preset='firm', transform='standard'),
        Deterministic(g3, mode='soft', priority=3,
                      transform='wide'),
    ],
    k_inner=0.1,
)
```

**Step 5** — 验证区分度。

---

## 8. 代码索引

| 约束类型 | 函数 | 文件 |
|---------|------|------|
| 翻译器 (T_alpha) | `build()`, `Deterministic`, `Chance`, `Robust`, `DRO` | `Constran.py` |
| 翻译器 (log_transform) | `build()`, `Deterministic`, ... | `Constran.py` |
| 确定性测试 | `cost_sat_hierarchical*` | `Constraints.py` |
| 机会约束测试 | `cost_sat_hierarchical4-10` | `Constraints.py` |
| 鲁棒测试 | `cost_robust1-2` | `RobustConstraints.py` |
| 生产级 MPC | — | `Hybridsystemtest.py` |
| 用户手册 | — | `ConstranUser_README.md` |

---

## 附录：快速参考卡片

```python
# ─── 核心函数 ───
sigma_k(x, k=1.0)      # 饱和: kx/√(1+(kx)²), 输出 ∈ (-1,1)
T_alpha(x, knots_g, knots_T)  # 多段 α 变换

# ─── 约束声明 ───
Deterministic(g_fn, mode='tunable', priority=1,
              transform='standard', tune_preset='firm')
Deterministic(g_fn, mode='soft', priority=2,
              transform='wide')
Deterministic(g_fn, mode='hard', priority=1,    # → Tunable β=1e7
              delta=1.5, transform='sharp')

# ─── 预设速查 ───
# transform: tight, standard, sharp, wide, log
# tune_preset: mild(0.1,1.0), standard(0.5,1.0), firm(1.0,2.0),
#              strong(2.0,2.0), nearhard(5.0,3.0)
# obj_transform: standard, flat, log

# ─── 构建 ───
cost_fn = build(obj_fn, constraints, k_inner=0.1,
                obj_transform='standard')
```

β —— 控制"多快触发"
$$
\beta \approx \frac{0.58}{\log(1 + g_{\text{accept}})}
$$

$g_{\text{accept}}$ = 你认为"可以容忍"的最大违反量。

可接受违反	β	含义
~10	0.2	宽过渡，大违反才感到
~1	0.8	标准
~0.1	6	小违反即触发
~0.01	60	很锐
~0.001	500+	几乎即触即满
δ —— 控制"最多影响多少"
内层内容（目标 + 内层约束）经 σ 后输出量级约 $[0, 0.7]$。

δ	效果
0.1–0.5	轻偏好，几乎不改变排名
1.0–2.0	与目标同级竞争
3.0–5.0	强偏好，通常压倒目标
>5	近似硬约束
四档套餐
场景	δ	β	g≈0.001	g≈1	g≈10
轻偏好	0.3	0.2	~0	+0.04	+0.13
标准软	1.0	1.0	+0.001	+0.57	+0.92
较强	2.0	5.0	+0.01	+1.92	+1.99
近似硬	3.0	50	+0.15	+3.00	+3.00
Soft 等于 Tunable 取 β→0, δ→∞ 的极限——对数增长永不饱和。选 Tunable 就是给这个增长加了个上限。

总结一下就是：

数学层： 三种模式的单调性都正确，不会破坏排序。

工程层： 选哪个、参数怎么设，取决于你对"违反"的语义定义：

你想说	用
"任何违反都不可接受，不管多小"	Hard, jnp.where
"违反越大越差，没有上限"	Soft, T(g)
"违反有上限——超过某个程度后都一样糟糕"	Tunable, 调 δ 定上限
"小违反可以忍，但要平滑过渡"	Tunable, 调 β 定容忍区宽度
"多点小违反 vs 单点大违反，我要区分"	Soft 或 Tunable β<1
"违反只看最坏的那个点"	用 lax.scan(max) 而不是 sum

---

