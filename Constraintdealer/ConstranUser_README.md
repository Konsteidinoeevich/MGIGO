# Constran 用户手册

**Constran** 是约束转化引擎——你写精确罚函数 `max(0, g(x))`，它自动组装成求解器能用的代价函数。

---

## 目录

1. [三步上手](#1-三步上手)
2. [四种约束类型](#2-四种约束类型)
3. [两种模式 + Tunable 连续谱](#3-两种模式--tunable-连续谱)
4. [Tunable 参数怎么选](#4-tunable-参数怎么选)
5. [容忍度与时域累加](#5-容忍度与时域累加)
6. [变换表与预设](#6-变换表与预设)
7. [语义速查表](#7-语义速查表)
8. [MPC 用法](#8-mpc-用法)
9. [常见问题](#9-常见问题)

---

## 1. 三步上手

```python
from Constraintdealer.Constran import *

# ① 写目标函数和约束 (精确罚函数形式: max(0, g(x)))
def my_obj(x, ctx):
    return jnp.sum((x[:2] - ctx['target'])**2) + 0.1 * jnp.sum(x[2:]**2)

def obstacle_violation(x, ctx):
    penetration = safe_dist - compute_min_dist(x, ctx)
    return jnp.sum(jnp.where(penetration > 0, penetration, 0.0))

# ② 声明约束
constraints = [
    Deterministic(obstacle_violation, mode='hard', priority=1),
    Deterministic(lambda x, ctx: jnp.sum(x[2:]**2) - efficiency_limit,
                  mode='tunable', priority=2, tune_preset='standard'),
]

# ③ 构建 → 给求解器
cost_fn = build(my_obj, constraints)
result = mmog_igo_optimizer_mpc(..., fitness_fn_total=cost_fn, context=ctx)
```

**你只需要写 `max(0, g(x))` 形式的约束函数。** 正值 = 违反，零 = 满足。Constran 自动做 T_alpha 变换、σ 嵌套、优先级组装。

---

## 2. 四种约束类型 + 内置聚合

| 类型 | 你写 | Constran 自动计算 g_raw |
|------|------|----------------------|
| `Deterministic` | `g(x)`（标量或向量） | `aggregate` 聚合 → 标量 |
| `Chance` | `g(x, ξ)` + 噪声分布 | 聚合 → $Q_{1-\alpha}$ |
| `Robust` | `g(x, ξ)` + 不确定性集 Ξ | 聚合 → $\max_{\Xi}$ |
| `DRO` | `g(x, ξ)` + 模糊集 | 聚合 → $\max_P Q_{1-\alpha}$ |

### 内置聚合 `aggregate`

**轨迹上每个采样点的违反构成一个向量。** 用 `aggregate` 指定聚合方式，
无需在 `g_fn` 里手写 `sum`/`mean`：

```python
# g_fn 直接返回每步的穿透向量, Constran 帮你聚合
Deterministic(
    lambda x, ctx: jnp.maximum(0.0, safe_dist - min_dists(x, ctx)),  # (H,) 向量
    aggregate='sum',    # 总穿透量 (默认)
    mode='soft', priority=1,
)

Deterministic(
    lambda x, ctx: jnp.maximum(0.0, safe_dist - min_dists(x, ctx)),
    aggregate='mean',   # 平均穿透, 与时域长度无关
    mode='soft', priority=1,
)
```

| aggregate | 语义 | 适用 |
|-----------|------|------|
| `'sum'`（默认） | 总违反量 | "全程擦边比单点碰撞差" |
| `'mean'` | 平均每步违反 | "只看步均, 与时域长度无关" |
| `'max'` | 最危险的一步 | "不允许任何一步超出极限" |
| `'count'` | 违反步数 | "广度比深度重要" |
| `'q90'`, `'q95'`, `'q99'` | 分位数 | "允许 10%/5%/1% 的采样点违规" |

### 聚合方式怎么选

**先问自己：这个约束是在管"总体"还是管"最坏"？**

| 约束类型 | 推荐聚合 | 原因 |
|---------|---------|------|
| **避障/碰撞** | `max` 或 `q95` | 安全性由最危险的点决定。`max`=一个点都不行，`q95`=允许 5% 点擦边 |
| **能耗/燃料** | `sum` | 总消耗量，累加才有意义 |
| **跟踪误差** | `mean` | 平均偏离，与时域长度解耦 |
| **曲率/平滑** | `mean` 或 `q90` | 整体舒适度，允许偶尔急转 |
| **速度/加速度限制** | `max` | 任何一步超限就是违规 |
| **终端约束** | 不加聚合（单点） | 只看最后一步 |

**分位数 `q95` 的空间含义**（不是概率！是对轨迹采样点排序）：

```python
# B-spline 200 采样点, q95 表示:
#   排序后取第 190 个（95%）, 允许 10 个最差的点被忽略
#   适合: 拐角处偶尔擦边可以忍, 但 95% 的轨迹必须是安全的

Deterministic(
    lambda x, ctx: jnp.maximum(0.0, safe_dist - min_dists(x, ctx)),
    aggregate='q95',   # 200 点中允许 10 点擦边
    mode='hard', priority=1,
)
```

**典型 B-spline 配置：**

```python
constraints = [
    # 碰撞: max — 最危险点决定安全
    Deterministic(collision_pen, aggregate='max', mode='hard', priority=1),
    # 曲率: mean — 整体平滑度
    Deterministic(curvature_pen, aggregate='mean', mode='soft', priority=2),
    # 速度: max — 不能有任何一步超速
    Deterministic(speed_pen, aggregate='max', mode='hard', priority=3),
]
```

**对比: sum vs max vs mean vs q95（同一个 B-spline 200 点轨迹）**

```
每点穿透: 180点擦0.01 + 10点碰0.5 + 10点撞2.0

sum=0.61:   全过程总穿透 → 最敏感，擦边也被放大
mean=0.11:  平均每步 → 与轨迹长度无关
max=0.33:   只看最危险点 → 2.0m 那一步决定一切
q95=0.17:   忽略最差的 10 个点 → 介于 mean 和 max 之间
count=0.76: 只看违规点数 → 200 点中 20 点违规
```

---

## 3. 两种模式 + Tunable 连续谱

没有 `jnp.where`。所有约束都是加性的。只有两种模式：

```python
# Tunable: δ·σ(β·T(g)) + inner  — β 控制软硬
# Soft:    T(g) + inner           — 最简, 无参数
```

| mode | β | 行为 | 何时用 |
|------|---|------|--------|
| `'soft'` | — | T(g) 直接加，无参数 | 最简场景 |
| `'tunable'` + `tune_preset='mild'` | 0.1 | 极软，大违反才触发 | 舒适/效率偏好 |
| `'tunable'` + `tune_preset='standard'` | 0.3 | 标准软 | 默认软约束 |
| `'tunable'` + `tune_preset='firm'` | 0.5 | 适中 | 重要偏好 |
| `'tunable'` + `tune_preset='strong'` | 1.0 | 较硬 | 较强约束 |
| `'tunable'` + `tune_preset='nearhard'` | 1.0 | 硬，δ 更大 | 近似硬 |
| `'hard'` (自动→ tunable β=1) | 1.0 | **硬约束** | 安全、物理极限 |

**`mode='hard'` 自动映射为 `tunable + β=1.0`。** 旧代码无需改动。

### 关键区别（g≥0 精确罚函数）

```
g=0.001 (微小违反):  β=0.1 → 几乎无感    β=1(hard) → 立刻感知
g=100   (严重违反):  β=0.1 → 温和惩罚    β=1(hard) → 强惩罚
```

`β=1` 在 $g \to 0^+$ 处提供了层级分离，同时保留违反区区分度。$β>1$ 会导致 σ 饱和、丢失信息。

---

## 4. Tunable 参数怎么选

### δ_soft — "最多扣几分"

内层内容经 σ 后输出约 $[0, 0.7]$。以此为参考：

| δ_soft | 效果 |
|--------|------|
| 0.3~0.5 | 轻偏好，用于硬约束（甜点） |
| 1.0~1.5 | 与目标同级竞争 |
| 2.0~3.0 | 强偏好，通常压倒目标 |

### β — "多快触发"（仅自定义时需设）

通常直接用 `tune_preset` 就够了。手动设 β 的场景：

| 可接受违反 | β | 效果 |
|-----------|-----|------|
| ~10 | 0.1 | 非常软 |
| ~1 | 0.5 | 标准过渡 |
| ~0.1 | 1.0 | 较锐（硬约束甜点） |
| ~0.01 | 5.0+ | 很锐 |

### 套餐速查

```python
# 轻微偏好
Deterministic(viol, mode='tunable', priority=3, tune_preset='mild')

# 标准软约束
Deterministic(viol, mode='tunable', priority=2, tune_preset='standard')

# 重要但可调
Deterministic(viol, mode='tunable', priority=2, tune_preset='firm')

# 近似硬
Deterministic(viol, mode='tunable', priority=1, tune_preset='nearhard')

# 硬约束 (最简写法)
Deterministic(viol, mode='hard', priority=1)
```

---

## 5. 容忍度与时域累加

**关键：你写的 `g_fn` 返回的是整个时域上的总和。** $g_{\text{total}} = \sum_t \max(0, \text{pen}_t)$。
时域越长累加越大，代价自动递增——不需要额外参数。

### 三档默认的容忍度

| 每步穿透 | H=1 (Soft) | H=10 | H=50 | H=200 | (Tunable) | (Hard) |
|---------|-----------|------|------|-------|-----------|--------|
| 0.01 | **+0.03** 轻触 | +0.08 | +0.14 | +0.30 | +0.04→+0.11 | **+0.23** 重罚 |
| 0.1 | +0.08 | +0.11 | +0.17 | +0.38 | +0.05→+0.12 | +0.25 |
| 1.0 | +0.23 | — | — | — | +0.09→+0.13 | +0.27 |

**Soft:** 累加敏感——200 步擦边代价远超 1 步。目标和约束平等竞争。  
**Tunable:** σ 压缩——边际递减，累积不如 Soft 敏感。  
**Hard:** 一步即重罚——$g=10^{-6}$ 就 $> 0.24$，超过任何无违反状态。保证"任何违反 > 任何满足"。

### 如果默认不满足需求

**调容忍度——改结点表的 $T$ 值：**

```python
from Constraintdealer.Constran import Deterministic, TRANSFORM_SOFT
import numpy as np

# 自定义: 让 Soft 对 g<0.1 更不敏感
my_soft = (
    np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e4, 1e6]),
    np.array([0.01, 0.02, 0.05, 0.15, 0.5, 1.5, 4.0, 8.0, 12.0]),
    #        ↑ 改这些小 g 的 T 值 → 调小 = 更不敏感
)

Deterministic(viol, mode='soft', priority=1,
              _transform_table=my_soft)
```

**调累加速度——改聚合策略：**

```python
# 方案 A (默认): 先 sum 再 T — 总穿透量
def g_fn(x, ctx):
    pens = compute_penetrations(x)  # (H,)
    return jnp.sum(jnp.maximum(0.0, pens))

# 方案 B: 先 T 再 sum — 每步独立压缩, 大穿透打折
def g_fn(x, ctx):
    pens = compute_penetrations(x)
    return jnp.sum(T_alpha(jnp.maximum(0.0, pens)))
    # 200×0.01 ≈ 200×0.02 = 4.0 走另一条累加路径
```

---

## 6. 变换表与预设

T_alpha 把原始 $g$ 映射到 $[0, 12]$ 范围。每层可选择变换风格：

```
transform='standard'  — 默认, 地板 T(0⁺)=0.7
transform='sharp'     — 地板 T(0⁺)=1.0, 小违反立刻重罚
transform='tight'     — 地板 T(0⁺)=0.3, 近 log 但无盲区
transform='wide'      — 地板 T(0⁺)=0.3, 宽线性区
transform='log'       — 原版 log_transform, 无地板
```

目标函数也有自己的变换：
```python
cost_fn = build(my_obj, constraints, obj_transform='standard')  # 默认
cost_fn = build(my_obj, constraints, obj_transform='flat')      # 更平, 适合超大范围
cost_fn = build(my_obj, constraints, obj_transform='log')       # 原版
```

**约束层 σ 默认 $k=0.2$**（拐点在 $g \approx 150$），12 个数量级全可区分。目标层 $k_{\text{inner}}=0.1$。

---

## 6. 语义速查表

### 完整语义 → 方案映射

```
"任何碰撞都不行"
  → Deterministic(viol, mode='hard', priority=1, transform='sharp')

"碰撞越深越差，但 10m 和 100m 差不多"
  → Deterministic(viol, mode='tunable', priority=1, tune_preset='firm')

"小擦边无所谓，别大撞就行"
  → Deterministic(viol, mode='tunable', priority=2, tune_preset='standard')

"只是偏好，好跟踪可补偿偶尔擦边"
  → Deterministic(viol, mode='soft', priority=3)

"安全就是安全，微小违反也要感知"
  → Deterministic(viol, mode='hard', priority=1, transform='sharp')

"越远离约束越好（奖励深度满足）"
  → 使用带符号的 g(x) 而非 max(0,g), mode='soft'
```

### 先 T 再 Sum（每点独立压缩再聚合）

当前 Constran 对 g_fn 返回的标量做 T——即**先 sum 再 T**。
如果想**先 T 再 sum**，直接在 g_fn 里做：

```python
def g_fn(x, ctx):
    pens = compute_penetrations(x)          # (200,) 向量
    return jnp.sum(log_transform(pens))     # 先 T 再 sum
    # Constran 会再 T 一次 → "双重 T", 无害
```

双重 T 不破坏单调性。详见 [ConstraintsTransformation_README.md](ConstraintsTransformation_README.md)。

---

## 7. MPC 用法

### 关键：build 一次，ctx 传动态信息

```python
# ✓ 正确
cost_fn = build(my_obj, constraints)    # ← 只调一次

for step in range(T_mpc):
    ctx = {'target': targets[step], 'obs_pos': obs[step], ...}
    result = solver(..., fitness_fn_total=cost_fn, context=ctx)

# ✗ 错误 — 每次都 rebuild → 每次都 JIT 重编译
for step in range(T_mpc):
    cost_fn = build(my_obj, constraints)  # 不要！
    result = solver(...)
```

### 多智能体

```python
agent_fns = build_multi_agent({
    0: (obj_agent0, [Deterministic(viol0, mode='hard', priority=1)]),
    1: (obj_agent1, []),
})
result = mmog_igo_rne_blocks_solver(..., fitness_fn_j=agent_fns[0], ...)
```

---

## 8. 常见问题

### Q: mode='hard' 和 mode='tunable' + tune_preset='nearhard' 有什么区别？

`mode='hard'` 自动设 β=1.0, δ=0.5。`nearhard` 设 β=1.0, δ=2.0。前者 δ 更小，违反区动态范围更大。

### Q: δ 设多大合适？

硬约束（β=1）：δ=0.3~0.5。软约束：δ=1.0~1.5。Tunable：用 `tune_preset` 自动选。

### Q: 违反区会太"平坦"吗？

用 $k=0.2$ 的约束层 σ，$g$ 从 $10^{-6}$ 到 $10^6$ 的 12 个数量级全可区分。只有 $g > 10^6$ 才开始饱和——这是物理极限。

### Q: 支持多少层约束？

任意 $M$。按 priority 排序后逐层嵌套。20 层测试通过。

### Q: 会被"深度满足"扰乱吗？

精确罚函数 $g \ge 0$ 天然不存在深度满足——满足时 $g=0$，$\mathcal{T}(0)=0$，贡献为零。

### Q: 约束函数里能做复杂计算吗？

可以。`g_fn` 可以是任意 JAX 计算——rollout 轨迹、MC 采样、查表。只要返回标量。
