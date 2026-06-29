# Constran — 通用黑箱优化 Cost 构造引擎

**Constran** 是一个将多目标 + 约束转化为单一标量 cost 的引擎，供 IGO/GMM 等零阶求解器使用。

核心理念：**σ 饱和嵌套 = 优先级编码**。外层先优化，内层在外层满足后精细优化。无需手动调权重，无需 `jnp.where` 分支。

---

## 目录

1. [三步上手](#1-三步上手)
2. [核心机制](#2-核心机制)
3. [优先级嵌套](#3-优先级嵌套)
4. [逐层 k 校准](#4-逐层-k-校准)
5. [约束类型与聚合](#5-约束类型与聚合)
6. [变换表与预设](#6-变换表与预设)
7. [MPC 用法](#7-mpc-用法)
8. [数值特性参考](#8-数值特性参考)
9. [常见问题](#9-常见问题)

---

## 1. 三步上手

```python
from Constraintdealer.Constran import *

# ① 写目标和约束 — 约束用精确罚函数 max(0, g(x))
def my_obj(x, ctx):
    return jnp.sum((x[:2] - ctx['target'])**2)

def collision_penalty(x, ctx):
    d = jnp.sqrt(jnp.sum((x[:,:2] - ctx['obs_pos'])**2, axis=-1))
    return jnp.maximum(0.0, ctx['safe_dist'] - d)  # 向量或标量

# ② 声明层级 — priority 越小越外层（越优先）
layers = [
    Deterministic(collision_penalty, mode='hard',   priority=1, aggregate='max'),
    Deterministic(lambda x,c: jnp.sum(x[2:]**2),
                  mode='tunable', priority=2, tune_preset='standard'),
]

# ③ 构建 → 给求解器
cost_fn = build(my_obj, layers)                     # auto_k=True 默认开启
result = solver(..., fitness_fn_total=cost_fn, context=ctx)
```

**你只需要写目标和违反函数。** Constran 自动处理：T_alpha 范围压缩 → σ 饱和 → 逐层嵌套 → 逐层 k 校准。

---

## 2. 核心机制

Constran 用三个构建块把任意数量、任意尺度的目标和约束组装成一个标量。

### 2.1 T_alpha — 多段对数变换

原始 $g$ 可能从 $10^{-6}$ 到 $10^6$（12 个数量级），直接塞给求解器会让大值淹没小值。T_alpha 用分段的 log 类变换把 $g$ 压缩到 $[0, 13]$：

```
T_alpha(g) = sign(g) × T_target(|g|)
```

通过对数-线性插值结点表实现。关键特性：
- **小 g 有地板**：$g \to 0^+$ 时 $T \to T_{\text{floor}} > 0$，确保微小违反立即被感知
- **大 g 近似 log**：$g$ 很大时 $T \approx \log(g)$，大范围可区分
- **g=0 恰好为 0**：满足时贡献为零

三档标定表（分辨率递增）：

| 表 | 分辨率 | 地板 T(0⁺) | 用途 |
|----|--------|-----------|------|
| `TRANSFORM_SOFT` | $10^{-2}$ | 0.03 | 偏好、微调 |
| `TRANSFORM_TUNABLE` | $10^{-4}$ | 0.15 | 通用可调 |
| `TRANSFORM_HARD` | $10^{-6}$ | 0.5 | 安全、硬约束 |

### 2.2 σ_k — 饱和函数

$$\sigma_k(x) = \frac{kx}{\sqrt{1 + (kx)^2}}, \quad \text{输出} \in (-1, 1)$$

$k$ 控制拐点位置：$\sigma$ 在 $x = 1/k$ 处达到 $1/\sqrt{2} \approx 0.707$。

| k | 拐点 (T 值) | 近似线性区 | 动态范围/层 |
|---|------------|----------|-----------|
| 0.2 | T=5 | T ∈ [0, 5] | ~3 decades |
| 0.5 | T=2 | T ∈ [0, 2] | ~2 decades |
| 1.0 | T=1 | T ∈ [0, 1] | ~1.5 decades |

小 k = 宽线性区 = 强优先级。大 k = 窄线性区 = 好穿透力。**二者不可兼得——逐层 k 校准解决这个矛盾（见 §4）。**

### 2.3 两种模式

所有层都是加性的（无分支）。只有两种模式：

```
Soft:    cost = σ_k( T(g) + inner )
Tunable: cost = σ_k( δ·σ₁(β·T(g)) + inner )
```

| 模式 | 公式 | 参数 | 何时用 |
|------|------|------|--------|
| **Soft** | $T(g) + \text{inner}$ | 无 | 偏好、微调、最简场景 |
| **Tunable** | $\delta \cdot \sigma_1(\beta \cdot T(g)) + \text{inner}$ | β, δ | 需要控制"软硬"的层 |

Tunable 的 β 控制"多快触发"（类似硬度），δ 控制"最多扣几分"（类似权重）。常用预设：

| tune_preset | β | δ | 语义 |
|-------------|----|----|------|
| `'mild'` | 0.1 | 1.0 | 极软，大违反才触发 |
| `'standard'` | 0.3 | 1.0 | 标准（默认） |
| `'firm'` | 0.5 | 1.5 | 适中 |
| `'strong'` | 1.0 | 1.5 | 较硬 |
| `'nearhard'` | 1.0 | 2.0 | 近似硬约束 |
| `'hard'`（自动→） | 1.0 | 1.5 | 硬约束 |

`mode='hard'` 自动映射为 `tunable + β=1.0`，无需手动设参数。

### 2.4 嵌套即优先级

`build()` 把所有层按 priority 排序后嵌套。**priority 数字越小越外层，越优先满足：**

```
最终 cost = σ_{k_n}(  contrib_n  +  σ_{k_{n-1}}(  contrib_{n-1}  +  ...  +  σ_{k₁}(  T(obj)  ) ... ))
                      ↑ 最外层 P1                            ↑ 最内层 Pn
```

每层 σ 把内部所有内容再压缩一次。**外层直接命中最终 cost，内层被层层压缩**——求解器自然优先优化外层。

**选择压力数据**（k=0.2 外层 vs k=1.0）：

| 外层违反 g | k=0.2 的导数 | k=1.0 的导数 | 说明 |
|-----------|------------|------------|------|
| 100 | 0.083 | 0.103 | k=0.2 在严重违反时也有信号 |
| 10 | 0.200 | 0.093 | k=0.2 峰值在中度违反区 |
| 1 | **0.355** | 0.093 | k=0.2 选择压力是 k=1.0 的 3.5× |
| 0.1 | 0.155 | 饱和 | k=1.0 进入死区 |

**k=0.2 的"缓坡"不是缺陷**——它让求解器在宽范围内始终有方向可走，不会像 k=1.0 那样遇到饱和死区。

---

## 3. 优先级嵌套

### 3.1 基本用法

```python
layers = [
    # P1 最外层：碰撞 — 任何一点都不行
    Deterministic(collision, mode='hard', priority=1, aggregate='max'),
    # P2：曲率 — 整体平滑，允许偶尔急转
    Deterministic(curvature, mode='tunable', priority=2, aggregate='mean',
                  tune_preset='standard'),
    # P3 最内层：跟踪 — 软偏好，好跟踪可补偿
    Deterministic(tracking, mode='soft', priority=3, aggregate='sum'),
]
cost_fn = build(my_obj, layers)
```

### 3.2 动态范围分配

外层获得更大的 cost 动态范围，内层被压缩——这就是优先级机制：

| 层 | 嵌套位置 | Cost 范围 | 可区分数 | 含义 |
|----|---------|----------|---------|------|
| collision (P1) | 外层 | 0.26 | ~2000 | 求解器主要优化目标 |
| curvature (P2) | 中层 | 0.035 | ~2900 | 在安全前提下优化 |
| tracking (P3) | 内层 | 0.007 | ~2900 | 在最内层微调 |

### 3.3 物理链嵌套（ODE 积分链）

对于 jerk → acc → vel → pos 积分链，**下导数（决策量）应嵌套在外层，上状态应嵌套在内层**：

```python
layers = [
    Deterministic(jerk_viol,  mode='tunable', priority=1, tune_preset='firm'),
    Deterministic(acc_viol,   mode='tunable', priority=2, tune_preset='standard'),
    Deterministic(vel_viol,   mode='tunable', priority=3, tune_preset='standard'),
    Deterministic(pos_viol,   mode='soft',    priority=4),
]
```

等价于隐式编码"加速度不能突变 → 速度不能突变 → 位置连续"——无需显式写 ODE。

**数值验证**：Physics（jerk 外→pos 内）vs Reversed（pos 外→jerk 内）：

| 场景 | Physics | Reversed |
|------|:-:|:-:|
| jerk=100, pos=0 | **0.271** | 0.006 |
| jerk=0, pos=100 | 0.006 | **0.271** |
| pos=10 固定, jerk 扫 0→1000 | Δ=**+0.277** | Δ=+0.006 |

Physics 顺序下 cost 随 jerk 剧烈变化、随 pos 几乎不变 → 求解器拼命降 jerk，在平滑前提下微调位置。

### 3.4 层数选择

| 层数 | 效果 |
|------|------|
| 1-2 | 弱优先级，目标和约束几乎平等竞争 |
| 3-5 | 推荐范围，优先级梯度清晰（如 Simple.py） |
| 6-10 | 需要 `auto_k=True`（默认），逐层 k 校准保证穿透 |
| >10 | 建议把底层合并到同一 priority，共享 σ 层 |

---

## 4. 逐层 k 校准

### 4.1 为什么需要

所有层用相同 k（如全局 k=0.2）时，最内层信号经过 n 层 σ 压缩，增益为 $0.2^n$：

| n | 最内层增益 | 可区分 f32 值 | 状态 |
|---|----------|-------------|------|
| 4 | $1.6 \times 10^{-3}$ | ~3000 | ✓ |
| 7 | $1.3 \times 10^{-5}$ | ~200 | 临界 |
| 10 | $1.0 \times 10^{-7}$ | ~5 | ☠ 死 |
| 12 | $4.1 \times 10^{-9}$ | ~1 | ☠ 求解器发散 |

### 4.2 几何 taper

`auto_k=True`（默认）时，`build()` 自动为每层分配不同的 k：

```
k_i = k_outer × r^(n-i)    [外层 k_n=0.2, 内层逐步增大到 k→1.0]
```

- 外层保持 k≈0.2（强优先级，3 decades 范围）
- 内层逐步增大到 k→1.0（深度无关穿透，增益恒为 1）
- 一旦 k 达到 1.0，再加层无额外压缩

**T_alpha 缩放补偿**：内层 k 增大后，σ 拐点左移（更容易饱和）。T_alpha 表同步缩放 `T_new = T_old × (0.2/k_i)`，保持 σ(T_max) 恒定。数学上，σ∘T 曲线在不同 k 下完全重合。

### 4.3 效果验证

n=10 层，2000 点密集采样：

| 层 | Equal k=0.2 | Auto-calibrated |
|----|------------|----------------|
| 外层 (P1) | 1429 ✓ | 1429 ✓ |
| 中层 | 1144 ✓ | 1144 ✓ |
| **内层 (P10)** | **5 ☠** | **1003 ✓** |

n=12 层：

| 层 | Equal k=0.2 | Auto-calibrated |
|----|------------|----------------|
| 外层 (P1) | 1429 ✓ | 1429 ✓ |
| 内层 (P12) | **1 ☠** | **786 ✓** |

全部满足 500-1000 可分辨值的需求。

### 4.4 控制参数

```python
cost_fn = build(obj, layers,
    auto_k=True,        # 默认开启，≥2 层时生效
    k_outer=0.2,        # 最外层 k，增大 = 加快外层收敛但减小范围
)
# 关闭校准 → 所有层用全局 k=0.2
cost_fn = build(obj, layers, auto_k=False)
```

更深嵌套用更大 target_gain：

```python
from Constraintdealer.Constran import auto_calibrate_k
ks = auto_calibrate_k(15, k_outer=0.2, target_gain=0.01)  # 更强穿透
```

---

## 5. 约束类型与聚合

### 5.1 四种约束

| 类型 | 输入 | Constran 自动 |
|------|------|-------------|
| `Deterministic` | `g(x)` | `aggregate` 聚合 → 标量 |
| `Chance` | `g(x,ξ)` + 噪声 | 聚合 → $Q_{1-\alpha}$ |
| `Robust` | `g(x,ξ)` + 不确定集 Ξ | 聚合 → $\max_{\Xi}$ |
| `DRO` | `g(x,ξ)` + 模糊集 | 聚合 → $\max_P Q_{1-\alpha}$ |

### 5.2 内置聚合

轨迹上每采样点构成向量。用 `aggregate` 指定聚合方式，无需在 `g_fn` 里手写 sum/mean：

| aggregate | 语义 | 适用 |
|-----------|------|------|
| `'sum'`（默认） | 总违反量 | 能耗、总偏差 |
| `'mean'` | 步均违反 | 跟踪误差（与时域解耦） |
| `'max'` | 最危险一步 | 避障、速度限制 |
| `'count'` | 违反步数 | 广度优于深度 |
| `'q90'`,`'q95'`,`'q99'` | 分位数 | 允许 10%/5%/1% 点擦边 |

### 5.3 聚合决策表

| 约束 | 推荐聚合 | 原因 |
|------|---------|------|
| 避障/碰撞 | `max` 或 `q95` | 最危险点决定安全 |
| 速度/加速度限制 | `max` | 任何一步超限即违规 |
| 曲率/平滑 | `mean` 或 `q90` | 整体舒适度 |
| 跟踪误差 | `mean` | 与时域解耦 |
| 能耗/燃料 | `sum` | 总消耗 |
| 终端约束 | 不加聚合 | 只看最后一步 |

### 5.4 语义速查

```
"任何碰撞都不行"
  → Deterministic(viol, mode='hard', priority=1, transform='sharp')
"小擦边无所谓，别大撞就行"
  → Deterministic(viol, mode='tunable', priority=2, tune_preset='standard')
"只是偏好，好跟踪可补偿偶尔擦边"
  → Deterministic(viol, mode='soft', priority=3)
```

---

## 6. 变换表与预设

### 6.1 约束变换

```
transform='soft'     — 地板 T(0⁺)=0.03, 分辨率 1e-2, 偏好/微调
transform='tunable'  — 地板 T(0⁺)=0.15, 分辨率 1e-4, 通用（默认）
transform='hard'     — 地板 T(0⁺)=0.5,  分辨率 1e-6, 安全约束
```

自定义变换表（调容忍度）：

```python
my_table = (
    np.array([1e-4, 1e-2, 1e-1, 1, 10, 100, 1e4, 1e6]),   # knots_g
    np.array([0.01, 0.05, 0.2, 0.5, 1.5, 4.0, 8.0, 12.0]), # knots_T
)
Deterministic(viol, mode='soft', priority=1, _transform_table=my_table)
```

### 6.2 目标变换

```python
cost_fn = build(my_obj, layers, obj_transform='standard')  # 默认
cost_fn = build(my_obj, layers, obj_transform='flat')      # 更平, 超大范围
cost_fn = build(my_obj, layers, obj_transform='log')       # 原版 log
```

---

## 7. MPC 用法

**关键：build 一次，ctx 传动态信息。**

```python
cost_fn = build(my_obj, layers)  # ← 只调一次

for step in range(T_mpc):
    ctx = {'target': targets[step], 'obs_pos': obs[step]}
    result = solver(..., fitness_fn_total=cost_fn, context=ctx)
```

多智能体：

```python
agent_fns = build_multi_agent({
    0: (obj_agent0, [Deterministic(v0, mode='hard', priority=1)]),
    1: (obj_agent1, []),
})
```

---

## 8. 数值特性参考

### 8.1 Float32 分辨率

IGO 求解器不用梯度，纯靠 float32（~7 位精度）比大小。如果多个样本 cost 相同 → 求解器无选择压力 → 发散。

Constran 保证每层 ≥ 500 可区分 f32 值（默认 auto_k）：

| 深度 | 最外层 | 中层 | 最内层 | 达标 |
|------|--------|------|--------|------|
| 4-8 | 1429 | 1144 | 1144 | ✓✓ |
| 10 | 1429 | 1144 | 1003 | ✓✓ |
| 12 | 1429 | 1143 | 786 | ✓ |
| 15 | 1429 | 1143 | ~500 | 需调 target_gain |

### 8.2 深度极限

| k 策略 | 最大安全深度 | 衰减模式 |
|--------|------------|---------|
| Equal k=0.2 | ~7 | 指数 $0.2^n$ |
| Equal k=0.5 | ~17 | 指数 $0.5^n$ |
| Equal k=1.0 | 200+ | 无衰减（增益恒 1） |
| Taper 0.2→1.0 | 15+ | 内层穿透保持 |

**k=1.0 定理**：σ₁'(0) = 1，所以 σ₁^(n)'(0) = 1。小信号穿透力与深度无关。

### 8.3 k 的选择压力

外层 k=0.2 的缓坡意味着求解器在中度违反区（g≈1-10）有最强选择压力（导数 0.36，是 k=1.0 的 3.5×），在严重违反区（g≈100）仍有信号（导数 0.08）。k=1.0 在 g>10 后进入饱和死区。

---

## 9. 常见问题

### Q: 嵌套顺序怎么确定？

**先安全，后性能，最后微调。** 碰撞 > 动力学限制 > 平滑 > 跟踪。积分链中下导数在外、状态在内。

### Q: soft vs tunable 怎么选？

不確定時用 `tunable + standard`。確定只是微調偏好用 `soft`。安全相關用 `hard`。

### Q: auto_k 会改变现有代码的行为吗？

外层 k 不变（仍为 0.2），内层 k 增大 → 内层穿透力更强、目标函数压缩更少。这对优化行为是改善。如需完全兼容，设 `auto_k=False`。

### Q: δ 设多大？

- 硬约束：δ = 0.3~0.5
- 软约束/偏好：δ = 1.0~1.5
- 用 `tune_preset` 自动选

### Q: 会被"深度满足"扰乱吗？

精确罚函数 g ≥ 0 天然不存在深度满足——满足时 g=0, T(0)=0, 贡献为零。

### Q: 先 sum 再 T 还是先 T 再 sum？

默认先 sum 再 T（g_fn 返回标量，Constran 做 T）。如需先 T 再 sum，直接在 g_fn 里做——双重 T 不破坏单调性。

---

## 附录：完整示例 (Simple.py 同款)

```python
from Constraintdealer.Constran import *

def my_obj(x, ctx):
    pos, vel = evaluate_trajectory(x, ctx)
    return (25.0 * jnp.sum((pos[:,1] - ctx['y_ref'])**2) +
             5.0 * jnp.sum((vel - ctx['v_ref'])**2))

layers = autodelta([
    # P1 最外层：避障 — 硬约束，max（最危险点）
    Deterministic(collision_pen, mode='hard', priority=1,
                  aggregate='max', transform='hard'),
    # P2：曲率 — tunable，mean（整体平滑）
    Deterministic(curvature_pen, mode='tunable', priority=2,
                  aggregate='mean', tune_preset='standard'),
    # P3：jerk — soft，mean（舒适偏好）
    Deterministic(jerk_pen, mode='soft', priority=3,
                  aggregate='mean', transform='soft'),
    # P4：车道偏离 — soft，sum（总偏离）
    Deterministic(lane_pen, mode='soft', priority=4,
                  aggregate='sum', transform='soft'),
    # P5 最内层：速度限制 — tunable，max（任何一步超限）
    Deterministic(speed_pen, mode='tunable', priority=5,
                  aggregate='max', tune_preset='firm'),
])

cost_fn = build(my_obj, layers)  # auto_k=True 默认
# → 外层 k≈0.2（强优先级），内层 k→0.49（穿透）
# → 每层 ≥ 1000 可区分 f32 值
```
