# Constran — 通用黑箱优化 Cost 构造引擎

**Constran** 将多目标 + 约束转化为单一标量 cost。

核心理念：**自相似 σ 嵌套**。obj/√2ⁿ⁺¹ → σₖ → [√2·σ₁ + Φ] × n → √2·σ₁。Φ=0 时完全透明，多层不衰减。

硬度由 **baseline**（0.5~2.0）单一旋钮控制，替代传统的 β/δ/权重调参。

---
## 1. 三步上手

```python
from Constraintdealer.Constran import *

def my_obj(x, ctx):
    return jnp.sum((x[:2] - ctx['target'])**2)

def collision_penalty(x, ctx):
    d = jnp.sqrt(jnp.sum((x[:,:2] - ctx['obs_pos'])**2, axis=-1))
    return jnp.maximum(0.0, ctx['safe_dist'] - d)

# ② 声明层级 — priority 小的在内层（被 σ·m 放大）
layers = [
    Deterministic(collision_penalty, mode='hard',   priority=1, aggregate='max'),
    Deterministic(lambda x,c: jnp.sum(x[2:]**2),
                  mode='tunable', priority=2),
    Deterministic(lambda x,c: jnp.sum((x[:2]-c['target'])**2),
                  mode='soft',    priority=3),
]

# ③ 构建
cost_fn = build(my_obj, layers)
```

**小 priority = 内层（被 σ·m 放大，影响大），大 priority = 外层（直接输出）。**
安全放内层自然被放大，舒适放外层。靠结构保证，不调参。

---
## 2. 核心机制

### 2.1 T_alpha — 多段对数变换

原始 g 从 1e-6 到 1e10（16 个数量级），T_alpha 压缩到 0~6：

```
T_alpha(g) = sign(g) × T_target(|g|)
```

三段式：**地板**（g < resolution，T 恒定）→ **log 增长** → **缓坡天花板**（g > ceiling，T 缓慢增长）。

三档标定表：

| 表 | 分辨率 | 地板 T(0⁺) | 天花板 T(∞) |
|----|--------|-----------|------------|
| `TRANSFORM_SOFT` | 1e-2 | 0.003 | 4.5 |
| `TRANSFORM_TUNABLE` | 1e-4 | 0.02 | 6.0 |
| `TRANSFORM_HARD` | 1e-6 | 0.08 | 6.5 |

### 2.2 自相似 σ 嵌套

$$\text{obj}/\sqrt{2}^{\,n+1} \;\to\; \sigma_k \;\to\; [\,\sqrt{2}\cdot\sigma_1 + \Phi\,] \times n \;\to\; \sqrt{2}\cdot\sigma_1$$

- k 只在最内层（目标函数），约束链全用 σ₁
- Φ=0 时层透明：任意层数无衰减
- 输出 ∈ (-√2, √2) ≈ (-1.41, 1.41)

### 2.3 baseline — 硬度旋钮

```python
Φ = max(0, T(g))               # 精确罚: 只罚违规
Φ = Φ + baseline               # 如果 g > resolution
```

| mode | baseline (默认) | 违规 Δ | 语义 |
|------|--------|--------|------|
| `soft` | 0.5 | ~0.23 | 柔和，跟内层竞争 |
| `tunable` | 1.3 | ~0.40 | 清晰约束 |
| `hard` | 2.0 | ~0.46 | 严格优先 |

baseline 可设任意连续值（0~2），替换传统 β/δ preset。

### 2.4 嵌套即优先级

```
P1 (内) → σ·m → P2 → σ·m → ... → Pn (外)
 ↑ 放大多次, 影响大        ↑ 直接输出, 影响小
 安全约束放这里           舒适约束放这里
```

---
## 3. 优先级嵌套

### 3.1 基本用法

```python
layers = [
    Deterministic(collision, mode='hard',   priority=1, aggregate='max'),
    Deterministic(curvature, mode='tunable', priority=2, aggregate='mean'),
    Deterministic(tracking, mode='soft',    priority=3, aggregate='sum'),
]
cost_fn = build(my_obj, layers)
```

### 3.2 不同约束不同语义

```python
layers = [
    # P1 (内): 避障 — baseline=2.0, 被放大, 最优先
    Deterministic(obs_g,  mode='hard',   priority=1, transform='hard', baseline=2.0),
    # P2: 车道 — baseline=1.3
    Deterministic(lane_g, mode='tunable', priority=2, transform='tunable'),
    # P3: 速度 — baseline=1.0 (自定义)
    Deterministic(spd_g,  mode='tunable', priority=3, transform='tunable', baseline=1.0),
    # P5 (外): 能耗 — baseline=0.5, 不放大
    Deterministic(erg_g,  mode='soft',    priority=5, transform='soft'),
]
```

---
## 4. 每层怎么设

### 决策

```
① 违反能被接受吗？
   ├─ 绝不行 → mode='hard',   baseline=2.0, priority=小
   ├─ 严重时可以 → mode='tunable', baseline=1.3, priority=中
   └─ 只是偏好 → mode='soft',    baseline=0.5, priority=大

② 一个坏点就毁全部吗？
   ├─ 是 → aggregate='max'
   ├─ 否 → aggregate='mean'
   └─ 看总量 → aggregate='sum'
```

### 常见语义速查

| 语义 | mode | baseline | priority | aggregate | transform |
|------|------|----------|----------|-----------|-----------|
| 避障/防撞 | `hard` | 2.0 | 1 (内) | `max` | `hard` |
| 车道偏离 | `tunable` | 1.3 | 2 | `q95` | `tunable` |
| 曲率 | `tunable` | 1.3 | 3 | `mean` | `tunable` |
| jerk/舒适 | `soft` | 0.5 | 4 | `mean` | `soft` |
| 跟踪 | `soft` | 0.5 | 5 (外) | `mean` | `soft` |

---
## 5. 约束类型

| 类型 | 类 | 数学形式 |
|------|-----|---------|
| 确定性 | `Deterministic` | $g(x) \le 0$ |
| 机会约束 | `Chance` | $\mathbb{P}(g\le 0) \ge 1-\alpha$ |
| 鲁棒约束 | `Robust` | $\max_{\xi\in\Xi} g(x,\xi) \le 0$ |
| 分布鲁棒 | `DRO` | $\inf_{\mathbb{Q}} \mathbb{Q}(g\le 0) \ge 1-\alpha$ |

聚合方式：`sum`、`mean`、`max`、`count`、`q90`/`q95`/`q99`。

### 5.4 B-spline 时域约束

B-spline 轨迹通过上百个时域采样点评估，推荐分位数聚合：

```python
Deterministic(bspline_violation, aggregate='q95', mode='tunable', priority=2)
```

| aggregate | 语义 |
|-----------|------|
| `'q95'` | **推荐**，鲁棒且不过度敏感 |
| `'q99'` | 接近 max，容错 1% 野点 |
| `'max'` | 太敏感，B-spline 不推荐 |

---
## 6. 变换表

| 表 | knots_g | knots_T |
|----|---------|---------|
| SOFT | [1e-2, 5e-2, 1e-1, 0.5, 1, 10, 100, 1e4, 1e6, 1e8, 1e10] | [0.003, 0.015, 0.06, 0.25, 0.7, 2.2, 3.5, 4.0, 4.2, 4.4, 4.5] |
| TUNABLE | [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 10, 100, 1e4, 1e6, 1e8, 1e10] | [0.02, 0.06, 0.15, 0.4, 0.8, 1.5, 3.0, 4.5, 5.0, 5.3, 5.7, 6.0] |
| HARD | [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 10, 100, 1e4, 1e6, 1e8, 1e10] | [0.08, 0.15, 0.3, 0.6, 1.2, 2.0, 3.0, 4.5, 5.5, 5.8, 6.2, 6.5] |

---
## 7. 数值特性

| 参数 | 值 | 说明 |
|------|-----|------|
| 结构 | obj/√2ⁿ⁺¹ → σₖ → [√2·σ₁ + Φ] × n → √2·σ₁ | 自相似，无衰减 |
| 输出范围 | (-√2, √2) ≈ (-1.41, 1.41) | 最终 σ·m 包裹 |
| Φ=0 透明 | 任意层数 = obj_only | 5×SOFT 验证 |
| baseline | 0.5 / 1.3 / 2.0 | SOFT / TUNABLE / HARD |
| k_inner | 0.1~1.0 | 按目标范围自选 |

---
## 8. 常见问题

**Q: priority 大的在外层？**
A: 对。小 priority = 内层 = 被 σ·m 放大 = 影响大。安全放内层，舒适放外层。

**Q: baseline 和以前的 β/δ 什么关系？**
A: baseline 替代了 β/δ。硬度由 baseline 单一旋钮控制，0.5~2.0 连续可调。

**Q: 天花板会卡死求解器吗？**
A: 不会。缓坡天花板，极端违规时 T 仍缓慢增长。

**Q: 地板会制造盲区吗？**
A: 故意的。分辨率以下的违规不算违规。
