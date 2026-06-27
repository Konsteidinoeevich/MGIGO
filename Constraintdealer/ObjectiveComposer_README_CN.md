# ObjectiveComposer —— 结构化子目标组合

## 1. 问题：手动调权重的困境

实际工程中的目标函数包含多个互相竞争的子项。传统做法是给每一项赋一个权重，然后加在一起：

```python
f = 15.0 * 跟踪误差 + 2.0 * 平滑性 + 0.1 * 控制代价
```

这样做能工作，但有众所周知的痛点：

| 痛点 | 原因 |
|------|------|
| **无界** | 权重 15 没有上限 —— 一条烂轨迹可以让 f 爆炸 |
| **跨项量级失配** | 跟踪 ~100，平滑性 ~1 —— 权重必须补偿量级差 |
| **逐问题重调** | 新环境 → 新权重 → 反复试错 |
| **无物理意义** | `w=15.0` 是什么意思？什么都解释不了 |

## 2. 解决方案：逐项饱和增益

每一项独立经过：

```
原始值 → log_transform → sigma_k(..., k=该项的k) → 有界于 [0, 1)
```

所有压缩后的项求和，再经过一层外部的 sigma：

```
f(x) = σ( Σ σ_k( T(term_i(x)), k=k_i ), k=k_outer )
```

### 为什么有效

| 属性 | 手动权重 | Compose Objective |
|------|---------|-------------------|
| 有界性 | 无界 | 每项 ∈ [0, 1) |
| 量级处理 | 手动（除以 S） | 自动（log_transform） |
| 增益控制 | 无单位权重 w | k → 物理单位的 knee |
| 防主导 | 一项可主导全部 | 全部有界 → 单项无法劫持总代价 |

## 3. k 即物理增益

**k 控制 knee（拐点）**——该项在什么原始值处达到半饱和：

```
raw_knee ≈ exp(1/k) - 1
```

| k | 原始 knee | 适用场景 |
|---|----------|----------|
| 5.0 | ~0.22 m | 横向跟踪 —— 亚米级精度很关键 |
| 3.0 | ~0.40 m | 近距障碍物缓冲 |
| 2.0 | ~0.65 m | 终端逼近精度 |
| 1.0 | ~1.7 | 均衡，中等灵敏度 |
| 0.5 | ~6.4 | 路径积分跟踪 —— 宽范围 |
| 0.2 | ~147 | 纵向跟踪 —— ~100m 量级 |
| 0.1 | ~2.2×10⁴ | 控制代价 —— 只惩罚极端浪费 |

**大 k = 高增益**：饱和快，对小变化敏感。
**小 k = 低增益**：线性区宽，只有很大值才受影响。

这比权重**更具可解释性**：knee 为 0.65m 意味着"我在乎亚米级的终端精度"。
权重 15.0 意味着……什么都解释不了。

## 4. 快速入门

### 4.1 推荐：自适应（语义角色）

```python
from Constraintdealer.ObjectiveComposer import compose_objective_adaptive

ctx_calib = {'target': jnp.array([8.0, 6.0]),
             'init_state': jnp.array([0., 0., 0., 0.])}

obj = compose_objective_adaptive([
    (跟踪函数,   'primary',    "跟踪"),       # P50：knee 在中位数
    (终端函数,   'secondary',  "终端"),       # P70
    (控制函数,   'tiebreaker', "控制"),       # P95
], n_dims=16, bounds=(-5.0, 5.0), n_samples=500,
   ctx_calib=ctx_calib)
# obj 是标准 (x, ctx) -> scalar 可调用对象，输出在 [0, 1)
```

### 4.2 生产级：动态 k

见 [§8 动态 k](#8-动态-k免-jax-重编译)。

### 4.3 与 Constran.build() 集成

```python
from Constraintdealer.Constran import build, Deterministic

# obj 来自 compose_objective_adaptive 或 compose_objective_dynamic
cost_fn = build(obj, [
    Deterministic(障碍函数,  mode='hard', priority=1),
    Deterministic(行人函数,  mode='hard', priority=2),
])

# 步骤 3：传给求解器
result = mmog_igo_optimizer_mpc(..., fitness_fn_total=cost_fn, ...)
```

### 4.4 诊断各项贡献

```python
from Constraintdealer.ObjectiveComposer import inspect_terms

info = inspect_terms(obj, x_test, ctx, [
    (跟踪函数,   0.5, "跟踪"),
    (终端函数,   2.0, "终端"),
    (控制函数,   1.0, "控制"),
])
for c in info['contributions']:
    print(f"{c['name']}: 原始={c['raw']:.3f} → 压缩={c['compressed']:.3f} [{c['saturated']}]")
# 输出：
#   跟踪: 原始=8.000 → 压缩=0.740 [knee]
#   终端: 原始=0.500 → 压缩=0.528 [knee]
#   控制: 原始=0.100 → 压缩=0.058 [linear]
```

这告诉你哪些项已饱和（拉满了 —— 其 k 可能太高），哪些处于 knee 区（灵敏度好），
哪些仍是线性的（k 可能太低，该项几乎没贡献）。

## 5. 前后对比

### 之前：手动权重

```python
def my_objective(z_flat, ctx):
    # 计算原始值
    track  = compute_tracking(z_flat, ctx)       # 范围 ~0–500
    final  = compute_final_error(z_flat, ctx)    # 范围 ~0–50
    smooth = compute_smoothness(z_flat, ctx)     # 范围 ~0–10
    ctrl   = compute_control(z_flat, ctx)        # 范围 ~0–5

    # 每个新问题都要调这些权重：
    f = 1.0 * track + 15.0 * final + 1.5 * smooth + 0.1 * ctrl

    # 期望求解器能在跨越 3 个数量级的值中做区分
    return f
```

### 之后：自适应

```python
from Constraintdealer.ObjectiveComposer import compose_objective_adaptive

obj = compose_objective_adaptive([
    (compute_tracking,   'primary',    "跟踪"),    # P50 — 自动 knee
    (compute_final_error,'secondary',  "终端"),    # P70
    (compute_smoothness, 'tiebreaker', "平滑"),    # P95
    (compute_control,    'tiebreaker', "控制"),    # P95
], n_dims=16, bounds=(-5,5), n_samples=500, ctx_calib=ctx)
# 完成。无需权重、无需 k。只需语义角色。
# 输出始终在 [0, 1)。直接传给 Constran.build()。
```

## 6. 选择 k 的指南

### 方法 A：物理 knee（"我知道什么值重要"）

1. 决定："该项在什么原始值处应该半饱和？"
2. `k = knee_to_k(raw_knee)`

```python
from Constraintdealer.ObjectiveComposer import knee_to_k

k_lat = knee_to_k(0.3)   # "0.3m 横向误差 → knee"
k_lon = knee_to_k(10.0)  # "10m 纵向误差 → knee"
```

### 方法 B：典型值 + 最大值（"我知道工作范围"）

1. 估计典型值和最大原始值
2. `k = suggest_k(典型值, 最大值)`

```python
from Constraintdealer.ObjectiveComposer import suggest_k

k = suggest_k(typical=2.0, max=100.0)
```

### 方法 C：从默认开始，诊断，迭代

1. 所有项从 `k=1.0` 开始
2. 对几个候选解跑 `inspect_terms()`
3. 若某项总是显示 `[saturated]` → 减小 k（拓宽范围）
4. 若某项总是显示 `[linear]` 且贡献很小 → 增大 k

## 7. 自适应 k：通过语义角色自动校准

当你**不知道**各项的量级时，用 `compose_objective_adaptive`。你不需要指定 k，
只需给每项分配一个**语义角色**，k 会从随机样本的经验分布中自动校准。

### 7.1 核心思想

| 你知道 | 你不知道 |
|--------|---------|
| 这是我的*主要*目标 | 它的典型原始值 |
| 这是*次要*偏好 | 它的最大值 |
| 这是*平局决胜*项 | 它的动态范围 |

角色映射到一个**分位数（percentile）**。knee 就取在该分位数处，k 自动导出。

| 角色 | 分位数 | knee 位置 | 含义 |
|------|--------|----------|------|
| `'primary'` | P50 | 中位数 | 宽线性区 — 最差一半才饱和 |
| `'secondary'` | P70 | 70分位 | 适中 — 70% 在线性区 |
| `'tiebreaker'` | P95 | 95分位 | 几乎全线性 — 仅极端值饱和 |

也可以直接传浮点数：`0.40` = P40, `0.99` = P99。

### 7.2 用法

```python
from Constraintdealer.ObjectiveComposer import compose_objective_adaptive

ctx_calib = {'target': jnp.array([8.0, 6.0]),
             'init_state': jnp.array([0., 0., 0., 0.])}

obj = compose_objective_adaptive([
    (跟踪函数,   'primary',    "跟踪"),       # P50
    (终端函数,   'secondary',  "终端"),       # P70
    (平滑函数,   'tiebreaker', "平滑"),       # P95
], n_dims=16, bounds=(-5.0, 5.0), n_samples=1000,
   ctx_calib=ctx_calib)
```

校准期间输出：
```
--- Adaptive K Calibration ---
  跟踪              : P50  knee=4.2371  k=0.6645
  终端              : P70  knee=0.8920  k=1.4752
  平滑              : P95  knee=6.1804  k=0.5100
```

### 7.3 样本来源（根据成本选择）

| 来源 | 参数 | 额外开销 | 适用场景 |
|------|------|---------|---------|
| 随机均匀 | `n_dims` + `bounds` | n_samples × term 调用 | 冷启动，term 便宜 |
| 预热样本 | `warmup_samples` | **零** | MPC：复用求解器第一步的样本 |
| 自定义生成器 | `sample_fn` | n_samples 次调用 | 已知决策空间结构 |

```python
# 零开销：复用求解器 warmup 阶段的样本
obj = compose_objective_adaptive([
    (track_fn, 'primary', "跟踪"),
], warmup_samples=mu_k_samples, ctx_calib=ctx)

# 低成本：随机均匀
obj = compose_objective_adaptive([
    (track_fn, 'primary', "跟踪"),
], n_dims=16, bounds=(-3.0, 3.0), n_samples=500, ctx_calib=ctx)
```

### 7.4 与 build() 集成

```python
from Constraintdealer.Constran import build, Deterministic

obj = compose_objective_adaptive([...], n_dims=16, ctx_calib=ctx)
cost_fn = build(obj, [Deterministic(obs_fn, mode='hard', priority=1)])
# cost_fn 直接给求解器
```

## 8. 动态 k：免 JAX 重编译

优化过程中 cost 量级会漂移：初期 ~500，收敛后 ~10。用初期样本校准的 k
收敛时 knee 太宽，各项掉进线性区底层——在最需要区分度的时候最不敏感。

**解法：k 放进 ctx，而非闭包。** ctx 本来就是动态的（MPC 每步变 target/障碍物），
修改 ctx 里的 k 值**不会触发 JAX 重编译**。

### 8.1 三个函数

| 函数 | 作用 |
|------|------|
| `calibrate_k_into_ctx(ctx, terms, ...)` | 初始校准 → 写入 `ctx['k_{name}']` |
| `compose_objective_dynamic(terms)` | 构建从 ctx 读取 k 的 cost fn |
| `update_k_from_elite(ctx, terms, elite_x)` | 用精英解更新 k（knee = 原始值×5） |

### 8.2 MPC 循环模式

```python
from Constraintdealer.ObjectiveComposer import (
    compose_objective_dynamic, calibrate_k_into_ctx, update_k_from_elite
)

# 1. 初始校准（一次）
ctx = calibrate_k_into_ctx(ctx, [
    (跟踪函数, 'primary',    '跟踪'),
    (终端函数, 'secondary',  '终端'),
], n_dims=16, bounds=(-5,5), n_samples=500)

# 2. 构建代价函数（一次）
obj = compose_objective_dynamic([
    (跟踪函数, 0.0, '跟踪'),    # k 占位——真正的 k 从 ctx 读取
    (终端函数, 0.0, '终端'),
])
cost_fn = Constran.build(obj, constraints)

# 3. MPC 循环——每步用精英更新 k
for step in range(T_mpc):
    ctx['target'] = new_target
    ctx['obs_pos'] = new_obstacles

    result = solver(..., fitness_fn_total=cost_fn, context=ctx)

    # 用精英解更新 k（零额外评估）
    elite_x = result.best_solution
    ctx = update_k_from_elite(ctx, terms, elite_x, multiplier=5.0)
```

**JAX 编译一次。** ctx 里的 k 值自由变化——不触发重编译。

### 8.3 `update_k_from_elite` 如何工作

对每项，在精英解上评估 `term_fn(elite_x, ctx)`，设 `knee = raw * multiplier`，
然后 `k = 1/log(1 + knee)`。`multiplier=5` 时，knee 在当前最佳值 5 倍处——
给优化留改善空间，同时保持敏感度。

```
更新前（随机样本校准）: knee=500, k=0.16  → σ(10)=0.36（平坦）
更新后（精英校准）:     knee=50,  k=0.26  → σ(10)=0.69（knee区！）
```

### 8.4 为什么 cost 平坦会杀死 IGO 求解器

当 cost 地形趋于平坦，整个 IGO 优化会通过一条因果链崩溃：

```
cost 变平坦（σ'(x) → 0）
  → argsort 变成随机噪声（所有候选看起来一样好/差）
    → 自然梯度指向随机方向
      → GMM 分布无引导地膨胀扩散
        → 采样飞到极端值（几百上千）
          → eig(S) → 0 或 ∞  →  NaN / 溢出
```

IGO 依赖排序来估计自然梯度。当 cost 无法区分好坏解时，排序等同于随机，
梯度是纯噪声。GMM 膨胀，采样发散，求解器**崩溃**。

`MPCsolverM22` 中现有的保护措施（特征值 clamp `[1e-3, 1e3]`、
IGO 权重 clip `±20`）是**治标的止血带**——只在最后一步拦截崩溃。
动态 k 是**治本**：让各项始终处于 knee 区，cost 永不平坦，
`argsort` 始终有信号，整条链根本不会启动。

**实验确认：** 在 500 次 IGO 迭代、12 步 horizon 的对比测试中，
最优解附近的 cost 敏感度为：

| 方案 | 最优解附近敏感度 | 状态 |
|------|-----------------|------|
| 动态 k | 0.0049 | 敏感——求解器能看到改进方向 |
| 静态 k | 0.0000 | 彻底平坦——求解器"瞎了" |
| 原始权重 | 0.000006 | 基本平坦——求解器"瞎了" |

## 9. API 参考

### `compose_objective_adaptive(terms, *, n_dims=..., bounds=..., ..., ctx_calib=..., k_outer=1.0)`

从语义角色自动校准 k。每项是 `(fn, role, name?)`。
角色：`'primary'`（P50）、`'secondary'`（P70）、`'tiebreaker'`（P95），
或浮点数分位数。必须提供 `ctx_calib` 和样本来源。

### `knee_to_k(raw_knee) -> float`

将物理 knee 转换为 k。`raw_knee = exp(1/k) - 1`。

### `k_to_knee(k) -> float`

将 k 转换为物理 knee。`knee_to_k` 的逆函数。

### `suggest_k(raw_typical, raw_max=None) -> float`

从典型值和最大值启发式推荐 k。

### `inspect_terms(composed_obj, x, ctx, terms_info) -> dict`

逐项诊断：原始值、压缩值、饱和状态。

## 10. 不适用的情况

- **单项目标**：用 `build_unconstrained()` 更简单
- **已经归一化**：如果所有项本来就自然落在 [0, 1]，直接求和即可
- **已知固定权重**：如果权重已经调好且工作正常，保持原样
- **基于梯度的求解器**：sigma 嵌套增加了非线性，请检查梯度方法是否容忍（IGO 无梯度，无此问题）

## 11. Demo

```bash
# 对比三种模式（不需要求解器）
uv run python Functiontest/ObjectiveComposer_demo.py

# 带 IGO 优化
uv run python Functiontest/ObjectiveComposer_demo.py --optimize

# 单一模式
uv run python Functiontest/ObjectiveComposer_demo.py --mode composed --optimize
```

## 12. 与 Constran 的关系

```
┌──────────────────────────────────────────────────┐
│  Constran.build(objective_fn, constraints)        │
│                                                   │
│  objective_fn 可以是:                              │
│    ├── 普通 Python 函数   (传统方式)                │
│    ├── compose_objective([...])   ← 新增          │
│    └── compose_objective_auto([...])  ← 新增      │
│                                                   │
│  Constran 负责:                                   │
│    ├── 约束嵌套 (hard/soft/tunable)               │
│    ├── 逐约束 log_transform + sigma_k             │
│    └── δ 自动分配                                  │
│                                                   │
│  ObjectiveComposer 负责:                          │
│    ├── 逐子项 log_transform + sigma_k             │
│    ├── 子项间加法竞争                              │
│    └── 基于 k 的增益控制                           │
└──────────────────────────────────────────────────┘
```

两层使用相同的数学原语（`sigma_k`、`log_transform`），但作用于不同层次：
ObjectiveComposer 结构化**最内层**（目标函数），Constran 结构化**外层**（约束）。

---

## 13. 与小增益定理的联系

我们基于 k 的增益控制与非线性控制理论中的**小增益定理**（Small Gain Theorem）
有深刻的结构平行。本节形式化这一联系。

### 11.1 回顾：小增益定理

经典小增益定理（Zames, 1966）指出：

> 考虑两个因果系统 $H_1$, $H_2$，具有有限 $\mathcal{L}_2$ 增益
> $\gamma_1$, $\gamma_2$。若 $\gamma_1 \cdot \gamma_2 < 1$，则
> $H_1$ 与 $H_2$ 的反馈互联是有限增益 $\mathcal{L}_2$ 稳定的。

简言之：**如果每个子系统对信号的放大最多为 $\gamma_i$，且放大乘积 < 1，
整个系统就是稳定的。**

小增益定理是一个更广泛统一框架的特例——**圆盘判据**（Circle Criterion）和
**无源性定理**（Passivity Theorem）都源自同一个半内积空间框架。

### 11.2 σ_k 作为扇区有界非线性

我们的饱和函数：

$$\sigma_k(x) = \frac{kx}{\sqrt{1 + (kx)^2}}$$

是一个**无记忆、扇区有界的非线性**——正是圆盘判据和小增益定理设计来处理的那种非线性。

**扇区刻画。** 对所有 $x \neq 0$：

$$0 < \frac{\sigma_k(x)}{x} \le k$$

即 $\sigma_k$ 位于**扇区 $[0, k]$** 中——下界为 0（无源性），上界为 k（增益极限）。

**Lipschitz / 增量增益。** 导数为：

$$\sigma_k'(x) = \frac{k}{(1 + (kx)^2)^{3/2}} \le k$$

所以 $|\sigma_k(x) - \sigma_k(y)| \le k|x-y|$。**k 就是 Lipschitz 常数**——
输入差异能经历的最大放大倍数。

**饱和（绝对有界）。** 对任意 x，无论 k 多大，$|\sigma_k(x)| < 1$。
这就是"饱和"性质——不管输入多大，输出有界。用控制术语：$\sigma_k$ 具有
**有限的 $\mathcal{L}_\infty$ 增益 1**，无论其 $\mathcal{L}_2$ 增益 k 是多少。

### 11.3 k 参数就是增益

在控制理论中，系统的"增益" $\gamma$ 是输出能量与输入能量的最坏情况比：

$$\gamma = \sup_{u \neq 0} \frac{\|H(u)\|}{\|u\|}$$

对我们的 $\sigma_k$，增量增益（Lipschitz 常数）恰好是 k。
这就是为什么我们称 k 为**增益**——这不是比喻，而是数学上精确的陈述：

| k | Lipschitz 增益 | 含义 |
|---|---------------|------|
| 5.0 | 5.0 | 高增益 —— 小输入变化被放大 5 倍 |
| 1.0 | 1.0 | 单位增益 —— 非扩张 |
| 0.5 | 0.5 | 低增益 —— 收缩映射，信号衰减 |
| 0.1 | 0.1 | 极低增益 —— 强衰减 |

### 11.4 嵌套结构作为增益有界互联

我们的 `compose_objective` 结构：

$$f(x) = \sigma_{k_{\text{outer}}}\!\left(\sum_i \sigma_{k_i}\!\big(\mathcal{T}(\text{term}_i(x))\big)\right)$$

可以解释为一个**增益有界子系统的前馈互联**：

```
term₁(x) → [T] → [σ_{k₁}] ↘
term₂(x) → [T] → [σ_{k₂}] → [ Σ ] → [σ_{k_outer}] → f(x)
term₃(x) → [T] → [σ_{k₃}] ↗
```

每个通道具有：
- **对数变换** $\mathcal{T}$：压缩动态范围（类似信号处理中的对数放大器）
- **饱和** $\sigma_{k_i}$：扇区有界非线性，增益 $k_i$，输出有界于 $(-1, 1)$

**加法器** $\Sigma$ 在最坏情况下增益 $\le \sum k_i$。

**外层** $\sigma_{k_{\text{outer}}}$ 提供**全局增益有界**：无论有多少项、总和多大，
输出始终在 $(-1, 1)$。

### 11.5 为什么"就这样能工作"——一个小增益式论证

**命题：** 组合后的目标函数不会被任何单项主导。

**证明梗概（小增益风格）：**

1. 每项子通道具有**有限增益** $k_i$（Lipschitz 常数）
2. 每通道输出**绝对有界**：$|\sigma_{k_i}(\cdot)| < 1$
3. $N$ 项之和的增益 $\le \sum k_i$，但输出有界于 $(-N, N)$
4. 外层 $\sigma_{k_{\text{outer}}}$ 将 $(-N, N)$ 压缩回 $(-1, 1)$
   ——具有**有限 $\mathcal{L}_\infty$ 增益**，与 $N$ 无关
5. 因此：**有界输入（原始项值）→ 有界输出（代价）**——
   一个**输入-状态稳定（ISS）**性质

与线性权重 $w_i \cdot \text{term}_i$ 不同——后者中一个大项就能让 $f \to \infty$，
饱和嵌套保证 $f \in (-1, 1)$ 始终成立。没有任何单项能"失稳"总体代价。

### 11.6 外层 σ 作为"环路增益"控制器

在反馈控制系统中，通常加一个控制器将环路增益降到 1 以下来保证稳定性。
在我们的前馈结构中：

$$\text{外层 } \sigma \text{ 就像一个自动增益控制器（AGC）}$$

- 若所有内层项都小 → 总和小 → 外层 σ 工作在线性区（增益 ≈ k_outer）→ 良好区分度
- 若某些内层项很大 → 总和大 → 外层 σ 饱和 → 总输出有界 → 防止爆炸
- 过渡是**连续**的（无硬削波）→ 梯度得以保留

这类似于通信中的**自动增益控制（AGC）**电路，或根据信号幅值自适应放大倍数的
**增益调度控制器**。

### 11.7 更深层的联系

**收缩映射。** 当 $k \le 1$ 时，每个 $\sigma_k$ 是 Banach 意义上的**收缩**：

$$|\sigma_k(x) - \sigma_k(y)| \le k|x-y| < |x-y|$$

收缩的复合仍是收缩。这联系到 **Banach 不动点定理**——迭代应用收敛到唯一不动点。
在我们的语境中：原始项值的微小变化不会导致代价的过大变化。

**圆盘判据（Circle Criterion）。** 对于 Lur'e 系统（线性动力学 + 扇区有界非线性反馈），
圆盘判据保证：若线性部分的 Nyquist 图避开由扇区边界确定的某个圆盘，则系统稳定。
我们的饱和 $\sigma_k \in [0, k]$ 是标准示例——扇区边界精确告诉我们非线性有多"激进"。

**无源性（Passivity）。** $\sigma_k$ 是**无源**的：对所有 $x$，$\sigma_k(x) \cdot x \ge 0$
（它是奇函数且 $\sigma_k(x)/x > 0$）。在互联无源系统中，稳定性由无源性定理保证——
与小增益定理的又一统一。

**ISS（输入-状态稳定）。** 整个复合满足：

$$\|f\|_\infty \le 1 \quad \text{无论输入幅值多大}$$

这是 ISS 的最强形式——输出**全局一致最终有界**，界为 1。k 的任何调参都不能改变这一保证。

### 11.8 实践含义：通过增益推理选择 k

控制论的视角给出了选择 k 的原则性方法：

| 控制概念 | 我们的等价 | 对 k 的含义 |
|----------|-----------|------------|
| 环路增益 | k（Lipschitz 常数）| k < 1：收缩；k > 1：放大 |
| 增益裕度 | k 在不稳定前能增大多少 | 外层 σ 提供无限增益裕度 |
| 扇区边界 | σ_k ∈ [0, k] | k 定义了线性工作区 |
| 饱和极限 | \|σ_k\| < 1 | 始终安全，与 k 无关 |
| 带宽 | 1/k（knee 位置）| 小 k = 宽带宽，大 k = 窄带宽 |
| 增益调度 | 每项不同 k | 每个子目标有独立的"通道增益" |

**核心洞察：** 你选的不是无单位的权重——你在**为一个非线性控制通道设计增益**，
其稳定性由饱和结构数学保证。外层 σ 是"安全网"，无论你加多少项、k 取多大，
总增益始终有界。

### 11.9 参考文献

- Zames, G. (1966). "On the input-output stability of time-varying nonlinear
  feedback systems." *IEEE Trans. Automatic Control*, 11(2–3).
- "A nonlinear small gain theorem for the analysis of control systems with
  saturation." *IEEE Trans. Automatic Control*, 1996.
- "Unified Necessary and Sufficient Conditions for the Robust Stability of
  Interconnected Sector-Bounded Systems." arXiv:1809.08742.
- Sontag, E.D. (2008). "Input to State Stability: Basic Concepts and Results."
  In *Nonlinear and Optimal Control Theory*, Springer.
- Khalil, H.K. (2002). *Nonlinear Systems*, 3rd ed. Prentice Hall.
  (第 5 章: 输入-输出稳定性, 第 10 章: 无源性, 圆盘判据)

---

*"你选的不是权重，是非线性控制通道的增益。"*
