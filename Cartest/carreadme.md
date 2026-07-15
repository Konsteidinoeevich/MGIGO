# Cartest — Frenet B-Spline Trajectory MPC

基于 Frenet 坐标系 + 五次 B 样条的 MPC 轨迹规划器。
IGO 黑箱优化 + Constran 约束引擎。

## 总体框架

```
  ┌─────────────┐    ┌───────────┐
  │ ReferencePath│    │  Scenario │   地图 & 场景
  │  道路几何    │    │ 障碍物+参数│
  └──────┬──────┘    └─────┬─────┘
         │                 │
         ▼                 ▼
  ┌──────────────────────────────────────────────┐
  │ frenet_traj                                   │
  │  to_vehicle_states    (Frenet → 车辆运动学)    │  正反变换
  │  from_vehicle_states  (车辆运动学 → Frenet)    │
  │  make_frenet_reference(maneuver → z_ref)      │
  └──────────────────────┬───────────────────────┘
         │               │
         ▼               ▼
  ┌─────────────┐   ┌──────────────────────────┐
  │ cost.py     │   │ build_context + warmstart │  规划准备
  │ Lyapunov 2阶│   │ + make_constraints        │
  └──────┬──────┘   └────────────┬─────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         └──────────────│  build_solver() │  Constran + IGO 一站
                        └────────┬────────┘
                                 │
                                 ▼  result.x → ctrl_s, ctrl_d
                        ┌──────────────────┐
                        │ execute_perfect  │  plan 的 t=1 状态
                        │ _tracking        │  直接作为下一步
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ reporting/plot   │
                        │ diagnose/eval    │
                        └──────────────────┘
```

> **注意**: 上图是当前已实现的**单阶段**管线。两阶段优化架构（Phase 1 探索 + Phase 2 精炼, 以及正反变换在其中的角色）的设计文档见 **[two_phase_design.md](two_phase_design.md)**。两阶段代码尚未实现。

**核心管线**：
- **正变换** `to_vehicle_states`: Frenet (s,d) → 车辆状态 (x,y,v,ψ,a_long,a_lat,…)，含曲率耦合
- **反变换** `from_vehicle_states`: 车辆状态 → Frenet，用于从地图/外部参考反解 `z_ref`
- **参考生成** `make_frenet_reference`: maneuver 描述 → `z_ref`，供 Lyapunov cost 跟踪
- **执行** `execute_perfect_tracking`: 假设完美跟踪，直接用 plan 预测的下一状态（评估开环 plan 质量）

## 文件结构

```
Cartest/
├── Simple.py                    # MPC demo 主程序
├── carreadme.md
├── basis/                       # 离线预计算
│   ├── spline.py                # → bspline_basis.npz
│   └── bspline_basis.npz        # (10控点, 5次, 10s时域)
├── core/                        # 核心变换 + 车辆模型
│   ├── frenet_traj.py           # evaluate, to_vehicle_states,
│   │                            # from_vehicle_states, make_frenet_reference
│   ├── reference_path.py        # StraightReference, CircularReference,
│   │                            # frenet↔cartesian
│   └── vehicle_model.py         # PointMassModel (摩擦圆积分)
├── planning/                    # MPC 规划
│   ├── cost.py                  # 2阶耦合 Lyapunov + build_context
│   ├── constraints.py           # 约束构建 (obs/lane/speed/acc/jerk)
│   ├── warmstart.py             # build_initial_mu
│   └── scenario.py              # 场景配置
├── execution/                   # 执行
│   └── execute.py               # execute_perfect_tracking / execute_point_mass
└── eval/                        # 评估 + 测试
    ├── diagnostics.py           # raw obj, g 值诊断
    ├── reporting.py             # StepReport 记录
    ├── plotting.py              # 可视化
    ├── eval_closed_loop.py      # 闭环评估 (收敛/超调/震荡/约束)
    └── test_frenet_invert.py    # 16 个测试
```

## 1. 地图 — ReferencePath

参考线 = 弧长参数化的光滑中心线。实现：

```python
evaluate(s)              → (x_r, y_r, θ_r, κ_r)      # 路径几何
frenet_to_cartesian(s,d) → (x, y)                    # Frenet → Cartesian
cartesian_to_frenet(x,y) → (s, d)                    # Cartesian → Frenet (反解)
```

内置 `StraightReference`（直路，`s=x, d=y` 平凡反解）。测试用 `CircularReference`（圆弧，`s=R·atan2(x,R-y)`, `d=R−√(x²+(R−y)²)` 闭式反解）。

自定义弯道继承 `ReferencePath` 并实现 `evaluate` 和 `cartesian_to_frenet`（可用 1D Newton 迭代，不需要 SQP）。

## 2. 场景 — Scenario

`scenario.py` 是所有场景参数的唯一来源。切换场景只需改一行 import：

```python
from Cartest.scenario import THREE_BLOCKING as scenario
```

每个场景是一个 dict：

```python
SCENE = {
    "obstacles": [
        {"x": 45.0, "y": -2.5, "r": 2.0},
        {"x": 65.0, "y":  0.5, "r": 2.0},
    ],
    "lane_hw":       2.0,      # 半车道宽度 (m)
    "obs_safe_dist": 0.1,      # RSS 反应时间 (s)
    "v_target":     18.0,      # 目标速度 (m/s)
    "init": {                  # 初始车辆状态
        "s": 0.0, "s_dot": 12.0, "s_ddot": 0.0,
        "d": -3.0, "d_dot":  0.0, "d_ddot": 0.0,
        "psi": 0.0,
    },
}
```

## 3. 初始状态 — FrenetState

`execute.py` 定义了 `FrenetState` 数据类，含 `to_ctx()` 方法：

```python
@dataclass
class FrenetState:
    s:      float   # 纵向位置 (m)
    s_dot:  float   # 纵向速度 (m/s)
    s_ddot: float   # 纵向加速度 (m/s²)
    d:      float   # 横向偏移 (m)
    d_dot:  float   # 横向速度 (m/s)
    d_ddot: float   # 横向加速度 (m/s²)
    psi:    float   # 航向角 (rad)
```

## 4. B 样条轨迹 & Cost

5 次 B 样条，10 控制点，10 秒时域，100 采样点。

```
P0, P1  夹紧: C0 (位置) + C1 (速度)
P2..P9  自由: 8 控制点/通道 × 2 = 16 维优化变量
```

**C2 (加速度) 不夹紧** — 实验表明在当前 jer k约束 (|j|≤2.0) 和 0.1s 执行步长下，
C2 夹紧锁死初始横向加速度，导致轨迹无法在合理时间内收敛。增加控制点数量
或添加三阶 cost 项均无帮助——问题是物理性的，不是优化性的。

### 耦合 Lyapunov 代价 (2阶)

s/d 两通道对称追踪位置误差，K 矩阵配置收敛速率：

```
e = [es, ed]    es = s − s_ref(t)    ed = d
s_ref(t) = s0 + v_target·t + (v0−v_target)/ω_s · (1−e^(−ω_s·t))

cost = Σ eᵀe + Σ (ė + K e)ᵀ(ė + K e) + Σ (ë + 2K ė + K² e)ᵀ(ë + 2K ė + K² e)

K = [[ω_s, 0], [0, ω_d]]   — α=0 解耦，各通道独立
```

收敛行为见下方 [Constructive Lyapunov 原理](#constructive-lyapunov-原理)。

### 参考轨迹生成

`make_frenet_reference(gen, ctx, maneuver)` 从高层描述生成 `z_ref`:

```python
# 变道
ref = make_frenet_reference(gen, ctx, {
    'type': 'lane_change', 'd_end': 3.5,
    't_start': 0.5, 't_duration': 3.0, 'v_desired': 20.0,
})
# 巡航
ref = make_frenet_reference(gen, ctx, {'type': 'cruise', 'v_desired': 25.0})
# 外部参考 (地图/其他planner)
ref = make_frenet_reference(gen, ctx, {
    'type': 'external', 'vehicle_states': y_ref,  # [T, 9]
})
```

所有模式统一走 `vehicle-level y_ref → from_vehicle_states → z_ref` 管道，
保证速度分解 `v² = (1−d·κ_r)²·s_dot² + d_dot²` 对直路和弯道都正确。

### Constructive Lyapunov 原理

当前 cost 是二阶 **Constructive Lyapunov Function (CLF)**。
"Constructive" 的含义是从低阶到高阶逐层构造，每层引入更高阶导数作为
"虚拟控制输入"，最终形成完整的 Lyapunov 函数。

**构造层次：**

```
层0 (位置):     V₀ = ||e||²                           ← 纯几何误差
层1 (速度):     V₁ = V₀ + ||ė + K·e||²                ← ė 作为"虚拟控制"驱动 e→0
层2 (加速度):   V₂ = V₁ + ||ë + 2K·ė + K²·e||²       ← ë 作为"虚拟控制"驱动 ė→−K·e
层3 (jerk, 可选): V₃ = …                              ← 需要 C2 夹紧 + 三阶 cost
```

每一层引入的"虚拟控制" `v_k = e^(k) + k·K·e^(k−1) + … + K^k·e`
把上一层的收敛速率绑定到 K 的特征值。

**为什么 α=0 解耦：** `K = [[ω_s, α], [α, ω_d]]`。α>0 时 s 和 d
通道互相耦合——纵向速度误差会影响横向 cost，优化器被迫在两个目标间
折中。实际测试显示 α=0.5 时 v 终值只能到目标的 80%。

**收敛速率由 ω_s, ω_d 预设，但受 B-spline + jerk 约束限制：**
- 横向: ~3s 收敛下限 (ω_d ≥ 4 后不再加速)
- 纵向: ~5s 收敛下限 (比横向慢，是瓶颈)

**为什么不夹紧 C2：** C2 夹紧强制 `d_ddot[0] = 0`。在当前 jerk 约束
(|j|≤2.0) 和 0.1s 执行步长下，每个 MPC 步只能产生 ~0.001m 的横向位移，
收敛时间超出合理范围。增加控制点数或加三阶 cost 项均无帮助——
问题是物理性的：从零加速度起步需要 jerk 缓慢爬升。

### 为什么需要正反变换 & 合理参考

本项目的核心是跟踪 Frenet 状态 `z = (s, d, s_dot, d_dot, s_ddot, d_ddot, …)`。
但参考信号来自外部——地图、场景、或 maneuver 描述——这些通常不在 Frenet 空间。

**正变换 `to_vehicle_states`** 把优化器产出的 Frenet 轨迹映射到物理车辆状态，
供约束检查（速度/加速度/jerk 必须满足物理极限）和诊断使用。
曲率耦合项（`(1−d·κ_r)` Jacobian、离心、Coriolis）保证在弯道上也是对的。

**反变换 `from_vehicle_states`** 把外部参考（GPS waypoints、地图 lane center、
上层 planner 输出）从车辆/Cartesian 空间转回 Frenet 空间，生成 `z_ref`。

**为什么不能直接在 Frenet 空间写 `s_dot_ref = v_ref`：**

车辆运动学的基本关系是：

```
v² = (1 − d·κ_r)² · s_dot² + d_dot²
```

如果直接写 `s_dot_ref = v_ref` 同时 `d_dot_ref ≠ 0`（变道有横向速度），
实际车速 `v_actual = √(v_ref² + d_dot_ref²) > v_ref`——参考本身就违反
物理约束。弯道上还有 `(1−d·κ_r)` 的修正。

**正确的管道：**

```
maneuver (如 "d → 3.5m, v → 20m/s")
    │
    ▼
构建 vehicle-level 参考 y_ref(t) = (x,y,v,ψ,a_long,a_lat,…)
    │  考虑路径几何 θ_r(s), κ_r(s)
    │  保证 v² = (1−d·κ_r)²·s_dot² + d_dot²
    ▼
from_vehicle_states(y_ref) → z_ref = (s_ref, s_dot_ref, …, d_ref, d_dot_ref, …)
    │
    ▼
Lyapunov cost 跟踪 z_ref
```

这个管道保证 `z_ref` 在几何上可行——无论直路弯道，`v_ref`, `d_ref`, `s_dot_ref` 三者始终满足运动学关系。

### 误差空间线性变换 — 无损耦合方案

当前 framework 中 α>0 会导致两通道相互拖累（见 [Constructive Lyapunov 原理](#constructive-lyapunov-原理)），
因为 K 矩阵的交叉项压低了最小特征值。下面给出一种替代方案：**在误差定义层做线性变换，
Lyapunov 层保持解耦**——耦合通过误差混合实现，不通过 K 矩阵实现。

#### 变换定义

设原始误差向量（各阶同理）：

```
e = [e_s, e_d]ᵀ   其中 e_s = s − s_ref,  e_d = d − d_ref
```

定义可逆线性变换 `T ∈ GL(2)`：

```
T = [[1,  α],
     [β,  1]],     det(T) = 1 − αβ > 0   ⇔   αβ < 1
```

变换后的误差：

```
[z̃]   [1  α] [e_s]     z̃ = e_s + α·e_d        ← 纵向误差混入横向
[w̃] = [β  1] [e_d]     w̃ = β·e_s + e_d        ← 横向误差混入纵向
```

原始误差可通过逆变换恢复：

```
e_s = (z̃ − α·w̃) / (1 − αβ)
e_d = (w̃ − β·z̃) / (1 − αβ)
```

`z̃→0, w̃→0 ⇒ e_s→0, e_d→0`——收敛性不丢失。

#### 与当前 K 矩阵耦合的本质区别

```
当前方案:    误差空间不耦合 → 在 Lyapunov 层用 K 的非对角元耦合
            → α 压低 min(λ_K) → 两边收敛都变慢

变换方案:    在误差定义层做线性混合 → Lyapunov 层用 K̃ = diag(ω_z, ω_w) 解耦
            → 每个虚拟控制只追自己方向, 不打架
```

在 `(z̃, w̃)` 空间里，constructive Lyapunov 各层完全解耦：

```
层0:  V₀ = z̃² + w̃²
层1:  V₁ = V₀ + (z̃̇ + ω_z·z̃)² + (w̃̇ + ω_w·w̃)²
层2:  V₂ = V₁ + (z̃̈ + 2ω_z·z̃̇ + ω_z²·z̃)² + (w̃̈ + 2ω_w·w̃̇ + ω_w²·w̃)²
```

**关键：** 层 1 的 Lyapunov 条件 `z̃̇ + ω_z·z̃` 展开：

```
z̃̇ + ω_z·z̃ = (ė_s + ω_z·e_s) + α·(ė_d + ω_z·e_d)
```

同一个 `ω_z` 同时作用在 s 和 d 的虚拟控制上——纵向的收敛速率被"借"给横向用，
但因为是同一个标量特征值，**纵向前进不减速**。

#### 特性

| 特性 | 当前 K 耦合 (α≠0) | 误差变换 (T 耦合) |
|------|-------------------|-------------------|
| λ_min 退化 | 是 (被压低) | 否 (仅取决于 min(ω_z, ω_w)) |
| 一通道牺牲加速另一通道 | 不会 (只共同变慢) | 可做到 (选 α·β<0) |
| 收敛保证 | K≻0 ⇒ 保证 | det(T)>0 ⇒ 保证 |
| Lyapunov 层复杂度 | K 有交叉项 | K̃=diag, 无交叉项 |

#### 超调映射

变换后误差 `z̃, w̃` 不超调不保证原始误差 `e_s, e_d` 不超调。反变换：

```
e_d = (w̃ − β·z̃) / (1−αβ)
e_s = (z̃ − α·w̃) / (1−αβ)
```

`e_d` 的超调条件：当 `β·ω_z·z̃₀ > ω_w·w̃₀` 时 `ė_d(0) > 0`，
即被帮助的通道在初始阶段不降反升。**超调 ≈ 正比于 `|β|`**，约每 0.01 的 β 贡献 ~0.08m 额外超调。

`ω_z ≈ ω_w` 时超调最小（两个通道衰减速率接近，`z̃` 不会长时间拖累 `e_d`）。

### 跨阶广义耦合（前瞻）

前述 `T ∈ GL(2)` 仅在同阶（位置与位置）间耦合。更一般的构造：
将两通道的误差向量展开到所需阶数（如位置、速度、加速度），
用 **分块矩阵** 做跨阶混合。

#### 定义

设两通道的误差状态向量（以 3 阶为例）：

```
𝐞_A = [e_s,  ė_s,  ë_s ]ᵀ     ← 纵向 (位置, 速度, 加速度)
𝐞_B = [e_d,  ė_d,  ë_d ]ᵀ     ← 横向
```

定义 6×6 分块变换：

```
[𝐳_A]   [ I      C_{B→A} ] [𝐞_A]
[𝐳_B] = [C_{A→B}   I      ] [𝐞_B]

其中 C_{B→A}, C_{A→B} ∈ ℝ^(3×3) 是跨阶耦合矩阵
```

展开形式（以 `C_{B→A}` 为例）：

```
C_{B→A} = [c^pp_ba  c^pv_ba  c^pa_ba]   ← 横向位置/速度/加速度 → 纵向各阶
           [c^vp_ba  c^vv_ba  c^va_ba]
           [c^ap_ba  c^av_ba  c^aa_ba]
```

例如 `c^pv_ba ≠ 0` 的含义：**横向速度误差 `ė_d` 混入纵向位置误差 `e_s`**。
这在物理上对应——变道时 `d_dot` 峰值出现，纵向通道"预知"横向正在运动，提前调整速度。

#### 稳定性条件

分块矩阵的逆存在条件（类比 `det(T) = 1−αβ > 0`）：

```
det( I − C_{B→A}·C_{A→B} ) ≠ 0
```

等价于 1 不是 `C_{B→A}·C_{A→B}` 的特征值。满足时 `𝐳_A, 𝐳_B → 0 ⇔ 𝐞_A, 𝐞_B → 0`，
收敛性不丢失。

在 `(𝐳_A, 𝐳_B)` 空间中，constructive Lyapunov 完全解耦：

```
V = ||𝐳_A||²_{Lya} + ||𝐳_B||²_{Lya}
```

其中 `||·||_{Lya}` 是各通道内部的 3 阶 Lyapunov 范数（用 `K_A, K_B` 对角阵）。

#### 超调分析

跨阶耦合引入了新的超调路径。以 `c^pv_ba`（横向速度→纵向位置）为例：

```
z_A,0 = e_s + c^pv_ba · ė_d  (+ 其他项)
```

变道过程中 `ė_d` 先正后负（先加速横移、后减速回正）。
当 `ė_d > 0` 时 `z_A,0` 被抬高，Lyapunov 条件迫使纵向"更用力"收敛；
当 `ė_d < 0` 时 `z_A,0` 被压低，可能产生反向超调。

**纵向的超调来源于横向速度的符号翻转被"注入"到纵向误差中。**
可以通过限制 `C_{B→A}` 的非对角元符号来控制——例如只用 `c^pp_ba`（位置→位置）
和 `c^vp_ba`（位置→速度），避免导数项（速度、加速度）的符号翻转污染。

#### 退化关系

| 耦合类型 | `C_{B→A}` | `C_{A→B}` | 等价于 |
|---------|-----------|-----------|--------|
| 无耦合 | 0 | 0 | 当前 `α=β=0` |
| 同阶位置耦合 | `diag(α,α,α)` | `diag(β,β,β)` | 前述 `T ∈ GL(2)` |
| 同阶全耦合 | `α·I` | `β·I` | 前述 `T ∈ GL(2)` 在各阶复制 |
| 跨阶 | 非对角 | 非对角 | **待实现** |

当前所有 cost（`cost.py`, `cost_transform.py`）都是 `C_{B→A} = diag(α,α,α)`, `C_{A→B} = diag(β,β,β)` 的退化版。

#### 实验验证

参数扫描（场景: d=3→0, v=12→20，横向+纵向同时有误差）：

```
ω_z ω_w  α     β        d收敛    v终值    d超调
────────────────────────────────────────────────────
1.0 4.0  0     0        4.1s    15.69   −0.06   ← T=I 基线
1.0 4.0  0.10 −0.10     3.3s    14.70   −0.75   ← 快 20%, 超调 0.75m
1.0 4.0  0.15 −0.15     3.3s    14.23   −1.15   ← 过度耦合
3.0 4.0  0.10 −0.10     7.8s    17.31   −0.15   ← ω_z≈ω_w, 超调小但慢
2.0 3.0  0.15 −0.15     4.9s    16.27   −0.69   ← 折中
```

**规律：**
- `|β|` 每增加 0.01，d 超调增加 ~0.08m，收敛加速 ~0.08s
- `ω_z ≈ ω_w` 时超调最小，因为两个通道衰减同步，不互相拖拽
- v 终值下降 ≈ `|β|·Δv_target`——纵向"支付"了横向加速的代价

#### 使用策略

超调是买收敛速度的代价，`|β|` 是计价单位。按场景选择：

| 场景 | α, β | 理由 |
|------|------|------|
| 正常巡航 | α=β=0 (退化) | 零超调，精准跟踪 lane center |
| 变道 | `|β| = 0.05~0.10` | 用 ~0.4m 超调换 20% 收敛加速 |
| 紧急绕行 | `|β| = 0.10~0.15` | 接受更大超调，优先快速偏离 |

#### 实现

`planning/cost_transform.py` — `make_objective_transform(gen, ω_z, ω_w, α_t, β_t)`。
α_t=β_t=0 时退化为现有 `make_objective` 的 `α=0` 版本，可直接替换使用。

## 5. 执行

两种模式，`execute.py` 中均有：

| 函数 | 用途 | 说明 |
|------|------|------|
| `execute_perfect_tracking` | **默认** | 直接用 plan 的 t=1 状态，假设底层控制器能精确跟踪 |
| `execute_point_mass` | 遗留 | Frenet 欧拉积分 + 摩擦圆，仅 κ_r=0 时正确 |

默认使用 `execute_perfect_tracking`——本项目评估的是开环 plan 质量
（跟踪/超调/震荡/约束满足），控制器的跟踪精度留给后续工作。

## 6. IGO 优化器

`build_solver()` 一站：Constran 约束组装 + solver 选择 + 参数初始化。

```python
solver = build_solver(obj_fn, dims=(gen.n_free, gen.n_free),
    constraints=make_constraints(gen, lane_hw, safe_dist),
    solver='m22', T=300, dt=0.3, K=3, B=64, B0=30, T_0=300,
    k_inner=1.0, obj_transform='standard',
)

result = solver(key, context=ctx, initial_mu=mu_init)
ctrl_s, ctrl_d = result.x[:gen.n_free], result.x[gen.n_free:]
```

支持 GMM 状态继承：`solver(key, context=ctx, warm_start=prev_result)`。

### 两阶段分步优化

> **状态**: 以下为设计方案，**代码尚未实现**。详细设计、实现计划与验证路径见 **[two_phase_design.md](two_phase_design.md)**。

完整规划问题（全局路径 + 精细跟踪 + 物理约束）很难在单个 IGO 中一次求解。
方案是用**同一套 B-spline 基**拆成两个阶段，通过 `solver_modes` 切换求解器配置。

#### 架构

```
地图 waypoints (Cartesian)
  │
  cartesian_to_frenet(x_i, y_i) → (s_i, d_i)
  │  最小二乘拟合 Greville → ctrl_init (粗 warmstart)
  │
  ▼
Phase 1: 探索 (coarse, Cartesian)
  粗 IGO:  K=3, dt=0.15, T_0 小 (定期重置)
          松约束 (ACC=10, JERK=5)
          简单 cost (几何可行 — 不碰障碍, 在车道内)
  输出:   y_ref [T×9] — 行为级意图
          (x, y, v, ψ, a_long, a_lat, j_long, j_lat, steer)
  │
  │  小 dt → GMM 移动慢 → 多模态保持 (左绕/右绕)
  │  决断: cost gap 超过阈值 → 胜出模态自然 dominate
  │
  ▼
from_vehicle_states(y_ref) → z_ref
  │  直路: s=x, d=y, s_dot=v, d_dot=0
  │  弯道: 自动处理 (1−d·κ_r) Jacobian, 离心/Coriolis 解耦
  │  z_ref = {s_ref, s_dot_ref, s_ddot_ref, s_dddot_ref,
  │           d_ref, d_dot_ref, d_ddot_ref, d_dddot_ref}
  │
  ▼
Phase 2: 精炼 (fine, Frenet)
  细 MPC:  K=3, dt=0.30, T_0 大 (不重置)
          紧约束 (ACC=5, JERK=2)
          Lyapunov cost 跟踪 z_ref
  输入:   Phase 1 的 z_ref 作为 cost 参考
  输出:   B-spline ctrl → 最终执行轨迹
  │
  │  大 dt → 分布快速收敛到局部最优
  │  T_0 大/不重置 → 锁死在 Phase 1 选定的绕行方向
```

#### MPC 步内流程

两个 Phase 在同一 MPC 步内顺序执行——不是两个独立进程：

```
每个 MPC 步 (0.1s):

  ctx = build_context(state, ...)
  mu  = warmstart(state)
  │
  ├─ Phase 1: dt=0.15, K=3, 松约束, 同 B-spline 基
  │     cost: 几何可行 + 闭环加速项
  │     输出: z_ref (直接由 B-spline evaluate 得到, Frenet)
  │
  ├─ Phase 2: dt=0.30, K=3, 紧约束, mode=mode_label
  │     cost: Lyapunov 跟踪 z_ref (纯跟踪)
  │     ctrl_coarse 直接 warmstart ctrl_fine (同基)
  │
  └─ execute(ctrl_fine) → next state
```

#### 约束嵌套方向：两 Phase 相反

Constran 的优先规则: **外层（priority 数字大）拥有最终话语权**。
外层的违规信号直接进入最终 cost，不受内层掩盖。

```
Phase 1 (探索):  obs (外, priority 最大) → lane → speed → acc → jerk (内)
                obs 在最外层 — 安全是最终防线，拥有最高话语权
                内层 jerk/acc 松 (ACC=8~10, JERK=5~8)，不限制探索

Phase 2 (精炼):  jerk (外, priority 最大) → acc → speed → lane → obs (内)
                jerk/acc 在最外层 — 物理可行性是最终防线
                obs 在内层 — z_ref 已解决几何，obs 仅作内层后备
```

两个约束构建独立，由 `solver_modes` 在 build 时各自编译好，运行时零开销切换。

#### Phase 1 实验验证

**warmstart 设计：** P1 的 warmstart 必须来自地图的全长度车道数据，
不能是 ramp 外推。每个模态（左/中/右车道）对应一组自由控点，
直接设为该车道的常数 `d_lane` 和 `s0 + v0·greville`。
B-spline 的 C0/C1 夹紧自动处理从当前 `d0` 到目标车道的过渡。

```python
# 正确：全车道 warmstart (地图给整条 lane)
ctrl_d = [d_lane, d_lane, ..., d_lane]  # n_free 个副本

# 错误：ramp 外推 (手工构造的渐变)
ctrl_d = d_start + (d_lane - d_start) * greville / total_time
```

**实验结果：**

| 配置 | 模态 | 发散 | 说明 |
|------|------|------|------|
| ramp warmstart | 单一 | 有 | 数值爆炸，cost 无法区分 |
| 全车道 warmstart, T=300, B=128 | {中, 右} | 无 | 双模态，零发散 |
| 全车道 warmstart, T=500, B=256 | {左, 中, 右} | 1/5 | 三模态全有 |

**关键认知：**

1. **Cost 使用 Constran 自动构建，不手工拼权重。** P1 的目标函数只保留
   最轻的速度引导（`sum(s_dot − v_target)²`），所有几何可行性
   （障碍物、车道边界）由 Constran 的 σ 嵌套约束自动处理。
   手工加权和 → 7 位有效数字不够用 → 数值发散。

2. **多模态来自地图 warmstart，不来自约束。** K=3 的 GMM 各分量
   初始化为地图提供的不同车道，每个分量自发探索该车道附近的几何空间。
   IGO 自然淘汰不好的分量。如果地图只给一条 lane → 单模态。

3. **障碍物约束生效需要足够的 IGO 轮次 + 样本。**
   T=300 B=128 以上才能让 P1 在约束边界附近找到可行轨迹。
   约束嵌套方向（obs 在最外层，拥有最终话语权）在探索阶段是正确的——先问"能过吗"，
   再在可行空间内搜。

#### 时序预算

两阶段 MPC 步的实测耗时（预编译，无 JIT 重编译）：

```
P1 (T=200 B=96  dt=0.15) + P2 (T=150 B=64 dt=0.30) = 544ms
P1 (T=300 B=128 dt=0.15) + P2 (T=150 B=64 dt=0.30) = 650ms
```

544ms 在 600ms 预算内，收敛时间与单阶段基线持平（3.4s vs 3.3s）。

#### 冲突时间 → 收敛速率 + 模态选择

Phase 1 每步重新评估，输出 `(z_ref, t_conflict, mode_label)`：
- `t_conflict` → Phase 2 选 `ω_d ≥ 5/(t_conflict − t_now)`，冲突前收敛
- `mode_label ∈ {正常, 变道, 让行, 紧急}` → Phase 2 选对应 solver mode

Phase 1 的 GMM (K=3) 在多模态间自然探索，cost 低的 dominate，
不良分量自动 → 0。Phase 2 跟随选定模态，同 K=3。

#### 模态切换不发散

每步 Phase 1 重新评估——环境变化自动切换模态。
同一 MPC 步内 Phase 2 从当前车辆状态直接跟踪新 `z_ref`，
Lyapunov cost 产生收敛力。B-spline C1 夹紧 + jerk 约束保证轨迹连续，
无需显式平滑过渡。若新 `z_ref` 几何不可达，Phase 2 自然牺牲跟踪精度保约束。

#### Phase 1 cost 设计：ref 作为闭环加速工具

Phase 1 和 Phase 2 共享同一套 Frenet B-spline 基。Phase 1 的 cost
不是"产出光滑可行参考"——Phase 2 负责光滑和可行性。Phase 1 的 cost 是
**"产出能驱动 Phase 2 闭环执行最快的 z_ref"**。

**原理：** Phase 2 是 Lyapunov 跟踪器——给定 `z_ref`，它的闭环收敛行为
（收敛时间、超调量）由 `z_ref` 的形状决定。Phase 1 可以利用这一点，
在 `z_ref` 的某些段故意设超调/欠调，让 Phase 2 的闭环执行提前到达目标。

```
标准 ref:    d_ref 单调 S 形 → Phase 2 闭环 d(t) 慢
激进 ref:    d_ref 在启动段故意越过目标 → Phase 2 追的过程中
            实际 d(t) 在更早时刻穿过目标线
```

**Phase 1 的 cost 结构：** 不直接最小化 `z_ref` 的 jerk 或光滑度，
而是**模拟 Phase 2 的闭环响应**，最小化"Phase 2 执行 `z_ref` 后的
实际轨迹到达目标的时间"：

```
cost_P1(z_ref) = cost_geo(z_ref)                          ← 几何可行
               + w_time · T_arrival(z_ref, model_P2)      ← 闭环到达时间
               + w_overshoot · penalty(d_actual 越界)      ← 安全边界
```

其中 `model_P2` 是 Phase 2 Lyapunov 跟踪动力学的简化模型（可解析：
`d_actual ≈ d_ref 经一阶滤波器 ω_d`）。Phase 1 搜 `z_ref` 时，
闭着眼睛知道 Phase 2 会怎么跟——然后设计出"跟出来最快"的形状。

**为什么误差耦合不够——ref 才是瓶颈：** 下午实验表明纯改耦合矩阵
（T 变换、跨阶 C 矩阵）效果有限。根本原因是 Phase 2 的 Lyapunov
跟踪器被 `z_ref` 的形状锁死了动态——耦合只能在给定 `z_ref` 下调配
冗余，不能改变跟踪包络。要加速闭环，必须从 `z_ref` 本身下手。

**求解器兼容性：** Phase 1 仍用 IGO 系求解器。后续将换为
`MPC_G_MS.py`（博弈 mixed-strategy Nash equilibrium 版本），
但优化目标不变——都是搜 `z_ref` 最小化闭环到达时间 + 几何代价。
Phase 1 的 cost 定义与求解器实现解耦。

#### cost 去硬编码

`z_ref` 全部来自 Phase 1 管道。当前 `cost.py` 内部硬编码的指数速度
和 `d_ref=0` 可替换为纯跟踪——接受外部 8 个参考量。

#### solver_modes 支持

`planning/solver_modes.py` 预编译 5 个模式。Phase 1 用 `active`/`aggressive`
(松约束、探索)，Phase 2 用 `standard` (紧约束、跟踪)。同一步内通过
`SolverModes.solve(name, …)` 切换。

## 7. 运行 & 测试

```bash
# 生成基函数矩阵 (只需一次)
uv run python Cartest/basis/spline.py

# MPC demo (首次运行自动 JIT 预热)
uv run python Cartest/Simple.py --steps 150 --seed 0
uv run python Cartest/Simple.py --steps 50 --no-plot

# 测试 (16个)
uv run python Cartest/eval/test_frenet_invert.py

# 闭环评估
uv run python Cartest/eval/eval_closed_loop.py --steps 150
```

输出 `Cartest/frenet_demo.gif`。

## 8. 关键设计决策

| 决策 | 理由 |
|------|------|
| Frenet 坐标 | 参考线承担曲率，B 样条 ctrl→物理量线性 |
| `to_vehicle_states` + `from_vehicle_states` 成对 | 正反变换同源，round-trip 自洽 |
| `cartesian_to_frenet` | 支持外部地图/参考反解，1D Newton 即可（不需 SQP） |
| `make_frenet_reference` 统一走 vehicle→Frenet | 保证速度分解考虑 κ_r 和 d_dot，直路弯道一致 |
| C0+C1 夹紧，C2 自由 | C2 夹紧 + jerk 约束 = 初始响应锁死，无法收敛 |
| `execute_perfect_tracking` | 评估开环 plan 质量，控制器跟踪留后 |
| Scenario 单文件 | 切场景只改一行 import |
| `FrenetState` dataclass | 类型安全，`.to_ctx()` 自动 |
| `build_solver()` | Constran + IGO + warm-start 一站 |
| 耦合 Lyapunov cost, α=0 | 解耦，各通道独立最速；α>0 破坏收敛 |
| 自相似 σ 嵌套约束 | jerk/acc/speed/lane/obs，因果积分链 |
| RSS 障碍物约束 | 横纵各自判定取 max |
| 诊断分离 | `cur_obs`=车辆真实距离, `g_max`=规划层约束压力 |

## 已知近似

`to_vehicle_states` 中：
- κ_r' = 0（忽略曲率沿弧长导数）— 直路和圆弧精确，回旋线有误差
- 简化 jerk 旋转（忽略离心 jerk 耦合项 `2·κ_r·v·a_long`）
- 运动学自行车转向模型（忽略轮胎侧偏）

`_build_vehicle_reference` 中：
- κ_r 在 s0 处取一次，全时域复用 — 常数曲率路径精确，变曲率路径有误差
- 由 `from_vehicle_states` 在末端纠正残差

## 依赖

- `jax` — 矩阵运算
- `numpy`, `scipy` — 基函数预计算
- `matplotlib` — 可视化
- `gmm_igo` — IGO 优化器
- `Constraintdealer.Constran` — σ 嵌套约束引擎
