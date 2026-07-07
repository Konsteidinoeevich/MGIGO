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
