# Cartest — Frenet B-Spline Trajectory MPC

基于 Frenet 坐标系 + 五次 B 样条的 MPC 轨迹规划器。IGO 黑箱优化 + Constran 自相似 σ 嵌套约束。

## 架构

```
                          ┌─────────────────────┐
                          │    ReferencePath     │  地图输入
                          │  evaluate(s) → x,y,θ,κ │
                          │  frenet→cartesian    │
                          └──────────┬──────────┘
                                     │
  ┌──────────┐    ┌──────────┐       ▼
  │  spline  │───▶│ frenet   │───▶ [s,d,ḃ,ḋ,s̈,d̈,s⃛,d⃛](t)  ← 规划
  │  B,dB,   │    │ _traj    │    evaluate_plan() → st, (x,y)
  │ d2B,d3B  │    └──────────┘         │
  └──────────┘                         │
                         ┌─────────────┼─────────────┐
                         ▼             ▼             ▼
                    ┌──────────┐  ┌──────────┐  ┌──────────┐
                    │ warmstart│  │   cost   │  │constraints│
                    │ Greville │  │ Σd²+Σv²  │  │P5→P1 嵌套│
                    │build_mu()│  │build_ctx()│ │g_values()│
                    └────┬─────┘  └────┬─────┘  └────┬─────┘
                         │             │               │
                         ▼             ▼               ▼
                    ┌──────────────────────────────────────┐
                    │         build_solver()               │
                    │  Constran.build() + IGO 一站         │
                    │  result = solver(key, ctx, mu_init)  │
                    └────────────────┬─────────────────────┘
                                     │
                                     ▼  result.x → ctrl_s, ctrl_d
  ┌──────────────┐    ┌──────────┐    ┌──────────────────────┐
  │vehicle_model │◀───│ execute  │◀───│  s̈_cmd = plan(t=dt)  │  ← 执行
  │摩擦圆 + Euler │    │FrenetState│   └──────────────────────┘
  └──────────────┘    └──────────┘
         │                  │
         ▼                  ▼
  ┌──────────┐    ┌──────────────────┐
  │ reporting│    │     plotting     │
  │StepReport│    │ setup/render/save│
  └──────────┘    └──────────────────┘
```

### 核心思想：Frenet → 车辆运动学变换

B 样条在 Frenet 坐标系 (s, d) 里规划。`to_vehicle_states()` 把 Frenet 导数变换到车辆坐标系：

```
v_t = (1-d·κ_r)·ḃ          切向速度（沿参考线）
v_n = ḋ                    法向速度（跨参考线）
v   = sqrt(v_t² + v_n²)   总速度

a_t = (1-d·κ_r)·s̈ - 2·κ_r·ḃ·ḋ               切向加速度
a_n = d̈ + κ_r·(1-d·κ_r)·ḃ²                   法向加速度（含离心项）

a_long =  a_t·cosΔψ + a_n·sinΔψ              旋转到车辆纵轴
a_lat  = -a_t·sinΔψ + a_n·cosΔψ              旋转到车辆横轴
```

弯曲参考线 (κ_r≠0) 下离心项可达 ACC_MAX 的 65%（R=100m, v=18m/s → κ_r·v²≈3.2 m/s²）。直路 (κ_r=0) 全部退化到简化形式。

**cost 和约束都走 `to_vehicle_states()`，不再混用原始 Frenet 导数。**

## 文件结构

```
Cartest/
├── spline.py            # B 样条基函数预计算 (B, dB, d2B, d3B, d4B)
├── frenet_traj.py       # Frenet B 样条: evaluate, evaluate_plan, to_vehicle_states
├── reference_path.py    # 参考线抽象 (Straight + 弯道接口)
├── warmstart.py         # Warm-start: build_initial_mu, mpc_warmstart
├── cost.py              # 目标函数 + build_context
├── constraints.py       # 约束 + compute_g_values + compute_summary
├── execute.py           # 执行: plan→model→FrenetState
├── vehicle_model.py     # 车辆模型 (摩擦圆 + Euler)
├── reporting.py         # StepReport 记录
├── plotting.py          # 可视化 (setup, render, save)
├── Simple.py            # MPC demo
└── bspline_basis.npz    # 预计算基函数矩阵
```

## 1. 地图输入：ReferencePath

参考线 = 弧长参数化的光滑中心线。`ReferencePath` 是抽象基类，两个核心方法：

```python
class ReferencePath:
    def evaluate(self, s) -> (x_r, y_r, θ_r, κ_r)
    def frenet_to_cartesian(self, s, d) -> (x, y)
```

### 内置：StraightReference

```python
from Cartest.reference_path import StraightReference

ref = StraightReference()
# x = s, y = d, θ = 0, κ = 0
```

### 自定义弯道

继承 `ReferencePath`，实现 `evaluate(s)`：

```python
class CircularReference(ReferencePath):
    def __init__(self, radius=100.0):
        self.R = radius

    def evaluate(self, s):
        θ = s / self.R                          # 弧长 → 角度
        x_r = self.R * jnp.sin(θ)               # 圆心在原点
        y_r = self.R * (1 - jnp.cos(θ))
        θ_r = θ                                  # 切线方向
        κ_r = jnp.full_like(s, 1.0 / self.R)    # 恒定曲率
        return x_r, y_r, θ_r, κ_r
```

参考线只用于**避障约束**的 Frenet→Cartesian 映射和车辆状态的 heading/曲率计算。规划本身完全在 (s, d) 空间进行。

## 2. B 样条轨迹

5 次 (quintic) B 样条，12 个控制点，10 秒规划时域，C⁴ 连续。

```
t_eval ∈ [0, 10]s, dt = 0.1s, T = 100 个采样点
```

### 夹紧边界条件（C0/C1/C2）

前 3 个控制点夹紧，保证从当前状态出发：

```
P0 = x0                                    → C0: 位置连续
P1 = P0 + (Δt_knot/5) · v0                → C1: 速度连续
P2 = 3·P1 − 2·P0 + (Δt²_knot/10) · a0   → C2: 加速度连续
```

后 9 个控制点是自由的（优化变量），θ = [ctrl_s(9) | ctrl_d(9)]，共 18 维。

### 基函数矩阵

预计算 `spline.py` 生成 `bspline_basis.npz`：

```bash
uv run python Cartest/spline.py
```

生成 B, dB, d2B, d3B, d4B 矩阵（各 [100, 12]）和 Greville 横坐标。

## 3. 车辆模型 & 执行

执行链：`execute_step()` 取 plan 的期望加速度 → 车辆模型前向仿真 → 返回 `FrenetState`。

```python
from Cartest.execute import execute_step, FrenetState

# execute_step 内部:
#   1. s_ddot_cmd = plan(t=dt)      ← plan 的意图
#   2. vehicle.step(cmd)            ← 摩擦圆 + Euler
#   3. return FrenetState(s, s_dot, s_ddot=ax, ...)  ← 模型的真实加速度
```

```python
# vehicle_model.py
class FrenetVehicleModel:
    def __init__(self, mu=0.85, dt=0.1):
        self.a_max = mu * 9.81    # ≈ 8.3 m/s² 摩擦圆

    def step(self, s0, d0, s_dot0, d_dot0, s_ddot_cmd, d_ddot_cmd):
        a_cmd = sqrt(s_ddot_cmd² + d_ddot_cmd²)
        scale = min(1.0, a_max / a_cmd)
        ax, ay = s_ddot_cmd * scale, d_ddot_cmd * scale
        # Euler 积分
        return s0 + s_dot0*dt, d0 + d_dot0*dt, \
               s_dot0 + ax*dt, d_dot0 + ay*dt, ax, ay
```

**换模型只需改 `vehicle_model.py`**——`execute_step` 只依赖 `step()` 接口。

## 4. 代价函数 & 约束

### 目标

```python
# cost.py — 也走 to_vehicle_states
d, v = _eval_all(theta, ctx, gen)   # d: Frenet 横向偏移, v: st[:,2] 总速度
cost = Σ(v - v_target)² + Σ(d)²
```

两个目标：进入车道中心 (d→0)，达到目标速度 (v→v_target)。B 样条自带 C⁴ 光滑，无需显式平滑惩罚。

### 约束：按积分链组织

物理量的因果积分链：

```
jerk (s⃛) ──∫──▶ acc (s̈) ──∫──▶ speed (ḃ) ──∫──▶ position (s,d)
 控制输入         中间量         中间量           输出
 最外层           ...            ...            最内层
 P5               P4             P3             P2, P1
```

约束按积分链从外到内排列（Constran self-similar σ 嵌套）：

| Priority | 层 | 约束 | 来源 |
|----------|----|------|------|
| P1 (内) | 避障 | 穿透深度 | `to_vehicle_states → x,y` |
| P2 | 车道 | `|d| ≤ lane_hw` | Frenet d |
| P3 | 速度 | `V_min ≤ v ≤ V_max` | `st[:,2]` (车辆总速度) |
| P4 | 加速度 | `max(|a_long|,|a_lat|,|a_total|) ≤ ACC_MAX` | `st[:,4:6]` |
| P5 (外) | jerk | `max(|j_long|,|j_lat|,|j_total|) ≤ JERK_MAX` | `st[:,6:8]` |

### 为什么这个顺序？

- 积分链因果：jerk → acc → speed → position → 避障距离。外层物理约束对了，内层避障才可执行。

### 模式

- **P1 (obstacle)**: `mode='hard'` → baseline=2.0 → 安全底线
- **P2-P5**: `mode='soft'` → baseline=0 → 无违规时透明 → 目标信号恢复

### 约束函数 — 统一走 to_vehicle_states

```python
def acc_g(theta, ctx):
    st = _eval_vehicle_states(theta, ctx, gen)   # [T, 9] 车辆状态
    a_long, a_lat = st[:, 4], st[:, 5]
    am = jnp.sqrt(a_long**2 + a_lat**2)
    # 三者取最大: 纵向 / 横向 / 合成幅值
    return jnp.maximum(
        jnp.maximum(0., jnp.abs(a_long) - ACC_MAX),
        jnp.maximum(jnp.maximum(0., jnp.abs(a_lat) - ACC_MAX),
                    jnp.maximum(0., am - ACC_MAX)),
    )

def jerk_g(theta, ctx):   # 同理, 用 st[:,6:8]
    ...
```

## 5. IGO 优化器

`build_solver()` 把 Constran 构建 + solver 选择 + 参数初始化 + best-x 提取全包了：

```python
from gmm_igo.solver_builder import build_solver

solver = build_solver(
    obj_fn, dims=(9, 9),
    constraints=constraints,      # Constran.build() 内部处理
    solver='m22',                 # 自动选: 'auto' / 'm22' / 'reuse_multi'
    T=500, dt=0.15, K=3, B=64, B0=20, T_0=250,
    k_inner=0.1, obj_transform='standard',
)

# MPC 循环中一行求解:
result = solver(key, context=ctx, initial_mu=mu_init)
ctrl_s, ctrl_d = result.x[:9], result.x[9:]
# result.cost, result.mu, result.S_or_L, result.pi 全可用

# 也可继承上一轮 GMM 状态:
result = solver(key, context=ctx, warm_start=prev_result)
```

IGO 是黑箱全局优化——不需要梯度，只要求值。

### MPC 主循环

```python
state = FrenetState(s=0, s_dot=12, s_ddot=0, d=-3, d_dot=0, d_ddot=0)

for step in range(steps):
    ctx      = build_context(gen, state, V_TARGET, LANE_HW, obs_pos, obs_rad)
    mu_init  = build_initial_mu(gen, state.s, state.s_dot, state.d)
    result   = solver(key, context=ctx, initial_mu=mu_init)
    frenet, st, (x, y) = gen.evaluate_plan(result.x[:9], result.x[9:], ctx)
    gv       = compute_g_values(st, d, x, y, obs_pos, obs_rad)
    sm       = compute_summary(st, d, x, y, obs_pos, obs_rad)
    state    = execute_step(gen, s, d, s_dot, d_dot, s_ddot, d_ddot, vehicle)
    report   = StepReport(...)
```

## 6. 运行

```bash
# 1. 生成基函数矩阵 (只需一次)
uv run python Cartest/spline.py

# 2. 运行 MPC demo
uv run python Cartest/Simple.py --steps 150 --seed 0

# 跳过绘图
uv run python Cartest/Simple.py --steps 50 --no-plot
```

输出 `Cartest/frenet_demo.gif`（执行轨迹 vs 规划轨迹的动画）。

## 7. 关键设计决策

| 决策 | 理由 |
|------|------|
| Frenet 坐标 | ctrl→jerk/acc 线性，消除 Cartesian 的非线性放大 |
| `to_vehicle_states` 统一 | cost/约束/reporting 同一口径，离心/Coriolis/(1-d·κ_r) 全包含 |
| 5 次 B 样条 | C⁴ 连续：jerk 连续，bang 有界 |
| 单侧夹紧 | 初始状态精确匹配；终端自由（MPC 只执行第一步） |
| `evaluate_plan()` 一站评估 | Frenet + 车辆状态 + Cartesian 一次调用，避免重复 |
| Greville-based warm-start | 精确常速轨迹：jerk=0, acc=0 |
| `build_initial_mu()` | 物理 warm-start → solver-ready mu_init，一行调用 |
| soft constraints (P2-P5) | 无违规时透明，目标信号不被 baseline 淹没 |
| hard constraint (P1) | 安全底线，baseline=2.0 不可协商 |
| 约束三取一罚函数 | `max(|long|-LIM, |lat|-LIM, |total|-LIM, 0)` 方向+幅值全覆盖 |
| `compute_g_values/summary` | 约束公式同源，reporting 不会跟约束不同步 |
| `build_context()` | ctx 构造统一入口，不再手写字典 |
| `FrenetState` dataclass | 类型安全，`.to_ctx()` 直接生成 ctx 条目 |
| `build_solver()` | Constran + solver + best-x 一站，warm_start 继承 GMM |
| execute → model → 真实状态 | 规划/仿真解耦，摩擦圆限幅后的真实加速度写回 state |

## 依赖

- `jax[cuda12]` — B 样条矩阵运算
- `numpy`, `scipy` — 基函数预计算
- `matplotlib` — 可视化
- `gmm_igo` — IGO 优化器
- `Constraintdealer.Constran` — σ 嵌套约束引擎
