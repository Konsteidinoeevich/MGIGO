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
  ┌─────────────┐   ┌──────────────────────────┐
  │ frenet_traj │   │ build_context + warmstart │  规划准备
  │ evaluate +  │   │ + make_constraints        │
  │ to_vehicle  │   └────────────┬─────────────┘
  └──────┬──────┘                │
         │                       ▼
         │              ┌─────────────────┐
         └──────────────│  build_solver() │  Constran + IGO 一站
                        └────────┬────────┘
                                 │
                                 ▼  result.x → ctrl_s, ctrl_d
  ┌──────────────┐    ┌──────────┐    ┌──────────────────┐
  │ BicycleModel │◀───│ execute  │◀───│  s̈_cmd = plan    │  执行
  │ 摩擦圆 + 转向│    │FrenetState│   │  (t=dt)           │
  └──────────────┘    └──────────┘    └──────────────────┘
         │
         ▼
  ┌──────────┐    ┌──────────────────┐
  │ reporting│    │     plotting     │
  │StepReport│    │ setup/render/save│
  └──────────┘    └──────────────────┘
```

**分三层**：
- **地图** (`ReferencePath`) — 道路中心线，提供 Frenet ↔ Cartesian 映射
- **场景** (`Scenario`) — 障碍物、车道宽度、目标速度、初始状态，一个 dict 全包
- **规划/执行** — B 样条轨迹优化 → IGO 求解 → 车辆模型仿真 → MPC 闭环

## 文件结构

```
Cartest/
├── spline.py            # B 样条基函数预计算
├── frenet_traj.py       # Frenet 轨迹: evaluate, evaluate_plan, to_vehicle_states
├── reference_path.py    # 参考线 (StraightReference + 弯道接口)
├── scenario.py          # 场景配置: 障碍物 + 道路参数 + 初始状态
├── warmstart.py         # Warm-start: build_initial_mu, mpc_warmstart
├── cost.py              # 目标函数 + build_context
├── constraints.py       # 约束构建 + compute_g_values + compute_summary
├── execute.py           # 执行桥: plan → model → FrenetState
├── vehicle_model.py     # 车辆模型 (BicycleModel: 摩擦圆 + 转向 + yaw)
├── diagnostics.py       # 诊断: raw obj, g 值, 车辆当前状态
├── reporting.py         # StepReport 记录
├── plotting.py          # 可视化
├── Simple.py            # MPC demo
└── bspline_basis.npz    # 预计算基函数矩阵
```

## 1. 地图 — ReferencePath

参考线 = 弧长参数化的光滑中心线。实现 `evaluate(s) → (x, y, θ, κ)` 和 `frenet_to_cartesian(s, d)`。

内置 `StraightReference`（直路）。自定义弯道只需继承并实现 `evaluate`。

规划在 (s, d) 空间进行，`to_vehicle_states()` 负责 Frenet → 车辆运动学变换（含离心项、Coriolis、Δψ 旋转）。

## 2. 场景 — Scenario

`scenario.py` 是所有场景参数的唯一来源。切换场景只需改一行 import：

```python
from Cartest.scenario import THREE_BLOCKING as scenario
```

每个场景是一个 dict：

```python
SCENE = {
    "obstacles": [
        {"x": 45.0, "y": -2.5, "r": 2.0},   # 障碍物列表
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

所有模块从场景取值，无重复定义。

## 3. 初始状态 — FrenetState

`execute.py` 定义了 `FrenetState` 数据类，包含完整车辆状态和 `to_ctx()` 方法：

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

场景中的 `init` 直接构造 `FrenetState`，经 `build_context()` 生成 solver 的 `ctx`。

## 4. B 样条轨迹

5 次 B 样条，12 控制点（前 3 夹紧 = C0/C1/C2 初始状态匹配），10 秒时域，100 采样点。后 9 控制点自由（18 维优化变量）。

`evaluate_plan()` 一站返回 Frenet + 车辆状态 [T,9] + Cartesian 坐标。

## 5. 车辆模型 & 执行

`BicycleModel`（转向 + 摩擦圆 + yaw 动力学）：

```
plan 输出 s̈_cmd, d̈_cmd
  → 旋转到车体 (a_long, a_lat)
  → 转向角 δ = arctan(a_lat·L/v²)
  → 摩擦圆限幅
  → ψ̇ = v·tan(δ)/L
  → Frenet 积分
  → 返回 FrenetState (含 ψ)
```

`execute_step()` 是纯桥接——取 plan 的指令，传模型仿真，透传结果。换模型只改 `vehicle_model.py`。

## 6. IGO 优化器

`build_solver()` 一站：Constran 约束组装 + solver 选择 + 参数初始化 + best-x 提取。

```python
solver = build_solver(obj_fn, dims=(9, 9),
    constraints=make_constraints(gen, lane_hw, safe_dist),
    solver='m22', T=300, dt=0.15, K=3, B=100, B0=45, T_0=300,
    k_inner=1.0, obj_transform='standard',
)

result = solver(key, context=ctx, initial_mu=mu_init)
ctrl_s, ctrl_d = result.x[:9], result.x[9:]
```

支持 GMM 状态继承：`solver(key, context=ctx, warm_start=prev_result)`。

## 7. 运行

```bash
uv run python Cartest/spline.py     # 生成基函数矩阵 (只需一次)
uv run python Cartest/Simple.py --steps 150 --seed 0
uv run python Cartest/Simple.py --steps 50 --no-plot
```

输出 `Cartest/frenet_demo.gif`。

## 8. 关键设计决策

| 决策 | 理由 |
|------|------|
| Frenet 坐标 | 参考线承担曲率，B 样条 ctrl→物理量线性 |
| `to_vehicle_states` 统一 | cost/约束/reporting 同口径 |
| Scenario 单文件 | 切场景只改一行 import |
| `FrenetState` dataclass | 类型安全，`.to_ctx()` 自动 |
| `BicycleModel` | 转向 + yaw + 摩擦圆，比点质量真 |
| `build_solver()` | Constran + IGO + warm-start 一站 |
| 自相似 σ 嵌套约束 | 外→内: jerk/acc/speed/lane/obs，因果积分链对齐 |
| RSS 障碍物约束 | 纵向制动距离 + 横向间隙，各自判定取 max |
| 诊断分离 | `cur_obs`=车辆真实距离, `g_max`=规划层约束压力 |

## 依赖

- `jax[cuda12]` — 矩阵运算
- `numpy`, `scipy` — 基函数预计算
- `matplotlib` — 可视化
- `gmm_igo` — IGO 优化器
- `Constraintdealer.Constran` — σ 嵌套约束引擎
