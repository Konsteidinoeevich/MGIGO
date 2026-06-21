import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.patches import Circle

project_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from gmm_igo.MPCsolver import igo_mog_optimizer
from Constraintdealer.Constran import build, Deterministic

# ==============================================================================
# 1. 配置参数
# ==============================================================================

CONFIG = {
    'dt': 0.1,
    'horizon': 12,        # 预测时域
    'dim': 2,             # 控制维度 [v, omega]
    'n_components': 4,    
    'pop_size': 100,      
    'elite_size': 40,     
    'opt_steps': 200,    
    'warmup_steps': 1000, 
    
    # 物理限制
    'max_v': 2.5,         
    'max_w': 1.0,         
    
    # 障碍物
    'obs_rows': 4,
    'obs_cols': 4,
    'obs_spacing': 4.0,   
    'obs_radius': 1.9,    
    'safe_margin': 0.0,   
}

TOTAL_DIM = CONFIG['horizon'] * CONFIG['dim']

# ==============================================================================
# 2. 饱和嵌套代价函数 (Auto-Scaling Saturation Nesting)
#
#    2-Level 严格优先级 (仿 Hybridsystemtest.py):
#      L1 (外层): 障碍物碰撞 → 1.5 + penetration, 碰撞区内连续梯度
#      L2 (内层): 跟踪 + 终端 + 平滑 + 控制 → 安全区内连续梯度
#
#    自动缩放:
#      L1_SCALE = H * N_obs * (1.5 + safe_dist)
#      L2_SCALE = (2H+15)*dist + ctrl_floor  (ctrl_floor 防 dist→0 时 NaN)
#
#    输出范围 (数学保证):
#      L1 激活 (碰撞):  [0.768, 1.000)
#      L2 仅 (无碰撞):  [0.000, 0.707)
#
#    参考: Functiontest/Hybridsystemtest.py + Hybrid_test_README.md
# ==============================================================================

@jax.jit
def saturate(x):
    return (x / (jnp.sqrt(1.0 + x**2)))


@jax.jit
def mpc_cost_fn(flat_actions, context):
    raw_actions = flat_actions.reshape((CONFIG['horizon'], CONFIG['dim']))
    vs = jnp.tanh(raw_actions[:, 0]) * CONFIG['max_v']
    ws = jnp.tanh(raw_actions[:, 1]) * CONFIG['max_w']

    x0, y0, theta0 = context['current_state']
    target = context['target_pos']
    dt = CONFIG['dt']

    thetas = theta0 + jnp.cumsum(ws * dt)
    thetas_prev = jnp.concatenate([jnp.array([theta0]), thetas[:-1]])

    dx = vs * jnp.cos(thetas_prev) * dt
    dy = vs * jnp.sin(thetas_prev) * dt

    trajectory = jnp.stack([x0 + jnp.cumsum(dx), y0 + jnp.cumsum(dy)], axis=1)

    dist_to_target = jnp.linalg.norm(trajectory - target, axis=1)
    cost_track = jnp.sum(dist_to_target) * 2.0
    cost_final = jnp.linalg.norm(trajectory[-1] - target) * 15.0

    cost_smooth = jnp.sum(jnp.diff(ws)**2) * 1.5 + jnp.sum(jnp.diff(vs)**2) * 0.5
    cost_v = jnp.sum(vs**2) * 1.0 + jnp.sum(ws**2) * 0.5

    # L2 raw: 总运行代价 + mu 正则化 (防 mu 在 tanh 饱和区无意义膨胀)
    mu_reg = 0.01 * (jnp.sum(raw_actions[:, 0]**2) + jnp.sum(raw_actions[:, 1]**2))
    f_total = cost_track + cost_final + cost_v + mu_reg

    # L1 raw: 障碍物碰撞 — 1.5 + penetration, 与 Hybrid 行人公式一致

    obs_pos = context['obs_pos']
    safe_dist = context['safe_distance']

    diff = trajectory[:, None, :] - obs_pos[None, :, :]
    min_dists = jnp.min(jnp.linalg.norm(diff, axis=-1), axis=-1)

    penetration = safe_dist - min_dists
    viol_obs = jnp.sum(jnp.where(penetration > 0.0, 1.5 + penetration, 0.0))

    

    # —— 自动缩放 (Hybrid README §4) ——
    H = CONFIG['horizon']
    init_pos = jnp.array([x0, y0])
    dist_init_to_target = jnp.linalg.norm(init_pos - target)

    # L1_SCALE: 单步单障碍的特征违反 = 1.5 + safe_dist
    #   不乘 H*N_obs — 让总违反量随步数和障碍数自然累加
    #   小擦边 (0.01m, 1步1障碍) → 归一化 ~0.44 → saturate ~0.4  (< L2 max)
    #   持续碰撞 (3步2障碍, 0.3m)  → 归一化 ~3.2 → saturate ~0.95 (> L2 max)
    L1_SCALE = 1.5 + safe_dist

    # L2_SCALE: 零控制跟踪 + 控制代价下界 (防 dist→0 时 NaN)
    ctrl_cost_floor = H * (CONFIG['max_v']**2 + CONFIG['max_w']**2)
    L2_SCALE = (2.0 * H + 15.0) * dist_init_to_target + ctrl_cost_floor + 1.0

    # —— 饱和嵌套 (加法模式, 非分支) ——
    #   viol/L1_SCALE + saturate(f/L2_SCALE): L1 和 L2 始终同时生效
    #   - 无碰撞:      res = saturate(0 + l2_sat)    纯跟踪
    #   - 小擦边+好跟踪: res = saturate(0.44+0.1)=0.48  比"无碰+烂跟踪"好 → 可穿缝
    #   - 小擦边+烂跟踪: res = saturate(0.44+0.7)=0.76  比"无碰+烂跟踪"差 → 被避免
    #   - 大碰撞:       res → 1.0                       严格避免
    res = saturate(
        viol_obs / L1_SCALE + saturate(f_total / L2_SCALE)
    )

    return res


# ======================================================================
# 2b. Constran-based cost function (declarative, build() from Constraintdealer)
#
#     Same 2-level structure as the original, but expressed declaratively:
#       L1 (outermost): obstacle collision → Deterministic(…, mode='soft')
#       L2 (innermost):  tracking + smooth + control + reg → objective_fn
#
#     Key differences from original auto-scaling:
#       - log_transform replaces viol_obs/L1_SCALE and f_total/L2_SCALE
#       - No per-problem scale estimation (L1_SCALE, L2_SCALE)
#       - penalize_only_soft=True: never rewards obstacle distance
#       - k_inner=0.1: wide dynamic range for objective
#
#     Soft mode: obstacle violation and tracking ALWAYS compete additively.
#     Small scrape + good tracking can beat no-scrape + bad tracking.
#     This preserves the original's "可穿缝" (squeeze-through-gap) behavior.
# ======================================================================

def _tracking_objective(z_flat, ctx):
    """L2 (innermost): tracking + final + smooth + control + mu regularization.

    Uses pre-computed trajectory from ctx['trajectory'] and raw controls
    from z_flat for regularization terms.
    """
    raw_actions = z_flat.reshape((CONFIG['horizon'], CONFIG['dim']))
    vs = jnp.tanh(raw_actions[:, 0]) * CONFIG['max_v']
    ws = jnp.tanh(raw_actions[:, 1]) * CONFIG['max_w']

    trajectory = ctx['trajectory']
    target = ctx['target_pos']

    dist_to_target = jnp.linalg.norm(trajectory - target, axis=1)
    cost_track = jnp.sum(dist_to_target) * 2.0
    cost_final = jnp.linalg.norm(trajectory[-1] - target) * 15.0

    cost_smooth = jnp.sum(jnp.diff(ws) ** 2) * 1.5 + jnp.sum(jnp.diff(vs) ** 2) * 0.5
    cost_v = jnp.sum(vs ** 2) * 1.0 + jnp.sum(ws ** 2) * 0.5
    mu_reg = 0.01 * (jnp.sum(raw_actions[:, 0] ** 2) + jnp.sum(raw_actions[:, 1] ** 2))

    return cost_track + cost_final + cost_v + mu_reg


def _obstacle_violation(z_flat, ctx):
    """L1 (outermost): obstacle collision violation (raw sum of 1.5 + penetration).

    Returns raw violation > 0 when any trajectory point penetrates an obstacle.
    Constran applies log_transform → sigma_k → additive nesting automatically.
    Uses the same 1.5 + penetration formula as the original for consistency.
    """
    trajectory = ctx['trajectory']
    obs_pos = ctx['obs_pos']
    safe_dist = ctx['safe_distance']

    diff = trajectory[:, None, :] - obs_pos[None, :, :]
    min_dists = jnp.min(jnp.linalg.norm(diff, axis=-1), axis=-1)
    penetration = safe_dist - min_dists
    return jnp.sum(jnp.where(penetration > 0.0, 1.5 + penetration, 0.0))


# Build the nested cost via Constran (no jit — we wrap with rollout + jit below)
# SOFT mode: obstacle always active, competes additively with tracking
# penalize_only_soft=True: log_transform(g) clamped to ≥ 0 — never rewards being far
_base_mpc_cost = build(
    _tracking_objective,
    [
        Deterministic(_obstacle_violation, mode='soft', priority=1),
    ],
    k_inner=0.1,
    penalize_only_soft=True,
    jit_cost=False,
)


@jax.jit
def mpc_cost_fn_constran(flat_actions, context):
    """Constran-wrapped MPC cost — single trajectory rollout, modular constraints.

    This is the solver-ready cost function. It:
    1. Computes the unicycle trajectory from flat_actions (same as original).
    2. Passes trajectory via extended context to the Constran-built cost.
    3. Constran handles log_transform, σ nesting, and additive soft competition.

    The solver sees the same (flat_actions, context) -> scalar interface.
    """
    raw_actions = flat_actions.reshape((CONFIG['horizon'], CONFIG['dim']))
    vs = jnp.tanh(raw_actions[:, 0]) * CONFIG['max_v']
    ws = jnp.tanh(raw_actions[:, 1]) * CONFIG['max_w']

    x0, y0, theta0 = context['current_state']
    dt = CONFIG['dt']

    thetas = theta0 + jnp.cumsum(ws * dt)
    thetas_prev = jnp.concatenate([jnp.array([theta0]), thetas[:-1]])

    dx = vs * jnp.cos(thetas_prev) * dt
    dy = vs * jnp.sin(thetas_prev) * dt
    trajectory = jnp.stack([x0 + jnp.cumsum(dx), y0 + jnp.cumsum(dy)], axis=1)

    # Extend context with computed trajectory
    ext_ctx = dict(context)
    ext_ctx['trajectory'] = trajectory
    return _base_mpc_cost(flat_actions, ext_ctx)


# ==============================================================================
# 3. 改进的决策辅助函数
# ==============================================================================

def shift_solution_with_diversity(mu_k, K, horizon, dim, key, action_range=1.0):
    """
    平移旧解，并在末端进行全域均匀探索。
    """
    mu_reshaped = mu_k.reshape(K, horizon, dim)
    
    # 1. 执行常规平移 (t=1~H-1 保持意图)
    mu_shifted = jnp.roll(mu_reshaped, shift=-1, axis=1)
    
    # 2. 生成完全独立的、均匀分布的末端探索
    # 假设控制量经过 tanh 映射，范围在 [-action_range, action_range]
    key, subkey = jax.random.split(key)
    # 产生 (K, dim) 的均匀分布随机数
    uniform_tail = jax.random.uniform(
        subkey, 
        shape=(K, dim), 
        minval=-action_range, 
        maxval=action_range
    )
    
    # 3. 替换末端
    # 这样每个分量在 H 时刻都有一个完全随机的初始搜索方向
    mu_shifted = mu_shifted.at[:, -1, :].set(uniform_tail)
    
    return mu_shifted.reshape(K, -1), key

# ==============================================================================
# 4. 主循环 (集成决策滞回逻辑)
# ==============================================================================

def _draw_frame(ax, robot_state, current_obs_pos, target_pos, mu_k, best_idx, comp_colors, step):
    """在当前 Matplotlib 画布上绘制一帧场景。"""
    ax.cla()
    ax.set_xlim(-2, 22)
    ax.set_ylim(-2, 22)
    ax.set_aspect('equal')

    obs_np = np.asarray(current_obs_pos)
    for center in obs_np:
        ax.add_patch(Circle(center, CONFIG['obs_radius'], color='red', alpha=0.3))

    robot_state_np = np.asarray(robot_state)
    target_np = np.asarray(target_pos)

    for k in range(mu_k.shape[0]):
        comp_u = mu_k[k].reshape(CONFIG['horizon'], CONFIG['dim'])
        c_vs = jnp.tanh(comp_u[:, 0]) * CONFIG['max_v']
        c_ws = jnp.tanh(comp_u[:, 1]) * CONFIG['max_w']
        c_thetas = robot_state_np[2] + jnp.cumsum(c_ws * CONFIG['dt'])
        c_thetas_pre = jnp.concatenate([jnp.array([robot_state_np[2]]), c_thetas[:-1]])
        c_traj = jnp.stack([
            robot_state_np[0] + jnp.cumsum(c_vs * jnp.cos(c_thetas_pre) * CONFIG['dt']),
            robot_state_np[1] + jnp.cumsum(c_vs * jnp.sin(c_thetas_pre) * CONFIG['dt'])
        ], axis=1)

        c_traj_np = np.asarray(c_traj)
        is_selected = (k == best_idx)
        lw = 4.5 if is_selected else 1.0
        alpha_val = 0.9 if is_selected else 0.2
        ax.plot(c_traj_np[:, 0], c_traj_np[:, 1], color=comp_colors[k], lw=lw, alpha=alpha_val, zorder=4 if is_selected else 3)

    ax.plot(target_np[0], target_np[1], 'g*', ms=12)
    ax.arrow(
        robot_state_np[0],
        robot_state_np[1],
        0.6 * np.cos(robot_state_np[2]),
        0.6 * np.sin(robot_state_np[2]),
        head_width=0.3,
        color='blue',
        zorder=5
    )
    ax.set_title(f"Step {step} | Sticky Decision (Hysteresis) | Comp {best_idx} Chosen")


def run_mpc_simulation(video_path="outcmaes/mpcmain21.gif", steps=350, fps=20,
                       cost_mode='original'):
    """Run MPC simulation.

    cost_mode: 'original'  → auto-scaling + saturate (current behavior)
               'constran'  → Constraintdealer.Constran.build() — declarative,
                              log_transform, soft additive nesting
    """
    # Select cost function
    if cost_mode == 'constran':
        cost_fn = mpc_cost_fn_constran
        mode_label = 'CONSTRAN'
    else:
        cost_fn = mpc_cost_fn
        mode_label = 'ORIGINAL'

    # Adjust video path for constran mode
    if cost_mode == 'constran' and video_path == "outcmaes/mpcmain21.gif":
        video_path = "outcmaes/mpcmain21_constran.gif"

    key = jax.random.PRNGKey(42)
    K = CONFIG['n_components']

    obs_initial_pos = generate_grid_obstacles(4, 4, CONFIG['obs_spacing'], 2.5, 2.5)
    obs_num = obs_initial_pos.shape[0]
    key, subkey = jax.random.split(key)
    obs_phases = jax.random.uniform(subkey, (obs_num, 2)) * 2 * jnp.pi

    mu_k = jax.random.normal(key, (K, TOTAL_DIM)) * 0.0
    L_inv_k = jnp.stack([jnp.eye(TOTAL_DIM) * 1.0 for _ in range(K)])
    pi_k_all = jnp.ones(K) / K

    last_best_idx = 0
    robot_state = jnp.array([0.0, 0.0, 0.0])
    target_final = jnp.array([18.0, 15.0])

    video_path = os.path.abspath(video_path)
    output_dir = os.path.dirname(video_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    comp_colors = plt.get_cmap('jet')(np.linspace(0, 1, K))
    writer = PillowWriter(fps=fps, metadata={'title': 'IGO MPC Simulation', 'artist': 'MPCmain21'})

    with writer.saving(fig, video_path, dpi=120):
        for t in range(steps):
            key, subkey = jax.random.split(key)

            offsets = 0.25 * jnp.stack([jnp.sin(t * 0.1 + obs_phases[:, 0]), jnp.cos(t * 0.12 + obs_phases[:, 1])], axis=1)
            current_obs_pos = obs_initial_pos + offsets
            target_pos = target_final + jnp.array([jnp.sin(t * 0.1), jnp.cos(t * 0.2)]) * 0.2

            context_data = {
                'current_state': robot_state, 'target_pos': target_pos,
                'obs_pos': current_obs_pos, 'obs_radius': CONFIG['obs_radius'],
                'safe_distance': CONFIG['obs_radius'] + CONFIG['safe_margin']
            }

            iter_steps = CONFIG['warmup_steps'] if t == 0 else CONFIG['opt_steps']
            mu_k, L_inv_k, pi_k_all = igo_mog_optimizer(
                subkey, iter_steps, 0.15, K, CONFIG['pop_size'], CONFIG['elite_size'],
                cost_fn, mu_k, L_inv_k, pi_k_all, context_data
            )

            hysteresis_bias = 0.00
            biased_pi = pi_k_all.at[last_best_idx].add(hysteresis_bias)
            best_idx = jnp.argmax(biased_pi)
            last_best_idx = best_idx

            best_flat = np.asarray(mu_k[best_idx])
            mu_str = np.array2string(best_flat, precision=3, suppress_small=True)
            print(f"Step {t:03d} | best component {int(best_idx)} | mu={mu_str}")

            best_u_seq = best_flat.reshape(CONFIG['horizon'], CONFIG['dim'])
            v_exec = jnp.tanh(3 * best_u_seq[0, 0]) * CONFIG['max_v']
            w_exec = jnp.tanh(3 * best_u_seq[0, 1]) * CONFIG['max_w']

            new_theta = robot_state[2] + w_exec * CONFIG['dt']
            new_x = robot_state[0] + v_exec * jnp.cos(robot_state[2]) * CONFIG['dt']
            new_y = robot_state[1] + v_exec * jnp.sin(robot_state[2]) * CONFIG['dt']
            robot_state = jnp.array([new_x, new_y, new_theta])

            mu_k, key = shift_solution_with_diversity(
                mu_k, K, CONFIG['horizon'], CONFIG['dim'], key
            )

            _draw_frame(ax, robot_state, current_obs_pos, target_pos, mu_k, best_idx, comp_colors, t)
            writer.grab_frame()

    plt.close(fig)

def generate_grid_obstacles(rows, cols, spacing, start_x, start_y):
    x = jnp.linspace(start_x, start_x + (cols-1)*spacing, cols)
    y = jnp.linspace(start_y, start_y + (rows-1)*spacing, rows)
    xx, yy = jnp.meshgrid(x, y)
    return jnp.stack([xx.ravel(), yy.ravel()], axis=1)

def parse_args():
    parser = argparse.ArgumentParser(description="Run MPC simulation and export video.")
    parser.add_argument(
        "--video",
        default=os.path.join("outcmaes", "mpcmain21.gif"),
        help="gif"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=350,
        help="simulation step"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="frequency"
    )
    parser.add_argument(
        "--mode",
        default='original',
        choices=['original', 'constran'],
        help="Cost function: 'original' (auto-scaling + saturate) or "
             "'constran' (Constraintdealer.Constran.build())"
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_mpc_simulation(video_path=cli_args.video, steps=cli_args.steps,
                       fps=cli_args.fps, cost_mode=cli_args.mode)