import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
import time

from gmm_igo.MPCsolver import igo_mog_optimizer

# ==============================================================================
# 1. 配置参数
# ==============================================================================

CONFIG = {
    'dt': 0.1,
    'horizon': 12,        # 预测时域
    'dim': 2,             # 控制维度 [v, omega]
    'n_components': 6,    
    'pop_size': 60,      
    'elite_size': 25,     
    'opt_steps': 250,     # 模拟优化轮次较少的情况
    'warmup_steps': 800, 
    
    # 物理限制
    'max_v': 2.5,         
    'max_w': 2.0,         
    
    # 障碍物
    'obs_rows': 4,
    'obs_cols': 4,
    'obs_spacing': 4.0,   
    'obs_radius': 2.0,    
    'safe_margin': 0.0,   
}

TOTAL_DIM = CONFIG['horizon'] * CONFIG['dim']

# ==============================================================================
# 2. Unicycle 动力学与代价函数 (保持原样，不破坏物理平衡)
# ==============================================================================

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
    
    obs_pos = context['obs_pos']
    safe_dist = context['safe_distance']
    
    diff = trajectory[:, None, :] - obs_pos[None, :, :]
    min_dists = jnp.min(jnp.linalg.norm(diff, axis=-1), axis=-1)
    
    collision_mask = jnp.where(min_dists < safe_dist, 1.0, -1.0)
    obs_activation = 0.5 + 0.5 * jnp.tanh(5.0 * collision_mask)
    cost_obstacle = 600.0 * jnp.sum(obs_activation * (safe_dist - min_dists))
    
    cost_smooth = jnp.sum(jnp.diff(ws)**2) * 1.5
    cost_v = jnp.sum(vs**2) * 0.1
    
    return cost_track + cost_final + cost_obstacle + cost_smooth + cost_v

# ==============================================================================
# 3. 改进的决策辅助函数
# ==============================================================================

def shift_solution_with_diversity(mu_k, K, horizon, dim):
    """在解推进时，为不同分量注入不同的末端探测倾向"""
    mu_reshaped = mu_k.reshape(K, horizon, dim)
    mu_shifted = jnp.roll(mu_reshaped, shift=-1, axis=1)
    
    # 分量个性化：即便它们目前很像，但在末端分别尝试直行、左转、右转
    # 这有助于在低优化轮次下维持分量的分布
    diversity_tail = jnp.array([
        [0.0, 0.0],   # 分量0: 维持现状
        [0.5, 0.5],   # 分量1: 尝试右偏
        [0.5, -0.5],  # 分量2: 尝试左偏
        [1.0, 0.0],   # 分量3: 尝试加速
        [-0.5, 0.0],  # 分量4: 尝试减速
        [0.0, 1.0],   # 分量5: 尝试急转
    ])
    mu_shifted = mu_shifted.at[:, -1, :].set(diversity_tail[:K])
    return mu_shifted.reshape(K, -1)

# ==============================================================================
# 4. 主循环 (集成决策滞回逻辑)
# ==============================================================================

def run_mpc_simulation():
    key = jax.random.PRNGKey(42)
    K = CONFIG['n_components']
    
    obs_initial_pos = generate_grid_obstacles(4, 4, CONFIG['obs_spacing'], 2.5, 2.5)
    obs_num = obs_initial_pos.shape[0]
    key, subkey = jax.random.split(key)
    obs_phases = jax.random.uniform(subkey, (obs_num, 2)) * 2 * jnp.pi
    
    mu_k = jax.random.normal(key, (K, TOTAL_DIM)) * 0.1
    L_inv_k = jnp.stack([jnp.eye(TOTAL_DIM) * 2.0 for _ in range(K)])
    pi_k_all = jnp.ones(K) / K
    
    # 决策层记录变量
    last_best_idx = 0
    
    robot_state = jnp.array([0.0, 0.0, 0.0])
    target_final = jnp.array([18.0, 15.0])
    history_robot = [robot_state[:2]]
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    comp_colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, K))
    
    for t in range(250):
        key, subkey = jax.random.split(key)
        
        offsets = 0.25 * jnp.stack([jnp.sin(t * 0.1 + obs_phases[:, 0]), jnp.cos(t * 0.12 + obs_phases[:, 1])], axis=1)
        current_obs_pos = obs_initial_pos + offsets
        target_pos = target_final + jnp.array([jnp.sin(t*0.05), jnp.cos(t*0.05)]) * 0.3
        
        context_data = {
            'current_state': robot_state, 'target_pos': target_pos,
            'obs_pos': current_obs_pos, 'obs_radius': CONFIG['obs_radius'],
            'safe_distance': CONFIG['obs_radius'] + CONFIG['safe_margin']
        }
        
        # 1. IGO 优化
        iter_steps = CONFIG['warmup_steps'] if t == 0 else CONFIG['opt_steps']
        mu_k, L_inv_k, pi_k_all = igo_mog_optimizer(
            subkey, iter_steps, 0.1, K, CONFIG['pop_size'], CONFIG['elite_size'], 
            mpc_cost_fn, mu_k, L_inv_k, pi_k_all, context_data
        )
        
        # 2. 决策层：引入滞回(Hysteresis)以消除折返
        # 给上一时刻的最佳分量 10% 的额外权重优势 (Bias)
        hysteresis_bias = 0.1
        biased_pi = pi_k_all.at[last_best_idx].add(hysteresis_bias)
        best_idx = jnp.argmax(biased_pi)
        last_best_idx = best_idx
        
        # 3. 提取并执行控制
        best_u_seq = mu_k[best_idx].reshape(CONFIG['horizon'], CONFIG['dim'])
        v_exec = jnp.tanh(best_u_seq[0, 0]) * CONFIG['max_v']
        w_exec = jnp.tanh(best_u_seq[0, 1]) * CONFIG['max_w']
        
        # 更新动力学
        new_theta = robot_state[2] + w_exec * CONFIG['dt']
        new_x = robot_state[0] + v_exec * jnp.cos(robot_state[2]) * CONFIG['dt']
        new_y = robot_state[1] + v_exec * jnp.sin(robot_state[2]) * CONFIG['dt']
        robot_state = jnp.array([new_x, new_y, new_theta])
        
        # 4. 解推进 (加入个性化扰动)
        mu_k = shift_solution_with_diversity(mu_k, K, CONFIG['horizon'], CONFIG['dim'])
        history_robot.append(robot_state[:2])
        
        # --- 绘图逻辑 ---
        if t % 1 == 0:
            ax.cla()
            ax.set_xlim(-2, 22); ax.set_ylim(-2, 22); ax.set_aspect('equal')
            for i in range(obs_num):
                ax.add_patch(Circle(current_obs_pos[i], CONFIG['obs_radius'], color='red', alpha=0.3))
            
            # 可视化所有分量的预测轨迹
            for k in range(K):
                comp_u = mu_k[k].reshape(CONFIG['horizon'], CONFIG['dim'])
                c_vs, c_ws = jnp.tanh(comp_u[:, 0]) * CONFIG['max_v'], jnp.tanh(comp_u[:, 1]) * CONFIG['max_w']
                c_thetas = robot_state[2] + jnp.cumsum(c_ws * CONFIG['dt'])
                c_thetas_pre = jnp.concatenate([jnp.array([robot_state[2]]), c_thetas[:-1]])
                c_traj = jnp.stack([
                    robot_state[0] + jnp.cumsum(c_vs * jnp.cos(c_thetas_pre) * CONFIG['dt']),
                    robot_state[1] + jnp.cumsum(c_vs * jnp.sin(c_thetas_pre) * CONFIG['dt'])
                ], axis=1)
                
                # 当前选中的分量画得最粗最亮
                is_selected = (k == best_idx)
                lw = 4.5 if is_selected else 1.0
                alpha_val = 0.9 if is_selected else 0.2
                ax.plot(c_traj[:, 0], c_traj[:, 1], color=comp_colors[k], lw=lw, alpha=alpha_val, zorder=4 if is_selected else 3)

            ax.plot(target_pos[0], target_pos[1], 'g*', ms=12)
            ax.arrow(robot_state[0], robot_state[1], 0.6*jnp.cos(robot_state[2]), 0.6*jnp.sin(robot_state[2]), 
                     head_width=0.3, color='blue', zorder=5)
            
            ax.set_title(f"Step {t} | Sticky Decision (Hysteresis) | Comp {best_idx} active")
            plt.pause(0.01)

def generate_grid_obstacles(rows, cols, spacing, start_x, start_y):
    x = jnp.linspace(start_x, start_x + (cols-1)*spacing, cols)
    y = jnp.linspace(start_y, start_y + (rows-1)*spacing, rows)
    xx, yy = jnp.meshgrid(x, y)
    return jnp.stack([xx.ravel(), yy.ravel()], axis=1)

if __name__ == "__main__":
    run_mpc_simulation()