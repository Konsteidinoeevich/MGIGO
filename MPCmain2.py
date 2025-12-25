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
    'horizon': 10,        # 稍微增加时域，让“远见”更清晰
    'dim': 2,             # 控制维度 [v, omega]
    'n_components': 6,    
    'pop_size': 60,      
    'elite_size': 25,     
    'opt_steps': 300,     
    'warmup_steps': 1000, 
    
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
# 2. Unicycle 动力学与代价函数
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
    cost_final = jnp.linalg.norm(trajectory[-1] - target) * 15.0 # 强化终点吸引
    
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
# 3. 辅助函数
# ==============================================================================

def shift_solution(mu_k, K, horizon, dim):
    mu_reshaped = mu_k.reshape(K, horizon, dim)
    mu_shifted = jnp.roll(mu_reshaped, shift=-1, axis=1)
    mu_shifted = mu_shifted.at[:, -1, :].set(jnp.zeros((dim,)))
    return mu_shifted.reshape(K, -1)

def generate_grid_obstacles(rows, cols, spacing, start_x, start_y):
    x = jnp.linspace(start_x, start_x + (cols-1)*spacing, cols)
    y = jnp.linspace(start_y, start_y + (rows-1)*spacing, rows)
    xx, yy = jnp.meshgrid(x, y)
    return jnp.stack([xx.ravel(), yy.ravel()], axis=1)

# ==============================================================================
# 4. 主循环
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
    
    robot_state = jnp.array([0.0, 0.0, 0.0])
    target_final = jnp.array([18.0, 15.0])
    history_robot = [robot_state[:2]]
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    # 定义分量轨迹的颜色
    comp_colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, K))
    
    for t in range(250):
        key, subkey = jax.random.split(key)
        
        # 障碍物与目标动态更新
        offsets = 0.25 * jnp.stack([jnp.sin(t * 0.1 + obs_phases[:, 0]), jnp.cos(t * 0.12 + obs_phases[:, 1])], axis=1)
        current_obs_pos = obs_initial_pos + offsets
        target_pos = target_final + jnp.array([jnp.sin(t*0.05), jnp.cos(t*0.05)]) * 0.3
        
        context_data = {
            'current_state': robot_state, 'target_pos': target_pos,
            'obs_pos': current_obs_pos, 'obs_radius': CONFIG['obs_radius'],
            'safe_distance': CONFIG['obs_radius'] + CONFIG['safe_margin']
        }
        
        # IGO 优化
        iter_steps = CONFIG['warmup_steps'] if t == 0 else CONFIG['opt_steps']

        t0=time.time()
        mu_k, L_inv_k, pi_k_all = igo_mog_optimizer(
            subkey, iter_steps, 0.1, K, CONFIG['pop_size'], CONFIG['elite_size'], 
            mpc_cost_fn, mu_k, L_inv_k, pi_k_all, context_data
        )
        
        mu_k.block_until_ready()
        t1=time.time()
        print(f"IGO优化耗时: {(t1-t0)*1000:.4f}ms")

        # 决策执行
        best_idx = jnp.argmax(pi_k_all)
        best_u_seq = mu_k[best_idx].reshape(CONFIG['horizon'], CONFIG['dim'])
        v_exec = jnp.tanh(best_u_seq[0, 0]) * CONFIG['max_v']
        w_exec = jnp.tanh(best_u_seq[0, 1]) * CONFIG['max_w']
        
        # 更新动力学
        new_theta = robot_state[2] + w_exec * CONFIG['dt']
        new_x = robot_state[0] + v_exec * jnp.cos(robot_state[2]) * CONFIG['dt']
        new_y = robot_state[1] + v_exec * jnp.sin(robot_state[2]) * CONFIG['dt']
        robot_state = jnp.array([new_x, new_y, new_theta])
        
        mu_k = shift_solution(mu_k, K, CONFIG['horizon'], CONFIG['dim'])
        history_robot.append(robot_state[:2])
        
        # --- 绘图逻辑 ---
        if t % 1 == 0:
            ax.cla()
            ax.set_xlim(-2, 22); ax.set_ylim(-2, 22); ax.set_aspect('equal')
            
            # 1. 障碍物
            for i in range(obs_num):
                ax.add_patch(Circle(current_obs_pos[i], CONFIG['obs_radius'], color='red', alpha=0.3))
            
            # 2. 多分量预测轨迹 (核心添加部分)
            for k in range(K):
                comp_u = mu_k[k].reshape(CONFIG['horizon'], CONFIG['dim'])
                c_vs = jnp.tanh(comp_u[:, 0]) * CONFIG['max_v']
                c_ws = jnp.tanh(comp_u[:, 1]) * CONFIG['max_w']
                c_thetas = robot_state[2] + jnp.cumsum(c_ws * CONFIG['dt'])
                c_thetas_pre = jnp.concatenate([jnp.array([robot_state[2]]), c_thetas[:-1]])
                c_traj = jnp.stack([
                    robot_state[0] + jnp.cumsum(c_vs * jnp.cos(c_thetas_pre) * CONFIG['dt']),
                    robot_state[1] + jnp.cumsum(c_vs * jnp.sin(c_thetas_pre) * CONFIG['dt'])
                ], axis=1)
                
                # 线宽由权重 pi_k_all 决定
                lw = 0.5 + 4.0 * float(pi_k_all[k])
                ax.plot(c_traj[:, 0], c_traj[:, 1], color=comp_colors[k], lw=lw, alpha=0.8)

            # 3. 历史与当前状态
            hist_np = np.array(history_robot)
            ax.plot(hist_np[:, 0], hist_np[:, 1], 'k--', alpha=0.3)
            ax.plot(target_pos[0], target_pos[1], 'g*', ms=12)
            ax.arrow(robot_state[0], robot_state[1], 0.6*jnp.cos(robot_state[2]), 0.6*jnp.sin(robot_state[2]), 
                     head_width=0.3, color='blue', zorder=5)
            
            ax.set_title(f"Step {t} | GMM Components Exploration | Best v={v_exec:.2f}")
            plt.pause(0.01)

    plt.ioff(); plt.show()

if __name__ == "__main__":
    run_mpc_simulation()