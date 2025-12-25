import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
from gmm_igo.MPCsolver import igo_mog_optimizer

# ==============================================================================
# 1. 定义 MPC 问题配置
# ==============================================================================

CONFIG = {
    'dt': 0.1,            # 时间步长
    'horizon': 8,        # 预测时域 (稍微加长以应对绕行需求)
    'dim': 2,             # 控制维度 (vx, vy)
    'n_components': 6,    # MoG 分量数 (增加分量以探索多条路径)
    'pop_size': 60,      # 种群大小 (增加以应对复杂地形)
    'elite_size': 25,     # 精英数量
    'opt_steps': 350,     # 常规帧优化步数
    'warmup_steps': 1500, # 初始冷启动 (环境复杂，需要长时间思考)
    
    # 障碍物配置
    'obs_rows': 4,
    'obs_cols': 4,
    'obs_spacing': 4.0,   # 间距 4m
    'obs_radius': 1.9,    # 半径 1.8m (非常紧密，缝隙仅 0.4m)
    'safe_margin': 0.1,   # 安全余量 (设小一点，否则缝隙会被视为“不可通过”)
}

TOTAL_DIM = CONFIG['horizon'] * CONFIG['dim']

# ==============================================================================
# 2. Cost Function (Tanh Activation 形式)
# ==============================================================================

def mpc_cost_fn(flat_actions, context):
    """
    Args:
        flat_actions: (H * D,)
        context: { ... 'obs_pos': (N, 2) ... }
    """
    actions = flat_actions.reshape((CONFIG['horizon'], CONFIG['dim']))
    x0 = context['current_pos']
    target = context['target_pos']
    dt = CONFIG['dt']
    
    # 动力学推演
    deltas = actions * dt
    trajectory = x0 + jnp.cumsum(deltas, axis=0)
    
    # --- A. 追踪代价 ---
    dist_to_target = jnp.linalg.norm(trajectory - target, axis=1)
    cost_track = jnp.sum(dist_to_target)
    
    # --- B. 障碍物代价 (Tanh 形式) ---
    obs_pos = context['obs_pos']
    obs_radius = context['obs_radius']
    safe_distance = context['safe_distance']
    
    # 计算距离矩阵: (Horizon, N_obs)
    diff = trajectory[:, None, :] - obs_pos[None, :, :]
    all_dists = jnp.linalg.norm(diff, axis=-1)
    
    # 取每一步距离最近的障碍物
    distances = jnp.min(all_dists, axis=-1)
    
    # Tanh 激活逻辑
    d_x = jnp.where(distances < obs_radius, 1.0, -1.0)
    O_x = 0.5 + 0.5 * jnp.tanh(5.0 * d_x)
    
    # 代价计算
    # 权重设高一点 (300.0)，因为缝隙很小，必须精确避让
    step_costs = O_x * (safe_distance - distances)
    obstacle_distance_cost = 100.0 * jnp.sum(step_costs)
    
    # --- C. 能量代价 ---
    cost_energy = jnp.sum(actions**2) * 0.05
    
    return cost_track + obstacle_distance_cost + cost_energy

# ==============================================================================
# 3. 辅助函数
# ==============================================================================

def shift_solution(mu_k, K, horizon, dim):
    mu_reshaped = mu_k.reshape(K, horizon, dim)
    mu_shifted = jnp.roll(mu_reshaped, shift=-1, axis=1)
    mu_shifted = mu_shifted.at[:, -1, :].set(jnp.zeros((dim,)))
    return mu_shifted.reshape(K, -1)

def generate_grid_obstacles(rows, cols, spacing, start_x, start_y):
    """生成 4x4 的网格坐标"""
    x = jnp.linspace(start_x, start_x + (cols-1)*spacing, cols)
    y = jnp.linspace(start_y, start_y + (rows-1)*spacing, rows)
    # 生成网格点
    xx, yy = jnp.meshgrid(x, y)
    # 展平为 (N, 2)
    coords = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    return coords

# ==============================================================================
# 4. 主循环
# ==============================================================================

def run_mpc_simulation():
    print(">>> Initialize MPC with 4x4 Moving Obstacle Grid...")
    key = jax.random.PRNGKey(999)
    
    K = CONFIG['n_components']
    D_total = TOTAL_DIM
    
    # --- 1. 初始化障碍物网格 ---
    # 机器人从 (0,0) 出发，目标在 (20, 15)
    # 我们把障碍物阵列放在中间，例如从 x=4, y=2 开始
    obs_initial_pos = generate_grid_obstacles(
        rows=CONFIG['obs_rows'], 
        cols=CONFIG['obs_cols'], 
        spacing=CONFIG['obs_spacing'], 
        start_x=2.0, 
        start_y=2.0
    )
    obs_num = obs_initial_pos.shape[0]
    
    # 为每个障碍物生成独立的随机相位，用于微动
    key, subkey = jax.random.split(key)
    obs_phases = jax.random.uniform(subkey, (obs_num, 2)) * 2 * jnp.pi
    
    # --- 2. 初始化优化器 ---
    mu_k = jax.random.normal(key, (K, D_total)) * 0.5
    L_inv_k = jnp.stack([jnp.eye(D_total) * 3.0 for _ in range(K)])
    pi_k_all = jnp.ones(K) / K
    
    robot_pos = jnp.array([0.0, 0.0]) # 稍微调整起点y，正对缝隙或障碍
    target_final = jnp.array([18.5, 14.0]) # 放在障碍阵列后面
    
    # 绘图
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    history_robot = [robot_pos]
    
    safe_distance_val = CONFIG['obs_radius'] + CONFIG['safe_margin']
    print(f">>> Config: Radius={CONFIG['obs_radius']}, Spacing={CONFIG['obs_spacing']}, Gap={CONFIG['obs_spacing'] - 2*CONFIG['obs_radius']:.2f}m")
    
    for t in range(150):
        key, subkey = jax.random.split(key)
        
        # --- A. 障碍物动态更新 (轻微行动) ---
        # 使用 sin/cos 让每个障碍物在原点附近画微小的圈或椭圆
        # 振幅 0.2m
        offsets = 0.2 * jnp.stack([
            jnp.sin(t * 0.1 + obs_phases[:, 0]),
            jnp.cos(t * 0.15 + obs_phases[:, 1])
        ], axis=1)
        
        current_obs_pos = obs_initial_pos + offsets
        
        # 目标微动
        target_pos = target_final + jnp.array([jnp.sin(t*0.05)*0.5, jnp.cos(t*0.05)*0.5])
        
        context_data = {
            'current_pos': robot_pos,
            'target_pos': target_pos,
            'obs_pos': current_obs_pos,
            'obs_radius': CONFIG['obs_radius'],
            'safe_distance': safe_distance_val
        }
        
        # --- B. 优化 ---
        current_steps = CONFIG['warmup_steps'] if t == 0 else CONFIG['opt_steps']
        
        t0 = time.time()
        # 学习率 delta_t 设为 0.2
        mu_k, L_inv_k, pi_k_all = igo_mog_optimizer(
            subkey, 
            current_steps, 
            0.1, 
            K, CONFIG['pop_size'], CONFIG['elite_size'], 
            mpc_cost_fn, 
            mu_k, L_inv_k, pi_k_all, 
            context_data
        )
        mu_k.block_until_ready()
        t1 = time.time()
        
        # --- C. 执行 ---
        best_comp_idx = jnp.argmax(pi_k_all)
        best_action_seq = mu_k[best_comp_idx].reshape(CONFIG['horizon'], CONFIG['dim'])
        current_action = best_action_seq[0]
        
        robot_pos = robot_pos + current_action * CONFIG['dt']
        mu_k = shift_solution(mu_k, K, CONFIG['horizon'], CONFIG['dim'])
        history_robot.append(robot_pos)
        
        print(f"Step {t:03d} | Calc: {(t1-t0)*1000:.1f}ms | Pos: {robot_pos}")
        
        # --- D. 绘图 ---
        if t % 1 == 0:
            ax.cla()
            ax.set_xlim(-2, 22)
            ax.set_ylim(-2, 22)
            ax.set_aspect('equal')
            
            # 画障碍物
            for i in range(obs_num):
                # 实体 (深红色)
                c_real = Circle(current_obs_pos[i], CONFIG['obs_radius'], color='firebrick', alpha=0.6, zorder=2)
                ax.add_patch(c_real)
                # 安全边界 (虚线)
                c_safe = Circle(current_obs_pos[i], safe_distance_val, color='gray', fill=False, linestyle=':', alpha=0.5)
                ax.add_patch(c_safe)
            
            # 历史轨迹
            hist_arr = np.array(history_robot)
            ax.plot(hist_arr[:, 0], hist_arr[:, 1], 'b.-', alpha=0.5, linewidth=1, label='History')
            
            # 机器人与目标
            ax.plot(robot_pos[0], robot_pos[1], 'bo', markersize=8, zorder=3, label='Robot')
            ax.plot(target_pos[0], target_pos[1], 'g*', markersize=12, zorder=3, label='Target')
            
            # 预测轨迹
            pred_traj = robot_pos + jnp.cumsum(best_action_seq * CONFIG['dt'], axis=0)
            ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'c-', linewidth=2, zorder=3, label='MPC Plan')
            
            ax.legend(loc='upper left', fontsize='small')
            ax.set_title(f"Step {t} | Grid 4x4 Dynamic | Gap ~0.4m")
            plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_mpc_simulation()