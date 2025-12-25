import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
import functools

from gmm_igo.MPCsolver import igo_mog_optimizer

# ==============================================================================
# 1. 配置参数
# ==============================================================================
CONFIG = {
    'dt': 0.1,
    'horizon': 12,        
    'dim': 2,
    'n_components': 2,    
    'pop_size': 60,       
    'elite_size': 25,     
    'opt_steps': 100,     
    'warmup_steps': 1000,
    'max_v': 2.8,         
    'max_w': 2.0,         
    'obs_radius': 1.6,    
    'safe_margin': 0.2,   
}

TOTAL_DIM = CONFIG['horizon'] * CONFIG['dim']
N_TASKS = 3 

# ==============================================================================
# 2. 核心逻辑：对障碍物轨迹建模的代价函数
# ==============================================================================

parallel_igo_solver = jax.vmap(
    igo_mog_optimizer,
    in_axes=(0, None, None, None, None, None, None, 0, 0, 0, {
        'current_state': 0, 
        'target_pos': 0, 
        'obs_traj': None,  # 修改点：传入预测轨迹 (Horizon, Num_Obs, 2)
        'obs_radius': None,
        'safe_distance': None,
        'strategy_id': 0, 
    })
)

@jax.jit
def mpc_cost_fn(flat_actions, context):
    raw_actions = flat_actions.reshape((CONFIG['horizon'], CONFIG['dim']))
    vs = jnp.tanh(raw_actions[:, 0]) * CONFIG['max_v']
    ws = jnp.tanh(raw_actions[:, 1]) * CONFIG['max_w']
    x0, y0, theta0 = context['current_state']
    
    thetas = theta0 + jnp.cumsum(ws * CONFIG['dt'])
    thetas_prev = jnp.concatenate([jnp.array([theta0]), thetas[:-1]])
    trajectory = jnp.stack([
        x0 + jnp.cumsum(vs * jnp.cos(thetas_prev) * CONFIG['dt']),
        y0 + jnp.cumsum(vs * jnp.sin(thetas_prev) * CONFIG['dt'])
    ], axis=1)

    # 关键修改：计算每一时刻轨迹点与对应时刻障碍物位置的距离
    obs_traj = context['obs_traj'] # (H, N_obs, 2)
    safe_dist = context['safe_distance']
    
    # 距离计算：(H, 1, 2) 与 (H, N_obs, 2) 广播
    diff = trajectory[:, None, :] - obs_traj
    dists = jnp.linalg.norm(diff, axis=-1)
    instant_risk_base = 1e3 * jnp.maximum(0, safe_dist - dists)

    # 保持原始非马尔可夫逻辑
    s_id = context['strategy_id']
    risk_0 = jnp.sum(instant_risk_base, axis=1)
    cos_t = jnp.cos(thetas_prev)[:, None]
    sin_t = jnp.sin(thetas_prev)[:, None]
    local_y = -diff[..., 0] * sin_t + diff[..., 1] * cos_t
    left_mask = jnp.where(local_y > 0, 15.0, 1.0) 
    risk_1 = jnp.sum(instant_risk_base * left_mask, axis=1)
    time_decay = jnp.exp(-0.5 * jnp.arange(CONFIG['horizon']))
    risk_2 = jnp.sum(instant_risk_base, axis=1) * time_decay

    selected_instant = jnp.where(s_id == 0, risk_0, jnp.where(s_id == 1, risk_1, risk_2))
    cum_risk = jnp.flip(jnp.cumsum(jnp.flip(selected_instant)))
    dist_to_target = jnp.linalg.norm(trajectory - context['target_pos'], axis=1)
    cum_dist = jnp.flip(jnp.cumsum(jnp.flip(dist_to_target)))

    return jnp.sum(cum_risk) * 6000.0 + jnp.sum(cum_dist) * 1.5 + jnp.sum(jnp.diff(ws)**2) * 2.0

# ==============================================================================
# 3. 动态障碍物预测器 (建模运动逻辑)
# ==============================================================================

def predict_obs_trajectories(t_start, obs_base, horizon, dt):
    """
    根据运动学逻辑预测未来 Horizon 步的障碍物位置
    """
    t_steps = t_start + jnp.arange(horizon) * dt
    
    # 1-4: 往复运动
    move_osc = 3.5 * jnp.sin(1.0 * t_steps[:, None] + jnp.arange(4)) # (H, 4)
    traj_osc = obs_base[:4][None, :, :] + jnp.stack([move_osc, jnp.zeros_like(move_osc)], axis=-1)
    
    # 5-7: 圆周运动
    angles = 1.2 * t_steps[:, None] + jnp.arange(3)
    traj_cir = obs_base[4:7][None, :, :] + jnp.stack([2.0 * jnp.cos(angles), 2.0 * jnp.sin(angles)], axis=-1)
    
    # 8-10: 线性漂移 (忽略实时随机噪声的预测，建模其平均趋势)
    move_drift = 1.5 * jnp.cos(0.8 * t_steps[:, None])
    traj_drift = obs_base[7:][None, :, :] + jnp.stack([move_drift, move_drift], axis=-1)
    
    return jnp.concatenate([traj_osc, traj_cir, traj_drift], axis=1)

# ==============================================================================
# 4. 主循环
# ==============================================================================

def get_trajectory_np(robot_state, action_seq):
    u = action_seq.reshape(CONFIG['horizon'], CONFIG['dim'])
    vs, ws = np.tanh(u[:, 0]) * CONFIG['max_v'], np.tanh(u[:, 1]) * CONFIG['max_w']
    x, y, th = robot_state
    pts = []
    for i in range(CONFIG['horizon']):
        x += vs[i] * np.cos(th) * CONFIG['dt']; y += vs[i] * np.sin(th) * CONFIG['dt']; th += ws[i] * CONFIG['dt']
        pts.append([x, y])
    return np.array(pts)

def shift_solution_batch(mu_k_batch):
    mu_reshaped = mu_k_batch.reshape(N_TASKS, CONFIG['n_components'], CONFIG['horizon'], CONFIG['dim'])
    mu_shifted = jnp.roll(mu_reshaped, shift=-1, axis=2)
    mu_shifted = mu_shifted.at[:, :, -1, :].set(0.0)
    return mu_shifted.reshape(N_TASKS, CONFIG['n_components'], -1)

def run_heterogeneous_functional_mpc():
    key = jax.random.PRNGKey(42)
    K = CONFIG['n_components']
    mu_k_batch = jax.random.normal(key, (N_TASKS, K, TOTAL_DIM)) * 0.1
    L_inv_k_batch = jnp.stack([jnp.stack([jnp.eye(TOTAL_DIM) * 2.0 for _ in range(K)]) for _ in range(N_TASKS)])
    pi_k_batch = jnp.ones((N_TASKS, K)) / K
    
    robot_state = jnp.array([0.0, 0.0, 0.0])
    target_final = jnp.array([24.0, 24.0])
    obs_base = jnp.array([
        [6., 6.], [12., 12.], [18., 18.], [5., 14.], 
        [14., 5.], [10., 18.], [18., 10.], 
        [8., 10.], [13., 15.], [16., 8.]
    ])

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for t in range(500):
        # 1. 获取未来时域内障碍物的真实/预测轨迹
        current_time = t * CONFIG['dt']
        obs_trajs = predict_obs_trajectories(current_time, obs_base, CONFIG['horizon'], CONFIG['dt'])
        
        # 为了模拟环境，给当前时刻的位置加一点随机噪声
        key, subkey = jax.random.split(key)
        noise = jax.random.uniform(subkey, (10, 2), minval=-0.2, maxval=0.2)
        current_obs_pos = obs_trajs[0] + noise

        # 2. 并行 MPC 求解
        batch_context = {
            'current_state': jnp.stack([robot_state] * N_TASKS),
            'target_pos': jnp.stack([target_final] * N_TASKS),
            'obs_traj': obs_trajs, # 传入完整预测序列
            'obs_radius': CONFIG['obs_radius'],
            'safe_distance': CONFIG['obs_radius'] + CONFIG['safe_margin'],
            'strategy_id': jnp.array([0, 1, 2])
        }

        key, subkey = jax.random.split(key)
        mu_k_batch, L_inv_k_batch, pi_k_batch = parallel_igo_solver(
            jax.random.split(subkey, N_TASKS), CONFIG['opt_steps'], 0.1, K, 
            CONFIG['pop_size'], CONFIG['elite_size'], mpc_cost_fn, 
            mu_k_batch, L_inv_k_batch, pi_k_batch, batch_context
        )

        # 3. 决策执行 (Strategy 0 裁判)
        eval_scores = [mpc_cost_fn(mu_k_batch[i, jnp.argmax(pi_k_batch[i])], 
                       {**batch_context, 'strategy_id': 0, 'current_state': robot_state, 'target_pos': target_final}) 
                       for i in range(N_TASKS)]
        best_idx = np.argmin(eval_scores)
        
        u_exec = mu_k_batch[best_idx, jnp.argmax(pi_k_batch[best_idx])].reshape(CONFIG['horizon'], CONFIG['dim'])[0]
        v_e, w_e = jnp.tanh(u_exec[0])*CONFIG['max_v'], jnp.tanh(u_exec[1])*CONFIG['max_w']
        robot_state = jnp.array([
            robot_state[0] + v_e * jnp.cos(robot_state[2]) * CONFIG['dt'],
            robot_state[1] + v_e * jnp.sin(robot_state[2]) * CONFIG['dt'],
            robot_state[2] + w_e * CONFIG['dt']
        ])

        if t % 1 == 0:
            ax.cla()
            ax.set_xlim(-2, 28); ax.set_ylim(-2, 28); ax.set_aspect('equal')
            for op in current_obs_pos:
                ax.add_patch(Circle(op, CONFIG['obs_radius'], color='gray', alpha=0.3))
            for i in range(N_TASKS):
                traj_np = get_trajectory_np(robot_state, mu_k_batch[i, jnp.argmax(pi_k_batch[i])])
                ax.plot(traj_np[:,0], traj_np[:,1], lw=2.5 if i==best_idx else 0.8)
            ax.plot(target_final[0], target_final[1], 'g*', ms=15)
            plt.pause(0.01)

        mu_k_batch = shift_solution_batch(mu_k_batch)

if __name__ == "__main__":
    run_heterogeneous_functional_mpc()