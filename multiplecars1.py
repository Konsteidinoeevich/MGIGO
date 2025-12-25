# mainM2_3Car_Final_Solution_with_Plot.py - 3车轨迹规划 MPC (添加绘图功能)

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax 
import functools 
import time 
from typing import Callable, Tuple, List, Any

# 导入 Matplotlib (新增)
import matplotlib.pyplot as plt 

# 假设 MPCsolverM.py 位于当前目录下
from gmm_igo.MPCsolverM import mmog_igo_optimizer_mpc 

# ======================================================================
# I. 配置参数
# ======================================================================

SEED = 42
DELTA_T = 0.1
DYN_DT = 0.1
M_MOG = 3
K_COMP = 8
B_SAMPLES = 60    
B_0_ELITE = 25    
T_0_RESTART = 100 

PRED_STEPS = 6    
CONTROL_DIM = 2   
D_I = PRED_STEPS * CONTROL_DIM 

DIMS_TUPLE = (D_I, D_I, D_I) 
D_TOTAL = sum(DIMS_TUPLE)
D_MAX = max(DIMS_TUPLE)

T_IGO_ITER = 600   
T_MPC_RUNS = 80   

# 成本函数权重和常量
W_TARGET = 10.0
W_SMOOTH = 0.5
W_COLL = 100.0
D_SAFE_SQ = 1.0 

# ======================================================================
# II. 车辆运动学模型 (保持不变)
# ======================================================================

@functools.partial(jit, static_argnames=['delta_t'])
def step_fn(state, control_input, delta_t: float = DYN_DT):
    x, y, theta, v = state
    a, w = control_input
    v_next = jnp.clip(v + a * delta_t, 0.0, 5.0) 
    theta_next = theta + w * delta_t
    v_avg = (v + v_next) / 2.0
    x_next = x + v_avg * jnp.cos(theta) * delta_t
    y_next = y + v_avg * jnp.sin(theta) * delta_t
    new_state = jnp.array([x_next, y_next, theta_next, v_next])
    new_xy = jnp.array([x_next, y_next])
    return new_state, new_xy

@functools.partial(jit, static_argnames=['pred_steps', 'delta_t'])
def predict_trajectory(initial_state, control_sequence, pred_steps, delta_t):
    controls = control_sequence.reshape((pred_steps, CONTROL_DIM))
    def scan_step(state, control_input):
        new_state, new_xy = step_fn(state, control_input, delta_t)
        return new_state, new_xy
    initial_xy = initial_state[:2].reshape((1, 2)) 
    final_state, xy_trajectory = lax.scan(
        scan_step, initial_state, controls
    )
    return jnp.concatenate([initial_xy, xy_trajectory], axis=0) 

# ======================================================================
# III. 修正后的成本函数 (保持不变)
# ======================================================================

def create_3car_mpc_fitness_fn(M_MOG: int, dims_tuple: Tuple[int, ...], PRED_STEPS: int):
    
    D_MAX = max(dims_tuple)
    D_ARRAY = jnp.array(dims_tuple)

    @jax.jit
    def fitness_fn_total(samples_overall, context):
        
        samples_M_padded = samples_overall.reshape((M_MOG, D_MAX))
        context_M = context.reshape((M_MOG, 6))
        
        initial_states = context_M[:, :4] 
        targets = context_M[:, 4:] 

        def vmap_body(m_idx):
            sample_row = samples_M_padded[m_idx] 
            control_sequence = lax.dynamic_slice(
                sample_row, 
                start_indices=(0,), 
                slice_sizes=(D_I,) 
            )
            
            initial_state = initial_states[m_idx]
            target_xy = targets[m_idx]
            
            traj_xy = predict_trajectory(initial_state, control_sequence, PRED_STEPS, DYN_DT)
            
            final_xy = traj_xy[-1]
            f_target = jnp.sum((final_xy - target_xy)**2)
            controls = control_sequence.reshape((PRED_STEPS, CONTROL_DIM))
            f_smooth = jnp.sum(jnp.diff(controls, axis=0)**2) 
            f_local = W_TARGET * f_target + W_SMOOTH * f_smooth
            
            return f_local, traj_xy

        f_local_M, traj_xy_M = vmap(vmap_body)(jnp.arange(M_MOG)) 
        f_local_total = jnp.sum(f_local_M)

        f_coll_total = 0.0
        for i in range(M_MOG):
            for j in range(i + 1, M_MOG):
                traj_i = traj_xy_M[i]
                traj_j = traj_xy_M[j]
                dist_sq = jnp.sum((traj_i - traj_j)**2, axis=1) 
                violation = jnp.clip(D_SAFE_SQ - dist_sq, a_min=0.0)
                f_coll_total += jnp.sum(violation**2) 
                
        f_total = f_local_total + W_COLL * f_coll_total
        return f_total

    return fitness_fn_total

# ======================================================================
# IV. 初始化函数 (保持不变)
# ======================================================================

def initialize_params_mmog_heterogeneous(key, M: int, K: int, dims: Tuple[int, ...]):
    D_max: int = max(dims)
    initial_mu_list: List[jnp.ndarray] = []
    initial_L_inv_list: List[jnp.ndarray] = []
    keys = random.split(key, M)

    for m in range(M):
        D_m = dims[m]
        key_mu, key_L = random.split(keys[m])
        mu_m_actual = random.uniform(key_mu, (K+1, D_m), minval=-0.2, maxval=0.2)
        mu_m_padded = jnp.pad(mu_m_actual, ((0, 0), (0, D_max - D_m)), mode='constant')
        initial_mu_list.append(mu_m_padded)

        L_inv_template = jnp.eye(D_m) * 2.0 
        L_inv_k_all = jnp.stack([L_inv_template] * (K + 1)) 
        L_inv_m_padded = jnp.pad(L_inv_k_all, ((0, 0), (0, D_max - D_m), (0, D_max - D_m)), mode='constant')
        initial_L_inv_list.append(L_inv_m_padded)

    initial_v_k = jnp.zeros((M, K))
    initial_mu_k_stacked = jnp.stack(initial_mu_list)     
    initial_L_inv_k_stacked = jnp.stack(initial_L_inv_list) 
    
    return initial_mu_k_stacked, initial_L_inv_k_stacked, initial_v_k, D_max

# ======================================================================
# V. 主程序运行逻辑 (MPC 滚动和 Warm Start)
# ======================================================================

if __name__ == '__main__':
    
    fitness_fn_total = create_3car_mpc_fitness_fn(M_MOG, DIMS_TUPLE, PRED_STEPS)

    key = random.PRNGKey(SEED)
    key_init, key_run = random.split(key)
    
    initial_mu_k, initial_L_inv_k, initial_v_k, D_max_check = initialize_params_mmog_heterogeneous(
        key_init, M_MOG, K_COMP, DIMS_TUPLE
    )
    
    print("--- 3 车轨迹规划 (M-MoG IGO MPC) 测试开始 ---")
    print(f"车辆 M={M_MOG}, 维度 D={DIMS_TUPLE}, 预测步长 T_p={PRED_STEPS}")
    print(f"IGO 迭代 T={T_IGO_ITER}, MPC 步长 T_MPC={T_MPC_RUNS}, 动力学步长 dt={DYN_DT}")
    
    # 初始状态和目标
    initial_context_data = jnp.array([
        [0.0, 0.0, 0.0, 1.0, 20.0, 5.0], 
        [0.0, 5.0, 0.0, 1.0, 20.0, 0.0], 
        [5.0, 2.5, jnp.pi/2, 1.0, 20.0, 2.5],
    ])
    
    # --- 绘图数据收集初始化 (新增) ---
    # 存储 M 辆车的 (x, y) 轨迹
    # 转换为 Python 列表，方便后续 NumPy/Matplotlib 处理
    trajectory_data = [[] for _ in range(M_MOG)]
    
    initial_xy = initial_context_data[:, :2]
    targets_xy = initial_context_data[:, 4:]
    
    for i in range(M_MOG):
        trajectory_data[i].append(initial_xy[i].tolist())
    # -----------------------------------

    # --- IGO Warm Start 初始化 ---
    mu_k_current = initial_mu_k  
    L_inv_k_current = initial_L_inv_k
    v_k_current = initial_v_k 

    current_initial_states = initial_context_data[:, :4]
    targets = initial_context_data[:, 4:]
    current_context = initial_context_data.flatten()
    
    @vmap
    def apply_control_and_step(state_and_control):
         state, control = state_and_control[:-CONTROL_DIM], state_and_control[-CONTROL_DIM:]
         new_state, _ = step_fn(state, control, DYN_DT)
         return new_state 

    # --- MPC 滚动循环 ---
    for mpc_step in range(T_MPC_RUNS):
        key_run, subkey = random.split(key_run)
        print(f"\n--- MPC Step {mpc_step + 1}/{T_MPC_RUNS} ---")
        
        start_time = time.time()
        
        # 1. IGO 优化 (匹配 3 个返回值)
        final_mu_k, final_L_inv_k, final_pi_k_all = mmog_igo_optimizer_mpc(
            subkey, T_IGO_ITER, DELTA_T, M_MOG, K_COMP, B_SAMPLES, B_0_ELITE, 
            DIMS_TUPLE, 
            T_0_RESTART, fitness_fn_total,
            mu_k_current, L_inv_k_current, v_k_current, 
            context=current_context 
        )
        final_mu_k.block_until_ready()
        elapsed_time = time.time() - start_time
        
        # 2. 从 final_pi_k_all 提取最优控制序列
        best_comp_indices = jnp.argmax(final_pi_k_all[:, :-1], axis=1) 
        mu_star_M_padded = final_mu_k[jnp.arange(M_MOG), best_comp_indices]
        
        # 3. 提取并应用第一步控制
        control_vec = mu_star_M_padded[:, :CONTROL_DIM] 
        states_and_controls = jnp.concatenate([current_initial_states, control_vec], axis=1)
        new_initial_states = apply_control_and_step(states_and_controls)

        # 4. 更新 Warm Start 参数
        mu_star_M_shift = mu_star_M_padded[:, CONTROL_DIM:D_I] 
        last_control = mu_star_M_padded[:, D_I - CONTROL_DIM : D_I]
        mu_star_M_new_sequence = jnp.concatenate([mu_star_M_shift, last_control], axis=1) 
        
        mu_k_current = final_mu_k.at[jnp.arange(M_MOG), 0, :D_I].set(mu_star_M_new_sequence) 
        L_inv_k_current = final_L_inv_k
        # v_k_current 保持不变

        # 5. 更新 Context 和轨迹数据 (新增)
        current_initial_states = new_initial_states
        current_context = jnp.concatenate([current_initial_states, targets], axis=1).flatten()
        
        # 收集当前位置
        current_xy = current_initial_states[:, :2].tolist() 
        for i in range(M_MOG):
            trajectory_data[i].append(current_xy[i])
        
        # 6. 结果打印
        f_mu_star = fitness_fn_total(mu_star_M_padded.flatten(), current_context)
        print(f"  IGO 耗时: {elapsed_time:.4f}s, 最终成本: {f_mu_star:.4e}")
        
        final_pos = current_initial_states[:, :2]
        dist_to_target = jnp.linalg.norm(final_pos - targets, axis=1)
        print(f"  当前距离目标: {dist_to_target}")
        
        if jnp.all(dist_to_target < 0.5):
            print("\n所有车辆接近目标 (距离 < 1.5m)，MPC 结束。")
            break
        
    print("\n--- MPC 滚动优化完成 ---")

    # ======================================================================
    # VI. 轨迹可视化 (新增)
    # ======================================================================
    
    print("\n--- 正在生成车辆轨迹图 ---")

    # Matplotlib 绘图配置
    plt.figure(figsize=(10, 8))
    
    # 定义绘图颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 蓝、橙、绿
    
    for i in range(M_MOG):
        # 将列表数据转换为 NumPy 数组
        traj = jnp.array(trajectory_data[i])
        
        # 绘制轨迹
        plt.plot(traj[:, 0], traj[:, 1], 
                 linestyle='-', 
                 marker='.', 
                 color=colors[i], 
                 alpha=0.8, 
                 label=f'Vehicle {i+1} Trajectory')
        
        # 标记起始点 (Start)
        plt.plot(traj[0, 0], traj[0, 1], 
                 marker='o', 
                 markersize=10, 
                 color=colors[i], 
                 markerfacecolor='none', 
                 linestyle='none')
        plt.text(traj[0, 0], traj[0, 1], f' Start {i+1}', fontsize=9, verticalalignment='bottom')

        # 标记目标点 (Goal)
        target_x, target_y = targets_xy[i].tolist()
        plt.plot(target_x, target_y, 
                 marker='*', 
                 markersize=12, 
                 color=colors[i], 
                 linestyle='none', 
                 label=f'Target {i+1}')
        plt.text(target_x, target_y, f' Target {i+1}', fontsize=9, verticalalignment='bottom')


    plt.title('3-Car Multi-Objective IGO MPC Trajectories')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # 确保X和Y轴比例相等，避免轨迹变形
    plt.show()