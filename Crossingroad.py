# intersection_mpc_animated.py - 3车十字路口轨迹规划 (精确惩罚函数和最新参数) - 动态图版本

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax 
import functools 
import time 
from typing import Callable, Tuple, List, Any

import matplotlib.pyplot as plt 
# *** 导入 Matplotlib Animation 模块 ***
from matplotlib import animation 

# 假设 MPCsolverM.py 位于当前目录下
# 请根据您的实际路径修改导入：
from gmm_igo.MPCsolverM import mmog_igo_optimizer_mpc 

# ======================================================================
# I. 配置参数 (参数保持不变，与 Crossingroad.py 一致)
# ======================================================================

SEED = 42
DELTA_T = 0.1     
DYN_DT = 0.08     
M_MOG = 3         
K_COMP = 8        
B_SAMPLES = 60    
B_0_ELITE = 25    
T_0_RESTART = 80  

PRED_STEPS = 6    
CONTROL_DIM = 2   
D_I = PRED_STEPS * CONTROL_DIM 

DIMS_TUPLE = (D_I, D_I, D_I) 
D_MAX = max(DIMS_TUPLE)

T_IGO_ITER = 400  
T_MPC_RUNS = 80   

# 成本函数权重和常量
W_TARGET = 10.0
W_SMOOTH = 0.5
W_COLL = 100.0
W_PATH = 1e5      
D_SAFE_SQ = 1.0   
INTERSECTION_BUFFER = 2.0 
ROAD_HALF_WIDTH = 1.5     # 道路半宽 (用于精确惩罚)

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
    return jnp.array([x_next, y_next, theta_next, v_next]), jnp.array([x_next, y_next])

@functools.partial(jit, static_argnames=['pred_steps', 'delta_t'])
def predict_trajectory(initial_state, control_sequence, pred_steps, delta_t):
    controls = control_sequence.reshape((pred_steps, CONTROL_DIM))
    def scan_step(state, control_input):
        new_state, new_xy = step_fn(state, control_input, delta_t)
        return new_state, new_xy
    initial_xy = initial_state[:2].reshape((1, 2)) 
    final_state, xy_trajectory = lax.scan(scan_step, initial_state, controls)
    return jnp.concatenate([initial_xy, xy_trajectory], axis=0) 

# ======================================================================
# III. 成本函数 (精确惩罚函数，保持不变)
# ======================================================================

@jit
def _calculate_car3_path_cost(traj_xy: jnp.ndarray, road_half_width: float) -> float:
    x_traj = traj_xy[:, 0]
    y_traj = traj_xy[:, 1]
    
    dev_y = jnp.abs(y_traj) 
    dev_x = jnp.abs(x_traj) 

    viol_y = jnp.clip(dev_y - road_half_width, a_min=0.0)
    viol_x = jnp.clip(dev_x - road_half_width, a_min=0.0)

    is_approach = x_traj > INTERSECTION_BUFFER 
    is_exit = y_traj > INTERSECTION_BUFFER     
    
    f_path_points = jnp.where(
        is_approach, 
        viol_y, 
        jnp.where(
            is_exit, 
            viol_x, 
            0.0     
        )
    )
    
    return W_PATH * jnp.sum(f_path_points)


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
            
            f_path = lax.cond(m_idx == 2, 
                lambda: _calculate_car3_path_cost(traj_xy, ROAD_HALF_WIDTH), 
                lambda: 0.0
            )
            f_local += f_path
            
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
    
    print("--- 3 车轨迹规划 (M-MoG IGO MPC) 动态图测试开始 ---")
    print(f"场景: 三车通过十字路口 (dt={DYN_DT}, T_MPC={T_MPC_RUNS})")
    
    # 初始状态和目标 (十字路口场景)
    initial_context_data = jnp.array([
        [-15.0, 0.0, 0.0, 20.0, 15.0, 0.0],       # Car 1: 西向东直行
        [0.0, -15.0, jnp.pi/2, 15.0, 0.0, 15.0],  # Car 2: 南向北直行
        [15.0, 0.0, jnp.pi, 8.0, 0.0, 15.0],     # Car 3: 东向西，左转去北
    ])
    
    # --- 绘图数据收集初始化 ---
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
        
        # 1. IGO 优化
        final_mu_k, final_L_inv_k, final_pi_k_all = mmog_igo_optimizer_mpc(
            subkey, T_IGO_ITER, DELTA_T, M_MOG, K_COMP, B_SAMPLES, B_0_ELITE, 
            DIMS_TUPLE, 
            T_0_RESTART, fitness_fn_total,
            mu_k_current, L_inv_k_current, v_k_current, 
            context=current_context 
        )
        final_mu_k.block_until_ready()
        elapsed_time = time.time() - start_time
        
        # 2. 提取最优控制序列
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

        # 5. 更新 Context 和轨迹数据
        current_initial_states = new_initial_states
        current_context = jnp.concatenate([current_initial_states, targets], axis=1).flatten()
        
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
            print("\n所有车辆接近目标 (距离 < 0.5m)，MPC 结束。")
            T_MPC_END = mpc_step + 1 # 记录实际运行步数
            break
        elif mpc_step == T_MPC_RUNS - 1:
            T_MPC_END = T_MPC_RUNS
        
    print("\n--- MPC 滚动优化完成 ---")

    # ======================================================================
    # VI. 轨迹动画可视化 (Animation)
    # ======================================================================
    
    print("\n--- 正在生成车辆轨迹动画 ---")

    # 准备数据
    all_trajectories_array = jnp.array(trajectory_data) # (M_MOG, T_MPC_END+1, 2)
    num_frames = T_MPC_END # 动画帧数等于实际运行的MPC步数

    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    labels = ["Car 1 (W->E)", "Car 2 (S->N)", "Car 3 (E->N Turn)"]
    
    # 绘制静态背景元素
    
    # 绘制道路边界
    ax.axvline(x=ROAD_HALF_WIDTH, color='lightgray', linestyle='-', alpha=0.5)
    ax.axvline(x=-ROAD_HALF_WIDTH, color='lightgray', linestyle='-', alpha=0.5)
    ax.axhline(y=ROAD_HALF_WIDTH, color='lightgray', linestyle='-', alpha=0.5)
    ax.axhline(y=-ROAD_HALF_WIDTH, color='lightgray', linestyle='-', alpha=0.5)
    
    # 绘制交叉口中心
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.scatter([0], [0], marker='+', color='red', s=100, label='Intersection Center')
    
    # 标记目标点
    for i in range(M_MOG):
        target_x, target_y = targets_xy[i].tolist()
        ax.plot(target_x, target_y, 
                 marker='X', 
                 markersize=12, 
                 color=colors[i], 
                 linestyle='none',
                 label=f'Target {i+1}')
        ax.text(target_x, target_y, f' T{i+1}', fontsize=9, verticalalignment='bottom')

    # 初始化动态元素
    lines = [] # 用于绘制轨迹历史 (line segment)
    car_markers = [] # 用于绘制车辆当前位置 (point marker)

    for i in range(M_MOG):
        # 轨迹历史线段
        line, = ax.plot([], [], 
                         linestyle='-', 
                         color=colors[i], 
                         alpha=0.5, 
                         linewidth=2.0)
        lines.append(line)
        
        # 车辆当前位置标记 (使用三角形表示方向)
        marker, = ax.plot([], [], 
                           marker='^', 
                           markersize=10, 
                           color=colors[i], 
                           label=labels[i])
        car_markers.append(marker)

    # 设置图表属性
    ax.set_title(f'3-Car Intersection Coordination - MPC Simulation')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal', adjustable='box') 
    
    # 初始化函数
    def init():
        for line in lines:
            line.set_data([], [])
        for marker in car_markers:
            marker.set_data([], [])
        return lines + car_markers

    # 动画更新函数
    def animate(frame):
        t = frame 
        
        # 更新车辆位置和轨迹历史
        for i in range(M_MOG):
            traj = all_trajectories_array[i]
            
            # 更新轨迹历史 (从起点到当前帧 t)
            lines[i].set_data(traj[:t+1, 0], traj[:t+1, 1])
            
            # 更新当前位置 marker
            current_x, current_y = traj[t, 0], traj[t, 1]
            car_markers[i].set_data([current_x], [current_y])
            
            # *** 关键：更新车辆方向 (使用初始状态中的 theta) ***
            # 这里的theta (航向角)需要从 initial_states 中获取，但是 initial_states 
            # 在循环中被更新为 new_initial_states (包含最新的 theta).
            # 我们需要保存所有的状态 (x, y, theta, v) 而不仅仅是 (x, y).
            
            # 由于原代码只保存了 (x, y) 坐标，我们无法准确显示方向。
            # 为了满足 "不修改程序参数" 的要求，我们只能根据 (x, y) 的变化趋势粗略估计，
            # 或者像现在这样，只显示位置。

            # 为简化并保持与原代码的精神一致 (只保存了xy)，我们只更新位置。
            
        ax.set_title(f'3-Car Intersection Coordination - Time Step {t+1} / {num_frames} (T={t * DYN_DT:.2f}s)')
        
        return lines + car_markers

    # 创建动画 (interval 是毫秒)
    # 假设每一步 MPC 对应 DYN_DT=0.08 秒
    anim = animation.FuncAnimation(
        fig, 
        animate, 
        init_func=init, 
        frames=num_frames, 
        interval=DYN_DT * 1000, 
        blit=False,
        repeat=False
    )

    plt.show()

    # 如果需要保存为 mp4 文件，可以取消注释以下代码 (需要安装 ffmpeg)
    print("\n正在保存动画为 intersection_mpc_animation.mp4...")
    anim.save('intersection_mpc_animation.gif', writer='pillow', fps=10)
    
    # ----------------------------------------------------------------------