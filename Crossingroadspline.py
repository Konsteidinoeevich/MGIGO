# Crossingroadspline.py - 3车十字路口轨迹规划 (B-spline M-MoG IGO MPC - FINAL COST FIX)

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax 
import functools 
import time 
from typing import Callable, Tuple, List, Any, Dict
import numpy as onp
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt 
from matplotlib import animation 

from gmm_igo.MPCsolverM import mmog_igo_optimizer_mpc 

# ======================================================================
# I. 全局配置 (适配 B-spline)
# ======================================================================

SEED = 42
DT = 0.1          # B-spline 时间步长 / MPC 滚动步长
DYN_DT = DT       # MPC 步进时间 
HORIZON = 100     # B-spline 轨迹点数
TOTAL_TIME = HORIZON * DT # 10.0 seconds
POLY_ORDER = 5    # B-spline 阶次

# --- B-spline 维度 ---
NUM_CONTROL_POINTS_FULL = 10 # N
NUM_CONTROL_POINTS_OPT = 8  # N - 2
D_I = 2 * NUM_CONTROL_POINTS_OPT # 总优化变量维度 (2 * 13 = 26)

# --- MPC/IGO 配置 ---
M_MOG = 3         
K_COMP = 8        
B_SAMPLES = 60    
B_0_ELITE = 25    
T_0_RESTART = 80  
T_IGO_ITER = 400  
T_MPC_RUNS = 80   

DIMS_TUPLE = (D_I, D_I, D_I) 
D_MAX = D_I

# --- 成本函数权重和常量 ---
# W_TARGET = 10.0 # 原始目标权重已移除，改为非对称加权
W_TARGET_LONG = 50.0  # 终端纵向目标权重 (参照 MPCmain4.py)
W_TARGET_LAT = 10.0   # 终端横向目标权重 (参照 MPCmain4.py)
W_COLL = 100.0
W_PATH = 5.0      
D_SAFE_SQ = 1.0   
INTERSECTION_BUFFER = 2.0 
ROAD_HALF_WIDTH = 1.5     
LANE_WIDTH = 3.7 
LANE_WIDTH_FLOAT = float(LANE_WIDTH) 

# B-spline 动态成本权重 (参照 MPCmain4.py)
W_ACC = 10.0 
W_LAT = 1.0 
W_JERK = 10.0 

# ======================================================================
# II. B-spline 矩阵和核心函数
# ======================================================================

# --- 1. 构造 5-Tap 滤波器矩阵 F (不变) ---
def create_5_tap_filter_matrix(N):
    F = onp.zeros((N, N))
    W = onp.array([1, 26, 66, 26, 1]) / 120.0
    for i in range(N):
        if i == 0 or i == N - 1:
            F[i, i] = 1.0
            continue
        for r in range(-2, 3):
            q_idx_raw = i + r
            weight = W[r + 2]
            j = q_idx_raw
            if j < 0:
                j_final = abs(j)
            elif j >= N:
                j_final = 2 * (N - 1) - j
            else:
                j_final = j
            F[i, j_final] += weight
    return jnp.array(F, dtype=jnp.float32)

F_MATRIX = create_5_tap_filter_matrix(NUM_CONTROL_POINTS_FULL)

# --- 2. B-spline 基函数和导数矩阵 (不变) ---
INTERNAL_KNOTS_COUNT = NUM_CONTROL_POINTS_FULL - POLY_ORDER 
KNOT_DELTA = TOTAL_TIME / INTERNAL_KNOTS_COUNT 
internal_knots = onp.arange(1, INTERNAL_KNOTS_COUNT + 1) * KNOT_DELTA
knots = onp.concatenate([onp.zeros(POLY_ORDER + 1), 
                         internal_knots, 
                         onp.full(POLY_ORDER, TOTAL_TIME)]) 
t_eval = onp.arange(HORIZON) * DT 

def compute_basis_matrix(knots, t_eval, k, nu):
    basis = onp.zeros((len(t_eval), NUM_CONTROL_POINTS_FULL))
    for i in range(NUM_CONTROL_POINTS_FULL):
        c = onp.zeros(NUM_CONTROL_POINTS_FULL); c[i] = 1.0
        spl = BSpline(knots, c, k=k, extrapolate=True) 
        basis[:, i] = spl(t_eval, nu=nu)
    return jnp.array(basis, dtype=jnp.float32)

N5_BASIS          = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=0) 
N5_PRIME          = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=1) 
N5_DOUBLE_PRIME   = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=2) 
N5_TRIPLE_PRIME   = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=3) 


# --- 3. 核心函数: 优化变量 -> 轨迹和导数 (不变) ---
@jit
def theta_to_trajectory(theta: jnp.ndarray, ctx: Dict[str, Any]):
    s_cur   = ctx['s_cur']; l_cur   = ctx['l_cur']; ds_cur  = ctx['ds_cur']
    
    # 1. 拆分优化变量 (长度 13)
    Qs_opt = theta[:NUM_CONTROL_POINTS_OPT]; 
    Ql_opt = theta[NUM_CONTROL_POINTS_OPT:]
    
    # 2. 计算 Q[0] 和 Q[1] 的锚定值 (C0 和 C1 连续性)
    s_anchor_0 = s_cur
    l_anchor_0 = l_cur
    s_anchor_1 = s_cur + ds_cur * DT
    l_anchor_1 = l_cur
    
    Qs_anchors = jnp.array([s_anchor_0, s_anchor_1])
    Ql_anchors = jnp.array([l_anchor_0, l_anchor_1])

    # 3. 重构完整的 Q 向量 (长度 15)
    Qs_full = jnp.concatenate([Qs_anchors, Qs_opt])
    Ql_full = jnp.concatenate([Ql_anchors, Ql_opt])

    # 4. 滤波和轨迹计算 (P = F @ Q_full)
    Ps = F_MATRIX @ Qs_full; Pl = F_MATRIX @ Ql_full
    
    s_traj   = N5_BASIS      @ Ps; l_traj   = N5_BASIS      @ Pl
    s_dot    = N5_PRIME      @ Ps; l_dot    = N5_PRIME      @ Pl
    s_ddot   = N5_DOUBLE_PRIME @ Ps; l_ddot   = N5_DOUBLE_PRIME @ Pl
    s_dddot  = N5_TRIPLE_PRIME @ Ps; l_dddot  = N5_TRIPLE_PRIME @ Pl

    # 5. 轨迹 (笛卡尔 X-Y, 假设直道近似: X=S, Y=L*W)
    traj_xy = jnp.stack([s_traj, l_traj * LANE_WIDTH_FLOAT], axis=1)

    return traj_xy, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot

# ======================================================================
# III. 轨迹预测和成本计算函数 (FIXED)
# ======================================================================

@jit
def get_traj_and_next_state(theta, initial_state_cartesian, car_idx):
    """
    根据 B-spline 参数计算完整轨迹和 MPC 下一步的状态。
    包含对 lax.switch 的参数解包修正。
    """
    x_cur, y_cur, theta_cur, v_cur = initial_state_cartesian
    
    # 1. 定义三个车的 Frenet 转换逻辑函数
    def car_1_frenet(operands): # Car 1: W->E (S=X, L=Y/W)
        _x, _y, _v = operands[0] # 修正: 显式解包 operands[0]
        return _x, _y / LANE_WIDTH_FLOAT, _v
    
    def car_2_frenet(operands): # Car 2: S->N (S=Y, L=-X/W)
        _x, _y, _v = operands[0] # 修正: 显式解包 operands[0]
        return _y, -_x / LANE_WIDTH_FLOAT, _v
    
    def car_3_frenet(operands): # Car 3: E->N Turn (S=-X, L=-Y/W)
        _x, _y, _v = operands[0] # 修正: 显式解包 operands[0]
        return -_x, -_y / LANE_WIDTH_FLOAT, _v

    # 2. 使用 jax.lax.switch 替代 if/elif/else 
    s_cur, l_cur, ds_cur = lax.switch(
        car_idx,
        (car_1_frenet, car_2_frenet, car_3_frenet),
        ((x_cur, y_cur, v_cur),) # 传入一个包含元组的元组
    )

    ctx_i = {'s_cur': s_cur, 'l_cur': l_cur, 'ds_cur': ds_cur}

    # 3. B-spline 轨迹生成 (保持不变)
    traj_xy, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot = theta_to_trajectory(theta, ctx_i)
    
    # 4. 提取下一步状态 (t = DT = 0.1s, 对应轨迹点索引 1)
    x_next, y_next = traj_xy[1, :]
    v_next = s_dot[1] 
    
    dx_dt_next = s_dot[1]
    dy_dt_next = l_dot[1] * LANE_WIDTH_FLOAT
    theta_next = jnp.arctan2(dy_dt_next, dx_dt_next)
    
    next_state = jnp.array([x_next, y_next, theta_next, v_next])
    
    return traj_xy, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot, next_state, ctx_i


def create_3car_mpc_fitness_fn_bspline(M_MOG: int, D_I: int, D_MAX: int):
    
    @jax.jit
    def fitness_fn_total(samples_overall, context):
        
        samples_M_padded = samples_overall.reshape((M_MOG, D_MAX))
        context_M = context.reshape((M_MOG, 6))
        
        initial_states_cartesian = context_M[:, :4] 
        targets = context_M[:, 4:] 

        def vmap_body(m_idx):
            theta = samples_M_padded[m_idx, :D_I] 
            
            # Trajectory generation and cost component extraction
            traj_xy, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot, _, _ = \
                get_traj_and_next_state(theta, initial_states_cartesian[m_idx], m_idx)
            
            # --- 成本计算 (基于 B-spline 导数) ---
            ddot_x = s_ddot                                     
            ddot_y = l_ddot * LANE_WIDTH_FLOAT                        
            a_mag = jnp.sqrt(ddot_x**2 + ddot_y**2)             
            dddot_x = s_dddot                                   
            dddot_y = l_dddot * LANE_WIDTH_FLOAT                      

            target_xy = targets[m_idx] 
            
            # 1. 终端代价 (Target) - **修正**为笛卡尔 X/Y 各自加权 (参照 MPCmain4.py)
            x_err = traj_xy[-1, 0] - target_xy[0]
            y_err = traj_xy[-1, 1] - target_xy[1]
            
            # 权重分配逻辑: X轴为主 (Car 1) 或 Y轴为主 (Car 2, 3)
            def car_1_weights():
                # Car 1: X是纵向 (W_TARGET_LONG), Y是横向 (W_TARGET_LAT)
                return W_TARGET_LONG, W_TARGET_LAT
            def car_2_or_3_weights():
                # Car 2 & 3: Y是纵向 (W_TARGET_LONG), X是横向 (W_TARGET_LAT)
                return W_TARGET_LAT, W_TARGET_LONG

            W_X_term, W_Y_term = lax.cond(m_idx == 0, car_1_weights, car_2_or_3_weights)
            
            cost_target = W_X_term * x_err**2 + W_Y_term * y_err**2 
            
            # 2. 动态/平滑性惩罚 (与 MPCmain4.py 哲学一致: 基于笛卡尔导数)
            cost_acc = W_ACC * jnp.sum(jnp.maximum(0.0, a_mag - 5.0)**2) 
            cost_lat = W_LAT * jnp.sum(ddot_y**2) 
            cost_jerk = W_JERK * jnp.sum(dddot_x**2 + dddot_y**2) 
            
            f_local = cost_target + cost_acc + cost_lat + cost_jerk
            
            # 3. Car 3 的路径惩罚
            is_car3 = m_idx == 2
            
            def calculate_car3_path_cost():
                 is_approach = s_traj < -INTERSECTION_BUFFER 
                 lateral_viol = jnp.clip(jnp.abs(l_traj * LANE_WIDTH_FLOAT) - ROAD_HALF_WIDTH, a_min=0.0)
                 f_path_car3 = W_PATH * jnp.sum(jnp.where(is_approach, lateral_viol, 0.0))
                 return f_path_car3
                 
            f_local += lax.cond(is_car3, calculate_car3_path_cost, lambda: 0.0)

            return f_local, traj_xy

        f_local_M, traj_xy_M = vmap(vmap_body)(jnp.arange(M_MOG)) 
        f_local_total = jnp.sum(f_local_M)

        # 4. 耦合成本：避碰 (Collision)
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
# IV. 初始化函数 (不变)
# ======================================================================

def initialize_params_mmog_heterogeneous(key, M: int, K: int, DIMS: Tuple[int, ...]):
    """使用 D_I=26 的直行轨迹作为初始猜测。"""
    D_max: int = max(DIMS)
    initial_mu_list: List[jnp.ndarray] = []
    initial_L_inv_list: List[jnp.ndarray] = []
    keys = random.split(key, M)

    for m in range(M):
        D_m = D_I 
        key_mu, key_L = random.split(keys[m])
        
        # 1. 构造初始 Q 向量 (直行/零横向)
        V_INIT = 1.0 
        Qs0_full = onp.arange(NUM_CONTROL_POINTS_FULL) * DT * V_INIT
        Ql0_full = onp.zeros(NUM_CONTROL_POINTS_FULL)
        
        # 2. 截取 Q[2] 到 Q[14] 作为优化的 theta0 (长度 13)
        Qs_opt_0 = jnp.array(Qs0_full[2:])
        Ql_opt_0 = jnp.array(Ql0_full[2:])
        theta0 = jnp.concatenate([Qs_opt_0, Ql_opt_0]) # 长度 26
        
        # 3. 构造 MoG mu_k
        mu_m_actual = jnp.stack([theta0] * (K + 1)) 
        mu_m_padded = jnp.pad(mu_m_actual, ((0, 0), (0, D_max - D_m)), mode='constant')
        initial_mu_list.append(mu_m_padded)

        # 4. 构造 L_inv_k 
        L_inv_template = jnp.eye(D_m) * 2.0 
        L_inv_k_all = jnp.stack([L_inv_template] * (K + 1)) 
        L_inv_m_padded = jnp.pad(L_inv_k_all, ((0, 0), (0, D_max - D_m), (0, D_max - D_m)), mode='constant')
        initial_L_inv_list.append(L_inv_m_padded)

    initial_v_k = jnp.zeros((M, K))
    initial_mu_k_stacked = jnp.stack(initial_mu_list)     
    initial_L_inv_k_stacked = jnp.stack(initial_L_inv_list) 
    
    return initial_mu_k_stacked, initial_L_inv_k_stacked, initial_v_k, D_max

# ======================================================================
# V. 主程序运行逻辑 (不变)
# ======================================================================

if __name__ == '__main__':
    
    fitness_fn_total = create_3car_mpc_fitness_fn_bspline(M_MOG, D_I, D_MAX)

    key = random.PRNGKey(SEED)
    key_init, key_run = random.split(key)
    
    initial_mu_k, initial_L_inv_k, initial_v_k, D_max_check = initialize_params_mmog_heterogeneous(
        key_init, M_MOG, K_COMP, DIMS_TUPLE
    )
    
    print("--- 3 车轨迹规划 (B-spline M-MoG IGO MPC) 动态图测试开始 ---")
    print(f"场景: 三车通过十字路口 (dt={DYN_DT}, D_I={D_I}, T_MPC={T_MPC_RUNS})")
    print(f"终端代价权重: 纵向={W_TARGET_LONG}, 横向={W_TARGET_LAT} (参照 MPCmain4.py)")
    
    # 初始状态和目标 (十字路口场景)
    initial_context_data = jnp.array([
        [-15.0, 0.0, 0.0, 1.0, 15.0, 0.0],       # Car 1: 西向东直行 (x, y, theta, v, target_x, target_y)
        [0.0, -15.0, jnp.pi/2, 1.0, 0.0, 15.0],  # Car 2: 南向北直行
        [15.0, 0.0, jnp.pi, 1.0, 0.0, 15.0],     # Car 3: 东向西，左转去北
    ])
    
    # --- 绘图数据收集初始化 ---
    trajectory_data = [[] for _ in range(M_MOG)] 
    current_initial_states = initial_context_data[:, :4] 
    targets = initial_context_data[:, 4:]
    for i in range(M_MOG):
        trajectory_data[i].append(current_initial_states[i, :2].tolist())
    
    # --- IGO Warm Start 初始化 ---
    mu_k_current = initial_mu_k  
    L_inv_k_current = initial_L_inv_k
    v_k_current = initial_v_k 

    current_context = initial_context_data.flatten()
    T_MPC_END = T_MPC_RUNS 

    # --- MPC 滚动循环 ---
    for mpc_step in range(T_MPC_RUNS):
        key_run, subkey = random.split(key_run)
        print(f"\n--- MPC Step {mpc_step + 1}/{T_MPC_RUNS} ---")
        
        start_time = time.time()
        
        # 1. IGO 优化
        final_mu_k, final_L_inv_k, final_pi_k_all = mmog_igo_optimizer_mpc(
            subkey, T_IGO_ITER, 0.12, M_MOG, K_COMP, B_SAMPLES, B_0_ELITE, 
            DIMS_TUPLE, 
            T_0_RESTART, fitness_fn_total,
            mu_k_current, L_inv_k_current, v_k_current, 
            context=current_context 
        )
        final_mu_k.block_until_ready()
        elapsed_time = time.time() - start_time
        
        # 2. 提取最优 theta
        best_comp_indices = jnp.argmax(final_pi_k_all[:, :-1], axis=1) 
        best_theta_M_padded = final_mu_k[jnp.arange(M_MOG), best_comp_indices]
        
        # 3. 轨迹生成并提取下一步状态 (next_state)
        new_initial_states = []
        delta_s_list = []
        
        for i in range(M_MOG):
            best_theta = best_theta_M_padded[i, :D_I]
            
            # 必须将 i 显式转换为 JAX 整数，因为它是 lax.switch 的索引
            i_jnp = jnp.array(i, dtype=jnp.int32)
            
            # 使用修正后的函数
            _, s_traj, _, _, _, _, _, _, _, next_state_i, _ = \
                get_traj_and_next_state(best_theta, current_initial_states[i], i_jnp)
            
            new_initial_states.append(next_state_i)
            
            # 计算 s 的平移量 (Warm Start 所需)
            delta_s = s_traj[1] - s_traj[0] 
            delta_s_list.append(delta_s)
        
        current_initial_states = jnp.stack(new_initial_states)

        # 4. 更新 Warm Start 参数 (重调度/平移)
        for i in range(M_MOG):
            best_theta_old = best_theta_M_padded[i, :D_I]
            
            # 1. 从 best_theta (长度 26) 中取出优化的 Qs 和 Ql (长度 13)
            Qs_k_old = best_theta_old[:NUM_CONTROL_POINTS_OPT]
            Ql_k_old = best_theta_old[NUM_CONTROL_POINTS_OPT:]
            
            # 2. X 坐标平移 Qs，Ql 不变 (应用于 13 个优化点)
            Qs_next_guess = Qs_k_old + delta_s_list[i] 
            Ql_next_guess = Ql_k_old 
            theta_next_guess = jnp.concatenate([Qs_next_guess, Ql_next_guess])
            
            # 将新的猜测值设置为 MoG 的第 0 个分量 (Warm Start)
            mu_k_current = mu_k_current.at[i, 0, :D_I].set(theta_next_guess)
            L_inv_k_current = final_L_inv_k
            
        # 5. 更新 Context 和轨迹数据
        current_context = jnp.concatenate([current_initial_states, targets], axis=1).flatten()
        
        current_xy = current_initial_states[:, :2].tolist() 
        for i in range(M_MOG):
            trajectory_data[i].append(current_xy[i])
        
        # 6. 结果打印
        f_mu_star = fitness_fn_total(best_theta_M_padded.flatten(), current_context)
        print(f"  IGO 耗时: {elapsed_time:.4f}s, 最终成本: {f_mu_star:.4e}")
        
        final_pos = current_initial_states[:, :2]
        dist_to_target = jnp.linalg.norm(final_pos - targets, axis=1)
        print(f"  当前距离目标: {dist_to_target}")
        
        if jnp.all(dist_to_target < 0.5): 
            print("\n所有车辆接近目标 (距离 < 0.5m)，MPC 结束。")
            T_MPC_END = mpc_step + 1 
            break
        elif mpc_step == T_MPC_RUNS - 1:
            T_MPC_END = T_MPC_RUNS
        
    print("\n--- MPC 滚动优化完成 ---")

    # ======================================================================
    # VI. 轨迹动画可视化 (Animation)
    # ======================================================================
    
    print("\n--- 正在生成车辆轨迹动画 ---")

    all_trajectories_array = jnp.array(trajectory_data) 
    num_frames = T_MPC_END + 1 

    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    labels = ["Car 1 (W->E)", "Car 2 (S->N)", "Car 3 (E->N Turn)"]
    
    # 绘制静态背景元素
    ax.axvline(x=ROAD_HALF_WIDTH, color='lightgray', linestyle='-', alpha=0.5)
    ax.axvline(x=-ROAD_HALF_WIDTH, color='lightgray', linestyle='-', alpha=0.5)
    ax.axhline(y=ROAD_HALF_WIDTH, color='lightgray', linestyle='-', alpha=0.5)
    ax.axhline(y=-ROAD_HALF_WIDTH, color='lightgray', linestyle='-', alpha=0.5)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.scatter([0], [0], marker='+', color='red', s=100, label='Intersection Center')
    
    # 标记目标点
    for i in range(M_MOG):
        target_x, target_y = targets[i].tolist()
        ax.plot(target_x, target_y, marker='X', markersize=12, color=colors[i], linestyle='none')

    # 初始化动态元素
    lines = [] 
    car_markers = [] 

    for i in range(M_MOG):
        line, = ax.plot([], [], linestyle='-', color=colors[i], alpha=0.5, linewidth=2.0)
        lines.append(line)
        
        marker, = ax.plot([], [], marker='^', markersize=10, color=colors[i], label=labels[i])
        car_markers.append(marker)

    # 设置图表属性
    ax.set_title(f'3-Car Intersection Coordination - B-spline MPC')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal', adjustable='box') 
    
    def init():
        for line in lines:
            line.set_data([], [])
        for marker in car_markers:
            marker.set_data([], [])
        return lines + car_markers

    def animate(frame):
        t = frame 
        for i in range(M_MOG):
            traj = all_trajectories_array[i]
            lines[i].set_data(traj[:t+1, 0], traj[:t+1, 1])
            current_x, current_y = traj[t, 0], traj[t, 1]
            car_markers[i].set_data([current_x], [current_y])
            
        ax.set_title(f'3-Car Intersection Coordination - Time Step {t} / {num_frames-1} (T={t * DYN_DT:.2f}s)')
        return lines + car_markers

    anim = animation.FuncAnimation(
        fig, 
        animate, 
        init_func=init, 
        frames=num_frames, 
        interval=DYN_DT * 1000, 
        blit=False,
        repeat=False
    )

    anim.save('intersection_mpc_bspline_animation.gif', writer='pillow', fps=10)
    print("动画已保存为 intersection_mpc_bspline_animation.gif")
            
    plt.show()