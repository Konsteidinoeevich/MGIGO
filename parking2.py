import jax
import jax.numpy as jnp
from jax import random, jit ,lax
import time
import numpy as onp
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from typing import Tuple 

# =====================================================================
# I. 优化器导入
# =====================================================================
from gmm_igo.MPCsolverM2 import mmog_igo_optimizer_mpc

# =====================================================================
# II. 常量和配置
# =====================================================================
DT = 0.1
HORIZON = 100
TOTAL_TIME = HORIZON * DT
POLY_ORDER = 5
NUM_CONTROL_POINTS_FULL = 9 
NUM_CONTROL_POINTS_OPT = 8
DIMS_TUPLE = (NUM_CONTROL_POINTS_OPT, NUM_CONTROL_POINTS_OPT) 
M_MOG = 2
D_MAX = max(DIMS_TUPLE)

# --- CLF 参数 ---
K1, K2, K3 = 1.0, 5.0, 1.0
SIGMA = 0.01
Q_VAL = jnp.sqrt(K1 / K3)
W_END = 1.0   
BAR_V = 3.0      # 最大线速度
BAR_O = 1.0      # 最大角速度（统一使用 BAR_O）

# --- M-MoG IGO 配置 ---
SEED = 42
DELTA_T = 0.1     
K_COMP = 8        
B_SAMPLES = 60   
B_0_ELITE = 25    
T_0_RESTART = 50 

# --- MPC 配置 ---
# 🌟 关键修正：统一迭代次数为 500，确保 JIT 编译只发生一次，提升 MPC 循环性能。
T_RUN_INIT = 500       
T_RUN_STEP = 500        
MPC_STEPS = 150           
L_INV_START_MAG = 2.0   

# --- 目标与初始状态 ---
X_TARGET, Y_TARGET, YAW_TARGET = 5.0, 3.0, -jnp.pi / 2.0 
INITIAL_X, INITIAL_Y, INITIAL_YAW, INITIAL_V = 0.0, 0.0, 0.0, 0.5 
INITIAL_CONTEXT = jnp.array([INITIAL_X, INITIAL_Y, INITIAL_YAW, INITIAL_V])

# =====================================================================
# III. B-spline 矩阵构造
# =====================================================================
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
            if j < 0: j_final = abs(j)
            elif j >= N: j_final = 2 * (N - 1) - j
            else: j_final = j
            F[i, j_final] += weight
    return jnp.array(F, dtype=jnp.float32)

F_MATRIX = create_5_tap_filter_matrix(NUM_CONTROL_POINTS_FULL)
INTERNAL_KNOTS_COUNT = NUM_CONTROL_POINTS_FULL - POLY_ORDER 
KNOT_DELTA = TOTAL_TIME / INTERNAL_KNOTS_COUNT 
internal_knots = onp.arange(1, INTERNAL_KNOTS_COUNT + 1) * KNOT_DELTA
knots = onp.concatenate([onp.zeros(POLY_ORDER + 1), internal_knots, onp.full(POLY_ORDER, TOTAL_TIME)]) 
t_eval = onp.arange(HORIZON) * DT

def compute_basis_matrix(knots, t_eval, k, nu):
    basis = onp.zeros((len(t_eval), NUM_CONTROL_POINTS_FULL))
    for i in range(NUM_CONTROL_POINTS_FULL):
        c = onp.zeros(NUM_CONTROL_POINTS_FULL); c[i] = 1.0
        spl = BSpline(knots, c, k=k, extrapolate=True) 
        basis[:, i] = spl(t_eval, nu=nu)
    return jnp.array(basis, dtype=jnp.float32)

N5_BASIS = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=0)
N5_PRIME = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=1) 
N5_DOUBLE_PRIME = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=2)

# =====================================================================
# IV. 轨迹解码函数
# =====================================================================
@jit
def theta_to_cartesian_trajectory(theta: jnp.ndarray, context: jnp.ndarray):
    x_cur, y_cur, theta_cur, v_cur = context[0], context[1], context[2], context[3]
    Qx_opt = theta[:NUM_CONTROL_POINTS_OPT]
    Qy_opt = theta[NUM_CONTROL_POINTS_OPT:]
    x_anchor_0 = x_cur; y_anchor_0 = y_cur
    #x_anchor_1 = x_cur + v_cur * jnp.cos(theta_cur) * DT
    #y_anchor_1 = y_cur + v_cur * jnp.sin(theta_cur) * DT
    Qx_anchors = jnp.array([x_anchor_0])
    Qy_anchors = jnp.array([y_anchor_0])
    Qx_full = jnp.concatenate([Qx_anchors, Qx_opt])
    Qy_full = jnp.concatenate([Qy_anchors, Qy_opt])
    Px_full = F_MATRIX @ Qx_full
    Py_full = F_MATRIX @ Qy_full
    x_traj  = N5_BASIS @ Px_full; y_traj  = N5_BASIS @ Py_full
    x_dot = N5_PRIME @ Px_full; y_dot = N5_PRIME @ Py_full
    x_ddot  = N5_DOUBLE_PRIME @ Px_full; y_ddot  = N5_DOUBLE_PRIME @ Py_full
    return x_traj, y_traj, x_dot, y_dot, x_ddot, y_ddot, Qx_full, Qy_full, Px_full, Py_full

# =====================================================================
# V. CLF 成本函数
# =====================================================================
@jit
def _cost_nu(nu_val, epsilon):
    TOL = 1e-6
    abs_nu = jnp.abs(nu_val)
    is_near_zero = abs_nu < TOL
    cost_non_zero = jnp.arctan(epsilon * abs_nu) - jnp.log(1 + epsilon**2 * nu_val**2) / (2 * epsilon * abs_nu + TOL)
    cost_near_zero = -1/12 * (epsilon * nu_val)**4 
    return jnp.where(is_near_zero, cost_near_zero, cost_non_zero)

@jit
def calculate_clf_terms(x, y, dx, dy, ddx, ddy):
    v_sq = dx**2 + dy**2
    v = jnp.sqrt(v_sq + 1e-8)
    omega = (ddy * dx - ddx * dy) / (v_sq + 1e-6)
    rho_sq = x**2 + y**2
    rho = jnp.sqrt(rho_sq)
    delta = jnp.arctan2(y, x) + jnp.pi
    theta = jnp.arctan2(dy, dx)
    gamma = jnp.arctan2(y, x) - theta + jnp.pi
    A = gamma + 0.5 * jnp.arctan(2 * K2 * delta)
    dV_drho = 2 * rho
    dV_dgamma = 2 * Q_VAL**2 * A
    term_d_delta = Q_VAL**2 * A * (2 * K2 / (1 + 4 * K2**2 * delta**2))
    dV_ddelta = 2 * delta + term_d_delta
    cos_gamma = (x * dx + y * dy) / (rho * v + 1e-6)
    sin_gamma = (x * dy - y * dx) / (rho * v + 1e-6)
    nu2 = -dV_dgamma
    nu1 = dV_drho * (-rho * cos_gamma) + dV_ddelta * sin_gamma + dV_dgamma * sin_gamma
    
    EPSILON1 = 2 * BAR_V / (jnp.pi * (rho + SIGMA))
    EPSILON2 = 2 * BAR_O / jnp.pi   # 使用 BAR_O
    
    return nu1, nu2, v, rho, omega, EPSILON1, EPSILON2

def create_unicycle_spline_clf_fitness_fn(x_target, y_target, yaw_target):
    @jax.jit
    def fitness_fn_total(samples_overall: jnp.ndarray, context: jnp.ndarray) -> jnp.float32:
        X_traj, Y_traj, dX_traj, dY_traj, ddX_traj, ddY_traj, _, _, _, _ = theta_to_cartesian_trajectory(samples_overall, context)
        
        def scan_body(carry, i):
            x, y, dx, dy, ddx, ddy = X_traj[i], Y_traj[i], dX_traj[i], dY_traj[i], ddX_traj[i], ddY_traj[i]
            nu1, nu2, v, rho, omega, EPSILON1, EPSILON2 = calculate_clf_terms(x, y, dx, dy, ddx, ddy)
            C1 = _cost_nu(nu1, EPSILON1)
            C2 = _cost_nu(nu2, EPSILON2)
            clip_max = jnp.pi/2 - 1e-3
            C3 = jnp.log(jnp.cos(jnp.clip(jnp.abs(v) / (EPSILON1 * (rho + 1e-6)), a_max=clip_max)))
            C4 = jnp.log(jnp.cos(jnp.clip(jnp.abs(omega) / EPSILON2, a_max=clip_max)))
            j_running_cost = C1 + C2 + C3 + C4
            total_J = carry + j_running_cost * DT 
            return total_J, None

        J_total_running, _ = lax.scan(scan_body, 0.0, jnp.arange(HORIZON))
        
        final_x = X_traj[-1]; final_y = Y_traj[-1]
        final_yaw = jnp.arctan2(dY_traj[-1], dX_traj[-1]) 
        C_end_pos = 100*(final_x - x_target)**2 + 100*(final_y - y_target)**2 + (final_yaw - yaw_target)**2
        yaw_err = jnp.arctan2(jnp.sin(final_yaw - yaw_target), jnp.cos(final_yaw - yaw_target))
        C_end_yaw = yaw_err**2
        C_terminal = W_END * (C_end_pos + C_end_yaw)
        return -(J_total_running) + C_terminal

    return fitness_fn_total

# =====================================================================
# VI. 初始化函数
# =====================================================================
def initialize_params_mmog_heterogeneous(key, M: int, K: int, dims: Tuple[int, ...]):
    D_max = max(dims)
    initial_mu_k_K = random.normal(key, (M, K, D_max)) * 0.1 
    initial_mu_k_K_plus_1 = jnp.zeros((M, 1, D_max))
    initial_mu_k = jnp.concatenate([initial_mu_k_K, initial_mu_k_K_plus_1], axis=1)
    return initial_mu_k, D_max

# =====================================================================
# VII. 主程序：MPC + 简单欧拉积分 + 实时绘图
# =====================================================================
def main():
    key = random.PRNGKey(SEED)
    key_init, key_run = random.split(key)
    
    fitness_fn_total = create_unicycle_spline_clf_fitness_fn(X_TARGET, Y_TARGET, YAW_TARGET)
    
    # --- 1. 优化器参数的初始 Warm Start ---
    initial_mu_k_ws, D_max = initialize_params_mmog_heterogeneous(key_init, M_MOG, K_COMP, DIMS_TUPLE)
    
    # 🌟 关键修改：初始化 L_inv_k 和 v_k 的 Warm Start (热启动)
    # L_inv_k (协方差逆矩阵) 初始冷启动值
    initial_L_inv_k_ws = L_INV_START_MAG * jnp.eye(D_MAX)
    initial_L_inv_k_ws = jnp.tile(initial_L_inv_k_ws[None, None, :, :], (M_MOG, K_COMP + 1, 1, 1))
    
    # v_k (log-weights) 初始冷启动值 (零向量)
    initial_v_k_ws = jnp.zeros((M_MOG, K_COMP))
    
    current_context = INITIAL_CONTEXT.copy()
    
    print("=== M-MoG IGO MPC 运动规划启动 ===")
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    history = []  # 记录历史位置
    
    for mpc_step in range(MPC_STEPS):
        T_CURRENT = T_RUN_INIT # 现在 T_RUN_INIT = T_RUN_STEP = 500
        
        print(f"\n--- MPC 步 {mpc_step+1}/{MPC_STEPS} | 优化迭代 {T_CURRENT} ---")
        print(f"当前状态: X={current_context[0]:.3f}, Y={current_context[1]:.3f}, "
              f"Yaw={jnp.rad2deg(current_context[2]):.1f}°, V={current_context[3]:.3f}")
        
        key_run, subkey = random.split(key_run)
        
        start_time = time.time()
        
        # 2. 调用优化器，使用上一轮的 Warm Start 参数
        final_mu_k, final_L_inv_k_raw, final_pi_k_all = mmog_igo_optimizer_mpc(
            subkey, T_CURRENT, DELTA_T, M_MOG, K_COMP, B_SAMPLES, B_0_ELITE, 
            DIMS_TUPLE, T_0_RESTART, fitness_fn_total,
            initial_mu_k=initial_mu_k_ws, 
            initial_L_inv_k=initial_L_inv_k_ws, # 🌟 热启动 L_inv_k
            initial_v_k=initial_v_k_ws,
            context=current_context 
        )
        
        final_mu_k.block_until_ready()
        
        opt_time = time.time() - start_time
        print(f"优化耗时: {opt_time:.3f}s")
        
        # 3. 结果提取和下一轮 Warm Start 更新
        best_comp_indices = jnp.argmax(final_pi_k_all[:, :-1], axis=1) 
        mu_star_M = final_mu_k[jnp.arange(M_MOG), best_comp_indices]
        best_theta = mu_star_M.flatten()
        
        # 更新 Warm Start 参数
        # mu_k: 直接传递整个矩阵 (B-spline 不需要序列平移)
        initial_mu_k_ws = final_mu_k
        # L_inv_k: 传递协方差矩阵的逆 (热启动)
        initial_L_inv_k_ws = final_L_inv_k_raw
        # v_k: 保持冷启动（零向量），依赖优化器内部的 T_0 重启逻辑管理
        # initial_v_k_ws = jnp.zeros((M_MOG, K_COMP)) # 保持不变，因为初始化时就是零向量

        # 4. 状态推进 (与 Crossingroad.py 理念一致：应用第一个控制)
        if mpc_step < MPC_STEPS - 1:
            X_traj, Y_traj, dX_traj, dY_traj, ddX_traj, ddY_traj, _, _, _, _ = \
                theta_to_cartesian_trajectory(best_theta, current_context)
            
            dx0 = dX_traj[0]; dy0 = dY_traj[0]
            v_cmd = dx0 * jnp.cos(current_context[2]) + dy0 * jnp.sin(current_context[2])
            
            v_sq = dx0**2 + dy0**2 + 1e-6
            omega_cmd = (ddY_traj[0] * dx0 - ddX_traj[0] * dy0) / v_sq
            
            new_x = current_context[0] + dx0 * DT
            new_y = current_context[1] + dy0 * DT
            new_yaw = current_context[2] + omega_cmd * DT
            new_v = v_cmd 
            
            current_context = jnp.array([new_x, new_y, new_yaw, new_v])
            
            direction = "前进" if v_cmd >= 0 else "后退"
            print(f"→ 新状态: X={new_x:.3f}, Y={new_y:.3f}, "
                  f"Yaw={jnp.rad2deg(new_yaw):.1f}°, V={new_v:.3f} ({direction})")
        
        # ------------------- 实时绘图 (保持不变) -------------------
        ax.cla()
        ax.set_xlim(-1, 6.5)
        ax.set_ylim(-1, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        X_traj, Y_traj, dX_traj, dY_traj, ddX_traj, ddY_traj, Qx_full, Qy_full, Px_full, Py_full = \
            theta_to_cartesian_trajectory(best_theta, current_context)
        
        ax.plot(X_TARGET, Y_TARGET, 'r*', markersize=20, label='Target')
        
        if history:
            hist = jnp.array(history)
            ax.plot(hist[:,0], hist[:,1], 'b-', linewidth=4, alpha=0.7, label='History')
        
        ax.plot(X_traj, Y_traj, 'cyan', linewidth=5, alpha=0.8, label='Planned Trajectory')
        ax.plot(current_context[0], current_context[1], 'go', markersize=12, label='Current Pos')
        
        ax.plot(Qx_full, Qy_full, 'rx', markersize=9, mew=3, label='Q (opt)')
        ax.plot(Px_full, Py_full, 'ko', markersize=7, label='P (filtered)')
        
        dist_to_goal = jnp.sqrt((current_context[0]-X_TARGET)**2 + (current_context[1]-Y_TARGET)**2)
        ax.set_title(f"MPC Step {mpc_step+1}/{MPC_STEPS} | "
                     f"Dist: {dist_to_goal:.2f}m | Time: {opt_time:.2f}s")
        
        ax.legend(loc='upper left')
        plt.pause(0.01)
        
        history.append([current_context[0], current_context[1]])
    
    print("\n=== MPC 规划结束 ===")
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()