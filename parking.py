# CLFParking_Pure_CLF_Cartesian.py
# 2025 真实量产级 MPC —— 引入 CLF 自动饱和与镇定机制 (笛卡尔坐标系)

import jax
import jax.numpy as jnp
from jax import jit, vmap 
import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from gmm_igo.MPCsolver import igo_mog_optimizer 
from scipy.interpolate import BSpline

# ====================== 1. 全局配置 (CLF 常数) ======================
DT = 0.05
HORIZON = 100
TOTAL_TIME = HORIZON * DT  # 5.0 seconds
POLY_ORDER = 5       

# --- 关键维度 ---
NUM_CONTROL_POINTS_FULL = 10 
NUM_CONTROL_POINTS_OPT = 8  
TOTAL_DIM = 2 * NUM_CONTROL_POINTS_OPT 

# --- CLF 约束常数 (与推导保持一致) ---
BAR_V = 3.0         # 最大允许线速度 \bar{v} (停车场景降低)
BAR_OMEGA = 1.0     # 最大允许角速度 \bar{\omega}
SIGMA = 0.1         # \sigma
K2 = 5.0            # CLF V 中的 k2
Q_SQ = 1.0          # CLF V 中的 q^2 = k1/k3

# --- 停车目标 ---
X_TARGET = 0.0      # 目标点 X 坐标
Y_TARGET = 0.0      # 目标点 Y 坐标
V_TARGET_TERMINAL = 0.0 # 目标终端速度
STOP_THRESHOLD = 0.1    # 停车阈值

# --- GMM 参数 ---
NUM_GAUSSIANS = 10 # 保持 K=20


# ====================== 2. 构造 5-Tap 滤波器矩阵 F (保持不变) ======================
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


# ====================== 3. B-spline 基函数和导数矩阵 (保持不变) ======================
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


# ====================== 3. 笛卡尔轨迹生成 (返回绘图所需的所有控制点) ======================
@jit
def theta_to_cartesian_trajectory(theta, ctx):
    x_cur, y_cur, theta_cur, v_cur = ctx['x_cur'], ctx['y_cur'], ctx['theta_cur'], ctx['v_cur']

    Qx_opt = theta[:NUM_CONTROL_POINTS_OPT]; 
    Qy_opt = theta[NUM_CONTROL_POINTS_OPT:]
    
    # 锚点 0: 当前位置 (固定轨迹起点)
    x_anchor_0 = x_cur; y_anchor_0 = y_cur 
    
    # 锚点 1: 当前速度和方向的外推点 (固定轨迹初始速度/方向)
    x_anchor_1 = x_cur + v_cur * jnp.cos(theta_cur) * DT
    y_anchor_1 = y_cur + v_cur * jnp.sin(theta_cur) * DT

    Qx_anchors = jnp.array([x_anchor_0, x_anchor_1])
    Qy_anchors = jnp.array([y_anchor_0, y_anchor_1])

    Qx_full = jnp.concatenate([Qx_anchors, Qx_opt])
    Qy_full = jnp.concatenate([Qy_anchors, Qy_opt])

    # 只对优化部分（从索引2开始）施加滤波，构造一个“偏置”滤波矩阵
    # 简单方式：对整个 Q_full 滤波，但强制前两个点不变
    Px_full = F_MATRIX @ Qx_full
    Py_full = F_MATRIX @ Qy_full
    
    x_traj   = N5_BASIS      @ Px_full; y_traj   = N5_BASIS      @ Py_full
    x_dot    = N5_PRIME      @ Px_full; y_dot    = N5_PRIME      @ Py_full
    x_ddot   = N5_DOUBLE_PRIME @ Px_full; y_ddot   = N5_DOUBLE_PRIME @ Py_full

    return x_traj, y_traj, x_dot, y_dot, x_ddot, y_ddot, Qx_full, Qy_full, Px_full, Py_full


# ====================== 4. Pure CLF Cost 函数 (保持不变) ======================
@jit
def cartesian_pure_clf_cost(theta, ctx):
    
    x_traj, y_traj, x_dot, y_dot, x_ddot, y_ddot, _, _, _, _ = \
        theta_to_cartesian_trajectory(theta, ctx)
    
    theta_cur_val = ctx['theta_cur'] 
    
    # --- 1. 从 B-spline 导数计算物理量 ---
    
    
    
    
    # 极坐标量 (CLF 变量)
    rho_traj = jnp.sqrt(x_traj**2 + y_traj**2)
    delta_traj = jnp.arctan2(y_traj, x_traj) + jnp.pi 
    theta_traj = jnp.arctan(y_dot/x_dot) 
    gamma_traj = delta_traj - theta_traj +jnp.pi
    tilde_gamma_traj = gamma_traj + 0.5 * jnp.arctan(2.0 * K2 * delta_traj)
    
    v_s_traj = -(x_traj*x_dot + y_traj*y_dot) / (rho_traj * jnp.cos(gamma_traj) + 1e-9) 
    omega_traj= v_s_traj * jnp.sin(gamma_traj) / (rho_traj + 1e-9) + (y_traj*x_dot)/(rho_traj**2) -(x_traj*y_dot)/(rho_traj**2) +(y_ddot*x_dot-x_ddot*y_dot)/(rho_traj**2)
    # --- 2. 计算 CLF Cost 辅助项 \nu_1, \nu_2 ---
    epsilon_1 = 2.0 * BAR_V / (jnp.pi * (rho_traj + SIGMA)) 
    epsilon_2 = 2.0 * BAR_OMEGA / jnp.pi
    
    bracket_term =  Q_SQ * tilde_gamma_traj * \
                       (1.0 + K2 / (1.0 + 4.0 * K2**2 * delta_traj**2))
    nu1_traj = -2.0 * rho_traj**2 * jnp.cos(gamma_traj) + \
               2.0 * jnp.sin(gamma_traj) * (delta_traj+bracket_term)
    
    nu2_traj = -2.0*Q_SQ * tilde_gamma_traj
    
    # --- 3. 构造 Cost 函数 L + E ---

    # A. CLF 误差惩罚项 L_nu1,L_nu2 
    l_nu1_abs = jnp.abs(nu1_traj)
    nu1_denom = 2.0 * epsilon_1 * l_nu1_abs + 1e-9 
    cost_clf_nu1 = jnp.sum(2.0 * (jnp.arctan(epsilon_1 * l_nu1_abs) - jnp.log(1.0 + epsilon_1**2 * nu1_traj**2) / nu1_denom))*DT

    cost_clf_nu2=jnp.sum(2.0 * (jnp.arctan(epsilon_2 * jnp.abs(nu2_traj)) - jnp.log(1.0 + epsilon_2**2 * nu2_traj**2) / (2.0 * epsilon_2 * jnp.abs(nu2_traj) + 1e-9)))*DT

    # B. 速度饱和惩罚项 L_v 
    v_s_abs = jnp.abs(v_s_traj) 
    v_limit_denom = epsilon_1 * rho_traj + 1e-9
    v_arg = v_s_abs / v_limit_denom
    
    cos_v_safe = jnp.maximum(jnp.cos(v_arg), 1e-9)
    cost_v_sat = -jnp.sum(jnp.log(cos_v_safe)) * DT


    # C. 角速度饱和惩罚项 L_omega 
    omega_arg = jnp.abs(omega_traj) / (epsilon_2 + 1e-9)
    
    cos_omega_safe = jnp.maximum(jnp.cos(omega_arg), 1e-9)
    cost_omega_sat = -jnp.sum(jnp.log(cos_omega_safe)) * DT
    
    # 阶段积分项总和 (加权)
    cost_stage_integral = 0.5 * (cost_clf_nu1+cost_clf_nu2) + cost_v_sat + cost_omega_sat


    # D. 终端代价 E 
    rho_end = rho_traj[-1]
    delta_end = delta_traj[-1]
    tilde_gamma_end = tilde_gamma_traj[-1]
    v_s_end = v_s_traj[-1] 

    V_end = rho_end**2 + delta_end**2 + Q_SQ * (tilde_gamma_end**2)
    
    cost_terminal =  V_end + v_s_end**2
    
    
    # 总代价
    total_cost = cost_stage_integral + 100*cost_terminal
    
    return total_cost

#@jit
#def get_cost_breakdown(theta, ctx):
    
    x_traj, y_traj, x_dot, y_dot, x_ddot, y_ddot, _, _, _, _ = \
        theta_to_cartesian_trajectory(theta, ctx)
    theta_cur_val = ctx['theta_cur'] 
    
    # --- 1. 物理量计算 (与 cost 函数一致) ---
    v_s_traj = x_dot * jnp.cos(theta_cur_val) + y_dot * jnp.sin(theta_cur_val)
    v_mag_sq = x_dot**2 + y_dot**2
    omega_traj = (x_dot * y_ddot - x_ddot * y_dot) / (v_mag_sq + 1e-9) 
    
    rho_traj = jnp.sqrt(x_traj**2 + y_traj**2)
    delta_traj = jnp.arctan2(y_traj, x_traj) + jnp.pi 
    theta_traj = jnp.arctan2(y_dot, x_dot) 
    gamma_traj = delta_traj - theta_traj
    tilde_gamma_traj = gamma_traj + 0.5 * jnp.arctan(2.0 * K2 * delta_traj)
    
    # --- 2. CLF 辅助项 (与 cost 函数一致) ---
    epsilon_1 = 2.0 * BAR_V / (jnp.pi * (rho_traj + SIGMA)) 
    epsilon_2 = 2.0 * BAR_OMEGA / jnp.pi
    
    nu1_bracket_term = delta_traj + Q_SQ * tilde_gamma_traj * \
                       (1.0 + K2 / (1.0 + 4.0 * K2**2 * delta_traj**2))
    nu1_traj = -2.0 * rho_traj**2 * jnp.cos(gamma_traj) + \
               2.0 * jnp.sin(gamma_traj) * nu1_bracket_term
    
    nu2_traj = -2.0*Q_SQ * tilde_gamma_traj
    
    # --- 3. 阶段代价 L (包含 *DT 保证一致性) ---

    # A. CLF 误差惩罚项 L_nu1, L_nu2
    l_nu1_abs = jnp.abs(nu1_traj)
    nu1_denom = 2.0 * epsilon_1 * l_nu1_abs + 1e-9 
    cost_clf_nu1_int = jnp.sum(2.0 * (jnp.arctan(epsilon_1 * l_nu1_abs) - jnp.log(1.0 + epsilon_1**2 * nu1_traj**2) / nu1_denom))*DT

    cost_clf_nu2_int = jnp.sum(2.0 * (jnp.arctan(epsilon_2 * jnp.abs(nu2_traj)) - jnp.log(1.0 + epsilon_2**2 * nu2_traj**2) / (2.0 * epsilon_2 * jnp.abs(nu2_traj) + 1e-9)))*DT
    
    cost_clf_total_weighted = 0.5 * (cost_clf_nu1_int + cost_clf_nu2_int)

    # B. 速度饱和惩罚项 L_v 
    v_s_abs = jnp.abs(v_s_traj)
    v_limit_denom = epsilon_1 * rho_traj + 1e-9
    v_arg = v_s_abs / v_limit_denom
    
    cos_v_safe = jnp.maximum(jnp.cos(v_arg), 1e-9)
    cost_v_sat = -jnp.sum(jnp.log(cos_v_safe)) * DT


    # C. 角速度饱和惩罚项 L_omega 
    omega_arg = jnp.abs(omega_traj) / (epsilon_2 + 1e-9)
    
    cos_omega_safe = jnp.maximum(jnp.cos(omega_arg), 1e-9)
    cost_omega_sat = -jnp.sum(jnp.log(cos_omega_safe)) * DT
    
    
    # --- 4. 终端代价 E (与 cost 函数一致) ---
    rho_end = rho_traj[-1]
    delta_end = delta_traj[-1]
    tilde_gamma_end = tilde_gamma_traj[-1]
    v_s_end = v_s_traj[-1] 

    V_end = rho_end**2 + delta_end**2 + Q_SQ * (tilde_gamma_end**2)
    cost_terminal = V_end + v_s_end**2 
    
    # 返回四个分量，它们的和等于 total_cost
    return cost_clf_total_weighted, cost_v_sat, cost_omega_sat, cost_terminal

# ！！！ 使用 vmap 并行计算所有高斯均值的代价 ！！！
vmapped_cost_eval = vmap(cartesian_pure_clf_cost, in_axes=(0, None))


# ====================== 6. 主循环 (停车场景) ======================
def run_parking_demo():
    print(f"CLF-based B-spline 停车规划 (已取消硬截断, 启用关键诊断信息)")
    key = jax.random.PRNGKey(0)

    
    robot_x = 10.0; robot_y = -5.0; robot_v = 1.0 
    #initial_yaw_to_origin = jnp.arctan2(-robot_y, -robot_x)
    robot_theta = 0.0
    
    # --- GMM 初始化：构建分散的初始均值 mu_k (保持不变) ---
    
    x_diff = robot_v * DT * onp.cos(robot_theta)
    y_diff = robot_v * DT * onp.sin(robot_theta)
    
    Qs_opt_init = onp.arange(NUM_CONTROL_POINTS_OPT) * x_diff + (robot_x + 2 * x_diff)
    Ql_opt_init = onp.arange(NUM_CONTROL_POINTS_OPT) * y_diff + (robot_y + 2 * y_diff)
    
    theta0 = jnp.concatenate([jnp.array(Qs_opt_init), jnp.array(Ql_opt_init)])
    
    INITIAL_PERTURBATION_STD = 2.0 
    
    mu_k_list = []
    key, *subkeys_init = jax.random.split(key, NUM_GAUSSIANS + 1) 
    for i in range(NUM_GAUSSIANS): 
        random_perturbation = jax.random.normal(subkeys_init[i], shape=(TOTAL_DIM,)) * INITIAL_PERTURBATION_STD
        mu_k_list.append(theta0 + random_perturbation)
    
    mu_k = jnp.stack(mu_k_list)
    
    L_inv_k = jnp.stack([jnp.eye(TOTAL_DIM) * 2] * NUM_GAUSSIANS);
    pi_k = jnp.ones(NUM_GAUSSIANS) / NUM_GAUSSIANS

    plt.ion(); fig, ax = plt.subplots(figsize=(10, 10)); history = []
    
    Qs_k_full = jnp.zeros(NUM_CONTROL_POINTS_FULL)
    Ql_k_full = jnp.zeros(NUM_CONTROL_POINTS_FULL)
    Ps_k = jnp.zeros(NUM_CONTROL_POINTS_FULL)
    Pl_k = jnp.zeros(NUM_CONTROL_POINTS_FULL)

    for t in range(500): 
        rho_cur = jnp.sqrt(robot_x**2 + robot_y**2)
        
        # 终止条件
        if rho_cur <= STOP_THRESHOLD and jnp.abs(robot_v) <= 0.1:
            print(f"已成功停车 (rho={rho_cur:.2f}m, v={robot_v:.2f}m/s)，停止规划.")
            break

        key, subkey = jax.random.split(key)
        
        # 更新 Context
        ctx = {'x_cur': robot_x, 'y_cur': robot_y, 'theta_cur': robot_theta, 'v_cur': robot_v}

        steps = 3000 if t == 0 else 800
        t0 = time.time()
        
        pi_k = jnp.ones(NUM_GAUSSIANS) / NUM_GAUSSIANS
        L_inv_k = jnp.stack([jnp.eye(TOTAL_DIM) * 2] * NUM_GAUSSIANS)
        
        # 优化
        mu_k, L_inv_k, pi_k = igo_mog_optimizer(subkey, steps, 0.12, NUM_GAUSSIANS, 60, 25, 
                                                cartesian_pure_clf_cost, 
                                                mu_k, L_inv_k, pi_k, ctx)
        t1 = time.time()

        # ----------------------------------------------------
        # 核心：选择成本最低的均值作为最优解
        
        costs = vmapped_cost_eval(mu_k, ctx)
        min_cost_index = jnp.argmin(costs)
        best_theta = mu_k[min_cost_index]
        
       # cost_clf_weighted, cost_v_sat, cost_omega_sat, cost_term = get_cost_breakdown(best_theta, ctx)
        
        x_traj, y_traj, x_dot, y_dot ,x_ddot, y_ddot, Qs_k_full, Ql_k_full, Ps_k, Pl_k = \
            theta_to_cartesian_trajectory(best_theta, ctx)
            
        # ----------------------------------------------------
        
        # --- 车辆执行：状态更新 (严格遵循 Unicycle Kinematics, 无截断) ---
        
        # 1. 提取 t=0 时刻的控制命令 (v_cmd, omega_cmd)
        # 1. 提取 t=0 时刻的控制命令 (v_cmd, omega_cmd)
        # 1. 提取规划轨迹 t=0 处的几何信息
        # 1. 提取 t=0 时刻的控制命令 (v_cmd, omega_cmd)
        # 笛卡尔坐标 -> 极坐标 (应用于整个时域 [0, T])
        rho_traj = jnp.sqrt(x_traj**2 + y_traj**2)
        delta_traj = jnp.arctan2(y_traj, x_traj) + jnp.pi 
        theta_traj = jnp.arctan(y_dot/x_dot) 
        gamma_traj = delta_traj - theta_traj +jnp.pi
        
        # tilde_gamma_traj (CLF 变量，仅用于诊断，不用于执行)
        # tilde_gamma_traj = gamma_traj + 0.5 * jnp.arctan(2.0 * K2 * delta_traj)

        # ----------------------------------------------------
        # 【执行器修正 A：严格按照 CLF 极坐标公式提取 V_CMD 和 OMEGA_CMD】
        
        # 1. 计算 v_s_traj (投影速度，整个时域 [0, T])
        # v_s_traj = -(x_dot + y_dot) / (rho_traj * jnp.cos(gamma_traj))
        # ！！！ 注意：这里 jnp.cos(gamma_traj) 可能导致数值不稳定，但我们必须遵循原公式
        v_s_traj = -(x_traj*x_dot + y_traj*y_dot) / (rho_traj * jnp.cos(gamma_traj) + 1e-9) 
        
        # 2. 计算 omega_traj (角速度，整个时域 [0, T])
        # omega_traj = v_s_traj * jnp.sin(gamma_traj) / (rho_traj + 1e-9) + 
        #              (y_traj * x_dot - x_traj * y_dot) / (rho_traj**2 + 1e-9) + 
        #              (y_ddot * x_dot - x_ddot * y_dot) / (rho_traj**2 + 1e-9)
        omega_traj= v_s_traj * jnp.sin(gamma_traj) / (rho_traj + 1e-9) + (y_traj*x_dot)/(rho_traj**2) -(x_traj*y_dot)/(rho_traj**2) +(y_ddot*x_dot-x_ddot*y_dot)/(rho_traj**2)
        
        # ----------------------------------------------------
        # 3. 提取 t=0 时刻的控制命令 (执行器命令)
        v_cmd_exec = v_s_traj[0]
        omega_cmd_exec = omega_traj[0]
        
        # 4. 诊断信息打印
        print("-" * 50)
        # ... (省略打印内容)
        print(f" {v_s_traj[0]:.3f}")
        print(f"    当前航向: THETA_CUR={jnp.rad2deg(robot_theta):.1f}°")
        print(f"    执行命令: V_CMD={v_cmd_exec:.3f} m/s | OMEGA_CMD={omega_cmd_exec:.3f} rad/s")
        print("-" * 50)

        # ----------------------------------------------------
        # 【执行器修正 B：高精度运动学积分（与模型解耦）】
        
        # 1. 航向更新
        robot_theta_new = robot_theta + omega_cmd_exec * DT 
        
        # 2. 计算平均航向角 (Midpoint/Runge-Kutta 2 阶近似，消除状态漂移)
        theta_avg = (robot_theta + robot_theta_new) / 2.0 

        # 3. 位置更新：使用平均航向角进行积分 (单轮车模型)
        # 严格遵循: d/dt(x) = v * cos(theta_avg)
        robot_x_new = robot_x + v_cmd_exec * jnp.cos(theta_avg) * DT
        robot_y_new = robot_y + v_cmd_exec * jnp.sin(theta_avg) * DT

        # 4. 更新状态变量
        robot_x = robot_x_new
        robot_y = robot_y_new
        robot_v = v_cmd_exec  # 执行的线速度 (有符号)
        
        # 5. 强制归一化 theta 到 [-pi, pi] 
        robot_theta=robot_theta_new
        #robot_theta = (robot_theta_unwrapped + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
        # ----------------------------------------------------
        v_s_next = v_cmd_exec # 用于打印

        history.append(jnp.array([robot_x, robot_y, robot_theta, robot_v]))

        # 打印 Cost Breakdown 
        min_cost_val = costs[min_cost_index]
        print(f"Step {t:02d} | X:{robot_x:6.1f}m Y:{robot_y:5.1f}m | Rho:{rho_cur:4.1f}m | V:{robot_v:4.1f}m/s | OPT:{(t1-t0)*1000:4.0f}ms")
       # print(f"       | Min Cost:{min_cost_val:.2f} | Stage Breakdown: CLF_nu:{cost_clf_weighted:.2f} | L_v:{cost_v_sat:.2f} | L_omega:{cost_omega_sat:.2f} | E_term:{cost_term:.2f}")
        
        # ----------------------------------------------------
        # Warm Start 修正: 确保连续性 (保持不变)
        
        theta_base_guess = best_theta
        
        mu_k_list = [theta_base_guess] 
        key, *subkeys = jax.random.split(key, NUM_GAUSSIANS) 
        RUNTIME_PERTURBATION_STD = 0.1
        
        for i in range(NUM_GAUSSIANS - 1): 
            random_perturbation = jax.random.normal(subkeys[i], shape=(TOTAL_DIM,)) * RUNTIME_PERTURBATION_STD
            mu_k_list.append(theta_base_guess + random_perturbation)
        
        mu_k = jnp.stack(mu_k_list)
        # ----------------------------------------------------

        if t % 1 == 0: 
            ax.cla()
            ax.set_xlim(-5, 15) 
            ax.set_ylim(-15, 5)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')

            # --- 绘制目标点 (原点) ---
            ax.plot(0, 0, 'r*', ms=20, label='Parking Target (0, 0)')
            
            # 绘制 Ego 历史轨迹
            hist = jnp.array(history)
            ax.plot(hist[:,0], hist[:,1], 'b-', lw=4, alpha=0.5, label='Ego History')

            # 绘制规划的中心线轨迹 (B-spline 采样点)
            ax.plot(x_traj, y_traj, 'cyan', lw=5, alpha=0.7, label='Planned Trajectory')
            # 标记轨迹的起点 (当前位置)
            ax.plot(x_traj[0], y_traj[0], 'go', ms=10)
            
            # --- 绘制 Q, P 点 ---
            ax.plot(Qs_k_full, Ql_k_full, 'rx', ms=8, mew=2, label='Optimization Variables (Q)')
            ax.plot(Ps_k, Pl_k, 'ko', ms=6, label='B-spline Control Points (P)')
            
            # --- 标题和图例 ---
            ax.set_title(f"CLF-MPC Cartesian Parking | X:{robot_x:.1f}m Y:{robot_y:.1f}m | Rho:{jnp.sqrt(robot_x**2 + robot_y**2):.1f}m | V:{robot_v:4.1f}m/s")
            ax.legend()
            plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_parking_demo()