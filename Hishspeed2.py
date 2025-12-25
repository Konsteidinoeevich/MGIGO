# MPCmain4_Final_Velocity_Projection_WarmStart.py
# 2025 真实量产级 MPC —— L1 Minimax EPF + 纵向速度投影 Warm Start (针对用户痛点优化)

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from gmm_igo.MPCsolver import igo_mog_optimizer 
from scipy.interpolate import BSpline
import numpy as onp

# ====================== 1. 全局配置 (关键更新) ======================
DT = 0.1
HORIZON = 100
TOTAL_TIME = HORIZON * DT  # 10.0 seconds
POLY_ORDER = 5       
NUM_CONTROL_POINTS_FULL = 15 
NUM_CONTROL_POINTS_OPT = 13  
TOTAL_DIM = 2 * NUM_CONTROL_POINTS_OPT 
LANE_WIDTH = 3.7   
V_MAX = 35.0; V_MIN = 10.0 
V_TARGET_TERMINAL = 25.0 
X_TARGET = 1000.0 
KNOT_DELTA = 1.0 # B-spline 结点间隔 (秒)

# Minimax 统一 EPF 惩罚系数
C_CRITICAL_EPF = 5000.0 
C_KINEMATIC_EPF = 1000.0 
MAX_KINEMATIC_RATIO = jnp.tan(jnp.radians(15.0)) 
L_MAX_BOUNDARY = 1.5 
TAU = 0.5; A_BRAKE = 4.0; A_ACCEL_MIN = -2.0; MARGIN_LEADING = 1.0; MARGIN_LAT = 0.2


# ====================== 2-5. 矩阵/B-spline/theta_to_trajectory/Cost (保持不变) ======================
# ... (Functions create_5_tap_filter_matrix, compute_basis_matrix, theta_to_trajectory, and overtake_cost remain identical to the previous version, 
# as the structural change is only in the warm start logic in the main loop.)
def create_5_tap_filter_matrix(N):
    F = onp.zeros((N, N))
    W = onp.array([1, 26, 66, 26, 1]) / 120.0
    for i in range(N):
        if i == 0 or i == N - 1: F[i, i] = 1.0; continue
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

POLY_ORDER = 5       
INTERNAL_KNOTS_COUNT = NUM_CONTROL_POINTS_FULL - POLY_ORDER 
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
N5_BASIS          = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=0) 
N5_PRIME          = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=1) 
N5_DOUBLE_PRIME   = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=2) 
N5_TRIPLE_PRIME   = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=3) 

@jit
def theta_to_trajectory(theta, ctx):
    s_cur   = ctx['s_cur']; l_cur   = ctx['l_cur']; ds_cur  = ctx['ds_cur']
    Qs_opt = theta[:NUM_CONTROL_POINTS_OPT]; Ql_opt = theta[NUM_CONTROL_POINTS_OPT:]
    s_anchor_0 = s_cur; l_anchor_0 = l_cur
    s_anchor_1 = s_cur + ds_cur * DT; l_anchor_1 = l_cur
    Qs_anchors = jnp.array([s_anchor_0, s_anchor_1]); Ql_anchors = jnp.array([l_anchor_0, l_anchor_1])
    Qs_full = jnp.concatenate([Qs_anchors, Qs_opt]); Ql_full = jnp.concatenate([Ql_anchors, Ql_opt])
    Ps = F_MATRIX @ Qs_full; Pl = F_MATRIX @ Ql_full
    s_traj   = N5_BASIS @ Ps; l_traj   = N5_BASIS @ Pl
    s_dot    = N5_PRIME @ Ps; l_dot    = N5_PRIME @ Pl
    s_ddot   = N5_DOUBLE_PRIME @ Ps; l_ddot   = N5_DOUBLE_PRIME @ Pl
    s_dddot  = N5_TRIPLE_PRIME @ Ps; l_dddot  = N5_TRIPLE_PRIME @ Pl
    traj = jnp.stack([s_traj, l_traj * LANE_WIDTH], axis=1)
    return traj, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot

@jit
def overtake_cost(theta, ctx):
    _, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot = theta_to_trajectory(theta, ctx)
    lead_boxes = ctx['lead_boxes']; V_lead_array = ctx['V_lead_array'] 
    Y_target = ctx['Y_target']; X_target = ctx['X_target'] 
    y_traj = l_traj * LANE_WIDTH; ddot_x = s_ddot; ddot_y = l_ddot * LANE_WIDTH
    a_mag = jnp.sqrt(ddot_x**2 + ddot_y**2)
    cost_stage_deviation = jnp.sum(10.0 * (y_traj - Y_target)**2 + 5.0 * (s_traj - X_target)**2)
    cost_term = 10.0 * (y_traj[-1] - Y_target)**2 + 50.0 * (s_traj[-1] - X_target)**2 + 50.0 * (s_dot[-1] - V_TARGET_TERMINAL)**2 
    cost_acc = 10.0 * jnp.sum(jnp.maximum(0.0, a_mag - 5.0)**2)
    cost_lat = 0.1 * jnp.sum(ddot_y**2); cost_jerk = 10.0 * jnp.sum(s_dddot**2 + l_dddot**2) 
    
    # 1. Kinematic EPF 
    kinematic_ratio = LANE_WIDTH * jnp.abs(l_dot) / (s_dot + 1e-6) 
    violation_kin = jnp.maximum(0.0, kinematic_ratio - MAX_KINEMATIC_RATIO)
    cost_EPF_kinematic = C_KINEMATIC_EPF * jnp.sum(violation_kin)

    # 2. RSS EPF 
    N = lead_boxes.shape[0]; V_ego_batch = jnp.tile(s_dot, (N, 1)); V_lead_batch = V_lead_array.reshape(N, 1)
    V_diff = V_ego_batch - V_lead_batch; V_diff_pos = jnp.maximum(0.0, V_diff) 
    margin_lon = V_ego_batch * TAU + V_diff_pos**2 / (2.0 * (A_BRAKE - A_ACCEL_MIN))
    s_traj_batch = jnp.tile(s_traj, (N, 1))
    lead_s_center = (lead_boxes[:, 0, 0] + lead_boxes[:, 1, 0]) / 2.0 
    s_lead_front = lead_boxes[:, 1, 0].reshape(N, 1); s_lead_rear = lead_boxes[:, 0, 0].reshape(N, 1) 
    s_ego_front = s_traj_batch + 2.4; s_ego_rear = s_traj_batch - 2.4
    clearance_trailing = s_lead_rear - s_ego_front; violation_lon_trailing = jnp.maximum(0.0, margin_lon - clearance_trailing) 
    clearance_leading = s_ego_rear - s_lead_front; violation_lon_leading = jnp.maximum(0.0, MARGIN_LEADING - clearance_leading) 
    ego_is_trailing = (s_traj_batch < lead_s_center.reshape(N, 1)).astype(jnp.float32)
    violation_lon = (violation_lon_trailing * ego_is_trailing) + (violation_lon_leading * (1.0 - ego_is_trailing))
    lead_y_center = (lead_boxes[:, 0, 1] + lead_boxes[:, 1, 1]) / 2.0 
    Y_SEP_ACT_BOXES = jnp.abs(lead_y_center.reshape(N, 1) - jnp.tile(y_traj, (N, 1))) - 0.95 - 0.95 
    violation_lat = jnp.maximum(0.0, MARGIN_LAT - Y_SEP_ACT_BOXES)
    S_SEP_THRESHOLD = 20.0 
    is_close_lon = (jnp.abs(s_traj_batch - lead_s_center.reshape(N, 1)) < S_SEP_THRESHOLD).astype(jnp.float32)
    cost_EPF_RSS = C_CRITICAL_EPF * (jnp.sum(violation_lon) + jnp.sum(violation_lat * is_close_lon))

    # 3. Boundary EPF 
    violation_boundary = jnp.maximum(0.0, jnp.abs(l_traj) - L_MAX_BOUNDARY)
    cost_EPF_Boundary = C_CRITICAL_EPF * jnp.sum(violation_boundary)
    
    # 4. Minimax Critical EPF 
    cost_EPF_Critical_Minimax = jnp.maximum(cost_EPF_RSS, cost_EPF_Boundary)

    return cost_term + cost_stage_deviation + cost_acc + cost_lat + cost_jerk + cost_EPF_kinematic + cost_EPF_Critical_Minimax


# ====================== 6. 主循环 (改进 Warm Start) ======================
def run_overtake_demo():
    print(f"2025 真实量产级 MPC —— 纵向速度投影 Warm Start 启动！")
    key = jax.random.PRNGKey(0)

    # 初始状态
    robot_s = 0.0; robot_l = 0.0; robot_ds = 30.0 
    
    # 障碍物初始化 
    lead_car_states = [
        {'s': 100.0, 'l': 0.0, 'v': 20.0},  
        {'s': 250.0, 'l': -1.0, 'v': 25.0}, 
        {'s': 400.0, 'l': 1.0, 'v': 30.0}   
    ]
    
    # 1. 构造初始 Q 向量 (长度 26)
    Qs0_full = onp.arange(NUM_CONTROL_POINTS_FULL) * KNOT_DELTA * 30.0 # Q_i = i * 30 m (V=30)
    Ql0_full = onp.zeros(NUM_CONTROL_POINTS_FULL)
    theta0 = jnp.concatenate([jnp.array(Qs0_full[2:]), jnp.array(Ql0_full[2:])])
    
    # 初始化 iGO-MoG 优化器的初始分布
    mu_k = jnp.stack([theta0] * 6); 
    L_inv_k = jnp.stack([jnp.eye(TOTAL_DIM) * 2.0] * 6); 
    pi_k = jnp.ones(6) / 6.0

    plt.ion(); fig, ax = plt.subplots(figsize=(15, 7)); history = []

    for t in range(250): 
        if robot_s >= X_TARGET:
            print(f"达到目标距离 {X_TARGET:.1f}m，停止规划.")
            break

        key, subkey = jax.random.split(key)
        l_target=0.0
        Y_TARGET = l_target * LANE_WIDTH 
        
        current_lead_boxes = []; V_lead_array = []
        for i in range(len(lead_car_states)):
            car = lead_car_states[i]
            car['s'] += car['v'] * DT 
            s_center = car['s']; y_center = car['l'] * LANE_WIDTH
            lead_box = [[s_center - 2.4, y_center - 0.95], [s_center + 2.4, y_center + 0.95]]
            current_lead_boxes.append(lead_box)
            V_lead_array.append(car['v'])
        
        JAX_lead_boxes = jnp.array(current_lead_boxes) 
        JAX_V_lead_array = jnp.array(V_lead_array)
        
        ctx = {'s_cur': robot_s, 'l_cur': robot_l, 'ds_cur': robot_ds, 
               'Y_target': Y_TARGET, 'lead_boxes': JAX_lead_boxes, 
               'X_target': X_TARGET, 'V_lead_array': JAX_V_lead_array}

        steps = 3000 if t == 0 else 450
        t0 = time.time()
        
        mu_k, L_inv_k, pi_k = igo_mog_optimizer(subkey, steps, 0.12, 6, 60, 25, overtake_cost, mu_k, L_inv_k, pi_k, ctx)
        t1 = time.time()

        best_theta = mu_k[jnp.argmax(pi_k)]
        
        best_traj, s_traj, l_traj, s_dot, l_dot, _, _, _, _ = theta_to_trajectory(best_theta, ctx)
        
        # --- 车辆执行：状态更新 ---
        robot_ds_next_raw = s_dot[1] 
        robot_ds_next = jnp.clip(robot_ds_next_raw, V_MIN, V_MAX) 

        robot_s_next = s_traj[1] 
        robot_l_next = l_traj[1]
        
        robot_s = robot_s_next; 
        robot_l = robot_l_next; 
        robot_ds = robot_ds_next
        
        history.append(jnp.array([robot_s, robot_l * LANE_WIDTH]))

        print(f"Step {t:02d} | S:{robot_s:6.1f}m L:{robot_l*LANE_WIDTH:5.1f}m V:{robot_ds:4.1f}m/s | OPT:{(t1-t0)*1000:4.0f}ms")

        # ！！！ 改进的 Warm Start 机制 ！！！
        
        # 1. 横向 Warm Start (Robust Shift: Index Shift + Extrapolation)
        Ql_k_old = best_theta[NUM_CONTROL_POINTS_OPT:]
        Ql_shifted = Ql_k_old[1:] 
        Ql_new_end = Ql_k_old[-1] 
        Ql_next_guess = jnp.concatenate([Ql_shifted, jnp.array([Ql_new_end])])
        
        # 2. 纵向 Warm Start (Velocity Projection: Reset Qs to linear speed profile)
        
        # Qs_opt 的索引 (i=0..12) 对应于 B-spline 全控制点 Qs[2] 到 Qs[14]
        # Qs[i] 大致影响 t = i * KNOT_DELTA 附近, 且 Qs_opt 索引从 2 开始。
        Qs_indices_shifted = jnp.arange(NUM_CONTROL_POINTS_OPT)
        # 优化变量 Qs_opt[i] 对应于 B-spline 索引 i+2
        Qs_time_factor = Qs_indices_shifted + 2.0 # (2.0, 3.0, 4.0, ..., 14.0)
        
        # Qs_next_guess = 新的起始绝对位置 + (时间因子 * 结点间隔) * 新的速度
        # 由于 KNOT_DELTA = 1.0s, 简化为:
        Qs_next_guess = robot_s + Qs_time_factor * robot_ds
        
        theta_next_guess = jnp.concatenate([Qs_next_guess, Ql_next_guess])
        
        # 用新的猜测更新 iGO-MoG 初始分布的均值
        mu_k = mu_k.at[:].set(theta_next_guess)
        
        
        # --- 绘图 (保持不变) ---
        if t % 1 == 0: 
            ax.cla(); ax.set_xlim(robot_s - 30, robot_s + 200); ax.set_ylim(-15, 15)
            # 绘制车道线和边界 
            for y in [-11.1, -7.4, -3.7, 0, 3.7, 7.4, 11.1]: ax.axhline(y, color='gray', ls='--', alpha=0.3)
            ax.axhline(L_MAX_BOUNDARY * LANE_WIDTH, color='black', ls='-', lw=2, label='Lane Boundary')
            ax.axhline(-L_MAX_BOUNDARY * LANE_WIDTH, color='black', ls='-', lw=2)

            # 绘制所有障碍物
            for car_state in lead_car_states:
                s_center = car_state['s']
                y_center = car_state['l'] * LANE_WIDTH
                ax.add_patch(Rectangle((s_center-2.4, y_center-0.95), 4.8, 1.9, 
                                       color='red', alpha=0.9, label='Lead Cars' if car_state == lead_car_states[0] else None))
            
            # 绘制自车 (Ego) 的体积
            ego_x_center = robot_s; ego_y_center = robot_l * LANE_WIDTH
            ego_rect_x = ego_x_center - 2.4; ego_rect_y = ego_y_center - 0.95
            ax.add_patch(Rectangle((ego_rect_x, ego_rect_y), 4.8, 1.9, 
                                   color='green', alpha=0.9, label='Ego Car Body'))
            
            hist = jnp.array(history)
            ax.plot(hist[:,0], hist[:,1], 'b-', lw=4, label='Ego History')
            ax.plot(s_traj, l_traj*LANE_WIDTH, 'cyan', lw=5, alpha=0.7, label='Ego Planned Centerline')
            ax.plot(s_traj[0], l_traj[0]*LANE_WIDTH, 'go', ms=15)
            
            # --- 绘制 Q, P 点 ---
            # Qs_k_old 这里指的是优化前的 theta_next_guess (即 Warm Start 结果)
            Qs_k_old_vis = theta_next_guess[:NUM_CONTROL_POINTS_OPT] 
            Ql_k_old_vis = theta_next_guess[NUM_CONTROL_POINTS_OPT:]
            
            s_anchor_0 = s_traj[0]; l_anchor_0 = l_traj[0] 
            s_anchor_1 = s_dot[0] * DT + s_traj[0]; l_anchor_1 = l_traj[0] 

            Qs_k_full = jnp.concatenate([jnp.array([s_anchor_0, s_anchor_1]), Qs_k_old_vis])
            Ql_k_full = jnp.concatenate([jnp.array([l_anchor_0, l_anchor_1]), Ql_k_old_vis])

            ax.plot(Qs_k_full, Ql_k_full*LANE_WIDTH, 'rx', ms=8, label='Optimization Variables (Q)')
            Ps_k = F_MATRIX @ Qs_k_full
            Pl_k = F_MATRIX @ Ql_k_full
            ax.plot(Ps_k, Pl_k*LANE_WIDTH, 'ko', ms=6, label='B-spline Control Points (P)')
            
            ax.set_title(f"2025 Production Overtake (Velocity Projection Warm Start) | S:{robot_s:.1f}m / {X_TARGET:.1f}m | V_cur:{robot_ds:4.1f}m/s")
            ax.legend()
            plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_overtake_demo()