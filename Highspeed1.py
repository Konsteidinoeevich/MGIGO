# MPCmain4_Final_RSS_EPF.py
# 2025 真实量产级 MPC —— 引入 L1 精确罚函数 (EPF) 强制 RSS 碰撞安全距离

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

# ====================== 1. 全局配置 (新增 RSS 和运动学常数) ======================
DT = 0.1
HORIZON = 100
TOTAL_TIME = HORIZON * DT  # 10.0 seconds
POLY_ORDER = 5       

# --- 关键维度 ---
NUM_CONTROL_POINTS_FULL = 10 
NUM_CONTROL_POINTS_OPT = 8  
TOTAL_DIM = 2 * NUM_CONTROL_POINTS_OPT 

# --- 关键常数 ---
LANE_WIDTH = 3.7   
V_MAX = 45.0; V_MIN = 10.0 
V_TARGET_TERMINAL = 33.0 
X_TARGET = 1500.0 

# EPF 和运动学约束常数
C_BOUNDARY =1e6
C_EPF = 1e4 
C_KINEMATIC = 10.0 
MAX_KINEMATIC_RATIO = jnp.tan(jnp.radians(15.0)) 

# ！！！ RSS 常数 ！！！
TAU = 0.3            # 反应时间 (s)
A_BRAKE = 4.0        # Ego 最大制动 (m/s^2)
A_ACCEL_MIN = -2.0   # Lead 最小加速度 (m/s^2, 考虑紧急制动)
MARGIN_LEADING = 10.0 # Ego 领先时的最小纵向安全距离 (m)
MARGIN_LAT = 0.8     # 最小横向安全间距 (m)
L_MAX_BOUNDARY = 1.5  # 最大允许横向位移 (Frenet l 坐标，1.5 = 左车道中心线到最左边界/右车道中心线到最右边界)


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
N5_TRIPLE_PRIME   = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=3) 


# ====================== 4. theta_to_trajectory (保持不变) ======================
@jit
def theta_to_trajectory(theta, ctx):
    s_cur   = ctx['s_cur']; l_cur   = ctx['l_cur']; ds_cur  = ctx['ds_cur']
    
    Qs_opt = theta[:NUM_CONTROL_POINTS_OPT]; 
    Ql_opt = theta[NUM_CONTROL_POINTS_OPT:]
    
    s_anchor_0 = s_cur
    l_anchor_0 = l_cur
    s_anchor_1 = s_cur + ds_cur * DT
    l_anchor_1 = l_cur
    
    Qs_anchors = jnp.array([s_anchor_0, s_anchor_1])
    Ql_anchors = jnp.array([l_anchor_0, l_anchor_1])

    Qs_full = jnp.concatenate([Qs_anchors, Qs_opt])
    Ql_full = jnp.concatenate([Ql_anchors, Ql_opt])

    Ps = F_MATRIX @ Qs_full; Pl = F_MATRIX @ Ql_full
    
    s_traj   = N5_BASIS      @ Ps; l_traj   = N5_BASIS      @ Pl
    s_dot    = N5_PRIME      @ Ps; l_dot    = N5_PRIME      @ Pl
    s_ddot   = N5_DOUBLE_PRIME @ Ps; l_ddot   = N5_DOUBLE_PRIME @ Pl
    s_dddot  = N5_TRIPLE_PRIME @ Ps; l_dddot  = N5_TRIPLE_PRIME @ Pl

    traj = jnp.stack([s_traj, l_traj * LANE_WIDTH], axis=1)

    return traj, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot


# ====================== 5. 代价函数 (RSS 约束) ======================
@jit
def overtake_cost(theta, ctx):
    _, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot = \
        theta_to_trajectory(theta, ctx)
    
    lead_boxes = ctx['lead_boxes'] 
    V_lead_array = ctx['V_lead_array'] # 障碍车速度 (N,)
    Y_target = ctx['Y_target'] 
    X_target = ctx['X_target'] 
    
    # --- 动力学项计算 (保持不变) ---
    y_traj = l_traj * LANE_WIDTH
    ddot_x = s_ddot; ddot_y = l_ddot * LANE_WIDTH                        
    a_mag = jnp.sqrt(ddot_x**2 + ddot_y**2)             
    dddot_x = s_dddot; dddot_y = l_dddot * LANE_WIDTH  
    
    # 基础代价 (保持不变)
    cost_stage_deviation = jnp.sum(1.0 * (y_traj - Y_target)**2 + 5.0 * (s_traj - X_target)**2)
    cost_term = 0.5 * (y_traj[-1] - Y_target)**2 + 10.0 * (s_traj[-1] - X_target)**2 + 50.0 * (s_dot[-1] - V_TARGET_TERMINAL)**2 
    cost_acc = 1.0 * jnp.sum(jnp.maximum(0.0, a_mag - 5.0)**2)
    cost_lat = 0.1 * jnp.sum(ddot_y**2) 
    cost_jerk = 1.0 * jnp.sum(dddot_x**2 + dddot_y**2) 
    
    # EPF 运动学耦合约束 (保持不变)
    kinematic_ratio = LANE_WIDTH * jnp.abs(l_dot) / (s_dot + 1e-6) 
    violation_kin = jnp.maximum(0.0, kinematic_ratio - MAX_KINEMATIC_RATIO)
    cost_EPF_kinematic = C_KINEMATIC * jnp.sum(violation_kin)
    
    violation_boundary = jnp.maximum(0.0, jnp.abs(l_traj) - L_MAX_BOUNDARY)
    cost_EPF_boundary = C_BOUNDARY * jnp.sum(violation_boundary)

    # ！！！ 碰撞代价：基于 RSS 精确罚函数 ！！！
    N = lead_boxes.shape[0] # 障碍物数量
    
    # 1. RSS 参数计算 (Shape (N, H))
    V_ego_batch = jnp.tile(s_dot, (N, 1))
    V_lead_batch = V_lead_array.reshape(N, 1)

    V_diff = V_ego_batch - V_lead_batch
    V_diff_pos = jnp.maximum(0.0, V_diff) 
    
    # 动态纵向安全裕度 Margin_lon (Ego trailing)
    margin_lon = V_ego_batch * TAU + V_diff_pos**2 / (2.0 * (A_BRAKE - A_ACCEL_MIN))
    
    # 2. 实际车体边界和中心点 (Shape (N, H) / (N, 1))
    s_traj_batch = jnp.tile(s_traj, (N, 1))
    
    lead_s_center = (lead_boxes[:, 0, 0] + lead_boxes[:, 1, 0]) / 2.0 # (N,)
    lead_y_center = (lead_boxes[:, 0, 1] + lead_boxes[:, 1, 1]) / 2.0 # (N,)
    
    # 3. 纵向 RSS 约束 (避免追尾和被追尾)
    
    # 障碍车前沿 S_lead_front (N, 1)
    s_lead_front = lead_boxes[:, 1, 0].reshape(N, 1) 
    # 障碍车后沿 S_lead_rear (N, 1)
    s_lead_rear = lead_boxes[:, 0, 0].reshape(N, 1) 
    
    # 自车前沿 S_ego_front (N, H)
    s_ego_front = s_traj_batch + 2.4
    # 自车后沿 S_ego_rear (N, H)
    s_ego_rear = s_traj_batch - 2.4

    # Case A: Ego Trailing (S_ego < S_lead)
    # 违规 = Margin_lon - (Lead_rear - Ego_front)
    clearance_trailing = s_lead_rear - s_ego_front 
    violation_lon_trailing = jnp.maximum(0.0, margin_lon - clearance_trailing) 
    
    # Case B: Ego Leading (S_ego > S_lead)
    # 违规 = Margin_leading - (Ego_rear - Lead_front)
    clearance_leading = s_ego_rear - s_lead_front
    violation_lon_leading = jnp.maximum(0.0, MARGIN_LEADING - clearance_leading) 
    
    # 判定 Ego Trailing/Leading (使用中心点，避免边界重叠时的判定歧义)
    ego_is_trailing = (s_traj_batch < lead_s_center.reshape(N, 1)).astype(jnp.float32)
    
    violation_lon = (violation_lon_trailing * ego_is_trailing) + (violation_lon_leading * (1.0 - ego_is_trailing))
    
    # 4. 横向 RSS 约束 (避免侧面碰撞)
    
    # 实际横向车体间距 (中心距 - 宽度)
    Y_SEP_ACT_BOXES = jnp.abs(lead_y_center.reshape(N, 1) - jnp.tile(y_traj, (N, 1))) - 0.95 - 0.95 
    
    # 横向违规 = Margin_lat - 实际间距
    violation_lat = jnp.maximum(0.0, MARGIN_LAT - Y_SEP_ACT_BOXES)
    
    # 仅在纵向距离小于 20m 时惩罚横向违规 (避免惩罚远处的车)
    S_SEP_THRESHOLD = 10.0 
    is_close_lon = (jnp.abs(s_traj_batch - lead_s_center.reshape(N, 1)) < S_SEP_THRESHOLD).astype(jnp.float32)
    
    # 总 RSS 碰撞代价
    cost_collision = C_EPF * jnp.sum(violation_lon) + C_EPF * jnp.sum(violation_lat * is_close_lon)

    return cost_term + cost_stage_deviation + cost_acc + cost_lat + cost_jerk + cost_EPF_kinematic + cost_collision + cost_EPF_boundary


# ====================== 6. 主循环 (更新 Context 传递 V_lead) ======================
def run_overtake_demo():
    print(f"2025 真实量产级 MPC —— L1 EPF RSS 安全距离 ({X_TARGET}m) 启动！")
    key = jax.random.PRNGKey(0)

    # 初始状态
    robot_s = 0.0; robot_l = 0.0; robot_ds = 30.0 
    
    # 障碍物初始化 
    lead_car_states = [
        {'s': 100.0, 'l': 0.0, 'v': 20.0},  # 障碍车 1: 中车道, 20 m/s
        {'s': 150.0, 'l': -1.0, 'v': 21.0}, # 障碍车 2: 右车道, 25 m/s
        {'s': 200.0, 'l': 1.0, 'v': 22.0},   # 障碍车 3: 左车道, 30 m/s 
        {'s': 225.0, 'l': 0.0, 'v': 23.0} ,  # 障碍车 4: 左车道, 30 m/s 
        {'s': 250.0, 'l': 1.0, 'v': 24.0} ,  # 障碍车 5: 左车道, 30 m/s 
        {'s': 300.0, 'l': -1.0, 'v': 25.0},   # 障碍车 6: 左车道, 30 m/s 
        {'s': 330.0, 'l': 1.0, 'v': 26.0},  # 障碍车 6: 左车道, 30 m/s 
        {'s': 360.0, 'l': 0.0, 'v': 27.0},   # 障碍车 6: 左车道, 30 m/s 
        {'s': 390.0, 'l': -1.0, 'v': 28.0},   # 障碍车 6: 左车道, 30 m/s 
        {'s': 420.0, 'l': 0.0, 'v': 29.0},   # 障碍车 6: 左车道, 30 m/s 
        {'s': 450.0, 'l': 1.0, 'v': 30.0},   # 障碍车 6: 左车道, 30 m/s 
        {'s': 480.0, 'l': -1.0, 'v': 30.0}   # 障碍车 6: 左车道, 30 m/s 
    ]
    
    # 1. 构造初始 Q 向量 (长度 26)
    Qs0_full = onp.arange(NUM_CONTROL_POINTS_FULL) * KNOT_DELTA * 30.0 / KNOT_DELTA
    Ql0_full = onp.zeros(NUM_CONTROL_POINTS_FULL)
    theta0 = jnp.concatenate([jnp.array(Qs0_full[2:]), jnp.array(Ql0_full[2:])])
    
    # 初始化 iGO-MoG 优化器的初始分布
    mu_k = jnp.stack([theta0] * 6); 
    L_inv_k = jnp.stack([jnp.eye(TOTAL_DIM) * 2.0] * 6); 
    pi_k = jnp.ones(6) / 6.0

    plt.ion(); fig, ax = plt.subplots(figsize=(15, 7)); history = []

    for t in range(350): 
        if robot_s >= X_TARGET:
            print(f"达到目标距离 {X_TARGET:.1f}m，停止规划.")
            break

        key, subkey = jax.random.split(key)
        
        l_target=0.0
        Y_TARGET = l_target * LANE_WIDTH 
        
        # ！！！ 障碍物状态更新，并收集 V_lead_array ！！！
        current_lead_boxes = []
        V_lead_array = []
        for i in range(len(lead_car_states)):
            car = lead_car_states[i]
            car['s'] += car['v'] * DT 
            s_center = car['s']
            y_center = car['l'] * LANE_WIDTH
            lead_box = [[s_center - 2.4, y_center - 0.95], [s_center + 2.4, y_center + 0.95]]
            current_lead_boxes.append(lead_box)
            V_lead_array.append(car['v'])
        
        JAX_lead_boxes = jnp.array(current_lead_boxes) 
        JAX_V_lead_array = jnp.array(V_lead_array)
        
        # ！！！ 更新 Context ！！！
        ctx = {'s_cur': robot_s, 'l_cur': robot_l, 'ds_cur': robot_ds, 
               'Y_target': Y_TARGET, 'lead_boxes': JAX_lead_boxes, 
               'X_target': X_TARGET, 'V_lead_array': JAX_V_lead_array}

        steps = 2000 if t == 0 else 450
        t0 = time.time()
        
        mu_k, L_inv_k, pi_k = igo_mog_optimizer(subkey, steps, 0.12, 6, 60, 25, overtake_cost, mu_k, L_inv_k, pi_k, ctx)
        t1 = time.time()

        best_theta = mu_k[jnp.argmax(pi_k)]
        
        best_traj, s_traj, l_traj, s_dot, l_dot, _, _, _, _ = theta_to_trajectory(best_theta, ctx)
        
        # --- 车辆执行：保持一致性更新 ---
        robot_ds_next_raw = s_dot[1] 
        robot_ds_next = jnp.clip(robot_ds_next_raw, V_MIN, V_MAX) 

        robot_s_next = robot_s + robot_ds_next * DT
        robot_l_next = l_traj[1]
        
        shift_s = robot_s_next - robot_s 
        
        robot_s = robot_s_next; 
        robot_l = robot_l_next; 
        robot_ds = robot_ds_next
        
        history.append(jnp.array([robot_s, robot_l * LANE_WIDTH]))

        print(f"Step {t:02d} | S:{robot_s:6.1f}m L:{robot_l*LANE_WIDTH:5.1f}m V:{robot_ds:4.1f}m/s (Target V:{V_TARGET_TERMINAL:.0f}m/s) | OPT:{(t1-t0)*1000:4.0f}ms")

        # --- 重调度 (Re-scheduling) ---
        Qs_k_old = best_theta[:NUM_CONTROL_POINTS_OPT]; 
        Ql_k_old = best_theta[NUM_CONTROL_POINTS_OPT:]
        
        Qs_next_guess = Qs_k_old + shift_s; 
        Ql_next_guess = Ql_k_old 
        theta_next_guess = jnp.concatenate([Qs_next_guess, Ql_next_guess])
        mu_k = mu_k.at[:].set(theta_next_guess)
        
        if t % 1 == 0: 
            ax.cla(); ax.set_xlim(robot_s - 30, robot_s + 200); ax.set_ylim(-15, 15)
            for y in [-11.1, -7.4, -3.7, 0, 3.7, 7.4, 11.1]: ax.axhline(y, color='gray', ls='--', alpha=0.3)
            
            # 绘制所有障碍物
            for car_state in lead_car_states:
                s_center = car_state['s']
                y_center = car_state['l'] * LANE_WIDTH
                ax.add_patch(Rectangle((s_center-2.4, y_center-0.95), 4.8, 1.9, 
                                       color='red', alpha=0.9, label='Lead Cars' if car_state == lead_car_states[0] else None))
            
            # 绘制自车 (Ego) 的体积
            ego_x_center = robot_s
            ego_y_center = robot_l * LANE_WIDTH
            ego_rect_x = ego_x_center - 2.4
            ego_rect_y = ego_y_center - 0.95
            ax.add_patch(Rectangle((ego_rect_x, ego_rect_y), 4.8, 1.9, 
                                   color='green', alpha=0.9, label='Ego Car Body'))
            
            hist = jnp.array(history)
            ax.plot(hist[:,0], hist[:,1], 'b-', lw=4, label='Ego History')
            ax.plot(s_traj, l_traj*LANE_WIDTH, 'cyan', lw=5, alpha=0.7, label='Ego Planned Centerline')
            ax.plot(s_traj[0], l_traj[0]*LANE_WIDTH, 'go', ms=15)
            
            # --- 绘制 Q, P 点 ---
            s_anchor_0 = s_traj[0]; l_anchor_0 = l_traj[0] 
            s_anchor_1 = s_dot[0] * DT + s_traj[0]; l_anchor_1 = l_traj[0] 

            Qs_k_full = jnp.concatenate([jnp.array([s_anchor_0, s_anchor_1]), Qs_k_old])
            Ql_k_full = jnp.concatenate([jnp.array([l_anchor_0, l_anchor_1]), Ql_k_old])

            ax.plot(Qs_k_full, Ql_k_full*LANE_WIDTH, 'rx', ms=8, label='Optimization Variables (Q)')
            Ps_k = F_MATRIX @ Qs_k_full
            Pl_k = F_MATRIX @ Ql_k_full
            ax.plot(Ps_k, Pl_k*LANE_WIDTH, 'ko', ms=6, label='B-spline Control Points (P)')
            
            ax.set_title(f"2025 Production Overtake (EPF RSS) | S:{robot_s:.1f}m / {X_TARGET:.1f}m | V_cur:{robot_ds:4.1f}m/s")
            ax.legend()
            plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_overtake_demo()