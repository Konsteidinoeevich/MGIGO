# MPCmain4_Final_Multi_Obstacle.py
# 2025 真实量产级 MPC —— 支持多障碍物 (左、中、右车道) 和长距离目标 (1000m)

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
# 假设 gmm_igo.MPCsolver.igo_mog_optimizer 是可用的
from gmm_igo.MPCsolver import igo_mog_optimizer 
from scipy.interpolate import BSpline
import numpy as onp

# ====================== 1. 全局配置 (更新目标距离) ======================
DT = 0.1
HORIZON = 100
TOTAL_TIME = HORIZON * DT  # 10.0 seconds
POLY_ORDER = 5       

# --- 关键维度 ---
NUM_CONTROL_POINTS_FULL = 15 
NUM_CONTROL_POINTS_OPT = 13  
TOTAL_DIM = 2 * NUM_CONTROL_POINTS_OPT 

# --- 关键常数 ---
LANE_WIDTH = 3.7   
V_MAX = 40.0 # 最大速度限制 (m/s)
V_MIN = 10.0 # 最小速度限制 (m/s)
V_TARGET_TERMINAL = 25.0 # 最终目标速度 (m/s)
X_TARGET = 1000.0 # ！！！ 新增：目标纵向距离 1000.0 m ！！！

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


# ====================== 5. 代价函数 (多障碍物碰撞检测) ======================
@jit
def overtake_cost(theta, ctx):
    _, s_traj, l_traj, s_dot, _, s_ddot, l_ddot, s_dddot, l_dddot = \
        theta_to_trajectory(theta, ctx)
    
    # ！！！ 从 ctx 中接收多障碍物边界 ！！！
    lead_boxes = ctx['lead_boxes'] # 形状 (N_cars, 2, 2)
    Y_target = ctx['Y_target'] 
    X_target = ctx['X_target'] 
    
    # --- 笛卡尔动力学转换 ---
    x_traj = s_traj
    y_traj = l_traj * LANE_WIDTH
    ddot_x = s_ddot; ddot_y = l_ddot * LANE_WIDTH                        
    a_mag = jnp.sqrt(ddot_x**2 + ddot_y**2)             
    dddot_x = s_dddot; dddot_y = l_dddot * LANE_WIDTH  
    
    # --- 代价项计算 ---
    
    # Stage Cost (横向拉力和纵向追踪)
    cost_stage_deviation = jnp.sum(10.0 * (y_traj - Y_target)**2 + 5.0 * (x_traj - X_target)**2)

    # 终端代价 (位置 + 速度)
    cost_term = (
        10.0 * (y_traj[-1] - Y_target)**2 + 
        50.0 * (x_traj[-1] - X_target)**2 +
        50.0 * (s_dot[-1] - V_TARGET_TERMINAL)**2 
    ) 

    # 1. 总加速度限制
    cost_acc = 10.0 * jnp.sum(jnp.maximum(0.0, a_mag - 5.0)**2)

    # 2. 横向加速度惩罚
    cost_lat = 0.1 * jnp.sum(ddot_y**2) 

    # 3. Jerk 惩罚
    cost_jerk = 10.0 * jnp.sum(dddot_x**2 + dddot_y**2) 

    # ！！！ 碰撞代价 (多障碍物向量化处理) ！！！
    H = HORIZON
    N = lead_boxes.shape[0]

    # Ego box properties (形状 (H,))
    ego_s_min = s_traj - 2.4; ego_s_max = s_traj + 2.4
    ego_y_min = l_traj*LANE_WIDTH - 0.95; ego_y_max = l_traj*LANE_WIDTH + 0.95 

    # 扩展 Ego 属性到 (N_cars, H) 形状，用于与障碍物批处理
    ego_s_min_batch = jnp.tile(ego_s_min, (N, 1))
    ego_s_max_batch = jnp.tile(ego_s_max, (N, 1))
    ego_y_min_batch = jnp.tile(ego_y_min, (N, 1))
    ego_y_max_batch = jnp.tile(ego_y_max, (N, 1))

    # 提取障碍物属性。形状 (N_cars, 1)
    lead_s_min = lead_boxes[:, 0, 0].reshape(N, 1) 
    lead_y_min = lead_boxes[:, 0, 1].reshape(N, 1)
    lead_s_max = lead_boxes[:, 1, 0].reshape(N, 1)
    lead_y_max = lead_boxes[:, 1, 1].reshape(N, 1)
    
    # 计算碰撞重叠面积 (形状 N_cars, H)
    overlap_s = jnp.maximum(0.0, jnp.minimum(lead_s_max, ego_s_max_batch) - jnp.maximum(lead_s_min, ego_s_min_batch))
    overlap_y = jnp.maximum(0.0, jnp.minimum(lead_y_max, ego_y_max_batch) - jnp.maximum(lead_y_min, ego_y_min_batch))
    
    # 对所有障碍物和所有时间步的重叠面积求和
    cost_collision = 1e3 * jnp.sum(overlap_s * overlap_y)

    return cost_term + cost_stage_deviation + cost_acc + cost_lat + cost_jerk + cost_collision 


# ====================== 6. 主循环 (多障碍物状态管理与执行) ======================
def run_overtake_demo():
    print(f" MPC —— 多障碍物/长距离规划 ({X_TARGET}m) 启动！")
    key = jax.random.PRNGKey(0)

    # 初始状态
    robot_s = 0.0; robot_l = 0.0; robot_ds = 30.0 
    
    # ！！！ 障碍物初始化 ！！！ (l: Frenet坐标，单位是车道数)
    # 假设 L_WIDTH = 3.7m: l=0 -> 0m (中), l=1 -> 3.7m (左), l=-1 -> -3.7m (右)
    lead_car_states = [
        {'s': 100.0, 'l': 0.0, 'v': 20.0},  # 障碍车 1: 中车道, 20 m/s
        {'s': 150.0, 'l': 0.0, 'v': 25.0}, # 障碍车 2: 右车道, 25 m/s
        {'s': 200.0, 'l': 1.0, 'v': 30.0}   # 障碍车 3: 左车道, 30 m/s (与 Ego 初速相同)
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

    for t in range(200): # 运行更长时间以覆盖 1000m 目标
        if robot_s >= X_TARGET:
            print(f"达到目标距离 {X_TARGET:.1f}m，停止规划.")
            break

        key, subkey = jax.random.split(key)
        
        l_target=1.0
        Y_TARGET = l_target * LANE_WIDTH 
        
        # ！！！ 障碍物状态更新和 Context 构造 ！！！
        current_lead_boxes = []
        for i in range(len(lead_car_states)):
            car = lead_car_states[i]
            # 更新 s 位置
            car['s'] += car['v'] * DT 
            
            # 构造边界盒 [s_min, y_min] 和 [s_max, y_max]
            s_center = car['s']
            y_center = car['l'] * LANE_WIDTH
            
            lead_box = [
                [s_center - 2.4, y_center - 0.95], 
                [s_center + 2.4, y_center + 0.95]
            ]
            current_lead_boxes.append(lead_box)
        
        JAX_lead_boxes = jnp.array(current_lead_boxes) # 形状 (N_cars, 2, 2)
        
        ctx = {'s_cur': robot_s, 'l_cur': robot_l, 'ds_cur': robot_ds, 
               'Y_target': Y_TARGET, 'lead_boxes': JAX_lead_boxes, 'X_target': X_TARGET}

        steps = 3000 if t == 0 else 450
        t0 = time.time()
        
        mu_k, L_inv_k, pi_k = igo_mog_optimizer(subkey, steps, 0.12, 6, 60, 25, overtake_cost, mu_k, L_inv_k, pi_k, ctx)
        t1 = time.time()

        best_theta = mu_k[jnp.argmax(pi_k)]
        
        best_traj, s_traj, l_traj, s_dot, l_dot, _, _, _, _ = theta_to_trajectory(best_theta, ctx)
        
        # --- 车辆执行：保持一致性更新 ---
        robot_ds_next_raw = s_dot[1] 
        robot_ds_next = jnp.clip(robot_ds_next_raw, V_MIN, V_MAX) 

        # 纵向位置：基于当前位置和饱和后的速度更新 
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
        
        if t % 1 == 0: # 降低绘图频率
            ax.cla(); ax.set_xlim(robot_s - 30, robot_s + 200); ax.set_ylim(-15, 15)
            for y in [-11.1, -7.4, -3.7, 0, 3.7, 7.4, 11.1]: ax.axhline(y, color='gray', ls='--', alpha=0.3)
            
            # ！！！ 绘制所有障碍物 ！！！
            for car_state in lead_car_states:
                s_center = car_state['s']
                y_center = car_state['l'] * LANE_WIDTH
                ax.add_patch(Rectangle((s_center-2.4, y_center-0.95), 4.8, 1.9, 
                                       color='red', alpha=0.9, label='Lead Cars' if car_state == lead_car_states[0] else None))
            
            hist = jnp.array(history)
            ax.plot(hist[:,0], hist[:,1], 'b-', lw=4, label='Ego History')
            ax.plot(s_traj, l_traj*LANE_WIDTH, 'cyan', lw=5, alpha=0.7, label='Ego Planned')
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
            
            ax.set_title(f"2025 Production Overtake (Multi-Obstacle) | S:{robot_s:.1f}m / {X_TARGET:.1f}m | V_cur:{robot_ds:4.1f}m/s")
            ax.legend()
            plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_overtake_demo()