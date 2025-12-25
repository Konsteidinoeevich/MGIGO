# MPCmain3_FINAL_CARTESIAN_TERMINAL_COST.py
# 2025 量产级 MPC —— 移除速度维持，终端代价全转为笛卡尔坐标系

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

# ====================== 全局配置 ======================
DT = 0.1
HORIZON = 100
TOTAL_TIME = HORIZON * DT  # 9.9 seconds
POLY_ORDER = 5       
NUM_CONTROL_POINTS = 15 
TOTAL_DIM = 2 * NUM_CONTROL_POINTS

# --- 关键常数 ---
LANE_WIDTH = 3.7   
CURVATURE_KAPPA = 0.0 # 假设直道


# ====================== 1. 构造 5-Tap 滤波器矩阵 F (不变) ======================
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

F_MATRIX = create_5_tap_filter_matrix(NUM_CONTROL_POINTS)


# ====================== 2. B-spline 基函数和导数矩阵 (节点匹配预测时间) ======================
INTERNAL_KNOTS_COUNT = NUM_CONTROL_POINTS - POLY_ORDER 
KNOT_DELTA = TOTAL_TIME / INTERNAL_KNOTS_COUNT 
internal_knots = onp.arange(1, INTERNAL_KNOTS_COUNT + 1) * KNOT_DELTA

knots = onp.concatenate([onp.zeros(POLY_ORDER + 1), 
                         internal_knots, 
                         onp.full(POLY_ORDER, TOTAL_TIME)]) 

t_eval = onp.arange(HORIZON) * DT

def compute_basis_matrix(knots, t_eval, k, nu):
    basis = onp.zeros((len(t_eval), NUM_CONTROL_POINTS))
    for i in range(NUM_CONTROL_POINTS):
        c = onp.zeros(NUM_CONTROL_POINTS); c[i] = 1.0
        spl = BSpline(knots, c, k=k, extrapolate=True) 
        basis[:, i] = spl(t_eval, nu=nu)
    return jnp.array(basis, dtype=jnp.float32)

N5_BASIS          = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=0) 
N5_PRIME          = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=1) 
N5_DOUBLE_PRIME   = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=2) 
N5_TRIPLE_PRIME   = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=3) 


# ====================== 3. theta_to_trajectory (不变) ======================
@jit
def theta_to_trajectory(theta, ctx):
    s_cur   = ctx['s_cur']; l_cur   = ctx['l_cur']; ds_cur  = ctx['ds_cur']
    
    Qs = theta[:15]; Ql = theta[15:]
    
    Qs = Qs.at[0].set(s_cur); Ql = Ql.at[0].set(l_cur)
    Qs = Qs.at[1].set(s_cur + ds_cur * DT); Ql = Ql.at[1].set(l_cur)

    Ps = F_MATRIX @ Qs; Pl = F_MATRIX @ Ql
    
    s_traj   = N5_BASIS      @ Ps; l_traj   = N5_BASIS      @ Pl
    s_dot    = N5_PRIME      @ Ps; l_dot    = N5_PRIME      @ Pl
    s_ddot   = N5_DOUBLE_PRIME @ Ps; l_ddot   = N5_DOUBLE_PRIME @ Pl
    s_dddot  = N5_TRIPLE_PRIME @ Ps; l_dddot  = N5_TRIPLE_PRIME @ Pl

    traj = jnp.stack([s_traj, l_traj * LANE_WIDTH], axis=1)

    return traj, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot

# ====================== 4. 代价函数 (移除速度维持 & 终端代价笛卡尔化) ======================
@jit
def overtake_cost(theta, ctx):
    _, s_traj, l_traj, s_dot, _, s_ddot, l_ddot, s_dddot, l_dddot = \
        theta_to_trajectory(theta, ctx)
    lead_box = ctx['lead_box']
    Y_target = ctx['Y_target'] # 笛卡尔目标 Y
    X_target = ctx['X_target'] # 笛卡尔目标 X
    V_ref= 30.0
    # --- Frenet 到 笛卡尔动力学转换 (基于直道 kappa=0) ---
    x_traj = s_traj
    y_traj = l_traj * LANE_WIDTH
    
    ddot_x = s_ddot                                     
    ddot_y = l_ddot * LANE_WIDTH                        
    a_mag = jnp.sqrt(ddot_x**2 + ddot_y**2)             # 笛卡尔坐标系下总加速度大小
    dddot_x = s_dddot                                   
    dddot_y = l_dddot * LANE_WIDTH                      # 笛卡尔坐标系下 Jerk 横向分量
    
    # --- 代价项计算 ---
    
    # ！！！新终端代价 (完全基于笛卡尔 X-Y) ！！！
    cost_term = 10.0 * (y_traj[-1] - Y_target)**2 + 50.0 * (x_traj[-1] - X_target)**2 

   
    # 1. 总加速度限制 (基于笛卡尔 a_mag)
    cost_acc = 10.0 * jnp.sum(jnp.maximum(0.0, a_mag - 5.0)**2)

    # 2. 横向加速度惩罚 (基于笛卡尔 ddot_y)
    cost_lat = 1.0 * jnp.sum(ddot_y**2) 

    # 3. Jerk 惩罚 (基于笛卡尔 dddot_x, dddot_y)
    cost_jerk = 10.0 * jnp.sum(dddot_x**2 + dddot_y**2) 

    # 碰撞 (基于 s/l*W, 接近笛卡尔)
    ego_s_min = s_traj - 2.4; ego_s_max = s_traj + 2.4
    ego_y_min = l_traj*LANE_WIDTH - 0.95; ego_y_max = l_traj*LANE_WIDTH + 0.95 
    lead_s_min, lead_y_min = lead_box[0]; lead_s_max, lead_y_max = lead_box[1]
    overlap_s = jnp.maximum(0.0, jnp.minimum(lead_s_max, ego_s_max) - jnp.maximum(lead_s_min, ego_s_min))
    overlap_y = jnp.maximum(0.0, jnp.minimum(lead_y_max, ego_y_max) - jnp.maximum(lead_y_min, ego_y_min))
    cost_collision = 1e3 * jnp.sum(overlap_s * overlap_y)

    return cost_term + cost_acc + cost_lat + cost_jerk + cost_collision 

# ====================== 5. 主循环 (传递笛卡尔目标) ======================
def run_overtake_demo():
    print("2025 真实量产级 MPC —— 动态/终端代价全笛卡尔化 [最终版] 启动！")
    key = jax.random.PRNGKey(0)

    # 恢复高速场景设置
    robot_s = 0.0; robot_l = 0.0; robot_ds = 30.0 
    
    Qs0 = onp.arange(NUM_CONTROL_POINTS) * KNOT_DELTA * 30.0 / KNOT_DELTA
    Qs0[0] = 0.0; Qs0[1] = 0.0 + robot_ds * DT 
    Ql0 = onp.zeros(NUM_CONTROL_POINTS)
    theta0 = jnp.concatenate([jnp.array(Qs0), jnp.array(Ql0)])
    
    mu_k = jnp.stack([theta0] * 6); L_inv_k = jnp.stack([jnp.eye(TOTAL_DIM) * 2.0] * 6); pi_k = jnp.ones(6) / 6.0

    lead_s = 100.0; lead_l = 0.0
    
    # 定义终端目标 X
    X_TARGET = 300.0

    plt.ion(); fig, ax = plt.subplots(figsize=(15, 7)); history = []

    for t in range(100):
        key, subkey = jax.random.split(key)
        
        # 动态计算 Frenet 目标 l 和笛卡尔目标 Y
        l_target=0.0
        Y_TARGET = l_target * LANE_WIDTH 
        
        lead_box = jnp.array([ [lead_s - 2.4, lead_l*LANE_WIDTH - 0.95], 
                              [lead_s + 2.4, lead_l*LANE_WIDTH + 0.95] ])
        
        # 传递笛卡尔目标 Y_target 和 X_target
        ctx = {'s_cur': robot_s, 'l_cur': robot_l, 'ds_cur': robot_ds, 
               'Y_target': Y_TARGET, 'lead_box': lead_box, 'X_target': X_TARGET}

        steps = 3000 if t == 0 else 450
        t0 = time.time()
        
        mu_k, L_inv_k, pi_k = igo_mog_optimizer(subkey, steps, 0.12, 6, 60, 25, overtake_cost, mu_k, L_inv_k, pi_k, ctx)
        t1 = time.time()

        best_theta = mu_k[jnp.argmax(pi_k)]
        
        best_traj, s_traj, l_traj, s_dot, l_dot, _, _, _, _ = theta_to_trajectory(best_theta, ctx)
        
        robot_s_next = s_traj[1]; robot_l_next = l_traj[1]; robot_ds_next = s_dot[1] 
        shift_s = robot_s_next - robot_s 
        
        robot_s = robot_s_next; robot_l = robot_l_next; robot_ds = robot_ds_next
        history.append(jnp.array([robot_s, robot_l * LANE_WIDTH]))

        print(f"Step {t:02d} | S:{robot_s:6.1f}m L:{robot_l*LANE_WIDTH:5.1f}m V:{robot_ds:4.1f}m/s  OPT:{(t1-t0)*1000:4.0f}ms")

        Qs_k_old = best_theta[:10]; Ql_k_old = best_theta[10:]
        Qs_next_guess = Qs_k_old + shift_s; Ql_next_guess = Ql_k_old 
        theta_next_guess = jnp.concatenate([Qs_next_guess, Ql_next_guess])
        mu_k = mu_k.at[:].set(theta_next_guess)
        
        if t % 1 == 0:
            ax.cla()
            ax.set_xlim(robot_s - 30, robot_s + 120); ax.set_ylim(-15, 15)
            for y in [-11.1, -7.4, -3.7, 0, 3.7, 7.4, 11.1]: ax.axhline(y, color='gray', ls='--', alpha=0.3)
            ax.add_patch(Rectangle((lead_s-2.4, lead_l*LANE_WIDTH-0.95), 4.8, 1.9, color='red', alpha=0.9, label='Lead Car'))
            
            hist = jnp.array(history)
            ax.plot(hist[:,0], hist[:,1], 'b-', lw=4, label='Ego')
            ax.plot(best_traj[:,0], best_traj[:,1], 'cyan', lw=5, alpha=0.7)
            ax.plot(s_traj[0], l_traj[0]*LANE_WIDTH, 'go', ms=15)
            
            Qs_k = best_theta[:15].at[0].set(s_traj[0]).at[1].set(s_dot[0] * DT + s_traj[0])
            Ql_k = best_theta[15:].at[0].set(l_traj[0]).at[1].set(l_traj[0])

            ax.plot(Qs_k, Ql_k*LANE_WIDTH, 'rx', ms=8, label='Optimization Variables (Q)')
            Ps_k = F_MATRIX @ Qs_k
            Pl_k = F_MATRIX @ Ql_k
            ax.plot(Ps_k, Pl_k*LANE_WIDTH, 'ko', ms=6, label='B-spline Control Points (P)')
            
            ax.set_title(f"2025 Production Overtake | X_target={X_TARGET}m | Y_target={Y_TARGET:.1f}m | OPT={(t1-t0)*1000:.1f}ms (Final)")
            ax.legend()
            plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_overtake_demo()