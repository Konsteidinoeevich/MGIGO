# MPCmain3_FINAL_FULL_JERK.py
# 2025 量产级 MPC —— 引入曲率 kappa(s) 和完整的笛卡尔 Jerk 动力学转换

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
NUM_CONTROL_POINTS = 10 
TOTAL_DIM = 2 * NUM_CONTROL_POINTS

# --- 关键常数 ---
LANE_WIDTH = 3.7   


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


# ====================== 2. B-spline 基函数和导数矩阵 (不变) ======================
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
    
    Qs = theta[:10]; Ql = theta[10:]
    
    Qs = Qs.at[0].set(s_cur); Ql = Ql.at[0].set(l_cur)
    Qs = Qs.at[1].set(s_cur + ds_cur * DT); Ql = Ql.at[1].set(l_cur)

    Ps = F_MATRIX @ Qs; Pl = F_MATRIX @ Ql
    
    s_traj   = N5_BASIS      @ Ps; l_traj   = N5_BASIS      @ Pl 
    s_dot    = N5_PRIME      @ Ps; l_dot    = N5_PRIME      @ Pl
    s_ddot   = N5_DOUBLE_PRIME @ Ps; l_ddot   = N5_DOUBLE_PRIME @ Pl
    s_dddot  = N5_TRIPLE_PRIME @ Ps; l_dddot  = N5_TRIPLE_PRIME @ Pl

    traj = jnp.stack([s_traj, l_traj], axis=1)

    return traj, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot

# ====================== 4. 代价函数 (引入完整 Jerk 转换) ======================
@jit
def overtake_cost(theta, ctx):
    _, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot = \
        theta_to_trajectory(theta, ctx)
        
    lead_box = ctx['lead_box']
    Y_target = ctx['Y_target'] 
    X_target = ctx['X_target'] 
    
    # 获取曲率信息 (H 维向量)
    kappa = ctx['kappa']
    kappa_prime = ctx['kappa_prime']
    kappa_double_prime = ctx['kappa_double_prime'] # 新增
    theta_r = ctx['theta_r']

    # --- 1. Frenet 到 笛卡尔的完整非线性加速度转换 ---
    c = 1.0 - l_traj * kappa
    
    # Frenet 加速度分量
    A_parallel = s_ddot * c - s_dot**2 * l_traj * kappa_prime - 2.0 * s_dot * l_dot * kappa
    A_perp = l_ddot + s_dot**2 * kappa * c
    
    # 笛卡尔加速度
    cos_theta_r = jnp.cos(theta_r)
    sin_theta_r = jnp.sin(theta_r)
    ddot_x = A_parallel * cos_theta_r - A_perp * sin_theta_r
    ddot_y = A_parallel * sin_theta_r + A_perp * cos_theta_r
    a_mag = jnp.sqrt(ddot_x**2 + ddot_y**2)             

    # --- 2. ！！！Frenet 到 笛卡尔的完整非线性 Jerk 转换！！！ ---
    
    # 中间项: 路径角速度
    dot_theta_r = s_dot * kappa
    
    # 计算加速度分量的时间导数 (A_dot)
    
    # 2.1 A_dot_parallel (纵向 Jerk 分量)
    # 项 1: ddd_s * c
    term1 = s_dddot * c 
    # 项 2: -3 * ddot_s * dot_l * kappa
    term2 = -3.0 * s_ddot * l_dot * kappa
    # 项 3: -2 * dot_s * ddot_l * kappa
    term3 = -2.0 * s_dot * l_ddot * kappa
    # 项 4: -3 * dot_s * ddot_s * l * kappa_prime
    term4 = -3.0 * s_dot * s_ddot * l_traj * kappa_prime
    # 项 5: -3 * dot_s^2 * dot_l * kappa_prime
    term5 = -3.0 * s_dot**2 * l_dot * kappa_prime
    # 项 6: - dot_s^3 * l * kappa_double_prime
    term6 = -s_dot**3 * l_traj * kappa_double_prime

    A_dot_parallel = term1 + term2 + term3 + term4 + term5 + term6
    
    # 2.2 A_dot_perp (横向 Jerk 分量)
    # 项 1: ddd_l
    term_A = l_dddot
    # 项 2: 2 * dot_s * ddot_s * kappa * c
    term_B = 2.0 * s_dot * s_ddot * kappa * c
    # 项 3: dot_s^3 * kappa_prime * c
    term_C = s_dot**3 * kappa_prime * c
    # 项 4: - dot_s^2 * dot_l * kappa^2 
    term_D = -s_dot**2 * l_dot * kappa**2
    # 项 5: - dot_s^3 * l * kappa * kappa_prime 
    term_E = -s_dot**3 * l_traj * kappa * kappa_prime
    
    A_dot_perp = term_A + term_B + term_C + term_D + term_E

    # 2.3 笛卡尔 Jerk 最终旋转
    # Jerk Rotation terms (包括角速度 dot_theta_r)
    J_rot_x = -(A_parallel * sin_theta_r + A_perp * cos_theta_r) * dot_theta_r
    J_rot_y = (A_parallel * cos_theta_r - A_perp * sin_theta_r) * dot_theta_r

    # 最终笛卡尔 Jerk
    dddot_x = A_dot_parallel * cos_theta_r - A_dot_perp * sin_theta_r + J_rot_x
    dddot_y = A_dot_parallel * sin_theta_r + A_dot_perp * cos_theta_r + J_rot_y
    
    # --- 代价项计算 (全部基于笛卡尔 X-Y) ---
    
    x_traj = s_traj
    y_traj = l_traj 

    # 终端代价 (基于笛卡尔 X-Y)
    cost_term = 40.0 * (y_traj[-1] - Y_target)**2 + 5.0 * (x_traj[-1] - X_target)**2 

    # 1. 总加速度限制 (基于笛卡尔 a_mag)
    cost_acc = 10.0 * jnp.sum(jnp.maximum(0.0, a_mag - 7.0)**2)

    # 2. 横向加速度惩罚 (基于笛卡尔 ddot_y)
    cost_lat = 50.0 * jnp.sum(ddot_y**2) 

    # 3. Jerk 惩罚 (基于笛卡尔 dddot_x, dddot_y)
    cost_jerk = 10.0 * jnp.sum(dddot_x**2 + dddot_y**2) 

    # 碰撞 (l_traj/y_traj 已经是物理距离)
    ego_s_min = s_traj - 2.4; ego_s_max = s_traj + 2.4
    ego_y_min = l_traj - 0.95; ego_y_max = l_traj + 0.95 
    lead_s_min, lead_y_min = lead_box[0]; lead_s_max, lead_y_max = lead_box[1]
    overlap_s = jnp.maximum(0.0, jnp.minimum(lead_s_max, ego_s_max) - jnp.maximum(lead_s_min, ego_s_min))
    overlap_y = jnp.maximum(0.0, jnp.minimum(lead_y_max, ego_y_max) - jnp.maximum(lead_y_min, ego_y_min))
    cost_collision = 1e3 * jnp.sum(overlap_s * overlap_y)

    return cost_term + cost_acc + cost_lat + cost_jerk + cost_collision 


# --- ！！！参考路径属性生成函数 (引入 kappa_double_prime) ！！！ ---
def get_reference_path_properties(s_traj):
    """
    根据轨迹 s_traj 上的点，返回参考路径的 kappa, kappa', kappa'' 和 theta_r。
    
    此处为模拟：保持 kappa=0, kappa'=0, kappa''=0, theta_r=0，即为直道。
    实际应用中，这里需要连接 HDMap 或 Path Planner。
    """
    KAPPA = jnp.zeros_like(s_traj) 
    KAPPA_PRIME = jnp.zeros_like(s_traj) 
    KAPPA_DOUBLE_PRIME = jnp.zeros_like(s_traj) # 新增
    THETA_R = jnp.zeros_like(s_traj) 
    
    # 示例：如果需要测试曲率，可以取消注释并定义曲线
    #R = 100.0 # 半径
    #KAPPA_H = jnp.where(s_traj > 50.0, 1.0 / R, 0.0)
    #THETA_R_H = jnp.where(s_traj > 50.0, (s_traj - 50.0) / R, 0.0)
    
    return KAPPA, KAPPA_PRIME, KAPPA_DOUBLE_PRIME, THETA_R


# ====================== 5. 主循环 (传递所有曲率上下文) ======================
def run_overtake_demo():
    print("2025 真实量产级 MPC —— 曲率就绪 + 完整笛卡尔 Jerk 动力学 [最终版] 启动！")
    key = jax.random.PRNGKey(0)

    robot_s = 0.0; robot_l = 0.0; robot_ds = 12.0 
    
    Qs0 = onp.arange(NUM_CONTROL_POINTS) * KNOT_DELTA * robot_ds
    Qs0[0] = 0.0; Qs0[1] = 0.0 + robot_ds * DT 
    Ql0 = onp.zeros(NUM_CONTROL_POINTS)
    theta0 = jnp.concatenate([jnp.array(Qs0), jnp.array(Ql0)])
    
    mu_k = jnp.stack([theta0] * 6); L_inv_k = jnp.stack([jnp.eye(TOTAL_DIM) * 2.0] * 6); pi_k = jnp.ones(6) / 6.0

    lead_s = 48.0; lead_l = 0.0 # lead_l 是 Frenet 索引 (0.0=中心)
    
    X_TARGET = 150.0

    plt.ion(); fig, ax = plt.subplots(figsize=(15, 7)); history = []

    for t in range(100):
        key, subkey = jax.random.split(key)
        
        dist = lead_s - robot_s; l_target_idx = 1.0 if dist < 50 else 0.0
        Y_TARGET = l_target_idx * LANE_WIDTH 
        
        lead_y_center = lead_l * LANE_WIDTH
        lead_box = jnp.array([ [lead_s - 2.4, lead_y_center - 0.95], 
                              [lead_s + 2.4, lead_y_center + 0.95] ])
        
        # 1. 生成参考路径属性向量 (基于初始的 s 预测)
        initial_s_guess = Qs0
        KAPPA_H, KAPPA_PRIME_H, KAPPA_DOUBLE_PRIME_H, THETA_R_H = get_reference_path_properties(initial_s_guess[:HORIZON])
        
        # 传递所有需要的上下文 (包括 kappa'')
        ctx = {'s_cur': robot_s, 'l_cur': robot_l, 'ds_cur': robot_ds, 
               'Y_target': Y_TARGET, 'lead_box': lead_box, 'X_target': X_TARGET,
               'kappa': KAPPA_H, 'kappa_prime': KAPPA_PRIME_H, 
               'kappa_double_prime': KAPPA_DOUBLE_PRIME_H, 'theta_r': THETA_R_H}

        steps = 3000 if t == 0 else 450
        t0 = time.time()
        
        mu_k, L_inv_k, pi_k = igo_mog_optimizer(subkey, steps, 0.12, 6, 60, 25, overtake_cost, mu_k, L_inv_k, pi_k, ctx)
        t1 = time.time()

        best_theta = mu_k[jnp.argmax(pi_k)]
        
        best_traj, s_traj, l_traj, s_dot, l_dot, _, _, _, _ = theta_to_trajectory(best_theta, ctx)
        
        # 运动更新
        robot_s_next = s_traj[1]; robot_l_next = l_traj[1]; robot_ds_next = s_dot[1] 
        shift_s = robot_s_next - robot_s 
        
        robot_s = robot_s_next; robot_l = robot_l_next; robot_ds = robot_ds_next
        history.append(jnp.array([robot_s, robot_l])) 

        print(f"Step {t:02d} | S:{robot_s:6.1f}m Y:{robot_l:5.1f}m V:{robot_ds:4.1f}m/s Dist:{dist:4.1f}m OPT:{(t1-t0)*1000:4.0f}ms")

        # 控制点重调度 (Re-scheduling)
        Qs_k_old = best_theta[:10]; Ql_k_old = best_theta[10:]
        Qs_next_guess = Qs_k_old + shift_s; Ql_next_guess = Ql_k_old 
        theta_next_guess = jnp.concatenate([Qs_next_guess, Ql_next_guess])
        mu_k = mu_k.at[:].set(theta_next_guess)
        
        if t % 1 == 0:
            ax.cla()
            ax.set_xlim(robot_s - 30, robot_s + 120); ax.set_ylim(-15, 15)
            
            for lane_y in [-3.7*3, -3.7*2, -3.7, 0, 3.7, 3.7*2, 3.7*3]: 
                ax.axhline(lane_y, color='gray', ls='--', alpha=0.3)
            
            ax.add_patch(Rectangle((lead_s-2.4, lead_y_center-0.95), 4.8, 1.9, color='red', alpha=0.9, label='Lead Car'))
            
            hist = jnp.array(history)
            ax.plot(hist[:,0], hist[:,1], 'b-', lw=4, label='Ego')
            ax.plot(best_traj[:,0], best_traj[:,1], 'cyan', lw=5, alpha=0.7)
            ax.plot(s_traj[0], l_traj[0], 'go', ms=15)
            
            Qs_k = best_theta[:10].at[0].set(s_traj[0]).at[1].set(s_dot[0] * DT + s_traj[0])
            Ql_k = best_theta[10:].at[0].set(l_traj[0]).at[1].set(l_traj[0])

            ax.plot(Qs_k, Ql_k, 'rx', ms=8, label='Optimization Variables (Q)')
            Ps_k = F_MATRIX @ Qs_k
            Pl_k = F_MATRIX @ Ql_k
            ax.plot(Ps_k, Pl_k, 'ko', ms=6, label='B-spline Control Points (P)')
            
            ax.set_title(f"2025 Production Overtake | X_target={X_TARGET}m | Y_target={Y_TARGET:.1f}m | OPT={(t1-t0)*1000:.1f}ms (Final)")
            ax.legend()
            plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_overtake_demo()