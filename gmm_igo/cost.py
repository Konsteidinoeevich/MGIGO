# cost_function.py

import jax.numpy as jnp
from jax import jit
# 导入 File 1 中的常量
from problem.spline_filter import LANE_WIDTH
# 导入 File 2 中的参数化函数
from problem.parameterization import theta_to_trajectory

# ====================== 1. 微分平坦动力学函数 ======================
@jit
def flat_output_to_controls(x_dot, y_dot, x_ddot, y_ddot, x_dddot, y_dddot, x_quad_dot, y_quad_dot):
    """
    通过平坦输出的导数，计算车辆的曲率 K、曲率变化率 K_dot 和总加速度 a_mag。
    """
    
    # 1. 运动学量
    dot_sq = x_dot**2 + y_dot**2
    dot_mag = jnp.sqrt(dot_sq)
    dot_mag_sq = dot_sq
    
    # 2. 曲率 kappa (K)
    # K = (x_dot * y_ddot - x_ddot * y_dot) / (|v|^3)
    K_num = x_dot * y_ddot - x_ddot * y_dot
    K_den = dot_mag_sq**(3/2)
    K = K_num / jnp.where(dot_mag > 0.1, K_den, 0.1**(3/2)) # 避免除零
    
    # 3. 曲率变化率 dot_kappa (K_dot) - Jerk 的主要代理
    # K_dot = dK/dt
    K_dot_num_term1 = (x_dot * y_dddot - x_dddot * y_dot) * dot_mag_sq 
    K_dot_num_term2 = 3.0 * K_num * (x_dot * x_ddot + y_dot * y_ddot)
    K_dot_num = K_dot_num_term1 - K_dot_num_term2
    K_dot_den = dot_mag_sq**(5/2)
    K_dot = K_dot_num / jnp.where(dot_mag > 0.1, K_dot_den, 0.1**(5/2)) # 避免除零
    
    # 4. 总加速度 magnitude (a_mag)
    a_mag = jnp.sqrt(x_ddot**2 + y_ddot**2)
    
    return K, K_dot, a_mag

# ====================== 2. 代价函数 (基于微分平坦输出) ======================
@jit
def overtake_cost(theta, ctx):
    """
    计算超车任务的综合代价。
    """
    _, x_traj, y_traj, x_dot, y_dot, x_ddot, y_ddot, x_dddot, y_dddot, x_quad_dot, y_quad_dot = \
        theta_to_trajectory(theta, ctx)
        
    lead_box = ctx['lead_box']
    Y_target = ctx['Y_target'] 
    X_target = ctx['X_target'] 
    
    # 1. 计算动力学量 (K, K_dot, a_mag)
    K, K_dot, a_mag = flat_output_to_controls(x_dot, y_dot, x_ddot, y_ddot, x_dddot, y_dddot, x_quad_dot, y_quad_dot)
    
    # 2. 代价项计算 (全部基于笛卡尔 X-Y)
    
    # 终端代价 (惩罚终点 X/Y 偏差)
    cost_term = 40.0 * (y_traj[-1] - Y_target)**2 + 5.0 * (x_traj[-1] - X_target)**2 

    # 1. 动力学约束：加速度限制
    cost_acc = 10.0 * jnp.sum(jnp.maximum(0.0, a_mag - 7.0)**2)
    
    # 2. 转向输入平滑性惩罚 (惩罚曲率变化率 K_dot，即转向角速度)
    cost_K_dot = 50.0 * jnp.sum(K_dot**2) 
    
    # 3. 转向幅度惩罚 (惩罚曲率 K，即转向角)
    cost_K = 10.0 * jnp.sum(K**2)

    # 4. 碰撞惩罚 (X/Y)
    ego_x_min = x_traj - 2.4; ego_x_max = x_traj + 2.4
    ego_y_min = y_traj - 0.95; ego_y_max = y_traj + 0.95 
    lead_x_min, lead_y_min = lead_box[0]; lead_x_max, lead_y_max = lead_box[1]
    overlap_x = jnp.maximum(0.0, jnp.minimum(lead_x_max, ego_x_max) - jnp.maximum(lead_x_min, ego_x_min))
    overlap_y = jnp.maximum(0.0, jnp.minimum(lead_y_max, ego_y_max) - jnp.maximum(lead_y_min, ego_y_min))
    cost_collision = 1e3 * jnp.sum(overlap_x * overlap_y)

    return cost_term + cost_acc + cost_K_dot + cost_K + cost_collision