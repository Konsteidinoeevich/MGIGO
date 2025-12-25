# trajectory_parameterization.py (FINAL FIX for JAX SHAPE ERROR)

import jax.numpy as jnp
from jax import jit
import math
from typing import Dict, Any

# 导入所有配置和矩阵。
try:
    from problem.spline_filter import (
        DT, NUM_CONTROL_POINTS, F_MATRIX, N5_BASIS, N5_PRIME, N5_DOUBLE_PRIME, 
        N5_TRIPLE_PRIME, N5_QUAD_PRIME,
        NUM_CONTROL_POINTS_OPT, 
        LANE_WIDTH
    )
except ImportError:
    # 应急常量
    DT = 0.1
    NUM_CONTROL_POINTS_OPT = 13
    LANE_WIDTH = 3.7
    # 假设 F_MATRIX 等变量已被定义或导入

# 关键 FIX: 将 DT 和 LANE_WIDTH 转换为 Python 浮点数常量，避免 JAX 追踪错误地将其广播
# 这强制 JAX 在编译时将它们视为静态标量。
DT_FLOAT = float(DT)
LANE_WIDTH_FLOAT = float(LANE_WIDTH)

# ====================== 1. 轨迹参数化 (Frenet输出 S, L) ======================
@jit
def theta_to_trajectory(theta: jnp.ndarray, ctx: Dict[str, Any]):
    """
    将 26 维的优化变量 theta (Qs_opt, Ql_opt) 重构为 30 维的 Q_full，
    并转换为 Frenet 轨迹及其导数。
    """
    # 初始状态 (在最内层 vmap 中它们应该是 JAX 标量)
    s_cur   = ctx['s_cur']
    l_cur   = ctx['l_cur']
    ds_cur  = ctx['ds_cur'] # 初始速度 ds/dt
    
    # 1. 拆分优化变量 (长度 13)
    Qs_opt = theta[:NUM_CONTROL_POINTS_OPT]; 
    Ql_opt = theta[NUM_CONTROL_POINTS_OPT:]
    
    # 2. 计算 Q[0] 和 Q[1] 的锚定值 (长度 2) - C0 和 C1 连续性
    # 2. 计算 Q[0] 和 Q[1] 的锚定值 (长度 2) - C0 和 C1 连续性
    s_anchor_0 = s_cur
    l_anchor_0 = l_cur
    s_anchor_1 = s_cur + ds_cur * DT 
    l_anchor_1 = l_cur 
    
    # FIX: 使用 jnp.stack 明确创建 (2,) 形状的数组，避免 JAX 广播错误
    Qs_anchors = jnp.stack([s_anchor_0, s_anchor_1]) # 形状 (2,)
    Ql_anchors = jnp.stack([l_anchor_0, l_anchor_1]) # 形状 (2,)
    # 3. 重构完整的 Q 向量 (长度 15)
    Qs_full = jnp.concatenate([Qs_anchors, Qs_opt]) # 形状 (15,)
    Ql_full = jnp.concatenate([Ql_anchors, Ql_opt]) # 形状 (15,)

    # 4. 滤波 P = F @ Q_full
    Ps = F_MATRIX @ Qs_full; Pl = F_MATRIX @ Ql_full
    
    # 5. B-spline 评估: 0~3 阶导数
    s_traj   = N5_BASIS      @ Ps; l_traj   = N5_BASIS      @ Pl
    s_dot    = N5_PRIME      @ Ps; l_dot    = N5_PRIME      @ Pl
    s_ddot   = N5_DOUBLE_PRIME @ Ps; l_ddot   = N5_DOUBLE_PRIME @ Pl
    s_dddot  = N5_TRIPLE_PRIME @ Ps; l_dddot  = N5_TRIPLE_PRIME @ Pl

    return s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot

# ====================== 2. 动力学方程 (Frenet 到 笛卡尔的微分平坦近似) ======================
@jit
def compute_cartesian_kinematics(s_traj, l_traj, s_ddot, l_ddot, s_dddot, l_dddot):
    """
    假设在直道 (kappa=0)，将 Frenet 坐标系下的加速度和 Jerk 
    近似转换为笛卡尔坐标系下的控制/约束量。
    """
    
    # (1) 笛卡尔位置 (直道近似: X=S, Y=L*W)
    x_traj = s_traj
    y_traj = l_traj * LANE_WIDTH_FLOAT # FIX: 使用 LANE_WIDTH_FLOAT 确保是静态标量
    
    # (2) 笛卡尔加速度 (X/Y)
    ddot_x = s_ddot                                     
    ddot_y = l_ddot * LANE_WIDTH_FLOAT                        
    a_mag = jnp.sqrt(ddot_x**2 + ddot_y**2)             
    
    # (3) 笛卡尔 Jerk (X/Y)
    dddot_x = s_dddot                                   
    dddot_y = l_dddot * LANE_WIDTH_FLOAT                      
    
    return x_traj, y_traj, ddot_x, ddot_y, a_mag, dddot_x, dddot_y