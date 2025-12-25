# multiplecars.py - 多车十字路口 MPC 求解框架 (修正 JIT Tracer 错误)

import jax
import jax.numpy as jnp
from jax import vmap, lax, jit, random
import functools
from typing import Callable, Tuple, Any, Dict
from problem.spline_filter import DT, HORIZON, TOTAL_DIM, LANE_WIDTH 
from problem.parameterization import theta_to_trajectory
from gmm_igo.MPCsolverM2 import mmog_igo_optimizer_mpc 
    
from problem.spline_filter import NUM_CONTROL_POINTS_OPT 


# 笛卡尔状态维度: (x, y, v, psi)
CARTESIAN_STATE_DIM = 4 

# 常量 (碰撞和目标成本)
COLLISION_DISTANCE_SQ = 4.0**2 
Q_GOAL = 1.0     
R_COMFORT = 0.01 
C_COLLISION = 1000.0 

# ----------------------------------------------------------------------
# I. 轨迹生成辅助函数 (保持不变)
# ----------------------------------------------------------------------
# ... (frenet_approx_to_cartesian_trajectory, generate_trajectory, vmap_generate_trajectory_setup 保持不变) ...

@jit
def frenet_approx_to_cartesian_trajectory(frenet_trajectories: Tuple[jnp.ndarray, ...]):
    s_traj, l_traj, s_dot, l_dot, _, _, _, _ = frenet_trajectories
    x_traj = s_traj
    y_traj = l_traj * LANE_WIDTH 
    dot_x = s_dot
    dot_y = l_dot * LANE_WIDTH
    v_traj = jnp.sqrt(dot_x**2 + dot_y**2)
    psi_traj = jnp.arctan2(dot_y, dot_x)
    cartesian_trajectory = jnp.stack([x_traj, y_traj, v_traj, psi_traj], axis=1)
    return cartesian_trajectory

@jit
def generate_trajectory(params, initial_frenet_ctx):
    frenet_trajectories = theta_to_trajectory(params, initial_frenet_ctx)
    cartesian_trajectory = frenet_approx_to_cartesian_trajectory(frenet_trajectories)
    return cartesian_trajectory

def vmap_generate_trajectory_setup(num_vehicles):
    def generate_single_trajectory_wrapper(params, s_cur, l_cur, ds_cur):
        ctx = {'s_cur': s_cur, 'l_cur': l_cur, 'ds_cur': ds_cur} 
        return generate_trajectory(params, ctx)
    return jax.vmap(
        generate_single_trajectory_wrapper, 
        in_axes=(0, 0, 0, 0)
    )

# ----------------------------------------------------------------------
# II. 适应度函数 (核心修正：不再有 @jit 装饰器，并接受静态 V 和 param_dim)
# ----------------------------------------------------------------------

# NOTE: 这个函数不再使用 @jit 装饰器。它的 JIT 封装和静态参数处理将由 solve_intersection_mpc 完成。
def _collision_check_fn(positions_t, V_static):
    """检查时间步 t 的所有车辆对之间的碰撞，V_static 是静态值。"""
    i_indices, j_indices = jnp.triu_indices(V_static, k=1)
    
    def pairwise_dist_sq(p_i, p_j):
        return jnp.sum((p_i - p_j)**2)

    p_i_all = positions_t[i_indices]
    p_j_all = positions_t[j_indices]
    
    dist_sq_pairs = jax.vmap(pairwise_dist_sq)(p_i_all, p_j_all)
    min_dist_sq = jnp.min(dist_sq_pairs)
    
    collision_violation = jnp.maximum(0.0, COLLISION_DISTANCE_SQ - min_dist_sq)
    return collision_violation
    
# 核心适应度函数 (接受静态参数 V 和 param_dim)
def fitness_fn_multi_vehicle_inner(
    all_params_flattened: jnp.ndarray, 
    context: Dict[str, Any], 
    V_static: int, 
    param_dim_static: int
):
    """
    多车MPC的适应度函数 J(U, c)。
    
    Args:
        all_params_flattened: 扁平化的参数 (V * TOTAL_DIM,)
        context: 动态数组上下文 (s_cur, goal_points, etc.)
        V_static: 车辆数 (静态整数)
        param_dim_static: 参数维度 (静态整数)
    """
    
    # 1. 核心修正: 将扁平化参数重塑为 (V, TOTAL_DIM)，此时 V 和 param_dim 是静态的
    all_params = all_params_flattened.reshape(V_static, param_dim_static)
    
    # 从 Context 中提取 V 个车辆的初始状态
    initial_s_cur = context['initial_s_cur']
    initial_l_cur = context['initial_l_cur']
    initial_ds_cur = context['initial_ds_cur']
    goal_points = context['goal_points']      
    
    vmap_generate_trajectory = vmap_generate_trajectory_setup(V_static)

    # --- 1. 轨迹生成 ---
    all_trajectories = vmap_generate_trajectory(
        all_params, initial_s_cur, initial_l_cur, initial_ds_cur
    )

    # --- 2. 目标成本 J_Goal ---
    final_positions = all_trajectories[:, -1, :2] 
    goal_cost = Q_GOAL * jnp.sum(jnp.linalg.norm(final_positions - goal_points, axis=1)**2)
    
    # --- 3. 舒适性成本 J_Comfort ---
    comfort_cost = R_COMFORT * jnp.sum(all_params**2)
    
    # --- 4. 碰撞惩罚 J_Collision ---
    all_positions = all_trajectories[:, :, :2].transpose((1, 0, 2)) 
    
    vmap_collision_check = jax.vmap(
        _collision_check_fn, in_axes=(0, None)
    )
    all_violations = vmap_collision_check(all_positions, V_static) 
    collision_cost = C_COLLISION * jnp.sum(all_violations)
    
    # --- 5. 总成本 ---
    total_cost = goal_cost + comfort_cost + collision_cost
    
    return total_cost


# ----------------------------------------------------------------------
# III. MoG 参数初始化函数 (保持不变)
# ----------------------------------------------------------------------
# ... (initialize_mog_params 保持不变) ...

def initialize_mog_params(key: jnp.ndarray, num_vehicles: int, param_dim: int, K: int):
    M = num_vehicles
    key_mu, key_L, key_v = random.split(key, 3)
    initial_mu_k = random.uniform(key_mu, shape=(M, K, param_dim), minval=-0.5, maxval=0.5)
    initial_L_inv_k = jnp.tile(jnp.eye(param_dim)[None, None, :, :], (M, K, 1, 1))
    initial_v_k = jnp.zeros((M, K)) 
    return initial_mu_k, initial_L_inv_k, initial_v_k


# ----------------------------------------------------------------------
# IV. 主 MPC 求解函数 (关键修正：封装 JIT 和静态参数)
# ----------------------------------------------------------------------

def solve_intersection_mpc(
    key, current_frenet_states_V3, current_goals_V2, initial_mog_state, 
    V=2, 
    T_IGO=50, DELTA_T_IGO=0.5, K_MOG=8, B_SAMPLES=60, B0_ELITES=25
):
    param_dim = TOTAL_DIM # 26
    
    # 1. 设置动态上下文 (只包含 JAX 数组)
    dynamic_context = {
        'initial_s_cur': current_frenet_states_V3[:, 0],
        'initial_l_cur': current_frenet_states_V3[:, 1],
        'initial_ds_cur': current_frenet_states_V3[:, 2],
        'goal_points': current_goals_V2,
    }
    
    # 2. 创建 JIT 编译的适应度函数封装 (使用 static_argnames 传递静态参数)
    @functools.partial(
        jit, static_argnames=['V_static', 'param_dim_static']
    )
    def jit_fitness_fn(all_params_flattened, context, V_static, param_dim_static):
        return fitness_fn_multi_vehicle_inner(
            all_params_flattened, context, V_static, param_dim_static
        )

    # 3. 使用 partial 预绑定静态参数 V 和 param_dim，生成优化器所需的函数接口
    bound_fitness_fn = functools.partial(
        jit_fitness_fn, 
        V_static=V, 
        param_dim_static=param_dim
    )
    # bound_fitness_fn 现在只接受 (all_params_flattened, dynamic_context)

    # 4. 调用黑箱求解器
    initial_mu_k, initial_L_inv_k, initial_v_k = initial_mog_state
    DIMS_TUPLE = tuple([param_dim] * V)
    
    final_mu_k, final_L_inv_k, final_v_k_all = mmog_igo_optimizer_mpc(
        key, T_IGO, DELTA_T_IGO, V, K_MOG, B_SAMPLES, B0_ELITES, 
        DIMS_TUPLE, 0, 
        bound_fitness_fn,  # <--- 使用预绑定静态参数的函数
        initial_mu_k, initial_L_inv_k, initial_v_k,
        context=dynamic_context # <--- 传递动态上下文
    )
    
    # 5. 提取最佳参数
    v_exp = jnp.exp(final_v_k_all - jnp.max(final_v_k_all, axis=1, keepdims=True))
    final_pi_k = v_exp / jnp.sum(v_exp, axis=1, keepdims=True)
    best_comp_indices = jnp.argmax(final_pi_k, axis=1)
    best_params_V_P = final_mu_k[jnp.arange(V), best_comp_indices] 
    
    # 6. 生成最佳轨迹 (V x P 形状)
    vmap_generate_trajectory = vmap_generate_trajectory_setup(V)
    best_trajectories_V_N_D = vmap_generate_trajectory(
        best_params_V_P, 
        dynamic_context['initial_s_cur'], 
        dynamic_context['initial_l_cur'], 
        dynamic_context['initial_ds_cur']
    )
    
    next_mog_state = (final_mu_k, final_L_inv_k, final_v_k_all)
    
    return best_params_V_P, best_trajectories_V_N_D, next_mog_state

# ----------------------------------------------------------------------
# V. 示例运行 (保持不变)
# ----------------------------------------------------------------------

if __name__ == '__main__':
    print("--- 多车 MPC 求解器框架配置完成 ---")
    print(f"预测时域 HORIZON: {HORIZON}")
    print(f"单车优化维度 TOTAL_DIM: {TOTAL_DIM}")
    print("轨迹生成：调用 parameterization.theta_to_trajectory + 直道近似转换 (X=S, Y=L*LANE_WIDTH)")
    
    NUM_V = 2
    KEY = jax.random.PRNGKey(42)

    frenet_state_1 = jnp.array([0.0, 0.5, 5.0]) 
    frenet_state_2 = jnp.array([0.0, -0.5, 4.0])
    current_frenet_states = jnp.stack([frenet_state_1, frenet_state_2])
    
    goal_1 = jnp.array([100.0, 0.5])
    goal_2 = jnp.array([0.5, 100.0])
    current_goals = jnp.stack([goal_1, goal_2])
    
    K_MOG_COMPONENTS = 8
    
    key_init, key_solve = jax.random.split(KEY)
    initial_mog_state = initialize_mog_params(
        key_init, num_vehicles=NUM_V, param_dim=TOTAL_DIM, K=K_MOG_COMPONENTS
    )
    
    print("\n--- 启动 MPC 求解 ---")

    try:
        best_params, best_trajectories, final_mog_state = solve_intersection_mpc(
            key=key_solve, 
            current_frenet_states_V3=current_frenet_states, 
            current_goals_V2=current_goals, 
            initial_mog_state=initial_mog_state,
            V=NUM_V,
            K_MOG=K_MOG_COMPONENTS
        )
        
        print("\n--- 求解结果 ---")
        print(f"优化后的最佳参数形状: {best_params.shape} (V={NUM_V} x P={TOTAL_DIM})")
        print(f"最佳轨迹形状: {best_trajectories.shape} (V={NUM_V} x N={HORIZON} x D={CARTESIAN_STATE_DIM})")
        
    except Exception as e:
        print(f"\n致命错误：主求解函数执行失败，请检查 MPCsolverM.py 中的 mmog_igo_optimizer_mpc 函数是否可用且接口正确。")
        print(f"具体错误信息: {e}")