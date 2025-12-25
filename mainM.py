import jax
import jax.numpy as jnp
from jax import random, vmap, lax
import functools
import time
from gmm_igo.solverM1 import mmog_igo_optimizer

# ----------------------------------------------------------------------
# II. 自行车模型与 B-SPLINE 参数化
# ----------------------------------------------------------------------

# 2阶 B-spline (3个控制点)
@jax.jit
def bspline_2nd_order(t, control_points):
    """t 需归一化到 [0, 1]"""
    t_sq = t * t
    t_minus_1 = 1.0 - t
    t_minus_1_sq = t_minus_1 * t_minus_1
    
    B0 = t_minus_1_sq / 2.0
    B1 = (-2.0 * t_sq + 2.0 * t + 1.0) / 2.0
    B2 = t_sq / 2.0
    
    B_vec = jnp.stack([B0, B1, B2])
    return jnp.dot(B_vec, control_points)

# 2D 运动学自行车模型 (离散时间)
@jax.jit
def bicycle_dynamics(x, u, dt, L=2.5):
    """x: [px, py, theta, v], u: [a, delta]"""
    px, py, theta, v = x
    a, delta = u
    
    px_dot = v * jnp.cos(theta)
    py_dot = v * jnp.sin(theta)
    theta_dot = v / L * jnp.tan(delta)
    v_dot = a
    
    px_new = px + px_dot * dt
    py_new = py + py_dot * dt
    theta_new = theta + theta_dot * dt
    v_new = v + v_dot * dt
    
    v_new = jnp.clip(v_new, a_min=0.0, a_max=20.0) # 速度限制
    
    return jnp.stack([px_new, py_new, theta_new, v_new])

# --- 模拟与代价函数 ---

def simulate_and_cost(control_points_M, T_seg, N_steps, dt):
    """运行 M 段串联的模拟并计算总代价。"""
    M = control_points_M.shape[0]
    
    # 初始状态 [px, py, theta, v]
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0]) 
    
    # 参考目标 (终点：(40, 0), 速度 5m/s)
    x_ref = jnp.array([10.0 * M, 0.0, 0.0, 5.0]) 
    
    # 权重 (Q/R)
    Q_x = jnp.array([1.0, 1.0, 0.1, 0.5])
    R_u = jnp.array([0.01, 0.1])
    
    def segment_scan_fn(carry, segment_idx):
        x_t = carry
        cps = control_points_M[segment_idx]
        a_cps = cps[:3]
        delta_cps = cps[3:]
        
        def step_scan_fn(carry_inner, step_k):
            x_k, J_u_sum = carry_inner
            
            t_norm = step_k / N_steps 
            
            a_k = bspline_2nd_order(t_norm, a_cps)
            delta_k = bspline_2nd_order(t_norm, delta_cps)
            u_k = jnp.array([a_k, delta_k])
            
            x_k_plus_1 = bicycle_dynamics(x_k, u_k, dt)
            
            # 累计 u 的二次代价
            J_u_sum_new = J_u_sum + jnp.sum(R_u * u_k**2) * dt
            
            return (x_k_plus_1, J_u_sum_new), None

        init_inner = (x_t, 0.0)
        (x_final_seg, J_u_sum_seg), _ = lax.scan(
            step_scan_fn, init_inner, jnp.arange(N_steps)
        )
        
        return x_final_seg, J_u_sum_seg

    segment_indices = jnp.arange(M)
    x_final_M, J_u_M = lax.scan(
        segment_scan_fn, x0, segment_indices
    )
    
    # 终端代价
    terminal_cost = jnp.sum(Q_x * (x_final_M - x_ref)**2)
    
    total_u_cost = jnp.sum(J_u_M)
    total_cost = terminal_cost + total_u_cost
    
    return total_cost

# --- 目标函数 (fitness_fn_total) ---

M_SEGMENTS = 3   # 段数 M
D_SEG = 6        # 每段维度 D=6
N_STEPS = 100    # 每段模拟步数
T_TOTAL = 15.0    # 总时间 (秒)
DT_SEG = T_TOTAL / M_SEGMENTS / N_STEPS

@jax.jit
def fitness_fn_total(samples_overall, saturation_limit=10.0):
    """
    整体适应度函数 f(xi_1, ..., xi_B)。
    samples_overall: (M*D,) 形状的单个整体样本 \xi
    """
    M = M_SEGMENTS
    D = D_SEG
    
    # 解包：(M*D) -> (M, D)
    control_points_M = samples_overall.reshape((M, D))
    
    J_raw = simulate_and_cost(control_points_M, T_TOTAL / M, N_STEPS, DT_SEG)
    
    # 饱和函数：上限很小，不超过 10
    J_saturated = jnp.clip(J_raw, a_max=saturation_limit)
    
    return J_saturated 

# --- IV. 主程序运行逻辑 ---
if __name__ == '__main__':
    # ------------------- 配置 -------------------
    SEED = 42
    T_RUN = 800         # 循环轮数
    DELTA_T = 0.5       # 步长 
    K_COMP = 15         # 分量数量
    M_SEG = M_SEGMENTS  # 段数 M=4
    D_DIM = D_SEG       # 每段维度 D=6
    D_TOTAL = M_SEG * D_DIM # 总维度 24
    B_SAMPLES = 60     # 样本数量
    B_0_ELITE = 25      # 精英样本 (40%)
    
    # ------------------- 初始化 -------------------
    def initialize_params_mmog(key, M, K, D):
        key_mu, key_L, key_v = random.split(key, 3)
        
        # 1. 初始均值 mu_k (M, K, D) - 尽可能分散
        # 假设控制点 a in [-2, 2], delta in [-pi/4, pi/4]
        mu_flat = random.uniform(key_mu, (M * K * D,), minval=-3.0, maxval=3.0)
        initial_mu_k = mu_flat.reshape((M, K, D))

        # 2. 初始协方差 L_inv_k (M, K, D, D)
        L_inv_template = jnp.eye(D) * (1.0 / jnp.sqrt(4.0)) # 初始方差 Sigma = 4*I
        initial_L_inv_k = jnp.stack([L_inv_template] * (M * K)).reshape((M, K, D, D))
        
        # 3. 初始权重 v_k (M, K-1) - 均匀权重 (v=0)
        initial_v_k = jnp.zeros((M, K - 1))
        
        return initial_mu_k, initial_L_inv_k, initial_v_k

    key = random.PRNGKey(SEED)
    key_init, key_run = random.split(key)

    initial_mu_k, initial_L_inv_k, initial_v_k = initialize_params_mmog(
        key_init, M_SEG, K_COMP, D_DIM
    )
    
    # ------------------- 运行 -------------------
    print("--- 多段 M-MoG IGO 优化器开始 ---")
    print(f"总维度 D_total={D_TOTAL}, 段数 M={M_SEG}, 分量 K={K_COMP}, 迭代 T={T_RUN}")

    # 将数据转移到设备 (GPU 或 CPU)
    device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0]
    initial_mu_k = jax.device_put(initial_mu_k, device)
    
    t_start_opt = time.perf_counter()
    
    # 调用优化器
    final_mu_k, final_L_inv_k, final_pi_k = mmog_igo_optimizer(
        key_run, T_RUN, DELTA_T, M_SEG, K_COMP, B_SAMPLES, B_0_ELITE, fitness_fn_total,
        initial_mu_k, initial_L_inv_k, initial_v_k
    )

    # 强制同步计时
    final_mu_k.block_until_ready()
    t_opt = time.perf_counter() - t_start_opt

    print(f"[时间] M-MoG 主循环耗时 (同步): {t_opt:.4f} 秒 (T={T_RUN})")
    print(f"平均每轮迭代时间: {t_opt / T_RUN:.6f} 秒/iter")
    
    # --- 结果分析 ---
    # 找到最佳整体样本 (通过评估最终采样的精英样本)
    key_eval = random.PRNGKey(420)
    samples_M = _vmap_sample_from_mog_batch(
        random.split(key_eval, M_SEG * B_SAMPLES + M_SEG).reshape((M_SEG, B_SAMPLES + 1, 2))[:, 1:], 
        final_mu_k, final_L_inv_k, final_pi_k, B_SAMPLES
    )
    
    samples_overall_B = jnp.transpose(samples_M, (1, 0, 2)).reshape((B_SAMPLES, D_TOTAL))
    f_xi_final = vmap(fitness_fn_total)(samples_overall_B).block_until_ready()
    
    best_overall_cost = jnp.min(f_xi_final)
    
    print("\n--- 优化结果总结 ---")
    print(f"最终最佳采样代价 (饱和后): {best_overall_cost:.4f} (目标值接近 0)")