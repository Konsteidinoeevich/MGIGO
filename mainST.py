# main.py - 混合高斯信息几何优化器运行文件 (Styblinski-Tang + T0 重置)

import jax
import jax.numpy as jnp
from jax import random, vmap
import numpy as np
import time  

# 假设 solver.py 已经在同一目录下
from gmm_igo.solverr import igo_mog_optimizer 

# --- I. 配置参数 (使用原参数，D=4) ---
SEED = 42
T_RUN = 10000      # 循环轮数
DELTA_T = 0.08     # 步长 (您的原值)
K_COMP = 20      # 分量数量 (您的原值)
D_DIM = 32         # 维度 (您的值)
B_SAMPLES = 100    # 样本数量
B_0_ELITE = 30    # 精英样本
T_0_RESTART = 1000 # 重置周期 ($T_0$)

# --- II. 适应度函数 (Styblinski-Tang's Function) ---
@jax.jit
def fitness_fn(x):
    """
    Styblinski-Tang Function (D dimensions). 
    全局最优值在每个坐标 x_i ≈ -2.9035
    """
    term = x**4 - 16.0 * x**2 + 5.0 * x 
    return jnp.sum(term)

# --- III. 初始化函数 ---
def initialize_params(key, K, D):
    key_mu, key_L, key_pi = random.split(key, 3)
    
    # 调整初始化范围，包含 Styblinski-Tang 的全局最优点 (-2.9)
    initial_mu_k = random.uniform(key_mu, (K, D), minval=-4.0, maxval=4.0) 
    L_inv_template = jnp.eye(D) * jnp.sqrt(2.0)
    L_inv_k = jnp.stack([L_inv_template] * K)
    initial_pi_k = jnp.ones(K) / K
    
    return initial_mu_k, L_inv_k, initial_pi_k

# --- IV. 主程序运行逻辑 ---
if __name__ == '__main__':
    # === 全局计时开始 ===
    t_start_total = time.perf_counter()

    key = random.PRNGKey(SEED)
    key_init, key_run = random.split(key)

    # --- 初始化阶段 ---
    t_start_init = time.perf_counter()
    initial_mu_k, initial_L_inv_k, initial_pi_k = initialize_params(key_init, K_COMP, D_DIM)
    t_init = time.perf_counter() - t_start_init

    # --- 优化器运行阶段 (正式计时) ---
    t_start_opt = time.perf_counter()
    # 调用时参数顺序正确 (11个参数)
    final_mu_k, final_L_inv_k, final_pi_k = igo_mog_optimizer(
        key_run, T_RUN, DELTA_T, K_COMP, B_SAMPLES, B_0_ELITE, fitness_fn, T_0_RESTART,
        initial_mu_k, initial_L_inv_k, initial_pi_k
    )
    final_mu_k = jax.device_get(final_mu_k) 
    t_opt = time.perf_counter() - t_start_opt

    print(f"[时间] IGO 优化器主循环耗时: {t_opt:.4f} 秒")

    # --- 结果分析阶段 ---
    t_start_analysis = time.perf_counter()
    f_mu_all = vmap(fitness_fn)(final_mu_k)
    best_comp_idx = jnp.argmin(f_mu_all)
    t_analysis = time.perf_counter() - t_start_analysis

    # === 总耗时统计 ===
    t_total = time.perf_counter() - t_start_total
    
    # Styblinski-Tang Function 在 x_i ≈ -2.9035 处的理论目标值
    TARGET_F = -39.166166 * D_DIM # D=4 时，目标值为 -156.6647

    print("\n--- 优化结果总结 ---")
    print(f"最终权重 (pi_k) 最小值: {jnp.min(final_pi_k):.4f}, 最大值: {jnp.max(final_pi_k):.4f}")
    print(f"最佳分量 ({best_comp_idx}) 的适应度 f(mu): {f_mu_all[best_comp_idx]:.6e} (目标值: {TARGET_F:.4f})")

    print(f"\n--- 所有分量 µ_k 状态 (前 {D_DIM} 维坐标) ---")
    for k in range(K_COMP):
        mu_k = final_mu_k[k]
        pi_k = final_pi_k[k]
        f_k = f_mu_all[k]
        # 打印 D_DIM 维坐标
        print(f"K={k:2d}: π={pi_k:.4f} | f(µ)={f_k:.6e} | µ[:{D_DIM}]={mu_k[:D_DIM]}")

    # === 详细时间报告 ===
    print(jax.devices())
    print("\n" + "="*50)
    print("           运行时间详细报告")
    print("="*50)
    print(f"初始化阶段           : {t_init:8.4f} 秒")
    print(f"优化器主循环          : {t_opt:8.4f} 秒")
    print(f"结果分析与打印        : {t_analysis:8.4f} 秒")
    print("-" * 50)
    print(f"总耗时 (Total)        : {t_total:8.4f} 秒")
    print(f"平均每轮迭代时间      : {t_opt / T_RUN:8.6f} 秒/iter")
    print(f"当前时间              : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)