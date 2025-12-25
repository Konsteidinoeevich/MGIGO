# main.py - 混合高斯信息几何优化器运行文件 (增强打印所有分量 + 运行时间统计)

import jax
import jax.numpy as jnp
from jax import random, vmap
import numpy as np
import time  # <-- 新增：用于精确计时

# 假设 solver.py 已经在同一目录下
from gmm_igo.solver import igo_mog_optimizer 

# --- I. 配置参数 (使用当前配置) ---
SEED = 42
T_RUN = 1000      # 循环轮数
DELTA_T = 0.1     # 步长 (使用当前值)
K_COMP = 15       # 分量数量
D_DIM = 3         # 维度 (高维测试)
B_SAMPLES = 60    # 样本数量 (使用当前值)
B_0_ELITE = 25    # 精英样本 (使用当前值)

# --- II. 适应度函数 (目标：最小化多全局最优的周期函数) ---
@jax.jit
def fitness_fn(x):
    """
    Simpler Multi-Global Optima Function (3^D 个最优解)。
    全局最优点在每个坐标 x_i = -4, 0, 或 4 处。
    """
    dist_to_0 = x**2
    dist_to_p4 = (x - 3.0)**2
    dist_to_m4 = (x + 3.0)**2
    
    min_dist_sq = jnp.min(jnp.stack([dist_to_0, dist_to_p4, dist_to_m4]), axis=0)
    return jnp.sum(min_dist_sq)

# --- III. 初始化函数 ---
def initialize_params(key, K, D):
    key_mu, key_L, key_pi = random.split(key, 3)
    
    initial_mu_k = random.uniform(key_mu, (K, D), minval=-5.0, maxval=5.0)
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

    print("--- 外部调用 IGO 优化器开始 (最终理论和数值一致版) ---")
    print(f"维度 D={D_DIM}, 分量 K={K_COMP}, 迭代 T={T_RUN}")
    print(f"样本 B={B_SAMPLES}, 精英 B0={B_0_ELITE}, 步长 dt={DELTA_T}")
    print(f"[时间] 参数初始化耗时: {t_init:.4f} 秒")

    # --- 优化器运行阶段 ---
    t_start_opt = time.perf_counter()
    final_mu_k, final_L_inv_k, final_pi_k = igo_mog_optimizer(
        key_run, T_RUN, DELTA_T, K_COMP, B_SAMPLES, B_0_ELITE, fitness_fn,
        initial_mu_k, initial_L_inv_k, initial_pi_k
    )
    t_opt = time.perf_counter() - t_start_opt

    print(f"[时间] IGO 优化器主循环耗时: {t_opt:.4f} 秒")

    # --- 结果分析阶段 ---
    t_start_analysis = time.perf_counter()
    f_mu_all = vmap(fitness_fn)(final_mu_k)
    best_comp_idx = jnp.argmin(f_mu_all)
    t_analysis = time.perf_counter() - t_start_analysis

    # === 总耗时统计 ===
    t_total = time.perf_counter() - t_start_total

    print("\n--- 优化结果总结 ---")
    print(f"最终权重 (pi_k) 最小值: {jnp.min(final_pi_k):.4f}, 最大值: {jnp.max(final_pi_k):.4f}")
    print(f"最佳分量 ({best_comp_idx}) 的适应度 f(mu): {f_mu_all[best_comp_idx]:.6e} (目标值: 0.0)")

    print("\n--- 所有分量 µ_k 状态 (前5维坐标) ---")
    for k in range(K_COMP):
        mu_k = final_mu_k[k]
        pi_k = final_pi_k[k]
        f_k = f_mu_all[k]
        print(f"K={k:2d}: π={pi_k:.4f} | f(µ)={f_k:.6e} | µ[:5]={mu_k[:5]}")

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