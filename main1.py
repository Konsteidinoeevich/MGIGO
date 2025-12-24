# test_solver_timing.py - IGO-MoG 优化器计时测试用例

import jax
import jax.numpy as jnp
from jax import random, vmap
import functools 
import time 

# 假设 solver.py 已经在同一目录下
from gmm_igo.solver import igo_mog_optimizer 


# --- I. 配置参数 (单 MoG 配置) ---
SEED = 42
T_RUN = 1000       # T: 循环轮数
DELTA_T = 0.1     # 步长
K_COMP = 15        # K: 活跃分量数量 (总 K+1 个分量)
D_DIM = 8        # D: 搜索空间的维度
B_SAMPLES = 60    # B: 样本数量
B_0_ELITE = 25    # B_0: 精英样本数量
NUM_RUNS = 10     # 计时测试的运行次数
TARGET_VALUE = jnp.array([0.0,3.0,-3.0]) # 函数目标值


# --- II. 适应度函数 (单 MoG - Sphere) ---
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

# --- III. 初始化函数 (单 MoG) ---
def initialize_params_single_mog(key, K, D):
    """设置分散的初始均值和稳定的初始方差。"""
    key_mu, key_L = random.split(key)
    
    # 1. 初始均值 mu_k: (K+1, D) - 分散 (-5.0 到 5.0)
    initial_mu_k = random.uniform(key_mu, (K+1, D), minval=-5.0, maxval=5.0)
    
    # 2. 初始逆乔列斯基因子 L_inv_k: 对应信息矩阵 S = 2.0 * I
    L_inv_template = jnp.eye(D) * 1.5
    initial_L_inv_k = jnp.stack([L_inv_template] * (K + 1))
    
    # 3. 初始权重 pi_k: (K+1,) - 均匀权重 
    initial_pi_k = jnp.full((K + 1,), 1.0 / (K + 1))
    
    return initial_mu_k, initial_L_inv_k, initial_pi_k

# --- IV. 主程序运行逻辑 ---
if __name__ == '__main__':
    
    print("--- IGO-MoG 优化器计时测试开始 (基于 solver.py) ---")
    print(f"分量数量 K={K_COMP} (总 K+1={K_COMP+1}), 维度 D={D_DIM}, 迭代 T={T_RUN}")
    
    key = random.PRNGKey(SEED)
    key_init, key_run = random.split(key)
    
    initial_mu_k, initial_L_inv_k, initial_pi_k = initialize_params_single_mog(
        key_init, K_COMP, D_DIM
    )

    # --- 优化器运行阶段 ---
    print(f"\n--- 运行 IGO 优化器主循环 ({T_RUN} 轮) ---")
    
    # -----------------------------------------------------------------
    # [计时循环]
    # -----------------------------------------------------------------
    total_time = 0.0
    
    print(f"开始计时测试，运行 {NUM_RUNS} 次...")

    for run in range(NUM_RUNS):
        key_run, subkey = random.split(key_run)
        
        start_time = time.time()
        
        # 调用优化器
        final_mu_k, final_L_inv_k, final_pi_k_all = igo_mog_optimizer(
            subkey, T_RUN, DELTA_T, K_COMP + 1, B_SAMPLES, B_0_ELITE, 
            fitness_fn,
            initial_mu_k, initial_L_inv_k, initial_pi_k
        )
        
        # 确保 JAX 计算完成 (非常重要)
        final_mu_k.block_until_ready()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        if run == 0:
            print(f"第一次运行（包含 JIT 编译）耗时: {elapsed_time:.4f} 秒。")

    # -----------------------------------------------------------------
    # [结果打印]
    # -----------------------------------------------------------------
    avg_time = total_time / NUM_RUNS
    print(f"\n--- 计时结果 ({NUM_RUNS} 次运行) ---")
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"**平均每次运行耗时: {avg_time:.4f} 秒**")
    
    # --- 结果分析 (基于最后一次运行) ---
    print(f"\n--- 优化结果总结 ---")
    
    # 找到权重最高的均值 (最佳分量，总共 K+1 个)
    best_comp_index = jnp.argmax(final_pi_k_all) 
    mu_star = final_mu_k[best_comp_index]
    
    # 评估最佳样本的适应度
    f_mu_star = fitness_fn(mu_star)
    
    # 检查收敛目标
    mean_error = jnp.mean(jnp.min(jnp.abs(mu_star - TARGET_VALUE), axis=1))

    print(f"**最终适应度 f(mu*): {f_mu_star:.6e} (理论最优值: 0.0)**")
    print(f"**收敛精度检查: 平均误差到目标点 ({TARGET_VALUE:.1f}): {mean_error:.6e}**")
    print(f"最佳分量索引: {best_comp_index}, 权重: {final_pi_k_all[best_comp_index]:.4f}")

    print("\n--- 结论 ---")
    if mean_error < 1e-2:
        print(f"IGO-MoG 优化器成功收敛到目标点。")
    else:
        print("优化器未达到理想精度。")