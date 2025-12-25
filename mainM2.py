# mainM2.py - M-MoG IGO (MPC + T0 重启) 测试用例

import jax
import jax.numpy as jnp
from jax import random, vmap
import functools 
import numpy as np 
import time # [新增] 导入 time 模块用于计时
from gmm_igo.MPCsolverM1 import mmog_igo_optimizer_mpc


# --- I. 配置参数 (M-MoG + T0 + Context 配置) ---
SEED = 42
T_RUN = 450       # T: 循环轮数
DELTA_T = 0.1     # 步长
M_MOG = 3         # M: MoG 数量 (子空间数量)
K_COMP = 8        # K: 每个 MoG 的分量数量
D_DIM = 8         # D: 每个 MoG 搜索空间的维度
B_SAMPLES = 60    # B: 整体样本数量
B_0_ELITE = 25    # B_0: 精英样本数量
T_0_RESTART = 150 # T_0: 周期性热重启轮数 (每 100 轮重置方差和权重)
NUM_RUNS = 10     # [新增] 计时测试的运行次数

# --- II. 模拟动态 Context 和适应度函数 ---

# 全局目标值，用于适应度函数 (这将由 Context 动态覆盖)
TARGET_VALUE = 1.0 

def create_mpc_coupled_sphere_fitness_fn(M_MOG, D_DIM):
    """
    创建一个 Context 敏感的耦合 Sphere 函数。
    context: 假设是一个 JAX 数组，其中第一个元素是动态目标值。
    """
    COUPLING_WEIGHT = 0.1 # 耦合项权重，用于保证 M 个 MoG 倾向于收敛到相似的均值
    
    @jax.jit
    def fitness_fn_total(samples_overall, context):
        """
        samples_overall: (M*D,) - 展平后的总样本向量
        context: (D_context,) - 动态参数，context[0] 是动态目标值
        """
        # 从 Context 提取动态目标值
        dynamic_target_value = context[0] 
        
        # 1. 结构重塑: (M*D,) -> (M, D)
        samples_M = samples_overall.reshape((M_MOG, D_DIM))
        
        # 2. 局部项 (Sphere Term): 目标接近 dynamic_target_value
        # 适应度函数旨在最小化，全局最优 f=0.0
        f_local = jnp.sum((samples_M - dynamic_target_value)**2)
        
        # 3. 耦合项 (Coupling Term): 强制所有 MoG 保持一致
        samples_M_shifted = jnp.roll(samples_M, shift=-1, axis=0) 
        diff = samples_M - samples_M_shifted
        f_coupling = jnp.sum(diff**2)
        
        # 总目标函数
        f_total = f_local + COUPLING_WEIGHT * f_coupling
        return f_total

    return fitness_fn_total

# --- III. 初始化函数 (分散初始分布) ---
def initialize_params_mmog_dispersed(key, M, K, D):
    """设置分散的初始均值和稳定的初始方差。"""
    key_mu, key_L, key_v = random.split(key, 3)
    
    # 1. 初始均值 mu_k: (M, K+1, D) - 分散 (-3.0 到 3.0)
    initial_mu_k_active = random.uniform(key_mu, (M, K, D), minval=-3.0, maxval=3.0)
    initial_mu_k_bg = jnp.zeros((M, 1, D)) # 背景分量均值为 0
    initial_mu_k = jnp.concatenate([initial_mu_k_active, initial_mu_k_bg], axis=1)

    # 2. 初始逆乔列斯基因子 L_inv_k: 对应 Sigma = 1.0 * I (S = 1.0 * I)
    L_inv_template = jnp.eye(D) * 1.414 
    L_inv_k_all = jnp.stack([L_inv_template] * (K + 1))
    initial_L_inv_k = jnp.stack([L_inv_k_all] * M)
    
    # 3. 初始权重参数 v_k: (M, K) - 均匀权重 (v=0)
    initial_v_k = jnp.zeros((M, K))
    
    return initial_mu_k, initial_L_inv_k, initial_v_k

# --- IV. 主程序运行逻辑 ---
if __name__ == '__main__':
    
    fitness_fn_total = create_mpc_coupled_sphere_fitness_fn(M_MOG, D_DIM)

    key = random.PRNGKey(SEED)
    key_init, key_run = random.split(key)
    
    # [关键修正]：定义子空间的维度，必须是可哈希的 Python 元组 (tuple)
    dims_tuple = tuple([D_DIM] * M_MOG) 
    
    print("--- M-MoG IGO (MPC + T0 重启) 优化器测试开始 ---")
    print(f"MoG 数量 M={M_MOG}, 每个 MoG 维度 D={D_DIM}, 总维度 D_TOTAL={M_MOG * D_DIM}")
    print(f"迭代 T={T_RUN}, 周期性重启 T0={T_0_RESTART}")
    
    initial_mu_k, initial_L_inv_k, initial_v_k = initialize_params_mmog_dispersed(
        key_init, M_MOG, K_COMP, D_DIM
    )

    # **模拟动态 Context**
    current_context = jnp.array([2.0, 0.0, 0.0, 0.0]) 

    # --- 优化器运行阶段 ---
    print(f"\n--- 运行 IGO 优化器主循环 (目标值由 Context 决定: {current_context[0]:.1f}) ---")
    
    # -----------------------------------------------------------------
    # [计时循环]
    # -----------------------------------------------------------------
    total_time = 0.0
    
    print(f"开始计时测试，运行 {NUM_RUNS} 次...")

    # 我们只在第一次运行时打印 Traceback 和结果，之后的运行只用于计时
    for run in range(NUM_RUNS):
        # 确保每次运行使用不同的 key
        key_run, subkey = random.split(key_run)
        
        start_time = time.time()
        
        # 调用优化器
        final_mu_k, final_L_inv_k, final_pi_k_all = mmog_igo_optimizer_mpc(
            subkey, T_RUN, DELTA_T, M_MOG, K_COMP, B_SAMPLES, B_0_ELITE, 
            dims_tuple, 
            T_0_RESTART, fitness_fn_total,
            initial_mu_k, initial_L_inv_k, initial_v_k,
            context=current_context 
        )
        
        # 确保 JAX 计算完成 (非常重要)
        final_mu_k.block_until_ready()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        if run == 0:
            print(f"第一次运行耗时: {elapsed_time:.4f} 秒。")

    # -----------------------------------------------------------------
    # [结果打印]
    # -----------------------------------------------------------------
    avg_time = total_time / NUM_RUNS
    print(f"\n--- 计时结果 ({NUM_RUNS} 次运行) ---")
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"**平均每次运行耗时: {avg_time:.4f} 秒**")
    
    # --- 结果分析 (基于最后一次运行) ---
    TARGET_VALUE = current_context[0] 
    print(f"\n--- 优化结果总结与 T0 策略检查 (基于最后一次运行) ---")
    
    # 找到每个 MoG 中权重最高的非背景分量
    best_comp_indices = jnp.argmax(final_pi_k_all[:, :-1], axis=1) # 排除背景分量
    mu_star_M = final_mu_k[jnp.arange(M_MOG), best_comp_indices]
    
    # 评估最佳整体样本的适应度
    samples_overall_best = mu_star_M.flatten()
    f_mu_star = fitness_fn_total(samples_overall_best, current_context)
    
    # 检查收敛目标
    mean_error = jnp.mean(jnp.abs(mu_star_M - TARGET_VALUE))

    print(f"**最终适应度 f(mu*): {f_mu_star:.6e} (理论最优值: 0.0)**")
    print(f"**收敛精度检查: 平均误差到目标点 ({TARGET_VALUE:.1f}): {mean_error:.6e}**")
    
    print(f"\n--- M-MoG 最佳分量均值详情 ---")
    
    for m in range(M_MOG):
        k_idx = best_comp_indices[m]
        mu_m_k_star = mu_star_M[m]
        pi_m_k_star = final_pi_k_all[m, k_idx]
        
        print(f"MoG M={m} (K={k_idx}): 权重 π={pi_m_k_star:.4f}, 均值误差={jnp.mean(jnp.abs(mu_m_k_star - TARGET_VALUE)):.6e}")

    print("\n--- 结论 ---")
    if mean_error < 1e-2:
        print(f"M-MoG IGO 优化器在周期重启策略下成功收敛到动态目标 {TARGET_VALUE}。")
    else:
        print("优化器未达到理想精度，请检查学习率 (dt) 或 T0 设置。")