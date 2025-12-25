# mainM1.py - 对比：全热重启 vs 仅权重重置

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax 
import functools 
import time 
from typing import Tuple, List, Any

# 导入两个版本的优化器
# 确保您的求解器文件 gmm_igo/MPCsolver_Hetero.py 包含了这两个函数
from gmm_igo.MPCsolverM3 import mmog_igo_optimizer_mpc_weights_only_reset

# --- I. 配置参数 (异构 M-MoG + T0 + Context 配置) ---
SEED = 42
T_RUN = 500       # T: 循环轮数
DELTA_T = 0.1     # 步长
M_MOG = 3         # M: MoG 数量 (子空间数量)
K_COMP = 8        # K: 每个 MoG 的分量数量
DIMS_TUPLE = (2, 5, 8) # 关键: 异构维度
B_SAMPLES = 60    # B: 整体样本数量
B_0_ELITE = 25    # B_0: 精英样本数量
T_0_RESTART = 100 # T_0: 周期性热重启轮数
NUM_RUNS = 10     # 计时测试的运行次数
D_TOTAL = sum(DIMS_TUPLE)
D_MAX = max(DIMS_TUPLE)

# --- II. 异构适应度函数 (Context 敏感) ---

def create_mpc_coupled_sphere_fitness_fn_hetero(M_MOG, dims_tuple: Tuple[int, ...]):
    """
    创建一个 Context 敏感的耦合 Sphere 函数，支持异构维度切片和填充掩码。
    """
    COUPLING_WEIGHT = 0.1 
    D_MAX = max(dims_tuple)
    D_ARRAY = jnp.array(dims_tuple)

    @jax.jit
    def fitness_fn_total(samples_overall, context):
        """
        samples_overall: (M * D_MAX,) - 展平后的填充样本向量
        """
        dynamic_target_value = context[0] 
        
        # 1. 结构重塑: (M * D_MAX,) -> (M, D_MAX)
        samples_M_padded = samples_overall.reshape((M_MOG, D_MAX))
        
        # 2. 局部项 (Sphere Term): 使用掩码确保只计算实际维度
        
        def loop_body_local(m, f_sum):
            D_m = D_ARRAY[m]
            
            # 使用 lax.dynamic_slice 静态切片出整行 (D_MAX 维度是静态的)
            sample_m_padded = lax.dynamic_slice(
                samples_M_padded, 
                (m, 0),             
                (1, D_MAX) 
            )[0]
            
            # 创建掩码 (只保留前 D_m 个元素)
            mask = jnp.arange(D_MAX) < D_m
            
            # 将填充部分的样本和目标值都归零，确保只对实际 D_m 维度求和
            sample_m_actual = sample_m_padded * mask 
            target_masked = dynamic_target_value * mask
            
            return f_sum + jnp.sum((sample_m_actual - target_masked)**2)

        f_local = lax.fori_loop(0, M_MOG, loop_body_local, 0.0)

        # 3. 简化耦合项
        f_coupling_simplified = jnp.sum(samples_M_padded[:, :1] - jnp.roll(samples_M_padded[:, :1], -1, axis=0))**2
        
        f_total = f_local + COUPLING_WEIGHT * f_coupling_simplified
        return f_total

    return fitness_fn_total

# --- III. 异构初始化函数 (保持不变) ---

def initialize_params_mmog_heterogeneous(key, M: int, K: int, dims: Tuple[int, ...]):
    """
    初始化 M 个 MoG 的参数，每个 MoG 的维度 D_m 不同，并填充到 D_max。
    """
    D_max: int = max(dims)
    initial_mu_list: List[jnp.ndarray] = []
    initial_L_inv_list: List[jnp.ndarray] = []
    keys = random.split(key, M)

    for m in range(M):
        D_m = dims[m]
        key_mu, key_L = random.split(keys[m])
        
        mu_m_actual = random.uniform(key_mu, (K+1, D_m), minval=-3.0, maxval=3.0)
        mu_m_padded = jnp.pad(mu_m_actual, ((0, 0), (0, D_max - D_m)), mode='constant')
        initial_mu_list.append(mu_m_padded)

        L_inv_template = jnp.eye(D_m) * 1.414 
        L_inv_k_all = jnp.stack([L_inv_template] * (K + 1)) 
        
        L_inv_m_padded = jnp.pad(L_inv_k_all, ((0, 0), (0, D_max - D_m), (0, D_max - D_m)), mode='constant')
        initial_L_inv_list.append(L_inv_m_padded)

    # 初始权重 v_k 设为 0，对应均匀权重
    initial_v_k = jnp.zeros((M, K))
    
    initial_mu_k_stacked = jnp.stack(initial_mu_list)     
    initial_L_inv_k_stacked = jnp.stack(initial_L_inv_list) 
    
    return initial_mu_k_stacked, initial_L_inv_k_stacked, initial_v_k, D_max

# --- IV. 主程序运行逻辑 (已修改为对比实验) ---
if __name__ == '__main__':
    
    fitness_fn_total = create_mpc_coupled_sphere_fitness_fn_hetero(M_MOG, DIMS_TUPLE)

    key = random.PRNGKey(SEED)
    key_init, key_run = random.split(key)
    
    initial_mu_k, initial_L_inv_k, initial_v_k, D_max_check = initialize_params_mmog_heterogeneous(
        key_init, M_MOG, K_COMP, DIMS_TUPLE
    )
    
    current_context = jnp.array([2.0, 0.0, 0.0, 0.0]) 

    print("--- 异构维度 M-MoG IGO 优化器计时测试开始 ---")
    print(f"MoG 数量 M={M_MOG}, 维度 D={DIMS_TUPLE}, 总维度 D_TOTAL={D_TOTAL}")
    print(f"迭代 T={T_RUN}, 周期性重启 T0={T_0_RESTART}")
    print(f"\n--- 目标值由 Context 决定: {current_context[0]:.1f} ---")

    def run_experiment(optimizer_fn, label, initial_mu_k, initial_L_inv_k, initial_v_k, key_run):
        """运行指定的优化器并报告计时和结果。"""
        print(f"\n=======================================================")
        print(f"=== 运行策略: {label} (运行 {NUM_RUNS} 次) ===")
        print(f"=======================================================")
        
        total_time = 0.0
        final_mu_k = None
        final_pi_k_all = None
        
        # 确保每次实验都使用一个新的随机数种子
        key_run_exp, _ = random.split(key_run)
        
        for run in range(NUM_RUNS):
            key_run_exp, subkey = random.split(key_run_exp)
            
            start_time = time.time()
            
            # 调用选定的优化器函数
            mu, L_inv, pi = optimizer_fn(
                subkey, T_RUN, DELTA_T, M_MOG, K_COMP, B_SAMPLES, B_0_ELITE, 
                DIMS_TUPLE, 
                T_0_RESTART, fitness_fn_total,
                initial_mu_k, initial_L_inv_k, initial_v_k,
                context=current_context 
            )
            
            # 等待 JAX 计算完成 (用于精确计时)
            mu.block_until_ready()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            final_mu_k = mu
            final_pi_k_all = pi

            if run == 0:
                print(f"第一次运行耗时: {elapsed_time:.4f} 秒。")

        avg_time = total_time / NUM_RUNS
        print(f"\n--- 计时结果 ({NUM_RUNS} 次运行) ---")
        print(f"总耗时: {total_time:.4f} 秒")
        print(f"**平均每次运行耗时: {avg_time:.4f} 秒**")
        
        # --- 结果计算 ---
        TARGET_VALUE = current_context[0] 
        best_comp_indices = jnp.argmax(final_pi_k_all[:, :-1], axis=1) 
        mu_star_M_padded = final_mu_k[jnp.arange(M_MOG), best_comp_indices]
        
        samples_overall_best = mu_star_M_padded.flatten()
        f_mu_star = fitness_fn_total(samples_overall_best, current_context)
        
        mean_error_list = []
        for m in range(M_MOG):
            D_m = DIMS_TUPLE[m]
            mu_m_star_actual = mu_star_M_padded[m, :D_m] 
            mean_error_list.append(jnp.mean(jnp.abs(mu_m_star_actual - TARGET_VALUE)))

        mean_error = jnp.mean(jnp.array(mean_error_list))

        print(f"\n--- 优化结果总结 ---")
        print(f"**最终适应度 f(mu*): {f_mu_star:.6e} (理论最优值: 0.0)**")
        print(f"**收敛精度检查: 平均误差到目标点 ({TARGET_VALUE:.1f}): {mean_error:.6e}**")
        
        return avg_time, f_mu_star, mean_error

    # 分割 Key 用于两个独立的实验
    key_run_1, key_run_2 = random.split(key_run)

    # 2. 运行策略 B: 仅权重重置
    results_weights_only = run_experiment(
        mmog_igo_optimizer_mpc_weights_only_reset, 
        "B: 仅权重重置 (保留方差)", 
        initial_mu_k, initial_L_inv_k, initial_v_k, 
        key_run_2
    )

    # 最终性能对比
    print("\n=======================================================")
    print("=== 最终性能对比 (平均耗时 / 最终适应度) ===")
    print("=======================================================")
    print(f"策略 B (仅权重重置): 耗时 {results_weights_only[0]:.4f} 秒 | 适应度 {results_weights_only[1]:.6e} | 误差 {results_weights_only[2]:.6e}")