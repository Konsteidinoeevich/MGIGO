import jax
import jax.numpy as jnp
from jax import random, lax, jit
import time
# 修改导入名称，移除 _jit 后缀

print(f"{jax.devices()}")
from gmm_igo.MPCsolvermultipleproblems import parallel_mmog_igo_mpc

@jit
def f1_styblinski(x): return 0.5*jnp.sum(x**4 - 16.0 * x**2 + 5.0 * x)

@jit
def f2_quadratic(x): return jnp.sum(x**2)

@jit
def f3_multi_goal(x):
    min_dist_sq = jnp.min(jnp.stack([x**2, (x-3.0)**2, (x+3.0)**2]), axis=0)
    return jnp.sum(min_dist_sq)

@jit
def fitness_dispatcher(z_combined, context):
    """保持最小值逻辑"""
    task_id = context['task_id']
    res = lax.cond(
        task_id == 0,
        lambda x: 1.0 * f1_styblinski(x),
        lambda x: 1.0 * f3_multi_goal(x),
        operand=z_combined
    )
    return res 

def run_fast_loop():
    P, M, K, D_max = 4, 4, 20, 5
    T, DT, B, B0, T0 = 1000, 0.1, 60, 25, 200
    dims = (5, 5)
    
    key = random.PRNGKey(42)
    # 构造 context 数组
    context_P = {'task_id': jnp.array([0, 0, 1, 1])}

    # 1. 预热编译 (Warm-up) - 这一步会慢，但只执行一次
    print("正在预热编译，请稍候...")
    init_mu = random.uniform(key, (P, M, K, D_max))
    init_L = jnp.tile(jnp.eye(D_max) * 1.5, (P, M, K, 1, 1))
    init_v = jnp.zeros((P, M, K - 1))
    
    _ = parallel_mmog_igo_mpc(random.split(key, P), T, DT, M, K, B, B0, dims, T0, 
                              fitness_dispatcher, init_mu, init_L, init_v, context_P)

    # 2. 正式循环
    print(f"\n>>> 编译完成，开始循环测试 (P={P})...")
    for i in range(5):
        key, subkey = random.split(key)
        keys_P = random.split(subkey, P)
        
        # 模拟 MPC: 每次随机生成初始点
        mu_iter = random.uniform(keys_P[0], (P, M, K, D_max), minval=-3.0, maxval=3.0)
        
        t_start = time.perf_counter()
        
        # 并行执行
        mu_f, L_f, pi_f = parallel_mmog_igo_mpc(
            keys_P, T, DT, M, K, B, B0, dims, T0, 
            fitness_dispatcher, mu_iter, init_L, init_v, context_P
        )
        
        # 【核心】强制同步硬件，否则计时是假的
        mu_f.block_until_ready()
        
        t_end = time.perf_counter()
        print(f"循环 {i} 耗时: {t_end - t_start:.4f}s")
        for p_idx in range(P):
            print(f"\n--- 问题 {p_idx} (任务类型 ID: {context_P['task_id'][p_idx]}) 的结果 ---")
            
            # 找到该问题中每个块概率最大的分量索引
            # pi_f[p_idx] 维度是 (M, K)
            best_comp_indices = jnp.argmax(pi_f[p_idx], axis=1)
            
            for m in range(M):
                k_best = best_comp_indices[m]
                weight = pi_f[p_idx, m, k_best]
                mean_val = mu_f[p_idx, m, k_best]
                print(f"  块 {m} [最佳分量 {k_best}]: 权重={weight:.4f}, 均值={mean_val}")

if __name__ == "__main__":
    run_fast_loop()