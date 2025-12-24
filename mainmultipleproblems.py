import jax
import jax.numpy as jnp
from jax import random, lax, jit
# 修改导入名称，移除 _jit 后缀
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
    """求最小值逻辑：直接返回函数值"""
    task_id = context['task_id']
    res = lax.cond(
        task_id == 0,
        lambda x: 1.0 * f1_styblinski(x) + 0.0 * f2_quadratic(x),
        lambda x: 0.0 * f1_styblinski(x) + 1.0 * f3_multi_goal(x),
        operand=z_combined
    )
    return res 

def run_parallel_benchmark():
    P = 4; M = 2; K = 20; D_max = 5
    T = 2000; DT = 0.1; B = 60; B0 = 25; T0 = 200
    dims = (5, 5)
    
    key = random.PRNGKey(42)
    keys_P = random.split(key, P)

    # 初始化 (P, M, K, D_max) 等
    initial_mu = random.uniform(keys_P[0], (P, M, K, D_max), minval=-4.0, maxval=4.0)
    initial_L_inv = jnp.tile(jnp.eye(D_max) * 1.5, (P, M, K, 1, 1))
    initial_v = jnp.zeros((P, M, K - 1))

    # Context 构造
    context_P = { 'task_id': jnp.array([0, 0, 1, 1]) }

    print(f">>> 启动并行优化器 (P={P}, 模式: 最小值优化)...")
    
    # 直接调用 parallel_mmog_igo_mpc
    # 内部会自动触发 mmog_igo_optimizer_mpc 的 JIT 编译
    mu_final, L_inv_final, pi_final = parallel_mmog_igo_mpc(
        keys_P, T, DT, M, K, B, B0, dims, T0, 
        fitness_dispatcher,
        initial_mu, initial_L_inv, initial_v, context_P
    )

    for p in range(P):
        best_x_blocks = [mu_final[p, m, jnp.argmax(pi_final[p, m])] for m in range(M)]
        full_x = jnp.concatenate(best_x_blocks)
        f_val = fitness_dispatcher(full_x, {'task_id': context_P['task_id'][p]})
        print(f"问题 {p} [Task {context_P['task_id'][p]}]: 最优值 f(x) = {f_val:.6f}")

if __name__ == "__main__":
    run_parallel_benchmark()