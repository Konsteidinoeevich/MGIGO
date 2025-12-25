import jax
from jax import vmap
from gmm_igo.MPCsolverM2 import mmog_igo_optimizer_mpc

# 这是一个纯净的包装器，不带 jit，由底层 mmog_igo_optimizer_mpc 自带的 jit 保证速度
def parallel_mmog_igo_mpc(
    keys, T, dt, M, K, B, B0, dims, T_0, 
    fitness_fn_total, initial_mu, initial_L_inv, initial_v, context
):
    # 直接对带 jit 的底座函数进行 vmap
    # None 代表该参数对所有 P 个任务共享（广播），0 代表该参数随 P 变化（并行）
    solver_batch = vmap(
        mmog_igo_optimizer_mpc,
        in_axes=(0, None, None, None, None, None, None, None, None, None, 0, 0, 0, 0)
    )

    return solver_batch(
        keys, T, dt, M, K, B, B0, dims, T_0, 
        fitness_fn_total, initial_mu, initial_L_inv, initial_v, context
    )