# gmm_igo/MPCsolvermultipleproblems.py
import jax
from jax import vmap
from gmm_igo.MPCsolverM2 import mmog_igo_optimizer_mpc

def parallel_mmog_igo_mpc(
    keys, T, dt, M, K, B, B0, dims, T_0, 
    fitness_fn_total, initial_mu, initial_L_inv, initial_v, context
):
    """
    并行求解多个问题。不再使用顶层 JIT 避免 static_argnums 偏移问题。
    """
    # 显式定义闭包，捕获标量/静态参数 (T, dt, M, K, B, B0, dims, T_0, fitness_fn_total)
    def single_solve(k, i_mu, i_L, i_v, ctx):
        return mmog_igo_optimizer_mpc(
            key=k,
            T=T,
            dt=dt,
            M=M,
            K=K,
            B=B,
            B0=B0,
            dims=dims,
            T_0=T_0,
            fitness_fn_total=fitness_fn_total,
            initial_mu_k=i_mu,
            initial_L_inv_k=i_L,
            initial_v_k=i_v,
            context=ctx
        )

    # 这里的 in_axes 分别对应 keys, initial_mu, initial_L_inv, initial_v, context 的批次维度
    solver_vmap = vmap(single_solve, in_axes=(0, 0, 0, 0, 0))

    return solver_vmap(keys, initial_mu, initial_L_inv, initial_v, context)