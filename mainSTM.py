import jax
import jax.numpy as jnp
from jax import random
import time

# 导入你修改后的 MPC 求解器
from gmm_igo.MPCsolverM2 import mmog_igo_optimizer_mpc

# ======================================================================
# 1. 适应度函数 (参考 mainST.py, 但适配多块展平输入)
# ======================================================================
@jax.jit
def mpc_styblinski_tang_fitness(z_flattened, context):
    """
    Styblinski-Tang 函数。
    z_flattened: 所有块拼接后的展平向量 (B, M * D_MAX)
    """
    # 在实际 MPC 中，z_flattened 包含了所有 block 的填充数据
    # 这里简单地对整个向量求和，或者根据 dims 还原后求和
    term = z_flattened**4 - 16.0 * z_flattened**2 + 5.0 * z_flattened 
    return jnp.sum(term)

# ======================================================================
# 2. 初始化逻辑 (参考 mainST.py 的 initialize_params)
# ======================================================================
def init_mpc_params(key, M, K, D_MAX):
    key_mu, key_L = random.split(key)
    
    # 初始化均值 mu: (M, K, D_MAX)
    initial_mu = random.uniform(key_mu, (M, K, D_MAX), minval=-4.0, maxval=4.0)
    
    # 初始化 Cholesky 因子 L_inv (精度矩阵的平方根): (M, K, D_MAX, D_MAX)
    # 使用单位阵乘以根号 2，对齐 mainST.py 的逻辑
    L_template = jnp.eye(D_MAX) * jnp.sqrt(2.0)
    initial_L_inv = jnp.tile(L_template, (M, K, 1, 1))
    
    # 初始化混合权重对数项 v: (M, K-1)
    initial_v = jnp.zeros((M, K - 1))
    
    return initial_mu, initial_L_inv, initial_v

# ======================================================================
# 3. 执行主程序
# ======================================================================
def run_example():
    # --- 参数配置 ---
    SEED = 42
    T_RUN = 10000        # 迭代轮数
    DELTA_T = 0.1     # 步长
    M_BLOCKS = 3       # 块数量 (N)
    K_COMP = 10        # 每个块的分量数 (K)
    DIMS_TUPLE = (4, 4, 4) # 每个块的实际有效维度
    D_MAX = max(DIMS_TUPLE)
    
    B_SAMPLES = 60    # 样本数 (B)
    B_0_ELITE = 25     # 精英样本数 (B0)
    T_0_RESTART = 1000  # 重置周期 (T0)
    
    key = random.PRNGKey(SEED)
    key_init, key_solve = random.split(key)

    # --- 初始化 ---
    print(f"正在初始化: {M_BLOCKS} 块, 每块 {K_COMP} 分量, 最大维度 {D_MAX}...")
    init_mu, init_L, init_v = init_mpc_params(key_init, M_BLOCKS, K_COMP, D_MAX)
    
    # 模拟 context (即使函数内没用到也需要透传)
    current_context = jnp.array([0.0]) 

    # --- 调用 MPC 求解器 ---
    print("开始 IGO 优化循环...")
    start_t = time.perf_counter()
    
    final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
        key=key_solve,
        T=T_RUN,
        dt=DELTA_T,
        M=M_BLOCKS,
        K=K_COMP,
        B=B_SAMPLES,
        B0=B_0_ELITE,
        dims=DIMS_TUPLE,
        T_0=T_0_RESTART,
        fitness_fn_total=mpc_styblinski_tang_fitness,
        initial_mu_k=init_mu,
        initial_L_inv_k=init_L,
        initial_v_k=init_v,
        context=current_context
    )
    
    # 强制同步以准确计时
    final_mu.block_until_ready()
    duration = time.perf_counter() - start_t

    # --- 结果展示 ---
    print(f"\n优化完成！耗时: {duration:.4f} 秒")
    
    # 计算理论目标值 (Styblinski-Tang 全局最优约为 -39.16 * 总维度)
    total_dim = sum(DIMS_TUPLE)
    target_f = -39.166 * total_dim
    
    # 找到所有块中概率最大的分量
    best_pi_idx = jnp.argmax(final_pi, axis=1)
    
    print("-" * 50)
    print(f"理论全局最优值应接近: {target_f:.2f}")
    for m in range(M_BLOCKS):
        idx = best_pi_idx[m]
        mu_best = final_mu[m, idx, :DIMS_TUPLE[m]]
        print(f"块 {m} [最佳分量 {idx}]:")
        print(f"  权重 pi: {final_pi[m, idx]:.4f}")
        print(f"  均值 mu (前4维): {mu_best[:4]}")

if __name__ == "__main__":
    run_example()