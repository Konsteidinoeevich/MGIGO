import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import time 

# 导入已经根据 Algorithm 3 严格修改后的求解器
# 假设新求解器保存在 gmm_igo/MPCsolverM2.py 中
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer

# ======================================================================
# 1. 目标函数 (对齐 Algorithm 3 的最小化倾向)
# ======================================================================
@jit
def multi_goal_fitness_fn(z_flattened, context):
    """
    多目标点优化函数。
    Algorithm 3 默认按 f(z) 递增排序（即最小化问题）。
    """
    D_MAX = 8 
    
    # 还原各块的有效维度
    x1 = z_flattened[0:2]
    x2 = z_flattened[D_MAX : D_MAX+5]
    x3 = z_flattened[2*D_MAX : 2*D_MAX+8]
    x = jnp.concatenate([x1, x2, x3])
    
    # 定义目标点
    dist_to_0 = x**2
    dist_to_p3 = (x - 3.0)**2
    dist_to_m3 = (x + 3.0)**2
    
    # 找到每个维度距离最近的目标点的平方距离
    min_dist_sq = jnp.min(jnp.stack([dist_to_0, dist_to_p3, dist_to_m3]), axis=0)
    
    # 返回正的距离之和。根据算法第18行，值越小排名越靠前（权重越高）
    return jnp.sum(min_dist_sq)

# ======================================================================
# 2. 循环求解主程序
# ======================================================================
def run_loop_test():
    # --- 参数配置 (参考 Algorithm 3 给定的变量) ---
    SEED = 42
    T_RUN = 1000       # T: 迭代总步数
    DELTA_T = 0.1     # alpha_t: 学习率/步长
    M_MOG = 3          # N: 块数量
    K_COMP = 10        # K: 混合分量数
    DIMS_TUPLE = (2, 5, 8) 
    B_SAMPLES = 60     # B: 样本大小
    B_0_ELITE = 20     # B0: 选择的精英样本数
    T_0_RESTART = 100  # T0: 权重重置周期
    NUM_SOLVES = 5     
    
    D_MAX = max(DIMS_TUPLE)
    key = random.PRNGKey(SEED)

    # --- 初始状态设置 (Algorithm 3 步骤 2) ---
    # 初始化均值 mu
    init_mu = random.uniform(key, (M_MOG, K_COMP, D_MAX), minval=-4.0, maxval=4.0)
    # 初始化 Cholesky 因子 L (满足 S = LL^T)
    # 注意：算法中使用 L_inv 进行计算以提高效率
    init_L_inv = jnp.tile(jnp.eye(D_MAX) * jnp.sqrt(4.0), (M_MOG, K_COMP, 1, 1))
    # 初始化权重 logits (全 0 对应均匀分布 1/K)
    init_v = jnp.zeros((M_MOG, K_COMP)) 
    
    current_context = jnp.array([0.0, 1.0]) 

    print(f"--- 严格遵循 Algorithm 3 开始求解 (Solves={NUM_SOLVES}) ---")

    for i in range(NUM_SOLVES):
        key, subkey = random.split(key)
        start_time = time.time()
        
        # 调用严格对齐算法的求解器
        final_mu, final_L, final_pi = mmog_igo_optimizer(
            key=subkey, 
            T=T_RUN, 
            dt=DELTA_T, 
            M=M_MOG, 
            K=K_COMP, 
            B=B_SAMPLES, 
            B0=B_0_ELITE, 
            dims=DIMS_TUPLE, 
            T_0=T_0_RESTART, 
            fitness_fn_total=multi_goal_fitness_fn,
            initial_mu_k=init_mu, 
            initial_L_inv_k=init_L_inv, 
            initial_v_k=init_v,
            context=current_context 
        )
        
        final_mu.block_until_ready()
        elapsed = time.time() - start_time
        
        # 结果提取：选择权重最大的分量作为代表解
        best_comp_indices = jnp.argmax(final_pi, axis=1) # [M]
        
        res_summary = []
        for m in range(M_MOG):
            m_best_idx = best_comp_indices[m]
            res_summary.append(final_mu[m, m_best_idx, :DIMS_TUPLE[m]])
        
        print(f"第 {i+1} 次求解耗时: {elapsed:.4f}s")
        print(f"  块1(dim=2) 结果: {res_summary[0]}")
        print(f"  块3(dim=8) 结果样例: {res_summary[2][:3]}...")

if __name__ == "__main__":
    run_loop_test()