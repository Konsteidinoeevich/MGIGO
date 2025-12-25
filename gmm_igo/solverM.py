import jax
import jax.numpy as jnp
from jax import vmap, random, lax
import functools
from gmm_igo.solver import  _sample_from_component_l

# 从 MoG 中采样 B 个样本：输出形状 (B, D)
# B_sampler 是 (B,) 形状的随机数 key 数组
# vmap(fn, in_axes=(0, 0, None, None)) 在 M 维上并行
# 这里的 _sample_from_component_l 已经是 vmap 过 K 的版本，但我们将其视为一个 MoG 的单次采样。

# 假设我们已将 solver.py 中的所有辅助函数 (如 _logsumexp, _gaussian_log_pdf_l, _mixture_log_pdf_l, _sample_from_component_l, _get_elite_weights) 引入并做轻微修改以适应批处理。

# --- 2. 单个 MoG 的并行采样函数 (M 维并行) ---

# @functools.partial(vmap, in_axes=(0, 0, 0, 0), out_axes=0)
def _sample_from_mog_batch(key_m, mu_k_m, L_inv_k_m, pi_k_all_m, B):
    """
    从单个 MoG 分布中采样 B 个样本。
    - key_m: (B,) 形状的 key 数组
    - mu_k_m: (K, D), L_inv_k_m: (K, D, D), pi_k_all_m: (K,)
    - 返回: (B, D) 样本矩阵
    """
    K = mu_k_m.shape[0]
    # 随机选择 B 个分量索引
    comp_indices = random.choice(key_m[0], K, shape=(B,), p=pi_k_all_m)
    sample_keys = key_m # (B,) keys
    
    # _sample_from_component_l 已经定义在 solver.py 中
    vmap_sample_fn = vmap(_sample_from_component_l, in_axes=(0, 0, None, None))
    # 注意: L_inv_k_m 在这里是 (K, D, D)，mu_k_m 是 (K, D)
    samples_m = vmap_sample_fn(comp_indices, sample_keys, mu_k_m, L_inv_k_m)
    return samples_m

# vmap M 个 MoG 的采样 (M 维并行)
_vmap_sample_from_mog_batch = vmap(
    _sample_from_mog_batch, 
    in_axes=(0, 0, 0, 0, None), 
    out_axes=0
) # 输出: (M, B, D)

# --- 3. 跨 M-MoG 的整体目标函数评估和精英选择 ---

def _get_overall_elite_weights(samples_M, fitness_fn_total, B, B_0):
    """
    评估 f(xi_1, ..., xi_B) 并计算整体精英样本权重。
    
    - samples_M: (M, B, D)。
    - fitness_fn_total: 函数 f(xi_1, ..., xi_B) -> 标量。
    
    根据您的描述，目标函数是关于 B 个样本的整体函数。
    由于 JAX 的限制，我们假设目标函数可以接受 (B, M, D) 形式的样本并返回 B 个标量，
    或者，为了遵循典型的 CE/IGO 流程，假设目标函数对每个样本 \xi_i 有一个适应度值 f(\xi_i)，
    然后对这些适应度值进行排序。
    
    根据您第 3/4 条描述：
    3. 将 (\xi_{k,1}, ..., \xi_{k,M}) 打包为总体 \xi 的第 k 个分量。因此 \xi 也是有 B 个。
    4. 计算目标函数 f(\xi_1, ..., \xi_B) 并排序。
    
    这表明：
    - B 个样本是 \boldsymbol{\xi}_1, \dots, \boldsymbol{\xi}_B，其中 $\boldsymbol{\xi}_i = (\xi_{i,1}, \dots, \xi_{i,M})$。
    - $\xi_{i,m}$ 是第 $m$ 个 MoG 贡献的第 $i$ 个分量 (形状 D)。
    - 整体样本 $\boldsymbol{\xi}_i$ 的形状是 $(M \cdot D)$。
    
    因此，我们需要将 samples_M (M, B, D) 转换为 (B, M*D) 的 B 个整体样本。
    """
    
    # 转换: (M, B, D) -> (B, M, D) -> (B, M*D)
    # 整体样本: (B, M*D)
    samples_overall = jnp.transpose(samples_M, (1, 0, 2))
    samples_overall = samples_overall.reshape((B, -1))

    # 评估 B 个整体样本的适应度 f(\boldsymbol{\xi}_i)
    # 我们假设 f(\boldsymbol{\xi}_i) 是可 vmap 的，或者您定义了一个评估所有 B 个样本的函数
    f_xi = vmap(fitness_fn_total)(samples_overall) # (B,)
    
    # 排序和精英选择 (与 solver.py 相同)
    ranks = jnp.argsort(jnp.argsort(f_xi))
    is_elite = ranks < B_0
    
    # 精英权重 $\omega_i = I_{elite} / B$ (形状 B)
    elite_weights = jnp.where(is_elite, 1.0, 0.0) / B 
    
    return elite_weights

# --- 4. 单个 MoG 的更新逻辑 (基于 solver.py 的并行化) ---

# _update_step_k_l_single_component (来自 solver.py) 已经是单分量 k 的更新，
# 并且 vmap 了 B 个样本 i: _vmap_update_step_k_l (来自 solver.py) 已经 vmap K 个分量 k。

def _update_step_m_l_single_mog(
    k_indices, mu_k_t_m, L_inv_k_t_m, samples_m, elite_weights, 
    pi_k_all_m, mu_k_all_m, L_inv_k_all_m, delta_t,
    mu_K_t_m, L_inv_K_t_m 
):
    """
    单个 MoG 的完整更新步骤 (K 维并行)。
    注意：samples_m 是 (B, D)，elite_weights 是 (B,)
    """
    # 使用 solver.py 中定义的 K 维并行函数
    mu_k_t_plus_1_m, L_inv_k_t_plus_1_m, v_update_sum_k_m = _vmap_update_step_k_l(
        k_indices, mu_k_t_m, L_inv_k_t_m, samples_m, elite_weights, 
        pi_k_all_m, mu_k_all_m, L_inv_k_all_m, delta_t,
        mu_K_t_m, L_inv_K_t_m
    )
    return mu_k_t_plus_1_m, L_inv_k_t_plus_1_m, v_update_sum_k_m
    
# vmap M 个 MoG 的更新 (M 维并行)
_vmap_update_step_m_l = vmap(
    _update_step_m_l_single_mog, 
    in_axes=(None, 0, 0, 0, None, 0, 0, 0, None, 0, 0), # elite_weights 共享
    out_axes=(0, 0, 0)
) # 输出: (M, K, D), (M, K, D, D), (M, K)

# --- 5. 完整的 M-MoG 迭代步 ---

def _mmog_iteration_step(state, key_input, M, K, B, B_0, delta_t, fitness_fn_total):
    """一个完整的 M-MoG IGO 迭代步。"""
    
    mu_k_t, L_inv_k_t, v_k_t = state # (M, K, D), (M, K, D, D), (M, K-1)
    key, subkey = random.split(key_input)
    
    # 1. 计算 M 个 MoG 的 K 个权重 pi_k_t_all
    pi_k_pre = jnp.exp(v_k_t) # (M, K-1)
    pi_K_t = 1 / (1 + jnp.sum(pi_k_pre, axis=1, keepdims=True)) # (M, 1)
    pi_k_all_m = jnp.concatenate([pi_k_pre * pi_K_t, pi_K_t], axis=1) # (M, K)
    
    # 2. 从 M 个 MoG 中采样 B 个样本 (M 维并行)
    # 为 M*B 次采样准备随机数 key
    key_sample_M = random.split(subkey, M * B).reshape((M, B, 2))
    
    # M 个 MoG 的 K 个参数和 K 个权重 (M, K, D), (M, K, D, D), (M, K)
    samples_M = _vmap_sample_from_mog_batch(
        key_sample_M, mu_k_t, L_inv_k_t, pi_k_all_m, B
    ) # (M, B, D)

    # 3. 评估 f(xi_1, ..., xi_B) 并计算整体精英样本权重
    elite_weights = _get_overall_elite_weights(
        samples_M, fitness_fn_total, B, B_0
    ) # (B,)
    
    # 4. M 个 MoG 并行更新
    k_indices = jnp.arange(K) 
    mu_K_t, L_inv_K_t = mu_k_t[:, -1], L_inv_k_t[:, -1] # (M, D), (M, D, D)
    
    mu_k_t_plus_1, L_inv_k_t_plus_1, v_update_sum_k_M = _vmap_update_step_m_l(
        k_indices, mu_k_t, L_inv_k_t, samples_M, elite_weights, 
        pi_k_all_m, mu_k_t, L_inv_k_t, delta_t,
        mu_K_t, L_inv_K_t 
    ) # (M, K, D), (M, K, D, D), (M, K)
    
    # 5. 权重更新 (M 维并行)
    v_update_vec = v_update_sum_k_M[:, :K-1] # (M, K-1)
    
    # 应用裁剪 (MAX_V_UPDATE 和 MAX_V_K 假定为全局常量)
    MAX_V_UPDATE = 10.0
    v_update_norm = jnp.linalg.norm(v_update_vec, axis=1, keepdims=True) # (M, 1)
    
    v_update_safe = jnp.where(
        v_update_norm > MAX_V_UPDATE,
        v_update_vec * (MAX_V_UPDATE / v_update_norm),
        v_update_vec
    )
    
    v_k_t_plus_1 = v_k_t + delta_t * v_update_safe
    MAX_V_K = 70.0 
    v_k_t_plus_1 = jnp.clip(v_k_t_plus_1, a_max=MAX_V_K) # (M, K-1)
    
    new_state = (mu_k_t_plus_1, L_inv_k_t_plus_1, v_k_t_plus_1)
    
    return new_state, None

# --- 6. 主优化器函数 ---

def mmog_igo_optimizer_impl(
    key, T, delta_t, M, K, B, B_0, fitness_fn_total,
    initial_mu_k, initial_L_inv_k, initial_v_k
):
    """
    M-MoG IGO 优化器主逻辑。
    - initial_mu_k: (M, K, D)
    - initial_L_inv_k: (M, K, D, D)
    - initial_v_k: (M, K-1)
    """
    
        
    initial_state = (initial_mu_k, initial_L_inv_k, initial_v_k)
    
    bound_iteration_step = functools.partial(
        _mmog_iteration_step, 
        M=M, K=K, B=B, B_0=B_0, delta_t=delta_t, fitness_fn_total=fitness_fn_total
    )

    keys_iter = random.split(key, T)
    final_state, _ = lax.scan(bound_iteration_step, initial_state, keys_iter)

    final_mu_k, final_L_inv_k, final_v_k = final_state
    
    # 最终权重转换
    final_pi_k_pre = jnp.exp(final_v_k) # (M, K-1)
    final_pi_K = 1 / (1 + jnp.sum(final_pi_k_pre, axis=1, keepdims=True)) # (M, 1)
    final_pi_k_all = jnp.concatenate([final_pi_k_pre * final_pi_K, final_pi_K], axis=1) # (M, K)
    
    return final_mu_k, final_L_inv_k, final_pi_k_all

mmog_igo_optimizer = jax.jit(
    mmog_igo_optimizer_impl, 
    static_argnames=('T', 'delta_t', 'M', 'K', 'B', 'B_0', 'fitness_fn_total')
)