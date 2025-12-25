# solverM1.py - M 个混合高斯的信息几何优化器核心模块 (已集成修正 & 历史记录)

import jax
import jax.numpy as jnp
from jax import vmap, random, lax
import functools
import time 

# ----------------------------------------------------------------------
# I. 核心辅助函数 (信息矩阵 S_k 范式 - 从 solver.py 整合)
# ----------------------------------------------------------------------

@jax.jit
def _logsumexp(a, axis=None):
    """Numerically stable calculation of log(sum(exp(a)))."""
    return jnp.logaddexp.reduce(a, axis=axis)

@jax.jit
def _gaussian_log_pdf_l(xi, mu, L_inv):
    """计算 N(mu, (L_inv @ L_inv.T)^{-1}) 的对数概率密度。"""
    D = mu.shape[0] 
    diff = xi - mu
    
    y = L_inv @ diff
    mahalanobis_sq = jnp.sum(y**2)
    
    # log|Sigma| = -log|S| = -2 * log|L_inv|
    log_det_S_inv = -2 * jnp.sum(jnp.log(jnp.diag(L_inv)))
    
    log_pdf = -0.5 * (D * jnp.log(2 * jnp.pi) + log_det_S_inv + mahalanobis_sq)
    return log_pdf

_vmap_gaussian_log_pdf_l_k = vmap(_gaussian_log_pdf_l, in_axes=(None, 0, 0))

@jax.jit
def _mixture_log_pdf_l(xi, mu_k, L_inv_k, pi_k):
    """计算混合高斯 Log-PDF (信息矩阵范式)。"""
    log_pdfs_k = _vmap_gaussian_log_pdf_l_k(xi, mu_k, L_inv_k)
    log_pi_k = jnp.log(pi_k)
    log_weighted_pdfs = log_pi_k + log_pdfs_k
    return _logsumexp(log_weighted_pdfs)

@jax.jit
def _sample_from_component_l(idx, key_sample, mu_k_all, L_inv_k_all):
    """从 N(mu, S^{-1}) 中采样。"""
    mu_k = mu_k_all[idx]
    S_k = L_inv_k_all[idx] @ L_inv_k_all[idx].T # S_k 是信息矩阵
    
    # Cholesky 分解 S^{-1} 的协方差 L_Sigma
    L_Sigma = jnp.linalg.cholesky(jnp.linalg.inv(S_k))
    
    D = mu_k.shape[0]
    z = random.normal(key_sample, shape=(D,))
    return mu_k + L_Sigma @ z

# ----------------------------------------------------------------------
# II. K 个分量的并行更新逻辑 (从 solver.py 整合)
# ----------------------------------------------------------------------

# **基础修正:** 强制收紧 log alpha 的裁剪值 (用于数值稳定)
LOG_CLIP_VALUE = 10.0 

def _update_step_k_l_single_component(
    k_idx, mu_k_t, L_inv_k_t, samples, elite_weights, 
    pi_k_all, mu_k_all, L_inv_k_all, delta_t,
    mu_K_t, L_inv_K_t 
):
    """在信息矩阵 S_k 范式下，单个分量 k 的更新函数。"""
    S_k_t = L_inv_k_t @ L_inv_k_t.T # S_k 是信息矩阵
    D = mu_k_t.shape[0]
    
    # --- 1. Log(alpha) 和 Log(beta) ---
    vmap_log_pdf_k = vmap(_gaussian_log_pdf_l, in_axes=(0, None, None))
    log_norm_pdf_k = vmap_log_pdf_k(samples, mu_k_t, L_inv_k_t) 
    log_norm_pdf_K = vmap_log_pdf_k(samples, mu_K_t, L_inv_K_t)
    log_mog_xi = vmap(_mixture_log_pdf_l, in_axes=(0, None, None, None))(
        samples, mu_k_all, L_inv_k_all, pi_k_all
    )

    log_a_i = jnp.clip(log_norm_pdf_k - log_mog_xi, a_max=LOG_CLIP_VALUE)
    log_b_i = jnp.clip(log_norm_pdf_K - log_mog_xi, a_max=LOG_CLIP_VALUE)
    
    a_i = jnp.exp(log_a_i)
    b_i = jnp.exp(log_b_i)
    scaled_a_i = a_i * elite_weights
    
    # --- 2. S_k 更新 (信息矩阵 S_k 的自然梯度) ---
    diff = samples - mu_k_t
    diff_outer = vmap(lambda x: jnp.outer(x, x))(diff)
    
    Sigma_k_t = jnp.linalg.inv(S_k_t) # 协方差 Sigma_k = S_k^{-1}
    
    S_update_term_i = Sigma_k_t @ diff_outer @ Sigma_k_t - Sigma_k_t[None, :, :]
    
    sum_S_update = jnp.sum(scaled_a_i[:, None, None] * S_update_term_i, axis=0)
    
    S_k_t_plus_1_prop = S_k_t - delta_t * sum_S_update
    
    S_k_t_plus_1_prop = (S_k_t_plus_1_prop + S_k_t_plus_1_prop.T) / 2
    D_dim = S_k_t_plus_1_prop.shape[0]
    # **基础修正:** 强制提高正则化 (从 1e-6 提高到 1e-3 以避免奇异)
    S_k_t_plus_1_prop = S_k_t_plus_1_prop + jnp.eye(D_dim) * 1e-12 
    
    L_inv_k_t_plus_1 = jnp.linalg.cholesky(S_k_t_plus_1_prop)
    S_k_t_plus_1 = L_inv_k_t_plus_1 @ L_inv_k_t_plus_1.T


    # --- 3. mu_k 更新 ---
    weighted_diff = scaled_a_i[None, :] * diff.T 
    S_t_weighted_diff = S_k_t @ weighted_diff
    sum_term_vector = jnp.sum(S_t_weighted_diff, axis=1) # D 维向量

    mu_update_term = jnp.linalg.solve(S_k_t_plus_1, sum_term_vector)
    mu_k_t_plus_1 = mu_k_t + delta_t * mu_update_term

    # --- 4. 权重更新项 (v_k) ---
    v_update_sum = jnp.sum(elite_weights * (a_i - b_i))
    
    return mu_k_t_plus_1, L_inv_k_t_plus_1, v_update_sum

_vmap_update_step_k_l = vmap(
    _update_step_k_l_single_component, 
    in_axes=(0, 0, 0, None, None, None, None, None, None, None, None), 
    out_axes=(0, 0, 0)
)


# ----------------------------------------------------------------------
# III. M-MoG 并行采样和更新逻辑 (从 solverM1.py 整合)
# ----------------------------------------------------------------------

def _sample_from_mog_batch(key_m, mu_k_m, L_inv_k_m, pi_k_all_m, B):
    """从单个 MoG 分布中采样 B 个样本 (修复 JAX Key 逻辑)。"""
    K = mu_k_m.shape[0]
    
    key_comp, key_samples = random.split(key_m)
    comp_indices = random.choice(key_comp, K, shape=(B,), p=pi_k_all_m) 
    sample_keys = random.split(key_samples, B) 
    
    vmap_sample_fn = vmap(_sample_from_component_l, in_axes=(0, 0, None, None))
    samples_m = vmap_sample_fn(comp_indices, sample_keys, mu_k_m, L_inv_k_m)
    return samples_m

_vmap_sample_from_mog_batch = vmap(
    _sample_from_mog_batch, 
    in_axes=(0, 0, 0, 0, None), 
    out_axes=0
) 

def _get_overall_elite_weights(samples_M, fitness_fn_total, B, B_0):
    """评估 B 个整体样本 f(xi_1, ..., xi_B) 并计算精英权重。"""
    samples_overall = jnp.transpose(samples_M, (1, 0, 2)).reshape((B, -1))
    f_xi = vmap(fitness_fn_total)(samples_overall) 
    ranks = jnp.argsort(jnp.argsort(f_xi))
    is_elite = ranks < B_0
    elite_weights = jnp.where(is_elite, 1.0, 0.0) / B 
    return elite_weights

def _update_step_m_l_single_mog(
    k_indices, mu_k_t_m, L_inv_k_t_m, samples_m, elite_weights, 
    pi_k_all_m, mu_k_all_m, L_inv_k_all_m, delta_t,
    mu_K_t_m, L_inv_K_t_m 
):
    """单个 MoG 的完整更新步骤。"""
    mu_k_t_plus_1_m, L_inv_k_t_plus_1_m, v_update_sum_k_m = _vmap_update_step_k_l(
        k_indices, mu_k_t_m, L_inv_k_t_m, samples_m, elite_weights, 
        pi_k_all_m, mu_k_all_m, L_inv_k_all_m, delta_t,
        mu_K_t_m, L_inv_K_t_m
    )
    return mu_k_t_plus_1_m, L_inv_k_t_plus_1_m, v_update_sum_k_m
    
_vmap_update_step_m_l = vmap(
    _update_step_m_l_single_mog, 
    in_axes=(None, 0, 0, 0, None, 0, 0, 0, None, 0, 0), 
    out_axes=(0, 0, 0)
) 

# ----------------------------------------------------------------------
# IV. 完整的 M-MoG 迭代步 (JAX Scan/Loop) - 新增历史记录追踪
# ----------------------------------------------------------------------

def _mmog_iteration_step(state, key_input, M, K, B, B_0, delta_t, fitness_fn_total):
    """一个完整的 M-MoG IGO 迭代步，返回状态以供追踪。"""
    
    mu_k_t, L_inv_k_t, v_k_t = state 
    key, subkey = random.split(key_input)
    
    # 1. 计算 M 个 MoG 的 K 个权重 pi_k_t_all (M 维并行)
    pi_k_pre = jnp.exp(v_k_t) 
    pi_K_t = 1 / (1 + jnp.sum(pi_k_pre, axis=1, keepdims=True)) 
    pi_k_all_m = jnp.concatenate([pi_k_pre * pi_K_t, pi_K_t], axis=1) 
    
    # 2. 采样 B 个样本 (M 维并行)
    key_sample_M = random.split(subkey, M) 
    samples_M = _vmap_sample_from_mog_batch(
        key_sample_M, mu_k_t, L_inv_k_t, pi_k_all_m, B
    ) 

    # 3. 整体精英选择
    elite_weights = _get_overall_elite_weights(
        samples_M, fitness_fn_total, B, B_0
    ) 
    
    # 4. M 个 MoG 并行更新
    k_indices = jnp.arange(K) 
    mu_K_t, L_inv_K_t = mu_k_t[:, -1], L_inv_k_t[:, -1] 
    
    mu_k_t_plus_1, L_inv_k_t_plus_1, v_update_sum_k_M = _vmap_update_step_m_l(
        k_indices, mu_k_t, L_inv_k_t, samples_M, elite_weights, 
        pi_k_all_m, mu_k_t, L_inv_k_t, delta_t,
        mu_K_t, L_inv_K_t 
    ) 
    
    # 5. 权重更新
    v_update_vec = v_update_sum_k_M[:, :K-1] 
    
    MAX_V_UPDATE = 5.0
    MAX_V_K = 10.0 
    v_update_norm = jnp.linalg.norm(v_update_vec, axis=1, keepdims=True) 
    
    v_update_safe = jnp.where(
        v_update_norm > MAX_V_UPDATE,
        v_update_vec * (MAX_V_UPDATE / v_update_norm),
        v_update_vec
    )
    
    v_k_t_plus_1 = v_k_t + delta_t * v_update_safe
    v_k_t_plus_1 = jnp.clip(v_k_t_plus_1, a_max=MAX_V_K) 
    
    new_state = (mu_k_t_plus_1, L_inv_k_t_plus_1, v_k_t_plus_1)
    
    # 关键修改：返回 new_state 两次，以便 lax.scan 收集历史
    return new_state, new_state


# ----------------------------------------------------------------------
# V. 主优化器函数 (JIT 编译) - 新增历史记录返回
# ----------------------------------------------------------------------

def mmog_igo_optimizer_impl(
    key, T, delta_t, M, K, B, B_0, fitness_fn_total,
    initial_mu_k, initial_L_inv_k, initial_v_k
):
    """M-MoG IGO 优化器主逻辑，返回 L_inv_k 的历史记录。"""
    
    initial_state = (initial_mu_k, initial_L_inv_k, initial_v_k)
    
    bound_iteration_step = functools.partial(
        _mmog_iteration_step, 
        M=M, K=K, B=B, B_0=B_0, delta_t=delta_t, fitness_fn_total=fitness_fn_total
    )

    keys_iter = random.split(key, T)
    # lax.scan 收集历史
    final_state, history = lax.scan(bound_iteration_step, initial_state, keys_iter)

    history_mu_k, history_L_inv_k, history_v_k = history 

    final_mu_k, final_L_inv_k, final_v_k = final_state
    
    final_pi_k_pre = jnp.exp(final_v_k) 
    final_pi_K = 1 / (1 + jnp.sum(final_pi_k_pre, axis=1, keepdims=True)) 
    final_pi_k_all = jnp.concatenate([final_pi_k_pre * final_pi_K, final_pi_K], axis=1) 
    
    # 返回 L_inv 的历史
    return final_mu_k, final_L_inv_k, final_pi_k_all, history_L_inv_k

# JIT 编译整个优化器
mmog_igo_optimizer = jax.jit(
    mmog_igo_optimizer_impl, 
    static_argnames=('T', 'delta_t', 'M', 'K', 'B', 'B_0', 'fitness_fn_total')
)