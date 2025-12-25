# gmm_igo/solver.py - 混合高斯信息几何优化器核心模块 (仅新增 T_0 权重重置逻辑)

import jax
import jax.numpy as jnp
from jax import vmap, random, lax
import functools

# ----------------------------------------------------------------------
# I. 核心辅助函数 (信息矩阵 S_k 范式)
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

def _sample_from_component_l(idx, key_sample, mu_k_all, L_inv_k_all):
    """从 N(mu, S^{-1}) 中采样。"""
    mu_k = mu_k_all[idx]
    S_k = L_inv_k_all[idx] @ L_inv_k_all[idx].T # S_k 是信息矩阵
    
    # 协方差 Sigma = S^{-1}
    Sigma_k = jnp.linalg.inv(S_k)
    
    # Cholesky 分解协方差矩阵 L_Sigma
    L_Sigma = jnp.linalg.cholesky(Sigma_k)
    
    D = mu_k.shape[0]
    z = random.normal(key_sample, shape=(D,))
    return mu_k + L_Sigma @ z

def _get_elite_weights(samples, fitness_fn, B, B_0):
    """评估 f(xi) 并计算精英样本权重 (I_elite / B)。"""
    f_xi = vmap(fitness_fn)(samples)
    ranks = jnp.argsort(jnp.argsort(f_xi))
    is_elite = ranks < B_0
    return jnp.where(is_elite, 1.0, 0.0) / B 

# ----------------------------------------------------------------------
# II. K 个分量的并行更新逻辑 (信息矩阵 S_k 范式)
# ----------------------------------------------------------------------

LOG_CLIP_VALUE = 80.0 

def _update_step_k_l_single_component(
    k_idx, mu_k_t, L_inv_k_t, samples, elite_weights, 
    pi_k_all, mu_k_all, L_inv_k_all, delta_t,
    mu_K_t, L_inv_K_t 
):
    """在信息矩阵 S_k 范式下，单个分量 k 的更新函数。"""
    S_k_t = L_inv_k_t @ L_inv_k_t.T # S_k 是信息矩阵
    D = mu_k_t.shape[0]
    
    # --- 1. Log(alpha) 和 Log(beta) (使用 L_inv) ---
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
    
    S_k_t_plus_1_prop = S_k_t - delta_t * sum_S_update # S_k 是信息矩阵
    
    S_k_t_plus_1_prop = (S_k_t_plus_1_prop + S_k_t_plus_1_prop.T) / 2
    D_dim = S_k_t_plus_1_prop.shape[0]
    
    # 🌟 恢复：S_k 正则化为 1e-6 (不改变您的算法数值)
    REGULARIZATION_JITTER = 1e-6 
    S_k_t_plus_1_prop = S_k_t_plus_1_prop + jnp.eye(D_dim) * REGULARIZATION_JITTER 
    
    # 更新 L_inv
    L_inv_k_t_plus_1 = jnp.linalg.cholesky(S_k_t_plus_1_prop)
    S_k_t_plus_1 = L_inv_k_t_plus_1 @ L_inv_k_t_plus_1.T


    # --- 3. mu_k 更新 ---
    
    # 1. Sum = \sum_{i} \omega_i \alpha_i^k \cdot S_{k, t} (\mathbf{\xi}_i - \mu_{k, t})
    weighted_diff = scaled_a_i[None, :] * diff.T 
    
    S_t_weighted_diff = S_k_t @ weighted_diff
    sum_term_vector = jnp.sum(S_t_weighted_diff, axis=1) # D 维向量

    # 2. mu\_update\_term = S_{k, t+1}^{-1} \cdot Sum
    mu_update_term = jnp.linalg.solve(S_k_t_plus_1, sum_term_vector)

    # 3. 应用更新
    mu_k_t_plus_1 = mu_k_t + delta_t * mu_update_term

    # --- 4. 权重更新项 (v_k) ---
    v_update_sum = jnp.sum(elite_weights * (a_i - b_i))
    
    return mu_k_t_plus_1, L_inv_k_t_plus_1, v_update_sum

# ----------------------------------------------------------------------
# III. 迭代步和主优化器函数 (新增权重重置逻辑)
# ----------------------------------------------------------------------

_vmap_update_step_k_l = vmap(
    _update_step_k_l_single_component, 
    in_axes=(0, 0, 0, None, None, None, None, None, None, None, None), 
    out_axes=(0, 0, 0)
)

def _iteration_step(state, iter_data, B, B_0, K, delta_t, fitness_fn, T_0):
    """一个完整的 IGO-MoG 迭代步，用于 lax.scan。"""
    mu_k_t, L_inv_k_t, v_k_t = state
    key_input, iteration_idx = iter_data
    key, subkey = random.split(key_input)
    
    # -----------------------------------------------------------------
    # 🌟 新增：权重重置逻辑 (Diversity Maintenance)
    # -----------------------------------------------------------------
    is_restart_step = (iteration_idx % T_0) == 0
    # v_k = log(pi_k / pi_K)，均匀权重时 v_k=0
    v_k_reset = jnp.zeros(K - 1) 
    
    # 如果是重置步，将 v_k_t 强制设置为 v_k_reset (即均匀权重)，同时不改变 mu 和 L_inv
    v_k_t_safe = jnp.where(is_restart_step, v_k_reset, v_k_t)
    # -----------------------------------------------------------------

    # 1. 计算 K 个权重 pi_k_t_all
    pi_k_pre = jnp.exp(v_k_t_safe)
    pi_K_t = 1 / (1 + jnp.sum(pi_k_pre))
    pi_k_t_all = jnp.concatenate([pi_k_pre * pi_K_t, jnp.array([pi_K_t])])
    
    # 2. 从 MoG 分布中取样 B 个样本
    comp_indices = random.choice(subkey, K, shape=(B,), p=pi_k_t_all)
    sample_keys = random.split(subkey, B)
    vmap_sample_fn = vmap(_sample_from_component_l, in_axes=(0, 0, None, None))
    samples = vmap_sample_fn(comp_indices, sample_keys, mu_k_t, L_inv_k_t)

    # 3. 评估 f(xi) 并计算精英样本权重
    elite_weights = _get_elite_weights(samples, fitness_fn, B, B_0)
    
    # 4. K 个分量并行更新 (均值和方差)
    k_indices = jnp.arange(K) 
    mu_K_t, L_inv_K_t = mu_k_t[-1], L_inv_k_t[-1]
    
    mu_k_t_plus_1, L_inv_k_t_plus_1, v_update_sum_k = _vmap_update_step_k_l(
        k_indices, mu_k_t, L_inv_k_t, samples, elite_weights, 
        pi_k_t_all, mu_k_t, L_inv_k_t, delta_t,
        mu_K_t, L_inv_K_t 
    )
    
    # 5. 权重更新 (集成 v_k 裁剪)
    v_update_vec = v_update_sum_k[:K-1]
    MAX_V_UPDATE = 10.0 
    v_update_norm = jnp.linalg.norm(v_update_vec)
    
    v_update_safe = jnp.where(
        v_update_norm > MAX_V_UPDATE,
        v_update_vec * (MAX_V_UPDATE / v_update_norm),
        v_update_vec
    )
    
    v_k_t_plus_1 = v_k_t_safe + delta_t * v_update_safe # 施加更新时使用 v_k_t_safe
    MAX_V_K = 70.0 
    v_k_t_plus_1 = jnp.clip(v_k_t_plus_1, a_max=MAX_V_K)
    
    new_state = (mu_k_t_plus_1, L_inv_k_t_plus_1, v_k_t_plus_1)
    
    return new_state, None

def igo_mog_optimizer_impl(
    key, T, delta_t, K, B, B_0, fitness_fn, T_0, 
    initial_mu_k, initial_L_inv_k, initial_pi_k
):
    """主实现逻辑。"""
    
    if K < 2:
        raise ValueError("K must be 2 or greater for Mixture of Gaussians IGO.")
        
    pi_K_0 = initial_pi_k[-1]
    pi_k_0_pre = initial_pi_k[:-1]
    v_k_0 = jnp.log(pi_k_0_pre / pi_K_0)
        
    initial_state = (initial_mu_k, initial_L_inv_k, v_k_0)
    
    bound_iteration_step = functools.partial(
        _iteration_step, 
        B=B, B_0=B_0, K=K, delta_t=delta_t, fitness_fn=fitness_fn, T_0=T_0
    )

    keys_iter = random.split(key, T)
    iter_indices = jnp.arange(T)
    iter_data = (keys_iter, iter_indices)
    
    final_state, _ = lax.scan(bound_iteration_step, initial_state, iter_data)

    final_mu_k, final_L_inv_k, final_v_k = final_state
    
    final_pi_k_pre = jnp.exp(final_v_k)
    final_pi_K = 1 / (1 + jnp.sum(final_pi_k_pre))
    final_pi_k_all = jnp.concatenate([final_pi_k_pre * final_pi_K, jnp.array([final_pi_K])])
    
    return final_mu_k, final_L_inv_k, final_pi_k_all


igo_mog_optimizer = jax.jit(
    igo_mog_optimizer_impl, 
    static_argnames=('T', 'delta_t', 'K', 'B', 'B_0', 'fitness_fn', 'T_0')
)