# gmm_igo/solver.py - 混合高斯信息几何优化器核心模块 (最终信息矩阵范式)

import jax
import jax.numpy as jnp
from jax import vmap, random, lax
import functools

def _nan_check(x, name):
    jax.debug.print(
        "[NaN CHECK] {name}: nan={nan}, inf={inf}",
        name=name,
        nan=jnp.any(jnp.isnan(x)),
        inf=jnp.any(jnp.isinf(x)),
    )
    return x

# ----------------------------------------------------------------------
# I. 核心辅助函数 (信息矩阵 S_k 范式)
# ----------------------------------------------------------------------

@jax.jit
def _logsumexp(a, axis=None):
    """Numerically stable calculation of log(sum(exp(a)))."""
    return jnp.logaddexp.reduce(a, axis=axis)

@jax.jit
def _gaussian_log_pdf_l(xi, mu, L_inv):
    """计算 N(mu, (L_inv @ L_inv.T)^{-1}) 的对数概率密度。
       L_inv 是 Cholesky 因子 L 的逆矩阵的转置。
    """
    D = mu.shape[0] 
    diff = xi - mu
    
    # 协方差 Sigma = S^{-1} = (L_inv @ L_inv.T)^{-1}
    # S = L_inv @ L_inv.T
    
    # y = L_inv * diff
    y = L_inv @ diff
    mahalanobis_sq = jnp.sum(y**2)
    
    # log|Sigma| = log|S^{-1}| = -log|S| = -2 * log|L_inv|
    log_det_S_inv = -2 * jnp.sum(jnp.log(jnp.diag(L_inv)))
    
    # log(pdf) = -0.5 * (D * log(2*pi) + log|Sigma| + Mahalanobis^2)
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

    
    _nan_check(S_k_t, "S_k_t")

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

    _nan_check(a_i, "a_i")
    _nan_check(b_i, "b_i")
    _nan_check(scaled_a_i, "scaled_a_i")    
    # --- 2. S_k 更新 (信息矩阵 S_k 的自然梯度) ---
    diff = samples - mu_k_t
    diff_outer = vmap(lambda x: jnp.outer(x, x))(diff)
    
    # 协方差 Sigma = S^{-1}
    # 协方差更新项 (Sigma_t @ diff_outer @ Sigma_t - Sigma_t)
    # 原始 IGO S_k 更新项 (恢复到您的原始理论形式)
    
    Sigma_k_t = jnp.linalg.solve(S_k_t, jnp.eye(D)) # 协方差 Sigma_k = S_k^{-1}
    

    _nan_check(Sigma_k_t, "Sigma_k_t")

    # 恢复您最初的逻辑，但使用 Sigma_k = S_k^{-1}
    S_update_term_i = Sigma_k_t @ diff_outer @ Sigma_k_t - Sigma_k_t[None, :, :]
    
    # S_k_t+1 应该收敛于 S_target，但保持您的原始理论减法结构
    sum_S_update = jnp.sum(scaled_a_i[:, None, None] * S_update_term_i, axis=0)
    
    # S_{k,t+1} 的目标应是 S_{k,t} + \Delta S_k
    # \Delta S_k 形式在信息矩阵下难以直接推导，我们保持您原始的减法结构：
    S_k_t_plus_1_prop = S_k_t - delta_t * sum_S_update # S_k 是信息矩阵
    
    S_k_t_plus_1_prop = (S_k_t_plus_1_prop + S_k_t_plus_1_prop.T) / 2
    #D_dim = S_k_t_plus_1_prop.shape[0]
    #S_k_t_plus_1_prop = S_k_t_plus_1_prop + jnp.eye(D_dim) * 1e-6 
    
    S_k_t_plus_1_prop = (S_k_t_plus_1_prop + S_k_t_plus_1_prop.T) / 2

# Sigma = S^{-1}
    Sigma_k_t_plus_1 = jnp.linalg.solve(S_k_t_plus_1_prop, jnp.eye(D))

# 协方差正则
    Sigma_k_t_plus_1 = Sigma_k_t_plus_1 + 1e-10 * jnp.eye(D)

# 回到信息矩阵
    S_k_t_plus_1_prop = jnp.linalg.solve(Sigma_k_t_plus_1, jnp.eye(D))

# Cholesky
    L_inv_k_t_plus_1 = jnp.linalg.cholesky(S_k_t_plus_1_prop)
    # 更新 L_inv
  #  L_inv_k_t_plus_1 = jnp.linalg.cholesky(S_k_t_plus_1_prop)
    #S_k_t_plus_1 = L_inv_k_t_plus_1 @ L_inv_k_t_plus_1.T


    # --- 3. mu_k 更新 (严格遵循您的要求) ---
    
    # 1. Sum = \sum_{i} \omega_i \alpha_i^k \cdot S_{k, t} (\mathbf{\xi}_i - \mu_{k, t})
    weighted_diff = scaled_a_i[None, :] * diff.T 
    
    # S_{k,t} 是信息矩阵，乘 (\xi_i - \mu)
    S_t_weighted_diff = S_k_t @ weighted_diff
    sum_term_vector = jnp.sum(S_t_weighted_diff, axis=1) # D 维向量

   # sum_term_vector_for_update=jnp.linalg.solve(L_inv_k_t_plus_1, sum_term_vector)
    
   # mu_update_term =jnp.linalg.solve(L_inv_k_t_plus_1.T, sum_term_vector_for_update)
    # 2. mu\_update\_term = S_{k, t+1}^{-1} \cdot Sum
    #    使用 jnp.linalg.solve(A, b) 计算 A^{-1}b
    #    A = S_{k, t+1}, b = sum_term_vector
    mu_update_term = jnp.linalg.solve(S_k_t_plus_1_prop, sum_term_vector)

    # 3. 应用更新
    mu_k_t_plus_1 = mu_k_t + delta_t * mu_update_term

    # --- 4. 权重更新项 (v_k) ---
    v_update_sum = jnp.sum(elite_weights * (a_i - b_i))
    
    return mu_k_t_plus_1, L_inv_k_t_plus_1, v_update_sum

# ----------------------------------------------------------------------
# III. 迭代步和主优化器函数 (集成 v_k 裁剪)
# ----------------------------------------------------------------------

_vmap_update_step_k_l = vmap(
    _update_step_k_l_single_component, 
    in_axes=(0, 0, 0, None, None, None, None, None, None, None, None), 
    out_axes=(0, 0, 0)
)

def _iteration_step(state, key_input, B, B_0, K, delta_t, fitness_fn):
    """一个完整的 IGO-MoG 迭代步，用于 lax.scan。"""
    mu_k_t, L_inv_k_t, v_k_t = state
    key, subkey = random.split(key_input)
    
    # 1. 计算 K 个权重 pi_k_t_all
    pi_k_pre = jnp.exp(v_k_t)
    pi_K_t = 1 / (1 + jnp.sum(pi_k_pre))
    pi_k_t_all = jnp.concatenate([pi_k_pre * pi_K_t, jnp.array([pi_K_t])])
    
    # 2. 从 MoG 分布中取样 B 个样本
    comp_indices = random.choice(subkey, K, shape=(B,), p=pi_k_t_all)
    sample_keys = random.split(subkey, B)
    vmap_sample_fn = vmap(_sample_from_component_l, in_axes=(0, 0, None, None))
    samples = vmap_sample_fn(comp_indices, sample_keys, mu_k_t, L_inv_k_t)

    # 3. 评估 f(xi) 并计算精英样本权重
    elite_weights = _get_elite_weights(samples, fitness_fn, B, B_0)
    
    # 4. K 个分量并行更新
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
    
    v_k_t_plus_1 = v_k_t + delta_t * v_update_safe
    MAX_V_K = 70.0 
    v_k_t_plus_1 = jnp.clip(v_k_t_plus_1, a_max=MAX_V_K)
    
    new_state = (mu_k_t_plus_1, L_inv_k_t_plus_1, v_k_t_plus_1)
    
    return new_state, None

def igo_mog_optimizer_impl(
    key, T, delta_t, K, B, B_0, fitness_fn,
    initial_mu_k, initial_L_inv_k, initial_pi_k
):
    """主实现逻辑，注意参数 L_inv_k 是信息矩阵的 Cholesky 因子。"""
    
    if K < 2:
        raise ValueError("K must be 2 or greater for Mixture of Gaussians IGO.")
        
    pi_K_0 = initial_pi_k[-1]
    pi_k_0_pre = initial_pi_k[:-1]
    v_k_0 = jnp.log(pi_k_0_pre / pi_K_0)
        
    initial_state = (initial_mu_k, initial_L_inv_k, v_k_0)
    
    bound_iteration_step = functools.partial(
        _iteration_step, 
        B=B, B_0=B_0, K=K, delta_t=delta_t, fitness_fn=fitness_fn
    )

    keys_iter = random.split(key, T)
    final_state, _ = lax.scan(bound_iteration_step, initial_state, keys_iter)

    final_mu_k, final_L_inv_k, final_v_k = final_state
    
    final_pi_k_pre = jnp.exp(final_v_k)
    final_pi_K = 1 / (1 + jnp.sum(final_pi_k_pre))
    final_pi_k_all = jnp.concatenate([final_pi_k_pre * final_pi_K, jnp.array([final_pi_K])])
    
    return final_mu_k, final_L_inv_k, final_pi_k_all


igo_mog_optimizer = jax.jit(
    igo_mog_optimizer_impl, 
    static_argnames=('T', 'delta_t', 'K', 'B', 'B_0', 'fitness_fn')
)