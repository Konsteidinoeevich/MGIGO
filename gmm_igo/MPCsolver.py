# gmm_igo/solver.py - Modified for Context Propagation

import jax
import jax.numpy as jnp
from jax import vmap, random, lax
import functools

# ----------------------------------------------------------------------
# I. 核心辅助函数 (信息矩阵 S_k 范式)
# ----------------------------------------------------------------------

@jax.jit
def _logsumexp(a, axis=None):
    return jnp.logaddexp.reduce(a, axis=axis)

@jax.jit
def _gaussian_log_pdf_l(xi, mu, L_inv):
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
    log_pdfs_k = _vmap_gaussian_log_pdf_l_k(xi, mu_k, L_inv_k)
    log_pi_k = jnp.log(pi_k)
    log_weighted_pdfs = log_pi_k + log_pdfs_k
    return _logsumexp(log_weighted_pdfs)

@jax.jit
def _sample_from_component_l(idx, key_sample, mu_k_all, L_inv_k_all):
    mu_k = mu_k_all[idx]
    S_k = L_inv_k_all[idx] @ L_inv_k_all[idx].T 
    L_Sigma = jnp.linalg.cholesky(jnp.linalg.inv(S_k))
    D = mu_k.shape[0]
    z = random.normal(key_sample, shape=(D,))
    return mu_k + L_Sigma @ z

# [关键修改 1] 增加 context 参数
def _get_elite_weights(samples, fitness_fn, B, B_0, context):
    """
    评估 f(xi, context) 并计算精英样本权重。
    fitness_fn 签名必须为: fn(sample, context) -> scalar
    """
    # vmap: samples (axis 0) 变化, context (None) 广播共享
    f_xi = vmap(fitness_fn, in_axes=(0, None))(samples, context)
    ranks = jnp.argsort(jnp.argsort(f_xi))
    is_elite = ranks < B_0
    return jnp.where(is_elite, 1.0, 0.0) / B 

# ----------------------------------------------------------------------
# II. K 个分量的并行更新逻辑
# ----------------------------------------------------------------------

LOG_CLIP_VALUE = 80.0 

def _update_step_k_l_single_component(
    k_idx, mu_k_t, L_inv_k_t, samples, elite_weights, 
    pi_k_all, mu_k_all, L_inv_k_all, delta_t,
    mu_K_t, L_inv_K_t 
):
    S_k_t = L_inv_k_t @ L_inv_k_t.T 
    
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
    
    diff = samples - mu_k_t
    diff_outer = vmap(lambda x: jnp.outer(x, x))(diff)
    
    #Sigma_k_t = jnp.linalg.inv(S_k_t) 
    S_update_term_i = S_k_t @ diff_outer @ S_k_t - S_k_t[None, :, :]
    
    sum_S_update = jnp.sum(scaled_a_i[:, None, None] * S_update_term_i, axis=0)
    
    S_k_t_plus_1_prop = S_k_t - delta_t * sum_S_update 
    
    S_k_t_plus_1_prop = (S_k_t_plus_1_prop + S_k_t_plus_1_prop.T) / 2
    D_dim = S_k_t_plus_1_prop.shape[0]
    S_k_t_plus_1_prop = S_k_t_plus_1_prop + jnp.eye(D_dim) * 1e-6 
    
    L_inv_k_t_plus_1 = jnp.linalg.cholesky(S_k_t_plus_1_prop)
   # S_k_t_plus_1 = L_inv_k_t_plus_1 @ L_inv_k_t_plus_1.T

    weighted_diff = scaled_a_i[None, :] * diff.T 
    S_t_weighted_diff = S_k_t @ weighted_diff
    sum_term_vector = jnp.sum(S_t_weighted_diff, axis=1) 
    
    sum_term_vector_for_update=jnp.linalg.solve(L_inv_k_t_plus_1, sum_term_vector)
    mu_update_term =jnp.linalg.solve(L_inv_k_t_plus_1.T, sum_term_vector_for_update)
   # mu_update_term = jnp.linalg.solve(S_k_t_plus_1, sum_term_vector)
    mu_k_t_plus_1 = mu_k_t + delta_t * mu_update_term

    v_update_sum = jnp.sum(elite_weights * (a_i - b_i))
    
    return mu_k_t_plus_1, L_inv_k_t_plus_1, v_update_sum

# ----------------------------------------------------------------------
# III. 迭代步和主优化器函数
# ----------------------------------------------------------------------

_vmap_update_step_k_l = vmap(
    _update_step_k_l_single_component, 
    in_axes=(0, 0, 0, None, None, None, None, None, None, None, None), 
    out_axes=(0, 0, 0)
)

# [关键修改 2] 迭代步接受 context 参数
def _iteration_step(state, key_input, B, B_0, K, delta_t, fitness_fn, context):
    mu_k_t, L_inv_k_t, v_k_t = state
    key, subkey = random.split(key_input)
    
    pi_k_pre = jnp.exp(v_k_t)
    pi_K_t = 1 / (1 + jnp.sum(pi_k_pre))
    pi_k_t_all = jnp.concatenate([pi_k_pre * pi_K_t, jnp.array([pi_K_t])])
    
    comp_indices = random.choice(subkey, K, shape=(B,), p=pi_k_t_all)
    sample_keys = random.split(subkey, B)
    vmap_sample_fn = vmap(_sample_from_component_l, in_axes=(0, 0, None, None))
    samples = vmap_sample_fn(comp_indices, sample_keys, mu_k_t, L_inv_k_t)

    # 透传 context
    elite_weights = _get_elite_weights(samples, fitness_fn, B, B_0, context)
    
    k_indices = jnp.arange(K) 
    mu_K_t, L_inv_K_t = mu_k_t[-1], L_inv_k_t[-1]
    
    mu_k_t_plus_1, L_inv_k_t_plus_1, v_update_sum_k = _vmap_update_step_k_l(
        k_indices, mu_k_t, L_inv_k_t, samples, elite_weights, 
        pi_k_t_all, mu_k_t, L_inv_k_t, delta_t,
        mu_K_t, L_inv_K_t 
    )
    
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

# [关键修改 3] 主接口接受 context 参数
def igo_mog_optimizer_impl(
    key, T, delta_t, K, B, B_0, fitness_fn,
    initial_mu_k, initial_L_inv_k, initial_pi_k,
    context # <--- 新增
):
    if K < 2:
        raise ValueError("K must be 2 or greater.")
        
    pi_K_0 = initial_pi_k[-1]
    pi_k_0_pre = initial_pi_k[:-1]
    v_k_0 = jnp.log(pi_k_0_pre / pi_K_0)
        
    initial_state = (initial_mu_k, initial_L_inv_k, v_k_0)
    
    # 使用 partial 将 context 绑定到迭代步中
    # 这样 context 对 lax.scan 来说是环境常量，不会影响编译（如果形状不变）
    bound_iteration_step = functools.partial(
        _iteration_step, 
        B=B, B_0=B_0, K=K, delta_t=delta_t, fitness_fn=fitness_fn,
        context=context 
    )

    keys_iter = random.split(key, T)
    final_state, _ = lax.scan(bound_iteration_step, initial_state, keys_iter)

    final_mu_k, final_L_inv_k, final_v_k = final_state
    
    final_pi_k_pre = jnp.exp(final_v_k)
    final_pi_K = 1 / (1 + jnp.sum(final_pi_k_pre))
    final_pi_k_all = jnp.concatenate([final_pi_k_pre * final_pi_K, jnp.array([final_pi_K])])
    
    return final_mu_k, final_L_inv_k, final_pi_k_all

# 注意：context 不在 static_argnames 里，它是动态数据
igo_mog_optimizer = jax.jit(
    igo_mog_optimizer_impl, 
    static_argnames=('T', 'delta_t', 'K', 'B', 'B_0', 'fitness_fn')
)