import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools

# ======================================================================
# I. 核心辅助函数 (支持异构维度掩码)
# ======================================================================

@jit
def _logsumexp(a, axis=None):
    return jnp.logaddexp.reduce(a, axis=axis)

@jit
def _gaussian_log_pdf_l_masked(xi, mu, S, D_m):
    """基于精度矩阵 S 计算对数概率密度，支持维度掩码"""
    diff = (xi - mu)
    mask = jnp.arange(xi.shape[0]) < D_m
    diff = diff * mask
    
    mahalanobis_sq = jnp.dot(diff, jnp.dot(S, diff))
    sign, logdet_S = jnp.linalg.slogdet(S) # S = LL^T, 所以 logdet(S) = logdet(inv(Sigma))
    
    log_pdf = -0.5 * (D_m * jnp.log(2 * jnp.pi) - logdet_S + mahalanobis_sq)
    return log_pdf

# ======================================================================
# II. 单分量更新逻辑 (对齐 MPCsolverM2 核心公式)
# ======================================================================

def _update_component_core(
    k_idx, mu_k, S_k, samples, elite_weights, 
    pi_all, mu_all, S_all, delta_t,
    mu_base, S_base, D_m
):
    D_max = mu_k.shape[0]
    
    # 1. 计算当前分量、基准分量和整体 MoG 的对数 PDF
    log_pdf_k = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_k, S_k, D_m))(samples)
    log_pdf_base = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_base, S_base, D_m))(samples)
    
    def mog_pdf_fn(xi):
        l_pdfs = vmap(lambda m, s: _gaussian_log_pdf_l_masked(xi, m, s, D_m))(mu_all, S_all)
        return _logsumexp(jnp.log(pi_all + 1e-15) + l_pdfs)
    
    log_mog = vmap(mog_pdf_fn)(samples)

    # 2. IGO 权重项 (Line 26)
    a_i = jnp.exp(jnp.clip(log_pdf_k - log_mog, -70.0, 70.0))
    b_i = jnp.exp(jnp.clip(log_pdf_base - log_mog, -70.0, 70.0))
    
    # 3. 精度矩阵 S 更新 (Line 28)
    diff = (samples - mu_k)
    #Sigma_k = jnp.linalg.inv(S_k + jnp.eye(D_max) * 1e-6)
    
    def s_grad_fn(d):
        return S_k @ jnp.outer(d, d) @ S_k - S_k
    
    sum_S_grad = jnp.sum((elite_weights * a_i)[:, None, None] * vmap(s_grad_fn)(diff), axis=0)
    S_new = S_k - delta_t * sum_S_grad
    S_new = (S_new + S_new.T) / 2.0 + jnp.eye(D_max) * 1e-6

    # 4. 均值 mu 更新 (Line 30) - 使用更新后的 S_new
    grad_mu_terms = (S_k @ diff.T).T
    sum_mu_grad = jnp.sum((elite_weights * a_i)[:, None] * grad_mu_terms, axis=0)
    mu_new = mu_k + delta_t * jnp.linalg.solve(S_new, sum_mu_grad)
    
    v_delta = jnp.sum(elite_weights * (a_i - b_i))

    return mu_new, S_new, v_delta

# ======================================================================
# III. 优化器迭代步
# ======================================================================

def _step_fn(state, iter_data, M, K, B, B0, dt, dims_arr, T_0, fitness_fn, v_reset, context):
    mu, S, v, t = state
    key, _ = iter_data
    
    # T0 重置混合权重
    v = jnp.where((t % T_0) == 0, v_reset, v)
    
    def v_to_pi(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    
    pi_all = vmap(v_to_pi)(v)

    # 1. 采样
    def sample_block(m_idx, sub_key):
        comps = random.choice(sub_key, K, p=pi_all[m_idx], shape=(B,))
        def gen_sample(c_idx, s_key):
            cov = jnp.linalg.inv(S[m_idx, c_idx] + jnp.eye(S.shape[-1]) * 1e-7)
            return random.multivariate_normal(s_key, mu[m_idx, c_idx], cov)
        return vmap(gen_sample)(comps, random.split(sub_key, B))

    samples_m = vmap(sample_block)(jnp.arange(M), random.split(key, M))
    
    # 2. 评价与精英权重
    samples_flat = samples_m.transpose(1, 0, 2).reshape(B, -1)
    f_vals = vmap(lambda s: fitness_fn(s, context))(samples_flat)
    ranks = jnp.argsort(jnp.argsort(f_vals)) 
    w_hat = jnp.where(ranks < B0, 1.0/B, 0.0)

    # 3. 块并行更新
    def update_block(m_idx):
        D_m = dims_arr[m_idx]
        mu_base, S_base = mu[m_idx, K-1], S[m_idx, K-1]
        
        new_mu_m, new_S_m, v_deltas = vmap(
            _update_component_core,
            in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None)
        )(jnp.arange(K), mu[m_idx], S[m_idx], samples_m[m_idx], w_hat, 
          pi_all[m_idx], mu[m_idx], S[m_idx], dt, mu_base, S_base, D_m)
        
        return new_mu_m, new_S_m, v_deltas[:K-1]

    next_mu, next_S, next_v_deltas = vmap(update_block)(jnp.arange(M))
    next_v = jnp.clip(v + dt * next_v_deltas, -70.0, 70.0)
    
    return (next_mu, next_S, next_v, t + 1), None

# ======================================================================
# IV. 顶层入口
# ======================================================================

@functools.partial(jit, static_argnums=(1, 3, 4, 5, 7, 9))
def mmog_igo_optimizer_mpc(
    key, T, dt, M, K, B, B0, dims, T_0, 
    fitness_fn_total, initial_mu_k, initial_L_inv_k, initial_v_k, context
):
    dims_array = jnp.array(dims)
    v_reset = jnp.zeros((M, K-1))
    
    # 将 L_inv 转换为初始精度矩阵 S
    S_init = vmap(vmap(lambda L: L @ L.T))(initial_L_inv_k[:, :K, :, :])
    mu_init = initial_mu_k[:, :K, :]
    v_init = jnp.zeros((M, K-1)) 

    state = (mu_init, S_init, v_init, 0)
    
    loop_fn = functools.partial(
        _step_fn, M=M, K=K, B=B, B0=B0, dt=dt, 
        dims_arr=dims_array, T_0=T_0, 
        fitness_fn=fitness_fn_total, v_reset=v_reset, context=context
    )
    
    final_state, _ = lax.scan(loop_fn, state, (random.split(key, T), jnp.arange(T)))
    
    # 结果转换
    def v_to_pi_final(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        return jnp.concatenate([exps / (1.0 + jnp.sum(exps)), jnp.array([1.0 / (1.0 + jnp.sum(exps))])])
    
    final_pi = vmap(v_to_pi_final)(final_state[2])
    # 将精度矩阵转回 Cholesky 因子输出
    final_L = vmap(vmap(jnp.linalg.cholesky))(final_state[1])
    
    return final_state[0], final_L, final_pi