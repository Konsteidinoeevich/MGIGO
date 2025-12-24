import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools

# ======================================================================
# I. 核心数学组件 (严格对齐 Algorithm 3)
# ======================================================================

@jit
def _get_gaussian_log_pdf(x, mu, S, D_m):
    """计算对数概率密度 (严格基于精度矩阵 S, 对应步骤 26)"""
    diff = x - mu
    # 精度矩阵 S 的维度是固定的 D_max，但在计算时仅考虑有效维度 D_m
    mahalanobis_sq = jnp.dot(diff, jnp.dot(S, diff))
    sign, logdet_S = jnp.linalg.slogdet(S)
    return 0.5 * (logdet_S - D_m * jnp.log(2 * jnp.pi) - mahalanobis_sq)

# ======================================================================
# II. 单分量更新逻辑 (步骤 23-31)
# ======================================================================

def _update_component(k_idx, mu_k, S_k, samples, w_hat, pi_j, mu_all, S_all, alpha_t, D_m):
    # 1. 计算权重 a_{j,k,b} (步骤 26)
    log_p_k = vmap(lambda x: _get_gaussian_log_pdf(x, mu_k, S_k, D_m))(samples)
    
    def log_p_mix(x):
        all_pdfs = vmap(lambda m, s: _get_gaussian_log_pdf(x, m, s, D_m))(mu_all, S_all)
        return jax.scipy.special.logsumexp(jnp.log(pi_j + 1e-15) + all_pdfs)
    
    log_p_theta = vmap(log_p_mix)(samples)
    a_k_b = jnp.exp(jnp.clip(log_p_k - log_p_theta, -70, 70))

    # 2. 更新精度矩阵 S (步骤 28)
    diff = samples - mu_k
    def s_grad_fn(d):
        Sd = jnp.dot(S_k, d)
        return jnp.outer(Sd, Sd) - S_k 
    
    sum_S_grad = jnp.sum((w_hat * a_k_b)[:, None, None] * vmap(s_grad_fn)(diff), axis=0)
    S_next = S_k - alpha_t * sum_S_grad
    S_next = (S_next + S_next.T) / 2.0 + jnp.eye(S_k.shape[0]) * 1e-6
    
    # 3. 更新均值 mu (步骤 30) - 必须使用 S_next
    S_next_inv = jnp.linalg.inv(S_next)
    mu_grad_terms = vmap(lambda d: jnp.dot(S_k, d))(diff)
    sum_mu_grad = jnp.sum((w_hat * a_k_b)[:, None] * mu_grad_terms, axis=0)
    mu_next = mu_k + alpha_t * jnp.dot(S_next_inv, sum_mu_grad)

    return mu_next, S_next, a_k_b

# ======================================================================
# III. 主优化步 (修复 NameError 和 Tracer Error)
# ======================================================================

def _step_logic(state, iter_data, M, K, B, B0, alpha_t, dims_arr, T_0, fitness_fn, context):
    mu, S, v, t = state
    key, _ = iter_data
    D_max = mu.shape[-1] # 静态维度

    # 1. 混合权重 (步骤 33-36)
    pi = vmap(jax.nn.softmax)(v)

    # 2. 采样阶段 (步骤 4-13)
    def sample_block(m_idx, sub_key):
        comp_idx = random.choice(sub_key, K, p=pi[m_idx], shape=(B,))
        def gen(c_idx, s_key):
            # 修复：使用静态 D_max 避免 Tracer Error
            cov = jnp.linalg.inv(S[m_idx, c_idx] + jnp.eye(D_max) * 1e-7)
            return random.multivariate_normal(s_key, mu[m_idx, c_idx], cov)
        return vmap(gen)(comp_idx, random.split(sub_key, B))

    samples_m = vmap(sample_block)(jnp.arange(M), random.split(key, M))
    
    # 3. 排名与权重 (步骤 14-19)
    samples_flat = samples_m.transpose(1, 0, 2).reshape(B, -1)
    f_vals = vmap(lambda s: fitness_fn(s, context))(samples_flat)
    ranks = jnp.argsort(jnp.argsort(f_vals)) 
    w_hat = jnp.where(ranks < B0, 1.0/B, 0.0)

    # 4. 定义 update_block 解决 NameError (步骤 21-39)
    def update_block(m_idx):
        res_mu, res_S, a_k_b_all = vmap(
            _update_component,
            in_axes=(0, 0, 0, None, None, None, None, None, None, None)
        )(jnp.arange(K), mu[m_idx], S[m_idx], samples_m[m_idx], w_hat, pi[m_idx], mu[m_idx], S[m_idx], alpha_t, dims_arr[m_idx])
        
        # 权重更新逻辑 (步骤 34)
        a_K_b = a_k_b_all[K-1]
        v_grad = vmap(lambda a_k: jnp.sum(w_hat * (a_k - a_K_b)))(a_k_b_all)
        v_next = jnp.where((t + 1) % T_0 == 0, jnp.zeros(K), v[m_idx] + alpha_t * v_grad)
        
        return res_mu, res_S, v_next

    # 执行并行块更新
    new_mu, new_S, new_v = vmap(update_block)(jnp.arange(M))
    return (new_mu, new_S, new_v, t + 1), None

# ======================================================================
# IV. 入口函数
# ======================================================================

@functools.partial(jit, static_argnums=(1, 3, 4, 5, 7, 9))
def mmog_igo_optimizer(key, T, dt, M, K, B, B0, dims, T_0, fitness_fn_total, 
                       initial_mu_k, initial_L_inv_k, initial_v_k, context):
    
    # 将 L_inv 转换为精度矩阵 S (步骤 24)
    S_init = vmap(vmap(lambda L: L @ L.T))(initial_L_inv_k)
    v_init = jnp.zeros((M, K))
    
    state = (initial_mu_k, S_init, v_init, 0)
    
    step_fn = functools.partial(_step_logic, M=M, K=K, B=B, B0=B0, alpha_t=dt, 
                                dims_arr=jnp.array(dims), T_0=T_0, 
                                fitness_fn=fitness_fn_total, context=context)
    
    final_state, _ = lax.scan(step_fn, state, (random.split(key, T), jnp.arange(T)))
    
    # 转换回 Cholesky (步骤 29)
    final_L = vmap(vmap(jnp.linalg.cholesky))(final_state[1])
    return final_state[0], final_L, vmap(jax.nn.softmax)(final_state[2])