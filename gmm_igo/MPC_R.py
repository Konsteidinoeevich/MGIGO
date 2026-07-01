import jax
import jax.numpy as jnp
from jax import vmap, random, lax
import functools

# ----------------------------------------------------------------------
# I. 基础数值与概率计算函数
# ----------------------------------------------------------------------

MIN_EIG = 1e-2
MAX_EIG = 1e3

def _safe_spd_projection(S):
    """确保精度矩阵 S 的数值稳定性"""
    eigvals, eigvecs = jnp.linalg.eigh(S)
    eigvals = jnp.clip(eigvals, MIN_EIG, MAX_EIG)
    return eigvecs @ (eigvals[:, None] * eigvecs.T)

@jax.jit
def _gaussian_log_pdf_l(xi, mu, L_inv):
    """高斯对数概率密度"""
    D = mu.shape[0]
    diff = xi - mu
    mahalanobis_sq = jnp.sum((L_inv @ diff)**2)
    log_det_S_inv = -2 * jnp.sum(jnp.log(jnp.diag(L_inv)))
    return -0.5 * (D * jnp.log(2 * jnp.pi) + log_det_S_inv + mahalanobis_sq)

@jax.jit
def _mixture_log_pdf_l(xi, mu_k, L_inv_k, pi_k):
    """混合高斯对数概率密度"""
    log_pdfs = vmap(_gaussian_log_pdf_l, in_axes=(None, 0, 0))(xi, mu_k, L_inv_k)
    return jnp.logaddexp.reduce(jnp.log(pi_k) + log_pdfs)

@jax.jit
def _sample_from_component_l(idx, key, mu_k_all, L_inv_k_all):
    """从 GMM 分量采样"""
    mu = mu_k_all[idx]
    S = L_inv_k_all[idx] @ L_inv_k_all[idx].T
    L_sigma = jnp.linalg.cholesky(jnp.linalg.inv(S))
    return mu + L_sigma @ random.normal(key, shape=(mu.shape[0],))

# ----------------------------------------------------------------------
# II. 核心更新步：严格对齐用户公式
# ----------------------------------------------------------------------

@jax.jit
def _update_step_reuse_k(
    k_idx, mu_k_t, L_inv_k_t, all_samples, is_selected, 
    pi_curr, mu_curr, L_inv_curr, 
    pi_prev, mu_prev, L_inv_prev,
    delta_t, B_n, B_o
):
   
    S_k_t = L_inv_k_t @ L_inv_k_t.T
    N_total = all_samples.shape[0] # N_total = B_n + B_o

    # 1. 计算分子：高斯概率密度
    p_gaussian_t = vmap(lambda xi, mu, L_inv: jnp.exp(_gaussian_log_pdf_l(xi, mu, L_inv)), in_axes=(0, None, None))
    
    p_i_t = p_gaussian_t(all_samples, mu_k_t, L_inv_k_t)
    p_K_t = p_gaussian_t(all_samples, mu_curr[-1], L_inv_curr[-1])

    # 2. 混合概率密度分母计算
    def get_denom(sample_z, sample_idx):
        def compute_t():
            mix_probs = vmap(lambda mu, L, pi: pi * jnp.exp(_gaussian_log_pdf_l(sample_z, mu, L)), in_axes=(0, 0, 0))(mu_curr, L_inv_curr, pi_curr)
            return jnp.sum(mix_probs)
            
        def compute_prev():
            mix_probs = vmap(lambda mu, L, pi: pi * jnp.exp(_gaussian_log_pdf_l(sample_z, mu, L)), in_axes=(0, 0, 0))(mu_prev, L_inv_prev, pi_prev)
            return jnp.sum(mix_probs)
            
        return lax.cond(sample_idx < B_n, lambda _: compute_t(), lambda _: compute_prev(), None)

    # 通过显式输入数组 `jnp.arange(N_total)` 匹配形状
    denominators = vmap(get_denom, in_axes=(0, 0))(all_samples, jnp.arange(N_total))
    denominators = jnp.clip(denominators, a_min=1e-10) # 防止分母为 0

    # 3. 计算 a_tilde: log-space clip 然后 exp (对齐 M22, 防止数值爆炸)
    log_a_tilde_i = jnp.clip(jnp.log(p_i_t + 1e-15) - jnp.log(denominators), -20.0, 20.0)
    a_tilde_i = jnp.exp(log_a_tilde_i)
    log_a_tilde_K = jnp.clip(jnp.log(p_K_t + 1e-15) - jnp.log(denominators), -20.0, 20.0)
    a_tilde_K = jnp.exp(log_a_tilde_K)

    weighted_factor = is_selected / N_total

    # 4. 精度矩阵 S 更新公式
    diff = all_samples - mu_k_t
    S_diff = (S_k_t @ diff.T).T
    S_diff_outer = vmap(lambda x: jnp.outer(x, x))(S_diff)
    
    S_update = jnp.sum((weighted_factor * a_tilde_i)[:, None, None] * (S_diff_outer - S_k_t[None, :, :]), axis=0)
    
    MAX_S_UPDATE_NORM = 1e4
    S_up_norm = jnp.linalg.norm(S_update)
    S_update = jnp.where(
        S_up_norm > MAX_S_UPDATE_NORM,
        S_update * (MAX_S_UPDATE_NORM / (S_up_norm + 1e-9)),
        S_update,
    )
    S_next = _safe_spd_projection(S_k_t - delta_t * S_update)
    L_inv_next = jnp.linalg.cholesky(S_next)
    
    # 5. 均值 mu 更新公式
    mu_grad = jnp.sum((weighted_factor * a_tilde_i)[:, None] * S_diff, axis=0)
    mu_next = mu_k_t + delta_t * jnp.linalg.solve(S_next, mu_grad)
    
    # 6. 计算权重变化增量
    delta_i = jnp.sum(weighted_factor * a_tilde_i)
    delta_K = jnp.sum(weighted_factor * a_tilde_K)
    
    return mu_next, L_inv_next, delta_i - delta_K

# ----------------------------------------------------------------------
# III. 主循环控制
# ----------------------------------------------------------------------

def _iteration_step_reuse(state, key_input, B_n, B_o, K, delta_t, a_threshold, fitness_fn, context):
    mu_t, L_t, v_t, prev_mu, prev_L, prev_pi, old_z, old_f = state
    key, subkey = random.split(key_input)
    
    pi_pre = jnp.exp(v_t)
    pi_K = 1.0 / (1.0 + jnp.sum(pi_pre))
    pi_t = jnp.concatenate([pi_pre * pi_K, jnp.array([pi_K])])
    
    comp_idx = random.choice(subkey, K, shape=(B_n,), p=pi_t)
    new_z = vmap(_sample_from_component_l, in_axes=(0, 0, None, None))(
        comp_idx, random.split(subkey, B_n), mu_t, L_t
    )

    # 合并样本并计算适应度，透传 context 参数
    new_f = vmap(fitness_fn, in_axes=(0, None))(new_z, context)
    all_z = jnp.concatenate([new_z, old_z], axis=0)
    all_f = jnp.concatenate([new_f, old_f], axis=0)

    # 计算 p(z_b; \Lambda^t) 和 p(z_b; \Lambda^{t-1})
    log_p_curr = vmap(_mixture_log_pdf_l, in_axes=(0, None, None, None))(all_z, mu_t, L_t, pi_t)
    log_p_prev = vmap(_mixture_log_pdf_l, in_axes=(0, None, None, None))(all_z, prev_mu, prev_L, prev_pi)
    
    # 权重 omega 计算
    omega_new = jnp.ones(B_n)
    omega_old = jnp.exp(jnp.clip(log_p_curr[B_n:] - log_p_prev[B_n:], -10.0, 10.0))
    omega = jnp.concatenate([omega_new, omega_old], axis=0)
    
    # 归一化重要性权重，防止数值爆炸
    omega = omega / jnp.mean(omega)
    
    # 累加和计算 $\widehat{q}_{\Lambda^t}^{f}(z_{(b)})$
    N_total = B_n + B_o
    sort_indices = jnp.argsort(all_f)
    sorted_omega = omega[sort_indices]
    
    cumsum_omega = jnp.cumsum(sorted_omega)
    q_values = jnp.concatenate([jnp.zeros(1), cumsum_omega[:-1]]) / N_total
    
    sorted_is_selected = (q_values <= a_threshold).astype(jnp.float32)
    
    is_selected = jnp.zeros(N_total).at[sort_indices].set(sorted_is_selected)

    vmap_upd = vmap(_update_step_reuse_k, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None))
    mu_next, L_next, v_deltas = vmap_upd(
        jnp.arange(K), mu_t, L_t, all_z, is_selected, 
        pi_t, mu_t, L_t,
        prev_pi, prev_mu, prev_L,
        delta_t, B_n, B_o
    )
    
    v_next = jnp.clip(v_t + delta_t * jnp.clip(v_deltas[:K-1], -10.0, 10.0), a_max=70.0)
    
    # 【修复重点】：固定传递给下一轮 state 的 old_z 与 old_f 尺寸为 (B_o, D) 和 (B_o,)
    # 防止因 B_n 和 B_o 不相等导致的 shape mismatch 错误。
    next_old_z = all_z[:B_o]
    next_old_f = all_f[:B_o]
    
    return (mu_next, L_next, v_next, mu_t, L_t, pi_t, next_old_z, next_old_f), None

# ----------------------------------------------------------------------
# IV. 主优化器入口
# ----------------------------------------------------------------------

def igo_mog_reuse_optimizer_impl(key, T, delta_t, K, B_n, B_o, a_threshold, fitness_fn, 
                                 initial_mu, initial_L_inv, initial_pi, context):
    v_0 = jnp.log(initial_pi[:-1] / initial_pi[-1])
    
    key, subkey = random.split(key)
    init_old_z = vmap(_sample_from_component_l, in_axes=(0, 0, None, None))(
        random.choice(subkey, K, shape=(B_o,), p=initial_pi), 
        random.split(subkey, B_o), initial_mu, initial_L_inv
    )
    init_old_f = vmap(fitness_fn, in_axes=(0, None))(init_old_z, context)
    
    initial_state = (initial_mu, initial_L_inv, v_0, initial_mu, initial_L_inv, initial_pi, init_old_z, init_old_f)
    
    step_fn = functools.partial(
        _iteration_step_reuse, 
        B_n=B_n, B_o=B_o, K=K, delta_t=delta_t, 
        a_threshold=a_threshold, fitness_fn=fitness_fn, context=context
    )
    
    final_state, _ = lax.scan(step_fn, initial_state, random.split(key, T))
    
    f_mu, f_L, f_v = final_state[0], final_state[1], final_state[2]
    f_pi = jnp.concatenate([jnp.exp(f_v), jnp.array([1.0])])
    return f_mu, f_L, f_pi / jnp.sum(f_pi)

igo_mog_reuse_optimizer = jax.jit(
    igo_mog_reuse_optimizer_impl,
    static_argnames=('T', 'delta_t', 'K', 'B_n', 'B_o', 'a_threshold', 'fitness_fn')
)