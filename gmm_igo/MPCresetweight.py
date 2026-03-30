# MPCsolverM2_v2.py - 严格对齐 Algorithm 3 & 4 的改进型多块 IGO 优化器
import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools
from typing import Callable, Tuple, Any

# ======================================================================
# I. 核心辅助函数
# ======================================================================

@jit
def _logsumexp(a, axis=None):
    return jnp.logaddexp.reduce(a, axis=axis)

@jit
def _gaussian_log_pdf_l_masked(xi, mu, L_inv, D_m):
    diff = (xi - mu)
    mask = jnp.arange(xi.shape[0]) < D_m
    diff = diff * mask
    
    y = L_inv @ diff
    mahalanobis_sq = jnp.sum(y**2)
    
    diag_L = jnp.diag(L_inv)
    log_det_S_inv = -2.0 * jnp.sum(jnp.where(mask, jnp.log(diag_L + 1e-12), 0.0))
    
    log_pdf = -0.5 * (D_m * jnp.log(2 * jnp.pi) + log_det_S_inv + mahalanobis_sq)
    return log_pdf

# ======================================================================
# II. 单分量更新逻辑 (Algorithm 3 Steps 23-34)
# ======================================================================

def _update_step_k_l_single_component(
    k_idx, mu_k_t, L_inv_k_t, samples, elite_weights, 
    pi_all, mu_all, L_inv_all, delta_t,
    mu_baseline, L_inv_baseline, D_m
):
    D_max = mu_k_t.shape[0]
    S_k_t = L_inv_k_t @ L_inv_k_t.T
    
    # 1. 计算当前分量与 MoG 的对数 PDF
    log_pdf_k = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_k_t, L_inv_k_t, D_m))(samples)
    log_pdf_base = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_baseline, L_inv_baseline, D_m))(samples)
    
    def mog_pdf_fn(xi):
        l_pdfs = vmap(lambda m, l: _gaussian_log_pdf_l_masked(xi, m, l, D_m))(mu_all, L_inv_all)
        return _logsumexp(jnp.log(pi_all + 1e-15) + l_pdfs)
    
    log_mog = vmap(mog_pdf_fn)(samples)

    # 2. 计算权重项 a_i 和 b_i (Algorithm 3 Step 26)
    a_i = jnp.exp(jnp.clip(log_pdf_k - log_mog, a_min=-70.0, a_max=70.0))
    b_i = jnp.exp(jnp.clip(log_pdf_base - log_mog, a_min=-70.0, a_max=70.0))
    
    # 3. 均值 mu 更新 (Step 30)
    diff = (samples - mu_k_t)
    grad_mu = (S_k_t @ diff.T).T
    sum_mu = jnp.sum((elite_weights * a_i)[:, None] * grad_mu, axis=0)
    
    # 4. 精度矩阵 S 更新 (Step 28)
    #Sigma_k = jnp.linalg.inv(S_k_t + jnp.eye(D_max)*1e-6)
    def outer_prod_update(d):
        return S_k_t @ jnp.outer(d, d) @ S_k_t - S_k_t
    
    S_grads = vmap(outer_prod_update)(diff)
    sum_S = jnp.sum((elite_weights * a_i)[:, None, None] * S_grads, axis=0)

    S_new = S_k_t - delta_t * sum_S
    S_new = (S_new + S_new.T) / 2.0 + jnp.eye(D_max) * 1e-6
    L_inv_new = jnp.linalg.cholesky(S_new)

    mu_new = mu_k_t + delta_t * jnp.linalg.solve(S_new , sum_mu)
    
    # 5. 权重 v 更新增量 (Step 34)
    v_update_val = jnp.sum(elite_weights * (a_i - b_i))

    return mu_new, L_inv_new, v_update_val

# ======================================================================
# III. 主优化流程 (集成 Algorithm 4 权重筛选)
# ======================================================================

def _parallel_step_with_algo4(state, iter_data, M, K, B, B0, dt, dims_arr, T_0, fitness_fn, v_reset, context):
    mu, L_inv, v, z_star = state # z_star 为上一步的精英阈值
    key, idx = iter_data
    
    # 检测是否为重置步后的第一步 (Algorithm 4 应用时机)
    is_after_reset = (idx % T_0 == 1) & (idx > 0)
    
    # Algorithm 3 Step 33-38: 重置权重逻辑
    v_curr = jnp.where((idx % T_0) == 0, v_reset, v)
    
    def v_to_pi(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_exps = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_exps, jnp.array([1.0 / sum_exps])])
    
    pi_all = vmap(v_to_pi)(v_curr)

    # 1. 采样阶段 (Step 4-13)
    def sample_block(m_idx, sub_key):
        comps = random.choice(sub_key, K, p=pi_all[m_idx], shape=(B,))
        def sample_gaussian(c_idx, s_key):
            m_k = mu[m_idx, c_idx]
            S_k = L_inv[m_idx, c_idx] @ L_inv[m_idx, c_idx].T
            sigma = jnp.linalg.inv(S_k + jnp.eye(S_k.shape[0])*1e-7)
            return m_k + jnp.linalg.cholesky(sigma) @ random.normal(s_key, (m_k.shape[0],))
        return vmap(sample_gaussian)(comps, random.split(sub_key, B))

    keys_m = random.split(key, M)
    samples_m = vmap(sample_block)(jnp.arange(M), keys_m)
    
    # 2. 评价与排序 (Step 14-19)
    samples_flat = samples_m.transpose(1, 0, 2).reshape(B, -1)
    f_vals = vmap(lambda s: fitness_fn(s, context))(samples_flat)
    
    # 计算当前步的 B0 分位数
    sorted_f = jnp.sort(f_vals)
    current_z_B0 = sorted_f[B0 - 1]
    
    # --- Algorithm 4 核心逻辑 ---
    # 若刚重置，则权重 w = 1/B 当且仅当 f(z) <= min(z_star, current_z_B0)
    def weight_logic_algo4():
        threshold = jnp.minimum(z_star, current_z_B0)
        return jnp.where(f_vals <= threshold, 1.0/B, 0.0)

    # 正常步：直接取前 B0 个样本
    def weight_logic_algo3():
        ranks = jnp.argsort(jnp.argsort(f_vals))
        return jnp.where(ranks < B0, 1.0/B, 0.0)

    elite_weights = lax.cond(is_after_reset, weight_logic_algo4, weight_logic_algo3)

    # 3. 块并行更新 (Step 21-41)
    def update_m(m_idx):
        D_m = dims_arr[m_idx]
        mu_m = mu[m_idx]; L_m = L_inv[m_idx]; p_m = pi_all[m_idx]
        mu_base = mu_m[K-1]; L_base = L_m[K-1]
        
        res_mu, res_L, res_v_delta = vmap(
            _update_step_k_l_single_component,
            in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None)
        )(jnp.arange(K), mu_m, L_m, samples_m[m_idx], elite_weights, p_m, mu_m, L_m, dt, mu_base, L_base, D_m)
        
        return res_mu, res_L, res_v_delta[:K-1]

    new_mu, new_L, v_deltas = vmap(update_m)(jnp.arange(M))
    new_v = jnp.clip(v_curr + dt * v_deltas, -70.0, 70.0)
    
    # 记录当前步的 B0 分位数作为下一轮可能的 z_star
    return (new_mu, new_L, new_v, current_z_B0), None

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
    
    # 增加 z_star 的状态维护，初始设为极大值
    z_star_init = jnp.array(1e10) 
    state = (initial_mu_k[:, :K, :], initial_L_inv_k[:, :K, :, :], v_reset, z_star_init)
    
    keys = random.split(key, T)
    
    final_state, _ = lax.scan(
        functools.partial(
            _parallel_step_with_algo4, 
            M=M, K=K, B=B, B0=B0, dt=dt, 
            dims_arr=dims_array, T_0=T_0, 
            fitness_fn=fitness_fn_total, v_reset=v_reset,
            context=context
        ),
        state, (keys, jnp.arange(T))
    )
    
    final_mu, final_L, final_v, _ = final_state
    
    def v_to_pi_final(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    
    final_pi = vmap(v_to_pi_final)(final_v)
    return final_mu, final_L, final_pi