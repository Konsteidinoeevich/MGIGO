# gmm_igo/MPCsolverM2.py - 严格对齐 Algorithm 3 的多块 IGO 优化器
import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools
from typing import Callable, Tuple, Any

# ======================================================================
# I. 核心辅助函数 (信息几何范式)
# ======================================================================

@jit
def _logsumexp(a, axis=None):
    """数值稳定的 log(sum(exp(a)))"""
    return jnp.logaddexp.reduce(a, axis=axis)

@jit
def _gaussian_log_pdf_l_masked(xi, mu, L_inv, D_m):
    """
    计算对数概率密度，支持异构维度掩码。
    xi, mu 长度为 D_max, L_inv 为 D_max * D_max
    """
    # 仅提取有效维度 D_m 进行计算
    diff = (xi - mu)
    # 创建掩码
    mask = jnp.arange(xi.shape[0]) < D_m
    diff = diff * mask
    
    # 仅对有效区域进行矩阵运算
    y = L_inv @ diff
    mahalanobis_sq = jnp.sum(y**2)
    
    # log_det(S_inv) = -2 * sum(log(diag(L_inv)))
    # 注意：只计算前 D_m 个对角线元素
    diag_L = jnp.diag(L_inv)
    log_det_S_inv = -2.0 * jnp.sum(jnp.where(mask, jnp.log(diag_L + 1e-12), 0.0))
    
    log_pdf = -0.5 * (D_m * jnp.log(2 * jnp.pi) + log_det_S_inv + mahalanobis_sq)
    return log_pdf

# ======================================================================
# II. 单分量更新逻辑 (对齐 Algorithm 3 步骤 23-34)
# ======================================================================

def _update_step_k_l_single_component(
    k_idx, mu_k_t, L_inv_k_t, samples, elite_weights, 
    pi_all, mu_all, L_inv_all, delta_t,
    mu_baseline, L_inv_baseline, D_m
):
    """
    计算单个分量的自然梯度更新量。
    k_idx: 当前分量索引 (0 到 K-1)
    """
    D_max = mu_k_t.shape[0]
    S_k_t = L_inv_k_t @ L_inv_k_t.T
    
    # 1. 计算当前分量、基准分量和整体 MoG 的对数 PDF (向量化)
    log_pdf_k = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_k_t, L_inv_k_t, D_m))(samples)
    log_pdf_base = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_baseline, L_inv_baseline, D_m))(samples)
    
    def mog_pdf_fn(xi):
        # 内部 Vmap 计算所有分量的 PDF
        l_pdfs = vmap(lambda m, l: _gaussian_log_pdf_l_masked(xi, m, l, D_m))(mu_all, L_inv_all)
        return _logsumexp(jnp.log(pi_all + 1e-15) + l_pdfs)
    
    log_mog = vmap(mog_pdf_fn)(samples)

    # 2. 计算 IGO 权重项 alpha_i 和 beta_i (对齐 Algorithm 3)
    a_i = jnp.exp(jnp.clip(log_pdf_k - log_mog, a_min=-70.0, a_max=70.0))
    b_i = jnp.exp(jnp.clip(log_pdf_base - log_mog, a_min=-70.0, a_max=70.0))
    
    # 3. 均值 mu 更新 (步骤 26)
    diff = (samples - mu_k_t)
    mask = (jnp.arange(D_max) < D_m)[:, None]
    # S_k_t @ diff.T
    grad_mu = (S_k_t @ diff.T).T
    sum_mu = jnp.sum( (elite_weights * a_i)[:, None] * grad_mu, axis=0)
    
    # 4. 精度矩阵 S 更新 (步骤 28)
    # 使用 Sigma_k = (S_k)^{-1}
    Sigma_k = jnp.linalg.inv(S_k_t + jnp.eye(D_max)*1e-6)
    
    # 外积展开
    def outer_prod_update(d):
        return Sigma_k @ jnp.outer(d, d) @ Sigma_k - Sigma_k
    
    S_grads = vmap(outer_prod_update)(diff)
    sum_S = jnp.sum((elite_weights * a_i)[:, None, None] * S_grads, axis=0)

    S_new = S_k_t - delta_t * sum_S
    S_new = (S_new + S_new.T) / 2.0 + jnp.eye(D_max) * 1e-6
    L_inv_new = jnp.linalg.cholesky(S_new)

    # 应用更新 (使用 delta_t)
    mu_new = mu_k_t + delta_t * jnp.linalg.solve(S_new , sum_mu)
    
    # 5. 权重 v 更新增量 (步骤 33)
    v_update_val = jnp.sum(elite_weights * (a_i - b_i))

    return mu_new, L_inv_new, v_update_val

# ======================================================================
# III. 主优化流程 (支持 T0 重置与 JAX 并行)
# ======================================================================

def _parallel_step_with_t0_restart(state, iter_data, M, K, B, B0, dt, dims_arr, T_0, fitness_fn, v_reset, context):
    mu, L_inv, v = state
    key, idx = iter_data
    
    # T0 重置逻辑 (周期性重置混合权重，保持多样性)
    v = jnp.where((idx % T_0) == 0, v_reset, v)
    
    # 计算当前的 pi (Softmax, 长度为 K)
    def v_to_pi(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_exps = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_exps, jnp.array([1.0 / sum_exps])])
    
    pi_all = vmap(v_to_pi)(v)

    # 1. 采样阶段 (M 块并行采样)
    def sample_block(m_idx, sub_key):
        # 采样分量索引 (0 到 K-1)
        comps = random.choice(sub_key, K, p=pi_all[m_idx], shape=(B,))
        
        def sample_gaussian(c_idx, s_key):
            m_k = mu[m_idx, c_idx]
            S_k = L_inv[m_idx, c_idx] @ L_inv[m_idx, c_idx].T
            sigma = jnp.linalg.inv(S_k + jnp.eye(S_k.shape[0])*1e-7)
            return m_k + jnp.linalg.cholesky(sigma) @ random.normal(s_key, (m_k.shape[0],))
            
        return vmap(sample_gaussian)(comps, random.split(sub_key, B))

    keys_m = random.split(key, M)
    samples_m = vmap(sample_block)(jnp.arange(M), keys_m) # (M, B, D_max)
    
    # 2. 拼接与评价 (跨块耦合)
    samples_flat = samples_m.transpose(1, 0, 2).reshape(B, -1)
    f_vals = vmap(lambda s: fitness_fn(s, context))(samples_flat)
    
    # 计算精英权重 (IGO 排序逻辑)
    ranks = jnp.argsort(jnp.argsort(f_vals)) # 寻找最大值，故 f 越大 rank 越小
    elite_weights = jnp.where(ranks < B0, 1.0/B, 0.0)

    # 3. 块并行更新
    def update_m(m_idx):
        D_m = dims_arr[m_idx]
        mu_m = mu[m_idx]
        L_m = L_inv[m_idx]
        p_m = pi_all[m_idx]
        
        # 选取第 K 个分量作为 Baseline (Algorithm 3)
        mu_base = mu_m[K-1]
        L_base = L_m[K-1]
        
        # 对 K 个分量并行计算梯度
        res_mu, res_L, res_v_delta = vmap(
            _update_step_k_l_single_component,
            in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None)
        )(jnp.arange(K), mu_m, L_m, samples_m[m_idx], elite_weights, p_m, mu_m, L_m, dt, mu_base, L_base, D_m)
        
        # 只取前 K-1 个 v 的更新量
        return res_mu, res_L, res_v_delta[:K-1]

    new_mu, new_L, v_deltas = vmap(update_m)(jnp.arange(M))
    new_v = jnp.clip(v + dt * v_deltas, -70.0, 70.0)
    
    return (new_mu, new_L, new_v), None

# ======================================================================
# IV. 顶层入口函数
# ======================================================================

@functools.partial(jit, static_argnums=(1, 3, 4, 5, 7, 9))
def mmog_igo_optimizer_mpc(
    key, T, dt, M, K, B, B0, dims, T_0, 
    fitness_fn_total, initial_mu_k, initial_L_inv_k, initial_v_k, context
):
    """
    M: 块数量
    K: 每个 MoG 的分量总数
    dims: 元组，描述各块维度
    """
    dims_array = jnp.array(dims)
    v_reset = jnp.zeros((M, K-1))
    
    # 确保初始化维度正确 (K 个分量)
    mu_init = initial_mu_k[:, :K, :]
    L_init = initial_L_inv_k[:, :K, :, :]
    v_init = jnp.zeros((M, K-1)) 
    state = (mu_init, L_init, v_init)
    keys = random.split(key, T)
    
    # 使用 lax.scan 执行高效循环
    final_state, _ = lax.scan(
        functools.partial(
            _parallel_step_with_t0_restart, 
            M=M, K=K, B=B, B0=B0, dt=dt, 
            dims_arr=dims_array, T_0=T_0, 
            fitness_fn=fitness_fn_total, v_reset=v_reset,
            context=context  # <-- 必须添加这一行，确保 scan 内部可见
        ),
        state, (keys, jnp.arange(T))
    )
    
    # 最终输出结果，并将 v 转换回 pi
    def v_to_pi_final(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    
    final_pi = vmap(v_to_pi_final)(final_state[2])
    
    return final_state[0], final_state[1], final_pi

# JIT 编译
mmog_igo_optimizer_mpc = jit(
    mmog_igo_optimizer_mpc, 
    static_argnums=(1, 3, 4, 5, 7, 9)
)