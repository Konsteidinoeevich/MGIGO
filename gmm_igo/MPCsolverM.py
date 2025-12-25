# gmm_igo/MPCsolver_Hetero.py - 支持异构维度的 M-MoG IGO 优化器核心

import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools
from typing import Callable, Tuple, Any, List

# ======================================================================
# I. 核心辅助函数 (基于信息矩阵 S_k 范式)
# ======================================================================

@jit
def _logsumexp(a, axis=None):
    return jnp.logaddexp.reduce(a, axis=axis)

@jit
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
_vmap_gaussian_log_pdf_l_samples = vmap(_gaussian_log_pdf_l, in_axes=(0, None, None)) 

@jit
def _mixture_log_pdf_l(xi, mu_k, L_inv_k, pi_k):
    """计算混合高斯 Log-PDF。"""
    log_pdfs_k = _vmap_gaussian_log_pdf_l_k(xi, mu_k, L_inv_k)
    log_pi_k = jnp.log(pi_k)
    log_weighted_pdfs = log_pi_k + log_pdfs_k
    return _logsumexp(log_weighted_pdfs)

@jit
def _sample_from_component_l(idx, key_sample, mu_k_all, L_inv_k_all):
    """从 N(mu, S^{-1}) 中采样。"""
    mu_k = mu_k_all[idx]
    S_k = L_inv_k_all[idx] @ L_inv_k_all[idx].T 
    L_Sigma = jnp.linalg.cholesky(jnp.linalg.solve(S_k, jnp.eye(S_k.shape[0])))
    D = mu_k.shape[0]
    z = random.normal(key_sample, shape=(D,))
    return mu_k + L_Sigma @ z

_vmap_sample_from_component_l = vmap(_sample_from_component_l, in_axes=(0, 0, None, None))

# ======================================================================
# II. M-MoG 采样 (Static Dispatch)
# ======================================================================

@functools.partial(jit, static_argnames=['K', 'B', 'D_max', 'D_static'])
def _mog_sampler_static(carry_m, key_m, K: int, B: int, D_max: int, D_static: int):
    """ 维度 D_static 静态编译的采样函数。"""
    mu_m, L_inv_m, v_m = carry_m
    
    pi_pre = jnp.exp(v_m) 
    pi_K = 1 / (1 + jnp.sum(pi_pre))
    pi = jnp.concatenate([pi_pre * pi_K, jnp.array([pi_K])]) 
    
    key_comp, key_samples = random.split(key_m)
    comp_indices = random.choice(key_comp, K+1, (B,), p=pi) 
    sample_keys = random.split(key_samples, B) 
    
    # 关键：从 D_max 维度切片到 D_static 维度进行实际计算
    mu_m_actual = mu_m[:, :D_static]
    L_inv_m_actual = L_inv_m[:, :D_static, :D_static]
    
    samples_m_actual = _vmap_sample_from_component_l(
        comp_indices, sample_keys, mu_m_actual, L_inv_m_actual
    ) 
    # 关键：结果填充回 D_max 维度
    samples_m_padded = jnp.pad(samples_m_actual, ((0, 0), (0, D_max - D_static)), mode='constant') 
    return samples_m_padded

def _mog_sampler_dispatch(carry_m, key_m, K, B, D_max, D_m_tracer):
    """ 使用 lax.switch 进行维度分派。"""
    
    branches = [
        functools.partial(_mog_sampler_static, D_static=d+1, K=K, B=B, D_max=D_max)
        for d in range(D_max) 
    ]
    # D_m_tracer 是 JAX 数组，索引是 D_m - 1
    index = jnp.clip(D_m_tracer - 1, a_min=0, a_max=D_max - 1)
    
    return lax.switch(
        index,
        branches,
        carry_m, key_m
    )


def _sequential_mog_sampler(carry_M, key_samples_M, K, B, D_max, dims_array, M):
    """ M 个 MoG 的顺序采样，处理异构维度。"""
    mu_M, L_inv_M, v_M = carry_M
    samples_M_padded = jnp.zeros((M, B, D_max)) 
    
    def scan_body(m, samples_M_padded):
        key_m = key_samples_M[m]
        carry_m = (mu_M[m], L_inv_M[m], v_M[m])
        D_m = dims_array[m] # JAX 数组的维度
        
        samples_m_padded_new = _mog_sampler_dispatch(carry_m, key_m, K, B, D_max, D_m)
        
        return samples_M_padded.at[m].set(samples_m_padded_new)

    final_samples = lax.fori_loop(0, M, scan_body, samples_M_padded)
    return final_samples


def _get_overall_elite_weights(samples_M: jnp.ndarray, fitness_fn_total: Callable, 
                               B: int, B0: int, context: Any, M: int, D_max: int):
    """评估 f(xi, context) 并计算精英样本权重。"""
    # 样本汇聚: (M, B, D_max) -> (B, M * D_max)
    samples_overall = jnp.transpose(samples_M, (1, 0, 2)).reshape((B, M * D_max))
    
    # 传入 context (context 必须携带 dims_array 来让 fitness_fn 知道如何切片)
    f_xi = vmap(fitness_fn_total, in_axes=(0, None))(samples_overall, context) 
    
    ranks = jnp.argsort(jnp.argsort(f_xi))
    is_elite = ranks < B0
    return jnp.where(is_elite, 1.0/B, 0.0) 

# ======================================================================
# III. K 个分量维度感知更新逻辑 (Static Dispatch)
# ======================================================================

LOG_CLIP_VALUE = 80.0
@functools.partial(jit, static_argnames=('D_m',))
def _update_step_k_l_single_component_dim_aware(
    k_idx, mu_k_t, L_inv_k_t, samples, elite_weights, 
    pi_k_all, mu_k_all, L_inv_k_all, delta_t,
    mu_K_t, L_inv_K_t, D_m: int # 这里的 D_m 必须是 Python int
):
    """单个分量 k 的 IGO 自然梯度更新 (D_m 维度)。"""
    
    S_k_t = L_inv_k_t @ L_inv_k_t.T 
    
    log_norm_pdf_k = _vmap_gaussian_log_pdf_l_samples(samples, mu_k_t, L_inv_k_t) 
    log_norm_pdf_K = _vmap_gaussian_log_pdf_l_samples(samples, mu_K_t, L_inv_K_t)
    
    log_mog_xi = vmap(_mixture_log_pdf_l, in_axes=(0, None, None, None))(
        samples, mu_k_all, L_inv_k_all, pi_k_all
    )

    log_a_i = jnp.clip(log_norm_pdf_k - log_mog_xi, a_max=LOG_CLIP_VALUE)
    log_b_i = jnp.clip(log_norm_pdf_K - log_mog_xi, a_max=LOG_CLIP_VALUE)
    a_i = jnp.exp(log_a_i)
    b_i = jnp.exp(log_b_i)
    scaled_a_i = a_i * elite_weights
    
    # S_k 更新
    diff = samples - mu_k_t
    diff_outer = vmap(lambda x: jnp.outer(x, x))(diff)
    Sigma_k_t = jnp.linalg.solve(S_k_t, jnp.eye(D_m)) 
    S_update_term_i = Sigma_k_t @ diff_outer @ Sigma_k_t - Sigma_k_t[None, :, :]
    sum_S_update = jnp.sum(scaled_a_i[:, None, None] * S_update_term_i, axis=0)
    S_k_t_plus_1_prop = S_k_t - delta_t * sum_S_update
    S_k_t_plus_1_prop = (S_k_t_plus_1_prop + S_k_t_plus_1_prop.T) / 2
    S_k_t_plus_1_prop = S_k_t_plus_1_prop + jnp.eye(D_m) * 1e-12 
    L_inv_k_t_plus_1 = jnp.linalg.cholesky(S_k_t_plus_1_prop)

    # mu_k 更新
    weighted_diff = scaled_a_i[None, :] * diff.T 
    sum_term_vector = jnp.sum(S_k_t @ weighted_diff, axis=1) 
    mu_update_term = jnp.linalg.solve(L_inv_k_t_plus_1 @ L_inv_k_t_plus_1.T, sum_term_vector)
    mu_k_t_plus_1 = mu_k_t + delta_t * mu_update_term

    # 权重更新项 (v_k)
    v_update_sum = jnp.sum(elite_weights * (a_i - b_i))
    
    return mu_k_t_plus_1, L_inv_k_t_plus_1, v_update_sum

_vmap_update_step_k_l = vmap(
    _update_step_k_l_single_component_dim_aware, 
    in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None), 
    out_axes=(0, 0, 0)
)

@functools.partial(jit, static_argnames=['K', 'dt', 'D_max', 'D_static'])
def _mog_updater_static(carry_m, key_m, elite_weights, samples_m_padded, D_max: int, K: int, dt: float, D_static: int):
    """ 维度 D_static 静态编译的更新函数。"""
    mu_m, L_inv_m, v_m = carry_m
    
    # 关键：切片到 D_static
    mu_m_actual = mu_m[:, :D_static]
    L_inv_m_actual = L_inv_m[:, :D_static, :D_static]
    samples_m_actual = samples_m_padded[:, :D_static]

    pi_pre = jnp.exp(v_m)
    pi_K = 1 / (1 + jnp.sum(pi_pre))
    pi_k_t_all = jnp.concatenate([pi_pre * pi_K, jnp.array([pi_K])]) 
    k_indices = jnp.arange(K) 
    
    mu_K_t_unbatched = mu_m_actual[K, :]
    L_inv_K_t_unbatched = L_inv_m_actual[K, :, :]
    
    mu_new_k_actual, L_inv_new_k_actual, v_update_sum_k = _vmap_update_step_k_l(
        k_indices, mu_m_actual[:K], L_inv_m_actual[:K], samples_m_actual, elite_weights, 
        pi_k_t_all, mu_m_actual, L_inv_m_actual, dt,
        mu_K_t_unbatched, L_inv_K_t_unbatched, D_static
    ) 

    # 权重更新 
    v_update_vec = v_update_sum_k 
    MAX_V_UPDATE = 10.0
    v_update_norm = jnp.linalg.norm(v_update_vec)
    v_update_safe = jnp.where(v_update_norm > MAX_V_UPDATE, v_update_vec * (MAX_V_UPDATE / v_update_norm), v_update_vec)
    v_new = jnp.clip(v_m + dt * v_update_safe, a_max=70.0)

    # 关键：结果填充回 D_max
    mu_new_k_padded = jnp.pad(mu_new_k_actual, ((0, 0), (0, D_max - D_static)), mode='constant')
    L_inv_new_k_padded = jnp.pad(L_inv_new_k_actual, ((0, 0), (0, D_max - D_static), (0, D_max - D_static)), mode='constant')
    
    # K+1 分量 (背景分量) 是不变的
    mu_final = jnp.concatenate([mu_new_k_padded, mu_m[K:]]) 
    L_inv_final = jnp.concatenate([L_inv_new_k_padded, L_inv_m[K:]])
    
    return mu_final, L_inv_final, v_new

def _mog_updater_dispatch(carry_m, key_m, elite_weights, samples_m_padded, D_max, D_m_tracer, K, dt):
    """ 使用 lax.switch 进行维度分派。"""
    
    branches = [
        functools.partial(_mog_updater_static, D_static=d+1, K=K, dt=dt, D_max=D_max)
        for d in range(D_max)
    ]
    
    index = jnp.clip(D_m_tracer - 1, a_min=0, a_max=D_max - 1)
    
    return lax.switch(
        index,
        branches,
        carry_m, key_m, elite_weights, samples_m_padded
    )


def _sequential_mog_updater(carry_M, keys_updater_M, elite_weights, samples_M_padded, D_max, dims_array, K, dt, M):
    """ M 个 MoG 的顺序更新，处理异构维度。"""
    
    def scan_body(m, carry_M):
        mu_M, L_inv_M, v_M = carry_M
        
        key_m = keys_updater_M[m]
        samples_m_padded_m = samples_M_padded[m]
        D_m = dims_array[m]
        carry_m = (mu_M[m], L_inv_M[m], v_M[m])
        
        mu_m_post, L_inv_m_post, v_m_post = _mog_updater_dispatch(
            carry_m, key_m, elite_weights, samples_m_padded_m, 
            D_max, D_m, K, dt
        )
        
        mu_M = mu_M.at[m].set(mu_m_post)
        L_inv_M = L_inv_M.at[m].set(L_inv_m_post)
        v_M = v_M.at[m].set(v_m_post)
        
        return (mu_M, L_inv_M, v_M)
    
    final_carry_M = lax.fori_loop(0, M, scan_body, carry_M)
    return final_carry_M


# ======================================================================
# IV. M-MoG 迭代步 和 主优化器函数
# ======================================================================

def _parallel_step_with_t0_restart(
    carry, t, 
    M, K, B, B0, dt, dims_array, D_max, T_0, 
    initial_L_inv_k_static, initial_v_k_static, 
    fitness_fn_total, context
):
    """支持 T0 周期性重初始化和异构维度处理的迭代步。"""
    mu_M, L_inv_M, v_M = carry 
    
    # 1. 周期性重置检查 (使用 static 初始值)
    do_restart = (T_0 > 1) & ((t % T_0) == 0)
    new_L_inv_M = jnp.where(do_restart, initial_L_inv_k_static, L_inv_M)
    new_v_M = jnp.where(do_restart, initial_v_k_static, v_M)
    current_carry = (mu_M, new_L_inv_M, new_v_M)
    
    # 2. 生成 Key
    key_input = random.PRNGKey(t) 
    key_M, subkey = random.split(key_input)
    
    # 3. 采样 (处理异构维度)
    key_samples_M = random.split(subkey, M) 
    samples_M_padded = _sequential_mog_sampler(
        current_carry, key_samples_M, K, B, D_max, dims_array, M
    ) 

    # 4. 整体精英权重 (Context-aware, 传入 M, D_max)
    elite_weights = _get_overall_elite_weights(
        samples_M_padded, fitness_fn_total, B, B0, context, M, D_max
    )
    
    # 5. M 个 MoG 更新 (处理异构维度)
    keys_updater_M = random.split(subkey, M) 
    mu_post, L_inv_post, v_post = _sequential_mog_updater(
        current_carry, keys_updater_M, elite_weights, samples_M_padded, 
        D_max, dims_array, K, dt, M
    )
    
    new_carry = (mu_post, L_inv_post, v_post)
    return new_carry, None

def _compute_pi(v_m):
    """计算最终混合权重 pi (K+1 个分量)"""
    pi_pre = jnp.exp(v_m) 
    pi_K = 1 / (1 + jnp.sum(pi_pre))
    return jnp.concatenate([pi_pre * pi_K, jnp.array([pi_K])])

def mmog_igo_optimizer_mpc_impl(
    key: jnp.ndarray, T: int, dt: float, M: int, K: int, B: int, B0: int, 
    dims: Tuple[int, ...], # <--- 异构维度元组
    T_0: int, 
    fitness_fn_total: Callable,
    initial_mu_k: jnp.ndarray, initial_L_inv_k: jnp.ndarray, initial_v_k: jnp.ndarray,
    context: Any 
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    D_max: int = max(dims) # 计算最大维度
    dims_array = jnp.array(dims) # 转换为 JAX 数组

    initial_state = (initial_mu_k, initial_L_inv_k, initial_v_k)
    
    # 初始重启状态也使用填充后的数组
    initial_L_inv_k_static = initial_L_inv_k 
    initial_v_k_static = jnp.zeros((M, K)) 
    
    t_iterations = jnp.arange(T)
    
    bound_iteration_step = functools.partial(
        _parallel_step_with_t0_restart, 
        M=M, K=K, B=B, B0=B0, dt=dt, 
        dims_array=dims_array, # 传入 JAX 数组
        D_max=D_max, 
        T_0=T_0,
        initial_L_inv_k_static=initial_L_inv_k_static,
        initial_v_k_static=initial_v_k_static,
        fitness_fn_total=fitness_fn_total,
        context=context 
    )

    final_carry, _ = lax.scan(bound_iteration_step, initial_state, t_iterations)
    
    final_mu_M, final_L_inv_M, final_v_M = final_carry
    
    final_pi_M = vmap(_compute_pi)(final_v_M) 
    
    return final_mu_M, final_L_inv_M, final_pi_M

# JIT 编译
mmog_igo_optimizer_mpc = jit(
    mmog_igo_optimizer_mpc_impl, 
    static_argnames=(
        'T', 'dt', 'M', 'K', 'B', 'B0', 'dims', 'T_0', 
        'fitness_fn_total'
    )
)