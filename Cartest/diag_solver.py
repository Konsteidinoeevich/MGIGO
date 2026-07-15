"""诊断版 solver: 捕获爆炸瞬间的求解器内部状态。

直接复制 MPCsolverM22._step_fn, 添加 mu 监控和 NaN 检测。
不修改原文件——创建独立副本。
"""

import jax, jax.numpy as jnp
from jax import vmap, random, lax
import functools

MIN_EIG, MAX_EIG = 1e-3, 1e3

def _safe_spd_projection(S):
    eigvals, eigvecs = jnp.linalg.eigh(S)
    eigvals = jnp.maximum(eigvals, MIN_EIG)
    eigvals = jnp.minimum(eigvals, MAX_EIG)
    return eigvecs @ (eigvals[:, None] * eigvecs.T)

@jax.jit
def _logsumexp(a, axis=None):
    return jnp.logaddexp.reduce(a, axis=axis)

@jax.jit
def _project_spd(mat, eps=1e-3, max_eig=1e3):
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = jnp.linalg.eigh(sym)
    eigvals = jnp.clip(eigvals, eps, max_eig)
    return (eigvecs * eigvals) @ eigvecs.T

@jax.jit
def _gaussian_log_pdf_l_masked(xi, mu, S, D_m):
    diff = (xi - mu)
    mask = jnp.arange(xi.shape[0]) < D_m
    diff = diff * mask
    mahalanobis_sq = jnp.dot(diff, jnp.dot(S, diff))
    sign, logdet_S = jnp.linalg.slogdet(S)
    log_pdf = -0.5 * (D_m * jnp.log(2 * jnp.pi) - logdet_S + mahalanobis_sq)
    return log_pdf


def _update_component_core(k_idx, mu_k, S_k, samples, elite_weights,
                            pi_all, mu_all, S_all, delta_t,
                            mu_base, S_base, D_m):
    D_max = mu_k.shape[0]
    S_k = _project_spd(S_k)

    log_pdf_k = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_k, S_k, D_m))(samples)
    log_pdf_base = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_base, S_base, D_m))(samples)

    def mog_pdf_fn(xi):
        l_pdfs = vmap(lambda m, s: _gaussian_log_pdf_l_masked(xi, m, s, D_m))(mu_all, S_all)
        return _logsumexp(jnp.log(pi_all + 1e-15) + l_pdfs)

    log_mog = vmap(mog_pdf_fn)(samples)

    a_i = jnp.exp(jnp.clip(log_pdf_k - log_mog, -20.0, 20.0))
    b_i = jnp.exp(jnp.clip(log_pdf_base - log_mog, -20.0, 20.0))

    diff = (samples - mu_k)
    def s_grad_fn(d):
        return S_k @ jnp.outer(d, d) @ S_k - S_k

    sum_S_grad = jnp.sum((elite_weights * a_i)[:, None, None] * vmap(s_grad_fn)(diff), axis=0)
    S_new = S_k - delta_t * sum_S_grad
    S_new = (S_new + S_new.T) / 2.0
    S_new = _safe_spd_projection(S_new)

    grad_mu_terms = (S_k @ diff.T).T
    sum_mu_grad = jnp.sum((elite_weights * a_i)[:, None] * grad_mu_terms, axis=0)
    mu_new = mu_k + delta_t * jnp.linalg.solve(S_new, sum_mu_grad)

    v_delta = jnp.sum(elite_weights * (a_i - b_i))
    return mu_new, S_new, v_delta


def make_diag_step_fn(verbose_every=50):
    """返回带诊断的 _step_fn。每 verbose_every 步打印 mu/S 状态。"""

    def _step_fn(state, iter_data, M, K, B, B0, dt, dims_arr,
                 T_0, fitness_fn, v_reset, context):
        mu, S, v, t = state
        key, _ = iter_data

        v = jnp.where((t > 0) & ((t % T_0) == 0), v_reset, v)

        def v_to_pi(v_m):
            exps = jnp.exp(jnp.clip(v_m, -70, 70))
            sum_e = 1.0 + jnp.sum(exps)
            return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])

        pi_all = vmap(v_to_pi)(v)

        # 采样
        def sample_block(m_idx, sub_key):
            comps = random.choice(sub_key, K, p=pi_all[m_idx], shape=(B,))
            def gen_sample(c_idx, s_key):
                cov = jnp.linalg.inv(S[m_idx, c_idx] + jnp.eye(S.shape[-1]) * 1e-7)
                return random.multivariate_normal(s_key, mu[m_idx, c_idx], cov)
            return vmap(gen_sample)(comps, random.split(sub_key, B))

        samples_m = vmap(sample_block)(jnp.arange(M), random.split(key, M))
        samples_flat = samples_m.transpose(1, 0, 2).reshape(B, -1)
        f_vals = vmap(lambda s: fitness_fn(s, context))(samples_flat)

        # 排名 (原始 argsort)
        ranks = jnp.argsort(jnp.argsort(f_vals))
        w_hat = jnp.where(ranks < B0, 1.0 / B, 0.0)

        # 块更新
        def update_block(m_idx):
            D_m = dims_arr[m_idx]
            mu_base, S_base = mu[m_idx, K - 1], S[m_idx, K - 1]
            new_mu_m, new_S_m, v_deltas = vmap(
                _update_component_core,
                in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None)
            )(jnp.arange(K), mu[m_idx], S[m_idx], samples_m[m_idx], w_hat,
              pi_all[m_idx], mu[m_idx], S[m_idx], dt, mu_base, S_base, D_m)
            return new_mu_m, new_S_m, v_deltas[:K - 1]

        next_mu, next_S, next_v_deltas = vmap(update_block)(jnp.arange(M))
        next_v = jnp.clip(v + dt * next_v_deltas, -70.0, 70.0)

        # 诊断 (使用 jax.lax.cond 在 JIT 内安全打印)
        def _print_diag(_):
            jax.debug.print("  t={t}: mu_d=[{m0},{m1},{m2}] S_eig=[{s0},{s1},{s2}] "
                            "v=[{v0},{v1}] f_range=[{fr0},{fr1}]",
                            t=t,
                            m0=mu[1,0,0], m1=mu[1,1,0], m2=mu[1,2,0],
                            s0=jnp.linalg.eigh(S[1,0])[0][-1],
                            s1=jnp.linalg.eigh(S[1,1])[0][-1],
                            s2=jnp.linalg.eigh(S[1,2])[0][-1],
                            v0=v[1,0], v1=v[1,1],
                            fr0=jnp.min(f_vals), fr1=jnp.max(f_vals))
        jax.lax.cond(t % verbose_every == 0, _print_diag, lambda _: None, None)

        return (next_mu, next_S, next_v, t + 1), None

    return _step_fn


def diag_optimizer(key, T, dt, M, K, B, B0, dims, T_0,
                   fitness_fn_total, initial_mu_k, initial_L_inv_k,
                   initial_v_k, context, verbose_every=50):
    """带诊断的 mmog_igo_optimizer_mpc 替代版。"""
    dims_array = jnp.array(dims)
    v_reset = jnp.zeros((M, K - 1))
    S_init = vmap(vmap(lambda L: L @ L.T))(initial_L_inv_k[:, :K, :, :])
    mu_init = initial_mu_k[:, :K, :]
    v_init = initial_v_k
    state = (mu_init, S_init, v_init, 0)

    diag_step = make_diag_step_fn(verbose_every)
    loop_fn = functools.partial(
        diag_step, M=M, K=K, B=B, B0=B0, dt=dt,
        dims_arr=dims_array, T_0=T_0,
        fitness_fn=fitness_fn_total, v_reset=v_reset, context=context)

    final_state, _ = lax.scan(loop_fn, state, (random.split(key, T), jnp.arange(T)))

    def v_to_pi_final(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        return jnp.concatenate([exps / (1.0 + jnp.sum(exps)),
                                jnp.array([1.0 / (1.0 + jnp.sum(exps))])])
    final_pi = vmap(v_to_pi_final)(final_state[2])
    final_L = vmap(vmap(jnp.linalg.cholesky))(final_state[1])
    return final_state[0], final_L, final_pi


if __name__ == "__main__":
    # 快速自测
    print("diag_solver.py — import and use via monkey-patch")
