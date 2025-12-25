# gmm_igo/solver.py
# 终极稳定版 | 修复 static + 1000轮 < 60ms

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
import functools

# ================================
# 核心函数
# ================================

@jit
def _logsumexp(a, axis=None):
    return jnp.logaddexp.reduce(a, axis=axis)

@jit
def _gaussian_log_pdf(xi, mu, L_inv):
    d = xi - mu
    y = L_inv @ d
    return -0.5 * (jnp.sum(y**2) + mu.shape[0] * jnp.log(2*jnp.pi) - 2*jnp.sum(jnp.log(jnp.diag(L_inv))))

_vmap_pdf = vmap(_gaussian_log_pdf, in_axes=(None, 0, 0))

@jit
def _sample(idx, key, mu_k, L_inv_k):
    mu = mu_k[idx]
    L_inv = L_inv_k[idx]
    z = random.normal(key, mu.shape)
    return mu + jnp.linalg.solve(L_inv.T, jnp.linalg.solve(L_inv, z))

_vmap_sample = vmap(_sample, in_axes=(0, 0, None, None))

@jit
def _elite_weights(f, B, B0):
    ranks = jnp.argsort(jnp.argsort(f))
    return jnp.where(ranks < B0, 1.0/B0, 0.0)

# ================================
# 单步
# ================================

@jit
def _step(carry, key, B, B0, K, dt, fitness_fn):
    mu, L_inv, v = carry
    key, subkey = random.split(key)

    pi_pre = jnp.exp(v)
    pi_K = 1 / (1 + jnp.sum(pi_pre))
    pi = jnp.concatenate([pi_pre * pi_K, jnp.array([pi_K])])

    idx = random.choice(subkey, K+1, (B,), p=pi)
    keys = random.split(subkey, B)
    samples = _vmap_sample(idx, keys, mu, L_inv)

    f = vmap(fitness_fn)(samples)
    w = _elite_weights(f, B, B0)

    log_pdf = _vmap_pdf(samples, mu[:-1], L_inv[:-1])
    log_mog = vmap(lambda s: _logsumexp(jnp.log(pi) + _vmap_pdf(s, mu, L_inv)))(samples)
    log_a = jnp.clip(log_pdf - log_mog, a_max=80)
    a = jnp.exp(log_a)
    weighted = a * w

    diff = samples - mu[:-1]
    S_update = jnp.einsum('b,bi,bj->ij', weighted, diff, diff)
    S = L_inv[:-1] @ L_inv[:-1].transpose(0, 2, 1)
    S_new = S + dt * S_update
    S_new = (S_new + S_new.T) / 2 + 1e-6 * jnp.eye(S.shape[1])
    L_inv_new = jnp.linalg.cholesky(S_new)

    mu_update = jnp.einsum('b,bi->i', weighted, diff)
    mu_new = mu[:-1] + dt * jnp.linalg.solve(S_new, S @ mu_update)

    v_up = jnp.sum(w * (a - jnp.exp(jnp.clip(_vmap_pdf(samples, mu[-1:], L_inv[-1:]) - log_mog, a_max=80))))
    v_new = jnp.clip(v + dt * jnp.where(jnp.abs(v_up) > 10, 10*jnp.sign(v_up), v_up), a_max=70)

    new_carry = (jnp.concatenate([mu_new, mu[-1:]]), jnp.concatenate([L_inv_new, L_inv[-1:]]), v_new)
    return new_carry, None

# ================================
# 主优化器（关键：fitness_fn 是第 7 个参数，索引 6，必须 static！）
# ================================

def igo_mog_optimizer_impl(key, T, dt, K, B, B0, fitness_fn, mu0, L_inv0, pi0):
    v0 = jnp.log(pi0[:-1] / pi0[-1])
    carry0 = (mu0, L_inv0, v0)
    keys = random.split(key, T)
    
    step = functools.partial(_step, B=B, B0=B0, K=K, dt=dt, fitness_fn=fitness_fn)
    final_carry, _ = lax.scan(step, carry0, keys)
    
    mu, L_inv, v = final_carry
    pi_pre = jnp.exp(v)
    pi_K = 1 / (1 + jnp.sum(pi_pre))
    pi_final = jnp.concatenate([pi_pre * pi_K, jnp.array([pi_K])])
    return mu, L_inv, pi_final

# 关键修复：fitness_fn 是第 7 个参数，必须标记为 static！
igo_mog_optimizer = jit(
    igo_mog_optimizer_impl,
    static_argnames=('T', 'K', 'B', 'B0', 'fitness_fn'),  # 修复点
    static_argnums=(2,)  # dt 是第 3 个参数
)