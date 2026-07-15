"""圆形赛道 lane 约束发散诊断 + igo_weights 对比。

基于用户提供的复现脚本，添加：
1. 每步 GMM 采样 → cost 可区分性诊断
2. 可选: 用 igo_weights 替换 solver 的 argsort 排名

用法::

    uv run python Cartest/repro_divergence.py --steps 60 [--igo-weights]
"""

import sys, argparse, time, copy
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import jax, jax.numpy as jnp
import numpy as np
from jax import random, vmap

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.core.reference_path import CircularReference
from Cartest.planning.warmstart import tangent_warmstart
from Cartest.execution.execute import execute_perfect_tracking, FrenetState
from Cartest.planning.constraints import _eval_frenet, _eval_vehicle_states
from Constraintdealer.Constran import Deterministic, build as constran_build
from gmm_igo.solver_builder import build_solver
from gmm_igo.igo_weights import check_distinguishability

BASIS = Path('Cartest/basis')
R = 20.0
gen = FrenetBSplineTrajectory(BASIS / 'bspline_basis.npz', CircularReference(R, 0, 0))
n = gen.n_free
lh = 4.0
v_nom = 10.0
ACC_P1 = 5.5
V_MIN, V_MAX = 2.0, 35.0


# ═══════════════════════════════════════════════════════════════════════════
# Cost & constraints (与用户脚本一致)
# ═══════════════════════════════════════════════════════════════════════════

def p1_cost(gen, v_nom=10.0, wd=4.0):
    def f(theta, ctx):
        s, d, sd, dd, _, _, _, _ = gen.evaluate(
            theta[:n], theta[n:2*n],
            ctx['s0'], ctx['s_dot0'], 0, ctx['d0'], 0, 0)
        es = sd - v_nom
        ed = d - 0.0
        v1d = dd + wd * ed
        return jnp.mean(es**2) + jnp.mean(ed**2 + v1d**2)
    return f


def og(t, c):
    no = c['obs_pos'].shape[0]
    if no == 0:
        return jnp.zeros(gen.T)
    st = _eval_vehicle_states(t, c, gen)
    x, y, v = st[:, 0], st[:, 1], st[:, 2]
    dr = v * 0.1 + v**2 / 16.0
    dx = x[:, None] - c['obs_pos'][None, :, 0]
    dy = y[:, None] - c['obs_pos'][None, :, 1]
    r = c['obs_rad'][None, :]
    return jnp.maximum(
        jnp.maximum(0., dr[:, None] + r - jnp.abs(dx)),
        jnp.maximum(0., r - jnp.abs(dy)),
    ).max(axis=-1)


def lg(t, c):
    _, d, _, _, _, _, _, _ = _eval_frenet(t, c, gen)
    return jnp.maximum(0., jnp.abs(d) - lh)


def sg(t, c):
    v = _eval_vehicle_states(t, c, gen)[:, 2]
    return jnp.maximum(jnp.maximum(0., V_MIN - v), jnp.maximum(0., v - V_MAX))


def ag(t, c):
    st = _eval_vehicle_states(t, c, gen)
    al, at = st[:, 4], st[:, 5]
    am = jnp.sqrt(al**2 + at**2)
    return jnp.maximum(
        jnp.maximum(0., jnp.abs(al) - ACC_P1),
        jnp.maximum(jnp.maximum(0., jnp.abs(at) - ACC_P1),
                    jnp.maximum(0., am - ACC_P1)),
    )


p1_constraints = [
    Deterministic(lg, mode='hard', priority=1, aggregate='q95', transform='hard'),
    Deterministic(og, mode='hard', priority=2, aggregate='max', transform='hard'),
    Deterministic(ag, mode='tunable', priority=3, aggregate='max', transform='tunable'),
    Deterministic(sg, mode='soft', priority=4, aggregate='max', transform='soft'),
]


def lm(gen, s0, sd0):
    cs, _ = tangent_warmstart(gen, s0, sd0, 0.0)
    sl, dl = [], []
    for dv in [-3., 0., 3.]:
        sl.append(cs)
        dl.append(jnp.full(n, dv))
    return jnp.stack([jnp.stack(sl), jnp.stack(dl)]).astype(jnp.float32)


def sample_from_gmm(rng, mu, L_inv, pi, n_samples):
    """从 solver 的 GMM 采样 (复用 solver 采样逻辑)。"""
    M, K, D = mu.shape
    total_dim = M * D
    samples = jnp.zeros((n_samples, total_dim))
    for m in range(M):
        rng, sk = random.split(rng)
        comps = random.choice(sk, K, p=pi[m], shape=(n_samples,))
        S_all = L_inv[m] @ L_inv[m].transpose(0, 2, 1)
        for k in range(K):
            mask = comps == k
            n_k = int(jnp.sum(mask))
            if n_k == 0:
                continue
            rng, sk2 = random.split(rng)
            cov = jnp.linalg.inv(S_all[k] + jnp.eye(D) * 1e-7)
            z = random.multivariate_normal(sk2, mu[m, k], cov, shape=(n_k,))
            start = m * D
            samples = samples.at[jnp.where(mask)[0], start:start+D].set(z)
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run(steps=60, seed=42, use_igo=False, diag_samples=200):
    op = jnp.zeros((0, 2), dtype=jnp.float32)
    od = jnp.zeros(0, dtype=jnp.float32)

    # Normal solver (argsort)
    if use_igo:
        print("=== 使用 igo_weights (monkey-patch solver) ===")
        _patch_solver_ranking()
    else:
        print("=== 使用原始 argsort 排名 ===")

    solver = build_solver(
        p1_cost(gen), dims=(n, n), constraints=p1_constraints,
        solver='m22', T=500, dt=0.25, K=3, B=128, B0=50, T_0=200,
        k_inner=1.0, obj_transform='standard',
    )

    # 外部 cost 函数用于诊断
    diag_cost = constran_build(
        p1_cost(gen), p1_constraints,
        k_inner=1.0, obj_transform='standard',
    )

    key = random.PRNGKey(seed)
    st = FrenetState(s=0., s_dot=10., s_ddot=0., d=-3., d_dot=0., d_ddot=0., psi=0.)
    cw = {'s0': 0., 's_dot0': 10., 's_ddot0': 0., 'd0': -3., 'd_dot0': 0.,
          'd_ddot0': 0., 'v_ref': jnp.array([v_nom]), 'lane_hw': lh,
          'obs_pos': op, 'obs_rad': od}
    _ = solver(random.PRNGKey(999), context=cw, initial_mu=lm(gen, 0., 10.))

    header = (f"{'Step':>4s}  {'d':>10s}  {'v':>7s}  "
              f"{'cost_range':>12s}  {'tied':>8s}  {'note':>10s}")
    print(header)
    print("-" * len(header))

    for step in range(steps):
        key, sk = random.split(key)
        c = {'s0': float(st.s), 's_dot0': float(st.s_dot), 's_ddot0': 0.,
             'd0': float(st.d), 'd_dot0': 0., 'd_ddot0': 0.,
             'v_ref': jnp.array([v_nom]), 'lane_hw': lh,
             'obs_pos': op, 'obs_rad': od}

        r = solver(sk, context=c, initial_mu=lm(gen, st.s, st.s_dot))

        # ── 诊断: GMM 采样 + cost 可区分性 ──
        dk = random.PRNGKey(step * 997 + seed * 1009)  # 独立 key, 不干扰 solver
        xs = sample_from_gmm(dk, r.mu, r.S_or_L, r.pi, diag_samples)
        fv = vmap(lambda x: diag_cost(x, c))(xs)
        diag = check_distinguishability(fv)
        fr = float(jnp.max(fv) - jnp.min(fv))

        # 执行
        s, d, sd, dd, sdd, ddd, _, _ = gen.evaluate(
            r.x[:n], r.x[n:2*n], c['s0'], c['s_dot0'], 0, c['d0'], 0, 0)
        stv = gen.to_vehicle_states(s, d, sd, dd, sdd, ddd, _, _)
        st_prev = st
        st = execute_perfect_tracking(s, d, sd, dd, sdd, ddd, stv[1, 3])

        note = ""
        if diag['frac_tied'] > 0.3:
            note = "!!PLATEAU"
        elif diag['frac_tied'] > 0.1:
            note = "⚠ flat"

        print(f"{step:4d}  {float(st.d):+10.2f}  {float(st.s_dot):7.1f}  "
              f"{fr:12.3e}  {diag['n_tied']:4d}/{diag_samples}  {note:>10s}")

        if abs(float(st.d)) > 100.0:
            print(f"\n!!! EXPLOSION at step {step}: d={float(st.d):.2e} !!!")
            break


def _patch_solver_ranking():
    """Monkey-patch MPCsolverM22 的 ranking 为 compute_elite_weights。

    通过替换 solver_builder._get_m22 返回包装后的函数来实现。
    """
    from gmm_igo import MPCsolverM22 as _m22
    from gmm_igo.igo_weights import compute_elite_weights

    _orig_step_fn = _m22._step_fn

    def _patched_step_fn(state, iter_data, M, K, B, B0, dt, dims_arr,
                         T_0, fitness_fn, v_reset, context):
        """与原始 _step_fn 完全一致，仅替换 ranking 部分。"""
        # 复制原始 _step_fn 的逻辑，但用 igo_weights 替换 argsort
        mu, S, v, t = state
        key, _ = iter_data

        v = jnp.where((t > 0) & ((t % T_0) == 0), v_reset, v)

        def v_to_pi(v_m):
            exps = jnp.exp(jnp.clip(v_m, -70, 70))
            sum_e = 1.0 + jnp.sum(exps)
            return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])

        pi_all = _m22.vmap(v_to_pi)(v)  # use original vmap

        # 采样 (复用原始逻辑)
        def sample_block(m_idx, sub_key):
            comps = random.choice(sub_key, K, p=pi_all[m_idx], shape=(B,))
            def gen_sample(c_idx, s_key):
                cov = jnp.linalg.inv(S[m_idx, c_idx] + jnp.eye(S.shape[-1]) * 1e-7)
                return random.multivariate_normal(s_key, mu[m_idx, c_idx], cov)
            return _m22.vmap(gen_sample)(comps, random.split(sub_key, B))

        samples_m = _m22.vmap(sample_block)(jnp.arange(M), random.split(key, M))
        samples_flat = samples_m.transpose(1, 0, 2).reshape(B, -1)
        f_vals = _m22.vmap(lambda s: fitness_fn(s, context))(samples_flat)

        # ══ PATCHED: 用 compute_elite_weights 替换 argsort ══
        w_hat = compute_elite_weights(f_vals, B0)

        # 块更新 (复用原始逻辑)
        def update_block(m_idx):
            D_m = dims_arr[m_idx]
            mu_base, S_base = mu[m_idx, K - 1], S[m_idx, K - 1]
            new_mu_m, new_S_m, v_deltas = _m22.vmap(
                _m22._update_component_core,
                in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None)
            )(jnp.arange(K), mu[m_idx], S[m_idx], samples_m[m_idx], w_hat,
              pi_all[m_idx], mu[m_idx], S[m_idx], dt, mu_base, S_base, D_m)
            return new_mu_m, new_S_m, v_deltas[:K - 1]

        next_mu, next_S, next_v_deltas = _m22.vmap(update_block)(jnp.arange(M))
        next_v = jnp.clip(v + dt * next_v_deltas, -70.0, 70.0)
        return (next_mu, next_S, next_v, t + 1), None

    # Replace in solver_builder's getter
    import gmm_igo.solver_builder as _sb
    _orig_get = _sb._get_m22

    def _patched_get():
        import functools
        # Return a patched version of mmog_igo_optimizer_mpc
        orig_opt = _orig_get()
        # We can't easily patch the inner _step_fn since it's captured in
        # the closure of mmog_igo_optimizer_mpc at jit time.
        # Instead, rebuild the optimizer with the patched step function.
        @functools.partial(jax.jit, static_argnums=(1, 3, 4, 5, 7, 9))
        def patched_opt(key, T, dt, M, K, B, B0, dims, T_0,
                        fitness_fn_total, initial_mu_k, initial_L_inv_k,
                        initial_v_k, context):
            dims_array = jnp.array(dims)
            v_reset = jnp.zeros((M, K - 1))
            S_init = _m22.vmap(_m22.vmap(lambda L: L @ L.T))(initial_L_inv_k[:, :K, :, :])
            mu_init = initial_mu_k[:, :K, :]
            v_init = initial_v_k
            state = (mu_init, S_init, v_init, 0)
            loop_fn = functools.partial(
                _patched_step_fn, M=M, K=K, B=B, B0=B0, dt=dt,
                dims_arr=dims_array, T_0=T_0,
                fitness_fn=fitness_fn_total, v_reset=v_reset, context=context)
            final_state, _ = jax.lax.scan(loop_fn, state,
                                          (random.split(key, T), jnp.arange(T)))

            def v_to_pi_final(v_m):
                exps = jnp.exp(jnp.clip(v_m, -70, 70))
                return jnp.concatenate([exps / (1.0 + jnp.sum(exps)),
                                        jnp.array([1.0 / (1.0 + jnp.sum(exps))])])
            final_pi = _m22.vmap(v_to_pi_final)(final_state[2])
            final_L = _m22.vmap(_m22.vmap(jnp.linalg.cholesky))(final_state[1])
            return final_state[0], final_L, final_pi
        return patched_opt

    _sb._get_m22 = _patched_get
    print("  [patched: MPCsolverM22 ranking → compute_elite_weights]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--igo-weights", action="store_true")
    p.add_argument("--diag-samples", type=int, default=200)
    args = p.parse_args()
    run(steps=args.steps, seed=args.seed, use_igo=args.igo_weights,
        diag_samples=args.diag_samples)
