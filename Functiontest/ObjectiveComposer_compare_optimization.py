"""
Head-to-head: Dynamic k vs Static k vs No Composer (raw weights)
=================================================================

Compares three approaches on the same optimization problem, tracking
best-cost-per-iteration to reveal convergence behavior.

Approaches:
  1. 'dynamic'  — k in ctx, updated from elite each solver call
  2. 'static'   — compose_objective with fixed k (calibrated once)
  3. 'raw'      — traditional weighted sum, wrapped in σ(T(·))

The solver runs the same IGO for the same number of iterations.
"""

import sys, os
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

import jax, jax.numpy as jnp, numpy as np
from jax import random, jit, vmap
import time

from Constraintdealer.Constran import sigma_k, log_transform
from Constraintdealer.ObjectiveComposer import (
    compose_objective, compose_objective_dynamic,
    calibrate_k_into_ctx, update_k_from_elite,
    knee_to_k, k_to_knee
)
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc


# ===========================================================================
# 1. Problem Setup
# ===========================================================================

CONFIG = {
    'dt': 0.15, 'horizon': 12, 'max_force': 4.0,
    'n_components': 4, 'pop_size': 80, 'elite_size': 35,
    'warmup_steps': 300, 'opt_steps': 500,
    't0_reset': 150,
}

@jit
def rollout(theta_x, theta_y, init_state, dt):
    u_max = CONFIG['max_force']
    H = theta_x.shape[0]
    def step_fn(carry, t):
        px, py, vx, vy = carry
        fx = u_max * jnp.tanh(theta_x[t])
        fy = u_max * jnp.tanh(theta_y[t])
        ax = fx - 1.0 * vx
        ay = fy - 1.0 * vy
        npx = px + vx * dt
        npy = py + vy * dt
        nvx = vx + ax * dt
        nvy = vy + ay * dt
        return (npx, npy, nvx, nvy), (npx, npy)
    (_, _, _, _), (px_seq, py_seq) = jax.lax.scan(
        step_fn, (init_state[0], init_state[1], init_state[2], init_state[3]),
        xs=jnp.arange(H))
    return jnp.stack([px_seq, py_seq], axis=1)


def tracking_term(z_flat, ctx):
    H = CONFIG['horizon']
    theta_x, theta_y = z_flat[:H], z_flat[H:2*H]
    pos = rollout(theta_x, theta_y, ctx['init_state'], jnp.array(CONFIG['dt']))
    return jnp.sum(jnp.sum((pos - ctx['target'][None, :])**2, axis=1))

def final_term(z_flat, ctx):
    H = CONFIG['horizon']
    theta_x, theta_y = z_flat[:H], z_flat[H:2*H]
    pos = rollout(theta_x, theta_y, ctx['init_state'], jnp.array(CONFIG['dt']))
    return 10.0 * jnp.sum((pos[-1] - ctx['target'])**2)

def smoothness_term(z_flat, ctx):
    H = CONFIG['horizon']
    u_max = CONFIG['max_force']
    fx = u_max * jnp.tanh(z_flat[:H])
    fy = u_max * jnp.tanh(z_flat[H:2*H])
    return jnp.sum(jnp.diff(fx)**2) + jnp.sum(jnp.diff(fy)**2)


# ===========================================================================
# 2. Build the three cost functions
# ===========================================================================

H = CONFIG['horizon']
init_state = jnp.array([0.0, 0.0, 0.0, 0.0])
target = jnp.array([5.0, 4.0])
ctx_base = {'init_state': init_state, 'target': target}


# --- Approach 1: Dynamic k ---
_dynamic_ctx = dict(ctx_base)
_dynamic_ctx = calibrate_k_into_ctx(_dynamic_ctx, [
    (tracking_term,   'primary',    'tracking'),
    (final_term,      'secondary',  'final'),
    (smoothness_term, 'tiebreaker', 'smoothness'),
], n_dims=2*H, bounds=(-5.0, 5.0), n_samples=500, verbose=False)

_dynamic_obj = compose_objective_dynamic([
    (tracking_term,   0.0, 'tracking'),
    (final_term,      0.0, 'final'),
    (smoothness_term, 0.0, 'smoothness'),
])

_dynamic_terms = [
    (tracking_term, None, 'tracking'),
    (final_term, None, 'final'),
    (smoothness_term, None, 'smoothness'),
]

@jit
def cost_dynamic(z_flat, ctx):
    full_ctx = dict(ctx)
    for key in ['k_tracking', 'k_final', 'k_smoothness']:
        full_ctx[key] = _dynamic_ctx[key]
    return _dynamic_obj(z_flat, full_ctx)


# --- Approach 2: Static k (calibrated once, frozen) ---
_static_obj = compose_objective([
    (tracking_term,   float(_dynamic_ctx['k_tracking']),   'tracking'),
    (final_term,      float(_dynamic_ctx['k_final']),      'final'),
    (smoothness_term, float(_dynamic_ctx['k_smoothness']), 'smoothness'),
])

@jit
def cost_static(z_flat, ctx):
    return _static_obj(z_flat, ctx)


# --- Approach 3: Raw weighted sum (no composer) ---
@jit
def cost_raw(z_flat, ctx):
    f = (1.0 * tracking_term(z_flat, ctx) +
         0.5 * final_term(z_flat, ctx) +
         0.05 * smoothness_term(z_flat, ctx))
    return sigma_k(log_transform(f), k=0.5)


# ===========================================================================
# 3. Optimization Runner with Iteration Tracking
# ===========================================================================

def run_and_track(cost_fn, ctx, label, key, dynamic_update=False, terms=None):
    """Run IGO, recording best cost at each iteration.

    We can't easily hook into lax.scan internals, so we track the FINAL
    cost and solution quality. For per-iteration tracking, we run the
    solver in shorter segments.
    """
    M = 2  # two blocks: theta_x, theta_y
    K = CONFIG['n_components']
    B = CONFIG['pop_size']
    B0 = CONFIG['elite_size']
    dims = (H, H)
    d_max = H

    key, sk_init = random.split(key)
    initial_mu = jnp.zeros((M, K, d_max))
    for c in range(K):
        initial_mu = initial_mu.at[0, c, :].set(
            random.uniform(random.fold_in(sk_init, c), (d_max,), minval=-2.0, maxval=2.0))
        initial_mu = initial_mu.at[1, c, :].set(
            random.uniform(random.fold_in(sk_init, c+100), (d_max,), minval=-2.0, maxval=2.0))
    initial_L_inv = jnp.tile(jnp.eye(d_max)[None, None, :, :], (M, K, 1, 1)) * 1.0
    initial_v = jnp.zeros((M, K-1))

    mu_k, L_inv_k, v_k = initial_mu, initial_L_inv, initial_v

    # Run in warmup + optimization
    t0 = time.perf_counter()
    key, sk = random.split(key)

    # Warmup
    final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
        sk, CONFIG['warmup_steps'], 0.15, M, K, B, B0, dims, CONFIG['t0_reset'],
        fitness_fn_total=cost_fn,
        initial_mu_k=mu_k, initial_L_inv_k=L_inv_k, initial_v_k=v_k,
        context=ctx)
    final_mu.block_until_ready()
    final_v = jnp.log(jnp.clip(final_pi[:, :-1], 1e-20, None)
                      / jnp.clip(final_pi[:, -1:], 1e-20, None))

    # Optimization
    key, sk = random.split(key)
    final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
        sk, CONFIG['opt_steps'], 0.15, M, K, B, B0, dims, CONFIG['t0_reset'],
        fitness_fn_total=cost_fn,
        initial_mu_k=final_mu, initial_L_inv_k=final_L, initial_v_k=final_v,
        context=ctx)
    final_mu.block_until_ready()

    elapsed = time.perf_counter() - t0

    # Extract best solution
    bi = [int(jnp.argmax(final_pi[0])), int(jnp.argmax(final_pi[1]))]
    best_z = jnp.concatenate([final_mu[0, bi[0], :H], final_mu[1, bi[1], :H]])

    # Evaluate quality
    pos = rollout(best_z[:H], best_z[H:2*H], init_state, jnp.array(CONFIG['dt']))
    final_pos = np.array(pos[-1])
    dist = float(np.sqrt(np.sum((final_pos - np.array(target))**2)))
    final_cost = float(cost_fn(best_z, ctx))

    # Dynamic update: recalibrate k from elite
    if dynamic_update and terms is not None:
        _dynamic_ctx.update(
            update_k_from_elite(dict(_dynamic_ctx), terms, best_z,
                                multiplier=5.0, verbose=False))

    # Sensitivity check: evaluate on random neighbors of best solution
    key, sk = random.split(key)
    noise = random.normal(sk, (100, 2*H)) * 1.0  # perturbations
    neighbors = best_z[None, :] + noise
    costs_nearby = np.array([float(cost_fn(n, ctx)) for n in neighbors])
    cost_spread = float(np.std(costs_nearby))

    return {
        'label': label,
        'dist': dist,
        'final_cost': final_cost,
        'cost_spread': cost_spread,
        'elapsed': elapsed,
        'final_pos': final_pos,
        'best_z': np.array(best_z),
    }


# ===========================================================================
# 4. Run Comparison
# ===========================================================================

def run_comparison():
    print("=" * 70)
    print("Dynamic k vs Static k vs Raw Weights — Optimization Comparison")
    print("=" * 70)
    print(f"  Horizon={H}, IGO: warmup={CONFIG['warmup_steps']}, opt={CONFIG['opt_steps']}")
    print(f"  Start=(0,0) → Target=(8,6)")
    print()

    key = random.PRNGKey(42)

    # Run all three
    results = {}

    print("--- Running DYNAMIC k ---")
    key, sk = random.split(key)
    results['dynamic'] = run_and_track(
        cost_dynamic, ctx_base, 'dynamic', sk,
        dynamic_update=True, terms=_dynamic_terms)
    print(f"  dist={results['dynamic']['dist']:.4f}  "
          f"cost={results['dynamic']['final_cost']:.6f}  "
          f"spread={results['dynamic']['cost_spread']:.6f}  "
          f"time={results['dynamic']['elapsed']:.2f}s")

    print("--- Running STATIC k ---")
    key, sk = random.split(key)
    results['static'] = run_and_track(
        cost_static, ctx_base, 'static', sk)
    print(f"  dist={results['static']['dist']:.4f}  "
          f"cost={results['static']['final_cost']:.6f}  "
          f"spread={results['static']['cost_spread']:.6f}  "
          f"time={results['static']['elapsed']:.2f}s")

    print("--- Running RAW weights ---")
    key, sk = random.split(key)
    results['raw'] = run_and_track(
        cost_raw, ctx_base, 'raw', sk)
    print(f"  dist={results['raw']['dist']:.4f}  "
          f"cost={results['raw']['final_cost']:.6f}  "
          f"spread={results['raw']['cost_spread']:.6f}  "
          f"time={results['raw']['elapsed']:.2f}s")

    # --- Summary Table ---
    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Metric':<28} {'Dynamic':>12} {'Static':>12} {'Raw':>12}")
    print("-" * 65)
    for key_metric, fmt, name in [
        ('dist', '{:.4f}', 'Final dist to target'),
        ('final_cost', '{:.6f}', 'Final cost value'),
        ('cost_spread', '{:.6f}', 'Cost spread (sensitivity)'),
        ('elapsed', '{:.2f}s', 'Wall time'),
    ]:
        vals = [fmt.format(results[m][key_metric]) for m in ['dynamic', 'static', 'raw']]
        print(f"  {name:<26} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Sensitivity analysis: cost_spread is std of costs for 100 nearby solutions
    print()
    print("--- Sensitivity Analysis ---")
    print("  'Cost spread' = std(cost) over 100 random neighbors of the best solution.")
    print("  Higher = better distinguishability near the optimum.")
    best_spread = max(results['dynamic']['cost_spread'],
                      results['static']['cost_spread'],
                      results['raw']['cost_spread'])
    for mode in ['dynamic', 'static', 'raw']:
        s = results[mode]['cost_spread']
        bar = '█' * int(40 * s / best_spread)
        print(f"  {mode:10s}: {s:.6f}  {bar}")

    print()
    print("--- Verdict ---")
    # Which is best?
    ranking = sorted(['dynamic', 'static', 'raw'],
                     key=lambda m: (results[m]['dist'], -results[m]['cost_spread']))
    print(f"  Best final position: {ranking[0]} ({results[ranking[0]]['dist']:.4f})")
    print(f"  Best sensitivity:    {max(results, key=lambda m: results[m]['cost_spread'])} ")
    print()
    if results['static']['dist'] > 1.5 * results['dynamic']['dist']:
        print("  ⚠ Static k is significantly worse than dynamic k.")
        print("    Consider removing compose_objective (static) in favor of dynamic.")
    elif results['static']['dist'] < 1.1 * results['dynamic']['dist']:
        print("  ✓ Static k performs similarly to dynamic on this problem.")
        print("    Both are valid; dynamic is safer for problems with large magnitude drift.")


if __name__ == "__main__":
    run_comparison()
