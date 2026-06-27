"""
Dynamic Gamma vs Static Gamma — Nonlinear Constraint
======================================================

Narrow gap between two circular obstacles.
The solver must squeeze through a 0.2m gap.
Objective pulls toward target beyond the gap.

Three constraints (all hard):
  L1: outside circle A  (x+1)^2 + y^2 >= 3^2
  L2: outside circle B  (x-1)^2 + y^2 >= 3^2
  L3: y <= 2            (ceiling)

Feasible region: the narrow corridor between the two circles
at y>0, with a ceiling at y=2.
"""

import sys, os
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

import jax, jax.numpy as jnp, numpy as np
from jax import random, jit
import time

from Constraintdealer.Constran import (
    build, build_dynamic, Deterministic,
    calibrate_gamma_into_ctx, update_gamma_from_elite,
    sigma_k, log_transform,
)
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc


# ===========================================================================
# 1. Problem Setup — MUST pass through a narrow gate
# ===========================================================================
# Two large walls force the solver toward x=0.
# At x=0, three small obstacles create a tiny gate (0.15m wide) at y≈1.3.
# Bounds: x∈[-3,3], y∈[0,4]
#
# Obstacles (all hard):
#   Wall L:   x <= -0.2  (blocks left path)  → g = -0.2 - x
#   Wall R:   x >=  0.2  (blocks right path) → g = x - 0.2
#   Obst A:   circle at (0, 1.0), r=0.25  (lower blocker)
#   Obst B:   circle at (0, 1.6), r=0.25  (upper blocker)
#   Gate:     between y=1.25 and y=1.35, x∈[-0.2, 0.2]
#
# Objective: min distance to target (0, 3) — beyond the gate.

CONFIG = {
    'n_components': 4, 'pop_size': 80, 'elite_size': 35,
    'warmup_steps': 200, 'opt_steps': 500,
    't0_reset': 100,
}


def objective_fn(x, ctx):
    target = jnp.array([0.0, 3.0])
    return jnp.sum((x - target) ** 2)


def viol_wall_left(x, ctx):
    return -x[0] - 0.2    # >0 when x < -0.2 (left of corridor)

def viol_wall_right(x, ctx):
    return x[0] - 0.2     # >0 when x > 0.2 (right of corridor)

def viol_obst_low(x, ctx):
    return 0.25**2 - (x[0]**2 + (x[1]-1.0)**2)  # >0 inside lower obstacle

def viol_obst_high(x, ctx):
    return 0.25**2 - (x[0]**2 + (x[1]-1.6)**2)  # >0 inside upper obstacle


# ===========================================================================
# 2. Run and track
# ===========================================================================

def run_optimization(cost_fn, ctx, key, label):
    M, K = 1, CONFIG['n_components']
    B, B0 = CONFIG['pop_size'], CONFIG['elite_size']
    dims = (2,)
    d_max = 2

    key, sk = random.split(key)
    initial_mu = random.uniform(sk, (M, K, d_max), minval=-1.0, maxval=2.0)
    initial_L_inv = jnp.tile(jnp.eye(d_max)[None,None,:,:], (M,K,1,1)) * 1.0
    initial_v = jnp.zeros((M, K-1))

    t0 = time.perf_counter()
    key, sk = random.split(key)
    final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
        sk, CONFIG['warmup_steps'], 0.15, M, K, B, B0, dims, CONFIG['t0_reset'],
        fitness_fn_total=cost_fn,
        initial_mu_k=initial_mu, initial_L_inv_k=initial_L_inv,
        initial_v_k=initial_v, context=ctx)
    final_mu.block_until_ready()

    key, sk = random.split(key)
    final_v = jnp.log(jnp.clip(final_pi[:, :-1], 1e-20, None)
                      / jnp.clip(final_pi[:, -1:], 1e-20, None))
    final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
        sk, CONFIG['opt_steps'], 0.15, M, K, B, B0, dims, CONFIG['t0_reset'],
        fitness_fn_total=cost_fn,
        initial_mu_k=final_mu, initial_L_inv_k=final_L, initial_v_k=final_v,
        context=ctx)
    final_mu.block_until_ready()
    elapsed = time.perf_counter() - t0

    best_idx = int(jnp.argmax(final_pi[0]))
    best_x = np.array(final_mu[0, best_idx, :])

    # Compute total violation
    g_wl = max(0.0, viol_wall_left(jnp.array(best_x), ctx))
    g_wr = max(0.0, viol_wall_right(jnp.array(best_x), ctx))
    g_lo = max(0.0, viol_obst_low(jnp.array(best_x), ctx))
    g_hi = max(0.0, viol_obst_high(jnp.array(best_x), ctx))
    total_viol = g_wl + g_wr + g_lo + g_hi

    obj_val = float(objective_fn(jnp.array(best_x), ctx))
    cost_val = float(cost_fn(jnp.array(best_x), ctx))

    # Cost spread near optimum (violation sensitivity)
    key, sk = random.split(key)
    noise = random.normal(sk, (200, 2)) * 0.1
    neighbors = best_x[None, :] + noise
    costs_nearby = np.array([float(cost_fn(n, ctx)) for n in neighbors])
    spread = float(np.std(costs_nearby))

    # Count how many neighbors are feasible
    feasible = 0
    for n in neighbors:
        gw = float(viol_wall_left(n, ctx)); gx = float(viol_wall_right(n, ctx))
        gl = float(viol_obst_low(n, ctx)); gh = float(viol_obst_high(n, ctx))
        if gw <= 0 and gx <= 0 and gl <= 0 and gh <= 0:
            feasible += 1

    return {
        'label': label, 'best_x': best_x, 'total_viol': total_viol,
        'objective': obj_val, 'cost': cost_val,
        'cost_spread': spread, 'feasible_pct': feasible / len(neighbors),
        'elapsed': elapsed,
    }


# ===========================================================================
# 3. Run
# ===========================================================================

def main():
    key = random.PRNGKey(42)
    constraints = [
        Deterministic(viol_wall_left,  mode='hard', priority=1),
        Deterministic(viol_wall_right, mode='hard', priority=2),
        Deterministic(viol_obst_low,   mode='hard', priority=3),
        Deterministic(viol_obst_high,  mode='hard', priority=4),
    ]
    constraint_fns = [
        (viol_wall_left,  1), (viol_wall_right, 2),
        (viol_obst_low,   3), (viol_obst_high,  4),
    ]

    ctx = {}

    # --- Static ---
    print("=" * 50)
    print("Static Gamma (fixed build)")
    print("=" * 50)
    cost_static = build(objective_fn, constraints, jit_cost=False)
    key, sk = random.split(key)
    r_s = run_optimization(cost_static, ctx, sk, 'static')
    print(f"  x=({r_s['best_x'][0]:.4f},{r_s['best_x'][1]:.4f})  "
          f"viol={r_s['total_viol']:.6f}  spread={r_s['cost_spread']:.6f}  "
          f"feasible_neighbors={r_s['feasible_pct']*100:.0f}%")

    # --- Dynamic (pass 1) ---
    print()
    print("=" * 50)
    print("Dynamic Gamma (pass 1 — same as static)")
    print("=" * 50)
    ctx = calibrate_gamma_into_ctx({}, constraints)
    cost_dyn = build_dynamic(objective_fn, constraints, jit_cost=False)
    key, sk = random.split(key)
    r_d1 = run_optimization(cost_dyn, ctx, sk, 'dynamic p1')
    print(f"  x=({r_d1['best_x'][0]:.4f},{r_d1['best_x'][1]:.4f})  "
          f"viol={r_d1['total_viol']:.6f}  spread={r_d1['cost_spread']:.6f}  "
          f"feasible_neighbors={r_d1['feasible_pct']*100:.0f}%")

    # --- Dynamic (pass 2, gamma updated) ---
    print()
    print("=" * 50)
    print("Dynamic Gamma (pass 2 — gamma updated from elite)")
    print("=" * 50)
    ctx = update_gamma_from_elite(ctx, constraint_fns,
                                   jnp.array(r_d1['best_x']), verbose=True)
    key, sk = random.split(key)
    r_d2 = run_optimization(cost_dyn, ctx, sk, 'dynamic p2')
    print(f"  x=({r_d2['best_x'][0]:.4f},{r_d2['best_x'][1]:.4f})  "
          f"viol={r_d2['total_viol']:.6f}  spread={r_d2['cost_spread']:.6f}  "
          f"feasible_neighbors={r_d2['feasible_pct']*100:.0f}%")

    # --- Summary ---
    print()
    print("=" * 60)
    print(f"{'Metric':<30} {'Static':>12} {'Dynamic':>12}")
    print("-" * 55)
    for key_name, fmt, label in [
        ('total_viol', '{:.6f}', 'Total violation'),
        ('cost_spread', '{:.6f}', 'Cost spread (sensitivity)'),
        ('feasible_pct', '{:.0f}%', 'Feasible neighbors'),
    ]:
        vals = [fmt.format(r_s[key_name]), fmt.format(r_d2[key_name])]
        print(f"  {label:<28} {vals[0]:>12} {vals[1]:>12}")

    # Cost landscape scan: across the gap
    print()
    print("--- Cost landscape across gate (x=0, y in [0.8, 2.0]) ---")
    print(f"  {'y':>6}  {'static σ':>10}  {'dynamic σ':>10}  {'feasible?':>10}")
    for y_val in np.linspace(0.8, 2.0, 25):
        x = jnp.array([0.0, y_val])
        cs = float(cost_static(x, ctx))
        cd = float(cost_dyn(x, ctx))
        gw = float(viol_wall_left(x, ctx)); gx = float(viol_wall_right(x, ctx))
        gl = float(viol_obst_low(x, ctx)); gh = float(viol_obst_high(x, ctx))
        feasible = "✓" if (gw <= 0 and gx <= 0 and gl <= 0 and gh <= 0) else "✗"
        marker = "← GATE" if feasible else ""
        print(f"  {y_val:6.3f}  {cs:10.6f}  {cd:10.6f}  {feasible:>10} {marker}")


if __name__ == "__main__":
    main()
