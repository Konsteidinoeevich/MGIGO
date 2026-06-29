"""
Validate per-layer k calibration: equal-k vs auto-calibrated.

Builds actual Constran cost functions with 8 synthetic layers,
sweeps violations, and measures float32 distinguishability.

Key question: does auto_k=True keep ALL layers alive in f32?
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from Constraintdealer.Constran import (
    build, Deterministic, sigma_k, T_alpha,
    auto_calibrate_k, _rescale_transform_table, _make_transform_fn,
    TRANSFORM_SOFT, TRANSFORM_TUNABLE, CONSTRAINT_K,
)

EPS_F32 = 6e-8
TARGET_RESOLUTION = 1000  # minimum distinguishable values per layer
OUT = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════
# Build cost functions: equal-k vs calibrated
# ═══════════════════════════════════════════════════════════════════════

def _viol(key):
    return lambda x, ctx: ctx.get(key, jnp.array(0.0, dtype=jnp.float32))


def build_n_layer(n, auto_k=True):
    """Build an n-layer cost function. All layers soft + tunable mix."""
    obj_fn = lambda x, ctx: jnp.array(0.0, dtype=jnp.float32)
    constraints = []
    for i in range(n):
        mode = 'tunable' if i < n // 2 else 'soft'
        transform = 'tunable' if mode == 'tunable' else 'soft'
        constraints.append(
            Deterministic(_viol(f'L{i}'), mode=mode, priority=i+1,
                         transform=transform,
                         tune_preset='standard' if mode == 'tunable' else 'none')
        )
    return build(obj_fn, constraints, jit_cost=False, auto_k=auto_k)


# ═══════════════════════════════════════════════════════════════════════
# Measurement
# ═══════════════════════════════════════════════════════════════════════

def measure_layer(cost_fn, layer_key, n_layers):
    """Sweep one layer, return distinguishable value count and min Δcost."""
    g_sweep = np.logspace(-6, 4, 200)
    costs = []
    for g in g_sweep:
        ctx = {f'L{i}': 0.0 for i in range(n_layers)}
        ctx[layer_key] = float(g)
        costs.append(float(cost_fn(jnp.zeros(1), ctx)))
    costs = np.array(costs)

    # Deduplicate at float32 precision (round to 7 decimals)
    costs_rounded = np.round(costs, decimals=7)
    unique = np.unique(costs_rounded)
    n_distinct = len(unique)

    # Min non-zero difference
    diffs = np.diff(np.sort(unique))
    diffs = diffs[diffs > EPS_F32 * 0.1]
    min_dc = np.min(diffs) if len(diffs) > 0 else 0.0

    # Detection threshold: first g where cost > EPS_F32
    above = np.where(costs_rounded > EPS_F32)[0]
    det_thresh = float(g_sweep[above[0]]) if len(above) > 0 else float('inf')

    # Dynamic range
    cost_range = np.max(unique) - np.min(unique)

    return {
        'n_distinct': n_distinct,
        'min_dc': min_dc,
        'cost_range': cost_range,
        'det_thresh': det_thresh,
        'g_sweep': g_sweep,
        'costs': costs,
        'alive': n_distinct >= 50 and min_dc > EPS_F32,
    }


# ═══════════════════════════════════════════════════════════════════════
def run_validation():
    depths = [4, 6, 8, 10, 12]

    # Summary table
    print("=" * 90)
    print("PER-LAYER k CALIBRATION VALIDATION")
    print(f"  Target: ≥ {TARGET_RESOLUTION} distinguishable f32 values per layer")
    print(f"  f32 noise floor: {EPS_F32:.1e}")
    print("=" * 90)

    for n in depths:
        print(f"\n{'='*90}")
        print(f"  n = {n} layers")
        print(f"{'='*90}")

        # Build both versions
        cost_eq = build_n_layer(n, auto_k=False)
        cost_cal = build_n_layer(n, auto_k=True)

        # Get calibrated k values
        ks = auto_calibrate_k(n, k_outer=0.2)
        ks_str = ', '.join([f'{k:.3f}' for k in ks])

        # Priority order: L0=innermost (highest priority number), L{n-1}=outermost
        print(f"  Calibrated k (innermost→outermost): [{ks_str}]")
        print(f"  Gain (innermost): ∏k = {np.prod(ks):.2e}")
        print(f"  Gain (outermost): k = {ks[-1]:.3f}")
        print()

        header = (f"  {'Layer':>8s} | {'Pos':>6s} | "
                  f"{'Equal-k distinct':>16s} | {'Calib distinct':>16s} | "
                  f"{'Eq minΔc':>10s} | {'Cal minΔc':>10s} | "
                  f"{'Eq alive?':>10s} | {'Cal alive?':>10s}")
        sep = f"  {'-'*8}-+-{'-'*6}-+-{'-'*16}-+-{'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}"
        print(header)
        print(sep)

        all_cal_alive = True
        any_eq_dead = False

        for i in range(n):
            layer_key = f'L{i}'
            # priority=i+1 → sorted reverse=True:
            #   low i = low priority# = OUTERMOST (processed last in loop)
            #   high i = high priority# = INNERMOST (processed first)
            if i == 0:
                pos = 'OUTER'
            elif i < n//3:
                pos = 'outer'
            elif i >= n - n//3:
                pos = 'inner'
            else:
                pos = 'mid'

            r_eq = measure_layer(cost_eq, layer_key, n)
            r_cal = measure_layer(cost_cal, layer_key, n)

            eq_alive = '✓' if r_eq['alive'] else '☠ DEAD'
            cal_alive = '✓' if r_cal['alive'] else '☠ DEAD'

            if not r_cal['alive']:
                all_cal_alive = False
            if not r_eq['alive']:
                any_eq_dead = True

            print(f"  {layer_key:>8s} | {pos:>6s} | "
                  f"{r_eq['n_distinct']:>16d} | {r_cal['n_distinct']:>16d} | "
                  f"{r_eq['min_dc']:>10.2e} | {r_cal['min_dc']:>10.2e} | "
                  f"{eq_alive:>10s} | {cal_alive:>10s}")

        print()
        print(f"  All calibrated alive: {all_cal_alive}")
        print(f"  Any equal-k dead: {any_eq_dead}")

    # --- Figure: side-by-side for n=8 ---
    n_demo = 8
    cost_eq = build_n_layer(n_demo, auto_k=False)
    cost_cal = build_n_layer(n_demo, auto_k=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    g_fine = np.logspace(-6, 4, 200)

    for i in range(n_demo):
        ax = axes[i // 4, i % 4]
        layer_key = f'L{i}'

        r_eq = measure_layer(cost_eq, layer_key, n_demo)
        r_cal = measure_layer(cost_cal, layer_key, n_demo)

        ax.semilogx(r_eq['g_sweep'], r_eq['costs'], 'b-', lw=2,
                    label=f'equal k=0.2 ({r_eq["n_distinct"]} vals)')
        ax.semilogx(r_cal['g_sweep'], r_cal['costs'], 'r-', lw=2,
                    label=f'calibrated ({r_cal["n_distinct"]} vals)')
        ax.axhline(EPS_F32, color='gray', ls='--', lw=0.8, alpha=0.7)
        ax.set_xlabel('g'); ax.set_ylabel('cost')
        # L0=OUTERMOST (priority=1, processed last), L{n-1}=INNERMOST
        if i == 0:
            pos = 'OUTERMOST'
        elif i < n_demo // 3:
            pos = 'outer'
        elif i >= n_demo - n_demo // 3:
            pos = f'inner (×{0.2**(n_demo-i):.0e} gain)'
        else:
            pos = f'mid (×{0.2**(n_demo-i):.0e} gain)'
        ax.set_title(f'{layer_key} P{i+1} ({pos})', weight='bold', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    fig.suptitle(f'Per-Layer k Calibration: Equal vs Calibrated (n={n_demo})\n'
                 f'Blue=equal k=0.2, Red=auto-calibrated taper. '
                 f'Dashed=f32 noise floor.',
                 fontsize=12, weight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'calibration_validate.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved to {OUT / 'calibration_validate.png'}")


if __name__ == "__main__":
    run_validation()
