"""Closed-loop performance evaluation for Frenet B-spline MPC.

Records four metrics across an MPC run:
  1. Tracking:   Lyapunov cost per step (is the plan tracking the reference?)
  2. Overshoot:  does d(t) exceed the target lane? (lane change)
  3. Oscillation: std of d and v over a trailing window (steady-state stability)
  4. Constraints: g-values (obs, lane, speed, acc, jerk) per step
"""

from __future__ import annotations

import sys, time, argparse
from pathlib import Path
from dataclasses import dataclass, field

import jax, jax.numpy as jnp
import numpy as np
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.core.reference_path import StraightReference
from Cartest.planning.warmstart import build_initial_mu
from Cartest.planning.cost import make_objective, build_context
from Cartest.execution.execute import execute_perfect_tracking, FrenetState
from Cartest.planning.constraints import make_constraints, compute_g_values, compute_summary
from Cartest.planning.scenario import EMPTY as scenario
from gmm_igo.solver_builder import build_solver

BASIS = Path(__file__).resolve().parents[1] / "basis"


@dataclass
class EvalMetrics:
    """Per-step closed-loop metrics."""
    step:          int
    cost:          float
    d_trajectory:  list[float] = field(default_factory=list)
    v_trajectory:  list[float] = field(default_factory=list)
    overshoot_d:   float = 0.0            # max(0, |d| - d_target) if applicable
    oscillation_d: float = 0.0            # trailing std of d
    oscillation_v: float = 0.0            # trailing std of v
    g_obs:         float = 0.0
    g_lane:        float = 0.0
    g_speed:       float = 0.0
    g_acc:         float = 0.0
    g_jerk:        float = 0.0

    def to_dict(self):
        return {
            'step': self.step, 'cost': self.cost,
            'overshoot_d': self.overshoot_d,
            'oscillation_d': self.oscillation_d,
            'oscillation_v': self.oscillation_v,
            'g_obs': self.g_obs, 'g_lane': self.g_lane,
            'g_speed': self.g_speed, 'g_acc': self.g_acc, 'g_jerk': self.g_jerk,
        }


def run_eval(steps=150, seed=0, window=20):
    """Run MPC with closed-loop evaluation.

    Args:
        steps:  number of MPC iterations
        seed:   RNG seed
        window: trailing window size for oscillation metric

    Returns:
        list of EvalMetrics, one per step.
    """
    ref_path = StraightReference()
    gen = FrenetBSplineTrajectory(BASIS / "bspline_basis.npz", ref_path)

    obs_list  = scenario["obstacles"]
    lane_hw   = scenario["lane_hw"]
    safe_dist = scenario["obs_safe_dist"]
    v_target  = scenario["v_target"]
    obs_pos = jnp.array([[o["x"], o["y"]] for o in obs_list], dtype=jnp.float32)
    obs_rad = jnp.array([o["r"] for o in obs_list], dtype=jnp.float32)

    solver = build_solver(
        make_objective(gen, omega_s=1.0, omega_d=4.0, alpha=0.0),
        dims=(gen.n_free, gen.n_free),
        constraints=make_constraints(gen, lane_hw, safe_dist),
        solver='m22', T=300, dt=0.3, K=3, B=64, B0=30, T_0=300,
        k_inner=1.0, obj_transform='standard',
    )

    key = random.PRNGKey(seed)
    init = scenario["init"]
    state = FrenetState(s=init["s"], s_dot=init["s_dot"], s_ddot=init["s_ddot"],
                        d=init["d"], d_dot=init["d_dot"], d_ddot=init["d_ddot"],
                        psi=init.get("psi", 0.0))

    hx, hy, hv = [state.s], [state.d], [state.s_dot]
    all_metrics = []

    for step in range(steps):
        key, sk = random.split(key)

        ctx = build_context(gen, state, v_target, lane_hw, obs_pos, obs_rad)
        mu_init = build_initial_mu(gen, state.s, state.s_dot, state.d)

        t0 = time.time()
        result = solver(sk, context=ctx, initial_mu=mu_init)
        ms = (time.time() - t0) * 1000

        ctrl_s, ctrl_d = result.x[:gen.n_free], result.x[gen.n_free:]
        frenet, st, (x_cart, y_cart) = gen.evaluate_plan(ctrl_s, ctrl_d, ctx)
        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = frenet

        gv = compute_g_values(st, d, x_cart, y_cart, obs_pos, obs_rad,
                              lane_hw, safe_dist)

        # Execute
        state = execute_perfect_tracking(s, d, s_dot, d_dot, s_ddot, d_ddot,
                                         st[1, 3])
        hx.append(state.s); hy.append(state.d); hv.append(state.s_dot)

        # ── Compute metrics ──────────────────────────────────────
        m = EvalMetrics(step=step, cost=float(result.cost))

        # Overshoot: |d| beyond lane_hw (or target lane center)
        d_abs = float(jnp.max(jnp.abs(d)))
        m.overshoot_d = max(0.0, d_abs - lane_hw)

        # Oscillation: trailing std of d and v
        hy_arr = np.array(hy[-window:])
        hv_arr = np.array(hv[-window:])
        if len(hy_arr) >= 5:
            m.oscillation_d = float(np.std(hy_arr))
            m.oscillation_v = float(np.std(hv_arr))

        # Constraint violations
        m.g_obs   = float(gv['obs'])
        m.g_lane  = float(gv['lane'])
        m.g_speed = float(gv['spd'])
        m.g_acc   = float(gv['acc'])
        m.g_jerk  = float(gv['jerk'])

        all_metrics.append(m)

        # Progress
        if step % 10 == 0:
            viol = sum(1 for v in [m.g_obs, m.g_lane, m.g_speed, m.g_acc, m.g_jerk] if v > 0)
            print(f"[{step:3d}] cost={m.cost:6.2f}  "
                  f"overshoot={m.overshoot_d:.3f}  osc_d={m.oscillation_d:.3f}  "
                  f"violations={viol}  solve={ms:.0f}ms")

    return all_metrics, np.array(hx), np.array(hy), np.array(hv)


def print_summary(metrics):
    """Print end-of-run summary."""
    costs = [m.cost for m in metrics]
    overshoots = [m.overshoot_d for m in metrics]
    osc_d = [m.oscillation_d for m in metrics]
    osc_v = [m.oscillation_v for m in metrics]
    g_obs = [m.g_obs for m in metrics]
    g_lane = [m.g_lane for m in metrics]
    g_speed = [m.g_speed for m in metrics]
    g_acc = [m.g_acc for m in metrics]
    g_jerk = [m.g_jerk for m in metrics]

    # Steady-state: last 30 steps
    ss = slice(-30, None)

    print("\n" + "=" * 60)
    print("CLOSED-LOOP EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Tracking:")
    print(f"    cost mean={np.mean(costs):.2f}  final={costs[-1]:.2f}")
    print(f"    cost trend: {'↓ decreasing' if costs[-1] < np.mean(costs[:10]) else '⚠ flat/rising'}")
    print(f"  Overshoot:")
    print(f"    max overshoot d: {np.max(overshoots):.4f}")
    print(f"  Oscillation (steady-state std):")
    print(f"    d: {np.mean(osc_d[ss]):.4f}  v: {np.mean(osc_v[ss]):.4f}")
    print(f"  Constraints (max violation):")
    print(f"    obs={np.max(g_obs):.4f}  lane={np.max(g_lane):.4f}  "
          f"speed={np.max(g_speed):.4f}  acc={np.max(g_acc):.4f}  "
          f"jerk={np.max(g_jerk):.4f}")

    # Summary verdict
    ok = True
    if np.max(g_obs) > 0:
        print("  ⚠ obstacle violated"); ok = False
    if np.max(g_lane) > 0:
        print("  ⚠ lane violated"); ok = False
    if np.max(overshoots) > 0.1:
        print("  ⚠ overshoot detected"); ok = False
    if np.mean(osc_d[ss]) > 0.5:
        print("  ⚠ d oscillation in steady state"); ok = False
    if ok:
        print("  ✓ All checks passed")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=20,
                   help="trailing window for oscillation metric")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    metrics, hx, hy, hv = run_eval(steps=args.steps, seed=args.seed,
                                   window=args.window)
    print_summary(metrics)
