"""Frenet B-spline trajectory MPC demo."""

from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.core.reference_path import StraightReference
from Cartest.planning.warmstart import build_initial_mu
from Cartest.planning.cost import make_objective, build_context
from Cartest.execution.execute import execute_perfect_tracking
from Cartest.planning.constraints import make_constraints, compute_g_values, compute_summary
from Cartest.eval.reporting import StepReport
from Cartest.eval.plotting import setup_axes, render_frame, save_animation
from Cartest.eval.diagnostics import diagnose, print_diag
from Cartest.planning.scenario import EMPTY as scenario
from gmm_igo.solver_builder import build_solver


ROOT = Path(__file__).resolve().parent
BASIS = ROOT / "basis"


# ═══════════════════════════════════════════════════════════════════════
# MPC
# ═══════════════════════════════════════════════════════════════════════

def run(steps=150, seed=0, plot=True):
    ref_path = StraightReference()
    gen = FrenetBSplineTrajectory(BASIS / "bspline_basis.npz", ref_path)

    # Build solver ONCE
    # ── Scenario ──
    obs_list  = scenario["obstacles"]
    lane_hw   = scenario["lane_hw"]
    safe_dist = scenario["obs_safe_dist"]
    v_target  = scenario["v_target"]
    obs_pos = jnp.array([[o["x"], o["y"]] for o in obs_list], dtype=jnp.float32)
    obs_rad = jnp.array([o["r"] for o in obs_list], dtype=jnp.float32)

    solver = build_solver(
        make_objective(gen, omega_s=1.0, omega_d=4.0, alpha=0.0), dims=(gen.n_free, gen.n_free),
        constraints=make_constraints(gen, lane_hw, safe_dist),
        solver='m22', T=300, dt=0.3, K=3, B=64, B0=30, T_0=300,
        k_inner=1.0, obj_transform='standard',
    )

    key = random.PRNGKey(seed)

    from Cartest.execution.execute import FrenetState
    init = scenario["init"]
    state = FrenetState(s=init["s"], s_dot=init["s_dot"], s_ddot=init["s_ddot"],
                        d=init["d"], d_dot=init["d_dot"], d_ddot=init["d_ddot"],
                        psi=init.get("psi", 0.0))

    # ── JIT warm‑up: compile all JAX functions before the first timed step ──
    ctx_warm = build_context(gen, state, v_target, lane_hw, obs_pos, obs_rad)
    mu_warm = build_initial_mu(gen, state.s, state.s_dot, state.d)
    _ = solver(random.PRNGKey(999), context=ctx_warm, initial_mu=mu_warm)

    if plot:
        fig, ax_t, ax_k = setup_axes()

    hx, hy, hv = [state.s], [state.d], [state.s_dot]
    frames, reports = [], []

    for step in range(steps):
        key, sk = random.split(key)

        ctx = build_context(gen, state, v_target, lane_hw, obs_pos, obs_rad)
        mu_init = build_initial_mu(gen, state.s, state.s_dot, state.d)

        t0 = time.time()
        result = solver(sk, context=ctx, initial_mu=mu_init)
        ms = (time.time() - t0) * 1000

        ctrl_s, ctrl_d = result.x[:gen.n_free], result.x[gen.n_free:]

        # Evaluate plan → Frenet + vehicle states + Cartesian (one call)
        frenet, st, (x_cart, y_cart) = gen.evaluate_plan(ctrl_s, ctrl_d, ctx)
        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = frenet

        # Metrics
        gv = compute_g_values(st, d, x_cart, y_cart, obs_pos, obs_rad, lane_hw, safe_dist)
        sm = compute_summary(st, d, x_cart, y_cart, obs_pos, obs_rad)

        # Execute: use plan's predicted next state (perfect tracking)
        state = execute_perfect_tracking(s, d, s_dot, d_dot, s_ddot, d_ddot, st[1, 3])
        hx.append(state.s); hy.append(state.d); hv.append(state.s_dot)

        # Record
        report = StepReport(
            step=step,
            hx=np.array(hx), hy=np.array(hy), hv=np.array(hv),
            px=np.array(x_cart), py=np.array(y_cart), sp=np.array(st[:, 2]),
            a_long=np.array(st[:, 4]), a_lat=np.array(st[:, 5]),
            jm=np.array(jnp.sqrt(st[:, 6]**2 + st[:, 7]**2)),
            solve_ms=ms, min_obs=sm['min_obs'],
            max_along=sm['max_a_long'], max_alat=sm['max_a_lat'],
            max_jerk=sm['max_jerk'], cost=result.cost, g_values=gv,
        )
        reports.append(report)
        frames.append(report.to_frame_dict())

        print(report.print_cost())
        print(report.print_line())
        if step % 5 == 0:
            print_diag(diagnose(gen, result.x, ctx, safe_dist, state))

    if plot:
        np.savez(ROOT / "frenet_demo.npz",
                 hx=np.array(hx), hy=np.array(hy), hv=np.array(hv),
                 frames=np.array(frames, dtype=object))
        save_animation(fig, reports,
                       lambda i: render_frame(ax_t, ax_k, reports[i],
                                              obs_list, safe_dist, gen.dt),
                       ROOT / "frenet_demo.gif")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(steps=args.steps, seed=args.seed, plot=not args.no_plot)
