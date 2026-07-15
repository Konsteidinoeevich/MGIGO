"""Test P1 Frenet coordinate transform correctness on variable-curvature paths.

Validates that the Cartesian → Frenet pipeline (``_build_vehicle_reference``
→ ``from_vehicle_states``) produces kinematically consistent z_ref without
numerical divergence, even on tight S-bends (R_min=20 m).

Also tests asymmetric lane constraints (d_min/d_max) for "no right lane change".

Usage::

    uv run python Cartest/test_p1_frenet.py                    # all tests
    uv run python Cartest/test_p1_frenet.py --test s_bend      # S-bend only
    uv run python Cartest/test_p1_frenet.py --steps 30 --no-plot
"""

from __future__ import annotations

import sys, time, argparse
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.core.reference_path import StraightReference, CircularReference, GeneralPath
from Cartest.planning.warmstart import build_initial_mu
from Cartest.planning.cost import make_objective, make_objective_p1, build_context
from Cartest.execution.execute import execute_perfect_tracking, FrenetState
from Cartest.planning.constraints import make_constraints
from gmm_igo.solver_builder import build_solver

ROOT = Path(__file__).resolve().parent
BASIS = ROOT / "basis"


def run_test(name: str, ref_path, scenario_cfg: dict, *,
             steps: int = 60, seed: int = 42, verbose: bool = True):
    """Run two-phase MPC on a given reference path and scenario.

    Returns:
        dict with keys: d_hist, v_hist, a_lat_hist, jac_min_hist,
        kappa_range, t_p1_avg, t_p2_avg, converge_step, d_violation
    """
    gen = FrenetBSplineTrajectory(BASIS / "bspline_basis.npz", ref_path)

    lane_hw   = scenario_cfg["lane_hw"]
    safe_dist = scenario_cfg.get("obs_safe_dist", 0.1)
    v_target  = scenario_cfg["v_target"]
    d_min     = scenario_cfg.get("d_min", None)
    d_max     = scenario_cfg.get("d_max", None)
    acc_max   = scenario_cfg.get("acc_max", 5.0)
    jerk_max  = scenario_cfg.get("jerk_max", 2.0)
    init      = scenario_cfg["init"]

    obs_pos = jnp.array(scenario_cfg.get("obs_pos", []), dtype=jnp.float32)
    if obs_pos.ndim == 1 and obs_pos.size == 0:
        obs_pos = jnp.zeros((0, 2), dtype=jnp.float32)
    obs_rad = jnp.array(scenario_cfg.get("obs_rad", []), dtype=jnp.float32)
    if obs_rad.ndim == 1 and obs_rad.size == 0:
        obs_rad = jnp.zeros(0, dtype=jnp.float32)

    # ── Phase 1: exploration — weak lane preference, constraints handle safety ──
    d_target = scenario_cfg["init"].get("d_target", scenario_cfg["init"]["d"])
    solver_p1 = build_solver(
        make_objective_p1(gen, d_target=d_target),
        dims=(gen.n_free, gen.n_free),
        constraints=make_constraints(gen, lane_hw, safe_dist,
                                     acc_max=acc_max, jerk_max=jerk_max,
                                     d_min=d_min, d_max=d_max),
        solver='m22', T=200, dt=0.25, K=3, B=128, B0=50, T_0=100,
        k_inner=1.0, obj_transform='standard',
    )

    # ── Phase 2: refinement (light — already near optimum from P1 warmstart) ──
    solver_p2 = build_solver(
        make_objective(gen, omega_s=1.0, omega_d=4.0, alpha=0.0),
        dims=(gen.n_free, gen.n_free),
        constraints=make_constraints(gen, lane_hw, safe_dist,
                                     acc_max=acc_max, jerk_max=jerk_max,
                                     d_min=d_min, d_max=d_max),
        solver='m22', T=100, dt=0.25, K=3, B=48, B0=30, T_0=101,
        k_inner=1.0, obj_transform='standard',
    )

    key = random.PRNGKey(seed)
    state = FrenetState(
        s=init["s"], s_dot=init["s_dot"], s_ddot=init.get("s_ddot", 0.0),
        d=init["d"], d_dot=init.get("d_dot", 0.0), d_ddot=init.get("d_ddot", 0.0),
        psi=init.get("psi", 0.0),
    )

    # ── JIT warmup: compile both initial_mu and warm_start paths ──
    ctx_warm = build_context(gen, state, v_target, lane_hw, obs_pos, obs_rad)
    mu_warm = build_initial_mu(gen, state.s, state.s_dot, state.d)
    _ = solver_p1(random.PRNGKey(999), context=ctx_warm, initial_mu=mu_warm)
    _ = solver_p2(random.PRNGKey(998), context=ctx_warm, initial_mu=mu_warm)
    # Also compile the warm_start path (P2 receives P1's GMM state)
    r_warm = solver_p1(random.PRNGKey(997), context=ctx_warm, initial_mu=mu_warm)
    _ = solver_p2(random.PRNGKey(996), context=ctx_warm, warm_start=r_warm)

    d_hist, v_hist, a_lat_hist = [], [], []
    jac_min_hist, kappa_hist = [], []
    t_p1_hist, t_p2_hist = [], []
    d_violation_hist = []
    diag_d_p1 = []   # P1's d_ref at t=1 (what P1 intended)
    diag_d_p2 = []   # P2's planned d at t=1 (what P2 planned)
    diag_cost_p1 = [] # P1 cost
    diag_cost_p2 = [] # P2 cost
    n_free = gen.n_free

    for step in range(steps):
        key, k1, k2 = random.split(key, 3)
        ctx = build_context(gen, state, v_target, lane_hw, obs_pos, obs_rad)
        mu_init = build_initial_mu(gen, state.s, state.s_dot, state.d)

        # ── Phase 1 ──
        t0 = time.time()
        result_p1 = solver_p1(k1, context=ctx, initial_mu=mu_init)
        t_p1 = (time.time() - t0) * 1000

        # ── Extract z_ref ──
        ctrl_s_p1, ctrl_d_p1 = result_p1.x[:n_free], result_p1.x[n_free:]
        frenet_p1 = gen.evaluate(
            ctrl_s_p1, ctrl_d_p1,
            ctx["s0"], ctx["s_dot0"], ctx["s_ddot0"],
            ctx["d0"], ctx["d_dot0"], ctx["d_ddot0"])
        z_ref = {
            's_ref': frenet_p1[0], 's_dot_ref': frenet_p1[2],
            's_ddot_ref': frenet_p1[4],
            'd_ref': frenet_p1[1], 'd_dot_ref': frenet_p1[3],
            'd_ddot_ref': frenet_p1[5],
        }

        # ── Phase 2 ──
        ctx_p2 = {**ctx, 'z_ref': z_ref}
        t0 = time.time()
        result_p2 = solver_p2(k2, context=ctx_p2, warm_start=result_p1)
        t_p2 = (time.time() - t0) * 1000

        ctrl_s, ctrl_d = result_p2.x[:n_free], result_p2.x[n_free:]
        frenet, st, (x_cart, y_cart) = gen.evaluate_plan(ctrl_s, ctrl_d, ctx)
        s_arr, d_arr, s_dot, d_dot, s_ddot, d_ddot, _, _ = frenet

        # ── Diagnostics ──
        # Jacobian at all sample points
        _, _, _, kappa_arr = ref_path.evaluate(s_arr)
        jac = 1.0 - d_arr * kappa_arr
        jac_min = float(jnp.min(jnp.abs(jac)))

        # Kinematic consistency: v² ≈ (1-d·κ)²·s_dot² + d_dot²
        vt = jac * s_dot
        v_pred = jnp.sqrt(vt ** 2 + d_dot ** 2)
        v_actual = st[:, 2]

        # Lane violation
        if d_min is not None:
            viol_lo = jnp.maximum(0., d_min - d_arr)
        else:
            viol_lo = jnp.zeros_like(d_arr)
        if d_max is not None:
            viol_hi = jnp.maximum(0., d_arr - d_max)
        else:
            viol_hi = jnp.zeros_like(d_arr)
        d_viol = float(jnp.max(jnp.maximum(viol_lo, viol_hi)))

        # Execute
        state = execute_perfect_tracking(
            s_arr, d_arr, s_dot, d_dot, s_ddot, d_ddot, st[1, 3])

        d_hist.append(float(state.d))
        v_hist.append(float(state.s_dot))
        # Diagnostics: P1 intent vs P2 plan vs actual execution
        diag_d_p1.append(float(z_ref['d_ref'][1]))       # P1: intended d at t=0.1s
        diag_d_p2.append(float(d_arr[1]))                 # P2: planned d at t=0.1s
        diag_cost_p1.append(float(result_p1.cost))
        diag_cost_p2.append(float(result_p2.cost))
        a_lat_hist.append(float(st[0, 5]))
        jac_min_hist.append(jac_min)
        kappa_hist.append(float(kappa_arr[0]))  # kappa at current s
        d_violation_hist.append(d_viol)
        t_p1_hist.append(t_p1)
        t_p2_hist.append(t_p2)

        # Divergence check
        if abs(float(state.d)) > 100.0 or not np.isfinite(float(state.s_dot)):
            print(f"  DIVERGENCE at step {step}: d={state.d:.0f} v={state.s_dot:.0f}")
            break

        if verbose and step % 5 == 0:
            d_p1 = float(z_ref['d_ref'][1])
            d_p2 = float(d_arr[1])
            print(f"  step {step:3d} | P1→{d_p1:+.3f} P2→{d_p2:+.3f} "
                  f"exec={state.d:+.3f} v={state.s_dot:.1f} "
                  f"| cost: P1={result_p1.cost:.1f} P2={result_p2.cost:.1f} "
                  f"| jac={jac_min:.3f}")

    # ── Summary ──
    d_arr_np = np.array(d_hist)
    v_arr_np = np.array(v_hist)
    jac_arr = np.array(jac_min_hist)

    d_target = init.get("d_target", init["d"])
    d_error = abs(float(d_arr_np[-1]) - d_target)
    overshoot = float(np.max(np.abs(d_arr_np - d_target)))  # max deviation from target
    d_steady = d_arr_np[-min(20, len(d_arr_np)):]
    oscillation = float(np.std(d_steady)) if len(d_steady) >= 20 else 0.0

    # Settling time: first step where |d-d_target| < 0.2 for 5 consecutive
    settle = steps
    for i in range(5, len(d_arr_np)):
        if all(abs(d_arr_np[j] - d_target) < 0.2 for j in range(i, min(i + 5, len(d_arr_np)))):
            settle = i
            break

    v_error = abs(float(v_arr_np[-1]) - v_target)

    result = {
        "name": name,
        "d_initial": float(d_arr_np[0]),
        "d_final": float(d_arr_np[-1]),
        "d_target": d_target,
        "d_error": d_error,
        "overshoot": overshoot,
        "oscillation": oscillation,
        "settle_s": settle * 0.1,
        "v_initial": float(v_arr_np[0]),
        "v_final": float(v_arr_np[-1]),
        "v_target": v_target,
        "v_error": v_error,
        "jac_min_abs": float(np.min(jac_arr)) if len(jac_arr) > 0 else 1.0,
        "d_viol_max": float(np.max(d_violation_hist)),
        "a_lat_max": float(np.max(np.abs(a_lat_hist))),
        "t_p1_avg": float(np.mean(t_p1_hist)),
        "t_p2_avg": float(np.mean(t_p2_hist)),
        "t_total_avg": float(np.mean(t_p1_hist) + np.mean(t_p2_hist)),
        "diverged": abs(float(d_arr_np[-1])) > 100.0 or not np.isfinite(float(v_arr_np[-1])),
    }

    if verbose:
        ok = (not result["diverged"]
              and result["d_viol_max"] < 0.01
              and result["jac_min_abs"] > 0.5
              and result["d_error"] < 0.3
              and result["oscillation"] < 0.5)
        status = "OK" if ok else "FAIL"
        print(f"\n  [{status}] {name}")
        print(f"    d: {result['d_initial']:+.1f}→{result['d_final']:+.2f} "
              f"(target={d_target:+.1f})  err={d_error:.2f}m  "
              f"overshoot={overshoot:.2f}m  osc={oscillation:.3f}  "
              f"settle={result['settle_s']:.1f}s")
        print(f"    v: {result['v_initial']:.0f}→{result['v_final']:.1f} "
              f"(target={v_target:.0f})  err={v_error:.1f}m/s")
        print(f"    P1→{np.mean(diag_d_p1):+.3f}  P2→{np.mean(diag_d_p2):+.3f}  "
              f"exec={np.mean(d_hist):+.3f}  "
              f"(avg P1 plan / P2 plan / exec at t=0.1s)")
        print(f"    cost: P1_avg={np.mean(diag_cost_p1):.0f}  P2_avg={np.mean(diag_cost_p2):.0f}")
        print(f"    jac_min={result['jac_min_abs']:.3f}  "
              f"a_lat_max={result['a_lat_max']:.1f}  d_viol={result['d_viol_max']:.4f}")
        print(f"    P1={result['t_p1_avg']:.0f}ms  P2={result['t_p2_avg']:.0f}ms  "
              f"total={result['t_total_avg']:.0f}ms")

    return result


# ═══════════════════════════════════════════════════════════════════════
# Test matrix
# ═══════════════════════════════════════════════════════════════════════

def make_scenario(v_target: float, init_d: float, d_target: float,
                  init_v: float, lane_hw: float = 4.0,
                  d_min: float | None = None, d_max: float | None = None,
                  acc_max: float = 5.0, jerk_max: float = 2.0,
                  obs_list=None):
    """Build scenario dict for a single test case."""
    sc = {
        "lane_hw": lane_hw,
        "obs_safe_dist": 0.1,
        "v_target": v_target,
        "d_min": d_min if d_min is not None else -lane_hw,
        "d_max": d_max if d_max is not None else lane_hw,
        "acc_max": acc_max,
        "jerk_max": jerk_max,
        "init": {
            "s": 0.0, "s_dot": init_v, "s_ddot": 0.0,
            "d": init_d, "d_dot": 0.0, "d_ddot": 0.0,
            "psi": 0.0, "d_target": d_target,
        },
    }
    if obs_list:
        sc["obs_pos"] = jnp.array([[o["x"], o["y"]] for o in obs_list], dtype=jnp.float32)
        sc["obs_rad"] = jnp.array([o["r"] for o in obs_list], dtype=jnp.float32)
    return sc


def main():
    parser = argparse.ArgumentParser(description="Test P1 Frenet transform on variable-curvature paths")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "straight", "circle_100", "circle_20", "s_bend", "s_bend_obs"],
                        help="Which test(s) to run")
    parser.add_argument("--steps", type=int, default=60,
                        help="Number of MPC steps")
    parser.add_argument("--no-plot", action="store_true", default=True,
                        help="Disable plotting (always true for headless)")
    args = parser.parse_args()

    path_s_bend = BASIS / "path_s_bend.npz"
    if not path_s_bend.exists():
        print(f"Building S-bend path: {path_s_bend}")
        import subprocess
        subprocess.run([
            sys.executable, str(BASIS / "build_path.py"),
            "--type", "sinusoid", "--R-min", "20", "--n-cycles", "3",
            "--total-length", "200", "--out", str(path_s_bend),
        ], check=True)

    tests_to_run = []

    # T1: Straight — baseline lane change: d=-3→0, v=12→18
    if args.test in ("all", "straight"):
        tests_to_run.append(("T1_straight", StraightReference(),
            make_scenario(v_target=18.0, init_d=-3.0, d_target=0.0, init_v=12.0)))

    # T2: R=100m circle — constant κ, lane change + speed change
    if args.test in ("all", "circle_100"):
        tests_to_run.append(("T2_circle_R100", CircularReference(100.0, 0.0, 0.0),
            make_scenario(v_target=16.0, init_d=-3.0, d_target=0.0, init_v=12.0)))

    # T3: R=20m circle — tight turn, low speed lane change
    if args.test in ("all", "circle_20"):
        tests_to_run.append(("T3_circle_R20", CircularReference(20.0, 0.0, 0.0),
            make_scenario(v_target=10.0, init_d=-3.0, d_target=0.0, init_v=6.0)))

    # T4: S-bend R_min=20m — variable curvature, main test
    if args.test in ("all", "s_bend"):
        tests_to_run.append(("T4_s_bend", GeneralPath(str(path_s_bend)),
            make_scenario(v_target=10.0, init_d=-3.0, d_target=0.0, init_v=6.0)))

    # T5: S-bend + obstacle — obs safety + lane change
    if args.test in ("all", "s_bend_obs"):
        tests_to_run.append(("T5_s_bend_obs", GeneralPath(str(path_s_bend)),
            make_scenario(v_target=10.0, init_d=-3.0, d_target=0.0, init_v=6.0,
                          obs_list=[{"x": 80.0, "y": 5.0, "r": 2.0}])))

    print(f"Running {len(tests_to_run)} test(s) on {args.steps} steps each\n")

    results = []
    for name, ref_path, sc in tests_to_run:
        print(f"═══ {name} ═══")
        d_tgt = sc["init"]["d_target"]
        print(f"  path: {type(ref_path).__name__}")
        print(f"  d: {sc['init']['d']:+.1f}→{d_tgt:+.1f}  v: {sc['init']['s_dot']:.0f}→{sc['v_target']:.0f}  "
              f"d_range=[{sc['d_min']:.0f},{sc['d_max']:.0f}]")
        r = run_test(name, ref_path, sc, steps=args.steps)
        results.append(r)
        print()

    # ── Summary table ──
    print("═══ Summary ═══")
    print(f"{'Test':<18} {'d':>8} {'target':>7} {'err':>6} {'overshoot':>8} "
          f"{'osc':>6} {'settle':>7} {'v':>7} {'j_min':>6} {'t':>7} {'st':>4}")
    print("-" * 100)
    all_ok = True
    for r in results:
        ok = (not r["diverged"] and r["d_viol_max"] < 0.01
              and r["d_error"] < 0.3 and r["oscillation"] < 0.5
              and r["jac_min_abs"] > 0.5)
        all_ok = all_ok and ok
        print(f"{r['name']:<18} {r['d_final']:+7.2f} {r['d_target']:+6.1f} "
              f"{r['d_error']:5.2f} {r['overshoot']:7.2f} "
              f"{r['oscillation']:5.3f} {r['settle_s']:6.1f}s "
              f"{r['v_final']:6.1f} {r['jac_min_abs']:5.3f} "
              f"{r['t_total_avg']:6.0f}ms {'OK' if ok else 'FAIL':>4}")
    print()

    if all_ok:
        print("All tests passed.")
    else:
        print("Some tests FAILED — check divergence or constraint violations above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
