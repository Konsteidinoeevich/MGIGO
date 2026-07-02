"""Cost function for Frenet B-spline MPC.

Objective: lateral tracking (d → 0) + speed tracking (v → v_target).

Both quantities from frenet_traj — cost and constraints share the
same vehicle-state pipeline, no more mixing raw Frenet derivatives.
"""

from __future__ import annotations

import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════
# Context builder
# ═══════════════════════════════════════════════════════════════════════

def build_context(gen, state, v_ref, lane_hw, obs_pos, obs_rad):
    """Build ctx dict for cost/constraint evaluation.

    Args:
        gen:     FrenetBSplineTrajectory
        state:   FrenetState
        v_ref:   scalar or [T] reference speed
        lane_hw: scalar lane half‑width
        obs_pos: [N, 2], obs_rad: [N]
    """
    return {
        'v_ref':   jnp.full(gen.T, v_ref) if isinstance(v_ref, (int, float)) else v_ref,
        'lane_hw': lane_hw,
        'obs_pos': obs_pos,
        'obs_rad': obs_rad,
        **state.to_ctx(),
    }


# ═══════════════════════════════════════════════════════════════════════
# Objective
# ═══════════════════════════════════════════════════════════════════════

def _eval_all(theta, ctx, gen):
    """Unpack theta → Frenet trajectory.

    Returns (d, d_dot, s_dot, s_ddot) — lateral position/velocity, longitudinal velocity/accel.
    """
    n = gen.n_free
    ctrl_s_free = theta[:n]
    ctrl_d_free = theta[n:2 * n]

    s, d, s_dot, d_dot, s_ddot, d_ddot, _, _ = gen.evaluate(
        ctrl_s_free, ctrl_d_free,
        ctx["s0"], ctx["s_dot0"], ctx["s_ddot0"],
        ctx["d0"], ctx["d_dot0"], ctx["d_ddot0"],
    )
    return d, d_dot, s_dot, s_ddot


def make_objective(gen):
    """Build objective aligned with integrator chains.

    d-channel (3-integrator):  d⃛ → d̈ → d_dot → d
      virtual control:  d_dot* = -k₁·d
      cost:              Σ (d_dot + k₁·d)²        — damps automatically

    s-channel (3-integrator):  s⃛ → s̈ → s_dot → s
      virtual control:  s̈* = -k₂·(s_dot − v_target)
      cost:              Σ (s_dot − v_target)² + Σ (s̈ + k₂·(s_dot − v_target))²
                         — accel steers speed toward target, not against it
    """

    k_lat = 0.5   # lateral-convergence rate (≈ bandwidth of d_channel)
    k_spd = 0.5   # speed-tracking rate  (≈ bandwidth of s_channel)

    def obj_fn(theta, ctx):
        d, d_dot, s_dot, s_ddot = _eval_all(theta, ctx, gen)

        # d-channel: penalise departure from d_dot* = -k_lat·d
        lat_cost = jnp.sum((d_dot + k_lat * d) ** 2)

        # s-channel: speed tracking + accel steering
        speed_err = s_dot - ctx["v_ref"]
        spd_cost = jnp.sum(speed_err ** 2)
        acc_cost = jnp.sum((s_ddot + k_spd * speed_err) ** 2)

        return lat_cost + spd_cost + 0.1 * acc_cost

    return obj_fn
