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
    """Unpack theta → Frenet trajectory → vehicle states.

    Returns (d, v) — the two quantities the objective cares about.
    """
    n = gen.n_free
    ctrl_s_free = theta[:n]
    ctrl_d_free = theta[n:2 * n]

    s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
        ctrl_s_free, ctrl_d_free,
        ctx["s0"], ctx["s_dot0"], ctx["s_ddot0"],
        ctx["d0"], ctx["d_dot0"], ctx["d_ddot0"],
    )

    st = gen.to_vehicle_states(s, d, s_dot, d_dot,
                               s_ddot, d_ddot, s_dddot, d_dddot)
    v = st[:, 2]
    return d, v


def make_objective(gen):
    """Build objective: d → 0 (lane centre) + v → v_target."""

    def obj_fn(theta, ctx):
        d, v = _eval_all(theta, ctx, gen)
        speed_cost = jnp.sum((v - ctx["v_ref"]) ** 2)
        lat_cost = jnp.sum(d ** 2)
        return speed_cost + lat_cost

    return obj_fn
