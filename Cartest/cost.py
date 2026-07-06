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
    """Unpack theta → Frenet trajectory (up to jerk, all quantities)."""
    n = gen.n_free
    ctrl_s_free = theta[:n]
    ctrl_d_free = theta[n:2 * n]

    s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
        ctrl_s_free, ctrl_d_free,
        ctx["s0"], ctx["s_dot0"], ctx["s_ddot0"],
        ctx["d0"], ctx["d_dot0"], ctx["d_ddot0"],
    )
    return s, d, d_dot, d_ddot, d_dddot, s_dot, s_ddot, s_dddot


def make_objective(gen, omega_s: float = 1.0, omega_d: float = 1.0,
                   alpha: float = 0.5):
    """Coupled s/d Lyapunov cost — both channels track position errors.

    e = [es, ed]  with  es = s−s_ref,  ed = d

    s_ref_dot(t) = v_tgt + (v0 − v_tgt)·e^(−ω_s·t)
    s_ref(t)     = s0 + v_tgt·t + (v0−v_tgt)/ω_s · (1−e^(−ω_s·t))

    α² < ω_s·ω_d  ensures K ≻ 0.
    """

    t_arr = jnp.arange(gen.T) * gen.dt  # [T]

    def obj_fn(theta, ctx):
        s, d, d_dot, d_ddot, d_dddot, s_dot, s_ddot, s_dddot = _eval_all(theta, ctx, gen)

        v_tgt = ctx["v_ref"][0]
        s0    = ctx["s0"]
        v0    = ctx["s_dot0"]
        dv    = v0 - v_tgt                              # speed gap (negative if accelerating)
        lam   = omega_s
        exp_term = jnp.exp(-lam * t_arr)                 # [T]
        s_ref      = s0 + v_tgt * t_arr + dv / lam * (1.0 - exp_term)
        s_ref_dot  = v_tgt + dv * exp_term
        s_ref_ddot = -dv * lam * exp_term

        # position errors (both channels symmetric now)
        es = s - s_ref
        ed = d

        es_dot = s_dot   - s_ref_dot
        ed_dot = d_dot

        es_ddot = s_ddot  - s_ref_ddot
        ed_ddot = d_ddot

        # K matrix and coupling (same as before)
        K00, K01 = omega_s, alpha
        K10, K11 = alpha,   omega_d

        K2_00 = K00**2 + K01*K10
        K2_01 = K00*K01 + K01*K11
        K2_10 = K10*K00 + K11*K10
        K2_11 = K10*K01 + K11**2

        term0 = es**2 + ed**2

        v1_s = es_dot + K00*es + K01*ed
        v1_d = ed_dot + K10*es + K11*ed
        term1 = v1_s**2 + v1_d**2

        v2_s = es_ddot + 2.0*(K00*es_dot + K01*ed_dot) + K2_00*es + K2_01*ed
        v2_d = ed_ddot + 2.0*(K10*es_dot + K11*ed_dot) + K2_10*es + K2_11*ed
        term2 = v2_s**2 + v2_d**2

        return jnp.sum(term0) + jnp.sum(term1) + jnp.sum(term2)

    return obj_fn
