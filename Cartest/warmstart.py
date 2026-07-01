"""Warm-start strategies for Frenet B-spline MPC.

Two kinds of warm-start:

  Physics-based — Greville abscissae × current speed → exact constant-speed
  trajectory (jerk=0, acc=0).  Used for the very first step and whenever a
  "fresh start" from the current vehicle state is needed.

  GMM inheritance — carry forward the solver's GMM state (mu, L, pi) from
  the previous MPC step.  The distribution is already concentrated near the
  previous optimum; only needs to adapt to the shifted horizon.  Much faster
  convergence in steady-state.

Exported helpers return solver‑ready kwargs for ``build_solver()``.
"""

from __future__ import annotations

import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════
# Low-level: control-point generation
# ═══════════════════════════════════════════════════════════════════════

def tangent_warmstart(gen, s0: float, v_target: float, d0: float = 0.0):
    """Constant-speed free control points (P3..P11) using Greville abscissae.

    Clamped P0,P1,P2 + these free points → exact constant speed *v_target*,
    exact zero acceleration, exact zero jerk.
    """
    ctrl_s = s0 + v_target * gen.greville[3:]
    ctrl_d = jnp.full((gen.n_free,), d0, dtype=jnp.float32)
    return ctrl_s, ctrl_d


def shift_warmstart(ctrl_s_old, ctrl_d_old, v_target: float, dt_knot: float):
    """Shift previous solution forward by one control-point index.

    Preserves solver‑optimised trajectory shape.  Useful when obstacle
    avoidance is active and regenerating from scratch would lose the
    learned avoidance manoeuvre.
    """
    n = len(ctrl_s_old)
    ctrl_s = jnp.zeros_like(ctrl_s_old)
    ctrl_d = jnp.zeros_like(ctrl_d_old)

    ctrl_s = ctrl_s.at[:-1].set(ctrl_s_old[1:])
    ctrl_d = ctrl_d.at[:-1].set(ctrl_d_old[1:])

    ctrl_s = ctrl_s.at[-1].set(ctrl_s_old[-1] + v_target * dt_knot)
    ctrl_d = ctrl_d.at[-1].set(ctrl_d_old[-1])

    return ctrl_s, ctrl_d


# ═══════════════════════════════════════════════════════════════════════
# High-level: solver‑ready initialisation
# ═══════════════════════════════════════════════════════════════════════

def build_initial_mu(gen, s0: float, s_dot0: float, d0: float = 0.0, K: int = 3):
    """Physics-based initial GMM means for ``build_solver(initial_mu=…)``.

    Returns (M=2, K, D_max) — all K components identical, centred on the
    constant‑speed Greville warm‑start.
    """
    ctrl_s, ctrl_d = tangent_warmstart(gen, s0, s_dot0, d0)
    return jnp.stack([
        jnp.stack([ctrl_s] * K, axis=0),
        jnp.stack([ctrl_d] * K, axis=0),
    ], axis=0).astype(jnp.float32)


def mpc_warmstart(gen, s0: float, s_dot0: float, d0: float = 0.0,
                  prev_result=None, K: int = 3):
    """Return solver kwargs for one MPC step.

    If *prev_result* is None (first step or reset):
        physics‑based ``initial_mu``, fresh L/pi.

    If *prev_result* is given:
        ``warm_start=prev_result`` — full GMM state inheritance.
    """
    if prev_result is not None:
        return {'warm_start': prev_result}
    return {'initial_mu': build_initial_mu(gen, s0, s_dot0, d0, K)}
