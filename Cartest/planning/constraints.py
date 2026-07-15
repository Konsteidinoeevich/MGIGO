"""Constraint builders for Frenet B-spline MPC.

All kinematics constraints (acc, jerk, speed) use to_vehicle_states()
— the correct Frenet→vehicle transformation with curvature coupling.

Lane and obstacle use Frenet / Cartesian directly (no curvature coupling needed).

Per-sample penalty: max(max(0, |long|-LIM), max(0, |lat|-LIM))
Only the *worse* component is penalised, not both.
"""

from __future__ import annotations

import jax.numpy as jnp
from Constraintdealer.Constran import Deterministic


# ═══════════════════════════════════════════════════════════════════════
# Constraint parameters
# ═══════════════════════════════════════════════════════════════════════

# Physical limits (hardware — not scenario-specific)
V_MIN, V_MAX = 2.0, 35.0
ACC_MAX = 5.0          # m/s²
JERK_MAX = 2.0         # m/s³  (tight: comfort limit)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _eval_frenet(theta, ctx, gen):
    """Unpack theta → evaluate Frenet trajectory.

    theta = [ctrl_s_free(9) | ctrl_d_free(9)].
    """
    n = gen.n_free
    return gen.evaluate(
        theta[:n], theta[n:2 * n],
        ctx["s0"], ctx["s_dot0"], ctx["s_ddot0"],
        ctx["d0"], ctx["d_dot0"], ctx["d_ddot0"],
    )


def _eval_vehicle_states(theta, ctx, gen):
    """Unpack theta → evaluate → vehicle states [T, 9]."""
    s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = _eval_frenet(theta, ctx, gen)
    return gen.to_vehicle_states(s, d, s_dot, d_dot,
                                 s_ddot, d_ddot, s_dddot, d_dddot)


# ═══════════════════════════════════════════════════════════════════════
# Constraint factory
# ═══════════════════════════════════════════════════════════════════════

def make_constraints(gen, lane_hw: float, obs_safe_dist: float,
                     acc_max: float | None = None,
                     jerk_max: float | None = None,
                     d_min: float | None = None,
                     d_max: float | None = None):
    """Build constraint list for Frenet B-spline MPC.

    Self-similar σ nesting (outer = larger priority number = higher precedence):
      P1 (inner):  obs       — obstacle safety
      P2:          lane      — lane keeping
      P3:          speed     — speed limits
      P4:          acc       — acceleration
      P5 (outer):  jerk      — comfort / control input
    Outer constraints have the final say in the σ-nesting chain.

    Args:
        lane_hw:  symmetric half-width when d_min/d_max are None
        acc_max:  override module‑level ACC_MAX (None → use default 5.0)
        jerk_max: override module‑level JERK_MAX (None → use default 2.0)
        d_min:    lower lane boundary (Frenet d).  None → -lane_hw.
        d_max:    upper lane boundary (Frenet d).  None → +lane_hw.
                  Set d_max=0.0 to forbid right lane change, etc.
    """
    _acc_max = acc_max if acc_max is not None else ACC_MAX
    _jerk_max = jerk_max if jerk_max is not None else JERK_MAX
    _d_min = d_min if d_min is not None else -lane_hw
    _d_max = d_max if d_max is not None else lane_hw

    def obs_g(theta, ctx):
        """RSS: longitudinal + lateral safe distance per obstacle."""
        n_obs = ctx["obs_pos"].shape[0]
        if n_obs == 0:
            return jnp.zeros(gen.T)  # no obstacles → no violation

        st = _eval_vehicle_states(theta, ctx, gen)
        x, y, v = st[:, 0], st[:, 1], st[:, 2]
        rho = obs_safe_dist
        a_brake = 8.0

        d_rss = v * rho + v ** 2 / (2.0 * a_brake)                    # [T]

        dx = x[:, None] - ctx["obs_pos"][None, :, 0]                  # [T, N]
        dy = y[:, None] - ctx["obs_pos"][None, :, 1]                  # [T, N]
        r  = ctx["obs_rad"][None, :]                                   # [1, N]

        pen_x = jnp.maximum(0., d_rss[:, None] + r - jnp.abs(dx))
        pen_y = jnp.maximum(0., r - jnp.abs(dy))

        return jnp.maximum(pen_x, pen_y).max(axis=-1)  # worst axis × worst obs

    def lane_g(theta, ctx):
        """Lane boundary: d_min ≤ d ≤ d_max.  d from Frenet directly."""
        _, d, _, _, _, _, _, _ = _eval_frenet(theta, ctx, gen)
        return jnp.maximum(
            jnp.maximum(0., _d_min - d),
            jnp.maximum(0., d - _d_max),
        )

    def kinematics_g(theta, ctx):
        """Fused speed + acc + jerk from a single to_vehicle_states call."""
        st = _eval_vehicle_states(theta, ctx, gen)     # ← 1 call, shared
        v = st[:, 2]
        a_long, a_lat = st[:, 4], st[:, 5]
        j_long, j_lat = st[:, 6], st[:, 7]
        am = jnp.sqrt(a_long ** 2 + a_lat ** 2)
        jm = jnp.sqrt(j_long ** 2 + j_lat ** 2)

        g_speed = jnp.maximum(jnp.maximum(0., V_MIN - v), jnp.maximum(0., v - V_MAX))
        g_acc = jnp.maximum(
            jnp.maximum(0., jnp.abs(a_long) - _acc_max),
            jnp.maximum(jnp.maximum(0., jnp.abs(a_lat) - _acc_max),
                        jnp.maximum(0., am - _acc_max)),
        )
        g_jerk = jnp.maximum(
            jnp.maximum(0., jnp.abs(j_long) - _jerk_max),
            jnp.maximum(jnp.maximum(0., jnp.abs(j_lat) - _jerk_max),
                        jnp.maximum(0., jm - _jerk_max)),
        )
        return g_speed + g_acc + g_jerk  # element-wise sum → [T]

    return [
        Deterministic(obs_g,        mode='hard', priority=1, aggregate='max',
                       transform='hard'),
        Deterministic(lane_g,       mode='soft', priority=2, aggregate='q95',
                       transform='soft'),
        Deterministic(kinematics_g, mode='soft', priority=3, aggregate='max',
                       transform='soft'),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Metrics — reusable g‑value computation (matches constraint formulas)
# ═══════════════════════════════════════════════════════════════════════

def compute_g_values(st, d, x_cart, y_cart, obs_pos, obs_rad,
                     lane_hw: float, obs_safe_dist: float,
                     d_min: float | None = None,
                     d_max: float | None = None):
    """Compute per‑constraint g‑values for reporting.

    Args:
        st:      [T, 9] vehicle states
        d:       [T] Frenet lateral offset
        x_cart, y_cart: [T] Cartesian positions
        obs_pos: [N, 2], obs_rad: [N]
        lane_hw, obs_safe_dist: scenario parameters
        d_min:   lower lane boundary (None → -lane_hw)
        d_max:   upper lane boundary (None → +lane_hw)

    Returns:
        dict with keys lane, obs, jerk, acc, speed (obs=max, rest=q90).
    """
    v = st[:, 2]
    a_long, a_lat = st[:, 4], st[:, 5]
    j_long, j_lat = st[:, 6], st[:, 7]
    am = jnp.sqrt(a_long ** 2 + a_lat ** 2)
    jm = jnp.sqrt(j_long ** 2 + j_lat ** 2)

    _d_min = d_min if d_min is not None else -lane_hw
    _d_max = d_max if d_max is not None else lane_hw

    g_lane = jnp.quantile(
        jnp.maximum(
            jnp.maximum(0., _d_min - d),
            jnp.maximum(0., d - _d_max),
        ), 0.9)

    if obs_pos.shape[0] == 0:
        g_obs = 0.0
    else:
        rho = obs_safe_dist
        d_rss = v * rho + v ** 2 / (2.0 * 8.0)
        dx = x_cart[:, None] - obs_pos[None, :, 0]
        dy = y_cart[:, None] - obs_pos[None, :, 1]
        r  = obs_rad[None, :]
        pen_x = jnp.maximum(0., d_rss[:, None] + r - jnp.abs(dx))
        pen_y = jnp.maximum(0., r - jnp.abs(dy))
        g_obs = float(jnp.max(jnp.maximum(pen_x, pen_y)))

    g_jerk = float(jnp.max(
        jnp.maximum(
            jnp.maximum(0., jnp.abs(j_long) - JERK_MAX),
            jnp.maximum(jnp.maximum(0., jnp.abs(j_lat) - JERK_MAX),
                        jnp.maximum(0., jm - JERK_MAX)),
        )))  # max — matches constraint aggregate

    g_acc = float(jnp.max(
        jnp.maximum(
            jnp.maximum(0., jnp.abs(a_long) - ACC_MAX),
            jnp.maximum(jnp.maximum(0., jnp.abs(a_lat) - ACC_MAX),
                        jnp.maximum(0., am - ACC_MAX)),
        )))  # max — matches constraint aggregate

    g_spd = jnp.quantile(
        jnp.maximum(jnp.maximum(0., V_MIN - v), jnp.maximum(0., v - V_MAX)), 0.9)

    return {
        'lane': float(g_lane), 'obs': float(g_obs),
        'jerk': float(g_jerk), 'acc': float(g_acc), 'spd': float(g_spd),
    }


def compute_summary(st, d, x_cart, y_cart, obs_pos, obs_rad):
    """Compute summary metrics: min_obs_dist, max |a_long|, |a_lat|, |jerk|."""
    a_long, a_lat = st[:, 4], st[:, 5]
    j_long, j_lat = st[:, 6], st[:, 7]
    jm = jnp.sqrt(j_long ** 2 + j_lat ** 2)

    if obs_pos.shape[0] == 0:
        min_obs = 1e9
    else:
        dist = jnp.sqrt((x_cart[:, None] - obs_pos[None, :, 0]) ** 2 +
                        (y_cart[:, None] - obs_pos[None, :, 1]) ** 2) - obs_rad[None, :]
    return {
        'min_obs': min_obs,
        'max_a_long': float(jnp.max(jnp.abs(a_long))),
        'max_a_lat':  float(jnp.max(jnp.abs(a_lat))),
        'max_jerk':   float(jnp.max(jm)),
        'v':          float(st[0, 2]),
    }
