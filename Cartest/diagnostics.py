"""Cost diagnostics — raw values + actual cost, no manual nesting trace.

Reports what the optimizer actually sees: raw objective, constraint
violations, and the resulting Constran-built cost.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def diagnose(gen, theta, ctx, obs_safe_dist: float = 0.5, state=None):
    """Extract raw values behind the Constran-built cost.

    Does NOT manually trace the nesting — just reports raw objective
    and per-constraint g_raw so we can see what the optimizer is fighting.
    If *state* (FrenetState) is given, also shows vehicle's current distance.
    """
    n = gen.n_free
    ctrl_s, ctrl_d = theta[:n], theta[n:]

    s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
        ctrl_s, ctrl_d,
        ctx["s0"], ctx["s_dot0"], ctx["s_ddot0"],
        ctx["d0"], ctx["d_dot0"], ctx["d_ddot0"],
    )
    st = gen.to_vehicle_states(s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot)
    v = st[:, 2]
    a_long, a_lat = st[:, 4], st[:, 5]
    j_long, j_lat = st[:, 6], st[:, 7]
    x_cart, y_cart = gen.to_cartesian(s, d)

    # ── Raw objective (matches cost.py: s_dot, not v) ──
    raw_speed = float((s_dot[-1] - ctx["v_ref"][-1]) ** 2)
    raw_lat   = float(jnp.sum(d ** 2))
    raw_ddot  = float(jnp.sum(d_dot ** 2))
    raw_obj   = raw_speed + raw_lat + 0.1 * raw_ddot

    # ── Raw g values (before T_alpha) ──
    am = jnp.sqrt(a_long ** 2 + a_lat ** 2)
    jm = jnp.sqrt(j_long ** 2 + j_lat ** 2)

    g_spd_max = float(jnp.max(jnp.maximum(jnp.maximum(0., 2.0 - v), jnp.maximum(0., v - 35.0))))
    g_acc_max = float(jnp.max(
        jnp.maximum(jnp.maximum(0., jnp.abs(a_long) - 5.0),
                    jnp.maximum(jnp.maximum(0., jnp.abs(a_lat) - 5.0),
                                jnp.maximum(0., am - 5.0)))))
    g_jerk_max = float(jnp.max(
        jnp.maximum(jnp.maximum(0., jnp.abs(j_long) - 5.0),
                    jnp.maximum(jnp.maximum(0., jnp.abs(j_lat) - 5.0),
                                jnp.maximum(0., jm - 5.0)))))
    g_lane_max = float(jnp.max(jnp.maximum(0., jnp.abs(d) - ctx["lane_hw"])))
    rho = obs_safe_dist
    d_rss = v * rho + v ** 2 / (2.0 * 8.0)
    dx = x_cart[:, None] - ctx["obs_pos"][None, :, 0]
    dy = y_cart[:, None] - ctx["obs_pos"][None, :, 1]
    r  = ctx["obs_rad"][None, :]
    pen_x = jnp.maximum(0., d_rss[:, None] + r - jnp.abs(dx))
    pen_y = jnp.maximum(0., r - jnp.abs(dy))
    g_obs_max = float(jnp.max(jnp.maximum(pen_x, pen_y)))

    # Vehicle's current distance to nearest obstacle
    cur_dist = None
    if state is not None:
        cur_dx = state.s - ctx["obs_pos"][:, 0]
        cur_dy = state.d - ctx["obs_pos"][:, 1]
        cur_dist = float(jnp.min(jnp.sqrt(cur_dx ** 2 + cur_dy ** 2) - ctx["obs_rad"]))

    return {
        'raw_obj': raw_obj,
        'raw_speed': raw_speed, 'raw_lat': raw_lat, 'raw_ddot': raw_ddot,
        's_dot_mean': float(jnp.mean(s_dot)), 's_dot_min': float(jnp.min(s_dot)), 's_dot_max': float(jnp.max(s_dot)),
        'd_rms': float(jnp.sqrt(jnp.mean(d ** 2))),
        'd_min': float(jnp.min(d)), 'd_max': float(jnp.max(d)),
        'cur_obs': cur_dist,
        'g_max': {'obs': g_obs_max, 'lane': g_lane_max, 'spd': g_spd_max,
                  'acc': g_acc_max, 'jerk': g_jerk_max},
        'a_long_max': float(jnp.max(jnp.abs(a_long))),
        'a_lat_max':  float(jnp.max(jnp.abs(a_lat))),
        'jerk_max':   float(jnp.max(jm)),
    }


def print_diag(d):
    print(f"  DIAG: raw_obj={d['raw_obj']:.0f} "
          f"speed={d['raw_speed']:.0f} lat={d['raw_lat']:.0f} ddot={d['raw_ddot']:.0f}")
    print(f"        s_dot_mean={d['s_dot_mean']:.1f} s_dot=[{d['s_dot_min']:.1f},{d['s_dot_max']:.1f}]  "
          f"d_rms={d['d_rms']:.2f} d_range=[{d['d_min']:.1f},{d['d_max']:.1f}]")
    if d['cur_obs'] is not None:
        print(f"        cur_obs={d['cur_obs']:.1f}m  ", end="")
    print(f"g_max: obs={d['g_max']['obs']:.3f} lane={d['g_max']['lane']:.3f} "
          f"spd={d['g_max']['spd']:.3f} acc={d['g_max']['acc']:.3f} jerk={d['g_max']['jerk']:.3f}")
    print(f"        max|a_long|={d['a_long_max']:.1f} max|a_lat|={d['a_lat_max']:.1f} "
          f"max|jerk|={d['jerk_max']:.1f}")
