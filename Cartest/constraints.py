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
JERK_MAX = 5.0         # m/s³


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

def make_constraints(gen, lane_hw: float, obs_safe_dist: float):
    """Build constraint list for Frenet B-spline MPC.

    Self-similar nesting: 小priority=内层(被后续σ·m放大), 大priority=外层(直接输出)
      P1 (内): obstacle  — 安全约束, 天然最优先
      P2:     lane      — 车道约束
      P3:     speed     — 速度约束
      P4:     acc       — 加速度约束
      P5 (外): jerk      — 控制输入, 直接输出
    安全(obs/lane)自然优先于舒适(jerk/acc) — 靠结构保证, 不靠参数调
    """

    def obs_g(theta, ctx):
        """RSS: longitudinal braking + lateral clearance, per obstacle.

        Longitudinal:  if approaching an obstacle (dx decreasing),
                       must have enough distance to brake: |dx| ≥ d_RSS(v).
        Lateral:       |dy| ≥ r_obs  at the closest approach.
        Take the max of both, max over obstacles.
        """
        st = _eval_vehicle_states(theta, ctx, gen)
        x, y, v = st[:, 0], st[:, 1], st[:, 2]
        rho = obs_safe_dist
        a_brake = 8.0

        # Longitudinal RSS — only penalise when approaching (v > 0)
        # and when |dx| < d_RSS(v) — i.e. vehicle is too close for its speed
        d_rss = v * rho + v ** 2 / (2.0 * a_brake)                    # [T]

        dx = x[:, None] - ctx["obs_pos"][None, :, 0]                  # [T, N]
        dy = y[:, None] - ctx["obs_pos"][None, :, 1]                  # [T, N]
        r  = ctx["obs_rad"][None, :]                                   # [1, N]

        # Longitudinal violation: too close for current speed
        # Only when vehicle hasn't passed the obstacle (dx remains)
        pen_x = jnp.maximum(0., d_rss[:, None] + r - jnp.abs(dx))

        # Lateral violation: not enough clearance
        pen_y = jnp.maximum(0., r - jnp.abs(dy))

        return jnp.maximum(pen_x, pen_y).max(axis=-1)  # worst axis × worst obs

    def lane_g(theta, ctx):
        """Lane boundary |d| ≤ lane_hw.  d from Frenet directly."""
        _, d, _, _, _, _, _, _ = _eval_frenet(theta, ctx, gen)
        return jnp.maximum(0., jnp.abs(d) - lane_hw)

    def speed_g(theta, ctx):
        """Speed V_MIN ≤ v ≤ V_MAX.  v from to_vehicle_states."""
        st = _eval_vehicle_states(theta, ctx, gen)
        v = st[:, 2]
        return jnp.maximum(
            jnp.maximum(0., V_MIN - v),
            jnp.maximum(0., v - V_MAX),
        )

    def acc_g(theta, ctx):
        """Acc: max(|long|-LIM, |lat|-LIM, |total|-LIM, 0) per sample."""
        st = _eval_vehicle_states(theta, ctx, gen)
        a_long, a_lat = st[:, 4], st[:, 5]
        am = jnp.sqrt(a_long ** 2 + a_lat ** 2)
        return jnp.maximum(
            jnp.maximum(0., jnp.abs(a_long) - ACC_MAX),
            jnp.maximum(
                jnp.maximum(0., jnp.abs(a_lat) - ACC_MAX),
                jnp.maximum(0., am            - ACC_MAX),
            ),
        )

    def jerk_g(theta, ctx):
        """Jerk: max(|long|-LIM, |lat|-LIM, |total|-LIM, 0) per sample."""
        st = _eval_vehicle_states(theta, ctx, gen)
        j_long, j_lat = st[:, 6], st[:, 7]
        jm = jnp.sqrt(j_long ** 2 + j_lat ** 2)
        return jnp.maximum(
            jnp.maximum(0., jnp.abs(j_long) - JERK_MAX),
            jnp.maximum(
                jnp.maximum(0., jnp.abs(j_lat) - JERK_MAX),
                jnp.maximum(0., jm            - JERK_MAX),
            ),
        )

    return [
        # P1 (最内层): 避障 — baseline=2.0, 安全底线, max 感应单点穿透
        Deterministic(obs_g,   mode='hard', priority=1, aggregate='max',
                      transform='hard'),
        # P2-P5: comfort — mode='soft' → baseline=0
        # 无违规时 Φ=0, σ层透明, obj信号完全恢复
        Deterministic(lane_g,  mode='soft', priority=2, aggregate='q95',
                      transform='soft'),
        Deterministic(speed_g, mode='soft', priority=3, aggregate='max',
                      transform='soft'),
        Deterministic(acc_g,   mode='soft', priority=4, aggregate='max',
                      transform='soft'),
        Deterministic(jerk_g,  mode='soft', priority=5, aggregate='max',
                      transform='soft'),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Metrics — reusable g‑value computation (matches constraint formulas)
# ═══════════════════════════════════════════════════════════════════════

def compute_g_values(st, d, x_cart, y_cart, obs_pos, obs_rad,
                     lane_hw: float, obs_safe_dist: float):
    """Compute per‑constraint g‑values for reporting.

    Args:
        st:      [T, 9] vehicle states
        d:       [T] Frenet lateral offset
        x_cart, y_cart: [T] Cartesian positions
        obs_pos: [N, 2], obs_rad: [N]
        lane_hw, obs_safe_dist: scenario parameters

    Returns:
        dict with keys lane, obs, jerk, acc, speed (obs=max, rest=q90).
    """
    v = st[:, 2]
    a_long, a_lat = st[:, 4], st[:, 5]
    j_long, j_lat = st[:, 6], st[:, 7]
    am = jnp.sqrt(a_long ** 2 + a_lat ** 2)
    jm = jnp.sqrt(j_long ** 2 + j_lat ** 2)

    g_lane = jnp.quantile(jnp.maximum(0., jnp.abs(d) - lane_hw), 0.9)

    rho = obs_safe_dist
    d_rss = v * rho + v ** 2 / (2.0 * 8.0)
    dx = x_cart[:, None] - obs_pos[None, :, 0]
    dy = y_cart[:, None] - obs_pos[None, :, 1]
    r  = obs_rad[None, :]
    pen_x = jnp.maximum(0., d_rss[:, None] + r - jnp.abs(dx))
    pen_y = jnp.maximum(0., r - jnp.abs(dy))
    g_obs = float(jnp.max(jnp.maximum(pen_x, pen_y)))

    g_jerk = jnp.quantile(
        jnp.maximum(
            jnp.maximum(0., jnp.abs(j_long) - JERK_MAX),
            jnp.maximum(jnp.maximum(0., jnp.abs(j_lat) - JERK_MAX),
                        jnp.maximum(0., jm - JERK_MAX)),
        ), 0.9)

    g_acc = jnp.quantile(
        jnp.maximum(
            jnp.maximum(0., jnp.abs(a_long) - ACC_MAX),
            jnp.maximum(jnp.maximum(0., jnp.abs(a_lat) - ACC_MAX),
                        jnp.maximum(0., am - ACC_MAX)),
        ), 0.9)

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

    dist = jnp.sqrt((x_cart[:, None] - obs_pos[None, :, 0]) ** 2 +
                    (y_cart[:, None] - obs_pos[None, :, 1]) ** 2) - obs_rad[None, :]

    return {
        'min_obs': float(jnp.min(dist)),
        'max_a_long': float(jnp.max(jnp.abs(a_long))),
        'max_a_lat':  float(jnp.max(jnp.abs(a_lat))),
        'max_jerk':   float(jnp.max(jm)),
        'v':          float(st[0, 2]),
    }
