"""Vehicle models for simulation — decoupled from planning via step() interface.

Receives Frenet acceleration command (s̈_cmd, d̈_cmd) from the plan,
converts to vehicle controls (a_long, δ), simulates realistic dynamics,
returns the full next FrenetState.

Interface:
  step(s0, d0, s_dot0, d_dot0, s_ddot_cmd, d_ddot_cmd, psi0)
      → (s, d, s_dot, d_dot, s_ddot, d_ddot, psi)
"""

from __future__ import annotations

import jax.numpy as jnp


class BicycleModel:
    """Kinematic bicycle with friction circle and steering lag.

    Plan outputs Frenet commands.  Model converts to vehicle controls,
    simulates yaw + steering dynamics, returns actual Frenet state.

    μ = 0.85 → a_max ≈ 8.3 m/s²,  L = 2.8 m (wheelbase).
    """

    def __init__(self, mu: float = 0.85, wheelbase: float = 2.8,
                 steer_max: float = 0.6, dt: float = 0.1):
        self.a_max = mu * 9.81
        self.L = wheelbase
        self.steer_max = steer_max   # rad (~35°)
        self.dt = dt

    def step(self, s0, d0, s_dot0, d_dot0,
             s_ddot_cmd, d_ddot_cmd, psi0: float = 0.0):
        """One time step of the bicycle model."""
        v0 = jnp.sqrt(s_dot0 ** 2 + d_dot0 ** 2)

        # ── 1. Frenet cmd → vehicle-frame acc ──────────────────────
        # For straight reference (θ_r=0): Δψ = arctan2(d_dot0, s_dot0)
        dpsi0 = jnp.arctan2(d_dot0, s_dot0)
        cos_d = jnp.cos(dpsi0)
        sin_d = jnp.sin(dpsi0)

        # Rotation: Frenet acc → vehicle longitudinal / lateral
        a_long_des = s_ddot_cmd * cos_d + d_ddot_cmd * sin_d
        a_lat_des  = -s_ddot_cmd * sin_d + d_ddot_cmd * cos_d

        # ── 2. Desired steering angle from curvature ───────────────
        # a_lat = v²·κ = v²·tan(δ)/L  →  δ = arctan(a_lat·L / v²)
        vs = jnp.maximum(v0, 1.0)
        steer_des = jnp.arctan(a_lat_des * self.L / vs ** 2)
        steer_des = jnp.clip(steer_des, -self.steer_max, self.steer_max)

        # ── 3. Friction circle: clip combined acceleration ────────
        a_lat_kin = vs ** 2 * jnp.tan(steer_des) / self.L
        a_cmb = jnp.sqrt(a_long_des ** 2 + a_lat_kin ** 2)
        scale = jnp.minimum(1.0, self.a_max / (a_cmb + 1e-6))
        a_long = a_long_des * scale
        a_lat  = a_lat_kin * scale
        # Recompute actual steer from clipped lateral acc
        steer_act = jnp.arctan(a_lat * self.L / vs ** 2)

        # ── 4. Yaw dynamics ───────────────────────────────────────
        psi_dot = vs * jnp.tan(steer_act) / self.L
        psi_new = psi0 + psi_dot * self.dt

        # ── 5. Euler integration in vehicle frame ─────────────────
        v_new     = vs + a_long * self.dt
        v_new     = jnp.maximum(v_new, 1.0)   # no reverse

        # Position integration in Frenet (straight ref: θ_r=0)
        # Vehicle velocity in Frenet frame
        s_dot_new = v_new * jnp.cos(psi_new)      # ψ = 0 → along x
        d_dot_new = v_new * jnp.sin(psi_new)      # lateral component
        s_new     = s0     + 0.5 * (s_dot0 + s_dot_new) * self.dt
        d_new     = d0     + 0.5 * (d_dot0 + d_dot_new) * self.dt

        # Acceleration propagated to next step
        s_ddot_new = a_long * jnp.cos(psi_new)
        d_ddot_new = a_long * jnp.sin(psi_new)

        return s_new, d_new, s_dot_new, d_dot_new, s_ddot_new, d_ddot_new, psi_new
