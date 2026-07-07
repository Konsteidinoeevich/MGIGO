"""Vehicle models for simulation — decoupled from planning via step() interface.

Receives acceleration command (s̈_cmd, d̈_cmd) from the plan,
integrates directly with friction-circle limits.
"""

from __future__ import annotations

import jax.numpy as jnp


class PointMassModel:
    """Point-mass in Frenet — direct integration of commanded acceleration.

    Friction circle clips combined magnitude.  No yaw / steering delay.
    Assumes the plan is valid and the controller can track it.
    """

    def __init__(self, mu: float = 0.85, dt: float = 0.1):
        self.a_max = mu * 9.81
        self.dt = dt

    def step(self, s0, d0, s_dot0, d_dot0,
             s_ddot_cmd, d_ddot_cmd,
             psi0: float = 0.0):
        """Euler integration with friction-circle clipping.  Returns 7 values."""
        a_cmd = jnp.sqrt(s_ddot_cmd ** 2 + d_ddot_cmd ** 2)
        scale = jnp.minimum(1.0, self.a_max / (a_cmd + 1e-6))
        ax = s_ddot_cmd * scale
        ay = d_ddot_cmd * scale

        s_new     = s0     + s_dot0 * self.dt
        s_dot_new = s_dot0 + ax      * self.dt
        d_new     = d0     + d_dot0 * self.dt
        d_dot_new = d_dot0 + ay      * self.dt

        return s_new, d_new, s_dot_new, d_dot_new, ax, ay, 0.0


# Alias for backward compatibility
BicycleModel = PointMassModel
