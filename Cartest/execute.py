"""Execution: bridge from plan to vehicle model.

1. Extract commanded acceleration from the plan at t=dt.
2. Pass to vehicle model for forward simulation.
3. Return a FrenetState with the model's full next state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrenetState:
    """Vehicle state — Frenet position + velocity + acceleration + yaw."""
    s:      float
    s_dot:  float
    s_ddot: float
    d:      float
    d_dot:  float
    d_ddot: float
    psi:    float = 0.0

    def to_ctx(self):
        """Convert to ctx dict entries for cost/constraint evaluation."""
        return {
            's0': self.s, 's_dot0': self.s_dot, 's_ddot0': self.s_ddot,
            'd0': self.d, 'd_dot0': self.d_dot, 'd_ddot0': self.d_ddot,
        }


def execute_step(gen, s, d, s_dot, d_dot, s_ddot, d_ddot,
                 vehicle_model, psi0: float = 0.0) -> FrenetState:
    """Plan → vehicle model → next FrenetState."""
    s_ddot_cmd = float(s_ddot[1])
    d_ddot_cmd = float(d_ddot[1])

    s_new, d_new, s_dot_new, d_dot_new, s_ddot_new, d_ddot_new, psi_new = \
        vehicle_model.step(
            float(s[0]), float(d[0]),
            float(s_dot[0]), float(d_dot[0]),
            s_ddot_cmd, d_ddot_cmd, psi0,
        )

    return FrenetState(
        s=s_new, s_dot=s_dot_new, s_ddot=float(s_ddot_new),
        d=d_new, d_dot=d_dot_new, d_ddot=float(d_ddot_new),
        psi=float(psi_new),
    )
