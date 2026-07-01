"""Execution: bridge from plan to vehicle model.

1. Extract commanded acceleration from the plan at t=dt.
2. Pass to vehicle model for forward simulation.
3. Return a FrenetState with the model's full next state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrenetState:
    """Frenet vehicle state — position, velocity, acceleration."""
    s:      float
    s_dot:  float
    s_ddot: float
    d:      float
    d_dot:  float
    d_ddot: float

    def to_ctx(self):
        """Convert to ctx dict entries for cost/constraint evaluation."""
        return {
            's0': self.s, 's_dot0': self.s_dot, 's_ddot0': self.s_ddot,
            'd0': self.d, 'd_dot0': self.d_dot, 'd_ddot0': self.d_ddot,
        }


def execute_step(gen, s, d, s_dot, d_dot, s_ddot, d_ddot,
                 vehicle_model) -> FrenetState:
    """Plan → vehicle model → next FrenetState.

    Args:
        gen:   FrenetBSplineTrajectory
        s..d_ddot: [T] evaluated trajectory
        vehicle_model: object with step(s0,d0,s_dot0,d_dot0,cmd_s,cmd_d)

    Returns:
        FrenetState with the vehicle model's actual (clipped) acceleration.
    """
    # 1. Plan's intended acceleration at t=dt
    s_ddot_cmd = float(s_ddot[1])
    d_ddot_cmd = float(d_ddot[1])

    # 2. Forward simulation through vehicle model
    s_new, d_new, s_dot_new, d_dot_new, ax, ay = vehicle_model.step(
        float(s[0]), float(d[0]),
        float(s_dot[0]), float(d_dot[0]),
        s_ddot_cmd, d_ddot_cmd,
    )

    # 3. Return model's actual state
    return FrenetState(
        s=s_new, s_dot=s_dot_new, s_ddot=float(ax),
        d=d_new, d_dot=d_dot_new, d_ddot=float(ay),
    )
