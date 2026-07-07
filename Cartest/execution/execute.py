"""Execution: bridge from plan to vehicle model.

Two modes:
  - execute_point_mass:   Frenet Euler integration (legacy, κ_r=0 only)
  - execute_perfect_tracking:  use plan's predicted next state directly
    (assumes low‑level controller can perfectly track the plan).

For evaluating open‑loop plan quality (tracking / overshoot / oscillation /
constraint satisfaction), use execute_perfect_tracking.
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
        return {
            's0': self.s, 's_dot0': self.s_dot, 's_ddot0': self.s_ddot,
            'd0': self.d, 'd_dot0': self.d_dot, 'd_ddot0': self.d_ddot,
        }


def execute_perfect_tracking(s, d, s_dot, d_dot, s_ddot, d_ddot, psi_next):
    """Use plan's predicted next state directly (perfect tracking assumption).

    The plan already produces a consistent trajectory through the B-spline
    + to_vehicle_states pipeline.  This function reads the t=1 state from
    that trajectory — equivalent to assuming the low‑level controller can
    exactly track the plan.

    Args:
        s, d, s_dot, d_dot, s_ddot, d_ddot: Frenet trajectory arrays [T]
        psi_next: vehicle heading at t=1 from to_vehicle_states

    Returns:
        FrenetState at t=1
    """
    return FrenetState(
        s=float(s[1]), s_dot=float(s_dot[1]), s_ddot=float(s_ddot[1]),
        d=float(d[1]), d_dot=float(d_dot[1]), d_ddot=float(d_ddot[1]),
        psi=float(psi_next),
    )


def execute_point_mass(gen, s, d, s_dot, d_dot, s_ddot, d_ddot,
                       vehicle_model, psi0: float = 0.0) -> FrenetState:
    """Legacy: Frenet Euler integration via PointMassModel (no curvature coupling).

    Suitable for straight roads (κ_r=0) where Frenet accelerations equal
    vehicle‑frame accelerations.
    """
    s_ddot_cmd = float(s_ddot[1])
    d_ddot_cmd = float(d_ddot[1])

    s_new, d_new, s_dot_new, d_dot_new, ax, ay, _ = vehicle_model.step(
        float(s[0]), float(d[0]),
        float(s_dot[0]), float(d_dot[0]),
        s_ddot_cmd, d_ddot_cmd,
    )

    return FrenetState(
        s=s_new, s_dot=s_dot_new, s_ddot=float(ax),
        d=d_new, d_dot=d_dot_new, d_ddot=float(ay),
        psi=float(psi0),
    )


# Backward-compatible alias
execute_step = execute_point_mass
