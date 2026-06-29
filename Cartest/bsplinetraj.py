"""Quintic B-spline trajectory generator (degree=5, C⁴ continuous).

All kinematics from precomputed basis matrices::

    pos  = B   @ ctrl    [T, 2]
    vel  = dB  @ ctrl    [T, 2]
    acc  = d2B @ ctrl    [T, 2]
    jerk = d3B @ ctrl    [T, 2]

Clamping (12 control points)::

    P0 = current position             (C0)
    P1 = P0 + dt_knot/5 · v0          (C1: velocity)
    P2 = 3·P1 − 2·P0 + dt_knot²/10·a0 (C2: acceleration)
    P3..P11 = free (9 pts × 2 = 18 optimization variables)

Nominal free points use Greville abscissae × current speed to produce
an exact constant-speed trajectory with zero acceleration and jerk.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from pathlib import Path


class BSplineTrajectoryGenerator:
    """Clamped quintic B-spline trajectory generator."""

    def __init__(self, basis_path: Path | str):
        data = np.load(str(basis_path))

        self.B    = jnp.array(data["B"])
        self.dB   = jnp.array(data["dB"])
        self.d2B  = jnp.array(data["d2B"])
        self.d3B  = jnp.array(data["d3B"])
        self.d4B  = jnp.array(data["d4B"])
        self.greville = jnp.array(data["greville"])  # [n_ctrl]

        self.T = int(self.B.shape[0])
        self.n_ctrl = int(self.B.shape[1])
        self.dt = float(data["dt"])
        self.total_time = float(data["total_time"])
        self.degree = int(data["degree"])
        self.dt_knot = float(data["dt_knot"])

        self.n_free = self.n_ctrl - 3  # P0,P1,P2 clamped

    # ═══════════════════════════════════════════════════════════════════
    # Clamped endpoints
    # ═══════════════════════════════════════════════════════════════════

    def _clamped_points(self, x0, v0, a0):
        """P0, P1, P2 from current vehicle state."""
        p0 = x0
        p1 = p0 + (self.dt_knot / self.degree) * v0
        p2 = 3.0 * p1 - 2.0 * p0 + (self.dt_knot ** 2 / 10.0) * a0
        return p0, p1, p2

    # ═══════════════════════════════════════════════════════════════════
    # Nominal free control points
    # ═══════════════════════════════════════════════════════════════════

    def nominal_free_points(self, speed: float, road_y: float = 0.0):
        """Free control points (P3..P11) for constant-speed straight line.

        Uses Greville abscissae: ``x_i = speed * greville[i]`` gives exact
        constant velocity with zero acceleration and zero jerk.
        """
        x = speed * self.greville[3:]  # P3..P11
        y = jnp.full((self.n_free,), road_y)
        return jnp.stack([x, y], axis=1)

    # ═══════════════════════════════════════════════════════════════════
    # Trajectory evaluation
    # ═══════════════════════════════════════════════════════════════════

    def evaluate(self, ctrl_free, x0, v0, a0):
        """Evaluate trajectory from free control points + clamped state.

        Returns pos, vel, acc, jerk — all [T, 2].
        """
        p0, p1, p2 = self._clamped_points(x0, v0, a0)
        full = jnp.concatenate([p0[None, :], p1[None, :], p2[None, :], ctrl_free], axis=0)
        return (
            jnp.dot(self.B,   full),
            jnp.dot(self.dB,  full),
            jnp.dot(self.d2B, full),
            jnp.dot(self.d3B, full),
        )

    # ═══════════════════════════════════════════════════════════════════
    # Vehicle states [T, 8]: x, y, v, psi, a_long, a_lat, jerk_long, steer
    # ═══════════════════════════════════════════════════════════════════

    def to_vehicle_states(self, pos, vel, acc, jerk, wheel_base=2.8):
        v = jnp.linalg.norm(vel, axis=-1)
        vs = v + 1e-6

        psi = jnp.arctan2(vel[..., 1], vel[..., 0])

        a_long = (acc[..., 0] * vel[..., 0] + acc[..., 1] * vel[..., 1]) / vs
        a_lat  = (vel[..., 0] * acc[..., 1] - vel[..., 1] * acc[..., 0]) / vs

        curvature = a_lat / (vs ** 2)
        steer = jnp.arctan(curvature * wheel_base)

        jerk_long = (jerk[..., 0] * vel[..., 0] + jerk[..., 1] * vel[..., 1]) / vs

        return jnp.stack([
            pos[..., 0], pos[..., 1], v, psi, a_long, a_lat, jerk_long, steer,
        ], axis=-1)
