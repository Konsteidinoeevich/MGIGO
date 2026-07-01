"""Frenet-frame quintic B-spline trajectory generator.

Same basis matrices as the Cartesian version (spline.py), but control
points live in (s, d) Frenet space instead of (x, y).

    s-channel:  s(t) = B·ctrl_s   ḃ = dB·ctrl_s   s̈ = d2B·ctrl_s  s⃛ = d3B·ctrl_s
    d-channel:  d(t) = B·ctrl_d   ḋ = dB·ctrl_d   d̈ = d2B·ctrl_d  d⃛ = d3B·ctrl_d

All kinematics are linear in control points — no arctan2 projection,
no curvature-division chain.  The reference path absorbs the curvature.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from pathlib import Path

from Cartest.reference_path import ReferencePath


class FrenetBSplineTrajectory:
    """Clamped quintic B-spline in Frenet (s, d) coordinates."""

    def __init__(self, basis_path: Path | str, ref_path: ReferencePath):
        data = np.load(str(basis_path))

        self.B = jnp.array(data["B"])
        self.dB = jnp.array(data["dB"])
        self.d2B = jnp.array(data["d2B"])
        self.d3B = jnp.array(data["d3B"])
        self.d4B = jnp.array(data["d4B"])
        self.greville = jnp.array(data["greville"])

        self.T = int(self.B.shape[0])
        self.n_ctrl = int(self.B.shape[1])
        self.dt = float(data["dt"])
        self.total_time = float(data["total_time"])
        self.degree = int(data["degree"])
        self.dt_knot = float(data["dt_knot"])

        self.n_free = self.n_ctrl - 3  # P0,P1,P2 clamped → 9 free
        self.ref_path = ref_path

    # ═══════════════════════════════════════════════════════════════════
    # Clamped endpoints — same formula as Cartesian, per channel
    # ═══════════════════════════════════════════════════════════════════

    def _clamped_3pts(self, x0, v0, a0):
        """P0,P1,P2 from (position, velocity, acceleration) for one channel."""
        p0 = x0
        p1 = p0 + (self.dt_knot / self.degree) * v0
        p2 = 3.0 * p1 - 2.0 * p0 + (self.dt_knot ** 2 / 10.0) * a0
        return p0, p1, p2

    # ═══════════════════════════════════════════════════════════════════
    # Trajectory evaluation
    # ═══════════════════════════════════════════════════════════════════

    def evaluate(self, ctrl_s_free, ctrl_d_free,
                 s0, s_dot0, s_ddot0,
                 d0, d_dot0, d_ddot0):
        """Evaluate Frenet trajectory from free control points + clamped state.

        Args:
            ctrl_s_free: [n_free]   free s-channel control points
            ctrl_d_free: [n_free]   free d-channel control points
            s0, s_dot0, s_ddot0:    initial s position, velocity, acceleration
            d0, d_dot0, d_ddot0:    initial d position, velocity, acceleration

        Returns:
            s, d:        [T]     position in Frenet
            s_dot, d_dot: [T]     velocity (Frenet time-derivative)
            s_ddot, d_ddot: [T]   acceleration
            s_dddot, d_dddot: [T] jerk
        """
        # Clamped endpoints for each channel
        p0_s, p1_s, p2_s = self._clamped_3pts(s0, s_dot0, s_ddot0)
        p0_d, p1_d, p2_d = self._clamped_3pts(d0, d_dot0, d_ddot0)

        ctrl_s = jnp.concatenate([
            jnp.array([p0_s]), jnp.array([p1_s]), jnp.array([p2_s]), ctrl_s_free
        ], axis=0)
        ctrl_d = jnp.concatenate([
            jnp.array([p0_d]), jnp.array([p1_d]), jnp.array([p2_d]), ctrl_d_free
        ], axis=0)

        return (
            jnp.dot(self.B,   ctrl_s),   # s
            jnp.dot(self.B,   ctrl_d),   # d
            jnp.dot(self.dB,  ctrl_s),   # ḃ
            jnp.dot(self.dB,  ctrl_d),   # ḋ
            jnp.dot(self.d2B, ctrl_s),   # s̈
            jnp.dot(self.d2B, ctrl_d),   # d̈
            jnp.dot(self.d3B, ctrl_s),   # s⃛
            jnp.dot(self.d3B, ctrl_d),   # d⃛
        )

    # ═══════════════════════════════════════════════════════════════════
    # One-shot plan evaluation
    # ═══════════════════════════════════════════════════════════════════

    def evaluate_plan(self, ctrl_s_free, ctrl_d_free, ctx):
        """Evaluate B-spline → Frenet + vehicle states + Cartesian.

        Returns (frenet, vehicle_states, (x, y)) where:
          frenet = (s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot)
          vehicle_states = [T, 9]
          (x, y) = Cartesian positions [T] each
        """
        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = self.evaluate(
            ctrl_s_free, ctrl_d_free,
            ctx["s0"], ctx["s_dot0"], ctx["s_ddot0"],
            ctx["d0"], ctx["d_dot0"], ctx["d_ddot0"],
        )
        st = self.to_vehicle_states(
            s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot)
        x, y = self.to_cartesian(s, d)
        return (s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot), st, (x, y)

    # ═══════════════════════════════════════════════════════════════════
    # Cartesian mapping (via reference path) — obstacle checking
    # ═══════════════════════════════════════════════════════════════════

    def to_cartesian(self, s, d):
        """Frenet (s,d) → Cartesian (x,y) through reference path."""
        return self.ref_path.frenet_to_cartesian(s, d)

    # ═══════════════════════════════════════════════════════════════════
    # Vehicle states [T, 9]:  x, y, v, ψ, a_long, a_lat, j_long, j_lat, steer
    # ═══════════════════════════════════════════════════════════════════

    def to_vehicle_states(self, s, d, s_dot, d_dot,
                          s_ddot, d_ddot, s_dddot, d_dddot,
                          wheel_base=2.8):
        """Frenet → vehicle-frame kinematics with curvature coupling.

        Correct transformation accounting for:
          - (1-d·κ_r) Jacobian in longitudinal speed
          - Centrifugal acceleration κ_r·v_t² in lateral direction
          - Coriolis term 2·κ_r·ḃ·ḋ in tangential acceleration
          - Frame rotation by Δψ for a_long / a_lat

        All correction terms use the *given* reference path κ_r(s) —
        not optimization variables — so they don't introduce instability.
        For straight roads (κ_r=0) the formulas reduce to the simplified form.
        """
        # Reference path geometry at s(t)
        _, _, θ_r, κ_r = self.ref_path.evaluate(s)
        # κ_r' = dκ_r/ds — 0 for lines and circular arcs; ignored for now

        # ── Velocity ────────────────────────────────────────────
        # v = (1-d·κ_r)·ḃ·t + ḋ·n   in (t, n) basis
        vt = (1.0 - d * κ_r) * s_dot        # tangential component
        vn = d_dot                            # normal component
        v2 = vt ** 2 + vn ** 2
        v = jnp.sqrt(v2)
        vs = v + 1e-6

        # Vehicle heading: ψ = θ_r + Δψ,  Δψ = arctan2(vn, vt)
        dpsi = jnp.arctan2(vn, vt)
        psi = θ_r + dpsi
        cos_dpsi = vt / vs
        sin_dpsi = vn / vs

        # ── Acceleration (reference-path tangential / normal) ──
        # a_t = dv_t/dt - v_n·κ_r·ḃ
        # a_n = d̈ + κ_r·v_t·ḃ    (centrifugal)
        vt_dot = (1.0 - d * κ_r) * s_ddot - κ_r * s_dot * d_dot   # κ_r' term omitted
        a_t = vt_dot - vn * κ_r * s_dot
        a_n = d_ddot + κ_r * vt * s_dot

        # ── Rotate to vehicle frame ────────────────────────────
        # [a_long]   [ cosΔψ  sinΔψ] [a_t]
        # [a_lat ] = [-sinΔψ  cosΔψ] [a_n]
        a_long = a_t * cos_dpsi + a_n * sin_dpsi
        a_lat  = -a_t * sin_dpsi + a_n * cos_dpsi

        # ── Jerk: rotate Frenet jerk to vehicle frame ──────────
        # Centrifugal jerk correction (2·κ_r·v·a_long) omitted for now.
        j_long = s_dddot * cos_dpsi + d_dddot * sin_dpsi
        j_lat  = -s_dddot * sin_dpsi + d_dddot * cos_dpsi

        # ── Curvature & steer ──────────────────────────────────
        # dψ/dt = κ_r·ḃ + d(Δψ)/dt
        # d(Δψ)/dt = (vt·d̈ - vn·vt_dot) / v²
        ddpsi_dt = (vt * d_ddot - vn * vt_dot) / v2
        dpsi_dt = κ_r * s_dot + ddpsi_dt
        curvature = dpsi_dt / vs
        steer = jnp.arctan(curvature * wheel_base)

        # Cartesian position (for obstacle checking / visualization)
        x, y = self.to_cartesian(s, d)

        return jnp.stack([
            x, y, v, psi, a_long, a_lat, j_long, j_lat, steer,
        ], axis=-1)
