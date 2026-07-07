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

from Cartest.core.reference_path import ReferencePath


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

        self.n_free = self.n_ctrl - 2  # P0,P1 clamped (C0/C1), P2..P11 free
        self.ref_path = ref_path

    # ═══════════════════════════════════════════════════════════════════
    # Clamped endpoints — C0, C1 only (C2 left to optimizer)
    # ═══════════════════════════════════════════════════════════════════

    def _clamped_2pts(self, x0, v0):
        """P0,P1 from (position, velocity).  Acceleration free."""
        p0 = x0
        p1 = p0 + (self.dt_knot / self.degree) * v0
        return p0, p1

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
        p0_s, p1_s = self._clamped_2pts(s0, s_dot0)
        p0_d, p1_d = self._clamped_2pts(d0, d_dot0)

        ctrl_s = jnp.concatenate([
            jnp.array([p0_s]), jnp.array([p1_s]), ctrl_s_free
        ], axis=0)
        ctrl_d = jnp.concatenate([
            jnp.array([p0_d]), jnp.array([p1_d]), ctrl_d_free
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

    # ═══════════════════════════════════════════════════════════════════
    # Inverse: Vehicle states → Frenet states
    # ═══════════════════════════════════════════════════════════════════

    def from_vehicle_states(self, x, y, v, psi, a_long, a_lat,
                            j_long=None, j_lat=None, wheel_base=2.8):
        """Vehicle states → Frenet states (inverse of to_vehicle_states).

        Inverts the curvature-coupled transformation layer by layer:
          1. Cartesian (x,y) → Frenet (s,d) via ref_path
          2. (v, ψ) → (s_dot, d_dot) through Δψ decomp + Jacobian
          3. (a_long, a_lat) → (s_ddot, d_ddot) unrotate + remove centrifugal/Coriolis
          4. (j_long, j_lat) → (s_dddot, d_dddot) unrotate (optional)

        Args:
            x, y:        Cartesian position
            v:           speed [m/s]
            psi:         heading [rad]
            a_long, a_lat: longitudinal / lateral acceleration [m/s²]
            j_long, j_lat: longitudinal / lateral jerk [m/s³] (optional)
            wheel_base:  [m] (unused; accepted for symmetry with to_vehicle_states)

        Returns:
            (s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot)
        """
        # ── Layer 1: position ──────────────────────────────────────
        s, d = self.ref_path.cartesian_to_frenet(x, y)
        _, _, theta_r, kappa_r = self.ref_path.evaluate(s)

        # ── Layer 2: velocity ──────────────────────────────────────
        dpsi = psi - theta_r
        cos_dpsi = jnp.cos(dpsi)
        sin_dpsi = jnp.sin(dpsi)
        vt = v * cos_dpsi
        vn = v * sin_dpsi
        jac = 1.0 - d * kappa_r + 1e-8            # numerical guard
        s_dot = vt / jac
        d_dot = vn

        # ── Layer 3: acceleration ──────────────────────────────────
        # Rotate vehicle-frame acc back to Frenet (t, n) basis
        a_t = a_long * cos_dpsi - a_lat * sin_dpsi   # R(-Δψ)
        a_n = a_long * sin_dpsi + a_lat * cos_dpsi
        # Remove centrifugal and Coriolis coupling
        d_ddot = a_n - kappa_r * vt * s_dot          # strip centrifugal
        vt_dot = a_t + vn * kappa_r * s_dot          # strip Coriolis
        s_ddot = (vt_dot + kappa_r * s_dot * d_dot) / jac

        # ── Layer 4: jerk (optional) ───────────────────────────────
        if j_long is not None and j_lat is not None:
            s_dddot = j_long * cos_dpsi - j_lat * sin_dpsi
            d_dddot = j_long * sin_dpsi + j_lat * cos_dpsi
        else:
            s_dddot = jnp.zeros_like(s)
            d_dddot = jnp.zeros_like(s)

        return s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot


# ═══════════════════════════════════════════════════════════════════════
# Reference trajectory generator — maneuver → z_ref for Lyapunov tracking
# ═══════════════════════════════════════════════════════════════════════

def make_frenet_reference(gen, ctx, maneuver):
    """Generate Frenet reference z_ref for Lyapunov tracking cost.

    All modes (internal + external) build a vehicle‑level reference y_ref
    first, then convert to Frenet via from_vehicle_states().  This ensures
    the reference respects the path geometry — velocity decomposition
    through κ_r and d_dot, correct ψ_ref, etc.

    Args:
        gen:      FrenetBSplineTrajectory
        ctx:      context dict with s0, s_dot0, s_ddot0, d0, d_dot0, d_ddot0
        maneuver: dict with 'type' and type‑specific parameters.

    Maneuver types
    --------------
    lane_change:
        {'type': 'lane_change', 'd_end': 3.5, 't_start': 0.5,
         't_duration': 3.0, 'v_desired': 20.0}

    cruise:
        {'type': 'cruise', 'v_desired': 25.0}

    external:
        {'type': 'external', 'vehicle_states': y_ref}   # [T, 9]

    Returns
    -------
    dict with keys:
        s_ref, s_dot_ref, s_ddot_ref, s_dddot_ref,
        d_ref, d_dot_ref, d_ddot_ref, d_dddot_ref
    Each value is a [T] JAX array.
    """
    if maneuver['type'] == 'external':
        y_ref = maneuver['vehicle_states']
    else:
        y_ref = _build_vehicle_reference(gen, ctx, maneuver)
    return _frenet_from_vehicle(gen, y_ref)


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _frenet_from_vehicle(gen, y_ref):
    """Convert [T, 9] vehicle states → z_ref dict via from_vehicle_states."""
    s_ref, d_ref, s_dot_ref, d_dot_ref, s_ddot_ref, d_ddot_ref, \
        s_dddot_ref, d_dddot_ref = gen.from_vehicle_states(
            y_ref[:, 0], y_ref[:, 1],  # x, y
            y_ref[:, 2], y_ref[:, 3],  # v, ψ
            y_ref[:, 4], y_ref[:, 5],  # a_long, a_lat
            y_ref[:, 6], y_ref[:, 7],  # j_long, j_lat
        )
    return {
        's_ref': s_ref, 's_dot_ref': s_dot_ref,
        's_ddot_ref': s_ddot_ref, 's_dddot_ref': s_dddot_ref,
        'd_ref': d_ref, 'd_dot_ref': d_dot_ref,
        'd_ddot_ref': d_ddot_ref, 'd_dddot_ref': d_dddot_ref,
    }


def _build_vehicle_reference(gen, ctx, maneuver):
    """Build [T, 9] vehicle‑state reference from maneuver spec.

    Constructs a kinematically consistent vehicle trajectory:
      - Lateral:  quintic polynomial (C² smooth) for lane_change,
                  constant for cruise.
      - Speed:    exponential transition from current to target.
      - Heading:  derived from velocity decomposition (respects κ_r).
      - Position: integrated forward through the reference path.
      - Acc / jerk:  numerical differentiation of v and ψ.
    """
    T = gen.T
    dt = gen.dt
    t_arr = jnp.arange(T) * dt
    s0 = ctx['s0']
    v0 = ctx['s_dot0']
    d0 = ctx['d0']
    mtype = maneuver['type']

    # ── Speed profile ────────────────────────────────────────────
    v_tgt = maneuver.get('v_desired', v0)
    dv = v0 - v_tgt
    lam = 1.0
    exp_term = jnp.exp(-lam * t_arr)
    v_ref = v_tgt + dv * exp_term                               # [T]
    a_long_ref = -dv * lam * exp_term                           # d/dt analytical

    # ── Lateral profile ──────────────────────────────────────────
    if mtype == 'cruise':
        d_ref = jnp.full(T, d0)
        d_dot_ref = jnp.zeros(T)
        d_ddot_ref = jnp.zeros(T)
        d_dddot_ref = jnp.zeros(T)
    elif mtype == 'lane_change':
        d_end = maneuver['d_end']
        t_start = maneuver.get('t_start', 0.0)
        t_dur = maneuver['t_duration']
        dd = d_end - d0

        tau = jnp.clip((t_arr - t_start) / t_dur, 0.0, 1.0)
        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau3 * tau
        tau5 = tau4 * tau
        poly = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
        d_ref = d0 + dd * poly
        d_dot_ref = dd * (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / t_dur
        d_ddot_ref = dd * (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (t_dur * t_dur)
        d_dddot_ref = dd * (60.0 - 360.0 * tau + 360.0 * tau2) / (t_dur * t_dur * t_dur)
    else:
        raise ValueError(f"Unknown maneuver type: {mtype}")

    # ── s_dot from kinematics: v² = (1−d·κ)²·s_dot² + d_dot² ──
    # Evaluate reference-path curvature at s0 (approximate — valid for
    # constant κ or small s deviations; the from_vehicle_states call
    # at the end corrects any residual mismatch).
    _, _, _, kappa_r = gen.ref_path.evaluate(jnp.array([s0]))
    kap = kappa_r[0]
    jac = 1.0 - d_ref * kap
    # Guard against imaginary s_dot when v_ref < |d_dot_ref|
    radicand = jnp.maximum(0.0, v_ref ** 2 - d_dot_ref ** 2)
    s_dot_ref = jnp.sqrt(radicand) / (jac + 1e-8)

    # ── s_ref by cumulative integration ──────────────────────────
    s_ref = jnp.zeros(T)
    s_ref = s_ref.at[0].set(s0)
    s_ref = s_ref.at[1:].set(
        s0 + jnp.cumsum(s_dot_ref[:-1]) * dt
    )

    # ── Cartesian position via reference path ────────────────────
    x_ref, y_ref = gen.ref_path.frenet_to_cartesian(s_ref, d_ref)

    # ── Heading: ψ = θ_r + arctan2(d_dot, (1−d·κ)·s_dot) ───────
    _, _, theta_r, _ = gen.ref_path.evaluate(s_ref)
    vt_ref = jac * s_dot_ref
    dpsi_ref = jnp.arctan2(d_dot_ref, vt_ref)
    psi_ref = theta_r + dpsi_ref

    # ── Acceleration / jerk — analytical (no numerical diff) ──────
    # Straight road (κ_r=0):
    #   s_ddot  = (v·a_long − d_dot·d_ddot) / s_dot
    #   a_lat   = (s_dot·d_ddot − d_dot·s_ddot) / v
    #
    # Curved road (constant κ_r):
    #   r       = √(v² − d_dot²),   dr/dt = (v·a_long − d_dot·d_ddot) / r
    #   s_ddot  = (dr/dt·jac + κ_r·d_dot·r) / jac²
    #   vt_ref  = jac·s_dot = r,   vt_dot = dr/dt
    #   dψ/dt   = κ_r·s_dot + (vt_ref·d_ddot − d_dot·vt_dot) / v²
    #   a_lat   = v·dψ/dt
    r_ref = jnp.sqrt(radicand)
    dr_dt = (v_ref * a_long_ref - d_dot_ref * d_ddot_ref) / (r_ref + 1e-8)

    # s_ddot — accounts for κ_r when present
    s_ddot_ref = (dr_dt * jac + kap * d_dot_ref * r_ref) / (jac ** 2 + 1e-8)

    # a_lat = v · dψ/dt
    vt_dot = dr_dt
    dpsi_dt = kap * s_dot_ref + (r_ref * d_ddot_ref
                                  - d_dot_ref * vt_dot) / (v_ref ** 2 + 1e-8)
    a_lat_ref = v_ref * dpsi_dt

    # Jerk — analytical from the exponential speed profile + quintic lateral
    j_long_ref = dv * lam ** 2 * exp_term  # d²v_ref/dt²

    # s_dddot = d(s_ddot)/dt
    # M = v·a_long − d_dot·d_ddot,  dM/dt = a_long² + v·j_long − d_ddot² − d_dot·d_dddot
    dM_dt = (a_long_ref ** 2 + v_ref * j_long_ref
             - d_ddot_ref ** 2 - d_dot_ref * d_dddot_ref)
    s_dddot_ref = (dM_dt * r_ref - (v_ref * a_long_ref - d_dot_ref * d_ddot_ref) * dr_dt
                   ) / (r_ref ** 2 + 1e-8)
    # Correct for κ_r (approximate — higher-order κ_r terms omitted)
    s_dddot_ref = s_dddot_ref / (jac + 1e-8)

    # j_lat = d(a_lat)/dt
    # a_lat = (s_dot·d_ddot − d_dot·s_ddot) / v  (exact for κ_r=0; approx for κ_r≠0)
    num = s_dot_ref * d_ddot_ref - d_dot_ref * s_ddot_ref
    dnum_dt = (s_ddot_ref * d_ddot_ref + s_dot_ref * d_dddot_ref
               - d_ddot_ref * s_ddot_ref - d_dot_ref * s_dddot_ref)
    j_lat_ref = (dnum_dt * v_ref - num * a_long_ref) / (v_ref ** 2 + 1e-8)

    return jnp.stack([
        x_ref, y_ref, v_ref, psi_ref,
        a_long_ref, a_lat_ref, j_long_ref, j_lat_ref,
        jnp.zeros(T),  # steer (placeholder)
    ], axis=-1)
