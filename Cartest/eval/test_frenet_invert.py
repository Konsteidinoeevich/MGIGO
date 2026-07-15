"""Unit tests for Frenet ↔ Vehicle state round-trip consistency.

Verifies:
  1. to_vehicle_states ∘ from_vehicle_states ≈ identity  (κ_r=0 straight road)
  2. from_vehicle_states ∘ to_vehicle_states ≈ identity
  3. Zero-speed degeneracy
  4. Perfect tracking execution consistency
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory, make_frenet_reference
from Cartest.core.reference_path import StraightReference, CircularReference as CircularRefPath, ReferencePath
from Cartest.execution.execute import execute_perfect_tracking, FrenetState

BASIS = Path(__file__).resolve().parents[1] / "basis"


# ═══════════════════════════════════════════════════════════════════════
# Circular reference path for κ_r ≠ 0 testing
# ═══════════════════════════════════════════════════════════════════════

class CircularReference(ReferencePath):
    """Circular arc reference path with constant curvature κ_r = 1/R.

    Centre at (0, R), radius R, starting at (0, 0) heading +x (east).
    Arc-length s runs counter-clockwise.

        x_r(s) = R·sin(s/R)
        y_r(s) = R·(1 − cos(s/R))
        θ_r(s)  = s/R
        κ_r(s)  = 1/R
    """

    def __init__(self, radius: float = 100.0):
        self.R = radius
        self._kappa = 1.0 / radius

    def evaluate(self, s):
        R = self.R
        theta = s / R
        x_r = R * jnp.sin(theta)
        y_r = R * (1.0 - jnp.cos(theta))
        kappa_r = jnp.full_like(s, self._kappa)
        return x_r, y_r, theta, kappa_r

    def frenet_to_cartesian(self, s, d):
        # x = (R-d)·sin(s/R),  y = R − (R-d)·cos(s/R)
        R = self.R
        theta = s / R
        r_eff = R - d
        x = r_eff * jnp.sin(theta)
        y = R - r_eff * jnp.cos(theta)
        return x, y

    def cartesian_to_frenet(self, x, y):
        # Inverse of the above — closed form.
        # Distance from centre (0, R):  r_eff = √(x² + (R−y)²)
        # d = R − r_eff,  s = R·atan2(x, R−y)
        R = self.R
        r_eff = jnp.sqrt(x ** 2 + (R - y) ** 2)
        d = R - r_eff
        s = R * jnp.arctan2(x, R - y)
        return s, d


def _make_gen():
    """Create a FrenetBSplineTrajectory with StraightReference."""
    ref_path = StraightReference()
    return FrenetBSplineTrajectory(BASIS / "bspline_basis.npz", ref_path)


def test_from_vehicle_states_round_trip():
    """from_vehicle_states(to_vehicle_states(z)) ≈ z on straight road.

    On a straight road (κ_r=0), to_vehicle_states simplifies to:
      x=s, y=d, v=s_dot, ψ=0 (or arctan2(d_dot, s_dot)),
      a_long=s_ddot, a_lat=d_ddot, j_long=s_dddot, j_lat=d_dddot
    The inverse should recover z exactly.
    """
    gen = _make_gen()
    T = gen.T

    # Random Frenet state at each timestep
    key = np.random.RandomState(42)
    s = jnp.array(key.uniform(0, 100, T))
    d = jnp.array(key.uniform(-3, 3, T))
    s_dot = jnp.array(key.uniform(2, 30, T))
    d_dot = jnp.array(key.uniform(-2, 2, T))
    s_ddot = jnp.array(key.uniform(-3, 3, T))
    d_ddot = jnp.array(key.uniform(-2, 2, T))
    s_dddot = jnp.array(key.uniform(-1, 1, T))
    d_dddot = jnp.array(key.uniform(-1, 1, T))

    # Forward: Frenet → vehicle states
    st = gen.to_vehicle_states(s, d, s_dot, d_dot,
                                s_ddot, d_ddot, s_dddot, d_dddot)
    x, y = st[:, 0], st[:, 1]
    v, psi = st[:, 2], st[:, 3]
    a_long, a_lat = st[:, 4], st[:, 5]
    j_long, j_lat = st[:, 6], st[:, 7]

    # Inverse: vehicle states → Frenet
    s2, d2, s_dot2, d_dot2, s_ddot2, d_ddot2, s_dddot2, d_dddot2 = \
        gen.from_vehicle_states(x, y, v, psi, a_long, a_lat, j_long, j_lat)

    # Check round-trip accuracy (should be near machine precision for straight road)
    assert_allclose(np.array(s), np.array(s2), atol=1e-6, rtol=1e-6)
    assert_allclose(np.array(d), np.array(d2), atol=1e-6, rtol=1e-6)
    assert_allclose(np.array(s_dot), np.array(s_dot2), atol=1e-6, rtol=1e-6)
    assert_allclose(np.array(d_dot), np.array(d_dot2), atol=1e-6, rtol=1e-6)
    assert_allclose(np.array(s_ddot), np.array(s_ddot2), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(d_ddot), np.array(d_ddot2), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(s_dddot), np.array(s_dddot2), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(d_dddot), np.array(d_dddot2), atol=1e-5, rtol=1e-5)


def test_forward_inverse_round_trip():
    """to_vehicle_states(from_vehicle_states(y)) ≈ y on straight road.

    Generate vehicle states, invert to Frenet, then forward again.
    """
    gen = _make_gen()
    T = gen.T

    key = np.random.RandomState(99)
    x = jnp.array(key.uniform(0, 100, T))
    y = jnp.array(key.uniform(-3, 3, T))
    v = jnp.array(key.uniform(2, 30, T))
    psi = jnp.array(key.uniform(-0.5, 0.5, T))
    a_long = jnp.array(key.uniform(-3, 3, T))
    a_lat = jnp.array(key.uniform(-2, 2, T))
    j_long = jnp.array(key.uniform(-1, 1, T))
    j_lat = jnp.array(key.uniform(-1, 1, T))

    # Inverse: vehicle → Frenet
    s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = \
        gen.from_vehicle_states(x, y, v, psi, a_long, a_lat, j_long, j_lat)

    # Forward: Frenet → vehicle
    st = gen.to_vehicle_states(s, d, s_dot, d_dot,
                                s_ddot, d_ddot, s_dddot, d_dddot)

    # On straight road (κ_r=0): x=s, y=d, v=s_dot, a_long=s_ddot, a_lat=d_ddot
    assert_allclose(np.array(x), np.array(st[:, 0]), atol=1e-6, rtol=1e-6)
    assert_allclose(np.array(y), np.array(st[:, 1]), atol=1e-6, rtol=1e-6)
    assert_allclose(np.array(v), np.array(st[:, 2]), atol=1e-6, rtol=1e-6)
    assert_allclose(np.array(a_long), np.array(st[:, 4]), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(a_lat), np.array(st[:, 5]), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(j_long), np.array(st[:, 6]), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(j_lat), np.array(st[:, 7]), atol=1e-5, rtol=1e-5)


def test_zero_speed_degenerate():
    """v=0: Δψ degenerates but s_dot, d_dot → 0, and inverse is well-defined."""
    gen = _make_gen()
    T = 10

    x = jnp.ones(T) * 5.0
    y = jnp.ones(T) * 2.0
    v = jnp.zeros(T)
    psi = jnp.zeros(T)
    a_long = jnp.zeros(T)
    a_lat = jnp.zeros(T)

    # Should not raise / produce NaN
    s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = \
        gen.from_vehicle_states(x, y, v, psi, a_long, a_lat)

    # At zero speed on straight road:
    #   s=x=5, d=y=2, s_dot=0, d_dot=0, s_ddot=0, d_ddot=0
    assert_allclose(np.array(s), np.array(x), atol=1e-6)
    assert_allclose(np.array(d), np.array(y), atol=1e-6)
    assert_allclose(np.array(s_dot), np.zeros(T), atol=1e-6)
    assert_allclose(np.array(d_dot), np.zeros(T), atol=1e-6)
    assert_allclose(np.array(s_ddot), np.zeros(T), atol=1e-6)
    assert_allclose(np.array(d_ddot), np.zeros(T), atol=1e-6)


def test_perfect_tracking_consistency():
    """execute_perfect_tracking returns state matching plan's t=1 prediction."""
    gen = _make_gen()

    # Build a simple Frenet trajectory
    T = gen.T
    s_arr = jnp.arange(T, dtype=jnp.float32)
    d_arr = jnp.ones(T) * 2.0
    s_dot_arr = jnp.ones(T) * 10.0
    d_dot_arr = jnp.zeros(T)
    s_ddot_arr = jnp.zeros(T)
    d_ddot_arr = jnp.zeros(T)
    psi_next = 0.0

    state = execute_perfect_tracking(
        s_arr, d_arr, s_dot_arr, d_dot_arr,
        s_ddot_arr, d_ddot_arr, psi_next,
    )

    assert state.s == float(s_arr[1])
    assert state.d == float(d_arr[1])
    assert state.s_dot == float(s_dot_arr[1])
    assert state.d_dot == float(d_dot_arr[1])
    assert state.s_ddot == float(s_ddot_arr[1])
    assert state.d_ddot == float(d_ddot_arr[1])
    assert state.psi == float(psi_next)


def test_no_jerk_input():
    """from_vehicle_states without jerk → s_dddot, d_dddot zeros."""
    gen = _make_gen()
    x = jnp.array([10.0])
    y = jnp.array([1.0])
    v = jnp.array([15.0])
    psi = jnp.array([0.1])
    a_long = jnp.array([1.0])
    a_lat = jnp.array([-0.5])

    _, _, _, _, _, _, s_dddot, d_dddot = \
        gen.from_vehicle_states(x, y, v, psi, a_long, a_lat)

    assert float(s_dddot[0]) == 0.0
    assert float(d_dddot[0]) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Curved road tests (κ_r ≠ 0)
# ═══════════════════════════════════════════════════════════════════════

def _make_circular_gen(radius=100.0):
    """Create a FrenetBSplineTrajectory with CircularReference (κ_r=1/R)."""
    ref_path = CircularReference(radius)
    return FrenetBSplineTrajectory(BASIS / "bspline_basis.npz", ref_path)


def test_round_trip_curved():
    """from_vehicle_states(to_vehicle_states(z)) ≈ z on circular road (κ_r=0.01).

    On a curved road the Jacobian (1-d·κ_r), centrifugal (κ_r·vt·s_dot),
    and Coriolis (vn·κ_r·s_dot) terms are all active.  The round-trip
    must still recover z exactly.
    """
    gen = _make_circular_gen(radius=100.0)   # κ_r = 0.01 m⁻¹
    T = gen.T

    key = np.random.RandomState(42)
    # Keep d small enough that 1-d·κ_r stays well away from 0
    s = jnp.array(key.uniform(0, 50, T))
    d = jnp.array(key.uniform(-2, 2, T))
    s_dot = jnp.array(key.uniform(5, 25, T))
    d_dot = jnp.array(key.uniform(-1, 1, T))
    s_ddot = jnp.array(key.uniform(-2, 2, T))
    d_ddot = jnp.array(key.uniform(-1, 1, T))
    s_dddot = jnp.array(key.uniform(-0.5, 0.5, T))
    d_dddot = jnp.array(key.uniform(-0.5, 0.5, T))

    # Forward: Frenet → vehicle (with curvature coupling)
    st = gen.to_vehicle_states(s, d, s_dot, d_dot,
                                s_ddot, d_ddot, s_dddot, d_dddot)
    x, y = st[:, 0], st[:, 1]
    v, psi = st[:, 2], st[:, 3]
    a_long, a_lat = st[:, 4], st[:, 5]
    j_long, j_lat = st[:, 6], st[:, 7]

    # Inverse: vehicle → Frenet
    s2, d2, s_dot2, d_dot2, s_ddot2, d_ddot2, s_dddot2, d_dddot2 = \
        gen.from_vehicle_states(x, y, v, psi, a_long, a_lat, j_long, j_lat)

    assert_allclose(np.array(s), np.array(s2), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(d), np.array(d2), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(s_dot), np.array(s_dot2), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(d_dot), np.array(d_dot2), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(s_ddot), np.array(s_ddot2), atol=1e-4, rtol=1e-4)
    assert_allclose(np.array(d_ddot), np.array(d_ddot2), atol=1e-4, rtol=1e-4)
    assert_allclose(np.array(s_dddot), np.array(s_dddot2), atol=1e-4, rtol=1e-4)
    assert_allclose(np.array(d_dddot), np.array(d_dddot2), atol=1e-4, rtol=1e-4)


def test_curved_forward_inverse():
    """to_vehicle_states(from_vehicle_states(y)) ≈ y on circular road."""
    gen = _make_circular_gen(radius=100.0)
    T = gen.T

    key = np.random.RandomState(99)
    x = jnp.array(key.uniform(0, 50, T))
    y = jnp.array(key.uniform(-2, 2, T))
    v = jnp.array(key.uniform(5, 25, T))
    psi = jnp.array(key.uniform(-0.3, 0.3, T))
    a_long = jnp.array(key.uniform(-2, 2, T))
    a_lat = jnp.array(key.uniform(-1, 1, T))
    j_long = jnp.array(key.uniform(-0.5, 0.5, T))
    j_lat = jnp.array(key.uniform(-0.5, 0.5, T))

    # Inverse: vehicle → Frenet
    s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = \
        gen.from_vehicle_states(x, y, v, psi, a_long, a_lat, j_long, j_lat)

    # Forward: Frenet → vehicle
    st = gen.to_vehicle_states(s, d, s_dot, d_dot,
                                s_ddot, d_ddot, s_dddot, d_dddot)

    assert_allclose(np.array(x), np.array(st[:, 0]), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(y), np.array(st[:, 1]), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(v), np.array(st[:, 2]), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(a_long), np.array(st[:, 4]), atol=1e-4, rtol=1e-4)
    assert_allclose(np.array(a_lat), np.array(st[:, 5]), atol=1e-4, rtol=1e-4)
    assert_allclose(np.array(j_long), np.array(st[:, 6]), atol=1e-4, rtol=1e-4)
    assert_allclose(np.array(j_lat), np.array(st[:, 7]), atol=1e-4, rtol=1e-4)


def test_circular_geometry():
    """CircularReference: cartesian_to_frenet ∘ frenet_to_cartesian ≈ id."""
    ref = CircularReference(radius=50.0)  # κ_r = 0.02

    key = np.random.RandomState(7)
    T = 50
    s = jnp.array(key.uniform(0, 30, T))
    d = jnp.array(key.uniform(-3, 3, T))

    x, y = ref.frenet_to_cartesian(s, d)
    s2, d2 = ref.cartesian_to_frenet(x, y)

    assert_allclose(np.array(s), np.array(s2), atol=1e-5, rtol=1e-5)
    assert_allclose(np.array(d), np.array(d2), atol=1e-5, rtol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Physics spot-checks — independent verification of to_vehicle_states
# ═══════════════════════════════════════════════════════════════════════

def test_spot_circular_centripetal():
    """Circular path: d=0, constant speed → pure centripetal a_lat = κ_r·v²."""
    gen = _make_circular_gen(radius=100.0)  # κ_r = 0.01
    T = 4

    v = 20.0
    s_arr = jnp.array([0.0, 10.0, 20.0, 30.0])
    d_arr = jnp.zeros(T)
    s_dot_arr = jnp.full(T, v)
    d_dot_arr = jnp.zeros(T)
    s_ddot_arr = jnp.zeros(T)
    d_ddot_arr = jnp.zeros(T)
    s_dddot_arr = jnp.zeros(T)
    d_dddot_arr = jnp.zeros(T)

    st = gen.to_vehicle_states(s_arr, d_arr, s_dot_arr, d_dot_arr,
                                s_ddot_arr, d_ddot_arr, s_dddot_arr, d_dddot_arr)
    # d=0, d_dot=0 → vt=v, vn=0, Δψ=0
    # a_n = d_ddot + κ_r·vt·s_dot = 0 + 0.01·v·v = 0.01·v²
    # a_lat = a_n (since Δψ=0)
    expected_a_lat = 0.01 * v ** 2  # = 4.0
    expected_a_long = 0.0

    assert_allclose(np.array(st[:, 5]), np.full(T, expected_a_lat), atol=1e-6)
    assert_allclose(np.array(st[:, 4]), np.full(T, expected_a_long), atol=1e-6)
    # Speed should equal s_dot (d_dot=0, d=0, κ_r·d = 0)
    assert_allclose(np.array(st[:, 2]), np.full(T, v), atol=1e-6)
    # Heading should increase linearly: ψ = θ_r = s/R
    assert_allclose(np.array(st[:, 3]), np.array(s_arr / 100.0), atol=1e-6)


def test_spot_circular_offset_d():
    """Circular path with d≠0: vehicle on circle of radius R-d."""
    gen = _make_circular_gen(radius=100.0)  # κ_r = 0.01, R = 100
    T = 3

    v = 15.0
    d_val = 2.0  # offset outward → effective radius = 98 m
    s_arr = jnp.array([5.0, 15.0, 25.0])
    d_arr = jnp.full(T, d_val)
    s_dot_arr = jnp.full(T, v)
    d_dot_arr = jnp.zeros(T)
    s_ddot_arr = jnp.zeros(T)
    d_ddot_arr = jnp.zeros(T)
    s_dddot_arr = jnp.zeros(T)
    d_dddot_arr = jnp.zeros(T)

    st = gen.to_vehicle_states(s_arr, d_arr, s_dot_arr, d_dot_arr,
                                s_ddot_arr, d_ddot_arr, s_dddot_arr, d_dddot_arr)

    # vt = (1 - d·κ_r)·s_dot = (1 - 0.02)·15 = 14.7
    # a_n = κ_r·vt·s_dot = 0.01·14.7·15 = 2.205
    # Physics: centripetal acc = vt²/(R-d) = 14.7²/98 = 2.205  ✓
    jac = 1.0 - d_val * 0.01  # 0.98
    vt = jac * v  # 14.7
    expected_a_lat = 0.01 * vt * v  # κ_r·vt·s_dot = 2.205
    expected_speed = vt  # since vn=0

    assert_allclose(np.array(st[:, 5]), np.full(T, expected_a_lat), atol=1e-6)
    assert_allclose(np.array(st[:, 2]), np.full(T, expected_speed), atol=1e-6)
    # a_long should be 0 (no tangential acc)
    assert_allclose(np.array(st[:, 4]), np.full(T, 0.0), atol=1e-6)


def test_spot_straight_diagonal_velocity():
    """Straight road: constant diagonal velocity → no acceleration."""
    gen = _make_gen()  # StraightReference
    T = 3

    s_dot_val = 20.0
    d_dot_val = 3.0
    s_arr = jnp.array([10.0, 30.0, 50.0])
    d_arr = jnp.array([1.0, 4.0, 7.0])
    s_dot_arr = jnp.full(T, s_dot_val)
    d_dot_arr = jnp.full(T, d_dot_val)
    s_ddot_arr = jnp.zeros(T)
    d_ddot_arr = jnp.zeros(T)
    s_dddot_arr = jnp.zeros(T)
    d_dddot_arr = jnp.zeros(T)

    st = gen.to_vehicle_states(s_arr, d_arr, s_dot_arr, d_dot_arr,
                                s_ddot_arr, d_ddot_arr, s_dddot_arr, d_dddot_arr)

    expected_v = float(jnp.sqrt(s_dot_val ** 2 + d_dot_val ** 2))
    expected_psi = float(jnp.arctan2(d_dot_val, s_dot_val))

    assert_allclose(np.array(st[:, 2]), np.full(T, expected_v), atol=1e-6)
    assert_allclose(np.array(st[:, 3]), np.full(T, expected_psi), atol=1e-6)
    assert_allclose(np.array(st[:, 4]), np.full(T, 0.0), atol=1e-6)  # a_long=0
    assert_allclose(np.array(st[:, 5]), np.full(T, 0.0), atol=1e-6)  # a_lat=0


def test_spot_straight_constant_acceleration():
    """Straight road, d=0: constant s_ddot → a_long = s_ddot, a_lat = 0."""
    gen = _make_gen()  # StraightReference
    T = 4

    s_ddot_val = 2.0
    s_arr = jnp.array([0.0, 1.0, 2.0, 3.0])
    d_arr = jnp.zeros(T)
    # Linearly increasing speed: s_dot(t) = s_dot0 + s_ddot·t
    s_dot_arr = jnp.array([10.0, 12.0, 14.0, 16.0])
    d_dot_arr = jnp.zeros(T)
    s_ddot_arr = jnp.full(T, s_ddot_val)
    d_ddot_arr = jnp.zeros(T)
    s_dddot_arr = jnp.zeros(T)
    d_dddot_arr = jnp.zeros(T)

    st = gen.to_vehicle_states(s_arr, d_arr, s_dot_arr, d_dot_arr,
                                s_ddot_arr, d_ddot_arr, s_dddot_arr, d_dddot_arr)

    assert_allclose(np.array(st[:, 4]), np.full(T, s_ddot_val), atol=1e-6)
    assert_allclose(np.array(st[:, 5]), np.full(T, 0.0), atol=1e-6)
    assert_allclose(np.array(st[:, 2]), np.array(s_dot_arr), atol=1e-6)
    assert_allclose(np.array(st[:, 3]), np.full(T, 0.0), atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Reference generator tests
# ═══════════════════════════════════════════════════════════════════════

def _make_ctx(s0=0.0, s_dot0=15.0, s_ddot0=0.0, d0=0.0, d_dot0=0.0, d_ddot0=0.0):
    return {'s0': s0, 's_dot0': s_dot0, 's_ddot0': s_ddot0,
            'd0': d0, 'd_dot0': d_dot0, 'd_ddot0': d_ddot0}


def test_lane_change_reference():
    """Quintic lane change: C2 smooth, correct start/end values, kinematic consistency."""
    gen = _make_gen()
    ctx = _make_ctx(d0=0.0, s_dot0=15.0)

    ref = make_frenet_reference(gen, ctx, {
        'type': 'lane_change',
        'd_end': 3.5,
        't_start': 0.5,
        't_duration': 3.0,
        'v_desired': 20.0,
    })

    T = gen.T
    dt = gen.dt

    # Start (t < t_start): d = d0 = 0, d_dot = 0
    idx_before = int(0.4 / dt)
    assert_allclose(np.array(ref['d_ref'][:idx_before]), np.zeros(idx_before), atol=1e-6)
    assert_allclose(np.array(ref['d_dot_ref'][:idx_before]), np.zeros(idx_before), atol=1e-6)
    assert_allclose(np.array(ref['d_ddot_ref'][:idx_before]), np.zeros(idx_before), atol=1e-6)

    # End (t > t_start + t_duration): d = d_end = 3.5, d_dot = d_ddot = 0
    idx_after = int(4.0 / dt)
    assert_allclose(np.array(ref['d_ref'][idx_after:]), np.full(T - idx_after, 3.5), atol=1e-6)
    assert_allclose(np.array(ref['d_dot_ref'][idx_after:]), np.zeros(T - idx_after), atol=1e-6)
    assert_allclose(np.array(ref['d_ddot_ref'][idx_after:]), np.zeros(T - idx_after), atol=1e-6)

    # Kinematic consistency on straight road: v² = s_dot² + d_dot²
    # Reconstruct v_ref from the exponential profile
    v0, v_tgt = 15.0, 20.0
    lam = 1.0
    t_arr = np.arange(T) * dt
    v_ref_expected = v_tgt + (v0 - v_tgt) * np.exp(-lam * t_arr)
    v_ref_actual = np.sqrt(np.array(ref['s_dot_ref'])**2 + np.array(ref['d_dot_ref'])**2)
    assert_allclose(v_ref_actual, v_ref_expected, atol=1e-5)

    # Before maneuver: s_dot = v (d_dot=0)
    assert_allclose(float(ref['s_dot_ref'][0]), 15.0, atol=1e-5)
    # After maneuver: s_dot ≈ v_tgt (d_dot=0)
    assert float(ref['s_dot_ref'][-1]) > 19.0


def test_cruise_reference():
    """Cruise: d stays constant, speed transitions to target."""
    gen = _make_gen()
    ctx = _make_ctx(d0=1.0, s_dot0=10.0)

    ref = make_frenet_reference(gen, ctx, {
        'type': 'cruise',
        'v_desired': 25.0,
    })

    T = gen.T
    assert_allclose(np.array(ref['d_ref']), np.full(T, 1.0), atol=1e-6)
    assert_allclose(np.array(ref['d_dot_ref']), np.zeros(T), atol=1e-6)
    assert_allclose(np.array(ref['d_ddot_ref']), np.zeros(T), atol=1e-6)
    assert float(ref['s_dot_ref'][0]) == 10.0


def test_external_reference_round_trip():
    """External mode: from_vehicle_states correctly embedded in make_frenet_reference."""
    gen = _make_gen()
    T = gen.T

    # Build a simple vehicle-state reference on straight road
    key = np.random.RandomState(123)
    x_arr = jnp.array(key.uniform(0, 50, T))
    y_arr = jnp.array(key.uniform(-2, 2, T))
    v_arr = jnp.array(key.uniform(5, 25, T))
    psi_arr = jnp.zeros(T)
    a_long_arr = jnp.zeros(T)
    a_lat_arr = jnp.zeros(T)
    j_long_arr = jnp.zeros(T)
    j_lat_arr = jnp.zeros(T)
    steer_arr = jnp.zeros(T)
    y_ref = jnp.stack([x_arr, y_arr, v_arr, psi_arr,
                       a_long_arr, a_lat_arr, j_long_arr, j_lat_arr, steer_arr], axis=-1)

    ctx = _make_ctx()
    ref = make_frenet_reference(gen, ctx, {
        'type': 'external',
        'vehicle_states': y_ref,
    })

    # Straight road with ψ=0: s_ref ≈ x, d_ref ≈ y, s_dot ≈ v
    assert_allclose(np.array(ref['s_ref']), np.array(x_arr), atol=1e-6)
    assert_allclose(np.array(ref['d_ref']), np.array(y_arr), atol=1e-6)
    assert_allclose(np.array(ref['s_dot_ref']), np.array(v_arr), atol=1e-6)
    assert_allclose(np.array(ref['d_dot_ref']), np.zeros(T), atol=1e-6)


def test_curved_cruise_kinematics():
    """Curved road cruise: s_dot = v / (1−d·κ_r) when d_dot=0."""
    gen = _make_circular_gen(radius=100.0)  # κ_r = 0.01
    ctx = _make_ctx(d0=2.0, s_dot0=10.0)    # offset 2m outward

    ref = make_frenet_reference(gen, ctx, {
        'type': 'cruise',
        'v_desired': 15.0,
    })

    T = gen.T
    dt = gen.dt
    t_arr = np.arange(T) * dt

    # Reconstruct v_ref from exponential
    v0, v_tgt = 10.0, 15.0
    lam = 1.0
    v_ref_expected = v_tgt + (v0 - v_tgt) * np.exp(-lam * t_arr)

    # Kinematic consistency on curved road: v² = (1−d·κ)²·s_dot² + d_dot²
    kap = 0.01
    jac = 1.0 - np.array(ref['d_ref']) * kap
    v_ref_actual = np.sqrt((jac * np.array(ref['s_dot_ref']))**2
                           + np.array(ref['d_dot_ref'])**2)
    assert_allclose(v_ref_actual, v_ref_expected, atol=1e-5)

    # d_dot should be zero for cruise
    assert_allclose(np.array(ref['d_dot_ref']), np.zeros(T), atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# CircularReference (from reference_path) geometry tests
# ═══════════════════════════════════════════════════════════════════════

def test_circular_reference_geometry():
    """Verify CircularReference basic geometry from reference_path."""
    R, cx, cy = 20.0, 0.0, 0.0
    ref = CircularRefPath(R, cx, cy)

    # s=0 at top: (0, R)
    x, y, θ, κ = ref.evaluate(jnp.array([0.0]))
    assert_allclose(x, 0.0, atol=1e-6)
    assert_allclose(y, R, atol=1e-6)
    assert_allclose(κ, 1.0 / R, atol=1e-6)

    # s=πR/2 at right: (R, 0)
    x, y, θ, κ = ref.evaluate(jnp.array([jnp.pi * R / 2]))
    assert_allclose(x, R, atol=1e-4)
    assert_allclose(y, 0.0, atol=1e-4)

    # s=πR at bottom: (0, -R)
    x, y, θ, κ = ref.evaluate(jnp.array([jnp.pi * R]))
    assert_allclose(x, 0.0, atol=1e-4)
    assert_allclose(y, -R, atol=1e-4)

    # s=3πR/2 at left: (-R, 0)
    x, y, θ, κ = ref.evaluate(jnp.array([3 * jnp.pi * R / 2]))
    assert_allclose(x, -R, atol=1e-4)
    assert_allclose(y, 0.0, atol=1e-4)

    # arclength consistent: each quarter = 31.416m for R=20
    assert_allclose(jnp.pi * R / 2, 31.4159265, atol=1e-4)


def test_circular_reference_round_trip():
    """cartesian_to_frenet ∘ evaluate = identity on CircularReference."""
    R, cx, cy = 20.0, 0.0, 0.0
    ref = CircularRefPath(R, cx, cy)

    # Test at multiple arclengths
    s_in = jnp.linspace(0, 2 * jnp.pi * R, 200)
    x, y, θ_r, κ_r = ref.evaluate(s_in)
    s_out, d_out = ref.cartesian_to_frenet(x, y)

    # On centerline → d_out ≈ 0
    assert_allclose(np.array(d_out), 0.0, atol=1e-5)
    # s may wrap: s=0 and s=2πR are the same point
    s_diff = np.minimum(np.abs(np.array(s_out) - np.array(s_in)),
                        2 * np.pi * R - np.abs(np.array(s_out) - np.array(s_in)))
    assert_allclose(s_diff, 0.0, atol=2e-5)  # float32 sin/cos precision

    # Test with offsets: d ≠ 0
    d_in = jnp.array([-3.0, 0.0, 2.5])
    s_test = jnp.array([0.0, jnp.pi * R, 3 * jnp.pi * R / 2])
    for si, di in zip(s_test, d_in):
        x, y = ref.frenet_to_cartesian(si, di)
        s_out, d_out = ref.cartesian_to_frenet(x, y)
        assert_allclose(s_out, si, atol=1e-6)
        assert_allclose(d_out, di, atol=1e-6)

    # κ_r constant everywhere
    _, _, _, κ_all = ref.evaluate(jnp.linspace(0, 100, 50))
    assert_allclose(κ_all, 1.0 / R * jnp.ones(50), atol=1e-6)


def test_general_path_geometry():
    """GeneralPath.evaluate() returns correct positions at s = 0, L/4, L/2, 3L/4, L.

    Uses a simple sinusoidal S-bend with known geometry.
    """
    import sys, tempfile, os
    from pathlib import Path as _Path
    from Cartest.core.reference_path import GeneralPath as _GeneralPath

    # Build a small test path: single-sine S-bend, R_min=50m, 100m long
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from Cartest.basis.build_path import build_sinusoid, save_table

    with tempfile.TemporaryDirectory() as tmp:
        path_file = _Path(tmp) / "test_path.npz"
        data = build_sinusoid(R_min=50.0, n_cycles=1.0, L=100.0, ds=0.1)
        save_table(data, path_file)

        gp = _GeneralPath(str(path_file))

        # s = 0: should be at origin, heading 0 (sin integral starts at 0)
        x0, y0, th0, k0 = gp.evaluate(jnp.array(0.0))
        assert_allclose(np.array(x0), np.array(0.0), atol=0.01)
        assert_allclose(np.array(y0), np.array(0.0), atol=0.01)
        # κ(s) = (1/50)·cos(2π·s/100), κ(0) = 1/50 = 0.02
        assert_allclose(np.array(k0), np.array(0.02), atol=0.001)

        # s = 50 (half-way): κ(50) = -0.02 (inverted)
        x50, y50, th50, k50 = gp.evaluate(jnp.array(50.0))
        assert_allclose(np.array(k50), np.array(-0.02), atol=0.001)

        # s = 100 (end): κ(100) = 0.02 (back to max, same as start)
        x100, y100, th100, k100 = gp.evaluate(jnp.array(100.0))
        assert_allclose(np.array(k100), np.array(0.02), atol=0.001)


def test_general_path_round_trip():
    """GeneralPath: cartesian_to_frenet(frenet_to_cartesian(s,d)) = (s,d)."""
    import sys, tempfile
    from pathlib import Path as _Path
    from Cartest.core.reference_path import GeneralPath as _GeneralPath

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from Cartest.basis.build_path import build_sinusoid, save_table

    with tempfile.TemporaryDirectory() as tmp:
        path_file = _Path(tmp) / "test_path.npz"
        data = build_sinusoid(R_min=20.0, n_cycles=2.0, L=150.0, ds=0.05)
        save_table(data, path_file)

        gp = _GeneralPath(str(path_file))

        s_in = jnp.linspace(10.0, 140.0, 20)
        x, y = gp.frenet_to_cartesian(s_in, jnp.zeros_like(s_in))
        s_out, d_out = gp.cartesian_to_frenet(x, y)

        # Position should recover perfectly
        assert_allclose(np.array(s_out), np.array(s_in), atol=0.01)
        assert_allclose(np.array(d_out), np.zeros(20), atol=0.01)

        # With offset
        d_offsets = jnp.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        s_fixed = jnp.full_like(d_offsets, 75.0)
        x_off, y_off = gp.frenet_to_cartesian(s_fixed, d_offsets)
        _, d_back = gp.cartesian_to_frenet(x_off, y_off)
        assert_allclose(np.array(d_back), np.array(d_offsets), atol=0.02)


def test_general_path_curved_round_trip():
    """Full Frenet↔Vehicle round-trip through GeneralPath (variable curvature)."""
    import sys, tempfile
    from pathlib import Path as _Path
    from Cartest.core.reference_path import GeneralPath as _GeneralPath
    from Cartest.core.frenet_traj import FrenetBSplineTrajectory as _Gen

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from Cartest.basis.build_path import build_sinusoid, save_table

    with tempfile.TemporaryDirectory() as tmp:
        path_file = _Path(tmp) / "test_path.npz"
        data = build_sinusoid(R_min=30.0, n_cycles=2.0, L=200.0, ds=0.05)
        save_table(data, path_file)

        gp = _GeneralPath(str(path_file))
        basis = _Path(__file__).resolve().parents[2] / "Cartest" / "basis" / "bspline_basis.npz"
        gen = _Gen(str(basis), gp)

        T = gen.T
        key = np.random.RandomState(42)
        # Small excursions to stay within reasonable d range
        s_in = jnp.array(key.uniform(20, 180, T))
        d_in = jnp.array(key.uniform(-2, 2, T))
        s_dot = jnp.array(key.uniform(5, 15, T))
        d_dot = jnp.array(key.uniform(-1, 1, T))
        s_ddot = jnp.zeros(T)
        d_ddot = jnp.zeros(T)

        # Forward
        st = gen.to_vehicle_states(s_in, d_in, s_dot, d_dot,
                                    s_ddot, d_ddot,
                                    jnp.zeros(T), jnp.zeros(T))

        # Inverse
        s_out, d_out, s_dot_out, d_dot_out, s_ddot_out, d_ddot_out, _, _ = \
            gen.from_vehicle_states(st[:, 0], st[:, 1], st[:, 2], st[:, 3],
                                     st[:, 4], st[:, 5], st[:, 6], st[:, 7])

        assert_allclose(np.array(s_out), np.array(s_in), atol=0.05)
        assert_allclose(np.array(d_out), np.array(d_in), atol=0.05)
        assert_allclose(np.array(s_dot_out), np.array(s_dot), atol=0.1)
        assert_allclose(np.array(d_dot_out), np.array(d_dot), atol=0.05)


if __name__ == "__main__":
    test_from_vehicle_states_round_trip()
    print("✓ test_from_vehicle_states_round_trip passed")

    test_forward_inverse_round_trip()
    print("✓ test_forward_inverse_round_trip passed")

    test_zero_speed_degenerate()
    print("✓ test_zero_speed_degenerate passed")

    test_perfect_tracking_consistency()
    print("✓ test_perfect_tracking_consistency passed")

    test_no_jerk_input()
    print("✓ test_no_jerk_input passed")

    test_round_trip_curved()
    print("✓ test_round_trip_curved passed")

    test_curved_forward_inverse()
    print("✓ test_curved_forward_inverse passed")

    test_circular_geometry()
    print("✓ test_circular_geometry passed")

    test_spot_circular_centripetal()
    print("✓ test_spot_circular_centripetal passed")

    test_spot_circular_offset_d()
    print("✓ test_spot_circular_offset_d passed")

    test_spot_straight_diagonal_velocity()
    print("✓ test_spot_straight_diagonal_velocity passed")

    test_spot_straight_constant_acceleration()
    print("✓ test_spot_straight_constant_acceleration passed")

    test_lane_change_reference()
    print("✓ test_lane_change_reference passed")

    test_cruise_reference()
    print("✓ test_cruise_reference passed")

    test_external_reference_round_trip()
    print("✓ test_external_reference_round_trip passed")

    test_curved_cruise_kinematics()
    print("✓ test_curved_cruise_kinematics passed")

    test_circular_reference_geometry()
    print("✓ test_circular_reference_geometry passed")

    test_circular_reference_round_trip()
    print("✓ test_circular_reference_round_trip passed")

    test_general_path_geometry()
    print("✓ test_general_path_geometry passed")

    test_general_path_round_trip()
    print("✓ test_general_path_round_trip passed")

    test_general_path_curved_round_trip()
    print("✓ test_general_path_curved_round_trip passed")

    print("\nAll tests passed.")
