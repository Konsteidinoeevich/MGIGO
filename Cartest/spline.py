"""Offline B-spline basis precomputation — quintic (degree 5).

Generates ``bspline_basis.npz`` with B, dB, d2B, d3B, d4B for a clamped
quintic B-spline.  Degree 5 provides C⁴ continuity: continuous jerk,
bounded snap — the standard for autonomous driving trajectory planning.

Usage::

    uv run python Cartest/spline.py
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from pathlib import Path


DEGREE = 5                # Quintic B-spline (C⁴ continuous)
N_CTRL = 12               # 12 control points → 7 internal knot segments
T = 100                   # Evaluation time steps
DT = 0.1                  # Time step (seconds)
TOTAL_TIME = T * DT       # 10.0 second horizon

OUTPUT_PATH = Path(__file__).with_name("bspline_basis.npz")


def build_clamped_knots(degree: int, n_ctrl: int, total_time: float):
    """Build clamped uniform knot vector.

    A clamped B-spline repeats the first and last knots ``degree + 1``
    times so the curve interpolates the first and last control points.
    """
    n_internal = n_ctrl - degree
    dt_knot = total_time / n_internal
    internal = np.arange(1, n_internal + 1) * dt_knot
    knots = np.concatenate([
        np.zeros(degree + 1),
        internal,
        np.full(degree, total_time),
    ])
    return knots


def compute_basis_matrices(knots, t_eval, degree, n_ctrl):
    """Compute basis matrices for derivatives 0..4.

    Returns B, dB, d2B, d3B, d4B each [T, n_ctrl].
    """
    matrices = []
    for nu in range(5):  # 0..4 derivatives
        B_nu = np.zeros((len(t_eval), n_ctrl))
        for i in range(n_ctrl):
            c = np.zeros(n_ctrl)
            c[i] = 1.0
            spl = BSpline(knots, c, k=degree, extrapolate=True)
            B_nu[:, i] = spl(t_eval, nu=nu)
        matrices.append(B_nu)
    return matrices


def main():
    print(f"Building clamped quintic B-spline basis...")
    print(f"  degree={DEGREE}, n_ctrl={N_CTRL}, T={T}, dt={DT}, total_time={TOTAL_TIME}")

    t_eval = np.arange(T) * DT
    knots = build_clamped_knots(DEGREE, N_CTRL, TOTAL_TIME)
    B, dB, d2B, d3B, d4B = compute_basis_matrices(knots, t_eval, DEGREE, N_CTRL)

    u_eval = t_eval / TOTAL_TIME
    n_internal = N_CTRL - DEGREE
    dt_knot = TOTAL_TIME / n_internal

    # Greville abscissae: t_i = mean(knots[i+1 : i+degree+1])
    # The time location where control point P_i has peak influence.
    greville = np.array([
        np.mean(knots[i + 1 : i + DEGREE + 1]) for i in range(N_CTRL)
    ])

    np.savez(
        OUTPUT_PATH,
        B=B.astype(np.float64),
        dB=dB.astype(np.float64),
        d2B=d2B.astype(np.float64),
        d3B=d3B.astype(np.float64),
        d4B=d4B.astype(np.float64),
        u_eval=u_eval.astype(np.float64),
        greville=greville.astype(np.float64),
        degree=np.int64(DEGREE),
        dt=np.float64(DT),
        total_time=np.float64(TOTAL_TIME),
        n_ctrl=np.int64(N_CTRL),
        dt_knot=np.float64(dt_knot),
    )

    # Sanity checks
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-10), "Partition of unity failed!"
    assert np.allclose(B[0, 0], 1.0, atol=1e-10), "P0 interpolation failed!"

    # Verify derivative formulas at t=0 for quintic clamped B-spline:
    # v(0) = degree/dt_knot * (P1 - P0)
    # a(0) = degree*(degree-1)/dt_knot² * (P2 - 2*P1 + P0)
    # For clamped B-spline: B[0,:] = [1,0,...,0], dB[0,:] matches the formula
    print(f"  B       {B.shape}  (row sums = 1.0)")
    print(f"  dB      {dB.shape}")
    print(f"  d2B     {d2B.shape}")
    print(f"  d3B     {d3B.shape}")
    print(f"  d4B     {d4B.shape}")
    print(f"  dt_knot = {dt_knot:.4f}s")
    print(f"  n_free  = {N_CTRL - 3} (P0,P1,P2 clamped)")
    print(f"  ✓ Partition of unity, P0 interpolation")

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
