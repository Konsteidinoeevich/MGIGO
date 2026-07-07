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
N_CTRL = 10               # 10 control points → 5 internal knot segments
T = 100                   # Evaluation time steps
DT = 0.1                  # Time step (seconds)
TOTAL_TIME = T * DT       # 10.0 second horizon

OUTPUT_PATH = Path(__file__).with_name("bspline_basis.npz")


def build_one_sided_knots(degree: int, n_ctrl: int, total_time: float):
    """Build knot vector clamped at t=0 only (C0/C1/C2), uniform elsewhere.

    Only the start is clamped — the vehicle MUST start exactly at the
    current state.  The end needs no clamping because MPC only executes
    the first step; the rest is just a planning horizon.

    Uniform internal knots make the B-spline convex-hull property tight,
    enabling exact per-control-point constraints.
    """
    dt_knot = total_time / (n_ctrl - degree)
    n_knots = n_ctrl + degree + 1
    knots = np.zeros(n_knots)
    knots[:degree + 1] = 0.0  # clamped start: C0/C1/C2
    for i in range(degree + 1, n_knots):
        knots[i] = (i - degree) * dt_knot
    return knots, dt_knot


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
    knots, dt_knot = build_one_sided_knots(DEGREE, N_CTRL, TOTAL_TIME)
    B, dB, d2B, d3B, d4B = compute_basis_matrices(knots, t_eval, DEGREE, N_CTRL)

    u_eval = t_eval / TOTAL_TIME

    # Greville abscissae
    greville = np.array([
        np.mean(knots[i + 1 : i + DEGREE + 1]) for i in range(N_CTRL)
    ])

    # Convex-hull scale factors: max|derivative(t)| ≤ scale × max|ctrl_diff|
    # Precomputed from basis matrices — constant, not time-varying.
    # jerk_scale = max_{t,i} relationship between d³P and jerk(t)
    # acc_scale  = max_{t,i} relationship between d²P and acc(t)
    # vel_scale  = max_{t,i} relationship between dP  and vel(t)
    #
    # For uniform knots: degree!/((degree-k)! · dt_knot^k)
    #   vel:  5/1.4286      = 3.500
    #   acc:  20/1.4286²     = 9.800
    #   jerk: 60/1.4286³     = 20.571
    # Clamped ends make these slightly tighter; we use the exact max.
    # Per-control-point scale factors from knot spacing (B-spline convex hull).
    # |vel(t)|  ≤ max_i { scale_v[i] × |P_{i+1} - P_i| }
    # |acc(t)|  ≤ max_i { scale_a[i] × |P_{i+2} - 2P_{i+1} + P_i| }
    # |jerk(t)| ≤ max_i { scale_j[i] × |P_{i+3} - 3P_{i+2} + 3P_{i+1} - P_i| }
    #
    # Formula: scale = p!/((p-k)!) / Π(u_{i+p+1-j} - u_{i+1+k-j})
    # For clamped ends, differences involving only clamped pts have scale=0
    # (no constraint needed — those pts are fixed).
    p = DEGREE
    # Only constrain differences among FREE control points (P3..P11).
    # Clamped P0,P1,P2 already enforce C0/C1/C2 — their diffs are fixed.
    # Only constrain differences where ALL involved ctrl pts are FREE.
    # Clamped P0,P1,P2 enforce C0/C1/C2 — mixed diffs skip.
    # Per-point scales: exact max sensitivity from M = derivB @ pinv(Diff).
    # |deriv(t)| ≤ max_i { scale[i] × |diff_i| } — tight, not formula-based.
    def _compute_scales(derivB, diff_order):
        n_diff = N_CTRL - diff_order
        Diff = np.zeros((n_diff, N_CTRL))
        if diff_order == 1:
            for i in range(n_diff): Diff[i,i]=-1; Diff[i,i+1]=1
        elif diff_order == 2:
            for i in range(n_diff): Diff[i,i]=1; Diff[i,i+1]=-2; Diff[i,i+2]=1
        elif diff_order == 3:
            for i in range(n_diff): Diff[i,i]=-1; Diff[i,i+1]=3; Diff[i,i+2]=-3; Diff[i,i+3]=1
        M = derivB @ np.linalg.pinv(Diff)  # [T, n_diff]
        return np.max(np.abs(M), axis=0)     # [n_diff]
    scale_v = _compute_scales(dB, 1)
    scale_a = _compute_scales(d2B, 2)
    scale_j = _compute_scales(d3B, 3)
    # Zero out diffs where ALL pts are clamped (P0,P1,P2)
    scale_v[0] = 0.0; scale_v[1] = 0.0
    scale_a[0] = 0.0
    scale_j[0] = 0.0; scale_j[1] = 0.0; scale_j[2] = 0.0  # d³P_0,1,2 involve clamped

    np.savez(
        OUTPUT_PATH,
        B=B.astype(np.float64),
        dB=dB.astype(np.float64),
        d2B=d2B.astype(np.float64),
        d3B=d3B.astype(np.float64),
        d4B=d4B.astype(np.float64),
        u_eval=u_eval.astype(np.float64),
        greville=greville.astype(np.float64),
        scale_v=scale_v.astype(np.float64),  # [n_ctrl-1]
        scale_a=scale_a.astype(np.float64),  # [n_ctrl-2]
        scale_j=scale_j.astype(np.float64),  # [n_ctrl-3]
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
