"""Offline reference-path table builder — curvature profile → (s, x, y, θ, κ) .npz.

Generates a dense 1D lookup table for ``GeneralPath`` by integrating the
Frenet–Serret ODE from a user-supplied curvature profile κ(s).

Supports four input modes::

    # Sinusoidal S-bend (R_min defines max curvature)
    uv run python Cartest/basis/build_path.py --type sinusoid \
        --R-min 20 --n-cycles 3 --total-length 200 \
        --out Cartest/basis/path_s_bend.npz

    # Arbitrary curvature function (Python lambda string)
    uv run python Cartest/basis/build_path.py --type curvature \
        --kappa "(1/20)*sin(0.1*s)" --total-length 200 \
        --out Cartest/basis/path_custom.npz

    # CSV / waypoints — cubic-spline fit → arc-length reparameterisation → κ(s)
    uv run python Cartest/basis/build_path.py --type waypoints \
        --input waypoints.csv --total-length 200 \
        --out Cartest/basis/path_waypoints.npz

    # Multi-segment (line / arc / clothoid)
    uv run python Cartest/basis/build_path.py --type segments \
        --segments "L50,A100:-90,C50:0.02,L80" \
        --out Cartest/basis/path_segments.npz

The resulting .npz file can be loaded by ``GeneralPath`` (in reference_path.py)
for JAX-compatible online lookup.
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from scipy.integrate import cumulative_trapezoid


DEFAULT_DS = 0.05          # Default arc-length step (m) — ~4000 pts for 200 m
DEFAULT_OUT = Path(__file__).with_name("path_table.npz")


# ═══════════════════════════════════════════════════════════════════════
# Path builders
# ═══════════════════════════════════════════════════════════════════════

def build_sinusoid(R_min: float, n_cycles: int, L: float,
                   ds: float = DEFAULT_DS) -> dict:
    """Sinusoidal S-bend: κ(s) = κ_max · cos(2π·n_cycles·s/L).

    R_min defines the tightest turn — κ_max = 1 / R_min.
    """
    kappa_max = 1.0 / R_min
    omega = 2.0 * np.pi * n_cycles / L

    def kappa_fn(s):
        return kappa_max * np.cos(omega * s)

    return _integrate_ode(kappa_fn, L, ds)


def build_curvature(kappa_expr: str, L: float,
                    ds: float = DEFAULT_DS) -> dict:
    """Arbitrary curvature function given as a Python expression of s.

    Example: ``"(1/20)*sin(0.1*s)"``
    """
    import math  # noqa: F401 — available inside eval scope

    safe_ns = {"s": None, "np": np, "math": math, "sin": np.sin,
               "cos": np.cos, "exp": np.exp, "tanh": np.tanh, "pi": np.pi}

    def kappa_fn(s):
        safe_ns["s"] = s
        return eval(kappa_expr, {"__builtins__": {}}, safe_ns)

    return _integrate_ode(kappa_fn, L, ds)


def build_waypoints(xy: np.ndarray, ds: float = DEFAULT_DS) -> dict:
    """Cubic-spline arc-length reparameterisation of (x, y) waypoints.

    Steps:
      1. Fit periodic cubic splines x(t), y(t) to waypoints
      2. Compute arc-length s(t) = ∫√(x'² + y'²) dt
      3. Reparameterise: evaluate x(s), y(s) on uniform s-grid
      4. Numerically differentiate to get θ(s), κ(s)
    """
    from scipy.interpolate import CubicSpline

    # Parameterise by cumulative chord length
    t = np.concatenate([[0.0], np.cumsum(np.sqrt(
        np.diff(xy[:, 0]) ** 2 + np.diff(xy[:, 1]) ** 2))])

    # Fit splines
    cs_x = CubicSpline(t, xy[:, 0], bc_type='natural')
    cs_y = CubicSpline(t, xy[:, 1], bc_type='natural')

    # Arc-length as function of t
    t_fine = np.linspace(0, t[-1], max(10000, int(t[-1] / ds * 2)))
    dx_dt = cs_x.derivative(1)(t_fine)
    dy_dt = cs_y.derivative(1)(t_fine)
    ds_dt = np.sqrt(dx_dt ** 2 + dy_dt ** 2)
    s_t = cumulative_trapezoid(ds_dt, t_fine, initial=0.0)
    L = float(s_t[-1])

    # Uniform s grid
    s_grid = np.arange(0, L, ds)
    if s_grid[-1] < L - 1e-6:
        s_grid = np.append(s_grid, L)

    # Invert s(t) → t(s) via interpolation
    t_of_s = np.interp(s_grid, s_t, t_fine)

    # Evaluate at uniform s
    x = cs_x(t_of_s)
    y = cs_y(t_of_s)

    # Numerical derivatives
    dx_ds = np.gradient(x, ds)
    dy_ds = np.gradient(y, ds)
    theta = np.arctan2(dy_ds, dx_ds)
    d2x_ds2 = np.gradient(dx_ds, ds)
    d2y_ds2 = np.gradient(dy_ds, ds)
    kappa = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (dx_ds ** 2 + dy_ds ** 2) ** 1.5

    return {"s": s_grid, "x": x, "y": y, "theta": theta, "kappa": kappa,
            "L": L, "ds": ds, "description": f"waypoints ({len(xy)} pts)"}


def build_segments(spec_str: str, ds: float = DEFAULT_DS) -> dict:
    """Multi-segment path from a compact description string.

    Segments are comma-separated.  Each segment is a letter followed by params:

      L<length>           — straight line
      A<radius>:<deg>      — circular arc (positive deg = CCW / left turn)
      C<rate>              — clothoid (ds² curvature transition)

    Example: ``"L50,A100:-90,C0.02,L80"``
      = 50 m straight, 90° right turn on R=100, clothoid transition, 80 m straight.
    """
    segments = [seg.strip() for seg in spec_str.split(",")]
    s_list, x_list, y_list, theta_list, kappa_list = [], [], [], [], []

    x_cur, y_cur = 0.0, 0.0  # current Cartesian position
    theta_cur = 0.0           # current heading (rad)
    kappa_cur = 0.0           # current curvature
    s_cum = 0.0

    for seg_spec in segments:
        seg_type = seg_spec[0].upper()
        params = seg_spec[1:]

        if seg_type == "L":
            # Straight line
            length = float(params)
            s_seg = np.arange(0, length, ds)
            if s_seg[-1] < length - 1e-6:
                s_seg = np.append(s_seg, length)
            theta_seg = np.full_like(s_seg, theta_cur)
            kappa_seg = np.zeros_like(s_seg)
            x_seg = x_cur + s_seg * np.cos(theta_cur)
            y_seg = y_cur + s_seg * np.sin(theta_cur)
            # Update state
            x_cur, y_cur = x_seg[-1], y_seg[-1]
            theta_cur = theta_cur  # unchanged
            kappa_cur = 0.0

        elif seg_type == "A":
            # Circular arc
            R, deg = params.split(":")
            R = float(R)
            deg = float(deg)
            arc_len = abs(np.radians(deg) * R)
            s_seg = np.arange(0, arc_len, ds)
            if s_seg[-1] < arc_len - 1e-6:
                s_seg = np.append(s_seg, arc_len)
            sign = 1.0 if deg >= 0 else -1.0
            kappa_seg = np.full_like(s_seg, sign / R)
            theta_seg = theta_cur + kappa_seg * s_seg
            # Integrate position via cumulative trapezoid
            dx = np.cos(theta_seg)
            dy = np.sin(theta_seg)
            x_seg = x_cur + cumulative_trapezoid(dx, s_seg, initial=0.0)
            y_seg = y_cur + cumulative_trapezoid(dy, s_seg, initial=0.0)
            # Update state
            x_cur, y_cur = x_seg[-1], y_seg[-1]
            theta_cur = theta_seg[-1]
            kappa_cur = kappa_seg[-1]

        elif seg_type == "C":
            # Clothoid: linear curvature ramp
            rate = float(params)  # dκ/ds (1/m²)
            # Determine segment length from curvature change needed
            # For now, require explicit length via syntax C<rate>:<length>
            raise NotImplementedError(
                "Clothoid requires explicit length: C<rate>:<length>")

        else:
            raise ValueError(f"Unknown segment type '{seg_type}'")

        s_list.append(s_cum + s_seg)
        x_list.append(x_seg)
        y_list.append(y_seg)
        theta_list.append(theta_seg)
        kappa_list.append(kappa_seg)
        s_cum += float(s_seg[-1])

    s_all = np.concatenate(s_list)
    # Handle potential duplicate points at segment boundaries
    mask = np.concatenate([[True], np.diff(s_all) > 1e-10])
    s_all = s_all[mask]

    x_all = np.interp(s_all,
                      np.concatenate([s_list[0][:1], np.array([sl[-1] for sl in s_list])]),
                      np.concatenate([x_list[0][:1], np.array([xl[-1] for xl in x_list])]))

    return {"s": s_all, "x": x_all, "y": np.interp(s_all, s_all, np.concatenate(y_list)),
            "theta": np.interp(s_all, s_all, np.concatenate(theta_list)),
            "kappa": np.interp(s_all, s_all, np.concatenate(kappa_list)),
            "L": float(s_all[-1]), "ds": ds,
            "description": f"segments: {spec_str}"}


# ═══════════════════════════════════════════════════════════════════════
# Core: ODE integration
# ═══════════════════════════════════════════════════════════════════════

def _integrate_ode(kappa_fn, L: float, ds: float) -> dict:
    """Integrate Frenet–Serret ODE on uniform s-grid.

    dθ/ds = κ(s),  dx/ds = cos(θ),  dy/ds = sin(θ)
    """
    s = np.arange(0.0, L, ds)
    if s[-1] < L - 1e-9:
        s = np.append(s, L)

    kappa = np.asarray(kappa_fn(s), dtype=np.float64)
    theta = cumulative_trapezoid(kappa, s, initial=0.0)
    x = cumulative_trapezoid(np.cos(theta), s, initial=0.0)
    y = cumulative_trapezoid(np.sin(theta), s, initial=0.0)

    return {"s": s, "x": x, "y": y, "theta": theta, "kappa": kappa,
            "L": L, "ds": ds, "description": "custom κ(s)"}


# ═══════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════

def save_table(data: dict, path: Path):
    """Save path table to .npz, validated."""
    s = data["s"]
    x = data["x"]
    y = data["y"]
    theta = data["theta"]
    kappa = data["kappa"]

    # Validation
    assert s[0] == 0.0, f"s[0] must be 0, got {s[0]}"
    assert len(s) >= 100, f"too few points ({len(s)}), reduce ds"
    assert np.all(np.diff(s) > 0), "s must be strictly increasing"
    assert np.isfinite(x).all() and np.isfinite(y).all(), "NaN in path"
    assert np.isfinite(theta).all() and np.isfinite(kappa).all(), "NaN in theta/kappa"

    np.savez(
        path,
        s=s.astype(np.float64),
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        theta=theta.astype(np.float32),
        kappa=kappa.astype(np.float32),
        L=np.float64(data["L"]),
        ds=np.float64(data.get("ds", s[1] - s[0])),
    )

    ds_actual = s[1] - s[0]
    print(f"Saved {path}")
    print(f"  points:   {len(s):,}")
    print(f"  length:   {data['L']:.1f} m")
    print(f"  ds:       {ds_actual:.3f} m")
    print(f"  κ range:  [{kappa.min():.4f}, {kappa.max():.4f}]  1/m")
    print(f"  R range:  [{1/abs(kappa).max() if abs(kappa).max()>1e-12 else np.inf:.1f}, "
          f"{1/abs(kappa).min() if abs(kappa).min()>1e-12 else np.inf:.1f}] m  (min/max)")
    print(f"  x range:  [{x.min():.1f}, {x.max():.1f}] m")
    print(f"  y range:  [{y.min():.1f}, {y.max():.1f}] m")
    print(f"  type:     {data.get('description', 'custom')}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build reference-path lookup table for GeneralPath")
    parser.add_argument("--type", required=True,
                        choices=["sinusoid", "curvature", "waypoints", "segments"])
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Output .npz path")

    # Sinusoid params
    parser.add_argument("--R-min", type=float, default=20.0,
                        help="Minimum curvature radius (m)")
    parser.add_argument("--n-cycles", type=float, default=3.0,
                        help="Number of full sine cycles")
    parser.add_argument("--total-length", type=float, default=200.0,
                        help="Total path length (m)")

    # Curvature expression
    parser.add_argument("--kappa", type=str, default=None,
                        help="Python expression of s for κ(s), e.g. '(1/20)*sin(0.1*s)'")

    # Waypoints
    parser.add_argument("--input", type=str, default=None,
                        help="CSV file with x,y columns (no header)")

    # Segments
    parser.add_argument("--segments", type=str, default=None,
                        help="Segment spec string, e.g. 'L50,A100:-90,L80'")

    # Common
    parser.add_argument("--ds", type=float, default=DEFAULT_DS,
                        help=f"Arc-length step (m), default {DEFAULT_DS}")

    args = parser.parse_args()

    if args.type == "sinusoid":
        data = build_sinusoid(args.R_min, args.n_cycles,
                              args.total_length, args.ds)
        data["description"] = (f"S-bend: R_min={args.R_min}m, "
                               f"n_cycles={args.n_cycles}, L={args.total_length}m")

    elif args.type == "curvature":
        if args.kappa is None:
            parser.error("--type curvature requires --kappa")
        data = build_curvature(args.kappa, args.total_length, args.ds)
        data["description"] = f"κ(s) = {args.kappa}, L={args.total_length}m"

    elif args.type == "waypoints":
        if args.input is None:
            parser.error("--type waypoints requires --input")
        xy = np.loadtxt(args.input, delimiter=",", dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError("Waypoints file must have at least 2 columns (x, y)")
        data = build_waypoints(xy, args.ds)

    elif args.type == "segments":
        if args.segments is None:
            parser.error("--type segments requires --segments")
        data = build_segments(args.segments, args.ds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_table(data, out_path)


if __name__ == "__main__":
    main()
