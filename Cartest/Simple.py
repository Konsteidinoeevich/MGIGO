"""Quintic B-spline trajectory MPC demo — using Constran `build()` + `autodelta()`.

All kinematics from B-spline analytical derivatives.
Constraints: lane (soft), curvature (tunable), obstacle (hard).
"""

from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Cartest.bsplinetraj import BSplineTrajectoryGenerator
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc
from Constraintdealer.Constran import build, autodelta, Deterministic


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

V_TARGET = 18.0
IGO_STEPS, IGO_DT = 500, 0.15
B_SAMPLES, B0_ELITE, T0_RESET = 64, 20, 500
M, K = 2, 3  # block 0: X ctrl pts (9d), block 1: Y ctrl pts (9d)

W_POS_Y, W_VEL = 25.0, 5.0  # only these stay in objective; rest → constraints

ROAD_Y, WHEEL_BASE = 0.0, 2.8
LANE_HW, CURVATURE_MAX = 4.0, 0.15
OBS_SAFE_DIST = 2.0
V_MIN, V_MAX = 2.0, 35.0

OBSTACLES = [{"x": 60.0, "y": 2.5, "r": 2.0}]

OUTPUT = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════
# Trajectory evaluation helper (shared by objective + constraints)
# ═══════════════════════════════════════════════════════════════════════

def _eval_traj(theta, ctx, gen):
    """Unpack theta → evaluate trajectory.  theta = [X(9) | Y(9)] from M=2 blocks."""
    n = gen.n_free
    ctrl_free = jnp.stack([theta[:n], theta[n:2*n]], axis=1)  # [9, 2]
    pos, vel, acc, jerk = gen.evaluate(ctrl_free, ctx["x0"], ctx["v0"], ctx["a0"])
    st = gen.to_vehicle_states(pos, vel, acc, jerk)
    return pos, vel, acc, jerk, st


# ═══════════════════════════════════════════════════════════════════════
# Objective — only tracking (position + speed).  All other costs are
# constraints so they get T_alpha → σ nesting and proper scale handling.
# ═══════════════════════════════════════════════════════════════════════

def _make_objective_fn(gen):
    def obj_fn(theta, ctx):
        pos, vel, _, _, st = _eval_traj(theta, ctx, gen)
        speed = st[:,2]
        return (
            W_POS_Y * jnp.sum((pos[:,1] - ctx["y_ref"])**2) +
            W_VEL   * jnp.sum((speed    - ctx["v_ref"])**2)
        )
    return obj_fn


# ═══════════════════════════════════════════════════════════════════════
# Constraints — each returns a VECTOR [T] of per-sample-point values.
# Constran applies: aggregate → T_alpha → mode (hard/tunable/soft) → nest.
#
# Priority (outer to inner):
#   L1: obstacle  — hard,  max     (any point touching obstacle → violation)
#   L2: curvature — tunable, mean   (average curvature over trajectory)
#   L3: jerk      — soft,   mean   (smoothness preference)
#   L4: lane      — soft,   sum    (total lane deviation)
#   L5: speed lim — tunable, max    (any point exceeding V_MIN/V_MAX)
# ═══════════════════════════════════════════════════════════════════════

def _make_constraints(gen):
    def obs_g(theta, ctx):
        """碰撞: (T,) → max over obstacles then points. 最危险点决定安全."""
        pos, _, _, _, _ = _eval_traj(theta, ctx, gen)
        d = jnp.sqrt(jnp.sum((pos[:,None,:] - ctx["obs_pos"][None,:,:])**2, axis=-1))
        pen = jnp.maximum(0., OBS_SAFE_DIST + ctx["obs_rad"][None,:] - d)  # (T, N_obs)
        return jnp.min(pen, axis=-1)  # per-point: closest obstacle only

    def curv_g(theta, ctx):
        """曲率: (T,) → mean. 整体平滑度, 允许偶尔急转."""
        _, vel, acc, _, st = _eval_traj(theta, ctx, gen)
        speed, a_lat = st[:,2], st[:,5]
        curv = a_lat / (speed**2 + 1e-6)
        return jnp.maximum(0., jnp.abs(curv) - ctx["curv_max"])

    def jerk_g(theta, ctx):
        """Jerk: (T,) → mean. 舒适度偏好, 不平滑就扣分."""
        _, _, _, jerk, _ = _eval_traj(theta, ctx, gen)
        return jnp.abs(jerk[:, 0])  # longitudinal jerk magnitude

    def lane_g(theta, ctx):
        """车道偏离: (T,) → sum. 总偏离量, 允许偶尔压线."""
        pos, _, _, _, _ = _eval_traj(theta, ctx, gen)
        return jnp.maximum(0., jnp.abs(pos[:,1] - ROAD_Y) - ctx["lane_hw"])

    def speed_g(theta, ctx):
        """速度限制: (T,) → max. 任何一步超速就是违规."""
        _, _, _, _, st = _eval_traj(theta, ctx, gen)
        speed = st[:,2]
        return jnp.maximum(0., jnp.maximum(V_MIN - speed, speed - V_MAX))

    return autodelta([
        Deterministic(obs_g,   mode='hard',    priority=1, aggregate='max',
                      transform='hard'),
        Deterministic(curv_g,  mode='tunable', priority=2, aggregate='mean',
                      tune_preset='standard', transform='tunable'),
        Deterministic(jerk_g,  mode='soft',    priority=3, aggregate='mean',
                      transform='soft'),
        Deterministic(lane_g,  mode='soft',    priority=4, aggregate='sum',
                      transform='soft'),
        Deterministic(speed_g, mode='tunable', priority=5, aggregate='max',
                      tune_preset='firm', transform='tunable'),
    ])


# ═══════════════════════════════════════════════════════════════════════
# Warm-start
# ═══════════════════════════════════════════════════════════════════════

def _shift(ctrl_free, v_target, dt):
    s = jnp.zeros_like(ctrl_free)
    s = s.at[:-1].set(ctrl_free[1:])
    s = s.at[-1].set(ctrl_free[-1] + jnp.array([v_target * dt, 0.0]))
    return s


# ═══════════════════════════════════════════════════════════════════════
# MPC
# ═══════════════════════════════════════════════════════════════════════

def run(steps=150, seed=0, plot=True):
    gen = BSplineTrajectoryGenerator(OUTPUT / "bspline_basis.npz")
    D = gen.n_free * 2

    # Build cost function ONCE (Constran pattern: build before MPC loop)
    obj_fn = _make_objective_fn(gen)
    constraints = _make_constraints(gen)
    cost_fn = build(obj_fn, constraints, k_inner=0.1, obj_transform='standard')

    key = random.PRNGKey(seed)
    rx, ry, rv = 0.0, -3.0, 12.0
    rvx, rvy, ra = rv, 0.0, 0.0

    ctrl_free = gen.nominal_free_points(rv, ry)
    ctrl_x = ctrl_free[:, 0]  # [9]
    ctrl_y = ctrl_free[:, 1]  # [9]
    mu_init = jnp.stack([
        jnp.stack([ctrl_x, ctrl_x, ctrl_x], axis=0),
        jnp.stack([ctrl_y, ctrl_y, ctrl_y], axis=0),
    ], axis=0).astype(jnp.float32)  # [2, 3, 9]
    D_max = gen.n_free  # 9
    dims = (D_max, D_max)
    L_inv = jnp.tile(jnp.eye(D_max, dtype=jnp.float32)[None, None, :, :], (M, K, 1, 1))
    v_init = jnp.zeros((M, K-1), dtype=jnp.float32)

    obs_pos = jnp.array([[o["x"], o["y"]] for o in OBSTACLES], dtype=jnp.float32)
    obs_rad = jnp.array([o["r"] for o in OBSTACLES], dtype=jnp.float32)

    if plot:
        plt.ion(); fig, (ax_t, ax_k) = plt.subplots(1, 2, figsize=(16, 5))

    hx, hy, hv = [rx], [ry], [rv]
    frames = []

    for step in range(steps):
        key, sk = random.split(key)

        ctx = {
            "y_ref": jnp.full(gen.T, ROAD_Y),
            "v_ref": jnp.full(gen.T, V_TARGET),
            "x0": jnp.array([rx, ry], dtype=jnp.float32),
            "v0": jnp.array([rvx, rvy], dtype=jnp.float32),
            "a0": jnp.array([ra, 0.0], dtype=jnp.float32),
            "lane_hw": LANE_HW, "curv_max": CURVATURE_MAX,
            "obs_pos": obs_pos, "obs_rad": obs_rad,
        }

        t0 = time.time()
        mu_k, L_k, pi_k = mmog_igo_optimizer_mpc(
            sk, IGO_STEPS, IGO_DT, M, K, B_SAMPLES, B0_ELITE,
            dims, T0_RESET, cost_fn, mu_init, L_inv, v_init, ctx)
        mu_k.block_until_ready()
        ms = (time.time() - t0) * 1000

        best = int(jnp.argmax(pi_k[0]))
        ctrl_x = mu_k[0, best, :D_max]
        ctrl_y = mu_k[1, best, :D_max]
        ctrl_free = jnp.stack([ctrl_x, ctrl_y], axis=1)

        pos, vel, acc, jerk = gen.evaluate(ctrl_free, ctx["x0"], ctx["v0"], ctx["a0"])
        st = gen.to_vehicle_states(pos, vel, acc, jerk)

        prev_v = rv
        rx, ry = float(pos[1, 0]), float(pos[1, 1])
        rv = float(jnp.clip(st[1, 2], V_MIN, V_MAX))
        rvx, rvy = float(vel[1, 0]), float(vel[1, 1])
        ra = (rv - prev_v) / gen.dt

        hx.append(rx); hy.append(ry); hv.append(rv)

        a_long, a_lat, j_long = st[:,4], st[:,5], st[:,6]
        min_obs = float(jnp.min(jnp.sqrt(jnp.sum((pos[:,None,:]-obs_pos[None,:,:])**2, axis=-1))-obs_rad[None,:]))
        ma, ml, mj = float(jnp.max(jnp.abs(a_long))), float(jnp.max(jnp.abs(a_lat))), float(jnp.max(jnp.abs(j_long)))

        frames.append(dict(hx=np.array(hx), hy=np.array(hy), hv=np.array(hv),
                           px=np.array(pos[:,0]), py=np.array(pos[:,1]), sp=np.array(st[:,2]),
                           al=np.array(a_long), alat=np.array(a_lat), jl=np.array(j_long),
                           mo=ms, md=min_obs, ma=ma, ml=ml, mj=mj))

        ctrl_free = _shift(ctrl_free, V_TARGET, gen.dt)
        mu_init = jnp.stack([
            jnp.stack([ctrl_free[:, 0], ctrl_free[:, 0], ctrl_free[:, 0]], axis=0),
            jnp.stack([ctrl_free[:, 1], ctrl_free[:, 1], ctrl_free[:, 1]], axis=0),
        ], axis=0).astype(jnp.float32)

        print(f"step {step:3d} | x={rx:6.1f} y={ry:5.1f} v={rv:5.1f} | "
              f"a_long={ma:5.1f} a_lat={ml:5.1f} jerk={mj:5.1f} | "
              f"obs={min_obs:4.1f}m | {ms:5.0f}ms")

    if plot:
        np.savez(OUTPUT / "quintic_demo.npz", hx=np.array(hx), hy=np.array(hy), hv=np.array(hv),
                 frames=np.array(frames, dtype=object))
        def render(i):
            f = frames[i]; ax_t.cla(); ax_k.cla()
            ax_t.plot(f["hx"], f["hy"], "g-", lw=2, label="exec")
            ax_t.plot(f["px"], f["py"], "b--", lw=2, label="plan")
            for o in OBSTACLES:
                ax_t.add_patch(Circle((o["x"], o["y"]), o["r"], fc="r", alpha=.3))
                ax_t.add_patch(Circle((o["x"], o["y"]), o["r"]+OBS_SAFE_DIST, fc="none", ec="r", alpha=.3, ls="--"))
            ax_t.set_aspect("equal"); ax_t.legend(loc="upper left"); ax_t.grid(alpha=.25)
            ax_t.set_xlim(f["hx"][0]-10, f["px"].max()+20); ax_t.set_ylim(-6, 6)
            ax_t.set_title(f"step {i}  v={f['hv'][-1]:.1f}  obs={f['md']:.1f}m  {f['mo']:.0f}ms")
            t_ = np.arange(len(f["sp"])) * gen.dt
            ax_k.plot(t_, f["al"], label="a_long"); ax_k.plot(t_, f["alat"], label="a_lat")
            ax_k.plot(t_, f["jl"], label="jerk"); ax_k.legend(); ax_k.grid(alpha=.25)
            ax_k.set_title(f"max |a|={f['ma']:.1f}/{f['ml']:.1f}  jerk={f['mj']:.1f}")
        anim = FuncAnimation(fig, render, frames=len(frames), interval=80)
        anim.save(OUTPUT / "quintic_demo.gif", writer=PillowWriter(fps=12))
        plt.close(fig)
        print(f"saved {OUTPUT / 'quintic_demo.gif'}")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(steps=args.steps, seed=args.seed, plot=not args.no_plot)
