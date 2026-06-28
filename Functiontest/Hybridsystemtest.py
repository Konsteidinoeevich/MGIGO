"""
Hybrid System Receding-Horizon MPC — Mixed Terrain + Dynamic Interaction
=========================================================================

Real hybrid system: 2D vehicle on mixed terrain (normal road / ice patch).
Mode determined by POSITION — a real physical phenomenon.

Saturation nesting (3 levels, outer = more important):
  L(z) = saturate(
    viol_static > 0  ?  viol_static + 1.5 :           ← L1: static obstacles (certain)
    saturate(
      viol_chance > 0  ?  viol_chance + 1.5 :         ← L2: dynamic chance constraint
      saturate( f(z)/200 )                             ← L3: total running cost
  ))

Dynamic interaction: a pedestrian crosses the room. Its future position is
uncertain (Gaussian noise). Chance constraint via MC quantile:
  P( distance to pedestrian ≥ safe_margin ) ≥ 0.9

Reference: testMPC.py (MPC pattern), RobustConstraints.py (saturation nesting).
"""

import sys, time, math
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Circle
from jax import random, lax, jit, vmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc
from Constraintdealer.Constran import build, Deterministic, log_transform
from Constraintdealer.ObjectiveComposer import (
    compose_objective_dynamic, compose_objective_semantic,
    compose_objective_adaptive,
    calibrate_k_into_ctx, update_k_from_elite, knee_to_k
)


# ======================================================================
# 1. Configuration
# ======================================================================

CONFIG = {
    'dt': 0.15,
    'horizon': 14,
    'sim_steps': 70,
    'n_components': 3,
    'pop_size': 80,
    'elite_size': 35,
    'warmup_steps': 600,
    'opt_steps': 200,
    't0_reset': 100,
    'max_force': 4.0,
    'target': (8.5, 6.5),
}

# --- Environment: two ice patches, narrow corridors ---
ICE_PATCHES = [
    ((3.0, 3.5), 1.2),   # (center, radius) — large ice patch center-left
    ((7.0, 4.0), 0.9),   # smaller ice near target approach
]

# Dense pillars creating narrow passages
PILLARS = [
    (2.5, 1.5, 0.4), (5.0, 2.0, 0.45), (7.5, 1.5, 0.4),
    (2.0, 4.0, 0.4), (4.5, 4.5, 0.45), (7.0, 4.0, 0.4),
    (3.0, 6.0, 0.4), (6.0, 6.5, 0.45),
]
WALLS = [
    (0.0, 0.0, 9.5, 0.0),    # bottom
    (0.0, 8.0, 9.5, 8.0),    # top
    (0.0, 0.0, 0.0, 8.0),    # left
    (9.5, 0.0, 9.5, 3.0),    # right-lower
    (9.5, 5.0, 9.5, 8.0),    # right-upper (gap = passage to target)
]

# --- Two dynamic pedestrians crossing in different directions ---
PEDESTRIANS = [
    {'start': np.array([8.0, 2.0]), 'vel': np.array([-0.04, 0.06]), 'r': 0.35, 'sigma': 0.12},
]
CHANCE_PROB = 0.9
CHANCE_MC   = 30


# ======================================================================
# 2. Distance helpers (JAX JIT compatible)
# ======================================================================

@jit
def _dist_to_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    seg2 = dx*dx + dy*dy + 1e-12
    t = jnp.clip(((px-x1)*dx + (py-y1)*dy) / seg2, 0.0, 1.0)
    nx, ny = x1 + t*dx - px, y1 + t*dy - py
    return jnp.sqrt(nx*nx + ny*ny)

@jit
def _static_obstacle_violation(px, py):
    """Deterministic violation: walls + pillars."""
    viol = 0.0
    safe = 0.25
    for (x1, y1, x2, y2) in WALLS:
        d = _dist_to_segment(px, py, x1, y1, x2, y2)
        viol += jnp.maximum(0.0, safe - d)
    for (cx, cy, r) in PILLARS:
        d = jnp.sqrt((px-cx)**2 + (py-cy)**2) + 1e-12
        viol += jnp.maximum(0.0, (r + safe) - d)
    return viol


# ======================================================================
# 3. Hybrid Dynamics (position-dependent mode)
# ======================================================================

@jit
def _dynamics_normal(state, control, dt):
    px, py, vx, vy = state[0], state[1], state[2], state[3]
    fx, fy = control[0], control[1]
    mu = 1.0
    ax = fx - mu*vx;  ay = fy - mu*vy
    return jnp.array([px+vx*dt, py+vy*dt, vx+ax*dt, vy+ay*dt])

@jit
def _dynamics_ice(state, control, dt):
    px, py, vx, vy = state[0], state[1], state[2], state[3]
    fx, fy = control[0], control[1]
    mu = 0.1
    ax = fx - mu*vx;  ay = fy - mu*vy
    return jnp.array([px+vx*dt, py+vy*dt, vx+ax*dt, vy+ay*dt])

@jit
def _dynamics_gravel(state, control, dt):
    """Gravel: intermediate friction μ=0.8 + quadratic drag."""
    px, py, vx, vy = state[0], state[1], state[2], state[3]
    fx, fy = control[0], control[1]
    mu = 0.8
    v_mag = jnp.sqrt(vx**2 + vy**2) + 1e-6
    ax = fx - mu*vx - 0.3*vx*v_mag
    ay = fy - mu*vy - 0.3*vy*v_mag
    return jnp.array([px+vx*dt, py+vy*dt, vx+ax*dt, vy+ay*dt])

# 3rd terrain type
GRAVEL_PATCHES = [
    ((5.0, 6.0), 1.0),
]


# ======================================================================
# 4. Trajectory Rollout
# ======================================================================

def _rollout(theta_x, theta_y, init_state, dt, H):
    u_max = CONFIG['max_force']
    ice_cx = jnp.array([p[0][0] for p in ICE_PATCHES])
    ice_cy = jnp.array([p[0][1] for p in ICE_PATCHES])
    ice_r  = jnp.array([p[1] for p in ICE_PATCHES])
    grav_cx = jnp.array([p[0][0] for p in GRAVEL_PATCHES])
    grav_cy = jnp.array([p[0][1] for p in GRAVEL_PATCHES])
    grav_r  = jnp.array([p[1] for p in GRAVEL_PATCHES])

    def step_fn(carry, _):
        state, t = carry
        # Mode selection by position: normal(0) / ice(1) / gravel(2)
        d2_ice = (state[0]-ice_cx)**2 + (state[1]-ice_cy)**2
        d2_grav = (state[0]-grav_cx)**2 + (state[1]-grav_cy)**2
        in_ice = jnp.any(d2_ice < ice_r**2)
        in_gravel = jnp.any(d2_grav < grav_r**2)
        mode_idx = jnp.where(in_ice, 1, jnp.where(in_gravel, 2, 0))
        control = u_max * jnp.tanh(jnp.array([theta_x[t], theta_y[t]]))
        ns = lax.switch(
            mode_idx,
            [lambda s,c: _dynamics_normal(s,c,dt),
             lambda s,c: _dynamics_ice(s,c,dt),
             lambda s,c: _dynamics_gravel(s,c,dt)],
            state, control)
        return (ns, t+1), (ns, mode_idx)

    (final_state, _), (traj, mode_seq) = lax.scan(
        step_fn, (init_state, 0), xs=None, length=H)
    return traj, final_state, mode_seq


# ======================================================================
# 5. MPC Cost Function — 3-Level Saturation Nesting
# ======================================================================

@jit
def saturate(x, k=1.0):
    """Smooth saturation: k·x / √(1 + (k·x)²)

    - Output always in (0, 1) regardless of k
    - k controls the "knee": linear region ≈ x ∈ [0, 1/k]
      · small k → wide linear range, good for large violations (e.g. L1: k=0.05)
      · large k → narrow linear range, saturated early (e.g. L3: k=0.5)
    - Typical k by level: L1=0.05~0.1, L2=0.2~0.3, L3=0.5~1.0
    """
    kx = k * x
    return kx / jnp.sqrt(1.0 + kx**2)


# ======================================================================
# 5a. Original version: jnp.where branching (Hybrid 原版硬约束)
# ======================================================================
@jit
def mpc_cost_fn_where(z_flat, ctx):
    """Hybrid MPC cost — 3-level saturation nesting with jnp.where.

    L1 (outermost): static deterministic obstacle violation
    L2 (middle):    dynamic chance constraints
    L3 (innermost): total running cost f(z) summed over horizon

    Each level uses jnp.where to branch: violation → offset + gradient;
    no violation → pass through to inner level. Offsets (δ₁=1.2, δ₂=0.8)
    create strict discontinuity at viol=0, guaranteeing ANY L1 > ANY L2 > ANY L3.
    """
    (init_state, target, dt_val, _,
     ped1_traj, ped1_r, ped1_sigma, mc_key) = ctx
    H = z_flat.shape[0] // 2
    u_max = CONFIG['max_force']

    theta_x = z_flat[:H]
    theta_y = z_flat[H:2*H]

    traj, final_state, _ = _rollout(theta_x, theta_y, init_state, dt_val, H)
    positions = traj[:, :2]

    # L3 (innermost): total running cost
    dists = jnp.sum((positions - target[None, :])**2, axis=1)
    tracking_cost = jnp.sum(dists)
    final_cost = 15.0 * jnp.sum((final_state[:2] - target)**2)
    fx_a = u_max * jnp.tanh(theta_x)
    fy_a = u_max * jnp.tanh(theta_y)
    control_cost = 0.1 * (jnp.sum(fx_a**2) + jnp.sum(fy_a**2))
    f_total = tracking_cost + final_cost + control_cost

    # L2 (middle): dynamic chance constraints
    def ped_violation(ped_traj, ped_r, ped_sigma):
        safe_margin = 0.05
        ped_safe_dist = ped_r + safe_margin
        diff_to_ped = positions - ped_traj
        M = CHANCE_MC
        noise = random.normal(mc_key, (M, 2)) * ped_sigma
        def noisy_dist(ns):
            return jnp.sqrt(jnp.sum((diff_to_ped - ns[None, :])**2, axis=-1) + 1e-12)
        dist_noisy = vmap(noisy_dist)(noise)
        q_dist = jnp.quantile(dist_noisy, 1.0 - CHANCE_PROB, axis=0)
        return jnp.sum(jnp.where(ped_safe_dist - q_dist > 0.0, 1.5 + ped_safe_dist - q_dist, 0.0))

    viol_dynamic = ped_violation(ped1_traj, ped1_r, ped1_sigma)

    # L1 (outermost): static constraints
    viol_static = jnp.sum(vmap(
        lambda p: _static_obstacle_violation(p[0], p[1]))(positions))

    viol_bound = jnp.sum(
        jnp.where(positions[:, 0] < 0.0, 1.5 - positions[:, 0], 0.0) +
        jnp.where(positions[:, 0] > 9.5, 1.5 + positions[:, 0] - 9.5, 0.0) +
        jnp.where(positions[:, 1] < 0.0, 1.5 - positions[:, 1], 0.0) +
        jnp.where(positions[:, 1] > 8.0, 1.5 + positions[:, 1] - 8.0, 0.0))
    viol_static = viol_static + viol_bound

    # Auto-scaling
    dist_sq = jnp.sum((init_state[:2] - target)**2)
    L3_SCALE = (H + 15.0) * dist_sq + 0.1

    L2_SCALE = H * (1.5 + ped1_r + 0.05)

    _L1_PER_STEP = sum(p[2] + 0.25 for p in PILLARS) + len(WALLS) * 0.25 + 4 * 1.5
    L1_SCALE = H * _L1_PER_STEP

    # 3-Level nesting with jnp.where — strict hierarchy via discontinuity
    res = saturate(
        jnp.where(viol_static > 0.0, viol_static / L1_SCALE + 1.2,
        saturate(
            jnp.where(viol_dynamic > 0.0, viol_dynamic / L2_SCALE + 0.8,
            saturate(f_total / L3_SCALE)
        ))
    ))

    return res


# ======================================================================
# 5b. Smooth version: nested saturate WITHOUT jnp.where
#
#     Form: σ( v₁/S₁ + δ₁·σ(β₁·v₁) + σ( v₂/S₂ + δ₂·σ(β₂·v₂) + σ( f/S₃ )))
#
#     σ(x)          = saturate(x) = x/√(1+x²)       ← always k=1
#     σ(β·viol)     ≈ 𝕀_{viol>0}  smooth step       ← β controls hardness
#     δ₁, δ₂         = jump magnitude per level
#     β₁, β₂         = hardness: β→∞ → hard step     ← tunable knob
#
#     No jnp.where — all transitions are continuous.
#     β controls the hardness: β=100 → nearly binary jump at viol=0.
#     Nested saturates preserve per-level knees for gradient.
#
#     Hierarchy (δ₁=2.0, δ₂=0.8, β large):
#       L1 active:  [σ(2.0), 1.0) = [0.894, 1.0)
#       L2 only:    [σ(0.8), σ(0.8+σ(∞))) = [0.625, 0.833)
#       L3 only:    [0, σ(σ(∞))) = [0, 0.577)
#       min(L1)=0.894 > max(L2)=0.833 ✓  (δ₁ tuned larger for smooth overlap)
# ======================================================================
@jit
def mpc_cost_fn_smooth(z_flat, ctx):
    (init_state, target, dt_val, _,
     ped1_traj, ped1_r, ped1_sigma, mc_key) = ctx
    H = z_flat.shape[0] // 2
    u_max = CONFIG['max_force']

    theta_x = z_flat[:H]
    theta_y = z_flat[H:2*H]

    traj, final_state, _ = _rollout(theta_x, theta_y, init_state, dt_val, H)
    positions = traj[:, :2]

    # ---- L3 raw ----
    dists = jnp.sum((positions - target[None, :])**2, axis=1)
    tracking_cost = jnp.sum(dists)
    final_cost = 15.0 * jnp.sum((final_state[:2] - target)**2)
    fx_a = u_max * jnp.tanh(theta_x)
    fy_a = u_max * jnp.tanh(theta_y)
    control_cost = 0.1 * (jnp.sum(fx_a**2) + jnp.sum(fy_a**2))
    f_total = tracking_cost + final_cost + control_cost

    # ---- L2 raw ----
    def ped_violation(ped_traj, ped_r, ped_sigma):
        safe_margin = 0.05
        ped_safe_dist = ped_r + safe_margin
        diff_to_ped = positions - ped_traj
        M = CHANCE_MC
        noise = random.normal(mc_key, (M, 2)) * ped_sigma
        def noisy_dist(ns):
            return jnp.sqrt(jnp.sum((diff_to_ped - ns[None, :])**2, axis=-1) + 1e-12)
        dist_noisy = vmap(noisy_dist)(noise)
        q_dist = jnp.quantile(dist_noisy, 1.0 - CHANCE_PROB, axis=0)
        return jnp.sum(jnp.where(ped_safe_dist - q_dist > 0.0, 1.5 + ped_safe_dist - q_dist, 0.0))

    viol_dynamic = ped_violation(ped1_traj, ped1_r, ped1_sigma)

    # ---- L1 raw ----
    viol_static = jnp.sum(vmap(
        lambda p: _static_obstacle_violation(p[0], p[1]))(positions))

    viol_bound = jnp.sum(
        jnp.where(positions[:, 0] < 0.0, 1.5 - positions[:, 0], 0.0) +
        jnp.where(positions[:, 0] > 9.5, 1.5 + positions[:, 0] - 9.5, 0.0) +
        jnp.where(positions[:, 1] < 0.0, 1.5 - positions[:, 1], 0.0) +
        jnp.where(positions[:, 1] > 8.0, 1.5 + positions[:, 1] - 8.0, 0.0))
    viol_static = viol_static + viol_bound

    # ---- Auto-scaling (same as where version) ----
    dist_sq = jnp.sum((init_state[:2] - target)**2)
    L3_SCALE = (H + 15.0) * dist_sq + 0.1
    L2_SCALE = H * (1.5 + ped1_r + 0.05)
    _L1_PER_STEP = sum(p[2] + 0.25 for p in PILLARS) + len(WALLS) * 0.25 + 4 * 1.5
    L1_SCALE = H * _L1_PER_STEP

    # ---- Hardness knobs ----
    beta_1 = 2000.0  # static obstacles: 阶跃在 ~0.0005m 内完成 (极硬)
    beta_2 = 500.0   # dynamic pedestrian: 阶跃在 ~0.002m 内完成 (硬)
    delta_1 = 2.0    # L1 jump
    delta_2 = 0.8    # L2 jump

    # ---- 3-Level nested saturation WITHOUT jnp.where ----
    # σ(β·viol) ≈ 0 when viol=0, ≈ 1 when viol ≫ 1/β  (smooth indicator)
    # Each level: viol/SCALE (梯度) + δ·σ(β·viol) (阶跃) + saturate(下层)
    res = saturate(
        viol_static / L1_SCALE + delta_1 * saturate(beta_1 * viol_static) + saturate(
            viol_dynamic / L2_SCALE + delta_2 * saturate(beta_2 * viol_dynamic) + saturate(
                f_total / L3_SCALE
            )
        )
    )

    return res


# ======================================================================
# 5c. Constran version: build() from Constraintdealer.Constran
#
#     Define objective + constraints as independent functions, then let
#     Constran.build() handle log-transform, saturation nesting, and δ
#     auto-assignment. A shared rollout wrapper avoids redundant _rollout
#     calls — the trajectory is computed once and passed via extended ctx.
#
#     Advantages over manual 5a/5b:
#       - Declarative: specify WHAT (objective, constraints) not HOW (nesting)
#       - Auto δ: outermost hard → 1.5, inner hard → 3.0 (matches README)
#       - Reusable: same pattern works for any M-layer problem
#       - Solver-agnostic: output is a standard (x, ctx) -> scalar
#
#     Mapping:
#       L1 (outermost) ← Deterministic(_static_violation, mode='hard', priority=1)
#       L2 (middle)     ← Deterministic(_dynamic_violation, mode='hard', priority=2)
#       L3 (innermost)  ← _tracking_objective  (passed as objective_fn to build())
# ======================================================================

def _tracking_objective(z_flat, ctx):
    """L3 (innermost): tracking + final + control cost.

    Receives pre-computed trajectory in ctx['traj'] / ctx['final_state'].
    """
    traj = ctx['traj']
    final_state = ctx['final_state']
    target = ctx['target']
    H = z_flat.shape[0] // 2
    u_max = CONFIG['max_force']

    positions = traj[:, :2]
    theta_x = z_flat[:H]
    theta_y = z_flat[H:2*H]

    dists = jnp.sum((positions - target[None, :]) ** 2, axis=1)
    tracking_cost = jnp.sum(dists)
    final_cost = 15.0 * jnp.sum((final_state[:2] - target) ** 2)
    fx_a = u_max * jnp.tanh(theta_x)
    fy_a = u_max * jnp.tanh(theta_y)
    control_cost = 0.1 * (jnp.sum(fx_a ** 2) + jnp.sum(fy_a ** 2))
    return tracking_cost + final_cost + control_cost


def _static_violation(z_flat, ctx):
    """L1 (outermost): static obstacle violation (walls + pillars + bounds).

    Returns raw violation > 0 when any position penetrates an obstacle.
    Constran applies log_transform + δ + σ nesting automatically.
    """
    traj = ctx['traj']
    positions = traj[:, :2]

    viol = jnp.sum(vmap(
        lambda p: _static_obstacle_violation(p[0], p[1]))(positions))
    viol_bound = jnp.sum(
        jnp.where(positions[:, 0] < 0.0, 1.5 - positions[:, 0], 0.0) +
        jnp.where(positions[:, 0] > 9.5, 1.5 + positions[:, 0] - 9.5, 0.0) +
        jnp.where(positions[:, 1] < 0.0, 1.5 - positions[:, 1], 0.0) +
        jnp.where(positions[:, 1] > 8.0, 1.5 + positions[:, 1] - 8.0, 0.0))
    return viol + viol_bound


def _dynamic_violation(z_flat, ctx):
    """L2 (middle): dynamic pedestrian chance constraint.

    Computes the MC quantile internally (like the original code) and returns
    a scalar raw violation.  > 0 means the chance constraint is violated.
    We use Deterministic (not Chance) because the MC logic is self-contained;
    Constran only sees the final scalar and handles nesting.
    """
    traj = ctx['traj']
    ped_traj = ctx['ped_traj']
    ped_r = ctx['ped_r']
    ped_sigma = ctx['ped_sigma']
    mc_key = ctx['mc_key']
    positions = traj[:, :2]

    safe_margin = 0.05
    ped_safe_dist = ped_r + safe_margin
    diff_to_ped = positions - ped_traj
    M = CHANCE_MC
    noise = random.normal(mc_key, (M, 2)) * ped_sigma

    def noisy_dist(ns):
        return jnp.sqrt(jnp.sum((diff_to_ped - ns[None, :]) ** 2, axis=-1) + 1e-12)

    dist_noisy = vmap(noisy_dist)(noise)
    q_dist = jnp.quantile(dist_noisy, 1.0 - CHANCE_PROB, axis=0)
    return jnp.sum(jnp.where(
        ped_safe_dist - q_dist > 0.0, 1.5 + ped_safe_dist - q_dist, 0.0))


# Build the nested cost via Constran (no jit — we wrap with rollout + jit below)
_base_hybrid_cost = build(
    _tracking_objective,
    [
        Deterministic(_static_violation, mode='hard', priority=1),
        Deterministic(_dynamic_violation, mode='hard', priority=2),
    ],
    jit_cost=False,
)


@jit
def mpc_cost_fn_constran(z_flat, ctx):
    """Constran-wrapped hybrid MPC cost — single rollout, modular constraints.

    This is the solver-ready cost function. It:
    1. Unpacks the solver context (same format as where/smooth versions).
    2. Runs _rollout ONCE to get the trajectory.
    3. Passes trajectory data via an extended context dict to the Constran-built
       nested cost, which calls _tracking_objective, _static_violation, and
       _dynamic_violation independently.
    4. Constran handles log_transform, σ nesting, and δ offsets automatically.

    The solver sees this as a black box (z_flat, ctx) -> scalar — exactly the
    same interface as mpc_cost_fn_where and mpc_cost_fn_smooth.
    """
    (init_state, target, dt_val, _,
     ped1_traj, ped1_r, ped1_sigma, mc_key) = ctx
    H = z_flat.shape[0] // 2

    theta_x = z_flat[:H]
    theta_y = z_flat[H:2 * H]
    traj, final_state, _ = _rollout(theta_x, theta_y, init_state, dt_val, H)

    ext_ctx = {
        'traj': traj,
        'final_state': final_state,
        'target': target,
        'ped_traj': ped1_traj,
        'ped_r': ped1_r,
        'ped_sigma': ped1_sigma,
        'mc_key': mc_key,
    }
    return _base_hybrid_cost(z_flat, ext_ctx)


# ======================================================================
# 5d. Composed mode: dynamic k objective + constraint build
#
#     Objective sub-terms get per-term dynamic k (auto-calibrated from elite),
#     constraints get declarative build().  Zero manual weights.
# ======================================================================

# --- Sub-term functions (read trajectory from ctx['traj']) ---
def _tracking_term(z_flat, ctx):
    positions = ctx['traj'][:, :2]
    target = ctx['target']
    return jnp.sum(jnp.sum((positions - target)**2, axis=1))

def _final_term(z_flat, ctx):
    final_state = ctx['final_state']
    target = ctx['target']
    return 15.0 * jnp.sum((final_state[:2] - target)**2)

def _control_term(z_flat, ctx):
    H = z_flat.shape[0] // 2
    u_max = CONFIG['max_force']
    fx = u_max * jnp.tanh(z_flat[:H])
    fy = u_max * jnp.tanh(z_flat[H:2*H])
    return 0.1 * (jnp.sum(fx**2) + jnp.sum(fy**2))

# --- Calibration-only term functions (self-contained, do their own rollout) ---
def _tracking_calib(z_flat, ctx):
    H = z_flat.shape[0] // 2
    traj, _, _ = _rollout(z_flat[:H], z_flat[H:2*H],
                          ctx['init_state'], jnp.array(CONFIG['dt']), H)
    return jnp.sum(jnp.sum((traj[:,:2] - ctx['target'])**2, axis=1))

def _final_calib(z_flat, ctx):
    H = z_flat.shape[0] // 2
    _, fs, _ = _rollout(z_flat[:H], z_flat[H:2*H],
                        ctx['init_state'], jnp.array(CONFIG['dt']), H)
    return 15.0 * jnp.sum((fs[:2] - ctx['target'])**2)

def _control_calib(z_flat, ctx):
    H = z_flat.shape[0] // 2
    u_max = CONFIG['max_force']
    return 0.1 * (jnp.sum((u_max * jnp.tanh(z_flat[:H]))**2) +
                  jnp.sum((u_max * jnp.tanh(z_flat[H:2*H]))**2))

# Dynamic k objective — raw_output=True: no outer σ, build() handles compression
_composed_obj = compose_objective_dynamic([
    (_tracking_term, 0.0, 'tracking'),
    (_final_term,    0.0, 'final'),
    (_control_term,  0.0, 'control'),
], raw_output=True)

# Initial k: physically meaningful knees, then per-step elite updates refine them
_composed_ctx = {'target': jnp.array(CONFIG['target']),
                 'init_state': jnp.array([0.5, 0.5, 0.0, 0.0])}
_composed_ctx['k_tracking'] = float(knee_to_k(20.0))   # knee at ~20 tracking error
_composed_ctx['k_final']    = float(knee_to_k(3.0))    # knee at ~3 final error
_composed_ctx['k_control']  = float(knee_to_k(1.5))    # knee at ~1.5 control cost

# Constraints use build() with large k_inner — composed objective is already
# per-term compressed, so the inner sigma_k should be nearly linear (knee wide).
_composed_cost_base = build(
    lambda z, ctx: _composed_obj(z, ctx),
    [
        Deterministic(_static_violation, mode='hard', priority=1),
        Deterministic(_dynamic_violation, mode='hard', priority=2),
    ],
    k_inner=1.0,   # transparent for pre-compressed objective (sum in [0.5, 3])
    jit_cost=False,
)

@jit
def mpc_cost_fn_composed(z_flat, ctx):
    (init_state, target, dt_val, _,
     ped1_traj, ped1_r, ped1_sigma, mc_key) = ctx
    H = z_flat.shape[0] // 2
    theta_x = z_flat[:H]
    theta_y = z_flat[H:2*H]
    traj, final_state, _ = _rollout(theta_x, theta_y, init_state, dt_val, H)

    ext_ctx = {
        'traj': traj, 'final_state': final_state, 'target': target,
        'ped_traj': ped1_traj, 'ped_r': ped1_r, 'ped_sigma': ped1_sigma,
        'mc_key': mc_key,
    }
    for key in ['k_tracking', 'k_final', 'k_control']:
        ext_ctx[key] = _composed_ctx[key]
    return _composed_cost_base(z_flat, ext_ctx)


# ======================================================================
# 5e. Semantic mode: log_transform per term + build() for constraints
#
#     Objective: log_transform per term → sum (no σ, no floor)
#     Constraints: build() with mode='hard' — proven nesting structure
#
#     log_transform preserves magnitude ordering without a floor:
#       far:  log(1000) dominates log(2)    → "go to target"
#       near: log(10) dominates log(0.01)   → "stay efficient"
#     Natural transition.  No weights.  No normalization.
# ======================================================================

def _semantic_objective(z_flat, ctx):
    """Auto-derived structural factors + log_transform per term → sum.

    Factors derived from MPC structure (not tuned):
      - final = H (one-step error counts as much as whole horizon)
      - control = 1 / (H * max_force²) (normalize to position-error scale)
    log_transform = sign(x)·log(1+|x|) handles magnitude compression.
    """
    traj = ctx['traj']
    final_state = ctx['final_state']
    target = ctx['target']
    H = z_flat.shape[0] // 2
    u_max = CONFIG['max_force']
    positions = traj[:, :2]

    tracking_raw = jnp.sum(jnp.sum((positions - target[None, :]) ** 2, axis=1))
    final_raw = float(H) * jnp.sum((final_state[:2] - target) ** 2)
    fx = u_max * jnp.tanh(z_flat[:H])
    fy = u_max * jnp.tanh(z_flat[H:2*H])
    control_raw = (1.0 / (H * u_max**2)) * (jnp.sum(fx ** 2) + jnp.sum(fy ** 2))

    return (log_transform(tracking_raw) +
            log_transform(final_raw) +
            log_transform(control_raw))


# build() handles constraint nesting, T_alpha compression, and σ layering
_semantic_cost_base = build(
    _semantic_objective,
    [
        Deterministic(_static_violation, mode='hard', priority=1),
        Deterministic(_dynamic_violation, mode='hard', priority=2),
    ],
    jit_cost=False,
)


@jit
def mpc_cost_fn_semantic(z_flat, ctx):
    (init_state, target, dt_val, _,
     ped1_traj, ped1_r, ped1_sigma, mc_key) = ctx
    H = z_flat.shape[0] // 2
    theta_x = z_flat[:H]
    theta_y = z_flat[H:2 * H]
    traj, final_state, _ = _rollout(theta_x, theta_y, init_state, dt_val, H)

    ext_ctx = {
        'traj': traj, 'final_state': final_state, 'target': target,
        'ped_traj': ped1_traj, 'ped_r': ped1_r, 'ped_sigma': ped1_sigma,
        'mc_key': mc_key,
    }
    return _semantic_cost_base(z_flat, ext_ctx)


# Default: use the original jnp.where version
mpc_cost_fn = mpc_cost_fn_where


# ======================================================================
# 6. Solver Wrapper
# ======================================================================

def solver(key, T, dt, M, K, B, B0, dims, T_0, cost, mu, L_inv, v, context):
    """Wrapper around mmog_igo_optimizer_mpc."""
    final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
        key, T, dt, M, K, B, B0, dims, T_0,
        fitness_fn_total=cost, initial_mu_k=mu,
        initial_L_inv_k=L_inv, initial_v_k=v, context=context)
    # MPCsolverM22 returns 3 values; compute v from pi for warm start
    final_v = jnp.log(jnp.clip(final_pi[:, :-1], 1e-20, None)
                      / jnp.clip(final_pi[:, -1:], 1e-20, None))
    return final_mu, final_L, final_pi, final_v


# ======================================================================
# 7. Warm Start
# ======================================================================

def shift_solution_with_diversity(mu, K, H, key):
    """Shift mu by 1 step + random tail — exactly like testMPC.py.
    Only mu is touched; L_inv and pi/v stay as returned by solver."""
    mu_shifted = jnp.roll(mu, shift=-1, axis=-1)
    key, sk1, sk2 = jax.random.split(key, 3)
    mu_shifted = mu_shifted.at[0, :, -1].set(
        jax.random.uniform(sk1, (K,), minval=-1.0, maxval=1.0))
    mu_shifted = mu_shifted.at[1, :, -1].set(
        jax.random.uniform(sk2, (K,), minval=-1.0, maxval=1.0))
    return mu_shifted, key


# ======================================================================
# 8. Receding-Horizon MPC
# ======================================================================

def run_hybrid_mpc(cost_mode='where'):
    """Run hybrid MPC simulation.

    cost_mode: 'where'    → original jnp.where branching (strict hard constraints)
               'smooth'   → nested saturate with smooth step (no jnp.where)
               'constran' → Constraintdealer.Constran.build() — declarative,
                             modular objective + constraints, auto δ assignment
               'composed' → dynamic k objective + build() constraints —
                             zero manual weights, semantic roles only
    """
    # Select cost function
    if cost_mode == 'semantic':
        cost_fn = mpc_cost_fn_semantic
        mode_label = 'SEMANTIC (compose_objective_semantic — T_alpha + fixed k, zero params)'
    elif cost_mode == 'composed':
        cost_fn = mpc_cost_fn_composed
        mode_label = 'COMPOSED (dynamic k + semantic roles)'
    elif cost_mode == 'constran':
        cost_fn = mpc_cost_fn_constran
        mode_label = 'CONSTRAN (declarative, build() from Constraintdealer)'
    elif cost_mode == 'smooth':
        cost_fn = mpc_cost_fn_smooth
        mode_label = 'SMOOTH (nested σ, no jnp.where)'
    else:
        cost_fn = mpc_cost_fn_where
        mode_label = 'WHERE (jnp.where branching)'

    H = CONFIG['horizon'];  sim_steps = CONFIG['sim_steps']
    K = CONFIG['n_components'];  B = CONFIG['pop_size']
    B0 = CONFIG['elite_size'];  dt_val = CONFIG['dt']
    u_max = CONFIG['max_force']
    dims = (H, H);  M = 2;  d_max = H

    key = random.PRNGKey(42)
    key, key_init = random.split(key)

    # Init solver
    initial_mu = jnp.zeros((M, K, d_max))
    for comp in range(K):
        initial_mu = initial_mu.at[0, comp, :].set(
            random.uniform(random.fold_in(key_init, comp), (d_max,), minval=-1.0, maxval=1.0))
        initial_mu = initial_mu.at[1, comp, :].set(
            random.uniform(random.fold_in(key_init, comp+100), (d_max,), minval=-1.0, maxval=1.0))
    initial_L_inv = jnp.tile(jnp.eye(d_max, dtype=jnp.float32)[None,None,:,:], (M,K,1,1)) * 1.0
    initial_v = jnp.zeros((M, K-1), dtype=jnp.float32)

    # Init simulation
    state = jnp.array([0.5, 0.5, 0.0, 0.0])
    target = jnp.array(CONFIG['target'])
    mu_k, L_inv_k, v_k = initial_mu, initial_L_inv, initial_v

    # Dynamic obstacles state
    ped_pos = [np.array(p['start'], dtype=np.float64) for p in PEDESTRIANS]
    ped_hist = [[p.copy()] for p in ped_pos]

    traj_hist = [np.array(state)]
    mode_hist, ctrl_hist, time_hist = [], [], []

    n_ped = len(PEDESTRIANS)
    print(f"{'='*60}")
    print(f"Hybrid MPC: Vehicle on Mixed Terrain + {n_ped} Dynamic Pedestrians")
    print(f"Cost mode: {mode_label}")
    print(f"Horizon H={H} | Steps={sim_steps} | u_max={u_max}")
    print(f"Static obstacles: {len(PILLARS)} pillars + {len(WALLS)} walls")
    print(f"Ice patches: {len(ICE_PATCHES)}")
    print(f"Dynamic obstacles: {n_ped} pedestrians with chance constraint P≥{CHANCE_PROB}")
    for i, p in enumerate(PEDESTRIANS):
        print(f"  Ped {i}: start={p['start']}, vel={p['vel']}, r={p['r']}, σ={p['sigma']}")
    print(f"Target: {CONFIG['target']} | Start: (0.5, 0.5)")
    print(f"Saturation: L1(static) > L2(dynamic chance) > L3(run cost)")
    print(f"{'='*60}")

    for step in range(sim_steps):
        key, key_solver, key_mc = random.split(key, 3)

        # Predict pedestrian trajectories for next H steps
        ped_trajs = [jnp.array([pos + p['vel'] * (t + 1) for t in range(H)])
                     for pos, p in zip(ped_pos, PEDESTRIANS)]

        ctx = (state, target, jnp.array(dt_val), jnp.array(H),
               ped_trajs[0], jnp.array(PEDESTRIANS[0]['r']), jnp.array(PEDESTRIANS[0]['sigma']),
               key_mc)

        iters = CONFIG['warmup_steps'] if step == 0 else CONFIG['opt_steps']

        t0 = time.perf_counter()
        final_mu, final_L, final_pi, final_v = solver(
            key_solver, iters, dt_val, M, K, B, B0, dims, CONFIG['t0_reset'],
            cost_fn, mu_k, L_inv_k, v_k, ctx)
        final_mu.block_until_ready()
        elapsed = time.perf_counter() - t0
        time_hist.append(elapsed)

        # Extract & execute
        bi = [int(jnp.argmax(final_pi[0])), int(jnp.argmax(final_pi[1]))]
        tx = np.array(final_mu[0, bi[0], :H])
        ty = np.array(final_mu[1, bi[1], :H])

        # Dynamic k update: recalibrate from elite for next step
        if cost_mode == 'composed':
            best_z = jnp.concatenate([final_mu[0, bi[0], :H],
                                       final_mu[1, bi[1], :H]])
            _composed_ctx['init_state'] = state  # current vehicle state
            _composed_ctx.update(
                update_k_from_elite(dict(_composed_ctx),
                    [(_tracking_calib, 0.0, 'tracking'),
                     (_final_calib,    0.0, 'final'),
                     (_control_calib,  0.0, 'control')],
                    best_z,
                    multiplier=3.0, min_knee=0.1,
                    verbose=(step < 5 or step % 20 == 0)))

        u_exec = (jnp.array([0.0, 0.0]) if np.any(~np.isfinite(tx)) or np.any(~np.isfinite(ty))
                  else u_max * jnp.tanh(jnp.array([tx[0], ty[0]])))
        ctrl_hist.append(np.array(u_exec))

        # Advance vehicle with 3-mode hybrid dynamics
        px, py = state[0], state[1]
        in_ice = any((px-cx)**2 + (py-cy)**2 < r**2 for (cx,cy),r in ICE_PATCHES)
        in_grav = any((px-cx)**2 + (py-cy)**2 < r**2 for (cx,cy),r in GRAVEL_PATCHES)
        mode_idx = 1 if in_ice else (2 if in_grav else 0)
        ns = lax.switch(
            mode_idx,
            [lambda s,c: _dynamics_normal(s,c,dt_val),
             lambda s,c: _dynamics_ice(s,c,dt_val),
             lambda s,c: _dynamics_gravel(s,c,dt_val)],
            state, u_exec)

        mode_hist.append(mode_idx)
        traj_hist.append(np.array(ns))

        # Advance pedestrians (true dynamics with small random walk)
        for i in range(n_ped):
            ped_pos[i] = ped_pos[i] + PEDESTRIANS[i]['vel'] + np.random.normal(0, 0.02, 2)
            ped_pos[i] = np.clip(ped_pos[i], [0.3, 0.3], [9.0, 7.5])
            ped_hist[i].append(ped_pos[i].copy())

        # Warm start: shift mu (carry forward), reset L_inv & v to initial
        # Resetting L_inv (precision) and v (mixture weights) ensures the solver
        # starts each step with wide exploration rather than trapped in narrow modes.
        mu_k, key = shift_solution_with_diversity(final_mu, K, H, key)
        L_inv_k, v_k = initial_L_inv, initial_v
        state = ns

        # Print
        pos = np.array(state[:2])
        dist = np.sqrt(np.sum((pos - np.array(target))**2))
        dist_ped = min(np.sqrt(np.sum((pos - p)**2)) for p in ped_pos)
        mode_str = ["NORM", "ICE ", "GRAV"][mode_idx]
        print(f" Step{step:3d} | {mode_str:4s} | "
              f"u=({u_exec[0]:+.2f},{u_exec[1]:+.2f}) | "
              f"p=({pos[0]:.2f},{pos[1]:.2f}) | "
              f"min_dPed={dist_ped:.2f} | "
              f"T={dist:.3f} | {elapsed:.2f}s")

    # Summary
    traj = np.array(traj_hist)
    ctrl = np.array(ctrl_hist)
    modes = np.array(mode_hist)
    ped_arrs = [np.array(h) for h in ped_hist]
    total = sum(time_hist)

    # Min distance to any pedestrian
    min_dists = []
    for t in range(len(traj)):
        d = min(np.sqrt(np.sum((traj[t, :2] - p[t, :2])**2)) for p in ped_arrs)
        min_dists.append(d)
    min_d = np.min(min_dists)

    print(f"\n{'='*60}")
    print(f"Summary: {sim_steps} steps, {total:.1f}s "
          f"(warmup={time_hist[0]:.1f}s, reopt={np.mean(time_hist[1:]):.2f}s)")
    print(f"Final pos=({traj[-1,0]:.3f},{traj[-1,1]:.3f}) "
          f"dist_to_target={np.sqrt(np.sum((traj[-1,:2]-np.array(target))**2)):.3f}")
    print(f"Ice steps: {np.sum(modes)}/{len(modes)}")
    print(f"Min distance to any pedestrian: {min_d:.3f}")
    print(f"{'='*60}")

    _plot(traj, ctrl, modes, target, ped_arrs)
    return traj, ctrl, modes, ped_arrs


# ======================================================================
# 9. Plot
# ======================================================================

def _plot(traj, ctrl, modes, target, ped_arrs):
    out = Path("Hybrid_test"); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    T = len(modes)
    ped_colors = ['orange', 'magenta']

    # --- Position ---
    ax = axes[0, 0]
    mode_colors = {0: 'tab:blue', 1: 'tab:red', 2: 'tab:green'}
    colors = [mode_colors[m] for m in modes]
    ax.scatter(traj[1:, 0], traj[1:, 1], c=colors, s=25, zorder=3)
    ax.plot(traj[:, 0], traj[:, 1], 'gray', alpha=0.3, zorder=2)
    # Pedestrians
    for i, ped_traj in enumerate(ped_arrs):
        ax.plot(ped_traj[:, 0], ped_traj[:, 1], ped_colors[i], linewidth=2, alpha=0.7, zorder=2)
        ax.scatter(ped_traj[::8, 0], ped_traj[::8, 1], c=ped_colors[i], s=20, marker='s', zorder=3)
    # Markers
    ax.scatter([traj[0,0]], [traj[0,1]], c='green', s=120, marker='o',
               zorder=4, label='Start', edgecolors='black')
    ax.scatter([target[0]], [target[1]], c='gold', s=180, marker='*',
               zorder=5, label='Target', edgecolors='black')
    # Ice patches
    for (cx, cy), r in ICE_PATCHES:
        ax.add_patch(plt.Circle((cx, cy), r, color='cyan', alpha=0.2, zorder=1))
    ax.add_patch(plt.Circle(ICE_PATCHES[0][0], ICE_PATCHES[0][1], color='cyan', alpha=0.2, zorder=1, label='Ice (μ=0.1)'))
    # Gravel patches
    for (cx, cy), r in GRAVEL_PATCHES:
        ax.add_patch(plt.Circle((cx, cy), r, color='brown', alpha=0.2, zorder=1))
    if GRAVEL_PATCHES:
        ax.add_patch(plt.Circle(GRAVEL_PATCHES[0][0], GRAVEL_PATCHES[0][1], color='brown', alpha=0.2, zorder=1, label='Gravel (μ=0.8)'))
    # Obstacles
    for (cx, cy, r) in PILLARS:
        ax.add_patch(plt.Circle((cx, cy), r, color='red', alpha=0.5, zorder=1))
    for (x1,y1,x2,y2) in WALLS:
        ax.plot([x1,x2], [y1,y2], 'k-', linewidth=3, alpha=0.6, zorder=1)
    # Uncertainty circles
    for i, ped_traj in enumerate(ped_arrs):
        ax.add_patch(plt.Circle(ped_traj[-1], PEDESTRIANS[i]['sigma']*2,
                                color=ped_colors[i], alpha=0.12, zorder=1))
    ax.set_xlim(-0.5, 10); ax.set_ylim(-0.5, 8.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Position (blue=normal, red=ice, green=gravel)")
    ax.legend(loc='upper left', fontsize=7); ax.grid(alpha=0.3); ax.set_aspect('equal')

    # --- Velocity ---
    ax = axes[0, 1]
    ax.plot(traj[:, 2], label='vx', alpha=0.7)
    ax.plot(traj[:, 3], label='vy', alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("Velocity")
    ax.set_title("Velocity Profile"); ax.legend(); ax.grid(alpha=0.3)

    # --- Controls ---
    ax = axes[1, 0]
    ax.step(range(T), ctrl[:, 0], label='fx', where='mid', alpha=0.7)
    ax.step(range(T), ctrl[:, 1], label='fy', where='mid', alpha=0.7)
    ax.axhline(y=CONFIG['max_force'], color='gray', ls='--', alpha=0.3)
    ax.axhline(y=-CONFIG['max_force'], color='gray', ls='--', alpha=0.3)
    ax.set_xlabel("Step"); ax.set_ylabel("Force")
    ax.set_title("Executed Controls"); ax.legend(); ax.grid(alpha=0.3)

    # --- Mode + distance to pedestrians ---
    ax = axes[1, 1]
    ax.step(range(T), modes, 'k-', where='mid', linewidth=2, alpha=0.5, label='Mode (0=norm,1=ice,2=gravel)')
    for i, ped_traj in enumerate(ped_arrs):
        dist_ped = np.sqrt(np.sum((traj[1:, :2] - ped_traj[1:, :2])**2, axis=1))
        ax.plot(range(T), dist_ped, ped_colors[i], linewidth=1.5, label=f'dist ped{i}')
        ax.axhline(y=PEDESTRIANS[i]['r']+0.05, color=ped_colors[i], ls='--', alpha=0.3)
    ax.set_xlabel("Step"); ax.set_ylabel("Mode / Distance")
    ax.set_title("Mode + Distance to Pedestrians")
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    fig.suptitle("Hybrid MPC: 2 Ice Patches + 2 Pedestrians + Dense Obstacles", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out / "hybrid_mpc_hard.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out / 'hybrid_mpc_hard.png'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid MPC simulation")
    parser.add_argument('--mode', default='where',
                       choices=['where', 'smooth', 'constran', 'composed', 'semantic'],
                       help="Cost function: 'where' (jnp.where), "
                            "'smooth' (nested σ), "
                            "'constran' (Constran.build()), "
                            "'composed' (dynamic k + semantic roles), "
                            "'semantic' (compose_objective_semantic — T_alpha + fixed k)")
    args = parser.parse_args()
    run_hybrid_mpc(cost_mode=args.mode)
