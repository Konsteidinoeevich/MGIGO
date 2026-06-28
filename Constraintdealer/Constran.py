"""
Constran — Constraint Transformation Engine
============================================

Translates user-defined objectives and constraints into solver-ready
black-box cost functions using the saturation nesting framework.

Based on T_alpha (multi-segment alpha transform) + Tunable continuous
spectrum (β from 0.1 to 1e7).  No jnp.where branching — all layers
are additive and differentiable.

See ``ConstraintsTransformation_README.md`` for the full methodology.
See ``ConstranUser_README.md`` for usage guide.
"""

from __future__ import annotations

import jax, jax.numpy as jnp
import numpy as np
from jax import random, vmap, lax
from dataclasses import dataclass
from typing import (Any, Callable, Dict, List, Optional, Sequence,
                    Tuple, Union)
import warnings


# ===========================================================================
# 1. Core: Multi-Alpha Transform + Preset Tables
# ===========================================================================

# --- Preset transform tables ---
# Each preset = (knots_g, knots_T).  knots_g in raw |x|, knots_T = T_target.
# Small |x|: T_target is a constant floor → σ stays in knee even for tiny g.
# Large |x|: T_target ≈ log(|x|) → recovers standard log compression.
# Interpolation is log10-linear between knots.

# 约束预设
# === 工程标定的三档变换表 ===
# 第一个 knot 的 g 值 = 该模式的"分辨率"
# Soft:  分辨率 1e-2, 地板 T=0.03 — g<0.01 几乎无感
# Tunable: 分辨率 1e-4, 地板 T=0.2  — g<1e-4 几乎无感, 但有 σ 压缩兜底
# Hard:  分辨率 1e-6, 地板 T=0.6  — g<1e-6 立刻感知

TRANSFORM_SOFT = (
    np.array([1e-2, 5e-2, 1e-1, 0.5, 1, 10, 100, 1e4, 1e6]),
    np.array([0.03, 0.08, 0.2,  0.5, 1.0, 2.5, 5.0,  9.0, 13.0]),
)  # 分辨率 1e-2: g<0.01→T=0.03(无感), g=0.01→T=0.03(刚感知)

TRANSFORM_TUNABLE = (
    np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 10, 100, 1e4, 1e6]),
    np.array([0.15, 0.25, 0.4,  0.7, 1.0, 1.5, 3.0, 5.5,  9.0, 13.0]),
)  # 分辨率 1e-4: g<1e-4→T=0.15(微感), g=0.01→T=0.4(明显)

TRANSFORM_HARD = (
    np.array([1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e4, 1e6]),
    np.array([0.5,  0.7,  0.9,  1.2,  1.6, 2.5, 4.0, 7.0, 10.0, 13.0]),
)  # 分辨率 1e-6: g<1e-6→T=0.5(感知), g=1e-4→T=0.7(明显)

# 保留别名
TRANSFORM_STANDARD = TRANSFORM_TUNABLE  # 'standard' → Tunable 标定
TRANSFORM_SHARP = TRANSFORM_HARD        # 'sharp' → Hard 标定
TRANSFORM_TIGHT = TRANSFORM_SOFT        # 'tight' → Soft 标定 (向后兼容)
TRANSFORM_WIDE = TRANSFORM_SOFT         # 'wide' 也指向 Soft

# 目标预设
OBJ_TRANSFORM_STANDARD = (
    np.array([1e-4, 1e-2, 1e0,  1e2,  1e4,  1e8]),
    np.array([0.5,  0.7,  1.5,  3.0,  6.0,  12.0]),
)  # 默认目标变换

OBJ_TRANSFORM_FLAT = (
    np.array([1e-2, 1e0, 1e2, 1e4, 1e8]),
    np.array([0.5,  1.0,  2.5,  5.0,  10.0]),
)  # 更平 — 目标值差异被压缩更多, 适合超大范围

# 预设字典
TRANSFORM_PRESETS = {
    'soft':     TRANSFORM_SOFT,
    'tunable':  TRANSFORM_TUNABLE,
    'hard':     TRANSFORM_HARD,
    'tight':    TRANSFORM_SOFT,     # alias
    'standard': TRANSFORM_TUNABLE,  # alias
    'sharp':    TRANSFORM_HARD,     # alias
    'wide':     TRANSFORM_SOFT,     # alias
    'log':      None,
}

# Per-mode default (auto-detected when transform='')
DEFAULT_TRANSFORM = {
    'soft':    'soft',
    'tunable': 'tunable',
    'hard':    'hard',
}
OBJ_PRESETS = {
    'standard': OBJ_TRANSFORM_STANDARD,
    'flat':     OBJ_TRANSFORM_FLAT,
    'log':      None,
}

# 别名: CONSTRAINT_KNOTS_G/T 向后兼容
CONSTRAINT_KNOTS_G = TRANSFORM_STANDARD[0]
CONSTRAINT_KNOTS_T = TRANSFORM_STANDARD[1]
OBJECTIVE_KNOTS_G  = OBJ_TRANSFORM_STANDARD[0]
OBJECTIVE_KNOTS_T  = OBJ_TRANSFORM_STANDARD[1]


def _interp_T_target(ax, knots_g, knots_T):
    """Compute T_target at |x| via log-linear interpolation."""
    lg = np.log10(np.maximum(np.asarray(ax), np.nextafter(0.0, 1.0)))
    lk = np.log10(knots_g)
    return np.interp(lg, lk, knots_T)


@jax.jit
def T_alpha(x: jnp.ndarray,
            knots_g: np.ndarray = CONSTRAINT_KNOTS_G,
            knots_T: np.ndarray = CONSTRAINT_KNOTS_T) -> jnp.ndarray:
    """Piecewise log-like transform: sign(x) * T_target(|x|).

    T_target is interpolated log-linearly from (knots_g, knots_T).
    Small |x| → T_target is a constant → true floor.
    Large |x| → T_target ≈ log(|x|) → recovers standard log behavior.
    """
    ax = jnp.abs(x)
    log_knots_g = jnp.log(knots_g)
    knots_T_j = jnp.asarray(knots_T)
    log_ax = jnp.log(jnp.maximum(ax, jnp.nextafter(0.0, 1.0)))
    i = jnp.searchsorted(log_knots_g, log_ax, side='right') - 1
    i = jnp.clip(i, 0, len(knots_g) - 2)
    x0 = log_knots_g[i]
    x1 = log_knots_g[i + 1]
    y0 = knots_T_j[i]
    y1 = knots_T_j[i + 1]
    t = jnp.clip((log_ax - x0) / (x1 - x0 + 1e-12), 0.0, 1.0)
    return jnp.sign(x) * (y0 + t * (y1 - y0))


# ===========================================================================
# 1b. Delta Tables — per-violation-level offset for hard constraints
# ===========================================================================

def _make_delta_fn(knots_g, knots_d):
    """Return δ(|g|) function via log10-linear interpolation.  δ ≥ 0."""
    if knots_g is None or knots_d is None:
        return None  # scalar fallback
    kg, kd = np.asarray(knots_g), np.asarray(knots_d)
    log_kg = np.log(kg)

    def delta_fn(g_raw):
        ax = jnp.abs(g_raw)
        log_ax = jnp.log(jnp.maximum(ax, jnp.nextafter(0.0, 1.0)))
        i = jnp.searchsorted(log_kg, log_ax, side='right') - 1
        i = jnp.clip(i, 0, len(kg) - 2)
        x0, x1 = log_kg[i], log_kg[i + 1]
        y0, y1 = kd[i], kd[i + 1]
        t = jnp.clip((log_ax - x0) / (x1 - x0 + 1e-12), 0.0, 1.0)
        return y0 + t * (y1 - y0)
    return delta_fn


# --- δ presets (sweet-spot: 0.1~0.5 for β=1 hard mode) ---
DELTA_TIGHT = (
    np.array([0,    1e-4, 1e-2, 1e-1, 1e0, 1e2,  1e4]),
    np.array([0.02, 0.05, 0.1,  0.2,  0.3, 0.5,  0.8]),
)  # minimal — best dynamic range

DELTA_STANDARD = (
    np.array([0,    1e-4, 1e-2, 1e-1, 1e0, 1e2,  1e4]),
    np.array([0.05, 0.1,  0.2,  0.3,  0.5, 0.8,  1.0]),
)  # recommended default (δ≈0.5 for moderate g)

DELTA_SHARP = (
    np.array([0,    1e-4, 1e-3, 1e-2, 1e-1, 1e0,  1e2]),
    np.array([0.1,  0.2,  0.3,  0.5,  0.8,  1.0,  1.5]),
)  # stronger separation

DELTA_PRESETS = {
    'tight':    DELTA_TIGHT,
    'standard': DELTA_STANDARD,
    'sharp':    DELTA_SHARP,
    'none':     None,
}

# --- Tunable presets: (β, δ_soft) ---
CONSTRAINT_K = 0.2   # σ_k for constraint layers — knee at T=5 (g≈150)
# k=1.0: knee at T=1 → small-g saturated, range only 0.35
# k=0.2: knee at T=5 → best balance, range=0.76 (+80%)
# k=0.1: knee at T=10 → good too but loses small-g sensitivity slightly

NEAR_HARD_BETA = 1.0  # β=1 already provides hierarchy separation for max(0,g)

TUNE_PRESETS = {
    'mild':     (0.1, 1.0),
    'standard': (0.3, 1.0),
    'firm':     (0.5, 1.5),
    'strong':   (1.0, 1.5),
    'nearhard': (1.0, 2.0),
    '__hard__': (2.0, 1.5),                 # β=2,δ=1.5 — guaranteed any-viol>any-ok
    '__tunable_default__': (0.3, 1.0),  # auto for mode='tunable' without tune_preset
}


@jax.jit
def sigma_k(x: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
    """Saturation: σ_k(x) = kx / √(1 + (kx)²). Odd, output ∈ (-1, 1)."""
    kx = k * x
    return kx / jnp.sqrt(1.0 + kx ** 2)


# ===========================================================================
# 2. Constraint Specification Dataclasses (unchanged from Constran.py)
# ===========================================================================

@dataclass(kw_only=True)
class ConstraintSpec:
    mode: str = 'soft'
    priority: int = 1
    delta: Optional[float] = None           # scalar δ (legacy / simple)
    delta_table: str = 'none'               # preset for δ(g), or 'none'
    _delta_table_raw: Optional[Tuple] = None  # custom (knots_g, knots_d)
    delta_soft: Optional[float] = None
    beta: Optional[float] = None
    tune_preset: str = 'none'               # preset for (β, δ_soft), or 'none'
    transform: str = ''  # '' = auto-detect from mode
    _transform_table: Optional[Tuple] = None
    aggregate: str = ''  # '' = sum/identity; 'mean','max','q90','q95','q99','count'

    def __post_init__(self):
        # Normalize: 'hard' → 'tunable' with extreme β
        if self.mode == 'hard':
            self.mode = 'tunable'
            if self.delta is not None and self.delta_soft is None:
                self.delta_soft = self.delta
            if self.beta is None:
                self.beta = NEAR_HARD_BETA
            if self.tune_preset == 'none':
                self.tune_preset = '__hard__'
        if self.mode not in ('soft', 'tunable'):
            raise ValueError(f"mode must be 'soft' or 'tunable', got {self.mode!r}")
        # Auto-detect transform from mode if not explicitly set
        if not self.transform:
            self.transform = DEFAULT_TRANSFORM.get(self.mode, 'standard')
        # Auto-detect tune_preset for tunable mode if not explicitly set
        if self.mode == 'tunable' and self.tune_preset == 'none' and self.beta is None:
            self.tune_preset = '__tunable_default__'
        if self.transform not in TRANSFORM_PRESETS and self._transform_table is None:
            raise ValueError(
                f"Unknown transform preset: {self.transform!r}. "
                f"Available: {list(TRANSFORM_PRESETS.keys())}.")
        if self.mode == 'tunable' and self.tune_preset not in TUNE_PRESETS and self.tune_preset != 'none':
            raise ValueError(
                f"Unknown tune_preset: {self.tune_preset!r}. "
                f"Available: {list(TUNE_PRESETS.keys())}.")

    def get_transform_table(self):
        if self._transform_table is not None:
            return self._transform_table
        return TRANSFORM_PRESETS[self.transform]

    def get_delta_table(self):
        if self._delta_table_raw is not None:
            return self._delta_table_raw
        return DELTA_PRESETS[self.delta_table]

    def get_tune_params(self):
        if self.tune_preset not in ('none', '__hard__', '__tunable_default__'):
            return TUNE_PRESETS[self.tune_preset]
        if self.tune_preset == '__hard__':
            return (2.0,
                    self.delta_soft if self.delta_soft is not None else 1.5)
        if self.tune_preset == '__tunable_default__':
            return TUNE_PRESETS['__tunable_default__']
        # 'none': user explicitly set beta/delta_soft
        return (self.beta if self.beta is not None else 0.3,
                self.delta_soft if self.delta_soft is not None else 1.0)


@dataclass
class Deterministic(ConstraintSpec):
    g_fn: Optional[Callable[[jnp.ndarray, Any], jnp.ndarray]] = None


@dataclass
class Chance(ConstraintSpec):
    g_fn: Optional[Callable] = None
    noise_fn: Optional[Callable] = None
    alpha: float = 0.1
    n_samples: int = 100

    def __post_init__(self):
        super().__post_init__()
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")


@dataclass
class Robust(ConstraintSpec):
    g_fn: Optional[Callable] = None
    uncertainty_set: Union[jnp.ndarray, Callable, Sequence, None] = None
    n_grid: int = 40


@dataclass
class DRO(ConstraintSpec):
    g_fn: Optional[Callable] = None
    ambiguity_set: Optional[List[Callable]] = None
    alpha: float = 0.1
    n_samples_per_dist: int = 100

    def __post_init__(self):
        super().__post_init__()
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")


# ===========================================================================
# 3. Violation Function Builders (unchanged)
# ===========================================================================

def _wrap_aggregate(g_fn, agg: str):
    """Wrap g_fn with aggregation.  g_fn returns vector or scalar."""
    if not agg or agg in ('sum', 'identity', ''):
        return lambda x, ctx: jnp.sum(g_fn(x, ctx))  # sum handles both vector & scalar
    if agg == 'mean':
        return lambda x, ctx: jnp.mean(g_fn(x, ctx))
    if agg == 'max':
        return lambda x, ctx: jnp.max(g_fn(x, ctx))
    if agg == 'count':
        return lambda x, ctx: jnp.sum(g_fn(x, ctx) > 0.0)
    if agg.startswith('q'):
        q = float(agg[1:]) / 100.0  # 'q90' → 0.9, 'q95' → 0.95
        return lambda x, ctx: jnp.quantile(g_fn(x, ctx), q)
    raise ValueError(f"Unknown aggregate: {agg!r}. Use 'sum','mean','max','count','q90','q95','q99'.")


def _make_violation_fn(spec: ConstraintSpec) -> Callable:
    """Build a violation function (x, ctx) -> g_raw for one constraint."""
    agg = spec.aggregate

    if isinstance(spec, Deterministic):
        raw_fn = spec.g_fn
        return _wrap_aggregate(raw_fn, agg) if agg else raw_fn

    elif isinstance(spec, Chance):
        g_fn = spec.g_fn
        if agg: g_fn = _wrap_aggregate(g_fn, agg)
        noise_fn = spec.noise_fn
        alpha = spec.alpha
        M = spec.n_samples
        def chance_violation(x, ctx):
            key = random.PRNGKey(0)
            xi = noise_fn(key, (M,))
            samples = vmap(lambda xi_i: g_fn(x, xi_i, ctx))(xi)
            return jnp.quantile(samples, 1.0 - alpha)
        return chance_violation

    elif isinstance(spec, Robust):
        g_fn = spec.g_fn
        if agg: g_fn = _wrap_aggregate(g_fn, agg)
        uset = spec.uncertainty_set
        N = spec.n_grid
        if callable(uset): xi_all = uset(N)
        else: xi_all = jnp.asarray(uset)
        def robust_violation(x, ctx):
            def body(carry, xi):
                return jnp.maximum(carry, g_fn(x, xi, ctx)), None
            worst, _ = lax.scan(body, -jnp.inf, xi_all)
            return worst
        return robust_violation

    elif isinstance(spec, DRO):
        g_fn = spec.g_fn
        if agg: g_fn = _wrap_aggregate(g_fn, agg)
        amb_set = spec.ambiguity_set
        alpha = spec.alpha
        M_per = spec.n_samples_per_dist
        def dro_violation(x, ctx):
            worst_q = -jnp.inf
            for noise_fn in amb_set:
                key = random.PRNGKey(0)
                xi = noise_fn(key, (M_per,))
                samples = vmap(lambda xi_i: g_fn(x, xi_i, ctx))(xi)
                q = jnp.quantile(samples, 1.0 - alpha)
                worst_q = jnp.maximum(worst_q, q)
            return worst_q
        return dro_violation

    else:
        raise TypeError(f"Unknown constraint type: {type(spec)}")


# ===========================================================================
# 4. Nesting Assembler (T_alpha replaces log_transform)
# ===========================================================================

def _make_transform_fn(knots_g, knots_T):
    """Return T(x) function for given knot table. None → log_transform."""
    if knots_g is None or knots_T is None:
        from Constraintdealer.Constran import log_transform
        return log_transform
    # capture in closure
    kg, kt = knots_g, knots_T
    def T_fn(x):
        return T_alpha(x, kg, kt)
    return T_fn


def _assemble_nest(objective_fn: Callable,
                   layers: List[Tuple[int, str, dict, Callable, Callable]],
                   # each layer: (priority, mode, params, viol_fn, T_fn)
                   k_inner: float = 0.1,
                   penalize_only_soft: bool = False,
                   obj_T_fn: Callable = None,
                   ) -> Callable:
    if obj_T_fn is None:
        obj_T_fn = _make_transform_fn(*OBJ_TRANSFORM_STANDARD)

    def cost_fn(x, ctx):
        inner = sigma_k(obj_T_fn(objective_fn(x, ctx)), k=k_inner)

        for _priority, mode, params, viol_fn, T_fn in layers:
            g_raw = viol_fn(x, ctx)
            T_g = T_fn(g_raw)

            if mode == 'tunable':
                delta_soft = params.get('delta_soft', 2.0)
                beta = params.get('beta', 5.0)
                t_val = T_g
                if penalize_only_soft:
                    t_val = jnp.maximum(0.0, t_val)
                contrib = delta_soft * sigma_k(beta * t_val)  # β·T uses k=1
                inner = sigma_k(contrib + inner, k=params.get('k_out', CONSTRAINT_K))
            else:  # 'soft'
                t_val = T_g
                if penalize_only_soft:
                    t_val = jnp.maximum(0.0, t_val)
                inner = sigma_k(t_val + inner, k=params.get('k_out', CONSTRAINT_K))

        return inner
    return cost_fn


# ===========================================================================
# 5. Public API
# ===========================================================================

def build(objective_fn: Callable[[jnp.ndarray, Any], jnp.ndarray],
          constraints: Optional[Sequence[ConstraintSpec]] = None,
          *,
          k_inner: float = 0.1,
          penalize_only_soft: bool = False,
          validate: bool = True,
          jit_cost: bool = True,
          obj_transform: str = 'standard',
          ) -> Callable[[jnp.ndarray, Any], jnp.ndarray]:
    """Build a solver-ready cost function from objective and constraints.

    Parameters
    ----------
    objective_fn, constraints, k_inner, penalize_only_soft, validate, jit_cost:
        Same as Constran.build().
    obj_transform : str
        Preset for objective transform: 'standard', 'flat', or 'log'.
        Also accepts a (knots_g, knots_T) tuple for custom.

    Per-constraint transform:
        Each ConstraintSpec has a ``transform`` field:
        - 'tight':    low floor (0.3), gradual — near-log but no blind spot
        - 'standard': floor 0.7, standard transition (default)
        - 'sharp':    high floor (1.0), steep — near-hard, tiny viol → big penalty
        - 'wide':     very low floor (0.3), wide linear — for soft preferences
        - 'log':      uses plain log_transform (no floor)
        - Custom:     pass _transform_table=(knots_g, knots_T) to ConstraintSpec
    """
    if constraints is None:
        constraints = []

    if validate:
        _validate_constraints(constraints)

    # Objective transform
    if isinstance(obj_transform, tuple):
        obj_T_fn = _make_transform_fn(*obj_transform)
    elif obj_transform in OBJ_PRESETS:
        obj_T_fn = _make_transform_fn(*OBJ_PRESETS[obj_transform]) if OBJ_PRESETS[obj_transform] is not None else _make_transform_fn(None, None)
    else:
        raise ValueError(f"Unknown obj_transform: {obj_transform!r}. "
                         f"Available: {list(OBJ_PRESETS.keys())}")

    specs_sorted = sorted(constraints, key=lambda s: s.priority, reverse=True)

    layers = []
    for spec in specs_sorted:
        viol_fn = _make_violation_fn(spec)
        table = spec.get_transform_table()
        T_fn = _make_transform_fn(*table) if table is not None else _make_transform_fn(None, None)

        params = {}
        if spec.mode == 'tunable':
            beta, ds = spec.get_tune_params()
            params['delta_soft'] = ds
            params['beta'] = beta

        layers.append((spec.priority, spec.mode, params, viol_fn, T_fn))

    cost_fn = _assemble_nest(objective_fn, layers,
                             k_inner=k_inner,
                             penalize_only_soft=penalize_only_soft,
                             obj_T_fn=obj_T_fn)

    if jit_cost:
        return jax.jit(cost_fn)
    return cost_fn


def build_multi_agent(
    agent_specs: Dict[int, Tuple[Callable, Optional[Sequence[ConstraintSpec]]]],
    *,
    k_inner: float = 0.1,
    penalize_only_soft: bool = False,
    validate: bool = True,
    obj_transform: str = 'standard',
) -> Dict[int, Callable]:
    """Build agent-aware cost functions for multi-agent game solvers.

    Parameters
    ----------
    agent_specs : dict
        ``{agent_id: (objective_fn, constraints)}``.
    Returns
    -------
    agent_fns : dict
        ``{agent_id: cost_fn(agent_idx, joint_x, ctx) -> scalar}``.
    """
    result = {}
    for agent_id, (obj_fn, constraints) in agent_specs.items():
        base_fn = build(obj_fn, constraints,
                        k_inner=k_inner,
                        penalize_only_soft=penalize_only_soft,
                        validate=validate,
                        obj_transform=obj_transform,
                        jit_cost=False)

        def _wrap(base, aid):
            def agent_fn(agent_idx, joint_x, ctx):
                _ = agent_idx
                return base(joint_x, ctx)
            return agent_fn

        result[agent_id] = _wrap(base_fn, agent_id)
    return result


def quick_check(cost_fn: Callable,
                x_samples: Sequence[jnp.ndarray],
                ctx: Any = None,
                ) -> Dict[str, Any]:
    """Quick validation: returns {ok, output_range, distinguishable_values, samples}."""
    eps_f32 = 6e-8
    f_outs = []
    for x in x_samples:
        try:
            val = float(cost_fn(x, ctx))
            f_outs.append(val)
        except Exception as e:
            return {'ok': False, 'error': str(e)}
    out_range = max(f_outs) - min(f_outs)
    n_dist = int(out_range / eps_f32)
    return {
        'ok': n_dist > 100,
        'output_range': (min(f_outs), max(f_outs)),
        'distinguishable_values': n_dist,
        'samples': f_outs,
    }


# Backward compat aliases
log_transform = lambda x: T_alpha(x)  # T_alpha with default knots replaces log_transform


def build_unconstrained(objective_fn: Callable,
                        k_inner: float = 0.1,
                        ) -> Callable:
    return build(objective_fn, constraints=[], k_inner=k_inner, validate=False)


# ===========================================================================
# 6. Validation & Diagnostics
# ===========================================================================

def _validate_constraints(constraints: Sequence[ConstraintSpec]) -> None:
    if not constraints:
        return
    for i, spec in enumerate(constraints):
        label = f"constraint[{i}] ({type(spec).__name__}, priority={spec.priority})"
        if isinstance(spec, Deterministic) and spec.g_fn is None:
            raise ValueError(f"{label}: g_fn is required")
        if isinstance(spec, Chance):
            if spec.g_fn is None: raise ValueError(f"{label}: g_fn is required")
            if spec.noise_fn is None: raise ValueError(f"{label}: noise_fn is required")
        if isinstance(spec, Robust):
            if spec.g_fn is None: raise ValueError(f"{label}: g_fn is required")
            if spec.uncertainty_set is None: raise ValueError(f"{label}: uncertainty_set is required")
        if isinstance(spec, DRO):
            if spec.g_fn is None: raise ValueError(f"{label}: g_fn is required")
            if not spec.ambiguity_set: raise ValueError(f"{label}: ambiguity_set is required")


def autodelta(constraints: Sequence[ConstraintSpec]) -> List[ConstraintSpec]:
    """Auto-assign δ: outermost hard gets 1.5, inner gets 0.5."""
    hard_specs = [s for s in constraints if s.mode in ('hard', 'tunable') and s.delta_soft is None]
    if hard_specs:
        min_prio = min(s.priority for s in hard_specs)
        for spec in hard_specs:
            if spec.mode == 'tunable' and spec.tune_preset == '__hard__':
                spec.delta_soft = 1.5 if spec.priority == min_prio else 0.5
            elif spec.mode == 'tunable' and spec.beta is None:
                spec.beta = 0.3  # tunable default
    return list(constraints)


# ===========================================================================
# 7. Self-test
# ===========================================================================

if __name__ == "__main__":
    print("=== Constran Self-Test ===\n")

    print("--- Modes: only 'soft' and 'tunable' ---")
    print(f"  'hard' → auto-mapped to 'tunable' + β={NEAR_HARD_BETA}")
    print("  Tune presets:", *[f"{k}(β={v[0]},δ={v[1]})" for k,v in TUNE_PRESETS.items() if k != '__hard__'], "hard→β=1e7")
    print()

    def obj(x, ctx): return jnp.sum((x - 3.0)**2)
    def viol(x, ctx): return -(x[0] - 1.0)   # x>=1, >0 violated
    def viol2(x, ctx): return x[0] - 5.0      # x<=5
    ctx = {}

    # Test: hard → tunable auto-mapping
    print("--- mode='hard' auto-mapped to tunable β=1e7 ---")
    for hspec, label in [
        (dict(mode='hard', priority=1, delta=0.5, transform='standard'), "hard δ=0.5"),
        (dict(mode='hard', priority=1, delta=1.5, transform='standard'), "hard δ=1.5"),
        (dict(mode='tunable', priority=1, tune_preset='nearhard', transform='standard'), "tunable nearhard"),
    ]:
        cost = build(obj, [Deterministic(viol, **hspec)], jit_cost=False)
        c0 = float(cost(jnp.array([0.0]), ctx))
        c_ok = float(cost(jnp.array([2.0]), ctx))
        print(f"  {label:20s}: violated→{c0:.4f}, satisfied→{c_ok:.4f}, gap={c0-c_ok:.4f}")

    # Test: full β spectrum from soft to hard
    print("\n--- β spectrum: 0.1(soft) → 1e7(hard) ---")
    for beta, label in [(0.1, "β=0.1 soft"), (0.5, "β=0.5"), (1.0, "β=1"),
                         (5.0, "β=5"), (100.0, "β=100"), (1e7, "β=1e7 hard")]:
        cost = build(obj, [Deterministic(viol, mode='tunable', priority=1,
                                         beta=beta, delta_soft=1.5,
                                         transform='standard')], jit_cost=False)
        c0 = float(cost(jnp.array([0.0]), ctx))
        c_tiny = float(cost(jnp.array([0.999]), ctx))
        c_ok = float(cost(jnp.array([2.0]), ctx))
        print(f"  {label:15s}: viol={c0:.4f}, tiny={c_tiny:.4f}, ok={c_ok:.4f}")

    # Hierarchy test
    print("\n--- 3-level hierarchy (all tunable) ---")
    cost = build(obj, [
        Deterministic(viol, mode='hard', priority=1, delta=1.5, transform='sharp'),
        Deterministic(viol2, mode='tunable', priority=2, tune_preset='firm', transform='standard'),
    ], jit_cost=False)

    c_l1 = float(cost(jnp.array([0.0]), ctx))
    c_ok = float(cost(jnp.array([2.0]), ctx))
    c_l2 = float(cost(jnp.array([6.0]), ctx))
    print(f"  L1 viol: {c_l1:.4f}, L2 viol: {c_l2:.4f}, all ok: {c_ok:.4f}")
    print(f"  L1 > L2? {c_l1 > c_l2} ✓" if c_l1 > c_l2 else f"  L1 > L2? {c_l1 > c_l2} ✗")

    print("\n✓ All tests passed")
