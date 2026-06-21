"""
Constran — Constraint Transformation Engine
============================================

Translates user-defined objectives and constraints into solver-ready
black-box cost functions using the saturation nesting framework.

See ``ConstraintsTransformation_README.md`` for the full methodology.

Quick Start
-----------
>>> from Constraintdealer.Constran import *
>>>
>>> def my_obj(x, ctx):
...     return jnp.sum((x - ctx['target']) ** 2)
>>>
>>> constraints = [
...     Deterministic(lambda x, ctx: x[0] + x[1] + 4,
...                    mode='hard', priority=1),
...     Deterministic(lambda x, ctx: x[1] - x[0],
...                    mode='soft', priority=2),
... ]
>>>
>>> cost_fn = build(my_obj, constraints)
>>> # cost_fn is JAX-compatible: (x, ctx) -> scalar
>>> # Pass directly to any solver:
>>> # mmog_igo_optimizer_mpc(..., fitness_fn_total=cost_fn, ...)

Solver Compatibility
--------------------
- MPCsolverM22.mmog_igo_optimizer_mpc — fitness_fn_total
- MPCsolver.igo_mog_optimizer       — fitness_fn
- blockwise_mgigo.blockwise_mgigo   — objective_fn
- TSP.plackett_luce_igo_optimizer_tsp — fitness_fn_total
- Multi-agent (MPC_G, MPC_G_MS) — use build_multi_agent
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random, vmap, lax
from dataclasses import dataclass
from typing import (Any, Callable, Dict, List, Optional, Sequence,
                    Tuple, Union)
import warnings

# ===========================================================================
# 1. Core Math Utilities (JAX-jittable)
# ===========================================================================

@jax.jit
def sigma_k(x: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
    """Saturation: σ_k(x) = kx / √(1 + (kx)²). Odd, output ∈ (-1, 1)."""
    kx = k * x
    return kx / jnp.sqrt(1.0 + kx ** 2)


@jax.jit
def log_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Log transform: T(x) = sign(x)·log(1+|x|). Odd, compresses range."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


# ===========================================================================
# 2. Constraint Specification Dataclasses
# ===========================================================================

@dataclass(kw_only=True)
class ConstraintSpec:
    """Base specification for one constraint layer.

    Use subclasses: ``Deterministic``, ``Chance``, ``Robust``, ``DRO``.

    Parameters
    ----------
    mode : 'hard' | 'soft' | 'tunable'
    priority : 1 = highest priority (outermost layer).
    delta : δ for hard mode (auto-assigned if None).
    delta_soft, beta : parameters for tunable mode.
    """
    mode: str = 'soft'
    priority: int = 1
    delta: Optional[float] = None
    delta_soft: Optional[float] = None
    beta: Optional[float] = None

    def __post_init__(self):
        if self.mode not in ('hard', 'soft', 'tunable'):
            raise ValueError(
                f"mode must be 'hard', 'soft', or 'tunable', got {self.mode!r}")


@dataclass
class Deterministic(ConstraintSpec):
    """Deterministic constraint: g(x) ≤ 0.

    g_fn(x, ctx) -> scalar. Positive = violated, negative = satisfied.
    """
    g_fn: Optional[Callable[[jnp.ndarray, Any], jnp.ndarray]] = None


@dataclass
class Chance(ConstraintSpec):
    """Chance constraint: P(g(x,ξ) ≤ 0) ≥ 1-α.

    g_fn(x, xi, ctx) -> scalar.
    noise_fn(key, shape) -> noise samples from known distribution.
    alpha: risk level (0.1 means 90% probability).
    n_samples: MC sample count (default 100).
    """
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
    """Robust constraint: g(x,ξ) ≤ 0 for all ξ ∈ Ξ.

    g_fn(x, xi, ctx) -> scalar.
    uncertainty_set: 1D array of xi values, or callable build_set(n) -> array.
    n_grid: discretization points (default 40).
    """
    g_fn: Optional[Callable] = None
    uncertainty_set: Union[jnp.ndarray, Callable, Sequence, None] = None
    n_grid: int = 40


@dataclass
class DRO(ConstraintSpec):
    """Distributionally robust: inf_{P∈𝒫} P(g≤0) ≥ 1-α.

    g_fn(x, xi, ctx) -> scalar.
    ambiguity_set: list of noise_fn(key, shape) callables.
    alpha: risk level.
    n_samples_per_dist: MC samples per candidate distribution.
    """
    g_fn: Optional[Callable] = None
    ambiguity_set: Optional[List[Callable]] = None
    alpha: float = 0.1
    n_samples_per_dist: int = 100

    def __post_init__(self):
        super().__post_init__()
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")


# ===========================================================================
# 3. Internal: Violation Function Builders
# ===========================================================================

def _make_violation_fn(spec: ConstraintSpec) -> Callable:
    """Build a violation function (x, ctx) -> g_raw for one constraint.

    g_raw > 0 means violated, g_raw < 0 means satisfied.
    """
    if isinstance(spec, Deterministic):
        return spec.g_fn

    elif isinstance(spec, Chance):
        g_fn = spec.g_fn
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
        uset = spec.uncertainty_set
        N = spec.n_grid

        if callable(uset):
            xi_all = uset(N)
        else:
            xi_all = jnp.asarray(uset)

        def robust_violation(x, ctx):
            def body(carry, xi):
                return jnp.maximum(carry, g_fn(x, xi, ctx)), None
            worst, _ = lax.scan(body, -jnp.inf, xi_all)
            return worst

        return robust_violation

    elif isinstance(spec, DRO):
        g_fn = spec.g_fn
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
# 4. Internal: Nesting Assembler
# ===========================================================================

def _assemble_nest(objective_fn: Callable,
                   layers: List[Tuple[int, str, dict, Callable]],
                   k_inner: float = 0.1,
                   penalize_only_soft: bool = False,
                   ) -> Callable:
    """Build the nested saturation cost function.

    layers: list of (priority, mode, params, viol_fn), sorted from LOWEST
            to HIGHEST priority (inside-out build order).
    """
    def cost_fn(x, ctx):
        inner = sigma_k(log_transform(objective_fn(x, ctx)), k=k_inner)

        for _priority, mode, params, viol_fn in layers:
            g_raw = viol_fn(x, ctx)

            if mode == 'hard':
                delta = params.get('delta', 3.0)
                inner = sigma_k(
                    jnp.where(g_raw > 0,
                              log_transform(g_raw) + delta,
                              inner)
                )
            elif mode == 'tunable':
                delta_soft = params.get('delta_soft', 2.0)
                beta = params.get('beta', 5.0)
                contrib = delta_soft * sigma_k(beta * log_transform(g_raw))
                inner = sigma_k(contrib + inner)
            else:  # 'soft'
                t_val = log_transform(g_raw)
                if penalize_only_soft:
                    t_val = jnp.maximum(0.0, t_val)
                inner = sigma_k(t_val + inner)

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
          ) -> Callable[[jnp.ndarray, Any], jnp.ndarray]:
    """Build a solver-ready cost function from objective and constraints.

    Supports **any number M** of constraints with arbitrary priority
    ordering. Each constraint layer independently selects its type
    (Deterministic / Chance / Robust / DRO) and mode (Hard / Tunable / Soft).

    Parameters
    ----------
    objective_fn : callable
        ``objective_fn(x, ctx) -> scalar``. Raw objective (may be negative).
    constraints : list of ConstraintSpec
        ``Deterministic``, ``Chance``, ``Robust``, or ``DRO`` instances.
        Any number M ≥ 0. Sorted internally by priority.
    k_inner : float
        Innermost σ knee. Default 0.1 (good for f up to ~1e8).
    penalize_only_soft : bool
        If True, soft/tunable modes use max(0, T(g)) — never reward.
    validate : bool
        If True, check consistency and warn about ordering issues.
    jit_cost : bool
        If True (default), wrap the returned function with ``jax.jit``.
        Set to False if the solver already JIT-compiles the full loop
        (e.g., MPCsolverM22 via ``lax.scan``) — avoids redundant wrapping.

    Returns
    -------
    cost_fn : callable
        ``cost_fn(x, ctx) -> scalar``. JAX-compatible.
        Pass to any solver as ``fitness_fn_total`` or ``objective_fn``.

    **Performance — Avoiding JAX Recompilation in MPC:**

    Call ``build()`` **once** before the optimization loop. All dynamic
    information (changing target, moving obstacles, new opponent states)
    must flow through the ``ctx`` parameter — never by rebuilding the
    cost function. Rebuilding calls ``jax.jit`` on a new function object,
    which forces recompilation on every MPC step.

    Correct::

        cost_fn = build(obj, constraints)     # ONCE, before loop
        for t in range(T_mpc):
            ctx = {'target': targets[t], ...}  # update context
            result = solver(..., fitness_fn_total=cost_fn, context=ctx)

    Wrong (forces recompile every step)::

        for t in range(T_mpc):
            cost_fn = build(obj, constraints)  # DON'T DO THIS
            result = solver(..., fitness_fn_total=cost_fn, context=ctx)

    Examples
    --------
    >>> cost_fn = build(
    ...     lambda x, ctx: jnp.sum(x**2),
    ...     [
    ...         Deterministic(lambda x, ctx: x[0] + x[1] + 4,
    ...                        mode='hard', priority=1, delta=1.5),
    ...         Chance(lambda x, xi, ctx: x[0] + xi,
    ...                noise_fn=lambda k, s: random.normal(k, s),
    ...                alpha=0.1, mode='soft', priority=2),
    ...     ])
    """
    if constraints is None:
        constraints = []

    if validate:
        _validate_constraints(constraints)

    # Sort by priority: lowest priority first (= inside-out build order)
    specs_sorted = sorted(constraints, key=lambda s: s.priority, reverse=True)

    # Auto-assign δ for hard layers
    hard_specs = [s for s in specs_sorted if s.mode == 'hard']
    for i, spec in enumerate(hard_specs):
        if spec.delta is None:
            # Outermost Hard (highest priority = smallest priority number)
            is_outermost = (spec.priority == min(s.priority for s in hard_specs))
            spec.delta = 1.5 if is_outermost else 3.0

    layers = []
    for spec in specs_sorted:
        viol_fn = _make_violation_fn(spec)

        params = {}
        if spec.mode == 'hard':
            params['delta'] = spec.delta if spec.delta is not None else 3.0
        elif spec.mode == 'tunable':
            params['delta_soft'] = (spec.delta_soft if spec.delta_soft is not None
                                    else 2.0)
            params['beta'] = spec.beta if spec.beta is not None else 5.0

        layers.append((spec.priority, spec.mode, params, viol_fn))

    cost_fn = _assemble_nest(objective_fn, layers,
                             k_inner=k_inner,
                             penalize_only_soft=penalize_only_soft)

    if jit_cost:
        return jax.jit(cost_fn)
    return cost_fn


def build_multi_agent(
    agent_specs: Dict[int, Tuple[Callable, Optional[Sequence[ConstraintSpec]]]],
    *,
    k_inner: float = 0.1,
    penalize_only_soft: bool = False,
    validate: bool = True,
) -> Dict[int, Callable]:
    """Build agent-aware cost functions for multi-agent game solvers.

    Parameters
    ----------
    agent_specs : dict
        ``{agent_id: (objective_fn, constraints)}``.
    k_inner, penalize_only_soft, validate : see ``build()``.

    Returns
    -------
    agent_fns : dict
        ``{agent_id: cost_fn(agent_idx, joint_x, ctx) -> scalar}``.
        Compatible with MPC_G, MPC_G_S, MPC_G_MS solvers.

    Examples
    --------
    >>> agent_fns = build_multi_agent({
    ...     0: (lambda x, ctx: jnp.sum((x[:2]-ctx['t0'])**2), [
    ...         Deterministic(lambda x, ctx: x[0] - 1, mode='hard', priority=1)
    ...     ]),
    ...     1: (lambda x, ctx: jnp.sum((x[2:]-ctx['t1'])**2), []),
    ... })
    """
    result = {}
    for agent_id, (obj_fn, constraints) in agent_specs.items():
        base_fn = build(obj_fn, constraints,
                        k_inner=k_inner,
                        penalize_only_soft=penalize_only_soft,
                        validate=validate)

        # Wrap to multi-agent signature: (agent_idx, joint_x, ctx) -> scalar
        def _wrap(base, aid):
            def agent_fn(agent_idx, joint_x, ctx):
                _ = agent_idx  # passed by solver for routing
                return base(joint_x, ctx)
            return agent_fn

        result[agent_id] = _wrap(base_fn, agent_id)

    return result


def build_unconstrained(objective_fn: Callable,
                        k_inner: float = 0.1,
                        ) -> Callable:
    """Build cost function for unconstrained optimization.

    Equivalent to ``build(objective_fn, constraints=[])``.
    """
    return build(objective_fn, constraints=[], k_inner=k_inner, validate=False)


# ===========================================================================
# 6. Validation & Diagnostics
# ===========================================================================

def _validate_constraints(constraints: Sequence[ConstraintSpec]) -> None:
    """Check constraint configuration for common issues."""
    if not constraints:
        return

    for i, spec in enumerate(constraints):
        label = f"constraint[{i}] ({type(spec).__name__}, priority={spec.priority})"

        if isinstance(spec, Deterministic) and spec.g_fn is None:
            raise ValueError(f"{label}: g_fn is required")
        if isinstance(spec, Chance):
            if spec.g_fn is None:
                raise ValueError(f"{label}: g_fn is required")
            if spec.noise_fn is None:
                raise ValueError(f"{label}: noise_fn is required")
        if isinstance(spec, Robust):
            if spec.g_fn is None:
                raise ValueError(f"{label}: g_fn is required")
            if spec.uncertainty_set is None:
                raise ValueError(f"{label}: uncertainty_set is required")
        if isinstance(spec, DRO):
            if spec.g_fn is None:
                raise ValueError(f"{label}: g_fn is required")
            if not spec.ambiguity_set:
                raise ValueError(f"{label}: ambiguity_set is required")

    # Warn if hard layer is inside soft layer (ordering issue)
    specs_by_prio = sorted(constraints, key=lambda s: s.priority)
    seen_soft = False
    for spec in specs_by_prio:
        if spec.mode in ('soft', 'tunable'):
            seen_soft = True
        elif spec.mode == 'hard' and seen_soft:
            warnings.warn(
                f"Hard constraint (priority={spec.priority}) is inside "
                f"a soft/tunable layer. Hard layers should be OUTERMOST "
                f"(lower priority number). See README §5.7."
            )
            break


def quick_check(cost_fn: Callable,
                x_samples: Sequence[jnp.ndarray],
                ctx: Any = None,
                ) -> Dict[str, Any]:
    """Quick validation of a built cost function.

    Pass several x values representing different scenarios
    (e.g., feasible, small violation, large violation).

    >>> result = quick_check(cost_fn, [
    ...     jnp.array([2.0, -2.0]),   # all satisfied
    ...     jnp.array([5.0, 4.0]),    # L2 violated
    ...     jnp.array([-3.0, -3.0]),  # L1 violated
    ... ])
    >>> print(result['ok'])  # True if output range is healthy
    """
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


# ===========================================================================
# 7. Convenience: Auto-Assign Deltas
# ===========================================================================

def autodelta(constraints: Sequence[ConstraintSpec]) -> List[ConstraintSpec]:
    """Auto-assign δ values: outermost Hard gets 1.5, inner Hard get 3.0.

    Modifies constraints in-place and returns the list.
    """
    hard_specs = [s for s in constraints if s.mode == 'hard']
    if not hard_specs:
        return list(constraints)

    min_prio = min(s.priority for s in hard_specs)
    for spec in hard_specs:
        if spec.delta is None:
            spec.delta = 1.5 if spec.priority == min_prio else 3.0

    return list(constraints)
