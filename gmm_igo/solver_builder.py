"""
Solver Builder — unified interface to MGIGO solvers.

Mirrors the Constran.build() pattern: declare what you want, get back a
solver ready to run.  Handles solver selection, parameter initialisation,
and optional Constran constraint integration.

Usage
-----
>>> from gmm_igo.solver_builder import build_solver

# 1. Simplest: unconstrained 2D
>>> solver = build_solver(my_cost_fn, dims=(2,))
>>> result = solver(key)

# 2. With constraints (auto‑wired through Constran)
>>> from Constraintdealer.Constran import Deterministic
>>> solver = build_solver(my_obj, dims=(2,), constraints=[
...     Deterministic(lambda x, ctx: x[0] + x[1] - 1, mode='hard', priority=1),
... ])
>>> result = solver(key, context=ctx)

# 3. Multi‑block with sample reuse
>>> solver = build_solver(my_cost, dims=(2, 5, 3), solver='reuse_multi',
...                       B_n=50, B_o=30, a_threshold=0.3)
>>> result = solver(key)

# 4. Warm‑start from previous result
>>> result2 = solver(key, warm_start=result)

Solvers
-------
=============== ======================= ===================================
solver=          class                   when to use
=============== ======================= ===================================
"auto"          (heuristic)             single block → M22; multi → reuse
"m22"           MPCsolverM22            gold‑standard, no sample reuse
"reuse_single"  MPC_R                   single‑block with old‑sample reuse
"reuse_multi"   MPC_Rblockwise          multi‑block with old‑sample reuse
=============== ======================= ===================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

# ── resolvers (lazy to avoid circular imports) ────────────────────────

def _get_m22():
    from .MPCsolverM22 import mmog_igo_optimizer_mpc
    return mmog_igo_optimizer_mpc

def _get_reuse_single():
    from .MPC_R import igo_mog_reuse_optimizer
    return igo_mog_reuse_optimizer

def _get_reuse_multi():
    from .MPC_Rblockwise import blockwise_reuse_optimizer
    return blockwise_reuse_optimizer

def _get_m23():
    from .MPCsolverM23 import mmog_igo_optimizer_mpc
    return mmog_igo_optimizer_mpc

def _get_nash_rne():
    from .MPC_G import mmog_igo_rne_solver
    return mmog_igo_rne_solver

def _get_nash_rne_single():
    from .MPC_G_S import mmog_igo_rne_solver
    return mmog_igo_rne_solver

def _get_nash_rne_single_verbose():
    from .MPC_G_S_V import mmog_igo_rne_solver
    return mmog_igo_rne_solver

def _get_nash_rne_blocks():
    from .MPC_G_MS import mmog_igo_rne_blocks_solver
    return mmog_igo_rne_blocks_solver


# ── result type ───────────────────────────────────────────────────────

@dataclass
class SolverResult:
    """Returned by every solver produced by ``build_solver()``.

    Contains everything needed to warm‑start the next call.
    """
    x: jnp.ndarray                   # best solution found  (total_dims,)
    cost: float                      # cost at x
    mu: jnp.ndarray                  # (M, K, D_max)  component means
    S_or_L: jnp.ndarray              # (M, K, D_max, D_max)  precision or Cholesky
    pi: jnp.ndarray                  # (M, K)  mixture weights
    v: Optional[jnp.ndarray] = None  # (M, K-1)  log-odds weights (M22/M23/G)
    solver_name: str = ""
    # extra diagnostics (optional)
    diag: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        v_info = f", v={list(self.v.shape)}" if self.v is not None else ""
        return (f"SolverResult(solver={self.solver_name!r}, "
                f"cost={self.cost:.6f}, x={np.array(self.x)}{v_info})")


# ── internal helpers ──────────────────────────────────────────────────

def _pick_solver(solver: str, M: int) -> str:
    """Resolve 'auto' to a concrete solver."""
    if solver != "auto":
        return solver
    return "m22" if M == 1 else "reuse_multi"


def _validate_dims(dims: Tuple[int, ...]) -> None:
    if not isinstance(dims, tuple) or len(dims) == 0:
        raise ValueError(f"dims must be a non-empty tuple, got {dims!r}")
    for d in dims:
        if not isinstance(d, int) or d <= 0:
            raise ValueError(f"Each dimension must be a positive int, got {d}")


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def build_solver(
    objective_fn: Callable[[jnp.ndarray, Any], jnp.ndarray],
    dims: Tuple[int, ...],
    *,
    # ── solver selection ──
    solver: str = "auto",
    # ── constraints (Constran integration) ──
    constraints: Optional[Sequence] = None,
    k_inner: float = 0.1,
    obj_transform: str = "standard",
    # ── solver hyper‑parameters ──
    T: int = 300,
    dt: float = 0.1,
    K: int = 3,
    T_0: int = 50,
    # for M22
    B: int = 80,
    B0: int = 20,
    # for reuse solvers
    B_n: int = 40,
    B_o: int = 40,
    a_threshold: float = 0.25,
    # ── initialisation ──
    init_mu_range: Tuple[float, float] = (-4.0, 4.0),
    init_precision_scale: float = 4.0,
    # ── M23 ──
    nu_r: float = 1.0,
) -> Callable:
    """Build a solver function.

    Parameters
    ----------
    objective_fn : (x, ctx) -> scalar
        If *constraints* is None this is used directly as the cost.
        Otherwise it is the "inner" objective wrapped by Constran.

    dims : tuple of int
        Block dimensions, e.g. ``(2,)`` for a single 2‑D block or
        ``(2, 5, 3)`` for three blocks.

    solver : str
        ``"auto"`` | ``"m22"`` | ``"reuse_single"`` | ``"reuse_multi"``.
        ``"auto"`` picks M22 for single‑block, reuse_multi for multi‑block.

    constraints : list of ConstraintSpec, optional
        Passed to ``Constran.build()``.  If None the raw *objective_fn*
        is used directly.

    k_inner, obj_transform :
        Forwarded to ``Constran.build()``.  Ignored when *constraints* is
        None / empty.

    T : int
        Number of IGO iterations.

    dt : float
        Learning rate / step size.  Safe range: 0.05–0.3.

    K : int
        Number of GMM components per block.

    T_0 : int
        Weight‑reset period (M22 / reuse_multi only).

    B : int
        Samples per iteration (M22 only).

    B0 : int
        Elite samples per iteration (M22 only).  Must be < B.

    B_n : int
        New samples per iteration (reuse solvers only).

    B_o : int
        Old samples carried forward (reuse solvers only).

    a_threshold : float
        Elite quantile threshold (reuse solvers only).
        Safe range: 0.15–0.5.

    init_mu_range : (float, float)
        Uniform range for random initial component means.

    init_precision_scale : float
        Initial precision = scale * I  (i.e. covariance = I / scale).

    Returns
    -------
    solver_fn : callable
        ``solver_fn(key, *, context=None, warm_start=None) -> SolverResult``.
    """
    _validate_dims(dims)
    M = len(dims)
    D_max = max(dims)
    solver = _pick_solver(solver, M)

    # ── build cost function ──────────────────────────────────────────
    if constraints is not None and len(constraints) > 0:
        from Constraintdealer.Constran import build as constran_build
        cost_fn = constran_build(
            objective_fn, list(constraints),
            k_inner=k_inner, obj_transform=obj_transform,
        )
    else:
        cost_fn = objective_fn

    # ── select solver implementation ─────────────────────────────────
    _uses_v = False   # whether solver tracks log-odds v internally
    _uses_L = True    # L_inv (Cholesky) or S (precision) parametrisation

    if solver == "m22":
        _impl = _get_m22()
        _uses_v = True
    elif solver == "m23":
        _impl = _get_m23()
        _uses_v = True
    elif solver == "reuse_single":
        if M > 1:
            raise ValueError(
                f"reuse_single only supports M=1 block, got dims={dims} (M={M}). "
                f"Use solver='reuse_multi' for multi‑block problems."
            )
        _impl = _get_reuse_single()
    elif solver == "reuse_multi":
        _impl = _get_reuse_multi()
        _uses_L = False  # uses S (precision) directly
    else:
        raise ValueError(
            f"Unknown solver {solver!r}. "
            f"Choose: 'auto', 'm22', 'm23', 'reuse_single', 'reuse_multi'."
        )

    # ── default initialisers ─────────────────────────────────────────
    def _default_mu(key):
        return jax.random.uniform(
            key, (M, K, D_max),
            minval=init_mu_range[0], maxval=init_mu_range[1],
        )

    def _default_L():
        return jnp.tile(
            jnp.eye(D_max) * jnp.sqrt(init_precision_scale),
            (M, K, 1, 1),
        )

    def _default_S():
        return jnp.tile(
            jnp.eye(D_max) * init_precision_scale,
            (M, K, 1, 1),
        )

    def _default_pi():
        return jnp.ones((M, K)) / K

    def _default_v():
        return jnp.zeros((M, K - 1))

    # ── best‑x helpers ───────────────────────────────────────────────

    def _best_x_from_multi_block(mu, cost_fn, context, M, dims):
        K_ = mu.shape[1]
        idx_grid = jnp.stack(
            jnp.meshgrid(*[jnp.arange(K_) for _ in range(M)], indexing='ij'),
            axis=-1,
        ).reshape(-1, M)
        # Evaluate one‑by‑one to avoid vmap / jitted cost_fn dispatch mismatch
        best_x = None; best_c = jnp.inf
        for i in range(idx_grid.shape[0]):
            parts = [mu[j, idx_grid[i, j], :dims[j]] for j in range(M)]
            x_i = jnp.concatenate(parts)
            c_i = cost_fn(x_i, context)
            if c_i < best_c:
                best_c = c_i; best_x = x_i
        return best_x, best_c

    def _best_x_from_single_block(mu, cost_fn, context, dims):
        mu_eff = mu[0, :, :dims[0]]
        # Evaluate one‑by‑one to avoid vmap / jitted cost_fn dispatch mismatch
        best_i = 0; best_c = jnp.inf
        for i in range(mu_eff.shape[0]):
            c_i = cost_fn(mu_eff[i], context)
            if c_i < best_c:
                best_c = c_i; best_i = i
        return mu_eff[best_i], best_c

    def _best_x(mu, cost_fn, context, M, dims):
        if M == 1:
            return _best_x_from_single_block(mu, cost_fn, context, dims)
        return _best_x_from_multi_block(mu, cost_fn, context, M, dims)

    # ── pi ↔ v conversion ───────────────────────────────────────────

    def _pi_to_v(pi):
        return jnp.log(pi[..., :-1] / (pi[..., -1:] + 1e-10))

    def _v_to_pi(v):
        exps = jnp.exp(jnp.clip(v, -70, 70))
        sum_e = 1.0 + jnp.sum(exps, axis=-1, keepdims=True)
        return jnp.concatenate([exps / sum_e, 1.0 / sum_e], axis=-1)

    # ══════════════════════════════════════════════════════════════════
    # M22 / M23 solver (L_inv parametrisation, uses v)
    # ══════════════════════════════════════════════════════════════════
    if solver in ("m22", "m23"):
        def _solve(key, *, context=None,
                   initial_mu=None, initial_S_or_L=None, initial_pi=None,
                   warm_start=None):
            # Resolve with explicit > warm_start > default
            mu_k = initial_mu
            L_k  = initial_S_or_L
            v_k  = None
            if warm_start is not None:
                if mu_k is None:   mu_k = warm_start.mu
                if L_k is None:    L_k  = warm_start.S_or_L
                if (warm_start.v is not None and initial_pi is None):
                    v_k = warm_start.v
            if initial_pi is not None:
                v_k = _pi_to_v(initial_pi)
            if mu_k is None:  mu_k = _default_mu(key)
            if L_k is None:   L_k  = _default_L()
            if v_k is None:   v_k  = _default_v()

            kwargs = dict(
                key=key, T=T, dt=dt, M=M, K=K, B=B, B0=B0,
                dims=dims, T_0=T_0,
                fitness_fn_total=cost_fn,
                initial_mu_k=mu_k,
                initial_L_inv_k=L_k,
                initial_v_k=v_k,
                context=context,
            )
            if solver == "m23":
                kwargs["nu_r"] = nu_r

            final_mu, final_L, final_pi = _impl(**kwargs)

            x_best, cost_best = _best_x(final_mu, cost_fn, context, M, dims)
            final_v = _pi_to_v(final_pi)

            return SolverResult(
                x=x_best, cost=float(cost_best),
                mu=final_mu, S_or_L=final_L, pi=final_pi, v=final_v,
                solver_name=solver,
            )

    # ══════════════════════════════════════════════════════════════════
    # reuse_single (L_inv, single‑block, uses pi directly)
    # ══════════════════════════════════════════════════════════════════
    elif solver == "reuse_single":
        def _solve(key, *, context=None,
                   initial_mu=None, initial_S_or_L=None, initial_pi=None,
                   warm_start=None):
            mu_k = initial_mu
            L_k  = initial_S_or_L
            pi_k = initial_pi
            if warm_start is not None:
                if mu_k is None:  mu_k = warm_start.mu
                if L_k is None:   L_k  = warm_start.S_or_L
                if pi_k is None:  pi_k = warm_start.pi

            # Unwrap M=1 dim for the single‑block solver
            if mu_k is not None:  mu_0 = mu_k[0]       # (K, D)
            else:                 mu_0 = None
            if L_k is not None:   L_0  = L_k[0]         # (K, D, D)
            else:                 L_0  = None
            if pi_k is not None:  pi_0 = pi_k[0]        # (K,)
            else:                 pi_0 = None

            if mu_0 is None:  mu_0 = jax.random.uniform(
                key, (K, D_max), minval=init_mu_range[0], maxval=init_mu_range[1])
            if L_0 is None:   L_0  = jnp.tile(
                jnp.eye(D_max) * jnp.sqrt(init_precision_scale), (K, 1, 1))
            if pi_0 is None:  pi_0 = jnp.ones(K) / K

            final_mu, final_L, final_pi = _impl(
                key=key, T=T, delta_t=dt, K=K,
                B_n=B_n, B_o=B_o, a_threshold=a_threshold,
                fitness_fn=cost_fn,
                initial_mu=mu_0, initial_L_inv=L_0, initial_pi=pi_0,
                context=context,
            )

            mu_w = final_mu[None, :, :]     # (1, K, D)
            L_w  = final_L[None, :, :, :]    # (1, K, D, D)
            pi_w = final_pi[None, :]          # (1, K)

            x_best, cost_best = _best_x_from_single_block(
                mu_w, cost_fn, context, dims,
            )

            return SolverResult(
                x=x_best, cost=float(cost_best),
                mu=mu_w, S_or_L=L_w, pi=pi_w,
                solver_name="reuse_single",
            )

    # ══════════════════════════════════════════════════════════════════
    # reuse_multi (S precision, multi‑block, uses pi directly)
    # ══════════════════════════════════════════════════════════════════
    elif solver == "reuse_multi":
        def _solve(key, *, context=None,
                   initial_mu=None, initial_S_or_L=None, initial_pi=None,
                   warm_start=None):
            mu_k = initial_mu
            S_k  = initial_S_or_L
            pi_k = initial_pi
            if warm_start is not None:
                if mu_k is None:  mu_k = warm_start.mu
                if S_k is None:   S_k  = warm_start.S_or_L
                if pi_k is None:  pi_k = warm_start.pi
            if mu_k is None:  mu_k = _default_mu(key)
            if S_k is None:   S_k  = _default_S()
            if pi_k is None:  pi_k = _default_pi()

            final_mu, final_S, final_pi = _impl(
                key=key, T=T, dt=dt, M=M, K=K,
                B_n=B_n, B_o=B_o,
                a_threshold=a_threshold, T_0=T_0,
                dims=dims, fitness_fn=cost_fn,
                initial_mu=mu_k, initial_S=S_k, initial_pi=pi_k,
                context=context,
            )

            x_best, cost_best = _best_x(final_mu, cost_fn, context, M, dims)

            return SolverResult(
                x=x_best, cost=float(cost_best),
                mu=final_mu, S_or_L=final_S, pi=final_pi,
                solver_name="reuse_multi",
            )

    else:
        raise RuntimeError(f"Unreachable: {solver}")

    return _solve


# ══════════════════════════════════════════════════════════════════════
# Shortcut: one‑shot solve
# ══════════════════════════════════════════════════════════════════════

def solve(
        objective_fn: Callable,
        dims: Tuple[int, ...],
    key=None,
    *,
    context=None,
    **kwargs,
) -> SolverResult:
    """One‑shot: build a solver and run it once.

    Parameters
    ----------
    objective_fn, dims, **kwargs :
        Identical to ``build_solver()``.

    key : PRNGKey, optional
        Auto‑generated from seed 0 if None.

    context : anything, optional
        Passed as the second argument to *objective_fn*.

    Returns
    -------
    SolverResult
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    solver_fn = build_solver(objective_fn, dims, **kwargs)
    return solver_fn(key, context=context)


# ══════════════════════════════════════════════════════════════════════
# Multi‑agent Nash equilibrium solver
# ══════════════════════════════════════════════════════════════════════

# ── Nash result type ──────────────────────────────────────────────────

@dataclass
class NashResult:
    """Returned by ``build_nash_solver()``.

    Contains per‑agent solutions that together form a Nash equilibrium.
    """
    solutions: Dict[int, SolverResult]   # agent_id → solution
    joint_x: jnp.ndarray                  # concatenated joint decision vector
    per_agent_cost: jnp.ndarray           # (M_agent,)  real evaluated costs
    solver_name: str
    diag: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [f"NashResult(solver={self.solver_name!r}"]
        for aid, sr in sorted(self.solutions.items()):
            parts.append(f"  agent {aid}: cost={sr.cost:.4f}")
        if self.per_agent_cost is not None:
            parts.append(f"  per_agent_cost={self.per_agent_cost}")
        parts.append(")")
        return "\n".join(parts)


def build_nash_solver(
    agent_specs: Dict[int, Tuple[Callable, Optional[Sequence]]],
    dims: Tuple[int, ...],
    *,
    # ── solver selection ──
    solver: str = "auto",
    block_to_agent: Optional[Sequence[int]] = None,
    # ── Constran params ──
    k_inner: float = 0.1,
    obj_transform: str = "standard",
    # ── solver hyper‑parameters ──
    T: int = 300,
    dt: float = 0.1,
    K: int = 3,
    T_0: int = 50,
    B: int = 80,
    B0: int = 20,
    M_inner: int = 30,
    # ── initialisation ──
    init_mu_range: Tuple[float, float] = (-4.0, 4.0),
    init_precision_scale: float = 4.0,
) -> Callable:
    """Build a Nash equilibrium solver for multi‑agent games.

    Parameters
    ----------
    agent_specs : dict
        ``{agent_id: (objective_fn, constraints)}``.
        Same format as ``Constran.build_multi_agent()``.
        Each *objective_fn* is ``(x, ctx) -> scalar`` evaluating the cost
        for that agent given a joint decision vector *x*.

    dims : tuple of int
        Block dimensions.  With *block_to_agent* this can be any number
        of blocks; without it, ``len(dims) == len(agent_specs)`` and each
        agent gets one block.

    solver : str
        ``"auto"`` | ``"rne"`` | ``"rne_blocks"``.
        ``"auto"`` picks ``"rne"`` when each agent has one block,
        ``"rne_blocks"`` otherwise.

    block_to_agent : list of int, optional
        Maps block index → agent_id.  Required when agents have multiple
        blocks.  Length must equal ``len(dims)``.

    M_inner : int
        Number of Monte‑Carlo opponent samples per Nash inner loop.

    k_inner, obj_transform :
        Forwarded to ``Constran.build_multi_agent()``.

    T, dt, K, T_0, B, B0 :
        Same as ``build_solver()``.

    Returns
    -------
    solver_fn : callable
        ``solver_fn(key, *, context=None, warm_start=None) -> NashResult``.
    """
    # ── validate ────────────────────────────────────────────────────
    _validate_dims(dims)
    N_blocks = len(dims)
    M_agent = len(agent_specs)
    D_max = max(dims)

    if block_to_agent is None:
        if N_blocks != M_agent:
            raise ValueError(
                f"When block_to_agent is None, len(dims) ({N_blocks}) must "
                f"equal len(agent_specs) ({M_agent}). "
                f"Otherwise provide block_to_agent explicitly."
            )
        block_to_agent = list(range(M_agent))
    else:
        if len(block_to_agent) != N_blocks:
            raise ValueError(
                f"len(block_to_agent) ({len(block_to_agent)}) != "
                f"len(dims) ({N_blocks})"
            )
    block_to_agent_arr = jnp.array(block_to_agent)

    # ── pick solver ──────────────────────────────────────────────────
    blocks_per_agent = [sum(1 for b in block_to_agent if b == aid)
                        for aid in range(M_agent)]
    all_single_block = all(b == 1 for b in blocks_per_agent)

    if solver == "auto":
        solver = "rne" if all_single_block else "rne_blocks"

    if solver == "rne":
        _impl = _get_nash_rne()
        _uses_fitness_list = False
    elif solver == "rne_blocks":
        _impl = _get_nash_rne_blocks()
        _uses_fitness_list = False
    elif solver in ("rne_single", "rne_single_verbose"):
        _impl = (_get_nash_rne_single_verbose() if solver == "rne_single_verbose"
                 else _get_nash_rne_single())
        _uses_fitness_list = True
    else:
        raise ValueError(
            f"Unknown solver {solver!r}. "
            f"Choose: 'auto', 'rne', 'rne_single', 'rne_single_verbose', 'rne_blocks'."
        )

    # ── build per‑agent cost functions (via Constran) ────────────────
    from Constraintdealer.Constran import build_multi_agent as constran_build_ma

    agent_fns = constran_build_ma(
        agent_specs,
        k_inner=k_inner,
        obj_transform=obj_transform,
    )
    _agent_fn_list = [agent_fns[i] for i in range(M_agent)]

    # ── build solver function ────────────────────────────────────────
    def _solve(key, *, context=None,
               initial_mu=None, initial_S_or_L=None, initial_pi=None,
               warm_start=None):
        # Resolve params: explicit > warm_start > default
        mu_k = initial_mu
        L_k  = initial_S_or_L
        pi_k = initial_pi
        v_k  = None
        if warm_start is not None:
            ws = warm_start if isinstance(warm_start, dict) else warm_start.__dict__
            if mu_k is None:  mu_k = ws.get("mu", ws.get("initial_mu"))
            if L_k is None:   L_k  = ws.get("L_inv", ws.get("S_or_L"))
            if pi_k is None:  pi_k = ws.get("pi")
            v_k = ws.get("v")
        if pi_k is not None and v_k is None:
            v_k = jnp.log(pi_k[..., :-1] / (pi_k[..., -1:] + 1e-10))
        if mu_k is None:
            mu_k = jax.random.uniform(
                key, (N_blocks, K, D_max),
                minval=init_mu_range[0], maxval=init_mu_range[1])
        if L_k is None:
            L_k = jnp.tile(
                jnp.eye(D_max) * jnp.sqrt(init_precision_scale),
                (N_blocks, K, 1, 1))
        if v_k is None:
            v_k = jnp.zeros((N_blocks, K - 1))

        # Build fitness dispatcher
        if _uses_fitness_list:
            # MPC_G_S / MPC_G_S_V: list of (agent_idx, x, ctx) -> scalar
            fitness_fns = [
                (lambda i=i: lambda aid, x, ctx_: _agent_fn_list[i](i, x, ctx_))()
                for i in range(M_agent)
            ]
            fitness_fn_j = None
        else:
            # MPC_G / MPC_G_MS: single callable(agent_idx, x, ctx)
            def fitness_fn_j_(aid, x, ctx_):
                branches = [
                    (lambda i=i: _agent_fn_list[i](i, x, ctx_))
                    for i in range(M_agent)
                ]
                return lax.switch(jnp.clip(aid, 0, M_agent - 1), branches)
            fitness_fn_j = fitness_fn_j_
            fitness_fns = None

        # Call solver
        if solver == "rne":
            final_mu, final_L, final_pi = _impl(
                key=key, T=T, dt=dt, M_agent=M_agent, K=K,
                B=B, B0=B0, dims=dims, T_0=T_0,
                fitness_fn_j=fitness_fn_j,
                initial_mu_k=mu_k, initial_L_inv_k=L_k,
                context=context, M_inner=M_inner,
            )
            final_v = None
        elif solver == "rne_blocks":
            final_mu, final_L, final_pi, _metrics = _impl(
                key=key, T=T, dt=dt,
                N_blocks=N_blocks, M_agent=M_agent, K=K,
                B=B, B0=B0, dims=dims, T_0=T_0,
                fitness_fn_j=fitness_fn_j,
                initial_mu_k=mu_k, initial_L_inv_k=L_k,
                context=context, M_inner=M_inner,
                block_to_agent_idx=block_to_agent_arr,
                initial_v_k=v_k,
            )
            final_v = v_k  # for warm_start tracking
        else:  # rne_single / rne_single_verbose
            result = _impl(
                key=key, T=T, dt=dt, M_agent=M_agent, K=K,
                B=B, B0=B0, dims=dims, T_0=T_0,
                fitness_fns=tuple(fitness_fns),
                initial_mu_k=mu_k, initial_L_inv_k=L_k,
                context=context, M_inner=M_inner,
            )
            if solver == "rne_single_verbose":
                final_mu, final_L, final_pi, _metrics = result
            else:
                final_mu, final_L, final_pi = result
            final_v = None

        # ── build joint_x and evaluate real per‑agent costs ──────────
        solutions = {}
        joint_parts = []
        for aid in range(M_agent):
            my_blocks = [b for b in range(N_blocks)
                         if block_to_agent[b] == aid]
            blk_arr = jnp.array(my_blocks)
            mu_agent = final_mu[blk_arr]
            L_agent  = final_L[blk_arr]
            pi_agent = final_pi[blk_arr]

            best_k = jnp.argmax(pi_agent, axis=1)
            x_parts = [mu_agent[j, best_k[j], :dims[my_blocks[j]]]
                       for j in range(len(my_blocks))]
            x_agent = jnp.concatenate(x_parts) if x_parts else jnp.zeros(0)
            joint_parts.append(x_agent)

            solutions[aid] = SolverResult(
                x=x_agent, cost=0.0,   # filled below
                mu=mu_agent, S_or_L=L_agent, pi=pi_agent,
                solver_name=f"nash_{solver}_agent{aid}",
            )

        joint_x = jnp.concatenate(joint_parts)

        # Evaluate real per-agent costs on joint_x
        per_agent_cost = jnp.zeros(M_agent)
        for aid in range(M_agent):
            c = _agent_fn_list[aid](aid, joint_x, context)
            per_agent_cost = per_agent_cost.at[aid].set(c)
            solutions[aid] = SolverResult(
                x=solutions[aid].x, cost=float(c),
                mu=solutions[aid].mu, S_or_L=solutions[aid].S_or_L,
                pi=solutions[aid].pi,
                solver_name=solutions[aid].solver_name,
            )

        return NashResult(
            solutions=solutions,
            joint_x=joint_x,
            per_agent_cost=per_agent_cost,
            solver_name=solver,
        )

    return _solve


# ── quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Running directly needs absolute imports
    import sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    # Re-import ourselves with the fixed path
    from gmm_igo.solver_builder import build_solver, build_nash_solver

    import jax.numpy as jnp

    def quad(x, ctx):
        return (x[0] - 3.0)**2 + (x[1] + 2.0)**2

    print("── M22 (auto) ──")
    s = build_solver(quad, (2,), T=200)
    r = s(jax.random.PRNGKey(42))
    print(r)

    print("\n── M23 ──")
    s = build_solver(quad, (2,), solver="m23", T=200, nu_r=0.5)
    r = s(jax.random.PRNGKey(42))
    print(r)

    print("\n── reuse_single + reuse_multi ──")
    for name in ["reuse_single", "reuse_multi"]:
        s = build_solver(quad, (2,), solver=name, T=200)
        r = s(jax.random.PRNGKey(42))
        print(f"  {name}: {r}")

    print("\n── Per-call initial_mu override ──")
    s = build_solver(quad, (2,), T=100)
    custom_mu = jnp.ones((1, 3, 2)) * 2.5
    r1 = s(jax.random.PRNGKey(42), initial_mu=custom_mu)
    print(f"  with custom mu: cost={r1.cost:.6f}")
    r2 = s(jax.random.PRNGKey(42), warm_start=r1)
    print(f"  then warm_start: cost={r2.cost:.8f}")

    print("\n── Multi-block M22 (dims=(1,1)) ──")
    s = build_solver(quad, (1, 1), solver="m22", T=200)
    r = s(jax.random.PRNGKey(42))
    print(f"  {r}")

    print("\n── Nash rne_single ──")
    def o0(x, ctx): return (x[0]-3.0)**2
    def o1(x, ctx): return (x[1]+2.0)**2
    s = build_nash_solver({0:(o0,[]), 1:(o1,[])}, dims=(1,1),
                           solver="rne_single", T=150, M_inner=10)
    r = s(jax.random.PRNGKey(42))
    print(f"  costs={r.per_agent_cost}")
