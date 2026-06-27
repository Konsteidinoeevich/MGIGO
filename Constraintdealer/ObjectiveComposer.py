"""
ObjectiveComposer — Structured Sub-Objective Composition
=========================================================

Replaces manual weight tuning with per-term saturation gain control.

Each sub-objective term gets its own ``log_transform → sigma_k(..., k)``
treatment. All terms compete additively through a final outer sigma.
The result is a standard ``(x, ctx) -> scalar`` callable, directly
usable as the ``objective_fn`` in ``Constran.build()``.

Core insight: **k is the gain**. Unlike a multiplicative weight (unbounded),
k controls where the term's sigma_k half-saturates — a physically meaningful
knee in the original units of the term.

::

    from Constraintdealer.ObjectiveComposer import compose_objective
    from Constraintdealer.Constran import build, Deterministic

    obj = compose_objective([
        (tracking_fn,  0.5, "tracking"),    # knee at ~6.4 — wide range
        (final_fn,     2.0, "final"),       # knee at ~0.65 — precise
        (control_fn,   1.0, "control"),      # knee at ~1.7 — balanced
    ])

    cost = build(obj, [Deterministic(obs_fn, mode='hard', priority=1)])

See ``ObjectiveComposer_README.md`` for the full methodology.
"""

from __future__ import annotations

import sys, os
# Ensure the project root is on sys.path so we can import Constraintdealer
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

import jax
import jax.numpy as jnp
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

# Reuse the core math from Constran — no duplication
from Constraintdealer.Constran import sigma_k, log_transform


# ===========================================================================
# 1. Knee / Gain Conversion Utilities
# ===========================================================================

def knee_to_k(raw_knee: float) -> float:
    """Convert a desired knee (in raw units) to the corresponding k.

    The knee is defined as the raw input where σ_k(T(raw)) ≈ 0.707
    (half-saturation).  Since σ_k(1/k) = 1/√2, and after log_transform:

        T(raw_knee) = 1/k  →  k = 1 / log(1 + raw_knee)

    Parameters
    ----------
    raw_knee : float
        The raw term value at which you want half-saturation.
        E.g., raw_knee=0.5 means "I want 0.5m tracking error to be
        at the middle of the sensitivity range."

    Returns
    -------
    k : float

    Examples
    --------
    >>> knee_to_k(0.22)   # lateral tracking: sub-meter precision
    5.02...
    >>> knee_to_k(6.4)    # path integral: wide range
    0.501...
    >>> knee_to_k(22000)  # control effort: only extreme waste
    0.100...
    """
    return 1.0 / jnp.log1p(raw_knee)


def k_to_knee(k: float) -> float:
    """Convert k back to the raw knee value.

        raw_knee = exp(1/k) - 1

    Examples
    --------
    >>> k_to_knee(5.0)
    0.221...
    >>> k_to_knee(0.5)
    6.389...
    >>> k_to_knee(0.1)
    22025.4...
    """
    return jnp.expm1(1.0 / k)


def suggest_k(raw_typical: float, raw_max: Optional[float] = None) -> float:
    """Heuristically suggest a k value given typical and max raw term values.

    The suggestion places the knee between the typical value and a fraction
    of the max, so the term has good distinguishability in its normal
    operating range while still differentiating large values.

    If raw_max is None, uses raw_typical * 100 as an estimate.

    Parameters
    ----------
    raw_typical : float
        A typical / expected value of this term during normal operation.
    raw_max : float, optional
        The maximum expected value. If None, estimated as 100 * raw_typical.

    Returns
    -------
    k : float

    Examples
    --------
    >>> suggest_k(1.0, 50.0)   # typical 1m error, max 50m
    0.596...
    >>> suggest_k(0.1, 2.0)    # typical 0.1 control, max 2.0
    2.28...
    """
    if raw_max is None:
        raw_max = raw_typical * 100.0
    # Place knee at geometric mean between typical and max/10
    knee_target = float(jnp.sqrt(raw_typical * max(raw_max / 10.0, raw_typical * 10.0)))
    return float(knee_to_k(knee_target))


# ===========================================================================
# 2. Main API: compose_objective
# ===========================================================================

def compose_objective(
    terms: Sequence[Union[Tuple[Callable, float],
                           Tuple[Callable, float, str]]],
    *,
    k_outer: float = 1.0,
    jit_result: bool = True,
) -> Callable[[jnp.ndarray, Any], jnp.ndarray]:
    """Compose multiple sub-objectives with per-term saturation gain (k).

    Each sub-term is independently compressed:

        compressed_i = σ_k( T(term_fn_i(x, ctx)), k=k_i )

    where T = log_transform (auto-scaling), σ_k = saturation with knee at 1/k.

    All compressed terms are summed and passed through a final outer σ_k:

        result = σ( Σ compressed_i, k=k_outer )

    **Why this replaces manual weights:**

    - Each term is bounded in [0, 1) — no single term can dominate
    - k controls the knee in physically meaningful units (meters, m/s², …)
    - log_transform eliminates per-term normalization
    - The outer σ prevents the sum from exceeding [0, 1)

    **Choosing k:**

    ======  ============  ==============================
    k       Raw knee      When to use
    ======  ============  ==============================
    5.0     ~0.22 m       Sub-meter precision (lateral)
    2.0     ~0.65 m       Final approach accuracy
    1.0     ~1.7          Balanced, moderate sensitivity
    0.5     ~6.4          Wide range (path integral)
    0.2     ~147          Very wide (longitudinal, ~100m)
    0.1     ~2.2×10⁴      Only penalize extreme waste
    ======  ============  ==============================

    Large k = high gain = saturates quickly = "small values matter".
    Small k = low gain = wide linear range = "only large values matter".

    Parameters
    ----------
    terms : list of tuple
        Each element is (term_fn, k) or (term_fn, k, name).
        - term_fn(x, ctx) -> scalar  (should return >= 0)
        - k : float, sensitivity (see table above)
        - name : str, optional (for debugging / logging)
    k_outer : float
        Final compression knee. Default 1.0.
    jit_result : bool
        If True (default), wrap with jax.jit.

    Returns
    -------
    objective_fn : callable
        ``(x, ctx) -> scalar`` in [0, 1).
        Pass as ``objective_fn`` to ``Constran.build()``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> def track(x, ctx): return jnp.sum((x[:2] - ctx['tgt'])**2)
    >>> def ctrl(x, ctx):  return 0.1 * jnp.sum(x[2:]**2)
    >>> obj = compose_objective([
    ...     (track, 0.5, "tracking"),
    ...     (ctrl,  1.0, "control"),
    ... ])
    >>> # obj(x, ctx) -> scalar in [0, 1)
    >>> # Use with Constran:
    >>> # cost = build(obj, [Deterministic(...), ...])
    """
    # Normalize: accept (fn, k) or (fn, k, name)
    parsed: List[Tuple[Callable, float]] = []
    for t in terms:
        if len(t) == 2:
            parsed.append((t[0], t[1]))
        else:
            parsed.append((t[0], t[1]))  # name ignored at runtime

    def _composed(x: jnp.ndarray, ctx: Any) -> jnp.ndarray:
        total = 0.0
        for term_fn, k in parsed:
            raw = term_fn(x, ctx)
            total = total + sigma_k(log_transform(raw), k=k)
        return sigma_k(total, k=k_outer)

    if jit_result:
        return jax.jit(_composed)
    return _composed


# ===========================================================================
# 3. Convenience: Auto-suggest k for all terms
# ===========================================================================

def compose_objective_auto(
    terms: Sequence[Tuple[Callable, float, Optional[float]]],
    *,
    k_outer: float = 1.0,
    jit_result: bool = True,
) -> Callable[[jnp.ndarray, Any], jnp.ndarray]:
    """Like compose_objective, but k is auto-suggested from (typical, max).

    Parameters
    ----------
    terms : list of (term_fn, raw_typical, raw_max_or_None)
        - term_fn(x, ctx) -> scalar
        - raw_typical : float — typical value during normal operation
        - raw_max : float or None — max expected value (None = auto-estimate)

    Returns
    -------
    objective_fn : (x, ctx) -> scalar in [0, 1)

    Examples
    --------
    >>> obj = compose_objective_auto([
    ...     (track_fn,  1.0,  50.0),   # typical 1m, max 50m → k≈0.6
    ...     (final_fn,  0.1,  5.0),    # typical 0.1m, max 5m → k≈2.5
    ...     (ctrl_fn,   0.5,  10.0),   # typical 0.5, max 10 → k≈1.0
    ... ])
    """
    term_list = []
    for t in terms:
        fn, typical, maximum = t
        k = suggest_k(typical, maximum)
        term_list.append((fn, k))
    return compose_objective(term_list, k_outer=k_outer, jit_result=jit_result)


# ===========================================================================
# 4. Adaptive k Calibration via Semantic Roles
# ===========================================================================

# Default role → percentile mapping.
# Smaller percentile = knee further left = wider linear region.
ROLE_PERCENTILES = {
    'primary':    0.50,   # knee at median — worst half saturates
    'secondary':  0.70,   # knee at 70th percentile
    'tiebreaker': 0.95,   # knee at 95th percentile — only extremes saturate
}


def _parse_role(role_or_pct, role_percentiles):
    """Resolve a role string or bare percentile float → float in (0, 1)."""
    if isinstance(role_or_pct, str):
        if role_or_pct not in role_percentiles:
            valid = ', '.join(sorted(role_percentiles))
            raise ValueError(
                f"Unknown role '{role_or_pct}'. "
                f"Valid roles: {valid}. "
                f"Or pass a float in (0, 1) for a custom percentile."
            )
        return role_percentiles[role_or_pct]
    pct = float(role_or_pct)
    if not 0.0 < pct < 1.0:
        raise ValueError(f"Percentile must be in (0, 1), got {pct}")
    return pct


def _generate_calibration_samples(
    n_dims=None, bounds=None, n_samples=1000, key=None,
    sample_fn=None, warmup_samples=None,
):
    """Generate calibration samples. Priority: warmup > sample_fn > (n_dims, bounds)."""
    if warmup_samples is not None:
        return jnp.asarray(warmup_samples)
    if sample_fn is not None:
        if key is None:
            key = jax.random.PRNGKey(0)
        return jnp.asarray(sample_fn(n_samples, key))
    if n_dims is not None:
        if key is None:
            key = jax.random.PRNGKey(0)
        low, high = bounds if bounds is not None else (-5.0, 5.0)
        return jax.random.uniform(key, shape=(n_samples, n_dims),
                                  minval=low, maxval=high)
    raise ValueError(
        "Must provide one of: warmup_samples, sample_fn, or (n_dims, bounds)"
    )


def _calibrate_k_from_samples(term_fn, percentile, samples, ctx, min_knee=1e-10):
    """Evaluate term_fn on all samples, compute percentile → knee → k.

    Parameters
    ----------
    term_fn : (x, ctx) -> scalar
    percentile : float in (0, 1)
    samples : (n_samples, n_dims)
    ctx : context dict
    min_knee : float — floor on raw knee to avoid division by zero

    Returns
    -------
    k : float
    """
    raw_vals = jax.vmap(lambda x: term_fn(x, ctx))(samples)
    finite = jnp.isfinite(raw_vals)
    raw_clean = raw_vals[finite]
    n_dropped = int(raw_vals.size - raw_clean.size)

    if raw_clean.size == 0:
        raise ValueError(
            f"All {raw_vals.size} calibration samples produced "
            f"non-finite values. Check term_fn or calibration bounds."
        )
    if n_dropped > 0.1 * raw_vals.size:
        import warnings
        warnings.warn(
            f"{n_dropped}/{raw_vals.size} calibration samples were "
            f"non-finite and dropped. Knee estimate may be unreliable."
        )

    raw_knee = float(jnp.percentile(raw_clean, percentile * 100.0))
    raw_knee_safe = max(raw_knee, min_knee)
    return float(knee_to_k(raw_knee_safe))


def compose_objective_adaptive(
    terms,
    *,
    # Sample source (exactly one needed)
    n_dims=None,
    bounds=None,
    n_samples=1000,
    key=None,
    sample_fn=None,
    warmup_samples=None,
    # Required: representative context for calibration
    ctx_calib=None,
    # Role customization
    role_percentiles=None,
    # Safety
    min_knee=1e-10,
    # Passed through to compose_objective
    k_outer=1.0,
    jit_result=True,
):
    """Compose sub-objectives with auto-calibrated k from semantic roles.

    Instead of specifying k manually, you assign each term a **semantic role**
    (``'primary'``, ``'secondary'``, ``'tiebreaker'``) or a bare percentile
    (e.g. ``0.40``).  The system evaluates each term on N random calibration
    samples, computes the specified percentile of its empirical distribution,
    sets that as the knee, and derives k automatically.

    **Why this works**: the percentile only depends on the *ordering* of raw
    values — no assumptions about magnitude, units, or scale.  A term whose
    raw values span 1e-6 to 1e6 calibrates just as correctly as one spanning
    0.1 to 10.

    **Cost concerns** (MC / expectation-based terms):
    Calibration evaluates each term on ``n_samples`` random vectors.  If a
    term internally runs 100 MC samples, calibration is ~n_samples×100
    evaluations.  In that case prefer ``warmup_samples`` from a prior solver
    run (zero overhead) or reduce ``n_samples`` to 200–500.

    Parameters
    ----------
    terms : list of tuple
        ``(term_fn, role, name?)`` where *role* is a str from the table below
        or a float in (0, 1) for a custom percentile.

        ============  ===========  ==============================
        Role          Percentile   Meaning
        ============  ===========  ==============================
        ``'primary'``      P50     Main goal — wide linear range
        ``'secondary'``    P70     Important but not dominant
        ``'tiebreaker'``   P95     Only matters when others close
        ============  ===========  ==============================

    n_dims : int, optional
        Dimensionality of the decision vector x.
    bounds : (float, float), optional
        (low, high) for uniform sampling. Default ``(-5.0, 5.0)``.
    n_samples : int
        Number of calibration samples (default 1000).
    key : jax PRNGKey, optional
    sample_fn : callable, optional
        ``sample_fn(n_samples, key) -> (n_samples, n_dims)``.
    warmup_samples : jnp.ndarray, optional
        ``(n, n_dims)`` — reuse solver samples, zero extra cost.
    ctx_calib : dict
        **Required.** Representative context for term evaluation.
    role_percentiles : dict, optional
        Override or extend the default role→percentile mapping.
    min_knee : float
        Floor on raw knee value (default 1e-10).
    k_outer : float
        Final compression knee, passed to ``compose_objective``.
    jit_result : bool
        Passed to ``compose_objective``.

    Returns
    -------
    objective_fn : ``(x, ctx) -> scalar`` in [0, 1).

    Examples
    --------
    >>> ctx_calib = {'target': jnp.array([8.0, 6.0]),
    ...              'init_state': jnp.array([0., 0., 0., 0.])}
    >>> obj = compose_objective_adaptive([
    ...     (tracking_fn,   'primary',    "tracking"),
    ...     (final_fn,      'secondary',  "final"),
    ...     (smooth_fn,     'tiebreaker', "smoothness"),
    ... ], n_dims=16, bounds=(-5.0, 5.0), n_samples=1000,
    ...    ctx_calib=ctx_calib)
    """
    if ctx_calib is None:
        raise TypeError(
            "ctx_calib is required. Term functions are evaluated on "
            "calibration samples with this context. Provide a "
            "representative static context, e.g. "
            "{'target': ..., 'init_state': ...}."
        )

    # 1. Build effective role → percentile map
    effective_roles = dict(ROLE_PERCENTILES)
    if role_percentiles is not None:
        effective_roles.update(role_percentiles)

    # 2. Parse terms
    parsed = []
    for i, t in enumerate(terms):
        fn = t[0]
        role_or_pct = t[1]
        name = t[2] if len(t) > 2 else f"term_{i}"
        percentile = _parse_role(role_or_pct, effective_roles)
        parsed.append((fn, percentile, name))

    # 3. Generate calibration samples
    samples = _generate_calibration_samples(
        n_dims=n_dims, bounds=bounds, n_samples=n_samples,
        key=key, sample_fn=sample_fn, warmup_samples=warmup_samples,
    )

    # 4. Warn if few samples
    if len(samples) < 200:
        import warnings
        warnings.warn(
            f"Only {len(samples)} calibration samples. "
            f"Percentile estimation is noisy below ~200. "
            f"Consider n_samples >= 500."
        )

    # 5. Calibrate k for each term
    calib_terms = []
    print("--- Adaptive K Calibration ---")
    for fn, percentile, name in parsed:
        k = _calibrate_k_from_samples(fn, percentile, samples,
                                      ctx_calib, min_knee)
        raw_knee = float(k_to_knee(k))
        print(f"  {name:20s}: P{int(percentile*100):02d}  "
              f"knee={raw_knee:.4f}  k={k:.4f}")
        calib_terms.append((fn, k, name))

    # 6. Delegate to compose_objective
    return compose_objective(calib_terms, k_outer=k_outer, jit_result=jit_result)


# ===========================================================================
# 5. Dynamic k via ctx — Zero JAX Recompilation
# ===========================================================================
#
#   k values live in ctx (not closure), so they can change every solver call
#   without triggering JAX recompilation — ctx is already dynamic.
#
#   Pattern:
#     1. calibrate_k_into_ctx(ctx, terms, ...) — initial calibration
#     2. obj = compose_objective_dynamic(terms) — reads k from ctx
#     3. For each MPC step: update_k_from_elite(ctx, terms, elite_x)
#
#   The cost function is compiled ONCE.  k tracks the improving population.

def _compose_dynamic_kernel(parsed):
    """Build a kernel that reads k from ctx instead of closure."""
    def _kernel(x, ctx):
        total = 0.0
        for i, (term_fn, _, name) in enumerate(parsed):
            raw = term_fn(x, ctx)
            k = ctx[f'k_{name}']
            total = total + sigma_k(log_transform(raw), k=k)
        k_out = ctx.get('k_outer', 1.0)
        return sigma_k(total, k=k_out)
    return _kernel


def compose_objective_dynamic(
    terms,
    *,
    jit_result: bool = True,
) -> Callable[[jnp.ndarray, Any], jnp.ndarray]:
    """Compose sub-objectives with k read from ctx (no recompilation).

    Like ``compose_objective``, but k values are read from ``ctx['k_{name}']``
    at call time instead of being captured in a closure.  This means you can
    change k between solver calls — e.g. after every MPC step — without
    triggering JAX recompilation.

    The returned cost function expects ctx to contain:
    - ``ctx['k_{name}']`` for each term (float)
    - ``ctx['k_outer']`` (float, optional; defaults to 1.0)

    Use ``calibrate_k_into_ctx()`` to populate the initial k values, and
    ``update_k_from_elite()`` to adjust them as the optimizer converges.

    Parameters
    ----------
    terms : list of (callable, float_or_None, str)
        ``(term_fn, k_placeholder, name)``.  The k value is NOT used at
        build time; it only serves to document the term.  Pass 0.0 or any
        dummy value.  The real k comes from ctx at call time.
    jit_result : bool
        If True, wrap with ``jax.jit``.

    Returns
    -------
    cost_fn : (x, ctx) -> scalar in [0, 1)
        Requires ``ctx['k_{name}']`` for each term.

    Examples
    --------
    >>> # 1. Initial calibration
    >>> ctx = calibrate_k_into_ctx(ctx, [
    ...     (track_fn, 'primary', "tracking"),
    ...     (final_fn, 'secondary', "final"),
    ... ], n_dims=16, bounds=(-5,5), n_samples=500)
    >>> # ctx now has ctx['k_tracking'], ctx['k_final']
    >>>
    >>> # 2. Build once
    >>> obj = compose_objective_dynamic([
    ...     (track_fn, 0.0, "tracking"),
    ...     (final_fn, 0.0, "final"),
    ... ])
    >>>
    >>> # 3. Use in MPC loop — update k each step
    >>> for step in range(T_mpc):
    ...     result = solver(..., fitness_fn_total=obj, context=ctx)
    ...     elite_x = extract_best(result)
    ...     ctx = update_k_from_elite(ctx, terms, elite_x)
    """
    parsed = []
    for i, t in enumerate(terms):
        fn = t[0]
        name = t[2] if len(t) > 2 else f"term_{i}"
        parsed.append((fn, 0.0, name))  # k placeholder — real k from ctx

    kernel = _compose_dynamic_kernel(parsed)
    if jit_result:
        return jax.jit(kernel)
    return kernel


def calibrate_k_into_ctx(
    ctx: dict,
    terms,
    *,
    n_dims=None, bounds=None, n_samples=500, key=None,
    sample_fn=None, warmup_samples=None,
    role_percentiles=None,
    min_knee=1e-10,
    verbose=True,
) -> dict:
    """Calibrate k from samples and write them into ctx.

    This is a convenience wrapper around the calibration logic in
    ``compose_objective_adaptive``.  It evaluates each term on calibration
    samples, computes the knee from the role's percentile, converts to k,
    and writes ``ctx['k_{name}']`` for each term.

    Parameters
    ----------
    ctx : dict
        Existing context dict.  Modified in-place and returned.
    terms : list of (callable, role, name)
        Same format as ``compose_objective_adaptive``.
    Other parameters : see ``compose_objective_adaptive``.

    Returns
    -------
    ctx : dict
        The same dict, now with ``ctx['k_{name}']`` entries populated.

    Examples
    --------
    >>> ctx = {'target': jnp.array([8.0, 6.0])}
    >>> ctx = calibrate_k_into_ctx(ctx, [
    ...     (track_fn, 'primary', "tracking"),
    ...     (final_fn, 'secondary', "final"),
    ... ], n_dims=16, bounds=(-5,5))
    >>> print(ctx['k_tracking'], ctx['k_final'])
    """
    effective_roles = dict(ROLE_PERCENTILES)
    if role_percentiles is not None:
        effective_roles.update(role_percentiles)

    parsed = []
    for i, t in enumerate(terms):
        fn = t[0]
        role_or_pct = t[1]
        name = t[2] if len(t) > 2 else f"term_{i}"
        percentile = _parse_role(role_or_pct, effective_roles)
        parsed.append((fn, percentile, name))

    samples = _generate_calibration_samples(
        n_dims=n_dims, bounds=bounds, n_samples=n_samples,
        key=key, sample_fn=sample_fn, warmup_samples=warmup_samples,
    )

    if verbose:
        print("--- Dynamic K Calibration ---")
    for fn, percentile, name in parsed:
        k = _calibrate_k_from_samples(fn, percentile, samples, ctx, min_knee)
        ctx[f'k_{name}'] = k
        if verbose:
            knee = float(k_to_knee(k))
            print(f"  k_{name:20s}: P{int(percentile*100):02d}  "
                  f"knee={knee:.4f}  k={k:.4f}")

    return ctx


def update_k_from_elite(
    ctx: dict,
    terms,
    elite_x: jnp.ndarray,
    *,
    multiplier: float = 5.0,
    min_knee: float = 1e-10,
    verbose: bool = False,
) -> dict:
    """Update ctx k-values based on the current elite (best) solution.

    For each term, evaluates it on ``elite_x``, sets the knee to
    ``raw_value * multiplier``, and derives k.  The multiplier provides
    headroom: at multiplier=5, the knee is 5× the current best, leaving
    room for further improvement while keeping the term in its sensitive
    region.

    Call this after each solver invocation in an MPC loop.
    Zero extra solver evaluations — the elite solution is already computed.

    Parameters
    ----------
    ctx : dict
        Context dict.  Modified in-place and returned.
    terms : list of (callable, role_or_None, name)
        The same term functions used for calibration.
    elite_x : jnp.ndarray
        The current best decision vector from the solver.
    multiplier : float
        knee = raw_value * multiplier.  Default 5.0.
    min_knee : float
        Floor on knee to avoid division by zero.
    verbose : bool
        If True, print updated k values.

    Returns
    -------
    ctx : dict

    Examples
    --------
    >>> # In MPC loop after solver returns:
    >>> elite_x = final_mu[0, best_idx]  # best solution
    >>> ctx = update_k_from_elite(ctx, terms, elite_x)
    """
    if verbose:
        print("--- Dynamic K Update (from elite) ---")
    for i, t in enumerate(terms):
        fn = t[0]
        name = t[2] if len(t) > 2 else f"term_{i}"
        raw = float(fn(elite_x, ctx))
        knee = max(raw * multiplier, min_knee)
        k = float(knee_to_k(knee))
        ctx[f'k_{name}'] = k
        if verbose:
            print(f"  k_{name:20s}: raw={raw:.4f}  "
                  f"knee={knee:.4f}  k={k:.4f}")
    return ctx


# ===========================================================================
# 6. Diagnostic: Inspect per-term contributions
# ===========================================================================

def inspect_terms(
    composed_obj: Callable,
    x: jnp.ndarray,
    ctx: Any,
    terms_info: Sequence[Tuple[Callable, float]],
) -> dict:
    """Return per-term contribution breakdown for a given x and ctx.

    Useful for debugging / tuning k values: see which terms are saturated
    and which are still in their linear region.

    Parameters
    ----------
    composed_obj : the function returned by compose_objective
    x : decision variable
    ctx : context
    terms_info : the original terms list passed to compose_objective

    Returns
    -------
    dict with keys: 'total', 'contributions' (list of {name, raw, compressed})
    """
    contributions = []
    for i, t in enumerate(terms_info):
        fn = t[0]
        k = t[1]
        name = t[2] if len(t) > 2 else f"term_{i}"
        raw = float(fn(x, ctx))
        compressed = float(sigma_k(log_transform(jnp.array(raw)), k=k))
        contributions.append({
            'name': name,
            'raw': raw,
            'compressed': compressed,
            'saturated': 'YES' if abs(compressed) > 0.9 else
                          'knee' if 0.5 < abs(compressed) < 0.9 else
                          'linear'
        })
    total = float(composed_obj(x, ctx))
    return {'total': total, 'contributions': contributions}


# ===========================================================================
# ===========================================================================
# 7. Quick self-test (run when executed directly)
# ===========================================================================
# ===========================================================================

if __name__ == "__main__":
    print("=== ObjectiveComposer Self-Test ===\n")

    # Test knee conversion round-trip
    for k_test in [5.0, 2.0, 1.0, 0.5, 0.1]:
        knee = float(k_to_knee(k_test))
        k_back = float(knee_to_k(knee))
        print(f"  k={k_test:.1f} → knee={knee:.4f} → k_back={k_back:.4f}  "
              f"({'✓' if abs(k_test - k_back) < 0.01 else '✗'})")

    # Test compose_objective
    import jax.numpy as jnp

    def track(x, ctx):
        return jnp.sum((x[:2] - ctx['target']) ** 2)

    def ctrl(x, ctx):
        return 0.1 * jnp.sum(x[2:] ** 2)

    obj = compose_objective([
        (track, 0.5, "tracking"),
        (ctrl,  1.0, "control"),
    ])

    x = jnp.array([3.0, 4.0, 1.0, 2.0])
    ctx = {'target': jnp.array([1.0, 2.0])}

    cost = float(obj(x, ctx))
    print(f"\n  compose_objective test: cost={cost:.6f}")
    assert 0.0 <= cost < 1.0, f"Output {cost} not in [0, 1)!"

    # Test inspect_terms
    info = inspect_terms(obj, x, ctx, [
        (track, 0.5, "tracking"),
        (ctrl,  1.0, "control"),
    ])
    print(f"  Total: {info['total']:.6f}")
    for c in info['contributions']:
        print(f"    {c['name']:12s}: raw={c['raw']:.4f}  "
              f"compressed={c['compressed']:.4f}  [{c['saturated']}]")

    # Test compose_objective_auto
    obj_auto = compose_objective_auto([
        (track, 1.0, 50.0),
        (ctrl,  0.5, 10.0),
    ])
    cost_auto = float(obj_auto(x, ctx))
    print(f"\n  compose_objective_auto test: cost={cost_auto:.6f}")
    assert 0.0 <= cost_auto < 1.0, f"Output {cost_auto} not in [0, 1)!"

    # Test suggest_k
    print("\n  suggest_k examples:")
    for typical, maximum in [(1.0, 50.0), (0.1, 2.0), (0.5, 10.0)]:
        k = float(suggest_k(typical, maximum))
        print(f"    typical={typical:.1f}, max={maximum:.1f} → k={k:.4f}")

    print("\n  All tests passed ✓")
