"""
IGO Plateau-Aware Elite Weighting for Float32 Precision
========================================================

When Constran cost values collapse below float32 resolution (~7 digits),
``jnp.argsort(jnp.argsort(f_vals))`` assigns **distinct ranks to bit-identical
values**, turning elite selection into arbitrary noise.  The solver then
diverges — especially visible with B-spline dense control points.

This module implements the correct IGO treatment from Ollivier (2017, §3.2):
tied (indistinguishable) samples receive the **same weight** — the selection
function averaged over the tied quantile interval.

Usage in truncation-selection solvers
-------------------------------------
Replace the two-line ranking pattern in your solver::

    # OLD (e.g. MPCsolverM22.py line 126-127)
    ranks = jnp.argsort(jnp.argsort(f_vals))
    w_hat = jnp.where(ranks < B0, 1.0/B, 0.0)

    # NEW
    from gmm_igo.igo_weights import compute_elite_weights
    w_hat = compute_elite_weights(f_vals, B0)

Affected solver files (not modified — documented for reference):

- ``MPCsolverM22.py``, ``MPCsolverM23.py``, ``MPCsolver.py``
- ``MPC_G.py``, ``MPC_G_S.py``, ``MPC_G_MS.py``, ``MPC_G_S_V.py``
- ``TSP.py``, ``Pure_discrete.py``, ``solverr.py``
- ``MPCresetweight.py``, ``MPCsolver_differentobjects.py``

Usage in reuse solvers
----------------------
::

    # OLD (MPC_R.py line 150-158)
    sort_indices = jnp.argsort(all_f)
    sorted_omega = omega[sort_indices]
    cumsum_omega = jnp.cumsum(sorted_omega)
    q_values = jnp.concatenate([jnp.zeros(1), cumsum_omega[:-1]]) / N_total
    sorted_is_selected = (q_values <= a_threshold).astype(jnp.float32)
    is_selected = jnp.zeros(N_total).at[sort_indices].set(sorted_is_selected)

    # NEW
    from gmm_igo.igo_weights import compute_tied_quantiles
    q_values = compute_tied_quantiles(all_f, omega)
    is_selected = (q_values <= a_threshold).astype(jnp.float32)

Runtime diagnostic
------------------
::

    from gmm_igo.igo_weights import check_distinguishability
    diag = check_distinguishability(f_vals)
    if not diag['ok']:
        print(f"WARNING: {diag['frac_tied']*100:.0f}% samples tied")

References
----------
- Ollivier, Y. et al. (2017). "Information-Geometric Optimization Algorithms:
  A Unifying Picture via Invariance Principles." JMLR 18(18):1-65.
- Akimoto, Y. & Hansen, N. (2019). "CMA-ES with restarts with handling
  of ties." GECCO 2019.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "compute_elite_weights",
    "compute_tied_quantiles",
    "check_distinguishability",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Core: IGO plateau-aware elite truncation weights
# ═══════════════════════════════════════════════════════════════════════════

def compute_elite_weights(
    f_vals: jnp.ndarray,
    B0: int,
    eps: float = 1e-7,
) -> jnp.ndarray:
    """IGO plateau-aware elite truncation weights.

    Replaces the standard ``argsort(argsort(f_vals))`` ranking with proper
    tie handling: samples whose cost values differ by less than *eps* are
    treated as indistinguishable and receive identical weights.

    Parameters
    ----------
    f_vals : (B,) ndarray
        Scalar cost values.  **Lower = better.**
    B0 : int
        Number of elite samples to select.  Must be in [0, B].
    eps : float
        Relative distinguishability threshold.  Internally scaled by
        ``max(|f_vals|)`` to track float32 ULP across magnitudes.
        Default ``1e-7`` (≈ float32 machine epsilon at magnitude 1.0).
        The Constran output range is ~[-1.414, 1.414], so this default
        is appropriate for most setups.

    Returns
    -------
    w_hat : (B,) ndarray
        Elite weights.  Sum equals ``B0 / B``.  Values are either ``0``,
        ``1/B``, or a fractional weight for tie groups straddling the B0
        boundary.

    Algorithm (Ollivier 2017, §3.2)
    --------------------------------
    1. Sort *f_vals* ascending.
    2. For each position *i*, compute via ``searchsorted``:

       - ``strict_rank[i]`` — count of samples with cost < f[i] - eps
       - ``non_strict_rank[i]`` — count of samples with cost <= f[i] + eps
       - ``group_size[i] = non_strict_rank[i] - strict_rank[i]`` (>= 1)

    3. Weight assignment (truncation selection w(u) = 1_{u <= B0/B}):

       - ``strict_rank >= B0`` → 0                         (entire group above threshold)
       - ``non_strict_rank <= B0`` → 1/B                    (entire group below threshold)
       - ``strict_rank < B0 < non_strict_rank`` →           (group straddles threshold)
         ``(B0 - strict_rank) / group_size * 1/B``

    4. Inverse-permute back to original sample order.

    When no values are within *eps* of each other (the common case with
    well-separated costs), every ``group_size == 1`` and ``strict_rank == i``,
    so the result is identical to the old ``argsort(argsort(f_vals))`` pattern.
    """
    B = f_vals.shape[0]
    # Note: no Python `if B0 <= 0` etc. — must stay JIT-compatible.
    # The general formula handles B0=0 (→ all strict_rank >= 0 → weight 0)
    # and B0>=B (→ all non_strict_rank <= B → weight 1/B) correctly.

    # Scale eps by the max magnitude so it tracks float32 ULP.
    # float32 ULP ≈ 1.19e-7 at |x|=1, ≈ 4.77e-7 at |x|=5, etc.
    mag = jnp.maximum(jnp.max(jnp.abs(f_vals)), 1e-12)
    scaled_eps = eps * mag

    # 1. Sort ascending (lower cost = better)
    sort_idx = jnp.argsort(f_vals)
    sorted_f = f_vals[sort_idx]

    # 2. Tie-group detection via searchsorted with ±eps window
    #    searchsorted(a, v, side='right') = first index where a[k] > v
    #    → strict_rank[i] = #elements strictly less than sorted_f[i] (allowing eps)
    strict_rank = jnp.searchsorted(sorted_f, sorted_f - scaled_eps, side='right')
    non_strict_rank = jnp.searchsorted(sorted_f, sorted_f + scaled_eps, side='right')
    group_size = non_strict_rank - strict_rank  # always >= 1

    # 3. IGO plateau-aware weight
    one_over_B = 1.0 / B
    weight_sorted = jnp.where(
        strict_rank >= B0,
        0.0,  # entirely above threshold
        jnp.where(
            non_strict_rank <= B0,
            one_over_B,  # entirely below threshold
            # Straddling boundary → fractional weight
            (B0 - strict_rank).astype(jnp.float32)
            / group_size.astype(jnp.float32)
            * one_over_B,
        ),
    )

    # 4. Map back to original sample order
    inv_sort = jnp.argsort(sort_idx)
    return weight_sorted[inv_sort]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Tied quantiles for importance-weighted reuse solvers
# ═══════════════════════════════════════════════════════════════════════════

def compute_tied_quantiles(
    f_vals: jnp.ndarray,
    sample_weights: jnp.ndarray,
    eps: float = 1e-7,
) -> jnp.ndarray:
    """Compute importance-weighted quantile values with tie averaging.

    For reuse solvers (MPC_R, MPC_Rblockwise) that select samples by
    thresholding cumulative importance-weighted quantiles.  When *f_vals*
    contain ties, the naive ``argsort`` + ``cumsum`` assigns different
    quantile values to tied samples.  This function returns **identical**
    quantile values for all members of a tie group (the start of the
    group's quantile interval), so they receive the same selection decision.

    Parameters
    ----------
    f_vals : (N,) ndarray
        Scalar cost values.  Lower = better.
    sample_weights : (N,) ndarray
        Importance weights (e.g. omega in MPC_R).  Must be non-negative.
    eps : float
        Distinguishability threshold (same semantics as
        :func:`compute_elite_weights`).

    Returns
    -------
    q_values : (N,) ndarray
        Quantile value (in [0, 1]) for each sample, in the **original**
        sample order.  Tie-group members share the same value.
    """
    N = f_vals.shape[0]
    total_w = jnp.maximum(jnp.sum(sample_weights), 1e-30)  # safe-div, JIT compatible

    # Scale eps by magnitude for float32 ULP tracking
    mag = jnp.maximum(jnp.max(jnp.abs(f_vals)), 1e-12)
    scaled_eps = eps * mag

    # Sort by cost
    sort_idx = jnp.argsort(f_vals)
    sorted_f = f_vals[sort_idx]
    sorted_w = sample_weights[sort_idx]

    # Detect tie groups
    strict_rank = jnp.searchsorted(sorted_f, sorted_f - scaled_eps, side='right')

    # Cumulative weights: cumsum_w[i] = sum_{k=0}^{i} sorted_w[k]
    cumsum_w = jnp.cumsum(sorted_w)

    # start_q_raw[k] = cumulative weight of first k elements (0 for k=0)
    # i.e. cumsum_w padded with a leading zero, dropping the last element
    start_q_raw = jnp.concatenate(
        [jnp.zeros(1, dtype=sample_weights.dtype), cumsum_w[:-1]]
    )

    # Each position i belongs to a tie group starting at strict_rank[i].
    # All members of the same group share the same strict_rank value,
    # so they get the same start_q (the group's quantile start).
    group_start_q_raw = start_q_raw[strict_rank]
    group_start_q = group_start_q_raw / total_w

    # Map back to original order
    inv_sort = jnp.argsort(sort_idx)
    return group_start_q[inv_sort]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Lightweight runtime diagnostic
# ═══════════════════════════════════════════════════════════════════════════

def check_distinguishability(
    f_vals: jnp.ndarray,
    eps: float = 1e-7,
) -> dict:
    """Lightweight diagnostic: how many samples are float32-indistinguishable.

    Can be called inside solver loops to monitor plateau width.

    Parameters
    ----------
    f_vals : (B,) ndarray
        Scalar cost values from the current iteration.
    eps : float
        Distinguishability threshold.

    Returns
    -------
    dict with keys:
        n_tied : int
            Number of samples tied with at least one other.
        max_tie_group : int
            Size of the largest tie group.
        frac_tied : float
            Fraction of samples in a tie group (0–1).
        ok : bool
            True if ``frac_tied <= 0.2`` (fewer than 20% tied).
    """
    B = f_vals.shape[0]
    sorted_f = jnp.sort(f_vals)

    # Adaptive eps (same as compute_elite_weights)
    mag = jnp.maximum(jnp.max(jnp.abs(f_vals)), 1e-12)
    scaled_eps = eps * mag

    strict_rank = jnp.searchsorted(sorted_f, sorted_f - scaled_eps, side='right')
    non_strict_rank = jnp.searchsorted(sorted_f, sorted_f + scaled_eps, side='right')
    group_sizes = non_strict_rank - strict_rank

    is_tied = group_sizes > 1
    n_tied = int(jnp.sum(is_tied))
    max_tie_group = int(jnp.max(group_sizes))
    frac_tied = n_tied / B
    ok = frac_tied <= 0.2

    return {
        'n_tied': n_tied,
        'max_tie_group': max_tie_group,
        'frac_tied': float(frac_tied),
        'ok': bool(ok),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Self-tests
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    failed = 0
    passed = 0

    def check(condition, label):
        global passed, failed
        if condition:
            passed += 1
            print(f"  ✓ {label}")
        else:
            failed += 1
            print(f"  ✗ FAIL: {label}")

    def section(title):
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")

    # ── Test A: compute_elite_weights ──────────────────────────────────

    section("Test A: compute_elite_weights")

    # A1 — No ties, standard case
    print("\n  A1: No ties — should match old argsort(argsort(f_vals))")
    f_a1 = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=jnp.float32)
    w_new = compute_elite_weights(f_a1, B0=2)
    ranks_old = jnp.argsort(jnp.argsort(f_a1))
    w_old = jnp.where(ranks_old < 2, 1.0 / 5, 0.0)
    check(jnp.allclose(w_new, w_old), "weights match old behavior")
    check(abs(float(jnp.sum(w_new)) - 0.4) < 1e-6, f"sum = {float(jnp.sum(w_new)):.6f} (expected 0.4)")
    print(f"    weights: {w_new}")

    # A2 — Exact ties, group entirely below B0
    print("\n  A2: Exact ties, group entirely below B0")
    f_a2 = jnp.array([0.1, 0.1, 0.1, 2.0, 5.0], dtype=jnp.float32)
    w_a2 = compute_elite_weights(f_a2, B0=3)
    check(abs(float(w_a2[0]) - 0.2) < 1e-6, "tied sample 0 weight = 0.2")
    check(abs(float(w_a2[1]) - 0.2) < 1e-6, "tied sample 1 weight = 0.2")
    check(abs(float(w_a2[2]) - 0.2) < 1e-6, "tied sample 2 weight = 0.2")
    check(abs(float(w_a2[3]) - 0.0) < 1e-6, "non-elite weight = 0")
    check(abs(float(jnp.sum(w_a2)) - 0.6) < 1e-6, f"sum = {float(jnp.sum(w_a2)):.6f} (expected 0.6)")
    print(f"    weights: {w_a2}")

    # A3 — Exact ties, group straddles B0 (THE KEY TEST)
    print("\n  A3: Exact ties, group straddles B0 boundary")
    f_a3 = jnp.array([0.1, 0.1, 0.1, 2.0, 5.0], dtype=jnp.float32)
    w_a3 = compute_elite_weights(f_a3, B0=2)
    expected_w = 2.0 / 15.0  # (B0 - 0) / 3 * 1/5 = 2/15 ≈ 0.133...
    check(abs(float(w_a3[0]) - expected_w) < 1e-6,
          f"tied sample 0 weight = {float(w_a3[0]):.6f} (expected {expected_w:.6f})")
    check(abs(float(w_a3[1]) - expected_w) < 1e-6,
          f"tied sample 1 weight = {float(w_a3[1]):.6f} (expected {expected_w:.6f})")
    check(abs(float(w_a3[2]) - expected_w) < 1e-6,
          f"tied sample 2 weight = {float(w_a3[2]):.6f} (expected {expected_w:.6f})")
    check(abs(float(w_a3[3]) - 0.0) < 1e-6, "non-elite weight = 0")
    check(abs(float(jnp.sum(w_a3)) - 0.4) < 1e-6, f"sum = {float(jnp.sum(w_a3)):.6f} (expected 0.4)")
    print(f"    weights: {w_a3}  (all three tied get {expected_w:.4f})")

    # A4 — Float32-near ties (within eps)
    print("\n  A4: Float32-near values within eps=1e-7")
    f_a4 = jnp.array([0.1, 0.1 + 5e-8, 0.1 + 9e-8, 2.0], dtype=jnp.float32)
    w_a4 = compute_elite_weights(f_a4, B0=2, eps=1e-7)
    # First three should be treated as tied (diffs 5e-8, 4e-8 < 1e-7)
    # strict_rank = 0 for all three, non_strict_rank = 3, group_size = 3
    # B0=2 straddles: weight = (2-0)/3 * 1/4 = 2/12 = 1/6 ≈ 0.1667
    expected_a4 = 1.0 / 6.0
    check(abs(float(w_a4[0]) - expected_a4) < 1e-6,
          f"near-tied 0 weight = {float(w_a4[0]):.6f} (expected {expected_a4:.6f})")
    check(abs(float(w_a4[1]) - expected_a4) < 1e-6,
          f"near-tied 1 weight = {float(w_a4[1]):.6f}")
    check(abs(float(w_a4[2]) - expected_a4) < 1e-6,
          f"near-tied 2 weight = {float(w_a4[2]):.6f}")
    check(abs(float(w_a4[3]) - 0.0) < 1e-6, "far sample weight = 0")
    print(f"    weights: {w_a4}")

    # A5 — Float32-near values across eps boundary
    print("\n  A5: Float32-near values across eps boundary")
    f_a5 = jnp.array([0.1, 0.1 + 2e-7, 2.0, 5.0], dtype=jnp.float32)
    w_a5 = compute_elite_weights(f_a5, B0=2, eps=1e-7)
    # Gap 2e-7 > eps=1e-7 → NOT tied → standard behavior
    ranks_a5 = jnp.argsort(jnp.argsort(f_a5))
    w_old_a5 = jnp.where(ranks_a5 < 2, 0.25, 0.0)
    check(jnp.allclose(w_a5, w_old_a5), "not tied — matches old behavior")
    print(f"    weights: {w_a5}")

    # A6 — All tied
    print("\n  A6: All samples tied")
    f_a6 = jnp.full((5,), 0.5, dtype=jnp.float32)
    w_a6 = compute_elite_weights(f_a6, B0=2)
    # B0=2, B=5, all 5 tied at strict_rank=0, non_strict_rank=5
    # weight = (2-0)/5 * 1/5 = 2/25 = 0.08
    expected_a6 = 2.0 / 25.0
    for i in range(5):
        check(abs(float(w_a6[i]) - expected_a6) < 1e-6,
              f"sample {i} weight = {float(w_a6[i]):.6f} (expected {expected_a6:.6f})")
    check(abs(float(jnp.sum(w_a6)) - 0.4) < 1e-6, f"sum = {float(jnp.sum(w_a6)):.6f} (expected 0.4)")
    print(f"    weights: {w_a6}  (all equal, each gets {expected_a6:.4f})")

    # A7 — Sum invariant across all cases
    print("\n  A7: Sum invariant (sum(weights) == B0/B)")
    for label, fv, b0, expected_sum in [
        ("no ties", f_a1, 2, 0.4),
        ("ties below", f_a2, 3, 0.6),
        ("ties straddle", f_a3, 2, 0.4),
        ("near ties", f_a4, 2, 0.5),
        ("all tied", f_a6, 2, 0.4),
        ("B0=0", f_a1, 0, 0.0),
        ("B0=B", f_a1, 5, 1.0),
    ]:
        s = float(jnp.sum(compute_elite_weights(fv, b0)))
        check(abs(s - expected_sum) < 1e-6,
              f"sum invariant [{label}]: {s:.6f} == {expected_sum}")

    # ── Test B: compute_tied_quantiles ─────────────────────────────────

    section("Test B: compute_tied_quantiles")

    # B1 — Ties with uniform weights
    print("\n  B1: Ties with uniform weights")
    f_b1 = jnp.array([0.1, 0.1, 0.1, 2.0, 5.0], dtype=jnp.float32)
    w_b1 = jnp.ones(5, dtype=jnp.float32)
    q = compute_tied_quantiles(f_b1, w_b1, eps=1e-7)
    # Tied group (3 samples at 0.1): all should have same q_value (start of group = 0)
    check(abs(float(q[0]) - float(q[1])) < 1e-6, "tied samples 0,1 share same q")
    check(abs(float(q[1]) - float(q[2])) < 1e-6, "tied samples 1,2 share same q")
    check(float(q[0]) < float(q[3]), "tied group q < non-tied sample q")
    print(f"    q_values: {q}")

    # B2 — No ties: should match old cumulative quantile behavior
    print("\n  B2: No ties — matches old cumulative quantile")
    f_b2 = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=jnp.float32)
    w_b2 = jnp.ones(5, dtype=jnp.float32)
    q_new = compute_tied_quantiles(f_b2, w_b2, eps=1e-7)

    # Old behavior
    sort_idx_old = jnp.argsort(f_b2)
    sorted_w_old = w_b2[sort_idx_old]
    cumsum_old = jnp.cumsum(sorted_w_old)
    q_old_sorted = jnp.concatenate([jnp.zeros(1), cumsum_old[:-1]]) / 5.0
    inv_old = jnp.argsort(sort_idx_old)
    q_old = q_old_sorted[inv_old]

    check(jnp.allclose(q_new, q_old), "no ties — matches old behavior")
    print(f"    q_new: {q_new}")
    print(f"    q_old: {q_old}")

    # ── Test C: JIT compatibility ──────────────────────────────────────

    section("Test C: JIT compatibility")

    import jax

    jit_elite = jax.jit(compute_elite_weights)
    jit_quant = jax.jit(compute_tied_quantiles)

    f_c = jnp.array([0.1, 0.1, 0.1, 2.0, 5.0], dtype=jnp.float32)
    w_c = jit_elite(f_c, 2)
    check(jnp.allclose(w_c, compute_elite_weights(f_c, 2)),
          "JIT compute_elite_weights matches eager")

    q_c = jit_quant(f_c, jnp.ones(5, dtype=jnp.float32))
    check(jnp.allclose(q_c, compute_tied_quantiles(f_c, jnp.ones(5, dtype=jnp.float32))),
          "JIT compute_tied_quantiles matches eager")

    # check_distinguishability uses Python type conversion (int/float/bool)
    # — it is a diagnostic, not meant for JIT inside the step function.
    f_has_ties = jnp.array([0.1, 0.1, 0.1, 2.0, 5.0], dtype=jnp.float32)
    d_c = check_distinguishability(f_has_ties)
    check(not d_c['ok'], "check_distinguishability: 3-of-5 tied → ok=False")
    check(d_c['frac_tied'] == 0.6, f"frac_tied={d_c['frac_tied']} (expected 0.6)")

    # JIT in lax.scan
    def scan_body(carry, _):
        w = jit_elite(f_c, carry)
        return carry, jnp.sum(w)
    _, sums = jax.lax.scan(scan_body, 2, None, length=3)
    check(jnp.allclose(sums, jnp.array([0.4, 0.4, 0.4])), "lax.scan works")

    print("    JIT + lax.scan: OK")

    # ── Test D: Divergence scenario comparison ─────────────────────────

    section("Test D: Divergence scenario — all costs equal")

    print("\n  Simulating B=80, B0=20, all costs identical (extreme plateau)")
    B, B0 = 80, 20
    f_all_same = jnp.zeros(B, dtype=jnp.float32)

    # Old behavior: argsort(argsort) → distinct ranks for tied values
    # JAX argsort is stable → preserves input order for ties
    # → first B0 indices (0..19) always get elite weight
    # → this is input-order dependent, not cost-dependent
    ranks_old = jnp.argsort(jnp.argsort(f_all_same))
    w_old = jnp.where(ranks_old < B0, 1.0 / B, 0.0)
    n_elite_old = int(jnp.sum(w_old > 0))
    # Note: with stable sort, exactly B0=20 get weight 1/B=0.0125

    # New behavior: all tied → all get same fractional weight
    w_new = compute_elite_weights(f_all_same, B0)
    n_unique_new = len(set(float(x) for x in w_new))

    print(f"    Old: {n_elite_old} elite samples (arbitrary — depends on input order)")
    print(f"    Old elite weight range: [{float(jnp.min(w_old)):.4f}, {float(jnp.max(w_old)):.4f}]")
    print(f"    New: {n_unique_new} distinct weight value(s) (all tied samples treated equally)")
    print(f"    New all-sample weight: {float(w_new[0]):.6f}")
    print(f"    Old elite sum: {float(jnp.sum(w_old)):.4f}")
    print(f"    New elite sum: {float(jnp.sum(w_new)):.4f}")

    check(n_unique_new == 1, "all tied → exactly 1 distinct weight value")
    check(abs(float(w_new[0]) - float(w_new[B-1])) < 1e-6, "all samples same weight")
    check(abs(float(jnp.sum(w_new)) - B0 / B) < 1e-6, f"sum = {float(jnp.sum(w_new)):.6f} (expected {B0/B:.6f})")

    # ── Test E: check_distinguishability ───────────────────────────────

    section("Test E: check_distinguishability")

    d1 = check_distinguishability(f_a1, eps=1e-7)  # no ties
    print(f"\n  No-ties case: {d1}")
    check(d1['ok'], "no ties → ok=True")
    check(d1['n_tied'] == 0, "n_tied=0")
    check(d1['max_tie_group'] == 1, "max_tie_group=1")

    d2 = check_distinguishability(f_all_same, eps=1e-7)  # all tied
    print(f"  All-tied case: {d2}")
    check(not d2['ok'], "all tied → ok=False")
    check(d2['n_tied'] == B, f"n_tied={B}")
    check(d2['max_tie_group'] == B, f"max_tie_group={B}")
    check(abs(d2['frac_tied'] - 1.0) < 1e-6, "frac_tied=1.0")

    # ── Summary ────────────────────────────────────────────────────────

    section("Summary")
    total = passed + failed
    print(f"\n  {passed}/{total} passed")
    if failed > 0:
        print(f"  {failed} FAILURES")
        sys.exit(1)
    else:
        print("  All tests passed!")
        sys.exit(0)
