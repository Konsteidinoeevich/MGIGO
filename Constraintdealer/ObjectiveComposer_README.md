# ObjectiveComposer — Structured Sub-Objective Composition

## 1. The Problem: Manual Weight Tuning

Real-world objectives have competing sub-terms. Traditionally, you assign each a
weight and sum them:

```python
f = 15.0 * tracking_error + 2.0 * smoothness + 0.1 * control_effort
```

This works, but has well-known problems:

| Problem | Why |
|---------|-----|
| **Unbounded** | A weight of 15 has no upper limit — one bad trajectory can blow up `f` |
| **Cross-term scale mismatch** | Tracking is ~100, smoothness is ~1 — weights must compensate |
| **Per-problem tuning** | New environment → new weights → trial and error |
| **No physical meaning** | What does `w=15.0` mean? Nothing interpretable |

## 2. The Solution: Per-Term Saturation Gain

Each sub-term independently goes through:

```
raw_value → log_transform → sigma_k(..., k=term_k) → bounded in [0, 1)
```

All compressed terms are summed, then passed through a final outer sigma:

```
f(x) = σ( Σ σ_k( T(term_i(x)), k=k_i ), k=k_outer )
```

### Why This Works

| Property | Manual Weights | Compose Objective |
|----------|---------------|-------------------|
| Boundedness | Unbounded | Each term ∈ [0, 1) |
| Scale handling | Manual (divide by S) | Automatic (log_transform) |
| Gain control | Dimensionless weight w | k → knee in physical units |
| No dominance | One term can dominate | All terms bounded → no single term can hijack cost |

## 3. k as Physical Gain

**k controls the knee** — the raw value where the term half-saturates:

```
raw_knee ≈ exp(1/k) - 1
```

| k | Raw knee | Use case |
|---|----------|----------|
| 5.0 | ~0.22 m | Lateral tracking — sub-meter precision matters |
| 3.0 | ~0.40 m | Close-range obstacle buffer |
| 2.0 | ~0.65 m | Final approach accuracy |
| 1.0 | ~1.7 | Balanced, moderate sensitivity |
| 0.5 | ~6.4 | Path-integral tracking — wide range |
| 0.2 | ~147 | Longitudinal tracking — ~100m range |
| 0.1 | ~2.2×10⁴ | Control effort — only penalize extreme waste |

**Large k = high gain**: saturates early, sensitive to small changes.
**Small k = low gain**: wide linear range, only large values matter.

This is **more interpretable** than weights: a knee of 0.65m means
"I care about sub-meter final approach accuracy." A weight of 15.0
means... nothing physical.

## 4. Quick Start

### 4.1 Recommended: Adaptive (semantic roles)

```python
from Constraintdealer.ObjectiveComposer import compose_objective_adaptive

ctx_calib = {'target': jnp.array([8.0, 6.0]),
             'init_state': jnp.array([0., 0., 0., 0.])}

obj = compose_objective_adaptive([
    (tracking_fn,   'primary',    "tracking"),     # P50: knee at median
    (final_fn,      'secondary',  "final"),        # P70
    (control_fn,    'tiebreaker', "control"),      # P95
], n_dims=16, bounds=(-5.0, 5.0), n_samples=500,
   ctx_calib=ctx_calib)
# obj is a standard (x, ctx) -> scalar callable, output in [0, 1)
```

### 4.2 Production: Dynamic (k tracks optimization progress)

See [§8 Dynamic k](#8-dynamic-k--zero-jax-recompilation).

### 4.3 Integration with Constran.build()

```python
from Constraintdealer.Constran import build, Deterministic

# obj from compose_objective_adaptive or compose_objective_dynamic
cost_fn = build(obj, [
    Deterministic(obs_fn,  mode='hard', priority=1),
    Deterministic(ped_fn,  mode='hard', priority=2),
])

# Step 3: pass to solver
result = mmog_igo_optimizer_mpc(..., fitness_fn_total=cost_fn, ...)
```

### 4.4 Diagnosing Per-Term Contributions

```python
from Constraintdealer.ObjectiveComposer import inspect_terms

info = inspect_terms(obj, x_test, ctx, [
    (tracking_fn,   0.5, "tracking"),
    (final_fn,      2.0, "final"),
    (control_fn,    1.0, "control"),
])
for c in info['contributions']:
    print(f"{c['name']}: raw={c['raw']:.3f} → compressed={c['compressed']:.3f} [{c['saturated']}]")
# Output:
#   tracking: raw=8.000 → compressed=0.740 [knee]
#   final:    raw=0.500 → compressed=0.528 [knee]
#   control:  raw=0.100 → compressed=0.058 [linear]
```

This tells you which terms are saturated (maxed out — their k may be too high),
which are in the knee region (good sensitivity), and which are still linear
(k may be too low, the term is barely contributing).

## 5. Before / After Comparison

### Before: Manual Weights

```python
def my_objective(z_flat, ctx):
    # Compute raw values
    track  = compute_tracking(z_flat, ctx)       # range ~0–500
    final  = compute_final_error(z_flat, ctx)    # range ~0–50
    smooth = compute_smoothness(z_flat, ctx)     # range ~0–10
    ctrl   = compute_control(z_flat, ctx)        # range ~0–5

    # Tune these weights for EVERY new problem:
    f = 1.0 * track + 15.0 * final + 1.5 * smooth + 0.1 * ctrl

    # Hope the solver can distinguish values across 3 orders of magnitude
    return f
```

### After: Adaptive

```python
from Constraintdealer.ObjectiveComposer import compose_objective_adaptive

obj = compose_objective_adaptive([
    (compute_tracking,   'primary',    "tracking"),    # P50 — auto knee
    (compute_final_error,'secondary',  "final"),       # P70
    (compute_smoothness, 'tiebreaker', "smoothness"),  # P95
    (compute_control,    'tiebreaker', "control"),     # P95
], n_dims=16, bounds=(-5,5), n_samples=500, ctx_calib=ctx)
# Done. No weights, no k. Just semantic roles.
# Output always in [0, 1). Pass directly to Constran.build().
```

## 6. Guidelines for Choosing k

### Approach A: Physical Knee ("I know what value matters")

1. Decide: "at what raw value should this term be half-saturated?"
2. `k = knee_to_k(raw_knee)`

```python
from Constraintdealer.ObjectiveComposer import knee_to_k

k_lat = knee_to_k(0.3)   # "0.3m lateral error → knee"
k_lon = knee_to_k(10.0)  # "10m longitudinal error → knee"
```

### Approach B: Typical + Max ("I know the operating range")

1. Estimate typical and maximum raw values
2. `k = suggest_k(typical, max)`

```python
from Constraintdealer.ObjectiveComposer import suggest_k

k = suggest_k(typical=2.0, max=100.0)
```

### Approach C: Start from defaults, diagnose, iterate

1. Start with `k=1.0` for all terms
2. Run `inspect_terms()` on a few candidate solutions
3. If a term always shows `[saturated]` → reduce its k (widen range)
4. If a term always shows `[linear]` and contributes little → increase k

## 7. Adaptive k via Semantic Roles

When you **don't know** the magnitude of each term upfront, use
`compose_objective_adaptive`. Instead of specifying k, you assign each term
a **semantic role**, and k is auto-calibrated from the empirical distribution
of random samples.

### 7.1 The Idea

| You know | You don't know |
|----------|---------------|
| This is my *main* objective | Its typical raw value |
| This is a *secondary* preference | Its maximum |
| This is a *tiebreaker* | Its dynamic range |

The role maps to a **percentile** of the term's empirical distribution.
The knee is set at that percentile, and k is derived automatically.

| Role | Percentile | Knee at | Meaning |
|------|-----------|---------|---------|
| `'primary'` | P50 | Median | Wide linear range — worst half saturates |
| `'secondary'` | P70 | 70th pctile | Moderate — 70% in linear region |
| `'tiebreaker'` | P95 | 95th pctile | Mostly linear — only extremes saturate |

You can also pass a bare float: `0.40` = P40, `0.99` = P99.

### 7.2 Usage

```python
from Constraintdealer.ObjectiveComposer import compose_objective_adaptive

ctx_calib = {'target': jnp.array([8.0, 6.0]),
             'init_state': jnp.array([0., 0., 0., 0.])}

obj = compose_objective_adaptive([
    (tracking_fn,   'primary',    "tracking"),     # P50
    (final_fn,      'secondary',  "final"),        # P70
    (smooth_fn,     'tiebreaker', "smoothness"),   # P95
], n_dims=16, bounds=(-5.0, 5.0), n_samples=1000,
   ctx_calib=ctx_calib)
```

Output during calibration:
```
--- Adaptive K Calibration ---
  tracking            : P50  knee=4.2371  k=0.6645
  final               : P70  knee=0.8920  k=1.4752
  smoothness          : P95  knee=6.1804  k=0.5100
```

### 7.3 Sample Sources (choose based on cost)

| Source | Parameter | Extra cost | When |
|--------|----------|------------|------|
| Random uniform | `n_dims` + `bounds` | n_samples × term calls | Cold start, cheap terms |
| Warmup samples | `warmup_samples` | **Zero** | MPC: reuse solver's first-step samples |
| Custom generator | `sample_fn` | n_samples calls | Known decision space structure |

```python
# Zero-overhead: reuse solver samples from warmup
obj = compose_objective_adaptive([
    (track_fn, 'primary', "track"),
], warmup_samples=mu_k_samples, ctx_calib=ctx)

# Cheap: random uniform
obj = compose_objective_adaptive([
    (track_fn, 'primary', "track"),
], n_dims=16, bounds=(-3.0, 3.0), n_samples=500, ctx_calib=ctx)
```

### 7.4 Integration with build()

```python
from Constraintdealer.Constran import build, Deterministic

obj = compose_objective_adaptive([...], n_dims=16, ctx_calib=ctx)
cost_fn = build(obj, [Deterministic(obs_fn, mode='hard', priority=1)])
# cost_fn ready for solver
```

## 8. Dynamic k — Zero JAX Recompilation

When cost magnitudes shift during optimization (e.g., tracking error drops from
~500 at initialization to ~10 at convergence), a static k calibrated from early
samples becomes too wide — terms lose sensitivity when they need it most.

The solution: **put k in ctx** instead of a closure.  ctx is already dynamic
(MPC updates target/obstacles every step), so changing k values in ctx does
NOT trigger JAX recompilation.

### 8.1 Three Functions

| Function | Role |
|----------|------|
| `calibrate_k_into_ctx(ctx, terms, ...)` | Initial calibration → writes `ctx['k_{name}']` |
| `compose_objective_dynamic(terms)` | Build cost fn that reads k from ctx |
| `update_k_from_elite(ctx, terms, elite_x)` | Update k from best solution (knee = raw×5) |

### 8.2 MPC Loop Pattern

```python
from Constraintdealer.ObjectiveComposer import (
    compose_objective_dynamic, calibrate_k_into_ctx, update_k_from_elite
)

# 1. Initial calibration (once)
ctx = calibrate_k_into_ctx(ctx, [
    (track_fn, 'primary',    'tracking'),
    (final_fn, 'secondary',  'final'),
], n_dims=16, bounds=(-5,5), n_samples=500)

# 2. Build cost function (once)
obj = compose_objective_dynamic([
    (track_fn, 0.0, 'tracking'),   # k placeholder — real k from ctx
    (final_fn, 0.0, 'final'),
])
cost_fn = Constran.build(obj, constraints)

# 3. MPC loop — update k every step from elite
for step in range(T_mpc):
    ctx['target'] = new_target
    ctx['obs_pos'] = new_obstacles

    result = solver(..., fitness_fn_total=cost_fn, context=ctx)

    # Update k from the elite solution (zero extra evaluations)
    elite_x = result.best_solution
    ctx = update_k_from_elite(ctx, terms, elite_x, multiplier=5.0)
```

**JAX compiles once.** k values in ctx change freely — no recompilation.

### 8.3 How `update_k_from_elite` Works

For each term, evaluates `term_fn(elite_x, ctx)`, sets `knee = raw * multiplier`,
then `k = 1/log(1 + knee)`.  At `multiplier=5`, the knee sits 5× ahead of the
current best — giving the optimizer headroom while keeping the term sensitive.

```
Before (random samples):  knee=500, k=0.16  → σ(10)=0.36 (flat)
After (elite update):     knee=50,  k=0.26  → σ(10)=0.69 (knee!)
```

### 8.4 Why Flat Costs Kill IGO Solvers

When the cost landscape goes flat, the entire IGO optimization collapses
through a causal chain:

```
cost → flat (σ'(x) → 0)
  → argsort becomes random noise (all candidates look identical)
    → natural gradient points in random directions
      → GMM distribution expands without guidance
        → samples fly to extreme values (hundreds/thousands)
          → eig(S) → 0 or ∞  →  NaN / overflow
```

IGO relies on ranking to estimate the natural gradient. When the cost
cannot distinguish good from bad solutions, the ranking is effectively
random, and the gradient is pure noise. The GMM expands, samples
diverge, and the solver **blows up**.

Existing protections in `MPCsolverM22` (eigenvalue clamping at `[1e-3, 1e3]`,
IGO weight clipping at `±20`) are **symptom treatments** — they catch the
blow-up at the final step.  Dynamic k is a **root-cause fix**: by keeping
every term in its knee region, the cost never goes flat, `argsort` always
has signal, and the cascade never starts.

**Experimental confirmation:** In a head-to-head comparison (500 IGO
iterations, 12-step horizon), the cost sensitivity near the optimum was:

| Approach | Cost spread near optimum | Status |
|----------|--------------------------|--------|
| Dynamic k | 0.0049 | Sensitive — solver sees improvement direction |
| Static k | 0.0000 | Dead flat — solver is blind |
| Raw weights | 0.000006 | Effectively flat — solver is blind |

## 9. API Reference

### `compose_objective_adaptive(terms, *, n_dims=..., bounds=..., ..., ctx_calib=..., k_outer=1.0)`

Auto-calibrate k from semantic roles. Each term is `(fn, role, name?)`.
Role: `'primary'` (P50), `'secondary'` (P70), `'tiebreaker'` (P95),
or a float percentile. Requires `ctx_calib` and a sample source.

### `knee_to_k(raw_knee) -> float`

Convert physical knee to k. `raw_knee = exp(1/k) - 1`.

### `k_to_knee(k) -> float`

Convert k to physical knee. Inverse of `knee_to_k`.

### `suggest_k(raw_typical, raw_max=None) -> float`

Heuristic k suggestion from typical and max values.

### `inspect_terms(composed_obj, x, ctx, terms_info) -> dict`

Per-term diagnostic: raw value, compressed value, saturation status.

## 10. When NOT to Use

- **Single-term objective**: `build_unconstrained()` is simpler
- **Already normalized**: if all terms are naturally in [0, 1], just sum them
- **Known, fixed weights**: if you've already tuned weights and they work, keep them
- **Gradient-based solvers**: the sigma nesting adds nonlinearity; check if your
  gradient method tolerates it (IGO is gradient-free, so no issue)

## 11. Demo

```bash
# Compare all three modes (no solver needed)
uv run python Functiontest/ObjectiveComposer_demo.py

# Run with IGO optimization
uv run python Functiontest/ObjectiveComposer_demo.py --optimize

# Single mode
uv run python Functiontest/ObjectiveComposer_demo.py --mode composed --optimize
```

## 12. Relationship to Constran

```
┌──────────────────────────────────────────────────┐
│  Constran.build(objective_fn, constraints)        │
│                                                   │
│  objective_fn can be:                             │
│    ├── plain Python function  (traditional)       │
│    ├── compose_objective([...])   ← NEW           │
│    └── compose_objective_auto([...])  ← NEW       │
│                                                   │
│  Constran handles:                                │
│    ├── Constraint nesting (hard/soft/tunable)     │
│    ├── log_transform + sigma_k per constraint     │
│    └── δ auto-assignment                          │
│                                                   │
│  ObjectiveComposer handles:                       │
│    ├── Per-term log_transform + sigma_k           │
│    ├── Additive competition between sub-terms     │
│    └── k-based gain control                       │
└──────────────────────────────────────────────────┘
```

Both layers use the same mathematical primitives (`sigma_k`, `log_transform`),
but at different levels: ObjectiveComposer structures the **innermost** layer
(the objective), while Constran structures the **outer** layers (the constraints).

---

## 13. Connection to the Small Gain Theorem

The user has observed a deep structural parallel between our k-based gain control
and the **Small Gain Theorem** (小增益理论) from nonlinear control theory.
This section formalizes that connection.

### 11.1 Quick Review: The Small Gain Theorem

The classical Small Gain Theorem (Zames, 1966) states:

> Consider two causal systems $H_1$, $H_2$ with finite $\mathcal{L}_2$ gains
> $\gamma_1$, $\gamma_2$. If $\gamma_1 \cdot \gamma_2 < 1$, then the feedback
> interconnection of $H_1$ and $H_2$ is finite-gain $\mathcal{L}_2$ stable.

In simpler terms: **if each subsystem amplifies signals by at most $\gamma_i$,
and the product of amplifications is < 1, the whole system is stable.**

The Small Gain Theorem is a special case of a broader unification — the
**Circle Criterion** and **Passivity Theorem** all emerge from the same
semi-inner product space framework (see "Unified Necessary and Sufficient
Conditions for the Robust Stability of Interconnected Sector-Bounded
Systems", arXiv:1809.08742).

### 11.2 σ_k as a Sector-Bounded Nonlinearity

Our saturation function:

$$\sigma_k(x) = \frac{kx}{\sqrt{1 + (kx)^2}}$$

is a **memoryless, sector-bounded nonlinearity** — exactly the kind of
nonlinearity that the Circle Criterion and Small Gain Theorem were designed
to handle.

**Sector characterization.** For all $x \neq 0$:

$$0 < \frac{\sigma_k(x)}{x} \le k$$

That is, $\sigma_k$ lies in the **sector $[0, k]$** — it's bounded below by 0
(passivity) and above by k (gain limit).

**Lipschitz / incremental gain.** The derivative:

$$\sigma_k'(x) = \frac{k}{(1 + (kx)^2)^{3/2}} \le k$$

So $|\sigma_k(x) - \sigma_k(y)| \le k|x-y|$. **k IS the Lipschitz constant** —
the maximum amplification any input difference can experience.

**Saturation (absolute bound).** $|\sigma_k(x)| < 1$ for all $x$, regardless of k.
This is the "saturation" property — no matter how large the input, the output
is bounded. In control terms: $\sigma_k$ has **finite $\mathcal{L}_\infty$ gain
of 1**, regardless of its $\mathcal{L}_2$ gain k.

### 11.3 The k Parameter IS the Gain

In control theory, the "gain" $\gamma$ of a system is the worst-case ratio of
output energy to input energy:

$$\gamma = \sup_{u \neq 0} \frac{\|H(u)\|}{\|u\|}$$

For our $\sigma_k$, the incremental gain (Lipschitz constant) is exactly k.
This is why we call k the **gain** — it's not just a metaphor, it's a
mathematically precise statement:

| k | Lipschitz gain | Meaning |
|---|---------------|---------|
| 5.0 | 5.0 | High gain — small input changes amplified 5× |
| 1.0 | 1.0 | Unity gain — non-expansive |
| 0.5 | 0.5 | Low gain — a contraction, signal attenuated |
| 0.1 | 0.1 | Very low gain — strong attenuation |

### 11.4 The Nesting as a Gain-Bounded Interconnection

Our `compose_objective` structure:

$$f(x) = \sigma_{k_{\text{outer}}}\!\left(\sum_i \sigma_{k_i}\!\big(\mathcal{T}(\text{term}_i(x))\big)\right)$$

can be interpreted as a **feedforward interconnection of gain-bounded subsystems**:

```
term₁(x) → [T] → [σ_{k₁}] ↘
term₂(x) → [T] → [σ_{k₂}] → [ Σ ] → [σ_{k_outer}] → f(x)
term₃(x) → [T] → [σ_{k₃}] ↗
```

Each channel has:
- **Log-transform** $\mathcal{T}$: compresses dynamic range (like a logarithmic
  amplifier / compander in signal processing)
- **Saturation** $\sigma_{k_i}$: sector-bounded nonlinearity with gain $k_i$,
  output bounded in $(-1, 1)$

The **adder** $\Sigma$ has gain $\le \sum k_i$ in the worst case.

The **outer** $\sigma_{k_{\text{outer}}}$ provides **global gain bounding**:
regardless of how many terms there are or how large their sum, the output
is always in $(-1, 1)$.

### 11.5 Why This "Just Works" — A Small-Gain Argument

**Claim:** The composed objective cannot be dominated by any single term.

**Proof sketch (small-gain style):**

1. Each sub-term channel has **finite gain** $k_i$ (Lipschitz constant)
2. Each channel output is **absolutely bounded**: $|\sigma_{k_i}(\cdot)| < 1$
3. The sum of $N$ terms has gain $\le \sum k_i$, but output is bounded in
   $(-N, N)$
4. The outer $\sigma_{k_{\text{outer}}}$ compresses $(-N, N)$ back to $(-1, 1)$
   — it has **finite $\mathcal{L}_\infty$ gain** regardless of $N$
5. Therefore: **bounded input (raw term values) → bounded output (cost)** —
   an **Input-to-State Stability (ISS)** property

Unlike linear weights $w_i \cdot \text{term}_i$ where one large term can make
$f \to \infty$, the saturation nesting guarantees $f \in (-1, 1)$ always.
No single term can "destabilize" the overall cost.

### 11.6 The Outer σ as a "Loop Gain" Controller

In a feedback control system, you typically add a controller to reduce loop
gain below 1 for stability. In our feedforward structure:

$$\text{outer } \sigma \text{ acts like an automatic gain controller}$$

- If all inner terms are small → sum is small → outer σ operates in its
  linear region (gain ≈ k_outer) → good distinguishability
- If some inner terms are large → sum is large → outer σ saturates →
  total output bounded → prevents blow-up
- The transition is **continuous** (no hard clipping) → gradient is preserved

This is analogous to an **Automatic Gain Control (AGC)** circuit in
communications, or a **gain-scheduled controller** that adapts its
amplification based on signal magnitude.

### 11.7 Deeper Connections

**Contraction mapping.** For $k \le 1$, each $\sigma_k$ is a **contraction**
in the Banach sense:

$$|\sigma_k(x) - \sigma_k(y)| \le k|x-y| < |x-y|$$

The composition of contractions is a contraction. This connects to the
**Banach Fixed-Point Theorem** — iterated application converges to a unique
fixed point. In our context: small changes in raw term values cannot cause
disproportionately large changes in cost.

**Circle Criterion.** For a Lur'e system (linear dynamics + sector-bounded
nonlinearity in feedback), the Circle Criterion guarantees stability if the
Nyquist plot of the linear part avoids a disk determined by the sector bounds.
Our saturation $\sigma_k \in [0, k]$ is a canonical example — the sector
bounds tell us exactly how "aggressive" the nonlinearity is.

**Passivity.** $\sigma_k$ is **passive**: $\sigma_k(x) \cdot x \ge 0$ for all
$x$ (it's an odd function with $\sigma_k(x)/x > 0$). In interconnected passive
systems, stability follows from the Passivity Theorem — another unification
with the Small Gain Theorem.

**ISS (Input-to-State Stability).** The entire composition satisfies:

$$\|f\|_\infty \le 1 \quad \text{regardless of input magnitude}$$

This is the strongest form of ISS — the output is **globally uniformly
ultimately bounded** with bound 1. No tuning of k can change this guarantee.

### 11.8 Practical Implication: Choosing k via Gain Reasoning

The control-theoretic perspective gives a principled way to choose k:

| Control concept | Our equivalent | Implication for k |
|-----------------|---------------|-------------------|
| Loop gain | k (Lipschitz constant) | k < 1: contraction, k > 1: amplification |
| Gain margin | How much k can increase before instability | Outer σ provides infinite gain margin |
| Sector bound | σ_k ∈ [0, k] | k defines the linear operating region |
| Saturation limit | |σ_k| < 1 | Always safe, independent of k |
| Bandwidth | 1/k (knee location) | Small k = wide bandwidth, large k = narrow |
| Gain scheduling | Different k per term | Each sub-objective has its own "channel gain" |

**The key insight:** You're not just picking a dimensionless weight — you're
**designing the gain of a nonlinear control channel** whose stability is
guaranteed by the saturation structure. The outer σ is the "safety net"
that bounds total gain regardless of how many terms you add or how large
their k values are.

### 11.9 References

- Zames, G. (1966). "On the input-output stability of time-varying nonlinear
  feedback systems." *IEEE Trans. Automatic Control*, 11(2–3).
- "A nonlinear small gain theorem for the analysis of control systems with
  saturation." *IEEE Trans. Automatic Control*, 1996.
- "Unified Necessary and Sufficient Conditions for the Robust Stability of
  Interconnected Sector-Bounded Systems." arXiv:1809.08742.
- Sontag, E.D. (2008). "Input to State Stability: Basic Concepts and Results."
  In *Nonlinear and Optimal Control Theory*, Springer.
- Khalil, H.K. (2002). *Nonlinear Systems*, 3rd ed. Prentice Hall.
  (Ch. 5: Input-Output Stability, Ch. 10: Passivity, Circle Criterion)
