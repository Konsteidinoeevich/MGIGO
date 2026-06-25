# MGIGO: General-Purpose Black-Box Optimization Solver Framework

A black-box global optimization solver based on **IGO (Information-Geometric
Optimization)** + **GMM (Gaussian Mixture Model)**, extended with **MCTS-guided
discrete search** and **automatic constraint transformation**. Transforms
arbitrary constrained optimization problems into scalar cost functions, searching
for global optima in continuous, discrete, or mixed decision spaces.

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [Core Solvers](#2-core-solvers)
3. [Constraint Transformation Engine (Constran)](#3-constraint-transformation-engine)
4. [MCTS + IGO Hybrid Framework](#4-mcts--igo-hybrid-framework)
5. [Energy / Unit Commitment Applications](#5-energy--unit-commitment-applications)
6. [Application Domains](#6-application-domains)
7. [Quick Start](#7-quick-start)
8. [Project Structure](#8-project-structure)
9. [References](#9-references)

---

## 1. What It Does

### In One Sentence

**You write the objective and constraints in JAX. The solver handles the rest.**

### Optimization Landscape

| Problem Type | Solver | Constraint Engine |
|-------------|--------|-------------------|
| Unconstrained continuous optimization | MPCsolverM22, blockwise_mgigo | `build_unconstrained()` |
| Deterministic constraints (g(x) ≤ 0) | MPCsolverM22 | `Deterministic` |
| Chance constraints (P(safe) ≥ 95%) | MPCsolverM22 | `Chance` |
| Robust constraints (∀ξ ∈ Ξ) | MPCsolverM22 | `Robust` |
| Distributionally robust (worst distribution) | MPCsolverM22 | `DRO` |
| Mixed constraints + priorities | MPCsolverM22 | `build()` |
| Discrete-continuous hybrid | MPC_discrete_variables, Pure_discrete | `build()` |
| TSP / permutation optimization | TSP (Plackett-Luce) | — |
| Receding-horizon MPC (obstacle/tracking) | MPCsolverM22, MPCsolver | `build()` |
| Multi-agent Nash equilibrium | MPC_G, MPC_G_MS | `build_multi_agent()` |
| Block-structured optimization | blockwise_mgigo | `build()` |
| MCTS-guided combinatorial optimization | MCTSIGO | `build()` |

### Key Features

- **Black-box friendly**: no gradients needed — only f(x) values
- **Numerically stable**: log-transform + saturation nesting, full-range
  distinguishability in float32
- **Constraints as declarations**: 5 constraint types, 3 priority modes
  (hard/soft/tunable), arbitrary M layers
- **Zero JAX recompilation in MPC**: call `build()` once, pass dynamic info
  via `ctx` — no recompilation across MPC steps
- **Multi-agent**: Nash equilibrium via RNE, mixed strategies, multi-block games
- **MCTS + IGO integration**: discrete topology search (MCTS) + continuous
  parameter optimization (IGO) with bidirectional feedback

---

## 2. Core Solvers

### 2.1 Solver Inventory

All solvers live in [gmm_igo/](gmm_igo/). The recommended primary solver is
**MPCsolverM22**.

#### Primary Production Solvers

| Solver | File | Role |
|--------|------|------|
| `mmog_igo_optimizer_mpc` | [MPCsolverM22.py](gmm_igo/MPCsolverM22.py) | **Main workhorse** — multi-block IGO, weight clipping, `lax.scan` acceleration, dimension masking |
| `igo_mog_optimizer` | [MPCsolver.py](gmm_igo/MPCsolver.py) | Base solver, simpler interface |
| `mmog_igo_optimizer_mpc` (M23) | [MPCsolverM23.py](gmm_igo/MPCsolverM23.py) | M22 + exploration precision reset |
| `blockwise_mgigo` | [blockwise_mgigo.py](gmm_igo/blockwise_mgigo.py) | Block-structured MGIGO with auto dimension partitioning |

#### Multi-Agent / Game-Theoretic Solvers

| Solver | File | Role |
|--------|------|------|
| `mmog_igo_rne_solver` | [MPC_G.py](gmm_igo/MPC_G.py) | Nash equilibrium with MC inner loop, sample reuse |
| `mmog_igo_rne_solver` (S variant) | [MPC_G_S.py](gmm_igo/MPC_G_S.py) | Single-block GMM update for RNE |
| `mmog_igo_rne_blocks_solver` | [MPC_G_MS.py](gmm_igo/MPC_G_MS.py) | **Multi-block multi-agent** mixed strategies |
| Tracking variant | [MPC_G_S_V.py](gmm_igo/MPC_G_S_V.py) | Blockwise RNE with metrics tracking |

#### Discrete / Mixed-Integer Solvers

| Solver | File | Role |
|--------|------|------|
| Hybrid discrete-continuous | [MPC_discrete_variables.py](gmm_igo/MPC_discrete_variables.py) | Continuous blocks via GMM, discrete blocks via categorical natural gradient |
| Refined hybrid | [MPC_discrete_variable1.py](gmm_igo/MPC_discrete_variable1.py) | Separate step sizes for continuous vs discrete blocks |
| Pure discrete | [Pure_discrete.py](gmm_igo/Pure_discrete.py) | Categorical block updates only |
| TSP / permutations | [TSP.py](gmm_igo/TSP.py) | Plackett-Luce permutation model, sequential masked sampling |

#### Sample Reuse Variants

| Solver | File | Role |
|--------|------|------|
| Sample reuse | [MPC_ReuseOldSamples.py](gmm_igo/MPC_ReuseOldSamples.py) | Cholesky-parameterized IGO with old-sample importance weighting |
| Blockwise reuse | [MPC_Rblockwise.py](gmm_igo/MPC_Rblockwise.py) | Blockwise version with dimension masking |
| Weight reset | [MPCresetweight.py](gmm_igo/MPCresetweight.py) | Strict Algorithm 3/4 alignment, full reset strategy |

#### Multi-Problem / Multi-Objective

| Solver | File | Role |
|--------|------|------|
| Batched multi-problem | [MPCsolvermultipleproblems.py](gmm_igo/MPCsolvermultipleproblems.py) | `vmap`-based parallel solver wrapper |
| Multi-objective | [MPCsolver_differentobjects.py](gmm_igo/MPCsolver_differentobjects.py) | Separate elite weights per objective |

### 2.2 Unified Cost Function Signature

```python
def my_cost(x, ctx):
    # x: 1D jnp.array (decision variables)
    # ctx: any Python object (environment state, target positions, etc.)
    return scalar  # lower is better

result = solver(..., fitness_fn=cost_fn, context=ctx)
```

### 2.3 Blockwise MGIGO

For high-dimensional problems, [blockwise_mgigo.py](gmm_igo/blockwise_mgigo.py)
automatically partitions decision variables into blocks of dimension ≤ 10. Each
block maintains an independent GMM (K=3 components). Blocks are sampled
independently → concatenated → evaluated jointly → updated independently via
natural gradient. Block coupling is handled implicitly through the shared scalar
cost feedback.

---

## 3. Constraint Transformation Engine (Constran)

[Constraintdealer/Constran.py](Constraintdealer/Constran.py) is the **translator**
between you and the solver. You declare each constraint's **type** and **priority**;
it auto-assembles a solver-ready cost function using the sigma-saturation nesting
methodology.

### 3.1 Supported Constraint Types

| Type | Math | Method |
|------|------|--------|
| `Deterministic` | $g(x) \le 0$ | Direct evaluation, log-transformed violation |
| `Chance` | $P(g(x,\xi) \le 0) \ge 1-\alpha$ | MC quantile estimation |
| `Robust` | $g(x,\xi) \le 0 \;\; \forall\xi \in \Xi$ | `lax.scan` max over uncertainty set |
| `DRO` | $\inf_{P\in\mathcal{P}} P(g(x,\xi)\le 0) \ge 1-\alpha$ | Worst-case MC over ambiguity set |

All constraint values pass through `sign(x)·log(1+|x|)` log-transform,
eliminating numeric explosion regardless of raw magnitude.

### 3.2 Priority Nesting — Three Modes

| Mode | Mechanism | When to Use |
|------|-----------|-------------|
| **Hard** | Violation → cost > ALL inner-layer maxima | Safety, legal, physical limits |
| **Tunable** | $\delta \cdot \sigma(\beta \cdot T(g))$ — $\beta$ controls sharpness | Adjustable hardness |
| **Soft** | $T(g)$ added directly, no floor | Preferences, efficiency, comfort |

**Design rule**: Hard outside, Soft inside. For deep nests, $\delta$ adapts
automatically ($\sigma_1^{(n)}(1) = 1/\sqrt{n+1}$).

### 3.3 Three-Step Usage

```python
from Constraintdealer.Constran import *

# Step 1: Write the objective
def my_obj(x, ctx):
    return jnp.sum((x[:2] - ctx['target'])**2) + 0.1 * jnp.sum(x[2:]**2)

# Step 2: Declare constraints
constraints = autodelta([
    Deterministic(lambda x, ctx: -(x[0] - obs_x)**2 - (x[1] - obs_y)**2 + r_safe**2,
                  mode='hard', priority=1),
    Chance(lambda x, xi, ctx: jnp.linalg.norm(x[:2] + xi) - safe_dist,
           noise_fn=lambda key, shape: jax.random.normal(key, shape) * 0.1,
           alpha=0.05, mode='tunable', priority=2, delta_soft=2.0, beta=5.0),
    Deterministic(lambda x, ctx: jnp.sum(x[2:]**2) - efficiency_limit,
                  mode='soft', priority=3),
])

# Step 3: Build → plug directly into solver
cost_fn = build(my_obj, constraints)
result = solver(..., fitness_fn_total=cost_fn, context=ctx)
```

### 3.4 Documentation

- [ConstraintsTransformation_README.md](Constraintdealer/ConstraintsTransformation_README.md) —
  Full mathematical methodology: log-transform theory, sigma-k saturation, delta
  selection, deep nesting scale laws (700+ lines)
- [ConstranUser_README.md](Constraintdealer/ConstranUser_README.md) —
  User-facing quick-start manual with parameter selection guides

---

## 4. MCTS + IGO Hybrid Framework

The [MCTSIGO/](MCTSIGO/) directory implements a **Monte Carlo Tree Search +
IGO** hybrid for combinatorial optimization problems with discrete-continuous
structure.

### 4.1 Core Idea

> **MCTS handles discrete topology search. IGO handles continuous parameter
> optimization. Bidirectional feedback aligns them.**

The framework targets problems with a natural bipartite structure:
- **Upper layer (topology/structure)**: discrete choices (unit commitment,
  maneuver selection, asset inclusion)
- **Lower layer (parameters/scheduling)**: continuous optimization (power
  dispatch, speed profiles, weight allocation)

### 4.2 Components

| File | Role |
|------|------|
| [decision_unit.py](MCTSIGO/decision_unit.py) | `DecisionUnit`, `MacroAction`, `ProblemSpec` — template-based discrete decision space description |
| [fidelity_config.py](MCTSIGO/fidelity_config.py) | `FidelityConfig` — three-fidelity evaluation system (none/light/full) with auto-budget allocation |
| [guide_tree.py](MCTSIGO/guide_tree.py) | `IndexTree` + `GuideTreeSolver` — static index tree with UCT selection, sigma-wrapped Q, sequential 3-stage evaluation |
| [MCTS.md](MCTSIGO/MCTS.md) | **Design document** — full framework specification: tree semantics, blockwise decomposition, UCB1 scoring, three-tier evaluation pyramid, bidirectional feedback |
| [MCTSforMPC.md](MCTSIGO/MCTSforMPC.md) | **MPC specialization** — horizon index tree + multi-modal IGO for receding-horizon discrete-continuous MPC, multi-vehicle game formulation, multi-tree heterogeneous architecture |

### 4.3 Three-Tier Evaluation Pyramid

```
Layer 1 — Constran fast eval (μs, thousands of calls)
    No MC, no IGO iteration. Pure deterministic single-point eval.
    Same Constran cost function as layers 2/3 — no modeling gap.

Layer 2 — Light IGO (3-5s, 30-50 calls)
    T=100, B=30, MC=30. Full horizon, preliminary diagnostics.

Layer 3 — Full IGO (15-30s, 3-5 calls)
    T=300, B=200, MC=100. Chance/DRO, complete diagnostics.
```

### 4.4 Key Design Principles

- **Tree depth = number of decision units**, not time steps × states.
  Macro-action compression avoids exponential explosion
- **All costs are non-Markovian** (startup costs, SoC inheritance, ramp
  constraints) — handled by full-horizon forward rollout, not per-node
  decomposition
- **Multi-tree heterogeneous architecture**: different agents / asset types
  get structurally independent index trees, converging at a joint RNE/IGO solver
  (see [MCTSforMPC.md](MCTSIGO/MCTSforMPC.md) §9)

---

## 5. Energy / Unit Commitment Applications

The [Energy/](Energy/) directory applies MGIGO + Constran to power system
optimization — from 5-generator proof-of-concept to 50-generator stochastic
Unit Commitment with storage and renewables.

### 5.1 Key Files

| File | Description |
|------|-------------|
| [UnitCommitment_test.py](Energy/UnitCommitment_test.py) | 5-gen proof of concept: incremental encoding, single-step and 24h MPC modes |
| [UnitCommitment_50.py](Energy/UnitCommitment_50.py) | 50-gen UC with block decomposition (5 blocks × 10 dims), Constran hierarchy |
| [stochastic_uc.py](Energy/stochastic_uc.py) | Stochastic UC with battery storage, wind, solar, chance/DRO constraints |
| [Energy_storage.py](Energy/Energy_storage.py) | Physics-based storage models: Li-ion, Hydrogen, Pumped Hydro, CAES, SolarPV, Wind |
| [Generator.py](Energy/Generator.py) | Generator set builders, load curves, incremental ramp-compliant encoding |
| [plot_results.py](Energy/plot_results.py) | Visualization for UC results (dispatch, power mix, SoC, cost breakdown) |

### 5.2 Documentation

| File | Description |
|------|-------------|
| [UC_README.md](Energy/UC_README.md) | Methodology guide: random keys encoding, why no `jnp.where`, beta parameter selection, MPC strategy |
| [MCTS_UC_README.md](Energy/MCTS_UC_README.md) | Full MCTS+IGO mathematical specification for UC: sets, decision variables, constraint hierarchy, closed-loop design |

### 5.3 Key Design Insight

UC uses **pure continuous encoding** (random keys) for generator on/off decisions
— no discrete variables needed. Each generator gets one continuous variable
$x_i \in \mathbb{R}$, decoded via $p_i = P_{\max} \cdot \sigma(x_i)$. The GMM
naturally converges to Dirac deltas at $x_i \gg 0$ (on) or $x_i \ll 0$ (off).

This avoids the combinatorial explosion of MILP while handling non-convex costs
and uncertainty distributions natively — no SOS2 linearization, no Big-M, no
scenario-tree reduction.

### 5.4 Results (5-Generator)

| Metric | Single Timestep | 24h Rolling MPC |
|--------|----------------|-----------------|
| Max power imbalance | 0.83 MW | 5.0 MW (2.5%) |
| vs greedy | −18.5% cost | — |
| Solve time/step | 2.0s (with JIT) | 0.16s (steady-state) |
| p_min/p_max compliance | ✓ | ✓ |
| Ramp rate compliance | ✓ | ✓ |

---

## 6. Application Domains

### 6.1 Function Benchmarks & Visualization — [Functiontest/](Functiontest/)

| File | Description |
|------|-------------|
| [Constraints.py](Functiontest/Constraints.py) | 16 hierarchical constraint tests (deterministic, chance, nested) with animated GIF outputs |
| [RobustConstraints.py](Functiontest/RobustConstraints.py) | Robust constraint benchmarks with non-convex uncertainty sets |
| [Multimodal.py](Functiontest/Multimodal.py) | Automatic modality selection: 6 scenarios (two-basin, hard-select, prob-sensors, safe-fast, three-way, nested-swap) |
| [Hybridsystemtest.py](Functiontest/Hybridsystemtest.py) | **Full hybrid-system MPC**: 2D vehicle on mixed terrain (ice patches) + pedestrian crossing with chance constraints. 3-level saturation nesting |
| [Discretetest.py](Functiontest/Discretetest.py) | Discrete-continuous hybrid test |
| [utils.py](Functiontest/utils.py) / [utilss.py](Functiontest/utilss.py) | Visualization & post-processing utilities |

**Output directories**:
- `Functiontest/output_igo_test/` — Saturation/hierarchical constraint test outputs
- `Functiontest/Hybrid_test/` — Hybrid MPC visualizations
- `Functiontest/Multimodal_test/` — Multimodal scenario outputs (animations, contours, 3D surfaces)
- `Functiontest/Robust_test/` — Robust constraint test outputs

### 6.2 MPC Test Suite — [MPCtest/](MPCtest/)

| File | Description |
|------|-------------|
| [MPCmain21animate.py](MPCtest/MPCmain21animate.py) | Obstacle avoidance MPC with Constran-built cost, animated output |
| [TSPtest.py](MPCtest/TSPtest.py) | TSP test using Plackett-Luce IGO solver (20 random cities) |
| [mainM1.py](MPCtest/mainM1.py) | Multi-block test with heterogeneous dimensions (2,5,8). Weight-only vs full-reset strategies |
| [mainM2.py](MPCtest/mainM2.py) | M-MoG IGO with T0 restart, coupled sphere fitness |
| [mainM3.py](MPCtest/mainM3.py) | Full hot restart vs weight-only reset contrast |
| [cma-esversion.py](MPCtest/cma-esversion.py) | **CMA-ES comparison baseline** for obstacle avoidance MPC |

### 6.3 B-Spline Trajectory Optimization — [Cartest/](Cartest/)

| File | Description |
|------|-------------|
| [bsplinetraj.py](Cartest/bsplinetraj.py) | BSpline trajectory generator with offline precomputed basis matrices. Position/velocity/acceleration evaluation |
| [bsplinetrajwith1.py](Cartest/bsplinetrajwith1.py) | Extended generator with variable time parametrization (learnable time weights via softmax) |
| [spline.py](Cartest/spline.py) | Offline B-spline basis precomputation (SciPy `BSpline`). Outputs `bspline_basis.npz` |
| [Simple.py](Cartest/Simple.py) | Minimal warm-start demo with MPCsolverM22 |
| [Simple1.py](Cartest/Simple1.py) | Extended trajectory optimization: variable time, lane shifting, multi-block (x, y, θ) |
| [splinetraj.md](Cartest/splinetraj.md) | Architecture notes: multi-block solver for (x, y, time) trajectory variables |

### 6.4 Autonomous Intersection Crossing — [Crossingroad/](Crossingroad/)

Multi-vehicle intersection coordination using IGO. 3 agents with B-spline
trajectories, spatio-temporal overlap cost, Frenet parameterization.

| File | Description |
|------|-------------|
| [main.py](Crossingroad/main.py) | Entry point: loads geometry, builds hybrid intersection map, calls solver |
| [Costcomputation.py](Crossingroad/Costcomputation.py) | `LebesgueEvaluator`: bitmask-based spatio-temporal overlap cost |
| [kinematics.py](Crossingroad/kinematics.py) | Sigmoid-based trajectory mapping (velocity profiles from optimization variables) |
| [parameterization.py](Crossingroad/parameterization.py) | Frenet trajectory parameterization (26-dim optimization vector → B-spline trajectories) |
| [Upperlayeroptimization.py](Crossingroad/Upperlayeroptimization.py) | 3-agent × 3D parameter optimization via MPCsolver |
| [utils/](Crossingroad/utils/) | Intersection geometry (`geometry_reference.py`), hybrid grid map (`map_utils.py`) |

### 6.5 Multi-Agent Games — [MultipleTest/](MultipleTest/)

| File | Description |
|------|-------------|
| [Testgame.py](MultipleTest/Testgame.py) | Two-agent crossing game (horizontal vs vertical), blockwise RNE solver |
| [Trackgame.py](MultipleTest/Trackgame.py) | Three-vehicle highway game, multi-block multi-scale solver, bicycle model |
| [Transformation.py](MultipleTest/Transformation.py) | Low-level vehicle physics: bicycle model, pairwise footprint overlap, trajectory rollouts |
| [bsplineplanner.py](MultipleTest/bsplineplanner.py) | B-spline trajectory generator with vehicle state conversion |

### 6.6 Wrapper Validation Report — [wrapper_validation_report/](wrapper_validation_report/)

Validates that the Constran wrapper API produces **identical results** to
matched hand-crafted cost functions across 3 borrow-overtake driving scenarios
(safe, blocked, critical).

| File | Description |
|------|-------------|
| [wrapper_validation_report.html](wrapper_validation_report/wrapper_validation_report.html) | Full validation report with trajectory comparisons |
| [assets/](wrapper_validation_report/assets/) | Scenario videos (.mp4), metrics charts, safety evaluation tables, CSV summary |

---

## 7. Quick Start

### 7.1 Installation

```bash
git clone https://github.com/Konsteidinoeevich/MGIGO.git
cd MGIGO
uv sync  # or pip install -e .
```

Dependencies: JAX (CUDA 12), NumPy, Matplotlib, tqdm, SciPy (optional, for B-spline).

### 7.2 Run Examples

```bash
# Deterministic constraint tests (16 scenarios)
uv run python Functiontest/Constraints.py

# Robust constraint tests
uv run python Functiontest/RobustConstraints.py

# Hybrid system MPC (3-level nesting)
uv run python Functiontest/Hybridsystemtest.py

# Multimodal optimization
uv run python Functiontest/Multimodal.py

# MPC obstacle avoidance (with animation)
uv run python MPCtest/MPCmain21animate.py

# CMA-ES comparison baseline
uv run python MPCtest/cma-esversion.py

# Unit Commitment (5-gen proof of concept)
uv run python Energy/UnitCommitment_test.py --mode single
uv run python Energy/UnitCommitment_test.py --mode mpc

# TSP test
uv run python MPCtest/TSPtest.py

# B-spline trajectory optimization
uv run python Cartest/Simple.py
```

### 7.3 Minimal Example — Unconstrained

```python
import jax.numpy as jnp
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc

def my_cost(x, ctx):
    return (x[0] - 3.0)**2 + (x[1] + 2.0)**2

result = mmog_igo_optimizer_mpc(
    key, T=200, dt=0.1, M=1, K=3, B=80, B0=40,
    dims=[2], T_0=50,
    fitness_fn_total=my_cost,
    initial_mu_k=mu, initial_L_inv_k=L_inv, initial_v_k=v,
    context=None,
)
```

### 7.4 Minimal Example — Constrained

```python
from Constraintdealer.Constran import *

constraints = autodelta([
    Deterministic(lambda x, ctx: x[0] + x[1] + 4,     # x + y ≥ -4
                  mode='hard', priority=1),
    Deterministic(lambda x, ctx: x[1] - x[0],          # y ≤ x
                  mode='hard', priority=2),
])

cost_fn = build(lambda x, ctx: ((x[0]-6)**2 + (x[1]+4)**2) / 50, constraints)
result = mmog_igo_optimizer_mpc(..., fitness_fn_total=cost_fn, ...)
```

### 7.5 Minimal Example — Multi-Agent

```python
from Constraintdealer.Constran import build_multi_agent
from gmm_igo.MPC_G_MS import mmog_igo_rne_blocks_solver

agent_fns = build_multi_agent({
    0: (lambda x, ctx: jnp.sum((x[:2] - ctx['t0'])**2), []),
    1: (lambda x, ctx: jnp.sum((x[2:] - ctx['t1'])**2), []),
})

result = mmog_igo_rne_blocks_solver(
    key, T=150, dt=0.15, N_blocks=2, M_agent=2, K=3,
    B=80, B0=40, dims=[2, 2], T_0=50,
    fitness_fn_j=agent_fns,
    block_to_agent_idx=jnp.array([0, 1]),
    ...
)
```

---

## 8. Project Structure

```
gmm_igo/
│
├── gmm_igo/                              # Core solvers (JAX)
│   ├── MPCsolverM22.py                   # ★ Main solver (recommended)
│   ├── MPCsolverM23.py                   # M22 + exploration reset
│   ├── MPCsolver.py                      # Base solver
│   ├── blockwise_mgigo.py                # Block-structured MGIGO
│   ├── MPC_G.py                          # Multi-agent Nash (RNE)
│   ├── MPC_G_S.py                        # Single-block RNE
│   ├── MPC_G_MS.py                       # Multi-block multi-agent RNE
│   ├── TSP.py                            # Permutation optimization
│   ├── Pure_discrete.py                  # Categorical IGO
│   ├── MPC_discrete_variables.py         # Discrete-continuous hybrid
│   ├── MPC_discrete_variable1.py         # Refined hybrid
│   ├── MPC_ReuseOldSamples.py            # Sample-reuse IGO
│   ├── MPC_R.py / MPC_Rblockwise.py      # Reuse variants
│   ├── MPCresetweight.py                 # Weight-reset solver
│   ├── MPCsolvermultipleproblems.py      # Batched multi-problem
│   ├── MPCsolver_differentobjects.py     # Multi-objective IGO
│
├── Constraintdealer/                     # Constraint transformation engine ★
│   ├── Constran.py                       # Core API (build, constraint types)
│   ├── ConstraintsTransformation_README.md  # Full mathematical methodology
│   └── ConstranUser_README.md            # User manual with examples
│
├── MCTSIGO/                              # MCTS + IGO hybrid framework ★
│   ├── decision_unit.py                  # Decision space abstractions
│   ├── fidelity_config.py                # Three-fidelity evaluation config
│   ├── guide_tree.py                     # Index tree + guide tree solver
│   ├── MCTS.md                           # Framework design document
│   └── MCTSforMPC.md                     # MPC specialization document
│
├── Energy/                               # Unit Commitment applications ★
│   ├── UnitCommitment_test.py            # 5-gen proof of concept
│   ├── UnitCommitment_50.py              # 50-gen with block decomposition
│   ├── stochastic_uc.py                  # Stochastic UC + storage + renewables
│   ├── Energy_storage.py                 # Physics-based storage models
│   ├── Generator.py                      # Generator & load curve builders
│   ├── plot_results.py                   # UC visualization
│   ├── UC_README.md                      # Methodology guide
│   ├── MCTS_UC_README.md                 # MCTS+IGO UC specification
│   └── results/                          # Saved optimization results
│
├── Functiontest/                         # Benchmarks & visualizations
│   ├── Constraints.py                    # 16 hierarchical constraint tests
│   ├── RobustConstraints.py              # Robust constraint benchmarks
│   ├── Multimodal.py                     # Multimodal optimization tests
│   ├── Hybridsystemtest.py               # Full hybrid-system MPC
│   ├── Discretetest.py                   # Discrete-continuous hybrid
│   ├── utils.py / utilss.py              # Visualization utilities
│   ├── Hybrid_test/                      # Hybrid MPC output
│   ├── Multimodal_test/                  # Multimodal test output
│   ├── Robust_test/                      # Robust test output
│   └── output_igo_test/                  # Saturation test output
│
├── MPCtest/                              # MPC test suite
│   ├── MPCmain21animate.py               # Obstacle avoidance MPC
│   ├── TSPtest.py                        # TSP test
│   ├── mainM1.py / mainM2.py / mainM3.py # Multi-block strategy tests
│   ├── cma-esversion.py                  # CMA-ES comparison
│   └── outcmaes/                         # CMA-ES output
│
├── Cartest/                              # B-spline trajectory planning
│   ├── bsplinetraj.py                    # BSpline trajectory generator
│   ├── bsplinetrajwith1.py               # Variable-time extension
│   ├── spline.py                         # Basis precomputation
│   ├── Simple.py / Simple1.py            # Demos
│   └── bspline_basis.npz                 # Precomputed basis
│
├── Crossingroad/                         # Autonomous intersection crossing
│   ├── main.py                           # Entry point
│   ├── Costcomputation.py                # Spatio-temporal overlap cost
│   ├── kinematics.py                     # Sigmoid trajectory mapping
│   ├── parameterization.py               # Frenet parameterization
│   ├── Upperlayeroptimization.py         # IGO optimization layer
│   ├── spline_filter.py                  # B-spline filter & basis
│   └── utils/                            # Geometry & map utilities
│
├── MultipleTest/                         # Multi-agent game tests
│   ├── Testgame.py                       # Two-agent crossing game
│   ├── Trackgame.py                      # Three-vehicle highway game
│   ├── Transformation.py                 # Vehicle physics
│   └── bsplineplanner.py                 # B-spline path planner
│
├── pyproject.toml                        # Package config (uv)
├── uv.lock                               # Dependency lock file
└── README.md                             # This file
```

---

## 9. References

### Methods

- **MGIGO-MPC**: Information-Geometric Optimization with Gaussian Mixture Models
  for Model Predictive Control
- **IGO**: Information-Geometric Optimization (Ollivier et al., 2017)
- **MMOG-IGO**: Multi-Modal Optimization via Gaussian mixture IGO
- **RNE**: Recursive Nash Equilibrium via sample-based best-response estimation
- **MCTS**: Monte Carlo Tree Search with UCB1 (Kocsis & Szepesvári, 2006)

### For Engineering Usage:
- Lipeng Qi's CUDA-accelerated GMM-IGO: https://github.com/qlp71/IOC_AGV.git

---

*"You write the objective and constraints. The solver handles the rest."*
