import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from gmm_igo.MPCsolver import igo_mog_optimizer

# =============================================================================
# CONFIG
# =============================================================================

DT = 0.1
H = 10
DIM_U = 2               # [a, delta]
L = 2.5

N_COMPONENTS = 6        # MUST >= 2
POP_SIZE = 60
ELITE_SIZE = 25
OPT_STEPS = 300
WARMUP_STEPS = 1200

TOTAL_DIM = H * DIM_U

# =============================================================================
# Utilities
# =============================================================================

def wrap_angle(theta):
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def bicycle_step(x, u):
    px, py, psi, v = x
    a, delta = u

    px_next = px + DT * v * jnp.cos(psi)
    py_next = py + DT * v * jnp.sin(psi)
    psi_next = psi + DT * v * jnp.tan(delta) / L
    v_next = v + DT * a

    return jnp.array([px_next, py_next, psi_next, v_next])


def rollout(x0, U):
    def step(x, u):
        x_next = bicycle_step(x, u)
        return x_next, x_next

    _, traj = jax.lax.scan(step, x0, U)
    return traj


def body_error(x, target):
    dx = x[0] - target[0]
    dy = x[1] - target[1]

    cg = jnp.cos(target[2])
    sg = jnp.sin(target[2])

    ex =  cg * dx + sg * dy
    ey = -sg * dx + cg * dy
    epsi = wrap_angle(x[2] - target[2])

    return ex, ey, epsi


# =============================================================================
# Terminal potential (NON-monotone, parking-valid)
# =============================================================================

def terminal_potential(x, target):
    ex, ey, epsi = body_error(x, target)
    v = x[3]

    return (
        1.0 * (ex**2 + ey**2)
        + 2.0 * epsi**2
        + 1.5 * v**2
    )


# =============================================================================
# MPC cost (SINGLE cost_fn, SINGLE context)
# =============================================================================

def mpc_cost_fn(flat_u, context):
    U = flat_u.reshape(H, 2)

    x0 = context["x0"]
    target = context["target"]

    traj = rollout(x0, U)

    cost = 0.0

    # running cost: physical regularization only
    for k in range(H):
        a, delta = U[k]
        cost += 0.05 * a**2 + 0.02 * delta**2

    # terminal cost
    cost += 25.0 * terminal_potential(traj[-1], target)

    return cost


# =============================================================================
# Simulation
# =============================================================================

def run():
    key = jax.random.PRNGKey(0)

    # state: [x, y, psi, v]
    x = jnp.array([0.0, 0.0, 0.0, 0.0])

    # target: [x*, y*, psi*]
    target = jnp.array([10.0, 8.0, jnp.pi / 2])

    # -------------------------------------------------------------------------
    # Mixture initialization (HYBRID VIA μ ONLY)
    # -------------------------------------------------------------------------

    mu = []

    # forward / reverse
    for sgn in [+1, -1]:
        U = np.zeros((H, 2))
        U[:, 0] = 0.8 * sgn
        mu.append(U.reshape(-1))

    # left / right steering
    for sgn in [+1, -1]:
        U = np.zeros((H, 2))
        U[:, 1] = 0.4 * sgn
        mu.append(U.reshape(-1))

    # braking / micro reverse
    for sgn in [+1, -1]:
        U = np.zeros((H, 2))
        U[:, 0] = -0.5 * sgn
        mu.append(U.reshape(-1))

    mu = jnp.array(mu)

    L_inv = jnp.stack([
        jnp.eye(TOTAL_DIM) * 2.5 for _ in range(N_COMPONENTS)
    ])

    pi = jnp.ones(N_COMPONENTS) / N_COMPONENTS

    history = [x]

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    context = dict(x0=x, target=target)

    # -------------------------------------------------------------------------
    # MPC loop
    # -------------------------------------------------------------------------

    for t in range(120):
        key, sub = jax.random.split(key)

        context["x0"] = x

        mu, L_inv, pi = igo_mog_optimizer(
            sub,
            OPT_STEPS if t > 0 else WARMUP_STEPS,
            0.1, 6,
            POP_SIZE, ELITE_SIZE,
            mpc_cost_fn,
            mu, L_inv, pi,
            context          # ← SINGLE context (IMPORTANT)
        )

        best = int(jnp.argmax(pi))
        U_best = mu[best].reshape(H, 2)

        # apply first control
        x = bicycle_step(x, U_best[0])
        history.append(x)

        # receding horizon shift
        mu = jnp.roll(mu.reshape(N_COMPONENTS, H, 2), -1, axis=1)
        mu = mu.at[:, -1, :].set(0.0)
        mu = mu.reshape(N_COMPONENTS, -1)

        # ---------------- plotting ----------------
        ax.cla()
        hist = np.array(history)
        ax.plot(hist[:, 0], hist[:, 1], "b.-")
        ax.plot(target[0], target[1], "g*", markersize=12)

        ax.arrow(
            x[0], x[1],
            np.cos(x[2]), np.sin(x[2]),
            head_width=0.3, color="r"
        )

        ax.set_xlim(-2, 14)
        ax.set_ylim(-2, 14)
        ax.set_aspect("equal")
        ax.set_title(f"Hybrid Parking MPC via Mixture-IGO | best={best}")

        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run()
