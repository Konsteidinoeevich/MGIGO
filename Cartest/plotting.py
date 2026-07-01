"""Plotting utilities for MPC trajectory visualisation."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import numpy as np


def setup_axes():
    """Create (trajectory, kinematics) twin axes for MPC animation."""
    plt.ion()
    fig, (ax_traj, ax_kin) = plt.subplots(1, 2, figsize=(16, 5))
    return fig, ax_traj, ax_kin


def render_frame(ax_traj, ax_kin, report, obstacles, obs_safe_dist, gen_dt):
    """Render one MPC frame onto the given axes."""
    ax_traj.cla(); ax_kin.cla()

    # Trajectory subplot
    ax_traj.plot(report.hx, report.hy, "g-", lw=2, label="exec")
    ax_traj.plot(report.px, report.py, "b--", lw=2, label="plan")
    for o in obstacles:
        ax_traj.add_patch(Circle((o["x"], o["y"]), o["r"], fc="r", alpha=.3))
        ax_traj.add_patch(Circle((o["x"], o["y"]), o["r"] + obs_safe_dist,
                                  fc="none", ec="r", alpha=.3, ls="--"))
    ax_traj.set_aspect("equal"); ax_traj.legend(loc="upper left"); ax_traj.grid(alpha=.25)
    ax_traj.set_xlim(report.hx[0] - 10, report.px.max() + 20)
    ax_traj.set_ylim(-6, 6)
    ax_traj.set_title(
        f"step {report.step}  v={report.hv[-1]:.1f}  "
        f"obs={report.min_obs:.1f}m  {report.solve_ms:.0f}ms")

    # Kinematics subplot
    t_arr = np.arange(len(report.sp)) * gen_dt
    ax_kin.plot(t_arr, report.a_long, label="a_long")
    ax_kin.plot(t_arr, report.a_lat, label="a_lat")
    ax_kin.plot(t_arr, report.jm, label="jerk")
    ax_kin.legend(); ax_kin.grid(alpha=.25)
    ax_kin.set_title(
        f"max |a_long|={report.max_along:.1f}  "
        f"|a_lat|={report.max_alat:.1f}  jerk={report.max_jerk:.1f}")


def save_animation(fig, frames, render_fn, output_path, fps=12):
    """Create and save an MPC animation."""
    anim = FuncAnimation(fig, render_fn, frames=len(frames), interval=80)
    anim.save(str(output_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {output_path}")
