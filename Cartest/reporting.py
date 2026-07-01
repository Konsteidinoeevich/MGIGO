"""Step‑by‑step recording for MPC visualisation.

``StepReport`` collects everything needed to render one MPC frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class StepReport:
    """Single MPC step snapshot for animation / logging."""
    step:      int
    hx:        np.ndarray   # execution history x
    hy:        np.ndarray   # execution history y
    hv:        np.ndarray   # execution history v
    px:        np.ndarray   # planned path x
    py:        np.ndarray   # planned path y
    sp:        np.ndarray   # planned speed profile
    a_long:    np.ndarray   # planned a_long profile
    a_lat:     np.ndarray   # planned a_lat profile
    jm:        np.ndarray   # planned jerk magnitude profile
    solve_ms:  float        # solver time in ms
    min_obs:   float        # minimum obstacle distance
    max_along: float        # max |a_long|
    max_alat:  float        # max |a_lat|
    max_jerk:  float        # max |jerk|
    cost:      float        # total cost
    g_values:  dict         # per‑constraint g values

    def print_line(self):
        return (f"step {self.step:3d} | "
                f"x={self.hx[-1]:6.1f} y={self.hy[-1]:5.1f} "
                f"v={self.hv[-1]:5.1f} | "
                f"a_long={self.max_along:5.1f} a_lat={self.max_alat:5.1f} "
                f"jerk={self.max_jerk:5.1f} | "
                f"obs={self.min_obs:4.1f}m | {self.solve_ms:5.0f}ms")

    def print_cost(self):
        g = self.g_values
        return (f"        cost={self.cost:.4f} "
                f"g=[lane={g['lane']:.2f} obs={g['obs']:.2f} "
                f"jerk={g['jerk']:.1f} acc={g['acc']:.1f} spd={g['spd']:.1f}]")

    def to_frame_dict(self):
        return dict(
            hx=self.hx, hy=self.hy, hv=self.hv,
            px=self.px, py=self.py, sp=self.sp,
            al=self.a_long, alat=self.a_lat, jl=self.jm,
            mo=self.solve_ms, md=self.min_obs,
            ma=self.max_along, ml=self.max_alat, mj=self.max_jerk,
        )
