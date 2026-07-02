"""Scenario configurations — obstacles, road, speed target, initial state.

Each scenario is a dict.  Import the one you want.
"""

SINGLE_OFFSET = {
    "obstacles": [
        {"x": 60.0, "y": 2.5, "r": 2.0},
    ],
    "lane_hw":       4.0,
    "obs_safe_dist": 0.1,   # RSS reaction time (s) = MPC dt
    "v_target":     18.0,
    "init":          {"s": 0.0, "s_dot": 12.0, "s_ddot": 0.0,
                      "d": -3.0, "d_dot":  0.0, "d_ddot": 0.0,
                      "psi": 0.0},
}

THREE_BLOCKING = {
    "obstacles": [
        {"x": 45.0, "y": -2.5, "r": 2.0},
        {"x": 65.0, "y":  0.5, "r": 2.0},
        {"x": 80.0, "y": -1.0, "r": 1.5},
    ],
    "lane_hw":       2.0,
    "obs_safe_dist": 0.1,   # RSS reaction time (s)
    "v_target":     18.0,
    "init":          {"s": 0.0, "s_dot": 12.0, "s_ddot": 0.0,
                      "d": -3.0, "d_dot":  0.0, "d_ddot": 0.0,
                      "psi": 0.0},
}

EMPTY = {
    "obstacles":     [],
    "lane_hw":       4.0,
    "obs_safe_dist": 0.1,   # RSS reaction time (s) = MPC dt
    "v_target":     18.0,
    "init":          {"s": 0.0, "s_dot": 12.0, "s_ddot": 0.0,
                      "d":  0.0, "d_dot":  0.0, "d_ddot": 0.0,
                      "psi": 0.0},
}
