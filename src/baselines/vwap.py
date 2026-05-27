"""VWAP baseline: U-shape intraday volume profile.

Synthetic intraday volume v(t) = a + b * ((t - T/2) / T)^2 (parabolic open/
close-heavy "smile"). The agent fires MOs in proportion to the cumulative
volume profile (quantile-matched discretisation).
"""
from __future__ import annotations

import numpy as np

from ..common.params import Params


def make_vwap_policy(p: Params, *, a: float = 0.5, b: float = 4.0):
    """U-shape volume baseline. a sets the trough, b controls the smile depth."""
    n_grid = 4001
    t_grid = np.linspace(0.0, p.T, n_grid)
    v = a + b * ((t_grid - p.T / 2.0) / p.T) ** 2
    cv = np.cumsum(v)
    cv = cv / cv[-1]   # normalised cumulative volume

    # Quantile slot times: when cumulative volume reaches k / Q0, fire share k
    slot_times = np.empty(p.Q0)
    for k in range(p.Q0):
        target = (k + 1) / p.Q0
        idx = int(np.searchsorted(cv, target, side="left"))
        slot_times[k] = t_grid[min(idx, n_grid - 1)] - 1e-9

    def policy(t: float, q: int, S: float):
        k_target = int(np.sum(slot_times <= t))
        executed = p.Q0 - q
        fire = executed < k_target
        return (1e9, bool(fire))

    return policy
