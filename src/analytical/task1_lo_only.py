"""Analytic LO-only solver (Cartea, Jaimungal & Penalva 2015, §8.2).

Value-function ansatz:
    H(t, x, S, q) = x + q S + h(t, q),                              (8.5)
    h(t, q) = (1 / kappa) log omega(t, q),
with
    omega(t, q) = sum_{n=0..q} (lam_tilde^n / n!)
                  * exp(-kappa alpha (q - n)^2) (T - t)^n,           (8.10)
    lam_tilde = lam / e.

Optimal feedback depth:
    delta*(t, q) = (1 / kappa) [1 + log(omega(t, q) / omega(t, q-1))], (8.11)
valid for q >= 1.

This module was migrated from `src.task1_lo_only.solver` as part of the AMF
revision (Phase 1). The formulae are locked and become the analytical
reference for every RL agent's diagnostic plots.
"""
from __future__ import annotations
import math

import numpy as np

from ..common.params import Params


def omega(p: Params, t: float, q: int) -> float:
    if q <= 0:
        return 1.0
    tau = max(p.T - t, 0.0)
    lam_tilde = p.lam / math.e
    total = 0.0
    for n in range(q + 1):
        term = (lam_tilde ** n) / math.factorial(n)
        term *= math.exp(-p.kappa * p.alpha * (q - n) ** 2)
        term *= tau ** n
        total += term
    return total


def h_func(p: Params, t: float, q: int) -> float:
    if q <= 0:
        return 0.0
    return math.log(omega(p, t, q)) / p.kappa


def optimal_depth(p: Params, t: float, q: int) -> float:
    """delta*(t,q) per (8.11). Returns 0 if q <= 0."""
    if q <= 0:
        return 0.0
    num = omega(p, t, q)
    den = omega(p, t, q - 1)
    return (1.0 / p.kappa) * (1.0 + math.log(num / den))


def precompute_delta_grid(p: Params, t_grid: np.ndarray) -> np.ndarray:
    """Return delta[i, q] = delta*(t_grid[i], q) for q = 0..Q0."""
    M = len(t_grid)
    grid = np.zeros((M, p.Q0 + 1))
    for i, ti in enumerate(t_grid):
        for q in range(p.Q0 + 1):
            grid[i, q] = optimal_depth(p, ti, q)
    return grid


def make_lo_only_policy(p: Params, t_grid: np.ndarray):
    """Cached feedback policy. Task 1 uses no MOs, so mo_flag is always False."""
    grid = precompute_delta_grid(p, t_grid)
    dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else p.T

    def policy(t: float, q: int, S: float):
        if q <= 0:
            return (0.0, False)
        i = min(int(round(t / dt)), len(t_grid) - 1)
        depth = max(grid[i, q], 0.0)
        return (float(depth), False)
    return policy, grid


# Legacy alias retained for shim-compatibility with `src.task1_lo_only.solver`.
make_policy = make_lo_only_policy


def asymptotic_depth(p: Params, t: float, q: int) -> float:
    """Far-from-maturity asymptotic (CJP p.191)."""
    tau = max(p.T - t, 0.0)
    return (1.0 / p.kappa) * (1.0 + math.log(
        math.exp(-p.kappa * p.alpha) + (p.lam_tilde / max(q, 1)) * tau
    ))
