"""Explicit finite-difference solver for the constant-lambda QVI of the
Cartea, Jaimungal & Penalva 2015 §8 optimal-execution problem.

With h(t, q) = (1 / kappa) log omega(t, q), the reduced QVI reads
    max{ d_t h - phi q^2 + (e^{-1} lam / kappa) exp(-kappa [h(t,q) - h(t,q-1)]) ;
         -xi + h(t, q-1) - h(t, q) } = 0,
with h(t, 0) = 0 and face-lifted terminal
h(T^-, q) = max(-alpha q^2, -q xi).

This module was migrated from `src.task2_lo_mo.fd_solver` as part of the AMF
revision (Phase 1). The FDSolution interface (`h`, `mo_trigger`,
`make_policy()`) is FROZEN — downstream RL diagnostics depend on it.
"""
from __future__ import annotations
import math

import numpy as np

from ..common.params import Params


class FDSolution:
    """Container for the FD solution and feedback policies."""

    def __init__(self, p: Params, t_grid: np.ndarray,
                 h: np.ndarray, mo_trigger: np.ndarray):
        self.p = p
        self.t_grid = t_grid
        self.h = h
        self.mo_trigger = mo_trigger

    def depth(self, t_idx: int, q: int) -> float:
        if q <= 0:
            return 0.0
        return 1.0 / self.p.kappa + self.h[t_idx, q] - self.h[t_idx, q - 1]

    def depth_grid(self) -> np.ndarray:
        g = np.zeros_like(self.h)
        for q in range(1, g.shape[1]):
            g[:, q] = 1.0 / self.p.kappa + self.h[:, q] - self.h[:, q - 1]
        return g

    def make_policy(self):
        dt = self.t_grid[1] - self.t_grid[0]
        trigger = self.mo_trigger
        depth_grid = self.depth_grid()
        M = len(self.t_grid) - 1

        def policy(t: float, q: int, S: float):
            if q <= 0:
                return (0.0, False)
            i = min(int(round(t / dt)), M)
            if trigger[i, q]:
                return (0.0, True)
            delta = max(depth_grid[i, q], 0.0)
            return (float(delta), False)
        return policy


def terminal_face_lift(p: Params, q: int) -> float:
    """Face-lifted terminal value used by the LO+MO QVI.

    At T^- the agent can either accept the terminal quadratic block-impact
    penalty or cross the spread with q unit market orders.  The dynamic
    programme starts from the better of these two liquidation modes.
    """
    return max(-p.alpha * q * q, -p.xi * q)


def solve(p: Params, dt: float = 0.01, Q_max: int | None = None) -> FDSolution:
    """Solve the reduced QVI backward in time using explicit Euler.

    Stability requires dt * (lam / (e kappa)) * exp(kappa * d_max) < 1.
    For book parameters (lam=50/60, kappa=100, depths ~0.03), dt=0.01 is safe.
    """
    if Q_max is None:
        Q_max = p.Q0
    M = int(round(p.T / dt))
    t_grid = np.linspace(0.0, p.T, M + 1)
    h = np.zeros((M + 1, Q_max + 1))
    trigger = np.zeros_like(h, dtype=bool)

    h[M, 0] = 0.0
    for q in range(1, Q_max + 1):
        h[M, q] = terminal_face_lift(p, q)
        trigger[M, q] = (-p.xi * q) >= (-p.alpha * q * q)

    lam_scaled = p.lam / (math.e * p.kappa)

    for i in range(M - 1, -1, -1):
        h[i, 0] = 0.0
        for q in range(1, Q_max + 1):
            hq_next = h[i + 1, q]
            hqm1_next = h[i + 1, q - 1]
            drift = p.phi * q ** 2 - lam_scaled * math.exp(
                -p.kappa * (hq_next - hqm1_next)
            )
            h_cont = hq_next - dt * drift
            h_exec = h[i, q - 1] - p.xi
            if h_exec >= h_cont:
                h[i, q] = h_exec
                trigger[i, q] = True
            else:
                h[i, q] = h_cont
                trigger[i, q] = False

    return FDSolution(p, t_grid, h, trigger)


def validate_against_analytic(p: Params, dt: float = 0.005):
    """Cross-check FD vs analytic q=1, 2 solutions."""
    from ..analytical import task2_lo_mo as aq
    sol = solve(p, dt=dt, Q_max=max(p.Q0, 2))
    t = sol.t_grid
    ome_q1_fd = np.exp(p.kappa * sol.h[:, 1])
    ome_q1_an = aq.omega_q1(p, t)
    err1 = np.max(np.abs(ome_q1_fd - ome_q1_an) / np.abs(ome_q1_an))

    ome_q2_fd = np.exp(p.kappa * sol.h[:, 2])
    ome_q2_an = aq.omega_q2(p, t)
    err2 = np.max(np.abs(ome_q2_fd - ome_q2_an) / np.abs(ome_q2_an))

    return {"err_q1": float(err1), "err_q2": float(err2),
            "ome_q1_fd": ome_q1_fd, "ome_q1_an": ome_q1_an,
            "ome_q2_fd": ome_q2_fd, "ome_q2_an": ome_q2_an,
            "t_grid": t, "sol": sol}
