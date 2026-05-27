"""Almgren-Chriss (2001) deterministic execution schedule.

The continuous schedule for risk-aversion gamma > 0 and impact lam_ac > 0 is
    Q_t = Q_0 * sinh(eta (T - t)) / sinh(eta T),   eta = sqrt(gamma * lam_ac / sigma^2)

In the limit eta -> 0 this reduces to TWAP. We translate the continuous
trajectory into a discrete MO firing schedule on the env time grid: at each
env step we fire an MO whenever the cumulative target Q_target(t) calls for
one more share to be liquidated than has been fired so far.

Interface: matches `simulator.policy(t, q, S) -> (delta, mo_flag)`.
"""
from __future__ import annotations
import math

import numpy as np

from ..common.params import Params


def _ac_schedule(Q0: int, T: float, eta: float) -> callable:
    if eta <= 1e-9:
        # TWAP limit
        def Q_target(t: float) -> float:
            return Q0 * (1.0 - t / T)
        return Q_target
    sinh_eT = math.sinh(eta * T)

    def Q_target(t: float) -> float:
        tau = max(T - t, 0.0)
        return Q0 * math.sinh(eta * tau) / sinh_eT
    return Q_target


def make_almgren_chriss_policy(p: Params, *, eta: float = 0.05):
    """Closure with no state — the simulator passes (t, q, S) each step.

    Parameters
    ----------
    eta : urgency parameter sqrt(gamma * lam_ac / sigma^2). Set 0 to recover
          TWAP exactly.
    """
    Q_target_fn = _ac_schedule(p.Q0, p.T, eta)

    def policy(t: float, q: int, S: float):
        target = Q_target_fn(t)
        n_already_fired = p.Q0 - q
        n_should_be_fired = int(math.floor(p.Q0 - target + 1e-9))
        fire = n_already_fired < n_should_be_fired
        return (1e9, bool(fire))   # huge depth -> LO never fills

    return policy
