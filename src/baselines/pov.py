"""POV (Percentage Of Volume) baseline.

POV(rho) fires the agent's MO whenever cumulative external (market-wide) MO
arrivals exceed the agent's already-fired share count divided by rho. Since
the simulator does not expose external arrivals in its policy callback, we
substitute a deterministic proxy: at every env step, the expected number of
external MOs that arrived is approximately lam * t. We fire MOs to keep
n_already_fired close to floor(rho * lam * t).

This is a faithful POV approximation for a Poisson external-MO process at
rate lam.
"""
from __future__ import annotations
import math

from ..common.params import Params


def make_pov_policy(p: Params, *, rho: float = 0.10):
    """POV at participation rate rho (default 10%)."""
    rate = max(rho * p.lam, 1e-12)

    def policy(t: float, q: int, S: float):
        target_fired = int(math.floor(rate * t + 1e-9))
        # Cap at Q0 so we don't try to over-fire
        target_fired = min(target_fired, p.Q0)
        n_already_fired = p.Q0 - q
        fire = n_already_fired < target_fired
        return (1e9, bool(fire))

    return policy
