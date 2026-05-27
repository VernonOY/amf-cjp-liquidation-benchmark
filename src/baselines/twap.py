"""TWAP (time-weighted average price) baseline policies.

Migrated from `src.common.twap` (Phase 1, AMF revision). Same closed-form
behaviour; relocated to `src.baselines.twap` per the new layout. The unified
`make_<name>_policy(p, **kwargs)` baseline interface (Phase 3) will sit
alongside this file.
"""
from __future__ import annotations

import numpy as np

from ..common.params import Params


def make_twap_mo_policy(p: Params):
    """Evenly-spaced MO schedule: split Q0 into Q0 buckets of 1 share across T."""
    slot_times = np.linspace(p.T / p.Q0, p.T, p.Q0) - 1e-9

    def policy(t: float, q: int, S: float):
        k_target = int(np.sum(slot_times <= t))
        executed = p.Q0 - q
        fire = executed < k_target
        return (1e9, bool(fire))
    return policy


def make_twap_lo_policy(p: Params, depth: float):
    """Fixed-depth LO baseline. Unhit inventory penalised via terminal condition."""
    def policy(t: float, q: int, S: float):
        return (depth, False)
    return policy
