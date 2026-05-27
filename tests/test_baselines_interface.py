"""Smoke test for all baselines: signature compliance + monotone inventory.

Every baseline must:
  - Be constructible via `make_<name>_policy(p, **kwargs)`.
  - Return (delta, mo_flag) when called as `policy(t, q, S)`.
  - Drive the simulator to a terminal state with q monotone non-increasing.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.params import TASK2
from src.common.simulator import Simulator
from src.baselines.twap import make_twap_mo_policy
from src.baselines.almgren_chriss import make_almgren_chriss_policy
from src.baselines.vwap import make_vwap_policy
from src.baselines.pov import make_pov_policy
from src.baselines.passive import make_passive_policy
from src.baselines.aggressive import make_aggressive_policy


ALL_BASELINES = [
    ("twap", lambda p: make_twap_mo_policy(p)),
    ("ac_low", lambda p: make_almgren_chriss_policy(p, eta=0.01)),
    ("ac_high", lambda p: make_almgren_chriss_policy(p, eta=0.2)),
    ("vwap", lambda p: make_vwap_policy(p)),
    ("pov", lambda p: make_pov_policy(p, rho=0.10)),
    ("passive", lambda p: make_passive_policy(p)),
    ("aggressive", lambda p: make_aggressive_policy(p)),
]


@pytest.mark.parametrize("name,factory", ALL_BASELINES, ids=[x[0] for x in ALL_BASELINES])
def test_baseline_simulator_compatible(name, factory):
    p = TASK2
    sim = Simulator(p, dt=0.05, seed=0)
    policy = factory(p)
    path = sim.simulate(policy)
    # Inventory must be monotone non-increasing on the path
    assert (path.q[1:] <= path.q[:-1]).all(), f"{name}: q not monotone"
    # All depths must be finite and non-negative
    assert np.all(path.delta >= 0.0), f"{name}: negative depth"
    # Terminal value must be finite
    assert np.isfinite(path.terminal_value(p)), f"{name}: NaN terminal value"


def test_almgren_chriss_reduces_to_twap_in_limit():
    """As eta -> 0, AC schedule should match TWAP slot times exactly."""
    p = TASK2
    twap = make_twap_mo_policy(p)
    ac = make_almgren_chriss_policy(p, eta=1e-6)

    # Both should fire at multiples of T/Q0 (with eta -> 0).
    t_grid = np.linspace(0.0, p.T, 121)
    diffs = 0
    for q in range(1, p.Q0 + 1):
        for t in t_grid:
            _, mo_t = twap(t, q, 30.0)
            _, mo_a = ac(t, q, 30.0)
            if mo_t != mo_a:
                diffs += 1
    # Allow at most a handful of boundary differences from int rounding
    assert diffs <= 3, f"AC at eta=0 differs from TWAP at {diffs} grid points"


def test_pov_at_rho_one_matches_external_arrival_rate():
    p = TASK2
    pov = make_pov_policy(p, rho=1.0)
    # At t = 1 second, expected external MOs = lam * 1 ~ 0.83, so still 0 fires
    _, fire = pov(0.5, p.Q0, 30.0)
    assert fire is False
    # At t close to T, target_fired saturates at Q0
    _, fire = pov(p.T, 1, 30.0)
    assert fire is True
