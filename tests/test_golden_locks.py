"""Lock tests: assert the current codebase reproduces the golden snapshots.

These tests MUST stay green throughout Phases 1–5 of the AMF revision. Any
failure means a refactor has changed the numerical behavior of a frozen
interface, which is forbidden without explicit re-baselining.

Snapshots are produced by `tests/_golden/regenerate.py`. They cover:
  - FD solver h-grid on TASK2 (the reference for every RL diagnostic)
  - Analytic omega_q1, omega_q2 on a fine grid
  - Monte Carlo summary of the LO-only analytic policy on TASK1
  - Step-by-step (state, action, reward) trace of the legacy env

Bit-identical comparison (atol=rtol=0) is used wherever the upstream code is
itself deterministic (FD solver, analytic, env step). The MC test uses a
tighter `assert_allclose` with rtol=0 atol=0 — the simulator is seeded and
event-driven, so it should also be bit-identical.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.common.params import TASK1, TASK2  # noqa: E402
from src.common.simulator import Simulator  # noqa: E402


# ---------------------------------------------------------------------------
# FD solver h-grid on TASK2
# ---------------------------------------------------------------------------
def test_fd_h_task2_bit_identical(golden_dir):
    from src.numerical.fd_constant_lambda import solve as fd_solve

    snap = np.load(golden_dir / "h_fd_task2.npz")
    sol = fd_solve(TASK2, dt=float(snap["dt"]))
    np.testing.assert_array_equal(sol.h, snap["h"])
    np.testing.assert_array_equal(sol.mo_trigger, snap["mo_trigger"])
    np.testing.assert_array_equal(sol.t_grid, snap["t_grid"])


# ---------------------------------------------------------------------------
# Analytic q=1, q=2 omega curves
# ---------------------------------------------------------------------------
def test_omega_q1_q2_bit_identical(golden_dir):
    from src.analytical.task2_lo_mo import omega_q1, omega_q2

    snap = np.load(golden_dir / "omega_q1q2.npz")
    t = snap["t"]
    om1 = np.asarray(omega_q1(TASK2, t), dtype=np.float64)
    om2 = np.asarray(omega_q2(TASK2, t), dtype=np.float64)
    np.testing.assert_array_equal(om1, snap["omega_q1"])
    np.testing.assert_array_equal(om2, snap["omega_q2"])


# ---------------------------------------------------------------------------
# Monte Carlo with seeded simulator
# ---------------------------------------------------------------------------
def test_mc_task1_seed0_bit_identical(golden_dir):
    from src.analytical.task1_lo_only import make_lo_only_policy as make_task1_policy

    snap = np.load(golden_dir / "mc_task1_seed0.npz")
    sim = Simulator(TASK1, dt=0.05, seed=0)
    policy, _ = make_task1_policy(TASK1, sim.t_grid)
    out = sim.monte_carlo(policy, n_paths=2000, progress=False)

    np.testing.assert_array_equal(out["terminal"], snap["terminal"])
    np.testing.assert_array_equal(out["q_final"], snap["q_final"])
    np.testing.assert_array_equal(out["n_mo"], snap["n_mo"])
    np.testing.assert_array_equal(out["n_lo"], snap["n_lo"])
    np.testing.assert_array_equal(out["avg_price"], snap["avg_price"], strict=False)
    np.testing.assert_array_equal(out["twap_price"], snap["twap_price"])
    np.testing.assert_array_equal(out["t_grid"], snap["t_grid"])


# ---------------------------------------------------------------------------
# Env step-by-step trace (the key Phase 2 regression guard)
# ---------------------------------------------------------------------------
def test_env_trace_seed0_bit_identical(golden_dir):
    from src.rl.env import LiquidationEnv, EnvConfig

    snap = np.load(golden_dir / "env_trace_seed0.npz")
    env = LiquidationEnv(
        TASK2,
        EnvConfig(dt=float(snap["cfg_dt"]), n_depth=int(snap["cfg_n_depth"])),
        seed=int(snap["env_seed"]),
    )
    state = env.reset()
    states_i = [state[0]]
    states_q = [state[1]]
    rewards = []
    for a in snap["actions"]:
        next_state, r, _done, _info = env.step(int(a))
        rewards.append(r)
        states_i.append(next_state[0])
        states_q.append(next_state[1])

    np.testing.assert_array_equal(np.asarray(states_i, dtype=np.int64), snap["states_i"])
    np.testing.assert_array_equal(np.asarray(states_q, dtype=np.int64), snap["states_q"])
    np.testing.assert_array_equal(np.asarray(rewards, dtype=np.float64), snap["rewards"])


# ---------------------------------------------------------------------------
# Sanity meta-test: confirm all four snapshot files exist
# ---------------------------------------------------------------------------
EXPECTED_SNAPSHOTS = [
    "h_fd_task2.npz",
    "omega_q1q2.npz",
    "mc_task1_seed0.npz",
    "env_trace_seed0.npz",
]


@pytest.mark.parametrize("name", EXPECTED_SNAPSHOTS)
def test_snapshot_present(golden_dir, name):
    assert (golden_dir / name).exists(), (
        f"Missing snapshot {name}. Run `python3 tests/_golden/regenerate.py` "
        f"to (re)create — but only intentionally."
    )
