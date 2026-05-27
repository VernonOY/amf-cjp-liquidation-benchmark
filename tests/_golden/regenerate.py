"""Generate (or regenerate) the golden snapshot files in tests/_golden/.

These snapshots lock the current numerical behavior of the codebase so that
the AMF revision refactor (Phases 1–5) can be verified bit-identical at each
phase gate. Run this script ONCE before any refactor work begins, then never
again unless intentionally re-baselining (which requires a written rationale
in the commit message).

Snapshots produced:
- h_fd_task2.npz        — FDSolution.h on TASK2, dt=0.01 (the canonical
                          h(t,q) reference used by every RL diagnostic)
- omega_q1q2.npz        — analytic omega_q1, omega_q2 on linspace(0,T,601)
- mc_task1_seed0.npz    — Simulator(TASK1, dt=0.05, seed=0).monte_carlo
                          using the LO-only analytic policy, n_paths=2000
- env_trace_seed0.npz   — A captured (state, action, reward) trace from the
                          current LiquidationEnv. Used by Phase 2 to verify
                          the refactored env reproduces the legacy reward
                          step-by-step.

Usage:
    python3 tests/_golden/regenerate.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.common.params import TASK1, TASK2  # noqa: E402
from src.common.simulator import Simulator  # noqa: E402
from src.analytical.task1_lo_only import make_lo_only_policy as make_task1_policy  # noqa: E402
from src.analytical.task2_lo_mo import omega_q1, omega_q2  # noqa: E402
from src.numerical.fd_constant_lambda import solve as fd_solve  # noqa: E402
from src.rl.env import LiquidationEnv, EnvConfig  # noqa: E402


GOLDEN_DIR = Path(__file__).resolve().parent


def gen_h_fd_task2():
    sol = fd_solve(TASK2, dt=0.01)
    np.savez_compressed(
        GOLDEN_DIR / "h_fd_task2.npz",
        h=sol.h,
        mo_trigger=sol.mo_trigger,
        t_grid=sol.t_grid,
        dt=0.01,
    )
    print(f"  h_fd_task2.npz  shape={sol.h.shape}  L∞|h|={np.max(np.abs(sol.h)):.6e}")


def gen_omega_q1q2():
    t = np.linspace(0.0, TASK2.T, 601)
    om1 = omega_q1(TASK2, t)
    om2 = omega_q2(TASK2, t)
    np.savez_compressed(
        GOLDEN_DIR / "omega_q1q2.npz",
        t=t,
        omega_q1=np.asarray(om1, dtype=np.float64),
        omega_q2=np.asarray(om2, dtype=np.float64),
    )
    print(f"  omega_q1q2.npz  q1[0]={float(np.asarray(om1)[0]):.6e}  q2[0]={float(np.asarray(om2)[0]):.6e}")


def gen_mc_task1_seed0():
    sim = Simulator(TASK1, dt=0.05, seed=0)
    policy, _ = make_task1_policy(TASK1, sim.t_grid)
    out = sim.monte_carlo(policy, n_paths=2000, progress=False)
    np.savez_compressed(
        GOLDEN_DIR / "mc_task1_seed0.npz",
        terminal=out["terminal"],
        q_final=out["q_final"],
        n_mo=out["n_mo"],
        n_lo=out["n_lo"],
        avg_price=out["avg_price"],
        twap_price=out["twap_price"],
        t_grid=out["t_grid"],
    )
    print(f"  mc_task1_seed0.npz  E[terminal]={out['terminal'].mean():.6e}  E[q_T]={out['q_final'].mean():.4f}")


def gen_env_trace_seed0():
    """Capture a deterministic (state, action, reward) trace for the legacy env.

    Phase 2's refactored env must reproduce these tuples bit-identical when
    given the same actions in the same order. We use a fixed pseudo-random
    action sequence (not learned) so the trace depends only on the env's
    transition + reward functions, not on any agent.
    """
    env = LiquidationEnv(TASK2, EnvConfig(dt=0.5, n_depth=21), seed=0)
    rng = np.random.default_rng(12345)  # independent of env seed
    state = env.reset()
    states_i = [state[0]]
    states_q = [state[1]]
    actions = []
    rewards = []
    done = False
    while not done:
        a = int(rng.integers(env.num_actions))
        actions.append(a)
        next_state, r, done, _info = env.step(a)
        rewards.append(r)
        states_i.append(next_state[0])
        states_q.append(next_state[1])
        state = next_state
    np.savez_compressed(
        GOLDEN_DIR / "env_trace_seed0.npz",
        states_i=np.asarray(states_i, dtype=np.int64),
        states_q=np.asarray(states_q, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float64),
        env_seed=0,
        action_rng_seed=12345,
        cfg_dt=0.5,
        cfg_n_depth=21,
    )
    print(f"  env_trace_seed0.npz  n_steps={len(actions)}  sum(r)={sum(rewards):.6e}  q_final={states_q[-1]}")


def main():
    print(f"Regenerating golden snapshots in {GOLDEN_DIR}")
    gen_h_fd_task2()
    gen_omega_q1q2()
    gen_mc_task1_seed0()
    gen_env_trace_seed0()
    print("Done.")


if __name__ == "__main__":
    main()
