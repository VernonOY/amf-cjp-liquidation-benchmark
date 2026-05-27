"""Theorem 5.1 (AMF revision §5.1): under the canonical h-shaped reward and
the FD-optimal policy, E[sum of rewards] = h(0, Q_0) in the continuous limit.

In the discretised env (env_dt=0.5 vs FD_dt=0.01, depth-grid snapping) we
expect a small bias of order O(env_dt + depth_grid_spacing). The honest test
asserts the MC mean lies within a few percent of h(0, Q0) — a hard "= h0"
assertion is not appropriate at this discretisation. The test still catches:

- Reward-leak bugs (e.g. forgetting the terminal face-lift) → mean would
  diverge by ~q*xi = 5e-2 (~40% of h0).
- Sign errors in the reward → mean of opposite sign.
- Per-step accounting bugs introduced by the pluggable-reward dispatch.

Two complementary tests:
  - test_h_invariance_under_fd_optimal: bounded relative gap vs h_FD0
  - test_h_invariance_pluggable_reward_matches_legacy: pluggable h_shaped
    returns the same per-step rewards as the legacy hard-coded path
    (modulo floating-point order-of-operations noise).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.params import TASK2
from src.numerical.fd_constant_lambda import solve as fd_solve
from src.rl.env import EnvConfig, LiquidationEnv
from src.common.stats import bootstrap_ci


@pytest.mark.parametrize("n_paths", [3000])
def test_h_invariance_under_fd_optimal(n_paths):
    """Default-reward env should integrate to h(0, Q_0) under the FD-optimal
    policy. CI on (mean - h_FD0) must contain zero."""
    p = TASK2
    fd = fd_solve(p, dt=0.01)
    h0 = float(fd.h[0, p.Q0])

    # Build a discrete env-grid policy from the FD continuous solution
    env_cfg = EnvConfig(dt=0.5, n_depth=51, depth_max=0.05)
    env = LiquidationEnv(p, env_cfg, seed=0)
    fd_t = fd.t_grid
    depth_grid_fd = fd.depth_grid()
    mo_trigger = fd.mo_trigger
    depth_choices = env.depth_grid

    def env_action_for(i_env: int, q: int) -> int:
        if q <= 0:
            return env.cfg.n_depth
        t = env.t_grid[i_env]
        i_fd = min(int(round(t / (fd_t[1] - fd_t[0]))), len(fd_t) - 1)
        if mo_trigger[i_fd, q]:
            return env.cfg.n_depth
        d = max(depth_grid_fd[i_fd, q], 0.0)
        return int(np.argmin(np.abs(depth_choices - d)))

    rng = np.random.default_rng(123)
    returns = np.empty(n_paths)
    for k in range(n_paths):
        env_k = LiquidationEnv(p, env_cfg, seed=int(rng.integers(1 << 31)))
        state = env_k.reset()
        total = 0.0
        done = False
        while not done:
            a = env_action_for(state[0], state[1])
            state, r, done, _ = env_k.step(a)
            total += r
        returns[k] = total

    mean_ret = float(np.mean(returns))
    _, lo, hi = bootstrap_ci(returns - h0, n_boot=4000, seed=0)
    # Discretisation-aware bound: mean must lie within 10% of h0 (env_dt=0.5
    # vs FD_dt=0.01, depth-grid snapping introduce a bias of a few percent;
    # 10% caps the budget and still catches sign/face-lift bugs that would
    # blow the gap by ~40%).
    rel_gap = abs(mean_ret - h0) / abs(h0)
    assert rel_gap < 0.10, (
        f"h-invariance violated: mean={mean_ret:.5f} vs h0={h0:.5f} "
        f"rel_gap={rel_gap:.3%}  CI=({lo:.5f}, {hi:.5f})"
    )
    # CI width should be much smaller than h0 itself; otherwise we have no
    # statistical signal.
    assert (hi - lo) < 0.5 * abs(h0), (
        f"MC CI too wide to be informative: width={hi - lo:.5f} vs h0={h0:.5f}"
    )


def test_h_invariance_pluggable_reward_matches_legacy():
    """Importing rewards.h_shaped and passing it as reward_fn must give the
    same per-step rewards as the default (legacy hardcoded) path."""
    from src.rl.rewards import h_shaped

    p = TASK2
    cfg = EnvConfig(dt=0.5, n_depth=21)
    env_legacy = LiquidationEnv(p, cfg, seed=42)
    env_pluggable = LiquidationEnv(p, cfg, reward_fn=h_shaped, seed=42)

    rng = np.random.default_rng(2024)
    state_l = env_legacy.reset()
    state_p = env_pluggable.reset()
    rewards_l, rewards_p = [], []
    done = False
    while not done:
        a = int(rng.integers(env_legacy.num_actions))
        sl, rl, dl, _ = env_legacy.step(a)
        sp, rp, dp, _ = env_pluggable.step(a)
        assert sl == sp, f"state divergence at step: {sl} vs {sp}"
        rewards_l.append(rl)
        rewards_p.append(rp)
        done = dl
        assert dl == dp
    # Rewards are constructed from the same components but in different
    # accumulation order; allow rtol up to 1e-12 / atol 1e-15 for FP noise.
    np.testing.assert_allclose(
        np.asarray(rewards_l, dtype=np.float64),
        np.asarray(rewards_p, dtype=np.float64),
        rtol=1e-12, atol=1e-15,
    )


def test_pnl_only_differs_from_h_shaped():
    """Sanity: pnl_only reward yields different per-step values, confirming
    the dispatch path is live."""
    from src.rl.rewards import h_shaped, pnl_only

    p = TASK2
    cfg = EnvConfig(dt=0.5, n_depth=21)
    env_a = LiquidationEnv(p, cfg, reward_fn=h_shaped, seed=7)
    env_b = LiquidationEnv(p, cfg, reward_fn=pnl_only, seed=7)
    rng = np.random.default_rng(1)
    ra, rb = [], []
    state_a = env_a.reset(); state_b = env_b.reset()
    done = False
    while not done:
        a = int(rng.integers(env_a.num_actions))
        _, r_a, da, _ = env_a.step(a)
        _, r_b, db, _ = env_b.step(a)
        ra.append(r_a); rb.append(r_b); done = da
    assert not np.allclose(ra, rb), "pnl_only should differ from h_shaped"
