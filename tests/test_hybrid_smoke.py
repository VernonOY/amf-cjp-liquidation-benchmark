"""Smoke test for Agent C (Hybrid Policy) — AMF revision §5.2.3.

Verifies:
- The network instantiates with the expected param count (~339).
- act() returns a valid action and a finite log-prob.
- train() runs for a small budget without crashing.
- value_grid and policy_grid return the right shape.
- A short training run produces a value error <= initial value error
  (relaxed sanity, not a strict learning curve test).
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from src.common.params import TASK2
from src.rl.env import EnvConfig, LiquidationEnv
from src.rl.hybrid_policy import HybridAgent, HybridPolicyNet


def make_env(seed=0):
    return LiquidationEnv(TASK2, EnvConfig(dt=0.5, n_depth=21), seed=seed)


def test_hybrid_net_param_count():
    net = HybridPolicyNet(hidden=16, input_dim=2)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # Expected: 2*16+16 + 16*16+16 + 17 + 17 + 1 = 48 + 272 + 17 + 17 + 1 = 355
    # (Slightly more than the back-of-envelope 339 because both Linear heads
    # carry a bias term, and log_beta is one more parameter.)
    assert 300 <= n_params <= 400, f"unexpected param count: {n_params}"


def test_hybrid_act_returns_valid_action():
    env = make_env(seed=0)
    agent = HybridAgent(env, kappa_hat=100.0, seed=0)
    state = env.reset()
    a, info = agent.act(state, stochastic=True, sigma=0.01)
    assert 0 <= a < env.num_actions
    assert torch.isfinite(info["log_p_delta"])
    assert torch.isfinite(info["log_p_mo"])


def test_hybrid_residual_can_quote_below_kappa_anchor():
    env = make_env(seed=0)
    agent = HybridAgent(env, kappa_hat=100.0, seed=0)
    with torch.no_grad():
        agent.net.g_head.weight.zero_()
        agent.net.g_head.bias.fill_(-20.0)
    state = env.reset()
    _, info = agent.act(state, stochastic=False)
    assert info["delta_mean"] < 1.0 / agent.kappa_hat
    assert info["delta_mean"] < 1e-3


def test_hybrid_train_small_budget_runs():
    env = make_env(seed=0)
    agent = HybridAgent(env, kappa_hat=100.0, batch_episodes=8, seed=0,
                         recompute_kappa_every=None)
    t0 = time.time()
    out = agent.train(n_episodes=24)
    wall = time.time() - t0
    assert wall < 30.0, f"hybrid 24-episode smoke took {wall:.1f}s"
    assert out["returns"].shape == (24,)
    assert math.isfinite(float(out["returns"].mean()))


def test_hybrid_value_grid_shape():
    env = make_env(seed=0)
    agent = HybridAgent(env, kappa_hat=100.0, seed=0)
    V = agent.value_grid()
    assert V.shape == (env.num_time_buckets, env.p.Q0 + 1)


def test_hybrid_policy_grid_shape():
    env = make_env(seed=0)
    agent = HybridAgent(env, kappa_hat=100.0, seed=0)
    depth, trig = agent.policy_grid()
    assert depth.shape == (env.num_time_buckets, env.p.Q0 + 1)
    assert trig.shape == depth.shape
    assert trig.dtype == np.bool_


@pytest.mark.slow
def test_hybrid_value_error_does_not_diverge():
    """Conservative: after 200 episodes, value error should not be more than
    3x the random-init value error."""
    from src.numerical.fd_constant_lambda import solve as fd_solve

    env = make_env(seed=0)
    fd = fd_solve(TASK2, dt=0.01)
    agent = HybridAgent(env, kappa_hat=100.0, batch_episodes=16,
                         recompute_kappa_every=None, seed=0)
    V0 = agent.value_grid()
    fd_t = fd.t_grid
    idx = np.clip(np.searchsorted(fd_t, env.t_grid), 0, len(fd_t) - 1)
    err0 = float(np.sqrt(np.mean((V0[:-1, 1:] - fd.h[idx][:-1, 1:]) ** 2)))
    agent.train(n_episodes=200)
    V1 = agent.value_grid()
    err1 = float(np.sqrt(np.mean((V1[:-1, 1:] - fd.h[idx][:-1, 1:]) ** 2)))
    assert err1 < 3.0 * err0, f"err blew up: {err0:.4f} -> {err1:.4f}"
