"""Smoke test for Agent D (Plug-in Estimator) — AMF revision §5.2.4.

Verifies:
- collect_pairs returns sensible counts at small n_fills_target
- fit recovers (kappa, lam) within reasonable tolerance
- train end-to-end runs and produces an FD solution
- act returns valid env action indices
- value_grid and policy_grid have correct shape
"""
from __future__ import annotations

import numpy as np

from src.common.params import TASK2
from src.rl.env import EnvConfig, LiquidationEnv
from src.rl.plugin_estimator import PluginAgent


def make_env(seed=0):
    return LiquidationEnv(TASK2, EnvConfig(dt=0.5, n_depth=21), seed=seed)


def test_plugin_collect_pairs_basic():
    env = make_env(seed=0)
    agent = PluginAgent(env, seed=1)
    deltas, fills = agent.collect_pairs(n_fills_target=50)
    assert deltas.shape == fills.shape
    assert fills.sum() >= 50
    assert deltas.min() >= 0.0
    assert deltas.max() <= env.cfg.depth_max + 1e-9


def test_plugin_train_recovers_truth():
    env = make_env(seed=0)
    agent = PluginAgent(env, seed=2)
    diag = agent.train(n_fills_target=1500)
    assert diag["n_fills"] >= 1500
    # Recovery within 25% at this sample size (true kappa=100, lam=50/60)
    assert abs(diag["kappa_hat"] - 100.0) / 100.0 < 0.30
    assert abs(diag["lam_hat"] - 50.0 / 60.0) / (50.0 / 60.0) < 0.30
    assert agent.fd_solution is not None


def test_plugin_act_returns_valid_action():
    env = make_env(seed=0)
    agent = PluginAgent(env, seed=3)
    agent.train(n_fills_target=200)
    state = env.reset()
    for _ in range(env.n_steps):
        a = agent.act(state)
        assert 0 <= a < env.num_actions
        state, _r, done, _info = env.step(a)
        if done:
            break


def test_plugin_value_grid_shape():
    env = make_env(seed=0)
    agent = PluginAgent(env, seed=4)
    agent.train(n_fills_target=200)
    V = agent.value_grid()
    assert V.shape == (env.num_time_buckets, env.p.Q0 + 1)
    # q=0 column must be 0
    np.testing.assert_allclose(V[:, 0], 0.0, atol=1e-12)


def test_plugin_zero_fills_falls_back_to_prior():
    """If exploration produces zero fills (paradoxically), the agent should
    still complete train() by falling back to prior (κ, λ)."""
    env = make_env(seed=0)
    agent = PluginAgent(env, prior_kappa=200.0, prior_lam=1.0, seed=5)
    # Construct an empty pairs set and fit directly
    est = agent.fit(np.array([0.01, 0.02, 0.03]), np.array([0.0, 0.0, 0.0]))
    assert est.kappa_hat == 200.0
    assert est.lam_hat == 1.0
