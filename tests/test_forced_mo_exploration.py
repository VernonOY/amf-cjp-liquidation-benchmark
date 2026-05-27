"""Forced-MO exploration hooks used for the fairness ablation."""
from __future__ import annotations

from src.common.params import TASK2
from src.rl.double_dqn import DDQN
from src.rl.env import EnvConfig, LiquidationEnv
from src.rl.tabular_q import TabularQ


def make_env(seed=0):
    return LiquidationEnv(TASK2, EnvConfig(dt=0.5, n_depth=21), seed=seed)


def test_tabular_forced_mo_exploration_action():
    env = make_env()
    agent = TabularQ(env, seed=0)
    state = env.reset()
    assert agent.act(state, eps=0.0, mo_explore=1.0) == env.cfg.n_depth


def test_ddqn_forced_mo_exploration_action():
    env = make_env()
    agent = DDQN(env, seed=0)
    state = env.reset()
    assert agent.act(state, eps=0.0, mo_explore=1.0) == env.cfg.n_depth
