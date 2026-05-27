"""Train tabular Q-learning and DDQN on the liquidation environment."""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path as _Path

import numpy as np

from ..common.params import TASK2
from ..task2_lo_mo import fd_solver as fd
from .env import LiquidationEnv, EnvConfig
from .agent_tabular import TabularQ
from .agent_dqn import DDQN


OUT = _Path(__file__).resolve().parents[2] / "results" / "task3"


def reference_h_grid(p, n_time, fd_dt=0.01):
    """Solve FD at fine grid and resample h on (n_time, Q0+1) evaluated at the
    env's time buckets."""
    sol = fd.solve(p, dt=fd_dt)
    t_fd = sol.t_grid
    target_t = np.linspace(0.0, p.T, n_time)
    idx = np.clip(np.searchsorted(t_fd, target_t), 0, len(t_fd) - 1)
    h_resampled = sol.h[idx]  # (n_time, Q0+1)
    return h_resampled, sol


def train_tabular(n_episodes: int = 80000, dt: float = 1.0, seed: int = 0):
    OUT.mkdir(parents=True, exist_ok=True)
    p = TASK2
    env = LiquidationEnv(p, EnvConfig(dt=dt, n_depth=21, depth_max=0.05), seed=seed)
    h_ref, sol = reference_h_grid(p, env.num_time_buckets)

    print(f"Tabular Q: n_episodes={n_episodes}, dt={dt}, "
          f"time_buckets={env.num_time_buckets}, Q0={p.Q0}")
    agent = TabularQ(env, seed=seed)
    log = agent.train(n_episodes=n_episodes,
                      alpha_start=0.4, alpha_end=0.02,
                      eps_start=1.0, eps_end=0.02,
                      h_reference=h_ref, log_every=max(1, n_episodes // 20))
    with open(OUT / "tabular.pkl", "wb") as f:
        pickle.dump({"Q": agent.Q, "returns": log["returns"],
                     "rmse_trace": log["rmse_trace"],
                     "h_reference": h_ref, "env_cfg": env.cfg.__dict__,
                     "fd_solution": sol,
                     "depth_grid": env.depth_grid, "t_grid": env.t_grid,
                     "params": p}, f)
    return agent, log, sol


def train_ddqn(n_episodes: int = 6000, dt: float = 1.0, seed: int = 0):
    OUT.mkdir(parents=True, exist_ok=True)
    p = TASK2
    env = LiquidationEnv(p, EnvConfig(dt=dt, n_depth=21, depth_max=0.05), seed=seed)
    h_ref, sol = reference_h_grid(p, env.num_time_buckets)

    print(f"DDQN: n_episodes={n_episodes}, dt={dt}, "
          f"time_buckets={env.num_time_buckets}, Q0={p.Q0}")
    agent = DDQN(env, seed=seed)
    returns = agent.train(n_episodes=n_episodes,
                          eps_start=1.0, eps_end=0.05,
                          log_every=max(1, n_episodes // 12))
    depth, trig = agent.greedy_policy_grid()
    with open(OUT / "ddqn.pkl", "wb") as f:
        pickle.dump({"depth": depth, "trigger": trig,
                     "returns": returns, "h_reference": h_ref,
                     "env_cfg": env.cfg.__dict__,
                     "depth_grid": env.depth_grid, "t_grid": env.t_grid,
                     "fd_solution": sol,
                     "params": p}, f)
    return agent, returns, sol


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["tabular", "ddqn", "both"], default="both")
    ap.add_argument("--tabular-episodes", type=int, default=80000)
    ap.add_argument("--ddqn-episodes", type=int, default=6000)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.method in ("tabular", "both"):
        train_tabular(args.tabular_episodes, args.dt, args.seed)
    if args.method in ("ddqn", "both"):
        train_ddqn(args.ddqn_episodes, args.dt, args.seed)
