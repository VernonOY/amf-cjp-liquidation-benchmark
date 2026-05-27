"""Experiment 4 — RL failure-mode diagnostics (AMF revision §5.4, Fig 10).

Three panels:
  (a) MO under-use vs epsilon-bonus on MO action: confirms whether
      DDQN/Tabular MO under-use is exploration-bias driven (vs structural).
  (b) Action grid bias: value error vs depth grid resolution n_d.
  (c) State aggregation bias: requires CIR env (Phase 4); stubbed.

Parallelisation: each (agent, sweep_value, seed) cell is independent and is
sent to a joblib worker. To avoid PyTorch/MKL oversubscription inside the
worker process, we pin OMP/MKL threads to 1 inside the worker.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed

from ..common.params import REGIME_I, REGIME_II, TASK2
from ..common.simulator import Simulator
from ..baselines.twap import make_twap_mo_policy
from ..rl.env import EnvConfig, LiquidationEnv
from ..rl.rewards import h_shaped
from ..rl.tabular_q import TabularQ
from ..rl.double_dqn import DDQN
from ._runner import write_csv


def _set_single_thread():
    """Disable MKL/OMP multi-threading inside the joblib worker — joblib
    fans out across processes, so each worker should be single-threaded."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _mo_bonus_reward(epsilon: float):
    def _r(p, env, trans):
        base = h_shaped(p, env, trans)
        if trans["mo_flag"] and trans["q_pre"] > 0:
            base += epsilon
        return base
    return _r


def _mo_usage_rate(env, agent, n_paths: int, seed: int,
                   sim_dt: float = 0.05) -> dict:
    """Roll the env-trained agent under CANONICAL reward (no bonus)."""
    p = env.p
    if isinstance(agent, TabularQ):
        depth_idx = np.argmax(agent.Q, axis=2)
        n_dep = env.cfg.n_depth
        depth_grid = env.depth_grid
        dt_env = env.cfg.dt

        def policy(t, q, S):
            if q <= 0:
                return (0.0, False)
            i = min(int(round(t / dt_env)), env.num_time_buckets - 1)
            a = int(depth_idx[i, min(q, p.Q0)])
            if a == n_dep:
                return (0.0, True)
            return (float(depth_grid[a]), False)
    else:  # DDQN
        depth_grid, trigger = agent.greedy_policy_grid()
        dt_env = env.cfg.dt

        def policy(t, q, S):
            if q <= 0:
                return (0.0, False)
            i = min(int(round(t / dt_env)), env.num_time_buckets - 1)
            if trigger[i, min(q, p.Q0)]:
                return (0.0, True)
            return (float(max(depth_grid[i, min(q, p.Q0)], 0.0)), False)

    sim = Simulator(p, dt=sim_dt, seed=seed)
    out = sim.monte_carlo(policy, n_paths=n_paths, progress=False)
    sim_twap = Simulator(p, dt=sim_dt, seed=seed)
    out_twap = sim_twap.monte_carlo(make_twap_mo_policy(p),
                                     n_paths=n_paths, progress=False)
    return {
        "mo_per_path": float(np.mean(out["n_mo"])),
        "terminal_mean": float(np.mean(out["terminal"])),
        "premium_vs_twap": float(np.mean(out["terminal"] - out_twap["terminal"])),
        "clearance_prob": float(np.mean(out["q_final"] == 0)),
    }


# ---------------------------------------------------------------------------
# Cell workers
# ---------------------------------------------------------------------------
def _panel_a_cell(eps: float, seed: int, agent_id: str, n_episodes: int,
                   n_eval_paths: int) -> dict:
    _set_single_thread()
    p = TASK2
    reward_fn = _mo_bonus_reward(eps)
    env = LiquidationEnv(p, EnvConfig(dt=0.5, n_depth=21),
                          reward_fn=reward_fn, seed=seed)
    if agent_id == "A_tabular":
        agent = TabularQ(env, seed=seed + 10_000)
        agent.train(n_episodes=n_episodes)
    else:
        agent = DDQN(env, seed=seed + 10_000)
        agent.train(n_episodes=n_episodes, log_every=10**9)
    env_eval = LiquidationEnv(p, EnvConfig(dt=0.5, n_depth=21), seed=seed)
    metrics = _mo_usage_rate(env_eval, agent, n_paths=n_eval_paths,
                              seed=seed + 50_000)
    return {
        "panel": "a", "agent": agent_id,
        "sweep_value": float(eps), "seed": int(seed),
        "mo_per_path": metrics["mo_per_path"],
        "terminal_mean": metrics["terminal_mean"],
        "premium_vs_twap": metrics["premium_vs_twap"],
        "clearance_prob": metrics["clearance_prob"],
        "value_err_rmse": float("nan"),
    }


def _panel_b_cell(nd: int, seed: int, n_episodes: int) -> dict:
    _set_single_thread()
    from ..numerical.fd_constant_lambda import solve as fd_solve

    p = TASK2
    fd = fd_solve(p, dt=0.01)
    env = LiquidationEnv(p, EnvConfig(dt=0.5, n_depth=nd), seed=seed)
    agent = TabularQ(env, seed=seed + 10_000)
    agent.train(n_episodes=n_episodes)
    V = agent.value_grid()
    fd_t = fd.t_grid
    idx = np.clip(np.searchsorted(fd_t, env.t_grid), 0, len(fd_t) - 1)
    err = float(np.sqrt(np.mean(
        (V[:-1, 1:] - fd.h[idx][:-1, 1:]) ** 2
    )))
    return {
        "panel": "b", "agent": "A_tabular",
        "sweep_value": int(nd), "seed": int(seed),
        "mo_per_path": float("nan"),
        "terminal_mean": float("nan"),
        "premium_vs_twap": float("nan"),
        "clearance_prob": float("nan"),
        "value_err_rmse": err,
    }


def _panel_c_cell(scheme: str, seed: int, agent_id: str, n_episodes: int,
                   n_eval_paths: int, p=TASK2, env_dt: float = 0.5,
                   sim_dt: float = 0.05) -> dict:
    """Fairness ablation: give Tabular/DDQN the same 5% linearly decaying
    forced-MO exploration probability used by Hybrid PPO, or leave their
    default epsilon-greedy exploration unchanged.
    """
    _set_single_thread()
    env = LiquidationEnv(p, EnvConfig(dt=env_dt, n_depth=21), seed=seed)
    forced = scheme == "forced_mo_5pct"
    kwargs = {"mo_explore_start": 0.05, "mo_explore_end": 0.0} if forced else {}
    if agent_id == "A_tabular":
        agent = TabularQ(env, seed=seed + 10_000)
        agent.train(n_episodes=n_episodes, **kwargs)
    else:
        agent = DDQN(env, seed=seed + 10_000)
        agent.train(n_episodes=n_episodes, log_every=10**9, **kwargs)
    env_eval = LiquidationEnv(p, EnvConfig(dt=env_dt, n_depth=21), seed=seed)
    metrics = _mo_usage_rate(env_eval, agent, n_paths=n_eval_paths,
                              seed=seed + 70_000, sim_dt=sim_dt)
    return {
        "panel": "c", "agent": agent_id,
        "sweep_value": scheme, "seed": int(seed),
        "mo_per_path": metrics["mo_per_path"],
        "terminal_mean": metrics["terminal_mean"],
        "premium_vs_twap": metrics["premium_vs_twap"],
        "clearance_prob": metrics["clearance_prob"],
        "value_err_rmse": float("nan"),
    }


# ---------------------------------------------------------------------------
# Parallel runners
# ---------------------------------------------------------------------------
def _run_parallel(jobs, n_jobs: int, label: str) -> list[dict]:
    t0 = time.time()
    print(f"[{label}] launching {len(jobs)} cells on {n_jobs} workers", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(fn)(*args) for fn, args in jobs
    )
    dt = time.time() - t0
    print(f"[{label}] done in {dt:.1f}s ({dt / max(len(jobs), 1):.2f}s/cell)",
          flush=True)
    return list(results)


def panel_a_mo_underuse(epsilons, n_episodes, seeds, n_eval_paths: int = 200,
                          n_jobs: int = -1):
    jobs = [(_panel_a_cell, (eps, seed, ag, n_episodes, n_eval_paths))
            for eps in epsilons for seed in seeds
            for ag in ("A_tabular", "B_ddqn")]
    return _run_parallel(jobs, n_jobs=n_jobs, label="exp4 panel A")


def panel_b_action_grid_bias(n_depth_list, n_episodes, seeds, n_jobs: int = -1):
    jobs = [(_panel_b_cell, (nd, seed, n_episodes))
            for nd in n_depth_list for seed in seeds]
    return _run_parallel(jobs, n_jobs=n_jobs, label="exp4 panel B")


def panel_c_forced_mo_fairness(n_episodes, seeds, n_eval_paths: int = 500,
                               n_jobs: int = -1, p=TASK2,
                               env_dt: float = 0.5,
                               sim_dt: float = 0.05):
    jobs = [(_panel_c_cell,
             (scheme, seed, ag, n_episodes, n_eval_paths, p, env_dt, sim_dt))
            for scheme in ("default", "forced_mo_5pct")
            for seed in seeds
            for ag in ("A_tabular", "B_ddqn")]
    return _run_parallel(jobs, n_jobs=n_jobs, label="exp4 panel C")


def run(
    epsilons: Sequence[float] = (0.0, 1e-3, 1e-2),
    n_depth_list: Sequence[int] = (10, 21, 50),
    n_episodes: int = 500,
    seeds: Sequence[int] = (0, 1),
    n_eval_paths: int = 200,
    out_csv: str = "data/exp4/failure_modes.csv",
    n_jobs: int = -1,
) -> list[dict]:
    rows_a = panel_a_mo_underuse(epsilons, n_episodes, seeds,
                                   n_eval_paths=n_eval_paths, n_jobs=n_jobs)
    rows_b = panel_b_action_grid_bias(n_depth_list, n_episodes, seeds,
                                        n_jobs=n_jobs)
    rows_c = panel_c_forced_mo_fairness(n_episodes, seeds,
                                         n_eval_paths=n_eval_paths,
                                         n_jobs=n_jobs)
    rows = rows_a + rows_b + rows_c
    write_csv(rows, out_csv,
              columns=["panel", "agent", "sweep_value", "seed",
                       "mo_per_path", "terminal_mean", "premium_vs_twap",
                       "clearance_prob", "value_err_rmse"])
    print(f"[exp4] wrote {len(rows)} rows -> {out_csv}", flush=True)
    return rows


def _regime_setup(name: str):
    key = name.lower()
    if key in ("i", "1", "regime_i", "regime1"):
        return REGIME_I, 0.5, 0.05
    if key in ("ii", "2", "regime_ii", "regime2"):
        return REGIME_II, 1.0, 0.1
    raise ValueError(f"unknown regime {name!r}")


def run_fairness_only(
    *,
    regime: str = "i",
    budget: int = 100000,
    n_episodes: int | None = None,
    seeds: Sequence[int] = tuple(range(20)),
    n_eval_paths: int = 2000,
    out_csv: str = "data/exp4/forced_mo_fairness.csv",
    n_jobs: int = -1,
) -> list[dict]:
    """Run only the forced-MO fairness ablation from panel C.

    The default budget corresponds to the reviewer-facing n=1e5 point in
    Regime I.  For Regime II callers can pass budget=1e6 to match the scaled
    sample-complexity endpoint.
    """
    p, env_dt, sim_dt = _regime_setup(regime)
    if n_episodes is None:
        n_steps = int(round(p.T / env_dt)) + 1
        n_episodes = max(int(round(budget / n_steps)), 1)
    rows = panel_c_forced_mo_fairness(
        n_episodes=n_episodes,
        seeds=seeds,
        n_eval_paths=n_eval_paths,
        n_jobs=n_jobs,
        p=p,
        env_dt=env_dt,
        sim_dt=sim_dt,
    )
    write_csv(rows, out_csv,
              columns=["panel", "agent", "sweep_value", "seed",
                       "mo_per_path", "terminal_mean", "premium_vs_twap",
                       "clearance_prob", "value_err_rmse"])
    print(f"[exp4 fairness] wrote {len(rows)} rows -> {out_csv}", flush=True)
    return rows


if __name__ == "__main__":  # pragma: no cover
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--full", action="store_true",
                     help="Paper-grade: 5 eps levels, 4 n_d levels, 10 seeds")
    pa.add_argument("--fairness-only", action="store_true",
                     help="Run only panel C: default vs forced-MO fairness ablation")
    pa.add_argument("--regime", default="i",
                     choices=("i", "ii", "regime_i", "regime_ii"))
    pa.add_argument("--budget", type=int, default=100000,
                     help="Sample-transition budget used to derive n_episodes for fairness-only")
    pa.add_argument("--n-episodes", type=int, default=None,
                     help="Override training episodes for all cells")
    pa.add_argument("--seed-count", type=int, default=None)
    pa.add_argument("--n-eval-paths", type=int, default=None)
    pa.add_argument("--out", default="data/exp4/failure_modes.csv")
    pa.add_argument("--n-jobs", type=int, default=-1)
    args = pa.parse_args()
    if args.fairness_only:
        seed_count = args.seed_count or (20 if args.full else 3)
        rows = run_fairness_only(
            regime=args.regime,
            budget=args.budget,
            n_episodes=args.n_episodes,
            seeds=tuple(range(seed_count)),
            n_eval_paths=args.n_eval_paths or (2000 if args.full else 200),
            out_csv=args.out,
            n_jobs=args.n_jobs,
        )
    elif args.full:
        rows = run(
            epsilons=(0.0, 1e-3, 1e-2, 5e-2, 1e-1),
            n_depth_list=(5, 10, 21, 50, 100, 200),
            n_episodes=2000,
            seeds=tuple(range(10)),
            n_eval_paths=args.n_eval_paths or 500,
            out_csv=args.out,
            n_jobs=args.n_jobs,
        )
    else:
        rows = run(out_csv=args.out,
                   n_eval_paths=args.n_eval_paths or 200,
                   n_jobs=args.n_jobs)
    print(f"wrote {len(rows)} rows to {args.out}")
