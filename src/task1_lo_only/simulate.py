"""Simulate Task 1 paths and run the Monte Carlo study."""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path as _Path

import numpy as np

from ..common.params import TASK1
from ..common.simulator import Simulator
from ..common.twap import make_twap_mo_policy
from .solver import make_policy


OUT = _Path(__file__).resolve().parents[2] / "results" / "task1"


def run_sample_paths(n: int = 3, seed: int = 42, dt: float = 0.05):
    OUT.mkdir(parents=True, exist_ok=True)
    p = TASK1
    sim = Simulator(p, dt=dt, seed=seed)
    policy, delta_grid = make_policy(p, sim.t_grid)
    paths = [sim.simulate(policy) for _ in range(n)]
    with open(OUT / "sample_paths.pkl", "wb") as f:
        pickle.dump({"paths": paths, "delta_grid": delta_grid,
                     "t_grid": sim.t_grid, "params": p}, f)
    return paths, delta_grid


def run_mc(n_paths: int = 10000, seed: int = 0, dt: float = 0.05):
    OUT.mkdir(parents=True, exist_ok=True)
    p = TASK1
    sim = Simulator(p, dt=dt, seed=seed)
    policy, _ = make_policy(p, sim.t_grid)
    print(f"Task 1 MC: {n_paths} paths, dt={dt}s")
    mc_opt = sim.monte_carlo(policy, n_paths, progress=True)

    # TWAP baseline (MO-only): execute one MO per time bucket.
    sim_tw = Simulator(p, dt=dt, seed=seed)  # same seed -> same mid paths
    tw_policy = make_twap_mo_policy(p)
    print("Task 1 MC: TWAP baseline")
    mc_tw = sim_tw.monte_carlo(tw_policy, n_paths, progress=True)

    with open(OUT / "mc.pkl", "wb") as f:
        pickle.dump({"optimal": mc_opt, "twap": mc_tw, "params": p}, f)
    print_mc_summary(mc_opt, mc_tw)
    return mc_opt, mc_tw


def print_mc_summary(mc_opt, mc_tw):
    def pct(a):
        return np.nanmean(a), np.nanstd(a)

    m_opt, s_opt = pct(mc_opt["avg_price"])
    m_tw, s_tw = pct(mc_tw["twap_price"])
    print("\n=== Task 1 Monte Carlo Summary ===")
    print(f"Optimal LO avg price : mean {m_opt:.6f}, std {s_opt:.6f}")
    print(f"TWAP midprice       : mean {m_tw:.6f}, std {s_tw:.6f}")
    print(f"Avg terminal inventory (optimal): {mc_opt['q_final'].mean():.3f}")
    print(f"Pr[q_T = 0] (optimal) : {(mc_opt['q_final'] == 0).mean():.3f}")
    print(f"Mean filled LOs (optimal): {mc_opt['n_lo'].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=10000)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()

    run_sample_paths(n=3, seed=42, dt=args.dt)
    if not args.sample_only:
        run_mc(n_paths=args.n_paths, seed=args.seed, dt=args.dt)
