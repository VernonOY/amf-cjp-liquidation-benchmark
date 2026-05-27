"""Simulate Task 2 (LO+MO) optimal strategy and run MC studies."""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path as _Path

import numpy as np

from ..common.params import TASK2, Params
from ..common.simulator import Simulator
from ..common.twap import make_twap_mo_policy
from . import fd_solver as fd


OUT = _Path(__file__).resolve().parents[2] / "results" / "task2"


def _build_sim_and_policy(p: Params, sim_dt: float, fd_dt: float, seed: int):
    sol = fd.solve(p, dt=fd_dt, Q_max=p.Q0)
    # Align simulator dt to FD grid so policy lookup is direct.
    sim = Simulator(p, dt=sim_dt, seed=seed)
    # Build a snapped policy: nearest FD grid index.
    depth_grid = sol.depth_grid()
    trigger = sol.mo_trigger
    fd_t = sol.t_grid

    def policy(t: float, q: int, S: float):
        if q <= 0:
            return (0.0, False)
        i = int(np.clip(round(t / fd_dt), 0, len(fd_t) - 1))
        if trigger[i, q]:
            return (0.0, True)
        return (float(max(depth_grid[i, q], 0.0)), False)
    return sim, policy, sol


def run_sample_paths(p: Params = TASK2, n: int = 3, seed: int = 123,
                     sim_dt: float = 0.05, fd_dt: float = 0.01,
                     tag: str = "default"):
    OUT.mkdir(parents=True, exist_ok=True)
    sim, policy, sol = _build_sim_and_policy(p, sim_dt, fd_dt, seed)
    paths = [sim.simulate(policy) for _ in range(n)]
    with open(OUT / f"sample_paths_{tag}.pkl", "wb") as f:
        pickle.dump({"paths": paths, "fd_solution": sol,
                     "params": p, "sim_dt": sim_dt}, f)
    return paths, sol


def run_mc(p: Params = TASK2, n_paths: int = 10000, seed: int = 0,
           sim_dt: float = 0.05, fd_dt: float = 0.01, tag: str = "default"):
    OUT.mkdir(parents=True, exist_ok=True)
    sim, policy, sol = _build_sim_and_policy(p, sim_dt, fd_dt, seed)
    print(f"Task 2 MC ({tag}): {n_paths} paths, sim_dt={sim_dt}, fd_dt={fd_dt}")
    mc_opt = sim.monte_carlo(policy, n_paths, progress=True)

    sim_tw = Simulator(p, dt=sim_dt, seed=seed)
    mc_tw = sim_tw.monte_carlo(make_twap_mo_policy(p), n_paths, progress=True)

    with open(OUT / f"mc_{tag}.pkl", "wb") as f:
        pickle.dump({"optimal": mc_opt, "twap": mc_tw, "fd_solution": sol,
                     "params": p}, f)
    print_mc_summary(mc_opt, mc_tw, p)
    return mc_opt, mc_tw, sol


def print_mc_summary(mc_opt, mc_tw, p: Params):
    def stats(a):
        return np.nanmean(a), np.nanstd(a)

    m_opt, s_opt = stats(mc_opt["avg_price"])
    m_tw, _ = stats(mc_tw["twap_price"])
    shortfall = (p.S0 - mc_opt["avg_price"])
    print("\n=== Task 2 Monte Carlo Summary ===")
    print(f"Optimal avg price : mean {m_opt:.6f}, std {s_opt:.6f}")
    print(f"TWAP midprice     : mean {m_tw:.6f}")
    print(f"Mean MOs used     : {mc_opt['n_mo'].mean():.3f}")
    print(f"Mean LO fills     : {mc_opt['n_lo'].mean():.3f}")
    print(f"Pr[q_T = 0]       : {(mc_opt['q_final'] == 0).mean():.3f}")
    print(f"Mean implementation shortfall vs S0 : {np.nanmean(shortfall):.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=10000)
    ap.add_argument("--sim-dt", type=float, default=0.05)
    ap.add_argument("--fd-dt", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()

    run_sample_paths(TASK2, n=3, seed=123,
                     sim_dt=args.sim_dt, fd_dt=args.fd_dt, tag="default")
    if not args.sample_only:
        run_mc(TASK2, n_paths=args.n_paths, seed=args.seed,
               sim_dt=args.sim_dt, fd_dt=args.fd_dt, tag="default")
