"""Experiment 3 — Parameter robustness (AMF revision §7, Figs 15-16, Tab 6).

Three sub-experiments parallelised via joblib (per-cell fan-out):

  §7.1 Univariate sensitivity:
        sweep each parameter p in {lam, kappa, xi, alpha, phi} over
        multipliers; record FD-optimal premium and MO usage.

  §7.2 Bivariate sensitivity:
        2D grids over (kappa, phi) and (xi, alpha); produce two heatmaps.

  §7.3 Train-test misspecification:
        FD-optimal policy at (lam_train, kappa_train), evaluate at
        (lam_test, kappa_test). Reports premium gap.

Each cell does ONE FD solve + 2 MC sweeps (RL + TWAP). Cells are
embarrassingly parallel; joblib spreads them across cores. Caching is at
the call-of-FD level (we only solve FD once per param tuple even if the
same params show up multiple times — e.g. the diagonal of the misspec grid
equals the univariate result).
"""
from __future__ import annotations
import dataclasses
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed

from ..common.params import TASK2, Params
from ..common.simulator import Simulator
from ..baselines.twap import make_twap_mo_policy
from ..numerical.fd_constant_lambda import solve as fd_solve
from ._runner import write_csv


def _perturbed_params(base: Params, **overrides) -> Params:
    return dataclasses.replace(base, **overrides)


def _mean_premium(p: Params, policy, n_paths: int, seed: int) -> dict:
    sim_rl = Simulator(p, dt=0.05, seed=seed)
    out_rl = sim_rl.monte_carlo(policy, n_paths=n_paths, progress=False)
    sim_tw = Simulator(p, dt=0.05, seed=seed)
    out_tw = sim_tw.monte_carlo(make_twap_mo_policy(p), n_paths=n_paths, progress=False)
    return {
        "premium": float(np.mean(out_rl["terminal"] - out_tw["terminal"])),
        "mo_per_path": float(np.mean(out_rl["n_mo"])),
        "clearance_prob": float(np.mean(out_rl["q_final"] == 0)),
    }


# ---------------------------------------------------------------------------
# Cell workers (one cell = one row of the output CSV)
# ---------------------------------------------------------------------------
def _univariate_cell(name: str, mult: float, n_paths: int, seed: int) -> dict:
    base = TASK2
    base_val = getattr(base, name)
    p = _perturbed_params(base, **{name: base_val * mult})
    try:
        fd = fd_solve(p, dt=0.01)
    except Exception:
        return {"regime": "univariate", "param": name, "multiplier": float(mult),
                "premium": float("nan"), "mo_per_path": float("nan"),
                "clearance_prob": float("nan"), "degenerate": True}
    metrics = _mean_premium(p, fd.make_policy(), n_paths=n_paths, seed=seed)
    return {"regime": "univariate", "param": name, "multiplier": float(mult),
            **metrics, "degenerate": False}


def _bivariate_cell(pair: tuple[str, str], ma: float, mb: float,
                     n_paths: int, seed: int) -> dict:
    base = TASK2
    a, b = pair
    p = _perturbed_params(base, **{a: getattr(base, a) * ma,
                                    b: getattr(base, b) * mb})
    try:
        fd = fd_solve(p, dt=0.01)
    except Exception:
        return {"regime": "bivariate", "pair": f"{a}+{b}",
                "ma": float(ma), "mb": float(mb),
                "premium": float("nan"), "mo_per_path": float("nan"),
                "clearance_prob": float("nan"), "degenerate": True}
    metrics = _mean_premium(p, fd.make_policy(), n_paths=n_paths, seed=seed)
    return {"regime": "bivariate", "pair": f"{a}+{b}",
            "ma": float(ma), "mb": float(mb), **metrics, "degenerate": False}


def _misspec_cell(m_lam_train: float, m_kap_train: float,
                   m_lam_test: float, m_kap_test: float,
                   n_paths: int, seed: int) -> dict:
    base = TASK2
    p_train = _perturbed_params(base,
                                 lam=base.lam * m_lam_train,
                                 kappa=base.kappa * m_kap_train)
    p_test = _perturbed_params(base,
                                lam=base.lam * m_lam_test,
                                kappa=base.kappa * m_kap_test)
    try:
        fd_train = fd_solve(p_train, dt=0.01)
    except Exception:
        return {"regime": "misspec",
                "lam_train_mult": float(m_lam_train),
                "kap_train_mult": float(m_kap_train),
                "lam_test_mult": float(m_lam_test),
                "kap_test_mult": float(m_kap_test),
                "premium": float("nan"), "mo_per_path": float("nan"),
                "clearance_prob": float("nan")}
    policy_train = fd_train.make_policy()
    metrics = _mean_premium(p_test, policy_train, n_paths=n_paths, seed=seed)
    return {"regime": "misspec",
            "lam_train_mult": float(m_lam_train),
            "kap_train_mult": float(m_kap_train),
            "lam_test_mult": float(m_lam_test),
            "kap_test_mult": float(m_kap_test),
            **metrics}


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


def univariate_sensitivity(
    multipliers: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0),
    n_paths: int = 500,
    seed: int = 0,
    n_jobs: int = -1,
) -> list[dict]:
    param_names = [n for n in ("lam", "kappa", "xi", "alpha", "phi")
                    if getattr(TASK2, n) != 0.0]
    jobs = [(_univariate_cell, (name, mult, n_paths, seed))
            for name in param_names for mult in multipliers]
    return _run_parallel(jobs, n_jobs=n_jobs, label="exp3 univariate")


def bivariate_sensitivity(
    pairs: Sequence[tuple[str, str]] = (("kappa", "phi"), ("xi", "alpha")),
    multipliers: Sequence[float] = (0.5, 1.0, 2.0),
    n_paths: int = 300,
    seed: int = 0,
    n_jobs: int = -1,
) -> list[dict]:
    jobs = [(_bivariate_cell, (pair, ma, mb, n_paths, seed))
            for pair in pairs for ma in multipliers for mb in multipliers]
    return _run_parallel(jobs, n_jobs=n_jobs, label="exp3 bivariate")


def misspecification_grid(
    multipliers: Sequence[float] = (0.5, 1.0, 2.0),
    n_paths: int = 300,
    seed: int = 0,
    n_jobs: int = -1,
) -> list[dict]:
    jobs = [(_misspec_cell, (mlt, mkt, mle, mke, n_paths, seed))
            for mlt in multipliers for mkt in multipliers
            for mle in multipliers for mke in multipliers]
    return _run_parallel(jobs, n_jobs=n_jobs, label="exp3 misspec")


def run(
    out_dir: str = "data/exp3",
    multipliers: Sequence[float] = (0.5, 1.0, 2.0),
    n_paths: int = 300,
    n_jobs: int = -1,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows_u = univariate_sensitivity(multipliers=multipliers,
                                     n_paths=n_paths, n_jobs=n_jobs)
    write_csv(rows_u, out / "univariate.csv",
              columns=["regime", "param", "multiplier", "premium",
                       "mo_per_path", "clearance_prob", "degenerate"])
    print(f"[exp3] wrote {len(rows_u)} rows -> {out / 'univariate.csv'}", flush=True)

    rows_b = bivariate_sensitivity(multipliers=multipliers,
                                    n_paths=n_paths, n_jobs=n_jobs)
    write_csv(rows_b, out / "bivariate.csv",
              columns=["regime", "pair", "ma", "mb", "premium",
                       "mo_per_path", "clearance_prob", "degenerate"])
    print(f"[exp3] wrote {len(rows_b)} rows -> {out / 'bivariate.csv'}", flush=True)

    rows_m = misspecification_grid(multipliers=multipliers,
                                    n_paths=n_paths, n_jobs=n_jobs)
    write_csv(rows_m, out / "misspec.csv",
              columns=["regime", "lam_train_mult", "kap_train_mult",
                       "lam_test_mult", "kap_test_mult",
                       "premium", "mo_per_path", "clearance_prob"])
    print(f"[exp3] wrote {len(rows_m)} rows -> {out / 'misspec.csv'}", flush=True)

    return {"univariate": len(rows_u), "bivariate": len(rows_b),
            "misspec": len(rows_m)}


if __name__ == "__main__":  # pragma: no cover
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--full", action="store_true",
                     help="Paper-grade: 5 multipliers, n_paths=2000")
    pa.add_argument("--out-dir", default="data/exp3")
    pa.add_argument("--n-jobs", type=int, default=-1)
    args = pa.parse_args()
    if args.full:
        multipliers = (0.5, 0.75, 1.0, 1.5, 2.0)
        n_paths = 2000
    else:
        multipliers = (0.5, 1.0, 2.0)
        n_paths = 300
    counts = run(out_dir=args.out_dir, multipliers=multipliers,
                  n_paths=n_paths, n_jobs=args.n_jobs)
    print(f"exp3 wrote: {counts}", flush=True)
