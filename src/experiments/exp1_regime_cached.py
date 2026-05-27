"""Checkpointed sample-complexity runner.

This is a fault-tolerant wrapper around ``exp1_sample_complexity._one_run``.
Each (regime, agent, budget, seed) cell is cached as an individual JSON file
before the aggregate CSV is written, so long Regime-II sweeps can be resumed
without losing completed cells.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from joblib import Parallel, delayed

from ._runner import run_with_cache, write_csv
from .exp1_sample_complexity import (
    AGENT_IDS,
    REGIME_II_N_LEVELS,
    FULL_N_LEVELS,
    _fd_reference,
    _one_run,
    _regime_setup,
    make_figure,
)


CSV_COLUMNS = [
    "agent", "n", "seed", "value_err_rmse", "policy_err_l1",
    "premium_vs_twap", "premium_ci_low", "premium_ci_high",
    "clearance_prob", "wall_clock_sec", "kappa_hat",
]


def _cell(
    cache_dir: str,
    regime: str,
    agent_id: str,
    n: int,
    seed: int,
    n_eval_paths: int,
):
    p, _levels, fd_dt, env_dt, sim_dt, plugin_fill_cap = _regime_setup(regime)
    fd = _fd_reference(p, dt=fd_dt)
    key = (
        f"regime={regime}__agent={agent_id}__n={int(n)}__seed={int(seed)}"
        f"__eval={int(n_eval_paths)}__fd={fd_dt:g}__env={env_dt:g}__sim={sim_dt:g}"
    )
    return run_with_cache(
        cache_dir,
        key,
        lambda: _one_run(
            agent_id, int(n), int(seed), p, fd, int(n_eval_paths), 20_000,
            env_dt, sim_dt, fd_dt, plugin_fill_cap,
        ),
    )


def run_cached(
    *,
    regime: str = "ii",
    agents: Sequence[str] = AGENT_IDS,
    budgets: Sequence[int] | None = None,
    seeds: Sequence[int] = tuple(range(20)),
    n_eval_paths: int = 2000,
    cache_dir: str = "data/cache/exp1_regime_ii",
    out_csv: str = "data/exp1/regime_ii_sample_complexity_full.csv",
    fig_path: str | None = "figures/fig_regime_ii_sample_complexity.pdf",
    n_jobs: int = -1,
) -> list[dict]:
    if budgets is None:
        budgets = REGIME_II_N_LEVELS if regime.lower() in ("ii", "2", "regime_ii") else FULL_N_LEVELS
    jobs = [
        (agent_id, int(n), int(seed))
        for agent_id in agents
        for n in budgets
        for seed in seeds
    ]
    print(
        f"[exp1 cached] {len(jobs)} cells; regime={regime}; "
        f"agents={','.join(agents)}; budgets={list(budgets)}; "
        f"seeds={len(list(seeds))}; n_jobs={n_jobs}",
        flush=True,
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_cell)(cache_dir, regime, agent_id, n, seed, n_eval_paths)
        for agent_id, n, seed in jobs
    )
    rows = sorted(rows, key=lambda r: (r["agent"], int(r["n"]), int(r["seed"])))
    write_csv(rows, out_csv, columns=CSV_COLUMNS)
    if fig_path:
        make_figure(out_csv, fig_path)
    print(f"[exp1 cached] wrote {len(rows)} rows -> {out_csv}", flush=True)
    return rows


if __name__ == "__main__":  # pragma: no cover
    import argparse

    pa = argparse.ArgumentParser()
    pa.add_argument("--regime", default="ii", choices=("i", "ii", "regime_i", "regime_ii"))
    pa.add_argument("--agents", default=",".join(AGENT_IDS))
    pa.add_argument("--budgets", default=None,
                    help="Comma-separated budgets; default is regime full grid")
    pa.add_argument("--seed-count", type=int, default=20)
    pa.add_argument("--n-eval-paths", type=int, default=2000)
    pa.add_argument("--cache-dir", default="data/cache/exp1_regime_ii")
    pa.add_argument("--out", default="data/exp1/regime_ii_sample_complexity_full.csv")
    pa.add_argument("--fig", default="figures/fig_regime_ii_sample_complexity.pdf")
    pa.add_argument("--n-jobs", type=int, default=-1)
    args = pa.parse_args()

    budgets = None
    if args.budgets:
        budgets = tuple(int(float(x)) for x in args.budgets.split(","))
    agents = tuple(a.strip() for a in args.agents.split(",") if a.strip())
    run_cached(
        regime=args.regime,
        agents=agents,
        budgets=budgets,
        seeds=tuple(range(args.seed_count)),
        n_eval_paths=args.n_eval_paths,
        cache_dir=args.cache_dir,
        out_csv=args.out,
        fig_path=args.fig,
        n_jobs=args.n_jobs,
    )
