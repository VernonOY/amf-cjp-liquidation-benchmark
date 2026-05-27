"""Experiment 1 — Sample-complexity benchmark (AMF revision §5.3).

Produces Fig 9 (4-panel sample complexity) and Tab 5 (agent comparison).

Sweep: 4 agents x sample_count_grid x seeds.
  Agent A : Tabular Q-learning
  Agent B : Double DQN
  Agent C : Structure-Aware Hybrid Policy
  Agent D : Plug-in MLE Estimator

Per-cell metrics:
  value_err_rmse : ||V_RL - h_FD||_2 on (t, q >= 1, i < n_t) cells
  policy_err_l1  : mean |delta_RL - delta*| on the same cells
  premium_vs_twap: mean of (return_RL - return_TWAP) over n_eval MC paths
  premium_ci_*   : paired-bootstrap 95% CI (percentile)
  clearance_prob : P[Q_T = 0]
  wall_clock_sec : training wall-clock
  kappa_hat      : Agent D's MLE only

For the AMF MVS-ready demo we use a reduced default grid (4 sample sizes x
5 seeds = 20 trainings per agent). The full grid (7 sizes x 20 seeds) is
parameterisable via `n_levels` and `n_seeds`.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..common.params import REGIME_I, REGIME_II, TASK2
from ..common.simulator import Simulator
from ..common.stats import paired_bootstrap_ci
from ..baselines.twap import make_twap_mo_policy
from ..numerical.fd_constant_lambda import solve as fd_solve
from ..rl.env import EnvConfig, LiquidationEnv
from ..rl.hybrid_policy import HybridAgent
from ..rl.plugin_estimator import PluginAgent
from ..rl.tabular_q import TabularQ
from ..rl.double_dqn import DDQN
from ._runner import dispatch, write_csv


AGENT_IDS = ("A_tabular", "B_ddqn", "C_hybrid", "D_plugin")


def _fd_reference(p, dt: float = 0.01):
    return fd_solve(p, dt=dt)


def _eval_policy_premium(p, policy, twap_returns: np.ndarray, n_paths: int,
                          seed: int, sim_dt: float = 0.05) -> dict:
    sim = Simulator(p, dt=sim_dt, seed=seed)
    out = sim.monte_carlo(policy, n_paths=n_paths, progress=False)
    rl_returns = out["terminal"]
    diff = rl_returns - twap_returns
    _, lo, hi = paired_bootstrap_ci(rl_returns, twap_returns,
                                    n_boot=2000, seed=seed)
    return {
        "premium_vs_twap": float(np.mean(diff)),
        "premium_ci_low": float(lo),
        "premium_ci_high": float(hi),
        "clearance_prob": float(np.mean(out["q_final"] == 0)),
    }


def _twap_returns(p, n_paths: int, seed: int,
                  sim_dt: float = 0.05) -> np.ndarray:
    sim = Simulator(p, dt=sim_dt, seed=seed)
    twap = make_twap_mo_policy(p)
    out = sim.monte_carlo(twap, n_paths=n_paths, progress=False)
    return out["terminal"]


def _make_env_policy_from_agent(env: LiquidationEnv, agent, fd=None) -> callable:
    """Wrap an env-side agent into a simulator-side policy.

    For agents that emit (depth_grid, mo_trigger) we snap to the env's grid
    and dispatch by (i_env, q). For plug-in (FD-derived) we reuse the FD's
    make_policy() because it is already simulator-compatible.
    """
    if isinstance(agent, PluginAgent):
        if agent.fd_solution is None:
            raise RuntimeError("PluginAgent: train() before wrapping")
        return agent.fd_solution.make_policy()
    if isinstance(agent, TabularQ):
        depth_idx = np.argmax(agent.Q, axis=2)
        depth_grid = env.depth_grid
        n_dep = env.cfg.n_depth
        dt_env = env.cfg.dt
        Q = env.p.Q0

        def policy(t: float, q: int, S: float):
            if q <= 0:
                return (0.0, False)
            i = min(int(round(t / dt_env)), env.num_time_buckets - 1)
            q_clip = min(q, Q)
            a = int(depth_idx[i, q_clip])
            if a == n_dep:
                return (0.0, True)
            return (float(depth_grid[a]), False)
        return policy
    # DDQN / HybridAgent expose policy_grid() that returns (depth, trigger)
    depth_grid_full, trigger = agent.policy_grid(env) if hasattr(agent, "policy_grid") else agent.greedy_policy_grid()
    dt_env = env.cfg.dt
    Q = env.p.Q0

    def policy(t: float, q: int, S: float):
        if q <= 0:
            return (0.0, False)
        i = min(int(round(t / dt_env)), env.num_time_buckets - 1)
        q_clip = min(q, Q)
        if trigger[i, q_clip]:
            return (0.0, True)
        return (float(max(depth_grid_full[i, q_clip], 0.0)), False)
    return policy


def _value_error(env, V_rl: np.ndarray, fd_sol) -> float:
    fd_t = fd_sol.t_grid
    idx = np.clip(np.searchsorted(fd_t, env.t_grid), 0, len(fd_t) - 1)
    h_on_env = fd_sol.h[idx]
    diff = V_rl - h_on_env
    diff = diff[:-1, 1:]
    return float(np.sqrt(np.mean(diff ** 2)))


def _policy_error_l1(env, agent, fd_sol) -> float:
    if isinstance(agent, PluginAgent) and agent.fd_solution is None:
        return float("nan")
    if isinstance(agent, TabularQ):
        depth_idx = np.argmax(agent.Q, axis=2)
        n_dep = env.cfg.n_depth
        depth_grid = env.depth_grid
        rl_depth = np.zeros_like(fd_sol.h)
        fd_depth = fd_sol.depth_grid()
        n_match = 0
        total = 0.0
        for i_env in range(env.num_time_buckets - 1):
            t = env.t_grid[i_env]
            i_fd = min(int(round(t / (fd_sol.t_grid[1] - fd_sol.t_grid[0]))),
                        len(fd_sol.t_grid) - 1)
            for q in range(1, env.p.Q0 + 1):
                a = int(depth_idx[i_env, q])
                d_rl = 0.0 if a == n_dep else float(depth_grid[a])
                d_fd = max(fd_depth[i_fd, q], 0.0)
                total += abs(d_rl - d_fd)
                n_match += 1
        return float(total / max(n_match, 1))
    depth_rl, _ = agent.policy_grid(env) if hasattr(agent, "policy_grid") else agent.greedy_policy_grid()
    fd_depth = fd_sol.depth_grid()
    total = 0.0
    n_match = 0
    for i_env in range(env.num_time_buckets - 1):
        t = env.t_grid[i_env]
        i_fd = min(int(round(t / (fd_sol.t_grid[1] - fd_sol.t_grid[0]))),
                    len(fd_sol.t_grid) - 1)
        for q in range(1, env.p.Q0 + 1):
            total += abs(float(depth_rl[i_env, q]) - max(fd_depth[i_fd, q], 0.0))
            n_match += 1
    return float(total / max(n_match, 1))


def _one_run(agent_id: str, n: int, seed: int,
              p, fd, n_eval_paths: int, eval_seed_offset: int,
              env_dt: float, sim_dt: float, fd_dt: float,
              plugin_fill_cap: int | None = None) -> dict:
    """Train one agent at sample budget n with seed, then evaluate."""
    env_seed = seed
    train_seed = seed + 10_000
    eval_seed = seed + eval_seed_offset
    env = LiquidationEnv(p, EnvConfig(dt=env_dt, n_depth=21), seed=env_seed)

    t0 = time.time()
    # Convert sample budget n (env transitions) to episode count.
    # Use ceil so n=100 -> 1 episode and n=316 -> 3 episodes (no collapse at floor).
    n_eps = max(int(round(n / env.num_time_buckets)), 1)
    if agent_id == "A_tabular":
        agent = TabularQ(env, seed=train_seed)
        agent.train(n_episodes=n_eps)
        V = agent.value_grid()
        kappa_hat = float("nan")
    elif agent_id == "B_ddqn":
        agent = DDQN(env, seed=train_seed)
        agent.train(n_episodes=n_eps, log_every=10**9)
        V = agent.value_grid()
        kappa_hat = float("nan")
    elif agent_id == "C_hybrid":
        # Plan A: scale batch so we get O(n_eps^{1/2}) gradient updates —
        # this gives a real sample-complexity curve instead of saturating at
        # ~2 updates across most n levels (the failure mode we observed in
        # the initial Plan A spot check). Cap at 128 to keep gradient
        # variance bounded.
        import math
        bsz = max(min(int(round(math.sqrt(n_eps))), 128), 1)
        agent = HybridAgent(env, kappa_hat=100.0, seed=train_seed,
                             batch_episodes=bsz,
                             recompute_kappa_every=None)
        agent.train(n_episodes=n_eps)
        V = agent.value_grid()
        kappa_hat = float(agent.kappa_hat)
    elif agent_id == "D_plugin":
        # n = environment transitions; convert to expected fills
        # E[fills per step] ~ lam * exp(-kappa * E[delta]) * dt
        #                  ~ 0.83 * exp(-100 * 0.025) * 0.5 ~ 0.034
        expected_fill_per_step = max(p.lam * np.exp(-p.kappa * 0.025) * env.cfg.dt, 1e-6)
        n_fills_target = max(int(n * expected_fill_per_step), 30)
        if plugin_fill_cap is not None:
            n_fills_target = min(n_fills_target, int(plugin_fill_cap))
        agent = PluginAgent(env, prior_kappa=p.kappa, prior_lam=p.lam,
                             fd_dt=fd_dt, seed=train_seed)
        agent.train(n_fills_target=n_fills_target)
        V = agent.value_grid()
        kappa_hat = float(agent.estimates.kappa_hat) if agent.estimates else float("nan")
    else:
        raise ValueError(f"unknown agent_id {agent_id!r}")
    wall = time.time() - t0

    # Diagnostics
    value_err = _value_error(env, V, fd)
    pol_err = _policy_error_l1(env, agent, fd)

    # Premium evaluation
    twap_ret = _twap_returns(p, n_paths=n_eval_paths, seed=eval_seed,
                              sim_dt=sim_dt)
    sim_policy = _make_env_policy_from_agent(env, agent, fd=fd)
    metrics = _eval_policy_premium(p, sim_policy, twap_ret,
                                    n_paths=n_eval_paths, seed=eval_seed,
                                    sim_dt=sim_dt)

    return {
        "agent": agent_id,
        "n": int(n),
        "seed": int(seed),
        "value_err_rmse": float(value_err),
        "policy_err_l1": float(pol_err),
        "premium_vs_twap": metrics["premium_vs_twap"],
        "premium_ci_low": metrics["premium_ci_low"],
        "premium_ci_high": metrics["premium_ci_high"],
        "clearance_prob": metrics["clearance_prob"],
        "wall_clock_sec": float(wall),
        "kappa_hat": kappa_hat,
    }


def run(
    n_levels: Sequence[int] = (200, 1000, 4000),
    n_seeds: int = 3,
    n_eval_paths: int = 500,
    agents: Sequence[str] = AGENT_IDS,
    out_csv: str = "data/exp1/sample_complexity.csv",
    backend: str = "serial",
    n_jobs: int = -1,
    p=TASK2,
    fd_dt: float = 0.01,
    env_dt: float = 0.5,
    sim_dt: float = 0.05,
    plugin_fill_cap: int | None = None,
) -> list[dict]:
    """Run the full sample-complexity sweep. Returns rows + writes CSV."""
    fd = _fd_reference(p, dt=fd_dt)
    jobs = []
    for agent_id in agents:
        for n in n_levels:
            for seed in range(n_seeds):
                jobs.append((_one_run,
                             (agent_id, n, seed, p, fd, n_eval_paths, 20_000,
                              env_dt, sim_dt, fd_dt, plugin_fill_cap),
                             {}))
    rows = dispatch(jobs, backend=backend, n_jobs=n_jobs)
    write_csv(
        rows, out_csv,
        columns=["agent", "n", "seed", "value_err_rmse", "policy_err_l1",
                 "premium_vs_twap", "premium_ci_low", "premium_ci_high",
                 "clearance_prob", "wall_clock_sec", "kappa_hat"],
    )
    return rows


def make_figure(csv_path: str, fig_path: str) -> None:
    """Render Fig 9 (4-panel) from the CSV."""
    import csv
    import matplotlib.pyplot as plt
    from ..common.style import apply as apply_style

    apply_style()
    by_agent: dict[str, dict[int, list[dict]]] = {}
    with open(csv_path, "r") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            ag = r["agent"]
            n = int(r["n"])
            by_agent.setdefault(ag, {}).setdefault(n, []).append(r)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))
    ax_value, ax_policy, ax_prem, ax_clear = axes.ravel()
    panel_keys = [
        (ax_value, "value_err_rmse", r"$\|V_{\rm RL} - h\|_{\rm RMSE}$"),
        (ax_policy, "policy_err_l1", r"mean $|\delta_{\rm RL} - \delta^*|$"),
        (ax_prem, "premium_vs_twap", "Premium vs TWAP"),
        (ax_clear, "clearance_prob", r"$\mathbb{P}[Q_T = 0]$"),
    ]
    markers = {"A_tabular": "o", "B_ddqn": "s", "C_hybrid": "^", "D_plugin": "D"}
    legend_labels = {
        "A_tabular": "A Tabular",
        "B_ddqn": "B DDQN",
        "C_hybrid": "C Hybrid",
        "D_plugin": "D Plug-in",
    }
    tick_fs = 13
    label_fs = 15
    for ax, key, label in panel_keys:
        plotted_means = []
        for ag in sorted(by_agent.keys()):
            ns = sorted(by_agent[ag].keys())
            means = []
            for n in ns:
                vals = [float(r[key]) for r in by_agent[ag][n]
                        if r[key] not in ("nan", "")]
                if not vals:
                    means.append(float("nan"))
                else:
                    means.append(float(np.mean(vals)))
            plotted_means.extend([m for m in means if np.isfinite(m)])
            ax.plot(
                ns,
                means,
                marker=markers.get(ag, "o"),
                label=legend_labels.get(ag, ag),
                linewidth=2.0,
                markersize=6.5,
            )
        ax.set_xscale("log")
        if key in ("value_err_rmse", "policy_err_l1"):
            ax.set_yscale("log")
        ax.tick_params(labelsize=tick_fs)
        ax.set_xlabel(r"sample budget (env transitions)", fontsize=label_fs)
        ax.set_ylabel(label, fontsize=label_fs)
        ax.grid(True, alpha=0.3)
        if key == "premium_vs_twap":
            ymax = max(plotted_means) if plotted_means else 0.205
            ax.set_ylim(bottom=0.0, top=max(0.205, 1.1 * ymax))
    handles, labels = ax_value.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=13,
        frameon=True,
        handlelength=1.8,
        columnspacing=1.4,
    )
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


FULL_N_LEVELS = (100, 316, 1000, 3162, 10000, 31623, 100000)
REGIME_II_N_LEVELS = (1000, 3162, 10000, 31623, 100000, 316227, 1000000)
FULL_N_SEEDS = 20
FULL_N_EVAL_PATHS = 2000


def _regime_setup(name: str):
    key = name.lower()
    if key in ("i", "1", "regime_i", "regime1"):
        return REGIME_I, FULL_N_LEVELS, 0.01, 0.5, 0.05, None
    if key in ("ii", "2", "regime_ii", "regime2"):
        return REGIME_II, REGIME_II_N_LEVELS, 0.05, 1.0, 0.1, 1000
    raise ValueError(f"unknown regime {name!r}")


if __name__ == "__main__":  # pragma: no cover
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--smoke", action="store_true",
                     help="2-min sanity run (plug-in only, n=200)")
    pa.add_argument("--full", action="store_true",
                     help="Paper-grade run: 7 n levels x 20 seeds x 4 agents")
    pa.add_argument("--regime", default="i",
                     choices=("i", "ii", "regime_i", "regime_ii"),
                     help="Parameter regime: i = CJP small scale, ii = Path 1 institutional scale")
    pa.add_argument("--agents", default=None,
                     help="Comma-separated agent ids, e.g. D_plugin,A_tabular")
    pa.add_argument("--n-levels", default=None,
                     help="Comma-separated sample budgets overriding the regime defaults")
    pa.add_argument("--seeds", type=int, default=3)
    pa.add_argument("--n-eval-paths", type=int, default=None)
    pa.add_argument("--backend", default="serial",
                     choices=("serial", "joblib"))
    pa.add_argument("--n-jobs", type=int, default=-1,
                     help="Joblib worker count (-1 = all cores)")
    pa.add_argument("--out", default="data/exp1/sample_complexity.csv")
    pa.add_argument("--fig", default="figures/fig9_sample_complexity.pdf")
    args = pa.parse_args()
    p, full_levels, fd_dt, env_dt, sim_dt, plugin_fill_cap = _regime_setup(args.regime)
    agents = tuple(args.agents.split(",")) if args.agents else AGENT_IDS
    if args.n_levels:
        n_levels_override = tuple(int(float(x)) for x in args.n_levels.split(","))
    else:
        n_levels_override = None
    if args.smoke:
        smoke_levels = n_levels_override or ((10000,) if p.Q0 > 10 else (200,))
        smoke_agents = agents if args.agents else (("D_plugin", "A_tabular") if p.Q0 > 10 else ("D_plugin",))
        rows = run(n_levels=smoke_levels, n_seeds=args.seeds,
                    n_eval_paths=args.n_eval_paths or (100 if p.Q0 > 10 else 200),
                    agents=smoke_agents, out_csv=args.out,
                    backend=args.backend, n_jobs=args.n_jobs,
                    p=p, fd_dt=fd_dt, env_dt=env_dt, sim_dt=sim_dt,
                    plugin_fill_cap=plugin_fill_cap)
    elif args.full:
        levels = n_levels_override or full_levels
        print(f"[exp1] FULL run ({args.regime}): {len(levels)} n levels x "
               f"{FULL_N_SEEDS} seeds x 4 agents = "
               f"{len(levels) * FULL_N_SEEDS * len(agents)} cells; "
               f"backend={args.backend} n_jobs={args.n_jobs}")
        rows = run(n_levels=levels, n_seeds=FULL_N_SEEDS,
                    n_eval_paths=args.n_eval_paths or FULL_N_EVAL_PATHS,
                    agents=agents, out_csv=args.out,
                    backend=args.backend, n_jobs=args.n_jobs,
                    p=p, fd_dt=fd_dt, env_dt=env_dt, sim_dt=sim_dt,
                    plugin_fill_cap=plugin_fill_cap)
    else:
        rows = run(n_levels=n_levels_override or (200, 1000, 4000),
                    n_seeds=args.seeds,
                    n_eval_paths=args.n_eval_paths or 500,
                    agents=agents, out_csv=args.out,
                    backend=args.backend, n_jobs=args.n_jobs,
                    p=p, fd_dt=fd_dt, env_dt=env_dt, sim_dt=sim_dt,
                    plugin_fill_cap=plugin_fill_cap)
    make_figure(args.out, args.fig)
    print(f"wrote {len(rows)} rows to {args.out}, figure to {args.fig}")
