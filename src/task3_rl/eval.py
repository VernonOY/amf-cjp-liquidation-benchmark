"""Evaluate trained RL agents and compare against Task 2 analytic/FD."""
from __future__ import annotations
import pickle
from pathlib import Path as _Path

import numpy as np
import matplotlib.pyplot as plt

from ..common.params import TASK2
from ..common.simulator import Simulator
from ..common.twap import make_twap_mo_policy
from ..common import style as S
from ..task2_lo_mo import fd_solver as fd


S.apply()
OUT = _Path(__file__).resolve().parents[2] / "results" / "task3"


def _learned_policy_from_tabular(data):
    Q = data["Q"]
    depth_grid = data["depth_grid"]
    t_grid = data["t_grid"]
    nt = len(t_grid); n_depth = len(depth_grid)
    dt = t_grid[1] - t_grid[0]

    depth = np.zeros((nt, TASK2.Q0 + 1))
    trigger = np.zeros((nt, TASK2.Q0 + 1), dtype=bool)
    for i in range(nt):
        for q in range(1, TASK2.Q0 + 1):
            a = int(np.argmax(Q[i, q]))
            if a == n_depth:
                trigger[i, q] = True
            else:
                depth[i, q] = float(depth_grid[a])

    def policy(t, q, S):
        if q <= 0:
            return (0.0, False)
        i = int(np.clip(round(t / dt), 0, nt - 1))
        if trigger[i, q]:
            return (0.0, True)
        return (float(depth[i, q]), False)
    return policy, depth, trigger


def _learned_policy_from_ddqn(data):
    depth = data["depth"]; trigger = data["trigger"]
    t_grid = data["t_grid"]
    nt = len(t_grid); dt = t_grid[1] - t_grid[0]

    def policy(t, q, S):
        if q <= 0:
            return (0.0, False)
        i = int(np.clip(round(t / dt), 0, nt - 1))
        if trigger[i, q]:
            return (0.0, True)
        return (float(depth[i, q]), False)
    return policy, depth, trigger


def fig_rl_vs_fd(tabular_file="tabular.pkl", ddqn_file="ddqn.pkl", save=True):
    with open(OUT / tabular_file, "rb") as f:
        tab = pickle.load(f)
    try:
        with open(OUT / ddqn_file, "rb") as f:
            dqn = pickle.load(f)
    except FileNotFoundError:
        dqn = None
    p = tab["params"]
    sol_fine = fd.solve(p, dt=0.01)
    env_t = tab["t_grid"]

    fd_t = sol_fine.t_grid
    idx = np.clip(np.searchsorted(fd_t, env_t), 0, len(fd_t) - 1)
    depth_an = sol_fine.depth_grid()[idx]

    _, depth_tab, trig_tab = _learned_policy_from_tabular(tab)
    if dqn is not None:
        _, depth_dqn, trig_dqn = _learned_policy_from_ddqn(dqn)

    q_levels = [1, 3, 6, 10]
    fig, axes = plt.subplots(1, len(q_levels), figsize=(11.5, 2.9), sharey=True)
    for ax, q in zip(axes, q_levels):
        ax.plot(env_t, depth_an[:, q], color="black", lw=1.6, label="analytic (FD)")
        ax.plot(env_t, depth_tab[:, q], color="#1f3a5f", lw=0.9,
                marker="o", ms=2.5, mec="none", label="tabular Q")
        if dqn is not None:
            ax.plot(env_t, depth_dqn[:, q], color="#8b0000", lw=0.9,
                    marker="s", ms=2.5, mec="none", label="double DQN")
        mo_tab = env_t[trig_tab[:, q]]
        if len(mo_tab) > 0:
            ax.scatter(mo_tab, np.zeros_like(mo_tab), marker="x",
                       color="#1f3a5f", s=22, label="tabular MO")
        ax.set_title(fr"$q = {q}$")
        ax.set_xlabel(r"$t$ (s)")
        if q == q_levels[0]:
            ax.set_ylabel(r"Depth $\delta^\ast$")
        ax.legend(fontsize=6.5, loc="upper right", handlelength=1.5)
    fig.tight_layout()
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "fig1_depth_comparison.pdf")
        fig.savefig(OUT / "fig1_depth_comparison.png")
    plt.close(fig)


def fig_rmse_trace(save=True):
    with open(OUT / "tabular.pkl", "rb") as f:
        tab = pickle.load(f)
    trace = tab["rmse_trace"]
    if not trace:
        return
    eps_x, rmse = zip(*trace)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
    axes[0].plot(eps_x, rmse, color="#1f3a5f", lw=1.0, marker="o", ms=3,
                 mec="none")
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel(r"RMSE $[\,V_{\mathrm{RL}}(t,q) - h(t,q)\,]$")
    axes[0].set_title("Tabular Q-learning: value error")

    returns = tab["returns"]
    window = max(1, len(returns) // 200)
    smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
    axes[1].plot(np.arange(len(smooth)) + window, smooth,
                 color="#1f3a5f", lw=0.9)
    p = tab["params"]
    h0 = tab["h_reference"][0, -1]
    axes[1].axhline(h0, color="#8b0000", lw=1.0, ls="--",
                    label=fr"analytic $h(0,{p.Q0}) = {h0:.4f}$")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel(fr"Smoothed mean return (window $= {window}$)")
    axes[1].set_title("Tabular Q-learning: learning curve")
    axes[1].legend()
    fig.tight_layout()
    if save:
        fig.savefig(OUT / "fig2_learning_curve.pdf")
        fig.savefig(OUT / "fig2_learning_curve.png")
    plt.close(fig)


def bench_rl_mc(n_paths: int = 2000, sim_dt: float = 0.1, seed: int = 42):
    with open(OUT / "tabular.pkl", "rb") as f:
        tab = pickle.load(f)
    policy_tab, _, _ = _learned_policy_from_tabular(tab)
    p = tab["params"]

    sim = Simulator(p, dt=sim_dt, seed=seed)
    print(f"Task 3 MC (tabular RL policy): {n_paths} paths")
    mc_rl = sim.monte_carlo(policy_tab, n_paths, progress=True)

    sol = fd.solve(p, dt=0.01)
    fd_grid = sol.depth_grid()
    trigger = sol.mo_trigger
    fd_dt = sol.t_grid[1] - sol.t_grid[0]

    def policy_fd(t, q, S):
        if q <= 0: return (0.0, False)
        i = int(np.clip(round(t / fd_dt), 0, len(sol.t_grid) - 1))
        if trigger[i, q]: return (0.0, True)
        return (float(max(fd_grid[i, q], 0.0)), False)

    sim2 = Simulator(p, dt=sim_dt, seed=seed)
    mc_opt = sim2.monte_carlo(policy_fd, n_paths, progress=True)

    sim3 = Simulator(p, dt=sim_dt, seed=seed)
    mc_tw = sim3.monte_carlo(make_twap_mo_policy(p), n_paths, progress=True)

    with open(OUT / "mc_bench.pkl", "wb") as f:
        pickle.dump({"rl": mc_rl, "analytic": mc_opt, "twap": mc_tw,
                     "params": p}, f)

    print("\n=== Task 3 RL benchmark vs Task 2 and TWAP ===")
    for label, d in [("RL (tabular)", mc_rl), ("Analytic (FD)", mc_opt),
                     ("TWAP", mc_tw)]:
        mean = np.nanmean(d["avg_price"])
        std = np.nanstd(d["avg_price"])
        pr0 = (d["q_final"] == 0).mean()
        mo = d["n_mo"].mean()
        print(f"{label:<16s}: mean={mean:.6f}  std={std:.6f}  "
              f"Pr[q_T=0]={pr0:.3f}  MOs={mo:.2f}")
    return mc_rl, mc_opt, mc_tw


def fig_bench_summary(save=True):
    with open(OUT / "mc_bench.pkl", "rb") as f:
        d = pickle.load(f)
    p = d["params"]
    labels = ["Analytic (FD)", "RL (tabular)", "TWAP"]
    colors = ["black", "#1f3a5f", "#8b0000"]
    datasets = [d["analytic"], d["rl"], d["twap"]]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.6))
    for name, mc, color in zip(labels, datasets, colors):
        price = mc["avg_price"]
        if name == "TWAP":
            price = mc["twap_price"]
        finite = price[np.isfinite(price)]
        ax.hist(finite - p.S0, bins=50, alpha=0.55,
                histtype="stepfilled", color=color, edgecolor=color,
                linewidth=0.8, density=True,
                label=fr"{name} (mean $= {np.mean(finite - p.S0):.4f}$)")
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel(r"Average execution price $-\,S_0$ (\$)")
    ax.set_ylabel("Density")
    ax.set_title("Per-share implementation premium")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(OUT / "fig3_bench_summary.pdf")
        fig.savefig(OUT / "fig3_bench_summary.png")
    plt.close(fig)


def main():
    fig_rl_vs_fd()
    fig_rmse_trace()
    bench_rl_mc(n_paths=2000, sim_dt=0.1, seed=42)
    fig_bench_summary()
    print("Task 3 figures saved to", OUT)


if __name__ == "__main__":
    main()
