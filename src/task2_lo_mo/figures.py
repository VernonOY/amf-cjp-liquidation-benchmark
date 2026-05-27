"""Task 2 figure generation in academic serif style."""
from __future__ import annotations
import pickle
from pathlib import Path as _Path

import numpy as np
import matplotlib.pyplot as plt

from ..common.params import TASK2, Params
from ..common import style as S
from . import fd_solver as fd
from . import analytic_q12 as aq


S.apply()
OUT = _Path(__file__).resolve().parents[2] / "results" / "task2"


def fig_fd_vs_analytic(p: Params = TASK2, fd_dt: float = 0.005, save=True):
    """FD vs analytic omega(t,q) for q=1,2."""
    sol = fd.solve(p, dt=fd_dt, Q_max=max(p.Q0, 2))
    t = sol.t_grid
    ome_q1_fd = np.exp(p.kappa * sol.h[:, 1])
    ome_q1_an = aq.omega_q1(p, t)
    ome_q2_fd = np.exp(p.kappa * sol.h[:, 2])
    ome_q2_an = aq.omega_q2(p, t)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    axes[0].plot(t, ome_q1_an, color="black", lw=1.6, label="analytic")
    axes[0].plot(t, ome_q1_fd, color="#8b0000", lw=1.0, ls=(0, (4, 2)),
                 label="finite difference")
    axes[0].set_title(r"$\omega(t,1)$")
    axes[0].set_xlabel(r"$t$ (s)"); axes[0].set_ylabel(r"$\omega$"); axes[0].legend()
    axes[1].plot(t, ome_q2_an, color="black", lw=1.6, label="analytic")
    axes[1].plot(t, ome_q2_fd, color="#8b0000", lw=1.0, ls=(0, (4, 2)),
                 label="finite difference")
    tau2 = aq.critical_time_q2(p)
    if tau2 is not None:
        axes[1].axvline(tau2, color="#1f3a5f", lw=0.8, ls=":",
                        label=fr"$\tau_2 = {tau2:.2f}$")
    axes[1].set_title(r"$\omega(t,2)$")
    axes[1].set_xlabel(r"$t$ (s)"); axes[1].legend()
    fig.tight_layout()
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "fig1_fd_vs_analytic.pdf")
        fig.savefig(OUT / "fig1_fd_vs_analytic.png")
    plt.close(fig)


def fig_policy_maps(phi_list=(1e-6, 1e-5, 1e-4), save=True):
    fig, axes = plt.subplots(2, len(phi_list), figsize=(3.4 * len(phi_list), 5.6))
    for j, phi in enumerate(phi_list):
        p = Params(**{**TASK2.__dict__, "phi": phi})
        sol = fd.solve(p, dt=0.01)
        t = sol.t_grid
        depth = sol.depth_grid()
        trig = sol.mo_trigger

        colors = S.gradient_colors(p.Q0)
        ax_d = axes[0, j]
        for q, c in zip(range(1, p.Q0 + 1), colors):
            ax_d.plot(t, depth[:, q], color=c, lw=0.9, label=fr"$q={q}$")
        ax_d.set_title(fr"$\phi = {phi:.0e}$")
        ax_d.set_xlabel(r"$t$ (s)")
        if j == 0:
            ax_d.set_ylabel(r"Posted depth $\delta^\ast$")
            ax_d.legend(fontsize=6.5, ncol=2, loc="lower left", handlelength=1.2)

        ax_t = axes[1, j]
        mask = trig.copy()
        mask[-1, :] = False  # strip terminal row (always "executes")
        tt, qq = np.where(mask)
        cont_tt, cont_qq = np.where(~mask & (np.arange(trig.shape[1])[None, :] > 0))
        ax_t.scatter(t[cont_tt], cont_qq, s=4, color="#1f3a5f", alpha=0.15,
                     label="post LO")
        ax_t.scatter(t[tt], qq, s=6, color="#8b0000", alpha=0.8,
                     label="execute MO")
        ax_t.set_xlabel(r"$t$ (s)")
        ax_t.set_yticks(range(p.Q0 + 1))
        if j == 0:
            ax_t.set_ylabel(r"Inventory $q$")
            ax_t.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    if save:
        fig.savefig(OUT / "fig2_policy_maps.pdf")
        fig.savefig(OUT / "fig2_policy_maps.png")
    plt.close(fig)


def fig_sample_paths(tag: str = "default", save=True):
    with open(OUT / f"sample_paths_{tag}.pkl", "rb") as f:
        data = pickle.load(f)
    paths = data["paths"]; p = data["params"]

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.2))
    path_colors = ["#000000", "#1f3a5f", "#8b0000"]
    for pth, c in zip(paths, path_colors):
        axes[0, 0].plot(pth.t, pth.S, color=c, lw=0.9)
        axes[0, 1].step(pth.t, pth.delta, color=c, where="post", lw=0.9)
        mo_times = pth.t[pth.mo_sends]
        if len(mo_times) > 0:
            axes[0, 1].scatter(mo_times, np.zeros_like(mo_times),
                               marker="v", color=c, s=22, zorder=5, edgecolor="black",
                               linewidths=0.3)
        axes[1, 0].step(pth.t, pth.q, color=c, where="post", lw=0.9)
        if len(mo_times) > 0:
            axes[1, 0].scatter(mo_times, pth.q[pth.mo_sends],
                               marker="v", color=c, s=22, zorder=5,
                               edgecolor="black", linewidths=0.3)
        shares_sold = p.Q0 - pth.q
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(shares_sold > 0,
                           pth.X / np.maximum(shares_sold, 1), np.nan)
        axes[1, 1].plot(pth.t, avg, color=c, lw=0.9)
        twap = np.cumsum(pth.S) / np.arange(1, len(pth.S) + 1)
        axes[1, 1].plot(pth.t, twap, color=c, lw=0.8, ls=(0, (3, 2)), alpha=0.8)

    axes[0, 0].set_ylabel(r"Midprice $S_t$"); axes[0, 0].set_xlabel(r"$t$ (s)")
    axes[0, 1].set_ylabel(r"Depth $\delta^\ast_t$ ($\blacktriangledown$ = MO)")
    axes[0, 1].set_xlabel(r"$t$ (s)")
    axes[1, 0].set_ylabel(r"Inventory $Q_t$"); axes[1, 0].set_xlabel(r"$t$ (s)")
    axes[1, 1].set_ylabel(r"Average price / share"); axes[1, 1].set_xlabel(r"$t$ (s)")
    axes[0, 0].set_title("(a)"); axes[0, 1].set_title("(b)")
    axes[1, 0].set_title("(c)"); axes[1, 1].set_title("(d)")
    fig.tight_layout()
    if save:
        fig.savefig(OUT / "fig3_sample_paths.pdf")
        fig.savefig(OUT / "fig3_sample_paths.png")
    plt.close(fig)


def fig_mc_summary(tag: str = "default", save=True):
    with open(OUT / f"mc_{tag}.pkl", "rb") as f:
        data = pickle.load(f)
    mc = data["optimal"]; tw = data["twap"]; p = data["params"]

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.2))

    max_mo = max(int(mc["n_mo"].max()), 5)
    axes[0].hist(mc["n_mo"], bins=np.arange(max_mo + 2) - 0.5,
                 color="#4c4c4c", edgecolor="black", linewidth=0.6)
    axes[0].set_xlabel(r"Number of agent MOs")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(fr"MO usage (mean $= {mc['n_mo'].mean():.2f}$)")

    axes[1].hist(mc["q_final"], bins=np.arange(p.Q0 + 2) - 0.5,
                 color="#4c4c4c", edgecolor="black", linewidth=0.6)
    axes[1].set_xlabel(r"Terminal inventory $Q_T$")
    axes[1].set_ylabel("Frequency")
    pr0 = (mc["q_final"] == 0).mean()
    axes[1].set_title(fr"$\Pr[Q_T = 0] = {pr0:.3f}$")

    heat = mc["q_heat"]; t = mc["t_grid"]
    q_bins = np.arange(p.Q0 + 2) - 0.5
    H = np.zeros((p.Q0 + 1, len(t)))
    for j in range(len(t)):
        counts, _ = np.histogram(heat[:, j], bins=q_bins)
        H[:, j] = counts / counts.sum()
    im = axes[2].imshow(H, origin="lower", aspect="auto",
                        extent=[t[0], t[-1], -0.5, p.Q0 + 0.5],
                        cmap="cividis")
    axes[2].plot(t, heat.mean(axis=0), color="white", lw=1.2, ls="--",
                 label=r"mean $Q_t$")
    axes[2].set_xlabel(r"$t$ (s)"); axes[2].set_ylabel(r"$Q_t$")
    axes[2].set_title("Inventory distribution")
    axes[2].legend(loc="upper right")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    shares = p.Q0 - mc["q_final"]
    opt_avg = np.where(shares > 0, mc["avg_price"], np.nan)
    savings = opt_avg - tw["twap_price"]
    axes[3].hist(savings[~np.isnan(savings)], bins=40,
                 color="#4c4c4c", edgecolor="black", linewidth=0.4)
    axes[3].axvline(np.nanmean(savings), color="#8b0000", lw=1.2, ls="--",
                    label=fr"mean $= {np.nanmean(savings):.4f}$")
    axes[3].set_xlabel(r"Savings / share vs. TWAP (\$)")
    axes[3].set_ylabel("Frequency")
    axes[3].set_title("Per-share savings")
    axes[3].legend()

    fig.tight_layout()
    if save:
        fig.savefig(OUT / "fig4_mc_summary.pdf")
        fig.savefig(OUT / "fig4_mc_summary.png")
    plt.close(fig)


def fig_task1_vs_task2(save=True):
    root = OUT.parent
    try:
        with open(root / "task1" / "mc.pkl", "rb") as f:
            d1 = pickle.load(f)
    except FileNotFoundError:
        return
    with open(OUT / "mc_default.pkl", "rb") as f:
        d2 = pickle.load(f)

    p1 = d1["params"]; p2 = d2["params"]
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.4))
    shares1 = p1.Q0 - d1["optimal"]["q_final"]
    shares2 = p2.Q0 - d2["optimal"]["q_final"]
    with np.errstate(invalid="ignore"):
        p1_avg = np.where(shares1 > 0, d1["optimal"]["avg_price"], np.nan)
        p2_avg = np.where(shares2 > 0, d2["optimal"]["avg_price"], np.nan)
    tw1 = d1["twap"]["twap_price"]; tw2 = d2["twap"]["twap_price"]

    bins = 40
    ax.hist(p1_avg - tw1, bins=bins, alpha=0.55, color="#1f3a5f",
            edgecolor="black", linewidth=0.3,
            label=fr"Task 1 (LO only, $Q_0={p1.Q0}$)")
    ax.hist(p2_avg - tw2, bins=bins, alpha=0.55, color="#8b0000",
            edgecolor="black", linewidth=0.3,
            label=fr"Task 2 (LO+MO, $Q_0={p2.Q0}$)")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"Average execution price $-$ TWAP (\$)")
    ax.set_ylabel("Frequency")
    ax.set_title("Per-share savings over TWAP")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(OUT / "fig5_task1_vs_task2.pdf")
        fig.savefig(OUT / "fig5_task1_vs_task2.png")
    plt.close(fig)


def main():
    fig_fd_vs_analytic()
    fig_policy_maps()
    fig_sample_paths()
    fig_mc_summary()
    fig_task1_vs_task2()
    print("Task 2 figures saved to", OUT)


if __name__ == "__main__":
    main()
