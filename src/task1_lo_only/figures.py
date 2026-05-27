"""Reproduce Task 1 figures (Fig 8.2–8.5 style) in academic serif style."""
from __future__ import annotations
import pickle
from pathlib import Path as _Path

import numpy as np
import matplotlib.pyplot as plt

from ..common.params import TASK1, Params
from ..common import style as S
from .solver import precompute_delta_grid


S.apply()
OUT = _Path(__file__).resolve().parents[2] / "results" / "task1"


def fig_optimal_depth_curves(save=True):
    """Fig 8.2-style: δ*(t,q) vs t for two α values."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, alpha in zip(axes, [1e-4, 1e-3]):
        p = Params(**{**TASK1.__dict__, "alpha": alpha})
        t = np.linspace(0.0, p.T, 200)
        grid = precompute_delta_grid(p, t)
        colors = S.gradient_colors(p.Q0)
        for q, c in zip(range(1, p.Q0 + 1), colors):
            ax.plot(t, grid[:, q], color=c, lw=1.1, label=fr"$q={q}$")
        ax.set_title(fr"$\alpha = {alpha:g}$")
        ax.set_xlabel(r"Time $t$ (s)")
        ax.set_ylabel(r"Optimal depth $\delta^\ast(t,q)$")
    axes[-1].legend(loc="upper right", ncol=1, handlelength=1.6)
    fig.tight_layout()
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "fig1_optimal_depth.pdf")
        fig.savefig(OUT / "fig1_optimal_depth.png")
    plt.close(fig)


def fig_sample_paths(save=True):
    with open(OUT / "sample_paths.pkl", "rb") as f:
        data = pickle.load(f)
    paths = data["paths"]; p = data["params"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    path_colors = ["#000000", "#1f3a5f", "#8b0000"]
    for pth, c in zip(paths, path_colors):
        axes[0, 0].plot(pth.t, pth.S, color=c, lw=0.9)
        axes[0, 1].step(pth.t, pth.delta, color=c, where="post", lw=0.9)
        axes[1, 0].step(pth.t, pth.q, color=c, where="post", lw=0.9)
        shares_sold = p.Q0 - pth.q
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(shares_sold > 0,
                           pth.X / np.maximum(shares_sold, 1), np.nan)
        axes[1, 1].plot(pth.t, avg, color=c, lw=1.0)
        twap = np.cumsum(pth.S) / np.arange(1, len(pth.S) + 1)
        axes[1, 1].plot(pth.t, twap, color=c, lw=0.9, ls=(0, (3, 2)), alpha=0.85)

    axes[0, 0].set_ylabel(r"Midprice $S_t$"); axes[0, 0].set_xlabel(r"$t$ (s)")
    axes[0, 1].set_ylabel(r"Posted depth $\delta^\ast_t$"); axes[0, 1].set_xlabel(r"$t$ (s)")
    axes[1, 0].set_ylabel(r"Inventory $Q_t$"); axes[1, 0].set_xlabel(r"$t$ (s)")
    axes[1, 1].set_ylabel(r"Average price per share"); axes[1, 1].set_xlabel(r"$t$ (s)")
    axes[0, 0].set_title(r"(a)"); axes[0, 1].set_title(r"(b)")
    axes[1, 0].set_title(r"(c)"); axes[1, 1].set_title(r"(d)")
    fig.tight_layout()
    if save:
        fig.savefig(OUT / "fig2_sample_paths.pdf")
        fig.savefig(OUT / "fig2_sample_paths.png")
    plt.close(fig)


def fig_mc_summary(save=True):
    with open(OUT / "mc.pkl", "rb") as f:
        data = pickle.load(f)
    mc = data["optimal"]; tw = data["twap"]
    p = data["params"]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

    # (i) terminal-inventory histogram
    axes[0].hist(mc["q_final"], bins=np.arange(p.Q0 + 2) - 0.5,
                 color="#4c4c4c", edgecolor="black", linewidth=0.6)
    axes[0].set_xlabel(r"Terminal inventory $Q_T$")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(fr"$n = {len(mc['q_final'])}$ paths")

    # (ii) inventory heatmap over time.
    heat = mc["q_heat"]
    t = mc["t_grid"]
    q_bins = np.arange(p.Q0 + 2) - 0.5
    H = np.zeros((p.Q0 + 1, len(t)))
    for j in range(len(t)):
        counts, _ = np.histogram(heat[:, j], bins=q_bins)
        H[:, j] = counts / counts.sum()
    im = axes[1].imshow(H, origin="lower", aspect="auto",
                        extent=[t[0], t[-1], -0.5, p.Q0 + 0.5],
                        cmap="cividis")
    axes[1].plot(t, heat.mean(axis=0), color="white", lw=1.2, ls="--",
                 label=r"mean $Q_t$")
    axes[1].set_xlabel(r"Time $t$ (s)"); axes[1].set_ylabel(r"Inventory $Q_t$")
    axes[1].set_title("Inventory distribution over time")
    axes[1].legend(loc="upper right")
    cb = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label("Probability")

    # (iii) cost savings per share vs TWAP.
    shares = p.Q0 - mc["q_final"]
    opt_avg = np.where(shares > 0, mc["avg_price"], np.nan)
    savings = opt_avg - tw["twap_price"]
    axes[2].hist(savings[~np.isnan(savings)], bins=40,
                 color="#4c4c4c", edgecolor="black", linewidth=0.4)
    axes[2].axvline(np.nanmean(savings), color="#8b0000", lw=1.2, ls="--",
                    label=fr"mean $= {np.nanmean(savings):.4f}$")
    axes[2].set_xlabel(r"Savings / share vs. TWAP (\$)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Per-share savings")
    axes[2].legend()

    fig.tight_layout()
    if save:
        fig.savefig(OUT / "fig3_mc_summary.pdf")
        fig.savefig(OUT / "fig3_mc_summary.png")
    plt.close(fig)


def main():
    fig_optimal_depth_curves()
    fig_sample_paths()
    fig_mc_summary()
    print("Task 1 figures saved to", OUT)


if __name__ == "__main__":
    main()
