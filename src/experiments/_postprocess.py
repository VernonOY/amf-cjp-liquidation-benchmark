"""Postprocessing: render Figs 10, 15, 16 and Table 5 from experiment CSVs."""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np


def _read_rows(path: str | Path) -> list[dict]:
    with open(path, "r") as fp:
        return list(csv.DictReader(fp))


def make_fig10_failure_modes(exp4_csv: str, fig_path: str) -> None:
    """3-panel failure-mode plot. Panel A: MO under-use vs epsilon-bonus.
    Panel B: value error vs depth grid size. Panel C: reserved for CIR
    bucket count (Phase 4 follow-up, not in MVS)."""
    import matplotlib.pyplot as plt
    from ..common.style import apply as apply_style

    apply_style()
    rows = _read_rows(exp4_csv)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Panel (a) — MO under-use
    ax = axes[0]
    rows_a = [r for r in rows if r["panel"] == "a"]
    by_agent: dict[str, dict[float, list[float]]] = {}
    for r in rows_a:
        eps = float(r["sweep_value"])
        v = float(r["mo_per_path"]) if r["mo_per_path"] not in ("nan", "") else float("nan")
        by_agent.setdefault(r["agent"], {}).setdefault(eps, []).append(v)
    for ag in sorted(by_agent.keys()):
        eps = sorted(by_agent[ag].keys())
        means = [float(np.mean(by_agent[ag][e])) for e in eps]
        ax.plot(eps, means, marker="o", label=ag)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel(r"MO-bonus $\epsilon$")
    ax.set_ylabel("MO firings per path")
    ax.set_title("(a) MO under-use vs MO-bonus")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel (b) — Action-grid bias
    ax = axes[1]
    rows_b = [r for r in rows if r["panel"] == "b"]
    by_nd: dict[int, list[float]] = {}
    for r in rows_b:
        nd = int(r["sweep_value"])
        v = float(r["value_err_rmse"]) if r["value_err_rmse"] not in ("nan", "") else float("nan")
        by_nd.setdefault(nd, []).append(v)
    nds = sorted(by_nd.keys())
    means = [float(np.mean(by_nd[nd])) for nd in nds]
    ax.plot(nds, means, marker="s", color="C1", label="Tabular Q")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"action-grid size $n_d$")
    ax.set_ylabel(r"$\|V_{\rm RL} - h\|_{\rm RMSE}$")
    ax.set_title("(b) Action-grid bias")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path)
    import matplotlib.pyplot as plt2
    plt2.close(fig)


def make_fig15_robustness_heatmap(exp3_dir: str, fig_path: str) -> None:
    """Bivariate sensitivity heatmaps for (kappa, phi) and (xi, alpha)."""
    import matplotlib.pyplot as plt
    from ..common.style import apply as apply_style

    apply_style()
    rows = _read_rows(Path(exp3_dir) / "bivariate.csv")
    pairs = sorted({r["pair"] for r in rows})

    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.5))
    if len(pairs) == 1:
        axes = [axes]
    for ax, pair in zip(axes, pairs):
        rows_p = [r for r in rows if r["pair"] == pair]
        mas = sorted({float(r["ma"]) for r in rows_p})
        mbs = sorted({float(r["mb"]) for r in rows_p})
        H = np.full((len(mas), len(mbs)), np.nan)
        for r in rows_p:
            i = mas.index(float(r["ma"]))
            j = mbs.index(float(r["mb"]))
            H[i, j] = float(r["premium"]) if r["premium"] not in ("nan", "") else np.nan
        im = ax.imshow(H, aspect="auto", origin="lower", cmap="viridis",
                        extent=[min(mbs), max(mbs), min(mas), max(mas)])
        a, b = pair.split("+")
        ax.set_xlabel(f"{b} multiplier")
        ax.set_ylabel(f"{a} multiplier")
        ax.set_title(f"premium: {pair}")
        fig.colorbar(im, ax=ax)
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path)
    import matplotlib.pyplot as plt2
    plt2.close(fig)


def make_fig16_misspec(exp3_dir: str, fig_path: str) -> None:
    """Train-test misspecification: average premium across diagonal vs
    off-diagonal. We pool over (kap_train, lam_train) pairs and show how
    the deployed policy fares at each (kap_test, lam_test)."""
    import matplotlib.pyplot as plt
    from ..common.style import apply as apply_style

    apply_style()
    rows = _read_rows(Path(exp3_dir) / "misspec.csv")
    test_lams = sorted({float(r["lam_test_mult"]) for r in rows})
    test_kaps = sorted({float(r["kap_test_mult"]) for r in rows})
    # Pool: at each (lam_test, kap_test), average over all (lam_train, kap_train)
    H = np.full((len(test_lams), len(test_kaps)), np.nan)
    for i, lt in enumerate(test_lams):
        for j, kt in enumerate(test_kaps):
            vals = [float(r["premium"]) for r in rows
                    if float(r["lam_test_mult"]) == lt
                    and float(r["kap_test_mult"]) == kt
                    and r["premium"] not in ("nan", "")]
            if vals:
                H[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(H, aspect="auto", origin="lower", cmap="RdYlGn",
                    extent=[min(test_kaps), max(test_kaps),
                            min(test_lams), max(test_lams)])
    ax.set_xlabel(r"$\kappa$ test multiplier")
    ax.set_ylabel(r"$\lambda$ test multiplier")
    ax.set_title("Misspecification premium\n(averaged over train cells)")
    fig.colorbar(im, ax=ax, label="premium")
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path)
    import matplotlib.pyplot as plt2
    plt2.close(fig)


def make_table5_agent_summary(exp1_csv: str, out_path: str) -> None:
    """Compact agent-comparison table: per agent, n_to_reach_95pct_optimal,
    wall_clock_at_95pct, value_err at n=10^4, premium at n=10^4 (mean +/- std
    across seeds). Writes a Markdown table."""
    rows = _read_rows(exp1_csv)
    by_agent: dict[str, dict[int, list[dict]]] = {}
    for r in rows:
        ag = r["agent"]
        n = int(r["n"])
        by_agent.setdefault(ag, {}).setdefault(n, []).append(r)

    # Per agent, max premium across n -> use as proxy for the "optimal" target
    lines = ["| Agent | n at peak premium | wall (s) | value_err@n=10k | premium@n=10k |",
             "|---|---|---|---|---|"]
    for ag in sorted(by_agent.keys()):
        ns = sorted(by_agent[ag].keys())
        # Peak premium n
        prem_by_n = {n: float(np.mean([float(r["premium_vs_twap"])
                                         for r in by_agent[ag][n]
                                         if r["premium_vs_twap"] not in ("nan", "")]))
                     for n in ns}
        peak_n = max(prem_by_n, key=prem_by_n.get)
        wall_at_peak = float(np.mean([float(r["wall_clock_sec"])
                                       for r in by_agent[ag][peak_n]]))
        # n=10k slice (or nearest)
        n_target = min(ns, key=lambda x: abs(x - 10000))
        cells = by_agent[ag][n_target]
        ve = float(np.mean([float(r["value_err_rmse"]) for r in cells
                             if r["value_err_rmse"] not in ("nan", "")]))
        prem_mean = float(np.mean([float(r["premium_vs_twap"]) for r in cells
                                     if r["premium_vs_twap"] not in ("nan", "")]))
        prem_std = float(np.std([float(r["premium_vs_twap"]) for r in cells
                                   if r["premium_vs_twap"] not in ("nan", "")]))
        lines.append(f"| {ag} | {peak_n} | {wall_at_peak:.1f} | "
                      f"{ve:.4f} | {prem_mean:+.4f} ± {prem_std:.4f} |")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n")
