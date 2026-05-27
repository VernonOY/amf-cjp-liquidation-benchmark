"""Render all paper-grade figures from the full-run CSVs.

Inputs:
  data/exp1/sample_complexity_full.csv     (Tabular / DDQN / Plug-in)
  data/exp1/sample_complexity_hybrid_ppo.csv  (Hybrid PPO replacement)
  data/exp2_full/fd_cir.csv + cir_validation.csv
  data/exp3_full/{univariate,bivariate,misspec}.csv
  data/exp4/failure_modes_full.csv

Outputs:
  figures/fig9_sample_complexity.pdf  (4-panel sample complexity, merged hybrid)
  figures/fig10_failure_modes.pdf     (MO under-use + action-grid bias)
  figures/fig11_h_surface_cir.pdf
  figures/fig12_mo_trigger_cir.pdf
  figures/fig15_robustness_bivariate.pdf
  figures/fig16_robustness_misspec.pdf
  figures/fig_regime_ii_pareto.pdf
  figures/fig_cir_validation.pdf
  results/tab5_agent_comparison.md
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.common.style import apply as apply_style  # noqa: E402


def _read_rows(path):
    with open(path, "r") as fp:
        return list(csv.DictReader(fp))


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def render_fig9(exp1_other_csv, exp1_hybrid_csv, fig_path):
    """4-panel sample complexity: value_err, policy_err, premium, clearance."""
    import matplotlib.pyplot as plt
    apply_style()

    rows_other = _read_rows(exp1_other_csv)
    rows_hybrid = _read_rows(exp1_hybrid_csv)

    by_agent = defaultdict(lambda: defaultdict(list))
    for r in rows_other:
        if r["agent"] == "C_hybrid":
            continue  # superseded by the PPO rerun
        by_agent[r["agent"]][int(r["n"])].append(r)
    for r in rows_hybrid:
        by_agent[r["agent"]][int(r["n"])].append(r)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))
    ax_val, ax_pol, ax_prem, ax_clr = axes.ravel()
    panels = [
        (ax_val, "value_err_rmse", r"$\|V_{\rm RL} - h\|_{\rm RMSE}$", True),
        (ax_pol, "policy_err_l1", r"mean $|\delta_{\rm RL} - \delta^*|$", True),
        (ax_prem, "premium_vs_twap", "Premium vs TWAP", False),
        (ax_clr, "clearance_prob", r"$\mathbb{P}[Q_T = 0]$", False),
    ]
    nice = {"A_tabular": "A Tabular",
            "B_ddqn": "B DDQN",
            "C_hybrid": "C Hybrid",
            "D_plugin": "D Plug-in"}
    markers = {"A_tabular": "o", "B_ddqn": "s", "C_hybrid": "^", "D_plugin": "D"}
    colors = {"A_tabular": "C0", "B_ddqn": "C1",
               "C_hybrid": "C2", "D_plugin": "C3"}
    tick_fs = 13
    label_fs = 15

    for ax, key, label, logy in panels:
        for ag in sorted(by_agent.keys()):
            ns = sorted(by_agent[ag].keys())
            means, stds = [], []
            for n in ns:
                vals = [_f(r[key]) for r in by_agent[ag][n]]
                vals = [v for v in vals if np.isfinite(v)]
                if not vals:
                    means.append(np.nan); stds.append(0.0)
                    continue
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals) / np.sqrt(max(len(vals), 1))))
            means = np.array(means)
            stds = np.array(stds)
            ax.errorbar(ns, means, yerr=stds,
                         marker=markers.get(ag, "o"),
                         color=colors.get(ag), capsize=2,
                         label=nice.get(ag, ag), linewidth=2.0, markersize=6.5)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.tick_params(labelsize=tick_fs)
        ax.set_xlabel("sample budget (env transitions)", fontsize=label_fs)
        ax.set_ylabel(label, fontsize=label_fs)
        ax.grid(True, alpha=0.3, which="both")
        if key == "premium_vs_twap":
            ax.set_ylim(bottom=0.0, top=0.205)
    handles, labels = ax_val.get_legend_handles_labels()
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
    # No fig.suptitle — figure caption is written in LaTeX via \caption{...}.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  wrote {fig_path}", flush=True)


def render_fig10(exp4_csv, fig_path):
    """Failure-mode 2-panel."""
    import matplotlib.pyplot as plt
    apply_style()
    rows = _read_rows(exp4_csv)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    # Panel A
    ax = axes[0]
    rows_a = [r for r in rows if r["panel"] == "a"]
    by_ag = defaultdict(lambda: defaultdict(list))
    for r in rows_a:
        by_ag[r["agent"]][_f(r["sweep_value"])].append(_f(r["mo_per_path"]))
    for ag in sorted(by_ag.keys()):
        eps = sorted(by_ag[ag].keys())
        means = [float(np.mean(by_ag[ag][e])) for e in eps]
        stds = [float(np.std(by_ag[ag][e]) / np.sqrt(max(len(by_ag[ag][e]), 1)))
                for e in eps]
        ax.errorbar(eps, means, yerr=stds, marker="o",
                     label=ag, capsize=2, linewidth=1.5)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel(r"MO-bonus $\epsilon$ added during training")
    ax.set_ylabel("MO firings per evaluation path")
    ax.set_title("(a) MO under-use vs MO-bonus")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # Panel B
    ax = axes[1]
    rows_b = [r for r in rows if r["panel"] == "b"]
    by_nd = defaultdict(list)
    for r in rows_b:
        by_nd[int(_f(r["sweep_value"]))].append(_f(r["value_err_rmse"]))
    nds = sorted(by_nd.keys())
    means = [float(np.mean(by_nd[nd])) for nd in nds]
    stds = [float(np.std(by_nd[nd]) / np.sqrt(max(len(by_nd[nd]), 1)))
            for nd in nds]
    ax.errorbar(nds, means, yerr=stds, marker="s", color="C1",
                 capsize=2, linewidth=1.5, label="Tabular Q")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"action-grid size $n_d$")
    ax.set_ylabel(r"$\|V_{\rm RL} - h\|_{\rm RMSE}$")
    ax.set_title("(b) Action-grid bias")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # No fig.suptitle — caption is in LaTeX.
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  wrote {fig_path}", flush=True)


def render_fig_univariate(exp3_dir, fig_path):
    """Univariate parameter sensitivity (exp3 §7.1). Five parameters, each
    swept over multipliers {0.5, 0.75, 1.0, 1.5, 2.0}. We plot premium and
    MO usage as functions of the multiplier, one line per parameter."""
    import matplotlib.pyplot as plt
    apply_style()
    rows = _read_rows(Path(exp3_dir) / "univariate.csv")
    by_param = defaultdict(list)
    for r in rows:
        by_param[r["param"]].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    nice = {"lam": r"$\lambda$ (fill rate)",
             "kappa": r"$\kappa$ (depth decay)",
             "xi": r"$\xi$ (half-spread)",
             "alpha": r"$\alpha$ (terminal impact)",
             "phi": r"$\phi$ (running penalty)"}
    markers = {"lam": "o", "kappa": "s", "xi": "^", "alpha": "D", "phi": "v"}

    # Panel (a) — premium
    ax = axes[0]
    for p in sorted(by_param.keys()):
        rs = sorted(by_param[p], key=lambda x: _f(x["multiplier"]))
        ms = [_f(r["multiplier"]) for r in rs]
        ys = [_f(r["premium"]) for r in rs]
        ax.plot(ms, ys, marker=markers.get(p, "o"),
                 linewidth=1.5, markersize=5, label=nice.get(p, p))
    ax.set_xscale("log")
    ax.set_xlabel("parameter multiplier")
    ax.set_ylabel("FD-optimal premium vs TWAP")
    ax.set_title("(a) Premium")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best")

    # Panel (b) — MO usage
    ax = axes[1]
    for p in sorted(by_param.keys()):
        rs = sorted(by_param[p], key=lambda x: _f(x["multiplier"]))
        ms = [_f(r["multiplier"]) for r in rs]
        ys = [_f(r["mo_per_path"]) for r in rs]
        ax.plot(ms, ys, marker=markers.get(p, "o"),
                 linewidth=1.5, markersize=5, label=nice.get(p, p))
    ax.set_xscale("log")
    ax.set_xlabel("parameter multiplier")
    ax.set_ylabel("MO firings per path")
    ax.set_title("(b) MO usage")
    ax.grid(True, alpha=0.3, which="both")

    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  wrote {fig_path}", flush=True)


def render_fig15_bivariate(exp3_dir, fig_path):
    """Two bivariate sensitivity heatmaps."""
    import matplotlib.pyplot as plt
    apply_style()
    rows = _read_rows(Path(exp3_dir) / "bivariate.csv")
    pairs = sorted({r["pair"] for r in rows})
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.6))
    if len(pairs) == 1:
        axes = [axes]
    for ax, pair in zip(axes, pairs):
        rows_p = [r for r in rows if r["pair"] == pair]
        mas = sorted({_f(r["ma"]) for r in rows_p})
        mbs = sorted({_f(r["mb"]) for r in rows_p})
        H = np.full((len(mas), len(mbs)), np.nan)
        for r in rows_p:
            i = mas.index(_f(r["ma"]))
            j = mbs.index(_f(r["mb"]))
            H[i, j] = _f(r["premium"])
        im = ax.imshow(H, aspect="auto", origin="lower", cmap="viridis",
                        extent=[min(mbs), max(mbs), min(mas), max(mas)])
        a, b = pair.split("+")
        ax.set_xlabel(f"{b} multiplier")
        ax.set_ylabel(f"{a} multiplier")
        ax.set_title(f"({a}, {b}) sweep")
        fig.colorbar(im, ax=ax, label="premium vs TWAP", shrink=0.85)
        # Per-cell numeric annotation
        for i, ma in enumerate(mas):
            for j, mb in enumerate(mbs):
                if not np.isnan(H[i, j]):
                    color = "white" if H[i, j] < np.nanmean(H) else "black"
                    ax.text(mb, ma, f"{H[i,j]:.2f}",
                            ha="center", va="center", fontsize=7, color=color)
    # No fig.suptitle — caption is in LaTeX.
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  wrote {fig_path}", flush=True)


def render_fig16_misspec(exp3_dir, fig_path):
    """Train-test misspecification heatmap, two panels:
       (a) absolute premium of the deployed policy at the test cell
       (b) RELATIVE gap vs the matched-training premium (the diagonal of
           the train=test grid), which is the cleanest 'cost of mis-
           specification' object.
    """
    import matplotlib.pyplot as plt
    apply_style()
    rows = _read_rows(Path(exp3_dir) / "misspec.csv")
    test_lams = sorted({_f(r["lam_test_mult"]) for r in rows})
    test_kaps = sorted({_f(r["kap_test_mult"]) for r in rows})
    train_lams = sorted({_f(r["lam_train_mult"]) for r in rows})
    train_kaps = sorted({_f(r["kap_train_mult"]) for r in rows})

    # Panel (a): pool over training cells -> mean premium at each test cell
    H_abs = np.full((len(test_lams), len(test_kaps)), np.nan)
    for i, lt in enumerate(test_lams):
        for j, kt in enumerate(test_kaps):
            vals = [_f(r["premium"]) for r in rows
                    if _f(r["lam_test_mult"]) == lt
                    and _f(r["kap_test_mult"]) == kt
                    and np.isfinite(_f(r["premium"]))]
            if vals:
                H_abs[i, j] = float(np.mean(vals))

    # Panel (b): for each test cell, average premium when train==test minus
    # mean premium across all other training cells (cost of misspecification).
    # We compute: matched_diag[i,j] = premium when (lam_train, kap_train) = (test_i, test_j)
    # then gap[i,j] = matched_diag[i,j] - mean_off_diag premium at (i,j)
    matched = np.full((len(test_lams), len(test_kaps)), np.nan)
    mean_off = np.full((len(test_lams), len(test_kaps)), np.nan)
    for i, lt in enumerate(test_lams):
        for j, kt in enumerate(test_kaps):
            on = [_f(r["premium"]) for r in rows
                  if _f(r["lam_test_mult"]) == lt
                  and _f(r["kap_test_mult"]) == kt
                  and _f(r["lam_train_mult"]) == lt
                  and _f(r["kap_train_mult"]) == kt
                  and np.isfinite(_f(r["premium"]))]
            if on:
                matched[i, j] = float(np.mean(on))
            off = [_f(r["premium"]) for r in rows
                   if _f(r["lam_test_mult"]) == lt
                   and _f(r["kap_test_mult"]) == kt
                   and not (_f(r["lam_train_mult"]) == lt
                            and _f(r["kap_train_mult"]) == kt)
                   and np.isfinite(_f(r["premium"]))]
            if off:
                mean_off[i, j] = float(np.mean(off))
    H_gap = matched - mean_off

    # Plot on cell-index coordinates so labels sit cleanly in cell centres
    # (avoids the axis-tick overlap that occurs with the multiplier-extent).
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    n_l = len(test_lams)
    n_k = len(test_kaps)
    title_fs = 20
    label_fs = 18
    tick_fs = 16
    annot_fs = 14
    cbar_fs = 16

    def _text_color(im, value):
        r, g, b, _ = im.cmap(im.norm(value))
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "black" if luminance > 0.58 else "white"

    def _annotate(ax, im, H, fmt):
        for i in range(n_l):
            for j in range(n_k):
                if not np.isnan(H[i, j]):
                    ax.text(j, i, fmt.format(H[i, j]),
                            ha="center", va="center", fontsize=annot_fs,
                            color=_text_color(im, H[i, j]))

    ax = axes[0]
    im0 = ax.imshow(H_abs, aspect="auto", origin="lower", cmap="RdYlGn")
    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f"{k:g}" for k in test_kaps])
    ax.set_yticks(range(n_l))
    ax.set_yticklabels([f"{l:g}" for l in test_lams])
    ax.tick_params(labelsize=tick_fs)
    ax.set_xlabel(r"$\kappa$ test multiplier", fontsize=label_fs)
    ax.set_ylabel(r"$\lambda$ test multiplier", fontsize=label_fs)
    ax.set_title("(a) Mean premium", fontsize=title_fs)
    cbar0 = fig.colorbar(im0, ax=ax, shrink=0.85)
    cbar0.set_label("premium vs TWAP", fontsize=cbar_fs)
    cbar0.ax.tick_params(labelsize=tick_fs)
    _annotate(ax, im0, H_abs, "{:.2f}")

    ax = axes[1]
    vmax = float(np.nanmax(np.abs(H_gap))) if np.isfinite(np.nanmax(np.abs(H_gap))) else 0.05
    im1 = ax.imshow(H_gap, aspect="auto", origin="lower", cmap="RdYlGn",
                     vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f"{k:g}" for k in test_kaps])
    ax.set_yticks(range(n_l))
    ax.set_yticklabels([f"{l:g}" for l in test_lams])
    ax.tick_params(labelsize=tick_fs)
    ax.set_xlabel(r"$\kappa$ test multiplier", fontsize=label_fs)
    ax.set_ylabel(r"$\lambda$ test multiplier", fontsize=label_fs)
    ax.set_title("(b) Matched minus off-diagonal", fontsize=title_fs)
    cbar1 = fig.colorbar(im1, ax=ax, shrink=0.85)
    cbar1.set_label("gap (matched - off-diag)", fontsize=cbar_fs)
    cbar1.ax.tick_params(labelsize=tick_fs)
    _annotate(ax, im1, H_gap, "{:+.3f}")

    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(w_pad=2.4)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  wrote {fig_path}", flush=True)


def render_regime_ii_pareto(exp1_csv, fig_path):
    """Regime-II endpoint multi-metric view.

    The plot uses premium and clearance as the visible axes, colour for value
    RMSE, and marker area for wall-clock time. This makes the Regime-II
    cash-premium/accuracy/completion trade-off visible without relying on one
    scalar ranking.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    apply_style()
    rows = [r for r in _read_rows(exp1_csv) if int(_f(r["n"])) == 1_000_000]
    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r["agent"]].append(r)

    nice = {
        "A_tabular": "A Tabular",
        "B_ddqn": "B DDQN",
        "C_hybrid": "C Hybrid",
        "D_plugin": "D Plug-in",
    }
    markers = {"A_tabular": "o", "B_ddqn": "s", "C_hybrid": "^", "D_plugin": "D"}
    order = ["A_tabular", "B_ddqn", "C_hybrid", "D_plugin"]

    records = []
    for ag in order:
        rs = by_agent.get(ag, [])
        if not rs:
            continue
        rec = {
            "agent": ag,
            "premium": float(np.mean([_f(r["premium_vs_twap"]) for r in rs])),
            "premium_sd": float(np.std([_f(r["premium_vs_twap"]) for r in rs])),
            "clearance": float(np.mean([_f(r["clearance_prob"]) for r in rs])),
            "rmse": float(np.mean([_f(r["value_err_rmse"]) for r in rs])),
            "wall": float(np.mean([_f(r["wall_clock_sec"]) for r in rs])),
        }
        records.append(rec)

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    rmses = [r["rmse"] for r in records]
    norm = LogNorm(vmin=max(min(rmses), 1e-3), vmax=max(rmses))
    cmap = plt.get_cmap("viridis_r")

    for rec in records:
        size = 90 + 70 * np.log10(1.0 + rec["wall"])
        ax.errorbar(
            rec["premium"],
            rec["clearance"],
            xerr=rec["premium_sd"],
            fmt="none",
            ecolor="0.55",
            elinewidth=1.1,
            capsize=3,
            zorder=1,
        )
        ax.scatter(
            rec["premium"],
            rec["clearance"],
            s=size,
            marker=markers.get(rec["agent"], "o"),
            c=[rec["rmse"]],
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.8,
            zorder=2,
        )
        dx = 0.015 if rec["agent"] != "D_plugin" else -0.16
        dy = 0.035 if rec["agent"] != "A_tabular" else -0.065
        ax.text(
            rec["premium"] + dx,
            rec["clearance"] + dy,
            nice.get(rec["agent"], rec["agent"]),
            fontsize=10,
        )

    ax.set_xlabel("Premium vs TWAP (dollars per 100-share episode)", fontsize=12)
    ax.set_ylabel(r"Pre-terminal clearance $\mathbb{P}[Q_T=0]$", fontsize=12)
    ax.set_xlim(0.25, 1.75)
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        "higher cash premium",
        xy=(1.62, 0.05),
        xytext=(1.18, 0.05),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=9,
        va="center",
    )
    ax.annotate(
        "higher completion",
        xy=(0.34, 0.96),
        xytext=(0.34, 0.55),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=9,
        rotation=90,
        ha="center",
        va="center",
    )
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
        shrink=0.9,
    )
    cbar.set_label(r"value RMSE vs FD reference (log scale)", fontsize=11)
    ax.text(
        0.27,
        0.02,
        "marker area scales with wall-clock time",
        fontsize=9,
        color="0.35",
    )

    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  wrote {fig_path}", flush=True)


def render_table5(exp1_other_csv, exp1_hybrid_csv, out_path):
    """Tab 5: 4-row agent comparison."""
    rows_other = _read_rows(exp1_other_csv)
    rows_hybrid = _read_rows(exp1_hybrid_csv)
    by_agent = defaultdict(lambda: defaultdict(list))
    for r in rows_other:
        if r["agent"] == "C_hybrid":
            continue
        by_agent[r["agent"]][int(r["n"])].append(r)
    for r in rows_hybrid:
        by_agent[r["agent"]][int(r["n"])].append(r)

    nice = {"A_tabular": "A. Tabular Q",
             "B_ddqn": "B. DDQN",
             "C_hybrid": "C. Hybrid (PPO)",
             "D_plugin": "D. Plug-in MLE"}
    lines = ["| Agent | val_err RMSE @ n=10⁴ | premium @ n=10⁴ | premium @ n=10⁵ | wall @ n=10⁵ (s) |",
             "|---|---|---|---|---|"]
    for ag in ["A_tabular", "B_ddqn", "C_hybrid", "D_plugin"]:
        ns = sorted(by_agent[ag].keys())
        if not ns:
            continue
        n10k = min(ns, key=lambda x: abs(x - 10000))
        n100k = min(ns, key=lambda x: abs(x - 100000))
        ve = float(np.mean([_f(r["value_err_rmse"]) for r in by_agent[ag][n10k]
                             if np.isfinite(_f(r["value_err_rmse"]))]))
        prem10 = [_f(r["premium_vs_twap"]) for r in by_agent[ag][n10k]
                   if np.isfinite(_f(r["premium_vs_twap"]))]
        prem100 = [_f(r["premium_vs_twap"]) for r in by_agent[ag][n100k]
                    if np.isfinite(_f(r["premium_vs_twap"]))]
        wall = float(np.mean([_f(r["wall_clock_sec"]) for r in by_agent[ag][n100k]]))
        lines.append(
            f"| {nice.get(ag, ag)} | {ve:.4f} | "
            f"{np.mean(prem10):+.4f} ± {np.std(prem10):.4f} | "
            f"{np.mean(prem100):+.4f} ± {np.std(prem100):.4f} | "
            f"{wall:.1f} |"
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path}", flush=True)


def main():
    print("Rendering paper-grade figures and tables ...")
    # Reuse existing CIR renderer
    from src.experiments.exp2_stochastic_lambda import make_figures as exp2_figs
    exp2_figs(out_dir="data/exp2_full", fig_dir="figures")
    print("  wrote figures/fig11_h_surface_cir.pdf, fig12_mo_trigger_cir.pdf, fig_cir_validation.pdf",
          flush=True)

    render_fig9(
        exp1_other_csv="data/exp1/sample_complexity_full.csv",
        exp1_hybrid_csv="data/exp1/sample_complexity_hybrid_v3.csv",
        fig_path="figures/fig9_sample_complexity.pdf",
    )
    render_fig10(
        exp4_csv="data/exp4/failure_modes_full.csv",
        fig_path="figures/fig10_failure_modes.pdf",
    )
    render_fig_univariate(
        exp3_dir="data/exp3_full",
        fig_path="figures/fig_univariate_sensitivity.pdf",
    )
    # Fig 15 (bivariate sensitivity) intentionally removed from the paper:
    # the (kappa, phi) and (xi, alpha) sweeps degenerate to univariate
    # patterns (no horizontal variation across phi or alpha), so the
    # heatmaps duplicate the univariate figure with no added information.
    render_fig16_misspec(
        exp3_dir="data/exp3_full",
        fig_path="figures/fig16_robustness_misspec.pdf",
    )
    render_regime_ii_pareto(
        exp1_csv="data/exp1/regime_ii_sample_complexity_full.csv",
        fig_path="figures/fig_regime_ii_pareto.pdf",
    )
    render_table5(
        exp1_other_csv="data/exp1/sample_complexity_full.csv",
        exp1_hybrid_csv="data/exp1/sample_complexity_hybrid_v3.csv",
        out_path="results/tab5_agent_comparison.md",
    )


if __name__ == "__main__":
    main()
