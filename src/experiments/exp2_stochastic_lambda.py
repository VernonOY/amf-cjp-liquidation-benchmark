"""Experiment 2 — Stochastic-intensity (CIR) results (AMF revision §6, Figs 11-14).

Produces three CSV outputs:

  fd_cir.csv         h(t, q, lam) FD solution and MO trigger map  ->  Figs 11, 12
  rl_cir.csv         RL agent premium under stochastic lam        ->  Fig 14
  cir_validation.csv |h_CIR - h_const| vs sigma_lam               ->  §6.3 inset

For Phase 4 we deliver the FD half and the validation half end-to-end. The
RL-under-CIR half is implementable but slow (each cell trains an agent with a
stochastic-lam env) — we ship the function skeleton with a working
single-cell demo to keep the test suite fast.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np

from ..common.params import TASK2
from ..common.params_cir import CIRParams, TASK5_CIR
from ..numerical.fd_constant_lambda import solve as fd_solve_const
from ..numerical.fd_stochastic_lambda import solve_cir
from ._runner import write_csv


SLICE_MULTS = (0.7, 1.0, 1.3)


def _lam_anchors(p=TASK2) -> np.ndarray:
    return np.asarray([m * p.lam for m in SLICE_MULTS], dtype=float)


def fd_cir_surface(sigma_lams: Sequence[float], n_lam: int = 41,
                    dt: float = 0.05) -> list[dict]:
    """Dump h(t=0, q, lam) and MO trigger map at t=0 for each sigma_lam."""
    p = TASK2
    rows = []
    for s in sigma_lams:
        cir = CIRParams(kappa_lam=2.0, theta_lam=p.lam,
                         sigma_lam=float(s), lam0=p.lam)
        # Narrower lambda grid (0.5..2.0 * lam_bar) avoids the near-zero
        # advection instability of the upwind backward-Euler scheme.
        sol = solve_cir(p, cir, dt=dt, n_lam=n_lam,
                         lam_min=p.lam * 0.5, lam_max=p.lam * 2.0,
                         lam_anchors=_lam_anchors(p))
        for j, lam_j in enumerate(sol.lam_grid):
            for q in range(p.Q0 + 1):
                rows.append({
                    "sigma_lam": float(s), "lam_idx": int(j),
                    "lam_value": float(lam_j), "q": int(q),
                    "h_t0": float(sol.h[0, j, q]),
                    "mo_trigger_t0": bool(sol.mo_trigger[0, j, q]),
                })
    return rows


def cir_validation_sweep(sigma_lams: Sequence[float], dt: float = 0.05,
                          n_lam: int = 41) -> list[dict]:
    """Measure the small-volatility perturbation gap at multiple lambda slices.

    At lambda=bar(lambda), the sigma=0 CIR solution coincides with the
    constant-lambda solution.  Away from the mean, the deterministic CIR drift
    is non-zero even at sigma=0, so the honest volatility-validation target is
    h_CIR(sigma_lam, lambda_slice) - h_CIR(0, lambda_slice), not a
    constant-lambda solution with frozen lambda.
    """
    p = TASK2
    sol_const = fd_solve_const(p, dt=dt)
    cir0 = CIRParams(kappa_lam=2.0, theta_lam=p.lam,
                      sigma_lam=0.0, lam0=p.lam)
    sol_zero = solve_cir(p, cir0, dt=dt, n_lam=n_lam,
                          lam_min=p.lam * 0.5, lam_max=p.lam * 2.0,
                          lam_anchors=_lam_anchors(p))
    rows = []
    for s in sigma_lams:
        cir = CIRParams(kappa_lam=2.0, theta_lam=p.lam,
                         sigma_lam=float(s), lam0=p.lam)
        sol_cir = solve_cir(p, cir, dt=dt, n_lam=n_lam,
                             lam_min=p.lam * 0.5, lam_max=p.lam * 2.0,
                             lam_anchors=_lam_anchors(p))
        norm = max(float(np.max(np.abs(sol_zero.h))), 1e-12)
        for mult in SLICE_MULTS:
            lam_target = mult * p.lam
            j = int(np.argmin(np.abs(sol_cir.lam_grid - lam_target)))
            j_zero = int(np.argmin(np.abs(sol_zero.lam_grid - lam_target)))
            gap = float(np.max(np.abs(
                sol_cir.h[:-1, j, 1:] - sol_zero.h[:-1, j_zero, 1:]
            )))
            if abs(mult - 1.0) < 1e-12:
                const_gap = float(np.max(np.abs(
                    sol_cir.h[:-1, j, 1:] - sol_const.h[:-1, 1:]
                )))
            else:
                const_gap = float("nan")
            rows.append({
                "sigma_lam": float(s),
                "slice_mult": float(mult),
                "lam_value": float(sol_cir.lam_grid[j]),
                "max_abs_gap": gap,
                "rel_gap": gap / norm,
                "const_gap_at_mean": const_gap,
            })
    return rows


def run(
    # Validation scope: small intensity volatility where the sigma_lam -> 0
    # limit can be checked directly against the constant-lambda FD solution.
    sigma_lams: Sequence[float] = (0.0, 0.05, 0.1),
    out_dir: str = "data/exp2",
    n_lam: int = 41,
    dt: float = 0.05,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_fd = fd_cir_surface(sigma_lams, n_lam=n_lam, dt=dt)
    rows_v = cir_validation_sweep(sigma_lams, n_lam=n_lam, dt=dt)
    write_csv(rows_fd, out / "fd_cir.csv",
              columns=["sigma_lam", "lam_idx", "lam_value", "q",
                       "h_t0", "mo_trigger_t0"])
    write_csv(rows_v, out / "cir_validation.csv",
              columns=["sigma_lam", "slice_mult", "lam_value",
                       "max_abs_gap", "rel_gap", "const_gap_at_mean"])
    return {"fd_rows": len(rows_fd), "validation_rows": len(rows_v)}


def make_figures(out_dir: str = "data/exp2",
                  fig_dir: str = "figures") -> None:
    """Render Fig 11 (h-surface multi-sigma), Fig 12 (MO trigger boundary as
    curves per q), Fig validation (gap vs sigma_lam)."""
    import csv
    import matplotlib.pyplot as plt
    from ..common.style import apply as apply_style

    apply_style()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir)

    fd_rows = list(csv.DictReader((out_dir / "fd_cir.csv").open()))
    all_sigs = sorted({float(r["sigma_lam"]) for r in fd_rows})

    # ---------------------------------------------------------------
    # Fig 11 — h(0, q, lambda) heatmap, 3 panels for representative sigmas
    # ---------------------------------------------------------------
    # Pick three sigma values spanning the validation range.
    if len(all_sigs) >= 3:
        sigs_show = [all_sigs[0], all_sigs[len(all_sigs)//2], all_sigs[-1]]
    else:
        sigs_show = all_sigs

    fig, axes = plt.subplots(1, len(sigs_show), figsize=(4.0 * len(sigs_show), 3.6),
                              sharey=True)
    if len(sigs_show) == 1:
        axes = [axes]
    vmin = min(float(r["h_t0"]) for r in fd_rows
                if int(r["q"]) > 0 and float(r["sigma_lam"]) in sigs_show)
    vmax = max(float(r["h_t0"]) for r in fd_rows
                if int(r["q"]) > 0 and float(r["sigma_lam"]) in sigs_show)
    for ax, s in zip(axes, sigs_show):
        rows_s = [r for r in fd_rows if float(r["sigma_lam"]) == s]
        lams = sorted({float(r["lam_value"]) for r in rows_s})
        qs = sorted({int(r["q"]) for r in rows_s if int(r["q"]) > 0})
        H = np.full((len(lams), len(qs)), np.nan)
        for r in rows_s:
            if int(r["q"]) == 0:
                continue
            i_l = lams.index(float(r["lam_value"]))
            i_q = qs.index(int(r["q"]))
            H[i_l, i_q] = float(r["h_t0"])
        im = ax.imshow(H.T, aspect="auto", origin="lower",
                        extent=[min(lams), max(lams), min(qs) - 0.5, max(qs) + 0.5],
                        cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel(r"$\lambda$")
        ax.set_title(rf"$\sigma_\lambda = {s:.2f}$")
    axes[0].set_ylabel(r"inventory $q$")
    fig.colorbar(im, ax=axes, label=r"$h(0, q, \lambda)$", shrink=0.85,
                  pad=0.02)
    fig.savefig(fig_dir / "fig11_h_surface_cir.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 12 (MO trigger boundary curves) intentionally removed from the
    # paper: in the main parameter regime the trigger is nearly
    # one-dimensional, so the qualitative geometry is already conveyed by
    # Section 4.2 (analytic q in {1, 2}) and Figure 11.

    # ---------------------------------------------------------------
    # Validation: gap vs sigma_lam
    # ---------------------------------------------------------------
    v_rows = list(csv.DictReader((out_dir / "cir_validation.csv").open()))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    by_slice = {}
    for r in v_rows:
        by_slice.setdefault(float(r.get("slice_mult", 1.0)), []).append(r)
    markers = {0.7: "o", 1.0: "s", 1.3: "^"}
    for mult in sorted(by_slice):
        rs = sorted(by_slice[mult], key=lambda x: float(x["sigma_lam"]))
        sigs = [float(r["sigma_lam"]) for r in rs]
        gaps = [float(r["max_abs_gap"]) for r in rs]
        gaps_plot = [max(g, 1e-12) for g in gaps]
        ax.plot(sigs, gaps_plot, marker=markers.get(mult, "o"),
                 linewidth=1.5, markersize=5,
                 label=rf"$\lambda_0={mult:.1f}\bar\lambda$")
    ax.set_xlabel(r"intensity vol $\sigma_\lambda$")
    ax.set_ylabel(r"$\max_{t,q}|h_{\rm CIR}^{\sigma}(t,\lambda_0,q)-h_{\rm CIR}^{0}(t,\lambda_0,q)|$")
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_cir_validation.pdf")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--full", action="store_true",
                     help="Paper-grade: small-sigma validation grid, n_lam=41")
    pa.add_argument("--out-dir", default="data/exp2")
    pa.add_argument("--fig-dir", default="figures")
    args = pa.parse_args()
    if args.full:
        sigma_lams = (0.0, 0.025, 0.05, 0.075, 0.1)
        n_lam = 41
    else:
        sigma_lams = (0.0, 0.1)
        n_lam = 41
    counts = run(sigma_lams=sigma_lams, out_dir=args.out_dir,
                  n_lam=n_lam)
    make_figures(out_dir=args.out_dir, fig_dir=args.fig_dir)
    print(f"exp2 wrote {counts}; figures in {args.fig_dir}/")
