"""Render seed-level pairwise endpoint comparison tables.

The experiment CSVs store one aggregate row per training seed, not path-level
terminal wealth. Therefore this script reports paired seed-level endpoint
comparisons rather than a path-level Diebold-Mariano matrix.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.common.stats import bootstrap_ci


AGENT_ORDER = ("A_tabular", "B_ddqn", "C_hybrid", "D_plugin")
AGENT_LABELS = {
    "A_tabular": "A Tabular",
    "B_ddqn": "B DDQN",
    "C_hybrid": "C Hybrid",
    "D_plugin": "D Plug-in",
}


def _read_endpoint(
    csv_paths: str | list[str] | tuple[str, ...],
    n: int,
    *,
    skip_hybrid_in_first: bool = False,
) -> dict[str, dict[int, float]]:
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    out: dict[str, dict[int, float]] = {a: {} for a in AGENT_ORDER}
    for path_idx, csv_path in enumerate(csv_paths):
        with open(csv_path, "r") as fp:
            for row in csv.DictReader(fp):
                if int(row["n"]) != int(n):
                    continue
                ag = row["agent"]
                if ag not in out:
                    continue
                if skip_hybrid_in_first and path_idx == 0 and ag == "C_hybrid":
                    continue
                out[ag][int(row["seed"])] = float(row["premium_vs_twap"])
    return out


def _paired_diff(data: dict[str, dict[int, float]], a: str, b: str) -> np.ndarray:
    seeds = sorted(set(data[a]) & set(data[b]))
    if not seeds:
        raise ValueError(f"no overlapping seeds for {a} vs {b}")
    return np.array([data[a][s] - data[b][s] for s in seeds], dtype=float)


def _normal_pvalue(diff: np.ndarray) -> float:
    try:
        from scipy.stats import t as student_t
    except ImportError:
        return float("nan")
    n = diff.size
    if n < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd <= 0:
        return 0.0 if abs(float(np.mean(diff))) > 0 else 1.0
    stat = float(np.mean(diff)) / (sd / np.sqrt(n))
    return float(2.0 * (1.0 - student_t.cdf(abs(stat), df=n - 1)))


def _holm_adjust(pvals: list[float]) -> list[float]:
    p = np.array(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adjusted_sorted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        raw = (n - rank) * p[idx]
        running = max(running, raw)
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(x) for x in adjusted]


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "---"
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}"


def render_pairwise_endpoint_table(
    csv_path: str | list[str] | tuple[str, ...],
    *,
    n: int,
    out_tex: str,
    skip_hybrid_in_first: bool = False,
    seed: int = 20260522,
) -> None:
    data = _read_endpoint(csv_path, n=n, skip_hybrid_in_first=skip_hybrid_in_first)
    pairs: list[tuple[str, str, np.ndarray, float]] = []
    raw_p = []
    for i, a in enumerate(AGENT_ORDER):
        for b in AGENT_ORDER[i + 1:]:
            diff = _paired_diff(data, a, b)
            pval = _normal_pvalue(diff)
            pairs.append((a, b, diff, pval))
            raw_p.append(pval)
    adj_p = _holm_adjust(raw_p)

    lines = [
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Comparison & Mean diff. & 95\\% CI & $p$ & Holm $p$ \\\\",
        "\\midrule",
    ]
    for (a, b, diff, pval), padj in zip(pairs, adj_p):
        point, lo, hi = bootstrap_ci(diff, n_boot=10_000, seed=seed)
        lines.append(
            f"{AGENT_LABELS[a]} $-$ {AGENT_LABELS[b]} & "
            f"{point:+.3f} & $[{lo:+.3f},{hi:+.3f}]$ & "
            f"{_fmt_p(pval)} & {_fmt_p(padj)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    Path(out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(out_tex).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_tex}")


def main() -> None:
    render_pairwise_endpoint_table(
        "data/exp1/regime_ii_sample_complexity_full.csv",
        n=1_000_000,
        out_tex="paper/tables/pairwise_regime_ii_premium.tex",
    )
    render_pairwise_endpoint_table(
        (
            "data/exp1/sample_complexity_full.csv",
            "data/exp1/sample_complexity_hybrid_v3.csv",
        ),
        n=100_000,
        out_tex="paper/tables/pairwise_regime_i_premium.tex",
        skip_hybrid_in_first=True,
    )


if __name__ == "__main__":
    main()
