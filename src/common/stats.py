"""Statistical utilities for the AMF revision: bootstrap CIs, Diebold-Mariano
test, Benjamini-Hochberg FDR control.

Every experiment script (exp1-exp4) and most diagnostic tables depend on
these. Two design rules:

1. Reproducibility: all functions accept a `seed` argument and use
   `np.random.default_rng(seed)` exclusively. No global RNG mutation.

2. Backward compatibility: the function signatures here are part of the
   AMF revision contract — downstream experiment scripts import them by
   keyword arguments, so signature changes are breaking and require a
   bump in `CITATION.cff`.
"""
from __future__ import annotations
from typing import Callable

import numpy as np


__all__ = [
    "bootstrap_ci",
    "bca_ci",
    "paired_bootstrap_ci",
    "diebold_mariano",
    "benjamini_hochberg",
]


# ---------------------------------------------------------------------------
# Percentile bootstrap
# ---------------------------------------------------------------------------
def bootstrap_ci(
    x: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    stat: Callable[[np.ndarray], float] = np.mean,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for `stat(x)`.

    Returns (point_estimate, lo, hi).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("bootstrap_ci: empty input")
    rng = np.random.default_rng(seed)
    n = x.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_stats = np.fromiter(
        (stat(x[i]) for i in idx), dtype=np.float64, count=n_boot
    )
    lo = float(np.quantile(boot_stats, alpha / 2.0))
    hi = float(np.quantile(boot_stats, 1.0 - alpha / 2.0))
    return float(stat(x)), lo, hi


# ---------------------------------------------------------------------------
# Bias-corrected accelerated (BCa) bootstrap — wraps scipy
# ---------------------------------------------------------------------------
def bca_ci(
    x: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    stat: Callable[[np.ndarray], float] = np.mean,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """BCa bootstrap CI for `stat(x)`. Falls back to percentile if SciPy raises.

    Returns (point_estimate, lo, hi).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    point = float(stat(x))
    try:
        from scipy.stats import bootstrap as _scipy_bootstrap
    except ImportError:
        return bootstrap_ci(x, n_boot=n_boot, alpha=alpha, stat=stat, seed=seed)
    rng = np.random.default_rng(seed)
    try:
        res = _scipy_bootstrap(
            (x,), stat,
            confidence_level=1.0 - alpha,
            n_resamples=n_boot,
            method="BCa",
            random_state=rng,
            vectorized=False,
        )
        return point, float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        return bootstrap_ci(x, n_boot=n_boot, alpha=alpha, stat=stat, seed=seed)


# ---------------------------------------------------------------------------
# Paired bootstrap (for premium vs baseline comparisons)
# ---------------------------------------------------------------------------
def paired_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Paired percentile bootstrap CI for mean(x - y).

    `x` and `y` must have identical shape and be path-aligned.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"paired_bootstrap_ci: shape mismatch {x.shape} vs {y.shape}")
    d = x - y
    return bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)


# ---------------------------------------------------------------------------
# Diebold-Mariano test (HAC variance with Newey-West lags)
# ---------------------------------------------------------------------------
def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    h: int = 1,
    two_sided: bool = True,
) -> dict:
    """Diebold-Mariano test of equal predictive accuracy.

    Compares two loss series `loss_a` and `loss_b` (path-aligned). Under H_0
    of equal accuracy, the DM statistic is asymptotically N(0,1).

    Parameters
    ----------
    loss_a, loss_b : 1D arrays of equal length, path-aligned losses
    h : forecast horizon (Newey-West uses h-1 lags)
    two_sided : if False, returns one-sided p-value testing H_1: A worse than B

    Returns
    -------
    dict with keys 'dm_stat', 'p_value', 'mean_diff', 'se'.
    """
    from scipy.stats import norm

    a = np.asarray(loss_a, dtype=np.float64).ravel()
    b = np.asarray(loss_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"diebold_mariano: shape mismatch {a.shape} vs {b.shape}")
    d = a - b
    n = d.size
    d_bar = float(np.mean(d))

    # HAC variance (Newey-West) with floor( (n)^{1/3} ) lags or h-1
    L = max(h - 1, int(np.floor(n ** (1.0 / 3.0))))
    L = min(L, n - 1)
    gamma0 = float(np.mean((d - d_bar) ** 2))
    var_d = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        gk = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        var_d += 2.0 * w * gk
    var_d = max(var_d, 1e-30)
    se = np.sqrt(var_d / n)
    dm_stat = d_bar / se
    if two_sided:
        p = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))
    else:
        p = 1.0 - norm.cdf(dm_stat)
    return {"dm_stat": float(dm_stat), "p_value": float(p),
            "mean_diff": d_bar, "se": float(se)}


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR control
# ---------------------------------------------------------------------------
def benjamini_hochberg(
    pvals: np.ndarray,
    q: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg (1995) FDR procedure.

    Parameters
    ----------
    pvals : 1D array of p-values
    q     : target FDR

    Returns
    -------
    (rejected, adjusted)
      rejected : bool mask, True where H_0 is rejected
      adjusted : BH-adjusted p-values (monotone non-decreasing in original rank)
    """
    p = np.asarray(pvals, dtype=np.float64).ravel()
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    factor = n / np.arange(1, n + 1, dtype=np.float64)
    raw = ranked * factor
    # Enforce monotonicity from the largest down to smallest rank
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adjusted = np.empty_like(adj_sorted)
    adjusted[order] = adj_sorted
    rejected = adjusted <= q
    return rejected, adjusted
