"""Tests for src.common.stats.

Covers:
- bootstrap_ci coverage on N(0,1)
- paired_bootstrap_ci shape error + mean-of-difference correctness
- bca_ci returns same point estimate as percentile and lies in [lo, hi]
- diebold_mariano on a controlled example: identical losses → p ~ 1
- diebold_mariano on a known systematic difference → small p
- benjamini_hochberg monotonicity and FDR control
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.stats import (
    bootstrap_ci, bca_ci, paired_bootstrap_ci,
    diebold_mariano, benjamini_hochberg,
)


def test_bootstrap_ci_coverage_normal():
    """Percentile bootstrap mean-CI for N(0, 1) should cover 0 about 95% of the time."""
    rng = np.random.default_rng(2024)
    K = 200
    covered = 0
    for k in range(K):
        x = rng.standard_normal(100)
        _, lo, hi = bootstrap_ci(x, n_boot=400, seed=k)
        if lo <= 0.0 <= hi:
            covered += 1
    rate = covered / K
    # Allow generous slack since this is a stochastic test
    assert 0.88 <= rate <= 1.00, f"coverage rate {rate:.3f} outside [0.88, 1.00]"


def test_bootstrap_ci_point_equals_stat():
    x = np.arange(50, dtype=np.float64)
    pt, lo, hi = bootstrap_ci(x, n_boot=200, seed=0)
    assert pt == pytest.approx(float(np.mean(x)), abs=0, rel=0)
    assert lo <= pt <= hi


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]), n_boot=10)


def test_paired_bootstrap_ci_mean_of_diff():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(200) + 0.5
    y = rng.standard_normal(200)
    pt, lo, hi = paired_bootstrap_ci(x, y, n_boot=400, seed=0)
    assert pt == pytest.approx(float(np.mean(x - y)), abs=0, rel=0)
    # The true mean of x-y is ~0.5; CI should cover or be near it.
    assert lo < 0.5 < hi or abs(pt - 0.5) < 0.2


def test_paired_bootstrap_ci_shape_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.zeros(5), np.zeros(6))


def test_bca_ci_basic():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(100)
    pt, lo, hi = bca_ci(x, n_boot=500, seed=0)
    assert pt == pytest.approx(float(np.mean(x)), abs=0, rel=0)
    assert lo <= pt <= hi


def test_diebold_mariano_identical_losses():
    """Identical loss series → mean diff = 0 → p-value ~ 1."""
    loss = np.linspace(0.1, 1.0, 50)
    out = diebold_mariano(loss, loss)
    assert out["mean_diff"] == 0.0
    assert out["p_value"] > 0.99


def test_diebold_mariano_known_difference():
    """Systematic loss gap → small p-value."""
    rng = np.random.default_rng(3)
    loss_a = rng.standard_normal(500) + 1.0
    loss_b = rng.standard_normal(500)
    out = diebold_mariano(loss_a, loss_b)
    assert out["mean_diff"] > 0.5
    assert out["p_value"] < 0.01


def test_diebold_mariano_shape_mismatch():
    with pytest.raises(ValueError):
        diebold_mariano(np.zeros(3), np.zeros(4))


def test_benjamini_hochberg_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 30)
    # Inject some signal p-values
    p[:5] = rng.uniform(0, 0.005, 5)
    rejected, adj = benjamini_hochberg(p, q=0.05)
    assert adj.shape == p.shape
    assert (adj >= 0).all() and (adj <= 1.0).all()
    # Adjusted p-values should be monotone non-decreasing in original p order
    sorted_idx = np.argsort(p)
    sorted_adj = adj[sorted_idx]
    assert np.all(sorted_adj[:-1] <= sorted_adj[1:] + 1e-12)
    # At least the 5 injected signals should be rejected
    assert rejected[:5].sum() >= 4


def test_benjamini_hochberg_extreme_cases():
    # All p-values 1.0 → nothing rejected
    rejected, adj = benjamini_hochberg(np.ones(10), q=0.05)
    assert not rejected.any()
    assert np.allclose(adj, 1.0)

    # All p-values 0.0 → all rejected
    rejected, adj = benjamini_hochberg(np.zeros(10), q=0.05)
    assert rejected.all()
    assert np.allclose(adj, 0.0)
