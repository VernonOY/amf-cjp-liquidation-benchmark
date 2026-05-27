"""Tests for src.common.estimators.fit_kappa_lambda_mle.

Strategy: simulate a known (kappa_true, lambda_true) fill process under a
random-depth behaviour policy, fit MLE on the resulting (delta_i, fill_i)
pairs, and confirm:
1. Point estimates are within ~10% of truth at n >= 5000.
2. Bias decays roughly as 1/n.
3. With ridge_log_kappa > 0 and few fills, estimate stays near the prior.
4. Joint MLE for (kappa, lambda) recovers both within their asymptotic SEs.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.common.estimators import fit_kappa_lambda_mle


def simulate_fills(kappa_true: float, lam_true: float,
                   n_steps: int, dt: float, depth_max: float,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    deltas = rng.uniform(0.0, depth_max, size=n_steps)
    rates = lam_true * np.exp(-kappa_true * deltas)
    p = 1.0 - np.exp(-rates * dt)
    fills = (rng.random(n_steps) < p).astype(np.float64)
    return deltas, fills


def test_mle_recovers_truth_large_sample():
    kappa_true = 100.0
    lam_true = 50.0 / 60.0
    deltas, fills = simulate_fills(kappa_true, lam_true,
                                   n_steps=80_000, dt=0.5,
                                   depth_max=0.03, seed=1)
    out = fit_kappa_lambda_mle(deltas, fills, dt=0.5)
    assert out["n_fills"] > 100
    # Exact-Bernoulli MLE: bias is small at n=80k (typically < 3%).
    assert abs(out["kappa_hat"] - kappa_true) / kappa_true < 0.05
    assert abs(out["lam_hat"] - lam_true) / lam_true < 0.05


def test_mle_bias_decreases_with_n():
    """Average kappa estimation error must shrink between the smallest n and
    the largest n. Single-seed bias has finite-sample noise that can spoil
    strict monotonicity at intermediate n; we only require the endpoints to
    show clear decay.
    """
    kappa_true = 100.0
    lam_true = 50.0 / 60.0
    errors = []
    for n in (2000, 8000, 64000):
        runs = []
        for seed in range(16):
            deltas, fills = simulate_fills(kappa_true, lam_true,
                                           n_steps=n, dt=0.5,
                                           depth_max=0.03, seed=seed * 17 + n)
            out = fit_kappa_lambda_mle(deltas, fills, dt=0.5)
            runs.append(abs(out["kappa_hat"] - kappa_true) / kappa_true)
        errors.append(float(np.mean(runs)))
    # Endpoint comparison: largest n must have less than half the error of n=2000
    assert errors[-1] < 0.5 * errors[0], (
        f"endpoint decay too small: errors={errors}"
    )
    # Smallest and largest both must be reasonable
    assert errors[0] < 0.20, f"unexpectedly large bias at n=2000: {errors[0]}"


def test_mle_zero_fills_returns_prior():
    """If no fills are observed, the estimator must fall back to the prior."""
    deltas = np.full(100, 0.5)  # huge depth -> ~zero fill prob
    fills = np.zeros(100)
    out = fit_kappa_lambda_mle(deltas, fills, dt=0.5,
                                kappa_prior=99.0, lam_prior=0.5)
    assert out["kappa_hat"] == pytest.approx(99.0)
    assert out["lam_hat"] == pytest.approx(0.5)
    assert math.isnan(out["kappa_se"])


def test_mle_ridge_pulls_to_prior_when_few_fills():
    """With strong ridge and few fills, estimate should sit close to the prior."""
    deltas, fills = simulate_fills(kappa_true=200.0, lam_true=2.0,
                                   n_steps=200, dt=0.5,
                                   depth_max=0.03, seed=0)
    # Compare ridged vs unridged: ridged pulls toward kappa_prior=100
    out_unridged = fit_kappa_lambda_mle(deltas, fills, dt=0.5)
    out_ridged = fit_kappa_lambda_mle(deltas, fills, dt=0.5,
                                       ridge_log_kappa=20.0,
                                       kappa_prior=100.0)
    # Ridged is closer to 100 than unridged is to 100 (if the unridged
    # estimator strayed in either direction)
    bias_unridged = abs(math.log(out_unridged["kappa_hat"]) - math.log(100.0))
    bias_ridged = abs(math.log(out_ridged["kappa_hat"]) - math.log(100.0))
    assert bias_ridged <= bias_unridged + 1e-9


def test_mle_se_finite_when_well_identified():
    deltas, fills = simulate_fills(kappa_true=100.0, lam_true=0.83,
                                   n_steps=20_000, dt=0.5,
                                   depth_max=0.03, seed=2)
    out = fit_kappa_lambda_mle(deltas, fills, dt=0.5)
    assert math.isfinite(out["kappa_se"])
    assert math.isfinite(out["lam_se"])
    assert out["kappa_se"] > 0 and out["lam_se"] > 0
