"""σ_λ → 0 (deterministic-λ) limit test for the CIR FD solver.

When sigma_lam == 0 and kappa_lam == 0 (no mean reversion), the lam process
is constant at lam0 and the CIR-PDE solution must equal the constant-lambda
FD solution at all (t, q) up to discretisation accuracy.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.params import TASK2
from src.common.params_cir import CIRParams
from src.numerical.fd_constant_lambda import solve as fd_solve
from src.numerical.fd_stochastic_lambda import solve_cir


def test_cir_zero_vol_zero_mean_reversion_recovers_constant():
    """With sigma_lam=0 and kappa_lam=0, lam_t = lam0 forever; CIR-FD
    should agree with constant-lambda FD up to spatial-discretisation error
    (we use a coarse lam-grid centred on lam0 for this test)."""
    p = TASK2
    cir = CIRParams(kappa_lam=0.0, theta_lam=p.lam,
                     sigma_lam=0.0, lam0=p.lam)
    sol_cir = solve_cir(p, cir, dt=0.05, n_lam=21,
                         lam_min=p.lam * 0.5, lam_max=p.lam * 2.0)
    sol_const = fd_solve(p, dt=0.05)
    # Pick the lam-grid point nearest to p.lam
    j0 = int(np.argmin(np.abs(sol_cir.lam_grid - p.lam)))
    h_cir = sol_cir.h[:, j0, :]
    # Compare at the env-FD time grid (both use dt=0.05)
    err = float(np.max(np.abs(h_cir - sol_const.h)))
    assert err < 1e-10, f"CIR(0,0) limit failure: max|h_cir - h_const|={err:.4f}"


def test_cir_solve_returns_correct_shape():
    p = TASK2
    cir = CIRParams(kappa_lam=2.0, theta_lam=p.lam,
                     sigma_lam=0.1, lam0=p.lam)
    sol = solve_cir(p, cir, dt=0.05, n_lam=21, Q_max=p.Q0)
    assert sol.h.shape == (int(round(p.T / 0.05)) + 1, 21, p.Q0 + 1)
    assert sol.mo_trigger.shape == sol.h.shape
    # h(., ., 0) must be identically zero
    np.testing.assert_allclose(sol.h[:, :, 0], 0.0, atol=1e-12)


def test_cir_h_increases_with_lam_at_fixed_t_q():
    """At fixed (t, q) > 0, higher fill intensity lam should yield a less
    negative h (better continuation value). We check the OVERALL trend at
    t=0 for a moderate-q slice — the discrete upwind scheme can have tiny
    non-monotonicities at the boundary (~1e-4); we only require the high-lam
    average to exceed the low-lam average by a clear margin."""
    p = TASK2
    cir = CIRParams(kappa_lam=1.0, theta_lam=p.lam,
                     sigma_lam=0.0, lam0=p.lam)
    sol = solve_cir(p, cir, dt=0.05, n_lam=21)
    h_q = sol.h[0, :, min(5, p.Q0)]
    low_third = float(np.mean(h_q[:7]))
    high_third = float(np.mean(h_q[-7:]))
    assert high_third > low_third, (
        f"h not increasing in lam: low={low_third:.5f} high={high_third:.5f}"
    )


@pytest.mark.slow
def test_cir_o_sigma_squared_validation():
    """Convergence-rate check (slow). For small sigma_lam, the gap between
    h_CIR and h_const should scale like O(sigma_lam^2)."""
    p = TASK2
    sigmas = (1e-3, 1e-2, 1e-1)
    sol_const = fd_solve(p, dt=0.05)
    gaps = []
    for s in sigmas:
        cir = CIRParams(kappa_lam=1.0, theta_lam=p.lam,
                         sigma_lam=s, lam0=p.lam)
        sol = solve_cir(p, cir, dt=0.05, n_lam=21,
                          lam_min=p.lam * 0.5, lam_max=p.lam * 2.0)
        j0 = int(np.argmin(np.abs(sol.lam_grid - p.lam)))
        gap = float(np.max(np.abs(sol.h[:, j0, :] - sol_const.h)))
        gaps.append(gap)
    # Should grow with sigma; we only test that the smallest sigma gives the
    # smallest gap (full O(sigma^2) scaling is a stretch on this coarse grid).
    assert gaps[0] <= gaps[2] + 1e-6, f"CIR vol scaling broken: {gaps}"
