"""MLE estimators for the Cartea-Jaimungal-Penalva fill-process parameters.

The plug-in agent (Agent D, §5.2.4 of the AMF revision plan) and the
structure-aware hybrid policy (Agent C, §5.2.3) both depend on online
estimation of the fill rate `lambda` and the depth-decay parameter `kappa`
from observed (delta_i, fill_i, dt_i) triples.

Model: per-step Bernoulli fill probability is
    p_i = 1 - exp(-eta_i),    eta_i := lambda * exp(-kappa * delta_i) * dt_i.
We use the EXACT Bernoulli log-likelihood
    l(lambda, kappa) = sum_i [ y_i log(1 - exp(-eta_i)) - (1 - y_i) eta_i ]
rather than the leading-order Poisson approximation
    l_poisson(lambda, kappa) ~ sum_i [ y_i log(eta_i) - eta_i ],
because at typical macro-step sizes (dt ~ 0.5 s) the leading-order
approximation introduces a measurable kappa, lambda bias.

Joint MLE: lambda given kappa is found by 1-D Newton; kappa is solved on top
by 60-step bisection of the score equation d l / d kappa = 0.

A ridge prior on (log kappa, log lambda) is added for finite-sample
stabilisation. With default `ridge_log_kappa = ridge_log_lam = 0` no
regularisation is applied.
"""
from __future__ import annotations
import math

import numpy as np


__all__ = ["fit_kappa_lambda_mle"]


def fit_kappa_lambda_mle(
    deltas: np.ndarray,
    fills: np.ndarray,
    dt: float | np.ndarray,
    *,
    ridge_log_kappa: float = 0.0,
    ridge_log_lam: float = 0.0,
    kappa_prior: float = 100.0,
    lam_prior: float = 50.0 / 60.0,
    kappa_lo: float = 1.0,
    kappa_hi: float = 1e4,
    n_iter: int = 60,
) -> dict:
    """Joint MLE for (kappa, lambda) under the leading-order Poisson likelihood.

    Parameters
    ----------
    deltas : per-step posted depths (>= 0), shape (N,)
    fills  : per-step fill indicators (0 or 1), shape (N,)
    dt     : either a scalar step size or per-step array of step sizes
    ridge_log_kappa, ridge_log_lam : non-negative regularisation weights on
        squared log-deviation from the priors. Equivalent to a Gaussian prior
        on (log kappa, log lambda) with precision 2 * ridge_*.
    kappa_prior, lam_prior : prior means in the original scale.
    kappa_lo, kappa_hi : bisection bracket (must satisfy g(kappa_lo) and
        g(kappa_hi) of opposite sign for the unregularised root condition).
    n_iter : bisection iterations.

    Returns
    -------
    dict with keys: 'kappa_hat', 'lam_hat', 'kappa_se', 'lam_se',
                    'n_obs', 'n_fills'.
    """
    deltas = np.asarray(deltas, dtype=np.float64).ravel()
    fills = np.asarray(fills, dtype=np.float64).ravel()
    if deltas.shape != fills.shape:
        raise ValueError("fit_kappa_lambda_mle: deltas/fills shape mismatch")
    if np.isscalar(dt):
        dt_arr = np.full_like(deltas, float(dt))
    else:
        dt_arr = np.asarray(dt, dtype=np.float64).ravel()
        if dt_arr.shape != deltas.shape:
            raise ValueError("fit_kappa_lambda_mle: dt array shape mismatch")
    n_obs = deltas.size
    n_fills = float(fills.sum())

    if n_obs == 0:
        return {"kappa_hat": float(kappa_prior), "lam_hat": float(lam_prior),
                "kappa_se": float("nan"), "lam_se": float("nan"),
                "n_obs": 0, "n_fills": 0.0}

    if n_fills < 1.0:
        # No fills observed — likelihood is monotone in (lam, kappa) and the
        # MLE diverges. Collapse to the prior so downstream callers get a
        # well-defined estimate.
        return {"kappa_hat": float(kappa_prior), "lam_hat": float(lam_prior),
                "kappa_se": float("nan"), "lam_se": float("nan"),
                "n_obs": int(n_obs), "n_fills": 0.0}

    # Exact Bernoulli log-likelihood (we maximise it directly via scipy):
    #   eta_i = lam * w_i * dt_i,  w_i := exp(-kappa * delta_i)
    #   l(lam, kappa) = sum_i { y_i log(1 - exp(-eta_i)) - (1 - y_i) eta_i }
    # We parameterise (lam, kappa) via theta = (log lam, log kappa) so the
    # optimiser stays inside the positive orthant.
    from scipy.optimize import minimize  # local import keeps top-level light

    log_kappa_prior = math.log(kappa_prior)
    log_lam_prior = math.log(lam_prior)

    def neg_loglik(theta: np.ndarray) -> float:
        log_lam, log_kap = float(theta[0]), float(theta[1])
        lam = math.exp(log_lam)
        kap = math.exp(log_kap)
        # Clip the kappa*delta argument to avoid exp() overflow during the
        # L-BFGS line search (which can transiently visit kap ~ 1 with
        # lam ~ 1e4). eta_max=700 keeps exp(eta) below double-precision
        # overflow, well above any meaningful regime.
        kdelta = np.minimum(kap * deltas, 700.0)
        eta = lam * np.exp(-kdelta) * dt_arr
        eta = np.minimum(eta, 700.0)
        # Use np.log1p(-exp(-eta)) for the "fill" term but guard against eta=0
        eta_safe = np.maximum(eta, 1e-300)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_one_minus_exp_neg_eta = np.log(-np.expm1(-eta_safe) + 1e-300)
        ll = float((fills * log_one_minus_exp_neg_eta - (1.0 - fills) * eta).sum())
        # Ridge in log-domain
        if ridge_log_lam != 0.0:
            ll -= ridge_log_lam * (log_lam - log_lam_prior) ** 2
        if ridge_log_kappa != 0.0:
            ll -= ridge_log_kappa * (log_kap - log_kappa_prior) ** 2
        return -ll

    # Initial guess from the leading-order Poisson form
    w0 = np.exp(-kappa_prior * deltas) * dt_arr
    lam_init = max(n_fills, 1.0) / max(float(w0.sum()), 1e-12)
    theta0 = np.array([math.log(max(lam_init, 1e-6)),
                       math.log(max(kappa_prior, 1.0))], dtype=np.float64)
    bounds = [
        (math.log(1e-4), math.log(1e4)),
        (math.log(kappa_lo), math.log(kappa_hi)),
    ]
    res = minimize(neg_loglik, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-12})
    log_lam_hat, log_kap_hat = res.x
    lam_hat = float(math.exp(log_lam_hat))
    kappa_hat = float(math.exp(log_kap_hat))

    if not res.success and n_fills >= 1.0:
        # Optimiser failed but we have signal: keep the result anyway, flag SE
        # as NaN to alert downstream callers.
        lam_se = float("nan")
        kappa_se = float("nan")
    else:
        # Asymptotic SEs from the leading-order Poisson Fisher information.
        # For typical fill counts (n_fills > 100) the Poisson and Bernoulli SEs
        # agree to within ~1% and the Poisson form is closed-form.
        w_hat = np.exp(-kappa_hat * deltas) * dt_arr
        I_ll = n_fills / max(lam_hat ** 2, 1e-30)
        I_lk = float((deltas * w_hat).sum())
        I_kk = lam_hat * float((deltas ** 2 * w_hat).sum())
        det = I_ll * I_kk - I_lk * I_lk
        if det > 0:
            lam_se = float(np.sqrt(max(I_kk / det, 0.0)))
            kappa_se = float(np.sqrt(max(I_ll / det, 0.0)))
        else:
            lam_se = float("nan")
            kappa_se = float("nan")

    return {"kappa_hat": float(kappa_hat), "lam_hat": float(lam_hat),
            "kappa_se": kappa_se, "lam_se": lam_se,
            "n_obs": int(n_obs), "n_fills": float(n_fills)}
