"""Analytic closed-form solutions for q = 1, 2 of the LO+MO QVI
(Cartea, Jaimungal & Penalva 2015, §8.4).

Using h(t, q) = (1 / kappa) log omega(t, q), the QVI becomes
    max{ d_t omega - kappa phi q^2 omega + lam_tilde omega(t, q-1) ;
         e^{-kappa xi} omega(t, q-1) - omega(t, q) } = 0,
with omega(t, 0) = 1 and the face-lifted terminal
    omega(T^-, q) = e^{-kappa q xi} omega(T^-, q-1).

For q = 1 in the continuation regime (phi <= lam_tilde e^{kappa xi} / kappa):
    omega(t, 1) = e^{-kappa xi} e^{-kappa phi (T - t)}
                + lam_tilde (1 - e^{-kappa phi (T - t)}) / (kappa phi).  (8.27)

For q = 2 a critical time tau_2 splits continuation from execution; the
formula extends (8.27) by an integral term involving omega(., 1).

This module was migrated from `src.task2_lo_mo.analytic_q12` as part of the
AMF revision (Phase 1). Formulae are locked.
"""
from __future__ import annotations
import math

import numpy as np

from ..common.params import Params


def omega_q1(p: Params, t):
    """Closed-form omega(t, 1) per (8.27), continuation regime."""
    kappa, phi, xi = p.kappa, p.phi, p.xi
    lam_tilde = p.lam_tilde
    tau = np.maximum(p.T - np.asarray(t), 0.0)
    if phi == 0.0:
        return np.exp(-kappa * xi) + lam_tilde * tau
    return (np.exp(-kappa * xi) * np.exp(-kappa * phi * tau)
            + lam_tilde * (1.0 - np.exp(-kappa * phi * tau)) / (kappa * phi))


def critical_time_q2(p: Params) -> float | None:
    """Solve omega(tau_2, 1) = lam_tilde e^{kappa xi} /
                              (lam_tilde - 3 kappa phi e^{-kappa xi})
    for tau_2. Returns None if the continuation regime fails for q=2.
    """
    kappa, phi, xi = p.kappa, p.phi, p.xi
    lam_tilde = p.lam_tilde
    threshold = lam_tilde * math.exp(kappa * xi) / (3.0 * kappa)
    if phi >= threshold:
        return None
    denom = lam_tilde - 3.0 * kappa * phi * math.exp(-kappa * xi)
    target = lam_tilde * math.exp(-kappa * xi) / denom

    lo, hi = 0.0, p.T
    if omega_q1(p, hi) > target:
        return p.T
    if omega_q1(p, lo) < target:
        return 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if omega_q1(p, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def omega_q2(p: Params, t):
    """Analytic omega(t, 2)."""
    kappa, phi, xi = p.kappa, p.phi, p.xi
    lam_tilde = p.lam_tilde
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty_like(t_arr)

    tau2 = critical_time_q2(p)
    if tau2 is None:
        out[:] = math.exp(-2.0 * kappa * (xi + p.alpha))
        return out if out.size > 1 else float(out[0])

    ome1_tau2 = float(omega_q1(p, tau2))
    Upsilon = math.exp(-kappa * xi) * ome1_tau2

    for i, ti in enumerate(t_arr):
        if ti < tau2:
            us = np.linspace(ti, tau2, 200)
            g1_us = omega_q1(p, us)
            factor = np.exp(4.0 * kappa * phi * (ti - us))
            integral = np.trapz(factor * g1_us, us)
            out[i] = (Upsilon * math.exp(4.0 * kappa * phi * (ti - tau2))
                      + lam_tilde * integral)
        else:
            out[i] = math.exp(-kappa * xi) * float(omega_q1(p, ti))

    return out if out.size > 1 else float(out[0])


def depth_q(p: Params, t, q: int):
    """delta*(t, q) = (1 / kappa) * (1 + log(omega(t, q) / omega(t, q-1)))."""
    if q == 1:
        num = omega_q1(p, t); den = np.ones_like(np.asarray(t, dtype=float))
    elif q == 2:
        num = omega_q2(p, t); den = omega_q1(p, t)
    else:
        raise ValueError("analytic available only for q in {1, 2}")
    return (1.0 / p.kappa) * (1.0 + np.log(np.asarray(num) / np.asarray(den)))
