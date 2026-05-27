"""CIR-type fill-intensity parameters for the AMF revision §6.

    d lam_t = kappa_lam (theta_lam - lam_t) dt + sigma_lam sqrt(lam_t) dW_t

Feller condition for strict positivity:
    2 kappa_lam theta_lam >= sigma_lam^2.

`TASK5_CIR` is the baseline calibration used by Experiment 2 and the
CIR-aware RL env (`state_mode='tql'`).
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CIRParams:
    kappa_lam: float        # mean-reversion speed
    theta_lam: float        # long-run mean
    sigma_lam: float        # vol of vol
    lam0: float             # initial lambda
    feller_ok: bool = field(init=False)

    def __post_init__(self):
        # frozen dataclass: must use object.__setattr__
        ok = 2.0 * self.kappa_lam * self.theta_lam >= self.sigma_lam ** 2
        object.__setattr__(self, "feller_ok", bool(ok))


# Baseline calibration: same mean as TASK2.lam (50/60 per second), modest vol.
TASK5_CIR = CIRParams(
    kappa_lam=2.0,
    theta_lam=50.0 / 60.0,
    sigma_lam=0.1,
    lam0=50.0 / 60.0,
)
# 2 * 2 * 50/60 = 3.33 > 0.01 = sigma^2  -> Feller satisfied.
