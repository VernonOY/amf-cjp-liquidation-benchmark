"""Shared model parameters (Cartea–Jaimungal–Penalva, Ch. 8)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    T: float            # horizon (seconds)
    Q0: int             # initial inventory (shares)
    lam: float          # MO arrival intensity (per second)
    kappa: float        # fill-probability decay (1/$)
    sigma: float        # midprice volatility ($/sqrt(s))
    S0: float           # initial midprice ($)
    xi: float           # half-spread ($)
    alpha: float        # terminal impact penalty ($/share^2)
    phi: float          # running inventory penalty ($^2 s / share^2)

    @property
    def lam_tilde(self) -> float:
        """lambda / e, the effective rate after maximising δ·P(δ)."""
        import math
        return self.lam / math.e


TASK1 = Params(
    T=60.0, Q0=5, lam=50.0 / 60.0, kappa=100.0,
    sigma=0.01, S0=30.0, xi=0.0, alpha=0.001, phi=0.0,
)


TASK2 = Params(
    T=60.0, Q0=10, lam=50.0 / 60.0, kappa=100.0,
    sigma=0.01, S0=30.0, xi=0.005, alpha=0.001, phi=1e-5,
)


# Regime I is the small CJP Chapter 8 LO+MO benchmark used throughout the
# first AMF revision. Regime II is the institutional-scale extension used by
# the R2 Path 1 plan: inventory and horizon are scaled by 10x, lambda is
# scaled to preserve fills-per-share over the horizon, and alpha/phi are
# rescaled to keep terminal and running inventory penalties numerically
# meaningful without overwhelming the spread economics.
REGIME_I = TASK2

REGIME_II = Params(
    T=600.0, Q0=100, lam=50.0 / 60.0, kappa=100.0,
    sigma=0.01, S0=30.0, xi=0.005, alpha=1e-5, phi=1e-6,
)
