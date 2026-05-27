"""Event-driven simulator for the Cartea–Jaimungal–Penalva LO/MO model.

The agent posts a single LO of unit size at depth delta_t. Other traders' MOs
arrive as a Poisson process with rate lam. Each arriving MO lifts the agent's
LO with probability exp(-kappa * delta_t). The agent may also execute her own
MO at any time (Task 2 only). Between events the midprice follows
dS_t = sigma dW_t.

We adopt an Euler time grid with small step dt. Within each sub-interval, the
fill probability is approximated as lam * exp(-kappa * delta) * dt (valid when
dt is small). This matches the continuous-time model to first order and is the
standard discretisation used in the book's numerical experiments.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .params import Params


@dataclass
class Path:
    t: np.ndarray        # time grid, shape (M+1,)
    S: np.ndarray        # midprice on grid, shape (M+1,)
    q: np.ndarray        # inventory at each grid point, shape (M+1,)
    delta: np.ndarray    # posted LO depth on grid, shape (M+1,)
    X: np.ndarray        # accumulated cash on grid, shape (M+1,)
    lo_fills: np.ndarray # boolean, True if an LO filled in (t_i, t_{i+1}]
    mo_sends: np.ndarray # boolean, True if agent executed an MO in that step

    @property
    def q_final(self) -> int:
        return int(self.q[-1])

    def terminal_value(self, p: Params) -> float:
        """X_T + Q_T (S_T - alpha Q_T), i.e. mark-to-market cash after
        terminal block-impact liquidation.  Agent-initiated MOs pay xi
        through the cash process before this terminal branch is reached."""
        qT = self.q[-1]
        return float(self.X[-1] + qT * (self.S[-1] - p.alpha * qT))


class Simulator:
    """Path generator common to Task 1, Task 2, and RL training."""

    def __init__(self, p: Params, dt: float = 0.05, seed: int | None = None):
        self.p = p
        self.dt = dt
        self.n_steps = int(round(p.T / dt))
        self.t_grid = np.linspace(0.0, p.T, self.n_steps + 1)
        self.rng = np.random.default_rng(seed)

    def simulate(self, policy) -> Path:
        """Simulate one path. `policy(t, q, S) -> (delta, mo_flag)`.

        delta: LO depth (>=0). Set mo_flag=True to execute a unit MO at the
        start of the step (Task 2). Task 1 policies always set mo_flag=False.
        """
        p, dt = self.p, self.dt
        M = self.n_steps
        t = self.t_grid
        S = np.empty(M + 1); S[0] = p.S0
        q = np.empty(M + 1, dtype=np.int64); q[0] = p.Q0
        delta = np.zeros(M + 1)
        X = np.zeros(M + 1)
        lo_fills = np.zeros(M + 1, dtype=bool)
        mo_sends = np.zeros(M + 1, dtype=bool)

        sqrt_dt = np.sqrt(dt)
        for i in range(M):
            d_i, mo_flag = policy(t[i], int(q[i]), S[i])
            delta[i] = d_i

            # Agent MO first (executes at S_t - xi, reduces inventory by 1).
            if mo_flag and q[i] > 0:
                X[i+1] = X[i] + (S[i] - p.xi)
                q[i+1] = q[i] - 1
                mo_sends[i] = True
            else:
                X[i+1] = X[i]
                q[i+1] = q[i]

            # Midprice diffusion over the step.
            S[i+1] = S[i] + p.sigma * sqrt_dt * self.rng.standard_normal()

            # If still holding inventory, check whether an external MO filled
            # the posted LO during (t_i, t_{i+1}].
            if q[i+1] > 0 and d_i >= 0:
                # Effective rate of LO fills = lam * P(delta).
                rate = p.lam * np.exp(-p.kappa * d_i)
                if self.rng.random() < 1.0 - np.exp(-rate * dt):
                    # LO filled at price S_{t_i} + delta (pre-jump mid + depth).
                    X[i+1] += (S[i] + d_i)
                    q[i+1] -= 1
                    lo_fills[i] = True

            # If inventory hits zero, freeze remainder of the path.
            if q[i+1] == 0:
                for j in range(i+1, M):
                    S[j+1] = S[j] + p.sigma * sqrt_dt * self.rng.standard_normal()
                    q[j+1] = 0; X[j+1] = X[j]
                delta[i+1:] = 0.0
                break

        # Face-lift / forced liquidation at T: if the policy keeps firing MOs
        # at the last step and q > 0, let it liquidate the remainder at S_T - xi.
        if q[M] > 0:
            d_last, mo_last = policy(t[M], int(q[M]), S[M])
            if mo_last:
                # Consistent with book face-lift: all remaining shares crossed
                # at S_T - xi, no alpha penalty since agent acts at T^-.
                X[M] += int(q[M]) * (S[M] - p.xi)
                mo_sends[M] = True
                q[M] = 0

        delta[-1] = delta[-2] if M > 0 else 0.0
        return Path(t=t, S=S, q=q, delta=delta, X=X,
                    lo_fills=lo_fills, mo_sends=mo_sends)

    def monte_carlo(self, policy, n_paths: int, progress: bool = False):
        """Return dict of summary arrays over n_paths simulations."""
        terminal = np.empty(n_paths)
        q_final = np.empty(n_paths, dtype=np.int64)
        n_mo = np.empty(n_paths, dtype=np.int64)
        n_lo = np.empty(n_paths, dtype=np.int64)
        avg_price = np.empty(n_paths)
        twap_price = np.empty(n_paths)
        q_heat = np.zeros((n_paths, self.n_steps + 1), dtype=np.int64)

        for k in range(n_paths):
            path = self.simulate(policy)
            terminal[k] = path.terminal_value(self.p)
            q_final[k] = path.q_final
            n_mo[k] = int(path.mo_sends.sum())
            n_lo[k] = int(path.lo_fills.sum())
            shares_sold = self.p.Q0 - q_final[k]
            avg_price[k] = (path.X[-1] / shares_sold) if shares_sold > 0 else np.nan
            twap_price[k] = float(np.mean(path.S))
            q_heat[k] = path.q
            if progress and (k + 1) % max(1, n_paths // 10) == 0:
                print(f"  MC {k+1}/{n_paths}")
        return {
            "terminal": terminal,
            "q_final": q_final,
            "n_mo": n_mo,
            "n_lo": n_lo,
            "avg_price": avg_price,
            "twap_price": twap_price,
            "q_heat": q_heat,
            "t_grid": self.t_grid,
        }
