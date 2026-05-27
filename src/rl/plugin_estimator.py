"""Agent D — Plug-in Estimator (AMF revision §5.2.4).

The plug-in agent is a pure model-calibrator:
  1. Run a random-depth exploration policy until n_fills_target fills observed.
  2. Joint MLE for (kappa_hat, lam_hat) via exact-Bernoulli likelihood.
  3. Solve the FD QVI at (kappa_hat, lam_hat) -> closed-form depth schedule.
  4. Use the FD-derived policy as the deployed policy.

This is the sample-efficient upper bound on the four-agent sample-complexity
benchmark (Experiment 1). Its only cost is the exploration rollout + one FD
solve; gradient updates are not used.
"""
from __future__ import annotations
import dataclasses
import math

import numpy as np

from ..common.estimators import fit_kappa_lambda_mle
from ..common.params import Params
from ..numerical.fd_constant_lambda import FDSolution, solve as fd_solve
from .env import LiquidationEnv


@dataclasses.dataclass
class PluginEstimates:
    kappa_hat: float
    lam_hat: float
    kappa_se: float
    lam_se: float
    n_obs: int
    n_fills: float


class PluginAgent:
    """Plug-in MLE agent. Maintains its own (kappa_hat, lam_hat) state."""

    def __init__(
        self,
        env: LiquidationEnv,
        *,
        prior_kappa: float = 100.0,
        prior_lam: float = 50.0 / 60.0,
        ridge_sigma_loose: float = math.log(10.0),
        ridge_sigma_tight: float = math.log(2.0),
        ridge_n_threshold: int = 200,
        fd_dt: float = 0.01,
        seed: int | None = None,
    ):
        self.env = env
        self.prior_kappa = float(prior_kappa)
        self.prior_lam = float(prior_lam)
        self.ridge_sigma_loose = float(ridge_sigma_loose)
        self.ridge_sigma_tight = float(ridge_sigma_tight)
        self.ridge_n_threshold = int(ridge_n_threshold)
        self.fd_dt = float(fd_dt)
        self.rng = np.random.default_rng(seed)
        # Fitted state
        self.estimates: PluginEstimates | None = None
        self._fd_sol: FDSolution | None = None
        self._p_hat: Params | None = None

    # ----- step 1: random-delta exploration --------------------------------
    def collect_pairs(
        self,
        n_fills_target: int,
        *,
        max_steps: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run a random-depth exploration policy until n_fills_target fills.

        Returns (deltas, fills) per environment step. Each LO posted at a
        random depth in [0, env.cfg.depth_max]. No MOs are used (we only
        need the LO-fill data to identify (kappa, lam)).
        """
        env = self.env
        deltas: list[float] = []
        fills: list[float] = []
        n_filled = 0
        max_steps = max_steps or (n_fills_target * 200)
        step_budget = 0
        while n_filled < n_fills_target and step_budget < max_steps:
            state = env.reset()
            done = False
            while not done and n_filled < n_fills_target:
                d = float(self.rng.uniform(0.0, env.cfg.depth_max))
                a = int(np.argmin(np.abs(env.depth_grid - d)))
                ns, _r, done, info = env.step(a)
                deltas.append(env.depth_grid[a])
                f = 1.0 if info["lo_fill"] else 0.0
                fills.append(f)
                if f > 0:
                    n_filled += 1
                state = ns
                step_budget += 1
        return np.asarray(deltas, dtype=np.float64), np.asarray(fills, dtype=np.float64)

    # ----- step 2: joint MLE -----------------------------------------------
    def fit(self, deltas: np.ndarray, fills: np.ndarray) -> PluginEstimates:
        n_obs = len(deltas)
        ridge = (self.ridge_sigma_tight if n_obs < self.ridge_n_threshold
                 else self.ridge_sigma_loose)
        # Ridge weight = 0.5 / sigma^2 in the Gaussian-prior parameterisation.
        ridge_weight = 0.5 / (ridge ** 2)
        out = fit_kappa_lambda_mle(
            deltas, fills, dt=self.env.cfg.dt,
            ridge_log_kappa=ridge_weight,
            ridge_log_lam=ridge_weight,
            kappa_prior=self.prior_kappa,
            lam_prior=self.prior_lam,
        )
        est = PluginEstimates(
            kappa_hat=float(out["kappa_hat"]),
            lam_hat=float(out["lam_hat"]),
            kappa_se=float(out["kappa_se"]),
            lam_se=float(out["lam_se"]),
            n_obs=int(out["n_obs"]),
            n_fills=float(out["n_fills"]),
        )
        self.estimates = est
        return est

    # ----- step 3: FD solve at hat params ----------------------------------
    def _build_policy(self) -> None:
        if self.estimates is None:
            raise RuntimeError("PluginAgent: call fit() before solving the FD QVI")
        p0 = self.env.p
        self._p_hat = dataclasses.replace(
            p0, kappa=self.estimates.kappa_hat, lam=self.estimates.lam_hat
        )
        self._fd_sol = fd_solve(self._p_hat, dt=self.fd_dt)

    def train(self, n_fills_target: int, *, max_steps: int | None = None) -> dict:
        """End-to-end: collect data -> MLE -> FD solve. Returns diagnostics."""
        deltas, fills = self.collect_pairs(n_fills_target, max_steps=max_steps)
        est = self.fit(deltas, fills)
        self._build_policy()
        return {
            "n_obs": est.n_obs,
            "n_fills": est.n_fills,
            "kappa_hat": est.kappa_hat,
            "lam_hat": est.lam_hat,
            "kappa_se": est.kappa_se,
            "lam_se": est.lam_se,
        }

    # ----- step 4: deploy --------------------------------------------------
    def act(self, state) -> int:
        if self._fd_sol is None:
            raise RuntimeError("PluginAgent: train() before act()")
        i_env, q = state[0], state[1]
        if q <= 0:
            return self.env.cfg.n_depth  # no-op
        # Map env time-index to FD time-index by ratio of grids
        fd_t = self._fd_sol.t_grid
        t = self.env.t_grid[i_env]
        i_fd = min(int(round(t / (fd_t[1] - fd_t[0]))), len(fd_t) - 1)
        if self._fd_sol.mo_trigger[i_fd, q]:
            return self.env.cfg.n_depth
        d_target = self._fd_sol.depth_grid()[i_fd, q]
        d_target = max(d_target, 0.0)
        return int(np.argmin(np.abs(self.env.depth_grid - d_target)))

    @property
    def fd_solution(self) -> FDSolution | None:
        return self._fd_sol

    # ----- common diagnostic interface ------------------------------------
    def value_grid(self, env=None) -> np.ndarray:
        env = env or self.env
        if self._fd_sol is None:
            return np.zeros((env.num_time_buckets, env.p.Q0 + 1))
        fd_t = self._fd_sol.t_grid
        idx = np.clip(np.searchsorted(fd_t, env.t_grid), 0, len(fd_t) - 1)
        return self._fd_sol.h[idx]

    def policy_grid(self, env=None) -> tuple[np.ndarray, np.ndarray]:
        env = env or self.env
        nt = env.num_time_buckets
        Q = env.p.Q0
        depth = np.zeros((nt, Q + 1))
        trigger = np.zeros((nt, Q + 1), dtype=bool)
        if self._fd_sol is None:
            return depth, trigger
        fd_t = self._fd_sol.t_grid
        idx = np.clip(np.searchsorted(fd_t, env.t_grid), 0, len(fd_t) - 1)
        depth_full = self._fd_sol.depth_grid()
        trigger_full = self._fd_sol.mo_trigger
        for i, j in enumerate(idx):
            for q in range(1, Q + 1):
                if trigger_full[j, q]:
                    trigger[i, q] = True
                else:
                    depth[i, q] = max(depth_full[j, q], 0.0)
        return depth, trigger
