"""Diagnostic utilities common to all four RL agents (A-D).

Per the AMF revision plan §5.1 (Theorem 5.1): under the canonical h-shaped
reward, V*(t,q) = h(t,q). This means every agent's learned value function
admits a pointwise error metric against the FD reference h_FD(t,q), giving
us a uniform diagnostic across tabular Q, DDQN, hybrid, and plug-in.

Two helpers are exposed:
- `value_function_grid(env, agent)`  -> V_pi(t_idx, q), shape (n_t, Q0+1)
- `value_error(V_rl, fd_h, ...)`     -> L∞ or RMSE error vs FD reference
- `noise_floor_check(env, fd_sol, ...)` -> stderr-aware sanity check that
  an agent's MC return never beats FD optimum by more than `sigma_mult * stderr`.
"""
from __future__ import annotations
from typing import Literal, Protocol

import numpy as np


class _Agent(Protocol):
    """Minimal interface every agent implements for diagnostics."""
    def value_grid(self, env) -> np.ndarray: ...


def value_function_grid(env, agent) -> np.ndarray:
    """Dispatch to the agent's own value_grid implementation."""
    return agent.value_grid(env)


def value_error(
    V_rl: np.ndarray,
    fd_h: np.ndarray,
    env_t_grid: np.ndarray,
    fd_t_grid: np.ndarray,
    *,
    metric: Literal["linf", "rmse"] = "linf",
    exclude_terminal: bool = True,
    exclude_q0: bool = True,
) -> float:
    """Pointwise error between V_rl (env grid) and fd_h (fine FD grid).

    fd_h is resampled to env_t_grid by nearest-neighbour (same convention as
    legacy `rmse_vs_reference`). q=0 (absorbing) and the terminal row are
    excluded by default.
    """
    idx = np.clip(np.searchsorted(fd_t_grid, env_t_grid), 0, len(fd_t_grid) - 1)
    h_on_env = fd_h[idx]
    diff = V_rl - h_on_env
    if exclude_q0:
        diff = diff[:, 1:]
    if exclude_terminal:
        diff = diff[:-1, :]
    if metric == "linf":
        return float(np.max(np.abs(diff)))
    return float(np.sqrt(np.mean(diff ** 2)))


class EvalAnomaly(RuntimeError):
    """Raised when a learned policy's MC mean beats FD optimum by more than
    `sigma_mult` standard errors of the FD-optimal MC mean. Indicates a
    reward leak or env idiosyncrasy."""


def noise_floor_check(
    env_factory,
    fd_policy,
    p,
    *,
    n_paths: int = 1000,
    seed: int = 42,
    sigma_mult: float = 3.0,
) -> dict:
    """Run FD-optimal policy n_paths times, return (mean, stderr) of terminal
    value. Use the returned dict to compare any learned policy's MC mean.

    Note: we do NOT raise here; callers should compare and raise EvalAnomaly
    themselves so the diagnostic context (agent name, episode) is informative.
    """
    from ..common.simulator import Simulator

    sim = Simulator(p, dt=0.05, seed=seed)
    out = sim.monte_carlo(fd_policy, n_paths=n_paths, progress=False)
    mean = float(np.mean(out["terminal"]))
    stderr = float(np.std(out["terminal"]) / np.sqrt(n_paths))
    return {
        "fd_mc_mean": mean,
        "fd_mc_stderr": stderr,
        "n_paths": n_paths,
        "ceiling": mean + sigma_mult * stderr,
        "sigma_mult": sigma_mult,
    }
