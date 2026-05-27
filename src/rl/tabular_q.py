"""Tabular Q-learning on (time_bucket, inventory) state, discrete actions.

Because the analytic value function h(t, q) depends only on (t, q), a tabular
Q-learner with sufficient training converges to the optimal CJP §8 solution
(up to discretisation error). It is the cleanest baseline for the AMF
sample-complexity benchmark (Experiment 1).

Migrated from `src.task3_rl.agent_tabular` (Phase 1, AMF revision).
"""
from __future__ import annotations

import numpy as np

from .env import LiquidationEnv


class TabularQ:
    def __init__(self, env: LiquidationEnv, seed: int | None = None):
        self.env = env
        self.Q = np.zeros((env.num_time_buckets, env.p.Q0 + 1, env.num_actions))
        self.rng = np.random.default_rng(seed)

    def greedy_action(self, state: tuple[int, int]) -> int:
        i, q = state[0], state[1]
        if q == 0:
            return self.env.cfg.n_depth
        return int(np.argmax(self.Q[i, q]))

    def value_grid(self, env=None) -> np.ndarray:
        """V_pi(t_idx, q) = max_a Q[t_idx, q, a]. Shape (n_t, Q0+1).

        env arg accepted for the common Agent diagnostic interface; ignored
        because the tabular agent's Q-table already lives on env's grid.
        """
        return self.Q.max(axis=2)

    def act(self, state: tuple[int, int], eps: float,
            mo_explore: float = 0.0) -> int:
        if state[1] > 0 and mo_explore > 0.0 and self.rng.random() < mo_explore:
            return self.env.cfg.n_depth
        if self.rng.random() < eps:
            return int(self.rng.integers(self.env.num_actions))
        return self.greedy_action(state)

    def train(self, n_episodes: int = 50000,
              alpha_start: float = 0.3, alpha_end: float = 0.05,
              eps_start: float = 1.0, eps_end: float = 0.05,
              mo_explore_start: float = 0.0, mo_explore_end: float = 0.0,
              gamma: float = 1.0, log_every: int = 5000,
              h_reference: np.ndarray | None = None):
        env = self.env
        returns = np.zeros(n_episodes)
        rmse_trace = []
        for ep in range(n_episodes):
            frac = ep / max(1, n_episodes - 1)
            alpha = alpha_start + (alpha_end - alpha_start) * frac
            eps = eps_start * (eps_end / eps_start) ** frac
            mo_explore = mo_explore_start + (mo_explore_end - mo_explore_start) * frac

            state = env.reset()
            total_r = 0.0
            done = False
            while not done:
                a = self.act(state, eps, mo_explore=mo_explore)
                next_state, r, done, _ = env.step(a)
                total_r += r
                i, q = state
                ni, nq = next_state
                if done:
                    target = r
                else:
                    target = r + gamma * np.max(self.Q[ni, nq])
                self.Q[i, q, a] += alpha * (target - self.Q[i, q, a])
                state = next_state
            returns[ep] = total_r

            if h_reference is not None and (ep + 1) % log_every == 0:
                V_learned = self.Q.max(axis=2)
                rmse = rmse_vs_reference(env, V_learned, h_reference)
                rmse_trace.append((ep + 1, rmse))
                print(f"ep {ep+1:>6d}  mean_return={np.mean(returns[max(0,ep-999):ep+1]):.4f}"
                      f"  RMSE vs analytic h: {rmse:.4f}  eps={eps:.3f}")
        return {"returns": returns, "rmse_trace": rmse_trace}


def rmse_vs_reference(env: LiquidationEnv, V_learned: np.ndarray,
                      h_ref: np.ndarray) -> float:
    """RMSE of V_learned vs h_ref, resampled to env's t_grid, excluding q=0."""
    fd_t = np.linspace(0.0, env.p.T, h_ref.shape[0])
    env_t = env.t_grid
    idx = np.clip(np.searchsorted(fd_t, env_t), 0, len(fd_t) - 1)
    h_on_env = h_ref[idx]
    err = V_learned[:, 1:] - h_on_env[:, 1:]
    return float(np.sqrt(np.mean(err ** 2)))
