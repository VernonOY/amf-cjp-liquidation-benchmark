"""Gym-style RL environment matching the LO+MO dynamics of the CJP §8 model.

Phase-2 refactor: the reward is now a pluggable callable. The default
(`reward_fn=None`) reproduces the legacy h-centric reward bit-identical —
golden test `test_env_trace_seed0_bit_identical` guards this.

Theorem 5.1 (AMF revision §5.1): under the default reward and the analytic
optimal policy, E[sum of rewards] = h(0, Q_0) where h is the FD-reduced
value function. This gives every RL agent a pointwise diagnostic against the
analytic / FD reference.

Action space is discrete with (n_depth + 1) actions:
  0 .. n_depth-1 : post LO at depth grid point k
  n_depth        : execute a market order (one share)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np

from ..common.params import Params


def terminal_face_lift_value(p: Params, q: int) -> float:
    """Reduced terminal value for residual inventory at the face-lift."""
    return max(-p.alpha * q * q, -p.xi * q)


# ---------------------------------------------------------------------------
# Lambda processes (constant default, CIR in Phase 4)
# ---------------------------------------------------------------------------
class ConstantLam:
    """Trivial lambda process: lam_t == p.lam for all t."""
    def __init__(self, lam: float):
        self.lam = float(lam)
        self.bar = float(lam)

    def reset(self, rng) -> float:
        return self.lam

    def step(self, dt: float, rng) -> float:
        return self.lam

    def bucketize(self, lam: float) -> int:
        return 0


@dataclass
class EnvConfig:
    dt: float = 0.5
    n_depth: int = 21
    depth_min: float = 0.0
    depth_max: float = 0.05
    state_mode: Literal["tq", "tql"] = "tq"
    n_lam_buckets: int = 1


# trans dict carries:
#   q_pre, q_after_mo, q_after, delta, mo_flag, lo_filled, dt, done,
#   is_terminal_face_lift, S_prev, S_next, lam
RewardFn = Callable[[Params, "LiquidationEnv", dict], float]


class LiquidationEnv:
    """Single-agent discrete-action liquidation environment.

    Parameters
    ----------
    p : CJP §8 parameters
    cfg : env discretisation (dt, depth grid)
    reward_fn : pluggable reward. None -> legacy h-shaped (bit-identical to
                the Phase-1 env, golden test enforces this).
    lam_process : LamProcess-like; default ConstantLam(p.lam). CIR Phase 4.
    seed : RNG seed for fill / midprice dynamics.
    """

    def __init__(self, p: Params, cfg: EnvConfig | None = None,
                 *, reward_fn: Optional[RewardFn] = None,
                 lam_process=None,
                 seed: int | None = None):
        self.p = p
        self.cfg = cfg or EnvConfig()
        self.reward_fn = reward_fn  # None means legacy hard-coded path
        self.lam_process = lam_process if lam_process is not None else ConstantLam(p.lam)
        self.n_steps = int(round(p.T / self.cfg.dt))
        self.t_grid = np.linspace(0.0, p.T, self.n_steps + 1)
        self.rng = np.random.default_rng(seed)
        self.depth_grid = np.linspace(self.cfg.depth_min,
                                      self.cfg.depth_max,
                                      self.cfg.n_depth)
        self.n_actions = self.cfg.n_depth + 1

        self._i = 0
        self._q = p.Q0
        self._S = p.S0
        self._lam = float(p.lam)

    # --- Gym-style API -----------------------------------------------------
    @property
    def num_actions(self) -> int:
        return self.n_actions

    @property
    def num_time_buckets(self) -> int:
        return self.n_steps + 1

    def reset(self) -> tuple:
        self._i = 0
        self._q = self.p.Q0
        self._S = self.p.S0
        self._lam = float(self.lam_process.reset(self.rng))
        return self._observation()

    def _observation(self):
        if self.cfg.state_mode == "tql":
            return (self._i, self._q, int(self.lam_process.bucketize(self._lam)))
        return (self._i, self._q)

    def state_to_feat(self, state) -> np.ndarray:
        """Continuous feature vector used by DDQN / hybrid policy.

        tq  -> [t_idx / (n_t-1), q / Q0]
        tql -> [t_idx / (n_t-1), q / Q0, (lam - bar) / bar]  (Phase 4)
        """
        nt = max(1, self.num_time_buckets - 1)
        Q = max(1, self.p.Q0)
        i = state[0]
        q = state[1]
        feat = [i / nt, q / Q]
        if self.cfg.state_mode == "tql":
            lam_bar = max(getattr(self.lam_process, "bar", self.p.lam), 1e-12)
            feat.append((self._lam - lam_bar) / (2.0 * lam_bar))
        return np.asarray(feat, dtype=np.float32)

    def action_to_policy(self, a: int) -> tuple[float, bool]:
        if a == self.cfg.n_depth:
            return (0.0, True)
        return (float(self.depth_grid[a]), False)

    # --- step --------------------------------------------------------------
    def step(self, action: int):
        p, cfg = self.p, self.cfg
        dt = cfg.dt
        q_pre = self._q
        S_prev = self._S
        i = self._i
        delta, mo_flag = self.action_to_policy(action)

        q = q_pre
        mo_done = False
        if mo_flag and q > 0:
            q -= 1
            mo_done = True
        q_after_mo = q

        # Midprice diffusion
        sqrt_dt = np.sqrt(dt)
        S_next = S_prev + p.sigma * sqrt_dt * self.rng.standard_normal()

        # LO fill check (only if still holding and not doing MO)
        lo_filled = False
        if q > 0 and delta >= 0 and not mo_flag:
            rate = self._lam * np.exp(-p.kappa * delta)
            if self.rng.random() < 1.0 - np.exp(-rate * dt):
                q -= 1
                lo_filled = True

        # Advance time / lambda
        i += 1
        self._i, self._q, self._S = i, q, S_next
        self._lam = float(self.lam_process.step(dt, self.rng))

        done = (i >= self.n_steps) or (q == 0)
        is_terminal_face_lift = bool(done and q > 0)

        # Build the transition dict for the reward function
        trans = {
            "q_pre": q_pre,
            "q_after_mo": q_after_mo,
            "q_after": q,
            "delta": float(delta),
            "mo_flag": bool(mo_flag),
            "lo_filled": bool(lo_filled),
            "dt": float(dt),
            "done": bool(done),
            "is_terminal_face_lift": is_terminal_face_lift,
            "S_prev": float(S_prev),
            "S_next": float(S_next),
            "lam": float(self._lam),
        }

        if self.reward_fn is None:
            # LEGACY PATH (bit-identical to Phase-1 env).
            reward = 0.0
            reward -= p.phi * q_pre * q_pre * dt
            if mo_done:
                reward -= p.xi
            if lo_filled:
                reward += float(delta)
            if is_terminal_face_lift:
                reward += terminal_face_lift_value(p, q_after_mo)
        else:
            reward = float(self.reward_fn(p, self, trans))

        # Terminal face-lift: clear the inventory
        if is_terminal_face_lift:
            q = 0
            self._q = 0

        info = {
            "lo_fill": lo_filled,
            "mo_done": mo_done,
            "q_after": q,
            "delta": float(delta),
            "lam": float(self._lam),
        }
        return self._observation(), reward, done, info
