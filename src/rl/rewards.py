"""Pluggable reward functions for the liquidation environment.

`h_shaped` is the canonical reward of the AMF revision (Theorem 5.1): under
the optimal policy the expected cumulative reward equals h(0, Q_0) where
h(t, q) is the FD-reduced value function. This is the default reward and is
locked by the golden env-trace test.

`pnl_only` and `risk_averse` are alternative shapings used by Experiment 4
(failure-mode analysis) to demonstrate that pure-PnL rewards leak signal and
that risk-averse shapings change the qualitative MO-trigger geometry.
"""
from __future__ import annotations
from typing import Callable

from ..common.params import Params
from .env import terminal_face_lift_value


# Type alias used throughout src.rl
# trans dict carries: q_pre, q_post, delta, mo_flag, lo_filled, dt, done,
#                     is_terminal_face_lift, S_prev, S_next, lam
RewardFn = Callable[[Params, "object", dict], float]


def h_shaped(p: Params, _env, trans: dict) -> float:
    """Canonical h-centric reward (CJP §8, Theorem 5.1 of the AMF revision).

    Components:
      -phi * q^2 * dt              running inventory penalty (q is q_pre)
      -xi                          when agent fires its own MO
      +delta                       on each LO fill at posted depth
      max(-alpha q^2, -xi q)       forced face-lifted liquidation at T
    """
    r = 0.0
    r -= p.phi * trans["q_pre"] ** 2 * trans["dt"]
    if trans["mo_flag"] and trans["q_pre"] > 0:
        r -= p.xi
    if trans["lo_filled"]:
        r += float(trans["delta"])
    if trans["is_terminal_face_lift"]:
        r += terminal_face_lift_value(p, trans["q_after_mo"])
    return r


def pnl_only(p: Params, _env, trans: dict) -> float:
    """Raw cash + mark-to-market PnL change. Used by Experiment 4 to expose
    that PnL-only RL leaks price signal and under-uses MOs."""
    r = 0.0
    if trans["mo_flag"] and trans["q_pre"] > 0:
        # Agent sells one share at S - xi
        r += trans["S_prev"] - p.xi - trans["S_prev"]   # net cash flow ignoring drift
    if trans["lo_filled"]:
        r += trans["S_prev"] + float(trans["delta"]) - trans["S_prev"]
    # Mark-to-market inventory revaluation
    q_after = trans["q_after"]
    r += q_after * (trans["S_next"] - trans["S_prev"])
    if trans["is_terminal_face_lift"]:
        r += terminal_face_lift_value(p, trans["q_after_mo"])
    return r


def risk_averse(p: Params, _env, trans: dict, *, gamma: float = 2.0) -> float:
    """CARA utility on the h-centric base reward (Experiment 4 ablation)."""
    base = h_shaped(p, _env, trans)
    # 1 - exp(-gamma * base) / gamma, well-defined for small base
    import math
    return float((1.0 - math.exp(-gamma * base)) / gamma)
