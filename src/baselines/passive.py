"""Pure-passive baseline: post LO at a fixed depth, never MO.

Unfilled inventory at T pays the face-lift cost; useful as a "do-nothing"
extreme alongside `aggressive.py`.
"""
from __future__ import annotations

from ..common.params import Params


def make_passive_policy(p: Params, *, depth: float | None = None):
    """Post LO at fixed depth `depth`. Default = 2 / kappa."""
    d = float(depth if depth is not None else 2.0 / p.kappa)

    def policy(t: float, q: int, S: float):
        return (d, False)

    return policy
