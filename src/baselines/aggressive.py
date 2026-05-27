"""Pure-aggressive baseline: cross the spread at every step."""
from __future__ import annotations

from ..common.params import Params


def make_aggressive_policy(p: Params):
    """Immediately fire one market order whenever inventory remains."""
    def policy(t: float, q: int, S: float):
        return (0.0, q > 0)

    return policy
