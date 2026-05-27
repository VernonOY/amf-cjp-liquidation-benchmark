"""Deprecated shim. Use `src.analytical.task1_lo_only` instead."""
import warnings as _warnings
_warnings.warn(
    "src.task1_lo_only.solver is deprecated; use src.analytical.task1_lo_only",
    DeprecationWarning,
    stacklevel=2,
)

from ..analytical.task1_lo_only import (  # noqa: E402, F401
    omega,
    h_func,
    optimal_depth,
    precompute_delta_grid,
    make_lo_only_policy,
    make_policy,
    asymptotic_depth,
)
