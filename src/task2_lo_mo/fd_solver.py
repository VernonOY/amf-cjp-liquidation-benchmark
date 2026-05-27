"""Deprecated shim. Use `src.numerical.fd_constant_lambda` instead."""
import warnings as _warnings
_warnings.warn(
    "src.task2_lo_mo.fd_solver is deprecated; use src.numerical.fd_constant_lambda",
    DeprecationWarning,
    stacklevel=2,
)

from ..numerical.fd_constant_lambda import (  # noqa: E402, F401
    FDSolution,
    solve,
    validate_against_analytic,
)
