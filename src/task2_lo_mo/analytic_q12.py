"""Deprecated shim. Use `src.analytical.task2_lo_mo` instead."""
import warnings as _warnings
_warnings.warn(
    "src.task2_lo_mo.analytic_q12 is deprecated; use src.analytical.task2_lo_mo",
    DeprecationWarning,
    stacklevel=2,
)

from ..analytical.task2_lo_mo import (  # noqa: E402, F401
    omega_q1,
    omega_q2,
    critical_time_q2,
    depth_q,
)
