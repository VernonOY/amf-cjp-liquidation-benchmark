"""Deprecated location. Use `src.numerical.fd_constant_lambda` and
`src.analytical.task2_lo_mo`.
"""
import warnings as _warnings
_warnings.warn(
    "src.task2_lo_mo is deprecated; use src.numerical.fd_constant_lambda "
    "and src.analytical.task2_lo_mo",
    DeprecationWarning,
    stacklevel=2,
)
