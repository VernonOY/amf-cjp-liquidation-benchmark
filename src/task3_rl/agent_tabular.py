"""Deprecated shim. Use `src.rl.tabular_q` instead."""
import warnings as _warnings
_warnings.warn(
    "src.task3_rl.agent_tabular is deprecated; use src.rl.tabular_q",
    DeprecationWarning,
    stacklevel=2,
)

from ..rl.tabular_q import TabularQ, rmse_vs_reference  # noqa: E402, F401
