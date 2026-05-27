"""Deprecated shim. Use `src.rl.env` instead."""
import warnings as _warnings
_warnings.warn(
    "src.task3_rl.env is deprecated; use src.rl.env",
    DeprecationWarning,
    stacklevel=2,
)

from ..rl.env import LiquidationEnv, EnvConfig  # noqa: E402, F401
