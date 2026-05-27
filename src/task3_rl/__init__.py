"""Deprecated location. Use `src.rl.{env, tabular_q, double_dqn}`."""
import warnings as _warnings
_warnings.warn(
    "src.task3_rl is deprecated; use src.rl.{env, tabular_q, double_dqn}",
    DeprecationWarning,
    stacklevel=2,
)
