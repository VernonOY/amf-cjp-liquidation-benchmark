"""Deprecated shim. Use `src.rl.double_dqn` instead."""
import warnings as _warnings
_warnings.warn(
    "src.task3_rl.agent_dqn is deprecated; use src.rl.double_dqn",
    DeprecationWarning,
    stacklevel=2,
)

from ..rl.double_dqn import QNet, DDQN, state_to_feat  # noqa: E402, F401
