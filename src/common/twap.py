"""Deprecated shim. Use `src.baselines.twap` instead."""
import warnings as _warnings
_warnings.warn(
    "src.common.twap is deprecated; use src.baselines.twap",
    DeprecationWarning,
    stacklevel=2,
)

from ..baselines.twap import make_twap_mo_policy, make_twap_lo_policy  # noqa: E402, F401
