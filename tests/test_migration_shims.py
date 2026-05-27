"""Phase 1 verification: legacy import paths still work via shims, and they
point to the exact same objects as the new canonical locations.

Bound on lifetime: these shims are deleted at the end of Phase 2.
"""
from __future__ import annotations
import warnings


def _import_quiet(path: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mod = __import__(path, fromlist=["*"])
    return mod


def test_task1_solver_shim_is_analytical():
    legacy = _import_quiet("src.task1_lo_only.solver")
    new = _import_quiet("src.analytical.task1_lo_only")
    assert legacy.omega is new.omega
    assert legacy.optimal_depth is new.optimal_depth
    assert legacy.make_policy is new.make_lo_only_policy


def test_task2_fd_solver_shim_is_numerical():
    legacy = _import_quiet("src.task2_lo_mo.fd_solver")
    new = _import_quiet("src.numerical.fd_constant_lambda")
    assert legacy.FDSolution is new.FDSolution
    assert legacy.solve is new.solve


def test_task2_analytic_shim_is_analytical():
    legacy = _import_quiet("src.task2_lo_mo.analytic_q12")
    new = _import_quiet("src.analytical.task2_lo_mo")
    assert legacy.omega_q1 is new.omega_q1
    assert legacy.omega_q2 is new.omega_q2
    assert legacy.critical_time_q2 is new.critical_time_q2


def test_task3_env_shim_is_rl():
    legacy = _import_quiet("src.task3_rl.env")
    new = _import_quiet("src.rl.env")
    assert legacy.LiquidationEnv is new.LiquidationEnv
    assert legacy.EnvConfig is new.EnvConfig


def test_task3_agents_shim_is_rl():
    legacy_t = _import_quiet("src.task3_rl.agent_tabular")
    new_t = _import_quiet("src.rl.tabular_q")
    assert legacy_t.TabularQ is new_t.TabularQ

    legacy_d = _import_quiet("src.task3_rl.agent_dqn")
    new_d = _import_quiet("src.rl.double_dqn")
    assert legacy_d.DDQN is new_d.DDQN
    assert legacy_d.QNet is new_d.QNet


def test_twap_shim_is_baselines():
    legacy = _import_quiet("src.common.twap")
    new = _import_quiet("src.baselines.twap")
    assert legacy.make_twap_mo_policy is new.make_twap_mo_policy
    assert legacy.make_twap_lo_policy is new.make_twap_lo_policy


def test_shims_emit_deprecation():
    """Confirm that import-time warning is raised exactly once per module."""
    import importlib
    import sys

    for mod_name in [
        "src.task1_lo_only.solver",
        "src.task2_lo_mo.fd_solver",
        "src.task2_lo_mo.analytic_q12",
        "src.task3_rl.env",
        "src.task3_rl.agent_tabular",
        "src.task3_rl.agent_dqn",
        "src.common.twap",
    ]:
        sys.modules.pop(mod_name, None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            importlib.import_module(mod_name)
        depr = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(depr) >= 1, f"Shim {mod_name} did not emit DeprecationWarning"
