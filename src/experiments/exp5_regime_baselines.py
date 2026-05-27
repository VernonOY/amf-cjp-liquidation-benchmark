"""Regime-level extended baseline panel for AMF Path 1.

The main paper's Table 5 is a Regime-I snapshot.  Path 1 adds an
institutional-scale Regime II and needs a directly comparable baseline panel
to test whether static passive posting is a small-scale artefact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..baselines.aggressive import make_aggressive_policy
from ..baselines.almgren_chriss import make_almgren_chriss_policy
from ..baselines.passive import make_passive_policy
from ..baselines.pov import make_pov_policy
from ..baselines.twap import make_twap_mo_policy
from ..baselines.vwap import make_vwap_policy
from ..common.params import REGIME_I, REGIME_II, Params
from ..common.simulator import Simulator
from ..common.stats import paired_bootstrap_ci
from ..numerical.fd_constant_lambda import solve as fd_solve
from ..rl.env import EnvConfig, LiquidationEnv
from ..rl.plugin_estimator import PluginAgent
from ._runner import write_csv


def _regime_setup(name: str) -> tuple[str, Params, float, float, float]:
    key = name.lower()
    if key in ("i", "1", "regime_i", "regime1"):
        return "Regime I", REGIME_I, 0.01, 0.5, 0.05
    if key in ("ii", "2", "regime_ii", "regime2"):
        return "Regime II", REGIME_II, 0.05, 1.0, 0.1
    raise ValueError(f"unknown regime {name!r}")


def _eval_against_twap(
    *,
    regime_label: str,
    p: Params,
    strategy: str,
    policy,
    twap_terminal: np.ndarray,
    n_paths: int,
    sim_dt: float,
    seed: int,
) -> dict:
    sim = Simulator(p, dt=sim_dt, seed=seed)
    out = sim.monte_carlo(policy, n_paths=n_paths, progress=False)
    diff = out["terminal"] - twap_terminal
    _, lo, hi = paired_bootstrap_ci(out["terminal"], twap_terminal,
                                    n_boot=2000, seed=seed)
    return {
        "regime": regime_label,
        "strategy": strategy,
        "premium_vs_twap": float(np.mean(diff)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "clearance_prob": float(np.mean(out["q_final"] == 0)),
        "mo_per_path": float(np.mean(out["n_mo"])),
        "lo_per_path": float(np.mean(out["n_lo"])),
    }


def run_regime(
    regime: str = "ii",
    *,
    n_paths: int = 2000,
    seed: int = 42,
    plugin_fills: int = 1000,
) -> list[dict]:
    regime_label, p, fd_dt, env_dt, sim_dt = _regime_setup(regime)
    sim_twap = Simulator(p, dt=sim_dt, seed=seed)
    twap_policy = make_twap_mo_policy(p)
    twap_out = sim_twap.monte_carlo(twap_policy, n_paths=n_paths,
                                     progress=False)
    twap_terminal = twap_out["terminal"]
    rows = [{
        "regime": regime_label,
        "strategy": "TWAP",
        "premium_vs_twap": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "clearance_prob": float(np.mean(twap_out["q_final"] == 0)),
        "mo_per_path": float(np.mean(twap_out["n_mo"])),
        "lo_per_path": float(np.mean(twap_out["n_lo"])),
    }]

    policies = [
        ("Almgren-Chriss", make_almgren_chriss_policy(p)),
        ("VWAP", make_vwap_policy(p)),
        ("POV 10pct", make_pov_policy(p, rho=0.10)),
        ("Pure passive", make_passive_policy(p)),
        ("Pure aggressive", make_aggressive_policy(p)),
    ]
    fd = fd_solve(p, dt=fd_dt)
    policies.append(("FD Optimal", fd.make_policy()))

    env = LiquidationEnv(p, EnvConfig(dt=env_dt, n_depth=21), seed=seed)
    plugin = PluginAgent(env, prior_kappa=p.kappa, prior_lam=p.lam,
                          fd_dt=fd_dt, seed=seed + 10_000)
    plugin.train(n_fills_target=plugin_fills)
    policies.append((f"Plug-in MLE ({plugin_fills} fills)",
                     plugin.fd_solution.make_policy()))

    for idx, (name, policy) in enumerate(policies):
        rows.append(_eval_against_twap(
            regime_label=regime_label,
            p=p,
            strategy=name,
            policy=policy,
            twap_terminal=twap_terminal,
            n_paths=n_paths,
            sim_dt=sim_dt,
            seed=seed + 1000 + idx,
        ))
    return rows


def run(regimes: Sequence[str] = ("ii",), out_csv: str = "data/exp5/regime_baselines.csv",
        n_paths: int = 2000, seed: int = 42, plugin_fills: int = 1000) -> list[dict]:
    rows: list[dict] = []
    for regime in regimes:
        rows.extend(run_regime(regime, n_paths=n_paths, seed=seed,
                               plugin_fills=plugin_fills))
    write_csv(rows, out_csv,
              columns=["regime", "strategy", "premium_vs_twap", "ci_low",
                       "ci_high", "clearance_prob", "mo_per_path",
                       "lo_per_path"])
    return rows


if __name__ == "__main__":  # pragma: no cover
    import argparse

    pa = argparse.ArgumentParser()
    pa.add_argument("--regime", default="ii", choices=("i", "ii", "both"))
    pa.add_argument("--n-paths", type=int, default=2000)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--plugin-fills", type=int, default=1000)
    pa.add_argument("--out", default="data/exp5/regime_baselines.csv")
    args = pa.parse_args()

    regimes = ("i", "ii") if args.regime == "both" else (args.regime,)
    rows = run(regimes=regimes, out_csv=args.out, n_paths=args.n_paths,
               seed=args.seed, plugin_fills=args.plugin_fills)
    print(f"wrote {len(rows)} rows to {args.out}")
