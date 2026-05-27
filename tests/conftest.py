"""Pytest fixtures shared across the AMF revision test suite.

Provides:
- Deterministic Params instances (TASK1, TASK2) — re-exported from src.common.params
- A Simulator factory with explicit seed handling
- The path to tests/_golden/ for snapshot tests
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable when pytest runs from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.common.params import Params, TASK1, TASK2  # noqa: E402
from src.common.simulator import Simulator  # noqa: E402


GOLDEN_DIR = Path(__file__).resolve().parent / "_golden"


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    return GOLDEN_DIR


@pytest.fixture(scope="session")
def task1_params() -> Params:
    return TASK1


@pytest.fixture(scope="session")
def task2_params() -> Params:
    return TASK2


@pytest.fixture
def sim_factory():
    """Returns a callable `make_sim(p, dt=0.05, seed=0) -> Simulator`."""
    def _make(p: Params, dt: float = 0.05, seed: int = 0) -> Simulator:
        return Simulator(p, dt=dt, seed=seed)
    return _make


@pytest.fixture
def rng_seed0() -> np.random.Generator:
    return np.random.default_rng(0)
