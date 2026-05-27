"""Regression tests for the terminal-impact parameter alpha."""
from __future__ import annotations

import dataclasses

import numpy as np

from src.common.params import TASK2
from src.numerical.fd_constant_lambda import solve, terminal_face_lift


def test_terminal_face_lift_uses_alpha_branch_for_small_inventory():
    p = TASK2
    assert terminal_face_lift(p, 1) == -p.alpha
    assert terminal_face_lift(p, p.Q0) == -p.xi * p.Q0


def test_fd_solution_responds_to_alpha_stress():
    p_low = dataclasses.replace(TASK2, alpha=1e-6)
    p_base = dataclasses.replace(TASK2, alpha=1e-3)
    p_high = dataclasses.replace(TASK2, alpha=1.0)

    h_low = solve(p_low, dt=0.05).h
    h_base = solve(p_base, dt=0.05).h
    h_high = solve(p_high, dt=0.05).h

    assert np.max(np.abs(h_low - h_base)) > 1e-3
    assert np.max(np.abs(h_high - h_base)) > 1e-3
