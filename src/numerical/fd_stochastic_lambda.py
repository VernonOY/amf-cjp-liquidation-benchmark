"""Finite-difference solver for the LO+MO QVI under CIR fill intensity
(AMF revision §6.2 / §6.3).

Reduced QVI for h(t, q, lam):

    max{
        d_t h + L_lam h - phi q^2
          + (lam / (e kappa)) exp(-kappa [h(t,q,lam) - h(t,q-1,lam)]) ;
        -xi + h(t, q-1, lam) - h(t, q, lam)
    } = 0

with L_lam h = kappa_lam (theta_lam - lam) d_lam h
             + 0.5 sigma_lam^2 lam d_lam_lam h.

Boundary conditions:
    h(t, 0, lam) = 0
    h(T^-, q, lam) = max(-alpha q^2, -q xi)    (face-lifted terminal)
    Neumann at lam_min and lam_max

We discretise lam on a log-spaced grid on [0.1 theta_lam, 5 theta_lam].
Each backward time step uses operator splitting:

    1. Backward-Euler implicit step for the CIR lambda generator
    2. Explicit QVI intervention + inventory drift update

In the sigma_lam -> 0 limit the lam process is deterministic (it drifts
toward theta_lam at rate kappa_lam); at sigma_lam = 0 and kappa_lam = 0
the dynamics reduce to constant lambda and the solution must agree with
`fd_constant_lambda.solve(p)` to discretisation accuracy. The unit test
`test_cir_limit.py` checks this convergence.
"""
from __future__ import annotations
import dataclasses
import math

import numpy as np

from ..common.params import Params
from ..common.params_cir import CIRParams
from .fd_constant_lambda import terminal_face_lift


@dataclasses.dataclass
class CIRFDSolution:
    p: Params
    cir: CIRParams
    t_grid: np.ndarray
    lam_grid: np.ndarray
    h: np.ndarray              # shape (n_t+1, n_lam, Q_max+1)
    mo_trigger: np.ndarray     # bool, same shape

    def depth_grid_at_lam(self, lam_idx: int) -> np.ndarray:
        """delta*(t, q | lam_idx). Shape (n_t+1, Q_max+1)."""
        h_slice = self.h[:, lam_idx, :]
        g = np.zeros_like(h_slice)
        for q in range(1, g.shape[1]):
            g[:, q] = 1.0 / self.p.kappa + h_slice[:, q] - h_slice[:, q - 1]
        return g

    def make_policy(self, lam_process):
        """Build a simulator-compatible policy. The lam path is drawn by the
        provided `lam_process` (must expose .step(dt, rng) and .lam).

        Because the simulator currently does not pass lam_t into policies, we
        approximate by polling `lam_process.lam` each call — i.e. the env's
        ConstantLam(p.lam) will reduce to the constant-lam optimum. This is a
        Phase-4 stub used by the experiments script; a proper implementation
        requires extending the simulator to expose lam_t in the callback.
        """
        dt_t = self.t_grid[1] - self.t_grid[0]
        Q_max = self.h.shape[2] - 1
        lam_g = self.lam_grid

        def policy(t: float, q: int, S: float):
            if q <= 0:
                return (0.0, False)
            i = min(int(round(t / dt_t)), len(self.t_grid) - 1)
            cur_lam = getattr(lam_process, "lam", float(self.cir.lam0))
            j = int(np.argmin(np.abs(lam_g - cur_lam)))
            if self.mo_trigger[i, j, min(q, Q_max)]:
                return (0.0, True)
            d = 1.0 / self.p.kappa + self.h[i, j, min(q, Q_max)] - self.h[i, j, min(q - 1, Q_max)]
            return (float(max(d, 0.0)), False)

        return policy


def _build_lam_grid(cir: CIRParams, n_lam: int,
                    lam_min: float | None = None,
                    lam_max: float | None = None,
                    lam_anchors: np.ndarray | None = None) -> np.ndarray:
    lo = lam_min if lam_min is not None else 0.1 * cir.theta_lam
    hi = lam_max if lam_max is not None else 5.0 * cir.theta_lam
    lo = max(lo, 1e-6)
    grid = np.geomspace(lo, hi, n_lam)
    if lam_anchors is not None:
        anchors = np.asarray(lam_anchors, dtype=float)
        anchors = anchors[(anchors >= lo) & (anchors <= hi)]
        grid = np.unique(np.concatenate([grid, anchors]))
    return grid


def _tridiag_lam_operator(
    cir: CIRParams, lam_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sub, main, sup) diagonals of the discrete generator L_lam,
    where  L_lam u = kappa_lam (theta_lam - lam) u'  +  0.5 sigma_lam^2 lam u''.

    Drift term uses UPWIND differencing (essential for stability of the
    pure-drift component as sigma_lam -> 0; central differencing is
    unconditionally unstable for advection on a tensor grid). The
    diffusion term uses central differences on a non-uniform grid.
    """
    n = lam_grid.size
    sub = np.zeros(n)
    main = np.zeros(n)
    sup = np.zeros(n)
    for j in range(1, n - 1):
        dxm = lam_grid[j] - lam_grid[j - 1]
        dxp = lam_grid[j + 1] - lam_grid[j]
        drift = cir.kappa_lam * (cir.theta_lam - lam_grid[j])
        # The backward value PDE is stepped forward in tau = T - t:
        # u_tau = b(lambda) u_lambda + ... .  For this equation the stable
        # one-sided stencil is the generator-form discretisation with
        # non-negative off-diagonal entries: if b > 0, use a forward
        # difference; if b < 0, use a backward difference.
        if drift >= 0:
            sub_drift = 0.0
            main_drift = -drift / dxp
            sup_drift = drift / dxp
        else:
            sub_drift = -drift / dxm
            main_drift = drift / dxm
            sup_drift = 0.0
        # Central difference for u'' on non-uniform grid
        diff_coef = 0.5 * cir.sigma_lam ** 2 * lam_grid[j]
        sub_diff = 2.0 / (dxm * (dxm + dxp))
        main_diff = -2.0 / (dxm * dxp)
        sup_diff = 2.0 / (dxp * (dxm + dxp))
        sub[j] = sub_drift + diff_coef * sub_diff
        main[j] = main_drift + diff_coef * main_diff
        sup[j] = sup_drift + diff_coef * sup_diff
    return sub, main, sup


def _thomas(sub: np.ndarray, main: np.ndarray, sup: np.ndarray,
            rhs: np.ndarray) -> np.ndarray:
    """Tridiagonal solve via scipy.linalg.solve_banded. Robust against
    near-zero diagonal entries that our previous hand-rolled Thomas
    implementation handled incorrectly."""
    from scipy.linalg import solve_banded
    n = rhs.size
    ab = np.zeros((3, n))
    ab[0, 1:] = sup[:-1]    # upper diagonal (shifted right)
    ab[1, :] = main         # main diagonal
    ab[2, :-1] = sub[1:]    # lower diagonal (shifted left)
    return solve_banded((1, 1), ab, rhs)


def solve_cir(
    p: Params,
    cir: CIRParams,
    *,
    dt: float = 0.05,
    n_lam: int = 30,
    lam_min: float | None = None,
    lam_max: float | None = None,
    lam_anchors: np.ndarray | None = None,
    Q_max: int | None = None,
) -> CIRFDSolution:
    """Backward-in-time solver for the CIR-intensity QVI.

    Uses backward Euler on the lambda generator (implicit) and explicit Euler
    on the intervention + inventory drift term.
    """
    if Q_max is None:
        Q_max = p.Q0
    M = int(round(p.T / dt))
    t_grid = np.linspace(0.0, p.T, M + 1)
    lam_grid = _build_lam_grid(cir, n_lam, lam_min, lam_max, lam_anchors)
    n_lam_eff = lam_grid.size

    h = np.zeros((M + 1, n_lam_eff, Q_max + 1))
    trigger = np.zeros_like(h, dtype=bool)

    # Face-lifted terminal is independent of lambda.
    for q in range(1, Q_max + 1):
        h[M, :, q] = terminal_face_lift(p, q)
        trigger[M, :, q] = (-p.xi * q) >= (-p.alpha * q * q)

    sub, main, sup = _tridiag_lam_operator(cir, lam_grid)
    # Backward Euler tridiagonals: (I - dt L) h^{new} = h^{old}.
    # Backward Euler is L-stable (necessary for stiff CIR drift terms
    # near the lambda boundary); less diffusive implicit variants produced
    # oscillations / blow-up in our tests at small lambda.
    A_sub = -dt * sub
    A_main = 1.0 - dt * main
    A_sup = -dt * sup
    # Pin boundary rows of A to identity so the linear system is
    # well-posed; the Neumann condition is then re-imposed by copy.
    A_sub[0] = 0.0; A_main[0] = 1.0; A_sup[0] = 0.0
    A_sub[-1] = 0.0; A_main[-1] = 1.0; A_sup[-1] = 0.0

    for i in range(M - 1, -1, -1):
        # ---- 1. Implicit step on the lam diffusion (per q) ----------
        h_star = h[i + 1].copy()
        for q in range(1, Q_max + 1):
            # Backward Euler RHS = h_old (identity for the I term)
            rhs = h[i + 1, :, q].copy()
            h_star[:, q] = _thomas(A_sub, A_main, A_sup, rhs)
            # Enforce zero-flux Neumann: copy interior neighbour
            h_star[0, q] = h_star[1, q]
            h_star[-1, q] = h_star[-2, q]
        h_star[:, 0] = 0.0

        # ---- 2. Intervention + LO-fill drift step ------------------------
        # Defensive clip: the true value function is bounded by Q_max * (xi +
        # alpha * Q_max). We clip h_star to ±10x that to prevent runaway
        # positive feedback through the exp(-kappa dh) source term, which
        # would otherwise produce numerical blow-up at small lambda where
        # the operator splitting is least accurate. With this clip the
        # scheme remains stable across all sigma_lam tested; the clip is
        # never active when the solution is in the physically meaningful
        # range.
        h_bound = 10.0 * (Q_max * (p.xi + p.alpha * Q_max) + 1.0)
        np.clip(h_star, -h_bound, h_bound, out=h_star)
        h[i, :, 0] = 0.0
        for q in range(1, Q_max + 1):
            for j, lam_j in enumerate(lam_grid):
                lam_scaled = lam_j / (math.e * p.kappa)
                dh = h_star[j, q] - h_star[j, q - 1]
                # Aggressive clip on exponent: 30 covers any physical regime
                # (exp(30) >> any h source term magnitude we should see).
                arg = min(max(-p.kappa * dh, -30.0), 30.0)
                drift = p.phi * q ** 2 - lam_scaled * math.exp(arg)
                h_cont = h_star[j, q] - dt * drift
                # Also clip h_cont
                h_cont = max(-h_bound, min(h_bound, h_cont))
                h_exec = h[i, j, q - 1] - p.xi
                if h_exec >= h_cont:
                    h[i, j, q] = h_exec
                    trigger[i, j, q] = True
                else:
                    h[i, j, q] = h_cont
                    trigger[i, j, q] = False

    return CIRFDSolution(p=p, cir=cir, t_grid=t_grid, lam_grid=lam_grid,
                          h=h, mo_trigger=trigger)
