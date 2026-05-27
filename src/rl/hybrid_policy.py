"""Agent C — Structure-Aware Hybrid Policy (AMF revision §5.2.3).

The hybrid policy encodes the analytical identity
    delta*(t, q) = 1/kappa + h(t, q) - h(t, q-1)
directly into its parameterisation:

    delta_theta(t, q) = 1/kappa_hat + g_theta(t, q)            # LO depth
    pi_MO(t, q)       = sigma( beta * [ (h_theta(t,q-1) - h_theta(t,q)) - xi ] )

where g_theta, h_theta share a tiny two-hidden-layer trunk.  The residual
head is signed and bounded, so the deployed depth can move below the
1/kappa_hat anchor and snap to the zero-depth grid point near maturity.
g_theta == 0 reproduces the asymptotic anchor; non-trivial g_theta is a
finite-sample correction that the data must justify.

Training: REINFORCE with a learned baseline (h_theta itself).
- Episodes are rolled out under a Gaussian-noised continuous depth, snapped to
  the env's discrete depth grid for execution. The log-prob of the chosen
  discrete action is computed by treating the noise as a Gaussian-density
  over depth grid points, normalised to a categorical.
- The MO gate is a Bernoulli with logit beta*(h(t,q-1) - h(t,q) - xi); the
  REINFORCE log-prob of the chosen branch enters the loss.
- The value head h_theta is fitted by MSE to Monte-Carlo returns G_t in the
  same update (multi-objective loss with coefficients c_v=0.5, c_e=1e-3).
- Every `recompute_kappa_every` episodes, kappa_hat is re-estimated from the
  agent's own visited (delta, fill) pairs to keep the sample budget honest.
"""
from __future__ import annotations
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..common.estimators import fit_kappa_lambda_mle
from .env import LiquidationEnv


class HybridPolicyNet(nn.Module):
    """Shared two-hidden-layer trunk + g_head + h_head + log_beta.

    Total params: 2*16 + 16 + 16*16 + 16 + 17 + 17 + 1 = 339.
    """
    def __init__(self, hidden: int = 16, input_dim: int = 2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.g_head = nn.Linear(hidden, 1)
        self.h_head = nn.Linear(hidden, 1)
        self.log_beta = nn.Parameter(torch.tensor(math.log(50.0)))

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(feat)
        # The deployed depth is `1/kappa + g`, where g is a SMALL signed
        # correction. Empirically the optimal residual g(t, q) for the CJP
        # §8 problem is in [-0.005, +0.020]. We tanh-bound g into [-0.01, +0.04]
        # (a 4x slack around the truth) so it can't blow up to depth values
        # outside the depth grid [0, 0.05]. At init the raw Linear output is
        # ~N(0, 1), tanh -> [-1, 1], scaled -> g ~ [-0.01, +0.04], giving
        # initial delta = 1/kappa + g in [0, 0.05] — well-spread on the grid.
        g = 0.025 * (1.0 + torch.tanh(self.g_head(z).squeeze(-1))) - 0.01
        h = self.h_head(z).squeeze(-1)      # value baseline
        return g, h


class HybridAgent:
    def __init__(
        self,
        env: LiquidationEnv,
        kappa_hat: float,
        *,
        hidden: int = 16,
        # Plan A: lr 3e-3 was unstable under REINFORCE high-variance gradients;
        # 3e-4 (Adam default for noisy objectives) gives much smoother decay.
        lr: float = 3e-4,
        sigma_explore_start: float = 0.01,
        sigma_explore_end: float = 1e-4,
        # Plan A: batch 32 was variance-dominated; 128 cuts gradient stderr by 2x
        # and aligns with the regimes where REINFORCE actually converges.
        batch_episodes: int = 128,
        grad_clip: float = 1.0,
        # Plan A: c_v boosted from 0.5 -> 5.0 so h_theta fits the FD target
        # within the same wall budget as the policy head. Without this h
        # learns ~3x slower than g and policy_grid uses a stale h for MO
        # trigger logits, locking the MO action off.
        c_v: float = 5.0,
        # Plan A: MO entropy bonus was actively destabilising the rare-event
        # MO Bernoulli — dropped (c_e = 0). MO exploration now comes from the
        # learned h_theta + the sigma_explore noise on delta.
        c_e: float = 0.0,
        # Plan B: clipped-PPO mode. When `ppo_epochs > 1` each batch of
        # rollouts is reused for K=ppo_epochs gradient passes under the
        # clipped surrogate objective, giving 3-5x more learning per
        # episode at the cost of a 30-line reimplementation. With
        # ppo_epochs=1 the loss reduces to REINFORCE (with normalised
        # advantage) — that is the Plan A default.
        ppo_epochs: int = 4,
        ppo_clip: float = 0.2,
        recompute_kappa_every: int | None = None,
        seed: int | None = None,
    ):
        self.env = env
        self.p = env.p
        self.kappa_hat = float(kappa_hat)
        self.sigma_start = float(sigma_explore_start)
        self.sigma_end = float(sigma_explore_end)
        self.batch_episodes = int(batch_episodes)
        self.grad_clip = float(grad_clip)
        self.c_v = float(c_v)
        self.c_e = float(c_e)
        self.ppo_epochs = int(ppo_epochs)
        self.ppo_clip = float(ppo_clip)
        self.recompute_kappa_every = recompute_kappa_every
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        input_dim = 3 if env.cfg.state_mode == "tql" else 2
        self.net = HybridPolicyNet(hidden=hidden, input_dim=input_dim)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)
        # Cached fill-pair buffer for online kappa re-estimation
        self._delta_buf: list[float] = []
        self._fill_buf: list[float] = []

    # ----- featurisation ---------------------------------------------------
    def _feat(self, state) -> torch.Tensor:
        return torch.from_numpy(self.env.state_to_feat(state)).unsqueeze(0)

    def _feat_minus_q(self, state) -> torch.Tensor:
        s = list(state)
        s[1] = max(s[1] - 1, 0)
        return torch.from_numpy(self.env.state_to_feat(tuple(s))).unsqueeze(0)

    # ----- action sampling -------------------------------------------------
    def act(self, state, *, stochastic: bool = True, sigma: float = 0.0,
            mo_explore: float = 0.0):
        """Returns (action_idx, info_dict). info_dict carries log_p_delta,
        log_p_mo, value_hat for the REINFORCE update.

        `mo_explore` (only active when stochastic=True) is a forced MO
        exploration probability that is mixed with the network's gate
        output: effective p_MO = (1 - mo_explore) * p_MO_net + mo_explore.
        This is essential at training start because h_theta is initialized
        near zero, so the network's mo_logit = beta*(h_m - h_t - xi)
        evaluates to ~-beta*xi < 0 and the gate never fires without the
        forced bonus, starving the agent of MO experience.
        """
        feat = self._feat(state)
        feat_m = self._feat_minus_q(state)
        g_t, h_t = self.net(feat)
        g_m, h_m = self.net(feat_m)
        delta_mean = (1.0 / self.kappa_hat) + g_t
        beta = self.net.log_beta.exp().clamp(max=1000.0)
        mo_logit = beta * ((h_m - h_t) - self.p.xi)
        p_mo_net = torch.sigmoid(mo_logit).clamp(min=1e-6, max=1 - 1e-6)
        if stochastic and mo_explore > 0.0:
            # Mixture: epsilon-greedy on top of the network's gate
            p_mo = (1.0 - mo_explore) * p_mo_net + mo_explore
        else:
            p_mo = p_mo_net

        # MO Bernoulli draw
        if stochastic:
            mo_sample = bool(torch.bernoulli(p_mo).item())
        else:
            mo_sample = bool(p_mo.item() > 0.5)
        # log-prob under the network's p_mo (not the mixture) — REINFORCE
        # treats the exploration noise as off-policy and we accept the
        # standard small bias for stability.
        log_p_mo = torch.log(p_mo_net if mo_sample else 1 - p_mo_net)[0]

        if mo_sample:
            action_idx = self.env.cfg.n_depth
            log_p_delta = torch.tensor(0.0)
        else:
            # Categorical over depth grid points via Gaussian density
            depth_choices = torch.from_numpy(self.env.depth_grid.astype(np.float32))
            sig = max(sigma, 1e-4)
            # log-densities (constant 1/(sqrt(2 pi sigma)) drops out after softmax)
            log_density = -((depth_choices - delta_mean) ** 2) / (2.0 * sig * sig)
            log_probs = log_density - torch.logsumexp(log_density, dim=0)
            probs = log_probs.exp()
            if stochastic:
                action_idx = int(torch.multinomial(probs, 1).item())
            else:
                action_idx = int(torch.argmax(probs).item())
            log_p_delta = log_probs[action_idx]

        info = {
            "log_p_delta": log_p_delta,
            "log_p_mo": log_p_mo,
            "value_hat": h_t.squeeze(),
            "delta_mean": float(delta_mean.item()),
            "p_mo": float(p_mo.item()),
        }
        return action_idx, info

    # ----- log-prob recomputation for PPO ---------------------------------
    def _log_prob_and_value_for_action(self, state, action_idx: int,
                                        sigma: float):
        """Re-compute log pi(action | state) and h_theta(state) under the
        CURRENT network. Used by the PPO inner loop.

        Returns (log_p_delta, log_p_mo, h_value).
        """
        feat = self._feat(state)
        feat_m = self._feat_minus_q(state)
        g_t, h_t = self.net(feat)
        _, h_m = self.net(feat_m)
        delta_mean = (1.0 / self.kappa_hat) + g_t
        beta = self.net.log_beta.exp().clamp(max=1000.0)
        mo_logit = beta * ((h_m - h_t) - self.p.xi)
        p_mo = torch.sigmoid(mo_logit).clamp(min=1e-6, max=1 - 1e-6)

        if action_idx == self.env.cfg.n_depth:
            # MO branch
            log_p_mo = torch.log(p_mo)[0]
            log_p_delta = torch.tensor(0.0)
        else:
            log_p_mo = torch.log(1 - p_mo)[0]
            depth_choices = torch.from_numpy(self.env.depth_grid.astype(np.float32))
            sig = max(sigma, 1e-4)
            log_density = -((depth_choices - delta_mean) ** 2) / (2.0 * sig * sig)
            log_probs = log_density - torch.logsumexp(log_density, dim=0)
            log_p_delta = log_probs[action_idx]
        return log_p_delta, log_p_mo, h_t.squeeze()

    # ----- policy / value grid for diagnostics ----------------------------
    @torch.no_grad()
    def value_grid(self, env=None) -> np.ndarray:
        env = env or self.env
        nt = env.num_time_buckets
        Q = env.p.Q0
        V = np.zeros((nt, Q + 1))
        for i in range(nt):
            for q in range(Q + 1):
                feat = torch.from_numpy(env.state_to_feat((i, q))).unsqueeze(0)
                _, h = self.net(feat)
                V[i, q] = float(h.squeeze().item())
        return V

    @torch.no_grad()
    def policy_grid(self, env=None) -> tuple[np.ndarray, np.ndarray]:
        env = env or self.env
        nt = env.num_time_buckets
        Q = env.p.Q0
        depth = np.zeros((nt, Q + 1))
        trigger = np.zeros((nt, Q + 1), dtype=bool)
        depth_choices = env.depth_grid
        for i in range(nt):
            for q in range(1, Q + 1):
                feat = torch.from_numpy(env.state_to_feat((i, q))).unsqueeze(0)
                feat_m = torch.from_numpy(env.state_to_feat((i, q - 1))).unsqueeze(0)
                g_t, h_t = self.net(feat)
                _, h_m = self.net(feat_m)
                beta = self.net.log_beta.exp().clamp(max=1000.0)
                mo_logit = beta * ((h_m - h_t) - env.p.xi)
                if torch.sigmoid(mo_logit).item() > 0.5:
                    trigger[i, q] = True
                else:
                    # Snap the (possibly negative) raw mean `1/kappa + g` to
                    # the nearest valid depth grid point — same discretisation
                    # as `act()` uses during training, so eval and train see
                    # the same effective policy.
                    d_raw = (1.0 / self.kappa_hat) + g_t.squeeze().item()
                    a_idx = int(np.argmin(np.abs(depth_choices - d_raw)))
                    depth[i, q] = float(depth_choices[a_idx])
        return depth, trigger

    # ----- training --------------------------------------------------------
    def train(
        self,
        n_episodes: int,
        *,
        h_reference: np.ndarray | None = None,
        log_every: int = 200,
    ) -> dict:
        env = self.env
        all_returns = np.zeros(n_episodes)
        # Per-step rollout buffer: states + actions + old log-probs + returns
        # are reused for self.ppo_epochs gradient passes per batch.
        buf_states: list[tuple] = []
        buf_actions: list[int] = []
        buf_sigmas: list[float] = []
        buf_old_logp: list[float] = []        # log(pi_old(a|s)) sum delta+mo
        buf_returns: list[float] = []

        rmse_trace: list[tuple[int, float]] = []

        for ep in range(n_episodes):
            frac = ep / max(1, n_episodes - 1)
            sigma = self.sigma_start + (self.sigma_end - self.sigma_start) * frac
            # MO exploration probability: start at 5% per step, decay to 0.
            # Without this the network's gate is always off (h_theta init near
            # zero implies p_MO_net = sigmoid(-beta*xi) ~ 0), and the agent
            # never collects MO experience.
            mo_explore = 0.05 * (1.0 - frac)

            state = env.reset()
            ep_states: list[tuple] = []
            ep_actions: list[int] = []
            ep_old_logp: list[float] = []
            ep_rewards: list[float] = []
            done = False
            while not done:
                a, info = self.act(state, stochastic=True, sigma=sigma,
                                    mo_explore=mo_explore)
                # Snapshot the rollout-time log-prob (no grad through opt)
                logp_old = float(info["log_p_delta"].detach() + info["log_p_mo"].detach())
                ep_states.append(tuple(state))
                ep_actions.append(int(a))
                ep_old_logp.append(logp_old)
                ns, r, done, step_info = env.step(a)
                if a != env.cfg.n_depth:
                    self._delta_buf.append(info["delta_mean"])
                    self._fill_buf.append(1.0 if step_info["lo_fill"] else 0.0)
                ep_rewards.append(r)
                state = ns

            G = float(sum(ep_rewards))
            all_returns[ep] = G
            steps_per_ep = len(ep_rewards)
            remaining = np.cumsum(ep_rewards[::-1])[::-1]
            for t_step in range(steps_per_ep):
                buf_states.append(ep_states[t_step])
                buf_actions.append(ep_actions[t_step])
                buf_sigmas.append(float(sigma))
                buf_old_logp.append(ep_old_logp[t_step])
                buf_returns.append(float(remaining[t_step]))

            # ---- PPO update every batch_episodes -------------------------
            if (ep + 1) % self.batch_episodes == 0:
                returns_t = torch.tensor(buf_returns, dtype=torch.float32)
                old_logp_t = torch.tensor(buf_old_logp, dtype=torch.float32)
                # K-epoch PPO. The first epoch is essentially REINFORCE
                # because new_logp == old_logp -> ratio == 1; subsequent
                # epochs do the actual policy improvement.
                final_loss_pi = final_loss_v = None
                final_adv_std = 0.0
                for _epoch in range(max(self.ppo_epochs, 1)):
                    # Recompute log-pi and h under current params
                    new_logp_list: list[torch.Tensor] = []
                    val_list: list[torch.Tensor] = []
                    for s, a, sg in zip(buf_states, buf_actions, buf_sigmas):
                        lp_d, lp_m, h_v = self._log_prob_and_value_for_action(s, a, sg)
                        new_logp_list.append(lp_d + lp_m)
                        val_list.append(h_v)
                    new_logp = torch.stack(new_logp_list)
                    val_t = torch.stack(val_list)
                    adv = (returns_t - val_t.detach())
                    adv_std = adv.std().clamp_min(1e-8)
                    adv_norm = (adv - adv.mean()) / adv_std
                    final_adv_std = float(adv_std)
                    # Clipped surrogate objective
                    ratio = (new_logp - old_logp_t).exp()
                    surr1 = ratio * adv_norm
                    surr2 = torch.clamp(ratio,
                                         1.0 - self.ppo_clip,
                                         1.0 + self.ppo_clip) * adv_norm
                    loss_pi = -torch.min(surr1, surr2).mean()
                    loss_v = ((val_t - returns_t) ** 2).mean()
                    loss = loss_pi + self.c_v * loss_v
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                    self.opt.step()
                    with torch.no_grad():
                        self.net.log_beta.clamp_(min=math.log(1.0), max=math.log(1000.0))
                    final_loss_pi, final_loss_v = float(loss_pi), float(loss_v)

                if h_reference is not None and (ep + 1) % log_every == 0:
                    print(f"  [hybrid ep {ep+1:>5}] G={float(returns_t.mean()):+.4f} "
                          f"adv_std={final_adv_std:.3f} "
                          f"loss_pi={final_loss_pi:+.4f} "
                          f"loss_v={final_loss_v:.4f} "
                          f"beta={float(self.net.log_beta.exp()):.1f}",
                          flush=True)

                buf_states.clear(); buf_actions.clear(); buf_sigmas.clear()
                buf_old_logp.clear(); buf_returns.clear()

            # ---- online kappa re-estimation -------------------------------
            if (self.recompute_kappa_every is not None
                    and len(self._delta_buf) >= 200
                    and (ep + 1) % self.recompute_kappa_every == 0):
                deltas = np.asarray(self._delta_buf, dtype=np.float64)
                fills = np.asarray(self._fill_buf, dtype=np.float64)
                out = fit_kappa_lambda_mle(deltas, fills, dt=env.cfg.dt,
                                            kappa_prior=self.kappa_hat,
                                            lam_prior=env.p.lam,
                                            ridge_log_kappa=1.0)
                self.kappa_hat = float(out["kappa_hat"])
                # Cap the buffer to keep memory bounded
                if len(self._delta_buf) > 4000:
                    self._delta_buf = self._delta_buf[-2000:]
                    self._fill_buf = self._fill_buf[-2000:]

            # ---- diagnostics --------------------------------------------
            if h_reference is not None and (ep + 1) % log_every == 0:
                V = self.value_grid()
                fd_t = np.linspace(0.0, env.p.T, h_reference.shape[0])
                idx = np.clip(np.searchsorted(fd_t, env.t_grid),
                              0, len(fd_t) - 1)
                err = float(np.sqrt(np.mean(
                    (V[:-1, 1:] - h_reference[idx][:-1, 1:]) ** 2
                )))
                rmse_trace.append((ep + 1, err))

        return {"returns": all_returns, "rmse_trace": rmse_trace,
                "kappa_hat_final": self.kappa_hat}
