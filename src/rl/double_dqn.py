"""Double-DQN agent (PyTorch). Function-approximation complement to the
tabular Q-learner. Input features = [t_idx / num_time_buckets, q / Q0].

Migrated from `src.task3_rl.agent_dqn` (Phase 1, AMF revision).
"""
from __future__ import annotations
import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .env import LiquidationEnv


class QNet(nn.Module):
    def __init__(self, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def state_to_feat(env: LiquidationEnv, state):
    i, q = state
    return np.array([i / max(1, env.num_time_buckets - 1),
                     q / max(1, env.p.Q0)], dtype=np.float32)


class DDQN:
    def __init__(self, env: LiquidationEnv, seed: int | None = None,
                 lr: float = 1e-3, gamma: float = 1.0,
                 buffer: int = 50000, batch: int = 128,
                 target_sync: int = 500):
        torch.manual_seed(seed or 0)
        np.random.seed(seed or 0)
        random.seed(seed or 0)
        self.env = env
        self.gamma = gamma
        self.batch = batch
        self.target_sync = target_sync
        self.qnet = QNet(env.num_actions)
        self.target = copy.deepcopy(self.qnet)
        self.opt = optim.Adam(self.qnet.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer)

    def act(self, state, eps, mo_explore: float = 0.0):
        if state[1] > 0 and mo_explore > 0.0 and random.random() < mo_explore:
            return self.env.cfg.n_depth
        if random.random() < eps:
            return random.randrange(self.env.num_actions)
        with torch.no_grad():
            f = torch.from_numpy(state_to_feat(self.env, state)).unsqueeze(0)
            return int(torch.argmax(self.qnet(f), dim=-1).item())

    def value_grid(self, env=None) -> np.ndarray:
        """V(t_idx, q) = max_a Q_net(feat(t_idx, q))[a]. Shape (n_t, Q0+1)."""
        env = env or self.env
        nt = env.num_time_buckets
        Q = env.p.Q0
        V = np.zeros((nt, Q + 1))
        with torch.no_grad():
            for i in range(nt):
                for q in range(Q + 1):
                    f = torch.from_numpy(
                        np.array([i / max(1, nt - 1), q / max(1, Q)],
                                 dtype=np.float32)
                    ).unsqueeze(0)
                    V[i, q] = float(torch.max(self.qnet(f)).item())
        return V

    def greedy_policy_grid(self):
        env = self.env
        nt = env.num_time_buckets
        Q = env.p.Q0
        depth = np.zeros((nt, Q + 1))
        trigger = np.zeros((nt, Q + 1), dtype=bool)
        with torch.no_grad():
            for i in range(nt):
                for q in range(1, Q + 1):
                    f = torch.from_numpy(
                        np.array([i / max(1, nt - 1), q / max(1, Q)],
                                 dtype=np.float32)
                    ).unsqueeze(0)
                    a = int(torch.argmax(self.qnet(f), dim=-1).item())
                    if a == env.cfg.n_depth:
                        trigger[i, q] = True
                    else:
                        depth[i, q] = float(env.depth_grid[a])
        return depth, trigger

    def train(self, n_episodes: int = 5000,
              eps_start: float = 1.0, eps_end: float = 0.05,
              mo_explore_start: float = 0.0, mo_explore_end: float = 0.0,
              log_every: int = 500):
        env = self.env
        returns = np.zeros(n_episodes)
        step_count = 0
        for ep in range(n_episodes):
            frac = ep / max(1, n_episodes - 1)
            eps = eps_start * (eps_end / eps_start) ** frac
            mo_explore = mo_explore_start + (mo_explore_end - mo_explore_start) * frac

            state = env.reset()
            done = False
            total = 0.0
            while not done:
                a = self.act(state, eps, mo_explore=mo_explore)
                ns, r, done, _ = env.step(a)
                self.buffer.append((state, a, r, ns, done))
                state = ns
                total += r
                step_count += 1
                if len(self.buffer) >= self.batch:
                    self._learn()
                if step_count % self.target_sync == 0:
                    self.target.load_state_dict(self.qnet.state_dict())
            returns[ep] = total
            if (ep + 1) % log_every == 0:
                recent = np.mean(returns[max(0, ep - log_every + 1):ep + 1])
                print(f"[DDQN] ep {ep+1}/{n_episodes} mean_return={recent:.4f} eps={eps:.3f}")
        return returns

    def _learn(self):
        batch = random.sample(self.buffer, self.batch)
        s = torch.from_numpy(np.stack([state_to_feat(self.env, b[0]) for b in batch]))
        a = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        ns = torch.from_numpy(np.stack([state_to_feat(self.env, b[3]) for b in batch]))
        done = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        q_pred = self.qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_next = torch.argmax(self.qnet(ns), dim=-1)
            q_next = self.target(ns).gather(1, a_next.unsqueeze(1)).squeeze(1)
            q_target = r + self.gamma * (1 - done) * q_next
        loss = nn.functional.smooth_l1_loss(q_pred, q_target)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 5.0)
        self.opt.step()
