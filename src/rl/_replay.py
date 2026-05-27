"""Tiny replay buffer used by both DDQN and the (optional off-policy variant
of the) hybrid policy. Single-threaded; not designed for distributed RL."""
from __future__ import annotations
import random
from collections import deque
from typing import Any


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: int | None = None):
        self.buf: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, *transition: Any) -> None:
        self.buf.append(transition)

    def sample(self, batch_size: int):
        return self._rng.sample(self.buf, batch_size)
