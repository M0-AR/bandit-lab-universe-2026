"""Base policy interface."""
from __future__ import annotations
import numpy as np


class BanditPolicy:
    """All policies share: reset(k), select(rng) -> int, update(a, x)."""

    name = "base"

    def reset(self, k: int) -> None:
        self.k = k
        self.t = 0

    def select(self, rng: np.random.Generator) -> int:
        raise NotImplementedError

    def update(self, a: int, x: float) -> None:
        self.t += 1
