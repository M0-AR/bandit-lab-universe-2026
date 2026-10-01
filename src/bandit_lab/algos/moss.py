"""MOSS (L&S Ch.9, Alg.7). Minimax optimal: R_n <= 39 sqrt(kn)+sum Delta."""
from __future__ import annotations
import math
import numpy as np
from bandit_lab.algos.base import BanditPolicy


def logp(x: float) -> float:
    return math.log(max(1.0, x))


class MOSS(BanditPolicy):
    name = "MOSS"

    def __init__(self, n: int, sigma: float = 1.0):
        assert n >= 1
        self.n = n
        self.sigma = sigma

    def reset(self, k: int) -> None:
        super().reset(k)
        self.counts = np.zeros(k, dtype=int)
        self.sums = np.zeros(k)
        self._init_left = k

    def select(self, rng: np.random.Generator) -> int:
        if self._init_left > 0:
            return self.k - self._init_left
        idx = []
        for i in range(self.k):
            m = self.sums[i] / max(1, self.counts[i])
            bonus = self.sigma * math.sqrt(
                4.0 / max(1, self.counts[i]) * logp(self.n / (self.k * max(1, self.counts[i])))
            )
            idx.append(m + bonus)
        best = max(idx)
        return int(rng.choice([i for i in range(self.k) if idx[i] == best]))

    def update(self, a: int, x: float) -> None:
        if self._init_left > 0:
            self._init_left -= 1
        self.counts[a] += 1
        self.sums[a] += x
        super().update(a, x)
