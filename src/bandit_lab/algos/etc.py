"""Explore-Then-Commit (Lattimore & Szepesvari Ch.6, Alg.1)."""
from __future__ import annotations
import numpy as np
from bandit_lab.algos.base import BanditPolicy


class ETC(BanditPolicy):
    """Play each arm m times, then commit to empirical best.

    Theorem 6.1: R_n <= m*sum(Delta) + (n-mk)*sum(Delta*exp(-m*Delta^2/4))
    for 1-subgaussian noise.
    """

    name = "ETC"

    def __init__(self, m: int):
        assert m >= 1
        self.m = m

    def reset(self, k: int) -> None:
        super().reset(k)
        self.counts = np.zeros(k, dtype=int)
        self.sums = np.zeros(k)
        self.committed = -1

    def select(self, rng: np.random.Generator) -> int:
        if self.t < self.m * self.k:
            return int(self.t % self.k)
        if self.committed < 0:
            means = self.sums / np.maximum(1, self.counts)
            best = float(np.max(means))
            cand = [i for i in range(self.k) if means[i] == best]
            self.committed = int(rng.choice(cand))
        return self.committed

    def update(self, a: int, x: float) -> None:
        self.counts[a] += 1
        self.sums[a] += x
        super().update(a, x)

    @staticmethod
    def oracle_m(n: int, k: int, delta: float) -> int:
        """Gap-aware oracle tuning Eq.(6.5): m = ceil(4/D^2 log(nD^2/4))."""
        import math
        if delta <= 0:
            return max(1, int(round(n ** (2 / 3) / k)))
        v = 4.0 / (delta ** 2) * math.log(max(math.e, n * delta ** 2 / 4.0))
        return int(min(max(1, math.ceil(v)), n // k))
