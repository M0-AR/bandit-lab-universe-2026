"""UCB family (L&S Ch.7-8)."""
from __future__ import annotations
import math
import numpy as np
from bandit_lab.algos.base import BanditPolicy


class UCB(BanditPolicy):
    """UCB(delta): UCB_i = mu_hat + sqrt(2 log(1/delta)/T_i). Thm 7.1."""

    name = "UCB"

    def __init__(self, delta: float | None = None, sigma: float = 1.0):
        self.delta = delta  # if None, set to 1/n^2 at reset via n_hint
        self.sigma = sigma
        self.n_hint = 1000

    def reset(self, k: int, n_hint: int = 1000) -> None:
        super().reset(k)
        self.n_hint = n_hint
        if self.delta is None:
            self._delta = 1.0 / (n_hint ** 2)
        else:
            self._delta = self.delta
        self.counts = np.zeros(k, dtype=int)
        self.sums = np.zeros(k)

    def _bonus(self, n_i: int) -> float:
        if n_i == 0:
            return float("inf")
        return self.sigma * math.sqrt(2.0 * math.log(1.0 / self._delta) / n_i)

    def select(self, rng: np.random.Generator) -> int:
        means = np.array([
            self.sums[i] / self.counts[i] if self.counts[i] > 0 else 0.0
            for i in range(self.k)
        ])
        ucb = np.array([means[i] + self._bonus(int(self.counts[i])) for i in range(self.k)])
        best = float(np.max(ucb))
        cand = [i for i in range(self.k) if ucb[i] == best]
        return int(rng.choice(cand))

    def update(self, a: int, x: float) -> None:
        self.counts[a] += 1
        self.sums[a] += x
        super().update(a, x)


class AsymptoticallyOptimalUCB(BanditPolicy):
    """Alg.6: UCB with f(t)=1+t log^2(t). Thm 8.1, limsup R/log n <= sum 2/Delta."""

    name = "UCB*"

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def reset(self, k: int) -> None:
        super().reset(k)
        self.counts = np.zeros(k, dtype=int)
        self.sums = np.zeros(k)
        # play each arm once first (standard)
        self._init_left = k

    @staticmethod
    def f(t: int) -> float:
        t = max(t, 2)
        return 1.0 + t * (math.log(t) ** 2)

    def select(self, rng: np.random.Generator) -> int:
        if self._init_left > 0:
            return self.k - self._init_left
        t = self.t + 1
        lf = math.log(self.f(t))
        ucb = []
        for i in range(self.k):
            m = self.sums[i] / max(1, self.counts[i])
            ucb.append(m + self.sigma * math.sqrt(2.0 * lf / max(1, self.counts[i])))
        best = max(ucb)
        return int(rng.choice([i for i in range(self.k) if ucb[i] == best]))

    def update(self, a: int, x: float) -> None:
        if self._init_left > 0:
            self._init_left -= 1
        self.counts[a] += 1
        self.sums[a] += x
        super().update(a, x)
