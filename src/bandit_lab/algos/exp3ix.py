"""Exp3-IX (L&S Ch.12, Alg.10). Implicit exploration: Yhat = I{A=i}Y/(P_i+gamma)."""
from __future__ import annotations
import math
import numpy as np
from bandit_lab.algos.base import BanditPolicy


class Exp3IX(BanditPolicy):
    name = "Exp3-IX"

    def __init__(self, n: int, eta: float | None = None, gamma: float | None = None):
        self.n = n
        self.eta_in = eta
        self.gamma_in = gamma

    def reset(self, k: int) -> None:
        super().reset(k)
        if self.eta_in is None:
            self.eta = math.sqrt(2.0 * math.log(k + 1) / (self.n * k)) if self.n > 0 else 0.1
        else:
            self.eta = self.eta_in
        self.gamma = self.gamma_in if self.gamma_in is not None else self.eta / 2.0
        self.Lhat = np.zeros(k)
        self.P = np.ones(k) / k

    def _recompute(self):
        z = -self.eta * (self.Lhat - np.min(self.Lhat))
        e = np.exp(z)
        self.P = e / np.sum(e)

    def select(self, rng: np.random.Generator) -> int:
        self._recompute()
        return int(rng.choice(self.k, p=self.P))

    def update(self, a: int, x: float) -> None:
        y = 1.0 - x  # loss
        p = self.P[a]
        # IX estimator for chosen arm only
        self.Lhat[a] += y / (p + self.gamma)
        super().update(a, x)
