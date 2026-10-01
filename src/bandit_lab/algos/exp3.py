"""Exp3 (L&S Ch.11, Alg.9) with loss-based IX-free estimator.

Loss estimator: Yhat = 1 - I{A=i}(1-X)/P_i  (Eq.11.6), bounded <=1.
Thm 11.2: R <= sqrt(2 n k log k) with eta = sqrt(2 log k/(nk)).
"""
from __future__ import annotations
import math
import numpy as np
from bandit_lab.algos.base import BanditPolicy


class Exp3(BanditPolicy):
    name = "Exp3"

    def __init__(self, n: int, eta: float | None = None):
        self.n = n
        self.eta_in = eta

    def reset(self, k: int) -> None:
        super().reset(k)
        self.k = k
        if self.eta_in is None:
            self.eta = math.sqrt(2.0 * math.log(k) / (self.n * k)) if self.n > 0 else 0.1
        else:
            self.eta = self.eta_in
        self.Shat = np.zeros(k)  # cumulative estimated *rewards*
        self.P = np.ones(k) / k

    def _recompute(self):
        z = self.eta * (self.Shat - np.max(self.Shat))
        e = np.exp(z)
        self.P = e / np.sum(e)

    def select(self, rng: np.random.Generator) -> int:
        self._recompute()
        return int(rng.choice(self.k, p=self.P))

    def update(self, a: int, x: float) -> None:
        # x in [0,1] reward; loss Y=1-x; estimator per Alg.9 line 6
        p = max(self.P[a], 1e-12)
        xhat = np.zeros(self.k)
        # only chosen arm gets non-trivial update; others get 1 (loss 0 -> reward 1)
        for i in range(self.k):
            if i == a:
                xhat[i] = 1.0 - (1.0 - x) / p
            else:
                xhat[i] = 1.0  # unbiased? see proof: E[sum P Xhat]=E[X]; keep vector form
                # Correct incremental form: Shat_i += 1 - I{A=i}(1-X)/P_i
        # Actually implement exactly Alg.9:
        # Shat_i <- Shat_i + 1 - I{A=i}(1-X)/P_i
        self.Shat[a] += 1.0 - (1.0 - x) / p
        for i in range(self.k):
            if i != a:
                self.Shat[i] += 1.0
        super().update(a, x)
