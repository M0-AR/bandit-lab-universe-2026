"""KL-UCB for Bernoulli noise (L&S Ch.10, Alg.8).

Index: max{ q : d(p_hat, q) <= log f(t)/T_i }, d = Bernoulli KL.
Newton/bisection inversion; matches Thm 10.6 asymptotic rate
limsup R/log n <= sum Delta/d(mu_i, mu*).
"""
from __future__ import annotations
import math
import numpy as np
from bandit_lab.algos.base import BanditPolicy


def kl_bernoulli(p: float, q: float) -> float:
    eps = 1e-15
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_ucb_upper(p_hat: float, n_i: int, logf_over_n: float, tol: float = 1e-7) -> float:
    if n_i == 0:
        return 1.0
    if p_hat >= 1.0:
        return 1.0
    # solve d(p_hat, q) = rhs for q in [p_hat, 1]
    rhs = logf_over_n
    lo, hi = p_hat, 1.0
    # if even q=1 is within rhs, return 1
    if kl_bernoulli(p_hat, 1.0 - 1e-12) <= rhs:
        return 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(p_hat, mid) <= rhs:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


class KLUCB(BanditPolicy):
    name = "KL-UCB"

    def reset(self, k: int) -> None:
        super().reset(k)
        self.counts = np.zeros(k, dtype=int)
        self.sums = np.zeros(k)
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
        ups = []
        for i in range(self.k):
            p = self.sums[i] / max(1, self.counts[i])
            ups.append(kl_ucb_upper(p, int(self.counts[i]), lf / max(1, self.counts[i])))
        best = max(ups)
        return int(rng.choice([i for i in range(self.k) if ups[i] == best]))

    def update(self, a: int, x: float) -> None:
        if self._init_left > 0:
            self._init_left -= 1
        self.counts[a] += 1
        self.sums[a] += x
        super().update(a, x)
