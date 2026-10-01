"""Thompson Sampling (L&S Ch.36, Alg.23/25)."""
from __future__ import annotations
import numpy as np
from bandit_lab.algos.base import BanditPolicy


class ThompsonBernoulli(BanditPolicy):
    """Beta(1,1) prior per arm, Bernoulli likelihood. Posterior Beta(a+s, b+f)."""

    name = "TS-Bernoulli"

    def reset(self, k: int) -> None:
        super().reset(k)
        self.alpha = np.ones(k)
        self.beta = np.ones(k)

    def select(self, rng: np.random.Generator) -> int:
        samples = rng.beta(self.alpha, self.beta)
        best = float(np.max(samples))
        cand = [i for i in range(self.k) if samples[i] == best]
        return int(rng.choice(cand))

    def update(self, a: int, x: float) -> None:
        # x in {0,1}; tolerate continuous x by randomized rounding
        s = 1 if (x >= 0.5 if isinstance(x, (int, float)) else x) else 0
        if isinstance(x, float) and 0 < x < 1:
            s = 1 if np.random.rand() < x else 0
        self.alpha[a] += s
        self.beta[a] += 1 - s
        super().update(a, x)


class ThompsonGaussian(BanditPolicy):
    """Gaussian N(0, v0) prior, N(theta, sigma^2) likelihood, known sigma.

    Posterior per arm: N(mu_n, s2_n) with precision update.
    Matches Thm 36.3-style diffuse prior when v0 large.
    """

    name = "TS-Gaussian"

    def __init__(self, sigma: float = 1.0, v0: float = 10.0):
        self.sigma = sigma
        self.v0 = v0

    def reset(self, k: int) -> None:
        super().reset(k)
        self.mu = np.zeros(k)
        self.var = np.ones(k) * self.v0
        self.counts = np.zeros(k, dtype=int)

    def select(self, rng: np.random.Generator) -> int:
        samples = rng.normal(self.mu, np.sqrt(self.var))
        best = float(np.max(samples))
        return int(rng.choice([i for i in range(self.k) if samples[i] == best]))

    def update(self, a: int, x: float) -> None:
        prec0 = 1.0 / self.var[a]
        prec_obs = 1.0 / (self.sigma ** 2)
        prec1 = prec0 + prec_obs
        self.mu[a] = (prec0 * self.mu[a] + prec_obs * x) / prec1
        self.var[a] = 1.0 / prec1
        self.counts[a] += 1
        super().update(a, x)
