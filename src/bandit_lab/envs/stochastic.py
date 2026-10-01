"""Stochastic bandits (L&S Ch.4)."""
from __future__ import annotations
import numpy as np


class BernoulliBandit:
    """E_B^k: X ~ B(mu_a)."""

    def __init__(self, means: list[float] | np.ndarray):
        self.means = np.asarray(means, dtype=float)
        assert self.means.ndim == 1 and np.all((0 <= self.means) & (self.means <= 1))
        self.k = len(self.means)
        self.mu_star = float(np.max(self.means))
        self.gaps = self.mu_star - self.means

    def pull(self, a: int, rng: np.random.Generator) -> float:
        return 1.0 if rng.random() < self.means[a] else 0.0


class GaussianBandit:
    """E_N^k(sigma^2): X ~ N(mu_a, sigma^2). 1-subgaussian when sigma<=1."""

    def __init__(self, means: list[float] | np.ndarray, sigma: float = 1.0):
        self.means = np.asarray(means, dtype=float)
        self.sigma = sigma
        self.k = len(self.means)
        self.mu_star = float(np.max(self.means))
        self.gaps = self.mu_star - self.means

    def pull(self, a: int, rng: np.random.Generator) -> float:
        return float(rng.normal(self.means[a], self.sigma))
