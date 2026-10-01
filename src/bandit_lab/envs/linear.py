"""Stochastic linear bandits (L&S Ch.19). X = <theta*, a> + eta, eta 1-subgaussian."""
from __future__ import annotations
import numpy as np


class LinearBandit:
    def __init__(self, arms: np.ndarray, theta: np.ndarray, sigma: float = 1.0):
        self.arms = np.asarray(arms, dtype=float)  # (k,d)
        self.theta = np.asarray(theta, dtype=float)
        assert self.arms.shape[1] == self.theta.shape[0]
        self.sigma = sigma
        self.k = self.arms.shape[0]
        self.means = self.arms @ self.theta
        self.mu_star = float(np.max(self.means))
        self.gaps = self.mu_star - self.means

    def pull(self, a: int, rng: np.random.Generator) -> float:
        return float(self.arms[a] @ self.theta + rng.normal(0, self.sigma))

    @staticmethod
    def hypercube(d: int, theta: np.ndarray, sigma: float = 1.0) -> "LinearBandit":
        import itertools
        arms = np.array(list(itertools.product([-1.0, 1.0], repeat=d)))
        return LinearBandit(arms, theta, sigma)

    @staticmethod
    def unit_ball_sample(k: int, d: int, theta: np.ndarray, seed: int = 0,
                         sigma: float = 1.0) -> "LinearBandit":
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(k, d))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        return LinearBandit(A, theta, sigma)
