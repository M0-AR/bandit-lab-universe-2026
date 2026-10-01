"""Adversarial bandits (L&S Ch.11). Oblivious reward matrix x in [0,1]^{n x k}."""
from __future__ import annotations
import numpy as np


class AdversarialBandit:
    def __init__(self, rewards: np.ndarray):
        assert rewards.ndim == 2
        assert np.all((0 <= rewards) & (rewards <= 1))
        self.rewards = rewards
        self.n, self.k = rewards.shape
        self.best_fixed = float(np.max(np.sum(rewards, axis=0)))

    def reward(self, t: int, a: int) -> float:
        return float(self.rewards[t, a])

    @staticmethod
    def switching_two_arm(n: int, seed: int = 0) -> "AdversarialBandit":
        """Ch.31-style non-stationary demo: arm0 good first half, arm1 second half."""
        rng = np.random.default_rng(seed)
        R = np.zeros((n, 2))
        R[: n // 2, 0] = 1.0
        R[n // 2 :, 1] = 1.0
        return AdversarialBandit(R)

    @staticmethod
    def stochastic_like(means: list[float], n: int, seed: int = 0) -> "AdversarialBandit":
        rng = np.random.default_rng(seed)
        k = len(means)
        R = np.zeros((n, k))
        for t in range(n):
            for i, m in enumerate(means):
                R[t, i] = 1.0 if rng.random() < m else 0.0
        return AdversarialBandit(R)
