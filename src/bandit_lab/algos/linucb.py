"""LinUCB / OFUL (L&S Ch.19). Ridge least-squares + ellipsoidal bonus.

At = argmax_a <theta_hat, a> + sqrt(beta) * ||a||_{V^{-1}}   (Eq.19.13)
V_t = lambda I + sum A_s A_s^T ; theta_hat = V^{-1} sum A_s X_s.
Default beta from Thm 20.5 / Eq.19.8 with data-dependent log-det upgrade.
"""
from __future__ import annotations
import math
import numpy as np
from bandit_lab.algos.base import BanditPolicy


class LinUCB(BanditPolicy):
    name = "LinUCB"

    def __init__(self, d: int, lam: float = 1.0, m2: float = 1.0,
                 L: float = 1.0, delta: float = 1e-3, beta: float | None = None):
        self.d = d
        self.lam = lam
        self.m2 = m2
        self.L = L
        self.delta = delta
        self.beta_fixed = beta

    def reset(self, k: int = 0) -> None:
        super().reset(k)
        self.V = self.lam * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.theta = np.zeros(self.d)
        self.t = 0

    def _beta(self, n_hint: int = 10_000) -> float:
        if self.beta_fixed is not None:
            return self.beta_fixed
        # Eq.19.8 worst-case upper envelope (conservative, n_hint for bound only)
        det_ratio = (self.lam * self.d + n_hint * self.L ** 2) / (self.lam * self.d)
        s = math.sqrt(self.lam) * self.m2 + math.sqrt(
            2 * math.log(1 / self.delta) + self.d * math.log(max(1.0, det_ratio)))
        return s ** 2

    def select_arm(self, arms: np.ndarray, rng: np.random.Generator) -> int:
        """arms: (k,d) matrix. Returns index."""
        Vinv = np.linalg.inv(self.V)
        beta = self._beta()
        sb = math.sqrt(beta)
        scores = arms @ self.theta + sb * np.sqrt(np.einsum("ij,jk,ik->i", arms, Vinv, arms))
        best = float(np.max(scores))
        cand = [i for i in range(len(scores)) if scores[i] == best]
        return int(rng.choice(cand))

    # compat with finite-arm loop (arms = one-hot not used here)
    def select(self, rng: np.random.Generator) -> int:
        raise NotImplementedError("Use select_arm(arms, rng) for LinUCB")

    def update_ctx(self, a_vec: np.ndarray, x: float) -> None:
        self.V += np.outer(a_vec, a_vec)
        self.b += a_vec * x
        self.theta = np.linalg.solve(self.V, self.b)
        self.t += 1
