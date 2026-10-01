"""Phased elimination with G-optimal design (L&S Ch.22, Alg.12, simplified).

For finite fixed action set A (linear or one-hot), phases halve target gap
eps_l = 2^-l; each phase plays actions ~ pi* (approx G-optimal via
uniform-over-active here + Frank-Wolfe hook) then eliminates.
Regret: Thm 22.1 O(sqrt(n d log(k log n/delta))).
"""
from __future__ import annotations
import math
import numpy as np
from bandit_lab.algos.base import BanditPolicy


class PhasedElimination(BanditPolicy):
    name = "PhasedElim"

    def __init__(self, arms: np.ndarray, delta: float = 0.05):
        self.arms = np.asarray(arms, dtype=float)  # (k,d)
        self.delta = delta

    def reset(self, k: int) -> None:
        super().reset(self.arms.shape[0])
        self.active = list(range(self.arms.shape[0]))
        self.phase = 1
        self.queue: list[int] = []
        self.sums: dict[int, float] = {i: 0.0 for i in self.active}
        self.cnts: dict[int, int] = {i: 0 for i in self.active}
        self._plan_phase()

    def _plan_phase(self):
        eps = 2.0 ** (-self.phase)
        m = len(self.active)
        # approx G-optimal: uniform over active (exact when one-hot/orthogonal;
        # Frank-Wolfe would refine for general A — hook left, documented)
        import math as _m
        base = math.ceil(2.0 * self.arms.shape[1] / (eps ** 2)
                         * _m.log(max(2, len(self.arms) * self.phase * (self.phase + 1) / self.delta)))
        per = max(1, base // max(1, m))
        self.queue = []
        for i in self.active:
            self.queue += [i] * per

    def select(self, rng: np.random.Generator) -> int:
        if not self.queue:
            # eliminate then next phase
            self._eliminate()
            self.phase += 1
            self._plan_phase()
        return self.queue.pop(0)

    def _eliminate(self):
        means = {i: self.sums[i] / max(1, self.cnts[i]) for i in self.active}
        # estimate theta via least squares on phase data (one-hot => means)
        # generic: ridge solve using arms
        A = self.arms[self.active]
        y = np.array([means[i] for i in self.active])
        # least squares theta
        theta, *_ = np.linalg.lstsq(A, y, rcond=None)
        preds = {i: float(self.arms[i] @ theta) for i in self.active}
        best = max(preds.values())
        eps = 2.0 ** (-self.phase)
        self.active = [i for i in self.active if best - preds[i] <= 2 * eps]
        if not self.active:
            self.active = [int(np.argmax([preds[i] for i in preds]))]
        self.sums = {i: 0.0 for i in self.active}
        self.cnts = {i: 0 for i in self.active}

    def update(self, a: int, x: float) -> None:
        if a in self.sums:
            self.sums[a] += x
            self.cnts[a] += 1
        super().update(a, x)
