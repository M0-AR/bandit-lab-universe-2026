"""metrics, harness, validation helpers."""
from __future__ import annotations
import numpy as np


def run_stochastic(env, policy_fn, n: int, seed: int = 0):
    """policy_fn() -> fresh policy. Returns dict with regret, pulls, rewards."""
    rng = np.random.default_rng(seed)
    pol = policy_fn()
    if hasattr(pol, "reset"):
        try:
            pol.reset(env.k, n_hint=n)
        except TypeError:
            pol.reset(env.k)
    pulls = np.zeros(env.k, dtype=int)
    rewards = np.zeros(n)
    for t in range(n):
        a = int(pol.select(rng))
        x = float(env.pull(a, rng))
        pol.update(a, x)
        pulls[a] += 1
        rewards[t] = x
    opt = env.mu_star
    regret = n * opt - float(np.sum(rewards))
    pseudo = float(np.sum(env.gaps * pulls))
    return {"regret": regret, "pseudo_regret": pseudo, "pulls": pulls,
            "rewards": rewards, "mu_star": opt}


def run_adversarial(env, policy_fn, seed: int = 0):
    rng = np.random.default_rng(seed)
    pol = policy_fn()
    pol.reset(env.k)
    tot = 0.0
    for t in range(env.n):
        a = int(pol.select(rng))
        x = env.reward(t, a)
        pol.update(a, x)
        tot += x
    regret = env.best_fixed - tot
    return {"regret": regret, "reward": tot, "best": env.best_fixed}


def run_linear(env, rng_seed: int = 0):
    from bandit_lab.algos.linucb import LinUCB
    rng = np.random.default_rng(rng_seed)
    d = env.arms.shape[1]
    pol = LinUCB(d=d)
    pol.reset()
    tot = 0.0
    for _ in range(len(env.means) * 0 + 2000):  # placeholder, real loops in experiments
        pass
    return pol


def deflated_sharpe(returns: np.ndarray, trials: int = 10) -> float:
    """Bailey & Lopez de Prado (2014) deflated Sharpe approximation."""
    import math
    r = np.asarray(returns, float)
    if len(r) < 3 or np.std(r) == 0:
        return 0.0
    sr = float(np.mean(r) / np.std(r) * math.sqrt(252))
    # Haircut for multiple trials: SR_deflated ~ SR * (1 - (1-gamma)*...); use simple PSR-style penalty
    skew = float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + 1e-12))
    kurt = float(((r - r.mean()) ** 4).mean() / (r.std() ** 4 + 1e-12))
    # expected SR under null with trials (approx)
    import math as m
    e_sr = (1 - 0.5772156649) * m.sqrt(m.pi / 2) * m.sqrt(m.log(trials) / m.e) if trials > 1 else 0.0
    # simplified: penalize
    return float(sr - e_sr * 0.3 - 0.1 * abs(skew) - 0.05 * max(0, kurt - 3))
