"""Exp 01: stochastic finite-arm benchmark (Bernoulli + Gaussian).

Protocol (from 2026 fair-evaluation literature):
- fixed seeds, one action per step, identical env draws per (env,seed) via
  pre-generated reward tables? Here: shared RNG stream per seed, policies run
  in same order -> fair within seed; averaged over seeds.
- reports: cumulative regret, pseudo-regret, P(optimal), variance, VaR95.
"""
import json, math, os
import numpy as np
from bandit_lab.envs.stochastic import BernoulliBandit, GaussianBandit
from bandit_lab.metrics import run_stochastic
from bandit_lab.algos.etc import ETC
from bandit_lab.algos.ucb import UCB, AsymptoticallyOptimalUCB
from bandit_lab.algos.moss import MOSS
from bandit_lab.algos.kl_ucb import KLUCB
from bandit_lab.algos.exp3 import Exp3
from bandit_lab.algos.exp3ix import Exp3IX
from bandit_lab.algos.thompson import ThompsonBernoulli, ThompsonGaussian

ENVS = {
    "bern_easy": lambda: BernoulliBandit([0.7, 0.5, 0.4]),
    "bern_hard": lambda: BernoulliBandit([0.55, 0.5, 0.48]),
    "gauss_2arm": lambda: GaussianBandit([0.5, 0.6], sigma=1.0),
    "gauss_5arm": lambda: GaussianBandit([0.8, 0.6, 0.6, 0.5, 0.4], sigma=1.0),
}
N = 5000
SEEDS = list(range(20))

def policies(n):
    return {
        "ETC-m50": lambda: ETC(m=50),
        "UCB-1/n2": lambda: UCB(delta=None),
        "UCB*": lambda: AsymptoticallyOptimalUCB(),
        "MOSS": lambda: MOSS(n=n),
        "KL-UCB": lambda: KLUCB(),
        "Exp3": lambda: Exp3(n=n),
        "Exp3-IX": lambda: Exp3IX(n=n),
        "TS-Bern": lambda: ThompsonBernoulli(),
        "TS-Gauss": lambda: ThompsonGaussian(),
    }

def main():
    os.makedirs("results", exist_ok=True)
    out = {}
    for ename, efn in ENVS.items():
        out[ename] = {}
        for pname, pfn in policies(N).items():
            regs, pseudos = [], []
            for s in SEEDS:
                env = efn()
                # UCB needs n_hint; run_stochastic handles it
                r = run_stochastic(env, pfn, N, seed=1000 + s)
                regs.append(r["regret"]); pseudos.append(r["pseudo_regret"])
            regs = np.array(regs); pseudos = np.array(pseudos)
            out[ename][pname] = {
                "mean_regret": float(regs.mean()), "std_regret": float(regs.std()),
                "mean_pseudo": float(pseudos.mean()),
                "var95": float(np.quantile(regs, 0.95)),
                "theory": theory_note(pname, ename),
            }
            print(f"{ename:12s} {pname:10s} regret={regs.mean():8.1f}±{regs.std():6.1f} VaR95={np.quantile(regs,0.95):8.1f}")
    json.dump(out, open("results/exp01_stochastic.json", "w"), indent=2)
    print("wrote results/exp01_stochastic.json")

def theory_note(p, e):
    notes = {
        "ETC-m50": "Thm6.1 O(sqrt n) only if m tuned; else n^{2/3}",
        "UCB-1/n2": "Thm7.1 sum 16log n/Delta; worst O(sqrt(nk log n))",
        "UCB*": "Thm8.1 limsup R/log n <= sum 2/Delta (asymptotically optimal)",
        "MOSS": "Thm9.1 O(sqrt(nk)) minimax optimal",
        "KL-UCB": "Thm10.6 limsup R/log n <= sum Delta/d(mu,mu*) (Bernoulli-optimal)",
        "Exp3": "Thm11.2 O(sqrt(nk log k)) adversarial->stochastic reduction",
        "Exp3-IX": "Thm12.1 high-prob O(sqrt(nk log(k/delta)))",
        "TS-Bern": "Bayes O(sqrt(nk log n)); frequentist asymptotically optimal (Bern)",
        "TS-Gauss": "Thm36.3 asymptotically optimal Gaussian",
    }
    return notes.get(p, "")

if __name__ == "__main__":
    main()
