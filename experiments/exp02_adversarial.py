"""Exp 02: adversarial benchmark (oblivious + switching)."""
import json, os
import numpy as np
from bandit_lab.envs.adversarial import AdversarialBandit
from bandit_lab.metrics import run_adversarial
from bandit_lab.algos.exp3 import Exp3
from bandit_lab.algos.exp3ix import Exp3IX
from bandit_lab.algos.ucb import UCB

N = 5000
SEEDS = list(range(20))

def main():
    os.makedirs("results", exist_ok=True)
    envs = {
        "stoch_like": AdversarialBandit.stochastic_like([0.7, 0.5, 0.4], N, seed=1),
        "switching": AdversarialBandit.switching_two_arm(N),
    }
    out = {}
    for ename, env in envs.items():
        out[ename] = {}
        for pname, pfn in {
            "Exp3": lambda: Exp3(n=N),
            "Exp3-IX": lambda: Exp3IX(n=N),
            "UCB(misspecified)": lambda: UCB(delta=1e-4),
        }.items():
            regs = [run_adversarial(env, pfn, seed=2000 + s)["regret"] for s in SEEDS]
            regs = np.array(regs)
            out[ename][pname] = {"mean": float(regs.mean()), "std": float(regs.std())}
            print(f"{ename:12s} {pname:18s} regret={regs.mean():8.1f}±{regs.std():6.1f}")
    json.dump(out, open("results/exp02_adversarial.json", "w"), indent=2)
    print("wrote results/exp02_adversarial.json")

if __name__ == "__main__":
    main()
