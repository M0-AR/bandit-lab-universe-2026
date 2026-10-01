"""Exp 03: linear bandits (hypercube + unit-ball sample) + lower-bound check."""
import json, os
import numpy as np
from bandit_lab.envs.linear import LinearBandit
from bandit_lab.algos.linucb import LinUCB

def run(env, n, seed=0):
    rng = np.random.default_rng(seed)
    d = env.arms.shape[1]
    pol = LinUCB(d=d, delta=1e-3)
    pol.reset()
    tot = 0.0
    for _ in range(n):
        a = pol.select_arm(env.arms, rng)
        x = env.pull(a, rng)
        pol.update_ctx(env.arms[a], x)
        tot += x
    return n * env.mu_star - tot

def main():
    os.makedirs("results", exist_ok=True)
    out = {}
    N = 2000
    # hypercube d=4 worst-case direction (Thm 24.1 Omega(d sqrt n))
    d = 4
    theta = np.full(d, 1 / np.sqrt(N))
    env = LinearBandit.hypercube(d, theta)
    regs = [run(env, N, seed=3000 + s) for s in range(10)]
    out["hypercube_d4"] = {"mean": float(np.mean(regs)), "std": float(np.std(regs)),
                           "lower_thm24.1": float(np.exp(-2) / 8 * d * np.sqrt(N))}
    print("hypercube", np.mean(regs), "lower~", out["hypercube_d4"]["lower_thm24.1"])
    # unit ball sample
    theta2 = np.zeros(5); theta2[0] = 0.5
    env2 = LinearBandit.unit_ball_sample(k=50, d=5, theta=theta2, seed=0)
    regs2 = [run(env2, N, seed=4000 + s) for s in range(10)]
    out["unitball_k50_d5"] = {"mean": float(np.mean(regs2)), "std": float(np.std(regs2))}
    print("unitball", np.mean(regs2))
    json.dump(out, open("results/exp03_linear.json", "w"), indent=2)

if __name__ == "__main__":
    main()
