"""Make paper figures from results/*.json/csv (no lookahead, pure viz)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)

# Fig1: stochastic benchmark mean regret (bern_hard)
d = json.load(open("results/exp01_stochastic.json"))
for env in ["bern_easy", "bern_hard", "gauss_5arm"]:
    rows = sorted(d[env].items(), key=lambda kv: kv[1]["mean_regret"])
    names = [k for k, _ in rows]; vals = [v["mean_regret"] for _, v in rows]
    errs = [v["std_regret"] for _, v in rows]
    plt.figure(figsize=(8, 3.5))
    plt.bar(names, vals, yerr=errs, capsize=3)
    plt.xticks(rotation=25, ha="right"); plt.ylabel("mean cumulative regret (n=5000, 20 seeds)")
    plt.title(f"Stochastic benchmark — {env} (lower is better)")
    plt.tight_layout(); plt.savefig(f"results/figures/stoch_{env}.png", dpi=150); plt.close()

# Fig2: live market hit-rate by policy
live = pd.read_csv("results/exp04_live_market.csv")
m = live.groupby("policy")["hit_rate"].mean().sort_values()
plt.figure(figsize=(6, 3.2))
plt.bar(m.index, m.values); plt.ylabel("mean out-of-sample hit-rate (live 2y, walk-forward)")
plt.title("Live-market validation: bandits vs uniform (higher is better)")
plt.tight_layout(); plt.savefig("results/figures/live_hitrate.png", dpi=150); plt.close()

# Fig3: hidden pattern Q2 pulls
h = json.load(open("results/exp05_hidden_patterns.json"))
pulls = h["Q2_optimism_failure"]["pulls"]
plt.figure(figsize=(5, 3.2))
plt.bar(["a1=e1 (opt)", "a2=e2 (informative)", "a3 near-opt (uninformative)"], pulls)
plt.ylabel("pulls (n=3000)"); plt.title("Optimism failure: LinUCB starves informative a2")
plt.tight_layout(); plt.savefig("results/figures/optimism_failure.png", dpi=150); plt.close()
print("figures written:", os.listdir("results/figures"))
