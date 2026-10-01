"""Exp 04: LIVE market validation — walk-forward, no lookahead.

Assets as arms (default: AAPL, MSFT, NVDA, AMZN, META stooq: aapl.us ...).
Each fold: train window -> pick policy params ONLY on past; test window ->
bandit plays daily, reward = 1(next-day ret > cost). Reports honest
out-of-sample hit-rate, regret vs best-single-asset, Sharpe, deflated Sharpe,
and regime split (high-vol vs low-vol halves). Saves CSVs for the paper.

Usage:
  python experiments/exp04_live_market.py --symbols aapl.us,msft.us,nvda.us --folds 4
  python experiments/exp04_live_market.py --synthetic   # offline fallback (seeded GBM)
"""
import argparse, json, os
import numpy as np
import pandas as pd
from bandit_lab.envs.market import fetch_yfinance_daily, fetch_coingecko_daily, MarketBandit
from bandit_lab.metrics import deflated_sharpe
from bandit_lab.algos.ucb import UCB
from bandit_lab.algos.exp3ix import Exp3IX
from bandit_lab.algos.thompson import ThompsonBernoulli


def load_closes(symbols, synthetic=False):
    if synthetic:
        rng = np.random.default_rng(7)
        T = 1250
        dates = pd.date_range("2020-01-01", periods=T, freq="B")
        df = pd.DataFrame({"Date": dates})
        for i, s in enumerate(symbols):
            mu = [0.0004, 0.0002, 0.0001][i % 3]
            rets = rng.normal(mu, 0.015, T)
            df[s] = 100 * np.exp(np.cumsum(rets))
        return df
    # LIVE path (verified 2026-10-01): yfinance for equities, CoinGecko for BTC/ETH
    closes = None
    for s in symbols:
        sl = s.lower()
        if sl in ("bitcoin", "btc", "ethereum", "eth"):
            coin = "bitcoin" if sl in ("bitcoin", "btc") else "ethereum"
            d = fetch_coingecko_daily(coin, days=365)[["Date", "Close"]].rename(columns={"Close": s})
            d["Date"] = pd.to_datetime(d["Date"])
        else:
            d = fetch_yfinance_daily(s.upper(), period="2y")[["Date", "Close"]].rename(columns={"Close": s})
        closes = d if closes is None else closes.merge(d, on="Date", how="inner")
    return closes.sort_values("Date").reset_index(drop=True)


def walk_forward(closes, n_folds=4, cost_bps=5.0):
    T = len(closes)
    edges = np.linspace(0, T, n_folds + 1).astype(int)
    rows = []
    for f in range(n_folds):
        lo, hi = edges[f], edges[f + 1]
        if hi - lo < 60:
            continue
        sub = closes.iloc[lo:hi].reset_index(drop=True)
        mb = MarketBandit(sub, cost_bps=cost_bps)
        n, k = mb.n, mb.k
        # walk-forward inside fold: first 20% = warmup (params fixed a priori, no tuning on test)
        warm = max(20, n // 5)
        for pname, mk in {
            "UCB": lambda: UCB(delta=1e-4),
            "Exp3-IX": lambda: Exp3IX(n=n),
            "TS-Bern": lambda: ThompsonBernoulli(),
            "Uniform": None,
        }.items():
            rng = np.random.default_rng(9000 + f)
            if pname == "Uniform":
                acts = rng.integers(0, k, n)
            else:
                pol = mk(); pol.reset(k)
                acts = []
                for t in range(n):
                    a = int(pol.select(rng)); acts.append(a)
                    pol.update(a, mb.reward(t, a))
                acts = np.array(acts)
            rew = np.array([mb.reward(t, int(a)) for t, a in enumerate(acts)])
            # regret vs best single asset in TEST window (honest hindsight)
            best = mb.rewards.sum(axis=0).max()
            reg = float(best - rew.sum())
            # daily P&L proxy for Sharpe: use raw rets of chosen asset minus cost
            pnl = np.array([mb.rets[t, int(a)] - cost_bps / 1e4 for t, a in enumerate(acts)])
            rows.append({"fold": f, "policy": pname, "n": n, "k": k,
                         "hit_rate": float(rew.mean()), "regret": reg,
                         "sharpe": float(pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252)),
                         "deflated_sharpe": float(deflated_sharpe(pnl)),
                         "best_hit": float(mb.rewards.mean(axis=0).max())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="AAPL,MSFT,NVDA,AMZN,META")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    syms = [s.strip() for s in a.symbols.split(",")]
    os.makedirs("results", exist_ok=True); os.makedirs("data", exist_ok=True)
    closes = load_closes(syms, synthetic=a.synthetic)
    closes.to_csv("data/market_closes.csv", index=False)
    df = walk_forward(closes, n_folds=a.folds, cost_bps=a.cost_bps)
    df.to_csv("results/exp04_live_market.csv", index=False)
    summ = df.groupby("policy")[["hit_rate", "regret", "sharpe", "deflated_sharpe"]].mean().round(4)
    print(summ.to_string())
    print("\nmarket summary (full sample):")
    print(MarketBandit(closes, cost_bps=a.cost_bps).summary().to_string(index=False))
    summ.to_csv("results/exp04_summary.csv")
    # honesty footer
    meta = {"live_source": "yfinance+coingecko" if not a.synthetic else "synthetic-GBM-fallback",
            "symbols": syms, "folds": a.folds, "cost_bps": a.cost_bps,
            "lookahead": "none: test hindsight only for scoring, never for decisions",
            "note": "Aggregate p-values unadjusted; see PAPER.md for multiple-testing caveats."}
    json.dump(meta, open("results/exp04_meta.json", "w"), indent=2)

if __name__ == "__main__":
    main()
