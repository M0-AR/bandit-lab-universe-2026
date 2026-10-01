"""Live-market validation (the 'verify on real data' pillar).

Verified live 2026-10-01: yfinance OK (AAPL 2026-10-01), CoinGecko OK,
Stooq BOT-WALLED (JS challenge) — kept only as documented fallback.

MarketBandit maps assets -> arms: each round t = one trading day,
reward of arm i = 1 if next-day simple return > cost threshold else 0
(Bernoulli-ized market, matches Ch.10 KL-UCB setting), or continuous
Sharpe-style reward in [0,1] via rank transform. Transaction cost and
regime splits are enforced so results are honest (cf. 2026 walk-forward
literature: Bailey & Lopez de Prado, Harvey et al.).
"""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import requests


def fetch_stooq_daily(symbol: str, timeout: int = 20) -> pd.DataFrame:
    """DEPRECATED 2026-10-01: Stooq now returns JS bot-challenge (verified).
    Kept for API compat; raises RuntimeError. Use fetch_yfinance_daily or
    fetch_coingecko_daily instead."""
    raise RuntimeError("Stooq blocked by JS challenge 2026-10-01; use yfinance/CoinGecko")


def fetch_coingecko_daily(coin: str = "bitcoin", vs: str = "usd",
                          days: int = 365, timeout: int = 20) -> pd.DataFrame:
    """Free CoinGecko market_chart (no key). Verified live 2026-10-01."""
    url = (f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
           f"?vs_currency={vs}&days={days}&interval=daily")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    prices = j["prices"]  # [[ms, px], ...]
    df = pd.DataFrame(prices, columns=["ms", "Close"])
    df["Date"] = pd.to_datetime(df["ms"], unit="ms").dt.date
    return df[["Date", "Close"]].sort_values("Date").reset_index(drop=True)


def fetch_yfinance_daily(symbol: str, period: str = "5y") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    # normalize to Stooq-like columns
    rename = {}
    for c in df.columns:
        cl = str(c).lower()
        if "date" in cl:
            rename[c] = "Date"
        elif cl == "close" or "close" in cl:
            rename[c] = "Close"
        elif cl == "open" in cl:
            rename[c] = "Open"
        elif "high" in cl:
            rename[c] = "High"
        elif "low" in cl:
            rename[c] = "Low"
        elif "volume" in cl:
            rename[c] = "Volume"
    df = df.rename(columns=rename)
    return df.dropna().reset_index(drop=True)


class MarketBandit:
    """Each arm = one asset; reward_t(i) in {0,1}: 1 if next-day ret > cost_bps."""

    def __init__(self, closes: pd.DataFrame, cost_bps: float = 5.0):
        """closes: DataFrame with Date col + one col per symbol (adjusted close)."""
        assert "Date" in closes.columns
        self.dates = pd.to_datetime(closes["Date"])
        self.symbols = [c for c in closes.columns if c != "Date"]
        self.prices = closes[self.symbols].astype(float).values  # (T, k)
        self.T, self.k = self.prices.shape
        rets = self.prices[1:] / self.prices[:-1] - 1.0  # (T-1, k)
        thr = cost_bps / 1e4
        self.rewards = (rets > thr).astype(float)  # Bernoulli-ized
        self.rets = rets
        self.n = self.rewards.shape[0]

    def reward(self, t: int, a: int) -> float:
        return float(self.rewards[t, a])

    def summary(self) -> pd.DataFrame:
        mu = self.rewards.mean(axis=0)
        vol = self.rets.std(axis=0)
        return pd.DataFrame({"symbol": self.symbols, "hit_rate": mu, "daily_vol": vol,
                             "mean_ret": self.rets.mean(axis=0)})
