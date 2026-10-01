"""bandit_lab: reproducible bandit algorithms from scratch.

Implements Lattimore & Szepesvari (2020) Ch.6-12,19-20,36 from first
principles. No code is copied from any existing repo.
"""
from bandit_lab.algos.base import BanditPolicy
from bandit_lab.envs.stochastic import BernoulliBandit, GaussianBandit
from bandit_lab.envs.adversarial import AdversarialBandit
from bandit_lab.envs.linear import LinearBandit
from bandit_lab.envs.market import MarketBandit, fetch_yfinance_daily, fetch_coingecko_daily

__all__ = [
    "BanditPolicy",
    "BernoulliBandit",
    "GaussianBandit",
    "AdversarialBandit",
    "LinearBandit",
    "MarketBandit",
    "fetch_yfinance_daily",
    "fetch_coingecko_daily",
]
__version__ = "1.0.0"
