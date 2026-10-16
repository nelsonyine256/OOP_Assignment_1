"""Reusable logic for Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model.

Contains three classes and two helper functions:

* FishStock     - logistic growth with harvesting (and optional closed season)
* PriceModel    - bounded random-walk fish price (UGX per kg)
* RiskAssessor  - CV-based risk class, Monte Carlo simulation and Value-at-Risk
* weekly_revenue, revenue_stats - small helpers used by the notebook

The logic is the same as the notebook cells; type hints have been added.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


class FishStock:
    """Fish stock that grows logistically and is harvested every week.

    Parameters
    ----------
    r  : growth rate per week
    K  : carrying capacity (tonnes)
    N0 : starting stock (tonnes)
    h  : harvest rate, fraction of stock caught per week (0 to 1)
    """

    def __init__(self, r: float = 0.4, K: float = 10_000, N0: float = 4_000, h: float = 0.10) -> None:
        self.r, self.K, self.N0, self.h = r, K, N0, h

    # ---- theory: quick formulas we can compare simulations against ----
    @property
    def msy(self) -> float:
        """Maximum sustainable yield per week (tonnes) = rK/4."""
        return self.r * self.K / 4

    @property
    def equilibrium_stock(self) -> float:
        """Stock level the population settles at (tonnes). Zero if h >= r."""
        return max(0.0, self.K * (1 - self.h / self.r))

    @property
    def status(self) -> str:
        """Plain-language sustainability label based on h."""
        if self.h >= self.r:
            return "Collapse"                      # catch removes more than growth adds
        if self.h > self.r / 2:
            return "Overfishing"                   # stock survives but yield is below MSY
        return "Sustainable"

    # ---- simulation ----
    def simulate(self, weeks: int = 52, closed_weeks: Sequence[int] = ()) -> Tuple[np.ndarray, np.ndarray]:
        """Run the model forward.

        Parameters
        ----------
        weeks        : number of weeks to simulate
        closed_weeks : week-of-year numbers (0-51) with NO harvesting (closed season)

        Returns
        -------
        stock   : array of length weeks+1 (stock at the start of each week, plus the end)
        harvest : array of length weeks (tonnes caught in each week)
        """
        stock = np.zeros(weeks + 1)
        harvest = np.zeros(weeks)
        stock[0] = self.N0

        for t in range(weeks):
            N = stock[t]
            h_now = 0.0 if (t % 52) in closed_weeks else self.h   # closed season => no catch
            harvest[t] = h_now * N
            growth = self.r * N * (1 - N / self.K)
            stock[t + 1] = max(0.0, N + growth - harvest[t])       # stock cannot go negative
        return stock, harvest


class PriceModel:
    """Weekly fish price (UGX/kg) as a random walk with bounds.

    Parameters
    ----------
    start   : starting price
    low/high: price floor and ceiling
    step_sd : standard deviation of the weekly price change (UGX)
    seed    : random seed so results can be reproduced
    """

    def __init__(self, start: float = 12_000, low: float = 9_000, high: float = 16_000,
                 step_sd: float = 300, seed: int = 42) -> None:
        self.start, self.low, self.high, self.step_sd = start, low, high, step_sd
        self.rng = np.random.default_rng(seed)

    def simulate_paths(self, n_paths: int = 1, weeks: int = 52) -> np.ndarray:
        """Return an array of shape (n_paths, weeks) of weekly prices."""
        prices = np.zeros((n_paths, weeks))
        prices[:, 0] = self.start
        for t in range(1, weeks):
            step = self.rng.normal(0, self.step_sd, size=n_paths)          # random weekly change
            prices[:, t] = np.clip(prices[:, t - 1] + step, self.low, self.high)
        return prices


def weekly_revenue(harvest_tonnes: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Revenue (UGX) = harvest in kg x price per kg. 1 tonne = 1,000 kg."""
    return harvest_tonnes * 1_000 * prices


def revenue_stats(revenue: np.ndarray) -> Dict[str, float]:
    """Return mean, median, variance, standard deviation and CV of a revenue series."""
    mean = revenue.mean()
    std = revenue.std(ddof=1)            # ddof=1 = sample standard deviation
    return {
        "mean (UGX)": mean,
        "median (UGX)": np.median(revenue),
        "variance (UGX^2)": revenue.var(ddof=1),
        "std dev (UGX)": std,
        "CV (no units)": std / mean,     # coefficient of variation = std / mean
    }


class RiskAssessor:
    """Classify revenue risk and estimate Value-at-Risk by Monte Carlo.

    Parameters
    ----------
    low_cv, high_cv : CV cut-offs between Low / Medium / High risk
    var_level       : tail probability for VaR (0.05 = 5%)
    """

    def __init__(self, low_cv: float = 0.10, high_cv: float = 0.25, var_level: float = 0.05) -> None:
        self.low_cv, self.high_cv, self.var_level = low_cv, high_cv, var_level

    def classify(self, cv: float) -> str:
        """Turn a coefficient of variation into 'Low', 'Medium' or 'High'."""
        if cv < self.low_cv:
            return "Low"
        if cv < self.high_cv:
            return "Medium"
        return "High"

    def monte_carlo(self, fish: FishStock, price_model: PriceModel,
                    n_paths: int = 5_000, weeks: int = 52) -> np.ndarray:
        """Simulate annual revenue for many random price paths.

        The stock path is fixed by the harvest rate; only prices are random.
        Returns an array of n_paths annual revenues (UGX).
        """
        _, harvest = fish.simulate(weeks)
        prices = price_model.simulate_paths(n_paths, weeks)          # shape (n_paths, weeks)
        return weekly_revenue(harvest, prices).sum(axis=1)           # add up the weeks

    def value_at_risk(self, annual_revenues: np.ndarray) -> float:
        """Annual revenue level that is undercut only var_level of the time."""
        return float(np.percentile(annual_revenues, self.var_level * 100))
