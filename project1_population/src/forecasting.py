"""Forecasting models: an abstract base class and three children, plus a bootstrap."""

from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Type

import numpy as np


class Forecaster(ABC):
    """Abstract base class for all forecasting models.

    Children must implement `fit` and `_curve`.
    `predict` and `fitted` are shared by all children.
    """

    def __init__(self) -> None:
        self.years_ = None     # training years (filled in by fit)
        self.values_ = None    # training values (filled in by fit)

    @abstractmethod
    def fit(self, years: Sequence[int], values: Sequence[float]) -> "Forecaster":
        """Learn from the training data. Must return self."""

    @abstractmethod
    def _curve(self, years: Sequence[int]) -> np.ndarray:
        """Model value for each year in `years` (works for past and future years)."""

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the `horizon` years that come right AFTER the training data."""
        future_years = self.years_[-1] + np.arange(1, horizon + 1)
        return self._curve(future_years)

    def fitted(self) -> np.ndarray:
        """The model's values for the training years (how well it follows the past)."""
        return self._curve(self.years_)

    def _store(self, years: Sequence[int], values: Sequence[float]) -> None:
        """Small helper: save the training data as NumPy arrays."""
        self.years_ = np.asarray(years, dtype=int)
        self.values_ = np.asarray(values, dtype=float)


class LinearTrendForecaster(Forecaster):
    """(a) Straight-line trend: population = slope * year + intercept."""

    def fit(self, years, values):
        self._store(years, values)
        # polyfit(deg=1) finds the best-fitting straight line (least squares)
        self.coefs_ = np.polyfit(self.years_, self.values_, deg=1)
        return self

    def _curve(self, years):
        # polyval plugs the years into the fitted line
        return np.polyval(self.coefs_, years)


class ExponentialForecaster(Forecaster):
    """(b) Exponential / CAGR growth: population grows by the same % every year."""

    def fit(self, years, values):
        self._store(years, values)
        n_steps = len(self.values_) - 1
        # CAGR of the training data (same formula as in analysis.cagr)
        self.growth_ = (self.values_[-1] / self.values_[0]) ** (1 / n_steps) - 1
        return self

    def _curve(self, years):
        years_since_start = np.asarray(years) - self.years_[0]
        return self.values_[0] * (1 + self.growth_) ** years_since_start


def fibonacci_ratios(n: int) -> np.ndarray:
    """First n ratios F(k+1)/F(k) of the Fibonacci sequence 1, 1, 2, 3, 5, 8, 13, ...

    The first two ratios (1/1 and 2/1) jump around wildly, so we skip them and
    start at 3/2 = 1.5, then 5/3 = 1.67, 8/5 = 1.6, 13/8 = 1.625, ... -> 1.618 (the golden ratio).
    """
    fib = [1, 1]
    while len(fib) < n + 4:                      # build enough Fibonacci numbers
        fib.append(fib[-1] + fib[-2])
    ratios = [fib[k + 1] / fib[k] for k in range(len(fib) - 1)]
    return np.array(ratios[2:2 + n])


class FibonacciRatioForecaster(Forecaster):
    """(c) Last value x successive Fibonacci ratios.

    Forecast for step 1 = last_value x 1.5, step 2 = that x 1.667, step 3 = that x 1.6, ...
    For past (training) years there is no formula, so we simply return the actual values.
    """

    def fit(self, years, values):
        self._store(years, values)
        return self

    def _curve(self, years):
        years = np.asarray(years)
        steps_ahead = years - self.years_[-1]                      # 1 = next year, 2 = year after...
        result = np.interp(years, self.years_, self.values_)       # past years: actual values
        future = steps_ahead > 0
        if future.any():
            # cumulative product turns [1.5, 1.67, 1.6] into [1.5, 2.5, 4.0]
            factors = np.cumprod(fibonacci_ratios(steps_ahead.max()))
            result[future] = self.values_[-1] * factors[steps_ahead[future] - 1]
        return result


def bootstrap_interval(ModelClass: Type[Forecaster], years, values, horizon: int,
                       n_boot: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) 95% prediction interval arrays for the forecast.

    seed makes the random draws repeatable, so you get the same answer every run.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    base_model = ModelClass().fit(years, values)
    fitted = base_model.fitted()
    residuals = values - fitted                           # step 1

    simulated = np.empty((n_boot, horizon))               # one row per bootstrap forecast
    for b in range(n_boot):
        fake_values = fitted + rng.choice(residuals, size=len(values), replace=True)   # step 2
        refit = ModelClass().fit(years, fake_values)                                   # step 3
        future_noise = rng.choice(residuals, size=horizon, replace=True)
        simulated[b] = refit.predict(horizon) + future_noise

    lower, upper = np.percentile(simulated, [2.5, 97.5], axis=0)                       # step 4
    return lower, upper
