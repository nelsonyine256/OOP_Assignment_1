"""Forecasters: a base class and subclasses that each predict the NEXT value of a series."""

import numpy as np


class Forecaster:
    """Base class. Every forecaster predicts the NEXT value from a history."""

    def __init__(self, name: str = "Forecaster") -> None:
        self.name = name

    def forecast(self, history: np.ndarray) -> float:
        """Return the forecast for the day after `history` (a 1-D array)."""
        raise NotImplementedError("Subclasses must implement forecast().")

    def __repr__(self) -> str:
        return self.name


class MovingAverage(Forecaster):
    """Forecast = mean of the last `window` observations."""

    def __init__(self, window: int = 3) -> None:
        super().__init__(name=f"Moving average ({window}-day)")
        self.window = window

    def forecast(self, history: np.ndarray) -> float:
        return np.mean(history[-self.window:])   # last `window` values


class ExponentialSmoothing(Forecaster):
    """Simple exponential smoothing.

    level_t = alpha * value_t + (1 - alpha) * level_(t-1)
    The forecast is the latest level.
    alpha near 1 -> reacts fast to new data; near 0 -> very smooth.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__(name=f"Exp. smoothing (alpha={alpha:.1f})")
        self.alpha = alpha

    def forecast(self, history: np.ndarray) -> float:
        level = history[0]                        # start with the first value
        for value in history[1:]:
            level = self.alpha * value + (1 - self.alpha) * level
        return level


class LinearTrend(Forecaster):
    """Fit a straight line (value = slope*day + intercept) and extend it by one day."""

    def __init__(self) -> None:
        super().__init__(name="Linear trend")

    def forecast(self, history: np.ndarray) -> float:
        days = np.arange(len(history))
        slope, intercept = np.polyfit(days, history, deg=1)   # least-squares line
        next_day = len(history)
        return slope * next_day + intercept


class SeasonalNaive(Forecaster):
    """Forecast = the value from one season ago (default: 7 days = same weekday last week)."""

    def __init__(self, period: int = 7) -> None:
        super().__init__(name=f"Seasonal naive ({period}-day)")
        self.period = period

    def forecast(self, history: np.ndarray) -> float:
        return history[-self.period]
