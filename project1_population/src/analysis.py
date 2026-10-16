"""Small helper functions: growth, error metrics and classroom planning."""

from typing import Sequence, Tuple

import numpy as np

PRIMARY_SHARE = 0.18          # 18% of people are primary-school age
PUPILS_PER_CLASSROOM = 53


def yoy_growth(populations: Sequence[float]) -> np.ndarray:
    """Year-on-year growth in percent. Returns one fewer value than the input."""
    p = np.asarray(populations, dtype=float)
    return (p[1:] - p[:-1]) / p[:-1] * 100      # p[1:] = "this year", p[:-1] = "last year"


def cagr(populations: Sequence[float]) -> float:
    """Compound annual growth rate as a fraction (0.05 means 5% per year)."""
    p = np.asarray(populations, dtype=float)
    n_steps = len(p) - 1                          # number of yearly steps
    return (p[-1] / p[0]) ** (1 / n_steps) - 1


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, float, float]:
    """Return (MAE, RMSE, MAPE %) comparing actual and predicted values."""
    errors = actual - predicted
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors) / actual) * 100
    return mae, rmse, mape


def classrooms_needed(population_thousands: float) -> int:
    """Classrooms needed for a population given in thousands (rounded UP)."""
    pupils = population_thousands * 1000 * PRIMARY_SHARE
    return int(np.ceil(pupils / PUPILS_PER_CLASSROOM))
