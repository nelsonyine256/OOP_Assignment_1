"""Planning tools: equilibrium, walk-forward backtesting, alpha tuning, fleet sizing, simulation."""

import math
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.linalg import solve

from src.forecasters import ExponentialSmoothing, Forecaster


def find_equilibrium(demand_intercept: float, demand_slope: float,
                     supply_intercept: float, supply_slope: float) -> Tuple[float, float]:
    """Solve Qd = a - b*P  and  Qs = c + d*P  for the equilibrium (Q, P).

    Written as a linear system in x = [Q, P]:
        Q + b*P = a
        Q - d*P = c
    """
    A = np.array([[1,  demand_slope],
                  [1, -supply_slope]])
    b = np.array([demand_intercept, supply_intercept])
    Q, P = solve(A, b)
    return Q, P


def backtest(forecaster: Forecaster, data: Sequence[float],
             first_day: int = 4) -> Tuple[np.ndarray, float]:
    """Walk-forward backtest of one forecaster on one series.

    Parameters
    ----------
    forecaster : Forecaster
        Any object with a .forecast(history) method.
    data : array-like
        The full series (day 1 first).
    first_day : int
        First day (1-indexed) to forecast. Default 4 (so 3 days of history).

    Returns
    -------
    forecasts : np.ndarray   forecasts for days first_day ... last day
    mae : float              mean absolute error over those days
    """
    data = np.asarray(data, dtype=float)
    forecasts = []
    for day in range(first_day, len(data) + 1):
        history = data[: day - 1]                 # only days BEFORE `day`
        forecasts.append(forecaster.forecast(history))

    forecasts = np.array(forecasts)
    actual = data[first_day - 1:]                 # what really happened
    mae = np.mean(np.abs(actual - forecasts))
    return forecasts, mae


def tune_alpha(data: Sequence[float], alphas: Sequence[float] = np.arange(0.1, 1.0, 0.1),
               first_day: int = 4) -> Tuple[float, Dict[float, float]]:
    """Grid-search the best smoothing alpha. Returns (best_alpha, {alpha: mae})."""
    maes = {}
    for a in alphas:
        _, mae = backtest(ExponentialSmoothing(a), data, first_day)
        maes[round(float(a), 1)] = mae
    best_alpha = min(maes, key=maes.get)          # alpha with smallest MAE
    return best_alpha, maes


def plan_fleet(forecast_passengers: float, trips_per_vehicle: int = 8,
               seats: int = 14, buffer: float = 0.15) -> Tuple[float, float, int]:
    """How many vehicles are needed to carry the forecast passengers?

    Returns (raw_vehicles, vehicles_with_buffer, vehicles_to_deploy).
    """
    capacity = trips_per_vehicle * seats               # passengers per vehicle per day
    raw = forecast_passengers / capacity
    with_buffer = raw * (1 + buffer)
    deploy = max(1, math.ceil(with_buffer))            # round UP, at least 1
    return raw, with_buffer, deploy


def simulate_weekly_demand(days: int = 60, base: float = 50, noise_sd: float = 2.0,
                           seed: int = 42) -> np.ndarray:
    """Simulate daily passengers with a weekly pattern (day 1 = Monday).

    Weekly multipliers (Mon..Sun): Friday is busiest, Sunday is quietest.
    """
    weekly_factors = np.array([1.00, 1.00, 1.05, 1.10, 1.40, 1.15, 0.60])  # Mon..Sun
    rng = np.random.default_rng(seed)                  # reproducible randomness
    weekday = np.arange(days) % 7                      # 0=Mon ... 6=Sun
    demand = base * weekly_factors[weekday] + rng.normal(0, noise_sd, days)
    return np.round(demand).clip(min=0)                # whole passengers, never negative
