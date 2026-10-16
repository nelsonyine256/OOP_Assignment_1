"""Pytest unit tests for Mini-Project 5 (taxi planner)."""

import statistics

import numpy as np
import pytest

from src.forecasters import (ExponentialSmoothing, Forecaster, LinearTrend,
                             MovingAverage, SeasonalNaive)
from src.planning import backtest, find_equilibrium, plan_fleet, simulate_weekly_demand
from src.route import Route
from src.route_planner import RoutePlanner


# ------------------------------------------------------------------ normal cases
def test_route_revenue_and_stats():
    r = Route("Test", [10, 20, 30], fare=1000)
    assert list(r.daily_revenue()) == [10_000, 20_000, 30_000]
    assert r.total_revenue() == 60_000
    s = r.stats()
    assert s["mean"] == 20
    assert s["variance"] == pytest.approx(100)   # sample variance
    assert s["std_dev"] == pytest.approx(10)


def test_equilibrium_matches_hand_calculation():
    q, p = find_equilibrium(120, 0.02, 10, 0.03)
    assert p == pytest.approx(2200)
    assert q == pytest.approx(76)


def test_forecasters_on_simple_series():
    history = np.array([1.0, 2.0, 3.0, 4.0])
    assert MovingAverage(window=2).forecast(history) == pytest.approx(3.5)
    assert ExponentialSmoothing(alpha=1.0).forecast(history) == pytest.approx(4.0)  # = last value
    assert LinearTrend().forecast(history) == pytest.approx(5.0)                    # perfect line
    assert SeasonalNaive(period=2).forecast(history) == pytest.approx(3.0)


def test_backtest_length_and_perfect_forecast():
    data = [5, 5, 5, 5, 5, 5]
    forecasts, mae = backtest(MovingAverage(3), data, first_day=4)
    assert len(forecasts) == 3          # days 4, 5, 6
    assert mae == 0


def test_plan_fleet_rounds_up_with_buffer():
    raw, buffered, deploy = plan_fleet(112)        # exactly one vehicle of demand
    assert raw == pytest.approx(1.0)
    assert buffered == pytest.approx(1.15)
    assert deploy == 2                             # 1.15 -> round UP to 2


def test_simulation_is_reproducible():
    a = simulate_weekly_demand(seed=1)
    b = simulate_weekly_demand(seed=1)
    c = simulate_weekly_demand(seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert (a >= 0).all()


def test_route_planner_composition():
    route = Route("Test", [10, 20, 30, 40], fare=500)
    planner = RoutePlanner(route, LinearTrend())
    assert planner.next_day_passengers() == pytest.approx(50)
    assert planner.next_day_revenue() == pytest.approx(25_000)
    assert planner.vehicles_needed() == 1


# ------------------------------------------------------------------ edge cases
def test_edge_empty_route_statistics_raises():
    with pytest.raises(statistics.StatisticsError):
        Route("Empty", [], fare=1000).stats()


def test_edge_zero_passengers_still_deploys_one_vehicle():
    raw, _, deploy = plan_fleet(0)
    assert raw == 0
    assert deploy == 1                 # policy: at least one vehicle per route


def test_edge_base_forecaster_is_abstract():
    with pytest.raises(NotImplementedError):
        Forecaster().forecast(np.array([1.0, 2.0]))


def test_edge_seasonal_naive_needs_enough_history():
    with pytest.raises(IndexError):
        SeasonalNaive(period=7).forecast(np.array([1.0, 2.0, 3.0]))
