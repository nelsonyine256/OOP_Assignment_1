"""Tests for the forecasting models and the bootstrap."""
import numpy as np
import pytest

from src.forecasting import (ExponentialForecaster, FibonacciRatioForecaster, Forecaster,
                             LinearTrendForecaster, bootstrap_interval, fibonacci_ratios)

YEARS = np.arange(2015, 2022)


def test_linear_recovers_a_straight_line():
    values = 100 + 10 * (YEARS - 2015)
    forecast = LinearTrendForecaster().fit(YEARS, values).predict(2)
    assert forecast == pytest.approx([170.0, 180.0])


def test_exponential_recovers_constant_growth():
    values = 100 * 1.1 ** (YEARS - 2015)
    forecast = ExponentialForecaster().fit(YEARS, values).predict(1)
    assert forecast == pytest.approx([100 * 1.1 ** 7])


def test_fibonacci_ratios_and_first_forecast():
    assert fibonacci_ratios(3) == pytest.approx([1.5, 5 / 3, 1.6])
    forecast = FibonacciRatioForecaster().fit(YEARS, np.full(7, 100.0)).predict(1)
    assert forecast == pytest.approx([150.0])


def test_abstract_class_cannot_be_created():
    with pytest.raises(TypeError):
        Forecaster()


def test_bootstrap_is_reproducible_and_ordered():
    values = np.array([100, 104, 109, 113, 119, 124, 130], dtype=float)
    low1, up1 = bootstrap_interval(ExponentialForecaster, YEARS, values, 3, n_boot=200)
    low2, up2 = bootstrap_interval(ExponentialForecaster, YEARS, values, 3, n_boot=200)
    assert np.array_equal(low1, low2) and np.array_equal(up1, up2)   # same seed -> same answer
    assert np.all(low1 <= up1)
