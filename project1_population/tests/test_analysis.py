"""Tests for growth, error metrics and classroom planning."""
import numpy as np
import pytest

from src.analysis import cagr, classrooms_needed, error_metrics, yoy_growth


def test_yoy_growth_known_values():
    assert yoy_growth([100, 110, 121]) == pytest.approx([10.0, 10.0])


def test_cagr_known_value():
    # 100 -> 121 over 2 yearly steps is exactly 10% a year
    assert cagr([100, 110, 121]) == pytest.approx(0.10)


def test_error_metrics_perfect_and_imperfect():
    actual = np.array([100.0, 200.0])
    assert error_metrics(actual, actual) == (0.0, 0.0, 0.0)
    mae, rmse, mape = error_metrics(actual, np.array([110.0, 180.0]))
    assert mae == pytest.approx(15.0)
    assert mape == pytest.approx(10.0)


def test_classrooms_round_up_and_zero():
    assert classrooms_needed(1) == 4      # 1,000 people -> 180 pupils -> 3.4 -> 4
    assert classrooms_needed(0) == 0      # edge case: zero population
