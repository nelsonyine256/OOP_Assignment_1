"""Tests for the Region class."""
import numpy as np
import pytest

from rainfall import Region


def test_annual_total_and_mean_normal_case(mbarara):
    assert mbarara.annual_total() == 1040
    assert mbarara.monthly_mean() == pytest.approx(1040 / 12)


def test_wettest_and_driest_month(mbarara):
    assert mbarara.wettest_month() == ("Apr", 140)
    assert mbarara.driest_month() == ("Jul", 20)


def test_cv_is_zero_when_every_month_is_equal():
    flat = Region("Flat", [100] * 12)
    assert flat.coefficient_of_variation() == 0


def test_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        Region("Empty", [])


def test_wrong_length_raises_value_error():
    with pytest.raises(ValueError):
        Region("Short", [10, 20, 30])


def test_negative_rainfall_raises_value_error():
    with pytest.raises(ValueError):
        Region("Bad", [-1] + [10] * 11)


def test_cv_of_all_zero_rainfall_is_nan():
    """Edge case: zero mean means 0/0, so the CV is undefined (nan) and NumPy warns."""
    dry = Region("Desert", [0] * 12)
    with pytest.warns(RuntimeWarning):
        assert np.isnan(dry.coefficient_of_variation())


def test_repr_contains_name():
    assert "Mbarara" in repr(Region("Mbarara", [10] * 12))
