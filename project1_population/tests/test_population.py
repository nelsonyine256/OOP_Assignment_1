"""Tests for DistrictPopulation (normal case, dunder methods, invalid input)."""
import numpy as np
import pytest

from src.population import DistrictPopulation


def test_valid_district_has_len_and_repr():
    d = DistrictPopulation("Gulu", [2015, 2016, 2017], [320, 330, 345])
    assert len(d) == 3
    assert "Gulu" in repr(d) and "2015-2017" in repr(d)
    assert isinstance(d.populations, np.ndarray)


def test_unequal_lengths_rejected():
    with pytest.raises(ValueError):
        DistrictPopulation("X", [2015, 2016, 2017], [100, 110])


def test_negative_population_rejected():
    with pytest.raises(ValueError):
        DistrictPopulation("X", [2015, 2016], [100, -5])


def test_empty_input_rejected():   # edge case
    with pytest.raises(ValueError):
        DistrictPopulation("X", [], [])
