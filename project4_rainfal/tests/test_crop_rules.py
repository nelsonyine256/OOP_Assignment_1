"""Tests for CropRule and classification_table."""
import pytest

from rainfall import CropRule, classification_table


def test_classify_month_normal_cases(maize):
    assert maize.classify_month(150) == "Good for maize"
    assert maize.classify_month(50) == "Drought risk"
    assert maize.classify_month(250) == "Waterlogging risk"


def test_boundary_values_count_as_good(maize):
    """Edge case: exactly on the threshold is still suitable."""
    assert maize.status_code(100) == CropRule.GOOD
    assert maize.status_code(200) == CropRule.GOOD


def test_zero_rainfall_is_drought(maize):
    assert maize.status_code(0) == CropRule.DROUGHT


def test_invalid_range_raises_value_error():
    with pytest.raises(ValueError):
        CropRule("Broken", 200, 100, "test")


def test_good_months_for_mbarara(maize, mbarara):
    assert maize.good_months(mbarara) == ["Mar", "Apr", "Sep", "Oct", "Nov"]


def test_classification_table_shape(maize, mbarara):
    table = classification_table([mbarara], [maize])
    assert table.shape == (1, 12)
