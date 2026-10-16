"""Tests for rainy-season detection."""
from rainfall import classify_seasonality, find_rainy_seasons


def test_mbarara_is_bimodal(mbarara):
    assert classify_seasonality(mbarara.rainfall) == "bimodal"


def test_one_clear_peak_is_unimodal():
    single = [10, 20, 40, 80, 150, 200, 150, 80, 40, 20, 10, 5]
    assert classify_seasonality(single) == "unimodal"


def test_flat_rainfall_has_no_clear_season():
    """Edge case: constant rainfall has no peaks at all."""
    assert classify_seasonality([100] * 12) == "no clear season"


def test_december_peak_is_found_thanks_to_year_wrapping():
    """A plain find_peaks call ignores the last month; our wrapping must not."""
    rain = [50, 10, 10, 10, 10, 10, 10, 10, 10, 10, 60, 100]
    months, _ = find_rainy_seasons(rain)
    assert 11 in months          # index 11 = December
