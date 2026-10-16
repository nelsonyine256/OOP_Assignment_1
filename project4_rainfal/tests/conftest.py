"""Shared test fixtures (reusable sample data for every test file)."""
import pytest

from rainfall import CropRule, Region


@pytest.fixture
def mbarara() -> Region:
    """Mbarara's illustrative rainfall from the assignment brief."""
    return Region("Mbarara", [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90])


@pytest.fixture
def maize() -> CropRule:
    """A maize rule with a 100-200 mm suitable range."""
    return CropRule("Maize", 100, 200, "test source")
