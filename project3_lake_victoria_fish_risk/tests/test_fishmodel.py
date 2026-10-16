"""pytest unit tests for src/fishmodel.py. Run from the repo root with:  pytest"""
import sys
from pathlib import Path

import numpy as np

# Make `src` importable when pytest is run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fishmodel import FishStock, PriceModel, RiskAssessor, revenue_stats, weekly_revenue


# ---------- FishStock ----------
def test_msy_and_equilibrium_match_theory():
    """Normal case: MSY = rK/4 = 1000 t and equilibrium = K(1 - h/r)."""
    fish = FishStock(r=0.4, K=10_000, h=0.20)
    assert fish.msy == 1_000
    assert fish.equilibrium_stock == 5_000


def test_stock_converges_to_equilibrium():
    """Normal case: after many weeks the simulated stock sits at the theoretical equilibrium."""
    stock, harvest = FishStock(h=0.10).simulate(weeks=300)
    assert len(stock) == 301 and len(harvest) == 300
    assert abs(stock[-1] - 7_500) < 1


def test_zero_harvest_edge_case():
    """Edge case: h = 0 means no catch and the stock grows but never exceeds K."""
    stock, harvest = FishStock(h=0.0).simulate(weeks=200)
    assert harvest.sum() == 0
    assert stock.max() <= 10_000 + 1e-6


def test_collapse_when_h_exceeds_r():
    """Edge case: h >= r collapses the stock, and it never goes negative."""
    fish = FishStock(h=0.5)
    stock, _ = fish.simulate(weeks=100)
    assert fish.status == "Collapse"
    assert stock.min() >= 0
    assert stock[-1] < 1


def test_closed_season_has_no_harvest():
    """Closed weeks must have exactly zero catch."""
    _, harvest = FishStock(h=0.2).simulate(weeks=52, closed_weeks=range(20, 28))
    assert np.all(harvest[20:28] == 0)
    assert harvest[0] > 0


# ---------- PriceModel ----------
def test_price_stays_within_bounds():
    """Even with huge random steps the price is clipped to [low, high]."""
    prices = PriceModel(step_sd=50_000, seed=1).simulate_paths(n_paths=200, weeks=52)
    assert prices.min() >= 9_000 and prices.max() <= 16_000


def test_same_seed_gives_same_prices():
    """Reproducibility: same seed -> identical path."""
    a = PriceModel(seed=5).simulate_paths(3, 20)
    b = PriceModel(seed=5).simulate_paths(3, 20)
    assert np.array_equal(a, b)


# ---------- Revenue helpers ----------
def test_weekly_revenue_unit_conversion():
    """2 tonnes = 2,000 kg x UGX 10,000 = UGX 20,000,000."""
    assert weekly_revenue(np.array([2.0]), np.array([10_000.0]))[0] == 20_000_000


def test_cv_of_constant_revenue_is_zero():
    """Edge case: no variation means CV = 0."""
    stats = revenue_stats(np.array([5.0, 5.0, 5.0, 5.0]))
    assert stats["CV (no units)"] == 0


def test_cv_has_no_units():
    """CV must not change when revenue is rescaled (e.g. UGX -> UGX millions)."""
    x = np.array([10.0, 12.0, 9.0, 15.0])
    assert np.isclose(revenue_stats(x)["CV (no units)"], revenue_stats(x / 1e6)["CV (no units)"])


# ---------- RiskAssessor ----------
def test_classify_thresholds():
    """Boundaries: 0.10 is Medium, 0.25 is High; below 0.10 is Low."""
    ra = RiskAssessor()
    assert ra.classify(0.05) == "Low"
    assert ra.classify(0.10) == "Medium"
    assert ra.classify(0.25) == "High"


def test_var_is_below_mean_and_constant_case():
    """VaR (5th percentile) is below the mean; for identical values it equals that value."""
    ra = RiskAssessor()
    annual = ra.monte_carlo(FishStock(h=0.2), PriceModel(seed=1), n_paths=1_000)
    assert len(annual) == 1_000
    assert ra.value_at_risk(annual) < annual.mean()
    assert ra.value_at_risk(np.full(100, 7.0)) == 7.0
