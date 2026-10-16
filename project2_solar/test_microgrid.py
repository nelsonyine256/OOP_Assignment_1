"""Unit tests for the Solar Micro-Grid Dispatch Planner.

Run from the repository root with:   pytest
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from microgrid import (HybridMicroGrid, MicroGrid, generate_demand_csv,
                       load_demand_csv, make_fake_input, read_positive_number)


# ---------- normal cases ----------
def test_solve_day_normal_case():
    """D1=100, D2=85 has the exact answer solar=14, battery=29 (checked by hand)."""
    x, y = MicroGrid().solve_day(100, 85)
    assert np.allclose([x, y], [14, 29])


def test_solve_many_matches_day_by_day():
    """The vectorised call must give the same answers as solving each day separately."""
    grid = MicroGrid()
    demands = np.array([[100, 90, 120], [85, 80, 95]], dtype=float)   # 2 equations x 3 days
    one_by_one = np.array([grid.solve_day(d1, d2) for d1, d2 in demands.T]).T
    assert np.allclose(grid.solve_many(demands), one_by_one)


def test_hybrid_known_answer():
    """We planted solar=20, battery=15, diesel=10 and check the 3x3 solver finds it."""
    result = HybridMicroGrid().solve_day(100, 115, 45)
    assert np.allclose(result, [20, 15, 10])


def test_dispatch_repairs_infeasible_day():
    """Day 2 has D2 > 1.33*D1, so the exact battery value is negative. The plan must not be."""
    demands = np.array([[100, 60], [85, 90]], dtype=float)
    plan = MicroGrid().dispatch(demands, method="nnls")
    assert list(plan["feasible"]) == [True, False]
    assert (plan[["solar", "battery"]] >= 0).all().all()
    clipped = MicroGrid().dispatch(demands, method="clip")
    assert plan["residual"].iloc[1] <= clipped["residual"].iloc[1]   # nnls is never worse


# ---------- edge cases ----------
def test_zero_demand_gives_zero_dispatch():
    """Edge case: no demand at all means no energy is needed."""
    assert np.allclose(MicroGrid().solve_day(0, 0), [0, 0])


def test_singular_matrix_is_refused():
    """Edge case: dependent equations (row 2 = 2 x row 1) have no unique solution."""
    grid = MicroGrid([[1, 2], [2, 4]])
    assert grid.is_singular
    with pytest.raises(np.linalg.LinAlgError):
        grid.solve_day(10, 20)


def test_non_square_matrix_is_rejected():
    """Invalid value: a 2x3 matrix cannot define a unique dispatch."""
    with pytest.raises(ValueError):
        MicroGrid([[1, 2, 3], [4, 5, 6]])


def test_input_validation_rejects_bad_text():
    """Empty, non-numeric, nan and negative entries are rejected; the first good one wins."""
    fake = make_fake_input(["", "abc", "nan", "-5", "42"])
    assert read_positive_number("D1: ", fake) == 42.0


def test_csv_is_reproducible_and_validated():
    """Same seed -> same file contents; a CSV with negative demand is refused."""
    with tempfile.TemporaryDirectory() as folder:
        a = generate_demand_csv(Path(folder) / "a.csv", days=30, seed=1)
        b = generate_demand_csv(Path(folder) / "b.csv", days=30, seed=1)
        assert a.equals(b)
        assert len(load_demand_csv(Path(folder) / "a.csv")) == 30

        bad = Path(folder) / "bad.csv"
        bad.write_text("date,d1_kwh,d2_kwh\n2026-09-01,-5,80\n")
        with pytest.raises(ValueError):
            load_demand_csv(bad)
