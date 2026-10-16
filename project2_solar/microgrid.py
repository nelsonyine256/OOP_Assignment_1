"""Reusable code for Mini-Project 2: Solar Micro-Grid Dispatch Planner.

Everything here is copied verbatim from the notebook cells (no logic changed),
so the notebook and the tests use exactly the same code.
"""

import statistics
import timeit
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve, null_space
from scipy.optimize import nnls

# ---------- SETTINGS ----------
COEFF_MATRIX  = [[3, 2],               # 3x + 2y = D1
                 [4, 1]]
SOURCE_NAMES  = ("solar", "battery")
N_DAYS        = 30
SEED          = 42
CSV_PATH      = Path("demand_30_days.csv")
START_DATE    = "2026-09-01"
SOLAR_PRICE   = 150
BATTERY_PRICE = 450
TOLERANCE     = 1e-9


class MicroGrid:
    """A micro-grid whose daily dispatch is the solution of a square linear system A·v = d.

    Attributes
    ----------
    A : np.ndarray       the coefficient matrix
    labels : tuple       names of the unknowns, e.g. ("solar", "battery")
    det : float          determinant of A
    cond : float         condition number of A
    is_singular : bool   True if A does not have full rank (no unique solution)
    """

    def __init__(self, matrix=COEFF_MATRIX, labels=SOURCE_NAMES):
        self.A = np.array(matrix, dtype=float)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("The coefficient matrix must be square (same rows as columns).")
        self.size = self.A.shape[0]
        self.labels = tuple(labels)

        # The two well-posedness numbers
        self.det = np.linalg.det(self.A)
        self.cond = np.linalg.cond(self.A)
        # matrix_rank is a more trustworthy singularity test than "det == 0"
        self.is_singular = np.linalg.matrix_rank(self.A) < self.size

    # ---------- diagnostics ----------
    def report(self):
        """Print the determinant, condition number and what they mean."""
        print("Coefficient matrix A:")
        print(self.A)
        print(f"\nDeterminant      : {self.det:.4f}")
        print(f"Condition number : {self.cond:.4f}")
        if self.is_singular:
            print("\n-> SINGULAR: the equations are dependent. No unique solution exists.")
        elif self.cond < 10:
            print("\n-> WELL-POSED and WELL-CONDITIONED: one unique solution, and small errors "
                  "in the demands stay small in x and y.")
        elif self.cond < 1000:
            print("\n-> Well-posed but MODERATELY sensitive: expect noticeable error amplification.")
        else:
            print("\n-> Well-posed on paper but ILL-CONDITIONED: tiny demand errors can cause huge "
                  "swings in x and y. Do not trust the numbers blindly.")

    def _require_solvable(self):
        if self.is_singular:
            raise np.linalg.LinAlgError(
                "This system is singular (dependent equations): no unique solution. "
                "Call .report() to see the diagnostics.")

    # ---------- solving ----------
    def solve(self, demand):
        """Solve ONE day for any size of system. `demand` is a list like [D1, D2]."""
        self._require_solvable()
        return solve(self.A, np.asarray(demand, dtype=float))

    def solve_day(self, d1, d2):
        """Solve one day of the 2x2 problem. Returns (x, y) = (solar, battery) in kWh."""
        x, y = self.solve([d1, d2])
        return float(x), float(y)

    def solve_many(self, demands):
        """Solve MANY days in a single call.

        `demands` has shape (n_equations, n_days): each COLUMN is one day.
        Returns an array of the same shape: each column is that day's [x, y].
        """
        self._require_solvable()
        demands = np.asarray(demands, dtype=float)
        if demands.ndim != 2 or demands.shape[0] != self.size:
            raise ValueError(f"`demands` must have shape ({self.size}, n_days).")
        return solve(self.A, demands)      # scipy handles a matrix right-hand side natively

    # ---------- feasibility (used in Part 4) ----------
    def dispatch(self, demands, method="nnls", index=None):
        """Solve many days AND handle infeasible ones (negative energy is physically impossible).

        method="nnls": for infeasible days, use the best non-negative approximation (scipy nnls)
        method="clip": for infeasible days, just set negative values to 0

        Returns a DataFrame with one row per day:
            <labels...>  the energy plan (kWh), never negative
            feasible     True if the exact solution was already non-negative
            residual     how far the plan is from meeting the demands exactly (0 = perfect)
        """
        demands = np.asarray(demands, dtype=float)
        raw = self.solve_many(demands)                         # exact maths answer
        feasible = (raw >= -TOLERANCE).all(axis=0)             # a day is feasible if NO value is negative

        plan = raw.copy()
        if method == "clip":
            plan = np.clip(raw, 0, None)
        elif method == "nnls":
            for day in np.where(~feasible)[0]:                 # only repair the bad days
                plan[:, day], _ = nnls(self.A, demands[:, day])
        else:
            raise ValueError("method must be 'nnls' or 'clip'")
        plan = np.clip(plan, 0, None)                          # removes harmless -1e-15 rounding dust

        residual = np.linalg.norm(demands - self.A @ plan, axis=0)
        table = pd.DataFrame(plan.T, columns=self.labels, index=index)
        table["feasible"] = feasible
        table["residual"] = residual
        return table


def read_positive_number(prompt, input_func=input):
    """Keep asking until the user types a valid, non-negative, finite number. Returns a float."""
    while True:
        text = input_func(prompt).strip()

        if text == "":                                   # 1) empty
            print("  X Empty input. Please type a number.")
            continue
        try:
            value = float(text)                          # 2) numeric?
        except ValueError:
            print(f"  X '{text}' is not a number.")
            continue
        if not np.isfinite(value):                       # 3) catches 'nan' and 'inf'
            print("  X Please type a real number (not nan/inf).")
            continue
        if value < 0:                                    # 4) negative
            print("  X Demand cannot be negative.")
            continue
        return value


def interactive_session(micro_grid, input_func=input):
    """Ask for today's D1 and D2, solve, and say whether the answer is physically feasible."""
    d1 = read_positive_number("Daytime load D1 (kWh): ", input_func)
    d2 = read_positive_number("Critical-equipment load D2 (kWh): ", input_func)
    x, y = micro_grid.solve_day(d1, d2)

    print(f"\nSolar = {x:.2f} kWh | Battery = {y:.2f} kWh")
    if x < -TOLERANCE or y < -TOLERANCE:
        print("Warning: INFEASIBLE day (negative energy). These demands cannot be met exactly.")
    else:
        print("Feasible day.")


def make_fake_input(answers):
    """Build a stand-in for input() that replays a list of answers and echoes them."""
    replies = iter(answers)
    def fake_input(prompt):
        reply = next(replies)
        print(f"{prompt}{reply}")
        return reply
    return fake_input


def generate_demand_csv(path=CSV_PATH, days=N_DAYS, seed=SEED, start=START_DATE):
    """Create a realistic demand CSV with columns: date, d1_kwh, d2_kwh. Returns the DataFrame."""
    rng = np.random.default_rng(seed)                       # seeded -> reproducible
    dates = pd.date_range(start, periods=days, freq="D")
    is_weekday = dates.dayofweek < 5                        # Mon=0 ... Sun=6

    # Typical level for each day of the week, then add noise
    d1 = np.where(is_weekday, 110, 75) + rng.normal(0, 6, days)    # daytime load
    d2 = np.where(is_weekday, 85, 80) + rng.normal(0, 5, days)     # critical load (steady)

    # Special days (4 distinct days chosen at random)
    special = rng.choice(days, size=4, replace=False)
    d1[special[:3]] *= 0.55                                  # clinic closed: daytime load collapses
    d1[special[3]]  *= 1.60                                  # outreach campaign: daytime surge

    table = pd.DataFrame({"date": dates, "d1_kwh": d1.round(1), "d2_kwh": d2.round(1)})
    table.to_csv(path, index=False)
    return table


def load_demand_csv(path=CSV_PATH):
    """Read a demand CSV, validate it, and return a DataFrame (date, d1_kwh, d2_kwh)."""
    table = pd.read_csv(path, parse_dates=["date"])
    needed = {"date", "d1_kwh", "d2_kwh"}
    if not needed.issubset(table.columns):
        raise ValueError(f"CSV must contain the columns {sorted(needed)}")
    values = table[["d1_kwh", "d2_kwh"]]
    if values.isna().any().any() or (values < 0).any().any():
        raise ValueError("CSV contains missing or negative demand values.")
    return table


def to_demand_matrix(table):
    """Turn the table into the (2, n_days) array that MicroGrid.solve_many expects."""
    return table[["d1_kwh", "d2_kwh"]].to_numpy().T


def solve_with_loop(micro_grid, demands):
    """Solve day by day in a Python loop. Returns shape (2, n_days) like solve_many."""
    results = [micro_grid.solve_day(d1, d2) for d1, d2 in demands.T]   # .T so each row is one day
    return np.array(results).T


def describe(values):
    """Return mean, variance, standard deviation and coefficient of variation of a list of numbers."""
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    return {"mean (kWh)": mean,
            "variance (kWh^2)": statistics.variance(values),
            "std dev (kWh)": std,
            "coeff. of variation": std / mean}


def daily_cost(plan_table, solar_price=SOLAR_PRICE, battery_price=BATTERY_PRICE):
    """Return a Series: UGX cost for each day, from the plan's solar and battery kWh."""
    return solar_price * plan_table["solar"] + battery_price * plan_table["battery"]


def plot_dispatch(plan_table, cost_series, title="Daily solar vs battery usage and cost"):
    """Stacked bars of solar + battery (left axis) and daily cost line (right axis)."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    days = np.arange(len(plan_table))

    ax.bar(days, plan_table["solar"], color="#f5b301", label="Solar (kWh)")
    ax.bar(days, plan_table["battery"], bottom=plan_table["solar"],
           color="#3b7ddd", label="Battery (kWh)")

    # Star above infeasible days
    totals = plan_table["solar"] + plan_table["battery"]
    for d in days[~plan_table["feasible"].to_numpy()]:
        ax.text(d, totals.iloc[d] + 1.5, "*", ha="center", color="red", fontsize=16)

    ax.set_xticks(days)
    ax.set_xticklabels(plan_table.index.strftime("%d %b"), rotation=90)
    ax.set_xlabel("Day")
    ax.set_ylabel("Energy used (kWh)")
    ax.set_title(title)

    ax2 = ax.twinx()                                   # second y-axis on the right
    ax2.plot(days, cost_series.to_numpy(), color="crimson", marker="o", label="Daily cost (UGX)")
    ax2.set_ylabel("Daily cost (UGX)")

    # One combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.text(0.99, 0.01, "* = infeasible day (repaired with NNLS)", ha="right", color="red")

    fig.tight_layout()
    plt.show()


class HybridMicroGrid(MicroGrid):
    """MicroGrid extended with a diesel generator: a 3x3 system (solar, battery, diesel)."""

    DEFAULT_MATRIX = [[3, 2, 1],
                      [4, 1, 2],
                      [1, 1, 1]]

    def __init__(self, matrix=DEFAULT_MATRIX, labels=("solar", "battery", "diesel")):
        super().__init__(matrix, labels)          # parent does the checks (det, cond, rank)

    def solve_day(self, d1, d2, d3):
        """Solve one day. Returns (solar, battery, diesel) in kWh."""
        return tuple(float(v) for v in self.solve([d1, d2, d3]))

    def solutions_when_dependent(self, demand):
        """For a SINGULAR system: report whether it has infinitely many solutions or none.

        Uses least squares (lstsq). If the residual is ~0, the demands are consistent
        (infinitely many solutions); otherwise the equations contradict each other (none).
        """
        demand = np.asarray(demand, dtype=float)
        best_fit, *_ = np.linalg.lstsq(self.A, demand, rcond=None)
        gap = np.linalg.norm(self.A @ best_fit - demand)
        return best_fit, gap


def sensitivity_analysis(micro_grid, demand, pct=0.05, draws=1000, seed=SEED):
    """Monte Carlo test of how demand errors affect the dispatch.

    Returns a dict with the baseline answer, all perturbed answers, and the
    'amplification' of every trial (relative output change / relative input change).
    """
    rng = np.random.default_rng(seed)
    demand = np.asarray(demand, dtype=float)

    factors = rng.uniform(1 - pct, 1 + pct, size=(len(demand), draws))   # e.g. 0.97, 1.04, ...
    noisy = demand[:, None] * factors                                    # perturbed demands
    answers = micro_grid.solve_many(noisy)                               # all trials at once
    base = micro_grid.solve(demand)

    rel_in = np.linalg.norm(noisy - demand[:, None], axis=0) / np.linalg.norm(demand)
    rel_out = np.linalg.norm(answers - base[:, None], axis=0) / np.linalg.norm(base)
    return {"base": base, "answers": answers, "amplification": rel_out / rel_in}


def run_planner(csv_path=CSV_PATH, matrix=COEFF_MATRIX, solar_price=SOLAR_PRICE,
                battery_price=BATTERY_PRICE, method="nnls"):
    """Load a demand CSV, build a MicroGrid, and return (plan table, daily cost, monthly cost)."""
    table = load_demand_csv(csv_path)
    micro_grid = MicroGrid(matrix)
    plan_table = micro_grid.dispatch(to_demand_matrix(table), method=method, index=table["date"])
    cost_series = daily_cost(plan_table, solar_price, battery_price)
    return plan_table, cost_series, cost_series.sum()
