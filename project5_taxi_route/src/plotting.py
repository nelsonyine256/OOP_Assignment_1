"""Plotting helper for actual vs forecast revenue."""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.planning import backtest
from src.route import Route


def plot_actual_vs_forecast(routes: List[Route], day11: Dict[str, dict], first_day: int = 4):
    """One panel per route: actual revenue, backtest forecast, and day-11 forecast.

    `day11[route.name]` must hold the chosen "model", and the day-11 "revenue".
    Returns the matplotlib Figure (call plt.show() to display it).
    """
    fig, axes = plt.subplots(len(routes), 1, figsize=(9, 3.3 * len(routes)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, routes):
        model = day11[r.name]["model"]
        actual_rev = r.daily_revenue()
        days = np.arange(1, len(actual_rev) + 1)

        # Backtest forecasts (passengers -> revenue)
        bt_forecasts, _ = backtest(model, r.passengers, first_day)
        bt_days = np.arange(first_day, len(actual_rev) + 1)

        ax.plot(days, actual_rev, "o-", label="Actual revenue")
        ax.plot(bt_days, bt_forecasts * r.fare, "s--", label=f"Backtest forecast: {model}")
        ax.plot(len(actual_rev) + 1, day11[r.name]["revenue"], "r*", markersize=15,
                label=f"Day 11 forecast: UGX {day11[r.name]['revenue']:,.0f}")

        ax.set_title(r.name)
        ax.set_ylabel("Revenue (UGX)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Day")
    fig.tight_layout()
    return fig
