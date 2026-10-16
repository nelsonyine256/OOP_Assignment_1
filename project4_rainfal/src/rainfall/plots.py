"""Charts: rainfall lines, crop-suitability heatmap and monthly box plots."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .constants import MONTHS
from .crop_rules import CropRule, classification_table
from .region import Region
from .seasons import find_rainy_seasons


def plot_rainfall_lines(regions: Sequence[Region], mark_peaks: bool = True,
                        min_prominence: float = 40) -> None:
    """Line chart of monthly rainfall for every region (stars mark detected rainy-season peaks)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for region in regions:
        line, = ax.plot(MONTHS, region.rainfall, marker="o", linewidth=2, label=region.name)
        if mark_peaks:
            positions, _ = find_rainy_seasons(region.rainfall, min_prominence)
            ax.plot([MONTHS[p] for p in positions], region.rainfall[positions],
                    marker="*", markersize=18, linestyle="none", color=line.get_color())

    ax.set_title("Monthly rainfall by region (stars = detected rainy-season peaks)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rainfall (mm)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_suitability_heatmap(regions: Sequence[Region], crops: Sequence[CropRule]) -> None:
    """Heatmap: one row per (region, crop), one column per month.

    Colour = status (orange drought, green good, blue waterlogging).
    The number in each cell is that month's rainfall in mm.
    """
    codes = classification_table(regions, crops, as_codes=True)
    rainfall_lookup = {r.name: r.rainfall for r in regions}

    colours = ["#e9a03b", "#4caf50", "#3b7dd8"]          # index 0, 1, 2 = drought, good, waterlogging
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.imshow(codes.values, cmap=ListedColormap(colours), vmin=0, vmax=2, aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([f"{region} - {crop}" for region, crop in codes.index])

    # Write the rainfall (mm) inside each cell
    for row, (region_name, _crop) in enumerate(codes.index):
        for col in range(12):
            ax.text(col, row, f"{rainfall_lookup[region_name][col]:.0f}",
                    ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    ax.legend(handles=[Patch(color=colours[0], label="Drought risk"),
                       Patch(color=colours[1], label="Good"),
                       Patch(color=colours[2], label="Waterlogging risk")],
              loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    ax.set_title("Crop suitability by month and region (numbers = rainfall in mm)")
    plt.tight_layout()
    plt.show()


def plot_monthly_boxplots(yearly_rainfall: pd.DataFrame, region_name: str) -> None:
    """Box plot of rainfall for each month across many years.

    Parameters
    ----------
    yearly_rainfall : DataFrame with one row per year and 12 columns (Jan..Dec).
    region_name : str, used in the chart title.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([yearly_rainfall[m] for m in MONTHS])
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS)
    ax.set_title(f"Year-to-year rainfall variability: {region_name}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rainfall (mm)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()
