"""The Region class: one place and its 12 months of rainfall."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .constants import MONTHS


class Region:
    """A geographic region with 12 months of rainfall (mm).

    Attributes
    ----------
    name : str
        Region name, e.g. "Kampala".
    rainfall : numpy.ndarray
        Array of 12 monthly rainfall values in mm (January to December).
    """

    def __init__(self, name: str, monthly_rainfall: ArrayLike) -> None:
        rainfall = np.asarray(monthly_rainfall, dtype=float)   # list -> NumPy array

        # Validate early so mistakes are caught when the object is created
        if rainfall.shape != (12,):
            raise ValueError(f"{name}: expected 12 monthly values, got {rainfall.shape}")
        if (rainfall < 0).any():
            raise ValueError(f"{name}: rainfall cannot be negative")

        self.name = name
        self.rainfall = rainfall

    # ---- the five statistics asked for in the brief ------------------------
    def annual_total(self) -> float:
        """Total rainfall over the year (mm)."""
        return self.rainfall.sum()

    def monthly_mean(self) -> float:
        """Average monthly rainfall (mm)."""
        return self.rainfall.mean()

    def wettest_month(self) -> tuple[str, float]:
        """Return (month name, rainfall in mm) for the wettest month."""
        i = int(np.argmax(self.rainfall))      # argmax = POSITION of the biggest value
        return MONTHS[i], self.rainfall[i]

    def driest_month(self) -> tuple[str, float]:
        """Return (month name, rainfall in mm) for the driest month."""
        i = int(np.argmin(self.rainfall))      # argmin = POSITION of the smallest value
        return MONTHS[i], self.rainfall[i]

    def coefficient_of_variation(self) -> float:
        """Standard deviation / mean, as a percentage.

        Uses the population standard deviation (np.std default, ddof=0)
        because our 12 months are the whole year, not a sample of it.
        """
        return self.rainfall.std() / self.rainfall.mean() * 100

    # ---- convenience --------------------------------------------------------
    def summary(self) -> dict[str, object]:
        """All statistics in one dictionary (handy for building a table)."""
        wet_name, wet_mm = self.wettest_month()
        dry_name, dry_mm = self.driest_month()
        return {
            "Region": self.name,
            "Annual total (mm)": round(self.annual_total(), 1),
            "Monthly mean (mm)": round(self.monthly_mean(), 1),
            "Wettest month": f"{wet_name} ({wet_mm:.0f} mm)",
            "Driest month": f"{dry_name} ({dry_mm:.0f} mm)",
            "CV (%)": round(self.coefficient_of_variation(), 1),
        }

    def __repr__(self) -> str:
        """What you see when you print a Region."""
        return f"Region('{self.name}', annual total = {self.annual_total():.0f} mm)"
