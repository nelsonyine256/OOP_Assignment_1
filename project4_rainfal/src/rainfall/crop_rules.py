"""The CropRule class and a helper that classifies many regions and crops at once."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from .constants import MONTHS
from .region import Region


class CropRule:
    """Rainfall rule for one crop: a suitable range of monthly rainfall (mm).

    Parameters
    ----------
    crop : str
        Crop name, e.g. "Maize".
    min_mm, max_mm : float
        Lowest and highest monthly rainfall that still suits the crop.
    source : str
        Citation that justifies the thresholds.
    """

    # Status codes. Using named constants avoids "magic numbers" in the code.
    DROUGHT, GOOD, WATERLOGGING = 0, 1, 2

    def __init__(self, crop: str, min_mm: float, max_mm: float, source: str) -> None:
        if min_mm >= max_mm:
            raise ValueError("min_mm must be smaller than max_mm")
        self.crop = crop
        self.min_mm = min_mm
        self.max_mm = max_mm
        self.source = source

    def status_code(self, mm: float) -> int:
        """Return DROUGHT, GOOD or WATERLOGGING (0, 1 or 2) for one month's rainfall."""
        if mm < self.min_mm:
            return self.DROUGHT
        if mm > self.max_mm:
            return self.WATERLOGGING
        return self.GOOD

    def classify_month(self, mm: float) -> str:
        """Return a readable label for one month, e.g. 'Good for maize'."""
        code = self.status_code(mm)
        if code == self.GOOD:
            return f"Good for {self.crop.lower()}"
        if code == self.DROUGHT:
            return "Drought risk"
        return "Waterlogging risk"

    def classify_region(self, region: Region) -> list[str]:
        """Return the 12 monthly labels for a Region."""
        return [self.classify_month(mm) for mm in region.rainfall]

    def codes_for_region(self, region: Region) -> list[int]:
        """Return the 12 monthly status codes (numbers, used for the heatmap)."""
        return [self.status_code(mm) for mm in region.rainfall]

    def good_months(self, region: Region) -> list[str]:
        """Return the names of the months that suit this crop."""
        return [m for m, c in zip(MONTHS, self.codes_for_region(region)) if c == self.GOOD]

    def __repr__(self) -> str:
        return f"CropRule({self.crop}: {self.min_mm}-{self.max_mm} mm/month)"


def classification_table(regions: Sequence[Region], crops: Sequence[CropRule],
                         as_codes: bool = False) -> pd.DataFrame:
    """Classify every month for every (region, crop) pair.

    Returns a table with one row per (region, crop) and one column per month.
    If as_codes is True the cells hold 0/1/2 instead of text (for the heatmap).
    """
    rows = {}
    for region in regions:
        for crop in crops:
            if as_codes:
                rows[(region.name, crop.crop)] = crop.codes_for_region(region)
            else:
                rows[(region.name, crop.crop)] = crop.classify_region(region)
    return pd.DataFrame(rows, index=MONTHS).T     # .T flips so months run across the columns
