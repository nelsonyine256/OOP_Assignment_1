"""Population data model: one object per district."""

from typing import Sequence

import numpy as np


class DistrictPopulation:
    """Population history of ONE district.

    Parameters
    ----------
    name : str
        District name, e.g. "Kampala".
    years : list or array of int
        The years observed, e.g. [2015, ..., 2024].
    populations : list or array of numbers
        Population (in thousands) for each year, same order as `years`.

    Raises
    ------
    ValueError
        If `years` and `populations` have different lengths, are empty,
        or if any population is negative or missing (NaN).
    """

    def __init__(self, name: str, years: Sequence[int], populations: Sequence[float]) -> None:
        self.name = name
        # np.asarray turns lists into NumPy arrays (and leaves arrays alone)
        self.years = np.asarray(years, dtype=int)
        self.populations = np.asarray(populations, dtype=float)
        self._validate()   # stop immediately if something is wrong

    def _validate(self) -> None:
        """Check the input; raise ValueError with a clear message if it is bad."""
        if len(self.years) != len(self.populations):
            raise ValueError(
                f"{self.name}: years ({len(self.years)}) and populations "
                f"({len(self.populations)}) must have equal lengths."
            )
        if len(self.years) == 0:
            raise ValueError(f"{self.name}: no data supplied.")
        if np.any(np.isnan(self.populations)):
            raise ValueError(f"{self.name}: populations contain missing values.")
        if np.any(self.populations < 0):
            raise ValueError(f"{self.name}: populations cannot be negative.")

    def __len__(self) -> int:
        """Number of years of data, so len(district) works."""
        return len(self.years)

    def __repr__(self) -> str:
        """Readable text shown by print(district) or when the object is displayed."""
        return (f"DistrictPopulation(name='{self.name}', "
                f"years={self.years[0]}-{self.years[-1]}, n={len(self)}, "
                f"latest={self.populations[-1]:,.0f}k)")
