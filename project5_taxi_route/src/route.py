"""Route model: daily passenger counts, a fixed fare, revenue and basic statistics."""

import statistics
from typing import Dict, Sequence

import numpy as np


class Route:
    """A taxi route with daily passenger counts and a fixed fare.

    Parameters
    ----------
    name : str
        Route name, e.g. "Kampala-Ntinda".
    passengers : Sequence[float]
        Passengers carried on each day (day 1 first).
    fare : float
        Fare per passenger in UGX.
    """

    def __init__(self, name: str, passengers: Sequence[float], fare: float) -> None:
        self.name = name
        self.passengers = np.array(passengers, dtype=float)  # NumPy array, as required
        self.fare = fare

    def daily_revenue(self) -> np.ndarray:
        """Revenue for each day (UGX) = passengers x fare."""
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        """Total revenue over all days (UGX)."""
        return self.daily_revenue().sum()

    def stats(self) -> Dict[str, float]:
        """Descriptive statistics of daily passenger counts (uses the statistics module)."""
        counts = list(self.passengers)  # statistics module likes plain lists
        return {
            "mean": statistics.mean(counts),
            "variance": statistics.variance(counts),
            "std_dev": statistics.stdev(counts),
        }

    def __repr__(self) -> str:
        return f"Route({self.name!r}, {len(self.passengers)} days, fare=UGX {self.fare:,})"
