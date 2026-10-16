"""Rainy-season detection using scipy.signal.find_peaks."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks


def find_rainy_seasons(rainfall: ArrayLike,
                       min_prominence: float = 40) -> tuple[NDArray, NDArray]:
    """Find the rainy-season peaks in 12 months of rainfall.

    Parameters
    ----------
    rainfall : array of 12 monthly values (mm).
    min_prominence : float
        Minimum height (mm) a peak must stand above its surroundings to count
        as a rainy season. This is the one setting you may need to tune.

    Returns
    -------
    (month_positions, prominences) : two arrays, one entry per season found.
    """
    wrapped = np.tile(rainfall, 3)                       # Jan-Dec, Jan-Dec, Jan-Dec
    peaks, props = find_peaks(wrapped, prominence=min_prominence)

    in_middle_year = (peaks >= 12) & (peaks < 24)        # keep only peaks in the middle copy
    month_positions = peaks[in_middle_year] - 12         # convert back to 0..11
    prominences = props["prominences"][in_middle_year]
    return month_positions, prominences


def classify_seasonality(rainfall: ArrayLike, min_prominence: float = 40) -> str:
    """Return 'unimodal', 'bimodal', etc. from the number of rainy seasons found."""
    n_peaks = len(find_rainy_seasons(rainfall, min_prominence)[0])
    names = {0: "no clear season", 1: "unimodal", 2: "bimodal"}
    return names.get(n_peaks, f"{n_peaks} seasons")
