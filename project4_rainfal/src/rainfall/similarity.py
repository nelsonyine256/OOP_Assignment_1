"""Similarity and distance measures between rainfall vectors."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .region import Region


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine similarity between two vectors: (a . b) / (|a| * |b|).

    Returns 1 if they point the same way, 0 if they are perpendicular
    (unrelated), and -1 if they point in opposite directions.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Straight-line distance between two vectors (same units as the data, mm)."""
    return np.linalg.norm(np.asarray(a) - np.asarray(b))


def pearson_correlation(a: ArrayLike, b: ArrayLike) -> float:
    """Pearson correlation between two vectors (np.corrcoef returns a 2x2 matrix)."""
    return np.corrcoef(a, b)[0, 1]


def pairwise_matrix(regions: Sequence[Region],
                    measure: Callable[[ArrayLike, ArrayLike], float]) -> pd.DataFrame:
    """Apply `measure(a, b)` to every pair of regions and return a square table."""
    n = len(regions)
    result = np.zeros((n, n))
    for i, region_i in enumerate(regions):
        for j, region_j in enumerate(regions):
            result[i, j] = measure(region_i.rainfall, region_j.rainfall)
    names = [r.name for r in regions]
    return pd.DataFrame(result, index=names, columns=names)
