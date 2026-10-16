"""Tests for the similarity and distance functions."""
import numpy as np
import pytest
from scipy.spatial import distance

from rainfall import (cosine_similarity, euclidean_distance, pairwise_matrix,
                      pearson_correlation)


def test_cosine_of_identical_vectors_is_one():
    v = [120, 140, 180]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_of_perpendicular_vectors_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_matches_scipy(mbarara):
    other = np.array([120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130])
    expected = 1 - distance.cosine(mbarara.rainfall, other)
    assert cosine_similarity(mbarara.rainfall, other) == pytest.approx(expected)


def test_cosine_ignores_scale_but_euclidean_does_not():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, 2 * v) == pytest.approx(1.0)
    assert euclidean_distance(v, 2 * v) > 0


def test_pearson_of_perfectly_linear_data_is_one():
    assert pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_euclidean_distance_to_self_is_zero():
    assert euclidean_distance([5, 6, 7], [5, 6, 7]) == 0


def test_pairwise_matrix_is_symmetric_with_unit_diagonal(mbarara):
    from rainfall import Region
    other = Region("Other", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    matrix = pairwise_matrix([mbarara, other], cosine_similarity)
    assert matrix.shape == (2, 2)
    assert matrix.loc["Mbarara", "Other"] == pytest.approx(matrix.loc["Other", "Mbarara"])
    assert matrix.loc["Mbarara", "Mbarara"] == pytest.approx(1.0)
