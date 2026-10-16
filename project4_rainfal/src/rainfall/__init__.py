"""Rainfall Pattern & Crop Suitability Analyser (Mini-Project 4)."""
from .constants import MONTHS
from .crop_rules import CropRule, classification_table
from .region import Region
from .seasons import classify_seasonality, find_rainy_seasons
from .similarity import (cosine_similarity, euclidean_distance,
                         pairwise_matrix, pearson_correlation)

__all__ = ["MONTHS", "Region", "CropRule", "classification_table",
           "cosine_similarity", "euclidean_distance", "pearson_correlation",
           "pairwise_matrix", "find_rainy_seasons", "classify_seasonality"]
