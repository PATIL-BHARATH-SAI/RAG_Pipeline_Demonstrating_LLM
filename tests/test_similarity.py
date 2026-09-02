"""Unit tests for similarity metric normalization to percentage."""
import pytest
from app.services.retrieval.similarity import to_percentage

def test_cosine_similarity_percentage():
    assert to_percentage(1.0, metric="cosine") == 100.0
    assert to_percentage(0.0, metric="cosine") == 50.0
    assert to_percentage(-1.0, metric="cosine") == 0.0
    assert to_percentage(0.8, metric="cosine") == 90.0

def test_euclidean_distance_percentage():
    assert to_percentage(0.0, metric="euclidean") == 100.0
    assert to_percentage(1.0, metric="euclidean") == 50.0
    assert 0.0 <= to_percentage(5.0, metric="euclidean") <= 100.0

def test_dot_product_percentage():
    assert to_percentage(0.0, metric="dot") == 50.0
    assert to_percentage(5.0, metric="dot") > 90.0
    assert to_percentage(-5.0, metric="dot") < 10.0
