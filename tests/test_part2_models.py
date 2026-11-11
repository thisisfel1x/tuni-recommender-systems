
"""Tests for Part 2 recommendation models."""

import numpy as np
import pandas as pd

from run_all import index_data
from part2_models import content_based


def test_content_based_scores():
    ratings = pd.DataFrame({
        "user": [1, 1, 2],
        "item": [10, 20, 20],
        "rating": [5, 2, 5],
        "timestamp": [1, 2, 3],
    })

    movies = pd.DataFrame({
        "item": [10, 20, 30],
        "title": [
            "Action A (2000)",
            "Comedy B (1990)",
            "Action C (2000)",
        ],
        "genres": ["Action", "Comedy", "Action"],
    })

    data = index_data(ratings, movies["item"].to_numpy())

    model = content_based(data, movies, year_weight=1.0)
    scores = model(np.array([0, 1]))

    # One score per user and catalogue item.
    assert scores.shape == (2, 3)
    assert np.isfinite(scores).all()

    # User 1 liked an action movie.
    # The unseen action movie should score higher than comedy.
    action = np.where(data.item_ids == 30)[0][0]
    comedy = np.where(data.item_ids == 20)[0][0]

    assert scores[0, action] > scores[0, comedy]


def test_content_based_invalid_weight():
    ratings = pd.DataFrame({
        "user": [1],
        "item": [10],
        "rating": [5],
        "timestamp": [1],
    })

    movies = pd.DataFrame({
        "item": [10],
        "title": ["Film (2000)"],
        "genres": ["Action"],
    })

    data = index_data(ratings, movies["item"].to_numpy())

    import pytest

    with pytest.raises(ValueError):
        content_based(data, movies, year_weight=-1.0)