"""Tests for reproducible user-stratified sparsity sampling."""

import pandas as pd
import pytest

from part2.sparsity_ablation import sample_training_sets


def test_nested_sparsity_samples_retain_users():
    train = pd.DataFrame({
        "user": [1] * 5 + [2] * 10 + [3] * 2,
        "item": list(range(17)),
        "rating": [5] * 17,
        "timestamp": list(range(17)),
    })
    result = sample_training_sets(train, seed=123)
    ids = {frac: set(df["timestamp"]) for frac, df in result.items()}

    assert ids[0.2] <= ids[0.5] <= ids[1.0]
    assert len(result[1.0]) == len(train)
    assert all(set(df["user"]) == {1, 2, 3} for df in result.values())
    assert ids == {
        fraction: set(df["timestamp"])
        for fraction, df in sample_training_sets(train, seed=123).items()
    }


def test_sparsity_invalid_fraction():
    train = pd.DataFrame({"user": [1], "rating": [5]})
    with pytest.raises(ValueError):
        sample_training_sets(train, fractions=(0.0,))
