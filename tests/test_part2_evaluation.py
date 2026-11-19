
"""Tests for the common Part 2 evaluation."""

import numpy as np
import pandas as pd

from run_all import evaluate, index_data, most_popular
from part2.evaluate_validation import evaluate_per_user, paired_bootstrap


def test_per_user_matches_part1_metrics():
    train = pd.DataFrame({
        "user": [1, 1, 2, 2],
        "item": [10, 20, 10, 30],
        "rating": [5, 4, 5, 4],
        "timestamp": [1, 2, 3, 4],
    })
    val = pd.DataFrame({
        "user": [1, 2, 3],
        "item": [30, 20, 40],
        "rating": [5, 5, 5],
        "timestamp": [5, 6, 7],
    })
    data = index_data(train, [10, 20, 30, 40])
    model = most_popular(data, min_rating=4)

    reference = evaluate(model, data, val)
    per_user, coverage, cold_skipped = evaluate_per_user(
        model, data, val, batch_size=1
    )

    for metric in ("precision@10", "recall@10", "ndcg@10"):
        assert np.isclose(per_user[metric].mean(), reference[metric])
    assert np.isclose(coverage, reference["coverage"])
    assert cold_skipped == reference["n_cold_users_skipped"] == 1
    assert set(per_user["user"]) == {1, 2}


def test_paired_bootstrap_zero_difference():
    rows = []
    for user, ndcg in [(1, 0.1), (2, 0.4), (3, 0.2)]:
        for model in ("ItemKNN", "Other"):
            rows.append({"user": user, "model": model, "ndcg@10": ndcg})
    result = paired_bootstrap(pd.DataFrame(rows), samples=100)

    assert len(result) == 1
    assert result.iloc[0]["mean_ndcg_difference"] == 0
    assert result.iloc[0]["ci95_low"] == 0
    assert result.iloc[0]["ci95_high"] == 0