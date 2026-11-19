
"""Analyze recommendation performance for popular and long-tail items."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_all import (
    REL_THRESHOLD,
    load_ratings,
    temporal_split,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
)
from part2.evaluate_validation import paired_bootstrap


OUT_DIR = Path("results/part2")
GROUPS = ["Popular", "Long-tail"]


def main():
    predictions = pd.read_csv(
        OUT_DIR / "validation_per_user.csv"
    )

    if "recommended_items" not in predictions.columns:
        raise ValueError(
            "Rerun evaluate_validation.py to save recommendation lists"
        )

    ratings = load_ratings(Path("data"))
    train, validation, _ = temporal_split(
        ratings, 0.05, 0.05
    )

    movies = pd.read_csv(
        "data/movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["item", "title", "genres"],
        encoding="latin-1",
    )

    # Classify items using training popularity only.
    catalogue = np.sort(movies["item"].unique())

    counts = (
        train.groupby("item")
        .size()
        .reindex(catalogue, fill_value=0)
        .sort_values(ascending=False, kind="stable")
    )

    n_popular = int(np.ceil(0.2 * len(catalogue)))
    popular_items = set(counts.index[:n_popular])

    # Relevant validation items grouped by their popularity.
    relevant = validation[
        validation["rating"] >= REL_THRESHOLD
    ][["user", "item"]].copy()

    relevant["popularity_group"] = np.where(
        relevant["item"].isin(popular_items),
        "Popular",
        "Long-tail",
    )

    ground_truth = {
        (int(user), group): set(items)
        for (user, group), items in relevant.groupby(
            ["user", "popularity_group"]
        )["item"]
    }

    results = []
    exposure = []

    for record in predictions.itertuples(index=False):
        user = int(record.user)
        model = record.model
        ranked = json.loads(record.recommended_items)

        # Share of recommendations belonging to popular items.
        exposure.append({
            "model": model,
            "popular_recommendation_share": (
                sum(item in popular_items for item in ranked)
                / len(ranked)
            ),
        })

        for group in GROUPS:
            truth = ground_truth.get((user, group))

            # Evaluate only users with a relevant item
            # in the respective popularity group.
            if not truth:
                continue

            results.append({
                "model": model,
                "user": user,
                "popularity_group": group,
                "n_relevant": len(truth),
                "precision@10": precision_at_k(
                    ranked, truth
                ),
                "recall@10": recall_at_k(
                    ranked, truth
                ),
                "ndcg@10": ndcg_at_k(
                    ranked, truth
                ),
            })

    per_user = pd.DataFrame(results)

    # Verify identical users across models in each group.
    for group, group_df in per_user.groupby(
        "popularity_group"
    ):
        counts = group_df.groupby("model")["user"].nunique()
        assert counts.nunique() == 1, (
            f"Different evaluation users for {group}"
        )

    summary = (
        per_user.groupby(["popularity_group", "model"])
        .agg(
            n_users=("user", "nunique"),
            precision_at_10=("precision@10", "mean"),
            recall_at_10=("recall@10", "mean"),
            ndcg_at_10=("ndcg@10", "mean"),
        )
        .reset_index()
    )

    exposure_df = (
        pd.DataFrame(exposure)
        .groupby("model", as_index=False)
        .mean()
    )

    bootstrap_rows = []

    for group, group_df in per_user.groupby(
        "popularity_group"
    ):
        comparison = paired_bootstrap(
            group_df, reference="ItemKNN", samples=2000
        )
        comparison.insert(0, "popularity_group", group)
        bootstrap_rows.append(comparison)

    bootstrap_df = pd.concat(
        bootstrap_rows, ignore_index=True
    )

    per_user.to_csv(
        OUT_DIR / "validation_item_popularity_per_user.csv",
        index=False,
    )
    summary.to_csv(
        OUT_DIR / "validation_item_popularity.csv",
        index=False,
    )
    exposure_df.to_csv(
        OUT_DIR / "validation_item_exposure.csv",
        index=False,
    )
    bootstrap_df.to_csv(
        OUT_DIR / "validation_item_popularity_bootstrap.csv",
        index=False,
    )

    print("\nPerformance by item popularity:")
    print(summary.to_string(index=False))

    print("\nRecommendation exposure:")
    print(exposure_df.to_string(index=False))

    print("\nPaired bootstrap by item group:")
    print(bootstrap_df.to_string(index=False))


if __name__ == "__main__":
    main()