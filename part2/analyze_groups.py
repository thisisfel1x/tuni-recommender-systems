
"""Analyze recommendation quality across user activity groups."""

from pathlib import Path

import pandas as pd

from part2.evaluate_validation import paired_bootstrap


OUT_DIR = Path("results/part2")


def main():
    df = pd.read_csv(
        OUT_DIR / "validation_per_user.csv"
    )

    # Use the training history of each evaluated user.
    users = (
        df[df["model"] == "ItemKNN"][
            ["user", "train_history_size"]
        ]
        .drop_duplicates("user")
        .copy()
    )

    # Split users into three activity groups.
    users["activity_group"] = pd.qcut(
        users["train_history_size"],
        q=3,
        labels=["Low", "Medium", "High"],
    )

    print("\nUser activity groups:")
    print(
        users.groupby("activity_group", observed=True)
        ["train_history_size"]
        .agg(["count", "min", "median", "max"])
    )

    # Attach the same group labels to every model.
    df = df.merge(
        users[["user", "activity_group"]],
        on="user",
        how="inner",
        validate="many_to_one",
    )

    # Check that all models are evaluated on the same users.
    user_counts = df.groupby("model")["user"].nunique()
    assert user_counts.nunique() == 1

    # Calculate metrics for each activity group.
    summary = (
        df.groupby(
            ["activity_group", "model"],
            observed=True,
        )
        .agg(
            n_users=("user", "nunique"),
            precision_at_10=("precision@10", "mean"),
            recall_at_10=("recall@10", "mean"),
            ndcg_at_10=("ndcg@10", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT_DIR / "validation_user_activity.csv",
        index=False,
    )

    print("\nPerformance by user activity:")
    print(summary.to_string(index=False))

    # Paired bootstrap inside each activity group.
    bootstrap_results = []

    for group, group_df in df.groupby(
        "activity_group", observed=True
    ):
        comparison = paired_bootstrap(
            group_df,
            reference="ItemKNN",
            samples=2000,
        )

        comparison.insert(
            0, "activity_group", str(group)
        )
        bootstrap_results.append(comparison)

    bootstrap_df = pd.concat(
        bootstrap_results,
        ignore_index=True,
    )

    bootstrap_df.to_csv(
        OUT_DIR / "validation_user_activity_bootstrap.csv",
        index=False,
    )

    print("\nBootstrap by activity group:")
    print(bootstrap_df.to_string(index=False))


if __name__ == "__main__":
    main()