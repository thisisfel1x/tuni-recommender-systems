
"""Evaluate recommendation robustness to reduced training interactions.

Run from the repository root:
    python -m part2.sparsity_ablation

No test ratings are used. Hyperparameters stay fixed at the chosen
full-training validation settings, so the experiment isolates data scarcity.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from part1.run_all import (
    K,
    REL_THRESHOLD,
    index_data,
    item_knn,
    load_ratings,
    most_popular,
    temporal_split,
)
from part2.evaluate_validation import evaluate_per_user
from part2.models import (
    content_based,
    matrix_factorization,
    user_knn,
)


DATA_DIR = Path("data")
OUT_DIR = Path("results/part2")
SEED = 123
FRACTIONS = (1.0, 0.5, 0.2)


def sample_training_sets(
    train: pd.DataFrame,
    fractions=FRACTIONS,
    seed=SEED,
):
    """Create nested user-stratified training subsets.

    Every user retains at least one rating.
    Sampling is reproducible through the fixed random seed.
    """

    if train.empty:
        raise ValueError(
            "Training data must not be empty"
        )

    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError(
            "Fractions must be in (0, 1]"
        )

    rng = np.random.default_rng(seed)

    # One random ordering per user.
    user_orderings = [
        rng.permutation(positions)
        for _, positions in train.groupby(
            "user",
            sort=True,
        ).indices.items()
    ]

    samples = {}

    for fraction in fractions:
        selected = np.concatenate([
            positions[:max(
                1,
                int(np.ceil(len(positions) * fraction)),
            )]
            for positions in user_orderings
        ])

        samples[fraction] = train.iloc[
            np.sort(selected)
        ].copy()

    return samples


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ratings = load_ratings(DATA_DIR)

    movies = pd.read_csv(
        DATA_DIR / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["item", "title", "genres"],
        encoding="latin-1",
    )

    train, validation, _ = temporal_split(
        ratings,
        0.05,
        0.05,
    )

    catalogue = movies["item"].to_numpy()

    # Full training data defines the fixed seen-item mask.
    full_data = index_data(
        train,
        catalogue,
    )

    # Use identical validation users across conditions.
    eligible_users = set(
        validation.loc[
            validation["rating"] >= REL_THRESHOLD,
            "user",
        ]
    ) & set(train["user"])

    summary_rows = []
    user_tables = []

    for fraction, subset in sample_training_sets(train).items():
        data = index_data(
            subset,
            catalogue,
        )

        # Verify consistent user and item indexing.
        assert np.array_equal(
            data.user_ids,
            full_data.user_ids,
        )

        assert np.array_equal(
            data.item_ids,
            full_data.item_ids,
        )

        print(
            f"\nTraining fraction {fraction:.0%}: "
            f"{len(subset):,}/{len(train):,} interactions "
            f"({len(subset) / len(train):.1%} actual)",
            flush=True,
        )

        # Keep previously selected hyperparameters fixed.
        builders = {
            "MostPopular": lambda: most_popular(
                data,
                min_rating=4,
            ),
            "ItemKNN": lambda: item_knn(
                data,
                k=500,
                similarity="cosine",
            ),
            "ContentBased": lambda: content_based(
                data,
                movies,
                year_weight=1.0,
            ),
            "MatrixFactorization": lambda: matrix_factorization(
                data,
                n_factors=32,
                n_epochs=24,
                learning_rate=0.005,
                reg=0.01,
                seed=SEED,
            ),
            "UserKNN": lambda: user_knn(
                data,
                k=200,
                similarity="cosine",
            ),
        }

        for name, build in builders.items():
            print(
                f"  Evaluating {name}...",
                flush=True,
            )

            per_user, coverage, cold_skipped = evaluate_per_user(
                build(),
                data,
                validation,
                seen_override=full_data.seen,
            )

            if set(per_user["user"]) != eligible_users:
                raise AssertionError(
                    "Validation users changed between fractions"
                )

            summary_rows.append({
                "train_fraction": fraction,
                "train_interactions": len(subset),
                "actual_train_fraction": len(subset) / len(train),
                "model": name,
                f"precision@{K}": per_user[
                    f"precision@{K}"
                ].mean(),
                f"recall@{K}": per_user[
                    f"recall@{K}"
                ].mean(),
                f"ndcg@{K}": per_user[
                    f"ndcg@{K}"
                ].mean(),
                "coverage": coverage,
                "n_users": len(per_user),
                "n_cold_users_skipped": cold_skipped,
            })

            per_user.insert(
                0,
                "model",
                name,
            )

            per_user.insert(
                0,
                "train_fraction",
                fraction,
            )

            user_tables.append(per_user)

    summary = pd.DataFrame(summary_rows)

    # Compare reduced-training results with full training.
    full_ndcg = summary.loc[
        summary["train_fraction"] == 1.0
    ].set_index("model")[f"ndcg@{K}"]

    summary["ndcg_change_vs_full"] = (
        summary[f"ndcg@{K}"]
        - summary["model"].map(full_ndcg)
    )

    summary = summary.sort_values(
        ["model", "train_fraction"],
        ascending=[True, False],
    )

    summary.to_csv(
        OUT_DIR / "sparsity_summary.csv",
        index=False,
    )

    pd.concat(
        user_tables,
        ignore_index=True,
    ).to_csv(
        OUT_DIR / "sparsity_per_user.csv",
        index=False,
    )

    print("\nSparsity ablation summary:")

    print(
        summary[[
            "train_fraction",
            "model",
            f"ndcg@{K}",
            "ndcg_change_vs_full",
            "coverage",
            "n_users",
        ]].to_string(index=False)
    )

    print(f"\nSaved files in {OUT_DIR}")


if __name__ == "__main__":
    main()