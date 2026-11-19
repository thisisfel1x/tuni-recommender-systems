
"""Final Part 2 evaluation on the held-out test set.

Run once after all model selection is complete:

    python -m part2.evaluate_test

All hyperparameters were selected using validation only.
Models are refitted on train + validation and evaluated on test.
Do not tune hyperparameters based on these results.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from run_all import (
    K,
    evaluate,
    index_data,
    item_knn,
    load_ratings,
    most_popular,
    temporal_split,
)
from part2.evaluate_validation import (
    evaluate_per_user,
    paired_bootstrap,
)
from part2.models import (
    content_based,
    matrix_factorization,
    user_knn,
)


DATA_DIR = Path("data")
OUT_DIR = Path("results/part2")
SEED = 123


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(DATA_DIR)

    movies = pd.read_csv(
        DATA_DIR / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["item", "title", "genres"],
        encoding="latin-1",
    )

    train, validation, test = temporal_split(
        ratings,
        0.05,
        0.05,
    )

    # Refit using all data available before the test period.
    fit = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    data = index_data(
        fit,
        movies["item"].to_numpy(),
    )

    # All configurations were selected on validation.
    model_factories = {
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

    summaries = []
    per_user_tables = []
    reference_users = None

    for name, builder in model_factories.items():
        print(f"Evaluating {name} on test...", flush=True)

        scorer = builder()

        per_user, coverage, cold_skipped = evaluate_per_user(
            scorer,
            data,
            test,
        )

        # Every model must use the same evaluation users.
        current_users = set(per_user["user"])

        if reference_users is None:
            reference_users = current_users
        elif current_users != reference_users:
            raise AssertionError(
                "Models have different test user sets"
            )

        # Cross-check against the original Part 1 evaluator.
        original = evaluate(scorer, data, test)

        summary = {
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
        }

        for metric in (
            f"precision@{K}",
            f"recall@{K}",
            f"ndcg@{K}",
            "coverage",
        ):
            if not np.isclose(
                summary[metric],
                original[metric],
                atol=1e-10,
            ):
                raise AssertionError(
                    f"Evaluation mismatch: {name}, {metric}"
                )

        summaries.append(summary)

        per_user.insert(0, "model", name)
        per_user_tables.append(per_user)

        print(
            f"  NDCG@{K}: {summary[f'ndcg@{K}']:.4f}, "
            f"Coverage: {coverage:.4f}, "
            f"Users: {len(per_user)}",
            flush=True,
        )

    summary_df = pd.DataFrame(summaries).sort_values(
        f"ndcg@{K}",
        ascending=False,
    )

    per_user_df = pd.concat(
        per_user_tables,
        ignore_index=True,
    )

    bootstrap_df = paired_bootstrap(
        per_user_df,
        reference="ItemKNN",
        samples=2000,
    )

    summary_df.to_csv(
        OUT_DIR / "test_summary.csv",
        index=False,
    )

    per_user_df.to_csv(
        OUT_DIR / "test_per_user.csv",
        index=False,
    )

    bootstrap_df.to_csv(
        OUT_DIR / "test_paired_bootstrap.csv",
        index=False,
    )

    print("\nFinal test summary:")
    print(summary_df.to_string(index=False))

    print("\nPaired bootstrap (model - ItemKNN):")
    print(bootstrap_df.to_string(index=False))

    print(f"\nResults saved in {OUT_DIR}")


if __name__ == "__main__":
    main()