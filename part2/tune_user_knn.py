
"""Hyperparameter tuning for user-based kNN on validation data."""

from pathlib import Path
from itertools import product

import pandas as pd

from run_all import (
    load_ratings,
    temporal_split,
    index_data,
    evaluate,
)
from part2.models import user_knn


def main():
    data_dir = Path("data")
    output = Path("results/part2/user_knn_tuning.csv")

    ratings = load_ratings(data_dir)

    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["item", "title", "genres"],
        encoding="latin-1",
    )

    train, val, _ = temporal_split(ratings, 0.05, 0.05)
    train_data = index_data(
        train,
        movies["item"].to_numpy(),
    )

    results = []

    for k, similarity in product(
        [20, 50, 100, 200],
        ["cosine", "jaccard"],
    ):
        model = user_knn(
            train_data,
            k=k,
            similarity=similarity,
        )

        metrics = evaluate(model, train_data, val)

        results.append({
            "k": k,
            "similarity": similarity,
            **metrics,
        })

        print(
            f"k={k:<4} "
            f"similarity={similarity:<8} "
            f"NDCG@10={metrics['ndcg@10']:.4f} "
            f"Coverage={metrics['coverage']:.4f}",
            flush=True,
        )

        # Save after each completed configuration.
        output.parent.mkdir(exist_ok=True)
        pd.DataFrame(results).to_csv(output, index=False)

    df = pd.DataFrame(results)
    best = df.loc[df["ndcg@10"].idxmax()]

    print("\nBest User-based kNN Configuration")
    print(best.to_string())


if __name__ == "__main__":
    main()