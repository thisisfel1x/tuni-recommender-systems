
"""Reproduce Content-Based and Matrix Factorization tuning."""

from itertools import product
from pathlib import Path

import pandas as pd

from run_all import (
    load_ratings,
    temporal_split,
    index_data,
    evaluate,
)
from part2.models import content_based, matrix_factorization


def main():
    data_dir = Path("data")
    out_dir = Path("results/part2")
    out_dir.mkdir(exist_ok=True)

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
    data = index_data(train, movies["item"].to_numpy())

    # 1. Content-Based tuning
    content_results = []
    content_path = out_dir / "content_based_tuning.csv"

    for weight in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        scorer = content_based(
            data, movies, year_weight=weight
        )
        metrics = evaluate(scorer, data, val)

        content_results.append({
            "year_weight": weight,
            **metrics,
        })

        pd.DataFrame(content_results).to_csv(
            content_path, index=False
        )

        print(
            f"Content-Based weight={weight}: "
            f"NDCG@10={metrics['ndcg@10']:.4f}",
            flush=True,
        )

    # 2. Matrix Factorization tuning
    configurations = list(product(
        [16, 32],
        [8, 16],
        [0.02, 0.1],
    ))

    configurations += [
        (32, 24, 0.02),
        (32, 32, 0.02),
        (32, 24, 0.01),
        (32, 32, 0.01),
    ]

    mf_results = []
    mf_path = out_dir / "matrix_factorization_tuning.csv"

    for n_factors, n_epochs, reg in configurations:
        scorer = matrix_factorization(
            data,
            n_factors=n_factors,
            n_epochs=n_epochs,
            learning_rate=0.005,
            reg=reg,
            seed=123,
        )

        metrics = evaluate(scorer, data, val)

        mf_results.append({
            "n_factors": n_factors,
            "n_epochs": n_epochs,
            "reg": reg,
            "learning_rate": 0.005,
            **metrics,
        })

        pd.DataFrame(mf_results).to_csv(
            mf_path, index=False
        )

        print(
            f"MF factors={n_factors} "
            f"epochs={n_epochs} reg={reg}: "
            f"NDCG@10={metrics['ndcg@10']:.4f}",
            flush=True,
        )

    # 3. Show best configurations
    for name, rows in [
        ("Content-Based", content_results),
        ("Matrix Factorization", mf_results),
    ]:
        df = pd.DataFrame(rows)
        best = df.loc[df["ndcg@10"].idxmax()]
        print(f"\nBest {name}:")
        print(best.to_string())


if __name__ == "__main__":
    main()