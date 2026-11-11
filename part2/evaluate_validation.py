"""Part 2: fair validation comparison and paired user bootstrap.

Run from the repository root:
    python -m part2.evaluate_validation

Only train/validation are used. Do not use test here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from run_all import (
    K,
    REL_THRESHOLD,
    catalogue_coverage,
    evaluate,
    index_data,
    item_knn,
    load_ratings,
    most_popular,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    temporal_split,
)
from part2.models import content_based, matrix_factorization, user_knn


DATA_DIR = Path("data")
OUT_DIR = Path("results/part2")
SEED = 123
BOOTSTRAP_SAMPLES = 2000
REFERENCE = "ItemKNN"


def evaluate_per_user(scorer, data, validation, batch_size=128):
    """Use the same eligible users, seen filter and ranking as run_all.evaluate."""
    relevant = (
        validation[validation["rating"] >= REL_THRESHOLD]
        .groupby("user")["item"]
        .agg(set)
    )
    all_users = relevant.index.to_numpy()
    is_known = np.isin(all_users, data.user_ids)
    users = all_users[is_known]
    rows = np.searchsorted(data.user_ids, users)
    cold_skipped = int((~is_known).sum())
    train_counts = np.bincount(data.u, minlength=data.shape[0])

    records = []
    all_ranked_lists = []
    for start in range(0, len(users), batch_size):
        batch_rows = rows[start : start + batch_size]
        scores = np.asarray(scorer(batch_rows), dtype=np.float64).copy()
        expected_shape = (len(batch_rows), data.shape[1])
        if scores.shape != expected_shape or not np.isfinite(scores).all():
            raise ValueError(f"Invalid score matrix: expected {expected_shape}, got {scores.shape}")
        scores[data.seen[batch_rows].toarray()] = -np.inf
        top_indices = np.argsort(-scores, axis=1, kind="stable")[:, :K]

        for user, row, top in zip(
            users[start : start + batch_size], batch_rows, top_indices
        ):
            ranked_items = data.item_ids[top]
            truth = relevant.loc[user]
            all_ranked_lists.append(ranked_items)
            records.append({
                "user": int(user),
                "train_history_size": int(train_counts[row]),
                "n_relevant_validation": len(truth),
                f"precision@{K}": precision_at_k(ranked_items, truth),
                f"recall@{K}": recall_at_k(ranked_items, truth),
                f"ndcg@{K}": ndcg_at_k(ranked_items, truth),
            })

    per_user = pd.DataFrame(records)
    coverage = catalogue_coverage(all_ranked_lists, len(data.item_ids))
    return per_user, coverage, cold_skipped


def paired_bootstrap(per_user_results, reference=REFERENCE, samples=BOOTSTRAP_SAMPLES):
    """95% percentile CI for paired mean per-user NDCG differences."""
    values = per_user_results.pivot(index="user", columns="model", values=f"ndcg@{K}")
    if values.isna().any().any():
        raise ValueError("Models were not evaluated on identical users")
    if reference not in values:
        raise ValueError(f"Reference model {reference!r} missing")

    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    reference_values = values[reference].to_numpy()
    comparisons = []
    for model in values.columns:
        if model == reference:
            continue
        differences = values[model].to_numpy() - reference_values
        sampled_means = differences[indices].mean(axis=1)
        low, high = np.percentile(sampled_means, [2.5, 97.5])
        comparisons.append({
            "model": model,
            "reference": reference,
            "mean_ndcg_difference": float(differences.mean()),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "n_users": len(values),
            "bootstrap_samples": samples,
        })
    return pd.DataFrame(comparisons)


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
    train, validation, _ = temporal_split(ratings, 0.05, 0.05)
    data = index_data(train, movies["item"].to_numpy())

    # Build one model at a time, so progress is visible and memory stays moderate.
    model_factories = {
        "MostPopular": lambda: most_popular(data, min_rating=4),
        "ItemKNN": lambda: item_knn(data, k=500, similarity="cosine"),
        "ContentBased": lambda: content_based(data, movies, year_weight=1.0),
        "MatrixFactorization": lambda: matrix_factorization(
            data,
            n_factors=32,
            n_epochs=24,
            learning_rate=0.005,
            reg=0.01,
            seed=SEED,
        ),
        "UserKNN": lambda: user_knn(data, k=200, similarity="cosine"),
    }

    summaries, all_per_user = [], []
    for name, build_model in model_factories.items():
        print(f"Evaluating {name}...", flush=True)
        scorer = build_model()
        user_df, coverage, cold_skipped = evaluate_per_user(scorer, data, validation)
        user_df.insert(0, "model", name)
        all_per_user.append(user_df)
        summary = {
            "model": name,
            f"precision@{K}": user_df[f"precision@{K}"].mean(),
            f"recall@{K}": user_df[f"recall@{K}"].mean(),
            f"ndcg@{K}": user_df[f"ndcg@{K}"].mean(),
            "coverage": coverage,
            "n_users": len(user_df),
            "n_cold_users_skipped": cold_skipped,
        }
        summaries.append(summary)
        print(f"{name}: NDCG@{K}={summary[f'ndcg@{K}']:.4f}, "
              f"coverage={coverage:.4f}, users={len(user_df)}", flush=True)

        # Validate that our per-user aggregation matches Part 1 exactly.
        check = evaluate(scorer, data, validation)
        for metric in (f"precision@{K}", f"recall@{K}", f"ndcg@{K}", "coverage"):
            if not np.isclose(summary[metric], check[metric], atol=1e-10):
                raise AssertionError(f"{name} {metric}: new={summary[metric]}, old={check[metric]}")

    summary_df = pd.DataFrame(summaries).sort_values(f"ndcg@{K}", ascending=False)
    per_user_df = pd.concat(all_per_user, ignore_index=True)
    bootstrap_df = paired_bootstrap(per_user_df)

    summary_df.to_csv(OUT_DIR / "validation_summary.csv", index=False)
    per_user_df.to_csv(OUT_DIR / "validation_per_user.csv", index=False)
    bootstrap_df.to_csv(OUT_DIR / "validation_paired_bootstrap.csv", index=False)

    print("\nValidation summary:\n", summary_df.to_string(index=False))
    print("\nPaired bootstrap (model - ItemKNN):\n", bootstrap_df.to_string(index=False))
    print(f"\nSaved files in {OUT_DIR}")


if __name__ == "__main__":
    main()