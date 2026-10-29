"""Precision@K, Recall@K, NDCG@K and catalogue coverage (ranked against the full catalogue).

All per-user metrics use binary relevance: `relevant` is the set of items the user
liked in the test period, `ranked` is the recommendation list (best first).
Users without any relevant item must be skipped by the caller.
"""
import numpy as np

# https://towardsdatascience.com/mean-average-precision-at-k-map-k-clearly-explained-538d8e032d2/
# https://insidelearningmachines.com/precisionk_and_recallk/
def precision_at_k(ranked: np.ndarray, relevant: set, k: int = 10) -> float:
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / k

# https://insidelearningmachines.com/precisionk_and_recallk/
def recall_at_k(ranked: np.ndarray, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)

# https://www.evidentlyai.com/ranking-metrics/ndcg-metric
def ndcg_at_k(ranked: np.ndarray, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    # a hit at position i (0-based) gains 1 / log2(i + 2)
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    # best possible: all relevant items (at most k) at the top
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def catalogue_coverage(all_recommendations, n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's top-K list.

    `all_recommendations` is an iterable of recommendation lists (one per user).
    """
    recommended = set()
    for ranked in all_recommendations:
        recommended.update(int(item) for item in ranked)
    return len(recommended) / n_items
