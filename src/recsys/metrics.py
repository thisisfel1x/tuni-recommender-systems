"""Precision@K, Recall@K, NDCG@K and catalogue coverage (ranked against the full catalogue)."""
import numpy as np


def precision_at_k(ranked: np.ndarray, relevant: set, k: int = 10) -> float:
    raise NotImplementedError


def recall_at_k(ranked: np.ndarray, relevant: set, k: int = 10) -> float:
    raise NotImplementedError


def ndcg_at_k(ranked: np.ndarray, relevant: set, k: int = 10) -> float:
    raise NotImplementedError


def catalogue_coverage(all_recommendations, n_items: int) -> float:
    raise NotImplementedError
