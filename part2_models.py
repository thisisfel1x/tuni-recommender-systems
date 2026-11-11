
"""Additional recommendation models for Part 2."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def content_based(data, movies: pd.DataFrame, year_weight: float = 1.0):
    """Recommend movies based on genres and release years."""

    if year_weight < 0:
        raise ValueError("year_weight must be non-negative")

    # Match movie metadata to the full candidate catalogue.
    metadata = (
        movies.drop_duplicates(subset="item")
        .set_index("item")
        .reindex(data.item_ids)
    )

    # Represent genres as multi-hot encoded features.
    genre_lists = metadata["genres"].fillna("").str.split(
        "|", regex=False
    )
    genre_names = sorted({
        genre
        for genres in genre_lists
        for genre in genres
        if genre and genre != "(no genres listed)"
    })
    genre_to_col = {
        genre: col for col, genre in enumerate(genre_names)
    }

    genre_vectors = np.zeros(
        (len(data.item_ids), len(genre_names)),
        dtype=np.float32,
    )

    for row, genres in enumerate(genre_lists):
        for genre in genres:
            if genre in genre_to_col:
                genre_vectors[row, genre_to_col[genre]] = 1.0

    # Extract and normalize release years.
    years = pd.to_numeric(
        metadata["title"].str.extract(r"\((\d{4})\)\s*$")[0],
        errors="coerce",
    )

    if years.notna().any() and years.max() > years.min():
        normalized_year = (
            (years - years.min()) / (years.max() - years.min())
        ).fillna(0.5)
    else:
        normalized_year = pd.Series(0.5, index=years.index)

    recentness = normalized_year.to_numpy(dtype=np.float32)
    year_vectors = np.column_stack(
        (recentness, 1.0 - recentness)
    )

    # Combine and normalize item features.
    item_features = np.column_stack(
        (genre_vectors, year_weight * year_vectors)
    )
    item_norms = np.linalg.norm(
        item_features, axis=1, keepdims=True
    )
    item_features /= np.maximum(item_norms, 1e-12)

    # Build user profiles from positive training ratings only.
    liked = data.rating >= 4

    positive_history = csr_matrix(
        (
            np.ones(liked.sum(), dtype=np.float32),
            (data.u[liked], data.i[liked]),
        ),
        shape=data.shape,
    )

    user_profiles = positive_history @ item_features

    user_norms = np.linalg.norm(
        user_profiles, axis=1, keepdims=True
    )
    user_profiles /= np.maximum(user_norms, 1e-12)

    # Return a scorer compatible with Part 1 evaluation.
    return lambda rows: user_profiles[rows] @ item_features.T


def matrix_factorization(
    data,
    n_factors=16,
    n_epochs=8,
    learning_rate=0.005,
    reg=0.05,
    batch_size=4096,
    seed=123,
):
    """Biased matrix factorization trained on observed ratings."""

    if (
        n_factors < 1
        or n_epochs < 1
        or learning_rate <= 0
        or reg < 0
        or batch_size < 1
    ):
        raise ValueError("Invalid matrix factorization parameters")

    if len(data.rating) == 0:
        raise ValueError("Training data must contain ratings")

    rng = np.random.default_rng(seed)

    n_users, n_items = data.shape

    users = np.asarray(data.u, dtype=np.intp)
    items = np.asarray(data.i, dtype=np.intp)
    ratings = np.asarray(data.rating, dtype=np.float32)

    global_mean = float(ratings.mean())

    # Initialize latent factors.
    user_factors = rng.normal(
        0, 0.1, size=(n_users, n_factors)
    ).astype(np.float32)

    item_factors = rng.normal(
        0, 0.1, size=(n_items, n_factors)
    ).astype(np.float32)

    user_bias = np.zeros(n_users, dtype=np.float32)
    item_bias = np.zeros(n_items, dtype=np.float32)

    # Do not assign random preferences to unseen catalogue items.
    item_counts = np.bincount(items, minlength=n_items)
    item_factors[item_counts == 0] = 0.0

    # Train only on observed ratings.
    for epoch in range(n_epochs):
        order = rng.permutation(len(ratings))

        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]

            u = users[batch]
            i = items[batch]
            r = ratings[batch]

            # Snapshot factors before applying batch updates.
            pu = user_factors[u]
            qi = item_factors[i]

            predictions = (
                global_mean
                + user_bias[u]
                + item_bias[i]
                + np.sum(pu * qi, axis=1)
            )

            errors = r - predictions

            # Regularized gradients.
            grad_user_bias = learning_rate * (
                errors - reg * user_bias[u]
            )
            grad_item_bias = learning_rate * (
                errors - reg * item_bias[i]
            )
            grad_user = learning_rate * (
                errors[:, None] * qi - reg * pu
            )
            grad_item = learning_rate * (
                errors[:, None] * pu - reg * qi
            )

            # Accumulate updates for repeated user/item indices.
            np.add.at(user_bias, u, grad_user_bias)
            np.add.at(item_bias, i, grad_item_bias)
            np.add.at(user_factors, u, grad_user)
            np.add.at(item_factors, i, grad_item)

        if not (
            np.isfinite(user_factors).all()
            and np.isfinite(item_factors).all()
            and np.isfinite(user_bias).all()
            and np.isfinite(item_bias).all()
        ):
            raise FloatingPointError(
                "Matrix factorization training diverged"
            )

    # Compatible with the Part 1 evaluate() interface.
    def score(rows):
        return (
            global_mean
            + user_bias[rows, None]
            + item_bias[None, :]
            + user_factors[rows] @ item_factors.T
        )

    return score