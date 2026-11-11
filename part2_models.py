
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