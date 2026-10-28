"""Dataset characterisation"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def summarize(ratings: pd.DataFrame) -> dict:
    """#users, #items, #interactions, density, time span, long tail."""
    user_count = ratings["user"].nunique()
    item_count = ratings["item"].nunique()
    interaction_count = len(ratings)
    density = interaction_count / (user_count * item_count)

    # timestamps are unix seconds -> convert to dates
    dates = pd.to_datetime(ratings["timestamp"], unit="s")

    per_user = ratings.groupby("user").size()
    per_item = ratings.groupby("item").size().sort_values(ascending=False)

    # share of all ratings that go to the most popular 1% / 10% / 20% of items
    top_share = {}
    for fraction in [0.01, 0.10, 0.20]:
        n_top = int(item_count * fraction)
        top_share[fraction] = per_item.iloc[:n_top].sum() / interaction_count

    return {
        "user_count": user_count,
        "item_count": item_count,
        "interaction_count": interaction_count,
        "density": density,
        "min_date": dates.min(),
        "max_date": dates.max(),
        "ratings_per_user_min": int(per_user.min()),
        "ratings_per_user_median": float(per_user.median()),
        "ratings_per_user_max": int(per_user.max()),
        "ratings_per_item_min": int(per_item.min()),
        "ratings_per_item_median": float(per_item.median()),
        "ratings_per_item_max": int(per_item.max()),
        "items_with_less_than_10_ratings": int((per_item < 10).sum()),
        "top_item_share": top_share,  # {0.01: share, 0.1: share, 0.2: share}
        "rating_share": ratings["rating"].value_counts(normalize=True).sort_index().to_dict(),
    }


def make_figures(ratings: pd.DataFrame, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_user = ratings.groupby("user").size().sort_values(ascending=False)
    per_item = ratings.groupby("item").size().sort_values(ascending=False)

    # 1) Long tail of items: rank (1 = most popular) vs. number of ratings
    plt.figure()
    plt.loglog(range(1, len(per_item) + 1), per_item.values)
    plt.xlabel("Item rank")
    plt.ylabel("Number of ratings")
    plt.title("Ratings per item (long tail)")
    plt.savefig(out_dir / "ratings_per_item.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Same for users
    plt.figure()
    plt.loglog(range(1, len(per_user) + 1), per_user.values)
    plt.xlabel("User rank")
    plt.ylabel("Number of ratings")
    plt.title("Ratings per user")
    plt.savefig(out_dir / "ratings_per_user.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3) Rating distribution
    plt.figure()
    ratings["rating"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Rating")
    plt.ylabel("Number of ratings")
    plt.title("Rating distribution")
    plt.savefig(out_dir / "rating_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4) Activity over time: ratings per month
    months = pd.to_datetime(ratings["timestamp"], unit="s").dt.to_period("M")
    plt.figure()
    months.value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Month")
    plt.ylabel("Number of ratings")
    plt.title("Activity over time")
    plt.savefig(out_dir / "activity_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()
