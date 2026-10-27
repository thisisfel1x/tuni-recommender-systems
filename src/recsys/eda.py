"""Dataset characterisation"""
import pandas as pd


def summarize(ratings: pd.DataFrame) -> dict:
    """#users, #items, #interactions, density, time span."""
    raise NotImplementedError


def make_figures(ratings: pd.DataFrame, out_dir) -> None:
    """Interactions per user/item (log-log long tail), rating histogram, activity over time."""
    raise NotImplementedError
