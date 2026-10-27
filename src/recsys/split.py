"""Temporal train/validation/test split and optional low-activity filtering."""
import pandas as pd


def temporal_split(ratings: pd.DataFrame, val_frac: float, test_frac: float):
    """Return (train, val, test) cut by global timestamp. Document cut points in the report."""
    raise NotImplementedError
