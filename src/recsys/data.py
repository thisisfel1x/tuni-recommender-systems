"""Loading and parsing of MovieLens dataset"""
from pathlib import Path

import pandas as pd


def load_ratings(data_dir: Path) -> pd.DataFrame:
    raise NotImplementedError


def load_movies(data_dir: Path) -> pd.DataFrame:
    raise NotImplementedError


def load_users(data_dir: Path) -> pd.DataFrame:
    raise NotImplementedError
