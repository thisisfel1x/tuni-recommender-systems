"""Loading and parsing of MovieLens 1M (`::`-separated .dat files)."""
from pathlib import Path

import pandas as pd

ENCODING = "latin-1"

def _read(data_dir: Path, filename: str, names: list, dtype: dict) -> pd.DataFrame:
    path = Path(data_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"{filename} not found in {data_dir}")
    return pd.read_csv(path, sep="::", engine="python", header=None, names=names, dtype=dtype, encoding=ENCODING)


def load_ratings(data_dir: Path) -> pd.DataFrame:
    """Columns: user, item, rating, timestamp"""
    return _read(
        data_dir,
        "ratings.dat",
        ["user", "item", "rating", "timestamp"],
        {"user": "int32", "item": "int32", "rating": "int8", "timestamp": "int64"},
    )


def load_movies(data_dir: Path) -> pd.DataFrame:
    """Columns: item, title, year, genres"""
    # Keep in mind, that genres needs to be converted to a list first by parsing
    movies = _read(data_dir, "movies.dat", ["item", "title", "genres"], {"item": "int32"})
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).astype("int16")
    movies["genres"] = movies["genres"].str.split("|")
    return movies[["item", "title", "year", "genres"]]


def load_users(data_dir: Path) -> pd.DataFrame:
    """Columns: user, gender, age, occupation, zip"""
    return _read(
        data_dir,
        "users.dat",
        ["user", "gender", "age", "occupation", "zip"],
        {"user": "int32", "gender": "category", "age": "int8", "occupation": "int8", "zip": "string"},
    )
