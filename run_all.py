"""Single entry point that reproduces every number and figure.

Usage: python run_all.py --data-dir /path/to/ml-1m
"""
import argparse
import random
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from recsys import SEED  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing ratings.dat, movies.dat, users.dat")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--fig-dir", type=Path, default=Path("figures"))
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("pipeline not implemented yet")


if __name__ == "__main__":
    main()
