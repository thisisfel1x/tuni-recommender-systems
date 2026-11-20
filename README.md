# Recommender Systems Course Project, Group 15

Charleen Lang, Felix Weber, Vinicius Martins

## Setup

Python 3.11, dependencies pinned in `requirements.txt`

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download MovieLens 1M from https://grouplens.org/datasets/movielens/1m/ and unzip it.
The data directory must contain `ratings.dat` and `movies.dat`

## Run
This command reproduces the Part 1 results and figures.

```bash
python -m part1.run_all --data-dir /path/to/ml-1m --out-dir results/part1
```

## Part 1

- `results/part1/results.csv`: final test results, same columns as the course results template
- `results/part1/tuning_val.csv`: all validation runs used for tuning
- `results/part1/split_report.csv`, `results/part1/data_summary.json`: split and dataset statistics
- `figures/part1/`: plots used in the report

The seed is fixed to 123 (`SEED` in `part1/run_all.py`, can be changed with `--seed`).
Tests: `python -m pytest`

## Part 2

Part 2 adds three recommendation models: Content-Based Filtering, Matrix Factorization, and User-based kNN. Their implementations are in `part2/models.py`.

For the following commands, place `ratings.dat` and `movies.dat` in the `data/` directory.

Run the experiments from the repository root:

```bash
python -m part2.tune_content_mf
python -m part2.tune_user_knn
python -m part2.evaluate_validation
python -m part2.analyze_groups
python -m part2.analyze_item_popularity
python -m part2.sparsity_ablation
python -m part2.evaluate_test
```

Hyperparameters are selected using validation data only. The final models are retrained on train + validation and evaluated on the held-out test set.

Part 2 uses the same temporal split, evaluation metrics, and random seed (123) as Part 1. Statistical comparisons use paired bootstrap confidence intervals, and the additional analyses cover user activity, item popularity, and training-data sparsity (100%, 50%, and 20%).

Results are saved in `results/part2/`, including tuning results, per-user evaluations, bootstrap comparisons, sparsity experiments, and final test results.

Tests: `python -m pytest -v`