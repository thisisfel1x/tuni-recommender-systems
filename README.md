# Recommender Systems 

## 1. Requirements
- Python 3.11 (see `.python-version`)
- Dependencies are pinned in `requirements.txt`

```bash
uv venv --python 3.11 .venv # or: python3.11 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt # or: pip install -r requirements.txt
```

## 2. Data
Download MovieLens 1M from <https://grouplens.org/datasets/movielens/1m/>
and unzip it. The directory passed to `--data-dir` must contain `ratings.dat` (Part 1 only uses the ratings)

## 3. Run
```bash
python run_all.py --data-dir /path/to/ml-1m
```