#!/usr/bin/env python3
"""Train and save one or multiple w-prediction models from featured data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lidar_stability.ml.feature_engineering import build_w_training_dataset, load_featured_data


@dataclass
class TrainResult:
    model_name: str
    n_samples: int
    n_features: int
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    r2_mean: float
    r2_std: float
    folds: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and persist one or multiple models for w (omega) prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-glob",
        nargs="+",
        default=["Doback-Data/featured/DOBACK*.csv"],
        help="One or more glob patterns relative to repo root",
    )
    parser.add_argument(
        "--input-files",
        nargs="*",
        default=None,
        help="Optional explicit CSV paths relative to repo root",
    )
    parser.add_argument(
        "--contains",
        nargs="*",
        default=None,
        help="Optional substrings to filter selected file names (all must match)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If > 0, limit number of input files after filtering",
    )
    parser.add_argument(
        "--shuffle-files",
        action="store_true",
        help="Shuffle file order before applying --max-files",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Optional explicit target column. If omitted, auto-detection is used.",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="*",
        default=None,
        help="Optional explicit feature columns. If omitted, defaults from feature_engineering are used.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional pandas query string to filter rows before training.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf"],
        choices=["rf", "extra_trees", "gbr"],
        help="Models to train",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="K-Fold splits")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="src/lidar_stability/ml/models",
        help="Directory where model artifacts and metrics will be saved",
    )
    parser.add_argument(
        "--prefix",
        default="w_model",
        help="File prefix for artifacts",
    )

    return parser.parse_args()


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _iter_paths_from_globs(repo_root: Path, patterns: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(repo_root.glob(pattern)))
    return found


def resolve_input_files(args: argparse.Namespace, repo_root: Path) -> list[Path]:
    paths = _iter_paths_from_globs(repo_root, args.input_glob)

    if args.input_files:
        for rel in args.input_files:
            candidate = (repo_root / rel).resolve()
            if candidate.exists() and candidate.suffix.lower() == ".csv":
                paths.append(candidate)

    unique = sorted({p.resolve() for p in paths if p.exists() and p.suffix.lower() == ".csv"})

    if args.contains:
        needles = [n.lower() for n in args.contains]
        unique = [p for p in unique if all(n in p.name.lower() for n in needles)]

    if args.shuffle_files and unique:
        rng = np.random.default_rng(args.random_state)
        rng.shuffle(unique)

    if args.max_files > 0:
        unique = unique[: args.max_files]

    if not unique:
        raise FileNotFoundError("No input CSV files found after applying filters")

    return unique


def build_model(model_key: str, random_state: int):
    if model_key == "rf":
        return RandomForestRegressor(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        )
    if model_key == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        )
    if model_key == "gbr":
        return GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
        )
    raise ValueError(f"Unknown model key: {model_key}")


def train_kfold(model_key: str, X: pd.DataFrame, y: pd.Series, n_splits: int, random_state: int):
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = build_model(model_key, random_state + fold_idx)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        y_pred = model.predict(X.iloc[test_idx])
        y_true = y.iloc[test_idx]

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        fold_metrics.append({"fold": fold_idx, "rmse": rmse, "mae": mae, "r2": r2})

    final_model = build_model(model_key, random_state)
    final_model.fit(X, y)

    rmse_values = [m["rmse"] for m in fold_metrics]
    mae_values = [m["mae"] for m in fold_metrics]
    r2_values = [m["r2"] for m in fold_metrics]

    result = TrainResult(
        model_name=model_key,
        n_samples=int(len(X)),
        n_features=int(X.shape[1]),
        rmse_mean=float(np.mean(rmse_values)),
        rmse_std=float(np.std(rmse_values)),
        mae_mean=float(np.mean(mae_values)),
        mae_std=float(np.std(mae_values)),
        r2_mean=float(np.mean(r2_values)),
        r2_std=float(np.std(r2_values)),
        folds=fold_metrics,
    )

    return final_model, result


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root()

    csv_paths = resolve_input_files(args, repo_root)
    print(f"Selected files: {len(csv_paths)}")
    for p in csv_paths[:10]:
        print(f"  - {p.relative_to(repo_root)}")
    if len(csv_paths) > 10:
        print("  - ...")

    df = load_featured_data(csv_paths)
    if df.empty:
        raise RuntimeError("Loaded dataframe is empty")

    if args.query:
        before = len(df)
        df = df.query(args.query).copy()
        print(f"Rows after query filter: {len(df)} (before={before})")
        if df.empty:
            raise RuntimeError("No rows left after applying --query filter")

    X, y, used_features, _ = build_w_training_dataset(
        df,
        feature_columns=args.feature_columns,
        target_column=args.target_column,
    )

    n_splits = max(2, int(args.n_splits))
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = []
    for model_key in args.models:
        model, result = train_kfold(
            model_key=model_key,
            X=X,
            y=y,
            n_splits=n_splits,
            random_state=int(args.random_state),
        )

        model_path = out_dir / f"{args.prefix}_{model_key}.joblib"
        metrics_path = out_dir / f"{args.prefix}_{model_key}_metrics.json"

        artifact = {
            "model": model,
            "feature_columns": used_features,
            "target_name": "omega_rad_s",
            "model_key": model_key,
        }
        dump(artifact, model_path)

        payload = {
            "model": model_key,
            "input_glob": args.input_glob,
            "input_files": args.input_files,
            "contains": args.contains,
            "max_files": args.max_files,
            "shuffle_files": args.shuffle_files,
            "query": args.query,
            "target_column": args.target_column or "auto",
            "feature_columns": used_features,
            "kfold": n_splits,
            "n_samples": result.n_samples,
            "n_features": result.n_features,
            "rmse_mean": result.rmse_mean,
            "rmse_std": result.rmse_std,
            "mae_mean": result.mae_mean,
            "mae_std": result.mae_std,
            "r2_mean": result.r2_mean,
            "r2_std": result.r2_std,
            "folds": result.folds,
            "model_path": str(model_path),
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        leaderboard.append(
            {
                "model": model_key,
                "rmse_mean": result.rmse_mean,
                "mae_mean": result.mae_mean,
                "r2_mean": result.r2_mean,
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
            }
        )

        print(f"\n[{model_key}] training completed")
        print(f"  samples={result.n_samples} features={result.n_features}")
        print(f"  RMSE={result.rmse_mean:.6f} +- {result.rmse_std:.6f}")
        print(f"  MAE={result.mae_mean:.6f} +- {result.mae_std:.6f}")
        print(f"  R2={result.r2_mean:.4f} +- {result.r2_std:.4f}")
        print(f"  model: {model_path}")
        print(f"  metrics: {metrics_path}")

    leaderboard_path = out_dir / f"{args.prefix}_leaderboard.json"
    leaderboard_path.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    print(f"\nLeaderboard saved: {leaderboard_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

