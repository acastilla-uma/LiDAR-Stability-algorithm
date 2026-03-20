#!/usr/bin/env python3
"""Train a sprint-5 baseline model to predict w (omega) from featured data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Ensure package imports work when executed as a script from repo root.
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lidar_stability.ml.feature_engineering import build_w_training_dataset, load_featured_data


def train_rf_kfold(X, y, n_splits: int, random_state: int):
    """Train K-fold baseline and return metrics plus fitted final model."""
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=random_state + fold_idx,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        y_pred = model.predict(X.iloc[test_idx])
        y_true = y.iloc[test_idx]

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        fold_metrics.append({'fold': fold_idx, 'rmse': rmse, 'mae': mae, 'r2': r2})

    final_model = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    final_model.fit(X, y)

    rmse_values = [m['rmse'] for m in fold_metrics]
    mae_values = [m['mae'] for m in fold_metrics]
    r2_values = [m['r2'] for m in fold_metrics]

    summary = {
        'n_samples': int(len(X)),
        'n_features': int(X.shape[1]),
        'kfold': n_splits,
        'rmse_mean': float(np.mean(rmse_values)),
        'rmse_std': float(np.std(rmse_values)),
        'mae_mean': float(np.mean(mae_values)),
        'mae_std': float(np.std(mae_values)),
        'r2_mean': float(np.mean(r2_values)),
        'r2_std': float(np.std(r2_values)),
        'folds': fold_metrics,
    }

    return final_model, summary


def parse_args():
    parser = argparse.ArgumentParser(description='Train baseline RF model for w prediction')
    parser.add_argument(
        '--input-glob',
        default='Doback-Data/featured/DOBACK*.csv',
        help='Glob pattern for featured CSV files',
    )
    parser.add_argument(
        '--target-column',
        default=None,
        help='Optional explicit target column for w (omega). If omitted, gy/gz autodetection is used.',
    )
    parser.add_argument('--n-splits', type=int, default=5, help='Number of KFold splits')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument(
        '--output-model',
        default='output/results/w_model_rf.joblib',
        help='Path to save trained model artifact',
    )
    parser.add_argument(
        '--output-metrics',
        default='output/results/w_model_rf_metrics.json',
        help='Path to save training metrics JSON',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    csv_paths = sorted(repo_root.glob(args.input_glob))
    if not csv_paths:
        raise FileNotFoundError(f'No input files found with pattern: {args.input_glob}')

    df = load_featured_data(csv_paths)
    if df.empty:
        raise RuntimeError('Failed to load featured data')

    X, y, used_features, _ = build_w_training_dataset(df, target_column=args.target_column)

    model, summary = train_rf_kfold(
        X,
        y,
        n_splits=max(2, int(args.n_splits)),
        random_state=int(args.random_state),
    )

    model_path = (repo_root / args.output_model).resolve()
    metrics_path = (repo_root / args.output_metrics).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        'model': model,
        'feature_columns': used_features,
        'target_name': 'omega_rad_s',
    }
    dump(artifact, model_path)

    metrics_payload = {
        **summary,
        'feature_columns': used_features,
        'input_glob': args.input_glob,
        'target_column': args.target_column or 'auto',
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding='utf-8')

    print('Training completed')
    print(f"  samples={summary['n_samples']} features={summary['n_features']}")
    print(f"  RMSE(mean+-std)={summary['rmse_mean']:.6f} +- {summary['rmse_std']:.6f} rad/s")
    print(f"  MAE(mean+-std)={summary['mae_mean']:.6f} +- {summary['mae_std']:.6f} rad/s")
    print(f"  R2(mean+-std)={summary['r2_mean']:.4f} +- {summary['r2_std']:.4f}")
    print(f'  model: {model_path}')
    print(f'  metrics: {metrics_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

