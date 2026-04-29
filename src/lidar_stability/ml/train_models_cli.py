#!/usr/bin/env python3
"""Train and save one or multiple w-prediction models from featured data."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lidar_stability.ml.feature_engineering import build_w_training_dataset


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


MODEL_HPARAM_KEYS = {
    "rf": {
        "rf_n_estimators",
        "rf_min_samples_leaf",
        "rf_max_depth",
        "rf_max_features",
        "rf_min_samples_split",
    },
    "extra_trees": {
        "extra_trees_n_estimators",
        "extra_trees_min_samples_leaf",
        "extra_trees_max_depth",
        "extra_trees_max_features",
        "extra_trees_min_samples_split",
    },
    "gbr": {
        "gbr_n_estimators",
        "gbr_learning_rate",
        "gbr_max_depth",
        "gbr_subsample",
        "gbr_min_samples_leaf",
        "gbr_min_samples_split",
    },
}

SOURCE_FILE_COLUMN = "__source_file"


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
        default="gy",
        help="Explicit target column (must be 'gy').",
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
        "--cv-group-by",
        choices=["row", "source_file"],
        default="row",
        help="How to group rows when building CV folds.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf"],
        choices=["rf", "extra_trees", "gbr"],
        help="Models to train",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="K-Fold splits")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for tree-based models that support n_jobs (-1 uses all available cores)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="output/models",
        help="Directory where model artifacts and metrics will be saved",
    )
    parser.add_argument(
        "--prefix",
        default="w_model",
        help="File prefix for artifacts",
    )
    parser.add_argument(
        "--compact-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, stores all trained models in one compressed bundle and writes a single consolidated "
            "metrics JSON file (fewer output files)."
        ),
    )
    parser.add_argument(
        "--artifact-compress",
        type=int,
        default=3,
        help="Compression level for joblib artifacts (0-9). Higher is smaller but slower.",
    )
    parser.add_argument(
        "--run-config",
        action="append",
        default=None,
        help=(
            "Repeatable JSON object (or Python dict literal) defining one training run. "
            "Format: '{\"model\":\"rf\",\"run_id\":\"rf_400\",\"rf_n_estimators\":400}'. "
            "If provided, --models is ignored and all run configs are trained."
        ),
    )

    # RandomForest hyperparameters
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=400,
        help="RandomForest: number of trees",
    )
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=2,
        help="RandomForest: minimum samples required at leaf node",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="RandomForest: maximum tree depth (None=unlimited)",
    )
    parser.add_argument(
        "--rf-max-features",
        type=str,
        default="sqrt",
        help="RandomForest: max features at each split (sqrt, log2, or int)",
    )
    parser.add_argument(
        "--rf-min-samples-split",
        type=int,
        default=2,
        help="RandomForest: minimum samples required to split an internal node",
    )

    # ExtraTrees hyperparameters
    parser.add_argument(
        "--extra-trees-n-estimators",
        type=int,
        default=500,
        help="ExtraTrees: number of trees",
    )
    parser.add_argument(
        "--extra-trees-min-samples-leaf",
        type=int,
        default=2,
        help="ExtraTrees: minimum samples required at leaf node",
    )
    parser.add_argument(
        "--extra-trees-max-depth",
        type=int,
        default=None,
        help="ExtraTrees: maximum tree depth (None=unlimited)",
    )
    parser.add_argument(
        "--extra-trees-max-features",
        type=str,
        default="sqrt",
        help="ExtraTrees: max features at each split (sqrt, log2, or int)",
    )
    parser.add_argument(
        "--extra-trees-min-samples-split",
        type=int,
        default=2,
        help="ExtraTrees: minimum samples required to split an internal node",
    )

    # GradientBoosting hyperparameters
    parser.add_argument(
        "--gbr-n-estimators",
        type=int,
        default=300,
        help="GradientBoosting: number of boosting stages",
    )
    parser.add_argument(
        "--gbr-learning-rate",
        type=float,
        default=0.05,
        help="GradientBoosting: learning rate (shrinkage)",
    )
    parser.add_argument(
        "--gbr-max-depth",
        type=int,
        default=3,
        help="GradientBoosting: maximum tree depth",
    )
    parser.add_argument(
        "--gbr-subsample",
        type=float,
        default=0.85,
        help="GradientBoosting: fraction of samples used for fitting trees (0.0, 1.0]",
    )
    parser.add_argument(
        "--gbr-min-samples-leaf",
        type=int,
        default=1,
        help="GradientBoosting: minimum samples required at leaf node",
    )
    parser.add_argument(
        "--gbr-min-samples-split",
        type=int,
        default=2,
        help="GradientBoosting: minimum samples required to split internal node",
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


def load_featured_with_source(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        frame = pd.read_csv(p, low_memory=False)
        frame = frame.copy()
        frame[SOURCE_FILE_COLUMN] = p.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _parse_max_features(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().lower()
    if text in {"none", "null"}:
        return None
    if text in {"sqrt", "log2", "auto"}:
        return text
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return str(value)


def _collect_base_hyperparams(args: argparse.Namespace) -> dict:
    hyperparams = {
        key: value
        for key, value in vars(args).items()
        if key.startswith(("rf_", "extra_trees_", "gbr_"))
    }
    hyperparams["rf_max_features"] = _parse_max_features(hyperparams.get("rf_max_features"))
    hyperparams["extra_trees_max_features"] = _parse_max_features(hyperparams.get("extra_trees_max_features"))
    return hyperparams


def _sanitize_run_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    return safe.strip("_")


def _effective_model_hyperparams(model_key: str, merged_hyperparams: dict) -> dict:
    keys = MODEL_HPARAM_KEYS[model_key]
    return {k: merged_hyperparams[k] for k in keys if k in merged_hyperparams}


def build_training_runs(args: argparse.Namespace, base_hyperparams: dict) -> list[dict]:
    runs: list[dict] = []

    if args.run_config:
        for idx, raw in enumerate(args.run_config, start=1):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                try:
                    payload = ast.literal_eval(raw)
                except (ValueError, SyntaxError) as exc_eval:
                    raise ValueError(
                        f"Invalid JSON/object in --run-config #{idx}: {exc}"
                    ) from exc_eval

            if not isinstance(payload, dict):
                raise ValueError(f"--run-config #{idx} must be a JSON object")

            model_key = payload.get("model")
            if model_key not in MODEL_HPARAM_KEYS:
                raise ValueError(
                    f"--run-config #{idx} must include 'model' in {sorted(MODEL_HPARAM_KEYS.keys())}"
                )

            run_id_raw = payload.get("run_id", f"run{idx:02d}")
            run_id = _sanitize_run_id(run_id_raw)
            if not run_id:
                raise ValueError(f"Invalid empty run_id in --run-config #{idx}")

            overrides = {k: v for k, v in payload.items() if k not in {"model", "run_id"}}
            invalid_keys = sorted(k for k in overrides if k not in MODEL_HPARAM_KEYS[model_key])
            if invalid_keys:
                raise ValueError(
                    f"--run-config #{idx} has invalid keys for model '{model_key}': {invalid_keys}"
                )

            merged_hyperparams = dict(base_hyperparams)
            if "rf_max_features" in overrides:
                overrides["rf_max_features"] = _parse_max_features(overrides["rf_max_features"])
            if "extra_trees_max_features" in overrides:
                overrides["extra_trees_max_features"] = _parse_max_features(overrides["extra_trees_max_features"])
            merged_hyperparams.update(overrides)

            runs.append(
                {
                    "model_key": model_key,
                    "run_id": run_id,
                    "hyperparams": merged_hyperparams,
                }
            )
    else:
        for model_key in args.models:
            runs.append(
                {
                    "model_key": model_key,
                    "run_id": "default",
                    "hyperparams": dict(base_hyperparams),
                }
            )

    seen_labels = set()
    for run in runs:
        label = (run["model_key"], run["run_id"])
        if label in seen_labels:
            raise ValueError(
                "Duplicate run detected for model/run_id pair: "
                f"{run['model_key']}/{run['run_id']}. Use unique run_id values."
            )
        seen_labels.add(label)

    return runs


def build_model(model_key: str, random_state: int, hyperparams: dict = None, n_jobs: int = -1):
    """Build a model with specified hyperparameters.
    
    Args:
        model_key: 'rf', 'extra_trees', or 'gbr'
        random_state: Random seed for reproducibility
        hyperparams: Dict with model-specific hyperparameters (e.g., {'rf_n_estimators': 400})
                     If None, uses default values.
    """
    if hyperparams is None:
        hyperparams = {}

    if model_key == "rf":
        return RandomForestRegressor(
            n_estimators=hyperparams.get("rf_n_estimators", 400),
            random_state=random_state,
            n_jobs=int(n_jobs),
            min_samples_leaf=hyperparams.get("rf_min_samples_leaf", 2),
            min_samples_split=hyperparams.get("rf_min_samples_split", 2),
            max_depth=hyperparams.get("rf_max_depth", None),
            max_features=hyperparams.get("rf_max_features", "sqrt"),
        )
    if model_key == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=hyperparams.get("extra_trees_n_estimators", 500),
            random_state=random_state,
            n_jobs=int(n_jobs),
            min_samples_leaf=hyperparams.get("extra_trees_min_samples_leaf", 2),
            min_samples_split=hyperparams.get("extra_trees_min_samples_split", 2),
            max_depth=hyperparams.get("extra_trees_max_depth", None),
            max_features=hyperparams.get("extra_trees_max_features", "sqrt"),
        )
    if model_key == "gbr":
        return GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=hyperparams.get("gbr_n_estimators", 300),
            learning_rate=hyperparams.get("gbr_learning_rate", 0.05),
            max_depth=hyperparams.get("gbr_max_depth", 3),
            subsample=hyperparams.get("gbr_subsample", 0.85),
            min_samples_leaf=hyperparams.get("gbr_min_samples_leaf", 1),
            min_samples_split=hyperparams.get("gbr_min_samples_split", 2),
        )
    raise ValueError(f"Unknown model key: {model_key}")


def resolve_cv_groups(
    df: pd.DataFrame,
    cv_group_by: str,
    source_col: str = SOURCE_FILE_COLUMN,
) -> pd.Series | None:
    if cv_group_by == "row":
        return None
    if source_col not in df.columns:
        raise ValueError(
            f"Requested --cv-group-by {cv_group_by}, but source column '{source_col}' is missing"
        )
    return df[source_col].astype(str)


def build_cv_splits(
    X: pd.DataFrame,
    n_splits: int,
    random_state: int,
    groups: pd.Series | None = None,
    cv_group_by: str = "row",
) -> list[tuple[np.ndarray, np.ndarray]]:
    if groups is not None:
        n_groups = int(groups.nunique())
        if n_groups < n_splits:
            raise ValueError(
                f"--cv-group-by {cv_group_by} requires at least {n_splits} unique groups, "
                f"but only {n_groups} were found"
            )
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(X, groups=groups))

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(X))


def _to_contiguous_feature_matrix(X: pd.DataFrame) -> np.ndarray:
    return np.ascontiguousarray(X.to_numpy(dtype=float, copy=False))


def _to_contiguous_target_array(y: pd.Series) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(y, dtype=float))


def train_kfold(
    model_key: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    random_state: int,
    hyperparams: dict = None,
    n_jobs: int = -1,
    split_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
):
    """Train model with K-Fold cross-validation.
    
    Args:
        model_key: 'rf', 'extra_trees', or 'gbr'
        X: Features dataframe
        y: Target series
        n_splits: Number of K-Fold splits
        random_state: Random seed
        hyperparams: Dict with model-specific hyperparameters
    """
    if hyperparams is None:
        hyperparams = {}
    if split_indices is None:
        split_indices = build_cv_splits(X, n_splits=n_splits, random_state=random_state)

    X_values = _to_contiguous_feature_matrix(X)
    y_values = _to_contiguous_target_array(y)

    fold_metrics: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(split_indices, start=1):
        model = build_model(model_key, random_state + fold_idx, hyperparams, n_jobs=n_jobs)
        model.fit(X_values[train_idx], y_values[train_idx])

        y_pred = model.predict(X_values[test_idx])
        y_true = y_values[test_idx]

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        fold_metrics.append({"fold": fold_idx, "rmse": rmse, "mae": mae, "r2": r2})

    final_model = build_model(model_key, random_state, hyperparams, n_jobs=n_jobs)
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

    df = load_featured_with_source(csv_paths)
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
    clean_df = df.loc[X.index].copy()
    cv_groups = resolve_cv_groups(clean_df, args.cv_group_by)

    selected_files_rel = [str(p.relative_to(repo_root)) for p in csv_paths]
    training_context = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "command": " ".join(sys.argv),
        "selected_files": selected_files_rel,
        "selected_files_count": len(selected_files_rel),
        "rows_after_filters": int(len(df)),
        "resolved_target_name": "gy",
        "resolved_feature_columns": list(used_features),
        "cv_group_by": args.cv_group_by,
    }

    n_splits = max(2, int(args.n_splits))
    split_indices = build_cv_splits(
        X,
        n_splits=n_splits,
        random_state=int(args.random_state),
        groups=cv_groups,
        cv_group_by=args.cv_group_by,
    )
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_hyperparams = _collect_base_hyperparams(args)
    training_runs = build_training_runs(args, base_hyperparams)

    compact_output = bool(args.compact_output)
    compression = int(args.artifact_compress)
    if compression < 0 or compression > 9:
        raise ValueError("--artifact-compress must be between 0 and 9")

    bundle_path = out_dir / f"{args.prefix}_models.joblib"
    bundle_metrics_path = out_dir / f"{args.prefix}_metrics.json"
    bundled_artifacts: dict[str, dict] = {}
    consolidated_metrics: list[dict] = []

    leaderboard = []
    for run in training_runs:
        model_key = run["model_key"]
        run_id = run["run_id"]
        hyperparams = run["hyperparams"]

        model, result = train_kfold(
            model_key=model_key,
            X=X,
            y=y,
            n_splits=n_splits,
            random_state=int(args.random_state),
            hyperparams=hyperparams,
            n_jobs=int(args.n_jobs),
            split_indices=split_indices,
        )

        if run_id == "default" and not args.run_config:
            artifact_key = model_key
        else:
            artifact_key = f"{model_key}@{run_id}"

        if compact_output:
            model_path = f"{bundle_path}::{artifact_key}"
            metrics_path = str(bundle_metrics_path)
        elif run_id == "default" and not args.run_config:
            model_path = out_dir / f"{args.prefix}_{model_key}.joblib"
            metrics_path = out_dir / f"{args.prefix}_{model_key}_metrics.json"
        else:
            model_path = out_dir / f"{args.prefix}_{model_key}_{run_id}.joblib"
            metrics_path = out_dir / f"{args.prefix}_{model_key}_{run_id}_metrics.json"

        effective_hyperparams = _effective_model_hyperparams(model_key, hyperparams)
        metrics_summary = {
            "rmse_mean": result.rmse_mean,
            "rmse_std": result.rmse_std,
            "mae_mean": result.mae_mean,
            "mae_std": result.mae_std,
            "r2_mean": result.r2_mean,
            "r2_std": result.r2_std,
        }

        artifact = {
            "model": model,
            "feature_columns": used_features,
            "target_name": "gy",
            "model_key": model_key,
            "run_id": run_id,
            "params": effective_hyperparams,
            "n_samples": result.n_samples,
            "n_features": result.n_features,
            "metrics": metrics_summary,
            "folds": result.folds,
            "training_context": training_context,
        }
        if compact_output:
            bundled_artifacts[artifact_key] = artifact
        else:
            dump(artifact, model_path, compress=compression)

        payload = {
            "model": model_key,
            "run_id": run_id,
            "input_glob": args.input_glob,
            "input_files": args.input_files,
            "contains": args.contains,
            "max_files": args.max_files,
            "shuffle_files": args.shuffle_files,
            "query": args.query,
            "cv_group_by": args.cv_group_by,
            "target_column": args.target_column,
            "target_name": "gy",
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
            "params": effective_hyperparams,
            "model_path": str(model_path),
            "artifact_key": artifact_key,
            "training_context": training_context,
            "run_config": {
                "model": model_key,
                "run_id": run_id,
                "hyperparameters": effective_hyperparams,
            },
        }
        if compact_output:
            consolidated_metrics.append(payload)
        else:
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        leaderboard.append(
            {
                "model": model_key,
                "run_id": run_id,
                "label": f"{model_key}@{run_id}" if run_id != "default" else model_key,
                "rmse_mean": result.rmse_mean,
                "mae_mean": result.mae_mean,
                "r2_mean": result.r2_mean,
                "n_samples": result.n_samples,
                "n_features": result.n_features,
                "params": effective_hyperparams,
                "feature_columns": list(used_features),
                "target_name": "gy",
                "cv_group_by": args.cv_group_by,
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "training_context": training_context,
            }
        )

        print(f"\n[{model_key} | run_id={run_id}] training completed")
        print(f"  samples={result.n_samples} features={result.n_features}")
        print(f"  RMSE={result.rmse_mean:.6f} +- {result.rmse_std:.6f}")
        print(f"  MAE={result.mae_mean:.6f} +- {result.mae_std:.6f}")
        print(f"  R2={result.r2_mean:.4f} +- {result.r2_std:.4f}")
        print(f"  model: {model_path}")
        print(f"  metrics: {metrics_path}")

    if compact_output:
        bundle_payload = {
            "format": "bundle_v2",
            "prefix": args.prefix,
            "n_models": len(bundled_artifacts),
            "training_context": training_context,
            "artifacts": bundled_artifacts,
        }
        dump(bundle_payload, bundle_path, compress=compression)

        compact_metrics_payload = {
            "format": "compact_metrics_v2",
            "prefix": args.prefix,
            "n_runs": len(consolidated_metrics),
            "training_context": training_context,
            "runs": consolidated_metrics,
        }
        bundle_metrics_path.write_text(json.dumps(compact_metrics_payload), encoding="utf-8")

        print("\nCompact output enabled")
        print(f"  model bundle: {bundle_path}")
        print(f"  consolidated metrics: {bundle_metrics_path}")

    leaderboard_path = out_dir / f"{args.prefix}_leaderboard.json"
    leaderboard_path.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    print(f"\nLeaderboard saved: {leaderboard_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
