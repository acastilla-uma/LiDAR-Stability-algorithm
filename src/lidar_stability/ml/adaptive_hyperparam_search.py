#!/usr/bin/env python3
"""Adaptive hyperparameter search until reaching a target R2.

This script preserves the existing workflow around featured-data loading,
balancing, grouped holdout selection, and artifact export, but replaces random
hyperparameter sampling with Optuna-based sequential optimization.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, KFold

try:
    import optuna
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    optuna = None

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lidar_stability.ml.feature_engineering import DEFAULT_FEATURE_COLUMNS, build_w_training_dataset


@dataclass
class TrialResult:
    trial: int
    optuna_trial_number: int
    trial_state: str
    model: str
    params: dict
    objective_value: float | None
    trial_duration_seconds: float | None
    cv_r2_mean: float
    cv_r2_std: float
    cv_r2_min: float
    cv_r2_max: float
    cv_rmse_mean: float
    cv_rmse_min: float
    cv_rmse_max: float
    cv_mae_mean: float
    cv_mae_min: float
    cv_mae_max: float
    cv_fold_r2s: list[float]
    cv_fold_rmses: list[float]
    cv_fold_maes: list[float]
    holdout_r2: float | None
    holdout_rmse: float | None
    holdout_mae: float | None
    holdout_residual_mean: float | None
    holdout_residual_std: float | None
    holdout_abs_residual_mean: float | None
    generalization_gap: float | None
    is_feasible: bool


SOURCE_FILE_COLUMN = "__source_file"
# Feature presets removed: use explicit `--feature-columns` or default features


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for adaptive hyperparameter search.

    Returns:
        argparse.Namespace: parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Adaptive search of hyperparameters until reaching target R2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-glob",
        nargs="+",
        default=["Doback-Data/featured/DOBACK*.csv"],
        help="One or more glob patterns relative to repository root",
    )
    parser.add_argument(
        "--contains",
        nargs="*",
        default=None,
        help="Optional substrings to filter selected file names (all must match)",
    )
    # --max-files removed; all matching input files will be used
    parser.add_argument(
        "--model",
        choices=["rf", "extra_trees", "gbr"],
        default="rf",
        help="Regressor family to optimize",
    )
    parser.add_argument(
        "--target-r2",
        type=float,
        default=0.70,
        help="Stop when holdout R2 reaches this value and constraints are satisfied",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=80,
        help="Maximum optimization trials")
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Stop early if no holdout improvement for this many completed trials",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Folds for CV on training partition",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for tree-based models that support n_jobs (-1 uses all available cores)",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.20,
        help="Fraction used as final holdout partition",
    )
    parser.add_argument(
        "--max-generalization-gap",
        type=float,
        default=0.08,
        help="Max allowed (CV R2 mean - holdout R2)",
    )
    parser.add_argument(
        "--max-r2-std",
        type=float,
        default=0.08,
        help="Max allowed CV R2 std to avoid unstable models",
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
        help="Optional explicit feature columns",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="output/models")
    parser.add_argument("--prefix", default="adaptive_w_model")
    parser.add_argument(
        "--sampler",
        choices=["tpe", "random"],
        default="tpe",
        help="Optuna sampler used to propose new hyperparameter trials",
    )
    parser.add_argument(
        "--pruner",
        choices=["median", "none"],
        default="median",
        help="Optuna pruner used to stop underperforming trials early",
    )
    parser.add_argument(
        "--startup-trials",
        type=int,
        default=10,
        help="Number of startup trials before TPE/pruner become active",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name. Defaults to <prefix>_<model> when --resume is enabled.",
    )
    parser.add_argument(
        "--study-storage",
        default=None,
        help="SQLite file used to persist the Optuna study. Defaults to <output-dir>/<prefix>_<model>_study.sqlite3.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume an existing Optuna study when available",
    )
    return parser.parse_args()


def find_repo_root() -> Path:
    """Return repository root based on this file's location.

    The project places this module under src/lidar_stability/... so the
    repository root is a few parents up.
    """
    return Path(__file__).resolve().parents[3]


def iter_paths_from_globs(repo_root: Path, patterns: Iterable[str]) -> list[Path]:
    """Expand one or more glob patterns relative to `repo_root`.

    Returns a sorted list of matching Path objects.
    """
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(repo_root.glob(pattern)))
    return found


def resolve_input_files(args: argparse.Namespace, repo_root: Path) -> list[Path]:
    """Resolve and filter input CSV files from CLI args.

    Applies `--contains` and validates file existence.
    """
    paths = iter_paths_from_globs(repo_root, args.input_glob)
    unique = sorted({p.resolve() for p in paths if p.exists() and p.suffix.lower() == ".csv"})

    if args.contains:
        needles = [n.lower() for n in args.contains]
        unique = [p for p in unique if all(n in p.name.lower() for n in needles)]

    # --max-files removed: use all matched files after filtering

    if not unique:
        raise FileNotFoundError("No input CSV files found after applying filters")

    return unique


def resolve_feature_columns_from_args(args: argparse.Namespace) -> tuple[list[str] | None, str]:
    """Determine feature columns from explicit args or default preset.

    Returns (columns, resolution_mode) where resolution_mode is 'explicit' or 'default'.
    """
    if args.feature_columns:
        return list(args.feature_columns), "explicit"
    return list(DEFAULT_FEATURE_COLUMNS), "default"






def load_featured_with_source(paths: list[Path]) -> pd.DataFrame:
    """Read CSV files and attach a source filename column.

    Returns a concatenated DataFrame or empty DataFrame when no files.
    """
    frames: list[pd.DataFrame] = []
    for p in paths:
        frame = pd.read_csv(p, low_memory=False)
        frame = frame.copy()
        frame[SOURCE_FILE_COLUMN] = p.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def balanced_sample_by_source(
    df: pd.DataFrame,
    source_col: str,
    max_rows_per_source: int,
    random_state: int,
) -> pd.DataFrame:
    """Reduce per-source imbalance by downsampling each source group.

    If `max_rows_per_source` <= 0 or source column missing, returns input.
    """
    if max_rows_per_source <= 0 or source_col not in df.columns:
        return df

    rng = np.random.default_rng(random_state)
    chunks: list[pd.DataFrame] = []
    for _, group in df.groupby(source_col):
        if len(group) <= max_rows_per_source:
            chunks.append(group)
            continue
        idx = rng.choice(group.index.to_numpy(), size=max_rows_per_source, replace=False)
        chunks.append(group.loc[idx])

    out = pd.concat(chunks, ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def split_train_holdout_by_group(
    df: pd.DataFrame,
    holdout_frac: float,
    random_state: int,
    source_col: str | None = SOURCE_FILE_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train and holdout sets.

    If `source_col` is provided and present, groups are used to split at
    the group level; otherwise rows are shuffled and split.
    Returns (train_idx, holdout_idx).
    """
    if source_col is None or source_col not in df.columns:
        n = len(df)
        rng = np.random.default_rng(random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int((1.0 - holdout_frac) * n)
        return idx[:cut], idx[cut:]

    groups = df[source_col].astype(str)
    unique_groups = groups.unique().tolist()
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_groups)

    n_hold_groups = max(1, int(len(unique_groups) * holdout_frac))
    hold_groups = set(unique_groups[:n_hold_groups])

    hold_mask = groups.isin(hold_groups).to_numpy()
    hold_idx = np.where(hold_mask)[0]
    train_idx = np.where(~hold_mask)[0]

    if len(train_idx) == 0 or len(hold_idx) == 0:
        n = len(df)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int((1.0 - holdout_frac) * n)
        return idx[:cut], idx[cut:]

    return train_idx, hold_idx


def build_model(model_key: str, params: dict, random_state: int, n_jobs: int = -1):
    """Construct a scikit-learn regressor from a parameter dict.

    The function normalizes a few string/np types and maps the params
    to the expected constructor arguments for RF, ExtraTrees or GBR.
    """
    def normalize_max_features(value):
        # sklearn accepts {'sqrt','log2'}, float in (0,1], int >=1, or None.
        if value is None:
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            return value
        text = str(value).strip().lower()
        if text in {"none", "null"}:
            return None
        if text in {"sqrt", "log2"}:
            return text
        try:
            if "." in text:
                return float(text)
            return int(text)
        except ValueError:
            return value

    if model_key == "rf":
        return RandomForestRegressor(
            n_estimators=int(params["rf_n_estimators"]),
            min_samples_leaf=int(params["rf_min_samples_leaf"]),
            max_depth=params["rf_max_depth"],
            max_features=normalize_max_features(params["rf_max_features"]),
            min_samples_split=int(params["rf_min_samples_split"]),
            bootstrap=True,
            oob_score=True,
            random_state=random_state,
            n_jobs=int(n_jobs),
        )
    if model_key == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(params["extra_trees_n_estimators"]),
            min_samples_leaf=int(params["extra_trees_min_samples_leaf"]),
            max_depth=params["extra_trees_max_depth"],
            max_features=normalize_max_features(params["extra_trees_max_features"]),
            min_samples_split=int(params["extra_trees_min_samples_split"]),
            random_state=random_state,
            n_jobs=int(n_jobs),
        )
    if model_key == "gbr":
        return GradientBoostingRegressor(
            n_estimators=int(params["gbr_n_estimators"]),
            learning_rate=float(params["gbr_learning_rate"]),
            max_depth=int(params["gbr_max_depth"]),
            min_samples_leaf=int(params["gbr_min_samples_leaf"]),
            min_samples_split=int(params["gbr_min_samples_split"]),
            subsample=float(params["gbr_subsample"]),
            random_state=random_state,
        )
    raise ValueError(f"Unknown model key: {model_key}")


def ensure_optuna_available() -> None:
    """Raise helpful error when Optuna isn't installed.

    This script requires Optuna for the adaptive search loop.
    """
    if optuna is None:
        raise ImportError(
            "Optuna is required for adaptive_hyperparam_search.py. "
            "Install project dependencies again to include 'optuna'."
        )


def suggest_params(trial, model_key: str) -> dict:
    """Given an Optuna trial, suggest hyperparameters for `model_key`.

    Encapsulates parameter search spaces for each supported model.
    """
    def draw_max_depth(prefix: str) -> int | None:
        mode = trial.suggest_categorical(f"{prefix}_max_depth_mode", ["none", "bounded"])
        if mode == "none":
            return None
        return int(trial.suggest_int(f"{prefix}_max_depth", 8, 30))

    def draw_max_features(prefix: str) -> str | float:
        choice = trial.suggest_categorical(
            f"{prefix}_max_features_choice",
            ["sqrt", "log2", "0.5", "0.8"],
        )
        if choice in {"sqrt", "log2"}:
            return choice
        return float(choice)

    if model_key == "rf":
        return {
            "rf_n_estimators": int(trial.suggest_int("rf_n_estimators", 200, 1200)),
            "rf_min_samples_leaf": int(trial.suggest_int("rf_min_samples_leaf", 2, 12)),
            "rf_max_depth": draw_max_depth("rf"),
            "rf_max_features": draw_max_features("rf"),
            "rf_min_samples_split": int(trial.suggest_int("rf_min_samples_split", 4, 24)),
        }

    if model_key == "extra_trees":
        return {
            "extra_trees_n_estimators": int(trial.suggest_int("extra_trees_n_estimators", 250, 1400)),
            "extra_trees_min_samples_leaf": int(trial.suggest_int("extra_trees_min_samples_leaf", 2, 11)),
            "extra_trees_max_depth": draw_max_depth("extra_trees"),
            "extra_trees_max_features": draw_max_features("extra_trees"),
            "extra_trees_min_samples_split": int(trial.suggest_int("extra_trees_min_samples_split", 4, 24)),
        }

    return {
        "gbr_n_estimators": int(trial.suggest_int("gbr_n_estimators", 150, 900)),
        "gbr_learning_rate": float(
            trial.suggest_categorical("gbr_learning_rate", [0.02, 0.03, 0.05, 0.08, 0.10])
        ),
        "gbr_max_depth": int(trial.suggest_int("gbr_max_depth", 2, 6)),
        "gbr_min_samples_leaf": int(trial.suggest_int("gbr_min_samples_leaf", 2, 15)),
        "gbr_min_samples_split": int(trial.suggest_int("gbr_min_samples_split", 4, 25)),
        "gbr_subsample": float(trial.suggest_categorical("gbr_subsample", [0.65, 0.75, 0.85, 0.95])),
    }


def is_trial_feasible(result: TrialResult, args: argparse.Namespace) -> bool:
    """Return True when a trial satisfies all configured constraints.

    Constraints include minimum holdout R2, maximum generalization gap and
    maximum CV R2 standard deviation.
    """
    if result.holdout_r2 is None or result.generalization_gap is None:
        return False
    return (
        result.holdout_r2 >= float(args.target_r2)
        and result.generalization_gap <= float(args.max_generalization_gap)
        and result.cv_r2_std <= float(args.max_r2_std)
    )


def serialize_trial_result(result: TrialResult) -> dict:
    """Convert a TrialResult dataclass into a serializable dict.

    This is stored as an Optuna user attribute for later inspection.
    """
    return asdict(result)


def _coerce_float_list(value) -> list[float]:
    """Best-effort conversion of list-like payloads to float lists."""
    if not isinstance(value, (list, tuple, np.ndarray)):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _safe_stat(values: list[float], reducer, default: float | None = None) -> float | None:
    """Compute a scalar summary if values exist, otherwise return default."""
    if not values:
        return default
    return float(reducer(np.asarray(values, dtype=float)))





def _summarize_feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    """Extract tree-based feature importance if the model exposes it."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    importance = np.asarray(getattr(model, "feature_importances_"), dtype=float)
    if len(importance) != len(feature_columns):
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.DataFrame({"feature": feature_columns, "importance": importance})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _summarize_permutation_importance(
    model,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    feature_columns: list[str],
    random_state: int,
) -> pd.DataFrame:
    """Compute permutation importance on the evaluation partition."""
    if X_eval.empty or y_eval.empty:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    result = permutation_importance(
        model,
        X_eval,
        y_eval,
        scoring="r2",
        n_repeats=5,
        random_state=int(random_state),
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": np.asarray(result.importances_mean, dtype=float),
            "importance_std": np.asarray(result.importances_std, dtype=float),
        }
    )
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)





def deserialize_trial_result(payload: dict) -> TrialResult:
    """Reconstruct a TrialResult from a dict previously produced by
    `serialize_trial_result`.
    """
    cv_fold_r2s = _coerce_float_list(payload.get("cv_fold_r2s"))
    cv_fold_rmses = _coerce_float_list(payload.get("cv_fold_rmses"))
    cv_fold_maes = _coerce_float_list(payload.get("cv_fold_maes"))
    cv_r2_mean = float(payload.get("cv_r2_mean", np.mean(cv_fold_r2s) if cv_fold_r2s else np.nan))
    cv_r2_std = float(payload.get("cv_r2_std", np.std(cv_fold_r2s) if cv_fold_r2s else np.nan))
    cv_rmse_mean = float(payload.get("cv_rmse_mean", np.mean(cv_fold_rmses) if cv_fold_rmses else np.nan))
    cv_mae_mean = float(payload.get("cv_mae_mean", np.mean(cv_fold_maes) if cv_fold_maes else np.nan))
    return TrialResult(
        trial=int(payload["trial"]),
        optuna_trial_number=int(payload["optuna_trial_number"]),
        trial_state=str(payload["trial_state"]),
        model=str(payload["model"]),
        params=dict(payload["params"]),
        objective_value=payload.get("objective_value"),
        trial_duration_seconds=payload.get("trial_duration_seconds"),
        cv_r2_mean=cv_r2_mean,
        cv_r2_std=cv_r2_std,
        cv_r2_min=float(payload.get("cv_r2_min", min(cv_fold_r2s) if cv_fold_r2s else cv_r2_mean)),
        cv_r2_max=float(payload.get("cv_r2_max", max(cv_fold_r2s) if cv_fold_r2s else cv_r2_mean)),
        cv_rmse_mean=cv_rmse_mean,
        cv_rmse_min=float(payload.get("cv_rmse_min", min(cv_fold_rmses) if cv_fold_rmses else cv_rmse_mean)),
        cv_rmse_max=float(payload.get("cv_rmse_max", max(cv_fold_rmses) if cv_fold_rmses else cv_rmse_mean)),
        cv_mae_mean=cv_mae_mean,
        cv_mae_min=float(payload.get("cv_mae_min", min(cv_fold_maes) if cv_fold_maes else cv_mae_mean)),
        cv_mae_max=float(payload.get("cv_mae_max", max(cv_fold_maes) if cv_fold_maes else cv_mae_mean)),
        cv_fold_r2s=cv_fold_r2s,
        cv_fold_rmses=cv_fold_rmses,
        cv_fold_maes=cv_fold_maes,
        holdout_r2=payload.get("holdout_r2"),
        holdout_rmse=payload.get("holdout_rmse"),
        holdout_mae=payload.get("holdout_mae"),
        holdout_residual_mean=payload.get("holdout_residual_mean"),
        holdout_residual_std=payload.get("holdout_residual_std"),
        holdout_abs_residual_mean=payload.get("holdout_abs_residual_mean"),
        generalization_gap=payload.get("generalization_gap"),
        is_feasible=bool(payload.get("is_feasible", False)),
    )


def fit_final_model(
    model_key: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_jobs: int = -1,
):
    """Fit and return a final model on provided training data.

    This is used to produce the artifact model after search completes.
    """
    model = build_model(model_key, params=params, random_state=random_state, n_jobs=n_jobs)
    model.fit(X_train, y_train)
    return model


def build_cv_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series | None,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build cross-validation split index pairs.

    Uses GroupKFold when groups are provided, otherwise KFold with shuffling.
    """
    if train_groups is not None:
        n_groups = int(train_groups.nunique())
        if n_groups < n_splits:
            raise ValueError(
                f"GroupKFold requires at least {n_splits} unique groups, "
                f"but only {n_groups} were found"
            )
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(X_train, y_train, groups=train_groups))

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(X_train, y_train))


def _to_contiguous_feature_matrix(X: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to a contiguous numpy matrix of floats.

    Using contiguous arrays avoids potential overhead when passing data to
    compiled libraries in tight loops.
    """
    return np.ascontiguousarray(X.to_numpy(dtype=float, copy=False))


def _to_contiguous_target_array(y: pd.Series) -> np.ndarray:
    """Convert target Series to contiguous float numpy array.
    """
    return np.ascontiguousarray(np.asarray(y, dtype=float))


def objective_factory(
    *,
    args: argparse.Namespace,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_hold: pd.DataFrame,
    y_hold: pd.Series,
    train_groups: pd.Series | None,
) -> Callable:
    """Create an Optuna objective callable bound to the training data.

    The returned function will be called by Optuna and perform CV, optional
    pruning, and final evaluation on the holdout set. It records per-trial
    results as user attributes so they can be serialized into the study.
    """
    split_indices = build_cv_splits(
        X_train=X_train,
        y_train=y_train,
        train_groups=train_groups,
        n_splits=max(2, int(args.n_splits)),
        random_state=int(args.random_state),
    )
    X_train_values = _to_contiguous_feature_matrix(X_train)
    y_train_values = _to_contiguous_target_array(y_train)

    def objective(trial) -> float:
        trial_started = time.perf_counter()
        params = suggest_params(trial, args.model)
        trial.set_user_attr("resolved_params", params)

        r2_vals: list[float] = []
        rmse_vals: list[float] = []
        mae_vals: list[float] = []

        # Perform CV folds using pre-computed index pairs
        for fold_idx, (tr_idx, va_idx) in enumerate(split_indices, start=1):
            model = build_model(
                args.model,
                params=params,
                random_state=int(args.random_state) + trial.number + fold_idx,
                n_jobs=int(args.n_jobs),
            )
            model.fit(X_train_values[tr_idx], y_train_values[tr_idx])

            pred = model.predict(X_train_values[va_idx])
            true = y_train_values[va_idx]

            r2_vals.append(float(r2_score(true, pred)))
            rmse_vals.append(float(np.sqrt(mean_squared_error(true, pred))))
            mae_vals.append(float(mean_absolute_error(true, pred)))

            # Report intermediate value for Optuna pruning and progress
            trial.report(float(np.mean(r2_vals)), step=fold_idx)
            if args.pruner != "none" and trial.should_prune():
                partial_result = TrialResult(
                    trial=trial.number + 1,
                    optuna_trial_number=trial.number,
                    trial_state="PRUNED",
                    model=args.model,
                    params=params,
                    objective_value=float(np.mean(r2_vals)),
                    trial_duration_seconds=float(time.perf_counter() - trial_started),
                    cv_r2_mean=float(np.mean(r2_vals)),
                    cv_r2_std=float(np.std(r2_vals)),
                    cv_r2_min=float(np.min(r2_vals)),
                    cv_r2_max=float(np.max(r2_vals)),
                    cv_rmse_mean=float(np.mean(rmse_vals)),
                    cv_rmse_min=float(np.min(rmse_vals)),
                    cv_rmse_max=float(np.max(rmse_vals)),
                    cv_mae_mean=float(np.mean(mae_vals)),
                    cv_mae_min=float(np.min(mae_vals)),
                    cv_mae_max=float(np.max(mae_vals)),
                    cv_fold_r2s=list(r2_vals),
                    cv_fold_rmses=list(rmse_vals),
                    cv_fold_maes=list(mae_vals),
                    holdout_r2=None,
                    holdout_rmse=None,
                    holdout_mae=None,
                    holdout_residual_mean=None,
                    holdout_residual_std=None,
                    holdout_abs_residual_mean=None,
                    generalization_gap=None,
                    is_feasible=False,
                )
                trial.set_user_attr("trial_result", serialize_trial_result(partial_result))
                raise optuna.exceptions.TrialPruned("Pruned by configured Optuna pruner")

        # Fit a final model on the whole training partition to evaluate
        # performance on the held-out data.
        final_model = fit_final_model(
            args.model,
            params=params,
            X_train=X_train,
            y_train=y_train,
            random_state=int(args.random_state) + trial.number,
            n_jobs=int(args.n_jobs),
        )
        hold_pred = final_model.predict(X_hold)
        hold_residuals = np.asarray(y_hold, dtype=float) - np.asarray(hold_pred, dtype=float)
        hold_r2 = float(r2_score(y_hold, hold_pred))
        hold_rmse = float(np.sqrt(mean_squared_error(y_hold, hold_pred)))
        hold_mae = float(mean_absolute_error(y_hold, hold_pred))

        # Build a TrialResult summarizing CV and holdout metrics.
        result = TrialResult(
            trial=trial.number + 1,
            optuna_trial_number=trial.number,
            trial_state="COMPLETE",
            model=args.model,
            params=params,
            objective_value=float(np.mean(r2_vals)),
            trial_duration_seconds=float(time.perf_counter() - trial_started),
            cv_r2_mean=float(np.mean(r2_vals)),
            cv_r2_std=float(np.std(r2_vals)),
            cv_r2_min=float(np.min(r2_vals)),
            cv_r2_max=float(np.max(r2_vals)),
            cv_rmse_mean=float(np.mean(rmse_vals)),
            cv_rmse_min=float(np.min(rmse_vals)),
            cv_rmse_max=float(np.max(rmse_vals)),
            cv_mae_mean=float(np.mean(mae_vals)),
            cv_mae_min=float(np.min(mae_vals)),
            cv_mae_max=float(np.max(mae_vals)),
            cv_fold_r2s=list(r2_vals),
            cv_fold_rmses=list(rmse_vals),
            cv_fold_maes=list(mae_vals),
            holdout_r2=hold_r2,
            holdout_rmse=hold_rmse,
            holdout_mae=hold_mae,
            holdout_residual_mean=float(np.mean(hold_residuals)),
            holdout_residual_std=float(np.std(hold_residuals)),
            holdout_abs_residual_mean=float(np.mean(np.abs(hold_residuals))),
            generalization_gap=float(np.mean(r2_vals) - hold_r2),
            is_feasible=False,
        )
        result.is_feasible = is_trial_feasible(result, args)
        trial.set_user_attr("trial_result", serialize_trial_result(result))
        return result.objective_value if result.objective_value is not None else result.cv_r2_mean

    return objective


def create_sampler(args: argparse.Namespace):
    ensure_optuna_available()
    if args.sampler == "random":
        return optuna.samplers.RandomSampler(seed=int(args.random_state))
    return optuna.samplers.TPESampler(
        seed=int(args.random_state),
        multivariate=True,
        n_startup_trials=int(args.startup_trials),
    )


def create_pruner(args: argparse.Namespace):
    ensure_optuna_available()
    if args.pruner == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(args.startup_trials),
        n_warmup_steps=2,
    )


def to_sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def resolve_study_storage_path(repo_root: Path, out_dir: Path, raw_value: str | None, prefix: str, model: str) -> Path:
    if raw_value:
        path = Path(raw_value)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return path
    return (out_dir / f"{prefix}_{model}_study.sqlite3").resolve()


def resolve_study_name(args: argparse.Namespace) -> str:
    base = args.study_name or f"{args.prefix}_{args.model}"
    if args.resume:
        return base
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{base}_{suffix}"


def load_trial_histories(study) -> tuple[list[TrialResult], list[TrialResult]]:
    completed: list[TrialResult] = []
    pruned: list[TrialResult] = []
    for frozen_trial in sorted(study.trials, key=lambda item: item.number):
        payload = frozen_trial.user_attrs.get("trial_result")
        if not isinstance(payload, dict):
            continue
        result = deserialize_trial_result(payload)
        if result.trial_state == "PRUNED":
            pruned.append(result)
        elif result.trial_state == "COMPLETE":
            completed.append(result)
    return completed, pruned


def rebuild_patience_state(history: list[TrialResult]) -> tuple[float, int]:
    best_holdout = float("-inf")
    no_improve = 0
    for result in history:
        holdout_r2 = float(result.holdout_r2) if result.holdout_r2 is not None else float("-inf")
        if holdout_r2 > best_holdout:
            best_holdout = holdout_r2
            no_improve = 0
        else:
            no_improve += 1
    return best_holdout, no_improve


def select_best_trial(history: list[TrialResult]) -> tuple[TrialResult, bool]:
    feasible = [result for result in history if result.is_feasible]
    if feasible:
        best = max(
            feasible,
            key=lambda result: (
                float(result.holdout_r2) if result.holdout_r2 is not None else float("-inf"),
                result.cv_r2_mean,
                -result.trial,
            ),
        )
        return best, True

    best = max(
        history,
        key=lambda result: (
            float(result.holdout_r2) if result.holdout_r2 is not None else float("-inf"),
            result.cv_r2_mean,
            -result.trial,
        ),
    )
    return best, False


def run_search(args: argparse.Namespace) -> int:
    ensure_optuna_available()

    repo_root = find_repo_root()
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    args.study_name = resolve_study_name(args)
    args.study_storage = str(
        resolve_study_storage_path(
            repo_root=repo_root,
            out_dir=out_dir,
            raw_value=args.study_storage,
            prefix=args.prefix,
            model=args.model,
        )
    )
    study_storage_path = Path(args.study_storage)
    study_storage_path.parent.mkdir(parents=True, exist_ok=True)

    csv_paths = resolve_input_files(args, repo_root)
    print(f"Selected files: {len(csv_paths)}")

    resolved_feature_columns, feature_resolution_mode = resolve_feature_columns_from_args(args)

    raw_df = load_featured_with_source(csv_paths)
    if raw_df.empty:
        raise RuntimeError("Loaded dataframe is empty")

    X, y, used_features, clean_df = build_w_training_dataset(
        raw_df,
        feature_columns=resolved_feature_columns,
        target_column=args.target_column,
    )

    clean_df = clean_df.loc[X.index].copy()

    train_idx, hold_idx = split_train_holdout_by_group(
        clean_df,
        holdout_frac=float(args.holdout_frac),
        random_state=int(args.random_state),
    )

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_hold, y_hold = X.iloc[hold_idx], y.iloc[hold_idx]
    train_groups = None

    if len(X_train) < max(200, args.n_splits * 20) or len(X_hold) < 100:
        raise RuntimeError(
            "Not enough samples for robust train/holdout evaluation after filtering. "
            "Increase input size or reduce filtering."
        )

    print(f"Samples after cleaning: {len(X)} | train={len(X_train)} holdout={len(X_hold)}")
    print(f"Used features ({len(used_features)}): {used_features}")

    selected_files_rel = [str(p.relative_to(repo_root)) for p in csv_paths]
    training_context = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "command": " ".join(sys.argv),
        "selected_files": selected_files_rel,
        "selected_files_count": len(selected_files_rel),
        "rows_after_cleaning": int(len(X)),
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_hold)),
        "resolved_target_name": "gy",
        "resolved_feature_columns": list(used_features),
        "feature_resolution_mode": feature_resolution_mode,
    }

    objective = objective_factory(
        args=args,
        X_train=X_train,
        y_train=y_train,
        X_hold=X_hold,
        y_hold=y_hold,
        train_groups=train_groups,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=to_sqlite_url(study_storage_path),
        load_if_exists=bool(args.resume),
        direction="maximize",
        sampler=create_sampler(args),
        pruner=create_pruner(args),
    )

    history, pruned_history = load_trial_histories(study)
    best_holdout, no_improve = rebuild_patience_state(history)

    if any(result.is_feasible for result in history):
        print("Existing feasible Optuna trial found in resumed study. Skipping new optimization.")
    else:
        while len(study.trials) < int(args.max_trials):
            study.optimize(objective, n_trials=1)
            history, pruned_history = load_trial_histories(study)

            latest_complete = history[-1] if history else None
            latest_trial = study.trials[-1] if study.trials else None
            if latest_trial is None:
                break

            if latest_trial.state.name == "COMPLETE" and latest_complete is not None:
                holdout_r2 = float(latest_complete.holdout_r2) if latest_complete.holdout_r2 is not None else float("-inf")
                is_new_best = holdout_r2 > best_holdout
                if is_new_best:
                    best_holdout = holdout_r2
                    no_improve = 0
                else:
                    no_improve += 1

                print(
                    f"Trial {latest_complete.trial:03d} | "
                    f"CV_R2={latest_complete.cv_r2_mean:.4f}+-{latest_complete.cv_r2_std:.4f} | "
                    f"Holdout_R2={latest_complete.holdout_r2:.4f} | "
                    f"Gap={latest_complete.generalization_gap:.4f} | "
                    f"Feasible={latest_complete.is_feasible}"
                )

                if latest_complete.is_feasible:
                    print("Stopping criterion reached: feasible Optuna trial found.")
                    break

                if no_improve >= int(args.patience):
                    print("Early stop by patience: no holdout improvement among completed trials.")
                    break
            elif latest_trial.state.name == "PRUNED":
                payload = latest_trial.user_attrs.get("trial_result", {})
                pruned_result = deserialize_trial_result(payload) if isinstance(payload, dict) else None
                if pruned_result is not None:
                    print(
                        f"Trial {pruned_result.trial:03d} | "
                        f"CV_R2_partial={pruned_result.cv_r2_mean:.4f} | "
                        f"state=PRUNED"
                    )

    history, pruned_history = load_trial_histories(study)
    if not history:
        raise RuntimeError("Search failed to produce any completed trial")

    best, constraints_satisfied = select_best_trial(history)
    best_model_obj = fit_final_model(
        args.model,
        params=best.params,
        X_train=X_train,
        y_train=y_train,
        random_state=int(args.random_state),
        n_jobs=int(args.n_jobs),
    )
    holdout_pred = best_model_obj.predict(X_hold)
    holdout_frame = clean_df.iloc[hold_idx].copy()
    feature_importance_df = _summarize_feature_importance(best_model_obj, used_features)
    permutation_importance_df = _summarize_permutation_importance(
        best_model_obj,
        X_hold,
        y_hold,
        used_features,
        random_state=int(args.random_state),
    )

    model_path = out_dir / f"{args.prefix}_{args.model}_best.joblib"
    history_path = out_dir / f"{args.prefix}_{args.model}_history.json"
    leaderboard_path = out_dir / f"{args.prefix}_{args.model}_leaderboard.csv"
    leaderboard_json_path = out_dir / f"{args.prefix}_{args.model}_leaderboard.json"
    report_path = out_dir / f"{args.prefix}_{args.model}_report.md"
    holdout_path = out_dir / f"{args.prefix}_{args.model}_holdout_predictions.csv"

    artifact = {
        "model": best_model_obj,
        "feature_columns": used_features,
        "target_name": "gy",
        "target_column_input": args.target_column,
        "model_key": args.model,
        "run_id": f"adaptive_target_{str(args.target_r2).replace('.', '_')}",
        "params": dict(best.params),
        "n_samples": int(len(X_train)),
        "n_train_samples": int(len(X_train)),
        "n_holdout_samples": int(len(X_hold)),
        "n_search_rows": int(len(X)),
        "n_features": int(len(used_features)),
        "training_context": training_context,
        "search_engine": "optuna",
        "search_best": serialize_trial_result(best),
        "search_constraints": {
            "target_r2": args.target_r2,
            "max_generalization_gap": args.max_generalization_gap,
            "max_r2_std": args.max_r2_std,
            "constraints_satisfied": constraints_satisfied,
        },
        "study_name": args.study_name,
        "study_storage": str(study_storage_path),
        "report_path": "",
        "holdout_predictions_path": str(holdout_path),
    }

    dump(artifact, model_path, compress=3)

    sampler_name = study.sampler.__class__.__name__
    pruner_name = study.pruner.__class__.__name__
    history_payload = {
        "format": "adaptive_history_v3",
        "config": vars(args),
        "search_engine": "optuna",
        "study_name": args.study_name,
        "study_storage": str(study_storage_path),
        "sampler": sampler_name,
        "pruner": pruner_name,
        "training_context": training_context,
        "dataset": {
            "n_samples": int(len(X)),
            "n_train_samples": int(len(X_train)),
            "n_holdout_samples": int(len(X_hold)),
            "n_features": int(len(used_features)),
            "feature_columns": list(used_features),
            "target_name": "gy",
            "target_column_input": args.target_column,
        },
        "n_trials": int(len(study.trials)),
        "n_completed_trials": int(len(history)),
        "n_pruned_trials": int(len(pruned_history)),
        "constraints_satisfied": constraints_satisfied,
        "best": serialize_trial_result(best),
        "history": [serialize_trial_result(h) for h in history],
        "pruned_history": [serialize_trial_result(h) for h in pruned_history],
        "model_path": str(model_path),
        "report_path": str(report_path),
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    rank_df = pd.DataFrame([serialize_trial_result(h) for h in history]).sort_values(
        by=["holdout_r2", "cv_r2_mean"],
        ascending=False,
    )
    rank_df.to_csv(leaderboard_path, index=False)

    leaderboard_rows = []
    for h in history:
        row = serialize_trial_result(h)
        row.update(
            {
                "run_id": int(h.trial),
                "label": f"{args.model}@{h.trial}",
                "rmse_mean": float(h.cv_rmse_mean),
                "mae_mean": float(h.cv_mae_mean),
                "r2_mean": float(h.cv_r2_mean),
                "n_samples": int(len(X)),
                "n_features": int(len(used_features)),
                "params": dict(h.params),
                "feature_columns": list(used_features),
                "target_name": "gy",
                "model_path": str(model_path),
                "metrics_path": str(history_path),
                "report_path": "",
                "holdout_predictions_path": str(holdout_path),
                "training_context": training_context,
            }
        )
        leaderboard_rows.append(row)
    leaderboard_json_path.write_text(json.dumps(leaderboard_rows, indent=2), encoding="utf-8")

    # Persist holdout predictions for downstream analysis (notebook, plots)
    try:
        hold_df = holdout_frame.copy()
        hold_df["y_true"] = np.asarray(y_hold, dtype=float)
        hold_df["y_pred"] = np.asarray(holdout_pred, dtype=float)
        hold_df["residual"] = hold_df["y_true"] - hold_df["y_pred"]
        hold_df["abs_residual"] = np.abs(hold_df["residual"]).astype(float)
        hold_df.to_csv(holdout_path, index=False)
    except Exception:
        # Non-fatal: ensure search still completes even if saving holdout fails
        pass

    print("\nBest trial summary")
    print(
        f"  trial={best.trial} holdout_R2={best.holdout_r2:.4f} "
        f"cv_R2={best.cv_r2_mean:.4f}+-{best.cv_r2_std:.4f} "
        f"gap={best.generalization_gap:.4f} feasible={constraints_satisfied}"
    )
    print(f"  model: {model_path}")
    print(f"  history: {history_path}")
    print(f"  leaderboard: {leaderboard_path}")
    print(f"  leaderboard_json: {leaderboard_json_path}")
    print(f"  holdout_predictions: {holdout_path}")
    print(f"  study: {study_storage_path}")

    run_config = {
        "model": args.model,
        "run_id": f"adaptive_r2_{str(args.target_r2).replace('.', '_')}",
        **dict(best.params),
    }
    print("\nEquivalent --run-config for train_models_cli")
    print(json.dumps(run_config))

    return 0


def main() -> int:
    args = parse_args()
    return run_search(args)


if __name__ == "__main__":
    raise SystemExit(main())
