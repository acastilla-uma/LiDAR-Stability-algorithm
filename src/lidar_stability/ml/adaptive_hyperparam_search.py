#!/usr/bin/env python3
"""Adaptive hyperparameter search until reaching a target R2.

This script trains regression models iteratively, changes hyperparameters at each
trial, and stops early once the target R2 is achieved with anti-overfitting
constraints.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
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
class TrialResult:
    trial: int
    model: str
    params: dict
    cv_r2_mean: float
    cv_r2_std: float
    cv_rmse_mean: float
    cv_mae_mean: float
    holdout_r2: float
    holdout_rmse: float
    holdout_mae: float
    generalization_gap: float


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If > 0, limit number of input files after filtering",
    )
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
    parser.add_argument("--max-trials", type=int, default=80, help="Maximum optimization trials")
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Stop early if no holdout improvement for this many trials",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Folds for CV on training partition",
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
        "--max-rows",
        type=int,
        default=0,
        help="If > 0, randomly sample this many rows from full dataset",
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=3000,
        help="Cap rows per source file to reduce source imbalance. 0 disables",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Optional explicit target column. Auto-detection if omitted",
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
    return parser.parse_args()


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def iter_paths_from_globs(repo_root: Path, patterns: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(repo_root.glob(pattern)))
    return found


def resolve_input_files(args: argparse.Namespace, repo_root: Path) -> list[Path]:
    paths = iter_paths_from_globs(repo_root, args.input_glob)
    unique = sorted({p.resolve() for p in paths if p.exists() and p.suffix.lower() == ".csv"})

    if args.contains:
        needles = [n.lower() for n in args.contains]
        unique = [p for p in unique if all(n in p.name.lower() for n in needles)]

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
        frame["__source_file"] = p.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def balanced_sample_by_source(
    df: pd.DataFrame,
    source_col: str,
    max_rows_per_source: int,
    random_state: int,
) -> pd.DataFrame:
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
    source_col: str = "__source_file",
) -> tuple[np.ndarray, np.ndarray]:
    if source_col not in df.columns:
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


def build_model(model_key: str, params: dict, random_state: int):
    if model_key == "rf":
        return RandomForestRegressor(
            n_estimators=int(params["rf_n_estimators"]),
            min_samples_leaf=int(params["rf_min_samples_leaf"]),
            max_depth=params["rf_max_depth"],
            max_features=params["rf_max_features"],
            min_samples_split=int(params["rf_min_samples_split"]),
            bootstrap=True,
            oob_score=True,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_key == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(params["extra_trees_n_estimators"]),
            min_samples_leaf=int(params["extra_trees_min_samples_leaf"]),
            max_depth=params["extra_trees_max_depth"],
            max_features=params["extra_trees_max_features"],
            min_samples_split=int(params["extra_trees_min_samples_split"]),
            random_state=random_state,
            n_jobs=-1,
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


def sample_params(model_key: str, rng: np.random.Generator) -> dict:
    if model_key == "rf":
        max_depth = None if rng.random() < 0.20 else int(rng.integers(8, 31))
        return {
            "rf_n_estimators": int(rng.integers(200, 1201)),
            "rf_min_samples_leaf": int(rng.integers(2, 13)),
            "rf_max_depth": max_depth,
            "rf_max_features": rng.choice(["sqrt", "log2", 0.5, 0.8]),
            "rf_min_samples_split": int(rng.integers(4, 25)),
        }

    if model_key == "extra_trees":
        max_depth = None if rng.random() < 0.20 else int(rng.integers(8, 31))
        return {
            "extra_trees_n_estimators": int(rng.integers(250, 1401)),
            "extra_trees_min_samples_leaf": int(rng.integers(2, 12)),
            "extra_trees_max_depth": max_depth,
            "extra_trees_max_features": rng.choice(["sqrt", "log2", 0.5, 0.8]),
            "extra_trees_min_samples_split": int(rng.integers(4, 25)),
        }

    return {
        "gbr_n_estimators": int(rng.integers(150, 901)),
        "gbr_learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08, 0.10])),
        "gbr_max_depth": int(rng.integers(2, 7)),
        "gbr_min_samples_leaf": int(rng.integers(2, 16)),
        "gbr_min_samples_split": int(rng.integers(4, 26)),
        "gbr_subsample": float(rng.choice([0.65, 0.75, 0.85, 0.95])),
    }


def evaluate_trial(
    model_key: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_hold: pd.DataFrame,
    y_hold: pd.Series,
    train_groups: pd.Series | None,
    n_splits: int,
    random_state: int,
) -> TrialResult:
    if train_groups is not None and train_groups.nunique() >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X_train, y_train, groups=train_groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X_train, y_train)

    r2_vals: list[float] = []
    rmse_vals: list[float] = []
    mae_vals: list[float] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        model = build_model(model_key, params=params, random_state=random_state + fold_idx)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])

        pred = model.predict(X_train.iloc[va_idx])
        true = y_train.iloc[va_idx]
        r2_vals.append(float(r2_score(true, pred)))
        rmse_vals.append(float(np.sqrt(mean_squared_error(true, pred))))
        mae_vals.append(float(mean_absolute_error(true, pred)))

    final_model = build_model(model_key, params=params, random_state=random_state)
    final_model.fit(X_train, y_train)

    hold_pred = final_model.predict(X_hold)
    hold_r2 = float(r2_score(y_hold, hold_pred))
    hold_rmse = float(np.sqrt(mean_squared_error(y_hold, hold_pred)))
    hold_mae = float(mean_absolute_error(y_hold, hold_pred))

    return TrialResult(
        trial=-1,
        model=model_key,
        params=params,
        cv_r2_mean=float(np.mean(r2_vals)),
        cv_r2_std=float(np.std(r2_vals)),
        cv_rmse_mean=float(np.mean(rmse_vals)),
        cv_mae_mean=float(np.mean(mae_vals)),
        holdout_r2=hold_r2,
        holdout_rmse=hold_rmse,
        holdout_mae=hold_mae,
        generalization_gap=float(np.mean(r2_vals) - hold_r2),
    )


def run_search(args: argparse.Namespace) -> int:
    repo_root = find_repo_root()
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = resolve_input_files(args, repo_root)
    print(f"Selected files: {len(csv_paths)}")

    raw_df = load_featured_with_source(csv_paths)
    if raw_df.empty:
        raise RuntimeError("Loaded dataframe is empty")

    raw_df = balanced_sample_by_source(
        raw_df,
        source_col="__source_file",
        max_rows_per_source=int(args.max_rows_per_source),
        random_state=int(args.random_state),
    )

    if args.max_rows > 0 and len(raw_df) > args.max_rows:
        raw_df = raw_df.sample(n=args.max_rows, random_state=args.random_state).reset_index(drop=True)

    X, y, used_features, clean_df = build_w_training_dataset(
        raw_df,
        feature_columns=args.feature_columns,
        target_column=args.target_column,
    )

    clean_df = clean_df.loc[X.index].copy()
    source_groups = clean_df["__source_file"].astype(str) if "__source_file" in clean_df.columns else None

    train_idx, hold_idx = split_train_holdout_by_group(
        clean_df,
        holdout_frac=float(args.holdout_frac),
        random_state=int(args.random_state),
        source_col="__source_file",
    )

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_hold, y_hold = X.iloc[hold_idx], y.iloc[hold_idx]
    train_groups = source_groups.iloc[train_idx] if source_groups is not None else None

    if len(X_train) < max(200, args.n_splits * 20) or len(X_hold) < 100:
        raise RuntimeError(
            "Not enough samples for robust train/holdout evaluation after filtering. "
            "Increase input size or reduce filtering."
        )

    print(f"Samples after cleaning: {len(X)} | train={len(X_train)} holdout={len(X_hold)}")
    print(f"Used features ({len(used_features)}): {used_features}")

    rng = np.random.default_rng(args.random_state)

    history: list[TrialResult] = []
    best: TrialResult | None = None
    best_model_obj = None
    no_improve = 0

    for trial in range(1, int(args.max_trials) + 1):
        params = sample_params(args.model, rng)
        trial_result = evaluate_trial(
            model_key=args.model,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_hold=X_hold,
            y_hold=y_hold,
            train_groups=train_groups,
            n_splits=max(2, int(args.n_splits)),
            random_state=int(args.random_state) + trial,
        )
        trial_result.trial = trial
        history.append(trial_result)

        is_new_best = best is None or trial_result.holdout_r2 > best.holdout_r2
        if is_new_best:
            best = trial_result
            best_model_obj = build_model(args.model, params=params, random_state=int(args.random_state))
            best_model_obj.fit(X, y)
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Trial {trial:03d} | "
            f"CV_R2={trial_result.cv_r2_mean:.4f}+-{trial_result.cv_r2_std:.4f} | "
            f"Holdout_R2={trial_result.holdout_r2:.4f} | "
            f"Gap={trial_result.generalization_gap:.4f}"
        )

        reached_target = (
            trial_result.holdout_r2 >= float(args.target_r2)
            and trial_result.generalization_gap <= float(args.max_generalization_gap)
            and trial_result.cv_r2_std <= float(args.max_r2_std)
        )
        if reached_target:
            print("Stopping criterion reached: target R2 with stability/generalization constraints.")
            break

        if no_improve >= int(args.patience):
            print("Early stop by patience: no holdout improvement.")
            break

    if best is None or best_model_obj is None:
        raise RuntimeError("Search failed to produce any valid trial")

    artifact = {
        "model": best_model_obj,
        "feature_columns": used_features,
        "target_name": "omega_rad_s",
        "model_key": args.model,
        "search_best": asdict(best),
        "search_constraints": {
            "target_r2": args.target_r2,
            "max_generalization_gap": args.max_generalization_gap,
            "max_r2_std": args.max_r2_std,
        },
    }

    model_path = out_dir / f"{args.prefix}_{args.model}_best.joblib"
    history_path = out_dir / f"{args.prefix}_{args.model}_history.json"
    leaderboard_path = out_dir / f"{args.prefix}_{args.model}_leaderboard.csv"

    dump(artifact, model_path, compress=3)

    history_payload = {
        "config": vars(args),
        "n_trials": len(history),
        "best": asdict(best),
        "history": [asdict(h) for h in history],
        "model_path": str(model_path),
    }
    history_path.write_text(json.dumps(history_payload), encoding="utf-8")

    rank_df = pd.DataFrame([asdict(h) for h in history]).sort_values(
        by=["holdout_r2", "cv_r2_mean"],
        ascending=False,
    )
    rank_df.to_csv(leaderboard_path, index=False)

    print("\nBest trial summary")
    print(
        f"  trial={best.trial} holdout_R2={best.holdout_r2:.4f} "
        f"cv_R2={best.cv_r2_mean:.4f}+-{best.cv_r2_std:.4f} gap={best.generalization_gap:.4f}"
    )
    print(f"  model: {model_path}")
    print(f"  history: {history_path}")
    print(f"  leaderboard: {leaderboard_path}")

    if args.model == "rf":
        run_config = {
            "model": "rf",
            "run_id": f"adaptive_r2_{str(args.target_r2).replace('.', '_')}",
            "rf_n_estimators": best.params["rf_n_estimators"],
            "rf_min_samples_leaf": best.params["rf_min_samples_leaf"],
            "rf_max_depth": best.params["rf_max_depth"],
            "rf_max_features": best.params["rf_max_features"],
        }
        print("\nEquivalent --run-config for train_models_cli")
        print(json.dumps(run_config))

    return 0


def main() -> int:
    args = parse_args()
    return run_search(args)


if __name__ == "__main__":
    raise SystemExit(main())
