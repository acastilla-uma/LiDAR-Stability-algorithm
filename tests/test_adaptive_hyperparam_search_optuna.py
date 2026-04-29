from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from lidar_stability.ml import adaptive_hyperparam_search as ahs
from lidar_stability.ml import plot_models_leaderboard as pml
from lidar_stability.ml import train_models_cli as tmc


def _make_args(**overrides) -> argparse.Namespace:
    base = {
        "input_glob": ["Doback-Data/featured/DOBACK*.csv"],
        "contains": None,
        "max_files": 0,
        "model": "rf",
        "target_r2": 2.0,
        "max_trials": 1,
        "patience": 10,
        "n_splits": 3,
        "n_jobs": 1,
        "holdout_frac": 0.2,
        "max_generalization_gap": 0.5,
        "max_r2_std": 1.0,
        "max_rows": 0,
        "max_rows_per_source": 0,
        "target_column": "gy",
        "feature_columns": None,
        "cv_group_by": "source_file",
        "random_state": 42,
        "output_dir": "output/models",
        "prefix": "adaptive_test",
        "sampler": "random",
        "pruner": "none",
        "startup_trials": 1,
        "study_name": None,
        "study_storage": None,
        "resume": True,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_featured_inputs(repo_root: Path, n_files: int = 6, rows_per_file: int = 120) -> None:
    featured_dir = repo_root / "Doback-Data" / "featured"
    featured_dir.mkdir(parents=True, exist_ok=True)

    for file_idx in range(n_files):
        start = file_idx * rows_per_file
        values = np.arange(start, start + rows_per_file, dtype=float)
        frame = pd.DataFrame(
            {
                "row_value": values,
                "dummy_feature": values * 0.1,
            }
        )
        frame.to_csv(featured_dir / f"DOBACK024_202510{file_idx + 1:02d}.csv", index=False)


def _fake_build_w_training_dataset(
    raw_df: pd.DataFrame,
    feature_columns=None,
    target_column: str = "gy",
):
    work = raw_df.copy().reset_index(drop=True)
    feature = pd.to_numeric(work["row_value"], errors="coerce").fillna(0.0)
    X = pd.DataFrame({"feature": feature}, index=work.index)
    y = feature * 0.5 + 3.0
    clean_df = work.copy()
    return X, y, ["feature"], clean_df


@pytest.mark.skipif(ahs.optuna is None, reason="optuna is not installed")
def test_suggest_params_build_model_for_all_model_keys():
    expected_keys = {
        "rf": {"rf_n_estimators", "rf_min_samples_leaf", "rf_max_depth", "rf_max_features", "rf_min_samples_split"},
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
            "gbr_min_samples_leaf",
            "gbr_min_samples_split",
            "gbr_subsample",
        },
    }

    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 20)})
    y = pd.Series(np.linspace(0.0, 1.0, 20))

    for model_key, keys in expected_keys.items():
        study = ahs.optuna.create_study(direction="maximize")
        trial = study.ask()
        params = ahs.suggest_params(trial, model_key)
        assert set(params) == keys

        model = ahs.build_model(model_key, params=params, random_state=123)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        study.tell(trial, 0.0)


def test_load_trial_histories_splits_completed_and_pruned_trials():
    completed = ahs.TrialResult(
        trial=1,
        optuna_trial_number=0,
        trial_state="COMPLETE",
        model="rf",
        params={"rf_n_estimators": 200},
        objective_value=0.8,
        cv_r2_mean=0.8,
        cv_r2_std=0.1,
        cv_rmse_mean=1.0,
        cv_mae_mean=0.5,
        holdout_r2=0.75,
        holdout_rmse=1.1,
        holdout_mae=0.6,
        generalization_gap=0.05,
        is_feasible=True,
    )
    pruned = ahs.TrialResult(
        trial=2,
        optuna_trial_number=1,
        trial_state="PRUNED",
        model="rf",
        params={"rf_n_estimators": 300},
        objective_value=0.4,
        cv_r2_mean=0.4,
        cv_r2_std=0.0,
        cv_rmse_mean=2.0,
        cv_mae_mean=1.0,
        holdout_r2=None,
        holdout_rmse=None,
        holdout_mae=None,
        generalization_gap=None,
        is_feasible=False,
    )

    class FrozenTrial:
        def __init__(self, number: int, payload: dict):
            self.number = number
            self.user_attrs = {"trial_result": payload}

    class FakeStudy:
        def __init__(self, trials):
            self.trials = trials

    study = FakeStudy(
        [
            FrozenTrial(0, ahs.serialize_trial_result(completed)),
            FrozenTrial(1, ahs.serialize_trial_result(pruned)),
        ]
    )

    history, pruned_history = ahs.load_trial_histories(study)
    assert [item.trial_state for item in history] == ["COMPLETE"]
    assert [item.trial_state for item in pruned_history] == ["PRUNED"]


def test_train_models_cli_accepts_rf_min_samples_split_in_run_config():
    args = argparse.Namespace(
        run_config=[
            json.dumps(
                {
                    "model": "rf",
                    "run_id": "rf_optuna_best",
                    "rf_n_estimators": 350,
                    "rf_min_samples_leaf": 3,
                    "rf_max_depth": 12,
                    "rf_max_features": "sqrt",
                    "rf_min_samples_split": 9,
                }
            )
        ]
    )
    base_hyperparams = {
        "rf_n_estimators": 400,
        "rf_min_samples_leaf": 2,
        "rf_max_depth": None,
        "rf_max_features": "sqrt",
        "rf_min_samples_split": 2,
    }

    runs = tmc.build_training_runs(args, base_hyperparams)
    assert runs[0]["hyperparams"]["rf_min_samples_split"] == 9


def test_train_models_cli_build_model_respects_n_jobs_for_tree_models():
    rf_model = tmc.build_model("rf", random_state=42, hyperparams={}, n_jobs=7)
    extra_trees_model = tmc.build_model("extra_trees", random_state=42, hyperparams={}, n_jobs=5)
    gbr_model = tmc.build_model("gbr", random_state=42, hyperparams={}, n_jobs=3)

    assert rf_model.n_jobs == 7
    assert extra_trees_model.n_jobs == 5
    assert not hasattr(gbr_model, "n_jobs")


def test_train_models_cli_build_cv_splits_groups_source_files_without_leakage():
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    groups = pd.Series(
        ["file_a.csv"] * 3 +
        ["file_b.csv"] * 3 +
        ["file_c.csv"] * 3 +
        ["file_d.csv"] * 3
    )

    splits = tmc.build_cv_splits(
        X,
        n_splits=4,
        random_state=42,
        groups=groups,
        cv_group_by="source_file",
    )

    for train_idx, test_idx in splits:
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_adaptive_build_cv_splits_groups_source_files_without_leakage():
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(np.arange(12, dtype=float))
    groups = pd.Series(
        ["file_a.csv"] * 3 +
        ["file_b.csv"] * 3 +
        ["file_c.csv"] * 3 +
        ["file_d.csv"] * 3
    )

    splits = ahs.build_cv_splits(
        X_train=X,
        y_train=y,
        train_groups=groups,
        n_splits=4,
        random_state=42,
        cv_group_by="source_file",
    )

    for train_idx, test_idx in splits:
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_grouped_cv_requires_enough_source_files():
    X = pd.DataFrame({"x": np.arange(6, dtype=float)})
    groups = pd.Series(["file_a.csv"] * 3 + ["file_b.csv"] * 3)

    with pytest.raises(ValueError, match="requires at least 3 unique groups"):
        tmc.build_cv_splits(
            X,
            n_splits=3,
            random_state=42,
            groups=groups,
            cv_group_by="source_file",
        )


def test_train_models_cli_smoke_records_explicit_device_constants_in_artifact(tmp_path, monkeypatch):
    repo_root = tmp_path
    featured_dir = repo_root / "Doback-Data" / "featured"
    featured_dir.mkdir(parents=True, exist_ok=True)

    device_constants = {
        "DOBACK023_20251012.csv": {"k1": 1.15, "k2": 2.05, "k4_mm": 1100, "d1_m": 2.03, "coeff": 1.89, "s_mm": 2450, "alphav": 54},
        "DOBACK024_20251012.csv": {"k1": 1.15, "k2": 2.0, "k4_mm": 2500, "d1_m": 1.88, "coeff": 5.0, "s_mm": 2215, "alphav": 64},
        "DOBACK027_20251020.csv": {"k1": 1.50, "k2": 1.50, "k4_mm": 1100, "d1_m": 0.96, "coeff": 2.13, "s_mm": 1880, "alphav": 45},
        "DOBACK028_20251008.csv": {"k1": 1.15, "k2": 2.05, "k4_mm": 1100, "d1_m": 1.96, "coeff": 1.86, "s_mm": 2200, "alphav": 57},
    }
    requested_features = [
        "roll",
        "pitch",
        "ax",
        "ay",
        "az",
        "speed_kmh",
        "phi_lidar",
        "tri",
        "ruggedness",
        "k1",
        "k2",
        "k4_mm",
        "d1_m",
        "coeff",
        "s_mm",
        "alphav",
    ]

    for file_idx, (filename, constants) in enumerate(device_constants.items(), start=1):
        values = np.arange(60, dtype=float) + (file_idx * 100.0)
        frame = pd.DataFrame(
            {
                "roll": np.sin(values / 20.0),
                "pitch": np.cos(values / 25.0),
                "ax": values * 0.3,
                "ay": values * 0.2,
                "az": 980.0 + values * 0.1,
                "speed_kmh": 25.0 + values * 0.05,
                "phi_lidar": 0.01 + values / 10000.0,
                "tri": 0.2 + values / 1000.0,
                "ruggedness": 0.3 + values / 1200.0,
                "gy": 15.0 + values * 0.4,
                **constants,
            }
        )
        frame.to_csv(featured_dir / filename, index=False)

    monkeypatch.setattr(tmc, "find_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_models_cli.py",
            "--input-glob",
            "Doback-Data/featured/DOBACK*.csv",
            "--models",
            "rf",
            "--n-splits",
            "2",
            "--n-jobs",
            "1",
            "--rf-n-estimators",
            "10",
            "--cv-group-by",
            "source_file",
            "--feature-columns",
            *requested_features,
            "--output-dir",
            "output/models",
            "--prefix",
            "smoke_constants",
        ],
    )

    assert tmc.main() == 0

    metrics_path = repo_root / "output" / "models" / "smoke_constants_metrics.json"
    bundle_path = repo_root / "output" / "models" / "smoke_constants_models.joblib"
    assert metrics_path.exists()
    assert bundle_path.exists()

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_payload = metrics_payload["runs"][0]
    assert run_payload["cv_group_by"] == "source_file"
    assert run_payload["feature_columns"] == requested_features

    bundle_payload = joblib.load(bundle_path)
    artifact = bundle_payload["artifacts"]["rf"]
    assert artifact["feature_columns"] == requested_features
    assert artifact["training_context"]["cv_group_by"] == "source_file"


@pytest.mark.skipif(ahs.optuna is None, reason="optuna is not installed")
def test_run_search_persists_resumes_and_exports_compatible_history(tmp_path, monkeypatch):
    repo_root = tmp_path
    _write_featured_inputs(repo_root)

    monkeypatch.setattr(ahs, "find_repo_root", lambda: repo_root)
    monkeypatch.setattr(ahs, "build_w_training_dataset", _fake_build_w_training_dataset)

    args_first = _make_args(max_trials=1)
    assert ahs.run_search(args_first) == 0

    output_dir = repo_root / "output" / "models"
    history_path = output_dir / "adaptive_test_rf_history.json"
    study_path = output_dir / "adaptive_test_rf_study.sqlite3"
    model_path = output_dir / "adaptive_test_rf_best.joblib"

    assert history_path.exists()
    assert study_path.exists()
    assert model_path.exists()

    first_payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert first_payload["search_engine"] == "optuna"
    assert first_payload["n_trials"] == 1
    assert first_payload["n_completed_trials"] == 1
    assert first_payload["n_pruned_trials"] == 0
    assert first_payload["history"][0]["trial_state"] == "COMPLETE"
    assert "optuna_trial_number" in first_payload["history"][0]

    artifact = joblib.load(model_path)
    assert artifact["n_samples"] == first_payload["dataset"]["n_train_samples"]
    assert artifact["n_train_samples"] < first_payload["dataset"]["n_samples"]

    ranked = pml.load_metrics(output_dir, enrich_from_bundle=False)
    assert not ranked.empty
    assert {"model", "rmse_mean", "mae_mean", "r2_mean"}.issubset(ranked.columns)

    args_second = _make_args(max_trials=2)
    assert ahs.run_search(args_second) == 0

    second_payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert second_payload["n_trials"] == 2
    assert second_payload["n_completed_trials"] == 2
    assert second_payload["study_storage"].endswith("adaptive_test_rf_study.sqlite3")
