import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from lidar_stability.physics import StabilityEngine
from lidar_stability.pipeline.evaluate_stability import evaluate_dataframe


class ConstantOmegaModel:
    def __init__(self, value: float = 12.0) -> None:
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def _featured_df(n: int = 24) -> pd.DataFrame:
    idx = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "source_file": ["route-a.csv"] * (n // 2) + ["route-b.csv"] * (n - n // 2),
            "roll": np.sin(idx / 5.0) * 3.0,
            "pitch": np.cos(idx / 6.0) * 2.0,
            "ax": 0.1 + idx * 0.0,
            "ay": 2.0 + np.abs(np.sin(idx / 4.0)),
            "az": 9.8 + idx * 0.0,
            "speed_kmh": 20.0 + idx * 0.1,
            "phi_lidar": np.radians(1.0 + idx * 0.02),
            "tri": 0.2 + idx * 0.001,
            "ruggedness": 0.3 + idx * 0.001,
            "gy": 10.0 + idx * 0.1,
            "si": np.clip(0.95 - idx * 0.002, 0.0, 1.0),
        }
    )


def _artifact(feature_columns=None) -> dict:
    return {
        "model": ConstantOmegaModel(),
        "feature_columns": feature_columns
        or ["roll", "pitch", "ax", "ay", "az", "speed_kmh", "phi_lidar", "tri", "ruggedness"],
        "target_name": "gy",
        "target_unit": "deg_s",
        "prediction_unit": "deg_s",
        "model_key": "dummy",
        "run_id": "constant",
        "training_context": {"cv_group_by": "source_file"},
    }


def test_physics_risk_and_si_margin_semantics():
    engine = StabilityEngine()
    phi_crit = engine.critical_angle(degrees=False)

    assert np.isclose(engine.static_lidar_risk(phi_crit), 1.0)

    omega_crit = engine.omega_critical(ay_m_s2=2.0)
    low_dynamic = engine.dynamic_omega_risk(omega_crit * 0.25, omega_crit)
    high_dynamic = engine.dynamic_omega_risk(omega_crit * 0.75, omega_crit)
    assert high_dynamic > low_dynamic

    stable = engine.final_si_from_terms(static_risk=0.1, dynamic_risk=0.1)
    less_stable = engine.final_si_from_terms(static_risk=0.2, dynamic_risk=0.2)
    assert stable > less_stable
    assert stable <= 1.0


def test_evaluator_computes_static_dynamic_and_final_si():
    predictions, metrics = evaluate_dataframe(
        _featured_df(),
        _artifact(),
        engine=StabilityEngine(),
        group_column="source_file",
    )

    expected_columns = {
        "omega_pred_rad_s",
        "omega_true_rad_s",
        "si_static_lidar_risk",
        "si_static_lidar_margin",
        "si_dynamic_omega_risk",
        "si_risk_total",
        "si_pred",
        "si_true",
        "si_error",
    }
    assert expected_columns.issubset(predictions.columns)
    assert predictions["si_pred"].between(0.0, 1.0).all()
    assert set(["omega", "static_only_si", "final_si"]).issubset(metrics)
    assert metrics["physics"]["si_semantics"] == "stability_margin_1_is_maximum"
    assert metrics["final_si"]["grouped_evaluation"] is True


def test_evaluator_rejects_leakage_feature_columns():
    artifact = _artifact(feature_columns=["roll", "si"])
    with pytest.raises(ValueError, match="leakage-prone"):
        evaluate_dataframe(
            _featured_df(),
            artifact,
            engine=StabilityEngine(),
            group_column="source_file",
        )


def test_evaluator_requires_grouped_final_si_metrics_without_benchmark():
    artifact = _artifact()
    artifact["training_context"] = {"cv_group_by": "row"}
    with pytest.raises(ValueError, match="Final SI metrics require grouped"):
        evaluate_dataframe(_featured_df(), artifact, engine=StabilityEngine(), group_column="source_file")


def test_evaluator_accepts_top_level_grouped_split_metadata():
    artifact = _artifact()
    artifact["training_context"] = {}
    artifact["split_metadata"] = {"kind": "grouped", "group_by": "source_file", "n_splits": 3}

    _, metrics = evaluate_dataframe(_featured_df(), artifact, engine=StabilityEngine())

    assert metrics["final_si"]["grouped_evaluation"] is True


def test_omega_target_unit_override_does_not_change_prediction_unit():
    df = _featured_df()
    df["gy"] = np.pi
    artifact = _artifact()
    artifact["model"] = ConstantOmegaModel(180.0)
    artifact["prediction_unit"] = "deg_s"

    predictions, _ = evaluate_dataframe(
        df,
        artifact,
        engine=StabilityEngine(),
        omega_target_unit="rad_s",
        group_column="source_file",
    )

    assert np.isclose(predictions["omega_pred_rad_s"].iloc[0], np.pi)
    assert np.isclose(predictions["omega_true_rad_s"].iloc[0], np.pi)


def test_evaluate_stability_cli_writes_prediction_and_metrics(tmp_path: Path):
    data_path = tmp_path / "featured.csv"
    artifact_path = tmp_path / "model.joblib"
    output_csv = tmp_path / "predictions.csv"
    metrics_json = tmp_path / "metrics.json"

    _featured_df().to_csv(data_path, index=False)
    joblib.dump(_artifact(), artifact_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lidar_stability.pipeline.evaluate_stability",
            "--input-files",
            str(data_path),
            "--model-artifact",
            str(artifact_path),
            "--output-csv",
            str(output_csv),
            "--metrics-json",
            str(metrics_json),
            "--group-column",
            "source_file",
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
    )

    assert "Predictions written" in result.stdout
    predictions = pd.read_csv(output_csv)
    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert "si_pred" in predictions.columns
    assert "final_si" in metrics
