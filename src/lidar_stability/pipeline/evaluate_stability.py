"""Evaluate physics-layer stability index from an omega model artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lidar_stability.ml.feature_engineering import load_featured_data
from lidar_stability.physics import StabilityEngine


LEAKAGE_FEATURES = {
    "si",
    "si_mcu",
    "si_real",
    "si_pred",
    "si_pred_obs_w",
    "delta_si",
    "delta_si_static_fused",
    "delta_si_pred_obs_w",
    "si_dynamic_obs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final SI from a trained omega model and physics terms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-files", nargs="+", required=True, help="Featured CSV files to evaluate")
    parser.add_argument("--model-artifact", required=True, help="Joblib artifact or compact bundle path")
    parser.add_argument("--artifact-key", default=None, help="Artifact key inside compact bundle")
    parser.add_argument("--output-csv", required=True, help="Prediction CSV path")
    parser.add_argument("--metrics-json", required=True, help="Metrics JSON path")
    parser.add_argument("--vehicle-config", default=None, help="Optional vehicle YAML config")
    parser.add_argument("--si-column", default="si", help="Measured SI stability-margin column")
    parser.add_argument("--phi-lidar-column", default="phi_lidar", help="LiDAR cross-slope column in radians")
    parser.add_argument("--ay-column", default="ay", help="Lateral acceleration column")
    parser.add_argument("--ay-unit", choices=["m_s2", "cm_s2", "g"], default="m_s2")
    parser.add_argument("--design-ay-m-s2", type=float, default=None, help="Fallback ay if ay column is unavailable")
    parser.add_argument("--omega-target-column", default="gy", help="Measured omega target column for optional metrics")
    parser.add_argument("--omega-target-unit", choices=["deg_s", "rad_s"], default=None)
    parser.add_argument("--group-column", default=None, help="Explicit grouped evaluation column for final SI metrics")
    parser.add_argument("--benchmark-mode", action="store_true", help="Allow leakage-prone artifacts for benchmarks")
    return parser.parse_args()


def load_model_artifact(path: str | Path, artifact_key: str | None = None) -> dict[str, Any]:
    """Load a single model artifact or one entry from a compact bundle."""
    payload = joblib.load(path)
    if isinstance(payload, dict) and payload.get("format", "").startswith("bundle"):
        artifacts = payload.get("artifacts") or {}
        if not artifacts:
            raise ValueError("Model bundle has no artifacts")
        key = artifact_key or sorted(artifacts)[0]
        if key not in artifacts:
            raise KeyError(f"Artifact key '{key}' not found. Available: {sorted(artifacts)}")
        artifact = dict(artifacts[key])
        artifact.setdefault("bundle_path", str(path))
        artifact.setdefault("artifact_key", key)
        artifact.setdefault("bundle_training_context", payload.get("training_context", {}))
        return artifact
    if not isinstance(payload, dict):
        raise TypeError("Model artifact must be a dict payload")
    return payload


def leakage_columns(feature_columns: list[str]) -> list[str]:
    """Return leakage-prone feature names present in an artifact."""
    return sorted({col for col in feature_columns if col.lower() in LEAKAGE_FEATURES})


def _omega_to_rad_s(values: pd.Series | np.ndarray, unit: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if unit == "rad_s":
        return arr
    if unit == "deg_s":
        return np.radians(arr)
    raise ValueError(f"Unsupported omega unit: {unit}")


def _ay_to_m_s2(values: pd.Series | np.ndarray, unit: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if unit == "m_s2":
        return arr
    if unit == "cm_s2":
        return arr / 100.0
    if unit == "g":
        return arr * 9.80665
    raise ValueError(f"Unsupported ay unit: {unit}")


def _metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) == 0:
        return {"n": 0}
    block: dict[str, float | int] = {
        "n": int(len(y_t)),
        "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
        "mae": float(mean_absolute_error(y_t, y_p)),
        "residual_mean": float(np.mean(y_t - y_p)),
        "residual_std": float(np.std(y_t - y_p)),
    }
    block["r2"] = float(r2_score(y_t, y_p)) if len(y_t) >= 2 else float("nan")
    return block


def _risk_band_summary(si_true: np.ndarray, si_pred: np.ndarray) -> dict[str, dict[str, int]]:
    bands = {
        "danger": (0.0, 0.4),
        "warning": (0.4, 0.7),
        "safe": (0.7, 1.0000001),
    }
    summary: dict[str, dict[str, int]] = {}
    for label, (lo, hi) in bands.items():
        mask = (si_true >= lo) & (si_true < hi)
        summary[label] = {
            "n": int(mask.sum()),
            "predicted_danger": int(((si_pred < 0.4) & mask).sum()),
            "predicted_warning": int(((si_pred >= 0.4) & (si_pred < 0.7) & mask).sum()),
            "predicted_safe": int(((si_pred >= 0.7) & mask).sum()),
        }
    return summary


def _monotonicity_checks(
    engine: StabilityEngine,
    *,
    phi_lidar_rad: np.ndarray,
    omega_pred_rad_s: np.ndarray,
    ay_m_s2: np.ndarray,
) -> dict[str, int]:
    if len(phi_lidar_rad) == 0:
        return {"phi_violations": 0, "omega_violations": 0}
    phi = np.asarray(phi_lidar_rad, dtype=float)
    omega = np.asarray(omega_pred_rad_s, dtype=float)
    ay = np.asarray(ay_m_s2, dtype=float)
    base = engine.compute_terms(phi_lidar_rad=phi, omega_rad_s=omega, ay_m_s2=ay)["si_pred"]
    phi_more = engine.compute_terms(phi_lidar_rad=np.abs(phi) * 1.05, omega_rad_s=omega, ay_m_s2=ay)["si_pred"]
    omega_more = engine.compute_terms(phi_lidar_rad=phi, omega_rad_s=np.abs(omega) * 1.05, ay_m_s2=ay)["si_pred"]
    return {
        "phi_violations": int(np.sum(np.asarray(phi_more) > np.asarray(base) + 1e-12)),
        "omega_violations": int(np.sum(np.asarray(omega_more) > np.asarray(base) + 1e-12)),
    }


def _normalize_omega_unit(unit: str) -> str:
    if unit in {"deg/s", "degrees_per_second"}:
        return "deg_s"
    if unit in {"rad/s", "radians_per_second"}:
        return "rad_s"
    return unit


def _prediction_unit_from_artifact(artifact: dict[str, Any]) -> str:
    unit = artifact.get("prediction_unit") or artifact.get("target_unit") or "deg_s"
    return _normalize_omega_unit(str(unit))


def _target_unit_from_artifact(artifact: dict[str, Any], cli_unit: str | None) -> str:
    if cli_unit:
        return cli_unit
    unit = artifact.get("prediction_unit") or artifact.get("target_unit") or "deg_s"
    return _normalize_omega_unit(str(unit))


def _artifact_has_grouped_split(artifact: dict[str, Any]) -> bool:
    split_metadata = artifact.get("split_metadata") or artifact.get("split_policy") or {}
    if split_metadata.get("kind") == "grouped":
        return True
    if split_metadata.get("group_by") not in {None, "row"}:
        return True
    training_context = artifact.get("training_context") or {}
    context_split = training_context.get("split_policy") or {}
    if context_split.get("kind") == "grouped":
        return True
    return training_context.get("cv_group_by") not in {None, "row"}


def evaluate_dataframe(
    df: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    engine: StabilityEngine,
    si_column: str = "si",
    phi_lidar_column: str = "phi_lidar",
    ay_column: str = "ay",
    ay_unit: str = "m_s2",
    design_ay_m_s2: float | None = None,
    omega_target_column: str = "gy",
    omega_target_unit: str | None = None,
    group_column: str | None = None,
    benchmark_mode: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return prediction dataframe and metrics payload."""
    if df.empty:
        raise ValueError("Input dataframe is empty")

    model = artifact.get("model")
    if model is None:
        raise ValueError("Artifact is missing 'model'")
    feature_columns = list(artifact.get("feature_columns") or [])
    if not feature_columns:
        raise ValueError("Artifact is missing feature_columns")

    leaks = leakage_columns(feature_columns)
    if leaks and not benchmark_mode:
        raise ValueError(f"Artifact feature_columns contain leakage-prone columns: {leaks}")

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe is missing model feature columns: {missing}")
    if phi_lidar_column not in df.columns:
        raise KeyError(f"Input dataframe is missing '{phi_lidar_column}'")

    training_context = artifact.get("training_context") or {}
    split_metadata = artifact.get("split_metadata") or artifact.get("split_policy") or {}
    grouped_artifact = _artifact_has_grouped_split(artifact)
    grouped_evaluator = group_column is not None and group_column in df.columns
    if si_column in df.columns and not (grouped_artifact or benchmark_mode):
        raise ValueError(
            "Final SI metrics require grouped artifact split metadata. "
            "Use --benchmark-mode only for development baselines."
        )

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    valid_mask = X.notna().all(axis=1)
    if not valid_mask.any():
        raise ValueError("No valid rows after numeric feature conversion")

    result = df.loc[valid_mask].copy()
    X_valid = X.loc[valid_mask]

    pred_raw = np.asarray(model.predict(X_valid), dtype=float)
    pred_unit = _prediction_unit_from_artifact(artifact)
    omega_pred = _omega_to_rad_s(pred_raw, pred_unit)

    if ay_column in result.columns:
        ay_m_s2 = _ay_to_m_s2(pd.to_numeric(result[ay_column], errors="coerce").fillna(0.0), ay_unit)
        ay_source = ay_column
    elif design_ay_m_s2 is not None:
        ay_m_s2 = np.full(len(result), float(design_ay_m_s2), dtype=float)
        ay_source = "design_ay_m_s2"
    else:
        raise KeyError(f"Input dataframe is missing '{ay_column}' and no design ay fallback was provided")

    phi = pd.to_numeric(result[phi_lidar_column], errors="coerce").to_numpy(dtype=float)
    terms = engine.compute_terms(phi_lidar_rad=phi, omega_rad_s=omega_pred, ay_m_s2=ay_m_s2)

    result["omega_pred_raw"] = pred_raw
    result["omega_pred_unit"] = pred_unit
    result["omega_pred_rad_s"] = omega_pred
    result["ay_m_s2"] = ay_m_s2
    result["phi_crit_rad"] = terms["phi_crit_rad"]
    result["omega_crit_rad_s"] = terms["omega_crit_rad_s"]
    result["si_static_lidar_risk"] = terms["si_static_lidar_risk"]
    result["si_static_lidar"] = result["si_static_lidar_risk"]
    result["si_static_lidar_margin"] = engine.final_si_from_terms(result["si_static_lidar_risk"].to_numpy(dtype=float))
    result["si_dynamic_omega_risk"] = terms["si_dynamic_omega_risk"]
    result["si_risk_total"] = terms["si_risk_total"]
    result["si_pred"] = terms["si_pred"]

    metrics: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_input_rows": int(len(df)),
        "n_evaluated_rows": int(len(result)),
        "artifact": {
            "model_key": artifact.get("model_key"),
            "run_id": artifact.get("run_id"),
            "artifact_key": artifact.get("artifact_key"),
            "feature_columns": feature_columns,
            "target_name": artifact.get("target_name", "gy"),
            "target_unit": artifact.get("target_unit", "deg_s"),
            "prediction_unit": artifact.get("prediction_unit", pred_unit),
            "split_metadata": split_metadata,
            "training_context": training_context,
            "leakage_columns": leaks,
            "benchmark_mode": bool(benchmark_mode),
        },
        "physics": {
            "si_semantics": "stability_margin_1_is_maximum",
            "phi_lidar_column": phi_lidar_column,
            "phi_lidar_unit": "rad",
            "ay_source": ay_source,
            "ay_unit_input": ay_unit,
            "track_width_m": engine.get_vehicle_params()["track_width_m"],
            "cg_height_m": engine.get_vehicle_params()["cg_height_m"],
            "phi_crit_rad": float(engine.critical_angle(degrees=False)),
            "omega_crit_rad_s_mean": float(np.mean(result["omega_crit_rad_s"])),
            "omega_crit_rad_s_min": float(np.min(result["omega_crit_rad_s"])),
            "omega_crit_rad_s_max": float(np.max(result["omega_crit_rad_s"])),
        },
        "monotonicity": _monotonicity_checks(
            engine,
            phi_lidar_rad=phi,
            omega_pred_rad_s=omega_pred,
            ay_m_s2=ay_m_s2,
        ),
    }

    if omega_target_column in result.columns:
        true_unit = _target_unit_from_artifact(artifact, omega_target_unit)
        omega_true = _omega_to_rad_s(
            pd.to_numeric(result[omega_target_column], errors="coerce"),
            true_unit,
        )
        result["omega_true_rad_s"] = omega_true
        result["omega_true_unit"] = true_unit
        result["omega_error_rad_s"] = result["omega_true_rad_s"] - result["omega_pred_rad_s"]
        metrics["omega"] = _metric_block(omega_true, result["omega_pred_rad_s"].to_numpy(dtype=float))

    if si_column in result.columns:
        si_true = pd.to_numeric(result[si_column], errors="coerce").to_numpy(dtype=float)
        result["si_true"] = si_true
        result["si_error"] = result["si_true"] - result["si_pred"]
        metrics["static_only_si"] = _metric_block(si_true, result["si_static_lidar_margin"].to_numpy(dtype=float))
        metrics["final_si"] = _metric_block(si_true, result["si_pred"].to_numpy(dtype=float))
        metrics["final_si"]["grouped_evaluation"] = bool(grouped_artifact)
        if grouped_evaluator:
            metrics["final_si"]["group_column"] = group_column
            metrics["final_si"]["n_groups"] = int(result[group_column].astype(str).nunique())
        metrics["si_band_confusion"] = _risk_band_summary(si_true, result["si_pred"].to_numpy(dtype=float))

    return result, metrics


def main() -> int:
    args = parse_args()
    df = load_featured_data(args.input_files)
    artifact = load_model_artifact(args.model_artifact, args.artifact_key)
    engine = StabilityEngine(args.vehicle_config)
    predictions, metrics = evaluate_dataframe(
        df,
        artifact,
        engine=engine,
        si_column=args.si_column,
        phi_lidar_column=args.phi_lidar_column,
        ay_column=args.ay_column,
        ay_unit=args.ay_unit,
        design_ay_m_s2=args.design_ay_m_s2,
        omega_target_column=args.omega_target_column,
        omega_target_unit=args.omega_target_unit,
        group_column=args.group_column,
        benchmark_mode=args.benchmark_mode,
    )
    output_csv = Path(args.output_csv)
    metrics_json = Path(args.metrics_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Predictions written: {output_csv}")
    print(f"Metrics written: {metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
