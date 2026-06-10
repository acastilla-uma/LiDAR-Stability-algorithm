"""Ground-truth builders for static and dynamic stability datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def build_ground_truth(df: pd.DataFrame, engine) -> pd.DataFrame:
    """Build baseline static ground truth from IMU roll and measured SI.

    ``si_static`` is kept as a legacy risk/utilization column. New consumers
    should prefer explicit ``*_risk`` columns and final SI margin columns.
    """
    if df.empty:
        return df.copy()

    result = df.copy()
    roll_deg = _numeric(result, "roll_deg")
    si_real = _numeric(result, "si_mcu", default=np.nan)
    if si_real.isna().all():
        si_real = _numeric(result, "si")
    si_real = si_real.fillna(0.0)

    result["si_real"] = si_real.astype(float)
    result["si_static_risk"] = engine.si_static_batch(np.radians(roll_deg.to_numpy(dtype=float)))
    result["si_static_margin"] = engine.final_si_from_terms(result["si_static_risk"].to_numpy(dtype=float))
    result["si_static"] = result["si_static_risk"]
    result["delta_si"] = result["si_real"] - result["si_static"]
    return result


def build_enhanced_ground_truth(df: pd.DataFrame, engine) -> pd.DataFrame:
    """Build static + observed dynamic ground-truth columns for featured data."""
    if df.empty:
        return df.copy()

    result = df.copy()
    roll = _numeric(result, "roll")
    phi_lidar = _numeric(result, "phi_lidar")
    si_real = _numeric(result, "si")
    gy = _numeric(result, "gy")

    result["si_real"] = si_real.astype(float)
    result["si_static_imu_risk"] = engine.si_static_batch(np.radians(roll.to_numpy(dtype=float)))
    result["si_static_lidar_risk"] = engine.si_static_batch(phi_lidar.to_numpy(dtype=float))
    result["si_static_fused_risk"] = (
        result["si_static_imu_risk"].to_numpy(dtype=float) + result["si_static_lidar_risk"].to_numpy(dtype=float)
    ) / 2.0
    result["si_static_imu_margin"] = engine.final_si_from_terms(result["si_static_imu_risk"].to_numpy(dtype=float))
    result["si_static_lidar_margin"] = engine.final_si_from_terms(
        result["si_static_lidar_risk"].to_numpy(dtype=float)
    )
    result["si_static_fused_margin"] = engine.final_si_from_terms(
        result["si_static_fused_risk"].to_numpy(dtype=float)
    )
    result["omega_rad_s"] = np.radians(gy.to_numpy(dtype=float))
    result["si_dynamic_obs"] = result["si_real"] - result["si_static_fused_margin"]
    result["si_pred_obs_w"] = result["si_static_fused_margin"] + result["si_dynamic_obs"]
    result["delta_si_static_fused"] = result["si_real"] - result["si_static_fused_margin"]
    result["delta_si_pred_obs_w"] = result["si_real"] - result["si_pred_obs_w"]
    result["si_static_imu"] = result["si_static_imu_risk"]
    result["si_static_lidar"] = result["si_static_lidar_risk"]
    result["si_static_fused"] = result["si_static_fused_risk"]
    return result


def export_ground_truth(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a ground-truth dataframe to CSV and return the path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
