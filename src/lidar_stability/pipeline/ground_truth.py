"""Ground truth builders for static and sprint-5 enhanced stability datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _resolve_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise KeyError(f"Missing required columns. Tried: {candidates}")
    return None


def _to_numeric_series(df: pd.DataFrame, column: str | None, fill_value: float = 0.0) -> pd.Series:
    if column is None:
        return pd.Series(np.full(len(df), fill_value, dtype=float), index=df.index)
    return pd.to_numeric(df[column], errors='coerce').fillna(fill_value)


def _timestamp_to_elapsed_us(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors='coerce')
    valid = dt.dropna()
    if valid.empty:
        return pd.Series(np.full(len(series), np.nan, dtype=float), index=series.index)

    ref = valid.iloc[0]
    return (dt - ref).dt.total_seconds() * 1e6


def _extract_roll_pitch_si(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    roll_col = _resolve_column(df, ['roll_deg', 'roll'])
    pitch_col = _resolve_column(df, ['pitch_deg', 'pitch'], required=False)
    si_col = _resolve_column(df, ['si_mcu', 'si_real', 'si'])

    roll_deg = _to_numeric_series(df, roll_col)
    pitch_deg = _to_numeric_series(df, pitch_col)
    si_real = _to_numeric_series(df, si_col)
    return roll_deg, pitch_deg, si_real


def _extract_omega_rad_s(df: pd.DataFrame, omega_column: str | None = None) -> pd.Series:
    rad_candidates = ['omega_rad_s', 'w_rad_s', 'gy_rad_s']
    mdeg_candidates = ['gy', 'gz', 'gy_mdeg_s']

    if omega_column:
        omega_raw = _to_numeric_series(df, omega_column)
        if omega_column in mdeg_candidates:
            return np.radians(omega_raw / 1000.0)
        return omega_raw

    found_rad = _resolve_column(df, rad_candidates, required=False)
    if found_rad is not None:
        return _to_numeric_series(df, found_rad)

    found_mdeg = _resolve_column(df, mdeg_candidates, required=False)
    if found_mdeg is None:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)

    omega_mdeg_s = _to_numeric_series(df, found_mdeg)
    return np.radians(omega_mdeg_s / 1000.0)


def build_ground_truth(imu_df: pd.DataFrame, engine) -> pd.DataFrame:
    """Backward-compatible ground truth builder with SI_static and delta_si."""
    if imu_df.empty:
        logger.warning("Empty IMU DataFrame provided")
        return pd.DataFrame()

    roll_deg, pitch_deg, si_real = _extract_roll_pitch_si(imu_df)

    gt_df = pd.DataFrame(index=imu_df.index)
    if 't_us' in imu_df.columns:
        gt_df['t_us'] = pd.to_numeric(imu_df['t_us'], errors='coerce')
    elif 'timestamp' in imu_df.columns:
        gt_df['t_us'] = _timestamp_to_elapsed_us(imu_df['timestamp'])
    elif 'timeantwifi' in imu_df.columns:
        logger.warning("Using raw timeantwifi as fallback for t_us; prefer timestamp-based timing.")
        gt_df['t_us'] = pd.to_numeric(imu_df['timeantwifi'], errors='coerce')

    if 'timestamp' in imu_df.columns:
        gt_df['timestamp'] = imu_df['timestamp']

    gt_df['roll_deg'] = roll_deg
    gt_df['pitch_deg'] = pitch_deg
    gt_df['si_real'] = si_real

    roll_rad = np.radians(roll_deg.values)
    gt_df['si_static'] = engine.si_static_batch(roll_rad)
    gt_df['delta_si'] = gt_df['si_real'] - gt_df['si_static']

    _validate_ground_truth(gt_df)

    logger.info(f"Built ground truth dataset with {len(gt_df)} samples")
    logger.info(f"  SI_real range: [{gt_df['si_real'].min():.3f}, {gt_df['si_real'].max():.3f}]")
    logger.info(f"  SI_static range: [{gt_df['si_static'].min():.3f}, {gt_df['si_static'].max():.3f}]")
    logger.info(f"  delta_si mean: {gt_df['delta_si'].mean():.3f}, std: {gt_df['delta_si'].std():.3f}")

    return gt_df


def build_enhanced_ground_truth(
    data_df: pd.DataFrame,
    engine,
    static_fusion_alpha: float | None = None,
    tri_gain: float | None = None,
    tri_clip: float | None = None,
    omega_column: str | None = None,
    lidar_phi_column: str | None = None,
    tri_column: str | None = None,
) -> pd.DataFrame:
    """Build sprint-5 enriched ground truth with static and dynamic SI components."""
    if data_df.empty:
        logger.warning("Empty DataFrame provided to build_enhanced_ground_truth")
        return pd.DataFrame()

    roll_deg, pitch_deg, si_real = _extract_roll_pitch_si(data_df)
    omega_rad_s = _extract_omega_rad_s(data_df, omega_column=omega_column)

    phi_col = lidar_phi_column
    if phi_col is None:
        phi_col = _resolve_column(data_df, ['phi_lidar', 'phi_lidar_rad'], required=False)

    tri_col = tri_column
    if tri_col is None:
        tri_col = _resolve_column(data_df, ['tri', 'ruggedness'], required=False)

    phi_lidar_rad = _to_numeric_series(data_df, phi_col)
    tri_values = _to_numeric_series(data_df, tri_col)

    roll_rad = np.radians(roll_deg.values)
    phi_arr = phi_lidar_rad.values
    tri_arr = tri_values.values
    omega_arr = omega_rad_s.values

    gt_df = pd.DataFrame(index=data_df.index)
    if 'timestamp' in data_df.columns:
        gt_df['timestamp'] = data_df['timestamp']
    if 't_us' in data_df.columns:
        gt_df['t_us'] = pd.to_numeric(data_df['t_us'], errors='coerce')
    elif 'timestamp' in data_df.columns:
        gt_df['t_us'] = _timestamp_to_elapsed_us(data_df['timestamp'])
    elif 'timeantwifi' in data_df.columns:
        logger.warning("Using raw timeantwifi as fallback for t_us; prefer timestamp-based timing.")
        gt_df['t_us'] = pd.to_numeric(data_df['timeantwifi'], errors='coerce')

    gt_df['roll_deg'] = roll_deg
    gt_df['pitch_deg'] = pitch_deg
    gt_df['phi_lidar_deg'] = np.degrees(phi_arr)
    gt_df['tri'] = tri_values
    gt_df['omega_rad_s'] = omega_rad_s
    gt_df['si_real'] = si_real

    gt_df['si_static_imu'] = engine.si_static_batch(roll_rad)
    gt_df['si_static_lidar'] = engine.si_static_batch(phi_arr)

    effective_roll = engine.effective_roll_from_lidar_batch(
        roll_rad_array=roll_rad,
        phi_lidar_rad_array=phi_arr,
        tri_array=tri_arr,
        static_fusion_alpha=static_fusion_alpha,
        tri_gain=tri_gain,
        tri_clip=tri_clip,
    )
    gt_df['roll_eff_deg'] = np.degrees(effective_roll)
    gt_df['si_static_fused'] = engine.si_static_batch(effective_roll)

    gt_df['si_dynamic_obs'] = 1.0 - engine.dynamic_penalty_from_w_batch(omega_arr)
    gt_df['si_pred_obs_w'] = engine.si_combined_batch(
        roll_rad_array=roll_rad,
        omega_rad_s_array=omega_arr,
        phi_lidar_rad_array=phi_arr,
        tri_array=tri_arr,
        static_fusion_alpha=static_fusion_alpha,
        tri_gain=tri_gain,
        tri_clip=tri_clip,
    )

    gt_df['delta_si_static_imu'] = gt_df['si_real'] - gt_df['si_static_imu']
    gt_df['delta_si_static_fused'] = gt_df['si_real'] - gt_df['si_static_fused']
    gt_df['delta_si_pred_obs_w'] = gt_df['si_real'] - gt_df['si_pred_obs_w']

    _validate_ground_truth(
        gt_df.rename(columns={'si_static_imu': 'si_static', 'delta_si_static_imu': 'delta_si'})
    )

    logger.info(f"Built enhanced ground truth dataset with {len(gt_df)} samples")
    return gt_df


def _validate_ground_truth(df: pd.DataFrame):
    """Validate ground truth dataset sanity."""
    critical_cols = ['si_real', 'si_static', 'delta_si']
    for col in critical_cols:
        if col not in df.columns:
            continue
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"Column '{col}' has {nan_count} NaN values")

    if 'si_real' in df.columns and ((df['si_real'] < 0).any() or (df['si_real'] > 2).any()):
        logger.warning(
            f"SI_real contains values outside [0, 2]: min={df['si_real'].min():.3f}, "
            f"max={df['si_real'].max():.3f}"
        )

    if 'delta_si' in df.columns:
        delta_mean = df['delta_si'].mean()
        delta_std = df['delta_si'].std()
        if abs(delta_mean) > 0.5:
            logger.warning(f"delta_si mean is large: {delta_mean:.3f} (expected close to 0)")
        if delta_std > 1.0:
            logger.warning(f"delta_si std is large: {delta_std:.3f} (suggests high residual errors)")


def export_ground_truth(gt_df: pd.DataFrame, output_path: str):
    """Export ground truth dataset to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gt_df.to_csv(out, index=False)
    logger.info(f"Exported ground truth to {out}")
