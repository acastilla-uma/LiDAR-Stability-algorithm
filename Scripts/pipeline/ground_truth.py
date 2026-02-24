"""
Ground Truth Pipeline

Builds the ground truth dataset by combining IMU telemetry with physics engine.
Computes: SI_real (from MCU), SI_static (from physics), and delta_si (residual).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_ground_truth(imu_df: pd.DataFrame, engine) -> pd.DataFrame:
    """
    Build ground truth dataset from IMU data and physics engine.
    
    Args:
        imu_df: DataFrame from parse_imu() with columns including:
                [t_us, roll_deg, pitch_deg, si_mcu, ...]
        engine: StabilityEngine instance
        
    Returns:
        DataFrame with columns: [t_us, roll_deg, pitch_deg, si_real, si_static, delta_si]
    """
    if imu_df.empty:
        logger.warning("Empty IMU DataFrame provided")
        return pd.DataFrame()
    
    # Create output dataframe
    gt_df = pd.DataFrame()
    gt_df['t_us'] = imu_df['t_us']
    gt_df['roll_deg'] = imu_df['roll_deg']
    gt_df['pitch_deg'] = imu_df['pitch_deg']
    
    # SI_real is the MCU measurement
    gt_df['si_real'] = imu_df['si_mcu']
    
    # SI_static from physics (using roll angle from IMU)
    # Convert roll from degrees to radians for the engine
    roll_rad = np.radians(imu_df['roll_deg'].values)
    gt_df['si_static'] = engine.si_static_batch(roll_rad)
    
    # Delta SI is the residual (what ML will predict)
    gt_df['delta_si'] = gt_df['si_real'] - gt_df['si_static']
    
    # Validate data
    _validate_ground_truth(gt_df)
    
    logger.info(f"Built ground truth dataset with {len(gt_df)} samples")
    logger.info(f"  SI_real range: [{gt_df['si_real'].min():.3f}, {gt_df['si_real'].max():.3f}]")
    logger.info(f"  SI_static range: [{gt_df['si_static'].min():.3f}, {gt_df['si_static'].max():.3f}]")
    logger.info(f"  ΔSI mean: {gt_df['delta_si'].mean():.3f}, std: {gt_df['delta_si'].std():.3f}")
    
    return gt_df


def _validate_ground_truth(df: pd.DataFrame):
    """Validate ground truth dataset sanity."""
    # Check for NaN in critical columns
    critical_cols = ['si_real', 'si_static', 'delta_si']
    for col in critical_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"Column '{col}' has {nan_count} NaN values")
    
    # Check SI_real ranges
    if (df['si_real'] < 0).any() or (df['si_real'] > 2).any():
        logger.warning(f"SI_real contains values outside [0, 2]: "
                      f"min={df['si_real'].min():.3f}, max={df['si_real'].max():.3f}")
    
    # Check delta_si distribution (should be somewhat centered around 0)
    delta_mean = df['delta_si'].mean()
    delta_std = df['delta_si'].std()
    if abs(delta_mean) > 0.5:
        logger.warning(f"ΔSI mean is large: {delta_mean:.3f} "
                      "(expected close to 0)")
    if delta_std > 1.0:
        logger.warning(f"ΔSI std is large: {delta_std:.3f} "
                      "(suggests high residual errors)")


def export_ground_truth(gt_df: pd.DataFrame, output_path: str):
    """
    Export ground truth dataset to CSV.
    
    Args:
        gt_df: Ground truth DataFrame
        output_path: Path to save CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    gt_df.to_csv(output_path, index=False)
    logger.info(f"Exported ground truth to {output_path}")
