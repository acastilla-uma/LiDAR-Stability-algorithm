"""
Time Synchronization Module

Aligns GPS (UTC datetime) and IMU (monotonic microsecond counter) temporal bases.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def calculate_imu_absolute_timestamp(imu_df: pd.DataFrame, 
                                     session_start_utc: pd.Timestamp) -> pd.Series:
    """
    Convert IMU monotonic microsecond counter to absolute UTC timestamps.
    
    Args:
        imu_df: IMU DataFrame with 't_us' column (monotonic microseconds)
        session_start_utc: Session start time in UTC (from metadata or first GPS fix)
        
    Returns:
        pd.Series of absolute UTC timestamps
    """
    if imu_df.empty:
        return pd.Series(dtype='datetime64[ns]')
    
    # Convert microseconds to timedeltas relative to first IMU sample
    t_us_min = imu_df['t_us'].min()
    dt_us = imu_df['t_us'] - t_us_min
    dt_seconds = dt_us / 1e6
    
    # Create absolute timestamps
    timestamps = [session_start_utc + timedelta(seconds=float(s)) for s in dt_seconds]
    
    return pd.Series(timestamps, index=imu_df.index)


def merge_gps_imu(gps_df: pd.DataFrame, imu_df: pd.DataFrame, 
                  method='linear') -> pd.DataFrame:
    """
    Merge GPS and IMU data on a unified timeline.
    
    GPS provides absolute time references; IMU provides high-frequency samples.
    
    Args:
        gps_df: GPS DataFrame with 'timestamp_utc' column
        imu_df: IMU DataFrame with 't_us' column
        method: Interpolation method for gaps ('linear', 'nearest')
        
    Returns:
        Merged DataFrame with unified timeline
    """
    if gps_df.empty or imu_df.empty:
        logger.warning("Cannot merge: one or both DataFrames are empty")
        return pd.DataFrame()
    
    # Use GPS timestamp as the time reference
    session_start_gps = gps_df['timestamp_utc'].min()
    
    # Convert IMU timestamps to absolute
    imu_df_copy = imu_df.copy()
    imu_df_copy['timestamp_utc'] = calculate_imu_absolute_timestamp(
        imu_df_copy, session_start_gps
    )
    
    # Create unified timeline
    all_times = pd.concat([
        gps_df['timestamp_utc'],
        imu_df_copy['timestamp_utc']
    ]).drop_duplicates().sort_values().reset_index(drop=True)
    
    # Resample to unified timeline
    # First, set index and forward fill
    imu_reindexed = imu_df_copy.set_index('timestamp_utc').sort_index()
    gps_reindexed = gps_df.set_index('timestamp_utc').sort_index()
    
    # Create a column to track data source
    imu_reindexed['source'] = 'imu'
    gps_reindexed['source'] = 'gps'
    
    # Combine and sort; handle duplicate indices
    combined = pd.concat([imu_reindexed, gps_reindexed]).sort_index()
    
    # Forward fill
    combined = combined.ffill()
    
    # Remove duplicates in index by grouping and taking first
    combined = combined[~combined.index.duplicated(keep='first')]
    
    # Reindex to unified timeline using reindex_like or direct assignment
    combined = combined.reindex(all_times)
    # Forward fill again after reindex
    combined = combined.ffill()
    
    combined = combined.reset_index()
    combined.rename(columns={'index': 'timestamp_utc'}, inplace=True)
    
    logger.info(f"Merged {len(gps_df)} GPS records with {len(imu_df)} IMU records")
    logger.info(f"  Unified timeline: {len(combined)} samples")
    if 'source' in combined.columns:
        logger.info(f"  GPS ratio: {(combined['source'] == 'gps').sum() / len(combined) * 100:.1f}%")
    
    return combined
