"""Timestamp helpers for GPS and IMU data."""

from __future__ import annotations

import pandas as pd


def calculate_imu_absolute_timestamp(imu_df: pd.DataFrame, session_start) -> pd.Series:
    """Convert relative IMU microseconds into absolute timestamps."""
    start = pd.to_datetime(session_start)
    offsets = pd.to_timedelta(pd.to_numeric(imu_df["t_us"], errors="coerce").fillna(0), unit="us")
    return pd.Series(start + offsets, index=imu_df.index, name="timestamp")


def merge_gps_imu(gps_df: pd.DataFrame, imu_df: pd.DataFrame, tolerance_seconds: float = 1.0) -> pd.DataFrame:
    """Merge GPS samples onto the IMU timeline using nearest timestamp."""
    if gps_df.empty or imu_df.empty:
        return pd.DataFrame()

    gps = gps_df.copy()
    imu = imu_df.copy()
    gps_time_col = "timestamp_utc" if "timestamp_utc" in gps.columns else "timestamp"
    gps["timestamp"] = pd.to_datetime(gps[gps_time_col])
    if "timestamp" not in imu.columns:
        imu["timestamp"] = calculate_imu_absolute_timestamp(imu, gps["timestamp"].min())
    else:
        imu["timestamp"] = pd.to_datetime(imu["timestamp"])

    gps = gps.sort_values("timestamp")
    imu = imu.sort_values("timestamp")
    return pd.merge_asof(
        imu,
        gps,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
        suffixes=("_imu", "_gps"),
    )
