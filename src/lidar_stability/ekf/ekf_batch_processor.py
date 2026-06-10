"""Batch helpers for synchronized EKF input data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def match_gps_stability(gps_df: pd.DataFrame, stab_df: pd.DataFrame, tolerance_seconds: float = 1.0) -> pd.DataFrame:
    """Align GPS rows onto the stability timeline."""
    if gps_df.empty or stab_df.empty:
        return pd.DataFrame()

    gps = gps_df.copy()
    stab = stab_df.copy()
    gps["timestamp"] = pd.to_datetime(gps["timestamp"])
    stab["timestamp"] = pd.to_datetime(stab["timestamp"])
    return pd.merge_asof(
        stab.sort_values("timestamp"),
        gps.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    )


def split_segments(df: pd.DataFrame, max_gap_meters: float = 1000.0, min_points: int = 10) -> list[pd.DataFrame]:
    """Split a route on large spatial gaps and keep segments with enough points."""
    if df.empty:
        return []

    coords = df[["x_utm", "y_utm"]].to_numpy(dtype=float)
    distances = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    split_at = np.where(distances > max_gap_meters)[0] + 1
    pieces = np.split(df, split_at)
    return [piece.reset_index(drop=True) for piece in pieces if len(piece) >= min_points]
