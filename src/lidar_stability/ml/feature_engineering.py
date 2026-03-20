"""Feature engineering helpers for sprint 5 dynamic model (w prediction)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLUMNS = [
    'roll',
    'pitch',
    'ax',
    'ay',
    'az',
    'speed_kmh',
    'phi_lidar',
    'tri',
    'ruggedness',
]

TARGET_CANDIDATES = ['omega_rad_s', 'w_rad_s', 'gy_rad_s', 'gy', 'gz', 'gy_mdeg_s']


def _resolve_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise KeyError(f"Missing required columns. Tried: {candidates}")
    return None


def _to_numeric_series(df: pd.DataFrame, col: str | None, fill_value: float = np.nan) -> pd.Series:
    if col is None:
        return pd.Series(np.full(len(df), fill_value, dtype=float), index=df.index)
    return pd.to_numeric(df[col], errors='coerce')


def _resolve_feature_columns(df: pd.DataFrame, feature_columns: list[str] | None = None) -> list[str]:
    candidates = feature_columns or DEFAULT_FEATURE_COLUMNS
    usable = [col for col in candidates if col in df.columns]
    if not usable:
        raise KeyError(
            "No valid feature columns found in input dataframe. "
            f"Candidates were: {candidates}"
        )
    return usable


def _resolve_target_omega_rad_s(df: pd.DataFrame, target_column: str | None = None) -> pd.Series:
    if target_column:
        series = _to_numeric_series(df, target_column)
        if target_column in {'gy', 'gz', 'gy_mdeg_s'}:
            return np.radians(series / 1000.0)
        return series

    target = _resolve_column(df, TARGET_CANDIDATES)
    series = _to_numeric_series(df, target)
    if target in {'gy', 'gz', 'gy_mdeg_s'}:
        return np.radians(series / 1000.0)
    return series


def build_w_training_dataset(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    """Build supervised dataset for w prediction in rad/s."""
    if df.empty:
        raise ValueError("Input dataframe is empty")

    used_features = _resolve_feature_columns(df, feature_columns=feature_columns)

    X = df[used_features].apply(pd.to_numeric, errors='coerce')
    y = _resolve_target_omega_rad_s(df, target_column=target_column)

    clean_mask = X.notna().all(axis=1) & y.notna() & np.isfinite(y)
    X_clean = X.loc[clean_mask].copy()
    y_clean = y.loc[clean_mask].astype(float)
    clean_df = df.loc[clean_mask].copy()

    if X_clean.empty:
        raise ValueError(
            "No valid samples after numeric conversion and NaN filtering. "
            "Check feature and target columns."
        )

    return X_clean, y_clean, used_features, clean_df


def load_featured_data(paths: list[str | Path]) -> pd.DataFrame:
    """Load and concatenate one or multiple featured CSV files."""
    frames = []
    for p in paths:
        csv_path = Path(p)
        if not csv_path.exists():
            continue
        frames.append(pd.read_csv(csv_path))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
