"""Tests for Sprint 5: empirical static + dynamic w prediction pipeline blocks."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Add scripts folder to import path
scripts_path = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_path))

from lidar_stability.ml.feature_engineering import build_w_training_dataset
from lidar_stability.physics import StabilityEngine
from lidar_stability.pipeline import build_enhanced_ground_truth


@pytest.fixture
def engine():
    config_path = scripts_path / 'config' / 'vehicle.yaml'
    return StabilityEngine(str(config_path))


@pytest.fixture
def featured_like_df():
    n = 120
    idx = np.arange(n, dtype=float)

    return pd.DataFrame({
        'timestamp': pd.date_range('2026-03-01', periods=n, freq='s'),
        'timeantwifi': idx * 100000.0,
        'roll': np.sin(idx / 15.0) * 6.0,
        'pitch': np.cos(idx / 20.0) * 4.0,
        'ax': 50.0 + np.sin(idx / 10.0) * 3.0,
        'ay': -20.0 + np.cos(idx / 9.0) * 2.5,
        'az': 980.0 + np.sin(idx / 25.0) * 5.0,
        'si': 0.9 - np.sin(idx / 30.0) * 0.05,
        'gy': 180.0 + np.sin(idx / 12.0) * 35.0,
        'speed_kmh': 35.0 + np.cos(idx / 18.0) * 5.0,
        'phi_lidar': np.radians(2.0 + np.sin(idx / 14.0)),
        'tri': 0.4 + np.abs(np.sin(idx / 21.0)) * 0.2,
        'ruggedness': 0.5 + np.abs(np.cos(idx / 25.0)) * 0.15,
    })


class TestEnhancedGroundTruth:
    def test_enhanced_gt_columns(self, featured_like_df, engine):
        gt = build_enhanced_ground_truth(featured_like_df, engine)

        expected = [
            'si_real',
            'si_static_imu',
            'si_static_lidar',
            'si_static_fused',
            'si_dynamic_obs',
            'si_pred_obs_w',
            'delta_si_static_fused',
            'delta_si_pred_obs_w',
            'omega_rad_s',
        ]
        for col in expected:
            assert col in gt.columns, f'Missing expected column: {col}'

    def test_enhanced_gt_has_finite_core_values(self, featured_like_df, engine):
        gt = build_enhanced_ground_truth(featured_like_df, engine)

        core_cols = ['si_real', 'si_static_imu', 'si_static_fused', 'si_pred_obs_w', 'omega_rad_s']
        for col in core_cols:
            assert np.isfinite(gt[col]).all(), f'Column {col} must be finite'

    def test_fused_static_differs_when_lidar_changes(self, featured_like_df, engine):
        base = featured_like_df.copy()
        modified = featured_like_df.copy()
        modified['phi_lidar'] = modified['phi_lidar'] * 2.0

        gt_base = build_enhanced_ground_truth(base, engine)
        gt_mod = build_enhanced_ground_truth(modified, engine)

        diff = np.abs(gt_base['si_static_fused'] - gt_mod['si_static_fused']).mean()
        assert diff > 1e-4, 'Fused static SI should react to LiDAR slope changes'


class TestWFeatureEngineering:
    def test_build_w_training_dataset(self, featured_like_df):
        X, y, used_features, clean_df = build_w_training_dataset(featured_like_df)

        assert not X.empty
        assert not y.empty
        assert len(X) == len(y) == len(clean_df)
        assert len(used_features) >= 5
        assert np.isfinite(X.values).all()
        assert np.isfinite(y.values).all()

    def test_target_converted_to_rad_s(self, featured_like_df):
        _, y, _, _ = build_w_training_dataset(featured_like_df, target_column='gy')

        expected_first = np.radians(featured_like_df.iloc[0]['gy'] / 1000.0)
        assert np.isclose(y.iloc[0], expected_first)

