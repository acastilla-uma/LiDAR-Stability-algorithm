"""
Tests for Sprint 1: Parsers, Physics Engine, and Ground Truth

Run with: pytest scripts/tests/test_sprint1.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

# Add scripts to path
scripts_path = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_path))

from parsers import parse_gps, parse_imu
from physics import StabilityEngine
from pipeline import build_ground_truth


class TestGPSParser:
    """Tests for GPS parser."""
    
    @pytest.fixture
    def config_path(self):
        return scripts_path / 'config' / 'vehicle.yaml'
    
    def test_gps_parse_valid(self):
        """Test that GPS parser returns valid DataFrame."""
        gps_file = Path(__file__).parent.parent.parent / 'Doback-Data' / 'GPS' / 'GPS_DOBACK027_20250814_0.txt'
        
        if not gps_file.exists():
            pytest.skip("GPS test file not found")
        
        df = parse_gps(str(gps_file))
        
        assert not df.empty, "Parsed DataFrame should not be empty"
        assert len(df) > 0, "Should have at least one row"
        
        # Check columns
        expected_cols = ['timestamp_utc', 'lat', 'lon', 'alt', 'hdop', 'speed_kmh', 'device_id', 'session']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check for NaN in critical columns
        assert df['lat'].notna().all(), "Latitude should not have NaN"
        assert df['lon'].notna().all(), "Longitude should not have NaN"
        
        # Check data types
        assert df['timestamp_utc'].dtype == 'datetime64[ns]', "Timestamp should be datetime"
        assert pd.api.types.is_numeric_dtype(df['lat']), "Latitude should be numeric"
        assert pd.api.types.is_numeric_dtype(df['lon']), "Longitude should be numeric"
    
    def test_gps_rejects_corrupt(self):
        """Test that corrupt rows are properly rejected."""
        gps_file = Path(__file__).parent.parent.parent / 'Doback-Data' / 'GPS' / 'GPS_DOBACK027_20250814_0.txt'
        
        if not gps_file.exists():
            pytest.skip("GPS test file not found")
        
        df = parse_gps(str(gps_file))
        
        # Check that no absurd values exist
        assert (df['speed_kmh'] < 200).all(), "All speeds should be < 200 km/h"
        assert (df['lat'] >= 36).all() and (df['lat'] <= 44).all(), "Latitudes should be in Peninsula range"
        assert (df['lon'] >= -10).all() and (df['lon'] <= 5).all(), "Longitudes should be in Peninsula range"
    
    def test_gps_metadata(self):
        """Test that metadata is parsed correctly."""
        gps_file = Path(__file__).parent.parent.parent / 'Doback-Data' / 'GPS' / 'GPS_DOBACK027_20250814_0.txt'
        
        if not gps_file.exists():
            pytest.skip("GPS test file not found")
        
        df = parse_gps(str(gps_file))
        
        if not df.empty:
            assert df.iloc[0]['device_id'] in ['DOBACK027', 'DOBACK024', 'UNKNOWN']
            # Session should be an integer
            assert isinstance(df.iloc[0]['session'], (int, np.integer))


class TestIMUParser:
    """Tests for IMU/Stability parser."""
    
    def test_imu_parse_valid(self):
        """Test that IMU parser returns valid DataFrame."""
        imu_file = Path(__file__).parent.parent.parent / 'Doback-Data' / 'Stability' / 'ESTABILIDAD_DOBACK024_20250825_188.txt'
        
        if not imu_file.exists():
            pytest.skip("IMU test file not found")
        
        df = parse_imu(str(imu_file))
        
        assert not df.empty, "Parsed DataFrame should not be empty"
        assert len(df) >= 20000, "Should have many samples for a 10 Hz sensor"
        
        # Check columns
        expected_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll_deg', 'pitch_deg', 'yaw_deg', 't_us', 'si_mcu']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        # SI_mcu should be in valid range
        assert (df['si_mcu'] >= 0).all() and (df['si_mcu'] <= 2).all(), "SI should be in [0, 2]"
    
    def test_imu_skips_timestamps(self):
        """Test that timestamp marker lines are properly skipped."""
        imu_file = Path(__file__).parent.parent.parent / 'Doback-Data' / 'Stability' / 'ESTABILIDAD_DOBACK024_20250825_188.txt'
        
        if not imu_file.exists():
            pytest.skip("IMU test file not found")
        
        df = parse_imu(str(imu_file))
        
        # Check that no row contains AM or PM in numeric fields
        for col in ['ax', 'ay', 'az']:
            # These should all be numeric, not strings with AM/PM
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"
    
    def test_imu_frequency(self):
        """Test that IMU data is approximately 10 Hz."""
        imu_file = Path(__file__).parent.parent.parent / 'Doback-Data' / 'Stability' / 'ESTABILIDAD_DOBACK024_20250825_188.txt'
        
        if not imu_file.exists():
            pytest.skip("IMU test file not found")
        
        df = parse_imu(str(imu_file))
        
        if len(df) > 1:
            # Calculate dt in milliseconds
            dt_us = df['t_us'].diff().dropna()
            dt_ms = dt_us / 1000.0
            median_dt_ms = dt_ms.median()
            
            # Should be around 100 ms for 10 Hz
            assert 80 <= median_dt_ms <= 120, f"Median dt = {median_dt_ms:.2f} ms, expected ~100 ms for 10 Hz"


class TestStabilityEngine:
    """Tests for physics engine."""
    
    @pytest.fixture
    def engine(self):
        config_path = scripts_path / 'config' / 'vehicle.yaml'
        return StabilityEngine(str(config_path))
    
    def test_critical_angle(self, engine):
        """Test critical angle calculation."""
        phi_c_deg = engine.critical_angle(degrees=True)
        phi_c_rad = engine.critical_angle(degrees=False)
        
        # Expected: arctan(2.480 / (2 * 1.850)) = arctan(0.6703) ≈ 33.8°
        assert 33 <= phi_c_deg <= 34, f"Critical angle should be ~33.8°, got {phi_c_deg:.2f}°"
        assert 0.58 <= phi_c_rad <= 0.60, f"Critical angle should be ~0.589 rad, got {phi_c_rad:.4f} rad"
    
    def test_si_zero_roll(self, engine):
        """Test SI at zero roll."""
        si = engine.si_static(0.0)
        assert si == 0.0, f"SI at zero roll should be 0, got {si}"
    
    def test_si_critical_angle(self, engine):
        """Test SI at critical angle."""
        phi_c = engine.critical_angle(degrees=False)
        si = engine.si_static(phi_c)
        
        # SI should be approximately 1.0 at critical angle
        assert 0.95 <= si <= 1.05, f"SI at critical angle should be ~1.0, got {si:.3f}"
    
    def test_si_beyond_critical(self, engine):
        """Test SI beyond critical angle is unstable."""
        phi_c = engine.critical_angle(degrees=False)
        si = engine.si_static(phi_c + 0.1)
        
        # SI > 1.0 means unstable
        assert si > 1.0, f"SI beyond critical angle should be > 1.0, got {si:.3f}"
    
    def test_si_batch(self, engine):
        """Test batch SI calculation."""
        rolls = np.array([0.0, 0.1, 0.2, 0.3])
        si_batch = engine.si_static_batch(rolls)
        
        # Compare with single calculations
        for i, roll in enumerate(rolls):
            si_single = engine.si_static(roll)
            assert np.isclose(si_batch[i], si_single), "Batch should match single calculations"
    
    def test_si_from_degrees(self, engine):
        """Test SI calculation from degrees."""
        roll_deg = 10.0
        si_deg = engine.si_static_from_deg(roll_deg)
        
        roll_rad = np.radians(roll_deg)
        si_rad = engine.si_static(roll_rad)
        
        assert np.isclose(si_deg, si_rad), "Degree and radian calculations should match"
    
    def test_vehicle_params(self, engine):
        """Test vehicle parameters retrieval."""
        params = engine.get_vehicle_params()
        
        assert params['mass_kg'] == 18000
        assert params['track_width_m'] == 2.480
        assert params['cg_height_m'] == 1.850
        assert params['roll_inertia_kg_m2'] == 89300


class TestGroundTruth:
    """Tests for ground truth pipeline."""
    
    @pytest.fixture
    def imu_data(self):
        """Create minimal IMU test data."""
        return pd.DataFrame({
            't_us': np.arange(0, 10000, 100),
            'roll_deg': np.linspace(0, 10, 100),
            'pitch_deg': np.zeros(100),
            'si_mcu': np.linspace(0, 0.3, 100),
            'device_id': ['DOBACK024'] * 100,
            'session': [0] * 100
        })
    
    @pytest.fixture
    def engine(self):
        config_path = scripts_path / 'config' / 'vehicle.yaml'
        return StabilityEngine(str(config_path))
    
    def test_ground_truth_columns(self, imu_data, engine):
        """Test ground truth has correct columns."""
        gt = build_ground_truth(imu_data, engine)
        
        expected_cols = ['t_us', 'roll_deg', 'pitch_deg', 'si_real', 'si_static', 'delta_si']
        for col in expected_cols:
            assert col in gt.columns, f"Missing column: {col}"
    
    def test_ground_truth_no_nans(self, imu_data, engine):
        """Test ground truth has no NaN in critical columns."""
        gt = build_ground_truth(imu_data, engine)
        
        critical_cols = ['si_real', 'si_static', 'delta_si']
        for col in critical_cols:
            assert gt[col].notna().all(), f"Column {col} should not have NaN"
    
    def test_delta_si_distribution(self, imu_data, engine):
        """Test delta_si has reasonable distribution."""
        gt = build_ground_truth(imu_data, engine)
        
        delta_mean = gt['delta_si'].mean()
        delta_std = gt['delta_si'].std()
        
        # Mean should be close to 0 (physics captures most of SI)
        assert abs(delta_mean) < 0.3, f"ΔSI mean should be < 0.3, got {delta_mean:.3f}"
        # Std should be reasonable
        assert delta_std < 0.5, f"ΔSI std should be < 0.5, got {delta_std:.3f}"
    
    def test_ground_truth_consistency(self, imu_data, engine):
        """Test SI_final = SI_static + ΔSI."""
        gt = build_ground_truth(imu_data, engine)
        
        # Recompute and verify
        si_final = gt['si_static'] + gt['delta_si']
        assert np.allclose(si_final.values, gt['si_real'].values), \
            "SI_final should equal SI_static + ΔSI"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
