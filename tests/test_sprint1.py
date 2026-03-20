"""
Tests for Sprint 1: Batch Processing, Visualization, Physics Engine, and Ground Truth

Run with: pytest src/lidar_stability/tests/test_sprint1.py -v
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

from lidar_stability.parsers import batch_processor
from lidar_stability.parsers import route_visualizer
from lidar_stability.physics import StabilityEngine
from lidar_stability.pipeline import build_ground_truth


class TestBatchProcessor:
    """Tests for batch processing helpers."""

    def test_split_into_segments_filters_short(self):
        """Segments with fewer than 10 points should be discarded."""
        # Build a path with a large gap so we get two segments
        x = np.concatenate([np.arange(0, 12), np.arange(2000, 2010)])
        y = np.zeros_like(x, dtype=float)
        df = pd.DataFrame({"x_utm": x, "y_utm": y})

        segments = batch_processor.split_into_segments(df, max_gap_meters=1000)

        # First segment has 12 points -> kept; second has 10 points -> discarded by >10 rule
        assert len(segments) == 1, "Only segments with >10 points should remain"
        assert len(segments[0]) == 12

    def test_filter_isolated_points(self):
        """Isolated points should be removed based on local neighbor distance."""
        df = pd.DataFrame({
            "x_utm": [0, 1, 2, 1000],
            "y_utm": [0, 0, 0, 0],
        })

        filtered = batch_processor.filter_isolated_points(
            df,
            neighbor_distance=5,
            window_size=1,
            min_neighbors=1,
        )

        # The isolated far point should be removed
        assert len(filtered) == 3
        assert filtered["x_utm"].max() <= 2

    def test_match_by_timestamp(self):
        """Matching should return rows within tolerance and drop unmatched."""
        gps_df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-02-26 10:00:00",
                "2026-02-26 10:00:02",
            ]),
            "lat": [40.0, 40.1],
            "lon": [-3.7, -3.8],
        })

        stab_df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-02-26 10:00:00.500",
                "2026-02-26 10:00:05.000",
            ]),
            "si": [0.2, 0.3],
        })

        merged = batch_processor.match_by_timestamp(gps_df, stab_df, tolerance_seconds=1.0)

        assert merged is not None
        assert len(merged) == 1
        assert np.isclose(merged.iloc[0]["si"], 0.2)


class TestRouteVisualizer:
    """Tests for route visualization helpers."""

    def test_si_to_color_bounds(self):
        """Color mapping should clamp values and return hex strings."""
        assert route_visualizer.si_to_color(0.0) == "#FF0000"
        assert route_visualizer.si_to_color(1.0) == "#00AA00"
        assert route_visualizer.si_to_color(-1.0) == "#FF0000"
        assert route_visualizer.si_to_color(2.0) == "#00AA00"

    def test_load_route_data(self, tmp_path):
        """Load route data from a CSV with SI column detection."""
        csv_path = tmp_path / "route.csv"
        df = pd.DataFrame({
            "timestamp": ["2026-02-26 10:00:00", "2026-02-26 10:00:01"],
            "lat": [40.0, 40.0001],
            "lon": [-3.7, -3.7001],
            "si_total": [0.2, 0.3],
        })
        df.to_csv(csv_path, index=False)

        loaded = route_visualizer.load_route_data(csv_path)
        assert not loaded.empty
        assert "si" in loaded.columns
        assert loaded["si"].between(0, 1).all()

    def test_find_matching_files(self, tmp_path):
        """Pattern discovery should find base and segmented files."""
        base = tmp_path / "DOBACK024_20251005.csv"
        seg1 = tmp_path / "DOBACK024_20251005_seg1.csv"
        seg2 = tmp_path / "DOBACK024_20251005_seg2.csv"

        for path in [base, seg1, seg2]:
            pd.DataFrame({"lat": [40.0], "lon": [-3.7], "si": [0.2]}).to_csv(path, index=False)

        matches = route_visualizer.find_matching_files("DOBACK024_20251005", search_dir=tmp_path)
        names = sorted([p.name for p in matches])

        assert "DOBACK024_20251005.csv" in names
        assert "DOBACK024_20251005_seg1.csv" in names
        assert "DOBACK024_20251005_seg2.csv" in names


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
        
        # Expected: arctan(2.480 / (2 * 1.850)) = arctan(0.6703) â‰ˆ 33.8Â°
        assert 33 <= phi_c_deg <= 34, f"Critical angle should be ~33.8Â°, got {phi_c_deg:.2f}Â°"
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
        assert abs(delta_mean) < 0.3, f"Î”SI mean should be < 0.3, got {delta_mean:.3f}"
        # Std should be reasonable
        assert delta_std < 0.5, f"Î”SI std should be < 0.5, got {delta_std:.3f}"
    
    def test_ground_truth_consistency(self, imu_data, engine):
        """Test SI_final = SI_static + Î”SI."""
        gt = build_ground_truth(imu_data, engine)
        
        # Recompute and verify
        si_final = gt['si_static'] + gt['delta_si']
        assert np.allclose(si_final.values, gt['si_real'].values), \
            "SI_final should equal SI_static + Î”SI"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

